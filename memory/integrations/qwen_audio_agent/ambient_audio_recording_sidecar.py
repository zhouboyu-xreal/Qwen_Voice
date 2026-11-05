#!/usr/bin/env python3
"""Persist ambient PCM as WAV and asynchronously transcribe sealed batches.

This sidecar deliberately owns no ``MemoryRuntime`` or SQLite connection. It
only creates durable, lossless audio evidence and produces timestamped
transcript segments. Completed segments are sent directly over a private local
Unix socket to the primary Agent Memory sidecar, which remains the sole writer
for each owner. The Gateway is not on this data path.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import logging
import math
import os
import re
import socket
import sys
import time
import wave
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config import load_voice_runtime_config
from voice.voice_runtime import VoiceRuntime
from voice.ambient_audio_splitter import JointSilenceSplitter


_RECORDING_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")


def _load_config(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - project dependency
        raise RuntimeError("PyYAML is required to load config.yaml") from exc
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError("config.yaml must contain a mapping")
    return value


def _clean(value: Any, limit: int = 512) -> str:
    return str(value or "").replace("\0", "").strip()[:limit]


def _parse_timestamp(value: Any) -> datetime:
    text = _clean(value, 120)
    if not text:
        return datetime.now().astimezone()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"invalid recording startedAt: {text}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=datetime.now().astimezone().tzinfo)
    return parsed


@dataclass(frozen=True)
class _AudioBatch:
    index: int
    path: Path
    started_at: datetime
    ended_at: datetime
    sample_count: int
    sample_rate: int
    source: str
    cut_reason: str
    shared_cut_at: Optional[datetime]


@dataclass
class _AudioTrack:
    source: str
    sample_rate: Optional[int] = None
    batch_index: int = 0
    batch_started_at: Optional[datetime] = None
    batch_samples: int = 0
    writer: Optional[wave.Wave_write] = None
    partial_path: Optional[Path] = None


class _AgentMemoryIpcClient:
    """Small synchronous client for Agent Memory's local Unix socket."""

    def __init__(self, path: Path, owner_id: str) -> None:
        self._path = path.expanduser().resolve()
        self._owner_id = _clean(owner_id, 240)
        if not self._owner_id:
            raise ValueError("agent-memory owner id must be non-empty")

    def request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        payload = json.dumps({"method": method, "params": params}, ensure_ascii=False)
        last_error: Optional[Exception] = None
        for attempt in range(5):
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                    connection.settimeout(30.0)
                    connection.connect(str(self._path))
                    connection.sendall((payload + "\n").encode("utf-8"))
                    response = b""
                    while b"\n" not in response and len(response) <= 2_000_000:
                        chunk = connection.recv(65_536)
                        if not chunk:
                            break
                        response += chunk
                parsed = json.loads(response.split(b"\n", 1)[0].decode("utf-8"))
                if not isinstance(parsed, dict):
                    raise RuntimeError("agent-memory IPC returned an invalid response")
                if parsed.get("ok") is not True:
                    raise RuntimeError(str(parsed.get("error") or "agent-memory IPC failed"))
                result = parsed.get("result")
                return result if isinstance(result, dict) else {}
            except (OSError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt < 4:
                    time.sleep(0.5 * (attempt + 1))
        raise RuntimeError(f"agent-memory IPC unavailable: {last_error}")

    @property
    def owner_id(self) -> str:
        return self._owner_id


class AmbientAudioRecordingService:
    """Persist both tracks and ASR artifacts on their shared sample timeline."""

    def __init__(
        self,
        *,
        voice_config: Dict[str, Any],
        recordings_dir: Path,
        chunk_duration_seconds: float,
        agent_memory_ipc: _AgentMemoryIpcClient,
        logger: logging.Logger,
        split_silence_seconds: float = 0.8,
        split_quiet_max_rms_dbfs: float = -35.0,
        detector_factory=None,
    ) -> None:
        # Starting a recording must only open a durable WAV stream.  Loading
        # the offline ASR runtime here can take seconds (or longer on the
        # first run), which used to leave the Desktop record control looking
        # unresponsive before it had even accepted ``recording.start``.
        # Construct it in the single ASR worker when the first sealed batch is
        # actually ready to transcribe instead.
        self._voice_config = voice_config
        self._voice_runtime: Optional[VoiceRuntime] = None
        runtime_config = voice_config.get("runtime") if isinstance(voice_config, dict) else None
        sample_rate_value = (
            runtime_config.get("sample_rate", 16_000)
            if isinstance(runtime_config, dict)
            else 16_000
        )
        self._asr_sample_rate = max(1, int(sample_rate_value))
        self._chunk_duration_seconds = float(chunk_duration_seconds)
        self._split_silence_seconds = float(split_silence_seconds)
        if not math.isfinite(self._chunk_duration_seconds) or self._chunk_duration_seconds <= 0:
            raise ValueError("chunk-duration-seconds must be finite and positive")
        if not math.isfinite(self._split_silence_seconds) or self._split_silence_seconds <= 0:
            raise ValueError("split-silence-seconds must be finite and positive")
        self._split_quiet_max_rms_dbfs = float(split_quiet_max_rms_dbfs)
        if not math.isfinite(self._split_quiet_max_rms_dbfs) or not -120 <= self._split_quiet_max_rms_dbfs <= 0:
            raise ValueError("split quiet RMS threshold must be between -120 and 0 dBFS")
        self._detector_factory = detector_factory
        self._split_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ambient-split-vad")
        self._split_pending = []
        self._split_epoch = 0
        self._splitter = None
        self._split_detection_from = None
        self._shared_batch_started_at = None
        self._split_events = []
        self._tracks: Dict[str, _AudioTrack] = {}
        self._asr_progress: Dict[str, datetime] = {}
        self._buffered_segments: List[Dict[str, Any]] = []
        self._delivery_index = 0
        self._delivery_future: Optional[Future[Dict[str, Any]]] = None
        self._delivery_until: Optional[datetime] = None
        self._delivery_payload: Optional[Dict[str, Any]] = None
        self._pending_batches: Dict[Future[Dict[str, Any]], _AudioBatch] = {}
        self._failed_until: Optional[datetime] = None
        self._recordings_dir = recordings_dir.expanduser().resolve()
        self._recordings_dir.mkdir(parents=True, exist_ok=True)
        self._agent_memory_ipc = agent_memory_ipc
        self._logger = logger
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="ambient-audio-asr",
        )
        self._recording_id = ""
        self._recording_started_at: Optional[datetime] = None
        self._manifest_path: Optional[Path] = None
        self._audio_file_names: List[str] = []
        self._transcript_results: Dict[str, Dict[str, Any]] = {}
        self._failed_transcript_saves = 0
        self._pending: List[Future[Dict[str, Any]]] = []
        self._completed: List[Dict[str, Any]] = []
        self._failed_batches = 0
        self._stopping = False
        self._memory_finalized = False

    def start(self, params: Dict[str, Any]) -> Dict[str, Any]:
        recording_id = _clean(params.get("recordingId"), 128)
        if not _RECORDING_ID.fullmatch(recording_id):
            raise ValueError("recordingId must contain only letters, digits, '_' or '-'")
        if self._recording_id:
            if self._recording_id == recording_id:
                return self._result()
            raise RuntimeError("another ambient recording is already active")
        sources = params.get("sources", ["microphone"])
        if not isinstance(sources, list) or not sources or any(
            source not in ("microphone", "system") for source in sources
        ) or len(set(sources)) != len(sources):
            raise ValueError("sources must be a non-empty unique list of microphone/system")
        self._recording_started_at = _parse_timestamp(params.get("startedAt"))
        self._recording_id = recording_id
        recording_stamp = self._recording_started_at.astimezone().strftime("%Y%m%d_%H%M%S")
        self._manifest_path = self._recordings_dir / (
            f"{recording_stamp}_{self._recording_short_id()}.json"
        )
        self._audio_file_names = []
        self._transcript_results = {}
        self._failed_transcript_saves = 0
        self._tracks = {source: _AudioTrack(source) for source in sources}
        self._asr_progress = {source: self._recording_started_at for source in sources}
        self._buffered_segments = []
        self._delivery_index = 0
        self._published_until = self._recording_started_at
        self._stopping = False
        self._memory_finalized = False
        self._reset_splitter(self._recording_started_at)
        self._write_manifest()
        generation = self._agent_memory_ipc.request("memory_generation", {"ownerId": self._agent_memory_ipc.owner_id})
        self._memory_generation = generation.get("generation", "0")
        registration = self._agent_memory_ipc.request(
            "set_ambient_recording_state",
            {
                "ownerId": self._agent_memory_ipc.owner_id,
                "recordingId": recording_id,
                "active": True,
                "memoryGeneration": self._memory_generation,
            },
        )
        if registration.get("ok") is False:
            raise RuntimeError("Memory is resetting; recording was not started")
        self._logger.info(
            "ambient recording started recording_id=%s split_policy=joint_silence "
            "split_min_seconds=%.3f silence_seconds=%.3f hard_limit=none timeline_policy=sample_count",
            recording_id,
            self._chunk_duration_seconds,
            self._split_silence_seconds,
        )
        return self._result()

    def append_audio(self, params: Dict[str, Any]) -> Dict[str, Any]:
        self._require_active_recording(params)
        if self._stopping:
            raise RuntimeError("ambient recording is stopping")
        source = str(params.get("source") or "microphone")
        track = self._tracks.get(source)
        if track is None:
            raise ValueError("audio source is not enabled for this recording")
        sample_rate = int(params.get("sampleRate") or 0)
        if sample_rate < 8_000 or sample_rate > 192_000:
            raise ValueError("ambient audio sampleRate must be between 8000 and 192000")
        try:
            payload = base64.b64decode(str(params.get("audio") or ""), validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("audio must be valid base64 PCM16") from exc
        if len(payload) % 2:
            raise ValueError("audio must contain complete PCM16 samples")
        if not payload:
            return self._result()
        # Packet capture timestamps may jump when a device is reopened or a
        # renderer callback is delayed. Persist all received PCM consecutively;
        # both sources share the session origin and advance by sample duration.
        # Deliberately do not insert silence or drop samples to repair gaps.
        self._harvest_split_analysis()
        if track.sample_rate is not None and track.sample_rate != sample_rate:
            # A WAV cannot contain multiple sample rates. Preserve the format
            # boundary, continuing at the old batch's accumulated audio end.
            self._seal_current_batch(track, reason="sample_rate_change")
            track.sample_rate = None
            self._reset_splitter(track.batch_started_at)
        if track.sample_rate is None:
            track.sample_rate = sample_rate
            track.batch_started_at = track.batch_started_at or self._recording_started_at
            self._logger.info("ambient track capture format source=%s sample_rate=%s", source, sample_rate)
        actual_start = track.batch_started_at + timedelta(seconds=track.batch_samples / sample_rate)
        self._ensure_writer(track)
        track.writer.writeframes(payload)
        track.batch_samples += len(payload) // 2
        # Audio is already durable before analysis is queued. Analyse only a
        # short warmup before the gate, then continue until a shared quiet span
        # exists. There is deliberately no duration-based forced split.
        ended_at = actual_start + timedelta(seconds=len(payload) / (sample_rate * 2))
        if ended_at > self._split_detection_from:
            skip = max(0, math.ceil((self._split_detection_from - actual_start).total_seconds() * sample_rate))
            analysis_start = actual_start + timedelta(seconds=skip / sample_rate)
            future = self._split_executor.submit(self._splitter.accept, source, payload[skip * 2:],
                                                 sample_rate, analysis_start)
            self._split_pending.append((self._split_epoch, future))
        self._harvest_completed()
        return self._result(include_completed=True)

    def stop(self, params: Dict[str, Any]) -> Dict[str, Any]:
        self._require_active_recording(params)
        self._stopping = True
        self._split_epoch += 1
        for _, future in self._split_pending:
            future.cancel()
        self._split_pending.clear()
        for track in self._tracks.values():
            if track.batch_samples > 0:
                self._seal_current_batch(track)
            else:
                self._discard_empty_partial(track)
        self._harvest_completed()
        self._write_manifest()
        self._logger.info(
            "ambient recording stopped recording_id=%s pending_batches=%s",
            self._recording_id,
            len(self._pending),
        )
        return self._result(include_completed=True)

    def drain(self, params: Dict[str, Any]) -> Dict[str, Any]:
        self._require_active_recording(params)
        self._harvest_completed()
        result = self._result(include_completed=True)
        if self._stopping and not self._pending and not self._buffered_segments and self._delivery_future is None:
            self._finalize_memory_recording()
        return result

    def status(self, _params: Dict[str, Any]) -> Dict[str, Any]:
        self._harvest_completed()
        return self._result()

    def close(self, _params: Dict[str, Any]) -> Dict[str, Any]:
        if self._recording_id and not self._stopping:
            self.stop({"recordingId": self._recording_id})
        # Normal Desktop shutdown drains first. Never close a runtime while
        # its worker is still transcribing a sealed source file.
        while self._pending or self._delivery_payload is not None or self._buffered_segments:
            self._harvest_completed()
            if self._pending or self._delivery_future is not None:
                time.sleep(0.01)
        self._split_executor.shutdown(wait=True, cancel_futures=True)
        self._executor.shutdown(wait=True, cancel_futures=False)
        if self._recording_id and not self._memory_finalized:
            self._finalize_memory_recording()
        if self._voice_runtime is not None:
            self._voice_runtime.close()
        return {"closed": True}

    def _require_active_recording(self, params: Dict[str, Any]) -> None:
        recording_id = _clean(params.get("recordingId"), 128)
        if not self._recording_id or recording_id != self._recording_id:
            raise ValueError("recordingId does not identify the active recording")

    def _reset_splitter(self, started_at: datetime) -> None:
        self._split_epoch += 1
        for _, future in self._split_pending:
            future.cancel()
        self._split_pending.clear()
        self._shared_batch_started_at = started_at
        self._split_detection_from = started_at + timedelta(seconds=max(0, self._chunk_duration_seconds - 1))
        self._splitter = JointSilenceSplitter(
            sources=self._tracks, started_at=started_at,
            min_duration_seconds=self._chunk_duration_seconds,
            silence_seconds=self._split_silence_seconds, quiet_max_rms_dbfs=self._split_quiet_max_rms_dbfs,
            vad_config=self._voice_config.get("vad"), detector_factory=self._detector_factory)

    def _harvest_split_analysis(self) -> None:
        while self._split_pending and self._split_pending[0][1].done():
            epoch, future = self._split_pending.pop(0)
            if epoch != self._split_epoch or future.cancelled():
                continue
            try:
                cuts = future.result()
            except Exception as exc:
                self._logger.exception("ambient joint silence detection failed; saved PCM is retained")
                raise RuntimeError("ambient joint silence detection failed; stop and preserve recorded audio") from exc
            for cut_at in cuts:
                self._seal_shared_batch(cut_at)

    def _seal_shared_batch(self, cut_at: datetime) -> None:
        # Both analyses have covered this point, so both durable streams must
        # cover it too. Never pair unrelated quiet spans or synthesize silence.
        for track in self._tracks.values():
            if track.sample_rate is None or track.batch_started_at is None:
                raise RuntimeError("shared cut does not have audio from every source")
            end = track.batch_started_at + timedelta(seconds=track.batch_samples / track.sample_rate)
            if not track.batch_started_at < cut_at <= end + timedelta(microseconds=1):
                raise RuntimeError("shared cut falls outside a saved source interval")
        duration = (cut_at - self._shared_batch_started_at).total_seconds()
        for track in self._tracks.values():
            self._seal_current_batch(track, reason="joint_silence", cut_at=cut_at)
        self._split_events.append({"reason": "joint_silence", "cut_at": cut_at.isoformat(),
                                   "duration_seconds": duration, "silence_seconds": self._split_silence_seconds})
        self._shared_batch_started_at = cut_at
        self._logger.info("ambient shared batch sealed reason=joint_silence cut_at=%s "
                          "duration_s=%.3f silence_s=%.3f sources=%s", cut_at.isoformat(),
                          duration, self._split_silence_seconds, ",".join(self._tracks))

    def _partial_path(self, track: _AudioTrack, started_at: datetime, index: int) -> Path:
        stem = (f"{started_at.astimezone().strftime('%Y%m%d_%H%M%S')}"
                f"_{track.source}_{index:06d}_{self._recording_short_id()}")
        return self._recordings_dir / f"{stem}.partial.wav"

    @staticmethod
    def _open_writer(path: Path, sample_rate: int):
        writer = wave.open(str(path), "wb")
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        return writer

    def _ensure_writer(self, track: _AudioTrack) -> None:
        if track.writer is not None:
            return
        if track.batch_started_at is None or track.sample_rate is None:
            raise RuntimeError("ambient recording has not started")
        track.batch_index += 1
        track.partial_path = self._partial_path(track, track.batch_started_at, track.batch_index)
        track.writer = self._open_writer(track.partial_path, track.sample_rate)
        track.batch_samples = 0

    @staticmethod
    def _copy_frames(reader, writer, count: int) -> None:
        # A delayed source may arrive after the faster source has written past
        # the agreed cut. Split that disk-backed tail with bounded memory.
        while count:
            frames = reader.readframes(min(count, 65_536))
            if not frames:
                raise RuntimeError("saved PCM ended before the shared cut")
            writer.writeframes(frames)
            count -= len(frames) // 2

    def _seal_current_batch(self, track: _AudioTrack, *, reason="stop", cut_at=None) -> None:
        if track.partial_path is None or track.batch_started_at is None or not track.batch_samples:
            return
        if track.writer is not None:
            track.writer.close()
            track.writer = None
        partial_path = track.partial_path
        total_samples = track.batch_samples
        sample_count = total_samples if cut_at is None else round(
            (cut_at - track.batch_started_at).total_seconds() * track.sample_rate)
        if not 0 < sample_count <= total_samples:
            raise ValueError("cut must preserve a non-empty prefix of saved PCM")
        started_at = track.batch_started_at
        ended_at = started_at + timedelta(seconds=sample_count / track.sample_rate)
        final_path = partial_path.with_name(partial_path.name.replace(".partial.wav", ".wav"))
        tail_samples = total_samples - sample_count
        tail_writer = None
        tail_path = None
        if tail_samples:
            prefix_path = partial_path.with_suffix(".sealing")
            tail_path = self._partial_path(track, ended_at, track.batch_index + 1)
            try:
                with wave.open(str(partial_path), "rb") as reader:
                    with self._open_writer(prefix_path, track.sample_rate) as prefix_writer:
                        self._copy_frames(reader, prefix_writer, sample_count)
                    tail_writer = self._open_writer(tail_path, track.sample_rate)
                    self._copy_frames(reader, tail_writer, tail_samples)
                prefix_path.replace(final_path)
            except Exception:
                if tail_writer is not None:
                    tail_writer.close()
                prefix_path.unlink(missing_ok=True)
                if tail_path is not None:
                    tail_path.unlink(missing_ok=True)
                # The full original WAV remains available for stop/error
                # finalization if a split could not be completed.
                raise
            partial_path.unlink()
        else:
            partial_path.replace(final_path)
        batch = _AudioBatch(index=track.batch_index, path=final_path, started_at=started_at,
                            ended_at=ended_at, sample_count=sample_count, sample_rate=track.sample_rate,
                            source=track.source, cut_reason=reason, shared_cut_at=cut_at)
        self._audio_file_names.append(final_path.name)
        self._transcript_results[final_path.name] = {
            "audioFile": final_path.name, "transcriptFile": final_path.with_suffix(".asr.json").name,
            "source": track.source, "batchIndex": batch.index,
            "asrStatus": "pending", "saveStatus": "pending",
        }
        track.batch_started_at = ended_at
        track.batch_samples = tail_samples
        track.writer = tail_writer
        track.partial_path = tail_path
        if tail_samples:
            track.batch_index += 1
        future = self._executor.submit(self._transcribe_batch, batch)
        self._pending.append(future)
        self._pending_batches[future] = batch
        self._write_manifest()
        self._logger.info("ambient audio batch sealed recording_id=%s source=%s batch=%s "
                          "reason=%s duration_s=%.3f shared_cut_at=%s path=%s", self._recording_id,
                          track.source, batch.index, reason, sample_count / track.sample_rate,
                          cut_at.isoformat() if cut_at else "", final_path.name)

    def _discard_empty_partial(self, track: _AudioTrack) -> None:
        if track.writer is not None:
            track.writer.close()
        if track.partial_path is not None:
            track.partial_path.unlink(missing_ok=True)
        track.writer = None
        track.partial_path = None

    def _transcribe_batch(self, batch: _AudioBatch) -> Dict[str, Any]:
        voice_runtime = self._get_voice_runtime()
        audio, sample_rate = voice_runtime.read_audio_file(batch.path)
        report = voice_runtime.process_audio(
            audio,
            sample_rate,
            session_start=batch.started_at.isoformat(),
            audio_path=batch.path,
        )
        checksum = hashlib.sha256(batch.path.read_bytes()).hexdigest()
        relative_path = str(batch.path.relative_to(self._recordings_dir))
        segments: List[Dict[str, Any]] = []
        for segment in report.get("segments") or []:
            item = dict(segment)
            metadata = dict(item.get("metadata") or {})
            metadata.update({
                "recording_id": self._recording_id,
                "audio_source": batch.source,
                "ambient_asr_watermark_managed": True,
                "audio_file": relative_path,
                "audio_batch_index": batch.index,
                "audio_cut_reason": batch.cut_reason,
                "audio_timeline_policy": "sample_count",
                "audio_shared_cut_at": batch.shared_cut_at.isoformat() if batch.shared_cut_at else None,
                "audio_sha256": checksum,
            })
            item["metadata"] = metadata
            segments.append(item)
        result = {
            "batchId": f"{self._recording_id}:{batch.source}:{batch.index}",
            "recordingId": self._recording_id,
            "batchIndex": batch.index,
            "audioFile": relative_path,
            "audioSha256": checksum,
            "startedAt": batch.started_at.isoformat(),
            "endedAt": batch.ended_at.isoformat(),
            "durationSeconds": round(batch.sample_count / batch.sample_rate, 3),
            "audioSampleRate": batch.sample_rate,
            "speechSegmentCount": report.get("speech_segment_count", 0),
            "transcriptSegmentCount": len(segments),
            "source": batch.source,
            "cutReason": batch.cut_reason,
            "sharedCutAt": batch.shared_cut_at.isoformat() if batch.shared_cut_at else None,
            "segments": segments,
            "transcriptFile": batch.path.with_suffix(".asr.json").name,
            "transcriptSaveStatus": "saved",
        }
        # Persist the full result before the main loop pops segments for PWM.
        # File I/O runs on the ASR worker, never on the PCM append path.
        try:
            self._save_transcript_result(batch.path.with_suffix(".asr.json"), result)
            self._logger.info("ambient ASR result saved recording_id=%s source=%s batch=%s path=%s segments=%s",
                              self._recording_id, batch.source, batch.index,
                              result["transcriptFile"], len(segments))
        except Exception:
            # Recognition succeeded. Keep its text and coverage available to
            # PWM even when the independent JSON artifact cannot be written.
            result["transcriptSaveStatus"] = "failed"
            result["transcriptSaveError"] = "asr_result_write_failed"
            self._logger.exception("ambient ASR result save failed recording_id=%s source=%s batch=%s path=%s",
                                   self._recording_id, batch.source, batch.index, result["transcriptFile"])
        return result

    def _save_transcript_result(self, path: Path, result: Dict[str, Any]) -> None:
        temporary = path.with_name(path.name + ".tmp")
        try:
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(path)
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                self._logger.warning("ambient ASR temporary result cleanup failed path=%s", temporary.name)

    def _get_voice_runtime(self) -> VoiceRuntime:
        if self._voice_runtime is None:
            self._logger.info("initializing offline ASR for sealed ambient batch")
            self._voice_runtime = VoiceRuntime(
                self._voice_config,
                logger=self._logger.getChild("voice"),
            )
            self._asr_sample_rate = int(self._voice_runtime._sample_rate())
        return self._voice_runtime

    def _harvest_completed(self) -> None:
        if not self._stopping:
            self._harvest_split_analysis()
        remaining: List[Future[Dict[str, Any]]] = []
        for future in self._pending:
            if not future.done():
                remaining.append(future)
                continue
            try:
                completed = future.result()
                archive = self._transcript_results[completed["audioFile"]]
                archive.update(asrStatus="completed", saveStatus=completed["transcriptSaveStatus"],
                               transcriptSegmentCount=completed["transcriptSegmentCount"])
                if completed["transcriptSaveStatus"] == "failed":
                    self._failed_transcript_saves += 1
                    archive["errorCode"] = completed["transcriptSaveError"]
                self._asr_progress[completed["source"]] = _parse_timestamp(completed["endedAt"])
                self._buffered_segments.extend(completed.pop("segments"))
                self._completed.append(completed)
                self._logger.info(
                    "ambient audio batch transcribed recording_id=%s batch=%s segments=%s",
                    self._recording_id,
                    completed["batchIndex"],
                    completed["transcriptSegmentCount"],
                )
            except Exception:
                failed_batch = self._pending_batches[future]
                self._transcript_results[failed_batch.path.name].update(
                    asrStatus="failed", saveStatus="not_generated", errorCode="asr_failed")
                failed_start = failed_batch.started_at
                self._failed_until = min(self._failed_until, failed_start) if self._failed_until else failed_start
                self._failed_batches += 1
                self._logger.exception(
                    "ambient audio batch transcription failed recording_id=%s",
                    self._recording_id,
                )
            self._pending_batches.pop(future, None)
        self._pending = remaining
        if self._delivery_future is not None and self._delivery_future.done():
            # A failed/lost response leaves the exact payload intact for retry;
            # new segments must never be attached to the same batch id.
            try:
                delivery = self._delivery_future.result()
                if delivery.get("ok") is False:
                    if delivery.get("code") != "stale_memory_generation":
                        raise RuntimeError("Memory rejected an ambient transcript delivery")
                    # A reset permanently invalidated this input. Acknowledge
                    # the discard rather than retrying stale text forever.
                    self._logger.info("ambient transcript discarded after memory reset recording_id=%s", self._recording_id)
                self._delivery_index += 1
                self._published_until = self._delivery_until
                self._delivery_payload = None
            finally:
                self._delivery_future = None
        self._publish_transcripts(final=self._stopping and not self._pending)
        self._write_manifest()

    def _publish_transcripts(self, *, final: bool = False) -> None:
        if not self._tracks or self._delivery_future is not None:
            return
        if self._delivery_payload is not None:
            self._delivery_future = self._executor.submit(
                self._agent_memory_ipc.request, "observe_transcript_segments", self._delivery_payload,
            )
            return
        # Only advance past time covered by every source. Buffer the faster
        # source so the PWM assembler never receives a later answer before an
        # earlier question. Empty ASR batches still advance coverage.
        covered_until = min(self._asr_progress.values())
        if self._failed_until:
            covered_until = min(covered_until, self._failed_until)
        ordered = sorted(self._buffered_segments, key=lambda segment: (
            _parse_timestamp(segment["started_at"]),
            _parse_timestamp(segment["ended_at"]),
            segment.get("metadata", {}).get("audio_source", ""),
        ))
        count = 0
        for segment in ordered:
            if not final and _parse_timestamp(segment["ended_at"]) > covered_until:
                break
            count += 1
        if not final and count < len(ordered):
            covered_until = min(covered_until, _parse_timestamp(ordered[count]["started_at"]))
        if final and not self._failed_batches:
            covered_until = max(self._asr_progress.values())
        if count == 0 and covered_until <= self._published_until:
            return
        batch_id = f"{self._recording_id}:merged:{self._delivery_index + 1}"
        # IPC can wait on the PWM writer. Run it in the serial worker rather
        # than blocking the PCM append loop and losing incoming audio.
        self._delivery_payload = {
            "ownerId": self._agent_memory_ipc.owner_id,
            "recordingId": self._recording_id,
            "batchId": batch_id,
            "memoryGeneration": self._memory_generation,
            "segments": ordered[:count],
            "processedThrough": covered_until.isoformat(),
            "tags": ["qwen-audio-agent", "ambient-recording", self._recording_id],
        }
        self._delivery_future = self._executor.submit(
            self._agent_memory_ipc.request, "observe_transcript_segments", self._delivery_payload,
        )
        self._delivery_until = covered_until
        self._buffered_segments = ordered[count:]

    def _finalize_memory_recording(self) -> None:
        if self._memory_finalized:
            return
        result = self._agent_memory_ipc.request(
            "finalize_transcript_recording",
            {
                "ownerId": self._agent_memory_ipc.owner_id,
                "recordingId": self._recording_id,
                "memoryGeneration": self._memory_generation,
            },
        )
        if result.get("ok") is False and result.get("code") != "stale_memory_generation":
            raise RuntimeError("Memory rejected ambient recording finalization")
        self._memory_finalized = True
        self._logger.info(
            "ambient recording memory finalized recording_id=%s user_activity_queued=%s",
            self._recording_id,
            bool(result.get("userActivityQueued")),
        )

    def _result(self, *, include_completed: bool = False) -> Dict[str, Any]:
        completed = list(self._completed) if include_completed else []
        if include_completed:
            self._completed.clear()
        return {
            "recordingId": self._recording_id,
            "recording": bool(self._recording_id and not self._stopping),
            "stopping": bool(self._recording_id and self._stopping),
            "sources": list(self._tracks),
            "splitPolicy": "joint_silence",
            "timelinePolicy": "sample_count",
            "pendingSplitAnalysis": len(self._split_pending),
            "tracks": {source: {
                "sampleRate": track.sample_rate or 0,
                "openBatchSeconds": round(track.batch_samples / track.sample_rate, 3) if track.sample_rate else 0.0,
            } for source, track in self._tracks.items()},
            "pendingBatches": len(self._pending) + int(self._delivery_payload is not None),
            "failedBatches": self._failed_batches,
            "failedTranscriptSaves": self._failed_transcript_saves,
            "memoryFinalized": self._memory_finalized,
            "completedBatches": completed,
        }

    def _write_manifest(self) -> None:
        if self._manifest_path is None:
            return
        payload = {
            "recording_id": self._recording_id,
            "started_at": self._recording_started_at.isoformat() if self._recording_started_at else "",
            "tracks": {source: {
                "sample_rate": track.sample_rate,
                "open_batch_seconds": round(track.batch_samples / track.sample_rate, 3) if track.sample_rate else 0.0,
            } for source, track in self._tracks.items()},
            "asr_sample_rate": self._asr_sample_rate,
            "chunk_duration_seconds": self._chunk_duration_seconds,
            "split_policy": "joint_silence",
            "timeline_policy": "sample_count",
            "split_silence_seconds": self._split_silence_seconds,
            "split_quiet_max_rms_dbfs": self._split_quiet_max_rms_dbfs,
            "shared_split_events": list(self._split_events),
            "stopping": self._stopping,
            "pending_batches": len(self._pending) + int(self._delivery_payload is not None),
            "failed_batches": self._failed_batches,
            "failed_transcript_saves": self._failed_transcript_saves,
            "transcripts": list(self._transcript_results.values()),
            "audio_files": list(self._audio_file_names),
        }
        target = self._manifest_path
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(target)

    def _recording_short_id(self) -> str:
        compact = "".join(character for character in self._recording_id if character.isalnum())
        return compact[:6] or "record"


def _write(payload: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    sys.stdout.flush()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--state-dir",
        required=True,
        type=Path,
        help="Agent Memory runtime state directory.",
    )
    parser.add_argument("--config", default=REPO_ROOT / "config.yaml", type=Path)
    parser.add_argument("--recordings-dir", required=True, type=Path)
    parser.add_argument("--chunk-duration-seconds", type=float, default=600.0,
                        help="Minimum batch duration before seeking joint silence; no hard limit.")
    parser.add_argument("--agent-memory-ipc-socket", required=True, type=Path)
    parser.add_argument("--agent-memory-owner-id", required=True)
    parser.add_argument("--log-path", type=Path)
    args = parser.parse_args()
    if args.chunk_duration_seconds <= 0:
        raise ValueError("chunk-duration-seconds must be positive")
    log_path = args.log_path.expanduser().resolve() if args.log_path else None
    logger = logging.getLogger("agent_memory.ambient_recording")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(formatter)
    logger.addHandler(stderr_handler)
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    state_dir = args.state_dir.expanduser().resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    config = _load_config(args.config.expanduser().resolve())
    voice_config = load_voice_runtime_config(config)
    speaker_config = voice_config.get("speaker_identification")
    if not isinstance(speaker_config, dict):
        speaker_config = {}
        voice_config["speaker_identification"] = speaker_config
    speaker_config["state_dir"] = str(state_dir)
    service = AmbientAudioRecordingService(
        voice_config=voice_config,
        recordings_dir=args.recordings_dir,
        chunk_duration_seconds=args.chunk_duration_seconds,
        split_silence_seconds=float((voice_config.get("ambient_recording") or {}).get("split_silence_seconds", 0.8)),
        split_quiet_max_rms_dbfs=float((voice_config.get("ambient_recording") or {}).get("split_quiet_max_rms_dbfs", -35.0)),
        agent_memory_ipc=_AgentMemoryIpcClient(
            args.agent_memory_ipc_socket,
            args.agent_memory_owner_id,
        ),
        logger=logger,
    )
    logger.info(
        "ambient recording sidecar started state_dir=%s recordings_dir=%s "
        "chunk_duration_seconds=%.3f",
        state_dir,
        args.recordings_dir,
        args.chunk_duration_seconds,
    )
    methods = {
        "recording.start": service.start,
        "audio.append": service.append_audio,
        "recording.stop": service.stop,
        "recording.drain": service.drain,
        "recording.status": service.status,
        "close": service.close,
    }
    for line in sys.stdin:
        request: Optional[Dict[str, Any]] = None
        try:
            parsed = json.loads(line)
            if not isinstance(parsed, dict):
                raise ValueError("request must be a JSON object")
            request = parsed
            method = _clean(request.get("method"), 80)
            handler = methods.get(method)
            if handler is None:
                raise ValueError(f"unsupported method: {method}")
            response = {"id": request.get("id"), "result": handler(request.get("params") or {})}
        except Exception as exc:
            logger.exception("ambient recording sidecar request failed")
            response = {
                "id": request.get("id") if request else None,
                "error": f"{type(exc).__name__}: {exc}",
            }
        _write(response)
        if request and request.get("method") == "close":
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
