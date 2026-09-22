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

from memory.config import split_memory_config
from voice.voice_runtime import VoiceRuntime


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
    """Maintain one lossless recording and a serial offline-ASR worker."""

    def __init__(
        self,
        *,
        voice_config: Dict[str, Any],
        recordings_dir: Path,
        chunk_duration_seconds: float,
        agent_memory_ipc: _AgentMemoryIpcClient,
        logger: logging.Logger,
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
        self._recording_sample_rate: Optional[int] = None
        self._chunk_samples = 0
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
        self._batch_index = 0
        self._batch_started_at: Optional[datetime] = None
        self._batch_samples = 0
        self._writer: Optional[wave.Wave_write] = None
        self._partial_path: Optional[Path] = None
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
        self._recording_id = recording_id
        self._recording_started_at = _parse_timestamp(params.get("startedAt"))
        recording_stamp = self._recording_started_at.strftime("%Y%m%d_%H%M%S")
        self._manifest_path = self._recordings_dir / (
            f"{recording_stamp}_{self._recording_short_id()}.json"
        )
        self._audio_file_names = []
        self._batch_index = 0
        self._batch_started_at = self._recording_started_at
        self._batch_samples = 0
        self._recording_sample_rate = None
        self._chunk_samples = 0
        self._stopping = False
        self._memory_finalized = False
        self._write_manifest()
        self._logger.info(
            "ambient recording started recording_id=%s chunk_seconds=%.3f",
            recording_id,
            self._chunk_duration_seconds,
        )
        return self._result()

    def append_audio(self, params: Dict[str, Any]) -> Dict[str, Any]:
        self._require_active_recording(params)
        if self._stopping:
            raise RuntimeError("ambient recording is stopping")
        sample_rate = int(params.get("sampleRate") or 0)
        if sample_rate < 8_000 or sample_rate > 192_000:
            raise ValueError("ambient audio sampleRate must be between 8000 and 192000")
        if self._recording_sample_rate is None:
            self._recording_sample_rate = sample_rate
            self._chunk_samples = max(
                sample_rate,
                int(self._chunk_duration_seconds * sample_rate),
            )
            self._logger.info(
                "ambient recording capture format recording_id=%s sample_rate=%s",
                self._recording_id,
                sample_rate,
            )
        elif sample_rate != self._recording_sample_rate:
            raise ValueError(
                "ambient audio sampleRate changed within one recording: "
                f"got={sample_rate} expected={self._recording_sample_rate}"
            )
        try:
            payload = base64.b64decode(str(params.get("audio") or ""), validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("audio must be valid base64 PCM16") from exc
        if len(payload) % 2:
            payload = payload[:-1]
        offset = 0
        while offset < len(payload):
            self._ensure_writer()
            available_samples = self._chunk_samples - self._batch_samples
            take_bytes = min(len(payload) - offset, available_samples * 2)
            if take_bytes <= 0:
                self._seal_current_batch()
                continue
            chunk = payload[offset : offset + take_bytes]
            assert self._writer is not None
            # ``writeframes`` patches the WAV header after every append, making
            # a partially written file recoverable after an unexpected exit.
            self._writer.writeframes(chunk)
            self._batch_samples += len(chunk) // 2
            offset += len(chunk)
            if self._batch_samples >= self._chunk_samples:
                self._seal_current_batch()
        self._harvest_completed()
        return self._result(include_completed=True)

    def stop(self, params: Dict[str, Any]) -> Dict[str, Any]:
        self._require_active_recording(params)
        self._stopping = True
        if self._batch_samples > 0:
            self._seal_current_batch()
        else:
            self._discard_empty_partial()
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
        if self._stopping and not self._pending and not self._writer:
            self._finalize_memory_recording()
        return result

    def status(self, _params: Dict[str, Any]) -> Dict[str, Any]:
        self._harvest_completed()
        return self._result()

    def close(self, _params: Dict[str, Any]) -> Dict[str, Any]:
        if self._recording_id and not self._stopping:
            self.stop({"recordingId": self._recording_id})
        if self._voice_runtime is not None:
            self._voice_runtime.close()
        self._executor.shutdown(wait=False, cancel_futures=False)
        return {"closed": True}

    def _require_active_recording(self, params: Dict[str, Any]) -> None:
        recording_id = _clean(params.get("recordingId"), 128)
        if not self._recording_id or recording_id != self._recording_id:
            raise ValueError("recordingId does not identify the active recording")

    def _ensure_writer(self) -> None:
        if self._writer is not None:
            return
        if (
            self._batch_started_at is None
            or self._recording_sample_rate is None
            or self._chunk_samples <= 0
        ):
            raise RuntimeError("ambient recording has not started")
        self._batch_index += 1
        stem = (
            f"{self._batch_started_at.strftime('%Y%m%d_%H%M%S')}"
            f"_{self._batch_index:06d}_{self._recording_short_id()}"
        )
        self._partial_path = self._recordings_dir / f"{stem}.partial.wav"
        writer = wave.open(str(self._partial_path), "wb")
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(self._recording_sample_rate)
        self._writer = writer
        self._batch_samples = 0

    def _seal_current_batch(self) -> None:
        if self._writer is None or self._partial_path is None or self._batch_started_at is None:
            return
        writer = self._writer
        partial_path = self._partial_path
        sample_count = self._batch_samples
        started_at = self._batch_started_at
        self._writer = None
        self._partial_path = None
        self._batch_samples = 0
        writer.close()
        final_path = partial_path.with_name(partial_path.name.replace(".partial.wav", ".wav"))
        self._audio_file_names.append(final_path.name)
        partial_path.replace(final_path)
        assert self._recording_sample_rate is not None
        ended_at = started_at + timedelta(seconds=sample_count / self._recording_sample_rate)
        batch = _AudioBatch(
            index=self._batch_index,
            path=final_path,
            started_at=started_at,
            ended_at=ended_at,
            sample_count=sample_count,
            sample_rate=self._recording_sample_rate,
        )
        self._batch_started_at = ended_at
        self._pending.append(self._executor.submit(self._transcribe_batch, batch))
        self._write_manifest()
        self._logger.info(
            "ambient audio batch sealed recording_id=%s batch=%s duration_s=%.3f path=%s",
            self._recording_id,
            batch.index,
            sample_count / self._recording_sample_rate,
            final_path.name,
        )

    def _discard_empty_partial(self) -> None:
        if self._writer is not None:
            self._writer.close()
        if self._partial_path is not None:
            self._partial_path.unlink(missing_ok=True)
        self._writer = None
        self._partial_path = None

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
                "audio_file": relative_path,
                "audio_batch_index": batch.index,
                "audio_sha256": checksum,
            })
            item["metadata"] = metadata
            segments.append(item)
        memory_result = self._agent_memory_ipc.request(
            "observe_transcript_segments",
            {
                "ownerId": self._agent_memory_ipc.owner_id,
                "recordingId": self._recording_id,
                "batchId": f"{self._recording_id}:{batch.index}",
                "segments": segments,
                "tags": ["qwen-audio-agent", "ambient-recording", self._recording_id],
            },
        )
        return {
            "batchId": f"{self._recording_id}:{batch.index}",
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
            "memoryObserved": bool(memory_result.get("observed")),
        }

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
        remaining: List[Future[Dict[str, Any]]] = []
        for future in self._pending:
            if not future.done():
                remaining.append(future)
                continue
            try:
                completed = future.result()
                self._completed.append(completed)
                self._logger.info(
                    "ambient audio batch transcribed recording_id=%s batch=%s segments=%s",
                    self._recording_id,
                    completed["batchIndex"],
                    completed["transcriptSegmentCount"],
                )
            except Exception:
                self._failed_batches += 1
                self._logger.exception(
                    "ambient audio batch transcription failed recording_id=%s",
                    self._recording_id,
                )
        self._pending = remaining
        self._write_manifest()

    def _finalize_memory_recording(self) -> None:
        if self._memory_finalized:
            return
        result = self._agent_memory_ipc.request(
            "finalize_transcript_recording",
            {
                "ownerId": self._agent_memory_ipc.owner_id,
                "recordingId": self._recording_id,
            },
        )
        self._memory_finalized = True
        self._logger.info(
            "ambient recording memory finalized recording_id=%s episode_summary_queued=%s",
            self._recording_id,
            bool(result.get("episodeSummaryQueued")),
        )

    def _result(self, *, include_completed: bool = False) -> Dict[str, Any]:
        completed = list(self._completed) if include_completed else []
        if include_completed:
            self._completed.clear()
        return {
            "recordingId": self._recording_id,
            "recording": bool(self._recording_id and not self._stopping),
            "stopping": bool(self._recording_id and self._stopping),
            "sampleRate": self._recording_sample_rate or 0,
            "openBatchSeconds": round(
                self._batch_samples / self._recording_sample_rate,
                3,
            ) if self._recording_sample_rate else 0.0,
            "pendingBatches": len(self._pending),
            "failedBatches": self._failed_batches,
            "memoryFinalized": self._memory_finalized,
            "completedBatches": completed,
        }

    def _write_manifest(self) -> None:
        if self._manifest_path is None:
            return
        payload = {
            "recording_id": self._recording_id,
            "started_at": self._recording_started_at.isoformat() if self._recording_started_at else "",
            "sample_rate": self._recording_sample_rate,
            "asr_sample_rate": self._asr_sample_rate,
            "chunk_duration_seconds": self._chunk_duration_seconds,
            "stopping": self._stopping,
            "open_batch_seconds": round(
                self._batch_samples / self._recording_sample_rate,
                3,
            ) if self._recording_sample_rate else 0.0,
            "pending_batches": len(self._pending),
            "failed_batches": self._failed_batches,
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
    parser.add_argument("--config", default=REPO_ROOT / "config.yaml", type=Path)
    parser.add_argument("--recordings-dir", required=True, type=Path)
    parser.add_argument("--chunk-duration-seconds", type=float, default=600.0)
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

    config = _load_config(args.config.expanduser().resolve())
    _runtime_config, _manager_config, voice_config = split_memory_config(
        config,
        include_voice_runtime=True,
    )
    service = AmbientAudioRecordingService(
        voice_config=voice_config,
        recordings_dir=args.recordings_dir,
        chunk_duration_seconds=args.chunk_duration_seconds,
        agent_memory_ipc=_AgentMemoryIpcClient(
            args.agent_memory_ipc_socket,
            args.agent_memory_owner_id,
        ),
        logger=logger,
    )
    logger.info(
        "ambient recording sidecar started recordings_dir=%s chunk_duration_seconds=%.3f",
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
