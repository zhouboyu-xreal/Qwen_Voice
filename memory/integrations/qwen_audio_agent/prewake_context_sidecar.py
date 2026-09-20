#!/usr/bin/env python3
"""Local JSONL bridge for :mod:`voice.prewake_context`.

The process is intended to be launched by a local voice client while it is
listening for a wake word. PCM is retained only in process memory. The sole
``wake.snapshot`` result is a guarded text prompt for the next realtime turn;
it is not connected to Agent Memory storage.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from memory.config import split_memory_config
from voice.prewake_context import PreWakeContextRuntime, format_prewake_context_prompt


def _load_config(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - project dependency
        raise RuntimeError("PyYAML is required to load config.yaml") from exc
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError("config.yaml must contain a mapping")
    return value


def _finite_number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


class PreWakeContextService:
    def __init__(self, voice_config: Dict[str, Any], logger: logging.Logger) -> None:
        self._runtime = PreWakeContextRuntime(voice_config, logger=logger)
        self._logger = logger
        self._audio_requests = 0
        self._audio_bytes = 0
        self._last_audio_progress_log_at = time.monotonic()
        self._audio_progress_interval_seconds = 5.0
        self._logger.info(
            "prewake runtime ready sample_rate=%s asr_backend=%s buffer_seconds=%s "
            "guard_seconds=%s max_context_characters=%s",
            self._runtime.sample_rate,
            self._runtime.diagnostics()["asr_backend"],
            self._runtime.buffer.buffer_seconds,
            self._runtime.buffer.guard_seconds,
            self._runtime.buffer.max_context_characters,
        )

    def _log_audio_progress(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_audio_progress_log_at < self._audio_progress_interval_seconds:
            return
        self._last_audio_progress_log_at = now
        stats = self._runtime.diagnostics()
        self._logger.info(
            "prewake audio progress requests=%s received_bytes=%s received_seconds=%.3f "
            "buffered_seconds=%.3f vad_active=%s pending_asr=%s vad_speech_starts=%s "
            "asr_submitted=%s asr_completed=%s asr_failed=%s asr_dropped=%s",
            self._audio_requests,
            self._audio_bytes,
            stats["accepted_audio_seconds"],
            stats["buffered_audio_seconds"],
            stats["vad_active"],
            stats["pending_asr_segments"],
            stats["vad_speech_starts"],
            stats["asr_submitted_segments"],
            stats["asr_completed_segments"],
            stats["asr_failed_segments"],
            stats["asr_dropped_segments"],
        )

    def append_audio(self, params: Dict[str, Any]) -> Dict[str, Any]:
        sample_rate = int(params.get("sampleRate") or self._runtime.sample_rate)
        if sample_rate != self._runtime.sample_rate:
            raise ValueError(
                f"sampleRate={sample_rate} does not match configured "
                f"{self._runtime.sample_rate} Hz"
            )
        encoded = str(params.get("audio") or "")
        if not encoded:
            return {"accepted": False, "reason": "empty_audio"}
        try:
            pcm16 = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("audio must be base64 PCM16") from exc
        if len(pcm16) > 256 * 1024:
            raise ValueError("audio chunk exceeds 256 KiB")
        ended_at = _finite_number(params.get("endedAt"))
        self._runtime.accept_pcm16(pcm16, ended_at=ended_at)
        self._audio_requests += 1
        self._audio_bytes += len(pcm16)
        self._log_audio_progress()
        return {
            "accepted": True,
            "bufferedAudioSeconds": round(self._runtime.buffer.buffered_audio_seconds, 3),
        }

    def wake_snapshot(self, params: Dict[str, Any]) -> Dict[str, Any]:
        snapshot = self._runtime.consume_for_wake(
            wake_at=_finite_number(params.get("wakeAt")),
        )
        result = {
            "text": snapshot.text,
            "prompt": format_prewake_context_prompt(snapshot),
            "segmentCount": snapshot.segment_count,
            "droppedSegmentCount": snapshot.dropped_segment_count,
            "audioDurationSeconds": round(snapshot.audio_duration_seconds, 3),
            "windowStartedAt": snapshot.window_started_at,
            "windowEndedAt": snapshot.window_ended_at,
        }
        self._logger.info(
            "prewake snapshot segments=%s dropped=%s text_chars=%s audio_seconds=%.3f "
            "requests=%s submitted_asr=%s completed_asr=%s text=%r",
            snapshot.segment_count,
            snapshot.dropped_segment_count,
            len(snapshot.text),
            snapshot.audio_duration_seconds,
            self._audio_requests,
            self._runtime.diagnostics()["asr_submitted_segments"],
            self._runtime.diagnostics()["asr_completed_segments"],
            snapshot.text,
        )
        return result

    def clear(self, _params: Dict[str, Any]) -> Dict[str, Any]:
        self._log_audio_progress(force=True)
        self._runtime.clear()
        self._logger.info("prewake buffer cleared")
        return {"cleared": True}

    def health(self, _params: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "ok": True,
            "sampleRate": self._runtime.sample_rate,
            "bufferedAudioSeconds": round(self._runtime.buffer.buffered_audio_seconds, 3),
            "diagnostics": self._runtime.diagnostics(),
        }

    def close(self, _params: Dict[str, Any]) -> Dict[str, Any]:
        self._log_audio_progress(force=True)
        self._runtime.close()
        self._logger.info("prewake sidecar closed")
        return {"closed": True}


def _write(payload: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=REPO_ROOT / "config.yaml", type=Path)
    parser.add_argument("--log-path", type=Path)
    args = parser.parse_args()
    log_path = args.log_path.expanduser().resolve() if args.log_path else None
    logger = logging.getLogger("agent_memory.prewake_context")
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
    service = PreWakeContextService(voice_config, logger)
    logger.info("prewake sidecar started config=%s log_path=%s", args.config, log_path)
    methods = {
        "audio.append": service.append_audio,
        "wake.snapshot": service.wake_snapshot,
        "clear": service.clear,
        "health": service.health,
        "close": service.close,
    }
    for line in sys.stdin:
        request_id = None
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("request must be an object")
            request_id = request.get("id")
            method = str(request.get("method") or "")
            handler = methods.get(method)
            if handler is None:
                raise ValueError(f"unsupported method: {method}")
            params = request.get("params")
            result = handler(params if isinstance(params, dict) else {})
            _write({"id": request_id, "ok": True, "result": result})
            if method == "close":
                return 0
        except Exception as exc:  # keep the JSONL protocol alive after one bad request
            logger.exception("prewake sidecar request failed")
            _write({"id": request_id, "ok": False, "error": str(exc)})
    service.close({})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
