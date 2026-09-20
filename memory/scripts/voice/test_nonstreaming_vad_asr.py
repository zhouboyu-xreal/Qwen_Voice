#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
for import_root in (SRC_ROOT, ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from memory.config import split_memory_config  # noqa: E402
from voice.voice_runtime import VoiceRuntime  # noqa: E402


DEFAULT_CONFIG = ROOT / "config.yaml"
DEFAULT_AUDIO = (
    ROOT
    / "test_data/ambient_transcript/Eval_Ali/Eval_Ali_far/audio_dir/R8001_M8004_MS801.wav"
)
DEFAULT_RESULT_ROOT = ROOT / "tmp/nonstreaming_vad_asr"


@dataclass
class NonStreamingASREvent:
    segment_index: int
    start: float
    end: float
    duration: float
    frame_count: int
    speech_ratio: Optional[float]
    text: str


class RunLogger:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = run_dir / "run.log"
        self.events_path = run_dir / "events.jsonl"
        self._log_file = self.log_path.open("a", encoding="utf-8")
        self._events_file = self.events_path.open("a", encoding="utf-8")

    def log(self, message: str) -> None:
        print(message, flush=True)
        self._log_file.write(message + "\n")
        self._log_file.flush()

    def _log_level(self, level: str, message: str, *args: Any) -> None:
        if args:
            try:
                message = message % args
            except (TypeError, ValueError):
                message = " ".join([message, *(str(arg) for arg in args)])
        self.log(f"[{level}] {message}")

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._log_level("DEBUG", message, *args)

    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._log_level("INFO", message, *args)

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._log_level("WARNING", message, *args)

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._log_level("ERROR", message, *args)

    def exception(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._log_level("ERROR", message, *args)

    def event(self, event: NonStreamingASREvent) -> None:
        self._events_file.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")
        self._events_file.flush()

    def write_config(self, payload: Dict[str, Any]) -> None:
        (self.run_dir / "config.json").write_text(
            json.dumps(_json_safe(payload), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def close(self) -> None:
        self._log_file.close()
        self._events_file.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run frame-level VAD segmentation followed by non-streaming Whisper ASR."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--audio",
        type=Path,
        default=None,
        help="Optional audio override for quick tests. Defaults to config.yaml.",
    )
    parser.add_argument(
        "--max-duration-s",
        type=float,
        default=None,
        help="Override voice_runtime.max_duration_s for this test run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("YAML config support requires PyYAML.") from exc
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Expected YAML mapping in config file: {config_path}")
    _, _, voice_runtime_config = split_memory_config(
        config,
        include_voice_runtime=True,
    )
    if args.max_duration_s is not None:
        voice_runtime_config["max_duration_s"] = args.max_duration_s

    runtime_config = voice_runtime_config
    vad_config = voice_runtime_config.get("vad") or {}
    asr_section = voice_runtime_config.get("asr") or {}
    asr_backend = str(asr_section.get("backend") or "whisper")
    asr_profiles = asr_section.get("backends") or {}
    asr_config = asr_profiles.get(asr_backend) or asr_section
    output_config = voice_runtime_config.get("output") or {}
    if args.audio is not None:
        audio_path = args.audio.expanduser().resolve()
    else:
        audio_path = _resolve_path(
            runtime_config.get("audio") or DEFAULT_AUDIO,
            config_path.parent,
        )

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    result_root = DEFAULT_RESULT_ROOT
    run_dir = _create_run_dir(result_root, audio_path)
    logger = RunLogger(run_dir)
    logger.write_config({
        "config_path": config_path,
        "voice_runtime_config": voice_runtime_config,
        "audio": audio_path,
    })
    logger.log(f"Run directory: {run_dir}")

    try:
        voice_runtime = VoiceRuntime(voice_runtime_config, logger=logger)
        logger.log(
            "Non-streaming VAD+ASR test: "
            f"audio={audio_path}, frame={float(vad_config.get('frame_ms', 32.0)):.1f}ms, "
            f"sample_rate={int(runtime_config.get('sample_rate', 16000))}, "
            f"backend={asr_backend}, model={asr_config.get('model_id') or 'default'}"
        )

        voice_report = voice_runtime.process_audio_file(audio_path)
        events = _events_from_voice_report(voice_report, voice_runtime_config)
        for event in events:
            logger.event(event)
            if event.text or bool(output_config.get("print_empty", False)):
                logger.log(_format_asr_event(event))
        output_jsonl = output_config.get("output_jsonl")
        if output_jsonl:
            _write_jsonl(_resolve_path(output_jsonl, config_path.parent), events)
        logger.log(f"Saved log to: {logger.log_path}")
        logger.log(f"Saved events to: {logger.events_path}")
        logger.log(f"Finished with {len(events)} ASR segments.")
    finally:
        if "voice_runtime" in locals():
            voice_runtime.close()
        logger.close()


def _events_from_voice_report(
    report: Dict[str, Any],
    voice_runtime_config: Dict[str, Any],
) -> List[NonStreamingASREvent]:
    """Preserve the script's event output format from VoiceRuntime segments."""
    events: List[NonStreamingASREvent] = []
    vad_config = voice_runtime_config.get("vad") or {}
    frame_ms = max(1.0, float(vad_config.get("frame_ms", 32.0)))
    for index, segment in enumerate(report.get("segments") or [], 1):
        metadata = segment.get("metadata") or {}
        start = float(metadata.get("audio_start_s") or 0.0)
        end = float(metadata.get("audio_end_s") or start)
        duration = max(0.0, end - start)
        events.append(
            NonStreamingASREvent(
                segment_index=int(segment.get("segment_index") or index),
                start=round(start, 3),
                end=round(end, 3),
                duration=round(duration, 3),
                frame_count=max(1, round(duration * 1000.0 / frame_ms)),
                speech_ratio=None,
                text=str(segment.get("text") or ""),
            )
        )
    return events


def _resolve_path(value: Any, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _format_asr_event(event: NonStreamingASREvent) -> str:
    speech_ratio = (
        f"{event.speech_ratio:.2f}"
        if event.speech_ratio is not None
        else "n/a"
    )
    return (
        f"[ASR SEGMENT {event.segment_index:04d}] "
        f"time={event.start:8.3f}-{event.end:8.3f}s "
        f"duration={event.duration:6.3f}s "
        f"frames={event.frame_count} speech_ratio={speech_ratio} "
        f"text={event.text}"
    )


def _write_jsonl(output_path: Path, events: List[NonStreamingASREvent]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for event in events:
            file.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")
    print(f"Saved JSONL events to: {output_path}", flush=True)


def _create_run_dir(result_root: Path, audio_path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    audio_name = _safe_name(audio_path.stem)
    run_dir = result_root / f"{timestamp}_{audio_name}"
    suffix = 1
    while run_dir.exists():
        run_dir = result_root / f"{timestamp}_{audio_name}_{suffix:02d}"
        suffix += 1
    run_dir.mkdir(parents=True)
    return run_dir


def _safe_name(value: str) -> str:
    safe_chars = [
        char if char.isalnum() or char in ("-", "_", ".") else "_"
        for char in value
    ]
    return "".join(safe_chars).strip("._") or "audio"


def _json_safe(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return _json_safe(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


if __name__ == "__main__":
    main()
