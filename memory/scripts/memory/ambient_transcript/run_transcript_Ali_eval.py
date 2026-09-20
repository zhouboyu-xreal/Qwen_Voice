#!/usr/bin/env python3
"""Store Eval_Ali TextGrid transcripts through the unified memory manager."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from memory.memory_manager import (
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
    MemoryOperationReporter,
)
from memory.memory_runtime import MemoryRuntime
from memory.config import split_memory_config


DEFAULT_TEXTGRID = (
    REPO_ROOT
    / "test_data/ambient_transcript/Eval_Ali/Eval_Ali_far/textgrid_dir/R8001_M8004.TextGrid"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "tmp/ambient_transcript/Ali_eval"

_ENV_REF_RE = re.compile(r"\${([A-Za-z_][A-Za-z0-9_]*)}")
_NAME_RE = re.compile(r'name\s*=\s*"(?P<name>.*)"')
_XMIN_RE = re.compile(r"xmin\s*=\s*(?P<value>[-+]?\d+(?:\.\d+)?)")
_XMAX_RE = re.compile(r"xmax\s*=\s*(?P<value>[-+]?\d+(?:\.\d+)?)")
_TEXT_RE = re.compile(r'text\s*=\s*"(?P<text>(?:[^"]|"")*)"')


@dataclass
class TextGridSegment:
    speaker: str
    start_s: float
    end_s: float
    text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read Eval_Ali TextGrid intervals in time order and store them as "
            "ambient transcript memory episodes."
        )
    )
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config.yaml")
    parser.add_argument("--textgrid", type=Path, default=DEFAULT_TEXTGRID)
    parser.add_argument("--output-root-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--db-name", default="memory.db")
    parser.add_argument("--session-start", default="", help="ISO timestamp used as audio time zero.")
    parser.add_argument("--source-type", default="allday_recording")
    parser.add_argument("--max-pending-transcript-units", type=int, default=0)
    parser.add_argument("--max-pending-transcript-tokens", type=int, default=0)
    parser.add_argument("--min-pending-transcript-units", type=int, default=0)
    parser.add_argument("--min-pending-transcript-tokens", type=int, default=0)
    parser.add_argument("--max-gap-seconds", type=float, default=-1.0)
    parser.add_argument("--enable-reflect", action="store_true", help="Run memory reflection after storing episodes.")
    parser.add_argument("--disable-llm", action="store_true", help="Use heuristic fallback extraction only.")
    parser.add_argument("--llm-model")
    parser.add_argument("--llm-base-url")
    parser.add_argument("--llm-api-key")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_project_config(args.config)
    (
        memory_runtime_config,
        memory_manager_config,
    ) = split_memory_config(config)
    llm_config = memory_manager_config["llm"]
    embedding_config = memory_manager_config["embedding"]
    if args.disable_llm:
        llm_config["llm_api_key"] = ""
    transcript_segmentation_config = memory_runtime_config.setdefault(
        "allday_recording_segmentation",
        {},
    )
    if not isinstance(transcript_segmentation_config, dict):
        raise ValueError(
            "memory_runtime.allday_recording_segmentation must be a mapping",
        )
    if args.max_pending_transcript_units:
        transcript_segmentation_config["max_pending_transcript_units"] = (
            args.max_pending_transcript_units
        )
    if args.max_pending_transcript_tokens:
        transcript_segmentation_config["max_pending_transcript_tokens"] = (
            args.max_pending_transcript_tokens
        )
    if args.min_pending_transcript_units:
        transcript_segmentation_config["min_pending_transcript_units"] = (
            args.min_pending_transcript_units
        )
    if args.min_pending_transcript_tokens:
        transcript_segmentation_config["min_pending_transcript_tokens"] = (
            args.min_pending_transcript_tokens
        )
    if args.max_gap_seconds >= 0:
        transcript_segmentation_config["max_time_gap_seconds"] = args.max_gap_seconds

    textgrid_path = args.textgrid.expanduser().resolve()
    if not textgrid_path.exists():
        raise FileNotFoundError(f"TextGrid file not found: {textgrid_path}")
    segments = load_textgrid_segments(textgrid_path)
    if not segments:
        raise RuntimeError(f"No non-empty intervals found in TextGrid: {textgrid_path}")

    output_dir = resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(output_dir / "run.log", args.log_level)
    memory_log_path = output_dir / "memory_manager.log"
    memory_logger, memory_log_handler = configure_memory_logger(
        memory_log_path,
        args.log_level,
    )
    db_path = output_dir / args.db_name
    transcript_path = output_dir / "transcript.txt"
    report_path = output_dir / "report.json"

    session_start = parse_session_start(args.session_start)
    memory_segments = [
        textgrid_segment_to_memory_segment(item, session_start=session_start, index=index)
        for index, item in enumerate(segments, 1)
    ]
    write_transcript_txt(transcript_path, memory_segments)

    llm_config["llm_name"] = args.llm_model or llm_config.get("llm_name") or DEFAULT_LLM_MODEL
    llm_config["llm_base_url"] = args.llm_base_url or llm_config.get("llm_base_url") or DEFAULT_LLM_BASE_URL
    llm_config["llm_api_key"] = args.llm_api_key or llm_config.get("llm_api_key") or ""
    llm_config["llm_api_key"] = expand_env_refs(llm_config["llm_api_key"])

    operation_reporter = MemoryOperationReporter()
    runtime = MemoryRuntime(
        db_path=db_path,
        memory_runtime_config=memory_runtime_config,
        memory_manager_config=memory_manager_config,
        operation_reporter=operation_reporter,
        logger=memory_logger,
    )

    queued_episode_count = 0
    for segment_index, segment in enumerate(memory_segments, 1):
        ok = runtime.accept_single_transcript_segment(
            segment,
            source_type=args.source_type,
            tags=["Eval_Ali", textgrid_path.stem],
        )
        queued_episode_count += int(bool(ok.get("queued")))
        if ok.get("queued"):
            logging.info(
                "Queued transcript episode while processing segment %s/%s",
                segment_index,
                len(memory_segments),
            )

    reflect_result: Dict[str, Any] = {}
    if args.enable_reflect:
        reflect_timestamp = (
            memory_segments[-1].get("ended_at")
            if memory_segments
            else session_start.isoformat()
        )
        reflect_submit = runtime.trigger_memory_reflect(
            reflect_timestamp=reflect_timestamp,
        )
        if reflect_submit.get("queued") and not runtime.flush_pending_memory_inputs():
            raise RuntimeError("Timed out while draining queued memory reflect")
        reflect_result = operation_reporter.latest_report("memory_reflect") or reflect_submit
        queued_episode_count += int(
            bool((reflect_submit.get("pending_transcript_flush") or {}).get("queued"))
        )
        logging.info("Reflect result: %s", reflect_result)
    else:
        pending_transcript_count = (
            len(runtime._transcript_segmenter.pending_unit_snapshot())
            + int(runtime._transcript_utterance_assembler.has_pending_segments())
        )
        runtime.flush_pending_memory_inputs()
        queued_episode_count += int(
            pending_transcript_count > 0
            and not runtime._transcript_segmenter.has_pending_units()
            and not runtime._transcript_utterance_assembler.has_pending_segments()
        )
    store_operation_report = operation_reporter.operation_report("memory_store")
    logging.info(
        "Transcript input complete segments=%s pending=%s queued_episodes=%s stored_episodes=%s",
        len(memory_segments),
        len(runtime._transcript_segmenter.pending_unit_snapshot()),
        queued_episode_count,
        store_operation_report["succeeded"],
    )

    report = {
        "textgrid": str(textgrid_path),
        "db_path": str(db_path),
        "transcript_path": str(transcript_path),
        "memory_log_path": str(memory_log_path),
        "segment_count": len(memory_segments),
        "queued_episode_count": queued_episode_count,
        "stored_episode_count": store_operation_report["succeeded"],
        "store_operation_report": store_operation_report,
        "memory_operation_report": operation_reporter.snapshot(),
        "source_type": args.source_type,
        "session_start": session_start.isoformat(),
        "reflect_result": reflect_result,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info("Wrote report: %s", report_path)
    runtime.close()
    memory_logger.removeHandler(memory_log_handler)
    memory_log_handler.close()
    print(json.dumps(report, ensure_ascii=False, indent=2))


def load_project_config(config_path: Path) -> Dict[str, Any]:
    config_path = config_path.expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("YAML config support requires PyYAML.") from exc
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected YAML mapping in config file: {config_path}")
    return expand_env_refs(loaded)


def expand_env_refs(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: expand_env_refs(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_env_refs(item) for item in value]
    if isinstance(value, str):
        return _ENV_REF_RE.sub(lambda match: os.getenv(match.group(1), ""), value)
    return value


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return args.output_dir.expanduser().resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (args.output_root_dir / f"{timestamp}_{args.textgrid.stem}").expanduser().resolve()


def configure_logging(log_path: Path, log_level: str) -> None:
    level = getattr(logging, str(log_level or "INFO").upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def configure_memory_logger(
    log_path: Path,
    log_level: str,
) -> Tuple[logging.Logger, logging.Handler]:
    """Create a non-propagating logger for memory pipeline operations."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    memory_logger = logging.getLogger("memory.pipeline.ambient_transcript")
    for existing_handler in list(memory_logger.handlers):
        memory_logger.removeHandler(existing_handler)
        existing_handler.close()
    memory_logger.setLevel(getattr(logging, str(log_level or "INFO").upper(), logging.INFO))
    memory_logger.propagate = False
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    memory_logger.addHandler(handler)
    return memory_logger, handler


def parse_session_start(value: str) -> datetime:
    text = str(value or "").strip()
    if not text:
        return datetime.now(timezone.utc)
    normalized = text.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def load_textgrid_segments(textgrid_path: Path) -> List[TextGridSegment]:
    lines = textgrid_path.read_text(encoding="utf-8").splitlines()
    segments: List[TextGridSegment] = []
    current_tier = ""
    pending_start = None
    pending_end = None
    for raw_line in lines:
        line = raw_line.strip()
        name_match = _NAME_RE.fullmatch(line)
        if name_match:
            current_tier = unescape_text(name_match.group("name"))
            continue
        start_match = _XMIN_RE.fullmatch(line)
        if start_match:
            pending_start = float(start_match.group("value"))
            continue
        end_match = _XMAX_RE.fullmatch(line)
        if end_match:
            pending_end = float(end_match.group("value"))
            continue
        text_match = _TEXT_RE.fullmatch(line)
        if not text_match or pending_start is None or pending_end is None:
            continue
        text = unescape_text(text_match.group("text")).strip()
        if current_tier and text and pending_end > pending_start:
            segments.append(TextGridSegment(
                speaker=current_tier,
                start_s=pending_start,
                end_s=pending_end,
                text=text,
            ))
        pending_start = None
        pending_end = None
    return sorted(segments, key=lambda item: (item.start_s, item.end_s, item.speaker))


def unescape_text(value: str) -> str:
    return value.replace('""', '"')


def textgrid_segment_to_memory_segment(
    segment: TextGridSegment,
    *,
    session_start: datetime,
    index: int,
) -> Dict[str, Any]:
    started_at = session_start + timedelta(seconds=segment.start_s)
    ended_at = session_start + timedelta(seconds=segment.end_s)
    return {
        "speaker": segment.speaker,
        "role": "ambient_speaker",
        "text": segment.text,
        "started_at": started_at.isoformat(),
        "ended_at": ended_at.isoformat(),
        "segment_index": index,
        "metadata": {
            "audio_start_s": round(segment.start_s, 3),
            "audio_end_s": round(segment.end_s, 3),
        },
    }


def write_transcript_txt(path: Path, segments: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for segment in segments:
            meta = segment.get("metadata") or {}
            audio_start = float(meta.get("audio_start_s") or 0.0)
            audio_end = float(meta.get("audio_end_s") or 0.0)
            file.write(
                f"{segment.get('speaker')}\t"
                f"{audio_start:.3f}-{audio_end:.3f}\t"
                f"{segment.get('started_at')} - {segment.get('ended_at')}\t"
                f"{segment.get('text')}\n"
            )


if __name__ == "__main__":
    main()
