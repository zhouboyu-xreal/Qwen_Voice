#!/usr/bin/env python3
"""Continuously extract memory from multiple Eval_Ali TextGrid files.

All files share one PersonalWorldModel and one pair of databases. Each file is
treated as a separate session. The first speaker to produce a valid, nonempty
interval in each file is mapped to ``speaker_用户本人``. This is a synthetic
identity experiment: different files need not actually contain the same person.
TextGrid timestamps are relative, so the runner assigns a non-overlapping
synthetic timeline and records it in the report.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config import split_memory_config
from pwm_manager import PersonWorldModel
from task_scheduler import TaskOperationReporter
from run_transcript_Ali_eval import (
    TextGridSegment,
    configure_logging,
    configure_memory_logger,
    load_textgrid_segments,
    parse_session_start,
    textgrid_segment_to_memory_segment,
    write_transcript_txt,
)


DEFAULT_TEXTGRID_DIR = (
    REPO_ROOT / "test_data/ambient_transcript/Eval_Ali/Eval_Ali_far/textgrid_dir"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "tmp/ambient_transcript/Ali_eval_continuous"
MEMORY_COUNT_TABLES = (
    "memory_source_segments",
    "memory_facts",
    "memory_user_activity_items",
    "memory_user_claims",
    "memory_user_observed_patterns",
    "memory_user_intent_items",
)
REJECTED_INPUT_REASONS = {
    "expected_one_input_kind",
    "memory_disabled",
    "empty_turn",
    "empty_transcript_batch",
    "queue_rejected",
    "worker_queue_full",
    "worker_shutdown",
    "task_handler_not_registered",
    "task_disabled",
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config.yaml")
    parser.add_argument("--textgrid-dir", type=Path, default=DEFAULT_TEXTGRID_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--file-limit", type=int, default=0,
        help="Maximum number of TextGrid files to process; 0 means all files.",
    )
    parser.add_argument(
        "--session-start", default="",
        help="ISO time assigned to the first file's audio zero; defaults to a recent synthetic time.",
    )
    parser.add_argument(
        "--inter-file-gap-hours", type=float, default=24.0,
        help="Synthetic gap between the previous file's end and the next file's start.",
    )
    parser.add_argument("--llm-model")
    parser.add_argument("--llm-base-url")
    parser.add_argument("--llm-api-key")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    if args.file_limit < 0:
        parser.error("--file-limit must be greater than or equal to 0")
    if args.inter_file_gap_hours < 0:
        parser.error("--inter-file-gap-hours must be greater than or equal to 0")
    return args


def select_textgrid_files(directory: Path, file_limit: int) -> List[Path]:
    if not directory.is_dir():
        raise NotADirectoryError(f"TextGrid directory does not exist: {directory}")
    files = sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() == ".textgrid"
    )
    if file_limit:
        files = files[:file_limit]
    if not files:
        raise RuntimeError(f"No TextGrid files selected from {directory}")
    return files


def map_file_speakers(
    segments: Sequence[TextGridSegment], *, file_index: int,
) -> Tuple[List[TextGridSegment], str, Dict[str, str]]:
    """Choose the chronologically first valid speaker, not the first tier."""
    if not segments:
        raise ValueError("Cannot select a user speaker from an empty TextGrid")
    owner_speaker = min(
        segments, key=lambda item: (item.start_s, item.end_s, item.speaker)
    ).speaker
    other_speakers = sorted({item.speaker for item in segments} - {owner_speaker})
    mapping = {owner_speaker: "speaker_用户本人"}
    mapping.update({
        speaker: f"speaker_file{file_index:04d}_{index}"
        for index, speaker in enumerate(other_speakers, 1)
    })
    mapped = [
        TextGridSegment(
            speaker=mapping[item.speaker],
            start_s=item.start_s,
            end_s=item.end_s,
            text=item.text,
        )
        for item in segments
    ]
    return mapped, owner_speaker, mapping


def table_counts(pwm: PersonWorldModel) -> Dict[str, int]:
    counts = {
        table: int(pwm.database._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        for table in MEMORY_COUNT_TABLES
    }
    counts["user_states"] = int(
        pwm.user_model_database._conn.execute("SELECT COUNT(*) FROM user_states").fetchone()[0]
    )
    return counts


def operation_counts(reporter: TaskOperationReporter) -> Dict[str, Dict[str, int]]:
    return {
        kind: {
            field: int(values.get(field) or 0)
            for field in ("submitted", "completed", "succeeded", "failed", "rejected")
        }
        for kind, values in (reporter.snapshot().get("counts") or {}).items()
    }


def count_delta(
    before: Dict[str, Dict[str, int]], after: Dict[str, Dict[str, int]],
) -> Dict[str, Dict[str, int]]:
    return {
        kind: {
            field: value - before.get(kind, {}).get(field, 0)
            for field, value in values.items()
        }
        for kind, values in after.items()
        if values != before.get(kind, {})
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    textgrid_dir = args.textgrid_dir.expanduser().resolve()
    files = select_textgrid_files(textgrid_dir, args.file_limit)
    output_dir = (
        args.output_dir.expanduser().resolve() if args.output_dir else
        DEFAULT_OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(output_dir / "run.log", args.log_level)
    logging.info(
        "Continuous TextGrid extraction files=%s directory=%s file_limit=%s",
        len(files), textgrid_dir, args.file_limit,
    )

    sections = split_memory_config(args.config)
    llm_config = sections.llm_model
    for key, value in (
        ("llm_name", args.llm_model),
        ("llm_base_url", args.llm_base_url),
        ("llm_api_key", args.llm_api_key),
    ):
        if value is not None:
            llm_config[key] = value
    sections.memory_manager["user_memory_database_path"] = str(output_dir / "memory.db")
    sections.user_state["user_model_database_path"] = str(output_dir / "user_model.db")

    memory_logger, memory_log_handler = configure_memory_logger(
        output_dir / "memory_manager.log", args.log_level
    )
    reporter = TaskOperationReporter(recent_task_limit=10000)
    pwm: Optional[PersonWorldModel] = None
    results: List[Dict[str, Any]] = []
    if args.session_start:
        next_start = parse_session_start(args.session_start)
    else:
        next_start = datetime.now(timezone.utc) - timedelta(days=len(files))
    try:
        pwm = PersonWorldModel(
            memory_runtime_config=sections.memory_runtime,
            memory_manager_config=sections.memory_manager,
            user_state_config=sections.user_state,
            llm_model_config=llm_config,
            embedding_model_config=sections.embedding_model,
            task_scheduling_config=sections.task_scheduling,
            operation_reporter=reporter,
            logger=memory_logger,
        )
        for file_index, path in enumerate(files, 1):
            source_segments = load_textgrid_segments(path)
            if not source_segments:
                raise RuntimeError(f"No valid nonempty speech intervals: {path}")
            mapped_segments, owner_speaker, speaker_mapping = map_file_speakers(
                source_segments, file_index=file_index
            )
            memory_segments = [
                textgrid_segment_to_memory_segment(
                    item, session_start=next_start, index=index
                )
                for index, item in enumerate(mapped_segments, 1)
            ]
            transcript_path = output_dir / f"{file_index:04d}_{path.stem}_transcript.txt"
            write_transcript_txt(transcript_path, memory_segments)
            before = operation_counts(reporter)
            logging.info(
                "File started index=%s/%s name=%s segments=%s owner_speaker=%s session_start=%s",
                file_index, len(files), path.name, len(memory_segments), owner_speaker,
                next_start.isoformat(),
            )
            input_report = pwm.accept_memory_input(
                transcript_segments=memory_segments,
                tags=["Eval_Ali_continuous", f"file:{path.stem}"],
                ambient_recording_enabled=True,
            )
            if input_report.get("reason") in REJECTED_INPUT_REASONS:
                raise RuntimeError(f"Input rejected for {path.name}: {input_report}")
            finalization = pwm.finalize_memory_session(
                reason="ali_eval_file_boundary",
                tags=["Eval_Ali_continuous", f"file:{path.stem}"],
            )
            if not finalization.get("accepted", False):
                raise RuntimeError(f"Finalization rejected for {path.name}: {finalization}")
            if not pwm.wait_for_memory_tasks():
                raise RuntimeError(f"Memory tasks did not drain for {path.name}")
            state_tick = pwm.trigger_user_state_time_tick(force=True)
            if state_tick.get("reason") in {
                "worker_queue_full", "worker_shutdown", "task_handler_not_registered",
            }:
                raise RuntimeError(f"State tick rejected for {path.name}: {state_tick}")
            if not pwm.wait_for_memory_tasks():
                raise RuntimeError(f"State tasks did not drain for {path.name}")

            end_time = next_start + timedelta(
                seconds=max(item.end_s for item in source_segments)
            )
            delta = count_delta(before, operation_counts(reporter))
            result = {
                "file_index": file_index,
                "textgrid": str(path),
                "transcript": str(transcript_path),
                "segment_count": len(memory_segments),
                "owner_original_speaker": owner_speaker,
                "speaker_mapping": speaker_mapping,
                "synthetic_session_start": next_start.isoformat(),
                "synthetic_session_end": end_time.isoformat(),
                "input_report": input_report,
                "finalization": finalization,
                "state_tick": state_tick,
                "table_counts": table_counts(pwm),
                "operation_delta": delta,
            }
            results.append(result)
            write_json(output_dir / "file_progress.json", results)
            logging.info(
                "File completed index=%s/%s name=%s facts=%s claims=%s states=%s",
                file_index, len(files), path.name,
                result["table_counts"]["memory_facts"],
                result["table_counts"]["memory_user_claims"],
                result["table_counts"]["user_states"],
            )
            failures = {kind: counts for kind, counts in delta.items() if counts["failed"]}
            if failures:
                raise RuntimeError(f"Tasks failed while processing {path.name}: {failures}")
            next_start = end_time + timedelta(hours=args.inter_file_gap_hours)

        summary = {
            "textgrid_dir": str(textgrid_dir),
            "file_limit": args.file_limit,
            "files_processed": len(results),
            "segments_processed": sum(item["segment_count"] for item in results),
            "synthetic_timeline": True,
            "owner_selection": "chronologically earliest valid nonempty speech interval per file",
            "memory_db_path": str(output_dir / "memory.db"),
            "user_model_db_path": str(output_dir / "user_model.db"),
            "task_db_path": str(output_dir / "task_scheduler.db"),
            "table_counts": table_counts(pwm),
            "operation_counts": operation_counts(reporter),
            "files": results,
        }
        write_json(output_dir / "summary.json", summary)
        logging.info(
            "Continuous extraction finished files=%s segments=%s facts=%s states=%s",
            summary["files_processed"], summary["segments_processed"],
            summary["table_counts"]["memory_facts"],
            summary["table_counts"]["user_states"],
        )
        return 0
    finally:
        try:
            if pwm is not None:
                pwm.close()
        finally:
            memory_logger.removeHandler(memory_log_handler)
            memory_log_handler.close()


if __name__ == "__main__":
    raise SystemExit(main())
