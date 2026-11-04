#!/usr/bin/env python3
"""Extract PWM memory/state from Feishu-exported Markdown transcripts in order.

Example (run from the repository root)::

    python scripts/memory/ambient_transcript/run_feishu_meeting_extraction.py \
        --meeting-file '/path/to/文字记录：会议名称 2025年12月24日.md' \
        --owner-name 沈劲阳 --parse-only

Use --meeting-dir DIR --file-limit N to select the earliest N Markdown meetings
in that directory (non-recursive; 0 means all). Pass filenames after
--meeting-file together with --meeting-dir to select specific files, or pass
file paths without a directory. Multiple paths and repeated --meeting-file
are supported. All selected meetings are ordered by their actual meeting start
time, not filename or modification time, and share one PWM and databases.
Omit --parse-only to run extraction, without recall. Exact owner names may
include Feishu's " 在 <meeting room>" suffix; anonymous room/device speakers
are never inferred to be that person. Exported offsets are utterance STARTS,
not measured speech durations: inputs use point timestamps, with this
limitation recorded explicitly in the parsed transcript and report.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from config import split_memory_config
from pwm_manager import PersonWorldModel
from task_scheduler import TaskOperationReporter

try:  # Support both direct CLI execution and package imports in tests.
    from .run_transcript_Ali_eval import (
        configure_logging, configure_memory_logger, write_transcript_txt,
    )
except ImportError:
    from run_transcript_Ali_eval import (
        configure_logging, configure_memory_logger, write_transcript_txt,
    )


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "tmp/ambient_transcript/feishu_meeting"
from memory.feishu_transcript import (
    SHANGHAI, OWNER_SPEAKER, HEADER_RE, DATE_RE, TIME_RANGE_RE, TIMEZONE_RE,
    MARKDOWN_ESCAPE_RE, plain_markdown, speaker_name, resolve_meeting_start,
    read_meeting_metadata, parse_meeting,
)
MEMORY_TABLES = (
    "memory_source_segments", "memory_facts", "memory_user_activity_items",
    "memory_user_claims", "memory_user_observed_patterns", "memory_user_intent_items",
    "memory_user_threads", "memory_fact_user_claim_signals",
    "memory_fact_user_intent_signals", "memory_fact_user_action_signals",
)


def select_meeting_files(args: argparse.Namespace) -> List[Path]:
    """Resolve explicit files or sort directory exports before applying the limit."""
    directory = args.meeting_dir.expanduser().resolve() if args.meeting_dir else None
    if directory is not None and not directory.is_dir():
        raise NotADirectoryError(f"Meeting directory does not exist or is not a directory: {directory}")
    if args.meeting_file:
        candidates = []
        for path in args.meeting_file:
            source = path.expanduser()
            if directory is not None and not source.is_absolute():
                source = directory / source
            candidates.append(source.resolve())
    else:
        candidates = [
            path.resolve() for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() == ".md"
        ]
    if not candidates:
        raise ValueError(f"No Markdown meeting files found in {directory}")
    dated_files = []
    for source in dict.fromkeys(candidates):
        if not source.is_file():
            raise FileNotFoundError(f"Meeting file does not exist or is not a file: {source}")
        with source.open(encoding="utf-8-sig") as stream:
            metadata = read_meeting_metadata(stream)
        try:
            start = resolve_meeting_start(
                title=metadata["heading"] or metadata["title"],
                time_description=metadata["time_description"], filename=source.name,
                override=args.session_start,
            )
        except ValueError as exc:
            raise ValueError(f"Invalid meeting time in {source}: {exc}") from exc
        dated_files.append((start.timestamp(), str(source), source))
    dated_files.sort(key=lambda item: (item[0], item[1]))
    paths = [item[2] for item in dated_files]
    return paths[:args.file_limit] if args.file_limit else paths


def register_test_owner(pwm: PersonWorldModel, owner_name: str) -> None:
    """Register operator-supplied identity, not a fabricated fact or LLM guess.

    The production alias upsert requires a fact ID. A test fixture has none, so
    use the nullable evidence_fact_id and explicitly record test configuration
    as the evidence source. Call before submitting any inputs to the FIFO.
    """
    db = pwm.database
    canonical = pwm.memory_manager.world_owner_entity_name
    with db.transaction():
        owner_id = db.upsert_entity_nodes([{"name": canonical, "type": "PERSON"}])[canonical]
        if owner_name == canonical:
            return
        timestamp = datetime.now().astimezone().isoformat()
        db._conn.execute(
            """INSERT INTO memory_owner_entity_aliases (
                owner_entity_id, alias_name, alias_key, status, confidence,
                evidence_fact_id, evidence_type, evidence_text, source_speaker,
                created_at, updated_at
            ) VALUES (?, ?, ?, 'confirmed', 1.0, NULL, 'test_configuration', ?, ?, ?, ?)""",
            (owner_id, owner_name, db.normalize_owner_alias_key(owner_name),
             f"Test operator specified --owner-name={owner_name}", OWNER_SPEAKER,
             timestamp, timestamp),
        )


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--meeting-dir", type=Path,
                        help="Directory containing Feishu Markdown exports; non-recursive")
    parser.add_argument("--meeting-file", type=Path, nargs="+", action="extend", default=[],
                        help="Specific files, optionally relative to --meeting-dir; may be repeated; ordered by meeting time")
    parser.add_argument("--file-limit", type=int, default=0,
                        help="First N files after chronological sorting; 0 means all")
    parser.add_argument("--owner-name", required=True, help="Exact owner name in the export, e.g. 沈劲阳")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config.yaml")
    parser.add_argument("--output-dir", type=Path, help="New or empty directory; existing data is never reused")
    parser.add_argument("--session-start", default="", help="Single-file time override; naive ISO time means GMT+08")
    parser.add_argument("--segment-limit", type=int, default=0, help="First N chronological utterances per file; 0 means all")
    parser.add_argument("--parse-only", action="store_true", help="Export normalized inputs only; no models or databases")
    parser.add_argument("--llm-model")
    parser.add_argument("--llm-base-url")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    args = parser.parse_args(argv)
    if not args.meeting_file and args.meeting_dir is None:
        parser.error("supply --meeting-dir or --meeting-file")
    if args.file_limit < 0:
        parser.error("--file-limit must be nonnegative")
    if args.segment_limit < 0:
        parser.error("--segment-limit must be nonnegative")
    if args.session_start and len(args.meeting_file) != 1:
        parser.error("--session-start requires one explicit --meeting-file; other files use their own meeting times")
    return args


def meeting_output_dir(output_dir: Path, index: int, count: int) -> Path:
    return output_dir if count == 1 else output_dir / f"meeting_{index:04d}"


def extract_meeting(
    pwm: PersonWorldModel, reporter: TaskOperationReporter,
    meeting: Dict[str, Any], report: Dict[str, Any],
) -> None:
    before = reporter.snapshot()
    tags = ["feishu_meeting", f"meeting:{meeting['meeting_title']}"]
    # Keep all participants' utterances for pronoun/acceptance/dependency context.
    # Runtime assembles the whole file; drain it before submitting the next one.
    report["input_report"] = pwm.accept_memory_input(
        transcript_segments=meeting["segments"], tags=tags,
        ambient_recording_enabled=True,
    )
    input_report = report["input_report"]
    if input_report.get("accepted") is False or input_report.get("reason") in {
        "memory_disabled", "empty_transcript_batch", "queue_rejected",
        "worker_queue_full", "worker_shutdown", "task_handler_not_registered", "task_disabled",
    }:
        raise RuntimeError(f"Meeting input rejected: {input_report}")
    report["finalization"] = pwm.finalize_memory_session(reason="feishu_meeting_complete", tags=tags)
    if not report["finalization"].get("accepted", False):
        raise RuntimeError(f"Meeting finalization rejected: {report['finalization']}")
    if not pwm.wait_for_memory_tasks():
        raise RuntimeError("Memory extraction tasks did not drain")
    # Drain every pending batch using the timestamps persisted with its claims.
    report["state_tick"] = pwm.drain_pending_user_state_claims()
    if not pwm.wait_for_memory_tasks():
        raise RuntimeError("User state tasks did not drain")
    report["table_counts"] = {
        table: int(pwm.database._conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
        for table in MEMORY_TABLES
    }
    report["table_counts"]["user_states"] = int(
        pwm.user_model_database._conn.execute("SELECT COUNT(*) FROM user_states").fetchone()[0]
    )
    snapshot = reporter.snapshot()
    operation_delta = {}
    for kind, counts in snapshot["counts"].items():
        previous = before["counts"].get(kind, {})
        delta = {key: round(value - previous.get(key, 0), 2) for key, value in counts.items()}
        if any(delta.values()):
            operation_delta[kind] = delta
    previous_task_ids = {item["task_id"] for item in before["recent_tasks"]}
    current_events = [
        item for item in snapshot["recent_tasks"]
        if item["task_id"] not in previous_task_ids
    ]
    report["operation_delta"] = operation_delta
    report["failed_tasks"] = [
        item for item in current_events
        if item.get("event") == "finished" and item.get("succeeded") is False
    ]
    report["partial_tasks"] = [
        item for item in current_events
        if item.get("event") == "finished" and item.get("status") == "partial"
    ]
    failures = {kind: counts for kind, counts in operation_delta.items() if counts.get("failed")}
    # Disabled/empty follow-ups are normal, unlike a lost task or queue failure.
    unexpected_rejections = [
        item for item in current_events
        if item.get("event") == "rejected" and item.get("reason") not in {
            "task_disabled", "no_pending_signals", "no_pending_activities",
            "signals_already_submitted",
        }
    ]
    if unexpected_rejections:
        report["unexpected_rejections"] = unexpected_rejections
        raise RuntimeError(f"Task submissions rejected: {unexpected_rejections}")
    report["status"] = "completed_with_failures" if failures or report["partial_tasks"] else "completed"
    if failures or report["partial_tasks"]:
        logging.warning(
            "Meeting extraction completed with task failures source=%s failures=%s partial_task_ids=%s; "
            "continuing with the next file",
            meeting["source_file"], failures,
            [item["task_id"] for item in report["partial_tasks"]],
        )


def run_extraction(args: argparse.Namespace, meetings: Sequence[Dict[str, Any]], output_dir: Path) -> None:
    sections = split_memory_config(args.config)
    sections.memory_manager["user_memory_database_path"] = str(output_dir / "memory.db")
    sections.memory_manager["world_owner_entity_name"] = "用户本人"
    sections.user_state["user_model_database_path"] = str(output_dir / "user_model.db")
    for key, value in (("llm_name", args.llm_model), ("llm_base_url", args.llm_base_url)):
        if value is not None:
            sections.llm_model[key] = value
    logger, handler = configure_memory_logger(output_dir / "memory_manager.log", args.log_level)
    reporter = TaskOperationReporter(recent_task_limit=10000)
    pwm = None
    meeting_reports = [
        {**{key: value for key, value in meeting.items() if key != "segments"},
         "meeting_index": index, "status": "pending"}
        for index, meeting in enumerate(meetings, 1)
    ]
    report: Dict[str, Any] = (
        {key: value for key, value in meetings[0].items() if key != "segments"}
        if len(meetings) == 1 else {}
    )
    report.update({
        "status": "running", "config": str(args.config.expanduser().resolve()),
        "meeting_count": len(meetings), "meetings": meeting_reports,
        "memory_db_path": str(output_dir / "memory.db"),
        "user_model_db_path": str(output_dir / "user_model.db"),
        "task_db_path": str(output_dir / "task_scheduler.db"),
    })
    try:
        pwm = PersonWorldModel(
            memory_runtime_config=sections.memory_runtime,
            memory_manager_config=sections.memory_manager,
            user_state_config=sections.user_state,
            llm_model_config=sections.llm_model,
            embedding_model_config=sections.embedding_model,
            task_scheduling_config=sections.task_scheduling,
            operation_reporter=reporter, logger=logger,
        )
        register_test_owner(pwm, meetings[0]["owner_name"])
        for index, (meeting, meeting_report) in enumerate(zip(meetings, meeting_reports), 1):
            directory = meeting_output_dir(output_dir, index, len(meetings))
            directory.mkdir(parents=True, exist_ok=True)
            meeting_report["status"] = "running"
            logging.info("Meeting extraction start index=%s/%s source=%s", index, len(meetings), meeting["source_file"])
            try:
                extract_meeting(pwm, reporter, meeting, meeting_report)
            except Exception as exc:
                meeting_report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                raise
            finally:
                if len(meetings) > 1:
                    write_json(directory / "report.json", meeting_report)
            report["table_counts"] = meeting_report["table_counts"]
            logging.info(
                "Meeting extraction complete index=%s/%s status=%s counts=%s",
                index, len(meetings), meeting_report["status"], meeting_report["table_counts"],
            )
        report["status"] = (
            "completed_with_failures"
            if any(item["status"] == "completed_with_failures" for item in meeting_reports)
            else "completed"
        )
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        logging.exception("Meeting extraction failed")
        raise
    finally:
        try:
            if pwm is not None:
                pwm.close()
        finally:
            if len(meetings) == 1:
                for key in (
                    "input_report", "finalization", "state_tick", "table_counts",
                    "operation_delta", "failed_tasks", "partial_tasks", "unexpected_rejections",
                ):
                    if key in meeting_reports[0]:
                        report[key] = meeting_reports[0][key]
            report["operations"] = reporter.snapshot()
            write_json(output_dir / "report.json", report)
            logger.removeHandler(handler)
            handler.close()


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    meetings = []
    # Sort by export timestamps and apply the file limit before parsing speech.
    # Validate all selected sources before starting models or creating output.
    for source in select_meeting_files(args):
        meeting = parse_meeting(source, owner_name=args.owner_name, session_start=args.session_start)
        if args.segment_limit:
            meeting["segments"] = meeting["segments"][:args.segment_limit]
        meeting["selected_segment_count"] = len(meeting["segments"])
        meeting["selected_owner_segment_count"] = sum(
            item["speaker"] == OWNER_SPEAKER for item in meeting["segments"]
        )
        meeting["segment_limit"] = args.segment_limit
        meeting["last_utterance_start"] = meeting["segments"][-1]["started_at"]
        meetings.append(meeting)
    output_dir = (
        args.output_dir.expanduser().resolve() if args.output_dir else
        DEFAULT_OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f"Output directory must be new or empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(output_dir / "run.log", args.log_level)
    for index, meeting in enumerate(meetings, 1):
        directory = meeting_output_dir(output_dir, index, len(meetings))
        directory.mkdir(parents=True, exist_ok=True)
        write_json(directory / "parsed_meeting.json", meeting)
        write_transcript_txt(directory / "transcript.txt", meeting["segments"])
        logging.info(
            "Meeting parsed index=%s/%s source=%s owner=%s start=%s speakers=%s segments=%s/%s owner_segments=%s output=%s",
            index, len(meetings), meeting["source_file"], meeting["owner_name"], meeting["session_start"],
            len(meeting["speaker_mapping"]), meeting["selected_segment_count"], meeting["source_segment_count"],
            meeting["selected_owner_segment_count"], directory,
        )
    if len(meetings) > 1:
        write_json(output_dir / "parsed_meetings.json", [
            {**{key: value for key, value in meeting.items() if key != "segments"},
             "meeting_index": index, "output_dir": str(meeting_output_dir(output_dir, index, len(meetings)))}
            for index, meeting in enumerate(meetings, 1)
        ])
    if not args.parse_only:
        run_extraction(args, meetings, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
