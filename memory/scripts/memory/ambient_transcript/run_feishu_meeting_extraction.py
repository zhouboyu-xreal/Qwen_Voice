#!/usr/bin/env python3
"""Extract PWM memory/state from Feishu-exported Markdown transcripts in order.

Example (run from the repository root)::

    python scripts/memory/ambient_transcript/run_feishu_meeting_extraction.py \
        --meeting-file '/path/to/文字记录：会议名称 2025年12月24日.md' \
        --owner-name 沈劲阳 --parse-only

Pass multiple paths after --meeting-file, or repeat it, to process meetings in
the supplied order with one PWM instance and shared databases. Omit --parse-only
to run the configured extraction pipeline, without recall.
Only exact speaker-name matches identify the owner; anonymous room speakers
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
from typing import Any, Dict, List, Optional, Sequence

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
SHANGHAI = timezone(timedelta(hours=8))
OWNER_SPEAKER = "speaker_用户本人"
HEADER_RE = re.compile(r"^(?P<speaker>.+?)\s+(?P<hours>\d{1,3}):(?P<minutes>\d{2}):(?P<seconds>\d{2})$")
DATE_RE = re.compile(r"(?P<year>\d{4})[年/-](?P<month>\d{1,2})[月/-](?P<day>\d{1,2})日?")
TIME_RANGE_RE = re.compile(r"(?P<start>\d{1,2}:\d{2})\s*[-–—~～至]\s*(?P<end>\d{1,2}:\d{2})")
TIMEZONE_RE = re.compile(r"GMT\s*(?P<sign>[+-])(?P<hours>\d{1,2})(?::(?P<minutes>\d{2}))?")
MARKDOWN_ESCAPE_RE = re.compile(r"\\([!\"#$%&'()*+,\-./:;<=>?@\[\]^_`{|}~\\])")
MEMORY_TABLES = (
    "memory_source_segments", "memory_facts", "memory_user_activity_items",
    "memory_user_claims", "memory_user_observed_patterns", "memory_user_intent_items",
    "memory_user_threads", "memory_fact_user_claim_signals",
    "memory_fact_user_intent_signals", "memory_fact_user_action_signals",
)


def plain_markdown(text: str) -> str:
    """Undo Markdown punctuation escapes without interpreting document commands."""
    return MARKDOWN_ESCAPE_RE.sub(r"\1", text).strip()


def speaker_name(text: str) -> str:
    text = plain_markdown(text).strip()
    return text[1:].strip() if text.startswith("@") else text


def resolve_meeting_start(
    *, title: str, time_description: str, filename: str, override: str,
) -> datetime:
    if override:
        value = datetime.fromisoformat(override.replace("Z", "+00:00"))
        return value if value.tzinfo is not None else value.replace(tzinfo=SHANGHAI)
    match = DATE_RE.search(time_description) or DATE_RE.search(title) or DATE_RE.search(filename)
    time_range = TIME_RANGE_RE.search(time_description)
    if match is None or time_range is None:
        raise ValueError("Cannot determine meeting date/start time; supply --session-start ISO_TIMESTAMP")
    day = date(**{key: int(value) for key, value in match.groupdict().items()})
    tz = SHANGHAI
    zone_match = TIMEZONE_RE.search(time_description)
    if zone_match:
        hours = int(zone_match["hours"])
        minutes = int(zone_match["minutes"] or 0)
        if minutes >= 60:
            raise ValueError("Invalid GMT offset in meeting metadata")
        offset = timedelta(hours=hours, minutes=minutes)
        tz = timezone(offset if zone_match["sign"] == "+" else -offset)
    hour, minute = map(int, time_range["start"].split(":"))
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=tz)


def parse_meeting(path: Path, *, owner_name: str, session_start: str = "") -> Dict[str, Any]:
    """Read Feishu speaker/HH:MM:SS blocks, keeping text and source line numbers."""
    owner = speaker_name(owner_name)
    if not owner:
        raise ValueError("--owner-name must not be empty")
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    title = ""
    time_description = ""
    blocks: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    body: List[str] = []

    def finish_block() -> None:
        if current is None:
            return
        text = plain_markdown("\n".join(body))
        if not text:
            raise ValueError(f"Empty speech block at line {current['source_line']} in {path}")
        blocks.append({**current, "text": text})

    for line_number, raw_line in enumerate(lines, 1):
        line = plain_markdown(raw_line)
        header = HEADER_RE.fullmatch(line)
        if header:
            finish_block()
            hour, minute, second = (int(header[key]) for key in ("hours", "minutes", "seconds"))
            if minute >= 60 or second >= 60:
                raise ValueError(f"Invalid speech offset at line {line_number}: {line}")
            current = {
                "original_speaker": speaker_name(header["speaker"]),
                "offset_seconds": hour * 3600 + minute * 60 + second,
                "source_line": line_number,
            }
            body = []
        elif current is not None:
            body.append(raw_line)
        elif line.startswith("# ") and not title:
            title = line[2:].strip()
        elif line.lstrip("> ").startswith("会议主题："):
            title = line.lstrip("> ").split("：", 1)[1].strip() or title
        elif line.lstrip("> ").startswith("会议时间："):
            time_description = line.lstrip("> ").split("：", 1)[1].strip()
    finish_block()
    if not blocks:
        raise ValueError(f"No speaker/timestamp speech blocks found in {path}")
    # The topic may omit the date while the H1 includes it.
    heading = next((plain_markdown(line) for line in lines if line.startswith("# ")), "")
    start = resolve_meeting_start(
        title=heading or title, time_description=time_description,
        filename=path.name, override=session_start,
    )
    source_speakers = list(dict.fromkeys(item["original_speaker"] for item in blocks))
    if owner not in source_speakers:
        raise ValueError(f"Owner {owner!r} not found among speakers: {source_speakers}")
    # Never overwrite a non-owner export label that already uses the reserved ID.
    if OWNER_SPEAKER in source_speakers and owner != OWNER_SPEAKER:
        raise ValueError("A non-owner speaker already uses the reserved owner label")
    mapping = {name: OWNER_SPEAKER if name == owner else name for name in source_speakers}
    reordered = any(
        left["offset_seconds"] > right["offset_seconds"]
        for left, right in zip(blocks, blocks[1:])
    )
    blocks.sort(key=lambda item: item["offset_seconds"])  # Stable for simultaneous speakers.
    segments = []
    for index, item in enumerate(blocks, 1):
        timestamp = (start + timedelta(seconds=item["offset_seconds"])).isoformat()
        segments.append({
            "speaker": mapping[item["original_speaker"]],
            "role": "ambient_speaker", "text": item["text"],
            "started_at": timestamp, "ended_at": timestamp,
            "segment_index": index,
            "metadata": {
                "source": "feishu_markdown", "source_file": str(path),
                "source_line": item["source_line"],
                "original_speaker": item["original_speaker"],
                "offset_seconds": item["offset_seconds"],
                "audio_start_s": item["offset_seconds"],
                "audio_end_s": item["offset_seconds"],
                "end_time_basis": "unknown_use_start_time",
            },
        })
    return {
        "source_file": str(path), "meeting_title": title,
        "meeting_time_description": time_description,
        "session_start": start.isoformat(), "session_start_overridden": bool(session_start),
        "last_utterance_start": segments[-1]["started_at"],
        "timestamp_policy": "exported relative starts; ends unknown and set equal to starts",
        "owner_name": owner, "owner_speaker": OWNER_SPEAKER,
        "speaker_mapping": mapping, "source_segment_count": len(segments),
        "reordered_by_timestamp": reordered, "segments": segments,
    }


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
    parser.add_argument("--meeting-file", type=Path, nargs="+", action="extend", required=True,
                        help="One or more Feishu Markdown files in processing order; may be repeated")
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
    if args.segment_limit < 0:
        parser.error("--segment-limit must be nonnegative")
    if args.session_start and len(args.meeting_file) != 1:
        parser.error("--session-start only supports one meeting file; multiple files use their own meeting times")
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
    # Evaluate claims below the normal state-window threshold at this file's
    # data time, not the date this test runs or the next meeting's timestamp.
    report["state_tick"] = pwm.trigger_user_state_time_tick(
        now=datetime.fromisoformat(meeting["last_utterance_start"]), force=True,
    )
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
    # Parse every source before starting models so an invalid later file cannot
    # leave a partially executed extraction run. Do not sort the supplied paths.
    for path in args.meeting_file:
        source = path.expanduser().resolve()
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
