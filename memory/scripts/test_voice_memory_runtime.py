#!/usr/bin/env python3
"""Exercise the VoiceRuntime -> MemoryRuntime ingestion path."""

from __future__ import annotations

import argparse
import json
import sys
import time
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from memory_mcp_server import (  # noqa: E402
    build_service,
    load_config,
    resolve_mcp_server_paths,
)


DEFAULT_AUDIO_DIR = (
    ROOT
    / "test_data/ambient_transcript/Eval_Ali/Eval_Ali_far/filenames_formated_dir"
)
SUPPORTED_AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test VoiceRuntime transcription and MemoryRuntime storage together."
    )
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=None,
        help="Override memory_mcp_server.result_dir; database, logs, and report use its configured file names.",
    )
    parser.add_argument(
        "--override",
        action="store_true",
        help="Delete existing database, log, and report files before testing.",
    )
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--queue-timeout", type=float, default=60.0)
    parser.add_argument("--job-poll-interval", type=float, default=5.0)
    parser.add_argument("--job-timeout", type=float)
    parser.add_argument(
        "--max-duration-s",
        type=float,
        help="Override voice_runtime.max_duration_s for this test run.",
    )
    parser.add_argument("--source-type", default="allday_recording")
    parser.add_argument("--session-start")
    parser.add_argument("--tag", dest="tags", action="append", default=[])
    parser.add_argument("--max-files", type=int)
    parser.add_argument(
        "--query",
        help="Optional query to run after audio ingestion and queue flushing.",
    )
    return parser.parse_args()


def collect_audio_files(audio_dir: Path, max_files: int | None) -> List[Path]:
    directory = audio_dir.expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Audio directory not found: {directory}")
    files = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_SUFFIXES
    )
    if max_files is not None:
        if max_files <= 0:
            raise ValueError("--max-files must be positive")
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"No supported audio files found in {directory}")
    return files


def build_service_args(
    args: argparse.Namespace,
    *,
    db_path: Path,
    log_path: Path | None,
) -> Namespace:
    """Build the small Namespace expected by memory_mcp_server.build_service."""
    return Namespace(
        config=args.config.expanduser().resolve(),
        db_path=db_path,
        log_path=log_path,
        log_level=args.log_level,
        queue_timeout=args.queue_timeout,
        max_duration_s=args.max_duration_s,
    )


def wait_for_audio_job(
    service: Any,
    job_id: str,
    *,
    poll_interval: float,
    timeout: float | None,
    logger: Any,
) -> Dict[str, Any]:
    """Poll an asynchronous MCP audio job until it reaches a terminal state."""
    started_at = time.monotonic()
    terminal_statuses = {"ok", "partial_failure", "failed", "queue_flush_timeout"}
    while True:
        report = service.get_audio_processing_job(job_id=job_id)
        status = str(report.get("status") or "")
        logger.info(
            "Audio processing job status job_id=%s status=%s stage=%s file=%s/%s elapsed_ms=%s",
            job_id,
            status,
            report.get("current_stage"),
            report.get("current_file_index"),
            report.get("file_count"),
            report.get("elapsed_ms"),
        )
        if status in terminal_statuses:
            return report
        if timeout is not None and time.monotonic() - started_at > timeout:
            raise TimeoutError(f"Audio processing job timed out: {job_id}")
        time.sleep(max(0.1, float(poll_interval)))


def _remove_existing_file(path: Path) -> None:
    """Remove one output file while refusing to delete a directory."""
    if not path.exists() and not path.is_symlink():
        return
    if path.is_dir():
        raise IsADirectoryError(f"Refusing to delete directory: {path}")
    path.unlink()


def remove_existing_outputs(
    db_path: Path,
    log_path: Path | None,
    report_path: Path,
) -> None:
    """Remove files that could mix results from an earlier test run."""
    paths = [
        db_path,
        Path(f"{db_path}-wal"),
        Path(f"{db_path}-shm"),
        log_path,
        report_path,
    ]
    seen: set[Path] = set()
    for path in paths:
        if path is None:
            continue
        resolved_path = path.expanduser().resolve()
        if resolved_path in seen:
            continue
        seen.add(resolved_path)
        _remove_existing_file(resolved_path)


def main() -> int:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)
    server_config = config.get("memory_mcp_server") or {}
    if not isinstance(server_config, dict):
        raise ValueError("memory_mcp_server in config.yaml must be a mapping")
    result_dir = (
        args.result_dir.expanduser().resolve()
        if args.result_dir
        else None
    )
    server_paths = resolve_mcp_server_paths(
        server_config,
        config_path,
        result_dir=result_dir,
    )
    result_dir = server_paths["result_dir"]
    db_path = server_paths["db_path"]
    log_path = server_paths["log_path"]
    report_path = server_paths["report_path"]
    asr_result_dir = server_paths["asr_result_dir"]
    if db_path is None:
        raise ValueError("memory_mcp_server.db_name must be configured")
    if report_path is None:
        raise ValueError("memory_mcp_server.report_name must be configured")
    if args.override:
        remove_existing_outputs(db_path, log_path, report_path)
    audio_files = collect_audio_files(args.audio_dir, args.max_files)
    service, logger = build_service(
        build_service_args(args, db_path=db_path, log_path=log_path)
    )
    # ``build_service`` intentionally remains config-driven. This test-only
    # override keeps every generated artifact under ``--result-dir``.
    service.asr_result_dir = asr_result_dir
    try:
        logger.info(
            "Voice/memory runtime test started audio_dir=%s file_count=%s db_path=%s",
            args.audio_dir.expanduser().resolve(),
            len(audio_files),
            db_path,
        )
        processing_report = service.process_audio_files(
            files=[{"audio_path": str(path)} for path in audio_files],
            source_type=args.source_type,
            session_start=args.session_start,
            tags=list(args.tags),
        )
        job_id = str(processing_report.get("job_id") or "")
        if job_id:
            processing_report = wait_for_audio_job(
                service,
                job_id,
                poll_interval=args.job_poll_interval,
                timeout=args.job_timeout,
                logger=logger,
            )
        result: Dict[str, Any] = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "config_path": str(config_path),
            "audio_dir": str(args.audio_dir.expanduser().resolve()),
            "audio_files": [str(path) for path in audio_files],
            "result_dir": str(result_dir),
            "db_path": str(db_path),
            "asr_result_dir": str(asr_result_dir) if asr_result_dir else None,
            "override": bool(args.override),
            "max_duration_s": args.max_duration_s,
        }
        if args.query:
            result["recall"] = service.trigger_memory_recall(query=args.query)

        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Voice/memory runtime test finished report_path=%s", report_path)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    finally:
        service.close()


if __name__ == "__main__":
    raise SystemExit(main())
