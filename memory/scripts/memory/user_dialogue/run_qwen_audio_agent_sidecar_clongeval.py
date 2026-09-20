#!/usr/bin/env python3
"""Exercise the qwen-audio-agent MemoryRuntime sidecar with CLongEval data.

This is an integration runner, rather than a direct ``MemoryRuntime`` test.
It writes each parsed CLongEval context through the JSONL protocol used by
``AgentMemoryProvider``::

    observe -> flush -> close -> restart -> recall

Closing the first process is intentional: ``MemoryRuntime.close()`` drains its
store/reflect task queues, so recall in the restarted process observes a
durable, deterministic database instead of racing asynchronous extraction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import select
import shutil
import subprocess
import sys
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, TextIO, Tuple


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = (
    REPO_ROOT
    / "test_data"
    / "user_dialogue"
    / "CLongEval"
    / "1-2_long_conversation_memory"
    / "small.jsonl"
)
DEFAULT_OUTPUT_ROOT_DIR = REPO_ROOT / "tmp" / "qwen_audio_agent_sidecar"
DEFAULT_SIDECAR = REPO_ROOT / "integrations" / "qwen_audio_agent" / "agent_memory_sidecar.py"
DATE_HEADER_RE = re.compile(
    r"以下是(?P<date>\d{4}年\d{1,2}月\d{1,2}日)的对话记录\s*[:：]?"
)
ROLE_LINE_RE = re.compile(r"^[“”]?\s*(用户|AI|助手)\s*[:：]\s*(.*)$")
CONTEXT_END_MARKERS = ("请记住以上全部对话记录", "问题：", "问题:")


def load_records(path: Path) -> List[Dict[str, Any]]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"CLongEval input does not exist: {path}")
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                continue
            context = str(value.get("context") or "").strip()
            query = str(value.get("query") or "").strip()
            if not context or not query:
                logging.warning("Skipping record without context/query at line %s", line_number)
                continue
            records.append({
                **value,
                "context": context,
                "query": query,
                "answer": str(value.get("answer") or "").strip(),
                "id": str(value.get("id") or f"line_{line_number}"),
                "_source_line": line_number,
            })
    return records


def filter_records(
    records: Sequence[Dict[str, Any]],
    *,
    question_ids: Optional[Sequence[str]],
    start: int,
    limit: int,
) -> List[Dict[str, Any]]:
    selected = list(records)
    if question_ids:
        wanted = {str(value).strip() for value in question_ids if str(value).strip()}
        selected = [item for item in selected if str(item.get("id")) in wanted]
    if start:
        selected = selected[max(0, int(start)) :]
    if limit:
        selected = selected[: max(0, int(limit))]
    return selected


def _flush_message(messages: List[Tuple[str, str]], role: Optional[str], parts: List[str]) -> None:
    text = "\n".join(part for part in parts if part).strip().strip("“”")
    if text and role:
        messages.append((role, text))


def parse_context_days(context: str) -> List[Dict[str, Any]]:
    """Parse the dated 用户/AI conversation blocks used by CLongEval."""
    headers = list(DATE_HEADER_RE.finditer(context))
    if not headers:
        raise ValueError("Context does not contain a dated conversation header")
    days: List[Dict[str, Any]] = []
    for index, header in enumerate(headers):
        end = headers[index + 1].start() if index + 1 < len(headers) else len(context)
        section = context[header.end() : end]
        for marker in CONTEXT_END_MARKERS:
            marker_index = section.find(marker)
            if marker_index >= 0:
                section = section[:marker_index]
                break
        messages: List[Tuple[str, str]] = []
        role: Optional[str] = None
        parts: List[str] = []
        for raw_line in section.strip().strip("“”").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            match = ROLE_LINE_RE.match(line)
            if match:
                _flush_message(messages, role, parts)
                role = "user" if match.group(1) == "用户" else "assistant"
                parts = [match.group(2).strip()]
            elif role:
                parts.append(line.strip("“”"))
        _flush_message(messages, role, parts)

        pairs: List[Tuple[str, str]] = []
        pending_user: Optional[str] = None
        for message_role, text in messages:
            if message_role == "user":
                pending_user = text if not pending_user else f"{pending_user}\n{text}"
            elif pending_user:
                pairs.append((pending_user, text))
                pending_user = None
        if pairs:
            days.append({
                "date_text": header.group("date"),
                "date": datetime.strptime(header.group("date"), "%Y年%m月%d日"),
                "pairs": pairs,
            })
    if not days:
        raise ValueError("Context did not contain any complete user/assistant pairs")
    return days


def normalize_unique_timestamp(candidate: datetime, seen: set[str]) -> datetime:
    current = candidate
    while current.strftime("%Y-%m-%d %H:%M:%S") in seen:
        current += timedelta(seconds=1)
    seen.add(current.strftime("%Y-%m-%d %H:%M:%S"))
    return current


def context_group_id(context: str, group_index: int) -> str:
    digest = hashlib.sha1(context.encode("utf-8")).hexdigest()[:12]
    return f"context_{group_index:04d}_{digest}"


def group_records_by_context(
    records: Sequence[Dict[str, Any]],
) -> List[Tuple[str, List[Dict[str, Any]]]]:
    grouped: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()
    for record in records:
        grouped.setdefault(str(record["context"]), []).append(record)
    return list(grouped.items())


def gold_answer_in_context(answer: str, context: str) -> bool:
    normalized_answer = re.sub(r"\s+", "", str(answer or "").lower())
    normalized_context = re.sub(r"\s+", "", str(context or "").lower())
    return bool(normalized_answer) and normalized_answer in normalized_context


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run CLongEval conversation contexts through the qwen-audio-agent "
            "agent_memory JSONL sidecar."
        ),
    )
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config.yaml")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--sidecar", type=Path, default=DEFAULT_SIDECAR)
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
        help="Python interpreter with agent_memory dependencies installed.",
    )
    parser.add_argument("--output-root-dir", type=Path, default=DEFAULT_OUTPUT_ROOT_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="0 means all records.")
    parser.add_argument(
        "--question-id",
        action="append",
        help="Only run the specified record id. Can be passed multiple times.",
    )
    parser.add_argument(
        "--owner-prefix",
        default="clongeval",
        help="Prefix used to construct the isolated Qwen ownerId for each context.",
    )
    parser.add_argument(
        "--rpc-timeout-seconds",
        type=float,
        default=600.0,
        help="Maximum time to wait for each JSONL sidecar response.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def resolve_output_dir(args: argparse.Namespace) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = args.output_root_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else output_root / f"{args.input.stem}_{timestamp}"
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}. Pass --overwrite."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = output_dir
    return output_dir


def configure_logging(output_dir: Path, log_level: str) -> Path:
    log_path = output_dir / "run_qwen_audio_agent_sidecar_clongeval.log"
    logging.basicConfig(
        level=getattr(logging, str(log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
        force=True,
    )
    return log_path


class JsonlSidecarClient:
    """Small synchronous client for the provider's stdin/stdout JSONL contract."""

    def __init__(
        self,
        *,
        python_path: Path,
        sidecar_path: Path,
        config_path: Path,
        state_dir: Path,
        stderr_path: Path,
        timeout_seconds: float,
    ) -> None:
        self._timeout_seconds = max(1.0, float(timeout_seconds))
        self._request_id = 0
        self._stderr_handle: TextIO = stderr_path.open("w", encoding="utf-8")
        self._process = subprocess.Popen(
            [
                str(python_path),
                str(sidecar_path),
                "--state-dir",
                str(state_dir),
                "--config",
                str(config_path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr_handle,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        if self._process.stdin is None or self._process.stdout is None:
            self._stderr_handle.close()
            raise RuntimeError("Unable to open JSONL pipes for agent_memory sidecar")

    def request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if self._process.poll() is not None:
            raise RuntimeError(
                f"Sidecar exited before {method}: returncode={self._process.returncode}"
            )
        self._request_id += 1
        request_id = self._request_id
        payload = {"id": request_id, "method": method, "params": params}
        assert self._process.stdin is not None
        self._process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._process.stdin.flush()
        assert self._process.stdout is not None
        ready, _, _ = select.select([self._process.stdout], [], [], self._timeout_seconds)
        if not ready:
            raise TimeoutError(
                f"Timed out after {self._timeout_seconds:.1f}s waiting for sidecar {method}"
            )
        line = self._process.stdout.readline()
        if not line:
            raise RuntimeError(
                f"Sidecar closed stdout during {method}: returncode={self._process.poll()}"
            )
        try:
            response = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid sidecar JSON response to {method}: {line!r}") from exc
        if response.get("id") != request_id:
            raise RuntimeError(
                f"Unexpected sidecar response id for {method}: {response.get('id')!r}"
            )
        if response.get("error"):
            raise RuntimeError(f"Sidecar {method} failed: {response['error']}")
        result = response.get("result")
        if not isinstance(result, dict):
            raise RuntimeError(f"Sidecar {method} returned a non-object result")
        return result

    def close(self) -> Dict[str, Any]:
        try:
            result = self.request("close", {})
            self._process.wait(timeout=self._timeout_seconds)
            return result
        finally:
            self._cleanup_process()

    def abort(self) -> None:
        self._cleanup_process()

    def _cleanup_process(self) -> None:
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=10)
        if not self._stderr_handle.closed:
            self._stderr_handle.close()


def build_session_messages(
    days: Sequence[Dict[str, Any]],
    *,
    session_id: str,
) -> Tuple[List[Dict[str, Any]], int]:
    """Convert parsed benchmark dialogue into Qwen provider session messages."""
    messages: List[Dict[str, Any]] = []
    seen_timestamps: set[str] = set()
    turn_count = 0
    for day_index, day in enumerate(days, 1):
        for pair_index, (user, assistant) in enumerate(day["pairs"], 1):
            turn_count += 1
            timestamp = normalize_unique_timestamp(
                day["date"] + timedelta(seconds=pair_index),
                seen_timestamps,
            ).replace(tzinfo=timezone.utc)
            turn_id = f"{session_id}:turn:{turn_count:04d}"
            # conversation-sync.mjs persists this field as Date.now() epoch
            # milliseconds, and AgentMemoryProvider forwards that numeric value.
            created_at = int(timestamp.timestamp() * 1_000)
            messages.extend(
                [
                    {
                        "id": f"{turn_id}:user",
                        "role": "user",
                        "content": user,
                        "turnId": turn_id,
                        "createdAt": created_at,
                    },
                    {
                        "id": f"{turn_id}:assistant",
                        "role": "assistant",
                        "content": assistant,
                        "turnId": turn_id,
                        "createdAt": created_at,
                    },
                ]
            )
    return messages, turn_count


def run() -> int:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    sidecar_path = args.sidecar.expanduser().resolve()
    python_path = args.python.expanduser().resolve()
    for label, path in (("config", config_path), ("sidecar", sidecar_path), ("python", python_path)):
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")

    records = filter_records(
        load_records(args.input),
        question_ids=args.question_id,
        start=args.start,
        limit=args.limit,
    )
    if not records:
        raise RuntimeError("No CLongEval records selected")
    output_dir = resolve_output_dir(args)
    log_path = configure_logging(output_dir, args.log_level)
    state_dir = output_dir / "sidecar_state"
    groups = group_records_by_context(records)
    logging.info("Loaded records=%s context_groups=%s", len(records), len(groups))

    replay_summaries: List[Dict[str, Any]] = []
    replay_client = JsonlSidecarClient(
        python_path=python_path,
        sidecar_path=sidecar_path,
        config_path=config_path,
        state_dir=state_dir,
        stderr_path=output_dir / "sidecar_replay.stderr.log",
        timeout_seconds=args.rpc_timeout_seconds,
    )
    try:
        health = replay_client.request("health", {})
        logging.info("Replay sidecar health=%s", health)
        for group_index, (context, group_records) in enumerate(groups, 1):
            group_id = context_group_id(context, group_index)
            owner_id = f"{args.owner_prefix}:{group_id}"
            session_id = f"{group_id}:session"
            days = parse_context_days(context)
            messages, turn_count = build_session_messages(days, session_id=session_id)
            observe = replay_client.request(
                "observe",
                {
                    "ownerId": owner_id,
                    "sessionId": session_id,
                    "messages": messages,
                },
            )
            flush = replay_client.request("flush", {"ownerId": owner_id})
            replay_summaries.append(
                {
                    "context_group_id": group_id,
                    "owner_id": owner_id,
                    "session_id": session_id,
                    "record_count": len(group_records),
                    "parsed_days": len(days),
                    "turn_pairs": turn_count,
                    "observe": observe,
                    "flush": flush,
                }
            )
            logging.info(
                "Replayed group=%s turns=%s observed=%s flushed=%s reflect_queued=%s",
                group_id,
                turn_count,
                observe.get("messages", 0),
                flush.get("flushed"),
                flush.get("reflectQueued"),
            )
        close_result = replay_client.close()
        logging.info("Replay sidecar closed=%s", close_result)
    except Exception:
        replay_client.abort()
        raise

    results: List[Dict[str, Any]] = []
    recall_client = JsonlSidecarClient(
        python_path=python_path,
        sidecar_path=sidecar_path,
        config_path=config_path,
        state_dir=state_dir,
        stderr_path=output_dir / "sidecar_recall.stderr.log",
        timeout_seconds=args.rpc_timeout_seconds,
    )
    try:
        logging.info("Recall sidecar health=%s", recall_client.request("health", {}))
        for group_index, (context, group_records) in enumerate(groups, 1):
            group_id = context_group_id(context, group_index)
            owner_id = f"{args.owner_prefix}:{group_id}"
            for record in group_records:
                report = recall_client.request(
                    "recall",
                    {"ownerId": owner_id, "query": str(record["query"])},
                )
                memory_context = str(report.get("memory_context") or "")
                result = {
                    "id": record["id"],
                    "source_line": record.get("_source_line"),
                    "context_group_id": group_id,
                    "owner_id": owner_id,
                    "query": record["query"],
                    "answer": record.get("answer", ""),
                    "answer_in_recall_context": gold_answer_in_context(
                        record.get("answer", ""), memory_context
                    ),
                    "recall_status": str(
                        report.get("status") or ("ok" if memory_context else "empty")
                    ),
                    "actual_recall_mode": str(
                        report.get("actual_recall_mode") or "unknown"
                    ),
                    "recall_context_chars": len(memory_context),
                    "recall_context": memory_context,
                    "recall_report": report,
                }
                results.append(result)
                logging.info(
                    "Recalled id=%s group=%s chars=%s answer_in_context=%s",
                    record["id"],
                    group_id,
                    result["recall_context_chars"],
                    result["answer_in_recall_context"],
                )
        recall_close_result = recall_client.close()
        logging.info("Recall sidecar closed=%s", recall_close_result)
    except Exception:
        recall_client.abort()
        raise

    results_path = output_dir / "qwen_audio_agent_sidecar_clongeval_results.json"
    results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "input": str(args.input.expanduser().resolve()),
        "config": str(config_path),
        "sidecar": str(sidecar_path),
        "python": str(python_path),
        "output_dir": str(output_dir),
        "state_dir": str(state_dir),
        "log_path": str(log_path),
        "results_path": str(results_path),
        "records_processed": len(results),
        "context_groups": len(groups),
        "recall_nonempty": sum(bool(item["recall_context"]) for item in results),
        "gold_answer_in_recall_context": sum(
            bool(item["answer_in_recall_context"]) for item in results
        ),
        "gold_coverage_rate": (
            sum(bool(item["answer_in_recall_context"]) for item in results) / len(results)
            if results
            else 0.0
        ),
        "replay": replay_summaries,
    }
    summary_path = output_dir / "qwen_audio_agent_sidecar_clongeval_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info(
        "Finished records=%s groups=%s recall_nonempty=%s gold_coverage=%s/%s",
        summary["records_processed"],
        summary["context_groups"],
        summary["recall_nonempty"],
        summary["gold_answer_in_recall_context"],
        summary["records_processed"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
