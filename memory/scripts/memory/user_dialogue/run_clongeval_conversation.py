#!/usr/bin/env python3
"""Run MemoryNodeManager on the CLongEval conversation benchmark.

CLongEval conversation records are JSONL objects with this shape::

    {
        "context": "...dated Chinese conversations...",
        "query": "...question...",
        "answer": "...gold answer...",
        "id": "..."
    }

The same context is often reused by several questions. This runner groups
records by exact context, replays each context into one isolated memory DB,
and then runs all of that context's questions against the same memory runtime.
It uses the project config.yaml for the shared runtime and manager settings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import shutil
import sys
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
SCRIPT_ROOT = REPO_ROOT / "scripts" / "user_dialogue"
for import_root in (SRC_ROOT, REPO_ROOT, SCRIPT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from memory.memory_manager import MemoryOperationReporter
from memory.memory_runtime import MemoryRuntime
from memory.config import split_memory_config
from scripts.memory.user_dialogue.test_memory_store_fact_extraction import (
    configure_logging,
    load_project_config,
    log_memory_index_state,
    resolve_llm_args,
    validate_embedding_runtime,
)


DEFAULT_INPUT = (
    REPO_ROOT
    / "test_data"
    / "user_dialogue"
    / "CLongEval"
    / "1-2_long_conversation_memory"
    / "small.jsonl"
)
DEFAULT_OUTPUT_ROOT_DIR = REPO_ROOT / "tmp" / "clongeval_conversation"

DATE_HEADER_RE = re.compile(
    r"以下是(?P<date>\d{4}年\d{1,2}月\d{1,2}日)的对话记录\s*[:：]?"
)
ROLE_LINE_RE = re.compile(r"^[“”]?\s*(用户|AI|助手)\s*[:：]\s*(.*)$")
CONTEXT_END_MARKERS = ("请记住以上全部对话记录", "问题：", "问题:")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MemoryNodeManager on CLongEval conversation JSONL data."
    )
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config.yaml")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root-dir", type=Path, default=DEFAULT_OUTPUT_ROOT_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--existing-state-dir",
        type=Path,
        help=(
            "Reuse existing per-context DBs from "
            "<dir>/<context_group_id>/memory.db. When set, skip replay, "
            "store, and reflect; only rerun recall and reader answering."
        ),
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="0 means all records.")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes used to process independent context groups.",
    )
    parser.add_argument(
        "--question-id",
        action="append",
        help="Only run the specified record id. Can be passed multiple times.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--llm-model")
    parser.add_argument("--llm-base-url")
    parser.add_argument("--llm-api-key")
    parser.add_argument("--llm-timeout", type=int)
    parser.add_argument(
        "--llm-max-tokens",
        type=int,
        default=8192,
        help="Maximum output tokens for each memory extraction call.",
    )
    parser.add_argument(
        "--llm-thinking",
        choices=("disabled", "enabled", "auto"),
        default=None,
        help="Defaults to memory_manager.llm.llm_thinking in config.yaml.",
    )
    parser.add_argument(
        "--no-llm-json-mode",
        action="store_false",
        dest="llm_json_mode",
        help="Do not request provider-enforced JSON output.",
    )
    parser.set_defaults(llm_json_mode=None)
    parser.add_argument("--max-pending-interaction-turns", type=int)
    parser.add_argument("--max-pending-interaction-tokens", type=int)
    parser.add_argument("--enable-reflect", action="store_true")
    parser.add_argument(
        "--reflect-every-days",
        type=int,
        default=1,
        help="Run reflect after every N parsed conversation dates when enabled.",
    )
    parser.add_argument("--reflect-limit", type=int)
    parser.add_argument("--recall-top-k", type=int)
    parser.add_argument("--recall-budget")
    parser.add_argument(
        "--recall-mode",
        choices=("stage1", "stage2", "normal"),
        default=None,
        help=(
            "Override config.yaml recall_mode; when omitted, use the configured "
            "recall path."
        ),
    )
    parser.add_argument("--skip-embedding-validation", action="store_true")
    parser.add_argument(
        "--run-reader",
        action="store_true",
        help="Use the configured memory LLM to answer each question from recall context.",
    )
    parser.add_argument("--reader-max-context-chars", type=int, default=12000)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--manager-log-level", default="INFO")
    return parser.parse_args()


def load_records(path: Path) -> List[Dict[str, Any]]:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"CLongEval input does not exist: {path}")
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(item, dict):
                continue
            context = str(item.get("context") or "").strip()
            query = str(item.get("query") or "").strip()
            if not context or not query:
                logging.warning("Skipping record without context/query at line %s", line_number)
                continue
            records.append({
                **item,
                "context": context,
                "query": query,
                "answer": str(item.get("answer") or "").strip(),
                "id": str(item.get("id") or f"line_{line_number}"),
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


def parse_date(value: str) -> datetime:
    return datetime.strptime(value, "%Y年%m月%d日")


def _flush_message(messages: List[Tuple[str, str]], role: Optional[str], parts: List[str]) -> None:
    text = "\n".join(part for part in parts if part).strip().strip("“”")
    if text and role:
        messages.append((role, text))


def parse_context_days(context: str) -> List[Dict[str, Any]]:
    """Parse dated 用户/AI blocks from a CLongEval context string."""
    headers = list(DATE_HEADER_RE.finditer(context))
    if not headers:
        raise ValueError("Context does not contain a dated conversation header")

    days: List[Dict[str, Any]] = []
    for index, header in enumerate(headers):
        section_end = headers[index + 1].start() if index + 1 < len(headers) else len(context)
        section = context[header.end() : section_end]
        for marker in CONTEXT_END_MARKERS:
            marker_index = section.find(marker)
            if marker_index >= 0:
                section = section[:marker_index]
                break
        section = section.strip().strip("“”")

        messages: List[Tuple[str, str]] = []
        current_role: Optional[str] = None
        current_parts: List[str] = []
        for raw_line in section.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            match = ROLE_LINE_RE.match(line)
            if match:
                _flush_message(messages, current_role, current_parts)
                current_role = "user" if match.group(1) == "用户" else "assistant"
                current_parts = [match.group(2).strip()]
            elif current_role:
                current_parts.append(line.strip("“”"))
        _flush_message(messages, current_role, current_parts)

        pairs: List[Tuple[str, str]] = []
        pending_user: Optional[str] = None
        for role, text in messages:
            if role == "user":
                pending_user = text if not pending_user else f"{pending_user}\n{text}"
            elif pending_user:
                pairs.append((pending_user, text))
                pending_user = None

        if not pairs:
            logging.warning("No user/assistant pairs parsed for %s", header.group("date"))
            continue
        days.append({
            "date_text": header.group("date"),
            "date": parse_date(header.group("date")),
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


def resolve_existing_context_db(
    existing_state_dir: Path,
    *,
    context: str,
    group_id: str,
) -> Path:
    """Resolve a prior context DB, tolerating a changed filtered group index."""
    existing_root = Path(existing_state_dir).expanduser().resolve()
    direct_path = existing_root / group_id / "memory.db"
    if direct_path.is_file():
        return direct_path

    digest = hashlib.sha1(context.encode("utf-8")).hexdigest()[:12]
    matches = sorted(existing_root.glob(f"context_*_{digest}/memory.db"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            "Existing memory DB not found for context "
            f"{group_id} under {existing_root}"
        )
    raise RuntimeError(
        "Multiple existing memory DBs matched context "
        f"{group_id} under {existing_root}"
    )


def group_records_by_context(
    records: Sequence[Dict[str, Any]],
) -> List[Tuple[str, List[Dict[str, Any]]]]:
    grouped: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()
    for record in records:
        grouped.setdefault(str(record["context"]), []).append(record)
    return list(grouped.items())


def db_counts(db: Any) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for table in (
        "memory_episodes",
        "memory_facts",
    ):
        row = db._conn.execute(f"SELECT COUNT(*) AS count FROM {table}").fetchone()
        counts[table] = int(row["count"] if row else 0)
    return counts


def replay_context_into_memory(
    *,
    runtime: MemoryRuntime,
    db: Any,
    days: Sequence[Dict[str, Any]],
    group_id: str,
    enable_reflect: bool,
    reflect_every_days: int,
    reflect_limit: int,
) -> Dict[str, Any]:
    seen_timestamps: set[str] = set()
    pair_count = 0
    store_batches = 0
    reflect_runs = 0
    reflect_reports: List[Dict[str, Any]] = []
    last_reflected_day = 0
    last_turn_timestamp: Optional[datetime] = None
    ambient_watermark_updates = 0
    every_days = max(1, int(reflect_every_days or 1))

    for day_index, day in enumerate(days, 1):
        for pair_index, (user, assistant) in enumerate(day["pairs"]):
            pair_count += 1
            turn_timestamp = normalize_unique_timestamp(
                day["date"] + timedelta(seconds=pair_index),
                seen_timestamps,
            )
            store_report = runtime.accept_memory_input(
                interaction_turn={
                    "user_message": user,
                    "assistant_response": assistant,
                    "turn_timestamp": turn_timestamp,
                },
                tags=[
                    "clongeval_conversation",
                    f"context_group:{group_id}",
                    f"date:{day['date_text']}",
                    f"day_index:{day_index}",
                ],
                # Exercise the delayed-ASR path: each interaction stays in
                # the context manager's awaiting buffer until this benchmark
                # advances the simulated ambient-ASR watermark.
                ambient_recording_enabled=True,
            )
            last_turn_timestamp = turn_timestamp
            if store_report.get("queued"):
                store_batches += 1

        should_reflect = enable_reflect and (
            day_index % every_days == 0 or day_index == len(days)
        )
        if should_reflect:
            # Reflection creates an explicit partial boundary.  Advance
            # coverage through the final turn of this day before flushing, so
            # the flush does not need to force-release delayed input.
            watermark_report = runtime.update_ambient_asr_watermark(
                last_turn_timestamp,
                evaluate_episode_summary=False,
            )
            ambient_watermark_updates += 1
            if watermark_report.get("queued"):
                store_batches += 1
            # Facts are filtered by their local DB created_at date. Keep this
            # wall-clock timestamp separate from the benchmark dialogue time.
            reflect_timestamp = datetime.now().astimezone().isoformat()
            episode_input_flush = runtime.flush_pending_memory_inputs(
                evaluate_episode_summary=False,
            )
            episode_summary_submit = runtime.trigger_memory_episode_summary(
                reason="clongeval_reflect",
                tags=[
                    "clongeval_conversation",
                    f"context_group:{group_id}",
                    f"date:{day['date_text']}",
                    f"day_index:{day_index}",
                ],
            )
            reflect_submit = runtime.trigger_memory_reflect(
                limit=max(1, int(reflect_limit or 100)),
                reflect_timestamp=reflect_timestamp,
            )
            if episode_input_flush:
                store_batches += 1
            if (reflect_submit.get("pending_memory_input_flush") or {}).get("queued"):
                store_batches += 1
            if (
                episode_summary_submit.get("queued") or reflect_submit.get("queued")
            ) and not runtime.wait_for_memory_tasks():
                raise RuntimeError("Timed out while draining queued memory reflect")
            reflect_runs += 1
            last_reflected_day = day_index
            reflect_reports.append({
                "day_index": day_index,
                "date": day["date_text"],
                "ambient_watermark": watermark_report,
                "episode_summary": episode_summary_submit,
                "report": reflect_submit,
            })

    # The normal benchmark path reaches this point only after every
    # interaction has entered the runtime.  One final watermark advance
    # releases all remaining delayed input through the same append path.
    if last_turn_timestamp is not None:
        watermark_report = runtime.update_ambient_asr_watermark(
            last_turn_timestamp,
        )
        ambient_watermark_updates += 1
        if watermark_report.get("queued"):
            store_batches += 1
    runtime.flush_pending_memory_inputs()

    return {
        "parsed_days": len(days),
        "turn_pairs": pair_count,
        "store_batches": store_batches,
        "reflect_runs": reflect_runs,
        "ambient_watermark_updates": ambient_watermark_updates,
        "last_reflected_day": last_reflected_day,
        "reflect_reports": reflect_reports,
        "db_counts": db_counts(db),
    }


def normalize_for_match(value: Any) -> str:
    text = str(value or "").lower()
    return re.sub(r"\s+", "", text)


def gold_answer_in_context(answer: str, context: str) -> bool:
    normalized_answer = normalize_for_match(answer)
    return bool(normalized_answer) and normalized_answer in normalize_for_match(context)


def truncate_text(value: str, max_chars: int) -> str:
    text = str(value or "")
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 24)] + "...[truncated]"


def build_reader_prompt(query: str, memory_context: str) -> str:
    return (
        "你是一个长期记忆问答评测器。请只依据下面的记忆上下文回答问题，不要补充上下文之外的事实。\n"
        "问题可能是对原始对话的概括或改写，问题中的动作、时间和表达方式不一定与上下文逐字一致。\n"
        "回答时重点识别问题真正询问的实体、名称、属性或具体事实；只要上下文能够明确确定该答案，就应当回答，不要因为措辞差异而返回空字符串。\n"
        "例如，问题说‘推荐过哪本书’，上下文说‘正在读一本书，叫《小王子》’，则答案应为‘《小王子》’。\n"
        "答案必须有上下文证据支持，不能根据常识或猜测补全；如果上下文确实没有足够相关证据，才返回空字符串。\n"
        "答案尽量简短，并优先复用上下文中的原文表达，不要解释推理过程。\n"
        "只返回合法 JSON：{\"answer\": \"...\"}\n\n"
        f"问题：{query}\n\n"
        "记忆上下文：\n"
        f"{memory_context or '[empty]'}"
    )


def run_reader(manager: Any, query: str, memory_context: str, max_chars: int) -> str:
    if not memory_context.strip():
        return ""
    raw = manager._call_llm(
        build_reader_prompt(query, truncate_text(memory_context, max_chars))
    )
    parsed = manager._parse_json_object_from_llm_text(raw or "")
    if isinstance(parsed, dict):
        return str(parsed.get("answer") or parsed.get("final_answer") or "").strip()
    return str(raw or "").strip()


def prepare_runtime(
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    resolve_llm_args(args)
    config = load_project_config(args.config)
    memory_runtime_config, memory_manager_config = split_memory_config(config)
    llm_config = memory_manager_config["llm"]
    embedding_config = memory_manager_config["embedding"]
    if args.llm_thinking is None:
        args.llm_thinking = str(llm_config.get("llm_thinking") or "disabled")
    if args.llm_json_mode is None:
        args.llm_json_mode = bool(llm_config.get("llm_json_mode", True))
    args.reflect_limit = max(1, int(args.reflect_limit or memory_manager_config.get("reflect_limit", 100) or 100))
    args.recall_top_k = max(1, int(args.recall_top_k or memory_manager_config.get("recall_top_k", 8) or 8))
    args.recall_budget = str(args.recall_budget or memory_manager_config.get("recall_budget", "mid") or "mid")
    if args.recall_mode is not None:
        memory_manager_config["recall_mode"] = args.recall_mode
    args.recall_mode = str(
        memory_manager_config.get("recall_mode", "normal") or "normal"
    ).strip().lower()
    memory_input_config = memory_runtime_config.setdefault(
        "memory_context_manager",
        {},
    )
    segmentation_config = memory_input_config.setdefault("fact_extraction", {})
    if args.max_pending_interaction_turns is not None:
        segmentation_config["max_pending_units"] = args.max_pending_interaction_turns
    if args.max_pending_interaction_tokens is not None:
        segmentation_config["max_pending_tokens"] = args.max_pending_interaction_tokens
    llm_config["llm_name"] = str(args.llm_model)
    llm_config["llm_base_url"] = str(args.llm_base_url)
    llm_config["llm_api_key"] = str(args.llm_api_key or "")
    llm_config["llm_timeout"] = args.llm_timeout
    memory_manager_config["enable_entity_extraction"] = False
    return memory_runtime_config, memory_manager_config, llm_config, embedding_config


def resolve_output_dir(args: argparse.Namespace) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root_dir = Path(args.output_root_dir).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else output_root_dir / f"{args.input.stem}_{timestamp}"
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Output directory is not empty: {output_dir}. Pass --overwrite.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    args.output_root_dir = output_root_dir
    args.output_dir = output_dir
    return output_dir


def add_context_log_handler(log_path: Path) -> logging.Handler:
    """Mirror the current context's records into its own output directory."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    handler.setLevel(logging.NOTSET)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logging.getLogger().addHandler(handler)
    return handler


def remove_context_log_handler(handler: Optional[logging.Handler]) -> None:
    if handler is None:
        return
    root_logger = logging.getLogger()
    root_logger.removeHandler(handler)
    handler.close()


def configure_memory_logger(
    log_path: Path,
    log_level: str,
    logger_name: str,
) -> Tuple[logging.Logger, logging.Handler]:
    """Create a non-propagating logger dedicated to memory operations."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    memory_logger = logging.getLogger(logger_name)
    for existing_handler in list(memory_logger.handlers):
        memory_logger.removeHandler(existing_handler)
        existing_handler.close()
    memory_logger.setLevel(getattr(logging, str(log_level).upper(), logging.INFO))
    memory_logger.propagate = False
    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    handler.setLevel(logging.NOTSET)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    memory_logger.addHandler(handler)
    return memory_logger, handler


def remove_memory_logger_handler(
    memory_logger: Optional[logging.Logger],
    handler: Optional[logging.Handler],
) -> None:
    """Close a context-specific memory logger without affecting root logging."""
    if memory_logger is None or handler is None:
        return
    memory_logger.removeHandler(handler)
    handler.close()


def configure_worker_logging(
    log_path: Path,
    log_level: str,
    manager_log_level: str,
) -> None:
    """Configure file-only logging inside a worker process."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(getattr(logging, str(log_level).upper(), logging.INFO))
    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    root_logger.addHandler(handler)
    manager_level = getattr(
        logging,
        str(manager_log_level).upper(),
        logging.INFO,
    )
    logging.getLogger("agent.memory_node_manager").setLevel(manager_level)
    logging.getLogger("memory").setLevel(manager_level)
    logging.getLogger("memory.memory_manager").setLevel(manager_level)


def process_context_group(
    *,
    args: argparse.Namespace,
    group_index: int,
    total_groups: int,
    total_records: int,
    context: str,
    group_records: Sequence[Dict[str, Any]],
    memory_runtime_config: Dict[str, Any],
    memory_manager_config: Dict[str, Any],
    llm_config: Dict[str, Any],
    embedding_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Replay, reflect, recall, and optionally answer one isolated context."""
    group_id = context_group_id(context, group_index)
    group_dir = args.output_dir / group_id
    group_dir.mkdir(parents=True, exist_ok=True)
    reused_existing_db = args.existing_state_dir is not None
    db_path = (
        resolve_existing_context_db(
            args.existing_state_dir,
            context=context,
            group_id=group_id,
        )
        if reused_existing_db
        else group_dir / "memory.db"
    )
    context_log_path = group_dir / "context.log"
    logging.info(
        "Processing context group %s/%s id=%s records=%s context_chars=%s",
        group_index,
        total_groups,
        group_id,
        len(group_records),
        len(context),
    )
    days = [] if reused_existing_db else parse_context_days(context)
    context_log_handler = add_context_log_handler(context_log_path)
    memory_logger: Optional[logging.Logger] = None
    memory_log_handler: Optional[logging.Handler] = None
    runtime: Optional[MemoryRuntime] = None
    results: List[Dict[str, Any]] = []
    try:
        memory_logger, memory_log_handler = configure_memory_logger(
            group_dir / "memory.log",
            args.manager_log_level,
            f"memory.pipeline.{group_id}",
        )
        logging.info(
            "Context processing started id=%s records=%s parsed_days=%s "
            "reused_existing_db=%s db_path=%s",
            group_id,
            len(group_records),
            len(days),
            reused_existing_db,
            db_path,
        )
        operation_reporter = MemoryOperationReporter()
        runtime = MemoryRuntime(
            db_path=db_path,
            memory_runtime_config=memory_runtime_config,
            memory_manager_config=memory_manager_config,
            operation_reporter=operation_reporter,
            logger=memory_logger,
        )
        db = runtime.database
        manager = runtime.manager
        if not args.skip_embedding_validation:
            validate_embedding_runtime(manager, db, embedding_config)
        log_memory_index_state(db, f"before_context:{group_id}")
        if reused_existing_db:
            replay_stats = {
                "parsed_days": 0,
                "turn_pairs": 0,
                "store_batches": 0,
                "reflect_runs": 0,
                "last_reflected_day": 0,
                "reflect_reports": [],
                "db_counts": db_counts(db),
            }
        else:
            replay_stats = replay_context_into_memory(
                runtime=runtime,
                db=db,
                days=days,
                group_id=group_id,
                enable_reflect=args.enable_reflect,
                reflect_every_days=args.reflect_every_days,
                reflect_limit=args.reflect_limit,
            )
            if not runtime.flush_pending_memory_inputs():
                raise RuntimeError("Timed out while draining queued memory stores")
            if not runtime.wait_for_memory_tasks():
                raise RuntimeError(
                    "Timed out while draining queued memory store and reflect tasks"
                )
        log_memory_index_state(db, f"after_context:{group_id}")
        counts = db_counts(db)
        memory_operation_report = operation_reporter.snapshot()
        operation_counts = memory_operation_report.get("counts") or {}
        store_operation_report = operation_counts.get("memory_store") or {}
        reflect_operation_report = operation_counts.get("memory_reflect") or {}
        replay_stats["store_batches"] = int(store_operation_report.get("succeeded") or 0)
        replay_stats["reflect_runs"] = int(reflect_operation_report.get("submitted") or 0)
        replay_stats["store_flushes"] = int(store_operation_report.get("submitted") or 0)

        for record_index, record in enumerate(group_records, 1):
            recall_report = runtime.trigger_memory_recall(
                str(record["query"]),
            )
            recall_context = str(recall_report.get("memory_context") or "")
            covered = gold_answer_in_context(record.get("answer", ""), recall_context)
            hypothesis = ""
            if args.run_reader:
                hypothesis = run_reader(
                    manager,
                    str(record["query"]),
                    recall_context,
                    args.reader_max_context_chars,
                )
            actual_recall_mode = str(
                recall_report.get("actual_recall_mode") or "unknown"
            )
            memory_operation_report = operation_reporter.snapshot()
            operation_counts = memory_operation_report.get("counts") or {}
            recall_operation_report = operation_counts.get("recall") or {}
            recall_total_elapsed_ms = float(
                operation_reporter.latest_report("recall").get("elapsed_ms")
                or recall_report.get("elapsed_ms")
                or 0.0
            )
            result = {
                "id": record["id"],
                "source_line": record.get("_source_line"),
                "context_group_id": group_id,
                "query": record["query"],
                "answer": record.get("answer", ""),
                "hypothesis": hypothesis,
                "answer_in_recall_context": covered,
                "recall_mode": args.recall_mode,
                "requested_recall_mode": args.recall_mode,
                "actual_recall_mode": actual_recall_mode,
                "recall_status": str(
                    recall_report.get("status")
                    or ("ok" if recall_context else "empty")
                ),
                "store_total_elapsed_ms": float(store_operation_report.get("total_elapsed_ms") or 0.0),
                "reflect_total_elapsed_ms": float(reflect_operation_report.get("total_elapsed_ms") or 0.0),
                "recall_total_elapsed_ms": recall_total_elapsed_ms,
                "memory_total_elapsed_ms": round(
                    float(store_operation_report.get("total_elapsed_ms") or 0.0)
                    + float(reflect_operation_report.get("total_elapsed_ms") or 0.0)
                    + recall_total_elapsed_ms,
                    2,
                ),
                "recall_context_chars": len(recall_context or ""),
                "recall_context": recall_context,
                "db_path": str(db_path),
                "reused_existing_db": reused_existing_db,
                "db_counts": counts,
            }
            results.append(result)
            logging.info(
                "[%s/%s] context_record=%s/%s id=%s group=%s recall_chars=%s "
                "actual_recall_mode=%s answer_in_context=%s",
                group_index,
                total_groups,
                record_index,
                len(group_records),
                record["id"],
                group_id,
                len(recall_context or ""),
                actual_recall_mode,
                covered,
            )

        return {
            "context_group_id": group_id,
            "record_count": len(group_records),
            "context_chars": len(context),
            "db_path": str(db_path),
            "reused_existing_db": reused_existing_db,
            "context_log_path": str(context_log_path),
            "memory_log_path": str(group_dir / "memory.log"),
            "days": len(days),
            "replay": replay_stats,
            "db_counts": counts,
            "memory_operation_report": operation_reporter.snapshot(),
            "results": results,
        }
    finally:
        if runtime is not None:
            log_memory_index_state(runtime.database, f"finished_context:{group_id}")
            runtime.close()
        remove_memory_logger_handler(memory_logger, memory_log_handler)
        remove_context_log_handler(context_log_handler)


def run_context_group_worker(
    payload: Tuple[
        int,
        int,
        int,
        str,
        List[Dict[str, Any]],
        argparse.Namespace,
        Dict[str, Any],
        Dict[str, Any],
        Dict[str, Any],
        Dict[str, Any],
    ],
) -> Tuple[int, Dict[str, Any]]:
    """Process one CLongEval context in an isolated worker process."""
    (
        group_index,
        total_groups,
        total_records,
        context,
        group_records,
        args,
        memory_runtime_config,
        memory_manager_config,
        llm_config,
        embedding_config,
    ) = payload
    group_id = context_group_id(context, group_index)
    group_dir = args.output_dir / group_id
    configure_worker_logging(
        group_dir / "worker.log",
        args.log_level,
        args.manager_log_level,
    )
    logging.info(
        "Worker started for context group %s/%s id=%s records=%s",
        group_index,
        total_groups,
        group_id,
        len(group_records),
    )
    result = process_context_group(
        args=args,
        group_index=group_index,
        total_groups=total_groups,
        total_records=total_records,
        context=context,
        group_records=group_records,
        memory_runtime_config=memory_runtime_config,
        memory_manager_config=memory_manager_config,
        llm_config=llm_config,
        embedding_config=embedding_config,
    )
    logging.info(
        "Worker finished context group %s/%s id=%s recall_results=%s",
        group_index,
        total_groups,
        group_id,
        len(result.get("results") or []),
    )
    return group_index, result


def main() -> int:
    args = parse_args()
    if args.existing_state_dir:
        args.existing_state_dir = args.existing_state_dir.expanduser().resolve()
        if not args.existing_state_dir.is_dir():
            raise FileNotFoundError(
                f"--existing-state-dir does not exist: {args.existing_state_dir}"
            )
        if args.output_dir:
            requested_output_dir = args.output_dir.expanduser().resolve()
            if requested_output_dir == args.existing_state_dir:
                raise ValueError(
                    "--output-dir must differ from --existing-state-dir so the "
                    "existing memory DBs are not overwritten."
                )
    records = filter_records(
        load_records(args.input),
        question_ids=args.question_id,
        start=args.start,
        limit=args.limit,
    )
    if not records:
        raise RuntimeError("No CLongEval records selected")
    output_dir = resolve_output_dir(args)
    log_path = output_dir / "run_clongeval_conversation.log"
    configure_logging(log_path, args.log_level, args.manager_log_level)
    logging.info("Loaded CLongEval records=%s input=%s", len(records), args.input)

    (
        memory_runtime_config,
        memory_manager_config,
        llm_config,
        embedding_config,
    ) = prepare_runtime(args)
    groups = group_records_by_context(records)
    results_path = output_dir / "clongeval_conversation_results.json"
    results: List[Dict[str, Any]] = []
    group_summaries: List[Dict[str, Any]] = []
    result_count = 0
    recall_nonempty = 0
    gold_covered = 0
    reader_answered = 0

    workers = max(1, int(args.workers or 1))
    group_results_by_index: Dict[int, Dict[str, Any]] = {}
    group_errors_by_index: Dict[int, Dict[str, Any]] = {}
    if workers > 1:
        logging.info(
            "Processing %s context groups with %s worker processes",
            len(groups),
            workers,
        )
        payloads = [
            (
                group_index,
                len(groups),
                len(records),
                context,
                list(group_records),
                args,
                memory_runtime_config,
                memory_manager_config,
                llm_config,
                embedding_config,
            )
            for group_index, (context, group_records) in enumerate(groups, 1)
        ]
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(run_context_group_worker, payload): payload
                for payload in payloads
            }
            for future in as_completed(futures):
                payload = futures[future]
                group_index, _total, _records, context, group_records, *_configs = payload
                group_id = context_group_id(context, group_index)
                try:
                    result_index, group_result = future.result()
                    group_results_by_index[result_index] = group_result
                    logging.info(
                        "Context group ready index=%s id=%s records=%s",
                        result_index,
                        group_id,
                        len(group_records),
                    )
                except Exception as exc:
                    group_errors_by_index[group_index] = {
                        "context_group_id": group_id,
                        "record_count": len(group_records),
                        "status": "error",
                        "stage": "context_worker",
                        "error": str(exc),
                    }
                    logging.exception(
                        "Failed to process context group %s: %s",
                        group_id,
                        exc,
                    )
    else:
        for group_index, (context, group_records) in enumerate(groups, 1):
            try:
                group_results_by_index[group_index] = process_context_group(
                    args=args,
                    group_index=group_index,
                    total_groups=len(groups),
                    total_records=len(records),
                    context=context,
                    group_records=group_records,
                    memory_runtime_config=memory_runtime_config,
                    memory_manager_config=memory_manager_config,
                    llm_config=llm_config,
                    embedding_config=embedding_config,
                )
            except Exception as exc:
                group_id = context_group_id(context, group_index)
                group_errors_by_index[group_index] = {
                    "context_group_id": group_id,
                    "record_count": len(group_records),
                    "status": "error",
                    "stage": "context",
                    "error": str(exc),
                }
                logging.exception("Failed to process context group %s: %s", group_id, exc)

    for group_index, (context, group_records) in enumerate(groups, 1):
        if group_index in group_errors_by_index:
            group_summaries.append(group_errors_by_index[group_index])
            continue
        group_result = group_results_by_index.get(group_index)
        if group_result is None:
            group_summaries.append({
                "context_group_id": context_group_id(context, group_index),
                "record_count": len(group_records),
                "status": "error",
                "stage": "context_worker",
                "error": "Context worker returned no result",
            })
            continue
        group_summaries.append({
            key: value for key, value in group_result.items() if key != "results"
        })
        for result in group_result.get("results") or []:
            results.append(result)
            result_count += 1
            recall_nonempty += int(bool(result.get("recall_context")))
            gold_covered += int(result.get("answer_in_recall_context"))
            reader_answered += int(bool(result.get("hypothesis")))

    results_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = {
        "input": str(args.input.resolve()),
        "config": str(args.config.resolve()),
        "output_dir": str(output_dir),
        "results_path": str(results_path),
        "results_format": "json",
        "log_path": str(log_path),
        "workers": workers,
        "records_processed": result_count,
        "context_groups": len(groups),
        "recall_nonempty": recall_nonempty,
        "gold_answer_in_recall_context": gold_covered,
        "reader_answered": reader_answered,
        "gold_coverage_rate": gold_covered / result_count if result_count else 0.0,
        "recall_mode": args.recall_mode,
        "reused_existing_db": bool(args.existing_state_dir),
        "actual_recall_mode_counts": {
            path: sum(
                1
                for item in results
                if item.get("actual_recall_mode") == path
            )
            for path in sorted(
                {str(item.get("actual_recall_mode") or "unknown") for item in results}
            )
        },
        "enable_reflect": bool(args.enable_reflect),
        "group_summaries": group_summaries,
    }
    summary_path = output_dir / "clongeval_conversation_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info(
        "Finished CLongEval records=%s groups=%s recall_nonempty=%s gold_coverage=%s/%s",
        result_count,
        len(groups),
        recall_nonempty,
        gold_covered,
        result_count,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
