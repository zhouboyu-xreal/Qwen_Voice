#!/usr/bin/env python3
"""Build project memory for LoCoMo and emit evaluator-compatible predictions.

Each LoCoMo ``sample_id`` gets an isolated memory database.  The script replays
the dated conversation sessions into :class:`MemoryNodeManager`, optionally
reflects after sessions, recalls for every selected QA row, and asks a reader
model for a concise answer.  The JSON output keeps LoCoMo's native
``[{"sample_id": ..., "qa": [...]}]`` structure and is consumable by
``task_eval/evaluate_qa.py --model hermes-memory --hermes-answers-file``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import requests

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
for import_root in (SRC_ROOT, REPO_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from memory.memory_manager import (
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
    MemoryOperationReporter,
)
from memory.memory_runtime import MemoryRuntime
from memory.config import split_memory_config
from memory.utils import _as_embedding_vector
import memory.memory_database as memory_database_module


DEFAULT_INPUT = Path(
    "/Users/zhouboyu/Documents/xreal/项目/agent_memory/benchmark/locomo/data/locomo10.json"
)
DEFAULT_OUTPUT_ROOT_DIR = REPO_ROOT / "tmp" / "locomo"
DEFAULT_CONTEXT_KEY = "hermes-memory_context_text"
DEFAULT_PREDICTION_KEY = "hermes-memory_prediction"
NO_INFORMATION_ANSWER = "No information available."
_ENV_REF_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_SESSION_KEY_RE = re.compile(r"^session_(\d+)$")
_READER_HTTP_SESSION: Optional[requests.Session] = None


@dataclass(frozen=True)
class LocomoReplayStats:
    dialog_turns_total: int
    turn_pairs_total: int
    stored_pairs: int
    orphan_dialog_turns: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build MemoryNodeManager databases and answers for LoCoMo."
    )
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config.yaml")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root-dir", type=Path, default=DEFAULT_OUTPUT_ROOT_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--log-path", type=Path)
    parser.add_argument("--detail-output", type=Path)
    parser.add_argument("--context-key", default=DEFAULT_CONTEXT_KEY)
    parser.add_argument("--prediction-key", default=DEFAULT_PREDICTION_KEY)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="0 means all samples.")
    parser.add_argument("--sample-id", action="append", help="Only run this sample_id.")
    parser.add_argument("--qa-start", type=int, default=0)
    parser.add_argument("--qa-limit", type=int, default=0, help="0 means all QA rows.")
    parser.add_argument("--max-sessions", type=int, default=0, help="0 means all sessions.")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Worker processes for memory construction and recall; reader stays in main process.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--llm-model")
    parser.add_argument("--llm-base-url")
    parser.add_argument("--llm-api-key")
    parser.add_argument("--llm-timeout", type=int)
    parser.add_argument(
        "--llm-thinking", choices=("disabled", "enabled", "auto"), default=None
    )
    parser.add_argument("--no-llm-json-mode", action="store_false", dest="llm_json_mode")
    parser.set_defaults(llm_json_mode=True)
    parser.add_argument(
        "--memory-prompt-language", choices=("source", "en", "zh"), default=None
    )
    parser.add_argument(
        "--memory-output-language", choices=("source", "en", "zh"), default=None
    )
    parser.add_argument("--reader-model")
    parser.add_argument("--reader-base-url")
    parser.add_argument("--reader-api-key")
    parser.add_argument("--reader-timeout", type=int)
    parser.add_argument("--reader-max-tokens", type=int, default=8192)
    parser.add_argument("--reader-temperature", type=float, default=0.0)
    parser.add_argument("--reader-max-context-chars", type=int, default=16000)
    parser.add_argument("--recall-top-k", type=int, default=8)
    parser.add_argument("--recall-budget", choices=("low", "mid", "high"), default=None)
    parser.add_argument(
        "--recall-mode", choices=("stage1", "stage2", "normal"), default="normal"
    )
    parser.add_argument(
        "--recall-gate-mode", choices=("auto", "force"), default=None
    )
    parser.add_argument(
        "--recall-memory-source",
        action="append",
        choices=("assistant_wakeup", "allday_recording"),
    )
    parser.add_argument("--enable-reflect", action="store_true")
    parser.add_argument("--reflect-every-sessions", type=int, default=1)
    parser.add_argument("--reflect-limit", type=int, default=100)
    parser.add_argument(
        "--fact-extraction-interval",
        type=int,
        default=None,
        help=(
            "Override memory_runtime.memory_context_manager.fact_extraction."
            "max_pending_units "
            "from config.yaml."
        ),
    )
    parser.add_argument(
        "--fact-extraction-max-tokens",
        type=int,
        default=None,
        help=(
            "Override memory_runtime.memory_context_manager.fact_extraction."
            "max_pending_tokens "
            "from config.yaml."
        ),
    )
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--manager-log-level", default="INFO")
    return parser.parse_args()


def configure_logging(
    log_path: Path, log_level: str, manager_log_level: str, *, stream: bool = True
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(getattr(logging, str(log_level).upper(), logging.INFO))
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)
    if stream:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        root.addHandler(stream_handler)
@contextmanager
def sample_memory_logging(
    sample_state_dir: Path,
    manager_log_level: str,
) -> Iterator[Tuple[Path, logging.Logger]]:
    """Write memory pipeline logs beside one LoCoMo sample database."""
    sample_state_dir.mkdir(parents=True, exist_ok=True)
    log_path = sample_state_dir / "memory_manager.log"
    memory_logger = logging.getLogger(
        f"memory.pipeline.locomo.{sample_state_dir.name}"
    )
    for existing_handler in list(memory_logger.handlers):
        memory_logger.removeHandler(existing_handler)
        existing_handler.close()
    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    memory_logger.addHandler(handler)
    memory_logger.setLevel(getattr(logging, str(manager_log_level).upper(), logging.INFO))
    memory_logger.propagate = False
    try:
        yield log_path, memory_logger
    finally:
        memory_logger.removeHandler(handler)
        handler.close()


def _expand_env_refs(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_env_refs(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env_refs(item) for item in value]
    if isinstance(value, str):
        return _ENV_REF_RE.sub(lambda match: os.getenv(match.group(1), ""), value)
    return value


def load_project_config(config_path: Path) -> Dict[str, Any]:
    config_path = config_path.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")
    import yaml

    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected YAML mapping in config file: {config_path}")
    return _expand_env_refs(loaded)


def resolve_runtime_args(args: argparse.Namespace) -> None:
    config = load_project_config(args.config)
    model_config = config.get("model") if isinstance(config.get("model"), dict) else {}
    runtime_config, manager_config = split_memory_config(config)
    llm_config = manager_config["llm"]
    chat_config = config.get("chat") if isinstance(config.get("chat"), dict) else {}
    args.llm_model = args.llm_model or str(llm_config.get("llm_name") or model_config.get("default") or DEFAULT_LLM_MODEL)
    args.llm_base_url = args.llm_base_url or str(llm_config.get("llm_base_url") or model_config.get("base_url") or DEFAULT_LLM_BASE_URL)
    if args.llm_base_url.rstrip("/") == "https://api.deepseek.com":
        args.llm_base_url = "https://api.deepseek.com/v1"
    args.llm_api_key = args.llm_api_key or str(llm_config.get("llm_api_key") or model_config.get("api_key") or "")
    args.llm_timeout = args.llm_timeout or int(str(llm_config.get("llm_timeout", 120)))
    args.llm_thinking = args.llm_thinking or str(llm_config.get("llm_thinking") or "disabled")
    args.memory_prompt_language = args.memory_prompt_language or str(
        runtime_config.get("memory_prompt_language_mode")
        or manager_config.get("memory_prompt_language_mode")
        or "source"
    )
    args.memory_output_language = args.memory_output_language or str(
        runtime_config.get("memory_output_language_mode")
        or manager_config.get("memory_output_language_mode")
        or "source"
    )
    args.recall_budget = args.recall_budget or str(manager_config.get("recall_budget") or "mid")
    args.reader_model = args.reader_model or str(chat_config.get("llm_name") or args.llm_model)
    args.reader_base_url = args.reader_base_url or str(chat_config.get("llm_base_url") or args.llm_base_url)
    if args.reader_base_url.rstrip("/") == "https://api.deepseek.com":
        args.reader_base_url = "https://api.deepseek.com/v1"
    args.reader_api_key = args.reader_api_key or str(chat_config.get("llm_api_key") or args.llm_api_key)
    args.reader_timeout = args.reader_timeout or args.llm_timeout


def prepare_runtime_configs(
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    config = load_project_config(args.config)
    memory_runtime_config, memory_manager_config = split_memory_config(config)
    llm_config = memory_manager_config["llm"]
    embedding_config = memory_manager_config["embedding"]
    memory_input_config = memory_runtime_config.setdefault(
        "memory_context_manager",
        {},
    )
    segmentation_config = memory_input_config.setdefault("fact_extraction", {})
    configured_max_pending_turns = segmentation_config.get("max_pending_units")
    if configured_max_pending_turns in (None, ""):
        configured_max_pending_turns = segmentation_config.get("max_pending_units", 1)
    max_turns = max(1, int(args.fact_extraction_interval if args.fact_extraction_interval is not None else configured_max_pending_turns))
    args.fact_extraction_interval = max_turns
    segmentation_config["max_pending_units"] = max_turns
    configured_max_tokens = segmentation_config.get("max_pending_tokens")
    if args.fact_extraction_max_tokens is not None or configured_max_tokens not in (None, ""):
        max_tokens = max(1, int(args.fact_extraction_max_tokens if args.fact_extraction_max_tokens is not None else configured_max_tokens))
        args.fact_extraction_max_tokens = max_tokens
        segmentation_config["max_pending_tokens"] = max_tokens
    llm_config["llm_name"] = str(args.llm_model)
    llm_config["llm_base_url"] = str(args.llm_base_url)
    llm_config["llm_api_key"] = str(args.llm_api_key or "")
    llm_config["llm_timeout"] = int(args.llm_timeout)
    llm_config["llm_thinking"] = str(args.llm_thinking)
    llm_config["llm_json_mode"] = bool(args.llm_json_mode)
    memory_manager_config["memory_prompt_language_mode"] = str(args.memory_prompt_language)
    memory_manager_config["memory_output_language_mode"] = str(args.memory_output_language)
    memory_runtime_config["memory_prompt_language_mode"] = str(args.memory_prompt_language)
    memory_runtime_config["memory_output_language_mode"] = str(args.memory_output_language)
    return memory_runtime_config, memory_manager_config, llm_config, embedding_config


def resolve_output_paths(args: argparse.Namespace) -> Tuple[Path, Path]:
    if args.output_dir is None:
        args.output_dir = args.output_root_dir / f"{args.input.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.state_dir is None:
        args.state_dir = args.output_dir / "state"
    if args.log_path is None:
        args.log_path = args.output_dir / "run_locomo_memory_eval.log"
    output_path = args.output_dir / "locomo_memory.json"
    detail_path = args.detail_output or args.output_dir / "locomo_memory.details.json"
    return output_path, detail_path


def remove_existing_outputs(output_path: Path, detail_path: Path, state_dir: Path, overwrite: bool) -> None:
    existing = [path for path in (output_path, detail_path) if path.exists()]
    if state_dir.exists() and any(state_dir.iterdir()):
        existing.append(state_dir)
    if existing and not overwrite:
        raise FileExistsError("Output already exists; pass --overwrite to replace:\n  " + "\n  ".join(map(str, existing)))
    for path in (output_path, detail_path):
        if path.exists():
            path.unlink()
    if state_dir.exists():
        shutil.rmtree(state_dir)


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return [item for item in data if isinstance(item, dict)]


def filter_samples(samples: Sequence[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    selected = list(samples)
    if args.sample_id:
        wanted = {str(value).strip() for value in args.sample_id if str(value).strip()}
        selected = [item for item in selected if str(item.get("sample_id") or "").strip() in wanted]
    if args.start:
        selected = selected[args.start:]
    if args.limit:
        selected = selected[:args.limit]
    return selected


def parse_locomo_datetime(value: Any) -> datetime:
    text = str(value or "").strip()
    if not text:
        raise ValueError("Empty LoCoMo session timestamp")
    return datetime.strptime(text, "%I:%M %p on %d %B, %Y")


def format_memory_time(value: datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S")


def unique_timestamp(candidate: datetime, seen: set[str]) -> datetime:
    while format_memory_time(candidate) in seen:
        candidate += timedelta(seconds=1)
    seen.add(format_memory_time(candidate))
    return candidate


def sorted_locomo_sessions(sample: Dict[str, Any], max_sessions: int) -> List[Tuple[int, datetime, str, List[Dict[str, Any]]]]:
    conversation = sample.get("conversation") if isinstance(sample.get("conversation"), dict) else {}
    session_numbers = sorted(int(match.group(1)) for key in conversation for match in [_SESSION_KEY_RE.match(str(key))] if match)
    sessions: List[Tuple[int, datetime, str, List[Dict[str, Any]]]] = []
    for number in session_numbers:
        session_key = f"session_{number}"
        date_text = str(conversation.get(f"{session_key}_date_time") or "").strip()
        turns = conversation.get(session_key)
        sessions.append((number, parse_locomo_datetime(date_text), date_text, turns if isinstance(turns, list) else []))
    return sessions[:max_sessions] if max_sessions > 0 else sessions


def dialog_turn_text(turn: Dict[str, Any]) -> str:
    speaker = str(turn.get("speaker") or "Unknown speaker").strip()
    text = str(turn.get("text") or "").strip()
    if not text:
        return ""
    dia_id = str(turn.get("dia_id") or "").strip()
    caption = str(turn.get("blip_caption") or "").strip()
    result = f'{speaker} said: "{text}"'
    if caption:
        result += f" Shared image caption: {caption}"
    return f"[{dia_id}] {result}" if dia_id else result


def paired_dialog_turns(turns: Sequence[Dict[str, Any]], date_text: str) -> Tuple[List[Tuple[str, str, List[str]]], int]:
    pairs: List[Tuple[str, str, List[str]]] = []
    orphan_count = 0
    for index in range(0, len(turns), 2):
        if index + 1 >= len(turns):
            orphan_count += 1
            break
        first = turns[index] if isinstance(turns[index], dict) else {}
        second = turns[index + 1] if isinstance(turns[index + 1], dict) else {}
        first_text, second_text = dialog_turn_text(first), dialog_turn_text(second)
        if not first_text or not second_text:
            orphan_count += 1
            continue
        dia_ids = [str(turn.get("dia_id") or "").strip() for turn in (first, second)]
        pairs.append((f"LoCoMo conversation date: {date_text}\n{first_text}", f"LoCoMo conversation date: {date_text}\n{second_text}", [dia_id for dia_id in dia_ids if dia_id]))
    return pairs, orphan_count


def db_counts(db: Any) -> Dict[str, int]:
    tables = {
        "episodes": "memory_episodes",
        "facts": "memory_facts",
        "entities": "memory_entity_nodes",
    }
    return {
        name: int((db._conn.execute(f"SELECT COUNT(*) AS count FROM {table}").fetchone() or {"count": 0})["count"])
        for name, table in tables.items()
    }


def validate_runtime(manager: Any) -> None:
    embedding_cfg = getattr(manager, "_embedding_cfg", {}) or {}
    logging.info("Embedding runtime: python=%s faiss_available=%s provider=%s model=%s base_url=%s dimensions=%s", sys.version.split()[0], memory_database_module._HAS_FAISS, embedding_cfg.get("provider"), embedding_cfg.get("model"), embedding_cfg.get("base_url"), embedding_cfg.get("dimensions"))
    if not manager._ensure_embedding_client():
        raise RuntimeError("Failed to initialize embedding client")
    vector = _as_embedding_vector(manager._embedding_client.embed_text("LoCoMo embedding probe"))
    if vector is None:
        raise RuntimeError("Configured embedding provider returned an invalid vector")
    logging.info("Embedding probe succeeded: dimensions=%s norm=%.6f", vector.size, float((vector @ vector) ** 0.5))


def replay_sample_into_memory(
    runtime: MemoryRuntime,
    sample: Dict[str, Any],
    sessions: Sequence[Tuple[int, datetime, str, List[Dict[str, Any]]]],
    args: argparse.Namespace,
) -> Tuple[LocomoReplayStats, int]:
    sample_id = str(sample.get("sample_id") or "unknown_sample")
    seen_timestamps: set[str] = set()
    dialog_turns = turn_pairs = orphan_turns = reflect_runs = 0

    for session_position, (session_id, session_dt, date_text, turns) in enumerate(sessions, 1):
        dialog_turns += len(turns)
        pairs, orphaned = paired_dialog_turns(turns, date_text)
        orphan_turns += orphaned
        for pair_index, (user_message, assistant_response, dia_ids) in enumerate(pairs):
            turn_pairs += 1
            report = runtime.accept_single_interaction_turn(
                user_message,
                assistant_response,
                tags=["locomo", f"sample_id:{sample_id}", f"session_id:S{session_id}", *[f"dia_id:{dia_id}" for dia_id in dia_ids]],
                turn_timestamp=unique_timestamp(session_dt + timedelta(seconds=pair_index), seen_timestamps),
            )
        if args.enable_reflect and session_position % max(1, int(args.reflect_every_sessions)) == 0:
            reflect_submit = runtime.trigger_memory_reflect(
                limit=int(args.reflect_limit),
                reflect_timestamp=unique_timestamp(
                    session_dt + timedelta(seconds=max(1, len(pairs))),
                    seen_timestamps,
                ),
            )
            if reflect_submit.get("queued") and not runtime.flush_pending_memory_inputs():
                raise RuntimeError("Timed out while draining queued memory reflect")
            reflect_runs += 1

    if runtime.has_pending_interaction_turns() and not runtime.flush_pending_memory_inputs():
        raise RuntimeError("Timed out while draining queued memory stores")
    if args.enable_reflect and sessions:
        reflect_submit = runtime.trigger_memory_reflect(
            limit=int(args.reflect_limit),
            reflect_timestamp=unique_timestamp(
                sessions[-1][1] + timedelta(seconds=3599),
                seen_timestamps,
            ),
        )
        if reflect_submit.get("queued") and not runtime.flush_pending_memory_inputs():
            raise RuntimeError("Timed out while draining queued memory reflect")
        reflect_runs += 1
    if runtime.has_pending_interaction_turns():
        logging.warning(
            "Replay finished with %s pending turns for %s",
            len(runtime.get_pending_interaction_turns()),
            sample_id,
        )
    return (
        LocomoReplayStats(dialog_turns, turn_pairs, 0, orphan_turns),
        reflect_runs,
    )


def selected_qas(sample: Dict[str, Any], args: argparse.Namespace) -> List[Tuple[int, Dict[str, Any]]]:
    qas = sample.get("qa") if isinstance(sample.get("qa"), list) else []
    rows = [(index, qa) for index, qa in enumerate(qas) if isinstance(qa, dict)]
    if args.qa_start:
        rows = rows[args.qa_start:]
    return rows[:args.qa_limit] if args.qa_limit else rows


def build_sample_memory_context(
    args: argparse.Namespace,
    sample: Dict[str, Any],
    memory_runtime_config: Dict[str, Any],
    memory_manager_config: Dict[str, Any],
    llm_config: Dict[str, Any],
    embedding_config: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    sample_id = str(sample.get("sample_id") or "unknown_sample")
    sample_state_dir = args.state_dir / sample_id
    db_path = sample_state_dir / "memory.db"
    with sample_memory_logging(
        sample_state_dir,
        args.manager_log_level,
    ) as (memory_log_path, memory_logger):
        runtime: Optional[MemoryRuntime] = None
        try:
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
            validate_runtime(manager)
            sessions = sorted_locomo_sessions(sample, int(args.max_sessions))
            replay_stats, _ = replay_sample_into_memory(runtime, sample, sessions, args)
            if not runtime.flush_pending_memory_inputs():
                raise RuntimeError("Timed out while draining queued memory stores")
            memory_operation_report = operation_reporter.snapshot()
            operation_counts = memory_operation_report.get("counts") or {}
            store_operation_report = operation_counts.get("memory_store") or {}
            reflect_operation_report = operation_counts.get("memory_reflect") or {}
            replay_stats = LocomoReplayStats(
                replay_stats.dialog_turns_total,
                replay_stats.turn_pairs_total,
                int(store_operation_report.get("succeeded") or 0),
                replay_stats.orphan_dialog_turns,
            )
            reflect_runs = int(reflect_operation_report.get("submitted") or 0)
            counts = db_counts(db)
            recall_time_end = format_memory_time(
                sessions[-1][1] + timedelta(hours=1)
            ) if sessions else None
            output_qas = json.loads(json.dumps(sample.get("qa") or []))
            detail_rows: List[Dict[str, Any]] = []
            for qa_index, qa in selected_qas(sample, args):
                question = str(qa.get("question") or "").strip()
                recall_report = runtime.trigger_memory_recall(
                    question,
                    tags=["locomo", f"sample_id:{sample_id}"],
                    time_end=recall_time_end,
                )
                memory_context = str(recall_report.get("memory_context") or "")
                recall_operation_report = operation_reporter.latest_report("recall")
                recall_elapsed_ms = float(
                    recall_operation_report.get("elapsed_ms")
                    or recall_report.get("elapsed_ms")
                    or 0.0
                )
                output_qas[qa_index][args.context_key] = memory_context
                detail_rows.append({
                    "sample_id": sample_id,
                    "qa_index": qa_index,
                    "question": question,
                    "answer": qa.get("answer", qa.get("adversarial_answer")),
                    "category": qa.get("category"),
                    "evidence": qa.get("evidence") or [],
                    "db_path": str(db_path),
                    "memory_log_path": str(memory_log_path),
                    "db_counts": counts,
                    "recall_context": memory_context,
                    "recall_context_chars": len(memory_context),
                    "requested_recall_mode": recall_report.get("requested_recall_mode", args.recall_mode),
                    "actual_recall_mode": recall_report.get("actual_recall_mode", "unknown"),
                    "recall_status": recall_report.get("status", "empty"),
                    "recall_total_elapsed_ms": recall_elapsed_ms,
                })
            output = {
                "sample_id": sample_id,
                "qa": output_qas,
                "_memory_eval": {
                    "status": "ok",
                    "db_path": str(db_path),
                    "memory_log_path": str(memory_log_path),
                    "history_session_count": len(sessions),
                    "replayed_dialog_turns": replay_stats.dialog_turns_total,
                    "replayed_turn_pairs": replay_stats.turn_pairs_total,
                    "turn_pairs_with_stored_facts": replay_stats.stored_pairs,
                    "orphan_dialog_turns": replay_stats.orphan_dialog_turns,
                    "reflect_runs": reflect_runs,
                    "db_counts": counts,
                    "store_turn_calls": replay_stats.turn_pairs_total,
                    "store_flushes": int(store_operation_report.get("submitted") or 0),
                    "store_total_elapsed_ms": float(store_operation_report.get("total_elapsed_ms") or 0.0),
                    "reflect_total_elapsed_ms": float(reflect_operation_report.get("total_elapsed_ms") or 0.0),
                    "recall_total_elapsed_ms": float(
                        operation_reporter.operation_report("recall").get("total_elapsed_ms") or 0.0
                    ),
                    "memory_operation_report": operation_reporter.snapshot(),
                    "memory_total_elapsed_ms": round(
                        float(store_operation_report.get("total_elapsed_ms") or 0.0)
                        + float(reflect_operation_report.get("total_elapsed_ms") or 0.0)
                        + float(operation_reporter.operation_report("recall").get("total_elapsed_ms") or 0.0),
                        2,
                    ),
                },
            }
            return output, detail_rows
        finally:
            if runtime is not None:
                runtime.close()


def sample_worker(
    payload: Tuple[
        int,
        int,
        Dict[str, Any],
        argparse.Namespace,
        Dict[str, Any],
        Dict[str, Any],
        Dict[str, Any],
        Dict[str, Any],
    ],
) -> Tuple[int, Dict[str, Any], List[Dict[str, Any]]]:
    (
        index,
        total,
        sample,
        args,
        memory_runtime_config,
        memory_manager_config,
        llm_config,
        embedding_config,
    ) = payload
    sample_id = str(sample.get("sample_id") or f"sample_{index}")
    configure_logging(args.state_dir / sample_id / "run_locomo_memory_eval.log", args.log_level, args.manager_log_level, stream=False)
    logging.info("[%s/%s] Running LoCoMo sample %s", index, total, sample_id)
    output, detail_rows = build_sample_memory_context(
        args,
        sample,
        memory_runtime_config,
        memory_manager_config,
        llm_config,
        embedding_config,
    )
    logging.info("[%s/%s] Finished LoCoMo sample %s: facts=%s states=%s qa_rows=%s", index, total, sample_id, output["_memory_eval"]["db_counts"]["facts"], output["_memory_eval"]["db_counts"]["states"], len(detail_rows))
    return index, output, detail_rows


def reader_session() -> requests.Session:
    global _READER_HTTP_SESSION
    if _READER_HTTP_SESSION is None:
        _READER_HTTP_SESSION = requests.Session()
    return _READER_HTTP_SESSION


def extract_message_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(extract_message_content(item.get("text")) if isinstance(item, dict) else str(item) for item in value)
    return ""


def truncate_text(value: str, max_chars: int) -> str:
    return value if max_chars <= 0 or len(value) <= max_chars else value[:max(0, max_chars - 24)] + "\n[context truncated]"


def answer_with_reader(args: argparse.Namespace, question: str, category: Any, memory_context: str) -> str:
    if not memory_context.strip():
        return NO_INFORMATION_ANSWER
    category_text = str(category or "").strip()
    if category_text == "5":
        category_instruction = (
            "This is an adversarial or distractor question. Answer only when "
            "the retrieved memory directly supports the requested fact. Do not "
            "infer an answer from unrelated or merely similar memories. If the "
            "direct evidence is absent, answer exactly "
            f"{NO_INFORMATION_ANSWER}."
        )
    elif category_text == "1":
        category_instruction = (
            "This is a multi-hop question. Return only the requested atomic "
            "answer items in the order asked, separated by commas. Do not add "
            "numbering, explanations, or evidence."
        )
    elif category_text == "2":
        category_instruction = (
            "This is a temporal question. Return only the relevant date, year, "
            "duration, or concise relative time."
        )
    else:
        category_instruction = (
            "Return only the shortest direct answer phrase needed for the "
            "question."
        )
    prompt = (
        "Answer this LoCoMo long-term conversational-memory question using only the retrieved memory.\n"
        "Output only the final answer. Do not provide reasoning, explanations, "
        "introductions, conclusions, evidence, markdown, or a restatement of the question. "
        "Use exact names, dates, places, numbers, and short phrases from the memory whenever available. "
        "Do not paraphrase an answer when an exact answer appears in the memory. "
        "If multiple answer items are required, separate them with commas and do not add numbering. "
        f"{category_instruction}\n"
        f"If the memory is insufficient, answer exactly: {NO_INFORMATION_ANSWER}\n\n"
        f"Category: {category}\nQuestion: {question}\n\nRetrieved memory:\n"
        f"{truncate_text(memory_context, int(args.reader_max_context_chars))}\n\nAnswer:"
    )
    response = reader_session().post(
        f"{args.reader_base_url.rstrip('/')}/chat/completions",
        json={"model": args.reader_model, "messages": [{"role": "user", "content": prompt}], "temperature": float(args.reader_temperature), "max_tokens": int(args.reader_max_tokens), "stream": False},
        headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {args.reader_api_key}"} if args.reader_api_key else {})},
        timeout=int(args.reader_timeout),
    )
    response.raise_for_status()
    payload = response.json()
    choices = payload.get("choices") if isinstance(payload, dict) else []
    choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    answer = extract_message_content(message.get("content") or choice.get("text")).strip()
    if not answer:
        raise RuntimeError("Reader LLM returned an empty answer")
    return answer


def generate_reader_answers(args: argparse.Namespace, outputs: List[Dict[str, Any]], detail_rows: List[Dict[str, Any]]) -> Tuple[int, int]:
    output_by_id = {str(item.get("sample_id") or ""): item for item in outputs if isinstance(item, dict)}
    success = failures = 0
    for detail in detail_rows:
        if detail.get("status") == "error":
            continue
        sample = output_by_id.get(str(detail.get("sample_id") or ""))
        if not sample or not isinstance(sample.get("qa"), list):
            continue
        qa_index = int(detail["qa_index"])
        qa = sample["qa"][qa_index]
        try:
            prediction = answer_with_reader(args, str(detail.get("question") or ""), detail.get("category"), str(detail.get("recall_context") or ""))
            success += 1
            detail["reader_status"] = "ok"
        except Exception as exc:
            failures += 1
            prediction = NO_INFORMATION_ANSWER
            detail["reader_status"] = "error"
            detail["reader_error"] = str(exc)
            logging.exception("Reader failed for LoCoMo sample=%s qa_index=%s", detail.get("sample_id"), qa_index)
        qa[args.prediction_key] = prediction
        qa["hypothesis"] = prediction
        qa[f"{args.prediction_key}_context_text"] = str(detail.get("recall_context") or "")
        detail[args.prediction_key] = prediction
        detail["hypothesis"] = prediction
    return success, failures


def write_json(path: Path, value: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(value), ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    resolve_runtime_args(args)
    output_path, detail_path = resolve_output_paths(args)
    args.state_dir = args.state_dir.expanduser().resolve()
    args.log_path = args.log_path.expanduser().resolve()
    configure_logging(args.log_path, args.log_level, args.manager_log_level)
    remove_existing_outputs(output_path, detail_path, args.state_dir, bool(args.overwrite))
    args.state_dir.mkdir(parents=True, exist_ok=True)
    samples = filter_samples(load_dataset(args.input), args)
    if not samples:
        raise RuntimeError("No LoCoMo samples selected")
    (
        memory_runtime_config,
        memory_manager_config,
        llm_config,
        embedding_config,
    ) = prepare_runtime_configs(args)
    outputs_by_index: Dict[int, Dict[str, Any]] = {}
    details_by_index: Dict[int, List[Dict[str, Any]]] = {}
    errors: Dict[int, Dict[str, Any]] = {}
    workers = max(1, int(args.workers))
    payloads = [
        (
            index,
            len(samples),
            sample,
            args,
            memory_runtime_config,
            memory_manager_config,
            llm_config,
            embedding_config,
        )
        for index, sample in enumerate(samples, 1)
    ]
    if workers > 1:
        logging.info("Building LoCoMo memories with %s worker processes; reader answers run in main process", workers)
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(sample_worker, payload): payload for payload in payloads}
            for future in as_completed(futures):
                index, _total, sample, *_rest = futures[future]
                try:
                    result_index, output, details = future.result()
                    outputs_by_index[result_index], details_by_index[result_index] = output, details
                except Exception as exc:
                    errors[index] = {"sample_id": str(sample.get("sample_id") or ""), "status": "error", "stage": "memory_context", "error": str(exc)}
                    logging.exception("Failed LoCoMo sample %s", sample.get("sample_id"))
    else:
        for payload in payloads:
            index, total, sample, *_rest = payload
            logging.info("[%s/%s] Running LoCoMo sample %s", index, total, sample.get("sample_id"))
            try:
                output, details = build_sample_memory_context(
                    args,
                    sample,
                    memory_runtime_config,
                    memory_manager_config,
                    llm_config,
                    embedding_config,
                )
                outputs_by_index[index], details_by_index[index] = output, details
                logging.info(
                    "[%s/%s] Finished LoCoMo sample %s: facts=%s states=%s qa_rows=%s",
                    index,
                    total,
                    sample.get("sample_id"),
                    output["_memory_eval"]["db_counts"]["facts"],
                    output["_memory_eval"]["db_counts"]["states"],
                    len(details),
                )
            except Exception as exc:
                errors[index] = {"sample_id": str(sample.get("sample_id") or ""), "status": "error", "stage": "memory_context", "error": str(exc)}
                logging.exception("Failed LoCoMo sample %s", sample.get("sample_id"))
    outputs: List[Dict[str, Any]] = []
    detail_rows: List[Dict[str, Any]] = []
    for index, sample in enumerate(samples, 1):
        if index in errors:
            outputs.append({"sample_id": str(sample.get("sample_id") or ""), "qa": sample.get("qa") or [], "_memory_eval": errors[index]})
            detail_rows.append(errors[index])
        else:
            outputs.append(outputs_by_index[index])
            detail_rows.extend(details_by_index.get(index, []))
    reader_success, reader_failures = generate_reader_answers(args, outputs, detail_rows)
    write_json(output_path, outputs)
    write_json(detail_path, detail_rows)
    summary = {"input": str(args.input), "output": str(output_path), "detail_output": str(detail_path), "state_dir": str(args.state_dir), "samples_requested": len(samples), "samples_succeeded": len(samples) - len(errors), "samples_failed": len(errors), "reader_answers_succeeded": reader_success, "reader_answers_failed": reader_failures, "workers": workers, "recall_mode": args.recall_mode, "fact_extraction_interval": args.fact_extraction_interval}
    logging.info("LoCoMo evaluation complete: %s", json.dumps(summary, ensure_ascii=False))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
