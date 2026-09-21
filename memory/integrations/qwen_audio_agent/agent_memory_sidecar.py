#!/usr/bin/env python3
"""JSONL bridge from qwen-audio-agent to :class:`MemoryRuntime`.

The bridge intentionally receives completed Realtime transcripts instead of
audio. Qwen's Realtime provider owns VAD, ASR, interruption, and final-turn
semantics; this process owns only durable memory extraction and recall.

Each Gateway ``ownerId`` is mapped to an independent SQLite database below
``--state-dir``. This is required because the current agent-memory schema does
not include an owner column.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from memory.config import split_memory_config
from memory.memory_runtime import MemoryRuntime


_ENV_REF_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_SENSITIVE = re.compile(
    r"(?:api[_ -]?key|secret|token|password|passwd|credential|"
    r"验证码|密码|密钥|令牌|证件号|身份证|详细住址|病史|病历|诊断|用药|"
    r"\bsk-[a-z0-9_-]+|\b\d{11,19}\b)",
    re.IGNORECASE,
)
_MAX_SEEN_IDS = 20_000


def _clean(value: Any, limit: int = 8_000) -> str:
    return str(value or "").replace("\0", "").strip()[:limit]


def _owner_key(owner_id: str) -> str:
    return hashlib.sha256(owner_id.encode("utf-8")).hexdigest()


def _normalize_turn_timestamp(value: Any) -> Optional[str]:
    """Normalize Gateway epoch timestamps for MemoryRuntime's text schema."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        seconds = float(value)
        if seconds > 10_000_000_000:
            seconds /= 1_000.0
        try:
            return datetime.fromtimestamp(seconds).astimezone().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        except (OSError, OverflowError, ValueError):
            return None
    text = _clean(value, 120)
    return text or None


def _expand_env_refs(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_env_refs(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env_refs(item) for item in value]
    if isinstance(value, str):
        return _ENV_REF_RE.sub(lambda match: os.getenv(match.group(1), ""), value)
    return value


def _load_config(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - dependency is project-owned
        raise RuntimeError("PyYAML is required to load agent_memory config.yaml") from exc
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError("agent_memory config.yaml must contain a mapping")
    return _expand_env_refs(value)


def _message_key(owner_id: str, session_id: str, message: Dict[str, Any]) -> str:
    source = _clean(message.get("id"), 240) or _clean(message.get("turnId"), 240)
    if not source:
        source = hashlib.sha256(_clean(message.get("content")).encode("utf-8")).hexdigest()
    return hashlib.sha256(
        f"{owner_id}\0{session_id}\0{source}".encode("utf-8")
    ).hexdigest()


@dataclass
class OwnerRuntime:
    runtime: MemoryRuntime
    seen_path: Path
    seen_order: List[str] = field(default_factory=list)
    seen: set[str] = field(default_factory=set)

    def remember(self, keys: Iterable[str]) -> None:
        for key in keys:
            if key in self.seen:
                continue
            self.seen.add(key)
            self.seen_order.append(key)
        if len(self.seen_order) > _MAX_SEEN_IDS:
            self.seen_order = self.seen_order[-_MAX_SEEN_IDS:]
            self.seen = set(self.seen_order)
        temporary = self.seen_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.seen_order), encoding="utf-8")
        temporary.replace(self.seen_path)


class AgentMemoryRuntime:
    """Own isolated MemoryRuntime instances and expose JSONL-safe methods."""

    def __init__(
        self,
        *,
        state_dir: Path,
        memory_runtime_config: Dict[str, Any],
        memory_manager_config: Dict[str, Any],
        logger: logging.Logger,
    ) -> None:
        self._state_dir = state_dir.resolve()
        self._owner_dir = self._state_dir / "owners"
        self._owner_dir.mkdir(parents=True, exist_ok=True)
        self._runtime_config = dict(memory_runtime_config)
        self._manager_config = dict(memory_manager_config)
        self._logger = logger
        self._owners: Dict[str, OwnerRuntime] = {}

    def _owner_runtime(self, owner_id: str) -> OwnerRuntime:
        owner = _clean(owner_id, 240)
        if not owner:
            raise ValueError("ownerId must be non-empty")
        key = _owner_key(owner)
        existing = self._owners.get(key)
        if existing is not None:
            return existing
        owner_dir = self._owner_dir / key
        owner_dir.mkdir(parents=True, exist_ok=True)
        seen_path = owner_dir / "observed-message-ids.json"
        seen_order: List[str] = []
        if seen_path.exists():
            try:
                parsed = json.loads(seen_path.read_text(encoding="utf-8"))
                if isinstance(parsed, list):
                    seen_order = [str(item) for item in parsed[-_MAX_SEEN_IDS:]]
            except (OSError, ValueError, TypeError):
                self._logger.warning("Ignoring invalid observed-message index owner=%s", key)
        runtime = MemoryRuntime(
            db_path=owner_dir / "memory.db",
            memory_runtime_config=self._runtime_config,
            memory_manager_config=self._manager_config,
            logger=self._logger.getChild(f"owner.{key[:12]}"),
        )
        created = OwnerRuntime(
            runtime=runtime,
            seen_path=seen_path,
            seen_order=seen_order,
            seen=set(seen_order),
        )
        self._owners[key] = created
        return created

    @staticmethod
    def _assistant_text_by_turn(messages: List[Dict[str, Any]]) -> Dict[str, str]:
        values: Dict[str, List[str]] = {}
        for message in messages:
            if str(message.get("role") or "") != "assistant":
                continue
            turn_id = _clean(message.get("turnId"), 240)
            content = _clean(message.get("content"), 8_000)
            if turn_id and content and not _SENSITIVE.search(content):
                values.setdefault(turn_id, []).append(content)
        return {
            turn_id: "\n".join(dict.fromkeys(parts))
            for turn_id, parts in values.items()
        }

    def observe(self, params: Dict[str, Any]) -> Dict[str, Any]:
        owner_id = _clean(params.get("ownerId"), 240)
        session_id = _clean(params.get("sessionId"), 240)
        holder = self._owner_runtime(owner_id)
        owner_key = _owner_key(owner_id)
        messages = [
            item for item in params.get("messages") or [] if isinstance(item, dict)
        ]
        assistants = self._assistant_text_by_turn(messages)
        accepted_keys: List[str] = []
        skipped_sensitive = 0
        for message in messages:
            if str(message.get("role") or "") != "user":
                continue
            content = _clean(message.get("content"), 8_000)
            if not content:
                continue
            if _SENSITIVE.search(content):
                skipped_sensitive += 1
                continue
            key = _message_key(owner_id, session_id, message)
            if key in holder.seen:
                continue
            turn_id = _clean(message.get("turnId"), 240)
            report = holder.runtime.accept_single_interaction_turn(
                content,
                assistants.get(turn_id, ""),
                tags=["qwen-audio-agent"],
                turn_timestamp=_normalize_turn_timestamp(message.get("createdAt")),
            )
            if report.get("reason") == "memory_disabled":
                raise RuntimeError("agent_memory is disabled")
            accepted_keys.append(key)
        if accepted_keys:
            holder.remember(accepted_keys)
        result = {
            "observed": bool(accepted_keys),
            "messages": len(accepted_keys),
            "skippedSensitive": skipped_sensitive,
        }
        self._logger.info(
            "observe owner=%s session=%s input_messages=%s accepted_users=%s "
            "skipped_sensitive=%s",
            owner_key[:12],
            session_id[:80] or "[none]",
            len(messages),
            result["messages"],
            skipped_sensitive,
        )
        return result

    def finalize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Flush memory input and optionally close its semantic episode.

        ``checkpoint`` is for a transport-level close: it may flush and
        reflect, but never treats the reconnect as a conversation boundary.
        ``session_end`` is reserved for an explicit user-created new session;
        it submits store, episode summary, and reflection in that FIFO order.
        """
        owner_id = _clean(params.get("ownerId"), 240)
        session_id = _clean(params.get("sessionId"), 240)
        boundary = _clean(params.get("boundary"), 80).lower() or "checkpoint"
        if boundary not in {"checkpoint", "session_end"}:
            raise ValueError("boundary must be checkpoint or session_end")
        if boundary == "session_end" and not session_id:
            raise ValueError("sessionId must be non-empty for session_end")
        holder = self._owner_runtime(owner_id)
        input_flushed = holder.runtime.flush_pending_memory_inputs(
            evaluate_episode_summary=boundary != "session_end",
        )
        episode = None
        if boundary == "session_end":
            episode = holder.runtime.trigger_memory_episode_summary(
                reason=f"qwen_audio_agent_session_closed:{session_id}",
                source_type="assistant_wakeup",
                tags=["qwen-audio-agent", "session-close"],
            )
        # For an explicit session end this always queues after the summary;
        # for a checkpoint it preserves the existing best-effort reflect.
        reflect = (
            holder.runtime.trigger_memory_reflect()
            if boundary == "session_end" or input_flushed
            else None
        )
        result = {
            "finalized": boundary == "session_end" and bool((episode or {}).get("queued")),
            "boundary": boundary,
            "sessionId": session_id,
            "episodeSummaryQueued": bool((episode or {}).get("queued")),
            "inputFlushed": bool(input_flushed),
            "reflectQueued": bool((reflect or {}).get("queued")),
            "episode": episode,
            "reflect": reflect,
        }
        self._logger.info(
            "finalize owner=%s boundary=%s session=%s input_flushed=%s "
            "episode_summary_queued=%s reflect_queued=%s reason=%s",
            _owner_key(owner_id)[:12],
            boundary,
            session_id[:80],
            result["inputFlushed"],
            result["episodeSummaryQueued"],
            result["reflectQueued"],
            (episode or {}).get("reason") or "",
        )
        return result

    def recall(self, params: Dict[str, Any]) -> Dict[str, Any]:
        query = _clean(params.get("query"), 2_000)
        if not query:
            return {"status": "empty", "memory_context": ""}
        holder = self._owner_runtime(_clean(params.get("ownerId"), 240))
        tags = params.get("tags")
        result = holder.runtime.trigger_memory_recall(
            query,
            tags=[_clean(tag, 120) for tag in tags] if isinstance(tags, list) else None,
            time_end=_clean(params.get("timeEnd"), 120) or None,
            prompt_language=_clean(params.get("promptLanguage"), 16) or None,
        )
        self._logger.info(
            "recall owner=%s query_chars=%s status=%s context_chars=%s",
            _owner_key(_clean(params.get("ownerId"), 240))[:12],
            len(query),
            result.get("status") or "unknown",
            len(str(result.get("memory_context") or "")),
        )
        return result

    def health(self, _params: Dict[str, Any]) -> Dict[str, Any]:
        return {"ok": True, "activeOwners": len(self._owners)}

    def close(self, _params: Dict[str, Any]) -> Dict[str, Any]:
        failures = 0
        for holder in list(self._owners.values()):
            try:
                holder.runtime.close(timeout=None)
            except Exception:  # pragma: no cover - close must continue per owner
                failures += 1
                self._logger.exception("Failed to close owner memory runtime")
        self._owners.clear()
        result = {"closed": failures == 0, "failedOwners": failures}
        self._logger.info("close closed=%s failed_owners=%s", result["closed"], failures)
        return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-dir", required=True, type=Path)
    parser.add_argument("--config", default=REPO_ROOT / "config.yaml", type=Path)
    parser.add_argument(
        "--log-path",
        type=Path,
        help="Defaults to <state-dir>/agent-memory-sidecar.log.",
    )
    args = parser.parse_args()
    state_dir = args.state_dir.expanduser().resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    log_path = (
        args.log_path.expanduser().resolve()
        if args.log_path
        else state_dir / "agent-memory-sidecar.log"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("agent_memory.qwen_sidecar")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(formatter)
    logger.addHandler(stderr_handler)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    config = _load_config(args.config.expanduser().resolve())
    runtime_config, manager_config = split_memory_config(config)
    runtime = AgentMemoryRuntime(
        state_dir=state_dir,
        memory_runtime_config=runtime_config,
        memory_manager_config=manager_config,
        logger=logger,
    )
    logger.info("sidecar started state_dir=%s config=%s log_path=%s", state_dir, args.config, log_path)
    methods = {
        "observe": runtime.observe,
        "finalize": runtime.finalize,
        "recall": runtime.recall,
        "health": runtime.health,
        "close": runtime.close,
    }
    for line in sys.stdin:
        request: Optional[Dict[str, Any]] = None
        try:
            parsed = json.loads(line)
            if not isinstance(parsed, dict):
                raise ValueError("request must be a JSON object")
            request = parsed
            method = _clean(request.get("method"), 80)
            if method not in methods:
                raise ValueError(f"unsupported method: {method}")
            response = {"id": request.get("id"), "result": methods[method](request.get("params") or {})}
        except Exception as exc:
            logger.exception(
                "request failed method=%s",
                _clean((request or {}).get("method"), 80) or "[unknown]",
            )
            response = {
                "id": request.get("id") if request else None,
                "error": f"{type(exc).__name__}: {exc}",
            }
        print(json.dumps(response, ensure_ascii=False, default=str), flush=True)
        if request and request.get("method") == "close":
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
