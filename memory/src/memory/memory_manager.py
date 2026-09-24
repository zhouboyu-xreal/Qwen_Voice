#!/usr/bin/env python3
"""Unified memory manager inspired by MemPalace.

The public surface mirrors the current project's `MemoryNodeManager`, but the
internal model is deliberately unified:

1. Submitted source segments are organized into `memory_episodes`.
2. Extracted evidence becomes narrative `memory_facts`.
3. Traceable explicit and inductive propositions live in entity claims.
4. The legacy actionable-item projection is temporarily disabled while the
   intent and work-item layer is redesigned.

"""

from __future__ import annotations

import json
import logging
import math
import os
import queue
import re
import threading
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import requests

try:
    import jieba
except ImportError:  # pragma: no cover - exercised only in minimal installs
    jieba = None

from .embedding_client import EmbeddingClient
from .memory_database import SessionDB
from .prompts_en import (
    DERIVED_ENTITY_CLAIM_EXTRACTION_PROMPT_EN,
    ENTITY_CLAIM_RECONCILIATION_PROMPT_EN,
    EXPLICIT_ENTITY_CLAIM_EXTRACTION_PROMPT_EN,
    INDUCTIVE_ENTITY_CLAIM_EXTRACTION_PROMPT_EN,
    INTENT_EXTRACTION_PROMPT_EN,
    INTENT_RECONCILIATION_PROMPT_EN,
    EPISODE_SUMMARY_PROMPT_EN,
    MEMORY_RETRIEVED_FORMAT_PROMPT_EN,
    MEMORY_RETRIEVED_SECTION_SPECS_EN,
    RECALL_QUERY_ANALYSIS_PROMPT_EN,
    UNIFIED_MEMORY_EXTRACTION_PROMPT_EN,
)
from .prompts_zh import (
    DERIVED_ENTITY_CLAIM_EXTRACTION_PROMPT_ZH,
    ENTITY_CLAIM_RECONCILIATION_PROMPT_ZH,
    EXPLICIT_ENTITY_CLAIM_EXTRACTION_PROMPT_ZH,
    INDUCTIVE_ENTITY_CLAIM_EXTRACTION_PROMPT_ZH,
    INTENT_EXTRACTION_PROMPT_ZH,
    INTENT_RECONCILIATION_PROMPT_ZH,
    EPISODE_SUMMARY_PROMPT_ZH,
    MEMORY_RETRIEVED_FORMAT_PROMPT_ZH,
    MEMORY_RETRIEVED_SECTION_SPECS_ZH,
    RECALL_QUERY_ANALYSIS_PROMPT_ZH,
    UNIFIED_MEMORY_EXTRACTION_PROMPT_ZH,
)
from .utils import _cal_embedding_cosine_similarity

RecallTimeBounds = Optional[Tuple[Optional[str], Optional[str]]]

DEFAULT_LLM_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_LLM_MODEL = "deepseek-v4-flash"

_STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with",
    "about", "what", "which", "where", "when", "who", "why", "how", "did",
    "do", "does", "i", "me", "my", "you", "your", "we", "our", "is", "are",
    "was", "were", "be", "been", "being", "can", "could", "would", "should",
    "that", "this", "it", "as", "at", "by", "from", "have", "had", "has",
}

_COURTESY_PATTERNS = (
    "希望这个方法能帮到",
    "希望这能帮到",
    "希望对你有帮助",
    "希望对您有帮助",
    "有其他问题",
    "继续沟通",
    "随时告诉我",
    "不客气",
    "别客气",
    "很高兴能帮",
    "祝你",
    "祝您",
    "hope this helps",
    "hope that helps",
    "let me know if",
    "feel free to ask",
    "happy to help",
    "you are welcome",
    "you're welcome",
)

_ORDINARY_TIME_ENTITY_PATTERNS = (
    r"^(今天|昨天|前天|明天|后天|上周|本周|下周|上个月|这个月|下个月|最近|近期)$",
    r"^最近\d+(天|周|个月|月|年)$",
    r"^过去\d+(天|周|个月|月|年)$",
    r"^\d{4}[-/.]\d{1,2}[-/.]\d{1,2}$",
    r"^\d{1,2}:\d{2}(?::\d{2})?$",
    r"^\d+(分钟|小时|天|周|个月|月|年)$",
    r"^(today|yesterday|tomorrow|last week|this week|next week|last month|this month|next month|recently|lately)$",
    r"^(last|past|previous|next)\s+\d+\s+(day|days|week|weeks|month|months|year|years)$",
    r"^\d{4}[-/.]\d{1,2}[-/.]\d{1,2}$",
)

_ATTRIBUTE_ONLY_ENTITY_PATTERNS = (
    r"^(低|高|强|弱|轻|重|小|大|快|慢|短|长|稳定|灵活|固定|频繁|高频|低频|长期|短期).{0,8}$",
    r"^(low|high|strong|weak|light|heavy|fast|slow|short|long|stable|flexible|fixed|frequent)\s+[\w -]{0,24}$",
)

# These labels identify conversational roles rather than concrete entities.
# They may still be useful as diagnostics, but a role match alone must not be
# treated as an entity strong anchor during recall.
_LOW_VALUE_ENTITY_ALIASES = {
    "user": {
        "user", "the user", "用户", "我", "我自己", "me", "myself",
    },
    "assistant": {
        "assistant", "the assistant", "助手", "ai", "人工智能", "bot",
        "机器人", "agent",
    },
    "system": {
        "system", "the system", "系统",
    },
    "speaker": {
        "speaker", "说话人", "unknown", "unknown_speaker",
        "speaker_1", "speaker_2",
    },
}


def _now_text() -> str:
    return datetime.now().astimezone().isoformat()


def _to_timestamp_text(value: Any) -> str:
    if isinstance(value, datetime):
        # The LongMemEval scripts pass time filters as "YYYY-MM-DD HH:MM:SS".
        # Keep the same sortable format so SQLite string range filters work.
        return value.strftime("%Y-%m-%d %H:%M:%S")
    return str(value or "").strip()


def _compact_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


class MemoryOperationReporter:
    """Collect asynchronous memory operation results for benchmark callers."""

    def __init__(self, *, recent_task_limit: int = 200) -> None:
        self._lock = threading.Lock()
        self._task_seq = 0
        self._recent_task_limit = max(1, int(recent_task_limit or 200))
        self._counts: Dict[str, Dict[str, Any]] = {}
        self._latest_reports: Dict[str, Dict[str, Any]] = {}
        self._recent_tasks: List[Dict[str, Any]] = []

    @staticmethod
    def _empty_counts() -> Dict[str, Any]:
        return {
            "submitted": 0,
            "completed": 0,
            "succeeded": 0,
            "failed": 0,
            "rejected": 0,
            "inflight": 0,
            "total_elapsed_ms": 0.0,
        }

    def next_task_id(self, operation_type: str) -> str:
        clean_type = str(operation_type or "memory_task").strip() or "memory_task"
        with self._lock:
            self._task_seq += 1
            return f"{clean_type}-{self._task_seq}"

    def on_task_submitted(
        self,
        *,
        operation_type: str,
        task_id: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        safe_payload = self._json_compatible(payload or {})
        with self._lock:
            counts = self._counts.setdefault(operation_type, self._empty_counts())
            counts["submitted"] += 1
            counts["inflight"] += 1
            self._append_recent_locked({
                "event": "submitted",
                "operation_type": operation_type,
                "task_id": task_id,
                "payload": safe_payload,
                "timestamp": _now_text(),
            })

    def on_task_rejected(
        self,
        *,
        operation_type: str,
        task_id: str,
        reason: str,
    ) -> None:
        report = {
            "accepted": False,
            "status": "rejected",
            "reason": reason,
            "task_id": task_id,
        }
        with self._lock:
            counts = self._counts.setdefault(operation_type, self._empty_counts())
            counts["rejected"] += 1
            self._latest_reports[operation_type] = dict(report)
            self._append_recent_locked({
                "event": "rejected",
                "operation_type": operation_type,
                "task_id": task_id,
                "reason": reason,
                "timestamp": _now_text(),
            })

    def on_task_finished(
        self,
        *,
        operation_type: str,
        task_id: str,
        started_at: float,
        result: Any = None,
        error: Optional[BaseException] = None,
    ) -> None:
        elapsed_ms = round((time.monotonic() - started_at) * 1000, 2)
        succeeded = error is None and self._operation_succeeded(operation_type, result)
        report = self._operation_result_report(
            operation_type=operation_type,
            task_id=task_id,
            result=result,
            error=error,
            succeeded=succeeded,
            elapsed_ms=elapsed_ms,
        )
        with self._lock:
            counts = self._counts.setdefault(operation_type, self._empty_counts())
            counts["completed"] += 1
            counts["inflight"] = max(0, int(counts.get("inflight") or 0) - 1)
            counts["total_elapsed_ms"] = round(
                float(counts.get("total_elapsed_ms") or 0.0) + elapsed_ms,
                2,
            )
            counts["succeeded" if succeeded else "failed"] += 1
            self._latest_reports[operation_type] = dict(report)
            recent_event = {
                "event": "finished",
                "operation_type": operation_type,
                "task_id": task_id,
                "status": report.get("status"),
                "succeeded": succeeded,
                "elapsed_ms": elapsed_ms,
                "timestamp": _now_text(),
            }
            if error is not None:
                recent_event["error"] = str(error)
                recent_event["error_type"] = type(error).__name__
            self._append_recent_locked(recent_event)

    def on_recall_finished(self, report: Dict[str, Any]) -> None:
        elapsed_ms = float(report.get("elapsed_ms") or 0.0)
        status = str(report.get("status") or "").strip().lower()
        succeeded = status not in {"error", "failed"}
        recall_report = {
            key: value
            for key, value in report.items()
            if key != "memory_context"
        }
        with self._lock:
            counts = self._counts.setdefault("recall", self._empty_counts())
            counts["submitted"] += 1
            counts["completed"] += 1
            counts["total_elapsed_ms"] = round(
                float(counts.get("total_elapsed_ms") or 0.0) + elapsed_ms,
                2,
            )
            counts["succeeded" if succeeded else "failed"] += 1
            self._latest_reports["recall"] = recall_report
            self._append_recent_locked({
                "event": "finished",
                "operation_type": "recall",
                "task_id": f"recall-{counts['completed']}",
                "status": recall_report.get("status"),
                "actual_recall_mode": recall_report.get("actual_recall_mode"),
                "elapsed_ms": elapsed_ms,
                "timestamp": _now_text(),
            })

    def operation_report(self, operation_type: str) -> Dict[str, Any]:
        with self._lock:
            counts = dict(
                self._counts.get(operation_type) or self._empty_counts()
            )
            latest = self._latest_reports.get(operation_type)
        if latest:
            counts["latest_report"] = dict(latest)
        return counts

    def latest_report(self, operation_type: str) -> Dict[str, Any]:
        with self._lock:
            return dict(self._latest_reports.get(operation_type) or {})

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "counts": {
                    key: dict(value)
                    for key, value in self._counts.items()
                },
                "latest_reports": {
                    key: dict(value)
                    for key, value in self._latest_reports.items()
                },
                "recent_tasks": [dict(item) for item in self._recent_tasks],
            }

    def _append_recent_locked(self, event: Dict[str, Any]) -> None:
        self._recent_tasks.append(event)
        if len(self._recent_tasks) > self._recent_task_limit:
            del self._recent_tasks[: len(self._recent_tasks) - self._recent_task_limit]

    @classmethod
    def _json_compatible(cls, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, datetime):
            return _to_timestamp_text(value)
        if isinstance(value, dict):
            return {
                str(key): cls._json_compatible(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [cls._json_compatible(item) for item in value]
        return str(value)

    @staticmethod
    def _operation_succeeded(operation_type: str, result: Any) -> bool:
        failure_statuses = {
            "failed",
            "skipped",
            "error",
            "queue_rejected",
        }
        if operation_type == "memory_store":
            if not isinstance(result, dict):
                return bool(result)
            return (
                str(result.get("status") or "ok").strip().lower()
                not in failure_statuses
            )
        if operation_type in {
            "memory_episode_summary",
            "memory_entity_claim_update",
            "memory_future_commitment_update",
        }:
            if not isinstance(result, dict):
                return False
            return (
                str(result.get("status") or "ok").strip().lower()
                not in failure_statuses
            )
        return True

    @staticmethod
    def _operation_result_report(
        *,
        operation_type: str,
        task_id: str,
        result: Any,
        error: Optional[BaseException],
        succeeded: bool,
        elapsed_ms: float,
    ) -> Dict[str, Any]:
        if isinstance(result, dict):
            report = dict(result)
        else:
            report = {"result": result}
        result_status = str(report.get("status") or "").strip().lower()
        report.update({
            "accepted": True,
            "task_id": task_id,
            "operation_type": operation_type,
            "status": (
                "failed"
                if error is not None or not succeeded
                else (result_status or "ok")
            ),
            "total_elapsed_ms": elapsed_ms,
        })
        if operation_type in {"memory_store", "memory_episode_summary"}:
            report["stored"] = bool(succeeded and error is None)
        if error is not None:
            report["error_type"] = type(error).__name__
            report["error"] = str(error)
        return report


class MemoryNodeManager:
    """Compatibility manager backed by a unified index-first memory line."""

    def __init__(
        self,
        db: SessionDB,
        *,
        embedding_config: Optional[Dict[str, Any]] = None,
        memory_manager_config: Optional[Dict[str, Any]] = None,
        operation_reporter: Optional[MemoryOperationReporter] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._db = db
        self._logger = logger or logging.getLogger(__name__)
        self._operation_reporter = operation_reporter or MemoryOperationReporter()
        self._memory_cfg = dict(memory_manager_config or {})
        configured_embedding = self._memory_cfg.get("embedding")
        self._embedding_cfg = dict(
            embedding_config
            or (configured_embedding if isinstance(configured_embedding, dict) else {})
        )
        configured_llm = self._memory_cfg.get("llm")
        self._llm_cfg = dict(
            configured_llm if isinstance(configured_llm, dict) else {}
        )
        self._llm_model = str(self._llm_cfg.get("llm_name") or DEFAULT_LLM_MODEL)
        self._llm_base_url = self._normalize_llm_base_url(
            str(self._llm_cfg.get("llm_base_url") or DEFAULT_LLM_BASE_URL)
        )
        self._llm_api_key = self._resolve_env(self._llm_cfg.get("llm_api_key"))
        self._llm_timeout = int(self._llm_cfg.get("llm_timeout", 120) or 120)
        self._llm_json_mode = self._config_bool(
            self._llm_cfg.get("llm_json_mode", True),
            True,
        )
        self._llm_thinking = str(
            self._llm_cfg.get("llm_thinking", "disabled") or "disabled"
        )
        self._memory_prompt_language = str(
            self._memory_cfg.get("memory_prompt_language_mode")
            or self._memory_cfg.get("prompt_language_mode")
            or "source"
        )
        self._memory_enabled = bool(self._memory_cfg.get("memory_enabled", True))
        self._enable_memory_entity_claim_update = self._config_bool(
            self._memory_cfg.get("enable_memory_entity_claim_update", True),
            True,
        )
        self._enable_memory_future_commitment_update = self._config_bool(
            self._memory_cfg.get("enable_memory_future_commitment_update", True),
            True,
        )
        self._world_owner_entity_name = _compact_whitespace(
            self._memory_cfg.get("world_owner_entity_name") or "用户"
        ) or "用户"
        self._entity_claim_explicit_min_confidence = self._clamp_float(
            self._memory_cfg.get("entity_claim_explicit_min_confidence"),
            0.0,
            1.0,
            0.72,
        )
        self._entity_claim_derived_min_confidence = self._clamp_float(
            self._memory_cfg.get("entity_claim_derived_min_confidence"),
            0.0,
            1.0,
            0.75,
        )
        self._entity_claim_induction_min_support_facts = max(
            3,
            int(
                self._memory_cfg.get(
                    "entity_claim_induction_min_support_facts", 3,
                ) or 3
            ),
        )
        self._entity_claim_induction_min_episodes = max(
            3,
            int(self._memory_cfg.get("entity_claim_induction_min_episodes", 3) or 3),
        )
        self._entity_claim_induction_min_time_windows = max(
            2,
            int(self._memory_cfg.get("entity_claim_induction_min_time_windows", 2) or 2),
        )
        self._entity_claim_induction_evidence_limit = max(
            self._entity_claim_induction_min_support_facts,
            int(self._memory_cfg.get("entity_claim_induction_evidence_limit", 16) or 16),
        )
        self._initialize_recall_config()
        self._embedding_client: Optional[EmbeddingClient] = None
        self._task_queue_maxsize = max(
            1,
            int(self._memory_cfg.get("task_queue_maxsize", 100) or 100),
        )
        self._task_queue: queue.Queue[Dict[str, Any]] = queue.Queue(
            maxsize=self._task_queue_maxsize,
        )
        self._task_worker_thread: Optional[threading.Thread] = None
        self._task_worker_lock = threading.Lock()
        self._task_shutdown_event = threading.Event()
        self._memory_operation_lock = threading.RLock()
        self._task_completion_listener_lock = threading.Lock()
        self._task_completion_listeners: List[
            Callable[[Dict[str, Any]], None]
        ] = []

    def _initialize_recall_config(self) -> None:
        """Parse nested recall settings and keep legacy flat overrides working."""
        configured_recall = self._memory_cfg.get("recall")
        self._recall_cfg = dict(
            configured_recall if isinstance(configured_recall, dict) else {}
        )
        configured_stage1 = self._recall_cfg.get("recall_stage1")
        self._recall_stage1_cfg = dict(
            configured_stage1 if isinstance(configured_stage1, dict) else {}
        )
        configured_stage2 = self._recall_cfg.get("recall_stage2")
        self._recall_stage2_cfg = dict(
            configured_stage2 if isinstance(configured_stage2, dict) else {}
        )

        def recall_value(
            key: str,
            default: Any = None,
            *,
            stage: Optional[str] = None,
        ) -> Any:
            # Keep flat values as explicit overrides so existing callers that
            # patch ``memory_manager_config`` (for example CLI flags) retain
            # their current behavior while config.yaml uses the nested shape.
            if self._memory_cfg.get(key) is not None:
                return self._memory_cfg.get(key)
            stage_cfg = {
                "stage1": self._recall_stage1_cfg,
                "stage2": self._recall_stage2_cfg,
            }.get(stage, {})
            if key in stage_cfg and stage_cfg.get(key) is not None:
                return stage_cfg.get(key)
            if key in self._recall_cfg and self._recall_cfg.get(key) is not None:
                return self._recall_cfg.get(key)
            return default

        self._top_k = max(1, int(recall_value("recall_top_k", 8) or 8))
        self._recall_detailed_logging = self._config_bool(
            recall_value("recall_detailed_logging", False),
            False,
        )
        self._recall_budget = str(recall_value("recall_budget", "mid") or "mid")
        self._recall_mode = str(
            recall_value("recall_mode", "normal") or "normal"
        ).strip().lower()

        self._recall_stage1_entity_matched_score = self._clamp_float(
            recall_value(
                "recall_stage1_entity_matched_score",
                0.30,
                stage="stage1",
            ),
            0.0,
            1.0,
            0.30,
        )
        self._recall_stage1_topic_overlap_score_weight = self._clamp_float(
            recall_value(
                "recall_stage1_topic_overlap_score_weight",
                0.54,
                stage="stage1",
            ),
            0.0,
            1.0,
            0.54,
        )
        self._recall_stage1_keyword_overlap_score_weight = self._clamp_float(
            recall_value(
                "recall_stage1_keyword_overlap_score_weight",
                0.12,
                stage="stage1",
            ),
            0.0,
            1.0,
            0.12,
        )
        self._recall_stage1_time_score_weight = self._clamp_float(
            recall_value(
                "recall_stage1_time_score_weight",
                None,
                stage="stage1",
            ),
            0.0,
            1.0,
            0.12,
        )
        self._recall_stage1_contextual_time_score_multiplier = max(
            1.0,
            float(
                recall_value(
                    "recall_stage1_contextual_time_score_multiplier",
                    2.0,
                    stage="stage1",
                )
                or 2.0
            ),
        )
        self._recall_stage1_min_term_coverage = self._clamp_float(
            recall_value(
                "recall_stage1_evidence_profile_min_term_coverage",
                0.5,
                stage="stage1",
            ),
            0.0,
            1.0,
            0.5,
        )
        self._recall_stage1_episode_propagation_decay = self._clamp_float(
            recall_value(
                "recall_stage1_episode_propagation_decay",
                0.70,
                stage="stage1",
            ),
            0.0,
            1.0,
            0.70,
        )
        self._recall_stage1_association_propagation_decay = self._clamp_float(
            recall_value(
                "recall_stage1_association_propagation_decay",
                0.80,
                stage="stage1",
            ),
            0.0,
            1.0,
            0.80,
        )

        self._recall_stage2_fact_min_embedding_similarity = self._clamp_float(
            recall_value(
                "recall_stage2_fact_min_embedding_similarity",
                0.35,
                stage="stage2",
            ),
            0.0,
            1.0,
            0.35,
        )
        # The seed-retrieval floors are intentionally lower than the
        # thresholds required for a candidate to become direct evidence.
        self._recall_stage2_fact_strong_embedding_similarity = (
            self._clamp_float(
                recall_value(
                    "recall_stage2_fact_strong_embedding_similarity",
                    0.45,
                    stage="stage2",
                ),
                0.0,
                1.0,
                0.45,
            )
        )
        self._recall_stage2_strong_topic_pair_score = self._clamp_float(
            recall_value(
                "recall_stage2_strong_topic_pair_score",
                0.90,
                stage="stage2",
            ),
            0.0,
            1.0,
            0.90,
        )
        self._recall_stage2_embedding_score_weight = self._clamp_float(
            recall_value(
                "recall_stage2_embedding_score_weight",
                0.42,
                stage="stage2",
            ),
            0.0,
            1.0,
            0.42,
        )
        self._recall_stage2_topic_overlap_score_weight = self._clamp_float(
            recall_value(
                "recall_stage2_topic_overlap_score_weight",
                0.25,
                stage="stage2",
            ),
            0.0,
            1.0,
            0.25,
        )
        self._recall_stage2_keyword_match_score_weight = self._clamp_float(
            recall_value(
                "recall_stage2_keyword_match_score_weight",
                0.12,
                stage="stage2",
            ),
            0.0,
            1.0,
            0.12,
        )
        self._recall_stage2_bm25_score_weight = self._clamp_float(
            recall_value(
                "recall_stage2_bm25_score_weight",
                0.20,
                stage="stage2",
            ),
            0.0,
            1.0,
            0.20,
        )
        self._recall_stage2_entity_matched_score = self._clamp_float(
            recall_value(
                "recall_stage2_entity_matched_score",
                0.20,
                stage="stage2",
            ),
            0.0,
            1.0,
            0.20,
        )
        self._recall_stage2_time_score_weight = self._clamp_float(
            recall_value(
                "recall_stage2_time_score_weight",
                None,
                stage="stage2",
            ),
            0.0,
            1.0,
            0.12,
        )
        self._recall_stage2_contextual_time_score_multiplier = max(
            1.0,
            float(
                recall_value(
                    "recall_stage2_contextual_time_score_multiplier",
                    2.0,
                    stage="stage2",
                )
                or 2.0
            ),
        )
        self._recall_fact_time_score_half_life_seconds = max(
            1,
            int(recall_value("recall_fact_time_score_half_life_seconds", 604800) or 604800),
        )

        self._recall_context_char_budgets = {
            "low": max(
                1200,
                int(recall_value("recall_context_chars_low", 3200) or 3200),
            ),
            "mid": max(
                1800,
                int(recall_value("recall_context_chars_mid", 6000) or 6000),
            ),
            "high": max(
                2400,
                int(recall_value("recall_context_chars_high", 10000) or 10000),
            ),
        }
        shared_recall_context_budget = recall_value("recall_context_max_chars")
        if shared_recall_context_budget not in (None, ""):
            shared_budget = max(1200, int(shared_recall_context_budget or 0))
            self._recall_context_char_budgets = {
                key: shared_budget for key in self._recall_context_char_budgets
            }
        self._recall_entry_char_budgets = {
            "low": max(
                260,
                int(recall_value("recall_entry_chars_low", 520) or 520),
            ),
            "mid": max(
                320,
                int(recall_value("recall_entry_chars_mid", 760) or 760),
            ),
            "high": max(
                420,
                int(recall_value("recall_entry_chars_high", 1100) or 1100),
            ),
        }

    @staticmethod
    def _config_bool(value: Any, default: bool = False) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "y", "on", "enable", "enabled"}:
            return True
        if text in {"0", "false", "no", "n", "off", "disable", "disabled"}:
            return False
        return default

    @staticmethod
    def _resolve_env(value: Any) -> str:
        text = str(value or "").strip()
        match = re.fullmatch(r"\${([A-Za-z_][A-Za-z0-9_]*)}", text)
        if match:
            return os.environ.get(match.group(1), "").strip()
        return text

    @staticmethod
    def _normalize_llm_base_url(value: str) -> str:
        text = str(value or DEFAULT_LLM_BASE_URL).strip().rstrip("/")
        if text == "https://api.deepseek.com":
            return "https://api.deepseek.com/v1"
        return text or DEFAULT_LLM_BASE_URL

    # ── Runtime helpers used by benchmark scripts ───────────────────────

    def _ensure_embedding_client(self) -> bool:
        if self._embedding_client is None:
            self._embedding_client = EmbeddingClient(self._embedding_cfg)
        return True

    # ── Store path: raw segments -> episode -> facts -> index cards ──────

    @property
    def enabled(self) -> bool:
        return self._memory_enabled

    def set_logger(self, logger: Optional[logging.Logger]) -> None:
        """Set the logger used by memory store, entity-claim, and recall operations."""
        self._logger = logger or logging.getLogger(__name__)

    def set_embedding_client(self, embedding_client: Optional[EmbeddingClient]) -> None:
        """Share a pre-initialized embedding client with memory runtime callers."""
        self._embedding_client = embedding_client

    def register_task_completion_listener(
        self,
        listener: Callable[[Dict[str, Any]], None],
    ) -> None:
        """Register a non-blocking observer for completed asynchronous tasks.

        Listeners receive task metadata only after the manager's domain lock has
        been released.  They must not perform memory work synchronously; the
        runtime uses this hook only to update its local scheduling state and
        optionally enqueue follow-up tasks.
        """
        with self._task_completion_listener_lock:
            if listener not in self._task_completion_listeners:
                self._task_completion_listeners.append(listener)

    def unregister_task_completion_listener(
        self,
        listener: Callable[[Dict[str, Any]], None],
    ) -> None:
        """Remove a previously registered asynchronous task observer."""
        with self._task_completion_listener_lock:
            self._task_completion_listeners = [
                candidate
                for candidate in self._task_completion_listeners
                if candidate != listener
            ]

    def _emit_task_completion(self, event: Dict[str, Any]) -> None:
        """Notify observers without allowing an observer failure to stop work."""
        with self._task_completion_listener_lock:
            listeners = list(self._task_completion_listeners)
        for listener in listeners:
            try:
                listener(dict(event))
            except Exception:
                self._logger.exception(
                    "Memory task completion listener failed task_kind=%s task_id=%s",
                    event.get("task_kind"),
                    event.get("task_id"),
                )

    def _task_worker_loop(self) -> None:
        while not self._task_shutdown_event.is_set() or not self._task_queue.empty():
            try:
                task = self._task_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            task_kind = str(task.get("kind") or "")
            task_id = str(task.get("task_id") or "")
            started_at = float(task.get("started_at") or time.monotonic())
            result: Any = None
            error: Optional[BaseException] = None
            try:
                with self._memory_operation_lock:
                    if task_kind == "memory_store":
                        result = self._process_memory_store_task(**task["payload"])
                    elif task_kind == "memory_episode_summary":
                        result = self._process_memory_episode_summary_task(**task["payload"])
                    elif task_kind == "memory_entity_claim_update":
                        result = self._process_memory_entity_claim_update_task(**task["payload"])
                    elif task_kind == "memory_future_commitment_update":
                        result = self._process_memory_future_commitment_update_task(**task["payload"])
                    else:
                        raise ValueError(f"Unsupported memory async task: {task_kind}")
                self._operation_reporter.on_task_finished(
                    operation_type=task_kind,
                    task_id=task_id,
                    started_at=started_at,
                    result=result,
                )
            except Exception as exc:
                error = exc
                self._logger.exception("Async memory %s failed: %s", task.get("kind"), exc)
                self._operation_reporter.on_task_finished(
                    operation_type=task_kind,
                    task_id=task_id,
                    started_at=started_at,
                    error=exc,
                )
            finally:
                self._emit_task_completion({
                    "task_kind": task_kind,
                    "task_id": task_id,
                    "result": result,
                    "error": error,
                    "succeeded": error is None and self._operation_reporter._operation_succeeded(
                        task_kind,
                        result,
                    ),
                    "completion_context": dict(
                        task.get("completion_context") or {}
                    ),
                })
                self._task_queue.task_done()

    def _ensure_task_worker_locked(self) -> None:
        if self._task_worker_thread and self._task_worker_thread.is_alive():
            return
        self._task_worker_thread = threading.Thread(
            target=self._task_worker_loop,
            daemon=True,
            name="memory-node-worker",
        )
        self._task_worker_thread.start()

    def _submit_memory_task(
        self,
        *,
        task_kind: str,
        payload: Dict[str, Any],
        completion_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        task_id = self._operation_reporter.next_task_id(task_kind)
        with self._task_worker_lock:
            if self._task_shutdown_event.is_set():
                self._logger.warning("Memory worker is shut down; dropping %s task", task_kind)
                return self._reject_memory_task(
                    task_kind=task_kind,
                    task_id=task_id,
                    reason="worker_shutdown",
                )
            try:
                self._task_queue.put_nowait({
                    "kind": task_kind,
                    "payload": payload,
                    "task_id": task_id,
                    "started_at": time.monotonic(),
                    "completion_context": dict(completion_context or {}),
                })
            except queue.Full:
                self._logger.warning(
                    "Memory task queue is full; dropping %s (maxsize=%d)",
                    task_kind,
                    self._task_queue_maxsize,
                )
                return self._reject_memory_task(
                    task_kind=task_kind,
                    task_id=task_id,
                    reason="worker_queue_full",
                )
            self._operation_reporter.on_task_submitted(
                operation_type=task_kind,
                task_id=task_id,
                payload={
                    key: value
                    for key, value in payload.items()
                    if key != "raw_segments"
                },
            )
            self._ensure_task_worker_locked()
        return {
            "queued": True,
            "status": "queued",
            "task_id": task_id,
            "operation_type": task_kind,
        }

    def _reject_memory_task(
        self,
        *,
        task_kind: str,
        task_id: str,
        reason: str,
    ) -> Dict[str, Any]:
        self._operation_reporter.on_task_rejected(
            operation_type=task_kind,
            task_id=task_id,
            reason=reason,
        )
        return {
            "queued": False,
            "status": "rejected",
            "reason": reason,
            "task_id": task_id,
            "operation_type": task_kind,
        }

    def flush_task_queue(self, timeout: Optional[float] = None) -> bool:
        """Wait until all queued asynchronous memory tasks finish."""
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        while self._task_queue.unfinished_tasks:
            if deadline is not None and time.monotonic() >= deadline:
                return False
            time.sleep(0.01)
        return True

    def shutdown_task_worker(
        self,
        *,
        wait: bool = True,
        timeout: Optional[float] = None,
    ) -> bool:
        """Stop accepting tasks and optionally drain the memory worker.

        When ``wait`` is true, do not report shutdown as successful until every
        queued task has called ``task_done`` and the worker thread has exited.
        This is important because callers close the database immediately after
        shutdown; closing it while an entity-claim transaction is still running
        rolls back all state updates made by that transaction.
        """
        with self._task_worker_lock:
            self._task_shutdown_event.set()
            worker = self._task_worker_thread
        if not wait or worker is None:
            return not self._task_queue.unfinished_tasks

        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        while self._task_queue.unfinished_tasks:
            if not worker.is_alive():
                return False
            if deadline is not None and time.monotonic() >= deadline:
                return False
            time.sleep(0.01)

        remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
        worker.join(timeout=remaining)
        return not worker.is_alive() and not self._task_queue.unfinished_tasks

    def submit_memory_store_task(
        self,
        *,
        raw_segments: List[Dict[str, Any]],
        tags: List[str],
        prompt_language: str,
        completion_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Queue one normalized episode for ordered background storage."""
        if not self._memory_enabled or not raw_segments:
            reason = "memory_disabled" if not self._memory_enabled else "no_raw_segments"
            task_id = self._operation_reporter.next_task_id("memory_store")
            return self._reject_memory_task(
                task_kind="memory_store",
                task_id=task_id,
                reason=reason,
            )
        return self._submit_memory_task(
            task_kind="memory_store",
            payload={
                "raw_segments": raw_segments,
                "tags": tags,
                "prompt_language": prompt_language,
            },
            completion_context=completion_context,
        )

    def submit_memory_episode_summary_task(
        self,
        *,
        tags: List[str],
        prompt_language: str,
    ) -> Dict[str, Any]:
        """Queue episode summarization after preceding fact-store tasks."""
        if not self._memory_enabled:
            task_id = self._operation_reporter.next_task_id("memory_episode_summary")
            return self._reject_memory_task(
                task_kind="memory_episode_summary", task_id=task_id, reason="memory_disabled"
            )
        return self._submit_memory_task(
            task_kind="memory_episode_summary",
            payload={
                "tags": list(tags or []),
                "prompt_language": prompt_language,
            },
        )

    def _process_memory_episode_summary_task(
        self,
        *,
        tags: List[str],
        prompt_language: str,
    ) -> Dict[str, Any]:
        source_segments = self._db.get_unassigned_memory_source_segments(
            limit=240,
        )
        if not source_segments:
            return {
                "status": "empty",
                "reason": "no_unassigned_source_segments",
                "new_episode_count": 0,
                "fact_count": 0,
                "source_segment_count": 0,
            }
        source_started_at = _compact_whitespace(
            source_segments[0].get("started_at") or ""
        )
        source_ended_at = _compact_whitespace(
            source_segments[-1].get("ended_at")
            or source_segments[-1].get("started_at")
            or source_started_at
        )
        facts = self._db.get_unassigned_memory_facts_in_time_window(
            started_at=source_started_at,
            ended_at=source_ended_at,
            limit=80,
        )
        episode_info = self.generate_episode_from_source_segments(
            source_segments=source_segments,
            facts=facts,
            prompt_language=prompt_language,
        )
        if episode_info.get("status") != "ok":
            return episode_info
        with self._db.transaction():
            episode_id = self._db.insert_episode(
                episode_type="memory",
                title=episode_info["title"],
                summary=episode_info["summary"],
                participants=episode_info.get("participants") or [],
                started_at=episode_info.get("started_at") or _now_text(),
                ended_at=episode_info.get("ended_at") or episode_info.get("started_at") or _now_text(),
                canonical_topics=episode_info.get("canonical_topics") or [],
                entity_ids=episode_info.get("entity_ids") or [],
                metadata={
                    "tags": list(tags or []),
                    "fact_count": len(facts),
                    "source_segment_count": len(source_segments),
                    "generated_from_source_segments": True,
                    "evidence_fact_ids": list(episode_info.get("fact_ids") or []),
                },
            )
            self._upsert_episode_recall_document(
                episode_id=episode_id,
                title=episode_info["title"],
                summary=episode_info["summary"],
                participants=episode_info.get("participants") or [],
                entity_ids=episode_info.get("entity_ids") or [],
                canonical_topics=episode_info.get("canonical_topics") or [],
                started_at=episode_info.get("started_at") or _now_text(),
                ended_at=(
                    episode_info.get("ended_at")
                    or episode_info.get("started_at")
                    or _now_text()
                ),
            )
            topic_report = self._db.upsert_memory_topic_items(
                self._build_memory_topic_item_updates(
                    canonical_topics=episode_info.get("canonical_topics") or [],
                    episode_id=episode_id,
                )
            )
            attached = self._db.update_facts_episode_id(
                fact_ids=episode_info.get("fact_ids") or [], episode_id=episode_id,
            )
            attached_source_segment_rows = self._db.update_memory_source_segments_episode_id(
                source_segment_ids=episode_info.get("source_segment_ids") or [],
                episode_id=episode_id,
            )
        episode_info.update({
            "episode_id": episode_id,
            "fact_count": attached,
            # Source evidence is expanded from batch rows before episode
            # generation, so expose the logical segment count rather than
            # the number of physical persistence rows attached here.
            "source_segment_count": len(source_segments),
            "source_segment_row_count": attached_source_segment_rows,
            "new_episode_count": 1,
            "topic_items_created": int(topic_report.get("created_count", 0) or 0),
            "topic_items_updated": int(topic_report.get("updated_count", 0) or 0),
        })
        self._log_info("memory_store", "episode_summary_generated", {
            "episode_id": episode_id,
            "fact_count": attached,
            "source_segment_count": len(source_segments),
            "source_segment_row_count": attached_source_segment_rows,
            "title": episode_info.get("title") or "",
            "topic_items_created": episode_info.get("topic_items_created", 0),
            "topic_items_updated": episode_info.get("topic_items_updated", 0),
        })
        return episode_info

    def _process_memory_store_task(
        self,
        *,
        raw_segments: List[Dict[str, Any]],
        tags: List[str],
        prompt_language: str,
    ) -> Dict[str, Any]:
        store_started_at = time.monotonic()
        self._log_info("memory_store", "start", {
            "source_segment_count": len(raw_segments),
            "raw_segments": self._build_memory_segments_for_prompt(
                raw_segments,
                prompt_language=prompt_language,
            ),
        })
        if not raw_segments:
            elapsed_ms = round((time.monotonic() - store_started_at) * 1000, 2)
            self._log_info("memory_store", "finish", {
                "status": "skipped",
                "reason": "no_raw_segments",
                "total_elapsed_ms": elapsed_ms,
            })
            return {
                "status": "skipped",
                "reason": "no_raw_segments",
                "new_episode_count": 0,
                "new_fact_count": 0,
                "total_elapsed_ms": elapsed_ms,
            }
        with self._db.transaction():
            source_segment_ids = self._db.insert_memory_source_segments(
                segments=raw_segments,
            )
        extracted_info = self._extract_memory_fact_from_raw_segments(
            raw_segments,
            prompt_language=prompt_language,
        )
        facts = list(extracted_info.get("facts") or [])
        self._log_extracted_fact_info(facts=facts)
        with self._db.transaction():
            save_entity_info = self._store_extracted_memory_entities_into_db(
                participants=[],
                facts=facts,
            )
            save_fact_info = self._store_extracted_memory_facts_into_db(
                episode_id=None,
                facts=facts,
                tags=tags,
                episode_context_topics=None,
                entity_info=save_entity_info,
            )
            topic_report = self._db.upsert_memory_topic_items(
                save_fact_info.get("topic_item_updates") or []
            )
        report = {
            "status": "ok",
            "new_episode_count": 0,
            "new_fact_count": len(list(save_fact_info.get("fact_ids") or [])),
            "fact_ids": list(save_fact_info.get("fact_ids") or []),
            "entity_claim_signal_fact_count": int(
                save_fact_info.get("entity_claim_signal_fact_count") or 0
            ),
            "entity_claim_signal_count": int(
                save_fact_info.get("entity_claim_signal_count") or 0
            ),
            "future_commitment_signal_fact_count": int(
                save_fact_info.get("future_commitment_signal_fact_count") or 0
            ),
            "future_commitment_signal_count": int(
                save_fact_info.get("future_commitment_signal_count") or 0
            ),
            "source_segment_ids": source_segment_ids,
            "source_segment_count": len(raw_segments),
            "source_segment_row_count": len(source_segment_ids),
            "topic_items_created": int(topic_report.get("created_count", 0) or 0),
            "topic_items_updated": int(topic_report.get("updated_count", 0) or 0),
            "total_elapsed_ms": round((time.monotonic() - store_started_at) * 1000, 2),
        }
        self._log_info("memory_store", "finish", report)
        return report

    def _store_extracted_memory_entities_into_db(
        self,
        *,
        participants: List[str],
        facts: List[Dict[str, Any]],
    ) -> Dict[str, int]:
        """Persist only fact-grounded entities and explicit participants."""
        entity_names = self._episode_entity_names(
            participants=participants,
            facts=facts,
        )
        mapping = self._db.add_entity_names(entity_names)
        return {
            str(entity_name): int(entity_id)
            for entity_name, entity_id in mapping.items()
            if str(entity_name).strip() and str(entity_id).strip().isdigit()
        }

    def _upsert_memory_recall_document(
        self,
        *,
        object_type: str,
        object_id: int,
        identity_lines: Sequence[str],
        title: str = "",
        summary: str = "",
        entity_ids: Optional[Sequence[int]] = None,
        topic_keys: Optional[Sequence[str]] = None,
        time_start: str = "",
        time_end: str = "",
        status: str = "",
        confidence: float = 0.0,
        importance: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Write one canonical retrieval projection without ranking it.

        The source tables own domain state. This helper owns only the stable
        textual representation shared by lexical search and embeddings.
        """
        identity_text = "\n".join(
            _compact_whitespace(line)
            for line in identity_lines
            if _compact_whitespace(line)
        )
        return self._db.upsert_memory_recall_document(
            object_type=object_type,
            object_id=object_id,
            identity_text=identity_text,
            identity_text_embedding=self._generate_embedding_vector(identity_text),
            title=title,
            summary=summary,
            entity_ids=entity_ids,
            topic_keys=topic_keys,
            time_start=time_start,
            time_end=time_end,
            status=status,
            confidence=confidence,
            importance=importance,
            metadata=metadata,
        )

    def _upsert_fact_recall_document(
        self,
        *,
        fact_id: int,
        summary: str,
        keywords: Sequence[str],
        entities: Sequence[str],
        entity_ids: Sequence[int],
        fact_root_topic: str,
        fact_aspect_topic: str,
        event_time_key: str,
        dialogue_time_key: str,
        confidence: float,
        importance: float,
    ) -> int:
        """Project one atomic fact into the shared recall-document store."""
        keyword_text = " ".join(str(value) for value in keywords or [] if value)
        return self._upsert_memory_recall_document(
            object_type="fact",
            object_id=fact_id,
            title=_compact_whitespace(summary)[:120],
            summary=summary,
            identity_lines=[
                f"summary: {_compact_whitespace(summary)}",
                f"keywords: {keyword_text}",
                f"entities: {', '.join(entities or [])}",
                f"fact_root_topic: {fact_root_topic}",
                f"fact_aspect_topic: {fact_aspect_topic}",
            ],
            entity_ids=entity_ids,
            topic_keys=[fact_root_topic, fact_aspect_topic],
            time_start=event_time_key or dialogue_time_key,
            time_end=event_time_key or dialogue_time_key,
            confidence=confidence,
            importance=importance,
            metadata={"projection_version": "v1"},
        )

    def _upsert_episode_recall_document(
        self,
        *,
        episode_id: int,
        title: str,
        summary: str,
        participants: Sequence[str],
        entity_ids: Sequence[int],
        canonical_topics: Sequence[str],
        started_at: str,
        ended_at: str,
    ) -> int:
        """Project one experience-level summary without exposing raw segments."""
        return self._upsert_memory_recall_document(
            object_type="episode",
            object_id=episode_id,
            title=title,
            summary=summary,
            identity_lines=[
                f"title: {_compact_whitespace(title)}",
                f"summary: {_compact_whitespace(summary)}",
                f"participants: {', '.join(participants or [])}",
                f"canonical_topics: {', '.join(canonical_topics or [])}",
            ],
            entity_ids=entity_ids,
            topic_keys=canonical_topics,
            time_start=started_at,
            time_end=ended_at,
            confidence=0.85,
            importance=0.6,
            metadata={"projection_version": "v1"},
        )

    def _log_extracted_fact_info(
        self,
        *,
        facts: List[Dict[str, Any]],
    ) -> None:
        if not facts:
            return
        for index, fact in enumerate(facts, 1):
            metadata = fact.get("metadata") if isinstance(fact.get("metadata"), dict) else {}
            self._log_info(
                "memory_store",
                "extract_fact_signals",
                {
                    "fact_index": index,
                    "fact_count": len(facts),
                    "summary": fact.get("summary") or "",
                    "fact_type": fact.get("fact_type"),
                    "event_time_key": fact.get("event_time_key") or "",
                    "dialogue_time_key": fact.get("dialogue_time_key") or "",
                    "keywords": fact.get("keywords") or "",
                    "entities": fact.get("entities") or [],
                    "primary_entity": fact.get("primary_entity"),
                    "fact_root_topic": fact.get("fact_root_topic") or "",
                    "fact_aspect_topic": fact.get("fact_aspect_topic") or "",
                    "entity_claim_signal": fact.get("entity_claim_signal") or [],
                    "future_commitment_signals": fact.get("future_commitment_signals") or [],
                    "importance": fact.get("importance"),
                    "confidence": fact.get("confidence"),
                    "time_confidence": metadata.get("time_confidence") or "",
                    "where": metadata.get("where") or "",
                    "metadata": metadata,
                    "batch_fact_index": index,
                    "batch_fact_count": len(facts),
                },
            )

    def _extract_memory_fact_from_raw_segments(
        self,
        raw_segments: List[Dict[str, Any]],
        *,
        prompt_language: str,
    ) -> Dict[str, Any]:
        """Extract narrative facts with the unified fact prompt."""
        data = self._extract_memory_fact_with_llm(
            raw_segments,
            prompt_language=prompt_language,
        )
        if data and data.get("facts"):
            return data
        return {"facts": []}

    def generate_episode_from_source_segments(
        self,
        *,
        source_segments: Sequence[Dict[str, Any]],
        facts: Sequence[Dict[str, Any]],
        prompt_language: str = "zh",
    ) -> Dict[str, Any]:
        """Generate an experience-level episode from source evidence.

        Facts remain compact coverage anchors, but the chronological source
        segments are the authoritative narrative material. This prevents an
        episode from degrading into a sentence-by-sentence fact concatenation.
        """
        segment_list = [
            item for item in source_segments or []
            if isinstance(item, dict)
            and _compact_whitespace(item.get("text") or "")
        ]
        if not segment_list:
            return {
                "status": "empty",
                "new_episode_count": 0,
                "fact_count": 0,
                "source_segment_count": 0,
            }
        fact_list = [
            item for item in facts or []
            if isinstance(item, dict)
            and _compact_whitespace(item.get("summary") or item.get("text") or "")
        ]
        fact_payload = [
            {
                "id": item.get("id"),
                "summary": _compact_whitespace(item.get("summary") or item.get("text") or ""),
                "fact_type": item.get("fact_type") or "",
                "fact_root_topic": item.get("fact_root_topic") or "",
                "fact_aspect_topic": item.get("fact_aspect_topic") or "",
                "dialogue_time_key": item.get("dialogue_time_key") or "",
                "event_time_key": item.get("event_time_key") or "",
            }
            for item in fact_list
        ]
        source_text = self._build_memory_segments_for_prompt(
            segment_list,
            prompt_language=prompt_language,
        )
        prompt_template = (
            EPISODE_SUMMARY_PROMPT_EN
            if prompt_language == "en"
            else EPISODE_SUMMARY_PROMPT_ZH
        )
        prompt = (
            prompt_template
            .replace("{source_segments}", source_text)
            .replace("{evidence_facts}", json.dumps(fact_payload, ensure_ascii=False))
        )
        parsed: Dict[str, Any] = {}
        for _ in range(2):
            parsed = self._parse_json_object_from_llm_text(self._call_llm(prompt) or "") or {}
            if _compact_whitespace(parsed.get("summary") or ""):
                break
        summary = _compact_whitespace(parsed.get("summary") or "")
        if not summary:
            summary = self._fallback_generate_episode_summary_from_raw_segments(segment_list)
        title = _compact_whitespace(parsed.get("title") or "")
        if not title:
            title = self._fallback_generate_episode_title_from_raw_segments(segment_list)
        source_text_for_topics = " ".join(
            _compact_whitespace(item.get("text") or "") for item in segment_list
        )
        topics = self._normalize_episode_canonical_topics(
            parsed.get("canonical_topics"),
            fallback_text=source_text_for_topics,
            limit=3,
        )
        if not topics:
            topics = (
                self._normalize_unique_labels([
                    item.get("fact_root_topic")
                    for item in fact_payload
                    if item.get("fact_root_topic")
                ])[:3]
                or self._topic_candidates(summary)[:3]
            )
        participants = self._parse_participants_from_raw_segments(segment_list)
        entity_names = self._episode_entity_names(
            participants=participants,
            facts=fact_list,
        )
        episode_entity_ids = self._entity_ids_from_names_and_facts(
            names=entity_names,
            facts=fact_list,
        )
        start = _compact_whitespace(segment_list[0].get("started_at") or "") or _now_text()
        end = (
            _compact_whitespace(segment_list[-1].get("ended_at") or "")
            or _compact_whitespace(segment_list[-1].get("started_at") or "")
            or start
        )
        return {
            "status": "ok",
            "new_episode_count": 0,
            "title": title,
            "summary": summary,
            "canonical_topics": topics,
            "participants": participants,
            "entity_ids": episode_entity_ids,
            "fact_ids": [
                int(item["id"])
                for item in fact_list
                if str(item.get("id", "")).isdigit()
            ],
            "source_segment_ids": list(dict.fromkeys(
                int(item["id"])
                for item in segment_list
                if str(item.get("id", "")).isdigit()
            )),
            "started_at": start,
            "ended_at": end,
        }

    def _extract_memory_fact_with_llm(
        self,
        segments: List[Dict[str, Any]],
        *,
        prompt_language: str,
    ) -> Optional[Dict[str, Any]]:
        prompt_template = (
            UNIFIED_MEMORY_EXTRACTION_PROMPT_EN
            if prompt_language == "en"
            else UNIFIED_MEMORY_EXTRACTION_PROMPT_ZH
        )
        memory_topic_item_context = (
            self._collect_memory_topic_item_context(segments=segments)
            if prompt_language != "en"
            else {"canonical_topics": [], "aspect_topics": []}
        )
        prompt = (
            prompt_template
            .replace(
                "{existing_memory_topic_items}",
                self._format_memory_topic_items_for_prompt(memory_topic_item_context),
            )
            .replace(
                "{dialogue_batch}",
                self._build_memory_segments_for_prompt(
                    segments,
                    prompt_language=prompt_language,
                ),
            )
        )
        for attempt in range(2):
            result = self._call_llm(prompt)
            parsed = self._parse_json_object_from_llm_text(result or "")
            if parsed is not None:
                normalized = self._normalize_memory_fact_extraction_llm_output(
                    parsed,
                    segments,
                    prompt_language=prompt_language,
                )
                if normalized is not None:
                    return normalized
            if attempt == 0:
                self._logger.debug("Unified memory LLM extraction failed, retrying")
        return None

    def _call_llm(self, prompt: str) -> Optional[str]:
        if (
            not self._llm_api_key
            or not self._llm_base_url
            or str(self._llm_base_url).strip().lower() == "none"
        ):
            self._logger.debug(
                "Skipping LLM call because llm_api_key or llm_base_url is not configured"
            )
            return None
        url = f"{self._llm_base_url.rstrip('/')}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self._llm_api_key:
            headers["Authorization"] = f"Bearer {self._llm_api_key}"
        payload: Dict[str, Any] = {
            "model": self._llm_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "stream": False,
            "max_tokens": 2048,
        }
        if self._llm_json_mode:
            payload["response_format"] = {"type": "json_object"}
        if self._llm_thinking in {"disabled", "enabled"}:
            payload["thinking"] = {"type": self._llm_thinking}

        attempts = [payload]
        if "thinking" in payload:
            stripped = dict(payload)
            stripped.pop("thinking", None)
            attempts.append(stripped)
        if "response_format" in payload:
            stripped = dict(payload)
            stripped.pop("response_format", None)
            attempts.append(stripped)

        seen = set()
        for item in attempts:
            marker = json.dumps(sorted(item.keys()), ensure_ascii=False)
            if marker in seen:
                continue
            seen.add(marker)
            try:
                response = requests.post(url, json=item, headers=headers, timeout=self._llm_timeout)
                response.raise_for_status()
                data = response.json()
                choices = data.get("choices") or []
                if choices:
                    message = choices[0].get("message") or {}
                    content = message.get("content")
                    if content:
                        return str(content)
            except requests.RequestException as exc:
                text = str(exc).lower()
                if "response_format" in text or "json" in text or "thinking" in text:
                    continue
                self._logger.warning("Unified memory LLM call failed: %s", exc)
                return None
        return None

    @staticmethod
    def _parse_json_object_from_llm_text(text: str) -> Optional[Dict[str, Any]]:
        raw = str(text or "").strip()
        if not raw:
            return None
        if raw.startswith("```"):
            start = raw.find("{")
            end = raw.rfind("}")
            if start >= 0 and end > start:
                raw = raw[start : end + 1]
        else:
            start = raw.find("{")
            end = raw.rfind("}")
            if 0 <= start < end and (start > 0 or end < len(raw) - 1):
                raw = raw[start : end + 1]
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return data if isinstance(data, dict) else None

    def _normalize_memory_fact_extraction_llm_output(
        self,
        data: Dict[str, Any],
        raw_segments: List[Dict[str, Any]],
        *,
        prompt_language: str,
    ) -> Optional[Dict[str, Any]]:
        raw_facts = data.get("facts")
        if not isinstance(raw_facts, list):
            return None
        facts: List[Dict[str, Any]] = []
        dialogue_time_key = _to_timestamp_text(
            raw_segments[0].get("started_at") if raw_segments else ""
        ) or _now_text()
        for raw_fact in raw_facts:
            if not isinstance(raw_fact, dict):
                continue
            text = _compact_whitespace(raw_fact.get("text") or raw_fact.get("summary") or "")
            if not text:
                continue
            priority = self._normalize_priority(raw_fact.get("priority", 70))
            if priority < 60:
                continue
            keywords = self._normalize_string_list(raw_fact.get("keywords"), limit=18)
            if not keywords:
                keywords = self._keywords(text, limit=18)
            entities = self._normalize_entity_names(raw_fact.get("entities"))
            if not entities:
                entities = self._entities(text)
            primary_entity = self._normalize_primary_entity(
                raw_fact.get("primary_entity"),
                entities=entities,
            )
            if primary_entity:
                primary_entity_name = primary_entity["name"]
                if primary_entity_name not in entities:
                    entities = [primary_entity_name, *entities]
            primary_entity_name = _compact_whitespace(
                (primary_entity or {}).get("name") or ""
            ).lower()
            if primary_entity_name in {"assistant", "agent", "the assistant", "助手"} \
                    and self._is_low_value_assistant_closing(text):
                continue
            if primary_entity_name in {"user", "the user", "用户"} \
                    and self._is_low_value_user_acknowledgement(text):
                continue
            fact_topic_fallback = " ".join(keywords[:3]) if keywords else "general"
            fact_root_topic, fact_aspect_topic = self._normalize_fact_topic_fields(
                raw_fact.get("fact_root_topic"),
                raw_fact.get("fact_aspect_topic"),
                fallback_root_topic=fact_topic_fallback,
                fallback_aspect_topic=fact_topic_fallback,
            )
            entity_claim_signal = self._normalize_entity_claim_signal(
                raw_fact.get("entity_claim_signal"),
                fallback_entity=primary_entity,
            )
            future_commitment_signals = self._normalize_future_commitment_signals(
                raw_fact.get("future_commitment_signals"),
                fallback_entity=primary_entity,
            )
            event_time_key = _compact_whitespace(raw_fact.get("event_time_key") or "")
            facts.append({
                "summary": text,
                "fact_type": self._normalize_fact_type(raw_fact.get("fact_type")),
                "event_time_key": event_time_key,
                "dialogue_time_key": dialogue_time_key,
                "keywords": keywords,
                "entities": entities,
                "primary_entity": primary_entity,
                "entity_claim_signal": entity_claim_signal,
                "future_commitment_signals": future_commitment_signals,
                "fact_root_topic": fact_root_topic,
                "fact_aspect_topic": fact_aspect_topic,
                "importance": max(0.6, min(1.0, priority / 100.0)),
                "confidence": 0.9,
                "metadata": {
                    "extractor": "llm",
                    "priority": priority,
                    "time_confidence": _compact_whitespace(raw_fact.get("time_confidence") or "unknown"),
                    "where": _compact_whitespace(raw_fact.get("where") or ""),
                },
            })
        return {"facts": facts}

    def _normalize_fact_topic_fields(
        self,
        root_topic: Any,
        aspect_topic: Any,
        *,
        fallback_root_topic: Any,
        fallback_aspect_topic: Any,
    ) -> Tuple[str, str]:
        normalized_root = (
            self._normalize_topic_name(root_topic)
            or self._normalize_topic_name(fallback_root_topic)
            or "general"
        )
        normalized_aspect = (
            self._normalize_topic_name(aspect_topic)
            or self._normalize_topic_name(fallback_aspect_topic)
            or normalized_root
        )
        return normalized_root, normalized_aspect

    def _normalize_entity_claim_signal(
        self,
        value: Any,
        *,
        fallback_entity: Optional[Dict[str, str]] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if not isinstance(value, list):
            return []
        max_items = max(
            0,
            int(
                limit
                if limit is not None
                else self._memory_cfg.get("entity_claim_signal_max_per_fact", 3) or 3
            ),
        )
        if max_items <= 0:
            return []
        allowed_types = self._entity_claim_types()
        allowed_kinds = {"explicit_assertion", "pattern_observation"}
        normalized: List[Dict[str, Any]] = []
        seen: set[Tuple[str, str, str, str]] = set()
        for raw in value:
            if not isinstance(raw, dict):
                continue
            signal_kind = str(raw.get("signal_kind") or "").strip().lower()
            claim_type_hint = str(raw.get("claim_type_hint") or "").strip().lower()
            if signal_kind not in allowed_kinds or claim_type_hint not in allowed_types:
                continue
            if (
                signal_kind == "explicit_assertion"
                and claim_type_hint == "behavior_pattern"
            ):
                continue
            if (
                signal_kind == "pattern_observation"
                and claim_type_hint not in {"preference", "behavior_pattern"}
            ):
                continue
            claim_anchor = _compact_whitespace(
                raw.get("claim_anchor")
                or raw.get("anchor")
                or raw.get("attribute_name")
                or ""
            )
            evidence_basis = _compact_whitespace(
                raw.get("evidence_basis")
                or raw.get("evidence")
                or raw.get("reason")
                or ""
            )
            if (
                not claim_anchor
                or not evidence_basis
            ):
                continue
            confidence = self._clamp_float(raw.get("confidence"), 0.0, 1.0, 0.75)
            entity = raw.get("entity") or raw.get("primary_entity") or fallback_entity
            if isinstance(entity, dict):
                entity_name = _compact_whitespace(entity.get("name") or entity.get("text") or "")
                entity_type = _compact_whitespace(entity.get("type") or "CONCEPT").upper()
            else:
                entity_name = _compact_whitespace(entity)
                entity_type = "CONCEPT"
            entity_payload = (
                {"name": entity_name, "type": entity_type}
                if entity_name
                else None
            )
            key = (
                signal_kind,
                claim_type_hint,
                claim_anchor.lower(),
                evidence_basis.lower(),
            )
            if key in seen:
                continue
            seen.add(key)
            item: Dict[str, Any] = {
                "signal_kind": signal_kind,
                "claim_type_hint": claim_type_hint,
                "claim_anchor": claim_anchor,
                "evidence_basis": evidence_basis,
                "confidence": confidence,
            }
            if entity_payload:
                item["entity"] = entity_payload
            normalized.append(item)
            if len(normalized) >= max_items:
                break
        return normalized

    def _normalize_future_commitment_signals(
        self,
        value: Any,
        *,
        fallback_entity: Optional[Dict[str, str]] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Normalize fact-level evidence for the prospective world model."""
        if not isinstance(value, list):
            return []
        max_items = max(
            0,
            int(
                limit
                if limit is not None
                else self._memory_cfg.get("future_commitment_signal_max_per_fact", 2) or 2
            ),
        )
        if max_items <= 0:
            return []
        evidence_kind_aliases = {
            "goal_expression": "goal",
            "future_plan": "plan",
            "responsibility_commitment": "responsibility",
            "work_item": "responsibility",
            "plan_lifecycle_update": "lifecycle_update",
            "work_item_lifecycle_update": "lifecycle_update",
        }
        default_object_types = {
            "goal": ["goal"],
            "plan": ["plan"],
            "responsibility": ["work_item"],
            "lifecycle_update": ["plan", "work_item"],
        }
        normalized: List[Dict[str, Any]] = []
        seen: set[Tuple[str, str, str]] = set()
        for raw in value:
            if not isinstance(raw, dict):
                continue
            evidence_kind = evidence_kind_aliases.get(
                str(
                    raw.get("evidence_kind")
                    or raw.get("signal_type")
                    or raw.get("type")
                    or ""
                ).strip().lower(),
                str(raw.get("evidence_kind") or "").strip().lower(),
            )
            if evidence_kind not in default_object_types:
                continue
            raw_object_types = raw.get("candidate_object_types")
            if isinstance(raw_object_types, str):
                raw_object_types = re.split(r"[,，;；\s]+", raw_object_types)
            if not isinstance(raw_object_types, list):
                raw_object_types = []
            candidate_object_types = list(dict.fromkeys(
                str(item).strip().lower()
                for item in raw_object_types
                if str(item).strip().lower() in {"goal", "plan", "work_item"}
            )) or list(default_object_types[evidence_kind])
            user_role = str(raw.get("user_role") or "").strip().lower()
            if user_role not in {"owner", "participant", "responsible"}:
                continue
            assertion_source = str(raw.get("assertion_source") or "").strip().lower()
            if assertion_source not in {
                "self_statement", "third_party_report", "observed_event",
            }:
                continue
            explicitness = str(raw.get("explicitness") or "").strip().lower()
            if explicitness not in {"direct", "reported", "tentative"}:
                continue
            subject = raw.get("subject_entity") or raw.get("subject") or fallback_entity
            if isinstance(subject, dict):
                subject_name = _compact_whitespace(
                    subject.get("name") or subject.get("text") or ""
                )
            else:
                subject_name = _compact_whitespace(subject)
            if subject_name.lower() in {"我", "本人", "用户", "user", "the user"}:
                subject_name = self._world_owner_entity_name
            future_commitment_anchor = _compact_whitespace(
                raw.get("future_commitment_anchor") or raw.get("intent_anchor") or ""
            )[:240]
            evidence_basis = _compact_whitespace(
                raw.get("evidence_basis") or raw.get("evidence") or raw.get("reason") or ""
            )[:480]
            if not subject_name or not future_commitment_anchor or not evidence_basis:
                continue
            confidence = self._clamp_float(raw.get("confidence"), 0.0, 1.0, 0.75)
            key = (
                subject_name.lower(),
                evidence_kind,
                self._generate_topic_name_key(future_commitment_anchor),
            )
            if key in seen:
                continue
            seen.add(key)
            normalized.append({
                "subject_entity": subject_name,
                "evidence_kind": evidence_kind,
                "candidate_object_types": candidate_object_types,
                "user_role": user_role,
                "future_commitment_anchor": future_commitment_anchor,
                "assertion_source": assertion_source,
                "explicitness": explicitness,
                "evidence_basis": evidence_basis,
                "confidence": confidence,
            })
            if len(normalized) >= max_items:
                break
        return normalized

    def _build_dialogue_batch_for_prompt(
        self,
        turns: List[Dict[str, Any]],
        *,
        prompt_language: str,
    ) -> str:
        is_en = prompt_language == "en"
        time_label = "Conversation timestamp" if is_en else "对话发生时间"
        user_label = "User" if is_en else "用户"
        assistant_label = "Assistant" if is_en else "助手"
        blocks: List[str] = []
        for index, turn in enumerate(turns, 1):
            blocks.append(
                "\n".join([
                    f"[Turn {index}]",
                    f"{time_label}: {turn.get('turn_timestamp') or ''}",
                    f"{user_label}: {turn.get('user_message') or ''}",
                    f"{assistant_label}: {turn.get('assistant_response') or ''}",
                ])
            )
        return "\n\n".join(blocks)

    @staticmethod
    def _parse_participants_from_raw_segments(raw_segments: List[Dict[str, Any]]) -> List[str]:
        participants: List[str] = []
        seen: set[str] = set()
        for segment in raw_segments:
            speaker = _compact_whitespace(segment.get("speaker") or "")
            if not speaker:
                continue
            key = speaker.lower()
            if key in seen:
                continue
            seen.add(key)
            participants.append(speaker)
        return participants or ["unknown_speaker"]

    @staticmethod
    def _append_unique_text(values: List[str], value: Any, *, limit: int = 64) -> None:
        text = _compact_whitespace(value)
        if not text:
            return
        seen = {item.lower() for item in values}
        if text.lower() in seen:
            return
        values.append(text)
        if len(values) > limit:
            del values[limit:]

    def _entity_ids_for_names(self, names: Sequence[Any], *, limit: int = 64) -> List[int]:
        normalized: List[str] = []
        for name in names or []:
            self._append_unique_text(normalized, name, limit=limit)
        mapping = self._db.add_entity_names(normalized)
        ids: List[int] = []
        for name in normalized:
            entity_id = mapping.get(name)
            if entity_id and entity_id not in ids:
                ids.append(entity_id)
        return ids

    def _entity_ids_from_names_and_facts(
        self,
        *,
        names: Sequence[Any],
        facts: Sequence[Dict[str, Any]],
        limit: int = 64,
    ) -> List[int]:
        ids: List[int] = []
        for fact in facts or []:
            for value in fact.get("entity_ids") or []:
                try:
                    entity_id = int(value)
                except (TypeError, ValueError):
                    continue
                if entity_id and entity_id not in ids:
                    ids.append(entity_id)
                if len(ids) >= limit:
                    return ids
        for entity_id in self._entity_ids_for_names(names, limit=limit):
            if entity_id not in ids:
                ids.append(entity_id)
            if len(ids) >= limit:
                break
        return ids

    def _fact_entity_names(
        self,
        fact: Dict[str, Any],
        *,
        entities: Optional[Sequence[str]] = None,
    ) -> List[str]:
        names: List[str] = []
        normalized_entities = (
            list(entities)
            if entities is not None
            else self._normalize_entity_names(fact.get("entities"), limit=32)
        )
        for entity in normalized_entities:
            self._append_unique_text(names, entity)
        primary = fact.get("primary_entity")
        if isinstance(primary, dict):
            self._append_unique_text(names, primary.get("name") or primary.get("text"))
        else:
            self._append_unique_text(names, primary)
        return names

    def _episode_entity_names(
        self,
        *,
        participants: Sequence[str],
        facts: Sequence[Dict[str, Any]],
    ) -> List[str]:
        """Return the episode entity set from structured fact evidence only.

        Episode summaries and raw source segments are natural-language views,
        not entity sources: extracting names from either bypasses fact-level
        entity filtering and can create fragmented or incidental nodes.
        """
        names: List[str] = []
        for participant in participants or []:
            self._append_unique_text(names, participant)
        for fact in facts or []:
            for entity in self._fact_entity_names(fact):
                self._append_unique_text(names, entity)
        return names

    def _build_memory_segments_for_prompt(
        self,
        segments: List[Dict[str, Any]],
        *,
        prompt_language: str,
    ) -> str:
        is_en = prompt_language == "en"
        time_label = "Time" if is_en else "时间"
        speaker_label = "Speaker" if is_en else "说话人"
        text_label = "Text" if is_en else "文本"
        blocks: List[str] = []
        for index, segment in enumerate(segments, 1):
            started_at = segment.get("started_at") or ""
            ended_at = segment.get("ended_at") or started_at
            time_text = started_at if started_at == ended_at else f"{started_at} - {ended_at}"
            blocks.append(
                "\n".join([
                    f"[Segment {index}]",
                    f"{time_label}: {time_text}",
                    f"{speaker_label}: {segment.get('speaker') or ''}",
                    f"{text_label}: {segment.get('text') or ''}",
                ])
            )
        return "\n\n".join(blocks)

    def _is_indexable_memory_topic(self, value: Any) -> bool:
        """Reject fallback labels that must not become reusable topic names."""
        topic = self._normalize_topic_name(value)
        if not topic:
            return False
        return self._generate_topic_name_key(topic) not in {
            "general",
            "通用",
            "其他",
            "其它",
            "unknown",
            "未知",
        }

    def _build_memory_topic_item_updates(
        self,
        *,
        canonical_topics: Sequence[Any],
        aspect_topics: Sequence[Any] = (),
        fact_id: Optional[int] = None,
        episode_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Create compact topic-registry updates from persisted memory fields."""
        normalized_fact_id = int(fact_id or 0)
        normalized_episode_id = int(episode_id or 0)
        updates: List[Dict[str, Any]] = []
        canonical_keys: set[str] = set()

        def add(topic_kind: str, value: Any) -> None:
            topic_name = self._normalize_topic_name(value)
            if not topic_name or not self._is_indexable_memory_topic(topic_name):
                return
            topic_key = self._generate_topic_name_key(topic_name)
            if topic_kind == "aspect" and topic_key in canonical_keys:
                # An aspect equal to its fact root adds no finer-grained
                # vocabulary and would only duplicate prompt candidates.
                return
            item: Dict[str, Any] = {
                "topic_kind": topic_kind,
                "topic_name": topic_name,
                "topic_key": topic_key,
                "fact_ids": [normalized_fact_id] if normalized_fact_id > 0 else [],
                "episode_ids": [normalized_episode_id] if normalized_episode_id > 0 else [],
            }
            if not any(
                existing["topic_kind"] == topic_kind
                and existing["topic_key"] == topic_key
                for existing in updates
            ):
                updates.append(item)

        for topic in canonical_topics or ():
            topic_name = self._normalize_topic_name(topic)
            if topic_name and self._is_indexable_memory_topic(topic_name):
                canonical_keys.add(self._generate_topic_name_key(topic_name))
            add("canonical", topic)
        # Aspect topics are defined only by facts. Episode callers therefore
        # leave this sequence empty.
        if normalized_fact_id > 0:
            for topic in aspect_topics or ():
                add("aspect", topic)
        return updates

    def _collect_memory_topic_item_context(
        self,
        *,
        segments: Sequence[Dict[str, Any]],
        canonical_limit: int = 12,
        aspect_limit: int = 12,
    ) -> Dict[str, List[str]]:
        """Return only topic names lexically related to the incoming evidence."""
        query_text = " ".join(
            _compact_whitespace(segment.get("text") or "")
            for segment in segments or ()
            if isinstance(segment, dict)
        )
        query_terms = self._lexical_search_terms_for_text(
            query_text,
            limit=24,
            preserve_phrase=False,
        )
        if not query_terms:
            return {"canonical_topics": [], "aspect_topics": []}
        try:
            rows = self._db.list_memory_topic_items(limit=240)
        except Exception as exc:
            self._logger.debug("Failed to load memory topic item context: %s", exc)
            return {"canonical_topics": [], "aspect_topics": []}

        ranked: List[Tuple[float, Dict[str, Any]]] = []
        query_key = self._generate_topic_name_key(query_text)
        for row in rows:
            topic_name = self._normalize_topic_name(row.get("topic_name") or "")
            topic_key = self._generate_topic_name_key(topic_name) if topic_name else ""
            if not topic_name or not topic_key:
                continue
            score = self._topic_name_best_pair_similarity(
                query_terms,
                [topic_name],
            )
            if query_key and topic_key and topic_key in query_key:
                score = max(score, 0.9)
            if score < 0.5:
                continue
            ranked.append((score, row))

        result: Dict[str, List[str]] = {
            "canonical_topics": [],
            "aspect_topics": [],
        }
        limits = {
            "canonical": max(1, int(canonical_limit or 12)),
            "aspect": max(1, int(aspect_limit or 12)),
        }
        for _score, row in sorted(
            ranked,
            key=lambda item: (
                item[0],
                str(item[1].get("last_seen_at") or ""),
                int(item[1].get("fact_occurrence_count") or 0)
                + int(item[1].get("episode_occurrence_count") or 0),
            ),
            reverse=True,
        ):
            kind = str(row.get("topic_kind") or "").strip().lower()
            result_key = f"{kind}_topics"
            topic_name = self._normalize_topic_name(row.get("topic_name") or "")
            if (
                kind not in limits
                or not topic_name
                or topic_name in result[result_key]
                or len(result[result_key]) >= limits[kind]
            ):
                continue
            result[result_key].append(topic_name)
        return result

    @staticmethod
    def _format_memory_topic_items_for_prompt(
        topic_items: Dict[str, List[str]],
        *,
        max_chars: int = 1200,
    ) -> str:
        payload = {
            "canonical_topics": list(topic_items.get("canonical_topics") or []),
            "aspect_topics": list(topic_items.get("aspect_topics") or []),
        }
        while payload["canonical_topics"] or payload["aspect_topics"]:
            text = json.dumps(payload, ensure_ascii=False, indent=2)
            if len(text) <= max_chars:
                return text
            if len(payload["aspect_topics"]) >= len(payload["canonical_topics"]):
                payload["aspect_topics"].pop()
            else:
                payload["canonical_topics"].pop()
        return "[]"

    def _normalize_episode_canonical_topics(
        self,
        value: Any,
        *,
        fallback_text: str,
        limit: int,
    ) -> List[str]:
        raw_topics = self._coerce_topic_list(value)
        if not raw_topics:
            raw_topics = self._topic_candidates(fallback_text)
        normalized: List[str] = []
        seen: set[str] = set()
        for raw_topic in raw_topics:
            topic = self._normalize_topic_name(raw_topic)
            if not topic:
                continue
            key = topic.lower()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(topic)
            if len(normalized) >= max(1, int(limit or 5)):
                break
        return normalized

    @staticmethod
    def _coerce_topic_list(value: Any) -> List[Any]:
        if isinstance(value, str):
            return re.split(r"[,，;；\n]+", value)
        if isinstance(value, list):
            out: List[Any] = []
            for item in value:
                if isinstance(item, dict):
                    out.append(
                        item.get("canonical_topic")
                        or item.get("topic")
                        or item.get("name")
                        or item.get("text")
                    )
                else:
                    out.append(item)
            return out
        return []

    @staticmethod
    def _normalize_topic_name(value: Any) -> str:
        text = _compact_whitespace(value)
        text = text.strip("'\".,:;!?，。！？、；：（）()[]{}")
        if not text:
            return ""
        lower = text.lower()
        if lower in _STOPWORDS:
            return ""
        if any(pattern in lower for pattern in _COURTESY_PATTERNS):
            return ""
        if re.search(r"[。！？!?；;，,]", text):
            return ""
        chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
        if chinese_chars:
            if not (2 <= len(chinese_chars) <= 18):
                return ""
        elif not (2 <= len(text.split()) <= 6):
            return ""
        generic_topics = {
            "方案确定", "产品设计讨论", "部门协作", "问题讨论", "用户咨询",
            "solution finalized", "product design discussion",
            "team collaboration", "problem discussion", "user consultation",
        }
        if lower in generic_topics:
            return ""
        return text

    def _topic_name_similarity(self, left: str, right: str) -> float:
        left_terms = set(self._topic_similarity_terms(left))
        right_terms = set(self._topic_similarity_terms(right))
        if not left_terms or not right_terms:
            return 0.0
        overlap = len(left_terms & right_terms)
        union = len(left_terms | right_terms)
        return overlap / max(1, union)

    def _topic_similarity_terms(self, text: str) -> List[str]:
        clean = _compact_whitespace(text).lower()
        chinese_chars = "".join(re.findall(r"[\u4e00-\u9fff]", clean))
        if jieba is not None and chinese_chars:
            raw_tokens = [
                *jieba.lcut(clean, HMM=False),
                *jieba.cut_for_search(clean, HMM=False),
            ]
            tokens: List[str] = []
            seen: set[str] = set()
            for token in raw_tokens:
                normalized = _compact_whitespace(token).strip(
                    "'\".,:;!?，。！？、；：（）()[]{}"
                )
                if not normalized or not re.search(r"[0-9a-zA-Z\u4e00-\u9fff]", normalized):
                    continue
                key = normalized.lower()
                if key in seen:
                    continue
                seen.add(key)
                tokens.append(key)
            if tokens:
                return tokens
        if len(chinese_chars) >= 3:
            return [chinese_chars[i : i + 2] for i in range(len(chinese_chars) - 1)]
        return self._keywords(clean, limit=12)

    def _resolve_prompt_language_from_text(self, text: str, *, fallback: str = "zh") -> str:
        mode = str(self._memory_prompt_language or "source").strip().lower()
        if mode in {"en", "english", "force_en"}:
            return "en"
        if mode in {"zh", "chinese", "force_zh"}:
            return "zh"
        if re.search(r"[\u4e00-\u9fff]", str(text or "")):
            return "zh"
        return "en" if str(fallback).lower().startswith("en") else "zh"

    def _episode_title(self, turns: List[Dict[str, Any]]) -> str:
        for turn in turns:
            text = turn.get("user_message") or turn.get("assistant_response") or ""
            if text:
                return _compact_whitespace(text)[:96]
        return "assistant interaction episode"

    def _episode_summary(self, turns: List[Dict[str, Any]]) -> str:
        chunks: List[str] = []
        for turn in turns[:6]:
            if turn.get("user_message"):
                chunks.append(f"User: {turn['user_message']}")
            if turn.get("assistant_response"):
                chunks.append(f"Assistant: {turn['assistant_response'][:600]}")
        return "\n".join(chunks)

    def _fallback_generate_episode_title_from_raw_segments(self, raw_segments: List[Dict[str, Any]]) -> str:
        for segment in raw_segments:
            text = segment.get("text") or ""
            if text:
                return _compact_whitespace(text)[:96]
        return "memory episode"

    def _fallback_generate_episode_summary_from_raw_segments(self, raw_segments: List[Dict[str, Any]]) -> str:
        chunks: List[str] = []
        for segment in raw_segments[:10]:
            speaker = segment.get("speaker") or "speaker"
            text = _compact_whitespace(segment.get("text") or "")
            if not text:
                continue
            started_at = segment.get("started_at") or ""
            chunks.append(f"{started_at} {speaker}: {text[:600]}")
        return "\n".join(chunks)

    def _is_low_value_assistant_closing(self, text: str) -> bool:
        clean = _compact_whitespace(text)
        if not clean:
            return True
        lower = clean.lower()
        if any(pattern in lower for pattern in _COURTESY_PATTERNS):
            has_specific_action = any(
                marker in lower
                for marker in (
                    "建议", "需要", "决定", "计划", "截止", "预约", "购买",
                    "recommend", "suggest", "need to", "decide", "plan", "deadline",
                )
            )
            return not has_specific_action
        return False

    @staticmethod
    def _is_low_value_user_acknowledgement(text: str) -> bool:
        clean = _compact_whitespace(text)
        if not clean:
            return True
        lower = clean.lower()
        if len(clean) > 48:
            return False
        if any(marker in lower for marker in ("?", "？", "帮我", "需要", "想要", "计划", "决定", "安排", "提醒", "购买", "预约", "need", "want", "plan", "decide", "remind")):
            return False
        acknowledgement_markers = (
            "好的", "可以", "行", "嗯", "谢谢", "试一试", "听起来", "明白",
            "ok", "okay", "thanks", "thank you", "sounds good", "i'll try",
        )
        return any(marker in lower for marker in acknowledgement_markers)

    def _store_extracted_memory_facts_into_db(
        self,
        *,
        episode_id: Optional[int],
        facts: List[Dict[str, Any]],
        tags: List[str],
        episode_context_topics: Optional[Sequence[str]] = None,
        entity_info: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Any]:
        fact_ids: List[int] = []
        topic_item_updates: List[Dict[str, Any]] = []
        entity_claim_signal_mapping_updates: List[Dict[str, Any]] = []
        future_commitment_signal_mapping_updates: List[Dict[str, Any]] = []
        entity_claim_signal_fact_ids: List[int] = []
        future_commitment_signal_fact_ids: List[int] = []
        normalized_entity_info = {
            str(entity_name): int(entity_id)
            for entity_name, entity_id in (entity_info or {}).items()
            if str(entity_name).strip() and str(entity_id).strip().isdigit()
        }
        episode_context_entities = list(normalized_entity_info.keys())
        for fact in facts:
            keywords = self._normalize_string_list(
                fact.get("keywords"),
                limit=18,
            )
            if not keywords:
                keywords = self._keywords(fact.get("summary") or "", limit=18)
            entities = self._normalize_entity_names(fact.get("entities"))
            raw_metadata = fact.get("metadata")
            metadata = dict(raw_metadata) if isinstance(raw_metadata, dict) else {}
            fallback_topics = self._topic_candidates(fact["summary"])
            fact_root_topic = self._normalize_topic_name(
                fact.get("fact_root_topic")
                or next(iter(episode_context_topics or []), "")
            ) or self._normalize_topic_name(
                fallback_topics[0] if fallback_topics else ""
            ) or "general"
            fact_aspect_topic = self._normalize_topic_name(
                fact.get("fact_aspect_topic")
                or fact_root_topic
            ) or fact_root_topic
            fact_entities = self._fact_entity_names(fact, entities=entities)
            entity_ids = [
                normalized_entity_info[entity_name]
                for entity_name in fact_entities
                if entity_name in normalized_entity_info
            ]
            fact_metadata = {
                **metadata,
                "tags": tags,
                "episode_context_topics": list(episode_context_topics or []),
                "episode_context_entities": list(episode_context_entities or []),
            }
            fact_id = self._db.insert_fact(
                episode_id=episode_id,
                fact_type=fact["fact_type"],
                summary=fact["summary"],
                keywords=keywords,
                entities=entities,
                entity_ids=entity_ids,
                fact_root_topic=fact_root_topic,
                fact_aspect_topic=fact_aspect_topic,
                event_time_key=fact.get("event_time_key") or "",
                dialogue_time_key=fact.get("dialogue_time_key") or "",
                confidence=fact["confidence"],
                importance=fact["importance"],
                metadata=fact_metadata,
            )
            self._upsert_fact_recall_document(
                fact_id=fact_id,
                summary=fact["summary"],
                keywords=keywords,
                entities=entities,
                entity_ids=entity_ids,
                fact_root_topic=fact_root_topic,
                fact_aspect_topic=fact_aspect_topic,
                event_time_key=fact.get("event_time_key") or "",
                dialogue_time_key=fact.get("dialogue_time_key") or "",
                confidence=float(fact["confidence"]),
                importance=float(fact["importance"]),
            )
            fact_ids.append(fact_id)
            entity_claim_updates = (
                self._fact_entity_claim_signal_mapping_updates(fact_id, fact)
            )
            entity_claim_signal_mapping_updates.extend(entity_claim_updates)
            if any(
                str(item.get("signal_key") or "") != "__fact__"
                for item in entity_claim_updates
            ):
                entity_claim_signal_fact_ids.append(fact_id)
            future_commitment_updates = (
                self._fact_future_commitment_signal_mapping_updates(fact_id, fact)
            )
            future_commitment_signal_mapping_updates.extend(future_commitment_updates)
            if any(
                str(item.get("signal_key") or "") != "__fact__"
                for item in future_commitment_updates
            ):
                future_commitment_signal_fact_ids.append(fact_id)
            topic_item_updates.extend(
                self._build_memory_topic_item_updates(
                    canonical_topics=[fact_root_topic],
                    aspect_topics=[fact_aspect_topic],
                    fact_id=fact_id,
                )
            )
        self._db.upsert_fact_entity_claim_signal_mappings(
            entity_claim_signal_mapping_updates
        )
        self._db.upsert_fact_future_commitment_signal_mappings(
            future_commitment_signal_mapping_updates
        )
        return {
            "fact_ids": fact_ids,
            "topic_item_updates": topic_item_updates,
            "entity_claim_signal_fact_count": len(entity_claim_signal_fact_ids),
            "entity_claim_signal_count": sum(
                1
                for item in entity_claim_signal_mapping_updates
                if str(item.get("signal_key") or "") != "__fact__"
            ),
            "future_commitment_signal_fact_count": len(
                future_commitment_signal_fact_ids
            ),
            "future_commitment_signal_count": sum(
                1
                for item in future_commitment_signal_mapping_updates
                if str(item.get("signal_key") or "") != "__fact__"
            ),
        }

    # ── Entity-claim update: facts/episodes -> entity claims ─────────────

    def submit_memory_entity_claim_update_task(self, *_, **kwargs: Any) -> Dict[str, Any]:
        """Queue an entity-claim update after accepted memory tasks."""
        completion_context = kwargs.pop("completion_context", None)
        if not self._memory_enabled:
            task_id = self._operation_reporter.next_task_id(
                "memory_entity_claim_update"
            )
            return self._reject_memory_task(
                task_kind="memory_entity_claim_update",
                task_id=task_id,
                reason="memory_disabled",
            )
        return self._submit_memory_task(
            task_kind="memory_entity_claim_update",
            payload=dict(kwargs),
            completion_context=(
                dict(completion_context)
                if isinstance(completion_context, dict)
                else None
            ),
        )

    def submit_memory_future_commitment_update_task(self, *_, **kwargs: Any) -> Dict[str, Any]:
        """Queue a future-commitment projection after committed fact evidence."""
        completion_context = kwargs.pop("completion_context", None)
        if not self._memory_enabled or not self._enable_memory_future_commitment_update:
            task_id = self._operation_reporter.next_task_id("memory_future_commitment_update")
            return self._reject_memory_task(
                task_kind="memory_future_commitment_update",
                task_id=task_id,
                reason=("memory_disabled" if not self._memory_enabled else "future_commitment_update_disabled"),
            )
        return self._submit_memory_task(
            task_kind="memory_future_commitment_update",
            payload=dict(kwargs),
            completion_context=(
                dict(completion_context)
                if isinstance(completion_context, dict)
                else None
            ),
        )

    def _process_memory_entity_claim_update_task(
        self,
        limit: Optional[int] = None,
        reference_timestamp: Optional[Any] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Project new facts and completed episodes into entity claims."""
        update_started_at = time.monotonic()
        limit = max(
            1,
            int(limit or self._memory_cfg.get("entity_claim_limit") or 100),
        )
        if reference_timestamp is None:
            reference_timestamp = kwargs.get("timestamp") or _now_text()
        self._log_info("memory_entity_claim", "start", {
            "limit": limit,
            "timestamp": reference_timestamp,
        })
        with self._db.transaction():
            claim_report = self._update_memory_entity_claims(
                limit=limit,
                reference_timestamp=reference_timestamp,
            )
        report = {
            "status": (
                "ok"
                if (
                    claim_report.get("explicit", {}).get("fact_count", 0)
                    or claim_report.get("derived", {}).get("updated", 0)
                    or claim_report.get("inductive", {}).get("seed_fact_count", 0)
                )
                else "empty"
            ),
            "explicit_claims_updated": int(
                claim_report.get("explicit", {}).get("updated", 0) or 0
            ),
            "inductive_claims_updated": int(
                claim_report.get("inductive", {}).get("updated", 0) or 0
            ),
            "derived_claims_updated": int(
                claim_report.get("derived", {}).get("updated", 0) or 0
            ),
            "facts_marked_processed_for_memory_entity_claim": int(
                claim_report.get("explicit", {}).get("facts_marked_processed", 0) or 0
            ),
            "facts_marked_processed_for_entity_claim_induction": int(
                claim_report.get("inductive", {}).get("facts_marked_processed", 0) or 0
            ),
            "total_elapsed_ms": round(
                (time.monotonic() - update_started_at) * 1000,
                2,
            ),
        }
        self._log_info("memory_entity_claim", "finish", report)
        return report

    def _process_memory_future_commitment_update_task(
        self,
        *,
        limit: Optional[int] = None,
        reference_timestamp: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Settle pending future-facing evidence in isolated evidence groups."""
        limit = max(
            1,
            int(limit or self._memory_cfg.get("future_commitment_limit") or 100),
        )
        reference_timestamp = reference_timestamp or _now_text()
        facts = self._db.get_unprocessed_facts(
            processing_target="future_commitment_update",
            reference_timestamp=reference_timestamp,
            limit=limit,
            restrict_to_today=False,
        )
        report: Dict[str, Any] = {
            "status": "empty",
            "seed_fact_count": len(facts),
            "future_commitment_signal_count": 0,
            "evidence_group_count": 0,
            "candidate_count": 0,
            "applied_count": 0,
            "created": 0,
            "updated": 0,
            "facts_marked_processed": 0,
            "failed_group_count": 0,
        }
        self._log_future_commitment_facts_loaded(
            "future_commitment_update", facts, limit, reference_timestamp,
        )
        if not facts:
            self._log_info("memory_future_commitment_update", "finish", report)
            return report

        facts_by_id = {
            int(fact["id"]): fact
            for fact in facts
            if str(fact.get("id") or "").strip().isdigit()
        }
        future_commitment_signals = [
            signal
            for fact in facts_by_id.values()
            for signal in self._future_commitment_signals_from_fact(fact)
        ]
        report["future_commitment_signal_count"] = len(future_commitment_signals)
        if not future_commitment_signals:
            self._log_info("memory_future_commitment_update", "finish", report)
            return report

        world_owner_id = self._intent_world_owner_entity_id()
        evidence_groups = self._group_future_commitment_signals(
            future_commitment_signals,
            facts_by_id=facts_by_id,
        )
        report["evidence_group_count"] = len(evidence_groups)
        successful_group_fact_ids: set[int] = set()
        failed_group_fact_ids: set[int] = set()
        for group_index, group in enumerate(evidence_groups):
            group_facts = list(group["facts"])
            group_fact_ids = list(group["fact_ids"])
            language = self._resolve_prompt_language_from_text("\n".join(
                str(fact.get("summary") or "") for fact in group_facts
            ))
            prompt_template = (
                INTENT_EXTRACTION_PROMPT_EN
                if language == "en" else INTENT_EXTRACTION_PROMPT_ZH
            )
            raw = self._call_llm(
                prompt_template.replace("{world_owner_name}", self._world_owner_entity_name)
                .replace("{reference_timestamp}", str(reference_timestamp))
                .replace("{facts}", json.dumps(
                    self._future_commitment_group_fact_prompt_views(group),
                    ensure_ascii=False,
                    indent=2,
                ))
            )
            parsed = self._parse_json_object_from_llm_text(raw or "")
            candidate_list_fields = (
                "goal_candidates",
                "plan_candidates",
                "work_item_candidates",
            )
            if (
                parsed is None
                or not isinstance(parsed, dict)
                or any(not isinstance(parsed.get(field), list) for field in candidate_list_fields)
            ):
                report["failed_group_count"] += 1
                failed_group_fact_ids.update(group_fact_ids)
                self._log_info("memory_future_commitment_update", "group_error", {
                    "group_index": group_index,
                    "group_key": group["group_key"],
                    "fact_ids": group_fact_ids,
                    "error": "invalid_llm_future_commitment_extraction_response",
                })
                continue

            group_facts_by_id = {
                int(fact["id"]): fact
                for fact in group_facts
                if str(fact.get("id") or "").strip().isdigit()
            }
            entity_ids = self._intent_entity_name_to_id(
                group_facts,
                additional_names=[
                    signal["subject_entity"]
                    for signal in group["signals"]
                ],
            )
            entity_ids[self._world_owner_entity_name] = world_owner_id
            candidates = self._normalize_intent_candidates(
                parsed,
                facts_by_id=group_facts_by_id,
                entity_ids=entity_ids,
                world_owner_id=world_owner_id,
            )
            report["candidate_count"] += len(candidates)
            decisions = self._reconcile_intent_candidates(
                candidates,
                world_owner_id=world_owner_id,
                prompt_language=language,
                existing_by_type=self._retrieve_future_commitment_related_intent_objects(
                    candidates,
                    world_owner_id=world_owner_id,
                ),
            )
            with self._db.transaction():
                applied = self._apply_intent_candidates(candidates, decisions)
            successful_group_fact_ids.update(group_fact_ids)
            report["applied_count"] += len(applied)
            report["created"] += sum(item["created"] for item in applied)
            report["updated"] += sum(not item["created"] for item in applied)
            self._log_info("memory_future_commitment_update", "group_finish", {
                "group_index": group_index,
                "group_key": group["group_key"],
                "fact_ids": group_fact_ids,
                "future_commitment_signal_count": len(group["signals"]),
                "candidate_count": len(candidates),
                "created": sum(item["created"] for item in applied),
                "updated": sum(not item["created"] for item in applied),
            })

        completed_group_fact_ids = sorted(
            successful_group_fact_ids - failed_group_fact_ids
        )
        if completed_group_fact_ids:
            with self._db.transaction():
                report["facts_marked_processed"] += self._db.mark_facts_processed(
                    processing_target="future_commitment_update",
                    fact_ids=completed_group_fact_ids,
                )

        report["status"] = (
            "error" if report["failed_group_count"] == len(evidence_groups)
            else "ok"
        )
        self._log_info("memory_future_commitment_update", "finish", report)
        return report

    def _group_future_commitment_signals(
        self,
        future_commitment_signals: Sequence[Dict[str, Any]],
        *,
        facts_by_id: Dict[int, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Group compatible persisted signals while keeping each prompt small."""
        groups_by_key: Dict[str, List[Dict[str, Any]]] = {}
        for signal in future_commitment_signals:
            fact_id = int(signal["fact_id"])
            if fact_id not in facts_by_id:
                continue
            object_key = ",".join(signal["candidate_object_types"])
            # A group is a bounded extraction batch, not an assertion that its
            # signals describe the same prospective object.  Anchors remain in
            # the prompt view as semantic evidence, but must not fragment the
            # batch merely because the fact extractor phrased them differently.
            base_key = "|".join((
                str(signal["subject_entity"]).lower(),
                object_key,
            ))
            groups = groups_by_key.setdefault(base_key, [])
            target = next((
                group for group in groups
                if fact_id in group["fact_ids"] or len(group["fact_ids"]) < 8
            ), None)
            if target is None:
                target = {
                    "group_key": f"{base_key}#{len(groups) + 1}",
                    "signals": [],
                    "fact_ids": [],
                    "facts": [],
                }
                groups.append(target)
            target["signals"].append(signal)
            if fact_id not in target["fact_ids"]:
                target["fact_ids"].append(fact_id)
                target["facts"].append(facts_by_id[fact_id])
        return [
            group
            for groups in groups_by_key.values()
            for group in groups
        ]

    def _future_commitment_group_fact_prompt_views(
        self,
        group: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        signals_by_fact_id: Dict[int, List[Dict[str, Any]]] = {}
        for signal in group["signals"]:
            signals_by_fact_id.setdefault(int(signal["fact_id"]), []).append(signal)
        views: List[Dict[str, Any]] = []
        for fact in group["facts"]:
            view = self._intent_fact_prompt_view(fact)
            view["future_commitment_signals"] = [
                {
                    key: signal[key]
                    for key in (
                        "evidence_kind", "candidate_object_types", "subject_entity",
                        "user_role", "future_commitment_anchor",
                        "assertion_source", "explicitness",
                    )
                }
                for signal in signals_by_fact_id.get(int(fact.get("id") or 0), [])
            ]
            views.append(view)
        return views

    def _retrieve_future_commitment_related_intent_objects(
        self,
        candidates: Sequence[Dict[str, Any]],
        *,
        world_owner_id: int,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Retrieve candidate-ranked existing objects for reconciliation.

        ``canonical_key`` is deliberately only one high-precision feature. It
        cannot be a retrieval gate because separately extracted descriptions
        of the same future-facing object rarely use identical keys.
        """
        active_statuses = {
            "goal": ["active"],
            "plan": ["planned", "rescheduled"],
            "work_item": ["open", "in_progress", "blocked"],
        }
        related_by_type: Dict[str, List[Dict[str, Any]]] = {}
        for object_type, statuses in active_statuses.items():
            existing = self._db.get_intent_objects(
                object_type=object_type,
                world_owner_entity_id=world_owner_id,
                statuses=statuses,
                limit=64,
            )
            candidates_of_type = [
                candidate
                for candidate in candidates
                if candidate.get("object_type") == object_type
            ]
            selected_by_id: Dict[int, Dict[str, Any]] = {}
            for candidate in candidates_of_type:
                ranked = sorted(
                    (
                        (
                            self._score_intent_candidate_against_existing_object(
                                candidate,
                                item,
                            ),
                            item,
                        )
                        for item in existing
                    ),
                    key=lambda pair: pair[0],
                    reverse=True,
                )
                strong_matches = [
                    item for score, item in ranked
                    if score >= 0.35
                ][:4]
                # Lifecycle updates are especially important not to lose. If
                # lexical/structured evidence is sparse, expose only the two
                # best candidate-ranked objects rather than an arbitrary
                # recency slice of the entire active set.
                if (
                    not strong_matches
                    and candidate.get("lifecycle_evidence") != "none"
                ):
                    strong_matches = [item for _score, item in ranked[:2]]
                for item in strong_matches:
                    object_id = int(item.get("id") or 0)
                    if object_id > 0:
                        selected_by_id[object_id] = item
            related_by_type[object_type] = list(selected_by_id.values())[:16]
        return related_by_type

    def _build_intent_object_match_profile(
        self,
        object_type: str,
        item: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Expose only type-specific identity evidence for intent matching."""
        normalized_type = str(object_type or "").strip().lower()
        if normalized_type == "goal":
            text_values = [
                item.get("desired_outcome"),
                item.get("summary"),
                item.get("success_criteria"),
            ]
            entity_id = int(item.get("owner_entity_id") or 0)
            time_value = item.get("target_at") or ""
        elif normalized_type == "plan":
            text_values = [
                item.get("event_or_activity"),
                item.get("summary"),
                item.get("location_text"),
            ]
            entity_id = int(item.get("actor_entity_id") or 0)
            time_value = item.get("start_at") or ""
        elif normalized_type == "work_item":
            text_values = [
                item.get("action_text"),
                item.get("deliverable"),
                item.get("summary"),
            ]
            entity_id = int(item.get("responsible_entity_id") or 0)
            time_value = item.get("due_at") or item.get("start_at") or ""
        else:
            return {"text_values": [], "entity_id": 0, "time_value": "", "canonical_key": ""}
        canonical_value = _compact_whitespace(item.get("canonical_key") or "")
        return {
            "text_values": self._normalize_unique_labels(text_values, limit=6),
            "entity_id": entity_id,
            "time_value": _compact_whitespace(time_value),
            "canonical_key": (
                self._generate_topic_name_key(canonical_value)
                if canonical_value else ""
            ),
        }

    def _score_intent_candidate_against_existing_object(
        self,
        candidate: Dict[str, Any],
        existing: Dict[str, Any],
    ) -> float:
        """Rank a same-type existing object without requiring key equality."""
        object_type = str(candidate.get("object_type") or "").strip().lower()
        candidate_profile = self._build_intent_object_match_profile(object_type, candidate)
        existing_profile = self._build_intent_object_match_profile(object_type, existing)
        text_score = self._topic_name_best_pair_similarity(
            candidate_profile["text_values"],
            existing_profile["text_values"],
        )
        canonical_score = self._topic_name_best_pair_similarity(
            [candidate_profile["canonical_key"]],
            [existing_profile["canonical_key"]],
        )
        candidate_entity_id = int(candidate_profile["entity_id"] or 0)
        existing_entity_id = int(existing_profile["entity_id"] or 0)
        entity_score = (
            1.0
            if candidate_entity_id and candidate_entity_id == existing_entity_id
            else 0.0
        )
        candidate_day = str(candidate_profile["time_value"] or "")[:10]
        existing_day = str(existing_profile["time_value"] or "")[:10]
        time_score = 1.0 if candidate_day and candidate_day == existing_day else 0.0
        return round(
            min(
                1.0,
                text_score * 0.6
                + canonical_score * 0.25
                + entity_score * 0.1
                + time_score * 0.05,
            ),
            6,
        )

    def _intent_world_owner_entity_id(self) -> int:
        mapping = self._db.add_entity_names([self._world_owner_entity_name])
        return int(mapping[self._world_owner_entity_name])

    def _intent_fact_prompt_view(self, fact: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": fact.get("id"), "summary": fact.get("summary") or "",
            "fact_type": fact.get("fact_type") or "",
            "entities": fact.get("entities") or [],
            "primary_entity": fact.get("primary_entity") or {},
            "event_time": fact.get("event_time_key") or "",
            "dialogue_time": fact.get("dialogue_time_key") or "",
            "keywords": fact.get("keywords") or [],
            "topics": [fact.get("fact_root_topic") or "", fact.get("fact_aspect_topic") or ""],
        }

    def _intent_entity_name_to_id(
        self,
        facts: Sequence[Dict[str, Any]],
        *,
        additional_names: Sequence[Any] = (),
    ) -> Dict[str, int]:
        names: List[str] = [self._world_owner_entity_name]
        for fact in facts:
            names.extend(self._normalize_entity_names(fact.get("entities")))
            primary = fact.get("primary_entity")
            if isinstance(primary, dict):
                names.append(_compact_whitespace(primary.get("name") or ""))
        names.extend(_compact_whitespace(name) for name in additional_names)
        mapping = self._db.add_entity_names([name for name in names if name])
        return {str(name): int(entity_id) for name, entity_id in mapping.items()}

    def _normalize_intent_candidates(
        self,
        payload: Dict[str, Any],
        *,
        facts_by_id: Dict[int, Dict[str, Any]],
        entity_ids: Dict[str, int],
        world_owner_id: int,
    ) -> List[Dict[str, Any]]:
        """Normalize type-specific prompt lists into one internal sequence."""
        normalized: List[Dict[str, Any]] = []
        for field_name, object_type in (
            ("goal_candidates", "goal"),
            ("plan_candidates", "plan"),
            ("work_item_candidates", "work_item"),
        ):
            raw_candidates = payload.get(field_name)
            if not isinstance(raw_candidates, list):
                continue
            for raw_candidate in raw_candidates:
                candidate = self._normalize_intent_candidate(
                    raw_candidate,
                    object_type=object_type,
                    facts_by_id=facts_by_id,
                    entity_ids=entity_ids,
                    world_owner_id=world_owner_id,
                )
                if candidate is not None:
                    normalized.append(candidate)
                if len(normalized) >= 24:
                    return normalized
        return normalized

    def _normalize_intent_candidate(
        self,
        raw: Any,
        *,
        object_type: str,
        facts_by_id: Dict[int, Dict[str, Any]],
        entity_ids: Dict[str, int],
        world_owner_id: int,
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(raw, dict):
            return None
        if object_type not in {"goal", "plan", "work_item"}:
            return None
        evidence_ids = list(dict.fromkeys(
            int(value) for value in (raw.get("evidence_fact_ids") or [])
            if str(value).strip().isdigit() and int(value) in facts_by_id
        ))[:12]
        summary = _compact_whitespace(raw.get("summary") or "")[:480]
        if not summary or not evidence_ids:
            return None
        canonical_key = self._generate_topic_name_key(raw.get("canonical_key") or summary)
        confidence = self._clamp_float(raw.get("confidence"), 0.0, 1.0, 0.7)
        lifecycle_evidence = str(raw.get("lifecycle_evidence") or "none").strip().lower()
        allowed_lifecycle_evidence = {
            "goal": {"none", "confirmed", "completed", "cancelled", "blocked"},
            "plan": {"none", "confirmed", "occurred", "cancelled", "rescheduled", "blocked"},
            "work_item": {"none", "confirmed", "completed", "cancelled", "blocked"},
        }
        if lifecycle_evidence not in allowed_lifecycle_evidence[object_type]:
            lifecycle_evidence = "none"
        candidate: Dict[str, Any] = {
            "object_type": object_type, "summary": summary,
            "canonical_key": canonical_key, "confidence": confidence,
            "evidence_fact_ids": evidence_ids,
            "world_owner_entity_id": world_owner_id,
            "lifecycle_evidence": lifecycle_evidence,
        }
        def entity_id(value: Any, *, fallback: int = 0) -> int:
            name = _compact_whitespace(value or "")
            if name.lower() in {"我", "本人", "用户", "user", "the user"}:
                return world_owner_id
            return int(entity_ids.get(name) or fallback)
        if object_type == "goal":
            owner_name = _compact_whitespace(raw.get("owner_entity") or self._world_owner_entity_name)
            if owner_name.lower() in {"我", "本人", "用户", "user", "the user"}:
                owner_name = self._world_owner_entity_name
            owner_id = entity_id(owner_name, fallback=world_owner_id)
            desired_outcome = _compact_whitespace(raw.get("desired_outcome") or "")[:480]
            if not owner_id or not desired_outcome:
                return None
            candidate.update(owner_entity_id=owner_id, owner_name=owner_name, desired_outcome=desired_outcome,
                success_criteria=_compact_whitespace(raw.get("success_criteria") or "")[:480],
                target_at=_compact_whitespace(raw.get("target_at") or "")[:80])
        elif object_type == "plan":
            actor_name = _compact_whitespace(raw.get("actor_entity") or self._world_owner_entity_name)
            if actor_name.lower() in {"我", "本人", "用户", "user", "the user"}:
                actor_name = self._world_owner_entity_name
            actor_id = entity_id(actor_name, fallback=world_owner_id)
            event = _compact_whitespace(raw.get("event_or_activity") or "")[:320]
            if not actor_id or not event:
                return None
            location = _compact_whitespace(raw.get("location") or "")[:160]
            if location and location not in entity_ids:
                entity_ids.update(self._db.add_entity_names([location]))
            candidate.update(actor_entity_id=actor_id, actor_name=actor_name, event_or_activity=event,
                start_at=_compact_whitespace(raw.get("scheduled_start_at") or "")[:80],
                end_at=_compact_whitespace(raw.get("scheduled_end_at") or "")[:80],
                time_precision=(
                    str(raw.get("time_precision") or "unknown").lower()
                    if str(raw.get("time_precision") or "unknown").lower()
                    in {"exact", "day", "week", "relative", "unknown"}
                    else "unknown"
                ),
                location_text=location, location_entity_id=int(entity_ids.get(location) or 0),
                participant_names=self._normalize_string_list(raw.get("participants"), limit=12))
        else:
            responsible_name = _compact_whitespace(raw.get("responsible_entity") or "")
            if responsible_name.lower() in {"我", "本人", "用户", "user", "the user"}:
                responsible_name = self._world_owner_entity_name
            responsible_id = entity_id(responsible_name)
            action_text = _compact_whitespace(raw.get("action_text") or "")[:480]
            deliverable = _compact_whitespace(raw.get("deliverable") or "")[:320]
            responsibility_type = str(raw.get("responsibility_type") or "").lower()
            if not responsible_id or not action_text or responsibility_type not in {
                "personal_action", "commitment", "assigned", "external_commitment",
            }:
                return None
            due_at = _compact_whitespace(raw.get("due_at") or "")[:80]
            available_from = _compact_whitespace(raw.get("available_from") or "")[:80]
            has_completion_anchor = bool(deliverable or due_at) or responsibility_type in {
                "commitment", "assigned", "external_commitment",
            }
            if not has_completion_anchor:
                return None
            candidate.update(responsible_entity_id=responsible_id, responsible_name=responsible_name, action_text=action_text,
                deliverable=deliverable, responsibility_type=responsibility_type,
                due_at=due_at,
                start_at=available_from,
                priority=_compact_whitespace(raw.get("priority") or "")[:80],
                beneficiary_names=self._normalize_string_list(raw.get("beneficiary_entities"), limit=8),
                delegator_names=self._normalize_string_list(raw.get("delegator_entities"), limit=8),
                collaborator_names=self._normalize_string_list(raw.get("collaborator_entities"), limit=8))
        return candidate

    def _reconcile_intent_candidates(
        self,
        candidates: Sequence[Dict[str, Any]],
        *,
        world_owner_id: int,
        prompt_language: str,
        existing_by_type: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    ) -> Dict[int, Dict[str, Any]]:
        if existing_by_type is None:
            existing_by_type = self._retrieve_future_commitment_related_intent_objects(
                candidates,
                world_owner_id=world_owner_id,
            )
        existing_by_type = {
            object_type: list(existing_by_type.get(object_type) or [])
            for object_type in ("goal", "plan", "work_item")
        }
        decisions: Dict[int, Dict[str, Any]] = {}
        for index, candidate in enumerate(candidates):
            direct = next((
                item for item in existing_by_type[candidate["object_type"]]
                if str(item.get("canonical_key") or "") == candidate["canonical_key"]
            ), None)
            decisions[index] = {
                "operation": (
                    self._intent_operation_from_lifecycle_evidence(candidate)
                    if direct else "create"
                ),
                "target_object_type": candidate["object_type"] if direct else "",
                "target_object_id": int(direct["id"]) if direct else 0,
                "reason": "deterministic_canonical_key_match" if direct else "no_direct_match",
            }
        existing = [
            self._intent_object_prompt_view(object_type, item)
            for object_type, items in existing_by_type.items() for item in items
        ]
        if not candidates or not existing:
            return decisions
        prompt_template = (
            INTENT_RECONCILIATION_PROMPT_EN
            if prompt_language == "en" else INTENT_RECONCILIATION_PROMPT_ZH
        )
        raw = self._call_llm(
            prompt_template.replace("{candidates}", json.dumps(candidates, ensure_ascii=False, indent=2))
            .replace("{existing_objects}", json.dumps(existing, ensure_ascii=False, indent=2))
        )
        parsed = self._parse_json_object_from_llm_text(raw or "")
        if not isinstance(parsed, dict) or not isinstance(parsed.get("decisions"), list):
            return decisions
        valid_ids = {
            (object_type, int(item["id"]))
            for object_type, items in existing_by_type.items() for item in items
        }
        for raw_decision in parsed["decisions"]:
            if not isinstance(raw_decision, dict):
                continue
            index = raw_decision.get("candidate_index")
            if not isinstance(index, int) or not 0 <= index < len(candidates):
                continue
            operation = str(raw_decision.get("operation") or "create").lower()
            target_type = str(raw_decision.get("target_object_type") or "").lower()
            try:
                target_id = int(raw_decision.get("target_object_id") or 0)
            except (TypeError, ValueError):
                continue
            if operation not in {"create", "confirm", "update", "complete", "cancel", "reschedule", "block"}:
                continue
            if operation != "create" and (
                target_type != candidates[index]["object_type"]
                or (target_type, target_id) not in valid_ids
            ):
                continue
            decisions[index] = {
                "operation": operation, "target_object_type": target_type,
                "target_object_id": target_id,
                "reason": _compact_whitespace(raw_decision.get("reason") or "llm_reconciliation")[:240],
            }
        return decisions

    @staticmethod
    def _intent_operation_from_lifecycle_evidence(candidate: Dict[str, Any]) -> str:
        """Map fact-level lifecycle evidence to an update for a matched object."""
        lifecycle_evidence = str(candidate.get("lifecycle_evidence") or "none").lower()
        return {
            "confirmed": "confirm",
            "completed": "complete",
            "occurred": "complete",
            "cancelled": "cancel",
            "rescheduled": "reschedule",
            "blocked": "block",
        }.get(lifecycle_evidence, "update")

    @staticmethod
    def _intent_object_prompt_view(object_type: str, item: Dict[str, Any]) -> Dict[str, Any]:
        fields = {
            "goal": ("summary", "desired_outcome", "status", "target_at", "canonical_key"),
            "plan": ("summary", "event_or_activity", "status", "start_at", "end_at", "location_text", "canonical_key"),
            "work_item": ("summary", "action_text", "deliverable", "status", "due_at", "responsibility_type", "canonical_key"),
        }[object_type]
        return {"object_type": object_type, "id": item.get("id"), **{
            field: item.get(field) or "" for field in fields
        }}

    def _sync_future_commitment_recall_document(
        self,
        *,
        object_type: str,
        item: Dict[str, Any],
    ) -> None:
        """Refresh the retrieval projection for one goal, plan, or work item."""
        normalized_type = str(object_type or "").strip().lower()
        object_id = int(item.get("id") or 0)
        if normalized_type not in {"goal", "plan", "work_item"} or object_id <= 0:
            return
        entity_fields = {
            "goal": ("world_owner_entity_id", "owner_entity_id"),
            "plan": ("world_owner_entity_id", "actor_entity_id", "location_entity_id"),
            "work_item": ("world_owner_entity_id", "responsible_entity_id"),
        }[normalized_type]
        entity_ids = [
            int(item.get(field) or 0)
            for field in entity_fields
            if int(item.get(field) or 0) > 0
        ]
        names_by_id = self._db.get_entity_names_by_ids(entity_ids)
        if normalized_type == "goal":
            time_start = time_end = str(item.get("target_at") or "")
            identity_lines = [
                f"owner: {names_by_id.get(int(item.get('owner_entity_id') or 0), '')}",
                f"summary: {_compact_whitespace(item.get('summary') or '')}",
                f"desired_outcome: {_compact_whitespace(item.get('desired_outcome') or '')}",
                f"success_criteria: {_compact_whitespace(item.get('success_criteria') or '')}",
                f"canonical_key: {item.get('canonical_key') or ''}",
                f"status: {item.get('status') or ''}",
                f"target_at: {item.get('target_at') or ''}",
            ]
        elif normalized_type == "plan":
            time_start = str(item.get("start_at") or "")
            time_end = str(item.get("end_at") or time_start)
            identity_lines = [
                f"actor: {names_by_id.get(int(item.get('actor_entity_id') or 0), '')}",
                f"summary: {_compact_whitespace(item.get('summary') or '')}",
                f"event_or_activity: {_compact_whitespace(item.get('event_or_activity') or '')}",
                f"location: {_compact_whitespace(item.get('location_text') or '')}",
                f"canonical_key: {item.get('canonical_key') or ''}",
                f"status: {item.get('status') or ''}",
                f"start_at: {item.get('start_at') or ''}",
                f"end_at: {item.get('end_at') or ''}",
            ]
        else:
            time_start = str(item.get("start_at") or "")
            time_end = str(item.get("due_at") or time_start)
            identity_lines = [
                f"responsible: {names_by_id.get(int(item.get('responsible_entity_id') or 0), '')}",
                f"summary: {_compact_whitespace(item.get('summary') or '')}",
                f"action: {_compact_whitespace(item.get('action_text') or '')}",
                f"deliverable: {_compact_whitespace(item.get('deliverable') or '')}",
                f"canonical_key: {item.get('canonical_key') or ''}",
                f"status: {item.get('status') or ''}",
                f"due_at: {item.get('due_at') or ''}",
                f"priority: {item.get('priority') or ''}",
            ]
        self._upsert_memory_recall_document(
            object_type=normalized_type,
            object_id=object_id,
            title=_compact_whitespace(item.get("summary") or "")[:120],
            summary=_compact_whitespace(item.get("summary") or ""),
            identity_lines=identity_lines,
            entity_ids=entity_ids,
            topic_keys=[str(item.get("canonical_key") or "")],
            time_start=time_start,
            time_end=time_end,
            status=str(item.get("status") or ""),
            confidence=float(item.get("confidence") or 0.0),
            importance=0.7,
            metadata={
                "future_commitment_object_type": normalized_type,
                "projection_version": "v1",
            },
        )

    def _apply_intent_candidates(
        self,
        candidates: Sequence[Dict[str, Any]],
        decisions: Dict[int, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        applied: List[Dict[str, Any]] = []
        for index, candidate in enumerate(candidates):
            decision = decisions.get(index) or {"operation": "create", "target_object_id": 0}
            object_type = candidate["object_type"]
            operation = str(decision.get("operation") or "create")
            target_id = int(decision.get("target_object_id") or 0)
            target = self._db.get_intent_object(
                object_type=object_type, object_id=target_id,
            ) if target_id else None
            status = self._intent_status_for_operation(object_type, operation, target)
            payload = self._intent_storage_payload(candidate, status=status)
            created = target is None
            if created:
                object_id = self._db.create_intent_object(
                    object_type=object_type, payload=payload,
                )
                previous_status = ""
                previous_payload: Dict[str, Any] = {}
            else:
                object_id = int(target["id"])
                previous_status = str(target.get("status") or "")
                previous_payload = dict(target)
                update_payload = {
                    key: value for key, value in payload.items()
                    if value not in ("", None, 0, {}) or key in {"status", "confidence"}
                }
                self._db.update_intent_object(
                    object_type=object_type, object_id=object_id, payload=update_payload,
                )
            stored_object = self._db.get_intent_object(
                object_type=object_type,
                object_id=object_id,
            )
            if stored_object:
                self._sync_future_commitment_recall_document(
                    object_type=object_type,
                    item=stored_object,
                )
            evidence_role = {
                "create": "creation", "confirm": "confirmation", "complete": "completion",
                "cancel": "cancellation", "block": "block",
            }.get(operation, "update")
            self._db.upsert_intent_evidence([{
                "object_type": object_type, "object_id": object_id,
                "evidence_type": "fact", "evidence_id": fact_id,
                "role": evidence_role, "observed_at": self._intent_effective_at(candidate),
            } for fact_id in candidate["evidence_fact_ids"]])
            self._db.insert_intent_event(
                object_type=object_type, object_id=object_id,
                event_type="create" if created else operation,
                previous_status=previous_status, new_status=status,
                previous_payload=previous_payload, new_payload=payload,
                evidence_fact_ids=candidate["evidence_fact_ids"],
                effective_at=self._intent_effective_at(candidate),
                decision_source=str(decision.get("reason") or "intent_reconciliation"),
            )
            self._write_intent_entity_links(object_type, object_id, candidate)
            applied.append({"object_type": object_type, "object_id": object_id, "created": created})
        return applied

    @staticmethod
    def _intent_status_for_operation(
        object_type: str, operation: str, target: Optional[Dict[str, Any]]) -> str:
        defaults = {"goal": "active", "plan": "planned", "work_item": "open"}
        status = str((target or {}).get("status") or defaults[object_type])
        return {
            "goal": {"complete": "achieved", "cancel": "abandoned"},
            "plan": {"complete": "occurred", "cancel": "cancelled", "reschedule": "rescheduled"},
            "work_item": {"complete": "completed", "cancel": "cancelled", "block": "blocked"},
        }[object_type].get(operation, status)

    def _intent_storage_payload(self, candidate: Dict[str, Any], *, status: str) -> Dict[str, Any]:
        payload = {key: value for key, value in candidate.items() if key in {
            "world_owner_entity_id", "canonical_key", "summary", "owner_entity_id",
            "desired_outcome", "success_criteria", "target_at", "actor_entity_id",
            "event_or_activity", "start_at", "end_at", "time_precision", "location_entity_id",
            "location_text", "responsible_entity_id", "action_text", "deliverable",
            "responsibility_type", "due_at", "priority",
        }}
        payload.update(status=status, confidence=candidate["confidence"], metadata={
            "source_fact_ids": candidate["evidence_fact_ids"],
        })
        if candidate["object_type"] == "work_item":
            payload.update(
                completed_at=(self._intent_effective_at(candidate) if status == "completed" else ""),
                extractor_version="future_commitment_update_v1", prompt_version="v1",
            )
        return payload

    @staticmethod
    def _intent_effective_at(candidate: Dict[str, Any]) -> str:
        return str(candidate.get("due_at") or candidate.get("start_at") or candidate.get("target_at") or "")

    def _write_intent_entity_links(self, object_type: str, object_id: int, candidate: Dict[str, Any]) -> None:
        names_by_role = (
            {"actor": [candidate.get("actor_name") or ""], "participant": candidate.get("participant_names") or []}
            if object_type == "plan" else {
                "beneficiary": candidate.get("beneficiary_names") or [],
                "delegator": candidate.get("delegator_names") or [],
                "collaborator": candidate.get("collaborator_names") or [],
            }
        )
        if object_type == "work_item":
            names_by_role["responsible"] = [candidate.get("responsible_name") or ""]
        names = [name for values in names_by_role.values() for name in values if name]
        mapping = self._db.add_entity_names(names)
        self._db.upsert_intent_entities(
            object_type=object_type, object_id=object_id,
            entities=[
                {"entity_id": mapping[name], "role": role}
                for role, values in names_by_role.items() for name in values
                if name in mapping
            ],
        )

    @staticmethod
    def _entity_claim_types() -> set[str]:
        return {
            "identity_profile", "affiliation", "relationship", "preference",
            "constraint", "behavior_pattern",
        }

    def _update_memory_entity_claims(
        self,
        *,
        limit: int,
        reference_timestamp: Any,
    ) -> Dict[str, Dict[str, Any]]:
        """Project facts and completed episodes into traceable claim records."""
        disabled = {
            "enabled": 0, "updated": 0, "facts_marked_processed": 0,
        }
        if not self._enable_memory_entity_claim_update:
            return {
                "explicit": dict(disabled),
                "derived": dict(disabled),
                "inductive": dict(disabled),
            }

        explicit_facts = self._db.get_unprocessed_facts(
            processing_target="entity_claim",
            limit=limit,
            reference_timestamp=reference_timestamp,
        )
        explicit_report = self._update_explicit_entity_claims_from_facts(explicit_facts)
        explicit_completed = bool(explicit_report.pop("completed", False))
        affected_claim_ids = explicit_report.pop("affected_claim_ids", [])
        derived_report = self._update_derived_entity_claims_from_explicit_claims(
            affected_claim_ids=affected_claim_ids,
        ) if explicit_completed else {
            "enabled": 1,
            "seed_claim_count": 0,
            "candidate_count": 0,
            "updated": 0,
            "created": 0,
            "invalidated": 0,
            "completed": False,
            "error": "explicit_claim_update_incomplete",
        }
        derived_completed = bool(derived_report.pop("completed", False))
        # A valid empty result is a completed projection.  If either the
        # explicit or directly-derived pass fails, leave these facts pending so
        # the idempotent claim reconciliation can retry the full local chain.
        if explicit_completed and derived_completed:
            explicit_report["facts_marked_processed"] = self._db.mark_facts_processed(
                processing_target="entity_claim",
                fact_ids=[fact.get("id") for fact in explicit_facts],
            )
        else:
            explicit_report["facts_marked_processed"] = 0

        induction_seed_facts = self._db.get_unprocessed_facts(
            processing_target="entity_claim_induction",
            reference_timestamp=reference_timestamp,
            limit=max(1, min(limit, 100)),
            restrict_to_today=False,
            require_episode=True,
        )
        inductive_report = self._update_inductive_entity_claims_from_facts(
            induction_seed_facts,
        )
        if inductive_report.pop("completed", False):
            inductive_report["facts_marked_processed"] = self._db.mark_facts_processed(
                processing_target="entity_claim_induction",
                fact_ids=[fact.get("id") for fact in induction_seed_facts],
            )
        else:
            inductive_report["facts_marked_processed"] = 0
        self._log_info("memory_entity_claim", "entity_claim_update_finish", {
            "explicit": explicit_report,
            "derived": derived_report,
            "inductive": inductive_report,
        })
        return {
            "explicit": explicit_report,
            "derived": derived_report,
            "inductive": inductive_report,
        }

    def _claim_fact_prompt_view(self, fact: Dict[str, Any]) -> Dict[str, Any]:
        metadata = fact.get("metadata") if isinstance(fact.get("metadata"), dict) else {}
        return {
            "fact_id": fact.get("id"),
            "summary": fact.get("summary") or "",
            "fact_type": fact.get("fact_type") or "",
            "entities": fact.get("entities") or [],
            "primary_entity": fact.get("primary_entity") or metadata.get("primary_entity"),
            "keywords": fact.get("keywords") or [],
            "event_time": fact.get("event_time_key") or "",
            "dialogue_time": fact.get("dialogue_time_key") or "",
        }

    def _claim_entity_name_to_id(self, facts: Sequence[Dict[str, Any]]) -> Dict[str, int]:
        names = [
            name
            for fact in facts
            for name in self._normalize_entity_names(fact.get("entities") or [], limit=24)
        ]
        return self._db.add_entity_names(names)

    def _sync_entity_claim_recall_documents(
        self,
        claim_ids: Sequence[int],
    ) -> None:
        """Refresh projections for claims created, merged, or transitioned."""
        claims = self._db.get_entity_claims_by_ids(claim_ids)
        entity_ids = [
            int(entity_id)
            for claim in claims
            for entity_id in (
                claim.get("subject_entity_id"),
                claim.get("object_entity_id"),
                claim.get("source_actor_entity_id"),
            )
            if str(entity_id or "").strip().isdigit() and int(entity_id) > 0
        ]
        names_by_id = self._db.get_entity_names_by_ids(entity_ids)
        for claim in claims:
            subject_id = int(claim.get("subject_entity_id") or 0)
            object_id = int(claim.get("object_entity_id") or 0)
            source_actor_id = int(claim.get("source_actor_entity_id") or 0)
            claim_entity_ids = [
                entity_id for entity_id in (subject_id, object_id, source_actor_id)
                if entity_id > 0
            ]
            self._upsert_memory_recall_document(
                object_type="entity_claim",
                object_id=int(claim["id"]),
                title=_compact_whitespace(claim.get("claim_text") or "")[:120],
                summary=_compact_whitespace(claim.get("claim_text") or ""),
                identity_lines=[
                    f"subject: {names_by_id.get(subject_id, '')}",
                    f"claim: {_compact_whitespace(claim.get('claim_text') or '')}",
                    f"predicate: {claim.get('predicate') or ''}",
                    f"object: {names_by_id.get(object_id, '')}",
                    f"claim_type: {claim.get('claim_type') or ''}",
                    f"claim_origin: {claim.get('claim_origin') or ''}",
                    f"status: {claim.get('status') or ''}",
                ],
                entity_ids=claim_entity_ids,
                topic_keys=[
                    str(claim.get("claim_type") or ""),
                    str(claim.get("predicate") or ""),
                ],
                time_start=str(claim.get("valid_from") or ""),
                time_end=str(claim.get("valid_to") or ""),
                status=str(claim.get("status") or ""),
                confidence=float(claim.get("confidence") or 0.0),
                importance=0.65,
                metadata={
                    "claim_origin": str(claim.get("claim_origin") or ""),
                    "claim_type": str(claim.get("claim_type") or ""),
                    "predicate": str(claim.get("predicate") or ""),
                    "projection_version": "v1",
                },
            )

    @staticmethod
    def _entity_claim_origin_priority(origin: Any) -> int:
        """Origin is a hard reconciliation precedence, not a soft score."""
        return {"explicit": 3, "inductive": 2, "derived": 1}.get(
            str(origin or "").strip().lower(),
            0,
        )

    @staticmethod
    def _entity_claim_storage_payload(candidate: Dict[str, Any]) -> Dict[str, Any]:
        return {
            key: value
            for key, value in candidate.items()
            if key not in {
                "evidence_fact_ids", "support_fact_ids", "premise_claim_ids",
            }
        }

    @staticmethod
    def _entity_claim_support_ids(
        candidate: Dict[str, Any],
    ) -> List[int]:
        support_ids = candidate.get("support_fact_ids")
        if support_ids is None:
            support_ids = candidate.get("evidence_fact_ids") or []
        return [int(value) for value in support_ids if str(value).strip().isdigit()]

    def _retrieve_related_entity_claims(
        self,
        candidate: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Bound relation candidates to one entity and compatible claim type."""
        rows = self._db.get_entity_claims(
            subject_entity_id=int(candidate["subject_entity_id"]),
            claim_type=str(candidate["claim_type"]),
            statuses=["active", "candidate", "weakened"],
            limit=80,
        )
        predicate = str(candidate.get("predicate") or "")
        object_entity_id = int(candidate.get("object_entity_id") or 0)
        rows.sort(
            key=lambda row: (
                str(row.get("predicate") or "") != predicate,
                int(row.get("object_entity_id") or 0) != object_entity_id,
                -self._entity_claim_origin_priority(row.get("claim_origin")),
                -float(row.get("confidence") or 0.0),
            )
        )
        return rows[:24]

    def _finalize_entity_claim_relation_decision(
        self,
        candidate: Dict[str, Any],
        target: Optional[Dict[str, Any]],
        semantic_decision: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Translate text-only relation into a safe origin-aware write plan."""
        semantic_relation = str(
            semantic_decision.get("semantic_relation")
            or semantic_decision.get("relation")
            or "unrelated"
        )
        decision = {
            **semantic_decision,
            "semantic_relation": semantic_relation,
            "merge_into_target": False,
            "candidate_status": str(candidate.get("status") or "candidate"),
            "target_status": "",
            "target_evidence_role": "",
        }
        if not target:
            return decision
        candidate_priority = self._entity_claim_origin_priority(
            candidate.get("claim_origin")
        )
        target_priority = self._entity_claim_origin_priority(
            target.get("claim_origin")
        )
        if semantic_relation == "duplicate":
            if candidate_priority <= target_priority:
                decision["merge_into_target"] = True
                decision["target_evidence_role"] = (
                    "context" if candidate_priority < target_priority else "support"
                )
            else:
                # Keep a direct explicit assertion as its own, stronger claim,
                # while using it to support the older inductive conclusion.
                decision["target_evidence_role"] = "support"
            return decision
        if semantic_relation in {"supports", "refines"}:
            decision["target_evidence_role"] = (
                "context" if candidate_priority < target_priority else "support"
            )
            return decision
        if semantic_relation not in {"contradicts", "supersedes"}:
            return decision
        if candidate_priority < target_priority:
            # An inferred pattern can coexist as a tentative competing claim,
            # but it never changes an explicit claim's state.
            decision["candidate_status"] = "candidate"
            return decision
        decision["target_evidence_role"] = "counterexample"
        decision["target_status"] = (
            "superseded" if semantic_relation == "supersedes" else "weakened"
        )
        return decision

    def _classify_entity_claim_relations(
        self,
        candidates: Sequence[Dict[str, Any]],
        related_by_index: Dict[int, List[Dict[str, Any]]],
    ) -> List[List[Dict[str, Any]]]:
        if not any(related_by_index.values()):
            return [[] for _candidate in candidates]
        language = self._resolve_prompt_language_from_text(
            "\n".join(str(candidate.get("claim_text") or "") for candidate in candidates)
        )
        prompt_template = (
            ENTITY_CLAIM_RECONCILIATION_PROMPT_EN
            if language == "en" else ENTITY_CLAIM_RECONCILIATION_PROMPT_ZH
        )
        prompt_candidates = [
            {
                "candidate_claim_index": index,
                "claim_text": candidate.get("claim_text") or "",
            }
            for index, candidate in enumerate(candidates)
        ]
        existing_by_id = {
            int(claim["id"]): claim
            for claims in related_by_index.values()
            for claim in claims
            if str(claim.get("id") or "").strip().isdigit()
        }
        raw = self._call_llm(
            prompt_template.replace("{candidate_claims}", json.dumps(
                prompt_candidates, ensure_ascii=False, indent=2,
            )).replace("{existing_claims}", json.dumps(
                [
                    {"id": claim.get("id"), "claim_text": claim.get("claim_text") or ""}
                    for claim in existing_by_id.values()
                ],
                ensure_ascii=False, indent=2,
            ))
        )
        parsed = self._parse_json_object_from_llm_text(raw or "")
        if not parsed or not isinstance(parsed.get("decisions"), list):
            return [[] for _candidate in candidates]
        else:
            allowed_relations = {
                "duplicate", "supports", "contradicts", "refines", "supersedes",
            }
            semantic_resolved: List[List[Dict[str, Any]]] = [
                [] for _candidate in candidates
            ]
            for raw_decision in parsed["decisions"]:
                if not isinstance(raw_decision, dict):
                    continue
                try:
                    index = int(raw_decision.get("candidate_claim_index"))
                except (TypeError, ValueError):
                    continue
                if index < 0 or index >= len(candidates):
                    continue
                raw_relations = raw_decision.get("relations")
                if not isinstance(raw_relations, list):
                    continue
                allowed_ids = {
                    int(claim["id"])
                    for claim in related_by_index.get(index, [])
                    if str(claim.get("id") or "").strip().isdigit()
                }
                parsed_by_id: Dict[int, Dict[str, Any]] = {}
                for raw_relation in raw_relations:
                    if not isinstance(raw_relation, dict):
                        continue
                    try:
                        target_id = int(raw_relation.get("existing_claim_id"))
                    except (TypeError, ValueError):
                        continue
                    relation = str(
                        raw_relation.get("semantic_relation") or ""
                    ).strip().lower()
                    if target_id not in allowed_ids or relation not in allowed_relations:
                        continue
                    parsed_by_id[target_id] = {
                        "existing_claim_id": target_id,
                        "semantic_relation": relation,
                        "confidence": self._clamp_float(
                            raw_relation.get("confidence"), 0.0, 1.0, 0.7,
                        ),
                        "reason": _compact_whitespace(
                            raw_relation.get("reason") or ""
                        )[:240],
                    }
                semantic_resolved[index] = list(parsed_by_id.values())
        finalized: List[List[Dict[str, Any]]] = []
        for index, candidate in enumerate(candidates):
            candidate_decisions: List[Dict[str, Any]] = []
            for semantic_decision in semantic_resolved[index]:
                target_id = semantic_decision.get("existing_claim_id")
                target = next(
                    (
                        claim for claim in related_by_index.get(index, [])
                        if int(claim.get("id") or 0) == int(target_id or 0)
                    ),
                    None,
                )
                if target:
                    candidate_decisions.append(
                        self._finalize_entity_claim_relation_decision(
                            candidate, target, semantic_decision,
                        )
                    )
            finalized.append(candidate_decisions)
        return finalized

    def _write_entity_claim_evidence(
        self,
        *,
        claim_id: int,
        support_ids: Sequence[int],
        facts_by_id: Dict[int, Dict[str, Any]],
        confidence: float,
        support_role: str = "support",
    ) -> None:
        evidence: List[Dict[str, Any]] = []
        for fact_id in support_ids:
            fact = facts_by_id.get(int(fact_id))
            if not fact:
                continue
            evidence.append({
                "claim_id": claim_id,
                "evidence_type": "fact",
                "evidence_id": int(fact_id),
                "role": support_role,
                "weight": confidence,
                "observed_at": fact.get("event_time_key")
                or fact.get("dialogue_time_key") or "",
            })
        self._db.upsert_entity_claim_evidence(evidence)

    @staticmethod
    def _entity_claim_transition_effective_at(
        candidate: Dict[str, Any],
        *,
        support_ids: Sequence[int],
        facts_by_id: Dict[int, Dict[str, Any]],
    ) -> str:
        """Use the candidate's own temporal assertion, then its newest fact."""
        valid_from = _compact_whitespace(candidate.get("valid_from") or "")
        if valid_from:
            return valid_from
        observed_at = [
            _compact_whitespace(
                fact.get("event_time_key") or fact.get("dialogue_time_key") or ""
            )
            for fact_id in support_ids
            for fact in [facts_by_id.get(int(fact_id))]
            if fact
        ]
        return max((value for value in observed_at if value), default="")

    def _transition_entity_claim_target(
        self,
        *,
        target: Dict[str, Any],
        trigger_claim_id: int,
        candidate: Dict[str, Any],
        decision: Dict[str, Any],
        effective_at: str,
    ) -> bool:
        """Persist an origin-aware claim-state transition with its audit trail."""
        target_status = str(decision.get("target_status") or "")
        if not target_status:
            return False
        candidate_origin = str(candidate.get("claim_origin") or "")
        target_origin = str(target.get("claim_origin") or "")
        candidate_priority = self._entity_claim_origin_priority(candidate_origin)
        target_priority = self._entity_claim_origin_priority(target_origin)
        details = {
            "trigger_claim_snapshot": {
                "claim_text": candidate.get("claim_text") or "",
                "claim_origin": candidate_origin,
                "claim_type": candidate.get("claim_type") or "",
                "confidence": candidate.get("confidence"),
            },
            "target_claim_snapshot": {
                "claim_text": target.get("claim_text") or "",
                "claim_origin": target_origin,
                "claim_type": target.get("claim_type") or "",
                "confidence": target.get("confidence"),
            },
            "origin_priorities": {
                "trigger": candidate_priority,
                "target": target_priority,
            },
        }
        return self._db.transition_entity_claim_status(
            target_claim_id=int(target["id"]),
            new_status=target_status,
            trigger_claim_id=trigger_claim_id,
            semantic_relation=str(decision.get("semantic_relation") or ""),
            effective_at=effective_at,
            decision_source="entity_claim_reconciliation_v1:llm_semantics+origin_policy",
            semantic_confidence=self._clamp_float(
                decision.get("confidence"), 0.0, 1.0, 0.0,
            ),
            semantic_reason=_compact_whitespace(decision.get("reason") or "")[:240],
            policy_reason=(
                f"origin_precedence: trigger={candidate_origin}({candidate_priority}), "
                f"target={target_origin}({target_priority}), "
                f"target_status={target_status}"
            ),
            details=details,
        )

    def _reconcile_entity_claims(
        self,
        candidates: Sequence[Dict[str, Any]],
        *,
        facts_by_id: Dict[int, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Resolve new candidate claims before they become durable claims."""
        unique_candidates: List[Dict[str, Any]] = []
        seen: set[Tuple[Any, ...]] = set()
        for candidate in candidates:
            key = (
                candidate.get("subject_entity_id"), candidate.get("claim_type"),
                candidate.get("predicate"), candidate.get("object_entity_id") or 0,
                candidate.get("claim_text"), candidate.get("claim_origin"),
            )
            if key in seen:
                continue
            seen.add(key)
            unique_candidates.append(candidate)
        related_by_index = {
            index: self._retrieve_related_entity_claims(candidate)
            for index, candidate in enumerate(unique_candidates)
        }
        decisions = self._classify_entity_claim_relations(
            unique_candidates, related_by_index,
        )
        applied: List[Dict[str, Any]] = []
        for index, candidate in enumerate(unique_candidates):
            candidate_decisions = decisions[index]
            candidate_origin = str(candidate.get("claim_origin") or "")
            support_ids = self._entity_claim_support_ids(candidate)
            effective_at = self._entity_claim_transition_effective_at(
                candidate,
                support_ids=support_ids,
                facts_by_id=facts_by_id,
            )
            candidate_status = str(candidate.get("status") or "candidate")
            if any(
                str(decision.get("candidate_status") or "") == "candidate"
                for decision in candidate_decisions
            ):
                candidate_status = "candidate"

            def target_for(decision: Dict[str, Any]) -> Optional[Dict[str, Any]]:
                target_id = decision.get("existing_claim_id")
                return next(
                    (
                        item for item in related_by_index[index]
                        if int(item.get("id") or 0) == int(target_id or 0)
                    ),
                    None,
                )

            merge_decisions = [
                decision for decision in candidate_decisions
                if decision.get("merge_into_target") and target_for(decision)
            ]
            if merge_decisions:
                primary_merge = max(
                    merge_decisions,
                    key=lambda decision: self._entity_claim_origin_priority(
                        (target_for(decision) or {}).get("claim_origin")
                    ),
                )
                primary_target = target_for(primary_merge)
                assert primary_target is not None
                claim_id = int(primary_target["id"])
                for decision in candidate_decisions:
                    target = target_for(decision)
                    if not target:
                        continue
                    target_status = str(decision.get("target_status") or "")
                    target_evidence_role = str(
                        decision.get("target_evidence_role") or ""
                    )
                    if target_status:
                        self._transition_entity_claim_target(
                            target=target,
                            trigger_claim_id=claim_id,
                            candidate=candidate,
                            decision=decision,
                            effective_at=effective_at,
                        )
                    if target_evidence_role:
                        self._write_entity_claim_evidence(
                            claim_id=int(target["id"]),
                            support_ids=support_ids,
                            facts_by_id=facts_by_id,
                            confidence=float(candidate.get("confidence") or 0.0),
                            support_role=target_evidence_role,
                        )
                applied.append({
                    "claim": candidate, "claim_id": claim_id, "created": False,
                    "effective_origin": primary_target.get("claim_origin") or "",
                    "relations": candidate_decisions, "merged": True,
                })
                continue

            storage_payload = self._entity_claim_storage_payload(candidate)
            storage_payload["status"] = candidate_status
            claim_id, created = self._db.upsert_entity_claim(**storage_payload)
            self._write_entity_claim_evidence(
                claim_id=claim_id,
                support_ids=support_ids,
                facts_by_id=facts_by_id,
                confidence=float(candidate.get("confidence") or 0.0),
            )
            for decision in candidate_decisions:
                target = target_for(decision)
                if not target:
                    continue
                target_status = str(decision.get("target_status") or "")
                target_evidence_role = str(
                    decision.get("target_evidence_role") or ""
                )
                if target_status:
                    self._transition_entity_claim_target(
                        target=target,
                        trigger_claim_id=claim_id,
                        candidate=candidate,
                        decision=decision,
                        effective_at=effective_at,
                    )
                if target_evidence_role:
                    self._write_entity_claim_evidence(
                        claim_id=int(target["id"]),
                        support_ids=support_ids,
                        facts_by_id=facts_by_id,
                        confidence=float(candidate.get("confidence") or 0.0),
                        support_role=target_evidence_role,
                    )
            applied.append({
                "claim": candidate, "claim_id": claim_id, "created": created,
                "effective_origin": candidate_origin,
                "relations": candidate_decisions,
                "merged": False,
            })
        self._log_info("memory_entity_claim", "entity_claim_reconciled", {
            "candidate_count": len(unique_candidates),
            "applied_count": len(applied),
            "relations": [
                {
                    "claim_id": item["claim_id"], "origin": item["claim"].get("claim_origin"),
                    "relations": [
                        {
                            "existing_claim_id": decision.get("existing_claim_id"),
                            "semantic_relation": decision.get("semantic_relation"),
                        }
                        for decision in item["relations"]
                    ],
                    "merged": item["merged"],
                }
                for item in applied
            ],
        })
        recall_document_claim_ids = {
            int(item["claim_id"])
            for item in applied
            if int(item.get("claim_id") or 0) > 0
        }
        recall_document_claim_ids.update(
            int(decision["existing_claim_id"])
            for item in applied
            for decision in item.get("relations") or []
            if str(decision.get("existing_claim_id") or "").strip().isdigit()
            and int(decision["existing_claim_id"]) > 0
        )
        self._sync_entity_claim_recall_documents(
            sorted(recall_document_claim_ids)
        )
        return applied

    def _update_explicit_entity_claims_from_facts(
        self,
        facts: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        report: Dict[str, Any] = {
            "enabled": 1, "fact_count": len(facts), "claim_count": 0,
            "updated": 0, "created": 0, "completed": True,
        }
        if not facts:
            return report
        prompt_language = self._resolve_prompt_language_from_text(
            "\n".join(str(fact.get("summary") or "") for fact in facts[:20])
        )
        prompt_template = (
            EXPLICIT_ENTITY_CLAIM_EXTRACTION_PROMPT_EN
            if prompt_language == "en" else EXPLICIT_ENTITY_CLAIM_EXTRACTION_PROMPT_ZH
        )
        raw = self._call_llm(prompt_template.replace(
            "{facts}", json.dumps(
                [self._claim_fact_prompt_view(fact) for fact in facts[:40]],
                ensure_ascii=False, indent=2,
            ),
        ))
        parsed = self._parse_json_object_from_llm_text(raw or "")
        if parsed is None or not isinstance(parsed.get("claims"), list):
            report["completed"] = False
            report["error"] = "invalid_llm_claim_response"
            return report
        entity_ids = self._claim_entity_name_to_id(facts)
        facts_by_id = {
            int(fact["id"]): fact for fact in facts
            if str(fact.get("id") or "").strip().isdigit()
        }
        candidates: List[Dict[str, Any]] = []
        for raw_claim in parsed["claims"][:32]:
            claim = self._normalize_explicit_entity_claim(
                raw_claim, facts_by_id=facts_by_id, entity_ids=entity_ids,
            )
            if not claim:
                continue
            candidates.append(claim)
        applied = self._reconcile_entity_claims(candidates, facts_by_id=facts_by_id)
        report["updated"] = len(applied)
        report["created"] = sum(int(item["created"]) for item in applied)
        report["merged"] = sum(int(item["merged"]) for item in applied)
        report["claim_count"] = len(candidates)
        report["affected_claim_ids"] = sorted({
            int(claim_id)
            for item in applied
            for claim_id in (
                item.get("claim_id"),
                *[
                    decision.get("existing_claim_id")
                    for decision in item.get("relations") or []
                ],
            )
            if str(claim_id or "").strip().isdigit() and int(claim_id) > 0
        })
        return report

    def _normalize_explicit_entity_claim(
        self,
        raw: Any,
        *,
        facts_by_id: Dict[int, Dict[str, Any]],
        entity_ids: Dict[str, int],
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(raw, dict):
            return None
        claim_type = str(raw.get("claim_type") or "").strip().lower()
        if claim_type not in self._entity_claim_types() - {"behavior_pattern"}:
            return None
        subject = _compact_whitespace(raw.get("subject_entity") or "")
        subject_id = entity_ids.get(subject)
        predicate = re.sub(r"[^a-z0-9_]+", "_", str(raw.get("predicate") or "").lower()).strip("_")
        if not subject_id or not predicate:
            return None
        evidence_ids = list(dict.fromkeys(
            int(value) for value in (raw.get("evidence_fact_ids") or [])
            if str(value).strip().isdigit() and int(value) in facts_by_id
        ))[:12]
        if not evidence_ids or not all(
            subject in (facts_by_id[fact_id].get("entities") or [])
            for fact_id in evidence_ids
        ):
            return None
        object_name = _compact_whitespace(raw.get("object_entity") or "")
        object_id = entity_ids.get(object_name) if object_name else None
        raw_claim_text = _compact_whitespace(raw.get("claim_text") or "")
        if not object_id and not raw_claim_text:
            return None
        confidence = self._clamp_float(raw.get("confidence"), 0.0, 1.0, 0.7)
        claim_text = self._normalize_entity_claim_text(
            raw_claim_text,
            subject=subject,
            predicate=predicate,
            object_name=object_name,
        )
        if not claim_text:
            return None
        return {
            "subject_entity_id": subject_id,
            "predicate": predicate,
            "object_entity_id": object_id,
            "claim_text": claim_text,
            "claim_type": claim_type,
            "claim_origin": "explicit",
            "status": "active" if confidence >= self._entity_claim_explicit_min_confidence else "candidate",
            "confidence": confidence,
            "valid_from": _compact_whitespace(raw.get("valid_from") or ""),
            "valid_to": _compact_whitespace(raw.get("valid_to") or ""),
            "extractor_version": "entity_claim_explicit_v1",
            "prompt_version": "v1",
            "metadata": {"source_fact_count": len(evidence_ids)},
            "evidence_fact_ids": evidence_ids,
        }

    def _update_derived_entity_claims_from_explicit_claims(
        self,
        *,
        affected_claim_ids: Sequence[int],
    ) -> Dict[str, Any]:
        """Derive local conclusions from this batch's explicit-claim changes.

        The LLM proposes only premise-linked conclusions.  Entity IDs,
        premise activity, fact traceability, lifecycle transitions, and writes
        remain deterministic program responsibilities.
        """
        affected_ids = list(dict.fromkeys(
            int(value)
            for value in affected_claim_ids or []
            if str(value).strip().isdigit() and int(value) > 0
        ))
        report: Dict[str, Any] = {
            "enabled": 1,
            "affected_claim_count": len(affected_ids),
            "seed_claim_count": 0,
            "entity_group_count": 0,
            "input_claim_count": 0,
            "candidate_count": 0,
            "updated": 0,
            "created": 0,
            "invalidated": 0,
            "suppressed": 0,
            "completed": True,
        }
        if not affected_ids:
            return report

        # A premise may have been superseded during explicit reconciliation.
        # Invalidate its old proof before optionally creating a new proof from
        # the replacement explicit claim in this same pass.
        report["invalidated"] = self._invalidate_derived_claims_for_inactive_premises(
            affected_ids
        )
        seed_claims = [
            claim
            for claim in self._db.get_entity_claims_by_ids(affected_ids)
            if claim.get("claim_origin") == "explicit"
            and claim.get("status") == "active"
        ]
        report["seed_claim_count"] = len(seed_claims)
        if not seed_claims:
            return report

        changed_by_entity: Dict[int, List[Dict[str, Any]]] = {}
        for claim in seed_claims:
            endpoint_entity_ids = {
                int(entity_id)
                for entity_id in (
                    claim.get("subject_entity_id"),
                    claim.get("object_entity_id"),
                )
                if str(entity_id or "").strip().isdigit() and int(entity_id) > 0
            }
            for entity_id in endpoint_entity_ids:
                changed_by_entity.setdefault(entity_id, []).append(claim)
        report["entity_group_count"] = len(changed_by_entity)
        for entity_id, changed_claims in changed_by_entity.items():
            related_claims = self._retrieve_related_derived_explicit_claims(
                entity_id=entity_id,
                changed_claims=changed_claims,
            )
            report["input_claim_count"] += len(changed_claims) + len(related_claims)
            outcome = self._extract_derived_entity_claims(
                changed_claims=changed_claims,
                related_claims=related_claims,
            )
            if outcome is None:
                report["completed"] = False
                report["error"] = "invalid_llm_derived_claim_response"
                report["failed_entity_id"] = entity_id
                return report
            report["candidate_count"] += len(outcome)
            if not outcome:
                continue
            group_report = self._apply_derived_entity_claim_candidates(
                raw_candidates=outcome,
                changed_claims=changed_claims,
                related_claims=related_claims,
            )
            for key in ("updated", "created", "suppressed"):
                report[key] += int(group_report[key])
        return report

    def _invalidate_derived_claims_for_inactive_premises(
        self,
        premise_claim_ids: Sequence[int],
    ) -> int:
        """Propagate an explicit-premise lifecycle change to derived claims."""
        affected_derived_ids = self._db.invalidate_entity_claim_derivations_for_premises(
            premise_claim_ids
        )
        invalidated_claim_ids = self._db.get_entity_claim_ids_without_active_derivations(
            affected_derived_ids
        )
        changed_ids: List[int] = []
        for claim in self._db.get_entity_claims_by_ids(invalidated_claim_ids):
            if self._db.transition_entity_claim_status(
                target_claim_id=int(claim["id"]),
                new_status="invalidated",
                semantic_relation="premise_invalidated",
                decision_source="derived_claim_premise_lifecycle_v1",
                policy_reason="all_derivation_premises_inactive",
            ):
                changed_ids.append(int(claim["id"]))
        if changed_ids:
            self._sync_entity_claim_recall_documents(changed_ids)
        return len(changed_ids)

    def _retrieve_related_derived_explicit_claims(
        self,
        *,
        entity_id: int,
        changed_claims: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Load historical active explicit claims touching the anchor entity."""
        changed_claim_ids = {
            int(claim["id"])
            for claim in changed_claims
            if str(claim.get("id") or "").strip().isdigit()
        }
        claims_by_id: Dict[int, Dict[str, Any]] = {}
        for claim in self._db.get_entity_claims(
            entity_id=entity_id,
            claim_origin="explicit",
            statuses=["active"],
            limit=32,
        ):
            claim_id = int(claim["id"])
            if claim_id not in changed_claim_ids:
                claims_by_id[claim_id] = claim
        return sorted(
            claims_by_id.values(),
            key=lambda claim: -int(claim.get("id") or 0),
        )[:32]

    def _apply_derived_entity_claim_candidates(
        self,
        *,
        raw_candidates: Sequence[Dict[str, Any]],
        changed_claims: Sequence[Dict[str, Any]],
        related_claims: Sequence[Dict[str, Any]],
    ) -> Dict[str, int]:
        """Validate, reconcile, and record one subject's LLM conclusions."""
        report = {"updated": 0, "created": 0, "suppressed": 0}
        input_claims = [*changed_claims, *related_claims]
        input_by_id = {int(claim["id"]): claim for claim in input_claims}
        changed_claim_ids = {int(claim["id"]) for claim in changed_claims}
        entity_ids = {
            int(entity_id)
            for claim in input_claims
            for entity_id in (
                claim.get("subject_entity_id"), claim.get("object_entity_id"),
            )
            if str(entity_id or "").strip().isdigit() and int(entity_id) > 0
        }
        entity_names = self._db.get_entity_names_by_ids(sorted(entity_ids))
        candidates = [
            candidate
            for raw_claim in raw_candidates
            if (candidate := self._normalize_derived_entity_claim(
                raw_claim,
                input_claims_by_id=input_by_id,
                new_claim_ids=changed_claim_ids,
                entity_ids=entity_ids,
                entity_names=entity_names,
            )) is not None
        ]
        if not candidates:
            return report
        evidence_fact_ids = sorted({
            fact_id
            for candidate in candidates
            for fact_id in candidate["support_fact_ids"]
        })
        facts_by_id = {
            int(fact["id"]): fact
            for fact in self._db.get_memory_facts_by_ids(evidence_fact_ids)
            if str(fact.get("id") or "").strip().isdigit()
        }
        applied = self._reconcile_entity_claims(candidates, facts_by_id=facts_by_id)
        persisted_claims = {
            int(claim["id"]): claim
            for item in applied
            for claim in self._db.get_entity_claims_by_ids([item["claim_id"]])
            if str(claim.get("id") or "").strip().isdigit()
        }
        persisted_derived_ids: List[int] = []
        for item in applied:
            claim_id = int(item.get("claim_id") or 0)
            stored_claim = persisted_claims.get(claim_id)
            if not stored_claim or stored_claim.get("claim_origin") != "derived":
                report["suppressed"] += 1
                continue
            candidate = item["claim"]
            premise_ids = list(candidate["premise_claim_ids"])
            derivation_id = self._db.upsert_entity_claim_derivation(
                claim_id=claim_id,
                rule_id="llm_direct_entailment",
                rule_version="v1",
                derivation_key=(
                    f"llm_direct_entailment:v1:{claim_id}:"
                    + ",".join(str(value) for value in premise_ids)
                ),
                metadata={
                    "prompt_version": "v1",
                    "premise_claim_ids": premise_ids,
                    "generator": "derived_entity_claim_extraction",
                },
            )
            self._db.replace_entity_claim_derivation_premises(
                derivation_id=derivation_id,
                premise_claim_ids=premise_ids,
            )
            persisted_derived_ids.append(claim_id)
            report["created"] += int(item.get("created") or 0)
            report["updated"] += 1
        if persisted_derived_ids:
            self._sync_entity_claim_recall_documents(
                sorted(set(persisted_derived_ids))
            )
        return report

    def _derived_claim_prompt_view(
        self,
        claim: Dict[str, Any],
        *,
        entity_names: Dict[int, str],
    ) -> Dict[str, Any]:
        subject_id = int(claim.get("subject_entity_id") or 0)
        object_id = int(claim.get("object_entity_id") or 0)
        return {
            "id": int(claim["id"]),
            "subject_entity_id": subject_id,
            "subject_entity": entity_names.get(subject_id, ""),
            "predicate": claim.get("predicate") or "",
            "object_entity_id": object_id,
            "object_entity": entity_names.get(object_id, ""),
            "claim_type": claim.get("claim_type") or "",
            "claim_text": claim.get("claim_text") or "",
            "valid_from": claim.get("valid_from") or "",
            "valid_to": claim.get("valid_to") or "",
            "confidence": float(claim.get("confidence") or 0.0),
        }

    def _extract_derived_entity_claims(
        self,
        *,
        changed_claims: Sequence[Dict[str, Any]],
        related_claims: Sequence[Dict[str, Any]],
    ) -> Optional[List[Dict[str, Any]]]:
        input_claims = [*changed_claims, *related_claims]
        entity_ids = {
            int(entity_id)
            for claim in input_claims
            for entity_id in (
                claim.get("subject_entity_id"), claim.get("object_entity_id"),
            )
            if str(entity_id or "").strip().isdigit() and int(entity_id) > 0
        }
        entity_names = self._db.get_entity_names_by_ids(sorted(entity_ids))
        language = self._resolve_prompt_language_from_text(
            "\n".join(str(claim.get("claim_text") or "") for claim in input_claims)
        )
        prompt_template = (
            DERIVED_ENTITY_CLAIM_EXTRACTION_PROMPT_EN
            if language == "en" else DERIVED_ENTITY_CLAIM_EXTRACTION_PROMPT_ZH
        )
        raw = self._call_llm(prompt_template.replace(
            "{changed_claims}",
            json.dumps([
                self._derived_claim_prompt_view(
                    claim,
                    entity_names=entity_names,
                )
                for claim in changed_claims
            ], ensure_ascii=False, indent=2),
        ).replace(
            "{related_claims}",
            json.dumps([
                self._derived_claim_prompt_view(
                    claim,
                    entity_names=entity_names,
                )
                for claim in related_claims
            ], ensure_ascii=False, indent=2),
        ))
        parsed = self._parse_json_object_from_llm_text(raw or "")
        if parsed is None or not isinstance(parsed.get("claims"), list):
            return None
        return [item for item in parsed["claims"][:16] if isinstance(item, dict)]

    def _normalize_derived_entity_claim(
        self,
        raw: Dict[str, Any],
        *,
        input_claims_by_id: Dict[int, Dict[str, Any]],
        new_claim_ids: set[int],
        entity_ids: set[int],
        entity_names: Dict[int, str],
    ) -> Optional[Dict[str, Any]]:
        try:
            subject_entity_id = int(raw.get("subject_entity_id") or 0)
            object_entity_id = int(raw.get("object_entity_id") or 0)
        except (TypeError, ValueError):
            return None
        if subject_entity_id not in entity_ids or (
            object_entity_id and object_entity_id not in entity_ids
        ):
            return None
        claim_type = str(raw.get("claim_type") or "").strip().lower()
        if claim_type not in {"affiliation", "relationship", "constraint"}:
            return None
        predicate = re.sub(
            r"[^a-z0-9_]+", "_", str(raw.get("predicate") or "").lower(),
        ).strip("_")
        if not predicate:
            return None
        premise_ids = list(dict.fromkeys(
            int(value)
            for value in (raw.get("premise_claim_ids") or [])
            if str(value).strip().isdigit() and int(value) in input_claims_by_id
        ))[:8]
        if not premise_ids or not any(value in new_claim_ids for value in premise_ids):
            return None
        premises = [input_claims_by_id[value] for value in premise_ids]
        if not all(
            claim.get("claim_origin") == "explicit" and claim.get("status") == "active"
            for claim in premises
        ):
            return None
        conclusion_tuple = (
            subject_entity_id, predicate, object_entity_id, claim_type,
        )
        if any(
            conclusion_tuple == (
                int(claim.get("subject_entity_id") or 0),
                str(claim.get("predicate") or ""),
                int(claim.get("object_entity_id") or 0),
                str(claim.get("claim_type") or ""),
            )
            for claim in premises
        ):
            return None
        premise_confidence = min(
            float(claim.get("confidence") or 0.0) for claim in premises
        )
        confidence = min(
            self._clamp_float(raw.get("confidence"), 0.0, 1.0, 0.7),
            premise_confidence,
        )
        if confidence < self._entity_claim_derived_min_confidence:
            return None
        valid_from, valid_to = self._derived_claim_validity_bounds(premises)
        if valid_from and valid_to and valid_from > valid_to:
            return None
        support_fact_ids = sorted({
            fact_id
            for premise_id in premise_ids
            for fact_id in self._db.get_entity_claim_evidence_fact_ids([premise_id]).get(
                premise_id, []
            )
        })
        if not support_fact_ids:
            return None
        claim_text = self._normalize_entity_claim_text(
            raw.get("claim_text") or "",
            subject=entity_names.get(subject_entity_id, ""),
            predicate=predicate,
            object_name=entity_names.get(object_entity_id, ""),
        )
        if not claim_text:
            return None
        return {
            "subject_entity_id": subject_entity_id,
            "predicate": predicate,
            "object_entity_id": object_entity_id or None,
            "claim_text": claim_text,
            "claim_type": claim_type,
            "claim_origin": "derived",
            "status": "active",
            "confidence": confidence,
            "valid_from": valid_from,
            "valid_to": valid_to,
            "extractor_version": "entity_claim_derived_v1",
            "prompt_version": "v1",
            "metadata": {
                "derivation_type": "llm_direct_entailment",
                "premise_claim_ids": premise_ids,
            },
            "premise_claim_ids": premise_ids,
            "support_fact_ids": support_fact_ids,
        }

    @staticmethod
    def _derived_claim_validity_bounds(
        premises: Sequence[Dict[str, Any]],
    ) -> Tuple[str, str]:
        """Use the intersection of premise validity without inventing time."""
        starts = sorted({
            _compact_whitespace(claim.get("valid_from") or "")
            for claim in premises
            if _compact_whitespace(claim.get("valid_from") or "")
        })
        ends = sorted({
            _compact_whitespace(claim.get("valid_to") or "")
            for claim in premises
            if _compact_whitespace(claim.get("valid_to") or "")
        })
        return (starts[-1] if starts else "", ends[0] if ends else "")

    def _update_inductive_entity_claims_from_facts(
        self,
        seed_facts: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Use new facts to trigger bounded, exact-group evidence expansion."""
        report: Dict[str, Any] = {
            "enabled": 1, "seed_fact_count": len(seed_facts), "candidate_count": 0,
            "updated": 0, "created": 0, "completed": True,
        }
        if not seed_facts:
            return report
        groups: Dict[Tuple[int, str, str], Dict[str, Any]] = {}
        for fact in seed_facts:
            fact_id = int(fact.get("id") or 0)
            if fact_id <= 0:
                continue
            for signal_mapping in self._entity_claim_signals_from_fact(fact):
                key = (
                    int(signal_mapping["subject_entity_id"]),
                    str(signal_mapping["claim_type_hint"]),
                    str(signal_mapping["claim_anchor_key"]),
                )
                group = groups.setdefault(key, {
                    "subject": signal_mapping["subject"],
                    "subject_entity_id": int(signal_mapping["subject_entity_id"]),
                    "claim_type_hint": signal_mapping["claim_type_hint"],
                    "claim_anchor": signal_mapping["claim_anchor"],
                    "claim_anchor_key": signal_mapping["claim_anchor_key"],
                    "seed_facts_by_id": {},
                })
                group["seed_facts_by_id"][fact_id] = fact
        report["candidate_count"] = len(groups)
        for group in list(groups.values())[:12]:
            historical_facts = self._db.memory_facts_for_entity_claim_signal_group(
                subject_entity_id=int(group["subject_entity_id"]),
                claim_type_hint=str(group["claim_type_hint"]),
                claim_anchor_key=str(group["claim_anchor_key"]),
                limit=max(64, self._entity_claim_induction_evidence_limit * 4),
            )
            evidence_facts = self._select_inductive_entity_claim_evidence_facts(
                group=group,
                historical_facts=historical_facts,
            )
            distinct_episodes = {
                int(fact["episode_id"]) for fact in evidence_facts
                if str(fact.get("episode_id") or "").strip().isdigit()
            }
            time_windows = {
                str(fact.get("event_time_key") or fact.get("dialogue_time_key") or "")[:10]
                for fact in evidence_facts
                if str(fact.get("event_time_key") or fact.get("dialogue_time_key") or "")
            }
            if (
                len(evidence_facts) < self._entity_claim_induction_min_support_facts
                or len(distinct_episodes) < self._entity_claim_induction_min_episodes
                or len(time_windows) < self._entity_claim_induction_min_time_windows
            ):
                continue
            outcome = self._extract_inductive_entity_claims(
                group=group, facts=evidence_facts,
            )
            if outcome is None:
                report["completed"] = False
                report["error"] = "invalid_llm_induction_response"
                return report
            by_id = {int(fact["id"]): fact for fact in evidence_facts}
            applied = self._reconcile_entity_claims(outcome, facts_by_id=by_id)
            for item in applied:
                claim = item["claim"]
                if item["effective_origin"] != "inductive":
                    continue
                support_ids = self._entity_claim_support_ids(claim)
                if not support_ids:
                    continue
                claim_id = int(item["claim_id"])
                support_summary = self._db.get_entity_claim_support_fact_summary(
                    claim_id
                )
                if int(support_summary["support_count"]) <= 0:
                    continue
                self._db.upsert_entity_claim_induction(
                    claim_id=claim_id,
                    condition_text=str(claim.get("metadata", {}).get("condition_text") or ""),
                    support_count=int(support_summary["support_count"]),
                    first_observed_at=str(support_summary["first_observed_at"]),
                    last_observed_at=str(support_summary["last_observed_at"]),
                )
            report["updated"] += len(applied)
            report["created"] += sum(int(item["created"]) for item in applied)
        return report

    def _fact_entity_claim_signal_mapping_updates(
        self,
        fact_id: int,
        fact: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Project one fact's induction-eligible signals into exact group keys."""
        updates: List[Dict[str, Any]] = []
        fallback_entity = fact.get("primary_entity")
        raw_signals = self._normalize_entity_claim_signal(
            fact.get("entity_claim_signal"),
            fallback_entity=fallback_entity,
        )
        for signal in raw_signals:
            claim_type_hint = str(signal.get("claim_type_hint") or "").lower()
            signal_kind = str(signal.get("signal_kind") or "").lower()
            if claim_type_hint not in {"preference", "behavior_pattern"}:
                continue
            if signal_kind not in {"explicit_assertion", "pattern_observation"}:
                continue
            names = self._entities_for_entity_claim_signal(signal, fact)
            entity_mapping = self._db.add_entity_names(names)
            if not names or names[0] not in entity_mapping:
                continue
            claim_anchor = _compact_whitespace(signal.get("claim_anchor") or "")
            if not claim_anchor:
                continue
            claim_anchor_key = self._generate_topic_name_key(claim_anchor)
            if not claim_anchor_key:
                continue
            updates.append({
                "fact_id": int(fact_id),
                "signal_key": (
                    f"{int(entity_mapping[names[0]])}|{claim_type_hint}|"
                    f"{claim_anchor_key}"
                ),
                "subject": names[0],
                "subject_entity_id": int(entity_mapping[names[0]]),
                "claim_type_hint": claim_type_hint,
                "signal_kind": signal_kind,
                "claim_anchor": claim_anchor,
                "claim_anchor_key": claim_anchor_key,
                "confidence": float(signal.get("confidence") or 0.0),
            })
        return updates or [{
            "fact_id": int(fact_id),
            "signal_key": "__fact__",
        }]

    def _fact_future_commitment_signal_mapping_updates(
        self,
        fact_id: int,
        fact: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Project one fact's normalized prospective signals into DB rows."""
        updates: List[Dict[str, Any]] = []
        raw_signals = self._normalize_future_commitment_signals(
            fact.get("future_commitment_signals"),
            fallback_entity=fact.get("primary_entity"),
        )
        for signal in raw_signals:
            subject_name = _compact_whitespace(signal.get("subject_entity") or "")
            entity_mapping = self._db.add_entity_names([subject_name])
            if not subject_name or subject_name not in entity_mapping:
                continue
            future_commitment_anchor = _compact_whitespace(
                signal.get("future_commitment_anchor") or ""
            )
            future_commitment_anchor_key = self._generate_topic_name_key(future_commitment_anchor)
            if not future_commitment_anchor or future_commitment_anchor_key == "general":
                continue
            updates.append({
                "fact_id": int(fact_id),
                "signal_key": (
                    f"{int(entity_mapping[subject_name])}|{signal['evidence_kind']}|"
                    f"{future_commitment_anchor_key}"
                ),
                "subject_entity_id": int(entity_mapping[subject_name]),
                "evidence_kind": signal["evidence_kind"],
                "candidate_object_types": signal["candidate_object_types"],
                "user_role": signal["user_role"],
                "future_commitment_anchor": future_commitment_anchor,
                "future_commitment_anchor_key": future_commitment_anchor_key,
                "assertion_source": signal["assertion_source"],
                "explicitness": signal["explicitness"],
                "evidence_basis": signal["evidence_basis"],
                "confidence": signal["confidence"],
            })
        return updates or [{
            "fact_id": int(fact_id),
            "signal_key": "__fact__",
        }]

    def _select_inductive_entity_claim_evidence_facts(
        self,
        *,
        group: Dict[str, Any],
        historical_facts: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Keep new group facts, then add bounded episode-diverse history."""
        seed_by_id = dict(group.get("seed_facts_by_id") or {})
        candidates_by_id = {
            int(fact["id"]): fact
            for fact in [*seed_by_id.values(), *historical_facts]
            if str(fact.get("id") or "").strip().isdigit()
        }
        seed_ids = set(seed_by_id)

        ordered = sorted(
            candidates_by_id.values(),
            key=lambda fact: (
                str(fact.get("event_time_key") or fact.get("dialogue_time_key") or ""),
                int(fact.get("id") or 0),
            ),
            reverse=True,
        )
        ordered.sort(
            key=lambda fact: int(fact.get("id") or 0) not in seed_ids
        )
        selected: List[Dict[str, Any]] = []
        selected_ids: set[int] = set()
        selected_episodes: set[int] = set()
        for fact in ordered:
            fact_id = int(fact["id"])
            episode_id = int(fact.get("episode_id") or 0)
            if episode_id > 0 and episode_id in selected_episodes:
                continue
            selected.append(fact)
            selected_ids.add(fact_id)
            if episode_id > 0:
                selected_episodes.add(episode_id)
            if len(selected) >= self._entity_claim_induction_evidence_limit:
                break
        if len(selected) < self._entity_claim_induction_evidence_limit:
            for fact in ordered:
                fact_id = int(fact["id"])
                if fact_id in selected_ids:
                    continue
                selected.append(fact)
                selected_ids.add(fact_id)
                if len(selected) >= self._entity_claim_induction_evidence_limit:
                    break
        return sorted(
            selected,
            key=lambda fact: (
                str(fact.get("event_time_key") or fact.get("dialogue_time_key") or ""),
                int(fact.get("id") or 0),
            ),
        )

    def _extract_inductive_entity_claims(
        self,
        *,
        group: Dict[str, Any],
        facts: Sequence[Dict[str, Any]],
    ) -> Optional[List[Dict[str, Any]]]:
        language = self._resolve_prompt_language_from_text(
            "\n".join(str(fact.get("summary") or "") for fact in facts[:16])
        )
        prompt_template = (
            INDUCTIVE_ENTITY_CLAIM_EXTRACTION_PROMPT_EN
            if language == "en" else INDUCTIVE_ENTITY_CLAIM_EXTRACTION_PROMPT_ZH
        )
        raw = self._call_llm(
            prompt_template.replace("{induction_target}", json.dumps({
                "subject_entity": group["subject"],
                "claim_type_hint": group["claim_type_hint"],
                "claim_anchor": group["claim_anchor"],
            }, ensure_ascii=False, indent=2)).replace("{facts}", json.dumps([
                self._claim_fact_prompt_view(fact) for fact in facts[:32]
            ], ensure_ascii=False, indent=2))
        )
        parsed = self._parse_json_object_from_llm_text(raw or "")
        if parsed is None or not isinstance(parsed.get("claims"), list):
            return None
        facts_by_id = {
            int(fact["id"]): fact for fact in facts
            if str(fact.get("id") or "").strip().isdigit()
        }
        result: List[Dict[str, Any]] = []
        for raw_claim in parsed["claims"][:4]:
            claim = self._normalize_inductive_entity_claim(
                raw_claim, group=group, facts_by_id=facts_by_id,
            )
            if claim:
                result.append(claim)
        return result

    def _normalize_inductive_entity_claim(
        self,
        raw: Any,
        *,
        group: Dict[str, Any],
        facts_by_id: Dict[int, Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(raw, dict):
            return None
        claim_type = str(raw.get("claim_type") or "").strip().lower()
        if claim_type not in {"preference", "behavior_pattern"}:
            return None
        if _compact_whitespace(raw.get("subject_entity") or "") != group["subject"]:
            return None
        predicate = re.sub(r"[^a-z0-9_]+", "_", str(raw.get("predicate") or "").lower()).strip("_")
        if predicate not in {"prefers", "dislikes", "usually_does", "avoids", "has_routine"}:
            return None
        support_ids = list(dict.fromkeys(
            int(value) for value in (raw.get("support_fact_ids") or [])
            if str(value).strip().isdigit() and int(value) in facts_by_id
        ))[:24]
        episode_ids = {
            int(facts_by_id[fact_id]["episode_id"]) for fact_id in support_ids
            if str(facts_by_id[fact_id].get("episode_id") or "").strip().isdigit()
        }
        windows = {
            str(facts_by_id[fact_id].get("event_time_key") or facts_by_id[fact_id].get("dialogue_time_key") or "")[:10]
            for fact_id in support_ids
        } - {""}
        if (
            len(support_ids) < self._entity_claim_induction_min_support_facts
            or len(episode_ids) < self._entity_claim_induction_min_episodes
            or len(windows) < self._entity_claim_induction_min_time_windows
        ):
            return None
        raw_claim_text = _compact_whitespace(raw.get("claim_text") or "")
        if not raw_claim_text:
            return None
        confidence = self._clamp_float(raw.get("confidence"), 0.0, 1.0, 0.7)
        claim_text = self._normalize_entity_claim_text(
            raw_claim_text,
            subject=group["subject"],
            predicate=predicate,
        )
        return {
            "subject_entity_id": group["subject_entity_id"], "predicate": predicate,
            "claim_text": claim_text,
            "claim_type": claim_type, "claim_origin": "inductive",
            "status": "active",
            "confidence": confidence,
            "extractor_version": "entity_claim_induction_v1", "prompt_version": "v1",
            "metadata": {
                "condition_text": _compact_whitespace(raw.get("condition_text") or ""),
                "claim_anchor": group["claim_anchor"],
            },
            "support_fact_ids": support_ids,
        }

    @staticmethod
    def _normalize_entity_claim_text(
        value: Any,
        *,
        subject: str,
        predicate: str,
        object_name: str = "",
    ) -> str:
        """Return the claim's complete, reader-facing semantic proposition."""
        text = _compact_whitespace(value or "")[:480]
        if text:
            return text
        target = _compact_whitespace(object_name)
        return _compact_whitespace(f"{subject} {predicate} {target}")[:480]

    def _log_future_commitment_facts_loaded(
        self,
        processing_target: str,
        facts: List[Dict[str, Any]],
        limit: int,
        reference_timestamp: Any,
    ) -> None:
        """Log the independent fact batch consumed by future-commitment update."""
        self._log_info("memory_future_commitment_update", "facts_loaded", {
            "processing_target": processing_target,
            "fact_count": len(facts),
            "fact_ids": [fact.get("id") for fact in facts],
            "limit": limit,
            "reference_timestamp": reference_timestamp,
            "time_start": facts[0].get("dialogue_time_key") if facts else "",
            "time_end": facts[-1].get("dialogue_time_key") if facts else "",
        })

    @staticmethod
    def _generate_topic_name_key(value: Any) -> str:
        text = _compact_whitespace(value).lower()
        text = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "", text)
        return text or "general"

    @staticmethod
    def _normalize_unique_labels(
        values: Sequence[Any],
        *,
        limit: int = 20,
    ) -> List[str]:
        labels: List[str] = []
        seen: set[str] = set()
        for value in values:
            label = _compact_whitespace(value)
            if not label:
                continue
            key = label.lower()
            if key in seen:
                continue
            seen.add(key)
            labels.append(label)
            if len(labels) >= limit:
                break
        return labels

    def _topic_name_best_pair_similarity(
        self,
        left_aliases: Sequence[str],
        right_aliases: Sequence[str],
        *,
        allow_substring: bool = True,
    ) -> float:
        """Return the strongest similarity between any two topic aliases."""
        best = 0.0
        for left in left_aliases:
            left_key = self._generate_topic_name_key(left)
            for right in right_aliases:
                right_key = self._generate_topic_name_key(right)
                if left_key and right_key and left_key == right_key:
                    return 1.0
                if (
                    allow_substring
                    and left_key
                    and right_key
                    and (left_key in right_key or right_key in left_key)
                ):
                    best = max(best, 0.9)
                left_terms = set(self._topic_similarity_terms(str(left)))
                right_terms = set(self._topic_similarity_terms(str(right)))
                if not left_terms or not right_terms:
                    continue
                shared_terms = left_terms & right_terms
                if not shared_terms:
                    continue
                jaccard = len(shared_terms) / max(1, len(left_terms | right_terms))
                best = max(best, jaccard)

                # A shorter topic can be a lexical specialization of a
                # longer topic, e.g. "手机推广策略" and
                # "新手机产品推广策略". Require at least two shared
                # tokens so a single generic word cannot create a strong
                # match by itself.
                if len(shared_terms) >= 2:
                    left_coverage = len(shared_terms) / max(1, len(left_terms))
                    right_coverage = len(shared_terms) / max(1, len(right_terms))
                    best = max(best, left_coverage, right_coverage)
        return best

    def _recall_calculate_search_terms_overlap_with_topic_values(
        self,
        query_terms: Sequence[str],
        topic_values: Sequence[str],
        *,
        allow_substring: bool = True,
        minimum_pair_score: float = 0.5,
    ) -> Dict[str, Any]:
        """Measure how many distinct query-side topic terms a candidate covers.

        Unlike ``_topic_name_best_pair_similarity``, this is intentionally
        asymmetric: a candidate with many aliases cannot inflate its score.
        Each distinct query term contributes at most one match when any topic
        value provides sufficiently strong exact, substring, or token-overlap
        evidence.
        """
        normalized_terms: List[Tuple[str, str]] = []
        seen_term_keys: set[str] = set()
        for value in query_terms or ():
            term = self._recall_stage1_clean_anchor(value)
            term_key = self._generate_topic_name_key(term) if term else ""
            if not term or not term_key or term_key in seen_term_keys:
                continue
            seen_term_keys.add(term_key)
            normalized_terms.append((term, term_key))
        # Search-mode tokenization can emit both a compound term and its
        # nested fragments. Only the longest form should contribute to
        # query-side coverage, otherwise one topic phrase is counted several
        # times as independent evidence.
        normalized_terms = [
            (term, term_key)
            for term, term_key in normalized_terms
            if not any(
                term_key != other_key
                and len(term_key) < len(other_key)
                and term_key in other_key
                for _other_term, other_key in normalized_terms
            )
        ]

        normalized_topics: List[Tuple[str, str]] = []
        seen_topic_keys: set[str] = set()
        for value in topic_values or ():
            topic = self._recall_stage1_clean_anchor(value)
            topic_key = self._generate_topic_name_key(topic) if topic else ""
            if not topic or not topic_key or topic_key in seen_topic_keys:
                continue
            seen_topic_keys.add(topic_key)
            normalized_topics.append((topic, topic_key))

        matched_terms: List[str] = []
        matched_topic_values: List[str] = []
        best_pair_score = 0.0
        required_pair_score = self._clamp_float(
            minimum_pair_score,
            0.0,
            1.0,
            0.5,
        )
        for term, _term_key in normalized_terms:
            best_topic = ""
            best_term_score = 0.0
            for topic, _topic_key in normalized_topics:
                pair_score = self._topic_name_best_pair_similarity(
                    [term],
                    [topic],
                    allow_substring=allow_substring,
                )
                if pair_score > best_term_score:
                    best_term_score = pair_score
                    best_topic = topic
            best_pair_score = max(best_pair_score, best_term_score)
            if best_term_score < required_pair_score:
                continue
            matched_terms.append(term)
            if best_topic and best_topic not in matched_topic_values:
                matched_topic_values.append(best_topic)

        term_count = len(normalized_terms)
        matched_term_count = len(matched_terms)
        return {
            "matched_term_count": matched_term_count,
            "term_count": term_count,
            "coverage": round(matched_term_count / term_count, 4)
            if term_count
            else 0.0,
            "matched_terms": matched_terms,
            "matched_topic_values": matched_topic_values,
            "best_pair_score": round(best_pair_score, 4),
        }

    def _recall_calculate_search_terms_overlap_with_candidate_topics(
        self,
        candidate: Dict[str, Any],
        query_terms: Sequence[str],
        *,
        allow_substring: bool = True,
        minimum_pair_score: float = 0.5,
    ) -> Dict[str, Any]:
        """Extract one candidate's canonical topic values and score overlap."""
        raw = (
            candidate.get("_hydrated")
            if isinstance(candidate.get("_hydrated"), dict)
            else {}
        )
        topic_values: List[str] = []

        def extend_values(value: Any) -> None:
            if isinstance(value, (list, tuple, set)):
                for item in value:
                    extend_values(item)
                return
            topic = self._recall_stage1_clean_anchor(value)
            if topic and topic not in topic_values:
                topic_values.append(topic)

        extend_values(raw.get("fact_root_topic"))
        extend_values(raw.get("fact_aspect_topic"))

        overlap = self._recall_calculate_search_terms_overlap_with_topic_values(
            query_terms,
            topic_values,
            allow_substring=allow_substring,
            minimum_pair_score=minimum_pair_score,
        )
        return {
            **overlap,
            "topic_values": topic_values,
        }

    def _recall_calculate_search_terms_overlap_with_candidate_keywords(
        self,
        candidate: Dict[str, Any],
        query_terms: Sequence[str],
        *,
        allow_substring: bool = True,
        minimum_pair_score: float = 0.5,
    ) -> Dict[str, Any]:
        """Measure query-term overlap with one fact's extracted keywords.

        Keywords are a supplementary, fact-only signal.  They intentionally
        remain separate from canonical topic matching so callers can use them
        for ranking without treating them as a topic strong anchor.
        """
        if str(candidate.get("index_level") or "").strip().lower() != "fact":
            overlap = self._recall_calculate_search_terms_overlap_with_topic_values(
                query_terms,
                [],
                allow_substring=allow_substring,
                minimum_pair_score=minimum_pair_score,
            )
            return {
                key: value
                for key, value in overlap.items()
                if key != "matched_topic_values"
            } | {
                "matched_keyword_values": list(
                    overlap.get("matched_topic_values") or []
                ),
                "keyword_values": [],
            }

        raw = (
            candidate.get("_hydrated")
            if isinstance(candidate.get("_hydrated"), dict)
            else {}
        )
        raw_keywords = raw.get("keywords") or candidate.get("keywords") or ""
        keyword_values: List[str] = []
        seen_keyword_keys: set[str] = set()

        def add_keyword(value: Any) -> None:
            keyword = self._recall_stage1_clean_anchor(value)
            keyword_key = self._generate_topic_name_key(keyword) if keyword else ""
            if not keyword or not keyword_key or keyword_key in seen_keyword_keys:
                return
            seen_keyword_keys.add(keyword_key)
            keyword_values.append(keyword)

        keywords_are_structured = isinstance(raw_keywords, (list, tuple, set))
        raw_values = (
            raw_keywords
            if keywords_are_structured
            else re.split(r"[,，;；\n]+", str(raw_keywords))
        )
        for raw_value in raw_values:
            add_keyword(raw_value)
            # Legacy facts persist a whitespace-joined string. New facts use
            # a structured list so multi-word keywords retain their boundary.
            if not keywords_are_structured and isinstance(raw_value, str):
                for token in raw_value.split():
                    add_keyword(token)

        overlap = self._recall_calculate_search_terms_overlap_with_topic_values(
            query_terms,
            keyword_values,
            allow_substring=allow_substring,
            minimum_pair_score=minimum_pair_score,
        )
        return {
            key: value
            for key, value in overlap.items()
            if key != "matched_topic_values"
        } | {
            "matched_keyword_values": list(
                overlap.get("matched_topic_values") or []
            ),
            "keyword_values": keyword_values,
        }

    def _entity_claim_signals_from_fact(self, fact: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Read claim-level signals from their normalized fact mapping rows."""
        fact_id = int(fact.get("id") or 0)
        if fact_id <= 0:
            return []
        return list(
            self._db.get_fact_entity_claim_signal_mappings(fact_id).values()
        )

    def _future_commitment_signals_from_fact(self, fact: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Read prospective evidence solely from its normalized mapping rows."""
        fact_id = int(fact.get("id") or 0)
        if fact_id <= 0:
            return []
        return list(
            self._db.get_fact_future_commitment_signal_mappings(fact_id).values()
        )

    def _entities_for_entity_claim_signal(
        self,
        signal: Dict[str, Any],
        fact: Dict[str, Any],
    ) -> List[str]:
        entity = signal.get("entity") or signal.get("primary_entity")
        if isinstance(entity, dict):
            name = _compact_whitespace(entity.get("name") or entity.get("text") or "")
        else:
            name = _compact_whitespace(entity)
        if name:
            return [name]
        return self._entities_for_entity_claim_fact(fact)

    def _entities_for_entity_claim_fact(
        self,
        fact: Dict[str, Any],
    ) -> List[str]:
        metadata = fact.get("metadata") if isinstance(fact.get("metadata"), dict) else {}
        primary_entity = fact.get("primary_entity")
        if isinstance(primary_entity, dict):
            primary_name = _compact_whitespace(
                primary_entity.get("name") or primary_entity.get("text") or ""
            )
        else:
            primary_name = _compact_whitespace(primary_entity)
        if primary_name:
            return [primary_name]

        entities = [
            _compact_whitespace(value)
            for value in (fact.get("entities") or [])
            if _compact_whitespace(value)
        ]
        out: List[str] = []
        seen: set[str] = set()
        for entity in entities:
            clean = _compact_whitespace(entity)
            if not clean:
                continue
            key = clean.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(clean)
        return out[:1]

    def _should_route_fact_to_future_commitment_update(self, fact: Dict[str, Any]) -> bool:
        """Return whether a fact merits semantic review by the intent task.

        This is deliberately a high-recall, deterministic intake gate.  It does
        not classify a fact as a goal, plan, or work item, and it must not make
        database changes.  A future intent-extraction task will make that
        decision using the fact, its source context, and related active objects.
        """
        if not isinstance(fact, dict):
            return False
        summary = _compact_whitespace(fact.get("summary") or fact.get("text") or "")
        if not summary:
            return False

        fact_type = self._normalize_fact_type(fact.get("fact_type"))
        if fact_type == "recommendation":
            # Assistant suggestions are not user intent unless a separate fact
            # records the user's explicit acceptance or commitment.
            return False

        text = summary.lower()

        lifecycle_markers = (
            "已完成", "完成了", "做完", "办完", "已经发", "已发", "提交了",
            "已提交", "取消", "不做了", "不去了", "改期", "改到", "延期", "推迟",
            "延后", "提前", "受阻", "卡住", "无法", "没法", "等待确认", "等待资料",
            "completed", "finished", "done", "sent", "submitted", "cancelled",
            "canceled", "rescheduled", "postponed", "delayed", "blocked",
            "cannot proceed", "waiting for confirmation",
        )
        if any(marker in text for marker in lifecycle_markers):
            return True

        if fact_type == "commitment":
            return True

        goal_markers = (
            "长期目标", "目标是", "目标为", "希望达到", "希望实现", "争取", "致力于",
            "今年完成", "今年实现", "年内完成", "未来要实现", "want to achieve",
            "hope to achieve", "aim to", "goal is", "long-term goal", "work toward",
        )
        if any(marker in text for marker in goal_markers):
            return True

        responsibility_markers = (
            "需要完成", "必须", "负责", "答应", "承诺", "交付", "提交",
            "交给", "让我", "分配给我", "安排我", "请提醒", "帮我提醒",
            "need to", "have to", "must", "responsible for", "committed to",
            "promised to", "deliver", "submit", "assigned me", "remind me",
        )
        if any(marker in text for marker in responsibility_markers):
            return True

        future_time_markers = (
            "明天", "后天", "今晚", "明晚", "下周", "下个月", "明年", "本周",
            "这周", "周一", "周二", "周三", "周四", "周五", "周六", "周日", "周末",
            "tomorrow", "tonight", "next week", "next month", "next year",
            "this week", "on monday", "on tuesday", "on wednesday", "on thursday",
            "on friday", "this weekend",
        )
        future_activity_markers = (
            "前往", "去", "参加", "出席", "开会", "会面", "见面", "拜访", "出差",
            "旅行", "约", "预约", "安排", "活动", "会议", "飞", "乘车",
            "travel", "go to", "attend", "meet", "visit", "schedule", "book",
            "appointment", "trip", "flight", "meeting",
        )
        if (
            any(marker in text for marker in future_time_markers)
            and any(marker in text for marker in future_activity_markers)
        ):
            return True

        decision_markers = (
            "决定", "打算", "准备", "计划", "将要", "要去", "要做",
            "decided", "planning to", "plan to", "going to", "will ",
        )
        return fact_type in {"decision", "action", "instruction"} and any(
            marker in text for marker in decision_markers
        )

    @staticmethod
    def _is_low_value_followup_question(text: str) -> bool:
        lower = str(text or "").lower()
        if any(marker in lower for marker in ("提醒我", "帮我提醒", "请提醒", "帮我记", "请记住", "remind me", "remember this")):
            return False
        return any(
            marker in lower
            for marker in (
                "未明确接受", "未明确拒绝", "未明确回应", "是否愿意尝试",
                "是否采纳", "是否接受", "是否愿意", "用户未明确",
                "not explicitly accepted", "not explicitly rejected",
                "did not clearly respond", "whether the user is willing to try",
                "whether the user accepts",
            )
        )
    
    @staticmethod
    def _clamp_float(value: Any, low: float, high: float, default: float) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = default
        return max(low, min(high, number))

    def _log_info(self, scope: str, event: str, payload: Dict[str, Any]) -> None:
        record = {
            "scope": scope,
            "event": event,
            "payload": payload,
        }
        try:
            body = json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=False,
                indent=2,
                default=str,
            )
        except (TypeError, ValueError):
            body = json.dumps(
                {
                    "scope": scope,
                    "event": event,
                    "payload": str(payload),
                },
                ensure_ascii=False,
                sort_keys=False,
                indent=2,
            )
        self._logger.info("\n%s", body)

    @staticmethod
    def _format_log_text(value: Any, *, limit: int = 500) -> str:
        text = _compact_whitespace(value)
        if limit <= 0 or len(text) <= limit:
            return text
        return text[:limit] + "...[truncated]"

    @staticmethod
    def _normalize_recall_time_bound(value: Any, *, default_to_now: bool = False) -> str:
        if isinstance(value, datetime):
            return value.replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S")
        text = str(value or "").strip()
        if not text:
            return datetime.now().strftime("%Y-%m-%d %H:%M:%S") if default_to_now else ""
        normalized = text.replace("T", " ")
        if normalized.endswith("Z"):
            normalized = normalized[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(normalized).replace(tzinfo=None).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        except ValueError:
            pass
        match = re.search(r"(\d{4}-\d{1,2}-\d{1,2})\s+(\d{1,2}:\d{1,2}:\d{1,2})", normalized)
        if match:
            try:
                return datetime.strptime(
                    f"{match.group(1)} {match.group(2)}",
                    "%Y-%m-%d %H:%M:%S",
                ).strftime("%Y-%m-%d %H:%M:%S")
            except ValueError:
                pass
        match = re.search(r"(\d{4}-\d{1,2}-\d{1,2})", normalized)
        if match:
            try:
                return datetime.strptime(match.group(1), "%Y-%m-%d").strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            except ValueError:
                pass
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S") if default_to_now else text

    def _recall_log_candidate_items(
        self,
        items: Sequence[Dict[str, Any]],
        *,
        detailed: bool = False,
        limit: Optional[int] = 12,
        stage: str = "",
    ) -> Dict[str, Any]:
        """Serialize recall candidates for compact or detailed diagnostics."""
        candidates = list(items or [])
        visible_candidates = (
            candidates
            if limit is None
            else candidates[:max(0, int(limit or 0))]
        )
        rows: List[Dict[str, Any]] = []
        for item in visible_candidates:
            raw = item.get("_hydrated") if isinstance(item.get("_hydrated"), dict) else {}
            summary = (
                raw.get("summary")
                or item.get("summary_for_retrieval")
                or item.get("summary")
                or item.get("title")
                or ""
            )
            score_details_key = (
                "_recall_stage2_match_details"
                if str(stage or "").startswith("stage2")
                else "_recall_fast_match_details"
            )
            score_details = item.get(score_details_key)
            if not isinstance(score_details, dict) and not stage:
                score_details = item.get("_recall_stage2_match_details")
            score_details_present = isinstance(score_details, dict)
            score_details = (
                score_details if isinstance(score_details, dict) else {}
            )
            score_components = dict(
                score_details.get("score_components") or {}
            )
            if not score_details_present:
                score_components = dict(
                    item.get("_recall_score_components") or {}
                )
            row: Dict[str, Any] = {
                "target": f"{item.get('target_table')}#{item.get('target_id')}",
                "level": item.get("index_level") or item.get("_recall_type"),
                "score": item.get("_recall_score"),
                "rank": item.get("_recall_rank"),
                "embedding_similarity": item.get("embedding_similarity"),
                "bm25_score": (
                    item.get("_recall_bm25_score")
                    if item.get("_recall_bm25_score") is not None
                    else score_components.get("bm25_component_score")
                ),
                "score_components": score_components,
                "has_strong_anchor": bool(item.get("has_strong_anchor")),
                "strong_anchor_reasons": list(
                    item.get("strong_anchor_reasons") or []
                ),
                "time_start": item.get("time_start"),
                "summary": self._format_log_text(summary, limit=240),
            }
            if detailed:
                match_details_key = (
                    "_recall_stage2_match_details"
                    if str(stage or "").startswith("stage2")
                    else "_recall_fast_match_details"
                )
                match_details = item.get(match_details_key)
                if not isinstance(match_details, dict) and not stage:
                    match_details = item.get("_recall_stage2_match_details")
                match_details = (
                    dict(match_details)
                    if isinstance(match_details, dict)
                    else {}
                )
                decision = item.get("_recall_decision")
                decision = dict(decision) if isinstance(decision, dict) else {}
                source = str(item.get("_recall_candidate_source") or "")
                reasons: List[str] = []
                for value in (
                    item.get("evidence")
                    or item.get("_recall_fast_match_evidence")
                    or []
                ):
                    if str(value) and str(value) not in reasons:
                        reasons.append(str(value))
                for value in self._recall_candidate_source_channels(source):
                    if value not in reasons:
                        reasons.append(value)
                decision_reason = (
                    decision.get("decision_reason")
                    or match_details.get("filter_reason")
                    or item.get("_recall_drop_reason")
                    or ""
                )
                row.update({
                    "stage": stage,
                    "candidate_source": source,
                    "retrieval_reasons": reasons,
                    "title": self._format_log_text(
                        item.get("title") or raw.get("canonical_name") or "",
                        limit=240,
                    ),
                    "summary": self._format_log_text(summary, limit=500),
                    "identity_text": self._format_log_text(
                        item.get("identity_text") or raw.get("identity_text") or "",
                        limit=500,
                    ),
                    "time_end": item.get("time_end"),
                    "match": match_details,
                    "bm25_raw_score": item.get("_bm25_score"),
                    "bm25_score": item.get("_recall_bm25_score"),
                    "episode_seed_targets": list(
                        item.get("_stage2_episode_seed_targets") or []
                    ),
                    "accepted": decision.get("accepted"),
                    "decision_reason": decision_reason,
                })
            rows.append(row)
        return {
            "count": len(candidates),
            "items": rows,
        }

    @staticmethod
    def _parse_time_expression(
        query: str,
        *,
        reference_time: Optional[str] = None,
    ) -> Tuple[Optional[str], str, str]:
        """Parse lightweight time expressions from a recall query.

        This mirrors the voice_recording recall path but uses the recall
        timestamp as the relative-time anchor when one is provided. That keeps
        benchmark queries anchored to the question date instead of wall-clock
        time.
        """

        def parse_reference_time(value: Optional[str]) -> datetime:
            text = str(value or "").strip()
            if not text:
                return datetime.now()
            normalized = text.replace("T", " ")
            if normalized.endswith("Z"):
                normalized = normalized[:-1] + "+00:00"
            try:
                return datetime.fromisoformat(normalized)
            except ValueError:
                pass
            candidates = [normalized, normalized[:19], normalized[:16], normalized[:10]]
            for candidate in candidates:
                for fmt_text in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d %H:%M"):
                    try:
                        return datetime.strptime(candidate, fmt_text)
                    except ValueError:
                        continue
            return datetime.now()

        def fmt(value: datetime) -> str:
            return value.replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S")

        text = str(query or "")
        now = parse_reference_time(reference_time)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        clean_query = text

        m = re.search(r"(?:最近|近|过去)\s*(\d+)\s*(天|日|周|星期|个月|月|年)", text)
        if m:
            num = int(m.group(1))
            unit = m.group(2)
            if unit in ("天", "日"):
                delta = timedelta(days=num)
            elif unit in ("周", "星期"):
                delta = timedelta(weeks=num)
            elif unit in ("个月", "月"):
                delta = timedelta(days=num * 30)
            elif unit == "年":
                delta = timedelta(days=num * 365)
            else:
                delta = timedelta(days=num)
            clean_query = text[:m.start()] + text[m.end():]
            return fmt(now - delta), fmt(now), clean_query.strip()

        m = re.search(r"\b(?:last|past|previous|recent)\s+(\d+)\s+(day|days|week|weeks|month|months|year|years)\b", text, re.IGNORECASE)
        if m:
            num = int(m.group(1))
            unit = m.group(2).lower()
            if unit.startswith("day"):
                delta = timedelta(days=num)
            elif unit.startswith("week"):
                delta = timedelta(weeks=num)
            elif unit.startswith("month"):
                delta = timedelta(days=num * 30)
            else:
                delta = timedelta(days=num * 365)
            clean_query = text[:m.start()] + text[m.end():]
            return fmt(now - delta), fmt(now), clean_query.strip()

        m = re.search(r"最近\s*", text)
        if m:
            clean_query = text[:m.start()] + text[m.end():]
            return fmt(now - timedelta(days=7)), fmt(now), clean_query.strip()

        m = re.search(r"\b(?:recently|lately)\b", text, re.IGNORECASE)
        if m:
            clean_query = text[:m.start()] + text[m.end():]
            return fmt(now - timedelta(days=7)), fmt(now), clean_query.strip()

        # Parse a day-level Chinese date range before the single-date branch.
        # The end bound is exclusive, so a query covering Apr 27 through Apr
        # 29 searches until the start of Apr 30.
        m = re.search(
            r"(?<!\d)"
            r"(?:(?P<year1>\d{4})\s*年\s*)?"
            r"(?P<month1>\d{1,2})\s*月\s*(?P<day1>\d{1,2})\s*(?:日|号)?"
            r"\s*(?:到|至|[-~～])\s*"
            r"(?:(?P<year2>\d{4})\s*年\s*)?"
            r"(?P<month2>\d{1,2})\s*月\s*(?P<day2>\d{1,2})\s*(?:日|号)?"
            r"(?!\d)",
            text,
        )
        if m:
            year1_text = m.group("year1")
            year2_text = m.group("year2")
            month1 = int(m.group("month1"))
            day1 = int(m.group("day1"))
            month2 = int(m.group("month2"))
            day2 = int(m.group("day2"))
            year1 = int(year1_text) if year1_text else now.year
            if not year1_text:
                try:
                    if datetime(year1, month1, day1).date() > now.date():
                        year1 -= 1
                except ValueError:
                    pass
            if year2_text:
                year2 = int(year2_text)
            elif year1_text:
                year2 = year1
            else:
                year2 = year1 + (1 if (month2, day2) < (month1, day1) else 0)
            try:
                start = datetime(year1, month1, day1)
                end = datetime(year2, month2, day2) + timedelta(days=1)
                if end > start:
                    clean_query = text[:m.start()] + text[m.end():]
                    return fmt(start), fmt(end), clean_query.strip()
            except ValueError:
                pass

        m = re.search(r"(?:从)?\s*(\d{4})\s*年\s*(\d{1,2})\s*月\s*(?:到|至)\s*(?:(\d{4})\s*年)?\s*(\d{1,2})\s*月", text)
        if m:
            year1 = int(m.group(1))
            month1 = int(m.group(2))
            year2 = int(m.group(3)) if m.group(3) else year1
            month2 = int(m.group(4))
            try:
                start = datetime(year1, month1, 1)
                end = (
                    datetime(year2, month2 + 1, 1) - timedelta(seconds=1)
                    if month2 < 12
                    else datetime(year2, 12, 31, 23, 59, 59)
                )
                clean_query = text[:m.start()] + text[m.end():]
                return fmt(start), fmt(end), clean_query.strip()
            except ValueError:
                pass

        m = re.search(
            r"(?<!\d)(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日?",
            text,
        )
        if m:
            try:
                start = datetime(
                    int(m.group(1)),
                    int(m.group(2)),
                    int(m.group(3)),
                )
                clean_query = text[:m.start()] + text[m.end():]
                return None, fmt(start + timedelta(days=1)), clean_query.strip()
            except ValueError:
                pass

        # A month/day without a year is interpreted relative to the recall
        # reference year. For historical-memory queries, a date later than
        # the reference date most naturally refers to the previous year.
        m = re.search(
            r"(?<!\d)(\d{1,2})\s*月\s*(\d{1,2})\s*(?:日|号)?",
            text,
        )
        if m:
            month = int(m.group(1))
            day = int(m.group(2))
            try:
                start = datetime(now.year, month, day)
                if start.date() > now.date():
                    start = datetime(now.year - 1, month, day)
                clean_query = text[:m.start()] + text[m.end():]
                return None, fmt(start + timedelta(days=1)), clean_query.strip()
            except ValueError:
                pass

        m = re.search(r"(\d{4})\s*年\s*(\d{1,2})\s*月", text)
        if m:
            year = int(m.group(1))
            month = int(m.group(2))
            try:
                start = datetime(year, month, 1)
                end = (
                    datetime(year, month + 1, 1) - timedelta(seconds=1)
                    if month < 12
                    else datetime(year, 12, 31, 23, 59, 59)
                )
                clean_query = text[:m.start()] + text[m.end():]
                return fmt(start), fmt(end), clean_query.strip()
            except ValueError:
                pass

        m = re.search(
            r"(?<!\d)(\d{4})-(\d{1,2})-(\d{1,2})(?!\d)",
            text,
        )
        if m:
            try:
                start = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
                clean_query = text[:m.start()] + text[m.end():]
                return None, fmt(start + timedelta(days=1)), clean_query.strip()
            except ValueError:
                pass

        m = re.search(r"上(?:个)?(?:月|星期|周)", text)
        if m:
            unit = m.group()[1:]
            if "月" in unit:
                first_of_month = today_start.replace(day=1)
                end_of_last_month = first_of_month - timedelta(seconds=1)
                start_of_last_month = end_of_last_month.replace(day=1, hour=0, minute=0, second=0)
                start, end = start_of_last_month, end_of_last_month
            else:
                start_of_this_week = today_start - timedelta(days=today_start.weekday())
                start = start_of_this_week - timedelta(days=7)
                end = start_of_this_week
            clean_query = text[:m.start()] + text[m.end():]
            return fmt(start), fmt(end), clean_query.strip()

        m = re.search(r"\b(last month|last week|previous month|previous week)\b", text, re.IGNORECASE)
        if m:
            phrase = m.group(1).lower()
            if "month" in phrase:
                first_of_month = today_start.replace(day=1)
                end = first_of_month - timedelta(seconds=1)
                start = end.replace(day=1, hour=0, minute=0, second=0)
            else:
                end = today_start - timedelta(days=today_start.weekday())
                start = end - timedelta(days=7)
            clean_query = text[:m.start()] + text[m.end():]
            return fmt(start), fmt(end), clean_query.strip()

        m = re.search(r"(?:这个月|本月)", text)
        if m:
            clean_query = text[:m.start()] + text[m.end():]
            return fmt(today_start.replace(day=1)), fmt(now), clean_query.strip()

        m = re.search(r"\b(this month|current month)\b", text, re.IGNORECASE)
        if m:
            clean_query = text[:m.start()] + text[m.end():]
            return fmt(today_start.replace(day=1)), fmt(now), clean_query.strip()

        m = re.search(r"(?:本周|这一周)", text)
        if m:
            clean_query = text[:m.start()] + text[m.end():]
            return fmt(today_start - timedelta(days=today_start.weekday())), fmt(now), clean_query.strip()

        m = re.search(r"\b(this week|current week)\b", text, re.IGNORECASE)
        if m:
            clean_query = text[:m.start()] + text[m.end():]
            return fmt(today_start - timedelta(days=today_start.weekday())), fmt(now), clean_query.strip()

        m = re.search(r"昨天|昨日", text)
        if m:
            start = today_start - timedelta(days=1)
            clean_query = text[:m.start()] + text[m.end():]
            return fmt(start), fmt(start + timedelta(days=1)), clean_query.strip()

        m = re.search(r"\byesterday\b", text, re.IGNORECASE)
        if m:
            start = today_start - timedelta(days=1)
            clean_query = text[:m.start()] + text[m.end():]
            return fmt(start), fmt(start + timedelta(days=1)), clean_query.strip()

        m = re.search(r"前天|前日", text)
        if m:
            start = today_start - timedelta(days=2)
            clean_query = text[:m.start()] + text[m.end():]
            return fmt(start), fmt(start + timedelta(days=1)), clean_query.strip()

        m = re.search(r"今天|今日", text)
        if m:
            clean_query = text[:m.start()] + text[m.end():]
            return fmt(today_start), fmt(now), clean_query.strip()

        m = re.search(r"\btoday\b", text, re.IGNORECASE)
        if m:
            clean_query = text[:m.start()] + text[m.end():]
            return fmt(today_start), fmt(now), clean_query.strip()

        return None, fmt(now), text

    # ── Recall path: raw candidates -> unified rerank -> formatted evidence ─

    def process_memory_recall_immediately(
        self,
        query: str,
        tags: Optional[List[str]] = None,
        time_end: Optional[str] = None,
        prompt_language: str = "zh",
    ) -> Dict[str, Any]:
        """Run recall immediately against the latest committed memory snapshot."""
        # Recall is read-only and must not wait for the store/entity-claim worker's
        # long-running LLM or embedding work. A separate WAL reader gives it
        # a consistent committed snapshot without sharing the writer connection.
        with self._db.reader_transaction() as reader_db:
            return self._recall_sync(
                query=query,
                tags=tags,
                time_end=time_end,
                recall_mode=self._recall_mode,
                prompt_language=prompt_language,
                database=reader_db,
            )

    def _recall_sync(
        self,
        query: str,
        tags: Optional[List[str]] = None,
        time_end: Optional[str] = None,
        recall_mode: str = "normal",
        prompt_language: str = "zh",
        database: Optional[SessionDB] = None,
    ) -> Dict[str, Any]:
        requested_recall_mode = str(recall_mode or "normal").strip().lower()
        started_at = time.monotonic()
        if not self._memory_enabled or not str(query or "").strip():
            recall_report = {
                "memory_context": "",
                "requested_recall_mode": requested_recall_mode,
                "actual_recall_mode": "none",
                "status": "empty",
                "elapsed_ms": round((time.monotonic() - started_at) * 1000, 2),
            }
            self._operation_reporter.on_recall_finished(recall_report)
            return recall_report
        try:
            normalized_recall_mode = requested_recall_mode
            if normalized_recall_mode not in {"stage1", "stage2", "normal"}:
                raise ValueError(
                    "recall_mode must be one of: stage1, stage2, normal"
                )
            reference_time = self._normalize_recall_time_bound(
                time_end,
                default_to_now=True,
            )
            self._log_info("memory_recall", "start", {
                "query": self._format_log_text(query, limit=500),
                "top_k": self._top_k,
                "budget": self._recall_budget,
                "tags": tags or [],
                "requested_time_end": time_end,
                "time_end": reference_time,
                "recall_mode": normalized_recall_mode,
                "prompt_language": prompt_language,
            })

            parsed_time_start, parsed_time_end, time_stripped_query = self._parse_time_expression(
                query,
                reference_time=reference_time,
            )
            temporal_bounds: RecallTimeBounds = (
                parsed_time_start,
                parsed_time_end,
            )
            temporal_mode = self._infer_recall_temporal_mode(query)
            # Keep a successfully parsed time-only query empty for text
            # retrieval rather than reintroducing the removed time expression.
            self._log_info("memory_recall", "query_prepared", {
                "time_stripped_query": self._format_log_text(
                    time_stripped_query,
                    limit=500,
                ),
                "parsed_time_start": parsed_time_start,
                "parsed_time_end": parsed_time_end,
                "temporal_mode": temporal_mode,
            })

            memory_text: Optional[str]
            actual_recall_mode: str
            if normalized_recall_mode == "stage2":
                actual_recall_mode = "stage2"
                stage1_report = self._process_recall_stage1(
                    original_query=query,
                    time_stripped_query=time_stripped_query,
                    temporal_bounds=temporal_bounds,
                    reference_time=reference_time,
                    temporal_mode=temporal_mode,
                    prompt_language=prompt_language,
                    database=database,
                )
                memory_text = self._process_recall_stage2(
                    original_query=query,
                    time_stripped_query=time_stripped_query,
                    temporal_bounds=temporal_bounds,
                    reference_time=reference_time,
                    temporal_mode=temporal_mode,
                    stage1_report=stage1_report,
                    prompt_language=prompt_language,
                    database=database,
                )
            else:
                stage1_report = self._process_recall_stage1(
                    original_query=query,
                    time_stripped_query=time_stripped_query,
                    temporal_bounds=temporal_bounds,
                    reference_time=reference_time,
                    temporal_mode=temporal_mode,
                    prompt_language=prompt_language,
                    database=database,
                )
                if normalized_recall_mode == "stage1":
                    actual_recall_mode = "stage1"
                    memory_text = str(stage1_report.get("memory_context") or "")
                elif not stage1_report.get("trusted"):
                    actual_recall_mode = "stage2"
                    memory_text = self._process_recall_stage2(
                        original_query=query,
                        time_stripped_query=time_stripped_query,
                        temporal_bounds=temporal_bounds,
                        reference_time=reference_time,
                        temporal_mode=temporal_mode,
                        stage1_report=stage1_report,
                        prompt_language=prompt_language,
                        database=database,
                    )
                else:
                    memory_text = str(stage1_report.get("memory_context") or "")
                    actual_recall_mode = "stage1"
            recall_status = "ok" if memory_text else "empty"
            recall_report = {
                "memory_context": memory_text or "",
                "requested_recall_mode": normalized_recall_mode,
                "actual_recall_mode": actual_recall_mode,
                "temporal_mode": temporal_mode,
                "status": recall_status,
                "elapsed_ms": round((time.monotonic() - started_at) * 1000, 2),
                "recall_context_chars": len(memory_text or ""),
            }
            self._log_info("memory_recall", "finish", {
                "status": recall_status,
                "recall_mode": normalized_recall_mode,
                "actual_recall_mode": actual_recall_mode,
                "temporal_mode": temporal_mode,
                "elapsed_ms": round((time.monotonic() - started_at) * 1000, 2),
                "recall_context_chars": len(memory_text or ""),
                "recall_context": memory_text,
            })
            self._operation_reporter.on_recall_finished(recall_report)
            return recall_report
        except Exception as exc:
            self._log_info("memory_recall", "error", {
                "query": self._format_log_text(query, limit=500),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "elapsed_ms": round((time.monotonic() - started_at) * 1000, 2),
            })
            raise

    def _retrieve_recall_stage1_seed_candidates(
        self,
        *,
        terms: Sequence[str],
        query_entity_names: Sequence[str],
        direct_object_types: Sequence[str],
        prospective_query_profile: Dict[str, Any],
        temporal_bounds: RecallTimeBounds,
        temporal_mode: str,
        candidate_limits: Dict[str, int],
        database: Optional[SessionDB] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve Stage 1 direct seeds for permitted object types.

        Episode documents are intentionally not permitted here. They remain a
        stored projection but only participate later as fact-association
        boundaries.
        """
        seed_search_terms = self._recall_stage1_build_seed_search_terms(
            terms=terms,
            query_entity_names=query_entity_names,
        )
        return self._retrieve_recall_document_lexical_seed_candidates(
            terms=seed_search_terms,
            direct_object_types=direct_object_types,
            prospective_query_profile=prospective_query_profile,
            temporal_bounds=temporal_bounds,
            temporal_mode=temporal_mode,
            candidate_limits=candidate_limits,
            candidate_source="stage1_lexical",
            database=database,
        )

    def _retrieve_recall_document_lexical_seed_candidates(
        self,
        *,
        terms: Sequence[str],
        direct_object_types: Sequence[str],
        temporal_bounds: RecallTimeBounds,
        temporal_mode: str,
        candidate_limits: Dict[str, int],
        candidate_source: str,
        prospective_query_profile: Optional[Dict[str, Any]] = None,
        database: Optional[SessionDB] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve direct seeds from the shared document projection.

        Fact documents are hydrated back to ``memory_facts`` before they are
        returned, while derived-memory projections can become candidates
        directly.  For an explicit prospective time-slot query, prospective
        object types use a strict time browse instead of lexical terms.
        Episode is deliberately excluded because it is only a fact-association
        boundary.
        """
        direct_types = list(dict.fromkeys(
            str(value or "").strip().lower()
            for value in direct_object_types or []
            if str(value or "").strip().lower()
            in {"fact", "entity_claim", "goal", "plan", "work_item"}
        ))
        db = database or self._db
        is_prospective_time_slot_query = bool(
            (prospective_query_profile or {}).get("is_prospective_query")
        )
        candidates: List[Dict[str, Any]] = []
        for object_type in direct_types:
            candidate_limit = max(
                1,
                int(candidate_limits.get(object_type, 1) or 1),
            )
            # Fact time filtering depends on the query's temporal mode
            # (event time / dialogue time / both), while the shared document
            # stores one canonical time range.  Over-fetch fact documents and
            # apply the authoritative fact-level filter after hydration.
            retrieval_limit = (
                max(24, candidate_limit * 4)
                if object_type == "fact" and any(temporal_bounds or (None, None))
                else candidate_limit
            )
            document_time_start, document_time_end = (
                temporal_bounds or (None, None)
            ) if object_type != "fact" else (None, None)
            use_prospective_time_slot_browse = bool(
                is_prospective_time_slot_query
                and object_type in {"goal", "plan", "work_item"}
            )
            object_terms = () if use_prospective_time_slot_browse else terms
            if not object_terms and not use_prospective_time_slot_browse:
                # An empty query is meaningful only for an explicitly routed
                # prospective time-slot browse.  It must not turn fact or
                # claim retrieval into an unrelated "most recent" scan.
                continue
            object_candidate_source = (
                candidate_source.removesuffix("_lexical") + "_prospective_query"
                if use_prospective_time_slot_browse
                else candidate_source
            )
            rows = db.search_memory_recall_documents(
                object_type=object_type,
                terms=object_terms,
                statuses=self._recall_stage1_direct_document_statuses(object_type),
                time_start=document_time_start,
                time_end=document_time_end,
                limit=retrieval_limit,
                strict_time_filter=use_prospective_time_slot_browse,
            )
            candidates.extend(self._make_recall_document_candidates(
                rows=rows,
                candidate_source=object_candidate_source,
                temporal_bounds=temporal_bounds,
                temporal_mode=temporal_mode,
                per_type_limit=candidate_limit,
                database=db,
            ))
        return candidates

    @staticmethod
    def _recall_stage1_direct_document_statuses(
        object_type: str,
    ) -> Sequence[str]:
        """Keep Stage 1 focused on currently answerable derived objects."""
        return {
            "entity_claim": ("active",),
            "goal": ("active",),
            "plan": ("planned", "rescheduled"),
            "work_item": ("open", "blocked"),
        }.get(str(object_type or "").strip().lower(), ())

    @staticmethod
    def _recall_stage1_query_modes(
        query: str,
        *,
        temporal_bounds: RecallTimeBounds,
    ) -> List[str]:
        """Route only explicit Stage 1 knowledge/prospective wording.

        Facts remain the default evidence source. The rule route is purposely
        narrow: ambiguous requests fall through to Stage 2 rather than
        searching every derived-memory type in the fast path.
        """
        text = _compact_whitespace(query).lower()
        modes = ["factual"]
        knowledge_markers = (
            "偏好", "喜欢", "不喜欢", "习惯", "通常", "一般", "倾向",
            "画像", "特点", "性格", "preference", "prefer", "dislike",
            "habit", "usually", "tend to", "profile",
        )
        prospective_markers = (
            "明天", "后天", "接下来", "计划", "安排", "待办", "任务",
            "目标", "截止", "要去", "要做", "还要", "未完成", "提醒",
            "tomorrow", "next", "plan", "schedule", "todo", "task",
            "goal", "deadline", "upcoming", "remaining",
        )
        if any(marker in text for marker in knowledge_markers):
            modes.append("knowledge")
        if (
            any(marker in text for marker in prospective_markers)
            or bool((temporal_bounds or (None, None))[0])
            and any(marker in text for marker in ("去", "做", "安排", "计划", "任务", "行程"))
        ):
            modes.append("prospective")
        return modes

    @staticmethod
    def _recall_query_has_explicit_temporal_expression(query: str) -> bool:
        """Return whether the user text itself states a calendar/time window."""
        text = str(query or "")
        return bool(re.search(
            r"今天|今日|明天|后天|本周|这周|下周|下星期|本月|这个月|下个月|"
            r"今晚|明早|下午|上午|月底|月底前|未来|接下来|"
            r"\d{4}\s*(?:年|[-/.])\s*\d{1,2}|\d{1,2}\s*月\s*\d{1,2}|"
            r"\b(?:today|tomorrow|the day after tomorrow|this week|next week|"
            r"this month|next month|tonight|this afternoon|this morning|"
            r"upcoming|in the next)\b",
            text,
            re.IGNORECASE,
        ))

    def _recall_stage1_prospective_query_profile(
        self,
        *,
        original_query: str,
        temporal_bounds: RecallTimeBounds,
        reference_time: str,
    ) -> Dict[str, Any]:
        """Conservatively identify a current/future personal time-slot query.

        The profile gates the no-term prospective browse only.  It does not
        decide whether goals, plans, or work items may participate in normal
        lexical recall; a concrete query such as ``明天去天津的计划`` can still
        retrieve those objects through its text terms.
        """
        text = _compact_whitespace(original_query).lower()
        time_start, time_end = temporal_bounds or (None, None)
        reference = self._recall_stage1_parse_datetime(reference_time)
        window_start = self._recall_stage1_parse_datetime(time_start)
        window_end = self._recall_stage1_parse_datetime(time_end)
        has_time_window = bool(window_start or window_end)
        has_explicit_time_expression = (
            self._recall_query_has_explicit_temporal_expression(original_query)
        )
        current_or_future_window = bool(
            has_explicit_time_expression
            and has_time_window
            and (
                (window_end is not None and (reference is None or window_end >= reference))
                or (window_end is None and window_start is not None and (
                    reference is None or window_start >= reference
                ))
            )
        )
        slot_markers = (
            "安排", "计划", "日程", "行程", "待办", "任务", "事项", "提醒",
            "截止", "未完成", "还要", "剩下", "下一步", "要做", "需要做",
            "schedule", "agenda", "plan", "todo", "to do", "task",
            "deadline", "upcoming", "remaining", "what's next",
        )
        inventory_markers = (
            "有什么", "有哪些", "有没有", "什么安排", "什么计划", "做什么",
            "需要做什么", "要做什么", "还剩什么", "下一步", "几件",
            "what do i have", "what's on", "what is on", "what should i do",
            "any plans", "any tasks", "what remains", "what's next",
        )
        excluded_markers = (
            "怎么", "如何", "推荐", "建议", "天气", "新闻", "电影", "股价",
            "航班", "路线", "how to", "recommend", "weather", "news", "movie",
            "stock", "flight", "route",
        )
        command_markers = (
            "提醒我", "创建日程", "帮我创建", "帮我安排", "设置提醒",
            "remind me", "create a reminder", "schedule for me",
        )
        has_slot_marker = any(marker in text for marker in slot_markers)
        has_inventory_marker = any(marker in text for marker in inventory_markers)
        has_excluded_marker = any(marker in text for marker in excluded_markers)
        has_command_marker = any(marker in text for marker in command_markers)
        is_prospective_query = bool(
            current_or_future_window
            and has_slot_marker
            and has_inventory_marker
            and not has_excluded_marker
            and not has_command_marker
        )
        if not has_explicit_time_expression:
            reason = "no_explicit_time_expression"
        elif not has_time_window:
            reason = "no_time_window"
        elif not current_or_future_window:
            reason = "historical_time_window"
        elif has_command_marker:
            reason = "immediate_command"
        elif has_excluded_marker:
            reason = "external_or_advice_query"
        elif not has_slot_marker:
            reason = "no_prospective_slot_marker"
        elif not has_inventory_marker:
            reason = "not_inventory_query"
        else:
            reason = "current_or_future_prospective_time_slot"
        return {
            "is_prospective_query": is_prospective_query,
            "reason": reason,
            "has_explicit_time_expression": has_explicit_time_expression,
            "has_time_window": has_time_window,
            "current_or_future_window": current_or_future_window,
            "has_slot_marker": has_slot_marker,
            "has_inventory_marker": has_inventory_marker,
        }

    def _recall_stage2_prospective_query_profile(
        self,
        *,
        original_query: str,
        llm_value: Any,
        temporal_bounds: RecallTimeBounds,
        reference_time: str,
    ) -> Dict[str, Any]:
        """Validate the LLM's prospective time-slot routing decision."""
        if isinstance(llm_value, bool):
            llm_is_prospective = llm_value
        else:
            llm_is_prospective = str(llm_value or "").strip().lower() in {
                "1", "true", "yes",
            }
        time_start, time_end = temporal_bounds or (None, None)
        reference = self._recall_stage1_parse_datetime(reference_time)
        window_start = self._recall_stage1_parse_datetime(time_start)
        window_end = self._recall_stage1_parse_datetime(time_end)
        has_time_window = bool(window_start or window_end)
        has_explicit_time_expression = (
            self._recall_query_has_explicit_temporal_expression(original_query)
        )
        current_or_future_window = bool(
            has_explicit_time_expression
            and has_time_window
            and (
                (window_end is not None and (reference is None or window_end >= reference))
                or (window_end is None and window_start is not None and (
                    reference is None or window_start >= reference
                ))
            )
        )
        is_prospective_query = bool(
            llm_is_prospective and current_or_future_window
        )
        return {
            "is_prospective_query": is_prospective_query,
            "reason": (
                "llm_prospective_time_slot"
                if is_prospective_query
                else "llm_declined_prospective_time_slot"
                if not llm_is_prospective
                else "invalid_or_historical_time_window"
            ),
            "llm_is_prospective_query": llm_is_prospective,
            "has_explicit_time_expression": has_explicit_time_expression,
            "has_time_window": has_time_window,
            "current_or_future_window": current_or_future_window,
        }

    @staticmethod
    def _recall_stage1_direct_object_types(
        query_modes: Sequence[str],
    ) -> List[str]:
        modes = set(query_modes or [])
        object_types = ["fact"]
        if "knowledge" in modes:
            object_types.append("entity_claim")
        if "prospective" in modes:
            object_types.extend(("goal", "plan", "work_item"))
        return object_types

    @staticmethod
    def _normalize_recall_object_types(value: Any) -> List[str]:
        """Normalize LLM-routed Stage 2 object types without excluding facts."""
        allowed_types = {
            "fact", "entity_claim", "goal", "plan", "work_item",
        }
        raw_values = value if isinstance(value, (list, tuple, set)) else []
        object_types = ["fact"]
        for raw_value in raw_values:
            object_type = str(raw_value or "").strip().lower()
            if object_type in allowed_types and object_type not in object_types:
                object_types.append(object_type)
        return object_types

    def _recall_stage1_build_seed_search_terms(
        self,
        *,
        terms: Sequence[str],
        query_entity_names: Sequence[str],
    ) -> List[str]:
        """Build lexical terms for Stage 1 seed retrieval.

        Concrete query entities are intentionally added only here.  Candidate
        scoring and evidence coverage continue to use the original query
        terms plus their dedicated entity-matching signal.
        """
        high_value_entity_names = [
            entity_name
            for entity_name in self._normalize_entity_names(query_entity_names)
            if self._recall_stage1_entity_value_class(
                self._recall_stage1_normalize_match_text(entity_name)
            ) == "high"
        ]
        return self._build_recall_search_terms(
            "",
            keywords=[*high_value_entity_names, *list(terms or [])],
            entities=[],
        )

    def _process_recall_stage1(
        self,
        *,
        original_query: str,
        time_stripped_query: str,
        temporal_bounds: RecallTimeBounds,
        reference_time: str,
        temporal_mode: str = "dialogue_time",
        prompt_language: str = "zh",
        database: Optional[SessionDB] = None,
    ) -> Dict[str, Any]:
        """Run a deterministic, no-LLM recall path for high-confidence hits.

        Stage 1 is deliberately conservative. It returns a formatted context
        only when the retrieved candidates provide sufficient direct matching
        evidence. Otherwise it returns ``None`` so the caller can fall through
        to Stage 2 semantic retrieval.
        """
        started_at = time.monotonic()
        terms = self._build_recall_search_terms(
            time_stripped_query,
            keywords=[],
            entities=[],
        )
        is_contextual_query = self._recall_is_contextual_query(original_query)
        query_modes = self._recall_stage1_query_modes(
            original_query,
            temporal_bounds=temporal_bounds,
        )
        direct_object_types = self._recall_stage1_direct_object_types(
            query_modes,
        )
        prospective_query_profile = self._recall_stage1_prospective_query_profile(
            original_query=original_query,
            temporal_bounds=temporal_bounds,
            reference_time=reference_time,
        )

        candidate_limits = self._recall_stage1_candidate_limits(
            top_k=self._top_k,
        )
        seed_candidate_limits = candidate_limits["seed_limits"]
        selected_candidate_limits = candidate_limits["selected_limits"]
        association_per_relation_limit = candidate_limits[
            "association_per_relation_limit"
        ]
        query_entity_names = self._recall_stage1_resolve_query_entity_names(
            query=original_query,
            database=database,
        )
        self._log_info("memory_recall_stage1", "start", {
            "original_query": self._format_log_text(original_query, limit=500),
            "time_stripped_query": self._format_log_text(
                time_stripped_query,
                limit=500,
            ),
            "top_k": self._top_k,
            "budget": self._recall_budget,
            "terms": terms,
            "time_start": (temporal_bounds or (None, None))[0],
            "time_end": (temporal_bounds or (None, None))[1],
            "temporal_mode": temporal_mode,
            "candidate_limits": candidate_limits,
            "query_entity_names": query_entity_names,
            "query_modes": query_modes,
            "direct_object_types": direct_object_types,
            "prospective_query_profile": prospective_query_profile,
        })
        # direct recall
        seed_candidates = self._retrieve_recall_stage1_seed_candidates(
            terms=terms,
            query_entity_names=query_entity_names,
            direct_object_types=direct_object_types,
            prospective_query_profile=prospective_query_profile,
            temporal_bounds=temporal_bounds,
            temporal_mode=temporal_mode,
            candidate_limits=seed_candidate_limits,
            database=database,
        )
        self._log_recall_stage1_seed_candidates(
            seed_candidates=seed_candidates,
            seed_candidate_limits=seed_candidate_limits,
        )
        direct_candidates = self._recall_stage1_calculate_candidate_matching_score(
            candidates=seed_candidates,
            search_terms=terms,
            query_entity_names=query_entity_names,
            is_contextual_query=is_contextual_query,
            temporal_bounds=temporal_bounds,
        )
        self._log_recall_direct_candidates(
            stage_name="stage1",
            seed_candidates=seed_candidates,
            direct_candidates=direct_candidates,
        )
        # associative recall
        association_candidates = (
            self._retrieve_association_candidates_using_seed_candidates(
                seed_candidates=direct_candidates,
                temporal_bounds=temporal_bounds,
                temporal_mode=temporal_mode,
                limit=association_per_relation_limit,
                candidate_source_prefix="stage1",
                database=database,
            )
        )
        expanded_candidates = self._merge_recall_direct_and_associative_candidates(
            direct_candidates=direct_candidates,
            association_candidates=association_candidates,
            stage_name="stage1",
        )
        self._log_recall_association_candidates(
            stage_name="stage1",
            association_candidates=association_candidates,
            expanded_candidates=expanded_candidates,
        )
        semantic_query = self._recall_stage1_requires_semantic_search(
            time_stripped_query
        )
        selected_candidates = self._recall_stage1_rank_and_select_candidates(
            candidates=expanded_candidates,
            layer_limits=selected_candidate_limits,
        )
        self._log_recall_selected_candidates(
            stage_name="stage1",
            selected_candidates=selected_candidates,
            expanded_candidates=expanded_candidates,
        )

        evidence_profile = self._recall_stage1_build_evidence_profile(
            candidates=selected_candidates,
            query_terms=terms,
            query_entity_names=query_entity_names,
            query_modes=query_modes,
        )
        evidence_gate = bool(evidence_profile.get("trusted"))

        # A lexical fast path can surface evidence for a semantic question,
        # but it cannot safely perform the synthesis that the question asks
        # for (for example, comparison, trend, or full history).  Keep the
        # evidence for Stage 2 and do not prematurely return it as an answer.
        trusted = bool(selected_candidates) and evidence_gate and not semantic_query
        memory_text = self._build_memory_retrieved_format_text(
            entries=selected_candidates,
            prompt_language=prompt_language,
        )
        stage1_finish_payload = self._log_recall_stage1_finish_payload(
            selected_candidates=selected_candidates,
            semantic_query=semantic_query,
            evidence_profile=evidence_profile,
            trusted=trusted,
            memory_text=memory_text,
            started_at=started_at,
        )
        return {
            "memory_context": memory_text or "",
            # Stage 2 receives the original direct lexical pool, not the
            # Stage 1 association results or final presentation selection.
            # Stage 2 still has fact-only scoring during this transition, so
            # it receives only fact seeds. Stage 1 itself can already return
            # claims and prospective candidates from the new projection.
            "seed_candidates": [
                candidate for candidate in seed_candidates
                if str(candidate.get("index_level") or "") == "fact"
            ],
            "direct_seed_candidates": list(seed_candidates),
            "query_modes": query_modes,
            "evidence_profile": evidence_profile,
            "trusted": bool(trusted and memory_text),
            "elapsed_ms": stage1_finish_payload["elapsed_ms"],
        }

    @staticmethod
    def _recall_count_candidates_by_level(
        candidates: Sequence[Dict[str, Any]],
    ) -> Dict[str, int]:
        counts = {
            "fact": 0,
            "entity_claim": 0,
            "goal": 0,
            "plan": 0,
            "work_item": 0,
        }
        for candidate in candidates or []:
            level = str(candidate.get("index_level") or "")
            if level in counts:
                counts[level] += 1
        return counts

    def _log_recall_stage1_seed_candidates(
        self,
        *,
        seed_candidates: Sequence[Dict[str, Any]],
        seed_candidate_limits: Dict[str, int],
    ) -> None:
        """Log the lexical retrieval output before any Stage 1 scoring."""
        seed_candidates = list(seed_candidates or [])
        payload: Dict[str, Any] = {
            "seed_candidate_count": len(seed_candidates),
            "seed_candidate_limits": dict(seed_candidate_limits or {}),
            "seed_by_level": self._recall_count_candidates_by_level(
                seed_candidates
            ),
            "candidates": self._recall_log_candidate_items(
                seed_candidates,
                detailed=self._recall_detailed_logging,
                limit=None if self._recall_detailed_logging else 12,
                stage="stage1_seed",
            ),
        }
        self._log_info("memory_recall_stage1", "seeds_retrieved", payload)

    def _log_recall_direct_candidates(
        self,
        *,
        stage_name: str,
        seed_candidates: Sequence[Dict[str, Any]],
        direct_candidates: Sequence[Dict[str, Any]],
    ) -> None:
        """Log direct matcher acceptance for one recall stage."""
        stage_name = str(stage_name or "").strip().lower()
        if stage_name not in {"stage1", "stage2"}:
            raise ValueError(f"Unsupported recall stage: {stage_name}")
        seed_candidates = list(seed_candidates or [])
        direct_candidates = list(direct_candidates or [])
        payload: Dict[str, Any] = {
            "scored_candidate_count": len(seed_candidates),
            "direct_candidate_count": len(direct_candidates),
            "rejected_candidate_count": max(
                0, len(seed_candidates) - len(direct_candidates)
            ),
            "direct_by_level": self._recall_count_candidates_by_level(
                direct_candidates
            ),
            "direct_candidates": self._recall_log_candidate_items(
                direct_candidates,
                detailed=self._recall_detailed_logging,
                limit=None if self._recall_detailed_logging else 12,
                stage=f"{stage_name}_direct",
            ),
        }
        self._log_info(
            f"memory_recall_{stage_name}",
            "direct_candidates_scored",
            payload,
        )

    def _log_recall_association_candidates(
        self,
        *,
        stage_name: str,
        association_candidates: Sequence[Dict[str, Any]],
        expanded_candidates: Sequence[Dict[str, Any]],
    ) -> None:
        """Log association expansion and the expanded candidates."""
        stage_name = str(stage_name or "").strip().lower()
        if stage_name not in {"stage1", "stage2"}:
            raise ValueError(f"Unsupported recall stage: {stage_name}")
        association_candidates = list(association_candidates or [])
        expanded_candidates = list(expanded_candidates or [])
        association_by_relation = {"same_episode": 0}
        for candidate in association_candidates:
            relation = str(
                candidate.get("_recall_association_relation")
                or ""
            )
            if relation in association_by_relation:
                association_by_relation[relation] += 1
        payload: Dict[str, Any] = {
            "association_candidate_count": len(association_candidates),
            "association_by_relation": association_by_relation,
            "expanded_candidate_count": len(expanded_candidates),
            "expanded_by_level": self._recall_count_candidates_by_level(
                expanded_candidates
            ),
            "candidates": self._recall_log_candidate_items(
                expanded_candidates,
                detailed=self._recall_detailed_logging,
                limit=None if self._recall_detailed_logging else 12,
                stage=f"{stage_name}_expanded",
            ),
        }
        self._log_info(
            f"memory_recall_{stage_name}",
            "association_candidates_merged",
            payload,
        )

    def _log_recall_selected_candidates(
        self,
        *,
        stage_name: str,
        selected_candidates: Sequence[Dict[str, Any]],
        expanded_candidates: Sequence[Dict[str, Any]],
    ) -> None:
        """Log final candidate selection for one recall stage."""
        stage_name = str(stage_name or "").strip().lower()
        if stage_name not in {"stage1", "stage2"}:
            raise ValueError(f"Unsupported recall stage: {stage_name}")
        selected_candidates = list(selected_candidates or [])
        expanded_candidates = list(expanded_candidates or [])
        payload: Dict[str, Any] = {
            "expanded_candidate_count": len(expanded_candidates),
            "selected_candidate_count": len(selected_candidates),
            "selected_by_level": self._recall_count_candidates_by_level(
                selected_candidates
            ),
            "selected_candidates": self._recall_log_candidate_items(
                selected_candidates,
                detailed=self._recall_detailed_logging,
                limit=None if self._recall_detailed_logging else 12,
                stage=f"{stage_name}_selected",
            ),
            "selected_targets": [
                f"{item.get('target_table')}#{item.get('target_id')}"
                for item in selected_candidates
            ],
        }
        self._log_info(
            f"memory_recall_{stage_name}",
            "candidates_selected",
            payload,
        )

    def _log_recall_stage1_finish_payload(
        self,
        *,
        selected_candidates: Sequence[Dict[str, Any]],
        semantic_query: bool,
        evidence_profile: Dict[str, Any],
        trusted: bool,
        memory_text: str,
        started_at: float,
    ) -> Dict[str, Any]:
        """Build and log only the final Stage 1 trust/output decision."""
        selected_candidates = list(selected_candidates or [])
        status = "hit" if trusted and memory_text else "miss"
        if status == "hit":
            reason = ""
        elif not selected_candidates:
            reason = "no_selected_candidates"
        elif not memory_text:
            reason = "empty_formatted_context"
        elif semantic_query:
            reason = "semantic_query_requires_stage2"
        else:
            reason = "evidence_profile_below_threshold"
        stage1_finish_payload: Dict[str, Any] = {
            "status": status,
            "reason": reason,
            "trusted": bool(trusted),
            "semantic_query": bool(semantic_query),
            "selected_candidate_count": len(selected_candidates),
            "selected_by_level": self._recall_count_candidates_by_level(
                selected_candidates
            ),
            "strong_anchor_reasons": sorted({
                str(reason)
                for item in selected_candidates
                for reason in item.get("strong_anchor_reasons") or []
                if str(reason)
            }),
            "evidence_profile": dict(evidence_profile or {}),
            "retrieved_chars": len(memory_text or ""),
            "elapsed_ms": round((time.monotonic() - started_at) * 1000, 2),
        }
        self._log_info("memory_recall_stage1", "finish", stage1_finish_payload)
        return stage1_finish_payload

    def _retrieve_association_candidates_using_seed_candidates(
        self,
        *,
        seed_candidates: Sequence[Dict[str, Any]],
        temporal_bounds: RecallTimeBounds,
        temporal_mode: str,
        limit: int,
        candidate_source_prefix: str = "stage1",
        database: Optional[SessionDB] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve all fact associations for accepted direct candidates.

        This is the association orchestrator only.  The two relation families
        deliberately remain separate: derived-memory evidence is retrieved
        from explicit mappings, while fact-to-fact expansion uses shared
        episode membership.  Episode documents themselves are never
        candidates.
        """
        db = database or self._db
        document_evidence_candidates = (
            self._retrieve_evidence_fact_association_candidates(
                direct_candidates=seed_candidates,
                candidate_source_prefix=candidate_source_prefix,
                database=db,
            )
        )
        same_episode_candidates = (
            self._retrieve_same_episode_fact_association_candidates(
                seed_candidates=seed_candidates,
                temporal_bounds=temporal_bounds,
                temporal_mode=temporal_mode,
                limit=limit,
                candidate_source_prefix=candidate_source_prefix,
                database=db,
            )
        )
        return [*document_evidence_candidates, *same_episode_candidates]

    def _retrieve_same_episode_fact_association_candidates(
        self,
        *,
        seed_candidates: Sequence[Dict[str, Any]],
        temporal_bounds: RecallTimeBounds,
        temporal_mode: str,
        limit: int,
        candidate_source_prefix: str,
        database: SessionDB,
    ) -> List[Dict[str, Any]]:
        """Expand accepted fact seeds with other facts in the same episode."""
        fact_seed_candidates = [
            candidate
            for candidate in seed_candidates or []
            if str(candidate.get("index_level") or "") == "fact"
        ]
        seed_scores: Dict[int, float] = {}
        for candidate in fact_seed_candidates or []:
            try:
                fact_id = int(candidate.get("target_id"))
            except (TypeError, ValueError):
                continue
            seed_scores[fact_id] = max(
                seed_scores.get(fact_id, 0.0),
                self._clamp_float(candidate.get("_recall_score"), 0.0, 1.0, 0.0),
            )
        if not seed_scores:
            return []

        per_relation_limit = max(1, int(limit or 24))
        related_scores: Dict[int, Tuple[str, float]] = {}
        for pair in database.related_fact_pairs_by_episode_fact_ids(
            list(seed_scores),
            limit=per_relation_limit,
        ):
            seed_score = seed_scores.get(int(pair["seed_fact_id"]), 0.0)
            propagated_score = round(
                seed_score * self._recall_fact_association_decay("same_episode"),
                4,
            )
            related_fact_id = int(pair["related_fact_id"])
            existing = related_scores.get(related_fact_id)
            if existing is None or propagated_score > existing[1]:
                related_scores[related_fact_id] = (
                    "same_episode",
                    propagated_score,
                )
        if not related_scores:
            return []

        expanded: List[Dict[str, Any]] = []
        for fact in database.get_memory_facts_by_ids(list(related_scores)):
            fact_id = int(fact.get("id") or 0)
            relation_info = related_scores.get(fact_id)
            if not relation_info:
                continue
            relation, propagated_score = relation_info
            candidate = self._make_recall_document_candidate(
                row=fact,
                object_type="fact",
                candidate_source=(
                    f"{candidate_source_prefix}_episode_association"
                ),
                temporal_bounds=temporal_bounds,
                temporal_mode=temporal_mode,
            )
            if not candidate:
                continue
            candidate["_recall_association_score"] = propagated_score
            candidate["_recall_association_relation"] = relation
            candidate["_recall_score"] = propagated_score
            candidate["evidence"] = ["associative_recall"]
            candidate["matched"] = True
            candidate["candidate_score_threshold"] = 0.0
            candidate["filter_reason"] = ""
            candidate["has_strong_anchor"] = False
            candidate["strong_anchor_reasons"] = []
            match_details_key = (
                "_recall_stage2_match_details"
                if candidate_source_prefix == "stage2"
                else "_recall_fast_match_details"
            )
            candidate[match_details_key] = {
                "topic_match_info": {},
                "entity_match_info": {},
                "time_score_info": {},
                "score_components": {},
            }
            candidate["_recall_decision"] = {
                "accepted": True,
                "decision_reason": "accepted_associative_recall",
            }
            expanded.append(candidate)
        return expanded

    def _retrieve_evidence_fact_association_candidates(
        self,
        *,
        direct_candidates: Sequence[Dict[str, Any]],
        candidate_source_prefix: str,
        database: SessionDB,
    ) -> List[Dict[str, Any]]:
        """Expand claim/prospective candidates to their mapped fact evidence."""
        claim_candidates = [
            candidate for candidate in direct_candidates or []
            if str(candidate.get("index_level") or "") == "entity_claim"
        ]
        future_commitment_candidates = [
            candidate for candidate in direct_candidates or []
            if str(candidate.get("index_level") or "")
            in {"goal", "plan", "work_item"}
        ]
        evidence_by_fact_id: Dict[int, Tuple[str, float]] = {}

        claim_evidence = database.get_entity_claim_evidence_fact_ids(
            [candidate.get("target_id") for candidate in claim_candidates],
            limit=max(24, len(claim_candidates) * 4),
        )
        for candidate in claim_candidates:
            claim_id = int(candidate.get("target_id") or 0)
            propagated_score = round(
                self._clamp_float(candidate.get("_recall_score"), 0.0, 1.0, 0.0)
                * 0.90,
                4,
            )
            for fact_id in claim_evidence.get(claim_id, [])[:3]:
                existing = evidence_by_fact_id.get(fact_id)
                if existing is None or propagated_score > existing[1]:
                    evidence_by_fact_id[fact_id] = (
                        "claim_evidence", propagated_score,
                    )

        intent_evidence = database.get_intent_evidence_fact_ids(
            [
                (
                    str(candidate.get("index_level") or ""),
                    int(candidate.get("target_id") or 0),
                )
                for candidate in future_commitment_candidates
            ],
            limit=max(24, len(future_commitment_candidates) * 4),
        )
        for candidate in future_commitment_candidates:
            key = (
                str(candidate.get("index_level") or ""),
                int(candidate.get("target_id") or 0),
            )
            propagated_score = round(
                self._clamp_float(candidate.get("_recall_score"), 0.0, 1.0, 0.0)
                * 0.90,
                4,
            )
            for fact_id in intent_evidence.get(key, [])[:3]:
                existing = evidence_by_fact_id.get(fact_id)
                if existing is None or propagated_score > existing[1]:
                    evidence_by_fact_id[fact_id] = (
                        "intent_evidence", propagated_score,
                    )
        if not evidence_by_fact_id:
            return []

        candidates: List[Dict[str, Any]] = []
        for fact in database.get_memory_facts_by_ids(list(evidence_by_fact_id)):
            fact_id = int(fact.get("id") or 0)
            relation_info = evidence_by_fact_id.get(fact_id)
            if not relation_info:
                continue
            relation, propagated_score = relation_info
            candidate = self._make_recall_document_candidate(
                row=fact,
                object_type="fact",
                candidate_source=f"{candidate_source_prefix}_{relation}",
                # An evidence fact can describe the creation or update of a
                # future object before its scheduled time, so it must not be
                # rejected using the query's prospective time window.
                temporal_bounds=None,
                temporal_mode="none",
            )
            if not candidate:
                continue
            candidate.update({
                "_recall_association_score": propagated_score,
                "_recall_association_relation": relation,
                "_recall_score": propagated_score,
                "matched": True,
                "candidate_score_threshold": 0.0,
                "filter_reason": "",
                "has_strong_anchor": False,
                "strong_anchor_reasons": [],
                "_recall_decision": {
                    "accepted": True,
                    "decision_reason": f"accepted_{relation}",
                },
                "_recall_fast_match_details": {
                    "topic_match_info": {},
                    "entity_match_info": {},
                    "time_score_info": {},
                    "score_components": {},
                },
            })
            candidates.append(candidate)
        return candidates

    def _recall_fact_association_decay(self, relation: str) -> float:
        """Return the configured score decay for one fact association edge."""
        return {
            "same_episode": self._recall_stage1_episode_propagation_decay,
        }.get(relation, self._recall_stage1_association_propagation_decay)

    @staticmethod
    def _recall_stage1_candidate_limits(
        *,
        top_k: int,
    ) -> Dict[str, Any]:
        """Build explicit Stage 1 retrieval, expansion, and output budgets.

        ``top_k`` controls the fact quota. Derived objects receive small,
        independent quotas only when the Stage 1 query route requests them.
        Episode is intentionally absent: it is an association boundary, not a
        direct-recall candidate type.
        """
        k = max(1, int(top_k or 1))
        seed_per_level_limit = max(8, min(24, k * 2))
        return {
            # Direct lexical recall keeps a modest, fixed over-fetch ratio so
            # score filtering and association can work without inflating the
            # Stage 1 latency or candidate pool for contextual wording.
            "seed_limits": {
                "fact": seed_per_level_limit,
                "entity_claim": max(4, min(12, k)),
                "goal": max(4, min(10, k)),
                "plan": max(4, min(12, k)),
                "work_item": max(4, min(12, k)),
            },
            # The association loader applies this limit to same-episode facts.
            "association_per_relation_limit": max(4, min(12, k)),
            "selected_limits": {
                "fact": k,
                "entity_claim": max(2, min(4, k)),
                "goal": max(2, min(3, k)),
                "plan": max(2, min(4, k)),
                "work_item": max(2, min(4, k)),
            },
        }

    @staticmethod
    def _recall_candidate_source_channels(value: Any) -> set[str]:
        """Normalize prefixed candidate sources to retrieval channels."""
        source = str(value or "").strip().lower()
        if source == "both" or source.endswith("_both"):
            return {"entity_mapping", "lexical"}
        channels: set[str] = set()
        if source == "entity_mapping" or source.endswith("_entity_mapping"):
            channels.add("entity_mapping")
        if source == "lexical" or source.endswith("_lexical"):
            channels.add("lexical")
        return channels

    def _recall_stage1_calculate_candidate_matching_score(
        self,
        *,
        candidates: Sequence[Dict[str, Any]],
        search_terms: Sequence[str] = (),
        query_entity_names: Sequence[str] = (),
        is_contextual_query: bool = False,
        temporal_bounds: RecallTimeBounds = None,
    ) -> List[Dict[str, Any]]:
        """Score direct Stage 1 candidates and return accepted ones.

        This method scores only direct lexical candidates. Associated recall
        candidates receive their propagated score at expansion time instead.
        """
        direct_candidates: List[Dict[str, Any]] = []
        for candidate in candidates or []:
            matching_score_info = (
                self._recall_stage1_calculate_single_candidate_matching_score(
                    candidate,
                    search_terms=search_terms,
                    query_entity_names=query_entity_names,
                    is_contextual_query=is_contextual_query,
                    temporal_bounds=temporal_bounds,
                )
            )
            match_details = dict(
                matching_score_info.get("_recall_fast_match_details") or {}
            )
            candidate.update({
                key: value
                for key, value in matching_score_info.items()
                if key != "_recall_fast_match_details"
            })
            # Direct Stage 1 matching is represented by the normalized
            # strong-anchor fields; do not carry the legacy evidence list
            # forward on the candidate.
            candidate.pop("evidence", None)
            candidate.pop("_recall_fast_match_evidence", None)
            candidate["_recall_fast_match_details"] = match_details
            candidate["_recall_score"] = self._clamp_float(
                matching_score_info.get("score"),
                0.0,
                1.0,
                0.0,
            )
            candidate["_recall_candidate_source"] = (
                candidate.get("_recall_candidate_source") or "stage1_lexical"
            )
            entity_match_info = match_details.get("entity_match_info")
            if not isinstance(entity_match_info, dict):
                entity_match_info = {}
            if entity_match_info.get("matched_entity_names"):
                candidate["_recall_entity_names"] = list(
                    entity_match_info.get("matched_entity_names") or []
                )
            accepted = bool(
                matching_score_info.get("matched")
            )
            candidate["_recall_decision"] = {
                "accepted": accepted,
                "decision_reason": (
                    "accepted"
                    if accepted
                    else str(matching_score_info.get("filter_reason") or "not_matched")
                ),
            }
            if accepted:
                direct_candidates.append(candidate)
        return direct_candidates

    def _recall_stage1_rank_and_select_candidates(
        self,
        *,
        candidates: Sequence[Dict[str, Any]],
        layer_limits: Dict[str, int],
    ) -> List[Dict[str, Any]]:
        """Rank accepted Stage 1 candidates and select them by layer limits."""
        ranked_by_level: Dict[str, List[Dict[str, Any]]] = {
            str(level): []
            for level in (layer_limits or {})
        }
        for candidate in candidates or []:
            if not bool((candidate.get("_recall_decision") or {}).get("accepted")):
                continue
            level = str(candidate.get("index_level") or "")
            if level in ranked_by_level:
                ranked_by_level[level].append(candidate)

        def rank_key(item: Dict[str, Any]) -> Tuple[float, str, int]:
            return (
                float(item.get("_recall_score") or 0.0),
                str(item.get("time_start") or ""),
                int(item.get("target_id") or 0),
            )

        for candidates_for_level in ranked_by_level.values():
            candidates_for_level.sort(key=rank_key, reverse=True)

        layer_limits = {
            str(layer): max(0, int(limit or 0))
            for layer, limit in (layer_limits or {}).items()
        }
        selected_candidates: List[Dict[str, Any]] = []
        seen_targets: set[Tuple[str, int]] = set()
        max_selected_candidates = max(1, sum(layer_limits.values()))

        def append_candidate(candidate: Dict[str, Any]) -> bool:
            try:
                target = (
                    str(candidate.get("target_table") or ""),
                    int(candidate.get("target_id")),
                )
            except (TypeError, ValueError):
                return False
            if target in seen_targets or len(selected_candidates) >= max_selected_candidates:
                return False
            seen_targets.add(target)
            selected_candidates.append(candidate)
            return True

        selected_by_layer: Dict[str, int] = {layer: 0 for layer in layer_limits}
        for layer, limit in layer_limits.items():
            for candidate in ranked_by_level.get(layer, []):
                if selected_by_layer[layer] >= limit:
                    break
                if append_candidate(candidate):
                    selected_by_layer[layer] += 1

        return selected_candidates

    def _recall_stage1_calculate_single_candidate_matching_score(
        self,
        candidate: Dict[str, Any],
        *,
        search_terms: Sequence[str] = (),
        query_entity_names: Sequence[str] = (),
        is_contextual_query: bool = False,
        temporal_bounds: RecallTimeBounds = None,
    ) -> Dict[str, Any]:
        """Score one direct candidate with its type-specific Stage 1 policy."""
        if str(candidate.get("index_level") or "").strip().lower() != "fact":
            return self._recall_stage1_calculate_document_matching_score(
                candidate,
                search_terms=search_terms,
                query_entity_names=query_entity_names,
                is_contextual_query=is_contextual_query,
                temporal_bounds=temporal_bounds,
            )
        topic_match_info = (
            self._recall_calculate_search_terms_overlap_with_candidate_topics(
                candidate,
                search_terms,
                allow_substring=True,
            )
        )
        topic_coverage_ratio = float(topic_match_info.get("coverage") or 0.0)
        keyword_match_info = (
            self._recall_calculate_search_terms_overlap_with_candidate_keywords(
                candidate,
                search_terms,
                allow_substring=True,
            )
        )
        keyword_coverage_ratio = float(
            keyword_match_info.get("coverage") or 0.0
        )

        candidate_source = str(candidate.get("_recall_candidate_source") or "")
        candidate_sources = self._recall_candidate_source_channels(candidate_source)
        entity_match_info = self._recall_stage1_matched_entity_names(
            query_entity_names=query_entity_names,
            candidate_entity_names=candidate.get("entities") or [],
        )
        high_value_entity_matched = bool(
            entity_match_info.get("high_value_entity_matched")
        )

        topic_matched_overlap_score = min(
            self._recall_stage1_topic_overlap_score_weight,
            self._recall_stage1_topic_overlap_score_weight
            * topic_coverage_ratio)
        keyword_overlap_score = min(
            self._recall_stage1_keyword_overlap_score_weight,
            self._recall_stage1_keyword_overlap_score_weight
            * keyword_coverage_ratio,
        )
        entity_score = (
            self._recall_stage1_entity_matched_score
            if high_value_entity_matched
            else 0.0
        )
        time_score_info = self._calculate_recall_candidate_time_score(
            candidate,
            temporal_bounds=temporal_bounds,
        )
        time_weight = self._recall_stage1_time_score_weight
        if is_contextual_query:
            time_weight *= self._recall_stage1_contextual_time_score_multiplier

        score_components = {
            "topic_overlap": round(topic_matched_overlap_score, 4),
            "keyword_overlap": round(keyword_overlap_score, 4),
            "entity_matched": round(entity_score, 4),
            "time_score": round(
                time_weight * float(time_score_info.get("time_score") or 0.0),
                4,
            ),
        }
        score = min(1.0, sum(score_components.values()))
        strong_anchor_reasons: List[str] = []
        if topic_coverage_ratio > 0.0:
            strong_anchor_reasons.append("topic_match")
        if entity_score > 0.0:
            strong_anchor_reasons.append("entity_match")
        has_strong_anchor = bool(strong_anchor_reasons)

        fast_match_details = {
            "topic_match_info": {
                **topic_match_info,
                "keyword_match_info": dict(keyword_match_info),
            },
            "entity_match_info": dict(entity_match_info),
            "time_score_info": dict(time_score_info),
            "score_components": dict(score_components),
        }

        if not has_strong_anchor:
            return {
                "matched": False,
                "score": 0.0,
                "candidate_source": candidate_source,
                "candidate_sources": sorted(candidate_sources - {""}),
                "has_strong_anchor": has_strong_anchor,
                "strong_anchor_reasons": strong_anchor_reasons,
                "filter_reason": "no_strong_anchor",
                "_recall_fast_match_details": fast_match_details,
            }
        return {
            "matched": True,
            "score": round(score, 4),
            "candidate_source": candidate_source,
            "candidate_sources": sorted(candidate_sources - {""}),
            "filter_reason": "",
            "has_strong_anchor": has_strong_anchor,
            "strong_anchor_reasons": strong_anchor_reasons,
            "_recall_fast_match_details": fast_match_details,
        }

    def _recall_stage1_calculate_document_matching_score(
        self,
        candidate: Dict[str, Any],
        *,
        search_terms: Sequence[str],
        query_entity_names: Sequence[str],
        is_contextual_query: bool,
        temporal_bounds: RecallTimeBounds,
    ) -> Dict[str, Any]:
        """Score claim/prospective documents without treating them as facts."""
        object_type = str(candidate.get("index_level") or "").strip().lower()
        raw = (
            candidate.get("_hydrated")
            if isinstance(candidate.get("_hydrated"), dict)
            else {}
        )
        document_values = [
            raw.get("title"),
            raw.get("summary"),
            raw.get("identity_text"),
            *(raw.get("topic_keys") or []),
        ]
        document_match_info = (
            self._recall_calculate_search_terms_overlap_with_topic_values(
                search_terms,
                document_values,
                allow_substring=True,
                minimum_pair_score=0.5,
            )
        )
        document_coverage_ratio = float(
            document_match_info.get("coverage") or 0.0
        )
        entity_match_info = self._recall_stage1_matched_entity_names(
            query_entity_names=query_entity_names,
            candidate_entity_names=candidate.get("entities") or [],
        )
        high_value_entity_matched = bool(
            entity_match_info.get("high_value_entity_matched")
        )
        time_score_info = self._calculate_recall_candidate_time_score(
            candidate,
            temporal_bounds=temporal_bounds,
        )
        time_weight = self._recall_stage1_time_score_weight
        if is_contextual_query:
            time_weight *= self._recall_stage1_contextual_time_score_multiplier
        status = str(raw.get("status") or "").strip().lower()
        allowed_statuses = set(self._recall_stage1_direct_document_statuses(
            object_type,
        ))
        status_eligible = bool(status and status in allowed_statuses)
        document_overlap_score = min(
            self._recall_stage1_topic_overlap_score_weight,
            self._recall_stage1_topic_overlap_score_weight * document_coverage_ratio,
        )
        entity_score = (
            self._recall_stage1_entity_matched_score
            if high_value_entity_matched else 0.0
        )
        status_score = 0.10 if status_eligible else 0.0
        score_components = {
            "document_overlap": round(document_overlap_score, 4),
            "entity_matched": round(entity_score, 4),
            "time_score": round(
                time_weight * float(time_score_info.get("time_score") or 0.0),
                4,
            ),
            "status": round(status_score, 4),
        }
        strong_anchor_reasons: List[str] = []
        if document_coverage_ratio > 0.0:
            strong_anchor_reasons.append("document_match")
        if entity_score > 0.0:
            strong_anchor_reasons.append("entity_match")
        if (
            object_type in {"goal", "plan", "work_item"}
            and status_eligible
            and bool(time_score_info.get("within_temporal_bounds"))
        ):
            strong_anchor_reasons.append("temporal_future_commitment_match")
        has_strong_anchor = bool(strong_anchor_reasons)
        fast_match_details = {
            # Preserve the same detail shape consumed by the Stage 1 evidence
            # profile; document text takes the role that fact topics play for
            # atomic evidence.
            "topic_match_info": {
                **document_match_info,
                "topic_values": list(document_values),
                "keyword_match_info": {},
            },
            "entity_match_info": dict(entity_match_info),
            "time_score_info": dict(time_score_info),
            "score_components": dict(score_components),
        }
        if not status_eligible:
            filter_reason = "inactive_or_unsupported_document_status"
        elif not has_strong_anchor:
            filter_reason = "no_strong_anchor"
        else:
            filter_reason = ""
        return {
            "matched": not bool(filter_reason),
            "score": round(min(1.0, sum(score_components.values())), 4)
            if not filter_reason else 0.0,
            "candidate_source": str(
                candidate.get("_recall_candidate_source") or ""
            ),
            "candidate_sources": sorted(
                self._recall_candidate_source_channels(
                    candidate.get("_recall_candidate_source")
                ) - {""}
            ),
            "filter_reason": filter_reason,
            "has_strong_anchor": has_strong_anchor,
            "strong_anchor_reasons": strong_anchor_reasons,
            "_recall_fast_match_details": fast_match_details,
        }

    def _recall_stage1_build_effective_evidence_terms(
        self,
        *,
        query_terms: Sequence[str],
    ) -> Dict[str, Any]:
        """Filter query terms used only by Stage 1 evidence coverage.

        Retrieval and candidate scoring deliberately continue to use the full
        lexical query.  The profile only removes fixed question and dialogue
        framing terms that cannot be evidence anchors.
        """
        ignored_terms = {
            "什么", "哪个", "哪位", "哪种", "哪本", "哪部", "多少", "几",
            "何时", "哪里", "怎么", "如何", "是否", "吗", "呢",
            "what", "which", "who", "whom", "whose", "when", "where",
            "how", "whether", "howmany", "howmuch", "whichone",
            "whichkind", "whichbook", "whichmovie",
        }
        coverage_terms: List[str] = []
        seen_term_keys: set[str] = set()
        excluded_terms: List[str] = []
        for value in query_terms or ():
            term = self._recall_stage1_clean_anchor(value)
            term_key = self._generate_topic_name_key(term) if term else ""
            if not term or not term_key or term_key in seen_term_keys:
                continue
            seen_term_keys.add(term_key)
            if term_key in ignored_terms:
                excluded_terms.append(term)
            else:
                coverage_terms.append(term)

        return {
            "coverage_terms": coverage_terms,
            "excluded_terms": excluded_terms,
        }

    def _recall_stage1_build_evidence_profile(
        self,
        *,
        candidates: Sequence[Dict[str, Any]],
        query_terms: Sequence[str],
        query_entity_names: Sequence[str],
        query_modes: Sequence[str] = (),
    ) -> Dict[str, Any]:
        """Determine whether selected candidates cover the query's anchors.

        This deliberately consumes only the cached direct-match diagnostics in
        ``_recall_fast_match_details``.  It does not re-score candidates or
        create new topic/entity matches while evaluating the final Stage 1
        result.
        """
        effective_terms_info = self._recall_stage1_build_effective_evidence_terms(
            query_terms=query_terms,
        )
        query_term_by_key: Dict[str, str] = {}
        for value in effective_terms_info.get("coverage_terms") or []:
            term = self._recall_stage1_clean_anchor(value)
            term_key = self._generate_topic_name_key(term) if term else ""
            if not term_key or term_key in query_term_by_key:
                continue
            query_term_by_key[term_key] = term
        effective_term_keys = list(query_term_by_key)
        effective_term_key_set = set(effective_term_keys)
        high_value_entity_by_key: Dict[str, str] = {}
        for value in self._normalize_entity_names(query_entity_names or []):
            entity_key = self._recall_stage1_normalize_match_text(value)
            if (
                entity_key
                and self._recall_stage1_entity_value_class(entity_key) == "high"
            ):
                high_value_entity_by_key.setdefault(entity_key, value)

        matched_topic_term_keys: set[str] = set()
        matched_keyword_term_keys: set[str] = set()
        matched_entity_keys: set[str] = set()
        cached_match_candidate_count = 0
        for candidate in candidates or ():
            details = candidate.get("_recall_fast_match_details")
            if not isinstance(details, dict):
                continue
            topic_match_info = details.get("topic_match_info")
            topic_match_info = (
                topic_match_info
                if isinstance(topic_match_info, dict)
                else {}
            )
            keyword_match_info = topic_match_info.get("keyword_match_info")
            keyword_match_info = (
                keyword_match_info
                if isinstance(keyword_match_info, dict)
                else {}
            )
            entity_match_info = details.get("entity_match_info")
            entity_match_info = (
                entity_match_info
                if isinstance(entity_match_info, dict)
                else {}
            )
            candidate_matched = False
            for value in topic_match_info.get("matched_terms") or []:
                term_key = self._generate_topic_name_key(str(value or ""))
                if term_key in effective_term_key_set:
                    matched_topic_term_keys.add(term_key)
                    candidate_matched = True
            for value in keyword_match_info.get("matched_terms") or []:
                term_key = self._generate_topic_name_key(str(value or ""))
                if term_key in effective_term_key_set:
                    matched_keyword_term_keys.add(term_key)
                    candidate_matched = True
            for value in (
                entity_match_info.get("high_value_matched_entity_names") or []
            ):
                entity_key = self._recall_stage1_normalize_match_text(value)
                if entity_key in high_value_entity_by_key:
                    matched_entity_keys.add(entity_key)
                    candidate_matched = True
            if candidate_matched:
                cached_match_candidate_count += 1

        matched_term_keys = (
            matched_topic_term_keys | matched_keyword_term_keys
        )
        matched_query_terms = [
            query_term_by_key[key]
            for key in effective_term_keys
            if key in matched_term_keys
        ]
        missing_query_terms = [
            query_term_by_key[key]
            for key in effective_term_keys
            if key not in matched_term_keys
        ]
        matched_topic_query_terms = [
            query_term_by_key[key]
            for key in effective_term_keys
            if key in matched_topic_term_keys
        ]
        matched_keyword_query_terms = [
            query_term_by_key[key]
            for key in effective_term_keys
            if key in matched_keyword_term_keys
        ]
        query_term_count = len(effective_term_keys)
        query_term_coverage_ratio = (
            len(matched_term_keys) / query_term_count
            if query_term_count
            else 1.0
        )
        lexical_query_term_coverage_sufficient = (
            query_term_coverage_ratio >= self._recall_stage1_min_term_coverage
        )

        matched_high_value_entities = [
            high_value_entity_by_key[key]
            for key in high_value_entity_by_key
            if key in matched_entity_keys
        ]
        missing_high_value_entities = [
            high_value_entity_by_key[key]
            for key in high_value_entity_by_key
            if key not in matched_entity_keys
        ]
        high_value_entity_count = len(high_value_entity_by_key)
        entity_coverage = (
            len(matched_entity_keys) / high_value_entity_count
            if high_value_entity_count
            else 1.0
        )
        query_entity_coverage_sufficient = not missing_high_value_entities
        modes = set(query_modes or [])
        direct_claim_candidates = [
            candidate for candidate in candidates or []
            if str(candidate.get("index_level") or "") == "entity_claim"
            and not candidate.get("_recall_association_relation")
        ]
        direct_future_commitment_candidates = [
            candidate for candidate in candidates or []
            if str(candidate.get("index_level") or "")
            in {"goal", "plan", "work_item"}
            and not candidate.get("_recall_association_relation")
        ]
        claim_evidence_count = sum(
            1 for candidate in candidates or []
            if str(candidate.get("_recall_association_relation") or "")
            == "claim_evidence"
        )
        intent_evidence_count = sum(
            1 for candidate in candidates or []
            if str(candidate.get("_recall_association_relation") or "")
            == "intent_evidence"
        )
        has_temporal_future_commitment_match = any(
            "temporal_future_commitment_match"
            in (candidate.get("strong_anchor_reasons") or [])
            for candidate in direct_future_commitment_candidates
        )
        # A future-oriented question can intentionally omit the subject of
        # the planned object (for example, "我明天要去哪里？").  In that
        # case the resolved time window is itself the primary retrieval
        # anchor.  Do not reject an otherwise supported prospective object
        # merely because a question-word-shaped lexical term has no match.
        query_term_coverage_sufficient = (
            lexical_query_term_coverage_sufficient
            or (
                "prospective" in modes
                and has_temporal_future_commitment_match
            )
        )
        has_primary_query_coverage = bool(
            matched_topic_term_keys
            or matched_high_value_entities
            or has_temporal_future_commitment_match
        )
        knowledge_evidence_sufficient = (
            "knowledge" not in modes
            or bool(direct_claim_candidates and claim_evidence_count)
        )
        future_commitment_evidence_sufficient = (
            "prospective" not in modes
            or bool(direct_future_commitment_candidates and intent_evidence_count)
        )
        trusted = (
            bool(candidates)
            and has_primary_query_coverage
            and query_term_coverage_sufficient
            and query_entity_coverage_sufficient
            and knowledge_evidence_sufficient
            and future_commitment_evidence_sufficient
        )
        return {
            "trusted": trusted,
            "candidate_count": len(candidates),
            "cached_match_candidate_count": cached_match_candidate_count,
            "coverage_terms": list(effective_terms_info.get("coverage_terms") or []),
            "excluded_query_terms": list(effective_terms_info.get("excluded_terms") or []),
            "query_term_count": query_term_count,
            "matched_query_terms": matched_query_terms,
            "matched_topic_query_terms": matched_topic_query_terms,
            "matched_keyword_query_terms": matched_keyword_query_terms,
            "missing_query_terms": missing_query_terms,
            "query_term_coverage_ratio": round(query_term_coverage_ratio, 4),
            "query_term_coverage_threshold": (
                self._recall_stage1_min_term_coverage
            ),
            "lexical_query_term_coverage_sufficient": (
                lexical_query_term_coverage_sufficient
            ),
            "query_term_coverage_sufficient": query_term_coverage_sufficient,
            "high_value_query_entity_count": high_value_entity_count,
            "matched_high_value_entities": matched_high_value_entities,
            "missing_high_value_entities": missing_high_value_entities,
            "entity_coverage": round(entity_coverage, 4),
            "query_entity_coverage_sufficient": query_entity_coverage_sufficient,
            "query_modes": sorted(modes),
            "direct_entity_claim_count": len(direct_claim_candidates),
            "claim_evidence_count": claim_evidence_count,
            "knowledge_evidence_sufficient": knowledge_evidence_sufficient,
            "direct_future_commitment_count": len(direct_future_commitment_candidates),
            "intent_evidence_count": intent_evidence_count,
            "future_commitment_evidence_sufficient": future_commitment_evidence_sufficient,
            "has_temporal_future_commitment_match": has_temporal_future_commitment_match,
        }

    @staticmethod
    def _recall_stage1_normalize_match_text(value: Any) -> str:
        return _compact_whitespace(value).lower().strip("'\".,:;!?，。！？、；：（）()[]{}")

    @staticmethod
    def _recall_stage1_clean_anchor(value: Any) -> str:
        if isinstance(value, dict):
            value = value.get("name") or value.get("text") or ""
        anchor = _compact_whitespace(value)
        if not anchor:
            return ""
        normalized = anchor.lower().strip("'\".,:;!?，。！？、；：（）()[]{}")
        if normalized in {
            "general", "topic", "state", "entity", "user", "assistant",
            "用户", "助手", "unknown", "unknown_speaker",
        }:
            return ""
        if len(normalized) < 2 or len(normalized) > 80:
            return ""
        return anchor

    def _recall_stage1_resolve_query_entity_names(
        self,
        *,
        query: str,
        database: Optional[SessionDB] = None,
    ) -> List[str]:
        """Resolve entity anchors explicitly present in the query."""
        db = database or self._db
        rows = db.find_entity_nodes_in_text(str(query or ""), limit=12)
        return self._normalize_entity_names([
            row.get("name")
            for row in rows
        ], limit=12)

    def _recall_stage1_matched_entity_names(
        self,
        *,
        query_entity_names: Sequence[str],
        candidate_entity_names: Sequence[Any],
    ) -> Dict[str, Any]:
        """Classify exact entity matches by their anchor value.

        Role-like labels such as ``用户`` and ``助手`` remain visible in the
        diagnostics, but only concrete (high-value) entity matches qualify as
        an entity strong anchor.
        """
        candidate_names = self._normalize_entity_names(
            candidate_entity_names,
            limit=24,
        )
        candidate_high_value_keys: set[str] = set()
        candidate_low_value_keys: Dict[str, set[str]] = {}
        for name in candidate_names:
            normalized = self._recall_stage1_normalize_match_text(name)
            if not normalized:
                continue
            value_class = self._recall_stage1_entity_value_class(normalized)
            if value_class == "high":
                candidate_high_value_keys.add(normalized)
            else:
                candidate_low_value_keys.setdefault(value_class, set()).add(
                    normalized
                )

        matched: List[str] = []
        high_value_matched: List[str] = []
        low_value_matched: List[str] = []
        seen_matched: set[str] = set()
        for name in self._normalize_entity_names(list(query_entity_names or [])):
            normalized = self._recall_stage1_normalize_match_text(name)
            if not normalized:
                continue
            value_class = self._recall_stage1_entity_value_class(normalized)
            if value_class == "high":
                is_match = normalized in candidate_high_value_keys
            else:
                # Low-value aliases represent the same conversational role
                # even when the query and candidate use different languages
                # (for example ``用户`` and ``user``).
                is_match = bool(candidate_low_value_keys.get(value_class))
            if not is_match or normalized in seen_matched:
                continue
            seen_matched.add(normalized)
            matched.append(name)
            if value_class == "high":
                high_value_matched.append(name)
            else:
                low_value_matched.append(name)

        high_value_matched = list(high_value_matched)
        low_value_matched = list(low_value_matched)
        return {
            "matched_entity_names": matched,
            "high_value_matched_entity_names": high_value_matched,
            "low_value_matched_entity_names": low_value_matched,
            "entity_matched": bool(matched),
            "high_value_entity_matched": bool(high_value_matched),
            "low_value_entity_matched": bool(low_value_matched),
            "entity_strong_anchor": bool(high_value_matched),
            "high_value_entity_match_count": len(high_value_matched),
            "low_value_entity_match_count": len(low_value_matched),
        }

    @classmethod
    def _recall_stage1_entity_value_class(cls, value: Any) -> str:
        normalized = cls._recall_stage1_normalize_match_text(value)
        for value_class, aliases in _LOW_VALUE_ENTITY_ALIASES.items():
            if normalized in {
                cls._recall_stage1_normalize_match_text(alias)
                for alias in aliases
            }:
                return value_class
        return "high"

    @classmethod
    def _recall_stage1_contains_anchor(cls, query: str, anchor: str) -> bool:
        query_text = _compact_whitespace(query).lower()
        anchor_text = _compact_whitespace(anchor).lower()
        if not query_text or not anchor_text:
            return False
        if re.search(r"[\u4e00-\u9fff]", anchor_text):
            return anchor_text.replace(" ", "") in query_text.replace(" ", "")
        pattern = rf"(?<![a-z0-9]){re.escape(anchor_text)}(?![a-z0-9])"
        return re.search(pattern, query_text) is not None

    def _calculate_recall_candidate_time_score(
        self,
        candidate: Dict[str, Any],
        *,
        temporal_bounds: RecallTimeBounds = None,
    ) -> Dict[str, Any]:
        """Calculate continuous temporal proximity for one recall candidate.

        A closed ``[start, end]`` window gives every in-range candidate full
        temporal relevance. For an open start, proximity is measured from
        ``temporal_bounds.end`` so older memories before that end remain
        distinguishable.
        """
        raw = (
            candidate.get("_hydrated")
            if isinstance(candidate.get("_hydrated"), dict)
            else {}
        )
        window_start_text, window_end_text = temporal_bounds or (None, None)
        window_start = self._recall_stage1_parse_datetime(window_start_text)
        window_end = self._recall_stage1_parse_datetime(window_end_text)
        reference = window_end
        if reference is None:
            return {
                "time_score": 0.0,
                "time_distance_seconds": None,
                "candidate_time": "",
                "within_temporal_bounds": False,
                "half_life_seconds": 0,
            }

        time_values: List[Tuple[datetime, str]] = []

        def add_time(value: Any) -> None:
            parsed = self._recall_stage1_parse_datetime(value)
            text = _compact_whitespace(value)
            if parsed is not None and text:
                time_values.append((parsed, text))

        add_time(raw.get("dialogue_time_key"))
        add_time(candidate.get("time_end"))
        add_time(candidate.get("time_start"))
        half_life_seconds = self._recall_fact_time_score_half_life_seconds

        if not time_values:
            return {
                "time_score": 0.0,
                "time_distance_seconds": None,
                "candidate_time": "",
                "within_temporal_bounds": False,
                "half_life_seconds": half_life_seconds,
            }

        if window_start is not None and window_end is not None:
            def distance_to_window(value: datetime) -> float:
                if window_start is not None and value < window_start:
                    return (window_start - value).total_seconds()
                if window_end is not None and value > window_end:
                    return (value - window_end).total_seconds()
                return 0.0

            candidate_time, candidate_time_text = min(
                time_values,
                key=lambda item: distance_to_window(item[0]),
            )
            distance_seconds = distance_to_window(candidate_time)
        else:
            candidate_time, candidate_time_text = max(
                time_values,
                key=lambda item: item[0],
            )
            distance_seconds = abs((reference - candidate_time).total_seconds())

        time_score = math.exp(
            -max(0.0, distance_seconds) / max(1, half_life_seconds)
        )
        return {
            "time_score": round(float(time_score), 4),
            "time_distance_seconds": round(float(distance_seconds), 2),
            "candidate_time": candidate_time_text,
            "within_temporal_bounds": bool(distance_seconds <= 0.0),
            "half_life_seconds": int(half_life_seconds),
        }

    @staticmethod
    def _recall_stage1_parse_datetime(value: Any) -> Optional[datetime]:
        text = str(value or "").strip()
        if not text:
            return None
        normalized = text.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            try:
                parsed = datetime.strptime(text[:19], "%Y-%m-%d %H:%M:%S")
            except ValueError:
                return None
        if parsed.tzinfo is not None:
            return parsed.astimezone(timezone.utc).replace(tzinfo=None)
        return parsed

    @staticmethod
    def _recall_is_contextual_query(query: str) -> bool:
        lower = str(query or "").lower()
        markers = (
            "这个", "那个", "刚才", "前面", "继续", "然后", "目前", "接下来",
            "what about it", "that one", "continue", "then", "next",
        )
        return any(marker in lower for marker in markers)

    @staticmethod
    def _recall_stage1_requires_semantic_search(query: str) -> bool:
        lower = str(query or "").lower()
        markers = (
            "为什么", "为何", "原因", "如何", "怎么", "比较", "区别", "历史",
            "趋势", "变化", "全部", "所有", "之前", "之后", "最早", "第一次",
            "why", "how", "compare", "difference", "history", "trend", "before",
            "after", "all", "first",
        )
        return any(marker in lower for marker in markers)

    def _merge_recall_stage2_seed_candidates(
        self,
        *,
        stage1_lexical_candidates: Sequence[Dict[str, Any]],
        stage2_lexical_candidates: Sequence[Dict[str, Any]],
        stage2_embedding_candidates: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Deduplicate the independent Stage 2 direct-retrieval channels."""
        seed_candidates = [
            *list(stage1_lexical_candidates or []),
            *list(stage2_lexical_candidates or []),
            *list(stage2_embedding_candidates or []),
        ]

        merged_candidates: Dict[Tuple[str, int], Dict[str, Any]] = {}

        def merge_candidate_group(
            candidates: Sequence[Dict[str, Any]],
        ) -> None:
            for raw_candidate in candidates or []:
                try:
                    target = (
                        str(raw_candidate.get("target_table") or ""),
                        int(raw_candidate.get("target_id")),
                    )
                except (TypeError, ValueError):
                    continue
                candidate = dict(raw_candidate)
                existing = merged_candidates.get(target)
                if existing is None:
                    merged_candidates[target] = candidate
                    continue
                if float(candidate.get("_recall_bm25_score") or 0.0) > float(
                    existing.get("_recall_bm25_score") or 0.0
                ):
                    existing["_recall_bm25_score"] = candidate.get(
                        "_recall_bm25_score"
                    )
                    existing["_bm25_score"] = candidate.get("_bm25_score")
                entity_names = self._normalize_entity_names([
                    *(existing.get("_recall_entity_names") or []),
                    *(candidate.get("_recall_entity_names") or []),
                ], limit=24)
                if entity_names:
                    existing["_recall_entity_names"] = entity_names

        merge_candidate_group(stage1_lexical_candidates)
        merge_candidate_group(stage2_lexical_candidates)
        merge_candidate_group(stage2_embedding_candidates)
        return {
            "seed_candidates": seed_candidates,
            "stage1_lexical_candidates": list(stage1_lexical_candidates or []),
            "stage2_lexical_candidates": list(stage2_lexical_candidates or []),
            "stage2_embedding_candidates": list(stage2_embedding_candidates or []),
            "merged_candidates": list(merged_candidates.values()),
            "merged_count": len(merged_candidates),
            "seed_count": len(seed_candidates),
        }

    def _recall_stage2_calculate_candidate_matching_score(
        self,
        *,
        candidates: Sequence[Dict[str, Any]],
        search_terms: Sequence[str],
        query_terms: Sequence[str],
        query_embedding: Optional[np.ndarray],
        query_entity_names: Sequence[str] = (),
        is_contextual_query: bool = False,
        temporal_bounds: RecallTimeBounds = None,
    ) -> List[Dict[str, Any]]:
        """Score Stage 2 seeds in place and return accepted direct candidates."""
        direct_candidates: List[Dict[str, Any]] = []
        for candidate in candidates or []:
            if str(candidate.get("index_level") or "").strip().lower() != "fact":
                continue
            matching_score_info = (
                self._recall_stage2_calculate_single_candidate_matching_score(
                    candidate,
                    search_terms=search_terms,
                    query_terms=query_terms,
                    query_embedding=query_embedding,
                    query_entity_names=query_entity_names,
                    is_contextual_query=is_contextual_query,
                    temporal_bounds=temporal_bounds,
                )
            )
            match_details = dict(
                matching_score_info.get("_recall_stage2_match_details") or {}
            )
            candidate.update({
                key: value
                for key, value in matching_score_info.items()
                if key != "_recall_stage2_match_details"
            })
            candidate["_recall_stage2_match_details"] = match_details
            candidate["_recall_type"] = "fact"
            candidate["_recall_score"] = self._clamp_float(
                matching_score_info.get("score"),
                0.0,
                1.0,
                0.0,
            )
            filter_reason = str(
                matching_score_info.get("filter_reason") or ""
            )
            if filter_reason:
                candidate["_recall_drop_reason"] = filter_reason
            matched = bool(matching_score_info.get("matched"))
            candidate["_recall_decision"] = {
                "accepted": matched,
                "decision_reason": (
                    "accepted_stage2_direct_retrieval"
                    if matched
                    else filter_reason or "stage2_direct_score_zero"
                ),
            }
            if bool(
                (candidate.get("_recall_decision") or {}).get("accepted")
            ):
                direct_candidates.append(candidate)
        return direct_candidates

    def _merge_recall_direct_and_associative_candidates(
        self,
        *,
        direct_candidates: Sequence[Dict[str, Any]],
        association_candidates: Sequence[Dict[str, Any]],
        stage_name: str,
    ) -> List[Dict[str, Any]]:
        """Merge direct and propagated candidates without re-scoring relations.

        The same merge path is shared by Stage 1 and Stage 2. Association
        candidates carry common score and relation fields; their source marks
        which recall stage created the association.
        """
        if stage_name not in {"stage1", "stage2"}:
            raise ValueError(f"Unsupported recall stage: {stage_name}")
        candidates_by_target: Dict[Tuple[str, int], Dict[str, Any]] = {}
        for candidate in direct_candidates or []:
            try:
                target = (
                    str(candidate.get("target_table") or ""),
                    int(candidate.get("target_id")),
                )
            except (TypeError, ValueError):
                continue
            candidates_by_target[target] = candidate
        for association_candidate in association_candidates or []:
            try:
                target = (
                    str(association_candidate.get("target_table") or ""),
                    int(association_candidate.get("target_id")),
                )
            except (TypeError, ValueError):
                continue
            association_score = self._clamp_float(
                association_candidate.get("_recall_association_score"),
                0.0,
                1.0,
                0.0,
            )
            association_relation = str(
                association_candidate.get("_recall_association_relation") or ""
            )
            existing = candidates_by_target.get(target)
            if existing is None:
                candidates_by_target[target] = association_candidate
                continue
            existing_association_score = self._clamp_float(
                existing.get("_recall_association_score"),
                0.0,
                1.0,
                0.0,
            )
            if association_score >= existing_association_score:
                existing["_recall_association_score"] = association_score
                existing["_recall_association_relation"] = association_relation
            else:
                association_score = existing_association_score
                association_relation = str(
                    existing.get("_recall_association_relation") or ""
                )
            existing["_recall_association_score"] = association_score
            existing["_recall_association_relation"] = association_relation
            current_score = self._clamp_float(
                existing.get("_recall_score"),
                0.0,
                1.0,
                0.0,
            )
            existing["_recall_score"] = round(max(
                current_score,
                association_score,
            ), 4)
            evidence = list(
                existing.get("evidence")
                or existing.get("_recall_fast_match_evidence")
                or []
            )
            if "associative_recall" not in evidence:
                evidence.append("associative_recall")
            existing["evidence"] = evidence
            existing["matched"] = True
            existing["filter_reason"] = ""
            if not bool(
                (existing.get("_recall_decision") or {}).get("accepted")
            ):
                existing["_recall_decision"] = {
                    "accepted": True,
                    "decision_reason": "accepted_associative_recall",
                }

            # Preserve the best lexical BM25 evidence when an associated fact
            # is the same target as a direct seed.
            try:
                existing_bm25 = float(existing.get("_bm25_score"))
            except (TypeError, ValueError):
                existing_bm25 = None
            try:
                candidate_bm25 = float(association_candidate.get("_bm25_score"))
            except (TypeError, ValueError):
                candidate_bm25 = None
            if candidate_bm25 is not None and (
                existing_bm25 is None or candidate_bm25 < existing_bm25
            ):
                existing["_bm25_score"] = candidate_bm25
                existing["_recall_bm25_score"] = self._clamp_float(
                    association_candidate.get("_recall_bm25_score"),
                    0.0,
                    1.0,
                    0.0,
                )
                existing["_recall_bm25_rank"] = association_candidate.get(
                    "_recall_bm25_rank"
                )

        return list(candidates_by_target.values())

    def _recall_stage2_rank_and_select_candidates(
        self,
        *,
        candidates: Sequence[Dict[str, Any]],
        final_candidate_limits: Dict[str, int],
    ) -> List[Dict[str, Any]]:
        """Rank direct and associative candidates by their already-final score."""
        ranked_candidates = sorted(
            (
                candidate
                for candidate in candidates or []
                if str(candidate.get("index_level") or "") == "fact"
                and bool((candidate.get("_recall_decision") or {}).get("accepted"))
            ),
            key=lambda item: (
                float(item.get("_recall_score") or 0.0),
                str(item.get("time_start") or ""),
                int(item.get("target_id") or 0),
            ),
            reverse=True,
        )
        return ranked_candidates[: max(
            0,
            int(final_candidate_limits.get("fact", 0) or 0),
        )]

    def _retrieve_recall_stage2_seed_candidates(
        self,
        *,
        stage1_lexical_candidates: Sequence[Dict[str, Any]],
        direct_object_types: Sequence[str],
        prospective_query_profile: Dict[str, Any],
        search_terms: Sequence[str],
        query_embedding: Optional[np.ndarray],
        temporal_bounds: RecallTimeBounds,
        temporal_mode: str,
        seed_channel_limits: Dict[str, Dict[str, int]],
        database: Optional[SessionDB] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve and deduplicate all direct Stage 2 seed channels.

        Stage 1 contributes its existing direct seeds. Stage 2 adds candidates
        retrieved with its expanded search terms (using the same type-aware
        prospective time-slot strategy) and top full-corpus document
        embeddings. Association
        expansion is intentionally performed by the caller after direct
        scoring.
        """
        lexical_candidate_limits = dict(
            seed_channel_limits.get("stage2_lexical") or {}
        )
        embedding_candidate_limits = dict(
            seed_channel_limits.get("stage2_embedding") or {}
        )
        lexical_candidates = (
            self._retrieve_recall_document_lexical_seed_candidates(
                terms=list(search_terms),
                direct_object_types=direct_object_types,
                temporal_bounds=temporal_bounds,
                temporal_mode=temporal_mode,
                candidate_limits=lexical_candidate_limits,
                candidate_source="stage2_lexical",
                prospective_query_profile=prospective_query_profile,
                database=database,
            )
            if search_terms or prospective_query_profile.get(
                "is_prospective_query"
            )
            else []
        )
        embedding_candidates = self._retrieve_recall_document_embedding_seed_candidates(
            query_embedding=query_embedding,
            direct_object_types=direct_object_types,
            temporal_bounds=temporal_bounds,
            temporal_mode=temporal_mode,
            candidate_limits=embedding_candidate_limits,
            candidate_source="stage2_embedding",
            database=database,
        )
        merged_report = self._merge_recall_stage2_seed_candidates(
            stage1_lexical_candidates=stage1_lexical_candidates,
            stage2_lexical_candidates=lexical_candidates,
            stage2_embedding_candidates=embedding_candidates,
        )
        return list(merged_report.get("merged_candidates") or [])

    def _retrieve_recall_document_embedding_seed_candidates(
        self,
        *,
        query_embedding: Optional[np.ndarray],
        direct_object_types: Sequence[str],
        temporal_bounds: RecallTimeBounds,
        temporal_mode: str,
        candidate_limits: Dict[str, int],
        candidate_source: str,
        database: Optional[SessionDB] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve full-corpus embedding seeds from recall documents.

        Similarity is computed and capped separately for each memory object
        type.  This avoids allowing the much larger fact corpus to suppress
        claim or prospective candidates before their type-specific Stage 2
        matcher is available.
        """
        if query_embedding is None:
            return []
        direct_types = list(dict.fromkeys(
            str(value or "").strip().lower()
            for value in direct_object_types or []
            if str(value or "").strip().lower()
            in {"fact", "entity_claim", "goal", "plan", "work_item"}
        ))
        db = database or self._db
        candidates: List[Dict[str, Any]] = []
        for object_type in direct_types:
            candidate_limit = max(
                1,
                int(candidate_limits.get(object_type, 1) or 1),
            )
            rows = db.memory_recall_documents_with_identity_embeddings(
                object_type=object_type,
                statuses=self._recall_stage1_direct_document_statuses(object_type),
            )
            ranked_rows: List[Tuple[float, Dict[str, Any]]] = []
            for row in rows:
                similarity = max(0.0, _cal_embedding_cosine_similarity(
                    query_embedding,
                    row.get("identity_text_embedding"),
                ))
                if (
                    object_type == "fact"
                    and similarity < self._recall_stage2_fact_min_embedding_similarity
                ):
                    continue
                ranked_rows.append((similarity, row))
            ranked_rows.sort(
                key=lambda item: (
                    item[0],
                    str(item[1].get("time_end") or ""),
                    int(item[1].get("object_id") or 0),
                ),
                reverse=True,
            )
            retrieval_limit = (
                max(24, candidate_limit * 4)
                if any(temporal_bounds or (None, None))
                else candidate_limit
            )
            selected_rows = [
                row for _similarity, row in ranked_rows[:retrieval_limit]
            ]
            similarity_by_object_id = {
                int(row.get("object_id") or 0): similarity
                for similarity, row in ranked_rows[:retrieval_limit]
            }
            object_candidates = self._make_recall_document_candidates(
                rows=selected_rows,
                candidate_source=candidate_source,
                temporal_bounds=temporal_bounds,
                temporal_mode=temporal_mode,
                per_type_limit=candidate_limit,
                database=db,
            )
            for candidate in object_candidates:
                candidate["_recall_embedding_seed_similarity"] = round(
                    float(similarity_by_object_id.get(
                        int(candidate.get("target_id") or 0),
                        0.0,
                    )),
                    4,
                )
            candidates.extend(object_candidates)
        return candidates

    def _log_recall_stage2_seed_candidates(
        self,
        *,
        seed_candidates: Sequence[Dict[str, Any]],
    ) -> None:
        """Log all Stage 2 seed channels before direct matching."""
        seed_candidates = list(seed_candidates or [])
        source_counts = {
            source: sum(
                1
                for candidate in seed_candidates
                if candidate.get("_recall_candidate_source") == source
            )
            for source in (
                "stage1_lexical",
                "stage1_prospective_query",
                "stage2_lexical",
                "stage2_prospective_query",
                "stage2_embedding",
            )
        }
        payload: Dict[str, Any] = {
            "seed_candidate_count": len(seed_candidates),
            "seed_by_level": self._recall_count_candidates_by_level(
                seed_candidates
            ),
            "stage1_lexical_seed_count": source_counts["stage1_lexical"],
            "stage1_prospective_seed_count": source_counts[
                "stage1_prospective_query"
            ],
            "stage2_lexical_seed_count": source_counts["stage2_lexical"],
            "stage2_prospective_seed_count": source_counts[
                "stage2_prospective_query"
            ],
            "stage2_embedding_seed_count": source_counts["stage2_embedding"],
            "candidates": self._recall_log_candidate_items(
                seed_candidates,
                detailed=self._recall_detailed_logging,
                limit=None if self._recall_detailed_logging else 12,
                stage="stage2_seed",
            ),
        }
        self._log_info("memory_recall_stage2", "seeds_retrieved", payload)

    def _process_recall_stage2(
        self,
        *,
        original_query: str,
        time_stripped_query: str,
        temporal_bounds: RecallTimeBounds,
        reference_time: str,
        temporal_mode: str = "dialogue_time",
        stage1_report: Optional[Dict[str, Any]] = None,
        prompt_language: str = "zh",
        database: Optional[SessionDB] = None,
    ) -> str:
        """Run the existing LLM and semantic-search recall pipeline.

        This is intentionally separated from ``recall`` so a deterministic
        Stage 1 fast path can decide whether this more expensive path is
        necessary without duplicating its query preparation and logging.
        """
        stage_started_at = time.monotonic()
        time_stripped_query = str(time_stripped_query or "")
        fallback_temporal_bounds = temporal_bounds
        fallback_temporal_mode = self._normalize_recall_temporal_mode(
            temporal_mode
        )
        analysis_reference_time = str(reference_time)
        self._log_info("memory_recall_stage2", "start", {
            "original_query": self._format_log_text(original_query, limit=500),
            "time_stripped_query": self._format_log_text(
                time_stripped_query,
                limit=500,
            ),
            "top_k": self._top_k,
            "budget": self._recall_budget,
            "fallback_time_start": (fallback_temporal_bounds or (None, None))[0],
            "fallback_time_end": (fallback_temporal_bounds or (None, None))[1],
            "fallback_temporal_mode": fallback_temporal_mode,
            "reference_time": analysis_reference_time,
            "prompt_language": prompt_language,
        })
        query_analysis_info = self._analyze_recall_query(
            original_query,
            reference_time=analysis_reference_time,
            prompt_language=prompt_language,
        )
        temporal_resolution = self._resolve_recall_stage2_temporal_constraints(
            original_query=original_query,
            fallback_temporal_bounds=fallback_temporal_bounds,
            fallback_temporal_mode=fallback_temporal_mode,
            llm_temporal_bounds=query_analysis_info.get("temporal_bounds"),
            llm_temporal_mode=query_analysis_info.get("temporal_mode"),
        )
        temporal_bounds = temporal_resolution["effective_temporal_bounds"]
        temporal_mode = temporal_resolution["effective_temporal_mode"]
        llm_keywords = self._normalize_string_list(
            query_analysis_info.get("keywords"),
            limit=12,
        )
        llm_entities = self._normalize_entity_names(
            query_analysis_info.get("entities"),
            limit=12,
        )
        direct_object_types = self._normalize_recall_object_types(
            query_analysis_info.get("recall_object_types"),
        )
        prospective_query_profile = self._recall_stage2_prospective_query_profile(
            original_query=original_query,
            llm_value=query_analysis_info.get(
                "is_prospective_time_slot_query"
            ),
            temporal_bounds=temporal_bounds,
            reference_time=analysis_reference_time,
        )
        query_entity_names = self._normalize_entity_names(
            [
                *llm_entities,
                *self._recall_stage1_resolve_query_entity_names(
                    query=original_query,
                    database=database,
                ),
            ],
            limit=24,
        )
        is_contextual_query = self._recall_is_contextual_query(
            original_query
        )
        reference_time = (
            (temporal_bounds or (None, None))[1]
            or analysis_reference_time
        )
        # Stage 1 contributes its existing direct seeds as one Stage 2 source.
        # The Stage 2 lexical source itself is reserved for LLM-derived terms.
        supplement_terms = self._build_recall_search_terms(
            "",
            keywords=llm_keywords,
            entities=llm_entities,
        )
        query_terms = self._lexical_search_terms_for_text(
            time_stripped_query,
            limit=32,
            preserve_phrase=False,
        )
        retrieval_text = query_analysis_info.get("query_rewrite") or ""
        rewrite_terms = self._lexical_search_terms_for_text(
            retrieval_text,
            limit=32,
            preserve_phrase=False,
        )
        search_terms = list(dict.fromkeys([
            *supplement_terms,
            *rewrite_terms,
            *query_terms,
        ]))
        query_identity_text = self._format_recall_query_identity_text(
            time_stripped_query,
            retrieval_text=retrieval_text,
            keywords=llm_keywords,
            entities=llm_entities,
        )
        query_identity_embedding = self._generate_embedding_vector(query_identity_text)
        candidate_limits = self._recall_stage2_candidate_limits(
            top_k=self._top_k,
        )
        seed_channel_limits = candidate_limits["seed_channel_limits"]
        selected_candidate_limits = candidate_limits["selected_limits"]
        association_per_relation_limit = candidate_limits[
            "association_per_relation_limit"
        ]
        self._log_info("memory_recall_stage2", "query_analyzed", {
            "query_analysis_info": query_analysis_info,
            "temporal_resolution": temporal_resolution,
            "keywords": llm_keywords,
            "entities": llm_entities,
            "direct_object_types": direct_object_types,
            "prospective_query_profile": prospective_query_profile,
            "query_entity_names": query_entity_names,
            "is_contextual_query": is_contextual_query,
            "reference_time": reference_time,
            "supplement_terms": supplement_terms,
            "query_terms": query_terms,
            "rewrite_terms": rewrite_terms,
            "search_terms": search_terms,
            "retrieval_text": self._format_log_text(retrieval_text, limit=500),
            "identity_text": self._format_log_text(query_identity_text, limit=500),
            "query_embedding_available": query_identity_embedding is not None,
            "candidate_limits": candidate_limits,
            "budget": self._recall_budget,
            "temporal_mode": temporal_mode,
        })
        stage1_lexical_seed_candidates = list(
            (stage1_report or {}).get("direct_seed_candidates")
            or (stage1_report or {}).get("seed_candidates")
            or []
        )
        seed_candidates = self._retrieve_recall_stage2_seed_candidates(
            stage1_lexical_candidates=stage1_lexical_seed_candidates,
            direct_object_types=direct_object_types,
            prospective_query_profile=prospective_query_profile,
            search_terms=search_terms,
            query_embedding=query_identity_embedding,
            temporal_bounds=temporal_bounds,
            temporal_mode=temporal_mode,
            seed_channel_limits=seed_channel_limits,
            database=database,
        )
        self._log_recall_stage2_seed_candidates(
            seed_candidates=seed_candidates,
        )
        direct_candidates = self._recall_stage2_calculate_candidate_matching_score(
            candidates=seed_candidates,
            search_terms=search_terms,
            query_terms=query_terms,
            query_embedding=query_identity_embedding,
            query_entity_names=query_entity_names,
            is_contextual_query=is_contextual_query,
            temporal_bounds=temporal_bounds,
        )
        self._log_recall_direct_candidates(
            stage_name="stage2",
            seed_candidates=seed_candidates,
            direct_candidates=direct_candidates,
        )
        association_candidates = (
            self._retrieve_association_candidates_using_seed_candidates(
                seed_candidates=direct_candidates,
                temporal_bounds=temporal_bounds,
                temporal_mode=temporal_mode,
                limit=association_per_relation_limit,
                candidate_source_prefix="stage2",
                database=database,
            )
        )
        expanded_candidates = self._merge_recall_direct_and_associative_candidates(
            direct_candidates=direct_candidates,
            association_candidates=association_candidates,
            stage_name="stage2",
        )
        self._log_recall_association_candidates(
            stage_name="stage2",
            expanded_candidates=expanded_candidates,
            association_candidates=association_candidates,
        )

        ranked_candidates = self._recall_stage2_rank_and_select_candidates(
            candidates=expanded_candidates,
            final_candidate_limits=selected_candidate_limits,
        )
        self._log_recall_selected_candidates(
            stage_name="stage2",
            selected_candidates=ranked_candidates,
            expanded_candidates=expanded_candidates,
        )
        memory_text = self._build_memory_retrieved_format_text(
            entries=ranked_candidates,
            prompt_language=prompt_language,
        )
        self._log_info("memory_recall_stage2", "finish", {
            "status": "ok" if memory_text else "empty",
            "elapsed_ms": round((time.monotonic() - stage_started_at) * 1000, 2),
            "retrieved_chars": len(memory_text or ""),
        })
        return memory_text

    def _make_recall_document_candidates(
        self,
        *,
        rows: Sequence[Dict[str, Any]],
        candidate_source: str,
        temporal_bounds: RecallTimeBounds,
        temporal_mode: str,
        per_type_limit: int,
        database: SessionDB,
    ) -> List[Dict[str, Any]]:
        """Build and normalize candidates from one recall-document result set."""
        rows = list(rows or [])
        if not rows:
            return []
        limit = max(1, int(per_type_limit or 1))
        candidates: List[Dict[str, Any]] = []
        for row in rows:
            candidate = self._make_recall_document_candidate(
                row=row,
                candidate_source=candidate_source,
                temporal_bounds=temporal_bounds,
                temporal_mode=temporal_mode,
                database=database,
            )
            if candidate:
                candidates.append(candidate)
        self._normalize_recall_candidate_bm25_scores(candidates)
        return candidates[:limit]

    def _make_recall_document_candidate(
        self,
        *,
        row: Dict[str, Any],
        object_type: Optional[str] = None,
        candidate_source: str,
        temporal_bounds: RecallTimeBounds,
        temporal_mode: str = "dialogue_time",
        entity_names_by_id: Optional[Dict[int, str]] = None,
        database: Optional[SessionDB] = None,
    ) -> Optional[Dict[str, Any]]:
        """Convert a hydrated recall object into the shared candidate shape."""
        object_type = str(
            object_type or row.get("object_type") or ""
        ).strip().lower()
        if object_type == "fact":
            is_recall_document = (
                str(row.get("object_type") or "").strip().lower() == "fact"
            )
            if is_recall_document:
                if database is None:
                    return None
                try:
                    fact_id = int(row.get("object_id"))
                except (TypeError, ValueError):
                    return None
                fact_rows = database.get_memory_facts_by_ids([fact_id])
                if not fact_rows:
                    return None
                fact_row = dict(fact_rows[0])
                fact_row["_bm25_score"] = row.get("_bm25_score")
                row = fact_row
            try:
                target_id = int(row.get("id"))
            except (TypeError, ValueError):
                return None
            if not self._fact_matches_time_bounds(
                row,
                temporal_mode=temporal_mode,
                temporal_bounds=temporal_bounds,
            ):
                return None
            fact_times = self._fact_time_values(row, temporal_mode)
            time_start = fact_times[0] if fact_times else ""
            time_end = fact_times[-1] if fact_times else ""
            query_start, query_end = temporal_bounds or (None, None)
            if query_start and time_end and time_end < str(query_start):
                return None
            if query_end and time_start and time_start > str(query_end):
                return None
            hydrated = dict(row)
            hydrated.pop("episode_id", None)
            hydrated.pop("embedding", None)
            hydrated.pop("identity_text_embedding", None)
            hydrated.pop("canonical_name_embedding", None)
            metadata = dict(row.get("metadata") or {})
            metadata["_matched_via"] = [candidate_source]
            return {
                "target_table": "memory_facts",
                "target_id": target_id,
                "index_level": "fact",
                "memory_path": "fact",
                "title": _compact_whitespace(row.get("summary") or "")[:120],
                "summary_for_retrieval": _compact_whitespace(
                    row.get("summary") or ""
                ),
                "identity_text": _compact_whitespace(
                    row.get("identity_text") or ""
                ),
                "keywords": row.get("keywords") or "",
                "entities": row.get("entities") or [],
                "participants": row.get("participants") or [],
                "time_start": time_start,
                "time_end": time_end,
                "importance": row.get("importance") or 0.5,
                "confidence": row.get("confidence") or 0.8,
                "embedding": row.get("identity_text_embedding"),
                "metadata": metadata,
                "_hydrated": hydrated,
                "_bm25_score": row.get("_bm25_score"),
                "_recall_candidate_source": candidate_source,
            }

        target_table_by_type = {
            "entity_claim": "memory_entity_claims",
            "goal": "memory_goals",
            "plan": "memory_plans",
            "work_item": "memory_work_items",
        }
        target_table = target_table_by_type.get(object_type)
        if not target_table:
            return None
        try:
            target_id = int(row.get("object_id"))
        except (TypeError, ValueError):
            return None
        time_start = self._normalize_event_time_text(row.get("time_start"))
        time_end = self._normalize_event_time_text(row.get("time_end")) or time_start
        query_start, query_end = temporal_bounds or (None, None)
        if time_start or time_end:
            if query_start and time_end and time_end < str(query_start):
                return None
            if query_end and time_start and time_start > str(query_end):
                return None
        entity_ids = [
            int(entity_id)
            for entity_id in row.get("entity_ids") or []
            if str(entity_id).strip().isdigit() and int(entity_id) > 0
        ]
        if entity_names_by_id is None and database is not None:
            entity_names_by_id = database.get_entity_names_by_ids(entity_ids)
        entity_names_by_id = entity_names_by_id or {}
        entities = [
            entity_names_by_id[entity_id]
            for entity_id in entity_ids
            if entity_id in entity_names_by_id
        ]
        hydrated = dict(row)
        hydrated["entity_names"] = list(entities)
        metadata = dict(row.get("metadata") or {})
        metadata["_matched_via"] = [candidate_source]
        temporal_match = str(row.get("_recall_temporal_match") or "").strip()
        if temporal_match:
            metadata["_recall_temporal_match"] = temporal_match
        candidate = {
            "target_table": target_table,
            "target_id": target_id,
            "index_level": object_type,
            "memory_path": object_type,
            "title": _compact_whitespace(row.get("title") or row.get("summary") or "")[:120],
            "summary_for_retrieval": _compact_whitespace(row.get("summary") or ""),
            "identity_text": _compact_whitespace(row.get("identity_text") or ""),
            "keywords": [],
            "entities": entities,
            "participants": [],
            "time_start": time_start,
            "time_end": time_end,
            "importance": float(row.get("importance") or 0.0),
            "confidence": float(row.get("confidence") or 0.0),
            "embedding": row.get("identity_text_embedding"),
            "metadata": metadata,
            "_hydrated": hydrated,
            "_bm25_score": row.get("_bm25_score"),
            "_recall_candidate_source": candidate_source,
        }
        if temporal_match:
            candidate["_recall_temporal_match"] = temporal_match
        return candidate

    @staticmethod
    def _normalize_recall_candidate_bm25_scores(
        candidates: Sequence[Dict[str, Any]],
    ) -> None:
        """Normalize BM25 only within one type-specific retrieval channel."""
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for candidate in candidates or []:
            try:
                raw_bm25_score = float(candidate.get("_bm25_score"))
            except (TypeError, ValueError):
                candidate["_recall_bm25_score"] = 0.0
                continue
            if not math.isfinite(raw_bm25_score):
                candidate["_recall_bm25_score"] = 0.0
                continue
            scored.append((raw_bm25_score, candidate))
        scored.sort(key=lambda item: item[0])
        count = len(scored)
        for position, (_raw_bm25_score, candidate) in enumerate(scored):
            candidate["_recall_bm25_score"] = round(
                0.80 if count == 1 else 0.42 + 0.50 * (1.0 - position / (count - 1)),
                4,
            )
            candidate["_recall_bm25_rank"] = position + 1

    def _recall_stage2_calculate_single_candidate_matching_score(
        self,
        candidate: Dict[str, Any],
        *,
        search_terms: Sequence[str],
        query_terms: Sequence[str],
        query_embedding: Optional[np.ndarray],
        query_entity_names: Sequence[str] = (),
        is_contextual_query: bool = False,
        temporal_bounds: RecallTimeBounds = None,
    ) -> Dict[str, Any]:
        """Calculate matching score for one Stage 2 direct candidate only."""
        search_topic_match_info = (
            self._recall_calculate_search_terms_overlap_with_candidate_topics(
                candidate,
                search_terms,
                allow_substring=True,
            )
        )
        query_topic_match_info = (
            self._recall_calculate_search_terms_overlap_with_candidate_topics(
                candidate,
                query_terms,
                allow_substring=True,
            )
        )
        all_topic_values = list(
            search_topic_match_info.get("topic_values") or []
        )
        topic_match_ratio = float(
            search_topic_match_info.get("coverage") or 0.0
        )
        keyword_match_info = (
            self._recall_calculate_search_terms_overlap_with_candidate_keywords(
                candidate,
                search_terms,
                allow_substring=True,
            )
        )
        all_keyword_values = list(keyword_match_info.get("keyword_values") or [])
        keyword_match_ratio = float(keyword_match_info.get("coverage") or 0.0)
        similarity = max(0.0, _cal_embedding_cosine_similarity(
            query_embedding,
            candidate.get("embedding"),
        ))
        bm25_score = self._clamp_float(
            candidate.get("_recall_bm25_score"),
            0.0,
            1.0,
            0.0,
        )
        entity_match_info = self._recall_stage1_matched_entity_names(
            query_entity_names=query_entity_names,
            candidate_entity_names=candidate.get("entities") or [],
        )
        high_value_entity_matched = bool(
            entity_match_info.get("high_value_entity_matched")
        )
        entity_strong_anchor = bool(
            entity_match_info.get("entity_strong_anchor")
        )
        time_score_info = self._calculate_recall_candidate_time_score(
            candidate,
            temporal_bounds=temporal_bounds,
        )
        time_weight = self._recall_stage2_time_score_weight
        if is_contextual_query:
            time_weight *= self._recall_stage2_contextual_time_score_multiplier
        time_component_score = time_weight * float(
            time_score_info.get("time_score") or 0.0
        )

        strong_embedding_threshold = (
            self._recall_stage2_fact_strong_embedding_similarity
        )
        embedding_strong_anchor = bool(
            query_embedding is not None
            and candidate.get("embedding") is not None
            and similarity >= strong_embedding_threshold
        )
        best_topic_pair_score = float(
            query_topic_match_info.get("best_pair_score") or 0.0
        )
        term_strong_anchor = bool(
            int(query_topic_match_info.get("matched_term_count") or 0) > 0
            and best_topic_pair_score
            >= self._recall_stage2_strong_topic_pair_score
        )
        keyword_strong_anchor = bool(
            int(keyword_match_info.get("matched_term_count") or 0) > 0
            and float(keyword_match_info.get("best_pair_score") or 0.0)
            >= self._recall_stage2_strong_topic_pair_score
        )
        strong_anchor_reasons: List[str] = []
        if embedding_strong_anchor:
            strong_anchor_reasons.append("embedding_similarity")
        if term_strong_anchor:
            strong_anchor_reasons.append("topic_match")
        if keyword_strong_anchor:
            strong_anchor_reasons.append("keyword_match")
        if entity_strong_anchor:
            strong_anchor_reasons.append("entity_match")
        has_strong_anchor = bool(strong_anchor_reasons)

        embedding_score = (
            self._recall_stage2_embedding_score_weight * similarity
            if query_embedding is not None and candidate.get("embedding") is not None
            else 0.0
        )
        topic_match_score = (
            self._recall_stage2_topic_overlap_score_weight * topic_match_ratio
            if search_terms and all_topic_values
            else 0.0
        )
        keyword_match_score = (
            self._recall_stage2_keyword_match_score_weight * keyword_match_ratio
            if search_terms and all_keyword_values
            else 0.0
        )
        bm25_component_score = (
            self._recall_stage2_bm25_score_weight * bm25_score
        )
        entity_matching_score = (
            self._recall_stage2_entity_matched_score
            if high_value_entity_matched
            else 0.0
        )
        score = self._clamp_float(
            embedding_score
            + topic_match_score
            + keyword_match_score
            + bm25_component_score
            + entity_matching_score
            + time_component_score,
            0.0,
            1.0,
            0.0,
        )
        score_components = {
            "embedding_score": round(float(embedding_score), 4),
            "topic_match_score": round(float(topic_match_score), 4),
            "keyword_match_score": round(float(keyword_match_score), 4),
            "bm25_component_score": round(float(bm25_component_score), 4),
            "entity_matching_score": round(float(entity_matching_score), 4),
            "time_component_score": round(float(time_component_score), 4),
        }
        stage2_match_details = {
            "topic_match_info": {
                **search_topic_match_info,
                "query_anchor_match_info": dict(query_topic_match_info),
                "keyword_match_info": dict(keyword_match_info),
            },
            "entity_match_info": dict(entity_match_info),
            "time_score_info": dict(time_score_info),
            "score_components": dict(score_components),
        }
        return {
            "score": round(float(score), 4),
            "matched": has_strong_anchor,
            "filter_reason": "" if has_strong_anchor else "no_strong_anchor",
            "embedding_similarity": round(float(similarity), 4),
            "contextual_time_weight_applied": bool(is_contextual_query),
            "has_strong_anchor": has_strong_anchor,
            "strong_anchor_reasons": strong_anchor_reasons,
            "_recall_stage2_match_details": stage2_match_details,
        }

    @classmethod
    def _normalize_llm_recall_time_bound(cls, value: Any) -> Optional[str]:
        """Validate one LLM-supplied recall bound and normalize it to ISO."""
        normalized = cls._normalize_recall_time_bound(value, default_to_now=False)
        if not normalized:
            return None
        try:
            return datetime.strptime(normalized, "%Y-%m-%d %H:%M:%S").strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        except ValueError:
            return None

    @staticmethod
    def _recall_query_has_explicit_calendar_date(query: str) -> bool:
        """Return whether the query states a concrete calendar date/range."""
        text = str(query or "")
        return bool(re.search(
            r"\d{4}\s*(?:年|[-/.])\s*\d{1,2}"
            r"|\d{1,2}\s*月\s*\d{1,2}\s*(?:日|号)?",
            text,
        ))

    def _resolve_recall_stage2_temporal_constraints(
        self,
        *,
        original_query: str,
        fallback_temporal_bounds: RecallTimeBounds,
        fallback_temporal_mode: str,
        llm_temporal_bounds: Any,
        llm_temporal_mode: Any,
    ) -> Dict[str, Any]:
        """Validate Stage 2 temporal analysis and choose effective constraints."""
        fallback_start, fallback_end = fallback_temporal_bounds or (None, None)
        fallback_bounds = (
            self._normalize_llm_recall_time_bound(fallback_start),
            self._normalize_llm_recall_time_bound(fallback_end),
        )
        raw_llm_bounds = (
            llm_temporal_bounds if isinstance(llm_temporal_bounds, dict) else {}
        )
        llm_bounds = (
            self._normalize_llm_recall_time_bound(raw_llm_bounds.get("start")),
            self._normalize_llm_recall_time_bound(raw_llm_bounds.get("end")),
        )
        llm_bounds_valid = bool(llm_bounds[0] or llm_bounds[1])
        if llm_bounds_valid and all(llm_bounds):
            llm_bounds_valid = bool(llm_bounds[0] < llm_bounds[1])

        fallback_has_bounds = bool(fallback_bounds[0] or fallback_bounds[1])
        explicit_rule_bounds = bool(
            fallback_has_bounds
            and self._recall_query_has_explicit_calendar_date(original_query)
        )
        if llm_bounds_valid and not explicit_rule_bounds:
            effective_bounds = llm_bounds
            bounds_source = "llm"
        elif fallback_has_bounds:
            effective_bounds = fallback_bounds
            bounds_source = (
                "rule_conflict_override"
                if llm_bounds_valid and explicit_rule_bounds
                else "rule"
            )
        else:
            effective_bounds = (None, None)
            bounds_source = "none"

        llm_mode = self._normalize_recall_temporal_mode(llm_temporal_mode)
        fallback_mode = self._normalize_recall_temporal_mode(fallback_temporal_mode)
        effective_has_bounds = bool(effective_bounds[0] or effective_bounds[1])
        if effective_has_bounds:
            if llm_mode in {"event_time", "dialogue_time", "both"}:
                effective_mode = llm_mode
                mode_source = "llm"
            elif fallback_mode in {"event_time", "dialogue_time", "both"}:
                effective_mode = fallback_mode
                mode_source = "rule"
            else:
                effective_mode = "both"
                mode_source = "bounds_default"
        else:
            effective_mode = llm_mode if llm_mode != "none" else fallback_mode
            mode_source = "llm" if llm_mode != "none" else "rule"

        return {
            "fallback_temporal_bounds": fallback_bounds,
            "fallback_temporal_mode": fallback_mode,
            "llm_temporal_bounds": llm_bounds,
            "llm_temporal_bounds_valid": llm_bounds_valid,
            "llm_temporal_mode": llm_mode,
            "effective_temporal_bounds": effective_bounds,
            "effective_temporal_mode": effective_mode,
            "temporal_bounds_source": bounds_source,
            "temporal_mode_source": mode_source,
        }

    def _analyze_recall_query(
        self,
        query: str,
        *,
        reference_time: str,
        prompt_language: str,
    ) -> Dict[str, Any]:
        prompt_language = (
            "en"
            if str(prompt_language or "").strip().lower().startswith("en")
            else "zh"
        )
        prompt_template = (
            RECALL_QUERY_ANALYSIS_PROMPT_EN
            if prompt_language == "en"
            else RECALL_QUERY_ANALYSIS_PROMPT_ZH
        )
        prompt = (
            prompt_template
            .replace("{query}", str(query or ""))
            .replace("{reference_time}", str(reference_time or ""))
        )
        result = self._call_llm(prompt)
        parsed = self._parse_json_object_from_llm_text(result or "")
        return parsed if isinstance(parsed, dict) else {}

    def _build_recall_search_terms(
        self,
        query: str,
        *,
        keywords: Sequence[str],
        entities: Sequence[str],
    ) -> List[str]:
        """Build the shared, already-tokenized lexical query representation."""
        terms: List[str] = []
        seen: set[str] = set()

        def add_terms(values: Sequence[str]) -> None:
            for value in values:
                clean = re.sub(r"\s+", " ", str(value or "").strip()).lower()
                if not clean or clean in seen:
                    continue
                if len(clean) > 80 or re.search(r"[。！？!?；;，,]", clean):
                    continue
                seen.add(clean)
                terms.append(clean)
                if len(terms) >= 32:
                    return

        for value in [*keywords, *entities]:
            add_terms(self._lexical_search_terms_for_text(value))
            if len(terms) >= 32:
                break
        if len(terms) < 32:
            add_terms(
                self._lexical_search_terms_for_text(
                    query,
                    limit=32,
                    preserve_phrase=False,
                )
            )
        return terms

    @staticmethod
    def _recall_entity_lookup_aliases(query: Any) -> List[str]:
        """Return canonical role entities implied by first/second-person text.

        Memory entities use stable role names (``用户``/``助手`` or
        ``user``/``assistant``), while recall questions naturally use
        pronouns such as ``我`` and ``你``. These aliases are only used to
        query the entity-node index; the original query remains unchanged for
        time parsing, lexical search, scoring, and LLM analysis.
        """
        text = str(query or "")
        aliases: List[str] = []

        # Do not treat ``我们``/``你们`` as a single speaker role. The
        # conversational schema models the direct user and assistant roles
        # separately.
        is_chinese = bool(re.search(r"[\u4e00-\u9fff]", text))
        if re.search(r"我(?!们)", text) or re.search(
            r"\b(?:i|me|my|mine|myself)\b", text, re.IGNORECASE
        ):
            aliases.append("用户" if is_chinese else "user")
        if re.search(r"你(?!们)", text) or re.search(
            r"\b(?:you|your|yours|yourself)\b", text, re.IGNORECASE
        ):
            aliases.append("助手" if is_chinese else "assistant")
        return list(dict.fromkeys(aliases))

    def _lexical_search_terms_for_text(
        self,
        text: Any,
        *,
        limit: int = 32,
        preserve_phrase: bool = True,
    ) -> List[str]:
        """Tokenize one lexical value for the database FTS contract.

        The regular jieba tokens and search-mode sub-tokens mirror the stream
        used when ``memory_database`` builds ``lexical_index_text``. A
        whitespace-joined regular-token phrase is retained before individual
        tokens so short topic phrases remain searchable as a unit.
        """
        clean_text = _compact_whitespace(text)
        if not clean_text:
            return []
        values: List[str] = []
        seen: set[str] = set()

        def add(value: Any) -> None:
            if len(values) >= max(1, int(limit or 32)):
                return
            clean = re.sub(r"\s+", " ", str(value or "").strip()).lower()
            if not clean or clean in seen:
                return
            if len(clean) > 80 or re.search(r"[。！？!?；;，,]", clean):
                return
            chinese_count = len(re.findall(r"[\u4e00-\u9fff]", clean))
            if chinese_count and chinese_count < 2 and len(clean) < 2:
                return
            if not chinese_count and len(clean) < 2:
                return
            seen.add(clean)
            values.append(clean)

        chinese_text = "".join(re.findall(r"[\u4e00-\u9fff]", clean_text))
        if jieba is not None and chinese_text:
            regular_tokens = [
                _compact_whitespace(token)
                for token in jieba.lcut(clean_text, HMM=False)
            ]
            regular_tokens = [
                token for token in regular_tokens
                if token and re.search(r"[0-9a-zA-Z\u4e00-\u9fff]", token)
            ]
            if preserve_phrase and len(regular_tokens) > 1:
                add(" ".join(regular_tokens))
            for token in regular_tokens:
                add(token)
            for token in jieba.cut_for_search(clean_text, HMM=False):
                add(token)
            return values

        if preserve_phrase and not chinese_text:
            add(clean_text)

        # Minimal-install fallback: keep the complete Chinese run and the
        # same bigram coverage used by the legacy lexical path.
        for token in re.findall(
            r"[A-Za-z][A-Za-z0-9_.$'-]*|\d+(?:/\d+)?|[\u4e00-\u9fff]+",
            clean_text,
        ):
            if preserve_phrase or not re.fullmatch(r"[\u4e00-\u9fff]+", token):
                add(token)
            if re.fullmatch(r"[\u4e00-\u9fff]+", token):
                for index in range(len(token) - 1):
                    add(token[index : index + 2])
        return values

    def _recall_context_char_budget(self, budget: str) -> int:
        return int(
            self._recall_context_char_budgets.get(
                str(budget or "mid").lower(),
                self._recall_context_char_budgets["mid"],
            )
        )

    def _recall_entry_char_budget(self, budget: str) -> int:
        return int(
            self._recall_entry_char_budgets.get(
                str(budget or "mid").lower(),
                self._recall_entry_char_budgets["mid"],
            )
        )

    @staticmethod
    def _truncate_recall_line(text: Any, *, max_chars: int) -> str:
        clean = _compact_whitespace(text or "")
        if len(clean) <= max_chars:
            return clean
        return clean[: max(0, max_chars - 18)].rstrip() + "...[truncated]"

    @staticmethod
    def _normalize_event_time_text(value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        text = text.split("#", 1)[0].strip().replace("T", " ")
        if len(text) >= 19:
            return text[:19]
        if len(text) >= 10:
            return text[:10]
        return text

    @staticmethod
    def _normalize_recall_temporal_mode(value: Any) -> str:
        mode = str(value or "").strip().lower()
        aliases = {
            "event": "event_time",
            "event-time": "event_time",
            "dialogue": "dialogue_time",
            "dialogue-time": "dialogue_time",
            "conversation": "dialogue_time",
            "all": "both",
        }
        mode = aliases.get(mode, mode)
        return mode if mode in {"event_time", "dialogue_time", "both", "none"} else "none"

    @classmethod
    def _infer_recall_temporal_mode(cls, query: str) -> str:
        text = _compact_whitespace(query).lower()
        dialogue_markers = (
            "讨论", "聊", "提到", "问过", "说过", "谈过", "对话", "交流",
            "讨论了", "提及", "discuss", "talk", "mentioned", "asked",
            "conversation", "chat", "said",
        )
        event_markers = (
            "做了", "做过", "发生", "买了", "买过", "去过", "参加", "完成",
            "经历", "遇到", "使用过", "发生了什么", "what happened", "did",
            "bought", "visited", "attended", "completed", "experienced",
        )
        has_dialogue = any(marker in text for marker in dialogue_markers)
        has_event = any(marker in text for marker in event_markers)
        if has_dialogue and has_event:
            return "both"
        if has_event:
            return "event_time"
        if has_dialogue:
            return "dialogue_time"
        return "none"

    @classmethod
    def _fact_time_values(
        cls,
        fact: Dict[str, Any],
        temporal_mode: str,
    ) -> List[str]:
        mode = cls._normalize_recall_temporal_mode(temporal_mode)
        event_time = cls._normalize_event_time_text(fact.get("event_time_key"))
        dialogue_time = cls._normalize_event_time_text(fact.get("dialogue_time_key"))
        if mode == "event_time":
            return [event_time] if event_time else []
        if mode == "dialogue_time":
            return [dialogue_time] if dialogue_time else []
        if mode == "both":
            return sorted({value for value in (event_time, dialogue_time) if value})
        return []

    @classmethod
    def _fact_matches_time_bounds(
        cls,
        fact: Dict[str, Any],
        *,
        temporal_mode: str,
        temporal_bounds: RecallTimeBounds,
    ) -> bool:
        time_start, time_end = temporal_bounds or (None, None)
        mode = cls._normalize_recall_temporal_mode(temporal_mode)
        if mode == "none" or not (time_start or time_end):
            return True
        values = cls._fact_time_values(fact, mode)
        if not values:
            return False
        return any(
            (not time_start or value >= str(time_start))
            and (not time_end or value <= str(time_end))
            for value in values
        )

    def _build_memory_retrieved_format_text(
        self,
        *,
        entries: List[Dict[str, Any]],
        prompt_language: str,
    ) -> str:
        """Format selected direct/associated candidates for the assistant."""
        if not entries:
            return ""

        is_en = str(prompt_language or "").strip().lower().startswith("en")
        format_template = (
            MEMORY_RETRIEVED_FORMAT_PROMPT_EN
            if is_en
            else MEMORY_RETRIEVED_FORMAT_PROMPT_ZH
        )
        section_specs = (
            MEMORY_RETRIEVED_SECTION_SPECS_EN
            if is_en
            else MEMORY_RETRIEVED_SECTION_SPECS_ZH
        )
        note_prefix = "System note: " if is_en else "系统说明："
        if is_en:
            labels = {
                "fact": "narrative fact",
                "dialogue_time": "dialogue_time",
                "event_time": "event_time",
                "summary": "summary",
                "fact_root_topic": "fact_root_topic",
                "fact_aspect_topic": "fact_aspect_topic",
                "claim": "claim",
                "claim_origin": "claim_origin",
                "status": "status",
                "time": "time",
                "action": "action",
            }
        else:
            labels = {
                "fact": "叙事事实",
                "dialogue_time": "对话时间",
                "event_time": "事件时间",
                "summary": "摘要",
                "fact_root_topic": "事实根主题",
                "fact_aspect_topic": "事实方面主题",
                "claim": "主张",
                "claim_origin": "来源",
                "status": "状态",
                "time": "时间",
                "action": "行动",
            }

        grouped = {
            group_key: [
                entry for entry in entries
                if entry.get("index_level") == group_key
            ]
            for _title, _note, group_key in section_specs
        }
        sections: List[str] = []
        for title, note, group_key in section_specs:
            group = grouped[group_key]
            if not group:
                continue
            section_lines = [title, f"{note_prefix}{note}"]
            for index, entry in enumerate(group, 1):
                raw = entry.get("_hydrated") if isinstance(entry.get("_hydrated"), dict) else {}
                if group_key == "fact":
                    dialogue_time = (
                        self._normalize_event_time_text(raw.get("dialogue_time_key"))
                        or "unknown-dialogue-time"
                    )
                    event_time = (
                        self._normalize_event_time_text(raw.get("event_time_key"))
                        or "unknown-event-time"
                    )
                    block_lines = [
                        f"{index}. {labels['fact']}",
                        f"   {labels['dialogue_time']}: {dialogue_time}",
                        f"   {labels['event_time']}: {event_time}",
                        f"   {labels['summary']}: {raw.get('summary') or entry.get('summary_for_retrieval') or ''}",
                        f"   {labels['fact_root_topic']}: {raw.get('fact_root_topic') or ''}; {labels['fact_aspect_topic']}: {raw.get('fact_aspect_topic') or ''}",
                    ]
                elif group_key == "entity_claim":
                    block_lines = [
                        f"{index}. {labels['claim']}",
                        f"   {labels['summary']}: {raw.get('summary') or entry.get('summary_for_retrieval') or ''}",
                        f"   predicate: {raw.get('metadata', {}).get('predicate') or ''}",
                        f"   {labels['claim_origin']}: {raw.get('metadata', {}).get('claim_origin') or ''}",
                        f"   {labels['status']}: {raw.get('status') or ''}",
                    ]
                else:
                    time_value = raw.get("time_end") or raw.get("time_start") or ""
                    action_value = (
                        raw.get("summary")
                        or entry.get("summary_for_retrieval")
                        or ""
                    )
                    block_lines = [
                        f"{index}. {group_key}",
                        f"   {labels['summary']}: {action_value}",
                        f"   {labels['status']}: {raw.get('status') or ''}",
                        f"   {labels['time']}: {time_value or 'unknown'}",
                    ]
                section_lines.append("\n".join(block_lines))
            sections.append("\n".join(section_lines))
        return format_template.replace(
            "{memory_sections}",
            "\n\n".join(sections),
        ).strip()
    
    # ── Lightweight NLP heuristics ───────────────────────────────────────

    def _generate_embedding_vector(self, text: str) -> Optional[np.ndarray]:
        self._ensure_embedding_client()
        return self._embedding_client.embed_text(text) if self._embedding_client else None

    def _format_recall_query_identity_text(
        self,
        query: str,
        *,
        retrieval_text: str = "",
        keywords: Optional[Sequence[str]] = None,
        entities: Optional[Sequence[str]] = None,
    ) -> str:
        terms = list(keywords or [])
        if not terms:
            terms = self._lexical_search_terms_for_text(
                query,
                limit=32,
                preserve_phrase=False,
            )
        parts = [str(query or "").strip()]
        if retrieval_text and str(retrieval_text).strip() != str(query or "").strip():
            parts.append(f"retrieval: {str(retrieval_text).strip()}")
        if terms:
            parts.append(f"keywords: {' '.join(str(item) for item in terms)}")
        if entities:
            parts.append(f"entities: {' '.join(str(item) for item in entities)}")
        return "\n".join(parts)

    def _keywords(self, text: str, *, limit: int) -> List[str]:
        tokens = re.findall(r"[A-Za-z][A-Za-z0-9_.$'-]*|\d+(?:/\d+)?|[\u4e00-\u9fff]{2,}", str(text or "").lower())
        counts = Counter(
            self._normalize_keyword_term(token)
            for token in tokens
            if self._is_valid_keyword_term(self._normalize_keyword_term(token))
        )
        return [term for term, _count in counts.most_common(limit)]

    def _entities(self, text: str) -> List[str]:
        entities: List[str] = []
        for match in re.findall(r"\b[A-Z][A-Za-z0-9'&.-]*(?:\s+[A-Z][A-Za-z0-9'&.-]*){0,4}\b", str(text or "")):
            clean = _compact_whitespace(match)
            if self._is_valid_entity_name(clean) and clean not in entities:
                entities.append(clean)
            if len(entities) >= 12:
                break
        return entities

    def _topic_candidates(self, text: str) -> List[str]:
        keywords = self._keywords(text, limit=6)
        if not keywords:
            return []
        topics: List[str] = []
        for size in (3, 2):
            if len(keywords) >= size:
                topics.append(" ".join(keywords[:size]))
        topics.append(keywords[0])
        return list(dict.fromkeys(topics))[:3]
    
    def _infer_fact_type(self, text: str, *, speaker: str) -> str:
        lower = str(text or "").lower()
        if any(word in lower for word in ("prefer", "favorite", "like", "dislike", "would rather")):
            return "preference"
        if any(word in lower for word in ("decided", "i'll", "i will", "plan to", "going to")):
            return "decision" if speaker == "user" else "recommendation"
        if any(word in lower for word in ("need to", "have to", "should", "todo", "pick up", "return")):
            return "action"
        if any(word in lower for word in ("recommend", "suggest", "consider", "try")):
            return "recommendation"
        if "?" in text:
            return "request"
        return "context"

    @staticmethod
    def _normalize_priority(value: Any) -> int:
        try:
            return max(0, min(100, int(round(float(value)))))
        except (TypeError, ValueError):
            return 70

    @staticmethod
    def _normalize_fact_type(value: Any) -> str:
        text = str(value or "context").strip().lower()
        allowed = {
            "preference", "decision", "request", "recommendation", "action",
            "commitment", "open_question", "risk", "error", "context",
            "instruction", "other",
        }
        return text if text in allowed else "context"

    @staticmethod
    def _normalize_string_list(value: Any, *, limit: int = 12) -> List[str]:
        if isinstance(value, str):
            raw = re.split(r"[,，;；\n]+", value)
        elif isinstance(value, list):
            raw = value
        else:
            raw = []
        out: List[str] = []
        seen = set()
        for item in raw:
            text = MemoryNodeManager._normalize_keyword_term(item)
            if not MemoryNodeManager._is_valid_keyword_term(text) or text in seen:
                continue
            seen.add(text)
            out.append(text)
            if len(out) >= limit:
                break
        return out

    @staticmethod
    def _normalize_keyword_term(value: Any) -> str:
        text = _compact_whitespace(value)
        return text.strip("'\".,:;!?，。！？、；：（）()[]{}")

    @staticmethod
    def _is_valid_keyword_term(text: str) -> bool:
        clean = _compact_whitespace(text)
        if not clean:
            return False
        lower = clean.lower()
        if lower in _STOPWORDS:
            return False
        if any(pattern in lower for pattern in _COURTESY_PATTERNS):
            return False
        if re.search(r"[。！？!?；;，,]", clean):
            return False
        if re.search(r"(好的|谢谢|不客气|继续沟通|有其他问题|帮到您|帮到你)", clean):
            return False
        if re.search(r"^(好的|谢谢|嗯|行|可以|ok|okay|thanks)$", lower):
            return False
        chinese_chars = re.findall(r"[\u4e00-\u9fff]", clean)
        if chinese_chars and len(chinese_chars) > 10:
            return False
        if not chinese_chars and len(clean.split()) > 4:
            return False
        return len(clean) > 1

    @staticmethod
    def _normalize_entity_names(value: Any, *, limit: int = 16) -> List[str]:
        if isinstance(value, str):
            raw = re.split(r"[,，;；\n]+", value)
        elif isinstance(value, list):
            raw = value
        else:
            raw = []
        out: List[str] = []
        seen = set()
        for item in raw:
            if isinstance(item, dict):
                text = _compact_whitespace(item.get("name") or item.get("text") or "")
            else:
                text = _compact_whitespace(item)
            if not MemoryNodeManager._is_valid_entity_name(text) or text in seen:
                continue
            seen.add(text)
            out.append(text)
            if len(out) >= limit:
                break
        return out

    @staticmethod
    def _is_valid_entity_name(value: Any) -> bool:
        """Validate entity anchors using the shared extraction guidance."""
        text = _compact_whitespace(value).strip("'\".,:;!?，。！？、；：（）()[]{}")
        if not text:
            return False
        lower = text.lower()
        if lower in _STOPWORDS:
            return False
        if any(pattern in lower for pattern in _COURTESY_PATTERNS):
            return False
        if re.search(r"[。！？!?；;，,]", text):
            return False
        if len(text) > 48:
            return False
        if any(
            re.fullmatch(pattern, text, flags=re.IGNORECASE)
            for pattern in _ORDINARY_TIME_ENTITY_PATTERNS
        ):
            return False
        if any(
            re.fullmatch(pattern, text, flags=re.IGNORECASE)
            for pattern in _ATTRIBUTE_ONLY_ENTITY_PATTERNS
        ):
            if not re.search(
                r"(工作|压力|负担|时间|作息|活动|场景|问题|任务|状态|沟通|管理|"
                r"fatigue|burden|pressure|schedule|activity|scenario|task|condition|communication|management)",
                lower,
            ):
                return False
        chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
        if chinese_chars and len(chinese_chars) > 16:
            return False
        if not chinese_chars and len(text.split()) > 5:
            return False
        return len(text) > 1

    @classmethod
    def _normalize_primary_entity(
        cls,
        value: Any,
        *,
        entities: Sequence[str],
    ) -> Optional[Dict[str, str]]:
        """Normalize the single entity used for entity-state assignment."""
        name = ""
        entity_type = "CONCEPT"
        if isinstance(value, dict):
            name = _compact_whitespace(value.get("name") or value.get("text") or "")
            entity_type = _compact_whitespace(value.get("type") or "CONCEPT").upper()
        else:
            name = _compact_whitespace(value)
        if not name:
            if entities:
                name = _compact_whitespace(entities[0])
        if not name:
            return None
        allowed_types = {
            "PERSON", "ORGANIZATION", "LOCATION", "PRODUCT", "PROJECT",
            "TECHNOLOGY", "CONCEPT", "TOPIC", "PREFERENCE", "OTHER",
        }
        if entity_type not in allowed_types:
            entity_type = "CONCEPT"
        return {"name": name, "type": entity_type}

    def _recall_stage2_candidate_limits(
        self,
        *,
        top_k: int,
    ) -> Dict[str, Any]:
        """Build explicit Stage 2 retrieval, expansion, and output budgets.

        Stage 2 broadens retrieval through three independent seed channels.
        All direct object types receive bounded seed quotas, while final
        selection remains fact-only until type-specific Stage 2 matching and
        parent-aware selection are introduced.
        """
        k = max(1, int(top_k or 1))
        # Stage 2's LLM-expanded lexical and full-embedding channels need a
        # little more depth than the final fact quota to surface semantic
        # alternatives before direct scoring.
        seed_per_channel_limit = max(6, min(16, int(math.ceil(k * 1.5))))
        seed_limits = {
            "fact": seed_per_channel_limit,
            "entity_claim": max(4, min(10, k)),
            "goal": max(4, min(8, k)),
            "plan": max(4, min(10, k)),
            "work_item": max(4, min(10, k)),
        }
        return {
            # No merged-seed cap is needed: each independent channel is
            # already bounded, and deduplication can only reduce the pool.
            # Stage 1 lexical seeds retain their own Stage 1 retrieval limit.
            "seed_channel_limits": {
                "stage2_lexical": dict(seed_limits),
                "stage2_embedding": dict(seed_limits),
            },
            # Applied to same-episode fact expansion.
            "association_per_relation_limit": max(4, min(12, k)),
            "selected_limits": {
                "fact": k,
            },
        }
