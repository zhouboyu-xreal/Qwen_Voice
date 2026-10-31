#!/usr/bin/env python3
"""Unified memory manager inspired by MemPalace.

The public surface mirrors the current project's `MemoryNodeManager`, but the
internal model is deliberately unified:

1. Submitted source units become narrative `memory_facts`.
2. Longer source windows and their facts yield grounded user-activity observations.
3. Explicit and derived propositions live in user claims; repeated action patterns live separately.
4. The legacy actionable-item projection is temporarily disabled while the
   intent and work-item layer is redesigned.

"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import sqlite3
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import jieba
except ImportError:  # pragma: no cover - exercised only in minimal installs
    jieba = None

from embedding_client import EmbeddingClient
from llm_client import LLMClient
from .memory_database import UserMemoryDB
from .memory_types import MemoryUnit
from .user_thread_updater import UserThreadUpdater
from task_scheduler import TaskScheduler
from prompts_en import (
    DERIVED_USER_CLAIM_EXTRACTION_PROMPT_EN,
    USER_OBSERVED_PATTERN_RECONCILIATION_PROMPT_EN,
    USER_OBSERVED_PATTERN_SIGNAL_RELATION_PROMPT_EN,
    USER_CLAIM_RECONCILIATION_PROMPT_EN,
    EXPLICIT_USER_CLAIM_EXTRACTION_PROMPT_EN,
    USER_OBSERVED_PATTERN_EXTRACTION_PROMPT_EN,
    INTENT_EXTRACTION_PROMPT_EN,
    INTENT_LIFECYCLE_RECONCILIATION_PROMPT_EN,
    INTENT_RECONCILIATION_PROMPT_EN,
    USER_ACTIVITY_EXTRACTION_PROMPT_EN,
    MEMORY_RETRIEVED_FORMAT_PROMPT_EN,
    MEMORY_RETRIEVED_SECTION_SPECS_EN,
    RECALL_QUERY_ANALYSIS_PROMPT_EN,
    UNIFIED_MEMORY_EXTRACTION_PROMPT_EN,
)
from prompts_zh import (
    DERIVED_USER_CLAIM_EXTRACTION_PROMPT_ZH,
    USER_OBSERVED_PATTERN_RECONCILIATION_PROMPT_ZH,
    USER_OBSERVED_PATTERN_SIGNAL_RELATION_PROMPT_ZH,
    USER_CLAIM_RECONCILIATION_PROMPT_ZH,
    EXPLICIT_USER_CLAIM_EXTRACTION_PROMPT_ZH,
    USER_OBSERVED_PATTERN_EXTRACTION_PROMPT_ZH,
    INTENT_EXTRACTION_PROMPT_ZH,
    INTENT_LIFECYCLE_RECONCILIATION_PROMPT_ZH,
    INTENT_RECONCILIATION_PROMPT_ZH,
    USER_ACTIVITY_EXTRACTION_PROMPT_ZH,
    MEMORY_RETRIEVED_FORMAT_PROMPT_ZH,
    MEMORY_RETRIEVED_SECTION_SPECS_ZH,
    RECALL_QUERY_ANALYSIS_PROMPT_ZH,
    UNIFIED_MEMORY_EXTRACTION_PROMPT_ZH,
)
from .utils import _cal_embedding_cosine_similarity
from .user_action_similarity import action_lexical_similarity
from user_model import UserStateResolver

RecallTimeBounds = Optional[Tuple[Optional[str], Optional[str]]]

_PRIMARY_DOMAINS = UserStateResolver.STATE_DOMAINS

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
        "用户", "user", "the user", "用户本人", "user self", "user_self",
        "我", "我自己", "me", "myself",
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

# ``用户本人`` / ``user self`` identify the wearer whose personal world model
# this database represents. They are intentionally distinct from ordinary
# mentions of product users, customers, or consumers in ambient recordings.
_WORLD_OWNER_ENTITY_REFERENCE_ALIASES = {
    "用户本人", "user self", "user_self",
}
_AMBIGUOUS_USER_ENTITY_ALIASES = {
    "用户", "user", "the user",
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




class MemoryNodeManager:
    """Compatibility manager backed by a unified index-first memory line."""

    def __init__(
        self,
        db: UserMemoryDB,
        *,
        embedding_config: Optional[Dict[str, Any]] = None,
        memory_manager_config: Optional[Dict[str, Any]] = None,
        task_scheduler: TaskScheduler,
        llm_client: LLMClient,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._db = db
        self._logger = logger or logging.getLogger(__name__)
        self._operation_reporter = task_scheduler.operation_reporter
        self._task_scheduler = task_scheduler
        self._llm_client = llm_client
        self._memory_cfg = dict(memory_manager_config or {})
        configured_embedding = self._memory_cfg.get("embedding")
        self._embedding_cfg = dict(
            embedding_config
            or (configured_embedding if isinstance(configured_embedding, dict) else {})
        )
        self._memory_prompt_language = str(
            self._memory_cfg.get("memory_prompt_language_mode")
            or "source"
        )
        self._memory_enabled = bool(self._memory_cfg.get("memory_enabled", True))
        configured_world_owner_name = _compact_whitespace(
            self._memory_cfg.get("world_owner_entity_name") or ""
        )
        if configured_world_owner_name.lower() in _AMBIGUOUS_USER_ENTITY_ALIASES:
            raise ValueError(
                "memory.world_owner_entity_name must use '用户本人', "
                "'user self', or a concrete owner name; bare '用户' / 'user' "
                "is ambiguous with product users and consumers."
            )
        self._world_owner_entity_name = (
            configured_world_owner_name or "用户本人"
        )
        self._user_claim_explicit_min_confidence = self._clamp_float(
            self._memory_cfg.get("user_claim_explicit_min_confidence"),
            0.0,
            1.0,
            0.72,
        )
        self._user_claim_derived_min_confidence = self._clamp_float(
            self._memory_cfg.get("user_claim_derived_min_confidence"),
            0.0,
            1.0,
            0.75,
        )
        self._user_observed_pattern_min_support_signals = max(
            3,
            int(
                self._memory_cfg.get(
                    "user_observed_pattern_min_support_signals", 3,
                ) or 3
            ),
        )
        self._user_observed_pattern_signal_limit = max(
            self._user_observed_pattern_min_support_signals,
            int(self._memory_cfg.get("user_observed_pattern_signal_limit", 32) or 32),
        )
        self._initialize_recall_config()
        self._embedding_client: Optional[EmbeddingClient] = None
        self._user_thread_updater = UserThreadUpdater(
            memory_db=self._db, llm_client=self._llm_client,
            resolve_language=self._resolve_prompt_language_from_text, log=self._log_info,
            batch_size=self._memory_cfg.get("user_thread_activity_batch_size", 8),
            candidate_limit=self._memory_cfg.get("user_thread_candidate_limit", 12),
            event_context_limit=self._memory_cfg.get("user_thread_event_context_limit", 16),
        )
        self._task_scheduler.register_handler(
            "memory_fact_extraction", self._process_memory_fact_extraction_task,
        )
        self._task_scheduler.register_handler(
            "memory_user_activity_extraction", self._process_memory_user_activity_extraction_task,
        )
        self._task_scheduler.register_handler(
            "memory_user_thread_update", self._process_memory_user_thread_update_task,
        )
        self._task_scheduler.register_handler(
            "memory_user_claim_update", self._process_memory_user_claim_update_task,
        )
        self._task_scheduler.register_handler(
            "memory_user_observed_pattern_update",
            self._process_memory_user_observed_pattern_update_task,
        )
        self._task_scheduler.register_handler(
            "memory_user_intent_update", self._process_memory_user_intent_update_task,
        )
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

    # ── Runtime helpers used by benchmark scripts ───────────────────────

    def _ensure_embedding_client(self) -> bool:
        if self._embedding_client is None:
            self._embedding_client = EmbeddingClient(self._embedding_cfg)
        return True

    # ── Store path: source units -> facts -> user activity ───────────────

    @property
    def enabled(self) -> bool:
        return self._memory_enabled

    @property
    def world_owner_entity_name(self) -> str:
        return self._world_owner_entity_name

    def set_logger(self, logger: Optional[logging.Logger]) -> None:
        """Set the logger used by memory store, entity-claim, and recall operations."""
        self._logger = logger or logging.getLogger(__name__)
        self._llm_client.set_logger(self._logger)

    def set_embedding_client(self, embedding_client: Optional[EmbeddingClient]) -> None:
        """Share a pre-initialized embedding client with memory runtime callers."""
        self._embedding_client = embedding_client

    def _process_memory_user_thread_update_task(
        self, *, activity_ids: List[int], reference_timestamp: Any = None,
    ) -> Dict[str, Any]:
        """Resolve activity membership first, then incrementally update matters."""
        return self._user_thread_updater.process_update_task(
            activity_ids=activity_ids, reference_timestamp=reference_timestamp,
        )

    def _process_memory_user_activity_extraction_task(
        self,
        *,
        tags: List[str],
    ) -> Dict[str, Any]:
        source_units = self._db.get_unprocessed_memory_source_segments_for_user_activity(
            limit=240,
        )
        if not source_units:
            return {
                "status": "empty",
                "reason": "no_unprocessed_source_units",
                "new_user_activity_count": 0,
                "fact_count": 0,
                "source_unit_count": 0,
            }
        source_row_ids = list(dict.fromkeys(
            int(unit.source_unit_row_id)
            for unit in source_units
            if unit.source_unit_row_id is not None
        ))
        facts = self._db.get_memory_facts_for_source_segments(
            source_segment_ids=source_row_ids, limit=240,
        )
        activity_items = self._extract_user_activity_from_source_units(
            source_units=source_units,
            facts=facts,
        )
        if activity_items is None:
            return {"status": "failed", "reason": "invalid_user_activity_response"}
        source_started_at = _compact_whitespace(source_units[0].start_time) or _now_text()
        source_ended_at = _compact_whitespace(source_units[-1].end_time) or source_started_at
        fact_ids = {int(fact["id"]) for fact in facts}
        activity_ids: List[int] = []
        with self._db.transaction():
            for item in activity_items:
                supported_fact_ids = [
                    int(value) for value in item.get("evidence_fact_ids") or []
                    if str(value).isdigit() and int(value) in fact_ids
                ]
                activity_ids.append(self._db.insert_user_activity_item(
                    activity_text=item["activity_text"],
                    user_role=item["user_role"],
                    confidence=item["confidence"],
                    started_at=source_started_at,
                    ended_at=source_ended_at,
                    evidence_fact_ids=supported_fact_ids,
                    source_segment_ids=source_row_ids,
                    metadata={"tags": list(tags or []), "time_scope": "source_window"},
                ))
            processed_rows = self._db.mark_memory_source_segments_processed_for_user_activity(
                source_segment_ids=source_row_ids,
            )
        report = {
            "status": "ok",
            "new_user_activity_count": len(activity_ids),
            "user_activity_item_ids": activity_ids,
            "fact_count": len(facts),
            "source_unit_count": len(source_units),
            "source_unit_row_count": processed_rows,
        }
        self._log_info("memory_user_activity_extraction", "finish", report)
        return report

    def _process_memory_fact_extraction_task(
        self,
        *,
        units: List[MemoryUnit],
        preceding_context_units: Optional[List[MemoryUnit]] = None,
        tags: List[str],
    ) -> Dict[str, Any]:
        store_started_at = time.monotonic()
        units = self._valid_memory_units(units)
        preceding_units = list(preceding_context_units or [])
        if not units:
            elapsed_ms = round((time.monotonic() - store_started_at) * 1000, 2)
            report = {
                "status": "empty",
                "reason": "no_units",
                "attempt_count": 0,
                "new_fact_count": 0,
                "fact_ids": [],
                "source_unit_row_ids": [],
                "source_unit_count": 0,
                "source_unit_row_count": 0,
                "total_elapsed_ms": elapsed_ms,
            }
            self._log_info("memory_fact_extraction", "finish", report)
            return report
        with self._db.transaction():
            source_unit_row_ids = self._db.insert_memory_source_segments(
                units=units,
            )
        fact_info = self._extract_memory_fact_from_units(
            units,
            preceding_context_units=preceding_units,
        )
        if fact_info["status"] != "ok":
            report = {
                "status": fact_info["status"],
                "reason": fact_info["reason"],
                "attempt_count": fact_info["attempt_count"],
                "new_fact_count": 0,
                "fact_ids": [],
                "source_unit_row_ids": source_unit_row_ids,
                "source_unit_count": len(units),
                "source_unit_row_count": len(source_unit_row_ids),
                "total_elapsed_ms": round((time.monotonic() - store_started_at) * 1000, 2),
            }
            if fact_info.get("error_type"):
                report["error_type"] = fact_info["error_type"]
            self._log_info("memory_fact_extraction", "finish", report)
            return report
        facts = list(fact_info.get("facts") or [])
        self._log_extracted_fact_info(facts=facts)
        with self._db.transaction():
            save_fact_info = self._store_extracted_memory_facts_into_db(
                facts=facts,
                tags=tags,
            )
            self._db.insert_fact_source_mappings(
                fact_ids=save_fact_info.get("fact_ids") or [],
                source_segment_ids=source_unit_row_ids,
            )
            topic_report = self._db.upsert_memory_topic_items(
                save_fact_info.get("topic_item_updates") or []
            )
        report = {
            "status": fact_info["status"],
            "reason": fact_info["reason"],
            "attempt_count": fact_info["attempt_count"],
            "new_fact_count": len(list(save_fact_info.get("fact_ids") or [])),
            "fact_ids": list(save_fact_info.get("fact_ids") or []),
            "user_claim_signal_fact_count": int(
                save_fact_info.get("user_claim_signal_fact_count") or 0
            ),
            "user_claim_signal_count": int(
                save_fact_info.get("user_claim_signal_count") or 0
            ),
            "user_action_signal_fact_count": int(
                save_fact_info.get("user_action_signal_fact_count") or 0
            ),
            "user_action_signal_count": int(
                save_fact_info.get("user_action_signal_count") or 0
            ),
            "user_intent_signal_fact_count": int(
                save_fact_info.get("user_intent_signal_fact_count") or 0
            ),
            "user_intent_signal_count": int(
                save_fact_info.get("user_intent_signal_count") or 0
            ),
            "source_unit_row_ids": source_unit_row_ids,
            "source_unit_count": len(units),
            "source_unit_row_count": len(source_unit_row_ids),
            "topic_items_created": int(topic_report.get("created_count", 0) or 0),
            "topic_items_updated": int(topic_report.get("updated_count", 0) or 0),
            "total_elapsed_ms": round((time.monotonic() - store_started_at) * 1000, 2),
        }
        self._log_info("memory_fact_extraction", "finish", report)
        return report

    @staticmethod
    def _valid_memory_units(
        units: Sequence[MemoryUnit],
    ) -> List[MemoryUnit]:
        """Keep task units with usable text, without rebuilding their order."""
        return [
            unit
            for unit in units or ()
            if isinstance(unit, MemoryUnit)
            and _compact_whitespace(unit.text)
        ]

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
        fact_topic: str,
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
                f"fact_topic: {fact_topic}",
            ],
            entity_ids=entity_ids,
            topic_keys=[fact_topic],
            time_start=event_time_key or dialogue_time_key,
            time_end=event_time_key or dialogue_time_key,
            confidence=confidence,
            importance=importance,
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
                "memory_fact_extraction",
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
                    "fact_topic": fact.get("fact_topic") or "",
                    "user_claim_signals": fact.get("user_claim_signals") or [],
                    "user_action_signals": fact.get("user_action_signals") or [],
                    "user_intent_signals": fact.get("user_intent_signals") or [],
                    "importance": fact.get("importance"),
                    "confidence": fact.get("confidence"),
                    "time_confidence": metadata.get("time_confidence") or "",
                    "where": metadata.get("where") or "",
                    "metadata": metadata,
                    "batch_fact_index": index,
                    "batch_fact_count": len(facts),
                },
            )

    def _extract_memory_fact_from_units(
        self,
        units: Sequence[MemoryUnit],
        *,
        preceding_context_units: Optional[Sequence[MemoryUnit]] = None,
    ) -> Dict[str, Any]:
        """Return ok/empty/failed, facts, a reason and the extraction attempt count."""
        current_units = self._valid_memory_units(units)
        if not current_units:
            return {"status": "empty", "facts": [], "reason": "no_units", "attempt_count": 0}
        preceding_units = self._valid_memory_units(preceding_context_units or [])
        prompt_language = self._resolve_prompt_language_from_memory_units(
            [*current_units, *preceding_units],
        )
        self._log_info("memory_fact_extraction", "start", {
            "source_unit_count": len(current_units),
            "preceding_context_unit_count": len(preceding_units),
            "current_units": self._build_memory_units_for_prompt(
                current_units,
                prompt_language=prompt_language,
            ),
            "preceding_context_units": self._build_memory_units_for_prompt(
                preceding_units,
                prompt_language=prompt_language,
            ),
        })
        prompt_template = (
            UNIFIED_MEMORY_EXTRACTION_PROMPT_EN
            if prompt_language == "en"
            else UNIFIED_MEMORY_EXTRACTION_PROMPT_ZH
        )
        known_world_owner_aliases = [
            item["alias_name"]
            for item in self._db.get_owner_entity_aliases(
                self._intent_world_owner_entity_id(),
                status="confirmed",
                limit=128,
            )
        ]
        prompt = (
            prompt_template
            .replace(
                "{known_world_owner_aliases}",
                json.dumps(known_world_owner_aliases[:32], ensure_ascii=False),
            )
            .replace(
                "{dialogue_batch}",
                self._build_memory_units_for_prompt(
                    current_units,
                    prompt_language=prompt_language,
                ),
            )
            .replace(
                "{preceding_context}",
                self._build_memory_units_for_prompt(
                    preceding_units,
                    prompt_language=prompt_language,
                ) or ("(none)" if prompt_language == "en" else "（无）"),
            )
        )
        failure_info: Dict[str, Any] = {}
        for attempt in range(2):
            try:
                result = self._llm_client.call_llm(prompt)
            except Exception as exc:
                failure_info = {"reason": "llm_request_error", "error_type": type(exc).__name__}
                self._logger.exception("Unified memory LLM request failed attempt=%s", attempt + 1)
                self._log_info("memory_fact_extraction", "llm_output_diagnostics", {
                    "attempt": attempt + 1,
                    "parse_status": "request_error",
                    "error_type": type(exc).__name__,
                    "response_chars": 0,
                    "raw_fact_count": None,
                    "normalized_fact_count": 0,
                    "dropped_fact_counts": {},
                })
                continue
            response_text = str(result or "")
            response_chars = len(response_text)
            if not response_text.strip():
                failure_info = {"reason": "empty_llm_response"}
                self._log_info("memory_fact_extraction", "llm_output_diagnostics", {
                    "attempt": attempt + 1,
                    "parse_status": "empty_response",
                    "response_chars": response_chars,
                    "raw_fact_count": None,
                    "normalized_fact_count": 0,
                    "dropped_fact_counts": {},
                })
                if attempt == 0:
                    self._logger.debug("Unified memory LLM extraction failed, retrying")
                continue
            parsed = self._llm_client.parse_json_object_from_llm_text(result or "")
            if parsed is None:
                failure_info = {"reason": "invalid_llm_json_object"}
                self._log_info("memory_fact_extraction", "llm_output_diagnostics", {
                    "attempt": attempt + 1,
                    "parse_status": "invalid_json_object",
                    "response_chars": response_chars,
                    "raw_fact_count": None,
                    "normalized_fact_count": 0,
                    "dropped_fact_counts": {},
                })
            else:
                normalization_diagnostics: Dict[str, Any] = {}
                normalized = self._normalize_memory_fact_extraction_llm_output(
                    parsed,
                    current_units,
                    prompt_language=prompt_language,
                    known_world_owner_aliases=known_world_owner_aliases,
                    diagnostics=normalization_diagnostics,
                )
                self._log_info("memory_fact_extraction", "llm_output_diagnostics", {
                    "attempt": attempt + 1,
                    "parse_status": "parsed",
                    "response_chars": response_chars,
                    "response_keys": sorted(str(key) for key in parsed.keys()),
                    **normalization_diagnostics,
                })
                if normalized is not None:
                    facts = normalized["facts"]
                    return {
                        "status": "ok" if facts else "empty",
                        "facts": facts,
                        "reason": (
                            "" if facts else "no_retained_facts" if parsed["facts"] else "no_facts"
                        ),
                        "attempt_count": attempt + 1,
                    }
                failure_info = {
                    "reason": normalization_diagnostics.get("normalization_status")
                    or "invalid_llm_fact_response",
                }
            if attempt == 0:
                self._logger.debug("Unified memory LLM extraction failed, retrying")
        return {"status": "failed", "facts": [], "attempt_count": 2, **failure_info}

    def _extract_user_activity_from_source_units(
        self,
        *,
        source_units: Sequence[MemoryUnit],
        facts: Sequence[Dict[str, Any]],
    ) -> Optional[List[Dict[str, Any]]]:
        """Observe what the user is doing across a source window.

        Raw units are authoritative; facts provide compact evidence anchors.
        An empty list is valid when the window does not establish user activity.
        """
        unit_list = self._valid_memory_units(source_units)
        if not unit_list:
            return []
        prompt_language = self._resolve_prompt_language_from_memory_units(
            unit_list,
        )
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
                "fact_topic": item.get("fact_topic") or "",
                "dialogue_time_key": item.get("dialogue_time_key") or "",
                "event_time_key": item.get("event_time_key") or "",
            }
            for item in fact_list
        ]
        source_text = self._build_memory_units_for_prompt(
            unit_list,
            prompt_language=prompt_language,
        )
        prompt_template = (
            USER_ACTIVITY_EXTRACTION_PROMPT_EN
            if prompt_language == "en"
            else USER_ACTIVITY_EXTRACTION_PROMPT_ZH
        )
        prompt = (
            prompt_template
            .replace("{source_units}", source_text)
            .replace("{evidence_facts}", json.dumps(fact_payload, ensure_ascii=False))
        )
        allowed_fact_ids = {
            int(item["id"]) for item in fact_list
            if str(item.get("id") or "").isdigit()
        }
        for _ in range(2):
            parsed = self._llm_client.parse_json_object_from_llm_text(
                self._llm_client.call_llm(prompt) or ""
            ) or {}
            raw_items = parsed.get("user_activities")
            if not isinstance(raw_items, list):
                continue
            activities: List[Dict[str, Any]] = []
            seen: set[str] = set()
            for raw in raw_items[:12]:
                if not isinstance(raw, dict):
                    continue
                activity_text = _compact_whitespace(raw.get("activity_text") or "")
                user_role = str(raw.get("user_role") or "").strip().lower()
                if not activity_text or user_role not in {"actor", "participant", "contextual"}:
                    continue
                if activity_text in seen:
                    continue
                seen.add(activity_text)
                raw_evidence_ids = raw.get("evidence_fact_ids")
                if not isinstance(raw_evidence_ids, list):
                    raw_evidence_ids = []
                confidence = self._clamp_float(raw.get("confidence"), 0.0, 1.0, 0.5)
                if not math.isfinite(confidence):
                    confidence = 0.5
                activities.append({
                    "activity_text": activity_text,
                    "user_role": user_role,
                    "confidence": confidence,
                    "evidence_fact_ids": list(dict.fromkeys(
                        int(value) for value in raw_evidence_ids
                        if str(value).isdigit() and int(value) in allowed_fact_ids
                    )),
                })
            return activities
        return None

    def _normalize_memory_fact_extraction_llm_output(
        self,
        data: Dict[str, Any],
        units: Sequence[MemoryUnit],
        *,
        prompt_language: str,
        known_world_owner_aliases: Sequence[str],
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        diagnostics = diagnostics if diagnostics is not None else {}
        raw_facts = data.get("facts")
        if not isinstance(raw_facts, list):
            diagnostics.update({
                "normalization_status": "invalid_facts_field",
                "facts_field_type": type(raw_facts).__name__,
                "raw_fact_count": None,
                "normalized_fact_count": 0,
                "dropped_fact_counts": {},
                "fact_candidates": [],
            })
            return None
        facts: List[Dict[str, Any]] = []
        dropped_fact_counts: Dict[str, int] = {}
        candidate_diagnostics: List[Dict[str, Any]] = []

        def record_drop(reason: str) -> None:
            dropped_fact_counts[reason] = dropped_fact_counts.get(reason, 0) + 1

        def record_candidate(
            *,
            index: int,
            raw_fact: Any,
            result: str,
        ) -> None:
            if len(candidate_diagnostics) >= 12:
                return
            raw_payload = raw_fact if isinstance(raw_fact, dict) else {}
            candidate_diagnostics.append({
                "index": index,
                "result": result,
                "text": self._format_log_text(
                    raw_payload.get("text") or raw_payload.get("summary") or "",
                    limit=300,
                ),
                "fact_type": _compact_whitespace(raw_payload.get("fact_type") or ""),
            })

        dialogue_time_key = _to_timestamp_text(
            units[0].start_time if units else ""
        ) or _now_text()
        newly_confirmed_aliases: List[str] = []
        for index, raw_fact in enumerate(raw_facts, 1):
            if not isinstance(raw_fact, dict):
                record_drop("non_object_fact")
                record_candidate(
                    index=index,
                    raw_fact=raw_fact,
                    result="dropped_non_object_fact",
                )
                continue
            text = _compact_whitespace(raw_fact.get("text") or raw_fact.get("summary") or "")
            if not text:
                record_drop("missing_text")
                record_candidate(
                    index=index,
                    raw_fact=raw_fact,
                    result="dropped_missing_text",
                )
                continue
            keywords = self._normalize_string_list(raw_fact.get("keywords"), limit=18)
            if not keywords:
                keywords = self._keywords(text, limit=18)
            raw_entity_descriptors = self._normalize_entity_descriptors(
                raw_fact.get("entities"), limit=16,
            )
            alias_candidates = self._normalize_world_owner_alias_candidates(
                raw_fact.get("world_owner_alias_candidates"),
                fact_text=text,
                entity_names=[item["name"] for item in raw_entity_descriptors],
                source_units=units,
            )
            confirmed_aliases = self._confirmed_owner_aliases_in_fact(
                entity_names=[item["name"] for item in raw_entity_descriptors],
                fact_text=text,
                confirmed_aliases=[
                    *known_world_owner_aliases,
                    *newly_confirmed_aliases,
                    *(
                        item["alias_name"] for item in alias_candidates
                        if item["status"] == "confirmed"
                    ),
                ],
            )
            owner_alias_reference = bool(confirmed_aliases)
            allows_world_owner_reference = (
                self._units_include_world_owner(units) or owner_alias_reference
            )
            entity_descriptors = self._normalize_fact_entity_descriptors(
                raw_fact.get("entities"),
                allows_world_owner_reference=allows_world_owner_reference,
            )
            if owner_alias_reference:
                self._merge_entity_descriptor(entity_descriptors, {
                    "name": self._world_owner_entity_name,
                    "type": "PERSON",
                })
            entities = [item["name"] for item in entity_descriptors]
            if not entities:
                entity_descriptors = self._normalize_fact_entity_descriptors(
                    self._entities(text),
                    allows_world_owner_reference=allows_world_owner_reference,
                )
                entities = [item["name"] for item in entity_descriptors]
            primary_candidates = self._normalize_fact_entity_descriptors(
                raw_fact.get("primary_entity"),
                allows_world_owner_reference=allows_world_owner_reference,
            )
            if primary_candidates and not any(
                descriptor["name"] == primary_candidates[0]["name"]
                for descriptor in entity_descriptors
            ):
                self._merge_entity_descriptor(
                    entity_descriptors,
                    primary_candidates[0],
                )
                entities = [item["name"] for item in entity_descriptors]
            primary_entity = (
                primary_candidates[0]
                if primary_candidates
                else (entity_descriptors[0] if entity_descriptors else None)
            )
            if primary_entity is not None:
                primary_entity = next(
                    (
                        descriptor
                        for descriptor in entity_descriptors
                        if descriptor["name"] == primary_entity["name"]
                    ),
                    primary_entity,
                )
            primary_entity_name = _compact_whitespace(
                (primary_entity or {}).get("name") or ""
            ).lower()
            if primary_entity_name in {"assistant", "agent", "the assistant", "助手"} \
                    and self._is_low_value_assistant_closing(text):
                record_drop("low_value_assistant_closing")
                record_candidate(
                    index=index,
                    raw_fact=raw_fact,
                    result="dropped_low_value_assistant_closing",
                )
                continue
            fact_topic_fallback = " ".join(keywords[:3]) if keywords else "general"
            fact_topic = (
                self._normalize_topic_name(raw_fact.get("fact_topic"))
                or self._normalize_topic_name(fact_topic_fallback)
                or "general"
            )
            user_claim_signals = self._normalize_user_claim_signal(
                raw_fact.get("user_claim_signals"),
                allows_world_owner_reference=allows_world_owner_reference,
                source_units=units,
                owner_aliases=confirmed_aliases,
            )
            user_action_signals = self._normalize_user_action_signals(
                raw_fact.get("user_action_signals"),
                fact_entities=entities,
                allows_world_owner_reference=allows_world_owner_reference,
                source_units=units,
                owner_aliases=confirmed_aliases,
            )
            user_intent_signals = self._normalize_user_intent_signals(
                raw_fact.get("user_intent_signals"),
                fallback_entity=primary_entity,
                allows_world_owner_reference=allows_world_owner_reference,
                source_units=units,
                owner_aliases=confirmed_aliases,
            )
            event_time_key = _compact_whitespace(raw_fact.get("event_time_key") or "")
            facts.append({
                "summary": text,
                "fact_type": self._normalize_fact_type(raw_fact.get("fact_type")),
                "event_time_key": event_time_key,
                "dialogue_time_key": dialogue_time_key,
                "keywords": keywords,
                "entities": entities,
                "entity_descriptors": entity_descriptors,
                "primary_entity": primary_entity,
                "user_claim_signals": user_claim_signals,
                "user_action_signals": user_action_signals,
                "user_intent_signals": user_intent_signals,
                "world_owner_alias_candidates": alias_candidates,
                "resolved_world_owner_aliases": confirmed_aliases,
                "fact_topic": fact_topic,
                "importance": 0.5,
                "confidence": 0.9,
                "metadata": {
                    "extractor": "llm",
                    "time_confidence": _compact_whitespace(raw_fact.get("time_confidence") or "unknown"),
                    "where": _compact_whitespace(raw_fact.get("where") or ""),
                    "has_world_owner_speaker": (
                        any(
                            signal.get("owner_evidence_type") in {
                                "self_statement", "confirmed_response",
                            }
                            for signal in user_claim_signals
                        )
                        or any(
                            signal.get("assertion_source") in {
                                "self_statement", "confirmed_response",
                            }
                            for signal in user_intent_signals
                        )
                        or (
                            self._units_include_world_owner(units)
                            and any(
                                signal.get("assertion_source") == "observed_event"
                                for signal in user_intent_signals
                            )
                        )
                    ),
                    "has_world_owner_alias_reference": owner_alias_reference,
                },
            })
            newly_confirmed_aliases.extend(
                item["alias_name"] for item in alias_candidates
                if item["status"] == "confirmed"
            )
            record_candidate(
                index=index,
                raw_fact=raw_fact,
                result="accepted",
            )
        diagnostics.update({
            "normalization_status": "ok",
            "facts_field_type": "list",
            "raw_fact_count": len(raw_facts),
            "normalized_fact_count": len(facts),
            "dropped_fact_counts": dropped_fact_counts,
            "fact_candidates": candidate_diagnostics,
            "omitted_fact_candidate_diagnostics": max(
                0,
                len(raw_facts) - len(candidate_diagnostics),
            ),
        })
        return {"facts": facts}

    def _normalize_user_claim_signal(
        self,
        value: Any,
        *,
        allows_world_owner_reference: bool = False,
        source_units: Optional[Sequence[MemoryUnit]] = None,
        owner_aliases: Sequence[str] = (),
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if not isinstance(value, list):
            return []
        max_items = max(
            0,
            int(
                limit
                if limit is not None
                else self._memory_cfg.get("user_claim_signal_max_per_fact", 3) or 3
            ),
        )
        if max_items <= 0:
            return []
        allowed_types = self._user_claim_types()
        normalized: List[Dict[str, Any]] = []
        seen: set[Tuple[str, str, str]] = set()
        for raw in value:
            if not isinstance(raw, dict):
                continue
            claim_type_hint = str(raw.get("claim_type_hint") or "").strip().lower()
            if claim_type_hint not in allowed_types:
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
            subject_entity = _compact_whitespace(raw.get("subject_entity") or "")
            if not subject_entity:
                continue
            # user_claim signals are evidence about the configured wearer
            # only. A third-party entity may still appear in the fact or as
            # the object of a relation, but cannot be the claim subject.
            is_owner_reference = (
                self._is_world_owner_entity_reference(subject_entity)
                or subject_entity.lower() == self._world_owner_entity_name.lower()
            )
            if not allows_world_owner_reference or not is_owner_reference:
                continue
            owner_evidence = self._normalize_user_signal_owner_evidence(
                raw,
                source_units=source_units,
                owner_aliases=owner_aliases,
            )
            if owner_evidence is None:
                continue
            if owner_evidence["owner_evidence_type"] == "third_party_report":
                confidence = min(confidence, 0.65)
            key = (
                claim_type_hint,
                claim_anchor.lower(),
                evidence_basis.lower(),
            )
            if key in seen:
                continue
            seen.add(key)
            item: Dict[str, Any] = {
                "subject_entity": self._world_owner_entity_name,
                "claim_type_hint": claim_type_hint,
                "claim_anchor": claim_anchor,
                "evidence_basis": evidence_basis,
                "confidence": confidence,
                **owner_evidence,
            }
            normalized.append(item)
            if len(normalized) >= max_items:
                break
        return normalized

    def _normalize_user_action_signals(
        self,
        value: Any,
        *,
        fact_entities: Sequence[str],
        allows_world_owner_reference: bool,
        source_units: Optional[Sequence[MemoryUnit]] = None,
        owner_aliases: Sequence[str] = (),
    ) -> List[Dict[str, Any]]:
        """Retain only grounded, wearer-attributed concrete action evidence."""
        if not isinstance(value, list) or not allows_world_owner_reference:
            return []
        owner_name = self._world_owner_entity_name
        if owner_name not in fact_entities:
            return []
        normalized: List[Dict[str, Any]] = []
        seen: set[Tuple[str, str, str, str]] = set()
        for raw in value:
            if not isinstance(raw, dict):
                continue
            subject = _compact_whitespace(raw.get("subject_entity") or "")
            if not self._is_world_owner_entity_reference(subject) and (
                subject.lower() != owner_name.lower()
            ):
                continue
            action_kind = str(raw.get("action_kind") or "").strip().lower()
            if action_kind not in {"action", "choice", "participation"}:
                continue
            action_text = _compact_whitespace(raw.get("action_text") or "")[:480]
            evidence_basis = _compact_whitespace(raw.get("evidence_basis") or "")[:480]
            if not action_text or not evidence_basis:
                continue
            owner_evidence = self._normalize_user_signal_owner_evidence(
                raw,
                source_units=source_units,
                owner_aliases=owner_aliases,
            )
            if owner_evidence is None:
                continue
            key = (
                action_kind, action_text.lower(),
                owner_evidence["owner_evidence_text"].lower(),
                owner_evidence["other_evidence_text"].lower(),
            )
            if key in seen:
                continue
            seen.add(key)
            confidence = self._clamp_float(raw.get("confidence"), 0.0, 1.0, 0.7)
            if owner_evidence["owner_evidence_type"] == "third_party_report":
                confidence = min(confidence, 0.65)
            normalized.append({
                "subject_entity": owner_name,
                "action_kind": action_kind,
                "action_text": action_text,
                "evidence_basis": evidence_basis,
                "confidence": confidence,
                **owner_evidence,
            })
            if len(normalized) >= 3:
                break
        return normalized

    def _normalize_user_intent_signals(
        self,
        value: Any,
        *,
        fallback_entity: Optional[Dict[str, str]] = None,
        limit: Optional[int] = None,
        allows_world_owner_reference: bool = False,
        source_units: Optional[Sequence[MemoryUnit]] = None,
        owner_aliases: Sequence[str] = (),
    ) -> List[Dict[str, Any]]:
        """Normalize fact-level evidence for the prospective world model."""
        if not isinstance(value, list):
            return []
        max_items = max(
            0,
            int(
                limit
                if limit is not None
                else self._memory_cfg.get("user_intent_signal_max_per_fact", 2) or 2
            ),
        )
        if max_items <= 0:
            return []
        allowed_signal_purposes = {"future_intent", "lifecycle_update"}
        normalized: List[Dict[str, Any]] = []
        seen: set[Tuple[str, str, str, Tuple[str, ...]]] = set()
        for raw in value:
            if not isinstance(raw, dict):
                continue
            signal_purpose = str(raw.get("signal_purpose") or "").strip().lower()
            if signal_purpose not in allowed_signal_purposes:
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
            ))
            if not candidate_object_types:
                continue
            user_role = str(raw.get("user_role") or "").strip().lower()
            if user_role not in {"owner", "participant", "responsible"}:
                continue
            assertion_source = str(raw.get("assertion_source") or "").strip().lower()
            if assertion_source not in {
                "self_statement", "confirmed_response", "third_party_report",
                "observed_event",
            }:
                continue
            subject = raw.get("subject_entity") or raw.get("subject") or fallback_entity
            if isinstance(subject, dict):
                subject_name = _compact_whitespace(
                    subject.get("name") or subject.get("text") or ""
                )
            else:
                subject_name = _compact_whitespace(subject)
            if not allows_world_owner_reference or not (
                self._is_world_owner_entity_reference(subject_name)
                or subject_name.lower() == self._world_owner_entity_name.lower()
            ):
                continue
            subject_name = self._world_owner_entity_name
            user_intent_anchor = _compact_whitespace(
                raw.get("user_intent_anchor") or raw.get("intent_anchor") or ""
            )[:240]
            evidence_basis = _compact_whitespace(
                raw.get("evidence_basis") or raw.get("evidence") or raw.get("reason") or ""
            )[:480]
            if not subject_name or not user_intent_anchor or not evidence_basis:
                continue
            if assertion_source == "observed_event":
                if raw.get("owner_evidence_text") or raw.get("other_evidence_text"):
                    continue
                owner_evidence = {
                    "owner_evidence_text": "",
                    "other_evidence_text": "",
                }
            else:
                owner_evidence = self._normalize_user_signal_owner_evidence(
                    {**raw, "owner_evidence_type": assertion_source},
                    source_units=source_units,
                    owner_aliases=owner_aliases,
                )
                if owner_evidence is None:
                    continue
                owner_evidence.pop("owner_evidence_type")
            confidence = self._clamp_float(raw.get("confidence"), 0.0, 1.0, 0.75)
            if assertion_source == "third_party_report":
                confidence = min(confidence, 0.65)
            key = (
                subject_name.lower(),
                signal_purpose,
                self._generate_topic_name_key(user_intent_anchor),
                tuple(sorted(candidate_object_types)),
            )
            if key in seen:
                continue
            seen.add(key)
            normalized.append({
                "subject_entity": subject_name,
                "signal_purpose": signal_purpose,
                "candidate_object_types": candidate_object_types,
                "user_role": user_role,
                "user_intent_anchor": user_intent_anchor,
                "assertion_source": assertion_source,
                "evidence_basis": evidence_basis,
                "confidence": confidence,
                **owner_evidence,
            })
            if len(normalized) >= max_items:
                break
        return normalized

    @staticmethod
    def _is_world_owner_entity_reference(value: Any) -> bool:
        return _compact_whitespace(value).lower() in {
            alias.lower() for alias in _WORLD_OWNER_ENTITY_REFERENCE_ALIASES
        }

    @staticmethod
    def _units_include_world_owner(units: Sequence[MemoryUnit]) -> bool:
        """Return whether this fact task contains a verified wearer speaker."""
        return any(
            isinstance(unit, MemoryUnit)
            and any(
                _compact_whitespace(speaker) == "speaker_用户本人"
                for speaker in unit.speakers or []
            )
            for unit in units or ()
        )

    def _confirmed_owner_aliases_in_fact(
        self,
        *,
        entity_names: Sequence[str],
        fact_text: str,
        confirmed_aliases: Sequence[str],
    ) -> List[str]:
        """Resolve aliases co-referenced with the wearer, without another DB lookup."""
        if not any(
            self._is_world_owner_entity_reference(name)
            or _compact_whitespace(name).lower() == self._world_owner_entity_name.lower()
            for name in entity_names
        ):
            return []
        confirmed_keys = {
            self._db.normalize_owner_alias_key(name)
            for name in confirmed_aliases
        }
        aliases: List[str] = []
        for name in entity_names:
            normalized_name = _compact_whitespace(name)
            if (
                normalized_name
                and self._db.normalize_owner_alias_key(normalized_name) in confirmed_keys
                and normalized_name.casefold() in fact_text.casefold()
                and normalized_name not in aliases
            ):
                aliases.append(normalized_name)
        return aliases

    def _normalize_world_owner_alias_candidates(
        self,
        value: Any,
        *,
        fact_text: str,
        entity_names: Sequence[str],
        source_units: Sequence[MemoryUnit],
    ) -> List[Dict[str, Any]]:
        """Validate proposed owner names against current, speaker-attributed audio text."""
        if not isinstance(value, list):
            return []
        source_entries = [
            (_compact_whitespace(entry.get("speaker")), _compact_whitespace(entry.get("text")))
            for unit in source_units
            for entry in unit.entries()
        ]
        entity_keys = {
            self._db.normalize_owner_alias_key(name)
            for name in entity_names
        }
        candidates: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for raw in value[:4]:
            if not isinstance(raw, dict):
                continue
            alias_name = _compact_whitespace(raw.get("alias_name") or "")[:80]
            alias_key = self._db.normalize_owner_alias_key(alias_name)
            evidence_type = str(raw.get("evidence_type") or "").strip().lower()
            evidence_text = _compact_whitespace(raw.get("evidence_text") or "")[:480]
            response_text = _compact_whitespace(raw.get("owner_response_text") or "")[:240]
            if (
                len(alias_key) < 2 or alias_key in seen
                or alias_key not in entity_keys or alias_name not in fact_text
                or alias_name.lower() in _AMBIGUOUS_USER_ENTITY_ALIASES
                or self._is_world_owner_entity_reference(alias_name)
                or evidence_type not in {
                    "self_identification", "direct_address", "third_party_reference",
                }
                or not evidence_text or alias_name not in evidence_text
            ):
                continue
            matching_positions = [
                index for index, (speaker, text) in enumerate(source_entries)
                if evidence_text in text and (
                    speaker == "speaker_用户本人"
                    if evidence_type == "self_identification"
                    else speaker not in {"speaker_用户本人", "speaker_助手"}
                )
            ]
            if not matching_positions:
                continue
            if evidence_type == "self_identification":
                if not re.search(
                    rf"(?:我叫|我是|我的名字(?:是)?|本人叫|my name is|i am|i'm)\s*{re.escape(alias_name)}",
                    evidence_text,
                    re.IGNORECASE,
                ):
                    continue
                status = "confirmed"
                source_speaker = "speaker_用户本人"
            elif evidence_type == "direct_address":
                if not response_text or response_text in {
                    "嗯", "哦", "好", "好的", "知道了", "收到", "明白了",
                } or not any(
                    speaker == "speaker_用户本人" and response_text in text
                    for position in matching_positions
                    for speaker, text in source_entries[position + 1:]
                ):
                    continue
                status = "confirmed"
                source_speaker = source_entries[matching_positions[0]][0]
            else:
                if response_text:
                    continue
                status = "candidate"
                source_speaker = source_entries[matching_positions[0]][0]
            seen.add(alias_key)
            candidates.append({
                "alias_name": alias_name,
                "status": status,
                "confidence": self._clamp_float(raw.get("confidence"), 0.0, 1.0, 0.75),
                "evidence_type": evidence_type,
                "evidence_text": evidence_text,
                "source_speaker": source_speaker,
            })
        return candidates

    def _normalize_user_signal_owner_evidence(
        self,
        signal: Dict[str, Any],
        *,
        source_units: Optional[Sequence[MemoryUnit]],
        owner_aliases: Sequence[str] = (),
    ) -> Optional[Dict[str, str]]:
        """Validate the attributed quote, preserving whether it is hearsay."""
        evidence_type = str(signal.get("owner_evidence_type") or "").strip().lower()
        if evidence_type not in {"self_statement", "confirmed_response", "third_party_report"}:
            return None
        owner_text = _compact_whitespace(
            signal.get("owner_evidence_text") or ""
        ).strip("\"'“”")
        other_text = _compact_whitespace(
            signal.get("other_evidence_text") or ""
        ).strip("\"'“”")
        if evidence_type == "third_party_report":
            if (
                owner_text or not other_text or not any(
                    self._db.normalize_owner_alias_key(alias)
                    in self._db.normalize_owner_alias_key(other_text)
                    for alias in owner_aliases
                )
            ):
                return None
            if source_units is not None and not any(
                _compact_whitespace(entry.get("speaker") or "")
                not in {"speaker_用户本人", "speaker_助手"}
                and other_text in _compact_whitespace(entry.get("text") or "")
                for unit in source_units for entry in unit.entries()
            ):
                return None
            return {
                "owner_evidence_type": evidence_type,
                "owner_evidence_text": "",
                "other_evidence_text": other_text,
            }
        if not owner_text or (
            evidence_type == "confirmed_response" and not other_text
        ) or (
            evidence_type == "self_statement" and other_text
        ):
            return None
        if evidence_type == "confirmed_response" and owner_text in {
            "嗯", "哦", "好", "好的", "知道了", "收到", "明白了",
        }:
            return None
        if source_units is not None:
            source_entries = [
                (
                    _compact_whitespace(entry.get("speaker") or ""),
                    _compact_whitespace(entry.get("text") or ""),
                )
                for unit in source_units
                for entry in unit.entries()
            ]
            owner_positions = [
                index for index, (speaker, text) in enumerate(source_entries)
                if speaker == "speaker_用户本人" and owner_text in text
            ]
            if not owner_positions:
                return None
            if evidence_type == "confirmed_response" and not any(
                speaker != "speaker_用户本人" and other_text in text
                for owner_position in owner_positions
                for speaker, text in source_entries[:owner_position]
            ):
                return None
        return {
            "owner_evidence_type": evidence_type,
            "owner_evidence_text": owner_text,
            "other_evidence_text": other_text,
        }

    def _normalize_fact_entity_descriptors(
        self,
        value: Any,
        *,
        allows_world_owner_reference: bool,
        limit: int = 16,
    ) -> List[Dict[str, str]]:
        """Apply source-aware identity boundaries to extracted entities.

        A bare ``用户`` is ambiguous: in ambient audio it generally denotes a
        product user or customer, not the wearer. Only the reserved
        ``用户本人`` / ``user self`` reference from a tagged interaction may
        resolve to this database's world owner.
        """
        descriptors = self._normalize_entity_descriptors(value, limit=limit)
        normalized: List[Dict[str, str]] = []
        for descriptor in descriptors:
            name = _compact_whitespace(descriptor.get("name") or "")
            if not name or name.lower() in _AMBIGUOUS_USER_ENTITY_ALIASES:
                continue
            if self._is_world_owner_entity_reference(name):
                if not allows_world_owner_reference:
                    continue
                descriptor = {
                    "name": self._world_owner_entity_name,
                    "type": "PERSON",
                }
            self._merge_entity_descriptor(normalized, descriptor)
            if len(normalized) >= limit:
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
        user_label = "speaker_用户本人"
        assistant_label = "speaker_助手"
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
    def _parse_participants_from_units(units: Sequence[MemoryUnit]) -> List[str]:
        participants: List[str] = []
        seen: set[str] = set()
        for unit in units or []:
            for value in unit.speakers or []:
                speaker = _compact_whitespace(value)
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

    @classmethod
    def _merge_entity_descriptor(
        cls,
        descriptors: List[Dict[str, str]],
        value: Any,
    ) -> None:
        """Merge one typed entity without downgrading a concrete type."""
        if not isinstance(value, dict):
            return
        name = _compact_whitespace(value.get("name") or value.get("text") or "")
        if not cls._is_valid_entity_name(name):
            return
        entity_type = cls._normalize_entity_type(value.get("type"))
        for descriptor in descriptors:
            if _compact_whitespace(descriptor.get("name") or "") != name:
                continue
            if descriptor.get("type") == "OTHER" and entity_type != "OTHER":
                descriptor["type"] = entity_type
            return
        descriptors.append({"name": name, "type": entity_type})

    @classmethod
    def _fact_entity_descriptors(
        cls,
        fact: Dict[str, Any],
        *,
        entities: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, str]]:
        """Return typed entity descriptors while retaining name-only fallback."""
        metadata = fact.get("metadata") if isinstance(fact.get("metadata"), dict) else {}
        descriptor_source = fact.get("entity_descriptors")
        if descriptor_source is None:
            descriptor_source = metadata.get("entity_descriptors")
        descriptors = cls._normalize_entity_descriptors(
            descriptor_source if descriptor_source is not None else fact.get("entities"),
            limit=32,
        )
        names = (
            list(entities)
            if entities is not None
            else cls._normalize_entity_names(fact.get("entities"), limit=32)
        )
        for name in names:
            cls._merge_entity_descriptor(
                descriptors,
                {"name": name, "type": "OTHER"},
            )
        primary = fact.get("primary_entity")
        if isinstance(primary, dict):
            cls._merge_entity_descriptor(descriptors, primary)
        elif primary:
            cls._merge_entity_descriptor(
                descriptors,
                {"name": primary, "type": "OTHER"},
            )
        return descriptors

    def _build_memory_units_for_prompt(
        self,
        units: Sequence[MemoryUnit],
        *,
        prompt_language: str,
    ) -> str:
        is_en = prompt_language == "en"
        time_label = "Time" if is_en else "时间"
        unit_type_label = "Unit type" if is_en else "单元类型"
        speaker_label = "Speaker" if is_en else "说话人"
        text_label = "Text" if is_en else "文本"
        blocks: List[str] = []
        for index, unit in enumerate(units, 1):
            entry_blocks: List[str] = []
            for entry_index, text in enumerate(unit.texts or [], 1):
                started_at = (
                    unit.started_ats[entry_index - 1]
                    if entry_index <= len(unit.started_ats)
                    else ""
                ) or ""
                ended_at = (
                    unit.ended_ats[entry_index - 1]
                    if entry_index <= len(unit.ended_ats)
                    else started_at
                ) or started_at
                time_text = (
                    started_at
                    if started_at == ended_at
                    else f"{started_at} - {ended_at}"
                )
                speaker = (
                    unit.speakers[entry_index - 1]
                    if entry_index <= len(unit.speakers)
                    else ""
                )
                entry_blocks.append("\n".join([
                    f"{time_label}: {time_text}",
                    f"{speaker_label}: {speaker}",
                    f"{text_label}: {text}",
                ]))
            blocks.append("\n".join([
                f"[Unit {index}]",
                f"{unit_type_label}: {unit.unit_type or 'transcript'}",
                *entry_blocks,
            ]))
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
    ) -> List[Dict[str, Any]]:
        """Create compact topic-registry updates from persisted memory fields."""
        normalized_fact_id = int(fact_id or 0)
        updates: List[Dict[str, Any]] = []
        def add(topic_kind: str, value: Any) -> None:
            topic_name = self._normalize_topic_name(value)
            if not topic_name or not self._is_indexable_memory_topic(topic_name):
                return
            topic_key = self._generate_topic_name_key(topic_name)
            item: Dict[str, Any] = {
                "topic_kind": topic_kind,
                "topic_name": topic_name,
                "topic_key": topic_key,
                "fact_ids": [normalized_fact_id] if normalized_fact_id > 0 else [],
            }
            if not any(
                existing["topic_kind"] == topic_kind
                and existing["topic_key"] == topic_key
                for existing in updates
            ):
                updates.append(item)

        for topic in canonical_topics or ():
            add("canonical", topic)
        # Aspect topics are defined only by facts.
        if normalized_fact_id > 0:
            for topic in aspect_topics or ():
                add("aspect", topic)
        return updates

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

    def _resolve_prompt_language_from_memory_units(
        self,
        units: Sequence[MemoryUnit],
    ) -> str:
        """Resolve one LLM task's prompt language from its actual source text."""
        text = "\n".join(
            _compact_whitespace(unit.text)
            for unit in units
            if isinstance(unit, MemoryUnit)
        )
        return self._resolve_prompt_language_from_text(text)

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

    def _store_extracted_memory_facts_into_db(
        self,
        *,
        facts: List[Dict[str, Any]],
        tags: List[str],
    ) -> Dict[str, Any]:
        fact_ids: List[int] = []
        topic_item_updates: List[Dict[str, Any]] = []
        user_claim_signal_mapping_updates: List[Dict[str, Any]] = []
        user_action_signal_updates: List[Dict[str, Any]] = []
        user_intent_signal_mapping_updates: List[Dict[str, Any]] = []
        user_claim_signal_fact_ids: List[int] = []
        user_action_signal_fact_ids: List[int] = []
        user_intent_signal_fact_ids: List[int] = []
        owner_entity_id = self._intent_world_owner_entity_id()
        stored_confirmed_alias_keys = {
            str(row["alias_key"])
            for row in self._db.get_owner_entity_aliases(
                owner_entity_id, status="confirmed", limit=4096,
            )
        }
        for fact in facts:
            keywords = self._normalize_string_list(
                fact.get("keywords"),
                limit=18,
            )
            if not keywords:
                keywords = self._keywords(fact.get("summary") or "", limit=18)
            fact_entities = list(fact.get("entities") or [])
            entity_descriptors = list(fact.get("entity_descriptors") or [])
            raw_metadata = fact.get("metadata")
            metadata = dict(raw_metadata) if isinstance(raw_metadata, dict) else {}
            resolved_world_owner_aliases = list(fact["resolved_world_owner_aliases"])
            fallback_topics = self._topic_candidates(fact["summary"])
            fact_topic = (
                self._normalize_topic_name(fact.get("fact_topic"))
                or self._normalize_topic_name(
                    fallback_topics[0] if fallback_topics else ""
                )
                or "general"
            )
            owner_alias_keys = {
                self._db.normalize_owner_alias_key(alias)
                for alias in resolved_world_owner_aliases
            }
            entity_mapping = self._db.upsert_entity_nodes([
                descriptor for descriptor in entity_descriptors
                if self._db.normalize_owner_alias_key(descriptor.get("name") or "")
                not in owner_alias_keys
            ])
            if owner_alias_keys:
                for descriptor in entity_descriptors:
                    name = str(descriptor.get("name") or "")
                    if self._db.normalize_owner_alias_key(name) in owner_alias_keys:
                        entity_mapping[name] = owner_entity_id
            entity_ids = [
                entity_mapping[entity_name]
                for entity_name in fact_entities
                if entity_name in entity_mapping
            ]
            fact_metadata = {
                **metadata,
                "tags": tags,
                "entity_descriptors": entity_descriptors,
            }
            fact_id = self._db.insert_fact(
                fact_type=fact["fact_type"],
                summary=fact["summary"],
                keywords=keywords,
                entities=fact_entities,
                entity_ids=entity_ids,
                fact_topic=fact_topic,
                event_time_key=fact.get("event_time_key") or "",
                dialogue_time_key=fact.get("dialogue_time_key") or "",
                confidence=fact["confidence"],
                importance=fact["importance"],
                metadata=fact_metadata,
            )
            confirmed_candidates_by_key = {
                self._db.normalize_owner_alias_key(alias["alias_name"]): alias
                for alias in fact.get("world_owner_alias_candidates") or []
                if isinstance(alias, dict) and alias.get("status") == "confirmed"
            }
            for alias_name in resolved_world_owner_aliases:
                alias_key = self._db.normalize_owner_alias_key(alias_name)
                alias = confirmed_candidates_by_key.get(alias_key)
                if alias_key in stored_confirmed_alias_keys or alias is None:
                    # Known aliases need no new row, and fresh aliases need
                    # current-fact confirmation evidence before persistence.
                    continue
                self._db.upsert_owner_entity_alias(
                    owner_entity_id=owner_entity_id,
                    alias_name=alias_name,
                    status="confirmed",
                    confidence=alias["confidence"],
                    evidence_fact_id=fact_id,
                    evidence_type=alias["evidence_type"],
                    evidence_text=alias["evidence_text"],
                    source_speaker=alias["source_speaker"],
                )
                stored_confirmed_alias_keys.add(alias_key)
            self._upsert_fact_recall_document(
                fact_id=fact_id,
                summary=fact["summary"],
                keywords=keywords,
                entities=fact_entities,
                entity_ids=entity_ids,
                fact_topic=fact_topic,
                event_time_key=fact.get("event_time_key") or "",
                dialogue_time_key=fact.get("dialogue_time_key") or "",
                confidence=float(fact["confidence"]),
                importance=float(fact["importance"]),
            )
            fact_ids.append(fact_id)
            user_claim_updates = (
                self._fact_user_claim_signal_mapping_updates(fact_id, fact)
            )
            user_claim_signal_mapping_updates.extend(user_claim_updates)
            if user_claim_updates:
                user_claim_signal_fact_ids.append(fact_id)
            action_updates = self._fact_user_action_signal_updates(
                fact_id=fact_id,
                fact=fact,
                owner_entity_id=owner_entity_id,
            )
            user_action_signal_updates.extend(action_updates)
            if action_updates:
                user_action_signal_fact_ids.append(fact_id)
            user_intent_updates = (
                self._fact_user_intent_signal_mapping_updates(fact_id, fact)
            )
            user_intent_signal_mapping_updates.extend(user_intent_updates)
            if user_intent_updates:
                user_intent_signal_fact_ids.append(fact_id)
            topic_item_updates.extend(
                self._build_memory_topic_item_updates(
                    canonical_topics=[],
                    aspect_topics=[fact_topic],
                    fact_id=fact_id,
                )
            )
        self._db.upsert_fact_user_claim_signals(
            user_claim_signal_mapping_updates
        )
        self._db.upsert_fact_user_action_signals(user_action_signal_updates)
        self._db.upsert_fact_user_intent_signals(
            user_intent_signal_mapping_updates
        )
        return {
            "fact_ids": fact_ids,
            "topic_item_updates": topic_item_updates,
            "user_claim_signal_fact_count": len(user_claim_signal_fact_ids),
            "user_claim_signal_count": len(user_claim_signal_mapping_updates),
            "user_action_signal_fact_count": len(user_action_signal_fact_ids),
            "user_action_signal_count": len(user_action_signal_updates),
            "user_intent_signal_fact_count": len(
                user_intent_signal_fact_ids
            ),
            "user_intent_signal_count": len(user_intent_signal_mapping_updates),
        }

    def _process_memory_user_observed_pattern_update_task(
        self, *, signal_ids: Optional[Sequence[int]] = None,
        reference_timestamp: Optional[Any] = None,
        **_: Any,
    ) -> Dict[str, Any]:
        """Find similar past actions and reconcile resulting pattern candidates."""
        selected_ids = list(dict.fromkeys(
            int(value) for value in signal_ids or [] if int(value) > 0
        ))
        if not selected_ids:
            return {"status": "empty", "signal_count": 0,
                    "new_pattern_ids": [], "pattern_relation_event_ids": [],
                    "_completed_signal_ids": []}
        started_at = time.monotonic()
        update = self._update_user_observed_patterns_from_action_signals(
            selected_ids,
        )
        completed = bool(update.get("completed"))
        pattern_relation_event_ids = [int(event["id"])
            for event in self._db.get_user_observed_pattern_events_for_signal_ids(
                selected_ids
            )
        ] if completed else []
        report = {
            "status": "ok" if completed else "failed",
            "signal_count": len(selected_ids),
            "similar_action_batches": int(update.get("similar_action_batches") or 0),
            "pattern_relations_recorded": int(update.get("relations_recorded") or 0),
            "direct_support_signal_count": int(
                update.get("direct_support_signal_count") or 0
            ),
            "user_observed_patterns_created": int(update.get("created") or 0),
            "user_observed_patterns_updated": int(update.get("updated") or 0),
            "new_pattern_ids": update.get("new_pattern_ids") or [],
            "pattern_relation_event_ids": pattern_relation_event_ids,
            "reference_timestamp": reference_timestamp,
            "total_elapsed_ms": round((time.monotonic() - started_at) * 1000, 2),
            "_completed_signal_ids": selected_ids if completed else [],
        }
        if update.get("error"):
            report["error"] = update["error"]
        self._log_info("memory_user_observed_pattern_update", "finish", report)
        return report

    def _process_memory_user_claim_update_task(
        self,
        reference_timestamp: Optional[Any] = None,
        signal_ids: Optional[Sequence[int]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Project new facts into user claims."""
        update_started_at = time.monotonic()
        selected_signal_ids = list(dict.fromkeys(
            int(value) for value in signal_ids or [] if int(value) > 0
        ))
        if reference_timestamp is None:
            reference_timestamp = kwargs.get("timestamp") or _now_text()
        self._log_info("memory_user_claim", "start", {
            "timestamp": reference_timestamp,
            "user_claim_signal_count": len(selected_signal_ids),
        })
        claim_report = self._update_memory_user_claims(
            signal_ids=selected_signal_ids,
            reference_timestamp=reference_timestamp,
        )
        new_claim_ids = list(claim_report.get("new_claim_ids") or [])
        claim_relations = list(claim_report.get("claim_relations") or [])
        self._log_info("memory_user_claim", "user_claim_update_finish", claim_report)
        explicit_failed_count = int(
            claim_report.get("explicit", {}).get("failed_signal_count") or 0
        )
        completed_signal_count = len(
            claim_report.get("_completed_signal_ids") or []
        )
        failed_derived_seed_claim_ids = list(
            claim_report.get("derived", {}).get("failed_seed_claim_ids") or []
        )
        has_failures = explicit_failed_count > 0 or bool(failed_derived_seed_claim_ids)
        report = {
            "status": (
                "partial" if has_failures and completed_signal_count
                else "failed" if has_failures
                else "ok" if completed_signal_count else "empty"
            ),
            "explicit_failed_signal_count": explicit_failed_count,
            "failed_derived_seed_claim_ids": failed_derived_seed_claim_ids,
            "derived_failures": list(claim_report.get("derived", {}).get("failures") or []),
            "explicit_claims_updated": int(
                claim_report.get("explicit", {}).get("updated", 0) or 0
            ),
            "derived_claims_updated": int(
                claim_report.get("derived", {}).get("updated", 0) or 0
            ),
            "total_elapsed_ms": round(
                (time.monotonic() - update_started_at) * 1000,
                2,
            ),
        }
        report["new_claim_ids"] = new_claim_ids
        report["claim_relations"] = claim_relations
        report["reference_timestamp"] = reference_timestamp
        report["_completed_signal_ids"] = claim_report.get(
            "_completed_signal_ids", [],
        )
        self._log_info("memory_user_claim", "finish", report)
        return report

    def _process_memory_user_intent_update_task(
        self,
        *,
        reference_timestamp: Optional[Any] = None,
        signal_ids: Optional[Sequence[int]] = None,
    ) -> Dict[str, Any]:
        """Process independent signal batches and return only committed change events."""
        reference_timestamp = reference_timestamp or _now_text()
        selected_signal_ids = list(dict.fromkeys(
            int(value) for value in signal_ids or [] if int(value) > 0
        ))
        pending_signals = self._db.get_user_intent_signals_by_ids(
            selected_signal_ids,
        )
        facts = self._db.get_memory_facts_by_ids(
            sorted({int(signal["fact_id"]) for signal in pending_signals})
        )
        report: Dict[str, Any] = {
            "status": "empty",
            "reference_timestamp": str(reference_timestamp),
            "intent_event_ids": [],
            "_completed_signal_ids": [],
            "failed_signal_ids": [],
        }
        facts_by_id = {
            int(fact["id"]): fact
            for fact in facts
            if str(fact.get("id") or "").strip().isdigit()
        }
        signal_items: List[Dict[str, Any]] = []
        for signal in pending_signals:
            fact_id = int(signal["fact_id"])
            fact = facts_by_id.get(fact_id)
            if fact is None:
                continue
            signal_items.append({
                "signal_id": int(signal["signal_id"]),
                "fact_id": fact_id,
                "fact": fact,
                "signal": signal,
            })
        self._log_user_intent_signals_loaded(signal_items, reference_timestamp)
        available_signal_ids = {item["signal_id"] for item in signal_items}
        report["failed_signal_ids"] = sorted(set(selected_signal_ids) - available_signal_ids)
        if report["failed_signal_ids"]:
            self._log_info("memory_user_intent_update", "batch_error", {
                "phase": "input_validation",
                "signal_ids": report["failed_signal_ids"],
                "error": "missing_signal_or_supporting_fact",
            })
        if not signal_items:
            if report["failed_signal_ids"]:
                report["status"] = "failed"
            self._log_info("memory_user_intent_update", "finish", report)
            return report

        world_owner_id = self._intent_world_owner_entity_id()
        batch_size = max(
            1,
            int(self._memory_cfg.get("user_intent_signal_batch_size") or 8),
        )
        for batch_start in range(0, len(signal_items), batch_size):
            batch = signal_items[batch_start:batch_start + batch_size]
            batch_index = batch_start // batch_size
            language = self._resolve_prompt_language_from_text("\n".join(
                str(item["fact"].get("summary") or "") for item in batch
            ))
            prompt_template = (
                INTENT_EXTRACTION_PROMPT_EN
                if language == "en" else INTENT_EXTRACTION_PROMPT_ZH
            )
            batch_error = ""
            try:
                raw = self._llm_client.call_llm(
                    prompt_template.replace("{reference_timestamp}", str(reference_timestamp))
                    .replace("{signal_items}", json.dumps(
                        [self._user_intent_signal_prompt_view(item) for item in batch],
                        ensure_ascii=False,
                        indent=2,
                    ))
                )
                parsed = self._llm_client.parse_json_object_from_llm_text(raw or "")
            except Exception as exc:
                parsed = None
                batch_error = str(exc)
            if not isinstance(parsed, dict) or not isinstance(parsed.get("results"), list):
                report["failed_signal_ids"].extend(item["signal_id"] for item in batch)
                self._log_info("memory_user_intent_update", "batch_error", {
                    "batch_index": batch_index,
                    "signal_ids": [item["signal_id"] for item in batch],
                    "phase": "extraction",
                    "error": batch_error or "invalid_llm_user_intent_extraction_response",
                })
                continue

            try:
                normalized = self._normalize_intent_candidates(
                    parsed["results"],
                    signal_items=batch,
                    world_owner_id=world_owner_id,
                )
            except sqlite3.Error:
                raise
            except Exception as exc:
                report["failed_signal_ids"].extend(item["signal_id"] for item in batch)
                self._log_info("memory_user_intent_update", "batch_error", {
                    "batch_index": batch_index,
                    "signal_ids": [item["signal_id"] for item in batch],
                    "phase": "normalization",
                    "error_type": type(exc).__name__, "error": str(exc),
                })
                continue
            candidates = normalized["future_intent_candidates"]
            observations = normalized["intent_lifecycle_observations"]
            report["_completed_signal_ids"].extend(normalized["empty_signal_ids"])
            report["failed_signal_ids"].extend(normalized["failed_signal_ids"])
            phase = "future_intent_reconciliation"
            try:
                decisions = self._reconcile_future_intent_candidates(
                    candidates,
                    world_owner_id=world_owner_id,
                    prompt_language=language,
                )
                phase = "future_intent_save"
                applied = self._save_future_intent_candidates(
                    candidates,
                    decisions,
                    facts_by_id=facts_by_id,
                    reference_timestamp=reference_timestamp,
                )
                report["_completed_signal_ids"].extend(applied["completed_signal_ids"])
                report["failed_signal_ids"].extend(applied["failed_signal_ids"])
                report["intent_event_ids"].extend(applied["intent_event_ids"])
            except sqlite3.Error:
                raise
            except Exception as exc:
                report["failed_signal_ids"].extend(candidate["signal_id"] for candidate in candidates)
                self._log_info("memory_user_intent_update", "batch_error", {
                    "batch_index": batch_index,
                    "signal_ids": [candidate["signal_id"] for candidate in candidates],
                    "error_type": type(exc).__name__, "error": str(exc),
                    "phase": phase,
                })

            # Match after saving future candidates so this batch's newly stored
            # intents are visible too. A failure in either branch is independent.
            phase = "lifecycle_reconciliation"
            try:
                lifecycle_decisions = self._reconcile_intent_lifecycle_observations(
                    observations, facts_by_id=facts_by_id,
                    world_owner_id=world_owner_id, prompt_language=language,
                )
                phase = "lifecycle_save"
                saved = self._save_intent_lifecycle_observations(
                    observations, lifecycle_decisions, world_owner_id=world_owner_id,
                )
            except sqlite3.Error:
                raise
            except Exception as exc:
                saved = {
                    "completed_signal_ids": [],
                    "failed_signal_ids": [item["signal_id"] for item in observations],
                    "intent_event_ids": [], "intent_lifecycle_observation_ids": [],
                    "unmatched_observation_ids": [],
                }
                self._log_info("memory_user_intent_update", "batch_error", {
                    "batch_index": batch_index,
                    "signal_ids": saved["failed_signal_ids"],
                    "phase": phase,
                    "error_type": type(exc).__name__, "error": str(exc),
                })
            report["_completed_signal_ids"].extend(saved["completed_signal_ids"])
            report["failed_signal_ids"].extend(saved["failed_signal_ids"])
            report["intent_event_ids"].extend(saved["intent_event_ids"])
            self._log_info("memory_user_intent_update", "batch_finish", {
                "batch_index": batch_index,
                "signal_ids": [item["signal_id"] for item in batch],
                "candidate_count": len(candidates),
                "lifecycle_observation_count": len(observations),
                "lifecycle_observation_saved_count": len(saved["intent_lifecycle_observation_ids"]),
                "lifecycle_relation_count": (
                    len(saved["intent_lifecycle_observation_ids"])
                    - len(saved["unmatched_observation_ids"])
                ),
                "unmatched_lifecycle_observation_ids": saved["unmatched_observation_ids"],
            })

        report["intent_event_ids"] = sorted(set(report["intent_event_ids"]))
        report["_completed_signal_ids"] = sorted(set(report["_completed_signal_ids"]))
        # Never silently lose an input that received no terminal handler result.
        report["failed_signal_ids"] = sorted(
            set(report["failed_signal_ids"])
            | (set(selected_signal_ids) - set(report["_completed_signal_ids"]))
        )
        report["status"] = (
            "partial" if report["failed_signal_ids"] and report["_completed_signal_ids"]
            else "failed" if report["failed_signal_ids"] else "ok"
        )
        self._log_info("memory_user_intent_update", "finish", report)
        return report

    def _user_intent_signal_prompt_view(
        self,
        item: Dict[str, Any],
    ) -> Dict[str, Any]:
        signal = item["signal"]
        return {
            "signal_id": item["signal_id"],
            "fact": self._intent_fact_prompt_view(item["fact"]),
            "user_intent_signal": {
                key: signal.get(key)
                for key in (
                    "signal_purpose", "candidate_object_types", "subject_entity",
                    "user_role", "user_intent_anchor", "assertion_source",
                    "evidence_basis", "confidence",
                )
            },
        }

    def _retrieve_user_intent_related_intent_objects(
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
        related_by_type: Dict[str, List[Dict[str, Any]]] = {}
        for object_type in ("goal", "plan", "work_item"):
            existing = self._db.get_intent_objects(
                object_type=object_type,
                world_owner_entity_id=world_owner_id,
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
                for item in strong_matches:
                    object_id = int(item.get("id") or 0)
                    if object_id > 0:
                        selected_by_id[object_id] = item
            related_by_type[object_type] = list(selected_by_id.values())
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
            "topics": [fact.get("fact_topic") or ""],
        }

    def _normalize_intent_candidates(
        self,
        results: Sequence[Any],
        *,
        signal_items: Sequence[Dict[str, Any]],
        world_owner_id: int,
    ) -> Dict[str, Any]:
        """Validate and route batch results by signal purpose before normalizing each output."""
        results_by_id: Dict[int, List[Dict[str, Any]]] = {}
        for result in results:
            if not isinstance(result, dict):
                continue
            try:
                signal_id = int(result.get("signal_id"))
            except (TypeError, ValueError):
                continue
            results_by_id.setdefault(signal_id, []).append(result)

        candidates: List[Dict[str, Any]] = []
        observations: List[Dict[str, Any]] = []
        empty_signal_ids: List[int] = []
        errors: Dict[int, str] = {}
        for item in signal_items:
            signal_id = item["signal_id"]
            signal_results = results_by_id.get(signal_id) or []
            if len(signal_results) != 1 or not all(
                key in signal_results[0]
                for key in ("future_intent_candidate", "intent_lifecycle_observation")
            ):
                errors[signal_id] = "missing_or_duplicate_signal_result"
                continue
            result = signal_results[0]
            purpose = item["signal"].get("signal_purpose")
            output_key = {
                "future_intent": "future_intent_candidate",
                "lifecycle_update": "intent_lifecycle_observation",
            }.get(purpose)
            if output_key is None or any(
                result[key] is not None
                for key in ("future_intent_candidate", "intent_lifecycle_observation")
                if key != output_key
            ):
                errors[signal_id] = "invalid_signal_result_route"
                continue
            raw_candidate = result[output_key]
            if raw_candidate is None:
                empty_signal_ids.append(signal_id)
                continue
            fact_id = item["fact_id"]
            if (
                not isinstance(raw_candidate, dict)
                or raw_candidate.get("evidence_fact_ids") != [fact_id]
                or raw_candidate.get("object_type") not in item["signal"]["candidate_object_types"]
            ):
                errors[signal_id] = "invalid_signal_result_scope"
                continue
            if purpose == "lifecycle_update":
                observation = self._normalize_single_intent_lifecycle_observation(raw_candidate)
                if observation is None:
                    errors[signal_id] = "invalid_normalized_intent_lifecycle_observation"
                    continue
                observation["signal_id"] = signal_id
                observations.append(observation)
                continue
            fact = item["fact"]
            entity_ids_by_name = {
                _compact_whitespace(name): int(entity_id)
                for name, entity_id in zip(
                    fact.get("entities") or [], fact.get("entity_ids") or [],
                )
                if _compact_whitespace(name) and str(entity_id).isdigit()
                and int(entity_id) > 0
            }
            try:
                candidate = self._normalize_single_intent_candidate(
                    raw_candidate,
                    object_type=raw_candidate["object_type"],
                    facts_by_id={fact_id: fact},
                    entity_ids=entity_ids_by_name,
                    world_owner_id=world_owner_id,
                )
            except Exception as exc:
                errors[signal_id] = str(exc)
                continue
            if candidate is None:
                errors[signal_id] = "invalid_normalized_future_intent_candidate"
                continue
            candidate["signal_id"] = signal_id
            candidates.append(candidate)

        for signal_id, error in errors.items():
            self._log_info("memory_user_intent_update", "signal_error", {
                "signal_id": signal_id, "error": error,
            })
        return {
            "future_intent_candidates": candidates,
            "intent_lifecycle_observations": observations,
            "empty_signal_ids": empty_signal_ids,
            "failed_signal_ids": list(errors),
        }

    def _normalize_single_intent_lifecycle_observation(
        self, raw: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Normalize observed changes without reconstructing or updating an intent."""
        if any(not isinstance(raw.get(key), str) for key in (
            "object_type", "target_description", "observation_text", "lifecycle_evidence",
        )) or not isinstance(raw.get("effective_at", ""), str):
            return None
        object_type = str(raw.get("object_type") or "")
        allowed_evidence = {
            "goal": {"confirmed", "updated", "completed", "cancelled", "blocked"},
            "plan": {"confirmed", "updated", "occurred", "cancelled", "rescheduled", "blocked"},
            "work_item": {"confirmed", "updated", "completed", "cancelled", "blocked"},
        }
        lifecycle_evidence = str(raw.get("lifecycle_evidence") or "").strip().lower()
        target = _compact_whitespace(raw.get("target_description") or "")[:480]
        text = _compact_whitespace(raw.get("observation_text") or "")[:800]
        if not target or not text or lifecycle_evidence not in allowed_evidence.get(object_type, set()):
            return None
        effective_at = _compact_whitespace(raw.get("effective_at") or "")
        if effective_at:
            try:
                datetime.fromisoformat(effective_at.replace("Z", "+00:00"))
            except ValueError:
                return None
        raw_details = raw.get("details", {})
        if not isinstance(raw_details, dict):
            return None
        details: Dict[str, str] = {}
        for key in ("time_start", "time_end", "location", "progress_text"):
            value = raw_details.get(key)
            if value is None or value == "":
                continue
            if not isinstance(value, str):
                return None
            text_value = _compact_whitespace(value)[:480]
            if text_value:
                details[key] = text_value
        return {
            "object_type": object_type,
            "target_description": target,
            "observation_text": text,
            "lifecycle_evidence": lifecycle_evidence,
            "effective_at": effective_at,
            "confidence": self._clamp_float(raw.get("confidence"), 0.0, 1.0, 0.7),
            "evidence_fact_ids": list(raw["evidence_fact_ids"]),
            "details": details,
        }

    def _normalize_single_intent_candidate(
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
        details = raw.get("details")
        details = details if isinstance(details, dict) else {}
        evidence_ids = list(dict.fromkeys(
            int(value) for value in (raw.get("evidence_fact_ids") or [])
            if str(value).strip().isdigit() and int(value) in facts_by_id
        ))[:12]
        summary = _compact_whitespace(raw.get("summary") or "")[:480]
        primary_domain = self._normalize_primary_domain(raw.get("primary_domain"))
        if not summary or not evidence_ids or not primary_domain:
            return None
        allows_world_owner_reference = any(
            isinstance(facts_by_id[fact_id].get("metadata"), dict)
            and bool(
                facts_by_id[fact_id]["metadata"].get("has_world_owner_speaker")
                or facts_by_id[fact_id]["metadata"].get("has_world_owner_alias_reference")
            )
            for fact_id in evidence_ids
            if isinstance(facts_by_id.get(fact_id, {}).get("metadata"), dict)
        )
        canonical_key = self._generate_topic_name_key(raw.get("canonical_key") or summary)
        confidence = self._clamp_float(raw.get("confidence"), 0.0, 1.0, 0.7)
        candidate: Dict[str, Any] = {
            "object_type": object_type, "summary": summary,
            "canonical_key": canonical_key, "confidence": confidence,
            "primary_domain": primary_domain,
            "evidence_fact_ids": evidence_ids,
            "world_owner_entity_id": world_owner_id,
        }
        def entity_id(value: Any, *, fallback: int = 0) -> int:
            name = _compact_whitespace(value or "")
            if self._is_world_owner_entity_reference(name):
                return world_owner_id if allows_world_owner_reference else 0
            return int(entity_ids.get(name) or fallback)
        if object_type == "goal":
            owner_name = _compact_whitespace(raw.get("subject_entity") or "")
            if self._is_world_owner_entity_reference(owner_name):
                if not allows_world_owner_reference:
                    return None
                owner_name = self._world_owner_entity_name
            elif not owner_name and allows_world_owner_reference:
                owner_name = self._world_owner_entity_name
            owner_id = entity_id(
                owner_name,
                fallback=(world_owner_id if allows_world_owner_reference else 0),
            )
            desired_outcome = _compact_whitespace(details.get("desired_outcome") or "")[:480]
            if not owner_id or not desired_outcome:
                return None
            target_at = _compact_whitespace(
                details.get("target_at") or raw.get("time_end") or ""
            )[:80]
            candidate.update(
                owner_entity_id=owner_id,
                owner_name=owner_name,
                desired_outcome=desired_outcome,
                success_criteria=_compact_whitespace(details.get("success_criteria") or "")[:480],
                target_at=target_at,
                time_start="",
                time_end=target_at,
                details={
                    "desired_outcome": desired_outcome,
                    "success_criteria": _compact_whitespace(details.get("success_criteria") or "")[:480],
                    "target_at": target_at,
                },
            )
        elif object_type == "plan":
            actor_name = _compact_whitespace(raw.get("subject_entity") or "")
            if self._is_world_owner_entity_reference(actor_name):
                if not allows_world_owner_reference:
                    return None
                actor_name = self._world_owner_entity_name
            elif not actor_name and allows_world_owner_reference:
                actor_name = self._world_owner_entity_name
            actor_id = entity_id(
                actor_name,
                fallback=(world_owner_id if allows_world_owner_reference else 0),
            )
            event = _compact_whitespace(details.get("event_or_activity") or "")[:320]
            if not actor_id or not event:
                return None
            location = _compact_whitespace(details.get("location") or "")[:160]
            if location and location not in entity_ids:
                entity_ids.update(self._db.add_entity_names([location]))
            start_at = _compact_whitespace(raw.get("time_start") or "")[:80]
            end_at = _compact_whitespace(raw.get("time_end") or "")[:80]
            candidate.update(actor_entity_id=actor_id, actor_name=actor_name, event_or_activity=event,
                start_at=start_at,
                end_at=end_at,
                time_precision=(
                    str(details.get("time_precision") or "unknown").lower()
                    if str(details.get("time_precision") or "unknown").lower()
                    in {"exact", "day", "week", "relative", "unknown"}
                    else "unknown"
                ),
                location_text=location, location_entity_id=int(entity_ids.get(location) or 0),
                participant_names=self._normalize_string_list(details.get("participants"), limit=12),
                time_start=start_at,
                time_end=end_at,
                details={
                    "event_or_activity": event,
                    "time_precision": (
                        str(details.get("time_precision") or "unknown").lower()
                        if str(details.get("time_precision") or "unknown").lower()
                        in {"exact", "day", "week", "relative", "unknown"}
                        else "unknown"
                    ),
                    "location": location,
                    "participants": self._normalize_string_list(details.get("participants"), limit=12),
                })
        else:
            responsible_name = _compact_whitespace(raw.get("subject_entity") or "")
            if self._is_world_owner_entity_reference(responsible_name):
                if not allows_world_owner_reference:
                    return None
                responsible_name = self._world_owner_entity_name
            responsible_id = entity_id(responsible_name)
            action_text = _compact_whitespace(details.get("action_text") or "")[:480]
            deliverable = _compact_whitespace(details.get("deliverable") or "")[:320]
            responsibility_type = str(details.get("responsibility_type") or "").lower()
            if not responsible_id or not action_text or responsibility_type not in {
                "personal_action", "commitment", "assigned", "external_commitment",
            }:
                return None
            due_at = _compact_whitespace(raw.get("time_end") or "")[:80]
            available_from = _compact_whitespace(raw.get("time_start") or "")[:80]
            has_completion_anchor = bool(deliverable or due_at) or responsibility_type in {
                "commitment", "assigned", "external_commitment",
            }
            if not has_completion_anchor:
                return None
            candidate.update(responsible_entity_id=responsible_id, responsible_name=responsible_name, action_text=action_text,
                deliverable=deliverable, responsibility_type=responsibility_type,
                due_at=due_at,
                start_at=available_from,
                priority=_compact_whitespace(details.get("priority") or "")[:80],
                beneficiary_names=self._normalize_string_list(details.get("beneficiary_entities"), limit=8),
                delegator_names=self._normalize_string_list(details.get("delegator_entities"), limit=8),
                collaborator_names=self._normalize_string_list(details.get("collaborator_entities"), limit=8),
                time_start=available_from,
                time_end=due_at,
                details={
                    "action_text": action_text,
                    "deliverable": deliverable,
                    "responsibility_type": responsibility_type,
                    "priority": _compact_whitespace(details.get("priority") or "")[:80],
                    "beneficiary_entities": self._normalize_string_list(details.get("beneficiary_entities"), limit=8),
                    "delegator_entities": self._normalize_string_list(details.get("delegator_entities"), limit=8),
                    "collaborator_entities": self._normalize_string_list(details.get("collaborator_entities"), limit=8),
                })
        return candidate

    def _reconcile_future_intent_candidates(
        self,
        candidates: Sequence[Dict[str, Any]],
        *,
        world_owner_id: int,
        prompt_language: str,
        existing_by_type: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    ) -> Dict[int, Dict[str, Any]]:
        """Adjudicate candidate persistence, relationships, and existing-intent changes."""
        if not candidates:
            return {}
        if existing_by_type is None:
            existing_by_type = self._retrieve_user_intent_related_intent_objects(
                candidates,
                world_owner_id=world_owner_id,
            )
        existing_by_type = {
            object_type: [
                item for item in existing_by_type.get(object_type) or []
                if int(item.get("world_owner_entity_id") or 0) == world_owner_id
                and int(item.get("id") or 0) > 0
            ]
            for object_type in ("goal", "plan", "work_item")
        }
        decisions: Dict[int, Dict[str, Any]] = {}
        existing = [
            {
                **self._intent_object_prompt_view(object_type, item),
                "time_start": item.get("time_start") or "",
                "time_end": item.get("time_end") or "",
                "details": item.get("details") or {},
            }
            for object_type, items in existing_by_type.items() for item in items
        ]
        facts_by_id = {
            int(fact["id"]): fact
            for fact in self._db.get_memory_facts_by_ids(sorted({
                fact_id for candidate in candidates
                for fact_id in candidate["evidence_fact_ids"]
            }))
        }
        candidate_inputs = [{
            **candidate,
            "facts": [
                self._intent_fact_prompt_view(facts_by_id[fact_id])
                for fact_id in candidate["evidence_fact_ids"] if fact_id in facts_by_id
            ],
        } for candidate in candidates]
        prompt_template = (
            INTENT_RECONCILIATION_PROMPT_EN
            if prompt_language == "en" else INTENT_RECONCILIATION_PROMPT_ZH
        )
        try:
            raw = self._llm_client.call_llm(
                prompt_template.replace("{candidates}", json.dumps(candidate_inputs, ensure_ascii=False, indent=2))
                .replace("{existing_objects}", json.dumps(existing, ensure_ascii=False, indent=2))
            )
            parsed = self._llm_client.parse_json_object_from_llm_text(raw or "")
            if not isinstance(parsed, dict) or not isinstance(parsed.get("decisions"), list):
                raise ValueError("invalid_future_intent_reconciliation_response")
        except Exception as exc:
            self._log_info("memory_user_intent_update", "batch_error", {
                "candidate_indices": list(range(len(candidates))),
                "phase": "future_intent_reconciliation", "error": str(exc),
            })
            return decisions
        existing_by_id = {
            (object_type, int(item["id"])): item
            for object_type, items in existing_by_type.items() for item in items
        }
        allowed_statuses = {
            "goal": {"active", "blocked", "completed", "cancelled"},
            "plan": {"planned", "rescheduled", "blocked", "occurred", "cancelled"},
            "work_item": {"open", "in_progress", "blocked", "completed", "cancelled"},
        }
        detail_fields = {
            "goal": {"desired_outcome", "success_criteria", "target_at"},
            "plan": {"event_or_activity", "time_precision", "location_text", "participants"},
            "work_item": {
                "action_text", "deliverable", "responsibility_type", "priority",
                "beneficiary_entities", "delegator_entities", "collaborator_entities",
                "completed_at",
            },
        }
        results_by_index: Dict[int, List[Dict[str, Any]]] = {}
        for raw_decision in parsed["decisions"]:
            if not isinstance(raw_decision, dict):
                continue
            index = raw_decision.get("candidate_index")
            if type(index) is int and 0 <= index < len(candidates):
                results_by_index.setdefault(index, []).append(raw_decision)
        for index, candidate in enumerate(candidates):
            try:
                results = results_by_index.get(index, [])
                if len(results) != 1:
                    raise ValueError("missing_or_duplicate_future_intent_decision")
                result = results[0]
                if (type(result.get("save_new_intent")) is not bool
                        or not isinstance(result.get("reason"), str)):
                    raise ValueError("invalid_future_intent_decision_fields")
                relations = result.get("relations")
                updates = result.get("intent_updates")
                if (not isinstance(relations, list) or len(relations) > 1
                        or not isinstance(updates, list) or len(updates) > 1):
                    raise ValueError("invalid_future_intent_relations_or_updates")
                object_type = candidate["object_type"]
                normalized_relations = []
                normalized_updates = []
                target_id = None
                if relations:
                    relation = relations[0]
                    if not isinstance(relation, dict):
                        raise ValueError("invalid_future_intent_relation")
                    target_id = relation.get("target_intent_id")
                    if (type(target_id) is not int
                            or (object_type, target_id) not in existing_by_id
                            or relation.get("relation") not in {"same_intent", "supersedes", "conflicts"}):
                        raise ValueError("invalid_future_intent_target_or_relation")
                    normalized_relations.append({
                        "target_intent_id": target_id, "relation": relation["relation"],
                    })
                if updates:
                    update = updates[0]
                    if (not isinstance(update, dict) or target_id is None
                            or type(update.get("target_intent_id")) is not int
                            or update["target_intent_id"] != target_id
                            or update.get("status") not in allowed_statuses[object_type]):
                        raise ValueError("invalid_future_intent_update_target_or_status")
                    patch = update.get("field_updates")
                    if (not isinstance(patch, dict)
                            or set(patch) - {"summary", "time_start", "time_end", "details"}):
                        raise ValueError("invalid_future_intent_field_updates")
                    for field, value in patch.items():
                        if field == "details":
                            if not isinstance(value, dict) or set(value) - detail_fields[object_type]:
                                raise ValueError("invalid_future_intent_detail_updates")
                            for key, detail in value.items():
                                if key in {"participants", "beneficiary_entities", "delegator_entities", "collaborator_entities"}:
                                    if (not isinstance(detail, list) or not detail
                                            or any(not isinstance(name, str) or not name.strip() for name in detail)):
                                        raise ValueError("invalid_future_intent_detail_list")
                                elif not isinstance(detail, str) or not detail.strip():
                                    raise ValueError("invalid_future_intent_detail_value")
                        elif not isinstance(value, str) or not value.strip():
                            raise ValueError("future_intent_patch_must_not_clear_values")
                    if object_type == "goal" and "target_at" in patch.get("details", {}):
                        target_at = patch["details"]["target_at"]
                        if "time_end" in patch and patch["time_end"] != target_at:
                            raise ValueError("conflicting_goal_deadline_fields")
                        patch = {**patch, "time_end": target_at}
                    effective_at = update.get("effective_at")
                    if not isinstance(effective_at, str):
                        raise ValueError("invalid_future_intent_effective_at")
                    effective_at = effective_at.strip()
                    if effective_at:
                        datetime.fromisoformat(effective_at.replace("Z", "+00:00"))
                    normalized_updates.append({
                        "target_intent_id": target_id, "status": update["status"],
                        "field_updates": patch, "effective_at": effective_at,
                    })
                decisions[index] = {
                    "save_new_intent": result["save_new_intent"],
                    "relations": normalized_relations, "intent_updates": normalized_updates,
                    "reason": _compact_whitespace(result["reason"])[:480],
                }
            except Exception as exc:
                self._log_info("memory_user_intent_update", "signal_error", {
                    "signal_id": candidates[index].get("signal_id"), "candidate_index": index,
                    "phase": "future_intent_reconciliation", "error": str(exc),
                })
        # Do not apply contradictory patches to the same object in list order.
        updates_by_target: Dict[int, Dict[str, Any]] = {}
        conflicting_targets = set()
        for decision in decisions.values():
            for update in decision["intent_updates"]:
                fields = {"status": update["status"]}
                for field, value in update["field_updates"].items():
                    if field == "details":
                        fields.update({f"details.{key}": detail for key, detail in value.items()})
                    else:
                        fields[field] = value
                previous = updates_by_target.setdefault(update["target_intent_id"], {})
                if any(key in previous and previous[key] != value for key, value in fields.items()):
                    conflicting_targets.add(update["target_intent_id"])
                previous.update(fields)
        for index in list(decisions):
            if any(update["target_intent_id"] in conflicting_targets
                   for update in decisions[index]["intent_updates"]):
                del decisions[index]
                self._log_info("memory_user_intent_update", "signal_error", {
                    "signal_id": candidates[index].get("signal_id"), "candidate_index": index,
                    "phase": "future_intent_reconciliation", "error": "conflicting_future_intent_updates",
                })
        return decisions

    @staticmethod
    def _intent_object_prompt_view(object_type: str, item: Dict[str, Any]) -> Dict[str, Any]:
        fields = {
            "goal": ("summary", "desired_outcome", "status", "target_at", "canonical_key"),
            "plan": ("summary", "event_or_activity", "status", "start_at", "end_at", "location_text", "canonical_key"),
            "work_item": ("summary", "action_text", "deliverable", "status", "due_at", "responsibility_type", "canonical_key"),
        }[object_type]
        return {
            "object_type": object_type,
            "id": item.get("id"),
            "primary_domain": item.get("primary_domain") or "",
            **{field: item.get(field) or "" for field in fields},
        }

    def _sync_user_intent_recall_document(
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
                f"primary_domain: {item.get('primary_domain') or ''}",
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
                f"primary_domain: {item.get('primary_domain') or ''}",
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
                f"primary_domain: {item.get('primary_domain') or ''}",
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
                "user_intent_object_type": normalized_type,
                "primary_domain": item.get("primary_domain") or "",
                "projection_version": "v1",
            },
        )

    def _retrieve_intent_lifecycle_observation_cases(
        self, observations: Sequence[Dict[str, Any]], *,
        facts_by_id: Dict[int, Dict[str, Any]], world_owner_id: int,
    ) -> List[Dict[str, Any]]:
        """Retrieve same-type identity candidates without filtering intent lifecycle status."""
        existing_by_type = {
            object_type: self._db.get_intent_objects(
                object_type=object_type, world_owner_entity_id=world_owner_id, limit=120,
            )
            for object_type in {item["object_type"] for item in observations}
        }
        cases: List[Dict[str, Any]] = []
        for observation in observations:
            object_type = observation["object_type"]
            identity = {"object_type": object_type, "summary": observation["target_description"]}
            ranked = sorted(
                ((self._score_intent_candidate_against_existing_object(identity, item), item)
                 for item in existing_by_type[object_type]),
                key=lambda pair: pair[0], reverse=True,
            )
            cases.append({
                "intent_lifecycle_observation": {
                    key: observation[key] for key in (
                        "object_type", "target_description", "observation_text",
                        "lifecycle_evidence", "effective_at", "confidence", "details",
                    )
                },
                "fact": self._intent_fact_prompt_view(
                    facts_by_id[observation["evidence_fact_ids"][0]],
                ),
                "existing_intents": [
                    {
                        **self._intent_object_prompt_view(object_type, item),
                        "time_start": item.get("time_start") or "",
                        "time_end": item.get("time_end") or "",
                        "details": item.get("details") or {},
                        "updated_at": item.get("updated_at") or "",
                    }
                    for score, item in ranked[:6] if score >= 0.2
                ],
            })
        return cases

    def _reconcile_intent_lifecycle_observations(
        self, observations: Sequence[Dict[str, Any]], *,
        facts_by_id: Dict[int, Dict[str, Any]], world_owner_id: int, prompt_language: str,
    ) -> Dict[int, Dict[str, Any]]:
        """Match observations and adjudicate sparse intent updates without writing to storage."""
        if not observations:
            return {}
        decisions: Dict[int, Dict[str, Any]] = {}
        cases = self._retrieve_intent_lifecycle_observation_cases(
            observations, facts_by_id=facts_by_id, world_owner_id=world_owner_id,
        )
        matching_indices = {index for index, case in enumerate(cases) if case["existing_intents"]}
        decisions_by_index: Dict[int, List[Dict[str, Any]]] = {}
        if matching_indices:
            self._log_info("memory_user_intent_update", "lifecycle_matching_inputs", {
                "observation_cases": cases,
            })
            try:
                template = (
                    INTENT_LIFECYCLE_RECONCILIATION_PROMPT_EN if prompt_language == "en"
                    else INTENT_LIFECYCLE_RECONCILIATION_PROMPT_ZH
                )
                raw = self._llm_client.call_llm(template.replace(
                    "{observation_cases}", json.dumps(cases, ensure_ascii=False, indent=2),
                ))
                parsed = self._llm_client.parse_json_object_from_llm_text(raw or "")
                if not isinstance(parsed, dict) or not isinstance(parsed.get("decisions"), list):
                    raise ValueError("invalid_lifecycle_matching_response")
                for decision in parsed["decisions"]:
                    if not isinstance(decision, dict):
                        continue
                    index = decision.get("observation_index")
                    if type(index) is int and index in matching_indices:
                        decisions_by_index.setdefault(index, []).append(decision)
            except Exception as exc:
                self._log_info("memory_user_intent_update", "lifecycle_match_error", {
                    "observation_indices": sorted(matching_indices),
                    "error": str(exc),
                })

        allowed_statuses = {
            "goal": {"active", "blocked", "completed", "cancelled"},
            "plan": {"planned", "rescheduled", "blocked", "occurred", "cancelled"},
            "work_item": {"open", "in_progress", "blocked", "completed", "cancelled"},
        }
        detail_fields = {
            "goal": {"desired_outcome", "success_criteria", "target_at"},
            "plan": {"event_or_activity", "time_precision", "location_text", "participants"},
            "work_item": {
                "action_text", "deliverable", "responsibility_type", "priority", "completed_at",
                "beneficiary_entities", "delegator_entities", "collaborator_entities",
            },
        }
        for index, case in enumerate(cases):
            try:
                if not case["existing_intents"]:
                    decision = {"relations": [], "intent_updates": [], "reason": "no_related_intent"}
                else:
                    results = decisions_by_index.get(index, [])
                    if len(results) != 1:
                        raise ValueError("missing_or_duplicate_lifecycle_matching_decision")
                    decision = results[0]
                relations = decision.get("relations")
                updates = decision.get("intent_updates")
                if (not isinstance(relations, list) or len(relations) > 1
                        or not isinstance(updates, list) or len(updates) > 1
                        or not isinstance(decision.get("reason"), str)):
                    raise ValueError("invalid_lifecycle_decision_fields")
                object_type = case["intent_lifecycle_observation"]["object_type"]
                normalized_relations = []
                normalized_updates = []
                target_id = None
                if relations:
                    relation = relations[0]
                    if not isinstance(relation, dict):
                        raise ValueError("invalid_lifecycle_relation")
                    target_id = relation.get("target_intent_id")
                    if (type(target_id) is not int or target_id <= 0
                            or relation.get("relation") != "same_intent"
                            or target_id not in {item["id"] for item in case["existing_intents"]
                                                 if item["object_type"] == object_type}):
                        raise ValueError("invalid_lifecycle_matching_target")
                    normalized_relations.append({
                        "target_intent_id": target_id, "relation": "same_intent",
                    })
                if updates:
                    update = updates[0]
                    if (not isinstance(update, dict) or target_id is None
                            or type(update.get("target_intent_id")) is not int
                            or update["target_intent_id"] != target_id
                            or update.get("status") not in allowed_statuses[object_type]):
                        raise ValueError("invalid_lifecycle_update_target_or_status")
                    patch = update.get("field_updates")
                    if (not isinstance(patch, dict)
                            or set(patch) - {"summary", "time_start", "time_end", "details"}):
                        raise ValueError("invalid_lifecycle_field_updates")
                    for field, value in patch.items():
                        if field == "details":
                            if not isinstance(value, dict) or set(value) - detail_fields[object_type]:
                                raise ValueError("invalid_lifecycle_detail_updates")
                            for key, detail in value.items():
                                if key in {"participants", "beneficiary_entities", "delegator_entities", "collaborator_entities"}:
                                    if (not isinstance(detail, list) or not detail
                                            or any(not isinstance(name, str) or not name.strip() for name in detail)):
                                        raise ValueError("invalid_lifecycle_detail_list")
                                elif not isinstance(detail, str) or not detail.strip():
                                    raise ValueError("invalid_lifecycle_detail_value")
                        elif not isinstance(value, str) or not value.strip():
                            raise ValueError("lifecycle_patch_must_not_clear_values")
                    if object_type == "goal" and "target_at" in patch.get("details", {}):
                        target_at = patch["details"]["target_at"]
                        if "time_end" in patch and patch["time_end"] != target_at:
                            raise ValueError("conflicting_goal_deadline_fields")
                        patch = {**patch, "time_end": target_at}
                    effective_at = update.get("effective_at")
                    if not isinstance(effective_at, str):
                        raise ValueError("invalid_lifecycle_effective_at")
                    effective_at = effective_at.strip()
                    if effective_at:
                        datetime.fromisoformat(effective_at.replace("Z", "+00:00"))
                    normalized_updates.append({
                        "target_intent_id": target_id, "status": update["status"],
                        "field_updates": patch, "effective_at": effective_at,
                    })
                decisions[index] = {
                    "relations": normalized_relations, "intent_updates": normalized_updates,
                    "reason": _compact_whitespace(decision["reason"])[:480],
                }
            except Exception as exc:
                self._log_info("memory_user_intent_update", "signal_error", {
                    "observation_index": index, "phase": "lifecycle_reconciliation", "error": str(exc),
                })
        # Do not let incompatible batch updates overwrite one another in list order.
        updates_by_target: Dict[int, Dict[str, Any]] = {}
        conflicting_targets = set()
        for decision in decisions.values():
            for update in decision["intent_updates"]:
                fields = {"status": update["status"]}
                for field, value in update["field_updates"].items():
                    if field == "details":
                        fields.update({f"details.{key}": detail for key, detail in value.items()})
                    else:
                        fields[field] = value
                previous = updates_by_target.setdefault(update["target_intent_id"], {})
                if any(key in previous and previous[key] != value for key, value in fields.items()):
                    conflicting_targets.add(update["target_intent_id"])
                previous.update(fields)
        for index in list(decisions):
            if any(update["target_intent_id"] in conflicting_targets
                   for update in decisions[index]["intent_updates"]):
                del decisions[index]
                self._log_info("memory_user_intent_update", "signal_error", {
                    "observation_index": index, "phase": "lifecycle_reconciliation",
                    "error": "conflicting_lifecycle_updates",
                })
        return decisions

    def _save_intent_lifecycle_observations(
        self, observations: Sequence[Dict[str, Any]], decisions: Dict[int, Dict[str, Any]], *,
        world_owner_id: int,
    ) -> Dict[str, Any]:
        """Atomically save each observation, its adjudicated intent changes, and fact evidence."""
        report: Dict[str, Any] = {
            "intent_lifecycle_observation_ids": [],
            "intent_event_ids": [],
            "unmatched_observation_ids": [], "completed_signal_ids": [], "failed_signal_ids": [],
        }
        for index, observation in enumerate(observations):
            signal_id = observation["signal_id"]
            try:
                recorded_events = self._db.get_user_intent_events(source_signal_id=signal_id)
                if recorded_events:
                    report["intent_event_ids"].extend(event["id"] for event in recorded_events)
                    report["intent_lifecycle_observation_ids"].extend(
                        sorted({event["lifecycle_observation_id"] for event in recorded_events
                                if event["lifecycle_observation_id"] is not None})
                    )
                    report["completed_signal_ids"].append(signal_id)
                    continue
                decision = decisions.get(index)
                if decision is None:
                    raise ValueError("missing_valid_lifecycle_matching_decision")
                relation = decision["relations"][0] if decision["relations"] else None
                update = decision["intent_updates"][0] if decision["intent_updates"] else None
                event_id = None
                with self._db.transaction():
                    observation_id = self._db.upsert_user_intent_lifecycle_observation(observation)
                    if relation:
                        object_type = observation["object_type"]
                        target_id = relation["target_intent_id"]
                        target = self._db.get_intent_object(object_type=object_type, object_id=target_id)
                        if (not target
                                or target["world_owner_entity_id"] != world_owner_id):
                            raise ValueError("invalid_lifecycle_matching_target")
                        if update:
                            patch = update["field_updates"]
                            # Merge before calling the typed DB writer so sparse
                            # lifecycle patches cannot erase unrelated columns.
                            payload = {key: target[key] for key in (
                                "world_owner_entity_id", "canonical_key", "summary", "primary_domain",
                                "time_start", "time_end", "confidence", "metadata",
                            )}
                            entity_field = {
                                "goal": "owner_entity_id", "plan": "actor_entity_id",
                                "work_item": "responsible_entity_id",
                            }[object_type]
                            payload[entity_field] = target[entity_field]
                            payload.update({key: value for key, value in patch.items() if key != "details"})
                            payload["details"] = {**target["details"], **patch.get("details", {})}
                            if object_type == "goal":
                                if "time_end" in patch:
                                    payload["details"]["target_at"] = patch["time_end"]
                                elif "target_at" in patch.get("details", {}):
                                    payload["time_end"] = patch["details"]["target_at"]
                            payload["status"] = update["status"]
                            if any(payload[key] != target.get(key) for key in payload):
                                if not self._db.update_intent_object(
                                    object_type=object_type, object_id=target_id, payload=payload,
                                ):
                                    raise ValueError("lifecycle_intent_update_failed")
                                updated_target = self._db.get_intent_object(
                                    object_type=object_type, object_id=target_id,
                                )
                                if not updated_target:
                                    raise ValueError("updated_lifecycle_intent_missing")
                                event_id = self._db.insert_user_intent_event(
                                    intent_id=target_id,
                                    event_type="updated",
                                    previous_payload=target,
                                    new_payload=updated_target,
                                    source_signal_id=signal_id,
                                    lifecycle_observation_id=observation_id,
                                    effective_at=update["effective_at"],
                                )
                        facts = self._db.get_memory_facts_by_ids(observation["evidence_fact_ids"])
                        if len(facts) != len(observation["evidence_fact_ids"]):
                            raise ValueError("missing_lifecycle_evidence_fact")
                        self._db.upsert_intent_evidence([{
                            "object_type": object_type, "object_id": target_id,
                            "evidence_type": "fact", "evidence_id": fact["id"],
                            "role": observation["lifecycle_evidence"],
                            "observed_at": str(fact.get("dialogue_time_key") or fact.get("event_time_key") or ""),
                        } for fact in facts])
            except Exception as exc:
                report["failed_signal_ids"].append(signal_id)
                self._log_info("memory_user_intent_update", "signal_error", {
                    "signal_id": signal_id, "phase": "lifecycle_observation_save", "error": str(exc),
                })
                continue
            report["intent_lifecycle_observation_ids"].append(observation_id)
            report["completed_signal_ids"].append(signal_id)
            if event_id is not None:
                report["intent_event_ids"].append(event_id)
            if not relation:
                report["unmatched_observation_ids"].append(observation_id)
            self._log_info("memory_user_intent_update", "lifecycle_relation", {
                **decision, "signal_id": signal_id,
                "observation_id": observation_id, "intent_event_id": event_id,
            })
        return report

    def _save_future_intent_candidates(
        self,
        candidates: Sequence[Dict[str, Any]],
        decisions: Dict[int, Dict[str, Any]],
        *,
        facts_by_id: Dict[int, Dict[str, Any]],
        reference_timestamp: Any,
    ) -> Dict[str, Any]:
        """Atomically save new intents, existing-intent changes, and their fact evidence."""
        report: Dict[str, Any] = {
            "intent_event_ids": [],
            "completed_signal_ids": [],
            "failed_signal_ids": [],
        }
        for index, candidate in enumerate(candidates):
            signal_id = candidate["signal_id"]
            try:
                recorded_events = self._db.get_user_intent_events(source_signal_id=signal_id)
                if recorded_events:
                    report["intent_event_ids"].extend(event["id"] for event in recorded_events)
                    report["completed_signal_ids"].append(signal_id)
                    continue
                signal_event_ids: List[int] = []
                fact = facts_by_id[candidate["evidence_fact_ids"][0]]
                observed_at = _to_timestamp_text(
                    fact.get("dialogue_time_key")
                    or fact.get("event_time_key")
                    or reference_timestamp
                )
                decision = decisions.get(index)
                if decision is None:
                    raise ValueError("missing_valid_future_intent_decision")
                object_type = candidate["object_type"]
                relation = decision["relations"][0] if decision["relations"] else None
                update = decision["intent_updates"][0] if decision["intent_updates"] else None
                target_id = relation["target_intent_id"] if relation else 0
                with self._db.transaction():
                    target = self._db.get_intent_object(
                        object_type=object_type, object_id=target_id,
                    ) if target_id else None
                    if relation and (
                        not target or target["world_owner_entity_id"] != candidate["world_owner_entity_id"]
                    ):
                        raise ValueError("future_intent_target_missing_or_owner_mismatch")
                    if decision["save_new_intent"]:
                        status = {
                            "goal": "active", "plan": "planned", "work_item": "open",
                        }[object_type]
                        object_id = self._db.create_intent_object(
                            object_type=object_type,
                            payload=self._intent_storage_payload(candidate, status=status),
                        )
                        stored_object = self._db.get_intent_object(
                            object_type=object_type, object_id=object_id,
                        )
                        if not stored_object:
                            raise ValueError("created_future_intent_missing")
                        self._db.upsert_intent_evidence([{
                            "object_type": object_type, "object_id": object_id,
                            "evidence_type": "fact", "evidence_id": fact_id,
                            "role": "creation", "observed_at": observed_at,
                        } for fact_id in candidate["evidence_fact_ids"]])
                        signal_event_ids.append(self._db.insert_user_intent_event(
                            intent_id=object_id,
                            event_type="created",
                            new_payload=stored_object,
                            source_signal_id=signal_id,
                        ))
                    if target:
                        if update:
                            # The DB accepts typed full payloads. Merge the sparse patch
                            # first so unspecified columns and details cannot be erased.
                            patch = update["field_updates"]
                            payload = {key: target[key] for key in (
                                "world_owner_entity_id", "canonical_key", "summary", "primary_domain",
                                "time_start", "time_end", "confidence", "metadata",
                            )}
                            entity_field = {
                                "goal": "owner_entity_id", "plan": "actor_entity_id",
                                "work_item": "responsible_entity_id",
                            }[object_type]
                            payload[entity_field] = target[entity_field]
                            payload.update({key: value for key, value in patch.items() if key != "details"})
                            payload["details"] = {**target["details"], **patch.get("details", {})}
                            if object_type == "goal":
                                if "time_end" in patch:
                                    payload["details"]["target_at"] = patch["time_end"]
                                elif "target_at" in patch.get("details", {}):
                                    payload["time_end"] = patch["details"]["target_at"]
                            payload["status"] = update["status"]
                            if any(payload[key] != target.get(key) for key in payload):
                                if not self._db.update_intent_object(
                                    object_type=object_type, object_id=target_id, payload=payload,
                                ):
                                    raise ValueError("future_intent_update_failed")
                                updated_target = self._db.get_intent_object(
                                    object_type=object_type, object_id=target_id,
                                )
                                if not updated_target:
                                    raise ValueError("updated_future_intent_missing")
                                signal_event_ids.append(self._db.insert_user_intent_event(
                                    intent_id=target_id,
                                    event_type="updated",
                                    previous_payload=target,
                                    new_payload=updated_target,
                                    source_signal_id=signal_id,
                                    effective_at=update["effective_at"],
                                ))
                        self._db.upsert_intent_evidence([{
                            "object_type": object_type, "object_id": target_id,
                            "evidence_type": "fact", "evidence_id": fact_id,
                            "role": relation["relation"], "observed_at": observed_at,
                        } for fact_id in candidate["evidence_fact_ids"]])
            except Exception as exc:
                report["failed_signal_ids"].append(signal_id)
                self._log_info("memory_user_intent_update", "signal_error", {
                    "signal_id": signal_id, "error": str(exc),
                })
                continue
            report["completed_signal_ids"].append(signal_id)
            report["intent_event_ids"].extend(signal_event_ids)
            self._log_info("memory_user_intent_update", "future_intent_saved", {
                "candidate_index": index, "signal_id": signal_id,
                "intent_event_ids": signal_event_ids, **decision,
            })
        return report

    def _intent_storage_payload(self, candidate: Dict[str, Any], *, status: str) -> Dict[str, Any]:
        payload = {key: value for key, value in candidate.items() if key in {
            "world_owner_entity_id", "canonical_key", "summary", "owner_entity_id",
            "desired_outcome", "success_criteria", "target_at", "actor_entity_id",
            "event_or_activity", "start_at", "end_at", "time_precision", "location_entity_id",
            "location_text", "responsible_entity_id", "action_text", "deliverable",
            "responsibility_type", "due_at", "priority", "primary_domain",
        }}
        object_type = str(candidate.get("object_type") or "")
        if object_type == "goal":
            payload["time_start"] = ""
            payload["time_end"] = candidate.get("target_at") or ""
            payload["details"] = {
                "desired_outcome": candidate.get("desired_outcome") or "",
                "success_criteria": candidate.get("success_criteria") or "",
                "target_at": candidate.get("target_at") or "",
            }
        elif object_type == "plan":
            payload["time_start"] = candidate.get("start_at") or ""
            payload["time_end"] = candidate.get("end_at") or ""
            payload["details"] = {
                "event_or_activity": candidate.get("event_or_activity") or "",
                "time_precision": candidate.get("time_precision") or "unknown",
                "location_entity_id": candidate.get("location_entity_id") or 0,
                "location_text": candidate.get("location_text") or "",
                "participants": candidate.get("participant_names") or [],
            }
        elif object_type == "work_item":
            payload["time_start"] = candidate.get("start_at") or ""
            payload["time_end"] = candidate.get("due_at") or ""
            payload["details"] = {
                "action_text": candidate.get("action_text") or "",
                "deliverable": candidate.get("deliverable") or "",
                "responsibility_type": candidate.get("responsibility_type") or "personal_action",
                "priority": candidate.get("priority") or "",
                "beneficiary_entities": candidate.get("beneficiary_names") or [],
                "delegator_entities": candidate.get("delegator_names") or [],
                "collaborator_entities": candidate.get("collaborator_names") or [],
                "extractor_version": "user_intent_update_v1",
                "prompt_version": "v1",
            }
        payload.update(status=status, confidence=candidate["confidence"], metadata={
            "source_fact_ids": candidate["evidence_fact_ids"],
            "primary_domain": candidate["primary_domain"],
        })
        return payload

    @staticmethod
    def _user_claim_types() -> set[str]:
        return {
            "identity_profile", "affiliation", "relationship", "preference",
            "constraint", "behavior_pattern",
        }

    def _update_memory_user_claims(
        self,
        *,
        signal_ids: Sequence[int],
        reference_timestamp: Any,
    ) -> Dict[str, Any]:
        """Review selected claim signals against their facts and persist claims."""
        signals = self._db.get_user_signals_by_ids(
            "claim", signal_ids,
        )
        explicit_facts = self._db.get_memory_facts_by_ids(sorted({
            int(signal["fact_id"]) for signal in signals
        }))
        facts_by_id = {int(fact["id"]): fact for fact in explicit_facts}
        signals_by_id = {
            int(item["signal_id"]): item
            for fact in explicit_facts
            for item in self._get_user_claim_signals_from_fact(fact)
        }
        signal_items = [
            {"signal_id": int(row["signal_id"]), "fact": facts_by_id[int(row["fact_id"])],
             "signal": signals_by_id[int(row["signal_id"])]}
            for row in signals
            if int(row["fact_id"]) in facts_by_id
            and int(row["signal_id"]) in signals_by_id
        ]
        # Commit explicit results independently: a later derivation failure must
        # neither roll them back nor make their source signals pending again.
        with self._db.transaction():
            explicit_report = self._update_explicit_user_claims_from_signals(signal_items)
        missing_signal_count = len(signals) - len(signal_items)
        if missing_signal_count:
            explicit_report["failed_signal_count"] += missing_signal_count
            explicit_report["completed"] = False
        explicit_report.pop("completed", None)
        processed_signal_ids = explicit_report.pop("processed_signal_ids", [])
        new_claim_ids = list(explicit_report.get("new_claim_ids") or [])
        derived_report = self._update_derived_user_claims_from_explicit_claims(
            new_explicit_claim_ids=new_claim_ids,
        )
        all_new_claim_ids = sorted(
            set(new_claim_ids) | set(derived_report.get("new_claim_ids") or [])
        )
        all_claim_relations = {
            int(relation["relation_event_id"]): relation
            for source_report in (explicit_report, derived_report)
            for relation in source_report.get("claim_relations") or []
            if int(relation.get("relation_event_id") or 0) > 0
        }
        return {
            "explicit": explicit_report,
            "derived": derived_report,
            "new_claim_ids": all_new_claim_ids,
            "claim_relations": [
                all_claim_relations[event_id] for event_id in sorted(all_claim_relations)
            ],
            "_completed_signal_ids": processed_signal_ids,
        }

    def _explicit_claim_fact_prompt_view(self, fact: Dict[str, Any]) -> Dict[str, Any]:
        metadata = fact.get("metadata") if isinstance(fact.get("metadata"), dict) else {}
        entities = list(fact.get("entities") or [])
        has_world_owner = bool(
            (metadata.get("has_world_owner_speaker") or metadata.get("has_world_owner_alias_reference"))
            and any(
                _compact_whitespace(entity).lower()
                == self._world_owner_entity_name.lower()
                for entity in entities
            )
        )
        prompt_view = {
            "fact_id": fact.get("id"),
            "summary": fact.get("summary") or "",
            "fact_type": fact.get("fact_type") or "",
            "entities": entities,
            "world_owner_entity": (
                self._world_owner_entity_name if has_world_owner else ""
            ),
            "primary_entity": fact.get("primary_entity") or metadata.get("primary_entity"),
            "keywords": fact.get("keywords") or [],
            "event_time": fact.get("event_time_key") or "",
            "dialogue_time": fact.get("dialogue_time_key") or "",
        }
        return prompt_view

    def _explicit_claim_signal_prompt_view(self, item: Dict[str, Any]) -> Dict[str, Any]:
        signal = item["signal"]
        return {
            "signal_id": item["signal_id"],
            "user_claim_signal": {
                "subject_entity": _compact_whitespace(signal.get("subject") or ""),
                "claim_type_hint": str(signal.get("claim_type_hint") or ""),
                "claim_anchor": str(signal.get("claim_anchor") or ""),
                "evidence_basis": str(signal.get("evidence_basis") or ""),
                "confidence": float(signal.get("confidence") or 0.0),
            },
            "fact": self._explicit_claim_fact_prompt_view(item["fact"]),
        }

    def _sync_user_claim_recall_documents(
        self,
        claim_ids: Sequence[int],
    ) -> None:
        """Refresh projections for claims created, merged, or transitioned."""
        claims = self._db.get_user_claims_by_ids(claim_ids)
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
                object_type="user_claim",
                object_id=int(claim["id"]),
                title=_compact_whitespace(claim.get("claim_text") or "")[:120],
                summary=_compact_whitespace(claim.get("claim_text") or ""),
                identity_lines=[
                    f"subject: {names_by_id.get(subject_id, '')}",
                    f"claim: {_compact_whitespace(claim.get('claim_text') or '')}",
                    f"predicate: {claim.get('predicate') or ''}",
                    f"object: {names_by_id.get(object_id, '')}",
                    f"claim_type: {claim.get('claim_type') or ''}",
                    f"primary_domain: {claim.get('primary_domain') or ''}",
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
                    "primary_domain": str(claim.get("primary_domain") or ""),
                    "predicate": str(claim.get("predicate") or ""),
                    "projection_version": "v1",
                },
            )

    @staticmethod
    def _user_claim_origin_priority(origin: Any) -> int:
        """Return a deterministic ordering hint for claim-relation retrieval."""
        return {"explicit": 2, "derived": 1}.get(
            str(origin or "").strip().lower(),
            0,
        )

    @staticmethod
    def _user_claim_storage_payload(candidate: Dict[str, Any]) -> Dict[str, Any]:
        return {
            key: value
            for key, value in candidate.items()
            if key not in {
                "evidence_fact_ids", "support_fact_ids", "premise_claim_ids",
            }
        }

    @staticmethod
    def _get_user_claim_support_fact_ids(
        candidate: Dict[str, Any],
    ) -> List[int]:
        support_ids = candidate.get("support_fact_ids")
        if support_ids is None:
            support_ids = candidate.get("evidence_fact_ids") or []
        return [int(value) for value in support_ids if str(value).strip().isdigit()]

    def _retrieve_related_user_claims(
        self,
        candidate: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Bound relation candidates to one entity and compatible claim type."""
        rows = self._db.get_user_claims(
            subject_entity_id=int(candidate["subject_entity_id"]),
            claim_type=str(candidate["claim_type"]),
            statuses=["active", "candidate", "weakened", "superseded"],
            limit=80,
        )
        predicate = str(candidate.get("predicate") or "")
        object_entity_id = int(candidate.get("object_entity_id") or 0)
        rows.sort(
            key=lambda row: (
                str(row.get("predicate") or "") != predicate,
                int(row.get("object_entity_id") or 0) != object_entity_id,
                -self._user_claim_origin_priority(row.get("claim_origin")),
                -float(row.get("confidence") or 0.0),
            )
        )
        return rows[:24]

    def _finalize_user_claim_relation_decision(
        self,
        candidate: Dict[str, Any],
        target: Optional[Dict[str, Any]],
        semantic_decision: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Translate text-only claim semantics into audit/evidence actions.

        This layer deliberately does not decide which claim is currently true.
        The UserStateResolver applies origin, freshness, and context weights at
        read time; reconciliation only records the semantic relation.
        """
        semantic_relation = str(
            semantic_decision.get("semantic_relation")
            or semantic_decision.get("relation")
            or "unrelated"
        )
        decision = {
            **semantic_decision,
            "semantic_relation": semantic_relation,
            "target_evidence_role": "",
        }
        if not target:
            return decision
        if semantic_relation in {"duplicate", "supports", "refines"}:
            decision["target_evidence_role"] = "support"
        elif semantic_relation in {"contradicts", "supersedes"}:
            decision["target_evidence_role"] = "counterexample"
        return decision

    def _classify_user_claim_relations(
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
            USER_CLAIM_RECONCILIATION_PROMPT_EN
            if language == "en" else USER_CLAIM_RECONCILIATION_PROMPT_ZH
        )
        candidate_claim_groups = [
            {
                "candidate_claim_index": index,
                "candidate_claim": {
                    "claim_text": candidate.get("claim_text") or "",
                },
                "related_existing_claims": [
                    {
                        "id": claim.get("id"),
                        "claim_text": claim.get("claim_text") or "",
                    }
                    for claim in related_by_index.get(index, [])
                ],
            }
            for index, candidate in enumerate(candidates)
        ]
        raw = self._llm_client.call_llm(
            prompt_template.replace("{candidate_claim_groups}", json.dumps(
                candidate_claim_groups,
                ensure_ascii=False, indent=2,
            ))
        )
        parsed = self._llm_client.parse_json_object_from_llm_text(raw or "")
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
                        self._finalize_user_claim_relation_decision(
                            candidate, target, semantic_decision,
                        )
                    )
            finalized.append(candidate_decisions)
        return finalized

    def _write_user_claim_evidence(
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
        self._db.upsert_user_claim_evidence(evidence)

    @staticmethod
    def _user_claim_transition_effective_at(
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

    def _record_user_claim_relation(
        self,
        *,
        target: Dict[str, Any],
        trigger_claim_id: int,
        candidate: Dict[str, Any],
        decision: Dict[str, Any],
        effective_at: str,
    ) -> Optional[int]:
        """Persist semantic claim history without resolving effective state."""
        details = {
            "trigger_claim_snapshot": {
                "claim_text": candidate.get("claim_text") or "",
                "claim_origin": candidate.get("claim_origin") or "",
                "claim_type": candidate.get("claim_type") or "",
                "confidence": candidate.get("confidence"),
            },
            "target_claim_snapshot": {
                "claim_text": target.get("claim_text") or "",
                "claim_origin": target.get("claim_origin") or "",
                "claim_type": target.get("claim_type") or "",
                "confidence": target.get("confidence"),
            },
        }
        return self._db.record_user_claim_relation(
            target_claim_id=int(target["id"]),
            trigger_claim_id=trigger_claim_id,
            semantic_relation=str(decision.get("semantic_relation") or ""),
            effective_at=effective_at,
            decision_source="user_claim_reconciliation_v2:llm_semantics",
            semantic_confidence=self._clamp_float(
                decision.get("confidence"), 0.0, 1.0, 0.0,
            ),
            semantic_reason=_compact_whitespace(decision.get("reason") or "")[:240],
            details=details,
        )

    def _reconcile_user_claims(
        self,
        candidates: Sequence[Dict[str, Any]],
        *,
        facts_by_id: Dict[int, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Resolve new candidate claims before they become durable claims."""
        candidate_subject_ids = [
            candidate.get("subject_entity_id") for candidate in candidates or []
        ]
        subject_names = self._db.get_entity_names_by_ids(candidate_subject_ids)
        unique_candidates: List[Dict[str, Any]] = []
        seen: set[Tuple[Any, ...]] = set()
        for candidate in candidates:
            subject_id = int(candidate.get("subject_entity_id") or 0)
            if _compact_whitespace(subject_names.get(subject_id) or "").lower() != (
                self._world_owner_entity_name.lower()
            ):
                continue
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
            index: self._retrieve_related_user_claims(candidate)
            for index, candidate in enumerate(unique_candidates)
        }
        decisions = self._classify_user_claim_relations(
            unique_candidates, related_by_index,
        )
        applied: List[Dict[str, Any]] = []
        for index, candidate in enumerate(unique_candidates):
            candidate_decisions = decisions[index]
            candidate_origin = str(candidate.get("claim_origin") or "")
            support_ids = self._get_user_claim_support_fact_ids(candidate)
            effective_at = self._user_claim_transition_effective_at(
                candidate,
                support_ids=support_ids,
                facts_by_id=facts_by_id,
            )

            def target_for(decision: Dict[str, Any]) -> Optional[Dict[str, Any]]:
                target_id = decision.get("existing_claim_id")
                return next(
                    (
                        item for item in related_by_index[index]
                        if int(item.get("id") or 0) == int(target_id or 0)
                    ),
                    None,
                )

            storage_payload = self._user_claim_storage_payload(candidate)
            claim_id, created = self._db.upsert_user_claim(**storage_payload)
            self._write_user_claim_evidence(
                claim_id=claim_id,
                support_ids=support_ids,
                facts_by_id=facts_by_id,
                confidence=float(candidate.get("confidence") or 0.0),
            )
            persisted_relations: List[Dict[str, Any]] = []
            for decision in candidate_decisions:
                target = target_for(decision)
                if not target:
                    continue
                target_evidence_role = str(
                    decision.get("target_evidence_role") or ""
                )
                relation_event_id = self._record_user_claim_relation(
                    target=target,
                    trigger_claim_id=claim_id,
                    candidate=candidate,
                    decision=decision,
                    effective_at=effective_at,
                )
                if relation_event_id:
                    persisted_relations.append({
                        **decision,
                        "relation_event_id": int(relation_event_id),
                    })
                if target_evidence_role:
                    self._write_user_claim_evidence(
                        claim_id=int(target["id"]),
                        support_ids=support_ids,
                        facts_by_id=facts_by_id,
                        confidence=float(candidate.get("confidence") or 0.0),
                        support_role=target_evidence_role,
                    )
            applied.append({
                "claim": candidate, "claim_id": claim_id, "created": created,
                "effective_origin": candidate_origin,
                "relations": persisted_relations,
                "merged": False,
            })
        self._log_info("memory_user_claim", "user_claim_reconciled", {
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
        self._sync_user_claim_recall_documents(
            sorted(recall_document_claim_ids)
        )
        return applied

    def _update_explicit_user_claims_from_signals(
        self,
        signal_items: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Review each signal against its own fact before reconciling claims."""
        report: Dict[str, Any] = {
            "enabled": 1,
            "signal_count": len(signal_items),
            "fact_count": len({int(item["fact"]["id"]) for item in signal_items}),
            "signal_batch_count": 0,
            "failed_signal_count": 0,
            "claim_count": 0,
            "updated": 0,
            "created": 0,
            "merged": 0,
            "new_claim_ids": [],
            "claim_relations": [],
            "processed_signal_ids": [],
            "completed": True,
        }
        if not signal_items:
            return report
        extraction_claim_info = self._extract_explicit_user_claims_from_signals_using_llm(signal_items)
        report.update({
            "signal_batch_count": extraction_claim_info["signal_batch_count"],
            "failed_signal_count": extraction_claim_info["failed_signal_count"],
            "claim_count": extraction_claim_info["claim_count"],
            "processed_signal_ids": extraction_claim_info["processed_signal_ids"],
        })
        claim_candidates = extraction_claim_info["claim_candidates"]
        facts_by_id = {int(item["fact"]["id"]): item["fact"] for item in signal_items}
        applied = self._reconcile_user_claims(
            claim_candidates, facts_by_id=facts_by_id,
        ) if claim_candidates else []
        report["updated"] = len(applied)
        report["created"] = sum(int(item["created"]) for item in applied)
        report["merged"] = sum(int(item["merged"]) for item in applied)
        new_claim_ids: set[int] = set()
        claim_relations: Dict[Tuple[int, int, str], Dict[str, Any]] = {}
        for applied_item in applied:
            claim_id = int(applied_item["claim_id"])
            new_claim_ids.add(claim_id)
            for decision in applied_item.get("relations") or []:
                existing_claim_id = int(decision.get("existing_claim_id") or 0)
                if existing_claim_id <= 0 or existing_claim_id == claim_id:
                    continue
                semantic_relation = str(decision.get("semantic_relation") or "")
                key = (claim_id, existing_claim_id, semantic_relation)
                claim_relations[key] = {
                    "relation_event_id": int(decision["relation_event_id"]),
                    "new_claim_id": claim_id,
                    "existing_claim_id": existing_claim_id,
                    "semantic_relation": semantic_relation,
                    "confidence": float(decision.get("confidence") or 0.0),
                    "reason": str(decision.get("reason") or ""),
                }
        report["new_claim_ids"] = sorted(new_claim_ids)
        report["claim_relations"] = [
            claim_relations[key] for key in sorted(claim_relations)
        ]
        report["completed"] = report["failed_signal_count"] == 0
        return report

    def _extract_explicit_user_claims_from_signals_using_llm(
        self,
        signal_items: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Extract all signal batches and merge duplicate claims across batches."""
        result: Dict[str, Any] = {
            "signal_batch_count": 0,
            "failed_signal_count": 0,
            "claim_count": 0,
            "processed_signal_ids": [],
            "claim_candidates": [],
        }
        unique_claim_candidates: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
        batch_size = max(
            1, int(self._memory_cfg.get("user_claim_signal_batch_size") or 8),
        )
        for batch_start in range(0, len(signal_items), batch_size):
            batch_signal_items = signal_items[batch_start:batch_start + batch_size]
            result["signal_batch_count"] += 1
            prompt_language = self._resolve_prompt_language_from_text(
                "\n".join(str(item["fact"].get("summary") or "") for item in batch_signal_items)
            )
            prompt_template = (
                EXPLICIT_USER_CLAIM_EXTRACTION_PROMPT_EN
                if prompt_language == "en" else EXPLICIT_USER_CLAIM_EXTRACTION_PROMPT_ZH
            )
            try:
                raw = self._llm_client.call_llm(prompt_template.replace(
                    "{signal_items}", json.dumps(
                        [self._explicit_claim_signal_prompt_view(item) for item in batch_signal_items],
                        ensure_ascii=False, indent=2,
                    ),
                ))
                parsed = self._llm_client.parse_json_object_from_llm_text(raw or "")
            except Exception as exc:
                parsed = None
                self._log_info("memory_user_claim", "signal_batch_error", {
                    "signal_ids": [item["signal_id"] for item in batch_signal_items],
                    "error": str(exc),
                })
            if not isinstance(parsed, dict) or not isinstance(parsed.get("results"), list):
                result["failed_signal_count"] += len(batch_signal_items)
                continue

            results_by_id: Dict[int, List[Dict[str, Any]]] = {}
            for raw_result in parsed["results"]:
                if not isinstance(raw_result, dict):
                    continue
                try:
                    signal_id = int(raw_result.get("signal_id"))
                except (TypeError, ValueError):
                    continue
                results_by_id.setdefault(signal_id, []).append(raw_result)

            for item in batch_signal_items:
                signal_id = int(item["signal_id"])
                results = results_by_id.get(signal_id) or []
                if len(results) != 1 or not isinstance(results[0].get("claims"), list):
                    result["failed_signal_count"] += 1
                    continue
                fact = item["fact"]
                fact_id = int(fact["id"])
                entity_ids_by_name = {
                    _compact_whitespace(name): int(entity_id)
                    for name, entity_id in zip(
                        fact.get("entities") or [], fact.get("entity_ids") or [],
                    )
                    if _compact_whitespace(name) and str(entity_id).isdigit()
                    and int(entity_id) > 0
                }
                signal_candidates: List[Dict[str, Any]] = []
                valid = True
                for raw_claim in results[0]["claims"]:
                    if (
                        not isinstance(raw_claim, dict)
                        or raw_claim.get("evidence_fact_ids") != [fact_id]
                    ):
                        valid = False
                        break
                    claim = self._normalize_explicit_user_claim(
                        raw_claim,
                        facts_by_id={fact_id: fact},
                        entity_ids=entity_ids_by_name,
                    )
                    if claim is None:
                        valid = False
                        break
                    signal_candidates.append(claim)
                if not valid:
                    result["failed_signal_count"] += 1
                    continue
                result["claim_count"] += len(signal_candidates)
                result["processed_signal_ids"].append(signal_id)

                # Merge equivalent claims across every batch before reconciliation.
                for candidate in signal_candidates:
                    key = (
                        candidate["subject_entity_id"], candidate["claim_type"],
                        candidate["predicate"], candidate.get("object_entity_id") or 0,
                        candidate["claim_text"], candidate["claim_origin"],
                    )
                    existing = unique_claim_candidates.get(key)
                    if existing is None:
                        unique_claim_candidates[key] = candidate
                        continue
                    existing["evidence_fact_ids"] = list(dict.fromkeys(
                        [*existing["evidence_fact_ids"], *candidate["evidence_fact_ids"]]
                    ))
                    existing["metadata"]["source_fact_count"] = len(existing["evidence_fact_ids"])
                    existing["confidence"] = max(existing["confidence"], candidate["confidence"])
                    if candidate["status"] == "active":
                        existing["status"] = "active"
        result["claim_candidates"] = list(unique_claim_candidates.values())
        return result

    def _normalize_explicit_user_claim(
        self,
        raw: Any,
        *,
        facts_by_id: Dict[int, Dict[str, Any]],
        entity_ids: Dict[str, int],
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(raw, dict):
            return None
        claim_type = str(raw.get("claim_type") or "").strip().lower()
        if claim_type not in self._user_claim_types():
            return None
        evidence_ids = list(dict.fromkeys(
            int(value) for value in (raw.get("evidence_fact_ids") or [])
            if str(value).strip().isdigit() and int(value) in facts_by_id
        ))[:12]
        if not evidence_ids:
            return None
        subject = _compact_whitespace(raw.get("subject_entity") or "")
        if self._is_world_owner_entity_reference(subject):
            subject = self._world_owner_entity_name
        if subject.lower() != self._world_owner_entity_name.lower():
            return None
        # A claim can only be grounded in facts whose source metadata
        # explicitly identifies the wearer, and whose normalized entities
        # retain that exact subject.
        if not all(
            isinstance(facts_by_id[fact_id].get("metadata"), dict)
            and bool(
                facts_by_id[fact_id]["metadata"].get("has_world_owner_speaker")
                or facts_by_id[fact_id]["metadata"].get("has_world_owner_alias_reference")
            )
            and subject in (facts_by_id[fact_id].get("entities") or [])
            for fact_id in evidence_ids
        ):
            return None
        subject_id = entity_ids.get(subject)
        primary_domain = self._normalize_primary_domain(raw.get("primary_domain"))
        predicate = re.sub(r"[^a-z0-9_]+", "_", str(raw.get("predicate") or "").lower()).strip("_")
        if not subject_id or not predicate or not primary_domain:
            return None
        object_name = _compact_whitespace(raw.get("object_entity") or "")
        object_id = entity_ids.get(object_name) if object_name else None
        raw_claim_text = _compact_whitespace(raw.get("claim_text") or "")
        if not object_id and not raw_claim_text:
            return None
        confidence = self._clamp_float(raw.get("confidence"), 0.0, 1.0, 0.7)
        has_wearer_evidence = any(
            facts_by_id[fact_id]["metadata"].get("has_world_owner_speaker")
            for fact_id in evidence_ids
        )
        if not has_wearer_evidence:
            confidence = min(confidence, 0.65)
        claim_text = self._normalize_user_claim_text(
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
            "primary_domain": primary_domain,
            "status": (
                "active"
                if has_wearer_evidence and confidence >= self._user_claim_explicit_min_confidence
                else "candidate"
            ),
            "confidence": confidence,
            "valid_from": _compact_whitespace(raw.get("valid_from") or ""),
            "valid_to": _compact_whitespace(raw.get("valid_to") or ""),
            "extractor_version": "user_claim_explicit_v1",
            "prompt_version": "v1",
            "metadata": {"source_fact_count": len(evidence_ids)},
            "evidence_fact_ids": evidence_ids,
        }

    def _update_derived_user_claims_from_explicit_claims(
        self,
        *,
        new_explicit_claim_ids: Sequence[int],
    ) -> Dict[str, Any]:
        """Derive local conclusions from this batch's explicit-claim changes.

        The LLM proposes only premise-linked conclusions. Entity IDs,
        premise membership, fact traceability, and writes remain deterministic
        program responsibilities.
        """
        explicit_claim_ids = list(dict.fromkeys(
            int(value)
            for value in new_explicit_claim_ids or []
            if str(value).strip().isdigit() and int(value) > 0
        ))
        report: Dict[str, Any] = {
            "enabled": 1,
            "input_explicit_claim_count": len(explicit_claim_ids),
            "seed_claim_count": 0,
            "processed_seed_claim_count": 0,
            "successful_seed_claim_count": 0,
            "input_claim_count": 0,
            "candidate_count": 0,
            "updated": 0,
            "created": 0,
            "suppressed": 0,
            "new_claim_ids": [],
            "claim_relations": [],
            "failed_seed_claim_ids": [],
            "failures": [],
            "completed": True,
        }
        if not explicit_claim_ids:
            return report
        changed_claims = self._db.get_user_claims_by_ids(explicit_claim_ids)
        world_owner_entity_id = self._intent_world_owner_entity_id()
        seed_claims = [
            claim
            for claim in changed_claims
            if claim.get("claim_origin") == "explicit"
            and claim.get("status") == "active"
            and int(claim.get("subject_entity_id") or 0) == world_owner_entity_id
        ]
        report["seed_claim_count"] = len(seed_claims)
        if not seed_claims:
            return report

        new_derived_claim_ids: set[int] = set()
        claim_relations: Dict[Tuple[int, int, str], Dict[str, Any]] = {}
        for seed_claim in seed_claims:
            related_claims = self._retrieve_related_derived_explicit_claims(
                seed_claim=seed_claim,
                batch_seed_claims=seed_claims,
                world_owner_entity_id=world_owner_entity_id,
            )
            report["processed_seed_claim_count"] += 1
            report["input_claim_count"] += 1 + len(related_claims)
            failure: Dict[str, Any] = {}
            claim_report: Dict[str, Any] = {}
            try:
                outcome = self._extract_derived_user_claims(
                    changed_claims=[seed_claim],
                    related_claims=related_claims,
                )
                if outcome is None:
                    failure = {"reason": "invalid_llm_derived_claim_response"}
                elif outcome:
                    with self._db.transaction():
                        claim_report = self._apply_derived_user_claim_candidates(
                            raw_candidates=outcome,
                            changed_claims=[seed_claim],
                            related_claims=related_claims,
                        )
            except sqlite3.Error:
                # Storage/infrastructure failures are not invalid LLM outputs.
                raise
            except Exception as exc:
                failure = {
                    "reason": "derived_claim_processing_error",
                    "error_type": type(exc).__name__,
                }
                self._logger.exception(
                    "Derived user claim processing failed seed_claim_id=%s", seed_claim["id"],
                )
            if failure:
                report["completed"] = False
                report["failed_seed_claim_ids"].append(int(seed_claim["id"]))
                failure["seed_claim_id"] = int(seed_claim["id"])
                report["failures"].append(failure)
                self._log_info("memory_user_claim", "derived_seed_failed", failure)
                continue
            report["successful_seed_claim_count"] += 1
            report["candidate_count"] += len(outcome)
            if not outcome:
                continue
            for key in ("updated", "created", "suppressed"):
                report[key] += int(claim_report[key])
            new_derived_claim_ids.update(
                int(value)
                for value in claim_report.get("new_claim_ids") or []
                if int(value) > 0
            )
            for relation in claim_report.get("claim_relations") or []:
                key = (
                    int(relation["new_claim_id"]),
                    int(relation["existing_claim_id"]),
                    str(relation["semantic_relation"]),
                )
                claim_relations[key] = relation
        report["new_claim_ids"] = sorted(new_derived_claim_ids)
        report["claim_relations"] = [
            claim_relations[key] for key in sorted(claim_relations)
        ]
        return report

    def _retrieve_related_derived_explicit_claims(
        self,
        *,
        seed_claim: Dict[str, Any],
        batch_seed_claims: Sequence[Dict[str, Any]],
        world_owner_entity_id: int,
    ) -> List[Dict[str, Any]]:
        """Find premises for one new claim without grouping by its subject."""
        seed_claim_id = int(seed_claim["id"])
        object_entity_id = int(seed_claim.get("object_entity_id") or 0)
        primary_domain = str(seed_claim.get("primary_domain") or "")
        predicate = str(seed_claim.get("predicate") or "")
        claims_by_id: Dict[int, Dict[str, Any]] = {}
        batch_claim_ids = {int(claim["id"]) for claim in batch_seed_claims}

        def add_claims(claims: Sequence[Dict[str, Any]]) -> None:
            for claim in claims:
                claim_id = int(claim["id"])
                if (
                    claim_id != seed_claim_id
                    and claim.get("claim_origin") == "explicit"
                    and claim.get("status") == "active"
                    and int(claim.get("subject_entity_id") or 0)
                    == world_owner_entity_id
                ):
                    claims_by_id[claim_id] = claim

        # Other claims created in the same batch may be necessary premises
        # even when they refer to different objects.
        add_claims(batch_seed_claims)
        search_filters: List[Dict[str, Any]] = []
        if object_entity_id and object_entity_id != world_owner_entity_id:
            search_filters.append({"entity_id": object_entity_id})
        if primary_domain and primary_domain != "general":
            search_filters.append({"primary_domain": primary_domain})
        if predicate:
            search_filters.append({"predicate": predicate})
        # Keep a small recent owner-wide pool for cross-object and
        # cross-predicate deductions that the narrower filters cannot find.
        search_filters.append({})
        for filters in search_filters:
            add_claims(self._db.get_user_claims(
                subject_entity_id=world_owner_entity_id,
                claim_origin="explicit",
                statuses=["active"],
                limit=64,
                **filters,
            ))

        def relevance(claim: Dict[str, Any]) -> Tuple[bool, bool, bool, bool, bool, str, int]:
            claim_object_id = int(claim.get("object_entity_id") or 0)
            return (
                bool(object_entity_id and claim_object_id == object_entity_id),
                bool(predicate and claim.get("predicate") == predicate),
                bool(primary_domain and claim.get("primary_domain") == primary_domain),
                claim.get("claim_type") == seed_claim.get("claim_type"),
                int(claim["id"]) in batch_claim_ids,
                str(claim.get("updated_at") or ""),
                int(claim["id"]),
            )

        batch_related = sorted(
            (claim for claim in claims_by_id.values() if int(claim["id"]) in batch_claim_ids),
            key=relevance,
            reverse=True,
        )[:8]
        historical_related = sorted(
            (claim for claim in claims_by_id.values() if int(claim["id"]) not in batch_claim_ids),
            key=relevance,
            reverse=True,
        )[:32 - len(batch_related)]
        return [*batch_related, *historical_related]

    def _apply_derived_user_claim_candidates(
        self,
        *,
        raw_candidates: Sequence[Dict[str, Any]],
        changed_claims: Sequence[Dict[str, Any]],
        related_claims: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Validate, reconcile, and record one subject's LLM conclusions."""
        report: Dict[str, Any] = {
            "updated": 0, "created": 0, "suppressed": 0,
            "new_claim_ids": [], "claim_relations": [],
        }
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
            if (candidate := self._normalize_derived_user_claim(
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
        applied = self._reconcile_user_claims(candidates, facts_by_id=facts_by_id)
        persisted_claims = {
            int(claim["id"]): claim
            for item in applied
            for claim in self._db.get_user_claims_by_ids([item["claim_id"]])
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
            derivation_id = self._db.upsert_user_claim_derivation(
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
                    "generator": "derived_user_claim_extraction",
                },
            )
            self._db.replace_user_claim_derivation_premises(
                derivation_id=derivation_id,
                premise_claim_ids=premise_ids,
            )
            persisted_derived_ids.append(claim_id)
            report["new_claim_ids"].append(claim_id)
            for decision in item.get("relations") or []:
                existing_claim_id = int(decision.get("existing_claim_id") or 0)
                if existing_claim_id <= 0 or existing_claim_id == claim_id:
                    continue
                report["claim_relations"].append({
                    "relation_event_id": int(decision["relation_event_id"]),
                    "new_claim_id": claim_id,
                    "existing_claim_id": existing_claim_id,
                    "semantic_relation": str(decision.get("semantic_relation") or ""),
                    "confidence": float(decision.get("confidence") or 0.0),
                    "reason": str(decision.get("reason") or ""),
                })
            report["created"] += int(item.get("created") or 0)
            report["updated"] += 1
        if persisted_derived_ids:
            self._sync_user_claim_recall_documents(
                sorted(set(persisted_derived_ids))
            )
        report["new_claim_ids"] = sorted(set(report["new_claim_ids"]))
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
            "primary_domain": claim.get("primary_domain") or "",
            "claim_text": claim.get("claim_text") or "",
            "valid_from": claim.get("valid_from") or "",
            "valid_to": claim.get("valid_to") or "",
            "confidence": float(claim.get("confidence") or 0.0),
        }

    def _extract_derived_user_claims(
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
            DERIVED_USER_CLAIM_EXTRACTION_PROMPT_EN
            if language == "en" else DERIVED_USER_CLAIM_EXTRACTION_PROMPT_ZH
        )
        raw = self._llm_client.call_llm(prompt_template.replace(
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
        parsed = self._llm_client.parse_json_object_from_llm_text(raw or "")
        if parsed is None or not isinstance(parsed.get("claims"), list):
            return None
        return [item for item in parsed["claims"][:16] if isinstance(item, dict)]

    def _normalize_derived_user_claim(
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
        if _compact_whitespace(
            entity_names.get(subject_entity_id) or ""
        ).lower() != self._world_owner_entity_name.lower():
            return None
        claim_type = str(raw.get("claim_type") or "").strip().lower()
        if claim_type not in {"affiliation", "relationship", "constraint"}:
            return None
        primary_domain = self._normalize_primary_domain(raw.get("primary_domain"))
        if not primary_domain:
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
        if confidence < self._user_claim_derived_min_confidence:
            return None
        valid_from, valid_to = self._derived_claim_validity_bounds(premises)
        if valid_from and valid_to and valid_from > valid_to:
            return None
        support_fact_ids = sorted({
            fact_id
            for premise_id in premise_ids
            for fact_id in self._db.get_user_claim_evidence_fact_ids([premise_id]).get(
                premise_id, []
            )
        })
        if not support_fact_ids:
            return None
        claim_text = self._normalize_user_claim_text(
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
            "primary_domain": primary_domain,
            "status": "active",
            "confidence": confidence,
            "valid_from": valid_from,
            "valid_to": valid_to,
            "extractor_version": "user_claim_derived_v1",
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

    def _update_user_observed_patterns_from_action_signals(
        self, signal_ids: Sequence[int],
    ) -> Dict[str, Any]:
        """Update nearby patterns first, then discover patterns from action history."""
        seed_signals = self._db.get_user_action_signals_by_ids(signal_ids)
        report: Dict[str, Any] = {
            "candidate_count": 0, "updated": 0, "created": 0,
            "new_pattern_ids": [], "completed": True,
            "similar_action_batches": 0, "relations_recorded": 0,
            "direct_support_signal_count": 0,
        }
        if not seed_signals:
            return report
        subject_id = int(seed_signals[0]["subject_entity_id"])
        history = self._db.get_user_action_signals()
        pool = {int(row["signal_id"]): row for row in history}
        pool.update({int(row["signal_id"]): row for row in seed_signals})
        vectors = self._user_action_vectors_for_signals(list(pool.values()))
        existing_patterns = self._db.get_user_observed_patterns(
            subject_entity_id=subject_id, limit=None,
        )
        facts_by_id = {
            int(fact["id"]): fact
            for fact in self._db.get_memory_facts_by_ids(
                [int(signal["fact_id"]) for signal in seed_signals]
            )
        }
        # A pattern is retrieved directly from its own text and supporting
        # actions. Discovery's historical-signal cap must not hide it.
        related_by_seed: Dict[Tuple[int, ...], List[Dict[str, Any]]] = {}
        patterns_by_id = {int(pattern["id"]): pattern for pattern in existing_patterns}
        for signal in seed_signals:
            related = self._retrieve_similar_user_observed_patterns(
                signal=signal, patterns=existing_patterns, signals_by_id=pool,
                vectors=vectors,
            )
            if related:
                key = tuple(int(pattern["id"]) for pattern in related)
                related_by_seed.setdefault(key, []).append(signal)
        supported_signal_ids: set[int] = set()
        for pattern_ids, related_signals in related_by_seed.items():
            for offset in range(0, len(related_signals), 8):
                direct_update = self._update_observed_patterns_from_new_action_signals(
                    new_signals=related_signals[offset:offset + 8],
                    existing_patterns=[patterns_by_id[value] for value in pattern_ids],
                    facts_by_id=facts_by_id,
                )
                if direct_update is None:
                    report.update(completed=False, error="invalid_action_pattern_relation_response")
                    return report
                supported_signal_ids.update(direct_update["supported_signal_ids"])
                report["updated"] += len(direct_update["updated_pattern_ids"])
                report["relations_recorded"] += direct_update["relations_recorded"]
        report["direct_support_signal_count"] = len(supported_signal_ids)

        unresolved = [
            signal for signal in seed_signals
            if int(signal["signal_id"]) not in supported_signal_ids
        ]
        seed_ids = {int(signal["signal_id"]) for signal in seed_signals}
        neighborhoods = self._retrieve_similar_user_action_signal_batches(
            seeds=unresolved,
            history=[row for signal_id, row in pool.items() if signal_id not in seed_ids],
            vectors=vectors,
        )
        report["similar_action_batches"] = len(neighborhoods)
        for ordered, new_signal_ids in neighborhoods:
            facts_by_id = {
                int(fact["id"]): fact
                for fact in self._db.get_memory_facts_by_ids(
                    [int(signal["fact_id"]) for signal in ordered]
                )
            }
            if len(ordered) < self._user_observed_pattern_min_support_signals:
                continue
            report["candidate_count"] += 1
            patterns = self._extract_user_observed_patterns(
                subject_entity_id=subject_id,
                new_signal_ids=new_signal_ids,
                signals=ordered,
                facts_by_id=facts_by_id,
            )
            if patterns is None:
                report.update(completed=False, error="invalid_observed_pattern_response")
                return report
            for pattern in patterns:
                applied = self._reconcile_user_observed_pattern(
                    pattern, subject_entity_id=subject_id,
                )
                if applied is None:
                    report.update(completed=False, error="invalid_pattern_reconciliation")
                    return report
                report["created"] += int(applied["created"])
                report["updated"] += int(not applied["created"])
                report["relations_recorded"] += int(applied["relations_recorded"])
                if applied["created"]:
                    report["new_pattern_ids"].append(int(applied["pattern_id"]))
        report["new_pattern_ids"] = sorted(set(report["new_pattern_ids"]))
        return report

    def _update_observed_patterns_from_new_action_signals(
        self, *, new_signals: Sequence[Dict[str, Any]],
        existing_patterns: Sequence[Dict[str, Any]],
        facts_by_id: Dict[int, Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Classify new behavior against existing patterns before discovery."""
        result: Dict[str, Any] = {
            "supported_signal_ids": set(), "updated_pattern_ids": set(),
            "relations_recorded": 0,
        }
        if not new_signals or not existing_patterns:
            return result
        def representative_ids(pattern: Dict[str, Any]) -> List[int]:
            ids = pattern.get("support_signal_ids") or []
            return ids if len(ids) <= 4 else [ids[0], *ids[-3:]]

        evidence_ids = [
            signal_id
            for pattern in existing_patterns
            for signal_id in representative_ids(pattern)
        ]
        evidence_signals = self._db.get_user_action_signals_by_ids(evidence_ids)
        evidence_facts = {
            int(fact["id"]): fact
            for fact in self._db.get_memory_facts_by_ids([
                int(signal["fact_id"]) for signal in evidence_signals
            ])
        }
        evidence_by_id = {
            int(signal["signal_id"]): signal for signal in evidence_signals
        }

        def signal_view(signal: Dict[str, Any], fact: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "signal_id": int(signal["signal_id"]),
                "action_text": signal.get("action_text") or "",
                "action_kind": signal.get("action_kind") or "",
                "event_time": signal.get("event_time_key") or "",
                "location": signal.get("location") or "",
                "fact_id": int(signal["fact_id"]),
                "fact_summary": fact.get("summary") or "",
                "dialogue_time": fact.get("dialogue_time_key") or "",
            }

        new_by_id = {int(signal["signal_id"]): signal for signal in new_signals}
        pattern_by_id = {int(pattern["id"]): pattern for pattern in existing_patterns}
        language = self._resolve_prompt_language_from_text(
            "\n".join(str(signal.get("action_text") or "") for signal in new_signals)
        )
        template = (
            USER_OBSERVED_PATTERN_SIGNAL_RELATION_PROMPT_EN
            if language == "en" else USER_OBSERVED_PATTERN_SIGNAL_RELATION_PROMPT_ZH
        )
        raw = self._llm_client.call_llm(
            template.replace("{new_signals}", json.dumps([
                signal_view(signal, facts_by_id.get(int(signal["fact_id"])) or {})
                for signal in new_signals
            ], ensure_ascii=False, indent=2)).replace("{existing_patterns}", json.dumps([
                {
                    "id": int(pattern["id"]),
                    "pattern_text": pattern.get("pattern_text") or "",
                    "condition_text": pattern.get("condition_text") or "",
                    "confidence": pattern.get("confidence"),
                    "supporting_actions": [
                        signal_view(evidence_by_id[signal_id], evidence_facts.get(
                            int(evidence_by_id[signal_id]["fact_id"])
                        ) or {})
                        for signal_id in representative_ids(pattern)
                        if signal_id in evidence_by_id
                    ],
                }
                for pattern in existing_patterns
            ], ensure_ascii=False, indent=2))
        )
        parsed = self._llm_client.parse_json_object_from_llm_text(raw or "")
        if not parsed or not isinstance(parsed.get("decisions"), list):
            return None
        decisions: Dict[Tuple[int, int], Dict[str, Any]] = {}
        for item in parsed["decisions"]:
            if not isinstance(item, dict):
                continue
            try:
                signal_id = int(item.get("signal_id"))
                pattern_id = int(item.get("existing_pattern_id"))
            except (TypeError, ValueError):
                continue
            relation = str(item.get("semantic_relation") or "").strip().lower()
            if (
                signal_id not in new_by_id or pattern_id not in pattern_by_id
                or relation not in {"supports", "refines", "contradicts", "unrelated"}
            ):
                continue
            confidence = self._clamp_float(item.get("confidence"), 0.0, 1.0, 0.0)
            if confidence < 0.7 or relation == "unrelated":
                continue
            key = (signal_id, pattern_id)
            if key not in decisions or confidence > decisions[key]["confidence"]:
                decisions[key] = {
                    "semantic_relation": relation,
                    "confidence": confidence,
                    "reason": _compact_whitespace(item.get("reason") or "")[:240],
                }
        with self._db.transaction():
            for (signal_id, pattern_id), decision in decisions.items():
                signal = new_by_id[signal_id]
                snapshot = signal_view(
                    signal, facts_by_id.get(int(signal["fact_id"])) or {},
                )
                if decision["semantic_relation"] == "supports":
                    added = self._db.add_user_observed_pattern_signals(
                        pattern_id=pattern_id, signal_ids=[signal_id],
                    )
                    if added:
                        self._db.refresh_user_observed_pattern(
                            pattern_id=pattern_id,
                            confidence=float(pattern_by_id[pattern_id]["confidence"]),
                        )
                        result["updated_pattern_ids"].add(pattern_id)
                    result["supported_signal_ids"].add(signal_id)
                    if not added:
                        continue
                result["relations_recorded"] += int(
                    self._db.record_user_observed_pattern_relation(
                        target_pattern_id=pattern_id, trigger_pattern_id=None,
                        semantic_relation=decision["semantic_relation"],
                        support_signal_ids=[signal_id],
                        candidate_snapshot=snapshot,
                        semantic_confidence=decision["confidence"],
                        semantic_reason=decision["reason"],
                    )
                )
        return result

    def _user_action_vectors_for_signals(
        self, signals: Sequence[Dict[str, Any]],
    ) -> Dict[int, np.ndarray]:
        """Read cached action vectors and embed only missing action texts."""
        self._ensure_embedding_client()
        assert self._embedding_client is not None
        dimension = self._embedding_client.dimension
        vectors: Dict[int, np.ndarray] = {}
        missing: List[Dict[str, Any]] = []
        for signal in signals:
            signal_id = int(signal["signal_id"])
            blob = signal.get("action_text_embedding")
            vector = np.frombuffer(blob, dtype=np.float32) if blob else None
            if vector is not None and vector.size == dimension and np.all(np.isfinite(vector)):
                vectors[signal_id] = vector
            else:
                missing.append(signal)
        if missing:
            embedded = self._embedding_client.embed_batch(
                str(signal.get("action_text") or "") for signal in missing
            )
            updates: List[Tuple[int, np.ndarray]] = []
            for signal, vector in zip(missing, embedded):
                if vector is None:
                    continue
                array = np.asarray(vector, dtype=np.float32)
                if array.size != dimension or not np.all(np.isfinite(array)):
                    continue
                signal_id = int(signal["signal_id"])
                vectors[signal_id] = array
                updates.append((signal_id, array))
            if updates:
                self._db.set_user_action_signal_embeddings(updates)
        return vectors

    @staticmethod
    def _user_action_signal_similarity(
        left: Dict[str, Any], right: Dict[str, Any],
        vectors: Dict[int, np.ndarray],
    ) -> float:
        lexical = action_lexical_similarity(
            left.get("action_text"), right.get("action_text"),
        )
        left_vector = vectors.get(int(left["signal_id"]))
        right_vector = vectors.get(int(right["signal_id"]))
        semantic = (
            max(0.0, _cal_embedding_cosine_similarity(left_vector, right_vector))
            if left_vector is not None and right_vector is not None else 0.0
        )
        return max(lexical, semantic * 0.82)

    def _retrieve_similar_user_observed_patterns(
        self, *, signal: Dict[str, Any],
        patterns: Sequence[Dict[str, Any]],
        signals_by_id: Dict[int, Dict[str, Any]],
        vectors: Dict[int, np.ndarray],
    ) -> List[Dict[str, Any]]:
        """Retrieve existing patterns independently of discovery neighborhoods."""
        ranked: List[Tuple[float, Dict[str, Any]]] = []
        for pattern in patterns:
            score = action_lexical_similarity(
                signal.get("action_text"), pattern.get("pattern_text"),
            )
            for support_id in pattern.get("support_signal_ids") or []:
                support = signals_by_id.get(int(support_id))
                if support is not None:
                    score = max(score, self._user_action_signal_similarity(
                        signal, support, vectors,
                    ))
            if score >= 0.60:
                ranked.append((score, pattern))
        ranked.sort(key=lambda item: (-item[0], int(item[1]["id"])))
        return [pattern for _, pattern in ranked[:8]]

    def _retrieve_similar_user_action_signal_batches(
        self, *, seeds: Sequence[Dict[str, Any]],
        history: Sequence[Dict[str, Any]], vectors: Dict[int, np.ndarray],
    ) -> List[Tuple[List[Dict[str, Any]], set[int]]]:
        """Build novel-pattern batches from similar actions, without pattern quotas."""
        if not seeds:
            return []
        capacity = max(1, self._user_observed_pattern_signal_limit)
        seed_ids = {int(row["signal_id"]) for row in seeds}
        historical = [
            row for row in history if int(row["signal_id"]) not in seed_ids
        ]
        batches: List[Tuple[List[Dict[str, Any]], set[int]]] = []
        remaining = list(seeds)
        while remaining:
            anchor = remaining.pop(0)
            grouped = [anchor]
            still_unmatched = []
            for candidate in remaining:
                if (
                    len(grouped) < capacity
                    and self._user_action_signal_similarity(anchor, candidate, vectors) >= 0.60
                ):
                    grouped.append(candidate)
                else:
                    still_unmatched.append(candidate)
            remaining = still_unmatched
            ranked_history = sorted(
                (
                    (
                        max(self._user_action_signal_similarity(seed, row, vectors)
                            for seed in grouped),
                        row,
                    )
                    for row in historical
                ),
                key=lambda item: (-item[0], -int(item[1]["signal_id"])),
            )
            neighbors = [
                row for score, row in ranked_history if score >= 0.60
            ][:max(0, capacity - len(grouped))]
            batches.append((
                [*grouped, *neighbors],
                {int(row["signal_id"]) for row in grouped},
            ))
        return batches

    def _fact_user_claim_signal_mapping_updates(
        self,
        fact_id: int,
        fact: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Persist every normalized user-claim signal for one fact."""
        updates: List[Dict[str, Any]] = []
        raw_signals = self._normalize_user_claim_signal(
            fact.get("user_claim_signals"),
            allows_world_owner_reference=(
                isinstance(fact.get("metadata"), dict)
                and bool(fact["metadata"].get("has_world_owner_speaker")
                         or fact["metadata"].get("has_world_owner_alias_reference"))
            ),
            owner_aliases=fact.get("resolved_world_owner_aliases") or [],
        )
        for signal in raw_signals:
            claim_type_hint = str(signal.get("claim_type_hint") or "").lower()
            subject_entity = self._subject_entity_for_user_claim_signal(signal, fact)
            if not subject_entity:
                continue
            entity_mapping = self._db.upsert_entity_nodes([{
                "name": subject_entity,
                "type": "PERSON",
            }])
            if subject_entity not in entity_mapping:
                continue
            claim_anchor = _compact_whitespace(signal.get("claim_anchor") or "")
            if not claim_anchor:
                continue
            claim_anchor_key = self._generate_topic_name_key(claim_anchor)
            if not claim_anchor_key:
                continue
            evidence_basis = _compact_whitespace(
                signal.get("evidence_basis") or ""
            )
            signal_fingerprint = hashlib.sha256(
                "\x1f".join((
                    claim_type_hint,
                    claim_anchor_key,
                    evidence_basis.lower(),
                )).encode("utf-8")
            ).hexdigest()[:16]
            updates.append({
                "fact_id": int(fact_id),
                "signal_key": (
                    f"{int(entity_mapping[subject_entity])}|{claim_type_hint}|"
                    f"{claim_anchor_key}|{signal_fingerprint}"
                ),
                "subject": subject_entity,
                "subject_entity_id": int(entity_mapping[subject_entity]),
                "claim_type_hint": claim_type_hint,
                "claim_anchor": claim_anchor,
                "claim_anchor_key": claim_anchor_key,
                "evidence_basis": evidence_basis,
                "confidence": float(signal.get("confidence") or 0.0),
            })
        return updates

    def _fact_user_action_signal_updates(
        self,
        *,
        fact_id: int,
        fact: Dict[str, Any],
        owner_entity_id: int,
    ) -> List[Dict[str, Any]]:
        """Prepare persisted action evidence without routing it to claims."""
        if self._world_owner_entity_name not in (fact.get("entities") or []):
            return []
        updates: List[Dict[str, Any]] = []
        for signal in fact.get("user_action_signals") or []:
            if not isinstance(signal, dict):
                continue
            action_kind = str(signal.get("action_kind") or "")
            action_text = _compact_whitespace(signal.get("action_text") or "")
            owner_evidence_type = str(signal.get("owner_evidence_type") or "")
            if (
                signal.get("subject_entity") != self._world_owner_entity_name
                or action_kind not in {"action", "choice", "participation"}
                or not action_text
                or owner_evidence_type not in {
                    "self_statement", "confirmed_response", "third_party_report",
                }
            ):
                continue
            fingerprint = hashlib.sha256(
                "\x1f".join((
                    action_kind,
                    action_text.lower(),
                    owner_evidence_type,
                    str(signal.get("owner_evidence_text") or "").lower(),
                    str(signal.get("other_evidence_text") or "").lower(),
                )).encode("utf-8")
            ).hexdigest()[:16]
            updates.append({
                "fact_id": int(fact_id),
                "signal_key": f"{owner_entity_id}|{action_kind}|{fingerprint}",
                "subject_entity_id": int(owner_entity_id),
                "action_kind": action_kind,
                "action_text": action_text,
                "event_time_key": str(fact.get("event_time_key") or ""),
                "time_confidence": str(
                    (fact.get("metadata") or {}).get("time_confidence") or "unknown"
                ),
                "location": str((fact.get("metadata") or {}).get("where") or ""),
                "owner_evidence_type": owner_evidence_type,
                "owner_evidence_text": signal.get("owner_evidence_text") or "",
                "other_evidence_text": signal.get("other_evidence_text") or "",
                "evidence_basis": signal.get("evidence_basis") or "",
                "confidence": float(signal.get("confidence") or 0.0),
            })
        return updates

    def _fact_user_intent_signal_mapping_updates(
        self,
        fact_id: int,
        fact: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Project one fact's normalized prospective signals into DB rows."""
        updates: List[Dict[str, Any]] = []
        raw_signals = self._normalize_user_intent_signals(
            fact.get("user_intent_signals"),
            fallback_entity=fact.get("primary_entity"),
            allows_world_owner_reference=(
                isinstance(fact.get("metadata"), dict)
                and bool(fact["metadata"].get("has_world_owner_speaker")
                         or fact["metadata"].get("has_world_owner_alias_reference"))
            ),
            owner_aliases=fact.get("resolved_world_owner_aliases") or [],
        )
        fact_entities_by_name = {
            str(descriptor["name"]).lower(): descriptor
            for descriptor in self._fact_entity_descriptors(fact)
            if _compact_whitespace(descriptor.get("name") or "")
        }
        for signal in raw_signals:
            subject_name = _compact_whitespace(signal.get("subject_entity") or "")
            entity_descriptor = fact_entities_by_name.get(subject_name.lower())
            if entity_descriptor is None:
                continue
            subject_name = str(entity_descriptor["name"])
            entity_mapping = self._db.upsert_entity_nodes([entity_descriptor])
            if subject_name not in entity_mapping:
                continue
            user_intent_anchor = _compact_whitespace(
                signal.get("user_intent_anchor") or ""
            )
            user_intent_anchor_key = self._generate_topic_name_key(user_intent_anchor)
            if not user_intent_anchor or user_intent_anchor_key == "general":
                continue
            updates.append({
                "fact_id": int(fact_id),
                "signal_key": (
                    f"{int(entity_mapping[subject_name])}|{signal['signal_purpose']}|"
                    f"{user_intent_anchor_key}|"
                    f"{','.join(sorted(signal['candidate_object_types']))}"
                ),
                "subject_entity_id": int(entity_mapping[subject_name]),
                "signal_purpose": signal["signal_purpose"],
                "candidate_object_types": signal["candidate_object_types"],
                "user_role": signal["user_role"],
                "user_intent_anchor": user_intent_anchor,
                "user_intent_anchor_key": user_intent_anchor_key,
                "assertion_source": signal["assertion_source"],
                "evidence_basis": signal["evidence_basis"],
                "confidence": signal["confidence"],
            })
        return updates

    def _extract_user_observed_patterns(
        self,
        *,
        subject_entity_id: int,
        new_signal_ids: set[int],
        signals: Sequence[Dict[str, Any]],
        facts_by_id: Dict[int, Dict[str, Any]],
    ) -> Optional[List[Dict[str, Any]]]:
        language = self._resolve_prompt_language_from_text(
            "\n".join(str(signal.get("action_text") or "") for signal in signals)
        )
        prompt_template = (
            USER_OBSERVED_PATTERN_EXTRACTION_PROMPT_EN
            if language == "en" else USER_OBSERVED_PATTERN_EXTRACTION_PROMPT_ZH
        )
        raw = self._llm_client.call_llm(
            prompt_template.replace("{pattern_target}", json.dumps({
                "subject_entity": self._world_owner_entity_name,
                "subject_entity_id": subject_entity_id,
                "world_owner_entity": self._world_owner_entity_name,
                "new_signal_ids": sorted(new_signal_ids),
            }, ensure_ascii=False, indent=2)).replace("{action_signals}", json.dumps([
                {
                    "signal_id": int(signal["signal_id"]),
                    "action_kind": signal.get("action_kind") or "",
                    "event_time": signal.get("event_time_key") or "",
                    "action_text": signal.get("action_text") or "",
                    "location": signal.get("location") or "",
                    "owner_evidence_type": signal.get("owner_evidence_type") or "",
                    "owner_evidence_text": signal.get("owner_evidence_text") or "",
                    "other_evidence_text": signal.get("other_evidence_text") or "",
                    "fact_id": int(signal["fact_id"]),
                    "fact_summary": (
                        facts_by_id.get(int(signal["fact_id"])) or {}
                    ).get("summary") or "",
                    "dialogue_time": (
                        facts_by_id.get(int(signal["fact_id"])) or {}
                    ).get("dialogue_time_key") or "",
                }
                for signal in signals
            ], ensure_ascii=False, indent=2))
        )
        parsed = self._llm_client.parse_json_object_from_llm_text(raw or "")
        if parsed is None or not isinstance(parsed.get("patterns"), list):
            return None
        available_signal_ids = {int(signal["signal_id"]) for signal in signals}
        result: List[Dict[str, Any]] = []
        for raw_pattern in parsed["patterns"][:4]:
            pattern = self._normalize_user_observed_pattern(
                raw_pattern,
                subject_entity_id=subject_entity_id,
                new_signal_ids=new_signal_ids,
                available_signal_ids=available_signal_ids,
            )
            if pattern:
                result.append(pattern)
        return result

    def _normalize_user_observed_pattern(
        self,
        raw: Any,
        *,
        subject_entity_id: int,
        new_signal_ids: set[int],
        available_signal_ids: set[int],
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(raw, dict):
            return None
        primary_domain = self._normalize_primary_domain(raw.get("primary_domain"))
        if not primary_domain:
            return None
        if (
            _compact_whitespace(raw.get("subject_entity") or "")
            != self._world_owner_entity_name
        ):
            return None
        signal_ids = list(dict.fromkeys(
            int(value) for value in (raw.get("support_signal_ids") or [])
            if str(value).strip().isdigit() and int(value) in available_signal_ids
        ))[:24]
        if (
            len(signal_ids) < self._user_observed_pattern_min_support_signals
            or not new_signal_ids.intersection(signal_ids)
        ):
            return None
        pattern_text = _compact_whitespace(raw.get("pattern_text") or "")[:480]
        if not pattern_text:
            return None
        confidence = self._clamp_float(raw.get("confidence"), 0.0, 1.0, 0.7)
        return {
            "pattern_text": pattern_text,
            "condition_text": _compact_whitespace(raw.get("condition_text") or "")[:240],
            "primary_domain": primary_domain,
            "confidence": confidence,
            "support_signal_ids": signal_ids,
        }

    def _reconcile_user_observed_pattern(
        self, pattern: Dict[str, Any], *, subject_entity_id: int,
    ) -> Optional[Dict[str, Any]]:
        """Reuse an equivalent pattern or create a new one with relation events."""
        existing = self._db.get_user_observed_patterns(
            subject_entity_id=subject_entity_id, limit=200,
        )
        # Domain is a preference for the shortlist, not an exclusion: a pattern
        # can be assigned to a different domain in a later extraction.
        ranked = sorted(existing, key=lambda row: (
            int(row.get("primary_domain") == pattern["primary_domain"]),
            action_lexical_similarity(
                pattern["pattern_text"], row.get("pattern_text"),
            ),
        ), reverse=True)
        related = ranked[:16]
        relations = self._classify_user_observed_pattern_relations(pattern, related)
        if relations is None:
            return None
        existing_by_id = {int(row["id"]): row for row in related}
        merge_relations = [
            item for item in relations
            if item["semantic_relation"] in {"duplicate", "supports"}
        ]
        if merge_relations:
            selected = max(merge_relations, key=lambda item: (
                item["confidence"],
                float(existing_by_id[item["existing_pattern_id"]].get("confidence") or 0.0),
            ))
            pattern_id = selected["existing_pattern_id"]
            with self._db.transaction():
                added = self._db.add_user_observed_pattern_signals(
                    pattern_id=pattern_id, signal_ids=pattern["support_signal_ids"],
                )
                self._db.refresh_user_observed_pattern(
                    pattern_id=pattern_id, confidence=pattern["confidence"],
                )
                recorded_count = 0
                for relation in relations:
                    relation_type = relation["semantic_relation"]
                    target_id = relation["existing_pattern_id"]
                    if relation_type == "unrelated" or (
                        relation_type in {"duplicate", "supports"}
                        and (target_id != pattern_id or not added)
                    ):
                        continue
                    recorded_count += int(self._db.record_user_observed_pattern_relation(
                        target_pattern_id=target_id,
                        trigger_pattern_id=(pattern_id if target_id != pattern_id else None),
                        semantic_relation=relation_type,
                        support_signal_ids=pattern["support_signal_ids"],
                        candidate_snapshot=pattern,
                        semantic_confidence=relation["confidence"],
                        semantic_reason=relation["reason"],
                    ))
            return {"pattern_id": pattern_id, "created": False,
                    "relations_recorded": recorded_count}

        pattern_key = hashlib.sha256(json.dumps({
            "pattern_text": pattern["pattern_text"].casefold(),
            "condition_text": pattern["condition_text"].casefold(),
            "primary_domain": pattern["primary_domain"],
        }, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
        with self._db.transaction():
            pattern_id, created = self._db.upsert_user_observed_pattern(
                subject_entity_id=subject_entity_id, pattern_key=pattern_key,
                pattern_text=pattern["pattern_text"],
                condition_text=pattern["condition_text"],
                primary_domain=pattern["primary_domain"],
                confidence=pattern["confidence"],
            )
            added = self._db.add_user_observed_pattern_signals(
                pattern_id=pattern_id, signal_ids=pattern["support_signal_ids"],
            )
            recorded_count = 0
            if not created and added:
                recorded_count += int(self._db.record_user_observed_pattern_relation(
                    target_pattern_id=pattern_id,
                    trigger_pattern_id=None,
                    semantic_relation="supports",
                    support_signal_ids=pattern["support_signal_ids"],
                    candidate_snapshot=pattern,
                    semantic_confidence=pattern["confidence"],
                    semantic_reason="New action signals supported the existing pattern key.",
                ))
            for relation in relations:
                if relation["semantic_relation"] not in {"refines", "contradicts"}:
                    continue
                if relation["existing_pattern_id"] == pattern_id:
                    continue
                recorded_count += int(self._db.record_user_observed_pattern_relation(
                    target_pattern_id=relation["existing_pattern_id"],
                    trigger_pattern_id=pattern_id,
                    semantic_relation=relation["semantic_relation"],
                    support_signal_ids=pattern["support_signal_ids"],
                    candidate_snapshot=pattern,
                    semantic_confidence=relation["confidence"],
                    semantic_reason=relation["reason"],
                ))
        return {"pattern_id": pattern_id, "created": created,
                "relations_recorded": recorded_count}

    def _classify_user_observed_pattern_relations(
        self, candidate: Dict[str, Any], existing: Sequence[Dict[str, Any]],
    ) -> Optional[List[Dict[str, Any]]]:
        if not existing:
            return []
        evidence_ids = list(candidate["support_signal_ids"][-8:])
        evidence_ids.extend(
            signal_id
            for row in existing
            for signal_id in (row.get("support_signal_ids") or [])[-4:]
        )
        signals_by_id = {
            int(row["signal_id"]): row
            for row in self._db.get_user_action_signals_by_ids(evidence_ids)
        }
        facts_by_id = {
            int(row["id"]): row
            for row in self._db.get_memory_facts_by_ids([
                int(signal["fact_id"]) for signal in signals_by_id.values()
            ])
        }

        def evidence_view(signal_ids: Sequence[int], limit: int) -> List[Dict[str, Any]]:
            result: List[Dict[str, Any]] = []
            for signal_id in signal_ids[-limit:]:
                signal = signals_by_id.get(int(signal_id))
                if signal is None:
                    continue
                result.append({
                    "signal_id": int(signal_id),
                    "action_text": signal.get("action_text") or "",
                    "event_time": signal.get("event_time_key") or "",
                    "location": signal.get("location") or "",
                    "fact_summary": (
                        facts_by_id.get(int(signal["fact_id"])) or {}
                    ).get("summary") or "",
                })
            return result

        language = self._resolve_prompt_language_from_text(
            candidate["pattern_text"],
        )
        template = (
            USER_OBSERVED_PATTERN_RECONCILIATION_PROMPT_EN
            if language == "en" else USER_OBSERVED_PATTERN_RECONCILIATION_PROMPT_ZH
        )
        raw = self._llm_client.call_llm(
            template.replace("{candidate_pattern}", json.dumps(
                {**candidate, "supporting_actions": evidence_view(
                    candidate["support_signal_ids"], 8,
                )}, ensure_ascii=False, indent=2,
            )).replace("{existing_patterns}", json.dumps([
                {
                    "id": row["id"], "pattern_text": row["pattern_text"],
                    "condition_text": row.get("condition_text") or "",
                    "primary_domain": row.get("primary_domain") or "",
                    "support_signal_ids": row.get("support_signal_ids") or [],
                    "supporting_actions": evidence_view(
                        row.get("support_signal_ids") or [], 4,
                    ),
                }
                for row in existing
            ], ensure_ascii=False, indent=2))
        )
        parsed = self._llm_client.parse_json_object_from_llm_text(raw or "")
        if not parsed or not isinstance(parsed.get("relations"), list):
            return None
        allowed_ids = {int(row["id"]) for row in existing}
        allowed_relations = {"duplicate", "supports", "refines", "contradicts", "unrelated"}
        result: List[Dict[str, Any]] = []
        for item in parsed["relations"]:
            if not isinstance(item, dict):
                continue
            try:
                target_id = int(item.get("existing_pattern_id"))
            except (TypeError, ValueError):
                continue
            relation = str(item.get("semantic_relation") or "").strip().lower()
            if target_id not in allowed_ids or relation not in allowed_relations:
                continue
            result.append({
                "existing_pattern_id": target_id,
                "semantic_relation": relation,
                "confidence": self._clamp_float(item.get("confidence"), 0.0, 1.0, 0.7),
                "reason": _compact_whitespace(item.get("reason") or "")[:240],
            })
        return result

    @staticmethod
    def _normalize_user_claim_text(
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

    def _log_user_intent_signals_loaded(
        self,
        signal_items: Sequence[Dict[str, Any]],
        reference_timestamp: Any,
    ) -> None:
        """Log each intent signal together with the fact supporting its extraction."""
        self._log_info("memory_user_intent_update", "signals_loaded", {
            "signal_count": len(signal_items),
            "signal_ids": [item["signal_id"] for item in signal_items],
            "reference_timestamp": reference_timestamp,
            "signal_items": [
                self._user_intent_signal_prompt_view(item) for item in signal_items
            ],
        })

    @staticmethod
    def _generate_topic_name_key(value: Any) -> str:
        text = _compact_whitespace(value).lower()
        text = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "", text)
        return text or "general"

    @staticmethod
    def _normalize_primary_domain(value: Any) -> str:
        """Return a controlled semantic domain or an empty invalid marker."""
        domain = re.sub(
            r"[^a-z]+", "_", str(value or "").strip().lower(),
        ).strip("_")
        return domain if domain in _PRIMARY_DOMAINS else ""

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
        """Extract one candidate's fact topic and score its overlap."""
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

        extend_values(raw.get("fact_topic"))

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

    def _get_user_claim_signals_from_fact(self, fact: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Read claim-level signals from their normalized fact mapping rows."""
        fact_id = int(fact.get("id") or 0)
        if fact_id <= 0:
            return []
        return list(
            self._db.get_fact_user_claim_signals(fact_id).values()
        )

    def _subject_entity_for_user_claim_signal(
        self,
        signal: Dict[str, Any],
        fact: Dict[str, Any],
    ) -> str:
        """Return the wearer subject only when it is a fact entity."""
        name = _compact_whitespace(signal.get("subject_entity") or "")
        if name.lower() != self._world_owner_entity_name.lower():
            return ""
        for fact_entity in fact.get("entities") or []:
            canonical_name = _compact_whitespace(fact_entity)
            if canonical_name and canonical_name.lower() == name.lower():
                return canonical_name
        return ""

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
                database=reader_db,
            )

    def _recall_sync(
        self,
        query: str,
        tags: Optional[List[str]] = None,
        time_end: Optional[str] = None,
        recall_mode: str = "normal",
        database: Optional[UserMemoryDB] = None,
    ) -> Dict[str, Any]:
        requested_recall_mode = str(recall_mode or "normal").strip().lower()
        started_at = time.monotonic()
        prompt_language = self._resolve_prompt_language_from_text(query)
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
        database: Optional[UserMemoryDB] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve Stage 1 direct seeds for permitted object types.

        User-activity observations are not recall documents in this version.
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
        database: Optional[UserMemoryDB] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve direct seeds from the shared document projection.

        Fact documents are hydrated back to ``memory_facts`` before they are
        returned, while derived-memory projections can become candidates
        directly.  For an explicit prospective time-slot query, prospective
        object types use a strict time browse instead of lexical terms.
        User-activity observations are deliberately excluded in this version.
        """
        direct_types = list(dict.fromkeys(
            str(value or "").strip().lower()
            for value in direct_object_types or []
            if str(value or "").strip().lower()
            in {"fact", "user_claim", "goal", "plan", "work_item"}
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
            "user_claim": ("active",),
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
            object_types.append("user_claim")
        if "prospective" in modes:
            object_types.extend(("goal", "plan", "work_item"))
        return object_types

    @staticmethod
    def _normalize_recall_object_types(value: Any) -> List[str]:
        """Normalize LLM-routed Stage 2 object types without excluding facts."""
        allowed_types = {
            "fact", "user_claim", "goal", "plan", "work_item",
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
        database: Optional[UserMemoryDB] = None,
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
            "user_claim": 0,
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
        payload: Dict[str, Any] = {
            "association_candidate_count": len(association_candidates),
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
        candidate_source_prefix: str = "stage1",
        database: Optional[UserMemoryDB] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve all fact associations for accepted direct candidates.

        Claim and intent candidates expand only to their mapped fact evidence.
        """
        db = database or self._db
        document_evidence_candidates = (
            self._retrieve_evidence_fact_association_candidates(
                direct_candidates=seed_candidates,
                candidate_source_prefix=candidate_source_prefix,
                database=db,
            )
        )
        return document_evidence_candidates

    def _retrieve_evidence_fact_association_candidates(
        self,
        *,
        direct_candidates: Sequence[Dict[str, Any]],
        candidate_source_prefix: str,
        database: UserMemoryDB,
    ) -> List[Dict[str, Any]]:
        """Expand claim/prospective candidates to their mapped fact evidence."""
        claim_candidates = [
            candidate for candidate in direct_candidates or []
            if str(candidate.get("index_level") or "") == "user_claim"
        ]
        user_intent_candidates = [
            candidate for candidate in direct_candidates or []
            if str(candidate.get("index_level") or "")
            in {"goal", "plan", "work_item"}
        ]
        evidence_by_fact_id: Dict[int, Tuple[str, float]] = {}

        claim_evidence = database.get_user_claim_evidence_fact_ids(
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
                for candidate in user_intent_candidates
            ],
            limit=max(24, len(user_intent_candidates) * 4),
        )
        for candidate in user_intent_candidates:
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

    @staticmethod
    def _recall_stage1_candidate_limits(
        *,
        top_k: int,
    ) -> Dict[str, Any]:
        """Build explicit Stage 1 retrieval, expansion, and output budgets.

        ``top_k`` controls the fact quota. Derived objects receive small,
        independent quotas only when the Stage 1 query route requests them.
        Activity observations are deliberately not direct-recall candidates.
        """
        k = max(1, int(top_k or 1))
        seed_per_level_limit = max(8, min(24, k * 2))
        return {
            # Direct lexical recall keeps a modest, fixed over-fetch ratio so
            # score filtering and association can work without inflating the
            # Stage 1 latency or candidate pool for contextual wording.
            "seed_limits": {
                "fact": seed_per_level_limit,
                "user_claim": max(4, min(12, k)),
                "goal": max(4, min(10, k)),
                "plan": max(4, min(12, k)),
                "work_item": max(4, min(12, k)),
            },
            "selected_limits": {
                "fact": k,
                "user_claim": max(2, min(4, k)),
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
            strong_anchor_reasons.append("temporal_user_intent_match")
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
            if str(candidate.get("index_level") or "") == "user_claim"
            and not candidate.get("_recall_association_relation")
        ]
        direct_user_intent_candidates = [
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
        has_temporal_user_intent_match = any(
            "temporal_user_intent_match"
            in (candidate.get("strong_anchor_reasons") or [])
            for candidate in direct_user_intent_candidates
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
                and has_temporal_user_intent_match
            )
        )
        has_primary_query_coverage = bool(
            matched_topic_term_keys
            or matched_high_value_entities
            or has_temporal_user_intent_match
        )
        knowledge_evidence_sufficient = (
            "knowledge" not in modes
            or bool(direct_claim_candidates and claim_evidence_count)
        )
        user_intent_evidence_sufficient = (
            "prospective" not in modes
            or bool(direct_user_intent_candidates and intent_evidence_count)
        )
        trusted = (
            bool(candidates)
            and has_primary_query_coverage
            and query_term_coverage_sufficient
            and query_entity_coverage_sufficient
            and knowledge_evidence_sufficient
            and user_intent_evidence_sufficient
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
            "direct_user_claim_count": len(direct_claim_candidates),
            "claim_evidence_count": claim_evidence_count,
            "knowledge_evidence_sufficient": knowledge_evidence_sufficient,
            "direct_user_intent_count": len(direct_user_intent_candidates),
            "intent_evidence_count": intent_evidence_count,
            "user_intent_evidence_sufficient": user_intent_evidence_sufficient,
            "has_temporal_user_intent_match": has_temporal_user_intent_match,
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
        database: Optional[UserMemoryDB] = None,
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
        database: Optional[UserMemoryDB] = None,
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
        database: Optional[UserMemoryDB] = None,
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
            in {"fact", "user_claim", "goal", "plan", "work_item"}
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
        database: Optional[UserMemoryDB] = None,
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
        database: UserMemoryDB,
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
        database: Optional[UserMemoryDB] = None,
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
            "user_claim": "memory_user_claims",
            "goal": "memory_user_intent_items",
            "plan": "memory_user_intent_items",
            "work_item": "memory_user_intent_items",
        }
        target_table = target_table_by_type.get(object_type)
        if not target_table:
            return None
        try:
            target_id = int(row.get("object_id"))
        except (TypeError, ValueError):
            return None
        if object_type == "user_claim":
            if database is None:
                return None
            stored_claims = database.get_user_claims_by_ids([target_id])
            if not stored_claims:
                return None
            subject_id = int(stored_claims[0].get("subject_entity_id") or 0)
            subject_name = database.get_entity_names_by_ids([subject_id]).get(
                subject_id, ""
            )
            # Ignore legacy rows outside the user-self claim scope without
            # destructively rewriting a user's existing database.
            if _compact_whitespace(subject_name).lower() != (
                self._world_owner_entity_name.lower()
            ):
                return None
        if object_type in {"goal", "plan", "work_item"}:
            if database is None:
                return None
            intent_row = database.get_intent_object(
                object_type=object_type,
                object_id=target_id,
            )
            if not intent_row:
                return None
            hydrated_intent = dict(intent_row)
            hydrated_intent["object_id"] = target_id
            intent_metadata = dict(hydrated_intent.get("metadata") or {})
            document_metadata = row.get("metadata")
            if isinstance(document_metadata, dict):
                intent_metadata.update(document_metadata)
            hydrated_intent["metadata"] = intent_metadata
            hydrated_intent.update({
                key: value
                for key, value in row.items()
                if key in {
                    "title", "identity_text", "identity_text_embedding",
                    "entity_ids", "topic_keys", "importance", "_bm25_score",
                    "_recall_temporal_match",
                }
            })
            row = hydrated_intent
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
        result = self._llm_client.call_llm(prompt)
        parsed = self._llm_client.parse_json_object_from_llm_text(result or "")
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

        Memory entities use stable role names (``用户本人``/``助手`` or
        ``user self``/``assistant``), while recall questions naturally use
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
            aliases.extend(["用户本人", "user self"])
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
                "fact_topic": "fact_topic",
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
                "fact_topic": "事实主题",
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
                        f"   {labels['fact_topic']}: {raw.get('fact_topic') or ''}",
                    ]
                elif group_key == "user_claim":
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
    def _normalize_entity_type(value: Any) -> str:
        entity_type = _compact_whitespace(value or "OTHER").upper()
        allowed_types = {
            "PERSON", "ORGANIZATION", "LOCATION", "PRODUCT", "PROJECT",
            "TECHNOLOGY", "CONCEPT", "TOPIC", "PREFERENCE", "OTHER",
        }
        return entity_type if entity_type in allowed_types else "OTHER"

    @classmethod
    def _normalize_entity_descriptors(
        cls,
        value: Any,
        *,
        limit: int = 16,
    ) -> List[Dict[str, str]]:
        """Normalize prompt entity objects without discarding their types."""
        if isinstance(value, str):
            raw = re.split(r"[,，;；\n]+", value)
        elif isinstance(value, list):
            raw = value
        else:
            raw = []
        descriptors: List[Dict[str, str]] = []
        for item in raw:
            if isinstance(item, dict):
                candidate = {
                    "name": _compact_whitespace(
                        item.get("name") or item.get("text") or "",
                    ),
                    "type": cls._normalize_entity_type(item.get("type")),
                }
            else:
                candidate = {
                    "name": _compact_whitespace(item),
                    "type": "OTHER",
                }
            if not cls._is_valid_entity_name(candidate["name"]):
                continue
            cls._merge_entity_descriptor(descriptors, candidate)
            if len(descriptors) >= limit:
                break
        return descriptors

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
            "user_claim": max(4, min(10, k)),
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
            "selected_limits": {
                "fact": k,
            },
        }
