"""Runtime adapter between application input events and memory storage.

``MemoryNodeManager`` owns episode persistence, reflection, and recall.  This
adapter owns the short-lived interaction buffer and converts frontend-shaped
turns or transcript segments into the manager's raw episode segments.
"""

from __future__ import annotations

import logging
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .embedding_client import EmbeddingClient
from .memory_database import SessionDB

from .memory_manager import (
    MemoryOperationReporter,
    MemoryNodeManager,
    _compact_whitespace,
    _now_text,
    _to_timestamp_text,
)
from .memory_segmentation import (
    OnlineSemanticSegmenter,
    OnlineSegmentUnit,
    TranscriptUnitAssembler,
    build_online_segmentation_config,
    build_transcript_aggregation_config,
    build_transcript_segmentation_config,
    convert_interaction_turn_to_online_unit,
)

class MemoryRuntime:
    """Normalize application input and batch it before episode storage."""

    def __init__(
        self,
        *,
        db_path: Path | str,
        memory_runtime_config: Optional[Dict[str, Any]] = None,
        memory_manager_config: Optional[Dict[str, Any]] = None,
        operation_reporter: Optional[MemoryOperationReporter] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Initialize memory storage, manager, and runtime batching.

        Application-facing callers provide ``db_path`` and the two memory
        config mappings. The runtime owns the ``SessionDB``, embedding client,
        and standard ``MemoryNodeManager`` lifecycle.
        """
        self._logger = logger or logging.getLogger(__name__)
        manager_config = dict(memory_manager_config or {})
        configured_embedding = manager_config.get("embedding")
        embedding_config = (
            dict(configured_embedding)
            if isinstance(configured_embedding, dict)
            else {}
        )

        database = SessionDB(Path(db_path).expanduser().resolve())
        try:
            manager_logger = self._logger.getChild("memory_manager")
            manager = MemoryNodeManager(
                database,
                embedding_config=dict(embedding_config or {}),
                memory_manager_config=manager_config,
                operation_reporter=operation_reporter,
                logger=manager_logger,
            )
        except Exception:
            database.close()
            raise
        self._memory_database = database
        self._memory_manager = manager
        runtime_config = dict(memory_runtime_config or {})
        self._prompt_language_mode = str(
            runtime_config.get("memory_prompt_language_mode")
            or "source"
        ).strip().lower()
        self._embedding_client = EmbeddingClient(
            dict(embedding_config or getattr(manager, "_embedding_cfg", {}) or {}),
        )
        if hasattr(self._memory_manager, "set_embedding_client"):
            self._memory_manager.set_embedding_client(self._embedding_client)
        interaction_segmentation_config = build_online_segmentation_config(runtime_config)
        self._interaction_segmenter = OnlineSemanticSegmenter(
            self._embedding_client,
            interaction_segmentation_config,
        )
        transcript_segmentation_config = build_transcript_segmentation_config(
            runtime_config,
        )
        self._transcript_segmenter = OnlineSemanticSegmenter(
            self._embedding_client,
            transcript_segmentation_config,
        )
        self._transcript_unit_assembler = TranscriptUnitAssembler(
            build_transcript_aggregation_config(runtime_config),
        )
        interaction_episode_config = runtime_config.get(
            "assistant_wakeup_segmentation",
        )
        if not isinstance(interaction_episode_config, dict):
            interaction_episode_config = {}
        allday_segmentation_config = runtime_config.get("allday_recording_segmentation")
        if not isinstance(allday_segmentation_config, dict):
            allday_segmentation_config = {}
        self._transcript_segmentation_log_decisions = bool(
            allday_segmentation_config.get("log_decisions", False),
        )
        self._interaction_episode_source_type = "assistant_wakeup"
        self._transcript_episode_source_type = "allday_recording"
        self._episode_summary_limits = {
            self._interaction_episode_source_type: {
                "max_duration_seconds": max(0.0, float(
                    interaction_episode_config.get(
                        "episode_summary_max_duration_seconds", 1800.0,
                    )
                )),
                "max_tokens": max(0, int(
                    interaction_episode_config.get("episode_summary_max_tokens", 6000)
                )),
                "min_tokens_for_duration": max(0, int(
                    interaction_episode_config.get(
                        "episode_summary_min_tokens_for_duration", 1000,
                    )
                )),
            },
            self._transcript_episode_source_type: {
                "max_duration_seconds": max(0.0, float(
                    allday_segmentation_config.get(
                        "episode_summary_max_duration_seconds", 3600.0,
                    )
                )),
                "max_tokens": max(0, int(
                    allday_segmentation_config.get("episode_summary_max_tokens", 8000)
                )),
                "min_tokens_for_duration": max(0, int(
                    allday_segmentation_config.get(
                        "episode_summary_min_tokens_for_duration", 1500,
                    )
                )),
            },
        }
        self._interaction_episode_tags: List[str] = []
        self._interaction_episode_prompt_language = "zh"
        self._interaction_has_pending_episode_sources = False
        self._interaction_episode_started_at: Optional[datetime] = None
        self._interaction_episode_latest_at: Optional[datetime] = None
        self._interaction_episode_token_count = 0

        self._transcript_episode_tags: List[str] = []
        self._transcript_episode_prompt_language = "zh"
        self._transcript_has_pending_episode_sources = False
        self._transcript_episode_started_at: Optional[datetime] = None
        self._transcript_episode_latest_at: Optional[datetime] = None
        self._transcript_episode_token_count = 0

    def close(self, timeout: Optional[float] = 30.0) -> None:
        """Drain owned tasks and release resources created by this runtime."""
        try:
            self.flush_pending_memory_inputs(timeout=timeout)
        finally:
            try:
                shutdown_ok = self._memory_manager.shutdown_task_worker(
                    wait=True,
                    timeout=timeout,
                )
                if not shutdown_ok:
                    # A bounded shutdown timeout must not allow the database
                    # to close while a reflection transaction is still in
                    # flight. Continue draining without a deadline so all
                    # queued writes are committed before the connection closes.
                    self._logger.warning(
                        "Memory worker did not stop within timeout=%s; "
                        "continuing to drain queued tasks before closing database",
                        timeout,
                    )
                    shutdown_ok = self._memory_manager.shutdown_task_worker(
                        wait=True,
                        timeout=None,
                    )
                if not shutdown_ok:
                    raise RuntimeError(
                        "Memory worker stopped before completing all queued tasks"
                    )
            finally:
                if self._memory_database is not None:
                    self._memory_database.close()

    @property
    def manager(self) -> MemoryNodeManager:
        """Return the manager owned by this runtime for diagnostics/adapters."""
        return self._memory_manager

    @property
    def database(self) -> SessionDB:
        """Return the database used by the owned manager."""
        if self._memory_database is not None:
            return self._memory_database
        return self._memory_manager._db

    def accept_single_interaction_turn(
        self,
        user_message: str,
        assistant_response: str = "",
        *,
        tags: Optional[List[str]] = None,
        turn_timestamp: Optional[Any] = None,
        **extra: Any,
    ) -> Dict[str, Any]:
        """Buffer one interaction and store a complete batch when due."""
        if turn_timestamp is None:
            turn_timestamp = extra.get("timestamp")
        turn = {
            "user_message": _compact_whitespace(user_message),
            "assistant_response": _compact_whitespace(assistant_response),
            "tags": list(tags or []),
            "turn_timestamp": _to_timestamp_text(turn_timestamp) or _now_text(),
        }
        self._logger.info(
            "memory runtime received interaction turn timestamp=%s tags=%s "
            "user_chars=%s assistant_chars=%s user_message=%s assistant_response=%s",
            turn["turn_timestamp"],
            turn["tags"],
            len(turn["user_message"]),
            len(turn["assistant_response"]),
            turn["user_message"],
            turn["assistant_response"],
        )
        if not self._memory_manager.enabled:
            return {"queued": False, "reason": "memory_disabled"}
        if not turn["user_message"] and not turn["assistant_response"]:
            return {"queued": False, "reason": "empty_turn"}

        pending_units = self._interaction_segmenter.pending_unit_snapshot()
        incoming_unit = convert_interaction_turn_to_online_unit(
            turn,
            len(pending_units) + 1,
        )
        append_report = self._append_interaction_turn_unit(incoming_unit)
        return {
            "queued": bool(append_report.get("queued")),
            "reason": str(append_report.get("reason") or ""),
        }

    def _append_interaction_turn_unit(
        self,
        unit: OnlineSegmentUnit,
    ) -> Dict[str, Any]:
        """Append one interaction unit, flushing the prior batch at a boundary."""
        queued = False
        reason = "threshold_not_reached"
        pending_units = self._interaction_segmenter.pending_unit_snapshot()
        incoming_embedding = self._interaction_segmenter.embed_unit(
            unit,
        ).embedding
        if pending_units:
            should_finalize, boundary_decision = (
                self._interaction_segmenter.should_finalize_pending_units(
                    unit,
                    incoming_embedding,
                )
            )
            if should_finalize:
                flush_report = self._flush_pending_interaction_turns()
                queued = bool(flush_report.get("queued")) or queued
                reason = (
                    ""
                    if queued
                    else str(
                        flush_report.get("reason") or boundary_decision.reason
                    )
                )
                if self._interaction_segmenter.has_pending_units():
                    return {
                        "accepted": False,
                        "queued": queued,
                        "reason": reason,
                    }

        self._interaction_segmenter.append_pending_unit(
            unit,
            incoming_embedding,
        )
        return {
            "accepted": True,
            "queued": queued,
            "reason": "" if queued else reason,
        }

    def _flush_pending_interaction_turns(
        self,
        *,
        evaluate_episode_summary: bool = True,
    ) -> Dict[str, Any]:
        """Submit the buffered interaction turns and clear them when queued."""
        if not self._interaction_segmenter.has_pending_units():
            return {"queued": False, "reason": "no_pending_turns"}
        queue_report = self._trigger_memory_store_task_for_pending_interaction(
            evaluate_episode_summary=evaluate_episode_summary,
        )
        queued = bool(queue_report.get("queued"))
        if queued:
            self._interaction_segmenter.clear_pending_units()
        return {
            "queued": bool(queue_report.get("queued")),
            "reason": str(queue_report.get("reason") or ""),
            "episode_summary": queue_report.get("episode_summary"),
        }

    def accept_single_transcript_segment(
        self,
        segment: Dict[str, Any],
        *,
        source_type: str = "allday_recording",
        tags: Optional[List[str]] = None,
        is_last_segment: bool = False,
    ) -> Dict[str, Any]:
        """Buffer one transcript segment and queue completed transcript episodes."""
        if not self._memory_manager.enabled:
            return {"queued": False, "reason": "memory_disabled"}
        if not isinstance(segment, dict):
            return {"queued": False, "reason": "invalid_segment"}
        normalized_segment = self._normalize_single_transcript_segment(segment)
        if normalized_segment is None:
            return {"queued": False, "reason": "empty_segment"}
        self._logger.info(
            "memory runtime received transcript segment source_type=%s is_last_segment=%s "
            "speaker=%s started_at=%s ended_at=%s text_chars=%s text=%s",
            source_type,
            is_last_segment,
            normalized_segment["speaker"],
            normalized_segment["started_at"],
            normalized_segment["ended_at"],
            len(normalized_segment["text"]),
            normalized_segment["text"],
        )

        context = {
            "source_type": str(source_type or "allday_recording"),
            "tags": sorted(
                {
                    str(tag)
                    for tag in tags or []
                    if tag is not None and str(tag).strip()
                }
            ),
        }
        normalized_segment["_memory_context"] = dict(context)
        self._transcript_episode_source_type = context["source_type"]
        self._transcript_episode_tags = sorted(set(self._transcript_episode_tags).union(context["tags"]))
        queued = False
        completed_unit = self._transcript_unit_assembler.append_new_segment(
            normalized_segment,
        )
        if completed_unit is None:
            final_summary_report = None
            if is_last_segment:
                final_flush_report = self._flush_pending_transcript_segments(
                    reason="last_segment",
                    evaluate_episode_summary=False,
                )
                queued = bool(final_flush_report.get("queued")) or queued
                final_summary_report = self.trigger_memory_episode_summary(reason="last_segment")
            return {
                "queued": queued or bool((final_summary_report or {}).get("queued")),
                "reason": "threshold_not_reached",
                "episode_summary": final_summary_report,
            }
        append_report = self._append_transcript_semantic_unit(completed_unit)
        queued = bool(append_report.get("queued")) or queued
        if not append_report.get("accepted"):
            return {
                "queued": queued,
                "reason": str(append_report.get("reason") or "queue_rejected"),
            }
        if is_last_segment:
            final_flush_report = self._flush_pending_transcript_segments(
                reason="last_segment",
                evaluate_episode_summary=False,
            )
            queued = bool(final_flush_report.get("queued")) or queued
            final_summary_report = self.trigger_memory_episode_summary(reason="last_segment")
            queued = queued or bool(final_summary_report.get("queued"))
        else:
            final_summary_report = None
        return {
            "queued": queued,
            "reason": "" if queued else "threshold_not_reached",
            "episode_summary": final_summary_report,
        }

    def should_trigger_episode_summary(
        self,
        current_segment: Dict[str, Any],
        previous_segment: Optional[Dict[str, Any]] = None,
        *,
        source_type: Optional[str] = None,
        is_last_segment: bool = False,
    ) -> bool:
        """Return whether an input would exceed its active episode window.

        This deliberately uses only bounded duration and accumulated content.
        It is called after a complete store batch was accepted, so the batch
        that reaches the threshold belongs to the episode being summarized.
        ``previous_segment`` remains for call compatibility; gaps are no
        longer an episode trigger.
        """
        if is_last_segment:
            return True
        resolved_source_type = str(
            source_type
            or current_segment.get("source_type")
            or self._transcript_episode_source_type
        ).strip()
        is_interaction = resolved_source_type == self._interaction_episode_source_type
        limits = self._episode_summary_limits[
            self._interaction_episode_source_type
            if is_interaction
            else self._transcript_episode_source_type
        ]
        started_at = (
            self._interaction_episode_started_at
            if is_interaction
            else self._transcript_episode_started_at
        )
        accumulated_tokens = (
            self._interaction_episode_token_count
            if is_interaction
            else self._transcript_episode_token_count
        )
        if accumulated_tokens <= 0:
            return False

        max_tokens = int(limits["max_tokens"])
        if max_tokens > 0 and accumulated_tokens >= max_tokens:
            return True

        if started_at is None:
            return False
        max_duration_seconds = float(limits["max_duration_seconds"])
        if max_duration_seconds <= 0:
            return False
        current_time = (
            self._episode_summary_segment_time(current_segment)
            or (
                self._interaction_episode_latest_at
                if is_interaction
                else self._transcript_episode_latest_at
            )
        )
        if current_time is None:
            return False
        elapsed_seconds = max(0.0, (current_time - started_at).total_seconds())
        return (
            elapsed_seconds >= max_duration_seconds
            and accumulated_tokens >= int(limits["min_tokens_for_duration"])
        )

    @staticmethod
    def _episode_summary_segment_token_count(segment: Dict[str, Any]) -> int:
        text = "\n".join(
            _compact_whitespace(segment.get(field) or "")
            for field in ("text", "user_message", "assistant_response")
        ).strip()
        return len(re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9_.$'-]+|[^\s]", text))

    def _episode_summary_segment_time(
        self,
        segment: Dict[str, Any],
    ) -> Optional[datetime]:
        for field in (
            "turn_timestamp",
            "started_at",
            "ended_at",
            "timestamp",
            "start_time",
            "end_time",
        ):
            parsed = self._parse_runtime_timestamp(segment.get(field))
            if parsed is not None:
                return parsed
        return None

    def _record_episode_summary_input(
        self,
        source_type: str,
        segment: Dict[str, Any],
    ) -> None:
        token_count = self._episode_summary_segment_token_count(segment)
        if token_count <= 0:
            return
        started_at = self._episode_summary_segment_time(segment)
        if source_type == self._interaction_episode_source_type:
            if self._interaction_episode_started_at is None:
                self._interaction_episode_started_at = started_at
            if started_at is not None:
                self._interaction_episode_latest_at = started_at
            self._interaction_episode_token_count += token_count
            return
        if self._transcript_episode_started_at is None:
            self._transcript_episode_started_at = started_at
        if started_at is not None:
            self._transcript_episode_latest_at = started_at
        self._transcript_episode_token_count += token_count

    def _record_episode_summary_inputs(
        self,
        source_type: str,
        segments: Sequence[Dict[str, Any]],
    ) -> None:
        for segment in segments:
            if isinstance(segment, dict):
                self._record_episode_summary_input(source_type, segment)

    def _reset_episode_summary_window(self, source_type: str) -> None:
        if source_type == self._interaction_episode_source_type:
            self._interaction_episode_started_at = None
            self._interaction_episode_latest_at = None
            self._interaction_episode_token_count = 0
            return
        self._transcript_episode_started_at = None
        self._transcript_episode_latest_at = None
        self._transcript_episode_token_count = 0

    def _parse_runtime_timestamp(self, value: Any) -> Optional[datetime]:
        text = _to_timestamp_text(value)
        if not text:
            return None
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=datetime.now().astimezone().tzinfo)
            return parsed
        except (TypeError, ValueError):
            return None

    def trigger_memory_episode_summary(
        self,
        *,
        reason: str = "explicit",
        source_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        prompt_language: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Queue one completed source episode and its prospective update.

        Callers must first submit the episode's final store batch.  The
        manager's FIFO queue then preserves store, summary, and prospective
        update ordering without this function reaching back into runtime
        input buffers.
        """
        resolved_source_type = str(
            source_type or self._transcript_episode_source_type
        ).strip() or "allday_recording"
        is_transcript_episode = (
            resolved_source_type == self._transcript_episode_source_type
        )
        is_interaction_episode = (
            resolved_source_type == self._interaction_episode_source_type
        )
        default_tags = (
            self._transcript_episode_tags
            if is_transcript_episode
            else self._interaction_episode_tags
        )
        default_prompt_language = (
            self._transcript_episode_prompt_language
            if is_transcript_episode
            else self._interaction_episode_prompt_language
        )
        resolved_tags = list(default_tags if tags is None else tags or [])
        resolved_prompt_language = str(
            prompt_language or default_prompt_language
        ).strip() or "zh"

        if is_transcript_episode:
            has_pending_episode_sources = self._transcript_has_pending_episode_sources
        elif is_interaction_episode:
            has_pending_episode_sources = self._interaction_has_pending_episode_sources
        else:
            has_pending_episode_sources = False

        if not has_pending_episode_sources:
            return {
                "queued": False,
                "reason": "no_pending_episode_sources",
                "trigger_reason": reason,
            }
        report = self._memory_manager.submit_memory_episode_summary_task(
            source_type=resolved_source_type,
            tags=resolved_tags,
            prompt_language=resolved_prompt_language,
        )
        if bool(report.get("queued")):
            report["prospective_update"] = (
                self._memory_manager.submit_memory_prospective_update_task()
            )
            if is_transcript_episode:
                self._transcript_has_pending_episode_sources = False
                self._transcript_episode_tags = []
                self._reset_episode_summary_window(resolved_source_type)
            elif is_interaction_episode:
                self._interaction_has_pending_episode_sources = False
                self._interaction_episode_tags = []
                self._reset_episode_summary_window(resolved_source_type)
        else:
            report["prospective_update"] = {
                "queued": False,
                "reason": "episode_summary_not_queued",
            }
        report["trigger_reason"] = reason
        return report

    def _append_transcript_semantic_unit(
        self,
        unit: OnlineSegmentUnit,
        *,
        evaluate_episode_summary: bool = True,
    ) -> Dict[str, Any]:
        """Use one assembled utterance to decide whether the prior episode ends."""
        queued = False
        incoming_embedding = self._transcript_segmenter.embed_unit(
            unit,
        ).embedding
        pending_units = self._transcript_segmenter.pending_unit_snapshot()
        if pending_units:
            should_finalize, decision = (
                self._transcript_segmenter.should_finalize_pending_units(
                    unit,
                    incoming_embedding,
                )
            )
            self._log_transcript_segmentation_decision(
                unit,
                decision=decision,
            )
            store_report = (
                self._trigger_memory_store_task_for_pending_transcript(
                    reason=decision.reason,
                    evaluate_episode_summary=evaluate_episode_summary,
                )
                if should_finalize
                else None
            )
            if store_report is not None:
                queued = bool(store_report.get("queued")) or queued
                if self._transcript_segmenter.has_pending_units():
                    return {
                        "accepted": False,
                        "queued": queued,
                        "reason": str(store_report.get("reason") or "queue_rejected"),
                    }
        else:
            _, decision = self._transcript_segmenter.should_finalize_pending_units(
                unit,
                incoming_embedding,
            )
            self._log_transcript_segmentation_decision(unit, decision=decision)

        self._transcript_segmenter.append_pending_unit(
            unit,
            incoming_embedding,
        )
        return {"accepted": True, "queued": queued, "reason": ""}

    def _flush_pending_transcript_segments(
        self,
        *,
        reason: str = "explicit_flush",
        evaluate_episode_summary: bool = True,
    ) -> Dict[str, Any]:
        """Finalize the current utterance, then submit pending transcript units."""
        queued = False
        completed_unit = self._transcript_unit_assembler.flush()
        if completed_unit is not None:
            append_report = self._append_transcript_semantic_unit(
                completed_unit,
                evaluate_episode_summary=evaluate_episode_summary,
            )
            queued = bool(append_report.get("queued")) or queued
            if not append_report.get("accepted"):
                return {
                    "queued": queued,
                    "reason": str(append_report.get("reason") or "queue_rejected"),
                }
        store_report = self._trigger_memory_store_task_for_pending_transcript(
            reason=reason,
            evaluate_episode_summary=evaluate_episode_summary,
        )
        return {
            "queued": bool(store_report.get("queued")) or queued,
            "reason": str(store_report.get("reason") or ""),
        }

    def _trigger_memory_store_task_for_pending_transcript(
        self,
        *,
        reason: str,
        evaluate_episode_summary: bool = True,
    ) -> Dict[str, Any]:
        """Normalize and submit transcript spans held by the semantic segmenter."""
        pending_units = self._transcript_segmenter.pending_unit_snapshot()
        if not pending_units:
            return {"queued": False, "reason": "no_pending_segments"}
        segments = self._transcript_raw_segments_from_units(pending_units)
        context = self._transcript_context_from_units(pending_units)
        raw_segments, prompt_language = self._normalize_transcript_segments_into_memory_raw_segments(
            segments,
        )
        if not raw_segments:
            self._transcript_segmenter.clear_pending_units()
            return {"queued": False, "reason": "invalid_pending_segments"}
        self._log_info(
            "memory_runtime",
            "transcript_batch_detail",
            {
                "reason": reason,
                "source_type": context.get("source_type"),
                "tags": context.get("tags") or [],
                "raw_segment_count": len(segments),
                "semantic_unit_count": len(pending_units),
                "prompt_language": prompt_language,
                "segments": raw_segments,
            },
        )
        tags = {
            str(tag)
            for tag in context.get("tags") or []
            if tag is not None and str(tag).strip()
        }
        for segment in raw_segments:
            tags.update(
                str(tag)
                for tag in segment.get("tags") or []
                if tag is not None and str(tag).strip()
            )
        queue_report = self._memory_manager.submit_memory_store_task(
            raw_segments=raw_segments,
            source_type=str(context.get("source_type") or "allday_recording"),
            tags=sorted(tags),
            prompt_language=prompt_language,
        )
        self._transcript_episode_prompt_language = prompt_language
        queued = bool(queue_report.get("queued"))
        episode_summary_report = None
        if queued:
            self._transcript_has_pending_episode_sources = True
            resolved_source_type = str(
                context.get("source_type") or "allday_recording"
            )
            self._record_episode_summary_inputs(
                resolved_source_type,
                raw_segments,
            )
            self._logger.info(
                "transcript episode queued reason=%s raw_segment_count=%s semantic_unit_count=%s",
                reason,
                len(segments),
                len(pending_units),
            )
            self._transcript_segmenter.clear_pending_units()
            if evaluate_episode_summary and self.should_trigger_episode_summary(
                raw_segments[-1],
                source_type=resolved_source_type,
            ):
                episode_summary_report = self.trigger_memory_episode_summary(
                    reason="episode_limit",
                    source_type=resolved_source_type,
                )
        return {
            "queued": queued or bool((episode_summary_report or {}).get("queued")),
            "reason": (
                "episode_limit"
                if (episode_summary_report or {}).get("queued")
                else ""
                if queued
                else str(queue_report.get("reason") or "queue_rejected")
            ),
            "episode_summary": episode_summary_report,
        }

    @staticmethod
    def _transcript_raw_segments_from_units(
        units: Sequence[OnlineSegmentUnit],
    ) -> List[Dict[str, Any]]:
        """Expand the original ASR spans retained by transcript semantic units."""
        segments: List[Dict[str, Any]] = []
        for unit in units:
            raw = unit.raw if isinstance(unit.raw, dict) else {}
            raw_segments = raw.get("raw_segments")
            if not isinstance(raw_segments, list):
                continue
            segments.extend(
                dict(segment)
                for segment in raw_segments
                if isinstance(segment, dict)
            )
        return segments

    @staticmethod
    def _transcript_context_from_units(
        units: Sequence[OnlineSegmentUnit],
    ) -> Dict[str, Any]:
        """Derive one storage context from pending transcript semantic units."""
        source_type = "allday_recording"
        tags: set[str] = set()
        for unit in units:
            raw = unit.raw if isinstance(unit.raw, dict) else {}
            raw_segments = raw.get("raw_segments")
            if not isinstance(raw_segments, list):
                continue
            for segment in raw_segments:
                if not isinstance(segment, dict):
                    continue
                context = segment.get("_memory_context")
                if not isinstance(context, dict):
                    continue
                resolved_source_type = str(context.get("source_type") or "").strip()
                if resolved_source_type:
                    source_type = resolved_source_type
                tags.update(
                    str(tag)
                    for tag in context.get("tags") or []
                    if tag is not None and str(tag).strip()
                )
        return {"source_type": source_type, "tags": sorted(tags)}

    def _log_transcript_segmentation_decision(
        self,
        unit: OnlineSegmentUnit,
        *,
        decision: Optional[Any] = None,
        reason: Optional[str] = None,
    ) -> None:
        """Emit per-unit semantic boundary details when transcript logging is enabled."""
        if not self._transcript_segmentation_log_decisions:
            return
        raw = unit.raw if isinstance(unit.raw, dict) else {}
        raw_segments = raw.get("raw_segments") or []
        speaker_labels = raw.get("speaker_labels") or []
        resolved_reason = str(reason or getattr(decision, "reason", "append"))
        self._logger.info(
            "transcript segmentation decision unit=%s reason=%s raw_segment_count=%s "
            "token_count=%s speakers=%s cut_probability=%s score=%s "
            "semantic_surprise=%s cohesion_drop=%s time_gap_seconds=%s "
            "scoring_mode=%s rolling_tail_units=%s rolling_tail_tokens=%s text=%s",
            unit.index,
            resolved_reason,
            len(raw_segments),
            unit.token_count,
            ",".join(str(label) for label in speaker_labels),
            getattr(decision, "cut_probability", None),
            getattr(decision, "score", None),
            getattr(decision, "semantic_surprise", None),
            getattr(decision, "cohesion_drop", None),
            getattr(decision, "time_gap_seconds", None),
            getattr(decision, "scoring_mode", None),
            getattr(decision, "rolling_window_tail_units", None),
            getattr(decision, "rolling_window_tail_tokens", None),
            unit.text[:240],
        )

    def _log_info(self, scope: str, event: str, payload: Dict[str, Any]) -> None:
        """Emit a structured JSON log record for runtime diagnostics."""
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

    def trigger_memory_reflect(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Queue reflection after pending interaction and transcript storage."""
        interaction_flush_report = self._flush_pending_interaction_turns()
        transcript_flush_report = self._flush_pending_transcript_segments(
            reason="reflect",
        )
        report = self._memory_manager.submit_memory_reflect_task(*args, **kwargs) or {}
        report["pending_interaction_flush"] = interaction_flush_report
        report["pending_transcript_flush"] = transcript_flush_report
        
        return report
    
    def trigger_memory_recall(
        self,
        query: str,
        *,
        tags: Optional[List[str]] = None,
        time_end: Optional[str] = None,
        prompt_language: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run recall immediately through the manager without adding buffering."""
        resolved_prompt_language = str(prompt_language or "").strip().lower()
        if not resolved_prompt_language:
            resolved_prompt_language = self._resolve_prompt_language_from_segments(
                [{"text": str(query or "")}]
            )
        return self._memory_manager.process_memory_recall_immediately(
            query=str(query or ""),
            tags=tags,
            time_end=time_end,
            prompt_language=resolved_prompt_language,
        )

    def flush_pending_memory_inputs(
        self,
        timeout: Optional[float] = None,
        *,
        evaluate_episode_summary: bool = True,
    ) -> bool:
        """Submit runtime-buffered inputs without waiting for manager tasks."""
        if self._interaction_segmenter.has_pending_units():
            interaction_report = self._flush_pending_interaction_turns(
                evaluate_episode_summary=evaluate_episode_summary,
            )
            if not interaction_report.get("queued") and interaction_report.get("reason") not in {"", "no_pending_segments"}:
                return False
        transcript_flush_report = self._flush_pending_transcript_segments(
            reason="explicit_input_boundary",
            evaluate_episode_summary=evaluate_episode_summary,
        )
        return not (
            not transcript_flush_report.get("queued")
            and transcript_flush_report.get("reason") not in {"", "no_pending_segments"}
        )

    def wait_for_memory_tasks(self, timeout: Optional[float] = None) -> bool:
        """Wait until all tasks already submitted to the memory manager complete."""
        return self._memory_manager.flush_task_queue(timeout=timeout)

    def get_pending_interaction_turns(self) -> List[Dict[str, Any]]:
        """Return raw turns currently held by the interaction segmenter."""
        turns: List[Dict[str, Any]] = []
        for unit in self._interaction_segmenter.pending_unit_snapshot():
            if isinstance(unit.raw, dict):
                turns.append(dict(unit.raw))
        return turns

    def has_pending_interaction_turns(self) -> bool:
        """Return whether interaction turns are waiting for storage."""
        return self._interaction_segmenter.has_pending_units()

    def _trigger_memory_store_task_for_pending_interaction(
        self,
        *,
        evaluate_episode_summary: bool = True,
    ) -> Dict[str, Any]:
        """Read pending turns, then submit their memory-store task."""
        turns = self.get_pending_interaction_turns()
        (
            raw_segments,
            prompt_language,
        ) = self._normalize_interaction_turns_to_memory_raw_segments(turns)
        tags = sorted({tag for turn in turns for tag in turn.get("tags", [])})
        self._log_info(
            "memory_runtime",
            "interaction_batch_detail",
            {
                "source_type": "assistant_wakeup",
                "tags": tags,
                "raw_segment_count": len(raw_segments),
                "semantic_unit_count": len(
                    self._interaction_segmenter.pending_unit_snapshot()
                ),
                "prompt_language": prompt_language,
                "segments": raw_segments,
            },
        )
        queue_report = self._memory_manager.submit_memory_store_task(
            raw_segments=raw_segments,
            source_type="assistant_wakeup",
            tags=tags,
            prompt_language=prompt_language,
        )
        queued = bool(queue_report.get("queued"))
        episode_summary_report = None
        if queued:
            self._interaction_has_pending_episode_sources = True
            self._interaction_episode_tags = sorted(
                set(self._interaction_episode_tags).union(tags)
            )
            self._interaction_episode_prompt_language = prompt_language
            self._record_episode_summary_inputs(
                self._interaction_episode_source_type,
                turns,
            )
            if (
                evaluate_episode_summary
                and turns
                and self.should_trigger_episode_summary(
                    turns[-1],
                    source_type=self._interaction_episode_source_type,
                )
            ):
                episode_summary_report = self.trigger_memory_episode_summary(
                    reason="episode_limit",
                    source_type=self._interaction_episode_source_type,
                )
        return {
            "queued": queued or bool((episode_summary_report or {}).get("queued")),
            "reason": (
                "episode_limit"
                if (episode_summary_report or {}).get("queued")
                else ""
                if queued
                else str(queue_report.get("reason") or "queue_rejected")
            ),
            "episode_summary": episode_summary_report,
        }

    def _resolve_prompt_language_from_segments(
        self,
        records: Sequence[Dict[str, Any]],
    ) -> str:
        """Resolve prompt language for normalized segments or interaction turns."""
        mode = self._prompt_language_mode
        if mode in {"en", "english", "force_en"}:
            return "en"
        if mode in {"zh", "chinese", "force_zh"}:
            return "zh"
        sample = "\n".join(
            text
            for record in list(records)[:12]
            if isinstance(record, dict)
            for text in (
                str(record.get("text") or "").strip()
                or str(record.get("user_message") or "").strip()
                or str(record.get("assistant_response") or "").strip(),
            )
            if text
        )
        return "zh" if re.search(r"[\u4e00-\u9fff]", sample) else "en"

    @staticmethod
    def _transcript_segment_text(segment: Dict[str, Any]) -> str:
        """Extract and compact the transcript text from supported input fields."""
        return _compact_whitespace(
            segment.get("text")
            or segment.get("asr_text")
            or segment.get("reference_text")
            or segment.get("utterance")
            or ""
        )

    @staticmethod
    def _transcript_segment_start_time(segment: Dict[str, Any]) -> str:
        """Extract the normalized start timestamp from a transcript segment."""
        return _to_timestamp_text(
            segment.get("started_at")
            or segment.get("start_timestamp")
            or segment.get("timestamp")
            or segment.get("start")
        )

    @staticmethod
    def _transcript_segment_end_time(segment: Dict[str, Any]) -> str:
        """Extract the normalized end timestamp, falling back to the start time."""
        started_at = MemoryRuntime._transcript_segment_start_time(segment)
        return _to_timestamp_text(
            segment.get("ended_at")
            or segment.get("end_timestamp")
            or segment.get("timestamp_end")
            or segment.get("end")
            or started_at
        )

    def _normalize_interaction_turns_to_memory_raw_segments(
        self,
        turns: Sequence[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], str]:
        """Convert frontend interaction turns into manager-compatible segments."""

        prompt_language = self._resolve_prompt_language_from_segments(turns)
        segments: List[Dict[str, Any]] = []
        for turn_index, turn in enumerate(turns, 1):
            timestamp = _to_timestamp_text(turn.get("turn_timestamp")) or _now_text()
            tags = list(turn.get("tags") or [])
            user_text = _compact_whitespace(turn.get("user_message") or "")
            assistant_text = _compact_whitespace(turn.get("assistant_response") or "")
            if user_text:
                segments.append({
                    "speaker": "user" if prompt_language == "en" else "用户",
                    "text": user_text,
                    "started_at": timestamp,
                    "ended_at": timestamp,
                    "tags": tags,
                    "turn_index": turn_index,
                })
            if assistant_text:
                segments.append({
                    "speaker": "assistant" if prompt_language == "en" else "助手",
                    "text": assistant_text,
                    "started_at": timestamp,
                    "ended_at": timestamp,
                    "tags": tags,
                    "turn_index": turn_index,
                })
        return segments, prompt_language

    def _normalize_transcript_segments_into_memory_raw_segments(
        self,
        segments: Sequence[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], str]:
        """Validate and normalize transcript segments for episode storage."""
        normalized: List[Dict[str, Any]] = []
        for index, segment in enumerate(segments, 1):
            normalized_segment = self._normalize_single_transcript_segment(
                segment,
                fallback_index=index,
            )
            if normalized_segment is not None:
                normalized.append(normalized_segment)
        normalized = sorted(
            normalized,
            key=lambda item: (
                str(item.get("started_at") or ""),
                str(item.get("ended_at") or ""),
                str(item.get("speaker") or ""),
            ),
        )
        return normalized, self._resolve_prompt_language_from_segments(normalized)

    def _normalize_single_transcript_segment(
        self,
        segment: Any,
        *,
        fallback_index: int = 1,
    ) -> Optional[Dict[str, Any]]:
        """Normalize one frontend transcript record before utterance assembly."""
        if not isinstance(segment, dict):
            return None
        text = self._transcript_segment_text(segment)
        if not text:
            return None
        speaker = _compact_whitespace(
            segment.get("speaker")
            or segment.get("speaker_name")
            or segment.get("speaker_id")
            or "unknown_speaker"
        )
        started_at = self._transcript_segment_start_time(segment)
        ended_at = self._transcript_segment_end_time(segment)
        try:
            segment_index = int(segment.get("segment_index") or fallback_index)
        except (TypeError, ValueError):
            segment_index = fallback_index
        return {
            "speaker": speaker or "unknown_speaker",
            "text": text,
            "started_at": started_at or _now_text(),
            "ended_at": ended_at or started_at or _now_text(),
            "tags": list(segment.get("tags") or []),
            "segment_index": segment_index,
            "metadata": dict(segment.get("metadata") or {}),
        }
