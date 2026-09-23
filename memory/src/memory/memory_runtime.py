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
from .memory_context_manager import (
    MemoryContextManager,
    MemoryUnit,
    TranscriptUnitAssembler,
    build_episode_summary_config,
    build_online_segmentation_config,
    build_transcript_aggregation_config,
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
        memory_context_manager_config = build_online_segmentation_config(
            runtime_config,
        )
        self._transcript_unit_assembler = TranscriptUnitAssembler(
            build_transcript_aggregation_config(runtime_config),
        )
        self._transcript_ambient_recording_enabled = False
        self._memory_context_manager = MemoryContextManager(
            self._embedding_client,
            memory_context_manager_config,
            build_episode_summary_config(runtime_config),
        )
        self._transcript_segmentation_log_decisions = bool(
            runtime_config.get("log_segmentation_decisions", False),
        )
        self._episode_tags: List[str] = []
        self._episode_prompt_language = "zh"
        self._has_pending_episode_sources = False

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

    def accept_memory_input(
        self,
        *,
        interaction_turn: Optional[Dict[str, Any]] = None,
        transcript_segments: Optional[Sequence[Dict[str, Any]]] = None,
        tags: Optional[List[str]] = None,
        is_last_segment: bool = False,
        ambient_recording_enabled: bool = False,
    ) -> Dict[str, Any]:
        """Accept one interaction turn or one batch of transcript segments.

        Both input kinds enter one shared semantic buffer.  Transcript
        fragments first pass through :class:`TranscriptUnitAssembler`.
        ``ambient_recording_enabled`` only controls whether a detected fact
        boundary can submit immediately or must wait for ambient-ASR coverage.

        Exactly one of ``interaction_turn`` and ``transcript_segments`` must
        be supplied.  Transcript callers should pass the complete ASR batch;
        the runtime processes it in order and only completed assembled units
        are eligible for semantic boundary scoring.
        """
        has_interaction = isinstance(interaction_turn, dict)
        has_transcripts = transcript_segments is not None
        if has_interaction == has_transcripts:
            return {"queued": False, "reason": "expected_one_input_kind"}
        if not self._memory_manager.enabled:
            return {"queued": False, "reason": "memory_disabled"}

        normalized_tags = sorted({
            str(tag)
            for tag in tags or []
            if tag is not None and str(tag).strip()
        })
        if has_interaction:
            return self._accept_interaction_memory_input(
                dict(interaction_turn or {}),
                tags=normalized_tags,
                ambient_recording_enabled=ambient_recording_enabled,
            )
        return self._accept_transcript_memory_inputs(
            transcript_segments or [],
            tags=normalized_tags,
            is_last_segment=is_last_segment,
            ambient_recording_enabled=ambient_recording_enabled,
        )

    def update_ambient_asr_watermark(
        self,
        value: Any,
        *,
        evaluate_episode_summary: bool = True,
    ) -> Dict[str, Any]:
        """Advance ambient ASR coverage and process newly released input units."""
        self._memory_context_manager.update_ambient_asr_watermark(value)
        return self._process_awaiting_ambient_units(
            evaluate_episode_summary=evaluate_episode_summary,
        )

    def _accept_interaction_memory_input(
        self,
        interaction_turn: Dict[str, Any],
        *,
        tags: Sequence[str],
        ambient_recording_enabled: bool,
    ) -> Dict[str, Any]:
        """Normalize and append one delivered user/assistant interaction."""
        turn_timestamp = _to_timestamp_text(
            interaction_turn.get("turn_timestamp")
            or interaction_turn.get("timestamp")
        ) or _now_text()
        turn = {
            "user_message": _compact_whitespace(interaction_turn.get("user_message") or ""),
            "assistant_response": _compact_whitespace(interaction_turn.get("assistant_response") or ""),
            "tags": list(tags),
            "turn_timestamp": turn_timestamp,
        }
        if not turn["user_message"] and not turn["assistant_response"]:
            return {"queued": False, "reason": "empty_turn"}
        self._logger.info(
            "memory runtime received interaction turn ambient_recording_enabled=%s "
            "timestamp=%s tags=%s user_chars=%s assistant_chars=%s "
            "user_message=%s assistant_response=%s",
            ambient_recording_enabled,
            turn["turn_timestamp"],
            turn["tags"],
            len(turn["user_message"]),
            len(turn["assistant_response"]),
            turn["user_message"],
            turn["assistant_response"],
        )
        unit = convert_interaction_turn_to_online_unit(turn)
        append_report = self._append_memory_input_unit(
            unit,
            ambient_recording_enabled=ambient_recording_enabled,
        )
        return {
            "queued": bool(append_report.get("queued")),
            "reason": str(append_report.get("reason") or ""),
        }

    def _accept_transcript_memory_inputs(
        self,
        transcript_segments: Sequence[Dict[str, Any]],
        *,
        tags: Sequence[str],
        is_last_segment: bool,
        ambient_recording_enabled: bool,
    ) -> Dict[str, Any]:
        """Assemble and append one ordered ASR batch to the shared buffer."""
        normalized_segments = self._normalize_transcript_segments_for_input(
            transcript_segments,
        )
        if not normalized_segments:
            return {"queued": False, "reason": "empty_transcript_batch"}
        self._transcript_ambient_recording_enabled = bool(
            ambient_recording_enabled
        )
        for normalized_segment in normalized_segments:
            normalized_segment["tags"] = sorted({
                *normalized_segment.get("tags", []),
                *tags,
            })
            self._logger.info(
                "memory runtime received transcript segment ambient_recording_enabled=%s "
                "speaker=%s started_at=%s ended_at=%s "
                "text_chars=%s text=%s",
                ambient_recording_enabled,
                normalized_segment["speaker"],
                normalized_segment["started_at"],
                normalized_segment["ended_at"],
                len(normalized_segment["text"]),
                normalized_segment["text"],
            )

        self._episode_tags = sorted(
            set(self._episode_tags).union(tags)
        )
        queued = False
        for normalized_segment in normalized_segments:
            completed_unit = self._transcript_unit_assembler.append_new_segment(
                normalized_segment,
            )
            if completed_unit is None:
                continue
            append_report = self._append_memory_input_unit(
                completed_unit,
                ambient_recording_enabled=ambient_recording_enabled,
            )
            queued = bool(append_report.get("queued")) or queued
            if append_report.get("accepted") is False:
                return {
                    "queued": queued,
                    "reason": str(append_report.get("reason") or "queue_rejected"),
                }
        if ambient_recording_enabled:
            for normalized_segment in normalized_segments:
                self._memory_context_manager.update_ambient_asr_watermark(
                    normalized_segment.get("ended_at")
                    or normalized_segment.get("started_at"),
                )
            drain_report = self._process_awaiting_ambient_units(
                evaluate_episode_summary=True,
            )
            queued = bool(drain_report.get("queued")) or queued
            if not drain_report.get("accepted"):
                return {
                    "queued": queued,
                    "reason": str(drain_report.get("reason") or "queue_rejected"),
                }

        return {
            "queued": queued,
            "reason": "" if queued else "threshold_not_reached",
        }

    def trigger_memory_episode_summary(
        self,
        *,
        reason: str = "explicit",
        tags: Optional[List[str]] = None,
        prompt_language: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Queue one completed source episode and its prospective update.

        Callers must first submit the episode's final store batch.  The
        manager's FIFO queue then preserves store, summary, and prospective
        update ordering without this function reaching back into runtime
        input buffers.
        """
        resolved_tags = list(self._episode_tags if tags is None else tags or [])
        resolved_prompt_language = str(
            prompt_language or self._episode_prompt_language
        ).strip() or "zh"

        if not self._has_pending_episode_sources:
            return {
                "queued": False,
                "reason": "no_pending_episode_sources",
                "trigger_reason": reason,
            }
        report = self._memory_manager.submit_memory_episode_summary_task(
            tags=resolved_tags,
            prompt_language=resolved_prompt_language,
        )
        if bool(report.get("queued")):
            report["prospective_update"] = (
                self._memory_manager.submit_memory_prospective_update_task()
            )
            self._has_pending_episode_sources = False
            self._episode_tags = []
            self._memory_context_manager.reset_episode_summary_window()
        else:
            report["prospective_update"] = {
                "queued": False,
                "reason": "episode_summary_not_queued",
            }
        report["trigger_reason"] = reason
        return report

    def _append_memory_input_unit(
        self,
        unit: MemoryUnit,
        *,
        ambient_recording_enabled: bool,
        evaluate_episode_summary: bool = True,
    ) -> Dict[str, Any]:
        """Append one normalized input unit to the shared semantic buffer."""
        decision, finalized_units = self._memory_context_manager.insert_incoming_unit(
            unit,
            ambient_recording_enabled,
        )
        return self._handle_memory_context_boundary(
            unit,
            decision=decision,
            finalized_units=finalized_units,
            evaluate_episode_summary=evaluate_episode_summary,
        )

    def _process_awaiting_ambient_units(
        self,
        *,
        evaluate_episode_summary: bool,
        force: bool = False,
    ) -> Dict[str, Any]:
        """Submit boundary results released after ambient ASR coverage advances."""
        queued = False
        last_reason = ""
        for unit, decision, finalized_units in (
            self._memory_context_manager.process_awaiting_ambient_units(
                force=force,
            )
        ):
            append_report = self._handle_memory_context_boundary(
                unit,
                decision=decision,
                finalized_units=finalized_units,
                evaluate_episode_summary=evaluate_episode_summary,
            )
            queued = bool(append_report.get("queued")) or queued
            if not append_report.get("accepted"):
                return {
                    "accepted": False,
                    "queued": queued,
                    "reason": str(append_report.get("reason") or "queue_rejected"),
                }
            if append_report.get("queued"):
                last_reason = str(append_report.get("reason") or "")
        return {
            "accepted": True,
            "queued": queued,
            "reason": last_reason,
        }

    def _handle_memory_context_boundary(
        self,
        unit: MemoryUnit,
        *,
        decision: Any,
        finalized_units: Sequence[MemoryUnit],
        evaluate_episode_summary: bool,
    ) -> Dict[str, Any]:
        """Log one boundary decision and submit its finalized prefix, if any."""
        self._log_memory_context_manager_decision(unit, decision=decision)
        if not decision.should_finalize:
            return {
                "accepted": True,
                "queued": False,
                "reason": str(decision.reason or ""),
            }
        store_report = self._submit_memory_input_units(
            finalized_units,
            reason=decision.reason,
            evaluate_episode_summary=evaluate_episode_summary,
        )
        if not store_report.get("queued"):
            return {
                "accepted": False,
                "queued": False,
                "reason": str(store_report.get("reason") or "queue_rejected"),
            }
        return {
            "accepted": True,
            "queued": True,
            "reason": str(store_report.get("reason") or ""),
        }

    def _flush_pending_memory_input_units(
        self,
        *,
        reason: str = "explicit_flush",
        evaluate_episode_summary: bool = True,
    ) -> Dict[str, Any]:
        """Finalize assembled ASR, then submit all remaining shared input."""
        queued = False
        completed_unit = self._transcript_unit_assembler.flush()
        if completed_unit is not None:
            append_report = self._append_memory_input_unit(
                completed_unit,
                ambient_recording_enabled=self._transcript_ambient_recording_enabled,
                evaluate_episode_summary=evaluate_episode_summary,
            )
            queued = bool(append_report.get("queued")) or queued
            if not append_report.get("accepted"):
                return {
                    "queued": queued,
                    "reason": str(append_report.get("reason") or "queue_rejected"),
                }
        drain_report = self._process_awaiting_ambient_units(
            evaluate_episode_summary=evaluate_episode_summary,
            force=True,
        )
        queued = bool(drain_report.get("queued")) or queued
        if not drain_report.get("accepted"):
            return {
                "queued": queued,
                "reason": str(drain_report.get("reason") or "queue_rejected"),
            }
        store_report = self._trigger_memory_store_task_for_pending_memory_input(
            reason=reason,
            evaluate_episode_summary=evaluate_episode_summary,
        )
        return {
            "queued": bool(store_report.get("queued")) or queued,
            "reason": str(store_report.get("reason") or ""),
        }

    def _trigger_memory_store_task_for_pending_memory_input(
        self,
        *,
        reason: str,
        evaluate_episode_summary: bool = True,
    ) -> Dict[str, Any]:
        """Normalize and submit one shared semantic buffer to fact extraction."""
        pending_units = self._memory_context_manager.pending_unit_snapshot()
        if not pending_units:
            return {"queued": False, "reason": "no_pending_segments"}
        return self._submit_memory_input_units(
            pending_units,
            reason=reason,
            evaluate_episode_summary=evaluate_episode_summary,
            clear_segmenter=True,
        )

    def _submit_memory_input_units(
        self,
        units: Sequence[MemoryUnit],
        *,
        reason: str,
        evaluate_episode_summary: bool,
        clear_segmenter: bool = False,
    ) -> Dict[str, Any]:
        """Submit already selected shared units without changing their route."""
        pending_units = list(units)
        raw_segments, prompt_language = self._normalize_units_into_memory_raw_segments(
            pending_units,
        )
        if not raw_segments:
            if clear_segmenter:
                self._memory_context_manager.clear_pending_units()
            return {"queued": False, "reason": "invalid_pending_segments"}
        tags = {
            str(tag)
            for segment in raw_segments
            for tag in segment.get("tags") or []
            if tag is not None and str(tag).strip()
        }
        self._log_info(
            "memory_runtime",
            "memory_input_batch_detail",
            {
                "reason": reason,
                "tags": sorted(tags),
                "raw_segment_count": len(raw_segments),
                "semantic_unit_count": len(pending_units),
                "prompt_language": prompt_language,
                "segments": raw_segments,
            },
        )
        queue_report = self._memory_manager.submit_memory_store_task(
            raw_segments=raw_segments,
            tags=sorted(tags),
            prompt_language=prompt_language,
        )
        queued = bool(queue_report.get("queued"))
        episode_summary_report = None
        if queued:
            self._has_pending_episode_sources = True
            self._episode_prompt_language = prompt_language
            self._episode_tags = sorted(set(self._episode_tags).union(tags))
            episode_decision = self._memory_context_manager.record_stored_units(
                pending_units,
            )
            self._logger.info(
                "transcript episode queued reason=%s raw_segment_count=%s semantic_unit_count=%s",
                reason,
                len(raw_segments),
                len(pending_units),
            )
            if clear_segmenter:
                self._memory_context_manager.clear_pending_units()
            if evaluate_episode_summary and episode_decision.should_trigger:
                episode_summary_report = self.trigger_memory_episode_summary(
                    reason=f"episode_{episode_decision.reason}",
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

    def _log_memory_context_manager_decision(
        self,
        unit: MemoryUnit,
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
            "memory context decision started_at=%s ended_at=%s reason=%s raw_segment_count=%s "
            "token_count=%s speakers=%s cut_probability=%s score=%s "
            "semantic_surprise=%s cohesion_drop=%s time_gap_seconds=%s "
            "scoring_mode=%s rolling_tail_units=%s rolling_tail_tokens=%s text=%s",
            self._memory_context_manager.unit_timestamp(unit),
            self._memory_context_manager.unit_end_timestamp(unit),
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
        """Queue reflection after all pending shared input is stored."""
        input_flush_report = self._flush_pending_memory_input_units(
            reason="reflect",
        )
        report = self._memory_manager.submit_memory_reflect_task(*args, **kwargs) or {}
        report["pending_memory_input_flush"] = input_flush_report
        
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
        input_flush_report = self._flush_pending_memory_input_units(
            reason="explicit_input_boundary",
            evaluate_episode_summary=evaluate_episode_summary,
        )
        return not (
            not input_flush_report.get("queued")
            and input_flush_report.get("reason") not in {"", "no_pending_segments"}
        )

    def wait_for_memory_tasks(self, timeout: Optional[float] = None) -> bool:
        """Wait until all tasks already submitted to the memory manager complete."""
        return self._memory_manager.flush_task_queue(timeout=timeout)

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

    def _normalize_units_into_memory_raw_segments(
        self,
        units: Sequence[MemoryUnit],
    ) -> Tuple[List[Dict[str, Any]], str]:
        """Expand buffered units into normalized raw segments for storage."""
        normalized: List[Dict[str, Any]] = []
        raw_segments = [
            dict(segment)
            for unit in units
            for segment in (
                (unit.raw if isinstance(unit.raw, dict) else {}).get(
                    "raw_segments",
                )
                or []
            )
            if isinstance(segment, dict)
        ]
        for index, segment in enumerate(raw_segments, 1):
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

    def _normalize_transcript_segments_for_input(
        self,
        segments: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Normalize one inbound ASR batch before utterance assembly.

        This deliberately happens before calling ``TranscriptUnitAssembler``:
        an ASR sidecar can submit several chronological fragments at once, and
        the assembler must see their full order rather than an arbitrary
        per-RPC subset.
        """
        normalized: List[Dict[str, Any]] = []
        for index, segment in enumerate(segments or [], 1):
            item = self._normalize_single_transcript_segment(
                segment,
                fallback_index=index,
            )
            if item is not None:
                normalized.append(item)
        return sorted(
            normalized,
            key=lambda item: (
                str(item.get("started_at") or ""),
                str(item.get("ended_at") or ""),
                int(item.get("segment_index") or 0),
            ),
        )

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
