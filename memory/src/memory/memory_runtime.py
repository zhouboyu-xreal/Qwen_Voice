"""Runtime adapter between application input events and memory storage.

``MemoryNodeManager`` owns user-activity persistence, user-claim updates, and
recall. This
adapter owns the short-lived interaction buffer and converts frontend-shaped
turns or transcript segments into typed Manager-facing memory units.
"""

from __future__ import annotations

import logging
import json
import re
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from embedding_client import EmbeddingClient
from .memory_database import UserMemoryDB

from .memory_manager import (
    MemoryNodeManager,
    _compact_whitespace,
    _now_text,
    _to_timestamp_text,
)
from llm_client import LLMClient
from task_scheduler import TaskScheduler
from .memory_context_manager import (
    MemoryContextManager,
    TranscriptUnitAssembler,
    build_memory_context_manager_config,
    build_transcript_aggregation_config,
    convert_interaction_turn_to_online_unit,
)
from .memory_types import MemoryUnit


_MEMORY_INPUT_TOKEN_RE = re.compile(
    r"[\u4e00-\u9fff]|[A-Za-z0-9_.$'-]+|[^\s]"
)
_MEMORY_INPUT_DEDUP_MIN_TEXT_CHARS = 8
_MEMORY_INPUT_DEDUP_MAX_TIME_GAP_SECONDS = 120.0
_MEMORY_INPUT_DEDUP_MIN_FUZZY_RATIO = 0.90

class MemoryRuntime:
    """Normalize application input and batch it before user activity extraction."""

    def __init__(
        self,
        *,
        db_path: Path | str,
        database: UserMemoryDB,
        task_scheduler: TaskScheduler,
        llm_client: LLMClient,
        memory_runtime_config: Optional[Dict[str, Any]] = None,
        memory_manager_config: Optional[Dict[str, Any]] = None,
        embedding_config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Initialize input batching around already-composed shared resources."""
        self._logger = logger or logging.getLogger(__name__)
        manager_config = dict(memory_manager_config or {})
        runtime_config = dict(memory_runtime_config or {})
        if Path(db_path).expanduser().resolve() != database.db_path.expanduser().resolve():
            raise ValueError("Injected database path does not match db_path")
        resolved_embedding_config = dict(embedding_config or {})

        manager_logger = self._logger.getChild("memory_manager")
        self._llm_client = llm_client
        self._task_scheduler = task_scheduler
        manager = MemoryNodeManager(
            database,
            embedding_config=resolved_embedding_config,
            memory_manager_config=manager_config,
            task_scheduler=self._task_scheduler,
            llm_client=self._llm_client,
            logger=manager_logger,
        )
        self._memory_database = database
        self._memory_manager = manager
        self._embedding_client = EmbeddingClient(
            dict(resolved_embedding_config or getattr(manager, "_embedding_cfg", {}) or {}),
        )
        if hasattr(self._memory_manager, "set_embedding_client"):
            self._memory_manager.set_embedding_client(self._embedding_client)
        memory_context_manager_config = build_memory_context_manager_config(
            runtime_config,
        )
        self._transcript_unit_assembler = TranscriptUnitAssembler(
            build_transcript_aggregation_config(runtime_config),
        )
        self._transcript_ambient_recording_enabled = False
        self._memory_context_manager = MemoryContextManager(
            self._embedding_client,
            memory_context_manager_config,
        )
        self._user_activity_tags: List[str] = []

    @property
    def manager(self) -> MemoryNodeManager:
        """Return the manager owned by this runtime for diagnostics/adapters."""
        return self._memory_manager

    @property
    def database(self) -> UserMemoryDB:
        """Return the shared database injected by the composition root."""
        return self._memory_database

    def accept_memory_input(
        self,
        *,
        interaction_turn: Optional[Dict[str, Any]] = None,
        transcript_segments: Optional[Sequence[Dict[str, Any]]] = None,
        tags: Optional[List[str]] = None,
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
            ambient_recording_enabled=ambient_recording_enabled,
        )

    def update_ambient_asr_watermark(
        self,
        value: Any,
        *,
        evaluate_user_activity: bool = True,
    ) -> Dict[str, Any]:
        """Advance ambient ASR coverage and process newly released input units."""
        self._memory_context_manager.update_ambient_asr_watermark(value)
        return self._process_awaiting_ambient_units(
            evaluate_user_activity=evaluate_user_activity,
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
            "user_started_at": _to_timestamp_text(
                interaction_turn.get("user_started_at")
            ) or turn_timestamp,
            "user_ended_at": _to_timestamp_text(
                interaction_turn.get("user_ended_at")
            ) or turn_timestamp,
            "assistant_started_at": _to_timestamp_text(
                interaction_turn.get("assistant_started_at")
            ) or turn_timestamp,
            "assistant_ended_at": _to_timestamp_text(
                interaction_turn.get("assistant_ended_at")
            ) or turn_timestamp,
        }
        if not turn["user_message"] and not turn["assistant_response"]:
            return {"queued": False, "reason": "empty_turn"}
        self._logger.info(
            "memory runtime received interaction turn ambient_recording_enabled=%s "
            "user_started_at=%s user_ended_at=%s assistant_started_at=%s "
            "assistant_ended_at=%s tags=%s user_chars=%s assistant_chars=%s "
            "user_message=%s assistant_response=%s",
            ambient_recording_enabled,
            turn["user_started_at"],
            turn["user_ended_at"],
            turn["assistant_started_at"],
            turn["assistant_ended_at"],
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

        self._user_activity_tags = sorted(
            set(self._user_activity_tags).union(tags)
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
            watermark: Optional[datetime] = None
            for normalized_segment in normalized_segments:
                watermark = self._memory_context_manager.update_ambient_asr_watermark(
                    normalized_segment.get("ended_at")
                    or normalized_segment.get("started_at"),
                )
            self._logger.info(
                "memory runtime updated ambient ASR watermark batch_segments=%s "
                "batch_started_at=%s batch_ended_at=%s watermark=%s "
                "awaiting_unit_count=%s",
                len(normalized_segments),
                normalized_segments[0].get("started_at") or "",
                normalized_segments[-1].get("ended_at")
                or normalized_segments[-1].get("started_at")
                or "",
                _to_timestamp_text(watermark) or "",
                self._memory_context_manager.awaiting_ambient_unit_count(),
            )
            drain_report = self._process_awaiting_ambient_units(
                evaluate_user_activity=True,
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

    def _trigger_memory_user_activity(
        self,
        *,
        reason: str = "explicit",
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Queue activity observation for the completed source window.

        Callers must first submit the window's final fact batch. The
        scheduler's FIFO queue preserves this ordering without this function
        reaching back into runtime input buffers.
        """
        resolved_tags = list(self._user_activity_tags if tags is None else tags or [])

        if not self._memory_context_manager.has_pending_user_activity_units():
            return {
                "queued": False,
                "reason": "no_pending_user_activity_units",
                "trigger_reason": reason,
            }
        report = self._task_scheduler.submit(
            "memory_user_activity_extraction",
            {"tags": resolved_tags},
        )
        if bool(report.get("queued")):
            self._user_activity_tags = []
            self._memory_context_manager.reset_user_activity_window()
        report["trigger_reason"] = reason
        return report

    def _append_memory_input_unit(
        self,
        unit: MemoryUnit,
        *,
        ambient_recording_enabled: bool,
        evaluate_user_activity: bool = True,
    ) -> Dict[str, Any]:
        """Append one normalized input unit to the shared semantic buffer."""
        decision = self._memory_context_manager.insert_incoming_unit(
            unit,
            ambient_recording_enabled,
        )
        return self._handle_memory_fact_task_decision(
            decision=decision,
            evaluate_user_activity=evaluate_user_activity,
        )

    def _process_awaiting_ambient_units(
        self,
        *,
        evaluate_user_activity: bool,
        force: bool = False,
    ) -> Dict[str, Any]:
        """Process boundary results released after ambient ASR coverage advances."""
        queued = False
        last_reason = ""
        for _unit, decision in (
            self._memory_context_manager.iter_awaiting_ambient_units(
                force=force,
            )
        ):
            append_report = self._handle_memory_fact_task_decision(
                decision=decision,
                evaluate_user_activity=evaluate_user_activity,
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

    def _handle_memory_fact_task_decision(
        self,
        *,
        decision: Any,
        evaluate_user_activity: bool,
    ) -> Dict[str, Any]:
        """Log one boundary and schedule fact extraction for sealed chapters."""
        self._logger.info(
            "memory fact task decision trigger=%s "
            "reason=%s sealed_chapter_count=%s sealed_token_count=%s",
            decision.should_trigger,
            decision.reason,
            decision.sealed_chapter_count,
            decision.sealed_token_count,
        )
        if not decision.should_trigger:
            return {
                "accepted": True,
                "queued": False,
                "reason": decision.reason,
            }
        self._log_sealed_memory_input_chapters_for_fact_task(decision=decision)
        store_report = self._trigger_memory_store_task_for_sealed_memory_input()
        if not store_report.get("queued"):
            return {
                "accepted": False,
                "queued": False,
                "reason": str(store_report.get("reason") or "queue_rejected"),
            }
        user_activity_report = None
        activity_decision = self._memory_context_manager.evaluate_user_activity_trigger_decision()
        if evaluate_user_activity and activity_decision.should_trigger:
            user_activity_report = self._trigger_memory_user_activity(
                reason=f"user_activity_{activity_decision.reason}",
            )
        return {
            "accepted": True,
            "queued": True,
            "reason": (
                "user_activity_limit"
                if bool((user_activity_report or {}).get("queued"))
                else str(store_report.get("reason") or "")
            ),
            "user_activity": user_activity_report,
        }

    def _log_sealed_memory_input_chapters_for_fact_task(
        self,
        *,
        decision: Any,
    ) -> None:
        """Log the exact sealed chapter window before it is submitted.

        This is intentionally emitted only for a fact-task trigger.  The
        ordinary boundary log remains compact for append decisions, while this
        payload makes it possible to audit the chapter grouping that supplied
        one extraction request.
        """
        sealed_chapters = self._memory_context_manager.sealed_chapter_snapshot()
        chapter_payloads: List[Dict[str, Any]] = []
        for chapter_index, chapter in enumerate(sealed_chapters, 1):
            units = list(chapter.units)
            chapter_payloads.append({
                "chapter_index": chapter_index,
                "boundary_reason": chapter.boundary_reason,
                "unit_count": len(units),
                "token_count": sum(unit.token_count for unit in units),
                "started_at": (
                    units[0].start_time
                    if units
                    else ""
                ),
                "ended_at": (
                    units[-1].end_time
                    if units
                    else ""
                ),
                "units": [
                    {
                        "unit_index": unit_index,
                        "started_at": unit.start_time,
                        "ended_at": unit.end_time,
                        "token_count": unit.token_count,
                        "entry_count": len(unit.texts or []),
                        "text": unit.text,
                    }
                    for unit_index, unit in enumerate(units, 1)
                ],
            })
        self._log_info(
            "memory_runtime",
            "memory_fact_task_sealed_chapters",
            {
                "trigger_reason": str(getattr(decision, "reason", "") or ""),
                "sealed_chapter_count": len(chapter_payloads),
                "sealed_token_count": sum(
                    int(chapter.get("token_count") or 0)
                    for chapter in chapter_payloads
                ),
                "chapters": chapter_payloads,
            },
        )

    def _flush_pending_memory_input_units(
        self,
        *,
        reason: str = "explicit_flush",
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Finalize tail input, then append activity and fact follow-up fences."""
        queued = False
        completed_unit = self._transcript_unit_assembler.flush()
        if completed_unit is not None:
            append_report = self._append_memory_input_unit(
                completed_unit,
                ambient_recording_enabled=self._transcript_ambient_recording_enabled,
                evaluate_user_activity=False,
            )
            queued = bool(append_report.get("queued")) or queued
            if not append_report.get("accepted"):
                return {
                    "accepted": False,
                    "queued": queued,
                    "reason": str(append_report.get("reason") or "queue_rejected"),
                }
        drain_report = self._process_awaiting_ambient_units(
            evaluate_user_activity=False,
            force=True,
        )
        queued = bool(drain_report.get("queued")) or queued
        if not drain_report.get("accepted"):
            return {
                "accepted": False,
                "queued": queued,
                "reason": str(drain_report.get("reason") or "queue_rejected"),
            }
        self._memory_context_manager.seal_pending_units(boundary_reason=reason)
        store_report = self._trigger_memory_store_task_for_sealed_memory_input()
        user_activity_report = self._trigger_memory_user_activity(
            reason=reason,
            tags=tags,
        )
        fact_followup_dispatch = self._task_scheduler.submit_flush_fact_followup_tasks()
        submission_reports = [
            store_report,
            user_activity_report,
            fact_followup_dispatch,
        ]
        rejected_reasons = {
            "queue_rejected",
            "worker_queue_full",
            "worker_shutdown",
            "task_handler_not_registered",
            "memory_disabled",
            "task_disabled",
        }
        accepted = not any(
            str((task_report or {}).get("reason") or "") in rejected_reasons
            for task_report in submission_reports
        )
        return {
            "accepted": accepted,
            "queued": (
                bool(store_report.get("queued"))
                or queued
                or bool(user_activity_report.get("queued"))
                or bool(fact_followup_dispatch.get("queued"))
            ),
            "reason": str(store_report.get("reason") or ""),
            "user_activity": user_activity_report,
            "fact_followup_dispatch": fact_followup_dispatch,
        }

    def _trigger_memory_store_task_for_sealed_memory_input(
        self,
    ) -> Dict[str, Any]:
        """Submit the currently sealed chapters as one fact task."""
        sealed_chapters = self._memory_context_manager.sealed_chapter_snapshot()
        if not sealed_chapters:
            return {"queued": False, "reason": "no_sealed_chapters"}
        sealed_units = [
            unit
            for chapter in sealed_chapters
            for unit in chapter.units
        ]
        self._logger.info(
            "memory fact task submitting sealed_chapter_count=%s "
            "sealed_token_count=%s boundary_reasons=%s",
            len(sealed_chapters),
            sum(unit.token_count for unit in sealed_units),
            [chapter.boundary_reason for chapter in sealed_chapters],
        )
        pending_units = list(sealed_units)
        canonical_units, deduplication_report = (
            self._deduplicate_memory_input_units(pending_units)
        )
        preceding_context_units = (
            self._memory_context_manager.preceding_context_units_snapshot()
        )
        if not canonical_units:
            return {"queued": False, "reason": "invalid_pending_units"}
        tags = {
            str(tag)
            for unit in canonical_units
            for tag in unit.tags
            if tag is not None and str(tag).strip()
        }
        self._log_info(
            "memory_runtime",
            "memory_input_batch_detail",
            {
                "tags": sorted(tags),
                "original_unit_entry_count": deduplication_report[
                    "original_unit_entry_count"
                ],
                "semantic_unit_count": len(canonical_units),
                "original_semantic_unit_count": len(pending_units),
                "preceding_context_unit_count": len(preceding_context_units),
                "preceding_context_task_unit_count": len(preceding_context_units),
                "deduplication": deduplication_report,
                "preceding_context_units": self._memory_units_log_view(
                    preceding_context_units,
                ),
                "units": self._memory_units_log_view(canonical_units),
            },
        )
        store_completion_context = self._memory_store_completion_context(
            canonical_units,
        )
        queue_report = self._task_scheduler.submit(
            "memory_fact_extraction",
            {
                "units": canonical_units,
                "preceding_context_units": preceding_context_units,
                "tags": sorted(tags),
            },
            completion_context=store_completion_context,
            report_payload={"tags": sorted(tags)},
        )
        queued = bool(queue_report.get("queued"))
        if queued:
            self._memory_context_manager.clear_sealed_chapter_buffers()
            self._user_activity_tags = sorted(set(self._user_activity_tags).union(tags))
            self._memory_context_manager.record_stored_units(
                canonical_units,
            )
            self._memory_context_manager.record_preceding_context_units(
                sealed_chapters[-1].units,
            )
            self._logger.info(
                "memory user activity source queued semantic_unit_count=%s "
                "preceding_context_unit_count=%s",
                len(canonical_units),
                len(preceding_context_units),
            )
        return {
            **queue_report,
            "queued": queued,
            "reason": "" if queued else str(queue_report.get("reason") or "queue_rejected"),
        }

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

    def trigger_memory_recall(
        self,
        query: str,
        *,
        tags: Optional[List[str]] = None,
        time_end: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run recall immediately through the manager without adding buffering."""
        return self._memory_manager.process_memory_recall_immediately(
            query=str(query or ""),
            tags=tags,
            time_end=time_end,
        )

    def flush_pending_memory_inputs(
        self,
        *,
        reason: str = "explicit_input_boundary",
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Submit buffered tail input and its final follow-up tasks.

        This establishes an input boundary but does not wait for queued tasks
        to finish. Call :meth:`wait_for_memory_tasks` when completion is needed.
        """
        input_flush_report = self._flush_pending_memory_input_units(
            reason=reason,
            tags=tags,
        )
        return {
            "accepted": bool(input_flush_report.get("accepted", True)),
            "input_flush": input_flush_report,
            "user_activity": dict(input_flush_report.get("user_activity") or {}),
            "fact_followup_dispatch": dict(
                input_flush_report.get("fact_followup_dispatch") or {}
            ),
        }

    def wait_for_memory_tasks(self, timeout: Optional[float] = None) -> bool:
        """Wait until all tasks already submitted to the memory manager complete."""
        return self._task_scheduler.flush(timeout=timeout)

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

    @staticmethod
    def _memory_units_log_view(units: Sequence[MemoryUnit]) -> List[Dict[str, Any]]:
        """Return serializable task-unit details without Runtime embeddings."""
        return [
            {
                "unit_type": unit.unit_type,
                "speakers": list(unit.speakers or []),
                "texts": list(unit.texts or []),
                "started_ats": list(unit.started_ats or []),
                "ended_ats": list(unit.ended_ats or []),
                "tags": list(unit.tags or []),
            }
            for unit in units or []
            if _compact_whitespace(unit.text)
        ]

    def _memory_store_completion_context(
        self,
        units: Sequence[MemoryUnit],
    ) -> Dict[str, Any]:
        """Attach Runtime-only scheduling hints to one submitted store task."""
        unit_list = list(units or [])
        timestamps = [
            unit.start_time
            for unit in unit_list
            if unit.start_time
        ]
        end_timestamps = [
            unit.end_time
            for unit in unit_list
            if unit.end_time
        ]
        return {
            "contains_interaction_turn": any(
                unit.unit_type == "interaction"
                for unit in unit_list
            ),
            "input_time_start": timestamps[0] if timestamps else "",
            "input_time_end": (
                end_timestamps[-1]
                if end_timestamps
                else timestamps[-1]
                if timestamps
                else ""
            ),
        }

    def _deduplicate_memory_input_units(
        self,
        units: Sequence[MemoryUnit],
    ) -> Tuple[List[MemoryUnit], Dict[str, Any]]:
        """Drop transcript copies of interaction text from one store batch."""
        entries_by_unit = [
            self._memory_unit_entries(unit)
            for unit in units
        ]
        report: Dict[str, Any] = {
            "original_unit_entry_count": sum(
                len(entries) for entries in entries_by_unit
            ),
            "suppressed_entry_count": 0,
            "suppressed_entries": [],
        }
        suppressed_by_unit: Dict[int, set[int]] = {}
        for interaction_index, interaction_unit in enumerate(units):
            interaction_entries = self._interaction_unit_entries(
                interaction_unit,
                entries_by_unit[interaction_index],
            )
            if not interaction_entries:
                continue
            for transcript_index in self._iter_time_nearby_transcript_unit_indexes(
                units,
                interaction_index,
            ):
                transcript_entries = entries_by_unit[transcript_index]
                for entry_index, transcript_entry in enumerate(
                    transcript_entries,
                ):
                    if entry_index in suppressed_by_unit.get(transcript_index, set()):
                        continue
                    matched = self._find_interaction_duplicate_match(
                        transcript_entry,
                        interaction_entries,
                    )
                    if matched is None:
                        continue
                    interaction_entry, match_info = matched
                    suppressed_by_unit.setdefault(transcript_index, set()).add(
                        entry_index,
                    )
                    report["suppressed_entry_count"] += 1
                    report["suppressed_entries"].append({
                        "speaker": transcript_entry.get("speaker") or "",
                        "text": transcript_entry.get("text") or "",
                        "started_at": transcript_entry.get("started_at") or "",
                        "ended_at": transcript_entry.get("ended_at") or "",
                        "matched_interaction_speaker": interaction_entry.get(
                            "speaker",
                        ) or "",
                        "matched_interaction_text": interaction_entry.get(
                            "text",
                        ) or "",
                        **match_info,
                    })

        if not suppressed_by_unit:
            return list(units), report
        canonical_units: List[MemoryUnit] = []
        for unit_index, (unit, entries) in enumerate(
            zip(units, entries_by_unit),
        ):
            suppressed_indexes = suppressed_by_unit.get(unit_index)
            if not suppressed_indexes:
                canonical_units.append(unit)
                continue
            retained_entries = [
                entry
                for entry_index, entry in enumerate(entries)
                if entry_index not in suppressed_indexes
            ]
            if retained_entries:
                canonical_units.append(
                    self._rebuild_memory_input_unit(unit, retained_entries)
                )
        return canonical_units, report

    @staticmethod
    def _memory_unit_entries(
        unit: MemoryUnit,
    ) -> List[Dict[str, Any]]:
        return [
            {
                "text": _compact_whitespace(entry.get("text") or ""),
                "speaker": _compact_whitespace(entry.get("speaker") or ""),
                "started_at": _to_timestamp_text(entry.get("started_at")),
                "ended_at": _to_timestamp_text(
                    entry.get("ended_at") or entry.get("started_at"),
                ),
            }
            for entry in unit.entries()
            if _compact_whitespace(entry.get("text") or "")
        ]

    @staticmethod
    def _interaction_unit_entries(
        unit: MemoryUnit,
        entries: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if unit.unit_type != "interaction":
            return []
        return [
            entry
            for entry in entries
            if _compact_whitespace(entry.get("speaker") or "")
            in {"speaker_用户本人", "speaker_助手"}
        ]

    def _iter_time_nearby_transcript_unit_indexes(
        self,
        units: Sequence[MemoryUnit],
        interaction_index: int,
    ) -> Sequence[int]:
        """Return non-interaction units near one interaction in time order."""
        interaction_unit = units[interaction_index]
        interaction_started_at = MemoryContextManager._parse_timestamp(
            interaction_unit.start_time,
        )
        interaction_ended_at = MemoryContextManager._parse_timestamp(
            interaction_unit.end_time,
        )
        if interaction_started_at is None or interaction_ended_at is None:
            return []
        nearby_indexes: List[int] = []
        for candidate_index in range(interaction_index - 1, -1, -1):
            candidate = units[candidate_index]
            candidate_ended_at = MemoryContextManager._parse_timestamp(
                candidate.end_time,
            )
            if candidate_ended_at is None:
                continue
            if (
                interaction_started_at - candidate_ended_at
            ).total_seconds() > _MEMORY_INPUT_DEDUP_MAX_TIME_GAP_SECONDS:
                break
            if candidate.unit_type != "interaction":
                nearby_indexes.append(candidate_index)
        for candidate_index in range(interaction_index + 1, len(units)):
            candidate = units[candidate_index]
            candidate_started_at = MemoryContextManager._parse_timestamp(
                candidate.start_time,
            )
            if candidate_started_at is None:
                continue
            if (
                candidate_started_at - interaction_ended_at
            ).total_seconds() > _MEMORY_INPUT_DEDUP_MAX_TIME_GAP_SECONDS:
                break
            if candidate.unit_type != "interaction":
                nearby_indexes.append(candidate_index)
        return nearby_indexes

    def _find_interaction_duplicate_match(
        self,
        segment: Dict[str, Any],
        interaction_segments: Sequence[Dict[str, Any]],
    ) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
        best_match: Optional[Tuple[Dict[str, Any], Dict[str, Any]]] = None
        for reference in interaction_segments:
            text_match = self._memory_input_text_duplicate_match(
                str(segment.get("text") or ""),
                str(reference.get("text") or ""),
            )
            if text_match is None:
                continue
            match_info = {
                "match_kind": text_match["kind"],
                "text_similarity": text_match["similarity"],
            }
            if (
                best_match is None
                or match_info["text_similarity"]
                > best_match[1]["text_similarity"]
            ):
                best_match = (reference, match_info)
        return best_match

    @staticmethod
    def _memory_input_text_duplicate_match(
        left: str,
        right: str,
    ) -> Optional[Dict[str, Any]]:
        normalized_left = re.sub(
            r"[^0-9a-zA-Z\u4e00-\u9fff]+",
            "",
            str(left or "").lower(),
        )
        normalized_right = re.sub(
            r"[^0-9a-zA-Z\u4e00-\u9fff]+",
            "",
            str(right or "").lower(),
        )
        if (
            min(len(normalized_left), len(normalized_right))
            < _MEMORY_INPUT_DEDUP_MIN_TEXT_CHARS
        ):
            return None
        if normalized_left == normalized_right:
            return {"kind": "exact", "similarity": 1.0}
        shorter, longer = sorted(
            (normalized_left, normalized_right),
            key=len,
        )
        containment = len(shorter) / max(1, len(longer))
        if shorter in longer and containment >= _MEMORY_INPUT_DEDUP_MIN_FUZZY_RATIO:
            return {"kind": "contained", "similarity": round(containment, 4)}
        similarity = SequenceMatcher(
            None,
            normalized_left,
            normalized_right,
            autojunk=False,
        ).ratio()
        if similarity >= _MEMORY_INPUT_DEDUP_MIN_FUZZY_RATIO:
            return {"kind": "fuzzy", "similarity": round(similarity, 4)}
        return None

    @staticmethod
    def _rebuild_memory_input_unit(
        unit: MemoryUnit,
        entries: Sequence[Dict[str, Any]],
    ) -> MemoryUnit:
        retained_entries = [dict(entry) for entry in entries]
        texts = [
            _compact_whitespace(entry.get("text") or "")
            for entry in retained_entries
            if _compact_whitespace(entry.get("text") or "")
        ]
        speakers = [
            _compact_whitespace(entry.get("speaker") or "unknown_speaker")
            for entry in retained_entries
            if _compact_whitespace(entry.get("text") or "")
        ]
        started_ats = [
            _to_timestamp_text(entry.get("started_at"))
            for entry in retained_entries
            if _compact_whitespace(entry.get("text") or "")
        ]
        ended_ats = [
            _to_timestamp_text(entry.get("ended_at") or entry.get("started_at"))
            for entry in retained_entries
            if _compact_whitespace(entry.get("text") or "")
        ]
        text = " ".join(texts)
        return MemoryUnit(
            texts=texts,
            speakers=speakers,
            token_count=max(1, len(_MEMORY_INPUT_TOKEN_RE.findall(text))),
            started_ats=started_ats,
            ended_ats=ended_ats,
            unit_type=unit.unit_type,
            tags=list(unit.tags or []),
        )

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
        if speaker == "speaker_0":
            speaker = "speaker_用户本人"
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
