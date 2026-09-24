"""Runtime adapter between application input events and memory storage.

``MemoryNodeManager`` owns episode persistence, entity-claim updates, and
recall. This
adapter owns the short-lived interaction buffer and converts frontend-shaped
turns or transcript segments into the manager's raw episode segments.
"""

from __future__ import annotations

import logging
import json
import re
import threading
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


class _DerivedMemoryTaskScheduler:
    """Schedule claim and future-commitment projections after committed facts.

    The manager owns extraction and persistence.  This runtime-local helper
    merely coalesces successful store completions into follow-up tasks.  It
    deliberately has no timer yet: a task is scheduled when a batch threshold
    is reached, and session finalization always installs a FIFO drain fence.
    """

    _KIND_TO_TASK = {
        "entity_claim": "memory_entity_claim_update",
        "future_commitment": "memory_future_commitment_update",
    }

    def __init__(
        self,
        manager: MemoryNodeManager,
        *,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._manager = manager
        self._logger = logger or logging.getLogger(__name__)
        settings = dict(config or {})
        self._thresholds = {
            "entity_claim": {
                "default": {
                    "batches": max(1, int(
                        settings.get("entity_claim_min_signal_batches", 4) or 4
                    )),
                    "facts": max(1, int(
                        settings.get("entity_claim_min_signal_facts", 8) or 8
                    )),
                },
                "interaction": {
                    "batches": max(1, int(
                        settings.get(
                            "interaction_entity_claim_min_signal_batches", 3
                        ) or 3
                    )),
                    "facts": max(1, int(
                        settings.get(
                            "interaction_entity_claim_min_signal_facts", 5
                        ) or 5
                    )),
                },
            },
            "future_commitment": {
                "default": {
                    "batches": max(1, int(
                        settings.get(
                            "future_commitment_min_signal_batches", 4
                        ) or 4
                    )),
                    "facts": max(1, int(
                        settings.get(
                            "future_commitment_min_signal_facts", 5
                        ) or 5
                    )),
                },
                "interaction": {
                    "batches": max(1, int(
                        settings.get(
                            "interaction_future_commitment_min_signal_batches", 3
                        ) or 3
                    )),
                    "facts": max(1, int(
                        settings.get(
                            "interaction_future_commitment_min_signal_facts", 3
                        ) or 3
                    )),
                },
            },
        }
        self._lock = threading.RLock()
        self._state = {
            kind: {
                "generation": 0,
                "completed_generation": 0,
                "unscheduled_signal_batches": 0,
                "unscheduled_signal_facts": 0,
                "unscheduled_contains_interaction": False,
                "task_pending": False,
            }
            for kind in self._KIND_TO_TASK
        }
        self._force_pending_kinds: set[str] = set()
        self._force_failed_kinds: set[str] = set()

    def on_task_completion(self, event: Dict[str, Any]) -> None:
        """Consume a manager completion event on the manager worker thread."""
        task_kind = str(event.get("task_kind") or "")
        result = event.get("result")
        if task_kind == "memory_store":
            if not event.get("succeeded") or not isinstance(result, dict):
                return
            self._record_store_completion(
                result,
                completion_context=event.get("completion_context"),
            )
            return

        completion_context = event.get("completion_context")
        if not isinstance(completion_context, dict):
            return
        scheduler_context = completion_context.get("derived_task_scheduler")
        if not isinstance(scheduler_context, dict):
            return
        kind = str(scheduler_context.get("kind") or "")
        if self._KIND_TO_TASK.get(kind) != task_kind:
            return
        if bool(scheduler_context.get("force_drain")):
            self._finish_force_drain(
                kind,
                succeeded=bool(event.get("succeeded")),
            )
            return
        if not bool(scheduler_context.get("automatic")):
            return
        self._finish_automatic_task(kind, succeeded=bool(event.get("succeeded")))

    def force_drain(self) -> Dict[str, Dict[str, Any]]:
        """Append both projections after all previously queued memory work.

        This is intentionally unconditional.  The tasks query unprocessed
        signal mappings themselves, while the force marker suppresses an
        otherwise redundant automatic task from store completions that occur
        before this FIFO fence is reached.
        """
        with self._lock:
            self._force_pending_kinds = set(self._KIND_TO_TASK)
            self._force_failed_kinds.clear()
        reports = {
            "entity_claim": self._manager.submit_memory_entity_claim_update_task(
                completion_context={
                    "derived_task_scheduler": {
                        "kind": "entity_claim",
                        "force_drain": True,
                    }
                }
            ),
            "future_commitment": (
                self._manager.submit_memory_future_commitment_update_task(
                    completion_context={
                        "derived_task_scheduler": {
                            "kind": "future_commitment",
                            "force_drain": True,
                        }
                    }
                )
            ),
        }
        with self._lock:
            for kind, report in reports.items():
                if not bool((report or {}).get("queued")):
                    self._force_pending_kinds.discard(kind)
                    self._force_failed_kinds.add(kind)
                    self._mark_kind_needing_retry_locked(kind)
            force_round_finished = not self._force_pending_kinds
        if force_round_finished:
            self._finish_force_drain()
        return reports

    def snapshot(self) -> Dict[str, Any]:
        """Return scheduler state for diagnostics without exposing locks."""
        with self._lock:
            return {
                "thresholds": {
                    kind: dict(values) for kind, values in self._thresholds.items()
                },
                "force_pending_kinds": sorted(self._force_pending_kinds),
                "force_failed_kinds": sorted(self._force_failed_kinds),
                "tasks": {
                    kind: dict(values) for kind, values in self._state.items()
                },
            }

    def _record_store_completion(
        self,
        result: Dict[str, Any],
        *,
        completion_context: Any,
    ) -> None:
        context = (
            dict(completion_context)
            if isinstance(completion_context, dict)
            else {}
        )
        contains_interaction = bool(context.get("contains_interaction_turn"))
        signal_stats = {
            "entity_claim": {
                "signal_count": int(result.get("entity_claim_signal_count") or 0),
                "signal_fact_count": int(
                    result.get("entity_claim_signal_fact_count") or 0
                ),
            },
            "future_commitment": {
                "signal_count": int(
                    result.get("future_commitment_signal_count") or 0
                ),
                "signal_fact_count": int(
                    result.get("future_commitment_signal_fact_count") or 0
                ),
            },
        }
        self._logger.info(
            "memory store completion observed facts=%s entity_claim_signals=%s "
            "entity_claim_signal_facts=%s future_commitment_signals=%s "
            "future_commitment_signal_facts=%s contains_interaction_turn=%s",
            int(result.get("new_fact_count") or 0),
            signal_stats["entity_claim"]["signal_count"],
            signal_stats["entity_claim"]["signal_fact_count"],
            signal_stats["future_commitment"]["signal_count"],
            signal_stats["future_commitment"]["signal_fact_count"],
            contains_interaction,
        )
        with self._lock:
            for kind, stats in signal_stats.items():
                if stats["signal_count"] <= 0 or stats["signal_fact_count"] <= 0:
                    continue
                state = self._state[kind]
                state["generation"] += 1
                state["unscheduled_signal_batches"] += 1
                state["unscheduled_signal_facts"] += stats["signal_fact_count"]
                state["unscheduled_contains_interaction"] = bool(
                    state["unscheduled_contains_interaction"]
                    or contains_interaction
                )
        self._schedule_eligible_tasks()

    def _schedule_eligible_tasks(self) -> None:
        for kind in self._KIND_TO_TASK:
            self._schedule_kind_if_eligible(kind)

    def _schedule_kind_if_eligible(self, kind: str) -> None:
        with self._lock:
            if self._force_pending_kinds or self._state[kind]["task_pending"]:
                return
            state = self._state[kind]
            threshold_key = (
                "interaction"
                if state["unscheduled_contains_interaction"]
                else "default"
            )
            threshold = self._thresholds[kind][threshold_key]
            if (
                state["unscheduled_signal_batches"] < threshold["batches"]
                or state["unscheduled_signal_facts"] < threshold["facts"]
            ):
                return
            state["task_pending"] = True
            target_generation = int(state["generation"])

        completion_context = {
            "derived_task_scheduler": {
                "kind": kind,
                "automatic": True,
                "target_generation": target_generation,
            }
        }
        if kind == "entity_claim":
            report = self._manager.submit_memory_entity_claim_update_task(
                completion_context=completion_context,
            )
        else:
            report = self._manager.submit_memory_future_commitment_update_task(
                completion_context=completion_context,
            )
        if bool((report or {}).get("queued")):
            with self._lock:
                state = self._state[kind]
                state["unscheduled_signal_batches"] = 0
                state["unscheduled_signal_facts"] = 0
                state["unscheduled_contains_interaction"] = False
            self._logger.info(
                "memory derived task scheduled kind=%s target_generation=%s",
                kind,
                target_generation,
            )
            return
        with self._lock:
            self._state[kind]["task_pending"] = False
        self._logger.warning(
            "memory derived task rejected kind=%s reason=%s",
            kind,
            (report or {}).get("reason"),
        )

    def _finish_automatic_task(self, kind: str, *, succeeded: bool) -> None:
        with self._lock:
            state = self._state[kind]
            state["task_pending"] = False
            if succeeded:
                # FIFO means all store completion events observed before this
                # task begins are included in the task's DB query.
                state["completed_generation"] = int(state["generation"])
                state["unscheduled_signal_batches"] = 0
                state["unscheduled_signal_facts"] = 0
                state["unscheduled_contains_interaction"] = False
            else:
                self._mark_kind_needing_retry_locked(kind)
        if succeeded:
            self._schedule_kind_if_eligible(kind)

    def _finish_force_drain(
        self,
        kind: Optional[str] = None,
        *,
        succeeded: bool = True,
    ) -> None:
        """Settle one force-drain round without discarding failed evidence."""
        with self._lock:
            if kind:
                self._force_pending_kinds.discard(kind)
                if not succeeded:
                    self._force_failed_kinds.add(kind)
                    self._mark_kind_needing_retry_locked(kind)
            if self._force_pending_kinds:
                return
            failed_kinds = set(self._force_failed_kinds)
            self._force_failed_kinds.clear()
            for state_kind, state in self._state.items():
                if state_kind in failed_kinds:
                    continue
                state["completed_generation"] = int(state["generation"])
                state["unscheduled_signal_batches"] = 0
                state["unscheduled_signal_facts"] = 0
                state["unscheduled_contains_interaction"] = False
        if not failed_kinds:
            self._schedule_eligible_tasks()
        else:
            self._logger.warning(
                "memory derived force drain failed kinds=%s; retained pending evidence",
                sorted(failed_kinds),
            )

    def _mark_kind_needing_retry_locked(self, kind: str) -> None:
        """Preserve enough state for a later store completion or drain retry."""
        state = self._state[kind]
        threshold = self._thresholds[kind]["default"]
        state["unscheduled_signal_batches"] = max(
            int(state["unscheduled_signal_batches"]),
            threshold["batches"],
        )
        state["unscheduled_signal_facts"] = max(
            int(state["unscheduled_signal_facts"]),
            threshold["facts"],
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
        scheduler_config = runtime_config.get("derived_task_scheduling")
        self._derived_task_scheduler = _DerivedMemoryTaskScheduler(
            self._memory_manager,
            config=(
                dict(scheduler_config)
                if isinstance(scheduler_config, dict)
                else None
            ),
            logger=self._logger.getChild("derived_task_scheduler"),
        )
        self._manager_completion_listener = (
            self._derived_task_scheduler.on_task_completion
        )
        self._memory_manager.register_task_completion_listener(
            self._manager_completion_listener
        )

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
                    # to close while an entity-claim transaction is still in
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
                self._memory_manager.unregister_task_completion_listener(
                    self._manager_completion_listener
                )
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

    def _trigger_memory_episode_summary(
        self,
        *,
        reason: str = "explicit",
        tags: Optional[List[str]] = None,
        prompt_language: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Queue one completed source episode after its fact-store batches.

        Callers must first submit the episode's final store batch.  The
        manager's FIFO queue preserves this ordering without this function
        reaching back into runtime input buffers.
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
            self._has_pending_episode_sources = False
            self._episode_tags = []
            self._memory_context_manager.reset_episode_summary_window()
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
            self._memory_context_manager.iter_awaiting_ambient_units(
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
        tags: Optional[List[str]] = None,
        prompt_language: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Finalize tail input, then append episode and derived-task fences."""
        queued = False
        completed_unit = self._transcript_unit_assembler.flush()
        if completed_unit is not None:
            append_report = self._append_memory_input_unit(
                completed_unit,
                ambient_recording_enabled=self._transcript_ambient_recording_enabled,
                evaluate_episode_summary=False,
            )
            queued = bool(append_report.get("queued")) or queued
            if not append_report.get("accepted"):
                return {
                    "queued": queued,
                    "reason": str(append_report.get("reason") or "queue_rejected"),
                }
        drain_report = self._process_awaiting_ambient_units(
            evaluate_episode_summary=False,
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
            evaluate_episode_summary=False,
        )
        episode_summary_report = self._trigger_memory_episode_summary(
            reason=reason,
            tags=tags,
            prompt_language=prompt_language,
        )
        derived_task_reports = self._derived_task_scheduler.force_drain()
        return {
            "queued": (
                bool(store_report.get("queued"))
                or queued
                or bool(episode_summary_report.get("queued"))
                or any(
                    bool((report or {}).get("queued"))
                    for report in derived_task_reports.values()
                )
            ),
            "reason": str(store_report.get("reason") or ""),
            "episode_summary": episode_summary_report,
            "derived_tasks": derived_task_reports,
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
            completion_context=self._memory_store_completion_context(pending_units),
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
                episode_summary_report = self._trigger_memory_episode_summary(
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

    def finalize_memory_session(
        self,
        *,
        reason: str = "explicit_finalize",
        tags: Optional[List[str]] = None,
        prompt_language: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Append a final store → episode → derived-task FIFO fence.

        This is the lifecycle operation for a frontend session close or an
        ambient recording stop.  It deliberately does not wait for storage:
        the manager's single FIFO worker makes every submitted projection see
        all stores that precede it.
        """
        input_flush = self._flush_pending_memory_input_units(
            reason=reason,
            tags=tags,
            prompt_language=prompt_language,
        )
        return {
            "input_flush": input_flush,
            "episode_summary": dict(input_flush.get("episode_summary") or {}),
            "derived_tasks": dict(input_flush.get("derived_tasks") or {}),
        }
    
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
        tags: Optional[List[str]] = None,
        prompt_language: Optional[str] = None,
    ) -> bool:
        """Finalize buffered tail input without waiting for manager tasks."""
        input_flush_report = self._flush_pending_memory_input_units(
            reason="explicit_input_boundary",
            tags=tags,
            prompt_language=prompt_language,
        )
        return not (
            not input_flush_report.get("queued")
            and input_flush_report.get("reason") not in {"", "no_pending_segments"}
        )

    def wait_for_memory_tasks(self, timeout: Optional[float] = None) -> bool:
        """Wait until all tasks already submitted to the memory manager complete."""
        return self._memory_manager.flush_task_queue(timeout=timeout)

    def derived_task_scheduler_snapshot(self) -> Dict[str, Any]:
        """Return runtime-local derived-task scheduling diagnostics."""
        return self._derived_task_scheduler.snapshot()

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
        normalized: List[Tuple[Dict[str, Any], int]] = []
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
                normalized.append((normalized_segment, index))
        normalized.sort(
            key=lambda entry: self._memory_raw_segment_time_order_key(
                entry[0],
                stable_index=entry[1],
            ),
        )
        ordered_segments = [item for item, _stable_index in normalized]
        return ordered_segments, self._resolve_prompt_language_from_segments(
            ordered_segments,
        )

    def _memory_store_completion_context(
        self,
        units: Sequence[MemoryUnit],
    ) -> Dict[str, Any]:
        """Attach Runtime-only scheduling hints to one submitted store task."""
        unit_list = list(units or [])
        timestamps = [
            self._memory_context_manager.unit_timestamp(unit)
            for unit in unit_list
            if self._memory_context_manager.unit_timestamp(unit)
        ]
        end_timestamps = [
            self._memory_context_manager.unit_end_timestamp(unit)
            for unit in unit_list
            if self._memory_context_manager.unit_end_timestamp(unit)
        ]
        return {
            "contains_interaction_turn": any(
                isinstance(unit.raw, dict)
                and str(unit.raw.get("input_kind") or "") == "interaction"
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

    @staticmethod
    def _memory_raw_segment_time_order_key(
        segment: Dict[str, Any],
        *,
        stable_index: int,
    ) -> Tuple[int, float, int, float, int]:
        """Order source segments by absolute time, preserving equal-time input order.

        An ambient ASR batch generally uses UTC timestamps while Realtime
        interaction turns use local-offset timestamps.  Their strings cannot
        be compared lexically: ``09:46+00:00`` and ``17:46+08:00`` denote the
        same time.  Use the context manager's canonical parser and retain the
        flattened unit order as the tie breaker, which also preserves a turn's
        user-before-assistant source order.
        """
        started_at = MemoryContextManager._parse_timestamp(
            segment.get("started_at"),
        )
        ended_at = MemoryContextManager._parse_timestamp(
            segment.get("ended_at"),
        )
        return (
            0 if started_at is not None else 1,
            started_at.timestamp() if started_at is not None else float("inf"),
            0 if ended_at is not None else 1,
            ended_at.timestamp() if ended_at is not None else float("inf"),
            stable_index,
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
