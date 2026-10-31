"""FIFO execution and operation reporting for asynchronous memory tasks."""

from __future__ import annotations

import logging
import queue
import sqlite3
import threading
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence

from task_execution_database import TaskExecutionDB

if TYPE_CHECKING:
    from .memory_database import UserMemoryDB


TaskHandler = Callable[..., Any]


def _report_now_text() -> str:
    return datetime.now().astimezone().isoformat()


def _report_timestamp_text(value: datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S")


class TaskOperationReporter:
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
                "timestamp": _report_now_text(),
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
                "timestamp": _report_now_text(),
            })

    def on_task_finished(
        self,
        *,
        operation_type: str,
        task_id: str,
        started_at: float,
        result: Any = None,
        error: Optional[BaseException] = None,
    ) -> bool:
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
                "result": self._json_compatible(report),
                "elapsed_ms": elapsed_ms,
                "timestamp": _report_now_text(),
            }
            if error is not None:
                recent_event["error"] = str(error)
                recent_event["error_type"] = type(error).__name__
            self._append_recent_locked(recent_event)
        return succeeded

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
                "timestamp": _report_now_text(),
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
            return _report_timestamp_text(value)
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
        if operation_type == "memory_fact_extraction":
            if not isinstance(result, dict):
                return bool(result)
            return (
                str(result.get("status") or "ok").strip().lower()
                not in failure_statuses
            )
        if operation_type in {
            "memory_flush_fact_followups",
            "memory_user_activity_extraction",
            "memory_user_thread_update",
            "memory_user_claim_update",
            "memory_user_observed_pattern_update",
            "memory_user_intent_update",
            "user_state_update_from_user_claims",
            "user_state_update_from_user_threads",
            "user_state_update_from_user_observed_patterns",
            "user_prediction_update_from_intent_events",
            "user_proactive_policy_evaluation",
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
        if operation_type in {"memory_fact_extraction", "memory_user_activity_extraction"}:
            report["stored"] = bool(succeeded and error is None)
        if error is not None:
            report["error_type"] = type(error).__name__
            report["error"] = str(error)
        return report


class TaskScheduler:
    """Own the complete lifecycle of queued tasks on one FIFO worker."""

    _FACT_FOLLOWUP_TASK_KINDS = (
        "memory_user_claim_update",
        "memory_user_observed_pattern_update",
        "memory_user_intent_update",
    )
    _SIGNAL_TASK_TARGETS = {
        "memory_user_claim_update": "explicit_user_claim",
        "memory_user_observed_pattern_update": "user_action_signal_consolidation",
        "memory_user_intent_update": "user_intent_update",
    }

    def __init__(
        self,
        *,
        operation_reporter: TaskOperationReporter,
        maxsize: int = 100,
        logger: Optional[logging.Logger] = None,
        worker_name: str = "memory-task-worker",
        task_scheduling_config: Optional[Dict[str, Any]] = None,
        claim_state_database: Optional[UserMemoryDB] = None,
        memory_enabled: Any = True,
    ) -> None:
        self._operation_reporter = operation_reporter
        self._handlers: Dict[str, TaskHandler] = {}
        self._queue: queue.Queue[Dict[str, Any]] = queue.Queue(
            maxsize=max(1, int(maxsize)),
        )
        self._logger = logger or logging.getLogger(__name__)
        self._worker_name = worker_name
        self._worker_thread: Optional[threading.Thread] = None
        self._worker_lock = threading.Lock()
        self._execution_lock = threading.RLock()
        self._shutdown_event = threading.Event()
        settings = dict(
            task_scheduling_config
            if isinstance(task_scheduling_config, dict)
            else {}
        )
        self._memory_enabled = self._config_bool(memory_enabled, True)
        user_state_updates_enabled = self._config_bool(
            settings.get("enable_memory_user_state_update"), True,
        )
        self._task_enabled = {
            "memory_user_thread_update": self._config_bool(
                settings.get("enable_memory_user_thread_update"), True,
            ),
            "memory_user_claim_update": self._config_bool(
                settings.get("enable_memory_user_claim_update"), True,
            ),
            "memory_user_observed_pattern_update": self._config_bool(
                settings.get("enable_memory_user_observed_pattern_update"), True,
            ),
            "memory_user_intent_update": self._config_bool(
                settings.get("enable_memory_user_intent_update"), True,
            ),
            "user_state_update_from_user_claims": user_state_updates_enabled,
            "user_state_update_from_user_threads": user_state_updates_enabled,
            "user_state_update_from_user_observed_patterns": user_state_updates_enabled,
            "user_prediction_update_from_intent_events": self._config_bool(
                settings.get("enable_user_prediction"), True,
            ),
            "user_proactive_policy_evaluation": self._config_bool(
                settings.get("enable_user_proactive_policy_evaluation"), True,
            ),
        }
        self._user_prediction_horizon_hours = max(
            0.1, float(settings.get("user_prediction_horizon_hours", 6) or 6),
        )
        self._prediction_lock = threading.RLock()
        self._prediction_events_inflight: set[int] = set()
        self._claim_state_database = claim_state_database
        self._task_execution_db: Optional[TaskExecutionDB] = None
        if claim_state_database is not None and hasattr(claim_state_database, "db_path"):
            task_db_path = claim_state_database.db_path.with_name("task_scheduler.db")
            self._task_execution_db = TaskExecutionDB(task_db_path)
            try:
                signal_references = claim_state_database.get_user_signal_batches()
                self._task_execution_db.register_signals(**signal_references)
            except Exception:
                self._task_execution_db.close()
                raise
        self._claim_state_window_seconds = max(
            1.0, float(settings.get("user_state_claim_window_seconds", 1800) or 1800),
        )
        self._claim_state_min_changed_claims = max(
            1, int(settings.get("user_state_claim_min_changed_claims", 3) or 3),
        )
        self._claim_state_max_wait_seconds = max(
            self._claim_state_window_seconds,
            float(settings.get("user_state_claim_max_wait_seconds", 7200) or 7200),
        )
        self._claim_state_batch_limit = max(
            1, int(settings.get("user_state_claim_batch_limit", 80) or 80),
        )
        self._claim_state_lock = threading.Lock()
        self._claim_state_task_pending = False
        self._claim_state_force_requested = False
        self._next_claim_state_tick_at = (
            time.monotonic()
            + self._claim_state_window_seconds
            - (time.time() % self._claim_state_window_seconds)
        )
        self._fact_followup_thresholds = {
            "memory_user_claim_update": {
                "default": {
                    "batches": max(1, int(settings.get("user_claim_min_signal_batches", 4) or 4)),
                    "facts": max(1, int(settings.get("user_claim_min_signal_facts", 8) or 8)),
                },
                "interaction": {
                    "batches": max(1, int(settings.get("interaction_user_claim_min_signal_batches", 3) or 3)),
                    "facts": max(1, int(settings.get("interaction_user_claim_min_signal_facts", 5) or 5)),
                },
            },
            "memory_user_intent_update": {
                "default": {
                    "batches": max(1, int(settings.get("user_intent_min_signal_batches", 4) or 4)),
                    "facts": max(1, int(settings.get("user_intent_min_signal_facts", 5) or 5)),
                },
                "interaction": {
                    "batches": max(1, int(settings.get("interaction_user_intent_min_signal_batches", 3) or 3)),
                    "facts": max(1, int(settings.get("interaction_user_intent_min_signal_facts", 3) or 3)),
                },
            },
            "memory_user_observed_pattern_update": {
                "default": {
                    "batches": max(1, int(settings.get("user_action_min_signal_batches", 4) or 4)),
                    "facts": max(1, int(settings.get("user_action_min_signal_facts", 8) or 8)),
                },
                "interaction": {
                    "batches": max(1, int(settings.get("interaction_user_action_min_signal_batches", 3) or 3)),
                    "facts": max(1, int(settings.get("interaction_user_action_min_signal_facts", 5) or 5)),
                },
            },
        }
        self._fact_followup_state = {
            task_kind: {
                "generation": 0,
                "unscheduled_signal_batches": 0,
                "unscheduled_signal_facts": 0,
                "unscheduled_contains_interaction": False,
            }
            for task_kind in self._FACT_FOLLOWUP_TASK_KINDS
        }
        self._latest_input_time_end = ""
        self.register_handler(
            "memory_flush_fact_followups",
            self._process_flush_fact_followup_tasks,
        )

    @property
    def execution_lock(self) -> threading.RLock:
        """Shared lock for queued writes and synchronized domain reads."""
        return self._execution_lock

    @property
    def operation_reporter(self) -> TaskOperationReporter:
        """Reporter shared with synchronous operations such as recall."""
        return self._operation_reporter

    def _select_pending_signal_ids(self, target: str) -> List[int]:
        if self._task_execution_db is None or self._claim_state_database is None:
            return []
        signal_type = {
            "user_intent_update": "intent",
            "user_action_signal_consolidation": "action",
        }.get(target, "claim")
        pending_ids = self._task_execution_db.pending_signal_ids(target)
        if not pending_ids:
            return []
        with self._claim_state_database.reader_transaction() as reader_db:
            pending_signals = reader_db.get_user_signals_by_ids(
                signal_type, pending_ids,
            )
        valid_ids = {int(row["signal_id"]) for row in pending_signals}
        missing = set(pending_ids) - valid_ids
        if missing:
            self._task_execution_db.mark_signal_ids_completed(target, sorted(missing))
        return [signal_id for signal_id in pending_ids if signal_id in valid_ids]

    def _prepare_signal_task_payload(
        self, task_kind: str, payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Freeze pending signal IDs; task handlers enforce their own limits."""
        prepared = dict(payload)
        if task_kind == "memory_user_claim_update":
            if "signal_ids" not in prepared:
                prepared["signal_ids"] = self._select_pending_signal_ids(
                    "explicit_user_claim",
                )
        elif task_kind == "memory_user_intent_update":
            if "signal_ids" not in prepared:
                prepared["signal_ids"] = self._select_pending_signal_ids(
                    "user_intent_update",
                )
        elif task_kind == "memory_user_observed_pattern_update":
            if "signal_ids" not in prepared:
                prepared["signal_ids"] = self._select_pending_signal_ids(
                    "user_action_signal_consolidation",
                )
        return prepared

    def _complete_signal_task_result(
        self, task_kind: str, payload: Dict[str, Any], result: Any,
    ) -> None:
        """Complete the submitted snapshot and release unfinished IDs."""
        target = self._SIGNAL_TASK_TARGETS.get(task_kind)
        if target is None or self._task_execution_db is None:
            return
        completed_ids = result.pop("_completed_signal_ids", []) if isinstance(result, dict) else []
        if (task_kind == "memory_user_observed_pattern_update"
                and isinstance(result, dict)
                and str(result.get("status") or "").lower() not in {"ok", "empty"}):
            completed_ids = []
        completed = self._task_execution_db.finish_signal_task(
            target, payload.get("signal_ids") or [], completed_ids,
        )
        if isinstance(result, dict):
            result["signals_marked_processed"] = len(completed)

    def record_pending_user_state_claims(
        self, claim_ids: List[int], *,
        claim_relations: Sequence[Dict[str, Any]] = (),
        reference_timestamp: Any = None,
        source_task_id: str = "",
    ) -> int:
        if self._task_execution_db is None:
            return 0
        return self._task_execution_db.record_pending_user_state_claims(
            claim_ids,
            claim_relations=claim_relations,
            reference_timestamp=reference_timestamp,
            source_task_id=source_task_id,
        )

    def register_handler(self, task_kind: str, handler: TaskHandler) -> None:
        """Register a domain handler before its first task is submitted."""
        with self._worker_lock:
            if task_kind in self._handlers:
                raise ValueError(f"Task handler already registered: {task_kind}")
            self._handlers[task_kind] = handler

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

    def start(self) -> None:
        """Start the shared worker after all task handlers are registered."""
        with self._worker_lock:
            if not self._shutdown_event.is_set():
                self._ensure_worker_locked()
        if self._claim_state_database is not None:
            self.trigger_user_state_time_tick()

    def submit(
        self,
        task_kind: str,
        payload: Dict[str, Any],
        *,
        completion_context: Optional[Dict[str, Any]] = None,
        report_payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Enqueue and report a task, or return a recorded rejection."""
        # Enqueueing must not wait for the worker's long-running LLM handler.
        # The worker still holds _execution_lock for its shared writer connection;
        # signal validation above uses its own read-only connection.
        with self._worker_lock:
            task_id = self._operation_reporter.next_task_id(task_kind)
            if self._shutdown_event.is_set():
                reason = "worker_shutdown"
            elif not self._memory_enabled:
                reason = "memory_disabled"
            elif task_kind in self._task_enabled and not self._task_enabled[task_kind]:
                reason = "task_disabled"
            elif task_kind not in self._handlers:
                reason = "task_handler_not_registered"
            else:
                reason = None
            if reason is not None:
                return self._reject_locked(task_kind, task_id, reason)
            task_payload = self._prepare_signal_task_payload(task_kind, payload)
            task_completion_context = dict(completion_context or {})
            task = {
                "kind": task_kind,
                "payload": task_payload,
                "task_id": task_id,
                "started_at": time.monotonic(),
                "completion_context": task_completion_context,
            }
            if self._queue.full():
                return self._reject_locked(task_kind, task_id, "worker_queue_full")
            if task_kind == "memory_user_thread_update":
                if self._task_execution_db is None:
                    return self._reject_locked(task_kind, task_id, "task_execution_database_unavailable")
                activity_ids = task_payload.get("activity_ids")
                if activity_ids is None:
                    activity_ids = self._task_execution_db.pending_user_thread_activity_ids()
                task_payload["activity_ids"] = self._task_execution_db.reserve_user_thread_activities(
                    activity_ids, task_id=task_id,
                )
                if not task_payload["activity_ids"]:
                    return self._reject_locked(task_kind, task_id, "no_pending_activities")
            signal_target = self._SIGNAL_TASK_TARGETS.get(task_kind)
            if signal_target is not None:
                signal_ids = task_payload.get("signal_ids") or []
                if not signal_ids:
                    return self._reject_locked(task_kind, task_id, "no_pending_signals")
                if self._task_execution_db is None:
                    return self._reject_locked(task_kind, task_id, "task_execution_database_unavailable")
                try:
                    self._task_execution_db.reserve_signal_ids(signal_target, signal_ids)
                except ValueError:
                    return self._reject_locked(task_kind, task_id, "signals_already_submitted")
                except sqlite3.Error:
                    self._logger.exception(
                        "Failed to reserve signal IDs kind=%s id=%s", task_kind, task_id,
                    )
                    return self._reject_locked(task_kind, task_id, "signal_reservation_failed")
            try:
                self._operation_reporter.on_task_submitted(
                    operation_type=task_kind,
                    task_id=task_id,
                    payload=report_payload if report_payload is not None else task_payload,
                )
            except Exception:
                self._logger.exception(
                    "Failed to report task submission kind=%s id=%s",
                    task_kind,
                    task_id,
                )
            if task_kind == "memory_fact_extraction":
                self._record_input_time_end(
                    task["completion_context"].get("input_time_end")
                )
            if self._task_execution_db is not None:
                try:
                    self._task_execution_db.record_task_submission_info(
                        task_id, task_kind,
                    )
                except Exception:
                    self._logger.exception(
                        "Failed to persist task submission kind=%s id=%s",
                        task_kind, task_id,
                    )
            # Publish only after the reporter and task DB know the task. The
            # worker may take it immediately after put_nowait returns.
            self._queue.put_nowait(task)
            self._logger.info(
                "scheduled task queued task_kind=%s task_id=%s queue_depth=%s",
                task_kind, task_id, self._queue.qsize(),
            )
            self._ensure_worker_locked()
        return {
            "queued": True,
            "status": "queued",
            "task_id": task_id,
            "operation_type": task_kind,
        }

    def reject(self, task_kind: str, reason: str) -> Dict[str, Any]:
        """Record a domain-level rejection before a task enters the queue."""
        with self._worker_lock:
            task_id = self._operation_reporter.next_task_id(task_kind)
            return self._reject_locked(task_kind, task_id, reason)

    def _reject_locked(
        self, task_kind: str, task_id: str, reason: str,
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

    def set_logger(self, logger: Optional[logging.Logger]) -> None:
        self._logger = logger or logging.getLogger(__name__)

    def flush(self, timeout: Optional[float] = None) -> bool:
        """Wait until every accepted task has completed."""
        started_at = time.monotonic()
        deadline = None if timeout is None else started_at + max(0.0, timeout)
        next_progress_log_at = started_at + 30.0
        while self._queue.unfinished_tasks:
            worker = self._worker_thread
            if worker is not None and not worker.is_alive():
                return False
            now = time.monotonic()
            if deadline is not None and now >= deadline:
                return False
            if now >= next_progress_log_at:
                self._logger.warning(
                    "scheduled task drain waiting elapsed_seconds=%.1f "
                    "unfinished_tasks=%s queue_depth=%s worker_alive=%s",
                    now - started_at, self._queue.unfinished_tasks,
                    self._queue.qsize(), bool(worker and worker.is_alive()),
                )
                next_progress_log_at = now + 30.0
            time.sleep(0.01)
        return True

    def shutdown(
        self,
        *,
        wait: bool = True,
        timeout: Optional[float] = None,
    ) -> bool:
        """Stop accepting tasks and optionally drain and join the worker."""
        with self._worker_lock:
            self._shutdown_event.set()
            worker = self._worker_thread
        if worker is None:
            if self._task_execution_db is not None:
                self._task_execution_db.close()
                self._task_execution_db = None
            return not self._queue.unfinished_tasks
        if not wait:
            return not self._queue.unfinished_tasks

        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        while self._queue.unfinished_tasks:
            if not worker.is_alive():
                return False
            if deadline is not None and time.monotonic() >= deadline:
                return False
            time.sleep(0.01)

        remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
        worker.join(timeout=remaining)
        stopped = not worker.is_alive() and not self._queue.unfinished_tasks
        if stopped and self._task_execution_db is not None:
            self._task_execution_db.close()
            self._task_execution_db = None
        return stopped

    def _ensure_worker_locked(self) -> None:
        if self._worker_thread and self._worker_thread.is_alive():
            return
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name=self._worker_name,
        )
        self._worker_thread.start()

    def _worker_loop(self) -> None:
        while not self._shutdown_event.is_set() or not self._queue.empty():
            try:
                task = self._queue.get(timeout=0.2)
            except queue.Empty:
                self._maybe_tick_claim_state_window()
                continue
            try:
                self._execute_scheduled_task(task)
            except Exception:
                self._logger.exception(
                    "Scheduled task handler failed task_kind=%s task_id=%s",
                    task.get("kind"),
                    task.get("task_id"),
                )
            finally:
                self._queue.task_done()
            self._maybe_tick_claim_state_window()

    def _maybe_tick_claim_state_window(self) -> None:
        if self._claim_state_database is None:
            return
        now = time.monotonic()
        if now < self._next_claim_state_tick_at:
            return
        self._next_claim_state_tick_at = now + self._claim_state_window_seconds
        try:
            self.trigger_user_state_time_tick()
        except Exception:
            self._logger.exception("Claim state time-window tick failed")

    def _execute_scheduled_task(self, task: Dict[str, Any]) -> None:
        """Execute, apply result bookkeeping, report, then schedule follow-ups."""
        task_kind = str(task["kind"])
        task_id = str(task["task_id"])
        result: Any = None
        error: Optional[BaseException] = None
        handler_started_at = time.monotonic()
        self._logger.info(
            "scheduled task started task_kind=%s task_id=%s queue_wait_ms=%.1f",
            task_kind, task_id,
            (handler_started_at - float(task["started_at"])) * 1000,
        )
        try:
            with self._execution_lock:
                result = self._handlers[task_kind](**task["payload"])
        except Exception as exc:
            error = exc
            self._logger.exception(
                "Scheduled task failed task_kind=%s task_id=%s", task_kind, task_id,
            )
        handler_elapsed_ms = (time.monotonic() - handler_started_at) * 1000
        event = {
            "task_kind": task_kind,
            "task_id": task_id,
            "payload": task["payload"],
            "result": result,
            "error": error,
            "succeeded": error is None and self._operation_reporter._operation_succeeded(
                task_kind, result,
            ),
            "completion_context": dict(task.get("completion_context") or {}),
        }
        completion: Dict[str, Any] = {"postprocessed": False}
        try:
            completion = self._postprocess_completed_task(event)
        except Exception as exc:
            error = error or exc
            event["error"] = error
            event["succeeded"] = False
            self._logger.exception(
                "Scheduled task postprocessing failed task_kind=%s task_id=%s",
                task_kind, task_id,
            )
        try:
            succeeded = self._operation_reporter.on_task_finished(
                operation_type=task_kind,
                task_id=task_id,
                started_at=float(task["started_at"]),
                result=result,
                error=error,
            )
        except Exception as exc:
            succeeded = False
            error = error or exc
            self._logger.exception(
                "Failed to report task completion task_kind=%s task_id=%s",
                task_kind, task_id,
            )
        event["error"] = error
        event["succeeded"] = bool(succeeded)
        result_status = str(result.get("status") or "")[:80] if isinstance(result, dict) else ""
        self._logger.info(
            "scheduled task handler finished task_kind=%s task_id=%s succeeded=%s "
            "result_status=%s handler_elapsed_ms=%.1f total_elapsed_ms=%.1f",
            task_kind, task_id, bool(succeeded), result_status,
            handler_elapsed_ms, (time.monotonic() - float(task["started_at"])) * 1000,
        )
        if self._task_execution_db is not None:
            try:
                self._task_execution_db.record_task_completion_info(
                    task_id,
                    status=(result_status or "ok") if succeeded else "failed",
                    result=result if error is None else {
                        "error_type": type(error).__name__,
                    },
                )
            except Exception:
                self._logger.exception(
                    "Failed to persist task completion kind=%s id=%s",
                    task_kind, task_id,
                )
        self._schedule_completed_task(event, completion)
        self._logger.info(
            "scheduled task follow-up finished task_kind=%s task_id=%s queue_depth=%s",
            task_kind, task_id, self._queue.qsize(),
        )

    def trigger_user_state_time_tick(
        self, *, now: Optional[datetime] = None, force: bool = False,
    ) -> Dict[str, Any]:
        """Queue changed claims after the time window, or immediately on final flush."""
        if self._task_execution_db is None:
            return {"queued": False, "reason": "task_execution_database_unavailable"}
        effective_now = now or datetime.now().astimezone()
        if effective_now.tzinfo is None:
            effective_now = effective_now.astimezone()
        with self._execution_lock:
            with self._claim_state_lock:
                if self._claim_state_task_pending:
                    if force:
                        self._claim_state_force_requested = True
                    return {"queued": False, "reason": "claim_state_task_pending"}
                pending = self._task_execution_db.get_pending_user_state_claims(
                    limit=self._claim_state_batch_limit,
                )
                items = pending["items"]
                if not items:
                    return {"queued": False, "reason": "no_changed_claims"}
                first_detected = datetime.fromisoformat(
                    pending["first_detected_at"].replace("Z", "+00:00")
                )
                if first_detected.tzinfo is None:
                    first_detected = first_detected.astimezone()
                age_seconds = max(
                    0.0, (effective_now - first_detected).total_seconds(),
                )
                if not force and age_seconds < self._claim_state_window_seconds:
                    self._logger.info(
                        "claim state window deferred reason=window_not_elapsed "
                        "pending_claims=%s age_seconds=%.1f",
                        pending["pending_count"], age_seconds,
                    )
                    return {
                        "queued": False, "reason": "claim_window_not_elapsed",
                        "pending_claim_count": pending["pending_count"],
                    }
                if not force and (
                    pending["pending_count"] < self._claim_state_min_changed_claims
                    and age_seconds < self._claim_state_max_wait_seconds
                ):
                    self._logger.info(
                        "claim state window deferred reason=threshold_not_met "
                        "pending_claims=%s age_seconds=%.1f",
                        pending["pending_count"], age_seconds,
                    )
                    return {
                        "queued": False, "reason": "claim_threshold_not_met",
                        "pending_claim_count": pending["pending_count"],
                    }
                versions = [
                    (int(item["claim_id"]), int(item["generation"]))
                    for item in items
                ]
                newest = max(items, key=lambda item: item["last_detected_at"])
                with self._worker_lock:
                    latest_input_time_end = self._latest_input_time_end
                reference_timestamp = (
                    newest["reference_timestamp"] or latest_input_time_end or None
                )
                self._claim_state_task_pending = True
        state_payload = {
            "new_claim_ids": [claim_id for claim_id, _ in versions],
            "claim_relations": pending["claim_relations"],
            "reference_timestamp": reference_timestamp,
        }
        report = self.submit(
            "user_state_update_from_user_claims",
            state_payload,
            completion_context={
                "claim_state_window_versions": versions,
                "claim_state_final_flush": force,
            },
            report_payload={
                "new_claim_ids": state_payload["new_claim_ids"],
                "relation_event_count": len(state_payload["claim_relations"]),
                "reference_timestamp": reference_timestamp,
            },
        )
        if not report.get("queued"):
            with self._claim_state_lock:
                self._claim_state_task_pending = False
        self._logger.info(
            "claim state window evaluated pending_claims=%s selected_claims=%s "
            "age_seconds=%.1f force=%s queued=%s reason=%s",
            pending["pending_count"], len(versions), age_seconds,
            force, bool(report.get("queued")), report.get("reason") or "",
        )
        return {
            **report,
            "pending_claim_count": pending["pending_count"],
            "selected_claim_count": len(versions),
            "oldest_change_age_seconds": round(age_seconds, 1),
        }

    def submit_flush_fact_followup_tasks(self) -> Dict[str, Any]:
        """Queue a FIFO boundary; resolve pending signals only when it executes."""
        return self.submit("memory_flush_fact_followups", {})

    def _process_flush_fact_followup_tasks(self) -> Dict[str, Any]:
        """Submit unreserved signals produced by preceding fact tasks."""
        followup_reports = self._submit_final_fact_followup_tasks()
        failed_task_kinds = [
            task_kind
            for task_kind, report in followup_reports.items()
            if not report.get("queued")
            and report.get("reason") not in {"no_pending_signals", "task_disabled"}
        ]
        self._logger.info(
            "final memory follow-ups submitted claim_queued=%s "
            "observed_pattern_queued=%s intent_queued=%s",
            bool(followup_reports["memory_user_claim_update"].get("queued")),
            bool(followup_reports["memory_user_observed_pattern_update"].get("queued")),
            bool(followup_reports["memory_user_intent_update"].get("queued")),
        )
        return {
            "status": "failed" if failed_task_kinds else "ok",
            "fact_followup_tasks": followup_reports,
            "failed_task_kinds": failed_task_kinds,
        }

    def _submit_final_fact_followup_tasks(self) -> Dict[str, Dict[str, Any]]:
        with self._worker_lock:
            resolved_timestamp = self._latest_input_time_end
        reports: Dict[str, Dict[str, Any]] = {}
        for task_kind in self._FACT_FOLLOWUP_TASK_KINDS:
            payload: Dict[str, Any] = {
                "reference_timestamp": resolved_timestamp or None,
            }
            reports[task_kind] = self.submit(
                task_kind,
                payload,
                report_payload=payload,
            )
        return reports

    def _postprocess_completed_task(self, event: Dict[str, Any]) -> Dict[str, Any]:
        task_kind = str(event.get("task_kind") or "")
        postprocessors = {
            "memory_fact_extraction": self._postprocess_fact_extraction_task,
            "memory_user_activity_extraction": self._postprocess_user_activity_task,
            "memory_user_thread_update": self._postprocess_user_thread_update_task,
            "memory_user_claim_update": self._postprocess_user_claim_update_task,
            "memory_user_observed_pattern_update": self._postprocess_fact_followup_task,
            "memory_user_intent_update": self._postprocess_fact_followup_task,
            "user_state_update_from_user_claims": (
                self._postprocess_user_state_update_from_user_claims_task
            ),
            "user_prediction_update_from_intent_events": self._postprocess_user_prediction_task,
        }
        completion: Dict[str, Any] = {"postprocessed": False}
        postprocess = postprocessors.get(task_kind)
        if postprocess is not None:
            completion.update(postprocess(event) or {})
        completion["postprocessed"] = True
        return completion

    def _schedule_completed_task(
        self, event: Dict[str, Any], completion: Dict[str, Any],
    ) -> None:
        task_kind = str(event.get("task_kind") or "")
        schedulers = {
            "memory_fact_extraction": self._schedule_after_fact_extraction_task,
            "memory_user_activity_extraction": self._schedule_after_user_activity_task,
            "memory_user_thread_update": self._schedule_after_user_thread_update_task,
            "memory_user_observed_pattern_update": (
                self._schedule_after_user_observed_pattern_update_task
            ),
            "memory_user_intent_update": self._schedule_after_user_intent_update_task,
            "user_state_update_from_user_claims": (
                self._schedule_after_user_state_update_from_user_claims_task
            ),
            "user_prediction_update_from_intent_events": self._schedule_after_user_prediction_task,
        }
        schedule = schedulers.get(task_kind)
        if schedule is None:
            return
        if completion.get("postprocessed"):
            schedule(event, completion)

    def _postprocess_fact_extraction_task(
        self, event: Dict[str, Any],
    ) -> None:
        result = event.get("result")
        if (isinstance(result, dict) and self._task_execution_db is not None
                and self._claim_state_database is not None):
            signal_references = self._claim_state_database.get_user_signal_batches(
                result.get("fact_ids") or [],
            )
            self._task_execution_db.register_signals(
                **signal_references,
                source_task_id=str(event.get("task_id") or ""),
            )
        if event.get("succeeded") and isinstance(result, dict):
            self._record_fact_extraction_task_result(
                result, completion_context=event.get("completion_context"),
            )

    def _schedule_after_fact_extraction_task(
        self, event: Dict[str, Any], completion: Dict[str, Any],
    ) -> None:
        if (completion["postprocessed"] and event.get("succeeded")
                and isinstance(event.get("result"), dict)):
            for task_kind in self._FACT_FOLLOWUP_TASK_KINDS:
                self._schedule_fact_followup_if_eligible(task_kind)

    def _postprocess_user_activity_task(self, event: Dict[str, Any]) -> None:
        result = event.get("result")
        if event.get("succeeded") and isinstance(result, dict) and self._task_execution_db:
            self._task_execution_db.register_user_thread_activities(
                result.get("user_activity_item_ids") or [],
            )

    def _postprocess_user_thread_update_task(self, event: Dict[str, Any]) -> None:
        if self._task_execution_db is None:
            return
        result = event.get("result")
        # Earlier batches may have committed even if a later batch failed.
        completed = result.get("processed_activity_ids", []) if isinstance(result, dict) else []
        self._task_execution_db.finish_user_thread_activities(
            task_id=event["task_id"], activity_ids=event["payload"].get("activity_ids") or [],
            completed_ids=completed,
        )

    def _schedule_after_user_activity_task(
        self, event: Dict[str, Any], completion: Dict[str, Any],
    ) -> None:
        result = event.get("result")
        if not event.get("succeeded") or not isinstance(result, dict):
            return
        activity_ids = result.get("user_activity_item_ids") or []
        if not activity_ids:
            return
        if self._task_enabled["memory_user_thread_update"]:
            report = self.submit("memory_user_thread_update", {})
            if not report.get("queued"):
                self._logger.warning("User thread update not queued: %s", report)

    def _schedule_after_user_thread_update_task(
        self, event: Dict[str, Any], completion: Dict[str, Any],
    ) -> None:
        result = event.get("result")
        if not self._task_enabled["user_state_update_from_user_threads"]:
            return
        if event.get("error") is not None or not isinstance(result, dict):
            return
        changes = result.get("thread_changes") or []
        if not changes:
            return
        # A failed later batch may still have committed earlier changes. Those
        # completed batches also need their state projection.
        report = self.submit(
            "user_state_update_from_user_threads",
            {"thread_changes": changes, "runtime_context": {}},
            report_payload={
                "thread_ids": [change["thread"]["id"] for change in changes],
                "thread_event_ids": result.get("thread_event_ids") or [],
            },
        )
        if not report.get("queued"):
            self._logger.warning("Thread state update not queued: %s", report)

    def _postprocess_user_prediction_task(
        self, event: Dict[str, Any],
    ) -> None:
        context = event.get("completion_context") or {}
        event_ids = context.get("prediction_intent_event_ids") or []
        result = event.get("result") or {}
        with self._prediction_lock:
            self._prediction_events_inflight.difference_update(event_ids)
            if (event.get("succeeded") and isinstance(result, dict)
                    and result.get("status") in {"ok", "empty"}
                    and self._task_execution_db is not None):
                self._task_execution_db.complete_prediction_intent_events(
                    sorted(set(result.get("processed_intent_event_ids") or []) & set(event_ids)),
                )

    def _schedule_after_user_prediction_task(
        self, event: Dict[str, Any], completion: Dict[str, Any],
    ) -> None:
        result = event.get("result")
        if not event.get("succeeded") or not isinstance(result, dict):
            return
        new_prediction_ids = sorted({
            int(item["id"])
            for item in result.get("predictions") or []
            if isinstance(item, dict)
            and item.get("save_action") == "created"
            and str(item.get("id") or "").isdigit()
            and int(item["id"]) > 0
        })
        invalidated_prediction_ids = [
            int(value) for value in result.get("invalidated_prediction_ids") or []
            if str(value).isdigit() and int(value) > 0
        ]
        # Closing services for invalidated sources is cleanup, not a fresh LLM evaluation.
        prediction_ids = sorted(set(new_prediction_ids + invalidated_prediction_ids))
        if not prediction_ids:
            return
        reference_timestamp = (result.get("prediction_context") or {}).get("as_of")
        report = self.submit(
            "user_proactive_policy_evaluation",
            {
                "prediction_ids": prediction_ids,
                "source_prediction_task_id": str(event.get("task_id") or ""),
                "runtime_context": {},
                "reference_timestamp": reference_timestamp,
            },
            report_payload={
                "prediction_ids": prediction_ids,
                "source_prediction_task_id": str(event.get("task_id") or ""),
                "reference_timestamp": reference_timestamp,
            },
        )
        if not report.get("queued"):
            self._logger.warning(
                "proactive policy evaluation was not queued prediction_ids=%s reason=%s",
                prediction_ids, report.get("reason") or "unknown",
            )

    def _postprocess_user_state_update_from_user_claims_task(
        self, event: Dict[str, Any],
    ) -> Dict[str, Any]:
        context = event.get("completion_context") or {}
        versions = context.get("claim_state_window_versions")
        if not isinstance(versions, list):
            return {"completed_successfully": False, "force_requested": False}
        completed_successfully = False
        try:
            result = event.get("result")
            result_status = str(
                (result or {}).get("status")
                if isinstance(result, dict) else ""
            ).strip().lower()
            if event.get("succeeded") and result_status in {"ok", "empty"}:
                with self._execution_lock:
                    acknowledged = self._task_execution_db.acknowledge_user_state_claims(
                        versions,
                    )
                completed_successfully = True
                self._logger.info(
                    "claim state window completed task_id=%s acknowledged_claims=%s",
                    event.get("task_id"), acknowledged,
                )
            else:
                self._logger.warning(
                    "claim state window retained pending claims task_id=%s status=%s",
                    event.get("task_id"), result_status or "error",
                )
        finally:
            with self._claim_state_lock:
                self._claim_state_task_pending = False
                force_requested = self._claim_state_force_requested
                self._claim_state_force_requested = False
        return {
            "completed_successfully": completed_successfully,
            "force_requested": force_requested,
        }

    def _schedule_after_user_state_update_from_user_claims_task(
        self, event: Dict[str, Any], completion: Dict[str, Any],
    ) -> None:
        context = event.get("completion_context") or {}
        if (completion.get("completed_successfully") and (
                context.get("claim_state_final_flush")
                or completion.get("force_requested"))):
            self.trigger_user_state_time_tick(force=True)

    def _postprocess_fact_followup_task(
        self, event: Dict[str, Any],
    ) -> None:
        """Acknowledge the signal IDs completed by this task's handler."""
        task_kind = str(event.get("task_kind") or "")
        result = event.get("result")
        self._complete_signal_task_result(
            task_kind, event.get("payload") or {}, result,
        )

    def _postprocess_user_claim_update_task(
        self, event: Dict[str, Any],
    ) -> None:
        result = event.get("result")
        try:
            if event.get("succeeded") and isinstance(result, dict):
                self.record_pending_user_state_claims(
                    list(result.get("new_claim_ids") or []),
                    claim_relations=result.get("claim_relations") or [],
                    reference_timestamp=result.get("reference_timestamp"),
                    source_task_id=str(event.get("task_id") or ""),
                )
        except Exception:
            if isinstance(result, dict):
                result["_completed_signal_ids"] = []
            raise
        finally:
            self._postprocess_fact_followup_task(event)

    def _schedule_after_user_observed_pattern_update_task(
        self, event: Dict[str, Any], completion: Dict[str, Any],
    ) -> None:
        result = event.get("result")
        if (event.get("succeeded") and isinstance(result, dict)):
            new_pattern_ids = list(result.get("new_pattern_ids") or [])
            pattern_relation_event_ids = list(
                result.get("pattern_relation_event_ids") or []
            )
            if new_pattern_ids or pattern_relation_event_ids:
                payload = {
                    "new_pattern_ids": new_pattern_ids,
                    "pattern_relation_event_ids": pattern_relation_event_ids,
                    "reference_timestamp": result.get("reference_timestamp"),
                }
                self.submit(
                    "user_state_update_from_user_observed_patterns", payload,
                    report_payload={
                        "new_pattern_ids": new_pattern_ids,
                        "pattern_relation_event_ids": pattern_relation_event_ids,
                    },
                )

    def _schedule_after_user_intent_update_task(
        self, event: Dict[str, Any], completion: Dict[str, Any],
    ) -> None:
        result = event.get("result")
        if (not event.get("succeeded") or not isinstance(result, dict)
                or not result.get("intent_event_ids")):
            return
        event_ids = sorted({int(value) for value in result["intent_event_ids"] if int(value) > 0})
        with self._prediction_lock:
            completed = (
                self._task_execution_db.completed_prediction_intent_events(event_ids)
                if self._task_execution_db is not None else set()
            )
            selected = sorted(set(event_ids) - completed - self._prediction_events_inflight)
            if not selected:
                return
            payload = {
                "intent_event_ids": selected,
                "reference_timestamp": result.get("reference_timestamp"),
                "horizon_hours": self._user_prediction_horizon_hours,
            }
            report = self.submit(
                "user_prediction_update_from_intent_events", payload,
                completion_context={"prediction_intent_event_ids": selected},
                report_payload={"trigger_source": "intent_update", **payload},
            )
            if report.get("queued"):
                self._prediction_events_inflight.update(selected)

    def _record_fact_extraction_task_result(
        self, result: Dict[str, Any], *, completion_context: Any,
    ) -> None:
        context = dict(completion_context) if isinstance(completion_context, dict) else {}
        contains_interaction = bool(context.get("contains_interaction_turn"))
        signal_stats = {
            "memory_user_claim_update": {
                "signal_count": int(result.get("user_claim_signal_count") or 0),
                "signal_fact_count": int(result.get("user_claim_signal_fact_count") or 0),
            },
            "memory_user_intent_update": {
                "signal_count": int(result.get("user_intent_signal_count") or 0),
                "signal_fact_count": int(result.get("user_intent_signal_fact_count") or 0),
            },
            "memory_user_observed_pattern_update": {
                "signal_count": int(result.get("user_action_signal_count") or 0),
                "signal_fact_count": int(result.get("user_action_signal_fact_count") or 0),
            },
        }
        self._logger.info(
            "memory store completion observed facts=%s user_claim_signals=%s "
            "user_claim_signal_facts=%s user_intent_signals=%s "
            "user_intent_signal_facts=%s user_action_signals=%s "
            "user_action_signal_facts=%s contains_interaction_turn=%s input_time_end=%s",
            int(result.get("new_fact_count") or 0),
            signal_stats["memory_user_claim_update"]["signal_count"],
            signal_stats["memory_user_claim_update"]["signal_fact_count"],
            signal_stats["memory_user_intent_update"]["signal_count"],
            signal_stats["memory_user_intent_update"]["signal_fact_count"],
            signal_stats["memory_user_observed_pattern_update"]["signal_count"],
            signal_stats["memory_user_observed_pattern_update"]["signal_fact_count"],
            contains_interaction,
            context.get("input_time_end") or "",
        )
        for kind, stats in signal_stats.items():
            if stats["signal_count"] <= 0 or stats["signal_fact_count"] <= 0:
                continue
            state = self._fact_followup_state[kind]
            state["generation"] += 1
            state["unscheduled_signal_batches"] += 1
            state["unscheduled_signal_facts"] += stats["signal_fact_count"]
            state["unscheduled_contains_interaction"] = bool(
                state["unscheduled_contains_interaction"] or contains_interaction
            )

    def _schedule_fact_followup_if_eligible(self, kind: str) -> None:
        state = self._fact_followup_state[kind]
        mode = "interaction" if state["unscheduled_contains_interaction"] else "default"
        threshold = self._fact_followup_thresholds[kind][mode]
        if (
            state["unscheduled_signal_batches"] < threshold["batches"]
            or state["unscheduled_signal_facts"] < threshold["facts"]
        ):
            return
        target_generation = int(state["generation"])
        with self._worker_lock:
            reference_timestamp = self._latest_input_time_end
        followup_payload: Dict[str, Any] = {
            "reference_timestamp": reference_timestamp or None,
        }
        report = self.submit(
            kind,
            followup_payload,
            report_payload=followup_payload,
        )
        if report.get("queued"):
            state["unscheduled_signal_batches"] = 0
            state["unscheduled_signal_facts"] = 0
            state["unscheduled_contains_interaction"] = False
            self._logger.info(
                "memory fact follow-up task scheduled kind=%s target_generation=%s "
                "reference_timestamp=%s", kind, target_generation,
                reference_timestamp or "",
            )
            return
        if report.get("reason") == "no_pending_signals":
            state["unscheduled_signal_batches"] = 0
            state["unscheduled_signal_facts"] = 0
            state["unscheduled_contains_interaction"] = False
        self._logger.warning(
            "memory fact follow-up task rejected kind=%s reason=%s",
            kind, report.get("reason"),
        )

    def _record_input_time_end(self, value: Any) -> None:
        """Advance the input watermark while submit() holds _worker_lock."""
        candidate = str(value or "").strip()
        if not candidate:
            return
        current = self._latest_input_time_end
        if not current or self._is_later_timestamp(candidate, current):
            self._latest_input_time_end = candidate

    @staticmethod
    def _is_later_timestamp(candidate: str, current: str) -> bool:
        def epoch(value: str) -> Optional[float]:
            try:
                return datetime.fromisoformat(value.strip().replace("Z", "+00:00")).timestamp()
            except (TypeError, ValueError, OSError):
                return None
        candidate_epoch = epoch(candidate)
        current_epoch = epoch(current)
        if candidate_epoch is not None and current_epoch is not None:
            return candidate_epoch > current_epoch
        return candidate > current
