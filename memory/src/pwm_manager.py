"""Composition root for the personal world model."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from llm_client import LLMClient
from memory.memory_database import UserMemoryDB
from memory.memory_manager import MemoryNodeManager
from memory.memory_runtime import MemoryRuntime
from task_scheduler import TaskOperationReporter, TaskScheduler
from user_model import (
    UserModelDB, UserPrediction, UserProactiveService, UserStateResolver,
)


class PersonWorldModel:
    """Own the shared memory worker, evidence runtime, and user model."""

    def __init__(
        self,
        *,
        memory_runtime_config: Dict[str, Any],
        memory_manager_config: Dict[str, Any],
        user_state_config: Dict[str, Any],
        llm_model_config: Dict[str, Any],
        embedding_model_config: Dict[str, Any],
        task_scheduling_config: Dict[str, Any],
        operation_reporter: Optional[TaskOperationReporter] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._logger = logger or logging.getLogger(__name__)
        runtime_config = dict(memory_runtime_config)
        manager_config = dict(memory_manager_config)
        llm_config = dict(llm_model_config)
        embedding_config = dict(embedding_model_config)
        scheduler_config = dict(task_scheduling_config)
        memory_db_value = manager_config.get("user_memory_database_path")
        user_model_db_value = user_state_config.get("user_model_database_path")
        if not memory_db_value:
            raise ValueError(
                "memory_manager_config.user_memory_database_path is required"
            )
        if not user_model_db_value:
            raise ValueError(
                "user_state_config.user_model_database_path is required"
            )
        memory_db_path = Path(memory_db_value).expanduser().resolve()
        model_db_path = Path(user_model_db_value).expanduser().resolve()
        if model_db_path == memory_db_path:
            raise ValueError("User memory and user model database paths must be distinct")
        self._operation_reporter = operation_reporter or TaskOperationReporter()
        self._user_memory_database = UserMemoryDB(memory_db_path)
        try:
            self._user_model_database = UserModelDB(model_db_path)
            self._llm_client = LLMClient(
                memory_config={"llm": llm_config},
                logger=self._logger.getChild("llm_client"),
            )
            # Both memory and user-state handlers are registered before the
            # shared worker starts processing its FIFO.
            self._task_scheduler = TaskScheduler(
                operation_reporter=self._operation_reporter,
                maxsize=int(scheduler_config.get("task_queue_maxsize", 100) or 100),
                logger=self._logger.getChild("task_scheduler"),
                claim_state_database=self._user_memory_database,
                task_scheduling_config=scheduler_config,
                memory_enabled=manager_config.get("memory_enabled", True),
            )
            self._memory_runtime = MemoryRuntime(
                db_path=memory_db_path,
                database=self._user_memory_database,
                task_scheduler=self._task_scheduler,
                llm_client=self._llm_client,
                memory_runtime_config=runtime_config,
                memory_manager_config=manager_config,
                embedding_config=embedding_config,
                logger=self._logger.getChild("memory_runtime"),
            )
            self._user_state_resolver = UserStateResolver(
                memory_db=self._user_memory_database,
                model_db=self._user_model_database,
                task_scheduler=self._task_scheduler,
                llm_client=self._llm_client,
                user_state_config=user_state_config,
                world_owner_entity_name=(
                    self._memory_runtime.manager.world_owner_entity_name
                ),
                logger=self._logger.getChild("user_state_resolver"),
            )
            self._user_prediction = UserPrediction(
                db=self._user_memory_database,
                model_db=self._user_model_database,
                task_scheduler=self._task_scheduler,
                llm_client=self._llm_client,
                world_owner_entity_name=(
                    self._memory_runtime.manager.world_owner_entity_name
                ),
                logger=self._logger.getChild("user_prediction"),
            )
            self._user_proactive_service = UserProactiveService(
                memory_db=self._user_memory_database,
                model_db=self._user_model_database,
                prediction_reader=self._user_prediction.get_active_predictions,
                task_scheduler=self._task_scheduler,
                llm_client=self._llm_client,
                world_owner_entity_name=self._memory_runtime.manager.world_owner_entity_name,
                logger=self._logger.getChild("user_proactive_service"),
            )
            self._task_scheduler.start()
        except Exception:
            if hasattr(self, "_task_scheduler"):
                self._task_scheduler.shutdown(wait=True, timeout=0)
            if hasattr(self, "_user_model_database"):
                self._user_model_database.close()
            self._user_memory_database.close()
            raise
        self._closed = False
        self._memory_inputs_need_flush = False

    @property
    def memory_runtime(self) -> MemoryRuntime:
        return self._memory_runtime

    @property
    def memory_manager(self) -> MemoryNodeManager:
        return self._memory_runtime.manager

    @property
    def user_state_resolver(self) -> UserStateResolver:
        return self._user_state_resolver

    @property
    def user_prediction(self) -> UserPrediction:
        return self._user_prediction

    @property
    def user_proactive_service(self) -> UserProactiveService:
        return self._user_proactive_service

    @property
    def task_scheduler(self) -> TaskScheduler:
        return self._task_scheduler

    @property
    def llm_client(self) -> LLMClient:
        return self._llm_client

    @property
    def database(self) -> UserMemoryDB:
        return self._user_memory_database

    @property
    def user_model_database(self) -> UserModelDB:
        return self._user_model_database

    @property
    def operation_reporter(self) -> TaskOperationReporter:
        return self._operation_reporter

    def accept_memory_input(
        self,
        *,
        interaction_turn: Optional[Dict[str, Any]] = None,
        transcript_segments: Optional[Sequence[Dict[str, Any]]] = None,
        tags: Optional[List[str]] = None,
        ambient_recording_enabled: bool = False,
    ) -> Dict[str, Any]:
        report = self._memory_runtime.accept_memory_input(
            interaction_turn=interaction_turn,
            transcript_segments=transcript_segments,
            tags=tags,
            ambient_recording_enabled=ambient_recording_enabled,
        )
        if report.get("reason") not in {
            "expected_one_input_kind", "memory_disabled", "empty_transcript_batch",
        }:
            self._memory_inputs_need_flush = True
        return report

    def update_ambient_asr_watermark(
        self,
        value: Any,
        *,
        evaluate_user_activity: bool = True,
    ) -> Dict[str, Any]:
        """Release buffered inputs once ambient ASR has covered their time range."""
        return self._memory_runtime.update_ambient_asr_watermark(
            value,
            evaluate_user_activity=evaluate_user_activity,
        )

    def finalize_memory_session(
        self,
        *,
        reason: str = "explicit_finalize",
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        report = self._memory_runtime.flush_pending_memory_inputs(
            reason=reason,
            tags=tags,
        )
        input_flush = report.get("input_flush") or {}
        subreports = [
            input_flush,
            input_flush.get("user_activity") or {},
            input_flush.get("fact_followup_dispatch") or {},
        ]
        if bool(report.get("accepted", False)) and not any(
            item.get("reason") in {
                "queue_rejected", "worker_queue_full", "worker_shutdown",
                "task_handler_not_registered",
            }
            for item in subreports
            if isinstance(item, dict)
        ):
            self._memory_inputs_need_flush = False
        return report

    def trigger_memory_recall(
        self,
        query: str,
        *,
        tags: Optional[List[str]] = None,
        time_end: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self._memory_runtime.trigger_memory_recall(
            query,
            tags=tags,
            time_end=time_end,
        )

    def wait_for_memory_tasks(self, timeout: Optional[float] = None) -> bool:
        return self._task_scheduler.flush(timeout=timeout)

    def trigger_user_state_time_tick(
        self, *, now: Optional[datetime] = None, force: bool = False,
    ) -> Dict[str, Any]:
        """Evaluate the claim-to-state window, optionally draining it for tests."""
        return self._task_scheduler.trigger_user_state_time_tick(now=now, force=force)

    def get_user_states(
        self,
        *,
        state_domains: Optional[Sequence[str]] = None,
        state_types: Optional[Sequence[str]] = None,
        statuses: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Read the current user model through the resolver."""
        with self._task_scheduler.execution_lock:
            return self._user_state_resolver.get_current_states(
                state_domains=state_domains,
                state_types=state_types,
                statuses=statuses,
            )

    def get_active_predictions(
        self, *, as_of: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        with self._task_scheduler.execution_lock:
            return self._user_prediction.get_active_predictions(as_of=as_of)

    def record_prediction_outcome(
        self, *, prediction_id: int, status: str,
        observed_at: Optional[datetime] = None,
        evidence_refs: Optional[Sequence[Dict[str, Any]]] = None,
        note: str = "",
    ) -> bool:
        with self._task_scheduler.execution_lock:
            return self._user_prediction.record_prediction_outcome(
                prediction_id=prediction_id, status=status,
                observed_at=observed_at, evidence_refs=evidence_refs, note=note,
            )

    def get_proactive_services(
        self, *, statuses: Optional[Sequence[str]] = None, limit: int = 100,
    ) -> List[Dict[str, Any]]:
        with self._task_scheduler.execution_lock:
            owner_name = self.memory_manager.world_owner_entity_name
            owner_id = self._user_memory_database.get_entity_id_by_name(owner_name)
            if owner_id is None:
                return []
            return self._user_proactive_service.get_services(
                user_entity_id=owner_id, statuses=statuses, limit=limit,
            )

    def close(self, timeout: Optional[float] = 30.0) -> None:
        if self._closed:
            return
        if self._memory_inputs_need_flush:
            self.finalize_memory_session(reason="pwm_close")
            if self._memory_inputs_need_flush:
                raise RuntimeError("Pending memory inputs could not be submitted")
        safe_to_close_database = False
        try:
            if not self._task_scheduler.flush(timeout=timeout):
                self._logger.warning(
                    "PWM tasks did not drain within timeout=%s; continuing to drain",
                    timeout,
                )
                if not self._task_scheduler.flush(timeout=None):
                    raise RuntimeError("PWM task worker stopped with pending tasks")
            stopped = self._task_scheduler.shutdown(wait=True, timeout=timeout)
            if not stopped:
                stopped = self._task_scheduler.shutdown(wait=True, timeout=None)
            if not stopped:
                raise RuntimeError("PWM task worker did not stop cleanly")
            safe_to_close_database = True
        finally:
            if safe_to_close_database:
                self._user_model_database.close()
                self._user_memory_database.close()
                self._closed = True
