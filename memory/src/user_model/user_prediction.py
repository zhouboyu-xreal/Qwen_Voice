"""Intent-event-driven forecasts of user preparation behavior."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

from prompts_en import (
    USER_PREDICTION_NEAR_TERM_PREPARATION_PROMPT_EN,
    USER_PREDICTION_INTENT_UPDATE_PROMPT_EN,
)
from prompts_zh import (
    USER_PREDICTION_NEAR_TERM_PREPARATION_PROMPT_ZH,
    USER_PREDICTION_INTENT_UPDATE_PROMPT_ZH,
)

if TYPE_CHECKING:
    from llm_client import LLMClient
    from memory.memory_database import UserMemoryDB
    from task_scheduler import TaskScheduler
    from .user_model_database import UserModelDB


class UserPrediction:
    """Generate, inspect and verify preparation forecasts without changing intent."""

    TASK_KIND = "user_prediction_update_from_intent_events"
    MODEL_VERSION = "intent_preparation_v5"
    ACTIVE_INTENT_STATUSES = {
        "goal": {"active"}, "plan": {"planned", "rescheduled"},
        "work_item": {"open", "in_progress", "blocked"},
    }

    def __init__(
        self, *, db: UserMemoryDB, model_db: UserModelDB, llm_client: LLMClient,
        task_scheduler: TaskScheduler,
        world_owner_entity_name: str = "用户本人",
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._db = db
        self._model_db = model_db
        self._llm_client = llm_client
        self._task_scheduler = task_scheduler
        self._owner_name = world_owner_entity_name
        self._logger = logger or logging.getLogger(__name__)
        task_scheduler.register_handler(
            self.TASK_KIND, self.process_user_prediction_update_from_intent_events_task,
        )

    def _log_event(self, event: str, payload: Dict[str, Any], *, level: int = logging.INFO) -> None:
        if self._logger.isEnabledFor(level):
            self._logger.log(level, "\n%s", json.dumps(
                {"scope": self.TASK_KIND, "event": event, "payload": payload},
                ensure_ascii=False, indent=2, default=str,
            ))

    @staticmethod
    def _as_datetime(value: Any, reference: datetime) -> Optional[datetime]:
        if isinstance(value, datetime):
            result = value
        else:
            text = str(value or "").strip()
            if not text:
                return None
            try:
                result = datetime.fromisoformat(text.replace("Z", "+00:00"))
            except ValueError:
                return None
        if result.tzinfo is None:
            result = result.replace(tzinfo=reference.tzinfo)
        return result.astimezone(reference.tzinfo)

    @staticmethod
    def _now(as_of: Optional[datetime]) -> datetime:
        local = datetime.now().astimezone()
        if as_of is None:
            return local
        if as_of.tzinfo is None:
            return as_of.replace(tzinfo=local.tzinfo)
        return as_of

    @staticmethod
    def _hash(value: Any) -> str:
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _text_list(value: Any) -> List[str]:
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()][:8]

    @staticmethod
    def _intent_anchor_text(intent: Dict[str, Any]) -> str:
        object_type = intent.get("object_type")
        if object_type == "plan":
            return str(intent.get("start_at") or "")
        if object_type == "work_item":
            return str(intent.get("due_at") or "")
        return str(intent.get("target_at") or "")

    def _intent_snapshot(self, intent: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "object_type": intent["object_type"], "id": int(intent["id"]),
            "summary": intent.get("summary") or "",
            "anchor_at": self._intent_anchor_text(intent),
            "time_start": intent.get("time_start") or "",
            "time_end": intent.get("time_end") or "",
            "time_precision": intent.get("time_precision") or "unknown",
            "primary_domain": intent.get("primary_domain") or "general",
            "status": intent.get("status") or "", "details": intent.get("details") or {},
            "updated_at": intent.get("updated_at") or "",
        }

    def _build_user_prediction_prediction_inputs_using_intents(
        self, *, owner_id: int, as_of: datetime,
        horizon_end: datetime, intent_event_ids: Sequence[int],
    ) -> Dict[str, Any]:
        events = self._db.get_user_intent_events(event_ids=intent_event_ids)
        if {int(event["id"]) for event in events} != set(intent_event_ids):
            raise ValueError("Prediction input contains missing intent events")
        intent_ids = sorted({int(event["intent_id"]) for event in events})
        rows = self._db.get_user_intent_items_by_ids(intent_ids)
        if {int(intent["id"]) for intent in rows} != set(intent_ids):
            raise ValueError("Prediction event references a missing intent")
        if any(
            int(intent.get("world_owner_entity_id") or 0) != owner_id
            or int(intent.get("primary_entity_id") or 0) != owner_id
            for intent in rows
        ):
            raise ValueError("Prediction events must refer to the world owner's intents")
        intents = [self._intent_snapshot(intent) for intent in rows]
        fact_ids_by_intent = {}
        for intent in intents:
            key = (intent["object_type"], intent["id"])
            # Query each intent separately so the shared query limit cannot starve another case.
            fact_ids_by_intent[intent["id"]] = self._db.get_intent_evidence_fact_ids([key]).get(key, [])
        fact_ids = sorted({fact_id for ids in fact_ids_by_intent.values() for fact_id in ids})
        related_facts = {}
        for fact in self._db.get_memory_facts_by_ids(fact_ids) if fact_ids else []:
            observed_at = self._as_datetime(fact.get("dialogue_time_key"), as_of)
            if observed_at is not None and observed_at > as_of:
                continue
            related_facts[int(fact["id"])] = {
                "fact_id": int(fact["id"]), "summary": fact.get("summary") or "",
                "event_time_key": fact.get("event_time_key") or "",
                "dialogue_time_key": fact.get("dialogue_time_key") or "",
            }
        existing = self._model_db.get_user_predictions_by_intent_ids(
            intent_ids, statuses=["pending"], user_entity_id=owner_id,
        )
        evidence = self._model_db.get_user_prediction_evidence([row["id"] for row in existing])
        links = {}
        for link in evidence:
            links.setdefault(link["intent_id"], set()).add(link["prediction_id"])
        cases = []
        for row, intent in zip(rows, intents):
            related_events = [event for event in events if int(event["intent_id"]) == intent["id"]]
            latest = related_events[-1]
            # An older creation event can reach this task after a later update was persisted.
            changed_since_event = any(
                row.get(key) != value for key, value in latest["new_payload"].items()
            )
            cases.append({
                "event_type": "updated" if changed_since_event or any(
                    event["event_type"] == "updated" for event in related_events
                ) else "created",
                "current_intent": intent, "intent_events": related_events,
                "related_facts": [related_facts[fact_id]
                                  for fact_id in fact_ids_by_intent[intent["id"]]
                                  if fact_id in related_facts],
                "existing_predictions": [prediction for prediction in existing
                                         if prediction["id"] in links.get(intent["id"], set())],
            })

        states = self._model_db.get_user_states(
            state_types=["recurring_state"],
            statuses=["active"], limit=500,
        )
        active_states = []
        for state in states:
            valid_from = self._as_datetime(state.get("valid_from"), as_of)
            valid_to = self._as_datetime(state.get("valid_to"), as_of)
            if valid_from and valid_from >= horizon_end:
                continue
            if valid_to and valid_to <= as_of:
                continue
            active_states.append({
                "id": int(state["id"]), "state_text": state.get("state_text") or "",
                "state_domain": state.get("state_domain") or "general",
                "state_kind": state.get("state_kind") or "",
                "conditions": state.get("conditions") or {},
                "confidence": state.get("confidence"),
                "updated_at": state.get("updated_at") or "",
            })
        return {"intent_cases": cases, "recurring_states": active_states}

    def process_user_prediction_update_from_intent_events_task(
        self, *, intent_event_ids: Sequence[int],
        reference_timestamp: Optional[Any] = None,
        horizon_hours: float = 6.0,
    ) -> Dict[str, Any]:
        """Generate for created intents; retain or invalidate forecasts for updated intents."""
        started_at = time.monotonic()
        event_ids = sorted({int(value) for value in intent_event_ids if int(value) > 0})
        self._log_event("user_prediction_update_start", {
            "intent_event_ids": event_ids, "reference_timestamp": reference_timestamp,
            "horizon_hours": horizon_hours,
        })
        if not event_ids:
            self._log_event("finish", {
                "status": "empty", "reason": "no_intent_events",
                "total_elapsed_ms": round((time.monotonic() - started_at) * 1000, 2),
            })
            return {"status": "empty", "predictions": [], "prediction_assessments": [],
                    "retained_prediction_ids": [], "invalidated_prediction_ids": [],
                    "processed_intent_event_ids": []}
        local_now = self._now(None)
        now = self._as_datetime(reference_timestamp, local_now) or local_now
        if reference_timestamp is not None and self._as_datetime(reference_timestamp, local_now) is None:
            self._logger.warning(
                "Prediction reference time could not be parsed; using current time "
                "intent_event_ids=%s reference_timestamp=%r effective_reference_time=%s",
                event_ids, reference_timestamp, now.isoformat(),
            )
        end = now + timedelta(hours=max(0.1, float(horizon_hours)))
        owner_id = self._db.get_entity_id_by_name(self._owner_name)
        if owner_id is None:
            self._logger.error("Prediction world owner is missing intent_event_ids=%s owner_name=%s",
                               event_ids, self._owner_name)
            raise ValueError("Prediction world owner is missing")
        try:
            snapshot = self._build_user_prediction_prediction_inputs_using_intents(
                owner_id=owner_id, as_of=now, horizon_end=end, intent_event_ids=event_ids,
            )
        except Exception:
            self._logger.exception("Prediction input loading failed intent_event_ids=%s as_of=%s",
                                   event_ids, now.isoformat())
            raise
        states = snapshot["recurring_states"]
        self._log_event("inputs_loaded", {
            "intent_event_ids": event_ids, "owner_id": owner_id,
            "horizon_start": now.isoformat(), "horizon_end": end.isoformat(),
            "intent_cases": [{
                "event_type": case["event_type"], "current_intent": case["current_intent"],
                "intent_event_ids": [item["id"] for item in case["intent_events"]],
                "related_facts": case["related_facts"],
                "existing_prediction_ids": [item["id"] for item in case["existing_predictions"]],
            } for case in snapshot["intent_cases"]],
            "recurring_states": states,
        })
        updated_cases = [case for case in snapshot["intent_cases"] if case["event_type"] == "updated"]
        provided_prediction_ids = {item["id"] for case in updated_cases for item in case["existing_predictions"]}
        predictions, retained_prediction_ids, assessments = [], [], []
        status, error = "empty", ""
        try:
            created_cases = [case for case in snapshot["intent_cases"] if case["event_type"] == "created"]
            if created_cases:
                generated, assessed = self._generate_predictions_for_created_intents_with_llm(
                    created_cases, states=states, as_of=now, horizon_end=end,
                )
                predictions.extend(generated)
                assessments.extend(assessed)
            if updated_cases:
                generated, retained, assessed = self._update_predictions_for_updated_intents_with_llm(
                    updated_cases, states=states, as_of=now, horizon_end=end,
                )
                predictions.extend(generated)
                retained_prediction_ids.extend(retained)
                assessments.extend(assessed)
            status = "ok" if predictions or provided_prediction_ids or any(
                item["decision"] == "defer" for item in assessments
            ) else "empty"
        except Exception as exc:
            predictions, retained_prediction_ids, assessments = [], [], []
            status, error = "error", f"{type(exc).__name__}: {exc}"
            self._logger.exception(
                "Intent-event prediction evaluation failed intent_event_ids=%s as_of=%s "
                "pending_predictions_unchanged=True", event_ids, now.isoformat(),
            )
        try:
            with self._model_db.transaction():
                invalidated_prediction_ids = []
                if status != "error":
                    # Invalidate first so regenerating the same key cannot reuse a
                    # record that is about to be invalidated. All writes are atomic.
                    invalidated_prediction_ids = self._model_db.invalidate_user_predictions(
                        sorted(provided_prediction_ids - set(retained_prediction_ids)), user_entity_id=owner_id,
                    )
                saved = []
                for item in predictions:
                    prediction_id, save_action = self._model_db.save_user_prediction({
                        **item, "user_entity_id": owner_id,
                    })
                    saved.append({"id": prediction_id, "save_action": save_action, **item})
                if status != "error":
                    # Replace schedules for evaluated intents as one batch: a now/wait
                    # result must not erase a different behavior's deferred assessment.
                    clear_ids = [case["current_intent"]["id"] for case in snapshot["intent_cases"]]
                    self._model_db.delete_user_prediction_reassessment_schedules(
                        clear_ids, user_entity_id=owner_id,
                    )
                    for assessment in assessments:
                        if assessment["decision"] == "defer":
                            self._model_db.upsert_user_prediction_reassessment_schedule(
                                intent_id=assessment["intent_id"], user_entity_id=owner_id,
                                assessment_key=assessment["assessment_key"],
                                next_evaluate_at=assessment["next_evaluate_at"], reason=assessment["reason"],
                                evaluated_at=assessment["evaluated_at"], intent_basis_hash=assessment["intent_basis_hash"],
                                intent_event_ids=assessment["intent_event_ids"],
                            )
        except Exception:
            self._logger.exception(
                "Prediction result persistence failed intent_event_ids=%s as_of=%s transaction_rolled_back=True",
                event_ids, now.isoformat(),
            )
            raise
        # Log committed results only; a failed transaction must never look like a successful save.
        self._log_event("results_saved" if status != "error" else "results_not_saved", {
            "status": status, "intent_event_ids": event_ids,
            "predictions": [{
                **{key: item[key] for key in (
                    "id", "save_action", "behavior_text", "window_start", "window_end",
                    "conditions", "likelihood_band", "uncertainties", "rationale",
                )},
                "intent_id": item["basis"]["intent"]["id"],
                "state_ids": [state["id"] for state in item["basis"]["recurring_states"]],
            } for item in saved],
            "retained_prediction_ids": retained_prediction_ids,
            "invalidated_prediction_ids": invalidated_prediction_ids,
            "prediction_assessments": assessments,
            "reassessment_schedules_replaced": status != "error",
        })
        self._log_event("finish", {
            "status": status, "intent_event_ids": event_ids, "prediction_count": len(saved),
            "retained_prediction_ids": retained_prediction_ids,
            "invalidated_prediction_ids": invalidated_prediction_ids,
            "assessment_count": len(assessments), "error": error,
            "total_elapsed_ms": round((time.monotonic() - started_at) * 1000, 2),
        })
        return {"status": status, "predictions": saved,
                "retained_prediction_ids": retained_prediction_ids,
                "invalidated_prediction_ids": invalidated_prediction_ids,
                "prediction_assessments": assessments,
                "prediction_context": {
                    "as_of": now.isoformat(), "horizon_start": now.isoformat(),
                    "horizon_end": end.isoformat(), "predictor_version": self.MODEL_VERSION,
                    "intent_event_ids": event_ids,
                },
                "processed_intent_event_ids": event_ids if status in {"ok", "empty"} else [],
                "error": error}

    def _generate_predictions_for_created_intents_with_llm(
        self, cases: Sequence[Dict[str, Any]], *, states: Sequence[Dict[str, Any]],
        as_of: datetime, horizon_end: datetime,
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Assess individual behavior directions for newly created intents."""
        source = " ".join(case["current_intent"]["summary"] for case in cases)
        chinese = bool(re.search(r"[\u4e00-\u9fff]", source))
        template = (USER_PREDICTION_NEAR_TERM_PREPARATION_PROMPT_ZH if chinese
                    else USER_PREDICTION_NEAR_TERM_PREPARATION_PROMPT_EN)
        prompt = template.format(
            as_of=as_of.isoformat(), horizon_start=as_of.isoformat(), horizon_end=horizon_end.isoformat(),
            intent_cases=json.dumps(cases, ensure_ascii=False), states=json.dumps(states, ensure_ascii=False),
        )
        started_at = time.monotonic()
        self._log_event("llm_started", {
            "stage": "created", "intent_ids": [case["current_intent"]["id"] for case in cases],
            "prompt_language": "zh" if chinese else "en", "prompt_chars": len(prompt),
            "state_ids": [state["id"] for state in states],
            "horizon_start": as_of.isoformat(), "horizon_end": horizon_end.isoformat(),
        })
        self._log_event("llm_request", {"stage": "created", "prompt": prompt}, level=logging.DEBUG)
        raw = None
        try:
            raw = self._llm_client.call_llm(prompt)
            self._log_event("llm_response", {"stage": "created", "raw_response": raw}, level=logging.DEBUG)
            parsed = self._llm_client.parse_json_object_from_llm_text(raw or "")
            if not isinstance(parsed, dict) or set(parsed) != {"prediction_results"}:
                raise ValueError("Created-intent response must contain only prediction_results")
            predictions, assessments = self._normalize_prediction_results(
                parsed["prediction_results"], cases=cases, states=states, as_of=as_of, horizon_end=horizon_end,
            )
            self._log_event("llm_finished", {
                "stage": "created", "intent_ids": [case["current_intent"]["id"] for case in cases],
                "prediction_count": len(predictions), "assessments": assessments,
                "response_chars": len(raw or ""),
                "elapsed_ms": round((time.monotonic() - started_at) * 1000, 2),
            })
            return predictions, assessments
        except Exception as exc:
            self._log_prediction_response_failure(
                "created", raw=raw, error=exc, cases=cases, states=states,
                as_of=as_of, horizon_end=horizon_end,
            )
            raise

    def _log_prediction_response_failure(
        self, stage: str, *, raw: Any, error: Exception,
        cases: Sequence[Dict[str, Any]], states: Sequence[Dict[str, Any]],
        as_of: datetime, horizon_end: datetime,
    ) -> None:
        """Keep model output and validation scope together, only on failure."""
        payload = {
            "stage": stage, "error": f"{type(error).__name__}: {error}",
            "horizon_start": as_of.isoformat(), "horizon_end": horizon_end.isoformat(),
            "intent_cases": [{
                "current_intent": case["current_intent"],
                "intent_event_ids": [event["id"] for event in case["intent_events"]],
                "related_facts": case.get("related_facts", []),
                "existing_prediction_ids": [item["id"] for item in case["existing_predictions"]],
            } for case in cases],
            "allowed_state_ids": [state["id"] for state in states],
            "raw_response": raw,
        }
        self._logger.error(
            "User prediction response rejected diagnostics=%s",
            json.dumps(payload, ensure_ascii=False, default=str),
        )

    def _normalize_prediction_results(
        self, results: Any, *, cases: Sequence[Dict[str, Any]], states: Sequence[Dict[str, Any]],
        as_of: datetime, horizon_end: datetime,
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Normalize one assessment per behavior, allowing several per intent."""
        if not isinstance(results, list):
            raise ValueError("prediction_results must be an array")
        cases_by_id = {case["current_intent"]["id"]: case for case in cases}
        generated, assessments, seen_behaviors, seen_assessments = [], [], set(), set()
        counts = {}
        for index, result in enumerate(results):
            if not isinstance(result, dict) or set(result) != {
                "intent_id", "decision", "reason", "next_evaluate_at", "prediction",
            }:
                raise ValueError(f"Invalid prediction_results[{index}] fields")
            intent_id = result["intent_id"]
            if type(intent_id) is not int or intent_id not in cases_by_id:
                raise ValueError(f"Unknown intent_id in prediction_results[{index}]: {intent_id!r}")
            case = cases_by_id[intent_id]
            current = case["current_intent"]
            decision = result.get("decision")
            reason = result["reason"]
            if (decision not in {"predict_now", "defer", "wait_for_change"}
                    or not isinstance(reason, str) or not reason.strip()):
                raise ValueError(f"Invalid prediction_results[{index}] decision or reason")
            reason = reason.strip()
            if decision == "predict_now":
                prediction = result["prediction"]
                if not isinstance(prediction, dict) or set(prediction) != {
                    "intent_type", "state_ids", "behavior_text", "window_start", "window_end",
                    "conditions", "likelihood_band", "uncertainties", "rationale",
                }:
                    raise ValueError(f"prediction_results[{index}] requires one prediction object without intent_id")
                try:
                    new = self._normalize_case_predictions(
                        [{**prediction, "intent_id": intent_id}], cases=[case], states=states,
                        as_of=as_of, horizon_end=horizon_end,
                    )
                except ValueError as exc:
                    raise ValueError(f"prediction_results[{index}]: {exc}") from exc
                identity = (intent_id, self._normalize_behavior_text(new[0]["behavior_text"]))
                if identity in seen_behaviors:
                    raise ValueError("Duplicate predicted behavior")
                seen_behaviors.add(identity)
                counts[intent_id] = counts.get(intent_id, 0) + 1
                if counts[intent_id] > 5:
                    raise ValueError("Expected at most five new predictions per intent")
                generated.extend(new)
            elif result["prediction"] is not None:
                raise ValueError(f"prediction_results[{index}]: only predict_now may contain a prediction")
            next_evaluate_at = None
            if decision == "defer":
                value = result.get("next_evaluate_at")
                next_time = self._as_datetime(value, as_of)
                if (not isinstance(value, str) or not re.search(r"[T\s]\d{2}:\d{2}", value)
                        or next_time is None or next_time.tzinfo is None
                        or not re.search(r"(?:Z|[+-]\d{2}:\d{2})$", value)
                        or next_time <= as_of):
                    raise ValueError("Deferred assessment requires a future evaluation datetime")
                if current["status"] not in self.ACTIVE_INTENT_STATUSES.get(current["object_type"], set()):
                    raise ValueError("Terminal intents cannot be deferred for preparation prediction")
                next_evaluate_at = next_time.isoformat()
            elif result.get("next_evaluate_at") is not None:
                raise ValueError("Only defer may specify next_evaluate_at")
            assessment_key = self._hash({
                "intent_id": intent_id, "decision": decision, "reason": reason,
                "next_evaluate_at": next_evaluate_at,
                "prediction_key": new[0]["prediction_key"] if decision == "predict_now" else None,
            })
            if assessment_key in seen_assessments:
                raise ValueError("Duplicate behavior assessment")
            seen_assessments.add(assessment_key)
            assessments.append({
                "assessment_key": assessment_key,
                "intent_id": intent_id, "decision": decision, "reason": reason,
                "next_evaluate_at": next_evaluate_at, "evaluated_at": as_of.isoformat(),
                "intent_basis_hash": self._hash(current),
                "intent_event_ids": sorted({int(event["id"]) for event in case["intent_events"]}),
            })
        return generated, assessments

    def _update_predictions_for_updated_intents_with_llm(
        self, cases: Sequence[Dict[str, Any]], *, states: Sequence[Dict[str, Any]],
        as_of: datetime, horizon_end: datetime,
    ) -> tuple[List[Dict[str, Any]], List[int], List[Dict[str, Any]]]:
        """Reconcile retained IDs and per-behavior assessments in one LLM call."""
        source = " ".join(case["current_intent"]["summary"] for case in cases)
        chinese = bool(re.search(r"[\u4e00-\u9fff]", source))
        template = (USER_PREDICTION_INTENT_UPDATE_PROMPT_ZH if chinese
                    else USER_PREDICTION_INTENT_UPDATE_PROMPT_EN)
        prompt = template.format(
            as_of=as_of.isoformat(), horizon_start=as_of.isoformat(), horizon_end=horizon_end.isoformat(),
            intent_cases=json.dumps(cases, ensure_ascii=False), states=json.dumps(states, ensure_ascii=False),
        )
        started_at = time.monotonic()
        self._log_event("llm_started", {
            "stage": "updated", "intent_ids": [case["current_intent"]["id"] for case in cases],
            "existing_prediction_ids": [item["id"] for case in cases for item in case["existing_predictions"]],
            "prompt_language": "zh" if chinese else "en", "prompt_chars": len(prompt),
            "state_ids": [state["id"] for state in states],
            "horizon_start": as_of.isoformat(), "horizon_end": horizon_end.isoformat(),
        })
        self._log_event("llm_request", {"stage": "updated", "prompt": prompt}, level=logging.DEBUG)
        raw = None
        try:
            raw = self._llm_client.call_llm(prompt)
            self._log_event("llm_response", {"stage": "updated", "raw_response": raw}, level=logging.DEBUG)
            parsed = self._llm_client.parse_json_object_from_llm_text(raw or "")
            if not isinstance(parsed, dict) or set(parsed) != {"retained_prediction_ids", "prediction_results"}:
                raise ValueError("Updated-intent response requires retained_prediction_ids and prediction_results")
            predictions, retained, assessments = self._normalize_updated_intent_prediction_results(
                parsed["prediction_results"], retained_prediction_ids=parsed["retained_prediction_ids"],
                cases=cases, states=states,
                as_of=as_of, horizon_end=horizon_end,
            )
            self._log_event("llm_finished", {
                "stage": "updated", "intent_ids": [case["current_intent"]["id"] for case in cases],
                "prediction_count": len(predictions), "retained_prediction_ids": retained,
                "proposed_invalidated_prediction_ids": sorted({
                    item["id"] for case in cases for item in case["existing_predictions"]
                } - set(retained)),
                "assessments": assessments, "response_chars": len(raw or ""),
                "elapsed_ms": round((time.monotonic() - started_at) * 1000, 2),
            })
            return predictions, retained, assessments
        except Exception as exc:
            self._log_prediction_response_failure(
                "updated", raw=raw, error=exc, cases=cases, states=states,
                as_of=as_of, horizon_end=horizon_end,
            )
            raise

    def _normalize_updated_intent_prediction_results(
        self, results: Any, *, retained_prediction_ids: Any,
        cases: Sequence[Dict[str, Any]], states: Sequence[Dict[str, Any]],
        as_of: datetime, horizon_end: datetime,
    ) -> tuple[List[Dict[str, Any]], List[int], List[Dict[str, Any]]]:
        if (not isinstance(retained_prediction_ids, list)
                or any(type(value) is not int or value <= 0 for value in retained_prediction_ids)
                or len(set(retained_prediction_ids)) != len(retained_prediction_ids)):
            raise ValueError("retained_prediction_ids must contain unique positive integer IDs")
        retained = set(retained_prediction_ids)
        provided = {item["id"] for case in cases for item in case["existing_predictions"]}
        if retained - provided:
            raise ValueError(f"Unknown retained prediction IDs: {sorted(retained - provided)}")
        generated, assessments = self._normalize_prediction_results(
            results, cases=cases, states=states, as_of=as_of, horizon_end=horizon_end,
        )
        for case in cases:
            current = case["current_intent"]
            old = {item["id"]: item for item in case["existing_predictions"]}
            if (retained.intersection(old) and current["status"] not in
                    self.ACTIVE_INTENT_STATUSES.get(current["object_type"], set())):
                raise ValueError("Terminal intents cannot retain preparation predictions")
            retained_texts = {self._normalize_behavior_text(item["behavior_text"])
                              for key, item in old.items() if key in retained}
            if any(item["basis"]["intent"]["id"] == current["id"]
                   and self._normalize_behavior_text(item["behavior_text"]) in retained_texts
                   for item in generated):
                raise ValueError("New prediction duplicates a retained prediction")
        return generated, sorted(retained), assessments

    @staticmethod
    def _normalize_behavior_text(text: str) -> str:
        return re.sub(r"\s+", "", text).casefold()

    def _normalize_case_predictions(
        self, raw: Any, *, cases: Sequence[Dict[str, Any]], states: Sequence[Dict[str, Any]],
        as_of: datetime, horizon_end: datetime,
    ) -> List[Dict[str, Any]]:
        if not isinstance(raw, list) or len(raw) > 5:
            raise ValueError("Expected at most five predictions")
        intents = [case["current_intent"] for case in cases if case["current_intent"]["status"]
                   in self.ACTIVE_INTENT_STATUSES.get(case["current_intent"]["object_type"], set())]
        normalized = []
        seen = set()
        for index, candidate in enumerate(raw):
            try:
                accepted = self._validate_predictions(
                    [candidate], intents=intents, states=states, as_of=as_of, horizon_end=horizon_end,
                    intent_events=[event for case in cases for event in case["intent_events"]],
                )
            except ValueError as exc:
                raise ValueError(
                    f"Prediction candidate[{index}] for intent_ids="
                    f"{[case['current_intent']['id'] for case in cases]} rejected: {exc}"
                ) from exc
            item = accepted[0]
            identity = (item["basis"]["intent"]["id"], self._normalize_behavior_text(item["behavior_text"]))
            if identity in seen:
                raise ValueError("Duplicate predicted behavior")
            seen.add(identity)
            normalized.append(item)
        return normalized

    def _validate_predictions(
        self, raw: Sequence[Any], *, intents: Sequence[Dict[str, Any]],
        states: Sequence[Dict[str, Any]], as_of: datetime,
        horizon_end: datetime, intent_events: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        intent_by_key = {(item["object_type"], item["id"]): item for item in intents}
        state_by_id = {item["id"]: item for item in states}
        accepted = []
        for candidate in raw[:5]:
            if not isinstance(candidate, dict):
                raise ValueError("candidate_not_object")
            try:
                intent_type = str(candidate.get("intent_type") or "")
                intent_id = int(candidate.get("intent_id"))
                state_ids = sorted({int(value) for value in candidate.get("state_ids") or []})
            except (TypeError, ValueError) as exc:
                raise ValueError("invalid_intent_or_state_id_format") from exc
            intent = intent_by_key.get((intent_type, intent_id))
            if intent is None:
                raise ValueError(
                    f"unknown_or_inactive_intent_reference: intent_type={intent_type!r}, "
                    f"intent_id={intent_id}, allowed_intents={sorted(intent_by_key)}"
                )
            unknown_state_ids = [value for value in state_ids if value not in state_by_id]
            if unknown_state_ids:
                raise ValueError(
                    f"unknown_state_ids: {unknown_state_ids}, allowed_state_ids={sorted(state_by_id)}"
                )
            behavior = str(candidate.get("behavior_text") or "").strip()
            if not behavior:
                raise ValueError("empty_behavior_text")
            if re.sub(r"\s+", "", behavior).casefold() == re.sub(
                r"\s+", "", str(intent["summary"])
            ).casefold():
                raise ValueError("behavior_restates_intent_summary")
            band = str(candidate.get("likelihood_band") or "").lower().strip()
            beginning = self._as_datetime(candidate.get("window_start"), as_of)
            ending = self._as_datetime(candidate.get("window_end"), as_of)
            anchor_text = str(intent.get("anchor_at") or "")
            precise_anchor = bool(re.search(r"[T\s]\d{2}:\d{2}", anchor_text))
            if intent["object_type"] == "plan" and intent.get("time_precision") in {
                "day", "week", "unknown",
            }:
                precise_anchor = False
            # Keep coarse or unknown anchors as uncertainty, never midnight deadlines.
            intent_anchor = self._as_datetime(anchor_text, as_of) if precise_anchor else None
            reasons = []
            if band not in {"low", "medium", "high"}:
                reasons.append(f"invalid_likelihood_band: {band!r}")
            if beginning is None:
                reasons.append(f"invalid_window_start: {candidate.get('window_start')!r}")
            elif beginning < as_of:
                reasons.append(f"window_start_before_horizon: {beginning.isoformat()} < {as_of.isoformat()}")
            if ending is None:
                reasons.append(f"invalid_window_end: {candidate.get('window_end')!r}")
            else:
                if ending > horizon_end:
                    reasons.append(f"window_end_after_horizon: {ending.isoformat()} > {horizon_end.isoformat()}")
                if intent_anchor is not None and ending > intent_anchor:
                    reasons.append(f"window_end_after_intent_anchor: {ending.isoformat()} > {intent_anchor.isoformat()}")
            if beginning is not None and ending is not None and not beginning < ending:
                reasons.append(f"non_positive_window: {beginning.isoformat()} >= {ending.isoformat()}")
            if reasons:
                raise ValueError("; ".join(reasons))
            basis = {
                "intent": intent,
                "recurring_states": [state_by_id[value] for value in state_ids],
                "prediction_context": {
                    "as_of": as_of.isoformat(), "horizon_start": as_of.isoformat(),
                    "horizon_end": horizon_end.isoformat(), "predictor_version": self.MODEL_VERSION,
                    "intent_event_ids": sorted({int(event["id"]) for event in intent_events
                                                if int(event["intent_id"]) == intent_id}),
                },
            }
            key = self._hash({
                "kind": "near_term_preparation", "intent_type": intent_type,
                "intent_id": intent_id,
                "behavior": re.sub(r"\s+", " ", behavior).casefold(),
                "anchor": intent_anchor.isoformat() if intent_anchor is not None else anchor_text,
                "basis_hash": self._hash(basis),
                "window_start": beginning.isoformat(), "window_end": ending.isoformat(),
                "conditions": self._text_list(candidate.get("conditions")),
                "likelihood_band": band,
                "uncertainties": self._text_list(candidate.get("uncertainties")),
            })
            accepted.append({
                "prediction_key": key, "basis_hash": self._hash(basis),
                "behavior_text": behavior, "window_start": beginning.isoformat(),
                "window_end": ending.isoformat(),
                "conditions": self._text_list(candidate.get("conditions")),
                "likelihood_band": band,
                "uncertainties": self._text_list(candidate.get("uncertainties")),
                "rationale": str(candidate.get("rationale") or "").strip(),
                "basis": basis,
            })
        return accepted

    def get_active_predictions(
        self, *, as_of: Optional[datetime] = None,
        prediction_ids: Optional[Sequence[int]] = None,
    ) -> List[Dict[str, Any]]:
        """Read unexpired, unresolved hypotheses; no delivery decision here."""
        owner_id = self._db.get_entity_id_by_name(self._owner_name)
        if owner_id is None:
            self._log_event("active_predictions_read", {
                "reason": "world_owner_missing", "prediction_ids": prediction_ids,
            }, level=logging.DEBUG)
            return []
        reference = self._now(as_of)
        if prediction_ids is None:
            rows = self._model_db.get_active_user_predictions(
                user_entity_id=owner_id, as_of=reference.isoformat(),
            )
        else:
            rows = self._model_db.get_user_predictions_by_ids(prediction_ids)
        current = []
        for row in rows:
            ending = self._as_datetime(row.get("window_end"), reference)
            if (row.get("user_entity_id") != owner_id or row.get("status") != "pending"
                    or ending is None or ending <= reference):
                self._log_event("active_prediction_skipped", {
                    "prediction_id": row["id"], "reason": "owner_status_or_window_invalid",
                    "status": row.get("status"), "window_end": row.get("window_end"),
                    "as_of": reference.isoformat(),
                }, level=logging.DEBUG)
                continue
            basis = row.get("basis") or {}
            intent_basis = basis.get("intent") or {}
            try:
                intent_type = str(intent_basis["object_type"])
                intent_id = int(intent_basis["id"])
                state_basis = basis["recurring_states"]
                state_ids = [int(state["id"]) for state in state_basis]
            except (KeyError, TypeError, ValueError):
                self._log_event("active_prediction_skipped", {
                    "prediction_id": row["id"], "reason": "invalid_basis",
                }, level=logging.DEBUG)
                continue
            intent = self._db.get_intent_object(
                object_type=intent_type, object_id=intent_id,
            )
            if (not intent or intent.get("status") not in self.ACTIVE_INTENT_STATUSES.get(intent_type, set())
                    or int(intent.get("world_owner_entity_id") or 0) != owner_id
                    or int(intent.get("primary_entity_id") or 0) != owner_id):
                self._log_event("active_prediction_skipped", {
                    "prediction_id": row["id"], "intent_id": intent_id,
                    "reason": "source_intent_missing_inactive_or_foreign",
                    "intent_status": intent.get("status") if intent else None,
                }, level=logging.DEBUG)
                continue
            states = {int(item["id"]): item for item in self._model_db.get_user_states_by_ids(state_ids)}
            if any(
                state_id not in states
                or states[state_id].get("status") != "active"
                or states[state_id].get("updated_at") != source.get("updated_at")
                or (
                    self._as_datetime(states[state_id].get("valid_to"), reference) is not None
                    and self._as_datetime(states[state_id].get("valid_to"), reference) <= reference
                )
                for state_id, source in zip(state_ids, state_basis)
            ):
                self._log_event("active_prediction_skipped", {
                    "prediction_id": row["id"], "state_ids": state_ids,
                    "reason": "supporting_state_missing_changed_inactive_or_expired",
                }, level=logging.DEBUG)
                continue
            current.append(row)
        self._log_event("active_predictions_read", {
            "as_of": reference.isoformat(), "requested_prediction_ids": prediction_ids,
            "loaded_count": len(rows), "active_prediction_ids": [row["id"] for row in current],
        }, level=logging.DEBUG)
        return current

    def record_prediction_outcome(
        self, *, prediction_id: int, status: str,
        observed_at: Optional[datetime] = None,
        evidence_refs: Optional[Sequence[Dict[str, Any]]] = None,
        note: str = "",
    ) -> bool:
        """Attach an external validation result without creating memory evidence."""
        outcome = {
            "observed_at": self._now(observed_at).isoformat(),
            "evidence_refs": list(evidence_refs or []), "note": str(note),
        }
        changed = self._model_db.record_user_prediction_outcome(
            prediction_id=prediction_id, outcome_status=status, outcome=outcome,
        )
        self._log_event("prediction_outcome_recorded", {
            "prediction_id": prediction_id, "status": status, "changed": changed,
            "outcome": outcome,
        })
        return changed
