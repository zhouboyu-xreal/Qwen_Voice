"""Generate persistent service opportunities without dispatching them to an Agent."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence

if TYPE_CHECKING:
    from llm_client import LLMClient
    from task_scheduler import TaskScheduler
    from memory.memory_database import UserMemoryDB
    from .user_model_database import UserModelDB

from prompts_en import USER_PROACTIVE_POLICY_EVALUATION_PROMPT_EN
from prompts_zh import USER_PROACTIVE_POLICY_EVALUATION_PROMPT_ZH


class UserProactiveService:
    """LLM policy owns service content/timing; execution belongs to a downstream Agent."""

    TASK_KIND = "user_proactive_policy_evaluation"
    POLICY_VERSION = "llm_services_v2"
    SERVICE_TYPES = {"notify", "suggest", "clarify", "offer_assistance", "prepare", "delegate"}

    def __init__(
        self, *, memory_db: UserMemoryDB, model_db: UserModelDB,
        prediction_reader: Callable[..., List[Dict[str, Any]]], task_scheduler: TaskScheduler,
        llm_client: LLMClient,
        world_owner_entity_name: str = "用户本人",
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._model_db = model_db
        self._memory_db = memory_db
        self._prediction_reader = prediction_reader
        self._owner_name = world_owner_entity_name
        self._task_scheduler = task_scheduler
        self._llm_client = llm_client
        self._logger = logger or logging.getLogger(__name__)
        task_scheduler.register_handler(self.TASK_KIND, self.generate_proactive_services_from_predictions)

    def _log_event(self, event: str, payload: Dict[str, Any], *, level: int = logging.INFO) -> None:
        if self._logger.isEnabledFor(level):
            self._logger.log(level, "\n%s", json.dumps(
                {"scope": self.TASK_KIND, "event": event, "payload": payload},
                ensure_ascii=False, indent=2, default=str,
            ))

    @staticmethod
    def _as_datetime(value: Any, reference: datetime) -> Optional[datetime]:
        try:
            result = value if isinstance(value, datetime) else datetime.fromisoformat(
                str(value or "").replace("Z", "+00:00")
            )
        except ValueError:
            return None
        if result.tzinfo is None:
            result = result.replace(tzinfo=reference.tzinfo)
        return result.astimezone(reference.tzinfo)

    @classmethod
    def _active_user_states(
        cls, *, as_of: datetime, model_db: UserModelDB,
    ) -> List[Dict[str, Any]]:
        states = model_db.get_user_states(
            statuses=["active"], limit=500,
        )
        active = []
        for state in states:
            valid_from = cls._as_datetime(state.get("valid_from"), as_of)
            valid_to = cls._as_datetime(state.get("valid_to"), as_of)
            if valid_from and valid_from > as_of:
                continue
            if valid_to and valid_to <= as_of:
                continue
            active.append(state)
        return active

    def _build_service_evaluation_input(
        self, prediction: Dict[str, Any], *, states: Sequence[Dict[str, Any]], as_of: datetime,
    ) -> Dict[str, Any]:
        basis = prediction.get("basis") or {}
        source = basis.get("intent") or {}
        intent = self._memory_db.get_intent_object(
            object_type=source["object_type"], object_id=int(source["id"]),
        )
        key = (intent["object_type"], int(intent["id"]))
        fact_ids = self._memory_db.get_intent_evidence_fact_ids([key], limit=20).get(key, [])
        facts, excluded_future_fact_ids = [], []
        for fact in self._memory_db.get_memory_facts_by_ids(fact_ids):
            observed = self._as_datetime(fact.get("dialogue_time_key"), as_of)
            if observed and observed > as_of:
                excluded_future_fact_ids.append(fact["id"])
                continue
            facts.append({name: fact.get(name) for name in (
                "id", "summary", "fact_type", "event_time_key", "dialogue_time_key",
            )})
        supporting_ids = {int(item["id"]) for item in basis.get("recurring_states") or []}
        relevant = [state for state in states if (
            int(state["id"]) in supporting_ids
            or state.get("state_domain") in {intent.get("primary_domain"), "general"}
        )]
        relevant.sort(key=lambda state: int(state["id"]) not in supporting_ids)
        case = {
            "prediction": prediction, "current_intent": intent, "related_facts": facts,
            "relevant_user_states": [{name: state.get(name) for name in (
                "id", "state_domain", "state_type", "state_kind", "state_topic", "state_text",
                "conditions", "confidence", "valid_from", "valid_to", "last_observed_at",
            )} for state in relevant[:40]],
            "existing_services": self._model_db.get_user_proactive_services(
                prediction_id=int(prediction["id"]), user_entity_id=int(prediction["user_entity_id"]),
                limit=1000,
            ),
        }
        self._log_event("inputs_loaded", {
            "prediction": {key: prediction.get(key) for key in (
                "id", "behavior_text", "window_start", "window_end", "conditions",
                "likelihood_band", "uncertainties", "rationale",
            )},
            "current_intent": intent, "related_facts": facts,
            "future_fact_ids_excluded": excluded_future_fact_ids,
            "relevant_user_states": case["relevant_user_states"],
            "existing_services": [{key: item.get(key) for key in (
                "id", "service_type", "objective", "status", "reason",
                "next_evaluate_at", "recheck_condition", "expires_at",
            )} for item in case["existing_services"]],
            "as_of": as_of.isoformat(),
        })
        return case

    def _evaluate_with_llm(self, case: Dict[str, Any], *, as_of: datetime,
                           runtime_context: Dict[str, Any], agent_capabilities: Dict[str, Any]) -> Any:
        source_text = json.dumps(case, ensure_ascii=False)
        chinese = bool(re.search(r"[\u4e00-\u9fff]", source_text))
        prompt_template = (
            USER_PROACTIVE_POLICY_EVALUATION_PROMPT_ZH
            if chinese
            else USER_PROACTIVE_POLICY_EVALUATION_PROMPT_EN
        )
        prompt = prompt_template.format(
            as_of=as_of.isoformat(),
            evaluation_case=source_text,
            runtime_context=json.dumps(runtime_context, ensure_ascii=False),
            agent_capabilities=json.dumps(agent_capabilities, ensure_ascii=False),
        )
        started_at = time.monotonic()
        prediction_id = case["prediction"]["id"]
        self._log_event("llm_started", {
            "prediction_id": prediction_id, "as_of": as_of.isoformat(),
            "prompt_language": "zh" if chinese else "en", "prompt_chars": len(prompt),
            "runtime_context": runtime_context, "agent_capabilities": agent_capabilities,
        })
        raw = None
        try:
            raw = self._llm_client.call_llm(prompt)
            parsed = self._llm_client.parse_json_object_from_llm_text(raw or "")
            if not isinstance(parsed, dict):
                self._logger.error(
                    "proactive service invalid JSON prediction_id=%s response=%r", prediction_id, raw,
                )
            return parsed
        except Exception:
            self._logger.exception(
                "proactive service LLM call or parse failed prediction_id=%s elapsed_ms=%.2f response=%r",
                prediction_id, (time.monotonic() - started_at) * 1000, raw,
            )
            raise

    def _normalize_service_results(
        self, response: Any, *, case: Dict[str, Any], as_of: datetime,
        runtime_context: Dict[str, Any], agent_capabilities: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        if not isinstance(response, dict) or not isinstance(response.get("service_results"), list):
            raise ValueError("Proactive response must contain a service_results list")
        existing = {int(item["id"]): item for item in case["existing_services"]}
        open_ids = {key for key, item in existing.items() if item["status"] in {"ready", "waiting"}}
        seen_ids, seen_keys, results = set(), set(), []

        def future_time(value: Any) -> datetime:
            if not isinstance(value, str):
                raise ValueError("Service time must be a timezone-aware ISO timestamp")
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None or parsed <= as_of:
                raise ValueError(f"Service time must be after evaluation time: {value!r}")
            return parsed.astimezone(as_of.tzinfo)

        for raw in response["service_results"]:
            fields = {"service_id", "prediction_id", "decision", "reason", "service",
                      "next_evaluate_at", "recheck_condition", "expires_at"}
            if not isinstance(raw, dict) or set(raw) != fields:
                raise ValueError("Invalid service result fields")
            item = dict(raw)
            if (type(item["prediction_id"]) is not int
                    or item["prediction_id"] != case["prediction"]["id"]):
                raise ValueError("Service references an unrelated prediction")
            decision = item["decision"]
            if not isinstance(decision, str) or decision not in {"recommend_service", "defer", "suppress"}:
                raise ValueError("Unsupported proactive decision")
            if not isinstance(item["reason"], str) or not item["reason"].strip():
                raise ValueError("Service decision requires a reason")
            service_id = item["service_id"]
            if service_id is not None and (type(service_id) is not int or service_id not in open_ids):
                raise ValueError("Service update must reference an open service of this prediction")
            if decision == "suppress":
                if (item["service"] is not None or item["expires_at"] is not None
                        or item["next_evaluate_at"] is not None or item["recheck_condition"] != ""):
                    raise ValueError("Suppression must not include a new service or schedule")
            else:
                service = item["service"]
                if (not isinstance(service, dict) or set(service) != {
                    "service_type", "objective", "agent_guidance", "constraints",
                }):
                    raise ValueError("Service requires type, objective, Agent guidance and constraints")
                kind = service["service_type"]
                if not isinstance(kind, str) or kind not in self.SERVICE_TYPES:
                    raise ValueError("Unsupported service type")
                for name in ("objective", "agent_guidance"):
                    if not isinstance(service[name], str) or not service[name].strip():
                        raise ValueError(f"Missing service {name}")
                if not isinstance(service["constraints"], list) or any(
                    not isinstance(value, str) or not value.strip() for value in service["constraints"]
                ):
                    raise ValueError("Service constraints must be a list of nonempty strings")
                key_text = kind + ":" + re.sub(r"\s+", "", service["objective"]).casefold()
                item["service_key"] = hashlib.sha256(key_text.encode()).hexdigest()
                if item["service_key"] in seen_keys:
                    raise ValueError("Duplicate service objective in one response")
                seen_keys.add(item["service_key"])
                if service_id is None:
                    match = next((old for old in existing.values()
                                  if old["service_key"] == item["service_key"]), None)
                    if match:
                        service_id = int(match["id"])
                        if service_id not in open_ids:
                            raise ValueError("Closed services cannot be silently reactivated")
                        item["service_id"] = service_id
                expiry = future_time(item["expires_at"])
                item["expires_at"] = expiry.isoformat()
                if not isinstance(item["recheck_condition"], str):
                    raise ValueError("Recheck condition must be text")
                if decision == "recommend_service":
                    if item["next_evaluate_at"] is not None or item["recheck_condition"]:
                        raise ValueError("Ready services cannot have a waiting schedule")
                    if kind not in agent_capabilities.get("supported_service_types", []):
                        raise ValueError("Ready service requires explicitly available Agent capability")
                    if kind in {"prepare", "delegate"}:
                        if kind not in agent_capabilities.get("authorized_service_types", []):
                            raise ValueError("Background service requires explicit authorization")
                    elif runtime_context.get("can_present") is not True:
                        raise ValueError("Ready conversational service requires known presentation availability")
                else:
                    next_at = item["next_evaluate_at"]
                    if next_at is not None:
                        next_time = future_time(next_at)
                        if next_time >= expiry:
                            raise ValueError("Reevaluation must precede service expiry")
                        item["next_evaluate_at"] = next_time.isoformat()
                    elif not item["recheck_condition"].strip():
                        raise ValueError("Waiting service requires a reevaluation time or condition")
            if service_id is not None:
                if service_id in seen_ids:
                    raise ValueError("Existing service appears more than once")
                seen_ids.add(service_id)
            results.append(item)
        if seen_ids != open_ids:
            raise ValueError("Every open existing service must be explicitly updated or suppressed")
        return results

    def _save_service_results(
        self, results: Sequence[Dict[str, Any]], *, case: Dict[str, Any], as_of: datetime,
        source_prediction_task_id: str,
    ) -> List[Dict[str, Any]]:
        existing = {int(item["id"]): item for item in case["existing_services"]}
        saved = []
        with self._model_db.transaction():
            for item in results:
                if item["decision"] == "suppress" and item["service_id"] is None:
                    saved.append({**item, "save_action": "not_saved"})
                    continue
                if item["decision"] == "suppress":
                    payload = {**existing[item["service_id"]], "service_id": item["service_id"],
                               "status": "cancelled", "reason": item["reason"],
                               "next_evaluate_at": None, "recheck_condition": ""}
                else:
                    payload = {**item["service"], "service_id": item["service_id"],
                               "service_key": item["service_key"], "reason": item["reason"],
                               "status": "ready" if item["decision"] == "recommend_service" else "waiting",
                               **{name: item[name] for name in (
                                   "next_evaluate_at", "recheck_condition", "expires_at",
                               )}}
                payload.update(
                    prediction_id=case["prediction"]["id"],
                    user_entity_id=case["prediction"]["user_entity_id"],
                    policy_version=self.POLICY_VERSION, evaluated_at=as_of.isoformat(),
                    source_prediction_task_id=source_prediction_task_id,
                )
                service_id, action = self._model_db.save_user_proactive_service(payload)
                saved.append({**item, "service_id": service_id, "status": payload["status"],
                              "save_action": action})
        self._log_event("services_saved", {
            "prediction_id": case["prediction"]["id"],
            "source_prediction_task_id": source_prediction_task_id,
            "service_results": saved,
        })
        return saved

    def generate_proactive_services_from_predictions(
        self, *, prediction_ids: Sequence[int], source_prediction_task_id: str = "",
        runtime_context: Optional[Dict[str, Any]] = None,
        agent_capabilities: Optional[Dict[str, Any]] = None,
        reference_timestamp: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Persist service opportunities; ready does not mean dispatched or delivered."""
        started_at = time.monotonic()
        self._log_event("proactive_services_generation_start", {
            "prediction_ids": prediction_ids, "source_prediction_task_id": source_prediction_task_id,
            "reference_timestamp": reference_timestamp,
            "runtime_context": runtime_context, "agent_capabilities": agent_capabilities,
        })
        now = datetime.now().astimezone()
        if reference_timestamp is not None:
            reference = self._as_datetime(reference_timestamp, now)
            if reference is None:
                raise ValueError("Invalid proactive policy reference_timestamp")
            now = reference
        if runtime_context is not None and not isinstance(runtime_context, dict):
            raise ValueError("runtime_context must be an object")
        context = dict(runtime_context or {})
        if agent_capabilities is not None and not isinstance(agent_capabilities, dict):
            raise ValueError("agent_capabilities must be an object")
        capabilities = dict(agent_capabilities or {})
        for name in ("supported_service_types", "authorized_service_types"):
            values = capabilities.get(name, [])
            if not isinstance(values, list) or any(
                not isinstance(value, str) or value not in self.SERVICE_TYPES for value in values
            ):
                raise ValueError(f"Invalid capability list: {name}")
            capabilities[name] = sorted(set(values))
        if any(type(value) is not int or value <= 0 for value in prediction_ids):
            raise ValueError("prediction_ids must contain positive integer IDs")
        ids = sorted(set(prediction_ids))
        rows = {int(row["id"]): row for row in self._model_db.get_user_predictions_by_ids(ids)}
        active = {int(row["id"]): row for row in self._prediction_reader(prediction_ids=ids, as_of=now)}
        states = self._active_user_states(as_of=now, model_db=self._model_db) if active else []
        owner_id = self._memory_db.get_entity_id_by_name(self._owner_name)
        self._log_event("predictions_loaded", {
            "source_prediction_task_id": source_prediction_task_id,
            "requested_prediction_ids": ids, "loaded_prediction_ids": sorted(rows),
            "active_prediction_ids": sorted(active), "owner_id": owner_id,
            "active_state_count": len(states), "as_of": now.isoformat(),
        })
        results, failures, skipped, closed_ids, processed = [], [], [], [], []
        for prediction_id in ids:
            prediction = rows.get(prediction_id)
            if not prediction or prediction.get("user_entity_id") != owner_id:
                skipped.append({"prediction_id": prediction_id, "reason": "missing_or_foreign_prediction"})
                self._log_event("prediction_skipped", skipped[-1])
                continue
            if prediction_id not in active:
                ending = self._as_datetime(prediction.get("window_end"), now)
                status = "expired" if prediction["status"] == "pending" and ending and ending <= now else "cancelled"
                closed = self._model_db.close_user_proactive_services(
                    prediction_id=prediction_id, user_entity_id=owner_id, status=status,
                    reason="Source prediction is no longer effective", as_of=now.isoformat(),
                )
                closed_ids.extend(closed)
                self._log_event("prediction_skipped", {
                    "prediction_id": prediction_id, "reason": "inactive_prediction",
                    "prediction_status": prediction["status"], "window_end": prediction.get("window_end"),
                    "closed_service_ids": closed, "closed_service_status": status,
                    "as_of": now.isoformat(),
                })
                skipped.append({"prediction_id": prediction_id, "reason": "inactive_prediction"})
                continue
            expired = self._model_db.close_user_proactive_services(
                prediction_id=prediction_id, user_entity_id=owner_id, status="expired",
                reason="Service validity period ended", as_of=now.isoformat(), expired_only=True,
            )
            closed_ids.extend(expired)
            if expired:
                self._log_event("services_expired", {
                    "prediction_id": prediction_id, "expired_service_ids": expired, "as_of": now.isoformat(),
                })
            response = None
            evaluation_started_at = time.monotonic()
            stage = "build_inputs"
            try:
                case = self._build_service_evaluation_input(prediction, states=states, as_of=now)
                stage = "llm_evaluation"
                response = self._evaluate_with_llm(
                    case, as_of=now, runtime_context=context, agent_capabilities=capabilities,
                )
                stage = "normalize_response"
                normalized = self._normalize_service_results(
                    response, case=case, as_of=now, runtime_context=context, agent_capabilities=capabilities,
                )
                stage = "save_services"
                results.extend(self._save_service_results(
                    normalized, case=case, as_of=now, source_prediction_task_id=source_prediction_task_id,
                ))
                processed.append(prediction_id)
                self._log_event("prediction_finished", {
                    "prediction_id": prediction_id, "status": "ok", "result_count": len(normalized),
                    "elapsed_ms": round((time.monotonic() - evaluation_started_at) * 1000, 2),
                })
            except Exception as exc:
                self._logger.exception(
                    "proactive service evaluation failed prediction_id=%s source_prediction_task_id=%s "
                    "stage=%s elapsed_ms=%.2f existing_unexpired_services_unchanged=True response=%s",
                    prediction_id, source_prediction_task_id, stage,
                    (time.monotonic() - evaluation_started_at) * 1000, response,
                )
                failures.append({"prediction_id": prediction_id, "error": f"{type(exc).__name__}: {exc}"})
        closed_ids.extend(item["service_id"] for item in results if item.get("status") == "cancelled")
        report = {
            "status": "error" if failures else "ok" if processed or closed_ids else "empty",
            "evaluated_count": len(processed), "failed_count": len(failures),
            "processed_prediction_ids": processed,
            "new_service_ids": [item["service_id"] for item in results if item["save_action"] == "created"],
            "updated_service_ids": [item["service_id"] for item in results if item["save_action"] == "updated"],
            "ready_service_ids": [item["service_id"] for item in results if item.get("status") == "ready"],
            "waiting_service_ids": [item["service_id"] for item in results if item.get("status") == "waiting"],
            "closed_service_ids": sorted(set(closed_ids)),
            "skipped_predictions": skipped, "failed_predictions": failures, "service_results": results,
        }
        self._log_event("finish", {
            **{key: value for key, value in report.items() if key != "service_results"},
            "source_prediction_task_id": source_prediction_task_id, "prediction_ids": ids,
            "result_count": len(results),
            "total_elapsed_ms": round((time.monotonic() - started_at) * 1000, 2),
        })
        return report

    def get_services(
        self, *, user_entity_id: Optional[int] = None, statuses: Optional[Sequence[str]] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Read saved opportunities, not a dispatch/interruptibility decision."""
        services = self._model_db.get_user_proactive_services(
            user_entity_id=user_entity_id, statuses=statuses, limit=limit,
        )
        self._log_event("services_read", {
            "user_entity_id": user_entity_id, "statuses": statuses, "limit": limit,
            "service_ids": [item["id"] for item in services],
        }, level=logging.DEBUG)
        return services
