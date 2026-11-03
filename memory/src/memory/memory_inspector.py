"""Bounded, read-only views of committed PWM records for user-facing clients.

This adapter never initializes schemas or starts a PersonWorldModel. Missing
databases are an empty/unavailable view, not a reason to create runtime state.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator


INTENT_TYPES = {"goal", "plan", "work_item"}
STATE_TYPES = {"enduring_state", "recurring_state", "situational_state"}
ACTIVITY_ROLES = {"actor", "participant", "contextual"}
CURRENT_INTENT_STATUSES = {"active", "planned", "rescheduled", "open", "in_progress", "blocked"}
CURRENT_STATE_STATUSES = {"active", "weakened"}
STATUSES = CURRENT_INTENT_STATUSES | CURRENT_STATE_STATUSES | {"completed", "cancelled", "occurred", "expired"}
DETAIL_PAGE_SIZE = 20
SOURCE_PAGE_SIZE = 6


def _json(value: Any, fallback: Any) -> Any:
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
        return parsed if isinstance(parsed, type(fallback)) else fallback
    except (ValueError, TypeError):
        return fallback


def _positive(value: Any, name: str, maximum: int = 1_000_000) -> int:
    if isinstance(value, bool) or not str(value).isdigit() or not 1 <= int(value) <= maximum:
        raise ValueError(f"invalid {name}")
    return int(value)


@contextmanager
def _reader(path: Path) -> Iterator[sqlite3.Connection | None]:
    if not path.is_file():
        yield None
        return
    connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True, timeout=1.0)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA query_only=ON")
        connection.execute("BEGIN")
        yield connection
    finally:
        connection.rollback()
        connection.close()


def _table(connection: sqlite3.Connection | None, name: str) -> bool:
    return connection is not None and connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,),
    ).fetchone() is not None


def _bounded(value: Any, limit: int = 4000) -> str:
    return str(value or "").replace("\0", "")[:limit]


def _reference_ids(value: Any) -> list[int]:
    return list(dict.fromkeys(item for item in _json(value, [])
                              if type(item) is int and 0 < item <= 2**53 - 1))


def _instant(value: Any) -> str:
    if value in (None, ""):
        return ""
    if not isinstance(value, str) or len(value) > 40:
        raise ValueError("invalid activity time")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError("invalid activity time") from None
    if parsed.tzinfo is None:
        raise ValueError("activity time requires timezone")
    return parsed.isoformat()


class MemoryInspector:
    def __init__(self, memory_path: Path, model_path: Path, *, owner_name: str = "用户本人") -> None:
        self.memory_path = memory_path
        self.model_path = model_path
        self.owner_name = owner_name

    def inspect(self, params: dict[str, Any]) -> dict[str, Any]:
        section = params.get("section")
        if section == "activities":
            return self._activities(params)
        if section not in {"intents", "states"}:
            raise ValueError("section must be intents or states")
        types = INTENT_TYPES if section == "intents" else STATE_TYPES
        object_type = str(params.get("type") or "")
        if object_type and object_type not in types:
            raise ValueError("invalid type")
        scope = str(params.get("scope") or "current")
        status = str(params.get("status") or "")
        if scope not in {"current", "history", "all"} or status and status not in STATUSES:
            raise ValueError("invalid scope or status")
        page = _positive(params.get("page", 1), "page")
        size = _positive(params.get("pageSize", 20), "pageSize", 100)
        item_id = _positive(params["id"], "id", 2**53 - 1) if params.get("id") is not None else None
        evidence_page = _positive(params.get("evidencePage", 1), "evidencePage")
        history_page = _positive(params.get("historyPage", 1), "historyPage")
        table = "memory_user_intent_items" if section == "intents" else "user_states"
        path = self.memory_path if section == "intents" else self.model_path
        result: dict[str, Any] = {
            "supported": True, "available": False, "section": section,
            "fetchedAt": datetime.now().astimezone().isoformat(),
            "page": page, "pageSize": size, "total": 0, "items": [],
        }
        with _reader(path) as connection:
            if not _table(connection, table):
                result["reason"] = "not_created" if connection is None else "schema_unavailable"
                if item_id is not None:
                    result["item"] = None
                return result
            result["available"] = True
            clauses, values = [], []
            if section == "intents":
                if not _table(connection, "memory_entity_nodes"):
                    return {**result, "available": False, "reason": "schema_unavailable"}
                clauses.append("world_owner_entity_id IN (SELECT id FROM memory_entity_nodes WHERE name=?)")
                values.append(self.owner_name)
            if item_id is not None:
                clauses.append("id=?")
                values.append(item_id)
            else:
                if object_type:
                    clauses.append(f"{'object_type' if section == 'intents' else 'state_type'}=?")
                    values.append(object_type)
                if params.get("domain"):
                    clauses.append(f"{'primary_domain' if section == 'intents' else 'state_domain'}=?")
                    values.append(_bounded(params["domain"], 80))
                if status:
                    clauses.append("status=?")
                    values.append(status)
                if scope != "all":
                    current = sorted(CURRENT_INTENT_STATUSES if section == "intents" else CURRENT_STATE_STATUSES)
                    clauses.append(f"status {'NOT IN' if scope == 'history' else 'IN'} ({','.join('?' for _ in current)})")
                    values.extend(current)
            where = " WHERE " + " AND ".join(clauses) if clauses else ""
            if item_id is None:
                result["total"] = connection.execute(f"SELECT COUNT(*) FROM {table}{where}", values).fetchone()[0]
                rows = connection.execute(
                    f"SELECT * FROM {table}{where} ORDER BY updated_at DESC,id DESC LIMIT ? OFFSET ?",
                    (*values, size, (page - 1) * size),
                ).fetchall()
                result["items"] = [self._record(dict(row), section) for row in rows]
                return result
            row = connection.execute(f"SELECT * FROM {table}{where}", values).fetchone()
            result["item"] = self._record(dict(row), section) if row else None
            if row is None:
                return result
            result.pop("items")
            if section == "intents":
                self._intent_detail(connection, result, dict(row), evidence_page, history_page)
            else:
                self._state_detail(connection, result, item_id, evidence_page, history_page)
            return result

    def _activities(self, params: dict[str, Any]) -> dict[str, Any]:
        role = params.get("role", "")
        if not isinstance(role, str) or role and role not in ACTIVITY_ROLES:
            raise ValueError("invalid activity role")
        after, before = _instant(params.get("from")), _instant(params.get("to"))
        if after and before and datetime.fromisoformat(after) >= datetime.fromisoformat(before):
            raise ValueError("invalid activity time range")
        page = _positive(params.get("page", 1), "page")
        size = _positive(params.get("pageSize", 20), "pageSize", 100)
        item_id = _positive(params["id"], "id", 2**53 - 1) if params.get("id") is not None else None
        evidence_page = _positive(params.get("evidencePage", 1), "evidencePage")
        source_page = _positive(params.get("sourcePage", 1), "sourcePage")
        result = {"supported": True, "available": False, "section": "activities",
                  "fetchedAt": datetime.now().astimezone().isoformat(),
                  "page": page, "pageSize": size, "total": 0, "items": []}
        with _reader(self.memory_path) as connection:
            if not _table(connection, "memory_user_activity_items"):
                result["reason"] = "not_created" if connection is None else "schema_unavailable"
                if item_id is not None:
                    result["item"] = None
                return result
            result["available"] = True
            if item_id is not None:
                row = connection.execute("SELECT * FROM memory_user_activity_items WHERE id=?", (item_id,)).fetchone()
                result["item"] = self._activity_record(dict(row)) if row else None
                result.pop("items")
                if row:
                    self._activity_detail(connection, result, dict(row), evidence_page, source_page)
                return result
            clauses, values = [], []
            start = "COALESCE(julianday(started_at),julianday(ended_at))"
            end = "COALESCE(julianday(ended_at),julianday(started_at))"
            if role:
                clauses.append("user_role=?")
                values.append(role)
            # Overlap with [from, to); compare instants, never ISO text across timezones.
            for value, expression, operator in ((after, end, ">="), (before, start, "<")):
                if value:
                    clauses.append(f"{expression} {operator} julianday(?)")
                    values.append(value)
            where = " WHERE " + " AND ".join(clauses) if clauses else ""
            result["total"] = connection.execute(
                f"SELECT COUNT(*) FROM memory_user_activity_items{where}", values,
            ).fetchone()[0]
            rows = connection.execute(
                f"SELECT * FROM memory_user_activity_items{where} ORDER BY {start} DESC,id DESC LIMIT ? OFFSET ?",
                (*values, size, (page - 1) * size),
            ).fetchall()
            result["items"] = [self._activity_record(dict(row)) for row in rows]
            return result

    @staticmethod
    def _activity_record(row: dict[str, Any]) -> dict[str, Any]:
        item = {key: row.get(key) for key in (
            "id", "user_role", "confidence", "started_at", "ended_at", "created_at", "updated_at",
        )}
        item["activity_text"] = _bounded(row.get("activity_text"))
        item["time_scope"] = "source_window" if _json(row.get("metadata"), {}).get("time_scope") == "source_window" else ""
        item["evidence_fact_count"] = len(_reference_ids(row.get("evidence_fact_ids")))
        item["source_segment_count"] = len(_reference_ids(row.get("source_segment_ids")))
        return item

    def _activity_detail(self, connection, result, row, evidence_page, source_page):
        fact_ids = _reference_ids(row.get("evidence_fact_ids"))
        selected = fact_ids[(evidence_page - 1) * DETAIL_PAGE_SIZE:evidence_page * DETAIL_PAGE_SIZE]
        evidence = [{"evidence_type": "fact", "evidence_id": key, "role": "support"} for key in selected]
        self._resolve_evidence(connection, evidence, include_sources=False)
        result["evidence"] = {"items": evidence, "total": len(fact_ids),
                              "page": evidence_page, "pageSize": DETAIL_PAGE_SIZE}
        source_ids = _reference_ids(row.get("source_segment_ids"))
        selected = source_ids[(source_page - 1) * SOURCE_PAGE_SIZE:source_page * SOURCE_PAGE_SIZE]
        by_id = {}
        if selected and _table(connection, "memory_source_segments"):
            placeholders = ",".join("?" for _ in selected)
            by_id = {item["id"]: item for item in connection.execute(
                f"SELECT id,text,started_at,ended_at FROM memory_source_segments WHERE id IN ({placeholders})", selected,
            )}
        sources = []
        for key in selected:
            source = by_id.get(key)
            sources.append({"id": key, "resolved": source is not None,
                            "text": _bounded(source["text"], 2000) if source else "",
                            "started_at": source["started_at"] if source else None,
                            "ended_at": source["ended_at"] if source else None})
        result["sources"] = {"items": sources, "total": len(source_ids),
                             "page": source_page, "pageSize": SOURCE_PAGE_SIZE}

    @staticmethod
    def _record(row: dict[str, Any], section: str) -> dict[str, Any]:
        keys = ("id", "status", "confidence", "created_at", "updated_at")
        extra = ("object_type", "summary", "primary_domain", "primary_entity_id", "time_start", "time_end") if section == "intents" else (
            "state_domain", "state_type", "state_kind", "state_topic", "state_text", "valid_from", "valid_to", "last_observed_at",
        )
        item = {key: row.get(key) for key in (*keys, *extra)}
        for key in ("details",) if section == "intents" else ("conditions",):
            item[key] = _json(row.get(key), {})
        return item

    @staticmethod
    def _page(connection: sqlite3.Connection, table: str, where: str, values: tuple, page: int) -> dict:
        if not _table(connection, table):
            return {"items": [], "total": 0, "page": page, "pageSize": DETAIL_PAGE_SIZE}
        total = connection.execute(f"SELECT COUNT(*) FROM {table} WHERE {where}", values).fetchone()[0]
        # Evidence tables have no integer id; rowid is stable within this snapshot.
        rows = connection.execute(
            f"SELECT * FROM {table} WHERE {where} ORDER BY rowid DESC LIMIT ? OFFSET ?",
            (*values, DETAIL_PAGE_SIZE, (page - 1) * DETAIL_PAGE_SIZE),
        ).fetchall()
        return {"items": [dict(row) for row in rows], "total": total, "page": page, "pageSize": DETAIL_PAGE_SIZE}

    def _intent_detail(self, connection, result, row, evidence_page, history_page):
        result["evidence"] = self._page(connection, "memory_user_intent_evidence", "object_type=? AND object_id=?", (row["object_type"], row["id"]), evidence_page)
        result["history"] = self._page(connection, "memory_user_intent_events", "intent_id=?", (row["id"],), history_page)
        self._resolve_evidence(connection, result["evidence"]["items"])
        entity = connection.execute("SELECT name FROM memory_entity_nodes WHERE id=?", (row["primary_entity_id"],)).fetchone()
        result["item"]["entity_name"] = entity[0] if entity else ""
        self._events(result["history"]["items"])

    def _state_detail(self, connection, result, item_id, evidence_page, history_page):
        result["evidence"] = self._page(connection, "user_state_evidence", "state_id=?", (item_id,), evidence_page)
        result["history"] = self._page(connection, "user_state_events", "state_id=?", (item_id,), history_page)
        with _reader(self.memory_path) as memory:
            self._resolve_evidence(memory, result["evidence"]["items"])
        self._events(result["history"]["items"])

    @staticmethod
    def _events(events):
        for event in events:
            for field in ("previous_payload", "new_payload"):
                event[field] = _json(event.get(field), {})
            # Payloads contain domain fields, never embeddings or raw model prompts.
            for field in ("trigger_claim_ids", "trigger_intent_objects"):
                if field in event:
                    event[field] = _json(event[field], [])

    def _resolve_evidence(self, connection, items, *, include_sources=True):
        specs = {
            "fact": ("memory_facts", "summary"), "user_claim": ("memory_user_claims", "claim_text"),
            "user_activity": ("memory_user_activity_items", "activity_text"),
            "user_thread": ("memory_user_threads", "current_summary"),
            "user_thread_event": ("memory_user_thread_events", "summary"),
            "user_observed_pattern": ("memory_user_observed_patterns", "pattern_text"),
            "user_action_signal": ("memory_fact_user_action_signals", "action_text"),
            **{kind: ("memory_user_intent_items", "summary") for kind in INTENT_TYPES},
        }
        for evidence in items:
            kind, key = evidence["evidence_type"], evidence["evidence_id"]
            evidence.update(resolved=False, text="", source_segments=[])
            if "details" in evidence:
                evidence["details"] = _json(evidence["details"], {})
            spec = specs.get(kind)
            if spec is None or not _table(connection, spec[0]):
                continue
            id_column = "signal_id" if kind == "user_action_signal" else "id"
            row = connection.execute(f"SELECT * FROM {spec[0]} WHERE {id_column}=?", (key,)).fetchone()
            if row is None or kind in INTENT_TYPES and row["object_type"] != kind:
                continue
            row = dict(row)
            evidence.update(resolved=True, text=_bounded(row.get(spec[1])), occurred_at=row.get("event_time_key") or row.get("last_observed_at") or row.get("updated_at") or "")
            if not include_sources:
                continue
            facts = [key] if kind == "fact" else _json(row.get("evidence_fact_ids"), [])
            segments = _json(row.get("source_segment_ids"), [])
            if kind == "user_action_signal":
                facts = [row["fact_id"]]
            if kind == "user_claim" and _table(connection, "memory_user_claim_evidence"):
                facts = [item[0] for item in connection.execute(
                    "SELECT evidence_id FROM memory_user_claim_evidence WHERE claim_id=? AND evidence_type='fact' LIMIT 10", (key,),
                )]
            if kind == "user_observed_pattern" and _table(connection, "memory_user_observed_pattern_signals") and _table(connection, "memory_fact_user_action_signals"):
                facts = [item[0] for item in connection.execute(
                    "SELECT s.fact_id FROM memory_user_observed_pattern_signals p JOIN memory_fact_user_action_signals s ON s.signal_id=p.signal_id WHERE p.pattern_id=? LIMIT 10", (key,),
                )]
            if kind in INTENT_TYPES and _table(connection, "memory_user_intent_evidence"):
                facts = [item[0] for item in connection.execute(
                    "SELECT evidence_id FROM memory_user_intent_evidence WHERE object_type=? AND object_id=? AND evidence_type='fact' LIMIT 10", (kind, key),
                )]
            facts = [value for value in facts if type(value) is int and value > 0][:10]
            if facts and _table(connection, "memory_fact_source_mapping"):
                placeholders = ",".join("?" for _ in facts)
                segments.extend(item[0] for item in connection.execute(
                    f"SELECT DISTINCT source_segment_id FROM memory_fact_source_mapping WHERE fact_id IN ({placeholders}) LIMIT 6", facts,
                ))
            segments = sorted({value for value in segments if type(value) is int and value > 0})[:6]
            if segments and _table(connection, "memory_source_segments"):
                placeholders = ",".join("?" for _ in segments)
                evidence["source_segments"] = [{"id": item["id"], "text": _bounded(item["text"], 2000), "started_at": item["started_at"], "ended_at": item["ended_at"]} for item in connection.execute(
                    f"SELECT id,text,started_at,ended_at FROM memory_source_segments WHERE id IN ({placeholders}) ORDER BY id", segments,
                )]
