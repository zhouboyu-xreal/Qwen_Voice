#!/usr/bin/env python3
"""SQLite storage for the unified memory prototype.

The schema defines the current unified memory line:

    memory_source_segments + memory_facts -> memory_user_activity_items
                           memory_facts -> user_claims / user-intent items

`memory_recall_documents` is a derived retrieval projection. Every retrievable
memory object writes one document there; source tables retain only domain data.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import unicodedata
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .memory_types import MemoryUnit

try:
    import jieba
except ImportError:  # pragma: no cover - exercised only in minimal installs
    jieba = None

_IDENTITY_FTS_TABLES = {
    "memory_recall_documents": "memory_recall_documents_identity_fts",
}

_LEXICAL_DATE_PATTERNS = (
    re.compile(r"(?<!\d)\d{4}[-/.年]\d{1,2}[-/.月]\d{1,2}(?:日)?(?!\d)"),
    re.compile(r"(?<!\d)\d{1,2}月\d{1,2}日?(?!\d)"),
)


def _lexical_index_text(identity_text: Any) -> str:
    """Convert display identity text into deterministic FTS search tokens.

    Recall documents keep the original ``identity_text`` for embeddings and
    display. FTS receives a separate token stream so Chinese text can be
    searched by words instead of relying on SQLite's default tokenizer.
    Dates are excluded here because fact time is filtered through structured
    time columns rather than lexical coincidence.
    """
    text_lines: List[str] = []
    for line in str(identity_text or "").splitlines():
        clean_line = line.strip()
        colon_positions = [
            position
            for position in (clean_line.find(":"), clean_line.find("："))
            if position >= 0
        ]
        if colon_positions:
            # identity_text is formatted as one ``field: value`` per line.
            # Index only the value so labels such as ``summary`` and
            # ``entities`` do not become searchable memory content.
            clean_line = clean_line[min(colon_positions) + 1 :].strip()
        if clean_line:
            text_lines.append(clean_line)
    text = "\n".join(text_lines)
    for pattern in _LEXICAL_DATE_PATTERNS:
        text = pattern.sub(" ", text)
    if not text.strip():
        return ""

    if jieba is not None:
        # Search mode keeps useful sub-tokens for Chinese compounds while the
        # regular cut preserves the complete domain phrase when available.
        raw_tokens = [
            *jieba.lcut(text, HMM=False),
            *jieba.cut_for_search(text, HMM=False),
        ]
    else:
        # Keep the database usable in minimal environments. This fallback is
        # deliberately simple; production installs should include jieba.
        raw_tokens = []
        for match in re.findall(
            r"[A-Za-z][A-Za-z0-9_.$'-]*|\d+(?:/\d+)?|[\u4e00-\u9fff]+",
            text,
        ):
            if re.fullmatch(r"[\u4e00-\u9fff]+", match):
                raw_tokens.append(match)
                raw_tokens.extend(match)
                raw_tokens.extend(
                    match[index : index + 2]
                    for index in range(len(match) - 1)
                )
            else:
                raw_tokens.append(match)

    tokens: List[str] = []
    seen: set[str] = set()
    for raw_token in raw_tokens:
        token = re.sub(r"\s+", "", str(raw_token or "")).strip()
        if not token or re.fullmatch(r"[^\w\u4e00-\u9fff]+", token):
            continue
        if token not in seen:
            seen.add(token)
            tokens.append(token)
    return " ".join(tokens)


def local_now_text() -> str:
    return datetime.now().astimezone().isoformat()


def _coerce_reference_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value or "").strip()
        if not text:
            dt = datetime.now().astimezone()
        else:
            try:
                dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
            except ValueError:
                dt = datetime.strptime(text, "%Y-%m-%d %H:%M:%S")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.now().astimezone().tzinfo)
    return dt


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _json_loads(value: Any, default: Any) -> Any:
    if value in (None, ""):
        return default
    try:
        return json.loads(value)
    except Exception:
        return default


def _embedding_to_blob(embedding: Optional[np.ndarray]) -> Optional[bytes]:
    if embedding is None:
        return None
    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    return vector.tobytes()


def _blob_to_embedding(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    try:
        vector = np.frombuffer(bytes(value), dtype=np.float32)
    except Exception:
        return None
    return vector.reshape(1, -1).astype(np.float32)


class UserMemoryDB:
    """SQLite facade for the current memory schema and its derived indexes."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            timeout=30.0,
        )
        self._conn.row_factory = sqlite3.Row
        self._transaction_depth = 0
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def close(self) -> None:
        self._conn.close()

    def open_reader(self) -> "UserMemoryDB":
        """Open a query-only connection without running schema initialization."""
        reader = object.__new__(UserMemoryDB)
        reader.db_path = self.db_path
        reader._conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            timeout=30.0,
        )
        reader._conn.row_factory = sqlite3.Row
        reader._transaction_depth = 0
        reader._conn.execute("PRAGMA query_only=ON")
        reader._conn.execute("PRAGMA foreign_keys=ON")
        reader._conn.execute("PRAGMA busy_timeout=30000")
        return reader

    @contextmanager
    def reader_transaction(self):
        """Read one committed SQLite snapshot and close its connection."""
        reader = self.open_reader()
        try:
            reader._conn.execute("BEGIN")
            yield reader
        finally:
            try:
                reader._conn.rollback()
            finally:
                reader.close()

    @contextmanager
    def transaction(self):
        """Group database mutations into one commit or rollback boundary."""
        is_outermost = self._transaction_depth == 0
        if is_outermost:
            self._conn.execute("BEGIN")
        self._transaction_depth += 1
        try:
            yield self
        except Exception:
            self._transaction_depth -= 1
            if is_outermost:
                self._conn.rollback()
            raise
        else:
            self._transaction_depth -= 1
            if is_outermost:
                self._conn.commit()

    def _commit_if_needed(self) -> None:
        """Commit standalone writes while deferring commits in a transaction."""
        if self._transaction_depth == 0:
            self._conn.commit()

    def _init_schema(self) -> None:
        if self._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' "
            "AND name='memory_user_observed_pattern_occurrences'"
        ).fetchone():
            self._conn.close()
            raise RuntimeError(
                "Observed-pattern evidence uses action signal IDs now; "
                "back up the retired database and use a fresh path"
            )
        for table in (
            "memory_fact_user_claim_signals",
            "memory_fact_user_intent_signals",
        ):
            if self._conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                (table,),
            ).fetchone() and "signal_id" not in {
                row["name"] for row in self._conn.execute(f"PRAGMA table_info({table})")
            }:
                self._conn.close()
                raise RuntimeError(
                    f"{table} uses the retired schema without signal_id; "
                    "back up the database and use a fresh path"
                )
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS memory_user_activity_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                activity_text TEXT NOT NULL CHECK(length(trim(activity_text)) > 0),
                user_role TEXT NOT NULL CHECK(user_role IN ('actor', 'participant', 'contextual')),
                confidence REAL NOT NULL DEFAULT 0.0 CHECK(confidence BETWEEN 0.0 AND 1.0),
                started_at TEXT,
                ended_at TEXT,
                evidence_fact_ids TEXT NOT NULL DEFAULT '[]',
                source_segment_ids TEXT NOT NULL DEFAULT '[]',
                metadata TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS memory_user_threads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                creation_key TEXT NOT NULL UNIQUE,
                title TEXT NOT NULL CHECK(length(trim(title)) > 0),
                identity_description TEXT NOT NULL,
                current_summary TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('active', 'closed')),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS memory_user_thread_links (
                thread_id INTEGER NOT NULL REFERENCES memory_user_threads(id),
                object_type TEXT NOT NULL CHECK(object_type IN ('user_activity', 'fact')),
                object_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY(thread_id, object_type, object_id)
            );
            CREATE INDEX IF NOT EXISTS idx_user_thread_links_object
                ON memory_user_thread_links(object_type, object_id);
            CREATE TABLE IF NOT EXISTS memory_user_thread_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id INTEGER NOT NULL REFERENCES memory_user_threads(id),
                event_key TEXT NOT NULL UNIQUE,
                event_type TEXT NOT NULL CHECK(event_type IN
                    ('progress', 'change', 'blocked', 'resolved', 'closed', 'reopened')),
                summary TEXT NOT NULL CHECK(length(trim(summary)) > 0),
                evidence_activity_ids TEXT NOT NULL DEFAULT '[]',
                evidence_fact_ids TEXT NOT NULL DEFAULT '[]',
                occurred_at TEXT,
                known_at TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_user_thread_events_thread
                ON memory_user_thread_events(thread_id, id);

            CREATE TABLE IF NOT EXISTS memory_source_segments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                processed_for_user_activity INTEGER NOT NULL DEFAULT 0,
                text TEXT NOT NULL,
                started_at TEXT NOT NULL DEFAULT '',
                ended_at TEXT NOT NULL DEFAULT '',
                tags TEXT NOT NULL DEFAULT '[]',
                metadata TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS memory_facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fact_type TEXT NOT NULL DEFAULT 'context',
                summary TEXT NOT NULL,
                keywords TEXT NOT NULL DEFAULT '[]',
                entities TEXT NOT NULL DEFAULT '[]',
                entity_ids TEXT NOT NULL DEFAULT '[]',
                fact_topic TEXT NOT NULL DEFAULT '',
                event_time_key TEXT NOT NULL DEFAULT '',
                dialogue_time_key TEXT NOT NULL DEFAULT '',
                confidence REAL NOT NULL DEFAULT 0.85,
                importance REAL NOT NULL DEFAULT 0.5,
                metadata TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS memory_fact_source_mapping (
                fact_id INTEGER NOT NULL,
                source_segment_id INTEGER NOT NULL,
                PRIMARY KEY(fact_id, source_segment_id),
                FOREIGN KEY(fact_id) REFERENCES memory_facts(id) ON DELETE CASCADE,
                FOREIGN KEY(source_segment_id) REFERENCES memory_source_segments(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS memory_recall_documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                object_type TEXT NOT NULL,
                object_id INTEGER NOT NULL,
                title TEXT NOT NULL DEFAULT '',
                summary TEXT NOT NULL DEFAULT '',
                identity_text TEXT NOT NULL DEFAULT '',
                identity_text_embedding BLOB,
                entity_ids TEXT NOT NULL DEFAULT '[]',
                topic_keys TEXT NOT NULL DEFAULT '[]',
                time_start TEXT NOT NULL DEFAULT '',
                time_end TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT '',
                confidence REAL NOT NULL DEFAULT 0.0,
                importance REAL NOT NULL DEFAULT 0.0,
                metadata TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(object_type, object_id)
            );

            CREATE TABLE IF NOT EXISTS memory_entity_nodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                type TEXT NOT NULL DEFAULT 'OTHER',
                created_at TEXT NOT NULL DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS memory_owner_entity_aliases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                owner_entity_id INTEGER NOT NULL,
                alias_name TEXT NOT NULL,
                alias_key TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'candidate',
                confidence REAL NOT NULL DEFAULT 0.0,
                evidence_fact_id INTEGER,
                evidence_type TEXT NOT NULL DEFAULT '',
                evidence_text TEXT NOT NULL DEFAULT '',
                source_speaker TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(owner_entity_id, alias_key),
                FOREIGN KEY(owner_entity_id) REFERENCES memory_entity_nodes(id) ON DELETE CASCADE,
                FOREIGN KEY(evidence_fact_id) REFERENCES memory_facts(id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS memory_fact_user_claim_signals (
                signal_id INTEGER PRIMARY KEY AUTOINCREMENT,
                fact_id INTEGER NOT NULL,
                signal_key TEXT NOT NULL,
                subject_entity_id INTEGER,
                claim_type_hint TEXT NOT NULL DEFAULT '',
                claim_anchor TEXT NOT NULL DEFAULT '',
                claim_anchor_key TEXT NOT NULL DEFAULT '',
                evidence_basis TEXT NOT NULL DEFAULT '',
                confidence REAL NOT NULL DEFAULT 0.0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(fact_id, signal_key),
                FOREIGN KEY(fact_id) REFERENCES memory_facts(id) ON DELETE CASCADE,
                FOREIGN KEY(subject_entity_id) REFERENCES memory_entity_nodes(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS memory_fact_user_action_signals (
                signal_id INTEGER PRIMARY KEY AUTOINCREMENT,
                fact_id INTEGER NOT NULL,
                signal_key TEXT NOT NULL,
                subject_entity_id INTEGER NOT NULL,
                action_kind TEXT NOT NULL CHECK(action_kind IN ('action', 'choice', 'participation')),
                action_text TEXT NOT NULL CHECK(length(trim(action_text)) > 0),
                action_text_embedding BLOB,
                event_time_key TEXT NOT NULL DEFAULT '',
                location TEXT NOT NULL DEFAULT '',
                owner_evidence_type TEXT NOT NULL,
                owner_evidence_text TEXT NOT NULL DEFAULT '',
                other_evidence_text TEXT NOT NULL DEFAULT '',
                evidence_basis TEXT NOT NULL DEFAULT '',
                confidence REAL NOT NULL DEFAULT 0.0 CHECK(confidence BETWEEN 0.0 AND 1.0),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(fact_id, signal_key),
                FOREIGN KEY(fact_id) REFERENCES memory_facts(id) ON DELETE CASCADE,
                FOREIGN KEY(subject_entity_id) REFERENCES memory_entity_nodes(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS memory_fact_user_intent_signals (
                signal_id INTEGER PRIMARY KEY AUTOINCREMENT,
                fact_id INTEGER NOT NULL,
                signal_key TEXT NOT NULL,
                subject_entity_id INTEGER,
                signal_purpose TEXT NOT NULL DEFAULT '',
                candidate_object_types TEXT NOT NULL DEFAULT '[]',
                operation_hint TEXT NOT NULL DEFAULT '',
                user_role TEXT NOT NULL DEFAULT '',
                user_intent_anchor TEXT NOT NULL DEFAULT '',
                user_intent_anchor_key TEXT NOT NULL DEFAULT '',
                assertion_source TEXT NOT NULL DEFAULT '',
                evidence_basis TEXT NOT NULL DEFAULT '',
                confidence REAL NOT NULL DEFAULT 0.0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(fact_id, signal_key),
                FOREIGN KEY(fact_id) REFERENCES memory_facts(id) ON DELETE CASCADE,
                FOREIGN KEY(subject_entity_id) REFERENCES memory_entity_nodes(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS memory_entity_mapping (
                entity_id INTEGER PRIMARY KEY,
                fact_id TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(entity_id) REFERENCES memory_entity_nodes(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS memory_topic_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic_kind TEXT NOT NULL,
                topic_name TEXT NOT NULL,
                topic_key TEXT NOT NULL,
                fact_occurrence_count INTEGER NOT NULL DEFAULT 0,
                first_seen_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(topic_kind, topic_key)
            );

            CREATE TABLE IF NOT EXISTS memory_topic_mapping (
                topic_item_id INTEGER PRIMARY KEY,
                fact_ids TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(topic_item_id) REFERENCES memory_topic_items(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS memory_user_claims (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject_entity_id INTEGER NOT NULL,
                predicate TEXT NOT NULL,
                object_entity_id INTEGER NOT NULL DEFAULT 0,
                claim_text TEXT NOT NULL DEFAULT '',
                claim_type TEXT NOT NULL,
                claim_origin TEXT NOT NULL CHECK(claim_origin IN ('explicit', 'derived')),
                primary_domain TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'candidate',
                confidence REAL NOT NULL DEFAULT 0.7,
                valid_from TEXT NOT NULL DEFAULT '',
                valid_to TEXT NOT NULL DEFAULT '',
                source_actor_entity_id INTEGER,
                extractor_version TEXT NOT NULL DEFAULT '',
                prompt_version TEXT NOT NULL DEFAULT '',
                metadata TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(subject_entity_id) REFERENCES memory_entity_nodes(id) ON DELETE CASCADE,
                FOREIGN KEY(source_actor_entity_id) REFERENCES memory_entity_nodes(id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS memory_user_claim_evidence (
                claim_id INTEGER NOT NULL,
                evidence_type TEXT NOT NULL,
                evidence_id INTEGER NOT NULL,
                role TEXT NOT NULL DEFAULT 'support',
                weight REAL NOT NULL DEFAULT 1.0,
                observed_at TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(claim_id, evidence_type, evidence_id, role),
                FOREIGN KEY(claim_id) REFERENCES memory_user_claims(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS memory_user_claim_derivations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                claim_id INTEGER NOT NULL,
                rule_id TEXT NOT NULL,
                rule_version TEXT NOT NULL DEFAULT '',
                derivation_key TEXT NOT NULL UNIQUE,
                status TEXT NOT NULL DEFAULT 'active',
                derived_at TEXT NOT NULL DEFAULT '',
                invalidated_at TEXT NOT NULL DEFAULT '',
                metadata TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(claim_id) REFERENCES memory_user_claims(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS memory_user_claim_premises (
                derivation_id INTEGER NOT NULL,
                premise_claim_id INTEGER NOT NULL,
                premise_role TEXT NOT NULL DEFAULT 'support',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(derivation_id, premise_claim_id, premise_role),
                FOREIGN KEY(derivation_id) REFERENCES memory_user_claim_derivations(id) ON DELETE CASCADE,
                FOREIGN KEY(premise_claim_id) REFERENCES memory_user_claims(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS memory_user_claim_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                target_claim_id INTEGER NOT NULL,
                trigger_claim_id INTEGER,
                event_type TEXT NOT NULL DEFAULT 'status_transition',
                previous_status TEXT NOT NULL,
                new_status TEXT NOT NULL,
                semantic_relation TEXT NOT NULL DEFAULT '',
                effective_at TEXT NOT NULL DEFAULT '',
                decision_source TEXT NOT NULL DEFAULT '',
                semantic_confidence REAL,
                semantic_reason TEXT NOT NULL DEFAULT '',
                policy_reason TEXT NOT NULL DEFAULT '',
                details TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                FOREIGN KEY(target_claim_id) REFERENCES memory_user_claims(id) ON DELETE CASCADE,
                FOREIGN KEY(trigger_claim_id) REFERENCES memory_user_claims(id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS memory_user_observed_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject_entity_id INTEGER NOT NULL,
                pattern_key TEXT NOT NULL,
                pattern_text TEXT NOT NULL,
                condition_text TEXT NOT NULL DEFAULT '',
                primary_domain TEXT NOT NULL DEFAULT '',
                confidence REAL NOT NULL DEFAULT 0.7,
                metadata TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(subject_entity_id, pattern_key),
                FOREIGN KEY(subject_entity_id) REFERENCES memory_entity_nodes(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS memory_user_observed_pattern_signals (
                pattern_id INTEGER NOT NULL,
                signal_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY(pattern_id, signal_id),
                FOREIGN KEY(pattern_id) REFERENCES memory_user_observed_patterns(id) ON DELETE CASCADE,
                FOREIGN KEY(signal_id) REFERENCES memory_fact_user_action_signals(signal_id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS memory_user_observed_pattern_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                target_pattern_id INTEGER NOT NULL,
                trigger_pattern_id INTEGER,
                semantic_relation TEXT NOT NULL,
                support_signal_ids TEXT NOT NULL DEFAULT '[]',
                candidate_snapshot TEXT NOT NULL DEFAULT '{}',
                semantic_confidence REAL NOT NULL DEFAULT 0.0,
                semantic_reason TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                UNIQUE(target_pattern_id, semantic_relation, support_signal_ids),
                FOREIGN KEY(target_pattern_id) REFERENCES memory_user_observed_patterns(id) ON DELETE CASCADE,
                FOREIGN KEY(trigger_pattern_id) REFERENCES memory_user_observed_patterns(id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS memory_user_intent_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                object_type TEXT NOT NULL CHECK(object_type IN ('goal', 'plan', 'work_item')),
                world_owner_entity_id INTEGER NOT NULL,
                primary_entity_id INTEGER NOT NULL,
                canonical_key TEXT NOT NULL,
                summary TEXT NOT NULL,
                primary_domain TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL,
                time_start TEXT NOT NULL DEFAULT '',
                time_end TEXT NOT NULL DEFAULT '',
                confidence REAL NOT NULL DEFAULT 0.7,
                details TEXT NOT NULL DEFAULT '{}',
                metadata TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(world_owner_entity_id) REFERENCES memory_entity_nodes(id) ON DELETE CASCADE,
                FOREIGN KEY(primary_entity_id) REFERENCES memory_entity_nodes(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS memory_user_intent_lifecycle_observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_id INTEGER NOT NULL UNIQUE,
                object_type TEXT NOT NULL CHECK(object_type IN ('goal', 'plan', 'work_item')),
                target_description TEXT NOT NULL CHECK(length(trim(target_description)) > 0),
                observation_text TEXT NOT NULL CHECK(length(trim(observation_text)) > 0),
                lifecycle_evidence TEXT NOT NULL CHECK(
                    (object_type = 'goal' AND lifecycle_evidence IN (
                        'confirmed', 'updated', 'completed', 'cancelled', 'blocked'
                    )) OR (object_type = 'plan' AND lifecycle_evidence IN (
                        'confirmed', 'updated', 'occurred', 'cancelled', 'rescheduled', 'blocked'
                    )) OR (object_type = 'work_item' AND lifecycle_evidence IN (
                        'confirmed', 'updated', 'completed', 'cancelled', 'blocked'
                    ))
                ),
                effective_at TEXT NOT NULL DEFAULT '',
                confidence REAL NOT NULL DEFAULT 0.7 CHECK(confidence BETWEEN 0.0 AND 1.0),
                details TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(signal_id) REFERENCES memory_fact_user_intent_signals(signal_id)
                    ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS memory_user_intent_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                intent_id INTEGER NOT NULL,
                event_type TEXT NOT NULL CHECK(event_type IN ('created', 'updated')),
                previous_payload TEXT NOT NULL DEFAULT '{}',
                new_payload TEXT NOT NULL,
                source_signal_id INTEGER NOT NULL,
                lifecycle_observation_id INTEGER,
                effective_at TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                UNIQUE(source_signal_id, intent_id, event_type),
                FOREIGN KEY(intent_id) REFERENCES memory_user_intent_items(id) ON DELETE CASCADE,
                FOREIGN KEY(source_signal_id) REFERENCES memory_fact_user_intent_signals(signal_id)
                    ON DELETE CASCADE,
                FOREIGN KEY(lifecycle_observation_id)
                    REFERENCES memory_user_intent_lifecycle_observations(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS memory_user_intent_evidence (
                object_type TEXT NOT NULL,
                object_id INTEGER NOT NULL,
                evidence_type TEXT NOT NULL,
                evidence_id INTEGER NOT NULL,
                role TEXT NOT NULL DEFAULT 'support',
                observed_at TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(object_type, object_id, evidence_type, evidence_id, role)
            );

            CREATE INDEX IF NOT EXISTS idx_memory_facts_event_time ON memory_facts(event_time_key);
            CREATE INDEX IF NOT EXISTS idx_memory_facts_dialogue_time ON memory_facts(dialogue_time_key);
            CREATE INDEX IF NOT EXISTS idx_memory_recall_documents_object
            ON memory_recall_documents(object_type, object_id);
            CREATE INDEX IF NOT EXISTS idx_memory_recall_documents_type_status_time
            ON memory_recall_documents(object_type, status, time_end, updated_at);
            CREATE INDEX IF NOT EXISTS idx_memory_source_segments_activity
            ON memory_source_segments(processed_for_user_activity, id);
            CREATE INDEX IF NOT EXISTS idx_memory_fact_source_segment
            ON memory_fact_source_mapping(source_segment_id, fact_id);
            CREATE INDEX IF NOT EXISTS idx_memory_topic_items_kind_seen
            ON memory_topic_items(topic_kind, last_seen_at DESC);
            CREATE INDEX IF NOT EXISTS idx_memory_user_claims_subject
            ON memory_user_claims(subject_entity_id, claim_type, claim_origin, status);
            CREATE INDEX IF NOT EXISTS idx_memory_user_observed_patterns_subject
            ON memory_user_observed_patterns(subject_entity_id, primary_domain, updated_at);
            CREATE INDEX IF NOT EXISTS idx_memory_user_observed_pattern_signals_signal
            ON memory_user_observed_pattern_signals(signal_id, pattern_id);
            CREATE INDEX IF NOT EXISTS idx_memory_user_observed_pattern_events_target
            ON memory_user_observed_pattern_events(target_pattern_id, id DESC);
            CREATE INDEX IF NOT EXISTS idx_memory_owner_entity_aliases_name
            ON memory_owner_entity_aliases(alias_key, status);
            CREATE INDEX IF NOT EXISTS idx_memory_user_claim_evidence_claim
            ON memory_user_claim_evidence(claim_id, role, evidence_type);
            CREATE INDEX IF NOT EXISTS idx_memory_user_claim_derivations_claim
            ON memory_user_claim_derivations(claim_id, status, updated_at);
            CREATE INDEX IF NOT EXISTS idx_memory_user_claim_premises_claim
            ON memory_user_claim_premises(premise_claim_id, derivation_id);
            CREATE INDEX IF NOT EXISTS idx_memory_fact_user_claim_signals_group
            ON memory_fact_user_claim_signals(
                subject_entity_id, claim_type_hint, claim_anchor_key, fact_id
            );
            CREATE INDEX IF NOT EXISTS idx_memory_fact_user_action_signals_subject_time
            ON memory_fact_user_action_signals(subject_entity_id, event_time_key, fact_id);
            CREATE INDEX IF NOT EXISTS idx_memory_fact_user_intent_signals_group
            ON memory_fact_user_intent_signals(
                subject_entity_id, user_intent_anchor_key, signal_purpose, fact_id
            );
            CREATE INDEX IF NOT EXISTS idx_memory_user_claim_events_target
            ON memory_user_claim_events(target_claim_id, effective_at DESC, id DESC);
            CREATE INDEX IF NOT EXISTS idx_memory_user_claim_events_trigger
            ON memory_user_claim_events(trigger_claim_id, id DESC);
            CREATE INDEX IF NOT EXISTS idx_memory_user_intent_items_owner_type_status
            ON memory_user_intent_items(world_owner_entity_id, object_type, status, updated_at DESC);
            CREATE INDEX IF NOT EXISTS idx_memory_user_intent_items_primary_time
            ON memory_user_intent_items(object_type, primary_entity_id, time_start, time_end);
            CREATE INDEX IF NOT EXISTS idx_memory_user_intent_evidence_object
            ON memory_user_intent_evidence(object_type, object_id, role);
            CREATE INDEX IF NOT EXISTS idx_memory_user_intent_events_intent
            ON memory_user_intent_events(intent_id, id);
            """
        )
        # These tables belonged to the abandoned read-time user-state snapshot
        # prototype.  User states are now incrementally updated durable objects.
        self._init_identity_fts()
        self._commit_if_needed()

    def _init_identity_fts(self) -> None:
        """Create and populate the tokenized BM25 index for recall documents."""
        for source_table, fts_table in _IDENTITY_FTS_TABLES.items():
            self._conn.execute(
                f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS {fts_table} USING fts5(
                    lexical_index_text
                )
                """
            )
            existing_ids = {
                int(row["rowid"])
                for row in self._conn.execute(
                    f"SELECT rowid FROM {fts_table}"
                ).fetchall()
            }
            source_rows = self._conn.execute(
                f"SELECT id, identity_text FROM {source_table}"
            ).fetchall()
            for row in source_rows:
                row_id = int(row["id"])
                if row_id in existing_ids:
                    continue
                self._conn.execute(
                    f"INSERT INTO {fts_table} (rowid, lexical_index_text) VALUES (?, ?)",
                    (row_id, _lexical_index_text(row["identity_text"])),
                )

    def _sync_identity_fts(
        self,
        *,
        source_table: str,
        row_id: int,
        identity_text: str,
    ) -> None:
        """Keep the tokenized BM25 document synchronized with its source row."""
        fts_table = _IDENTITY_FTS_TABLES[source_table]
        self._conn.execute(
            f"DELETE FROM {fts_table} WHERE rowid = ?",
            (int(row_id),),
        )
        self._conn.execute(
            f"INSERT INTO {fts_table} (rowid, lexical_index_text) VALUES (?, ?)",
            (int(row_id), _lexical_index_text(identity_text)),
        )

    def upsert_memory_recall_document(
        self,
        *,
        object_type: str,
        object_id: int,
        identity_text: str,
        identity_text_embedding: Optional[np.ndarray],
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
        """Create or refresh one derived retrieval document.

        ``object_type`` and ``object_id`` identify the authoritative row in a
        domain table. This projection deliberately has no foreign key because
        it spans several source tables; callers own its lifecycle alongside
        the corresponding source-object write.
        """
        normalized_type = str(object_type or "").strip().lower()
        normalized_id = int(object_id or 0)
        if not normalized_type or normalized_id <= 0:
            raise ValueError("object_type and a positive object_id are required")
        now = local_now_text()
        normalized_entity_ids = list(dict.fromkeys(
            int(value)
            for value in entity_ids or []
            if str(value).strip().isdigit() and int(value) > 0
        ))
        normalized_topic_keys = list(dict.fromkeys(
            str(value).strip()
            for value in topic_keys or []
            if str(value).strip()
        ))
        self._conn.execute(
            """
            INSERT INTO memory_recall_documents (
                object_type, object_id, title, summary,
                identity_text, identity_text_embedding, entity_ids, topic_keys,
                time_start, time_end, status, confidence, importance, metadata,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(object_type, object_id) DO UPDATE SET
                title = excluded.title,
                summary = excluded.summary,
                identity_text = excluded.identity_text,
                identity_text_embedding = excluded.identity_text_embedding,
                entity_ids = excluded.entity_ids,
                topic_keys = excluded.topic_keys,
                time_start = excluded.time_start,
                time_end = excluded.time_end,
                status = excluded.status,
                confidence = excluded.confidence,
                importance = excluded.importance,
                metadata = excluded.metadata,
                updated_at = excluded.updated_at
            """,
            (
                normalized_type,
                normalized_id,
                str(title or ""),
                str(summary or ""),
                str(identity_text or ""),
                _embedding_to_blob(identity_text_embedding),
                _json_dumps(normalized_entity_ids),
                _json_dumps(normalized_topic_keys),
                str(time_start or ""),
                str(time_end or ""),
                str(status or ""),
                float(confidence or 0.0),
                float(importance or 0.0),
                _json_dumps(metadata or {}),
                now,
                now,
            ),
        )
        row = self._conn.execute(
            """
            SELECT id FROM memory_recall_documents
            WHERE object_type = ? AND object_id = ?
            """,
            (normalized_type, normalized_id),
        ).fetchone()
        assert row is not None
        document_id = int(row["id"])
        self._sync_identity_fts(
            source_table="memory_recall_documents",
            row_id=document_id,
            identity_text=str(identity_text or ""),
        )
        self._commit_if_needed()
        return document_id

    @staticmethod
    def _terms_to_fts_query(terms: Sequence[str]) -> str:
        quoted: List[str] = []
        for term in terms or []:
            clean = re.sub(r"\s+", " ", str(term or "").strip())
            clean = clean.replace('"', '""')
            if clean:
                quoted.append(f'"{clean}"')
            if len(quoted) >= 12:
                break
        return " OR ".join(quoted)

    @staticmethod
    def _normalize_search_terms(terms: Optional[Sequence[str]]) -> List[str]:
        """Normalize lexical terms that were already tokenized upstream.

        Query tokenization belongs to ``MemoryNodeManager`` so recall and
        reflect use one consistent lexical policy. The database layer only
        deduplicates whitespace-normalized terms before building the FTS
        expression; jieba remains part of index-text construction above.
        """
        normalized: List[str] = []
        for term in terms or []:
            clean = re.sub(r"\s+", " ", str(term or "").strip()).lower()
            if clean and clean not in normalized:
                normalized.append(clean)
            if len(normalized) >= 32:
                return normalized
        return normalized

    def insert_user_activity_item(
        self,
        *,
        activity_text: str,
        user_role: str,
        confidence: float,
        started_at: str,
        ended_at: str,
        evidence_fact_ids: Sequence[int],
        source_segment_ids: Sequence[int],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        now = local_now_text()
        cur = self._conn.execute(
            """
            INSERT INTO memory_user_activity_items (
                activity_text, user_role, confidence, started_at, ended_at,
                evidence_fact_ids, source_segment_ids,
                metadata, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                activity_text,
                user_role,
                float(confidence),
                started_at,
                ended_at,
                _json_dumps(list(evidence_fact_ids)),
                _json_dumps(list(source_segment_ids)),
                _json_dumps(metadata or {}),
                now,
                now,
            ),
        )
        self._commit_if_needed()
        return int(cur.lastrowid)

    def get_user_activity_items_by_ids(
        self,
        activity_ids: Sequence[int],
    ) -> List[Dict[str, Any]]:
        ids = list(dict.fromkeys(
            int(value)
            for value in activity_ids or []
            if str(value).strip().isdigit() and int(value) > 0
        ))
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        rows = self._conn.execute(
            f"SELECT * FROM memory_user_activity_items WHERE id IN ({placeholders})",
            ids,
        ).fetchall()
        by_id = {int(row["id"]): self._row_to_dict(row) for row in rows}
        result: List[Dict[str, Any]] = []
        for activity_id in ids:
            item = by_id.get(activity_id)
            if item is None:
                continue
            if not isinstance(item.get("evidence_fact_ids"), list):
                item["evidence_fact_ids"] = []
            if not isinstance(item.get("source_segment_ids"), list):
                item["source_segment_ids"] = []
            if not isinstance(item.get("metadata"), dict):
                item["metadata"] = {}
            result.append(item)
        return result

    def get_user_threads_by_ids(self, thread_ids: Sequence[int]) -> List[Dict[str, Any]]:
        ids = list(dict.fromkeys(int(value) for value in thread_ids))
        if not ids:
            return []
        rows = self._conn.execute(
            f"SELECT * FROM memory_user_threads WHERE id IN ({','.join('?' for _ in ids)})",
            ids,
        ).fetchall()
        by_id = {int(row['id']): dict(row) for row in rows}
        return [by_id[value] for value in ids if value in by_id]

    def search_user_threads(
        self, terms: Sequence[str], *, activity_id: int, limit: int = 12,
    ) -> List[Dict[str, Any]]:
        """Recall identities, not membership decisions; include closed lexical matches.

        Existing activity links take precedence on retries. A small recent-active
        fallback handles references whose project name is omitted from the activity.
        """
        limit = max(1, int(limit))
        linked = self._conn.execute(
            "SELECT t.* FROM memory_user_threads t JOIN memory_user_thread_links l "
            "ON t.id=l.thread_id WHERE l.object_type='user_activity' AND l.object_id=? "
            "ORDER BY t.id DESC LIMIT ?", (activity_id, limit),
        ).fetchall()
        normalized_terms = self._normalize_search_terms(terms)
        matched = []
        if normalized_terms:
            text_column = "lower(title || ' ' || identity_description || ' ' || current_summary)"
            conditions = [f"{text_column} LIKE ? ESCAPE '\\'" for _ in normalized_terms]
            parameters = [
                '%' + term.replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_') + '%'
                for term in normalized_terms
            ]
            matched = self._conn.execute(
                "SELECT * FROM memory_user_threads WHERE " + ' OR '.join(conditions)
                + " ORDER BY (" + ' + '.join(f"({condition})" for condition in conditions)
                + ") DESC, updated_at DESC, id DESC LIMIT ?",
                [*parameters, *parameters, limit],
            ).fetchall()
        recent = self._conn.execute(
            "SELECT * FROM memory_user_threads WHERE status='active' "
            "ORDER BY updated_at DESC, id DESC LIMIT ?", (min(3, limit),),
        ).fetchall()
        by_id = {}
        for row in [*linked, *matched, *recent]:
            by_id.setdefault(int(row['id']), dict(row))
        return list(by_id.values())[:limit]

    def get_user_thread_events(self, thread_id: int, *, limit: int = 16) -> List[Dict[str, Any]]:
        """Bound prompt context only; stored history is never truncated."""
        rows = self._conn.execute(
            "SELECT * FROM memory_user_thread_events WHERE thread_id=? "
            "ORDER BY id DESC LIMIT ?", (int(thread_id), max(1, int(limit))),
        ).fetchall()
        return [self._row_to_dict(row) for row in reversed(rows)]

    def get_user_thread_events_by_ids(self, event_ids: Sequence[int]) -> List[Dict[str, Any]]:
        """Load only the immutable changes supplied by a thread-update task."""
        ids = list(dict.fromkeys(int(value) for value in event_ids))
        by_id = {}
        for start in range(0, len(ids), 400):
            chunk = ids[start:start + 400]
            rows = self._conn.execute(
                "SELECT * FROM memory_user_thread_events WHERE id IN ("
                + ','.join('?' for _ in chunk) + ')', chunk,
            ).fetchall()
            by_id.update({int(row['id']): self._row_to_dict(row) for row in rows})
        return [by_id[value] for value in ids if value in by_id]

    def save_user_thread(
        self, *, thread_id: Optional[int], creation_key: str,
        title: str, identity_description: str, current_summary: str, status: str,
    ) -> int:
        now = local_now_text()
        if thread_id is None:
            self._conn.execute(
                "INSERT INTO memory_user_threads "
                "(creation_key,title,identity_description,current_summary,status,created_at,updated_at) "
                "VALUES (?,?,?,?,?,?,?) ON CONFLICT(creation_key) DO NOTHING",
                (creation_key, title, identity_description, current_summary, status, now, now),
            )
            thread_id = int(self._conn.execute(
                "SELECT id FROM memory_user_threads WHERE creation_key=?", (creation_key,),
            ).fetchone()['id'])
        else:
            cursor = self._conn.execute(
                "UPDATE memory_user_threads SET title=?,identity_description=?,current_summary=?,"
                "status=?,updated_at=? WHERE id=?",
                (title, identity_description, current_summary, status, now, int(thread_id)),
            )
            if cursor.rowcount != 1:
                raise ValueError(f'Unknown user thread: {thread_id}')
        self._commit_if_needed()
        return int(thread_id)

    def link_user_thread_evidence(
        self, thread_id: int, *, activity_ids: Sequence[int], fact_ids: Sequence[int],
    ) -> None:
        now = local_now_text()
        self._conn.executemany(
            "INSERT OR IGNORE INTO memory_user_thread_links "
            "(thread_id,object_type,object_id,created_at) VALUES (?,?,?,?)",
            [(int(thread_id), object_type, int(value), now)
             for object_type, ids in (('user_activity', activity_ids), ('fact', fact_ids))
             for value in set(ids)],
        )
        self._commit_if_needed()

    def insert_user_thread_event(
        self, thread_id: int, *, event_type: str, summary: str,
        evidence_activity_ids: Sequence[int], evidence_fact_ids: Sequence[int],
        occurred_at: Optional[str], known_at: str,
    ) -> Optional[int]:
        # Retry-safe for the same grounded delta; different evidence remains auditable.
        activity_ids = sorted(set(evidence_activity_ids))
        fact_ids = sorted(set(evidence_fact_ids))
        event_key = hashlib.sha256(_json_dumps([
            thread_id, event_type, summary.strip(), activity_ids, fact_ids, occurred_at,
        ]).encode('utf-8')).hexdigest()
        cursor = self._conn.execute(
            "INSERT OR IGNORE INTO memory_user_thread_events "
            "(thread_id,event_key,event_type,summary,evidence_activity_ids,evidence_fact_ids,"
            "occurred_at,known_at,created_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (thread_id, event_key, event_type, summary.strip(), _json_dumps(activity_ids),
             _json_dumps(fact_ids), occurred_at, known_at, local_now_text()),
        )
        self._commit_if_needed()
        return int(cursor.lastrowid) if cursor.rowcount else None

    def insert_memory_source_segments(
        self,
        *,
        units: Sequence[MemoryUnit],
    ) -> List[int]:
        """Persist one submitted source-unit batch as one physical row.

        The table records batch lifecycle (pending / activity-observed),
        while ``metadata.units`` preserves the ordered typed units needed by
        activity extraction. Returning a one-item list preserves the existing
        caller contract of ``source_segment_ids``.
        """
        now = local_now_text()
        normalized_units: List[Dict[str, Any]] = []
        for unit in units or []:
            if not isinstance(unit, MemoryUnit):
                continue
            entries = unit.entries()
            if not entries:
                continue
            tags = list(dict.fromkeys(
                str(tag).strip()
                for tag in unit.tags or []
                if str(tag).strip()
            ))
            normalized_units.append({
                "unit_type": unit.unit_type,
                "texts": [entry["text"] for entry in entries],
                "speakers": [entry["speaker"] for entry in entries],
                "started_ats": [entry["started_at"] for entry in entries],
                "ended_ats": [entry["ended_at"] for entry in entries],
                "token_count": max(1, int(unit.token_count or 0)),
                "tags": tags,
            })
        if not normalized_units:
            return []

        batch_tags = list(dict.fromkeys(
            tag
            for item in normalized_units
            for tag in item["tags"]
        ))
        batch_started_at = next(
            (
                started_at
                for item in normalized_units
                for started_at in item["started_ats"]
                if started_at
            ),
            "",
        )
        batch_ended_at = next(
            (
                ended_at or started_at
                for item in reversed(normalized_units)
                for started_at, ended_at in zip(
                    reversed(item["started_ats"]),
                    reversed(item["ended_ats"]),
                )
                if ended_at or started_at
            ),
            batch_started_at,
        )
        cur = self._conn.execute(
            """
            INSERT INTO memory_source_segments (
                text, started_at, ended_at,
                tags, metadata, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "\n".join(
                    f"{speaker or 'unknown'}: {text}"
                    for unit in normalized_units
                    for speaker, text in zip(
                        unit["speakers"],
                        unit["texts"],
                    )
                ),
                batch_started_at,
                batch_ended_at,
                _json_dumps(batch_tags),
                _json_dumps({
                    "storage_format": "unit_batch_v1",
                    "unit_count": len(normalized_units),
                    "units": normalized_units,
                }),
                now,
                now,
            ),
        )
        self._commit_if_needed()
        return [int(cur.lastrowid)]

    def get_unprocessed_memory_source_segments_for_user_activity(
        self,
        *,
        limit: int = 240,
    ) -> List[MemoryUnit]:
        """Return typed source units waiting for activity observation.

        Each row is a physical batch; ``metadata.units`` stores the ordered
        logical units used for activity extraction.
        """
        rows = self._conn.execute(
            """
            SELECT *
            FROM memory_source_segments
            WHERE processed_for_user_activity = 0
            ORDER BY id ASC
            LIMIT ?
            """,
            (max(1, int(limit or 240)),),
        ).fetchall()
        source_units: List[MemoryUnit] = []
        for row in rows:
            batch_row = self._row_to_dict(row)
            batch_metadata = (
                batch_row.get("metadata")
                if isinstance(batch_row.get("metadata"), dict)
                else {}
            )
            batch_units = batch_metadata.get("units")
            if not isinstance(batch_units, list):
                continue
            for raw_unit in batch_units:
                if not isinstance(raw_unit, dict):
                    continue
                texts = [
                    str(value).strip()
                    for value in raw_unit.get("texts") or []
                    if str(value).strip()
                ]
                if not texts:
                    continue
                speakers = [
                    str(value).strip()
                    for value in raw_unit.get("speakers") or []
                ]
                started_ats = [
                    str(value).strip()
                    for value in raw_unit.get("started_ats") or []
                ]
                ended_ats = [
                    str(value).strip()
                    for value in raw_unit.get("ended_ats") or []
                ]
                if len(speakers) < len(texts):
                    speakers.extend([""] * (len(texts) - len(speakers)))
                if len(started_ats) < len(texts):
                    started_ats.extend([
                        str(batch_row.get("started_at") or "").strip()
                    ] * (len(texts) - len(started_ats)))
                if len(ended_ats) < len(texts):
                    ended_ats.extend([
                        str(
                            batch_row.get("ended_at")
                            or batch_row.get("started_at")
                            or ""
                        ).strip()
                    ] * (len(texts) - len(ended_ats)))
                unit_type = str(raw_unit.get("unit_type") or "").strip()
                source_units.append(MemoryUnit(
                    texts=texts,
                    speakers=speakers[:len(texts)],
                    token_count=max(
                        1,
                        int(raw_unit.get("token_count") or 0),
                    ),
                    started_ats=started_ats[:len(texts)],
                    ended_ats=ended_ats[:len(texts)],
                    unit_type=(
                        unit_type
                        if unit_type in {"transcript", "interaction"}
                        else "transcript"
                    ),
                    tags=[
                        str(tag).strip()
                        for tag in raw_unit.get("tags") or []
                        if str(tag).strip()
                    ],
                    source_unit_row_id=int(batch_row["id"]),
                ))
        def unit_time_key(unit: MemoryUnit) -> Tuple[float, int]:
            timestamp = unit.start_time or unit.end_time
            if not timestamp:
                return (float("inf"), int(unit.source_unit_row_id or 0))
            try:
                return (
                    _coerce_reference_datetime(timestamp).timestamp(),
                    int(unit.source_unit_row_id or 0),
                )
            except (TypeError, ValueError, OverflowError):
                return (float("inf"), int(unit.source_unit_row_id or 0))

        return sorted(source_units, key=unit_time_key)

    def mark_memory_source_segments_processed_for_user_activity(
        self,
        *,
        source_segment_ids: Sequence[int],
    ) -> int:
        """Mark an observed source window, including windows with no activity."""
        normalized_ids = list(dict.fromkeys(
            int(value)
            for value in source_segment_ids or []
            if str(value).strip().isdigit() and int(value) > 0
        ))
        if not normalized_ids:
            return 0
        placeholders = ",".join("?" for _ in normalized_ids)
        cur = self._conn.execute(
            f"""
            UPDATE memory_source_segments
            SET processed_for_user_activity = 1, updated_at = ?
            WHERE id IN ({placeholders})
            """,
            (local_now_text(), *normalized_ids),
        )
        self._commit_if_needed()
        return int(cur.rowcount or 0)

    def insert_fact_source_mappings(
        self, *, fact_ids: Sequence[int], source_segment_ids: Sequence[int]
    ) -> None:
        pairs = [
            (int(fact_id), int(source_id))
            for fact_id in fact_ids for source_id in source_segment_ids
        ]
        self._conn.executemany(
            "INSERT OR IGNORE INTO memory_fact_source_mapping (fact_id, source_segment_id) VALUES (?, ?)",
            pairs,
        )
        self._commit_if_needed()

    def get_memory_facts_for_source_segments(
        self, *, source_segment_ids: Sequence[int], limit: int = 240
    ) -> List[Dict[str, Any]]:
        ids = [int(value) for value in source_segment_ids]
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        rows = self._conn.execute(
            f"""SELECT DISTINCT fact.* FROM memory_facts AS fact
                JOIN memory_fact_source_mapping AS mapping ON mapping.fact_id = fact.id
                WHERE mapping.source_segment_id IN ({placeholders})
                ORDER BY fact.dialogue_time_key ASC, fact.id ASC LIMIT ?""",
            (*ids, max(1, int(limit))),
        ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def upsert_memory_topic_items(
        self,
        items: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Upsert topic registry items and merge their fact IDs.

        ``memory_topic_items`` holds one reusable topic name per
        ``(topic_kind, topic_key)``. Its companion mapping row deliberately
        stores the aggregated evidence IDs as JSON arrays, matching the
        compact topic-directory model used by this project.
        """
        grouped: Dict[str, Dict[str, Any]] = {}

        def normalize_ids(value: Any) -> List[int]:
            values = value if isinstance(value, (list, tuple, set)) else [value]
            normalized: List[int] = []
            for raw in values:
                try:
                    item_id = int(raw)
                except (TypeError, ValueError):
                    continue
                if item_id > 0 and item_id not in normalized:
                    normalized.append(item_id)
            return normalized

        for item in items or []:
            if not isinstance(item, dict):
                continue
            topic_kind = str(item.get("topic_kind") or "").strip().lower()
            topic_name = str(item.get("topic_name") or "").strip()
            topic_key = str(item.get("topic_key") or "").strip().lower()
            if topic_kind not in {"canonical", "aspect"} or not topic_name or not topic_key:
                continue
            group_key = f"{topic_kind}\x1f{topic_key}"
            grouped_item = grouped.setdefault(
                group_key,
                {
                    "topic_kind": topic_kind,
                    "topic_name": topic_name,
                    "topic_key": topic_key,
                    "fact_ids": [],
                },
            )
            for field in ("fact_ids",):
                for item_id in normalize_ids(item.get(field)):
                    if item_id not in grouped_item[field]:
                        grouped_item[field].append(item_id)

        report = {
            "topic_item_ids": [],
            "created_count": 0,
            "updated_count": 0,
            "fact_links_added": 0,
        }
        if not grouped:
            return report

        now = local_now_text()
        for item in grouped.values():
            existing = self._conn.execute(
                """
                SELECT id FROM memory_topic_items
                WHERE topic_kind = ? AND topic_key = ?
                """,
                (item["topic_kind"], item["topic_key"]),
            ).fetchone()
            if existing is None:
                cur = self._conn.execute(
                    """
                    INSERT INTO memory_topic_items (
                        topic_kind, topic_name, topic_key,
                        fact_occurrence_count,
                        first_seen_at, last_seen_at, created_at, updated_at
                    ) VALUES (?, ?, ?, 0, ?, ?, ?, ?)
                    """,
                    (
                        item["topic_kind"],
                        item["topic_name"],
                        item["topic_key"],
                        now,
                        now,
                        now,
                        now,
                    ),
                )
                topic_item_id = int(cur.lastrowid)
                report["created_count"] += 1
            else:
                topic_item_id = int(existing["id"])

            mapping = self._conn.execute(
                """
                SELECT fact_ids FROM memory_topic_mapping
                WHERE topic_item_id = ?
                """,
                (topic_item_id,),
            ).fetchone()
            existing_fact_ids = normalize_ids(
                _json_loads(mapping["fact_ids"], []) if mapping else []
            )
            new_fact_ids = [
                fact_id for fact_id in item["fact_ids"]
                if fact_id not in existing_fact_ids
            ]
            merged_fact_ids = [*existing_fact_ids, *new_fact_ids]

            self._conn.execute(
                """
                INSERT INTO memory_topic_mapping (
                    topic_item_id, fact_ids, created_at, updated_at
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(topic_item_id) DO UPDATE SET
                    fact_ids = excluded.fact_ids,
                    updated_at = excluded.updated_at
                """,
                (
                    topic_item_id,
                    _json_dumps(merged_fact_ids),
                    now,
                    now,
                ),
            )
            if new_fact_ids:
                self._conn.execute(
                    """
                    UPDATE memory_topic_items
                    SET fact_occurrence_count = fact_occurrence_count + ?,
                        last_seen_at = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        len(new_fact_ids),
                        now,
                        now,
                        topic_item_id,
                    ),
                )
                report["updated_count"] += 1
            report["topic_item_ids"].append(topic_item_id)
            report["fact_links_added"] += len(new_fact_ids)

        self._commit_if_needed()
        return report

    def list_memory_topic_items(self, *, limit: int = 240) -> List[Dict[str, Any]]:
        """Load recent topic registry items with their compact evidence IDs."""
        rows = self._conn.execute(
            """
            SELECT item.*, mapping.fact_ids
            FROM memory_topic_items AS item
            LEFT JOIN memory_topic_mapping AS mapping
                ON mapping.topic_item_id = item.id
            ORDER BY item.last_seen_at DESC, item.id DESC
            LIMIT ?
            """,
            (max(1, int(limit or 240)),),
        ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def get_user_signal_batches(
        self, fact_ids: Optional[Sequence[int]] = None,
    ) -> Dict[str, List[int]]:
        """Return stable signal IDs; signal content remains in the memory DB."""
        ids = sorted({
            int(value) for value in fact_ids or []
            if str(value).strip().isdigit() and int(value) > 0
        })
        if fact_ids is not None and not ids:
            return {
                "claim_signal_ids": [],
                "intent_signal_ids": [],
                "action_signal_ids": [],
            }
        condition = (
            f"AND fact_id IN ({','.join('?' for _ in ids)})" if fact_ids is not None
            else ""
        )
        claim_rows = self._conn.execute(
            "SELECT signal_id FROM memory_fact_user_claim_signals "
            f"WHERE subject_entity_id IS NOT NULL {condition} "
            "ORDER BY fact_id, signal_key",
            ids,
        ).fetchall()
        intent_rows = self._conn.execute(
            "SELECT signal_id FROM memory_fact_user_intent_signals "
            f"WHERE subject_entity_id IS NOT NULL {condition} ORDER BY fact_id, signal_key",
            ids,
        ).fetchall()
        action_rows = self._conn.execute(
            "SELECT signal_id FROM memory_fact_user_action_signals "
            f"WHERE subject_entity_id IS NOT NULL {condition} ORDER BY fact_id, signal_key",
            ids,
        ).fetchall()
        return {
            "claim_signal_ids": [int(row["signal_id"]) for row in claim_rows],
            "intent_signal_ids": [int(row["signal_id"]) for row in intent_rows],
            "action_signal_ids": [int(row["signal_id"]) for row in action_rows],
        }

    def get_user_signals_by_ids(
        self, signal_type: str, signal_ids: Sequence[int],
    ) -> List[Dict[str, Any]]:
        table = {
            "claim": "memory_fact_user_claim_signals",
            "intent": "memory_fact_user_intent_signals",
            "action": "memory_fact_user_action_signals",
        }.get(signal_type)
        if table is None:
            raise ValueError(f"Unknown signal type: {signal_type}")
        columns = "signal_id, fact_id, signal_key"
        if signal_type == "intent":
            columns += ", signal_purpose"
        ids = sorted({int(value) for value in signal_ids if int(value) > 0})
        rows: List[Dict[str, Any]] = []
        for offset in range(0, len(ids), 500):
            chunk = ids[offset:offset + 500]
            placeholders = ",".join("?" for _ in chunk)
            rows.extend(dict(row) for row in self._conn.execute(
                f"SELECT {columns} FROM {table} "
                f"WHERE signal_id IN ({placeholders}) ORDER BY signal_id",
                chunk,
            ).fetchall())
        return rows

    def get_user_signals_by_fact_ids(
        self, signal_type: str, fact_ids: Sequence[int],
    ) -> List[Dict[str, Any]]:
        table = {
            "claim": "memory_fact_user_claim_signals",
            "intent": "memory_fact_user_intent_signals",
            "action": "memory_fact_user_action_signals",
        }.get(signal_type)
        if table is None:
            raise ValueError(f"Unknown signal type: {signal_type}")
        ids = sorted({int(value) for value in fact_ids if int(value) > 0})
        rows: List[Dict[str, Any]] = []
        for offset in range(0, len(ids), 500):
            chunk = ids[offset:offset + 500]
            placeholders = ",".join("?" for _ in chunk)
            rows.extend(dict(row) for row in self._conn.execute(
                f"SELECT signal_id, fact_id, signal_key FROM {table} "
                f"WHERE fact_id IN ({placeholders}) ORDER BY fact_id, signal_id",
                chunk,
            ).fetchall())
        return rows

    def upsert_fact_user_action_signals(
        self,
        signals: Sequence[Dict[str, Any]],
    ) -> int:
        """Persist fact-grounded user actions without creating a claim."""
        now = local_now_text()
        changed = 0
        for signal in signals or []:
            self._conn.execute(
                """
                INSERT INTO memory_fact_user_action_signals (
                    fact_id, signal_key, subject_entity_id, action_kind,
                    action_text, event_time_key, location, owner_evidence_type,
                    owner_evidence_text, other_evidence_text, evidence_basis,
                    confidence, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(fact_id, signal_key) DO UPDATE SET
                    action_text_embedding = CASE
                        WHEN memory_fact_user_action_signals.action_text = excluded.action_text
                        THEN memory_fact_user_action_signals.action_text_embedding
                        ELSE NULL END,
                    action_text = excluded.action_text,
                    event_time_key = excluded.event_time_key,
                    location = excluded.location,
                    owner_evidence_type = excluded.owner_evidence_type,
                    owner_evidence_text = excluded.owner_evidence_text,
                    other_evidence_text = excluded.other_evidence_text,
                    evidence_basis = excluded.evidence_basis,
                    confidence = excluded.confidence,
                    updated_at = excluded.updated_at
                """,
                (
                    int(signal["fact_id"]), str(signal["signal_key"]),
                    int(signal["subject_entity_id"]), str(signal["action_kind"]),
                    str(signal["action_text"]), str(signal.get("event_time_key") or ""),
                    str(signal.get("location") or ""),
                    str(signal["owner_evidence_type"]),
                    str(signal.get("owner_evidence_text") or ""),
                    str(signal.get("other_evidence_text") or ""),
                    str(signal.get("evidence_basis") or ""),
                    float(signal.get("confidence") or 0.0), now, now,
                ),
            )
            changed += 1
        self._commit_if_needed()
        return changed

    def get_fact_user_action_signals(self, fact_id: int) -> List[Dict[str, Any]]:
        """Read one fact's action evidence for inspection and future consumers."""
        if int(fact_id or 0) <= 0:
            return []
        rows = self._conn.execute(
            "SELECT * FROM memory_fact_user_action_signals "
            "WHERE fact_id = ? ORDER BY signal_id",
            (int(fact_id),),
        ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def get_user_action_signals_by_ids(
        self, signal_ids: Sequence[int],
    ) -> List[Dict[str, Any]]:
        """Load the complete action signals selected for pattern extraction."""
        ids = sorted({int(value) for value in signal_ids if int(value) > 0})
        signals: List[Dict[str, Any]] = []
        for offset in range(0, len(ids), 500):
            chunk = ids[offset:offset + 500]
            placeholders = ",".join("?" for _ in chunk)
            rows = self._conn.execute(
                "SELECT * FROM memory_fact_user_action_signals "
                f"WHERE signal_id IN ({placeholders})",
                chunk,
            ).fetchall()
            signals.extend(self._row_to_dict(row) for row in rows)
        return sorted(signals, key=lambda signal: (
            int(signal["fact_id"]), int(signal["signal_id"])
        ))

    def get_user_action_signals(self) -> List[Dict[str, Any]]:
        """Read all user-self actions before semantic neighborhood selection."""
        rows = self._conn.execute(
            "SELECT * FROM memory_fact_user_action_signals "
            "ORDER BY signal_id DESC",
        ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def set_user_action_signal_embeddings(
        self, embeddings: Sequence[Tuple[int, np.ndarray]],
    ) -> None:
        """Cache action-text embeddings for later pattern updates."""
        for signal_id, embedding in embeddings:
            self._conn.execute(
                "UPDATE memory_fact_user_action_signals "
                "SET action_text_embedding = ? WHERE signal_id = ?",
                (_embedding_to_blob(embedding), int(signal_id)),
            )
        self._commit_if_needed()

    def upsert_fact_user_claim_signals(
        self,
        mappings: Sequence[Dict[str, Any]],
    ) -> int:
        """Persist fact-level signals for explicit user-claim extraction."""
        now = local_now_text()
        changed = 0
        for mapping in mappings or []:
            try:
                fact_id = int(mapping["fact_id"])
            except (KeyError, TypeError, ValueError):
                continue
            signal_key = str(mapping.get("signal_key") or "").strip()
            try:
                subject_entity_id = int(mapping["subject_entity_id"])
            except (KeyError, TypeError, ValueError):
                subject_entity_id = 0
            claim_type_hint = str(mapping.get("claim_type_hint") or "").strip()
            claim_anchor = str(mapping.get("claim_anchor") or "").strip()
            claim_anchor_key = str(mapping.get("claim_anchor_key") or "").strip()
            evidence_basis = str(mapping.get("evidence_basis") or "").strip()
            if fact_id <= 0 or (
                subject_entity_id <= 0
                or not claim_type_hint
                or not claim_anchor
                or not claim_anchor_key
            ):
                continue
            if not signal_key:
                signal_fingerprint = hashlib.sha256(
                    "\x1f".join((
                        claim_type_hint,
                        claim_anchor_key,
                        evidence_basis.lower(),
                    )).encode("utf-8")
                ).hexdigest()[:16]
                signal_key = (
                    f"{subject_entity_id}|{claim_type_hint}|"
                    f"{claim_anchor_key}|{signal_fingerprint}"
                )
            self._conn.execute(
                """
                INSERT INTO memory_fact_user_claim_signals (
                    fact_id, signal_key, subject_entity_id, claim_type_hint,
                    claim_anchor, claim_anchor_key, evidence_basis,
                    confidence, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(fact_id, signal_key) DO UPDATE SET
                    claim_anchor = excluded.claim_anchor,
                    evidence_basis = excluded.evidence_basis,
                    confidence = MAX(
                        memory_fact_user_claim_signals.confidence,
                        excluded.confidence
                    ),
                    updated_at = excluded.updated_at
                """,
                (
                    fact_id, signal_key,
                    subject_entity_id or None, claim_type_hint,
                    claim_anchor, claim_anchor_key, evidence_basis,
                    float(mapping.get("confidence") or 0.0), now, now,
                ),
            )
            changed += 1
        self._commit_if_needed()
        return changed

    def get_fact_user_claim_signals(
        self,
        fact_id: int,
    ) -> Dict[str, Dict[str, Any]]:
        """Load the normalized claim signals persisted for one stored fact."""
        if int(fact_id or 0) <= 0:
            return {}
        rows = self._conn.execute(
            """
            SELECT signal.*, entity.name AS subject
            FROM memory_fact_user_claim_signals AS signal
            LEFT JOIN memory_entity_nodes AS entity
                ON entity.id = signal.subject_entity_id
            WHERE signal.fact_id = ?
              AND signal.subject_entity_id IS NOT NULL
            ORDER BY signal.claim_type_hint ASC,
                     signal.claim_anchor_key ASC,
                     signal.signal_key ASC
            """,
            (int(fact_id),),
        ).fetchall()
        return {
            str(mapping["signal_key"]): mapping
            for row in rows
            for mapping in [self._row_to_dict(row)]
        }

    def upsert_fact_user_intent_signals(
        self,
        mappings: Sequence[Dict[str, Any]],
    ) -> int:
        """Persist normalized intent and lifecycle evidence signals for one fact."""
        now = local_now_text()
        changed = 0
        for mapping in mappings or []:
            try:
                fact_id = int(mapping["fact_id"])
            except (KeyError, TypeError, ValueError):
                continue
            signal_key = str(mapping.get("signal_key") or "").strip()
            try:
                subject_entity_id = int(mapping["subject_entity_id"])
            except (KeyError, TypeError, ValueError):
                subject_entity_id = 0
            signal_purpose = str(mapping.get("signal_purpose") or "").strip().lower()
            user_role = str(mapping.get("user_role") or "").strip().lower()
            user_intent_anchor = str(mapping.get("user_intent_anchor") or "").strip()
            user_intent_anchor_key = str(mapping.get("user_intent_anchor_key") or "").strip()
            assertion_source = str(mapping.get("assertion_source") or "").strip().lower()
            candidate_object_types = list(dict.fromkeys(
                str(value).strip().lower()
                for value in (mapping.get("candidate_object_types") or [])
                if str(value).strip().lower() in {"goal", "plan", "work_item"}
            ))
            if fact_id <= 0 or (
                subject_entity_id <= 0
                or signal_purpose not in {"future_intent", "lifecycle_update"}
                or not candidate_object_types
                or user_role not in {"owner", "participant", "responsible"}
                or not user_intent_anchor
                or not user_intent_anchor_key
                or assertion_source not in {
                    "self_statement", "confirmed_response", "third_party_report",
                    "observed_event",
                }
            ):
                continue
            if not signal_key:
                signal_key = (
                    f"{subject_entity_id}|{signal_purpose}|{user_intent_anchor_key}|"
                    f"{','.join(sorted(candidate_object_types))}"
                )
            self._conn.execute(
                """
                INSERT INTO memory_fact_user_intent_signals (
                    fact_id, signal_key, subject_entity_id, signal_purpose,
                    candidate_object_types, operation_hint, user_role,
                    user_intent_anchor, user_intent_anchor_key,
                    assertion_source, evidence_basis,
                    confidence, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(fact_id, signal_key) DO UPDATE SET
                    candidate_object_types = excluded.candidate_object_types,
                    user_role = excluded.user_role,
                    user_intent_anchor = excluded.user_intent_anchor,
                    assertion_source = excluded.assertion_source,
                    evidence_basis = excluded.evidence_basis,
                    confidence = MAX(
                        memory_fact_user_intent_signals.confidence,
                        excluded.confidence
                    ),
                    updated_at = excluded.updated_at
                """,
                (
                    fact_id, signal_key, subject_entity_id or None, signal_purpose,
                    _json_dumps(candidate_object_types), "", user_role,
                    user_intent_anchor, user_intent_anchor_key,
                    assertion_source,
                    str(mapping.get("evidence_basis") or "").strip(),
                    float(mapping.get("confidence") or 0.0), now, now,
                ),
            )
            changed += 1
        self._commit_if_needed()
        return changed

    def get_fact_user_intent_signals(
        self,
        fact_id: int,
    ) -> Dict[str, Dict[str, Any]]:
        """Load a fact's prospective signals keyed by stable signal identity."""
        if int(fact_id or 0) <= 0:
            return {}
        rows = self._conn.execute(
            """
            SELECT signal.*, entity.name AS subject
            FROM memory_fact_user_intent_signals AS signal
            LEFT JOIN memory_entity_nodes AS entity ON entity.id = signal.subject_entity_id
            WHERE signal.fact_id = ?
              AND signal.subject_entity_id IS NOT NULL
            ORDER BY signal.signal_purpose, signal.user_intent_anchor_key
            """,
            (int(fact_id),),
        ).fetchall()
        mappings: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            mapping = self._row_to_dict(row)
            mapping["candidate_object_types"] = _json_loads(
                mapping.get("candidate_object_types"), []
            )
            mapping["subject_entity"] = str(mapping.get("subject") or "")
            mappings[str(mapping["signal_key"])] = mapping
        return mappings

    def get_user_intent_signals_by_ids(
        self,
        signal_ids: Sequence[int],
    ) -> List[Dict[str, Any]]:
        """Load the complete intent signals selected for one update task."""
        ids = sorted({int(value) for value in signal_ids if int(value) > 0})
        signals: List[Dict[str, Any]] = []
        for offset in range(0, len(ids), 500):
            chunk = ids[offset:offset + 500]
            placeholders = ",".join("?" for _ in chunk)
            rows = self._conn.execute(
                "SELECT signal.*, entity.name AS subject "
                "FROM memory_fact_user_intent_signals AS signal "
                "LEFT JOIN memory_entity_nodes AS entity "
                "ON entity.id = signal.subject_entity_id "
                f"WHERE signal.signal_id IN ({placeholders}) "
                "AND signal.subject_entity_id IS NOT NULL "
                "ORDER BY signal.signal_id",
                chunk,
            ).fetchall()
            for row in rows:
                signal = self._row_to_dict(row)
                signal["candidate_object_types"] = _json_loads(
                    signal.get("candidate_object_types"), []
                )
                signal["subject_entity"] = str(signal.get("subject") or "")
                signals.append(signal)
        return signals

    def upsert_user_intent_lifecycle_observation(
        self, observation: Dict[str, Any],
    ) -> int:
        """Save one extracted observation using its source signal as stable identity."""
        signal_id = int(observation["signal_id"])
        object_type = str(observation["object_type"])
        source = self._conn.execute(
            "SELECT signal_purpose, candidate_object_types "
            "FROM memory_fact_user_intent_signals WHERE signal_id = ?",
            (signal_id,),
        ).fetchone()
        if (not source or source["signal_purpose"] != "lifecycle_update"
                or object_type not in _json_loads(source["candidate_object_types"], [])):
            raise ValueError("Observation must reference a lifecycle signal of the same object type")
        now = local_now_text()
        self._conn.execute(
            """
            INSERT INTO memory_user_intent_lifecycle_observations (
                signal_id, object_type, target_description, observation_text,
                lifecycle_evidence, effective_at, confidence, details,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(signal_id) DO UPDATE SET
                object_type = excluded.object_type,
                target_description = excluded.target_description,
                observation_text = excluded.observation_text,
                lifecycle_evidence = excluded.lifecycle_evidence,
                effective_at = excluded.effective_at,
                confidence = excluded.confidence,
                details = excluded.details,
                updated_at = excluded.updated_at
            """,
            (
                signal_id, object_type, str(observation["target_description"]),
                str(observation["observation_text"]), str(observation["lifecycle_evidence"]),
                str(observation.get("effective_at") or ""),
                float(observation.get("confidence", 0.7)),
                _json_dumps(observation.get("details") or {}), now, now,
            ),
        )
        row = self._conn.execute(
            "SELECT id FROM memory_user_intent_lifecycle_observations WHERE signal_id = ?",
            (signal_id,),
        ).fetchone()
        self._commit_if_needed()
        return int(row["id"])

    def get_user_intent_lifecycle_observations_by_ids(
        self, observation_ids: Sequence[int],
    ) -> List[Dict[str, Any]]:
        """Load observations; source facts remain accessible through their signal IDs."""
        ids = sorted({int(value) for value in observation_ids if int(value) > 0})
        observations: List[Dict[str, Any]] = []
        for offset in range(0, len(ids), 500):
            chunk = ids[offset:offset + 500]
            placeholders = ",".join("?" for _ in chunk)
            rows = self._conn.execute(
                "SELECT * FROM memory_user_intent_lifecycle_observations "
                f"WHERE id IN ({placeholders}) ORDER BY id",
                chunk,
            ).fetchall()
            for row in rows:
                item = dict(row)
                item["details"] = _json_loads(item["details"], {})
                observations.append(item)
        return observations

    def upsert_user_claim(
        self,
        *,
        subject_entity_id: int,
        predicate: str,
        object_entity_id: Optional[int] = None,
        claim_text: str = "",
        claim_type: str,
        claim_origin: str,
        primary_domain: str = "",
        status: str = "candidate",
        confidence: float = 0.7,
        valid_from: str = "",
        valid_to: str = "",
        source_actor_entity_id: Optional[int] = None,
        extractor_version: str = "",
        prompt_version: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> tuple[int, bool]:
        """Insert a semantic claim or refresh its confidence and payload."""
        now = local_now_text()
        object_id = int(object_entity_id or 0)
        subject_id = int(subject_entity_id)
        origin = str(claim_origin or "").strip().lower()
        if origin not in {"explicit", "derived"}:
            raise ValueError("claim_origin must be explicit or derived")
        predicate = str(predicate or "").strip()
        claim_text = re.sub(r"\s+", " ", str(claim_text or "")).strip()
        existing = self._conn.execute(
            """
            SELECT id, confidence, status FROM memory_user_claims
            WHERE subject_entity_id = ? AND predicate = ? AND object_entity_id = ?
              AND claim_text = ? AND claim_type = ? AND claim_origin = ?
            """,
            (subject_id, predicate, object_id, claim_text, claim_type, origin),
        ).fetchone()
        if existing:
            claim_id = int(existing["id"])
            self._conn.execute(
                """
                UPDATE memory_user_claims
                SET claim_text = ?, primary_domain = ?, confidence = ?, valid_from = ?, valid_to = ?,
                    source_actor_entity_id = ?, extractor_version = ?, prompt_version = ?,
                    metadata = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    claim_text, str(primary_domain or ""),
                    max(float(existing["confidence"] or 0.0), float(confidence or 0.0)),
                    valid_from, valid_to, source_actor_entity_id, extractor_version,
                    prompt_version, _json_dumps(metadata or {}), now, claim_id,
                ),
            )
            self._commit_if_needed()
            return claim_id, False
        cur = self._conn.execute(
            """
            INSERT INTO memory_user_claims (
                subject_entity_id, predicate, object_entity_id, claim_text,
                claim_type, claim_origin, primary_domain, status, confidence, valid_from, valid_to,
                source_actor_entity_id, extractor_version, prompt_version, metadata,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                subject_id, predicate, object_id, claim_text,
                claim_type, origin, str(primary_domain or ""), status, float(confidence or 0.0),
                valid_from, valid_to, source_actor_entity_id, extractor_version,
                prompt_version, _json_dumps(metadata or {}), now, now,
            ),
        )
        self._commit_if_needed()
        return int(cur.lastrowid), True

    def upsert_user_observed_pattern(
        self, *, subject_entity_id: int, pattern_key: str, pattern_text: str,
        condition_text: str = "", primary_domain: str = "",
        confidence: float = 0.7, metadata: Optional[Dict[str, Any]] = None,
    ) -> tuple[int, bool]:
        """Create or refresh a repeated-behavior pattern for the user."""
        now = local_now_text()
        subject_id = int(subject_entity_id)
        normalized_key = re.sub(r"\s+", " ", str(pattern_key or "")).strip().lower()
        text = re.sub(r"\s+", " ", str(pattern_text or "")).strip()
        if not normalized_key or not text:
            raise ValueError("pattern_key and pattern_text are required")
        existing = self._conn.execute(
            "SELECT id, confidence, metadata FROM memory_user_observed_patterns "
            "WHERE subject_entity_id = ? AND pattern_key = ?",
            (subject_id, normalized_key),
        ).fetchone()
        merged_metadata = dict(_json_loads(existing["metadata"], {}) or {}) if existing else {}
        merged_metadata.update(metadata or {})
        if existing:
            pattern_id = int(existing["id"])
            self._conn.execute(
                "UPDATE memory_user_observed_patterns SET pattern_text = ?, "
                "condition_text = ?, primary_domain = ?, confidence = ?, "
                "metadata = ?, updated_at = ? WHERE id = ?",
                (
                    text, str(condition_text or ""), str(primary_domain or ""),
                    max(float(existing["confidence"] or 0.0), float(confidence or 0.0)),
                    _json_dumps(merged_metadata), now, pattern_id,
                ),
            )
            self._commit_if_needed()
            return pattern_id, False
        cur = self._conn.execute(
            "INSERT INTO memory_user_observed_patterns ("
            "subject_entity_id, pattern_key, pattern_text, condition_text, "
            "primary_domain, confidence, metadata, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                subject_id, normalized_key, text, str(condition_text or ""),
                str(primary_domain or ""), float(confidence or 0.0),
                _json_dumps(merged_metadata), now, now,
            ),
        )
        self._commit_if_needed()
        return int(cur.lastrowid), True

    def add_user_observed_pattern_signals(
        self, *, pattern_id: int, signal_ids: Sequence[int],
    ) -> int:
        """Attach fact-grounded action signals to an observed pattern."""
        now = local_now_text()
        changed = 0
        for signal_id in sorted({int(value) for value in signal_ids if int(value) > 0}):
            self._conn.execute(
                "INSERT OR IGNORE INTO memory_user_observed_pattern_signals "
                "(pattern_id, signal_id, created_at) VALUES (?, ?, ?)",
                (int(pattern_id), signal_id, now),
            )
            changed += int(self._conn.execute("SELECT changes()").fetchone()[0] > 0)
        self._commit_if_needed()
        return changed

    def refresh_user_observed_pattern(
        self, *, pattern_id: int, confidence: float,
    ) -> None:
        """Keep the established wording when new evidence supports a pattern."""
        self._conn.execute(
            "UPDATE memory_user_observed_patterns SET confidence = MAX(confidence, ?), "
            "updated_at = ? WHERE id = ?",
            (float(confidence), local_now_text(), int(pattern_id)),
        )
        self._commit_if_needed()

    def record_user_observed_pattern_relation(
        self, *, target_pattern_id: int, trigger_pattern_id: Optional[int],
        semantic_relation: str, support_signal_ids: Sequence[int],
        candidate_snapshot: Dict[str, Any], semantic_confidence: float,
        semantic_reason: str,
    ) -> bool:
        """Keep a deduplicated audit event for pattern reconciliation."""
        ids = sorted({int(value) for value in support_signal_ids if int(value) > 0})
        cursor = self._conn.execute(
            "INSERT OR IGNORE INTO memory_user_observed_pattern_events "
            "(target_pattern_id, trigger_pattern_id, semantic_relation, "
            "support_signal_ids, candidate_snapshot, semantic_confidence, "
            "semantic_reason, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                int(target_pattern_id),
                int(trigger_pattern_id) if trigger_pattern_id else None,
                str(semantic_relation), _json_dumps(ids),
                _json_dumps(candidate_snapshot), float(semantic_confidence),
                str(semantic_reason), local_now_text(),
            ),
        )
        self._commit_if_needed()
        return cursor.rowcount > 0

    def get_user_observed_patterns(
        self, *, subject_entity_id: Optional[int] = None,
        limit: Optional[int] = 200,
    ) -> List[Dict[str, Any]]:
        """Read observed patterns together with their supporting signals."""
        where = " WHERE subject_entity_id = ?" if subject_entity_id is not None else ""
        params: List[Any] = (
            [int(subject_entity_id)] if subject_entity_id is not None else []
        )
        limit_sql = ""
        if limit is not None:
            limit_sql = " LIMIT ?"
            params.append(max(1, int(limit)))
        rows = self._conn.execute(
            "SELECT * FROM memory_user_observed_patterns" + where
            + " ORDER BY updated_at DESC, id DESC" + limit_sql, params,
        ).fetchall()
        patterns = [self._row_to_dict(row) for row in rows]
        for pattern in patterns:
            evidence = self._conn.execute(
                "SELECT signal_id FROM memory_user_observed_pattern_signals "
                "WHERE pattern_id = ? ORDER BY signal_id",
                (int(pattern["id"]),),
            ).fetchall()
            pattern["support_signal_ids"] = [
                int(row["signal_id"]) for row in evidence
            ]
        return patterns

    def get_user_observed_patterns_by_ids(
        self, pattern_ids: Sequence[int],
    ) -> List[Dict[str, Any]]:
        ids = sorted({int(value) for value in pattern_ids if int(value) > 0})
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        rows = self._conn.execute(
            f"SELECT * FROM memory_user_observed_patterns WHERE id IN ({placeholders})",
            ids,
        ).fetchall()
        patterns = [self._row_to_dict(row) for row in rows]
        for pattern in patterns:
            evidence = self._conn.execute(
                "SELECT signal_id FROM memory_user_observed_pattern_signals "
                "WHERE pattern_id = ? ORDER BY signal_id",
                (int(pattern["id"]),),
            ).fetchall()
            pattern["support_signal_ids"] = [int(row["signal_id"]) for row in evidence]
        return patterns

    def get_user_observed_pattern_events_for_signal_ids(
        self, signal_ids: Sequence[int],
    ) -> List[Dict[str, Any]]:
        """Read relation events created from a task's action-signal snapshot."""
        selected = {int(value) for value in signal_ids if int(value) > 0}
        if not selected:
            return []
        rows_by_id: Dict[int, sqlite3.Row] = {}
        ids = sorted(selected)
        for offset in range(0, len(ids), 500):
            chunk = ids[offset:offset + 500]
            placeholders = ",".join("?" for _ in chunk)
            rows = self._conn.execute(
                "SELECT DISTINCT event.* FROM memory_user_observed_pattern_events AS event "
                "JOIN json_each(event.support_signal_ids) AS signal "
                f"WHERE CAST(signal.value AS INTEGER) IN ({placeholders})",
                chunk,
            ).fetchall()
            rows_by_id.update({int(row["id"]): row for row in rows})
        events = []
        for row in sorted(rows_by_id.values(), key=lambda item: int(item["id"]), reverse=True):
            event = self._row_to_dict(row)
            event["support_signal_ids"] = _json_loads(
                event.get("support_signal_ids"), [],
            )
            event["candidate_snapshot"] = _json_loads(
                event.get("candidate_snapshot"), {},
            )
            events.append(event)
        return events

    def get_user_observed_pattern_relation_events_by_ids(
        self, event_ids: Sequence[int],
    ) -> List[Dict[str, Any]]:
        """Load exact pattern relation events referenced by a state update task."""
        ids = sorted({int(value) for value in event_ids if int(value) > 0})
        events: List[Dict[str, Any]] = []
        for offset in range(0, len(ids), 500):
            chunk = ids[offset:offset + 500]
            placeholders = ",".join("?" for _ in chunk)
            rows = self._conn.execute(
                "SELECT * FROM memory_user_observed_pattern_events "
                f"WHERE id IN ({placeholders})",
                chunk,
            ).fetchall()
            for row in rows:
                event = self._row_to_dict(row)
                event["support_signal_ids"] = _json_loads(
                    event.get("support_signal_ids"), [],
                )
                event["candidate_snapshot"] = _json_loads(
                    event.get("candidate_snapshot"), {},
                )
                events.append(event)
        return sorted(events, key=lambda event: int(event["id"]))

    def record_user_claim_relation(
        self,
        *,
        target_claim_id: int,
        trigger_claim_id: int,
        semantic_relation: str,
        effective_at: str = "",
        decision_source: str = "",
        semantic_confidence: Optional[float] = None,
        semantic_reason: str = "",
        details: Optional[Dict[str, Any]] = None,
    ) -> Optional[int]:
        """Return the persisted relation event ID without changing claim status."""
        target_id = int(target_claim_id)
        trigger_id = int(trigger_claim_id)
        if target_id <= 0 or trigger_id <= 0 or target_id == trigger_id:
            return None
        relation = str(semantic_relation or "").strip().lower()
        if not relation or relation == "unrelated":
            return None
        occurred_at = str(effective_at or "")
        existing = self._conn.execute(
            """
            SELECT id FROM memory_user_claim_events
            WHERE target_claim_id = ? AND trigger_claim_id = ?
              AND event_type = 'relation_observed' AND semantic_relation = ?
              AND effective_at = ?
            LIMIT 1
            """,
            (target_id, trigger_id, relation, occurred_at),
        ).fetchone()
        if existing:
            return int(existing["id"])
        row = self._conn.execute(
            "SELECT status FROM memory_user_claims WHERE id = ?",
            (target_id,),
        ).fetchone()
        if not row:
            return None
        status = str(row["status"] or "")
        cur = self._conn.execute(
            """
            INSERT INTO memory_user_claim_events (
                target_claim_id, trigger_claim_id, event_type,
                previous_status, new_status, semantic_relation, effective_at,
                decision_source, semantic_confidence, semantic_reason,
                policy_reason, details, created_at
            ) VALUES (?, ?, 'relation_observed', ?, ?, ?, ?, ?, ?, ?, '', ?, ?)
            """,
            (
                target_id,
                trigger_id,
                status,
                status,
                relation,
                occurred_at,
                str(decision_source or ""),
                float(semantic_confidence) if semantic_confidence is not None else None,
                str(semantic_reason or ""),
                _json_dumps(details or {}),
                local_now_text(),
            ),
        )
        self._commit_if_needed()
        return int(cur.lastrowid)

    @staticmethod
    def _intent_object_spec(object_type: str) -> tuple[str, tuple[str, ...]]:
        normalized = str(object_type or "").strip().lower()
        if normalized not in {"goal", "plan", "work_item"}:
            raise ValueError("object_type must be goal, plan, or work_item")
        return "memory_user_intent_items", (
            "object_type", "world_owner_entity_id", "primary_entity_id",
            "canonical_key", "summary", "primary_domain", "status",
            "time_start", "time_end", "confidence", "details", "metadata",
        )

    @staticmethod
    def _intent_payload_to_row(
        object_type: str,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Map the typed manager payload onto the unified physical row."""
        normalized = str(object_type or "").strip().lower()
        entity_field = {
            "goal": "owner_entity_id",
            "plan": "actor_entity_id",
            "work_item": "responsible_entity_id",
        }.get(normalized)
        if not entity_field:
            raise ValueError("object_type must be goal, plan, or work_item")
        details = dict(payload.get("details") or {}) if isinstance(payload.get("details"), dict) else {}
        if normalized == "goal":
            details.update({
                key: payload[key]
                for key in ("desired_outcome", "success_criteria", "target_at")
                if key in payload
            })
            time_start = str(payload.get("time_start") or "")
            time_end = str(payload.get("time_end") or payload.get("target_at") or "")
        elif normalized == "plan":
            details.update({
                key: payload[key]
                for key in (
                    "event_or_activity", "time_precision", "location_entity_id", "location_text",
                )
                if key in payload
            })
            time_start = str(payload.get("time_start") or payload.get("start_at") or "")
            time_end = str(payload.get("time_end") or payload.get("end_at") or "")
        else:
            details.update({
                key: payload[key]
                for key in (
                    "action_text", "deliverable", "responsibility_type", "priority",
                    "completed_at", "extractor_version", "prompt_version",
                )
                if key in payload
            })
            time_start = str(payload.get("time_start") or payload.get("start_at") or "")
            time_end = str(payload.get("time_end") or payload.get("due_at") or "")
        return {
            "object_type": normalized,
            "world_owner_entity_id": payload.get("world_owner_entity_id"),
            "primary_entity_id": payload.get(entity_field),
            "canonical_key": payload.get("canonical_key"),
            "summary": payload.get("summary"),
            "primary_domain": payload.get("primary_domain"),
            "status": payload.get("status"),
            "time_start": time_start,
            "time_end": time_end,
            "confidence": payload.get("confidence"),
            "details": details,
            "metadata": payload.get("metadata"),
        }

    @staticmethod
    def _intent_row_to_api(row: sqlite3.Row | Dict[str, Any]) -> Dict[str, Any]:
        item = dict(row)
        details = _json_loads(item.get("details"), {})
        details = details if isinstance(details, dict) else {}
        metadata = _json_loads(item.get("metadata"), {})
        item["details"] = details
        item["metadata"] = metadata if isinstance(metadata, dict) else {}
        object_type = str(item.get("object_type") or "")
        item["world_owner_entity_id"] = int(item.get("world_owner_entity_id") or 0)
        item["primary_entity_id"] = int(item.get("primary_entity_id") or 0)
        if object_type == "goal":
            item.update(
                owner_entity_id=item["primary_entity_id"],
                desired_outcome=details.get("desired_outcome") or "",
                success_criteria=details.get("success_criteria") or "",
                target_at=details.get("target_at") or item.get("time_end") or "",
            )
        elif object_type == "plan":
            item.update(
                actor_entity_id=item["primary_entity_id"],
                event_or_activity=details.get("event_or_activity") or "",
                start_at=item.get("time_start") or "",
                end_at=item.get("time_end") or "",
                time_precision=details.get("time_precision") or "unknown",
                location_entity_id=details.get("location_entity_id"),
                location_text=details.get("location_text") or "",
            )
        elif object_type == "work_item":
            item.update(
                responsible_entity_id=item["primary_entity_id"],
                action_text=details.get("action_text") or "",
                deliverable=details.get("deliverable") or "",
                responsibility_type=details.get("responsibility_type") or "personal_action",
                due_at=item.get("time_end") or "",
                start_at=item.get("time_start") or "",
                priority=details.get("priority") or "",
                completed_at=details.get("completed_at") or "",
                extractor_version=details.get("extractor_version") or "",
                prompt_version=details.get("prompt_version") or "",
            )
        return item

    def create_intent_object(self, *, object_type: str, payload: Dict[str, Any]) -> int:
        """Persist one normalized Goal, Plan, or Work item candidate."""
        table, columns = self._intent_object_spec(object_type)
        normalized_payload = self._intent_payload_to_row(object_type, payload)
        now = local_now_text()
        values: List[Any] = []
        for column in columns:
            value = normalized_payload.get(column)
            if column in {"metadata", "details"}:
                value = _json_dumps(value if isinstance(value, dict) else {})
            elif column == "confidence":
                value = float(value or 0.0)
            elif column.endswith("_entity_id"):
                value = int(value) if value not in (None, "", 0) else None
            elif column == "object_type":
                value = str(value or object_type)
            else:
                value = str(value or "")
            values.append(value)
        placeholders = ", ".join("?" for _ in columns)
        cur = self._conn.execute(
            f"INSERT INTO {table} ({', '.join(columns)}, created_at, updated_at) "
            f"VALUES ({placeholders}, ?, ?)",
            (*values, now, now),
        )
        self._commit_if_needed()
        return int(cur.lastrowid)

    def update_intent_object(
        self, *, object_type: str, object_id: int, payload: Dict[str, Any]
    ) -> bool:
        """Update a known object without erasing fields absent from an event."""
        table, allowed_columns = self._intent_object_spec(object_type)
        normalized_payload = self._intent_payload_to_row(object_type, payload)
        existing_row = self._conn.execute(
            f"SELECT * FROM {table} WHERE id = ? AND object_type = ?",
            (int(object_id), str(object_type).strip().lower()),
        ).fetchone()
        if not existing_row:
            return False
        existing = dict(existing_row)
        for json_field in ("details", "metadata"):
            existing[json_field] = _json_loads(existing.get(json_field), {})
            if not isinstance(existing[json_field], dict):
                existing[json_field] = {}
        if "details" in payload or any(
            key in payload for key in (
                "desired_outcome", "success_criteria", "target_at", "event_or_activity",
                "time_precision", "location_entity_id", "location_text", "action_text",
                "deliverable", "responsibility_type", "priority", "completed_at",
                "extractor_version", "prompt_version",
            )
        ):
            normalized_payload["details"] = {
                **existing["details"],
                **{
                    key: value
                    for key, value in (normalized_payload.get("details") or {}).items()
                    if value not in ("", None, 0, {}, [])
                },
            }
        if "metadata" in payload:
            normalized_payload["metadata"] = {
                **existing["metadata"],
                **(normalized_payload.get("metadata") or {}),
            }
        else:
            normalized_payload.pop("metadata", None)
        if "details" not in payload and not any(
            key in payload for key in (
                "desired_outcome", "success_criteria", "target_at", "event_or_activity",
                "time_precision", "location_entity_id", "location_text", "action_text",
                "deliverable", "responsibility_type", "priority", "completed_at",
                "extractor_version", "prompt_version",
            )
        ):
            normalized_payload.pop("details", None)
        assignments: List[str] = []
        values: List[Any] = []
        for column in allowed_columns:
            if column not in normalized_payload:
                continue
            value = normalized_payload[column]
            if column in {"metadata", "details"}:
                value = _json_dumps(value if isinstance(value, dict) else {})
            elif column == "confidence":
                value = float(value or 0.0)
            elif column.endswith("_entity_id"):
                value = int(value) if value not in (None, "", 0) else None
            elif column == "object_type":
                value = str(value or object_type)
            else:
                value = str(value or "")
            assignments.append(f"{column} = ?")
            values.append(value)
        if not assignments:
            return False
        assignments.append("updated_at = ?")
        values.extend([local_now_text(), int(object_id)])
        cur = self._conn.execute(
            f"UPDATE {table} SET {', '.join(assignments)} WHERE id = ?", values
        )
        self._commit_if_needed()
        return bool(cur.rowcount)

    def get_intent_objects(
        self,
        *,
        object_type: str,
        world_owner_entity_id: Optional[int] = None,
        primary_domain: Optional[str] = None,
        statuses: Optional[Sequence[str]] = None,
        limit: int = 120,
    ) -> List[Dict[str, Any]]:
        table, _columns = self._intent_object_spec(object_type)
        clauses: List[str] = ["object_type = ?"]
        params: List[Any] = [str(object_type).strip().lower()]
        if world_owner_entity_id is not None:
            clauses.append("world_owner_entity_id = ?")
            params.append(int(world_owner_entity_id))
        if primary_domain is not None:
            clauses.append("primary_domain = ?")
            params.append(str(primary_domain))
        if statuses:
            placeholders = ", ".join("?" for _ in statuses)
            clauses.append(f"status IN ({placeholders})")
            params.extend(str(status) for status in statuses)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        rows = self._conn.execute(
            f"SELECT * FROM {table}{where} ORDER BY updated_at DESC, id DESC LIMIT ?",
            (*params, max(1, int(limit or 120))),
        ).fetchall()
        return [self._intent_row_to_api(row) for row in rows]

    def get_user_intent_items_by_ids(
        self, intent_ids: Sequence[int],
    ) -> List[Dict[str, Any]]:
        """Read typed intent items by their IDs in the unified table."""
        ids = sorted({int(value) for value in intent_ids if int(value) > 0})
        items: List[Dict[str, Any]] = []
        for offset in range(0, len(ids), 500):
            chunk = ids[offset:offset + 500]
            placeholders = ",".join("?" for _ in chunk)
            rows = self._conn.execute(
                f"SELECT * FROM memory_user_intent_items WHERE id IN ({placeholders}) ORDER BY id",
                chunk,
            ).fetchall()
            items.extend(self._intent_row_to_api(row) for row in rows)
        return items

    def insert_user_intent_event(
        self,
        *,
        intent_id: int,
        event_type: str,
        new_payload: Dict[str, Any],
        source_signal_id: int,
        previous_payload: Optional[Dict[str, Any]] = None,
        lifecycle_observation_id: Optional[int] = None,
        effective_at: str = "",
    ) -> int:
        """Record an applied intent mutation; callers share its write transaction."""
        if event_type not in {"created", "updated"}:
            raise ValueError("Intent event must be created or updated")
        if (event_type == "created" and previous_payload) or (
            event_type == "updated" and not previous_payload
        ):
            raise ValueError("Intent event has an invalid previous payload")
        intent = self._conn.execute(
            "SELECT * FROM memory_user_intent_items WHERE id = ?", (int(intent_id),),
        ).fetchone()
        signal = self._conn.execute(
            "SELECT subject_entity_id FROM memory_fact_user_intent_signals WHERE signal_id = ?",
            (int(source_signal_id),),
        ).fetchone()
        if (not intent or not signal
                or signal["subject_entity_id"] != intent["world_owner_entity_id"]):
            raise ValueError("Intent event must reference a same-owner source signal")
        if lifecycle_observation_id is not None:
            observation = self._conn.execute(
                "SELECT signal_id, object_type FROM memory_user_intent_lifecycle_observations "
                "WHERE id = ?", (int(lifecycle_observation_id),),
            ).fetchone()
            if (not observation or observation["signal_id"] != int(source_signal_id)
                    or observation["object_type"] != intent["object_type"]):
                raise ValueError("Intent event observation does not match its source and type")
        # Store physical business fields, excluding timestamps and typed API aliases.
        _, columns = self._intent_object_spec(intent["object_type"])
        current = self._intent_row_to_api(intent)
        previous = {key: previous_payload[key] for key in columns} if previous_payload else {}
        updated = {key: new_payload[key] for key in columns}
        if updated != {key: current[key] for key in columns}:
            raise ValueError("Intent event must describe the persisted intent")
        if event_type == "updated" and previous == updated:
            raise ValueError("Intent event must contain an actual change")
        cursor = self._conn.execute(
            """INSERT INTO memory_user_intent_events (
                intent_id, event_type, previous_payload, new_payload,
                source_signal_id, lifecycle_observation_id, effective_at, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                int(intent_id), event_type, _json_dumps(previous), _json_dumps(updated),
                int(source_signal_id), lifecycle_observation_id,
                str(effective_at or ""), local_now_text(),
            ),
        )
        self._commit_if_needed()
        return int(cursor.lastrowid)

    def get_user_intent_events(
        self, *, event_ids: Optional[Sequence[int]] = None,
        source_signal_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Read applied changes in event order, optionally scoped to a source signal."""
        ids = None if event_ids is None else sorted({
            int(value) for value in event_ids if int(value) > 0
        })
        events: List[Dict[str, Any]] = []
        chunks = [None] if ids is None else [
            ids[offset:offset + 500] for offset in range(0, len(ids), 500)
        ]
        for chunk in chunks:
            clauses: List[str] = []
            params: List[Any] = []
            if chunk is not None:
                clauses.append(f"id IN ({','.join('?' for _ in chunk)})")
                params.extend(chunk)
            if source_signal_id is not None:
                clauses.append("source_signal_id = ?")
                params.append(int(source_signal_id))
            where = " WHERE " + " AND ".join(clauses) if clauses else ""
            rows = self._conn.execute(
                f"SELECT * FROM memory_user_intent_events{where} ORDER BY id", params,
            ).fetchall()
            for row in rows:
                item = dict(row)
                for key in ("previous_payload", "new_payload"):
                    item[key] = _json_loads(item[key], {})
                events.append(item)
        return events

    def get_intent_object(
        self, *, object_type: str, object_id: int
    ) -> Optional[Dict[str, Any]]:
        table, _columns = self._intent_object_spec(object_type)
        row = self._conn.execute(
            f"SELECT * FROM {table} WHERE id = ? AND object_type = ?",
            (int(object_id), str(object_type).strip().lower()),
        ).fetchone()
        return self._intent_row_to_api(row) if row else None

    def get_entity_id_by_name(self, name: str) -> Optional[int]:
        row = self._conn.execute(
            "SELECT id FROM memory_entity_nodes WHERE name = ?", (str(name),)
        ).fetchone()
        return int(row["id"]) if row else None


    def upsert_intent_evidence(self, evidence: Sequence[Dict[str, Any]]) -> int:
        now = local_now_text()
        changed = 0
        for item in evidence or []:
            object_type = str(item.get("object_type") or "").strip().lower()
            evidence_type = str(item.get("evidence_type") or "fact").strip().lower()
            role = str(item.get("role") or "support").strip().lower()
            try:
                object_id = int(item["object_id"])
                evidence_id = int(item["evidence_id"])
            except (KeyError, TypeError, ValueError):
                continue
            if (
                object_type not in {"goal", "plan", "work_item"}
                or evidence_type != "fact"
                or object_id <= 0 or evidence_id <= 0
            ):
                continue
            self._conn.execute(
                """
                INSERT INTO memory_user_intent_evidence (
                    object_type, object_id, evidence_type, evidence_id, role,
                    observed_at, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(object_type, object_id, evidence_type, evidence_id, role)
                DO UPDATE SET observed_at = excluded.observed_at, updated_at = excluded.updated_at
                """,
                (object_type, object_id, evidence_type, evidence_id, role,
                 str(item.get("observed_at") or ""), now, now),
            )
            changed += 1
        self._commit_if_needed()
        return changed

    def get_user_claim_events(
        self,
        claim_id: int,
        *,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Return status-history events for one claim, newest event first."""
        rows = self._conn.execute(
            """
            SELECT * FROM memory_user_claim_events
            WHERE target_claim_id = ?
            ORDER BY effective_at DESC, id DESC
            LIMIT ?
            """,
            (int(claim_id), max(1, int(limit or 100))),
        ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def get_user_claim_events_for_claim_ids(
        self,
        claim_ids: Sequence[int],
        *,
        limit: int = 400,
    ) -> Dict[int, List[Dict[str, Any]]]:
        """Return transition history grouped by target claim for state resolution."""
        ids = list(dict.fromkeys(
            int(value)
            for value in claim_ids or []
            if str(value).strip().isdigit() and int(value) > 0
        ))
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        rows = self._conn.execute(
            f"""
            SELECT * FROM memory_user_claim_events
            WHERE target_claim_id IN ({placeholders})
            ORDER BY target_claim_id ASC, effective_at DESC, id DESC
            LIMIT ?
            """,
            (*ids, max(1, int(limit or 400))),
        ).fetchall()
        result: Dict[int, List[Dict[str, Any]]] = {claim_id: [] for claim_id in ids}
        for row in rows:
            event = self._row_to_dict(row)
            result.setdefault(int(event["target_claim_id"]), []).append(event)
        return result

    def get_user_claim_relation_events_by_ids(
        self, event_ids: Sequence[int],
    ) -> List[Dict[str, Any]]:
        """Load the exact relation events referenced by a state-update batch."""
        ids = list(dict.fromkeys(
            int(value) for value in event_ids or []
            if str(value).strip().isdigit() and int(value) > 0
        ))
        if not ids:
            return []
        rows_by_id: Dict[int, Dict[str, Any]] = {}
        for start in range(0, len(ids), 400):
            chunk = ids[start:start + 400]
            placeholders = ",".join("?" for _ in chunk)
            rows = self._conn.execute(
                f"SELECT * FROM memory_user_claim_events "
                f"WHERE event_type = 'relation_observed' AND id IN ({placeholders})",
                chunk,
            ).fetchall()
            rows_by_id.update(
                (int(row["id"]), self._row_to_dict(row)) for row in rows
            )
        return [rows_by_id[event_id] for event_id in ids if event_id in rows_by_id]

    def apply_user_claim_status_decision(
        self, *, claim_id: int, new_status: str, trigger_claim_id: int,
        relation_event_id: int, effective_at: str, reason: str,
        decision_source: str,
    ) -> bool:
        """Change a claim's current status and retain the resolver decision."""
        row = self._conn.execute(
            "SELECT status FROM memory_user_claims WHERE id = ?", (int(claim_id),)
        ).fetchone()
        previous_status = str(row["status"] or "") if row else ""
        if (
            new_status not in {"weakened", "superseded"}
            or previous_status not in {"active", "weakened"}
            or previous_status == new_status
            or (new_status == "weakened" and previous_status != "active")
        ):
            return False
        self._conn.execute(
            "UPDATE memory_user_claims SET status = ?, updated_at = ? WHERE id = ?",
            (str(new_status), local_now_text(), int(claim_id)),
        )
        self._conn.execute(
            """
            INSERT INTO memory_user_claim_events (
                target_claim_id, trigger_claim_id, event_type,
                previous_status, new_status, semantic_relation, effective_at,
                decision_source, semantic_reason, policy_reason, details, created_at
            ) VALUES (?, ?, 'status_transition', ?, ?, '', ?, ?, '', ?, ?, ?)
            """,
            (
                int(claim_id), int(trigger_claim_id), previous_status,
                str(new_status), str(effective_at or ""),
                str(decision_source or ""), str(reason or ""),
                _json_dumps({"relation_event_id": int(relation_event_id)}),
                local_now_text(),
            ),
        )
        self._conn.execute(
            """
            UPDATE memory_recall_documents
            SET status = ?, updated_at = ?
            WHERE object_type = 'user_claim' AND object_id = ?
            """,
            (str(new_status), local_now_text(), int(claim_id)),
        )
        self._commit_if_needed()
        return True

    def get_user_claims(
        self,
        *,
        entity_id: Optional[int] = None,
        subject_entity_id: Optional[int] = None,
        primary_domain: Optional[str] = None,
        claim_type: Optional[str] = None,
        claim_origin: Optional[str] = None,
        predicate: Optional[str] = None,
        statuses: Optional[Sequence[str]] = None,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        clauses: List[str] = []
        params: List[Any] = []
        if entity_id is not None:
            clauses.append("(subject_entity_id = ? OR object_entity_id = ?)")
            params.extend((int(entity_id), int(entity_id)))
        for column, value in (
            ("subject_entity_id", subject_entity_id),
            ("primary_domain", primary_domain),
            ("claim_type", claim_type),
            ("claim_origin", claim_origin),
            ("predicate", predicate),
        ):
            if value is not None:
                clauses.append(f"{column} = ?")
                params.append(value)
        if statuses:
            placeholders = ",".join("?" for _ in statuses)
            clauses.append(f"status IN ({placeholders})")
            params.extend(str(value) for value in statuses)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        rows = self._conn.execute(
            f"SELECT * FROM memory_user_claims{where} ORDER BY updated_at DESC, id DESC LIMIT ?",
            (*params, max(1, int(limit or 200))),
        ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def get_user_claims_by_ids(
        self,
        claim_ids: Sequence[int],
    ) -> List[Dict[str, Any]]:
        """Load claims in caller-specified order for projection refreshes."""
        ids = list(dict.fromkeys(
            int(value)
            for value in claim_ids or []
            if str(value).strip().isdigit() and int(value) > 0
        ))
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        rows = self._conn.execute(
            f"SELECT * FROM memory_user_claims WHERE id IN ({placeholders})",
            ids,
        ).fetchall()
        by_id = {int(row["id"]): self._row_to_dict(row) for row in rows}
        return [by_id[claim_id] for claim_id in ids if claim_id in by_id]


    def get_entity_names_by_ids(
        self,
        entity_ids: Sequence[int],
    ) -> Dict[int, str]:
        """Return stable entity display names for recall-document projection."""
        ids = list(dict.fromkeys(
            int(value)
            for value in entity_ids or []
            if str(value).strip().isdigit() and int(value) > 0
        ))
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        rows = self._conn.execute(
            f"SELECT id, name FROM memory_entity_nodes WHERE id IN ({placeholders})",
            ids,
        ).fetchall()
        return {
            int(row["id"]): str(row["name"] or "")
            for row in rows
            if str(row["name"] or "").strip()
        }

    def memory_recall_documents_with_identity_embeddings(
        self,
        *,
        object_type: str,
        statuses: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Load embeddable recall projections for one direct object type.

        Callers own type-specific status and temporal policies.  Keeping this
        method projection-only lets Stage 2 use the same full-embedding
        retrieval source for facts and all derived memory objects.
        """
        normalized_type = str(object_type or "").strip().lower()
        if not normalized_type:
            return []
        clauses = [
            "object_type = ?",
            "identity_text_embedding IS NOT NULL",
        ]
        params: List[Any] = [normalized_type]
        if statuses:
            normalized_statuses = [
                str(value).strip()
                for value in statuses
                if str(value).strip()
            ]
            if not normalized_statuses:
                return []
            placeholders = ",".join("?" for _ in normalized_statuses)
            clauses.append(f"status IN ({placeholders})")
            params.extend(normalized_statuses)
        rows = self._conn.execute(
            f"""
            SELECT *
            FROM memory_recall_documents
            WHERE {' AND '.join(clauses)}
            ORDER BY updated_at DESC, id DESC
            """,
            params,
        ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def search_memory_recall_documents(
        self,
        *,
        object_type: str,
        terms: Optional[Sequence[str]] = None,
        statuses: Optional[Sequence[str]] = None,
        time_start: Optional[str] = None,
        time_end: Optional[str] = None,
        limit: int = 120,
        strict_time_filter: bool = False,
    ) -> List[Dict[str, Any]]:
        """Lexically retrieve one object type from the recall projection.

        This method does not impose product retrieval policy. A requested time range first retrieves
        documents whose stored interval overlaps the range.  When
        ``strict_time_filter`` is false, a bounded lexical fallback fills a
        short result set; fallback rows are annotated so callers can apply a
        time penalty during ranking instead of silently treating them as exact
        temporal matches.
        """
        normalized_type = str(object_type or "").strip().lower()
        if not normalized_type:
            return []
        row_limit = max(1, int(limit or 120))
        clauses = ["document.object_type = ?"]
        params: List[Any] = [normalized_type]
        if statuses:
            normalized_statuses = [
                str(value).strip()
                for value in statuses
                if str(value).strip()
            ]
            if not normalized_statuses:
                return []
            placeholders = ",".join("?" for _ in normalized_statuses)
            clauses.append(f"document.status IN ({placeholders})")
            params.extend(normalized_statuses)
        base_where = " AND ".join(clauses)
        normalized_time_start = str(time_start or "").strip()
        normalized_time_end = str(time_end or "").strip()
        time_clauses: List[str] = []
        time_params: List[Any] = []
        # A recall document represents either a point-in-time item or a time
        # interval.  Treat missing time as unknown rather than as a match.
        # This keeps a time-bounded first pass precise; the optional fallback
        # below is the explicit recovery path for coarse or missing metadata.
        if normalized_time_start:
            time_clauses.append(
                "COALESCE(NULLIF(document.time_end, ''), "
                "NULLIF(document.time_start, '')) >= ?"
            )
            time_params.append(normalized_time_start)
        if normalized_time_end:
            time_clauses.append(
                "COALESCE(NULLIF(document.time_start, ''), "
                "NULLIF(document.time_end, '')) <= ?"
            )
            time_params.append(normalized_time_end)
        timed_where = " AND ".join([base_where, *time_clauses])
        has_time_filter = bool(time_clauses)
        normalized_terms = self._normalize_search_terms(terms)
        fts_table = _IDENTITY_FTS_TABLES["memory_recall_documents"]

        def query_rows(
            *,
            where: str,
            query_params: Sequence[Any],
            query_limit: int,
        ) -> List[sqlite3.Row]:
            if normalized_terms:
                match_query = self._terms_to_fts_query(normalized_terms)
                if match_query:
                    try:
                        return self._conn.execute(
                            f"""
                            SELECT document.*, bm25({fts_table}) AS bm25_score
                            FROM {fts_table}
                            JOIN memory_recall_documents AS document
                                ON document.id = {fts_table}.rowid
                            WHERE {where} AND {fts_table} MATCH ?
                            ORDER BY bm25({fts_table}) ASC,
                                     document.updated_at DESC, document.id DESC
                            LIMIT ?
                            """,
                            (*query_params, match_query, query_limit),
                        ).fetchall()
                    except sqlite3.Error:
                        pass
                like_clauses = [
                    "LOWER(COALESCE(document.identity_text, '')) LIKE ?"
                    for _term in normalized_terms[:12]
                ]
                if like_clauses:
                    return self._conn.execute(
                        f"""
                        SELECT document.*
                        FROM memory_recall_documents AS document
                        WHERE {where} AND ({" OR ".join(like_clauses)})
                        ORDER BY document.updated_at DESC, document.id DESC
                        LIMIT ?
                        """,
                        (
                            *query_params,
                            *[f"%{term}%" for term in normalized_terms[:12]],
                            query_limit,
                        ),
                    ).fetchall()
            return self._conn.execute(
                f"""
                SELECT document.*
                FROM memory_recall_documents AS document
                WHERE {where}
                ORDER BY NULLIF(document.time_end, '') DESC,
                         document.updated_at DESC, document.id DESC
                LIMIT ?
                """,
                (*query_params, query_limit),
            ).fetchall()

        ranked_rows: List[Tuple[sqlite3.Row, str]] = [
            (row, "strict")
            for row in query_rows(
                where=timed_where,
                query_params=[*params, *time_params],
                query_limit=row_limit,
            )
        ]
        if has_time_filter and not strict_time_filter and len(ranked_rows) < row_limit:
            seen_ids = {int(row["id"]) for row, _match in ranked_rows}
            fallback_rows = query_rows(
                where=base_where,
                query_params=params,
                query_limit=row_limit,
            )
            ranked_rows.extend(
                (row, "fallback")
                for row in fallback_rows
                if int(row["id"]) not in seen_ids
            )

        result: List[Dict[str, Any]] = []
        for row, temporal_match in ranked_rows[:row_limit]:
            item = self._row_to_dict(row)
            if "bm25_score" in row.keys():
                item["_bm25_score"] = row["bm25_score"]
            if has_time_filter:
                item["_recall_temporal_match"] = temporal_match
            result.append(item)
        return result

    def get_user_claim_evidence_fact_ids(
        self,
        claim_ids: Sequence[int],
        *,
        limit: int = 120,
    ) -> Dict[int, List[int]]:
        """Return support-fact IDs for bounded claim candidates."""
        ids = list(dict.fromkeys(
            int(value)
            for value in claim_ids or []
            if str(value).strip().isdigit() and int(value) > 0
        ))
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        rows = self._conn.execute(
            f"""
            SELECT claim_id, evidence_id
            FROM memory_user_claim_evidence
            WHERE claim_id IN ({placeholders})
              AND evidence_type = 'fact'
              AND role = 'support'
            ORDER BY claim_id ASC, weight DESC, observed_at DESC, evidence_id DESC
            LIMIT ?
            """,
            (*ids, max(1, int(limit or 120))),
        ).fetchall()
        result: Dict[int, List[int]] = {claim_id: [] for claim_id in ids}
        for row in rows:
            claim_id = int(row["claim_id"])
            fact_id = int(row["evidence_id"])
            if fact_id not in result.setdefault(claim_id, []):
                result[claim_id].append(fact_id)
        return result

    def get_intent_evidence_fact_ids(
        self,
        objects: Sequence[Tuple[str, int]],
        *,
        limit: int = 120,
    ) -> Dict[Tuple[str, int], List[int]]:
        """Return fact evidence for bounded goal, plan, and work-item rows."""
        normalized: List[Tuple[str, int]] = []
        seen: set[Tuple[str, int]] = set()
        for object_type, object_id in objects or []:
            normalized_type = str(object_type or "").strip().lower()
            if normalized_type not in {"goal", "plan", "work_item"}:
                continue
            try:
                normalized_id = int(object_id)
            except (TypeError, ValueError):
                continue
            key = (normalized_type, normalized_id)
            if normalized_id <= 0 or key in seen:
                continue
            seen.add(key)
            normalized.append(key)
        if not normalized:
            return {}
        object_clauses = " OR ".join(
            "(object_type = ? AND object_id = ?)" for _item in normalized
        )
        params: List[Any] = [
            value
            for object_type, object_id in normalized
            for value in (object_type, object_id)
        ]
        rows = self._conn.execute(
            f"""
            SELECT object_type, object_id, evidence_id
            FROM memory_user_intent_evidence
            WHERE evidence_type = 'fact'
              AND ({object_clauses})
            ORDER BY object_type ASC, object_id ASC, observed_at DESC, evidence_id DESC
            LIMIT ?
            """,
            (*params, max(1, int(limit or 120))),
        ).fetchall()
        result: Dict[Tuple[str, int], List[int]] = {
            item: [] for item in normalized
        }
        for row in rows:
            key = (str(row["object_type"]), int(row["object_id"]))
            fact_id = int(row["evidence_id"])
            if fact_id not in result.setdefault(key, []):
                result[key].append(fact_id)
        return result

    def upsert_user_claim_evidence(self, evidence: Sequence[Dict[str, Any]]) -> int:
        now = local_now_text()
        changed = 0
        for item in evidence or []:
            try:
                claim_id = int(item["claim_id"])
                evidence_id = int(item["evidence_id"])
            except (KeyError, TypeError, ValueError):
                continue
            evidence_type = str(item.get("evidence_type") or "fact")
            role = str(item.get("role") or "support")
            if claim_id <= 0 or evidence_id <= 0 or evidence_type != "fact":
                continue
            self._conn.execute(
                """
                INSERT INTO memory_user_claim_evidence (
                    claim_id, evidence_type, evidence_id, role, weight, observed_at,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(claim_id, evidence_type, evidence_id, role) DO UPDATE SET
                    weight = MAX(memory_user_claim_evidence.weight, excluded.weight),
                    observed_at = excluded.observed_at, updated_at = excluded.updated_at
                """,
                (claim_id, evidence_type, evidence_id, role, float(item.get("weight") or 1.0),
                 str(item.get("observed_at") or ""), now, now),
            )
            changed += 1
        self._commit_if_needed()
        return changed

    def upsert_user_claim_derivation(
        self,
        *,
        claim_id: int,
        rule_id: str,
        rule_version: str,
        derivation_key: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Create or reactivate one auditable proof for a derived claim."""
        now = local_now_text()
        self._conn.execute(
            """
            INSERT INTO memory_user_claim_derivations (
                claim_id, rule_id, rule_version, derivation_key, status,
                derived_at, invalidated_at, metadata, created_at, updated_at
            ) VALUES (?, ?, ?, ?, 'active', ?, '', ?, ?, ?)
            ON CONFLICT(derivation_key) DO UPDATE SET
                claim_id = excluded.claim_id,
                rule_id = excluded.rule_id,
                rule_version = excluded.rule_version,
                status = 'active',
                derived_at = excluded.derived_at,
                invalidated_at = '',
                metadata = excluded.metadata,
                updated_at = excluded.updated_at
            """,
            (
                int(claim_id), str(rule_id or ""), str(rule_version or ""),
                str(derivation_key or ""), now, _json_dumps(metadata or {}),
                now, now,
            ),
        )
        row = self._conn.execute(
            "SELECT id FROM memory_user_claim_derivations WHERE derivation_key = ?",
            (str(derivation_key or ""),),
        ).fetchone()
        derivation_id = int(row["id"]) if row else 0
        self._commit_if_needed()
        return derivation_id

    def replace_user_claim_derivation_premises(
        self,
        *,
        derivation_id: int,
        premise_claim_ids: Sequence[int],
        premise_role: str = "support",
    ) -> int:
        """Replace one derivation's exact premise set atomically."""
        normalized = list(dict.fromkeys(
            int(value)
            for value in premise_claim_ids or []
            if str(value).strip().isdigit() and int(value) > 0
        ))
        if int(derivation_id or 0) <= 0:
            return 0
        now = local_now_text()
        self._conn.execute(
            "DELETE FROM memory_user_claim_premises WHERE derivation_id = ?",
            (int(derivation_id),),
        )
        self._conn.executemany(
            """
            INSERT INTO memory_user_claim_premises (
                derivation_id, premise_claim_id, premise_role, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            [
                (int(derivation_id), premise_id, str(premise_role or "support"), now, now)
                for premise_id in normalized
            ],
        )
        self._commit_if_needed()
        return len(normalized)

    def invalidate_user_claim_derivations_for_premises(
        self,
        premise_claim_ids: Sequence[int],
    ) -> List[int]:
        """Invalidate proofs whose changed premise is no longer active.

        The caller owns the consequential derived-claim status transition, since
        it needs to preserve the existing claim-event audit trail.
        """
        premise_ids = list(dict.fromkeys(
            int(value)
            for value in premise_claim_ids or []
            if str(value).strip().isdigit() and int(value) > 0
        ))
        if not premise_ids:
            return []
        placeholders = ",".join("?" for _ in premise_ids)
        rows = self._conn.execute(
            f"""
            SELECT DISTINCT derivation.id, derivation.claim_id
            FROM memory_user_claim_derivations AS derivation
            JOIN memory_user_claim_premises AS premise
              ON premise.derivation_id = derivation.id
            JOIN memory_user_claims AS premise_claim
              ON premise_claim.id = premise.premise_claim_id
            WHERE derivation.status = 'active'
              AND premise.premise_claim_id IN ({placeholders})
              AND premise_claim.status != 'active'
            """,
            premise_ids,
        ).fetchall()
        if not rows:
            return []
        now = local_now_text()
        derivation_ids = [int(row["id"]) for row in rows]
        derivation_placeholders = ",".join("?" for _ in derivation_ids)
        self._conn.execute(
            f"""
            UPDATE memory_user_claim_derivations
            SET status = 'invalidated', invalidated_at = ?, updated_at = ?
            WHERE id IN ({derivation_placeholders})
            """,
            (now, now, *derivation_ids),
        )
        self._commit_if_needed()
        return list(dict.fromkeys(int(row["claim_id"]) for row in rows))

    def get_user_claim_ids_without_active_derivations(
        self,
        claim_ids: Sequence[int],
    ) -> List[int]:
        """Return derived-claim IDs for which every stored proof is inactive."""
        ids = list(dict.fromkeys(
            int(value)
            for value in claim_ids or []
            if str(value).strip().isdigit() and int(value) > 0
        ))
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        rows = self._conn.execute(
            f"""
            SELECT claim.id
            FROM memory_user_claims AS claim
            WHERE claim.id IN ({placeholders})
              AND claim.claim_origin = 'derived'
              AND claim.status = 'active'
              AND NOT EXISTS (
                  SELECT 1
                  FROM memory_user_claim_derivations AS derivation
                  WHERE derivation.claim_id = claim.id
                    AND derivation.status = 'active'
              )
            """,
            ids,
        ).fetchall()
        return [int(row["id"]) for row in rows]

    def insert_fact(
        self,
        *,
        fact_type: str,
        summary: str,
        keywords: Sequence[str],
        entities: Sequence[str],
        entity_ids: Optional[Sequence[int]],
        fact_topic: str,
        event_time_key: str,
        dialogue_time_key: str,
        confidence: float,
        importance: float,
        metadata: Optional[Dict[str, Any]],
    ) -> int:
        now = local_now_text()
        keyword_values = (
            [
                value
                for value in str(keywords).split()
                if value
            ]
            if isinstance(keywords, str)
            else [
                str(value).strip()
                for value in keywords or []
                if str(value).strip()
            ]
        )
        cur = self._conn.execute(
            """
            INSERT INTO memory_facts (
                fact_type,
                summary, keywords, entities, entity_ids, fact_topic,
                event_time_key, dialogue_time_key,
                confidence, importance, metadata, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                fact_type,
                summary,
                _json_dumps(keyword_values),
                _json_dumps(list(entities or [])),
                _json_dumps([int(value) for value in entity_ids or []]),
                str(fact_topic or ""),
                event_time_key,
                dialogue_time_key,
                float(confidence),
                float(importance),
                _json_dumps(metadata or {}),
                now,
                now,
            ),
        )
        fact_id = int(cur.lastrowid)
        self.insert_entity_memory_mappings([
            {
                "entity_id": int(entity_id),
                "fact_id": [fact_id],
            }
            for entity_id in entity_ids or []
        ])
        self._commit_if_needed()
        return fact_id

    def upsert_entity_nodes(
        self,
        entities: Iterable[Any],
    ) -> Dict[str, int]:
        """Create entity nodes and conservatively promote their node types.

        ``memory_entity_nodes.name`` remains the current identity key.  A
        concrete type supplied by fact extraction may upgrade a legacy
        ``OTHER`` node, but never replaces a previously established concrete
        type.  That avoids a later weak observation silently relabelling an
        entity while still allowing the fact-store path to repair old nodes.
        """
        now = local_now_text()
        allowed_types = {
            "PERSON", "ORGANIZATION", "LOCATION", "PRODUCT", "PROJECT",
            "TECHNOLOGY", "CONCEPT", "TOPIC", "PREFERENCE", "OTHER",
        }
        normalized: Dict[str, str] = {}
        for entity in entities:
            if isinstance(entity, dict):
                clean = str(entity.get("name") or entity.get("text") or "").strip()
                entity_type = str(entity.get("type") or "OTHER").strip().upper()
            else:
                clean = str(entity or "").strip()
                entity_type = "OTHER"
            if not clean:
                continue
            if entity_type not in allowed_types:
                entity_type = "OTHER"
            previous_type = normalized.get(clean)
            if previous_type is not None:
                if previous_type == "OTHER" and entity_type != "OTHER":
                    normalized[clean] = entity_type
                continue
            self._conn.execute(
                "INSERT OR IGNORE INTO memory_entity_nodes (name, type, created_at) VALUES (?, ?, ?)",
                (clean, entity_type, now),
            )
            normalized[clean] = entity_type
        for name, entity_type in normalized.items():
            if entity_type == "OTHER":
                continue
            self._conn.execute(
                "UPDATE memory_entity_nodes SET type = ? WHERE name = ? AND type = 'OTHER'",
                (entity_type, name),
            )
        self._commit_if_needed()
        if not normalized:
            return {}
        placeholders = ",".join("?" for _ in normalized)
        rows = self._conn.execute(
            f"SELECT id, name FROM memory_entity_nodes WHERE name IN ({placeholders})",
            list(normalized),
        ).fetchall()
        return {str(row["name"]): int(row["id"]) for row in rows}

    def add_entity_names(self, names: Iterable[str]) -> Dict[str, int]:
        """Compatibility wrapper for legacy name-only callers."""
        return self.upsert_entity_nodes(names)

    @staticmethod
    def normalize_owner_alias_key(value: str) -> str:
        return re.sub(r"\s+", "", unicodedata.normalize("NFKC", value).casefold())

    def get_owner_entity_aliases(
        self,
        owner_entity_id: int,
        *,
        status: Optional[str] = None,
        limit: int = 32,
    ) -> List[Dict[str, Any]]:
        """Read known names for the database's canonical world owner."""
        clauses = ["owner_entity_id = ?"]
        params: List[Any] = [int(owner_entity_id)]
        if status:
            clauses.append("status = ?")
            params.append(str(status))
        rows = self._conn.execute(
            f"SELECT * FROM memory_owner_entity_aliases WHERE {' AND '.join(clauses)} "
            "ORDER BY confidence DESC, updated_at DESC, id DESC LIMIT ?",
            (*params, max(1, int(limit))),
        ).fetchall()
        return [dict(row) for row in rows]

    def upsert_owner_entity_alias(
        self,
        *,
        owner_entity_id: int,
        alias_name: str,
        status: str,
        confidence: float,
        evidence_fact_id: int,
        evidence_type: str,
        evidence_text: str,
        source_speaker: str,
    ) -> int:
        """Keep the strongest observed evidence without downgrading a confirmed alias."""
        alias_name = str(alias_name or "").strip()
        alias_key = self.normalize_owner_alias_key(alias_name)
        if not alias_key or status not in {"candidate", "confirmed"}:
            raise ValueError("invalid owner alias")
        now = local_now_text()
        self._conn.execute(
            """
            INSERT INTO memory_owner_entity_aliases (
                owner_entity_id, alias_name, alias_key, status, confidence,
                evidence_fact_id, evidence_type, evidence_text, source_speaker,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(owner_entity_id, alias_key) DO UPDATE SET
                status = CASE
                    WHEN excluded.status = 'confirmed' THEN 'confirmed'
                    ELSE memory_owner_entity_aliases.status
                END,
                alias_name = CASE
                    WHEN (excluded.status = 'confirmed' AND memory_owner_entity_aliases.status != 'confirmed')
                      OR excluded.confidence > memory_owner_entity_aliases.confidence
                    THEN excluded.alias_name ELSE memory_owner_entity_aliases.alias_name
                END,
                evidence_fact_id = CASE
                    WHEN (excluded.status = 'confirmed' AND memory_owner_entity_aliases.status != 'confirmed')
                      OR excluded.confidence > memory_owner_entity_aliases.confidence
                    THEN excluded.evidence_fact_id ELSE memory_owner_entity_aliases.evidence_fact_id
                END,
                evidence_type = CASE
                    WHEN (excluded.status = 'confirmed' AND memory_owner_entity_aliases.status != 'confirmed')
                      OR excluded.confidence > memory_owner_entity_aliases.confidence
                    THEN excluded.evidence_type ELSE memory_owner_entity_aliases.evidence_type
                END,
                evidence_text = CASE
                    WHEN (excluded.status = 'confirmed' AND memory_owner_entity_aliases.status != 'confirmed')
                      OR excluded.confidence > memory_owner_entity_aliases.confidence
                    THEN excluded.evidence_text ELSE memory_owner_entity_aliases.evidence_text
                END,
                source_speaker = CASE
                    WHEN (excluded.status = 'confirmed' AND memory_owner_entity_aliases.status != 'confirmed')
                      OR excluded.confidence > memory_owner_entity_aliases.confidence
                    THEN excluded.source_speaker ELSE memory_owner_entity_aliases.source_speaker
                END,
                confidence = CASE
                    WHEN excluded.status = 'confirmed' AND memory_owner_entity_aliases.status != 'confirmed'
                    THEN excluded.confidence
                    ELSE MAX(memory_owner_entity_aliases.confidence, excluded.confidence)
                END,
                updated_at = excluded.updated_at
            """,
            (
                int(owner_entity_id), alias_name, alias_key, status,
                max(0.0, min(1.0, float(confidence))), int(evidence_fact_id),
                str(evidence_type), str(evidence_text), str(source_speaker), now, now,
            ),
        )
        self._commit_if_needed()
        row = self._conn.execute(
            "SELECT id FROM memory_owner_entity_aliases "
            "WHERE owner_entity_id = ? AND alias_key = ?",
            (int(owner_entity_id), alias_key),
        ).fetchone()
        return int(row["id"])

    def find_entity_nodes_in_text(
        self,
        text: str,
        *,
        limit: int = 12,
    ) -> List[Dict[str, Any]]:
        """Find stored entity names occurring verbatim in a query text."""
        clean_text = str(text or "").strip()
        if not clean_text:
            return []
        rows = self._conn.execute(
            """
            SELECT id, name
            FROM memory_entity_nodes
            WHERE length(name) >= 2
              AND instr(lower(?), lower(name)) > 0
            ORDER BY length(name) DESC, id ASC
            LIMIT ?
            """,
            (clean_text, max(1, int(limit or 12))),
        ).fetchall()
        return [dict(row) for row in rows]

    def memory_entity_mappings_by_entity_ids(
        self,
        entity_ids: Sequence[int],
    ) -> List[Dict[str, Any]]:
        """Load mapping rows for a bounded set of entity IDs."""
        ids = list(dict.fromkeys(
            int(value)
            for value in entity_ids
            if value is not None
        ))
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        rows = self._conn.execute(
            f"""
            SELECT entity_id, fact_id
            FROM memory_entity_mapping
            WHERE entity_id IN ({placeholders})
            """,
            ids,
        ).fetchall()
        return [
            {
                "entity_id": int(row["entity_id"]),
                "fact_id": _json_loads(row["fact_id"], []),
            }
            for row in rows
        ]

    def insert_entity_memory_mappings(
        self,
        mappings: Sequence[Dict[str, Any]],
    ) -> int:
        """Merge fact links into one row per entity."""
        if not mappings:
            return 0

        mapping_fields = (
            "fact_id",
        )

        def normalize_ids(value: Any) -> List[int]:
            if isinstance(value, str):
                parsed = _json_loads(value, None)
                values = parsed if isinstance(parsed, list) else [value]
            else:
                values = value if isinstance(value, (list, tuple, set)) else [value]
            normalized: List[int] = []
            for item in values:
                if item in (None, ""):
                    continue
                try:
                    item_id = int(item)
                except (TypeError, ValueError):
                    continue
                if item_id not in normalized:
                    normalized.append(item_id)
            return normalized

        grouped: Dict[int, Dict[str, List[int]]] = {}
        for mapping in mappings:
            try:
                entity_id = int(mapping["entity_id"])
            except (KeyError, TypeError, ValueError):
                continue
            entity_mapping = grouped.setdefault(
                entity_id,
                {field: [] for field in mapping_fields},
            )
            for field in mapping_fields:
                for item_id in normalize_ids(mapping.get(field)):
                    if item_id not in entity_mapping[field]:
                        entity_mapping[field].append(item_id)

        if not grouped:
            return 0

        now = local_now_text()
        changed_count = 0
        for entity_id, mapping in grouped.items():
            existing = self._conn.execute(
                """
                SELECT fact_id
                FROM memory_entity_mapping
                WHERE entity_id = ?
                """,
                (entity_id,),
            ).fetchone()
            merged = dict(mapping)
            if existing:
                for field in mapping_fields:
                    previous_ids = normalize_ids(existing[field])
                    merged[field] = previous_ids + [
                        item_id
                        for item_id in mapping[field]
                        if item_id not in previous_ids
                    ]
                self._conn.execute(
                    """
                    UPDATE memory_entity_mapping
                    SET fact_id = ?, updated_at = ?
                    WHERE entity_id = ?
                    """,
                    (
                        _json_dumps(merged["fact_id"]),
                        now,
                        entity_id,
                    ),
                )
            else:
                self._conn.execute(
                    """
                    INSERT INTO memory_entity_mapping (
                        entity_id, fact_id, created_at, updated_at
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (
                        entity_id,
                        _json_dumps(merged["fact_id"]),
                        now,
                        now,
                    ),
                )
            changed_count += 1
        self._commit_if_needed()
        return changed_count
    
    def get_memory_facts_by_ids(self, fact_ids: Sequence[int]) -> List[Dict[str, Any]]:
        ids = [int(value) for value in fact_ids if value is not None]
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        rows = self._conn.execute(
            f"""
            SELECT fact.*, document.identity_text,
                   document.identity_text_embedding
            FROM memory_facts AS fact
            LEFT JOIN memory_recall_documents AS document
                ON document.object_type = 'fact' AND document.object_id = fact.id
            WHERE fact.id IN ({placeholders})
            """,
            ids,
        ).fetchall()
        by_id = {int(row["id"]): self._row_to_dict(row) for row in rows}
        return [by_id[item] for item in ids if item in by_id]

    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        item = dict(row)
        if "keywords" in item:
            raw_keywords = item["keywords"]
            parsed_keywords = _json_loads(raw_keywords, None)
            if isinstance(parsed_keywords, list):
                item["keywords"] = [
                    str(value).strip()
                    for value in parsed_keywords
                    if str(value).strip()
                ]
            else:
                # Facts stored before keyword lists were serialized used a
                # whitespace-joined string. Their original phrase boundaries
                # cannot be recovered, so retain the former token behavior.
                item["keywords"] = [
                    value
                    for value in str(raw_keywords or "").split()
                    if value
                ]
        for key in (
            "entities",
            "entity_ids",
            "canonical_topics",
            "participants",
            "topic_keys",
            "tags",
            "metadata",
            "payload",
            "details",
            "previous_payload",
            "new_payload",
            "evidence_fact_ids",
            "evidence_activity_ids",
            "fact_ids",
            "source_segment_ids",
        ):
            if key in item:
                item[key] = _json_loads(
                    item[key],
                    {} if key in {"metadata", "payload", "details"} else [],
                )
        for key in (
            "embedding",
            "identity_text_embedding",
        ):
            if key in item:
                item[key] = _blob_to_embedding(item[key])
        return item
