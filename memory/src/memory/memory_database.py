#!/usr/bin/env python3
"""SQLite storage for the unified memory prototype.

The schema defines the current unified memory line:

    memory_source_segments -> memory_episodes <- memory_facts
                                             -> entity_claims / intent-execution

`memory_recall_documents` is a derived retrieval projection. Every retrievable
memory object writes one document there; source tables retain only domain data.
"""

from __future__ import annotations

import json
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

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


class SessionDB:
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

    def open_reader(self) -> "SessionDB":
        """Open a query-only connection without running schema initialization."""
        reader = object.__new__(SessionDB)
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
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS memory_episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                episode_type TEXT NOT NULL,
                title TEXT NOT NULL DEFAULT '',
                summary TEXT NOT NULL DEFAULT '',
                participants TEXT NOT NULL DEFAULT '[]',
                entity_ids TEXT NOT NULL DEFAULT '[]',
                canonical_topics TEXT NOT NULL DEFAULT '[]',
                started_at TEXT,
                ended_at TEXT,
                metadata TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS memory_source_segments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                episode_id INTEGER,
                text TEXT NOT NULL,
                started_at TEXT NOT NULL DEFAULT '',
                ended_at TEXT NOT NULL DEFAULT '',
                tags TEXT NOT NULL DEFAULT '[]',
                metadata TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(episode_id) REFERENCES memory_episodes(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS memory_facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                episode_id INTEGER,
                fact_type TEXT NOT NULL DEFAULT 'context',
                summary TEXT NOT NULL,
                keywords TEXT NOT NULL DEFAULT '[]',
                entities TEXT NOT NULL DEFAULT '[]',
                entity_ids TEXT NOT NULL DEFAULT '[]',
                fact_root_topic TEXT NOT NULL DEFAULT '',
                fact_aspect_topic TEXT NOT NULL DEFAULT '',
                event_time_key TEXT NOT NULL DEFAULT '',
                dialogue_time_key TEXT NOT NULL DEFAULT '',
                confidence REAL NOT NULL DEFAULT 0.85,
                importance REAL NOT NULL DEFAULT 0.5,
                metadata TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(episode_id) REFERENCES memory_episodes(id) ON DELETE CASCADE
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

            CREATE TABLE IF NOT EXISTS memory_fact_entity_claim_signal_mapping (
                fact_id INTEGER NOT NULL,
                signal_key TEXT NOT NULL,
                subject_entity_id INTEGER,
                claim_type_hint TEXT NOT NULL DEFAULT '',
                signal_kind TEXT NOT NULL DEFAULT 'none',
                claim_anchor TEXT NOT NULL DEFAULT '',
                claim_anchor_key TEXT NOT NULL DEFAULT '',
                confidence REAL NOT NULL DEFAULT 0.0,
                processed_for_memory_entity_claim INTEGER NOT NULL DEFAULT 0,
                processed_for_memory_entity_claim_induction INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(fact_id, signal_key),
                FOREIGN KEY(fact_id) REFERENCES memory_facts(id) ON DELETE CASCADE,
                FOREIGN KEY(subject_entity_id) REFERENCES memory_entity_nodes(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS memory_fact_prospective_signal_mapping (
                fact_id INTEGER NOT NULL,
                signal_key TEXT NOT NULL,
                subject_entity_id INTEGER,
                evidence_kind TEXT NOT NULL DEFAULT '',
                candidate_object_types TEXT NOT NULL DEFAULT '[]',
                operation_hint TEXT NOT NULL DEFAULT '',
                user_role TEXT NOT NULL DEFAULT '',
                prospective_anchor TEXT NOT NULL DEFAULT '',
                prospective_anchor_key TEXT NOT NULL DEFAULT '',
                assertion_source TEXT NOT NULL DEFAULT '',
                explicitness TEXT NOT NULL DEFAULT '',
                evidence_basis TEXT NOT NULL DEFAULT '',
                confidence REAL NOT NULL DEFAULT 0.0,
                processed_for_memory_prospective_update INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(fact_id, signal_key),
                FOREIGN KEY(fact_id) REFERENCES memory_facts(id) ON DELETE CASCADE,
                FOREIGN KEY(subject_entity_id) REFERENCES memory_entity_nodes(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS memory_entity_mapping (
                entity_id INTEGER PRIMARY KEY,
                episode_id TEXT NOT NULL DEFAULT '[]',
                fact_id TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(entity_id) REFERENCES memory_entity_nodes(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS memory_fact_episode_mapping (
                fact_id INTEGER NOT NULL,
                episode_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(fact_id, episode_id),
                FOREIGN KEY(fact_id) REFERENCES memory_facts(id) ON DELETE CASCADE,
                FOREIGN KEY(episode_id) REFERENCES memory_episodes(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS memory_topic_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic_kind TEXT NOT NULL,
                topic_name TEXT NOT NULL,
                topic_key TEXT NOT NULL,
                fact_occurrence_count INTEGER NOT NULL DEFAULT 0,
                episode_occurrence_count INTEGER NOT NULL DEFAULT 0,
                first_seen_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(topic_kind, topic_key)
            );

            CREATE TABLE IF NOT EXISTS memory_topic_mapping (
                topic_item_id INTEGER PRIMARY KEY,
                fact_ids TEXT NOT NULL DEFAULT '[]',
                episode_ids TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(topic_item_id) REFERENCES memory_topic_items(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS memory_entity_claims (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject_entity_id INTEGER NOT NULL,
                predicate TEXT NOT NULL,
                object_entity_id INTEGER NOT NULL DEFAULT 0,
                claim_text TEXT NOT NULL DEFAULT '',
                claim_type TEXT NOT NULL,
                claim_origin TEXT NOT NULL,
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

            CREATE TABLE IF NOT EXISTS memory_entity_claim_evidence (
                claim_id INTEGER NOT NULL,
                evidence_type TEXT NOT NULL,
                evidence_id INTEGER NOT NULL,
                role TEXT NOT NULL DEFAULT 'support',
                weight REAL NOT NULL DEFAULT 1.0,
                observed_at TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(claim_id, evidence_type, evidence_id, role),
                FOREIGN KEY(claim_id) REFERENCES memory_entity_claims(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS memory_entity_claim_induction (
                claim_id INTEGER PRIMARY KEY,
                condition_text TEXT NOT NULL DEFAULT '',
                support_count INTEGER NOT NULL DEFAULT 0,
                first_observed_at TEXT NOT NULL DEFAULT '',
                last_observed_at TEXT NOT NULL DEFAULT '',
                consolidation_version TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(claim_id) REFERENCES memory_entity_claims(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS memory_entity_claim_derivations (
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
                FOREIGN KEY(claim_id) REFERENCES memory_entity_claims(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS memory_entity_claim_premises (
                derivation_id INTEGER NOT NULL,
                premise_claim_id INTEGER NOT NULL,
                premise_role TEXT NOT NULL DEFAULT 'support',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(derivation_id, premise_claim_id, premise_role),
                FOREIGN KEY(derivation_id) REFERENCES memory_entity_claim_derivations(id) ON DELETE CASCADE,
                FOREIGN KEY(premise_claim_id) REFERENCES memory_entity_claims(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS memory_entity_claim_events (
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
                FOREIGN KEY(target_claim_id) REFERENCES memory_entity_claims(id) ON DELETE CASCADE,
                FOREIGN KEY(trigger_claim_id) REFERENCES memory_entity_claims(id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS memory_goals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                world_owner_entity_id INTEGER NOT NULL,
                owner_entity_id INTEGER NOT NULL,
                canonical_key TEXT NOT NULL,
                summary TEXT NOT NULL,
                desired_outcome TEXT NOT NULL DEFAULT '',
                success_criteria TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'active',
                target_at TEXT NOT NULL DEFAULT '',
                confidence REAL NOT NULL DEFAULT 0.7,
                metadata TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(world_owner_entity_id, owner_entity_id, canonical_key),
                FOREIGN KEY(world_owner_entity_id) REFERENCES memory_entity_nodes(id) ON DELETE CASCADE,
                FOREIGN KEY(owner_entity_id) REFERENCES memory_entity_nodes(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS memory_plans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                world_owner_entity_id INTEGER NOT NULL,
                actor_entity_id INTEGER NOT NULL,
                canonical_key TEXT NOT NULL,
                summary TEXT NOT NULL,
                event_or_activity TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'planned',
                start_at TEXT NOT NULL DEFAULT '',
                end_at TEXT NOT NULL DEFAULT '',
                time_precision TEXT NOT NULL DEFAULT 'unknown',
                location_entity_id INTEGER,
                location_text TEXT NOT NULL DEFAULT '',
                confidence REAL NOT NULL DEFAULT 0.7,
                metadata TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(world_owner_entity_id, actor_entity_id, canonical_key),
                FOREIGN KEY(world_owner_entity_id) REFERENCES memory_entity_nodes(id) ON DELETE CASCADE,
                FOREIGN KEY(actor_entity_id) REFERENCES memory_entity_nodes(id) ON DELETE CASCADE,
                FOREIGN KEY(location_entity_id) REFERENCES memory_entity_nodes(id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS memory_plan_entities (
                plan_id INTEGER NOT NULL,
                entity_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(plan_id, entity_id, role),
                FOREIGN KEY(plan_id) REFERENCES memory_plans(id) ON DELETE CASCADE,
                FOREIGN KEY(entity_id) REFERENCES memory_entity_nodes(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS memory_work_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                world_owner_entity_id INTEGER NOT NULL,
                responsible_entity_id INTEGER NOT NULL,
                canonical_key TEXT NOT NULL,
                summary TEXT NOT NULL,
                action_text TEXT NOT NULL DEFAULT '',
                deliverable TEXT NOT NULL DEFAULT '',
                responsibility_type TEXT NOT NULL DEFAULT 'personal_action',
                status TEXT NOT NULL DEFAULT 'open',
                due_at TEXT NOT NULL DEFAULT '',
                start_at TEXT NOT NULL DEFAULT '',
                priority TEXT NOT NULL DEFAULT '',
                confidence REAL NOT NULL DEFAULT 0.7,
                completed_at TEXT NOT NULL DEFAULT '',
                extractor_version TEXT NOT NULL DEFAULT '',
                prompt_version TEXT NOT NULL DEFAULT '',
                metadata TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(world_owner_entity_id, responsible_entity_id, canonical_key),
                FOREIGN KEY(world_owner_entity_id) REFERENCES memory_entity_nodes(id) ON DELETE CASCADE,
                FOREIGN KEY(responsible_entity_id) REFERENCES memory_entity_nodes(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS memory_work_item_entities (
                work_item_id INTEGER NOT NULL,
                entity_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(work_item_id, entity_id, role),
                FOREIGN KEY(work_item_id) REFERENCES memory_work_items(id) ON DELETE CASCADE,
                FOREIGN KEY(entity_id) REFERENCES memory_entity_nodes(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS memory_intent_evidence (
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

            CREATE TABLE IF NOT EXISTS memory_intent_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                object_type TEXT NOT NULL,
                object_id INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                previous_status TEXT NOT NULL DEFAULT '',
                new_status TEXT NOT NULL DEFAULT '',
                previous_payload TEXT NOT NULL DEFAULT '{}',
                new_payload TEXT NOT NULL DEFAULT '{}',
                evidence_fact_ids TEXT NOT NULL DEFAULT '[]',
                effective_at TEXT NOT NULL DEFAULT '',
                decision_source TEXT NOT NULL DEFAULT '',
                reason TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS memory_goal_work_item_mappings (
                goal_id INTEGER NOT NULL,
                work_item_id INTEGER NOT NULL,
                relation TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(goal_id, work_item_id, relation),
                FOREIGN KEY(goal_id) REFERENCES memory_goals(id) ON DELETE CASCADE,
                FOREIGN KEY(work_item_id) REFERENCES memory_work_items(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS memory_plan_work_item_mappings (
                plan_id INTEGER NOT NULL,
                work_item_id INTEGER NOT NULL,
                relation TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(plan_id, work_item_id, relation),
                FOREIGN KEY(plan_id) REFERENCES memory_plans(id) ON DELETE CASCADE,
                FOREIGN KEY(work_item_id) REFERENCES memory_work_items(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS memory_work_item_relations (
                source_work_item_id INTEGER NOT NULL,
                target_work_item_id INTEGER NOT NULL,
                relation TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(source_work_item_id, target_work_item_id, relation),
                FOREIGN KEY(source_work_item_id) REFERENCES memory_work_items(id) ON DELETE CASCADE,
                FOREIGN KEY(target_work_item_id) REFERENCES memory_work_items(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_memory_facts_event_time ON memory_facts(event_time_key);
            CREATE INDEX IF NOT EXISTS idx_memory_facts_dialogue_time ON memory_facts(dialogue_time_key);
            CREATE INDEX IF NOT EXISTS idx_memory_recall_documents_object
            ON memory_recall_documents(object_type, object_id);
            CREATE INDEX IF NOT EXISTS idx_memory_recall_documents_type_status_time
            ON memory_recall_documents(object_type, status, time_end, updated_at);
            CREATE INDEX IF NOT EXISTS idx_memory_source_segments_episode
            ON memory_source_segments(episode_id, id);
            CREATE INDEX IF NOT EXISTS idx_memory_fact_episode_episode
            ON memory_fact_episode_mapping(episode_id, updated_at);
            CREATE INDEX IF NOT EXISTS idx_memory_topic_items_kind_seen
            ON memory_topic_items(topic_kind, last_seen_at DESC);
            CREATE INDEX IF NOT EXISTS idx_memory_entity_claims_subject
            ON memory_entity_claims(subject_entity_id, claim_type, claim_origin, status);
            CREATE INDEX IF NOT EXISTS idx_memory_entity_claim_evidence_claim
            ON memory_entity_claim_evidence(claim_id, role, evidence_type);
            CREATE INDEX IF NOT EXISTS idx_memory_entity_claim_derivations_claim
            ON memory_entity_claim_derivations(claim_id, status, updated_at);
            CREATE INDEX IF NOT EXISTS idx_memory_entity_claim_premises_claim
            ON memory_entity_claim_premises(premise_claim_id, derivation_id);
            CREATE INDEX IF NOT EXISTS idx_memory_fact_entity_claim_signal_group
            ON memory_fact_entity_claim_signal_mapping(
                subject_entity_id, claim_type_hint, claim_anchor_key, fact_id
            );
            CREATE INDEX IF NOT EXISTS idx_memory_fact_prospective_signal_group
            ON memory_fact_prospective_signal_mapping(
                subject_entity_id, prospective_anchor_key, evidence_kind, fact_id
            );
            CREATE INDEX IF NOT EXISTS idx_memory_fact_entity_claim_processing
            ON memory_fact_entity_claim_signal_mapping(
                processed_for_memory_entity_claim, fact_id
            );
            CREATE INDEX IF NOT EXISTS idx_memory_fact_entity_claim_induction_processing
            ON memory_fact_entity_claim_signal_mapping(
                processed_for_memory_entity_claim_induction, fact_id
            );
            CREATE INDEX IF NOT EXISTS idx_memory_fact_prospective_update_processing
            ON memory_fact_prospective_signal_mapping(
                processed_for_memory_prospective_update, fact_id
            );
            CREATE INDEX IF NOT EXISTS idx_memory_entity_claim_events_target
            ON memory_entity_claim_events(target_claim_id, effective_at DESC, id DESC);
            CREATE INDEX IF NOT EXISTS idx_memory_entity_claim_events_trigger
            ON memory_entity_claim_events(trigger_claim_id, id DESC);
            CREATE INDEX IF NOT EXISTS idx_memory_goals_owner_status
            ON memory_goals(world_owner_entity_id, owner_entity_id, status);
            CREATE INDEX IF NOT EXISTS idx_memory_plans_actor_status_time
            ON memory_plans(world_owner_entity_id, actor_entity_id, status, start_at);
            CREATE INDEX IF NOT EXISTS idx_memory_work_items_responsible_status_due
            ON memory_work_items(world_owner_entity_id, responsible_entity_id, status, due_at);
            CREATE INDEX IF NOT EXISTS idx_memory_intent_evidence_object
            ON memory_intent_evidence(object_type, object_id, role);
            CREATE INDEX IF NOT EXISTS idx_memory_intent_events_object
            ON memory_intent_events(object_type, object_id, effective_at DESC, id DESC);
            """
        )
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

    def insert_episode(
        self,
        *,
        episode_type: str,
        title: str,
        summary: str,
        participants: Sequence[str],
        started_at: str,
        ended_at: str,
        canonical_topics: Optional[Sequence[str]] = None,
        entity_ids: Optional[Sequence[int]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        now = local_now_text()
        episode_metadata = dict(metadata or {})
        normalized_topics = list(canonical_topics or [])
        cur = self._conn.execute(
            """
            INSERT INTO memory_episodes (
                episode_type, title, summary, participants,
                entity_ids, canonical_topics, started_at, ended_at,
                metadata, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                episode_type,
                title,
                summary,
                _json_dumps(list(participants or [])),
                _json_dumps([int(value) for value in entity_ids or []]),
                _json_dumps(normalized_topics),
                started_at,
                ended_at,
                _json_dumps(episode_metadata),
                now,
                now,
            ),
        )
        episode_id = int(cur.lastrowid)
        self.insert_entity_memory_mappings([
            {
                "entity_id": int(entity_id),
                "episode_id": [episode_id],
            }
            for entity_id in entity_ids or []
        ])
        self._commit_if_needed()
        return episode_id

    def insert_memory_source_segments(
        self,
        *,
        segments: Sequence[Dict[str, Any]],
    ) -> List[int]:
        """Persist one submitted source-segment batch as one physical row.

        The table records batch lifecycle (unassigned / attached episode),
        while ``metadata.segments`` preserves the ordered logical segments
        needed by episode generation.  Returning a one-item list preserves
        the existing caller contract of ``source_segment_ids``.
        """
        now = local_now_text()
        normalized_segments: List[Dict[str, Any]] = []
        for segment in segments or []:
            if not isinstance(segment, dict):
                continue
            text = str(segment.get("text") or "").strip()
            if not text:
                continue
            tags = [
                str(tag).strip()
                for tag in (segment.get("tags") or [])
                if str(tag).strip()
            ]
            metadata = dict(
                segment.get("metadata")
                if isinstance(segment.get("metadata"), dict)
                else {}
            )
            for key in ("turn_index", "segment_index"):
                if key in segment and key not in metadata:
                    metadata[key] = segment[key]
            started_at = str(segment.get("started_at") or "").strip()
            ended_at = str(
                segment.get("ended_at") or segment.get("started_at") or ""
            ).strip()
            normalized_segments.append({
                "speaker": str(segment.get("speaker") or "").strip(),
                "text": text,
                "started_at": started_at,
                "ended_at": ended_at,
                "tags": tags,
                "metadata": metadata,
            })
        if not normalized_segments:
            return []

        batch_tags = list(dict.fromkeys(
            tag
            for item in normalized_segments
            for tag in item["tags"]
        ))
        batch_started_at = next(
            (item["started_at"] for item in normalized_segments if item["started_at"]),
            "",
        )
        batch_ended_at = next(
            (
                item["ended_at"] or item["started_at"]
                for item in reversed(normalized_segments)
                if item["ended_at"] or item["started_at"]
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
                    f"{item['speaker'] or 'unknown'}: {item['text']}"
                    for item in normalized_segments
                ),
                batch_started_at,
                batch_ended_at,
                _json_dumps(batch_tags),
                _json_dumps({
                    "storage_format": "segment_batch_v1",
                    "segment_count": len(normalized_segments),
                    "segments": normalized_segments,
                }),
                now,
                now,
            ),
        )
        self._commit_if_needed()
        return [int(cur.lastrowid)]

    def get_unassigned_memory_source_segments(
        self,
        *,
        limit: int = 240,
    ) -> List[Dict[str, Any]]:
        """Return logical source evidence waiting to be summarized.

        Each row is a physical batch; ``metadata.segments`` stores the
        ordered logical segments used for episode generation.
        """
        rows = self._conn.execute(
            """
            SELECT *
            FROM memory_source_segments
            WHERE episode_id IS NULL
            ORDER BY id ASC
            LIMIT ?
            """,
            (max(1, int(limit or 240)),),
        ).fetchall()
        source_segments: List[Dict[str, Any]] = []
        for row in rows:
            batch_row = self._row_to_dict(row)
            batch_metadata = (
                batch_row.get("metadata")
                if isinstance(batch_row.get("metadata"), dict)
                else {}
            )
            batch_segments = batch_metadata.get("segments")
            if not isinstance(batch_segments, list):
                continue
            for segment_index, raw_segment in enumerate(batch_segments):
                if not isinstance(raw_segment, dict):
                    continue
                text = str(raw_segment.get("text") or "").strip()
                if not text:
                    continue
                segment_metadata = (
                    raw_segment.get("metadata")
                    if isinstance(raw_segment.get("metadata"), dict)
                    else {}
                )
                source_segments.append({
                    "id": batch_row["id"],
                    "source_segment_row_id": batch_row["id"],
                    "speaker": str(raw_segment.get("speaker") or "").strip(),
                    "text": text,
                    "started_at": str(raw_segment.get("started_at") or "").strip(),
                    "ended_at": str(
                        raw_segment.get("ended_at")
                        or raw_segment.get("started_at")
                        or ""
                    ).strip(),
                    "tags": list(raw_segment.get("tags") or []),
                    "metadata": dict(segment_metadata),
                    "batch_segment_index": segment_index,
                })
        return source_segments

    def update_memory_source_segments_episode_id(
        self,
        *,
        source_segment_ids: Sequence[int],
        episode_id: int,
    ) -> int:
        """Attach persisted source evidence to its generated episode."""
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
            SET episode_id = ?, updated_at = ?
            WHERE id IN ({placeholders})
            """,
            (int(episode_id), local_now_text(), *normalized_ids),
        )
        self._commit_if_needed()
        return int(cur.rowcount or 0)

    def upsert_memory_topic_items(
        self,
        items: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Upsert topic registry items and merge their fact/episode IDs.

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
                    "episode_ids": [],
                },
            )
            for field in ("fact_ids", "episode_ids"):
                for item_id in normalize_ids(item.get(field)):
                    if item_id not in grouped_item[field]:
                        grouped_item[field].append(item_id)

        report = {
            "topic_item_ids": [],
            "created_count": 0,
            "updated_count": 0,
            "fact_links_added": 0,
            "episode_links_added": 0,
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
                        fact_occurrence_count, episode_occurrence_count,
                        first_seen_at, last_seen_at, created_at, updated_at
                    ) VALUES (?, ?, ?, 0, 0, ?, ?, ?, ?)
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
                SELECT fact_ids, episode_ids FROM memory_topic_mapping
                WHERE topic_item_id = ?
                """,
                (topic_item_id,),
            ).fetchone()
            existing_fact_ids = normalize_ids(
                _json_loads(mapping["fact_ids"], []) if mapping else []
            )
            existing_episode_ids = normalize_ids(
                _json_loads(mapping["episode_ids"], []) if mapping else []
            )
            new_fact_ids = [
                fact_id for fact_id in item["fact_ids"]
                if fact_id not in existing_fact_ids
            ]
            new_episode_ids = [
                episode_id for episode_id in item["episode_ids"]
                if episode_id not in existing_episode_ids
            ]
            merged_fact_ids = [*existing_fact_ids, *new_fact_ids]
            merged_episode_ids = [*existing_episode_ids, *new_episode_ids]

            self._conn.execute(
                """
                INSERT INTO memory_topic_mapping (
                    topic_item_id, fact_ids, episode_ids, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(topic_item_id) DO UPDATE SET
                    fact_ids = excluded.fact_ids,
                    episode_ids = excluded.episode_ids,
                    updated_at = excluded.updated_at
                """,
                (
                    topic_item_id,
                    _json_dumps(merged_fact_ids),
                    _json_dumps(merged_episode_ids),
                    now,
                    now,
                ),
            )
            if new_fact_ids or new_episode_ids:
                self._conn.execute(
                    """
                    UPDATE memory_topic_items
                    SET fact_occurrence_count = fact_occurrence_count + ?,
                        episode_occurrence_count = episode_occurrence_count + ?,
                        last_seen_at = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        len(new_fact_ids),
                        len(new_episode_ids),
                        now,
                        now,
                        topic_item_id,
                    ),
                )
                report["updated_count"] += 1
            report["topic_item_ids"].append(topic_item_id)
            report["fact_links_added"] += len(new_fact_ids)
            report["episode_links_added"] += len(new_episode_ids)

        self._commit_if_needed()
        return report

    def list_memory_topic_items(self, *, limit: int = 240) -> List[Dict[str, Any]]:
        """Load recent topic registry items with their compact evidence IDs."""
        rows = self._conn.execute(
            """
            SELECT item.*, mapping.fact_ids, mapping.episode_ids
            FROM memory_topic_items AS item
            LEFT JOIN memory_topic_mapping AS mapping
                ON mapping.topic_item_id = item.id
            ORDER BY item.last_seen_at DESC, item.id DESC
            LIMIT ?
            """,
            (max(1, int(limit or 240)),),
        ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def get_unprocessed_facts(
        self,
        *,
        processing_target: str = "entity_claim",
        reference_timestamp: Any,
        limit: int = 100,
        restrict_to_today: bool = True,
        require_episode: bool = False,
    ) -> List[Dict[str, Any]]:
        processing_mappings = {
            "entity_claim": (
                "memory_fact_entity_claim_signal_mapping",
                "processed_for_memory_entity_claim",
            ),
            "entity_claim_induction": (
                "memory_fact_entity_claim_signal_mapping",
                "processed_for_memory_entity_claim_induction",
            ),
            "prospective_update": (
                "memory_fact_prospective_signal_mapping",
                "processed_for_memory_prospective_update",
            ),
        }
        target = str(processing_target or "entity_claim").strip().lower()
        try:
            processing_table, processing_column = processing_mappings[target]
        except KeyError as exc:
            raise ValueError(
                "processing_target must be 'entity_claim', "
                "'entity_claim_induction', or 'prospective_update'"
            ) from exc

        clauses: List[str] = [f"processing.{processing_column} = 0"]
        if target == "prospective_update":
            # ``__fact__`` records that extraction found no prospective signal;
            # it is intentionally not work for the prospective projection.
            clauses.append("processing.signal_key != '__fact__'")
        params: List[Any] = []
        if require_episode:
            clauses.append("fact.episode_id IS NOT NULL")
        if restrict_to_today:
            local_now = _coerce_reference_datetime(reference_timestamp).astimezone()
            event_date = local_now.date().isoformat()
            clauses.append("substr(fact.created_at, 1, 10) = ?")
            params.append(event_date)
        where = "WHERE " + " AND ".join(clauses) if clauses else ""
        rows = self._conn.execute(
            f"""
            SELECT DISTINCT fact.*
            FROM memory_facts AS fact
            JOIN {processing_table} AS processing ON processing.fact_id = fact.id
            {where}
            ORDER BY replace(substr(fact.created_at, 1, 19), 'T', ' ') ASC, fact.id ASC
            LIMIT ?
            """,
            (*params, max(1, int(limit or 100))),
        ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def get_unassigned_memory_facts_in_time_window(
        self,
        *,
        started_at: str,
        ended_at: str,
        limit: int = 80,
    ) -> List[Dict[str, Any]]:
        """Load unassigned facts whose dialogue time belongs to one source window."""
        start = str(started_at or "").strip()
        end = str(ended_at or started_at or "").strip()
        if not start or not end:
            return []
        rows = self._conn.execute(
            """
            SELECT *
            FROM memory_facts
            WHERE episode_id IS NULL
              AND dialogue_time_key >= ?
              AND dialogue_time_key <= ?
            ORDER BY dialogue_time_key ASC, id ASC
            LIMIT ?
            """,
            (
                start,
                end,
                max(1, int(limit or 80)),
            ),
        ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def update_facts_episode_id(
        self,
        *,
        fact_ids: Sequence[int],
        episode_id: int,
    ) -> int:
        """Attach existing facts to an episode and mirror entity mappings."""
        normalized_ids = [int(value) for value in fact_ids if str(value).strip().isdigit()]
        if not normalized_ids:
            return 0
        placeholders = ",".join("?" for _ in normalized_ids)
        rows = self._conn.execute(
            f"SELECT id, entity_ids FROM memory_facts WHERE id IN ({placeholders})",
            normalized_ids,
        ).fetchall()
        now = local_now_text()
        cur = self._conn.execute(
            f"UPDATE memory_facts SET episode_id = ?, updated_at = ? WHERE id IN ({placeholders})",
            (int(episode_id), now, *normalized_ids),
        )
        self._conn.execute(
            f"DELETE FROM memory_fact_episode_mapping "
            f"WHERE fact_id IN ({placeholders}) AND episode_id != ?",
            (*normalized_ids, int(episode_id)),
        )
        self.insert_fact_episode_mappings([
            {
                "fact_id": fact_id,
                "episode_id": int(episode_id),
            }
            for fact_id in normalized_ids
        ])
        mappings = []
        for row in rows:
            for entity_id in _json_loads(row["entity_ids"], default=[]):
                if str(entity_id).strip().isdigit():
                    mappings.append({"entity_id": int(entity_id), "episode_id": [int(episode_id)]})
        if mappings:
            self.insert_entity_memory_mappings(mappings)
        self._commit_if_needed()
        return int(cur.rowcount or 0)

    def insert_fact_episode_mappings(
        self,
        mappings: Sequence[Dict[str, Any]],
    ) -> int:
        """Persist stable membership links from facts to memory episodes."""
        normalized_pairs = {
            (int(mapping["fact_id"]), int(mapping["episode_id"]))
            for mapping in mappings or []
            if str(mapping.get("fact_id") or "").strip().isdigit()
            and str(mapping.get("episode_id") or "").strip().isdigit()
            and int(mapping["fact_id"]) > 0
            and int(mapping["episode_id"]) > 0
        }
        if not normalized_pairs:
            return 0
        fact_ids = sorted({fact_id for fact_id, _episode_id in normalized_pairs})
        episode_ids = sorted(
            {episode_id for _fact_id, episode_id in normalized_pairs}
        )
        fact_placeholders = ",".join("?" for _ in fact_ids)
        episode_placeholders = ",".join("?" for _ in episode_ids)
        existing_fact_ids = {
            int(row["id"])
            for row in self._conn.execute(
                f"SELECT id FROM memory_facts WHERE id IN ({fact_placeholders})",
                fact_ids,
            ).fetchall()
        }
        existing_episode_ids = {
            int(row["id"])
            for row in self._conn.execute(
                f"SELECT id FROM memory_episodes WHERE id IN ({episode_placeholders})",
                episode_ids,
            ).fetchall()
        }
        now = local_now_text()
        changed_count = 0
        for fact_id, episode_id in normalized_pairs:
            if fact_id not in existing_fact_ids or episode_id not in existing_episode_ids:
                continue
            self._conn.execute(
                """
                INSERT INTO memory_fact_episode_mapping (
                    fact_id, episode_id, created_at, updated_at
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(fact_id, episode_id) DO NOTHING
                """,
                (fact_id, episode_id, now, now),
            )
            changed_count += 1
        self._commit_if_needed()
        return changed_count

    def mark_facts_processed(
        self,
        *,
        processing_target: str,
        fact_ids: Sequence[int],
    ) -> int:
        processing_mappings = {
            "entity_claim": (
                "memory_fact_entity_claim_signal_mapping",
                "processed_for_memory_entity_claim",
            ),
            "entity_claim_induction": (
                "memory_fact_entity_claim_signal_mapping",
                "processed_for_memory_entity_claim_induction",
            ),
            "prospective_update": (
                "memory_fact_prospective_signal_mapping",
                "processed_for_memory_prospective_update",
            ),
        }
        target = str(processing_target or "").strip().lower()
        try:
            processing_table, processing_column = processing_mappings[target]
        except KeyError as exc:
            raise ValueError(
                "processing_target must be 'entity_claim', "
                "'entity_claim_induction', or 'prospective_update'"
            ) from exc
        ids = [int(value) for value in fact_ids if value is not None]
        if not ids:
            return 0
        placeholders = ",".join("?" for _ in ids)
        affected_fact_count = int(self._conn.execute(
            f"SELECT COUNT(DISTINCT fact_id) FROM {processing_table} "
            f"WHERE fact_id IN ({placeholders})",
            ids,
        ).fetchone()[0] or 0)
        cur = self._conn.execute(
            f"""
            UPDATE {processing_table}
            SET {processing_column} = 1,
                updated_at = ?
            WHERE fact_id IN ({placeholders})
            """,
            (local_now_text(), *ids),
        )
        self._commit_if_needed()
        return affected_fact_count

    def upsert_fact_entity_claim_signal_mappings(
        self,
        mappings: Sequence[Dict[str, Any]],
    ) -> int:
        """Index fact-level claim signals for exact induction-group expansion."""
        now = local_now_text()
        changed = 0
        for mapping in mappings or []:
            try:
                fact_id = int(mapping["fact_id"])
            except (KeyError, TypeError, ValueError):
                continue
            signal_key = str(mapping.get("signal_key") or "").strip()
            is_sentinel = signal_key == "__fact__"
            try:
                subject_entity_id = int(mapping["subject_entity_id"])
            except (KeyError, TypeError, ValueError):
                subject_entity_id = 0
            claim_type_hint = str(mapping.get("claim_type_hint") or "").strip()
            signal_kind = str(mapping.get("signal_kind") or "").strip()
            claim_anchor = str(mapping.get("claim_anchor") or "").strip()
            claim_anchor_key = str(mapping.get("claim_anchor_key") or "").strip()
            if fact_id <= 0:
                continue
            if is_sentinel:
                subject_entity_id = 0
                claim_type_hint = ""
                signal_kind = "none"
                claim_anchor = ""
                claim_anchor_key = ""
                has_real_signal = self._conn.execute(
                    """
                    SELECT 1 FROM memory_fact_entity_claim_signal_mapping
                    WHERE fact_id = ? AND signal_key != '__fact__'
                    LIMIT 1
                    """,
                    (fact_id,),
                ).fetchone()
                if has_real_signal is not None:
                    continue
            elif (
                subject_entity_id <= 0
                or not claim_type_hint
                or not signal_kind
                or not claim_anchor
                or not claim_anchor_key
            ):
                continue
            if not signal_key:
                signal_key = (
                    f"{subject_entity_id}|{claim_type_hint}|{claim_anchor_key}"
                )
            if not is_sentinel:
                self._conn.execute(
                    "DELETE FROM memory_fact_entity_claim_signal_mapping "
                    "WHERE fact_id = ? AND signal_key = '__fact__'",
                    (fact_id,),
                )
            self._conn.execute(
                """
                INSERT INTO memory_fact_entity_claim_signal_mapping (
                    fact_id, signal_key, subject_entity_id, claim_type_hint,
                    signal_kind, claim_anchor, claim_anchor_key, confidence,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(fact_id, signal_key) DO UPDATE SET
                    signal_kind = excluded.signal_kind,
                    claim_anchor = excluded.claim_anchor,
                    confidence = MAX(
                        memory_fact_entity_claim_signal_mapping.confidence,
                        excluded.confidence
                    ),
                    updated_at = excluded.updated_at
                """,
                (
                    fact_id, signal_key,
                    subject_entity_id or None, claim_type_hint, signal_kind,
                    claim_anchor, claim_anchor_key,
                    float(mapping.get("confidence") or 0.0), now, now,
                ),
            )
            changed += 1
        self._commit_if_needed()
        return changed

    def memory_facts_for_entity_claim_signal_group(
        self,
        *,
        subject_entity_id: int,
        claim_type_hint: str,
        claim_anchor_key: str,
        limit: int = 64,
    ) -> List[Dict[str, Any]]:
        """Load bounded historical facts for one exact induction group."""
        rows = self._conn.execute(
            """
            SELECT fact.*
            FROM memory_fact_entity_claim_signal_mapping AS signal
            JOIN memory_facts AS fact ON fact.id = signal.fact_id
            WHERE signal.subject_entity_id = ?
              AND signal.claim_type_hint = ?
              AND signal.claim_anchor_key = ?
              AND fact.episode_id IS NOT NULL
            ORDER BY fact.dialogue_time_key DESC, fact.id DESC
            LIMIT ?
            """,
            (
                int(subject_entity_id), str(claim_type_hint or ""),
                str(claim_anchor_key or ""), max(1, int(limit or 64)),
            ),
        ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def get_fact_entity_claim_signal_mappings(
        self,
        fact_id: int,
    ) -> Dict[str, Dict[str, Any]]:
        """Load the normalized claim signals persisted for one stored fact."""
        if int(fact_id or 0) <= 0:
            return {}
        rows = self._conn.execute(
            """
            SELECT signal.*, entity.name AS subject
            FROM memory_fact_entity_claim_signal_mapping AS signal
            LEFT JOIN memory_entity_nodes AS entity
                ON entity.id = signal.subject_entity_id
            WHERE signal.fact_id = ?
              AND signal.signal_key != '__fact__'
            ORDER BY signal.claim_type_hint ASC,
                     signal.claim_anchor_key ASC
            """,
            (int(fact_id),),
        ).fetchall()
        return {
            str(mapping["signal_key"]): mapping
            for row in rows
            for mapping in [self._row_to_dict(row)]
        }

    def upsert_fact_prospective_signal_mappings(
        self,
        mappings: Sequence[Dict[str, Any]],
    ) -> int:
        """Persist normalized future-facing evidence signals for one fact."""
        now = local_now_text()
        changed = 0
        for mapping in mappings or []:
            try:
                fact_id = int(mapping["fact_id"])
            except (KeyError, TypeError, ValueError):
                continue
            signal_key = str(mapping.get("signal_key") or "").strip()
            is_sentinel = signal_key == "__fact__"
            try:
                subject_entity_id = int(mapping["subject_entity_id"])
            except (KeyError, TypeError, ValueError):
                subject_entity_id = 0
            evidence_kind = str(mapping.get("evidence_kind") or "").strip().lower()
            user_role = str(mapping.get("user_role") or "").strip().lower()
            prospective_anchor = str(mapping.get("prospective_anchor") or "").strip()
            prospective_anchor_key = str(mapping.get("prospective_anchor_key") or "").strip()
            assertion_source = str(mapping.get("assertion_source") or "").strip().lower()
            explicitness = str(mapping.get("explicitness") or "").strip().lower()
            candidate_object_types = list(dict.fromkeys(
                str(value).strip().lower()
                for value in (mapping.get("candidate_object_types") or [])
                if str(value).strip().lower() in {"goal", "plan", "work_item"}
            ))
            if fact_id <= 0:
                continue
            if is_sentinel:
                subject_entity_id = 0
                evidence_kind = user_role = ""
                prospective_anchor = prospective_anchor_key = ""
                assertion_source = explicitness = ""
                candidate_object_types = []
                has_real_signal = self._conn.execute(
                    """
                    SELECT 1 FROM memory_fact_prospective_signal_mapping
                    WHERE fact_id = ? AND signal_key != '__fact__'
                    LIMIT 1
                    """,
                    (fact_id,),
                ).fetchone()
                if has_real_signal is not None:
                    continue
            elif (
                subject_entity_id <= 0
                or evidence_kind not in {"goal", "plan", "responsibility", "lifecycle_update"}
                or user_role not in {"owner", "participant", "responsible"}
                or not prospective_anchor
                or not prospective_anchor_key
                or assertion_source not in {"self_statement", "third_party_report", "observed_event"}
                or explicitness not in {"direct", "reported", "tentative"}
            ):
                continue
            if not signal_key:
                signal_key = (
                    f"{subject_entity_id}|{evidence_kind}|{prospective_anchor_key}"
                )
            if not is_sentinel:
                self._conn.execute(
                    "DELETE FROM memory_fact_prospective_signal_mapping "
                    "WHERE fact_id = ? AND signal_key = '__fact__'",
                    (fact_id,),
                )
            self._conn.execute(
                """
                INSERT INTO memory_fact_prospective_signal_mapping (
                    fact_id, signal_key, subject_entity_id, evidence_kind,
                    candidate_object_types, operation_hint, user_role,
                    prospective_anchor, prospective_anchor_key,
                    assertion_source, explicitness, evidence_basis,
                    confidence, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(fact_id, signal_key) DO UPDATE SET
                    candidate_object_types = excluded.candidate_object_types,
                    user_role = excluded.user_role,
                    prospective_anchor = excluded.prospective_anchor,
                    assertion_source = excluded.assertion_source,
                    explicitness = excluded.explicitness,
                    evidence_basis = excluded.evidence_basis,
                    confidence = MAX(
                        memory_fact_prospective_signal_mapping.confidence,
                        excluded.confidence
                    ),
                    updated_at = excluded.updated_at
                """,
                (
                    fact_id, signal_key, subject_entity_id or None, evidence_kind,
                    _json_dumps(candidate_object_types), "", user_role,
                    prospective_anchor, prospective_anchor_key,
                    assertion_source, explicitness,
                    str(mapping.get("evidence_basis") or "").strip(),
                    float(mapping.get("confidence") or 0.0), now, now,
                ),
            )
            changed += 1
        self._commit_if_needed()
        return changed

    def get_fact_prospective_signal_mappings(
        self,
        fact_id: int,
    ) -> Dict[str, Dict[str, Any]]:
        """Load a fact's prospective signals keyed by stable signal identity."""
        if int(fact_id or 0) <= 0:
            return {}
        rows = self._conn.execute(
            """
            SELECT signal.*, entity.name AS subject
            FROM memory_fact_prospective_signal_mapping AS signal
            LEFT JOIN memory_entity_nodes AS entity ON entity.id = signal.subject_entity_id
            WHERE signal.fact_id = ?
              AND signal.signal_key != '__fact__'
            ORDER BY signal.evidence_kind, signal.prospective_anchor_key
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

    def upsert_entity_claim(
        self,
        *,
        subject_entity_id: int,
        predicate: str,
        object_entity_id: Optional[int] = None,
        claim_text: str = "",
        claim_type: str,
        claim_origin: str,
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
        predicate = str(predicate or "").strip()
        claim_text = re.sub(r"\s+", " ", str(claim_text or "")).strip()
        existing = self._conn.execute(
            """
            SELECT id, confidence, status FROM memory_entity_claims
            WHERE subject_entity_id = ? AND predicate = ? AND object_entity_id = ?
              AND claim_text = ? AND claim_type = ? AND claim_origin = ?
            """,
            (subject_id, predicate, object_id, claim_text, claim_type, claim_origin),
        ).fetchone()
        if existing:
            claim_id = int(existing["id"])
            self._conn.execute(
                """
                UPDATE memory_entity_claims
                SET claim_text = ?, status = ?, confidence = ?, valid_from = ?, valid_to = ?,
                    source_actor_entity_id = ?, extractor_version = ?, prompt_version = ?,
                    metadata = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    claim_text, status,
                    max(float(existing["confidence"] or 0.0), float(confidence or 0.0)),
                    valid_from, valid_to, source_actor_entity_id, extractor_version,
                    prompt_version, _json_dumps(metadata or {}), now, claim_id,
                ),
            )
            self._commit_if_needed()
            return claim_id, False
        cur = self._conn.execute(
            """
            INSERT INTO memory_entity_claims (
                subject_entity_id, predicate, object_entity_id, claim_text,
                claim_type, claim_origin, status, confidence, valid_from, valid_to,
                source_actor_entity_id, extractor_version, prompt_version, metadata,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                subject_id, predicate, object_id, claim_text,
                claim_type, claim_origin, status, float(confidence or 0.0),
                valid_from, valid_to, source_actor_entity_id, extractor_version,
                prompt_version, _json_dumps(metadata or {}), now, now,
            ),
        )
        self._commit_if_needed()
        return int(cur.lastrowid), True

    def transition_entity_claim_status(
        self,
        *,
        target_claim_id: int,
        new_status: str,
        trigger_claim_id: Optional[int] = None,
        semantic_relation: str = "",
        effective_at: str = "",
        decision_source: str = "",
        semantic_confidence: Optional[float] = None,
        semantic_reason: str = "",
        policy_reason: str = "",
        details: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Change a claim status and retain the causally linked history event.

        The event is deliberately written only for an actual status transition;
        evidence additions and duplicate merges do not make a claim appear to
        have been abandoned or revived.
        """
        target_id = int(target_claim_id)
        status = str(new_status or "").strip()
        if not status:
            return False
        occurred_at = str(effective_at or "").strip()
        now = local_now_text()
        with self.transaction():
            row = self._conn.execute(
                "SELECT status FROM memory_entity_claims WHERE id = ?",
                (target_id,),
            ).fetchone()
            if not row:
                return False
            previous_status = str(row["status"] or "")
            if previous_status == status:
                return False
            # A superseding claim closes the prior claim's validity interval.
            # We leave valid_to untouched for a weakened claim: it may still
            # describe a partially valid or temporarily interrupted pattern.
            if status == "superseded" and occurred_at:
                self._conn.execute(
                    """
                    UPDATE memory_entity_claims
                    SET status = ?,
                        valid_to = CASE
                            WHEN valid_to = '' OR valid_to > ? THEN ?
                            ELSE valid_to
                        END,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (status, occurred_at, occurred_at, now, target_id),
                )
            else:
                self._conn.execute(
                    "UPDATE memory_entity_claims SET status = ?, updated_at = ? WHERE id = ?",
                    (status, now, target_id),
                )
            self._conn.execute(
                """
                INSERT INTO memory_entity_claim_events (
                    target_claim_id, trigger_claim_id, event_type,
                    previous_status, new_status, semantic_relation, effective_at,
                    decision_source, semantic_confidence, semantic_reason,
                    policy_reason, details, created_at
                ) VALUES (?, ?, 'status_transition', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    target_id,
                    int(trigger_claim_id) if trigger_claim_id is not None else None,
                    previous_status,
                    status,
                    str(semantic_relation or "").strip(),
                    occurred_at,
                    str(decision_source or "").strip(),
                    float(semantic_confidence) if semantic_confidence is not None else None,
                    str(semantic_reason or "").strip(),
                    str(policy_reason or "").strip(),
                    _json_dumps(details or {}),
                    now,
                ),
            )
        return True

    @staticmethod
    def _intent_object_spec(object_type: str) -> tuple[str, tuple[str, ...]]:
        normalized = str(object_type or "").strip().lower()
        specs = {
            "goal": ("memory_goals", (
                "world_owner_entity_id", "owner_entity_id", "canonical_key", "summary",
                "desired_outcome", "success_criteria", "status", "target_at",
                "confidence", "metadata",
            )),
            "plan": ("memory_plans", (
                "world_owner_entity_id", "actor_entity_id", "canonical_key", "summary",
                "event_or_activity", "status", "start_at", "end_at", "time_precision",
                "location_entity_id", "location_text", "confidence", "metadata",
            )),
            "work_item": ("memory_work_items", (
                "world_owner_entity_id", "responsible_entity_id", "canonical_key", "summary",
                "action_text", "deliverable", "responsibility_type", "status", "due_at",
                "start_at", "priority", "confidence", "completed_at", "extractor_version",
                "prompt_version", "metadata",
            )),
        }
        try:
            return specs[normalized]
        except KeyError as exc:
            raise ValueError("object_type must be goal, plan, or work_item") from exc

    def create_intent_object(self, *, object_type: str, payload: Dict[str, Any]) -> int:
        """Persist one normalized Goal, Plan, or Work item candidate."""
        table, columns = self._intent_object_spec(object_type)
        now = local_now_text()
        values: List[Any] = []
        for column in columns:
            value = payload.get(column)
            if column == "metadata":
                value = _json_dumps(value if isinstance(value, dict) else {})
            elif column == "confidence":
                value = float(value or 0.0)
            elif column.endswith("_entity_id"):
                value = int(value) if value not in (None, "", 0) else None
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
        assignments: List[str] = []
        values: List[Any] = []
        for column in allowed_columns:
            if column not in payload:
                continue
            value = payload[column]
            if column == "metadata":
                value = _json_dumps(value if isinstance(value, dict) else {})
            elif column == "confidence":
                value = float(value or 0.0)
            elif column.endswith("_entity_id"):
                value = int(value) if value not in (None, "", 0) else None
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
        statuses: Optional[Sequence[str]] = None,
        limit: int = 120,
    ) -> List[Dict[str, Any]]:
        table, _columns = self._intent_object_spec(object_type)
        clauses: List[str] = []
        params: List[Any] = []
        if world_owner_entity_id is not None:
            clauses.append("world_owner_entity_id = ?")
            params.append(int(world_owner_entity_id))
        if statuses:
            placeholders = ", ".join("?" for _ in statuses)
            clauses.append(f"status IN ({placeholders})")
            params.extend(str(status) for status in statuses)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        rows = self._conn.execute(
            f"SELECT * FROM {table}{where} ORDER BY updated_at DESC, id DESC LIMIT ?",
            (*params, max(1, int(limit or 120))),
        ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def get_intent_object(
        self, *, object_type: str, object_id: int
    ) -> Optional[Dict[str, Any]]:
        table, _columns = self._intent_object_spec(object_type)
        row = self._conn.execute(
            f"SELECT * FROM {table} WHERE id = ?", (int(object_id),)
        ).fetchone()
        return self._row_to_dict(row) if row else None

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
                or evidence_type not in {"fact", "episode"}
                or object_id <= 0 or evidence_id <= 0
            ):
                continue
            self._conn.execute(
                """
                INSERT INTO memory_intent_evidence (
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

    def insert_intent_event(
        self,
        *,
        object_type: str,
        object_id: int,
        event_type: str,
        previous_status: str = "",
        new_status: str = "",
        previous_payload: Optional[Dict[str, Any]] = None,
        new_payload: Optional[Dict[str, Any]] = None,
        evidence_fact_ids: Optional[Sequence[int]] = None,
        effective_at: str = "",
        decision_source: str = "",
        reason: str = "",
    ) -> int:
        normalized_type = str(object_type or "").strip().lower()
        if normalized_type not in {"goal", "plan", "work_item"}:
            return 0
        fact_ids = [int(value) for value in (evidence_fact_ids or []) if str(value).strip().isdigit()]
        cur = self._conn.execute(
            """
            INSERT INTO memory_intent_events (
                object_type, object_id, event_type, previous_status, new_status,
                previous_payload, new_payload, evidence_fact_ids, effective_at,
                decision_source, reason, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (normalized_type, int(object_id), str(event_type or "update"),
             str(previous_status or ""), str(new_status or ""),
             _json_dumps(previous_payload or {}), _json_dumps(new_payload or {}),
             _json_dumps(list(dict.fromkeys(fact_ids))), str(effective_at or ""),
             str(decision_source or ""), str(reason or ""), local_now_text()),
        )
        self._commit_if_needed()
        return int(cur.lastrowid)

    def upsert_intent_entities(
        self,
        *,
        object_type: str,
        object_id: int,
        entities: Sequence[Dict[str, Any]],
    ) -> int:
        normalized_type = str(object_type or "").strip().lower()
        table = {"plan": "memory_plan_entities", "work_item": "memory_work_item_entities"}.get(normalized_type)
        id_column = "plan_id" if normalized_type == "plan" else "work_item_id"
        if not table or int(object_id or 0) <= 0:
            return 0
        now = local_now_text()
        changed = 0
        for entity in entities or []:
            try:
                entity_id = int(entity["entity_id"])
            except (KeyError, TypeError, ValueError):
                continue
            role = str(entity.get("role") or "").strip().lower()
            if entity_id <= 0 or not role:
                continue
            self._conn.execute(
                f"""
                INSERT INTO {table} ({id_column}, entity_id, role, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT({id_column}, entity_id, role) DO UPDATE SET updated_at = excluded.updated_at
                """,
                (int(object_id), entity_id, role, now, now),
            )
            changed += 1
        self._commit_if_needed()
        return changed

    def upsert_goal_work_item_mapping(self, *, goal_id: int, work_item_id: int, relation: str) -> bool:
        if int(goal_id or 0) <= 0 or int(work_item_id or 0) <= 0:
            return False
        now = local_now_text()
        self._conn.execute(
            """
            INSERT INTO memory_goal_work_item_mappings (goal_id, work_item_id, relation, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(goal_id, work_item_id, relation) DO UPDATE SET updated_at = excluded.updated_at
            """,
            (int(goal_id), int(work_item_id), str(relation or "advances"), now, now),
        )
        self._commit_if_needed()
        return True

    def upsert_plan_work_item_mapping(self, *, plan_id: int, work_item_id: int, relation: str) -> bool:
        if int(plan_id or 0) <= 0 or int(work_item_id or 0) <= 0:
            return False
        now = local_now_text()
        self._conn.execute(
            """
            INSERT INTO memory_plan_work_item_mappings (plan_id, work_item_id, relation, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(plan_id, work_item_id, relation) DO UPDATE SET updated_at = excluded.updated_at
            """,
            (int(plan_id), int(work_item_id), str(relation or "prepares"), now, now),
        )
        self._commit_if_needed()
        return True

    def get_entity_claim_events(
        self,
        claim_id: int,
        *,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Return status-history events for one claim, newest event first."""
        rows = self._conn.execute(
            """
            SELECT * FROM memory_entity_claim_events
            WHERE target_claim_id = ?
            ORDER BY effective_at DESC, id DESC
            LIMIT ?
            """,
            (int(claim_id), max(1, int(limit or 100))),
        ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def get_entity_claims(
        self,
        *,
        entity_id: Optional[int] = None,
        subject_entity_id: Optional[int] = None,
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
            f"SELECT * FROM memory_entity_claims{where} ORDER BY updated_at DESC, id DESC LIMIT ?",
            (*params, max(1, int(limit or 200))),
        ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def get_entity_claims_by_ids(
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
            f"SELECT * FROM memory_entity_claims WHERE id IN ({placeholders})",
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

        Episodes intentionally remain valid projection rows but callers choose
        whether they are a direct-recall type.  This method does not impose
        that product policy itself.  A requested time range first retrieves
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

    def get_entity_claim_evidence_fact_ids(
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
            FROM memory_entity_claim_evidence
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
            FROM memory_intent_evidence
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

    def upsert_entity_claim_evidence(self, evidence: Sequence[Dict[str, Any]]) -> int:
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
            if claim_id <= 0 or evidence_id <= 0 or evidence_type not in {"fact", "episode"}:
                continue
            self._conn.execute(
                """
                INSERT INTO memory_entity_claim_evidence (
                    claim_id, evidence_type, evidence_id, role, weight, observed_at,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(claim_id, evidence_type, evidence_id, role) DO UPDATE SET
                    weight = MAX(memory_entity_claim_evidence.weight, excluded.weight),
                    observed_at = excluded.observed_at, updated_at = excluded.updated_at
                """,
                (claim_id, evidence_type, evidence_id, role, float(item.get("weight") or 1.0),
                 str(item.get("observed_at") or ""), now, now),
            )
            changed += 1
        self._commit_if_needed()
        return changed

    def get_entity_claim_support_fact_summary(self, claim_id: int) -> Dict[str, Any]:
        """Aggregate unique support facts so incremental induction never shrinks counts."""
        row = self._conn.execute(
            """
            SELECT
                COUNT(DISTINCT evidence_id) AS support_count,
                MIN(NULLIF(observed_at, '')) AS first_observed_at,
                MAX(NULLIF(observed_at, '')) AS last_observed_at
            FROM memory_entity_claim_evidence
            WHERE claim_id = ? AND evidence_type = 'fact' AND role = 'support'
            """,
            (int(claim_id),),
        ).fetchone()
        return {
            "support_count": int(row["support_count"] or 0) if row else 0,
            "first_observed_at": str(row["first_observed_at"] or "") if row else "",
            "last_observed_at": str(row["last_observed_at"] or "") if row else "",
        }

    def upsert_entity_claim_induction(
        self,
        *,
        claim_id: int,
        condition_text: str,
        support_count: int,
        first_observed_at: str,
        last_observed_at: str,
        consolidation_version: str = "v1",
    ) -> None:
        now = local_now_text()
        self._conn.execute(
            """
            INSERT INTO memory_entity_claim_induction (
                claim_id, condition_text, support_count,
                first_observed_at, last_observed_at,
                consolidation_version, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(claim_id) DO UPDATE SET
                condition_text = excluded.condition_text,
                support_count = excluded.support_count,
                first_observed_at = excluded.first_observed_at,
                last_observed_at = excluded.last_observed_at,
                consolidation_version = excluded.consolidation_version,
                updated_at = excluded.updated_at
            """,
            (int(claim_id), str(condition_text or ""),
             max(0, int(support_count or 0)),
             str(first_observed_at or ""), str(last_observed_at or ""),
             str(consolidation_version or "v1"), now, now),
        )
        self._commit_if_needed()

    def upsert_entity_claim_derivation(
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
            INSERT INTO memory_entity_claim_derivations (
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
            "SELECT id FROM memory_entity_claim_derivations WHERE derivation_key = ?",
            (str(derivation_key or ""),),
        ).fetchone()
        derivation_id = int(row["id"]) if row else 0
        self._commit_if_needed()
        return derivation_id

    def replace_entity_claim_derivation_premises(
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
            "DELETE FROM memory_entity_claim_premises WHERE derivation_id = ?",
            (int(derivation_id),),
        )
        self._conn.executemany(
            """
            INSERT INTO memory_entity_claim_premises (
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

    def invalidate_entity_claim_derivations_for_premises(
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
            FROM memory_entity_claim_derivations AS derivation
            JOIN memory_entity_claim_premises AS premise
              ON premise.derivation_id = derivation.id
            JOIN memory_entity_claims AS premise_claim
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
            UPDATE memory_entity_claim_derivations
            SET status = 'invalidated', invalidated_at = ?, updated_at = ?
            WHERE id IN ({derivation_placeholders})
            """,
            (now, now, *derivation_ids),
        )
        self._commit_if_needed()
        return list(dict.fromkeys(int(row["claim_id"]) for row in rows))

    def get_entity_claim_ids_without_active_derivations(
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
            FROM memory_entity_claims AS claim
            WHERE claim.id IN ({placeholders})
              AND claim.claim_origin = 'derived'
              AND claim.status = 'active'
              AND NOT EXISTS (
                  SELECT 1
                  FROM memory_entity_claim_derivations AS derivation
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
        episode_id: Optional[int],
        fact_type: str,
        summary: str,
        keywords: Sequence[str],
        entities: Sequence[str],
        entity_ids: Optional[Sequence[int]],
        fact_root_topic: str,
        fact_aspect_topic: str,
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
                episode_id, fact_type,
                summary, keywords, entities, entity_ids, fact_root_topic,
                fact_aspect_topic, event_time_key, dialogue_time_key,
                confidence, importance, metadata, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                episode_id,
                fact_type,
                summary,
                _json_dumps(keyword_values),
                _json_dumps(list(entities or [])),
                _json_dumps([int(value) for value in entity_ids or []]),
                str(fact_root_topic or ""),
                str(fact_aspect_topic or ""),
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
        if episode_id is not None:
            self.insert_fact_episode_mappings([
                {
                    "fact_id": fact_id,
                    "episode_id": int(episode_id),
                }
            ])
        self.insert_entity_memory_mappings([
            {
                "entity_id": int(entity_id),
                "episode_id": [episode_id] if episode_id is not None else [],
                "fact_id": [fact_id],
            }
            for entity_id in entity_ids or []
        ])
        self._commit_if_needed()
        return fact_id

    def add_entity_names(self, names: Iterable[str]) -> Dict[str, int]:
        now = local_now_text()
        normalized: List[str] = []
        for name in names:
            clean = str(name or "").strip()
            if not clean or clean in normalized:
                continue
            normalized.append(clean)
            self._conn.execute(
                "INSERT OR IGNORE INTO memory_entity_nodes (name, type, created_at) VALUES (?, ?, ?)",
                (clean, "OTHER", now),
            )
        self._commit_if_needed()
        if not normalized:
            return {}
        placeholders = ",".join("?" for _ in normalized)
        rows = self._conn.execute(
            f"SELECT id, name FROM memory_entity_nodes WHERE name IN ({placeholders})",
            normalized,
        ).fetchall()
        return {str(row["name"]): int(row["id"]) for row in rows}

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
            SELECT entity_id, episode_id, fact_id
            FROM memory_entity_mapping
            WHERE entity_id IN ({placeholders})
            """,
            ids,
        ).fetchall()
        return [
            {
                "entity_id": int(row["entity_id"]),
                "episode_id": _json_loads(row["episode_id"], []),
                "fact_id": _json_loads(row["fact_id"], []),
            }
            for row in rows
        ]

    def insert_entity_memory_mappings(
        self,
        mappings: Sequence[Dict[str, Any]],
    ) -> int:
        """Merge fact and episode links into one row per entity."""
        if not mappings:
            return 0

        mapping_fields = (
            "episode_id",
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
                SELECT episode_id, fact_id
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
                    SET episode_id = ?, fact_id = ?, updated_at = ?
                    WHERE entity_id = ?
                    """,
                    (
                        _json_dumps(merged["episode_id"]),
                        _json_dumps(merged["fact_id"]),
                        now,
                        entity_id,
                    ),
                )
            else:
                self._conn.execute(
                    """
                    INSERT INTO memory_entity_mapping (
                        entity_id, episode_id, fact_id, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        entity_id,
                        _json_dumps(merged["episode_id"]),
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

    def memory_episodes_by_ids(self, episode_ids: Sequence[int]) -> List[Dict[str, Any]]:
        ids = [int(value) for value in episode_ids if value is not None]
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        rows = self._conn.execute(
            f"SELECT * FROM memory_episodes WHERE id IN ({placeholders})",
            ids,
        ).fetchall()
        by_id = {int(row["id"]): self._row_to_dict(row) for row in rows}
        return [by_id[item] for item in ids if item in by_id]

    def memory_facts_by_episode_ids(
        self,
        episode_ids: Sequence[int],
        *,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        """Return facts belonging to a bounded set of episodes."""
        ids = list(dict.fromkeys(
            int(value)
            for value in episode_ids
            if value is not None
        ))
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
            WHERE fact.episode_id IN ({placeholders})
            ORDER BY fact.dialogue_time_key ASC, fact.id ASC
            LIMIT ?
            """,
            (*ids, max(1, int(limit or 200))),
        ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def memory_episode_facts_for_entity_id(
        self,
        entity_id: int,
        *,
        limit: int = 240,
    ) -> List[Dict[str, Any]]:
        """Load completed-episode facts directly linked to one entity."""
        entity_token = f"%,{int(entity_id)},%"
        rows = self._conn.execute(
            """
            SELECT fact.*, document.identity_text,
                   document.identity_text_embedding
            FROM memory_facts AS fact
            LEFT JOIN memory_recall_documents AS document
                ON document.object_type = 'fact' AND document.object_id = fact.id
            WHERE fact.episode_id IS NOT NULL
              AND (',' || replace(replace(replace(replace(fact.entity_ids, ' ', ''), '\n', ''), '[', ''), ']', '') || ',') LIKE ?
            ORDER BY fact.dialogue_time_key ASC, fact.id ASC
            LIMIT ?
            """,
            (entity_token, max(1, int(limit or 240))),
        ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def related_fact_pairs_by_episode_fact_ids(
        self,
        fact_ids: Sequence[int],
        *,
        limit: int = 200,
    ) -> List[Dict[str, int]]:
        """Return other facts sharing an episode with each supplied fact."""
        ids = list(dict.fromkeys(
            int(value)
            for value in fact_ids or []
            if str(value).strip().isdigit() and int(value) > 0
        ))
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        rows = self._conn.execute(
            f"""
            SELECT seed.fact_id AS seed_fact_id,
                   related.fact_id AS related_fact_id
            FROM memory_fact_episode_mapping AS seed
            INNER JOIN memory_fact_episode_mapping AS related
                ON related.episode_id = seed.episode_id
            WHERE seed.fact_id IN ({placeholders})
              AND related.fact_id != seed.fact_id
            ORDER BY seed.fact_id ASC, related.fact_id ASC
            LIMIT ?
            """,
            (*ids, max(1, int(limit or 200))),
        ).fetchall()
        return [
            {
                "seed_fact_id": int(row["seed_fact_id"]),
                "related_fact_id": int(row["related_fact_id"]),
            }
            for row in rows
        ]

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
            "details",
            "previous_payload",
            "new_payload",
            "evidence_fact_ids",
            "fact_ids",
            "episode_ids",
        ):
            if key in item:
                item[key] = _json_loads(
                    item[key],
                    {} if key in {"metadata", "details"} else [],
                )
        for key in (
            "embedding",
            "identity_text_embedding",
        ):
            if key in item:
                item[key] = _blob_to_embedding(item[key])
        return item
