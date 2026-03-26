"""
Semantic Gravity Memory — SQLite Storage Backend

Full implementation of BaseStorage using only sqlite3 (stdlib).
Thread-safe writes via threading.Lock. WAL mode for concurrent reads.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from typing import Any, Dict, List, Optional, Tuple

from semantic_gravity_memory.models import (
    Activation,
    AntibodyMemory,
    Contradiction,
    Crystal,
    Entity,
    Event,
    ProspectiveMemory,
    Relation,
    SalienceVector,
    Schema,
)
from semantic_gravity_memory.storage.base import BaseStorage
from semantic_gravity_memory.utils import now_iso, safe_json_dumps, safe_json_loads


class SQLiteBackend(BaseStorage):
    """SQLite-backed memory storage. Accepts ':memory:' for testing."""

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self._migrate()

    # -----------------------------------------------------------------
    # Schema migration
    # -----------------------------------------------------------------

    def _migrate(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS events (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                ts            TEXT    NOT NULL,
                actor         TEXT    NOT NULL,
                kind          TEXT    NOT NULL,
                content       TEXT    NOT NULL,
                context_json  TEXT    DEFAULT '{}',
                salience      REAL    DEFAULT 0,
                embedding_json TEXT
            );

            CREATE TABLE IF NOT EXISTS entities (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                name          TEXT    NOT NULL UNIQUE,
                kind          TEXT    NOT NULL DEFAULT 'concept',
                first_seen_ts TEXT    NOT NULL,
                last_seen_ts  TEXT    NOT NULL,
                salience      REAL    DEFAULT 0,
                mention_count INTEGER DEFAULT 0,
                metadata_json TEXT    DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS crystals (
                id                   INTEGER PRIMARY KEY AUTOINCREMENT,
                created_ts           TEXT    NOT NULL,
                updated_ts           TEXT    NOT NULL,
                title                TEXT    NOT NULL,
                theme                TEXT    NOT NULL,
                summary              TEXT    NOT NULL,
                compressed_narrative TEXT    DEFAULT '',
                source_event_ids_json TEXT   DEFAULT '[]',
                entity_ids_json      TEXT    DEFAULT '[]',
                salience_json        TEXT    DEFAULT '{}',
                confidence           REAL    DEFAULT 0.5,
                self_state           TEXT    DEFAULT 'general',
                future_implications  TEXT    DEFAULT '',
                unresolved           TEXT    DEFAULT '',
                contradiction_state  TEXT    DEFAULT 'clean',
                valid_from_ts        TEXT,
                valid_to_ts          TEXT,
                embedding_json       TEXT,
                memory_type          TEXT    DEFAULT 'episodic',
                access_count         INTEGER DEFAULT 0,
                last_accessed_ts     TEXT,
                decay_rate           REAL    DEFAULT 0.1,
                version              INTEGER DEFAULT 1,
                parent_crystal_id    INTEGER,
                schema_id            INTEGER
            );

            CREATE TABLE IF NOT EXISTS relations (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                source_type  TEXT    NOT NULL,
                source_id    INTEGER NOT NULL,
                target_type  TEXT    NOT NULL,
                target_id    INTEGER NOT NULL,
                relation     TEXT    NOT NULL,
                weight       REAL    DEFAULT 0.5,
                context_json TEXT    DEFAULT '{}',
                created_ts   TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS contradictions (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                ts                TEXT    NOT NULL,
                topic             TEXT    NOT NULL,
                claim_a           TEXT    NOT NULL,
                claim_b           TEXT    NOT NULL,
                evidence_event_a  INTEGER,
                evidence_event_b  INTEGER,
                resolution_state  TEXT    DEFAULT 'open',
                resolution_ts     TEXT,
                notes             TEXT    DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS activations (
                id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                ts                 TEXT    NOT NULL,
                query              TEXT    NOT NULL,
                active_self_state  TEXT    DEFAULT 'general',
                crystal_ids_json   TEXT    DEFAULT '[]',
                entity_ids_json    TEXT    DEFAULT '[]',
                scene_json         TEXT    DEFAULT '{}',
                quality_score      REAL
            );

            CREATE TABLE IF NOT EXISTS prospective_memories (
                id                     INTEGER PRIMARY KEY AUTOINCREMENT,
                created_ts             TEXT    NOT NULL,
                trigger_description    TEXT    NOT NULL,
                trigger_embedding_json TEXT,
                payload_crystal_id     INTEGER NOT NULL,
                expiry_ts              TEXT,
                fired                  INTEGER DEFAULT 0,
                fired_ts               TEXT
            );

            CREATE TABLE IF NOT EXISTS schemas (
                id                     INTEGER PRIMARY KEY AUTOINCREMENT,
                created_ts             TEXT    NOT NULL,
                updated_ts             TEXT    NOT NULL,
                name                   TEXT    NOT NULL,
                description            TEXT    DEFAULT '',
                pattern                TEXT    DEFAULT '',
                source_crystal_ids_json TEXT   DEFAULT '[]',
                slot_definitions_json  TEXT    DEFAULT '{}',
                activation_count       INTEGER DEFAULT 0,
                embedding_json         TEXT
            );

            CREATE TABLE IF NOT EXISTS antibodies (
                id                     INTEGER PRIMARY KEY AUTOINCREMENT,
                created_ts             TEXT    NOT NULL,
                trigger_description    TEXT    NOT NULL,
                trigger_embedding_json TEXT,
                suppress_crystal_id    INTEGER NOT NULL,
                reason                 TEXT    DEFAULT '',
                active                 INTEGER DEFAULT 1
            );

            -- Indexes
            CREATE INDEX IF NOT EXISTS idx_events_ts        ON events(ts DESC);
            CREATE INDEX IF NOT EXISTS idx_entities_name     ON entities(name);
            CREATE INDEX IF NOT EXISTS idx_entities_salience ON entities(salience DESC);
            CREATE INDEX IF NOT EXISTS idx_crystals_ts       ON crystals(created_ts DESC);
            CREATE INDEX IF NOT EXISTS idx_crystals_type     ON crystals(memory_type);
            CREATE INDEX IF NOT EXISTS idx_relations_src     ON relations(source_type, source_id);
            CREATE INDEX IF NOT EXISTS idx_relations_tgt     ON relations(target_type, target_id);
            CREATE INDEX IF NOT EXISTS idx_contradictions_st ON contradictions(resolution_state);
            CREATE INDEX IF NOT EXISTS idx_activations_ts    ON activations(ts DESC);
            CREATE INDEX IF NOT EXISTS idx_prospective_fired ON prospective_memories(fired);
            CREATE INDEX IF NOT EXISTS idx_antibodies_active ON antibodies(active);
            """
        )
        self.conn.commit()
        # Gravitational mass column (added in v0.2 — safe to re-run)
        try:
            self.conn.execute("ALTER TABLE crystals ADD COLUMN grav_mass REAL DEFAULT 0")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_crystals_mass ON crystals(grav_mass DESC)")
            self.conn.commit()
        except sqlite3.OperationalError:
            pass  # Column already exists

    # -----------------------------------------------------------------
    # Row ↔ Model converters
    # -----------------------------------------------------------------

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> Event:
        return Event(
            id=row["id"],
            ts=row["ts"],
            actor=row["actor"],
            kind=row["kind"],
            content=row["content"],
            context=safe_json_loads(row["context_json"], {}),
            salience=float(row["salience"] or 0),
            embedding=safe_json_loads(row["embedding_json"]),
        )

    @staticmethod
    def _row_to_entity(row: sqlite3.Row) -> Entity:
        return Entity(
            id=row["id"],
            name=row["name"],
            kind=row["kind"],
            first_seen_ts=row["first_seen_ts"],
            last_seen_ts=row["last_seen_ts"],
            salience=float(row["salience"] or 0),
            mention_count=int(row["mention_count"] or 0),
            metadata=safe_json_loads(row["metadata_json"], {}),
        )

    @staticmethod
    def _row_to_crystal(row: sqlite3.Row) -> Crystal:
        return Crystal(
            id=row["id"],
            created_ts=row["created_ts"],
            updated_ts=row["updated_ts"],
            title=row["title"],
            theme=row["theme"],
            summary=row["summary"],
            compressed_narrative=row["compressed_narrative"] or "",
            source_event_ids=safe_json_loads(row["source_event_ids_json"], []),
            entity_ids=safe_json_loads(row["entity_ids_json"], []),
            salience=SalienceVector.from_dict(safe_json_loads(row["salience_json"], {})),
            confidence=float(row["confidence"] or 0.5),
            self_state=row["self_state"] or "general",
            future_implications=row["future_implications"] or "",
            unresolved=row["unresolved"] or "",
            contradiction_state=row["contradiction_state"] or "clean",
            valid_from_ts=row["valid_from_ts"],
            valid_to_ts=row["valid_to_ts"],
            embedding=safe_json_loads(row["embedding_json"]),
            memory_type=row["memory_type"] or "episodic",
            access_count=int(row["access_count"] or 0),
            last_accessed_ts=row["last_accessed_ts"],
            decay_rate=float(row["decay_rate"] if row["decay_rate"] is not None else 0.1),
            version=int(row["version"] or 1),
            parent_crystal_id=row["parent_crystal_id"],
            schema_id=row["schema_id"],
        )

    @staticmethod
    def _row_to_relation(row: sqlite3.Row) -> Relation:
        return Relation(
            id=row["id"],
            source_type=row["source_type"],
            source_id=int(row["source_id"]),
            target_type=row["target_type"],
            target_id=int(row["target_id"]),
            relation=row["relation"],
            weight=float(row["weight"] or 0.5),
            context=safe_json_loads(row["context_json"], {}),
            created_ts=row["created_ts"],
        )

    @staticmethod
    def _row_to_contradiction(row: sqlite3.Row) -> Contradiction:
        return Contradiction(
            id=row["id"],
            ts=row["ts"],
            topic=row["topic"],
            claim_a=row["claim_a"],
            claim_b=row["claim_b"],
            evidence_event_a=row["evidence_event_a"],
            evidence_event_b=row["evidence_event_b"],
            resolution_state=row["resolution_state"] or "open",
            resolution_ts=row["resolution_ts"],
            notes=row["notes"] or "",
        )

    @staticmethod
    def _row_to_activation(row: sqlite3.Row) -> Activation:
        return Activation(
            id=row["id"],
            ts=row["ts"],
            query=row["query"],
            active_self_state=row["active_self_state"] or "general",
            crystal_ids=safe_json_loads(row["crystal_ids_json"], []),
            entity_ids=safe_json_loads(row["entity_ids_json"], []),
            scene=safe_json_loads(row["scene_json"], {}),
            quality_score=row["quality_score"],
        )

    @staticmethod
    def _row_to_prospective(row: sqlite3.Row) -> ProspectiveMemory:
        return ProspectiveMemory(
            id=row["id"],
            created_ts=row["created_ts"],
            trigger_description=row["trigger_description"],
            trigger_embedding=safe_json_loads(row["trigger_embedding_json"]),
            payload_crystal_id=int(row["payload_crystal_id"]),
            expiry_ts=row["expiry_ts"],
            fired=bool(row["fired"]),
            fired_ts=row["fired_ts"],
        )

    @staticmethod
    def _row_to_schema(row: sqlite3.Row) -> Schema:
        return Schema(
            id=row["id"],
            created_ts=row["created_ts"],
            updated_ts=row["updated_ts"],
            name=row["name"],
            description=row["description"] or "",
            pattern=row["pattern"] or "",
            source_crystal_ids=safe_json_loads(row["source_crystal_ids_json"], []),
            slot_definitions=safe_json_loads(row["slot_definitions_json"], {}),
            activation_count=int(row["activation_count"] or 0),
            embedding=safe_json_loads(row["embedding_json"]),
        )

    @staticmethod
    def _row_to_antibody(row: sqlite3.Row) -> AntibodyMemory:
        return AntibodyMemory(
            id=row["id"],
            created_ts=row["created_ts"],
            trigger_description=row["trigger_description"],
            trigger_embedding=safe_json_loads(row["trigger_embedding_json"]),
            suppress_crystal_id=int(row["suppress_crystal_id"]),
            reason=row["reason"] or "",
            active=bool(row["active"]),
        )

    # -----------------------------------------------------------------
    # Events
    # -----------------------------------------------------------------

    def insert_event(self, event: Event) -> int:
        ts = event.ts or now_iso()
        with self._lock:
            cur = self.conn.execute(
                """INSERT INTO events (ts, actor, kind, content, context_json, salience, embedding_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    ts,
                    event.actor,
                    event.kind,
                    event.content,
                    safe_json_dumps(event.context),
                    event.salience,
                    safe_json_dumps(event.embedding) if event.embedding else None,
                ),
            )
            self.conn.commit()
            return int(cur.lastrowid)  # type: ignore[arg-type]

    def get_event(self, event_id: int) -> Optional[Event]:
        row = self.conn.execute("SELECT * FROM events WHERE id=?", (event_id,)).fetchone()
        return self._row_to_event(row) if row else None

    def recent_events(self, limit: int = 50) -> List[Event]:
        rows = self.conn.execute("SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [self._row_to_event(r) for r in rows]

    # -----------------------------------------------------------------
    # Entities
    # -----------------------------------------------------------------

    def upsert_entity(self, entity: Entity) -> int:
        ts = entity.last_seen_ts or now_iso()
        with self._lock:
            existing = self.conn.execute("SELECT * FROM entities WHERE name=?", (entity.name,)).fetchone()
            if existing:
                merged_meta = safe_json_loads(existing["metadata_json"], {})
                merged_meta.update(entity.metadata)
                new_salience = float(existing["salience"] or 0) + entity.salience
                new_count = int(existing["mention_count"] or 0) + max(entity.mention_count, 1)
                self.conn.execute(
                    """UPDATE entities
                       SET last_seen_ts=?, salience=?, mention_count=?, metadata_json=?, kind=?
                       WHERE id=?""",
                    (ts, new_salience, new_count, safe_json_dumps(merged_meta), entity.kind or existing["kind"], existing["id"]),
                )
                self.conn.commit()
                return int(existing["id"])
            first_ts = entity.first_seen_ts or ts
            cur = self.conn.execute(
                """INSERT INTO entities (name, kind, first_seen_ts, last_seen_ts, salience, mention_count, metadata_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (entity.name, entity.kind, first_ts, ts, entity.salience, max(entity.mention_count, 1), safe_json_dumps(entity.metadata)),
            )
            self.conn.commit()
            return int(cur.lastrowid)  # type: ignore[arg-type]

    def get_entity(self, entity_id: int) -> Optional[Entity]:
        row = self.conn.execute("SELECT * FROM entities WHERE id=?", (entity_id,)).fetchone()
        return self._row_to_entity(row) if row else None

    def get_entity_by_name(self, name: str) -> Optional[Entity]:
        row = self.conn.execute("SELECT * FROM entities WHERE name=?", (name,)).fetchone()
        return self._row_to_entity(row) if row else None

    def top_entities(self, limit: int = 50) -> List[Entity]:
        rows = self.conn.execute(
            "SELECT * FROM entities ORDER BY salience DESC, last_seen_ts DESC LIMIT ?", (limit,)
        ).fetchall()
        return [self._row_to_entity(r) for r in rows]

    # -----------------------------------------------------------------
    # Crystals
    # -----------------------------------------------------------------

    def insert_crystal(self, crystal: Crystal) -> int:
        ts = crystal.created_ts or now_iso()
        with self._lock:
            cur = self.conn.execute(
                """INSERT INTO crystals (
                       created_ts, updated_ts, title, theme, summary, compressed_narrative,
                       source_event_ids_json, entity_ids_json, salience_json,
                       confidence, self_state, future_implications, unresolved,
                       contradiction_state, valid_from_ts, valid_to_ts, embedding_json,
                       memory_type, access_count, last_accessed_ts, decay_rate,
                       version, parent_crystal_id, schema_id
                   ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    ts,
                    crystal.updated_ts or ts,
                    crystal.title,
                    crystal.theme,
                    crystal.summary,
                    crystal.compressed_narrative,
                    safe_json_dumps(crystal.source_event_ids),
                    safe_json_dumps(crystal.entity_ids),
                    safe_json_dumps(crystal.salience.to_dict()),
                    crystal.confidence,
                    crystal.self_state,
                    crystal.future_implications,
                    crystal.unresolved,
                    crystal.contradiction_state,
                    crystal.valid_from_ts,
                    crystal.valid_to_ts,
                    safe_json_dumps(crystal.embedding) if crystal.embedding else None,
                    crystal.memory_type,
                    crystal.access_count,
                    crystal.last_accessed_ts,
                    crystal.decay_rate,
                    crystal.version,
                    crystal.parent_crystal_id,
                    crystal.schema_id,
                ),
            )
            self.conn.commit()
            return int(cur.lastrowid)  # type: ignore[arg-type]

    def update_crystal(self, crystal: Crystal) -> None:
        if crystal.id is None:
            raise ValueError("Cannot update crystal without an id")
        with self._lock:
            self.conn.execute(
                """UPDATE crystals SET
                       updated_ts=?, title=?, theme=?, summary=?, compressed_narrative=?,
                       source_event_ids_json=?, entity_ids_json=?, salience_json=?,
                       confidence=?, self_state=?, future_implications=?, unresolved=?,
                       contradiction_state=?, valid_from_ts=?, valid_to_ts=?, embedding_json=?,
                       memory_type=?, access_count=?, last_accessed_ts=?, decay_rate=?,
                       version=?, parent_crystal_id=?, schema_id=?
                   WHERE id=?""",
                (
                    crystal.updated_ts or now_iso(),
                    crystal.title,
                    crystal.theme,
                    crystal.summary,
                    crystal.compressed_narrative,
                    safe_json_dumps(crystal.source_event_ids),
                    safe_json_dumps(crystal.entity_ids),
                    safe_json_dumps(crystal.salience.to_dict()),
                    crystal.confidence,
                    crystal.self_state,
                    crystal.future_implications,
                    crystal.unresolved,
                    crystal.contradiction_state,
                    crystal.valid_from_ts,
                    crystal.valid_to_ts,
                    safe_json_dumps(crystal.embedding) if crystal.embedding else None,
                    crystal.memory_type,
                    crystal.access_count,
                    crystal.last_accessed_ts,
                    crystal.decay_rate,
                    crystal.version,
                    crystal.parent_crystal_id,
                    crystal.schema_id,
                    crystal.id,
                ),
            )
            self.conn.commit()

    def get_crystal(self, crystal_id: int) -> Optional[Crystal]:
        row = self.conn.execute("SELECT * FROM crystals WHERE id=?", (crystal_id,)).fetchone()
        return self._row_to_crystal(row) if row else None

    def all_crystals(self) -> List[Crystal]:
        rows = self.conn.execute("SELECT * FROM crystals ORDER BY id").fetchall()
        return [self._row_to_crystal(r) for r in rows]

    def recent_crystals(self, limit: int = 50) -> List[Crystal]:
        rows = self.conn.execute(
            "SELECT * FROM crystals ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [self._row_to_crystal(r) for r in rows]

    # -----------------------------------------------------------------
    # Gravitational retrieval
    # -----------------------------------------------------------------

    def crystals_by_entity_ids(self, entity_ids: List[int], limit: int = 50) -> List[Crystal]:
        if not entity_ids:
            return []
        ph = ",".join("?" * len(entity_ids))
        rows = self.conn.execute(
            f"""SELECT DISTINCT c.* FROM crystals c
                JOIN relations r ON (
                    (r.source_type='crystal' AND r.source_id=c.id
                     AND r.target_type='entity' AND r.target_id IN ({ph}))
                    OR
                    (r.target_type='crystal' AND r.target_id=c.id
                     AND r.source_type='entity' AND r.source_id IN ({ph}))
                )
                WHERE c.valid_to_ts IS NULL
                ORDER BY c.id DESC LIMIT ?""",
            (*entity_ids, *entity_ids, limit),
        ).fetchall()
        return [self._row_to_crystal(r) for r in rows]

    def top_crystals_by_mass(self, limit: int = 50) -> List[Crystal]:
        rows = self.conn.execute(
            "SELECT * FROM crystals WHERE valid_to_ts IS NULL ORDER BY grav_mass DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._row_to_crystal(r) for r in rows]

    def update_crystal_masses(self, mass_map: Dict[int, float]) -> None:
        if not mass_map:
            return
        with self._lock:
            for cid, mass in mass_map.items():
                self.conn.execute("UPDATE crystals SET grav_mass=? WHERE id=?", (mass, cid))
            self.conn.commit()

    def entity_names_and_ids(self) -> List[Tuple[int, str]]:
        rows = self.conn.execute(
            "SELECT id, name FROM entities ORDER BY salience DESC"
        ).fetchall()
        return [(int(r["id"]), r["name"]) for r in rows]

    # -----------------------------------------------------------------
    # Relations
    # -----------------------------------------------------------------

    def insert_relation(self, relation: Relation) -> int:
        ts = relation.created_ts or now_iso()
        with self._lock:
            cur = self.conn.execute(
                """INSERT INTO relations (source_type, source_id, target_type, target_id, relation, weight, context_json, created_ts)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (
                    relation.source_type,
                    relation.source_id,
                    relation.target_type,
                    relation.target_id,
                    relation.relation,
                    relation.weight,
                    safe_json_dumps(relation.context),
                    ts,
                ),
            )
            self.conn.commit()
            return int(cur.lastrowid)  # type: ignore[arg-type]

    def relations_from(self, source_type: str, source_id: int) -> List[Relation]:
        rows = self.conn.execute(
            "SELECT * FROM relations WHERE source_type=? AND source_id=?",
            (source_type, source_id),
        ).fetchall()
        return [self._row_to_relation(r) for r in rows]

    def relations_to(self, target_type: str, target_id: int) -> List[Relation]:
        rows = self.conn.execute(
            "SELECT * FROM relations WHERE target_type=? AND target_id=?",
            (target_type, target_id),
        ).fetchall()
        return [self._row_to_relation(r) for r in rows]

    def all_relations(self) -> List[Relation]:
        rows = self.conn.execute("SELECT * FROM relations ORDER BY id").fetchall()
        return [self._row_to_relation(r) for r in rows]

    # -----------------------------------------------------------------
    # Contradictions
    # -----------------------------------------------------------------

    def insert_contradiction(self, contradiction: Contradiction) -> int:
        ts = contradiction.ts or now_iso()
        with self._lock:
            cur = self.conn.execute(
                """INSERT INTO contradictions (ts, topic, claim_a, claim_b, evidence_event_a, evidence_event_b, resolution_state, resolution_ts, notes)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    ts,
                    contradiction.topic,
                    contradiction.claim_a,
                    contradiction.claim_b,
                    contradiction.evidence_event_a,
                    contradiction.evidence_event_b,
                    contradiction.resolution_state,
                    contradiction.resolution_ts,
                    contradiction.notes,
                ),
            )
            self.conn.commit()
            return int(cur.lastrowid)  # type: ignore[arg-type]

    def update_contradiction(self, contradiction: Contradiction) -> None:
        if contradiction.id is None:
            raise ValueError("Cannot update contradiction without an id")
        with self._lock:
            self.conn.execute(
                """UPDATE contradictions SET
                       topic=?, claim_a=?, claim_b=?, evidence_event_a=?, evidence_event_b=?,
                       resolution_state=?, resolution_ts=?, notes=?
                   WHERE id=?""",
                (
                    contradiction.topic,
                    contradiction.claim_a,
                    contradiction.claim_b,
                    contradiction.evidence_event_a,
                    contradiction.evidence_event_b,
                    contradiction.resolution_state,
                    contradiction.resolution_ts,
                    contradiction.notes,
                    contradiction.id,
                ),
            )
            self.conn.commit()

    def open_contradictions(self) -> List[Contradiction]:
        rows = self.conn.execute(
            "SELECT * FROM contradictions WHERE resolution_state='open' ORDER BY id DESC"
        ).fetchall()
        return [self._row_to_contradiction(r) for r in rows]

    def all_contradictions(self) -> List[Contradiction]:
        rows = self.conn.execute("SELECT * FROM contradictions ORDER BY id").fetchall()
        return [self._row_to_contradiction(r) for r in rows]

    # -----------------------------------------------------------------
    # Activations
    # -----------------------------------------------------------------

    def insert_activation(self, activation: Activation) -> int:
        ts = activation.ts or now_iso()
        with self._lock:
            cur = self.conn.execute(
                """INSERT INTO activations (ts, query, active_self_state, crystal_ids_json, entity_ids_json, scene_json, quality_score)
                   VALUES (?,?,?,?,?,?,?)""",
                (
                    ts,
                    activation.query,
                    activation.active_self_state,
                    safe_json_dumps(activation.crystal_ids),
                    safe_json_dumps(activation.entity_ids),
                    safe_json_dumps(activation.scene),
                    activation.quality_score,
                ),
            )
            self.conn.commit()
            return int(cur.lastrowid)  # type: ignore[arg-type]

    def recent_activations(self, limit: int = 50) -> List[Activation]:
        rows = self.conn.execute(
            "SELECT * FROM activations ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [self._row_to_activation(r) for r in rows]

    # -----------------------------------------------------------------
    # Prospective Memory
    # -----------------------------------------------------------------

    def insert_prospective(self, pm: ProspectiveMemory) -> int:
        ts = pm.created_ts or now_iso()
        with self._lock:
            cur = self.conn.execute(
                """INSERT INTO prospective_memories (created_ts, trigger_description, trigger_embedding_json, payload_crystal_id, expiry_ts, fired, fired_ts)
                   VALUES (?,?,?,?,?,?,?)""",
                (
                    ts,
                    pm.trigger_description,
                    safe_json_dumps(pm.trigger_embedding) if pm.trigger_embedding else None,
                    pm.payload_crystal_id,
                    pm.expiry_ts,
                    int(pm.fired),
                    pm.fired_ts,
                ),
            )
            self.conn.commit()
            return int(cur.lastrowid)  # type: ignore[arg-type]

    def active_prospective_memories(self) -> List[ProspectiveMemory]:
        rows = self.conn.execute(
            "SELECT * FROM prospective_memories WHERE fired=0 ORDER BY id"
        ).fetchall()
        return [self._row_to_prospective(r) for r in rows]

    def fire_prospective(self, pm_id: int, fired_ts: str) -> None:
        with self._lock:
            self.conn.execute(
                "UPDATE prospective_memories SET fired=1, fired_ts=? WHERE id=?",
                (fired_ts, pm_id),
            )
            self.conn.commit()

    # -----------------------------------------------------------------
    # Schemas
    # -----------------------------------------------------------------

    def insert_schema(self, schema: Schema) -> int:
        ts = schema.created_ts or now_iso()
        with self._lock:
            cur = self.conn.execute(
                """INSERT INTO schemas (created_ts, updated_ts, name, description, pattern, source_crystal_ids_json, slot_definitions_json, activation_count, embedding_json)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    ts,
                    schema.updated_ts or ts,
                    schema.name,
                    schema.description,
                    schema.pattern,
                    safe_json_dumps(schema.source_crystal_ids),
                    safe_json_dumps(schema.slot_definitions),
                    schema.activation_count,
                    safe_json_dumps(schema.embedding) if schema.embedding else None,
                ),
            )
            self.conn.commit()
            return int(cur.lastrowid)  # type: ignore[arg-type]

    def update_schema(self, schema: Schema) -> None:
        if schema.id is None:
            raise ValueError("Cannot update schema without an id")
        with self._lock:
            self.conn.execute(
                """UPDATE schemas SET
                       updated_ts=?, name=?, description=?, pattern=?,
                       source_crystal_ids_json=?, slot_definitions_json=?,
                       activation_count=?, embedding_json=?
                   WHERE id=?""",
                (
                    schema.updated_ts or now_iso(),
                    schema.name,
                    schema.description,
                    schema.pattern,
                    safe_json_dumps(schema.source_crystal_ids),
                    safe_json_dumps(schema.slot_definitions),
                    schema.activation_count,
                    safe_json_dumps(schema.embedding) if schema.embedding else None,
                    schema.id,
                ),
            )
            self.conn.commit()

    def all_schemas(self) -> List[Schema]:
        rows = self.conn.execute("SELECT * FROM schemas ORDER BY id").fetchall()
        return [self._row_to_schema(r) for r in rows]

    # -----------------------------------------------------------------
    # Antibodies
    # -----------------------------------------------------------------

    def insert_antibody(self, antibody: AntibodyMemory) -> int:
        ts = antibody.created_ts or now_iso()
        with self._lock:
            cur = self.conn.execute(
                """INSERT INTO antibodies (created_ts, trigger_description, trigger_embedding_json, suppress_crystal_id, reason, active)
                   VALUES (?,?,?,?,?,?)""",
                (
                    ts,
                    antibody.trigger_description,
                    safe_json_dumps(antibody.trigger_embedding) if antibody.trigger_embedding else None,
                    antibody.suppress_crystal_id,
                    antibody.reason,
                    int(antibody.active),
                ),
            )
            self.conn.commit()
            return int(cur.lastrowid)  # type: ignore[arg-type]

    def active_antibodies(self) -> List[AntibodyMemory]:
        rows = self.conn.execute(
            "SELECT * FROM antibodies WHERE active=1 ORDER BY id"
        ).fetchall()
        return [self._row_to_antibody(r) for r in rows]

    # -----------------------------------------------------------------
    # Meta
    # -----------------------------------------------------------------

    def set_meta(self, key: str, value: str) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )
            self.conn.commit()

    def get_meta(self, key: str, default: str = "") -> str:
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else default

    # -----------------------------------------------------------------
    # Export
    # -----------------------------------------------------------------

    def export_all(self) -> Dict[str, Any]:
        return {
            "meta": {r["key"]: r["value"] for r in self.conn.execute("SELECT * FROM meta")},
            "events": [self._row_to_event(r).__dict__ for r in self.conn.execute("SELECT * FROM events ORDER BY id")],
            "entities": [self._row_to_entity(r).__dict__ for r in self.conn.execute("SELECT * FROM entities ORDER BY id")],
            "crystals": [self._crystal_to_export_dict(r) for r in self.conn.execute("SELECT * FROM crystals ORDER BY id")],
            "relations": [self._row_to_relation(r).__dict__ for r in self.conn.execute("SELECT * FROM relations ORDER BY id")],
            "contradictions": [self._row_to_contradiction(r).__dict__ for r in self.conn.execute("SELECT * FROM contradictions ORDER BY id")],
            "activations": [self._row_to_activation(r).__dict__ for r in self.conn.execute("SELECT * FROM activations ORDER BY id")],
            "prospective_memories": [self._row_to_prospective(r).__dict__ for r in self.conn.execute("SELECT * FROM prospective_memories ORDER BY id")],
            "schemas": [self._row_to_schema(r).__dict__ for r in self.conn.execute("SELECT * FROM schemas ORDER BY id")],
            "antibodies": [self._row_to_antibody(r).__dict__ for r in self.conn.execute("SELECT * FROM antibodies ORDER BY id")],
        }

    def _crystal_to_export_dict(self, row: sqlite3.Row) -> dict:
        c = self._row_to_crystal(row)
        d = c.__dict__.copy()
        d["salience"] = c.salience.to_dict()
        return d

    # -----------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------

    def close(self) -> None:
        self.conn.close()
