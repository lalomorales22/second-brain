#!/usr/bin/env python3
"""
Semantic Gravity Memory Lab
single-file local desktop app for experimenting with a field-based memory system using Ollama.

what it does
- sleek black tkinter gui
- creates its own sqlite database
- talks to local ollama for chat + embeddings
- stores raw events, entities, memory crystals, contradictions, relations, activations
- performs semantic recall with embeddings
- reconstructs memory scenes for answers
- visualizes the active memory graph on a canvas

quick start
1) install python 3.10+
2) install ollama and start it
3) pull a chat model and an embedding model, or use the in-app buttons
4) run: python semantic_gravity_memory_lab.py

recommended local models
- chat: gemma3
- embeddings: all-minilm or embeddinggemma

this file avoids third-party python dependencies.
"""

from __future__ import annotations

import base64
import datetime as dt
import json
import math
import os
import queue
import random
import re
import shutil
import sqlite3
import subprocess
import threading
import time
import traceback
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog
except Exception as e:
    raise SystemExit(f"tkinter is required: {e}")

APP_TITLE = "semantic gravity memory lab"
APP_DIR = os.path.join(os.path.expanduser("~"), ".semantic_gravity_memory_lab")
DB_PATH = os.path.join(APP_DIR, "memory_lab.db")
EXPORT_DIR = os.path.join(APP_DIR, "exports")
OLLAMA_BASE = "http://localhost:11434/api"
DEFAULT_CHAT_MODEL = "gemma3"
DEFAULT_EMBED_MODEL = "all-minilm"
MAX_RECALL = 8
MAX_GRAPH_NODES = 28

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "so", "to", "of", "for", "on", "in", "at", "by", "with",
    "is", "it", "this", "that", "these", "those", "be", "been", "was", "were", "am", "are", "as", "from", "into",
    "about", "my", "your", "our", "their", "his", "her", "its", "me", "you", "we", "they", "he", "she", "them",
    "i", "im", "i'm", "dont", "don't", "do", "did", "does", "have", "has", "had", "can", "could", "would", "should",
    "what", "why", "how", "when", "where", "who", "which", "will", "just", "like", "really", "very", "more", "less",
}


def now_iso() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def ensure_dirs() -> None:
    os.makedirs(APP_DIR, exist_ok=True)
    os.makedirs(EXPORT_DIR, exist_ok=True)


def json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


def json_loads(s: Optional[str], fallback: Any) -> Any:
    if not s:
        return fallback
    try:
        return json.loads(s)
    except Exception:
        return fallback


def cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:80] or "memory"


def summarize_text(text: str, max_len: int = 240) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= max_len else text[: max_len - 1].rstrip() + "…"


class Database:
    def __init__(self, path: str):
        self.path = path
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self._migrate()

    def _migrate(self) -> None:
        cur = self.conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                actor TEXT NOT NULL,
                kind TEXT NOT NULL,
                content TEXT NOT NULL,
                context_json TEXT,
                salience REAL DEFAULT 0,
                embedding_json TEXT
            );

            CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                kind TEXT NOT NULL DEFAULT 'concept',
                first_seen_ts TEXT NOT NULL,
                last_seen_ts TEXT NOT NULL,
                salience REAL DEFAULT 0,
                metadata_json TEXT DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS memory_crystals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                title TEXT NOT NULL,
                theme TEXT NOT NULL,
                summary TEXT NOT NULL,
                semantic_signature_json TEXT,
                source_event_ids_json TEXT,
                entity_ids_json TEXT,
                emotional_salience REAL DEFAULT 0,
                practical_salience REAL DEFAULT 0,
                identity_salience REAL DEFAULT 0,
                temporal_salience REAL DEFAULT 0,
                uncertainty_salience REAL DEFAULT 0,
                confidence REAL DEFAULT 0.5,
                self_state TEXT DEFAULT 'general',
                future_implications TEXT DEFAULT '',
                unresolved TEXT DEFAULT '',
                contradiction_state TEXT DEFAULT 'clean',
                valid_from_ts TEXT,
                valid_to_ts TEXT,
                embedding_json TEXT,
                compressed_narrative TEXT DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS relations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_type TEXT NOT NULL,
                source_id INTEGER NOT NULL,
                target_type TEXT NOT NULL,
                target_id INTEGER NOT NULL,
                relation TEXT NOT NULL,
                weight REAL DEFAULT 0.5,
                context_json TEXT DEFAULT '{}',
                created_ts TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS contradictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                topic TEXT NOT NULL,
                claim_a TEXT NOT NULL,
                claim_b TEXT NOT NULL,
                evidence_event_a INTEGER,
                evidence_event_b INTEGER,
                resolution_state TEXT DEFAULT 'open',
                notes TEXT DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS activations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                query TEXT NOT NULL,
                active_self_state TEXT,
                retrieved_crystal_ids_json TEXT,
                retrieved_entity_ids_json TEXT,
                scene_json TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts DESC);
            CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
            CREATE INDEX IF NOT EXISTS idx_crystals_ts ON memory_crystals(ts DESC);
            CREATE INDEX IF NOT EXISTS idx_relations_src ON relations(source_type, source_id);
            CREATE INDEX IF NOT EXISTS idx_relations_tgt ON relations(target_type, target_id);
            """
        )
        self.conn.commit()
        self.set_meta("db_initialized_at", now_iso())

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute("INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))
        self.conn.commit()

    def get_meta(self, key: str, default: str = "") -> str:
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else default

    def insert_event(self, actor: str, kind: str, content: str, context: Optional[dict] = None, salience: float = 0.0,
                     embedding: Optional[List[float]] = None) -> int:
        cur = self.conn.execute(
            "INSERT INTO events(ts, actor, kind, content, context_json, salience, embedding_json) VALUES(?, ?, ?, ?, ?, ?, ?)",
            (now_iso(), actor, kind, content, json_dumps(context or {}), salience, json_dumps(embedding) if embedding else None),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def upsert_entity(self, name: str, kind: str = "concept", salience_boost: float = 0.15, metadata: Optional[dict] = None) -> int:
        ts = now_iso()
        row = self.conn.execute("SELECT id, salience, metadata_json FROM entities WHERE name=?", (name,)).fetchone()
        if row:
            merged_meta = json_loads(row["metadata_json"], {})
            merged_meta.update(metadata or {})
            self.conn.execute(
                "UPDATE entities SET last_seen_ts=?, salience=?, metadata_json=? WHERE id=?",
                (ts, float(row["salience"] or 0) + salience_boost, json_dumps(merged_meta), int(row["id"])),
            )
            self.conn.commit()
            return int(row["id"])
        cur = self.conn.execute(
            "INSERT INTO entities(name, kind, first_seen_ts, last_seen_ts, salience, metadata_json) VALUES(?, ?, ?, ?, ?, ?)",
            (name, kind, ts, ts, salience_boost, json_dumps(metadata or {})),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def insert_crystal(self, crystal: dict) -> int:
        cur = self.conn.execute(
            """
            INSERT INTO memory_crystals(
                ts, title, theme, summary, semantic_signature_json, source_event_ids_json, entity_ids_json,
                emotional_salience, practical_salience, identity_salience, temporal_salience, uncertainty_salience,
                confidence, self_state, future_implications, unresolved, contradiction_state,
                valid_from_ts, valid_to_ts, embedding_json, compressed_narrative
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                now_iso(), crystal["title"], crystal["theme"], crystal["summary"],
                json_dumps(crystal.get("semantic_signature", {})),
                json_dumps(crystal.get("source_event_ids", [])),
                json_dumps(crystal.get("entity_ids", [])),
                crystal.get("emotional_salience", 0.0),
                crystal.get("practical_salience", 0.0),
                crystal.get("identity_salience", 0.0),
                crystal.get("temporal_salience", 0.0),
                crystal.get("uncertainty_salience", 0.0),
                crystal.get("confidence", 0.5),
                crystal.get("self_state", "general"),
                crystal.get("future_implications", ""),
                crystal.get("unresolved", ""),
                crystal.get("contradiction_state", "clean"),
                crystal.get("valid_from_ts", now_iso()),
                crystal.get("valid_to_ts"),
                json_dumps(crystal.get("embedding", [])),
                crystal.get("compressed_narrative", ""),
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def insert_relation(self, source_type: str, source_id: int, target_type: str, target_id: int, relation: str,
                        weight: float = 0.5, context: Optional[dict] = None) -> int:
        cur = self.conn.execute(
            "INSERT INTO relations(source_type, source_id, target_type, target_id, relation, weight, context_json, created_ts) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (source_type, source_id, target_type, target_id, relation, weight, json_dumps(context or {}), now_iso()),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def insert_contradiction(self, topic: str, claim_a: str, claim_b: str,
                             evidence_event_a: Optional[int], evidence_event_b: Optional[int], notes: str = "") -> int:
        cur = self.conn.execute(
            "INSERT INTO contradictions(ts, topic, claim_a, claim_b, evidence_event_a, evidence_event_b, notes) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (now_iso(), topic, claim_a, claim_b, evidence_event_a, evidence_event_b, notes),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def insert_activation(self, query: str, active_self_state: str, crystal_ids: List[int], entity_ids: List[int], scene: dict) -> int:
        cur = self.conn.execute(
            "INSERT INTO activations(ts, query, active_self_state, retrieved_crystal_ids_json, retrieved_entity_ids_json, scene_json) VALUES (?, ?, ?, ?, ?, ?)",
            (now_iso(), query, active_self_state, json_dumps(crystal_ids), json_dumps(entity_ids), json_dumps(scene)),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def recent_events(self, limit: int = 40) -> List[sqlite3.Row]:
        return list(self.conn.execute("SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,)).fetchall())

    def recent_crystals(self, limit: int = 50) -> List[sqlite3.Row]:
        return list(self.conn.execute("SELECT * FROM memory_crystals ORDER BY id DESC LIMIT ?", (limit,)).fetchall())

    def recent_contradictions(self, limit: int = 50) -> List[sqlite3.Row]:
        return list(self.conn.execute("SELECT * FROM contradictions ORDER BY id DESC LIMIT ?", (limit,)).fetchall())

    def top_entities(self, limit: int = 50) -> List[sqlite3.Row]:
        return list(self.conn.execute("SELECT * FROM entities ORDER BY salience DESC, last_seen_ts DESC LIMIT ?", (limit,)).fetchall())

    def crystal_by_id(self, crystal_id: int) -> Optional[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM memory_crystals WHERE id=?", (crystal_id,)).fetchone()

    def entity_by_id(self, entity_id: int) -> Optional[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM entities WHERE id=?", (entity_id,)).fetchone()

    def crystal_embeddings(self) -> List[sqlite3.Row]:
        return list(
            self.conn.execute(
                """
                SELECT id, title, theme, summary, entity_ids_json, future_implications, unresolved,
                       embedding_json, emotional_salience, practical_salience, identity_salience,
                       temporal_salience, uncertainty_salience, self_state, contradiction_state
                FROM memory_crystals
                """
            ).fetchall()
        )

    def relation_rows(self) -> List[sqlite3.Row]:
        return list(self.conn.execute("SELECT * FROM relations ORDER BY id DESC LIMIT 500").fetchall())

    def export_json(self, filepath: str) -> None:
        payload = {
            "meta": {r["key"]: r["value"] for r in self.conn.execute("SELECT * FROM meta")},
            "events": [dict(r) for r in self.conn.execute("SELECT * FROM events ORDER BY id")],
            "entities": [dict(r) for r in self.conn.execute("SELECT * FROM entities ORDER BY id")],
            "memory_crystals": [dict(r) for r in self.conn.execute("SELECT * FROM memory_crystals ORDER BY id")],
            "relations": [dict(r) for r in self.conn.execute("SELECT * FROM relations ORDER BY id")],
            "contradictions": [dict(r) for r in self.conn.execute("SELECT * FROM contradictions ORDER BY id")],
            "activations": [dict(r) for r in self.conn.execute("SELECT * FROM activations ORDER BY id")],
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)


class OllamaClient:
    def __init__(self, base_url: str = OLLAMA_BASE):
        self.base_url = base_url.rstrip("/")

    def _post(self, endpoint: str, payload: dict, timeout: int = 180) -> dict:
        req = urllib.request.Request(
            self.base_url + endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8")
                return json.loads(body)
        except urllib.error.HTTPError as e:
            text = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"ollama http {e.code}: {text}")
        except urllib.error.URLError as e:
            raise RuntimeError(f"unable to reach ollama at {self.base_url}: {e}")

    def tags(self) -> dict:
        req = urllib.request.Request(self.base_url + "/tags", method="GET")
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            raise RuntimeError(str(e))

    def chat(self, model: str, messages: List[dict], system: Optional[str] = None, keep_alive: str = "20m") -> str:
        payload = {
            "model": model,
            "messages": ([{"role": "system", "content": system}] if system else []) + messages,
            "stream": False,
            "keep_alive": keep_alive,
        }
        data = self._post("/chat", payload, timeout=300)
        msg = data.get("message", {})
        return msg.get("content", "")

    def embed(self, model: str, text: str) -> List[float]:
        data = self._post("/embed", {"model": model, "input": text}, timeout=180)
        arr = data.get("embeddings", [])
        if not arr:
            return []
        return arr[0]


class MemoryEngine:
    def __init__(self, db: Database, ollama: OllamaClient, logger):
        self.db = db
        self.ollama = ollama
        self.log = logger

    def detect_self_state(self, text: str) -> str:
        t = text.lower()
        if any(x in t for x in ["school", "class", "exam", "assignment", "homework", "project"]):
            return "student"
        if any(x in t for x in ["client", "business", "contract", "invoice", "tax", "quote", "bid", "agency"]):
            return "founder"
        if any(x in t for x in ["joke", "comedy", "comedian", "standup", "bit"]):
            return "comic"
        if any(x in t for x in ["dad", "son", "wife", "family", "maximus", "jessica"]):
            return "family"
        if any(x in t for x in ["deploy", "code", "python", "php", "sqlite", "postgres", "rust", "go", "api"]):
            return "builder"
        return "general"

    def extract_entities(self, text: str) -> List[Tuple[str, str]]:
        found: List[Tuple[str, str]] = []
        tokens = re.findall(r"[A-Za-z0-9_\-\.]{3,}", text)
        counts: Dict[str, int] = {}
        for tok in tokens:
            low = tok.lower()
            if low in STOPWORDS or low.isdigit():
                continue
            counts[low] = counts.get(low, 0) + 1
        for k, c in sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:10]:
            kind = "tool" if any(x in k for x in ["ollama", "sqlite", "postgres", "python", "flask", "tkinter", "api"]) else "concept"
            found.append((k, kind))
        # camel caps / titles / product-ish names
        for m in re.findall(r"\b(?:[A-Z][a-z]+(?:[A-Z][a-z]+)+|[A-Z]{2,}[A-Za-z0-9\-]*)\b", text):
            if m.lower() not in {x[0] for x in found}:
                found.append((m, "entity"))
        return found[:14]

    def score_salience(self, text: str, self_state: str) -> Dict[str, float]:
        t = text.lower()
        emotional = 0.05 + (0.25 if any(x in t for x in ["love", "hate", "afraid", "stressed", "worried", "excited", "important"]) else 0.0)
        practical = 0.10 + (0.30 if any(x in t for x in ["need", "must", "todo", "deadline", "ship", "build", "implement", "fix"]) else 0.0)
        identity = 0.15 if self_state in {"student", "founder", "comic", "family", "builder"} else 0.05
        temporal = 0.25 if any(x in t for x in ["today", "tomorrow", "now", "urgent", "this week", "soon"]) else 0.08
        uncertainty = 0.30 if "?" in text or any(x in t for x in ["not sure", "unclear", "maybe", "confused", "contradiction"]) else 0.05
        return {
            "emotional_salience": clamp(emotional, 0, 1),
            "practical_salience": clamp(practical, 0, 1),
            "identity_salience": clamp(identity, 0, 1),
            "temporal_salience": clamp(temporal, 0, 1),
            "uncertainty_salience": clamp(uncertainty, 0, 1),
        }

    def detect_contradictions(self, text: str, event_id: int) -> None:
        t = text.lower()
        prefs = []
        for pat in [r"i prefer ([^.\n]+)", r"i like ([^.\n]+)", r"i hate ([^.\n]+)", r"i don't like ([^.\n]+)", r"i do not like ([^.\n]+)"]:
            for m in re.finditer(pat, t):
                prefs.append(m.group(0).strip())
        if not prefs:
            return
        prev = self.db.recent_events(limit=120)
        for old in prev:
            if int(old["id"]) == event_id:
                continue
            old_text = old["content"].lower()
            if ("i prefer" in old_text or "i like" in old_text or "i hate" in old_text or "i don't like" in old_text or "i do not like" in old_text):
                for p in prefs:
                    core = p.replace("i prefer ", "").replace("i like ", "").replace("i hate ", "").replace("i don't like ", "").replace("i do not like ", "")
                    if core and core in old_text and old_text != t and (("hate" in p or "don't like" in p or "do not like" in p) != ("hate" in old_text or "don't like" in old_text or "do not like" in old_text)):
                        self.db.insert_contradiction(core, old["content"], text, int(old["id"]), event_id, "auto-detected preference tension")
                        return

    def make_crystal(self, event_id: int, actor: str, text: str, embed_model: str) -> int:
        self_state = self.detect_self_state(text)
        ents = self.extract_entities(text)
        entity_ids = [self.db.upsert_entity(name, kind) for name, kind in ents]
        sal = self.score_salience(text, self_state)
        signature = {
            "self_state": self_state,
            "keywords": [name for name, _ in ents],
            "length": len(text),
            "contains_question": "?" in text,
        }
        summary = summarize_text(text, 220)
        future_implications = self._infer_future_implications(text)
        unresolved = self._infer_unresolved(text)
        title = self._make_title(text, ents)
        embedding = []
        try:
            embedding = self.ollama.embed(embed_model, f"theme: {title}\nsummary: {summary}\nfuture: {future_implications}\n")
        except Exception as e:
            self.log(f"embed warning while creating crystal: {e}")
        crystal = {
            "title": title,
            "theme": title,
            "summary": summary,
            "semantic_signature": signature,
            "source_event_ids": [event_id],
            "entity_ids": entity_ids,
            "confidence": 0.72 if actor == "user" else 0.58,
            "self_state": self_state,
            "future_implications": future_implications,
            "unresolved": unresolved,
            "contradiction_state": "tension" if unresolved else "clean",
            "compressed_narrative": f"{title}: {summary}",
            "embedding": embedding,
            **sal,
        }
        crystal_id = self.db.insert_crystal(crystal)
        for ent_id in entity_ids:
            self.db.insert_relation("crystal", crystal_id, "entity", ent_id, "mentions", weight=0.6)
        self.db.insert_relation("event", event_id, "crystal", crystal_id, "crystallized_into", weight=0.9)
        return crystal_id

    def _make_title(self, text: str, ents: List[Tuple[str, str]]) -> str:
        if ents:
            head = ", ".join([e[0] for e in ents[:3]])
            return summarize_text(head, 64)
        words = [w for w in re.findall(r"[A-Za-z0-9_\-]+", text) if w.lower() not in STOPWORDS]
        return summarize_text(" ".join(words[:6]) or "memory crystal", 64)

    def _infer_future_implications(self, text: str) -> str:
        t = text.lower()
        bits = []
        if any(x in t for x in ["build", "implement", "prototype", "app"]):
            bits.append("likely leads to implementation work")
        if any(x in t for x in ["tax", "invoice", "contract", "bid"]):
            bits.append("may affect business/compliance decisions")
        if any(x in t for x in ["exam", "assignment", "class"]):
            bits.append("may affect academic workload")
        if "?" in text:
            bits.append("open question likely to resurface")
        return "; ".join(bits) or "latent relevance may increase when adjacent topics reactivate"

    def _infer_unresolved(self, text: str) -> str:
        t = text.lower()
        if "?" in text:
            return summarize_text(text, 160)
        for key in ["todo", "need to", "must", "should", "unclear", "not sure", "figure out"]:
            if key in t:
                return summarize_text(text, 160)
        return ""

    def ingest_event(self, actor: str, kind: str, text: str, context: Optional[dict], embed_model: str) -> Tuple[int, int]:
        self_state = self.detect_self_state(text)
        sal = self.score_salience(text, self_state)
        combined_salience = sum(sal.values()) / 5.0
        embedding = []
        try:
            embedding = self.ollama.embed(embed_model, text)
        except Exception as e:
            self.log(f"embed warning: {e}")
        event_id = self.db.insert_event(actor, kind, text, context=context, salience=combined_salience, embedding=embedding)
        crystal_id = self.make_crystal(event_id, actor, text, embed_model)
        self.detect_contradictions(text, event_id)
        return event_id, crystal_id

    def retrieve_scene(self, query: str, embed_model: str, self_state: Optional[str] = None) -> dict:
        active_self = self_state or self.detect_self_state(query)
        q_embed = []
        try:
            q_embed = self.ollama.embed(embed_model, query)
        except Exception as e:
            self.log(f"embed warning during retrieve: {e}")
        scored: List[Tuple[float, sqlite3.Row]] = []
        for row in self.db.crystal_embeddings():
            emb = json_loads(row["embedding_json"], [])
            score = cosine_similarity(q_embed, emb) if q_embed and emb else 0.0
            if row["self_state"] == active_self:
                score += 0.12
            score += 0.25 * float(row["practical_salience"] or 0)
            score += 0.12 * float(row["identity_salience"] or 0)
            if row["contradiction_state"] == "tension":
                score += 0.04
            scored.append((score, row))
        scored.sort(key=lambda x: x[0], reverse=True)
        picked = [r for s, r in scored[:MAX_RECALL] if s > 0.05 or len(scored) <= MAX_RECALL]

        entity_ids = []
        scene_crystals = []
        for row in picked:
            eids = json_loads(row["entity_ids_json"], [])
            entity_ids.extend(eids)
            scene_crystals.append({
                "id": row["id"],
                "title": row["title"],
                "summary": row["summary"],
                "future_implications": row["future_implications"],
                "unresolved": row["unresolved"],
                "self_state": row["self_state"],
                "score": next((s for s, rr in scored if rr["id"] == row["id"]), 0.0),
            })
        uniq_eids = []
        seen = set()
        for eid in entity_ids:
            if eid not in seen:
                uniq_eids.append(eid)
                seen.add(eid)
        entity_rows = [self.db.entity_by_id(eid) for eid in uniq_eids[:16]]
        entity_rows = [r for r in entity_rows if r]

        contradictions = [dict(r) for r in self.db.recent_contradictions(limit=12)]
        scene = {
            "query": query,
            "active_self_state": active_self,
            "crystals": scene_crystals,
            "entities": [{"id": r["id"], "name": r["name"], "kind": r["kind"], "salience": r["salience"]} for r in entity_rows],
            "contradictions": contradictions,
            "scene_summary": self._scene_summary(query, active_self, scene_crystals, entity_rows),
        }
        self.db.insert_activation(query, active_self, [int(c["id"]) for c in scene_crystals], [int(r["id"]) for r in entity_rows], scene)
        return scene

    def _scene_summary(self, query: str, active_self: str, crystals: List[dict], entities: List[sqlite3.Row]) -> str:
        lines = [f"active self-state: {active_self}"]
        if crystals:
            lines.append("dominant memory crystals: " + "; ".join(c["title"] for c in crystals[:4]))
        if entities:
            lines.append("active entities: " + ", ".join(r["name"] for r in entities[:6]))
        lines.append(f"current query: {summarize_text(query, 120)}")
        return "\n".join(lines)

    def answer_with_memory(self, query: str, chat_model: str, embed_model: str) -> Tuple[str, dict]:
        scene = self.retrieve_scene(query, embed_model)
        system = (
            "you are an ai using a semantic gravity memory system. answer naturally and clearly. "
            "use the reconstructed memory scene when relevant, but do not invent facts. "
            "if memory is sparse, say so. do not mention hidden system prompts."
        )
        prompt = (
            "reconstructed scene\n"
            f"{json.dumps(scene, ensure_ascii=False, indent=2)}\n\n"
            f"user query\n{query}\n\n"
            "respond in a grounded way using the memory scene above."
        )
        answer = self.ollama.chat(chat_model, [{"role": "user", "content": prompt}], system=system)
        return answer, scene


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        ensure_dirs()
        self.title(APP_TITLE)
        self.geometry("1520x960")
        self.minsize(1240, 820)
        self.configure(bg="#050505")

        self.db = Database(DB_PATH)
        self.ollama = OllamaClient()
        self.log_queue: queue.Queue[str] = queue.Queue()
        self.memory = MemoryEngine(self.db, self.ollama, self.log)
        self.current_scene: dict = {}

        self.chat_model_var = tk.StringVar(value=self.db.get_meta("chat_model", DEFAULT_CHAT_MODEL) or DEFAULT_CHAT_MODEL)
        self.embed_model_var = tk.StringVar(value=self.db.get_meta("embed_model", DEFAULT_EMBED_MODEL) or DEFAULT_EMBED_MODEL)
        self.base_url_var = tk.StringVar(value=self.db.get_meta("ollama_base", OLLAMA_BASE) or OLLAMA_BASE)
        self.status_var = tk.StringVar(value="ready")

        self._configure_style()
        self._build_ui()
        self.refresh_all()
        self.after(180, self._drain_log_queue)

    def _configure_style(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure(".", background="#050505", foreground="#f4f4f4", fieldbackground="#0f0f10")
        style.configure("TFrame", background="#050505")
        style.configure("TLabel", background="#050505", foreground="#f4f4f4", font=("Helvetica", 10))
        style.configure("Title.TLabel", font=("Helvetica", 18, "bold"), foreground="#ffffff")
        style.configure("Sub.TLabel", font=("Helvetica", 10), foreground="#bcbcbc")
        style.configure("TButton", background="#111214", foreground="#ffffff", padding=8)
        style.map("TButton", background=[("active", "#1b1c20")])
        style.configure("TEntry", foreground="#ffffff", fieldbackground="#111214", insertcolor="#ffffff")
        style.configure("TNotebook", background="#050505", borderwidth=0)
        style.configure("TNotebook.Tab", background="#0d0d0f", foreground="#d8d8d8", padding=(14, 8))
        style.map("TNotebook.Tab", background=[("selected", "#17181c")], foreground=[("selected", "#ffffff")])
        style.configure("Treeview", background="#0f1012", foreground="#ececec", fieldbackground="#0f1012", rowheight=26, borderwidth=0)
        style.map("Treeview", background=[("selected", "#262a31")])
        style.configure("Treeview.Heading", background="#121318", foreground="#ffffff", relief="flat")

    def _build_ui(self) -> None:
        root = ttk.Frame(self)
        root.pack(fill="both", expand=True)

        top = ttk.Frame(root)
        top.pack(fill="x", padx=18, pady=(16, 10))

        left_top = ttk.Frame(top)
        left_top.pack(side="left", fill="x", expand=True)
        ttk.Label(left_top, text="semantic gravity memory lab", style="Title.TLabel").pack(anchor="w")
        ttk.Label(left_top, text="field-activated memory • memory crystals • contradictions • reconstructive recall • local ollama", style="Sub.TLabel").pack(anchor="w", pady=(4, 0))

        right_top = ttk.Frame(top)
        right_top.pack(side="right")
        ttk.Label(right_top, text="chat model").grid(row=0, column=0, sticky="e", padx=4)
        ttk.Entry(right_top, textvariable=self.chat_model_var, width=18).grid(row=0, column=1, padx=4)
        ttk.Label(right_top, text="embed model").grid(row=0, column=2, sticky="e", padx=4)
        ttk.Entry(right_top, textvariable=self.embed_model_var, width=18).grid(row=0, column=3, padx=4)
        ttk.Button(right_top, text="save", command=self.save_settings).grid(row=0, column=4, padx=4)

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True, padx=18, pady=(0, 10))

        self.chat_tab = ttk.Frame(self.notebook)
        self.graph_tab = ttk.Frame(self.notebook)
        self.crystals_tab = ttk.Frame(self.notebook)
        self.events_tab = ttk.Frame(self.notebook)
        self.system_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.chat_tab, text="chat + memory")
        self.notebook.add(self.graph_tab, text="graph")
        self.notebook.add(self.crystals_tab, text="crystals")
        self.notebook.add(self.events_tab, text="events")
        self.notebook.add(self.system_tab, text="system")

        self._build_chat_tab()
        self._build_graph_tab()
        self._build_crystals_tab()
        self._build_events_tab()
        self._build_system_tab()

        bottom = ttk.Frame(root)
        bottom.pack(fill="x", padx=18, pady=(0, 14))
        ttk.Label(bottom, textvariable=self.status_var, style="Sub.TLabel").pack(side="left")
        ttk.Button(bottom, text="refresh", command=self.refresh_all).pack(side="right")

    def _build_chat_tab(self) -> None:
        wrap = ttk.Frame(self.chat_tab)
        wrap.pack(fill="both", expand=True, padx=8, pady=8)
        wrap.columnconfigure(0, weight=3)
        wrap.columnconfigure(1, weight=2)
        wrap.rowconfigure(0, weight=1)
        wrap.rowconfigure(1, weight=0)

        left = ttk.Frame(wrap)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        left.rowconfigure(0, weight=1)
        left.rowconfigure(1, weight=0)
        left.columnconfigure(0, weight=1)

        self.chat_output = tk.Text(left, bg="#08090b", fg="#f6f6f6", insertbackground="#ffffff", wrap="word", relief="flat", padx=16, pady=16, font=("Menlo", 11))
        self.chat_output.grid(row=0, column=0, sticky="nsew")
        self.chat_output.tag_configure("user", foreground="#ffffff")
        self.chat_output.tag_configure("assistant", foreground="#c8d7ff")
        self.chat_output.tag_configure("scene", foreground="#8ea0b8")
        self.chat_output.tag_configure("header", foreground="#8a8a8a")

        entry_bar = ttk.Frame(left)
        entry_bar.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        entry_bar.columnconfigure(0, weight=1)
        self.chat_input = tk.Text(entry_bar, height=5, bg="#101114", fg="#ffffff", insertbackground="#ffffff", wrap="word", relief="flat", padx=12, pady=10, font=("Menlo", 11))
        self.chat_input.grid(row=0, column=0, sticky="ew")
        btns = ttk.Frame(entry_bar)
        btns.grid(row=0, column=1, sticky="ns", padx=(10, 0))
        ttk.Button(btns, text="send", command=self.send_chat).pack(fill="x")
        ttk.Button(btns, text="ingest only", command=self.ingest_only).pack(fill="x", pady=(8, 0))
        ttk.Button(btns, text="clear view", command=lambda: self.chat_output.delete("1.0", "end")).pack(fill="x", pady=(8, 0))

        right = ttk.Frame(wrap)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        scene_frame = ttk.Frame(right)
        scene_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 10))
        ttk.Label(scene_frame, text="active reconstructed scene", style="Sub.TLabel").pack(anchor="w")
        self.scene_text = tk.Text(scene_frame, bg="#08090b", fg="#d7d7d7", insertbackground="#ffffff", wrap="word", relief="flat", padx=14, pady=14, font=("Menlo", 10))
        self.scene_text.pack(fill="both", expand=True, pady=(6, 0))

        expl_frame = ttk.Frame(right)
        expl_frame.grid(row=1, column=0, sticky="nsew")
        ttk.Label(expl_frame, text="how this memory works", style="Sub.TLabel").pack(anchor="w")
        self.explain_text = tk.Text(expl_frame, bg="#08090b", fg="#d7d7d7", wrap="word", relief="flat", padx=14, pady=14, font=("Menlo", 10))
        self.explain_text.pack(fill="both", expand=True, pady=(6, 0))
        expl = (
            "1. every message becomes an event atom\n"
            "2. entities are extracted and reinforced\n"
            "3. the event is compressed into a memory crystal\n"
            "4. each crystal gets salience dimensions and an embedding\n"
            "5. recall is field activation, not just exact keyword search\n"
            "6. the app reconstructs a scene from active crystals + entities + contradictions\n"
            "7. ollama answers from that reconstructed scene\n"
            "8. the graph tab shows what became active in the current cognitive gravity well"
        )
        self.explain_text.insert("1.0", expl)
        self.explain_text.configure(state="disabled")

    def _build_graph_tab(self) -> None:
        wrap = ttk.Frame(self.graph_tab)
        wrap.pack(fill="both", expand=True, padx=8, pady=8)
        wrap.columnconfigure(0, weight=4)
        wrap.columnconfigure(1, weight=2)
        wrap.rowconfigure(0, weight=1)

        self.graph_canvas = tk.Canvas(wrap, bg="#060708", highlightthickness=0)
        self.graph_canvas.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        side = ttk.Frame(wrap)
        side.grid(row=0, column=1, sticky="nsew")
        side.rowconfigure(1, weight=1)
        ttk.Label(side, text="graph legend", style="Sub.TLabel").pack(anchor="w")
        legend = tk.Text(side, bg="#08090b", fg="#d7d7d7", wrap="word", relief="flat", padx=12, pady=12, height=8, font=("Menlo", 10))
        legend.pack(fill="x", pady=(6, 10))
        legend.insert("1.0", "crystal nodes = memory crystals\nentity nodes = extracted concepts/tools/people\nedge thickness = relationship weight\nnode size = salience\ncurrent scene gets a brighter halo\n")
        legend.configure(state="disabled")

        ttk.Label(side, text="scene payload", style="Sub.TLabel").pack(anchor="w")
        self.graph_detail = tk.Text(side, bg="#08090b", fg="#d7d7d7", wrap="word", relief="flat", padx=12, pady=12, font=("Menlo", 10))
        self.graph_detail.pack(fill="both", expand=True, pady=(6, 0))

    def _build_crystals_tab(self) -> None:
        wrap = ttk.Frame(self.crystals_tab)
        wrap.pack(fill="both", expand=True, padx=8, pady=8)
        wrap.columnconfigure(0, weight=3)
        wrap.columnconfigure(1, weight=2)
        wrap.rowconfigure(0, weight=1)

        left = ttk.Frame(wrap)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)
        cols = ("id", "title", "self_state", "practical", "identity", "uncertainty")
        self.crystal_tree = ttk.Treeview(left, columns=cols, show="headings")
        for c, w in [("id", 60), ("title", 320), ("self_state", 110), ("practical", 85), ("identity", 85), ("uncertainty", 95)]:
            self.crystal_tree.heading(c, text=c)
            self.crystal_tree.column(c, width=w, anchor="w")
        self.crystal_tree.grid(row=0, column=0, sticky="nsew")
        self.crystal_tree.bind("<<TreeviewSelect>>", self.show_selected_crystal)

        right = ttk.Frame(wrap)
        right.grid(row=0, column=1, sticky="nsew")
        ttk.Label(right, text="crystal detail", style="Sub.TLabel").pack(anchor="w")
        self.crystal_detail = tk.Text(right, bg="#08090b", fg="#d7d7d7", wrap="word", relief="flat", padx=12, pady=12, font=("Menlo", 10))
        self.crystal_detail.pack(fill="both", expand=True, pady=(6, 0))

    def _build_events_tab(self) -> None:
        wrap = ttk.Frame(self.events_tab)
        wrap.pack(fill="both", expand=True, padx=8, pady=8)
        wrap.columnconfigure(0, weight=1)
        wrap.rowconfigure(0, weight=1)
        cols = ("id", "ts", "actor", "kind", "salience", "content")
        self.events_tree = ttk.Treeview(wrap, columns=cols, show="headings")
        widths = {"id": 60, "ts": 160, "actor": 80, "kind": 120, "salience": 80, "content": 780}
        for c in cols:
            self.events_tree.heading(c, text=c)
            self.events_tree.column(c, width=widths[c], anchor="w")
        self.events_tree.grid(row=0, column=0, sticky="nsew")

    def _build_system_tab(self) -> None:
        wrap = ttk.Frame(self.system_tab)
        wrap.pack(fill="both", expand=True, padx=8, pady=8)
        wrap.columnconfigure(0, weight=1)
        wrap.rowconfigure(1, weight=1)

        ctl = ttk.Frame(wrap)
        ctl.grid(row=0, column=0, sticky="ew")
        ttk.Label(ctl, text="ollama base").grid(row=0, column=0, sticky="e", padx=4)
        ttk.Entry(ctl, textvariable=self.base_url_var, width=40).grid(row=0, column=1, padx=4)
        ttk.Button(ctl, text="test ollama", command=self.test_ollama).grid(row=0, column=2, padx=4)
        ttk.Button(ctl, text="pull chat model", command=self.pull_chat_model).grid(row=0, column=3, padx=4)
        ttk.Button(ctl, text="pull embed model", command=self.pull_embed_model).grid(row=0, column=4, padx=4)
        ttk.Button(ctl, text="export json", command=self.export_json).grid(row=0, column=5, padx=4)
        ttk.Button(ctl, text="seed demo memory", command=self.seed_demo).grid(row=0, column=6, padx=4)

        self.system_log = tk.Text(wrap, bg="#08090b", fg="#d7d7d7", wrap="word", relief="flat", padx=12, pady=12, font=("Menlo", 10))
        self.system_log.grid(row=1, column=0, sticky="nsew", pady=(12, 0))
        self.system_log.insert("1.0", f"app dir: {APP_DIR}\ndb path: {DB_PATH}\n")

    def save_settings(self) -> None:
        self.db.set_meta("chat_model", self.chat_model_var.get().strip())
        self.db.set_meta("embed_model", self.embed_model_var.get().strip())
        self.db.set_meta("ollama_base", self.base_url_var.get().strip())
        self.ollama.base_url = self.base_url_var.get().strip().rstrip("/")
        self.status_var.set("settings saved")
        self.log("saved settings")

    def log(self, message: str) -> None:
        self.log_queue.put(f"[{time.strftime('%H:%M:%S')}] {message}")

    def _drain_log_queue(self) -> None:
        try:
            while True:
                msg = self.log_queue.get_nowait()
                self.system_log.insert("end", msg + "\n")
                self.system_log.see("end")
        except queue.Empty:
            pass
        self.after(180, self._drain_log_queue)

    def append_chat(self, who: str, text: str, tag: str) -> None:
        self.chat_output.insert("end", f"\n{who}\n", "header")
        self.chat_output.insert("end", text.strip() + "\n", tag)
        self.chat_output.see("end")

    def send_chat(self) -> None:
        text = self.chat_input.get("1.0", "end").strip()
        if not text:
            return
        self.chat_input.delete("1.0", "end")
        self.append_chat("you", text, "user")
        self.status_var.set("thinking with memory...")
        threading.Thread(target=self._send_chat_worker, args=(text,), daemon=True).start()

    def _send_chat_worker(self, text: str) -> None:
        try:
            self.save_settings()
            embed_model = self.embed_model_var.get().strip()
            chat_model = self.chat_model_var.get().strip()
            self.memory.ingest_event("user", "chat_message", text, {"channel": "chat"}, embed_model)
            answer, scene = self.memory.answer_with_memory(text, chat_model, embed_model)
            self.memory.ingest_event("assistant", "chat_response", answer, {"channel": "chat", "query": text}, embed_model)
            self.current_scene = scene
            self.after(0, lambda: self.append_chat("assistant", answer, "assistant"))
            self.after(0, lambda: self.scene_text.delete("1.0", "end"))
            self.after(0, lambda: self.scene_text.insert("1.0", json.dumps(scene, ensure_ascii=False, indent=2)))
            self.after(0, self.refresh_all)
            self.after(0, lambda: self.status_var.set("ready"))
        except Exception as e:
            err = f"error: {e}\n{traceback.format_exc()}"
            self.log(err)
            self.after(0, lambda: self.append_chat("system", err, "scene"))
            self.after(0, lambda: self.status_var.set("error"))

    def ingest_only(self) -> None:
        text = self.chat_input.get("1.0", "end").strip()
        if not text:
            return
        self.chat_input.delete("1.0", "end")
        self.append_chat("you", text, "user")
        def worker():
            try:
                self.memory.ingest_event("user", "note", text, {"channel": "manual_ingest"}, self.embed_model_var.get().strip())
                self.after(0, self.refresh_all)
                self.after(0, lambda: self.status_var.set("note ingested"))
            except Exception as e:
                self.log(str(e))
        threading.Thread(target=worker, daemon=True).start()

    def refresh_all(self) -> None:
        self.refresh_crystals()
        self.refresh_events()
        self.refresh_graph()

    def refresh_crystals(self) -> None:
        for i in self.crystal_tree.get_children():
            self.crystal_tree.delete(i)
        for row in self.db.recent_crystals(limit=120):
            self.crystal_tree.insert("", "end", iid=str(row["id"]), values=(
                row["id"], row["title"], row["self_state"],
                f"{float(row['practical_salience'] or 0):.2f}",
                f"{float(row['identity_salience'] or 0):.2f}",
                f"{float(row['uncertainty_salience'] or 0):.2f}",
            ))

    def refresh_events(self) -> None:
        for i in self.events_tree.get_children():
            self.events_tree.delete(i)
        for row in self.db.recent_events(limit=200):
            self.events_tree.insert("", "end", values=(
                row["id"], row["ts"], row["actor"], row["kind"], f"{float(row['salience'] or 0):.2f}", summarize_text(row["content"], 160)
            ))

    def show_selected_crystal(self, event=None) -> None:
        sel = self.crystal_tree.selection()
        if not sel:
            return
        row = self.db.crystal_by_id(int(sel[0]))
        if not row:
            return
        payload = dict(row)
        payload["semantic_signature_json"] = json_loads(payload.get("semantic_signature_json"), {})
        payload["source_event_ids_json"] = json_loads(payload.get("source_event_ids_json"), [])
        payload["entity_ids_json"] = json_loads(payload.get("entity_ids_json"), [])
        self.crystal_detail.delete("1.0", "end")
        self.crystal_detail.insert("1.0", json.dumps(payload, ensure_ascii=False, indent=2))

    def refresh_graph(self) -> None:
        c = self.graph_canvas
        c.delete("all")
        w = max(c.winfo_width(), 900)
        h = max(c.winfo_height(), 700)

        scene = self.current_scene or {}
        active_crystals = scene.get("crystals", [])
        active_entities = scene.get("entities", [])

        crystals = self.db.recent_crystals(limit=18)
        entities = self.db.top_entities(limit=12)

        node_specs = []
        for row in crystals[:MAX_GRAPH_NODES]:
            node_specs.append({
                "type": "crystal",
                "id": int(row["id"]),
                "label": row["title"],
                "size": 16 + int(18 * float((row["practical_salience"] or 0) + (row["identity_salience"] or 0))),
                "active": any(int(x["id"]) == int(row["id"]) for x in active_crystals),
            })
        for row in entities[:MAX_GRAPH_NODES]:
            node_specs.append({
                "type": "entity",
                "id": int(row["id"]),
                "label": row["name"],
                "size": 10 + int(8 * float(row["salience"] or 0)),
                "active": any(int(x["id"]) == int(row["id"]) for x in active_entities),
            })

        positions: Dict[Tuple[str, int], Tuple[float, float]] = {}
        cx, cy = w / 2, h / 2
        crystal_nodes = [n for n in node_specs if n["type"] == "crystal"]
        entity_nodes = [n for n in node_specs if n["type"] == "entity"]

        for i, n in enumerate(crystal_nodes):
            ang = (2 * math.pi * i) / max(1, len(crystal_nodes))
            r = min(w, h) * 0.30 + (20 if n["active"] else 0)
            positions[(n["type"], n["id"])] = (cx + math.cos(ang) * r, cy + math.sin(ang) * r)
        for i, n in enumerate(entity_nodes):
            ang = (2 * math.pi * i) / max(1, len(entity_nodes))
            r = min(w, h) * 0.14 + (16 if n["active"] else 0)
            positions[(n["type"], n["id"])] = (cx + math.cos(ang) * r, cy + math.sin(ang) * r)

        for rel in self.db.relation_rows():
            s_key = (rel["source_type"], int(rel["source_id"]))
            t_key = (rel["target_type"], int(rel["target_id"]))
            if s_key in positions and t_key in positions:
                x1, y1 = positions[s_key]
                x2, y2 = positions[t_key]
                width = 1 + 3 * float(rel["weight"] or 0)
                color = "#2c3440" if rel["relation"] != "mentions" else "#23303a"
                c.create_line(x1, y1, x2, y2, fill=color, width=width)

        for n in node_specs:
            x, y = positions[(n["type"], n["id"])]
            r = n["size"]
            fill = "#18202b" if n["type"] == "crystal" else "#11261a"
            outline = "#96b3ff" if n["active"] and n["type"] == "crystal" else ("#7cffb0" if n["active"] else "#40454f")
            if n["active"]:
                c.create_oval(x - r - 8, y - r - 8, x + r + 8, y + r + 8, outline="#2d3d58", width=2)
            c.create_oval(x - r, y - r, x + r, y + r, fill=fill, outline=outline, width=2)
            c.create_text(x, y, text=("C" if n["type"] == "crystal" else "E"), fill="#ffffff", font=("Helvetica", max(9, min(14, r // 2)), "bold"))
            c.create_text(x, y + r + 14, text=summarize_text(n["label"], 20), fill="#cfd4dd", font=("Helvetica", 9))

        detail = {
            "current_scene": scene,
            "graph_nodes": len(node_specs),
            "graph_edges": len(self.db.relation_rows()),
            "note": "active halos show the current memory field that got activated for the latest query",
        }
        self.graph_detail.delete("1.0", "end")
        self.graph_detail.insert("1.0", json.dumps(detail, ensure_ascii=False, indent=2))

    def test_ollama(self) -> None:
        def worker():
            try:
                self.save_settings()
                tags = self.ollama.tags()
                models = [m.get("name") for m in tags.get("models", [])][:20]
                self.log("ollama reachable")
                self.log("available models: " + ", ".join(models) if models else "no models listed")
                self.after(0, lambda: self.status_var.set("ollama reachable"))
            except Exception as e:
                self.log(f"ollama test failed: {e}")
                self.after(0, lambda: self.status_var.set("ollama test failed"))
        threading.Thread(target=worker, daemon=True).start()

    def _pull_model_worker(self, model_name: str) -> None:
        try:
            if not shutil.which("ollama"):
                raise RuntimeError("ollama cli not found in PATH")
            self.log(f"pulling model: {model_name}")
            proc = subprocess.Popen(["ollama", "pull", model_name], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            assert proc.stdout is not None
            for line in proc.stdout:
                self.log(line.rstrip())
            code = proc.wait()
            if code != 0:
                raise RuntimeError(f"ollama pull exited with {code}")
            self.log(f"model ready: {model_name}")
            self.after(0, lambda: self.status_var.set(f"pulled {model_name}"))
        except Exception as e:
            self.log(f"pull failed: {e}")
            self.after(0, lambda: self.status_var.set("model pull failed"))

    def pull_chat_model(self) -> None:
        model = self.chat_model_var.get().strip()
        threading.Thread(target=self._pull_model_worker, args=(model,), daemon=True).start()

    def pull_embed_model(self) -> None:
        model = self.embed_model_var.get().strip()
        threading.Thread(target=self._pull_model_worker, args=(model,), daemon=True).start()

    def export_json(self) -> None:
        filepath = filedialog.asksaveasfilename(
            title="export memory json",
            defaultextension=".json",
            initialdir=EXPORT_DIR,
            initialfile=f"memory_export_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            filetypes=[("json", "*.json")],
        )
        if not filepath:
            return
        try:
            self.db.export_json(filepath)
            self.log(f"exported json: {filepath}")
            self.status_var.set("export complete")
        except Exception as e:
            self.log(f"export failed: {e}")
            self.status_var.set("export failed")

    def seed_demo(self) -> None:
        demo = [
            "i prefer single-file python and php apps with sqlite for prototypes",
            "i'm building a new kind of memory system that stores memory crystals instead of raw chunks",
            "i need a black sleek gui that visualizes how memory gets activated",
            "i might later connect this to postgres, but for now i want local-first and semantic",
            "there is tension between simple storage and true reconstructive memory",
        ]
        def worker():
            for line in demo:
                try:
                    self.memory.ingest_event("user", "seed", line, {"channel": "seed_demo"}, self.embed_model_var.get().strip())
                except Exception as e:
                    self.log(f"seed issue: {e}")
            self.after(0, self.refresh_all)
            self.after(0, lambda: self.status_var.set("seeded demo memory"))
        threading.Thread(target=worker, daemon=True).start()


def main() -> None:
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
