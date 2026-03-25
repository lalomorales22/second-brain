"""
Semantic Gravity Memory

A living memory system for AI — structured crystals, temporal gravity,
spreading activation, contradiction tracking, and consolidation.

Zero external Python dependencies.

Quick start::

    from semantic_gravity_memory import Memory

    memory = Memory()                                  # SQLite, no embeddings
    memory.ingest("I prefer Python for prototypes")
    scene  = memory.recall("what language?")
    memory.consolidate()

With Ollama embeddings::

    memory = Memory(ollama_model="all-minilm")

"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional, Tuple

from semantic_gravity_memory.models import (
    SalienceVector,
    Event,
    Entity,
    Crystal,
    Relation,
    Contradiction,
    Activation,
    ProspectiveMemory,
    Schema,
    AntibodyMemory,
)
from semantic_gravity_memory.storage.sqlite_backend import SQLiteBackend
from semantic_gravity_memory.embeddings.ollama import OllamaEmbedder
from semantic_gravity_memory.core.engine import MemoryEngine

__version__ = "0.1.0"

_DEFAULT_DIR = os.path.join(os.path.expanduser("~"), ".semantic_gravity_memory")
_DEFAULT_DB = os.path.join(_DEFAULT_DIR, "memory.db")


class Memory:
    """One-line entry point to the Semantic Gravity Memory system.

    Args:
        db_path:       Path to SQLite database. Defaults to
                       ``~/.semantic_gravity_memory/memory.db``.
                       Pass ``":memory:"`` for an ephemeral in-memory store.
        ollama_model:  If set, enables Ollama embeddings with this model name
                       (e.g. ``"all-minilm"``).
        ollama_url:    Ollama API base URL (default ``http://localhost:11434/api``).
        max_recall:    Maximum crystals returned per recall (default 8).
        carrying_capacity: Max active crystals before consolidation evicts (default 500).

    Example::

        memory = Memory()
        memory.ingest("I always use Python for prototypes")
        scene = memory.recall("what tools does the user prefer?")
        print(scene["crystals"])
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        ollama_model: Optional[str] = None,
        ollama_url: str = "http://localhost:11434/api",
        max_recall: int = 8,
        carrying_capacity: int = 500,
    ):
        # Storage
        if db_path is None:
            os.makedirs(_DEFAULT_DIR, exist_ok=True)
            db_path = _DEFAULT_DB
        elif db_path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._storage = SQLiteBackend(db_path)

        # Embedder (optional)
        embedder = None
        if ollama_model:
            embedder = OllamaEmbedder(model=ollama_model, base_url=ollama_url)

        # Engine
        self._engine = MemoryEngine(
            self._storage, embedder,
            max_recall=max_recall,
            carrying_capacity=carrying_capacity,
        )

    # ----- Core operations ---------------------------------------------------

    def ingest(
        self,
        text: str,
        actor: str = "user",
        kind: str = "chat_message",
        context: Optional[Dict] = None,
    ) -> Tuple[int, int]:
        """Ingest text → event + crystal. Returns ``(event_id, crystal_id)``."""
        return self._engine.ingest(text, actor, kind, context)

    def recall(
        self,
        query: str,
        self_state: Optional[str] = None,
        now_ts: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Reconstruct a memory scene for *query*. Returns a scene dict."""
        return self._engine.recall(query, self_state, now_ts)

    def answer(
        self,
        query: str,
        chat_fn: Callable[[str], str],
        self_state: Optional[str] = None,
        now_ts: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Recall → call *chat_fn* with grounded prompt → ingest response.

        Returns ``(answer_text, scene)``.
        """
        return self._engine.answer(query, chat_fn, self_state, now_ts)

    # ----- Consolidation -----------------------------------------------------

    def consolidate(self, now_ts: Optional[str] = None) -> Dict[str, Any]:
        """Run a single consolidation pass. Returns a log dict."""
        return self._engine.consolidate(now_ts)

    def start_daemon(self, heartbeat_seconds: float = 300.0) -> None:
        """Start the background consolidation daemon."""
        self._engine.start_daemon(heartbeat_seconds)

    def stop_daemon(self) -> None:
        """Stop the background consolidation daemon."""
        self._engine.stop_daemon()

    # ----- Feedback & control ------------------------------------------------

    def feedback(self, activation_id: int, quality: float, self_state: str = "general") -> None:
        """Record quality feedback for a retrieval."""
        self._engine.feedback(activation_id, quality, self_state)

    def set_prospective(
        self,
        trigger: str,
        crystal_id: int,
        embedding: Optional[List[float]] = None,
        expiry_ts: Optional[str] = None,
    ) -> int:
        """Create a prospective memory (future-triggered recall)."""
        return self._engine.set_prospective(trigger, crystal_id, embedding, expiry_ts)

    def suppress(self, crystal_id: int, reason: str, trigger: str = "") -> int:
        """Create an antibody to suppress a crystal from future recalls."""
        return self._engine.suppress(crystal_id, reason, trigger)

    # ----- Inspection --------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        """Memory health metrics."""
        return self._engine.stats()

    def export(self) -> Dict[str, Any]:
        """Full dump of all memory data (JSON-serializable)."""
        return self._engine.export()

    # ----- Lifecycle ---------------------------------------------------------

    def close(self) -> None:
        """Shut down daemon and close database connection."""
        self._engine.stop_daemon()
        self._storage.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __repr__(self) -> str:
        s = self._engine.stats()
        return (
            f"<Memory crystals={s['active_crystals']} "
            f"entities={s['total_entities']} "
            f"schemas={s['schemas']}>"
        )


__all__ = [
    # Public API
    "Memory",
    # Models
    "SalienceVector",
    "Event",
    "Entity",
    "Crystal",
    "Relation",
    "Contradiction",
    "Activation",
    "ProspectiveMemory",
    "Schema",
    "AntibodyMemory",
    # Backends
    "SQLiteBackend",
    "OllamaEmbedder",
    # Engine (for advanced use)
    "MemoryEngine",
]
