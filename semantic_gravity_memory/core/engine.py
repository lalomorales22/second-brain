"""
Semantic Gravity Memory — Memory Engine

The main orchestrator.  Ties together:
  Crystal forge, retrieval, consolidation, temporal engine,
  metamemory, and the immune system.

Usage:
    engine = MemoryEngine(storage, embedder)
    engine.ingest("user said something")
    scene = engine.recall("what did the user say?")
    engine.consolidate()
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional, Tuple

from semantic_gravity_memory.storage.base import BaseStorage
from semantic_gravity_memory.embeddings.base import BaseEmbedder
from semantic_gravity_memory.core.crystal_forge import CrystalForge
from semantic_gravity_memory.core.retrieval import RetrievalEngine
from semantic_gravity_memory.core.consolidation import (
    Consolidator,
    ConsolidationDaemon,
)
from semantic_gravity_memory.core.metamemory import MetaMemory
from semantic_gravity_memory.core.temporal import create_prospective
from semantic_gravity_memory.core.immune import create_antibody
from semantic_gravity_memory.utils import summarize_text


class MemoryEngine:
    """Top-level orchestrator for the Semantic Gravity Memory system."""

    def __init__(
        self,
        storage: BaseStorage,
        embedder: Optional[BaseEmbedder] = None,
        max_recall: int = 8,
        carrying_capacity: int = 500,
    ):
        self.storage = storage
        self.embedder = embedder

        # Sub-systems
        self.forge = CrystalForge(storage, embedder)
        self.metamemory = MetaMemory(storage)
        self.retrieval = RetrievalEngine(
            storage, embedder, metamemory=self.metamemory,
            max_recall=max_recall,
        )
        self.consolidator = Consolidator(
            storage, carrying_capacity=carrying_capacity,
        )
        self._daemon: Optional[ConsolidationDaemon] = None

    # -----------------------------------------------------------------
    # Ingestion
    # -----------------------------------------------------------------

    def ingest(
        self,
        text: str,
        actor: str = "user",
        kind: str = "chat_message",
        context: Optional[Dict] = None,
    ) -> Tuple[int, int]:
        """Ingest text into the memory system.

        Returns (event_id, crystal_id).
        """
        return self.forge.ingest(text, actor, kind, context)

    # -----------------------------------------------------------------
    # Recall
    # -----------------------------------------------------------------

    def recall(
        self,
        query: str,
        self_state: Optional[str] = None,
        now_ts: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Reconstruct a memory scene for the given query.

        Returns a scene dict with crystals, entities, contradictions,
        prospective fires, suppressions, and a narrative summary.
        """
        return self.retrieval.recall(query, self_state, now_ts)

    # -----------------------------------------------------------------
    # Answer (recall + LLM)
    # -----------------------------------------------------------------

    def answer(
        self,
        query: str,
        chat_fn: Callable[[str], str],
        self_state: Optional[str] = None,
        now_ts: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Recall context, call *chat_fn* with a grounded prompt,
        ingest the response, and return (answer_text, scene).

        *chat_fn* receives a formatted prompt string and returns the
        model's response text.
        """
        scene = self.recall(query, self_state, now_ts)
        prompt = self._format_prompt(scene, query)
        answer_text = chat_fn(prompt)

        # Ingest the assistant response
        self.ingest(
            answer_text,
            actor="assistant",
            kind="chat_response",
            context={"query": query, "activation_id": scene.get("activation_id")},
        )
        return answer_text, scene

    # -----------------------------------------------------------------
    # Consolidation
    # -----------------------------------------------------------------

    def consolidate(self, now_ts: Optional[str] = None) -> Dict[str, Any]:
        """Run a single consolidation pass (synchronous)."""
        return self.consolidator.run_pass(now_ts)

    def start_daemon(self, heartbeat_seconds: float = 300.0) -> None:
        """Start the background consolidation daemon."""
        if self._daemon and self._daemon.running:
            return
        self._daemon = ConsolidationDaemon(
            self.consolidator, heartbeat_seconds,
        )
        self._daemon.start()

    def stop_daemon(self) -> None:
        """Stop the background consolidation daemon."""
        if self._daemon:
            self._daemon.stop()

    @property
    def daemon_running(self) -> bool:
        return self._daemon is not None and self._daemon.running

    # -----------------------------------------------------------------
    # Feedback
    # -----------------------------------------------------------------

    def feedback(
        self,
        activation_id: int,
        quality: float,
        self_state: str = "general",
    ) -> None:
        """Record quality feedback for a retrieval activation."""
        self.metamemory.record_feedback(activation_id, quality, self_state)

    # -----------------------------------------------------------------
    # Prospective memory
    # -----------------------------------------------------------------

    def set_prospective(
        self,
        trigger_description: str,
        payload_crystal_id: int,
        trigger_embedding: Optional[List[float]] = None,
        expiry_ts: Optional[str] = None,
    ) -> int:
        """Create a prospective memory (future-triggered recall)."""
        return create_prospective(
            self.storage,
            trigger_description,
            payload_crystal_id,
            trigger_embedding,
            expiry_ts,
        )

    # -----------------------------------------------------------------
    # Immune system
    # -----------------------------------------------------------------

    def suppress(
        self,
        crystal_id: int,
        reason: str,
        trigger_description: str = "",
        trigger_embedding: Optional[List[float]] = None,
    ) -> int:
        """Create an antibody to suppress a crystal from future recalls."""
        return create_antibody(
            self.storage,
            trigger_description=trigger_description or f"suppress crystal {crystal_id}",
            suppress_crystal_id=crystal_id,
            reason=reason,
            trigger_embedding=trigger_embedding,
        )

    # -----------------------------------------------------------------
    # Export & stats
    # -----------------------------------------------------------------

    def export(self) -> Dict[str, Any]:
        """Full dump of all memory data."""
        return self.storage.export_all()

    def stats(self) -> Dict[str, Any]:
        """Memory health metrics."""
        all_c = self.storage.all_crystals()
        active = [c for c in all_c if not c.valid_to_ts]
        semantic = [c for c in active if c.memory_type == "semantic"]
        episodic = [c for c in active if c.memory_type == "episodic"]

        return {
            "total_crystals": len(all_c),
            "active_crystals": len(active),
            "dormant_crystals": len(all_c) - len(active),
            "semantic_crystals": len(semantic),
            "episodic_crystals": len(episodic),
            "total_entities": len(self.storage.top_entities(limit=10000)),
            "open_contradictions": len(self.storage.open_contradictions()),
            "schemas": len(self.storage.all_schemas()),
            "active_antibodies": len(self.storage.active_antibodies()),
            "active_prospective": len(self.storage.active_prospective_memories()),
            "domain_confidences": self.metamemory.all_domain_confidences(),
            "consolidation_log": self.consolidator.get_log(limit=5),
            "working_memory": self.retrieval.working_memory.contents(),
        }

    # -----------------------------------------------------------------
    # Prompt formatting
    # -----------------------------------------------------------------

    @staticmethod
    def _format_prompt(scene: Dict[str, Any], query: str) -> str:
        lines = [
            "You are an AI using a semantic gravity memory system.",
            "Answer naturally using the reconstructed memory scene below.",
            "If memory is sparse, say so. Do not invent facts.",
            "",
            "=== Reconstructed Memory Scene ===",
            f"Self-state: {scene.get('active_self_state', 'general')}",
        ]

        crystals = scene.get("crystals", [])
        if crystals:
            lines.append("\n--- Active Memories ---")
            for c in crystals:
                tag = c.get("memory_type", "episodic")
                lines.append(f"[{tag}] {c.get('title', '')}: {c.get('summary', '')}")
                unresolved = c.get("unresolved", "")
                if unresolved:
                    lines.append(f"  Unresolved: {unresolved}")

        entities = scene.get("entities", [])
        if entities:
            lines.append("\n--- Active Entities ---")
            lines.append(", ".join(e["name"] for e in entities[:10]))

        contradictions = scene.get("contradictions", [])
        if contradictions:
            lines.append("\n--- Open Contradictions ---")
            for ct in contradictions[:4]:
                lines.append(
                    f"  {ct.get('topic', '?')}: "
                    f"{summarize_text(ct.get('claim_a', ''), 60)} vs "
                    f"{summarize_text(ct.get('claim_b', ''), 60)}"
                )

        lines.append(f"\n--- User Query ---\n{query}")
        lines.append("\nRespond in a grounded way using the scene above.")
        return "\n".join(lines)
