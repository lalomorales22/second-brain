"""
Semantic Gravity Memory — Crystal Forge

The formation pipeline: raw text → structured memory crystal.
Orchestrates entity extraction, salience scoring, self-state detection,
embedding, contradiction checking, and relation creation.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from semantic_gravity_memory.models import (
    Crystal,
    Entity,
    Event,
    Relation,
    SalienceVector,
)
from semantic_gravity_memory.storage.base import BaseStorage
from semantic_gravity_memory.embeddings.base import BaseEmbedder
from semantic_gravity_memory.core.entity_extractor import (
    extract_entities,
    extract_relationships,
    find_co_occurrences,
)
from semantic_gravity_memory.core.salience import score_salience
from semantic_gravity_memory.core.self_state import SelfStateDetector
from semantic_gravity_memory.core.contradiction import ContradictionDetector
from semantic_gravity_memory.utils import now_iso, summarize_text, content_tokens


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACTOR_CONFIDENCE: Dict[str, float] = {
    "user": 0.78,
    "assistant": 0.62,
    "system": 0.50,
}

SEMANTIC_INDICATORS = frozenset({
    "i always", "i never", "i prefer", "i believe", "my rule",
    "in general", "typically", "usually", "my approach is",
    "the way i", "my philosophy", "as a principle",
})


# ---------------------------------------------------------------------------
# Crystal Forge
# ---------------------------------------------------------------------------


class CrystalForge:
    """Transforms raw text into structured memory crystals."""

    def __init__(
        self,
        storage: BaseStorage,
        embedder: Optional[BaseEmbedder] = None,
        self_state_detector: Optional[SelfStateDetector] = None,
        contradiction_detector: Optional[ContradictionDetector] = None,
    ):
        self.storage = storage
        self.embedder = embedder
        self.self_state_detector = self_state_detector or SelfStateDetector(storage)
        self.contradiction_detector = contradiction_detector or ContradictionDetector(storage)

    # -----------------------------------------------------------------
    # Main pipeline
    # -----------------------------------------------------------------

    def ingest(
        self,
        text: str,
        actor: str = "user",
        kind: str = "chat_message",
        context: Optional[Dict] = None,
    ) -> Tuple[int, int]:
        """Full ingestion: text → event + crystal.

        Returns (event_id, crystal_id).
        """
        # 1. Entity extraction
        raw_entities = extract_entities(text)
        entity_names = [name for name, _ in raw_entities]

        # 2. Self-state detection
        self_state = self.self_state_detector.detect(text, entity_names)

        # 3. Salience scoring (with optional novelty from embeddings)
        recent_embeddings = self._recent_embeddings(limit=20)
        text_embedding = self._embed(text)
        salience = score_salience(
            text,
            self_state=self_state,
            recent_embeddings=recent_embeddings,
            query_embedding=text_embedding,
        )

        # 4. Store event
        event = Event(
            actor=actor,
            kind=kind,
            content=text,
            context=context or {},
            salience=salience.combined(),
            embedding=text_embedding,
        )
        event_id = self.storage.insert_event(event)

        # 5. Upsert entities
        entity_ids: List[int] = []
        entity_name_to_id: Dict[str, int] = {}
        for name, ent_kind in raw_entities:
            ent = Entity(name=name, kind=ent_kind, salience=0.15)
            eid = self.storage.upsert_entity(ent)
            entity_ids.append(eid)
            entity_name_to_id[name] = eid

        # 6. Co-occurrence relations
        for name_a, name_b in find_co_occurrences(raw_entities):
            id_a = entity_name_to_id.get(name_a)
            id_b = entity_name_to_id.get(name_b)
            if id_a and id_b:
                self.storage.insert_relation(Relation(
                    source_type="entity", source_id=id_a,
                    target_type="entity", target_id=id_b,
                    relation="co_occurred", weight=0.4,
                ))

        # 7. Extracted relationships
        for subj, pred, obj in extract_relationships(text):
            subj_id = entity_name_to_id.get(subj)
            obj_id = entity_name_to_id.get(obj)
            if subj_id and obj_id:
                self.storage.insert_relation(Relation(
                    source_type="entity", source_id=subj_id,
                    target_type="entity", target_id=obj_id,
                    relation=pred, weight=0.7,
                ))

        # 8. Build crystal
        title = self._make_title(text, raw_entities)
        summary = summarize_text(text, 220)
        future_impl = self._infer_future_implications(text)
        unresolved = self._infer_unresolved(text)
        memory_type = self._classify_memory_type(text)

        crystal_embedding = self._embed(
            f"theme: {title}\nsummary: {summary}\nfuture: {future_impl}"
        )

        crystal = Crystal(
            title=title,
            theme=title,
            summary=summary,
            compressed_narrative=f"{title}: {summary}",
            source_event_ids=[event_id],
            entity_ids=entity_ids,
            salience=salience,
            confidence=ACTOR_CONFIDENCE.get(actor, 0.55),
            self_state=self_state,
            future_implications=future_impl,
            unresolved=unresolved,
            contradiction_state="tension" if unresolved else "clean",
            valid_from_ts=now_iso(),
            embedding=crystal_embedding,
            memory_type=memory_type,
            decay_rate=self._decay_rate(salience, actor),
        )
        crystal_id = self.storage.insert_crystal(crystal)

        # 9. Crystal→entity and event→crystal relations
        for eid in entity_ids:
            self.storage.insert_relation(Relation(
                source_type="crystal", source_id=crystal_id,
                target_type="entity", target_id=eid,
                relation="mentions", weight=0.6,
            ))
        self.storage.insert_relation(Relation(
            source_type="event", source_id=event_id,
            target_type="crystal", target_id=crystal_id,
            relation="crystallized_into", weight=0.9,
        ))

        # 10. Contradiction check
        contradictions = self.contradiction_detector.check_all(text, event_id)
        for contra in contradictions:
            self.storage.insert_contradiction(contra)
        if contradictions:
            crystal.id = crystal_id
            crystal.contradiction_state = "tension"
            self.storage.update_crystal(crystal)

        # 11. Self-state learning
        self.self_state_detector.learn(entity_names, self_state)

        return event_id, crystal_id

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    def _embed(self, text: str) -> Optional[List[float]]:
        if not self.embedder:
            return None
        try:
            return self.embedder.embed(text)
        except Exception:
            return None

    def _recent_embeddings(self, limit: int = 20) -> List[List[float]]:
        recent = self.storage.recent_crystals(limit=limit)
        return [c.embedding for c in recent if c.embedding]

    @staticmethod
    def _make_title(text: str, entities: List[Tuple[str, str]]) -> str:
        if entities:
            return summarize_text(", ".join(e[0] for e in entities[:4]), 64)
        tokens = content_tokens(text)
        if tokens:
            return summarize_text(" ".join(tokens[:6]), 64)
        return "memory crystal"

    @staticmethod
    def _infer_future_implications(text: str) -> str:
        t = text.lower()
        bits: List[str] = []
        if any(x in t for x in ("build", "implement", "prototype", "app", "create")):
            bits.append("likely leads to implementation work")
        if any(x in t for x in ("tax", "invoice", "contract", "bid", "legal")):
            bits.append("may affect business or compliance decisions")
        if any(x in t for x in ("exam", "assignment", "class", "study")):
            bits.append("may affect academic workload")
        if any(x in t for x in ("deploy", "launch", "release", "ship")):
            bits.append("production impact expected")
        if any(x in t for x in ("meeting", "call", "presentation")):
            bits.append("upcoming coordination event")
        if "?" in text:
            bits.append("open question likely to resurface")
        return "; ".join(bits) or "latent relevance"

    @staticmethod
    def _infer_unresolved(text: str) -> str:
        t = text.lower()
        if "?" in text:
            return summarize_text(text, 160)
        for key in ("todo", "need to", "must", "should", "unclear",
                     "not sure", "figure out", "have to", "gotta", "haven't"):
            if key in t:
                return summarize_text(text, 160)
        return ""

    @staticmethod
    def _classify_memory_type(text: str) -> str:
        t = text.lower()
        for indicator in SEMANTIC_INDICATORS:
            if indicator in t:
                return "semantic"
        return "episodic"

    @staticmethod
    def _decay_rate(salience: SalienceVector, actor: str) -> float:
        base = 0.15 if actor == "user" else 0.25
        return max(0.02, base - salience.combined() * 0.12)
