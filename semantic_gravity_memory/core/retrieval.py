"""
Semantic Gravity Memory — Gravitational Retrieval & Activation

Revolutionary retrieval: memories aren't searched, they're awakened.

Pipeline:
  Phase 1  Entity Gateway    — cue extraction → graph traversal (zero embeddings)
  Phase 2  Gravity Orbit     — heaviest crystals always accessible (pre-computed mass)
  Phase 3  Resonance Field   — recent recall neighborhoods stay primed (semantic priming)
  Phase 4  Selective scoring  — embeddings compared ONLY on the narrowed candidate set
  Phase 5  Spreading activation — energy propagates through the relation graph
  Phase 6  Scene reconstruction — activated crystals + entities + contradictions

Traditional RAG: query → embed → compare ALL vectors → top-k.
This system:     query → cue → graph → gravity → resonance → selective embed → spread → scene.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from semantic_gravity_memory.models import (
    Activation,
    Crystal,
    Entity,
    Relation,
    SalienceVector,
)
from semantic_gravity_memory.storage.base import BaseStorage
from semantic_gravity_memory.embeddings.base import BaseEmbedder
from semantic_gravity_memory.core.self_state import SelfStateDetector
from semantic_gravity_memory.core.temporal import (
    check_prospective_triggers,
    crystal_strength,
    fire_all_triggered,
    recency_score,
    reinforce_crystal,
)
from semantic_gravity_memory.core.immune import check_antibodies
from semantic_gravity_memory.core.metamemory import MetaMemory
from semantic_gravity_memory.utils import (
    cosine_similarity,
    now_iso,
    summarize_text,
)


# =========================================================================
# Spreading Activation (standalone, testable)
# =========================================================================


def spread_activation(
    storage: BaseStorage,
    seed_scores: Dict[Tuple[str, int], float],
    max_hops: int = 3,
    hop_decay: float = 0.5,
    min_energy: float = 0.01,
) -> Dict[Tuple[str, int], float]:
    """Propagate activation energy through the relation graph.

    Starts from *seed_scores* ``{(node_type, node_id): energy}`` and
    follows relation edges bidirectionally, decaying energy at each hop.

    Returns ``{(node_type, node_id): accumulated_energy}`` for all
    reached nodes (including seeds).
    """
    energies: Dict[Tuple[str, int], float] = dict(seed_scores)
    frontier: List[Tuple[str, int, float]] = [
        (ntype, nid, energy) for (ntype, nid), energy in seed_scores.items()
    ]

    for _hop in range(max_hops):
        next_frontier: List[Tuple[str, int, float]] = []
        for node_type, node_id, energy in frontier:
            if energy < min_energy:
                continue

            # Forward edges
            for rel in storage.relations_from(node_type, node_id):
                key = (rel.target_type, rel.target_id)
                propagated = energy * rel.weight * hop_decay
                if propagated >= min_energy and propagated > energies.get(key, 0.0):
                    energies[key] = propagated
                    next_frontier.append((rel.target_type, rel.target_id, propagated))

            # Reverse edges
            for rel in storage.relations_to(node_type, node_id):
                key = (rel.source_type, rel.source_id)
                propagated = energy * rel.weight * hop_decay
                if propagated >= min_energy and propagated > energies.get(key, 0.0):
                    energies[key] = propagated
                    next_frontier.append((rel.source_type, rel.source_id, propagated))

        frontier = next_frontier
        if not frontier:
            break

    return energies


# =========================================================================
# Working Memory Buffer
# =========================================================================


class WorkingMemoryBuffer:
    """Limited-capacity buffer of crystal ids that persist across queries.

    Crystals in the buffer get a retrieval bonus without needing an
    embedding match.  When capacity is exceeded the oldest entry is evicted.
    """

    def __init__(self, capacity: int = 7):
        self.capacity = capacity
        self._buffer: List[int] = []

    def add(self, crystal_id: int) -> None:
        if crystal_id in self._buffer:
            self._buffer.remove(crystal_id)
        self._buffer.append(crystal_id)
        while len(self._buffer) > self.capacity:
            self._buffer.pop(0)

    def add_many(self, crystal_ids: List[int]) -> None:
        for cid in crystal_ids:
            self.add(cid)

    def contents(self) -> List[int]:
        return list(self._buffer)

    def contains(self, crystal_id: int) -> bool:
        return crystal_id in self._buffer

    def clear(self) -> None:
        self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)


# =========================================================================
# Resonance Field (semantic priming)
# =========================================================================


class ResonanceField:
    """Persistent activation neighborhood — semantic priming across queries.

    When you recall memories about "Python programming", not only do those
    crystals stay warm, but their entire graph neighborhood does too.
    Follow-up questions about related topics are nearly instant because the
    relevant crystals are already primed.

    This mirrors priming effects in human cognition: thinking about dogs
    makes it easier to recall cats, pets, walks, veterinarians...
    """

    def __init__(self, capacity: int = 60, decay_rate: float = 0.15):
        self.capacity = capacity
        self.decay_rate = decay_rate
        self._field: Dict[int, float] = {}  # crystal_id → resonance strength

    def activate(self, crystal_ids: List[int], storage: BaseStorage) -> None:
        """Prime the field with recalled crystals and their graph neighbors."""
        # Decay existing resonance first
        self._decay()

        # Primary crystals get full resonance
        for cid in crystal_ids:
            self._field[cid] = 1.0

        # 1-hop graph neighbors get partial resonance
        for cid in crystal_ids:
            for rel in storage.relations_from("crystal", cid):
                if rel.target_type == "crystal":
                    self._field[rel.target_id] = max(
                        self._field.get(rel.target_id, 0),
                        rel.weight * 0.5,
                    )
            for rel in storage.relations_to("crystal", cid):
                if rel.source_type == "crystal":
                    self._field[rel.source_id] = max(
                        self._field.get(rel.source_id, 0),
                        rel.weight * 0.5,
                    )

        # Evict weakest if over capacity
        if len(self._field) > self.capacity:
            sorted_items = sorted(self._field.items(), key=lambda x: x[1])
            for k, _ in sorted_items[: len(self._field) - self.capacity]:
                del self._field[k]

    def _decay(self) -> None:
        to_remove = []
        for k in self._field:
            self._field[k] *= (1.0 - self.decay_rate)
            if self._field[k] < 0.05:
                to_remove.append(k)
        for k in to_remove:
            del self._field[k]

    def resonant_ids(self, min_strength: float = 0.1) -> List[int]:
        """Crystal IDs with resonance above threshold."""
        return [cid for cid, s in self._field.items() if s >= min_strength]

    def bonus(self, crystal_id: int) -> float:
        """Resonance bonus for a crystal (0 to 0.30)."""
        return self._field.get(crystal_id, 0.0) * 0.30

    def __len__(self) -> int:
        return len(self._field)


# =========================================================================
# Entity Cue Matching
# =========================================================================


def match_entity_cues(
    query: str,
    entity_names_and_ids: List[Tuple[int, str]],
) -> List[int]:
    """Find entity IDs whose names appear in the query text.

    Cue-dependent retrieval: specific words in the query directly
    activate associated memory traces through the knowledge graph,
    bypassing embedding comparison entirely.

    This is O(entities) string matching — fast even with thousands of
    entities because names are short strings.
    """
    query_lower = query.lower()
    matched: List[int] = []
    for eid, name in entity_names_and_ids:
        if len(name) >= 2 and name.lower() in query_lower:
            matched.append(eid)
    return matched


# =========================================================================
# Retrieval Engine
# =========================================================================


class RetrievalEngine:
    """Gravitational recall: query → scene dict.

    Three phases BEFORE any embedding comparison:
      1. Entity Gateway  — extract cues from query, walk graph to crystals
      2. Gravity Orbit   — always include the heaviest (most important) crystals
      3. Resonance Field — primed neighborhood from recent recalls

    Only then: selective embedding scoring on the narrowed candidate set,
    followed by spreading activation and scene reconstruction.
    """

    def __init__(
        self,
        storage: BaseStorage,
        embedder: Optional[BaseEmbedder] = None,
        self_state_detector: Optional[SelfStateDetector] = None,
        metamemory: Optional[MetaMemory] = None,
        max_recall: int = 8,
        max_seeds: int = 12,
        max_hops: int = 3,
        hop_decay: float = 0.5,
        working_memory_capacity: int = 7,
    ):
        self.storage = storage
        self.embedder = embedder
        self.self_state_detector = self_state_detector or SelfStateDetector(storage)
        self.metamemory = metamemory or MetaMemory(storage)
        self.max_recall = max_recall
        self.max_seeds = max_seeds
        self.max_hops = max_hops
        self.hop_decay = hop_decay
        self.working_memory = WorkingMemoryBuffer(working_memory_capacity)
        self.resonance = ResonanceField()

        # Cached entity names for cue matching (refreshed periodically)
        self._entity_cache: List[Tuple[int, str]] = []
        self._entity_cache_size: int = 0

    # -----------------------------------------------------------------
    # Entity cache
    # -----------------------------------------------------------------

    def _refresh_entity_cache(self) -> None:
        """Reload entity names from storage for cue matching."""
        self._entity_cache = self.storage.entity_names_and_ids()
        self._entity_cache_size = len(self._entity_cache)

    # -----------------------------------------------------------------
    # Main recall — Gravitational Retrieval
    # -----------------------------------------------------------------

    def recall(
        self,
        query: str,
        self_state: Optional[str] = None,
        now_ts: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute the gravitational recall pipeline.

        Returns a scene dict suitable for grounding an LLM response.
        """
        ts = now_ts or now_iso()

        # 1. Embed query (single Ollama call)
        q_embedding = self._embed(query)

        # 2. Detect self-state
        active_self = self_state or self.self_state_detector.detect(query)

        # ============================================================
        # PHASE 1: ENTITY GATEWAY (zero embeddings, pure graph)
        # Extract concept cues from the query text, then walk the
        # knowledge graph to find connected crystals.
        # ============================================================
        self._refresh_entity_cache()
        cue_entity_ids = match_entity_cues(query, self._entity_cache)
        gateway_crystals = self.storage.crystals_by_entity_ids(
            cue_entity_ids, limit=40,
        ) if cue_entity_ids else []

        # ============================================================
        # PHASE 2: GRAVITY ORBIT (pre-computed mass)
        # The heaviest crystals are always on the tip of the tongue.
        # Like ACT-R base-level activation: frequently accessed, salient
        # memories have higher resting activation and require less cue
        # strength to trigger.
        # ============================================================
        orbit_crystals = self.storage.top_crystals_by_mass(limit=30)

        # ============================================================
        # PHASE 3: RESONANCE FIELD (semantic priming)
        # Crystals from recent recall neighborhoods are already warm.
        # This makes follow-up questions nearly instant.
        # ============================================================
        resonance_ids = self.resonance.resonant_ids()
        resonance_crystals: List[Crystal] = []
        for rid in resonance_ids:
            c = self.storage.get_crystal(rid)
            if c and not c.valid_to_ts:
                resonance_crystals.append(c)

        # Working memory crystals
        wm_crystals: List[Crystal] = []
        for wid in self.working_memory.contents():
            c = self.storage.get_crystal(wid)
            if c and not c.valid_to_ts:
                wm_crystals.append(c)

        # ============================================================
        # UNION: Deduplicate candidates from all phases
        # ============================================================
        candidates: Dict[int, Crystal] = {}
        for c in gateway_crystals + orbit_crystals + resonance_crystals + wm_crystals:
            if c.id is not None and c.id not in candidates:
                candidates[c.id] = c

        # ============================================================
        # PHASE 4: SELECTIVE SCORING (embeddings only on candidates)
        # Instead of scoring ALL crystals, we score only the narrowed
        # set gathered by the first three phases.
        # ============================================================
        seed_scores: Dict[Tuple[str, int], float] = {}
        for c in candidates.values():
            if c.id is None:
                continue
            score = self._score_crystal(c, q_embedding, active_self, ts)
            if score > 0.03:
                seed_scores[("crystal", c.id)] = score

        # ============================================================
        # PHASE 5: SPREADING ACTIVATION (unchanged — graph propagation)
        # ============================================================
        sorted_seeds = sorted(seed_scores.items(), key=lambda x: -x[1])
        top_seeds = dict(sorted_seeds[: self.max_seeds])
        activated = spread_activation(
            self.storage, top_seeds,
            max_hops=self.max_hops,
            hop_decay=self.hop_decay,
        )

        # Split crystal / entity energies
        crystal_energies: Dict[int, float] = {}
        entity_energies: Dict[int, float] = {}
        for (ntype, nid), energy in activated.items():
            if ntype == "crystal":
                crystal_energies[nid] = energy
            elif ntype == "entity":
                entity_energies[nid] = energy

        # 6. Prospective memory triggers
        triggered_pms = check_prospective_triggers(
            self.storage, query, q_embedding, now_ts=ts,
        )
        fired_crystal_ids = fire_all_triggered(self.storage, triggered_pms, now_ts=ts)
        for cid in fired_crystal_ids:
            crystal_energies[cid] = max(crystal_energies.get(cid, 0.0), 0.80)

        # 7. Antibody filtering
        suppressions = check_antibodies(
            self.storage, query, q_embedding,
            set(crystal_energies.keys()),
        )
        for sup in suppressions:
            crystal_energies.pop(sup["crystal_id"], None)

        # 8. Pick top crystals
        top_ids = sorted(crystal_energies, key=crystal_energies.get, reverse=True)[  # type: ignore[arg-type]
            : self.max_recall
        ]

        # 9. Build scene
        scene_crystals = self._build_crystal_entries(top_ids, crystal_energies)
        scene_entities = self._build_entity_entries(entity_energies)
        contradictions = [
            {"id": c.id, "topic": c.topic, "claim_a": c.claim_a, "claim_b": c.claim_b, "state": c.resolution_state}
            for c in self.storage.open_contradictions()
        ]

        scene: Dict[str, Any] = {
            "query": query,
            "active_self_state": active_self,
            "crystals": scene_crystals,
            "entities": scene_entities,
            "contradictions": contradictions[:8],
            "prospective_fired": fired_crystal_ids,
            "suppressions": suppressions,
            "working_memory": self.working_memory.contents(),
            "scene_narrative": self._narrative(active_self, scene_crystals, scene_entities, query),
        }

        # 10. Record activation
        activation = Activation(
            ts=ts,
            query=query,
            active_self_state=active_self,
            crystal_ids=top_ids,
            entity_ids=list(entity_energies.keys())[:16],
            scene=scene,
        )
        act_id = self.storage.insert_activation(activation)
        scene["activation_id"] = act_id

        # 11. Reinforce recalled crystals + update working memory + prime resonance
        for cid in top_ids:
            try:
                reinforce_crystal(self.storage, cid, now_ts=ts)
            except ValueError:
                pass
        self.working_memory.add_many(top_ids)
        self.resonance.activate(top_ids, self.storage)

        return scene

    # -----------------------------------------------------------------
    # Crystal scoring
    # -----------------------------------------------------------------

    def _score_crystal(
        self,
        crystal: Crystal,
        q_embedding: Optional[List[float]],
        active_self: str,
        now_ts: str,
    ) -> float:
        score = 0.0

        # Embedding similarity
        if q_embedding and crystal.embedding:
            score += cosine_similarity(q_embedding, crystal.embedding)

        # Self-state match
        if crystal.self_state == active_self and active_self != "general":
            score += 0.12

        # Episodic: add recency bonus
        if crystal.memory_type == "episodic" and crystal.created_ts:
            score += recency_score(crystal.created_ts, now_ts=now_ts) * 0.15

        # Working memory bonus
        if crystal.id is not None and self.working_memory.contains(crystal.id):
            score += 0.20

        # Resonance bonus (semantic priming from recent recalls)
        if crystal.id is not None:
            score += self.resonance.bonus(crystal.id)

        # Crystal strength (decay + reinforcement)
        score += crystal_strength(crystal, now_ts=now_ts) * 0.25

        # Domain confidence modulation
        domain_conf = self.metamemory.domain_confidence(crystal.self_state)
        score *= (0.5 + domain_conf * 0.5)  # range: 0.5× to 1.0×

        return score

    # -----------------------------------------------------------------
    # Scene builders
    # -----------------------------------------------------------------

    def _build_crystal_entries(
        self,
        crystal_ids: List[int],
        energies: Dict[int, float],
    ) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        for cid in crystal_ids:
            c = self.storage.get_crystal(cid)
            if not c:
                continue
            entries.append({
                "id": c.id,
                "title": c.title,
                "summary": c.summary,
                "theme": c.theme,
                "activation_energy": round(energies.get(cid, 0.0), 4),
                "self_state": c.self_state,
                "memory_type": c.memory_type,
                "confidence": c.confidence,
                "future_implications": c.future_implications,
                "unresolved": c.unresolved,
                "version": c.version,
            })
        return entries

    def _build_entity_entries(
        self,
        energies: Dict[int, float],
    ) -> List[Dict[str, Any]]:
        top_eids = sorted(energies, key=energies.get, reverse=True)[:16]  # type: ignore[arg-type]
        entries: List[Dict[str, Any]] = []
        for eid in top_eids:
            e = self.storage.get_entity(eid)
            if not e:
                continue
            entries.append({
                "id": e.id,
                "name": e.name,
                "kind": e.kind,
                "salience": e.salience,
                "mention_count": e.mention_count,
                "activation_energy": round(energies.get(eid, 0.0), 4),
            })
        return entries

    @staticmethod
    def _narrative(
        active_self: str,
        crystals: List[Dict],
        entities: List[Dict],
        query: str,
    ) -> str:
        lines = [f"active self-state: {active_self}"]
        if crystals:
            lines.append(
                "dominant crystals: "
                + "; ".join(c["title"] for c in crystals[:4])
            )
        if entities:
            lines.append(
                "active entities: "
                + ", ".join(e["name"] for e in entities[:6])
            )
        lines.append(f"query: {summarize_text(query, 120)}")
        return "\n".join(lines)

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
