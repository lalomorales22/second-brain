"""
Semantic Gravity Memory — Data Models

All core data structures. Zero external dependencies.
Every memory object in the system is defined here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SalienceVector:
    """Multi-dimensional salience scoring.

    Six dimensions capturing *why* something matters, not just *how much*:
      emotional  — personal significance, stress, excitement, attachment
      practical  — actionable, deadlines, tasks, things to build or fix
      identity   — relates to who the user is (role, self-concept, values)
      temporal   — time-sensitive, urgent, expiring, happening soon
      uncertainty — unresolved, confusing, contradictory, open questions
      novelty    — new, surprising, first encounter with this topic
    """

    emotional: float = 0.0
    practical: float = 0.0
    identity: float = 0.0
    temporal: float = 0.0
    uncertainty: float = 0.0
    novelty: float = 0.0

    def combined(self, weights: Optional[Dict[str, float]] = None) -> float:
        """Weighted combination of all dimensions into a single score."""
        w = weights or {
            "emotional": 0.15,
            "practical": 0.25,
            "identity": 0.15,
            "temporal": 0.20,
            "uncertainty": 0.15,
            "novelty": 0.10,
        }
        total = 0.0
        for dim, weight in w.items():
            total += getattr(self, dim, 0.0) * weight
        return total

    def peak_dimension(self) -> str:
        """Return the name of the highest-scoring dimension."""
        dims = self.to_dict()
        return max(dims, key=dims.get)  # type: ignore[arg-type]

    def to_dict(self) -> Dict[str, float]:
        return {
            "emotional": self.emotional,
            "practical": self.practical,
            "identity": self.identity,
            "temporal": self.temporal,
            "uncertainty": self.uncertainty,
            "novelty": self.novelty,
        }

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, float]]) -> SalienceVector:
        if not d:
            return cls()
        return cls(
            emotional=float(d.get("emotional", 0.0)),
            practical=float(d.get("practical", 0.0)),
            identity=float(d.get("identity", 0.0)),
            temporal=float(d.get("temporal", 0.0)),
            uncertainty=float(d.get("uncertainty", 0.0)),
            novelty=float(d.get("novelty", 0.0)),
        )


# ---------------------------------------------------------------------------
# Core memory objects
# ---------------------------------------------------------------------------


@dataclass
class Event:
    """Raw experience record — the atomic unit of input.

    Every user message, assistant response, and system observation
    starts life as an Event before any processing.
    """

    id: Optional[int] = None
    ts: str = ""
    actor: str = ""  # "user", "assistant", "system"
    kind: str = ""  # "chat_message", "note", "observation", "seed"
    content: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    salience: float = 0.0
    embedding: Optional[List[float]] = None


@dataclass
class Entity:
    """Named concept extracted from the event stream.

    Entities are reinforced over time — each mention bumps salience
    and mention_count, creating a natural importance signal.
    """

    id: Optional[int] = None
    name: str = ""
    kind: str = "concept"  # "concept", "tool", "person", "place", "project"
    first_seen_ts: str = ""
    last_seen_ts: str = ""
    salience: float = 0.0
    mention_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Crystal:
    """Memory crystal — the primary unit of the memory system.

    A crystal is a structured, compressed memory object formed from
    one or more events. It carries multi-dimensional salience,
    temporal anchoring, contradiction awareness, decay characteristics,
    and version history.

    Key fields beyond basic RAG:
      memory_type       — "episodic" (what happened) or "semantic" (what I know)
      access_count      — how often recalled, drives reinforcement
      last_accessed_ts  — when last recalled, drives decay calculation
      decay_rate        — 0.0 = permanent, 1.0 = ephemeral
      version           — incremented on update, enables belief evolution tracking
      parent_crystal_id — links to merged parent (for consolidation)
      schema_id         — which abstracted schema this crystal belongs to
    """

    id: Optional[int] = None
    created_ts: str = ""
    updated_ts: str = ""
    title: str = ""
    theme: str = ""
    summary: str = ""
    compressed_narrative: str = ""
    source_event_ids: List[int] = field(default_factory=list)
    entity_ids: List[int] = field(default_factory=list)
    salience: SalienceVector = field(default_factory=SalienceVector)
    confidence: float = 0.5
    self_state: str = "general"
    future_implications: str = ""
    unresolved: str = ""
    contradiction_state: str = "clean"  # "clean", "tension", "resolved"
    valid_from_ts: Optional[str] = None
    valid_to_ts: Optional[str] = None
    embedding: Optional[List[float]] = None
    memory_type: str = "episodic"  # "episodic" or "semantic"
    access_count: int = 0
    last_accessed_ts: Optional[str] = None
    decay_rate: float = 0.1
    version: int = 1
    parent_crystal_id: Optional[int] = None
    schema_id: Optional[int] = None


@dataclass
class Relation:
    """Typed weighted edge between any two memory objects.

    source_type/target_type can be: "event", "crystal", "entity", "schema"
    relation examples: "mentions", "crystallized_into", "contradicts",
                       "co_occurred", "caused", "temporal_cluster"
    """

    id: Optional[int] = None
    source_type: str = ""
    source_id: int = 0
    target_type: str = ""
    target_id: int = 0
    relation: str = ""
    weight: float = 0.5
    context: Dict[str, Any] = field(default_factory=dict)
    created_ts: str = ""


@dataclass
class Contradiction:
    """Recorded tension between two claims.

    The system stores conflict instead of silently overwriting.
    resolution_state tracks the lifecycle:
      "open"       — conflict detected, not yet resolved
      "resolved_a" — claim_a won (more evidence, user confirmed)
      "resolved_b" — claim_b won
      "dissolved"  — conflict no longer relevant
    """

    id: Optional[int] = None
    ts: str = ""
    topic: str = ""
    claim_a: str = ""
    claim_b: str = ""
    evidence_event_a: Optional[int] = None
    evidence_event_b: Optional[int] = None
    resolution_state: str = "open"
    resolution_ts: Optional[str] = None
    notes: str = ""


@dataclass
class Activation:
    """Record of a single recall/retrieval event.

    Stores what the memory system surfaced for a given query,
    including metamemory quality feedback.
    """

    id: Optional[int] = None
    ts: str = ""
    query: str = ""
    active_self_state: str = "general"
    crystal_ids: List[int] = field(default_factory=list)
    entity_ids: List[int] = field(default_factory=list)
    scene: Dict[str, Any] = field(default_factory=dict)
    quality_score: Optional[float] = None


@dataclass
class ProspectiveMemory:
    """Future-triggered recall — remembering to remember.

    When incoming text matches the trigger embedding above a threshold,
    the payload crystal gets injected into the active scene.
    """

    id: Optional[int] = None
    created_ts: str = ""
    trigger_description: str = ""
    trigger_embedding: Optional[List[float]] = None
    payload_crystal_id: int = 0
    expiry_ts: Optional[str] = None
    fired: bool = False
    fired_ts: Optional[str] = None


@dataclass
class Schema:
    """Abstracted pattern extracted from multiple crystals.

    After seeing many similar crystals, the consolidation daemon
    abstracts them into a Schema — a reusable template that
    guides future recall and prediction.
    """

    id: Optional[int] = None
    created_ts: str = ""
    updated_ts: str = ""
    name: str = ""
    description: str = ""
    pattern: str = ""
    source_crystal_ids: List[int] = field(default_factory=list)
    slot_definitions: Dict[str, Any] = field(default_factory=dict)
    activation_count: int = 0
    embedding: Optional[List[float]] = None


@dataclass
class AntibodyMemory:
    """Immune system memory — suppresses known-bad recall patterns.

    Created when a user corrects a bad retrieval. The antibody
    sits in the retrieval path and actively suppresses the target
    crystal from entering scenes when similar triggers appear.
    """

    id: Optional[int] = None
    created_ts: str = ""
    trigger_description: str = ""
    trigger_embedding: Optional[List[float]] = None
    suppress_crystal_id: int = 0
    reason: str = ""
    active: bool = True
