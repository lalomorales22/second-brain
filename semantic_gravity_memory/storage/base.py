"""
Semantic Gravity Memory — Abstract Storage Interface

Every storage backend (SQLite, Postgres, in-memory) must implement this.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from semantic_gravity_memory.models import (
    Activation,
    AntibodyMemory,
    Contradiction,
    Crystal,
    Entity,
    Event,
    ProspectiveMemory,
    Relation,
    Schema,
)


class BaseStorage(ABC):
    """Abstract contract for all memory storage backends."""

    # -- Events ---------------------------------------------------------------

    @abstractmethod
    def insert_event(self, event: Event) -> int:
        """Store an event, return its assigned id."""
        ...

    @abstractmethod
    def get_event(self, event_id: int) -> Optional[Event]:
        ...

    @abstractmethod
    def recent_events(self, limit: int = 50) -> List[Event]:
        """Most recent events, newest first."""
        ...

    # -- Entities -------------------------------------------------------------

    @abstractmethod
    def upsert_entity(self, entity: Entity) -> int:
        """Insert or update by name. Returns entity id."""
        ...

    @abstractmethod
    def get_entity(self, entity_id: int) -> Optional[Entity]:
        ...

    @abstractmethod
    def get_entity_by_name(self, name: str) -> Optional[Entity]:
        ...

    @abstractmethod
    def top_entities(self, limit: int = 50) -> List[Entity]:
        """Entities sorted by salience descending."""
        ...

    # -- Crystals -------------------------------------------------------------

    @abstractmethod
    def insert_crystal(self, crystal: Crystal) -> int:
        ...

    @abstractmethod
    def update_crystal(self, crystal: Crystal) -> None:
        """Update an existing crystal (must have a valid id)."""
        ...

    @abstractmethod
    def get_crystal(self, crystal_id: int) -> Optional[Crystal]:
        ...

    @abstractmethod
    def all_crystals(self) -> List[Crystal]:
        ...

    @abstractmethod
    def recent_crystals(self, limit: int = 50) -> List[Crystal]:
        """Most recent crystals, newest first."""
        ...

    # -- Relations ------------------------------------------------------------

    @abstractmethod
    def insert_relation(self, relation: Relation) -> int:
        ...

    @abstractmethod
    def relations_from(self, source_type: str, source_id: int) -> List[Relation]:
        ...

    @abstractmethod
    def relations_to(self, target_type: str, target_id: int) -> List[Relation]:
        ...

    @abstractmethod
    def all_relations(self) -> List[Relation]:
        ...

    # -- Contradictions -------------------------------------------------------

    @abstractmethod
    def insert_contradiction(self, contradiction: Contradiction) -> int:
        ...

    @abstractmethod
    def update_contradiction(self, contradiction: Contradiction) -> None:
        ...

    @abstractmethod
    def open_contradictions(self) -> List[Contradiction]:
        """Contradictions with resolution_state == 'open'."""
        ...

    @abstractmethod
    def all_contradictions(self) -> List[Contradiction]:
        ...

    # -- Activations ----------------------------------------------------------

    @abstractmethod
    def insert_activation(self, activation: Activation) -> int:
        ...

    @abstractmethod
    def recent_activations(self, limit: int = 50) -> List[Activation]:
        ...

    # -- Prospective Memory ---------------------------------------------------

    @abstractmethod
    def insert_prospective(self, pm: ProspectiveMemory) -> int:
        ...

    @abstractmethod
    def active_prospective_memories(self) -> List[ProspectiveMemory]:
        """Unfired, non-expired prospective memories."""
        ...

    @abstractmethod
    def fire_prospective(self, pm_id: int, fired_ts: str) -> None:
        ...

    # -- Schemas --------------------------------------------------------------

    @abstractmethod
    def insert_schema(self, schema: Schema) -> int:
        ...

    @abstractmethod
    def update_schema(self, schema: Schema) -> None:
        ...

    @abstractmethod
    def all_schemas(self) -> List[Schema]:
        ...

    # -- Antibodies -----------------------------------------------------------

    @abstractmethod
    def insert_antibody(self, antibody: AntibodyMemory) -> int:
        ...

    @abstractmethod
    def active_antibodies(self) -> List[AntibodyMemory]:
        """Antibodies where active == True."""
        ...

    # -- Meta -----------------------------------------------------------------

    @abstractmethod
    def set_meta(self, key: str, value: str) -> None:
        ...

    @abstractmethod
    def get_meta(self, key: str, default: str = "") -> str:
        ...

    # -- Export ---------------------------------------------------------------

    @abstractmethod
    def export_all(self) -> Dict[str, Any]:
        """Full dump of all tables as a dict (JSON-serializable)."""
        ...

    # -- Lifecycle ------------------------------------------------------------

    @abstractmethod
    def close(self) -> None:
        ...

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
