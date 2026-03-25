"""Core engine modules for the Semantic Gravity Memory system."""

from semantic_gravity_memory.core.entity_extractor import (
    extract_entities,
    extract_relationships,
    find_co_occurrences,
)
from semantic_gravity_memory.core.salience import score_salience
from semantic_gravity_memory.core.self_state import SelfStateDetector
from semantic_gravity_memory.core.contradiction import ContradictionDetector
from semantic_gravity_memory.core.crystal_forge import CrystalForge
from semantic_gravity_memory.core.temporal import (
    crystal_strength,
    reinforce_crystal,
    cluster_crystals,
    create_episode_relations,
    get_episode_members,
    temporal_gravity,
    temporal_proximity_bonus,
    recency_score,
    create_prospective,
    check_prospective_triggers,
    fire_prospective,
    fire_all_triggered,
    version_crystal,
    get_crystal_history,
    belief_at_version,
    decay_all_crystals,
    auto_cluster,
)

__all__ = [
    "extract_entities",
    "extract_relationships",
    "find_co_occurrences",
    "score_salience",
    "SelfStateDetector",
    "ContradictionDetector",
    "CrystalForge",
    "crystal_strength",
    "reinforce_crystal",
    "cluster_crystals",
    "create_episode_relations",
    "get_episode_members",
    "temporal_gravity",
    "temporal_proximity_bonus",
    "recency_score",
    "create_prospective",
    "check_prospective_triggers",
    "fire_prospective",
    "fire_all_triggered",
    "version_crystal",
    "get_crystal_history",
    "belief_at_version",
    "decay_all_crystals",
    "auto_cluster",
    "RetrievalEngine",
    "WorkingMemoryBuffer",
    "spread_activation",
    "MetaMemory",
    "create_antibody",
    "check_antibodies",
    "deactivate_antibody",
    "Consolidator",
    "ConsolidationDaemon",
    "MemoryEngine",
]

from semantic_gravity_memory.core.retrieval import (
    RetrievalEngine,
    WorkingMemoryBuffer,
    spread_activation,
)
from semantic_gravity_memory.core.metamemory import MetaMemory
from semantic_gravity_memory.core.immune import (
    create_antibody,
    check_antibodies,
    deactivate_antibody,
)
from semantic_gravity_memory.core.consolidation import (
    Consolidator,
    ConsolidationDaemon,
)
from semantic_gravity_memory.core.engine import MemoryEngine
