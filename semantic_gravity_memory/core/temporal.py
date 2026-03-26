"""
Semantic Gravity Memory — Temporal Engine

Time as a first-class dimension.

Components:
  Crystal strength   — decay + reinforcement math
  Temporal clustering — auto-episode formation from time-proximate crystals
  Temporal gravity    — proximity bonus for retrieval scoring
  Recency weighting   — smooth boost for recent memories
  Prospective memory  — future-triggered recall
  Memory versioning   — belief evolution tracking with snapshots
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from semantic_gravity_memory.models import (
    Crystal,
    ProspectiveMemory,
    Relation,
)
from semantic_gravity_memory.storage.base import BaseStorage
from semantic_gravity_memory.utils import (
    clamp,
    cosine_similarity,
    exponential_decay,
    is_expired,
    now_iso,
    safe_json_dumps,
    safe_json_loads,
    seconds_between,
    hours_since,
)


# =========================================================================
# 1. Crystal Strength (decay + reinforcement)
# =========================================================================


def crystal_strength(
    crystal: Crystal,
    now_ts: Optional[str] = None,
    reinforcement_per_access: float = 0.03,
    max_reinforcement: float = 0.30,
) -> float:
    """Compute current effective strength of a crystal.

    strength = (base * decay_factor) + reinforcement_bonus

    base         = salience.combined() * confidence
    decay_factor = e^(-decay_rate * hours_elapsed)
    reinforcement = min(access_count * per_access, max_reinforcement)

    The time reference is last_accessed_ts (or created_ts if never accessed).
    Pass *now_ts* to get a deterministic result (useful for tests).
    """
    base = crystal.salience.combined() * crystal.confidence

    # Hours since last interaction with this crystal
    ref_ts = crystal.last_accessed_ts or crystal.created_ts
    if ref_ts:
        if now_ts:
            hours = seconds_between(ref_ts, now_ts) / 3600.0
        else:
            hours = hours_since(ref_ts)
    else:
        hours = 0.0

    decayed = exponential_decay(base, crystal.decay_rate, hours)
    reinforcement = min(crystal.access_count * reinforcement_per_access, max_reinforcement)

    return clamp(decayed + reinforcement)


def reinforce_crystal(
    storage: BaseStorage,
    crystal_id: int,
    now_ts: Optional[str] = None,
) -> Crystal:
    """Mark a crystal as accessed: bump access_count, reset last_accessed_ts.

    This resets the decay clock and adds a permanent reinforcement tick.
    """
    crystal = storage.get_crystal(crystal_id)
    if not crystal:
        raise ValueError(f"Crystal {crystal_id} not found")
    crystal.access_count += 1
    crystal.last_accessed_ts = now_ts or now_iso()
    crystal.updated_ts = crystal.last_accessed_ts
    storage.update_crystal(crystal)
    return crystal


# =========================================================================
# 2. Temporal Clustering (auto-episode formation)
# =========================================================================


def cluster_crystals(
    crystals: List[Crystal],
    window_hours: float = 4.0,
) -> List[List[Crystal]]:
    """Group time-proximate crystals into episode clusters.

    Walks through crystals sorted by created_ts and starts a new cluster
    whenever the gap between consecutive crystals exceeds *window_hours*.
    """
    if not crystals:
        return []
    valid = [c for c in crystals if c.created_ts]
    if not valid:
        return []
    sorted_c = sorted(valid, key=lambda c: c.created_ts)
    clusters: List[List[Crystal]] = [[sorted_c[0]]]
    for c in sorted_c[1:]:
        prev_ts = clusters[-1][-1].created_ts
        gap_hours = seconds_between(prev_ts, c.created_ts) / 3600.0
        if gap_hours <= window_hours:
            clusters[-1].append(c)
        else:
            clusters.append([c])
    return clusters


def create_episode_relations(
    storage: BaseStorage,
    clusters: List[List[Crystal]],
) -> int:
    """Store temporal_cluster relations for each episode.

    Creates a chain of relations between consecutive crystals in each
    cluster.  Skips if the relation already exists to allow re-running.
    Returns the number of new relations created.
    """
    created = 0
    for cluster in clusters:
        if len(cluster) < 2:
            continue
        for i in range(len(cluster) - 1):
            src = cluster[i]
            tgt = cluster[i + 1]
            if src.id is None or tgt.id is None:
                continue
            # Check for existing link to avoid duplicates
            existing = storage.relations_from("crystal", src.id)
            already = any(
                r.target_type == "crystal"
                and r.target_id == tgt.id
                and r.relation == "temporal_cluster"
                for r in existing
            )
            if not already:
                storage.insert_relation(Relation(
                    source_type="crystal",
                    source_id=src.id,
                    target_type="crystal",
                    target_id=tgt.id,
                    relation="temporal_cluster",
                    weight=0.6,
                ))
                created += 1
    return created


def get_episode_members(
    storage: BaseStorage,
    crystal_id: int,
) -> List[int]:
    """Walk temporal_cluster relations to find all crystals in the same episode.

    Follows the chain in both directions from the given crystal.
    """
    visited: set = {crystal_id}
    queue = [crystal_id]
    while queue:
        cid = queue.pop(0)
        # Forward links
        for r in storage.relations_from("crystal", cid):
            if r.relation == "temporal_cluster" and r.target_type == "crystal":
                if r.target_id not in visited:
                    visited.add(r.target_id)
                    queue.append(r.target_id)
        # Backward links
        for r in storage.relations_to("crystal", cid):
            if r.relation == "temporal_cluster" and r.source_type == "crystal":
                if r.source_id not in visited:
                    visited.add(r.source_id)
                    queue.append(r.source_id)
    return sorted(visited)


# =========================================================================
# 3. Temporal Gravity (proximity bonus)
# =========================================================================


def temporal_gravity(
    ts_a: str,
    ts_b: str,
    reference_hours: float = 8.0,
) -> float:
    """Score how temporally close two timestamps are.

    Returns 1.0 for identical timestamps, decays smoothly toward 0.
    reference_hours controls the 50 % decay point.
    """
    if not ts_a or not ts_b:
        return 0.0
    hours = seconds_between(ts_a, ts_b) / 3600.0
    return 1.0 / (1.0 + hours / reference_hours)


def temporal_proximity_bonus(
    crystal: Crystal,
    anchor_ts: str,
    reference_hours: float = 8.0,
) -> float:
    """Bonus for a crystal being close in time to an anchor timestamp."""
    ref = crystal.created_ts or crystal.updated_ts
    if not ref:
        return 0.0
    return temporal_gravity(ref, anchor_ts, reference_hours)


# =========================================================================
# 4. Recency Weighting
# =========================================================================


def recency_score(
    ts: str,
    now_ts: Optional[str] = None,
    half_life_hours: float = 48.0,
) -> float:
    """Smooth recency score.  Recent → high, old → low, never zero.

    half_life_hours is the point at which the score drops to 0.5.
    """
    if not ts:
        return 0.0
    if now_ts:
        hours = seconds_between(ts, now_ts) / 3600.0
    else:
        hours = hours_since(ts)
    return 1.0 / (1.0 + hours / half_life_hours)


# =========================================================================
# 5. Prospective Memory Engine
# =========================================================================


def create_prospective(
    storage: BaseStorage,
    trigger_description: str,
    payload_crystal_id: int,
    trigger_embedding: Optional[List[float]] = None,
    expiry_ts: Optional[str] = None,
) -> int:
    """Create a prospective memory (remembering to remember).

    Returns the prospective memory id.
    """
    pm = ProspectiveMemory(
        trigger_description=trigger_description,
        trigger_embedding=trigger_embedding,
        payload_crystal_id=payload_crystal_id,
        expiry_ts=expiry_ts,
    )
    return storage.insert_prospective(pm)


def check_prospective_triggers(
    storage: BaseStorage,
    text: str,
    embedding: Optional[List[float]] = None,
    similarity_threshold: float = 0.65,
    now_ts: Optional[str] = None,
) -> List[ProspectiveMemory]:
    """Check incoming text against all active prospective memories.

    Matching strategies (tried in order):
      1. Embedding cosine similarity ≥ threshold
      2. Trigger description substring match in text

    Expired memories are silently skipped.
    Returns the list of triggered (but not yet fired) memories.
    """
    active = storage.active_prospective_memories()
    triggered: List[ProspectiveMemory] = []
    for pm in active:
        # Skip expired
        if pm.expiry_ts:
            if now_ts:
                try:
                    from semantic_gravity_memory.utils import parse_iso
                    if parse_iso(pm.expiry_ts) < parse_iso(now_ts):
                        continue
                except (ValueError, TypeError):
                    pass
            elif is_expired(pm.expiry_ts):
                continue

        # Strategy 1: embedding similarity
        if embedding and pm.trigger_embedding:
            sim = cosine_similarity(embedding, pm.trigger_embedding)
            if sim >= similarity_threshold:
                triggered.append(pm)
                continue

        # Strategy 2: text substring
        if pm.trigger_description:
            if pm.trigger_description.lower() in text.lower():
                triggered.append(pm)
    return triggered


def fire_prospective(
    storage: BaseStorage,
    pm: ProspectiveMemory,
    now_ts: Optional[str] = None,
) -> None:
    """Mark a prospective memory as fired."""
    if pm.id is not None:
        storage.fire_prospective(pm.id, now_ts or now_iso())


def fire_all_triggered(
    storage: BaseStorage,
    triggered: List[ProspectiveMemory],
    now_ts: Optional[str] = None,
) -> List[int]:
    """Fire all triggered prospective memories.

    Returns the payload crystal ids that should be injected into the scene.
    """
    payload_ids: List[int] = []
    ts = now_ts or now_iso()
    for pm in triggered:
        fire_prospective(storage, pm, ts)
        payload_ids.append(pm.payload_crystal_id)
    return payload_ids


# =========================================================================
# 6. Memory Versioning
# =========================================================================

_HISTORY_PREFIX = "crystal_history_"


def version_crystal(
    storage: BaseStorage,
    crystal_id: int,
    updates: Dict,
    now_ts: Optional[str] = None,
) -> int:
    """Create a new version of a crystal, preserving the old state as a snapshot.

    1. Snapshot current state into meta storage
    2. Apply *updates* (dict of field_name → new_value)
    3. Increment version, update updated_ts
    4. Return the new version number
    """
    crystal = storage.get_crystal(crystal_id)
    if not crystal:
        raise ValueError(f"Crystal {crystal_id} not found")

    # Snapshot current state
    snapshot: Dict = {
        "version": crystal.version,
        "ts": crystal.updated_ts or crystal.created_ts,
        "title": crystal.title,
        "theme": crystal.theme,
        "summary": crystal.summary,
        "compressed_narrative": crystal.compressed_narrative,
        "salience": crystal.salience.to_dict(),
        "confidence": crystal.confidence,
        "self_state": crystal.self_state,
        "memory_type": crystal.memory_type,
        "contradiction_state": crystal.contradiction_state,
        "future_implications": crystal.future_implications,
        "unresolved": crystal.unresolved,
    }
    history_key = f"{_HISTORY_PREFIX}{crystal_id}"
    history = safe_json_loads(storage.get_meta(history_key, "[]"), [])
    history.append(snapshot)
    storage.set_meta(history_key, safe_json_dumps(history))

    # Apply updates
    for field, value in updates.items():
        if hasattr(crystal, field):
            setattr(crystal, field, value)
    crystal.version += 1
    crystal.updated_ts = now_ts or now_iso()
    storage.update_crystal(crystal)
    return crystal.version


def get_crystal_history(
    storage: BaseStorage,
    crystal_id: int,
) -> List[Dict]:
    """Retrieve the version history of a crystal.

    Returns a list of snapshot dicts ordered by version (oldest first).
    """
    history_key = f"{_HISTORY_PREFIX}{crystal_id}"
    history = safe_json_loads(storage.get_meta(history_key, "[]"), [])
    return sorted(history, key=lambda s: s.get("version", 0))


def belief_at_version(
    storage: BaseStorage,
    crystal_id: int,
    version: int,
) -> Optional[Dict]:
    """Get the snapshot of a crystal at a specific version.

    Returns None if that version doesn't exist in history.
    """
    for snap in get_crystal_history(storage, crystal_id):
        if snap.get("version") == version:
            return snap
    return None


# =========================================================================
# 7. Gravitational Mass (pre-computed retrieval tier)
# =========================================================================


def gravitational_mass(
    crystal: Crystal,
    now_ts: Optional[str] = None,
) -> float:
    """Compute a crystal's gravitational mass for retrieval tiering.

    Heavy crystals are always accessible (inner orbit). Light crystals
    require stronger cues to awaken. Mass is pre-computed during
    consolidation and stored, not computed at query time.

    mass = (salience + reinforcement + recency_boost) * (1 - decay_penalty)

    This mirrors base-level activation in ACT-R: frequently accessed,
    salient, recently used memories have higher resting activation.
    """
    import math

    # Salience contribution (0-1 range)
    sal = crystal.salience.combined()

    # Reinforcement from access (logarithmic — diminishing returns)
    reinforcement = math.log1p(crystal.access_count) * 0.15

    # Recency contribution
    ref_ts = crystal.last_accessed_ts or crystal.created_ts
    rec = recency_score(ref_ts, now_ts=now_ts, half_life_hours=72.0) if ref_ts else 0.0

    # Confidence weight
    conf = crystal.confidence

    # Semantic crystals get a permanence bonus (they've been graduated)
    type_bonus = 0.15 if crystal.memory_type == "semantic" else 0.0

    # Decay penalty (high decay rate + old = lighter)
    decay_penalty = crystal.decay_rate * (1.0 - rec) * 0.4

    mass = (sal * conf + reinforcement + rec * 0.3 + type_bonus) * (1.0 - decay_penalty)
    return clamp(mass)


# =========================================================================
# 8. Batch helpers (for consolidation / daemon use)
# =========================================================================


def decay_all_crystals(
    storage: BaseStorage,
    now_ts: Optional[str] = None,
    dormant_threshold: float = 0.02,
) -> Tuple[int, int]:
    """Compute strength for every crystal and mark weak ones dormant.

    Sets valid_to_ts on crystals whose strength falls below threshold.
    Returns (total_checked, newly_dormant).
    """
    all_c = storage.all_crystals()
    checked = 0
    dormant = 0
    for c in all_c:
        if c.valid_to_ts:
            continue  # already dormant
        checked += 1
        strength = crystal_strength(c, now_ts=now_ts)
        if strength < dormant_threshold:
            c.valid_to_ts = now_ts or now_iso()
            c.updated_ts = c.valid_to_ts
            storage.update_crystal(c)
            dormant += 1
    return checked, dormant


def auto_cluster(
    storage: BaseStorage,
    window_hours: float = 4.0,
) -> int:
    """Run temporal clustering on all active crystals and store relations.

    Returns the number of new relations created.
    """
    active = [c for c in storage.all_crystals() if not c.valid_to_ts]
    clusters = cluster_crystals(active, window_hours)
    return create_episode_relations(storage, clusters)
