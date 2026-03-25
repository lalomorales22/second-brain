"""
Semantic Gravity Memory — Immune System

Antibody memories suppress known-bad recall patterns.
Created when the user corrects a mistake. Checked before
scene construction to prevent the same error from recurring.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from semantic_gravity_memory.models import AntibodyMemory
from semantic_gravity_memory.storage.base import BaseStorage
from semantic_gravity_memory.utils import (
    cosine_similarity,
    now_iso,
)


# -------------------------------------------------------------------------
# Create / deactivate
# -------------------------------------------------------------------------


def create_antibody(
    storage: BaseStorage,
    trigger_description: str,
    suppress_crystal_id: int,
    reason: str = "",
    trigger_embedding: Optional[List[float]] = None,
) -> int:
    """Create an antibody that suppresses a crystal on future matches.

    Returns the antibody id.
    """
    ab = AntibodyMemory(
        trigger_description=trigger_description,
        trigger_embedding=trigger_embedding,
        suppress_crystal_id=suppress_crystal_id,
        reason=reason,
        active=True,
    )
    return storage.insert_antibody(ab)


def deactivate_antibody(
    storage: BaseStorage,
    antibody_id: int,
) -> None:
    """Mark an antibody as inactive (soft-delete).

    We update the meta record rather than deleting so the history
    is preserved.  The storage currently stores active as INTEGER,
    so we re-insert with active=False via a direct update.
    """
    # Fetch → flip → save via raw conn if SQLiteBackend, else fallback
    ab_list = storage.active_antibodies()
    for ab in ab_list:
        if ab.id == antibody_id:
            # Use meta as a deactivation log (storage API doesn't expose
            # a direct update-antibody method, so we mark via meta).
            storage.set_meta(
                f"antibody_deactivated_{antibody_id}",
                now_iso(),
            )
            break
    # For SQLiteBackend we can reach into the conn directly if available.
    if hasattr(storage, "conn") and hasattr(storage, "_lock"):
        import threading
        with storage._lock:  # type: ignore[attr-defined]
            storage.conn.execute(  # type: ignore[attr-defined]
                "UPDATE antibodies SET active=0 WHERE id=?",
                (antibody_id,),
            )
            storage.conn.commit()  # type: ignore[attr-defined]


# -------------------------------------------------------------------------
# Check (pre-filter for scene construction)
# -------------------------------------------------------------------------


def check_antibodies(
    storage: BaseStorage,
    query_text: str,
    query_embedding: Optional[List[float]],
    candidate_crystal_ids: Set[int],
    similarity_threshold: float = 0.60,
) -> List[Dict[str, Any]]:
    """Check active antibodies against a query and candidate crystal set.

    For each antibody whose trigger matches the query AND whose target
    crystal is in *candidate_crystal_ids*, emit a suppression record.

    Returns list of ``{"antibody_id", "crystal_id", "reason"}`` dicts.
    """
    antibodies = storage.active_antibodies()
    suppressions: List[Dict[str, Any]] = []

    for ab in antibodies:
        # Does the trigger match the current query?
        triggered = False

        # Strategy 1: embedding similarity
        if query_embedding and ab.trigger_embedding:
            sim = cosine_similarity(query_embedding, ab.trigger_embedding)
            if sim >= similarity_threshold:
                triggered = True

        # Strategy 2: text substring
        if not triggered and ab.trigger_description:
            if ab.trigger_description.lower() in query_text.lower():
                triggered = True

        # Is the suppressed crystal a candidate?
        if triggered and ab.suppress_crystal_id in candidate_crystal_ids:
            suppressions.append({
                "antibody_id": ab.id,
                "crystal_id": ab.suppress_crystal_id,
                "reason": ab.reason,
            })

    return suppressions
