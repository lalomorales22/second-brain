"""
Semantic Gravity Memory — MetaMemory

Memory about memory. Tracks retrieval quality per domain,
calibrates confidence, and identifies under-used or over-used crystals.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from semantic_gravity_memory.storage.base import BaseStorage
from semantic_gravity_memory.utils import (
    now_iso,
    safe_json_dumps,
    safe_json_loads,
)

_DOMAIN_KEY = "metamemory_domain_stats"
_FEEDBACK_KEY = "metamemory_feedback_log"


class MetaMemory:
    """Tracks and calibrates retrieval quality."""

    def __init__(self, storage: BaseStorage):
        self.storage = storage

    # -----------------------------------------------------------------
    # Feedback recording
    # -----------------------------------------------------------------

    def record_feedback(
        self,
        activation_id: int,
        quality: float,
        self_state: str = "general",
        now_ts: Optional[str] = None,
    ) -> None:
        """Record quality feedback for a retrieval activation.

        quality: 0.0 = terrible, 1.0 = perfect.
        Also updates the running per-domain stats.
        """
        # Append to feedback log
        log = safe_json_loads(self.storage.get_meta(_FEEDBACK_KEY, "[]"), [])
        log.append({
            "activation_id": activation_id,
            "quality": quality,
            "self_state": self_state,
            "ts": now_ts or now_iso(),
        })
        # Keep last 500 entries to avoid unbounded growth
        if len(log) > 500:
            log = log[-500:]
        self.storage.set_meta(_FEEDBACK_KEY, safe_json_dumps(log))

        # Update domain stats
        stats = safe_json_loads(self.storage.get_meta(_DOMAIN_KEY, "{}"), {})
        if self_state not in stats:
            stats[self_state] = {"count": 0, "total_quality": 0.0}
        stats[self_state]["count"] += 1
        stats[self_state]["total_quality"] += quality
        self.storage.set_meta(_DOMAIN_KEY, safe_json_dumps(stats))

    # -----------------------------------------------------------------
    # Domain confidence
    # -----------------------------------------------------------------

    def domain_confidence(self, self_state: str) -> float:
        """Average quality score for a self-state domain.

        Returns 0.5 (neutral) if there's no data yet.
        """
        stats = safe_json_loads(self.storage.get_meta(_DOMAIN_KEY, "{}"), {})
        entry = stats.get(self_state)
        if not entry or entry.get("count", 0) == 0:
            return 0.5
        return entry["total_quality"] / entry["count"]

    def all_domain_confidences(self) -> Dict[str, float]:
        """Confidence scores for all tracked domains."""
        stats = safe_json_loads(self.storage.get_meta(_DOMAIN_KEY, "{}"), {})
        result: Dict[str, float] = {}
        for state, entry in stats.items():
            if entry.get("count", 0) > 0:
                result[state] = entry["total_quality"] / entry["count"]
            else:
                result[state] = 0.5
        return result

    # -----------------------------------------------------------------
    # Retrieval history analysis
    # -----------------------------------------------------------------

    def crystal_recall_counts(self) -> Dict[int, int]:
        """Count how many times each crystal has appeared in activations.

        Scans recent activations to build a frequency map.
        """
        activations = self.storage.recent_activations(limit=200)
        counts: Dict[int, int] = {}
        for act in activations:
            for cid in act.crystal_ids:
                counts[cid] = counts.get(cid, 0) + 1
        return counts

    def never_recalled_crystals(self) -> List[int]:
        """Crystal ids that have never appeared in any activation."""
        recalled = set(self.crystal_recall_counts().keys())
        all_crystals = self.storage.all_crystals()
        return [c.id for c in all_crystals if c.id is not None and c.id not in recalled]

    def most_recalled_crystals(self, limit: int = 10) -> List[int]:
        """Crystal ids sorted by recall frequency, descending."""
        counts = self.crystal_recall_counts()
        return sorted(counts, key=counts.get, reverse=True)[:limit]  # type: ignore[arg-type]

    def feedback_log(self, limit: int = 50) -> List[Dict]:
        """Recent feedback entries."""
        log = safe_json_loads(self.storage.get_meta(_FEEDBACK_KEY, "[]"), [])
        return log[-limit:]
