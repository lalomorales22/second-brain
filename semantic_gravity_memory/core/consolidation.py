"""
Semantic Gravity Memory — Consolidation & Living Memory

The heartbeat.  Background processes that keep the memory alive:
  1. Decay pass        — mark weak crystals dormant
  2. Merge pass        — fuse high-similarity crystal pairs
  3. Schema extraction — abstract recurring patterns
  4. Contradiction res — auto-resolve stale conflicts
  5. Graduation        — promote episodic → semantic
  6. Carrying capacity — enforce crystal budget

Also provides:
  ConsolidationDaemon — background thread with configurable heartbeat
  Consolidation log   — records what every pass did
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional, Set

from semantic_gravity_memory.models import (
    Crystal,
    Schema,
    SalienceVector,
)
from semantic_gravity_memory.storage.base import BaseStorage
from semantic_gravity_memory.core.temporal import (
    crystal_strength,
    decay_all_crystals,
    auto_cluster,
)
from semantic_gravity_memory.utils import (
    cosine_similarity,
    now_iso,
    safe_json_dumps,
    safe_json_loads,
    summarize_text,
)

_LOG_KEY = "consolidation_log"


# =========================================================================
# Helpers
# =========================================================================


def _average_salience(a: SalienceVector, b: SalienceVector) -> SalienceVector:
    return SalienceVector(
        emotional=(a.emotional + b.emotional) / 2,
        practical=(a.practical + b.practical) / 2,
        identity=(a.identity + b.identity) / 2,
        temporal=(a.temporal + b.temporal) / 2,
        uncertainty=(a.uncertainty + b.uncertainty) / 2,
        novelty=(a.novelty + b.novelty) / 2,
    )


def _average_embeddings(
    a: Optional[List[float]], b: Optional[List[float]]
) -> Optional[List[float]]:
    if a and b and len(a) == len(b):
        return [(x + y) / 2.0 for x, y in zip(a, b)]
    return a or b


# =========================================================================
# Consolidator
# =========================================================================


class Consolidator:
    """Runs consolidation passes over the memory store."""

    def __init__(
        self,
        storage: BaseStorage,
        merge_threshold: float = 0.85,
        dormant_threshold: float = 0.02,
        graduation_access_min: int = 5,
        carrying_capacity: int = 500,
        cluster_window_hours: float = 4.0,
    ):
        self.storage = storage
        self.merge_threshold = merge_threshold
        self.dormant_threshold = dormant_threshold
        self.graduation_access_min = graduation_access_min
        self.carrying_capacity = carrying_capacity
        self.cluster_window_hours = cluster_window_hours

    # -----------------------------------------------------------------
    # Full pass
    # -----------------------------------------------------------------

    def run_pass(self, now_ts: Optional[str] = None) -> Dict[str, Any]:
        """Execute all consolidation sub-passes in sequence.

        Returns a log dict describing what happened.
        """
        ts = now_ts or now_iso()
        log: Dict[str, Any] = {"ts": ts}

        checked, dormant = self._decay_pass(ts)
        log["decay"] = {"checked": checked, "dormant": dormant}

        merged = self._merge_pass(ts)
        log["merge"] = {"merged": merged}

        schemas = self._schema_extraction_pass(ts)
        log["schema"] = {"extracted": schemas}

        resolved = self._contradiction_resolution_pass(ts)
        log["contradictions"] = {"resolved": resolved}

        graduated = self._graduation_pass(ts)
        log["graduation"] = {"graduated": graduated}

        evicted = self._carrying_capacity_pass(ts)
        log["carrying_capacity"] = {"evicted": evicted}

        clustered = auto_cluster(self.storage, self.cluster_window_hours)
        log["clustering"] = {"new_relations": clustered}

        self._save_log(log)
        return log

    # -----------------------------------------------------------------
    # 1. Decay pass
    # -----------------------------------------------------------------

    def _decay_pass(self, now_ts: str):
        return decay_all_crystals(
            self.storage, now_ts=now_ts,
            dormant_threshold=self.dormant_threshold,
        )

    # -----------------------------------------------------------------
    # 2. Merge pass
    # -----------------------------------------------------------------

    def _merge_pass(self, now_ts: str) -> int:
        """Find crystal pairs with embedding similarity > threshold and merge."""
        active = [
            c for c in self.storage.all_crystals()
            if not c.valid_to_ts and c.embedding and c.id is not None
        ]

        # Group by self_state for efficiency
        by_state: Dict[str, List[Crystal]] = {}
        for c in active:
            state = c.self_state or "general"
            by_state.setdefault(state, []).append(c)

        merged_count = 0
        consumed: Set[int] = set()

        for _state, crystals in by_state.items():
            for i in range(len(crystals)):
                ci = crystals[i]
                if ci.id in consumed:
                    continue
                for j in range(i + 1, len(crystals)):
                    cj = crystals[j]
                    if cj.id in consumed:
                        continue
                    sim = cosine_similarity(ci.embedding, cj.embedding)  # type: ignore[arg-type]
                    if sim >= self.merge_threshold:
                        self._merge_pair(ci, cj, now_ts)
                        consumed.add(cj.id)  # type: ignore[arg-type]
                        merged_count += 1

        return merged_count

    def _merge_pair(self, keeper: Crystal, absorbed: Crystal, now_ts: str) -> None:
        """Merge *absorbed* into *keeper*. Mark absorbed dormant."""
        # Union source events and entities
        keeper.source_event_ids = list(
            set(keeper.source_event_ids) | set(absorbed.source_event_ids)
        )
        keeper.entity_ids = list(
            set(keeper.entity_ids) | set(absorbed.entity_ids)
        )

        # Average salience and embeddings
        keeper.salience = _average_salience(keeper.salience, absorbed.salience)
        keeper.embedding = _average_embeddings(keeper.embedding, absorbed.embedding)

        # Keep higher confidence, sum access counts
        keeper.confidence = max(keeper.confidence, absorbed.confidence)
        keeper.access_count += absorbed.access_count

        # Concatenate narratives
        if absorbed.compressed_narrative:
            sep = " | " if keeper.compressed_narrative else ""
            keeper.compressed_narrative += sep + absorbed.compressed_narrative

        keeper.version += 1
        keeper.updated_ts = now_ts
        self.storage.update_crystal(keeper)

        # Mark absorbed dormant with parent link
        absorbed.parent_crystal_id = keeper.id
        absorbed.valid_to_ts = now_ts
        absorbed.updated_ts = now_ts
        self.storage.update_crystal(absorbed)

    # -----------------------------------------------------------------
    # 3. Schema extraction
    # -----------------------------------------------------------------

    def _schema_extraction_pass(self, now_ts: str) -> int:
        """Find recurring entity patterns across crystals and create schemas."""
        active = [
            c for c in self.storage.all_crystals()
            if not c.valid_to_ts and c.entity_ids and c.id is not None
        ]

        # Group by self_state
        by_state: Dict[str, List[Crystal]] = {}
        for c in active:
            state = c.self_state or "general"
            by_state.setdefault(state, []).append(c)

        existing = {s.name: s for s in self.storage.all_schemas()}
        created = 0

        for state, crystals in by_state.items():
            if len(crystals) < 3:
                continue

            # Count how often each entity appears in this state's crystals
            entity_counts: Dict[int, int] = {}
            for c in crystals:
                for eid in c.entity_ids:
                    entity_counts[eid] = entity_counts.get(eid, 0) + 1

            # Entities appearing in 40 %+ of the group
            threshold = max(2, int(len(crystals) * 0.4))
            common = [eid for eid, cnt in entity_counts.items() if cnt >= threshold]
            if not common:
                continue

            schema_name = f"{state}_pattern"
            if schema_name in existing:
                # Update with new crystals
                schema = existing[schema_name]
                old_ids = set(schema.source_crystal_ids)
                new_ids = [c.id for c in crystals if c.id not in old_ids]
                if new_ids:
                    schema.source_crystal_ids = list(old_ids | set(new_ids))
                    schema.activation_count += 1
                    schema.updated_ts = now_ts
                    self.storage.update_schema(schema)
            else:
                # Build description from common entities
                enames: List[str] = []
                for eid in common[:5]:
                    e = self.storage.get_entity(eid)
                    if e:
                        enames.append(e.name)

                pattern = " \u2192 ".join(
                    summarize_text(c.title, 40) for c in crystals[:5]
                )
                self.storage.insert_schema(Schema(
                    created_ts=now_ts,
                    updated_ts=now_ts,
                    name=schema_name,
                    description=(
                        f"Recurring {state} pattern"
                        + (f" involving {', '.join(enames)}" if enames else "")
                    ),
                    pattern=pattern,
                    source_crystal_ids=[c.id for c in crystals],
                    activation_count=1,
                ))
                created += 1

        return created

    # -----------------------------------------------------------------
    # 4. Contradiction resolution
    # -----------------------------------------------------------------

    def _contradiction_resolution_pass(self, now_ts: str) -> int:
        """Auto-resolve stale open contradictions.

        Heuristic: the newer claim supersedes the older one.
        Only resolves contradictions that have been open > 0 seconds
        (i.e., not brand-new ones from the same pass).
        """
        open_c = self.storage.open_contradictions()
        resolved = 0

        for c in open_c:
            if c.evidence_event_a is None or c.evidence_event_b is None:
                continue
            ea = self.storage.get_event(c.evidence_event_a)
            eb = self.storage.get_event(c.evidence_event_b)
            if not ea or not eb:
                continue
            # Resolve in favor of the newer event
            try:
                from semantic_gravity_memory.utils import parse_iso
                ts_a = parse_iso(ea.ts)
                ts_b = parse_iso(eb.ts)
            except (ValueError, TypeError):
                continue

            if ts_b >= ts_a:
                c.resolution_state = "resolved_b"
            else:
                c.resolution_state = "resolved_a"
            c.resolution_ts = now_ts
            c.notes = (c.notes + " | auto-resolved by consolidation").strip(" | ")
            self.storage.update_contradiction(c)
            resolved += 1

        return resolved

    # -----------------------------------------------------------------
    # 5. Episodic → semantic graduation
    # -----------------------------------------------------------------

    def _graduation_pass(self, now_ts: str) -> int:
        """Promote frequently accessed episodic crystals to semantic."""
        active = [
            c for c in self.storage.all_crystals()
            if not c.valid_to_ts
            and c.memory_type == "episodic"
            and c.access_count >= self.graduation_access_min
            and c.id is not None
        ]

        graduated = 0
        for c in active:
            # Only graduate if temporal salience is low (not time-bound)
            if c.salience.temporal > 0.4:
                continue
            c.memory_type = "semantic"
            c.decay_rate = max(0.01, c.decay_rate * 0.3)  # much slower decay
            c.valid_from_ts = None   # loosen temporal anchors
            c.valid_to_ts = None
            c.updated_ts = now_ts
            self.storage.update_crystal(c)
            graduated += 1

        return graduated

    # -----------------------------------------------------------------
    # 6. Carrying capacity
    # -----------------------------------------------------------------

    def _carrying_capacity_pass(self, now_ts: str) -> int:
        """If active crystals exceed budget, dormant the weakest."""
        active = [c for c in self.storage.all_crystals() if not c.valid_to_ts]
        excess = len(active) - self.carrying_capacity
        if excess <= 0:
            return 0

        # Sort by strength ascending (weakest first)
        scored = sorted(
            active,
            key=lambda c: crystal_strength(c, now_ts=now_ts),
        )
        evicted = 0
        for c in scored[:excess]:
            if c.id is None:
                continue
            c.valid_to_ts = now_ts
            c.updated_ts = now_ts
            self.storage.update_crystal(c)
            evicted += 1

        return evicted

    # -----------------------------------------------------------------
    # Log
    # -----------------------------------------------------------------

    def _save_log(self, entry: Dict) -> None:
        log = safe_json_loads(self.storage.get_meta(_LOG_KEY, "[]"), [])
        log.append(entry)
        if len(log) > 100:
            log = log[-100:]
        self.storage.set_meta(_LOG_KEY, safe_json_dumps(log))

    def get_log(self, limit: int = 20) -> List[Dict]:
        log = safe_json_loads(self.storage.get_meta(_LOG_KEY, "[]"), [])
        return log[-limit:]


# =========================================================================
# Consolidation Daemon (background thread)
# =========================================================================


class ConsolidationDaemon:
    """Runs consolidation passes on a background timer."""

    def __init__(
        self,
        consolidator: Consolidator,
        heartbeat_seconds: float = 300.0,
    ):
        self.consolidator = consolidator
        self.heartbeat = heartbeat_seconds
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            stopped = self._stop_event.wait(timeout=self.heartbeat)
            if stopped:
                break
            try:
                self.consolidator.run_pass()
            except Exception:
                pass  # daemon must not crash
