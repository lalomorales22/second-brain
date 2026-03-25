"""
Semantic Gravity Memory — Self-State Detection

Detects the user's active self-state (role / context) from text.
Starts with seed vocabulary, then learns from entity co-occurrences.
Can discover entirely new states from recurring entity clusters.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Set

from semantic_gravity_memory.storage.base import BaseStorage
from semantic_gravity_memory.utils import safe_json_dumps, safe_json_loads


# ---------------------------------------------------------------------------
# Seed vocabulary
# ---------------------------------------------------------------------------

SEED_STATES: Dict[str, frozenset] = {
    "builder": frozenset({
        "code", "python", "api", "deploy", "database", "server", "framework",
        "debug", "test", "git", "docker", "kubernetes", "function", "class",
        "variable", "compile", "runtime", "dependency", "package", "import",
        "error", "bug", "refactor", "commit", "branch", "merge", "pull",
        "endpoint", "schema", "query", "migration", "model", "view",
        "controller", "route", "middleware", "backend", "frontend",
        "cli", "terminal", "ssh", "config", "env", "lint",
    }),
    "founder": frozenset({
        "client", "business", "contract", "invoice", "revenue", "startup",
        "pitch", "funding", "partnership", "agency", "proposal", "bid",
        "quote", "margin", "profit", "customer", "sales", "marketing",
        "roi", "kpi", "metrics", "growth", "churn", "acquisition",
        "retention", "pipeline", "deal", "negotiation", "competitor",
        "pricing", "subscription", "saas", "b2b",
    }),
    "student": frozenset({
        "class", "exam", "assignment", "homework", "study", "professor",
        "grade", "course", "semester", "university", "college", "lecture",
        "textbook", "thesis", "dissertation", "research", "lab", "campus",
        "gpa", "credit", "enrollment", "syllabus", "midterm", "final",
    }),
    "creative": frozenset({
        "design", "art", "music", "comedy", "standup", "writing", "story",
        "video", "content", "creative", "aesthetic", "visual", "animation",
        "illustration", "photography", "film", "podcast", "blog", "post",
        "brand", "logo", "color", "font", "layout", "composition",
    }),
    "family": frozenset({
        "dad", "mom", "son", "daughter", "wife", "husband", "family",
        "kids", "home", "weekend", "brother", "sister", "parent", "child",
        "birthday", "holiday", "vacation", "dinner", "school", "daycare",
        "pet", "dog", "cat", "house",
    }),
    "researcher": frozenset({
        "paper", "hypothesis", "data", "experiment", "analysis",
        "methodology", "findings", "literature", "citation", "journal",
        "peer", "review", "abstract", "conclusion", "dataset",
        "statistical", "correlation", "significance", "sample",
    }),
}

# Patterns that signal an explicit context switch
TRANSITION_PATTERNS = [
    re.compile(r"(?:ok|okay|alright|now|so)\s+(?:about|regarding|for|let me|switching to)\b", re.I),
    re.compile(r"(?:different topic|change of subject|moving on|on another note)\b", re.I),
    re.compile(r"(?:back to|returning to|let's talk about)\b", re.I),
]

META_KEY = "self_state_learning"


class SelfStateDetector:
    """Detects and learns the user's active self-state."""

    def __init__(self, storage: Optional[BaseStorage] = None):
        self.storage = storage
        self._learned: Dict[str, Dict[str, int]] = {}
        self._custom_states: Dict[str, Set[str]] = {}
        if storage:
            self._load_learning_data()

    # -----------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------

    def _load_learning_data(self) -> None:
        if not self.storage:
            return
        raw = self.storage.get_meta(META_KEY, "{}")
        data = safe_json_loads(raw, {})
        self._learned = data.get("entity_states", {})
        self._custom_states = {
            k: set(v) for k, v in data.get("custom_states", {}).items()
        }

    def _save_learning_data(self) -> None:
        if not self.storage:
            return
        data = {
            "entity_states": self._learned,
            "custom_states": {k: sorted(v) for k, v in self._custom_states.items()},
        }
        self.storage.set_meta(META_KEY, safe_json_dumps(data))

    # -----------------------------------------------------------------
    # Detection
    # -----------------------------------------------------------------

    def detect(self, text: str, entity_names: Optional[List[str]] = None) -> str:
        """Detect the active self-state from text and extracted entities.

        Priority:
          1. Seed vocabulary scoring
          2. Learned entity → state co-occurrences
          3. Custom (discovered) states
          4. Default "general"
        """
        t = text.lower()
        tokens = set(re.findall(r"[a-z0-9_\-]+", t))

        has_transition = any(p.search(text) for p in TRANSITION_PATTERNS)

        # Score each seed state
        state_scores: Dict[str, float] = {}
        for state, vocab in SEED_STATES.items():
            hits = len(tokens & vocab)
            if hits > 0:
                state_scores[state] = float(hits)

        # Score custom (discovered) states
        for state, vocab in self._custom_states.items():
            hits = len(tokens & vocab)
            if hits > 0:
                state_scores[state] = state_scores.get(state, 0.0) + float(hits)

        # Score from learned entity → state co-occurrences
        if entity_names:
            for ename in entity_names:
                low = ename.lower()
                if low in self._learned:
                    for state, count in self._learned[low].items():
                        state_scores[state] = state_scores.get(state, 0.0) + count * 0.3

        if not state_scores:
            return "general"

        best_state = max(state_scores, key=state_scores.get)  # type: ignore[arg-type]
        best_score = state_scores[best_state]

        if best_score < 1.0 and not has_transition:
            return "general"

        return best_state

    # -----------------------------------------------------------------
    # Learning
    # -----------------------------------------------------------------

    def learn(self, entity_names: List[str], detected_state: str) -> None:
        """Record that these entities co-occurred with this state."""
        if detected_state == "general" or not entity_names:
            return
        for ename in entity_names:
            low = ename.lower()
            if low not in self._learned:
                self._learned[low] = {}
            self._learned[low][detected_state] = (
                self._learned[low].get(detected_state, 0) + 1
            )
        self._save_learning_data()

    def discover_state(self, entity_names: List[str]) -> Optional[str]:
        """Create a new state if unmatched entities cluster enough.

        If 3+ entities appear together without matching any known state,
        create a new custom state named after the dominant entity.
        """
        unmatched: List[str] = []
        for ename in entity_names:
            low = ename.lower()
            in_seed = any(low in vocab for vocab in SEED_STATES.values())
            in_custom = any(low in vocab for vocab in self._custom_states.values())
            if not in_seed and not in_custom:
                unmatched.append(low)

        if len(unmatched) < 3:
            return None

        # Skip if these entities already map strongly to a known state
        for ename in unmatched:
            if ename in self._learned:
                if max(self._learned[ename].values(), default=0) >= 3:
                    return None

        new_state = f"context_{unmatched[0]}"
        self._custom_states[new_state] = set(unmatched)
        self._save_learning_data()
        return new_state
