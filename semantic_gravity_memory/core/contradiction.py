"""
Semantic Gravity Memory — Contradiction Detection

Detects three types of contradiction:
  1. Preference — "I like X" vs "I hate X"
  2. Factual   — "X uses REST" vs "X uses GraphQL"
  3. Temporal   — "I'll finish X by Friday" + evidence of non-completion

Also handles belief evolution (not every change is a contradiction)
and resolution suggestions based on evidence weight and recency.
"""

from __future__ import annotations

import re
from typing import List, Optional, Set, Tuple

from semantic_gravity_memory.models import Contradiction, Event
from semantic_gravity_memory.storage.base import BaseStorage
from semantic_gravity_memory.utils import now_iso, parse_iso, summarize_text


# ---------------------------------------------------------------------------
# Preference patterns
# ---------------------------------------------------------------------------

_PREF_POSITIVE = [
    re.compile(
        r"\bi\s+(?:like|love|prefer|enjoy|appreciate|use|choose|recommend)\s+"
        r"(.{3,60}?)(?:[.,;!?\n]|$)",
        re.I,
    ),
]

_PREF_NEGATIVE = [
    re.compile(
        r"\bi\s+(?:hate|dislike|avoid|dropped|stopped using"
        r"|don't like|do not like|can't stand)\s+"
        r"(.{3,60}?)(?:[.,;!?\n]|$)",
        re.I,
    ),
]

# ---------------------------------------------------------------------------
# Factual claim patterns
# ---------------------------------------------------------------------------

_FACTUAL_PATTERNS = [
    re.compile(r"(\S+)\s+(?:is|uses?|runs?|works?\s+with)\s+(\S+)", re.I),
    re.compile(r"(\S+)\s+costs?\s+\$?([\d,.]+)", re.I),
]

# ---------------------------------------------------------------------------
# Temporal commitment patterns
# ---------------------------------------------------------------------------

_COMMITMENT_PATTERNS = [
    re.compile(
        r"i'?ll?\s+(?:have|get|finish|complete|do|deliver|ship)\s+"
        r"(.{3,60}?)\s+by\s+(.{3,30}?)(?:[.,;!?\n]|$)",
        re.I,
    ),
    re.compile(
        r"(.{3,40}?)\s+(?:will be|should be|is going to be)\s+"
        r"(?:done|ready|finished|complete)\s+by\s+"
        r"(.{3,30}?)(?:[.,;!?\n]|$)",
        re.I,
    ),
]

# ---------------------------------------------------------------------------
# Growth / evolution patterns (suppress false positives)
# ---------------------------------------------------------------------------

_GROWTH_PATTERNS = [
    re.compile(r"i'm\s+(?:learning|getting better at|improving|now able to|starting to)", re.I),
    re.compile(r"i\s+(?:used to|didn't used to|initially|at first|before)", re.I),
    re.compile(r"(?:now i|these days i|i've since|i've grown)", re.I),
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_preferences(text: str) -> List[Tuple[str, str]]:
    """Extract (subject, sentiment) pairs. Sentiment is 'positive' or 'negative'."""
    results: List[Tuple[str, str]] = []
    for pat in _PREF_POSITIVE:
        for m in pat.finditer(text):
            subj = m.group(1).strip().lower().rstrip(".,;!?")
            if subj and len(subj) >= 2:
                results.append((subj, "positive"))
    for pat in _PREF_NEGATIVE:
        for m in pat.finditer(text):
            subj = m.group(1).strip().lower().rstrip(".,;!?")
            if subj and len(subj) >= 2:
                results.append((subj, "negative"))
    return results


def _extract_factual_claims(text: str) -> List[Tuple[str, str]]:
    """Extract (subject, object) pairs from factual statements."""
    results: List[Tuple[str, str]] = []
    for pat in _FACTUAL_PATTERNS:
        for m in pat.finditer(text):
            groups = m.groups()
            if len(groups) >= 2:
                subj = groups[0].strip().lower().rstrip(".,;:")
                obj = groups[1].strip().lower().rstrip(".,;:")
                if subj and obj and len(subj) >= 2 and len(obj) >= 1:
                    results.append((subj, obj))
    return results


def _is_growth(text: str) -> bool:
    """True if the text signals belief evolution, not contradiction."""
    return any(p.search(text) for p in _GROWTH_PATTERNS)


def _subjects_overlap(a: str, b: str) -> bool:
    """Check if two subject strings refer to the same thing."""
    a, b = a.lower().strip(), b.lower().strip()
    if a == b:
        return True
    if a in b or b in a:
        return True
    tokens_a = set(a.split())
    tokens_b = set(b.split())
    if not tokens_a or not tokens_b:
        return False
    overlap = len(tokens_a & tokens_b)
    return overlap >= min(len(tokens_a), len(tokens_b)) * 0.5


# ---------------------------------------------------------------------------
# Detector class
# ---------------------------------------------------------------------------


class ContradictionDetector:
    """Detects contradictions across memory history."""

    def __init__(self, storage: BaseStorage):
        self.storage = storage

    # -----------------------------------------------------------------
    # Per-type checks
    # -----------------------------------------------------------------

    def check_preferences(self, text: str, event_id: int) -> List[Contradiction]:
        """Preference contradictions: "I like X" vs "I hate X"."""
        if _is_growth(text):
            return []

        current_prefs = _extract_preferences(text)
        if not current_prefs:
            return []

        contradictions: List[Contradiction] = []
        recent = self.storage.recent_events(limit=200)

        for old in recent:
            if old.id == event_id:
                continue
            if _is_growth(old.content):
                continue
            old_prefs = _extract_preferences(old.content)
            for cur_subj, cur_sent in current_prefs:
                for old_subj, old_sent in old_prefs:
                    if not _subjects_overlap(cur_subj, old_subj):
                        continue
                    if cur_sent != old_sent:
                        contradictions.append(Contradiction(
                            ts=now_iso(),
                            topic=cur_subj,
                            claim_a=summarize_text(old.content, 200),
                            claim_b=summarize_text(text, 200),
                            evidence_event_a=old.id,
                            evidence_event_b=event_id,
                            resolution_state="open",
                            notes=f"preference: was {old_sent}, now {cur_sent}",
                        ))
        return contradictions

    def check_factual(self, text: str, event_id: int) -> List[Contradiction]:
        """Factual contradictions: same subject, different object."""
        current_claims = _extract_factual_claims(text)
        if not current_claims:
            return []

        contradictions: List[Contradiction] = []
        recent = self.storage.recent_events(limit=200)

        for old in recent:
            if old.id == event_id:
                continue
            old_claims = _extract_factual_claims(old.content)
            for cur_subj, cur_obj in current_claims:
                for old_subj, old_obj in old_claims:
                    if _subjects_overlap(cur_subj, old_subj) and cur_obj != old_obj:
                        contradictions.append(Contradiction(
                            ts=now_iso(),
                            topic=cur_subj,
                            claim_a=f"{old_subj} \u2192 {old_obj}",
                            claim_b=f"{cur_subj} \u2192 {cur_obj}",
                            evidence_event_a=old.id,
                            evidence_event_b=event_id,
                            resolution_state="open",
                            notes="factual: same subject, different claims",
                        ))
        return contradictions

    def check_temporal(self, text: str, event_id: int) -> List[Contradiction]:
        """Temporal contradictions: missed commitments."""
        contradictions: List[Contradiction] = []
        recent = self.storage.recent_events(limit=200)

        for old in recent:
            if old.id == event_id:
                continue
            for pat in _COMMITMENT_PATTERNS:
                for m in pat.finditer(old.content):
                    task_desc = m.group(1).strip()
                    date_desc = m.group(2).strip()
                    if task_desc.lower() in text.lower():
                        incompletion = re.search(
                            r"(?:haven't|hasn't|didn't|not done|not finished"
                            r"|still|yet|behind on|delayed)",
                            text, re.I,
                        )
                        if incompletion:
                            contradictions.append(Contradiction(
                                ts=now_iso(),
                                topic=task_desc,
                                claim_a=f"committed: {summarize_text(old.content, 160)}",
                                claim_b=f"status: {summarize_text(text, 160)}",
                                evidence_event_a=old.id,
                                evidence_event_b=event_id,
                                resolution_state="open",
                                notes=f"temporal: commitment by '{date_desc}' appears unmet",
                            ))
        return contradictions

    # -----------------------------------------------------------------
    # Combined check
    # -----------------------------------------------------------------

    def check_all(self, text: str, event_id: int) -> List[Contradiction]:
        """Run all detection passes, deduplicate by (topic, event pair)."""
        all_c: List[Contradiction] = []
        all_c.extend(self.check_preferences(text, event_id))
        all_c.extend(self.check_factual(text, event_id))
        all_c.extend(self.check_temporal(text, event_id))

        seen: Set[tuple] = set()
        deduped: List[Contradiction] = []
        for c in all_c:
            key = (c.topic, c.evidence_event_a, c.evidence_event_b)
            if key not in seen:
                seen.add(key)
                deduped.append(c)
        return deduped

    # -----------------------------------------------------------------
    # Resolution
    # -----------------------------------------------------------------

    def suggest_resolution(self, contradiction: Contradiction) -> Optional[str]:
        """Suggest a resolution based on recency."""
        if contradiction.evidence_event_a is None or contradiction.evidence_event_b is None:
            return None
        event_a = self.storage.get_event(contradiction.evidence_event_a)
        event_b = self.storage.get_event(contradiction.evidence_event_b)
        if not event_a or not event_b:
            return None
        try:
            ts_a = parse_iso(event_a.ts)
            ts_b = parse_iso(event_b.ts)
            if ts_b > ts_a:
                return f"resolved_b: newer claim (event {event_b.id}) likely supersedes"
            return f"resolved_a: newer claim (event {event_a.id}) likely supersedes"
        except (ValueError, TypeError):
            return None
