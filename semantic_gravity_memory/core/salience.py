"""
Semantic Gravity Memory — Salience Scoring

Six-dimensional salience scoring for text.
Keyword-seeded with context-sensitive boosting and embedding-based novelty.
"""

from __future__ import annotations

import re
from typing import List, Optional

from semantic_gravity_memory.models import SalienceVector
from semantic_gravity_memory.utils import clamp, cosine_similarity


# ---------------------------------------------------------------------------
# Keyword dictionaries per dimension
# ---------------------------------------------------------------------------

EMOTIONAL_KEYWORDS = frozenset({
    "love", "hate", "afraid", "stressed", "worried", "excited", "important",
    "care", "feel", "angry", "happy", "sad", "frustrated", "anxious", "proud",
    "disappointed", "grateful", "scared", "annoyed", "thrilled", "upset",
    "overwhelmed", "lonely", "hopeful", "desperate", "furious",
    "devastated", "ecstatic", "miserable", "terrified", "passionate",
    "heartbroken", "relieved", "ashamed", "embarrassed",
})

PRACTICAL_KEYWORDS = frozenset({
    "need", "must", "todo", "deadline", "ship", "build", "implement", "fix",
    "deploy", "launch", "release", "bug", "error", "issue", "task", "done",
    "finish", "complete", "start", "create", "setup", "install", "configure",
    "update", "upgrade", "migrate", "test", "debug", "refactor", "optimize",
    "solve", "resolve", "plan", "schedule", "deliver", "commit", "push",
    "merge", "review", "approve", "submit", "send", "run", "execute",
})

IDENTITY_PHRASES = frozenset({
    "i am", "i'm a", "my role", "my job", "i work", "my company", "my team",
    "as a", "i consider", "my background", "my experience", "my expertise",
    "i specialize", "my passion", "i believe", "my philosophy", "my approach",
    "i value", "my style", "i identify", "my profession", "my career",
    "i prefer", "i always", "i never", "i tend to", "my habit",
})

TEMPORAL_PHRASES = frozenset({
    "today", "tomorrow", "now", "urgent", "this week", "soon", "yesterday",
    "last week", "next month", "by friday", "deadline", "due", "schedule",
    "asap", "immediately", "tonight", "this morning", "right now",
    "next week", "this month", "by end of", "overdue", "running late",
    "time sensitive", "expiring", "before", "after", "until",
})

UNCERTAINTY_PHRASES = frozenset({
    "not sure", "unclear", "maybe", "confused", "don't know", "might",
    "possibly", "uncertain", "wondering", "how do", "what if", "should i",
    "could be", "perhaps", "debatable", "ambiguous", "either", "or maybe",
    "hard to say", "tricky", "complicated", "i think", "i guess",
    "not certain", "idk", "hmm", "tough call",
})

NOVELTY_PHRASES = frozenset({
    "first time", "never before", "new", "just discovered", "didn't know",
    "interesting", "surprising", "unusual", "unexpected", "different",
    "changed", "just learned", "til", "apparently", "turns out",
    "who knew", "i had no idea", "breakthrough", "revelation",
    "game changer", "mind blown", "never seen", "brand new",
    "novel", "innovative", "pioneering",
})


# ---------------------------------------------------------------------------
# Scoring internals
# ---------------------------------------------------------------------------


def _count_phrase_matches(text_lower: str, phrases: frozenset) -> int:
    count = 0
    for phrase in phrases:
        if phrase in text_lower:
            count += 1
    return count


def _phrase_score(count: int, base: float = 0.05, per_match: float = 0.12, cap: float = 0.85) -> float:
    return min(cap, base + count * per_match)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def score_salience(
    text: str,
    self_state: str = "general",
    recent_embeddings: Optional[List[List[float]]] = None,
    query_embedding: Optional[List[float]] = None,
) -> SalienceVector:
    """Score text across all 6 salience dimensions.

    Args:
        text:              the input text to score
        self_state:        current self-state (boosts identity if non-general)
        recent_embeddings: embeddings of recent crystals (for novelty detection)
        query_embedding:   embedding of current text (for novelty comparison)
    """
    t = text.lower()

    # Keyword / phrase matching
    emotional = _phrase_score(_count_phrase_matches(t, EMOTIONAL_KEYWORDS))
    practical = _phrase_score(_count_phrase_matches(t, PRACTICAL_KEYWORDS))
    identity = _phrase_score(_count_phrase_matches(t, IDENTITY_PHRASES))
    temporal = _phrase_score(_count_phrase_matches(t, TEMPORAL_PHRASES))
    uncertainty = _phrase_score(_count_phrase_matches(t, UNCERTAINTY_PHRASES))
    novelty = _phrase_score(_count_phrase_matches(t, NOVELTY_PHRASES))

    # --- Context boosting ---

    # Questions boost uncertainty
    question_count = text.count("?")
    if question_count > 0:
        uncertainty += 0.15 * min(question_count, 3)

    # Exclamation marks boost emotional
    exclaim_count = text.count("!")
    if exclaim_count > 0:
        emotional += 0.10 * min(exclaim_count, 3)

    # ALL-CAPS words (3+ letters) boost emotional
    caps_words = len(re.findall(r"\b[A-Z]{3,}\b", text))
    if caps_words > 0:
        emotional += 0.05 * min(caps_words, 4)

    # Non-general self-state boosts identity
    if self_state != "general":
        identity += 0.12

    # Imperative / request sentences boost practical
    if re.match(r"^(please |can you |could you |help me |i need )", t):
        practical += 0.10

    # Named-day deadlines boost temporal
    if re.search(
        r"\b(by|before|until|due)\s+"
        r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday"
        r"|tomorrow|tonight|eod|eow|eom)\b",
        t,
    ):
        temporal += 0.20

    # URLs or file paths boost practical
    if re.search(r"https?://|/[a-z]+/[a-z]+", t):
        practical += 0.08

    # --- Novelty from embedding similarity ---
    if recent_embeddings and query_embedding:
        similarities = [
            cosine_similarity(query_embedding, emb)
            for emb in recent_embeddings
            if emb
        ]
        if similarities:
            avg_sim = sum(similarities) / len(similarities)
            novelty_from_emb = clamp(1.0 - avg_sim, 0.0, 0.6)
            novelty = max(novelty, novelty_from_emb)

    return SalienceVector(
        emotional=clamp(emotional),
        practical=clamp(practical),
        identity=clamp(identity),
        temporal=clamp(temporal),
        uncertainty=clamp(uncertainty),
        novelty=clamp(novelty),
    )
