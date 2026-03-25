"""
Semantic Gravity Memory — Utilities

Hand-rolled math, time, text, and serialization helpers.
Zero external dependencies.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import re
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------


def now_iso() -> str:
    """Current local time as ISO 8601 string, seconds precision."""
    return dt.datetime.now().isoformat(timespec="seconds")


def parse_iso(s: str) -> dt.datetime:
    """Parse an ISO 8601 string back to a datetime."""
    return dt.datetime.fromisoformat(s)


def seconds_between(ts_a: str, ts_b: str) -> float:
    """Absolute seconds between two ISO timestamps."""
    a = parse_iso(ts_a)
    b = parse_iso(ts_b)
    return abs((b - a).total_seconds())


def hours_since(ts: str) -> float:
    """Hours elapsed since the given ISO timestamp."""
    then = parse_iso(ts)
    now = dt.datetime.now()
    return max(0.0, (now - then).total_seconds() / 3600.0)


def is_expired(ts: Optional[str]) -> bool:
    """True if the given ISO timestamp is in the past (or None)."""
    if not ts:
        return False
    try:
        return parse_iso(ts) < dt.datetime.now()
    except (ValueError, TypeError):
        return False


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two vectors. Pure Python, no numpy."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def exponential_decay(initial: float, decay_rate: float, time_elapsed: float) -> float:
    """Exponential decay: initial * e^(-rate * elapsed).

    Args:
        initial: starting strength (0.0 to 1.0 typically)
        decay_rate: how fast it decays (0.0 = never, 1.0 = fast)
        time_elapsed: time units elapsed (hours by convention)

    Returns:
        Current strength after decay.
    """
    if decay_rate <= 0.0:
        return initial
    return initial * math.exp(-decay_rate * time_elapsed)


def reinforcement_boost(current: float, boost: float = 0.15, ceiling: float = 1.0) -> float:
    """Apply reinforcement boost with diminishing returns near ceiling."""
    if current >= ceiling:
        return ceiling
    headroom = ceiling - current
    return current + boost * headroom


def sigmoid(x: float) -> float:
    """Standard sigmoid activation: 1 / (1 + e^(-x))."""
    if x < -500:
        return 0.0
    if x > 500:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


def clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp a value between lo and hi."""
    return max(lo, min(hi, v))


def weighted_average(values: List[float], weights: List[float]) -> float:
    """Weighted average. Returns 0 if weights sum to zero."""
    if not values or not weights or len(values) != len(weights):
        return 0.0
    total_weight = sum(weights)
    if total_weight == 0.0:
        return 0.0
    return sum(v * w for v, w in zip(values, weights)) / total_weight


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


def slugify(text: str, max_len: int = 80) -> str:
    """Convert text to a URL/filename-safe slug."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:max_len] or "memory"


def summarize_text(text: str, max_len: int = 240) -> str:
    """Truncate text to max_len, collapsing whitespace."""
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "\u2026"


def word_tokens(text: str) -> List[str]:
    """Split text into lowercase word tokens, stripping punctuation."""
    return [w.lower() for w in re.findall(r"[A-Za-z0-9_\-]+", text) if len(w) >= 2]


STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "if", "then", "so", "to", "of",
    "for", "on", "in", "at", "by", "with", "is", "it", "this", "that",
    "these", "those", "be", "been", "was", "were", "am", "are", "as",
    "from", "into", "about", "my", "your", "our", "their", "his", "her",
    "its", "me", "you", "we", "they", "he", "she", "them", "do", "did",
    "does", "have", "has", "had", "can", "could", "would", "should",
    "what", "why", "how", "when", "where", "who", "which", "will",
    "just", "like", "really", "very", "more", "less", "not", "no",
    "im", "dont", "ive", "ill", "its",
})


def content_tokens(text: str) -> List[str]:
    """Word tokens with stopwords removed."""
    return [w for w in word_tokens(text) if w not in STOPWORDS]


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def safe_json_dumps(obj: Any) -> str:
    """JSON serialize, never raises."""
    try:
        return json.dumps(obj, ensure_ascii=False)
    except (TypeError, ValueError):
        return "{}"


def safe_json_loads(s: Optional[str], fallback: Any = None) -> Any:
    """JSON deserialize, returns fallback on any failure."""
    if not s:
        return fallback
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError, ValueError):
        return fallback
