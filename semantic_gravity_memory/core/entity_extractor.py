"""
Semantic Gravity Memory — Entity Extractor

Multi-pass entity extraction from raw text.
Hand-built NLP — no spaCy, no NLTK, no external dependencies.

Passes:
  1. Capitalized phrases (proper nouns, product names)
  2. CamelCase / PascalCase / ALLCAPS identifiers
  3. Technology and tool names (pattern-matched)
  4. Quoted strings
  5. Frequency-significant content words (fallback)
  6. Relationship extraction (verb-preposition patterns)
  7. Co-occurrence pairing
  8. Deduplication and merging
"""

from __future__ import annotations

import re
from typing import Dict, List, Set, Tuple

from semantic_gravity_memory.utils import STOPWORDS, content_tokens


# ---------------------------------------------------------------------------
# Known technology / tool patterns
# ---------------------------------------------------------------------------

TECH_PATTERNS = frozenset({
    "python", "python3", "javascript", "typescript", "rust", "golang", "java",
    "ruby", "php", "swift", "kotlin", "scala", "elixir", "clojure", "perl",
    "react", "vue", "angular", "svelte", "nextjs", "nuxt", "remix", "astro",
    "flask", "django", "fastapi", "express", "rails", "laravel", "spring",
    "sqlite", "postgres", "postgresql", "mysql", "mongodb", "redis",
    "elasticsearch", "supabase", "firebase", "dynamodb", "cassandra",
    "docker", "kubernetes", "k8s", "terraform", "ansible", "nginx", "apache",
    "aws", "gcp", "azure", "vercel", "heroku", "netlify", "cloudflare",
    "git", "github", "gitlab", "bitbucket",
    "linux", "macos", "windows", "ubuntu", "debian", "fedora",
    "ollama", "openai", "anthropic", "claude", "chatgpt", "gemini",
    "tkinter", "qt", "gtk", "electron", "tauri",
    "api", "rest", "graphql", "grpc", "websocket",
    "npm", "pip", "cargo", "brew", "apt", "yarn", "pnpm",
    "json", "yaml", "toml", "csv", "xml", "protobuf",
    "html", "css", "sass", "tailwind",
    "pytest", "jest", "mocha", "unittest", "vitest",
    "webpack", "vite", "esbuild", "rollup",
    "solana", "ethereum", "bitcoin", "blockchain",
    "stripe", "paypal", "quickbooks", "plaid",
    "figma", "sketch", "canva",
    "slack", "discord", "telegram",
    "notion", "obsidian", "logseq",
})

# ---------------------------------------------------------------------------
# Relationship verb patterns
# ---------------------------------------------------------------------------

# Optional article skip for patterns
_ART = r"(?:the\s+|a\s+|an\s+|my\s+|our\s+)?"

RELATIONSHIP_PATTERNS = [
    (re.compile(rf"(?:deployed|deploying|deploy)\s+{_ART}(\S+)\s+(?:to|on)\s+{_ART}(\S+)", re.I), "deployed_to"),
    (re.compile(rf"(?:built|building|build)\s+{_ART}(\S+)\s+(?:with|using)\s+{_ART}(\S+)", re.I), "built_with"),
    (re.compile(rf"{_ART}(\S+)\s+(?:uses?|using)\s+{_ART}(\S+)", re.I), "uses"),
    (re.compile(rf"{_ART}(\S+)\s+(?:runs?|running)\s+(?:on|in)\s+{_ART}(\S+)", re.I), "runs_on"),
    (re.compile(rf"(?:migrat(?:ed|ing|e))\s+(?:from\s+)?{_ART}(\S+)\s+to\s+{_ART}(\S+)", re.I), "migrated_to"),
    (re.compile(rf"(?:connect(?:ed|ing|s)?)\s+{_ART}(\S+)\s+to\s+{_ART}(\S+)", re.I), "connects_to"),
    (re.compile(rf"(?:replaced?|replacing)\s+{_ART}(\S+)\s+with\s+{_ART}(\S+)", re.I), "replaced_by"),
    (re.compile(rf"{_ART}(\S+)\s+(?:depends?\s+on|requires?)\s+{_ART}(\S+)", re.I), "depends_on"),
]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalize_entity_name(name: str) -> str:
    """Normalize for dedup: lowercase, strip trailing version numbers."""
    n = name.lower().strip().rstrip(".")
    n = re.sub(r"(\d+\.?\d*\.?\d*)$", "", n).rstrip("-_")
    return n


def _classify_entity(name: str) -> str:
    """Determine entity kind from name patterns."""
    low = name.lower()
    norm = _normalize_entity_name(name)
    if norm in TECH_PATTERNS or low in TECH_PATTERNS:
        return "tool"
    if re.match(r"^https?://", name):
        return "url"
    if re.match(r"^\S+@\S+\.\S+$", name):
        return "email"
    if re.search(r"\.(py|js|ts|rs|go|java|php|rb|css|html|json|yaml|toml|sql|sh|md)$", low):
        return "file"
    return "concept"


# ---------------------------------------------------------------------------
# Extraction passes
# ---------------------------------------------------------------------------


def _extract_capitalized_phrases(text: str) -> List[str]:
    """Pass 1: Consecutive capitalized words (proper nouns, product names)."""
    phrases: List[str] = []
    # Multi-word: "Google Cloud Platform", "South Bay Solutions"
    for m in re.finditer(r"(?<![.\w])([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)", text):
        phrase = m.group(1).strip()
        if len(phrase) >= 4 and phrase.lower() not in STOPWORDS:
            phrases.append(phrase)
    # Single capitalized words after sentence boundaries (fixed-width lookbehind)
    for m in re.finditer(r"(?<=[.!?] )([A-Z][a-z]{2,})", text):
        w = m.group(1)
        if w.lower() not in STOPWORDS:
            phrases.append(w)
    return phrases


def _extract_camel_case(text: str) -> List[str]:
    """Pass 2: CamelCase, PascalCase, ALLCAPS identifiers."""
    found: List[str] = []
    for m in re.findall(r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b", text):
        found.append(m)
    for m in re.findall(r"\b[A-Z]{2,}(?:[-_][A-Za-z0-9]+)*\b", text):
        if m.lower() not in STOPWORDS and len(m) >= 2:
            found.append(m)
    return found


def _extract_tech_names(text: str) -> List[str]:
    """Pass 3: Known technology and tool names."""
    found: List[str] = []
    tokens = set(re.findall(r"[A-Za-z0-9_\-]+", text.lower()))
    for tok in tokens:
        norm = _normalize_entity_name(tok)
        if norm in TECH_PATTERNS and len(norm) >= 2:
            found.append(tok)
    return found


def _extract_quoted_strings(text: str) -> List[str]:
    """Pass 4: Strings inside quotes."""
    found: List[str] = []
    for m in re.findall(r'["\u201c]([^"\u201d]{2,60})["\u201d]', text):
        m = m.strip()
        if m and m.lower() not in STOPWORDS:
            found.append(m)
    for m in re.findall(r"'([^']{2,60})'", text):
        m = m.strip()
        if m and m.lower() not in STOPWORDS and not re.match(r"^[ts] ", m):
            found.append(m)
    return found


def _extract_frequent_content(text: str, threshold: int = 2) -> List[str]:
    """Pass 5: Content words appearing multiple times (significance signal)."""
    tokens = content_tokens(text)
    counts: Dict[str, int] = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    return [w for w, c in sorted(counts.items(), key=lambda x: -x[1]) if c >= threshold][:8]


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def _deduplicate_and_merge(raw_entities: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Deduplicate entities, merge variants.

    "Python", "python", "python3" → keep best form, merge.
    """
    groups: Dict[str, List[Tuple[str, str]]] = {}
    for name, kind in raw_entities:
        norm = _normalize_entity_name(name)
        if not norm or len(norm) < 2:
            continue
        if norm not in groups:
            groups[norm] = []
        groups[norm].append((name, kind))

    # Merge when one normalized name is a substring of another
    merged_keys: Set[str] = set()
    norm_list = sorted(groups.keys(), key=len)
    for i, short in enumerate(norm_list):
        if short in merged_keys:
            continue
        for long_key in norm_list[i + 1:]:
            if long_key in merged_keys:
                continue
            if short in long_key:
                groups[long_key] = groups.get(long_key, []) + groups.get(short, [])
                merged_keys.add(short)
                break

    results: List[Tuple[str, str]] = []
    for norm, variants in groups.items():
        if norm in merged_keys:
            continue
        best_name = max(
            variants,
            key=lambda x: (any(c.isupper() for c in x[0]), len(x[0])),
        )[0]
        best_kind = "concept"
        for _, k in variants:
            if k == "tool":
                best_kind = "tool"
                break
            if k not in ("concept",):
                best_kind = k
        results.append((best_name, best_kind))

    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_entities(text: str) -> List[Tuple[str, str]]:
    """Run the full multi-pass entity extraction pipeline.

    Returns list of (name, kind) tuples, deduplicated and merged.
    """
    raw: List[Tuple[str, str]] = []

    for phrase in _extract_capitalized_phrases(text):
        raw.append((phrase, _classify_entity(phrase)))

    for name in _extract_camel_case(text):
        raw.append((name, _classify_entity(name)))

    for name in _extract_tech_names(text):
        raw.append((name, "tool"))

    for name in _extract_quoted_strings(text):
        raw.append((name, _classify_entity(name)))

    # Fallback: frequency-significant words if we found very little
    if len(raw) < 3:
        for name in _extract_frequent_content(text):
            raw.append((name, _classify_entity(name)))

    return _deduplicate_and_merge(raw)


def extract_relationships(text: str) -> List[Tuple[str, str, str]]:
    """Extract (subject, predicate, object) relationship tuples."""
    results: List[Tuple[str, str, str]] = []
    for pattern, predicate in RELATIONSHIP_PATTERNS:
        for m in pattern.finditer(text):
            subj = m.group(1).strip().rstrip(".,;:")
            obj = m.group(2).strip().rstrip(".,;:")
            if (subj.lower() not in STOPWORDS
                    and obj.lower() not in STOPWORDS
                    and len(subj) >= 2 and len(obj) >= 2):
                results.append((subj, predicate, obj))
    return results


def find_co_occurrences(entities: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Generate co-occurrence pairs from entities found in the same text."""
    names = [name for name, _ in entities]
    pairs: List[Tuple[str, str]] = []
    seen: Set[Tuple[str, str]] = set()
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pair = tuple(sorted([names[i], names[j]]))
            if pair not in seen:
                seen.add(pair)
                pairs.append(pair)  # type: ignore[arg-type]
    return pairs
