# Semantic Gravity Memory

A living memory system for AI. Structured crystals, temporal gravity, spreading activation, contradiction tracking, and background consolidation. Zero external Python dependencies.

```python
from semantic_gravity_memory import Memory

memory = Memory()
memory.ingest("I always prefer Python for prototypes")
scene = memory.recall("what tools does the user like?")
memory.consolidate()
```

## What makes this different

This is not RAG. This is not a vector store with a chat wrapper.

| Typical AI memory | Semantic Gravity Memory |
|---|---|
| Text chunks in a vector DB | Structured **memory crystals** with 25+ fields |
| Single relevance score | **6-dimensional salience** (emotional, practical, identity, temporal, uncertainty, novelty) |
| Overwrite old info silently | Store **contradictions** as tension, resolve later |
| Flat retrieval by cosine sim | **Spreading activation** through a knowledge graph |
| No time awareness | **Temporal gravity** — decay, reinforcement, episode clustering, prospective memory |
| Static | **Living** — background consolidation merges, abstracts, decays, graduates |
| No self-awareness | **Metamemory** — tracks its own retrieval accuracy per domain |
| No error correction | **Immune system** — antibodies suppress known-bad recall patterns |

## Install

```bash
pip install .
```

Or for development:

```bash
pip install -e .
```

No PyPI dependencies. The entire system uses only the Python standard library.

## Quick start

### Without Ollama (pure CPU, no embeddings)

```python
from semantic_gravity_memory import Memory

memory = Memory(db_path=":memory:")  # in-memory, or omit for persistent
memory.ingest("I'm building a Flask API with SQLite")
memory.ingest("The client deadline is next Friday")
memory.ingest("I always prefer single-file apps")

scene = memory.recall("what am I working on?")
print(scene["crystals"])
print(scene["scene_narrative"])
```

### With Ollama (semantic embeddings)

```bash
ollama pull all-minilm
```

```python
memory = Memory(ollama_model="all-minilm")
memory.ingest("I deployed the app to Docker with Postgres")
scene = memory.recall("what infrastructure am I using?")
```

### With a chat model

```python
answer, scene = memory.answer(
    "what should I focus on today?",
    chat_fn=lambda prompt: ollama_chat(prompt),  # your LLM call
)
```

## Architecture

```
semantic_gravity_memory/
├── __init__.py              # Memory — one-line public API
├── models.py                # 10 dataclasses (Crystal, Event, Entity, etc.)
├── utils.py                 # Hand-rolled math, time, text helpers
├── storage/
│   ├── base.py              # Abstract storage interface
│   └── sqlite_backend.py    # SQLite implementation (10 tables, WAL mode)
├── embeddings/
│   ├── base.py              # Abstract embedder
│   └── ollama.py            # Ollama HTTP embedder (urllib only)
├── core/
│   ├── engine.py            # MemoryEngine orchestrator
│   ├── crystal_forge.py     # Text → Crystal pipeline
│   ├── entity_extractor.py  # 8-pass entity extraction (no spaCy)
│   ├── salience.py          # 6-dimensional salience scoring
│   ├── self_state.py        # Self-state detection with learning
│   ├── contradiction.py     # Preference, factual, temporal conflict detection
│   ├── temporal.py          # Decay, clustering, gravity, prospective memory, versioning
│   ├── retrieval.py         # Spreading activation, scene reconstruction, working memory
│   ├── consolidation.py     # Background daemon — merge, decay, graduate, enforce capacity
│   ├── metamemory.py        # Per-domain confidence calibration
│   └── immune.py            # Antibody suppression of known-bad patterns
└── gui/
    └── app.py               # Optional tkinter desktop app
```

## Core concepts

### Memory crystals

The fundamental unit. Not a text chunk — a structured object with title, theme, summary, 6-dimensional salience, confidence, self-state, future implications, unresolved items, contradiction state, temporal anchoring, decay rate, version history, and an embedding vector.

### Events and entities

Every input becomes an **event** (raw record). The entity extractor pulls out **entities** — tools, people, projects, concepts — and tracks them across conversations. Entities accumulate salience through reinforcement.

### Salience vector

Six dimensions capturing *why* something matters:
- **Emotional** — stress, excitement, personal significance
- **Practical** — deadlines, tasks, actionable items
- **Identity** — role, self-concept, values
- **Temporal** — urgency, time-sensitivity
- **Uncertainty** — open questions, confusion
- **Novelty** — new, surprising, first encounter

### Self-state

Who the user is *right now* affects what they remember. The system detects self-states (builder, founder, student, creative, family, researcher) from text, learns entity-state associations over time, and can discover entirely new states.

### Contradiction tracking

"I like JavaScript" followed by "I hate JavaScript" doesn't silently overwrite. It creates a **contradiction** record with both claims and evidence. Contradictions can be auto-resolved by consolidation or left as tension for the LLM to acknowledge.

### Temporal engine

- **Decay**: crystals weaken over time (exponential decay with reinforcement on access)
- **Episodes**: time-proximate crystals auto-cluster into episodes
- **Gravity**: crystals close in time pull each other into recall scenes
- **Prospective memory**: future-triggered recall ("remind me about X when Y happens")
- **Versioning**: crystals track belief evolution with snapshots

### Spreading activation

Retrieval isn't just cosine similarity. Activation energy propagates through the relation graph — crystal to entity to crystal — decaying at each hop. This surfaces associatively related memories, not just keyword matches.

### Consolidation (the heartbeat)

A background daemon runs periodic passes:
1. **Decay** — mark weak crystals dormant
2. **Merge** — fuse crystals with >85% embedding similarity
3. **Schema extraction** — abstract recurring patterns into templates
4. **Contradiction resolution** — auto-resolve stale conflicts
5. **Graduation** — promote frequently-accessed episodic memories to semantic
6. **Carrying capacity** — enforce a crystal budget, forcing abstraction

### Metamemory

The system tracks its own accuracy. Per-domain confidence scores (e.g., "85% useful for code questions, 60% for personal preferences") modulate retrieval scoring.

### Immune system

Antibodies suppress crystals that previously led to bad answers. Created on user correction, checked before every scene construction.

## API reference

### Memory(db_path, ollama_model, ollama_url, max_recall, carrying_capacity)

- `ingest(text, actor, kind, context)` → `(event_id, crystal_id)`
- `recall(query, self_state, now_ts)` → scene dict
- `answer(query, chat_fn, self_state, now_ts)` → `(answer_text, scene)`
- `consolidate(now_ts)` → log dict
- `start_daemon(heartbeat_seconds)` / `stop_daemon()`
- `feedback(activation_id, quality, self_state)`
- `set_prospective(trigger, crystal_id, embedding, expiry_ts)` → pm_id
- `suppress(crystal_id, reason, trigger)` → antibody_id
- `stats()` → health metrics dict
- `export()` → full data dump
- `close()`

### Scene dict

```python
{
    "query": "...",
    "active_self_state": "builder",
    "crystals": [{"id", "title", "summary", "activation_energy", "memory_type", ...}],
    "entities": [{"id", "name", "kind", "salience", "mention_count", ...}],
    "contradictions": [{"topic", "claim_a", "claim_b", "state"}],
    "prospective_fired": [crystal_ids],
    "suppressions": [{"antibody_id", "crystal_id", "reason"}],
    "working_memory": [crystal_ids],
    "scene_narrative": "textual summary",
    "activation_id": int,
}
```

## GUI

```bash
python -m semantic_gravity_memory.gui.app
```

Or after pip install:

```bash
semantic-gravity-lab
```

Requires tkinter and a running Ollama server.

## Requirements

- Python 3.10+
- No pip dependencies (stdlib only)
- Optional: Ollama for embeddings and chat
- Optional: tkinter for GUI

## Design principles

- **Memories are structured objects, not text blobs.** Crystals have 25+ fields.
- **Contradiction is data, not error.** Tension is stored, not flattened.
- **Time is a dimension, not metadata.** Decay, reinforcement, gravity, prospective triggers.
- **The memory breathes.** Consolidation runs even when idle.
- **Forgetting is a feature.** Carrying capacity forces abstraction.
- **The memory learns about itself.** Metamemory tracks its own reliability.
- **Simple entry, deep interior.** `Memory()` — one line, one import.
