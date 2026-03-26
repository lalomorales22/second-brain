# Second Brain

A living memory system for AI. Structured crystals, temporal gravity, spreading activation, contradiction tracking, background consolidation, and a **3D brain visualization** you can fly through.

Zero external Python dependencies. Local-first. Your data never leaves your machine.

```bash
./install.sh
second-brain
# → http://localhost:8487
```

```python
from semantic_gravity_memory import Memory

memory = Memory()
memory.ingest("I always prefer Python for prototypes")
scene = memory.recall("what tools does the user like?")
memory.consolidate()
```

## What makes this different

This is not RAG. This is not a vector store with a chat wrapper.

| Typical AI memory | Second Brain |
|---|---|
| Text chunks in a vector DB | Structured **memory crystals** with 25+ fields |
| Single relevance score | **6-dimensional salience** (emotional, practical, identity, temporal, uncertainty, novelty) |
| Overwrite old info silently | Store **contradictions** as tension, resolve later |
| Flat retrieval by cosine sim | **Gravitational retrieval** — entity gateway, gravity orbits, resonance priming, THEN spreading activation |
| No time awareness | **Temporal gravity** — decay, reinforcement, episode clustering, prospective memory |
| Static | **Living** — background consolidation merges, abstracts, decays, graduates, recomputes gravitational mass |
| No self-awareness | **Metamemory** — tracks its own retrieval accuracy per domain |
| No error correction | **Immune system** — antibodies suppress known-bad recall patterns |
| Terminal or flat dashboard | **3D brain** — Three.js WebGL, bloom, real activation waves, streaming chat |

## Install

```bash
chmod +x install.sh && ./install.sh
```

This checks Python, installs the package, sets up the `second-brain` command, checks Ollama, and pulls an embedding model if needed.

Or manually:

```bash
pip install -e .                    # or: pip install --user --break-system-packages -e .
ollama pull all-minilm              # embedding model
ollama pull gpt-oss:20b             # chat model (or any model you have)
```

No PyPI dependencies. The entire engine uses only the Python standard library.

## Quick start

### Launch the 3D brain

```bash
second-brain
```

Opens `http://localhost:8487`. Select your Ollama chat model from the dropdown, type a message, and watch the brain light up.

### Options

```bash
second-brain --port 9000 --chat-model llama3.2 --embed-model all-minilm
```

### Python API (without the UI)

```python
from semantic_gravity_memory import Memory

# No Ollama needed — works without embeddings
memory = Memory(db_path=":memory:")
memory.ingest("I'm building a Flask API with SQLite")
memory.ingest("The client deadline is next Friday")
memory.ingest("I always prefer single-file apps")

scene = memory.recall("what am I working on?")
print(scene["crystals"])
print(scene["scene_narrative"])
```

With Ollama embeddings:

```python
memory = Memory(ollama_model="all-minilm")
memory.ingest("I deployed the app to Docker with Postgres")
scene = memory.recall("what infrastructure am I using?")
```

With a chat model:

```python
answer, scene = memory.answer(
    "what should I focus on today?",
    chat_fn=lambda prompt: your_llm_call(prompt),
)
```

## How it works

```
You type something
     |
     v
Text gets crystallized into a structured memory object
(title, theme, salience, entities, confidence, decay rate...)
     |
     v
Entities extracted: "Python", "Sarah", "deadline"
     |
     v
Relations formed in the knowledge graph
     |
     v
You ask a question later
     |
     v
Phase 1: Entity Gateway — words in your query activate connected crystals via the graph
Phase 2: Gravity Orbit — heaviest memories are always on the tip of the tongue
Phase 3: Resonance Field — recent topics stay primed
     |
     v
~50 candidates scored with embeddings (not thousands)
     |
     v
Spreading activation ripples through the graph
     |
     v
Top 8 crystals become the "scene" — the AI only sees these, not your whole database
     |
     v
AI responds, response gets crystallized as a new memory
```

Every node in the 3D visualization corresponds to a real crystal or entity. Every glow reflects real gravitational mass. When nodes light up during a query, those are the actual memories being used.

## 3D Brain Visualization

A web-based Three.js interface that renders your memory as a living neural network. Everything you see is real data — no fake animations.

### What you see

- **Crystal nodes** — glowing spheres sized and brightened by **gravitational mass**. Heavy memories (frequently accessed, high salience) burn bright. Light memories are dim specks. What you see is what the memory actually weighs.
- **Entity nodes** — smaller icosahedra. Brightness = real mention count and salience.
- **Edges** — colored by relation type: blue=mentions, green=co-occurred, amber=temporal cluster, red=contradicts. Opacity = real edge weight.
- **Colors** — blue=builder, gold=founder, green=student, purple=creative, orange=family.
- **Bloom postprocessing** — UnrealBloomPass gives everything a natural glow.
- **Orbit controls** — zoom, rotate, pan, slow auto-rotate.

### What happens when you ask a question

1. **Recall** — the gravitational retrieval engine awakens relevant memories. Activated nodes light up in the graph. Inactive nodes dim. This is real — those are the actual crystals the engine selected.
2. **Activation wave** — energy ripples outward from the active cluster.
3. **Streaming response** — tokens appear as the AI generates them via Server-Sent Events.
4. **Thinking** — if the model reasons (e.g. `<think>` tags), thinking appears in a collapsible purple block above the answer.
5. **Ingestion** — the response is automatically stored as a new crystal.

### What the idle brain shows

At rest, every node sits at its true gravitational mass brightness. No fake pulsing. Heavy memories glow strong. Light ones are dim. Heavier nodes drift less — they're anchored. The resting brain IS a map of what matters most.

## Gravitational Retrieval

**Memories aren't searched. They're awakened.**

Traditional RAG loads every vector and compares against all of them. With a large database, that's slow. This system does something different — three phases before any embedding math happens:

### Phase 1: Entity Gateway

Extract concept cues from the query text. If you ask about "Python", the system finds the Python entity in the knowledge graph and follows relations to connected crystals. Zero embeddings. Pure graph traversal.

### Phase 2: Gravity Orbit

Every crystal has a pre-computed **gravitational mass** based on salience, access frequency, recency, and reinforcement. The heaviest crystals are always accessible — like tip-of-tongue memories that don't need to be searched for. Mass is recomputed every consolidation cycle.

```
mass = (salience * confidence + reinforcement + recency + type_bonus) * (1 - decay_penalty)
```

### Phase 3: Resonance Field

Semantic priming across queries. When you recall memories about Python, the entire graph neighborhood stays warm. Follow-up questions about programming, tools, or code are nearly instant because related crystals are already primed. This decays naturally between conversations.

### Then: Selective Scoring + Spreading Activation

Only after narrowing to ~40-60 candidates does the system compare embeddings. Then spreading activation propagates energy through the graph — crystal to entity to crystal — surfacing associatively related memories, not just keyword matches.

**Result:** Instead of scoring thousands of crystals, the system scores dozens. Scales to massive databases.

## Architecture

```
semantic_gravity_memory/
├── __init__.py              # Memory — one-line public API
├── models.py                # 10 dataclasses (Crystal, Event, Entity, etc.)
├── utils.py                 # Hand-rolled math, time, text helpers
├── storage/
│   ├── base.py              # Abstract storage interface (42 methods)
│   └── sqlite_backend.py    # SQLite (10 tables, WAL mode, gravitational mass)
├── embeddings/
│   ├── base.py              # Abstract embedder
│   └── ollama.py            # Ollama HTTP embedder (urllib only)
├── core/
│   ├── engine.py            # MemoryEngine orchestrator
│   ├── crystal_forge.py     # Text → Crystal pipeline
│   ├── entity_extractor.py  # 8-pass NLP entity extraction (no spaCy)
│   ├── salience.py          # 6-dimensional salience scoring
│   ├── self_state.py        # Self-state detection with learning
│   ├── contradiction.py     # Preference, factual, temporal conflict detection
│   ├── temporal.py          # Decay, clustering, gravity, gravitational mass, prospective memory
│   ├── retrieval.py         # Gravitational retrieval, resonance field, spreading activation
│   ├── consolidation.py     # Background daemon — merge, decay, graduate, recompute mass
│   ├── metamemory.py        # Per-domain confidence calibration
│   └── immune.py            # Antibody suppression of known-bad patterns
├── api/
│   └── server.py            # Stdlib HTTP server, SSE streaming, REST endpoints
├── ui/
│   ├── index.html           # SPA shell with Three.js 3D brain
│   ├── style.css            # Dark theme, glow aesthetic
│   └── brain.js             # 3D scene, force layout, streaming chat
└── gui/
    └── app.py               # Optional tkinter desktop app
```

## Core concepts

### Why "crystals"?

When you type "I had a meeting with Sarah about the API deadline being moved to Friday and I'm stressed about it," that messy sentence gets **crystallized** into a structured object:

```
title:        "API deadline moved to Friday"
theme:        "work pressure"
summary:      "Meeting with Sarah — API deadline shifted to Friday, causing stress"
self_state:   "professional"
memory_type:  "episodic"
salience:
  emotional:  0.6    (stress)
  practical:  0.8    (deadline)
  temporal:   0.7    (time-sensitive)
  identity:   0.2
  uncertainty: 0.3
  novelty:    0.4
entities:     [Sarah, API, Friday]
confidence:   0.7
decay_rate:   0.1
```

Raw text to structured, multi-dimensional, decay-aware object. Like how a mineral forms from a disordered solution into an organized lattice — the system takes unstructured language and locks it into a shape with defined facets.

And like real crystals, they have physical properties:
- **Mass** — gravitational weight in retrieval (heavy = always accessible)
- **Decay** — they weaken over time without reinforcement
- **They merge** — consolidation fuses similar crystals into one
- **They graduate** — episodic to semantic, like carbon to diamond under pressure
- **They hold tension** — contradictions are stored, not overwritten
- **They evolve** — version history tracks how beliefs change

It's not a metaphor slapped on top. The data structure actually behaves like the name implies.

### Events and entities

Every input becomes an **event** (raw record). The entity extractor runs 8 passes to pull out **entities** — tools, people, projects, concepts — and tracks them across conversations. Entities accumulate salience through reinforcement. They are the primary retrieval gateway — when you mention "Python," the system finds the Python entity and walks its graph connections to related crystals before ever comparing embeddings.

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

- **Decay** — crystals weaken over time (exponential decay with reinforcement on access)
- **Episodes** — time-proximate crystals auto-cluster into episodes
- **Gravity** — crystals close in time pull each other into recall scenes
- **Gravitational mass** — pre-computed importance determines retrieval tier and visual brightness
- **Prospective memory** — future-triggered recall ("remind me about X when Y happens")
- **Versioning** — crystals track belief evolution with snapshots

### Consolidation (the heartbeat)

A background daemon runs periodic passes:
1. **Decay** — mark weak crystals dormant
2. **Merge** — fuse crystals with >85% embedding similarity
3. **Schema extraction** — abstract recurring patterns into templates
4. **Contradiction resolution** — auto-resolve stale conflicts
5. **Graduation** — promote frequently-accessed episodic memories to semantic
6. **Carrying capacity** — enforce a crystal budget, forcing abstraction
7. **Gravitational mass** — recompute mass for all active crystals

### Metamemory

The system tracks its own accuracy. Per-domain confidence scores modulate retrieval scoring. Bad at personal questions? That domain gets weighted down.

### Immune system

Antibodies suppress crystals that previously led to bad answers. Created on user correction, checked before every scene construction.

## API reference

### Memory(db_path, ollama_model, ollama_url, max_recall, carrying_capacity)

| Method | Returns | Description |
|--------|---------|-------------|
| `ingest(text, actor, kind, context)` | `(event_id, crystal_id)` | Store a memory |
| `recall(query, self_state, now_ts)` | scene dict | Retrieve relevant memories |
| `answer(query, chat_fn, self_state)` | `(answer, scene)` | Recall + LLM + auto-ingest |
| `consolidate(now_ts)` | log dict | Run consolidation pass |
| `start_daemon(heartbeat_seconds)` | | Start background consolidation |
| `stop_daemon()` | | Stop background consolidation |
| `feedback(activation_id, quality)` | | Rate a retrieval |
| `set_prospective(trigger, crystal_id)` | pm_id | Set future-triggered recall |
| `suppress(crystal_id, reason)` | antibody_id | Block a crystal from recall |
| `stats()` | dict | Memory health metrics |
| `export()` | dict | Full data dump |

### REST API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | 3D brain UI |
| `GET` | `/api/stats` | Memory health metrics |
| `GET` | `/api/graph` | Crystals, entities, relations for 3D visualization |
| `GET` | `/api/models` | List installed Ollama models |
| `GET` | `/api/config` | Current server config |
| `GET` | `/api/export` | Full data dump |
| `POST` | `/api/answer` | Streaming chat (SSE) with memory grounding |
| `POST` | `/api/ingest` | Ingest text into memory |
| `POST` | `/api/recall` | Recall a memory scene |
| `POST` | `/api/consolidate` | Trigger consolidation |
| `POST` | `/api/feedback` | Record quality feedback |
| `POST` | `/api/config` | Update server config |

## Data storage

All memory data is stored in `~/.semantic_gravity_memory/memory.db` (SQLite). Each user starts fresh. Your data never leaves your machine.

## Requirements

- Python 3.10+
- Zero pip dependencies (stdlib only)
- Optional: [Ollama](https://ollama.com) for embeddings and chat
- Optional: tkinter for the desktop GUI
- Three.js is loaded from CDN (no npm/node needed)

## Tests

```bash
python -m pytest tests/ -q
# 326 passed
```

## Design principles

- **Memories are structured objects, not text blobs.** Crystals have 25+ fields.
- **Contradiction is data, not error.** Tension is stored, not flattened.
- **Time is a dimension, not metadata.** Decay, reinforcement, gravity, prospective triggers.
- **Retrieval is gravitational, not linear.** Graph-first, not vector-first.
- **The memory breathes.** Consolidation runs even when idle.
- **Forgetting is a feature.** Carrying capacity forces abstraction.
- **What you see is real.** Every glow, every pulse, every dim node reflects actual data.
- **The memory learns about itself.** Metamemory tracks its own reliability.
- **Simple entry, deep interior.** `Memory()` — one line, one import.
- **Your data is yours.** Local SQLite, no cloud, no telemetry.
