# Second Brain
<img width="1263" height="830" alt="Screenshot 2026-03-25 at 11 54 14 PM" src="https://github.com/user-attachments/assets/5d4b5cb1-bbfe-49f0-b159-889e8bb2bb90" />

A persistent memory engine for AI agents with a **3D brain visualization** you can explore. Memory records with multi-axis importance scoring, multi-phase retrieval (graph-first, then embeddings), spreading activation, contradiction tracking, temporal decay, and background consolidation.

Zero external Python dependencies. Local-first. Everything in one SQLite file. Your data never leaves your machine.

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
| Text chunks in a vector DB | Structured memory records (internally called "crystals") with 25+ fields |
| Single relevance score | Importance scored across 6 axes (emotional, practical, identity, temporal, uncertainty, novelty) via keyword matching with context boosters |
| Overwrite old info silently | Store **contradictions** as tension — both claims preserved, resolved later |
| Flat retrieval by cosine sim | **Multi-phase retrieval** — entity gateway, importance tier, recency priming, THEN embedding comparison on ~50 candidates |
| No time awareness | **Temporal engine** — exponential decay, reinforcement on access, episode clustering, prospective memory |
| Static | **Background consolidation** — daemon merges, decays, graduates, enforces capacity, recomputes importance |
| No self-awareness | **Metamemory** — tracks its own retrieval accuracy per domain |
| No error correction | **Suppression rules** — context-aware filters block known-bad recall patterns |
| Terminal or flat dashboard | **3D brain** — Three.js WebGL, bloom, real activation waves driven by actual retrieval data |

## Install

### One-liner (from anywhere)

```bash
pip install git+https://github.com/lalomorales22/second-brain.git
```

Then use it in your own code:

```python
from semantic_gravity_memory import Memory
memory = Memory()
```

### Full setup (with 3D brain + terminal command)

```bash
git clone https://github.com/lalomorales22/second-brain.git
cd second-brain
chmod +x install.sh && ./install.sh
```

This checks Python, installs the package, sets up the `second-brain` command, checks Ollama, and pulls an embedding model if needed.

### Manual

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

### Drop it into any app

Second Brain is a library, not just a UI. Three lines to give any chatbot, agent, or app persistent memory:

```bash
pip install git+https://github.com/lalomorales22/second-brain.git
```

```python
from semantic_gravity_memory import Memory

# In your chatbot / agent / app
memory = Memory(ollama_model="all-minilm")

# Every conversation turn
memory.ingest(user_message)
scene = memory.recall(user_message)  # top 8 relevant memories
# Feed scene["crystals"] into your LLM prompt as context

# Run periodically — the memory tidies itself
memory.consolidate()
```

Works with any LLM, any framework, any language that can call Python. The 3D brain is optional — the engine stands alone.

## How it works

```
You type something
     |
     v
Text gets structured into a memory record
(title, theme, importance scores, entities, confidence, decay rate...)
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
Phase 1: Entity Gateway — words in your query find connected memories via the graph
Phase 2: Importance Tier — highest-importance memories are always accessible (like tip-of-tongue)
Phase 3: Recency Cache — recent topics and their neighborhoods stay primed
     |
     v
~50 candidates scored with embeddings (not thousands)
     |
     v
Spreading activation ripples through the graph (BFS with energy decay per hop)
     |
     v
Top 8 memories become the "scene" — the AI only sees these, not your whole database
     |
     v
AI responds, response gets stored as a new memory
```

Every node in the 3D visualization corresponds to a real memory or entity. Every glow reflects the real importance score. When nodes light up during a query, those are the actual memories being retrieved.

## 3D Brain Visualization

A web-based Three.js interface that renders your memory as an explorable graph. Everything you see is real data — no fake animations.

### What you see

- **Memory nodes** — glowing spheres sized and brightened by importance score. Frequently accessed, high-salience memories burn bright. Weak ones are dim specks.
- **Entity nodes** — smaller icosahedra. Brightness = mention count and salience.
- **Edges** — colored by relation type: blue=mentions, green=co-occurred, amber=temporal cluster, red=contradicts. Opacity = edge weight.
- **Colors** — blue=builder, gold=founder, green=student, purple=creative, orange=family.
- **Bloom postprocessing** — UnrealBloomPass gives everything a natural glow.
- **Orbit controls** — zoom, rotate, pan, slow auto-rotate.

### What happens when you ask a question

1. **Recall** — the retrieval engine surfaces relevant memories. Activated nodes light up. Inactive nodes dim. These are the actual memories the engine selected.
2. **Activation wave** — energy ripples outward from the active cluster.
3. **Streaming response** — tokens appear as the AI generates them via Server-Sent Events.
4. **Thinking** — if the model reasons (e.g. `<think>` tags), thinking appears in a collapsible purple block above the answer.
5. **Ingestion** — the response is automatically stored as a new memory.

### What the idle brain shows

At rest, every node sits at its true importance-score brightness. No fake pulsing. Heavy memories glow strong. Light ones are dim. The resting brain is a map of what matters most in the memory.

## How retrieval actually works

Traditional RAG compares every embedding on every query. When you have hundreds of memories and most aren't relevant, that's wasteful. This system narrows candidates before touching embeddings:

### Phase 1: Entity Gateway

Extract entity names from the query text. If you ask about "Python", the system finds the Python entity in the knowledge graph and follows relations to connected memories. Zero embeddings. Pure graph traversal.

### Phase 2: Importance Tier

Called "gravity orbit" in the code. Every memory has a pre-computed importance score based on salience, access frequency (logarithmic), recency, confidence, and memory type. The top 30 are always accessible — like frequently-used items at the top of your mind. Score is recomputed every consolidation cycle.

```
importance = (salience * confidence + reinforcement + recency + type_bonus) * (1 - decay_penalty)
```

### Phase 3: Recency Cache

Called "resonance field" in the code. This is semantic priming: when you recall memories about Python, the entire graph neighborhood stays warm. Follow-up questions about programming, tools, or code benefit because related memories are already in the candidate set. Decays naturally between conversations.

### Then: Selective Scoring + Spreading Activation

Only after narrowing to ~40-60 candidates does the system compare embeddings. Then spreading activation propagates energy through the graph — memory to entity to memory — surfacing associatively related results. Spreading activation is a real cognitive science concept (Collins & Loftus, 1975), applied here as BFS with energy decay per hop.

**Result:** Instead of scoring every memory, the system scores dozens. For the personal/agent memory scale this targets (hundreds to low thousands of items), retrieval is fast and context-aware.

## Architecture

```
semantic_gravity_memory/
├── __init__.py              # Memory — one-line public API
├── models.py                # 10 dataclasses (Crystal, Event, Entity, etc.)
├── utils.py                 # Hand-rolled math, time, text helpers
├── storage/
│   ├── base.py              # Abstract storage interface (42 methods)
│   └── sqlite_backend.py    # SQLite (10 tables, WAL mode)
├── embeddings/
│   ├── base.py              # Abstract embedder
│   └── ollama.py            # Ollama HTTP embedder (urllib only)
├── core/
│   ├── engine.py            # MemoryEngine orchestrator
│   ├── crystal_forge.py     # Text → memory record pipeline
│   ├── entity_extractor.py  # Multi-pass regex entity extraction (no spaCy)
│   ├── salience.py          # 6-axis importance scoring (keyword matching + context boosters)
│   ├── self_state.py        # Self-state detection with learning
│   ├── contradiction.py     # Preference, factual, temporal conflict detection
│   ├── temporal.py          # Decay, clustering, importance scoring, prospective memory
│   ├── retrieval.py         # Multi-phase retrieval, spreading activation
│   ├── consolidation.py     # Background daemon — merge, decay, graduate, enforce capacity
│   ├── metamemory.py        # Per-domain confidence calibration
│   └── immune.py            # Context-aware suppression of known-bad patterns
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

### Memory records ("crystals")

When you type "I had a meeting with Sarah about the API deadline being moved to Friday and I'm stressed about it," that sentence gets structured into a record with 25+ fields:

```
title:        "API deadline moved to Friday"
theme:        "work pressure"
summary:      "Meeting with Sarah — API deadline shifted to Friday, causing stress"
self_state:   "professional"
memory_type:  "episodic"
salience:
  emotional:  0.6    (stress keywords detected)
  practical:  0.8    (deadline keywords detected)
  temporal:   0.7    (time-sensitive keywords detected)
  identity:   0.2
  uncertainty: 0.3
  novelty:    0.4
entities:     [Sarah, API, Friday]
confidence:   0.7
decay_rate:   0.1
```

The name "crystal" comes from the process — raw unstructured text compressed into a structured, multi-dimensional object. Like a mineral forming from a disordered solution into an organized lattice.

These records have lifecycle properties:
- **Importance score** — determines retrieval tier and visual brightness
- **Decay** — weakens over time without access (Ebbinghaus-style exponential decay)
- **Merge** — consolidation fuses similar records into one
- **Graduate** — episodic to semantic after repeated access
- **Contradiction** — conflicting claims stored as tension, not overwritten
- **Version history** — tracks how beliefs change over time

### Events and entities

Every input becomes an **event** (raw record). The entity extractor runs multiple regex passes to pull out **entities** — tools, people, projects, concepts — and tracks them across conversations. Entities accumulate salience through reinforcement. They are the primary retrieval gateway — when you mention "Python," the system finds the Python entity and walks its graph connections before ever comparing embeddings.

The entity extractor uses: capitalized phrase matching, CamelCase/ALLCAPS detection, a dictionary of ~100 known tech names, quoted string extraction, and frequency-significant fallback. It won't beat spaCy on accuracy, but it works with zero dependencies.

### Importance scoring (6 axes)

Six dimensions capturing *why* something matters:
- **Emotional** — stress, excitement, personal significance
- **Practical** — deadlines, tasks, actionable items
- **Identity** — role, self-concept, values
- **Temporal** — urgency, time-sensitivity
- **Uncertainty** — open questions, confusion
- **Novelty** — new, surprising, first encounter

Scored via keyword matching against curated word lists with context boosters (question marks boost uncertainty, exclamation marks boost emotional, etc.). It's simple pattern matching — not deep NLP — but the axes capture useful signal about the *type* of importance, not just the amount.

### Self-state

Who the user is *right now* affects what they remember. The system detects self-states (builder, founder, student, creative, family, researcher) from text using keyword matching, learns entity-state associations over time, and can discover new states from recurring patterns.

### Contradiction tracking

"I like JavaScript" followed by "I hate JavaScript" doesn't silently overwrite. It creates a **contradiction** record with both claims and evidence. Includes growth detection — "I used to like X, now I prefer Y" is recognized as evolution, not contradiction. Contradictions can be auto-resolved during consolidation (newer claim wins) or left as tension for the LLM to acknowledge.

### Temporal engine

- **Decay** — memories weaken over time (exponential decay, reinforced on access)
- **Episodes** — time-proximate memories auto-cluster
- **Importance score** — pre-computed, determines retrieval tier and visual brightness
- **Prospective memory** — future-triggered recall ("when deployment comes up, surface this memory")
- **Versioning** — tracks belief evolution with snapshots

### Consolidation (the heartbeat)

A background daemon thread runs periodic passes:
1. **Decay** — mark weak memories dormant
2. **Merge** — fuse memories with >85% embedding similarity
3. **Schema extraction** — abstract recurring patterns into templates
4. **Contradiction resolution** — auto-resolve stale conflicts
5. **Graduation** — promote frequently-accessed episodic memories to semantic
6. **Carrying capacity** — enforce a memory budget, forcing abstraction through merges
7. **Importance recomputation** — recalculate scores for retrieval tiering

### Metamemory

The system tracks its own retrieval accuracy. Per-domain confidence scores (based on user feedback) modulate retrieval scoring. Low-confidence domains get weighted down.

### Suppression system

Called the "immune system" in the code. When a user flags a bad retrieval, a suppression rule is created with a trigger context. Future queries matching that trigger exclude the suppressed memory. It's a context-aware blocklist — the memory isn't deleted, just filtered out for matching queries.

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
| `suppress(crystal_id, reason)` | antibody_id | Block a memory from recall |
| `stats()` | dict | Memory health metrics |
| `export()` | dict | Full data dump |

### REST API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | 3D brain UI |
| `GET` | `/api/stats` | Memory health metrics |
| `GET` | `/api/graph` | Memories, entities, relations for 3D visualization |
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

### WebGL on Linux

The 3D brain requires WebGL. On some Linux systems (especially ARM devices like NVIDIA Jetson, Raspberry Pi, or machines with software rendering), the browser may block WebGL by default. If the 3D view shows "3D brain unavailable", chat and memory still work — only the visualization is affected.

Fixes:
- **Chrome/Chromium**: launch with `--ignore-gpu-blocklist --enable-gpu-rasterization --enable-webgl`
- **Chrome flags**: go to `chrome://flags/#ignore-gpu-blocklist` and enable it, then restart
- **Firefox**: often has better WebGL support on Linux — try `firefox http://localhost:8487`
- **Access from another device**: `second-brain` binds to `0.0.0.0`, so you can open `http://<your-ip>:8487` from a phone, tablet, or another computer on your network

macOS and Windows typically have no WebGL issues.

## Tests

```bash
python -m unittest discover tests/ -v
# 326 passed
```

## Design principles

- **Memories are structured objects, not text blobs.** Records have 25+ fields.
- **Contradiction is data, not error.** Tension is stored, not flattened.
- **Time is a dimension, not metadata.** Decay, reinforcement, prospective triggers.
- **Retrieval is multi-phase, not brute-force.** Graph-first, then embeddings on candidates.
- **The memory tidies itself.** Consolidation runs in the background.
- **Forgetting is a feature.** Carrying capacity forces abstraction.
- **What you see is real.** Every glow, every dim node reflects actual data.
- **The memory learns about itself.** Metamemory tracks its own reliability.
- **Simple entry, deep interior.** `Memory()` — one line, one import.
- **Your data is yours.** Local SQLite, no cloud, no telemetry.
