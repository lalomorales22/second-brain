# Semantic Gravity Memory — Build Plan

Zero external Python dependencies. Every algorithm hand-built. `pip install` gives you a living memory.

---

## Architecture

```
semantic_gravity_memory/
├── __init__.py                 # Public API: Memory(backend, embedder)
├── models.py                   # All dataclasses (Crystal, Event, Entity, etc.)
├── utils.py                    # Math, time, text helpers — all hand-rolled
├── storage/
│   ├── __init__.py
│   ├── base.py                 # Abstract storage interface
│   └── sqlite_backend.py       # SQLite implementation (stdlib only)
├── embeddings/
│   ├── __init__.py
│   ├── base.py                 # Abstract embedder interface
│   └── ollama.py               # Ollama local embedder (urllib only)
├── core/
│   ├── __init__.py
│   ├── engine.py               # Main MemoryEngine orchestrator
│   ├── crystal_forge.py        # Event → Crystal pipeline
│   ├── entity_extractor.py     # Hand-built NLP entity extraction
│   ├── salience.py             # Multi-dimensional salience scoring
│   ├── self_state.py           # Learned self-state detection
│   ├── contradiction.py        # Temporal + factual + preference conflict detection
│   ├── temporal.py             # Decay, clustering, temporal gravity, prospective memory
│   ├── retrieval.py            # Spreading activation, scene reconstruction
│   ├── consolidation.py        # Background daemon — sleep cycles, merging, schemas
│   ├── metamemory.py           # Confidence calibration from feedback
│   └── immune.py               # Antibody memories — suppress known-bad patterns
└── gui/                        # Optional tkinter app (Phase 6)
    └── app.py
tests/
├── __init__.py
├── test_models.py
├── test_utils.py
├── test_storage.py
├── test_crystal_forge.py
├── test_entity_extractor.py
├── test_salience.py
├── test_contradiction.py
├── test_temporal.py
├── test_retrieval.py
├── test_consolidation.py
├── test_metamemory.py
├── test_immune.py
└── test_integration.py
pyproject.toml
```

---

## Phase 1: Foundation Layer
**Status: COMPLETE**

The skeleton. Every data structure, the storage contract, the SQLite engine, the embedder interface, and math/time/text utilities. Nothing intelligent yet — but everything built so that intelligence has a place to live.

### Files to Create
- [x] `pyproject.toml` — package config, zero dependencies
- [x] `semantic_gravity_memory/__init__.py` — public exports
- [x] `semantic_gravity_memory/models.py` — all 10 dataclasses:
  - `SalienceVector` (6-dimensional: emotional, practical, identity, temporal, uncertainty, novelty)
  - `Event` (raw experience atom)
  - `Entity` (named concept with mention tracking)
  - `Crystal` (the core memory unit — 25+ fields including memory_type, access_count, decay_rate, version, parent linkage)
  - `Relation` (typed weighted edge)
  - `Contradiction` (recorded tension with resolution tracking)
  - `Activation` (recall event with metamemory quality_score)
  - `ProspectiveMemory` (future-triggered recall)
  - `Schema` (abstracted pattern from crystal clusters)
  - `AntibodyMemory` (immune system — suppresses known-bad recall)
- [x] `semantic_gravity_memory/utils.py` — hand-rolled:
  - `cosine_similarity` — pure Python dot product
  - `exponential_decay` — `initial * e^(-rate * elapsed)`
  - `sigmoid` — activation function
  - `now_iso`, `parse_iso`, `seconds_between`, `hours_since` — temporal math
  - `slugify`, `summarize_text` — text compression
  - `safe_json_loads`, `safe_json_dumps` — defensive serialization
- [x] `semantic_gravity_memory/storage/base.py` — abstract interface (40+ methods)
- [x] `semantic_gravity_memory/storage/sqlite_backend.py` — full SQLite implementation:
  - 10 tables with indexes
  - WAL mode, thread-safe writes
  - JSON serialization for complex fields
  - SalienceVector round-trip
  - Full export to dict
- [x] `semantic_gravity_memory/embeddings/base.py` — abstract embedder
- [x] `semantic_gravity_memory/embeddings/ollama.py` — Ollama HTTP embedder (urllib only)
- [x] `tests/test_models.py` — model creation, serialization, SalienceVector math
- [x] `tests/test_utils.py` — every utility function
- [x] `tests/test_storage.py` — full CRUD for all 10 entity types

### Acceptance Criteria
- `python -m pytest tests/test_models.py tests/test_utils.py tests/test_storage.py` — all green
- Zero imports outside Python stdlib
- Every model round-trips through SQLite without data loss
- SalienceVector combined score computes correctly with custom weights
- Temporal utilities give correct time math
- Storage handles upserts, updates, and edge cases

---

## Phase 2: Crystal Formation Pipeline
**Status: COMPLETE**

The perceptual system. Raw text goes in, structured crystals come out. Entity extraction, salience scoring, self-state detection — all hand-built, no spaCy, no NLTK.

### Files to Create
- `semantic_gravity_memory/core/entity_extractor.py`
  - Multi-pass extraction: capitalized phrases, compound nouns, camelCase, tool names
  - Co-occurrence tracking (entities that appear together get relation edges)
  - Entity merging (detect "Python" and "python3" as the same entity)
  - Relationship extraction ("deployed X to Y" → relation between X and Y)
- `semantic_gravity_memory/core/salience.py`
  - Keyword-seeded scoring for all 6 salience dimensions
  - Novelty detection: compare incoming text against recent crystal embeddings
  - Context-sensitive boosting (urgency words boost temporal, question marks boost uncertainty)
- `semantic_gravity_memory/core/self_state.py`
  - Seed vocabulary for initial states (builder, founder, student, etc.)
  - Learning: track which entity clusters activate with which self-states
  - State discovery: if a new cluster appears that doesn't match known states, create one
  - State transitions: detect when user switches context ("ok now about my side project...")
- `semantic_gravity_memory/core/crystal_forge.py`
  - Full pipeline: text → entities → salience → embedding → crystal
  - Title generation from entity clusters
  - Future implications inference
  - Unresolved item detection
  - Compressed narrative generation
  - Initial memory_type assignment (episodic by default)
  - Confidence scoring (user input > assistant output > system observation)
- `semantic_gravity_memory/core/contradiction.py`
  - Preference contradictions ("I like X" vs "I hate X")
  - Temporal contradictions ("I'll finish by Friday" → [Saturday] "haven't started")
  - Factual contradictions ("the API uses REST" vs "the API uses GraphQL")
  - Belief evolution tracking (not every change is a contradiction — some are growth)
  - Resolution suggestions based on evidence weight
- `tests/test_crystal_forge.py`
- `tests/test_entity_extractor.py`
- `tests/test_salience.py`
- `tests/test_contradiction.py`

### Acceptance Criteria
- Entity extraction finds compound names, tools, relationships
- Salience scores vary meaningfully across different input types
- Self-state detection works for seed states and discovers new ones
- Crystal forge produces complete crystals with all fields populated
- Contradiction detector catches preference, temporal, and factual conflicts
- All tests green, zero external dependencies

---

## Phase 3: Temporal Engine
**Status: COMPLETE**

Time as a first-class dimension. Memories decay, cluster into episodes, exert gravitational pull based on temporal proximity, and can trigger future recall.

### Files to Create
- `semantic_gravity_memory/core/temporal.py`
  - **Exponential decay with reinforcement**: every crystal's strength decays over time, but accessing it resets the curve. Formula: `strength = base * e^(-decay_rate * hours_since_last_access) + (access_count * reinforcement_bonus)`
  - **Temporal clustering**: events within a configurable window (default 4 hours) form an episode. Episodes are first-class objects stored as relations between their member crystals.
  - **Temporal gravity**: when retrieving, crystals close in time to each other get a proximity bonus. Three tax conversations this week have more combined pull than three spread across months.
  - **Recency weighting**: configurable curve that boosts recent crystals without killing old important ones. Old + high salience can still beat new + low salience.
  - **Temporal contradiction detection**: "I'll do X by date Y" + current_date > Y + no completion event = temporal contradiction.
  - **Prospective memory engine**: check incoming text against active trigger embeddings. If similarity exceeds threshold, fire the prospective memory and inject its payload crystal into the scene.
  - **Memory versioning**: when a crystal gets updated (new info on same topic), increment version and store the previous state as a snapshot. Enables "what did I believe about X last month?"
- `tests/test_temporal.py`

### Acceptance Criteria
- Decay reduces crystal strength over simulated time
- Reinforcement (access) resets decay curve
- Temporal clusters form automatically from time-proximate events
- Prospective memories fire when trigger conditions match
- Memory versions track belief evolution
- All temporal math uses hand-rolled functions from utils.py

---

## Phase 4: Retrieval & Activation
**Status: COMPLETE**

The recall system. Not keyword search — spreading activation through a knowledge graph, scene reconstruction from activated fragments, and metamemory feedback loops.

### Files to Create
- `semantic_gravity_memory/core/retrieval.py`
  - **Spreading activation**: start from query embedding matches, propagate energy along relation edges with decay per hop. 3 hops max by default. Energy = `initial_score * edge_weight * hop_decay_factor`.
  - **Scene reconstruction**: gather activated crystals + entities + contradictions + prospective memories. Build a coherent scene dict with:
    - active self-state
    - dominant crystals (sorted by activation energy)
    - active entities (with mention counts and salience)
    - relevant contradictions
    - fired prospective memories
    - scene narrative summary
  - **Antibody filtering**: before returning scene, check all active antibodies. If an antibody's trigger matches the query AND its target crystal is in the scene, suppress that crystal and log the suppression.
  - **Working memory buffer**: maintain a limited-capacity (configurable, default 7) set of "currently held" crystals that persist across multiple queries within a session. These get a retrieval bonus without needing embedding match.
  - **Episodic vs. semantic retrieval**: episodic crystals retrieved by temporal proximity + embedding. Semantic crystals retrieved by embedding only (they're timeless).
- `semantic_gravity_memory/core/metamemory.py`
  - **Quality tracking**: after each retrieval, accept a quality signal (explicit: "good answer" / "that's wrong", or implicit: follow-up question suggests confusion).
  - **Per-domain calibration**: track accuracy by self-state, by entity cluster, by crystal theme. Build confidence modifiers: "when recalling about code, I'm 85% useful; about personal preferences, 60%."
  - **Retrieval history analysis**: which crystals keep getting recalled? Which never do? Feed this into consolidation.
- `semantic_gravity_memory/core/immune.py`
  - **Antibody creation**: when a user corrects a recall ("no, that's wrong"), create an antibody that suppresses the bad crystal for that trigger context.
  - **Antibody decay**: antibodies weaken over time too. If the underlying crystal gets updated/versioned, the antibody may no longer apply.
  - **Antibody check**: fast pre-filter before scene construction.
- `tests/test_retrieval.py`
- `tests/test_metamemory.py`
- `tests/test_immune.py`

### Acceptance Criteria
- Spreading activation reaches related crystals through 2-3 hops
- Scene reconstruction produces coherent context for LLM grounding
- Antibodies suppress known-bad retrievals
- Working memory buffer persists across queries in a session
- Metamemory tracks per-domain confidence
- All tests green

---

## Phase 5: Consolidation & Living Memory
**Status: COMPLETE**

The heartbeat. Background processes that keep the memory alive: merging, abstracting, decaying, competing. The memory breathes even when no one is talking.

### Files to Create
- `semantic_gravity_memory/core/consolidation.py`
  - **Consolidation daemon**: background thread with configurable heartbeat (default: every 5 minutes). Runs a consolidation pass:
    1. **Decay pass**: apply temporal decay to all crystals. Crystals below a strength threshold get marked dormant.
    2. **Merge pass**: find crystal pairs with cosine similarity > 0.85. Merge into a super-crystal: combine source events, merge entity sets, average salience vectors, keep the better title, concatenate narratives, increment version. Old crystals get `parent_crystal_id` pointing to the merge.
    3. **Schema extraction pass**: find 3+ crystals with overlapping entity sets and similar themes. Abstract into a Schema: "When user is debugging, they typically check logs → reproduce → isolate test."
    4. **Contradiction resolution pass**: for open contradictions, check evidence weights. If one side has 3x the evidence, auto-resolve.
    5. **Episodic → semantic graduation**: crystals accessed 5+ times with low temporal salience graduate from episodic to semantic. Their temporal anchors loosen; they become "things I know" rather than "things that happened."
    6. **Carrying capacity enforcement**: if total active crystals exceed a configurable limit (default: 500), force-merge or dormant the weakest until under budget. This prevents unbounded growth and forces abstraction.
  - **Manual triggers**: `memory.consolidate()` runs a pass immediately.
  - **Consolidation log**: every pass records what it did (merged N crystals, decayed M, graduated K). Useful for debugging and transparency.
- `semantic_gravity_memory/core/engine.py`
  - **Main orchestrator**: ties together crystal forge, retrieval, consolidation, temporal engine, metamemory, and immune system.
  - `ingest(text, actor, kind, context)` → event + crystal + entity extraction + contradiction check
  - `recall(query)` → scene reconstruction with spreading activation
  - `answer(query, chat_fn)` → recall + format scene + call external chat function
  - `consolidate()` → manual consolidation trigger
  - `feedback(activation_id, quality)` → metamemory signal
  - `set_prospective(trigger, crystal_id, expiry)` → future-triggered recall
  - `suppress(crystal_id, reason)` → create antibody
  - `export()` → full memory dump
  - `stats()` → memory health metrics
- `tests/test_consolidation.py`

### Acceptance Criteria
- Consolidation daemon runs on a background thread
- Crystal merging reduces count while preserving information
- Schema extraction identifies recurring patterns
- Episodic → semantic graduation works after repeated access
- Carrying capacity prevents unbounded growth
- Engine ties all subsystems together cleanly
- All tests green

---

## Phase 6: API Surface, Packaging & GUI
**Status: COMPLETE**

The shell. A clean public API, pip-installable package, optional GUI, and comprehensive testing.

### Files to Create / Modify
- `semantic_gravity_memory/__init__.py` — finalize public API:
  ```python
  from semantic_gravity_memory import Memory

  memory = Memory()  # SQLite + Ollama defaults
  memory.ingest("the user prefers Python for prototypes")
  scene = memory.recall("what language does the user like?")
  memory.consolidate()
  ```
- `semantic_gravity_memory/gui/app.py` — tkinter GUI rebuilt on clean API:
  - Chat + memory tab
  - Graph visualization (force-directed layout, not just circular)
  - Crystal browser with inline editing
  - Temporal timeline view
  - Consolidation dashboard (what merged, what decayed, what graduated)
  - Metamemory confidence display
  - Antibody manager
- `pyproject.toml` — finalized for PyPI:
  - `pip install semantic-gravity-memory` (core, no GUI)
  - `pip install semantic-gravity-memory[gui]` (with tkinter noted)
  - `pip install semantic-gravity-memory[ollama]` (convenience alias)
- `tests/test_integration.py` — end-to-end:
  - Ingest 50 messages across 5 simulated days
  - Verify crystal formation, entity extraction, contradiction detection
  - Run consolidation, verify merging and schema extraction
  - Verify temporal decay after simulated time passage
  - Verify prospective memory firing
  - Verify antibody suppression
  - Benchmark: 1000 crystals, recall under 200ms
- Documentation:
  - Updated README.md with new architecture
  - API reference (docstrings, not a separate site)
  - Examples directory

### Acceptance Criteria
- `pip install .` works from the repo root
- `from semantic_gravity_memory import Memory` gives you a working memory system in 1 line
- GUI shows all memory subsystems visually
- Integration tests pass end-to-end
- No external Python dependencies in the core package
- All tests green: `python -m pytest tests/`

---

## Dependency Rules (Enforced Across All Phases)

1. **Zero PyPI dependencies for core**: sqlite3, json, math, datetime, re, threading, urllib, dataclasses, abc, typing, collections, hashlib, os, time — all stdlib.
2. **Ollama is a system service, not a Python dependency**: we talk to it over HTTP with urllib.
3. **tkinter is optional**: GUI is a separate extra, not required for the core memory engine.
4. **Every algorithm is ours**: cosine similarity, exponential decay, spreading activation, temporal clustering, entity extraction, schema abstraction — all hand-built.
5. **Tests use unittest** (stdlib). No pytest dependency required (but compatible with it).

---

## Design Principles

- **Memories are structured objects, not text blobs.** Crystals have 25+ fields. This is a database of meaning, not a vector store.
- **Contradiction is data, not error.** The system stores tension instead of flattening it.
- **Time is a dimension, not metadata.** Temporal proximity, decay, and prospective triggering are first-class operations.
- **The memory breathes.** Consolidation runs even when idle. Memories compete, merge, decay, and graduate.
- **Forgetting is a feature.** Carrying capacity forces abstraction. Decay prevents noise accumulation.
- **The memory learns about itself.** Metamemory tracks its own reliability. Antibodies prevent repeated mistakes.
- **Simple entry, deep interior.** `Memory()` gives you everything. One line. One import.
