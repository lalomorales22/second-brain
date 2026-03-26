# Handoff: Second Brain — Next Session

## What exists

A complete AI memory system called **Second Brain** (`semantic-gravity-memory`). Zero external Python dependencies. 326 tests, all passing. Lives at [github.com/lalomorales22/second-brain](https://github.com/lalomorales22/second-brain).

### The engine (done, solid, tested)
- **Memory crystals** — structured memory objects with 25+ fields, 6-dimensional salience, contradiction tracking, temporal decay, versioning
- **Entity extraction** — hand-built 8-pass NLP, no spaCy
- **Spreading activation** — energy propagates through a knowledge graph, not just cosine similarity
- **Temporal engine** — decay, reinforcement, episode clustering, prospective memory, gravity
- **Consolidation daemon** — background thread that merges, decays, graduates, extracts schemas, enforces carrying capacity
- **Immune system** — antibodies suppress known-bad recall patterns
- **Metamemory** — tracks its own accuracy per domain
- **One-line API**: `from semantic_gravity_memory import Memory`

### The current GUI (functional but flat)
- tkinter desktop app, dark theme, 5 tabs (chat, graph, crystals, stats, system)
- Combobox dropdowns for Ollama models (auto-populated from local server)
- Graph tab uses a hand-built force-directed spring-electric layout
- It works. It's clean. But it looks like a 2D dashboard. **No wow factor.**

## What to build next: THE UI

The engine is the brain. Now it needs a **face**. The goal is to make the memory system feel alive — like you're looking into a living, breathing neural network.

### Vision: 3D world-like environment

Think less "database viewer" and more "you're inside the brain looking at thoughts forming."

#### Graph visualization — the centerpiece
- **3D force-directed graph** — crystals and entities floating in space, orbiting, pulsing
- **Depth and parallax** — nodes closer to camera are larger/brighter, distant ones fade
- **Glow and particle effects** — active crystals pulse with light, spreading activation visualized as energy waves traveling along edges
- **Organic motion** — nodes drift slightly even at rest, like neurons idling. Not static. Alive.
- **Zoom and rotate** — mouse wheel zooms, drag rotates the camera around the scene
- **Clustering visible** — temporal episodes form visible nebula-like clusters, self-states have color-coded regions
- **Real-time activation** — when you send a query, watch the activation energy ripple through the graph in real time

#### Crystal nodes
- **Size = strength** — strong crystals are large glowing orbs, weak ones are dim specks
- **Color = self-state** — builder=blue, founder=gold, student=green, creative=purple, family=warm orange
- **Pulse = recent access** — just-recalled crystals pulse brighter then fade
- **Ring = memory type** — semantic crystals have a solid ring, episodic have a dashed/fading ring
- **Contradiction halos** — crystals in tension have a red flicker

#### Entity nodes
- **Smaller, satellite-like** — orbit around their most connected crystal
- **Brightness = mention count** — frequently mentioned entities glow hotter
- **Trails** — when entities connect to multiple crystals, faint trails show the connections

#### Edges
- **Thickness = weight** — strong relations are thick bright lines, weak ones are thin and translucent
- **Type coloring** — mentions=blue, co_occurred=green, temporal_cluster=amber, contradicts=red
- **Animated flow** — tiny particles travel along edges when activation is spreading

#### Scene reconstruction visualization
- When a recall happens, the relevant subgraph **lifts forward** and the rest dims. Like a thought emerging from background noise. The scene narrative appears as floating text near the active cluster.

#### Consolidation visualization
- When consolidation runs, show it: crystals merging (two orbs flowing into one), dormant crystals fading and sinking, graduated crystals changing color from episodic to semantic. The brain tidying itself up in real time.

### Technology options
- **Option A: tkinter + Canvas 3D projection** — keep zero dependencies, hand-roll 3D→2D projection with rotation matrices. Impressive but limited.
- **Option B: Web-based (Flask/FastAPI + Three.js/WebGL)** — the engine stays Python, the UI is a local web app with a 3D scene. Most visual potential. Three.js gives particles, bloom, postprocessing for free.
- **Option C: PyOpenGL** — real 3D in a desktop window. Heavyweight but native.
- **Recommended: Option B** — Flask serves the Memory API as JSON endpoints, a single HTML page with Three.js renders the 3D brain. Still local-first, still `pip install`, but the visuals go from "dashboard" to "wow."

### The experience we want
1. User opens the app → sees a slowly rotating 3D brain-like structure
2. They type a message → watch crystals form in real time, edges snap into place
3. They ask a question → see activation energy ripple outward from the query point, relevant crystals light up and pull forward
4. They trigger consolidation → watch merges happen (two orbs becoming one), weak nodes dissolve, schemas crystallize
5. Contradictions show as red tension lines crackling between opposing crystals
6. The whole thing breathes — subtle ambient motion, particles drifting, gentle glow pulsation

### Files to expect
- `semantic_gravity_memory/api/server.py` — lightweight Flask/FastAPI serving Memory as REST endpoints
- `semantic_gravity_memory/ui/` — static HTML/JS/CSS with Three.js scene
- Or if staying tkinter: enhanced `gui/app.py` with 3D projection math on Canvas

### What NOT to change
- The engine (`core/`, `storage/`, `models.py`, `utils.py`) is done and tested. Don't refactor it.
- The `Memory` class API is stable. Build the UI on top of it.
- Keep tests passing. 326 tests, all green.

## Key files to read first
- `semantic_gravity_memory/__init__.py` — the `Memory` class, the public API
- `semantic_gravity_memory/core/engine.py` — `MemoryEngine` orchestrator
- `semantic_gravity_memory/core/retrieval.py` — `RetrievalEngine`, scene dict structure
- `semantic_gravity_memory/gui/app.py` — current tkinter GUI (reference for what exists)
- `tasks.md` — the full 6-phase build plan (all complete)
- `README.md` — architecture overview and API reference
