# Second Brain — How It Actually Works
### The "I don't read code" guide

---

## The Big Picture (30 seconds)

You type something. The brain **doesn't** search through everything you've ever told it. Instead, it works like YOUR brain:

1. Your words get turned into a **"vibe"** (a math fingerprint)
2. That vibe gets compared to the vibes of stored memories
3. The **best matches wake up** and pass energy to their neighbors
4. Only the **top 8 memories** make it to the AI
5. The AI reads those 8 memories and responds

**The AI never sees your whole database.** It only sees 8 hand-picked memories per question.

---

## What Happens When You Type a Message

### Step 1: Your words become a fingerprint
Your message gets sent to Ollama (the local AI running on your machine), which converts it into a list of ~384 numbers. This is called an **embedding** — think of it as the "vibe" or "DNA" of what you said.

> "I love writing jokes about AI" → [0.23, -0.15, 0.87, 0.04, ...]

### Step 2: Compare vibes with stored memories
Every memory in the system also has a fingerprint. The system compares your fingerprint against ALL stored memory fingerprints using **cosine similarity** (basically: how much do these two vibes point in the same direction?).

> This is like walking into a library and every book starts glowing based on how related it is to what you're thinking about.

### Step 3: Bonus points
Raw vibe-matching isn't enough. Memories also get bonus points for:
- **Recency** — stuff you talked about recently gets a boost
- **Context match** — if you're in "work mode," work memories score higher
- **Working memory** — the last 7 memories you recalled stick around (like keeping browser tabs open)
- **Strength** — memories you've recalled many times are stronger

### Step 4: Spreading activation (the cool part)
The top 12 scoring memories become **seeds**. Then energy **spreads through the graph** — like ripples in water:

```
You ask about "Python"
  → "Python" crystal lights up
    → energy flows to connected entity "programming"
      → "programming" activates nearby crystal about "code review"
        → which activates "debugging" entity
```

This means the system can recall things that are **indirectly related** to your question — not just keyword matches. It thinks associatively, like you do.

### Step 5: Pick the winners
After activation spreads (3 hops max), the system picks the **top 8 crystals** with the most energy. These are your active memories.

### Step 6: The AI reads the scene
Those 8 memories get formatted into a **prompt** that looks like:

```
Here's what I remember:
[episodic] "Pizza debate": User prefers thin crust over deep dish
[semantic] "Python preference": User always prototypes in Python
[episodic] "Tuesday standup": User mentioned deadline is Friday
...

User's question: "What should I work on today?"
```

The AI (Gemma, Llama, etc.) reads this and responds. **It only sees these 8 memories**, not your whole database.

### Step 7: The response becomes a memory too
The AI's answer gets ingested back into the system as a new crystal. The brain remembers what it said.

---

## What's a Crystal?

A **crystal** is one compressed memory. When you type "I had pizza for lunch and it was amazing," the system creates a crystal like:

| Field | Value |
|-------|-------|
| **title** | "Pizza lunch experience" |
| **summary** | "User had pizza for lunch and loved it" |
| **self_state** | "personal" |
| **memory_type** | "episodic" (a thing that happened) |
| **salience** | emotional: 0.3, practical: 0.1, novelty: 0.2 |
| **entities** | ["pizza", "lunch"] |

Over time, if you keep talking about pizza, those crystals can **merge** into a semantic crystal: "User loves pizza, especially thin crust." Episodic (what happened) graduates to semantic (what I know).

---

## What's an Entity?

An **entity** is a recurring concept the system extracted: a person, tool, place, project, or idea. Every time you mention "Python," the Python entity gets reinforced — its mention count goes up and its salience grows.

Entities are the **connectors** between crystals. Two memories that both mention "Python" are connected through that entity.

---

## What's the Graph?

The 3D visualization you see is the **knowledge graph**:

- **Big glowing orbs** = crystals (memories)
- **Small floating shapes** = entities (concepts)
- **Lines between them** = relationships

The colors mean things:
- **Blue** crystals = builder/professional context
- **Gold** = founder mode
- **Green** = student/learning
- **Purple** = creative
- **Orange** = family
- **Red flickering** = contradiction (two memories disagree)

Line colors:
- **Blue** lines = "mentions" (crystal talks about an entity)
- **Green** lines = "co-occurred" (appeared together)
- **Amber** lines = "temporal cluster" (happened around the same time)
- **Red** lines = "contradicts" (memories in tension)

---

## What's Consolidation?

Like sleep for the brain. When you hit "consolidate":

1. **Similar crystals merge** — if you said "I like Python" five different times, they become one strong crystal
2. **Weak memories fade** — things with low salience that you never recalled again decay away
3. **Patterns emerge** — repeated patterns get abstracted into **schemas** (templates)
4. **Episodic → Semantic** — specific events graduate to general knowledge

---

## How Does It Handle Lots of Data? (Gravitational Retrieval)

Here's what's revolutionary: **it does NOT search through everything.**

Most AI memory systems (RAG) compare your question against every single stored memory. That's like a librarian walking past every book to find the right one. Slow.

This system works like your actual brain: **memories aren't searched, they're awakened.**

### The 3-Phase Retrieval (before any AI math happens):

**Phase 1: Entity Gateway** (instant, zero AI)
> The system looks at the WORDS in your question. If you ask about "Python", it finds the "Python" entity in the graph and follows its connections to find related crystals. No math. Pure graph walking.

**Phase 2: Gravity Orbit** (instant, pre-computed)
> Every crystal has a "gravitational mass" — computed during consolidation based on how important, how frequently accessed, and how recent it is. The heaviest crystals are always ready, like tip-of-the-tongue memories. They don't need to be searched.

**Phase 3: Resonance Field** (instant, cached)
> Whatever you were just talking about stays "warm." Ask about Python, and next time you ask about coding, those Python memories are still primed. Like how thinking about dogs makes it easier to remember cats.

**THEN** — and only then — does it compare embeddings. But instead of comparing against ALL 1000+ crystals, it only checks the ~40-60 candidates from the three phases above.

Think of it like: instead of walking past every book, the librarian already knows which shelf you need (entity gateway), has your favorite books on the counter (gravity orbit), and remembers what you were reading last time (resonance field).

---

## The Flow, Visually

```
  YOU TYPE SOMETHING
         │
         ▼
  ┌─────────────┐
  │  Embed query │  ← Ollama turns your words into numbers
  └──────┬──────┘
         ▼
  ┌──────────────────────────────────────────┐
  │  PHASE 1: Entity Gateway                  │
  │  Find "Python" in graph → connected       │  ← No AI math, pure graph
  │  crystals (~20 candidates)                │
  ├──────────────────────────────────────────┤
  │  PHASE 2: Gravity Orbit                   │
  │  Always include the "heavy" crystals      │  ← Pre-computed, instant
  │  (top 30 by importance)                   │
  ├──────────────────────────────────────────┤
  │  PHASE 3: Resonance Field                 │
  │  Include primed crystals from recent      │  ← Cached, instant
  │  conversations                            │
  └──────┬───────────────────────────────────┘
         ▼
  ┌─────────────────┐
  │ Score ~50 crystals│  ← Compare fingerprints (FAST — only candidates)
  └──────┬──────────┘
         ▼
  ┌─────────────────┐
  │ Top 12 → spread  │  ← Energy ripples through the graph
  │   activation      │
  └──────┬──────────┘
         ▼
  ┌─────────────────┐
  │  Pick top 8      │  ← Your "active scene"
  └──────┬──────────┘
         ▼
  ┌─────────────────┐
  │ Format prompt +   │  ← AI reads 8 memories + your question
  │ send to AI model  │
  └──────┬──────────┘
         ▼
  ┌─────────────────┐
  │  AI responds      │  ← Only sees 8 crystals, not everything
  └──────┬──────────┘
         ▼
  ┌─────────────────┐
  │ Response becomes  │  ← The brain remembers what it said
  │  a new crystal    │
  └─────────────────┘
```

---

## Quick Reference

| Term | What it means |
|------|--------------|
| **Crystal** | One compressed memory |
| **Entity** | A recurring concept (person, tool, idea) |
| **Relation** | A connection between two things |
| **Embedding** | A math fingerprint of text (~384 numbers) |
| **Salience** | How important something is (6 dimensions) |
| **Self-state** | Your current context (builder, creative, etc.) |
| **Spreading activation** | Energy rippling through connected memories |
| **Working memory** | The last 7 things recalled (like open tabs) |
| **Consolidation** | Brain cleanup — merge, decay, and abstract |
| **Schema** | A pattern extracted from many similar crystals |
| **Antibody** | A filter that blocks bad recall patterns |
| **Prospective memory** | A future trigger ("remind me when X comes up") |

---

## TL;DR

Your brain stores thousands of memories but you only think about ~7 things at once. This system works the same way. It doesn't dump your whole database on the AI — it reconstructs a tiny, relevant **scene** of 8 memories using vibe-matching and associative spreading. The AI only ever sees that small scene plus your question. The 3D visualization is showing you the full graph of everything stored, with the active scene glowing bright.
