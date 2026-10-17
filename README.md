# Second Brain

A local memory workspace you can add to, search, and talk to. Your notes, documents,
images, audio, and video live beside a 3D map of real memories and relationships.
Ollama provides embeddings and chat; originals stay on disk.

## Start

```bash
./start.sh
# Open http://127.0.0.1:8487
```

On macOS, double-click **Start Second Brain.command**. On Windows, double-click
**start.bat**. The launcher creates an isolated `.venv` if necessary and installs
the optional media dependencies. Python 3.10+ and a running Ollama installation
are required.

### Desktop window

To use the workspace in its own window instead of a browser tab (Windows and
macOS, through the system web view via `pywebview`):

```bash
pip install -e '.[media,desktop]'
second-brain-desktop
```

On Windows, double-click **start-desktop.bat**. The window starts Ollama if it is
not already running, serves the same workspace on `127.0.0.1` (port 8487, or a
free port if that one is busy), and stops the server when you close it.

An Electron shell with the same behaviour lives in `desktop/`. It needs Node.js
and uses the repo's `.venv` for the server:

```bash
cd desktop && npm install && npm start
```

On Windows, double-click **start-electron.bat**, which does the same. It is a
dev-machine wrapper: Python and the models are not bundled into the app.

Recommended installed models for this workspace:

```bash
ollama pull embeddinggemma-2
# Use a vision chat model, for example:
ollama pull qwen3.5:9b
./start.sh --embed-model embeddinggemma-2 --chat-model qwen3.5:9b
```

The app auto-detects installed models. You can change them in **Models & settings**.
Selecting a different embedding model rebuilds the index before switching. If a
rebuild fails, the old index remains usable. Chat and embedding choices persist.

## What you can add

| Input | How it is indexed | Limits |
| --- | --- | --- |
| Notes, text, Markdown, code, CSV, JSON, subtitles | Complete text split into overlapping passages | UTF-8, 1 million extracted characters |
| PDF | Selectable text, retaining page markers | 500 pages; scanned pages need OCR first |
| DOCX | Paragraph text | Text only; embedded pictures/tables are not structurally reconstructed |
| PNG, JPEG, WebP | Native image embeddings | Vision embedding model required |
| MP4, MOV, WebM, MKV | Up to 12 evenly spaced image frames with timestamps | 5 minutes; soundtrack and unsampled events are not indexed |
| WAV, MP3, M4A, OGG, FLAC, AAC | Native audio embeddings in 30-second segments | 5 minutes; search/playback, no transcription |

All files are limited to **64 MB**. Byte-identical uploads are deduplicated. A
failed import does not leave a partially indexed source. Text can be imported
without Ollama using lexical search; media requires a compatible embedding model.

**EmbeddingGemma 2 through Ollama** accepts text, image, and audio inputs. Videos
are decoded into individual frames because Ollama's embed API does not accept
native video. Images and frames retrieved for a question are also attached to a
vision-capable chat model (up to three per answer). Audio matches have playback
links; the chat model is told that they are not transcripts.

PDF parsing uses `pypdf`; audio/video decoding uses FFmpeg, either installed on
PATH or bundled by `imageio-ffmpeg`. These are optional: `pip install -e '.[media]'`.

## Talk to your material

1. Use **Add to your brain** or drag files into the import dialog.
2. Open **Source library** to preview originals, inspect passages, and jump to
   video frames or audio segments.
3. Ask a question in **Talk to your brain**. Relevant sources are shown beneath
   the answer; click one to inspect the evidence. Models are prompted to use
   `[S1]` citations, but generated claims still need judgment.
4. Use the search field for direct retrieval without generating an answer.

Your messages become memories. Assistant replies are retained in conversation
history, but the web app does **not** promote them into factual memory nodes.
Conversation history survives reloads. There is one local conversation history
per database, not a multi-user or multi-conversation system.

## A graph grounded in your data

- **Purple spheres:** structured memories. Size reflects stored importance; rings
  indicate semantic memories.
- **Mint polyhedra:** concepts extracted from text. Their size reflects mentions.
- **Amber diamonds:** original source files, linked to their corresponding memory.
- **Solid lines:** stored source links, mentions, co-occurrences, temporal links,
  or contradictions. Click a node to inspect the relationship names.
- **Optional dashed lines:** model-derived similarity suggestions with cosine
  scores. Turn on **Similar** to see them. These are not verified factual links.

The force layout uses those edges to form clusters. Initial positions are
repeatable, existing nodes retain their positions as new nodes arrive, and
there are no decorative starfield nodes. **Fit**, **Orbit**, and **All** control
navigation and highlighting. The graph displays a bounded recent/important
subset (up to 100 source files) for browser performance; the source library
and retrieval remain separate from that visual limit.

Concept extraction and importance scoring remain rule-based. A shared term or
high embedding similarity is a useful clue, not a proof that two claims agree.
Similarity thresholds are conservative heuristics, not calibrated probabilities.

## Data and backups

`./start.sh` uses:

```text
.data/memory.db          # SQLite: memories, conversations, sources, embeddings
.data/memory-assets/     # Original files and decoded media excerpts
```

Back up **both**. **Export index** downloads text/metadata, not the original
binary files. No test content is automatically installed in a new database.

The `second-brain` package command retains the historical default location:
`~/.semantic_gravity_memory/memory.db`, with an adjacent `memory-assets` folder.
Choose any location with `--db`:

```bash
.venv/bin/second-brain --db /path/to/my-memory.db --port 9000
```

The server binds to `127.0.0.1` by default. It has no user authentication; use it
as a personal local app. `--host` is available for explicitly configured trusted
networks. Local models keep inference local. Three.js is loaded from jsDelivr;
the library/chat UI works independently if the 3D dependencies fail to load.

## Python memory engine

The original engine is still available separately, with no required pip dependencies:

```python
from semantic_gravity_memory import Memory

with Memory(db_path='my-memory.db') as memory:
    memory.ingest('I prefer Python for prototypes.')
    scene = memory.recall('What language do I like?')
    memory.consolidate()
```

It includes structured memories, entity relations, importance scoring, temporal
decay, contradiction tracking, spreading activation, feedback, and consolidation.
The web workspace adds a persistent source library and source-grounded chat on top.

## API

| Endpoint | Purpose |
| --- | --- |
| `POST /api/sources?name=filename.ext` | Upload raw bytes; returns an import job |
| `GET /api/jobs/{id}` | Import/reindex progress, result, or error |
| `GET /api/sources` | Source library |
| `GET /api/sources/{id}` | Metadata and extracted passages |
| `GET /api/sources/{id}/file` | Original file, with byte-range support |
| `POST /api/search` | Retrieve source excerpts and memories for `{query}` |
| `POST /api/answer` | Grounded chat via SSE for `{query}` |
| `GET /api/history` | Recent persistent conversation |
| `GET /api/graph` | Actual visible nodes and relationships |
| `GET/POST /api/config` | Settings; embedding changes return a rebuild job |
| `POST /api/ingest` | Save a short text memory for `{text}` |
| `POST /api/consolidate` | Consolidate memories |
| `GET /api/export` | Export memory plus source text/metadata |

## Validate

```bash
.venv/bin/python -m unittest discover -s tests -v
node --check semantic_gravity_memory/ui/workspace.js
node --check semantic_gravity_memory/ui/brain.js
```

The test suite covers the memory engine and regressions in source ingestion,
model rebuilds, evidence preservation, streamed responses, and API behavior.
