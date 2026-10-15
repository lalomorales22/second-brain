# Using your Second Brain

Launch with **Start Second Brain.command** on macOS, or run `./start.sh` and open
[the local workspace](http://127.0.0.1:8487).

## Add something worth keeping

Click **Add to your brain**. Drop files into the outlined area, or give a thought
a title and save it as a note. Imports show progress and explicit errors. The
original is preserved even after it has been converted into searchable excerpts.

Documents, photos, audio, and video each have a card in **Source library**.
Open a card to inspect the source, read its extracted text, play it, download
it, or ask a question about it. A repeated upload of the same bytes points to
the existing source rather than making duplicate memories.

Video imports are a sample of up to 12 frames across a clip. Click a thumbnail
to jump the player to that moment. The soundtrack is not searched. Audio imports
are indexed in 30-second segments and remain playable, but there is no generated
transcript. Clips must be no longer than five minutes. Scanned PDFs require OCR
outside the app before text import.

## Follow an idea

The search bar finds matching excerpts and related memories. Similarity scores
are rankings, not confidence that a statement is true. Try descriptions in your
own words: “the launch date we agreed on,” “a blue dashboard,” or “a ringing bell.”

Chat retrieves material from your library and memory before generating an answer.
Sources appear as clickable cards beneath the answer. A vision-capable chat
model can inspect matching photos and video frames. Audio can be retrieved and
played, but the assistant cannot quote speech without a transcript.

Your words become memories. The assistant's replies are saved in history without
being turned into factual memories of their own. Reloading restores the recent
conversation. Use **Save as a note** for something you want to keep as a source.

## Explore the map

Drag to orbit, scroll to zoom, and click a node to see its actual connections.
Purple spheres are memories, mint polyhedra are concepts, and amber diamonds
are source files. **Fit** reframes the map, **Orbit** toggles gentle rotation,
and **All** clears the focus.

**Similar** adds dashed embedding-similarity suggestions. These are optional and
are clearly distinguished from stored relationships. There are no placeholder
nodes: an empty database starts with an empty map.

At narrow window sizes the graph and chat stack vertically. Tap the view name
in the top bar to switch between the map and library.

## Choose your models

Open **Models & settings**. The chat model writes answers; the memory model finds
related material. EmbeddingGemma 2 is the preferred memory model when installed.
A local vision model such as Qwen3.5 can also inspect image evidence.

Changing the memory model rebuilds embeddings in the background. Existing
vectors are replaced only after all replacements are ready. Errors preserve
the previous index. Imported media needs a multimodal embedding model.

**Consolidate memories** applies the original engine's decay/merge/importance
logic to memory records. Source originals and their searchable passages remain
separate and are not deleted by consolidation.

## Keep a backup

With the launcher, copy `.data/memory.db` and `.data/memory-assets/` together.
Close the server before a simple file-copy backup, or use a SQLite-aware backup.
The JSON **Export index** is useful for inspecting text and metadata; it does not
include binary originals. Keep both the database and assets for a full backup.
