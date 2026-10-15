# Workspace review and improvements

The earlier app had a useful memory engine and a graph, but the web ingestion
path accepted only text and there was no source library. The changes focus on
making it usable as a personal, ingestable second brain.

- Added original-file preservation, chunked document ingestion, native image and
  audio embeddings, sampled video frames, deduplication, job progress, source
  previews, timestamp seeking, and source-backed answers.
- Rebuilt the dark interface around the map, source library, and conversation.
  Chat/library initialization is independent of WebGL and the 3D CDN imports.
- Replaced the decorative starfield with only actual data nodes. Added source
  nodes, deterministic initial layout, stable positions on refresh, explicit
  relationship inspection, graph focus controls, and optional similarity edges.
- Fixed web chat dropping user messages while storing assistant answers as
  memories. User statements are now preserved; generated replies stay in history.
- Fixed reasoning text leaking into saved answers when think tags split across
  streaming chunks. Native reasoning fields are not shown or stored as answers.
- Fixed model controls that displayed choices but did not change the live
  embedder. Settings now use the server's actual configuration; model changes
  rebuild vectors atomically and persist across restarts.
- Added EmbeddingGemma query/document prefixes, response validation, and model
  profile tracking. Sources from different embedding profiles are not compared.
- Preserved full text for retrieval and answer grounding. The old forge embedded
  a short summary, and graph-first candidate selection could omit older memories.
  A semantic candidate pass now supplements the original retrieval pipeline.
- Limited upload sizes, validated paths, disabled cross-origin writes, restricted
  media decoding to local protocols, served executable document types as downloads,
  and made loopback the default bind address.
- Added an isolated launcher and optional media dependencies. The original core
  memory library still has no mandatory third-party Python dependencies.

Remaining practical limits are documented in README.md: sampled video without
its soundtrack, audio retrieval without transcription, selectable-text PDFs,
rule-based concept extraction, a bounded graph view, and a single local chat
history. This is a local personal workspace, not an authenticated hosted service.
