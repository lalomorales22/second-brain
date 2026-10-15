"""Atomic model changes: never compare vectors from different embedding spaces."""
import json

from semantic_gravity_memory.embeddings.ollama import OllamaEmbedder


def install_embedder(memory, model, url, library=None):
    """Prepare replacement vectors first, then commit and swap the live embedder.

    Caller serializes this operation against ingestion and recall. A failed model
    call leaves every old vector and the live model unchanged.
    """
    storage = memory._storage
    embedder = OllamaEmbedder(model, url) if model else None
    profile = embedder.profile if embedder else ''
    old_profile = storage.get_meta('embedding_profile')
    updates = []
    missing = storage.conn.execute("SELECT 1 FROM crystals WHERE embedding_json IS NULL OR embedding_json='null' LIMIT 1").fetchone()
    if library:
        missing = missing or storage.conn.execute('SELECT 1 FROM source_chunks WHERE profile<>? LIMIT 1', (profile,)).fetchone()
    if embedder and (profile != old_profile or missing):
        targets = [
            ('events', 'embedding_json', 'content'),
            ('crystals', 'embedding_json', None),
            ('schemas', 'embedding_json', 'description'),
            ('prospective_memories', 'trigger_embedding_json', 'trigger_description'),
            ('antibodies', 'trigger_embedding_json', 'trigger_description'),
        ]
        # Validate the model even when the database is empty.
        embedder.embed_query('memory')
        for table, column, text_column in targets:
            for row in storage.conn.execute(f'SELECT * FROM {table}').fetchall():
                if table == 'crystals':
                    event_ids = json.loads(row['source_event_ids_json'] or '[]')
                    events = [storage.get_event(eid) for eid in event_ids]
                    text = '\n'.join(e.content for e in events if e) or row['summary']
                else:
                    text = row[text_column] or ''
                # Existing long chat memories are bounded for the model; sources
                # themselves use complete chunked text in the source library.
                fn = embedder.embed_query if 'trigger' in column else embedder.embed_document
                vector = fn(text[:12000]) if text else None
                updates.append((table, column, row['id'], json.dumps(vector)))
        if library:
            updates.extend(library.prepare_reindex(embedder))
        with storage._lock, storage.conn:
            for table, column, row_id, value in updates:
                storage.conn.execute(f'UPDATE {table} SET {column}=? WHERE id=?', (value, row_id))
            storage.conn.execute("INSERT OR REPLACE INTO meta VALUES ('embedding_profile', ?)", (profile,))
            storage.conn.execute("INSERT OR REPLACE INTO meta VALUES ('embedding_model', ?)", (model,))
            if library:
                storage.conn.execute('UPDATE source_chunks SET profile=?', (profile,))
    engine = memory._engine
    storage.set_meta('embedding_model', model or '')
    storage.set_meta('embedding_configured', 'yes')
    engine.embedder = embedder
    engine.forge.embedder = embedder
    engine.retrieval.embedder = embedder
    if library:
        library.embedder = embedder
    return len(updates)
