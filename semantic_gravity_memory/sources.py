"""Persistent local source library: originals, searchable passages, and media excerpts.

Text/images need only stdlib + Ollama. PDF and audio/video decoding use optional
media extras. No uploaded bytes or local paths are sent anywhere except Ollama.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import io
import json
import math
import mimetypes
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import uuid
import zipfile
from xml.etree import ElementTree

from semantic_gravity_memory.utils import cosine_similarity, now_iso

MAX_BYTES = 64 * 1024 * 1024
MAX_TEXT = 1_000_000
MAX_DURATION = 300
TEXT_EXT = {'.txt', '.md', '.markdown', '.csv', '.json', '.py', '.js', '.ts', '.tsx', '.jsx', '.css', '.html', '.log', '.yaml', '.yml', '.srt', '.vtt'}
IMAGE_EXT = {'.png', '.jpg', '.jpeg', '.webp'}
VIDEO_EXT = {'.mp4', '.mov', '.webm', '.mkv'}
AUDIO_EXT = {'.wav', '.mp3', '.m4a', '.ogg', '.flac', '.aac'}


def ffmpeg_path():
    executable = shutil.which('ffmpeg')
    if executable:
        return executable
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return None


def capabilities():
    try:
        import pypdf  # noqa: F401
        pdf = True
    except ImportError:
        pdf = False
    from semantic_gravity_memory import youtube
    return {'pdf': pdf, 'media': bool(ffmpeg_path()), 'youtube': youtube.available(), 'max_bytes': MAX_BYTES,
            'max_duration': MAX_DURATION, 'video_frames': 12,
            'extensions': sorted(TEXT_EXT | IMAGE_EXT | VIDEO_EXT | AUDIO_EXT | {'.pdf', '.docx'})}


def chunk_text(text, size=2800, overlap=240):
    """Cover the whole text, preserving paragraphs when possible."""
    text = text.replace('\x00', '').strip()
    if not text:
        raise ValueError('This file has no extractable text. Scanned PDFs need OCR first.')
    if len(text) > MAX_TEXT:
        raise ValueError('Text is limited to 1 million characters per source. Split this document first.')
    pieces = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            boundary = text.rfind('\n', start + size // 2, end)
            if boundary > start:
                end = boundary
        pieces.append(text[start:end])
        if end == len(text):
            break
        start = max(start + 1, end - overlap)
    return pieces


def extract_document(path, extension):
    if extension == '.pdf':
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ValueError('PDF import requires the media extra: pip install -e ".[media]"') from exc
        reader = PdfReader(path)
        if reader.is_encrypted:
            raise ValueError('Unlock this PDF before importing it.')
        if len(reader.pages) > 500:
            raise ValueError('PDFs are limited to 500 pages per source.')
        text = '\n\n'.join(f'[Page {i + 1}]\n{page.extract_text() or ""}' for i, page in enumerate(reader.pages))
        if not any((page.extract_text() or '').strip() for page in reader.pages):
            raise ValueError('This PDF has no selectable text. Run OCR first, or import its pages as images.')
        return text
    if extension == '.docx':
        with zipfile.ZipFile(path) as archive:
            info = archive.getinfo('word/document.xml')
            if info.file_size > MAX_TEXT * 8:
                raise ValueError('This Word document is too large to extract.')
            root = ElementTree.fromstring(archive.read(info))
        ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
        return '\n'.join(''.join(p.itertext()) for p in root.findall('.//w:p', ns))
    try:
        return path.read_text(encoding='utf-8-sig')
    except UnicodeDecodeError as exc:
        raise ValueError('Text files must use UTF-8 encoding.') from exc


def _ffmpeg(args, timeout=90):
    executable = ffmpeg_path()
    if not executable:
        raise ValueError('Audio/video import requires FFmpeg or: pip install -e ".[media]"')
    process = subprocess.run([executable, '-nostdin', '-hide_banner', '-protocol_whitelist', 'file,pipe', *args],
                             capture_output=True, timeout=timeout)
    if process.returncode:
        detail = process.stderr.decode(errors='replace')[-600:]
        raise ValueError('Could not decode this media file. ' + detail)
    return process


def media_duration(path):
    executable = ffmpeg_path()
    if not executable:
        raise ValueError('Audio/video import requires FFmpeg or the media extra.')
    result = subprocess.run([executable, '-nostdin', '-hide_banner', '-protocol_whitelist', 'file,pipe', '-i', str(path)],
                            capture_output=True, timeout=20)
    match = re.search(r'Duration: (\d+):(\d+):(\d+(?:\.\d+)?)', result.stderr.decode(errors='replace'))
    if not match:
        raise ValueError('Could not read the duration of this media file.')
    h, m, s = map(float, match.groups())
    duration = h * 3600 + m * 60 + s
    if not 0 < duration <= MAX_DURATION:
        raise ValueError(f'Import clips up to {MAX_DURATION // 60} minutes. Split longer recordings first.')
    return duration


@dataclass
class PreparedSource:
    """An original that is saved and indexed but not yet in the database (see SourceLibrary.prepare)."""
    source_id: str = ''
    name: str = ''
    kind: str = 'text'
    digest: str = ''
    size: int = 0
    path: Path | None = None
    description: str = ''
    metadata: dict = field(default_factory=dict)
    chunks: list = field(default_factory=list)
    profile: str = ''
    duplicate: dict | None = None


class SourceLibrary:
    def __init__(self, storage, asset_dir, embedder=None):
        self.storage = storage
        self.conn = storage.conn
        self.asset_dir = Path(asset_dir).resolve()
        self.asset_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        self.conn.executescript('''
            CREATE TABLE IF NOT EXISTS sources (
                id TEXT PRIMARY KEY, name TEXT NOT NULL, kind TEXT NOT NULL,
                mime TEXT NOT NULL, sha256 TEXT NOT NULL UNIQUE, bytes INTEGER,
                created_ts TEXT, path TEXT, description TEXT, metadata TEXT,
                crystal_id INTEGER
            );
            CREATE TABLE IF NOT EXISTS source_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
                ordinal INTEGER, text TEXT NOT NULL, media_path TEXT,
                media_kind TEXT, start_seconds REAL, end_seconds REAL,
                embedding_json TEXT, profile TEXT
            );
            CREATE INDEX IF NOT EXISTS source_chunks_source ON source_chunks(source_id);
        ''')
        self.conn.commit()

    def ingest(self, name, data, progress=None, metadata=None):
        """Store and index one original. `metadata` is merged into the source's metadata
        (it may override the default 'coverage' description).

        This is prepare() followed by commit(). The server runs prepare(), which is slow
        (extraction and embedding), without its global lock so chat and search stay
        responsive, and takes the lock only for commit(), one short transaction.
        """
        return self.commit(self.prepare(name, data, progress, metadata))

    def prepare(self, name, data, progress=None, metadata=None):
        """Validate, save the original, extract, and embed, without writing to the database.

        Returns a PreparedSource for commit(). If identical bytes are already stored, it
        carries that source as `duplicate` and nothing is extracted or embedded. On
        failure the saved original is removed.
        """
        progress = progress or (lambda message: None)
        extra_metadata = dict(metadata or {})
        embedder = self.embedder  # one embedder (and profile) for every chunk of this source
        name = Path(name.replace('\\', '/')).name[:200]
        extension = Path(name).suffix.lower()
        if not data or len(data) > MAX_BYTES:
            raise ValueError('Choose a nonempty file smaller than 64 MB.')
        if extension not in TEXT_EXT | IMAGE_EXT | VIDEO_EXT | AUDIO_EXT | {'.pdf', '.docx'}:
            raise ValueError('Unsupported file type. Use text, PDF, DOCX, PNG/JPEG/WebP, audio, or video.')
        digest = hashlib.sha256(data).hexdigest()
        existing = self._stored_digest(digest)
        if existing:
            return PreparedSource(duplicate={**self.get(existing), 'duplicate': True})
        source_id = uuid.uuid4().hex
        folder = self.asset_dir / source_id
        folder.mkdir()
        path = folder / ('original' + extension)
        path.write_bytes(data)
        chunks = []
        metadata = {}
        kind = 'text'
        try:
            progress('Reading ' + name)
            if extension in IMAGE_EXT:
                kind = 'image'
                chunks = [{'text': f'Image: {name}', 'media_path': str(path), 'media_kind': 'image'}]
                metadata['coverage'] = 'Original image indexed directly; visual answers require a vision chat model.'
            elif extension in AUDIO_EXT | VIDEO_EXT:
                kind = 'video' if extension in VIDEO_EXT else 'audio'
                duration = media_duration(path)
                metadata['duration'] = duration
                if kind == 'video':
                    # Evenly sample the full clip. These are visual snapshots,
                    # not a claim that every frame or its audio was indexed.
                    count = min(12, max(2, math.ceil(duration / 5)))
                    last_frame = duration - min(0.15, duration / 4)
                    times = [last_frame * i / (count - 1) for i in range(count)]
                    for i, at in enumerate(times):
                        output = folder / f'frame-{i:03d}.jpg'
                        _ffmpeg(['-ss', str(at), '-i', str(path), '-frames:v', '1',
                                 '-vf', 'scale=768:768:force_original_aspect_ratio=decrease',
                                 '-y', str(output)])
                        if not output.exists():
                            raise ValueError('A video frame could not be decoded.')
                        chunks.append({'text': f'Video frame from {name} at {at:.1f} seconds.',
                                       'media_path': str(output), 'media_kind': 'image', 'start_seconds': at})
                    metadata['coverage'] = f'{count} sampled frames across {duration:.1f}s. Audio is not indexed; brief events may be missed.'
                else:
                    for i, at in enumerate(range(0, math.ceil(duration), 30)):
                        output = folder / f'audio-{i:03d}.wav'
                        _ffmpeg(['-ss', str(at), '-i', str(path), '-t', '30', '-vn',
                                 '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le', '-y', str(output)])
                        chunks.append({'text': f'Audio excerpt from {name}, {at}–{min(at + 30, duration):.1f}s. Not transcribed.',
                                       'media_path': str(output), 'media_kind': 'audio',
                                       'start_seconds': at, 'end_seconds': min(at + 30, duration)})
                    metadata['coverage'] = 'Complete recording indexed in 30-second audio segments. Semantic search and playback; no transcript.'
            else:
                text = extract_document(path, extension)
                chunks = [{'text': part} for part in chunk_text(text)]
                metadata['coverage'] = 'Complete extracted text, split into overlapping passages.'
            metadata.update(extra_metadata)
            if kind != 'text' and not embedder:
                raise ValueError('Select a multimodal embedding model, such as EmbeddingGemma 2, before importing media.')
            for i, chunk in enumerate(chunks):
                progress(f'Indexing {name} · {i + 1}/{len(chunks)}')
                chunk['embedding'] = self._embed_chunk(chunk, embedder) if embedder else None
            description = (chunks[0]['text'][:500] if kind == 'text' else metadata['coverage'])
        except Exception:
            shutil.rmtree(folder, ignore_errors=True)
            raise
        return PreparedSource(source_id=source_id, name=name, kind=kind, digest=digest, size=len(data),
                              path=path, description=description, metadata=metadata, chunks=chunks,
                              profile=embedder.profile if embedder else '')

    def commit(self, prepared):
        """Write a prepared source and its passages in one short transaction.

        If identical bytes were committed meanwhile, this copy is discarded and the stored
        source is returned with duplicate=True. On failure nothing is written and the
        saved original is removed.
        """
        if prepared.duplicate:
            return prepared.duplicate
        folder = prepared.path.parent
        try:
            with self.storage._lock, self.conn:
                row = self.conn.execute('SELECT id FROM sources WHERE sha256=?', (prepared.digest,)).fetchone()
                if not row:
                    self.conn.execute('INSERT INTO sources VALUES (?,?,?,?,?,?,?,?,?,?,NULL)',
                                      (prepared.source_id, prepared.name, prepared.kind,
                                       mimetypes.guess_type(prepared.name)[0] or 'application/octet-stream',
                                       prepared.digest, prepared.size, now_iso(), str(prepared.path),
                                       prepared.description, json.dumps(prepared.metadata)))
                    for i, chunk in enumerate(prepared.chunks):
                        self.conn.execute('''INSERT INTO source_chunks
                            (source_id,ordinal,text,media_path,media_kind,start_seconds,end_seconds,embedding_json,profile)
                            VALUES (?,?,?,?,?,?,?,?,?)''',
                            (prepared.source_id, i, chunk['text'], chunk.get('media_path'), chunk.get('media_kind'),
                             chunk.get('start_seconds'), chunk.get('end_seconds'), json.dumps(chunk['embedding']),
                             prepared.profile))
        except Exception:
            shutil.rmtree(folder, ignore_errors=True)
            raise
        if row:
            shutil.rmtree(folder, ignore_errors=True)
            return {**self.get(row['id']), 'duplicate': True}
        return self.get(prepared.source_id)

    def _stored_digest(self, digest):
        """Id of the source with these exact bytes, if any."""
        with self.storage._lock:
            row = self.conn.execute('SELECT id FROM sources WHERE sha256=?', (digest,)).fetchone()
        return row['id'] if row else None

    @staticmethod
    def _embed_chunk(chunk, embedder):
        if chunk.get('media_path'):
            return embedder.embed_media(Path(chunk['media_path']).read_bytes(), chunk['media_kind'])
        return embedder.embed_document(chunk['text'])

    def prepare_reindex(self, embedder):
        updates = []
        for row in self.conn.execute('SELECT * FROM source_chunks').fetchall():
            vector = self._embed_chunk(dict(row), embedder)
            updates.append(('source_chunks', 'embedding_json', row['id'], json.dumps(vector)))
        return updates

    def find_by_metadata(self, key, value):
        """The source whose metadata has key == value (e.g. a YouTube video_id), or None."""
        if not re.fullmatch(r'[a-z_]+', key):
            raise ValueError('Invalid metadata key.')
        row = self.conn.execute('SELECT id FROM sources WHERE json_extract(metadata, ?) = ? LIMIT 1',
                                ('$.' + key, value)).fetchone()
        return self.get(row['id']) if row else None

    def list(self):
        return [self._public(row) for row in self.conn.execute('''SELECT s.*, COUNT(c.id) AS chunks
            FROM sources s LEFT JOIN source_chunks c ON c.source_id=s.id
            GROUP BY s.id ORDER BY s.created_ts DESC''')]

    def get(self, source_id):
        row = self.conn.execute('''SELECT s.*, COUNT(c.id) AS chunks FROM sources s
            LEFT JOIN source_chunks c ON c.source_id=s.id WHERE s.id=? GROUP BY s.id''', (source_id,)).fetchone()
        if not row:
            raise ValueError('Source not found.')
        result = self._public(row)
        result['passages'] = [self._chunk_public(r) for r in self.conn.execute(
            'SELECT * FROM source_chunks WHERE source_id=? ORDER BY ordinal', (source_id,))]
        return result

    def _public(self, row):
        return {k: row[k] for k in ('id', 'name', 'kind', 'mime', 'bytes', 'created_ts', 'description', 'crystal_id', 'chunks')} | {
            'metadata': json.loads(row['metadata']), 'url': f'/api/sources/{row["id"]}/file'}

    @staticmethod
    def _chunk_public(row):
        result = {key: row[key] for key in ('id', 'source_id', 'ordinal', 'text', 'media_kind', 'start_seconds', 'end_seconds')}
        if row['media_path']:
            result['preview_url'] = f'/api/chunks/{row["id"]}/media'
        return result

    def asset(self, source_id=None, chunk_id=None):
        if source_id:
            row = self.conn.execute('SELECT path, mime, name FROM sources WHERE id=?', (source_id,)).fetchone()
        else:
            row = self.conn.execute('SELECT media_path AS path FROM source_chunks WHERE id=?', (chunk_id,)).fetchone()
        if not row or not row['path']:
            raise ValueError('Media not found.')
        path = Path(row['path']).resolve()
        if self.asset_dir not in path.parents or not path.is_file():
            raise ValueError('Media not found.')
        return path

    def search(self, query, limit=8):
        query_vector = self.embedder.embed_query(query) if self.embedder else None
        terms = set(re.findall(r'\w+', query.lower())) - {'the', 'a', 'an', 'my', 'what', 'is', 'in', 'of', 'and', 'to'}
        hits = []
        rows = self.conn.execute('''SELECT c.*, s.name, s.kind FROM source_chunks c
            JOIN sources s ON s.id=c.source_id''').fetchall()
        for row in rows:
            vector = json.loads(row['embedding_json'] or 'null')
            similarity = None
            if query_vector and vector and row['profile'] == self.embedder.profile and len(query_vector) == len(vector):
                similarity = cosine_similarity(query_vector, vector)
            words = set(re.findall(r'\w+', (row['name'] + ' ' + row['text']).lower()))
            lexical = len(terms & words) / max(1, len(terms))
            score = (similarity or 0) + lexical * 0.15
            if (similarity is None or similarity < 0.22) and lexical == 0:
                continue
            hit = self._chunk_public(row)
            hit.update(name=row['name'], kind=row['kind'], score=round(score, 4),
                       similarity=round(similarity, 4) if similarity is not None else None,
                       url=f'/api/sources/{row["source_id"]}/file')
            hits.append(hit)
        hits.sort(key=lambda hit: hit['score'], reverse=True)
        # EmbeddingGemma's unrelated multimodal pairs can still score around
        # 0.6. Avoid presenting every nearest neighbor as useful evidence.
        floor = 0.65 if self.embedder and 'embeddinggemma-2' in self.embedder.model else 0.25
        best = hits[0]['score'] if hits else 0
        hits = [hit for hit in hits if hit['score'] >= best - 0.15
                and (hit['similarity'] is None or hit['similarity'] >= floor
                     or hit['score'] - hit['similarity'] >= 0.10)]
        counts = {}
        result = []
        for hit in hits:
            source_id = hit['source_id']
            if counts.get(source_id, 0) >= 3:
                continue
            counts[source_id] = counts.get(source_id, 0) + 1
            result.append(hit)
            if len(result) == limit:
                break
        return result

    def graph(self):
        sources = self.list()[:100]
        visible_ids = {source['id'] for source in sources}
        vectors = {}
        for row in self.conn.execute('SELECT source_id, embedding_json, profile FROM source_chunks ORDER BY ordinal'):
            if row['source_id'] in visible_ids and self.embedder and row['profile'] == self.embedder.profile:
                vector = json.loads(row['embedding_json'] or 'null')
                if vector:
                    vectors.setdefault(row['source_id'], []).append(vector)
        # Mean vectors summarize each source for a conservative similarity edge.
        centroids = {sid: [sum(v[i] for v in vs) / len(vs) for i in range(len(vs[0]))]
                     for sid, vs in vectors.items()}
        relations = []
        seen = set()
        for source in sources:
            sid = source['id']
            if source['crystal_id']:
                relations.append(dict(source_type='source', source_id=sid, target_type='crystal',
                                      target_id=source['crystal_id'], relation='source_memory', weight=1.0))
            if sid not in centroids:
                continue
            matches = sorted(((cosine_similarity(centroids[sid], v), other)
                              for other, v in centroids.items() if other != sid), reverse=True)[:3]
            for similarity, other in matches:
                pair = tuple(sorted((sid, other)))
                if similarity >= 0.82 and pair not in seen:
                    seen.add(pair)
                    relations.append(dict(source_type='source', source_id=sid, target_type='source',
                                          target_id=other, relation='semantic_similarity', weight=round(similarity, 4)))
        return sources, relations
