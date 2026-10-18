"""Import a Hugging Face dataset (up to 10 GB) as one source.

Every file of the dataset (or of one folder of it) is downloaded and kept with the
source. Search indexes a representative sample, because embedding gigabytes on a laptop
CPU isn't practical: the dataset card, its splits and columns, sample rows as text, and
sample images and audio clips, which the multimodal embedding model reads natively.

Rows come from Hugging Face's dataset viewer API, which reads Parquet and other formats
without extra libraries; CSV, JSON Lines, JSON, and text files are sampled locally when
the viewer has no rows. Gated or private datasets need an HF_TOKEN environment variable.
"""
from __future__ import annotations

import csv
import io
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote, unquote, urlencode, urlparse

from semantic_gravity_memory import downloads
from semantic_gravity_memory.sources import AUDIO_EXT, IMAGE_EXT, _ffmpeg, chunk_text

HUB = 'https://huggingface.co'
VIEWER = 'https://datasets-server.huggingface.co'
MAX_DOWNLOAD = downloads.MAX_DOWNLOAD
TEXT_BUDGET = 300_000  # characters of card and rows to index
PASSAGE_CHARS = 2400
CELL_CHARS = 600
ROWS_PER_SPLIT = 200
MAX_SPLITS = 6
MAX_IMAGES = 24
MAX_AUDIO = 12
MEDIA_BYTES = 25 * 1024 * 1024
NAME = re.compile(r'[A-Za-z0-9_.-]+')
TABLE_EXT = {'.csv', '.tsv', '.jsonl', '.ndjson', '.json', '.txt'}


@dataclass
class DatasetRef:
    repo_id: str  # "owner/name" (or a legacy single name)
    revision: str = ''
    path: str = ''

    @property
    def label(self):
        return self.repo_id + (f'/{self.path}' if self.path else '')


def parse_dataset(url):
    """A DatasetRef for a Hugging Face dataset link; ValueError for anything else."""
    text = str(url or '').strip()
    if not re.match(r'^[a-z][a-z0-9+.-]*://', text, re.I):
        text = 'https://' + text
    parsed = urlparse(text)
    host = (parsed.hostname or '').lower()
    if host.startswith('www.'):
        host = host[4:]
    parts = [unquote(p) for p in parsed.path.split('/') if p]
    if host not in ('huggingface.co', 'hf.co'):
        raise ValueError('Paste a link to a Hugging Face dataset, such as https://huggingface.co/datasets/owner/name')
    if not parts or parts[0] != 'datasets':
        raise ValueError('That link is a Hugging Face model or Space. Only datasets can be imported: '
                         'https://huggingface.co/datasets/owner/name')
    parts = parts[1:]
    if not parts or not NAME.fullmatch(parts[0]):
        raise ValueError('Paste a link to a Hugging Face dataset, such as https://huggingface.co/datasets/owner/name')
    if len(parts) >= 2 and NAME.fullmatch(parts[1]) and parts[1] not in ('tree', 'blob', 'resolve', 'viewer'):
        repo_id, rest = f'{parts[0]}/{parts[1]}', parts[2:]
    else:
        repo_id, rest = parts[0], parts[1:]
    revision = path = ''
    if rest and rest[0] in ('tree', 'blob', 'resolve') and len(rest) >= 2:
        revision, path = rest[1], '/'.join(rest[2:])
        if not re.fullmatch(r'[A-Za-z0-9._-]+', revision):
            raise ValueError('That branch name is not supported.')
        if any(part in ('.', '..') for part in path.split('/')):
            raise ValueError('That folder path is not supported.')
    elif rest and rest[0] not in ('viewer',):
        raise ValueError('That link points to a page inside the dataset. Paste the dataset link or one of its folders.')
    return DatasetRef(repo_id, revision, path.strip('/'))


def _headers():
    token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
    return {'Authorization': f'Bearer {token}'} if token else {}


def _explain(exc, ref):
    if isinstance(exc, urllib.error.HTTPError):
        if exc.code in (401, 403):
            return ValueError(f'{ref.repo_id} is gated or private. Accept its terms on huggingface.co, set an '
                              'HF_TOKEN environment variable with your access token, and restart the app.')
        if exc.code == 404:
            return ValueError(f'Hugging Face has no dataset {ref.repo_id}'
                              + (f' at revision "{ref.revision}"' if ref.revision else '') + '.')
        return ValueError(f'Hugging Face returned an error ({exc.code}). Try again in a moment.')
    return ValueError(f'Could not reach Hugging Face: {getattr(exc, "reason", exc)}')


def get_json(url, timeout=60):
    request = urllib.request.Request(url, headers={'User-Agent': downloads.USER_AGENT, **_headers()})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        link = response.headers.get('Link') or ''
        return json.load(response), link


def fetch_info(ref):
    """Dataset metadata at the requested revision (including its exact commit sha)."""
    url = f'{HUB}/api/datasets/{ref.repo_id}' + (f'/revision/{quote(ref.revision)}' if ref.revision else '')
    try:
        info, _ = get_json(url)
    except (urllib.error.URLError, OSError) as exc:
        raise _explain(exc, ref) from exc
    if not re.fullmatch(r'[0-9a-f]{40}', info.get('sha') or ''):
        raise ValueError(f'Hugging Face did not report a revision for {ref.repo_id}.')
    return info


def list_files(ref, sha):
    """Every file under the dataset (or folder) at `sha`: [(path, size)], following pages."""
    url = f'{HUB}/api/datasets/{ref.repo_id}/tree/{sha}' + (f'/{quote(ref.path)}' if ref.path else '') + '?recursive=true'
    files = []
    try:
        while url:
            entries, link = get_json(url)
            for entry in entries:
                path = entry.get('path') or ''
                if entry.get('type') == 'file' and path != '.gitattributes' and not any(
                        p in ('', '.', '..') for p in path.split('/')):
                    files.append((path, int((entry.get('lfs') or {}).get('size') or entry.get('size') or 0)))
            match = re.search(r'<([^>]+)>;\s*rel="next"', link)
            url = match[1] if match else None
    except (urllib.error.URLError, OSError) as exc:
        raise _explain(exc, ref) from exc
    if not files:
        raise ValueError(f'No files were found in {ref.label}.')
    return files


def _cell(value, kind, names=None):
    """A readable text version of one cell."""
    if value is None:
        return ''
    if kind == 'ClassLabel' and isinstance(value, int) and names and 0 <= value < len(names):
        return str(names[value])
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    text = re.sub(r'\s+', ' ', text).strip()
    return text if len(text) <= CELL_CHARS else text[:CELL_CHARS] + ' …'


@dataclass
class DatasetSnapshot:
    ref: DatasetRef
    sha: str
    info: dict
    files: list  # [(path, size)]
    card: str = ''
    splits: list = field(default_factory=list)  # [{'config', 'split', 'rows'}]
    columns: list = field(default_factory=list)  # ['name (type)']
    row_lines: list = field(default_factory=list)  # [(heading, line)]
    media: list = field(default_factory=list)  # chunk dicts with media_path
    notes: list = field(default_factory=list)
    viewer_rows: bool = False  # rows came from the dataset viewer API

    @property
    def total_bytes(self):
        return sum(size for _, size in self.files)

    @property
    def key(self):
        return f'{self.ref.repo_id}:{self.ref.path}@{self.sha}'

    @property
    def url(self):
        return f'{HUB}/datasets/{self.ref.repo_id}' + (f'/tree/{self.sha}/{quote(self.ref.path)}' if self.ref.path else '')

    def summary_lines(self):
        card = self.info.get('cardData') or {}

        def listed(key):
            value = card.get(key)
            return ', '.join(map(str, value)) if isinstance(value, list) else (str(value) if value else '')
        lines = [f'Dataset: {HUB}/datasets/{self.ref.repo_id}']
        if self.ref.path:
            lines.append(f'Folder: {self.ref.path}')
        lines.append(f'Revision: {self.sha[:7]}')
        for label, key in (('Name', 'pretty_name'), ('License', 'license'), ('Tasks', 'task_categories'),
                           ('Languages', 'language'), ('Size category', 'size_categories')):
            if listed(key):
                lines.append(f'{label}: {listed(key)}')
        lines.append(f'Files: {len(self.files)}, {downloads.human(self.total_bytes)} (all kept locally)')
        if self.splits:
            lines.append('Splits: ' + ', '.join(
                f'{s["config"]}/{s["split"]}' + (f' ({s["rows"]:,} rows)' if s.get('rows') else '') for s in self.splits))
        if self.columns:
            lines.append('Columns: ' + ', '.join(self.columns))
        return lines

    def overview(self):
        text = '\n'.join([f'# {self.ref.label} (Hugging Face dataset)', *self.summary_lines(), *self.notes, '', 'Files:'])
        listed = 0
        for path, size in self.files:
            line = f'\n{path} ({downloads.human(size)})'
            if len(text) + len(line) > PASSAGE_CHARS:
                break
            text += line
            listed += 1
        if listed < len(self.files):
            text += f'\n… and {len(self.files) - listed} more files'
        return text

    def passages(self):
        """Text passages: the overview, the card, and grouped sample rows."""
        result = [self.overview()]
        if self.card.strip():
            result += [f'Dataset card · {self.ref.repo_id}\n{piece}' for piece in chunk_text(self.card, size=PASSAGE_CHARS)]
        current_heading, buffer = None, ''
        for heading, line in self.row_lines:
            if heading != current_heading or len(buffer) + len(line) + 1 > PASSAGE_CHARS:
                if buffer:
                    result.append(buffer)
                current_heading, buffer = heading, heading
            buffer += '\n' + line
        if buffer:
            result.append(buffer)
        return result

    def chunks(self):
        return [{'text': p} for p in self.passages()] + self.media

    def document(self):
        parts = [f'# {self.ref.label}', '', *self.summary_lines(), *self.notes, '', '## Files', '',
                 *(f'- {path} ({downloads.human(size)})' for path, size in self.files), '']
        if self.card.strip():
            parts += ['## Dataset card', '', self.card.strip(), '']
        if self.row_lines:
            parts += ['## Sample rows', '']
            heading = None
            for h, line in self.row_lines:
                if h != heading:
                    parts += ['', f'### {h}', '']
                    heading = h
                parts.append(f'- {line}')
        return '\n'.join(parts) + '\n'

    def filename(self):
        label = self.ref.repo_id.replace('/', ' · ') + (' › ' + ' › '.join(self.ref.path.split('/')) if self.ref.path else '')
        return re.sub(r'[\\/:*?"<>|\x00-\x1f]+', '-', label)[:150] + ' (Hugging Face).md'

    def metadata(self):
        images = sum(1 for m in self.media if m['media_kind'] == 'image')
        audio = sum(1 for m in self.media if m['media_kind'] == 'audio')
        sampled = len(self.row_lines)

        def count(n, noun):
            return f'{n:,} {noun}' + ('' if n == 1 else 's')
        covered = ['the dataset overview'] + (['its card'] if self.card.strip() else []) + [
            count(n, noun) for n, noun in ((sampled, 'sample row'), (images, 'image'), (audio, 'audio clip')) if n]
        coverage = (f'Hugging Face dataset: all {count(len(self.files), "file")} ({downloads.human(self.total_bytes)}) '
                    f'are stored locally. Search covers {", ".join(covered[:-1]) + " and " + covered[-1] if len(covered) > 1 else covered[0]}'
                    '; the rest of the data is kept but not indexed.')
        return {'origin': 'huggingface', 'dataset': self.ref.repo_id, 'revision': self.ref.revision, 'sha': self.sha,
                'path': self.ref.path, 'url': self.url, 'dataset_key': self.key, 'files': len(self.files),
                'bytes': self.total_bytes, 'rows_sampled': sampled, 'images': images, 'audio': audio,
                'coverage': coverage}


def _strip_front_matter(text):
    return re.sub(r'\A---\s*\n.*?\n---\s*\n', '', text, count=1, flags=re.S)


def download_files(snapshot, folder, progress):
    """Download every file into folder/files, keeping the dataset's own paths."""
    total = snapshot.total_bytes
    if total > MAX_DOWNLOAD:
        raise ValueError(f'{snapshot.ref.label} is {downloads.human(total)}, over the 10 GB limit. '
                         f'Paste a link to one of its folders instead (…/tree/main/<folder>).')
    downloads.check_space(folder, total)
    done = 0
    for number, (path, size) in enumerate(snapshot.files, 1):
        url = f'{HUB}/datasets/{snapshot.ref.repo_id}/resolve/{snapshot.sha}/{quote(path)}'
        target = Path(folder) / 'files' / Path(*path.split('/'))
        try:
            done += downloads.stream_to_file(
                url, target, progress, label=f'Downloading {snapshot.ref.repo_id} · file {number}/{len(snapshot.files)}',
                headers=_headers(), done_before=done, grand_total=total)
        except (urllib.error.URLError, OSError) as exc:
            raise _explain(exc, snapshot.ref) from exc


def sample_rows_from_viewer(snapshot, folder, progress, budget):
    """Sample rows, images, and audio through the dataset viewer API. Returns False if the
    viewer has nothing for this dataset."""
    repo = snapshot.ref.repo_id
    try:
        listing, _ = get_json(f'{VIEWER}/splits?' + urlencode({'dataset': repo}))
    except (urllib.error.URLError, OSError, ValueError):
        return False
    splits = listing.get('splits') or []
    if not splits:
        return False
    used = 0
    images = audio = 0
    for entry in splits[:MAX_SPLITS]:
        config, split = entry['config'], entry['split']
        record = {'config': config, 'split': split, 'rows': None}
        snapshot.splits.append(record)
        heading = f'Rows from {repo} · {config}/{split}'
        for offset in range(0, ROWS_PER_SPLIT, 100):
            if used >= budget:
                break
            progress(f'Sampling rows · {config}/{split}')
            query = urlencode({'dataset': repo, 'config': config, 'split': split, 'offset': offset, 'length': 100})
            try:
                page, _ = get_json(f'{VIEWER}/rows?{query}', timeout=90)
            except (urllib.error.URLError, OSError, ValueError):
                break
            features = {f['name']: f.get('type') or {} for f in page.get('features') or []}
            record['rows'] = page.get('num_rows_total')
            if not snapshot.columns:
                snapshot.columns = [f'{name} ({_type_name(kind)})' for name, kind in features.items()]
            rows = page.get('rows') or []
            for item in rows:
                index, row = item.get('row_idx'), item.get('row') or {}
                text_parts, media_cells = [], []
                for name, value in row.items():
                    kind = (features.get(name) or {}).get('_type')
                    if kind == 'Image' and isinstance(value, dict) and value.get('src'):
                        media_cells.append(('image', value['src']))
                    elif kind == 'Audio' and isinstance(value, list) and value and value[0].get('src'):
                        media_cells.append(('audio', value[0]['src']))
                    else:
                        cell = _cell(value, kind, (features.get(name) or {}).get('names'))
                        if cell:
                            text_parts.append(f'{name}: {cell}')
                line = f'Row {index} · ' + ' · '.join(text_parts) if text_parts else f'Row {index}'
                if text_parts and used < budget:
                    snapshot.row_lines.append((heading, line))
                    used += len(line)
                described = ' · '.join(text_parts)[:300]
                for media_kind, src in media_cells:
                    if media_kind == 'image' and images < MAX_IMAGES:
                        path = _fetch_media(src, Path(folder) / 'samples' / f'{config}-{split}-{index}-image', 'image')
                        if path:
                            images += 1
                            snapshot.media.append({'text': f'Image from {repo}, {config}/{split} row {index}.'
                                                   + (f' {described}' if described else ''),
                                                   'media_path': str(path), 'media_kind': 'image'})
                    elif media_kind == 'audio' and audio < MAX_AUDIO:
                        path = _fetch_media(src, Path(folder) / 'samples' / f'{config}-{split}-{index}-audio', 'audio')
                        if path:
                            audio += 1
                            snapshot.media.append({'text': f'Audio from {repo}, {config}/{split} row {index}, first 30 seconds.'
                                                   + (f' {described}' if described else ''),
                                                   'media_path': str(path), 'media_kind': 'audio',
                                                   'start_seconds': 0, 'end_seconds': 30})
            if len(rows) < 100:
                break
    return bool(snapshot.row_lines or snapshot.media)


def _type_name(kind):
    name = kind.get('_type') or 'value'
    if name == 'Value':
        return kind.get('dtype', 'value')
    if name == 'ClassLabel':
        return 'class: ' + ', '.join(map(str, (kind.get('names') or [])[:6]))
    return name.lower()


def _fetch_media(url, stem, kind):
    """Download one sample image or audio clip next to the source. Returns its path or None."""
    raw = Path(str(stem) + '.download')
    try:
        downloads.stream_to_file(url, raw, limit=MEDIA_BYTES, headers=_headers(), timeout=60)
    except (urllib.error.URLError, OSError, ValueError):
        return None
    return _finish_media(raw, stem, kind)


def _finish_media(raw, stem, kind):
    """Images keep their format; audio becomes 30 seconds of 16 kHz mono WAV."""
    data = raw.read_bytes()[:16]
    try:
        if kind == 'image':
            ext = ('.png' if data.startswith(b'\x89PNG') else '.webp' if data[8:12] == b'WEBP'
                   else '.jpg' if data.startswith(b'\xff\xd8') else None)
            if not ext:
                return None
            target = Path(str(stem) + ext)
            raw.replace(target)
            return target
        target = Path(str(stem) + '.wav')
        _ffmpeg(['-i', str(raw), '-t', '30', '-vn', '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le', '-y', str(target)])
        return target if target.exists() and target.stat().st_size > 1000 else None
    except Exception:
        return None
    finally:
        raw.unlink(missing_ok=True)


def sample_local_files(snapshot, folder, progress, budget):
    """Fallback and extra media: read text tables and media files that were downloaded."""
    root = Path(folder) / 'files'
    used = sum(len(line) for _, line in snapshot.row_lines)
    # Local media only when the viewer supplied none, so the same picture isn't indexed twice.
    images = MAX_IMAGES if any(m['media_kind'] == 'image' for m in snapshot.media) else 0
    audio = MAX_AUDIO if any(m['media_kind'] == 'audio' for m in snapshot.media) else 0
    for path, size in snapshot.files:
        local = root / Path(*path.split('/'))
        ext = local.suffix.lower()
        if not local.exists():
            continue
        if ext in IMAGE_EXT and images < MAX_IMAGES and size <= MEDIA_BYTES:
            images += 1
            snapshot.media.append({'text': f'Image file {path} from {snapshot.ref.repo_id}.',
                                   'media_path': str(local), 'media_kind': 'image'})
        elif ext in AUDIO_EXT and audio < MAX_AUDIO:
            clip = _finish_audio_copy(local, Path(folder) / 'samples' / f'file-{audio}')
            if clip:
                audio += 1
                snapshot.media.append({'text': f'Audio file {path} from {snapshot.ref.repo_id}, first 30 seconds.',
                                       'media_path': str(clip), 'media_kind': 'audio',
                                       'start_seconds': 0, 'end_seconds': 30})
        elif ext in TABLE_EXT and used < budget and path.lower() != 'readme.md' and not snapshot.viewer_rows:
            progress(f'Sampling {path}')
            for line in _table_lines(local, ext):
                if used >= budget:
                    break
                snapshot.row_lines.append((f'Rows from {snapshot.ref.repo_id} · {path}', line))
                used += len(line)


def _finish_audio_copy(local, stem):
    Path(stem).parent.mkdir(parents=True, exist_ok=True)
    target = Path(str(stem) + '.wav')
    try:
        _ffmpeg(['-i', str(local), '-t', '30', '-vn', '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le', '-y', str(target)])
    except Exception:
        return None
    return target if target.exists() and target.stat().st_size > 1000 else None


def _table_lines(path, ext, limit=ROWS_PER_SPLIT):
    """Readable lines for the first rows of a local CSV, TSV, JSON Lines, JSON, or text file."""
    try:
        with open(path, encoding='utf-8-sig', errors='replace') as handle:
            if ext in ('.csv', '.tsv'):
                reader = csv.reader(handle, delimiter='\t' if ext == '.tsv' else ',')
                header = next(reader, None) or []
                for number, row in enumerate(reader):
                    if number >= limit:
                        break
                    cells = [f'{h}: {_cell(v, "Value")}' for h, v in zip(header, row) if v]
                    if cells:
                        yield f'Row {number} · ' + ' · '.join(cells)
            elif ext in ('.jsonl', '.ndjson'):
                for number, line in enumerate(handle):
                    if number >= limit:
                        break
                    try:
                        item = json.loads(line)
                    except ValueError:
                        continue
                    yield f'Row {number} · ' + _record(item)
            elif ext == '.json':
                text = handle.read(5_000_000)
                try:
                    data = json.loads(text)
                except ValueError:
                    return
                if isinstance(data, list):
                    items = data
                elif isinstance(data, dict):  # often {"data": [...]} or {"train": [...]}
                    items = next((v for v in data.values() if isinstance(v, list)), [data])
                else:
                    items = [data]
                for number, item in enumerate(items[:limit]):
                    yield f'Row {number} · ' + _record(item)
            else:
                text = handle.read(PASSAGE_CHARS * 20)
                for number, piece in enumerate(chunk_text(text, size=PASSAGE_CHARS - 200)):
                    yield f'Part {number + 1} · {piece}'
    except (OSError, ValueError, csv.Error, StopIteration):
        return


def _record(item):
    if isinstance(item, dict):
        return ' · '.join(f'{k}: {_cell(v, "Value")}' for k, v in item.items() if v not in (None, ''))
    return _cell(item, 'Value')


def build(ref, info, folder, progress=None, budget=None):
    """Download the dataset into `folder`, then sample what to index. Returns a DatasetSnapshot."""
    progress = progress or (lambda message: None)
    budget = TEXT_BUDGET if budget is None else budget
    sha = info['sha']
    progress(f'Listing files in {ref.label}')
    snapshot = DatasetSnapshot(ref, sha, info, list_files(ref, sha))
    download_files(snapshot, folder, progress)
    card_file = Path(folder) / 'files' / 'README.md'
    if not ref.path and card_file.exists():
        snapshot.card = _strip_front_matter(card_file.read_text(encoding='utf-8', errors='replace'))[:100_000]
    viewer_rows = False
    if not ref.path:  # the viewer describes the whole dataset, not one folder
        progress(f'Sampling rows from {ref.repo_id}')
        viewer_rows = sample_rows_from_viewer(snapshot, folder, progress, budget)
    snapshot.viewer_rows = viewer_rows
    if not viewer_rows:
        snapshot.notes.append('Rows were sampled from the downloaded files.')
    sample_local_files(snapshot, folder, progress, budget)
    return snapshot
