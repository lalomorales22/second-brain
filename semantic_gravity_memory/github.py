"""Import a public GitHub repository (or one of its folders) as one searchable source.

The repository archive (up to 10 GB) is streamed over HTTPS into a temporary file (no
git needed) and read from there; no files are extracted, and the archive is deleted after. Documentation and source code become passages that each
name their file and line range, so answers can point at the exact code. Dependencies,
build output, binaries, lock files, and minified code are skipped, and a text budget
keeps embedding time reasonable on a laptop CPU.
"""
from __future__ import annotations

import io
import re
import tarfile
import tempfile
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

from semantic_gravity_memory import downloads

MAX_DOWNLOAD = downloads.MAX_DOWNLOAD  # 10 GB, streamed to a temporary file
MAX_FILE_BYTES = 512 * 1024
TEXT_BUDGET = 800_000  # characters of file text indexed; roughly 300 passages
PASSAGE_CHARS = 2400

NAME = re.compile(r'[A-Za-z0-9_.-]+')
NOT_OWNERS = {'orgs', 'topics', 'settings', 'marketplace', 'explore', 'sponsors', 'features',
              'about', 'login', 'join', 'search', 'collections', 'trending', 'notifications'}
SKIP_DIRS = {'node_modules', 'vendor', 'bower_components', 'dist', 'build', 'out', 'target',
             'coverage', '__pycache__', 'venv', 'env', 'site-packages', 'third_party',
             'Pods', 'DerivedData', 'jspm_packages'}
LOCK_FILES = {'package-lock.json', 'yarn.lock', 'pnpm-lock.yaml', 'npm-shrinkwrap.json',
              'cargo.lock', 'poetry.lock', 'pipfile.lock', 'composer.lock', 'gemfile.lock',
              'go.sum', 'packages.lock.json', 'mix.lock', 'pubspec.lock', 'uv.lock', 'bun.lockb'}
BINARY_EXT = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.ico', '.icns', '.tiff', '.psd', '.svg',
              '.mp3', '.mp4', '.mov', '.wav', '.ogg', '.flac', '.webm', '.mkv', '.avi', '.m4a',
              '.zip', '.gz', '.tgz', '.bz2', '.xz', '.7z', '.rar', '.tar', '.jar', '.war',
              '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', '.woff', '.woff2',
              '.ttf', '.otf', '.eot', '.exe', '.dll', '.so', '.dylib', '.a', '.o', '.obj',
              '.class', '.pyc', '.pyo', '.wasm', '.bin', '.dat', '.db', '.sqlite', '.sqlite3',
              '.pkl', '.pt', '.pth', '.onnx', '.safetensors', '.h5', '.npy', '.npz', '.parquet'}
DATA_EXT = {'.csv', '.tsv', '.jsonl', '.ndjson'}
DOC_EXT = {'.md', '.mdx', '.markdown', '.rst', '.txt', '.adoc'}
MANIFESTS = {'package.json', 'pyproject.toml', 'setup.py', 'setup.cfg', 'requirements.txt',
             'cargo.toml', 'go.mod', 'gemfile', 'pom.xml', 'build.gradle', 'build.gradle.kts',
             'composer.json', 'dockerfile', 'docker-compose.yml', 'docker-compose.yaml',
             'makefile', 'cmakelists.txt', 'deno.json', 'tsconfig.json'}
CONFIG_EXT = {'.json', '.yaml', '.yml', '.toml', '.xml', '.ini', '.cfg', '.lock'}


@dataclass
class RepoRef:
    owner: str
    repo: str
    ref: str = ''  # branch, tag, or commit; empty means the default branch
    path: str = ''  # folder or file inside the repository

    @property
    def slug(self):
        return f'{self.owner}/{self.repo}'

    @property
    def label(self):
        return self.slug + (f'/{self.path}' if self.path else '')


def parse_repo(url):
    """A RepoRef for a GitHub repository, folder, or file link; ValueError for anything else."""
    text = str(url or '').strip()
    ssh = re.fullmatch(r'git@github\.com:([^/\s]+)/([^/\s]+?)(?:\.git)?/?', text)
    if ssh:
        owner, repo, rest = ssh[1], ssh[2], []
    else:
        if not re.match(r'^[a-z][a-z0-9+.-]*://', text, re.I):
            text = 'https://' + text
        parsed = urlparse(text)
        host = (parsed.hostname or '').lower()
        if host.startswith('www.'):
            host = host[4:]
        parts = [unquote(p) for p in parsed.path.split('/') if p]
        if host != 'github.com' or len(parts) < 2:
            raise ValueError('Paste a link to a public GitHub repository, such as https://github.com/owner/name')
        owner, repo, rest = parts[0], parts[1], parts[2:]
    repo = repo[:-4] if repo.endswith('.git') else repo
    if not (NAME.fullmatch(owner) and NAME.fullmatch(repo)) or owner.lower() in NOT_OWNERS or repo in ('.', '..'):
        raise ValueError('Paste a link to a public GitHub repository, such as https://github.com/owner/name')
    ref = path = ''
    if rest:
        if rest[0] not in ('tree', 'blob') or len(rest) < 2:
            raise ValueError('That link points to a page inside the repository. Paste the repository link, or a link to one of its folders.')
        ref, path = rest[1], '/'.join(rest[2:])
        if not re.fullmatch(r'[A-Za-z0-9._/-]+', ref) or '..' in ref.split('/'):
            raise ValueError('That branch or tag name is not supported.')
        if any(part in ('.', '..') for part in path.split('/')):
            raise ValueError('That folder path is not supported.')
    return RepoRef(owner, repo, ref, path.strip('/'))


@dataclass
class RepoFile:
    path: str
    text: str


def skip_reason(path, size):
    """Why a file is left out, or None to include it."""
    parts = path.split('/')
    name = parts[-1].lower()
    ext = '.' + name.rsplit('.', 1)[-1] if '.' in name else ''
    if any(p in SKIP_DIRS for p in parts[:-1]):
        return 'dependencies or build output'
    if any(p.startswith('.') and p != '.github' for p in parts):
        return 'hidden files'
    if name in LOCK_FILES:
        return 'lock files'
    if ext in BINARY_EXT:
        return 'binary or media'
    if name.endswith(('.min.js', '.min.css', '.map', '.bundle.js', '.chunk.js')):
        return 'minified or generated'
    if size > MAX_FILE_BYTES:
        return 'over 512 KB'
    if ext in DATA_EXT and size > 100 * 1024:
        return 'large data files'
    return None


def decode(raw):
    """UTF-8 text, or None for binary content."""
    if b'\x00' in raw[:8000]:
        return None
    try:
        return raw.decode('utf-8-sig')
    except UnicodeDecodeError:
        return None


def looks_minified(text):
    lines = text.count('\n') + 1
    return len(text) > 5000 and len(text) / lines > 300


def priority(path):
    """Lower sorts first: what to keep when the text budget runs out."""
    name = path.rsplit('/', 1)[-1].lower()
    ext = '.' + name.rsplit('.', 1)[-1] if '.' in name else ''
    depth = path.count('/')
    parts = path.lower().split('/')
    if name.startswith('readme'):
        rank = 0
    elif ext in DOC_EXT or parts[0] in ('docs', 'doc', 'documentation'):
        rank = 1
    elif name in MANIFESTS:
        rank = 2
    elif (any(p in ('test', 'tests', '__tests__', 'spec', 'specs') for p in parts[:-1])
          or name.startswith('test_') or '.test.' in name or '.spec.' in name or name.endswith('_test.go')):
        rank = 4
    elif ext in CONFIG_EXT or ext in DATA_EXT:
        rank = 5
    else:
        rank = 3
    return rank, depth, path


def file_passages(path, text):
    """Split one file at line boundaries; each piece names the file and its lines."""
    lines = text.splitlines()
    pieces, buffer, size, start = [], [], 0, 1
    for number, line in enumerate(lines, 1):
        if len(line) > PASSAGE_CHARS:
            line = line[:PASSAGE_CHARS] + ' …'
        if buffer and size + len(line) + 1 > PASSAGE_CHARS:
            pieces.append((start, number - 1, buffer))
            buffer, size, start = [], 0, number
        buffer.append(line)
        size += len(line) + 1
    if buffer:
        pieces.append((start, len(lines), buffer))
    return [f'File: {path} (lines {a}–{b})\n' + '\n'.join(body).rstrip()
            for a, b, body in pieces if '\n'.join(body).strip()]


def _fence(text):
    longest = max((len(run) for run in re.findall(r'`+', text)), default=0)
    return '`' * max(3, longest + 1)


@dataclass
class RepoSnapshot:
    ref: RepoRef
    commit: str
    files: list = field(default_factory=list)  # included RepoFiles, in reading order
    skipped: Counter = field(default_factory=Counter)

    @property
    def key(self):
        return f'{self.ref.slug}:{self.ref.path}@{self.commit or self.ref.ref or "HEAD"}'

    @property
    def url(self):
        base = f'https://github.com/{self.ref.slug}'
        at = self.commit or self.ref.ref
        if not at:
            return base
        return f'{base}/tree/{at}' + (f'/{quote(self.ref.path)}' if self.ref.path else '')

    @property
    def text_chars(self):
        return sum(len(f.text) for f in self.files)

    def summary_lines(self):
        lines = [f'Repository: https://github.com/{self.ref.slug}']
        if self.ref.path:
            lines.append(f'Folder: {self.ref.path}')
        lines.append(f'Branch or tag: {self.ref.ref or "default branch"}'
                     + (f' · commit {self.commit[:7]}' if self.commit else ''))
        lines.append(f'Indexed: {len(self.files)} files, {self.text_chars // 1024} KB of text')
        if self.skipped:
            lines.append('Skipped: ' + ', '.join(f'{count} {reason}' for reason, count in self.skipped.most_common()))
        return lines

    def passages(self):
        """An overview with the file tree, then every file in labelled pieces. Small
        neighbouring pieces share a passage, which keeps embedding time down."""
        head = [f'# {self.ref.label} (GitHub repository)', *self.summary_lines(), '', 'Files:']
        overview = '\n'.join(head)
        listed = 0
        for f in self.files:
            if len(overview) + len(f.path) + 1 > PASSAGE_CHARS:
                break
            overview += '\n' + f.path
            listed += 1
        if listed < len(self.files):
            overview += f'\n… and {len(self.files) - listed} more files'
        result = [overview]
        for f in self.files:
            for piece in file_passages(f.path, f.text):
                if len(result) > 1 and len(result[-1]) + len(piece) + 2 <= PASSAGE_CHARS:
                    result[-1] += '\n\n' + piece
                else:
                    result.append(piece)
        return result

    def document(self):
        """The stored original: a readable Markdown digest of what was indexed."""
        parts = [f'# {self.ref.label}', '', *self.summary_lines(), '', '## Files', '',
                 *(f'- {f.path}' for f in self.files), '']
        for f in self.files:
            fence = _fence(f.text)
            language = f.path.rsplit('.', 1)[-1] if '.' in f.path.rsplit('/', 1)[-1] else ''
            parts += [f'## {f.path}', '', fence + language, f.text.rstrip(), fence, '']
        return '\n'.join(parts)

    def filename(self):
        label = f'{self.ref.owner} · {self.ref.repo}' + (
            ' › ' + ' › '.join(self.ref.path.split('/')) if self.ref.path else '')
        return re.sub(r'[\\/:*?"<>|\x00-\x1f]+', '-', label)[:150] + ' (GitHub).md'

    def metadata(self):
        coverage = (f'GitHub repository snapshot: {len(self.files)} files of docs and code, split into '
                    'passages labelled with file and line numbers.')
        if self.skipped:
            coverage += ' Skipped ' + ', '.join(f'{count} {reason}' for reason, count in self.skipped.most_common()) + '.'
        return {'origin': 'github', 'repo': self.ref.slug, 'ref': self.ref.ref, 'path': self.ref.path,
                'commit': self.commit, 'url': self.url, 'repo_key': self.key,
                'files': len(self.files), 'skipped': dict(self.skipped), 'coverage': coverage}


def download(ref, target, progress=None):
    """Stream the repository's .tar.gz archive from GitHub into `target` (up to 10 GB)."""
    url = f'https://codeload.github.com/{ref.slug}/tar.gz/{quote(ref.ref or "HEAD", safe="/")}'
    try:
        return downloads.stream_to_file(url, target, progress, label=f'Downloading {ref.slug}', limit=MAX_DOWNLOAD)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            where = f' with a branch or tag named "{ref.ref}"' if ref.ref else ''
            raise ValueError(f'GitHub has no public repository {ref.slug}{where}. '
                             'Private repositories are not supported.') from exc
        raise ValueError(f'GitHub returned an error ({exc.code}). Try again in a moment.') from exc
    except urllib.error.URLError as exc:
        raise ValueError(f'Could not reach GitHub: {exc.reason}') from exc


def read_archive(source, ref, budget=None):
    """Pick the files to index from an archive (bytes or a path to a .tar.gz), streaming it
    without extracting anything to disk. Returns a RepoSnapshot."""
    budget = TEXT_BUDGET if budget is None else budget
    candidates, skipped = [], Counter()
    try:
        opened = (tarfile.open(fileobj=io.BytesIO(source), mode='r:gz') if isinstance(source, (bytes, bytearray))
                  else tarfile.open(name=str(source), mode='r:gz'))
        with opened as tar:
            commit = tar.pax_headers.get('comment', '')
            if not re.fullmatch(r'[0-9a-f]{40}', commit or ''):
                commit = ''
            for member in tar:
                if not member.isfile():
                    continue  # folders, links, and devices are never followed or extracted
                parts = member.name.split('/')[1:]  # drop the archive's top folder
                if not parts or any(p in ('', '.', '..') for p in parts):
                    continue
                path = '/'.join(parts)
                if ref.path and not (path == ref.path or path.startswith(ref.path + '/')):
                    continue
                reason = skip_reason(path, member.size)
                if reason:
                    skipped[reason] += 1
                    continue
                text = decode(tar.extractfile(member).read(MAX_FILE_BYTES + 1))
                if text is None:
                    skipped['binary or media'] += 1
                elif looks_minified(text):
                    skipped['minified or generated'] += 1
                elif text.strip():
                    candidates.append(RepoFile(path, text))
    except (tarfile.TarError, EOFError, OSError) as exc:
        raise ValueError('GitHub returned an archive that could not be read.') from exc
    chosen, used = [], 0
    for f in sorted(candidates, key=lambda f: priority(f.path)):
        if used + len(f.text) > budget:
            skipped['over the size budget'] += 1
            continue
        chosen.append(f)
        used += len(f.text)
    if not chosen:
        where = f' in {ref.path}' if ref.path else ''
        raise ValueError(f'No readable docs or code were found{where}.')
    # The top-level README (of the repository or the chosen folder) first, then path order.
    top = ref.path + '/' if ref.path else ''

    def top_readme(path):
        rest = path[len(top):]
        return path.startswith(top) and '/' not in rest and rest.lower().startswith('readme')

    chosen.sort(key=lambda f: (not top_readme(f.path), f.path))
    return RepoSnapshot(ref, commit, chosen, skipped)


def fetch_repository(url_or_ref, progress=None):
    """Download a repository and choose what to index."""
    progress = progress or (lambda message: None)
    ref = url_or_ref if isinstance(url_or_ref, RepoRef) else parse_repo(url_or_ref)
    progress(f'Downloading {ref.label} from GitHub')
    with tempfile.TemporaryDirectory(prefix='second-brain-') as folder:
        archive_path = Path(folder) / 'repository.tar.gz'
        download(ref, archive_path, progress)
        progress(f'Reading {ref.label}')
        return read_archive(archive_path, ref)
