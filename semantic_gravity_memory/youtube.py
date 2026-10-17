"""Import YouTube captions as searchable transcripts using yt-dlp.

Only the video's details and one caption track are fetched; the video and its
audio are never downloaded. The transcript becomes an ordinary Markdown text
source, so it is chunked, embedded, deduplicated, and cited like any other file.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from html import unescape
from urllib.parse import parse_qs, urlparse

VIDEO_ID = re.compile(r'[A-Za-z0-9_-]{11}')
WATCH_HOSTS = {'youtube.com', 'm.youtube.com', 'music.youtube.com', 'youtube-nocookie.com'}
MAX_TRACK_BYTES = 32 * 1024 * 1024
PARAGRAPH_SECONDS = 60      # start a new timestamped paragraph after a sentence end past this
PARAGRAPH_MAX_SECONDS = 120  # ...and always after this, for captions without punctuation


def available():
    try:
        import yt_dlp  # noqa: F401
    except ImportError:
        return False
    return True


def video_id(url):
    """Return the 11-character id of a single YouTube video link, or raise ValueError."""
    text = str(url or '').strip()
    if not re.match(r'^[a-z][a-z0-9+.-]*://', text, re.I):
        text = 'https://' + text
    parsed = urlparse(text)
    host = (parsed.hostname or '').lower()
    if host.startswith('www.'):
        host = host[4:]
    parts = [p for p in parsed.path.split('/') if p]
    candidate = ''
    if host == 'youtu.be' and parts:
        candidate = parts[0]
    elif host in WATCH_HOSTS:
        if parts[:1] == ['watch']:
            candidate = parse_qs(parsed.query).get('v', [''])[0]
        elif len(parts) >= 2 and parts[0] in ('shorts', 'live', 'embed', 'v'):
            candidate = parts[1]
    if not VIDEO_ID.fullmatch(candidate):
        raise ValueError('Paste a link to a single YouTube video, such as https://www.youtube.com/watch?v=…')
    return candidate


def watch_url(vid):
    return f'https://www.youtube.com/watch?v={vid}'


def clock(seconds):
    seconds = max(0, int(seconds))
    return f'{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}'


def _clean(text):
    return re.sub(r'\s+', ' ', unescape(text)).strip()


def parse_json3(raw):
    """YouTube's json3 caption format -> [(start_seconds, text)]."""
    data = json.loads(raw)
    cues = []
    for event in data.get('events', []):
        text = _clean(''.join(seg.get('utf8', '') for seg in event.get('segs', [])))
        if text:
            cues.append((event.get('tStartMs', 0) / 1000, text))
    return cues


def parse_vtt(text):
    """WebVTT -> [(start_seconds, text)], dropping the rolling repeats auto-captions use."""
    cues, previous = [], ''
    stamp = re.compile(r'(?:(\d+):)?(\d{2}):(\d{2})[.,](\d{3})\s+-->')
    for block in re.split(r'\r?\n\r?\n+', text):
        lines = block.strip().splitlines()
        for i, line in enumerate(lines):
            match = stamp.match(line.strip())
            if not match:
                continue
            h, m, s, ms = match.groups()
            start = int(h or 0) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000
            for body in lines[i + 1:]:
                line_text = _clean(re.sub(r'<[^>]+>', '', body))
                if line_text and line_text != previous:
                    cues.append((start, line_text))
                    previous = line_text
            break
    return cues


def paragraphs(cues):
    """Group caption cues into readable paragraphs, each keeping its start time."""
    result, start, words, last = [], None, [], ''
    for at, text in cues:
        if text == last:
            continue
        last = text
        if start is not None and words:
            span = at - start
            if span >= PARAGRAPH_MAX_SECONDS or (span >= PARAGRAPH_SECONDS and re.search(r'[.?!]["\')\]]?$', words[-1])):
                result.append((start, ' '.join(words)))
                start, words = None, []
        if start is None:
            start = at
        words.append(text)
    if words:
        result.append((start, ' '.join(words)))
    return result


def _formats(table, key):
    """Usable formats for one caption language, best first (json3, then plain VTT)."""
    usable = [f for f in table.get(key) or [] if f.get('url') and str(f.get('protocol', 'https')).startswith('http')]
    for ext in ('json3', 'vtt'):
        for fmt in usable:
            if fmt.get('ext') == ext:
                return fmt
    return None


def pick_track(info, language=None):
    """Choose a caption track: (language_key, format, 'manual'|'auto-generated'|'auto-translated').

    Order: human captions in the requested or spoken language, then English human
    captions, then the speech-recognition track in that language, then English.
    """
    manual = {k: v for k, v in (info.get('subtitles') or {}).items() if k != 'live_chat'}
    auto = info.get('automatic_captions') or {}
    spoken = (info.get('language') or '').strip()
    wanted = (language or spoken or 'en').strip()
    base = wanted.split('-')[0].lower()

    def family(table, prefix):
        return [k for k in table if k.split('-')[0].lower() == prefix and not k.endswith('-orig')]

    for key in [wanted, *family(manual, base), 'en', *family(manual, 'en')]:
        fmt = _formats(manual, key)
        if fmt:
            return key, fmt, 'manual'
    for key in [f'{wanted}-orig', wanted, f'{base}-orig', base, 'en-orig', 'en']:
        fmt = _formats(auto, key)
        if fmt:
            translated = not key.endswith('-orig') and spoken and key.split('-')[0] != spoken.split('-')[0]
            return key, fmt, 'auto-translated' if translated else 'auto-generated'
    for key in manual:
        fmt = _formats(manual, key)
        if fmt:
            return key, fmt, 'manual'
    raise ValueError('This video has no captions to import. YouTube only offers transcripts when the '
                     'uploader adds captions or automatic captions are enabled.')


@dataclass
class Transcript:
    video_id: str
    title: str
    cues: list = field(default_factory=list)
    channel: str = ''
    duration: float | None = None
    upload_date: str = ''
    language: str = ''
    language_name: str = ''
    captions: str = 'manual'

    @property
    def url(self):
        return watch_url(self.video_id)

    @property
    def published(self):
        d = self.upload_date
        return f'{d[:4]}-{d[4:6]}-{d[6:8]}' if re.fullmatch(r'\d{8}', d or '') else ''

    def filename(self):
        safe = re.sub(r'[\\/:*?"<>|\x00-\x1f]+', '-', self.title or '').strip(' .-')[:150]
        return f'{safe or self.video_id} (YouTube).md'

    def document(self):
        label = self.language_name or self.language or 'unknown language'
        lines = [f'# {self.title or self.video_id}', '']
        if self.channel:
            lines.append(f'Channel: {self.channel}')
        if self.published:
            lines.append(f'Published: {self.published}')
        if self.duration:
            lines.append(f'Length: {clock(self.duration)}')
        lines += [f'Video: {self.url}', f'Transcript: {label}, {self.captions} captions from YouTube', '']
        for start, text in paragraphs(self.cues):
            lines += [f'[{clock(start)}] {text}', '']
        return '\n'.join(lines).rstrip() + '\n'

    def metadata(self):
        label = self.language_name or self.language
        coverage = (f'Complete YouTube transcript ({label}, {self.captions} captions), split into overlapping '
                    'passages. Speakers are not labeled.')
        if self.captions != 'manual':
            coverage += ' Automatic captions can mishear names and technical terms.'
        return {'origin': 'youtube', 'video_id': self.video_id, 'url': self.url, 'title': self.title,
                'channel': self.channel, 'duration': self.duration, 'published': self.published,
                'language': self.language, 'captions': self.captions, 'coverage': coverage}


class _QuietLogger:
    """Keep yt-dlp's console output out of the server log; failures surface as exceptions."""
    def debug(self, message):
        pass
    info = warning = error = debug


def _error_text(exc):
    text = re.sub(r'\x1b\[[0-9;]*m', '', str(exc))
    text = re.sub(r'^(ERROR:\s*)?(\[[^\]]+\]\s*)?([A-Za-z0-9_-]{11}:\s*)?', '', text).strip()
    return text or 'YouTube did not return this video.'


def fetch_transcript(url, language=None, progress=None):
    """Fetch the best caption track for a video and return a Transcript."""
    progress = progress or (lambda message: None)
    vid = video_id(url)
    try:
        import yt_dlp
    except ImportError as exc:
        raise ValueError('YouTube import needs yt-dlp: pip install -e ".[youtube]"') from exc
    options = {'skip_download': True, 'quiet': True, 'no_warnings': True, 'noplaylist': True,
               'socket_timeout': 30, 'logger': _QuietLogger()}
    with yt_dlp.YoutubeDL(options) as ydl:
        progress('Reading YouTube video details')
        try:
            info = ydl.extract_info(watch_url(vid), download=False)
        except yt_dlp.utils.DownloadError as exc:
            raise ValueError('YouTube: ' + _error_text(exc)) from exc
        if info.get('live_status') in ('is_live', 'is_upcoming', 'post_live'):
            raise ValueError('This stream has no finished transcript yet. Try again after it has ended and been processed.')
        key, fmt, captions = pick_track(info, language)
        progress(f'Downloading {captions} captions · {info.get("title") or vid}')
        try:
            with ydl.urlopen(fmt['url']) as response:
                raw = response.read(MAX_TRACK_BYTES + 1)
        except Exception as exc:
            raise ValueError('Could not download the caption track: ' + _error_text(exc)) from exc
    if len(raw) > MAX_TRACK_BYTES:
        raise ValueError('This caption track is too large to import.')
    try:
        cues = parse_json3(raw) if fmt.get('ext') == 'json3' else parse_vtt(raw.decode('utf-8', 'replace'))
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError('YouTube returned a caption track that could not be read.') from exc
    if not cues:
        raise ValueError('The caption track for this video is empty.')
    return Transcript(video_id=vid, title=info.get('title') or vid, cues=cues, channel=info.get('channel') or info.get('uploader') or '',
                      duration=info.get('duration'), upload_date=info.get('upload_date') or '',
                      language=key.removesuffix('-orig'), language_name=fmt.get('name') or '', captions=captions)
