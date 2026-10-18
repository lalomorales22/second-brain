"""Streaming HTTPS downloads to disk, with a size cap, a free-space guard, and progress.

Large downloads (repositories and datasets up to 10 GB) never pass through memory: they
are written to disk in 1 MB blocks, and a partial file is removed if anything fails.
"""
from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path

MAX_DOWNLOAD = 10 * 1024 ** 3  # 10 GB
SPARE_DISK = 1024 ** 3  # always leave at least 1 GB free
BLOCK = 1024 * 1024
USER_AGENT = 'second-brain'


def human(size):
    """1536 -> '1.5 KB'; 3_400_000_000 -> '3.2 GB'."""
    size = float(size)
    for unit in ('bytes', 'KB', 'MB', 'GB'):
        if size < 1024 or unit == 'GB':
            return f'{size:.0f} {unit}' if unit == 'bytes' else f'{size:.1f} {unit}'
        size /= 1024
    return f'{size:.1f} GB'


def free_space(folder):
    folder = Path(folder)
    while not folder.exists() and folder != folder.parent:
        folder = folder.parent
    return shutil.disk_usage(folder).free


def check_space(folder, needed):
    """Raise ValueError unless `needed` bytes fit while keeping SPARE_DISK free."""
    available = free_space(folder)
    if needed + SPARE_DISK > available:
        raise ValueError(f'Not enough disk space: this needs {human(needed)}, and only '
                         f'{human(max(0, available - SPARE_DISK))} can be used.')


def stream_to_file(url, target, progress=None, label='Downloading', limit=MAX_DOWNLOAD,
                   headers=None, timeout=60, done_before=0, grand_total=None):
    """Download `url` into `target`; returns the number of bytes written.

    `done_before` and `grand_total` let a multi-file download report overall progress.
    HTTP errors propagate (as urllib.error.HTTPError) so callers can explain them.
    """
    progress = progress or (lambda message: None)
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={'User-Agent': USER_AGENT, **(headers or {})})
    written = 0
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response, open(target, 'wb') as out:
            length = int(response.headers.get('Content-Length') or 0)
            if length > limit:
                raise ValueError(f'{label} is {human(length)}, over the {human(limit)} limit.')
            if length:
                check_space(target.parent, length)
            last_report = 0
            while True:
                block = response.read(BLOCK)
                if not block:
                    break
                written += len(block)
                if written > limit:
                    raise ValueError(f'{label} is over the {human(limit)} limit.')
                out.write(block)
                if written - last_report >= 8 * BLOCK or written == length:
                    last_report = written
                    if not length and free_space(target.parent) < SPARE_DISK:
                        raise ValueError('Stopped downloading: the disk is almost full.')
                    done = done_before + written
                    total = grand_total or (done_before + length if length else None)
                    progress(f'{label} · {human(done)}' + (f' of {human(total)}' if total else ''))
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    return written
