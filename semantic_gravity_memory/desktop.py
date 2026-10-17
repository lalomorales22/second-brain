"""Open the Second Brain workspace in its own window instead of a browser tab.

The window is a native web view (WebView2 on Windows, WebKit on macOS) provided
by pywebview. The regular HTTP server runs in a background thread on 127.0.0.1
and is shut down when the window closes.
"""
from __future__ import annotations

import argparse
import html
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

from semantic_gravity_memory.api import server as api_server

APP_TITLE = 'Second Brain'
HOST = '127.0.0.1'


def default_db_path():
    """Use <repo>/.data/memory.db from a checkout (same as start.bat), else the package default."""
    root = Path(__file__).resolve().parent.parent
    if (root / 'pyproject.toml').exists():
        return str(root / '.data' / 'memory.db')
    return None


def _url_ok(url, timeout=2):
    try:
        with urllib.request.urlopen(url, timeout=timeout):
            return True
    except Exception:
        return False


def _wait_for(url, seconds):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if _url_ok(url):
            return True
        time.sleep(0.5)
    return False


def ensure_ollama(ollama_url):
    """Start the Ollama desktop app if nothing answers on its port. Best effort."""
    tags = ollama_url.rstrip('/') + '/tags'
    if _url_ok(tags):
        return True
    try:
        if sys.platform == 'win32':
            app = Path(os.environ.get('LOCALAPPDATA', '')) / 'Programs' / 'Ollama' / 'ollama app.exe'
            if not app.exists():
                return False
            flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
            subprocess.Popen([str(app)], creationflags=flags, close_fds=True)
        elif sys.platform == 'darwin':
            if not Path('/Applications/Ollama.app').exists():
                return False
            subprocess.Popen(['open', '-a', 'Ollama'])
        else:
            return False
    except OSError:
        return False
    return _wait_for(tags, 20)


def pick_port(preferred, host=HOST):
    """Use the preferred port when free, otherwise any free port."""
    for port in (preferred, 0):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind((host, port))
            except OSError:
                continue
            return probe.getsockname()[1]
    raise OSError('No free TCP port available on ' + host)


class ServerThread(threading.Thread):
    """Runs the workspace HTTP server; `ready` is set once it is bound or has failed."""

    def __init__(self, **options):
        super().__init__(name='second-brain-server', daemon=True)
        self.options = options
        self.server = None
        self.memory = None
        self.error = None
        self.ready = threading.Event()

    def run(self):
        try:
            self.server, self.memory = api_server.create_server(**self.options)
        except Exception as exc:  # reported to the window by main()
            self.error = exc
            self.ready.set()
            return
        # Closing the window should not wait for an in-flight chat generation.
        self.server.block_on_close = False
        self.ready.set()
        try:
            self.server.serve_forever()
        finally:
            self.server.server_close()
            self.memory.close()

    def stop(self):
        if self.server is not None and self.is_alive():
            self.server.shutdown()
            self.join(timeout=10)


def _show_error(webview, message):
    print(message, file=sys.stderr, flush=True)
    body = ('<body style="font: 15px system-ui; margin: 2em; color: #ddd; background: #111">'
            f'<h2>{APP_TITLE} could not start</h2><pre style="white-space: pre-wrap">'
            f'{html.escape(message)}</pre></body>')
    webview.create_window(APP_TITLE, html=body, width=720, height=360)
    webview.start()


def main(argv=None):
    parser = argparse.ArgumentParser(description='Second Brain — open the workspace in its own window')
    parser.add_argument('--port', type=int, default=api_server.DEFAULT_PORT,
                        help='preferred port; a free one is used when it is busy')
    parser.add_argument('--db', dest='db_path', default=default_db_path(),
                        help='SQLite database path; assets are stored beside it')
    parser.add_argument('--ollama-url', default=api_server.DEFAULT_OLLAMA_URL)
    parser.add_argument('--embed-model')
    parser.add_argument('--chat-model')
    parser.add_argument('--width', type=int, default=1400)
    parser.add_argument('--height', type=int, default=900)
    args = parser.parse_args(argv)

    try:
        import webview
    except ImportError:
        print('pywebview is not installed. Run: pip install -e ".[media,desktop]"', file=sys.stderr)
        return 1

    if not ensure_ollama(args.ollama_url):
        print('Ollama is not running; embeddings and chat will be unavailable until it starts.', flush=True)

    try:
        port = pick_port(args.port)
    except OSError as exc:
        _show_error(webview, str(exc))
        return 1

    thread = ServerThread(port=port, host=HOST, db_path=args.db_path, ollama_url=args.ollama_url,
                          embed_model=args.embed_model, chat_model=args.chat_model)
    thread.start()
    thread.ready.wait(120)
    url = f'http://{HOST}:{port}/'
    if thread.error is not None or thread.server is None or not _wait_for(url + 'api/config', 30):
        _show_error(webview, f'The local server did not start: {thread.error or "timed out"}')
        thread.stop()
        return 1

    # Keep web-view storage (and so UI preferences) next to the database rather than in a throwaway profile.
    storage = None
    if args.db_path:
        storage = str(Path(args.db_path).resolve().parent / 'webview-profile')
        os.makedirs(storage, exist_ok=True)

    webview.create_window(APP_TITLE, url, width=args.width, height=args.height, min_size=(900, 600))
    try:
        webview.start(private_mode=False, storage_path=storage)
    finally:
        thread.stop()
    return 0


if __name__ == '__main__':
    sys.exit(main())
