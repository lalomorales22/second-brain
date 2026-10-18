"""Local text and media embeddings using Ollama's /api/embed endpoint."""
from __future__ import annotations

import base64
import json
import math
import re
import urllib.error
import urllib.request
from typing import List

from semantic_gravity_memory.embeddings.base import BaseEmbedder

# Ollama embeds a text in a single batch, so one input is limited to the batch size
# (2,048 tokens by default). Dense text gets there in far fewer characters than prose:
# code, Japanese or Chinese, and especially numbers, which Gemma splits one digit per token.
TOO_LARGE = re.compile(r'too large|exceeds the context|context length|input length', re.I)
MIN_SPLIT_CHARS = 64


class InputTooLarge(RuntimeError):
    """Ollama rejected an input as longer than it can embed in one pass."""


def _split(text):
    """Two halves of a text, cut at a line break (or else a space) near the middle."""
    middle = len(text) // 2
    window = len(text) // 4
    for separator in ('\n', ' '):
        before = text.rfind(separator, middle - window, middle + 1)
        after = text.find(separator, middle, middle + window)
        candidates = [i for i in (before, after) if i > 0]
        if candidates:
            cut = min(candidates, key=lambda i: abs(i - middle)) + 1
            return text[:cut], text[cut:]
    return text[:middle], text[middle:]


def _normalize(vector):
    length = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / length for v in vector]


class OllamaEmbedder(BaseEmbedder):
    def __init__(self, model='all-minilm', base_url='http://localhost:11434/api', timeout=120):
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self._dimension = 0

    @property
    def profile(self):
        # Prefix changes also change the coordinate space and require reindexing.
        return self.model + ('|gemma-search-v1' if 'embeddinggemma' in self.model else '|raw-v1')

    def _request(self, inputs):
        payload = json.dumps({'model': self.model, 'input': inputs, 'truncate': False}).encode()
        req = urllib.request.Request(self.base_url + '/embed', data=payload,
                                     headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                data = json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors='replace')
            if exc.code == 413 or TOO_LARGE.search(detail):
                raise InputTooLarge(f'Ollama embedding failed: {detail}') from exc
            raise RuntimeError(f'Ollama embedding failed: {detail}') from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f'Cannot reach Ollama at {self.base_url}: {exc.reason}') from exc
        vectors = data.get('embeddings', [])
        expected = len(inputs) if isinstance(inputs, list) else 1
        if len(vectors) != expected:
            raise RuntimeError('Ollama returned an incomplete embedding batch.')
        for vector in vectors:
            if not vector or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in vector):
                raise RuntimeError('Ollama returned an invalid embedding.')
            if self._dimension and len(vector) != self._dimension:
                raise RuntimeError('Ollama changed embedding dimensions during this session.')
            self._dimension = len(vector)
        return vectors

    def _embed_text(self, text, prefix=''):
        """Embed text; if it is too long for one pass, embed its halves (each with the
        same task prefix) and combine them weighted by length, so all of it still counts."""
        try:
            return self._request(prefix + text)[0]
        except InputTooLarge:
            if len(text) < MIN_SPLIT_CHARS:
                raise
        left, right = _split(text)
        a, b = self._embed_text(left, prefix), self._embed_text(right, prefix)
        wa, wb = len(left), len(right)
        return _normalize([(x * wa + y * wb) / (wa + wb) for x, y in zip(a, b)])

    def embed(self, text: str) -> List[float]:
        return self._embed_text(text)

    def embed_query(self, text):
        return self._embed_text(text, 'task: search result | query: ' if 'embeddinggemma' in self.model else '')

    def embed_document(self, text):
        return self._embed_text(text, 'title: none | text: ' if 'embeddinggemma' in self.model else '')

    def embed_batch(self, texts):
        if not texts:
            return []
        try:
            return self._request(texts)
        except InputTooLarge:
            return [self.embed(text) for text in texts]

    def embed_media(self, data: bytes, kind: str):
        if kind not in ('image', 'audio'):
            raise ValueError('Use image frames for video embeddings.')
        return self._request({kind: base64.b64encode(data).decode('ascii')})[0]

    def dimension(self):
        return self._dimension
