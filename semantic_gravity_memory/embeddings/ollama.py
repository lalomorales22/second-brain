"""Local text and media embeddings using Ollama's /api/embed endpoint."""
from __future__ import annotations

import base64
import json
import math
import urllib.error
import urllib.request
from typing import List

from semantic_gravity_memory.embeddings.base import BaseEmbedder


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
            raise RuntimeError(f'Ollama embedding failed: {exc.read().decode(errors="replace")}') from exc
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

    def embed(self, text: str) -> List[float]:
        return self._request(text)[0]

    def embed_query(self, text):
        if 'embeddinggemma' in self.model:
            text = 'task: search result | query: ' + text
        return self.embed(text)

    def embed_document(self, text):
        if 'embeddinggemma' in self.model:
            text = 'title: none | text: ' + text
        return self.embed(text)

    def embed_batch(self, texts):
        return self._request(texts) if texts else []

    def embed_media(self, data: bytes, kind: str):
        if kind not in ('image', 'audio'):
            raise ValueError('Use image frames for video embeddings.')
        return self._request({kind: base64.b64encode(data).decode('ascii')})[0]

    def dimension(self):
        return self._dimension
