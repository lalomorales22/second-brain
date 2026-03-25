"""
Semantic Gravity Memory — Ollama Embedder

Talks to a local Ollama instance over HTTP using only urllib (stdlib).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import List, Optional

from semantic_gravity_memory.embeddings.base import BaseEmbedder


class OllamaEmbedder(BaseEmbedder):
    """Generate embeddings via a local Ollama server."""

    def __init__(
        self,
        model: str = "all-minilm",
        base_url: str = "http://localhost:11434/api",
        timeout: int = 120,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._dimension: int = 0

    def embed(self, text: str) -> List[float]:
        payload = json.dumps({"model": self.model, "input": text}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/embed",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama HTTP {e.code}: {body}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Cannot reach Ollama at {self.base_url}: {e}") from e

        embeddings = data.get("embeddings", [])
        if not embeddings:
            return []
        vec = embeddings[0]
        if vec and self._dimension == 0:
            self._dimension = len(vec)
        return vec

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        # Ollama /api/embed supports a list input natively
        payload = json.dumps({"model": self.model, "input": texts}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/embed",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception:
            # Fallback to sequential if batch fails
            return [self.embed(t) for t in texts]

        embeddings = data.get("embeddings", [])
        if embeddings and embeddings[0] and self._dimension == 0:
            self._dimension = len(embeddings[0])
        return embeddings

    def dimension(self) -> int:
        return self._dimension
