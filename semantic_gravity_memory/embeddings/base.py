"""
Semantic Gravity Memory — Abstract Embedder Interface
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List


class BaseEmbedder(ABC):
    """Contract for all embedding providers."""

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Generate an embedding vector for the given text."""
        ...

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts. Default: sequential calls."""
        return [self.embed(t) for t in texts]

    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension, or 0 if unknown until first call."""
        ...
