"""Embedding generation for SkillMind memory system."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import EmbeddingConfig


class EmbeddingEngine:
    """
    Generate embeddings for memory content.

    Supports:
    - sentence-transformers (local, free, default)
    - openai (API-based, requires key)
    """

    def __init__(self, config: EmbeddingConfig | None = None):
        self.config = config or EmbeddingConfig()
        self._model: Any = None
        self._cache: dict[str, list[float]] = {}
        self._cache_path: Path | None = None

    def _load_model(self) -> None:
        """Lazy-load the embedding model."""
        if self._model is not None:
            return

        if self.config.provider == "sentence-transformers":
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.config.model)
        elif self.config.provider == "openai":
            import openai

            self._model = openai.OpenAI()
        else:
            raise ValueError(f"Unknown embedding provider: {self.config.provider}")

    def _cache_key(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        key = self._cache_key(text)
        if key in self._cache:
            return self._cache[key]

        self._load_model()

        if self.config.provider == "sentence-transformers":
            vec = self._model.encode(text, normalize_embeddings=True)
            result = vec.tolist()
        elif self.config.provider == "openai":
            response = self._model.embeddings.create(
                input=text,
                model=self.config.model,
            )
            result = response.data[0].embedding
        else:
            raise ValueError(f"Unknown provider: {self.config.provider}")

        self._cache[key] = result
        return result

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        # Check cache first
        results: list[list[float] | None] = []
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        for i, text in enumerate(texts):
            key = self._cache_key(text)
            if key in self._cache:
                results.append(self._cache[key])
            else:
                results.append(None)
                uncached_indices.append(i)
                uncached_texts.append(text)

        if not uncached_texts:
            return results  # type: ignore

        self._load_model()

        if self.config.provider == "sentence-transformers":
            vecs = self._model.encode(uncached_texts, normalize_embeddings=True)
            new_embeddings = [v.tolist() for v in vecs]
        elif self.config.provider == "openai":
            response = self._model.embeddings.create(
                input=uncached_texts,
                model=self.config.model,
            )
            new_embeddings = [d.embedding for d in response.data]
        else:
            raise ValueError(f"Unknown provider: {self.config.provider}")

        for idx, emb in zip(uncached_indices, new_embeddings):
            results[idx] = emb
            self._cache[self._cache_key(uncached_texts[uncached_indices.index(idx)])] = emb

        return results  # type: ignore

    @property
    def dimension(self) -> int:
        return self.config.dimension

    def save_cache(self, path: Path) -> None:
        """Persist embedding cache to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._cache, f)

    def load_cache(self, path: Path) -> None:
        """Load embedding cache from disk."""
        if path.exists():
            with open(path) as f:
                self._cache = json.load(f)
        self._cache_path = path
