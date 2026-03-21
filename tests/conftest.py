"""Shared test fixtures for SkillMind."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from skillmind.config import EmbeddingConfig, SkillMindConfig, StoreConfig
from skillmind.embeddings import EmbeddingEngine
from skillmind.models import Memory, MemorySource, MemoryType
from skillmind.trainer import Trainer


class MockEmbeddingEngine(EmbeddingEngine):
    """Deterministic mock embeddings for testing (no model download needed)."""

    def __init__(self, dimension: int = 384):
        config = EmbeddingConfig(provider="sentence-transformers", dimension=dimension)
        super().__init__(config)
        self._dimension = dimension

    def _load_model(self) -> None:
        pass  # No model needed

    def embed(self, text: str) -> list[float]:
        """Generate a deterministic pseudo-embedding from text hash."""
        import hashlib

        h = hashlib.sha256(text.encode()).digest()
        # Expand hash to fill dimension
        raw = []
        for i in range(self._dimension):
            raw.append(h[i % len(h)] / 255.0)
        # Normalize
        norm = sum(x * x for x in raw) ** 0.5
        return [x / norm for x in raw]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]

    @property
    def dimension(self) -> int:
        return self._dimension


@pytest.fixture
def mock_engine():
    """Deterministic embedding engine (no GPU/model needed)."""
    return MockEmbeddingEngine(dimension=384)


@pytest.fixture
def tmp_dir():
    """Temporary directory for test data."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def chroma_config(tmp_dir):
    """Config for ChromaDB backend in temp dir."""
    return SkillMindConfig(
        data_dir=str(tmp_dir),
        store=StoreConfig(backend="chroma", chroma_path=str(tmp_dir / "chroma")),
    )


@pytest.fixture
def faiss_config(tmp_dir):
    """Config for FAISS backend in temp dir."""
    return SkillMindConfig(
        data_dir=str(tmp_dir),
        store=StoreConfig(backend="faiss", faiss_path=str(tmp_dir / "faiss")),
    )


@pytest.fixture
def sample_memories() -> list[Memory]:
    """A set of diverse test memories."""
    return [
        Memory(
            id="mem-user-1",
            type=MemoryType.USER,
            topic="role",
            title="SEO Freelancer",
            content="Antonio is an SEO freelancer based in Koblenz, Germany.",
            tags=["seo", "freelancer"],
            source=MemorySource.MANUAL,
            confidence=1.0,
        ),
        Memory(
            id="mem-feedback-1",
            type=MemoryType.FEEDBACK,
            topic="pdf_generation",
            title="PDF Quality Standards",
            content="Always use real Umlaute (ä/ö/ü/ß), never ae/oe/ue. Graphics must be 600 DPI minimum.",
            tags=["pdf", "quality", "umlaute"],
            source=MemorySource.CONVERSATION,
            confidence=0.95,
        ),
        Memory(
            id="mem-feedback-2",
            type=MemoryType.FEEDBACK,
            topic="communication",
            title="Use Ich-Form",
            content="Client communication must use Ich-Form (first person singular), never Wir-Form. Solo freelancer.",
            tags=["communication", "german", "style"],
            source=MemorySource.CONVERSATION,
            confidence=0.9,
        ),
        Memory(
            id="mem-project-1",
            type=MemoryType.PROJECT,
            topic="paroc",
            title="Paroc GmbH SEO Project",
            content="Active SEO project for Paroc GmbH. Finnish parent company. Contact: marketing team in Germany.",
            tags=["paroc", "seo", "client"],
            source=MemorySource.MANUAL,
            confidence=0.85,
        ),
        Memory(
            id="mem-reference-1",
            type=MemoryType.REFERENCE,
            topic="notion",
            title="Notion To-Do Databases",
            content="Notion To-Do DBs for clients: Paroc (ID: abc123), X-Bionic (ID: def456).",
            tags=["notion", "todo", "clients"],
            source=MemorySource.MANUAL,
            confidence=1.0,
        ),
        Memory(
            id="mem-skill-1",
            type=MemoryType.SKILL,
            topic="seo",
            title="Content Mapping Workflow",
            content="8-phase content mapping workflow: data collection, 5000+ KWs, portfolio matching, "
            "TF-IDF clustering, org chart, report. Reusable across all SEO clients.",
            tags=["seo", "content", "workflow", "mapping"],
            source=MemorySource.MANUAL,
            confidence=0.9,
        ),
    ]
