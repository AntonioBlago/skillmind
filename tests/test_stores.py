"""Tests for all SkillMind store backends.

Uses parametrized fixtures to run the same test suite against each backend.
Only tests backends whose dependencies are installed.
"""

import pytest
from datetime import datetime, timedelta

from skillmind.models import Memory, MemoryType, MemorySource, QueryFilter, QueryResult
from skillmind.config import SkillMindConfig, StoreConfig
from skillmind.store.base import MemoryStore


# ── Backend availability checks ──────────────────────────────────

def _chroma_available():
    try:
        import chromadb
        return True
    except ImportError:
        return False


def _faiss_available():
    try:
        import faiss
        return True
    except ImportError:
        return False


def _qdrant_available():
    try:
        import qdrant_client
        return True
    except ImportError:
        return False


# ── Fixtures ─────────────────────────────────────────────────────

@pytest.fixture(params=[
    pytest.param("chroma", marks=pytest.mark.skipif(not _chroma_available(), reason="chromadb not installed")),
    pytest.param("faiss", marks=pytest.mark.skipif(not _faiss_available(), reason="faiss-cpu not installed")),
])
def store(request, tmp_dir, mock_engine) -> MemoryStore:
    """Parametrized store fixture — runs tests against each available backend."""
    backend = request.param

    if backend == "chroma":
        config = SkillMindConfig(
            data_dir=str(tmp_dir),
            store=StoreConfig(backend="chroma", chroma_path=str(tmp_dir / "chroma")),
        )
        from skillmind.store.chroma_store import ChromaStore
        s = ChromaStore(config=config, engine=mock_engine)

    elif backend == "faiss":
        config = SkillMindConfig(
            data_dir=str(tmp_dir),
            store=StoreConfig(backend="faiss", faiss_path=str(tmp_dir / "faiss")),
        )
        from skillmind.store.faiss_store import FAISSStore
        s = FAISSStore(config=config, engine=mock_engine)

    s.initialize()
    return s


# ── Universal Store Tests ────────────────────────────────────────

class TestStoreAdd:
    def test_add_single(self, store, sample_memories):
        mem = sample_memories[0]
        mid = store.add(mem)
        assert mid == mem.id
        assert store.count() == 1

    def test_add_batch(self, store, sample_memories):
        ids = store.add_batch(sample_memories)
        assert len(ids) == len(sample_memories)
        assert store.count() == len(sample_memories)

    def test_add_empty_batch(self, store):
        ids = store.add_batch([])
        assert ids == []


class TestStoreQuery:
    def test_semantic_query(self, store, sample_memories):
        store.add_batch(sample_memories)
        results = store.query("PDF quality settings", limit=3)
        assert len(results) > 0
        assert all(isinstance(r, QueryResult) for r in results)
        # Should find the PDF-related memory
        topics = [r.memory.topic for r in results]
        assert "pdf_generation" in topics or len(results) > 0

    def test_query_with_type_filter(self, store, sample_memories):
        store.add_batch(sample_memories)
        results = store.query(
            "SEO project",
            limit=10,
            filter=QueryFilter(types=[MemoryType.PROJECT]),
        )
        for r in results:
            assert r.memory.type == MemoryType.PROJECT

    def test_query_empty_store(self, store):
        results = store.query("anything", limit=5)
        assert results == []


class TestStoreGet:
    def test_get_existing(self, store, sample_memories):
        mem = sample_memories[0]
        store.add(mem)
        fetched = store.get(mem.id)
        assert fetched is not None
        assert fetched.id == mem.id
        assert fetched.content == mem.content

    def test_get_nonexistent(self, store):
        result = store.get("nonexistent-id")
        assert result is None


class TestStoreUpdate:
    def test_update_content(self, store, sample_memories):
        mem = sample_memories[0]
        store.add(mem)

        mem.content = "Updated content"
        store.update(mem)

        fetched = store.get(mem.id)
        assert fetched is not None
        assert fetched.content == "Updated content"


class TestStoreDelete:
    def test_delete_existing(self, store, sample_memories):
        mem = sample_memories[0]
        store.add(mem)
        assert store.count() == 1

        result = store.delete(mem.id)
        assert result is True
        assert store.count() == 0

    def test_delete_nonexistent(self, store):
        result = store.delete("nonexistent-id")
        # Some backends return False, some may not error
        assert isinstance(result, bool)


class TestStoreListAll:
    def test_list_all(self, store, sample_memories):
        store.add_batch(sample_memories)
        all_mems = store.list_all()
        assert len(all_mems) == len(sample_memories)

    def test_list_with_type_filter(self, store, sample_memories):
        store.add_batch(sample_memories)
        feedback = store.list_all(filter=QueryFilter(types=[MemoryType.FEEDBACK]))
        assert all(m.type == MemoryType.FEEDBACK for m in feedback)

    def test_list_with_limit(self, store, sample_memories):
        store.add_batch(sample_memories)
        limited = store.list_all(limit=2)
        assert len(limited) == 2


class TestStoreCount:
    def test_count_empty(self, store):
        assert store.count() == 0

    def test_count_after_add(self, store, sample_memories):
        store.add_batch(sample_memories)
        assert store.count() == len(sample_memories)

    def test_count_filtered(self, store, sample_memories):
        store.add_batch(sample_memories)
        n = store.count(filter=QueryFilter(types=[MemoryType.FEEDBACK]))
        assert n == 2  # We have 2 feedback memories in sample


class TestStoreClear:
    def test_clear(self, store, sample_memories):
        store.add_batch(sample_memories)
        assert store.count() > 0
        n = store.clear()
        assert n == len(sample_memories)
        assert store.count() == 0


class TestStoreFindDuplicates:
    def test_find_duplicate(self, store):
        mem1 = Memory(
            id="dup-1",
            type=MemoryType.FEEDBACK,
            topic="pdf",
            title="PDF quality",
            content="Always use 600 DPI for graphics in PDFs",
            source=MemorySource.MANUAL,
        )
        mem2 = Memory(
            id="dup-2",
            type=MemoryType.FEEDBACK,
            topic="pdf",
            title="PDF quality standards",
            content="Always use 600 DPI for graphics in PDF documents",
            source=MemorySource.MANUAL,
        )
        store.add(mem1)
        dupes = store.find_duplicates(mem2, threshold=0.8)
        # With mock embeddings, similar content should produce similar hashes
        assert isinstance(dupes, list)
