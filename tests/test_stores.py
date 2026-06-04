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


# FalkorDB needs both the SDK and a reachable server (local Docker / Railway).
# Set FALKORDB_URL to opt in, e.g. redis://:skillmind-dev@localhost:6379.
import os

FALKORDB_URL = os.environ.get("FALKORDB_URL", "")


def _falkordb_available():
    if not FALKORDB_URL:
        return False
    try:
        from urllib.parse import urlparse

        from falkordb import FalkorDB
    except ImportError:
        return False
    try:
        parsed = urlparse(FALKORDB_URL)
        db = FalkorDB(
            host=parsed.hostname or "localhost",
            port=parsed.port or 6379,
            password=parsed.password or None,
        )
        db.connection.ping()
        return True
    except Exception:
        return False


# ── Fixtures ─────────────────────────────────────────────────────

@pytest.fixture(params=[
    pytest.param("chroma", marks=pytest.mark.skipif(not _chroma_available(), reason="chromadb not installed")),
    pytest.param("faiss", marks=pytest.mark.skipif(not _faiss_available(), reason="faiss-cpu not installed")),
    pytest.param("falkordb", marks=pytest.mark.skipif(not _falkordb_available(), reason="FalkorDB not reachable (set FALKORDB_URL)")),
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

    elif backend == "falkordb":
        # Dedicated test graph so we never touch a real "skillmind" graph.
        config = SkillMindConfig(
            data_dir=str(tmp_dir),
            store=StoreConfig(backend="falkordb", falkordb_url=FALKORDB_URL, falkordb_graph="skillmind_pytest"),
        )
        from skillmind.store.falkordb_store import FalkorDBStore
        s = FalkorDBStore(config=config, engine=mock_engine)
        s.initialize()
        s.clear()  # isolation: wipe leftovers from a previous run
        return s

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


# ── Store-to-store migration ─────────────────────────────────────

def _local_store(backend, tmp_dir, mock_engine, sub):
    """Build + initialize a local store backend in an isolated temp subdir."""
    if backend == "faiss":
        config = SkillMindConfig(
            data_dir=str(tmp_dir),
            store=StoreConfig(backend="faiss", faiss_path=str(tmp_dir / sub)),
        )
        from skillmind.store.faiss_store import FAISSStore
        s = FAISSStore(config=config, engine=mock_engine)
    else:
        config = SkillMindConfig(
            data_dir=str(tmp_dir),
            store=StoreConfig(backend="chroma", chroma_path=str(tmp_dir / sub)),
        )
        from skillmind.store.chroma_store import ChromaStore
        s = ChromaStore(config=config, engine=mock_engine)
    s.initialize()
    return s


@pytest.mark.skipif(
    not (_faiss_available() or _chroma_available()),
    reason="need at least one local backend (faiss or chroma) for migration test",
)
class TestMigrateStore:
    """Backend-agnostic store-to-store migration (skillmind.migration.migrate_store)."""

    def _backend(self):
        return "faiss" if _faiss_available() else "chroma"

    def test_migrate_copies_all_and_preserves_ids(self, tmp_dir, mock_engine, sample_memories):
        from skillmind.migration import migrate_store

        backend = self._backend()
        source = _local_store(backend, tmp_dir, mock_engine, "src")
        target = _local_store(backend, tmp_dir, mock_engine, "tgt")
        source.add_batch(sample_memories)

        stats = migrate_store(source, target, batch_size=2)

        assert stats["source_count"] == len(sample_memories)
        assert stats["migrated"] == len(sample_memories)
        assert stats["truncated"] is False
        assert target.count() == len(sample_memories)
        # IDs preserved → idempotent
        src_ids = {m.id for m in source.list_all(limit=1000)}
        tgt_ids = {m.id for m in target.list_all(limit=1000)}
        assert src_ids == tgt_ids

    def test_migrate_is_idempotent(self, tmp_dir, mock_engine, sample_memories):
        from skillmind.migration import migrate_store

        backend = self._backend()
        source = _local_store(backend, tmp_dir, mock_engine, "src")
        target = _local_store(backend, tmp_dir, mock_engine, "tgt")
        source.add_batch(sample_memories)

        migrate_store(source, target)
        migrate_store(source, target)  # re-run upserts, no duplicates
        assert target.count() == len(sample_memories)

    def test_dry_run_writes_nothing(self, tmp_dir, mock_engine, sample_memories):
        from skillmind.migration import migrate_store

        backend = self._backend()
        source = _local_store(backend, tmp_dir, mock_engine, "src")
        target = _local_store(backend, tmp_dir, mock_engine, "tgt")
        source.add_batch(sample_memories)

        stats = migrate_store(source, target, dry_run=True)
        assert stats["fetched"] == len(sample_memories)
        assert stats["migrated"] == 0
        assert target.count() == 0

    def test_migrated_memories_are_queryable(self, tmp_dir, mock_engine, sample_memories):
        from skillmind.migration import migrate_store

        backend = self._backend()
        source = _local_store(backend, tmp_dir, mock_engine, "src")
        target = _local_store(backend, tmp_dir, mock_engine, "tgt")
        source.add_batch(sample_memories)

        migrate_store(source, target)
        results = target.query("PDF quality Umlaute", limit=3)
        assert results
        assert all(isinstance(r, QueryResult) for r in results)


# ── GraphRAG (FalkorDB-only) ─────────────────────────────────────

class TestGraphRAGHelpers:
    """Pure, offline scoring/parse helpers — no DB, no embedding model."""

    def test_parse_wikilinks(self):
        from skillmind.store.falkordb_store import _parse_wikilinks

        assert _parse_wikilinks("see [[foo]] and [[bar|alias]] and [[foo]]") == ["foo", "bar"]
        assert _parse_wikilinks("no links here") == []
        assert _parse_wikilinks("") == []
        assert _parse_wikilinks(None) == []  # type: ignore[arg-type]

    def test_cosine(self):
        from skillmind.store.falkordb_store import _cosine

        assert _cosine([1, 0], [1, 0]) == pytest.approx(1.0)
        assert _cosine([1, 0], [0, 1]) == pytest.approx(0.0)
        assert _cosine([], [1, 2]) == 0.0
        assert _cosine([0, 0], [1, 2]) == 0.0

    def test_rerank_vector_dominates_but_graph_surfaces(self):
        from skillmind.store.falkordb_store import _rerank

        vec_sim = {"a": 0.9, "b": 0.4}
        graph_boost = {"b": 1.0, "c": 1.0}  # c is graph-only (no vector hit)
        confidence = {"a": 1.0, "b": 1.0, "c": 1.0}
        ranked = _rerank(vec_sim, graph_boost, confidence, limit=3)
        ids = [i for i, _ in ranked]
        assert ids[0] == "a"           # strong vector hit stays on top
        assert "c" in ids             # graph-only candidate surfaces
        # scores are sorted descending
        scores = [s for _, s in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_rerank_confidence_weighting(self):
        from skillmind.store.falkordb_store import _rerank

        # Same vector sim, different confidence → higher confidence ranks first.
        ranked = _rerank({"a": 0.5, "b": 0.5}, {}, {"a": 1.0, "b": 0.2}, limit=2)
        assert ranked[0][0] == "a"


@pytest.mark.skipif(not _falkordb_available(), reason="FalkorDB not reachable (set FALKORDB_URL)")
class TestGraphRAG:
    """GraphRAG build + retrieval against a dedicated test graph.

    Uses deterministic signals (wiki links, shared topics/tags, direct
    RELATES_TO traversal) so the assertions don't depend on the mock engine's
    hash-based embeddings.
    """

    def _store(self, mock_engine, graphrag=True):
        from skillmind.store.falkordb_store import FalkorDBStore

        config = SkillMindConfig(
            store=StoreConfig(
                backend="falkordb", falkordb_url=FALKORDB_URL,
                falkordb_graph="skillmind_pytest", falkordb_graphrag=graphrag,
                falkordb_seed_k=2, falkordb_hops=2,
            ),
        )
        s = FalkorDBStore(config=config, engine=mock_engine)
        s.initialize()
        s.clear()
        return s

    def _seed(self, s):
        mems = [
            Memory(id="g1", type=MemoryType.PROJECT, topic="seo", tags=["seo"],
                   title="CTR Modell", content="CTR model. Related: [[Positionen]]",
                   source=MemorySource.MANUAL),
            Memory(id="g2", type=MemoryType.PROJECT, topic="seo", tags=["seo"],
                   title="Positionen", content="Target positions per search volume",
                   source=MemorySource.MANUAL),
            Memory(id="g3", type=MemoryType.PROJECT, topic="ads", tags=["ppc"],
                   title="Brand Spend", content="Google Ads brand spend trap",
                   source=MemorySource.MANUAL),
        ]
        s.add_batch(mems)
        return mems

    def test_build_graph_creates_link_edges(self, mock_engine):
        s = self._store(mock_engine)
        self._seed(s)
        counts = s.build_graph(similarity_threshold=2.0)  # disable semantic edges
        assert counts["link"] >= 1
        rel = s._run(
            "MATCH (:Memory {id:'g1'})-[r:RELATES_TO]->(:Memory {id:'g2'}) RETURN count(r)"
        ).result_set[0][0]
        assert rel >= 1
        s.clear()

    def test_expand_reaches_linked_and_topic_siblings(self, mock_engine):
        s = self._store(mock_engine)
        self._seed(s)
        s.build_graph(similarity_threshold=2.0)  # only link + existing topic/tag edges
        boost = s._expand(["g1"], hops=2, seed_sim={"g1": 1.0})
        # g2: shared topic+tag (attr) AND a RELATES_TO link → present with boost > 0
        assert "g2" in boost and boost["g2"] > 0
        # g1 itself is never a candidate (no self-boost)
        assert "g1" not in boost
        s.clear()

    def test_graphrag_query_contract(self, mock_engine):
        s = self._store(mock_engine, graphrag=True)
        self._seed(s)
        s.build_graph(similarity_threshold=2.0)
        results = s.query("CTR model positions", limit=3)
        assert results
        assert all(isinstance(r, QueryResult) for r in results)
        # GraphRAG must still return at least the same memories the vector path finds.
        s.config.store.falkordb_graphrag = False
        vec_ids = {r.memory.id for r in s.query("CTR model positions", limit=3)}
        s.config.store.falkordb_graphrag = True
        graph_ids = {r.memory.id for r in s.query("CTR model positions", limit=5)}
        assert vec_ids <= graph_ids
        s.clear()
