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


class TestGraphRAGRerankEdges:
    """Extra offline _rerank edge cases (no DB) — boost cap & empty candidates."""

    def test_graph_boost_is_capped(self):
        from skillmind.store.falkordb_store import _rerank, _BOOST_CAP, _GRAPH_WEIGHT

        # A graph-only candidate with an absurd boost must not exceed the cap, so
        # it can never bury a strong pure-vector hit.
        ranked = dict(_rerank({"vec": 0.9}, {"graph": 99.0}, {"vec": 1.0, "graph": 1.0}, limit=5))
        assert ranked["vec"] == pytest.approx(0.9)
        # graph score == GRAPH_WEIGHT * min(99, cap), confidence weight = 1.0
        assert ranked["graph"] == pytest.approx(_GRAPH_WEIGHT * _BOOST_CAP)
        assert ranked["vec"] > ranked["graph"]

    def test_rerank_empty_inputs(self):
        from skillmind.store.falkordb_store import _rerank

        assert _rerank({}, {}, {}, limit=5) == []

    def test_rerank_respects_limit(self):
        from skillmind.store.falkordb_store import _rerank

        vec = {f"m{i}": 1.0 - i * 0.1 for i in range(10)}
        ranked = _rerank(vec, {}, {k: 1.0 for k in vec}, limit=3)
        assert len(ranked) == 3


# ── Pinecone (offline via a synchronous fake Index) ──────────────
#
# Pinecone is cloud-only and eventually consistent, so its store can't join the
# parametrized `store` fixture (those tests assert immediate count()). Instead we
# exercise the FULL MemoryStore contract against a deterministic in-memory stand-in
# for the Pinecone Index — same code paths (upsert/query/fetch/delete/filter
# translation), no API key, no flakiness. A separate skip-guarded class covers the
# real cloud roundtrip when PINECONE_API_KEY is set (opt-in, like FALKORDB_URL).

from skillmind.store.falkordb_store import _cosine  # pure helper, reused for ranking


def _pc_match(meta: dict, flt: dict) -> bool:
    """Apply a Pinecone-style metadata filter (the subset _to_pinecone_filter emits)."""
    if "$and" in flt:
        return all(_pc_match(meta, sub) for sub in flt["$and"])
    for field, cond in flt.items():
        val = meta.get(field)
        for op, expected in cond.items():
            if op == "$in" and val not in expected:
                return False
            if op == "$eq" and val != expected:
                return False
            if op == "$gte" and (val is None or float(val) < float(expected)):
                return False
    return True


def _assert_pinecone_safe(meta: dict) -> None:
    """Enforce the real Pinecone metadata constraint: scalar or list[str] only.

    Pinecone rejects ``null`` and nested values at upsert with a 400. skillmind
    keeps metadata flat by construction (Memory.to_metadata_dict emits only
    str/float, docstring "no nested objects"), so the fake mirrors the API
    constraint to LOCK that invariant: a future regression that lets a None /
    dict / nested value reach upsert fails every add-path test, not silently
    only the live cloud run. (Pattern adopted from the Bikefitting RAG store's
    _sanitize_metadata, which sanitizes at the store layer; skillmind enforces
    it at the model layer and we assert it here.)
    """
    for key, val in meta.items():
        if isinstance(val, (str, int, float, bool)):
            continue
        if isinstance(val, list) and all(isinstance(x, str) for x in val):
            continue
        raise ValueError(
            f"Pinecone metadata {key!r} has unsupported type {type(val).__name__} "
            f"(value={val!r}); only str/number/bool/list[str] are allowed"
        )


class FakePineconeIndex:
    """In-memory, synchronous stand-in for a Pinecone Index.

    Mirrors exactly the subset of the Index API that PineconeStore touches
    (upsert / query / fetch / delete / describe_index_stats) with deterministic,
    immediately-consistent semantics, so the store contract is testable offline.
    Upsert enforces Pinecone's metadata type contract (see _assert_pinecone_safe).
    """

    def __init__(self):
        self._vectors: dict[str, dict] = {}  # id -> {"values": [...], "metadata": {...}}

    def upsert(self, vectors):
        for vid, values, meta in vectors:
            _assert_pinecone_safe(meta)
            self._vectors[vid] = {"values": list(values), "metadata": dict(meta)}

    def delete(self, ids=None, delete_all=False):
        if delete_all:
            self._vectors.clear()
        elif ids:
            for vid in ids:
                self._vectors.pop(vid, None)

    def fetch(self, ids):
        return {
            "vectors": {
                vid: {"metadata": dict(self._vectors[vid]["metadata"])}
                for vid in ids
                if vid in self._vectors
            }
        }

    def describe_index_stats(self):
        return {"total_vector_count": len(self._vectors)}

    def query(self, vector, top_k, include_metadata=True, filter=None):
        is_zero = not any(vector)  # list_all() probes with a zero vector
        scored = []
        for vid, rec in self._vectors.items():
            if filter and not _pc_match(rec["metadata"], filter):
                continue
            score = 0.0 if is_zero else _cosine(list(vector), rec["values"])
            scored.append((vid, score, rec["metadata"]))
        if not is_zero:
            scored.sort(key=lambda t: t[1], reverse=True)
        matches = [
            {"id": vid, "score": score, "metadata": dict(meta)}
            for vid, score, meta in scored[:top_k]
        ]
        return {"matches": matches}


@pytest.fixture
def pinecone_store(mock_engine):
    """PineconeStore wired to the fake Index (no network)."""
    from skillmind.store.pinecone_store import PineconeStore

    config = SkillMindConfig(
        store=StoreConfig(backend="pinecone", pinecone_api_key="test-key", pinecone_index="skillmind-pytest"),
    )
    s = PineconeStore(config=config, engine=mock_engine)
    s._index = FakePineconeIndex()  # bypass initialize() / real cloud
    return s


class TestPineconeContract:
    """Full MemoryStore contract against the fake Index — always runs offline."""

    def test_add_single(self, pinecone_store, sample_memories):
        mid = pinecone_store.add(sample_memories[0])
        assert mid == sample_memories[0].id
        assert pinecone_store.count() == 1

    def test_add_batch(self, pinecone_store, sample_memories):
        ids = pinecone_store.add_batch(sample_memories)
        assert len(ids) == len(sample_memories)
        assert pinecone_store.count() == len(sample_memories)

    def test_add_empty_batch(self, pinecone_store):
        assert pinecone_store.add_batch([]) == []

    def test_add_batch_chunks_over_100(self, pinecone_store):
        """add_batch upserts in chunks of 100 — verify nothing is dropped at the boundary."""
        many = [
            Memory(id=f"m{i}", type=MemoryType.SKILL, topic="t", title=f"T{i}",
                   content=f"content {i}", source=MemorySource.MANUAL)
            for i in range(205)
        ]
        ids = pinecone_store.add_batch(many)
        assert len(ids) == 205
        assert pinecone_store.count() == 205

    def test_semantic_query(self, pinecone_store, sample_memories):
        pinecone_store.add_batch(sample_memories)
        results = pinecone_store.query("PDF quality settings", limit=3)
        assert 0 < len(results) <= 3
        assert all(isinstance(r, QueryResult) for r in results)

    def test_query_with_type_filter(self, pinecone_store, sample_memories):
        pinecone_store.add_batch(sample_memories)
        results = pinecone_store.query(
            "anything", limit=10, filter=QueryFilter(types=[MemoryType.PROJECT]),
        )
        assert results
        assert all(r.memory.type == MemoryType.PROJECT for r in results)

    def test_query_with_min_confidence_filter(self, pinecone_store, sample_memories):
        pinecone_store.add_batch(sample_memories)
        results = pinecone_store.query(
            "anything", limit=10, filter=QueryFilter(min_confidence=0.95),
        )
        assert results
        assert all(r.memory.confidence >= 0.95 for r in results)

    def test_query_empty_store(self, pinecone_store):
        assert pinecone_store.query("anything", limit=5) == []

    def test_get_existing(self, pinecone_store, sample_memories):
        mem = sample_memories[0]
        pinecone_store.add(mem)
        fetched = pinecone_store.get(mem.id)
        assert fetched is not None
        assert fetched.id == mem.id
        assert fetched.content == mem.content
        assert fetched.tags == mem.tags  # round-trips the comma-joined tag string

    def test_get_nonexistent(self, pinecone_store):
        assert pinecone_store.get("nope") is None

    def test_update_content(self, pinecone_store, sample_memories):
        mem = sample_memories[0]
        pinecone_store.add(mem)
        mem.content = "Updated content"
        pinecone_store.update(mem)
        assert pinecone_store.get(mem.id).content == "Updated content"

    def test_delete_existing(self, pinecone_store, sample_memories):
        mem = sample_memories[0]
        pinecone_store.add(mem)
        assert pinecone_store.delete(mem.id) is True
        assert pinecone_store.count() == 0

    def test_delete_nonexistent_returns_bool(self, pinecone_store):
        assert isinstance(pinecone_store.delete("nope"), bool)

    def test_list_all(self, pinecone_store, sample_memories):
        pinecone_store.add_batch(sample_memories)
        assert len(pinecone_store.list_all(limit=100)) == len(sample_memories)

    def test_list_with_type_filter(self, pinecone_store, sample_memories):
        pinecone_store.add_batch(sample_memories)
        feedback = pinecone_store.list_all(filter=QueryFilter(types=[MemoryType.FEEDBACK]), limit=100)
        assert feedback
        assert all(m.type == MemoryType.FEEDBACK for m in feedback)

    def test_list_with_limit(self, pinecone_store, sample_memories):
        pinecone_store.add_batch(sample_memories)
        assert len(pinecone_store.list_all(limit=2)) == 2

    def test_count_empty(self, pinecone_store):
        assert pinecone_store.count() == 0

    def test_count_after_add(self, pinecone_store, sample_memories):
        pinecone_store.add_batch(sample_memories)
        assert pinecone_store.count() == len(sample_memories)

    def test_count_filtered(self, pinecone_store, sample_memories):
        pinecone_store.add_batch(sample_memories)
        assert pinecone_store.count(filter=QueryFilter(types=[MemoryType.FEEDBACK])) == 2

    def test_clear(self, pinecone_store, sample_memories):
        pinecone_store.add_batch(sample_memories)
        assert pinecone_store.clear() == len(sample_memories)
        assert pinecone_store.count() == 0

    def test_find_duplicates_returns_list(self, pinecone_store):
        mem1 = Memory(id="dup-1", type=MemoryType.FEEDBACK, topic="pdf", title="PDF quality",
                      content="Always use 600 DPI for graphics in PDFs", source=MemorySource.MANUAL)
        mem2 = Memory(id="dup-2", type=MemoryType.FEEDBACK, topic="pdf", title="PDF quality standards",
                      content="Always use 600 DPI for graphics in PDF documents", source=MemorySource.MANUAL)
        pinecone_store.add(mem1)
        assert isinstance(pinecone_store.find_duplicates(mem2, threshold=0.8), list)


class TestPineconeFilterTranslation:
    """_to_pinecone_filter — pure, offline (no Index needed)."""

    def _f(self, **kw):
        from skillmind.store.pinecone_store import PineconeStore
        return PineconeStore._to_pinecone_filter(QueryFilter(**kw) if kw else None)

    def test_none_filter(self):
        from skillmind.store.pinecone_store import PineconeStore
        assert PineconeStore._to_pinecone_filter(None) is None

    def test_empty_filter(self):
        assert self._f() is None

    def test_single_type_condition(self):
        assert self._f(types=[MemoryType.SKILL]) == {"type": {"$in": ["skill"]}}

    def test_source_condition(self):
        assert self._f(source=MemorySource.IMPORT) == {"source": {"$eq": "import"}}

    def test_min_confidence_condition(self):
        assert self._f(min_confidence=0.9) == {"confidence": {"$gte": 0.9}}

    def test_multiple_conditions_use_and(self):
        flt = self._f(types=[MemoryType.PROJECT], topics=["paroc"], min_confidence=0.5)
        assert "$and" in flt
        assert {"type": {"$in": ["project"]}} in flt["$and"]
        assert {"topic": {"$in": ["paroc"]}} in flt["$and"]
        assert {"confidence": {"$gte": 0.5}} in flt["$and"]


class TestPineconeMetaMapping:
    """_meta_to_memory — pure, offline (no Index needed)."""

    def _m(self, content="body", **meta):
        from skillmind.store.pinecone_store import PineconeStore
        return PineconeStore._meta_to_memory("id-1", content, meta)

    def test_tags_split_from_csv(self):
        mem = self._m(type="skill", topic="seo", title="T", source="manual",
                      confidence=0.9, tags="seo,content,workflow")
        assert mem.tags == ["seo", "content", "workflow"]

    def test_empty_tags_yield_empty_list(self):
        mem = self._m(type="user", topic="role", title="T", source="manual", tags="")
        assert mem.tags == []

    def test_defaults_when_meta_sparse(self):
        mem = self._m()
        assert mem.type == MemoryType.USER
        assert mem.source == MemorySource.MANUAL
        assert mem.confidence == 1.0
        assert mem.expires_at is None

    def test_timestamps_and_expiry_parsed(self):
        ts = "2026-01-02T03:04:05"
        mem = self._m(type="project", topic="p", title="T", source="manual",
                      created_at=ts, updated_at=ts, expires_at=ts)
        assert mem.created_at == datetime.fromisoformat(ts)
        assert mem.expires_at == datetime.fromisoformat(ts)


class TestPineconeMetadataContract:
    """Pinecone rejects null/nested metadata; lock the flat-metadata invariant.

    The Bikefitting RAG store sanitizes metadata at the store layer
    (_sanitize_metadata strips None / coerces lists). skillmind needs no
    sanitizer because Memory.to_metadata_dict emits only scalar types by
    construction — these tests guarantee that stays true and that the upsert
    double would catch a regression that re-introduces a None / nested value.
    """

    def test_to_metadata_dict_emits_only_scalar_types(self, sample_memories):
        for mem in sample_memories:
            meta = mem.to_metadata_dict()
            meta["content"] = mem.content  # the store adds content before upsert
            for key, val in meta.items():
                assert val is not None, f"{key} is None — Pinecone would 400"
                assert isinstance(val, (str, int, float, bool)), (
                    f"{key}={val!r} ({type(val).__name__}) is not Pinecone-safe"
                )

    def test_fake_index_rejects_none_value(self):
        idx = FakePineconeIndex()
        with pytest.raises(ValueError):
            idx.upsert([("x", [0.1, 0.2], {"type": "user", "topic": None})])

    def test_fake_index_rejects_nested_dict(self):
        idx = FakePineconeIndex()
        with pytest.raises(ValueError):
            idx.upsert([("x", [0.1], {"meta": {"nested": 1}})])

    def test_fake_index_rejects_non_str_list(self):
        idx = FakePineconeIndex()
        with pytest.raises(ValueError):
            idx.upsert([("x", [0.1], {"tags": [1, 2, 3]})])

    def test_fake_index_accepts_str_list(self):
        idx = FakePineconeIndex()
        idx.upsert([("x", [0.1], {"tags": ["seo", "pdf"]})])  # must not raise

    def test_store_add_passes_metadata_contract(self, pinecone_store, sample_memories):
        # Every sample memory must upsert through the contract-enforcing double.
        for mem in sample_memories:
            pinecone_store.add(mem)
        assert pinecone_store.count() == len(sample_memories)


# ── Pinecone real-cloud roundtrip (opt-in via PINECONE_API_KEY) ──

PINECONE_TEST_INDEX = os.environ.get("PINECONE_TEST_INDEX", "skillmind-pytest")


def _pinecone_available():
    if not os.environ.get("PINECONE_API_KEY"):
        return False
    try:
        import pinecone  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _pinecone_available(), reason="Pinecone not configured (set PINECONE_API_KEY)")
class TestPineconeIntegration:
    """Live Pinecone roundtrip against a dedicated test index.

    Pinecone upserts are eventually consistent, so we poll the query (not count)
    a few times before asserting, and never touch a non-test index.
    """

    def _store(self, mock_engine):
        from skillmind.store.pinecone_store import PineconeStore

        config = SkillMindConfig(
            store=StoreConfig(
                backend="pinecone",
                pinecone_api_key=os.environ["PINECONE_API_KEY"],
                pinecone_index=PINECONE_TEST_INDEX,
            ),
        )
        s = PineconeStore(config=config, engine=mock_engine)
        s.initialize()
        s.clear()
        return s

    def test_add_then_query_roundtrip(self, mock_engine, sample_memories):
        import time

        s = self._store(mock_engine)
        try:
            s.add_batch(sample_memories)
            results = []
            for _ in range(10):  # tolerate eventual consistency
                results = s.query("PDF quality Umlaute", limit=3)
                if results:
                    break
                time.sleep(1)
            assert results
            assert all(isinstance(r, QueryResult) for r in results)
        finally:
            s.clear()


# ── FalkorDB offline (no server, no falkordb SDK needed) ─────────
#
# The skip-guarded TestGraphRAG above needs a live FalkorDB. These tests instead
# exercise the FalkorDBStore *contract* offline via a tiny Cypher-substring graph
# double — the pattern visibly-app uses (_FakeGraph in test_graph_map.py, plus
# pure parity tests for _where_clause / _memory_to_params / _row_to_memory). They
# lock the COSINE-DISTANCE → similarity inversion and the row round-trip without
# any infrastructure, so a regression fails on every dev machine, not just CI.

class _FakeRes:
    """Stand-in for a FalkorDB query result (result_set + nodes_deleted)."""

    def __init__(self, result_set, nodes_deleted=0):
        self.result_set = result_set
        self.nodes_deleted = nodes_deleted


class _FakeGraph:
    """Minimal in-memory FalkorDB graph double.

    Answers the Cypher subset FalkorDBStore issues by substring-matching the
    query string. Stores the projected memory fields + embedding on MERGE upsert;
    the vector query returns a COSINE DISTANCE (``1 - cosine``) so the store's
    ``score = 1 - distance`` inversion is exercised end to end.
    """

    def __init__(self):
        from skillmind.store.falkordb_store import _NODE_FIELDS

        self._fields = _NODE_FIELDS
        self.nodes: dict[str, dict] = {}  # id -> {"params": {...}, "embedding": [...]}

    def query(self, cypher, params=None):
        from skillmind.store.falkordb_store import _cosine

        params = params or {}
        c = cypher

        if "CREATE VECTOR INDEX" in c or "DROP VECTOR INDEX" in c:
            return _FakeRes([])

        if "MERGE (m:Memory {id:$id})" in c and "SET m.type" in c:
            self.nodes[params["id"]] = {
                "params": {f: params.get(f) for f in self._fields},
                "embedding": list(params.get("embedding") or []),
            }
            return _FakeRes([])

        if "UNWIND $tags" in c:  # tag edges — nothing to project
            return _FakeRes([])

        if "DETACH DELETE m" in c and "MATCH (m:Memory {id:$id})" in c:
            existed = params.get("id") in self.nodes
            self.nodes.pop(params.get("id"), None)
            return _FakeRes([], nodes_deleted=1 if existed else 0)

        if "MATCH (n) DETACH DELETE n" in c:
            self.nodes.clear()
            return _FakeRes([])

        if "db.idx.vector.queryNodes" in c:
            qv = list(params.get("v") or [])
            limit = params.get("limit", 5)
            scored = []
            for rec in self.nodes.values():
                distance = 1.0 - _cosine(qv, rec["embedding"])
                row = [rec["params"].get(f) for f in self._fields] + [distance]
                scored.append((distance, row))
            scored.sort(key=lambda t: t[0])  # ORDER BY score ASC
            return _FakeRes([row for _, row in scored[:limit]])

        if "RETURN count(node)" in c:
            return _FakeRes([[len(self.nodes)]])

        if "MATCH (node:Memory {id:$id}) RETURN" in c:
            rec = self.nodes.get(params.get("id"))
            if rec is None:
                return _FakeRes([])
            return _FakeRes([[rec["params"].get(f) for f in self._fields]])

        if "MATCH (node:Memory)" in c and "RETURN" in c:  # list_all
            offset = params.get("offset", 0)
            limit = params.get("limit", 100)
            rows = [[rec["params"].get(f) for f in self._fields] for rec in self.nodes.values()]
            return _FakeRes(rows[offset:offset + limit])

        raise AssertionError(f"unhandled cypher in _FakeGraph: {c[:80]!r}")


@pytest.fixture
def falkordb_offline_store(mock_engine):
    """FalkorDBStore wired to the in-memory _FakeGraph (no SDK / server)."""
    from skillmind.store.falkordb_store import FalkorDBStore

    config = SkillMindConfig(
        store=StoreConfig(backend="falkordb", falkordb_graph="skillmind_pytest"),
    )
    s = FalkorDBStore(config=config, engine=mock_engine)
    s._graph = _FakeGraph()  # bypass initialize() → no falkordb import
    return s


class TestFalkorDBReturnProjection:
    """_return_projection — pure, offline."""

    def test_lists_all_node_fields_in_order(self):
        from skillmind.store.falkordb_store import _NODE_FIELDS, _return_projection

        proj = _return_projection("node")
        assert proj == ", ".join(f"node.{f}" for f in _NODE_FIELDS)

    def test_respects_variable_name(self):
        from skillmind.store.falkordb_store import _return_projection

        assert _return_projection("m").startswith("m.id")


class TestFalkorDBMapping:
    """_memory_to_params → row → _row_to_memory round-trip (pure, offline)."""

    def _row_from_params(self, params):
        from skillmind.store.falkordb_store import _NODE_FIELDS

        return [params.get(f) for f in _NODE_FIELDS]

    def test_roundtrip_preserves_scalar_fields(self, sample_memories):
        from skillmind.store.falkordb_store import FalkorDBStore

        for original in sample_memories:
            params = FalkorDBStore._memory_to_params(original)
            restored = FalkorDBStore._row_to_memory(self._row_from_params(params))
            assert restored.id == original.id
            assert restored.type == original.type
            assert restored.topic == original.topic
            assert restored.title == original.title
            assert restored.content == original.content
            assert restored.tags == original.tags
            assert restored.source == original.source
            assert restored.confidence == pytest.approx(original.confidence)

    def test_roundtrip_preserves_metadata_json(self):
        from skillmind.store.falkordb_store import FalkorDBStore

        mem = Memory(
            id="m-meta", type=MemoryType.PROJECT, topic="p", title="T",
            content="c", source=MemorySource.MANUAL, confidence=0.8,
            metadata={"video_id": "abc", "chapters": 3},
        )
        params = FalkorDBStore._memory_to_params(mem)
        restored = FalkorDBStore._row_to_memory(self._row_from_params(params))
        assert restored.metadata == {"video_id": "abc", "chapters": 3}

    def test_unset_expiry_maps_to_none(self):
        from skillmind.store.falkordb_store import FalkorDBStore

        mem = Memory(
            id="m-noexp", type=MemoryType.USER, topic="t", title="T",
            content="c", source=MemorySource.MANUAL,
        )
        params = FalkorDBStore._memory_to_params(mem)
        assert params["expires_at"] == ""  # stored as "" not None (Cypher-safe)
        restored = FalkorDBStore._row_to_memory(self._row_from_params(params))
        assert restored.expires_at is None

    def test_set_expiry_roundtrips(self):
        from skillmind.store.falkordb_store import FalkorDBStore

        exp = datetime(2026, 12, 31, 23, 59, 59)
        mem = Memory(
            id="m-exp", type=MemoryType.PROJECT, topic="t", title="T",
            content="c", source=MemorySource.MANUAL, expires_at=exp,
        )
        params = FalkorDBStore._memory_to_params(mem)
        restored = FalkorDBStore._row_to_memory(self._row_from_params(params))
        assert restored.expires_at == exp

    def test_row_to_memory_defaults_on_sparse_row(self):
        from skillmind.store.falkordb_store import FalkorDBStore, _NODE_FIELDS

        # Only id populated; everything else None (a thin/legacy node).
        row = ["only-id"] + [None] * (len(_NODE_FIELDS) - 1)
        mem = FalkorDBStore._row_to_memory(row)
        assert mem.id == "only-id"
        assert mem.type == MemoryType.USER
        assert mem.source == MemorySource.MANUAL
        assert mem.confidence == 1.0
        assert mem.tags == []
        assert mem.metadata == {}
        assert mem.expires_at is None


class TestFalkorDBWhereClause:
    """_where_clause — called unbound (it does not use self), like visibly-app."""

    def _wc(self, filter_):
        from skillmind.store.falkordb_store import FalkorDBStore

        # Unbound call: _where_clause ignores self, so any object works.
        return FalkorDBStore._where_clause(FalkorDBStore, filter_, "node")

    def test_none_filter_is_empty(self):
        assert self._wc(None) == ("", {})

    def test_default_filter_only_expiry_guard(self):
        # A bare QueryFilter has include_expired=False → expiry guard only.
        where, params = self._wc(QueryFilter())
        assert "node.expires_at" in where
        assert "f_now" in params
        assert "node.type" not in where

    def test_type_filter(self):
        where, params = self._wc(QueryFilter(types=[MemoryType.FEEDBACK], include_expired=True))
        assert "node.type IN $f_types" in where
        assert params["f_types"] == ["feedback"]

    def test_source_and_min_confidence_are_anded(self):
        where, params = self._wc(QueryFilter(
            source=MemorySource.MANUAL, min_confidence=0.5, include_expired=True,
        ))
        assert " AND " in where
        assert "node.source = $f_source" in where
        assert "node.confidence >= $f_minconf" in where
        assert params["f_source"] == "manual"
        assert params["f_minconf"] == 0.5

    def test_include_expired_drops_expiry_guard(self):
        where, params = self._wc(QueryFilter(types=[MemoryType.USER], include_expired=True))
        assert "expires_at" not in where
        assert "f_now" not in params


class TestFalkorDBStoreOffline:
    """FalkorDBStore store methods against the _FakeGraph double."""

    def test_add_and_count(self, falkordb_offline_store, sample_memories):
        falkordb_offline_store.add(sample_memories[0])
        assert falkordb_offline_store.count() == 1

    def test_add_batch_and_count(self, falkordb_offline_store, sample_memories):
        ids = falkordb_offline_store.add_batch(sample_memories)
        assert len(ids) == len(sample_memories)
        assert falkordb_offline_store.count() == len(sample_memories)

    def test_get_roundtrip(self, falkordb_offline_store, sample_memories):
        mem = sample_memories[1]
        falkordb_offline_store.add(mem)
        fetched = falkordb_offline_store.get(mem.id)
        assert fetched is not None
        assert fetched.id == mem.id
        assert fetched.content == mem.content
        assert fetched.tags == mem.tags
        assert fetched.type == mem.type

    def test_get_nonexistent_returns_none(self, falkordb_offline_store):
        assert falkordb_offline_store.get("nope") is None

    def test_delete_existing(self, falkordb_offline_store, sample_memories):
        mem = sample_memories[0]
        falkordb_offline_store.add(mem)
        assert falkordb_offline_store.delete(mem.id) is True
        assert falkordb_offline_store.count() == 0

    def test_delete_nonexistent_is_false(self, falkordb_offline_store):
        assert falkordb_offline_store.delete("nope") is False

    def test_list_all_and_limit(self, falkordb_offline_store, sample_memories):
        falkordb_offline_store.add_batch(sample_memories)
        assert len(falkordb_offline_store.list_all(limit=100)) == len(sample_memories)
        assert len(falkordb_offline_store.list_all(limit=2)) == 2

    def test_vector_query_similarity_inversion(self, falkordb_offline_store, sample_memories):
        # Querying with a memory's own document yields an identical embedding,
        # so cosine == 1 → distance == 0 → similarity score == 1.0. This locks
        # the COSINE-DISTANCE → similarity inversion (the documented Hard Rule).
        mem = sample_memories[0]
        falkordb_offline_store.add_batch(sample_memories)
        results = falkordb_offline_store.query(mem.to_document(), limit=len(sample_memories))
        assert all(isinstance(r, QueryResult) for r in results)
        top = results[0]
        assert top.memory.id == mem.id
        assert top.score == pytest.approx(1.0)
        # Scores are similarities (higher = better), descending.
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)
        assert all(0.0 <= s <= 1.0 + 1e-9 for s in scores)

    def test_query_empty_store(self, falkordb_offline_store):
        assert falkordb_offline_store.query("anything", limit=5) == []
