"""FalkorDB backend for SkillMind memory store.

FalkorDB is a Redis-based graph database with a built-in vector index. It is the
only SkillMind backend that combines vector similarity search (classic RAG) with
graph traversal, which enables GraphRAG retrieval on top of the same store.

This module implements the full ``MemoryStore`` contract using the vector index
(``query`` = vector KNN, drop-in compatible with the other backends). Each memory
is also linked to deterministic ``:Topic`` and ``:Tag`` nodes via ``HAS_TOPIC`` /
``HAS_TAG`` edges, so the knowledge graph is already populated for the GraphRAG
retrieval mode added on top of this skeleton.

Verified against falkordb/falkordb (graph module v41809, SDK falkordb==1.6.1):
the vector index ``score`` is a COSINE DISTANCE (identical vectors -> ~0), so we
expose ``similarity = 1 - distance`` to satisfy the QueryResult contract
(higher = more relevant).
"""

from __future__ import annotations

import json
import math
import re
from datetime import datetime
from typing import Any
from urllib.parse import urlparse

from ..config import SkillMindConfig
from ..embeddings import EmbeddingEngine
from ..models import Memory, MemorySource, MemoryType, QueryFilter, QueryResult
from .base import MemoryStore

# Scalar properties projected back from a :Memory node (embedding intentionally
# excluded to avoid transferring the full vector on every read).
_NODE_FIELDS = (
    "id", "type", "topic", "title", "content", "tags",
    "source", "confidence", "created_at", "updated_at", "expires_at",
    "metadata_json",
)

# GraphRAG re-rank weights (see _rerank). The vector similarity stays the
# dominant term; the graph boost only lifts items that are connected to a
# strong seed, so GraphRAG returns the vector hits PLUS connected memories.
#
# Shared topic/tag connections are weighted by the INVERSE DEGREE of the shared
# node: sharing a rare tag is a strong signal, sharing a hub tag (e.g. "youtube"
# on 18 memories) is almost noise. Without this, popular attribute nodes connect
# to every seed and their neighbours swamp the ranking (the hub problem).
_GRAPH_WEIGHT = 0.25          # how much the (normalized) graph boost adds on top of vec sim
_ATTR_WEIGHT = 1.0            # applied to the inverse-degree sum of shared attributes
_RELATES_WEIGHT = 0.5         # per RELATES_TO connection at hop 1
_HOP_DISCOUNT = 0.5           # multiplier per extra RELATES_TO hop
_BOOST_CAP = 1.0              # graph boost is capped so it can't bury a strong vector hit

# Matches Obsidian-style [[wiki links]] used in structured memories. The current
# auto-captured memories carry none, but build_graph() turns any it finds into
# RELATES_TO edges, so structured memories link up automatically once imported.
_WIKILINK_RE = re.compile(r"\[\[([^\]\|]+?)(?:\|[^\]]*)?\]\]")


def _return_projection(var: str = "node") -> str:
    """Cypher RETURN projection of the scalar memory fields, in _NODE_FIELDS order."""
    return ", ".join(f"{var}.{f}" for f in _NODE_FIELDS)


def _parse_wikilinks(content: str) -> list[str]:
    """Extract ``[[name]]`` link targets from memory content (pure, offline-testable).

    Handles ``[[name]]`` and ``[[name|alias]]``; returns de-duplicated, trimmed
    target names in first-seen order.
    """
    seen: dict[str, None] = {}
    for raw in _WIKILINK_RE.findall(content or ""):
        name = raw.strip()
        if name:
            seen.setdefault(name, None)
    return list(seen)


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two equal-length vectors (0 if either is empty/zero)."""
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _rerank(
    vec_sim: dict[str, float],
    graph_boost: dict[str, float],
    confidence: dict[str, float],
    limit: int,
) -> list[tuple[str, float]]:
    """Combine vector similarity, graph proximity and confidence into a ranking.

    Pure function (no DB) so the scoring is unit-testable offline. ``vec_sim`` and
    ``graph_boost`` are keyed by memory id; the candidate set is their union. The
    final score is ``(vec_sim + GRAPH_WEIGHT * graph_boost)`` gently weighted by
    confidence, so a graph-connected memory can outrank a weak pure-vector hit
    while strong vector hits stay on top. Returns ``(id, score)`` sorted desc.
    """
    candidates = set(vec_sim) | set(graph_boost)
    scored: list[tuple[str, float]] = []
    for cid in candidates:
        boost = min(graph_boost.get(cid, 0.0), _BOOST_CAP)
        base = vec_sim.get(cid, 0.0) + _GRAPH_WEIGHT * boost
        conf = confidence.get(cid, 1.0)
        scored.append((cid, base * (0.5 + 0.5 * conf)))
    scored.sort(key=lambda t: t[1], reverse=True)
    return scored[:limit]


class FalkorDBStore(MemoryStore):
    """FalkorDB-backed memory store (graph + vector, supports GraphRAG)."""

    def __init__(self, config: SkillMindConfig, engine: EmbeddingEngine):
        super().__init__(config, engine)
        self._db: Any = None
        self._graph: Any = None
        self._graph_name = config.store.falkordb_graph

    # ------------------------------------------------------------------ setup

    def initialize(self) -> None:
        from falkordb import FalkorDB

        url = self.config.store.falkordb_url
        parsed = urlparse(url)
        host = parsed.hostname or "localhost"
        port = parsed.port or 6379
        password = parsed.password or self.config.store.falkordb_password or None

        self._db = FalkorDB(host=host, port=port, password=password)
        self._graph = self._db.select_graph(self._graph_name)
        self._create_vector_index()

    def _create_vector_index(self) -> None:
        """Create the vector index once (CREATE VECTOR INDEX errors if it exists)."""
        dim = self.engine.dimension
        try:
            self._graph.query(
                "CREATE VECTOR INDEX FOR (m:Memory) ON (m.embedding) "
                f"OPTIONS {{dimension: {dim}, similarityFunction: 'cosine'}}"
            )
        except Exception as exc:  # noqa: BLE001 - index-already-exists is expected
            if "already" not in str(exc).lower() and "exist" not in str(exc).lower():
                raise

    def _drop_vector_index(self) -> None:
        """Drop the vector index if present (no-op when it does not exist)."""
        try:
            self._graph.query("DROP VECTOR INDEX FOR (m:Memory) ON (m.embedding)")
        except Exception as exc:  # noqa: BLE001 - index-missing is expected
            if "not" not in str(exc).lower() and "exist" not in str(exc).lower():
                raise

    @property
    def graph(self) -> Any:
        if self._graph is None:
            self.initialize()
        return self._graph

    def _run(self, query: str, params: dict | None = None) -> Any:
        return self.graph.query(query, params=params or {})

    # -------------------------------------------------------------------- write

    def add(self, memory: Memory) -> str:
        embedding = self.engine.embed(memory.to_document())
        self._upsert(memory, embedding)
        return memory.id

    def add_batch(self, memories: list[Memory]) -> list[str]:
        if not memories:
            return []
        embeddings = self.engine.embed_batch([m.to_document() for m in memories])
        for memory, embedding in zip(memories, embeddings):
            self._upsert(memory, embedding)
        return [m.id for m in memories]

    def _upsert(self, memory: Memory, embedding: list[float]) -> None:
        params = self._memory_to_params(memory)
        params["embedding"] = embedding

        # Upsert the memory node + its vector, then the deterministic topic edge.
        self._run(
            "MERGE (m:Memory {id:$id}) "
            "SET m.type=$type, m.topic=$topic, m.title=$title, m.content=$content, "
            "    m.tags=$tags, m.source=$source, m.confidence=$confidence, "
            "    m.created_at=$created_at, m.updated_at=$updated_at, "
            "    m.expires_at=$expires_at, m.metadata_json=$metadata_json, "
            "    m.embedding=vecf32($embedding) "
            "WITH m "
            "MERGE (t:Topic {name:$topic}) MERGE (m)-[:HAS_TOPIC]->(t)",
            params,
        )

        # Tag edges (UNWIND on an empty list is a no-op, so guard in Python).
        if memory.tags:
            self._run(
                "MATCH (m:Memory {id:$id}) "
                "UNWIND $tags AS tag "
                "MERGE (tg:Tag {name:tag}) MERGE (m)-[:HAS_TAG]->(tg)",
                {"id": memory.id, "tags": memory.tags},
            )

    def update(self, memory: Memory) -> None:
        memory.updated_at = datetime.utcnow()
        self.add(memory)  # MERGE handles upsert

    def delete(self, memory_id: str) -> bool:
        res = self._run(
            "MATCH (m:Memory {id:$id}) DETACH DELETE m",
            {"id": memory_id},
        )
        return getattr(res, "nodes_deleted", 0) > 0

    def clear(self) -> int:
        n = self.count()
        # The graph is dedicated to SkillMind, so wipe every node (Memory + derived
        # Topic/Tag/Entity). A persisted vector index does NOT reliably re-index
        # nodes inserted after a bulk DETACH DELETE, so drop and recreate it to
        # guarantee the store is queryable again after a clear.
        self._run("MATCH (n) DETACH DELETE n")
        self._drop_vector_index()
        self._create_vector_index()
        return n

    # --------------------------------------------------------------------- read

    def query(
        self,
        text: str,
        limit: int = 5,
        filter: QueryFilter | None = None,
    ) -> list[QueryResult]:
        """Semantic search. GraphRAG multi-hop when ``store.falkordb_graphrag``."""
        if self.config.store.falkordb_graphrag:
            return self._graphrag_query(text, limit, filter)
        return self._vector_query(text, limit, filter)

    def _vector_query(
        self,
        text: str,
        limit: int = 5,
        filter: QueryFilter | None = None,
    ) -> list[QueryResult]:
        """Plain vector KNN — drop-in identical to Pinecone & co (RAG path)."""
        embedding = self.engine.embed(text)
        where, params = self._where_clause(filter, "node")

        # Over-fetch when filtering, since the KNN runs before the WHERE prunes.
        k = limit if not where else min(limit * 10, 1000)
        params.update({"k": k, "v": embedding, "limit": limit})

        cypher = (
            "CALL db.idx.vector.queryNodes('Memory','embedding',$k, vecf32($v)) "
            "YIELD node, score "
            f"{where} "
            f"RETURN {_return_projection('node')}, score "
            "ORDER BY score ASC "
            "LIMIT $limit"
        )
        res = self._run(cypher, params)

        results: list[QueryResult] = []
        for row in res.result_set:
            *fields, distance = row
            memory = self._row_to_memory(fields)
            # score = cosine distance -> similarity (higher = more relevant)
            results.append(QueryResult(memory=memory, score=1.0 - float(distance)))
        return results

    def _graphrag_query(
        self,
        text: str,
        limit: int,
        filter: QueryFilter | None,
    ) -> list[QueryResult]:
        """GraphRAG: vector seeds -> graph expansion -> combined re-rank.

        1. Seed: vector KNN over a generous pool (covers the plain-RAG hits).
        2. Expand: from the top ``seed_k`` seeds, walk shared topics/tags and
           RELATES_TO edges (up to ``falkordb_hops``) to pull in connected memories.
        3. Re-rank: ``_rerank`` blends vector similarity, graph proximity and
           confidence. Graph-only candidates get a real vector score from their
           stored embedding so they compete fairly.
        4. Assemble the top ``limit`` as QueryResult.
        """
        store = self.config.store
        seed_k = max(1, store.falkordb_seed_k)
        hops = max(1, min(store.falkordb_hops, 3))

        q_vec = self.engine.embed(text)
        where, fparams = self._where_clause(filter, "node")

        # Seed pool: enough vector hits to act as a solid RAG baseline.
        pool = min(max(seed_k, limit) * 8, 500)
        sparams = {"k": pool, "v": q_vec, "limit": pool, **fparams}
        res = self._run(
            "CALL db.idx.vector.queryNodes('Memory','embedding',$k, vecf32($v)) "
            "YIELD node, score "
            f"{where} "
            f"RETURN {_return_projection('node')}, score "
            "ORDER BY score ASC LIMIT $limit",
            sparams,
        )

        memories: dict[str, Memory] = {}
        vec_sim: dict[str, float] = {}
        confidence: dict[str, float] = {}
        ordered_ids: list[str] = []
        for row in res.result_set:
            *fields, distance = row
            mem = self._row_to_memory(fields)
            memories[mem.id] = mem
            vec_sim[mem.id] = 1.0 - float(distance)
            confidence[mem.id] = mem.confidence
            ordered_ids.append(mem.id)

        if not ordered_ids:
            return []

        seed_ids = ordered_ids[:seed_k]
        graph_boost = self._expand(seed_ids, hops, vec_sim)

        # Fetch memories/embeddings for graph-only candidates (small set) so they
        # get a fair vector score instead of relying on the graph boost alone.
        missing = [cid for cid in graph_boost if cid not in memories]
        if missing:
            mres = self._run(
                f"MATCH (node:Memory) WHERE node.id IN $ids "
                f"RETURN {_return_projection('node')}, node.embedding",
                {"ids": missing},
            )
            for row in mres.result_set:
                *fields, emb = row
                mem = self._row_to_memory(fields)
                memories[mem.id] = mem
                confidence[mem.id] = mem.confidence
                vec_sim[mem.id] = max(0.0, _cosine(q_vec, list(emb or [])))

        ranked = _rerank(vec_sim, graph_boost, confidence, limit)
        return [
            QueryResult(memory=memories[cid], score=score)
            for cid, score in ranked
            if cid in memories
        ]

    def _expand(
        self,
        seed_ids: list[str],
        hops: int,
        seed_sim: dict[str, float],
    ) -> dict[str, float]:
        """Walk the graph from seeds, returning {candidate_id: graph_boost}.

        Two connection types, each weighted by the originating seed's similarity
        to the query (so connections from strong seeds count more):
        - shared :Topic / :Tag, weighted by the INVERSE DEGREE of the shared node
          (a rare tag is a strong signal; a hub tag shared by dozens of memories
          contributes almost nothing — this defeats the hub problem);
        - :RELATES_TO chains up to ``hops``, discounted per extra hop.

        The accumulated boost is normalized by the number of seeds, so it stays an
        average per-seed influence (roughly bounded) rather than growing with the
        seed count and overwhelming the dominant vector similarity term.
        """
        boost: dict[str, float] = {}

        # Degree of each attribute reachable from the seeds (how many memories
        # share it). Looked up below so common tags are down-weighted.
        deg_res = self._run(
            "MATCH (s:Memory) WHERE s.id IN $ids "
            "MATCH (s)-[:HAS_TOPIC|HAS_TAG]->(a) "
            "WITH DISTINCT a "
            "MATCH (a)<-[:HAS_TOPIC|HAS_TAG]-(m:Memory) "
            "RETURN id(a), count(m)",
            {"ids": seed_ids},
        )
        degree = {aid: max(1, int(cnt)) for aid, cnt in deg_res.result_set}

        # Shared topic/tag siblings (Memory -> attr <- Memory), inverse-degree weighted.
        attr = self._run(
            "MATCH (s:Memory) WHERE s.id IN $ids "
            "MATCH (s)-[:HAS_TOPIC|HAS_TAG]->(a)<-[:HAS_TOPIC|HAS_TAG]-(r:Memory) "
            "WHERE r.id <> s.id "
            "RETURN s.id, r.id, id(a)",
            {"ids": seed_ids},
        )
        for seed, cand, aid in attr.result_set:
            sim = seed_sim.get(seed, 0.0)
            inv = 1.0 / degree.get(aid, 1)
            boost[cand] = boost.get(cand, 0.0) + sim * _ATTR_WEIGHT * inv

        # RELATES_TO chains (hops validated to 1..3, safe to interpolate).
        rel = self._run(
            "MATCH (s:Memory) WHERE s.id IN $ids "
            f"MATCH p = (s)-[:RELATES_TO*1..{int(hops)}]-(r:Memory) "
            "WHERE r.id <> s.id "
            "RETURN s.id, r.id, min(length(p))",
            {"ids": seed_ids},
        )
        for seed, cand, dist in rel.result_set:
            sim = seed_sim.get(seed, 0.0)
            boost[cand] = boost.get(cand, 0.0) + (
                sim * _RELATES_WEIGHT * (_HOP_DISCOUNT ** (int(dist) - 1))
            )

        # Normalize by seed count so the boost is an average per-seed influence
        # rather than scaling with how many seeds happen to connect.
        n_seeds = max(1, len(seed_ids))
        for cid in boost:
            boost[cid] /= n_seeds

        # Seeds themselves are already in vec_sim; don't double-count.
        for sid in seed_ids:
            boost.pop(sid, None)
        return boost

    def build_graph(
        self,
        similarity_threshold: float = 0.78,
        max_neighbors: int = 6,
        rebuild: bool = False,
    ) -> dict[str, int]:
        """Materialize :RELATES_TO edges for GraphRAG (idempotent).

        Two sources, matching the design doc:
        - ``[[name]]`` wiki links in content -> RELATES_TO {kind:'link'} when the
          target resolves to a memory id, title or metadata ``name``;
        - embedding similarity: each memory's top ``max_neighbors`` neighbors with
          similarity >= ``similarity_threshold`` -> RELATES_TO {kind:'semantic',
          weight: sim}.

        HAS_TOPIC/HAS_TAG edges already exist from add(); this only adds the
        memory-to-memory layer. Returns counts of edges created per kind.
        Set ``rebuild=True`` to drop existing RELATES_TO edges first.
        """
        if rebuild:
            self._run("MATCH ()-[r:RELATES_TO]->() DELETE r")

        memories = self.list_all(limit=10000)
        counts = {"link": 0, "semantic": 0}

        # Build lookup for [[name]] resolution: id, lowercased title, metadata name.
        by_key: dict[str, str] = {}
        for m in memories:
            by_key[m.id] = m.id
            if m.title:
                by_key.setdefault(m.title.strip().lower(), m.id)
            meta_name = (m.metadata or {}).get("name")
            if meta_name:
                by_key.setdefault(str(meta_name).strip().lower(), m.id)

        # 1. Wiki-link edges.
        for m in memories:
            for target in _parse_wikilinks(m.content):
                tid = by_key.get(target) or by_key.get(target.lower())
                if tid and tid != m.id:
                    self._run(
                        "MATCH (a:Memory {id:$a}), (b:Memory {id:$b}) "
                        "MERGE (a)-[e:RELATES_TO]->(b) "
                        "SET e.kind='link', e.weight=1.0",
                        {"a": m.id, "b": tid},
                    )
                    counts["link"] += 1

        # 2. Semantic-similarity edges (top-k neighbors above the threshold).
        for m in memories:
            neighbors = self._vector_query(m.to_document(), limit=max_neighbors + 1)
            for r in neighbors:
                if r.memory.id == m.id or r.score < similarity_threshold:
                    continue
                # Undirected intent: store one direction, MERGE keeps it idempotent.
                self._run(
                    "MATCH (a:Memory {id:$a}), (b:Memory {id:$b}) "
                    "MERGE (a)-[e:RELATES_TO]->(b) "
                    "SET e.kind=coalesce(e.kind,'semantic'), "
                    "    e.weight=CASE WHEN e.weight IS NULL THEN $w "
                    "             ELSE (CASE WHEN e.weight>$w THEN e.weight ELSE $w END) END",
                    {"a": m.id, "b": r.memory.id, "w": float(r.score)},
                )
                counts["semantic"] += 1

        return counts

    def link_sequence(
        self,
        ordered_ids: list[str],
        rel: str = "NEXT",
        group_key: str | None = None,
    ) -> int:
        """Chain memories in a fixed order via directed ``:NEXT`` edges.

        Used to preserve the original reading order of a sequence — e.g. the
        chapters of a YouTube video — so a graph traversal can walk the content
        forwards/backwards. Idempotent (MERGE). ``group_key`` (e.g. a video id)
        is stamped on each edge so several sequences can share the graph without
        their chains getting tangled. Returns the number of edges created.
        """
        ids = [i for i in ordered_ids if i]
        if len(ids) < 2:
            return 0
        rel = re.sub(r"[^A-Za-z_]", "", rel) or "NEXT"
        created = 0
        for a, b in zip(ids, ids[1:]):
            if a == b:
                continue
            self._run(
                f"MATCH (a:Memory {{id:$a}}), (b:Memory {{id:$b}}) "
                f"MERGE (a)-[e:{rel}]->(b) "
                f"SET e.group_key=$g",
                {"a": a, "b": b, "g": group_key or ""},
            )
            created += 1
        return created

    def get(self, memory_id: str) -> Memory | None:
        res = self._run(
            f"MATCH (node:Memory {{id:$id}}) RETURN {_return_projection('node')}",
            {"id": memory_id},
        )
        if not res.result_set:
            return None
        return self._row_to_memory(res.result_set[0])

    def list_all(
        self,
        filter: QueryFilter | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Memory]:
        where, params = self._where_clause(filter, "node")
        params.update({"limit": limit, "offset": offset})
        res = self._run(
            f"MATCH (node:Memory) {where} "
            f"RETURN {_return_projection('node')} "
            "ORDER BY node.created_at DESC SKIP $offset LIMIT $limit",
            params,
        )
        return [self._row_to_memory(row) for row in res.result_set]

    def count(self, filter: QueryFilter | None = None) -> int:
        where, params = self._where_clause(filter, "node")
        res = self._run(
            f"MATCH (node:Memory) {where} RETURN count(node)",
            params,
        )
        if not res.result_set:
            return 0
        return int(res.result_set[0][0])

    # ------------------------------------------------------------- filter / map

    def _where_clause(self, filter: QueryFilter | None, var: str) -> tuple[str, dict]:
        """Build a Cypher WHERE fragment + params from a QueryFilter.

        Returns ("" , {}) when no conditions apply. Mirrors the metadata filtering
        used by the other backends (tags are intentionally not filtered, matching
        base._build_where_filter / QdrantStore).
        """
        if not filter:
            return "", {}

        conditions: list[str] = []
        params: dict[str, Any] = {}

        if filter.types:
            params["f_types"] = [t.value for t in filter.types]
            conditions.append(f"{var}.type IN $f_types")
        if filter.topics:
            params["f_topics"] = filter.topics
            conditions.append(f"{var}.topic IN $f_topics")
        if filter.source:
            params["f_source"] = filter.source.value
            conditions.append(f"{var}.source = $f_source")
        if filter.min_confidence > 0:
            params["f_minconf"] = filter.min_confidence
            conditions.append(f"{var}.confidence >= $f_minconf")
        if not filter.include_expired:
            params["f_now"] = datetime.utcnow().isoformat()
            # expires_at is stored as "" when unset; ISO strings compare correctly.
            conditions.append(f"({var}.expires_at = '' OR {var}.expires_at >= $f_now)")

        if not conditions:
            return "", params
        return "WHERE " + " AND ".join(conditions), params

    @staticmethod
    def _memory_to_params(memory: Memory) -> dict:
        return {
            "id": memory.id,
            "type": memory.type.value,
            "topic": memory.topic,
            "title": memory.title,
            "content": memory.content,
            "tags": memory.tags,
            "source": memory.source.value,
            "confidence": memory.confidence,
            "created_at": memory.created_at.isoformat(),
            "updated_at": memory.updated_at.isoformat(),
            "expires_at": memory.expires_at.isoformat() if memory.expires_at else "",
            "metadata_json": json.dumps(memory.metadata or {}),
        }

    @staticmethod
    def _row_to_memory(row: list) -> Memory:
        """Map a row projected in _NODE_FIELDS order back to a Memory."""
        data = dict(zip(_NODE_FIELDS, row))
        expires = data.get("expires_at") or ""
        meta_raw = data.get("metadata_json") or "{}"
        try:
            metadata = json.loads(meta_raw)
        except (ValueError, TypeError):
            metadata = {}
        return Memory(
            id=str(data["id"]),
            type=MemoryType(data.get("type") or "user"),
            topic=data.get("topic") or "",
            title=data.get("title") or "",
            content=data.get("content") or "",
            tags=list(data.get("tags") or []),
            source=MemorySource(data.get("source") or "manual"),
            confidence=float(data["confidence"]) if data.get("confidence") is not None else 1.0,
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.utcnow(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else datetime.utcnow(),
            expires_at=datetime.fromisoformat(expires) if expires else None,
            metadata=metadata,
        )
