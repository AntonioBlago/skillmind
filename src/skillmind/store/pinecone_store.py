"""Pinecone backend for SkillMind memory store."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from ..config import SkillMindConfig
from ..embeddings import EmbeddingEngine
from ..models import Memory, MemoryType, MemorySource, QueryFilter, QueryResult
from .base import MemoryStore


class PineconeStore(MemoryStore):
    """
    Pinecone-backed memory store.

    Cloud-hosted, scales infinitely. Best for teams or multi-device sync.
    Requires PINECONE_API_KEY.
    """

    def __init__(self, config: SkillMindConfig, engine: EmbeddingEngine):
        super().__init__(config, engine)
        self._index: Any = None

    def initialize(self) -> None:
        from pinecone import Pinecone, ServerlessSpec

        pc = Pinecone(api_key=self.config.store.pinecone_api_key)
        index_name = self.config.store.pinecone_index

        # Create index if it doesn't exist
        existing = [idx.name for idx in pc.list_indexes()]
        if index_name not in existing:
            pc.create_index(
                name=index_name,
                dimension=self.engine.dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )

        self._index = pc.Index(index_name)

    @property
    def index(self) -> Any:
        if self._index is None:
            self.initialize()
        return self._index

    def add(self, memory: Memory) -> str:
        embedding = self.engine.embed(memory.to_document())
        meta = memory.to_metadata_dict()
        meta["content"] = memory.content  # Pinecone doesn't store documents separately

        self.index.upsert(vectors=[(memory.id, embedding, meta)])
        return memory.id

    def add_batch(self, memories: list[Memory]) -> list[str]:
        if not memories:
            return []
        docs = [m.to_document() for m in memories]
        embeddings = self.engine.embed_batch(docs)

        vectors = []
        for mem, emb in zip(memories, embeddings):
            meta = mem.to_metadata_dict()
            meta["content"] = mem.content
            vectors.append((mem.id, emb, meta))

        # Pinecone batch limit is 100
        for i in range(0, len(vectors), 100):
            self.index.upsert(vectors=vectors[i : i + 100])

        return [m.id for m in memories]

    def query(
        self,
        text: str,
        limit: int = 5,
        filter: QueryFilter | None = None,
    ) -> list[QueryResult]:
        embedding = self.engine.embed(text)
        pc_filter = self._to_pinecone_filter(filter)

        kwargs: dict[str, Any] = {
            "vector": embedding,
            "top_k": limit,
            "include_metadata": True,
        }
        if pc_filter:
            kwargs["filter"] = pc_filter

        results = self.index.query(**kwargs)

        query_results: list[QueryResult] = []
        for match in results.get("matches", []):
            meta = match.get("metadata", {})
            content = meta.pop("content", "")
            memory = self._meta_to_memory(match["id"], content, meta)
            query_results.append(QueryResult(memory=memory, score=match["score"]))

        return query_results

    def get(self, memory_id: str) -> Memory | None:
        result = self.index.fetch(ids=[memory_id])
        vectors = result.get("vectors", {})
        if memory_id in vectors:
            meta = vectors[memory_id].get("metadata", {})
            content = meta.pop("content", "")
            return self._meta_to_memory(memory_id, content, meta)
        return None

    def update(self, memory: Memory) -> None:
        memory.updated_at = datetime.utcnow()
        embedding = self.engine.embed(memory.to_document())
        meta = memory.to_metadata_dict()
        meta["content"] = memory.content
        self.index.upsert(vectors=[(memory.id, embedding, meta)])

    def delete(self, memory_id: str) -> bool:
        try:
            self.index.delete(ids=[memory_id])
            return True
        except Exception:
            return False

    def list_all(
        self,
        filter: QueryFilter | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Memory]:
        # Pinecone doesn't support list without query — use zero vector
        zero_vec = [0.0] * self.engine.dimension
        pc_filter = self._to_pinecone_filter(filter)

        kwargs: dict[str, Any] = {
            "vector": zero_vec,
            "top_k": limit + offset,
            "include_metadata": True,
        }
        if pc_filter:
            kwargs["filter"] = pc_filter

        results = self.index.query(**kwargs)
        memories: list[Memory] = []
        for match in results.get("matches", [])[offset:]:
            meta = match.get("metadata", {})
            content = meta.pop("content", "")
            memories.append(self._meta_to_memory(match["id"], content, meta))

        return memories

    def count(self, filter: QueryFilter | None = None) -> int:
        if filter is None:
            stats = self.index.describe_index_stats()
            return stats.get("total_vector_count", 0)
        return len(self.list_all(filter=filter, limit=10000))

    def clear(self) -> int:
        n = self.count()
        self.index.delete(delete_all=True)
        return n

    @staticmethod
    def _to_pinecone_filter(filter: QueryFilter | None) -> dict | None:
        """Convert QueryFilter to Pinecone filter syntax."""
        if not filter:
            return None

        conditions: dict[str, Any] = {}

        if filter.types:
            conditions["type"] = {"$in": [t.value for t in filter.types]}
        if filter.topics:
            conditions["topic"] = {"$in": filter.topics}
        if filter.source:
            conditions["source"] = {"$eq": filter.source.value}
        if filter.min_confidence > 0:
            conditions["confidence"] = {"$gte": filter.min_confidence}

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions
        return {"$and": [{k: v} for k, v in conditions.items()]}

    @staticmethod
    def _meta_to_memory(memory_id: str, content: str, meta: dict) -> Memory:
        tags = meta.get("tags", "")
        return Memory(
            id=memory_id,
            type=MemoryType(meta.get("type", "user")),
            topic=meta.get("topic", ""),
            title=meta.get("title", ""),
            content=content,
            tags=tags.split(",") if tags else [],
            source=MemorySource(meta.get("source", "manual")),
            confidence=float(meta.get("confidence", 1.0)),
            created_at=datetime.fromisoformat(meta["created_at"]) if meta.get("created_at") else datetime.utcnow(),
            updated_at=datetime.fromisoformat(meta["updated_at"]) if meta.get("updated_at") else datetime.utcnow(),
            expires_at=datetime.fromisoformat(meta["expires_at"]) if meta.get("expires_at") else None,
        )
