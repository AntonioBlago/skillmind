"""Qdrant backend for SkillMind memory store."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from ..config import SkillMindConfig
from ..embeddings import EmbeddingEngine
from ..models import Memory, MemoryType, MemorySource, QueryFilter, QueryResult
from .base import MemoryStore


class QdrantStore(MemoryStore):
    """
    Qdrant-backed memory store.

    Local or cloud, excellent filtering, high performance.
    Supports both self-hosted (free) and Qdrant Cloud.
    """

    def __init__(self, config: SkillMindConfig, engine: EmbeddingEngine):
        super().__init__(config, engine)
        self._client: Any = None
        self._collection = config.store.qdrant_collection

    def initialize(self) -> None:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        if self.config.store.qdrant_api_key:
            self._client = QdrantClient(
                url=self.config.store.qdrant_url,
                api_key=self.config.store.qdrant_api_key,
            )
        else:
            self._client = QdrantClient(url=self.config.store.qdrant_url)

        # Create collection if not exists
        collections = [c.name for c in self._client.get_collections().collections]
        if self._collection not in collections:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(
                    size=self.engine.dimension,
                    distance=Distance.COSINE,
                ),
            )

    @property
    def client(self) -> Any:
        if self._client is None:
            self.initialize()
        return self._client

    def add(self, memory: Memory) -> str:
        from qdrant_client.models import PointStruct

        embedding = self.engine.embed(memory.to_document())
        payload = self._memory_to_payload(memory)

        self.client.upsert(
            collection_name=self._collection,
            points=[
                PointStruct(
                    id=memory.id,
                    vector=embedding,
                    payload=payload,
                )
            ],
        )
        return memory.id

    def add_batch(self, memories: list[Memory]) -> list[str]:
        if not memories:
            return []
        from qdrant_client.models import PointStruct

        docs = [m.to_document() for m in memories]
        embeddings = self.engine.embed_batch(docs)

        points = [
            PointStruct(
                id=m.id,
                vector=emb,
                payload=self._memory_to_payload(m),
            )
            for m, emb in zip(memories, embeddings)
        ]

        # Qdrant batch limit ~100 points per upsert
        for i in range(0, len(points), 100):
            self.client.upsert(
                collection_name=self._collection,
                points=points[i : i + 100],
            )
        return [m.id for m in memories]

    def query(
        self,
        text: str,
        limit: int = 5,
        filter: QueryFilter | None = None,
    ) -> list[QueryResult]:
        embedding = self.engine.embed(text)
        qdrant_filter = self._to_qdrant_filter(filter)

        kwargs: dict[str, Any] = {
            "collection_name": self._collection,
            "query_vector": embedding,
            "limit": limit,
            "with_payload": True,
        }
        if qdrant_filter:
            kwargs["query_filter"] = qdrant_filter

        results = self.client.search(**kwargs)

        query_results: list[QueryResult] = []
        for hit in results:
            memory = self._payload_to_memory(hit.id, hit.payload)
            query_results.append(QueryResult(memory=memory, score=hit.score))

        return query_results

    def get(self, memory_id: str) -> Memory | None:
        results = self.client.retrieve(
            collection_name=self._collection,
            ids=[memory_id],
            with_payload=True,
        )
        if results:
            return self._payload_to_memory(results[0].id, results[0].payload)
        return None

    def update(self, memory: Memory) -> None:
        memory.updated_at = datetime.utcnow()
        self.add(memory)  # Qdrant upsert handles updates

    def delete(self, memory_id: str) -> bool:
        from qdrant_client.models import PointIdsList

        try:
            self.client.delete(
                collection_name=self._collection,
                points_selector=PointIdsList(points=[memory_id]),
            )
            return True
        except Exception:
            return False

    def list_all(
        self,
        filter: QueryFilter | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Memory]:
        qdrant_filter = self._to_qdrant_filter(filter)

        kwargs: dict[str, Any] = {
            "collection_name": self._collection,
            "limit": limit,
            "offset": offset,
            "with_payload": True,
        }
        if qdrant_filter:
            kwargs["scroll_filter"] = qdrant_filter

        results, _ = self.client.scroll(**kwargs)

        return [self._payload_to_memory(r.id, r.payload) for r in results]

    def count(self, filter: QueryFilter | None = None) -> int:
        if filter is None:
            info = self.client.get_collection(self._collection)
            return info.points_count
        return len(self.list_all(filter=filter, limit=100000))

    def clear(self) -> int:
        n = self.count()
        self.client.delete_collection(self._collection)
        self.initialize()  # Recreate empty collection
        return n

    @staticmethod
    def _to_qdrant_filter(filter: QueryFilter | None) -> Any:
        if not filter:
            return None

        from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue, Range

        conditions: list[Any] = []

        if filter.types:
            conditions.append(
                FieldCondition(
                    key="type",
                    match=MatchAny(any=[t.value for t in filter.types]),
                )
            )
        if filter.topics:
            conditions.append(
                FieldCondition(
                    key="topic",
                    match=MatchAny(any=filter.topics),
                )
            )
        if filter.source:
            conditions.append(
                FieldCondition(
                    key="source",
                    match=MatchValue(value=filter.source.value),
                )
            )
        if filter.min_confidence > 0:
            conditions.append(
                FieldCondition(
                    key="confidence",
                    range=Range(gte=filter.min_confidence),
                )
            )

        if not conditions:
            return None
        return Filter(must=conditions)

    @staticmethod
    def _memory_to_payload(memory: Memory) -> dict:
        return {
            "type": memory.type.value,
            "topic": memory.topic,
            "title": memory.title,
            "content": memory.content,
            "tags": memory.tags,
            "source": memory.source.value,
            "confidence": memory.confidence,
            "created_at": memory.created_at.isoformat(),
            "updated_at": memory.updated_at.isoformat(),
            "expires_at": memory.expires_at.isoformat() if memory.expires_at else None,
        }

    @staticmethod
    def _payload_to_memory(memory_id: str, payload: dict) -> Memory:
        return Memory(
            id=str(memory_id),
            type=MemoryType(payload.get("type", "user")),
            topic=payload.get("topic", ""),
            title=payload.get("title", ""),
            content=payload.get("content", ""),
            tags=payload.get("tags") or [],
            source=MemorySource(payload.get("source", "manual")),
            confidence=float(payload.get("confidence", 1.0)),
            created_at=datetime.fromisoformat(payload["created_at"]) if payload.get("created_at") else datetime.utcnow(),
            updated_at=datetime.fromisoformat(payload["updated_at"]) if payload.get("updated_at") else datetime.utcnow(),
            expires_at=datetime.fromisoformat(payload["expires_at"]) if payload.get("expires_at") else None,
        )
