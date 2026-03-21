"""ChromaDB backend for SkillMind memory store."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from ..config import SkillMindConfig
from ..embeddings import EmbeddingEngine
from ..models import Memory, MemoryType, MemorySource, QueryFilter, QueryResult
from .base import MemoryStore


class ChromaStore(MemoryStore):
    """
    ChromaDB-backed memory store.

    Local, embedded, zero-cost. Best for solo developers.
    Handles ~100k memories easily.
    """

    def __init__(self, config: SkillMindConfig, engine: EmbeddingEngine):
        super().__init__(config, engine)
        self._client: Any = None
        self._collection: Any = None

    def initialize(self) -> None:
        import chromadb

        self._client = chromadb.PersistentClient(path=self.config.store.chroma_path)
        self._collection = self._client.get_or_create_collection(
            name="skillmind_memories",
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def collection(self) -> Any:
        if self._collection is None:
            self.initialize()
        return self._collection

    def add(self, memory: Memory) -> str:
        embedding = self.engine.embed(memory.to_document())
        self.collection.add(
            ids=[memory.id],
            embeddings=[embedding],
            documents=[memory.content],
            metadatas=[memory.to_metadata_dict()],
        )
        return memory.id

    def add_batch(self, memories: list[Memory]) -> list[str]:
        if not memories:
            return []
        docs = [m.to_document() for m in memories]
        embeddings = self.engine.embed_batch(docs)
        self.collection.add(
            ids=[m.id for m in memories],
            embeddings=embeddings,
            documents=[m.content for m in memories],
            metadatas=[m.to_metadata_dict() for m in memories],
        )
        return [m.id for m in memories]

    def query(
        self,
        text: str,
        limit: int = 5,
        filter: QueryFilter | None = None,
    ) -> list[QueryResult]:
        embedding = self.engine.embed(text)
        where = self._build_where_filter(filter)

        kwargs: dict[str, Any] = {
            "query_embeddings": [embedding],
            "n_results": limit,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = self.collection.query(**kwargs)

        query_results: list[QueryResult] = []
        if results and results["ids"] and results["ids"][0]:
            for i, mid in enumerate(results["ids"][0]):
                meta = results["metadatas"][0][i]
                distance = results["distances"][0][i]
                # Chroma cosine distance: 0 = identical, 2 = opposite
                score = max(0.0, 1.0 - distance / 2.0)

                memory = self._meta_to_memory(
                    mid, results["documents"][0][i], meta
                )
                query_results.append(QueryResult(memory=memory, score=score))

        return query_results

    def get(self, memory_id: str) -> Memory | None:
        try:
            result = self.collection.get(
                ids=[memory_id],
                include=["documents", "metadatas"],
            )
            if result and result["ids"]:
                return self._meta_to_memory(
                    result["ids"][0],
                    result["documents"][0],
                    result["metadatas"][0],
                )
        except Exception:
            pass
        return None

    def update(self, memory: Memory) -> None:
        memory.updated_at = datetime.utcnow()
        embedding = self.engine.embed(memory.to_document())
        self.collection.update(
            ids=[memory.id],
            embeddings=[embedding],
            documents=[memory.content],
            metadatas=[memory.to_metadata_dict()],
        )

    def delete(self, memory_id: str) -> bool:
        try:
            self.collection.delete(ids=[memory_id])
            return True
        except Exception:
            return False

    def list_all(
        self,
        filter: QueryFilter | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Memory]:
        where = self._build_where_filter(filter)
        kwargs: dict[str, Any] = {
            "include": ["documents", "metadatas"],
            "limit": limit,
            "offset": offset,
        }
        if where:
            kwargs["where"] = where

        result = self.collection.get(**kwargs)
        memories: list[Memory] = []
        if result and result["ids"]:
            for i, mid in enumerate(result["ids"]):
                memories.append(
                    self._meta_to_memory(mid, result["documents"][i], result["metadatas"][i])
                )
        return memories

    def count(self, filter: QueryFilter | None = None) -> int:
        if filter is None:
            return self.collection.count()
        return len(self.list_all(filter=filter, limit=100000))

    def clear(self) -> int:
        n = self.collection.count()
        if n > 0:
            all_ids = self.collection.get(limit=n)["ids"]
            self.collection.delete(ids=all_ids)
        return n

    @staticmethod
    def _meta_to_memory(memory_id: str, content: str, meta: dict) -> Memory:
        """Reconstruct a Memory from Chroma metadata."""
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
