"""Abstract base class for all SkillMind memory stores."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING

from ..models import Memory, MemoryType, QueryFilter, QueryResult

if TYPE_CHECKING:
    from ..config import SkillMindConfig
    from ..embeddings import EmbeddingEngine


class MemoryStore(ABC):
    """
    Abstract interface for memory storage backends.

    All backends must implement: add, query, get, update, delete, list_all, count, clear.
    """

    def __init__(self, config: SkillMindConfig, engine: EmbeddingEngine):
        self.config = config
        self.engine = engine

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the store (create collections, tables, indices)."""
        ...

    @abstractmethod
    def add(self, memory: Memory) -> str:
        """
        Add a memory to the store.

        Returns the memory ID.
        """
        ...

    @abstractmethod
    def add_batch(self, memories: list[Memory]) -> list[str]:
        """Add multiple memories at once. Returns list of IDs."""
        ...

    @abstractmethod
    def query(
        self,
        text: str,
        limit: int = 5,
        filter: QueryFilter | None = None,
    ) -> list[QueryResult]:
        """
        Semantic search across memories.

        Args:
            text: Natural language query
            limit: Max results
            filter: Optional metadata filters

        Returns:
            Ranked list of QueryResult (memory + relevance score)
        """
        ...

    @abstractmethod
    def get(self, memory_id: str) -> Memory | None:
        """Get a specific memory by ID."""
        ...

    @abstractmethod
    def update(self, memory: Memory) -> None:
        """Update an existing memory (re-embeds content)."""
        ...

    @abstractmethod
    def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID. Returns True if found and deleted."""
        ...

    @abstractmethod
    def list_all(
        self,
        filter: QueryFilter | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Memory]:
        """List memories with optional filtering (no semantic search)."""
        ...

    @abstractmethod
    def count(self, filter: QueryFilter | None = None) -> int:
        """Count memories, optionally filtered."""
        ...

    @abstractmethod
    def clear(self) -> int:
        """Delete ALL memories. Returns count of deleted items."""
        ...

    def cleanup_expired(self) -> int:
        """Remove expired memories. Returns count removed."""
        now = datetime.utcnow()
        expired = self.list_all(
            filter=QueryFilter(include_expired=True),
            limit=10000,
        )
        count = 0
        for mem in expired:
            if mem.expires_at and mem.expires_at < now:
                self.delete(mem.id)
                count += 1
        return count

    def find_duplicates(self, memory: Memory, threshold: float = 0.92) -> list[QueryResult]:
        """Find potential duplicates of a memory."""
        results = self.query(
            text=memory.to_document(),
            limit=5,
            filter=QueryFilter(types=[memory.type]),
        )
        return [r for r in results if r.score >= threshold and r.memory.id != memory.id]

    def _build_where_filter(self, filter: QueryFilter | None) -> dict | None:
        """Build a metadata filter dict (used by Chroma, Qdrant, etc.)."""
        if not filter:
            return None

        conditions: list[dict] = []

        if filter.types:
            conditions.append(
                {"type": {"$in": [t.value for t in filter.types]}}
            )
        if filter.topics:
            conditions.append(
                {"topic": {"$in": filter.topics}}
            )
        if filter.source:
            conditions.append({"source": {"$eq": filter.source.value}})
        if filter.min_confidence > 0:
            conditions.append({"confidence": {"$gte": filter.min_confidence}})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}
