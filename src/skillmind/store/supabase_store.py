"""Supabase (pgvector) backend for SkillMind memory store."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from ..config import SkillMindConfig
from ..embeddings import EmbeddingEngine
from ..models import Memory, MemoryType, MemorySource, QueryFilter, QueryResult
from .base import MemoryStore


# SQL to create the memories table (run once via Supabase dashboard or migration)
SETUP_SQL = """
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create memories table
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    topic TEXT NOT NULL,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    tags TEXT[] DEFAULT '{}',
    source TEXT DEFAULT 'manual',
    confidence REAL DEFAULT 1.0,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now(),
    expires_at TIMESTAMPTZ,
    metadata JSONB DEFAULT '{}',
    embedding VECTOR({dimension})
);

-- Index for vector similarity search
CREATE INDEX IF NOT EXISTS memories_embedding_idx
    ON memories USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- Index for metadata filtering
CREATE INDEX IF NOT EXISTS memories_type_idx ON memories (type);
CREATE INDEX IF NOT EXISTS memories_topic_idx ON memories (topic);

-- RPC function for similarity search
CREATE OR REPLACE FUNCTION match_memories(
    query_embedding VECTOR({dimension}),
    match_count INT DEFAULT 5,
    filter_types TEXT[] DEFAULT NULL,
    filter_topics TEXT[] DEFAULT NULL,
    min_confidence REAL DEFAULT 0.0
)
RETURNS TABLE (
    id TEXT,
    type TEXT,
    topic TEXT,
    title TEXT,
    content TEXT,
    tags TEXT[],
    source TEXT,
    confidence REAL,
    created_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ,
    expires_at TIMESTAMPTZ,
    metadata JSONB,
    similarity REAL
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        m.id, m.type, m.topic, m.title, m.content, m.tags, m.source,
        m.confidence, m.created_at, m.updated_at, m.expires_at, m.metadata,
        1 - (m.embedding <=> query_embedding) AS similarity
    FROM memories m
    WHERE
        (filter_types IS NULL OR m.type = ANY(filter_types))
        AND (filter_topics IS NULL OR m.topic = ANY(filter_topics))
        AND m.confidence >= min_confidence
        AND (m.expires_at IS NULL OR m.expires_at > now())
    ORDER BY m.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;
"""


class SupabaseStore(MemoryStore):
    """
    Supabase/pgvector-backed memory store.

    Cloud-synced, SQL + vectors in one, free tier available.
    Best for multi-device sync and team sharing.
    """

    def __init__(self, config: SkillMindConfig, engine: EmbeddingEngine):
        super().__init__(config, engine)
        self._client: Any = None

    def initialize(self) -> None:
        from supabase import create_client

        self._client = create_client(
            self.config.store.supabase_url,
            self.config.store.supabase_key,
        )

    @property
    def client(self) -> Any:
        if self._client is None:
            self.initialize()
        return self._client

    @property
    def table(self) -> str:
        return self.config.store.supabase_table

    def get_setup_sql(self) -> str:
        """Return SQL to set up the Supabase table. Run this once."""
        return SETUP_SQL.format(dimension=self.engine.dimension)

    def add(self, memory: Memory) -> str:
        embedding = self.engine.embed(memory.to_document())
        row = self._memory_to_row(memory, embedding)
        self.client.table(self.table).insert(row).execute()
        return memory.id

    def add_batch(self, memories: list[Memory]) -> list[str]:
        if not memories:
            return []
        docs = [m.to_document() for m in memories]
        embeddings = self.engine.embed_batch(docs)
        rows = [self._memory_to_row(m, e) for m, e in zip(memories, embeddings)]

        # Supabase batch insert
        self.client.table(self.table).insert(rows).execute()
        return [m.id for m in memories]

    def query(
        self,
        text: str,
        limit: int = 5,
        filter: QueryFilter | None = None,
    ) -> list[QueryResult]:
        embedding = self.engine.embed(text)

        # Use the RPC function for vector search
        params: dict[str, Any] = {
            "query_embedding": embedding,
            "match_count": limit,
        }
        if filter:
            if filter.types:
                params["filter_types"] = [t.value for t in filter.types]
            if filter.topics:
                params["filter_topics"] = filter.topics
            if filter.min_confidence > 0:
                params["min_confidence"] = filter.min_confidence

        result = self.client.rpc("match_memories", params).execute()

        query_results: list[QueryResult] = []
        for row in result.data or []:
            memory = self._row_to_memory(row)
            score = float(row.get("similarity", 0.0))
            query_results.append(QueryResult(memory=memory, score=score))

        return query_results

    def get(self, memory_id: str) -> Memory | None:
        result = (
            self.client.table(self.table)
            .select("*")
            .eq("id", memory_id)
            .execute()
        )
        if result.data:
            return self._row_to_memory(result.data[0])
        return None

    def update(self, memory: Memory) -> None:
        memory.updated_at = datetime.utcnow()
        embedding = self.engine.embed(memory.to_document())
        row = self._memory_to_row(memory, embedding)
        self.client.table(self.table).upsert(row).execute()

    def delete(self, memory_id: str) -> bool:
        result = (
            self.client.table(self.table)
            .delete()
            .eq("id", memory_id)
            .execute()
        )
        return bool(result.data)

    def list_all(
        self,
        filter: QueryFilter | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Memory]:
        query = self.client.table(self.table).select("*")

        if filter:
            if filter.types:
                query = query.in_("type", [t.value for t in filter.types])
            if filter.topics:
                query = query.in_("topic", filter.topics)
            if filter.source:
                query = query.eq("source", filter.source.value)
            if filter.min_confidence > 0:
                query = query.gte("confidence", filter.min_confidence)
            if not filter.include_expired:
                query = query.or_(f"expires_at.is.null,expires_at.gt.{datetime.utcnow().isoformat()}")

        result = query.range(offset, offset + limit - 1).execute()
        return [self._row_to_memory(row) for row in result.data or []]

    def count(self, filter: QueryFilter | None = None) -> int:
        query = self.client.table(self.table).select("id", count="exact")

        if filter and filter.types:
            query = query.in_("type", [t.value for t in filter.types])

        result = query.execute()
        return result.count or 0

    def clear(self) -> int:
        n = self.count()
        self.client.table(self.table).delete().neq("id", "").execute()
        return n

    def _memory_to_row(self, memory: Memory, embedding: list[float]) -> dict:
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
            "expires_at": memory.expires_at.isoformat() if memory.expires_at else None,
            "metadata": memory.metadata,
            "embedding": embedding,
        }

    @staticmethod
    def _row_to_memory(row: dict) -> Memory:
        return Memory(
            id=row["id"],
            type=MemoryType(row["type"]),
            topic=row["topic"],
            title=row["title"],
            content=row["content"],
            tags=row.get("tags") or [],
            source=MemorySource(row.get("source", "manual")),
            confidence=float(row.get("confidence", 1.0)),
            created_at=datetime.fromisoformat(row["created_at"]) if row.get("created_at") else datetime.utcnow(),
            updated_at=datetime.fromisoformat(row["updated_at"]) if row.get("updated_at") else datetime.utcnow(),
            expires_at=datetime.fromisoformat(row["expires_at"]) if row.get("expires_at") else None,
            metadata=row.get("metadata") or {},
        )
