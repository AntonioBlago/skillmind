"""Vector store backends for SkillMind."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import MemoryStore

if TYPE_CHECKING:
    from ..config import SkillMindConfig
    from ..embeddings import EmbeddingEngine


def create_store(config: SkillMindConfig, engine: EmbeddingEngine) -> MemoryStore:
    """Factory: create the appropriate store backend from config."""
    backend = config.store.backend.lower()

    if backend == "chroma":
        from .chroma_store import ChromaStore

        return ChromaStore(config=config, engine=engine)
    elif backend == "pinecone":
        from .pinecone_store import PineconeStore

        return PineconeStore(config=config, engine=engine)
    elif backend == "supabase":
        from .supabase_store import SupabaseStore

        return SupabaseStore(config=config, engine=engine)
    elif backend == "qdrant":
        from .qdrant_store import QdrantStore

        return QdrantStore(config=config, engine=engine)
    elif backend == "faiss":
        from .faiss_store import FAISSStore

        return FAISSStore(config=config, engine=engine)
    else:
        raise ValueError(
            f"Unknown backend: {backend}. "
            f"Supported: chroma, pinecone, supabase, qdrant, faiss"
        )


__all__ = ["MemoryStore", "create_store"]
