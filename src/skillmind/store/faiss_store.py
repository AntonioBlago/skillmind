"""FAISS + JSON backend for SkillMind memory store."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from ..config import SkillMindConfig
from ..embeddings import EmbeddingEngine
from ..models import Memory, MemoryType, MemorySource, QueryFilter, QueryResult
from .base import MemoryStore


class FAISSStore(MemoryStore):
    """
    FAISS-backed memory store with JSON sidecar for metadata.

    Fully local, zero network, fastest for small-to-medium collections.
    Best for: offline use, air-gapped environments, maximum speed.
    """

    def __init__(self, config: SkillMindConfig, engine: EmbeddingEngine):
        super().__init__(config, engine)
        self._index: Any = None
        self._memories: dict[str, Memory] = {}  # id -> Memory
        self._id_to_idx: dict[str, int] = {}  # memory_id -> faiss row index
        self._idx_to_id: dict[int, str] = {}  # faiss row index -> memory_id
        self._next_idx: int = 0
        self._data_dir = Path(config.store.faiss_path)

    def initialize(self) -> None:
        import faiss

        self._data_dir.mkdir(parents=True, exist_ok=True)

        index_path = self._data_dir / "index.faiss"
        meta_path = self._data_dir / "memories.json"

        if index_path.exists() and meta_path.exists():
            self._index = faiss.read_index(str(index_path))
            self._load_metadata(meta_path)
        else:
            self._index = faiss.IndexFlatIP(self.engine.dimension)  # Inner product (cosine on normalized vecs)
            self._memories = {}
            self._id_to_idx = {}
            self._idx_to_id = {}
            self._next_idx = 0

    @property
    def index(self) -> Any:
        if self._index is None:
            self.initialize()
        return self._index

    def _save(self) -> None:
        """Persist index and metadata to disk."""
        import faiss

        self._data_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self._data_dir / "index.faiss"))
        self._save_metadata(self._data_dir / "memories.json")

    def _save_metadata(self, path: Path) -> None:
        data = {
            "memories": {mid: m.model_dump(mode="json") for mid, m in self._memories.items()},
            "id_to_idx": self._id_to_idx,
            "idx_to_id": {str(k): v for k, v in self._idx_to_id.items()},
            "next_idx": self._next_idx,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    def _load_metadata(self, path: Path) -> None:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        self._memories = {}
        for mid, mdata in data.get("memories", {}).items():
            # Fix datetime fields
            for field in ("created_at", "updated_at", "expires_at"):
                if mdata.get(field) and isinstance(mdata[field], str):
                    try:
                        mdata[field] = datetime.fromisoformat(mdata[field])
                    except (ValueError, TypeError):
                        mdata[field] = None
            self._memories[mid] = Memory(**mdata)

        self._id_to_idx = data.get("id_to_idx", {})
        self._idx_to_id = {int(k): v for k, v in data.get("idx_to_id", {}).items()}
        self._next_idx = data.get("next_idx", 0)

    def add(self, memory: Memory) -> str:
        embedding = self.engine.embed(memory.to_document())
        vec = np.array([embedding], dtype=np.float32)

        idx = self._next_idx
        self.index.add(vec)
        self._memories[memory.id] = memory
        self._id_to_idx[memory.id] = idx
        self._idx_to_id[idx] = memory.id
        self._next_idx += 1

        self._save()
        return memory.id

    def add_batch(self, memories: list[Memory]) -> list[str]:
        if not memories:
            return []

        docs = [m.to_document() for m in memories]
        embeddings = self.engine.embed_batch(docs)
        vecs = np.array(embeddings, dtype=np.float32)

        start_idx = self._next_idx
        self.index.add(vecs)

        for i, mem in enumerate(memories):
            idx = start_idx + i
            self._memories[mem.id] = mem
            self._id_to_idx[mem.id] = idx
            self._idx_to_id[idx] = mem.id

        self._next_idx = start_idx + len(memories)
        self._save()
        return [m.id for m in memories]

    def query(
        self,
        text: str,
        limit: int = 5,
        filter: QueryFilter | None = None,
    ) -> list[QueryResult]:
        if self.index.ntotal == 0:
            return []

        embedding = self.engine.embed(text)
        vec = np.array([embedding], dtype=np.float32)

        # Search more than needed if filtering
        search_k = min(limit * 5, self.index.ntotal) if filter else min(limit, self.index.ntotal)
        scores, indices = self.index.search(vec, search_k)

        query_results: list[QueryResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            mid = self._idx_to_id.get(int(idx))
            if mid is None or mid not in self._memories:
                continue

            memory = self._memories[mid]

            # Apply filters
            if filter:
                if filter.types and memory.type not in filter.types:
                    continue
                if filter.topics and memory.topic not in filter.topics:
                    continue
                if filter.source and memory.source != filter.source:
                    continue
                if memory.confidence < filter.min_confidence:
                    continue
                if not filter.include_expired and memory.expires_at and memory.expires_at < datetime.utcnow():
                    continue

            # Normalize score to 0-1 range (cosine similarity from inner product)
            norm_score = max(0.0, min(1.0, float(score)))
            query_results.append(QueryResult(memory=memory, score=norm_score))

            if len(query_results) >= limit:
                break

        return query_results

    def get(self, memory_id: str) -> Memory | None:
        return self._memories.get(memory_id)

    def update(self, memory: Memory) -> None:
        # FAISS doesn't support in-place updates — delete and re-add
        # For simplicity, we just update metadata and rebuild would fix embeddings
        memory.updated_at = datetime.utcnow()

        if memory.id in self._memories:
            # Update the embedding in the index
            old_idx = self._id_to_idx[memory.id]
            embedding = self.engine.embed(memory.to_document())

            # FAISS IndexFlatIP doesn't support remove — we mark as updated in metadata
            # and the old vector becomes orphaned (cleaned up on rebuild)
            self._memories[memory.id] = memory

            # Add new vector
            vec = np.array([embedding], dtype=np.float32)
            new_idx = self._next_idx
            self.index.add(vec)

            # Remap
            del self._idx_to_id[old_idx]
            self._id_to_idx[memory.id] = new_idx
            self._idx_to_id[new_idx] = memory.id
            self._next_idx += 1

            self._save()

    def delete(self, memory_id: str) -> bool:
        if memory_id not in self._memories:
            return False

        idx = self._id_to_idx.pop(memory_id, None)
        if idx is not None:
            self._idx_to_id.pop(idx, None)
        del self._memories[memory_id]

        self._save()
        return True

    def list_all(
        self,
        filter: QueryFilter | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Memory]:
        all_memories = list(self._memories.values())

        if filter:
            filtered: list[Memory] = []
            for m in all_memories:
                if filter.types and m.type not in filter.types:
                    continue
                if filter.topics and m.topic not in filter.topics:
                    continue
                if filter.source and m.source != filter.source:
                    continue
                if m.confidence < filter.min_confidence:
                    continue
                if not filter.include_expired and m.expires_at and m.expires_at < datetime.utcnow():
                    continue
                filtered.append(m)
            all_memories = filtered

        return all_memories[offset : offset + limit]

    def count(self, filter: QueryFilter | None = None) -> int:
        if filter is None:
            return len(self._memories)
        return len(self.list_all(filter=filter, limit=100000))

    def clear(self) -> int:
        import faiss

        n = len(self._memories)
        self._index = faiss.IndexFlatIP(self.engine.dimension)
        self._memories.clear()
        self._id_to_idx.clear()
        self._idx_to_id.clear()
        self._next_idx = 0
        self._save()
        return n

    def rebuild_index(self) -> None:
        """Rebuild FAISS index from scratch (cleans up orphaned vectors)."""
        import faiss

        self._index = faiss.IndexFlatIP(self.engine.dimension)
        old_memories = dict(self._memories)
        self._id_to_idx.clear()
        self._idx_to_id.clear()
        self._next_idx = 0

        if old_memories:
            mems = list(old_memories.values())
            docs = [m.to_document() for m in mems]
            embeddings = self.engine.embed_batch(docs)
            vecs = np.array(embeddings, dtype=np.float32)
            self._index.add(vecs)

            for i, mem in enumerate(mems):
                self._id_to_idx[mem.id] = i
                self._idx_to_id[i] = mem.id
            self._next_idx = len(mems)

        self._save()
