"""
SkillMind Review Queue — pending memories for human review before storing.

Auto-detected memories from conversations land here first.
User reviews and approves/rejects before they go to Pinecone.

Queue stored locally as JSON (not in vector DB until approved).
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from .models import Memory, MemoryType, MemorySource
from .store.base import MemoryStore
from .trainer import Trainer


class ReviewQueue:
    """
    Holds pending memories for human review.

    Flow:
        Auto-listener detects something → add_pending()
        User reviews → list_pending()
        User approves → approve() → stored in Pinecone
        User rejects → reject() → deleted
        User edits → edit_pending() → then approve()
    """

    def __init__(self, queue_path: str | Path = ".skillmind/review_queue.json"):
        self.queue_path = Path(queue_path)
        self.queue_path.parent.mkdir(parents=True, exist_ok=True)
        self._queue: list[dict[str, Any]] = self._load()

    def _load(self) -> list[dict[str, Any]]:
        if self.queue_path.exists():
            try:
                return json.loads(self.queue_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return []
        return []

    def _save(self):
        self.queue_path.write_text(
            json.dumps(self._queue, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

    def add_pending(
        self,
        content: str,
        memory_type: str,
        topic: str,
        title: str = "",
        source: str = "conversation",
        tags: list[str] | None = None,
        trigger: str = "",
    ) -> dict:
        """Add a memory to the review queue (not stored in vector DB yet)."""
        entry = {
            "id": str(uuid.uuid4())[:8],
            "content": content,
            "type": memory_type,
            "topic": topic,
            "title": title or content[:60] + ("..." if len(content) > 60 else ""),
            "source": source,
            "tags": tags or [],
            "trigger": trigger,
            "detected_at": datetime.utcnow().isoformat(),
            "status": "pending",
        }
        self._queue.append(entry)
        self._save()
        return entry

    def list_pending(self) -> list[dict]:
        """Get all pending (unreviewed) memories."""
        return [e for e in self._queue if e.get("status") == "pending"]

    def count_pending(self) -> int:
        return len(self.list_pending())

    def get(self, entry_id: str) -> dict | None:
        """Get a specific entry by ID (supports partial ID)."""
        for e in self._queue:
            if e["id"] == entry_id or e["id"].startswith(entry_id):
                return e
        return None

    def approve(self, entry_id: str, trainer: Trainer) -> Memory | None:
        """Approve a pending memory → store in vector DB via Trainer."""
        entry = self.get(entry_id)
        if not entry or entry["status"] != "pending":
            return None

        memory = trainer.learn(
            content=entry["content"],
            title=entry.get("title"),
            source=MemorySource(entry.get("source", "conversation")),
            force_type=MemoryType(entry["type"]),
            force_topic=entry.get("topic") or None,
            tags=entry.get("tags"),
        )

        entry["status"] = "approved"
        entry["approved_at"] = datetime.utcnow().isoformat()
        if memory:
            entry["memory_id"] = memory.id
        self._save()

        return memory

    def approve_all(self, trainer: Trainer) -> list[Memory]:
        """Approve all pending memories."""
        memories = []
        for entry in self.list_pending():
            mem = self.approve(entry["id"], trainer)
            if mem:
                memories.append(mem)
        return memories

    def reject(self, entry_id: str, reason: str = "") -> bool:
        """Reject a pending memory → remove from queue."""
        entry = self.get(entry_id)
        if not entry:
            return False

        entry["status"] = "rejected"
        entry["rejected_at"] = datetime.utcnow().isoformat()
        entry["reject_reason"] = reason
        self._save()
        return True

    def reject_all(self) -> int:
        """Reject all pending memories."""
        count = 0
        for entry in self.list_pending():
            entry["status"] = "rejected"
            entry["rejected_at"] = datetime.utcnow().isoformat()
            count += 1
        self._save()
        return count

    def edit_pending(self, entry_id: str, **kwargs) -> dict | None:
        """Edit a pending entry before approving."""
        entry = self.get(entry_id)
        if not entry or entry["status"] != "pending":
            return None

        for key in ["content", "type", "topic", "title", "tags"]:
            if key in kwargs and kwargs[key] is not None:
                entry[key] = kwargs[key]

        entry["edited_at"] = datetime.utcnow().isoformat()
        self._save()
        return entry

    def cleanup(self, keep_days: int = 7) -> int:
        """Remove old approved/rejected entries."""
        cutoff = datetime.utcnow().timestamp() - (keep_days * 86400)
        before = len(self._queue)

        self._queue = [
            e for e in self._queue
            if e.get("status") == "pending"
            or datetime.fromisoformat(e.get("detected_at", "2000-01-01")).timestamp() > cutoff
        ]

        self._save()
        return before - len(self._queue)

    def stats(self) -> dict:
        """Queue statistics."""
        statuses = {"pending": 0, "approved": 0, "rejected": 0}
        for e in self._queue:
            s = e.get("status", "pending")
            statuses[s] = statuses.get(s, 0) + 1
        return {
            "total": len(self._queue),
            **statuses,
        }
