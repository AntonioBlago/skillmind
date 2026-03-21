"""
SkillMind Trainer — auto-classify, deduplicate, consolidate memories.

The Trainer is the intelligence layer that converts raw observations
into structured, deduplicated, high-quality memories.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any

from .models import Memory, MemoryType, MemorySource, QueryFilter, QueryResult
from .store.base import MemoryStore


# Classification keywords per memory type
TYPE_SIGNALS: dict[MemoryType, list[str]] = {
    MemoryType.USER: [
        "i am", "my role", "i work", "my background", "i prefer",
        "ich bin", "mein beruf", "meine rolle",
    ],
    MemoryType.FEEDBACK: [
        "don't", "always", "never", "stop", "keep doing", "prefer",
        "nicht", "immer", "nie", "aufhören", "bitte",
        "correct", "wrong", "right way", "better to",
    ],
    MemoryType.PROJECT: [
        "deadline", "sprint", "release", "client", "meeting",
        "projekt", "kunde", "termin", "phase", "milestone",
        "budget", "timeline", "status",
    ],
    MemoryType.REFERENCE: [
        "url", "link", "dashboard", "wiki", "confluence",
        "linear", "jira", "slack", "channel", "board",
        "api key", "endpoint", "documentation at",
    ],
    MemoryType.SKILL: [
        "how to", "pattern", "architecture", "workflow",
        "best practice", "template", "boilerplate",
        "anleitung", "vorlage", "workflow",
    ],
}

# Topic extraction patterns
TOPIC_PATTERNS: list[tuple[str, str]] = [
    (r"\b(pdf|fpdf|reportlab)\b", "pdf_generation"),
    (r"\b(seo|keyword|ranking|serp|gsc)\b", "seo"),
    (r"\b(notion)\b", "notion"),
    (r"\b(linkedin|social\s*media)\b", "social_media"),
    (r"\b(csv|excel|xlsx)\b", "data_export"),
    (r"\b(git|commit|branch|merge)\b", "git"),
    (r"\b(docker|container|kubernetes)\b", "devops"),
    (r"\b(react|vue|angular|frontend)\b", "frontend"),
    (r"\b(python|pip|pyproject)\b", "python"),
    (r"\b(test|pytest|unittest)\b", "testing"),
    (r"\b(api|endpoint|rest|graphql)\b", "api"),
    (r"\b(database|sql|postgres|supabase)\b", "database"),
    (r"\b(kunde|client|angebot|offer)\b", "client_work"),
]


class Trainer:
    """
    Automatically classifies, deduplicates, and consolidates memories.

    Usage:
        trainer = Trainer(store)
        memory = trainer.learn("User prefers dark mode in PDFs", source=MemorySource.CONVERSATION)
        trainer.consolidate()  # periodic cleanup
    """

    def __init__(
        self,
        store: MemoryStore,
        duplicate_threshold: float = 0.90,
        sanitize: bool = True,
        sanitizer: Any = None,
    ):
        self.store = store
        self.duplicate_threshold = duplicate_threshold
        self.sanitize = sanitize

        if sanitize:
            if sanitizer:
                self._sanitizer = sanitizer
            else:
                from .sanitizer import create_default_sanitizer
                self._sanitizer = create_default_sanitizer()
        else:
            self._sanitizer = None

    def learn(
        self,
        content: str,
        title: str | None = None,
        source: MemorySource = MemorySource.CONVERSATION,
        force_type: MemoryType | None = None,
        force_topic: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Memory | None:
        """
        Take raw input, classify it, check for duplicates, and store.

        Returns the Memory if stored, None if it was a duplicate.
        """
        # 0. Sanitize sensitive data before anything else
        if self._sanitizer:
            result = self._sanitizer.sanitize(content)
            content = result.sanitized
            if title:
                title = self._sanitizer.sanitize_memory_content(title)
            if result.was_modified:
                metadata = metadata or {}
                metadata["sanitized"] = True
                metadata["redaction_count"] = result.redaction_count

        # 1. Classify type
        mem_type = force_type or self._classify_type(content)

        # 2. Extract topic
        topic = force_topic or self._extract_topic(content)

        # 3. Generate title if not provided
        if not title:
            title = self._generate_title(content, mem_type)

        # 4. Build memory
        memory = Memory(
            type=mem_type,
            topic=topic,
            title=title,
            content=content,
            tags=tags or self._extract_tags(content),
            source=source,
            confidence=self._estimate_confidence(content, source),
            metadata=metadata or {},
            expires_at=self._estimate_expiry(mem_type),
        )

        # 5. Check for duplicates
        duplicates = self.store.find_duplicates(memory, threshold=self.duplicate_threshold)
        if duplicates:
            # Merge into the most relevant existing memory
            existing = duplicates[0].memory
            merged = self._merge_memories(existing, memory)
            self.store.update(merged)
            return merged

        # 6. Store new memory
        self.store.add(memory)
        return memory

    def consolidate(self) -> dict[str, int]:
        """
        Periodic maintenance: merge similar memories, expire stale ones, rebalance.

        Returns stats dict.
        """
        stats = {"merged": 0, "expired": 0, "updated": 0}

        # 1. Remove expired memories
        stats["expired"] = self.store.cleanup_expired()

        # 2. Decay confidence of old project memories
        project_memories = self.store.list_all(
            filter=QueryFilter(types=[MemoryType.PROJECT]),
            limit=1000,
        )
        now = datetime.utcnow()
        for mem in project_memories:
            age_days = (now - mem.created_at).days
            if age_days > 30 and mem.confidence > 0.3:
                mem.confidence = max(0.3, mem.confidence - 0.05 * (age_days // 30))
                self.store.update(mem)
                stats["updated"] += 1

        # 3. Find and merge near-duplicates within same type
        for mem_type in MemoryType:
            memories = self.store.list_all(
                filter=QueryFilter(types=[mem_type]),
                limit=1000,
            )
            merged_ids: set[str] = set()

            for mem in memories:
                if mem.id in merged_ids:
                    continue

                dupes = self.store.find_duplicates(mem, threshold=self.duplicate_threshold)
                for dupe in dupes:
                    if dupe.memory.id in merged_ids:
                        continue
                    # Merge dupe into mem
                    merged = self._merge_memories(mem, dupe.memory)
                    self.store.update(merged)
                    self.store.delete(dupe.memory.id)
                    merged_ids.add(dupe.memory.id)
                    stats["merged"] += 1

        return stats

    def _classify_type(self, content: str) -> MemoryType:
        """Classify memory type from content using keyword scoring."""
        content_lower = content.lower()
        scores: dict[MemoryType, int] = {t: 0 for t in MemoryType}

        for mem_type, keywords in TYPE_SIGNALS.items():
            for kw in keywords:
                if kw in content_lower:
                    scores[mem_type] += 1

        best = max(scores, key=scores.get)  # type: ignore
        if scores[best] == 0:
            return MemoryType.FEEDBACK  # Default: treat as feedback/preference
        return best

    def _extract_topic(self, content: str) -> str:
        """Extract primary topic from content."""
        content_lower = content.lower()
        for pattern, topic in TOPIC_PATTERNS:
            if re.search(pattern, content_lower):
                return topic
        return "general"

    def _extract_tags(self, content: str) -> list[str]:
        """Extract relevant tags from content."""
        content_lower = content.lower()
        tags: list[str] = []
        for pattern, topic in TOPIC_PATTERNS:
            if re.search(pattern, content_lower):
                tags.append(topic)
        return tags[:5]  # Max 5 tags

    def _generate_title(self, content: str, mem_type: MemoryType) -> str:
        """Generate a short title from content."""
        # Take first sentence or first 80 chars
        first_line = content.split("\n")[0].strip()
        first_sentence = first_line.split(".")[0].strip()
        if len(first_sentence) > 80:
            first_sentence = first_sentence[:77] + "..."
        return f"[{mem_type.value}] {first_sentence}"

    def _estimate_confidence(self, content: str, source: MemorySource) -> float:
        """Estimate initial confidence based on source and content."""
        base = {
            MemorySource.MANUAL: 1.0,
            MemorySource.CONVERSATION: 0.85,
            MemorySource.GIT_COMMIT: 0.9,
            MemorySource.FILE_CHANGE: 0.7,
            MemorySource.IMPORT: 0.95,
            MemorySource.SKILL_SEEKERS: 0.8,
        }
        return base.get(source, 0.8)

    def _estimate_expiry(self, mem_type: MemoryType) -> datetime | None:
        """Set expiry for ephemeral memory types."""
        if mem_type == MemoryType.PROJECT:
            return datetime.utcnow() + timedelta(days=90)
        return None  # User, feedback, reference, skill don't expire

    def _merge_memories(self, existing: Memory, new: Memory) -> Memory:
        """Merge new memory into existing one."""
        # Combine content if meaningfully different
        if new.content not in existing.content:
            existing.content = f"{existing.content}\n\n---\n\n{new.content}"

        # Merge tags
        existing.tags = list(set(existing.tags + new.tags))[:10]

        # Boost confidence on repeated observations
        existing.confidence = min(1.0, existing.confidence + 0.05)

        # Update timestamp
        existing.updated_at = datetime.utcnow()

        # Extend expiry if project memory is re-confirmed
        if existing.type == MemoryType.PROJECT and existing.expires_at:
            existing.expires_at = datetime.utcnow() + timedelta(days=90)

        return existing
