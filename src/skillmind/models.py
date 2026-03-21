"""Core data models for SkillMind memory system."""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    """Types of memory entries."""

    USER = "user"
    FEEDBACK = "feedback"
    PROJECT = "project"
    REFERENCE = "reference"
    SKILL = "skill"


class MemorySource(str, Enum):
    """Where the memory was captured from."""

    CONVERSATION = "conversation"
    GIT_COMMIT = "git_commit"
    FILE_CHANGE = "file_change"
    MANUAL = "manual"
    IMPORT = "import"
    SKILL_SEEKERS = "skill_seekers"


class Memory(BaseModel):
    """A single memory entry."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: MemoryType
    topic: str = Field(..., description="Primary topic tag, e.g. 'pdf_generation', 'paroc'")
    title: str = Field(..., description="Short descriptive title")
    content: str = Field(..., description="The actual knowledge")
    tags: list[str] = Field(default_factory=list, description="Additional searchable tags")
    source: MemorySource = Field(default=MemorySource.MANUAL)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime | None = Field(default=None)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_document(self) -> str:
        """Flatten to a single string for embedding."""
        parts = [self.title, self.content]
        if self.tags:
            parts.append(f"Tags: {', '.join(self.tags)}")
        return "\n".join(parts)

    def to_metadata_dict(self) -> dict[str, Any]:
        """Metadata dict for vector DB storage (no nested objects)."""
        return {
            "type": self.type.value,
            "topic": self.topic,
            "title": self.title,
            "source": self.source.value,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else "",
            "tags": ",".join(self.tags),
        }


class QueryFilter(BaseModel):
    """Filter criteria for memory queries."""

    types: list[MemoryType] | None = None
    topics: list[str] | None = None
    tags: list[str] | None = None
    source: MemorySource | None = None
    min_confidence: float = 0.0
    include_expired: bool = False


class QueryResult(BaseModel):
    """A single query result with relevance score."""

    memory: Memory
    score: float = Field(default=0.0, description="Relevance score (0-1, higher = more relevant)")


class SkillEntry(BaseModel):
    """A learned skill from external source (Skill Seekers pipeline)."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    version: str = "1.0"
    source_url: str = ""
    content: str = ""
    sections: list[dict[str, Any]] = Field(default_factory=list)
    last_synced: datetime = Field(default_factory=datetime.utcnow)
    auto_update: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContextRule(BaseModel):
    """Rule for auto-loading memories/skills based on triggers."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    trigger: str = Field(..., description="Pattern: 'file:*.pdf', 'project:paroc', 'topic:seo'")
    load_topics: list[str] = Field(default_factory=list)
    load_types: list[MemoryType] = Field(default_factory=list)
    load_skill_ids: list[str] = Field(default_factory=list)
    priority: int = Field(default=0, description="Higher = loaded first")
