"""Tests for SkillMind data models."""

import pytest
from datetime import datetime, timedelta

from skillmind.models import (
    Memory,
    MemoryType,
    MemorySource,
    QueryFilter,
    QueryResult,
    SkillEntry,
    ContextRule,
)


class TestMemory:
    def test_create_memory(self):
        mem = Memory(
            type=MemoryType.FEEDBACK,
            topic="pdf",
            title="PDF quality",
            content="Use 600 DPI",
        )
        assert mem.id  # UUID auto-generated
        assert mem.type == MemoryType.FEEDBACK
        assert mem.confidence == 1.0
        assert mem.created_at <= datetime.utcnow()

    def test_to_document(self):
        mem = Memory(
            type=MemoryType.USER,
            topic="role",
            title="SEO Expert",
            content="10 years experience",
            tags=["seo", "expert"],
        )
        doc = mem.to_document()
        assert "SEO Expert" in doc
        assert "10 years experience" in doc
        assert "seo" in doc

    def test_to_metadata_dict(self):
        mem = Memory(
            type=MemoryType.PROJECT,
            topic="paroc",
            title="Client project",
            content="Active project",
            tags=["client", "seo"],
            source=MemorySource.CONVERSATION,
        )
        meta = mem.to_metadata_dict()
        assert meta["type"] == "project"
        assert meta["topic"] == "paroc"
        assert meta["source"] == "conversation"
        assert meta["tags"] == "client,seo"
        assert isinstance(meta["created_at"], str)

    def test_memory_with_expiry(self):
        future = datetime.utcnow() + timedelta(days=90)
        mem = Memory(
            type=MemoryType.PROJECT,
            topic="sprint",
            title="Sprint 5",
            content="Sprint 5 ends March 30",
            expires_at=future,
        )
        assert mem.expires_at > datetime.utcnow()


class TestQueryFilter:
    def test_default_filter(self):
        qf = QueryFilter()
        assert qf.types is None
        assert qf.min_confidence == 0.0
        assert qf.include_expired is False

    def test_typed_filter(self):
        qf = QueryFilter(
            types=[MemoryType.FEEDBACK, MemoryType.USER],
            min_confidence=0.5,
        )
        assert len(qf.types) == 2
        assert qf.min_confidence == 0.5


class TestQueryResult:
    def test_query_result(self):
        mem = Memory(
            type=MemoryType.FEEDBACK,
            topic="test",
            title="Test",
            content="Test content",
        )
        qr = QueryResult(memory=mem, score=0.85)
        assert qr.score == 0.85
        assert qr.memory.topic == "test"


class TestSkillEntry:
    def test_create_skill(self):
        skill = SkillEntry(
            name="react",
            version="18.2",
            source_url="https://react.dev",
            content="React documentation skill",
        )
        assert skill.id
        assert skill.name == "react"
        assert skill.auto_update is False


class TestContextRule:
    def test_file_trigger(self):
        rule = ContextRule(
            trigger="file:*.pdf",
            load_topics=["pdf_generation"],
            priority=10,
        )
        assert rule.trigger == "file:*.pdf"
        assert "pdf_generation" in rule.load_topics
