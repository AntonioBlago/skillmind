"""Tests for SkillMind Trainer (auto-classify, dedup, consolidate)."""

import pytest
from datetime import datetime, timedelta

from skillmind.models import Memory, MemoryType, MemorySource, QueryFilter
from skillmind.trainer import Trainer
from skillmind.config import SkillMindConfig, StoreConfig


@pytest.fixture
def trainer_with_store(tmp_dir, mock_engine):
    """Create a Trainer backed by a Chroma store."""
    try:
        import chromadb
    except ImportError:
        pytest.skip("chromadb not installed")

    from skillmind.store.chroma_store import ChromaStore

    config = SkillMindConfig(
        data_dir=str(tmp_dir),
        store=StoreConfig(backend="chroma", chroma_path=str(tmp_dir / "chroma")),
    )
    store = ChromaStore(config=config, engine=mock_engine)
    store.initialize()
    return Trainer(store)


class TestTrainerClassification:
    def test_classify_feedback(self, trainer_with_store):
        mem = trainer_with_store.learn(
            "Don't use Wir-Form in client emails, always use Ich-Form",
            source=MemorySource.CONVERSATION,
        )
        assert mem is not None
        assert mem.type == MemoryType.FEEDBACK

    def test_classify_user(self, trainer_with_store):
        mem = trainer_with_store.learn(
            "I am a senior SEO consultant with 10 years experience",
            source=MemorySource.CONVERSATION,
        )
        assert mem is not None
        assert mem.type == MemoryType.USER

    def test_classify_project(self, trainer_with_store):
        mem = trainer_with_store.learn(
            "The deadline for the Paroc audit is next Thursday. Budget is 5000 EUR.",
            source=MemorySource.CONVERSATION,
        )
        assert mem is not None
        assert mem.type == MemoryType.PROJECT

    def test_classify_reference(self, trainer_with_store):
        mem = trainer_with_store.learn(
            "Bug tracking is in Linear, check https://linear.app/team/project",
            source=MemorySource.CONVERSATION,
        )
        assert mem is not None
        assert mem.type == MemoryType.REFERENCE

    def test_force_type(self, trainer_with_store):
        mem = trainer_with_store.learn(
            "Some generic content",
            force_type=MemoryType.SKILL,
            source=MemorySource.MANUAL,
        )
        assert mem is not None
        assert mem.type == MemoryType.SKILL


class TestTrainerTopicExtraction:
    def test_pdf_topic(self, trainer_with_store):
        mem = trainer_with_store.learn(
            "Always use 600 DPI in PDF reports",
            source=MemorySource.CONVERSATION,
        )
        assert mem is not None
        assert "pdf" in mem.topic.lower()

    def test_seo_topic(self, trainer_with_store):
        mem = trainer_with_store.learn(
            "Keyword mapping should match the portfolio structure",
            source=MemorySource.CONVERSATION,
        )
        assert mem is not None
        assert mem.topic in ("seo", "general")

    def test_force_topic(self, trainer_with_store):
        mem = trainer_with_store.learn(
            "Some content",
            force_topic="custom_topic",
            source=MemorySource.MANUAL,
        )
        assert mem is not None
        assert mem.topic == "custom_topic"


class TestTrainerConsolidate:
    def test_consolidate_empty(self, trainer_with_store):
        stats = trainer_with_store.consolidate()
        assert stats["merged"] == 0
        assert stats["expired"] == 0

    def test_expire_old_project(self, trainer_with_store):
        old_mem = Memory(
            type=MemoryType.PROJECT,
            topic="old_project",
            title="Old project",
            content="This project ended months ago",
            expires_at=datetime.utcnow() - timedelta(days=1),
        )
        trainer_with_store.store.add(old_mem)
        assert trainer_with_store.store.count() == 1

        stats = trainer_with_store.consolidate()
        assert stats["expired"] == 1
        assert trainer_with_store.store.count() == 0


class TestTrainerConfidence:
    def test_manual_has_high_confidence(self, trainer_with_store):
        mem = trainer_with_store.learn(
            "Important fact",
            source=MemorySource.MANUAL,
        )
        assert mem is not None
        assert mem.confidence == 1.0

    def test_conversation_has_lower_confidence(self, trainer_with_store):
        mem = trainer_with_store.learn(
            "Something mentioned in conversation",
            source=MemorySource.CONVERSATION,
        )
        assert mem is not None
        assert mem.confidence < 1.0

    def test_project_gets_expiry(self, trainer_with_store):
        mem = trainer_with_store.learn(
            "Sprint deadline is Friday",
            force_type=MemoryType.PROJECT,
            source=MemorySource.CONVERSATION,
        )
        assert mem is not None
        assert mem.expires_at is not None
        assert mem.expires_at > datetime.utcnow()
