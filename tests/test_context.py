"""Tests for SkillMind Context Generator."""

import pytest

from skillmind.context import ContextGenerator
from skillmind.models import ContextRule, MemoryType, QueryFilter
from skillmind.config import SkillMindConfig, StoreConfig


@pytest.fixture
def context_gen(tmp_dir, mock_engine, sample_memories):
    """Context generator with populated store."""
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
    store.add_batch(sample_memories)

    return ContextGenerator(store, max_tokens=4000)


class TestContextGeneration:
    def test_generate_basic(self, context_gen):
        ctx = context_gen.generate(query="PDF generation")
        assert "# SkillMind Context" in ctx
        assert len(ctx) > 50

    def test_generate_with_file(self, context_gen):
        ctx = context_gen.generate(current_file="report.pdf")
        assert "# SkillMind Context" in ctx

    def test_generate_with_topic(self, context_gen):
        ctx = context_gen.generate(current_topic="SEO audit for Paroc")
        assert "# SkillMind Context" in ctx

    def test_includes_user_memories(self, context_gen):
        ctx = context_gen.generate(query="anything")
        # User memories should always be included
        assert "User" in ctx or "user" in ctx.lower()

    def test_with_context_rule(self, context_gen):
        rule = ContextRule(
            trigger="file:*.pdf",
            load_topics=["pdf_generation"],
            priority=10,
        )
        context_gen.add_rule(rule)

        ctx = context_gen.generate(current_file="report.pdf")
        assert len(ctx) > 50

    def test_generate_to_file(self, context_gen, tmp_dir):
        path = context_gen.generate_to_file(
            tmp_dir / "context.md",
            query="SEO workflow",
        )
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert "# SkillMind Context" in content

    def test_empty_store(self, tmp_dir, mock_engine):
        try:
            import chromadb
        except ImportError:
            pytest.skip("chromadb not installed")

        from skillmind.store.chroma_store import ChromaStore

        config = SkillMindConfig(
            data_dir=str(tmp_dir),
            store=StoreConfig(backend="chroma", chroma_path=str(tmp_dir / "chroma_empty")),
        )
        store = ChromaStore(config=config, engine=mock_engine)
        store.initialize()
        gen = ContextGenerator(store)

        ctx = gen.generate(query="anything")
        assert "No relevant memories" in ctx
