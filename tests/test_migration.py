"""Tests for migration from Claude Code markdown memories."""

import pytest
from pathlib import Path

from skillmind.migration import parse_memory_file, discover_memory_files, migrate_memories
from skillmind.models import MemoryType
from skillmind.trainer import Trainer
from skillmind.config import SkillMindConfig, StoreConfig


@pytest.fixture
def memory_files(tmp_dir) -> Path:
    """Create sample Claude Code memory markdown files."""
    mem_dir = tmp_dir / "memory"
    mem_dir.mkdir()

    # File with frontmatter
    (mem_dir / "feedback_pdf_quality.md").write_text(
        "---\n"
        "name: feedback_pdf_quality\n"
        "description: PDF quality standards\n"
        "type: feedback\n"
        "---\n\n"
        "Always use real Umlaute. Graphics 600 DPI minimum.\n",
        encoding="utf-8",
    )

    # File with frontmatter
    (mem_dir / "user_role.md").write_text(
        "---\n"
        "name: user_role\n"
        "description: User role and background\n"
        "type: user\n"
        "---\n\n"
        "SEO Freelancer based in Koblenz, Germany.\n",
        encoding="utf-8",
    )

    # File without frontmatter
    (mem_dir / "project_notes.md").write_text(
        "Paroc project is in Phase 2. Deadline end of March.\n",
        encoding="utf-8",
    )

    # MEMORY.md (should be skipped)
    (mem_dir / "MEMORY.md").write_text(
        "# Memory Index\n- [feedback_pdf_quality.md](feedback_pdf_quality.md)\n",
        encoding="utf-8",
    )

    return mem_dir


class TestParseMemoryFile:
    def test_parse_with_frontmatter(self, memory_files):
        result = parse_memory_file(memory_files / "feedback_pdf_quality.md")
        assert result is not None
        assert result["name"] == "feedback_pdf_quality"
        assert result["type"] == "feedback"
        assert "Umlaute" in result["content"]

    def test_parse_without_frontmatter(self, memory_files):
        result = parse_memory_file(memory_files / "project_notes.md")
        assert result is not None
        assert result["name"] == "project_notes"
        assert "Paroc" in result["content"]

    def test_parse_nonexistent(self, tmp_dir):
        result = parse_memory_file(tmp_dir / "nonexistent.md")
        assert result is None


class TestDiscoverMemoryFiles:
    def test_discover_from_dir(self, memory_files):
        files = discover_memory_files(memory_files)
        # Should find 3 files (not MEMORY.md — wait, actually discover returns all .md)
        # MEMORY.md is skipped in migrate_memories, not in discover
        assert len(files) >= 3

    def test_discover_single_file(self, memory_files):
        files = discover_memory_files(memory_files / "user_role.md")
        assert len(files) == 1


class TestMigrateMemories:
    def test_dry_run(self, memory_files, tmp_dir, mock_engine):
        try:
            import chromadb
        except ImportError:
            pytest.skip("chromadb not installed")

        from skillmind.store.chroma_store import ChromaStore

        config = SkillMindConfig(
            data_dir=str(tmp_dir / "sm"),
            store=StoreConfig(backend="chroma", chroma_path=str(tmp_dir / "sm" / "chroma")),
        )
        store = ChromaStore(config=config, engine=mock_engine)
        store.initialize()
        trainer = Trainer(store)

        stats = migrate_memories(trainer, source_dir=memory_files, dry_run=True)
        assert stats["files_found"] >= 3
        assert stats["imported"] >= 3
        assert store.count() == 0  # Dry run — nothing stored

    def test_actual_import(self, memory_files, tmp_dir, mock_engine):
        try:
            import chromadb
        except ImportError:
            pytest.skip("chromadb not installed")

        from skillmind.store.chroma_store import ChromaStore

        config = SkillMindConfig(
            data_dir=str(tmp_dir / "sm"),
            store=StoreConfig(backend="chroma", chroma_path=str(tmp_dir / "sm" / "chroma")),
        )
        store = ChromaStore(config=config, engine=mock_engine)
        store.initialize()
        trainer = Trainer(store)

        stats = migrate_memories(trainer, source_dir=memory_files, dry_run=False)
        assert stats["imported"] >= 3
        assert store.count() >= 3
