"""Tests for SkillMind Listeners."""

import pytest

from skillmind.listener import ConversationListener, FileListener
from skillmind.models import MemoryType
from skillmind.trainer import Trainer
from skillmind.config import SkillMindConfig, StoreConfig


@pytest.fixture
def trainer(tmp_dir, mock_engine):
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


class TestConversationListener:
    def test_detect_correction(self, trainer):
        listener = ConversationListener(trainer)
        messages = [
            {"role": "user", "content": "Don't use Wir-Form in my client emails, always use Ich-Form since I'm a solo freelancer"},
        ]
        memories = listener.extract_from_messages(messages)
        assert len(memories) >= 1
        assert memories[0].type == MemoryType.FEEDBACK

    def test_detect_reference(self, trainer):
        listener = ConversationListener(trainer)
        messages = [
            {"role": "user", "content": "All bugs are tracked in Linear at https://linear.app/team/project-board"},
        ]
        memories = listener.extract_from_messages(messages)
        assert len(memories) >= 1
        assert memories[0].type == MemoryType.REFERENCE

    def test_detect_project_context(self, trainer):
        listener = ConversationListener(trainer)
        messages = [
            {"role": "user", "content": "The release deadline is next Thursday, we need to freeze all merges before that"},
        ]
        memories = listener.extract_from_messages(messages)
        assert len(memories) >= 1
        assert memories[0].type == MemoryType.PROJECT

    def test_skip_short_messages(self, trainer):
        listener = ConversationListener(trainer)
        messages = [
            {"role": "user", "content": "ok"},
            {"role": "user", "content": "yes"},
            {"role": "assistant", "content": "Some long response that should be skipped"},
        ]
        memories = listener.extract_from_messages(messages)
        assert len(memories) == 0

    def test_skip_assistant_messages(self, trainer):
        listener = ConversationListener(trainer)
        messages = [
            {"role": "assistant", "content": "Don't forget to always use Ich-Form in client communication"},
        ]
        memories = listener.extract_from_messages(messages)
        assert len(memories) == 0


class TestFileListener:
    def test_claude_md_change(self, trainer):
        listener = FileListener(trainer)
        mem = listener.on_file_change("project/CLAUDE.md", event="modified")
        assert mem is not None
        assert mem.type == MemoryType.PROJECT
        assert "CLAUDE.md" in mem.content

    def test_skip_pyc_files(self, trainer):
        listener = FileListener(trainer)
        mem = listener.on_file_change("src/__pycache__/module.pyc", event="modified")
        assert mem is None

    def test_skip_hidden_dirs(self, trainer):
        listener = FileListener(trainer)
        mem = listener.on_file_change(".git/objects/abc123", event="created")
        assert mem is None
