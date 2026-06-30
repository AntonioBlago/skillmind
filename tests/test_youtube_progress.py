"""Tests for YouTubeLearner progress reporting (progress_cb wiring).

These run fully offline: network (metadata/transcript), the Claude knowledge
extraction, and memory storage are all monkeypatched, so the test only asserts
that the progress callback fires once per phase with monotonically increasing
step numbers.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from skillmind.video.youtube_learner import YouTubeLearner


@pytest.fixture
def offline_learner():
    """A YouTubeLearner with all network/Claude/storage calls stubbed out."""
    trainer = MagicMock()
    # trainer.learn returns a truthy object so memories list fills up
    trainer.learn.return_value = MagicMock(id="m1")

    yt = YouTubeLearner(trainer=trainer)
    yt._get_metadata = MagicMock(return_value={
        "title": "Test Video", "author": "Tester", "duration": 600,
        "tags": [], "video_id": "abcdef12345",
        "url": "https://www.youtube.com/watch?v=abcdef12345",
    })
    yt._get_transcript = MagicMock(return_value="Some transcript text. " * 50)
    yt._extract_knowledge = MagicMock(return_value={
        "title": "Test Video", "summary": "A summary.",
        "key_takeaways": ["A", "B"], "topic": "testing", "tags": ["t1"],
    })
    return yt


def test_progress_cb_fires_all_four_phases(offline_learner):
    calls: list[tuple[int, int, str]] = []
    offline_learner.learn(
        "https://www.youtube.com/watch?v=abcdef12345",
        progress_cb=lambda step, total, msg: calls.append((step, total, msg)),
    )

    steps = [c[0] for c in calls]
    assert steps == [1, 2, 3, 4], f"expected phases 1..4, got {steps}"
    assert all(c[1] == 4 for c in calls), "total should always be 4"
    assert all(isinstance(c[2], str) and c[2] for c in calls), "each phase needs a message"


def test_learn_works_without_progress_cb(offline_learner):
    """Backward compatibility: omitting progress_cb must not break learn()."""
    memories = offline_learner.learn("https://www.youtube.com/watch?v=abcdef12345")
    assert isinstance(memories, list)
    assert len(memories) >= 1


def test_progress_cb_exception_is_swallowed(offline_learner):
    """A throwing callback must not abort the learning run."""
    def bad_cb(step, total, msg):
        raise RuntimeError("boom")

    memories = offline_learner.learn(
        "https://www.youtube.com/watch?v=abcdef12345", progress_cb=bad_cb,
    )
    assert len(memories) >= 1


class _Cp1252Stdout:
    """stdout double that rejects non-cp1252 glyphs until reconfigured to UTF-8."""

    def __init__(self) -> None:
        self.encoding = "cp1252"
        self.errors = "strict"

    def reconfigure(self, *, encoding=None, errors=None):
        if encoding:
            self.encoding = encoding
        if errors:
            self.errors = errors

    def write(self, text: str) -> int:
        text.encode(self.encoding, errors=self.errors)
        return len(text)

    def flush(self) -> None:
        pass


def test_emoji_title_does_not_crash_on_cp1252_stdout(offline_learner, monkeypatch):
    """Regression: a 🧠-prefixed title once aborted learn() on a Windows console.

    learn() now calls force_utf8_io() up front, so the progress line
    ``[SkillMind] 2/4 Hole Transkript: 🧠…`` prints instead of raising
    UnicodeEncodeError at the emoji's column.
    """
    offline_learner._get_metadata = MagicMock(return_value={
        "title": "🧠 Wie du Ablehnung im Verkauf meisterst",
        "author": "Tester", "duration": 600, "tags": [],
        "video_id": "abcdef12345",
        "url": "https://www.youtube.com/watch?v=abcdef12345",
    })
    monkeypatch.setattr("sys.stdout", _Cp1252Stdout())

    memories = offline_learner.learn("https://www.youtube.com/watch?v=abcdef12345")
    assert len(memories) >= 1


def test_no_transcript_skips_to_reference(offline_learner):
    """When no transcript is found, only phases 1+2 fire and a reference is stored."""
    offline_learner._get_transcript = MagicMock(return_value="")
    calls: list[int] = []
    memories = offline_learner.learn(
        "https://www.youtube.com/watch?v=abcdef12345",
        progress_cb=lambda step, total, msg: calls.append(step),
    )
    assert calls == [1, 2], f"only metadata+transcript phases expected, got {calls}"
    assert len(memories) == 1
