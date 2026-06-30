"""Tests for skillmind.ioutil.force_utf8_io.

Regression cover for the Windows cp1252 crash where a single print() carrying
an emoji (e.g. a 🧠 at the start of a YouTube title) aborted learn_youtube with
``'charmap' codec can't encode character '\\U0001f9e0'``.
"""

from __future__ import annotations

import pytest

from skillmind.ioutil import force_utf8_io


class _Cp1252Stream:
    """Minimal stdout double mimicking a Windows cp1252 console.

    Writes fail on non-cp1252 glyphs until ``reconfigure`` flips it to UTF-8 —
    the exact failure/fix path force_utf8_io() exercises in production.
    """

    def __init__(self) -> None:
        self.encoding = "cp1252"
        self.errors = "strict"
        self.buffer: list[str] = []

    def reconfigure(self, *, encoding=None, errors=None):
        if encoding:
            self.encoding = encoding
        if errors:
            self.errors = errors

    def write(self, text: str) -> int:
        # Emulate TextIOWrapper: encode through the currently active codec.
        text.encode(self.encoding, errors=self.errors)
        self.buffer.append(text)
        return len(text)

    def flush(self) -> None:
        pass


EMOJI_LINE = "[SkillMind] 2/4 Hole Transkript: 🧠 …"


def test_force_utf8_io_switches_stream_to_utf8(monkeypatch):
    out, err = _Cp1252Stream(), _Cp1252Stream()
    monkeypatch.setattr("sys.stdout", out)
    monkeypatch.setattr("sys.stderr", err)

    # Before: the emoji progress line crashes, just like the original bug.
    with pytest.raises(UnicodeEncodeError):
        out.write(EMOJI_LINE)

    force_utf8_io()

    assert out.encoding == "utf-8" and out.errors == "replace"
    assert err.encoding == "utf-8" and err.errors == "replace"
    # After: the same write succeeds.
    assert out.write(EMOJI_LINE) > 0


def test_force_utf8_io_noop_without_reconfigure(monkeypatch):
    """Streams lacking reconfigure (plain pipes, pytest capture) stay untouched."""

    class _Plain:
        encoding = "utf-8"

    plain = _Plain()
    monkeypatch.setattr("sys.stdout", plain)
    monkeypatch.setattr("sys.stderr", plain)
    force_utf8_io()  # must not raise


def test_force_utf8_io_is_idempotent(monkeypatch):
    out = _Cp1252Stream()
    monkeypatch.setattr("sys.stdout", out)
    monkeypatch.setattr("sys.stderr", _Cp1252Stream())
    force_utf8_io()
    force_utf8_io()
    assert out.encoding == "utf-8"


def test_force_utf8_io_survives_unreconfigurable_stream(monkeypatch):
    """A stream whose reconfigure raises must be swallowed, not propagated."""

    class _Stubborn:
        encoding = "cp1252"

        def reconfigure(self, *, encoding=None, errors=None):
            raise ValueError("already detached")

    monkeypatch.setattr("sys.stdout", _Stubborn())
    monkeypatch.setattr("sys.stderr", _Stubborn())
    force_utf8_io()  # must not raise
