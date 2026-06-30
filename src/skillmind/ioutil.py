"""Process-level I/O helpers."""

from __future__ import annotations

import sys


def force_utf8_io() -> None:
    """Make stdout/stderr tolerate any Unicode glyph.

    On Windows the console streams default to a legacy code page (cp1252),
    so a single ``print()`` carrying an emoji or a ``→`` arrow raises
    ``UnicodeEncodeError`` and aborts an otherwise-successful run — e.g. the
    YouTube learner's progress line ``[SkillMind] 2/4 Hole Transkript: 🧠…``
    crashes at the exact column the emoji lands on.

    Reconfiguring both streams to UTF-8 with ``errors="replace"`` makes every
    print site in the process encoding-safe at once. Idempotent, and a no-op on
    streams that don't support ``reconfigure`` (plain pipes, pytest capture).
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            # Stream already detached or not reconfigurable — leave it as-is.
            pass
