"""Markdown / plain-text directory source.

Harvests a folder of ``.md`` / ``.markdown`` / ``.txt`` files into
:class:`RawDocument` objects, parsing optional YAML frontmatter for ``title`` /
``tags`` / ``source``. This generalizes the legacy Claude-Code memory migration
into the unified enrichment pipeline.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import yaml

from .base import KnowledgeSource, RawDocument

_EXTENSIONS = {".md", ".markdown", ".txt"}
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.+?)\n---\s*\n(.*)$", re.DOTALL)


class MarkdownDirectorySource(KnowledgeSource):
    """Yield one RawDocument per markdown/text file in a directory tree."""

    name = "markdown"

    def __init__(
        self,
        path: str | Path,
        recursive: bool = True,
        skip_names: set[str] | None = None,
    ):
        self.path = Path(path)
        self.recursive = recursive
        # Bundle entry-point files are not standalone documents.
        self.skip_names = {n.lower() for n in (skip_names or {"index.md", "log.md", "readme.md"})}

    def discover(self) -> Iterable[RawDocument]:
        if self.path.is_file():
            files = [self.path]
        else:
            globber = self.path.rglob("*") if self.recursive else self.path.glob("*")
            files = sorted(p for p in globber if p.suffix.lower() in _EXTENSIONS)

        for path in files:
            if path.name.lower() in self.skip_names:
                continue
            doc = self._read(path)
            if doc is not None:
                yield doc

    def _read(self, path: Path) -> RawDocument | None:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return None

        frontmatter: dict = {}
        body = text.strip()
        match = _FRONTMATTER_RE.match(text)
        if match:
            try:
                loaded = yaml.safe_load(match.group(1))
                if isinstance(loaded, dict):
                    frontmatter = loaded
            except yaml.YAMLError:
                frontmatter = {}
            body = match.group(2).strip()

        if not body:
            return None

        title = frontmatter.get("title")
        if not title:
            heading = re.match(r"^#\s+(.+?)\s*$", body, re.MULTILINE)
            title = heading.group(1).strip() if heading else path.stem

        tags = frontmatter.get("tags")
        if isinstance(tags, str):
            tags = [t.strip() for t in re.split(r"[,;]", tags) if t.strip()]
        elif not isinstance(tags, list):
            tags = []

        source_url = frontmatter.get("source") or frontmatter.get("source_url") or frontmatter.get("url")

        return RawDocument(
            identifier=str(path),
            content=body,
            title=str(title),
            source_url=str(source_url) if source_url else None,
            tags=[str(t) for t in tags][:10],
            metadata={"original_file": str(path)},
        )
