"""Pluggable knowledge sources for the SkillMind enrichment loop."""

from __future__ import annotations

from .base import CallableSource, KnowledgeSource, RawDocument
from .markdown_source import MarkdownDirectorySource
from .web_source import WebSource

__all__ = [
    "KnowledgeSource",
    "RawDocument",
    "CallableSource",
    "MarkdownDirectorySource",
    "WebSource",
    "create_source",
]


def create_source(source_type: str, target, **kwargs) -> KnowledgeSource:
    """Factory: build a knowledge source by type name.

    Args:
        source_type: ``markdown`` | ``web``
        target: path (markdown) or URL / list of URLs (web)
        **kwargs: forwarded to the concrete source constructor
    """
    st = source_type.lower()
    if st in ("markdown", "md", "dir", "directory"):
        return MarkdownDirectorySource(target, **kwargs)
    if st in ("web", "url", "http"):
        return WebSource(target, **kwargs)
    raise ValueError(f"Unknown source type: {source_type}. Supported: markdown, web")
