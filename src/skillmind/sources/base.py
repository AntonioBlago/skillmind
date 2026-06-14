"""
Knowledge source abstraction — the pluggable input side of the enrichment loop.

This mirrors the pluggable-source pattern from Google's knowledge-catalog
(``okf/src/enrichment_agent/sources/base.py``) but is deliberately decoupled
from google-adk / BigQuery: a source simply *discovers* :class:`RawDocument`
objects, and the :class:`~skillmind.enrichment.EnrichmentRunner` turns each into
a deduplicated, classified memory via the Trainer.

Implement a new source by subclassing :class:`KnowledgeSource` and yielding
``RawDocument`` instances from :meth:`discover`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable


@dataclass
class RawDocument:
    """A single unit of knowledge harvested from a source, pre-classification."""

    identifier: str
    content: str
    title: str | None = None
    source_url: str | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class KnowledgeSource(ABC):
    """Abstract pluggable knowledge source.

    Subclasses yield :class:`RawDocument` objects from :meth:`discover`. The
    ``name`` identifies the adapter (recorded in each memory's metadata so the
    provenance of every concept stays traceable).
    """

    name: str = "source"

    @abstractmethod
    def discover(self) -> Iterable[RawDocument]:
        """Yield raw documents to be enriched into memories."""
        raise NotImplementedError


class CallableSource(KnowledgeSource):
    """Wrap any iterable / generator of ``RawDocument`` as a source.

    Handy for ad-hoc enrichment without writing a full subclass::

        src = CallableSource("notes", lambda: (RawDocument(...) for ...))
    """

    def __init__(self, name: str, factory):
        self.name = name
        self._factory = factory

    def discover(self) -> Iterable[RawDocument]:
        result = self._factory() if callable(self._factory) else self._factory
        return result
