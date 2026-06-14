"""
Enrichment loop — drive pluggable knowledge sources into the memory store.

This is the "second brain" ingestion engine: it walks any
:class:`~skillmind.sources.base.KnowledgeSource`, hands every discovered
:class:`~skillmind.sources.base.RawDocument` to the
:class:`~skillmind.trainer.Trainer` (which sanitizes, classifies, deduplicates
and stores it), and reports what happened. The resulting memories can then be
exported as an OKF bundle via :class:`~skillmind.exporters.okf.OKFExporter`,
closing the loop:  *any source → SkillMind store → OKF*.

Conceptually this is the portable analogue of knowledge-catalog's
``enrichment_agent`` runner, minus the google-adk / BigQuery coupling.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from .models import MemorySource, MemoryType
from .sources.base import KnowledgeSource, RawDocument
from .trainer import Trainer


class EnrichmentRunner:
    """Run a knowledge source through the Trainer into the store."""

    def __init__(self, trainer: Trainer):
        self.trainer = trainer

    def run(
        self,
        source: KnowledgeSource,
        dry_run: bool = False,
        default_type: MemoryType | None = None,
        default_tags: list[str] | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """
        Enrich the store from a single source.

        Args:
            source: the knowledge source to harvest
            dry_run: discover and report, but do not store
            default_type: force this memory type (else Trainer classifies)
            default_tags: tags added to every harvested document
            limit: stop after this many documents (None = no limit)

        Returns:
            Stats dict: source, discovered, imported, skipped_duplicate,
            skipped_empty, documents.
        """
        stats: dict[str, Any] = {
            "source": source.name,
            "discovered": 0,
            "imported": 0,
            "skipped_duplicate": 0,
            "skipped_empty": 0,
            "documents": [],
        }

        for doc in source.discover():
            if limit is not None and stats["discovered"] >= limit:
                break
            stats["discovered"] += 1

            if not isinstance(doc, RawDocument) or not doc.content.strip():
                stats["skipped_empty"] += 1
                continue

            tags = list(dict.fromkeys((doc.tags or []) + (default_tags or [])))
            metadata = self._build_metadata(source, doc)

            if dry_run:
                stats["documents"].append({
                    "identifier": doc.identifier,
                    "title": doc.title,
                    "tags": tags,
                })
                stats["imported"] += 1
                continue

            memory = self.trainer.learn(
                content=doc.content,
                title=doc.title,
                source=MemorySource.IMPORT,
                force_type=default_type,
                tags=tags or None,
                metadata=metadata,
            )

            if memory:
                stats["imported"] += 1
                stats["documents"].append({
                    "id": memory.id,
                    "title": memory.title,
                    "type": memory.type.value,
                    "topic": memory.topic,
                })
            else:
                stats["skipped_duplicate"] += 1

        return stats

    @staticmethod
    def _build_metadata(source: KnowledgeSource, doc: RawDocument) -> dict[str, Any]:
        metadata: dict[str, Any] = dict(doc.metadata or {})
        metadata.setdefault("enrichment_source", source.name)
        metadata.setdefault("source_identifier", doc.identifier)
        metadata["enriched_at"] = datetime.utcnow().isoformat()
        if doc.source_url:
            metadata.setdefault("source_url", doc.source_url)
        return metadata
