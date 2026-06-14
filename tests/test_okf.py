"""Tests for the OKF (Open Knowledge Format) integration.

Covers three areas:

* **Exporter** — spec-compliance of the produced bundle (concept frontmatter
  carries ``type``, ``index.md`` has NO frontmatter, graph edges are standard
  relative markdown links, ``# Citations`` appears when provenance exists,
  ``log.md`` uses ISO date headings).
* **Importer** — round-trip of SkillMind-produced bundles (type/topic preserved
  via ``skillmind_*`` keys), import of foreign bundles (freeform ``type`` mapped),
  reserved files skipped, dry-run is non-mutating.
* **Enrichment** — a markdown directory source flows through the Trainer into
  the store; dry-run is non-mutating; the source factory behaves.
"""

from __future__ import annotations

import yaml
import pytest

from skillmind.enrichment import EnrichmentRunner
from skillmind.exporters.okf import OKFExporter, TYPE_FOLDERS
from skillmind.exporters.okf_viz import TYPE_COLORS, OKFVisualizer
from skillmind.importers.okf import (
    OKF_TYPE_MAP,
    discover_concept_files,
    import_okf_bundle,
    parse_concept_file,
)
from skillmind.models import Memory, MemorySource, MemoryType, QueryFilter, QueryResult
from skillmind.sources import (
    MarkdownDirectorySource,
    WebSource,
    create_source,
)
from skillmind.store.base import MemoryStore
from skillmind.trainer import Trainer


# ── Fixtures ──────────────────────────────────────────────────────


class InMemoryStore(MemoryStore):
    """Backend-free MemoryStore double for tests.

    Avoids the chroma/faiss/qdrant dependency entirely (per project preference,
    chroma-based test fixtures are skipped). ``query`` resolves to ``[]`` so the
    inherited ``find_duplicates`` never merges — every ``Trainer.learn`` stores a
    fresh memory, which keeps round-trip and count assertions deterministic and
    needs no embedding model download.
    """

    def __init__(self):  # noqa: D401 — deliberately bypasses base (no config/engine)
        self._mem: dict[str, Memory] = {}

    def initialize(self) -> None:
        pass

    def add(self, memory: Memory) -> str:
        self._mem[memory.id] = memory
        return memory.id

    def add_batch(self, memories: list[Memory]) -> list[str]:
        return [self.add(m) for m in memories]

    def query(self, text: str, limit: int = 5, filter: QueryFilter | None = None) -> list[QueryResult]:
        return []

    def get(self, memory_id: str) -> Memory | None:
        return self._mem.get(memory_id)

    def update(self, memory: Memory) -> None:
        self._mem[memory.id] = memory

    def delete(self, memory_id: str) -> bool:
        return self._mem.pop(memory_id, None) is not None

    def list_all(self, filter: QueryFilter | None = None, limit: int = 100, offset: int = 0) -> list[Memory]:
        mems = list(self._mem.values())
        if filter and filter.types:
            allowed = set(filter.types)
            mems = [m for m in mems if m.type in allowed]
        return mems[offset:offset + limit]

    def count(self, filter: QueryFilter | None = None) -> int:
        return len(self.list_all(filter=filter, limit=10 ** 9))

    def clear(self) -> int:
        n = len(self._mem)
        self._mem.clear()
        return n


@pytest.fixture
def trainer_with_store():
    """Trainer over a backend-free in-memory store (sanitizer off for determinism)."""
    return Trainer(InMemoryStore(), sanitize=False)


def _parse_frontmatter(text: str) -> dict:
    """Extract and parse YAML frontmatter from a markdown document."""
    assert text.startswith("---\n"), "expected frontmatter delimiter"
    _, fm, _ = text.split("---\n", 2) if text.count("---\n") >= 2 else ("", "", "")
    # Robust split: take everything between the first two '---' lines.
    parts = text.split("---", 2)
    return yaml.safe_load(parts[1]) or {}


# ── Exporter: bundle structure & spec-compliance ─────────────────


class TestOKFExportStructure:
    def test_creates_bundle_skeleton(self, tmp_dir, sample_memories):
        bundle = tmp_dir / "kb"
        exporter = OKFExporter(bundle, bundle_title="Test KB")
        stats = exporter.export(sample_memories)

        assert stats["total"] == len(sample_memories)
        assert stats["concepts_created"] == len(sample_memories)
        assert (bundle / "index.md").exists()
        assert (bundle / "log.md").exists()
        assert (bundle / "README.md").exists()
        # Every type-folder that received a concept must exist.
        for folder, _ in TYPE_FOLDERS.values():
            assert (bundle / folder).exists()

    def test_concept_files_written_per_type(self, tmp_dir, sample_memories):
        bundle = tmp_dir / "kb"
        OKFExporter(bundle).export(sample_memories)

        assert (bundle / "skills" / "Content Mapping Workflow.md").exists()
        assert (bundle / "users" / "SEO Freelancer.md").exists()
        assert (bundle / "references" / "Notion To-Do Databases.md").exists()


class TestOKFSpecCompliance:
    def test_concept_frontmatter_has_required_type(self, tmp_dir, sample_memories):
        bundle = tmp_dir / "kb"
        OKFExporter(bundle).export(sample_memories)

        text = (bundle / "skills" / "Content Mapping Workflow.md").read_text(encoding="utf-8")
        fm = _parse_frontmatter(text)
        # `type` is the ONLY required OKF frontmatter key.
        assert fm["type"] == "skill"
        assert fm["title"] == "Content Mapping Workflow"
        # Producer keys enable lossless round-trip.
        assert fm["skillmind_id"] == "mem-skill-1"
        assert fm["skillmind_topic"] == "seo"
        assert "timestamp" in fm

    def test_index_has_no_frontmatter(self, tmp_dir, sample_memories):
        bundle = tmp_dir / "kb"
        OKFExporter(bundle, bundle_title="My Brain").export(sample_memories)

        index = (bundle / "index.md").read_text(encoding="utf-8")
        # Per spec the index is NOT a concept — it must not start with '---'.
        assert not index.lstrip().startswith("---")
        assert index.startswith("# My Brain")

    def test_index_carries_descriptions(self, tmp_dir, sample_memories):
        bundle = tmp_dir / "kb"
        OKFExporter(bundle).export(sample_memories)
        index = (bundle / "index.md").read_text(encoding="utf-8")
        # Progressive disclosure: index entries include a one-line description.
        assert "[Content Mapping Workflow](skills/Content Mapping Workflow.md)" in index
        assert " — " in index

    def test_graph_edges_are_relative_markdown_links(self, tmp_dir, sample_memories):
        bundle = tmp_dir / "kb"
        OKFExporter(bundle).export(sample_memories)

        # user + skill share the 'seo' tag → they reference each other.
        text = (bundle / "skills" / "Content Mapping Workflow.md").read_text(encoding="utf-8")
        assert "# Related" in text
        assert "](../users/SEO Freelancer.md)" in text
        # OKF uses standard markdown links, NEVER Obsidian wikilinks.
        assert "[[" not in text
        # No absolute URLs masquerading as graph edges in Related.
        related = text.split("# Related", 1)[1]
        assert "](http" not in related.split("# Citations")[0]

    def test_citations_emitted_for_provenance(self, tmp_dir):
        bundle = tmp_dir / "kb"
        mem = Memory(
            id="mem-cite",
            type=MemoryType.REFERENCE,
            topic="seo",
            title="CTR Study",
            content="Empirical CTR curve by SERP position.",
            metadata={"source_url": "https://example.com/ctr-study"},
        )
        OKFExporter(bundle).export([mem])

        text = (bundle / "references" / "CTR Study.md").read_text(encoding="utf-8")
        assert "# Citations" in text
        assert "https://example.com/ctr-study" in text
        # The same URI also becomes the canonical `resource` frontmatter key.
        fm = _parse_frontmatter(text)
        assert fm["resource"] == "https://example.com/ctr-study"

    def test_no_citations_without_provenance(self, tmp_dir, sample_memories):
        bundle = tmp_dir / "kb"
        OKFExporter(bundle).export(sample_memories)
        # sample memories carry no source_url/citations → no Citations section.
        text = (bundle / "users" / "SEO Freelancer.md").read_text(encoding="utf-8")
        assert "# Citations" not in text

    def test_log_has_iso_date_heading(self, tmp_dir, sample_memories):
        bundle = tmp_dir / "kb"
        OKFExporter(bundle).export(sample_memories)
        log = (bundle / "log.md").read_text(encoding="utf-8")
        assert log.startswith("# Log")
        # ISO 8601 date heading + leading-bold-word prose convention.
        import re

        assert re.search(r"^## \d{4}-\d{2}-\d{2}$", log, re.MULTILINE)
        assert "**Export**" in log or "**Rebuild**" in log
        assert "**Creation**" in log


class TestOKFFilenames:
    def test_safe_filename_strips_forbidden_chars(self):
        assert OKFExporter._safe_filename('a/b:c*?"<>|d') == "abcd"
        assert OKFExporter._safe_filename("Heading — dash") == "Heading - dash"
        assert OKFExporter._safe_filename("") == "untitled"


class TestOKFSync:
    def test_sync_skips_existing(self, tmp_dir, sample_memories):
        bundle = tmp_dir / "kb"
        exporter = OKFExporter(bundle)
        exporter.export(sample_memories)

        stats = exporter.sync(sample_memories)
        assert stats["concepts_created"] == 0
        assert stats["concepts_skipped"] == len(sample_memories)

    def test_sync_writes_new(self, tmp_dir, sample_memories):
        bundle = tmp_dir / "kb"
        exporter = OKFExporter(bundle)
        exporter.export(sample_memories)

        extra = Memory(
            id="mem-new-1",
            type=MemoryType.SKILL,
            topic="git",
            title="Git Rebase Workflow",
            content="Use interactive rebase to clean up history before merging.",
        )
        stats = exporter.sync(sample_memories + [extra])
        assert stats["concepts_created"] == 1
        assert (bundle / "skills" / "Git Rebase Workflow.md").exists()


# ── Importer: round-trip, foreign bundles, reserved files ────────


class TestOKFImportRoundTrip:
    def test_discover_skips_reserved_files(self, tmp_dir, sample_memories):
        bundle = tmp_dir / "kb"
        OKFExporter(bundle).export(sample_memories)

        files = discover_concept_files(bundle)
        names = {f.name.lower() for f in files}
        assert "index.md" not in names
        assert "log.md" not in names
        assert "readme.md" not in names
        assert len(files) == len(sample_memories)

    def test_round_trip_preserves_type_and_topic(self, tmp_dir, sample_memories, trainer_with_store):
        bundle = tmp_dir / "kb"
        OKFExporter(bundle).export(sample_memories)

        stats = import_okf_bundle(trainer_with_store, bundle)
        assert stats["skipped_error"] == 0
        assert stats["imported"] + stats["skipped_duplicate"] == stats["files_found"]

        stored = trainer_with_store.store.list_all(limit=100)
        types = {m.type for m in stored}
        assert MemoryType.SKILL in types
        assert MemoryType.USER in types
        assert MemoryType.REFERENCE in types
        # skillmind_topic survives the round-trip for the skill concept.
        skills = [m for m in stored if m.type == MemoryType.SKILL]
        assert any(m.topic == "seo" for m in skills)

    def test_dry_run_does_not_store(self, tmp_dir, sample_memories, trainer_with_store):
        bundle = tmp_dir / "kb"
        OKFExporter(bundle).export(sample_memories)

        before = trainer_with_store.store.count()
        stats = import_okf_bundle(trainer_with_store, bundle, dry_run=True)
        assert stats["imported"] > 0
        assert trainer_with_store.store.count() == before


class TestOKFImportForeign:
    def _write_foreign_concept(self, base):
        folder = base / "glossary"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "tf-idf.md").write_text(
            "---\n"
            "type: glossary\n"
            "title: TF-IDF\n"
            "tags: [nlp, ranking]\n"
            "resource: https://en.wikipedia.org/wiki/Tf%E2%80%93idf\n"
            "---\n\n"
            "# TF-IDF\n\n"
            "Term frequency-inverse document frequency weights terms by rarity.\n\n"
            "# Related\n\n- [Something](other.md)\n",
            encoding="utf-8",
        )
        return base

    def test_foreign_type_mapped_and_sections_stripped(self, tmp_dir, trainer_with_store):
        base = self._write_foreign_concept(tmp_dir / "foreign")
        stats = import_okf_bundle(trainer_with_store, base)
        assert stats["imported"] == 1

        stored = trainer_with_store.store.list_all(limit=10)
        assert len(stored) == 1
        mem = stored[0]
        # 'glossary' is not a MemoryType, so OKF_TYPE_MAP routes it to REFERENCE.
        assert OKF_TYPE_MAP["glossary"] == MemoryType.REFERENCE
        assert mem.type == MemoryType.REFERENCE
        # Folder name became the topic hint.
        assert mem.topic == "glossary"
        # Producer-generated trailer + leading title were stripped from content.
        assert "# Related" not in mem.content
        assert not mem.content.lstrip().startswith("# TF-IDF")
        assert "Term frequency" in mem.content
        # `resource` was captured as source provenance metadata.
        assert mem.metadata.get("source_url", "").startswith("https://")


class TestParseConceptFile:
    def test_parse_without_frontmatter(self, tmp_dir):
        p = tmp_dir / "note.md"
        p.write_text("# Plain Note\n\nJust some body text.\n", encoding="utf-8")
        parsed = parse_concept_file(p)
        assert parsed is not None
        assert parsed["frontmatter"] == {}
        assert parsed["title"] == "Plain Note"

    def test_parse_with_frontmatter(self, tmp_dir):
        p = tmp_dir / "note.md"
        p.write_text(
            "---\ntype: skill\ntitle: My Skill\n---\n\nBody here.\n",
            encoding="utf-8",
        )
        parsed = parse_concept_file(p)
        assert parsed is not None
        assert parsed["frontmatter"]["type"] == "skill"
        assert parsed["title"] == "My Skill"


# ── Enrichment loop: pluggable sources → memories ────────────────


class TestEnrichmentMarkdownSource:
    def _seed_docs(self, base):
        base.mkdir(parents=True, exist_ok=True)
        (base / "a.md").write_text(
            "---\ntitle: Docker Basics\ntags: [docker, devops]\n---\n\n"
            "Use multi-stage builds to keep container images small.\n",
            encoding="utf-8",
        )
        (base / "b.md").write_text(
            "# Pytest Tips\n\nUse fixtures to share setup across tests.\n",
            encoding="utf-8",
        )
        # Reserved bundle file → must be skipped by the source.
        (base / "index.md").write_text("# Index\n\nshould be skipped\n", encoding="utf-8")
        return base

    def test_markdown_source_discovers_docs(self, tmp_dir):
        base = self._seed_docs(tmp_dir / "notes")
        docs = list(MarkdownDirectorySource(base).discover())
        ids = {d.title for d in docs}
        assert "Docker Basics" in ids
        assert "Pytest Tips" in ids
        assert "Index" not in ids  # reserved file skipped

    def test_enrichment_run_imports(self, tmp_dir, trainer_with_store):
        base = self._seed_docs(tmp_dir / "notes")
        source = MarkdownDirectorySource(base)
        runner = EnrichmentRunner(trainer_with_store)

        before = trainer_with_store.store.count()
        stats = runner.run(source, default_tags=["enriched"])
        assert stats["discovered"] == 2
        assert stats["imported"] >= 1
        assert trainer_with_store.store.count() > before

    def test_enrichment_dry_run_is_non_mutating(self, tmp_dir, trainer_with_store):
        base = self._seed_docs(tmp_dir / "notes")
        source = MarkdownDirectorySource(base)
        runner = EnrichmentRunner(trainer_with_store)

        before = trainer_with_store.store.count()
        stats = runner.run(source, dry_run=True)
        assert stats["imported"] == 2
        assert trainer_with_store.store.count() == before

    def test_enrichment_respects_limit(self, tmp_dir, trainer_with_store):
        base = self._seed_docs(tmp_dir / "notes")
        source = MarkdownDirectorySource(base)
        runner = EnrichmentRunner(trainer_with_store)
        stats = runner.run(source, limit=1)
        assert stats["discovered"] == 1
        assert stats["imported"] == 1

    def test_enrichment_forces_type(self, tmp_dir, trainer_with_store):
        base = self._seed_docs(tmp_dir / "notes")
        runner = EnrichmentRunner(trainer_with_store)
        runner.run(MarkdownDirectorySource(base), default_type=MemoryType.SKILL)
        stored = trainer_with_store.store.list_all(limit=10)
        assert stored
        assert all(m.type == MemoryType.SKILL for m in stored)


class TestSourceFactory:
    def test_markdown_aliases(self, tmp_dir):
        for alias in ("markdown", "md", "dir", "directory"):
            assert isinstance(create_source(alias, str(tmp_dir)), MarkdownDirectorySource)

    def test_web_aliases(self):
        for alias in ("web", "url", "http"):
            assert isinstance(create_source(alias, "https://example.com"), WebSource)

    def test_unknown_source_raises(self):
        with pytest.raises(ValueError):
            create_source("telepathy", "wherever")


class TestWebSourceHtmlToText:
    def test_strips_tags_and_scripts(self):
        html_doc = (
            "<html><head><title>T</title><style>x{}</style></head>"
            "<body><script>evil()</script><p>Hello</p><p>World</p></body></html>"
        )
        text = WebSource._html_to_text(html_doc)
        assert "Hello" in text
        assert "World" in text
        assert "evil()" not in text
        assert "<p>" not in text


# ── Local visualizer: bundle → self-contained interactive HTML ───


class TestOKFVisualizer:
    def _bundle(self, tmp_dir, sample_memories):
        bundle = tmp_dir / "kb"
        OKFExporter(bundle).export(sample_memories)
        return bundle

    def test_build_graph_has_node_per_concept(self, tmp_dir, sample_memories):
        graph = OKFVisualizer(self._bundle(tmp_dir, sample_memories)).build_graph()
        assert graph["stats"]["concepts"] == len(sample_memories)
        ids = {n["data"]["id"] for n in graph["nodes"]}
        assert "skills/Content Mapping Workflow" in ids
        assert "users/SEO Freelancer" in ids
        # Node types reflect the skillmind/OKF concept types.
        assert set(graph["types"]) >= {"user", "feedback", "project", "reference", "skill"}

    def test_edges_resolve_relative_links(self, tmp_dir, sample_memories):
        graph = OKFVisualizer(self._bundle(tmp_dir, sample_memories)).build_graph()
        pairs = {(e["data"]["source"], e["data"]["target"]) for e in graph["edges"]}
        # skill + user share the 'seo' tag → exporter emits a Related edge.
        assert ("skills/Content Mapping Workflow", "users/SEO Freelancer") in pairs
        # No dangling edges: every endpoint is a real node.
        ids = {n["data"]["id"] for n in graph["nodes"]}
        for src, tgt in pairs:
            assert src in ids and tgt in ids

    def test_nodes_colored_by_type(self, tmp_dir, sample_memories):
        graph = OKFVisualizer(self._bundle(tmp_dir, sample_memories)).build_graph()
        by_id = {n["data"]["id"]: n["data"] for n in graph["nodes"]}
        assert by_id["users/SEO Freelancer"]["color"] == TYPE_COLORS["user"]
        assert by_id["skills/Content Mapping Workflow"]["color"] == TYPE_COLORS["skill"]

    def test_bodies_present_with_title_stripped(self, tmp_dir, sample_memories):
        graph = OKFVisualizer(self._bundle(tmp_dir, sample_memories)).build_graph()
        body = graph["bodies"]["skills/Content Mapping Workflow"]
        assert body  # detail panel renders the body
        # The duplicate leading "# <title>" is removed (the panel shows the title itself).
        assert not body.lstrip().startswith("# Content Mapping Workflow")

    def test_build_writes_self_contained_html(self, tmp_dir, sample_memories):
        out = OKFVisualizer(self._bundle(tmp_dir, sample_memories), title="My Brain").build()
        assert out.name == "okf-graph.html"
        doc = out.read_text(encoding="utf-8")
        # Graph data is embedded (no fetch) and the libs are referenced.
        assert "window.BUNDLE" in doc
        assert "cytoscape" in doc
        assert "users/SEO Freelancer" in doc
        assert "My Brain" in doc

    def test_resolve_link_walks_relative_path(self):
        assert OKFVisualizer._resolve_link("skills", "../users/SEO Freelancer.md") == "users/SEO Freelancer"
        assert OKFVisualizer._resolve_link("skills", "../projects/p.md") == "projects/p"
        # Absolute URLs are not graph edges.
        assert OKFVisualizer._resolve_link("skills", "https://example.com") is None

    def test_render_html_escapes_script_breakout(self, tmp_dir):
        viz = OKFVisualizer(tmp_dir)  # _render_html does not touch the filesystem
        graph = {
            "types": ["skill"],
            "nodes": [{"data": {"id": "s/x", "label": "X", "type": "skill",
                                 "color": "#000", "size": 30, "description": "",
                                 "resource": "", "tags": []}}],
            "edges": [],
            "bodies": {"s/x": "danger </script><img>"},
            "stats": {"concepts": 1, "edges": 0, "types": 1},
        }
        doc = viz._render_html(graph)
        # A literal "</script>" inside concept data must be neutralized so it
        # cannot break out of the embedding <script> tag.
        assert "</script><img>" not in doc
        assert "<\\/script><img>" in doc

    def test_missing_bundle_raises(self, tmp_dir):
        with pytest.raises(NotADirectoryError):
            OKFVisualizer(tmp_dir / "does-not-exist").build_graph()

    def test_type_derived_from_folder_for_foreign_bundle(self, tmp_dir):
        base = tmp_dir / "foreign" / "glossary"
        base.mkdir(parents=True)
        (base / "tf-idf.md").write_text(
            "---\ntitle: TF-IDF\n---\n\n# TF-IDF\n\nWeights terms by rarity.\n",
            encoding="utf-8",
        )
        graph = OKFVisualizer(tmp_dir / "foreign").build_graph()
        node = graph["nodes"][0]["data"]
        # No skillmind_type / type frontmatter → fall back to the folder name.
        assert node["type"] == "glossary"
