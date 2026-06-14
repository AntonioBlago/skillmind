"""
SkillMind OKF (Open Knowledge Format) Exporter

Exports SkillMind memories as a spec-compliant OKF bundle — the vendor-neutral
knowledge format from Google's knowledge-catalog project (``okf/SPEC.md``).

An OKF bundle is just a directory of markdown files with YAML frontmatter:

    bundle/
    ├── README.md             # human note (not a concept — importers skip it)
    ├── index.md              # entry point, NO frontmatter, sectioned link list
    ├── log.md                # change history, newest-first, ISO date headings
    ├── skills/
    │   └── <concept>.md
    ├── references/
    ├── feedback/
    ├── projects/
    └── users/

Spec essentials honored here:

- A **concept** = one markdown file. Its **concept ID** is the bundle-relative
  path minus ``.md`` (e.g. ``skills/git-rebase`` for ``skills/git-rebase.md``).
- Concept frontmatter: only ``type`` is REQUIRED. We also emit the optional
  ``title``, ``description``, ``resource``, ``tags`` and ``timestamp`` keys,
  plus ``skillmind_*`` producer keys so the bundle round-trips losslessly back
  into a store via :class:`~skillmind.importers.okf.OKFImporter`.
- **Graph edges are standard markdown links** ``[text](relative/path.md)`` — NOT
  Obsidian ``[[wikilinks]]``. Related concepts and inline mentions both use them.
- ``index.md`` has NO frontmatter; entries carry the linked concept's
  description for progressive disclosure.
- ``log.md`` uses ISO 8601 ``YYYY-MM-DD`` date headings, newest first, with the
  leading-bold-word prose convention (``**Creation**``, ``**Update**``…).
- Provenance is rendered as a numbered ``# Citations`` section at the bottom of
  each concept doc.
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from ..models import Memory, MemoryType


# Map memory types to OKF bundle sub-directories and display labels.
TYPE_FOLDERS: dict[MemoryType, tuple[str, str]] = {
    MemoryType.SKILL: ("skills", "Skills"),
    MemoryType.REFERENCE: ("references", "References"),
    MemoryType.FEEDBACK: ("feedback", "Feedback"),
    MemoryType.PROJECT: ("projects", "Projects"),
    MemoryType.USER: ("users", "User Profiles"),
}

# Metadata keys that may carry a canonical resource URI for a concept.
_RESOURCE_KEYS = ("resource", "source_url", "url", "canonical_url", "original_url")

# Metadata keys that may carry provenance worth citing.
_CITATION_KEYS = ("source_url", "url", "canonical_url", "original_url", "original_file")


class OKFExporter:
    """Export SkillMind memories to an Open Knowledge Format bundle."""

    def __init__(self, bundle_path: str | Path, bundle_title: str = "SkillMind Knowledge Base"):
        self.bundle_path = Path(bundle_path)
        self.bundle_title = bundle_title

    # ── Public API ────────────────────────────────────────────────

    def export(
        self,
        memories: list[Memory],
        full_rebuild: bool = False,
    ) -> dict[str, Any]:
        """
        Export all memories to the OKF bundle.

        Args:
            memories: Memory objects to export
            full_rebuild: If True, clear concept folders before writing

        Returns:
            Stats dict (concepts_created, concepts_updated, total).
        """
        self._ensure_dirs()

        if full_rebuild:
            for folder, _ in TYPE_FOLDERS.values():
                folder_path = self.bundle_path / folder
                if folder_path.exists():
                    for f in folder_path.glob("*.md"):
                        f.unlink()

        id_path_map = self._build_id_path_map(memories)
        title_id_map = {m.id: m.title for m in memories}

        stats: dict[str, Any] = {"concepts_created": 0, "concepts_updated": 0, "total": len(memories)}
        created: list[Memory] = []
        updated: list[Memory] = []

        for mem in memories:
            filepath = self._concept_path(mem)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            existed = filepath.exists()
            content = self._render_concept(mem, id_path_map, title_id_map, memories)
            filepath.write_text(content, encoding="utf-8")
            if existed:
                stats["concepts_updated"] += 1
                updated.append(mem)
            else:
                stats["concepts_created"] += 1
                created.append(mem)

        self._write_index(memories)
        self._write_readme(len(memories))
        self._append_log(created, updated, full_rebuild=full_rebuild)

        return stats

    def sync(
        self,
        memories: list[Memory],
        existing_ids: set[str] | None = None,
    ) -> dict[str, Any]:
        """
        Incremental sync — only write concepts not already in the bundle.

        Args:
            memories: Full memory list (compared against existing concept files)
            existing_ids: Optional pre-scanned set of skillmind IDs in the bundle

        Returns:
            Stats dict.
        """
        self._ensure_dirs()

        if existing_ids is None:
            existing_ids = self._scan_existing_ids()

        id_path_map = self._build_id_path_map(memories)
        title_id_map = {m.id: m.title for m in memories}
        new_memories = [m for m in memories if m.id not in existing_ids]

        stats: dict[str, Any] = {
            "concepts_created": 0,
            "concepts_skipped": len(existing_ids),
            "total": len(memories),
        }

        for mem in new_memories:
            filepath = self._concept_path(mem)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            content = self._render_concept(mem, id_path_map, title_id_map, memories)
            filepath.write_text(content, encoding="utf-8")
            stats["concepts_created"] += 1

        if new_memories:
            self._write_index(memories)
            self._write_readme(len(memories))
            self._append_log(new_memories, [], full_rebuild=False)

        return stats

    # ── Concept rendering ─────────────────────────────────────────

    def _render_concept(
        self,
        mem: Memory,
        id_path_map: dict[str, Path],
        title_id_map: dict[str, str],
        all_memories: list[Memory],
    ) -> str:
        """Render a single memory as a spec-compliant OKF concept document."""
        from_path = self._concept_path(mem)
        description = self._derive_description(mem)

        # ── Frontmatter (type REQUIRED; rest optional / producer keys) ──
        front: dict[str, Any] = {
            "type": mem.type.value,
            "title": mem.title,
        }
        if description:
            front["description"] = description
        resource = self._derive_resource(mem)
        if resource:
            front["resource"] = resource
        if mem.tags:
            front["tags"] = list(mem.tags)
        front["timestamp"] = self._iso(mem.updated_at)

        # skillmind producer keys — enable lossless round-trip on import.
        front["skillmind_id"] = mem.id
        front["skillmind_type"] = mem.type.value
        front["skillmind_topic"] = mem.topic
        front["skillmind_source"] = mem.source.value
        front["confidence"] = mem.confidence
        front["created"] = self._iso(mem.created_at)
        front["updated"] = self._iso(mem.updated_at)
        if mem.expires_at:
            front["expires"] = self._iso(mem.expires_at)

        lines: list[str] = ["---"]
        lines.append(
            yaml.safe_dump(front, sort_keys=False, allow_unicode=True, default_flow_style=False).rstrip()
        )
        lines.append("---")
        lines.append("")

        # ── Title + body (with inline relative-link injection) ──
        lines.append(f"# {mem.title}")
        lines.append("")
        body = self._inject_links(mem.content, mem, id_path_map, title_id_map, from_path)
        lines.append(body)
        lines.append("")

        # ── Related concepts (graph edges as relative markdown links) ──
        related = [
            m for m in all_memories
            if m.id != mem.id and (m.topic == mem.topic or set(m.tags) & set(mem.tags))
        ]
        if related:
            lines.append("# Related")
            lines.append("")
            for r in sorted(related[:10], key=lambda x: x.title):
                link = self._rel_link(from_path, self._concept_path(r))
                rdesc = self._derive_description(r)
                suffix = f" — {rdesc}" if rdesc else ""
                lines.append(f"- [{r.title}]({link}){suffix}")
            lines.append("")

        # ── Citations (provenance, numbered, bottom of doc) ──
        citations = self._build_citations(mem)
        if citations:
            lines.append("# Citations")
            lines.append("")
            for i, (label, target) in enumerate(citations, start=1):
                if target:
                    lines.append(f"{i}. [{label}]({target})")
                else:
                    lines.append(f"{i}. {label}")
            lines.append("")

        return "\n".join(lines).rstrip() + "\n"

    def _inject_links(
        self,
        content: str,
        mem: Memory,
        id_path_map: dict[str, Path],
        title_id_map: dict[str, str],
        from_path: Path,
    ) -> str:
        """Turn the first mention of another concept's title into a relative MD link."""
        # Skip titles already inside an existing markdown link to avoid double-linking.
        for other_id, title in title_id_map.items():
            if other_id == mem.id or len(title) < 5:
                continue
            target = id_path_map.get(other_id)
            if target is None:
                continue
            link = self._rel_link(from_path, target)
            pattern = re.escape(title)
            # Negative lookbehind/ahead so we don't relink text already in [..](..) or [[..]].
            if re.search(rf"(?<!\[)(?<!\[\[){pattern}(?!\])(?!\]\])", content, re.IGNORECASE):
                content = re.sub(
                    rf"(?<!\[)(?<!\[\[){pattern}(?!\])(?!\]\])",
                    f"[{title}]({link})",
                    content,
                    count=1,
                    flags=re.IGNORECASE,
                )
        return content

    # ── index.md (NO frontmatter, sectioned, progressive disclosure) ──

    def _write_index(self, memories: list[Memory]) -> None:
        type_map = self._group_by_type(memories)
        topic_map = self._group_by_topic(memories)

        lines: list[str] = []
        lines.append(f"# {self.bundle_title}")
        lines.append("")
        lines.append(
            f"Open Knowledge Format bundle — {len(memories)} concepts. "
            f"Generated {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC."
        )
        lines.append("")

        # By category — each entry carries the concept's description.
        type_order = [
            MemoryType.SKILL, MemoryType.REFERENCE, MemoryType.FEEDBACK,
            MemoryType.PROJECT, MemoryType.USER,
        ]
        for mt in type_order:
            mems = type_map.get(mt.value, [])
            if not mems:
                continue
            _, label = TYPE_FOLDERS[mt]
            lines.append(f"## {label}")
            lines.append("")
            for m in sorted(mems, key=lambda x: x.title):
                rel = self._concept_relpath(m)
                desc = self._derive_description(m)
                suffix = f" — {desc}" if desc else ""
                lines.append(f"- [{m.title}]({rel}){suffix}")
            lines.append("")

        # By topic — a second access path (still standard markdown links).
        if topic_map:
            lines.append("## By Topic")
            lines.append("")
            for topic in sorted(topic_map.keys()):
                mems = topic_map[topic]
                lines.append(f"### {topic.replace('_', ' ').title()}")
                lines.append("")
                for m in sorted(mems, key=lambda x: x.title):
                    rel = self._concept_relpath(m)
                    lines.append(f"- [{m.title}]({rel})")
                lines.append("")

        (self.bundle_path / "index.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    # ── log.md (ISO date headings, newest first, bold-word prose) ──

    def _append_log(
        self,
        created: list[Memory],
        updated: list[Memory],
        full_rebuild: bool,
    ) -> None:
        log_path = self.bundle_path / "log.md"
        today = datetime.utcnow().strftime("%Y-%m-%d")

        # Build today's new prose entries (leading-bold-word convention).
        entries: list[str] = []
        verb = "Rebuild" if full_rebuild else "Export"
        entries.append(
            f"**{verb}** — {len(created)} concept(s) created, {len(updated)} updated."
        )
        for m in created[:50]:
            rel = self._concept_relpath(m)
            entries.append(f"**Creation** — added [{m.title}]({rel}).")
        if len(created) > 50:
            entries.append(f"**Note** — {len(created) - 50} further creations omitted from log.")

        new_block = "\n".join(f"- {e}" for e in entries)

        # Merge into existing log, keeping date sections newest-first.
        if log_path.exists():
            existing = log_path.read_text(encoding="utf-8")
            body = existing.split("\n", 2)[-1] if existing.startswith("# Log") else existing
            body = body.lstrip("\n")
        else:
            body = ""

        today_heading = f"## {today}"
        if body.startswith(today_heading):
            # Insert today's new lines right under the existing today heading.
            after = body[len(today_heading):].lstrip("\n")
            merged_body = f"{today_heading}\n\n{new_block}\n\n{after}".rstrip() + "\n"
        else:
            merged_body = f"{today_heading}\n\n{new_block}\n\n{body}".rstrip() + "\n"

        out = f"# Log\n\n{merged_body}"
        log_path.write_text(out, encoding="utf-8")

    # ── README (human note, not a concept) ───────────────────────

    def _write_readme(self, count: int) -> None:
        readme = self.bundle_path / "README.md"
        content = f"""# {self.bundle_title}

This directory is an **Open Knowledge Format (OKF)** bundle produced by
[SkillMind](https://skill-mind.com). It currently holds **{count} concepts**.

## How to read it

- Start at [index.md](index.md) — it lists every concept grouped by category
  and topic, each with a one-line description.
- Each concept is a single markdown file with YAML frontmatter. The **concept
  ID** is the file path minus `.md` (e.g. `skills/git-rebase`).
- Concepts link to each other with **standard markdown relative links**
  `[text](../folder/concept.md)` — these are the knowledge graph edges.
- [log.md](log.md) records the change history, newest first.

## How to re-import it

```bash
skillmind import-okf <path-to-this-bundle>
```

The `skillmind_*` frontmatter keys let SkillMind reconstruct each memory's id,
type, topic, source and timestamps losslessly. Bundles from other OKF producers
(without those keys) import too — SkillMind classifies them on the way in.
"""
        readme.write_text(content, encoding="utf-8")

    # ── Provenance / citations ────────────────────────────────────

    def _build_citations(self, mem: Memory) -> list[tuple[str, str | None]]:
        """Collect (label, target) citation pairs from a memory's provenance."""
        citations: list[tuple[str, str | None]] = []
        seen: set[str] = set()

        # Explicit citation list in metadata (strings or {label,url} dicts).
        raw = mem.metadata.get("citations")
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict):
                    target = item.get("url") or item.get("target") or item.get("href")
                    label = item.get("label") or item.get("title") or target or "Source"
                    if target and target in seen:
                        continue
                    if target:
                        seen.add(target)
                    citations.append((str(label), str(target) if target else None))
                elif isinstance(item, str) and item not in seen:
                    seen.add(item)
                    citations.append((item, item if _looks_like_uri(item) else None))

        # Single-value provenance keys.
        for key in _CITATION_KEYS:
            val = mem.metadata.get(key)
            if isinstance(val, str) and val and val not in seen:
                seen.add(val)
                label = "Source file" if key == "original_file" else val
                citations.append((label, val if _looks_like_uri(val) else val))

        return citations

    # ── Helpers ───────────────────────────────────────────────────

    def _ensure_dirs(self) -> None:
        self.bundle_path.mkdir(parents=True, exist_ok=True)
        for folder, _ in TYPE_FOLDERS.values():
            (self.bundle_path / folder).mkdir(exist_ok=True)

    def _concept_path(self, mem: Memory) -> Path:
        folder, _ = TYPE_FOLDERS.get(mem.type, ("general", "General"))
        return self.bundle_path / folder / f"{self._safe_filename(mem.title)}.md"

    def _concept_relpath(self, mem: Memory) -> str:
        """Bundle-root-relative POSIX path to a concept (for index.md links)."""
        folder, _ = TYPE_FOLDERS.get(mem.type, ("general", "General"))
        return f"{folder}/{self._safe_filename(mem.title)}.md"

    def _build_id_path_map(self, memories: list[Memory]) -> dict[str, Path]:
        return {m.id: self._concept_path(m) for m in memories}

    @staticmethod
    def _rel_link(from_path: Path, to_path: Path) -> str:
        """Relative POSIX markdown link from one concept file to another."""
        rel = os.path.relpath(to_path, start=from_path.parent)
        return rel.replace(os.sep, "/")

    def _scan_existing_ids(self) -> set[str]:
        ids: set[str] = set()
        for folder, _ in TYPE_FOLDERS.values():
            folder_path = self.bundle_path / folder
            if not folder_path.exists():
                continue
            for f in folder_path.glob("*.md"):
                try:
                    text = f.read_text(encoding="utf-8")
                except OSError:
                    continue
                match = re.search(r"^skillmind_id:\s*['\"]?(.+?)['\"]?\s*$", text, re.MULTILINE)
                if match:
                    ids.add(match.group(1).strip())
        return ids

    @staticmethod
    def _derive_description(mem: Memory) -> str:
        """One-line description: explicit metadata, else first sentence of content."""
        explicit = mem.metadata.get("description")
        if isinstance(explicit, str) and explicit.strip():
            return _truncate(explicit.strip().replace("\n", " "), 160)

        for raw_line in mem.content.splitlines():
            line = raw_line.strip().lstrip("#").strip()
            if not line:
                continue
            sentence = re.split(r"(?<=[.!?])\s", line)[0].strip()
            return _truncate(sentence or line, 160)
        return ""

    @staticmethod
    def _derive_resource(mem: Memory) -> str | None:
        for key in _RESOURCE_KEYS:
            val = mem.metadata.get(key)
            if isinstance(val, str) and _looks_like_uri(val):
                return val
        return None

    @staticmethod
    def _iso(dt: datetime) -> str:
        """ISO 8601 timestamp (treat naive datetimes as UTC, per store convention)."""
        if dt.tzinfo is None:
            return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        return dt.isoformat()

    @staticmethod
    def _safe_filename(title: str) -> str:
        # Normalise smart punctuation FIRST, then strip filesystem-illegal chars.
        # (Order matters: converting curly quotes to ASCII must happen before the
        # strip, or e.g. „…" would re-introduce a literal " — illegal on Windows.)
        safe = title.replace("—", "-").replace("–", "-")
        safe = safe.replace("’", "'").replace("‘", "'").replace("‚", "'")
        safe = safe.replace("“", '"').replace("”", '"').replace("„", '"')
        safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", safe)
        safe = safe.strip(". ")
        return safe[:100] if safe else "untitled"

    @staticmethod
    def _group_by_topic(memories: list[Memory]) -> dict[str, list[Memory]]:
        groups: dict[str, list[Memory]] = {}
        for m in memories:
            groups.setdefault(m.topic, []).append(m)
        return groups

    @staticmethod
    def _group_by_type(memories: list[Memory]) -> dict[str, list[Memory]]:
        groups: dict[str, list[Memory]] = {}
        for m in memories:
            groups.setdefault(m.type.value, []).append(m)
        return groups


def _looks_like_uri(value: str) -> bool:
    return bool(re.match(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://", value)) or value.startswith("//")


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text[: limit - 1]
    if " " in cut:
        cut = cut[: cut.rfind(" ")]
    return cut + "…"
