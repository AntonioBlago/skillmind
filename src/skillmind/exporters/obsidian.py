"""
SkillMind Obsidian Vault Exporter

Implements the Karpathy LLM Wiki pattern with categorized folders:

    vault/
    ├── CLAUDE.md              # Wiki maintenance instructions
    ├── raw/                   # Raw source material (user-managed)
    └── wiki/
        ├── index.md           # Master index
        ├── log.md             # Operation history
        ├── skills/            # Skill memories
        ├── references/        # Reference memories
        ├── feedback/          # Feedback memories
        ├── projects/          # Project memories
        ├── users/             # User profile memories
        └── topics/            # Topic MOC (Map of Content) pages

Each memory page has:
- YAML frontmatter with Obsidian-native tags, dates, aliases, links
- Content with [[wikilinks]] to related memories
- Category and topic backlinks
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from ..models import Memory, MemoryType


# Map memory types to folder names and display labels
TYPE_FOLDERS = {
    MemoryType.SKILL: ("skills", "Skills"),
    MemoryType.REFERENCE: ("references", "References"),
    MemoryType.FEEDBACK: ("feedback", "Feedback"),
    MemoryType.PROJECT: ("projects", "Projects"),
    MemoryType.USER: ("users", "User Profiles"),
}


class ObsidianExporter:
    """Export SkillMind memories to an Obsidian vault (Karpathy wiki pattern)."""

    def __init__(self, vault_path: str | Path):
        self.vault_path = Path(vault_path)
        self.wiki_path = self.vault_path / "wiki"
        self.raw_path = self.vault_path / "raw"

    # ── Public API ───────────────────────────────────────────────��

    def export(
        self,
        memories: list[Memory],
        full_rebuild: bool = False,
    ) -> dict[str, Any]:
        """
        Export all memories to the Obsidian vault.

        Args:
            memories: List of Memory objects to export
            full_rebuild: If True, clear wiki/ before writing

        Returns:
            Stats dict with counts of created/updated files
        """
        self._ensure_dirs()

        if full_rebuild:
            for folder, _ in TYPE_FOLDERS.values():
                folder_path = self.wiki_path / folder
                if folder_path.exists():
                    for f in folder_path.glob("*.md"):
                        f.unlink()
            for f in self.wiki_path.glob("*.md"):
                f.unlink()
            topics_path = self.wiki_path / "topics"
            if topics_path.exists():
                for f in topics_path.glob("*.md"):
                    f.unlink()

        # Build lookup maps
        title_map = self._build_title_map(memories)
        topic_map = self._group_by_topic(memories)
        type_map = self._group_by_type(memories)

        stats = {"pages_created": 0, "pages_updated": 0, "total": len(memories)}

        # Write each memory as a wiki page in its category folder
        for mem in memories:
            filepath = self._memory_filepath(mem)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            existed = filepath.exists()
            content = self._render_page(mem, title_map, memories)
            filepath.write_text(content, encoding="utf-8")
            if existed:
                stats["pages_updated"] += 1
            else:
                stats["pages_created"] += 1

        # Write index, topic pages, log, CLAUDE.md, Obsidian config
        self._write_index(memories, topic_map, type_map)
        self._append_log(stats, len(memories))

        claude_md = self.vault_path / "CLAUDE.md"
        if not claude_md.exists():
            self._write_claude_md(claude_md)

        self._write_obsidian_config(memories)

        return stats

    def sync(
        self,
        memories: list[Memory],
        existing_ids: set[str] | None = None,
    ) -> dict[str, Any]:
        """
        Incremental sync -- only write new/updated memories.

        Args:
            memories: Full memory list (will compare against existing files)
            existing_ids: Optional set of IDs already in vault (optimization)

        Returns:
            Stats dict
        """
        self._ensure_dirs()

        if existing_ids is None:
            existing_ids = self._scan_existing_ids()

        title_map = self._build_title_map(memories)
        new_memories = [m for m in memories if m.id not in existing_ids]

        stats = {"pages_created": 0, "pages_skipped": len(existing_ids), "total": len(memories)}

        for mem in new_memories:
            filepath = self._memory_filepath(mem)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            content = self._render_page(mem, title_map, memories)
            filepath.write_text(content, encoding="utf-8")
            stats["pages_created"] += 1

        if new_memories:
            topic_map = self._group_by_topic(memories)
            type_map = self._group_by_type(memories)
            self._write_index(memories, topic_map, type_map)
            self._append_log(stats, len(new_memories))

        return stats

    # ── Page Rendering ────────────────────────────────────────────

    def _render_page(
        self,
        mem: Memory,
        title_map: dict[str, str],
        all_memories: list[Memory],
    ) -> str:
        """Render a single memory as an Obsidian-compatible markdown page."""
        lines: list[str] = []
        folder_name, type_label = TYPE_FOLDERS.get(mem.type, ("general", "General"))

        # ── YAML frontmatter (Obsidian-native format) ──
        lines.append("---")
        lines.append(f"id: \"{mem.id}\"")
        lines.append(f"type: {mem.type.value}")
        lines.append(f"category: \"{type_label}\"")
        lines.append(f"topic: \"{mem.topic}\"")
        lines.append(f"confidence: {mem.confidence}")
        lines.append(f"source: {mem.source.value}")

        # Dates in ISO format (Obsidian recognizes these)
        lines.append(f"created: {mem.created_at.strftime('%Y-%m-%dT%H:%M:%S')}")
        lines.append(f"updated: {mem.updated_at.strftime('%Y-%m-%dT%H:%M:%S')}")
        if mem.expires_at:
            lines.append(f"expires: {mem.expires_at.strftime('%Y-%m-%dT%H:%M:%S')}")

        # Obsidian-native tags (YAML array format -- shows in tag pane)
        all_tags = [mem.type.value, mem.topic.replace(" ", "_")]
        all_tags += [t.replace(" ", "_") for t in mem.tags]
        # Deduplicate while preserving order
        seen: set[str] = set()
        unique_tags = []
        for t in all_tags:
            if t not in seen:
                seen.add(t)
                unique_tags.append(t)
        lines.append("tags:")
        for t in unique_tags:
            lines.append(f"  - {t}")

        # Aliases for Obsidian search
        lines.append(f"aliases:")
        lines.append(f"  - \"{self._safe_filename(mem.title)}\"")

        # Links as frontmatter (for Dataview plugin queries)
        lines.append(f"category_link: \"[[{type_label}]]\"")
        lines.append(f"topic_link: \"[[{self._topic_page_name(mem.topic)}]]\"")

        lines.append("---")
        lines.append("")

        # ── Title ──
        lines.append(f"# {mem.title}")
        lines.append("")

        # ── Metadata bar ──
        lines.append(f"> **Category:** [[{type_label}]] | **Topic:** [[{self._topic_page_name(mem.topic)}]] | **Confidence:** {mem.confidence:.0%}")
        lines.append(f"> **Created:** {mem.created_at.strftime('%Y-%m-%d %H:%M')} | **Updated:** {mem.updated_at.strftime('%Y-%m-%d %H:%M')}")
        if mem.tags:
            tag_links = " ".join(f"#{t.replace(' ', '_')}" for t in mem.tags)
            lines.append(f"> **Tags:** {tag_links}")
        lines.append("")

        # ── Content with wikilinks injected ──
        content = self._inject_wikilinks(mem.content, title_map, mem.id)
        lines.append(content)
        lines.append("")

        # ── Related memories (same topic or overlapping tags) ──
        related = [
            m for m in all_memories
            if m.id != mem.id and (m.topic == mem.topic or set(m.tags) & set(mem.tags))
        ]
        if related:
            lines.append("## Related")
            for r in sorted(related[:10], key=lambda x: x.title):
                r_folder, _ = TYPE_FOLDERS.get(r.type, ("general", "General"))
                lines.append(f"- [[{r_folder}/{self._safe_filename(r.title)}|{r.title[:80]}]]")
            lines.append("")

        return "\n".join(lines)

    def _inject_wikilinks(
        self,
        content: str,
        title_map: dict[str, str],
        exclude_id: str,
    ) -> str:
        """Replace mentions of other memory titles with [[wikilinks]]."""
        for mem_id, title in title_map.items():
            if mem_id == exclude_id:
                continue
            if len(title) < 4:
                continue
            safe = self._safe_filename(title)
            pattern = re.escape(title)
            if re.search(pattern, content, re.IGNORECASE):
                content = re.sub(
                    pattern,
                    f"[[{safe}|{title}]]",
                    content,
                    count=1,
                    flags=re.IGNORECASE,
                )
        return content

    # ── Index Generation ──────────────────────────────────────────

    def _write_index(
        self,
        memories: list[Memory],
        topic_map: dict[str, list[Memory]],
        type_map: dict[str, list[Memory]],
    ) -> None:
        """Write the master index.md (Karpathy-style)."""
        lines: list[str] = []
        lines.append("---")
        lines.append("tags:")
        lines.append("  - index")
        lines.append("  - MOC")
        lines.append("---")
        lines.append("")
        lines.append("# Wiki Index")
        lines.append("")
        lines.append(f"*{len(memories)} knowledge entries | Last export: {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC*")
        lines.append("")

        # By Category (Type)
        lines.append("## By Category")
        lines.append("")
        type_order = [MemoryType.SKILL, MemoryType.REFERENCE, MemoryType.FEEDBACK, MemoryType.PROJECT, MemoryType.USER]
        for mt in type_order:
            folder, label = TYPE_FOLDERS[mt]
            mems = type_map.get(mt.value, [])
            if mems:
                lines.append(f"### [[{label}]] ({len(mems)})")
                for m in sorted(mems, key=lambda x: x.title):
                    date_str = m.created_at.strftime('%Y-%m-%d')
                    lines.append(f"- [[{folder}/{self._safe_filename(m.title)}|{m.title[:80]}]] *({date_str})*")
                lines.append("")

        # By Topic
        lines.append("## By Topic")
        lines.append("")
        for topic in sorted(topic_map.keys()):
            mems = topic_map[topic]
            lines.append(f"### [[{self._topic_page_name(topic)}]] ({len(mems)})")
            for m in sorted(mems, key=lambda x: x.title):
                folder, _ = TYPE_FOLDERS.get(m.type, ("general", "General"))
                lines.append(f"- [[{folder}/{self._safe_filename(m.title)}|{m.title[:80]}]]")
            lines.append("")

        # Tag cloud
        all_tags: dict[str, int] = {}
        for m in memories:
            for t in m.tags:
                all_tags[t] = all_tags.get(t, 0) + 1
        if all_tags:
            lines.append("## Tags")
            lines.append("")
            for tag, count in sorted(all_tags.items(), key=lambda x: -x[1]):
                lines.append(f"- #{tag.replace(' ', '_')} ({count})")
            lines.append("")

        (self.wiki_path / "index.md").write_text("\n".join(lines), encoding="utf-8")

        # Write category index pages
        for mt in type_order:
            folder, label = TYPE_FOLDERS[mt]
            mems = type_map.get(mt.value, [])
            self._write_category_index(folder, label, mems)

        # Write topic MOC pages
        topics_dir = self.wiki_path / "topics"
        topics_dir.mkdir(exist_ok=True)
        for topic, mems in topic_map.items():
            self._write_topic_page(topic, mems)

    def _write_category_index(self, folder: str, label: str, memories: list[Memory]) -> None:
        """Write a category index page inside each type folder."""
        folder_path = self.wiki_path / folder
        folder_path.mkdir(exist_ok=True)

        lines: list[str] = []
        lines.append("---")
        lines.append("tags:")
        lines.append("  - MOC")
        lines.append(f"  - {folder}")
        lines.append("---")
        lines.append("")
        lines.append(f"# {label}")
        lines.append("")
        lines.append(f"*{len(memories)} entries*")
        lines.append("")

        # Group by topic within this category
        by_topic: dict[str, list[Memory]] = {}
        for m in memories:
            by_topic.setdefault(m.topic, []).append(m)

        for topic in sorted(by_topic.keys()):
            mems = by_topic[topic]
            lines.append(f"## [[{self._topic_page_name(topic)}|{topic.replace('_', ' ').title()}]]")
            for m in sorted(mems, key=lambda x: x.created_at, reverse=True):
                date_str = m.created_at.strftime('%Y-%m-%d')
                tags_str = " ".join(f"#{t.replace(' ', '_')}" for t in m.tags[:3])
                lines.append(f"- [[{self._safe_filename(m.title)}|{m.title[:70]}]] *({date_str})* {tags_str}")
            lines.append("")

        (folder_path / f"_{label}_.md").write_text("\n".join(lines), encoding="utf-8")

    def _write_topic_page(self, topic: str, memories: list[Memory]) -> None:
        """Write a topic MOC (Map of Content) page in topics/ folder."""
        page_name = self._topic_page_name(topic)
        lines: list[str] = []

        lines.append("---")
        lines.append("tags:")
        lines.append("  - MOC")
        lines.append(f"  - {topic.replace(' ', '_')}")
        lines.append("---")
        lines.append("")
        lines.append(f"# {topic.replace('_', ' ').title()}")
        lines.append("")
        lines.append(f"*{len(memories)} entries in this topic*")
        lines.append("")

        by_type: dict[str, list[Memory]] = {}
        for m in memories:
            by_type.setdefault(m.type.value, []).append(m)

        for tname, mems in sorted(by_type.items()):
            folder, label = TYPE_FOLDERS.get(MemoryType(tname), ("general", "General"))
            lines.append(f"## {label}")
            for m in sorted(mems, key=lambda x: x.created_at, reverse=True):
                date_str = m.created_at.strftime('%Y-%m-%d')
                lines.append(f"- [[{folder}/{self._safe_filename(m.title)}|{m.title[:70]}]] *({date_str})* - {m.content[:60]}...")
            lines.append("")

        filepath = self.wiki_path / "topics" / f"{page_name}.md"
        filepath.write_text("\n".join(lines), encoding="utf-8")

    # ── Log ───────────────────────────────────────────────────────

    def _append_log(self, stats: dict, count: int) -> None:
        """Append an entry to log.md."""
        log_path = self.wiki_path / "log.md"
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

        if not log_path.exists():
            header = "# Operation Log\n\n"
        else:
            header = ""

        entry = (
            f"\n## {timestamp}\n"
            f"- Processed: {count} memories\n"
            f"- Created: {stats.get('pages_created', 0)} pages\n"
            f"- Updated: {stats.get('pages_updated', 0)} pages\n"
            f"- Skipped: {stats.get('pages_skipped', 0)} pages\n"
        )

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(header + entry)

    # ── CLAUDE.md ─────────────────────────────────────────────────

    def _write_claude_md(self, path: Path) -> None:
        """Write the CLAUDE.md that teaches Claude how to maintain this wiki."""
        content = """# SkillMind Wiki Vault

This Obsidian vault is auto-generated and maintained by SkillMind.

## Structure

```
vault/
├── CLAUDE.md              # This file
├── raw/                   # Drop raw source material here
└── wiki/
    ├── index.md           # Master index (by category + topic)
    ├── log.md             # Operation history
    ├── skills/            # Skill memories (patterns, workflows, how-tos)
    ├── references/        # Reference memories (URLs, dashboards, tools)
    ├── feedback/          # Feedback memories (corrections, confirmed approaches)
    ├── projects/          # Project memories (deadlines, client context)
    ├── users/             # User profile memories (role, preferences)
    └── topics/            # Topic MOC pages (Map of Content)
```

## How to Use

### Query the wiki
Ask questions about any topic. Claude reads the index and relevant wiki
pages to find answers across all categories.

### Add new knowledge
1. Drop raw material into `/raw/`
2. Ask Claude to "ingest this into the wiki"
3. Or use SkillMind MCP: `remember` / `learn_youtube` / etc.
4. Run `export_obsidian` or `sync_obsidian` to update the vault

### Maintain the wiki
- `sync_obsidian` — incrementally add new memories
- `export_obsidian` with `full_rebuild=true` — regenerate everything
- The index and topic MOCs are auto-maintained

## Conventions

- Pages are organized in **category folders** (skills/, references/, etc.)
- YAML frontmatter includes Obsidian-native `tags:` array
- `created:` and `updated:` dates in ISO format
- `[[Wikilinks]]` connect related concepts across categories
- Topic MOC pages in `topics/` group memories by subject
- Category index pages (`_Skills_.md`, etc.) list all entries per type
- `#hashtags` used for quick filtering in Obsidian
"""
        path.write_text(content, encoding="utf-8")

    # ── Obsidian Config (.obsidian/) ─────────────────────────────

    def _write_obsidian_config(self, memories: list[Memory]) -> None:
        """Write .obsidian/ config with graph groups, bookmarks, and appearance."""
        obsidian_dir = self.vault_path / ".obsidian"
        obsidian_dir.mkdir(exist_ok=True)

        # Collect unique topics for graph groups
        topics = sorted({m.topic for m in memories})

        # ── Graph view config with colored groups ──
        graph_config = {
            "collapse-filter": False,
            "search": "",
            "showTags": True,
            "showAttachments": False,
            "hideUnresolved": False,
            "showOrphans": True,
            "collapse-color-groups": False,
            "colorGroups": [
                # Category groups (by folder path)
                {"query": "path:wiki/skills", "color": {"a": 1, "rgb": 1745920}},       # Orange #1A9F00 -> actually let me use proper colors
                {"query": "path:wiki/references", "color": {"a": 1, "rgb": 3116773}},    # Blue
                {"query": "path:wiki/feedback", "color": {"a": 1, "rgb": 1752220}},      # Teal
                {"query": "path:wiki/projects", "color": {"a": 1, "rgb": 16087326}},     # Orange-red
                {"query": "path:wiki/users", "color": {"a": 1, "rgb": 8323327}},         # Purple
                {"query": "path:wiki/topics", "color": {"a": 1, "rgb": 5592575}},        # Light blue
                # Tag groups
                {"query": "tag:#MOC", "color": {"a": 1, "rgb": 16776960}},               # Yellow
                {"query": "tag:#index", "color": {"a": 1, "rgb": 16777215}},             # White
            ],
            "collapse-display": False,
            "showArrow": True,
            "textFadeMultiplier": 0,
            "nodeSizeMultiplier": 1.2,
            "lineSizeMultiplier": 1,
            "collapse-forces": False,
            "centerStrength": 0.5,
            "repelStrength": 10,
            "linkStrength": 1,
            "linkDistance": 250,
            "scale": 1,
            "close": False,
        }

        (obsidian_dir / "graph.json").write_text(
            json.dumps(graph_config, indent=2), encoding="utf-8"
        )

        # ── App settings ──
        app_config = {
            "showLineNumber": True,
            "strictLineBreaks": False,
            "readableLineLength": True,
            "showFrontmatter": False,
            "foldHeading": True,
            "foldIndent": True,
            "defaultViewMode": "preview",
            "livePreview": True,
        }

        (obsidian_dir / "app.json").write_text(
            json.dumps(app_config, indent=2), encoding="utf-8"
        )

        # ── Appearance (dark theme) ──
        appearance_config = {
            "baseFontSize": 16,
            "theme": "obsidian",
        }

        (obsidian_dir / "appearance.json").write_text(
            json.dumps(appearance_config, indent=2), encoding="utf-8"
        )

        # ── Bookmarks (pin key pages) ──
        bookmarks = {
            "items": [
                {"type": "file", "ctime": 0, "path": "wiki/index.md", "title": "Wiki Index"},
                {"type": "file", "ctime": 0, "path": "wiki/log.md", "title": "Operation Log"},
                {"type": "file", "ctime": 0, "path": "CLAUDE.md", "title": "Wiki Instructions"},
            ]
        }

        # Add category index bookmarks
        for mt in [MemoryType.SKILL, MemoryType.REFERENCE, MemoryType.FEEDBACK, MemoryType.PROJECT, MemoryType.USER]:
            folder, label = TYPE_FOLDERS[mt]
            bookmarks["items"].append({
                "type": "file", "ctime": 0,
                "path": f"wiki/{folder}/_{label}_.md",
                "title": label,
            })

        (obsidian_dir / "bookmarks.json").write_text(
            json.dumps(bookmarks, indent=2), encoding="utf-8"
        )

        # ── Core plugins (enable graph, bookmarks, tags, search) ──
        core_plugins = [
            "file-explorer", "global-search", "graph", "tag-pane",
            "bookmarks", "outline", "backlink", "page-preview",
            "command-palette", "editor-status", "word-count",
        ]

        (obsidian_dir / "core-plugins.json").write_text(
            json.dumps(core_plugins, indent=2), encoding="utf-8"
        )

        # ── Starred / pinned for older Obsidian versions ──
        starred = {
            "items": [
                {"type": "file", "path": "wiki/index.md"},
                {"type": "file", "path": "wiki/log.md"},
            ]
        }

        (obsidian_dir / "starred.json").write_text(
            json.dumps(starred, indent=2), encoding="utf-8"
        )

    # ── Helpers ───────────────────────────────────────────────────

    def _ensure_dirs(self) -> None:
        """Create vault directory structure."""
        self.vault_path.mkdir(parents=True, exist_ok=True)
        self.wiki_path.mkdir(exist_ok=True)
        self.raw_path.mkdir(exist_ok=True)
        for folder, _ in TYPE_FOLDERS.values():
            (self.wiki_path / folder).mkdir(exist_ok=True)
        (self.wiki_path / "topics").mkdir(exist_ok=True)

    def _memory_filepath(self, mem: Memory) -> Path:
        """Get the file path for a memory's wiki page (in its category folder)."""
        folder, _ = TYPE_FOLDERS.get(mem.type, ("general", "General"))
        return self.wiki_path / folder / f"{self._safe_filename(mem.title)}.md"

    def _scan_existing_ids(self) -> set[str]:
        """Scan existing wiki pages for their memory IDs (from frontmatter)."""
        ids: set[str] = set()
        for f in self.wiki_path.rglob("*.md"):
            if f.name.startswith("_") or f.name in ("index.md", "log.md"):
                continue
            try:
                text = f.read_text(encoding="utf-8")
                match = re.search(r'^id:\s*"?(.+?)"?\s*$', text, re.MULTILINE)
                if match:
                    ids.add(match.group(1).strip())
            except Exception:
                continue
        return ids

    @staticmethod
    def _safe_filename(title: str) -> str:
        """Convert a title to a safe filename (no extension)."""
        safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '', title)
        safe = safe.replace('\u2014', '-').replace('\u2013', '-')
        safe = safe.replace('\u2019', "'").replace('\u201c', '"').replace('\u201d', '"')
        safe = safe.strip(". ")
        return safe[:100] if safe else "untitled"

    @classmethod
    def _topic_page_name(cls, topic: str) -> str:
        """Convert a topic to a MOC page name."""
        safe_topic = cls._safe_filename(topic.replace('_', ' ').title())
        return f"Topic - {safe_topic}"

    @staticmethod
    def _build_title_map(memories: list[Memory]) -> dict[str, str]:
        """Build {id: title} map for wikilink injection."""
        return {m.id: m.title for m in memories}

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
