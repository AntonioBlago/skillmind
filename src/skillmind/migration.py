"""
Migration script: import existing Claude Code markdown memories into SkillMind.

Reads the flat .claude/projects/*/memory/*.md files and converts them
into structured Memory objects in the vector store.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import yaml

from .models import Memory, MemoryType, MemorySource, QueryFilter
from .store.base import MemoryStore
from .trainer import Trainer


# Map markdown frontmatter type to MemoryType
TYPE_MAP = {
    "user": MemoryType.USER,
    "feedback": MemoryType.FEEDBACK,
    "project": MemoryType.PROJECT,
    "reference": MemoryType.REFERENCE,
    "skill": MemoryType.SKILL,
}


def parse_memory_file(path: Path) -> dict[str, Any] | None:
    """
    Parse a Claude Code memory markdown file with YAML frontmatter.

    Expected format:
        ---
        name: memory_name
        description: one-line description
        type: user|feedback|project|reference
        ---
        Content here...
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None

    # Parse YAML frontmatter
    match = re.match(r"^---\s*\n(.+?)\n---\s*\n(.*)$", text, re.DOTALL)
    if not match:
        # No frontmatter — treat entire file as content
        return {
            "name": path.stem,
            "description": "",
            "type": "feedback",
            "content": text.strip(),
            "file_path": str(path),
        }

    try:
        frontmatter = yaml.safe_load(match.group(1))
    except yaml.YAMLError:
        frontmatter = {}

    content = match.group(2).strip()

    return {
        "name": frontmatter.get("name", path.stem),
        "description": frontmatter.get("description", ""),
        "type": frontmatter.get("type", "feedback"),
        "content": content,
        "file_path": str(path),
    }


def discover_memory_files(base_dir: str | Path | None = None) -> list[Path]:
    """
    Find all Claude Code memory markdown files.

    Searches:
    - ~/.claude/projects/*/memory/*.md
    - Specific path if provided
    """
    files: list[Path] = []

    if base_dir:
        base = Path(base_dir)
        if base.is_file():
            files.append(base)
        elif base.is_dir():
            files.extend(sorted(base.glob("*.md")))
        return files

    # Default: search all Claude Code memory directories
    claude_dir = Path.home() / ".claude" / "projects"
    if claude_dir.exists():
        for project_dir in claude_dir.iterdir():
            memory_dir = project_dir / "memory"
            if memory_dir.is_dir():
                for md_file in sorted(memory_dir.glob("*.md")):
                    if md_file.name != "MEMORY.md":  # Skip index file
                        files.append(md_file)

    return files


def migrate_memories(
    trainer: Trainer,
    source_dir: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    Import existing Claude Code markdown memories into SkillMind.

    Args:
        trainer: Trainer instance (handles dedup and classification)
        source_dir: Directory to import from (default: auto-discover)
        dry_run: If True, parse and report but don't store

    Returns:
        Migration stats dict
    """
    files = discover_memory_files(source_dir)

    stats = {
        "files_found": len(files),
        "imported": 0,
        "skipped_duplicate": 0,
        "skipped_error": 0,
        "memories": [],
    }

    for path in files:
        parsed = parse_memory_file(path)
        if not parsed:
            stats["skipped_error"] += 1
            continue

        # Map type
        mem_type = TYPE_MAP.get(parsed["type"], MemoryType.FEEDBACK)

        # Extract topic from filename
        topic = _extract_topic_from_name(parsed["name"])

        if dry_run:
            stats["memories"].append({
                "file": str(path),
                "name": parsed["name"],
                "type": mem_type.value,
                "topic": topic,
                "content_preview": parsed["content"][:100] + "...",
            })
            stats["imported"] += 1
            continue

        # Import via trainer (handles dedup)
        memory = trainer.learn(
            content=parsed["content"],
            title=parsed.get("description") or parsed["name"],
            source=MemorySource.IMPORT,
            force_type=mem_type,
            force_topic=topic,
            tags=_extract_tags_from_name(parsed["name"]),
            metadata={
                "original_file": str(path),
                "original_name": parsed["name"],
                "migrated_at": datetime.utcnow().isoformat(),
            },
        )

        if memory:
            stats["imported"] += 1
            stats["memories"].append({
                "id": memory.id,
                "title": memory.title,
                "type": memory.type.value,
                "topic": memory.topic,
            })
        else:
            stats["skipped_duplicate"] += 1

    return stats


def migrate_store(
    source: MemoryStore,
    target: MemoryStore,
    *,
    batch_size: int = 100,
    include_expired: bool = True,
    dry_run: bool = False,
    max_memories: int = 10000,
    progress_cb: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Copy all memories from one store backend to another.

    Memories keep their original IDs, so the migration is idempotent (re-running
    upserts the same records). The target store re-embeds each memory with its own
    engine, which means source and target MUST use the same embedding model /
    dimension — pass a shared EmbeddingEngine when constructing both stores.

    Note on pagination: some backends (Pinecone) cannot truly page through results,
    so we fetch up to ``max_memories`` in a single ``list_all`` call rather than
    iterating ``offset``. If the source holds more than ``max_memories``, the result
    is flagged as ``truncated`` so the caller can warn instead of silently dropping.

    Args:
        source: store to read from (already initialized)
        target: store to write to (already initialized)
        batch_size: how many memories to write per ``add_batch`` call
        include_expired: also migrate expired memories (default True — a migration
            should not silently drop data)
        dry_run: count and fetch, but do not write to the target
        max_memories: hard cap on how many memories to fetch from the source
        progress_cb: optional callback(done, total) for progress reporting

    Returns:
        Stats dict: source_count, fetched, migrated, batches, truncated.
    """
    qf = QueryFilter(include_expired=include_expired)
    source_count = source.count(filter=qf)
    memories = source.list_all(filter=qf, limit=max_memories, offset=0)

    stats: dict[str, Any] = {
        "source_count": source_count,
        "fetched": len(memories),
        "migrated": 0,
        "batches": 0,
        "truncated": source_count > len(memories),
    }

    if dry_run or not memories:
        return stats

    for i in range(0, len(memories), batch_size):
        batch = memories[i : i + batch_size]
        target.add_batch(batch)
        stats["migrated"] += len(batch)
        stats["batches"] += 1
        if progress_cb:
            progress_cb(stats["migrated"], len(memories))

    return stats


def _extract_topic_from_name(name: str) -> str:
    """Extract topic from memory filename like 'feedback_pdf_quality'."""
    # Remove type prefix
    for prefix in ("user_", "feedback_", "project_", "reference_", "skill_"):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    # Convert underscores to readable topic
    return name.replace("_", " ").strip() or "general"


def _extract_tags_from_name(name: str) -> list[str]:
    """Extract tags from memory filename."""
    parts = name.replace("-", "_").split("_")
    # Filter out type prefixes and short words
    tags = [p for p in parts if len(p) > 2 and p not in ("user", "feedback", "project", "reference")]
    return tags[:5]
