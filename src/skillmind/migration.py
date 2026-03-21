"""
Migration script: import existing Claude Code markdown memories into SkillMind.

Reads the flat .claude/projects/*/memory/*.md files and converts them
into structured Memory objects in the vector store.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from .models import Memory, MemoryType, MemorySource
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
