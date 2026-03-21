"""
SkillMind Setup Script — one-command setup that:

1. Discovers ALL existing memory/knowledge sources:
   - Claude Code markdown memories (~/.claude/projects/*/memory/*.md)
   - CLAUDE.md files (project instructions)
   - Markdown docs in project directories
   - Existing Skill Seekers outputs
2. Parses and classifies each source
3. Imports into the vector store (deduplicating along the way)
4. Generates initial context rules
5. Reports what was imported

Usage:
    python -m skillmind.setup
    python -m skillmind.setup --backend chroma --scan-dir /path/to/project
    python -m skillmind.setup --dry-run
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from .config import SkillMindConfig, StoreConfig
from .embeddings import EmbeddingEngine
from .migration import discover_memory_files, parse_memory_file
from .models import ContextRule, Memory, MemorySource, MemoryType
from .store import create_store
from .trainer import Trainer


def discover_all_sources(
    scan_dirs: list[str] | None = None,
) -> dict[str, list[Path]]:
    """
    Discover ALL knowledge sources on the system.

    Returns a dict of source_type -> list of file paths.
    """
    sources: dict[str, list[Path]] = {
        "claude_memories": [],
        "claude_md": [],
        "markdown_docs": [],
        "skill_seekers": [],
    }

    home = Path.home()

    # 1. Claude Code memory files
    claude_dir = home / ".claude" / "projects"
    if claude_dir.exists():
        for project_dir in claude_dir.iterdir():
            memory_dir = project_dir / "memory"
            if memory_dir.is_dir():
                for md in sorted(memory_dir.glob("*.md")):
                    if md.name != "MEMORY.md":
                        sources["claude_memories"].append(md)

    # 2. CLAUDE.md files (project instructions)
    search_roots = [home / ".claude"]
    if scan_dirs:
        search_roots.extend(Path(d) for d in scan_dirs)

    for root in search_roots:
        if root.exists():
            for claude_md in root.rglob("CLAUDE.md"):
                # Skip node_modules, .git, etc.
                parts = claude_md.parts
                if any(p.startswith(".") or p == "node_modules" for p in parts):
                    continue
                sources["claude_md"].append(claude_md)

    # 3. Markdown docs in common project locations
    project_dirs = [
        home / "PycharmProjects",
        home / "projects",
        home / "Documents",
        home / "OneDrive",
    ]
    if scan_dirs:
        project_dirs.extend(Path(d) for d in scan_dirs)

    for proj_root in project_dirs:
        if proj_root.exists():
            # Only go 3 levels deep to avoid scanning everything
            for md in proj_root.glob("**/README.md"):
                depth = len(md.relative_to(proj_root).parts)
                if depth <= 3:
                    sources["markdown_docs"].append(md)

    # 4. Skill Seekers output directories
    skill_seekers_dirs = [
        home / "output",
        home / ".skill-seekers",
    ]
    for ss_dir in skill_seekers_dirs:
        if ss_dir.exists():
            for skill_md in ss_dir.rglob("SKILL.md"):
                sources["skill_seekers"].append(skill_md)

    return sources


def import_claude_memories(
    trainer: Trainer,
    files: list[Path],
    dry_run: bool = False,
) -> list[dict]:
    """Import Claude Code markdown memory files."""
    results = []
    for path in files:
        parsed = parse_memory_file(path)
        if not parsed:
            results.append({"file": str(path), "status": "error", "reason": "parse_failed"})
            continue

        type_map = {
            "user": MemoryType.USER,
            "feedback": MemoryType.FEEDBACK,
            "project": MemoryType.PROJECT,
            "reference": MemoryType.REFERENCE,
            "skill": MemoryType.SKILL,
        }
        mem_type = type_map.get(parsed.get("type", ""), MemoryType.FEEDBACK)

        if dry_run:
            results.append({
                "file": str(path),
                "status": "would_import",
                "type": mem_type.value,
                "name": parsed["name"],
                "preview": parsed["content"][:80],
            })
            continue

        # Extract topic from filename
        name = parsed["name"]
        for prefix in ("user_", "feedback_", "project_", "reference_", "skill_"):
            if name.startswith(prefix):
                name = name[len(prefix):]
                break
        topic = name.replace("_", " ").strip() or "general"

        mem = trainer.learn(
            content=parsed["content"],
            title=parsed.get("description") or parsed["name"],
            source=MemorySource.IMPORT,
            force_type=mem_type,
            force_topic=topic,
            tags=name.split("_")[:5],
            metadata={"original_file": str(path), "migrated_at": datetime.utcnow().isoformat()},
        )

        results.append({
            "file": str(path),
            "status": "imported" if mem else "duplicate",
            "type": mem_type.value,
            "id": mem.id if mem else None,
        })

    return results


def import_claude_md(
    trainer: Trainer,
    files: list[Path],
    dry_run: bool = False,
) -> list[dict]:
    """Import CLAUDE.md project instruction files as reference memories."""
    results = []
    for path in files:
        try:
            content = path.read_text(encoding="utf-8")
        except Exception:
            results.append({"file": str(path), "status": "error"})
            continue

        if len(content) < 50:
            continue

        # Extract project name from path
        project_name = path.parent.name
        if project_name in (".", ".claude"):
            project_name = "global"

        if dry_run:
            results.append({
                "file": str(path),
                "status": "would_import",
                "type": "reference",
                "project": project_name,
                "size": len(content),
            })
            continue

        mem = trainer.learn(
            content=content[:5000],  # Cap at 5k chars
            title=f"CLAUDE.md: {project_name}",
            source=MemorySource.IMPORT,
            force_type=MemoryType.REFERENCE,
            force_topic=project_name,
            tags=["claude_md", "project_instructions", project_name],
            metadata={"original_file": str(path)},
        )

        results.append({
            "file": str(path),
            "status": "imported" if mem else "duplicate",
            "id": mem.id if mem else None,
        })

    return results


def import_skill_seekers(
    trainer: Trainer,
    files: list[Path],
    dry_run: bool = False,
) -> list[dict]:
    """Import Skill Seekers SKILL.md files as skill memories."""
    results = []
    for path in files:
        try:
            content = path.read_text(encoding="utf-8")
        except Exception:
            results.append({"file": str(path), "status": "error"})
            continue

        skill_name = path.parent.name

        if dry_run:
            results.append({
                "file": str(path),
                "status": "would_import",
                "type": "skill",
                "skill": skill_name,
                "size": len(content),
            })
            continue

        mem = trainer.learn(
            content=content[:8000],
            title=f"Skill: {skill_name}",
            source=MemorySource.SKILL_SEEKERS,
            force_type=MemoryType.SKILL,
            force_topic=skill_name,
            tags=["skill_seekers", "skill", skill_name],
            metadata={"original_file": str(path), "source": "skill_seekers"},
        )

        results.append({
            "file": str(path),
            "status": "imported" if mem else "duplicate",
            "id": mem.id if mem else None,
        })

    return results


def run_setup(
    backend: str = "chroma",
    data_dir: str = ".skillmind",
    scan_dirs: list[str] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    Full setup: discover, import, structure all existing knowledge.

    Returns comprehensive stats dict.
    """
    # 1. Initialize config and store
    config = SkillMindConfig(
        data_dir=data_dir,
        store=StoreConfig(backend=backend, chroma_path=f"{data_dir}/chroma", faiss_path=f"{data_dir}/faiss"),
    )
    if not dry_run:
        config.save()

    engine = EmbeddingEngine(config.embedding)
    store = create_store(config, engine)
    store.initialize()
    trainer = Trainer(store)

    # 2. Discover all sources
    print("Discovering knowledge sources...")
    sources = discover_all_sources(scan_dirs)

    stats: dict[str, Any] = {
        "discovered": {k: len(v) for k, v in sources.items()},
        "total_discovered": sum(len(v) for v in sources.values()),
        "results": {},
    }

    print(f"  Claude memories:     {len(sources['claude_memories'])}")
    print(f"  CLAUDE.md files:     {len(sources['claude_md'])}")
    print(f"  Markdown docs:       {len(sources['markdown_docs'])}")
    print(f"  Skill Seekers:       {len(sources['skill_seekers'])}")
    print(f"  Total:               {stats['total_discovered']}")
    print()

    # 3. Import each source type
    if sources["claude_memories"]:
        print(f"{'[DRY RUN] ' if dry_run else ''}Importing Claude memories...")
        results = import_claude_memories(trainer, sources["claude_memories"], dry_run)
        imported = sum(1 for r in results if r["status"] in ("imported", "would_import"))
        dupes = sum(1 for r in results if r["status"] == "duplicate")
        print(f"  -> {imported} imported, {dupes} duplicates skipped")
        stats["results"]["claude_memories"] = results

    if sources["claude_md"]:
        print(f"{'[DRY RUN] ' if dry_run else ''}Importing CLAUDE.md files...")
        results = import_claude_md(trainer, sources["claude_md"], dry_run)
        imported = sum(1 for r in results if r["status"] in ("imported", "would_import"))
        print(f"  -> {imported} imported")
        stats["results"]["claude_md"] = results

    if sources["skill_seekers"]:
        print(f"{'[DRY RUN] ' if dry_run else ''}Importing Skill Seekers skills...")
        results = import_skill_seekers(trainer, sources["skill_seekers"], dry_run)
        imported = sum(1 for r in results if r["status"] in ("imported", "would_import"))
        print(f"  -> {imported} imported")
        stats["results"]["skill_seekers"] = results

    # 4. Summary
    total_imported = sum(
        sum(1 for r in results if r["status"] in ("imported", "would_import"))
        for results in stats["results"].values()
    )
    stats["total_imported"] = total_imported

    if not dry_run:
        total_in_store = store.count()
        stats["total_in_store"] = total_in_store
        print(f"\nTotal memories in store: {total_in_store}")
    else:
        print(f"\n[DRY RUN] Would import {total_imported} items. Run without --dry-run to execute.")

    return stats


def main():
    """CLI entry point for setup."""
    import argparse

    parser = argparse.ArgumentParser(description="SkillMind Setup — import all existing knowledge")
    parser.add_argument("--backend", "-b", default="chroma", choices=["chroma", "pinecone", "supabase", "qdrant", "faiss"])
    parser.add_argument("--data-dir", "-d", default=".skillmind")
    parser.add_argument("--scan-dir", "-s", action="append", dest="scan_dirs", help="Additional directories to scan")
    parser.add_argument("--dry-run", action="store_true", help="Preview without importing")
    args = parser.parse_args()

    print("=" * 60)
    print("SkillMind Setup")
    print("=" * 60)
    print()

    stats = run_setup(
        backend=args.backend,
        data_dir=args.data_dir,
        scan_dirs=args.scan_dirs,
        dry_run=args.dry_run,
    )

    # Save report
    report_path = Path(args.data_dir) / "setup_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main()
