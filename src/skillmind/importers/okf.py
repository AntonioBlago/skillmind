"""
OKF (Open Knowledge Format) bundle importer.

Reads a directory of OKF concept files — markdown with YAML frontmatter, per
Google's knowledge-catalog ``okf/SPEC.md`` — and feeds each concept into the
SkillMind store through :class:`~skillmind.trainer.Trainer` (so classification,
sanitization and deduplication all apply on the way in).

Two cases are handled:

1. **SkillMind-produced bundles** (have ``skillmind_*`` frontmatter keys): the
   original type, topic, id and timestamps are honored for a lossless round-trip.
2. **Foreign bundles** (only the OKF-standard ``type`` / ``title`` / ``tags`` …):
   the freeform ``type`` is mapped to the nearest :class:`MemoryType` when
   possible, otherwise the Trainer classifies the concept from its content.

Files named ``index.md``, ``log.md`` and ``README.md`` are skipped — per the
spec these are the bundle entry point / change log, not concepts.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from ..models import MemorySource, MemoryType
from ..trainer import Trainer


# Reserved bundle file names that are not concepts.
_RESERVED_FILES = {"index.md", "log.md", "readme.md"}

# Folder names SkillMind itself uses — these encode a type, not a topic.
_RESERVED_FOLDERS = {"skills", "references", "feedback", "projects", "users", "general"}

# Map OKF / foreign `type` strings onto SkillMind memory types. Anything not
# listed here falls through to Trainer classification.
OKF_TYPE_MAP: dict[str, MemoryType] = {
    "user": MemoryType.USER,
    "profile": MemoryType.USER,
    "persona": MemoryType.USER,
    "feedback": MemoryType.FEEDBACK,
    "correction": MemoryType.FEEDBACK,
    "preference": MemoryType.FEEDBACK,
    "project": MemoryType.PROJECT,
    "task": MemoryType.PROJECT,
    "milestone": MemoryType.PROJECT,
    "reference": MemoryType.REFERENCE,
    "resource": MemoryType.REFERENCE,
    "link": MemoryType.REFERENCE,
    "glossary": MemoryType.REFERENCE,
    "table": MemoryType.REFERENCE,
    "dataset": MemoryType.REFERENCE,
    "skill": MemoryType.SKILL,
    "pattern": MemoryType.SKILL,
    "playbook": MemoryType.SKILL,
    "workflow": MemoryType.SKILL,
    "how-to": MemoryType.SKILL,
    "guide": MemoryType.SKILL,
}

# Headings that mark producer-generated sections we strip before storing.
_GENERATED_SECTION_RE = re.compile(
    r"\n#{1,3}\s+(Related|Citations|See also|References)\s*\n.*$",
    re.IGNORECASE | re.DOTALL,
)


def parse_concept_file(path: Path) -> dict[str, Any] | None:
    """
    Parse one OKF concept file into a normalized dict.

    Returns keys: frontmatter (dict), body (str), title (str), concept_id (str),
    or None if the file is unreadable.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None

    match = re.match(r"^---\s*\n(.+?)\n---\s*\n(.*)$", text, re.DOTALL)
    if match:
        try:
            frontmatter = yaml.safe_load(match.group(1)) or {}
        except yaml.YAMLError:
            frontmatter = {}
        body = match.group(2).strip()
    else:
        frontmatter = {}
        body = text.strip()

    if not isinstance(frontmatter, dict):
        frontmatter = {}

    # A leading "# Title" heading in the body duplicates the frontmatter title.
    title = frontmatter.get("title")
    if not title:
        heading = re.match(r"^#\s+(.+?)\s*$", body, re.MULTILINE)
        title = heading.group(1).strip() if heading else path.stem

    return {
        "frontmatter": frontmatter,
        "body": body,
        "title": str(title),
        "concept_id": _concept_id(path),
    }


def discover_concept_files(bundle_path: str | Path) -> list[Path]:
    """Find all OKF concept files in a bundle (skips index/log/readme)."""
    base = Path(bundle_path)
    if base.is_file():
        return [base]
    files: list[Path] = []
    for md in sorted(base.rglob("*.md")):
        if md.name.lower() in _RESERVED_FILES:
            continue
        files.append(md)
    return files


def import_okf_bundle(
    trainer: Trainer,
    bundle_path: str | Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    Import an OKF bundle into the store via the Trainer.

    Args:
        trainer: Trainer instance (handles classification + dedup)
        bundle_path: path to the bundle directory (or a single concept file)
        dry_run: parse and report, but do not store

    Returns:
        Stats dict: files_found, imported, skipped_duplicate, skipped_error, concepts.
    """
    base = Path(bundle_path)
    files = discover_concept_files(base)

    stats: dict[str, Any] = {
        "bundle": str(base),
        "files_found": len(files),
        "imported": 0,
        "skipped_duplicate": 0,
        "skipped_error": 0,
        "concepts": [],
    }

    for path in files:
        parsed = parse_concept_file(path)
        if not parsed:
            stats["skipped_error"] += 1
            continue

        fm = parsed["frontmatter"]
        body = _strip_generated_sections(parsed["body"])
        body = _strip_leading_title(body, parsed["title"])

        if not body.strip():
            stats["skipped_error"] += 1
            continue

        mem_type = _resolve_type(fm)
        topic = _resolve_topic(fm, path, base)
        tags = _resolve_tags(fm)
        metadata = _build_metadata(fm, parsed["concept_id"], path)

        if dry_run:
            stats["concepts"].append({
                "file": str(path),
                "concept_id": parsed["concept_id"],
                "title": parsed["title"],
                "type": mem_type.value if mem_type else "auto",
                "topic": topic or "auto",
            })
            stats["imported"] += 1
            continue

        memory = trainer.learn(
            content=body,
            title=parsed["title"],
            source=MemorySource.IMPORT,
            force_type=mem_type,
            force_topic=topic,
            tags=tags or None,
            metadata=metadata,
        )

        if memory:
            stats["imported"] += 1
            stats["concepts"].append({
                "id": memory.id,
                "concept_id": parsed["concept_id"],
                "title": memory.title,
                "type": memory.type.value,
                "topic": memory.topic,
            })
        else:
            stats["skipped_duplicate"] += 1

    return stats


class OKFImporter:
    """Object-oriented wrapper around :func:`import_okf_bundle`."""

    def __init__(self, trainer: Trainer):
        self.trainer = trainer

    def import_bundle(self, bundle_path: str | Path, dry_run: bool = False) -> dict[str, Any]:
        return import_okf_bundle(self.trainer, bundle_path, dry_run=dry_run)


# ── Internal helpers ──────────────────────────────────────────────


def _concept_id(path: Path) -> str:
    """Best-effort concept ID = bundle-relative path minus .md (slash-joined)."""
    # Use the closest parent folder + stem; full bundle-relative path is not
    # always recoverable from a single Path, so keep the last 2 components.
    parts = path.with_suffix("").parts
    return "/".join(parts[-2:]) if len(parts) >= 2 else path.stem


def _resolve_type(fm: dict[str, Any]) -> MemoryType | None:
    """SkillMind type wins; else map the OKF `type`; else let the Trainer decide."""
    sm_type = fm.get("skillmind_type") or fm.get("skillmind-type")
    if isinstance(sm_type, str):
        try:
            return MemoryType(sm_type.strip().lower())
        except ValueError:
            pass

    okf_type = fm.get("type")
    if isinstance(okf_type, str):
        key = okf_type.strip().lower()
        try:
            return MemoryType(key)
        except ValueError:
            pass
        return OKF_TYPE_MAP.get(key)

    return None


def _resolve_topic(fm: dict[str, Any], path: Path, base: Path) -> str | None:
    sm_topic = fm.get("skillmind_topic") or fm.get("skillmind-topic")
    if isinstance(sm_topic, str) and sm_topic.strip():
        return sm_topic.strip()

    # Use the containing folder as a topic hint, unless it's a reserved
    # SkillMind type-folder (those encode a type, not a subject).
    folder = path.parent.name.lower()
    if folder and folder not in _RESERVED_FOLDERS and path.parent != base:
        return folder.replace("-", " ").replace("_", " ").strip()

    return None  # Trainer will classify the topic from content.


def _resolve_tags(fm: dict[str, Any]) -> list[str]:
    tags = fm.get("tags")
    if isinstance(tags, list):
        return [str(t).strip() for t in tags if str(t).strip()][:10]
    if isinstance(tags, str) and tags.strip():
        return [t.strip() for t in re.split(r"[,;]", tags) if t.strip()][:10]
    return []


def _build_metadata(fm: dict[str, Any], concept_id: str, path: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "okf_concept_id": concept_id,
        "original_file": str(path),
        "imported_at": datetime.utcnow().isoformat(),
    }

    okf_type = fm.get("type")
    if isinstance(okf_type, str):
        metadata["okf_type"] = okf_type

    sm_id = fm.get("skillmind_id") or fm.get("skillmind-id")
    if isinstance(sm_id, str):
        metadata["okf_source_id"] = sm_id

    resource = fm.get("resource")
    if isinstance(resource, str) and resource.strip():
        metadata["source_url"] = resource.strip()

    desc = fm.get("description")
    if isinstance(desc, str) and desc.strip():
        metadata["description"] = desc.strip()

    return metadata


def _strip_generated_sections(body: str) -> str:
    """Remove producer-generated '# Related' / '# Citations' trailers."""
    return _GENERATED_SECTION_RE.sub("", body).rstrip()


def _strip_leading_title(body: str, title: str) -> str:
    """Drop a leading '# Title' heading so it isn't duplicated in the content."""
    stripped = body.lstrip()
    heading = f"# {title}"
    if stripped.startswith(heading):
        return stripped[len(heading):].lstrip("\n")
    # Generic: remove any single leading H1.
    return re.sub(r"^#\s+.+?\n+", "", stripped, count=1)
