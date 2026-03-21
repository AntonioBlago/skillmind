"""
SkillMind Context Generator — builds dynamic, focused context
for Claude Code conversations based on current activity.

Replaces static MEMORY.md loading with intelligent retrieval.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

from .models import ContextRule, Memory, MemoryType, QueryFilter
from .store.base import MemoryStore


class ContextGenerator:
    """
    Generates focused context for the current conversation/file/topic.

    Instead of loading ALL memories into Claude Code's context,
    this generates a small, relevant context document.
    """

    def __init__(self, store: MemoryStore, max_tokens: int = 4000):
        self.store = store
        self.max_tokens = max_tokens
        self.rules: list[ContextRule] = []

    def add_rule(self, rule: ContextRule) -> None:
        self.rules.append(rule)

    def generate(
        self,
        current_file: str | None = None,
        current_topic: str | None = None,
        query: str | None = None,
    ) -> str:
        """
        Build context document for current situation.

        Args:
            current_file: Currently open file path
            current_topic: Topic of current conversation
            query: Direct semantic query

        Returns:
            Markdown-formatted context string (< max_tokens)
        """
        relevant_memories: list[tuple[Memory, float]] = []

        # 1. Apply context rules (deterministic)
        rule_memories = self._apply_rules(current_file, current_topic)
        for mem in rule_memories:
            relevant_memories.append((mem, 1.0))

        # 2. Semantic search based on query or file context
        search_text = self._build_search_text(current_file, current_topic, query)
        if search_text:
            results = self.store.query(search_text, limit=10)
            for r in results:
                # Avoid duplicates
                if not any(m.id == r.memory.id for m, _ in relevant_memories):
                    relevant_memories.append((r.memory, r.score))

        # 3. Always include user profile memories (high priority)
        user_memories = self.store.list_all(
            filter=QueryFilter(types=[MemoryType.USER]),
            limit=3,
        )
        for mem in user_memories:
            if not any(m.id == mem.id for m, _ in relevant_memories):
                relevant_memories.insert(0, (mem, 1.0))

        # 4. Always include high-confidence feedback
        feedback_memories = self.store.list_all(
            filter=QueryFilter(types=[MemoryType.FEEDBACK], min_confidence=0.9),
            limit=5,
        )
        for mem in feedback_memories:
            if not any(m.id == mem.id for m, _ in relevant_memories):
                relevant_memories.append((mem, 0.95))

        # 5. Sort by relevance and truncate to fit token budget
        relevant_memories.sort(key=lambda x: x[1], reverse=True)
        return self._format_context(relevant_memories)

    def generate_to_file(
        self,
        output_path: str | Path,
        current_file: str | None = None,
        current_topic: str | None = None,
        query: str | None = None,
    ) -> Path:
        """Generate context and write to file (for Claude Code injection)."""
        context = self.generate(current_file, current_topic, query)
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(context, encoding="utf-8")
        return path

    def _apply_rules(
        self, current_file: str | None, current_topic: str | None
    ) -> list[Memory]:
        """Apply context rules to determine what to load."""
        memories: list[Memory] = []

        for rule in sorted(self.rules, key=lambda r: r.priority, reverse=True):
            triggered = False

            # Check file pattern triggers
            if current_file and rule.trigger.startswith("file:"):
                pattern = rule.trigger[5:]
                if self._match_glob(current_file, pattern):
                    triggered = True

            # Check topic triggers
            if current_topic and rule.trigger.startswith("topic:"):
                topic = rule.trigger[6:]
                if topic.lower() in current_topic.lower():
                    triggered = True

            # Check project triggers
            if rule.trigger.startswith("project:"):
                project = rule.trigger[8:]
                if current_topic and project.lower() in current_topic.lower():
                    triggered = True
                if current_file and project.lower() in current_file.lower():
                    triggered = True

            if triggered:
                if rule.load_topics:
                    mems = self.store.list_all(
                        filter=QueryFilter(topics=rule.load_topics),
                        limit=10,
                    )
                    memories.extend(mems)
                if rule.load_types:
                    mems = self.store.list_all(
                        filter=QueryFilter(types=rule.load_types),
                        limit=10,
                    )
                    memories.extend(mems)

        return memories

    def _build_search_text(
        self,
        current_file: str | None,
        current_topic: str | None,
        query: str | None,
    ) -> str:
        """Build a search query from available context."""
        parts: list[str] = []
        if query:
            parts.append(query)
        if current_topic:
            parts.append(current_topic)
        if current_file:
            # Extract meaningful parts from file path
            p = Path(current_file)
            parts.append(p.stem)
            if p.suffix:
                parts.append(p.suffix.lstrip("."))
        return " ".join(parts)

    def _format_context(self, memories: list[tuple[Memory, float]]) -> str:
        """Format memories into a readable context document."""
        if not memories:
            return "# SkillMind Context\n\nNo relevant memories found for current context.\n"

        lines: list[str] = ["# SkillMind Context", ""]
        char_budget = self.max_tokens * 4  # Rough chars-to-tokens ratio
        chars_used = 0

        # Group by type
        by_type: dict[MemoryType, list[tuple[Memory, float]]] = {}
        for mem, score in memories:
            by_type.setdefault(mem.type, []).append((mem, score))

        type_order = [
            MemoryType.USER,
            MemoryType.FEEDBACK,
            MemoryType.PROJECT,
            MemoryType.REFERENCE,
            MemoryType.SKILL,
        ]

        for mem_type in type_order:
            entries = by_type.get(mem_type, [])
            if not entries:
                continue

            section_header = f"## {mem_type.value.title()} Memories\n"
            lines.append(section_header)
            chars_used += len(section_header)

            for mem, score in entries:
                entry = f"### {mem.title}\n{mem.content}\n"
                if chars_used + len(entry) > char_budget:
                    lines.append("_(truncated — more memories available via `skillmind recall`)_\n")
                    break
                lines.append(entry)
                chars_used += len(entry)

        lines.append(f"\n---\n_Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC | "
                      f"{len(memories)} memories loaded_\n")
        return "\n".join(lines)

    @staticmethod
    def _match_glob(filepath: str, pattern: str) -> bool:
        """Simple glob matching for file patterns."""
        import fnmatch
        return fnmatch.fnmatch(filepath, pattern) or fnmatch.fnmatch(
            Path(filepath).name, pattern
        )
