"""
SkillMind Listener — watches git events, file changes, and conversations
to automatically extract and store memories.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from .models import Memory, MemoryType, MemorySource
from .trainer import Trainer


class GitListener:
    """
    Listens to git events and extracts relevant memories.

    Can be triggered via:
    - Git hooks (post-commit, post-merge)
    - Claude Code hooks (settings.json)
    - Manual CLI invocation
    """

    def __init__(self, trainer: Trainer, repo_path: str = "."):
        self.trainer = trainer
        self.repo_path = repo_path

    def on_commit(self, commit_hash: str | None = None) -> list[Memory]:
        """Extract learnings from a git commit."""
        memories: list[Memory] = []

        if commit_hash is None:
            commit_hash = self._run_git("rev-parse", "HEAD").strip()

        # Get commit details
        msg = self._run_git("log", "-1", "--format=%B", commit_hash).strip()
        diff_stat = self._run_git("diff", "--stat", f"{commit_hash}~1..{commit_hash}")
        files_changed = self._run_git("diff", "--name-only", f"{commit_hash}~1..{commit_hash}")

        # Extract: new dependencies added?
        if any(f in files_changed for f in ["package.json", "requirements.txt", "pyproject.toml", "Cargo.toml"]):
            mem = self.trainer.learn(
                content=f"Dependency change detected in commit {commit_hash[:8]}: {msg}\nFiles: {diff_stat}",
                title=f"Dependency update: {msg[:60]}",
                source=MemorySource.GIT_COMMIT,
                force_type=MemoryType.PROJECT,
                tags=["dependencies", "git"],
            )
            if mem:
                memories.append(mem)

        # Extract: config changes?
        config_files = [f for f in files_changed.split("\n") if any(
            f.endswith(ext) for ext in [".yml", ".yaml", ".json", ".toml", ".env.example"]
        )]
        if config_files:
            mem = self.trainer.learn(
                content=f"Config change in commit {commit_hash[:8]}: {', '.join(config_files)}\nMessage: {msg}",
                title=f"Config update: {msg[:60]}",
                source=MemorySource.GIT_COMMIT,
                force_type=MemoryType.PROJECT,
                tags=["config", "git"],
            )
            if mem:
                memories.append(mem)

        return memories

    def on_merge(self, branch: str | None = None) -> list[Memory]:
        """Extract learnings from a branch merge."""
        if branch is None:
            branch = self._run_git("rev-parse", "--abbrev-ref", "HEAD").strip()

        memories: list[Memory] = []

        mem = self.trainer.learn(
            content=f"Branch merged into {branch} at {datetime.utcnow().isoformat()}",
            title=f"Merge event: {branch}",
            source=MemorySource.GIT_COMMIT,
            force_type=MemoryType.PROJECT,
            tags=["git", "merge", branch],
        )
        if mem:
            memories.append(mem)

        return memories

    def _run_git(self, *args: str) -> str:
        try:
            result = subprocess.run(
                ["git", *args],
                capture_output=True,
                text=True,
                cwd=self.repo_path,
                timeout=10,
            )
            return result.stdout
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return ""


class FileListener:
    """
    Watches file changes and extracts relevant memories.

    Triggered via Claude Code hooks or file system watchers.
    """

    def __init__(self, trainer: Trainer):
        self.trainer = trainer

    def on_file_change(self, path: str, event: str = "modified") -> Memory | None:
        """React to a file change event."""
        p = Path(path)

        # Skip non-interesting files
        if p.suffix in (".pyc", ".pyo", ".class", ".o"):
            return None
        if any(part.startswith(".") for part in p.parts[1:]) and p.name != ".env.example":
            return None

        # CLAUDE.md changes → sync instructions
        if p.name == "CLAUDE.md":
            return self.trainer.learn(
                content=f"CLAUDE.md was {event} at {path}. Project instructions may have changed.",
                title=f"CLAUDE.md {event}",
                source=MemorySource.FILE_CHANGE,
                force_type=MemoryType.PROJECT,
                tags=["claude_md", "project_config"],
            )

        # New client folder detection
        if event == "created" and p.is_dir() and "clients" in str(p).lower():
            return self.trainer.learn(
                content=f"New client folder created: {path}",
                title=f"New client: {p.name}",
                source=MemorySource.FILE_CHANGE,
                force_type=MemoryType.PROJECT,
                tags=["client", "new_project"],
            )

        return None


class ConversationListener:
    """
    Extracts memories from Claude Code conversations.

    Parses conversation transcripts to find:
    - User corrections (feedback)
    - User preferences (user memories)
    - Project context (project memories)
    - External references (reference memories)
    """

    def __init__(self, trainer: Trainer):
        self.trainer = trainer

    def extract_from_messages(self, messages: list[dict[str, str]]) -> list[Memory]:
        """
        Analyze conversation messages and extract learnable content.

        Args:
            messages: List of {"role": "user"|"assistant", "content": "..."}

        Returns:
            List of newly created/updated memories.
        """
        memories: list[Memory] = []

        for msg in messages:
            if msg.get("role") != "user":
                continue

            content = msg.get("content", "")
            if not content or len(content) < 20:
                continue

            # Detect corrections / feedback
            if self._is_correction(content):
                mem = self.trainer.learn(
                    content=content,
                    source=MemorySource.CONVERSATION,
                    force_type=MemoryType.FEEDBACK,
                )
                if mem:
                    memories.append(mem)

            # Detect external references
            elif self._has_reference(content):
                mem = self.trainer.learn(
                    content=content,
                    source=MemorySource.CONVERSATION,
                    force_type=MemoryType.REFERENCE,
                )
                if mem:
                    memories.append(mem)

            # Detect project context (deadlines, status updates)
            elif self._is_project_context(content):
                mem = self.trainer.learn(
                    content=content,
                    source=MemorySource.CONVERSATION,
                    force_type=MemoryType.PROJECT,
                )
                if mem:
                    memories.append(mem)

        return memories

    def _is_correction(self, text: str) -> bool:
        """Detect if user is correcting assistant behavior."""
        patterns = [
            r"\bdon'?t\b.*\b(do|use|add|make)\b",
            r"\bstop\b.*\b(doing|adding|using)\b",
            r"\bnever\b.*\b(do|use|add)\b",
            r"\balways\b.*\b(use|do|make|prefer)\b",
            r"\bnicht\b.*\b(machen|verwenden|nutzen)\b",
            r"\bimmer\b.*\b(verwenden|machen|nutzen)\b",
            r"\bno,?\s*(not|that'?s wrong|incorrect)\b",
            r"\bplease\b.*\b(remember|note|keep in mind)\b",
        ]
        text_lower = text.lower()
        return any(re.search(p, text_lower) for p in patterns)

    def _has_reference(self, text: str) -> bool:
        """Detect external references in text."""
        patterns = [
            r"https?://\S+",
            r"\b(dashboard|wiki|confluence|linear|jira|slack)\b.*\b(at|in|on|under)\b",
            r"\b(tracked in|documented at|found in|check)\b",
        ]
        return any(re.search(p, text.lower()) for p in patterns)

    def _is_project_context(self, text: str) -> bool:
        """Detect project context information."""
        patterns = [
            r"\b(deadline|due|by|until)\b.*\b(monday|tuesday|wednesday|thursday|friday|tomorrow|next week)\b",
            r"\b(release|deploy|launch|freeze)\b.*\b(on|at|by|before)\b",
            r"\b(sprint|milestone|phase)\b.*\b(\d+|one|two|three)\b",
            r"\b(budget|hours|days|weeks)\b.*\b(left|remaining|allocated)\b",
            r"\b(termin|frist|bis|deadline)\b",
        ]
        return any(re.search(p, text.lower()) for p in patterns)
