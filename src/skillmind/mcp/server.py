"""
SkillMind MCP Server — exposes memory CRUD + context tools to Claude Code.

Usage in Claude Code settings.json:
    {
        "mcpServers": {
            "skillmind": {
                "command": "python",
                "args": ["-m", "skillmind.mcp.server"],
                "env": {
                    "SKILLMIND_BACKEND": "chroma"
                }
            }
        }
    }
"""

from __future__ import annotations

import json
import sys
from typing import Any

from ..config import SkillMindConfig
from ..context import ContextGenerator
from ..embeddings import EmbeddingEngine
from ..migration import migrate_memories
from ..models import Memory, MemorySource, MemoryType, QueryFilter
from ..store import create_store
from ..trainer import Trainer


def create_server():
    """Create and configure the MCP server."""
    from fastmcp import FastMCP

    mcp = FastMCP("skillmind")

    # Initialize components
    config = SkillMindConfig.load().resolve_env()
    engine = EmbeddingEngine(config.embedding)
    store = create_store(config, engine)
    store.initialize()
    trainer = Trainer(store)
    context_gen = ContextGenerator(store, max_tokens=config.context_max_tokens)

    # ─── Memory CRUD Tools ────────────────────────────────────────

    @mcp.tool()
    def remember(
        content: str,
        title: str = "",
        type: str = "feedback",
        topic: str = "",
        tags: str = "",
    ) -> str:
        """
        Store a new memory. The trainer auto-classifies, deduplicates, and indexes it.

        Args:
            content: The knowledge to remember
            title: Short title (auto-generated if empty)
            type: Memory type: user, feedback, project, reference, skill
            topic: Primary topic tag (auto-detected if empty)
            tags: Comma-separated tags
        """
        tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else None
        force_type = MemoryType(type) if type else None
        force_topic = topic if topic else None

        memory = trainer.learn(
            content=content,
            title=title or None,
            source=MemorySource.MANUAL,
            force_type=force_type,
            force_topic=force_topic,
            tags=tag_list,
        )

        if memory:
            return json.dumps({
                "status": "stored",
                "id": memory.id,
                "type": memory.type.value,
                "topic": memory.topic,
                "title": memory.title,
            }, indent=2)
        return json.dumps({"status": "duplicate", "message": "Similar memory already exists"})

    @mcp.tool()
    def recall(
        query: str,
        limit: int = 5,
        type: str = "",
        topic: str = "",
        min_confidence: float = 0.0,
    ) -> str:
        """
        Semantic search across all memories. Returns the most relevant matches.

        Args:
            query: Natural language search query
            limit: Max number of results (default 5)
            type: Filter by type (user/feedback/project/reference/skill)
            topic: Filter by topic
            min_confidence: Minimum confidence threshold (0-1)
        """
        qf = QueryFilter(min_confidence=min_confidence)
        if type:
            qf.types = [MemoryType(type)]
        if topic:
            qf.topics = [topic]

        results = store.query(query, limit=limit, filter=qf)

        return json.dumps({
            "count": len(results),
            "results": [
                {
                    "id": r.memory.id,
                    "type": r.memory.type.value,
                    "topic": r.memory.topic,
                    "title": r.memory.title,
                    "content": r.memory.content,
                    "score": round(r.score, 3),
                    "confidence": r.memory.confidence,
                    "tags": r.memory.tags,
                }
                for r in results
            ],
        }, indent=2)

    @mcp.tool()
    def forget(memory_id: str) -> str:
        """
        Delete a specific memory by ID.

        Args:
            memory_id: The ID of the memory to delete
        """
        success = store.delete(memory_id)
        return json.dumps({
            "status": "deleted" if success else "not_found",
            "id": memory_id,
        })

    @mcp.tool()
    def update_memory(
        memory_id: str,
        content: str = "",
        title: str = "",
        topic: str = "",
        tags: str = "",
        confidence: float = -1,
    ) -> str:
        """
        Update an existing memory.

        Args:
            memory_id: ID of memory to update
            content: New content (empty = keep current)
            title: New title (empty = keep current)
            topic: New topic (empty = keep current)
            tags: New comma-separated tags (empty = keep current)
            confidence: New confidence 0-1 (negative = keep current)
        """
        memory = store.get(memory_id)
        if not memory:
            return json.dumps({"status": "not_found", "id": memory_id})

        if content:
            memory.content = content
        if title:
            memory.title = title
        if topic:
            memory.topic = topic
        if tags:
            memory.tags = [t.strip() for t in tags.split(",") if t.strip()]
        if confidence >= 0:
            memory.confidence = min(1.0, max(0.0, confidence))

        store.update(memory)
        return json.dumps({
            "status": "updated",
            "id": memory.id,
            "title": memory.title,
        })

    # ─── Context & Intelligence Tools ─────────────────────────────

    @mcp.tool()
    def context(
        file: str = "",
        topic: str = "",
        query: str = "",
    ) -> str:
        """
        Generate focused context for the current situation.
        Returns only relevant memories, not everything.

        Args:
            file: Currently open file path
            topic: Current conversation topic
            query: Direct semantic query
        """
        ctx = context_gen.generate(
            current_file=file or None,
            current_topic=topic or None,
            query=query or None,
        )
        return ctx

    @mcp.tool()
    def consolidate() -> str:
        """
        Run maintenance: merge duplicates, expire stale memories, rebalance confidence.
        Run this periodically (e.g., weekly) to keep the memory store clean.
        """
        stats = trainer.consolidate()
        return json.dumps({
            "status": "done",
            "merged": stats["merged"],
            "expired": stats["expired"],
            "updated": stats["updated"],
        }, indent=2)

    @mcp.tool()
    def memory_stats() -> str:
        """Show memory store statistics: counts by type, total, backend info."""
        total = store.count()
        by_type = {}
        for mt in MemoryType:
            by_type[mt.value] = store.count(filter=QueryFilter(types=[mt]))

        return json.dumps({
            "backend": config.store.backend,
            "total_memories": total,
            "by_type": by_type,
            "embedding_model": config.embedding.model,
        }, indent=2)

    @mcp.tool()
    def list_memories(
        type: str = "",
        topic: str = "",
        limit: int = 20,
    ) -> str:
        """
        List memories with optional filtering (no semantic search).

        Args:
            type: Filter by type (user/feedback/project/reference/skill)
            topic: Filter by topic
            limit: Max results (default 20)
        """
        qf = QueryFilter()
        if type:
            qf.types = [MemoryType(type)]
        if topic:
            qf.topics = [topic]

        memories = store.list_all(filter=qf, limit=limit)
        return json.dumps({
            "count": len(memories),
            "memories": [
                {
                    "id": m.id,
                    "type": m.type.value,
                    "topic": m.topic,
                    "title": m.title,
                    "content": m.content[:200] + ("..." if len(m.content) > 200 else ""),
                    "confidence": m.confidence,
                    "created_at": m.created_at.isoformat(),
                }
                for m in memories
            ],
        }, indent=2)

    @mcp.tool()
    def import_markdown_memories(source_dir: str = "", dry_run: bool = True) -> str:
        """
        Import existing Claude Code markdown memory files into SkillMind.

        Args:
            source_dir: Path to memory directory (default: auto-discover ~/.claude/)
            dry_run: If true, preview what would be imported without actually importing
        """
        stats = migrate_memories(
            trainer=trainer,
            source_dir=source_dir or None,
            dry_run=dry_run,
        )
        return json.dumps(stats, indent=2, default=str)

    # ─── Video & YouTube Learning Tools ───────────────────────────

    @mcp.tool()
    def learn_youtube(
        video_url: str,
        topic: str = "",
        tags: str = "",
    ) -> str:
        """
        Learn knowledge from a YouTube video. Extracts transcript,
        structures key insights, and stores as memories.

        Args:
            video_url: YouTube video URL or video ID
            topic: Override auto-detected topic
            tags: Comma-separated tags
        """
        from ..video.youtube_learner import YouTubeLearner

        yt = YouTubeLearner(trainer=trainer)
        tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else None
        memories = yt.learn(video_url, force_topic=topic or None, tags=tag_list)
        return json.dumps({
            "status": "learned",
            "memories_created": len(memories),
            "memories": [
                {"id": m.id, "type": m.type.value, "title": m.title, "topic": m.topic}
                for m in memories
            ],
        }, indent=2)

    @mcp.tool()
    def learn_youtube_channel(
        channel_id: str,
        max_videos: int = 5,
        topic: str = "",
    ) -> str:
        """
        Learn from latest videos of a YouTube channel.

        Args:
            channel_id: YouTube channel ID
            max_videos: Max videos to process (default 5)
            topic: Override topic for all videos
        """
        from ..video.youtube_learner import YouTubeLearner

        yt = YouTubeLearner(trainer=trainer)
        memories = yt.learn_channel(channel_id, max_videos, force_topic=topic or None)
        return json.dumps({
            "status": "learned",
            "memories_created": len(memories),
        }, indent=2)

    @mcp.tool()
    def learn_video(
        video_path: str,
        topic: str = "",
        tags: str = "",
        extract_audio: bool = False,
    ) -> str:
        """
        Learn from a local video file or screen recording.
        Extracts frames, runs OCR, optionally transcribes audio.

        Args:
            video_path: Path to video file (MP4, AVI, etc.)
            topic: Override auto-detected topic
            tags: Comma-separated tags
            extract_audio: Also transcribe audio via Whisper
        """
        from ..video.video_learner import VideoLearner

        vl = VideoLearner(trainer=trainer)
        tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else None
        memories = vl.learn(
            video_path, force_topic=topic or None,
            tags=tag_list, extract_audio=extract_audio,
        )
        return json.dumps({
            "status": "learned",
            "memories_created": len(memories),
            "memories": [
                {"id": m.id, "type": m.type.value, "title": m.title}
                for m in memories
            ],
        }, indent=2)

    @mcp.tool()
    def record_screen(
        duration: int = 30,
        fps: int = 15,
        filename: str = "",
    ) -> str:
        """
        Record the screen for a given duration.

        Args:
            duration: Recording duration in seconds (default 30)
            fps: Frames per second (default 15)
            filename: Output filename (auto-generated if empty)
        """
        from ..video.screen_recorder import ScreenRecorder

        recorder = ScreenRecorder(output_dir=f"{config.data_dir}/recordings")
        path = recorder.record(
            duration=duration, fps=fps,
            output=filename or None,
        )
        return json.dumps({
            "status": "recorded",
            "path": path,
            "duration": duration,
        }, indent=2)

    @mcp.tool()
    def screenshot() -> str:
        """Take a screenshot and save it."""
        from ..video.screen_recorder import ScreenRecorder

        recorder = ScreenRecorder(output_dir=f"{config.data_dir}/recordings")
        path = recorder.screenshot()
        return json.dumps({"status": "captured", "path": path})

    return mcp


def main():
    """Run the MCP server (stdio transport)."""
    server = create_server()
    server.run()


if __name__ == "__main__":
    main()
