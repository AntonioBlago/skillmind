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

import asyncio
import json
import sys
from typing import Any

# Import Context at module scope so that pydantic (via FastMCP 3.x) can resolve
# the string forward-ref produced by `from __future__ import annotations` when it
# inspects tool signatures like `ctx: Context = None`. Forward-refs are evaluated
# against the function's module globals, not the enclosing function locals, so a
# local `from fastmcp import Context` inside create_server() is not enough.
try:
    from fastmcp import Context
except ImportError:  # fastmcp is an optional extra; create_server() re-checks
    Context = Any  # type: ignore[assignment,misc]

from ..config import SkillMindConfig
from ..context import ContextGenerator
from ..embeddings import EmbeddingEngine
from ..migration import migrate_memories
from ..models import Memory, MemorySource, MemoryType, QueryFilter
from ..store import create_store
from ..trainer import Trainer


def create_server():
    """Create and configure the MCP server."""
    from fastmcp import Context, FastMCP

    mcp = FastMCP("skillmind")

    # Initialize components
    config = SkillMindConfig.load().resolve_env()
    engine = EmbeddingEngine(config.embedding)
    store = create_store(config, engine)
    store.initialize()
    trainer = Trainer(store)
    context_gen = ContextGenerator(store, max_tokens=config.context_max_tokens)

    from ..review import ReviewQueue
    review_queue = ReviewQueue(queue_path=f"{config.data_dir}/review_queue.json")

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
    async def learn_youtube(
        video_url: str,
        topic: str = "",
        tags: str = "",
        ctx: Context = None,
    ) -> str:
        """
        Learn knowledge from a YouTube video. Extracts transcript,
        structures key insights, and stores as memories.

        Emits live MCP progress notifications (4 phases) plus a heartbeat
        during the long Claude extraction step, so clients can show status.

        Args:
            video_url: YouTube video URL or video ID
            topic: Override auto-detected topic
            tags: Comma-separated tags
        """
        from ..video.youtube_learner import YouTubeLearner

        yt = YouTubeLearner(trainer=trainer)
        tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else None

        # Shared state: written by the worker thread via the callback,
        # read by the heartbeat loop on the event loop thread.
        state = {"step": 0, "total": 4, "msg": "Starte YouTube-Lernen…"}

        def _cb(step: int, total: int, msg: str) -> None:
            state["step"], state["total"], state["msg"] = step, total, msg

        if ctx is not None:
            await ctx.report_progress(progress=0, total=4, message=state["msg"])

        task = asyncio.create_task(
            asyncio.to_thread(yt.learn, video_url, topic or None, tag_list, _cb)
        )

        # Heartbeat: re-emit progress every 2s (keepalive during the 1–2 min
        # Claude call) and log a line whenever the phase advances.
        last_step = -1
        while not task.done():
            if ctx is not None:
                await ctx.report_progress(
                    progress=state["step"], total=state["total"], message=state["msg"]
                )
                if state["step"] != last_step:
                    await ctx.info(f"[{state['step']}/{state['total']}] {state['msg']}")
                    last_step = state["step"]
            await asyncio.sleep(2)

        memories = await task  # re-raises any exception from the worker thread

        if ctx is not None:
            await ctx.report_progress(
                progress=4, total=4, message="Fertig — in Pinecone gespeichert"
            )

        # Build markdown from stored memory content (no extra network call —
        # metadata is reconstructed from the first stored memory).
        first = memories[0] if memories else None
        first_meta = (first.metadata or {}) if first else {}
        md_metadata = {
            "title": first.title if first else "",
            "author": first_meta.get("author", ""),
            "duration": first_meta.get("duration", 0),
            "url": first_meta.get("video_url", ""),
        }
        knowledge = {
            "title": first.title if first else "",
            "topic": topic or "youtube",
            "tags": tag_list or [],
            "summary": first.content if first else "",
            "key_takeaways": first_meta.get("key_takeaways", []),
        }
        markdown = yt.format_markdown(md_metadata, knowledge)

        return json.dumps({
            "status": "learned",
            "memories_created": len(memories),
            "memories": [
                {"id": m.id, "type": m.type.value, "title": m.title, "topic": m.topic}
                for m in memories
            ],
            "markdown": markdown,
        }, indent=2)

    @mcp.tool()
    async def learn_youtube_channel(
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
        memories = await yt.learn_channel_async(channel_id, max_videos, force_topic=topic or None)
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

    # ─── Custom Pattern Management ────────────────────────────

    @mcp.tool()
    def add_pattern(
        pattern: str,
        memory_type: str = "feedback",
        topic: str = "",
        description: str = "",
    ) -> str:
        """
        Add a custom auto-detection pattern. When a user message matches the regex,
        it will be automatically saved as a memory of the given type.

        Args:
            pattern: Regex pattern (e.g. "\\bSEO\\b.*\\b(tipp|trick|hack)\\b")
            memory_type: Memory type to assign: user, feedback, project, reference, skill
            topic: Force this topic (empty = auto-detect)
            description: What this pattern detects (for your reference)

        Examples:
            pattern="\\brezept\\b|\\brezeptur\\b", memory_type="skill", topic="cooking"
            pattern="\\b(paroc|x-bionic|tanzraum)\\b", memory_type="project", topic="client_work"
            pattern="\\bpreis\\b.*\\b(pro|monat|stunde)\\b", memory_type="project", topic="pricing"
        """
        from ..config import CustomPattern as CP

        # Validate regex
        import re
        try:
            re.compile(pattern)
        except re.error as e:
            return json.dumps({"status": "error", "message": f"Invalid regex: {e}"})

        # Validate memory type
        try:
            MemoryType(memory_type)
        except ValueError:
            return json.dumps({"status": "error", "message": f"Invalid type: {memory_type}. Use: user, feedback, project, reference, skill"})

        # Add to config
        new_pattern = CP(pattern=pattern, memory_type=memory_type, topic=topic, description=description)
        config.listener.custom_patterns.append(new_pattern)
        config.save()

        return json.dumps({
            "status": "added",
            "pattern": pattern,
            "memory_type": memory_type,
            "topic": topic or "(auto-detect)",
            "description": description,
            "total_patterns": len(config.listener.custom_patterns),
        }, indent=2)

    @mcp.tool()
    def list_patterns() -> str:
        """List all custom auto-detection patterns."""
        patterns = config.listener.custom_patterns
        return json.dumps({
            "count": len(patterns),
            "patterns": [
                {
                    "index": i,
                    "pattern": p.pattern,
                    "memory_type": p.memory_type,
                    "topic": p.topic or "(auto-detect)",
                    "description": p.description,
                }
                for i, p in enumerate(patterns)
            ],
        }, indent=2)

    @mcp.tool()
    def remove_pattern(index: int) -> str:
        """
        Remove a custom pattern by its index number (from list_patterns).

        Args:
            index: Pattern index to remove (0-based)
        """
        patterns = config.listener.custom_patterns
        if index < 0 or index >= len(patterns):
            return json.dumps({"status": "error", "message": f"Index {index} out of range (0-{len(patterns)-1})"})

        removed = patterns.pop(index)
        config.save()

        return json.dumps({
            "status": "removed",
            "pattern": removed.pattern,
            "description": removed.description,
            "remaining": len(patterns),
        }, indent=2)

    # ─── Obsidian Vault Export Tools ─────────────────────────────

    @mcp.tool()
    def export_obsidian(
        vault_path: str = "",
        full_rebuild: bool = False,
    ) -> str:
        """
        Export all memories to an Obsidian vault (Karpathy wiki pattern).

        Creates a wiki/ folder with interlinked markdown pages, an index,
        topic MOC pages, and an operation log. Open the vault in Obsidian
        to see the knowledge graph.

        Args:
            vault_path: Path to Obsidian vault (uses config if empty)
            full_rebuild: If true, regenerate all wiki pages from scratch
        """
        from ..exporters.obsidian import ObsidianExporter

        path = vault_path or config.obsidian.vault_path
        if not path:
            return json.dumps({"status": "error", "message": "No vault_path provided. Pass vault_path or set obsidian.vault_path in config."})

        exporter = ObsidianExporter(path)
        memories = store.list_all(limit=10000)
        stats = exporter.export(memories, full_rebuild=full_rebuild)

        # Save path to config if not already set
        if not config.obsidian.vault_path:
            config.obsidian.vault_path = path
            config.save()

        return json.dumps({
            "status": "exported",
            "vault_path": str(path),
            **stats,
        }, indent=2)

    @mcp.tool()
    def sync_obsidian(vault_path: str = "") -> str:
        """
        Incrementally sync new memories to an existing Obsidian vault.
        Only writes pages for memories not already in the vault.

        Args:
            vault_path: Path to Obsidian vault (uses config if empty)
        """
        from ..exporters.obsidian import ObsidianExporter

        path = vault_path or config.obsidian.vault_path
        if not path:
            return json.dumps({"status": "error", "message": "No vault_path. Run export_obsidian first or set obsidian.vault_path in config."})

        exporter = ObsidianExporter(path)
        memories = store.list_all(limit=10000)
        stats = exporter.sync(memories)

        return json.dumps({
            "status": "synced",
            "vault_path": str(path),
            **stats,
        }, indent=2)

    # ─── OKF (Open Knowledge Format) Tools ───────────────────────

    @mcp.tool()
    def export_okf(
        bundle_path: str,
        full_rebuild: bool = False,
        title: str = "SkillMind Knowledge Base",
    ) -> str:
        """
        Export all memories to a spec-compliant OKF (Open Knowledge Format) bundle.

        OKF is a vendor-neutral knowledge format (markdown + YAML frontmatter)
        from Google's knowledge-catalog project. The bundle is a portable
        "second brain": an index.md entry point, per-category concept folders,
        standard markdown relative links as graph edges, a change log, and
        numbered Citations for provenance. Re-importable via import_okf_bundle.

        Args:
            bundle_path: Directory to write the OKF bundle into
            full_rebuild: If true, clear concept folders before writing
            title: Bundle title shown in index.md / README.md
        """
        from ..exporters.okf import OKFExporter

        if not bundle_path:
            return json.dumps({"status": "error", "message": "bundle_path is required."})

        exporter = OKFExporter(bundle_path, bundle_title=title)
        memories = store.list_all(limit=10000)
        stats = exporter.export(memories, full_rebuild=full_rebuild)

        return json.dumps({
            "status": "exported",
            "bundle_path": str(bundle_path),
            "entry_point": f"{bundle_path}/index.md",
            **stats,
        }, indent=2)

    @mcp.tool()
    def import_okf_bundle(bundle_path: str, dry_run: bool = False) -> str:
        """
        Import an external OKF bundle into the store.

        Reads a directory of OKF concept files (markdown + YAML frontmatter),
        maps each concept's freeform `type` onto a SkillMind memory type (or
        lets the Trainer classify it), and stores it with deduplication.
        Bundles produced by export_okf round-trip losslessly via their
        skillmind_* frontmatter keys.

        Args:
            bundle_path: Path to the OKF bundle directory (or a single concept file)
            dry_run: If true, parse and report but do not store
        """
        from ..importers.okf import import_okf_bundle as _import_okf

        if not bundle_path:
            return json.dumps({"status": "error", "message": "bundle_path is required."})

        stats = _import_okf(trainer, bundle_path, dry_run=dry_run)
        return json.dumps({
            "status": "dry_run" if dry_run else "imported",
            **stats,
        }, indent=2)

    @mcp.tool()
    def visualize_okf(
        bundle_path: str,
        output: str = "okf-graph.html",
        title: str = "",
    ) -> str:
        """
        Render an OKF bundle as a self-contained local HTML knowledge graph.

        Reads an OKF bundle directory and writes one standalone HTML file
        (Cytoscape.js force-directed graph + marked.js detail panel) modeled on
        Google's knowledge-catalog OKF viewer, but server-free: the whole graph
        is embedded, so the file works straight from disk (file://). Edges are
        the bundle's relative markdown links; nodes are colored by concept type.

        Args:
            bundle_path: Path to an existing OKF bundle directory (e.g. from export_okf)
            output: HTML file name written inside the bundle
            title: Graph title (defaults to the bundle folder name)
        """
        from ..exporters.okf_viz import OKFVisualizer

        if not bundle_path:
            return json.dumps({"status": "error", "message": "bundle_path is required."})

        visualizer = OKFVisualizer(bundle_path, title=title or None)
        try:
            out_path = visualizer.build(output_name=output)
        except NotADirectoryError as exc:
            return json.dumps({"status": "error", "message": str(exc)})

        stats = visualizer.build_graph()["stats"]
        return json.dumps({
            "status": "rendered",
            "html_path": str(out_path),
            "open_with": out_path.resolve().as_uri(),
            **stats,
        }, indent=2)

    @mcp.tool()
    def enrich_source(
        source_type: str,
        source: str,
        force_type: str = "",
        tags: str = "",
        limit: int = 0,
        dry_run: bool = False,
    ) -> str:
        """
        Enrich the store from a pluggable knowledge source (the 'second brain' loop).

        Harvests documents from a source, then sanitizes, classifies and
        deduplicates each one into a memory via the Trainer. The result can be
        exported as an OKF bundle (export_okf), closing the loop:
        any source -> SkillMind store -> OKF.

        Args:
            source_type: 'markdown' (a file/directory) or 'web' (a URL)
            source: path for markdown, or URL for web
            force_type: optional memory type (user|feedback|project|reference|skill)
            tags: comma-separated tags added to every harvested document
            limit: stop after N documents (0 = no limit)
            dry_run: discover and report without storing
        """
        from ..enrichment import EnrichmentRunner
        from ..sources import create_source
        from ..models import MemoryType as _MT

        if not source_type or not source:
            return json.dumps({"status": "error", "message": "source_type and source are required."})

        try:
            knowledge_source = create_source(source_type, source)
        except ValueError as exc:
            return json.dumps({"status": "error", "message": str(exc)})

        mem_type = None
        if force_type:
            try:
                mem_type = _MT(force_type)
            except ValueError:
                return json.dumps({"status": "error", "message": f"Unknown force_type: {force_type}"})

        tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else None
        runner = EnrichmentRunner(trainer)
        stats = runner.run(
            knowledge_source,
            dry_run=dry_run,
            default_type=mem_type,
            default_tags=tag_list,
            limit=limit or None,
        )
        return json.dumps({
            "status": "dry_run" if dry_run else "enriched",
            **stats,
        }, indent=2)

    # ─── Review Mode & Queue Tools ─────────────────────────────

    @mcp.tool()
    def set_review_mode(mode: str) -> str:
        """
        Set how auto-detected memories are handled.

        Args:
            mode: One of:
                - 'review' = queue for approval first (default, safest)
                - 'auto' = store directly without review (fastest)
                - 'off' = disable auto-detection completely
        """
        if mode not in ("review", "auto", "off"):
            return json.dumps({"status": "error", "message": "Mode must be: review, auto, or off"})

        config.listener.review_mode = mode
        config.save()
        return json.dumps({
            "status": "updated",
            "review_mode": mode,
            "description": {
                "review": "Auto-detected memories go to review queue first",
                "auto": "Auto-detected memories stored directly in vector DB",
                "off": "Auto-detection disabled, only manual remember works",
            }[mode],
        }, indent=2)

    @mcp.tool()
    def get_review_mode() -> str:
        """Show the current review mode (review/auto/off)."""
        return json.dumps({
            "review_mode": config.listener.review_mode,
            "pending_count": review_queue.count_pending(),
        }, indent=2)

    @mcp.tool()
    def review_pending() -> str:
        """
        Show all pending memories waiting for review.
        Auto-detected memories land here first before being stored in the vector DB.
        Review each one and approve or reject.
        """
        pending = review_queue.list_pending()
        stats = review_queue.stats()

        return json.dumps({
            "pending": stats["pending"],
            "approved_total": stats["approved"],
            "rejected_total": stats["rejected"],
            "items": [
                {
                    "id": e["id"],
                    "type": e["type"],
                    "topic": e["topic"],
                    "title": e["title"],
                    "content": e["content"][:300] + ("..." if len(e["content"]) > 300 else ""),
                    "trigger": e.get("trigger", ""),
                    "detected_at": e["detected_at"],
                }
                for e in pending
            ],
        }, indent=2)

    @mcp.tool()
    def approve_memory(entry_id: str) -> str:
        """
        Approve a pending memory and store it in the vector database.

        Args:
            entry_id: ID from review_pending (supports partial ID match)
        """
        memory = review_queue.approve(entry_id, trainer)
        if memory:
            return json.dumps({
                "status": "approved",
                "id": entry_id,
                "memory_id": memory.id,
                "type": memory.type.value,
                "topic": memory.topic,
                "title": memory.title,
                "remaining_pending": review_queue.count_pending(),
            }, indent=2)
        return json.dumps({"status": "not_found", "id": entry_id})

    @mcp.tool()
    def reject_memory(entry_id: str, reason: str = "") -> str:
        """
        Reject a pending memory (won't be stored).

        Args:
            entry_id: ID from review_pending
            reason: Why rejected (optional, for learning)
        """
        success = review_queue.reject(entry_id, reason)
        return json.dumps({
            "status": "rejected" if success else "not_found",
            "id": entry_id,
            "reason": reason,
            "remaining_pending": review_queue.count_pending(),
        }, indent=2)

    @mcp.tool()
    def approve_all_pending() -> str:
        """Approve all pending memories at once."""
        memories = review_queue.approve_all(trainer)
        return json.dumps({
            "status": "approved_all",
            "count": len(memories),
            "memories": [
                {"id": m.id, "type": m.type.value, "title": m.title[:60]}
                for m in memories
            ],
        }, indent=2)

    @mcp.tool()
    def reject_all_pending() -> str:
        """Reject all pending memories at once."""
        count = review_queue.reject_all()
        return json.dumps({
            "status": "rejected_all",
            "count": count,
        }, indent=2)

    @mcp.tool()
    def edit_pending(
        entry_id: str,
        content: str = "",
        type: str = "",
        topic: str = "",
        title: str = "",
    ) -> str:
        """
        Edit a pending memory before approving it.

        Args:
            entry_id: ID from review_pending
            content: New content (empty = keep)
            type: New type (empty = keep)
            topic: New topic (empty = keep)
            title: New title (empty = keep)
        """
        kwargs = {}
        if content:
            kwargs["content"] = content
        if type:
            kwargs["type"] = type
        if topic:
            kwargs["topic"] = topic
        if title:
            kwargs["title"] = title

        entry = review_queue.edit_pending(entry_id, **kwargs)
        if entry:
            return json.dumps({
                "status": "edited",
                "entry": {
                    "id": entry["id"],
                    "type": entry["type"],
                    "topic": entry["topic"],
                    "title": entry["title"],
                    "content": entry["content"][:200],
                },
            }, indent=2)
        return json.dumps({"status": "not_found", "id": entry_id})

    return mcp


def main():
    """Run the MCP server (stdio transport)."""
    server = create_server()
    server.run()


if __name__ == "__main__":
    main()
