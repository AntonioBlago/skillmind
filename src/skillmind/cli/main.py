"""
SkillMind CLI — command-line interface for memory management.

Usage:
    skillmind init                    # Initialize SkillMind in current project
    skillmind remember "content"      # Store a memory
    skillmind recall "query"          # Semantic search
    skillmind list                    # List all memories
    skillmind forget <id>             # Delete a memory
    skillmind import                  # Import from Claude Code markdown files
    skillmind consolidate             # Cleanup and merge duplicates
    skillmind context                 # Generate context for current situation
    skillmind stats                   # Show store statistics
    skillmind serve                   # Start MCP server
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from ..config import SkillMindConfig
from ..embeddings import EmbeddingEngine
from ..models import MemorySource, MemoryType, QueryFilter
from ..store import create_store
from ..trainer import Trainer

console = Console()


def _get_components(config_path: str | None = None):
    """Initialize all SkillMind components."""
    config = SkillMindConfig.load(config_path).resolve_env()
    engine = EmbeddingEngine(config.embedding)
    store = create_store(config, engine)
    store.initialize()
    trainer = Trainer(store)
    return config, engine, store, trainer


@click.group()
@click.option("--config", "-c", default=None, help="Path to config.yml")
@click.pass_context
def cli(ctx: click.Context, config: str | None) -> None:
    """SkillMind — Active memory & skill layer for AI coding assistants."""
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config


@cli.command()
@click.option("--backend", "-b", default="chroma", type=click.Choice(["chroma", "pinecone", "supabase", "qdrant", "faiss", "falkordb"]))
@click.option("--data-dir", "-d", default=".skillmind")
def init(backend: str, data_dir: str) -> None:
    """Initialize SkillMind in the current project."""
    config = SkillMindConfig(
        data_dir=data_dir,
        store={"backend": backend},
    )
    config.save()

    # Create directory structure
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    (data_path / "exports").mkdir(exist_ok=True)

    console.print(f"[green]Initialized SkillMind with {backend} backend[/green]")
    console.print(f"  Config: {data_dir}/config.yml")
    console.print(f"  Data:   {data_dir}/")
    console.print()
    console.print("[dim]Next steps:[/dim]")
    console.print("  skillmind import          # Import existing Claude Code memories")
    console.print("  skillmind remember '...'  # Store a new memory")
    console.print("  skillmind recall '...'    # Search memories")


@cli.command()
@click.argument("content")
@click.option("--title", "-t", default="", help="Short title")
@click.option("--type", "-T", "mem_type", default="", type=click.Choice(["", "user", "feedback", "project", "reference", "skill"]))
@click.option("--topic", default="", help="Primary topic tag")
@click.option("--tags", default="", help="Comma-separated tags")
@click.pass_context
def remember(ctx: click.Context, content: str, title: str, mem_type: str, topic: str, tags: str) -> None:
    """Store a new memory."""
    _, _, store, trainer = _get_components(ctx.obj.get("config_path"))

    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else None
    force_type = MemoryType(mem_type) if mem_type else None

    memory = trainer.learn(
        content=content,
        title=title or None,
        source=MemorySource.MANUAL,
        force_type=force_type,
        force_topic=topic or None,
        tags=tag_list,
    )

    if memory:
        console.print(f"[green]Stored:[/green] {memory.title}")
        console.print(f"  ID:    {memory.id}")
        console.print(f"  Type:  {memory.type.value}")
        console.print(f"  Topic: {memory.topic}")
    else:
        console.print("[yellow]Skipped: similar memory already exists[/yellow]")


@cli.command()
@click.argument("query")
@click.option("--limit", "-n", default=5, help="Max results")
@click.option("--type", "-T", "mem_type", default="")
@click.option("--topic", default="")
@click.pass_context
def recall(ctx: click.Context, query: str, limit: int, mem_type: str, topic: str) -> None:
    """Semantic search across all memories."""
    _, _, store, _ = _get_components(ctx.obj.get("config_path"))

    qf = QueryFilter()
    if mem_type:
        qf.types = [MemoryType(mem_type)]
    if topic:
        qf.topics = [topic]

    results = store.query(query, limit=limit, filter=qf)

    if not results:
        console.print("[dim]No matching memories found.[/dim]")
        return

    table = Table(title=f"Results for: {query}")
    table.add_column("Score", style="cyan", width=6)
    table.add_column("Type", style="magenta", width=10)
    table.add_column("Topic", style="green", width=15)
    table.add_column("Title", width=40)
    table.add_column("ID", style="dim", width=12)

    for r in results:
        table.add_row(
            f"{r.score:.2f}",
            r.memory.type.value,
            r.memory.topic,
            r.memory.title[:40],
            r.memory.id[:12],
        )

    console.print(table)

    # Show full content of top result
    if results:
        top = results[0]
        console.print(f"\n[bold]Top result:[/bold] {top.memory.title}")
        console.print(top.memory.content)


@cli.command("list")
@click.option("--type", "-T", "mem_type", default="")
@click.option("--topic", default="")
@click.option("--limit", "-n", default=20)
@click.pass_context
def list_memories(ctx: click.Context, mem_type: str, topic: str, limit: int) -> None:
    """List all memories with optional filtering."""
    _, _, store, _ = _get_components(ctx.obj.get("config_path"))

    qf = QueryFilter()
    if mem_type:
        qf.types = [MemoryType(mem_type)]
    if topic:
        qf.topics = [topic]

    memories = store.list_all(filter=qf, limit=limit)

    table = Table(title=f"Memories ({len(memories)})")
    table.add_column("Type", style="magenta", width=10)
    table.add_column("Topic", style="green", width=15)
    table.add_column("Title", width=45)
    table.add_column("Confidence", style="cyan", width=10)
    table.add_column("ID", style="dim", width=12)

    for m in memories:
        table.add_row(
            m.type.value,
            m.topic,
            m.title[:45],
            f"{m.confidence:.2f}",
            m.id[:12],
        )

    console.print(table)


@cli.command()
@click.argument("memory_id")
@click.pass_context
def forget(ctx: click.Context, memory_id: str) -> None:
    """Delete a memory by ID."""
    _, _, store, _ = _get_components(ctx.obj.get("config_path"))

    # Support partial ID matching
    if len(memory_id) < 36:
        all_mems = store.list_all(limit=10000)
        matches = [m for m in all_mems if m.id.startswith(memory_id)]
        if len(matches) == 0:
            console.print(f"[red]No memory found starting with {memory_id}[/red]")
            return
        if len(matches) > 1:
            console.print(f"[yellow]Ambiguous ID, {len(matches)} matches. Be more specific.[/yellow]")
            return
        memory_id = matches[0].id

    success = store.delete(memory_id)
    if success:
        console.print(f"[green]Deleted memory {memory_id}[/green]")
    else:
        console.print(f"[red]Memory not found: {memory_id}[/red]")


@cli.command("import")
@click.option("--source", "-s", default="", help="Path to memory directory")
@click.option("--dry-run", is_flag=True, help="Preview without importing")
@click.pass_context
def import_memories(ctx: click.Context, source: str, dry_run: bool) -> None:
    """Import existing Claude Code markdown memories into SkillMind."""
    from ..migration import migrate_memories

    _, _, store, trainer = _get_components(ctx.obj.get("config_path"))

    console.print(f"[bold]{'DRY RUN — ' if dry_run else ''}Importing memories...[/bold]")

    stats = migrate_memories(
        trainer=trainer,
        source_dir=source or None,
        dry_run=dry_run,
    )

    console.print(f"  Files found:     {stats['files_found']}")
    console.print(f"  [green]Imported:        {stats['imported']}[/green]")
    console.print(f"  [yellow]Skipped (dupes): {stats['skipped_duplicate']}[/yellow]")
    console.print(f"  [red]Skipped (error): {stats['skipped_error']}[/red]")

    if dry_run and stats["memories"]:
        console.print("\n[bold]Would import:[/bold]")
        for m in stats["memories"]:
            console.print(f"  [{m.get('type', '?')}] {m.get('name', '?')}: {m.get('content_preview', '')[:60]}")
        console.print("\n[dim]Run without --dry-run to actually import.[/dim]")


_BACKENDS = ["chroma", "pinecone", "supabase", "qdrant", "faiss", "falkordb"]


@cli.command("migrate")
@click.option("--from", "from_backend", required=True, type=click.Choice(_BACKENDS), help="Source backend")
@click.option("--to", "to_backend", required=True, type=click.Choice(_BACKENDS), help="Target backend")
@click.option("--batch-size", default=100, help="Memories written per batch")
@click.option("--dry-run", is_flag=True, help="Count source memories without writing")
@click.option("--yes", "-y", is_flag=True, help="Skip the confirmation prompt")
@click.pass_context
def migrate(ctx: click.Context, from_backend: str, to_backend: str, batch_size: int, dry_run: bool, yes: bool) -> None:
    """Copy all memories from one store backend to another (e.g. pinecone -> falkordb).

    Both backends are read from the same config — the source uses the credentials
    for --from, the target those for --to. Memories keep their IDs (idempotent) and
    are re-embedded by the target with the shared embedding model.
    """
    from ..migration import migrate_store

    if from_backend == to_backend:
        console.print("[red]--from and --to must differ.[/red]")
        sys.exit(1)

    config = SkillMindConfig.load(ctx.obj.get("config_path")).resolve_env()
    engine = EmbeddingEngine(config.embedding)

    src_cfg = config.model_copy(deep=True)
    src_cfg.store.backend = from_backend
    source = create_store(src_cfg, engine)

    tgt_cfg = config.model_copy(deep=True)
    tgt_cfg.store.backend = to_backend
    target = create_store(tgt_cfg, engine)

    try:
        source.initialize()
        target.initialize()
    except Exception as exc:  # noqa: BLE001 - surface connection/setup errors clearly
        console.print(f"[red]Failed to initialize stores: {exc}[/red]")
        sys.exit(1)

    src_total = source.count(filter=QueryFilter(include_expired=True))
    console.print(f"[bold]Migrating {from_backend} -> {to_backend}[/bold]")
    console.print(f"  Source ({from_backend}) holds [cyan]{src_total}[/cyan] memories.")
    console.print(f"  Embedding model: {config.embedding.model} (dim {config.embedding.dimension})")

    if dry_run:
        stats = migrate_store(source, target, batch_size=batch_size, dry_run=True)
        console.print(f"[yellow]DRY RUN — would migrate {stats['fetched']} memories.[/yellow]")
        if stats["truncated"]:
            console.print(f"[red]Warning: source has more than the fetch cap; {stats['fetched']} of {stats['source_count']} would be copied.[/red]")
        return

    if src_total == 0:
        console.print("[yellow]Nothing to migrate — source is empty.[/yellow]")
        return

    if not yes:
        click.confirm(f"Write {src_total} memories into the {to_backend} store?", abort=True)

    with console.status(f"Migrating to {to_backend}..."):
        stats = migrate_store(
            source,
            target,
            batch_size=batch_size,
            progress_cb=lambda done, total: None,
        )

    console.print(f"  [green]Migrated: {stats['migrated']} memories in {stats['batches']} batches[/green]")
    if stats["truncated"]:
        console.print(f"  [red]Warning: only {stats['fetched']} of {stats['source_count']} were fetched (fetch cap). Re-run with a higher cap.[/red]")

    tgt_total = target.count(filter=QueryFilter(include_expired=True))
    console.print(f"  Target ({to_backend}) now holds [cyan]{tgt_total}[/cyan] memories.")


@cli.command()
@click.pass_context
def consolidate(ctx: click.Context) -> None:
    """Cleanup: merge duplicates, expire stale memories."""
    _, _, store, trainer = _get_components(ctx.obj.get("config_path"))

    console.print("[bold]Running consolidation...[/bold]")
    stats = trainer.consolidate()

    console.print(f"  [green]Merged:  {stats['merged']}[/green]")
    console.print(f"  [yellow]Expired: {stats['expired']}[/yellow]")
    console.print(f"  [cyan]Updated: {stats['updated']}[/cyan]")


@cli.command()
@click.option("--file", "-f", default="", help="Current file path")
@click.option("--topic", "-t", default="", help="Current topic")
@click.option("--query", "-q", default="", help="Direct query")
@click.option("--output", "-o", default="", help="Write to file instead of stdout")
@click.pass_context
def context(ctx: click.Context, file: str, topic: str, query: str, output: str) -> None:
    """Generate focused context for current situation."""
    from ..context import ContextGenerator

    _, _, store, _ = _get_components(ctx.obj.get("config_path"))
    gen = ContextGenerator(store)

    ctx_text = gen.generate(
        current_file=file or None,
        current_topic=topic or None,
        query=query or None,
    )

    if output:
        Path(output).write_text(ctx_text, encoding="utf-8")
        console.print(f"[green]Context written to {output}[/green]")
    else:
        console.print(ctx_text)


@cli.command()
@click.pass_context
def stats(ctx: click.Context) -> None:
    """Show memory store statistics."""
    config, _, store, _ = _get_components(ctx.obj.get("config_path"))

    total = store.count()

    table = Table(title="SkillMind Statistics")
    table.add_column("Metric", style="bold")
    table.add_column("Value", style="cyan")

    table.add_row("Backend", config.store.backend)
    table.add_row("Embedding model", config.embedding.model)
    table.add_row("Total memories", str(total))

    for mt in MemoryType:
        n = store.count(filter=QueryFilter(types=[mt]))
        table.add_row(f"  {mt.value}", str(n))

    console.print(table)


@cli.command()
@click.option("--transport", "-t", default="stdio", type=click.Choice(["stdio", "http"]))
@click.option("--port", "-p", default=8765, type=int)
def serve(transport: str, port: int) -> None:
    """Start the SkillMind MCP server."""
    from ..mcp.server import create_server

    server = create_server()

    console.print(f"[bold green]Starting SkillMind MCP server ({transport})...[/bold green]")

    if transport == "stdio":
        server.run()
    else:
        server.run(transport="streamable-http", host="0.0.0.0", port=port)


# ── Video & YouTube Commands ──────────────────────────────────


@cli.command("learn-youtube")
@click.argument("video_url")
@click.option("--topic", default="", help="Override topic")
@click.option("--tags", default="", help="Comma-separated tags")
@click.pass_context
def learn_youtube(ctx: click.Context, video_url: str, topic: str, tags: str) -> None:
    """Learn knowledge from a YouTube video."""
    from ..video.youtube_learner import YouTubeLearner

    _, _, store, trainer = _get_components(ctx.obj.get("config_path"))
    yt = YouTubeLearner(trainer=trainer)

    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else None
    console.print(f"[bold]Learning from YouTube: {video_url}[/bold]")

    memories = yt.learn(video_url, force_topic=topic or None, tags=tag_list)

    console.print(f"[green]{len(memories)} memories created:[/green]")
    for m in memories:
        console.print(f"  [{m.type.value}] {m.title}")


@cli.command("learn-channel")
@click.argument("channel_id")
@click.option("--max", "-m", "max_videos", default=5, help="Max videos to process")
@click.option("--topic", default="", help="Override topic")
@click.pass_context
def learn_channel(ctx: click.Context, channel_id: str, max_videos: int, topic: str) -> None:
    """Learn from latest videos of a YouTube channel."""
    from ..video.youtube_learner import YouTubeLearner

    _, _, store, trainer = _get_components(ctx.obj.get("config_path"))
    yt = YouTubeLearner(trainer=trainer)

    console.print(f"[bold]Learning from channel: {channel_id} (max {max_videos} videos)[/bold]")
    memories = yt.learn_channel(channel_id, max_videos, force_topic=topic or None)
    console.print(f"[green]{len(memories)} memories created[/green]")


@cli.command("learn-video")
@click.argument("video_path")
@click.option("--topic", default="", help="Override topic")
@click.option("--tags", default="", help="Comma-separated tags")
@click.option("--audio", is_flag=True, help="Also transcribe audio via Whisper")
@click.pass_context
def learn_video(ctx: click.Context, video_path: str, topic: str, tags: str, audio: bool) -> None:
    """Learn from a local video file or screen recording."""
    from ..video.video_learner import VideoLearner

    _, _, store, trainer = _get_components(ctx.obj.get("config_path"))
    vl = VideoLearner(trainer=trainer)

    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else None
    console.print(f"[bold]Learning from video: {video_path}[/bold]")

    memories = vl.learn(video_path, force_topic=topic or None, tags=tag_list, extract_audio=audio)
    console.print(f"[green]{len(memories)} memories created[/green]")


@cli.command("record")
@click.option("--duration", "-d", default=30, help="Duration in seconds")
@click.option("--fps", default=15, help="Frames per second")
@click.option("--output", "-o", default="", help="Output filename")
@click.pass_context
def record_screen(ctx: click.Context, duration: int, fps: int, output: str) -> None:
    """Record the screen."""
    from ..video.screen_recorder import ScreenRecorder

    config, _, _, _ = _get_components(ctx.obj.get("config_path"))
    recorder = ScreenRecorder(output_dir=f"{config.data_dir}/recordings")

    console.print(f"[bold]Recording screen for {duration}s at {fps} FPS...[/bold]")
    path = recorder.record(duration=duration, fps=fps, output=output or None)
    console.print(f"[green]Saved: {path}[/green]")


@cli.command("screenshot")
@click.option("--output", "-o", default="", help="Output filename")
@click.pass_context
def take_screenshot(ctx: click.Context, output: str) -> None:
    """Take a screenshot."""
    from ..video.screen_recorder import ScreenRecorder

    config, _, _, _ = _get_components(ctx.obj.get("config_path"))
    recorder = ScreenRecorder(output_dir=f"{config.data_dir}/recordings")

    path = recorder.screenshot(output=output or None)
    console.print(f"[green]Saved: {path}[/green]")


# ── Obsidian Vault Export Commands ────────────────────────────


@cli.command("export")
@click.argument("vault_path")
@click.option("--full-rebuild", is_flag=True, help="Regenerate all wiki pages from scratch")
@click.pass_context
def export_obsidian(ctx: click.Context, vault_path: str, full_rebuild: bool) -> None:
    """Export all memories to an Obsidian vault (Karpathy wiki pattern)."""
    from ..exporters.obsidian import ObsidianExporter

    config, _, store, _ = _get_components(ctx.obj.get("config_path"))
    exporter = ObsidianExporter(vault_path)
    memories = store.list_all(limit=10000)

    console.print(f"[bold]Exporting {len(memories)} memories to {vault_path}[/bold]")
    stats = exporter.export(memories, full_rebuild=full_rebuild)

    console.print(f"  [green]Created: {stats['pages_created']} pages[/green]")
    console.print(f"  [yellow]Updated: {stats['pages_updated']} pages[/yellow]")
    console.print(f"  Open {vault_path} in Obsidian to see the knowledge graph.")

    if not config.obsidian.vault_path:
        config.obsidian.vault_path = vault_path
        config.save()


@cli.command("sync")
@click.option("--vault", "-v", default="", help="Vault path (uses config if empty)")
@click.pass_context
def sync_obsidian(ctx: click.Context, vault: str) -> None:
    """Incrementally sync new memories to an existing Obsidian vault."""
    from ..exporters.obsidian import ObsidianExporter

    config, _, store, _ = _get_components(ctx.obj.get("config_path"))
    vault_path = vault or config.obsidian.vault_path

    if not vault_path:
        console.print("[red]No vault path. Run 'skillmind export <path>' first.[/red]")
        return

    exporter = ObsidianExporter(vault_path)
    memories = store.list_all(limit=10000)

    console.print(f"[bold]Syncing {len(memories)} memories to {vault_path}[/bold]")
    stats = exporter.sync(memories)

    console.print(f"  [green]Created: {stats['pages_created']} new pages[/green]")
    console.print(f"  [dim]Skipped: {stats['pages_skipped']} existing pages[/dim]")


# ── OKF (Open Knowledge Format) Commands ─────────────────────


@cli.command("export-okf")
@click.argument("bundle_path")
@click.option("--full-rebuild", is_flag=True, help="Clear concept folders before writing")
@click.option("--title", default="SkillMind Knowledge Base", help="Bundle title (shown in index.md)")
@click.pass_context
def export_okf(ctx: click.Context, bundle_path: str, full_rebuild: bool, title: str) -> None:
    """Export all memories to a spec-compliant OKF bundle (portable second brain)."""
    from ..exporters.okf import OKFExporter

    _, _, store, _ = _get_components(ctx.obj.get("config_path"))
    exporter = OKFExporter(bundle_path, bundle_title=title)
    memories = store.list_all(limit=10000)

    console.print(f"[bold]Exporting {len(memories)} memories to OKF bundle {bundle_path}[/bold]")
    stats = exporter.export(memories, full_rebuild=full_rebuild)

    console.print(f"  [green]Created: {stats['concepts_created']} concepts[/green]")
    console.print(f"  [yellow]Updated: {stats['concepts_updated']} concepts[/yellow]")
    console.print(f"  Entry point: {bundle_path}/index.md")


@cli.command("import-okf")
@click.argument("bundle_path")
@click.option("--dry-run", is_flag=True, help="Preview without importing")
@click.pass_context
def import_okf(ctx: click.Context, bundle_path: str, dry_run: bool) -> None:
    """Import an external OKF bundle into the store (classified + deduplicated)."""
    from ..importers.okf import import_okf_bundle

    _, _, store, trainer = _get_components(ctx.obj.get("config_path"))

    console.print(f"[bold]{'DRY RUN — ' if dry_run else ''}Importing OKF bundle {bundle_path}[/bold]")
    stats = import_okf_bundle(trainer, bundle_path, dry_run=dry_run)

    console.print(f"  Concepts found:  {stats['files_found']}")
    console.print(f"  [green]Imported:        {stats['imported']}[/green]")
    console.print(f"  [yellow]Skipped (dupes): {stats['skipped_duplicate']}[/yellow]")
    console.print(f"  [red]Skipped (error): {stats['skipped_error']}[/red]")

    if dry_run and stats["concepts"]:
        console.print("\n[bold]Would import:[/bold]")
        for c in stats["concepts"][:50]:
            console.print(f"  [{c.get('type', '?')}] {c.get('title', '?')} ({c.get('concept_id', '')})")
        console.print("\n[dim]Run without --dry-run to actually import.[/dim]")


@cli.command("viz-okf")
@click.argument("bundle_path")
@click.option("--output", "-o", default="okf-graph.html", help="HTML file name (written inside the bundle)")
@click.option("--title", default="", help="Graph title (defaults to the bundle folder name)")
@click.option("--open", "open_browser", is_flag=True, help="Open the graph in the default browser when done")
def viz_okf(bundle_path: str, output: str, title: str, open_browser: bool) -> None:
    """Render an OKF bundle as a self-contained local HTML knowledge graph (no server)."""
    import webbrowser
    from ..exporters.okf_viz import OKFVisualizer

    visualizer = OKFVisualizer(bundle_path, title=title or None)
    console.print(f"[bold]Visualizing OKF bundle {bundle_path}[/bold]")
    out_path = visualizer.build(output_name=output)
    stats = visualizer.build_graph()["stats"]

    console.print(f"  [green]Concepts: {stats['concepts']}[/green]  "
                  f"[cyan]Links: {stats['edges']}[/cyan]  "
                  f"[dim]Types: {stats['types']}[/dim]")
    console.print(f"  Graph written to {out_path}")
    console.print("  [dim]Open it in any browser (CDN needed only on first load).[/dim]")

    if open_browser:
        webbrowser.open(out_path.resolve().as_uri())


@cli.command("enrich")
@click.option("--source-type", "-t", required=True, type=click.Choice(["markdown", "web"]), help="Knowledge source type")
@click.option("--source", "-s", "source_target", required=True, help="Path (markdown) or URL (web)")
@click.option("--type", "force_type", default="", type=click.Choice(["", "user", "feedback", "project", "reference", "skill"]), help="Force memory type")
@click.option("--tag", "tags", multiple=True, help="Tag added to every harvested document (repeatable)")
@click.option("--limit", default=0, help="Stop after N documents (0 = no limit)")
@click.option("--dry-run", is_flag=True, help="Discover without storing")
@click.pass_context
def enrich(
    ctx: click.Context,
    source_type: str,
    source_target: str,
    force_type: str,
    tags: tuple,
    limit: int,
    dry_run: bool,
) -> None:
    """Enrich the store from a pluggable knowledge source (the 'second brain' loop)."""
    from ..enrichment import EnrichmentRunner
    from ..sources import create_source

    _, _, store, trainer = _get_components(ctx.obj.get("config_path"))

    source = create_source(source_type, source_target)
    runner = EnrichmentRunner(trainer)
    mem_type = MemoryType(force_type) if force_type else None

    console.print(f"[bold]{'DRY RUN — ' if dry_run else ''}Enriching from {source.name}: {source_target}[/bold]")
    stats = runner.run(
        source,
        dry_run=dry_run,
        default_type=mem_type,
        default_tags=list(tags) if tags else None,
        limit=limit or None,
    )

    console.print(f"  Discovered:      {stats['discovered']}")
    console.print(f"  [green]Imported:        {stats['imported']}[/green]")
    console.print(f"  [yellow]Skipped (dupes): {stats['skipped_duplicate']}[/yellow]")
    console.print(f"  [dim]Skipped (empty): {stats['skipped_empty']}[/dim]")


# ── Setup Command ─────────────────────────────────────────────


@cli.command("setup")
@click.option("--backend", "-b", default="chroma", type=click.Choice(["chroma", "pinecone", "supabase", "qdrant", "faiss", "falkordb"]))
@click.option("--scan-dir", "-s", multiple=True, help="Additional directories to scan")
@click.option("--dry-run", is_flag=True, help="Preview without importing")
@click.pass_context
def setup(ctx: click.Context, backend: str, scan_dir: tuple, dry_run: bool) -> None:
    """Full setup: discover, import, and structure ALL existing knowledge."""
    from ..setup import run_setup

    config_path = ctx.obj.get("config_path")
    data_dir = ".skillmind"
    if config_path:
        cfg = SkillMindConfig.load(config_path)
        data_dir = cfg.data_dir

    console.print("[bold]SkillMind Setup[/bold]")
    console.print("=" * 50)

    stats = run_setup(
        backend=backend,
        data_dir=data_dir,
        scan_dirs=list(scan_dir) if scan_dir else None,
        dry_run=dry_run,
    )

    console.print(f"\n[bold green]Setup complete! {stats['total_imported']} items imported.[/bold green]")


if __name__ == "__main__":
    cli()
