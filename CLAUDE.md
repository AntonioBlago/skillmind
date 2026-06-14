# SkillMind

**Active Skill Listener & Trainer** — structured memory layer for AI coding assistants.

## What is this?

SkillMind replaces flat markdown memory files with a vector-database-backed memory system that:
- **Listens** to git events, file changes, and conversations
- **Trains** by auto-classifying, deduplicating, and consolidating knowledge
- **Surfaces** only relevant context per conversation (not everything)

## Architecture

```
src/skillmind/
├── models.py          # Memory, QueryFilter, QueryResult (Pydantic)
├── config.py          # SkillMindConfig (YAML-based)
├── embeddings.py      # EmbeddingEngine (sentence-transformers | openai)
├── trainer.py         # Auto-classify, dedup, merge, consolidate
├── sanitizer.py       # Redact secrets / personal data before storing
├── listener.py        # GitListener, FileListener, ConversationListener
├── context.py         # ContextGenerator — dynamic context for Claude Code
├── review.py          # Human-in-the-loop approval queue for pending memories
├── migration.py       # Import existing Claude Code markdown memories
├── setup.py           # Guided one-shot project setup (backend + initial scan)
├── enrichment.py      # EnrichmentRunner — any source → store → OKF ("second brain")
├── store/
│   ├── base.py        # Abstract MemoryStore interface
│   ├── chroma_store.py    # ChromaDB backend (default)
│   ├── faiss_store.py     # FAISS + JSON backend
│   ├── qdrant_store.py    # Qdrant backend
│   ├── pinecone_store.py  # Pinecone backend
│   ├── supabase_store.py  # Supabase/pgvector backend
│   └── falkordb_store.py  # FalkorDB graph+vector backend (GraphRAG)
├── sources/           # Pluggable enrichment inputs (decoupled from google-adk)
│   ├── base.py        # KnowledgeSource / RawDocument abstraction
│   ├── markdown_source.py # Harvest a folder of .md/.txt files
│   └── web_source.py  # Fetch URLs → readable text (stdlib-only by default)
├── exporters/
│   ├── obsidian.py    # Obsidian vault exporter (Karpathy wiki pattern)
│   ├── okf.py         # OKFExporter — spec-compliant OKF bundle (vendor-neutral)
│   └── okf_viz.py     # OKFVisualizer — self-contained local HTML knowledge graph (no server)
├── importers/
│   └── okf.py         # OKFImporter — read foreign OKF bundles via Trainer
├── video/
│   ├── youtube_learner.py # Learn from YouTube videos / channels
│   ├── video_learner.py   # Learn from local video files
│   └── screen_recorder.py # Screen recording + screenshots
├── mcp/
│   └── server.py      # MCP server (31 tools for Claude Code)
└── cli/
    └── main.py        # CLI (click-based)
```

**OKF integration** (Open Knowledge Format, Google's [knowledge-catalog](https://github.com/GoogleCloudPlatform/knowledge-catalog)): SkillMind adopts only the FORMAT/SPEC and the enrichment-loop *concept* — it does NOT vendor Google's `enrichment_agent` (tied to google-adk + BigQuery, not portable). An OKF bundle is a vendor-neutral directory of markdown concept files with YAML frontmatter, relative wiki links, an `index.md` (no frontmatter), a `log.md` (newest-first ISO date headings) and `# Citations` for provenance. Round-trip safe: SkillMind-produced bundles carry `skillmind_*` keys so type/topic/id/timestamps survive re-import.

## Key Commands

```bash
pip install -e ".[chroma]"          # Install with Chroma backend
skillmind init --backend chroma     # Initialize
skillmind setup                     # Guided setup (pick backend + initial scan)
skillmind import                    # Import existing Claude Code memories
skillmind migrate --from chroma --to faiss   # Move memories between backends
skillmind remember "content"        # Store memory
skillmind recall "query"            # Semantic search
skillmind list                      # List all
skillmind consolidate               # Cleanup
skillmind serve                     # Start MCP server
skillmind export ~/MyWiki           # Export to Obsidian vault
skillmind sync                      # Incremental sync to vault
skillmind export-okf ~/bundle       # Export as a portable OKF bundle
skillmind import-okf ~/bundle       # Import a foreign OKF bundle
skillmind viz-okf ~/bundle --open   # Render bundle as a local HTML knowledge graph
skillmind enrich --markdown ~/notes # Enrichment loop: source → store → OKF
skillmind learn-youtube <url>       # Learn from a YouTube video
skillmind learn-channel <id>        # Learn from a YouTube channel
skillmind learn-video <path>        # Learn from a local video file
skillmind record / screenshot       # Capture screen → memory
```

## Store Backends

All 6 implement the same `MemoryStore` interface (add, query, get, update, delete, list_all, count, clear):

| Backend | Best for | Requires |
|---------|----------|----------|
| **chroma** | Solo dev, local, default | `pip install chromadb` |
| **faiss** | Offline, air-gapped, fastest | `pip install faiss-cpu` |
| **qdrant** | Self-hosted or cloud, great filtering | Qdrant server |
| **pinecone** | Multi-device cloud sync | API key |
| **supabase** | SQL + vectors, team sharing | Supabase project |
| **falkordb** | GraphRAG (graph + vector, multi-hop) | FalkorDB server (`falkordb>=1.6`) |

> **FalkorDB note:** the vector index `score` is a COSINE DISTANCE (0 = identical), so the store exposes `similarity = 1 - distance`. GraphRAG multi-hop retrieval is opt-in via `store.falkordb_graphrag`.

## MCP Server Tools

31 tools exposed via FastMCP. Core memory: `remember`, `recall`, `forget`, `update_memory`, `context`, `consolidate`, `memory_stats`, `list_memories`, `import_markdown_memories`. OKF / enrichment: `export_okf`, `import_okf_bundle`, `visualize_okf`, `enrich_source`. Obsidian: `export_obsidian`, `sync_obsidian`. Learning: `learn_youtube`, `learn_youtube_channel`, `learn_video`, `record_screen`, `screenshot`. Patterns: `add_pattern`, `list_patterns`, `remove_pattern`. Review queue: `set_review_mode`, `get_review_mode`, `review_pending`, `approve_memory`, `reject_memory`, `approve_all_pending`, `reject_all_pending`, `edit_pending`.

## Memory Types

- **user** — role, preferences, expertise
- **feedback** — corrections, confirmed approaches
- **project** — deadlines, client context, status (auto-expires 90d)
- **reference** — external URLs, dashboards, wikis
- **skill** — patterns, workflows, how-tos

## Testing

```bash
pytest tests/ -v
```

On Windows the system Python is too old; run under 3.12 with the src layout on the path:

```powershell
$env:PYTHONPATH = "src"; py -3.12 -m pytest tests/ -q
```

**Backend test strategy** (no live infra required):
- **chroma is skipped, always** — kept only as a `skipif`-guarded param in the parametrized `store` fixture, never the default or a live test backend. Backend-agnostic tests use a test double (`InMemoryStore` / `FakePineconeIndex`), not a real vector store.
- **Pinecone & FalkorDB run fully offline** via doubles in `tests/test_stores.py`: `FakePineconeIndex` (synchronous in-memory index that also enforces Pinecone's metadata contract — only str/number/bool/list[str], no None/nested) and `_FakeGraph` (Cypher-substring graph double; locks the `score = 1 - distance` similarity inversion). Pure parity tests cover `_where_clause`, `_memory_to_params`↔`_row_to_memory`, `_to_pinecone_filter`, `_meta_to_memory`, and the GraphRAG `_rerank`/`_cosine` helpers.
- **Opt-in live integration:** set `PINECONE_API_KEY` (+ optional `PINECONE_TEST_INDEX`) or `FALKORDB_URL` to run the skip-guarded round-trip tests against real infrastructure.
