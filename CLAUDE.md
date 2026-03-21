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
├── listener.py        # GitListener, FileListener, ConversationListener
├── context.py         # ContextGenerator — dynamic context for Claude Code
├── migration.py       # Import existing Claude Code markdown memories
├── store/
│   ├── base.py        # Abstract MemoryStore interface
│   ├── chroma_store.py    # ChromaDB backend (default)
│   ├── pinecone_store.py  # Pinecone backend
│   ├── supabase_store.py  # Supabase/pgvector backend
│   ├── qdrant_store.py    # Qdrant backend
│   └── faiss_store.py     # FAISS + JSON backend
├── mcp/
│   └── server.py      # MCP server (10 tools for Claude Code)
└── cli/
    └── main.py        # CLI (click-based)
```

## Key Commands

```bash
pip install -e ".[chroma]"          # Install with Chroma backend
skillmind init --backend chroma     # Initialize
skillmind import                    # Import existing Claude Code memories
skillmind remember "content"        # Store memory
skillmind recall "query"            # Semantic search
skillmind list                      # List all
skillmind consolidate               # Cleanup
skillmind serve                     # Start MCP server
```

## Store Backends

All 5 implement the same `MemoryStore` interface (add, query, get, update, delete, list_all, count, clear):

| Backend | Best for | Requires |
|---------|----------|----------|
| **chroma** | Solo dev, local, default | `pip install chromadb` |
| **faiss** | Offline, air-gapped, fastest | `pip install faiss-cpu` |
| **qdrant** | Self-hosted or cloud, great filtering | Qdrant server |
| **pinecone** | Multi-device cloud sync | API key |
| **supabase** | SQL + vectors, team sharing | Supabase project |

## MCP Server Tools

10 tools exposed via FastMCP: `remember`, `recall`, `forget`, `update_memory`, `context`, `consolidate`, `memory_stats`, `list_memories`, `import_markdown_memories`.

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
