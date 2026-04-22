<p align="center">
  <img src="docs/assets/logo.png" alt="SkillMind" width="200"/>
</p>

# SkillMind

[![PyPI version](https://badge.fury.io/py/skillmind.svg)](https://pypi.org/project/skillmind/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/skillmind.svg)](https://pypi.org/project/skillmind/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/skillmind.svg)](https://pypi.org/project/skillmind/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MCP Integration](https://img.shields.io/badge/MCP-Integrated-blue.svg)](https://modelcontextprotocol.io)
[![Website](https://img.shields.io/badge/Website-skill--mind.com-orange.svg)](https://www.skill-mind.com/)

**Active Skill Listener & Trainer** — the structured memory layer for AI coding assistants.

Replace flat markdown memory files with a vector-database-backed system that **listens**, **learns**, and **surfaces** only relevant context per conversation.

> 🌐 **[Visit skill-mind.com](https://skill-mind.com/)** — Documentation, setup guides, and live demos.

---

## The Problem

| Current state | SkillMind |
|---|---|
| 15+ flat markdown memory files | One vector DB with semantic search |
| Everything loaded into context | Only relevant memories surfaced |
| Manual save/forget | Auto-listen, auto-classify, auto-expire |
| No deduplication | Automatic dedup + consolidation |
| No protection | Sanitizer redacts API keys, PII before storing |

## Quick Start (3 Commands)

```bash
# 1. Install
pip install skillmind[pinecone]

# 2. Initialize
skillmind init --backend pinecone

# 3. Import your existing Claude Code memories
skillmind import ~/.claude/projects/*/memory/
```

**That's it.** Your memories are now in a vector DB with semantic search.

## How It Works

```
┌────────────┐    ┌────────────┐    ┌─────────────────┐
│  LISTENER   │───▶│  TRAINER    │───▶│  VECTOR STORE    │
│             │    │             │    │                  │
│ • Git hooks │    │ • Classify  │    │ • Chroma (local) │
│ • File watch│    │ • Sanitize  │    │ • Pinecone       │
│ • Convos    │    │ • Dedup     │    │ • Supabase       │
│ • YouTube   │    │ • Merge     │    │ • Qdrant         │
│ • Screen    │    │ • Expire    │    │ • FAISS          │
└────────────┘    └────────────┘    └─────────────────┘
                                            │
                                            ▼
                                    ┌─────────────────┐
                                    │  CLAUDE CODE     │
                                    │  (MCP Server)    │
                                    │  23 tools        │
                                    └─────────────────┘
```

## 5 Vector Store Backends

| Backend | Best for | Requires |
|---------|----------|----------|
| **ChromaDB** | Solo dev, local, default | `pip install skillmind[chroma]` |
| **Pinecone** | Multi-device cloud sync | API key |
| **Supabase** | SQL + vectors, team sharing | Supabase project |
| **Qdrant** | Self-hosted or cloud, great filtering | Qdrant server |
| **FAISS** | Offline, air-gapped, fastest | `pip install skillmind[faiss]` |

All backends implement the same interface — switch anytime with zero data loss.

## 23 MCP Tools for Claude Code

Once installed, Claude Code gets these tools:

### Memory CRUD
| Tool | What it does |
|---|---|
| `remember` | Store memory (auto-classified, sanitized, deduped) |
| `recall` | Semantic search across all memories |
| `forget` | Delete a memory |
| `update_memory` | Edit existing memory |
| `context` | Generate focused context for current file/topic |
| `consolidate` | Merge duplicates, expire stale, cleanup |
| `memory_stats` | Counts by type, backend info |
| `list_memories` | Filter by type/topic |
| `import_markdown_memories` | Bulk import from Claude Code markdown files |

### Video & YouTube Learning
| Tool | What it does |
|---|---|
| `learn_youtube` | Extract knowledge from YouTube video |
| `learn_youtube_channel` | Learn from channel's latest videos |
| `learn_video` | Learn from local video/screen recording |
| `record_screen` | Record screen to MP4 |
| `screenshot` | Capture screenshot |

### Custom Patterns
| Tool | What it does |
|---|---|
| `add_pattern` | Create auto-detection regex (e.g. client names, SEO terms) |
| `list_patterns` | Show all custom patterns |
| `remove_pattern` | Delete a pattern |

### Review Queue
| Tool | What it does |
|---|---|
| `set_review_mode` | Switch: `review` (queue first), `auto` (store directly), `off` |
| `get_review_mode` | Show current mode |
| `review_pending` | Show queued memories waiting for approval |
| `approve_memory` | Approve single memory → store in vector DB |
| `reject_memory` | Reject single memory → discard |
| `approve_all_pending` | Approve all at once |
| `reject_all_pending` | Clear the queue |
| `edit_pending` | Edit content/type/topic before approving |

### MCP Setup (Claude Code settings.json)

```json
{
  "mcpServers": {
    "skillmind": {
      "command": "python",
      "args": ["-m", "skillmind.mcp.server"],
      "env": {
        "PINECONE_API_KEY": "your-key",
        "SKILLMIND_BACKEND": "pinecone"
      }
    }
  }
}
```

## Memory Types

| Type | What | Auto-expires |
|---|---|---|
| **user** | Role, preferences, expertise | No |
| **feedback** | Corrections, confirmed approaches | No |
| **project** | Deadlines, client context, status | 90 days |
| **reference** | External URLs, dashboards, wikis | No |
| **skill** | Patterns, workflows, how-tos | No |

## Obsidian Vault Export (Karpathy Wiki Pattern)

Export your entire memory system as an Obsidian vault with interlinked wiki pages, inspired by [Andrej Karpathy's LLM wiki approach](https://gist.github.com/karpathy/1dd0294ef9567971c1e4348a90d69285).

### Complete Setup Guide

#### Step 1: Install SkillMind

```bash
# Install with your preferred backend + MCP server
pip install skillmind[pinecone,mcp,youtube]

# Or install everything
pip install skillmind[all]
```

#### Step 2: Configure the MCP Server

Add SkillMind to your Claude Code `settings.json`:

```json
{
  "mcpServers": {
    "skillmind": {
      "command": "python",
      "args": ["-m", "skillmind.mcp.server"],
      "env": {
        "PINECONE_API_KEY": "your-pinecone-key",
        "SKILLMIND_BACKEND": "pinecone",
        "ANTHROPIC_API_KEY": "your-anthropic-key"
      }
    }
  }
}
```

Or create a `.env` file in your project root:

```bash
PINECONE_API_KEY=your-pinecone-key
SKILLMIND_BACKEND=pinecone
ANTHROPIC_API_KEY=your-anthropic-key
```

#### Step 3: Import Existing Memories

If you already have Claude Code markdown memories:

```bash
skillmind import
```

This scans `~/.claude/projects/*/memory/*.md`, auto-classifies each memory, deduplicates, and stores them in your vector DB.

#### Step 4: Install Obsidian (Free)

1. Download from [obsidian.md](https://obsidian.md) (Windows/Mac/Linux)
2. Install and open it
3. No account required - works 100% offline

#### Step 5: Export to Obsidian Vault

**Via CLI:**

```bash
# Full export
skillmind export ~/Documents/MyWiki

# Incremental sync (only new memories)
skillmind sync --vault ~/Documents/MyWiki
```

**Via MCP tools (inside Claude Code):**

```
> export all my memories to an Obsidian vault at ~/Documents/MyWiki
```

Claude Code will call `export_obsidian` automatically.

#### Step 6: Open the Vault in Obsidian

1. Open Obsidian
2. Click **"Open folder as vault"**
3. Select the vault folder (e.g. `~/Documents/MyWiki`)
4. Press **Ctrl+G** to open the knowledge graph
5. Graph groups are pre-configured with colors per category

#### Step 7: Keep it in Sync

Every time you add new memories (via `remember`, `learn_youtube`, etc.), sync them:

```bash
skillmind sync
```

Or via MCP: `sync_obsidian`

### What Gets Generated

```
MyWiki/
├── .obsidian/             # Pre-configured: graph groups, bookmarks, plugins
├── CLAUDE.md              # Wiki maintenance instructions for Claude
├── raw/                   # Drop raw source material here (articles, PDFs)
└── wiki/
    ├── index.md           # Master index by category and topic
    ├── log.md             # Operation history
    ├── skills/            # Skill memories (green in graph)
    ├── references/        # Reference memories (blue in graph)
    ├── feedback/          # Feedback memories (teal in graph)
    ├── projects/          # Project memories (orange in graph)
    ├── users/             # User profile memories (purple in graph)
    └── topics/            # Topic MOC pages (light blue in graph)
```

**Each wiki page includes:**
- YAML frontmatter with Obsidian-native `tags`, `created`, `updated` dates
- `[[Wikilinks]]` to related memories across categories
- Topic and category backlinks
- `#hashtags` for quick filtering
- Confidence scores and source metadata

**Graph groups (pre-configured colors):**
| Category | Color | Obsidian Query |
|---|---|---|
| Skills | Green | `path:wiki/skills` |
| References | Blue | `path:wiki/references` |
| Feedback | Teal | `path:wiki/feedback` |
| Projects | Orange | `path:wiki/projects` |
| Users | Purple | `path:wiki/users` |
| Topics (MOC) | Light Blue | `path:wiki/topics` |
| Index pages | Yellow | `tag:#MOC` |

### Optional: YouTube Learning with Proxy

If YouTube blocks direct connections, add [ScraperAPI](https://www.scraperapi.com/pricing?fp_ref=antonio28) proxy support:

```bash
# In .env
VPN_PROXY_API_KEY=your-scraperapi-key
SCRAPER_Vendor=scraperapi
```

Then learn from YouTube videos:

```bash
skillmind learn-youtube "https://youtube.com/watch?v=..."
skillmind learn-youtube-channel "UCxxxxx" --limit 5
```

The extracted knowledge is auto-stored as memories and can be synced to your Obsidian vault.

## Built-in Sanitizer

API keys, emails, phone numbers, IBANs, and PII are **automatically redacted** before storing:

```
Input:  "My Pinecone key is pcsk_2mrRyA_9BSbdn5i..."
Stored: "My Pinecone key is [REDACTED:PINECONE_KEY]"
```

Configurable allowlists, custom patterns, and name anonymization.

## CLI Reference

```bash
skillmind init --backend chroma     # Initialize with backend
skillmind remember "content"        # Store a memory
skillmind recall "query"            # Semantic search
skillmind list                      # List all memories
skillmind list --type feedback      # Filter by type
skillmind forget <id>               # Delete
skillmind import                    # Import Claude Code memories
skillmind consolidate               # Cleanup & deduplicate
skillmind context                   # Generate context for current dir
skillmind stats                     # Show statistics
skillmind serve                     # Start MCP server
```

## Installation

```bash
# Core + one backend
pip install skillmind[chroma]       # Local (default)
pip install skillmind[pinecone]     # Cloud sync
pip install skillmind[supabase]     # SQL + vectors
pip install skillmind[qdrant]       # Self-hosted
pip install skillmind[faiss]        # Offline

# Add-ons
pip install skillmind[youtube]      # YouTube learning
pip install skillmind[video]        # Screen recording + OCR
pip install skillmind[mcp]          # MCP server for Claude Code

# Everything
pip install skillmind[all]
```

## Architecture

```
src/skillmind/
├── models.py          # Memory, QueryFilter, QueryResult (Pydantic)
├── config.py          # SkillMindConfig (YAML-based)
├── embeddings.py      # EmbeddingEngine (sentence-transformers | OpenAI)
├── trainer.py         # Auto-classify, dedup, merge, consolidate
├── sanitizer.py       # Redact API keys, PII before storage
├── listener.py        # GitListener, FileListener, ConversationListener
├── context.py         # ContextGenerator — dynamic context injection
├── migration.py       # Import existing Claude Code markdown memories
├── store/
│   ├── base.py            # Abstract MemoryStore interface
│   ├── chroma_store.py    # ChromaDB backend
│   ├── pinecone_store.py  # Pinecone backend
│   ├── supabase_store.py  # Supabase/pgvector backend
│   ├── qdrant_store.py    # Qdrant backend
│   └── faiss_store.py     # FAISS + JSON backend
├── video/
│   ├── youtube_learner.py     # YouTube transcript extraction
│   ├── video_learner.py       # Local video learning
│   └── screen_recorder.py     # Screen capture
├── mcp/
│   └── server.py      # FastMCP server (14 tools)
└── cli/
    └── main.py        # CLI (click-based)
```

## Complementary MCP Tools

SkillMind works great alongside other MCP servers:

| MCP Server | What it adds | Install |
|---|---|---|
| **[Visibly AI MCP](https://www.antonioblago.com/de/entwickler/mcp)** | SEO skills: keyword research, backlinks, site audit, GSC queries, on-page analysis | Hosted MCP — no install; add HTTP config to `settings.json` |
| **[Notion MCP](https://mcp.notion.com/)** | Read/write Notion pages, databases, todos | Built-in |
| **[Playwright MCP](https://github.com/anthropics/mcp-playwright)** | Browser automation, screenshots, web scraping | Built-in |

### Example: SEO Memory + Skills

```bash
# SkillMind installs locally; Visibly AI is a hosted MCP (no install needed)
pip install skillmind[pinecone,mcp]
```

```json
{
  "mcpServers": {
    "skillmind": {
      "command": "python",
      "args": ["-m", "skillmind.mcp.server"],
      "env": { "SKILLMIND_BACKEND": "pinecone", "PINECONE_API_KEY": "..." }
    },
    "visiblyai": {
      "type": "http",
      "url": "https://mcp.visibly-ai.com/mcp",
      "headers": { "Authorization": "Bearer YOUR_KEY" }
    }
  }
}
```

SkillMind remembers your SEO preferences, client keywords, and audit findings. Visibly AI provides the live SEO data. Together: persistent SEO intelligence.

## Used by

### [peec-ai-skills](https://github.com/AntonioBlago/peec-ai-skills)

A Claude Code skill repository for **Peec AI** (brand-visibility tracking in LLM search) that uses SkillMind as its **cross-project memory layer**. The `skillmind-learner` skill writes causal, evidence-backed patterns to SkillMind after a Peec AI growth-loop outcome is measured, and recalls matching patterns as priors for the growth-agent orchestrator on the next run — so lessons learned on project A inform decisions on project B.

See [`skills/skillmind-learner/SKILL.md`](https://github.com/AntonioBlago/peec-ai-skills/blob/main/skills/skillmind-learner/SKILL.md) for the full integration contract (read vs. write mode, required tags, pattern schema, handoff points from the other 5 Peec skills).

## Contributing

PRs welcome! See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT — see [LICENSE](LICENSE).

---

Built by [Antonio Blago](https://antonioblago.de) | [skill-mind.com](https://www.skill-mind.com)
