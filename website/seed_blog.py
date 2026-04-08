"""
Seed the first blog article into MySQL.
Run once on PythonAnywhere: python seed_blog.py
"""

import os
import sys

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from database import init_db, create_post, publish_post, get_post_by_slug

# Read the article HTML
article_path = os.path.join(os.path.dirname(__file__), "content", "blog", "en", "obsidian-wiki-ai-second-brain.html")
with open(article_path, encoding="utf-8") as f:
    content_en = f.read()

# Initialize DB
init_db()
print("Database initialized.")

# Check if already seeded
existing = get_post_by_slug("obsidian-wiki-ai-second-brain", "en")
if existing:
    print(f"Article already exists (ID: {existing['id']}). Skipping.")
    sys.exit(0)

# Create EN version
en_id = create_post({
    "title": "How to Build an AI-Powered Obsidian Wiki with SkillMind",
    "slug": "obsidian-wiki-ai-second-brain",
    "content": content_en,
    "excerpt": "Turn your AI assistant's memory into a visual knowledge graph. Export SkillMind memories to Obsidian with wikilinks, topic maps, and color-coded graph groups.",
    "meta_description": "Build an AI second brain with SkillMind and Obsidian. Export vector DB memories as interlinked wiki pages with graph visualization. Karpathy LLM wiki pattern made easy.",
    "featured_image_url": "/static/img/obsidian-knowledge-graph.png",
    "language": "en",
    "author": "Antonio Blago",
    "categories": '["Tutorial", "AI Knowledge Management"]',
    "tags": '["obsidian", "knowledge-graph", "second-brain", "karpathy", "llm-wiki", "mcp", "claude-code", "vector-database", "rag", "personal-wiki"]',
    "status": "draft",
})
print(f"EN article created (ID: {en_id})")

# Publish it
publish_post(en_id)
print("EN article published.")

# Create DE version
de_id = create_post({
    "title": "KI-Wissensgraph mit Obsidian und SkillMind aufbauen",
    "slug": "obsidian-wiki-ai-second-brain",
    "content": content_en,  # Same content for now
    "excerpt": "Verwandle das Gedaechtnis deines KI-Assistenten in einen visuellen Wissensgraphen mit Wikilinks, Topic Maps und farbcodierten Graph-Gruppen.",
    "meta_description": "Baue ein KI-Second-Brain mit SkillMind und Obsidian. Exportiere Vektor-DB-Erinnerungen als verlinkte Wiki-Seiten mit Graph-Visualisierung.",
    "featured_image_url": "/static/img/obsidian-knowledge-graph.png",
    "language": "de",
    "author": "Antonio Blago",
    "categories": '["Tutorial", "KI-Wissensmanagement"]',
    "tags": '["obsidian", "wissensgraph", "second-brain", "karpathy", "llm-wiki", "mcp", "claude-code"]',
    "status": "draft",
})
print(f"DE article created (ID: {de_id})")

publish_post(de_id)
print("DE article published.")

print("\nDone! Blog articles seeded.")
