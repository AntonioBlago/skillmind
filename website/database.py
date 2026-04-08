"""
SkillMind Blog — MySQL database layer.
PythonAnywhere EU: seooptimierung$skillmind_db
"""

import hashlib
import os
import re
from contextlib import contextmanager
from datetime import datetime

import pymysql
import pymysql.cursors


DB_CONFIG = {
    "host": os.environ.get("MYSQL_HOST", "seooptimierung.mysql.eu.pythonanywhere-services.com"),
    "user": os.environ.get("MYSQL_USER", "seooptimierung"),
    "password": os.environ.get("MYSQL_PASSWORD", ""),
    "database": os.environ.get("MYSQL_DATABASE", "seooptimierung$skillmind_db"),
    "charset": "utf8mb4",
    "cursorclass": pymysql.cursors.DictCursor,
}


@contextmanager
def get_db():
    """Get a database connection (context manager)."""
    conn = pymysql.connect(**DB_CONFIG)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Create blog_posts table if it doesn't exist."""
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS blog_posts (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    title VARCHAR(500) NOT NULL,
                    slug VARCHAR(200) NOT NULL UNIQUE,
                    content MEDIUMTEXT,
                    excerpt TEXT,
                    meta_description VARCHAR(500),
                    featured_image_url VARCHAR(1000),

                    language VARCHAR(5) NOT NULL DEFAULT 'en',
                    translation_of_id INT NULL,

                    status ENUM('draft', 'published', 'archived') DEFAULT 'draft',
                    author VARCHAR(100) DEFAULT 'Antonio Blago',
                    published_at TIMESTAMP NULL,

                    categories JSON,
                    tags JSON,

                    word_count INT DEFAULT 0,
                    reading_time INT DEFAULT 0,
                    content_hash VARCHAR(64),

                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

                    FOREIGN KEY (translation_of_id) REFERENCES blog_posts(id) ON DELETE SET NULL,
                    INDEX idx_slug (slug),
                    INDEX idx_status_lang (status, language),
                    INDEX idx_published (published_at)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """)


# ── CRUD ──────────────────────────────────────────────────────


def slugify(title: str) -> str:
    """Generate URL-safe slug from title."""
    slug = title.lower().strip()
    # German umlauts
    for old, new in [("ae", "ae"), ("oe", "oe"), ("ue", "ue"), ("ss", "ss"),
                     ("\u00e4", "ae"), ("\u00f6", "oe"), ("\u00fc", "ue"), ("\u00df", "ss")]:
        slug = slug.replace(old, new)
    slug = re.sub(r"[^a-z0-9\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug[:200]


def _word_count(html: str) -> int:
    """Count words in HTML content."""
    text = re.sub(r"<[^>]+>", " ", html or "")
    return len(text.split())


def create_post(data: dict) -> int:
    """Create a new blog post. Returns post ID or -1."""
    title = data.get("title", "").strip()
    if not title:
        return -1

    slug = data.get("slug") or slugify(title)
    content = data.get("content", "")
    wc = _word_count(content)

    with get_db() as conn:
        with conn.cursor() as cur:
            # Ensure unique slug
            cur.execute("SELECT id FROM blog_posts WHERE slug = %s", (slug,))
            if cur.fetchone():
                slug = f"{slug}-{int(datetime.now().timestamp())}"

            cur.execute("""
                INSERT INTO blog_posts
                (title, slug, content, excerpt, meta_description, featured_image_url,
                 language, status, author, categories, tags, word_count, reading_time, content_hash)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                title, slug, content,
                data.get("excerpt", ""),
                data.get("meta_description", ""),
                data.get("featured_image_url", ""),
                data.get("language", "en"),
                data.get("status", "draft"),
                data.get("author", "Antonio Blago"),
                pymysql.converters.escape_string(str(data.get("categories", "[]"))),
                pymysql.converters.escape_string(str(data.get("tags", "[]"))),
                wc,
                max(1, wc // 200),
                hashlib.md5(content.encode()).hexdigest(),
            ))
            return cur.lastrowid


def update_post(post_id: int, data: dict) -> bool:
    """Update an existing blog post."""
    sets = []
    vals = []
    for key in ["title", "slug", "content", "excerpt", "meta_description",
                 "featured_image_url", "language", "status", "author"]:
        if key in data:
            sets.append(f"{key} = %s")
            vals.append(data[key])

    if "categories" in data:
        sets.append("categories = %s")
        vals.append(str(data["categories"]))
    if "tags" in data:
        sets.append("tags = %s")
        vals.append(str(data["tags"]))

    if "content" in data:
        wc = _word_count(data["content"])
        sets.append("word_count = %s")
        vals.append(wc)
        sets.append("reading_time = %s")
        vals.append(max(1, wc // 200))
        sets.append("content_hash = %s")
        vals.append(hashlib.md5(data["content"].encode()).hexdigest())

    if not sets:
        return False

    vals.append(post_id)
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(f"UPDATE blog_posts SET {', '.join(sets)} WHERE id = %s", tuple(vals))
            return cur.rowcount > 0


def publish_post(post_id: int) -> bool:
    """Publish a post."""
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE blog_posts SET status='published', published_at=NOW() WHERE id=%s",
                (post_id,)
            )
            return cur.rowcount > 0


def unpublish_post(post_id: int) -> bool:
    """Set post back to draft."""
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE blog_posts SET status='draft' WHERE id=%s", (post_id,))
            return cur.rowcount > 0


def delete_post(post_id: int) -> bool:
    """Soft-delete (archive) a post."""
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE blog_posts SET status='archived' WHERE id=%s", (post_id,))
            return cur.rowcount > 0


def get_post(post_id: int) -> dict | None:
    """Get a single post by ID."""
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM blog_posts WHERE id = %s", (post_id,))
            return cur.fetchone()


def get_post_by_slug(slug: str, language: str = "en") -> dict | None:
    """Get a published post by slug and language."""
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM blog_posts WHERE slug = %s AND language = %s AND status = 'published'",
                (slug, language),
            )
            return cur.fetchone()


def get_posts(language: str = "en", status: str = "published",
              limit: int = 20, offset: int = 0) -> list[dict]:
    """Get paginated list of posts."""
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM blog_posts WHERE language = %s AND status = %s "
                "ORDER BY published_at DESC, created_at DESC LIMIT %s OFFSET %s",
                (language, status, limit, offset),
            )
            return cur.fetchall()


def get_all_posts(limit: int = 100) -> list[dict]:
    """Get all posts (admin view, all languages/statuses)."""
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, title, slug, language, status, author, published_at, "
                "word_count, reading_time, created_at, updated_at "
                "FROM blog_posts WHERE status != 'archived' "
                "ORDER BY updated_at DESC LIMIT %s",
                (limit,),
            )
            return cur.fetchall()


def count_posts(language: str = "en", status: str = "published") -> int:
    """Count posts."""
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) as cnt FROM blog_posts WHERE language = %s AND status = %s",
                (language, status),
            )
            return cur.fetchone()["cnt"]
