"""
SkillMind Website — Flask + Babel multi-language (EN/DE).
Deploy on PythonAnywhere EU.
"""

import os
from pathlib import Path

import yaml
from flask import Flask, render_template, redirect, request, g, url_for, abort
from flask_babel import Babel, gettext as _

app = Flask(__name__)
app.config["SECRET_KEY"] = "skillmind-web-2026"
app.config["BABEL_DEFAULT_LOCALE"] = "en"
app.config["BABEL_SUPPORTED_LOCALES"] = ["en", "de"]
app.config["BABEL_TRANSLATION_DIRECTORIES"] = "translations"

SUPPORTED = app.config["BABEL_SUPPORTED_LOCALES"]


def get_locale():
    """Get locale from URL prefix."""
    lang = g.get("lang", None)
    if lang and lang in SUPPORTED:
        return lang
    return request.accept_languages.best_match(SUPPORTED, "en")


babel = Babel(app, locale_selector=get_locale)


@app.route("/")
def index_redirect():
    lang = request.accept_languages.best_match(SUPPORTED, "en")
    return redirect(f"/{lang}/")


@app.route("/<lang>/")
def index(lang):
    if lang not in SUPPORTED:
        return redirect("/en/")
    g.lang = lang
    return render_template("index.html", lang=lang)


@app.route("/<lang>/docs/")
def docs(lang):
    if lang not in SUPPORTED:
        return redirect("/en/docs/")
    g.lang = lang
    return render_template("docs.html", lang=lang)


@app.route("/<lang>/features/")
def features(lang):
    if lang not in SUPPORTED:
        return redirect("/en/features/")
    g.lang = lang
    return render_template("features.html", lang=lang)


# ── Blog ──────────────────────────────────────────────────────

BLOG_DIR = Path(__file__).parent / "content" / "blog"


def _load_blog_posts(lang: str) -> list[dict]:
    """Load all blog posts for a language, sorted by date desc."""
    posts = []
    blog_path = BLOG_DIR / lang
    if not blog_path.exists():
        return posts
    for f in sorted(blog_path.glob("*.yml"), reverse=True):
        try:
            with open(f, encoding="utf-8") as fh:
                meta = yaml.safe_load(fh)
            if meta and meta.get("status") == "published":
                meta["slug"] = f.stem
                posts.append(meta)
        except Exception:
            continue
    return posts


def _load_blog_post(lang: str, slug: str) -> dict | None:
    """Load a single blog post by slug."""
    meta_path = BLOG_DIR / lang / f"{slug}.yml"
    content_path = BLOG_DIR / lang / f"{slug}.html"
    if not meta_path.exists() or not content_path.exists():
        return None
    try:
        with open(meta_path, encoding="utf-8") as f:
            meta = yaml.safe_load(f)
        with open(content_path, encoding="utf-8") as f:
            meta["content"] = f.read()
        meta["slug"] = slug
        return meta
    except Exception:
        return None


@app.route("/<lang>/blog/")
def blog_index(lang):
    if lang not in SUPPORTED:
        return redirect("/en/blog/")
    g.lang = lang
    posts = _load_blog_posts(lang)
    return render_template("blog_index.html", lang=lang, posts=posts)


@app.route("/<lang>/blog/<slug>/")
def blog_post(lang, slug):
    if lang not in SUPPORTED:
        return redirect(f"/en/blog/{slug}/")
    g.lang = lang
    post = _load_blog_post(lang, slug)
    if not post:
        abort(404)
    return render_template("blog_post.html", lang=lang, post=post)


@app.errorhandler(404)
def not_found(e):
    return render_template("index.html", lang="en"), 404


if __name__ == "__main__":
    app.run(debug=True, port=5000)
