"""
SkillMind Website — Flask + Babel multi-language (EN/DE).
Blog CMS with TinyMCE + MySQL on PythonAnywhere EU.
"""

import json
import os

from flask import Flask, render_template, redirect, request, g, url_for, abort
from flask_babel import Babel, gettext as _

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "skillmind-web-2026")
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

# ── Blog Admin (TinyMCE CMS) ─────────────────────────────────

from blog_admin import admin_bp  # noqa: E402
app.register_blueprint(admin_bp)

# Initialize DB tables on first request
_db_initialized = False


@app.before_request
def ensure_db():
    global _db_initialized
    if not _db_initialized:
        try:
            from database import init_db
            init_db()
            _db_initialized = True
        except Exception:
            pass  # DB not available (local dev without MySQL)


# ── Static Pages ─────────────────────────────────────────────


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


# ── Blog (MySQL-backed) ──────────────────────────────────────


@app.route("/<lang>/blog/")
def blog_index(lang):
    if lang not in SUPPORTED:
        return redirect("/en/blog/")
    g.lang = lang
    try:
        from database import get_posts
        posts = get_posts(language=lang, status="published")
        # Parse JSON fields for template
        for p in posts:
            for field in ("categories", "tags"):
                if isinstance(p.get(field), str):
                    try:
                        p[field] = json.loads(p[field])
                    except (json.JSONDecodeError, TypeError):
                        p[field] = []
    except Exception:
        posts = []
    return render_template("blog_index.html", lang=lang, posts=posts)


@app.route("/<lang>/blog/<slug>/")
def blog_post(lang, slug):
    if lang not in SUPPORTED:
        return redirect(f"/en/blog/{slug}/")
    g.lang = lang
    try:
        from database import get_post_by_slug
        post = get_post_by_slug(slug, language=lang)
        if not post:
            abort(404)
        # Parse JSON fields
        for field in ("categories", "tags"):
            if isinstance(post.get(field), str):
                try:
                    post[field] = json.loads(post[field])
                except (json.JSONDecodeError, TypeError):
                    post[field] = []
    except Exception:
        abort(404)
    return render_template("blog_post.html", lang=lang, post=post)


@app.errorhandler(404)
def not_found(e):
    return render_template("index.html", lang="en"), 404


if __name__ == "__main__":
    app.run(debug=True, port=5000)
