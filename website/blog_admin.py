"""
SkillMind Blog Admin — TinyMCE editor + CRUD routes.
Protected by simple password (BLOG_ADMIN_PASSWORD env var).
"""

import functools
import json
import os

from flask import Blueprint, render_template, request, redirect, url_for, jsonify, session, abort

from . import database as db

admin_bp = Blueprint("blog_admin", __name__, url_prefix="/admin/blog")

ADMIN_PASSWORD = os.environ.get("BLOG_ADMIN_PASSWORD", "skillmind2026")


def login_required(f):
    """Simple password-based auth decorator."""
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("blog_admin"):
            return redirect(url_for("blog_admin.login"))
        return f(*args, **kwargs)
    return decorated


@admin_bp.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        if request.form.get("password") == ADMIN_PASSWORD:
            session["blog_admin"] = True
            return redirect(url_for("blog_admin.index"))
        error = "Wrong password"
    return render_template("admin/blog/login.html", error=error)


@admin_bp.route("/logout")
def logout():
    session.pop("blog_admin", None)
    return redirect("/")


@admin_bp.route("/")
@login_required
def index():
    posts = db.get_all_posts()
    return render_template("admin/blog/index.html", posts=posts)


@admin_bp.route("/new")
@login_required
def new():
    return render_template("admin/blog/editor.html", post=None)


@admin_bp.route("/<int:post_id>/edit")
@login_required
def edit(post_id):
    post = db.get_post(post_id)
    if not post:
        abort(404)
    # Parse JSON fields for template
    if isinstance(post.get("categories"), str):
        try:
            post["categories"] = json.loads(post["categories"])
        except (json.JSONDecodeError, TypeError):
            post["categories"] = []
    if isinstance(post.get("tags"), str):
        try:
            post["tags"] = json.loads(post["tags"])
        except (json.JSONDecodeError, TypeError):
            post["tags"] = []
    return render_template("admin/blog/editor.html", post=post)


@admin_bp.route("/save", methods=["POST"])
@admin_bp.route("/<int:post_id>/save", methods=["POST"])
@login_required
def save(post_id=None):
    data = request.get_json() or {}

    # Parse categories/tags from comma-separated strings
    if isinstance(data.get("categories"), str):
        data["categories"] = json.dumps([c.strip() for c in data["categories"].split(",") if c.strip()])
    elif isinstance(data.get("categories"), list):
        data["categories"] = json.dumps(data["categories"])

    if isinstance(data.get("tags"), str):
        data["tags"] = json.dumps([t.strip() for t in data["tags"].split(",") if t.strip()])
    elif isinstance(data.get("tags"), list):
        data["tags"] = json.dumps(data["tags"])

    if post_id:
        success = db.update_post(post_id, data)
        return jsonify({"status": "updated" if success else "error", "id": post_id})
    else:
        new_id = db.create_post(data)
        return jsonify({"status": "created" if new_id > 0 else "error", "id": new_id})


@admin_bp.route("/<int:post_id>/publish", methods=["POST"])
@login_required
def publish(post_id):
    db.publish_post(post_id)
    return jsonify({"status": "published"})


@admin_bp.route("/<int:post_id>/unpublish", methods=["POST"])
@login_required
def unpublish(post_id):
    db.unpublish_post(post_id)
    return jsonify({"status": "draft"})


@admin_bp.route("/<int:post_id>/delete", methods=["POST"])
@login_required
def delete(post_id):
    db.delete_post(post_id)
    return jsonify({"status": "archived"})
