"""
SkillMind Website — Flask + Babel multi-language (EN/DE).
Deploy on PythonAnywhere EU.
"""

from flask import Flask, render_template, redirect, request, g, url_for
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


@app.errorhandler(404)
def not_found(e):
    return render_template("index.html", lang="en"), 404


if __name__ == "__main__":
    app.run(debug=True, port=5000)
