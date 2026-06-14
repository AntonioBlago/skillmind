"""Web page source — fetch URLs and harvest their readable text.

Dependency-free by default: uses ``urllib`` from the stdlib plus a lightweight
HTML→text reduction. For production-grade extraction install ``requests`` +
``readability-lxml`` and pass an extractor, but the built-in path is enough to
feed a "second brain" enrichment loop from arbitrary web pages.
"""

from __future__ import annotations

import html
import re
import urllib.request
from typing import Iterable

from .base import KnowledgeSource, RawDocument

_USER_AGENT = "SkillMind-OKF-Enrichment/1.0 (+https://skill-mind.com)"
_SCRIPT_STYLE_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\n\s*\n\s*\n+")


class WebSource(KnowledgeSource):
    """Yield one RawDocument per fetched URL (best-effort HTML→text)."""

    name = "web"

    def __init__(self, urls: list[str] | str, timeout: float = 20.0):
        self.urls = [urls] if isinstance(urls, str) else list(urls)
        self.timeout = timeout

    def discover(self) -> Iterable[RawDocument]:
        for url in self.urls:
            doc = self._fetch(url)
            if doc is not None:
                yield doc

    def _fetch(self, url: str) -> RawDocument | None:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                charset = resp.headers.get_content_charset() or "utf-8"
                raw = resp.read().decode(charset, errors="replace")
        except Exception:  # network/decoding errors → skip this URL gracefully
            return None

        title_match = _TITLE_RE.search(raw)
        title = html.unescape(_TAG_RE.sub("", title_match.group(1)).strip()) if title_match else url

        text = self._html_to_text(raw)
        if not text.strip():
            return None

        return RawDocument(
            identifier=url,
            content=text,
            title=title or url,
            source_url=url,
            metadata={"source_url": url, "fetched_from": url},
        )

    @staticmethod
    def _html_to_text(raw: str) -> str:
        no_scripts = _SCRIPT_STYLE_RE.sub(" ", raw)
        # Preserve block boundaries as newlines before stripping tags.
        blocks = re.sub(r"(?i)</(p|div|section|article|li|h[1-6]|br)\s*>", "\n", no_scripts)
        text = _TAG_RE.sub("", blocks)
        text = html.unescape(text)
        text = "\n".join(line.strip() for line in text.splitlines())
        text = _WS_RE.sub("\n\n", text)
        return text.strip()
