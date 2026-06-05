"""
SkillMind YouTube Learner — extract knowledge from YouTube videos
and store as structured memories.

Reuses patterns from content-automation project (youtube_to_blog.py, podcast_to_blog.py)
but focused on knowledge extraction, not blog generation.

Usage:
    learner = YouTubeLearner(trainer)
    memories = learner.learn("https://www.youtube.com/watch?v=VIDEO_ID")
    memories = learner.learn_channel("CHANNEL_ID", max_videos=5)
    memories = learner.learn_playlist("PLAYLIST_ID")
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from typing import Any, Callable

from ..models import Memory, MemorySource, MemoryType
from ..trainer import Trainer


class YouTubeLearner:
    """
    Extract knowledge from YouTube videos and convert to SkillMind memories.

    Supports:
    - Single video learning (transcript → structured knowledge)
    - Channel batch learning (latest N videos)
    - Playlist learning
    - Podcast episodes (long-form, chunked)
    - ScraperAPI proxy support (set VPN_PROXY_API_KEY + SCRAPER_Vendor=scraperapi)
    """

    def __init__(
        self,
        trainer: Trainer,
        language: str = "de",
        anthropic_api_key: str | None = None,
        claude_model: str = "claude-sonnet-4-6",
    ):
        self.trainer = trainer
        self.language = language
        self.api_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.claude_model = claude_model

        # Duration (seconds) derived from transcript timing — used as a fallback
        # when yt-dlp metadata is blocked and only oEmbed (no duration) is available.
        self._transcript_duration: int = 0

        # Timed transcript segments [(start_seconds, text)], captured during
        # transcript fetch. Used to split the transcript along chapter boundaries
        # for batched, order-preserving knowledge extraction.
        self._transcript_segments: list[tuple[float, str]] = []

        # Proxy config (ScraperAPI or generic)
        self._scraper_api_key = self._get_scraper_api_key()
        self._proxy_url = self._build_proxy_url()

    def _get_scraper_api_key(self) -> str | None:
        """Get ScraperAPI key if configured."""
        vendor = os.environ.get("SCRAPER_Vendor", "").lower()
        api_key = os.environ.get("VPN_PROXY_API_KEY", "")
        if vendor == "scraperapi" and api_key:
            return api_key
        return None

    def _build_proxy_url(self) -> str | None:
        """Build proxy URL from environment variables."""
        if self._scraper_api_key:
            return f"http://scraperapi:{self._scraper_api_key}@proxy-server.scraperapi.com:8001"
        return os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY") or None

    def _scraper_fetch(self, url: str, timeout: int = 30) -> str:
        """Fetch a URL via ScraperAPI direct URL mode (avoids SSL issues)."""
        import requests

        if self._scraper_api_key:
            api_url = f"http://api.scraperapi.com?api_key={self._scraper_api_key}&url={url}"
            resp = requests.get(api_url, timeout=timeout)
        elif self._proxy_url:
            proxies = {"http": self._proxy_url, "https": self._proxy_url}
            resp = requests.get(url, timeout=timeout, proxies=proxies)
        else:
            resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return resp.text

    def _get_ytdlp_proxy_args(self) -> list[str]:
        """Get yt-dlp --proxy arguments (with --no-check-certificates for ScraperAPI)."""
        if not self._proxy_url:
            return []
        args = ["--proxy", self._proxy_url]
        if self._scraper_api_key:
            args.append("--no-check-certificates")
        return args

    # ── Async wrappers (prevent blocking MCP event loop) ─────────

    async def learn_async(
        self,
        video_url: str,
        force_topic: str | None = None,
        tags: list[str] | None = None,
        progress_cb: Callable[[int, int, str], None] | None = None,
    ) -> list[Memory]:
        """Non-blocking version of learn() for use in async MCP tools."""
        return await asyncio.to_thread(self.learn, video_url, force_topic, tags, progress_cb)

    async def learn_channel_async(
        self,
        channel_id: str,
        max_videos: int = 5,
        force_topic: str | None = None,
    ) -> list[Memory]:
        """Non-blocking version of learn_channel() for use in async MCP tools."""
        return await asyncio.to_thread(self.learn_channel, channel_id, max_videos, force_topic)

    # ── Single Video ──────────────────────────────────────────────

    def learn(
        self,
        video_url: str,
        force_topic: str | None = None,
        tags: list[str] | None = None,
        progress_cb: Callable[[int, int, str], None] | None = None,
    ) -> list[Memory]:
        """
        Learn from a single YouTube video.

        Steps:
        1. Extract video ID
        2. Fetch metadata (title, author, duration, description)
        3. Fetch transcript
        4. Extract structured knowledge via Claude API
        5. Store as memories (skill + reference)

        progress_cb: optional callback(step, total, message) invoked before each
        phase. Used by async MCP tools to emit live MCP progress notifications.
        The callback runs in the worker thread, so keep it cheap and non-blocking.
        """
        total_steps = 4

        def _emit(step: int, msg: str) -> None:
            print(f"[SkillMind] {step}/{total_steps} {msg}", flush=True)
            if progress_cb:
                try:
                    progress_cb(step, total_steps, msg)
                except Exception:
                    pass

        video_id = self._extract_video_id(video_url)
        _emit(1, f"Lade Metadaten für {video_id}…")
        metadata = self._get_metadata(video_id)
        title = metadata.get("title", video_id)
        _emit(2, f"Hole Transkript: {title[:60]}…")
        transcript = self._get_transcript(video_id)

        # If yt-dlp metadata was blocked (duration 0) but the transcript carried
        # timing info, recover the duration from the last transcript timestamp.
        if not metadata.get("duration") and self._transcript_duration:
            metadata["duration"] = self._transcript_duration

        if not transcript:
            print(f"[SkillMind] No transcript available, storing reference only", flush=True)
            # Store just the reference
            mem = self.trainer.learn(
                content=f"YouTube video: {metadata.get('title', video_id)}\nURL: {metadata.get('url', video_url)}\nNo transcript available.",
                title=f"Video: {metadata.get('title', video_id)[:60]}",
                source=MemorySource.SKILL_SEEKERS,
                force_type=MemoryType.REFERENCE,
                force_topic=force_topic or "youtube",
                tags=tags or ["youtube", "video"],
            )
            return [mem] if mem else []

        # Extract knowledge (batched; per-chapter when the video has chapters)
        chapters = metadata.get("chapters") or []
        _emit(3, f"Extrahiere Wissen via Claude ({len(transcript)} Zeichen, "
                 f"{len(chapters)} Kapitel, kann 1–2 Min dauern)…")
        knowledge, chapter_knows = self._extract_with_chapters(transcript, metadata)
        # Keep the chapters on the knowledge dict for the markdown render.
        knowledge["chapters"] = chapter_knows
        memories: list[Memory] = []

        _emit(4, f"Speichere Memories (FalkorDB)… {len(chapter_knows)} Kapitel")
        # Store main knowledge as a skill memory
        subtopics = knowledge.get("subtopics", [])
        base_title = knowledge.get("title", metadata.get("title", "YouTube Video"))
        mem = self.trainer.learn(
            content=knowledge["summary"],
            title=base_title[:80],
            source=MemorySource.SKILL_SEEKERS,
            force_type=MemoryType.SKILL,
            force_topic=force_topic or knowledge.get("topic") or None,
            tags=(tags or []) + knowledge.get("tags", []) + subtopics + ["youtube"],
            metadata={
                "video_id": video_id,
                "video_url": metadata.get("url", ""),
                "duration": metadata.get("duration", 0),
                "author": metadata.get("author", ""),
                "subtopics": subtopics,
                "key_takeaways": knowledge.get("key_takeaways", []),
                "chapter_count": len(chapter_knows),
            },
        )
        if mem:
            memories.append(mem)

        # Store one memory per chapter, preserving order via a :NEXT graph chain.
        chapter_mem_ids: list[str] = []
        for ck in chapter_knows:
            c_idx = int(ck.get("chapter_index", 0))
            c_title = ck.get("chapter_title") or f"Kapitel {c_idx + 1}"
            c_summary = (ck.get("summary") or "").strip()
            if not c_summary:
                continue
            cmem = self.trainer.learn(
                content=c_summary,
                title=f"{base_title} - Kap. {c_idx + 1}: {c_title}"[:80],
                source=MemorySource.SKILL_SEEKERS,
                force_type=MemoryType.SKILL,
                force_topic=force_topic or ck.get("topic") or knowledge.get("topic") or None,
                tags=(tags or []) + (ck.get("tags") or []) + ["youtube", "chapter"],
                metadata={
                    "video_id": video_id,
                    "video_url": metadata.get("url", ""),
                    "chapter_index": c_idx,
                    "chapter_start": ck.get("chapter_start", 0),
                    "chapter_title": c_title,
                    "source_video": video_id,
                    "key_takeaways": ck.get("key_takeaways", []),
                },
            )
            if cmem:
                memories.append(cmem)
                chapter_mem_ids.append(cmem.id)

        # Chain chapter memories in order (FalkorDB :NEXT edges) if supported.
        if len(chapter_mem_ids) >= 2:
            store = getattr(self.trainer, "store", None)
            if store is not None and hasattr(store, "link_sequence"):
                try:
                    n = store.link_sequence(chapter_mem_ids, rel="NEXT", group_key=video_id)
                    print(f"[SkillMind] Kapitel-Reihenfolge verkettet: {n} :NEXT-Kanten", flush=True)
                except Exception as e:
                    print(f"[SkillMind] :NEXT chaining übersprungen: {e}", flush=True)

        # Store the overall key takeaways as separate memories (no chapters case
        # keeps the original behaviour; with chapters they complement the chain).
        for takeaway in knowledge.get("key_takeaways", [])[:5]:
            mem = self.trainer.learn(
                content=takeaway,
                title=f"Takeaway: {takeaway[:60]}",
                source=MemorySource.SKILL_SEEKERS,
                force_type=MemoryType.SKILL,
                force_topic=force_topic or knowledge.get("topic") or None,
                tags=["youtube", "takeaway"],
                metadata={"source_video": video_id},
            )
            if mem:
                memories.append(mem)

        # Print markdown summary
        self._print_markdown(metadata, knowledge)

        # Store video as reference
        ref_content = (
            f"YouTube: {metadata.get('title', '')}\n"
            f"URL: {metadata.get('url', '')}\n"
            f"Author: {metadata.get('author', '')}\n"
            f"Duration: {metadata.get('duration', 0) // 60} min"
        )
        mem = self.trainer.learn(
            content=ref_content,
            title=f"Video: {metadata.get('title', video_id)[:60]}",
            source=MemorySource.SKILL_SEEKERS,
            force_type=MemoryType.REFERENCE,
            force_topic=force_topic or knowledge.get("topic") or None,
            tags=["youtube", "video_reference"],
        )
        if mem:
            memories.append(mem)

        return memories

    # ── Channel Batch ─────────────────────────────────────────────

    def learn_channel(
        self,
        channel_id: str,
        max_videos: int = 5,
        force_topic: str | None = None,
    ) -> list[Memory]:
        """Learn from the latest N videos of a YouTube channel."""
        print(f"[SkillMind] Fetching channel video list ({channel_id})...", flush=True)
        videos = self._get_channel_videos(channel_id, max_videos)
        print(f"[SkillMind] Found {len(videos)} videos to process", flush=True)
        all_memories: list[Memory] = []

        for i, video in enumerate(videos, 1):
            try:
                print(f"\n[SkillMind] === Video {i}/{len(videos)}: {video.get('title', video['url'])[:60]} ===", flush=True)
                memories = self.learn(
                    video["url"],
                    force_topic=force_topic,
                    tags=["channel:" + channel_id],
                )
                all_memories.extend(memories)
            except Exception as e:
                print(f"[SkillMind] FAILED: {video.get('title', video['url'])}: {e}", flush=True)
                continue

        return all_memories

    # ── Playlist ──────────────────────────────────────────────────

    def learn_playlist(
        self,
        playlist_url: str,
        max_videos: int = 10,
        force_topic: str | None = None,
    ) -> list[Memory]:
        """Learn from videos in a YouTube playlist."""
        videos = self._get_playlist_videos(playlist_url, max_videos)
        all_memories: list[Memory] = []

        for video in videos:
            try:
                memories = self.learn(video["url"], force_topic=force_topic)
                all_memories.extend(memories)
            except Exception as e:
                print(f"Failed: {e}")
                continue

        return all_memories

    def _extract_with_chapters(
        self, transcript: str, metadata: dict
    ) -> tuple[dict, list[dict]]:
        """Return (overall_knowledge, ordered_chapter_knowledge).

        When the video has chapters AND timed transcript segments are available,
        each chapter is summarized separately (order preserved) and the overall
        view is synthesized from the chapter summaries. Otherwise falls back to
        whole-transcript batched extraction with no per-chapter knowledge.
        """
        chapters = metadata.get("chapters") or []
        sections = (
            self._split_transcript_by_chapters(chapters, self._transcript_segments)
            if chapters else []
        )
        if not self.api_key or len(sections) < 2:
            # No usable chapters → single overall knowledge, no chapter memories.
            return self._extract_knowledge(transcript, metadata), []

        import anthropic

        client = anthropic.Anthropic(api_key=self.api_key)
        chapter_knows = self._extract_chapters(client, sections, metadata)
        overall = self._synthesize_overall(client, chapter_knows, metadata)
        return overall, chapter_knows

    # ── Markdown Output ───────────────────────────────────────────

    @staticmethod
    def format_markdown(metadata: dict, knowledge: dict) -> str:
        """Format learning result as a readable markdown string."""
        title = knowledge.get("title") or metadata.get("title", "YouTube Video")
        author = metadata.get("author", "Unknown")
        duration = metadata.get("duration", 0) // 60
        url = metadata.get("url", "")
        topic = knowledge.get("topic", "")
        subtopics = knowledge.get("subtopics", [])
        tags = knowledge.get("tags", [])
        summary = knowledge.get("summary", "")
        takeaways = knowledge.get("key_takeaways", [])

        lines = [
            f"# {title}",
            f"**Author:** {author} | **Duration:** {duration} min | **Topic:** {topic}",
        ]
        if url:
            lines.append(f"**URL:** {url}")
        if subtopics:
            lines.append(f"**Subtopics:** {', '.join(str(s) for s in subtopics)}")
        if tags:
            lines.append(f"**Tags:** {', '.join(str(t) for t in tags)}")
        lines.append("")

        if takeaways:
            lines.append("## Key Takeaways")
            for i, t in enumerate(takeaways, 1):
                lines.append(f"{i}. {t}")
            lines.append("")

        chapters = knowledge.get("chapters") or []
        if chapters:
            lines.append(f"## Kapitel ({len(chapters)}) - in Reihenfolge")
            for ck in chapters:
                idx = int(ck.get("chapter_index", 0)) + 1
                start = float(ck.get("chapter_start", 0) or 0)
                ts = f"{int(start // 60):02d}:{int(start % 60):02d}"
                lines.append(f"{idx}. [{ts}] {ck.get('chapter_title', '')}")
            lines.append("")

        if summary:
            lines.append("## Summary")
            lines.append(summary.strip())
            lines.append("")

        return "\n".join(lines)

    def _print_markdown(self, metadata: dict, knowledge: dict) -> None:
        """Print markdown summary to stdout."""
        print(f"\n{'=' * 60}", flush=True)
        print(self.format_markdown(metadata, knowledge), flush=True)
        print(f"{'=' * 60}\n", flush=True)

    # ── Transcript Extraction ─────────────────────────────────────

    def _get_transcript(self, video_id: str) -> str:
        """Fetch transcript, trying youtube-transcript-api first, then yt-dlp.

        Each method has explicit timeouts to prevent indefinite hangs.
        """
        # Reset timed segments so repeated calls don't accumulate stale timing.
        self._transcript_segments = []

        # Method 1: youtube-transcript-api with timeout wrapper
        try:
            transcript = self._get_transcript_api(video_id)
            if transcript:
                return transcript
        except Exception:
            pass

        # Method 2: yt-dlp with proxy (better for long videos, auto-subs)
        try:
            return self._get_transcript_ytdlp(video_id)
        except Exception:
            pass

        return ""

    def _get_transcript_api(self, video_id: str, timeout: int = 15) -> str:
        """Fetch transcript via youtube-transcript-api with timeout protection.

        Runs the blocking API call in a daemon thread so the main thread
        can abandon it after `timeout` seconds without waiting for cleanup.
        """
        import threading

        from youtube_transcript_api import YouTubeTranscriptApi

        # GenericProxyConfig causes SSL errors with ScraperAPI,
        # so we only use it for non-ScraperAPI proxies
        if self._proxy_url and not self._scraper_api_key:
            from youtube_transcript_api.proxies import GenericProxyConfig
            ytt = YouTubeTranscriptApi(proxy_config=GenericProxyConfig(
                http_url=self._proxy_url,
                https_url=self._proxy_url,
            ))
        else:
            ytt = YouTubeTranscriptApi()

        result_container: list[str] = []
        error_container: list[Exception] = []

        def _fetch() -> None:
            try:
                try:
                    entries = ytt.fetch(video_id, languages=[self.language, "en", "de"])
                except Exception:
                    transcript_list = ytt.list(video_id)
                    first = next(iter(transcript_list))
                    entries = first.fetch()
                result_container.append(" ".join(entry.text for entry in entries))
                try:
                    segs: list[tuple[float, str]] = []
                    for entry in entries:
                        txt = (getattr(entry, "text", "") or "").strip()
                        if txt:
                            segs.append((float(getattr(entry, "start", 0) or 0), txt))
                    self._transcript_segments = segs
                except Exception:
                    pass
                try:
                    last = entries[-1]
                    self._transcript_duration = int(getattr(last, "start", 0) + getattr(last, "duration", 0))
                except Exception:
                    pass
            except Exception as exc:
                error_container.append(exc)

        t = threading.Thread(target=_fetch, daemon=True)
        t.start()
        t.join(timeout=timeout)

        if t.is_alive():
            # Thread still running — abandon it (daemon thread dies with process)
            raise TimeoutError(
                f"youtube-transcript-api timed out after {timeout}s for {video_id}"
            )
        if error_container:
            raise error_container[0]
        if result_container:
            return result_container[0]
        return ""

    def _get_transcript_ytdlp(self, video_id: str) -> str:
        """Fetch transcript via yt-dlp subtitles."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            url = f"https://www.youtube.com/watch?v={video_id}"
            cmd = [
                sys.executable, "-m", "yt_dlp",
                *self._get_ytdlp_proxy_args(),
                "--write-auto-sub",
                "--sub-lang", f"{self.language},en",
                "--sub-format", "json3",
                "--skip-download",
                "--output", os.path.join(tmpdir, f"sub_{video_id}"),
                url,
            ]
            self._run_subprocess_safe(cmd, timeout=60)

            for lang in [self.language, "en"]:
                sub_path = os.path.join(tmpdir, f"sub_{video_id}.{lang}.json3")
                if os.path.exists(sub_path):
                    with open(sub_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    texts = []
                    segments: list[tuple[float, str]] = []
                    last_ms = 0
                    for event in data.get("events", []):
                        ev_start_ms = event.get("tStartMs", 0)
                        ev_end = ev_start_ms + event.get("dDurationMs", 0)
                        if ev_end > last_ms:
                            last_ms = ev_end
                        ev_parts = []
                        for seg in event.get("segs", []):
                            text = seg.get("utf8", "").strip()
                            if text and text != "\n":
                                texts.append(text)
                                ev_parts.append(text)
                        if ev_parts:
                            segments.append((ev_start_ms / 1000.0, " ".join(ev_parts)))
                    if last_ms:
                        self._transcript_duration = int(last_ms / 1000)
                    self._transcript_segments = segments
                    return " ".join(texts)

        return ""

    # ── Metadata ──────────────────────────────────────────────────

    def _run_subprocess_safe(self, cmd: list[str], timeout: int = 30) -> subprocess.CompletedProcess:
        """Run a subprocess with timeout, ensuring the process is killed on timeout."""
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
            return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
            raise

    def _get_metadata(self, video_id: str) -> dict:
        """Get video metadata via yt-dlp (rich) or oEmbed (basic fallback)."""
        # Try yt-dlp first (richer metadata), with proxy if configured
        try:
            cmd = [
                sys.executable, "-m", "yt_dlp",
                *self._get_ytdlp_proxy_args(),
                "--dump-json", "--skip-download",
                f"https://www.youtube.com/watch?v={video_id}",
            ]
            result = self._run_subprocess_safe(cmd, timeout=30)
            if result.returncode == 0:
                data = json.loads(result.stdout)
                return {
                    "title": data.get("title", ""),
                    "author": data.get("uploader", ""),
                    "description": data.get("description", "")[:500],
                    "duration": data.get("duration", 0) or 0,
                    "tags": data.get("tags", [])[:10],
                    "chapters": self._normalize_chapters(data.get("chapters")),
                    "video_id": video_id,
                    "url": f"https://www.youtube.com/watch?v={video_id}",
                }
        except Exception:
            pass

        # Fallback: oEmbed via ScraperAPI direct URL mode
        try:
            oembed_url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
            data = json.loads(self._scraper_fetch(oembed_url, timeout=15))
            return {
                "title": data.get("title", ""),
                "author": data.get("author_name", ""),
                "description": "",
                "duration": 0,
                "tags": [],
                "chapters": [],
                "video_id": video_id,
                "url": f"https://www.youtube.com/watch?v={video_id}",
            }
        except Exception:
            return {
                "title": "", "author": "", "description": "",
                "duration": 0, "tags": [], "chapters": [], "video_id": video_id,
                "url": f"https://www.youtube.com/watch?v={video_id}",
            }

    # ── Knowledge Extraction ──────────────────────────────────────

    def _extract_knowledge(self, transcript: str, metadata: dict) -> dict:
        """Extract structured knowledge from a (whole) transcript via Claude.

        Batches the full transcript instead of truncating: the text is split
        into char-bounded windows, each window is summarized, then a final
        synthesis call merges the window summaries into one knowledge object.
        For a normal-length video this is a single window (one extract call)
        plus one synthesis call.
        """
        if not self.api_key:
            # No API key — return raw transcript as knowledge. topic stays empty
            # so the trainer auto-detects it from the content instead of forcing
            # the platform name ("youtube") as the subject.
            return {
                "title": metadata.get("title", "YouTube Video"),
                "summary": transcript[:2000],
                "key_takeaways": [],
                "topic": "",
                "subtopics": [],
                "tags": metadata.get("tags", []),
            }

        import anthropic

        client = anthropic.Anthropic(api_key=self.api_key)

        windows = self._window_text(transcript, max_chars=28000)
        if len(windows) == 1:
            return self._llm_extract(client, windows[0], metadata)

        # Long transcript: extract each window, then synthesize one overall view.
        partials = []
        for i, win in enumerate(windows, 1):
            print(f"[SkillMind] Batch {i}/{len(windows)} ({len(win)} Zeichen)…", flush=True)
            partials.append(self._llm_extract(client, win, metadata))
        return self._synthesize_overall(client, partials, metadata)

    # ── LLM helpers (shared by whole-transcript and per-chapter paths) ──

    def _llm_extract(self, client: Any, content: str, metadata: dict,
                     section_title: str | None = None) -> dict:
        """One Claude call → one knowledge dict for the given text block."""
        scope = (
            f"Dieser Abschnitt ist das Kapitel \"{section_title}\" des Videos."
            if section_title else
            "Dies ist (ein Teil des) Video-Transkripts."
        )
        prompt = f"""Extrahiere das Kernwissen aus diesem YouTube-Video-Transkript.
Fokussiere dich auf wiederverwendbare Erkenntnisse, Workflows, Tipps und Fakten.
{scope}

## Video-Infos
- Titel: {metadata.get('title', 'Unbekannt')}
- Autor: {metadata.get('author', 'Unbekannt')}
- Dauer: {metadata.get('duration', 0) // 60} Minuten

## Transkript
{content}

## Aufgabe
Extrahiere das Wissen in folgendem YAML-Format (zwischen --- Markern):

---
title: "Praegnanter Wissenstitel"
topic: "inhaltliches Hauptthema (z.B. 'large-language-models', NICHT die Quelle 'youtube')"
subtopics: [unterthema1, unterthema2, unterthema3]
tags: [tag1, tag2, tag3, tag4, tag5]
summary: |
  Ausfuehrliche Zusammenfassung des Kernwissens (500-1000 Woerter).
  Strukturiert mit Abschnitten. Fokus auf wiederverwendbare Erkenntnisse.
key_takeaways:
  - "Erste wichtige Erkenntnis"
  - "Zweite wichtige Erkenntnis"
  - "Dritte wichtige Erkenntnis"
  - "Vierte wichtige Erkenntnis"
  - "Fuenfte wichtige Erkenntnis"
---

Wichtig: 'topic' ist das inhaltliche Thema des Videos, nicht die Plattform.
'subtopics' sind 3-6 feinere Unterthemen. 'tags' sind kurze Schlagworte (kebab-case).
Antworte NUR mit dem YAML-Block."""

        response = client.messages.create(
            model=self.claude_model,
            max_tokens=8192,
            messages=[{"role": "user", "content": prompt}],
            timeout=120.0,
        )
        return self._parse_knowledge_yaml(response.content[0].text, metadata)

    def _synthesize_overall(self, client: Any, partials: list[dict], metadata: dict) -> dict:
        """Merge per-window/chapter knowledge dicts into one overall knowledge."""
        if not partials:
            return self._parse_knowledge_yaml("", metadata)
        if len(partials) == 1:
            return partials[0]

        blocks = []
        for i, p in enumerate(partials, 1):
            tks = "\n".join(f"  - {t}" for t in (p.get("key_takeaways") or []))
            blocks.append(
                f"### Teil {i}: {p.get('title','')}\n"
                f"{(p.get('summary') or '')[:1500]}\n"
                f"Takeaways:\n{tks}"
            )
        joined = "\n\n".join(blocks)
        prompt = f"""Fasse die folgenden Teil-Zusammenfassungen eines YouTube-Videos
zu EINER kohaerenten Gesamt-Wissensbasis zusammen.

## Video-Infos
- Titel: {metadata.get('title', 'Unbekannt')}
- Autor: {metadata.get('author', 'Unbekannt')}

## Teil-Zusammenfassungen
{joined}

## Aufgabe
Erzeuge eine konsolidierte Gesamtsicht im YAML-Format (zwischen --- Markern):

---
title: "Praegnanter Gesamttitel"
topic: "inhaltliches Hauptthema (NICHT 'youtube')"
subtopics: [unterthema1, unterthema2, unterthema3]
tags: [tag1, tag2, tag3, tag4, tag5]
summary: |
  Konsolidierte Gesamtzusammenfassung (600-1000 Woerter).
key_takeaways:
  - "Wichtigste Gesamt-Erkenntnis 1"
  - "Wichtigste Gesamt-Erkenntnis 2"
  - "Wichtigste Gesamt-Erkenntnis 3"
  - "Wichtigste Gesamt-Erkenntnis 4"
  - "Wichtigste Gesamt-Erkenntnis 5"
---

Antworte NUR mit dem YAML-Block."""
        response = client.messages.create(
            model=self.claude_model,
            max_tokens=8192,
            messages=[{"role": "user", "content": prompt}],
            timeout=120.0,
        )
        return self._parse_knowledge_yaml(response.content[0].text, metadata)

    # ── Chapters & batching ───────────────────────────────────────

    @staticmethod
    def _normalize_chapters(raw: Any) -> list[dict]:
        """Normalize yt-dlp chapters to [{title, start, end}] with floats."""
        if not raw or not isinstance(raw, list):
            return []
        out: list[dict] = []
        for i, ch in enumerate(raw):
            if not isinstance(ch, dict):
                continue
            try:
                start = float(ch.get("start_time", 0) or 0)
            except (TypeError, ValueError):
                start = 0.0
            end_raw = ch.get("end_time")
            try:
                end = float(end_raw) if end_raw is not None else None
            except (TypeError, ValueError):
                end = None
            title = str(ch.get("title") or f"Kapitel {i + 1}").strip()
            out.append({"title": title, "start": start, "end": end})
        out.sort(key=lambda c: c["start"])
        return out

    def _split_transcript_by_chapters(
        self, chapters: list[dict], segments: list[tuple[float, str]]
    ) -> list[dict]:
        """Assign timed transcript segments to chapters by start time.

        Returns [{index, title, start, end, text}] for chapters that received
        text. Needs timed segments; without them chapter splitting is impossible.
        """
        if not chapters or not segments:
            return []
        segs = sorted(segments, key=lambda s: s[0])
        result: list[dict] = []
        for idx, ch in enumerate(chapters):
            start = ch["start"]
            # End = explicit end, else next chapter start, else +inf.
            end = ch.get("end")
            if end is None:
                end = chapters[idx + 1]["start"] if idx + 1 < len(chapters) else float("inf")
            parts = [t for (s, t) in segs if start <= s < end]
            text = " ".join(parts).strip()
            if text:
                result.append({
                    "index": idx,
                    "title": ch["title"],
                    "start": start,
                    "end": None if end == float("inf") else end,
                    "text": text,
                })
        return result

    @staticmethod
    def _window_text(text: str, max_chars: int = 28000) -> list[str]:
        """Split text into <=max_chars windows, breaking on whitespace."""
        text = text or ""
        if len(text) <= max_chars:
            return [text] if text else [""]
        windows: list[str] = []
        i = 0
        n = len(text)
        while i < n:
            end = min(i + max_chars, n)
            if end < n:
                brk = text.rfind(" ", i + int(max_chars * 0.6), end)
                if brk != -1:
                    end = brk
            windows.append(text[i:end].strip())
            i = end
        return [w for w in windows if w]

    def _extract_chapters(
        self, client: Any, chapter_sections: list[dict], metadata: dict
    ) -> list[dict]:
        """Extract knowledge per chapter (one Claude call each), order preserved.

        Long chapters are internally windowed+synthesized. Each returned dict is
        the chapter knowledge plus its index/title/start for memory metadata.
        """
        out: list[dict] = []
        total = len(chapter_sections)
        for sec in chapter_sections:
            print(f"[SkillMind] Kapitel {sec['index'] + 1}/{total}: {sec['title'][:50]}…", flush=True)
            windows = self._window_text(sec["text"], max_chars=28000)
            if len(windows) == 1:
                know = self._llm_extract(client, windows[0], metadata, section_title=sec["title"])
            else:
                partials = [
                    self._llm_extract(client, w, metadata, section_title=sec["title"])
                    for w in windows
                ]
                know = self._synthesize_overall(client, partials, metadata)
            know["chapter_index"] = sec["index"]
            know["chapter_title"] = sec["title"]
            know["chapter_start"] = sec["start"]
            out.append(know)
        return out

    def _parse_knowledge_yaml(self, text: str, metadata: dict) -> dict:
        """Parse YAML knowledge extraction response.

        Robust against markdown code fences and a missing closing ``---`` marker
        (which happens when a long response is truncated by max_tokens) — in that
        case we parse everything after the opening ``---``.
        """
        import yaml

        text = text.strip()
        # Strip markdown code fences if the model wrapped the block.
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\s*\n", "", text)
            text = re.sub(r"\n```\s*$", "", text).strip()

        # Prefer a fully delimited block; fall back to "from first --- onwards"
        # so a truncated (unterminated) YAML block is still usable.
        fm_match = re.match(r'^---\s*\n(.*?)\n---', text, re.DOTALL)
        if not fm_match:
            fm_match = re.match(r'^---\s*\n(.*)', text, re.DOTALL)
        if fm_match:
            try:
                data = yaml.safe_load(fm_match.group(1))
                return {
                    "title": data.get("title", metadata.get("title", "")),
                    "summary": data.get("summary", ""),
                    "key_takeaways": data.get("key_takeaways", []),
                    "topic": data.get("topic", "") or "",
                    "subtopics": data.get("subtopics", []) or [],
                    "tags": data.get("tags", []) or [],
                }
            except yaml.YAMLError:
                pass

        # Fallback — leave topic empty so the trainer auto-detects from content.
        return {
            "title": metadata.get("title", "YouTube Video"),
            "summary": text[:2000],
            "key_takeaways": [],
            "topic": "",
            "subtopics": [],
            "tags": metadata.get("tags", []),
        }

    # ── Channel/Playlist Helpers ──────────────────────────────────

    def _get_channel_videos(self, channel_id: str, max_results: int = 5) -> list[dict]:
        """Fetch latest videos from channel via RSS."""
        import xml.etree.ElementTree as ET

        try:
            rss_url = f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
            rss_text = self._scraper_fetch(rss_url, timeout=15)

            root = ET.fromstring(rss_text)
            ns = {
                "atom": "http://www.w3.org/2005/Atom",
                "yt": "http://www.youtube.com/xml/schemas/2015",
            }

            videos = []
            for entry in root.findall("atom:entry", ns)[:max_results]:
                video_id = entry.find("yt:videoId", ns).text
                title = entry.find("atom:title", ns).text
                videos.append({
                    "video_id": video_id,
                    "title": title,
                    "url": f"https://www.youtube.com/watch?v={video_id}",
                })
            return videos
        except Exception:
            return []

    def _get_playlist_videos(self, playlist_url: str, max_results: int = 10) -> list[dict]:
        """Fetch videos from a playlist via yt-dlp."""
        try:
            cmd = [
                sys.executable, "-m", "yt_dlp",
                *self._get_ytdlp_proxy_args(),
                "--dump-json", "--flat-playlist",
                "--playlist-items", f"1-{max_results}",
                playlist_url,
            ]
            result = self._run_subprocess_safe(cmd, timeout=60)
            videos = []
            for line in result.stdout.strip().split("\n"):
                if line.strip():
                    try:
                        data = json.loads(line)
                        videos.append({
                            "video_id": data["id"],
                            "title": data.get("title", ""),
                            "url": f"https://www.youtube.com/watch?v={data['id']}",
                        })
                    except (json.JSONDecodeError, KeyError):
                        continue
            return videos
        except Exception:
            return []

    @staticmethod
    def _extract_video_id(url: str) -> str:
        patterns = [
            r'(?:v=|/v/|youtu\.be/)([a-zA-Z0-9_-]{11})',
            r'^([a-zA-Z0-9_-]{11})$',
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        raise ValueError(f"Could not extract video ID from: {url}")
