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
from typing import Any

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
    ) -> list[Memory]:
        """Non-blocking version of learn() for use in async MCP tools."""
        return await asyncio.to_thread(self.learn, video_url, force_topic, tags)

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
    ) -> list[Memory]:
        """
        Learn from a single YouTube video.

        Steps:
        1. Extract video ID
        2. Fetch metadata (title, author, duration, description)
        3. Fetch transcript
        4. Extract structured knowledge via Claude API
        5. Store as memories (skill + reference)
        """
        video_id = self._extract_video_id(video_url)
        metadata = self._get_metadata(video_id)
        transcript = self._get_transcript(video_id)

        if not transcript:
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

        # Extract knowledge
        knowledge = self._extract_knowledge(transcript, metadata)
        memories: list[Memory] = []

        # Store main knowledge as a skill memory
        mem = self.trainer.learn(
            content=knowledge["summary"],
            title=knowledge.get("title", metadata.get("title", "YouTube Video"))[:80],
            source=MemorySource.SKILL_SEEKERS,
            force_type=MemoryType.SKILL,
            force_topic=force_topic or knowledge.get("topic", "youtube"),
            tags=(tags or []) + knowledge.get("tags", []) + ["youtube"],
            metadata={
                "video_id": video_id,
                "video_url": metadata.get("url", ""),
                "duration": metadata.get("duration", 0),
                "author": metadata.get("author", ""),
                "key_takeaways": knowledge.get("key_takeaways", []),
            },
        )
        if mem:
            memories.append(mem)

        # Store individual key takeaways as separate memories
        for takeaway in knowledge.get("key_takeaways", [])[:5]:
            mem = self.trainer.learn(
                content=takeaway,
                title=f"Takeaway: {takeaway[:60]}",
                source=MemorySource.SKILL_SEEKERS,
                force_type=MemoryType.SKILL,
                force_topic=force_topic or knowledge.get("topic", "youtube"),
                tags=["youtube", "takeaway"],
                metadata={"source_video": video_id},
            )
            if mem:
                memories.append(mem)

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
            force_topic=force_topic or "youtube",
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
        videos = self._get_channel_videos(channel_id, max_videos)
        all_memories: list[Memory] = []

        for video in videos:
            try:
                memories = self.learn(
                    video["url"],
                    force_topic=force_topic,
                    tags=["channel:" + channel_id],
                )
                all_memories.extend(memories)
            except Exception as e:
                print(f"Failed to learn from {video.get('title', video['url'])}: {e}")
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

    # ── Transcript Extraction ─────────────────────────────────────

    def _get_transcript(self, video_id: str) -> str:
        """Fetch transcript, trying youtube-transcript-api first, then yt-dlp.

        Each method has explicit timeouts to prevent indefinite hangs.
        """
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
        """Fetch transcript via youtube-transcript-api with timeout protection."""
        import concurrent.futures

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

        def _fetch() -> str:
            try:
                entries = ytt.fetch(video_id, languages=[self.language, "en", "de"])
                return " ".join(entry.text for entry in entries)
            except Exception:
                transcript_list = ytt.list(video_id)
                first = next(iter(transcript_list))
                entries = first.fetch()
                return " ".join(entry.text for entry in entries)

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_fetch)
            try:
                return future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                future.cancel()
                raise TimeoutError(
                    f"youtube-transcript-api timed out after {timeout}s for {video_id}"
                )

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
            subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            for lang in [self.language, "en"]:
                sub_path = os.path.join(tmpdir, f"sub_{video_id}.{lang}.json3")
                if os.path.exists(sub_path):
                    with open(sub_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    texts = []
                    for event in data.get("events", []):
                        for seg in event.get("segs", []):
                            text = seg.get("utf8", "").strip()
                            if text and text != "\n":
                                texts.append(text)
                    return " ".join(texts)

        return ""

    # ── Metadata ──────────────────────────────────────────────────

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
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                data = json.loads(result.stdout)
                return {
                    "title": data.get("title", ""),
                    "author": data.get("uploader", ""),
                    "description": data.get("description", "")[:500],
                    "duration": data.get("duration", 0) or 0,
                    "tags": data.get("tags", [])[:10],
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
                "video_id": video_id,
                "url": f"https://www.youtube.com/watch?v={video_id}",
            }
        except Exception:
            return {
                "title": "", "author": "", "description": "",
                "duration": 0, "tags": [], "video_id": video_id,
                "url": f"https://www.youtube.com/watch?v={video_id}",
            }

    # ── Knowledge Extraction ──────────────────────────────────────

    def _extract_knowledge(self, transcript: str, metadata: dict) -> dict:
        """Extract structured knowledge from transcript using Claude API."""
        if not self.api_key:
            # No API key — return raw transcript as knowledge
            return {
                "title": metadata.get("title", "YouTube Video"),
                "summary": transcript[:2000],
                "key_takeaways": [],
                "topic": "youtube",
                "tags": metadata.get("tags", []),
            }

        import anthropic

        client = anthropic.Anthropic(api_key=self.api_key)

        # Chunk if very long
        max_chars = 80000
        content = transcript[:max_chars]

        prompt = f"""Extrahiere das Kernwissen aus diesem YouTube-Video-Transkript.
Fokussiere dich auf wiederverwendbare Erkenntnisse, Workflows, Tipps und Fakten.

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
topic: "hauptthema"
tags: [tag1, tag2, tag3]
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

Antworte NUR mit dem YAML-Block."""

        response = client.messages.create(
            model=self.claude_model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )

        return self._parse_knowledge_yaml(response.content[0].text, metadata)

    def _parse_knowledge_yaml(self, text: str, metadata: dict) -> dict:
        """Parse YAML knowledge extraction response."""
        import yaml

        text = text.strip()
        fm_match = re.match(r'^---\s*\n(.*?)\n---', text, re.DOTALL)
        if fm_match:
            try:
                data = yaml.safe_load(fm_match.group(1))
                return {
                    "title": data.get("title", metadata.get("title", "")),
                    "summary": data.get("summary", ""),
                    "key_takeaways": data.get("key_takeaways", []),
                    "topic": data.get("topic", "youtube"),
                    "tags": data.get("tags", []),
                }
            except yaml.YAMLError:
                pass

        # Fallback
        return {
            "title": metadata.get("title", "YouTube Video"),
            "summary": text[:2000],
            "key_takeaways": [],
            "topic": "youtube",
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
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
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
