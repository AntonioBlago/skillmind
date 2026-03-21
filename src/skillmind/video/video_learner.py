"""
SkillMind Video Learner — extract knowledge from local video files
and screen recordings using frame analysis + OCR.

Workflow:
1. Sample key frames from video (every N seconds or on scene change)
2. OCR text from frames (code, UI text, terminal output)
3. Extract audio transcript (Whisper or external)
4. Combine visual + audio knowledge
5. Store as structured memories

Dependencies:
    pip install opencv-python pillow pytesseract

Optional:
    pip install openai-whisper   # For local audio transcription

Usage:
    learner = VideoLearner(trainer)
    memories = learner.learn("recording.mp4")
    memories = learner.learn("tutorial.mp4", extract_audio=True)
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from ..models import Memory, MemorySource, MemoryType
from ..trainer import Trainer


class VideoLearner:
    """
    Extract knowledge from local video files and screen recordings.

    Combines:
    - Frame sampling (key frames, scene changes)
    - OCR (code, terminal text, UI elements)
    - Audio transcription (optional, via Whisper)
    - Claude API for knowledge structuring
    """

    def __init__(
        self,
        trainer: Trainer,
        anthropic_api_key: str | None = None,
        claude_model: str = "claude-sonnet-4-6",
    ):
        self.trainer = trainer
        self.api_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.claude_model = claude_model

    def learn(
        self,
        video_path: str,
        force_topic: str | None = None,
        tags: list[str] | None = None,
        extract_audio: bool = False,
        frame_interval: int = 5,
        max_frames: int = 50,
    ) -> list[Memory]:
        """
        Learn from a local video file.

        Args:
            video_path: Path to video file (MP4, AVI, etc.)
            force_topic: Override auto-detected topic
            tags: Additional tags
            extract_audio: Also transcribe audio via Whisper
            frame_interval: Sample a frame every N seconds
            max_frames: Maximum number of frames to analyze

        Returns:
            List of created memories.
        """
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        print(f"Learning from video: {path.name}")

        # 1. Extract key frames
        print("  Extracting key frames...")
        frames = self._extract_frames(str(path), frame_interval, max_frames)
        print(f"  → {len(frames)} frames sampled")

        # 2. OCR text from frames
        print("  Running OCR on frames...")
        ocr_texts = self._ocr_frames(frames)
        unique_texts = self._deduplicate_ocr(ocr_texts)
        print(f"  → {len(unique_texts)} unique text blocks extracted")

        # 3. Audio transcription (optional)
        transcript = ""
        if extract_audio:
            print("  Transcribing audio...")
            transcript = self._transcribe_audio(str(path))
            word_count = len(transcript.split()) if transcript else 0
            print(f"  → {word_count} words transcribed")

        # 4. Combine and structure knowledge
        print("  Structuring knowledge...")
        knowledge = self._structure_knowledge(
            ocr_texts=unique_texts,
            transcript=transcript,
            video_name=path.stem,
            duration=self._get_duration(str(path)),
        )

        # 5. Store as memories
        memories: list[Memory] = []

        # Main summary
        if knowledge.get("summary"):
            mem = self.trainer.learn(
                content=knowledge["summary"],
                title=knowledge.get("title", f"Video: {path.stem}")[:80],
                source=MemorySource.SKILL_SEEKERS,
                force_type=MemoryType.SKILL,
                force_topic=force_topic or knowledge.get("topic", "video"),
                tags=(tags or []) + knowledge.get("tags", []) + ["video", "screen_recording"],
                metadata={
                    "video_path": str(path),
                    "duration": knowledge.get("duration", 0),
                    "frame_count": len(frames),
                    "has_audio": bool(transcript),
                },
            )
            if mem:
                memories.append(mem)

        # Code snippets found via OCR
        for snippet in knowledge.get("code_snippets", [])[:5]:
            mem = self.trainer.learn(
                content=snippet,
                title=f"Code from {path.stem}"[:60],
                source=MemorySource.SKILL_SEEKERS,
                force_type=MemoryType.SKILL,
                force_topic=force_topic or "code",
                tags=["video", "code", "ocr"],
            )
            if mem:
                memories.append(mem)

        # Key takeaways
        for takeaway in knowledge.get("key_takeaways", [])[:5]:
            mem = self.trainer.learn(
                content=takeaway,
                title=f"Takeaway: {takeaway[:60]}",
                source=MemorySource.SKILL_SEEKERS,
                force_type=MemoryType.SKILL,
                force_topic=force_topic or knowledge.get("topic", "video"),
                tags=["video", "takeaway"],
            )
            if mem:
                memories.append(mem)

        # Cleanup temp frames
        for frame_path in frames:
            try:
                os.remove(frame_path)
            except OSError:
                pass

        print(f"  → {len(memories)} memories created")
        return memories

    # ── Frame Extraction ──────────────────────────────────────────

    def _extract_frames(
        self, video_path: str, interval: int = 5, max_frames: int = 50
    ) -> list[str]:
        """Extract key frames from video at regular intervals."""
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_step = int(fps * interval)

        frames: list[str] = []
        tmpdir = tempfile.mkdtemp(prefix="skillmind_frames_")

        frame_idx = 0
        while frame_idx < total_frames and len(frames) < max_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            frame_path = os.path.join(tmpdir, f"frame_{frame_idx:06d}.png")
            cv2.imwrite(frame_path, frame)
            frames.append(frame_path)
            frame_idx += frame_step

        cap.release()
        return frames

    # ── OCR ───────────────────────────────────────────────────────

    def _ocr_frames(self, frame_paths: list[str]) -> list[str]:
        """Run OCR on frame images to extract text."""
        try:
            import pytesseract
            from PIL import Image
        except ImportError:
            # Fallback: no OCR available
            return []

        texts: list[str] = []
        for path in frame_paths:
            try:
                img = Image.open(path)
                text = pytesseract.image_to_string(img, lang="deu+eng")
                text = text.strip()
                if text and len(text) > 20:  # Skip near-empty frames
                    texts.append(text)
            except Exception:
                continue

        return texts

    def _deduplicate_ocr(self, texts: list[str], similarity_threshold: float = 0.8) -> list[str]:
        """Remove near-duplicate OCR results (consecutive similar frames)."""
        if not texts:
            return []

        unique: list[str] = [texts[0]]
        for text in texts[1:]:
            # Simple similarity: character overlap ratio
            if unique:
                last = unique[-1]
                overlap = sum(1 for a, b in zip(text, last) if a == b)
                max_len = max(len(text), len(last))
                similarity = overlap / max_len if max_len > 0 else 0
                if similarity < similarity_threshold:
                    unique.append(text)
            else:
                unique.append(text)

        return unique

    # ── Audio Transcription ───────────────────────────────────────

    def _transcribe_audio(self, video_path: str) -> str:
        """Transcribe audio from video using Whisper."""
        try:
            import whisper

            model = whisper.load_model("base")
            result = model.transcribe(video_path, language="de")
            return result.get("text", "")
        except ImportError:
            pass

        # Fallback: extract audio with ffmpeg and use external transcription
        try:
            import subprocess

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                audio_path = tmp.name

            subprocess.run(
                ["ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
                 "-ar", "16000", "-ac", "1", audio_path, "-y"],
                capture_output=True, timeout=120,
            )

            # Try whisper on the extracted audio
            try:
                import whisper

                model = whisper.load_model("base")
                result = model.transcribe(audio_path, language="de")
                return result.get("text", "")
            except ImportError:
                pass
            finally:
                os.unlink(audio_path)
        except Exception:
            pass

        return ""

    # ── Knowledge Structuring ─────────────────────────────────────

    def _structure_knowledge(
        self,
        ocr_texts: list[str],
        transcript: str,
        video_name: str,
        duration: int,
    ) -> dict:
        """Structure extracted content into organized knowledge."""
        # Detect code snippets in OCR text
        code_snippets: list[str] = []
        non_code_texts: list[str] = []

        for text in ocr_texts:
            if self._looks_like_code(text):
                code_snippets.append(text)
            else:
                non_code_texts.append(text)

        if self.api_key and (non_code_texts or transcript):
            return self._structure_with_claude(
                ocr_texts=non_code_texts,
                code_snippets=code_snippets,
                transcript=transcript,
                video_name=video_name,
                duration=duration,
            )

        # No API: simple concatenation
        summary_parts = []
        if non_code_texts:
            summary_parts.append("## Visible Text\n" + "\n\n".join(non_code_texts[:10]))
        if transcript:
            summary_parts.append("## Audio Transcript\n" + transcript[:3000])
        if code_snippets:
            summary_parts.append("## Code Snippets\n```\n" + "\n---\n".join(code_snippets[:5]) + "\n```")

        return {
            "title": f"Video: {video_name}",
            "summary": "\n\n".join(summary_parts) if summary_parts else f"Screen recording: {video_name}",
            "key_takeaways": [],
            "code_snippets": code_snippets[:5],
            "topic": "video",
            "tags": ["video"],
            "duration": duration,
        }

    def _structure_with_claude(
        self,
        ocr_texts: list[str],
        code_snippets: list[str],
        transcript: str,
        video_name: str,
        duration: int,
    ) -> dict:
        """Use Claude to structure extracted video knowledge."""
        import anthropic
        import yaml

        client = anthropic.Anthropic(api_key=self.api_key)

        ocr_section = "\n\n".join(ocr_texts[:15]) if ocr_texts else "(keine)"
        code_section = "\n---\n".join(code_snippets[:5]) if code_snippets else "(keine)"
        audio_section = transcript[:4000] if transcript else "(keine)"

        prompt = f"""Analysiere die folgenden Inhalte aus einem Screen-Recording/Video und extrahiere das Kernwissen.

## Video: {video_name} ({duration // 60} Minuten)

## OCR-Text aus Frames:
{ocr_section}

## Code-Snippets aus Frames:
{code_section}

## Audio-Transkript:
{audio_section}

## Aufgabe
Extrahiere das Wissen im YAML-Format:

---
title: "Praegnanter Titel"
topic: "hauptthema"
tags: [tag1, tag2, tag3]
summary: |
  Strukturierte Zusammenfassung (300-800 Woerter).
  Was wird gezeigt? Welche Workflows/Tools/Techniken?
key_takeaways:
  - "Erkenntnis 1"
  - "Erkenntnis 2"
  - "Erkenntnis 3"
code_snippets:
  - "Bereinigter Code-Snippet 1"
  - "Bereinigter Code-Snippet 2"
---

Antworte NUR mit dem YAML-Block."""

        response = client.messages.create(
            model=self.claude_model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )

        text = response.content[0].text.strip()
        fm_match = re.match(r'^---\s*\n(.*?)\n---', text, re.DOTALL)
        if fm_match:
            try:
                data = yaml.safe_load(fm_match.group(1))
                return {
                    "title": data.get("title", f"Video: {video_name}"),
                    "summary": data.get("summary", ""),
                    "key_takeaways": data.get("key_takeaways", []),
                    "code_snippets": data.get("code_snippets", code_snippets[:5]),
                    "topic": data.get("topic", "video"),
                    "tags": data.get("tags", []),
                    "duration": duration,
                }
            except yaml.YAMLError:
                pass

        return {
            "title": f"Video: {video_name}",
            "summary": text[:2000],
            "key_takeaways": [],
            "code_snippets": code_snippets[:5],
            "topic": "video",
            "tags": [],
            "duration": duration,
        }

    # ── Helpers ───────────────────────────────────────────────────

    @staticmethod
    def _looks_like_code(text: str) -> bool:
        """Heuristic: does this OCR text look like code?"""
        code_indicators = [
            r'\bdef\s+\w+', r'\bclass\s+\w+', r'\bimport\s+', r'\bfrom\s+\w+\s+import',
            r'\bfunction\s+', r'\bconst\s+', r'\blet\s+', r'\bvar\s+',
            r'[{}\[\]();]', r'=>', r'\.then\(', r'async\s+',
            r'\$\s', r'pip\s+install', r'npm\s+', r'git\s+',
            r'^\s*(#|//|/\*)', r'localhost:\d+',
        ]
        matches = sum(1 for p in code_indicators if re.search(p, text, re.MULTILINE))
        return matches >= 2

    @staticmethod
    def _get_duration(video_path: str) -> int:
        """Get video duration in seconds."""
        try:
            import cv2

            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.release()
            return int(frames / fps)
        except Exception:
            return 0
