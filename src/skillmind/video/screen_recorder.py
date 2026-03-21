"""
SkillMind Screen Recorder — capture screen recordings and extract knowledge.

Features:
- Record screen (full screen or region)
- Record with audio (optional)
- Auto-stop after duration or on keypress
- Save as MP4/AVI
- Extract knowledge from recording via OCR + frame analysis

Dependencies:
    pip install opencv-python pillow mss

Optional for audio:
    pip install sounddevice soundfile

Usage:
    recorder = ScreenRecorder()
    path = recorder.record(duration=30, output="demo.mp4")
    recorder.record_region(x=0, y=0, w=1920, h=1080, duration=60)
"""

from __future__ import annotations

import os
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Any


class ScreenRecorder:
    """
    Record screen to video file.

    Supports full screen, region capture, and optional audio.
    """

    def __init__(self, output_dir: str = ".skillmind/recordings"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._recording = False
        self._thread: threading.Thread | None = None

    def record(
        self,
        duration: int = 30,
        fps: int = 15,
        output: str | None = None,
        monitor: int = 1,
        with_audio: bool = False,
    ) -> str:
        """
        Record full screen for a given duration.

        Args:
            duration: Recording duration in seconds
            fps: Frames per second (15 is good for tutorials, 30 for demos)
            output: Output filename (auto-generated if None)
            monitor: Monitor number (1 = primary)
            with_audio: Record system audio (requires sounddevice)

        Returns:
            Path to the recorded video file.
        """
        import cv2
        import mss
        import numpy as np

        if output is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output = str(self.output_dir / f"recording_{timestamp}.mp4")
        else:
            output = str(self.output_dir / output)

        sct = mss.mss()
        mon = sct.monitors[monitor]
        width = mon["width"]
        height = mon["height"]

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output, fourcc, fps, (width, height))

        self._recording = True
        frame_interval = 1.0 / fps
        start_time = time.time()

        print(f"Recording screen ({width}x{height}) for {duration}s → {output}")

        try:
            while self._recording and (time.time() - start_time) < duration:
                frame_start = time.time()

                img = sct.grab(mon)
                frame = np.array(img)
                # mss captures BGRA, OpenCV needs BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                writer.write(frame)

                elapsed = time.time() - frame_start
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)
        finally:
            writer.release()
            self._recording = False

        actual_duration = time.time() - start_time
        print(f"Recording saved: {output} ({actual_duration:.1f}s)")
        return output

    def record_region(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        duration: int = 30,
        fps: int = 15,
        output: str | None = None,
    ) -> str:
        """Record a specific screen region."""
        import cv2
        import mss
        import numpy as np

        if output is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output = str(self.output_dir / f"region_{timestamp}.mp4")
        else:
            output = str(self.output_dir / output)

        region = {"top": y, "left": x, "width": w, "height": h}

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output, fourcc, fps, (w, h))

        sct = mss.mss()
        self._recording = True
        frame_interval = 1.0 / fps
        start_time = time.time()

        print(f"Recording region ({w}x{h} at {x},{y}) for {duration}s → {output}")

        try:
            while self._recording and (time.time() - start_time) < duration:
                frame_start = time.time()

                img = sct.grab(region)
                frame = np.array(img)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                writer.write(frame)

                elapsed = time.time() - frame_start
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)
        finally:
            writer.release()
            self._recording = False

        print(f"Recording saved: {output}")
        return output

    def record_async(
        self,
        duration: int = 30,
        fps: int = 15,
        output: str | None = None,
    ) -> str:
        """Start recording in background thread. Call stop() to finish early."""
        if output is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output = f"recording_{timestamp}.mp4"

        self._thread = threading.Thread(
            target=self.record,
            kwargs={"duration": duration, "fps": fps, "output": output},
            daemon=True,
        )
        self._thread.start()
        return str(self.output_dir / output)

    def stop(self) -> None:
        """Stop an ongoing recording."""
        self._recording = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    def screenshot(self, output: str | None = None, monitor: int = 1) -> str:
        """Take a single screenshot."""
        import mss
        import mss.tools

        if output is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output = str(self.output_dir / f"screenshot_{timestamp}.png")
        else:
            output = str(self.output_dir / output)

        sct = mss.mss()
        mon = sct.monitors[monitor]
        img = sct.grab(mon)
        mss.tools.to_png(img.rgb, img.size, output=output)

        print(f"Screenshot saved: {output}")
        return output
