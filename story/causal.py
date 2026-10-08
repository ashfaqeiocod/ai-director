"""causal.py — the "no peeking" guard (rule 1 of the MVP 2 spec).

A live system can't see the future. Our clip file is fully on disk, so it would be easy to
cheat by accident (e.g. analyse the next 6 seconds). To make that impossible, EVERY part of the
story system reads video and audio only through a CausalReader:

    playhead = Playhead()                 # updated ~5x/s from the browser's <video>.currentTime
    reader = CausalReader(clip_folder, playhead)
    reader.audio(t0, t1)                  # 16 kHz samples between t0 and t1
    reader.video().frames(t0, t1, fps=10) # frames between t0 and t1
    reader.window_clip(t0, t1, out_path)  # a small mp4 of [t0, t1] for Gemini

If anything asks for a time later than the playhead, it raises FutureAccessError
instead of returning data. tests/test_story_causal.py checks this.
"""
from __future__ import annotations

import json
import subprocess
import threading
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
from scipy.io import wavfile

from story.common import ffmpeg_exe

# Float rounding slack: 1 microsecond. Anything later than the playhead by more than this is "future".
EPSILON = 1e-6


class FutureAccessError(Exception):
    """Something tried to read video or audio from after the playhead."""


class Playhead:
    """The current position of the live video, in seconds. Thread-safe.

    It only moves forward: like a real live broadcast, time never goes back. A smaller value
    (e.g. the user seeking backwards) is ignored.
    """

    def __init__(self, start: float = 0.0):
        self._t = start
        self._lock = threading.Lock()

    def now(self) -> float:
        with self._lock:
            return self._t

    def advance(self, t: float) -> float:
        """Move to t if t is later than now. Returns the (possibly unchanged) playhead."""
        with self._lock:
            if t > self._t:
                self._t = t
            return self._t


class CausalReader:
    """Reads a normalized clip folder (clip.mp4 + audio.wav), but never past the playhead."""

    def __init__(self, folder: Path, playhead: Playhead):
        self.folder = Path(folder)
        self.playhead = playhead
        meta = json.loads((self.folder / "meta.json").read_text())
        self.duration: float = meta["duration"]
        self.fps: float = meta["fps"]
        self.width: int = meta["width"]
        self.height: int = meta["height"]
        self.has_audio: bool = meta["has_audio"]
        # mmap: the wav stays on disk, only the slices we ask for are read into memory.
        self.sample_rate, self._samples = wavfile.read(self.folder / "audio.wav", mmap=True)

    # ------------------------------------------------------------------ the guard
    def check(self, t: float) -> None:
        """Raise FutureAccessError if t is after the playhead."""
        now = self.playhead.now()
        if t > now + EPSILON:
            raise FutureAccessError(f"asked for t={t:.3f}s but the playhead is at {now:.3f}s")

    # ------------------------------------------------------------------ audio
    def audio(self, t0: float, t1: float) -> np.ndarray:
        """Mono float32 samples (-1..1) from t0 to t1 seconds."""
        self.check(t1)
        a = max(0, int(round(t0 * self.sample_rate)))
        b = max(a, int(round(t1 * self.sample_rate)))
        return np.asarray(self._samples[a:b], dtype=np.float32) / 32768.0

    # ------------------------------------------------------------------ video
    def video(self) -> CausalVideo:
        """A frame reader. Each user (fast layer, specialist...) opens its own, because
        an OpenCV video file can't be shared between threads."""
        return CausalVideo(self)

    def window_clip(self, t0: float, t1: float, out: Path, height: int, with_audio: bool,
                    fps: float | None = None) -> Path:
        """Cut [t0, t1] into a small mp4 (for Gemini). t1 must not be after the playhead."""
        self.check(t1)
        t0 = max(0.0, t0)
        vf = f"scale=-2:{height // 2 * 2}" + (f",fps={fps}" if fps else "")
        cmd = [ffmpeg_exe(), "-y", "-loglevel", "error",
               "-ss", f"{t0:.3f}", "-i", str(self.folder / "clip.mp4"),
               # -t: nothing after t1 ends up in the output file.
               "-t", f"{t1 - t0:.3f}", "-vf", vf,
               "-c:v", "libx264", "-preset", "ultrafast", "-crf", "30", "-pix_fmt", "yuv420p"]
        cmd += ["-c:a", "aac", "-b:a", "48k", "-ac", "1"] if (with_audio and self.has_audio) else ["-an"]
        subprocess.run(cmd + [str(out)], check=True)
        return out


class CausalVideo:
    """Sequential frame reader for one consumer. Frame i shows the moment i / fps."""

    def __init__(self, reader: CausalReader):
        self.reader = reader
        self.cap = cv2.VideoCapture(str(reader.folder / "clip.mp4"))
        self.next_index = 0          # index of the frame the next cap.grab() returns

    def frame_time(self, index: int) -> float:
        return index / self.reader.fps

    def frames(self, t0: float, t1: float, fps: float | None = None) -> Iterator[tuple[float, np.ndarray]]:
        """Yield (time, BGR frame) for frames with t0 < time <= t1, about `fps` per second
        (None = every frame). t1 must not be after the playhead."""
        self.reader.check(t1)
        src_fps = self.reader.fps
        first = int(np.floor(t0 * src_fps)) + 1 if t0 >= 0 else 0
        last = int(np.floor(t1 * src_fps + EPSILON))
        step = 1 if not fps else max(1, int(round(src_fps / fps)))
        wanted = [i for i in range(first, last + 1) if i % step == 0]
        for index in wanted:
            frame = self._read(index)
            if frame is None:
                return
            yield self.frame_time(index), frame

    def _read(self, index: int) -> np.ndarray | None:
        # Double guard: never decode a frame later than the playhead.
        self.reader.check(self.frame_time(index))
        # Short jumps forward: skip frames by grabbing (fast). Long jumps or going back: seek.
        if index < self.next_index or index - self.next_index > 2 * self.reader.fps:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            self.next_index = index
        while self.next_index < index:
            if not self.cap.grab():
                return None
            self.next_index += 1
        ok, frame = self.cap.read()
        self.next_index += 1
        return frame if ok else None

    def close(self) -> None:
        self.cap.release()
