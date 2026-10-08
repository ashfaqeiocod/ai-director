"""engine.py — StoryEngine: the heart of MVP 2 (filled in step by step).

The browser's <video> is the master clock. It sends its currentTime ~5x/s ("ticks").
The engine moves the Playhead forward and, on its own background thread, runs whatever
layers are due up to that playhead. Results go to the dashboard via `publish(message)`.
"""
from __future__ import annotations

import threading
import time
from collections.abc import Callable
from pathlib import Path

from story.causal import CausalReader, Playhead

Publish = Callable[[dict], None]


class StoryEngine:
    def __init__(self, clip_folder: Path, cfg: dict, mode: str, publish: Publish, send_audio: bool = True):
        self.cfg = cfg
        self.mode = mode
        self.publish = publish
        self.send_audio = send_audio
        self.playhead = Playhead()
        self.reader = CausalReader(clip_folder, self.playhead)
        self.paused = True
        self.ticks = 0
        self.stop_flag = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def on_tick(self, t: float, paused: bool) -> None:
        """Called for every playhead message from the browser."""
        self.ticks += 1
        self.paused = paused
        self.playhead.advance(min(t, self.reader.duration))

    def stop(self) -> None:
        self.stop_flag.set()
        self.thread.join(timeout=5)

    def _loop(self) -> None:
        last_print = 0.0
        while not self.stop_flag.is_set():
            if time.time() - last_print >= 1.0:
                last_print = time.time()
                print(f"[story] playhead {self.playhead.now():7.2f}s  ticks {self.ticks}  paused {self.paused}")
            time.sleep(0.05)
