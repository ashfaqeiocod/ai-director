"""engine.py — StoryEngine: runs the story layers in step with the live video.

The browser's <video> is the master clock. It sends its currentTime ~5x/s ("ticks"); on_tick()
moves the Playhead forward. The engine's own background thread then runs whatever is due, but
only up to the playhead (CausalReader refuses anything later):

  - FAST layer every `fast.tick_s` (1 s) of video.
  - If the engine falls more than `playhead.max_lag_s` behind (slow Mac, busy CPU), it does NOT
    try to catch up on old seconds: it jumps to the newest second, like a live system, and logs
    the lag ("LAG" in the signal log).

Everything it finds is sent to the dashboard with publish(message).
"""
from __future__ import annotations

import threading
import time
import traceback
from collections.abc import Callable
from pathlib import Path

from story.causal import CausalReader, Playhead
from story.fast_layer import FastLayer, FastResult
from story.fusion import Fusion

Publish = Callable[[dict], None]


class StoryEngine:
    def __init__(self, clip_folder: Path, cfg: dict, mode: str, publish: Publish, send_audio: bool = True):
        self.cfg = cfg
        self.mode = mode
        self.publish = publish
        self.send_audio = send_audio
        self.clip = Path(clip_folder).name
        self.playhead = Playhead()
        self.reader = CausalReader(clip_folder, self.playhead)
        self.fusion = Fusion(cfg, mode)
        self.fast = FastLayer(cfg, self.reader)
        self.paused = True
        self.ended = False
        self.ticks = 0
        self.fast_done_to = 0.0          # the fast layer has analysed everything up to here
        self.stop_flag = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True, name="story-engine")

    # ------------------------------------------------------------------ control
    def start(self) -> None:
        self.thread.start()

    def on_tick(self, t: float, paused: bool, ended: bool = False) -> None:
        """Called for every playhead message from the browser."""
        self.ticks += 1
        self.paused = paused
        self.ended = self.ended or ended
        self.playhead.advance(min(t, self.reader.duration))

    def stop(self) -> None:
        self.stop_flag.set()
        if self.thread.is_alive() and threading.current_thread() is not self.thread:
            self.thread.join(timeout=10)

    def log(self, kind: str, message: str, t: float | None = None) -> None:
        """A line in the dashboard's signal log."""
        self.publish({"type": "log", "kind": kind, "message": message,
                      "t": round(self.playhead.now() if t is None else t, 2)})

    # ------------------------------------------------------------------ main loop
    def _loop(self) -> None:
        self.log("INFO", f"audio model: {self.fast.audio.model_name}"
                         + (f" ({self.fast.audio.error})" if self.fast.audio.error else ""))
        while not self.stop_flag.is_set():
            try:
                did_work = self._step()
            except Exception as error:
                traceback.print_exc()
                self.log("ERROR", f"{type(error).__name__}: {error}")
                did_work = False
            if not did_work:
                time.sleep(0.02)

    def _step(self) -> bool:
        """Run whatever is due. Returns True if something ran."""
        tick_s = self.cfg["fast"]["tick_s"]
        now = self.playhead.now()
        # The last partial second at the end of the clip is analysed too.
        at_end = now >= self.reader.duration - 1e-3 and now > self.fast_done_to + 0.05
        if now - self.fast_done_to < tick_s and not at_end:
            return False
        t0 = self.fast_done_to
        lag = now - (t0 + tick_s)
        if lag > self.cfg["playhead"]["max_lag_s"]:
            # Too far behind: skip the old seconds and analyse the newest one, like a live system.
            self.log("LAG", f"backend {lag:.1f}s behind the video: skipping {now - tick_s - t0:.1f}s")
            t0 = now - tick_s
        t1 = min(t0 + tick_s, now)
        result = self.fast.step(t0, t1)
        self.fast_done_to = t1
        self._on_fast(result, lag=max(0.0, self.playhead.now() - t1))
        return True

    # ------------------------------------------------------------------ layer results
    def _on_fast(self, r: FastResult, lag: float) -> None:
        intensity = self.fusion.intensity(r)
        self.publish({"type": "fast", "t": round(r.t, 2), "audio": round(r.audio, 3), "motion": round(r.motion, 3),
                      "crowd": round(r.crowd, 3), "whistle": round(r.whistle, 2), "loudness_db": r.loudness_db,
                      "cuts": r.cuts, "replay": r.replay, "intensity": intensity, "lag": round(lag, 2),
                      "audio_model": r.audio_model})
        if "cheer" in r.triggers:
            self.log("CHEER", f"crowd/excitement jump to {r.audio:.2f} ({r.loudness_db:.0f} dB)", r.t)
        if r.whistle_hit:
            self.log("WHISTLE", f"whistle (score {r.whistle:.2f})", r.t)
        if r.cuts:
            self.log("CUT", f"{r.cuts} scene cut(s)" + (" → replay hint" if r.replay else ""), r.t)
