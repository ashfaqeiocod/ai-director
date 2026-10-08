"""engine.py — the frame-by-frame core shared by the CLI (main.py) and the dashboard.

Think of it like a game loop. On every tick the Engine:
  1. reads ONE frame from each of the 3 cameras (same moment in time),
  2. gets detections for each camera, either by running the model now ("live" mode)
     or by reading the cached JSON from Step 4 ("replay" mode),
  3. updates the ball tracker and the player tracker of each camera,
  4. asks the Director which camera should be live,
  5. yields a FrameResult with everything that happened.

Whoever consumes the results decides what to do with them: write videos, push them
to a web page, print them. A real live system later would have the same shape.
"""
from __future__ import annotations

import multiprocessing
import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
import supervision as sv

from src.ball_tracker import BallState, BallTracker
from src.common import cam_path, detections_path, video_info
from src.detect import Detector, load_detections, save_detections
from src.director import Director

LIVE, REPLAY = "live", "replay"


@dataclass
class CamResult:
    frame: np.ndarray                 # the raw camera image (BGR)
    players: sv.Detections            # tracked players (with tracker_id)
    num_players: int                  # players detected this frame (before tracking)
    ball_box: list[float] | None      # raw ball detection, if any
    ball: BallState                   # tracker output: detected / predicted / lost


@dataclass
class FrameResult:
    index: int
    time: float                       # seconds since the start of the clip
    cams: list[CamResult]
    scores: list[float]
    chosen: int                       # live camera, 1-based
    reason: str
    events: list[dict] = field(default_factory=list)   # CUT / BALL_LOST / BALL_FOUND
    processing_fps: float = 0.0
    detection_fps: float = 0.0        # how many frames per second really got NEW detections


class Engine:
    def __init__(self, cfg: dict, mode: str = REPLAY, detector: Detector | None = None):
        self.cfg = cfg
        self.mode = mode
        self.num_cams = cfg["video"]["num_cams"]
        self.info = video_info(cam_path(cfg, 1))
        self.fps = self.info["fps"]
        self.total_frames = self.info["frames"]
        self._stop = False

        live = cfg.get("live", {})
        self.detect_every_n = max(1, int(live.get("detect_every_n", 1)))
        self.realtime = bool(live.get("realtime", False))
        self.pool = None
        self.detector = None
        if mode == LIVE:
            self.cache = None
            if not self.realtime:
                # Reuse a detector if the caller already loaded one (models are slow to load).
                # (Real-time mode loads its own detector in a separate process instead.)
                self.detector = detector or Detector(cfg)
            if live.get("parallel_cams", False) and not self.realtime:
                # One thread per camera; they share the same loaded model (no extra model in RAM).
                self.pool = ThreadPoolExecutor(max_workers=self.num_cams)
        else:
            missing = [c for c in range(1, self.num_cams + 1) if not detections_path(cfg, c).exists()]
            if missing:
                raise FileNotFoundError(f"No cached detections for cams {missing}. Run detection (live mode) first.")
            self.cache = [load_detections(detections_path(cfg, c))["frames"] for c in range(1, self.num_cams + 1)]

        bt = cfg["ball_tracker"]
        size = (self.info["width"], self.info["height"])
        self.ball_trackers = [BallTracker(bt["buffer_size"], bt["max_jump_px"], bt["max_predict_frames"], size)
                              for _ in range(self.num_cams)]
        self.player_trackers = [sv.ByteTrack(frame_rate=round(self.fps)) for _ in range(self.num_cams)]
        self.director = Director(cfg["director"], self.fps, self.num_cams)

    def stop(self) -> None:
        """Ask the engine to stop after the current frame."""
        self._stop = True

    def frames(self) -> Iterator[FrameResult]:
        """Yield one FrameResult per synced frame of the 3 cameras."""
        readers = [sv.get_video_frames_generator(str(cam_path(self.cfg, c))) for c in range(1, self.num_cams + 1)]
        live_cache: list[list[dict]] = [[] for _ in range(self.num_cams)]
        frame_meter, detect_meter = RateMeter(), RateMeter(window=10)
        last_dets: list[dict] = [{"players": [], "ball": None} for _ in range(self.num_cams)]
        background = None
        if self.mode == LIVE and self.realtime:
            background = BackgroundDetector(self.cfg)
            background.wait_ready(lambda: self._stop)   # don't start the clip with no boxes
        clock_start = time.time()

        for index, images in enumerate(zip(*readers)):
            if self._stop:
                break

            # 1. Get detections for this frame (see the "live" section in config.yaml).
            if self.mode == REPLAY:
                dets = [self._cached(c, index) for c in range(self.num_cams)]
            elif background is not None:
                # Real-time: hand the newest frames to the background detector when it is free,
                # and keep using its latest answer until a newer one arrives.
                background.submit(images)
                last_dets = background.latest() or last_dets
                dets = last_dets
            elif index % self.detect_every_n == 0:
                dets = last_dets = self._detect_all(images)
                detect_meter.tick()
            else:
                dets = last_dets       # skipped frame: reuse the previous detections
            if self.mode == LIVE:
                for c in range(self.num_cams):
                    live_cache[c].append(dets[c])

            # 2. Trackers + director.
            cams = [self._track(c, image, dets[c]) for c, image in enumerate(images)]
            decision = self.director.decide(index, [cm.ball for cm in cams], [cm.num_players for cm in cams],
                                            self.info["width"])

            # Real-time mode plays the video at its normal speed, whatever the detector does.
            if background is not None:
                delay = clock_start + (index + 1) / self.fps - time.time()
                if delay > 0:
                    time.sleep(delay)

            # Real processing speed over the last frames, shown honestly in the dashboard.
            frame_meter.tick()
            if self.mode == REPLAY:
                detection_fps = 0.0                # replay reads the cache, no detector runs
            elif background is not None:
                detection_fps = background.rate()
            else:
                detection_fps = detect_meter.rate()

            yield FrameResult(index, index / self.fps, cams, decision.scores, decision.cam, decision.reason,
                              decision.events, frame_meter.rate(), detection_fps)

        if background is not None:
            background.close()
        # A complete live run becomes the cache, so replay mode can use it later.
        # Only when EVERY frame was really detected, so replay never gets worse data.
        full_quality = self.detect_every_n == 1 and not self.realtime
        if self.mode == LIVE and full_quality and not self._stop and len(live_cache[0]) == self.total_frames:
            for c in range(self.num_cams):
                info = video_info(cam_path(self.cfg, c + 1))
                save_detections(detections_path(self.cfg, c + 1), c + 1, info, self.detector.describe(), live_cache[c])

    def _detect_all(self, images) -> list[dict]:
        """Run the detector on one frame of every camera (in parallel if configured)."""
        if self.pool is not None:
            return list(self.pool.map(self.detector.detect, images))
        return [self.detector.detect(image) for image in images]

    def _cached(self, cam_index: int, frame_index: int) -> dict:
        frames = self.cache[cam_index]
        return frames[frame_index] if frame_index < len(frames) else {"players": [], "ball": None}

    def _track(self, cam_index: int, image: np.ndarray, det: dict) -> CamResult:
        ball_box = det["ball"]["box"] if det["ball"] else None
        ball = self.ball_trackers[cam_index].update(ball_box)
        players = _to_detections(det["players"])
        tracked = self.player_trackers[cam_index].update_with_detections(players)
        return CamResult(image, tracked, len(det["players"]), ball_box, ball)


class RateMeter:
    """Counts events per second over the last `window` events (e.g. frames or detections)."""

    def __init__(self, window: int = 30):
        self.times: deque[float] = deque(maxlen=window)

    def tick(self) -> None:
        self.times.append(time.time())

    def rate(self) -> float:
        if len(self.times) < 2:
            return 0.0
        return round((len(self.times) - 1) / max(self.times[-1] - self.times[0], 1e-6), 1)


class BackgroundDetector:
    """Runs detection in a separate PROCESS, always on the NEWEST frames (real-time mode).

    Why a process and not a thread: Python lets only one thread run Python code at a
    time, so a busy detector thread makes the video stutter. A separate process has its
    own Python, and it runs at low priority, so the video always gets the CPU first.
    It loads its own copy of the model (about 0.5 GB extra RAM).

    The video keeps playing at normal speed. While the detector is busy, new frames are
    simply not detected; the engine reuses the last answer. On a fast machine almost every
    frame gets detected; on a slow one the boxes update less often.
    """

    def __init__(self, cfg: dict):
        ctx = multiprocessing.get_context("spawn")
        self.conn, child_conn = ctx.Pipe()
        self.process = ctx.Process(target=_detector_process, args=(cfg, child_conn), daemon=True)
        self.process.start()
        self.ready = False             # the model is still loading in the other process
        self.busy = False
        self.result: list[dict] | None = None   # newest answer, not yet picked up
        self.meter = RateMeter(window=10)

    def submit(self, images) -> None:
        """Send frames to the detector, but only if it is free (otherwise skip them)."""
        self._collect()
        if self.ready and not self.busy:
            self.conn.send([np.ascontiguousarray(image) for image in images])
            self.busy = True

    def latest(self) -> list[dict] | None:
        """The newest finished detections, or None if nothing new arrived."""
        self._collect()
        result, self.result = self.result, None
        return result

    def wait_ready(self, should_stop) -> None:
        """Block until the model is loaded in the other process (or until stopped)."""
        while not self.ready and not should_stop():
            self._collect()
            time.sleep(0.1)

    def rate(self) -> float:
        """Detections finished per second (over the last few)."""
        return self.meter.rate()

    def close(self) -> None:
        try:
            if self.busy:
                self.conn.recv()
            self.conn.send(None)       # "please exit"
        except (EOFError, OSError):
            pass
        self.process.join(timeout=10)
        if self.process.is_alive():
            self.process.terminate()

    def _collect(self) -> None:
        """Read every message the detector process has sent so far."""
        while self.conn.poll():
            message = self.conn.recv()
            if message == "ready":
                self.ready = True
            elif isinstance(message, str):          # the process reported an error
                raise RuntimeError(message)
            else:
                self.result, self.busy = message, False
                self.meter.tick()


def _detector_process(cfg: dict, conn) -> None:
    """Runs in the separate detector process: load the model, then answer requests."""
    os.nice(10)                        # low priority: the video process comes first
    try:
        detector = Detector(cfg)
        pool = ThreadPoolExecutor(max_workers=3) if cfg.get("live", {}).get("parallel_cams") else None
        conn.send("ready")
        while True:
            images = conn.recv()
            if images is None:
                return
            if pool is not None:
                conn.send(list(pool.map(detector.detect, images)))
            else:
                conn.send([detector.detect(image) for image in images])
    except Exception as error:
        conn.send(f"detector process failed: {type(error).__name__}: {error}")


def _to_detections(players: list[dict]) -> sv.Detections:
    """Convert our JSON player list back into a supervision Detections object."""
    if not players:
        return sv.Detections.empty()
    return sv.Detections(
        xyxy=np.array([p["box"] for p in players], dtype=np.float32),
        confidence=np.array([p["conf"] for p in players], dtype=np.float32),
        class_id=np.zeros(len(players), dtype=int),
    )
