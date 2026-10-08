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

import time
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


class Engine:
    def __init__(self, cfg: dict, mode: str = REPLAY, detector: Detector | None = None):
        self.cfg = cfg
        self.mode = mode
        self.num_cams = cfg["video"]["num_cams"]
        self.info = video_info(cam_path(cfg, 1))
        self.fps = self.info["fps"]
        self.total_frames = self.info["frames"]
        self._stop = False

        if mode == LIVE:
            # Reuse a detector if the caller already loaded one (models are slow to load).
            self.detector = detector or Detector(cfg)
            self.cache = None
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
        tick = time.time()
        fps_avg = 0.0

        for index, images in enumerate(zip(*readers)):
            if self._stop:
                break
            cams = []
            for c, image in enumerate(images):
                det = self._detections(c, index, image)
                if self.mode == LIVE:
                    live_cache[c].append(det)
                cams.append(self._track(c, image, det))

            decision = self.director.decide(index, [cm.ball for cm in cams], [cm.num_players for cm in cams],
                                            self.info["width"])
            # Smoothed processing speed (frames per second), shown honestly in the dashboard.
            now = time.time()
            instant = 1.0 / max(now - tick, 1e-6)
            fps_avg = instant if index == 0 else 0.9 * fps_avg + 0.1 * instant
            tick = now

            yield FrameResult(index, index / self.fps, cams, decision.scores, decision.cam, decision.reason,
                              decision.events, round(fps_avg, 1))

        # A complete live run becomes the cache, so replay mode can use it later.
        if self.mode == LIVE and not self._stop and len(live_cache[0]) == self.total_frames:
            for c in range(self.num_cams):
                info = video_info(cam_path(self.cfg, c + 1))
                save_detections(detections_path(self.cfg, c + 1), c + 1, info, self.detector.describe(), live_cache[c])

    def _detections(self, cam_index: int, frame_index: int, image: np.ndarray) -> dict:
        if self.mode == LIVE:
            return self.detector.detect(image)
        frames = self.cache[cam_index]
        return frames[frame_index] if frame_index < len(frames) else {"players": [], "ball": None}

    def _track(self, cam_index: int, image: np.ndarray, det: dict) -> CamResult:
        ball_box = det["ball"]["box"] if det["ball"] else None
        ball = self.ball_trackers[cam_index].update(ball_box)
        players = _to_detections(det["players"])
        tracked = self.player_trackers[cam_index].update_with_detections(players)
        return CamResult(image, tracked, len(det["players"]), ball_box, ball)


def _to_detections(players: list[dict]) -> sv.Detections:
    """Convert our JSON player list back into a supervision Detections object."""
    if not players:
        return sv.Detections.empty()
    return sv.Detections(
        xyxy=np.array([p["box"] for p in players], dtype=np.float32),
        confidence=np.array([p["conf"] for p in players], dtype=np.float32),
        class_id=np.zeros(len(players), dtype=int),
    )
