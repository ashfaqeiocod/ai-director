"""ball_tracker.py — Step 5: smooth the ball position for ONE camera.

The detector sometimes misses the ball, or fires on something that is not the ball
(a white sock, a line). The BallTracker cleans this up frame by frame:

- It remembers the last N ball centers.
- A detection that jumps too far from the recent average is rejected as an outlier.
- If the ball is missing, it guesses where the ball is from its last speed
  ("predicted") for a few frames, then gives up ("lost").

Every frame gets one status: "detected", "predicted" or "lost".
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

DETECTED, PREDICTED, LOST = "detected", "predicted", "lost"


@dataclass
class BallState:
    status: str                              # detected / predicted / lost
    center: tuple[float, float] | None       # (x, y) in pixels of this camera, None if lost
    rejected: bool = False                   # True if a detection was thrown away as an outlier


class BallTracker:
    def __init__(self, buffer_size: int, max_jump_px: float, max_predict_frames: int, frame_size: tuple[int, int]):
        self.buffer: deque[np.ndarray] = deque(maxlen=buffer_size)
        self.max_jump_px = max_jump_px
        self.max_predict_frames = max_predict_frames
        self.width, self.height = frame_size
        self.velocity = np.zeros(2)          # pixels per frame
        self.last_center: np.ndarray | None = None
        self.missed = 0                      # frames since the last accepted detection

    def update(self, ball_box: list[float] | None) -> BallState:
        """Feed this frame's ball box (x1, y1, x2, y2) or None; get the ball state back."""
        rejected = False
        if ball_box is not None:
            center = np.array([(ball_box[0] + ball_box[2]) / 2, (ball_box[1] + ball_box[3]) / 2])
            if self._is_outlier(center):
                rejected = True
            else:
                self._accept(center)
                return BallState(DETECTED, (float(center[0]), float(center[1])))

        return self._predict(rejected)

    def _is_outlier(self, center: np.ndarray) -> bool:
        """Too far from the average of recent positions? Then it's probably not the ball.

        The allowed distance grows with the number of missed frames, because a ball
        we haven't seen for a while may have moved further.
        """
        if not self.buffer:
            return False
        average = np.mean(self.buffer, axis=0)
        allowed = self.max_jump_px * (1 + self.missed)
        return float(np.linalg.norm(center - average)) > allowed

    def _accept(self, center: np.ndarray) -> None:
        if self.last_center is not None:
            # Speed = distance moved / frames it took.
            self.velocity = (center - self.last_center) / (self.missed + 1)
        self.last_center = center
        self.buffer.append(center)
        self.missed = 0

    def _predict(self, rejected: bool) -> BallState:
        """No usable detection this frame: guess from the last speed, or give up."""
        self.missed += 1
        if self.last_center is None or self.missed > self.max_predict_frames:
            self._reset()
            return BallState(LOST, None, rejected)
        guess = self.last_center + self.velocity * self.missed
        if not (0 <= guess[0] < self.width and 0 <= guess[1] < self.height):
            # The ball has rolled out of this camera's view.
            self._reset()
            return BallState(LOST, None, rejected)
        return BallState(PREDICTED, (float(guess[0]), float(guess[1])), rejected)

    def _reset(self) -> None:
        """Forget the history, so the next detection is accepted fresh."""
        self.buffer.clear()
        self.last_center = None
        self.velocity = np.zeros(2)
        self.missed = 0
