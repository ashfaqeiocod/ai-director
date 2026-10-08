"""director.py — Step 6: pick which camera is "live" on every frame.

Each frame, every camera gets a score:
  ball detected            +100
  ball predicted           +60
  ball near the horizontal center of that camera   +0..+30
  players visible          +1 each, max +10

Rules that stop the output from flickering:
  - Minimum shot length: never cut sooner than 2 s after the previous cut.
  - Hysteresis: another camera must beat the live one for 0.5 s in a row before we cut.
  - If no camera sees the ball: stay where we are; after 2 s lost, go to the fallback cam (cam2).

All numbers come from config.yaml ("director" section).
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from src.ball_tracker import DETECTED, PREDICTED, BallState


@dataclass
class Decision:
    cam: int                    # chosen camera (1, 2 or 3)
    scores: list[float]         # one score per camera
    reason: str                 # plain-English reason, goes into the CSV log
    events: list[dict]          # CUT / BALL_LOST / BALL_FOUND events that happened on this frame


def score_camera(ball: BallState, num_players: int, cam_width: int, cfg: dict) -> float:
    """Score one camera for one frame (see table at the top of this file)."""
    score = 0.0
    if ball.status == DETECTED:
        score += cfg["score_detected"]
    elif ball.status == PREDICTED:
        score += cfg["score_predicted"]
    if ball.center is not None:
        # 1.0 when the ball is exactly in the middle, 0.0 at the left/right edge.
        offset = abs(ball.center[0] - cam_width / 2) / (cam_width / 2)
        score += cfg["score_center_max"] * max(0.0, 1.0 - offset)
    score += min(num_players * cfg["score_per_player"], cfg["score_players_max"])
    return round(score, 1)


class Director:
    def __init__(self, cfg: dict, fps: float, num_cams: int = 3):
        self.cfg = cfg
        self.fps = fps
        self.num_cams = num_cams
        self.min_shot_frames = round(cfg["min_shot_seconds"] * fps)
        self.hysteresis_frames = max(1, round(cfg["hysteresis_seconds"] * fps))
        self.lost_frames_limit = round(cfg["lost_fallback_seconds"] * fps)
        self.current = cfg["start_cam"]
        self.shot_start = 0             # frame where the current shot began
        self.challenger: int | None = None
        self.challenger_frames = 0      # how long the challenger has been winning
        self.lost_frames = 0            # how long no camera has seen the ball
        self.ball_visible = False

    def decide(self, frame_idx: int, balls: list[BallState], player_counts: list[int], cam_width: int) -> Decision:
        scores = [score_camera(b, n, cam_width, self.cfg) for b, n in zip(balls, player_counts)]
        events: list[dict] = []
        shot_len = frame_idx - self.shot_start
        can_cut = shot_len >= self.min_shot_frames
        has_ball = any(b.status in (DETECTED, PREDICTED) for b in balls)

        # Ball visibility events (across all cameras).
        if has_ball and not self.ball_visible and frame_idx > 0:
            events.append({"type": "BALL_FOUND", "message": f"ball found on cam{self._best(scores)}"})
        if not has_ball and self.ball_visible:
            events.append({"type": "BALL_LOST", "message": "ball lost on all cameras"})
        self.ball_visible = has_ball

        if has_ball:
            self.lost_frames = 0
            reason = self._follow_ball(scores, can_cut, frame_idx, events)
        else:
            self.lost_frames += 1
            reason = self._ball_lost(can_cut, frame_idx, events)
        return Decision(self.current, scores, reason, events)

    def _best(self, scores: list[float]) -> int:
        return max(range(self.num_cams), key=lambda i: scores[i]) + 1

    def _follow_ball(self, scores: list[float], can_cut: bool, frame_idx: int, events: list[dict]) -> str:
        best = self._best(scores)
        if best == self.current or scores[best - 1] <= scores[self.current - 1]:
            self.challenger, self.challenger_frames = None, 0
            return "live cam has the best score"

        # Another camera is winning: count how long it has been winning in a row.
        if best == self.challenger:
            self.challenger_frames += 1
        else:
            self.challenger, self.challenger_frames = best, 1

        if self.challenger_frames < self.hysteresis_frames:
            return f"cam{best} better, waiting (hysteresis {self.challenger_frames}/{self.hysteresis_frames})"
        if not can_cut:
            return f"cam{best} better, waiting (min shot length)"
        return self._cut(best, frame_idx, f"cam{best} scored higher for {self.cfg['hysteresis_seconds']}s", events)

    def _ball_lost(self, can_cut: bool, frame_idx: int, events: list[dict]) -> str:
        self.challenger, self.challenger_frames = None, 0
        fallback = self.cfg["fallback_cam"]
        if self.lost_frames > self.lost_frames_limit and self.current != fallback:
            if can_cut:
                return self._cut(fallback, frame_idx, f"ball lost > {self.cfg['lost_fallback_seconds']}s, fallback", events)
            return "ball lost, waiting to go to fallback (min shot length)"
        return "no ball anywhere, holding current cam"

    def _cut(self, new_cam: int, frame_idx: int, reason: str, events: list[dict]) -> str:
        shot_seconds = (frame_idx - self.shot_start) / self.fps
        events.append({"type": "CUT", "message": f"cam{self.current} -> cam{new_cam} ({reason})",
                       "from": self.current, "to": new_cam, "shot_seconds": round(shot_seconds, 2)})
        self.current = new_cam
        self.shot_start = frame_idx
        self.challenger, self.challenger_frames = None, 0
        return "CUT: " + reason


class DecisionLog:
    """Writes outputs/director_log.csv one row at a time (so it works while the engine runs)."""

    def __init__(self, csv_path: Path, num_cams: int = 3):
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(csv_path, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(["frame", "time", "chosen_cam"] + [f"score_cam{i + 1}" for i in range(num_cams)] + ["reason"])

    def write(self, frame_idx: int, time_s: float, cam: int, scores: list[float], reason: str) -> None:
        self.writer.writerow([frame_idx, f"{time_s:.3f}", cam, *scores, reason])

    def close(self) -> None:
        self.file.close()


def shot_lengths(chosen: list[int], fps: float) -> list[float]:
    """Turn a per-frame list of chosen cams into a list of shot durations in seconds."""
    if not chosen:
        return []
    lengths, run = [], 1
    for prev, cur in zip(chosen, chosen[1:]):
        if cur == prev:
            run += 1
        else:
            lengths.append(run / fps)
            run = 1
    lengths.append(run / fps)
    return lengths
