"""render.py — Arm MVP Step 10: turn a list of shot decisions into a broadcast video with sound.

For every output frame (30 per second):
  - which shot is on air: the decision of the look it belongs to (10 looks per second, 3 frames each)
  - the virtual operator: the crop moves smoothly toward the shot's framing (eased, speed-limited), like a
    robotic camera head; during tense moments it slowly pushes in a little
  - the picture: the crop cut out of that camera's synced video; camera A's close-ups come from its 4K copy
  - transitions: CUT (next frame is the new shot), DISSOLVE (0.4 s cross-fade), MOVE (the same camera reframes:
    the operator glides from the old framing to the new one)
  - graphics (optional): camera tag, a LIVE / EDIT badge, and a lower third with the phase, a tension meter
    and the latest moment ("TAN LOOKS UP")
Sound: ONE continuous track from one phone (config arm.render.audio_cam): sound never jumps between phones.

Usage: called by arm/main.py (render_many). A decision needs .shot (cam, kind, rect, key) and .change.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from arm.common import athlete_word, cam_letters, settings, synced_video
from src.common import ffmpeg_exe
from studio.video import VideoWriter

EVENT_TEXT = {"LOOK_UP": "{who} LOOKS UP", "GESTURE": "{who} WAVES THE CROWD ON", "CHEER": "THE CROWD CHEERS",
              "SURGE": "{who} PUSHES", "NEAR_PIN": "{who} CLOSE TO A PIN", "PIN": "{who} WINS!",
              "REACTION": "{who} CELEBRATES"}


class Operator:
    """The virtual camera operator: moves the crop toward its target, smoothly and not too fast."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.rect: np.ndarray | None = None

    def jump(self, rect) -> None:
        self.rect = np.array(rect, dtype=float)

    def follow(self, target, fast: bool = False) -> np.ndarray:
        target = np.array(target, dtype=float)
        if self.rect is None:
            self.rect = target.copy()
        ease = self.cfg["move_ease"] if fast else self.cfg["follow_ease"]
        step = (target - self.rect) * ease
        limit = self.cfg["max_px_per_frame"] * (3 if fast else 1)
        step = np.clip(step, -limit, limit)
        self.rect = self.rect + step
        return self.rect


def keep_16x9(rect: np.ndarray, min_h: float) -> tuple[float, float, float, float]:
    """Keep the crop 16:9, at least min_h tall and inside the 1920x1080 picture."""
    x, y, w, h = rect
    h = float(np.clip(h, min_h, 1080)); w = h * 16 / 9
    cx, cy = x + rect[2] / 2, y + rect[3] / 2
    x = float(np.clip(cx - w / 2, 0, 1920 - w)); y = float(np.clip(cy - h / 2, 0, 1080 - h))
    return x, y, w, h


def push_in(rect, tension: float, amount: float, min_h: float) -> tuple[float, float, float, float]:
    """Tighten the framing by up to `amount` (share of its size) as tension rises: the slow push-in of TV."""
    x, y, w, h = rect
    k = 1 - amount * float(np.clip(tension, 0, 1))
    return keep_16x9(np.array([x + w * (1 - k) / 2, y + h * (1 - k) / 2, w * k, h * k]), min_h)


def crop(frames: dict[str, np.ndarray], full: dict[str, np.ndarray], cam: str, rect, size: tuple[int, int]) -> np.ndarray:
    """Cut the rect (1920x1080 pixels) out of the camera's frame and scale it to the output size.

    A tight crop of a camera that has a full-resolution copy is cut from that copy (sharper)."""
    x, y, w, h = rect
    source = frames[cam]
    scale = 1.0
    if cam in full and h < 1080 * 0.8:
        source = full[cam]
        scale = source.shape[0] / 1080
    x1, y1 = int(round(x * scale)), int(round(y * scale))
    x2, y2 = int(round((x + w) * scale)), int(round((y + h) * scale))
    piece = source[max(0, y1):y2, max(0, x1):x2]
    interp = cv2.INTER_AREA if piece.shape[0] > size[1] else cv2.INTER_CUBIC
    return cv2.resize(piece, size, interpolation=interp)


def draw_graphics(img: np.ndarray, shot, state: dict, event: str | None, badge: str) -> None:
    """Camera tag, badge, lower third (phase + tension meter + latest moment). Drawn in place."""
    H, W = img.shape[:2]
    s = W / 1920
    font = cv2.FONT_HERSHEY_DUPLEX
    tag = f"CAM {shot.cam}  {shot.kind.replace('_', ' ')}"
    overlay = img.copy()
    cv2.rectangle(overlay, (int(30 * s), int(30 * s)), (int((60 + 22 * len(tag)) * s), int(80 * s)), (20, 20, 20), -1)
    cv2.rectangle(overlay, (W - int(190 * s), int(30 * s)), (W - int(30 * s), int(80 * s)), (40, 40, 200), -1)
    # Lower third.
    y0 = H - int(150 * s)
    cv2.rectangle(overlay, (int(60 * s), y0), (int(760 * s), y0 + int(90 * s)), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    cv2.putText(img, tag, (int(45 * s), int(66 * s)), font, 0.9 * s, (255, 255, 255), max(1, int(2 * s)), cv2.LINE_AA)
    cv2.putText(img, badge, (W - int(175 * s), int(66 * s)), font, 0.9 * s, (255, 255, 255), max(1, int(2 * s)), cv2.LINE_AA)
    phase = state["phase"].replace("_", " ")
    cv2.putText(img, f"ARM WRESTLING  |  {phase}", (int(80 * s), y0 + int(35 * s)), font, 0.85 * s, (255, 255, 255),
                max(1, int(2 * s)), cv2.LINE_AA)
    # Tension meter.
    bx, by, bw, bh = int(80 * s), y0 + int(55 * s), int(300 * s), int(18 * s)
    cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (90, 90, 90), -1)
    filled = int(bw * float(np.clip(state["tension"], 0, 1)))
    colour = (60, 200, 255) if state["tension"] < 0.6 else (40, 80, 240)
    cv2.rectangle(img, (bx, by), (bx + filled, by + bh), colour, -1)
    cv2.putText(img, "TENSION", (bx + bw + int(15 * s), by + bh - int(2 * s)), font, 0.6 * s, (220, 220, 220),
                max(1, int(1 * s)), cv2.LINE_AA)
    if event:
        cv2.putText(img, event, (int(500 * s), by + bh - int(2 * s)), font, 0.7 * s, (120, 230, 255),
                    max(1, int(2 * s)), cv2.LINE_AA)


def event_caption(events: list[tuple[float, str]], t: float, hold_s: float) -> str | None:
    """The newest event of the last `hold_s` seconds, as a caption."""
    for when, name in reversed(events):
        if 0 <= t - when <= hold_s:
            word, _, who = name.partition(" ")
            text = EVENT_TEXT.get(word)
            if text and ("{who}" not in text or who):
                return text.format(who=athlete_word(who))
    return None


@dataclass
class Program:
    """One video to write: its decisions (one per look), the output file, and how it looks."""
    decisions: list
    out: Path
    graphics: bool = True
    badge: str = "AI LIVE"
    size: tuple[int, int] = (1920, 1080)
    push_in: bool = True              # the slow push-in on tension (off for the plain-camera baselines)


class ProgramState:
    """Everything one program remembers between frames: the operator, the shot on air, a running dissolve."""

    def __init__(self, program: Program, cfg: dict):
        self.p = program
        self.operator = Operator(cfg)
        self.on_air = None
        self.fade: tuple | None = None       # (old cam, old rect, frames left) during a dissolve
        self.move_left = 0                   # frames left of a MOVE (the same camera reframing)
        self.last_rect = None
        silent = program.out.with_name(program.out.stem + ".video.mp4")
        self.writer = VideoWriter(silent, *program.size, settings()["sync"]["fps"], crf=cfg["crf"], threads=cfg["threads"])

    def frame(self, n: int, frames: dict, full: dict, states: list, events: list) -> None:
        cfg = settings()["render"]
        fps = settings()["sync"]["fps"]
        step = round(fps / settings()["perceive"]["analysis_fps"])
        max_zoom = settings()["shots"]["max_zoom"]
        k = min(n // step, len(self.p.decisions) - 1)
        d = self.p.decisions[k]
        state = states[min(k, len(states) - 1)]
        min_h = 1080 / max_zoom.get(d.shot.cam, max_zoom["default"])
        target = push_in(d.shot.rect, state["tension"], cfg["push_in"], min_h) if self.p.push_in else d.shot.rect
        if self.on_air is None or d.shot.key != self.on_air.key:
            if self.on_air is not None and d.change == "DISSOLVE":
                self.fade = (self.on_air.cam, self.last_rect, round(cfg["dissolve_s"] * fps))
            if self.on_air is None or d.shot.cam != self.on_air.cam:
                self.operator.jump(target)                 # cut / dissolve: the new camera is already framed
            else:
                self.move_left = round(cfg["move_s"] * fps)  # same camera: glide to the new framing
            self.on_air = d.shot
        rect = keep_16x9(self.operator.follow(target, fast=self.move_left > 0), min_h)
        self.move_left = max(0, self.move_left - 1)
        self.last_rect = rect
        picture = crop(frames, full, d.shot.cam, rect, self.p.size)
        if self.fade is not None:
            old_cam, old_rect, left = self.fade
            alpha = 1 - left / round(cfg["dissolve_s"] * fps)
            old = crop(frames, full, old_cam, old_rect, self.p.size)
            picture = cv2.addWeighted(old, 1 - alpha, picture, alpha, 0)
            self.fade = (old_cam, old_rect, left - 1) if left > 1 else None
        if self.p.graphics:
            draw_graphics(picture, d.shot, state, event_caption(events, n / fps, cfg["caption_s"]), self.p.badge)
        self.writer.write(picture)


def render_many(programs: list[Program], timeline: dict) -> list[Path]:
    """Write all programs in ONE pass over the 5 cameras (decoding the videos is the slow part)."""
    cfg = settings()["render"]
    states = timeline["states"]
    events = [(s["t"], e) for s in states for e in s["events"]]
    caps = {c: cv2.VideoCapture(str(synced_video(c))) for c in cam_letters()}
    full_caps = {c: cv2.VideoCapture(str(synced_video(c, full_res=True))) for c in settings()["sync"]["keep_full_res"]
                 if synced_video(c, full_res=True).exists()}
    total = int(min(cap.get(cv2.CAP_PROP_FRAME_COUNT) for cap in caps.values()))
    runs = [ProgramState(p, cfg) for p in programs]
    try:
        for n in range(total):
            frames = {c: cap.read()[1] for c, cap in caps.items()}
            full = {c: cap.read()[1] for c, cap in full_caps.items()}
            for run in runs:
                run.frame(n, frames, full, states, events)
        for run in runs:
            run.writer.close()
    except Exception:
        for run in runs:
            run.writer.abort()
        raise
    finally:
        for cap in [*caps.values(), *full_caps.values()]:
            cap.release()
    for run in runs:
        add_audio(run.writer.path, synced_video(cfg["audio_cam"]), run.p.out)
        run.writer.path.unlink(missing_ok=True)
    return [p.out for p in programs]


def add_audio(video: Path, audio_from: Path, out: Path) -> None:
    """Copy the picture and put the audio of `audio_from` under it (both start at the synced 0)."""
    subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(video), "-i", str(audio_from),
                    "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest",
                    "-movflags", "+faststart", str(out)], check=True)
