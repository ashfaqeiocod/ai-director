"""render.py — Step 7: turn engine results into pictures and videos.

For every FrameResult from the engine, the Renderer builds:
  - the PROGRAM frame: the chosen camera, letterboxed to 1280x720, with a "CAM n" label;
  - 3 annotated camera tiles: player boxes + track IDs, ball marker + trail,
    red border and "LIVE" on the chosen camera;
  - the DEBUG frame: the 3 tiles side by side + a bar with timecode and scores.

It writes outputs/program.mp4 and outputs/debug.mp4 while the engine runs.
OpenCV can only write a codec browsers don't like (mp4v), so at the end ffmpeg
re-encodes both to H.264 and adds the original audio to program.mp4.
The dashboard calls the same Renderer, so it shows exactly what the videos contain.
"""
from __future__ import annotations

import subprocess
from collections import deque

import cv2
import numpy as np
import supervision as sv

from src.ball_tracker import DETECTED, PREDICTED
from src.common import ffmpeg_exe, path, video_info
from src.engine import FrameResult

# Colors are BGR (OpenCV order).
RED = (40, 40, 230)
GREEN = (80, 220, 80)        # ball detected
ORANGE = (0, 160, 255)       # ball predicted
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def letterbox(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Fit an image inside width x height without stretching; fill the rest with black."""
    h, w = image.shape[:2]
    scale = min(width / w, height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x, y = (width - new_w) // 2, (height - new_h) // 2
    canvas[y:y + new_h, x:x + new_w] = resized
    return canvas


def draw_label(image: np.ndarray, text: str, origin: tuple[int, int], scale: float = 1.0,
               color=WHITE, background=BLACK) -> None:
    """Text on a filled box, so it is readable on any background."""
    font, thickness = cv2.FONT_HERSHEY_SIMPLEX, max(1, int(2 * scale))
    (tw, th), base = cv2.getTextSize(text, font, scale, thickness)
    x, y = origin
    cv2.rectangle(image, (x - 6, y - th - 8), (x + tw + 6, y + base + 4), background, -1)
    cv2.putText(image, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def format_time(seconds: float) -> str:
    return f"{int(seconds // 60):02d}:{seconds % 60:05.2f}"


class Renderer:
    def __init__(self, cfg: dict, fps: float, num_cams: int = 3, write_videos: bool = True):
        self.cfg = cfg
        self.rcfg = cfg["render"]
        self.fps = fps
        self.num_cams = num_cams
        self.trails = [deque(maxlen=self.rcfg["trail_length"]) for _ in range(num_cams)]
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_scale=0.5, text_padding=3)
        self.write_videos = write_videos
        self.program_writer = self.debug_writer = None
        self.program_tmp = path(self.rcfg["program_path"]).with_name(".program_raw.mp4")
        self.debug_tmp = path(self.rcfg["debug_path"]).with_name(".debug_raw.mp4")

    # ---------- per frame ----------
    def render(self, result: FrameResult) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
        """Build (program frame, annotated cam tiles, debug frame) and write them to the videos."""
        tiles = [self.annotate_cam(result, c) for c in range(self.num_cams)]
        program = self.program_frame(result)
        debug = self.debug_frame(result, tiles)
        if self.write_videos:
            self._write(program, debug)
        return program, tiles, debug

    def program_frame(self, result: FrameResult) -> np.ndarray:
        raw = result.cams[result.chosen - 1].frame
        frame = letterbox(raw, self.rcfg["program_width"], self.rcfg["program_height"])
        draw_label(frame, f"CAM {result.chosen}", (24, 48), 1.0)
        return frame

    def annotate_cam(self, result: FrameResult, cam_index: int) -> np.ndarray:
        cam = result.cams[cam_index]
        image = cam.frame.copy()

        # Players: boxes and "#track id" labels.
        players = cam.players
        if len(players) > 0:
            image = self.box_annotator.annotate(image, players)
            if players.tracker_id is not None:
                labels = [f"#{t}" for t in players.tracker_id]
                image = self.label_annotator.annotate(image, players, labels=labels)

        # Ball: trail (older points are smaller) plus a marker. Green = detected, orange = predicted.
        trail = self.trails[cam_index]
        trail.append((cam.ball.center, cam.ball.status) if cam.ball.center else None)
        points = [p for p in trail if p is not None]
        for i, (center, status) in enumerate(points):
            color = GREEN if status == DETECTED else ORANGE
            cv2.circle(image, (int(center[0]), int(center[1])), 2 + i // 3, color, -1)
        if cam.ball.center is not None:
            color = GREEN if cam.ball.status == DETECTED else ORANGE
            x, y = int(cam.ball.center[0]), int(cam.ball.center[1])
            cv2.circle(image, (x, y), 16, color, 3)
            cv2.circle(image, (x, y), 3, color, -1)

        draw_label(image, f"CAM {cam_index + 1}", (16, 42), 1.0)
        status_color = {DETECTED: GREEN, PREDICTED: ORANGE}.get(cam.ball.status, (160, 160, 160))
        draw_label(image, f"ball: {cam.ball.status}", (16, 90), 0.8, color=status_color)
        if result.chosen == cam_index + 1:
            h, w = image.shape[:2]
            cv2.rectangle(image, (0, 0), (w - 1, h - 1), RED, 12)
            draw_label(image, "LIVE", (w - 110, 46), 1.1, background=RED)
        return image

    def debug_frame(self, result: FrameResult, tiles: list[np.ndarray]) -> np.ndarray:
        """3 tiles side by side (scaled to 720 px high) and an info bar underneath."""
        tile_h = 720
        scaled = [cv2.resize(t, (int(t.shape[1] * tile_h / t.shape[0]) // 2 * 2, tile_h),
                             interpolation=cv2.INTER_AREA) for t in tiles]
        row = np.hstack(scaled)
        bar = np.zeros((70, row.shape[1], 3), dtype=np.uint8)
        text = f"{format_time(result.time)}  frame {result.index}   LIVE: CAM {result.chosen}"
        cv2.putText(bar, text, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2, cv2.LINE_AA)
        scores = "   ".join(f"cam{i + 1}: {s:5.1f}" for i, s in enumerate(result.scores))
        cv2.putText(bar, scores, (14, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
        return np.vstack([row, bar])

    # ---------- video files ----------
    def _write(self, program: np.ndarray, debug: np.ndarray) -> None:
        if self.program_writer is None:
            self.program_tmp.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.program_writer = cv2.VideoWriter(str(self.program_tmp), fourcc, self.fps, program.shape[1::-1])
            self.debug_writer = cv2.VideoWriter(str(self.debug_tmp), fourcc, self.fps, debug.shape[1::-1])
        self.program_writer.write(program)
        self.debug_writer.write(debug)

    def abort(self) -> None:
        """Run was stopped early: throw the half-written videos away (keep any older finished ones)."""
        if self.program_writer is not None:
            self.program_writer.release()
            self.debug_writer.release()
            self.program_writer = self.debug_writer = None
        self.program_tmp.unlink(missing_ok=True)
        self.debug_tmp.unlink(missing_ok=True)

    def close(self) -> None:
        """Finish the videos: re-encode to H.264 and add the source audio to program.mp4."""
        if self.program_writer is None:
            return
        self.program_writer.release()
        self.debug_writer.release()
        self.program_writer = self.debug_writer = None
        source = path(self.cfg["video"]["source"])
        program_out, debug_out = path(self.rcfg["program_path"]), path(self.rcfg["debug_path"])

        cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(self.program_tmp)]
        if video_info(source)["has_audio"]:
            # Video from our render, audio copied untouched from the source clip.
            cmd += ["-i", str(source), "-map", "0:v", "-map", "1:a", "-c:a", "copy", "-shortest"]
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", str(program_out)]
        subprocess.run(cmd, check=True)
        subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(self.debug_tmp),
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart", str(debug_out)], check=True)
        self.program_tmp.unlink(missing_ok=True)
        self.debug_tmp.unlink(missing_ok=True)
