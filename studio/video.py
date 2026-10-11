"""video.py — small video helpers for MVP 3: an H.264 writer fed frame by frame, and a crop helper.

OpenCV's own writer only makes mp4v (browsers can't play it), so we pipe raw frames into ffmpeg instead.
We use libx264 (software). The Mac's hardware encoder (VideoToolbox) refused to open several sessions
at once here ("cannot create compression session"), and libx264 works on every machine.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np

from src.common import ffmpeg_exe


class VideoWriter:
    """Write BGR frames (all the same size) to an H.264 mp4, one frame at a time."""

    def __init__(self, path: Path, width: int, height: int, fps: float):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.part = path.with_name(path.stem + ".part.mp4")    # renamed when complete
        encoder = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-threads", "2"]
        cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", *encoder, "-pix_fmt", "yuv420p",
               "-g", str(round(fps)), "-movflags", "+faststart", str(self.part)]
        self.size = (width, height)
        self.process = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        if (frame.shape[1], frame.shape[0]) != self.size:
            raise ValueError(f"frame is {frame.shape[1]}x{frame.shape[0]}, writer expects {self.size}")
        self.process.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        self.process.stdin.close()
        if self.process.wait() != 0:
            raise RuntimeError(f"ffmpeg failed writing {self.path}")
        self.part.replace(self.path)

    def abort(self) -> None:
        self.process.stdin.close()
        self.process.wait()
        self.part.unlink(missing_ok=True)


def crop_to(image: np.ndarray, rect: tuple[float, float, float, float], width: int, height: int) -> np.ndarray:
    """Cut rect (x, y, w, h in source pixels) out of image and fit it into width x height.

    If the rect has another shape than the output, it is letterboxed (black bars), never stretched.
    Parts of the rect outside the image are black.
    """
    x, y, w, h = (int(round(v)) for v in rect)
    H, W = image.shape[:2]
    piece = np.zeros((h, w, 3), dtype=image.dtype)
    sx1, sy1, sx2, sy2 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if sx2 > sx1 and sy2 > sy1:
        piece[sy1 - y:sy2 - y, sx1 - x:sx2 - x] = image[sy1:sy2, sx1:sx2]
    scale = min(width / w, height / h)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(piece, (new_w, new_h), interpolation=interpolation)
    if (new_w, new_h) == (width, height):
        return resized
    canvas = np.zeros((height, width, 3), dtype=image.dtype)
    ox, oy = (width - new_w) // 2, (height - new_h) // 2
    canvas[oy:oy + new_h, ox:ox + new_w] = resized
    return canvas


def to_source(rect: tuple[float, float, float, float], width: int, height: int,
              point: tuple[float, float]) -> tuple[float, float]:
    """A point in a camera picture (width x height) -> the same point in the source (inverse of crop_to)."""
    x, y, w, h = rect
    scale = min(width / w, height / h)
    ox, oy = (width - w * scale) / 2, (height - h * scale) / 2
    return x + (point[0] - ox) / scale, y + (point[1] - oy) / scale


def to_camera(rect: tuple[float, float, float, float], width: int, height: int,
              point: tuple[float, float]) -> tuple[float, float]:
    """A point in the source -> the same point in a camera picture (what crop_to does to pixels)."""
    x, y, w, h = rect
    scale = min(width / w, height / h)
    ox, oy = (width - w * scale) / 2, (height - h * scale) / 2
    return ox + (point[0] - x) * scale, oy + (point[1] - y) * scale
