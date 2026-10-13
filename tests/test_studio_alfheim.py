"""Alfheim (full half, 3 fixed cameras): the calibration fine-tuning and the source names.

The calibration test draws a pitch with a KNOWN camera mapping, starts from a wrong one (like hand-read landmarks a
few metres off) and checks that the fine-tuning brings the drawn lines back onto the real ones.
"""
import cv2
import numpy as np

from studio.alfheim import dense, fine_tune, hint_pixels, line_distance
from studio.issia import pitch_lines
from studio.sources import source_label

# A camera high at the halfway line looking at the left third (like Alfheim cam1): pitch metres -> 1280x960 pixels.
SRC = np.float32([[-52.5, -34], [-20, -34], [-20, 34], [-52.5, 34]])
DST = np.float32([[700, 150], [1250, 170], [1100, 950], [40, 300]])
TRUE_H = cv2.getPerspectiveTransform(SRC, DST).astype(np.float64)


def picture() -> np.ndarray:
    """Dark green picture with the pitch lines drawn in white where TRUE_H puts them."""
    img = np.full((960, 1280, 3), (30, 90, 30), np.uint8)
    for line in pitch_lines():
        pts = cv2.perspectiveTransform(line[None], TRUE_H)[0].astype(np.int32)
        cv2.polylines(img, [pts], False, (255, 255, 255), 3)
    return img


def test_dense_points_are_evenly_spaced():
    pts = dense(np.array([[0.0, 0.0], [10.0, 0.0]]))
    assert len(pts) == 40
    assert np.allclose(np.diff(pts[:, 0]), 0.25)


def test_hint_pixels_follow_the_white_line():
    dist = line_distance(picture())
    near = cv2.perspectiveTransform(np.array([[[-52.5, 34], [-25, 34]]]), TRUE_H)[0]
    px = hint_pixels(dist, ["y", 34, *near[0], *near[1]])
    assert len(px) > 20
    on_pitch = cv2.perspectiveTransform(px[None], np.linalg.inv(TRUE_H))[0]
    assert np.abs(on_pitch[:, 1] - 34).max() < 0.6          # every hinted pixel is on the near touchline


def test_fine_tune_recovers_the_camera():
    dist = line_distance(picture())
    marks_pitch = np.array([[-52.5, 20.16], [-36, 20.16], [-36, -20.16], [-52.5, -20.16], [-41.5, 0.0]])
    marks_px = cv2.perspectiveTransform(marks_pitch[None], TRUE_H)[0] + np.array([[9, -6], [-7, 8], [6, 5], [-8, -4], [5, 7]])
    marks = np.c_[marks_pitch, marks_px]
    start = cv2.findHomography(marks[:, :2], marks[:, 2:], 0)[0]
    near = cv2.perspectiveTransform(np.array([[[-52.5, 34], [-25, 34]]]), TRUE_H)[0]
    H, gap = fine_tune(start, marks, dist, [["y", 34, *near[0], *near[1]]])
    # near side, far side, the empty middle (no lines there: only extrapolated), the box
    probe = np.array([[[-30.0, 30.0], [-25.0, -30.0], [-21.0, 0.0], [-40.0, 0.0]]])
    error = np.hypot(*(cv2.perspectiveTransform(probe, H)[0] - cv2.perspectiveTransform(probe, TRUE_H)[0]).T)
    before = np.hypot(*(cv2.perspectiveTransform(probe, start)[0] - cv2.perspectiveTransform(probe, TRUE_H)[0]).T)
    assert gap < 2.0
    assert error.max() < before.max() / 3                  # landmarks alone: up to ~100 px off; tuned: ~25
    assert error[[0, 1, 3]].max() < 10                      # where there are lines, a few px


def test_source_label_names_the_match():
    label = source_label("alfheim-2013-11-03-H1")
    assert "Strømsgodset" in label and "half 1" in label
