"""prepare.py — MVP 3 Step 1: cut a segment of a SoccerTrack half and check that annotations line up.

  venv/bin/python -m studio.prepare --segment dev

Writes data/studio/<segment>/:
  source.mp4        the 4K panorama for this segment only (frame-exact, 25 fps, H.264)
  segment.json      where the segment sits in the half (first frame), size, fps, frame count, audio yes/no
  pitch_to_image.npy  a 3x3 mapping from pitch metres to video pixels (fitted from the player annotations;
                      used ONLY to score where the ground-truth ball was)
  check_*.png       3 frames with the annotated players (green) and the ground-truth ball (yellow) drawn on

Frame numbers: segment frame i (0-based) = video frame first_frame + i of the half (0-based)
             = annotation frame first_frame + i + 1 + offset (the annotations count from 1).
`offset` is MEASURED here as a safety check (annotation_offset in segment.json): prepare tries offsets and
keeps the one where the annotated boxes cover the most "not grass" (= players). It should be ~0: the dataset's
annotations are in sync with its video. (The first dev cut was 24 frames early because the video stream starts
at timestamp 1.0 s and the fast seek did not allow for it; the measured offset of -23 corrected that segment.)
Use annotation_frame() / segment_time() everywhere, never first_frame directly.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

from src.common import ROOT, ffmpeg_exe, load_config, video_info
from studio.soccertrack import HALF_NAMES, data_dir, load_events, load_players


def segment_cfg(name: str) -> dict:
    segments = load_config()["studio"]["segments"]
    if name not in segments:
        raise SystemExit(f"unknown segment '{name}' (config.yaml studio.segments: {', '.join(segments)})")
    return segments[name]


def work_dir(name: str) -> Path:
    return ROOT / load_config()["studio"]["work_dir"] / name


def half_video(seg: dict) -> Path:
    return data_dir() / f"videos/{seg['match']}_{HALF_NAMES[seg['half']][0]}.mp4"


def load_segment(name: str) -> dict:
    """segment.json written by prepare (raises if prepare hasn't run)."""
    meta = work_dir(name) / "segment.json"
    if not meta.exists():
        raise SystemExit(f"segment '{name}' is not prepared: run  venv/bin/python -m studio.prepare --segment {name}")
    return json.loads(meta.read_text())


def annotation_frame(meta: dict, i: int) -> int:
    """The annotation frame number (gsr / ball / bas) that shows the same moment as segment frame i."""
    return meta["first_frame"] + i + 1 + meta.get("annotation_offset", 0)


def segment_time(meta: dict, annotation_frame_no: int) -> float:
    """Seconds into the segment of an annotation frame number (inverse of annotation_frame)."""
    return (annotation_frame_no - meta["first_frame"] - 1 - meta.get("annotation_offset", 0)) / meta["fps"]


def measure_offset(meta: dict, players: dict, video: Path, samples: int = 8, search: int = 75) -> tuple[int, float, float]:
    """Find the annotation time offset (frames). Returns (offset, fit at offset, fit at 0).

    Fit = average share of "not grass" pixels inside the annotated boxes: high when the boxes sit on players.
    """
    order = np.argsort(players["frame"], kind="stable")
    frames, boxes = players["frame"][order], np.stack([players[k][order] for k in ("x1", "y1", "x2", "y2")], 1)
    cap = cv2.VideoCapture(str(video))
    images = []
    for share in np.linspace(0.05, 0.95, samples):
        i = int(meta["frames"] * share)
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, image = cap.read()
        if ok:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            grass = (hsv[..., 0] > 30) & (hsv[..., 0] < 90) & (hsv[..., 1] > 60)
            images.append((i, grass))
    cap.release()

    def fit(offset: int) -> float:
        scores = []
        for i, grass in images:
            no = meta["first_frame"] + i + 1 + offset
            lo, hi = np.searchsorted(frames, [no, no + 1])
            for x1, y1, x2, y2 in boxes[lo:hi].astype(int):
                piece = grass[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
                if piece.size:
                    scores.append(1 - piece.mean())
        return float(np.mean(scores)) if scores else 0.0

    coarse = max(range(-search, search + 1, 5), key=fit)
    best = max(range(coarse - 5, coarse + 6), key=fit)
    return best, fit(best), fit(0)


def stream_start(video: Path) -> float:
    """Timestamp (s) of the video stream's first frame."""
    import av

    with av.open(str(video)) as c:
        st = c.streams.video[0]
        return float(st.start_time * st.time_base) if st.start_time else 0.0


def cut_segment(name: str, seg: dict, out: Path) -> dict:
    """Re-encode [start_s, end_s) of the half into out/source.mp4, frame-exact."""
    src = half_video(seg)
    info = video_info(src)
    fps = info["fps"]
    first = round(seg["start_s"] * fps)
    count = round((seg["end_s"] - seg["start_s"]) * fps)
    out.mkdir(parents=True, exist_ok=True)
    video = out / "source.mp4"
    print(f"[prepare] cutting {src.name} frames {first}..{first + count - 1} ({count / fps / 60:.1f} min) ...", flush=True)
    # Frame k of the half has the timestamp start + k / fps (this video's stream starts at 1.0 s, not 0).
    # -copyts keeps those original timestamps, so "select" can pick exactly [first, first + count) by time,
    # whatever frame the fast seek (-ss before -i, 5 s early) lands on. -frames:v stops decoding afterwards.
    t0 = stream_start(src) + first / fps
    half_frame = 0.5 / fps
    jump = max(0.0, t0 - 5)
    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", "-ss", f"{jump:.3f}", "-copyts", "-i", str(src),
           "-vf", f"select='between(t\\,{t0 - half_frame:.4f}\\,{t0 + (count - 0.5) / fps:.4f})',setpts=N/{fps}/TB",
           "-frames:v", str(count), "-r", str(fps), "-fps_mode", "cfr", "-output_ts_offset", "0",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p", "-g", str(round(fps)),
           "-movflags", "+faststart"]
    cmd += ["-af", f"atrim=start={t0:.4f}:duration={count / fps:.4f},asetpts=PTS-STARTPTS", "-c:a", "aac",
            "-t", f"{count / fps:.4f}"] if info["has_audio"] else ["-an"]
    subprocess.run(cmd + [str(video)], check=True)
    got = video_info(video)
    meta = {"segment": name, "match": seg["match"], "half": seg["half"], "first_frame": first,
            "frames": got["frames"], "fps": got["fps"], "width": got["width"], "height": got["height"],
            "has_audio": got["has_audio"], "start_s": first / fps, "end_s": (first + got["frames"]) / fps}
    if got["frames"] != count:
        print(f"[prepare] WARNING: expected {count} frames, got {got['frames']}")
    (out / "segment.json").write_text(json.dumps(meta, indent=2))
    print(f"[prepare] source.mp4: {got['width']}x{got['height']} @ {got['fps']} fps, {got['frames']} frames, "
          f"audio: {'yes' if got['has_audio'] else 'no'}")
    return meta


def fit_pitch_to_image(players: dict, first: int, count: int) -> tuple[np.ndarray, float]:
    """Fit pitch metres -> pixels from the annotated players' feet (both are given for every player).

    Returns the 3x3 matrix and the median error in pixels. The lens has a little distortion, so this
    is approximate; it is only used for SCORING (was the ground-truth ball inside the live camera's view?).
    """
    sel = (players["frame"] > first) & (players["frame"] <= first + count) & np.isfinite(players["pitch_x"])
    idx = np.flatnonzero(sel)[:: max(1, int(sel.sum()) // 20000)]
    pitch = np.stack([players["pitch_x"][idx], players["pitch_y"][idx]], axis=1).astype(np.float32)
    feet = np.stack([(players["x1"][idx] + players["x2"][idx]) / 2, players["y2"][idx]], axis=1).astype(np.float32)
    matrix, _ = cv2.findHomography(pitch, feet, cv2.RANSAC, 15.0)
    error = np.linalg.norm(project(matrix, pitch) - feet, axis=1)
    return matrix, float(np.median(error))


def project(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 3x3 mapping to an (N, 2) array of points."""
    return cv2.perspectiveTransform(points.reshape(-1, 1, 2).astype(np.float32), matrix).reshape(-1, 2)


def gt_ball(seg: dict) -> dict[str, np.ndarray]:
    """Ground-truth ball (pitch metres) of the half. SCORING ONLY: it is rebuilt from future events."""
    z = np.load(data_dir() / f"ball/{seg['match']}_{HALF_NAMES[seg['half']][0]}_ball.npz")
    return {k: z[k] for k in ("frame", "x", "y", "status")}


def draw_check(meta: dict, players: dict, matrix: np.ndarray, out: Path) -> list[Path]:
    """Save 3 frames with the annotations drawn on, to check by eye that they line up."""
    ball = gt_ball(meta)
    cap = cv2.VideoCapture(str(out / "source.mp4"))
    paths = []
    for share in (0.1, 0.5, 0.9):
        i = int(meta["frames"] * share)
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, image = cap.read()
        if not ok:
            continue
        frame_no = annotation_frame(meta, i)
        rows = np.flatnonzero(players["frame"] == frame_no)
        for r in rows:
            cv2.rectangle(image, (int(players["x1"][r]), int(players["y1"][r])),
                          (int(players["x2"][r]), int(players["y2"][r])), (80, 220, 80), 2)
        b = np.flatnonzero(ball["frame"] == frame_no)
        if len(b):
            x, y = project(matrix, np.array([[ball["x"][b[0]], ball["y"][b[0]]]]))[0]
            cv2.circle(image, (int(x), int(y)), 14, (0, 230, 255), 3)
        cv2.putText(image, f"segment frame {i} = annotation frame {frame_no}  ({len(rows)} people)",
                    (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
        path = out / f"check_{i:05d}.png"
        cv2.imwrite(str(path), image)
        paths.append(path)
    cap.release()
    return paths


def prepare(name: str, force: bool = False) -> dict:
    seg = segment_cfg(name)
    out = work_dir(name)
    if (out / "segment.json").exists() and not force:
        meta = load_segment(name)
        print(f"[prepare] {name} already prepared ({meta['frames']} frames); --force to redo")
    else:
        meta = cut_segment(name, seg, out)
    players = load_players(seg["match"], seg["half"])
    if "annotation_offset" not in meta or force:
        offset, fit_best, fit_zero = measure_offset(meta, players, out / "source.mp4")
        meta["annotation_offset"] = offset
        (out / "segment.json").write_text(json.dumps(meta, indent=2))
        print(f"[prepare] annotation offset: {offset} frames ({offset / meta['fps']:+.2f} s); "
              f"boxes on players {fit_best:.2f} vs {fit_zero:.2f} without it")
    matrix, error = fit_pitch_to_image(players, annotation_frame(meta, 0) - 1, meta["frames"])
    np.save(out / "pitch_to_image.npy", matrix)
    print(f"[prepare] pitch->image mapping: median error {error:.1f} px (scoring only)")
    for p in draw_check(meta, players, matrix, out):
        print(f"[prepare] check picture: {p.relative_to(ROOT)}")
    goals = [e for e in load_events(seg["match"], seg["half"])
             if e["label"] == "GOAL" and 0 <= segment_time(meta, e["frame"]) < meta["frames"] / meta["fps"]]
    print(f"[prepare] ground-truth goals in this segment: {len(goals)} "
          f"(at {', '.join(f'{segment_time(meta, g['frame']):.1f} s' for g in goals)})")
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--segment", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    prepare(args.segment, args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
