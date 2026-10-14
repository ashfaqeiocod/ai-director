"""showcase.py — one pre-rendered "how it works" video of a stadium run, for demos (MVP 4).

Why: the live dashboard on the M1 shows a low frame rate, so viewers can't see what really happens. This builds ONE
smooth 1920x1080 video from a finished `studio.stadium --aim director` run, showing the whole chain at once:
  1. the 3 STATIC cameras that are really installed (the Alfheim rig at halfway);
  2. the stitched panorama, with the boxes MAIN and CLOSE cut out of it right now (red = on air);
  3. the MAIN and CLOSE pictures (made by software, no robotic camera);
  4. the PROGRAM (what the AI director puts on air), the director's reason and the latest cuts.
Nothing is detected or decided here: it only reads the run's files (cameras.jsonl, director_log.csv, cam_*.mp4,
program.mp4) and the source videos, so it is exactly what the run did.

Usage:
  venv/bin/python -m studio.showcase                          # latest director run, its first 60 s
  venv/bin/python -m studio.showcase --run <run folder> --from 30 --seconds 90
Writes <run folder>/showcase.mp4.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from src.common import ROOT
from studio.panorama import Panorama
from studio.sources import get_source
from studio.stadium import FrameReader, Pinhole, Prefetch, View, settings
from studio.video import VideoWriter

W, H = 1920, 1080
LEFT_W = 1152                     # left column: cameras -> panorama -> cut-outs
RIGHT_X, RIGHT_W = 1184, 712      # right column: program + director
RED, YELLOW, CYAN = (40, 40, 230), (40, 210, 240), (230, 200, 40)
WHITE, GREY = (240, 240, 240), (150, 150, 150)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def latest_run() -> Path:
    """The newest folder with a program.mp4 under outputs/stadium/."""
    runs = sorted((ROOT / settings()["outputs_dir"]).glob("*/*-director/program.mp4"), key=lambda p: p.parent.name)
    if not runs:
        raise SystemExit("no director run found: run `venv/bin/python -m studio.stadium --aim director ...` first")
    return runs[-1].parent


def text(img: np.ndarray, s: str, x: int, y: int, scale: float = 0.6, colour=WHITE, thick: int = 1) -> None:
    cv2.putText(img, s, (x, y), FONT, scale, colour, thick, cv2.LINE_AA)


def heading(img: np.ndarray, number: str, title: str, x: int, y: int) -> None:
    """A step heading like "1  Three static cameras" above a panel."""
    cv2.circle(img, (x + 11, y - 6), 13, YELLOW, -1)
    text(img, number, x + 5, y, 0.55, (20, 20, 20), 2)
    text(img, title, x + 32, y, 0.62, WHITE, 1)


def fit(img: np.ndarray, w: int, h: int) -> np.ndarray:
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def outline(pano: Panorama, state: dict, scale: float) -> np.ndarray | None:
    """Where a camera's 1280x720 picture sits in the panorama: its border as a polyline (scaled panorama pixels)."""
    if "pan" not in state:
        return None
    pin = Pinhole(View(state["pan"], state["tilt"], state["hfov"]), 1280, 720)
    t = np.linspace(0, 1, 24)
    border = np.concatenate([np.column_stack([t * 1280, np.zeros_like(t)]),
                             np.column_stack([np.full_like(t, 1280), t * 720]),
                             np.column_stack([(1 - t) * 1280, np.full_like(t, 720)]),
                             np.column_stack([np.zeros_like(t), (1 - t) * 720])])
    uv = pano.rays_to_pano(pin.rays(border)) * scale
    return np.round(uv).astype(np.int32)


def read_log(path: Path) -> dict[int, dict]:
    """director_log.csv by frame index."""
    with open(path) as f:
        return {int(r["frame"]): r for r in csv.DictReader(f)}


def build(run: Path, from_s: float, seconds: float) -> Path:
    segment = run.parent.name
    src = get_source(segment)
    pano = Panorama.load(src.folder)
    fps = src.fps
    log = read_log(run / "director_log.csv")
    first = min(log)                                   # the run's first match frame
    skip = round(from_s * fps)
    count = min(round(seconds * fps), len(log) - skip)
    if count <= 0:
        raise SystemExit(f"--from {from_s} is past the end of the run ({len(log) / fps:.0f} s)")

    readers = {f"cam{c}": FrameReader(src.video(c), first + skip) for c in (1, 2, 3)}
    readers["pano"] = FrameReader(src.folder / "panorama.mp4", first + skip)
    for name in ("program", "main", "close"):
        readers[name] = FrameReader(run / ("program.mp4" if name == "program" else f"cam_{name}.mp4"), skip)
    prefetch = Prefetch(readers)

    states = open(run / "cameras.jsonl")
    for _ in range(skip):
        states.readline()

    pano_scale = LEFT_W / pano.width
    pano_h = round(pano.height * pano_scale)
    out = run / "showcase.mp4"
    writer = VideoWriter(out, W, H, fps, crf=21, threads=4)
    cuts: list[tuple[float, str]] = []                 # (match time, camera name) of the latest cuts
    last_cam = None
    print(f"[showcase] {run.relative_to(ROOT)}: {count / fps:.0f} s from +{from_s:g} s -> {out.name}", flush=True)
    try:
        for n in range(count):
            frames = prefetch.get()
            if frames is None:
                break
            cams = json.loads(states.readline())
            row = log[first + skip + n]
            on_air, t = row["cam_name"], float(row["time"])
            if on_air != last_cam:
                cuts.append((t, on_air))
                cuts = cuts[-7:]
                last_cam = on_air

            canvas = np.full((H, W, 3), 18, np.uint8)
            text(canvas, "AI Director - how one broadcast is made from 3 static cameras", 24, 30, 0.75, WHITE, 2)
            text(canvas, f"match time {int(t // 60):02d}:{t % 60:04.1f}", W - 250, 30, 0.6, GREY)

            # 1. the 3 installed static cameras (BOX L / BOX R are cam1 / cam3 as they are)
            heading(canvas, "1", "3 static cameras, installed together at halfway (real footage)", 24, 72)
            tile_w, tile_h = LEFT_W // 3 - 8, round((LEFT_W // 3 - 8) * 0.75)
            for k, c in enumerate((1, 2, 3)):
                x = 24 + k * (tile_w + 12)
                canvas[84:84 + tile_h, x:x + tile_w] = fit(frames[f"cam{c}"], tile_w, tile_h)
                live = (c == 1 and on_air == "BOX L") or (c == 3 and on_air == "BOX R")
                cv2.rectangle(canvas, (x, 84), (x + tile_w - 1, 84 + tile_h - 1), RED if live else GREY, 3 if live else 1)
                text(canvas, f"CAM {c}", x + 8, 84 + tile_h - 10, 0.55)

            # 2. the stitched panorama with the cut-out boxes
            y2 = 84 + tile_h + 44
            heading(canvas, "2", "Stitched into one panorama; MAIN and CLOSE are cut out of it by software", 24, y2 - 12)
            pano_img = fit(frames["pano"], LEFT_W, pano_h)
            for cam_id, colour in (("main", YELLOW), ("close", CYAN)):
                line = outline(pano, cams["cams"][cam_id], pano_scale)
                if line is None:
                    continue
                live = cams["on_air"] == cam_id
                cv2.polylines(pano_img, [line], True, RED if live else colour, 4 if live else 2, cv2.LINE_AA)
                top = line[line[:, 1].argmin()]
                text(pano_img, cam_id.upper(), int(top[0]) + 6, max(int(top[1]) + 22, 22), 0.6, RED if live else colour, 2)
            canvas[y2:y2 + pano_h, 24:24 + LEFT_W] = pano_img

            # 3. the cut-out pictures
            y3 = y2 + pano_h + 44
            heading(canvas, "3", "The cut-outs: virtual cameras, nothing moves in the stadium", 24, y3 - 12)
            cut_h = H - 16 - y3
            cut_w = round(cut_h * 16 / 9)
            for k, (name, colour) in enumerate((("main", YELLOW), ("close", CYAN))):
                x = 24 + k * (cut_w + 16)
                canvas[y3:y3 + cut_h, x:x + cut_w] = fit(frames[name], cut_w, cut_h)
                live = cams["on_air"] == name
                cv2.rectangle(canvas, (x, y3), (x + cut_w - 1, y3 + cut_h - 1), RED if live else colour, 3 if live else 1)
                text(canvas, name.upper(), x + 8, y3 + 24, 0.6, colour, 2)

            # 4. the program + the director
            heading(canvas, "4", "AI director picks what goes on air", RIGHT_X, 72)
            prog_h = round(RIGHT_W * 9 / 16)
            canvas[84:84 + prog_h, RIGHT_X:RIGHT_X + RIGHT_W] = fit(frames["program"], RIGHT_W, prog_h)
            cv2.rectangle(canvas, (RIGHT_X, 84), (RIGHT_X + RIGHT_W - 1, 84 + prog_h - 1), RED, 3)
            cv2.rectangle(canvas, (RIGHT_X + RIGHT_W - 96, 96), (RIGHT_X + RIGHT_W - 16, 126), RED, -1)
            text(canvas, "LIVE", RIGHT_X + RIGHT_W - 84, 119, 0.65, WHITE, 2)
            y = 84 + prog_h + 44
            simulated = "  (3D, simulated)" if on_air.startswith("GOAL") else ""
            text(canvas, f"ON AIR: {on_air}{simulated}", RIGHT_X, y, 0.9, RED, 2)
            text(canvas, f"game state: {row['state']}", RIGHT_X, y + 36, 0.6, GREY)
            reason = row["reason"]
            for k in range(0, min(len(reason), 150), 50):           # wrap the reason over up to 3 lines
                text(canvas, reason[k:k + 50], RIGHT_X, y + 70 + k // 50 * 26, 0.6)
            y = y + 170
            text(canvas, "Latest cuts", RIGHT_X, y, 0.65, YELLOW, 2)
            for k, (ct, name) in enumerate(reversed(cuts)):
                text(canvas, f"{int(ct // 60):02d}:{ct % 60:04.1f}   {name}", RIGHT_X, y + 34 + k * 30, 0.6,
                     WHITE if k == 0 else GREY)
            text(canvas, "Alfheim dataset (Simula), non-commercial research use", RIGHT_X, H - 20, 0.45, GREY)
            writer.write(canvas)
            if n % (round(fps) * 10) == 0:
                print(f"[showcase] {n / fps:.0f} / {count / fps:.0f} s", flush=True)
    except BaseException:
        writer.abort()
        raise
    finally:
        prefetch.close()
        states.close()
    writer.close()
    print(f"[showcase] done: {out.relative_to(ROOT)}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Pre-rendered 'how it works' video of a stadium director run")
    ap.add_argument("--run", type=Path, help="run folder (default: the latest director run)")
    ap.add_argument("--from", dest="from_s", type=float, default=0.0, help="seconds into the run")
    ap.add_argument("--seconds", type=float, default=60.0)
    args = ap.parse_args()
    build(args.run.resolve() if args.run else latest_run(), args.from_s, args.seconds)


if __name__ == "__main__":
    main()
