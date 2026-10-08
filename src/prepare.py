"""prepare.py — Steps 2–3: turn the input from config.yaml into 3 synced camera videos.

Reads the `input:` section of config.yaml and writes:
    data/cams/cam1.mp4, cam2.mp4, cam3.mp4   the 3 cameras (same size, fps and frame count)
    data/raw/source.mp4                      the reference video (its sound goes into program.mp4)
    data/cams/input.json                     a "fingerprint" of the input settings used

Two modes:
  split  ONE video (YouTube link or file) is cut into 3 vertical strips: left / middle / right.
  files  3 separate clips, one per camera. Each is lined up with start_seconds, resized to
         the same size (black bars keep the shape), converted to the same fps and cut to the
         same number of frames.

The fingerprint makes re-runs safe: if you change the input section (or replace one of your
files), the next run notices, deletes the old cameras/detections/outputs, and starts fresh.
If nothing changed, this step is skipped.

Run it on its own:  venv/bin/python src/prepare.py          (add --force to redo)
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

if __package__ in (None, ""):  # allow "python src/prepare.py"
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import crop
from src.common import cam_path, detections_path, load_config, path, video_info
from src.download import InputError, convert, fetch, is_url, resolve_path, scaled_size

MIN_SECONDS = 3.0   # shorter clips can't show anything useful (min shot length is 2 s)


# ---------------------------------------------------------------- fingerprint
def input_signature(cfg: dict) -> dict:
    """Everything that changes the camera pixels, and nothing else.

    Detection thresholds are NOT included: changing them must not throw away the cameras.
    """
    inp = cfg["input"]
    mode = inp.get("mode")
    section = inp.get(mode, {})
    signature = {"mode": mode, "settings": section,
                 "max_pixels": cfg["video"]["max_pixels"], "num_cams": cfg["video"]["num_cams"]}
    # For local files, also remember size + modified time, so replacing a file is noticed.
    signature["files"] = {s: _file_stamp(s) for s in _sources(cfg) if s and not is_url(s)}
    return json.loads(json.dumps(signature, sort_keys=True))


def _sources(cfg: dict) -> list[str]:
    inp = cfg["input"]
    if inp.get("mode") == "files":
        return [str(c.get("source", "")) for c in inp["files"].get("cams", [])]
    return [str(inp.get("split", {}).get("source", ""))]


def _file_stamp(source: str) -> list | None:
    file = resolve_path(source)
    if not file.is_file():
        return None
    stat = file.stat()
    return [stat.st_size, int(stat.st_mtime)]


def _signature_path(cfg: dict) -> Path:
    return path(cfg["video"]["cams_dir"]) / "input.json"


def is_up_to_date(cfg: dict) -> bool:
    """True if the camera files on disk were made from the current input settings."""
    sig_file = _signature_path(cfg)
    files = [cam_path(cfg, c) for c in range(1, cfg["video"]["num_cams"] + 1)] + [path(cfg["video"]["source"])]
    if not sig_file.exists() or not all(f.exists() for f in files):
        return False
    return json.loads(sig_file.read_text()) == input_signature(cfg)


def write_signature(cfg: dict) -> None:
    sig_file = _signature_path(cfg)
    sig_file.parent.mkdir(parents=True, exist_ok=True)
    sig_file.write_text(json.dumps(input_signature(cfg), indent=2))


# ---------------------------------------------------------------- main entry
def prepare(force: bool = False) -> None:
    cfg = load_config()
    mode = cfg["input"].get("mode")
    if mode not in ("split", "files"):
        raise InputError(f'input.mode in config.yaml must be "split" or "files" (it is "{mode}").')

    outs = [cam_path(cfg, c) for c in range(1, cfg["video"]["num_cams"] + 1)]
    if not force and is_up_to_date(cfg):
        print(f"[prepare] cameras are up to date ({mode} mode), skipping (use --force to redo)")
        crop.verify(path(cfg["video"]["source"]), outs)
        return

    # 1. Check everything (and download links) BEFORE deleting anything, so a typo in
    #    config.yaml never throws away the previous cameras and detections.
    plan = plan_split(cfg) if mode == "split" else plan_files(cfg)
    # 2. The new input is valid: remove the old generated files and build the new cameras.
    clean_generated(cfg)
    if mode == "split":
        make_split(cfg, plan)           # crop.py checks the 3 cameras itself
    else:
        make_files(cfg, plan)
        crop.verify(path(cfg["video"]["source"]), outs)
    write_signature(cfg)


def clean_generated(cfg: dict) -> None:
    """Delete files made from the OLD input. Only generated files; never your own videos."""
    n = cfg["video"]["num_cams"]
    generated = [cam_path(cfg, c) for c in range(1, n + 1)] + [path(cfg["video"]["source"]), _signature_path(cfg)]
    generated += [detections_path(cfg, c) for c in range(1, n + 1)]
    for key in ("program_path", "debug_path"):
        generated.append(path(cfg["render"][key]))
    log_csv = path(cfg["director"]["log_csv"])
    generated += [log_csv, log_csv.with_name(log_csv.name + ".part")]
    removed = [f for f in generated if f.exists()]
    for f in removed:
        f.unlink()
    if removed:
        print(f"[prepare] input changed: removed {len(removed)} old generated files (cameras, detections, outputs)")


# ---------------------------------------------------------------- mode: split
def plan_split(cfg: dict) -> dict:
    """Check the split settings and work out what to cut. Changes nothing on disk (except downloads)."""
    split = cfg["input"]["split"]
    src = fetch(str(split.get("source", "")), cfg["video"]["max_height"])
    info = video_info(src)
    start, length = _split_window(info["duration"], split)
    fps = round(info["fps"], 3)
    w, h = scaled_size(info["width"], info["height"], cfg["video"]["max_pixels"])
    return {"src": src, "start": start, "length": length, "fps": fps, "frames": math.floor(length * fps),
            "scale": f"scale={w}:{h}" if (w, h) != (info["width"], info["height"]) else "", "size": (w, h)}


def make_split(cfg: dict, plan: dict) -> None:
    """One video -> data/raw/source.mp4 -> cut into 3 strips by crop.py."""
    w, h = plan["size"]
    print(f"[prepare] split: {plan['src'].name} | using {plan['start']:.2f}s → {plan['start'] + plan['length']:.2f}s "
          f"({plan['frames']} frames @ {plan['fps']} fps) | {w}x{h}")
    convert(plan["src"], path(cfg["video"]["source"]), plan["start"], plan["frames"], plan["fps"], plan["scale"],
            keep_audio=True)
    crop.crop(force=True)


def _split_window(duration: float, split: dict) -> tuple[float, float]:
    """Work out (start, length) in seconds from trim_last_seconds / start_seconds / max_seconds."""
    trim_last = split.get("trim_last_seconds")
    if trim_last:
        start = max(0.0, duration - float(trim_last))
    else:
        start = float(split.get("start_seconds") or 0)
        if start < 0 or start >= duration:
            raise InputError(f"input.split.start_seconds ({start}) must be between 0 and the video length "
                             f"({duration:.1f}s).")
    length = duration - start
    if split.get("max_seconds"):
        length = min(length, float(split["max_seconds"]))
    if length < MIN_SECONDS:
        raise InputError(f"Only {length:.1f}s of video left to use; need at least {MIN_SECONDS:.0f}s. "
                         "Check start_seconds / max_seconds / trim_last_seconds.")
    return start, length


# ---------------------------------------------------------------- mode: files
def plan_files(cfg: dict) -> dict:
    """Check the files settings and work out what to cut. Changes nothing on disk (except downloads)."""
    files = cfg["input"]["files"]
    cams = files.get("cams") or []
    n = cfg["video"]["num_cams"]
    if len(cams) != n:
        raise InputError(f"input.files.cams must list exactly {n} clips (cam1, cam2, cam3); found {len(cams)}.")
    width, height = (int(v) for v in files["size"])
    if width <= 0 or height <= 0 or width % 2 or height % 2:
        raise InputError(f"input.files.size must be two even numbers, like [1280, 720] (it is {files['size']}).")
    fps = float(files["fps"])
    audio_from = int(files.get("audio_from") or 1)
    if not 1 <= audio_from <= n:
        raise InputError(f"input.files.audio_from must be 1, 2 or 3 (it is {audio_from}).")

    sources = [fetch(str(c.get("source", "")), cfg["video"]["max_height"]) for c in cams]
    infos = [video_info(s) for s in sources]
    starts = [float(c.get("start_seconds") or 0) for c in cams]
    for i, (info, start) in enumerate(zip(infos, starts)):
        if start < 0 or start >= info["duration"]:
            raise InputError(f"cam{i + 1}: start_seconds ({start}) must be between 0 and the clip length "
                             f"({info['duration']:.1f}s) — {sources[i]}")

    # All cameras get the same length: whatever the SHORTEST clip has left after its start point.
    length = min(info["duration"] - start for info, start in zip(infos, starts))
    if files.get("max_seconds"):
        length = min(length, float(files["max_seconds"]))
    if length < MIN_SECONDS:
        raise InputError(f"After lining up the clips only {length:.1f}s overlap; need at least {MIN_SECONDS:.0f}s. "
                         "Check start_seconds and that the clips cover the same moment.")
    return {"sources": sources, "infos": infos, "starts": starts, "fps": fps, "size": (width, height),
            "frames": math.floor(length * fps), "audio_from": audio_from}


def make_files(cfg: dict, plan: dict) -> None:
    """3 separate clips -> 3 camera files with the same size, fps and frame count."""
    sources, infos, starts = plan["sources"], plan["infos"], plan["starts"]
    (width, height), fps, frames, audio_from = plan["size"], plan["fps"], plan["frames"], plan["audio_from"]
    # Resize to width x height without stretching: shrink to fit, then add black bars.
    fit = (f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
           f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1")
    for i, (src, info, start) in enumerate(zip(sources, infos, starts)):
        print(f"[prepare] cam{i + 1}: {src.name} ({info['width']}x{info['height']} @ {info['fps']:.2f} fps) | "
              f"from {start:.2f}s | {frames} frames → {width}x{height} @ {fps:g} fps")
        convert(src, cam_path(cfg, i + 1), start, frames, fps, fit, keep_audio=False)

    # The reference video: the audio camera again, WITH its sound (render copies the sound from it).
    audio_src = sources[audio_from - 1]
    if not infos[audio_from - 1]["has_audio"]:
        print(f"[prepare] note: cam{audio_from}'s clip has no sound, so program.mp4 will be silent "
              "(change input.files.audio_from to use another clip's sound)")
    convert(audio_src, path(cfg["video"]["source"]), starts[audio_from - 1], frames, fps, fit, keep_audio=True)


if __name__ == "__main__":
    try:
        prepare(force="--force" in sys.argv)
    except InputError as error:
        sys.exit(f"\n[prepare] PROBLEM: {error}")
