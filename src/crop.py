"""crop.py — Step 3 (split mode): cut the source video into 3 "cameras".

The source frame is cut into 3 equal vertical strips: left (cam1), middle (cam2)
and right (cam3). One ffmpeg pass produces all 3 files, so they stay perfectly in sync.
Strip width is rounded down to an even number because the H.264 encoder needs even sizes.
"""
from __future__ import annotations

import subprocess
import sys

if __package__ in (None, ""):  # allow "python src/crop.py"
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from src.common import cam_path, ffmpeg_exe, load_config, path, video_info


def strip_width(width: int, num_cams: int) -> int:
    """Width of one camera strip, rounded down to an even number."""
    w = width // num_cams
    return w - (w % 2)


def crop(force: bool = False) -> None:
    cfg = load_config()
    src = path(cfg["video"]["source"])
    n = cfg["video"]["num_cams"]
    outs = [cam_path(cfg, i + 1) for i in range(n)]
    if all(o.exists() for o in outs) and not force:
        print("[crop] camera files already exist, skipping (use --force to redo)")
    else:
        outs[0].parent.mkdir(parents=True, exist_ok=True)
        info = video_info(src)
        w, h = strip_width(info["width"], n), info["height"]
        # filter_complex: split the input into n copies, crop each copy to its strip.
        parts = [f"[0:v]split={n}" + "".join(f"[s{i}]" for i in range(n))]
        parts += [f"[s{i}]crop={w}:{h}:{i * w}:0[c{i}]" for i in range(n)]
        cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(src), "-filter_complex", ";".join(parts)]
        for i, o in enumerate(outs):
            cmd += ["-map", f"[c{i}]", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                    "-pix_fmt", "yuv420p", "-an", str(o)]
        subprocess.run(cmd, check=True)
    verify(src, outs)


def verify(src, outs) -> None:
    """Check every camera has the same fps and frame count as the source."""
    ref = video_info(src)
    for o in outs:
        info = video_info(o)
        same = abs(info["fps"] - ref["fps"]) < 0.01 and info["frames"] == ref["frames"]
        print(f"[crop] {o.name}: {info['width']}x{info['height']} | fps {info['fps']:.3f} | "
              f"{info['frames']} frames | {'OK' if same else 'MISMATCH'}")
        if not same:
            raise RuntimeError(f"{o.name} does not match source ({ref['frames']} frames @ {ref['fps']:.3f})")


if __name__ == "__main__":
    crop(force="--force" in sys.argv)
