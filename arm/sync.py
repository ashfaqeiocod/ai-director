"""sync.py — Arm MVP Step 1: line up the 5 phone videos on ONE timeline, frame-exact.

Five people pressed "record" by hand, so every video starts at a slightly different moment, and the phones
film at different frame rates (one at 60 fps, some with a variable frame rate). Two clues tell us how to line up:
  1. metadata: every phone writes when it started recording, but only to the second (and the Samsung writes
     when it STOPPED). Good for a first guess (about 1 s).
  2. sound: all 5 phones heard the same room (claps, table knocks, cheers). Sliding one sound track over
     another until they match ("cross-correlation", GCC-PHAT) gives the offset to a fraction of a millisecond.
We measure all 10 camera pairs and solve for one offset per camera (camera A = 0). If the pairs agree
(A->B + B->C = A->C) the sync is trustworthy. Then every video is cut to the part that ALL cameras filmed and
re-timed to a constant 30 fps, so frame N of every camera is the same moment.

Usage:
  venv/bin/python -m arm.sync            # measure + write the synced copies + check video
  venv/bin/python -m arm.sync --measure  # only measure and print (fast, writes nothing)
Output: data/arm/synced/cam{A..E}.mp4 (+ camA_full.mp4 at 4K), sync.json, sync_check.mp4.
"""
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import av
import cv2
import numpy as np

from arm.common import cam_letters, raw_video, settings, sync_file, synced_video
from src.common import ffmpeg_exe
from studio.video import VideoWriter


# ---------- reading the files ----------

def file_info(video: Path) -> dict:
    """Phone model, start time from the metadata, and where the video and audio tracks start and end."""
    with av.open(str(video)) as c:
        tags = dict(c.metadata)
        v, a = c.streams.video[0], c.streams.audio[0]
        v_start = float(v.start_time * v.time_base) if v.start_time is not None else 0.0
        a_start = float(a.start_time * a.time_base) if a.start_time is not None else 0.0
        v_dur = float(v.duration * v.time_base) if v.duration else float(c.duration / av.time_base)
        info = {
            "phone": tags.get("com.apple.quicktime.model") or tags.get("com.android.manufacturer") or "Android",
            "container_start": float(c.start_time / av.time_base) if c.start_time else 0.0,
            "video_start": v_start, "video_end": v_start + v_dur,
            "audio_start": a_start,
            "fps": float(v.average_rate), "width": v.codec_context.width, "height": v.codec_context.height,
            "frames": v.frames,
        }
    info["recorded_at"] = recorded_at(tags, v_dur)
    return info


def recorded_at(tags: dict, duration: float) -> str | None:
    """When the phone started recording (ISO time, second precision), from the metadata.

    iPhones write the START time in com.apple.quicktime.creationdate. Android phones write `creation_time`,
    which on this Samsung is the moment recording STOPPED, so we subtract the duration.
    """
    if "com.apple.quicktime.creationdate" in tags:
        return datetime.fromisoformat(tags["com.apple.quicktime.creationdate"]).isoformat()
    if "creation_time" in tags:
        end = datetime.fromisoformat(tags["creation_time"].replace("Z", "+00:00"))
        return (end - timedelta(seconds=duration)).isoformat()
    return None


def load_audio(video: Path, rate: int) -> np.ndarray:
    """The whole sound track as mono float32 at `rate` samples/s (sample 0 = the audio track's start)."""
    chunks = []
    with av.open(str(video)) as c:
        resampler = av.AudioResampler(format="flt", layout="mono", rate=rate)
        for frame in c.decode(audio=0):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
        for out in resampler.resample(None):
            chunks.append(out.to_ndarray().reshape(-1))
    return np.concatenate(chunks).astype(np.float32)


# ---------- measuring offsets ----------

def gcc_phat(ref: np.ndarray, other: np.ndarray, rate: int, center_s: float, search_s: float) -> tuple[float, float]:
    """How many seconds later a sound appears in `other` than in `ref`, and how sure we are.

    GCC-PHAT: compare the two tracks in the frequency domain, keeping only the timing (phase) of every
    frequency and ignoring how loud it is. Different phone microphones colour the sound differently; the
    timing of claps and knocks is the same, which gives one sharp peak at the right lag.
    Only lags within center_s +- search_s are considered. Confidence = the peak divided by the next-best
    peak at least 20 ms away (above ~3 is a clear match).
    """
    n = 1 << int(np.ceil(np.log2(len(ref) + len(other))))
    spec = np.conj(np.fft.rfft(ref, n)) * np.fft.rfft(other, n)
    freqs = np.fft.rfftfreq(n, 1 / rate)
    spec[(freqs < 150) | (freqs > 3800)] = 0           # skip rumble (air con) and the edge of the band
    spec /= np.abs(spec) + 1e-12
    corr = np.fft.irfft(spec, n)
    lags = np.arange(n)
    lags[lags > n // 2] -= n                            # the second half of the result = negative lags
    window = np.abs(lags / rate - center_s) <= search_s
    idx = np.flatnonzero(window)
    best = idx[np.argmax(corr[idx])]
    peak = corr[best]
    # Sub-sample peak position: fit a parabola through the peak and its two neighbours.
    left, right = corr[(best - 1) % n], corr[(best + 1) % n]
    denom = left - 2 * peak + right
    shift = 0.5 * (left - right) / denom if denom != 0 else 0.0
    far = idx[np.abs(lags[idx] - lags[best]) > 0.02 * rate]
    runner_up = corr[far].max() if len(far) else 1e-9
    return (lags[best] + shift) / rate, float(peak / max(runner_up, 1e-9))


def solve_offsets(pairs: dict[tuple[str, str], tuple[float, float]], cams: list[str]) -> dict[str, float]:
    """One offset per camera from all pair measurements (least squares, camera cams[0] fixed at 0).

    pairs[(i, j)] = (lag, confidence) where lag = offset_j - offset_i. Sure pairs count more.
    """
    rows, rhs, weights = [], [], []
    for (i, j), (lag, conf) in pairs.items():
        row = np.zeros(len(cams))
        row[cams.index(j)], row[cams.index(i)] = 1, -1
        rows.append(row); rhs.append(lag); weights.append(min(conf, 20.0))
    fix = np.zeros(len(cams)); fix[0] = 1
    rows.append(fix); rhs.append(0.0); weights.append(100.0)
    w = np.sqrt(np.array(weights))
    solution, *_ = np.linalg.lstsq(np.array(rows) * w[:, None], np.array(rhs) * w, rcond=None)
    return {c: float(solution[k]) for k, c in enumerate(cams)}


def measure() -> dict:
    """Read every camera, measure the 10 pair offsets and solve. Offsets are in FILE time: an event that
    happens at global time g is at file time g + offset[cam] in that camera's file."""
    s = settings()["sync"]
    rate = s["sample_rate"]
    cams = cam_letters()
    infos = {c: file_info(raw_video(c)) for c in cams}
    for c in cams:      # the config's name is nicer than what the phone writes (the Samsung writes none)
        infos[c]["phone"] = settings()["cameras"][c]["phone"]
    audio = {c: load_audio(raw_video(c), rate) for c in cams}
    started = {c: datetime.fromisoformat(infos[c]["recorded_at"]) for c in cams}

    pairs = {}
    for i, j in itertools.combinations(cams, 2):
        # Metadata guess: if phone j started 1 s later, a sound comes 1 s EARLIER in j's file.
        guess = -(started[j] - started[i]).total_seconds() - (infos[j]["audio_start"] - infos[i]["audio_start"])
        lag, conf = gcc_phat(audio[i], audio[j], rate, guess, s["search_s"])
        # Sample 0 of each track is at that track's audio_start in its file: convert to file time.
        pairs[(i, j)] = (lag + infos[j]["audio_start"] - infos[i]["audio_start"], conf)

    offsets = solve_offsets(pairs, cams)
    residuals = {f"{i}-{j}": round((lag - (offsets[j] - offsets[i])) * 1000, 2)
                 for (i, j), (lag, _) in pairs.items()}
    # The part every camera filmed, in global time (global 0 = camera A's file time 0).
    start = max(infos[c]["video_start"] - offsets[c] for c in cams)
    end = min(infos[c]["video_end"] - offsets[c] for c in cams)
    return {
        "cameras": {c: {**infos[c], "offset_s": round(offsets[c], 5)} for c in cams},
        "pairs": {f"{i}-{j}": {"lag_s": round(lag, 5), "confidence": round(conf, 2)}
                  for (i, j), (lag, conf) in pairs.items()},
        "pair_residual_ms": residuals,
        "worst_residual_ms": max(abs(r) for r in residuals.values()),
        "common_start_s": round(start, 4), "common_end_s": round(end, 4),
    }


# ---------- writing the synced copies ----------

def write_synced(result: dict) -> None:
    """Cut every camera to the common window and re-time it to a constant frame rate (ffmpeg)."""
    s = settings()["sync"]
    fps = s["fps"]
    frames = int((result["common_end_s"] - result["common_start_s"]) * fps)
    duration = frames / fps
    result["frames"], result["fps"], result["duration_s"] = frames, fps, duration
    for cam, info in result["cameras"].items():
        # ffmpeg's -ss counts from the container start, so take that off the file time.
        seek = result["common_start_s"] + info["offset_s"] - info["container_start"]
        info["seek_s"] = round(seek, 5)
        sizes = [(False, f"scale={s['width']}:{s['height']}")]
        if cam in s["keep_full_res"]:
            sizes.append((True, "null"))
        for full, scale in sizes:
            out = synced_video(cam, full_res=full)
            out.parent.mkdir(parents=True, exist_ok=True)
            cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", "-ss", f"{seek:.5f}", "-i", str(raw_video(cam)),
                   "-t", f"{duration:.5f}", "-vf", f"fps={fps},{scale}", "-frames:v", str(frames),
                   "-c:v", "libx264", "-preset", "veryfast", "-crf", str(s["crf"]), "-pix_fmt", "yuv420p",
                   "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
                   "-movflags", "+faststart", str(out)]
            print(f"  writing {out.name} (seek {seek:.3f} s, {frames} frames)")
            subprocess.run(cmd, check=True)


def verify(result: dict) -> None:
    """Measure the pair offsets again on the SYNCED copies: they should all be ~0 ms now."""
    rate = settings()["sync"]["sample_rate"]
    cams = list(result["cameras"])
    audio = {c: load_audio(synced_video(c), rate) for c in cams}
    after = {}
    for c in cams[1:]:
        lag, conf = gcc_phat(audio[cams[0]], audio[c], rate, 0.0, 0.5)
        after[f"{cams[0]}-{c}"] = round(lag * 1000, 2)
    result["after_sync_ms"] = after


def check_video(result: dict) -> None:
    """sync_check.mp4: the 5 synced cameras in a grid + their sound waves around 'now', to check by eye.

    If the sync is right, a clap shows as a spike at the same place in all 5 waves (the red line = now).
    """
    rate = 8000
    cams = list(result["cameras"])
    caps = {c: cv2.VideoCapture(str(synced_video(c))) for c in cams}
    audio = {c: load_audio(synced_video(c), rate) for c in cams}
    tile_w, tile_h = 640, 360
    writer = VideoWriter(synced_video("A").parent / "sync_check.mp4", tile_w * 3, tile_h * 2, result["fps"])
    try:
        for n in range(result["frames"]):
            canvas = np.zeros((tile_h * 2, tile_w * 3, 3), np.uint8)
            for k, c in enumerate(cams):
                ok, frame = caps[c].read()
                if ok:
                    x, y = (k % 3) * tile_w, (k // 3) * tile_h
                    canvas[y:y + tile_h, x:x + tile_w] = cv2.resize(frame, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
                    cv2.putText(canvas, f"CAM {c}", (x + 10, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            draw_waves(canvas[tile_h:, tile_w * 2:], audio, n / result["fps"], rate)
            cv2.putText(canvas, f"t = {n / result['fps']:6.2f} s  frame {n}", (tile_w * 2 + 10, tile_h * 2 - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            writer.write(canvas)
        writer.close()
    except Exception:
        writer.abort()
        raise
    finally:
        for cap in caps.values():
            cap.release()


def draw_waves(tile: np.ndarray, audio: dict[str, np.ndarray], t: float, rate: int, half_s: float = 0.5) -> None:
    """Draw each camera's sound wave from t-0.5 s to t+0.5 s, one row per camera, red line at t."""
    h, w = tile.shape[:2]
    row_h = (h - 40) // len(audio)
    for k, (cam, wave) in enumerate(audio.items()):
        a, b = int((t - half_s) * rate), int((t + half_s) * rate)
        piece = np.zeros(b - a, np.float32)
        lo, hi = max(a, 0), min(b, len(wave))
        if hi > lo:
            piece[lo - a:hi - a] = wave[lo:hi]
        env = np.abs(piece).reshape(w, -1).max(axis=1) if len(piece) % w == 0 else \
            np.abs(np.interp(np.linspace(0, len(piece) - 1, w), np.arange(len(piece)), piece))
        env = env / (env.max() + 1e-6)
        mid = k * row_h + row_h // 2
        for x in range(w):
            dy = int(env[x] * row_h * 0.45)
            cv2.line(tile, (x, mid - dy), (x, mid + dy), (90, 220, 90), 1)
        cv2.putText(tile, cam, (4, mid + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.line(tile, (w // 2, 0), (w // 2, h - 40), (0, 0, 255), 1)


def report(result: dict) -> None:
    print("Camera offsets (file time of global 0; metadata start time):")
    for cam, info in result["cameras"].items():
        print(f"  {cam}  {info['phone']:<14} {info['fps']:5.2f} fps {info['width']}x{info['height']}  "
              f"offset {info['offset_s'] * 1000:8.1f} ms   recorded_at {info['recorded_at']}")
    print("Pairs (lag ms, confidence, residual after solving ms):")
    for key, p in result["pairs"].items():
        print(f"  {key}: {p['lag_s'] * 1000:8.1f}  conf {p['confidence']:5.1f}  residual {result['pair_residual_ms'][key]:6.2f}")
    limit = settings()["sync"]["max_pair_error_ms"]
    verdict = "OK" if result["worst_residual_ms"] <= limit else "UNSURE (pairs disagree)"
    print(f"Worst pair disagreement: {result['worst_residual_ms']:.2f} ms -> {verdict}")
    print(f"Common window: {result['common_start_s']:.3f} .. {result['common_end_s']:.3f} s "
          f"({result['common_end_s'] - result['common_start_s']:.2f} s)")
    if "after_sync_ms" in result:
        print(f"Re-measured on the synced copies (should be ~0 ms): {result['after_sync_ms']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--measure", action="store_true", help="only measure and print the offsets")
    args = parser.parse_args()
    result = measure()
    if not args.measure:
        write_synced(result)
        verify(result)
        sync_file().write_text(json.dumps(result, indent=2))
        print("  writing sync_check.mp4")
        check_video(result)
    report(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
