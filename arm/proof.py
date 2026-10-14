"""proof.py — the "behind the AI's eyes" video: proof that the AI watches the 5 phones and cuts by itself.

One 1920x1080 video, frame by frame, showing at once:
  top left    the AI live broadcast (data/arm/out/program_live.mp4) — what the viewer gets
  top right   the AI's thinking: time, phase, tension / crowd meters, who leads, the latest moment, and the list of
              cuts with their reasons (the one on air highlighted)
  bottom      the 5 phones, with what the AI detects drawn on them: the two athletes (black / tan boxes), their
              faces with strain and joy, the grip, readable audience faces. The camera on air has a red border
              and its crop (the digital zoom) is drawn as a box. Under each phone: its best shot now and the score.
Everything shown is what the live director knew at that moment (nothing from the future).

Usage: venv/bin/python -m arm.proof   (after arm.main; writes data/arm/out/proof_behind_the_ai.mp4)
"""
from __future__ import annotations

import cv2
import numpy as np

from arm.common import cam_letters, out_dir, settings, synced_video
from arm.director import judge_boost, run_live
from arm.judge import known_at, load_judge
from arm.match import load_timeline
from arm.render import add_audio, event_caption
from arm.shots import all_shots, event_boosts
from studio.video import VideoWriter

W, H = 1920, 1080
TILE_W, TILE_H = 384, 216
FONT = cv2.FONT_HERSHEY_DUPLEX
RED, WHITE, GREY, YELLOW = (60, 60, 230), (255, 255, 255), (150, 150, 150), (80, 220, 255)
ATHLETE_COLOUR = {"black": (255, 120, 60), "tan": (60, 170, 255)}


def text(img, s, x, y, scale=0.6, colour=WHITE, thick=1):
    cv2.putText(img, s, (int(x), int(y)), FONT, scale, colour, thick, cv2.LINE_AA)


def bar(img, x, y, w, h, value, colour, label):
    cv2.rectangle(img, (x, y), (x + w, y + h), (70, 70, 70), -1)
    cv2.rectangle(img, (x, y), (x + int(w * float(np.clip(value, 0, 1))), y + h), colour, -1)
    text(img, f"{label} {value:.2f}", x + w + 10, y + h - 2, 0.55)


def draw_tile(frame: np.ndarray, look: dict, cam: str, on_air, best, label: str) -> np.ndarray:
    """One phone, small, with the AI's detections drawn on it."""
    s = TILE_W / 1920
    tile = cv2.resize(frame, (TILE_W, TILE_H), interpolation=cv2.INTER_AREA)
    for f in look["audience_faces"]:
        x1, y1, x2, y2 = (int(v * s) for v in f["box"])
        cv2.rectangle(tile, (x1, y1), (x2, y2), (90, 200, 90), 1)
    for name in ("black", "tan"):
        a = look[name]
        if not a:
            continue
        x1, y1, x2, y2 = (int(v * s) for v in a["box"])
        cv2.rectangle(tile, (x1, y1), (x2, y2), ATHLETE_COLOUR[name], 1)
        face = a["face"]
        if face:
            fx1, fy1, fx2, fy2 = (int(v * s) for v in face["box"])
            cv2.rectangle(tile, (fx1, fy1), (fx2, fy2), YELLOW, 1)
            text(tile, f"S{face['strain']:.1f} J{face['joy']:.1f}", fx1, max(10, fy1 - 3), 0.33, YELLOW)
    gx, gy = (int(v * s) for v in look["grip_point"])
    cv2.circle(tile, (gx, gy), 5, (0, 255, 0) if look["grip"] else GREY, 2)
    live = on_air.cam == cam
    if live:
        x, y, w, h = (int(v * s) for v in on_air.rect)
        cv2.rectangle(tile, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.rectangle(tile, (0, 0), (TILE_W - 1, TILE_H - 1), RED, 6)
        cv2.rectangle(tile, (TILE_W - 62, 4), (TILE_W - 6, 24), RED, -1)
        text(tile, "LIVE", TILE_W - 56, 20, 0.5, WHITE, 1)
    cv2.rectangle(tile, (0, 0), (150, 24), (0, 0, 0), -1)
    text(tile, f"{cam} {label}", 5, 17, 0.45)
    return tile


def main() -> int:
    timeline = load_timeline()
    judge = load_judge()
    decisions = run_live(timeline, judge)
    states = timeline["states"]
    setup = timeline["setup"]
    cams = cam_letters()
    labels = {c: settings()["cameras"][c]["label"] for c in cams}
    fps = settings()["sync"]["fps"]
    step = round(fps / settings()["perceive"]["analysis_fps"])
    events = [(s["t"], e) for s in states for e in s["events"]]
    cuts = [(k, d) for k, d in enumerate(decisions) if k == 0 or d.shot.key != decisions[k - 1].shot.key]

    # Per look: the best shot of every camera (exactly what the live director compared).
    seen: list[tuple[float, str]] = []
    best_per_look = []
    for k, state in enumerate(states):
        seen += [(state["t"], e) for e in state["events"]]
        boosts = judge_boost(event_boosts(seen, state["t"]), judge, state["t"], live=True)
        shots = all_shots({c: timeline["cams"][c][k] for c in cams}, state, setup, boosts)
        best_per_look.append({c: max((s for s in shots if s.cam == c), key=lambda s: s.value, default=None) for c in cams})

    program = cv2.VideoCapture(str(out_dir() / "program_live.mp4"))
    caps = {c: cv2.VideoCapture(str(synced_video(c))) for c in cams}
    total = int(min(cap.get(cv2.CAP_PROP_FRAME_COUNT) for cap in caps.values()))
    silent = out_dir() / "proof_behind_the_ai.video.mp4"
    writer = VideoWriter(silent, W, H, fps, crf=21, threads=4)
    try:
        for n in range(total):
            ok, prog = program.read()
            frames = {c: cap.read()[1] for c, cap in caps.items()}
            k = min(n // step, len(decisions) - 1)
            d, state = decisions[k], states[k]
            t = n / fps
            canvas = np.full((H, W, 3), 18, np.uint8)
            canvas[0:720, 0:1280] = cv2.resize(prog, (1280, 720), interpolation=cv2.INTER_AREA) if ok else 0

            # The AI's thinking (top right).
            x0 = 1300
            text(canvas, "BEHIND THE AI'S EYES", x0, 40, 0.9, WHITE, 2)
            text(canvas, f"t = {t:5.2f} s   (the AI only knows the past)", x0, 75, 0.55, GREY)
            text(canvas, f"phase: {state['phase'].replace('_', ' ')}", x0, 115, 0.8, YELLOW, 2)
            bar(canvas, x0, 135, 300, 16, state["tension"], (60, 80, 240), "tension")
            bar(canvas, x0, 162, 300, 16, state["crowd"], (90, 200, 90), "crowd")
            lead = "black leads" if state["lean"] > 0.15 else "tan leads" if state["lean"] < -0.15 else "even"
            text(canvas, f"grip height {state['upright']:.2f}   lean {state['lean']:+.2f} ({lead})", x0, 205, 0.55)
            moment = event_caption(events, t, 2.0)
            text(canvas, f"moment: {moment or '-'}", x0, 235, 0.6, (120, 230, 255) if moment else GREY)
            answer = known_at(judge, t, live=True)
            if answer:
                text(canvas, f"Gemini (asked {answer['event_time']} s): {answer['caption'][:34]}", x0, 262, 0.45, GREY)
            text(canvas, "CUTS (decided live) and WHY", x0, 300, 0.6, WHITE, 1)
            shown = [c for c in cuts if c[0] <= k][-9:]
            y = 328
            for i, (ck, cd) in enumerate(shown):
                current = i == len(shown) - 1
                colour = YELLOW if current else (200, 200, 200)
                text(canvas, f"{cd.t:5.1f}s {cd.change or 'START':<8} {cd.shot.cam} {cd.shot.kind}", x0, y, 0.5, colour, 1)
                reason = cd.reason if len(cd.reason) <= 62 else cd.reason[:60] + ".."
                text(canvas, reason, x0 + 14, y + 19, 0.4, colour if current else GREY)
                y += 43

            # The 5 phones with the AI's detections + each camera's best shot now.
            for i, cam in enumerate(cams):
                look = timeline["cams"][cam][k]
                best = best_per_look[k][cam]
                tile = draw_tile(frames[cam], look, cam, d.shot, best, labels[cam])
                x = i * TILE_W
                canvas[730:730 + TILE_H, x:x + TILE_W] = tile
                if best is not None:
                    value = best.value / 110
                    colour = RED if d.shot.cam == cam else (180, 140, 60)
                    cv2.rectangle(canvas, (x + 8, 958), (x + 8 + int((TILE_W - 16) * min(1, value)), 978), colour, -1)
                    cv2.rectangle(canvas, (x + 8, 958), (x + TILE_W - 8, 978), GREY, 1)
                    text(canvas, f"best: {best.kind} {best.value:.0f}", x + 8, 1000, 0.5)
                    p = best.parts
                    text(canvas, f"fit {p['fit']:.1f} emo {p['emotion']:.1f} evt {p['event']:.1f} clar {p['clarity']:.1f}",
                         x + 8, 1022, 0.4, GREY)
            text(canvas, "boxes: blue = black shirt, orange = tan shirt, yellow = face (S strain, J joy), green = audience face,"
                 " circle = grip; cyan box = the AI's crop (zoom) on the live camera", 10, 1062, 0.45, GREY)
            writer.write(canvas)
        writer.close()
    except Exception:
        writer.abort()
        raise
    finally:
        program.release()
        for cap in caps.values():
            cap.release()
    out = out_dir() / "proof_behind_the_ai.mp4"
    add_audio(silent, synced_video(settings()["render"]["audio_cam"]), out)
    silent.unlink(missing_ok=True)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
