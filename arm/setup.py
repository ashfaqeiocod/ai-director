"""setup.py — Arm MVP "camera setup": before the match, learn where the two athletes sit in every camera.

A real crew frames every camera before the start and knows who is where. Here the AI does that once per camera,
from the FIRST frame only (nothing that happens later is used, so it stays fair for a live run):
  1. Gemini looks at the first frame (one still per camera = 5 requests in total, cached) and boxes the athlete
     in the black shirt, the athlete in the tan shirt and their gripped hands. It also says which athlete is
     nearest the camera and what kind of view it is.
  2. Those boxes are matched to the local pose model's people in the same frame (best overlap), so from now on
     every look can recognise the two athletes by position: a person whose box overlaps an athlete's seat.
Why not rules? On this footage shirt colour and size were not enough: photographers in black stand right behind
the grip, and the audience standing looks bigger than the seated athletes.

Usage: venv/bin/python -m arm.setup   (writes data/arm/cache/setup.json; prints what it found)
"""
from __future__ import annotations

import json
from typing import Literal

import cv2
from pydantic import BaseModel

from arm.common import cache_dir, cam_letters, synced_video
from arm.gemini import ask, jpeg
from arm.match import iou, load_cache


class Box(BaseModel):
    ymin: int
    xmin: int
    ymax: int
    xmax: int


class CameraSetup(BaseModel):
    black_athlete: Box
    tan_athlete: Box
    gripped_hands: Box
    nearest_to_camera: Literal["black", "tan", "neither"]
    view: Literal["side", "behind_black", "behind_tan", "wide", "other"]
    audience_faces_visible: int


PROMPT = """This is the first frame of one camera filming an arm-wrestling match in an office.
Two athletes sit at a small table with their right hands gripped: one wears a BLACK polo shirt, the other a TAN
(light brown / beige) polo shirt. Other people around them are the audience or photographers (ignore them).
Return:
- black_athlete: the box of the whole visible body of the athlete in the black shirt who is seated at the table,
- tan_athlete: the same for the athlete in the tan shirt,
- gripped_hands: a box around their two gripped hands,
  (all boxes as integers 0..1000 relative to the image: ymin, xmin, ymax, xmax)
- nearest_to_camera: which athlete is clearly nearest the camera ("neither" if about the same distance),
- view: "side" (we see the table from the side, both athletes in profile), "behind_black" / "behind_tan"
  (we film over that athlete's back or shoulder), "wide" (a wide shot with the audience), "other",
- audience_faces_visible: how many audience faces are visible."""


def to_pixels(box: dict, width: int, height: int) -> list[float]:
    return [box["xmin"] * width / 1000, box["ymin"] * height / 1000, box["xmax"] * width / 1000, box["ymax"] * height / 1000]


def setup_camera(cam: str) -> dict:
    cap = cv2.VideoCapture(str(synced_video(cam)))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot read camera {cam}")
    H, W = frame.shape[:2]
    answer, info = ask(f"setup_{cam}", [jpeg(frame, 1280)], PROMPT, CameraSetup)
    first = load_cache(cam)["frames"][0]["people"]
    result = {"view": answer["view"], "nearest": answer["nearest_to_camera"],
              "audience_faces": answer["audience_faces_visible"], "gemini": info,
              "grip": to_pixels(answer["gripped_hands"], W, H)}
    picks = {}
    for name in ("black", "tan"):
        gem = to_pixels(answer[f"{name}_athlete"], W, H)
        # The pose model's person that overlaps Gemini's box best = this athlete's "seat" (its box is tighter
        # and has body points). If none overlaps well, keep Gemini's box.
        scores = [(iou(gem, p["box"]), k) for k, p in enumerate(first)]
        best, k = max(scores) if scores else (0.0, -1)
        picks[name] = (best, k, gem)
    # Both athletes matched to the same person (one hides behind the other, camera C): the weaker match keeps
    # Gemini's own box.
    if picks["black"][1] == picks["tan"][1]:
        weaker = "black" if picks["black"][0] < picks["tan"][0] else "tan"
        picks[weaker] = (0.0, -1, picks[weaker][2])
    for name, (best, k, gem) in picks.items():
        result[name] = {"seat": first[k]["box"] if best >= 0.3 else gem, "gemini_box": gem, "match_iou": round(best, 2)}
    return result


def load_setup() -> dict:
    return json.loads((cache_dir() / "setup.json").read_text())


def main() -> int:
    setup = {}
    for cam in cam_letters():
        setup[cam] = setup_camera(cam)
        s = setup[cam]
        print(f"cam {cam}: view {s['view']:<12} nearest {s['nearest']:<7} black seat match {s['black']['match_iou']:.2f} "
              f"tan seat match {s['tan']['match_iou']:.2f}  audience faces {s['audience_faces']}  "
              f"({'cached' if s['gemini']['cached'] else s['gemini']['model'] + ' ' + str(s['gemini']['seconds']) + ' s'})")
    (cache_dir() / "setup.json").write_text(json.dumps(setup, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
