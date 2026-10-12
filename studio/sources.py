"""sources.py — the footage a live (or replayed) run plays, behind one small interface.

Two kinds:
  SoccerTrackSource("eval")   one 4K panorama cut into virtual cameras; pitch map, goal cameras, kick-off check;
                              answer key = the real goal times
  MuvsSource(19)              3 fans' phones filming the same match (MUVS); every camera is its own; no pitch
                              map; answer key = which camera a human picked every 3 s

Everything that differs between them lives here, so studio.live and the dashboard don't need to care.
Names: a source's `key` is used for folders and URLs ("eval", "muvs-P19").
"""
from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import numpy as np

from src.common import ROOT, load_config, video_info


def get_source(key: str):
    if key.startswith("muvs-P"):
        return MuvsSource(int(key[len("muvs-P"):]))
    return SoccerTrackSource(key)


def ready_sources() -> list[str]:
    """Everything that can be played: SoccerTrack segments with cameras + detections, MUVS periods with detections."""
    from studio.prepare import work_dir

    out = [s for s in load_config()["studio"]["segments"]
           if (work_dir(s) / "cams" / "cam5_detector_rects.npy").exists() and (work_dir(s) / "detections" / "cam1.json").exists()]
    muvs = ROOT / load_config()["studio"]["muvs"]["dir"]
    for folder in sorted(muvs.glob("P*")):
        if (folder / "detections" / "cam1.json").exists() and (folder / "cam1.mp4").exists():
            out.append(f"muvs-{folder.name}")
    return out


class SoccerTrackSource:
    kind = "soccertrack"
    goalwatch = True

    def __init__(self, segment: str):
        from studio.cameras import load_cams
        from studio.detect import PanoramaDetections, detect_step
        from studio.prepare import load_segment

        self.key = self.segment = segment
        self.cfg = load_config()["studio"]
        self.meta = load_segment(segment)
        self.cams = load_cams(segment)
        self.fps, self.frames, self.has_audio = self.meta["fps"], self.meta["frames"], self.meta["has_audio"]
        self.step = detect_step(self.fps)
        self.on_air = self.cfg["director"]["on_air_cams"]
        self.det = PanoramaDetections(segment)

    def video(self, cam: int) -> Path:
        from studio.cameras import cam_video

        return cam_video(self.segment, cam, "detector")

    def pipeline(self, feed):
        from studio.cameras import cams_dir
        from studio.pipeline import Pipeline
        from studio.prepare import work_dir

        return Pipeline(self.cfg, self.cams, self.meta, np.load(cams_dir(self.segment) / "cam5_detector_rects.npy"),
                        np.load(work_dir(self.segment) / "pitch_to_image.npy"), feed, "detector", self.step)

    def detection_at(self, frame: int) -> dict:
        return self.det.at(frame)

    def story_clip(self) -> tuple[Path, dict]:
        """The video the story layer watches (the close-up, with the match sound) + its settings."""
        from story.common import story_config
        from studio.story_pass import deep_merge, story_clip, without_missing_sound

        cfg = without_missing_sound(deep_merge(story_config(), self.cfg["story"]["feeds"]["close"]), self.has_audio)
        return story_clip(self.segment, "close_detector"), cfg

    def overlays(self, i: int, pipe, rects: list) -> dict:
        """What the detector saw, per on-air camera, in 0..1 of the camera picture (boxes, dots, ball)."""
        from studio.cameras import CLOSE_CAM
        from studio.detect import DETECT_CAMS
        from studio.video import to_camera

        w, h = self.cams["size"]
        frame = i - i % self.step
        here = self.det.at(frame)
        ball = pipe.world.state.ball                       # the map's ball (seen or remembered)
        out = {}
        for k, cam in enumerate(self.cams["cams"]):
            c = cam["cam"]
            if c not in self.on_air:
                continue
            item: dict = {}
            if c in DETECT_CAMS:
                det = self.det.dets[c]["frames"].get(frame, {})
                item["players"] = [[round(v / s, 3) for v, s in zip(p["box"], (w, h, w, h))] for p in det.get("players", [])]
            elif c == CLOSE_CAM:
                pts = [to_camera(rects[k], w, h, p) for p in here["players"]]
                item["dots"] = [[round(x / w, 3), round(y / h, 3)] for x, y in pts if 0 <= x < w and 0 <= y < h]
            if ball is not None:
                bx, by = to_camera(rects[k], w, h, ball)
                if 0 <= bx < w and 0 <= by < h:
                    item["ball"] = [round(bx / w, 3), round(by / h, 3)]
            out[str(c)] = item
        return out

    def info(self) -> dict:
        """Static things the page needs: cameras, camera views on the pitch, answer key."""
        from studio.cameras import CLOSE_CAM, cams_dir
        from studio.evaluate import truth_events
        from studio.pitchmap import Projection
        from studio.prepare import work_dir

        proj = Projection(np.load(work_dir(self.segment) / "pitch_to_image.npy"))
        rects = np.load(cams_dir(self.segment) / "cam5_detector_rects.npy")
        return {"kind": self.kind, "key": self.key, "cams": self.cams, "fps": self.fps, "duration": self.frames / self.fps,
                "on_air_cams": self.on_air, "goal_cams": self.cams.get("goal_cams", []),
                "pitch": {"coverage": {str(c["cam"]): proj.coverage(c["rect"]) for c in self.cams["cams"] if c["rect"]},
                          "close_coverage": {str(i): proj.coverage(tuple(rects[i]), n=4) for i in range(0, len(rects), 25)},
                          "close_cam": CLOSE_CAM},
                "truth": [{"t": round(e["t"], 2), "label": e["label"], "team": e["team"]} for e in truth_events(self.meta)
                          if e["label"] in ("GOAL", "SHOT")], "human": []}


class MuvsSource:
    kind = "muvs"
    goalwatch = False                  # no calibration: no goal cameras, no pitch map

    def __init__(self, period: int):
        self.cfg_all = load_config()["studio"]
        m = self.cfg_all["muvs"]
        self.period, self.key = period, f"muvs-P{period}"
        self.folder = ROOT / m["dir"] / f"P{period}"
        self.meta = json.loads((self.folder / "meta.json").read_text())
        info = video_info(self.folder / "cam1.mp4")
        self.fps, self.has_audio = info["fps"], False
        self.frames = min(video_info(self.folder / f"cam{c}.mp4")["frames"] for c in (1, 2, 3))
        names = m["cam_names"]
        self.cams = {"size": [info["width"], info["height"]],
                     "cams": [{"cam": c, "name": names[f"cam{c}"]["name"], "shot_type": names[f"cam{c}"]["shot_type"],
                               "rect": None} for c in (1, 2, 3)]}
        # The director's settings for phones: only these cameras, a sensible first camera.
        self.cfg = copy.deepcopy(self.cfg_all)
        self.cfg["director"].update(m["director"])
        self.on_air = self.cfg["director"]["on_air_cams"]
        self.dets = {c: json.loads((self.folder / "detections" / f"cam{c}.json").read_text()) for c in (1, 2, 3)}
        self.step = self.dets[1]["step"]
        self.thr = m["ball_tracker"]["min_confidence"]

    def video(self, cam: int) -> Path:
        return self.folder / f"cam{cam}.mp4"

    def pipeline(self, feed):
        from studio.pipeline import IndependentPipeline

        quality = {**self.cfg_all["quality"], **self.cfg_all["muvs"]["quality"]}
        return IndependentPipeline(self.cfg, self.cams, self.fps, feed, self.step, self.cfg_all["muvs"]["ball_tracker"],
                                   quality)

    def detection_at(self, frame: int) -> dict:
        return {c: self.dets[c]["frames"].get(str(frame), {"players": [], "ball": None}) for c in (1, 2, 3)}

    def story_clip(self) -> tuple[Path, dict]:
        """The story layer watches one phone (the side view, `story_cam`), as an MVP 2 clip (no sound)."""
        from story.common import story_config
        from story.intake import clip_dir, intake
        from studio.story_pass import deep_merge, without_missing_sound

        cam = self.cfg_all["muvs"]["story_cam"]
        name = f"{self.key.lower()}-cam{cam}"
        if not (clip_dir(name) / "meta.json").exists():
            intake(str(self.video(cam)), name=name)
        cfg = deep_merge(story_config(), {"sport": self.cfg_all["muvs"]["story_sport"]})
        return clip_dir(name), without_missing_sound(cfg, False)

    def overlays(self, i: int, pipe, rects: list) -> dict:
        w, h = self.cams["size"]
        frame = i - i % self.step
        out = {}
        for c in (1, 2, 3):
            det = self.dets[c]["frames"].get(str(frame), {})
            item = {"players": [[round(v / s, 3) for v, s in zip(p["box"], (w, h, w, h))] for p in det.get("players", [])]}
            b = pipe.balls.get(c)
            if b is not None and b.center is not None:
                item["ball"] = [round(b.center[0] / w, 3), round(b.center[1] / h, 3)]
            out[str(c)] = item
        return out

    def human_picks(self) -> list[dict]:
        rows = csv.DictReader(open(self.folder / "human_cuts.csv"))
        return [{"start": float(r["start_s"]), "end": float(r["end_s"]), "cam": int(r["camera"].replace("cam", ""))}
                for r in rows]

    def info(self) -> dict:
        return {"kind": self.kind, "key": self.key, "cams": self.cams, "fps": self.fps, "duration": self.frames / self.fps,
                "on_air_cams": self.on_air, "goal_cams": [], "pitch": None, "truth": [],
                "human": self.human_picks(), "credit": self.meta.get("credit", "")}
