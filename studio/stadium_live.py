"""stadium_live.py — the virtual stadium played LIVE for the dashboard's playground page (MVP 4 Step 6).

The match plays at real speed (the wall clock). Every shown frame:
  - the pitch map gets the detections of every 1/5 s that has passed (studio.multicam via StadiumPipeline);
  - the operators aim MAIN / CLOSE, the robotic heads move (motor limits), the TV director picks what is on air;
  - the 5 real-pixel feeds (MAIN, CLOSE, WIDE, BOX L, BOX R) and the program go out as JPEGs (MJPEG <img> streams),
    the state (time, on air, every camera's view, the map) goes out ~10 times a second over the WebSocket, and every
    cut and robotic-camera move at once.
The 3D cameras (GOAL L / GOAL R) are drawn by the page itself, at the time this loop sends.
When the Mac is slower than real time, frames are skipped (decoded but not drawn), never delayed: like MVP 3 live.

Same interface as studio.live.LiveStudio (start / stop / status / jpeg), so the dashboard's live WebSocket and MJPEG
routes serve it unchanged. Uses only the current frame and the past (causal).
"""
from __future__ import annotations

import threading
import time
from datetime import datetime
from typing import Callable

import cv2
import numpy as np

from src.common import load_config
from studio.panorama import Panorama
from studio.stadium import Stadium, StadiumPipeline, placeholder, program_picture, settings


class StadiumLive:
    def __init__(self, source, publish: Callable[[dict], None] | None = None, start_s: float = 0.0,
                 seconds: float | None = None):
        self.src = source
        self.segment = source.key
        self.publish = publish or (lambda m: None)
        self.cfg = settings()
        self.fps = source.fps
        self.start_frame = max(0, min(round(start_s * self.fps), source.frames - 1))
        end = source.frames if not seconds else self.start_frame + round(seconds * self.fps)
        self.end = min(source.frames, end)
        self.stop_flag = threading.Event()
        self.thread: threading.Thread | None = None
        self.jpegs: dict[str, tuple[int, bytes]] = {}
        self.jpeg_lock = threading.Lock()
        self.status = {"state": "idle", "t": 0.0, "lag_s": 0.0, "shown_fps": 0.0, "skipped": 0}
        self.run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-stadium-live"

    # ------------------------------------------------------------------ control (same as LiveStudio)
    def start(self) -> None:
        self.status["state"] = "starting"
        self.thread = threading.Thread(target=self._run, daemon=True, name="stadium-live")
        self.thread.start()

    def stop(self) -> None:
        self.stop_flag.set()
        if self.thread and self.thread.is_alive() and threading.current_thread() is not self.thread:
            self.thread.join(timeout=30)

    def jpeg(self, name: str) -> tuple[int, bytes] | None:
        with self.jpeg_lock:
            return self.jpegs.get(name)

    def _put_jpeg(self, name: str, image: np.ndarray, width: int, quality: int) -> None:
        h, w = image.shape[:2]
        if w > width:
            image = cv2.resize(image, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            with self.jpeg_lock:
                seq = self.jpegs.get(name, (0, b""))[0] + 1
                self.jpegs[name] = (seq, buf.tobytes())

    def _run(self) -> None:
        try:
            self._loop()
        except Exception as error:      # noqa: BLE001 — report to the page, don't die silently
            import traceback

            traceback.print_exc()
            self.status["state"] = "error"
            self.publish({"type": "live_error", "message": f"{type(error).__name__}: {error}"})

    # ------------------------------------------------------------------ the loop
    def _loop(self) -> None:
        src, fps, cfg = self.src, self.fps, self.cfg
        live_cfg = cfg["live"]
        self.status["state"] = "preparing"
        stadium = Stadium(cfg, Panorama.load(src.folder))
        pipe = StadiumPipeline(src, stadium, "director")
        w, h = stadium.size
        real = [c for c in stadium.cameras if c.built_from in ("panorama", "source")]
        caps = {"pano": cv2.VideoCapture(str(src.folder / "panorama.mp4"))}
        for c in real:
            if c.built_from == "source":
                caps[c.source_cam] = cv2.VideoCapture(str(src.video(c.source_cam)))
        for cap in caps.values():
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
        # warm up the map + heads on the last few seconds before the start (a crew is already filming)
        for d in range(max(0, self.start_frame - round(live_cfg["warmup_s"] * fps)), self.start_frame):
            if d % src.step == 0:
                pipe.detections(d, d / fps, detector=src.detection_at(d))
        credit = "Alfheim dataset (Simula), non-commercial research"
        self.status["state"] = "live"
        self.publish({"type": "info", "message": f"stadium live from {self.start_frame / fps / 60:.0f}:"
                                                 f"{self.start_frame / fps % 60:02.0f}"})
        wall0 = time.time()
        i = self.start_frame - 1                  # the last frame read
        next_det = self.start_frame + (-self.start_frame) % src.step
        shown, skipped, last_state = 0, 0, 0.0
        widths = {cid: head.target_width for cid, head in stadium.heads.items()}
        fast = {cid: False for cid in stadium.heads}
        on_air = None
        while not self.stop_flag.is_set():
            target = self.start_frame + int((time.time() - wall0) * fps)
            if target >= self.end:
                break
            if target <= i:
                time.sleep(0.003)
                continue
            # skipped frames are decoded (grab) but not drawn: the videos stay in step with the clock
            for cap in caps.values():
                for _ in range(target - i - 1):
                    cap.grab()
            frames = {k: cap.read()[1] for k, cap in caps.items()}
            if any(f is None for f in frames.values()):
                break
            skipped += target - i - 1
            dt = (target - i) / fps
            i, t = target, target / fps
            while next_det <= i:                  # every detection step that has passed, in order
                pipe.detections(next_det, next_det / fps, detector=src.detection_at(next_det))
                next_det += src.step
            decision, cards, shots = pipe.frame(i, t, dt)
            pictures = {}
            for c in real:
                pictures[c.id] = stadium.picture(shots[c.id], frames["pano"], frames)
                self._put_jpeg(c.id, pictures[c.id], live_cfg["tile_width"], live_cfg["jpeg_quality"])
            cam = stadium.cameras[decision.cam - 1]
            picture = pictures.get(cam.id)
            if picture is None:                   # a 3D camera: the page draws it over the program
                picture = placeholder(cam.name, w, h)
            self._put_jpeg("program", program_picture(picture, cam, t, getattr(pipe.base, "score", [0, 0]),
                                                      {**cfg, "program": {**cfg["program"], "scorebug": False}}, credit),
                           live_cfg["program_width"], live_cfg["jpeg_quality"])
            shown += 1
            # events: cuts at once, robotic camera moves (a new zoom, the start of a fast move)
            for e in decision.events:
                if e["type"] == "CUT":
                    a, b = stadium.cameras[e["from"] - 1], stadium.cameras[e["to"] - 1]
                    self.publish({"type": "cut", "t": round(t, 2), "from": a.id, "to": b.id,
                                  "shot_s": e["shot_seconds"], "message": f"{a.name} -> {b.name}: {decision.reason[5:]}"})
            for cid, head in stadium.heads.items():
                if abs(head.target_width - widths[cid]) >= 2:
                    self.publish({"type": "ptz", "t": round(t, 2), "cam": cid,
                                  "message": f"{head.cam.name} zooms to {head.target_width:.0f} m"})
                    widths[cid] = head.target_width
                now_fast = head.speed_deg_s > cfg["director"]["settle_deg_s"]
                if now_fast and not fast[cid]:
                    self.publish({"type": "ptz", "t": round(t, 2), "cam": cid,
                                  "message": f"{head.cam.name} swings ({head.speed_deg_s:.0f} deg/s)"})
                fast[cid] = now_fast
            if decision.cam != on_air:
                on_air = decision.cam
            now = time.time()
            if now - last_state >= 1 / live_cfg["state_hz"]:
                last_state = now
                elapsed = now - wall0
                self.status.update({"t": round(t, 2), "lag_s": round(elapsed - (t - self.start_frame / fps), 2),
                                    "shown_fps": round(shown / max(elapsed, 1e-6), 1), "skipped": skipped})
                self.publish(self._state(t, decision, shots, pipe))
        for cap in caps.values():
            cap.release()
        self.status["state"] = "ended"
        self.publish({"type": "info", "message": f"stadium live ended at {i / fps / 60:.0f}:{i / fps % 60:02.0f}"})

    def _state(self, t: float, decision, shots: dict, pipe: StadiumPipeline) -> dict:
        """What the page draws ~10x per second: the clock, what is on air, every camera's view, the pitch map."""
        st = pipe.pitch
        cams = {}
        for cid, shot in shots.items():
            c = shot.cam
            s = {k: v for k, v in shot.state.items() if k in ("look_at", "field_width_m", "pan", "hfov", "speed_deg_s")}
            if "pan" not in s and s.get("look_at"):    # fixed real cameras: their pan from where they look
                dx, dy = s["look_at"][0] - c.position_m[0], s["look_at"][1] - c.position_m[1]
                s["pan"] = round(float(np.degrees(np.arctan2(dx, -dy))), 2)
            cams[cid] = s
        on_air = shots[list(shots)[decision.cam - 1]].cam.id
        return {"type": "state", "t": round(t, 2), "on_air": on_air, "director_state": decision.state,
                "reason": decision.reason, "cams": cams,
                "ball": [round(v, 1) for v in st.ball] if st and st.ball and st.ball_status != "lost" else None,
                "ball_status": st.ball_status if st else "lost",
                "players": [[round(p[0], 1), round(p[1], 1)] for p in (st.players if st else [])],
                "zone": st.zone if st else "unknown", "score": getattr(pipe.base, "score", [0, 0]),
                "status": dict(self.status)}
