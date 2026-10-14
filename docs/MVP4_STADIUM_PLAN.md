# MVP 4 — Virtual Stadium (T2 camera setup) — implementation handoff

This plan is self-contained so a new Claude Code session can build it. It lives in the repo
as `docs/MVP4_STADIUM_PLAN.md`.

---

## 1. Why we are doing this
MVP 3 works on the Alfheim half (Tromsø IL – Strømsgodset, 2013-11-03, first half, 47 min, final score 0–0).
It uses 3 real FIXED cameras that all stand together high at the halfway line. The director cuts correctly,
but the output looks like CCTV:
- no camera ever pans or zooms;
- every shot comes from the same spot.

A real product would install a real camera set at the ground. We can't film there, so we **simulate the
camera set our product would install** (setup "T2", section 3).
- Each simulated camera produces **its own feed**.
- The director sees all feeds (like a TV control room's monitor wall), **aims and zooms the robotic ones** and
  **cuts between them**.
- Goal: a program that looks like a real TV broadcast, and a test bed for the director's cut, zoom and follow
  decisions.

The user is a frontend developer (JS/TS) who is new to Python. Explain everything in plain English, in
messages and in code comments.

## 2. Repo rules (from CLAUDE.md — must follow)
- First action: read `CLAUDE.md`, `PROGRESS.md`, `docs/MVP3_SPEC.md` (the director/cards design).
- New code goes in `studio/` and the dashboard (`dashboard/`).
  - `src/` (MVP 1) and `story/` (MVP 2) get **additive changes only, defaults unchanged**.
  - After any change that could touch MVP 1, run the MVP 1 quick check (`TESTING.md` section 1).
- **Never read beyond the playhead.** Live decisions use only the current and past frames (causal). The browser
  `<video>` is the master clock in the dashboard.
- All tunable numbers go in `config.yaml`; nothing is hard-coded.
- Hardware: MacBook M1 8 GB.
  - Frame-by-frame generators; never load a whole video into memory.
  - Max 1080p working resolution.
  - Load models once.
- **Commit after each step.**
  - Never commit `.env`, videos, model weights, downloaded data (ZXY), or `outputs/`.
  - Never print `HF_TOKEN` / `GEMINI_API_KEY`.
- Add an "MVP 4" section to `PROGRESS.md`; update it after each step.
- Alfheim licence: non-commercial research only. Credit it on every output: `CREDIT` in `studio/alfheim.py`.
- Run commands with `venv/bin/python -m studio.<module>`.
- Coding style: simple, typed, short functions, "why" comments, no clever abstractions.

## 3. Camera setups compared (the "infra guide" — goes into the doc as-is)

**Human TV crews:**
- Camera counts: lower league 4–6 cameras, top league 12–20, cup finals 40+.
- Core cameras:
  - MAIN: platform high in the main stand at halfway; follows play.
  - TIGHT: next to MAIN; player close-ups.
  - High box cameras L/R: at the edge of each penalty box.
  - Low box cameras.
  - Behind each goal: high and low.
  - Handheld/steadicam on the touchline.
  - Reverse angle (opposite stand).
  - Extras: spidercam, super slow-motion, cameras in the goal net.
- During live play about 70% of airtime is MAIN and about 15% TIGHT, both from the halfway platform. They stay
  on one side of the pitch, so left and right never flip for the viewer (the 180° rule).
- The other cameras are mostly for replays and stoppages.

**Automated products (Veo, Pixellot, Spiideo):**
- One panorama unit with 3–4 lenses at halfway, 10–15 m high.
- The AI crops a virtual "cameraman" view out of it.
- Cheap, but close-ups are soft and every shot comes from one side.

**Our tiers:**
| Tier | Cameras | Result |
|---|---|---|
| T1 | 1 panorama rig at halfway (~8K total width) | virtual MAIN / CLOSE / WIDE; watchable, like Veo |
| **T2 (chosen MVP)** | T1 + 2 fixed 4K cameras behind the goals (~6 m high) + 1 robotic close-up camera on the platform | feels like TV: sharp close-ups, real shot/goal angles, replays; behind-goal views also help the goal check |
| T3 | T2 + touchline robotic camera + reverse angle + crowd microphone | celebrations, reactions, sound |

All tiers also need:
- an edge GPU box at the ground (Jetson Orin or an RTX PC) for detection;
- a PoE/fibre network;
- a 5–10 s broadcast delay;
- a stream out.

**What Alfheim can simulate:** Alfheim *is* a low-resolution T1 rig (3 cameras of 1280x960 at halfway; players
are only about 35 px tall).
- From the real footage: the T1 cameras.
- From a 3D scene: the T2 behind-goal cameras.

## 4. What exists today (reuse, don't rewrite)
| Thing | Where | Notes |
|---|---|---|
| Alfheim videos | `data/alfheim/2013-11-03_H1/cam1-3.mp4` (1280x960, 30 fps, 84,780 frames, colour-fixed), `cam*_480.mp4` light copies, `meta.json` | cam1 = left third, cam2 = middle, cam3 = right third |
| Calibration | `data/alfheim/2013-11-03_H1/calibration.json` → `camN.pitch_to_image` (3x3 homography) | Pitch coordinates are metres, origin at the centre spot, x to the right −52.5..52.5, y towards the NEAR touchline −34..34 (see `studio/pitchmap.py` docstring). Made by `studio/alfheim.py --calibrate`. |
| Cached detections | `data/alfheim/2013-11-03_H1/detections/cam1-3.json` | players + ball per frame, 5 frames/s |
| Source wrapper | `studio/sources.py` → `AlfheimSource(IssiaSource)`, `get_source("alfheim-2013-11-03-H1")` | `.H[cam]`, `.cams`, `.detection_at(frame)`, `layout` |
| Pitch map | `studio/pitchmap.py` (`Projection`, `zone_of`, ball trail/speed, triggers) | ball + players in metres, causal |
| Virtual operator | `studio/cameras.py:96` `VirtualOperator` | dead zone + smoothing + max speed, crop rect in pixels; extend it, don't duplicate it |
| Crop/resize | `studio/video.py` `crop_to(image, rect, w, h)`, `VideoWriter` | |
| Camera cards | `studio/cards.py` `CamCard` (ball, players, shot_type, quality, action_in_view, zone, map_intensity) | the director's input |
| Director | `studio/director.py` `DirectorV2(cfg, fps, num_cams)` → `decide(i, t, cards, story_events, ...)` | config `studio.alfheim.director` (on_air_cams, follow_cams, danger_cams, stands, ...) |
| Shared per-frame brain | `studio/pipeline.py` | map → cards → goal confirmation → director |
| Live loop | `studio/live.py` (`--segment alfheim-2013-11-03-H1 --start 23:40`) | wall clock = now; cached or Colab detector |
| Goal watcher | `studio/goalwatch.py` (Gemini `gemini-3.5-flash-lite`) | known false goals on Alfheim (match was 0–0): loose kick-off check + stadium big screen |
| Dashboard | `dashboard/server.py`, `dashboard/studio_api.py`, `dashboard/static/studio.html` + `studio.css` | FastAPI, MJPEG `<img>` streams, WebSocket state |

## 5. The T2 camera plan we simulate
| id | Name | Position (pitch m) | Kind | Built from |
|---|---|---|---|---|
| main | MAIN | halfway, main stand, ~15 m high (same spot as the Alfheim rig) | virtual robotic camera: director aims + zooms, follows play | real panorama crop |
| close | CLOSE | next to MAIN | robotic close-up, motor speed limits | real panorama crop, tight (soft: an honest limit of 1280x960 sources) |
| wide | WIDE | same spot | fixed, whole pitch | real panorama, scaled |
| boxl / boxr | BOX L / BOX R | same spot, aimed at each third | fixed | real Alfheim cam1 / cam3 (as-is) |
| goall / goalr | GOAL L / GOAL R | behind each goal, centred, ~6 m high, ~8 m behind the goal line | fixed | 3D scene |

- All positions, heights, lens limits (min/max field width in metres) and motor speeds go in `config.yaml` →
  `studio.stadium`.
- Changing to T1 or T3 is just a different camera list.

## 6. Architecture (one idea that keeps it simple)
**Every camera is a math model** (position, look-at point, field of view): a real pinhole camera.
- Python (`studio/stadium.py`) holds all the models. For each camera and frame it computes:
  - **its card**, by projecting the ball and players (pitch metres) through that camera's model. No pixels
    are needed, so the director can judge the 3D cameras too.
  - **its picture**:
    - real cameras: a perspective crop of the panorama, rendered in Python;
    - 3D cameras: drawn in the browser by Three.js from the same camera model, exported as JSON.
- The director only ever sees cards and sends commands (`aim`, `zoom`, `cut`), exactly like a real
  control room with robotic heads.

## 7. Steps (do in order; check, update PROGRESS.md and commit after each)

### Step 0 — Document
- Save this plan as `docs/MVP4_STADIUM_PLAN.md`.
- Add an "MVP 4 (Virtual Stadium): CURRENT" line to the CLAUDE.md phase list (MVP 3 becomes DONE).
- Add the MVP 4 section to PROGRESS.md.
- Commit: `MVP 4 Step 0: plan + camera infra guide`.

### Step 1 — Panorama (`studio/panorama.py`)
- The 3 cameras share one viewpoint, so they stitch with homographies:
  - map cam1 and cam3 into cam2's picture with `H2 @ inv(H1)` and `H2 @ inv(H3)` (H = pitch_to_image);
  - then project onto a **cylinder** so the far ends are not stretched;
  - blend the seams (simple linear feather is enough).
- Output: `data/alfheim/2013-11-03_H1/panorama.mp4`, about 3600–4000 px wide.
  - Crop to the pitch plus a bit of stand.
  - Use the same fps and frame count as the sources.
  - Save `panorama.json` with the pitch → panorama mapping (a homography if the warp is planar; otherwise a
    lookup function), so map points can be placed in panorama pixels.
- Process frame by frame; expect this to take about the length of the half or longer. Add `--minutes N` for quick
  tests.
- **Check:** `panorama_check.jpg` with pitch lines drawn from the calibration. Lines must be continuous across
  both seams (≤ 3 px jump). Compare the player count in the seam areas against the source cams.
- Pitfall: the near touchline is very oblique (see the `studio/alfheim.py` docstring). If the homography seam
  mismatches on players (they aren't on the pitch plane), feather a wider seam rather than chasing perfection.

### Step 2 — Camera devices + real feeds (`studio/stadium.py`)
- `Camera` dataclass:
  - `id, name, kind ("ptz" | "fixed"), built_from ("panorama" | "source" | "3d")`;
  - `position_m (x, y, z)`, `look_at_m`, `field_width_m`, limits (`min/max_field_width_m`,
    `max_pan_deg_s`, `max_zoom_ratio_s`).
- `PTZHead`, the robotic head:
  - `aim(x_m, y_m)`, `zoom(field_width_m)`;
  - `step(dt)` moves towards the target with a dead zone, smoothing and speed limits;
  - reuse the logic of `VirtualOperator`, but in pitch metres/degrees instead of pixels;
  - `state()` gives the current look-at and field width.
- `project(camera, points_m) -> image points`: a pinhole model; `in_view`.
- Card for a camera: build a `CamCard` from projected ball/players (ball in view, players in view, `action_in_view`,
  `shot_type` from field width: wide > 50 m, medium 25–50, close < 25).
- Real picture for panorama cameras: a perspective crop of the panorama around the look-at point
  (`crop_to` is fine to start; a true perspective warp is better if time allows). Output 1280x720.
- CLI:
  ```
  venv/bin/python -m studio.stadium --segment alfheim-2013-11-03-H1 --start 23:40 --seconds 300 --aim follow
  ```
  - Writes `outputs/stadium/<segment>/<run>/cam_main.mp4`, `cam_close.mp4`, `cam_wide.mp4`, `cam_boxl.mp4`,
    `cam_boxr.mp4` plus `cameras.jsonl` (each camera's state per frame).
  - In this step MAIN/CLOSE follow a simple aim: ball from the map, else the player group.
- **Check:** watch all 5 feeds for 5 minutes. MAIN contains the ball in ≥ 90% of the frames where the ball is
  detected. No move is faster than the motor limit (assert in a test).

### Step 3 — Player positions for the 3D scene (`studio/zxy.py`)
- Download the match's ZXY sensor file from the Alfheim dataset page (https://datasets.simula.no/alfheim/;
  check the exact file URL there) into `data/alfheim/zxy/` (gitignored).
- Format (check it): timestamp, tag id, x, y in metres from a pitch corner, heading, speed, etc., 20 Hz.
  Convert to our centre-origin coordinates; check whether y is flipped against the camera's near side.
- **Important:** in Alfheim only the **home team (Tromsø) wears sensors**. The away team and the referee come
  from our detections on the pitch map.
  - Smooth them in pitch space (a simple tracker with constant velocity, holding position for up to 1 s when
    missing).
- Ball: Alfheim has no ball data.
  - Use our map's ball (`studio/pitchmap.py`) and interpolate gaps of up to 2 s.
  - Height is 0, except for long fast balls: give those a parabola from the speed.
  - This is for drawing only.
- Time alignment: find the video-time offset of the ZXY clock by matching ZXY home-player positions to our detected
  players over a few minutes (minimise the average distance). Check visually at the kick-off.
- Output: `data/alfheim/2013-11-03_H1/twin.jsonl`, one row per 0.1 s:
  `{t, home:[{id,x,y,speed}], away:[...], ref:[...], ball:{x,y,z}|null}`.
- **Check:** draw `twin` positions onto real cam2 frames at 5 times. Home players must sit on the real players
  (≤ ~2 m).

### Step 4 — 3D scene (`dashboard/static/stadium.html`, Three.js from cdnjs with a pinned version)
- Scene:
  - 105x68 m grass pitch with correct lines, goals with nets, simple stand boxes;
  - players as low-poly figures in team colours (Tromsø white/red? check the video; Strømsgodset: check), with
    a simple run/idle animation chosen by speed, facing their direction of movement;
  - the ball.
- Loads `twin.jsonl` (served by a new endpoint in `dashboard/studio_api.py`) and the camera list (from
  `config.yaml` via the API).
- Renders GOAL L / GOAL R (and a free "director's view" for debugging) as tiles synced to a time: the page
  clock, or the time sent by the live loop over the WebSocket.
- Export: `MediaRecorder` on `canvas.captureStream()` → WebM per camera, then convert with ffmpeg to
  `cam_goall.mp4` / `cam_goalr.mp4` (an endpoint takes the upload).
- **Check:**
  - open `http://localhost:8000/static/stadium.html`, play 1 min;
  - the 3D players move like the real ones at the same second;
  - the behind-goal views look right (goal centred, pitch going away from the camera).

### Step 5 — Director with robotic cameras and TV rules (studio/, additive)
- Add a stadium mode to the pipeline: cards come from `studio/stadium.py` for all 7 cameras.
- Camera commands every frame:
  - MAIN: aim at the ball, with lead room in the direction of play; zoom by game state:
    - open play ~40 m wide;
    - attack in the box ~28 m;
    - goal kick / restart ~60 m.
  - CLOSE: aim at the player nearest the ball (or the group after a stoppage), ~12 m wide.
- TV cutting rules (config `studio.stadium.director`):
  - MAIN is the default (~70%);
  - CLOSE after a stoppage, foul, shot or ball out, for 2–4 s, then back to MAIN;
  - GOAL L/R for shots or attacks inside that box only if the camera's card shows the action in view; otherwise BOX L/R;
  - WIDE for goal kicks and kick-offs;
  - minimum shot 2 s; never cut while a robotic camera is still moving fast (wait until it settles).
- Replays (optional, if time allows): after a shot, a 5–8 s replay from GOAL L/R with a "REPLAY" wipe, played when
  play stops.
- Goal watcher: keep it on the REAL cameras. The 3D cameras are drawn from our own data, so asking Gemini about
  them would be circular. Note in the doc that real T2 behind-goal cameras would replace this.
- Reuse `DirectorV2`; add new behaviour behind config, keeping the MVP 3 defaults the same.

### Step 6 — Playground page
- New dashboard page `dashboard/static/stadium.html`, or a "Stadium" tab in `studio.html` (follow
  `studio.html`/`studio.css` patterns). It shows:
  - a monitor wall of all 7 feeds (5 MJPEG `<img>` from Python + 2 Three.js canvases), the on-air one with a red border;
  - a large program view;
  - a small pitch map with each camera's position and view cone (looks great and explains the setup);
  - a log of cuts and robotic-camera moves;
  - a TV scorebug on the program (teams, score, match clock) and the Alfheim credit.
- Start/stop and the start time work like the current studio page.

### Step 7 — Test runs
- 5 minutes from 23:40, then the full half.
- Outputs: every `cam_*.mp4`, `program.mp4`, `director_log.csv`, `cameras.jsonl`.
- Report in PROGRESS.md:
  - airtime per camera (target MAIN ~60–75%);
  - number of cuts, average shot length, shots < 2 s (must be 0);
  - ball in MAIN frame %;
  - an honest note on picture quality (CLOSE is soft).
- Run the MVP 1 quick check (`TESTING.md` §1).

## 8. Known limits (say them honestly to the user)
- CLOSE and zoomed MAIN shots are soft: 1280x960 sources. A real T2 has a 4K panorama and a real robotic camera.
- The behind-goal cameras look like a video game, and away-team movement is less smooth (detections, not sensors).
- The ball has no real height or data; gaps are interpolated.
- All real-pixel cameras share one viewpoint (true for T1/T2 too).

## 9. Critical files
- New:
  - `studio/panorama.py`, `studio/stadium.py`, `studio/zxy.py`;
  - `dashboard/static/stadium.html` (+ endpoints in `dashboard/studio_api.py`);
  - `docs/MVP4_STADIUM_PLAN.md`;
  - tests in `tests/test_studio_stadium.py` (camera projection, PTZ speed limits, card from projection,
    ZXY coordinate conversion).
- Changed (additive): `studio/pipeline.py`, `studio/director.py`, `config.yaml` (new `studio.stadium` block),
  `PROGRESS.md`, `CLAUDE.md` (phase line).
- Untouched: `src/`, `story/`.

## 10. Verification (end to end)
1. `venv/bin/python -m pytest tests/` passes (new and old tests).
2. `panorama_check.jpg`: lines continuous across the seams.
3. `twin` check pictures: home players within ~2 m of the real ones.
4. 5-minute stadium run: all 7 feeds play.
   - Program: no shot < 2 s, MAIN airtime 60–75%, ball in MAIN ≥ 90% of detected frames.
   - Watch it with the user.
5. Playground page: feeds, program, map with camera cones, and log all update live.
6. MVP 1 quick check passes.
