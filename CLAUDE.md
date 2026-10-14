# AI Sports Director

## Phases
- **MVP 1 (camera switching): DONE** (Steps 1–9 below). Its rules still apply and it must keep working:
  run the quick check in `TESTING.md` (section 1) after any change that could touch it.
- **MVP 2 (Live Story Reader): DONE.** Full spec: `docs/MVP2_STORY_SPEC.md`.
  All MVP 2 code lives in `story/` (plus a Story tab in the dashboard and `colab/`). Its non-negotiable rules
  (spec section 2) apply at all times: never read video/audio beyond the playhead (`CausalReader`), the browser
  `<video>` is the master clock, store `event_time` + `logged_at` for every story item, don't break MVP 1,
  never commit `.env`, clips, model weights or outputs. Progress is tracked in the "MVP 2" section of `PROGRESS.md`.
  Secrets: `GEMINI_API_KEY` in `.env` (stop and ask the user if it is missing or fails).

- **MVP 3 (Directed Story): DONE** (open items in `PROGRESS.md`). Full spec: `docs/MVP3_SPEC.md` (read it fully before working on MVP 3);
  big picture: `docs/AI_LIVE_DIRECTOR_PLAN.md`. Combines MVP 1 + MVP 2 on real SoccerTrack v2 footage
  (5 virtual cameras, camera cards, director state machine). All MVP 3 code lives in `studio/` and only USES
  `src/` and `story/` (never change their defaults). Ground truth (`bas/`, `ball/`) is for scoring only.
  Progress: "MVP 3" section of `PROGRESS.md`. Secrets: `HF_TOKEN` in `.env` (never print it).
- **MVP 4 (Virtual Stadium): CURRENT.** Full plan: `docs/MVP4_STADIUM_PLAN.md` (read it fully before working on
  MVP 4). Simulates a T2 camera setup on the Alfheim half (real panorama cameras + 3D behind-goal cameras, robotic
  MAIN/CLOSE steered by the director). Code in `studio/` and the dashboard; `src/` and `story/` defaults unchanged.
  Progress: "MVP 4" section of `PROGRESS.md`.
- **Arm MVP (real footage, arm wrestling): BUILT** (labels + votes pending). Spec: `docs/ARM_MVP_SPEC.md`. A home-made
  arm-wrestling match filmed by 5 phones (`data/arm/raw/`, real colleagues: never commit, never upload except the
  capped Gemini stills: hard cap 200 requests, counted in `data/arm/cache/gemini_calls.json`). Code in `arm/` (+ the
  `/arm` dashboard pages); only USES `src/`, `story/`, `studio/`. Progress: "Arm MVP" section of `PROGRESS.md`.

The rest of this file is the MVP 1 brief; the Working mode, Hardware rules and Coding style sections apply to MVP 2 and MVP 3 too.

# MVP 1 — camera switching (done)

## Goal
Build an offline Python prototype that:
1. Takes one football video and splits it into 3 "camera" feeds (left / middle / right).
2. Detects the ball and players in each feed independently, as if they were 3 real cameras.
3. Automatically produces ONE broadcast video that switches between the 3 cameras based on where the ball is.
4. Shows everything in a local web dashboard while it runs: the 3 camera feeds, the program output and a live decision log.

This is a proof of concept. Working end-to-end matters more than polish.
No streaming to the internet, no PTZ, no model training in this phase. The only UI is the local dashboard (Step 9).

## About me
- Frontend developer (JS/TS), new to Python and computer vision.
- Explain what each file does in plain English, in code comments and in your messages.
- After each step, tell me in 2–3 simple sentences what you built and how I can check it.

## Working mode (read first, every session)
- Work autonomously. Do Steps 1–9 in order without waiting for me between steps, unless you are blocked.
- Stop and ask me ONLY for: the Roboflow API key (if `.env` is missing or the key fails), a login, or a decision that changes the plan.
- First action in every session: read `PROGRESS.md`. If it does not exist, create it from the template below.
- System check before Step 1: check what is installed (brew, python3, ffmpeg, git, yt-dlp) and install anything missing with Homebrew. Install all Python packages inside the project venv only. Never use `sudo`. Never touch files outside this project folder.
- After each step: run its check yourself, fix problems until it passes, update `PROGRESS.md`, then `git commit -m "Step N: <name>"`. Never commit `.env`, videos or model weights.
- If a step fails 3 times in a row with the same error, write the error and what you tried into `PROGRESS.md` and stop.
- At the end: write `TESTING.md` with simple steps for me to test everything myself (commands, what files to open, what I should see), and print the same in chat.

### PROGRESS.md template
```
# Progress — AI Sports Director MVP 1
Last updated: <date time>

## Status
| Step | Status | Notes |
|---|---|---|
| 0. System check | not started | |
| 1. Setup | not started | |
| 2. Download | not started | |
| 3. Crop | not started | |
| 4. Detect | not started | |
| 5. Ball tracking | not started | |
| 6. Director | not started | |
| 7. Render | not started | |
| 8. One command | not started | |
| 9. Dashboard | not started | |

## Key results
(ball rate per camera, number of cuts, processing fps)

## Decisions made
## Problems / open issues
## Next action
```
Status values: not started / in progress / done / blocked.

## Hardware rules (important)
- Machine: MacBook M1, 8 GB RAM, no NVIDIA GPU.
- Ultralytics: use `device="mps"`, fall back to CPU if MPS errors.
- Never load a whole video into memory. Process frame by frame (generators).
- Max working resolution 1080p. Downscale if the source is larger.
- Load each model once and reuse it.
- Cache detections to disk per camera, so re-running the director/render steps does NOT re-run detection.

## Architecture rule (important)
- Build the core as a frame-by-frame engine, not a batch-only pipeline.
- `src/engine.py`: an `Engine` that reads one synced frame from all 3 cameras at a time and yields a `FrameResult` (frame index, time, detections per cam, ball status per cam, the 3 scores, chosen cam, and an event if something happened: CUT, BALL_LOST, BALL_FOUND).
- The engine supports two sources of detections: run the model now (live mode) or read the cached JSON (replay mode).
- The CLI (`main.py`) and the dashboard both consume this same engine. This is the same shape a real live system will need later.

## Input video
- URL: https://www.youtube.com/watch?v=RVWMU6p0hDQ (≈22 s Veo clip, amateur youth match)
- The source is a Veo follow-cam (it pans/zooms to follow play), so the ball will mostly be in the middle crop. This is expected: MVP 1 proves the pipeline, not director quality.
- The footage shows minors. Keep all video files and outputs out of git. Never upload them anywhere.

## Tech stack
| Purpose | Tool |
|---|---|
| Download | `yt-dlp` |
| Cut / convert / audio | `ffmpeg` (system install via Homebrew) |
| Ball + player detection (primary) | Roboflow pretrained model via the `inference` package: `football-players-detection-3zvbc/11` (classes: ball, goalkeeper, player, referee). Key from `ROBOFLOW_API_KEY` in `.env`. If a model version fails, check Roboflow Universe for the latest version number. |
| Ball upgrade (only if ball rate is poor) | Roboflow ball-only model `football-ball-detection-rejhg/4` + `supervision.InferenceSlicer` (tiling so the ball looks bigger) |
| Fallback detector | Ultralytics `yolo26s.pt` (COCO classes: person=0, sports ball=32); if unavailable, `yolo11s.pt` |
| Player tracking | `supervision.ByteTrack` |
| Ball tracking | Custom `BallTracker` (see Step 5), approach based on github.com/roboflow/sports |
| Drawing + video IO | `supervision` annotators, `sv.get_video_frames_generator`, `sv.VideoSink`, OpenCV |
| Config | `config.yaml` (all tunable numbers live here, nothing hard-coded) |
| Secrets | `python-dotenv` |
| Dashboard backend | `FastAPI` + `uvicorn` |
| Video to browser | MJPEG streams (`multipart/x-mixed-replace`), shown with plain `<img>` tags |
| Logs + live scores | WebSocket (JSON messages) |
| Dashboard frontend | One static `index.html` + Tailwind CDN + vanilla JS (no build step) |

## Repo structure
```
ai-director/
├── CLAUDE.md
├── README.md            # how to install and run, in simple English
├── requirements.txt
├── config.yaml
├── .env                 # ROBOFLOW_API_KEY=...  (gitignored)
├── .gitignore           # venv/, .env, data/, outputs/, *.pt, *.mp4
├── data/
│   ├── raw/             # source.mp4
│   └── cams/            # cam1.mp4, cam2.mp4, cam3.mp4
├── outputs/
│   └── detections/      # cam1.json, cam2.json, cam3.json
├── src/
│   ├── download.py      # Step 2
│   ├── crop.py          # Step 3
│   ├── detect.py        # Step 4
│   ├── ball_tracker.py  # Step 5
│   ├── director.py      # Step 6
│   ├── engine.py        # frame-by-frame core used by CLI and dashboard
│   ├── render.py        # Step 7
│   └── main.py          # Step 8: runs everything; skips a step if its output already exists (add --force to redo)
└── dashboard/
    ├── server.py        # Step 9: FastAPI app (streams, WebSocket, controls)
    └── static/
        └── index.html   # the dashboard page
```

## Build steps
Do these in order. Run and verify each step before starting the next.

### Step 1: Setup
- Python venv, `requirements.txt`, `.gitignore`, `config.yaml`, `.env` template, `README.md`.
- Verify: all imports work, `torch.backends.mps.is_available()` prints True, ffmpeg is found.

### Step 2: Download
- `yt-dlp` → `data/raw/source.mp4` (≤1080p, mp4 with audio).
- Print duration, fps, resolution, frame count.

### Step 3: Crop into 3 cameras
- ffmpeg: split into 3 equal vertical strips (left / middle / right) → `data/cams/cam1.mp4`, `cam2.mp4`, `cam3.mp4`.
- Verify all 3 have the same fps and frame count as the source.

### Step 4: Detect
- For EACH camera separately, run ball + player detection on every frame.
- Keep only the single highest-confidence ball per frame per camera (there is only one ball).
- Ball confidence threshold low (start 0.15), players normal (0.3), both in `config.yaml`.
- Save `outputs/detections/camN.json`: per frame → list of players (box, class, confidence) and ball (box, confidence) or null.
- Print ball detection rate per camera (% of frames with a ball) and average ms per frame.
- If the overall ball rate is under 40%: try the ball-only model, then add `InferenceSlicer`. Report rates before and after, and keep the best setup.

### Step 5: Ball tracking
- `BallTracker` per camera:
  - Keep a buffer of the last N ball centers (N in config, start 10).
  - Reject a detection that jumps too far from the recent average (outlier).
  - If the ball is missing, predict it from the last velocity for up to ~10 frames, then mark it lost.
- Each frame gets a status: `detected` / `predicted` / `lost`.
- Track players with `ByteTrack` (stable IDs).

### Step 6: Director (choose a camera for every frame)
- Score each camera per frame:
  - ball `detected`: +100
  - ball `predicted`: +60
  - ball close to the horizontal center of that camera: +0 to +30
  - players in that camera: +1 each, max +10
- If no camera has the ball: stay on the current camera; if lost for more than 2 s, go to cam2.
- Minimum shot length: 2.0 s.
- Hysteresis: switch only if another camera has scored higher for 0.5 s in a row.
- Every value above lives in `config.yaml`.
- Save `outputs/director_log.csv`: frame, time, chosen cam, the 3 scores, reason for the decision.

### Step 7: Render
- `outputs/program.mp4`: the final broadcast, with the chosen camera's frame at each moment, scaled to 1280×720 (letterbox if needed), small "CAM n" label top-left, original audio copied from the source with ffmpeg.
- `outputs/debug.mp4`: the 3 cameras side by side, with player boxes and track IDs, a ball marker plus short trail (different color for detected vs predicted), a red border and "LIVE" on the chosen camera, and a timecode plus the 3 scores at the bottom.

### Step 8: One command
- `python src/main.py` runs Steps 2–7 end to end.
- Print a summary: ball rate per camera, number of cuts, average shot length, total processing time.

### Step 9: Live dashboard (local web UI)
Start with `python -m dashboard.server`, then open http://localhost:8000.

Layout:
- Top-left, large: PROGRAM output with a "LIVE" badge and the current CAM label.
- Below it: 3 camera tiles (cam1, cam2, cam3) with player boxes, ball marker and a score bar under each. The chosen camera gets a red border.
- Right side: live log panel. Auto-scroll, newest at the bottom, color by type (CUT, BALL_LOST, BALL_FOUND, INFO, ERROR), filter by type.
- Bottom: timeline bar of the whole clip, colored by which camera was live, with a moving playhead.
- Stats strip: processing fps, current time/frame, ball rate per cam, number of cuts, average shot length.
- Controls: Start, Pause, Stop/Reset; mode switch Live / Replay; speed 0.5× / 1× / 2× (replay only); inputs for min shot length and hysteresis, applied on the next Start.
- When a run finishes: download buttons for `program.mp4`, `debug.mp4`, `director_log.csv`.

Modes:
- Live: runs detection frame by frame and plays at whatever speed the M1 manages. Show the real fps honestly.
- Replay: uses cached detections and plays smoothly at real time (or the chosen speed). For demos. Disabled until a Live run has cached detections.

Technical rules:
- The engine runs in a background thread. Frames go into small queues (maxsize 2) that drop old frames, so the UI never lags behind.
- MJPEG endpoints: `/stream/program`, `/stream/cam1`, `/stream/cam2`, `/stream/cam3`. JPEG quality ~70; cam tiles max 640 px wide, program max 1280 px, to save CPU.
- WebSocket `/ws`: per-frame state (time, scores, chosen cam, ball status) throttled to ~10 messages/s; events (CUT, BALL_LOST…) sent immediately.
- REST: `POST /start`, `POST /pause`, `POST /stop`, `GET /config`, `POST /config`, `GET /files/{name}`.
- Every event is also appended to `outputs/run_log.jsonl`.
- `program.mp4`, `debug.mp4` and `director_log.csv` are written while the engine runs, same as the CLI.

Done when: I open the page, click Start, and see the 3 cams and the program playing together, the red border moving on every cut, and the log updating live.

## Definition of done (MVP 1)
- One command produces `program.mp4`, `debug.mp4` and `director_log.csv`.
- The dashboard runs a full clip in both Live and Replay mode, with the log updating live.
- No shot shorter than 2 s.
- Runs on M1 8 GB without crashing.

## Out of scope for MVP 1 (MVP 2 adds its own scope, see the spec)
Streaming to the internet, OBS, PTZ control, training custom models, team classification, event detection, replays of highlights, authentication, deployment, any UI other than the local dashboard.

## Coding style
- Simple, readable Python with type hints and short functions.
- Plain-English comments explaining the "why".
- No clever abstractions; this is a prototype a beginner must be able to read.
