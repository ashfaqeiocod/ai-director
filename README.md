# AI Sports Director — MVP 1

An offline prototype that turns football video into a "broadcast":

1. Takes **one** wide video (YouTube link or file) and cuts it into 3 cameras (left / middle / right),
   **or** takes **3 separate camera clips** and lines them up in time.
2. Finds the ball and players in each camera separately.
3. Picks which camera is "live" on every frame, based on where the ball is.
4. Renders `outputs/program.mp4` (the broadcast) and `outputs/debug.mp4` (all 3 cams with boxes).
5. Shows everything live in a local web dashboard.

## Guides

| I want to… | Read |
|---|---|
| set it up on a new computer (Mac, Windows, Linux) | [docs/NEW_DEVICE_SETUP.md](docs/NEW_DEVICE_SETUP.md) |
| test with my own video(s) | [docs/USING_YOUR_OWN_VIDEOS.md](docs/USING_YOUR_OWN_VIDEOS.md) |
| check everything step by step | [TESTING.md](TESTING.md) |
| understand / tune Live mode | [docs/LIVE_MODE.md](docs/LIVE_MODE.md) |

## Install (once)

Needs Python 3.10–3.13 (and Node.js only for YouTube links). Full guide: [docs/NEW_DEVICE_SETUP.md](docs/NEW_DEVICE_SETUP.md).
Nothing is installed system-wide: ffmpeg and yt-dlp come as pip packages inside the venv.

```bash
cd ai-director
python3 -m venv venv
venv/bin/pip install -r requirements.txt
cp .env.example .env                    # then put your Roboflow API key in .env
venv/bin/python src/check_setup.py      # says "Ready." or exactly what to fix
```
(Windows: use `venv\Scripts\python` instead of `venv/bin/python`.)

## Run

```bash
venv/bin/python src/main.py            # all steps; skips steps whose output already exists
venv/bin/python src/main.py --force    # redo everything
venv/bin/python -m dashboard.server    # dashboard at http://localhost:8000
```

Which video is used is set in the `input:` section at the top of `config.yaml`.
Single steps can also be run on their own: `venv/bin/python src/prepare.py`, `src/detect.py`.

## Where things are

| Path | What |
|---|---|
| `config.yaml` | every tunable number (thresholds, scores, shot lengths) |
| `src/prepare.py` | Steps 2–3: turn the `input:` (1 video or 3 clips) into 3 synced cameras; notices input changes |
| `src/download.py` | helper: download YouTube links / read files, convert with ffmpeg |
| `src/crop.py` | helper (split mode): cut one video into 3 strips, check frame counts |
| `src/check_setup.py` | checks a computer is ready (Python, packages, ffmpeg, key, GPU, input files) |
| `clips/` | put your own test videos here (never uploaded to git) |
| `src/detect.py` | Step 4: ball + player detection, cached to `outputs/detections/` |
| `src/ball_tracker.py` | Step 5: smooths the ball position (detected / predicted / lost) |
| `src/director.py` | Step 6: scores cameras and decides when to cut |
| `src/engine.py` | the frame-by-frame core used by the CLI and the dashboard |
| `src/render.py` | Step 7: writes `program.mp4` and `debug.mp4` |
| `src/main.py` | Step 8: runs everything with one command |
| `dashboard/` | Step 9: local web dashboard |
| `TESTING.md` | step-by-step tests |
| `docs/LIVE_MODE.md` | Live mode settings, and why real-time mode behaves as it does |

Videos and outputs are never committed to git (the footage shows minors).
