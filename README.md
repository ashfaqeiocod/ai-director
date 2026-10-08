# AI Sports Director — MVP 1

An offline prototype that turns one football video into a "broadcast":

1. Downloads a short match clip.
2. Cuts it into 3 fake cameras (left / middle / right strips).
3. Finds the ball and players in each camera separately.
4. Picks which camera is "live" on every frame, based on where the ball is.
5. Renders `outputs/program.mp4` (the broadcast) and `outputs/debug.mp4` (all 3 cams with boxes).
6. Shows everything live in a local web dashboard.

## Install (once)

Needs: macOS with Python 3.10–3.13 and Node.js (yt-dlp uses Node to read YouTube pages).
Nothing is installed system-wide: ffmpeg and yt-dlp come as pip packages inside the venv.

```bash
cd ai-director
python3 -m venv venv
venv/bin/pip install -r requirements.txt
cp .env.example .env        # then put your Roboflow API key in .env
```

## Run

```bash
venv/bin/python src/main.py            # all steps; skips steps whose output already exists
venv/bin/python src/main.py --force    # redo everything
venv/bin/python -m dashboard.server    # dashboard at http://localhost:8000
```

Single steps can also be run on their own: `venv/bin/python src/download.py`, `src/crop.py`, `src/detect.py`.

## Where things are

| Path | What |
|---|---|
| `config.yaml` | every tunable number (thresholds, scores, shot lengths) |
| `src/download.py` | Step 2: download the clip with yt-dlp |
| `src/crop.py` | Step 3: split into 3 camera videos with ffmpeg |
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
