# AI Sports Director

Two prototypes in one repo: **MVP 1** (camera switching, below) and **MVP 2** (Live Story Reader, at the end).

## MVP 1 — camera switching

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

## MVP 2 — Live Story Reader

Drop any soccer clip into the dashboard, press **Go Live**, and an AI watches it **as if it were live**
(it never reads video or sound past the playhead) and writes the story as it happens: intensity meter, events
(goals, shots, saves…), mood, a commentary line every ~8 s with its honest delay, the score and a "story so far".

```bash
venv/bin/python -m dashboard.server        # then open http://localhost:8000/story
venv/bin/python -m story.evaluate --clip eng-cze --run latest   # proof mode report
```

Needs `GEMINI_API_KEY` in `.env` (free key from https://aistudio.google.com). Optional: the Colab notebook
`colab/specialist_server.ipynb` runs a ball-action-spotting model on a free GPU (paste its URL in the dashboard).

| Layer | Where | What |
|---|---|---|
| Fast | Mac, every 1 s | crowd noise (YAMNet), whistle, motion, scene cuts → intensity + triggers |
| Specialist | Colab GPU | OpenSportsLab action spotting (shot, goal, free kick, tackle…) |
| Smart | Gemini, every ~8 s + triggers | last 6 s of video (+ sound) → phase, mood, events, commentary |
| Memory | Gemini, every 60 s | story so far, momentum, teams |
| Fusion | Mac | one timeline, intensity, score (no double goals on replays) |

Result on a 3:22 broadcast highlight (England 3-0 Czechia), hybrid mode: **3/3 goals, 80% of other headline
events, average delay 3.8 s**. Full guide: [TESTING_MVP2.md](TESTING_MVP2.md) · demo script: [DEMO.md](DEMO.md) ·
spec: [docs/MVP2_STORY_SPEC.md](docs/MVP2_STORY_SPEC.md) · results: [compare.md](compare.md).

| Path | What |
|---|---|
| `story/` | all MVP 2 code: `intake` (clips), `causal` (no-peeking guard), `fast_layer`, `specialist_client`, `llm/` (Gemini + templates), `fusion`, `engine`, `evaluate`, `soccernet` |
| `dashboard/story_api.py`, `dashboard/static/story.html` | the Story tab |
| `colab/` | the Colab notebook (built from `specialist_server.py` by `build_notebook.py`) |
| `tests/` | `venv/bin/python -m pytest tests -q` |
