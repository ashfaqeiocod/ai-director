# Progress — AI Sports Director MVP 1
Last updated: 2026-10-08

## Status
| Step | Status | Notes |
|---|---|---|
| 0. System check | done | brew, python3.13, git, node present. ffmpeg + yt-dlp now come from pip inside venv (see Decisions) |
| 1. Setup | done | venv, requirements, config.yaml, .gitignore, .env.example, README. Imports OK, MPS True, ffmpeg found |
| 2. Download | done | 22.08 s, 29.97 fps, 1920x1080, 660 frames, audio yes |
| 3. Crop | done | 3 × 640x1080, 660 frames each, same fps as source |
| 4. Detect | blocked | `.env` has placeholder `PASTE_KEY_HERE`; need real ROBOFLOW_API_KEY |
| 5. Ball tracking | in progress | code written (src/ball_tracker.py) |
| 6. Director | in progress | code written (src/director.py) |
| 7. Render | not started | |
| 8. One command | not started | |
| 9. Dashboard | not started | |

## Key results
(ball rate per camera, number of cuts, processing fps)

## Decisions made
- Python 3.13 (already installed) is used: `inference` 1.7.3 supports it. No extra Python installed.
- Homebrew has no prebuilt ffmpeg for macOS 14.0 (it would compile everything from source for hours),
  so ffmpeg comes from the `imageio-ffmpeg` pip package and yt-dlp from pip, both inside the venv.
  Video info uses PyAV instead of ffprobe. Nothing was installed system-wide.
- yt-dlp needs a JS runtime for YouTube now: we pass `--js-runtimes node` (Node v22 is installed).
- Source is re-encoded to constant frame rate so "frame N" means the same moment in every cam.
- Model caches (inference, ultralytics, matplotlib) are kept inside the project (`models_cache/`, `.yolo_config/`).
- `sv.ByteTrack` is deprecated in supervision 0.30 but still works; used as the spec says.

## Problems / open issues
- Roboflow key missing (placeholder in .env).

## Next action
Get the Roboflow key, then run Step 4 detection.
