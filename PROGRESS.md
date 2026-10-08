# Progress — AI Sports Director MVP 1
Last updated: 2026-10-08 16:10

## Status
| Step | Status | Notes |
|---|---|---|
| 0. System check | done | brew, python3.13, git, node present. ffmpeg + yt-dlp come from pip inside the venv (see Decisions) |
| 1. Setup | done | venv, pinned requirements, config.yaml, .gitignore, .env.example, README. Imports OK, MPS True, ffmpeg found |
| 2. Download | done | 22.08 s, 29.97 fps, 1920x1080, 660 frames, audio yes |
| 3. Crop | done | 3 × 640x1080, 660 frames each, same fps as source (verified by counting packets) |
| 4. Detect | done | Roboflow football-players-detection-3zvbc/11 via CoreML, ~300 ms/frame/cam |
| 5. Ball tracking | done | detected/predicted/lost per frame; ByteTrack IDs; 1 outlier rejected |
| 6. Director | done | 0 cuts on this clip (correct, see below); forced test cuts at exactly 2.0 s |
| 7. Render | done | program.mp4 1280x720 h264 + aac, 660 frames; debug.mp4 1278x790 h264 |
| 8. One command | done | `venv/bin/python src/main.py`, ~14 s when cached |
| 9. Dashboard | done | Replay 1× ≈ real time, 2× ≈ 65 fps; Live ≈ 1 fps; controls, WS, MJPEG, downloads tested |

## Key results
- Ball rate (frames with a ball detection): cam1 1% · cam2 53% · cam3 0% · **any cam 54%**.
- Ball tracker statuses: cam2 detected 349 / predicted 118 / lost 193 frames.
- Cuts: **0** (cam2 live for the full 22.02 s). With `start_cam: 1` → 1 cut at frame 60 (2.00 s).
- Processing: detection ~300 ms per cam-frame (≈1 fps live for 3 cams); render ~49 fps; dashboard replay up to ~65 fps.
- Memory: dashboard process ~870 MB during a live run.

## Decisions made
- Python 3.13 (already installed) is used: `inference` 1.7.3 supports it. No extra Python installed.
- Homebrew has no prebuilt ffmpeg for macOS 14.0 (it would compile for hours, pulling in extra Pythons),
  so ffmpeg comes from the `imageio-ffmpeg` pip package and yt-dlp from pip, both inside the venv.
  Video info uses PyAV instead of ffprobe. Nothing was installed system-wide.
- yt-dlp needs a JS runtime for YouTube now: we pass `--js-runtimes node` (Node v22 is installed).
- Source is re-encoded to constant frame rate so "frame N" means the same moment in every cam.
- Detection runs on CoreML (M1 GPU/Neural Engine): 300 ms/frame vs 5.4 s/frame on CPU.
- Each 640x1080 cam strip is padded (centred) onto a black 1920x1080 canvas before detection. The model
  squashes tall images into its square input and then found almost nothing (0 players). Padding fixed it
  (203 players / 11 balls on 60 sample frames). Each camera still only sees its own pixels.
- "Ball rate under 40%" is measured as **any-cam** rate (54%), not the per-cam average (~18%): with a
  follow-cam the ball is in one strip at a time, so a per-cam average can never pass 33%. Full-frame
  detection agreed with the strip detector on 57/60 sampled frames, so the ball-only model + slicer
  upgrade was not needed (`use_ball_model` / `use_slicer` remain available in config.yaml).
- 0 cuts is correct: on full frames the ball was in the middle third in 60/60 sampled frames (Veo follow-cam).
- `sv.ByteTrack` is deprecated in supervision 0.30 but still works; versions pinned in requirements.txt.
- Dashboard: video files are written by a background thread and the 4 JPEGs are encoded in parallel, so
  replay keeps real time. Stop discards the half-made videos/CSV (the last complete ones are kept).
- Model caches (inference, ultralytics, matplotlib) are kept inside the project (`models_cache/`, `.yolo_config/`).

## Problems / open issues
- This clip can't demonstrate director quality (ball always in the middle strip). A wide, static-camera
  clip would exercise real cuts — that changes the plan's input, so it is the user's call.
- ByteTrack IDs churn a bit (79 IDs in cam2 for ~15 people) because players are small and occluded.

## Next action
MVP 1 complete. See TESTING.md.
