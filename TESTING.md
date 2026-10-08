# How to test AI Sports Director — MVP 1

Run every command from the project folder in your own Terminal:

```bash
cd ~/Documents/ai-director
```

Everything is already downloaded, cut and detected, so the tests below start fast.

---

## 1. One command: the whole pipeline (about 15 s)

```bash
venv/bin/python src/main.py
```

What you should see:
- `[download] duration 30.00s | fps 25.000 | 2880x720 | 747 frames | audio: True`
- `[crop] cam1.mp4 / cam2.mp4 / cam3.mp4: 960x720 … 747 frames | OK`
- `[detect] ball rate: cam1 24% | cam2 34% | cam3 15% | any 70%`
- 5 `[director]` lines, one per cut:
  cam2 → cam1 (2.0 s) → cam2 (5.3 s) → cam3 (10.7 s) → cam2 (17.2 s) → cam1 (24.0 s)
- A `SUMMARY` block: ball rate, `Cuts: 5`, shot lengths, shortest shot 2.00 s, total time.

The clip is the **last 30 s** of a static panoramic camera
(https://www.youtube.com/watch?v=HBTFZwMdcCw, set in `config.yaml` → `video.url` and `video.trim_last_seconds`).
The ball travels across the whole pitch, so the director really switches cameras.
(The first clip, a Veo follow-cam with the ball always in the middle, is kept in `data/archive/`.)

Steps whose output already exists are skipped. To redo things:

| Command | What happens | Time |
|---|---|---|
| `rm outputs/program.mp4 && venv/bin/python src/main.py` | re-runs director + render only | ~15 s |
| `venv/bin/python src/main.py --force` | re-download, re-crop, re-detect, re-render (needs internet, Node.js and your Roboflow key) | ~13 min |

### Open the results

```bash
open outputs/program.mp4      # the broadcast: 1280x720, "CAM n" label, with sound
open outputs/debug.mp4        # 3 cams side by side, boxes + IDs, ball marker, red LIVE border, scores at the bottom
open outputs/director_log.csv # one row per frame: chosen cam, 3 scores, reason
```

Each "camera" is one third of the 2880x720 panorama (960x720): cam1 = left goal, cam2 = centre, cam3 = right goal.
In `program.mp4` it is letterboxed to 1280x720 (thin black bars on the sides).
In `debug.mp4` watch the red LIVE border follow the ball: left at ~0:03, right at ~0:11–0:17, left again at ~0:24.

### What the director gets wrong (good to know)

With `ball_confidence: 0.15` the director made 6 cuts, one of them wrong: a pitch marking in cam3 was detected
as a "ball" (confidence ~0.2) for many frames and stole the shot from the real ball in cam1 (confidence ~0.9).
Raising the threshold to 0.3 removed it. Try it yourself: set `detection.ball_confidence` to `0.15`, run
`rm outputs/program.mp4 && venv/bin/python src/main.py`, and look for the cut to cam3 at 4.2 s.

### Check that no shot is shorter than 2 s

```bash
venv/bin/python -c "
import csv; rows=list(csv.DictReader(open('outputs/director_log.csv')))
cams=[r['chosen_cam'] for r in rows]; cuts=[i for i in range(1,len(cams)) if cams[i]!=cams[i-1]]
starts=[0]+cuts; print('cuts:',len(cuts),'| shot lengths (s):',[round((b-a)/25,2) for a,b in zip(starts,starts[1:]+[len(cams)])])"
```

---

## 2. The dashboard

```bash
venv/bin/python -m dashboard.server
```

Open http://localhost:8000 in your browser. Stop the server with `Ctrl+C`.

### Replay mode (smooth, real time, for demos)
1. Pick **Replay**, speed **1×**, then click **Start**.
2. You should see:
   - The big **PROGRAM** view with a red **LIVE** badge and the current CAM label.
   - 3 camera tiles with player boxes and a green ball circle (orange = predicted). The live camera has a **red border**.
   - Score bars under each tile (the camera with the ball is around 100–140; the others are low).
   - The **decision log** on the right filling up (BALL_FOUND / BALL_LOST / INFO …). Use the checkboxes to filter by type.
   - The **timeline** at the bottom coloured by the live camera, with a white playhead moving along.
   - The **stats strip**: processing fps (about 30 at 1×), time and frame, ball rate per cam, cuts, average shot.
3. The clip ends after about 30 s, then a few seconds of "finishing videos". Then **download buttons** appear for
   `program.mp4`, `debug.mp4` and `director_log.csv`.
4. Try **2×** (about 15 s) and **0.5×** (about 60 s). Try **Pause** then **Resume**, and **Stop / Reset**.

### Live mode (runs the detector on every frame)
1. Pick **Live**, then click **Start**. The first time takes a few seconds to load the model.
2. It plays at the speed the M1 manages: about **1 fps**, because each frame needs 3 detections of about 300 ms.
   The stats strip shows that real number. A full clip (747 frames) takes about 15–17 minutes.
3. When it finishes it saves fresh detections, so Replay always uses the latest Live run.

---

## 3. Watching the cuts

1. In the dashboard, choose **Replay** and click **Start**.
2. You should see the **red border move** 5 times: CAM 1 at 0:02, CAM 2 at 0:05, CAM 3 at 0:11, CAM 2 at 0:17, CAM 1 at 0:24.
   Each cut appears as a red `CUT camX -> camY (…)` line in the log, and the timeline shows blue / green / amber blocks.
3. Change **min shot** (e.g. 4) or **hysteresis** (e.g. 1.0) in the header and Start again: fewer, later cuts.
   Set them back to 2 and 0.5 afterwards.

---

## 4. Live mode settings (tune per machine)

All in `config.yaml`. Restart the dashboard after changing them (Ctrl+C, then start it again).

```yaml
detection:
  onnx_providers: [CoreMLExecutionProvider, CPUExecutionProvider]   # where the model runs
live:
  parallel_cams: true    # detect the 3 cams at the same time (~25% faster, same results)
  detect_every_n: 1      # 1 = every frame; 3 = every 3rd frame, reuse in between
  realtime: false        # true = video plays at normal speed, detector runs in the background
```

| Machine | Suggested settings | What you see |
|---|---|---|
| **M1 8 GB, best quality** (default) | `detect_every_n: 1`, `realtime: false` | ~1 fps, every frame detected. Also refreshes the Replay cache. |
| **M1 8 GB, faster** | `detect_every_n: 3` | ~2.6 fps, boxes refresh every 3rd frame |
| **M1 8 GB, smooth** | `realtime: true` | ~27 fps video (like Replay), but boxes/ball update only ~every 2 s |
| **Faster Mac / NVIDIA PC** | `realtime: true` | Smooth video. When "detector fps" in the stats strip is close to the video fps, Live looks just like Replay. |
| **NVIDIA GPU** | `onnx_providers: [CUDAExecutionProvider, CPUExecutionProvider]` + `venv/bin/pip install onnxruntime-gpu`; `device: "cuda"` | much faster detection |

Notes on `realtime: true`:
- It starts a second process with its own copy of the model: about 0.5–1 GB more RAM and ~15 s start-up
  (the log says "starting the detector process"; the clip starts when it is ready).
- It does NOT update the Replay cache (only the full-quality setting does).
- The stats strip shows both numbers honestly: "video · detector" fps.

## 5. Single steps (optional)

```bash
venv/bin/python src/download.py   # Step 2
venv/bin/python src/crop.py       # Step 3 (prints the frame-count check)
venv/bin/python src/detect.py     # Step 4 (prints ball rate; add --force to re-detect, ~10 min)
```

## Troubleshooting

- **Streams freeze or buttons stop responding**: keep the dashboard open in only ONE browser tab. Each tab
  holds 4 video streams open, and browsers allow about 6 connections per site.
- **"address already in use"**: another dashboard is still running. Close it, or run `lsof -i :8000` to find it.
- **Detection is very slow (seconds per frame)**: CoreML isn't being used. Run from a normal Terminal, not a sandboxed tool.
- **Roboflow error**: check `ROBOFLOW_API_KEY` in `.env`. If Roboflow fails, the code falls back to a YOLO model automatically.
- **Download fails**: yt-dlp needs Node.js (`node --version`) and internet access.
