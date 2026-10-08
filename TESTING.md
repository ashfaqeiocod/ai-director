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
- `[download] … 22.08s | fps 29.970 | 1920x1080 | 660 frames | audio: True`
- `[crop] cam1.mp4 / cam2.mp4 / cam3.mp4: 640x1080 … 660 frames | OK`
- `[detect] ball rate: cam1 1% | cam2 53% | cam3 0% | any 54%`
- A `SUMMARY` block: ball rate, `Cuts: 0`, average shot, total time.

Steps whose output already exists are skipped. To redo things:

| Command | What happens | Time |
|---|---|---|
| `rm outputs/program.mp4 && venv/bin/python src/main.py` | re-runs director + render only | ~15 s |
| `venv/bin/python src/main.py --force` | re-download, re-crop, re-detect, re-render (needs internet, Node.js and your Roboflow key) | ~12 min |

### Open the results

```bash
open outputs/program.mp4      # the broadcast: 1280x720, "CAM 2" label, with sound
open outputs/debug.mp4        # 3 cams side by side, boxes + IDs, ball marker, red LIVE border, scores at the bottom
open outputs/director_log.csv # one row per frame: chosen cam, 3 scores, reason
```

In `program.mp4` you see a tall strip centred in a black 16:9 frame. That is expected:
each "camera" is one third of the original 1080p frame (640x1080), letterboxed to 1280x720.

### Why there are 0 cuts

The source is a Veo follow-cam: it pans to keep the ball in the middle of the picture.
We checked the full uncropped frames, and the ball was in the middle third every time it was visible.
So cam2 always has the ball and the director correctly stays on cam2.
MVP 1 proves the pipeline, not the director quality. Section 3 shows how to see a real cut.

### Check that no shot is shorter than 2 s

```bash
venv/bin/python -c "
import csv; rows=list(csv.DictReader(open('outputs/director_log.csv')))
cams=[r['chosen_cam'] for r in rows]; cuts=[i for i in range(1,len(cams)) if cams[i]!=cams[i-1]]
starts=[0]+cuts; print('cuts:',len(cuts),'| shot lengths (s):',[round((b-a)/29.97,2) for a,b in zip(starts,starts[1:]+[len(cams)])])"
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
   - The big **PROGRAM** view with a red **LIVE** badge and "CAM 2".
   - 3 camera tiles with player boxes and a green ball circle (orange = predicted). CAM 2 has a **red border**.
   - Score bars under each tile (CAM 2 is around 100–140; the others are low).
   - The **decision log** on the right filling up (BALL_FOUND / BALL_LOST / INFO …). Use the checkboxes to filter by type.
   - The **timeline** at the bottom turning green (cam2) with a white playhead moving along.
   - The **stats strip**: processing fps (about 30 at 1×), time and frame, ball rate per cam, cuts, average shot.
3. The clip ends after about 22 s, then a few seconds of "finishing videos". Then **download buttons** appear for
   `program.mp4`, `debug.mp4` and `director_log.csv`.
4. Try **2×** (about 11 s) and **0.5×** (about 44 s). Try **Pause** then **Resume**, and **Stop / Reset**.

### Live mode (runs the detector on every frame)
1. Pick **Live**, then click **Start**. The first time takes a few seconds to load the model.
2. It plays at the speed the M1 manages: about **1 fps**, because each frame needs 3 detections of about 300 ms.
   The stats strip shows that real number. A full clip takes about 12–15 minutes.
3. When it finishes it saves fresh detections, so Replay always uses the latest Live run.

---

## 3. How to see a cut

The clip never needs a cut (see above), so force one:
1. In `config.yaml`, change `start_cam: 2` to `start_cam: 1`.
2. In the dashboard, choose **Replay** and click **Start**. Restarting the server isn't needed; config is re-read on every Start.
3. For the first 2 s CAM 1 is live and has no ball. At **0:02** (the minimum shot length) the **red border jumps to CAM 2**.
   The log shows a red `CUT cam1 -> cam2 (cam2 scored higher for 0.5s)`, and the timeline turns from blue to green.
4. Try **min shot = 4** in the header: the cut then happens at 0:04.
5. Change `start_cam` back to `2` afterwards.

---

## 4. Single steps (optional)

```bash
venv/bin/python src/download.py   # Step 2
venv/bin/python src/crop.py       # Step 3 (prints the frame-count check)
venv/bin/python src/detect.py     # Step 4 (prints ball rate; add --force to re-detect, ~10 min)
```

## Troubleshooting

- **"address already in use"**: another dashboard is still running. Close it, or run `lsof -i :8000` to find it.
- **Detection is very slow (seconds per frame)**: CoreML isn't being used. Run from a normal Terminal, not a sandboxed tool.
- **Roboflow error**: check `ROBOFLOW_API_KEY` in `.env`. If Roboflow fails, the code falls back to a YOLO model automatically.
- **Download fails**: yt-dlp needs Node.js (`node --version`) and internet access.
