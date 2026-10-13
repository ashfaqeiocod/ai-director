# MVP 3 — Live test environment: implementation flow

Written 2026-10-13 for a NEW Claude Code session. Read `CLAUDE.md`, `docs/MVP3_SPEC.md` and the MVP 3 section of
`PROGRESS.md` first. All MVP 3 rules apply: code in `studio/` (+ dashboard), `src/` and `story/` only get
ADDITIVE changes with unchanged defaults, never read beyond the playhead, commit after each step, never commit
`.env`, footage, model files or outputs, run the MVP 1 quick check (`TESTING.md` section 1) at the end.

## Goal (in the user's words)
"Use the current dashboard. Start from anywhere on the timeline: the story starts like a new session, exactly
as if the broadcast started there. A REAL live test: the detector runs while the match plays (on Colab's GPU),
latency handled correctly. Show every latency (Gemini too) on the timeline and list all events in the dashboard."
Keep it simple: no new pages, no new modes beyond what is listed here.

## Where things stand (2026-10-13 evening)
- Main match: **Alfheim** `alfheim-2013-11-03-H1` (47 min, 3 fixed cameras, 30 fps, 1280x960, no sound).
  `data/alfheim/2013-11-03_H1/`: cam1-3.mp4, cam1-3_480.mp4 (light copies), story_mosaic.mp4, calibration.json,
  meta.json. Detection (cached, for fallback + comparison) runs unattended: `data/alfheim/pipeline.log`,
  checkpoints in `detections/camN.json` every 300 frames per camera; to resume after a stop just run it again:
  `venv/bin/python -c "from pathlib import Path; from studio.intensity import detect; detect(Path('data/alfheim/2013-11-03_H1'))"`
  (own Terminal or outside the sandbox: CoreML).
- **Memory warning:** the M1 has 8 GB. On 2026-10-13 15:50 swap was 8.8/10 GB (IDE + 3 Claude sessions + Chrome)
  and detection stalled ("stuck", 0% CPU). Close other apps while detection or a live run is going.
- Gemini: Lite-first chain + 503 retry on the next model + `python -m studio.gemini_check` (commit e9cd2be).
- Live mode today (`studio/live.py` `LiveStudio`): wall clock = match clock, starts at 0:00, detections are
  READ FROM THE CACHE (`src.detection_at`), Gemini story + goal watcher + Colab action model run live.
- Colab notebook `colab/specialist_server.ipynb` (built from `colab/specialist_server.py` by
  `colab/build_notebook.py`): action-spotting model only, WS `/stream`, tunnel via ngrok or Cloudflare.
- The user has added `ROBOFLOW_API_KEY` to Colab's Secrets.
- Detector facts: Roboflow `football-players-detection-3zvbc/11` via `inference`; the ONNX model takes a
  1280x1280 input (33600 anchors). `src/detect.py` `Detector.detect` pads a frame to 16:9, keeps players >= 0.3,
  the single best ball >= 0.3. M1: ~0.4 s per frame -> too slow for live (3 cams x 5/s = 15 frames/s).

---

## Step 1 — Start from anywhere (Mac only)

### 1a. Story engine: optional start time (additive, default 0)
`story/engine.py` `StoryEngine.__init__(..., start_s: float = 0.0)`:
- `self.playhead = Playhead(start_s)`, `self.fast_done_to = start_s`,
  `self.next_story_at = start_s + interval_s`, `self.next_memory_at = start_s + memory.interval_s`.
- Without this the fast layer would race through every old second (or log a misleading LAG skip) and the first
  story window would not be the new start.
- `story/specialist_client.py`: its `sent_until` starts at 0 -> start at the engine's start (pass it through, or
  read `reader.playhead.now()` when the session opens).
- Test: an engine started at 600 s asks for its first story window [600 + 8 - 9, 608], first memory at 660 s,
  never reads a frame < 600 - window (CausalReader still guards the future).

### 1b. Live loop: `LiveStudio(..., start_s=0.0)`
`studio/live.py`:
- Seek every capture (light and full) to `start_frame = round(start_s * fps)`; `i = start_frame - 1`.
- Clock: `start_s + (time.time() - wall_start) * speed` (capped at the end).
- A fresh pipeline (teams are learned again in the first seconds, possession/score start at 0): that IS the
  "new session". The goal watcher and story engine get `start_s` (story engine: 1a).
- Recording: `run.json` gets `"start_s"`; `director_log.csv`, `map.jsonl`, events keep ABSOLUTE match times.
  `program.mp4` starts at `start_s`: Replay must play it with that offset (studio_api + page).
- `summary` adds `start_s` and `end_s`; "valid" rules unchanged.
- CLI: `--start 23:40` (mm:ss or seconds).

### 1c. Dashboard
- `dashboard/studio_api.py` live start endpoint takes `start_s` (validate 0 <= start_s < duration - 10).
- `dashboard/static/studio.html` Live setup panel: a "Start at" box (mm:ss, default 0:00). While the setup panel
  is open, the timeline shows the whole match; a click on it fills "Start at". The run picker shows
  "from 23:40" for runs that did not start at 0.
- Replay of a run that started later: the timeline still shows absolute match time; the program video offset is
  `start_s`.

Done when: Go live at 23:40 -> cameras show 23:40, the first caption covers ~23:40-23:49, "story so far" builds
from there, no LAG lines; the run replays in sync.

---

## Step 2 — The detector on Colab's GPU (real live)

### 2a. Colab server: `/detect`
`colab/specialist_server.py` (then `python colab/build_notebook.py` to rebuild the notebook):
- `pip install inference-gpu` in the install cell. CHECK it does not break `opensportslib`'s torch; if it does,
  run the detector in its own process (a second uvicorn on another port behind the same tunnel path, or a
  subprocess the server talks to). Note the result in PROGRESS.md.
- Key: `from google.colab import userdata; os.environ["ROBOFLOW_API_KEY"] = userdata.get("ROBOFLOW_API_KEY")`
  (never print it). Load `football-players-detection-3zvbc/11` once, warm up, report ms per frame.
- SAME output as `src/detect.Detector.detect`: pad to 16:9 (copy `pad_to_aspect`), players (player, goalkeeper,
  referee) >= 0.3, one best ball >= 0.3, boxes in the ORIGINAL picture's pixels.
- WS `/detect`: client sends binary = 4-byte header length + JSON `{"frame": int, "cam": int, "sent": float}` +
  JPEG; server replies JSON `{"frame", "cam", "sent", "players", "ball", "infer_ms"}`. Batch the 3 cameras of one
  frame if the client sends them together (faster on GPU).
- `/health` adds `"detector": {"model", "ms_per_frame", "ready"}`.
- Self-test cell: 30 synthetic frames -> ms per frame; must be well under 66 ms (15 frames/s).

### 2b. Mac client: `studio/remote_detect.py`
- `RemoteDetector(url)`: a background thread with one websocket; `submit(frame, {cam: image})` never blocks
  (JPEG quality 85 at the full 1280x960: the model works at 1280, the 640 light copies would lose the ball);
  at most 2 frames in flight, older unsent ones are dropped and counted.
- `results()` returns the answers that arrived since the last call, with `rtt_ms` and `age_s`
  (= live clock now - frame time).
- Status for the page: connected / ms per frame on GPU / round trip / dropped.

### 2c. Live loop uses it (`studio/live.py`)
- New option `detector = "cached" | "colab"` (setup panel; "colab" needs the Colab URL test to show the detector).
- Full-resolution pictures of ALL cameras are needed only at detection frames (5/s): keep one full-res capture
  per camera and `grab()` the frames in between, `retrieve()` only at detection frames. Measure the CPU cost on
  the M1; if it drops the shown fps below 20, send 960x720 instead and compare ball rates.
- Results are applied WHEN THEY ARRIVE, in frame order, with their own frame time:
  `pipe.detections(det_frame, det_frame / fps, detector=result, images=images_of[det_frame])` (keep the last
  ~2 s of detection-frame light images in a small ring buffer for the shirt colours). A result older than
  `max_age_s` (config, start 1.5 s) is ignored and counted. The director therefore acts on what it knew at
  that moment: honest live behaviour; no program delay buffer (keep it simple).
- Fallback: Colab not connected / no answer for `max_age_s` -> use the cached detection of that frame if it
  exists, status "detector: cached (Colab late)"; never block the loop.
- Record `detections_live.jsonl` (frame, cam, sent, received, rtt, players, ball) so the run can be replayed
  and compared.

### 2d. Parity check (before trusting it)
60 s of Alfheim, Colab vs cached: ball rate per camera, players per frame, director cuts. Expect the same
within a few %. Write the numbers into PROGRESS.md.

---

## Step 3 — Latency on the timeline + all events

### 3a. Collect (live, recorded in `latency.jsonl`, also published as `{"type": "latency", ...}`)
| source | happened (`event_time` / window end) | known (`logged_at`) | extra |
|---|---|---|---|
| Gemini story | window_end | logged_at | model, key number, request seconds (from the LLM call log) |
| Gemini goal look | look window end | logged_at | model, answer (goal/shot/save) |
| Colab action model | event_time | logged_at | class, confidence |
| Detector | frame time | arrival | rtt_ms, GPU ms, cached/colab |
| Map trigger / director cut | time | same time | (0 delay, for reference) |

### 3b. Show (dashboard, Live and Replay the same)
- Timeline lane "Latency": each AI item is a thin bar from "happened" to "known"; colour by source; hover =
  source, model, delay. Long bar = slow, readable at a glance.
- Small strip above the timeline: Gemini delay (last + average), Colab delay, detector round trip, frames shown
  per second.
- Feed tab "All events": one table — happened, known, delay, source (+ model), what. Click = jump there
  (Replay). Filter by source.
- Run summary (run.json + report): average / 90% delay per source, results ignored as too late.

---

## Step 4 — Proof runs and report
1. `python -m studio.gemini_check` + Colab Test (both lights green).
2. Alfheim Live, Hybrid + Colab detector, three 5-minute starts (0:00, 23:40, 41:00) -> check the new-session
   behaviour and the latency numbers.
3. One full half (47 min), Hybrid + Colab detector. Keep the Mac free (memory!).
4. Compare with the cached-detector run: cuts, ball on air (where measurable), danger coverage, delays.
5. PROGRESS.md entry + TESTING_MVP3.md section 0A updated; update the HTML report artifact
   (`scratchpad/report/src.html` -> republish) if the user asks.

## What the user does (and when)
- Now: nothing on Colab until Step 2a is built (the notebook changes).
- After Step 2a: the secret must be named exactly `ROBOFLOW_API_KEY` with "Notebook access" on; open the new
  notebook -> Runtime -> T4 GPU -> Run all -> copy the printed URL -> Studio -> Live -> Colab URL -> Test.
- Before a long run: close other apps (8 GB Mac), run `python -m studio.gemini_check`.

## Order of commits
1. Step 1a+1b (+ tests) · 2. Step 1c · 3. Step 2a (notebook) · 4. Step 2b+2c (+ tests with a fake server)
· 5. Step 2d numbers · 6. Step 3 · 7. Step 4 docs.
