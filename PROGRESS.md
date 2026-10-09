# Progress — AI Sports Director MVP 1
Last updated: 2026-10-08 19:30

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
### Clip 1: Veo follow-cam (RVWMU6p0hDQ), first test
- Ball rate (frames with a ball detection): cam1 1% · cam2 53% · cam3 0% · **any cam 54%**.
- Ball tracker statuses: cam2 detected 349 / predicted 118 / lost 193 frames.
- Cuts: **0** (cam2 live for the full 22.02 s). With `start_cam: 1` → 1 cut at frame 60 (2.00 s).
- Processing: detection ~300 ms per cam-frame (≈1 fps live for 3 cams); render ~49 fps; dashboard replay up to ~65 fps.
- Memory: dashboard process ~870 MB during a live run.

### Clip 2 (current): last 30 s of a static panoramic camera (HBTFZwMdcCw)
- Source 3840x960 → scaled to 2880x720 (1080p pixel budget), 25 fps, 747 frames; cams 960x720.
- Ball rate (threshold 0.3): cam1 24% · cam2 34% · cam3 15% · any 70%. Detection ~315 ms/frame/cam.
- **5 cuts**, following the ball: cam2 → cam1 (2.0 s) → cam2 (5.3 s) → cam3 (10.7 s) → cam2 (17.2 s) → cam1 (24.0 s).
  Shots 2.0–6.8 s, average 5.0 s. Checked against debug frames at 1/3/5/10/15/20/27 s.
- With threshold 0.15 there were 6 cuts, one wrong (4.2 s → cam3): a static pitch marking in cam3 detected as
  "ball" (conf 0.17–0.30) beat the real ball in cam1 (conf 0.8–0.96). Threshold 0.3 fixed it.
- Clip 1 (Veo follow-cam, 0 cuts) files are kept in `data/archive/RVWMU6p0hDQ/`.

## Decisions made
- `ball_confidence` raised 0.15 → 0.3 (see Clip 2). The engine applies the current threshold when reading
  cached detections too, so changing it needs no re-detection.
- New clip settings: `video.trim_last_seconds` (keep last N s), `video.max_pixels` (1080p pixel budget, so
  wide panoramas keep detail), `detection.model_aspect` (pad any strip to 16:9 before detection).
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

- Live mode settings (config.yaml `live:` + `detection.onnx_providers`): parallel cams (~25% faster),
  detect_every_n (frame skipping), realtime (detector in a separate low-priority process, so the video
  isn't blocked by Python's GIL). M1 numbers: n=1 ≈ 1 fps; n=3 ≈ 2.6 fps; realtime ≈ 27 fps video with
  ~0.6 detector rounds/s. Only n=1 + realtime=false refreshes the Replay cache.

- **Any-video input** (`input:` in config.yaml, `src/prepare.py`): `split` = one video (link or file) cut in 3;
  `files` = 3 camera clips, each aligned with `start_seconds`, letterboxed to one size, converted to one fps,
  cut to the same frame count. A fingerprint (`data/cams/input.json`) makes main.py redo preparation +
  detection only when the input (or a local file) changed. Input is validated BEFORE old results are deleted.
  Tested: mixed sizes/fps (1280x960@30 + 960x720@25), a 2 s offset (frame-exact: error 2.5 at the right
  frame vs ≥17 one frame off), local-file split, skip/touch detection, 9 error cases, dashboard guard.
- Cross-platform: ONNX providers filtered to what the machine has (CPU fallback); `os.nice` guarded (Windows);
  `src/check_setup.py` checks a new device. Docs: docs/NEW_DEVICE_SETUP.md, docs/USING_YOUR_OWN_VIDEOS.md.

## Problems / open issues
- Ball on a camera border can be seen by 2 cams at once (54 of 747 frames); the director's hysteresis keeps
  this from flickering, but the ball-at-the-edge gets no "center" bonus, so the cut can come a little late.
- ByteTrack IDs churn a bit (79 IDs in cam2 for ~15 people) because players are small and occluded.

## Next action
MVP 1 complete. See TESTING.md.

---

# MVP 2 — Live Story Reader
Spec: `docs/MVP2_STORY_SPEC.md`. Last updated: 2026-10-09

## Status
| Step | Status | Notes |
|---|---|---|
| 0. Plan + environment | done | packages, YAMNet, MVP 1 check OK; Gemini key OK, model gemini-3.5-flash (2–3 s per 6 s window) |
| 1. Intake + live player | done | file/URL/upload → clip.mp4 (≤720p, 25 fps) + audio.wav; Story tab plays with sound; ~5 ticks/s; guard tests 6/6 |
| 2. Fast layer | done | YAMNet (MediaPipe) + loudness + tonal whistle + motion + cuts/replay hint; ~37 ms per 1-s step; cheers 3 s after each of the 3 goals |
| 3. Story engine + logs | done | engine cadence (fast 1 s, story 8 s + triggers, memory 60 s), template narrator, fusion events/score, logs in outputs/story/<clip>/<run_id>/; tests 17/17 |
| 4. Smart layer (Gemini) | done | gemini-3.5-flash-lite (free tier), 27–30 AI lines per 3:22 clip, avg delay 3.3–3.5 s, 429/503/400 + daily-quota fallbacks |
| 5. Memory layer | done | Gemini story so far every 60 s; final summary matches the match (3 goals, white dominated) |
| 6. Specialist (Colab) | done | user's Colab T4 via ngrok: 254 ms per inference, RTF 0.13; shot at 37.0 s reported 3.7 s later |
| 7. Fusion, score, modes | done | 4 modes compared on the same clip (compare.md): hybrid best; no double goals on replays; score white 3 – red 0 |
| 8. Story tab UI | done | full clip plays with sound, all panels update live, no freeze, lag ≤ 0.2 s; Colab off / Gemini quota out fall back |
| 9. Proof mode | done | report.md per run + compare.md; hybrid: goals 3/3, other headline 80%, avg delay 3.8 s; narration 👍 needs user ratings |
| 10. Docs | done | TESTING_MVP2.md, DEMO.md, README MVP 2 section, compare.md |

## Key results (MVP 2)
- Test clips: `eng-cze` (England 3-0 Czechia highlights, 201 s broadcast, youtube kJEY-UFTDk8), `pano` (MVP 1 panoramic, 30 s, no commentary).
- Ground truth `data/clips/eng-cze/truth.csv`: DRAFT written by Claude from frames + on-screen score
  (13 events: 3 goals at 61.5 / 94.0 / 148.5 s, 6 shots, 2 saves, 1 free kick, 1 corner). Please verify.
- specialist_only on the Mac CPU (not live-capable): shots at 37 s and 92.6 s and 123.8 s found; RTF ~7, delay ~35 s.
  6 of 7 truth events up to 100 s had intensity in the run's top 20%.
- Intake: 201 s YouTube clip downloaded + normalized in 67 s; 30 s local file in 3 s.
- Browser → backend playhead: ~5 ticks/s, follows play / pause (checked with headless Chrome).
- eng-cze goals (from the on-screen score): 1-0 at ~60 s, 2-0 at ~93 s, 3-0 at ~146 s. Fast layer CHEER triggers at
  63, 96, 150 s (+ big chances at 26, 39, 126 s; intro jingle at 10 s). Live run: lag ≤ 0.5 s.
- Whistle check: a synthetic referee whistle at commentary level is caught in 198/200 seconds; on the clean
  clip only 4/200 seconds fire (end music / final whistle area).

## Decisions made (MVP 2)
- `google-genai` pinned to 2.8.0: newer versions need pydantic ≥ 2.12.5, but Roboflow `inference` (MVP 1)
  needs pydantic < 2.12. 2.8.0 has everything we need (structured output, inline video, VideoMetadata fps).
- MediaPipe 1.1.0 runs YAMNet on the M1 CPU in ~4 ms per second of audio. Model in `models_cache/yamnet/` (gitignored).
- Fast layer, audio: YAMNet hears mostly "Speech" on TV highlights (crowd classes ≤ 0.09), so excitement is
  35% crowd classes (x3 gain) + 65% loudness vs the median of the last 30 s (past only). YAMNet "Whistle"
  was 0 on the whole clip, so the whistle needs a strict tone check: one narrow peak (≥ 50% of the 2–4.5 kHz
  band energy, ≥ 25 dB above the band median) for ≥ 4 frames in a row. Speech harmonics fail it.
- CHEER = excitement ≥ 0.45 and ≥ 0.25 above the quietest of the last 4 s, max once per 6 s, not in the first 10 s.
- Replay hint = ≥ 3 hard cuts within 8 s. On edited highlights this also fires on normal fast editing, so it is
  only a hint; Gemini's `is_replay` is the main replay signal when Gemini runs.
- Engine: AI calls run in a worker thread; `logged_at` = playhead when the answer ARRIVED. Templates (no AI) are
  used in fast_only/specialist_only and whenever Gemini is unavailable, so the commentary never stops.
- Fusion dedupe: same type within 4 s = one event (goals 60 s, so a goal shown again later can't count twice);
  confidences combine noisy-or; `logged_at` keeps the first report.
- Specialist model facts (read from opensportslib 0.3.1 source + the HF config.yaml): frames 224x398 RGB, /255,
  ImageNet mean/std; stride int(25 / extract_fps 2) = 12 -> ~2.08 fps; clip_len 100 (~48 s); output per frame =
  softmax over background + 12 classes. Server calls `E2EModel.predict()` on a rolling buffer (only frames already
  sent = past). The GRU looks both ways in time, so a frame is reported once it is `report_delay_s` (2 s) old.
- Server code lives in `colab/specialist_server.py` (tested locally on CPU, same code path as Colab) and is copied
  into the self-contained notebook by `colab/build_notebook.py`. Tunnel: ngrok (secret NGROK_AUTHTOKEN) or a
  Cloudflare quick tunnel without an account.
- Only Gemini's `is_replay` blocks new events. The scene-cut replay hint fired on normal highlight editing and
  would have dropped the specialist's 37 s shot (`fusion.fast_replay_blocks_events: false`).
- Default model switched to `gemini-3.5-flash-lite` (fallback `gemini-3.1-flash-lite`) after hitting the free
  tier's 20 requests/day on gemini-3.5-flash. Flash Lite rejects `thinking_budget: 0` with a generic 400: the
  provider retries once without it. A daily-quota 429 switches to the fallback model for the rest of the run.
- Bug found by the causal guard: the engine rounded the window end UP (8.168 -> 8.17 > playhead) -> refused.
  Now rounded down; regression test added. (It had silently dropped the cheer-triggered looks at the goals.)
- Replays: Flash Lite called live goals "replays" (and the scene-cut hint in the prompt pushed it that way). Hint
  removed from the prompt, stricter replay definition, per-event `in_replay` flag (only replayed events dropped).
- Goal dedupe: a 60 s merge window merged goal 2 (33 s after goal 1 in the highlights) into goal 1. Now the
  scoreboard decides: a goal with a scoreboard total above the counted goals is new; the 60 s window still
  merges replays when no new score is shown.
- Previous model note: `gemini-3.5-flash` (fallback `gemini-2.5-flash` on HTTP 503). 45 models on the key; newest Flash
  `gemini-3.8-flash` answered a one-word test in 65 s / 15 s and gave 503 "high demand"; `gemini-3.7-flash` gave
  503 on 2 of 3 real windows and 28 s on the third. `gemini-3.5-flash` on real 6 s windows: 2.1–2.7 s, ~1.5k tokens,
  goal found at 60.8 s (truth 61.5), replay at 78–84 s correctly flagged with no events, scoreboard read (2-0).
  `check_gemini` now skips models slower than `llm.max_test_latency_s` (5 s) on a warm test call.

## Problems / open issues (MVP 2)
- 2026-10-09 user runs on yt-lfzra492gdw (LaLiga Real Madrid–Barcelona, 2:22): 1st hybrid run 3 goals, 2nd only 1.
  Causes found and fixed: (1) 6 s looks every 8 s left a 2 s blind gap -> window_s 9 (overlap);
  (2) a second open Story tab parked at 2:20 sent its own playhead -> only the tab that pressed Go Live drives
  the clock now; (3) Flash Lite misreads the scoreboard sometimes ("3-1" for 2-1) -> goals < 20 s apart always
  merge; (4) Flash Lite invents goals (free kick on the crossbar at 38 s on eng-cze) -> with a visible scoreboard a
  goal is "pending" until the score graphic goes up, dropped after 30 s. The "goal at 115 s" on yt-lfzra492gdw
  was a disallowed goal (referee signalling 2:00–2:03, scoreboard stayed 2-1): now correctly not counted.
- Team attribution by Flash Lite is unreliable (Barcelona's goal reported as "white"): use the score ± buttons, or
  gemini-3.5-flash with a paid key.
- Specialist-only mode cannot find goals on highlights (the model never fires GOAL there), by design of the modes.
- Free Gemini tier: Flash models 5 req/min + 20 req/day (one clip needs ~30) -> default is Flash Lite (15/min, 500/day).
  Flash Lite over-labels replays as live (replayed shots become false alarms; replayed goals are merged by the
  scoreboard/60 s rule). With a paid key, gemini-3.5-flash recognised replays correctly in tests.
- Narration 👍 target (≥ 80%) is unmeasured until the user rates lines in a run.
- truth.csv for eng-cze is a Claude draft; the user should check it.
- Honest test (no audio to Gemini): 2/3 goals (the goal-2 window had no clear visual of the ball crossing the line).
- Specialist on the M1 CPU: 12–23 s per 48 s window (RTF 6–11): too slow for live, hence Colab.
- The spotting model never fires GOAL on eng-cze (offline, full context): it finds the SHOT for goals 1 and 3,
  misses goal 2. Goals therefore rely on Gemini (or specialist SHOT + Gemini goal).
- Seeking is blocked during a run (live means live; the AI already "saw" that part). The Playhead only moves forward.
- `.env` has no GEMINI_API_KEY yet.

## Next action (MVP 2)
MVP 2 complete. User: rate commentary lines (👍/👎) in a run to measure narration accuracy, check truth.csv,
and see TESTING_MVP2.md / DEMO.md. MVP 1 quick check passed on 2026-10-09 (5 cuts, shortest shot 2.00 s).