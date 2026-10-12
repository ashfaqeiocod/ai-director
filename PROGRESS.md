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
- 2026-10-09 longer test: esp-cro (Spain 4-1 Croatia extended highlights, 9:49, Sony LIV ZN7ZYybxpxM), goals-only
  truth drafted by Claude from the score graphic (75.5, 179.5, 214.5, 443.5, 521.5 s), fast + Gemini (Flash Lite),
  Colab offline. Score graphic read correctly every time (0-0 -> ... -> 4-1). Goals at the right moment (±10 s):
  run 1 0/5, run 2 2/5 after fixes (merged goals keep the EARLIEST time; the story prompt no longer gives Gemini the
  counted score, which it copied into its "score on screen" answer). Flash Lite mostly recognises goals in the
  replay 15-25 s later. Tried and REMOVED: re-timing a confirmed goal to the loudest crowd roar between the
  old/new score readings (errors -30 to +100 s, because Flash Lite's score readings lag).
  Conclusion: on the free tier, trust the score graphic for the score; for goal timing use gemini-3.5-flash
  (paid key) and/or the Colab specialist (hybrid).
- NOW card: big score = stable score-graphic reading (same value read twice in the last 3 looks); small line =
  goals counted per colour.
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
---

# MVP 3 — Directed Story
Spec: `docs/MVP3_SPEC.md`. Plan: `docs/AI_LIVE_DIRECTOR_PLAN.md`. Last updated: 2026-10-11

## Status
| Step | Status | Notes |
|---|---|---|
| 0. Plan + data access | done | HF token OK; 128057 2nd half (3.45 GB, 4096x1080, 25 fps, 70,350 frames, audio); SoccerNet NDA requested |
| 1. Intake + calibration check | done | dev segment cut; annotation offset measured (-23 frames); overlays checked by eye |
| 2. Virtual cameras | done | 5 cams x 9000 frames, ~15 fps render (libx264) |
| 3. Detection + cards | done | football model on CoreML (sandbox off), ~0.6-1 frames/s; ball ~16% of frames; parked spare ball ignored |
| 4. Story layer | done | close-up feed (delay ~4 s); goals 0/4 on the panorama, 3/4 camera-only on broadcast |
| 5. Director v2 | done (tuning) | 7 unit tests; oracle smoke test: cuts at exactly 3.0 s |
| 6. Run + render | done (tuning) | program.mp4 + debug.mp4 + logs, ~9 fps |
| 7. Studio tab | built, untested in browser | /studio |
| 8. Evaluation | done (first version) | ball on air (all / not WIDE), WIDE share, shots, reactions + false reactions, story R/P, bad feed, baselines |
| 9. Docs | in progress | TESTING_MVP3.md draft; plan doc to update with the findings |

## Key results (MVP 3)
- Baseline on dev (never cut): always CENTRE = 52% ground-truth ball on air; always WIDE = 100% but 100% WIDE time.
- Detector on this 4K panorama (60 sample frames, 3 cams): football model finds the ball in 9/57 frames (~16%),
  345 ms per cam-frame on CoreML. Tiling (InferenceSlicer 640) 2/57 at 2.8 s; ball-only model + tiling 2/57 at 1.4 s.
  So the plain model stays. Worse than the plan's "4 in 10": the ball is a few pixels on this camera.
- Story on the WIDE panorama (dev, Flash Lite, 44 looks, avg delay 6.2 s): missed the goal at 62 s completely and
  repeated "blue builds from the back" for 6 min. The goal is real (attack on the left goal at 61-62 s, kick-off
  at ~85 s) but players are a few pixels tall on the panorama.

## Decisions made (MVP 3)
- Footage: SoccerTrack v2 (CC BY 4.0). No open dataset has full matches from several real broadcast angles
  with sound (see plan section 14), so the 4K panorama is cut into 5 virtual cameras.
- Match 128057 (M7, a dataset TEST match), 2nd half: 5 goals at frames 2282, 37446, 46854, 53875, 67791.
  Dev segment 00:30-06:30, eval segment 23:00-37:00.
- The dataset's ball track is rebuilt from the annotated events (straight lines between them), so it knows
  the future: scoring only. Per-frame player positions (gsr) may drive an "oracle" mode, always labelled.

- Annotation offset: on dev the annotated players lined up ~23 frames EARLIER than expected. Cause found: MY cut,
  not the dataset. The half's video stream starts at timestamp 1.0 s and the fast seek didn't allow for it, so dev
  starts 24 frames early (checked by decoding the half frame by frame). The dataset IS in sync. dev keeps its
  measured offset (-23, corrects everything; no need to redo it); new cuts select frames by their original
  timestamps (-copyts) and were verified frame-exact (frames 0/250/499 of a 20 s test cut). prepare still measures
  the offset as a safety check (`annotation_offset` in segment.json); lookups use annotation_frame()/segment_time().
- Segment cuts stop decoding right after the segment (-frames:v): the dev cut took 5.5 min because ffmpeg read
  the whole 47-min half.
- Video encoding: libx264. VideoToolbox (hardware) refused 5 parallel sessions ("cannot create compression session").
- Ground-truth player boxes cover the feet area (small boxes), fine for counting and aiming.
- Director: the oracle "players group" gets MVP 1's centred bonus (else it tied with WIDE at 70 and never cut).
- Story feed: default is the CLOSE-up (detector-aimed for honest runs, oracle-aimed for oracle runs), with the
  match sound added. A story run with lag skips or > 3 AI errors is marked invalid and never used.
- Scoring: "ball on air" counts WIDE as always right, so the report leads with ball on air EXCLUDING WIDE time,
  plus WIDE share, false reactions and never-cut baselines (compare.md).

## Problems / open issues (MVP 3)
- SANDBOX: inside Claude Code's sandbox, CoreML cannot write its compiled model to the macOS temp folder, so
  MVP 1's Detector silently falls back to generic YOLO (ball rate ~2%). This also affects MVP 1's live mode when
  run from the sandbox. Detection was therefore run with the sandbox disabled. studio.detect now refuses the fallback.
- A story pass run at the same time as detection fell 240 s behind and lost Gemini calls: discarded. Real-time
  story passes must run alone.
- Detection speed with the real model while other jobs ran: 0.44 frames/s (3 cams); to re-measure alone.
- Gemini free tier: ~500 Flash Lite calls/day; a 14-min eval story pass needs ~120. Plan eval passes per day.
- The HF token was pasted in chat: regenerate it on huggingface.co when setup is done and update `.env`.
- Disk: ~19 GB free before the 6.7 GB download.

- dev results (detector + hybrid story, Colab T4): ball on air excl. WIDE 77% (always-CENTRE baseline 52%),
  WIDE time 20% (69% before the players' group could stand in for an unseen ball), shortest shot 3.0 s,
  degraded cam 0% on air, goal reaction 0/1 (no layer saw the goal).
- Story on the close-up (dev, oracle aim, Flash Lite): avg delay 4.0 s (wide: 6.2 s), saw the kick-off after the
  goal (87.6 s, true ~85 s) but not the goal. Colab specialist (hybrid): no GOAL; many headers/crosses that the
  truth doesn't have (dev truth: 76 passes, 62 dribbles, 1 free kick, 1 goal) -> it doesn't transfer to this camera.
- Audio: no crowd at this amateur match (YAMNet crowd ~0); "cheer" triggers every 10-30 s, so audio can't flag goals.
- eval, ORACLE run (annotated players aim + score; Gemini story on the oracle close-up, 109 looks, 0 errors,
  avg delay 5.0 s): ball on air excl. WIDE 71% (target 80% missed), WIDE 11%, 71 cuts (5.1/min, avg 11.7 s),
  shortest 3.0 s, degraded cam2 0% on air, goal reactions 0/3, story shots recall 40% / precision 67% (delay 10 s),
  goals 0/3. Gemini called "shot" 6 s and 5 s before goals 2 and 3, and "kick-off" 26 s after goal 1: it sees
  the attacks and restarts, never the goal itself.
- eval, DETECTOR run (final, honest; Gemini on the detector close-up, 117 looks, 0 errors, delay 4.2 s):
  ball on air excl. WIDE 74% (target 60% met; always-CENTRE baseline 43%), WIDE 24%, shortest normal shot 3.0 s
  (one 0.24 s shot = the allowed escape from the blurred camera), degraded cam2 0% on air, 135 cuts (9.6/min,
  avg 6.2 s: busier than a human would cut), goals 0/3 found -> 0/3 reactions, 0 false reactions.
  compare.md: outputs/studio/eval/compare.md.
- CAMERA-ONLY TEST on real broadcast (SoccerNet, Arsenal 3-4 Liverpool 2016, 2nd half 0-20 min, 4 goals;
  NO sound to the AI, score graphic blacked out, Flash Lite, real time, 114 looks):
  goals 3/4 within +-10 s (608.5, 1051.3, 1123.3 s vs truth 608.4, 1051, 1123; the 4th at 226 s vs 209 s, with
  celebrations reported at 212/218 s), 2 false goals = REPLAYS of goals called live (none in raw camera feeds),
  shots 4/6. So the AI CAN see goals from the cameras alone when the camera shows them well. On SoccerTrack
  (one far panoramic camera) it found 0/4. Conclusion: camera placement/framing (close-ups, a view near the
  goal) matters more than commentary. Note: Gemini still read a score once (another banner), not used for goals.
- Two Claude sessions worked on the project at once (2026-10-12): the other one also ran eval detection, added the
  parked-ball filter (commit 8a8764a) and an eval story pass; they clashed on the Gemini quota. One session only.
- 2026-10-12 ~07:00: the MacBook was on BATTERY (4%) and kept sleeping: renders slowed (eval cams 74 min), and an
  eval story pass "played" while asleep with 0 Gemini looks (marked invalid; story_pass now detects this).
- REAL multi-camera footage (2026-10-12): MUVS (Zenodo 10.5281/zenodo.20708683, CC BY 4.0), fans' phones filming
  Chicago Fire 0-1 Charlotte FC (MLS, 2024-05-15, goal 60') from 3 seats at once, synced to 0.07 s, plus a human's
  camera pick per 3 s. No sound track. `python -m studio.muvs --period 19` -> data/muvs/P19/cam1-3.mp4 (1080p,
  25 fps, 2875 frames each, 115 s), human_cuts.csv, meta.json; 4K originals deleted. P19 = 2nd-half STOPPAGE
  TIME (ribbon: CHI 0 CLT 1, 48:43): Chicago attack, CORNER KICK (~75-105 s, on the ribbon) and a goalmouth
  scramble at ~105 s (likely the keeper's late double save from the match report). Cam views: cam1 high corner,
  cam2 side, cam3 low behind a fan's head (natural "bad feed"). Human picks: mostly cam2 (29 of the last 30).
  Next MUVS stretch: P15 (5.6 min, 3 cams, 7.6 GB). Other searches (PolyU/Zhejiang 5-7 cam set: author emailed
  2026-10-12; Alfheim; MVFouls NDA) are in the "MVP 3 Footage Recommendation" doc.
- No-sound intensity fix (2026-10-12, studio/story_pass.py `without_missing_sound`): story intensity is 0.4 audio +
  0.2 motion + 0.2 specialist + 0.2 llm; on a video with NO sound track audio was stuck at 0 but still counted
  (max 60; same strong motion scored 30). Now audio weight -> 0 for such videos only (same motion: 90). Videos
  with a sound track (MVP 2 clips, SoccerTrack) are unchanged; MVP 2 defaults untouched.
- Vision-only intensity (studio/intensity.py, settings studio.vision_intensity): detector at 5 fps on each cam,
  phone movement removed (phase correlation, sign checked), pressure near goal, player speed, STOP trigger.
  P19 (12.8 min detection, football model): v1 mean 49 inside the corner/scramble window (75-110 s) vs 47
  outside; v2 (only players standing on pitch-green grass, per-second max pressure) 67 vs 57, peak at 111 s
  (just after). Real hit: STOP triggers cluster at 86-93 s = play stopped for the corner. Weak part: the model's
  "goalkeeper" label is unreliable on fan footage (linesman in yellow, stewards = "goalkeeper"), so pressure
  saturates. Player bunching per cam rises on cam2 toward the corner (7.5 -> 12.4) but a high cam (cam1) looks
  bunched all the time. Verdict: a helper signal, not enough alone; meaning must come from specialist + Gemini.
  Tuned on P19 only: check on P15 before trusting any setting. The whole clip is late attacking play, so its
  "calm" baseline is not calm.

## Next action (MVP 3)
PLUG THE MAC IN. Then eval: story pass close_oracle (14 min, alone) -> detection (~2.5 h, sandbox off)
-> close-up aimed by the detector -> story pass close_detector -> 2 runs + reports + compare.md.
dev is done (eval cameras are rendered, eval segment cut).
