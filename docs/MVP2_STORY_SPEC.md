# MVP 2 — Live Story Reader (build spec for Claude Code)

## 1. Goal
Prove that AI can watch a sports clip **as if it were live** and continuously understand and log
**what is happening, how intense it is, how it feels, and the story so far**, shown live in the dashboard.

- Test sport: soccer (sport-specific parts live in config, so other sports can be added later).
- Input: ANY soccer clip the user drops into the dashboard (file upload or YouTube URL).
- MVP 2 is separate from camera switching (MVP 1). Do not change MVP 1 behaviour.

**Demo in one sentence:** the user drops a clip, presses **Go Live**, the video plays at normal speed with sound,
and the Story tab fills up live with an intensity meter, events, mood, commentary and "story so far".

## 2. Non-negotiable rules
1. **Causal / no peeking.** No component may read video or audio beyond the current playhead. Enforce it in code
   with a guard class (`CausalReader`) that raises `FutureAccessError` if anything asks for t > playhead.
   Add a unit test for this guard.
2. **Real speed.** The browser `<video>` element is the master clock (it plays the clip with sound).
   It sends its `currentTime` to the backend over WebSocket ~5 times per second. The backend processes up to
   that playhead only. If the backend falls behind, it skips ahead (like a live system) and logs the lag.
3. **Honest latency.** Every story item stores both `event_time` (when it happened in the clip) and
   `logged_at` (playhead time when the AI produced it). The dashboard shows the delay.
4. **MVP 1 must keep working.** At the end, run the MVP 1 quick check from `TESTING.md` and confirm it still passes.
5. **Never commit** `.env`, clips, model weights, or outputs.

## 3. Architecture (layers)
```
Browser <video> plays clip (master clock) ──playhead──▶ Story Engine (Mac)
                                                         │
  FAST LAYER (Mac, every 1 s)                            │
    audio excitement + whistle (MediaPipe YAMNet)        │
    motion intensity + scene-cut/replay hint (OpenCV)    │──▶ intensity meter + triggers
                                                         │
  SPECIALIST LAYER (Colab GPU, via tunnel)               │
    OpenSportsLib ball action spotting (12 classes)      │──▶ precise events (shot, goal, tackle…)
                                                         │
  SMART LAYER (Gemini, every ~6–10 s + on triggers)      │
    last ~6 s clip (frames + audio) + specialist hints   │──▶ story JSON (phase, sentiment, narration…)
    + story-so-far                                       │
                                                         │
  MEMORY LAYER (Gemini text-only, every ~60 s)           │──▶ story so far, score, momentum
                                                         │
  FUSION ──▶ timeline + intensity + score ──▶ dashboard Story tab (WebSocket) + logs on disk
```

Modes (selectable in dashboard and `config.yaml`), used later to prove the hybrid is best:
`fast_only` · `specialist_only` (fast + specialist) · `gemini_only` (fast + Gemini) · `hybrid` (all layers).

Graceful fallback, automatic and shown in the UI:
- Colab not connected → run without the specialist layer.
- Gemini key missing or rate-limited → keep running; narration falls back to simple templates from fast/specialist events.
- No internet → fast layer only.

## 4. Repo additions (keep existing structure)
```
ai-director/
├── story/                     # NEW — all MVP 2 code (Python package)
│   ├── __init__.py
│   ├── intake.py              # upload / YouTube → normalized mp4 + 16 kHz wav
│   ├── causal.py              # CausalReader (video frames + audio up to playhead only)
│   ├── fast_layer.py          # audio (YAMNet via MediaPipe) + whistle + motion + scene cut
│   ├── specialist_client.py   # talks to the Colab server over WebSocket
│   ├── llm/
│   │   ├── base.py            # StoryLLM interface (swappable provider)
│   │   └── gemini.py          # Gemini implementation
│   ├── schema.py              # Pydantic models: StoryUpdate, Event, Memory
│   ├── fusion.py              # merge layers, dedupe events, intensity, score
│   ├── engine.py              # StoryEngine: orchestrates layers per playhead tick
│   ├── evaluate.py            # proof mode: compare with ground truth, write report
│   └── soccernet.py           # optional: import SoccerNet Labels-v2.json as ground truth
├── colab/
│   └── specialist_server.ipynb  # self-contained notebook (server code inside the notebook)
├── dashboard/                 # add a "Story" tab; MVP 1 tab unchanged
├── data/clips/                # user clips (gitignored)
├── outputs/story/<clip>/      # logs per run (gitignored)
├── tests/test_story_*.py      # causal guard, schema, dedupe, fusion
├── docs/MVP2_STORY_SPEC.md    # this file
└── config.yaml                # add a `story:` section
```

## 5. Story data model (`story/schema.py`, Pydantic)
```python
Phase = Literal["build_up","attack","chance","goal","celebration","stoppage","foul",
                "set_piece","injury","replay","break","unknown"]
Sentiment = Literal["calm","building","tense","excited","joy","frustration",
                    "anger","disappointment","relief","unknown"]
EventType = Literal["goal","shot","save","chance","foul","card_yellow","card_red","corner",
                    "free_kick","penalty","throw_in","offside","substitution","tackle","cross",
                    "header","out","kickoff","whistle","injury","celebration","argument","none"]

class Event(BaseModel):
    type: EventType
    event_time: float          # seconds in clip
    logged_at: float           # playhead when produced
    source: Literal["fast","specialist","gemini","fusion","manual"]
    team: str | None           # jersey colour, e.g. "white"
    confidence: float          # 0..1
    note: str | None

class StoryUpdate(BaseModel):
    window_start: float; window_end: float; logged_at: float
    phase: Phase
    events: list[Event]
    team_in_control: str | None
    intensity: int             # 0..100
    sentiment: Sentiment
    narration: str             # max ~25 words, present tense, like a commentator
    is_replay: bool            # broadcast replays must NOT create new events (no double goals)
    score: str | None          # e.g. "1-0" if known

class Memory(BaseModel):
    logged_at: float
    story_so_far: str          # 2–4 sentences
    score: str | None
    momentum: str              # e.g. "white pressing for 5 minutes"
    teams: dict[str, str]      # colour → short description
```
Gemini must return JSON matching `StoryUpdate` (use the SDK's structured output with a response schema).

## 6. Config (`config.yaml` → `story:` section, all tunable)
```yaml
story:
  mode: hybrid                   # fast_only | specialist_only | gemini_only | hybrid
  send_audio_to_llm: true        # set false to test without TV commentary (honest test)
  fast:
    tick_s: 1.0
    audio_classes: [Cheering, Crowd, Applause, Whistle, Chant, Roar, Shout]
    excitement_trigger_jump: 0.25
    whistle_min_score: 0.35
  specialist:
    enabled: true
    server_url: ""               # pasted from Colab (ngrok / cloudflared URL), also editable in UI
    send_fps: null               # null = read from model config returned by /health
    frame_height: 224
    stride_s: 2.0
    min_confidence: 0.3
    dedupe_window_s: 2.0
  llm:
    provider: gemini
    model: ""                    # empty = auto-pick newest Flash model available to the key (Step 0)
    window_s: 6.0
    interval_s: 8.0              # normal cadence
    trigger_min_gap_s: 3.0       # extra calls on triggers, but never closer than this
    max_requests_per_minute: 8   # stay under the free tier; adjust after checking AI Studio limits
    video_height: 360
  memory:
    interval_s: 60
  fusion:
    intensity_weights: {audio: 0.4, motion: 0.2, specialist: 0.2, llm: 0.2}
  eval:
    match_window_s: 10
```

## 7. Build steps
Work through these in order. After each step: run its check, fix until it passes, update `PROGRESS.md`
(add an "MVP 2" section), commit `MVP2 step N: <name>`, and explain to the user in 2–3 simple sentences
what was built and how to check it (the user is a beginner in Python and AI).

### Step 0 — Plan and environment
- Read this spec, the existing code and `CLAUDE.md`. Update `CLAUDE.md`: mark MVP 1 as done (keep its rules),
  add an "MVP 2" section pointing to this spec, and keep the working-mode rules.
- Add to `requirements.txt`: `google-genai`, `mediapipe`, `pydantic`, `websockets`, `python-multipart` (plus anything needed).
- `.env`: add `GEMINI_API_KEY=` placeholder. If it's empty, **stop and ask the user** (key from aistudio.google.com → Get API key).
- With the key: list available models (`client.models.list()`), choose the newest Flash model that works with this key,
  record it in `config.yaml` and `PROGRESS.md`. Make one tiny test call. Log the model name and latency.
- Check: imports work, Gemini test call succeeds, MVP 1 quick check still passes.

### Step 1 — Clip intake + live player
- `story/intake.py`: accept an uploaded file or a YouTube URL (reuse the yt-dlp code from MVP 1).
  Normalize with ffmpeg to `data/clips/<name>/clip.mp4` (H.264 + AAC, max 720p, 25 fps) and `audio.wav` (16 kHz mono).
- Dashboard: new **Story** tab with drag-and-drop upload, URL box, clip list, and an HTML5 `<video>` player **with sound**.
- Browser sends playhead (`currentTime`, paused/playing) over WebSocket ~5×/s.
- `story/causal.py`: `CausalReader` for frames and audio, with the guard and unit test (rule 1).
- Check: upload a clip, press Go Live, the video plays with sound, and the backend logs playhead ticks.

### Step 2 — Fast layer
- Audio: MediaPipe Audio Classifier with the YAMNet `.tflite` model (download it in setup, gitignore it).
  Every `tick_s`: classify the last ~1 s of audio, and compute excitement (0..1) from the configured classes + loudness.
  Whistle: YAMNet "Whistle" score **plus** a simple tonal band check (~2–4.5 kHz), because whistles are often
  confused with cheering.
  If MediaPipe fails on this Mac, fall back to loudness + spectral-band features and note it in PROGRESS.md.
- Motion: low-res frame difference / optical-flow magnitude → motion (0..1).
- Scene cut + replay hint: detect hard cuts (histogram difference); several cuts close together or a cut into
  slow motion → `replay_hint`.
- Triggers: excitement jump, whistle, (later) specialist SHOT/GOAL/FREE_KICK.
- Check: run a clip; the intensity meter moves with crowd noise; whistles and cheers show in the log at sensible times.

### Step 3 — Story engine + logs
- `story/engine.py`: on each playhead tick, call the layers that are due, collect results, hand them to fusion,
  and push updates to the dashboard over WebSocket.
- Write logs to `outputs/story/<clip>/<run_id>/`: `story.jsonl`, `events.jsonl`, `intensity.csv`,
  `llm_calls.jsonl` (request time, latency, tokens, errors), `specialist_calls.jsonl`, `memory.jsonl`, `run_meta.json`
  (mode, model, config snapshot).
- Check: a run in `fast_only` mode produces logs and live dashboard updates.

### Step 4 — Smart layer (Gemini)
- `story/llm/base.py`: `StoryLLM` interface (`describe_window(...) -> StoryUpdate`, `summarize(...) -> Memory`).
  `story/llm/gemini.py` implements it with `google-genai`.
- Window input: cut the last `window_s` seconds (ending exactly at the playhead) to a small mp4 (≤ `video_height`p)
  **with audio** if `send_audio_to_llm` is true, and send it inline. If inline video doesn't work well,
  fall back to ~4–6 JPEG frames + a WAV snippet.
- Prompt includes: sport, team colours known so far, the latest Memory, recent events (last ~60 s), and specialist
  hints (if any). Instruct the model: describe only what is visible or audible, mark replays as replays,
  never invent scores, keep narration under 25 words, and answer only with the JSON schema.
- Cadence: every `interval_s`, plus extra calls on triggers (respect `trigger_min_gap_s` and `max_requests_per_minute`).
  On HTTP 429 → exponential backoff, show "rate limited" in the UI, and keep the clip playing.
- Check: on a 3–5 minute clip, narration appears every ~8 s, reads sensibly, and latency is shown per update.

### Step 5 — Memory layer
- Every `memory.interval_s`, a text-only call: story so far, score, momentum, team colours.
- The dashboard shows the "Story so far" panel and score.
- Check: after 3+ minutes, the summary matches what happened.

### Step 6 — Specialist layer (Colab + tunnel)
- `colab/specialist_server.ipynb` (self-contained; the user uploads it to Colab):
  1. Install `opensportslib`, `fastapi`, `uvicorn`, `pyngrok`, `nest_asyncio`; check the GPU (`nvidia-smi`).
  2. Load the ball action spotting model `OpenSportsLab/OSL-loc-snbas-2025-e2e` (12 classes: PASS, DRIVE, HEADER,
     HIGH PASS, OUT, CROSS, THROW IN, SHOT, BALL PLAYER BLOCK, PLAYER SUCCESSFUL TACKLE, FREE KICK, GOAL).
     Read the model card and the opensportslib docs for the exact API, input size, fps and clip length.
     Warm it up and run a self-test on a short sample.
  3. FastAPI server: `GET /health` (model, classes, expected fps, frame size, window length, GPU name),
     `WS /stream` receives JPEG frames with timestamps, keeps a ring buffer, and every `stride_s` runs inference on the
     latest window (past frames only) and returns events `{class, event_time, confidence, infer_ms}`.
  4. Tunnel: pyngrok using `NGROK_AUTHTOKEN` from Colab Secrets. Fallback with no account: a cloudflared quick tunnel.
     Print the public URL big and clear, and test `/health` through it.
  5. A markdown cell at the top: "How to run" in 5 simple steps, for a beginner.
- `story/specialist_client.py`: sends frames at the model's fps and size, receives events, maps class names to
  `EventType` (e.g. SHOT→shot, GOAL→goal, FREE KICK→free_kick, THROW IN→throw_in, OUT→out,
  PLAYER SUCCESSFUL TACKLE→tackle, CROSS→cross, HEADER→header; PASS/DRIVE/HIGH PASS/BLOCK are kept as
  low-priority "play" signals, not shown as headline events), dedupes within `dedupe_window_s`, and reports the
  real-time factor (inference time ÷ stride).
- Dashboard: a "Colab URL" box + connection light + real-time factor.
- **Stop and give the user exact steps** to run the notebook and paste the URL. Continue when connected.
- If opensportslib can't run on Colab after 3 serious attempts: document why in PROGRESS.md, and implement the same
  server with the closest working alternative from the OpenSportsLab model zoo or the SoccerNet repos.
  Keep the same API so nothing else changes.
- Check: on a broadcast clip with a shot or goal, the specialist event appears within a few seconds, and the
  real-time factor is < 1.

### Step 7 — Fusion, score and modes
- `story/fusion.py`: merge events from all layers (same type within the dedupe window → keep one, combine confidence,
  keep every source), ignore new events while `is_replay` is true, compute fused intensity with the configured weights,
  and track the score (goal = specialist GOAL and/or Gemini goal with confidence ≥ threshold, excluding replays).
  Manual buttons "+1 / −1 per team" override the score.
- Triggers from the specialist (SHOT/GOAL/FREE KICK) force a Gemini call (respecting the rate limits).
- Mode selector works for all 4 modes, and the mode is stored in `run_meta.json`.
- Check: the same clip in `hybrid` vs `gemini_only` shows different, logged behaviour; no double goal on replays.

### Step 8 — Dashboard Story tab (final UI)
Layout:
- Left (large): video player with sound; **Go Live / Pause / Stop**; mode selector; "send audio to AI" toggle.
- Top right: **NOW card** with phase, sentiment (colour + emoji), intensity bar, team in control, and score.
- Middle right: **live commentary log** (newest at top, each line shows clip time + delay, with 👍/👎 rating buttons).
- Bottom: **event timeline** with icons (⚽ goal, 🎯 shot, 🧤 save, ✋ foul, 🟨/🟥 card, 🚩 corner, 🔁 replay)
  and an **intensity graph** over time with a playhead line.
- Side panel: **Story so far**, momentum, team colours.
- Status bar: Gemini ●, Colab ●, Audio ● lights; requests/minute used; average delay; real-time factor.
- Keep the existing stack (FastAPI + static HTML + Tailwind CDN + vanilla JS; Chart.js from a CDN is fine).
- Check: a full clip plays with everything updating live and nothing freezing; it also works with Colab off (fallback).

### Step 9 — Proof mode (evaluation)
- Ground truth: `data/clips/<name>/truth.csv` with columns `time_s,event,team` (written by the user in ~15 min),
  or imported from SoccerNet `Labels-v2.json` via `story/soccernet.py`.
- `python -m story.evaluate --clip <name> --run <run_id>` → `outputs/story/<clip>/<run_id>/report.md`:
  per event type, precision / recall within `match_window_s`; average and max delay; narration accuracy from
  👍/👎 ratings; intensity peaks vs truth events.
- Dashboard: in proof mode, truth markers on the timeline with ✅ matched / ❌ missed / ⚠️ false alarm.
- Add `compare.md`: a small table across modes for the same clip (fill it from the reports).
- Check: the report generates for at least one clip with truth data.

### Step 10 — Docs
- `TESTING_MVP2.md`: beginner steps to run the demo, the Colab connection, and how to read the report.
- `README.md`: a short MVP 2 section.
- `DEMO.md`: a 3-minute demo script (which clip, what to point at).

## 8. Definition of done
- Drop any soccer clip → Go Live → story, mood, events, intensity and score update live while the video plays with sound.
- Works in all 4 modes; falls back gracefully when Colab or Gemini is unavailable.
- Causal guard test passes. No future data is ever read.
- One evaluation report exists, with targets for a broadcast clip with truth data:
  goals ≥ 90% caught, other headline events ≥ 60%, average delay ≤ 10 s, narration 👍 ≥ 80%.
- MVP 1 still passes its quick check.

## 9. Out of scope (do not build)
Camera switching changes, phone or live network streams, spoken commentary (TTS), training or fine-tuning models,
the UniSoccer layer, authentication, deployment.

## 10. Notes for test clips
- Best first clip: 5–10 minute extended highlights or a full half with goals (broadcast, with crowd sound).
- Honest test: run the same clip with `send_audio_to_llm: false`, so the AI can't simply repeat the TV commentator.
- Amateur-style test: the static panoramic clip from MVP 1 (no commentary).
- Broadcast clips include replays: they must be marked `is_replay` and must not create duplicate events.
