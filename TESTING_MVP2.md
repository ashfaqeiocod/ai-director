# How to test MVP 2 — Live Story Reader

Run every command from the project folder in your own Terminal:

```bash
cd ~/Documents/ai-director
```

What it does: you play a soccer clip in the dashboard, and the AI watches it **as if it were live**
(it never looks past the point the video has reached). While the clip plays with sound, the Story tab fills
up with an intensity meter, events (goals, shots, saves…), the mood, a commentary line every ~8 seconds,
the score and a "story so far".

---

## 0. One-time checks (2 minutes)

```bash
venv/bin/python -m pytest tests -q          # 37 tests, all should pass ("37 passed")
venv/bin/python -m story.check_gemini       # is the Gemini key OK? (prints the models and a test call)
```

- The key lives in `.env` as `GEMINI_API_KEY=...` (from https://aistudio.google.com → Get API key).
- The model is set in `config.yaml` → `story.llm.model`. With a **free** key use `gemini-3.5-flash-lite`
  (the default: 15 requests/minute, 500/day). The normal Flash models only allow 20 requests **per day** on the
  free tier, and one 3-minute clip needs about 30. With a paid key, `gemini-3.5-flash` is a bit sharper.
  See your limits at https://aistudio.google.com → *Usage / Rate limits*.

## 1. Start the dashboard

```bash
venv/bin/python -m dashboard.server
```

Open **http://localhost:8000/story** (the MVP 1 camera director is still at http://localhost:8000).
Stop the server with `Ctrl+C`.

## 2. Bring in a clip

Any one of these:
- **Drag a video file** onto the "Drop a video here" box, or click it to choose a file.
- **Paste a YouTube link** in the box under it and click **Import** (needs Node.js and internet).
- Command line: `venv/bin/python -m story.intake ~/Movies/match.mp4` (or a YouTube link, add `--name derby`).

It is converted to 720p / 25 fps + a 16 kHz sound file in `data/clips/<name>/` (never committed to git).
A 3-minute YouTube clip takes about a minute. The clip then appears in the list; click it.

The clip used for all results below: **eng-cze** = "England 3-0 Czechia" highlights (3:22,
https://www.youtube.com/watch?v=kJEY-UFTDk8).

## 3. Go Live (fast + Gemini, no Colab needed)

1. Click the clip **eng-cze**. Pick mode **fast + Gemini**. Leave "send audio to AI" on.
2. Click **● Go Live**. The video plays with sound and a red **LIVE** badge.
3. What you should see:
   - **Status bar** (top right): Gemini ● green, Audio ● green, `req/min` around 8/12, `avg delay` 3–5 s, `lag` ~0.0 s.
   - **NOW card**: phase (e.g. "build up", "goal"), mood emoji + colour, intensity bar, team in control, **score**.
     The big score is the TV score graphic as the AI reads it (shown once it was read the same way twice).
     The small line under it is "counted" goals per jersey colour; the AI can mix up team colours, so fix it with
     the ± buttons. A goal shows faded on the timeline until the score graphic confirms it (or is dropped after ~30 s).
   - **Live commentary** (newest at the top): a line every ~8 s, each with the clip time, the **delay**
     (how long after that moment the AI wrote it) and "AI" or "template". Rate lines with 👍 / 👎.
   - **Event timeline**: icons appear (🦶 free kick ~0:33, 🎯 shot + 🧤 save ~0:38–0:50, ⚽ goals at ~1:01, ~1:34, ~2:28),
     purple bands = replays.
   - **Intensity graph**: orange line jumps after every goal; white line = playhead.
   - **Story so far** (after 1:00, then every minute): 2–4 sentences, momentum, team colours.
   - **Signal log** (bottom): CHEER / WHISTLE / CUT / LLM / EVENT / SCORE lines. Use the checkboxes to filter.
4. **Pause** stops the video and the AI. **Go Live** again resumes. **Stop** ends the run. You can't seek back
   during a run (live means live).
5. At the end the log says `run stopped (clip ended)` and where the logs are.

Expected (gemini_only, Flash Lite): ~28 AI lines, average delay ~3.5 s, final score `white 3 – red 0` or close.

## 4. Connect the Colab specialist (ball action spotting on a free GPU)

1. Go to https://colab.research.google.com → **File → Upload notebook** → choose `colab/specialist_server.ipynb`.
2. Menu **Runtime → Change runtime type → T4 GPU → Save**.
3. (Optional) ngrok: put your token in Colab's 🔑 Secrets as `NGROK_AUTHTOKEN`. Without it the notebook uses a
   free Cloudflare tunnel.
4. Menu **Runtime → Run all** (3–5 minutes the first time). A green box shows a URL like
   `https://xxxx.ngrok-free.app`.
5. In the dashboard, paste it into the **Colab specialist** box → **Connect**. The light turns green and shows
   "on Tesla T4".
6. Pick mode **hybrid (all)** → **● Go Live**. Extra lines `SPECIALIST shot @0:37.0` appear, the status bar shows
   **RTF ~0.13** (the GPU needs 0.13 s for every 1 s of video: under 1 = keeps up with live).

If Colab stops (it does after ~90 min idle on the free tier) the light turns red and the run simply continues
without it. Run the notebook again and paste the new URL.

## 5. Proof mode (how good is it?)

1. Tick **proof mode** before Go Live (needs `data/clips/<clip>/truth.csv`; eng-cze has one).
2. Under the timeline, truth markers appear as the clip plays: ✅ matched, ❌ missed, ⚠️ false alarm.
3. After the run, make the report:

```bash
venv/bin/python -m story.evaluate --clip eng-cze --run latest     # or the run id shown in the dashboard
venv/bin/python -m story.evaluate --clip eng-cze --compare         # table across all runs
open outputs/story/eng-cze/*/report.md                             # or open the newest one in your editor
```

How to read `report.md`:
- **Targets**: goals caught ≥ 90%, other headline events ≥ 60%, average delay ≤ 10 s, narration 👍 ≥ 80%.
  "n/a" for narration means nobody rated lines in that run: click 👍/👎 during a run.
- **Per event type**: recall = how many real events were found; precision = how many found events were real.
  A match must be within ±10 s (`story.eval.match_window_s`).
- **Delay**: logged_at − real time. This is the honest "how late was the AI" number.
- **Each truth event**: the ✅ / ❌ list.

**truth.csv** is a simple table you can write yourself in ~15 minutes while watching the clip:

```
time_s,event,team
61.5,goal,white
146.0,corner,white
```

Events use these names: goal, shot, save, chance, foul, card_yellow, card_red, corner, free_kick, penalty,
offside, substitution, injury. Teams = jersey colour. The eng-cze truth file was drafted by Claude from the
video frames and the on-screen score: please check it.
SoccerNet games: `venv/bin/python -m story.soccernet --labels Labels-v2.json --clip <name> --half 1 --start 600 --home white --away red`.

## 6. The honest test (no TV commentary)

Untick **send audio to AI** (or `story.send_audio_to_llm: false`) and run the same clip. Gemini then only sees
the pictures, so it can't simply repeat what the commentator says. Compare the two reports with `--compare`.

## 7. Where the logs are

`outputs/story/<clip>/<run_id>/` (never committed):

| File | What |
|---|---|
| `run_meta.json` | mode, model, settings, summary (events, score, delays) |
| `story.jsonl` | every commentary line with window, `logged_at`, phase, mood, replay flag |
| `events.jsonl` | every event each time it is created or merged (sources, confidence, delay) |
| `intensity.csv` | one row per second: audio, motion, intensity, lag |
| `llm_calls.jsonl` | every Gemini request: latency, tokens, errors |
| `specialist_calls.jsonl` | every Colab answer: inference ms, RTF |
| `memory.jsonl` | every "story so far" |
| `signals.jsonl` | cheers, whistles, cuts, lag, errors |
| `ratings.jsonl` | your 👍 / 👎 |
| `report.md` | the evaluation (after `story.evaluate`) |

## 8. Which videos to test with

The models were trained on (and Gemini works best with) **TV broadcast football**. Best results:

| ✅ Good test clips | Why |
|---|---|
| Official "extended highlights" of a league/international match, 3–10 min (league or federation YouTube channels: Premier League, LaLiga, Bundesliga, UEFA, England, etc.) | wide main camera, crowd sound, a **visible score graphic** (the system uses it to confirm goals and tell replays apart) |
| A full half or a 10–15 min stretch of a broadcast match | real-time flow (no editing cuts), fewer surprises; costs more requests (~10 per minute of video) |
| 720p or 1080p, 25/30/50 fps, landscape | the specialist model expects a normal TV picture |

| ❌ Avoid | Why |
|---|---|
| "Top 10 goals" / compilations from several matches | the score graphic jumps between matches, many goals seconds apart |
| Fan edits with music over the sound, slow-motion montages, skills videos | no match flow; everything looks like a replay |
| Vertical phone videos, TV filmed with a phone, very low quality | the specialist and Gemini can't see the ball |
| Video games (FIFA / eFootball) | not real football |
| Clips under ~1 minute | no story to tell, and the first 10 s are warm-up for the audio baseline |

Tips:
- Short 2-minute highlight reels are the hardest: goals 20–40 s apart, lots of quick cuts. 5–10 minute
  "extended highlights" are a fairer test.
- Write a `truth.csv` (section 5) for every clip you want numbers for, otherwise the report can't score it.
- Free key: ~10 Gemini requests per minute of video, 500 per day for Flash Lite → about 45 minutes of video per day.
- Only keep **one** Story tab running a clip. Other open tabs just watch (they no longer move the AI's clock).
- Disallowed goals (offside, VAR) are a good tough case: the system only counts a goal when the score graphic
  goes up, so a disallowed goal shows as faded on the timeline and is dropped after ~30 s.

## 9. MVP 1 still works

```bash
venv/bin/python src/main.py     # should end with "Cuts: 5" and "Shortest shot: 2.00s"
```

## Troubleshooting

- **Gemini light red, "daily quota used"**: the free limit for that model is reached for today. Switch
  `story.llm.model` to another model (e.g. `gemini-3.1-flash-lite`) or wait until tomorrow. The run keeps going
  with simple "template" lines meanwhile.
- **"rate limited"**: too many requests per minute. Lower `story.llm.max_requests_per_minute` in `config.yaml`.
- **Colab light red**: the notebook stopped or the URL changed. Run it again and paste the new URL.
- **No sound**: click 🔊 next to Stop; some browsers start muted.
- **Seeking doesn't work during a run**: on purpose. Stop, then Go Live starts again from 0:00.
