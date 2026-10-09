# 3-minute demo — Live Story Reader (MVP 2)

**One sentence:** we drop a soccer clip in, press Go Live, and an AI watches it like a live broadcast —
never peeking ahead — and writes the story as it happens: intensity, events, mood, commentary, score.

## Before the demo (10 minutes earlier)

1. `cd ~/Documents/ai-director && venv/bin/python -m dashboard.server`
2. Open http://localhost:8000/story in **one** browser tab, sound on.
3. Optional but impressive: start the Colab notebook (TESTING_MVP2.md §4), paste the URL, **Connect** → green light.
4. Click the clip **eng-cze** (England 3-0 Czechia, 3:22). Mode **hybrid (all)** (or **fast + Gemini** without Colab).
   Tick **proof mode**.
5. Check the free quota has room (aistudio.google.com → Usage): one run uses ~30 Flash Lite requests.

## The script

| Time | Do / point at | Say |
|---|---|---|
| 0:00 | Press **● Go Live** | "The browser's video player is the clock. 5 times a second it tells the backend where it is. The AI is never allowed to read a frame or a sound after that point — there's a guard in the code that throws an error if anything tries." |
| 0:10 | **Intensity graph** + **Audio** light | "The fast layer runs on this Mac every second: crowd noise with Google's YAMNet audio model, a whistle detector, motion and scene cuts." |
| 0:30 | **Commentary** panel, the **delay** on each line | "Every 8 seconds Gemini gets the last 6 seconds of video with sound and writes a line. Each line shows *when* it happened and *how late* the AI was — about 3 to 4 seconds. That's the honest latency." |
| 0:40 | Timeline: 🦶 free kick, 🎯 shot, 🧤 save | "Events land on the timeline. Hover one: you see which layers found it — Gemini, the Colab action-spotting model, or both — and the confidence." |
| 1:00 | **Status bar: Colab ● RTF 0.13** | "This model runs on a free Colab GPU. RTF 0.13 means it needs 0.13 s per second of video — comfortably live." |
| 1:05 | **⚽ at 1:01**, the **score** changes to white 1 – red 0, CHEER in the signal log | "Goal. The crowd noise spiked, Gemini saw the ball go in, the score updates — about 4 seconds after the real moment." |
| 1:15 | Purple **replay** band, commentary line marked "replay" | "Broadcasts replay goals. Replays must not count twice. The system checks the on-screen score: same score means replay, higher score means a new goal." |
| 1:35 | Second **⚽**, score 2 – 0 | "Second goal, only 33 seconds after the first in this highlight reel — counted correctly because the scoreboard went up." |
| 2:00 | **Story so far** panel | "Once a minute a text-only call writes the story so far and the momentum." |
| 2:28 | Third **⚽** after the 🚩 corner | "Corner, header, 3-0." |
| 2:45 | **Proof mode** markers ✅ under the timeline | "We wrote down what really happened. Green ticks are events the AI caught live." |
| 3:21 | Clip ends, log: `run stopped` | Open `outputs/story/eng-cze/<run>/report.md` (or run `venv/bin/python -m story.evaluate --clip eng-cze --run latest`). "All 3 goals caught, 80% of the other headline events, average delay under 4 seconds." |

## If something goes wrong live

- Gemini light red: free quota used → the commentary continues with simple "template" lines. Say so: "it degrades
  gracefully". Switch `story.llm.model` to `gemini-3.1-flash-lite` and restart for the next run.
- Colab light red: the run continues without it (hybrid falls back automatically).
- Honest-test variant (if asked "is it just repeating the commentator?"): untick **send audio to AI** and run again.
