# Mode comparison — eng-cze (England 3-0 Czechia highlights, 3:22)

Same clip, same code (2026-10-09), Gemini `gemini-3.5-flash-lite` (free tier), Colab specialist on a Tesla T4,
proof mode against `data/clips/eng-cze/truth.csv` (13 events, drafted by Claude from the video — please verify).
Match window ±10 s. Filled from `outputs/story/eng-cze/<run>/report.json` (`python -m story.evaluate --clip eng-cze --compare`).

| Mode | Audio to AI | Goals caught | Other headline events | Avg delay | Final score | Run |
|---|---|---|---|---|---|---|
| fast_only | – | 0% (0/3) | 0% | – | – | 20261009-115208-fast_only |
| specialist_only (fast + Colab) | – | 0% (0/3) | 30% | 3.5 s | – | 20261009-114828-specialist_only |
| gemini_only (fast + Gemini) | yes | 100% (3/3) | 50% | 5.9 s | white 3 – red 0 | 20261009-114449-gemini_only |
| **hybrid (all layers)** | yes | **100% (3/3)** | **80%** | **3.8 s** | **white 3 – red 0** | 20261009-114058-hybrid |
| hybrid, honest test | **no** | 67% (2/3) | 60% | 4.4 s | white 2 – red 0 | 20261009-115557-hybrid |

Narration 👍 is not filled in: it needs a human to rate the commentary lines during a run.

What this shows:
- **Hybrid is best**: Gemini finds the goals and writes the story, the specialist adds shots/saves and an earlier
  first report (lower delay), the fast layer drives the intensity meter (13/13 truth events fall in the run's top-20%
  intensity) and triggers extra Gemini looks on crowd roars.
- The specialist alone never fires GOAL on this clip (it finds the shots), so goals rely on Gemini.
- Without sound (honest test) Gemini still catches 2 of 3 goals from the picture alone.
- Replays: goals shown again in replays were not counted twice in any run (scoreboard-aware goal merging).
