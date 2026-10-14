# Testing the Arm MVP (AI director on the 5-phone arm-wrestling footage)

Everything runs on the Mac. The videos stay in `data/arm/` (never committed: real colleagues).

## 1. One command (about 25 s with the caches, ~3 min with the videos)
```bash
venv/bin/python -m arm.main --no-render   # decisions + report only (~25 s)
venv/bin/python -m arm.main               # + all videos (~3 min)
```
What you should see: `[sync] cached`, `[perception] cached`, `[camera setup] cached`, `[judge] cached`, then a table:
`AI live 16 shots, 0 180-degree breaks, moments 12/14`, the editor, and the baselines (one phone A..E, random,
round robin, motion). Re-running sends **0** Gemini requests (everything is cached).

First time on a new machine (no caches) it also runs:
| step | command | time |
|---|---|---|
| sync | `venv/bin/python -m arm.sync` | ~25 min on a busy M1 (re-encodes 5 videos + the 4K copy) |
| perception | `venv/bin/python -m arm.perceive` | ~5 min |
| camera setup | `venv/bin/python -m arm.setup` | 5 Gemini calls, ~30 s |
| judge | `venv/bin/python -m arm.judge` | 16 Gemini calls, ~2-5 min on the free tier |

## 2. Watch the results
```bash
open data/arm/out/program_live.mp4     # the AI live broadcast (it only saw the past), with graphics
open data/arm/out/program_edit.mp4     # the AI editor's version (it saw the whole match)
open data/arm/synced/sync_check.mp4    # the 5 synced phones + their sound waves (spikes line up = synced)
open data/arm/out/report.md            # the scorecard + Gemini's notes
open data/arm/out/proof_behind_the_ai.mp4  # "behind the AI's eyes": broadcast + what the AI sees on 5 phones + why it cut
                                           # (made by: venv/bin/python -m arm.proof, ~3 min)
open data/arm/out/director_log_live.csv
```
In `program_live.mp4` look for: tan smiling as he lifts his head (~16.5 s, close-up from C), the crowd laughing
(~21 s, E), black driving tan's hand down (23-26 s, grip from behind), and the end: "BLACK WINS!" with tan's hand
on the table (31.3 s). The lower third shows the phase, a tension meter and the moment.

## 3. The dashboard
```bash
venv/bin/python -m dashboard.server        # then open http://localhost:8000/arm
```
(If port 8000 is busy, another session's dashboard is running: use that one, the Arm pages are in it too.)
- `/arm` — the program (switch AI live / AI editor), the 5 synced phones with the on-air one in red, timelines
  (shots per camera, phase, tension + crowd, moments, Gemini), cut log with reasons, scorecard.
- `/arm/label` — **please label the match**: play it, pick "who" and press G (go), S (surge), N (near pin),
  P (pin + winner), R (reaction), H (great moment), E (end) at the right moments; Save. Then
  `venv/bin/python -m arm.main --no-render` scores the AI against YOUR labels (an independent test).
- `/arm/vote` — the blind test: two versions side by side without names; pick the better one. Ask colleagues.
  The results table fills in as votes arrive (`data/arm/votes.jsonl`).

## 4. Tests
```bash
venv/bin/python -m pytest tests/test_arm.py -q    # 8 tests: sync, 180-degree rule, jump cuts, min shot, editor, Gemini cap
```

## 5. Things to know
- Gemini budget: hard cap 200 requests in total for this footage (`data/arm/cache/gemini_calls.json`); 31 used.
- Gemini on the free tier answered in 3-27 s per still: fine for the editor, mostly too late for live.
- Until you label the match, "moments caught" uses the AI's own detected moments (the report says so).
