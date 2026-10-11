"""main.py — MVP 3 in one command: every step for one segment, skipping what is already done.

  venv/bin/python -m studio.main --segment dev
  venv/bin/python -m studio.main --segment eval --story-mode gemini_only

Steps (each one is also its own command, see docs/MVP3_SPEC.md):
  1 prepare      cut the segment from the half, check the annotations line up        (studio.prepare)
  2 cameras      5 virtual cameras; the close-up aimed by the oracle                  (studio.cameras)
  3 detect       MVP 1 detector on cams 1-3 (slow on the M1: ~1 h for 14 min)         (studio.detect)
    cameras      the close-up again, aimed by the detector                           (studio.cameras --only 5)
  4 story        MVP 2 story engine on the panorama at REAL speed (takes the clip's length)  (studio.story_pass)
  6 run          direct it twice: detector and oracle                                 (studio.run)
  8 evaluate     report.md for both runs + compare.md                                 (studio.evaluate)
"""
from __future__ import annotations

import argparse
import json

from studio import cameras, detect, evaluate, prepare, run, story_pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--segment", required=True)
    parser.add_argument("--story-mode", default="gemini_only",
                        choices=["fast_only", "specialist_only", "gemini_only", "hybrid"])
    parser.add_argument("--specialist-url", default="")
    parser.add_argument("--skip-detector", action="store_true", help="oracle run only (no 1 h detection)")
    args = parser.parse_args()
    seg = args.segment

    prepare.prepare(seg)
    cams = cameras.cams_dir(seg)
    if not all(cameras.cam_video(seg, c).exists() for c in cameras.FIXED_CAMS) \
            or not cameras.cam_video(seg, cameras.CLOSE_CAM, "oracle").exists():
        cameras.render(seg, aim="oracle")
    if not args.skip_detector:
        detect.run(seg)
        if not (cams / "cam5_detector_rects.npy").exists():
            cameras.render(seg, only=[cameras.CLOSE_CAM], aim="detector")

    runs_file = prepare.work_dir(seg) / "story_runs.json"
    runs = json.loads(runs_file.read_text()) if runs_file.exists() else []
    story = [r for r in runs if r["mode"] == args.story_mode and r["speed"] == 1.0]
    story_id = story[-1]["run_id"] if story else story_pass.run(seg, args.story_mode, 1.0, args.specialist_url)

    for oracle in ([True] if args.skip_detector else [False, True]):
        out = run.run(seg, oracle, story_id, no_story=False, max_seconds=None)
        evaluate.evaluate(seg, out.name)
    evaluate.compare(seg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
