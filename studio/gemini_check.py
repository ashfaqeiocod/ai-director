"""gemini_check.py — before a long live run: which Gemini model works on which key, right now?

Run:  venv/bin/python -m studio.gemini_check

Sends every model in the studio's chain (config.yaml -> studio.llm: model, fallback_model, extra_models) one REAL
story request on every key: a 6-second match clip + a JSON answer, like the live run sends. Prints OK + seconds, or
the error (503 = Google's model is overloaded right now, 429 = quota used up, 404 = this key can't use it).
Keys are never printed, only their number (0 = GEMINI_API_KEY, 1 = GEMINI_API_KEY1 ...).

Each check uses 1 request of that model's daily quota per key (Lite models: 500/day each, Flash: 20/day).
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.common import ROOT, ffmpeg_exe, load_config

SCHEMA = {"type": "object", "properties": {"narration": {"type": "string"}, "events": {"type": "array", "items": {"type": "string"}}},
          "required": ["narration", "events"]}
PROMPT = "Football clip. In one sentence, what happens? List the events (pass, shot, tackle...)."


def chain() -> list[str]:
    llm = {**load_config()["story"]["llm"], **load_config()["studio"].get("llm", {})}
    return list(dict.fromkeys(m for m in [llm["model"], llm.get("fallback_model"), *(llm.get("extra_models") or [])] if m))


def test_clip() -> bytes:
    """6 s of a prepared match (the ISSIA or Alfheim story mosaic), 960 px wide, no sound."""
    sources = [*sorted((ROOT / "data" / "alfheim").glob("*_H[12]/story_mosaic.mp4")),
               *sorted((ROOT / "data" / "issia").glob("T*/story_mosaic.mp4"))]
    if not sources:
        sys.exit("No story_mosaic.mp4 found: prepare ISSIA or Alfheim first")
    out = Path(tempfile.mkstemp(suffix=".mp4")[1])
    subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-ss", "60", "-t", "6", "-i", str(sources[0]),
                    "-vf", "scale=960:-2", "-an", str(out)], check=True)
    data = out.read_bytes()
    out.unlink()
    return data


def main() -> int:
    from dotenv import load_dotenv
    from google import genai
    from google.genai import types

    from story.common import gemini_keys

    load_dotenv(ROOT / ".env")
    keys = gemini_keys()
    if not keys:
        sys.exit("No GEMINI_API_KEY in .env")
    video = test_clip()
    models = chain()
    print(f"{len(keys)} keys, {len(models)} models (in order of use): {', '.join(models)}\n")

    def probe(job: tuple[str, int]) -> tuple[str, int, str, bool]:
        model, k = job
        client = genai.Client(api_key=keys[k], http_options=types.HttpOptions(timeout=60000))
        start = time.time()
        try:
            r = client.models.generate_content(
                model=model, contents=[types.Part(inline_data=types.Blob(data=video, mime_type="video/mp4"),
                                                  video_metadata=types.VideoMetadata(fps=2)), PROMPT],
                config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=SCHEMA))
            return model, k, f"OK   {time.time() - start:5.1f}s  {json.loads(r.text)['narration'][:60]}", True
        except Exception as error:  # noqa: BLE001 — report anything (quota, overload, network)
            text = str(error).replace(keys[k], "***").split(". {")[0]
            return model, k, f"FAIL {time.time() - start:5.1f}s  {text[:80]}", False

    with ThreadPoolExecutor(6) as pool:
        results = list(pool.map(probe, [(m, k) for m in models for k in range(len(keys))]))
    for model, k, text, _ in results:
        print(f"  {model:32s} key {k}  {text}")
    working = sorted({m for m, _, _, ok in results if ok}, key=models.index)
    print(f"\nWorking now: {', '.join(working) or 'none'}")
    if models[0] not in working:
        print(f"WARNING: the first model ({models[0]}) failed: calls start on the next working model")
    return 0 if working else 1


if __name__ == "__main__":
    sys.exit(main())
