"""check_gemini.py — Step 0 of MVP 2: is the Gemini key OK, and which model do we use?

Run:  venv/bin/python -m story.check_gemini

1. Reads GEMINI_API_KEY from .env.
2. Lists the models this key can use and picks the newest "Flash" model (fast + cheap).
3. Makes one tiny test call and prints the latency.
4. Writes the chosen model name into config.yaml (story.llm.model) if it was empty.
"""
from __future__ import annotations

import re
import sys
import time

from story.common import ROOT, gemini_key, story_config

# Model names we never want for story reading (special-purpose variants).
SKIP_WORDS = ("lite", "image", "tts", "audio", "live", "embedding", "thinking", "8b", "exp", "robotics", "computer")


def flash_candidates(names: list[str]) -> list[str]:
    """Flash models, newest version first; a stable model beats a preview of the same version."""
    found = []
    for name in names:
        short = name.removeprefix("models/")
        m = re.fullmatch(r"gemini-(\d+(?:\.\d+)?)-flash(-preview[\w.-]*)?", short)
        if not m or any(w in short for w in SKIP_WORDS):
            continue
        found.append((float(m.group(1)), m.group(2) is None, short))
    return [s for _, _, s in sorted(found, reverse=True)]


def test_call(client, model: str) -> float:
    """One tiny request. Returns the latency in seconds (raises on any error)."""
    start = time.time()
    reply = client.models.generate_content(model=model, contents="Reply with the single word: ready")
    if not (reply.text or "").strip():
        raise RuntimeError("empty reply")
    return time.time() - start


def save_model_in_config(model: str) -> None:
    """Write `model: "<name>"` under story.llm in config.yaml, keeping every comment."""
    file = ROOT / "config.yaml"
    text = file.read_text()
    llm_block = text.index("  llm:\n    provider:")
    start = text.index('    model: ""', llm_block)
    file.write_text(text[:start] + f'    model: "{model}"' + text[start + len('    model: ""'):])


def main() -> int:
    key = gemini_key()
    if not key:
        print("GEMINI_API_KEY is missing in .env.\n"
              "  Get one at https://aistudio.google.com -> 'Get API key', then add this line to .env:\n"
              "  GEMINI_API_KEY=your-key")
        return 1
    from google import genai

    client = genai.Client(api_key=key)
    try:
        names = [m.name for m in client.models.list()
                 if "generateContent" in (m.supported_actions or [])]
    except Exception as error:
        print(f"Could not list models (is the key right?): {error}")
        return 1
    candidates = flash_candidates(names)
    print(f"{len(names)} models available; Flash candidates (newest first): {candidates}")

    configured = story_config()["llm"].get("model") or ""
    for model in ([configured] if configured else []) + candidates:
        try:
            latency = test_call(client, model)
        except Exception as error:
            print(f"  {model}: FAILED ({str(error)[:160]})")
            continue
        print(f"  {model}: OK, test call took {latency:.2f} s")
        if not configured:
            save_model_in_config(model)
            print(f"Saved story.llm.model = {model} in config.yaml")
        return 0
    print("No Flash model worked with this key.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
