"""gemini.py — the ONLY door to Gemini in Arm MVP: still pictures in, JSON out, with a hard budget and a cache.

Rules (agreed with the user):
  - Local models do the per-frame work. Gemini is asked rarely, and only with still JPEGs (never video or sound).
  - A hard cap on the TOTAL number of requests ever sent for this footage (config arm.gemini.max_total_calls,
    200). The count is kept on disk (data/arm/cache/gemini_calls.json), so it survives restarts.
  - Every answer is cached by (purpose, picture bytes, prompt): a rerun sends nothing and costs nothing.
  - The API key is never printed or logged (only the key's NUMBER).
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np

from arm.common import cache_dir, settings
from src.common import path
from story.common import gemini_keys


class BudgetExceeded(Exception):
    pass


# Models that answered "quota used up" (429) in this run: not asked again until the next run.
_out_of_quota: set[str] = set()


def _calls_file() -> Path:
    """ONE counter for every recorded match (the 200 cap is for all our footage together), kept with match 1."""
    return path(settings()["gemini"].get("calls_file", "data/arm/cache/gemini_calls.json"))


def calls_used() -> int:
    file = _calls_file()
    return json.loads(file.read_text())["count"] if file.exists() else 0


def _count_call(purpose: str, model: str, seconds: float, ok: bool, error: str = "") -> None:
    file = _calls_file()
    data = json.loads(file.read_text()) if file.exists() else {"count": 0, "log": []}
    data["count"] += 1
    data["log"].append({"purpose": purpose, "model": model, "seconds": round(seconds, 2), "ok": ok, "error": error,
                        "at": time.strftime("%Y-%m-%d %H:%M:%S")})
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(json.dumps(data, indent=1))


def jpeg(image: np.ndarray, width: int) -> bytes:
    """A BGR picture as a JPEG, at most `width` pixels wide (smaller = cheaper and faster)."""
    h, w = image.shape[:2]
    if w > width:
        image = cv2.resize(image, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
    ok, data = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return data.tobytes()


def ask(purpose: str, images: list[bytes], prompt: str, schema, cache_key_extra: str = "") -> tuple[dict, dict]:
    """Ask Gemini about one or more JPEGs. Returns (answer as a dict, info: cached / model / seconds).

    `schema` is a pydantic model: Gemini must answer in exactly that JSON shape.
    """
    cfg = settings()["gemini"]
    digest = hashlib.sha256()
    for part in (purpose, prompt, cache_key_extra, *images):
        digest.update(part if isinstance(part, bytes) else part.encode())
    key = digest.hexdigest()[:24]
    cache = cache_dir() / "gemini" / f"{purpose}_{key}.json"
    if cache.exists():
        saved = json.loads(cache.read_text())
        return saved["answer"], {**saved["info"], "cached": True}
    if calls_used() >= cfg["max_total_calls"]:
        raise BudgetExceeded(f"Gemini budget of {cfg['max_total_calls']} calls is used up")
    keys = gemini_keys()
    if not keys:
        raise RuntimeError("GEMINI_API_KEY is missing in .env")
    from google import genai
    from google.genai import errors, types

    parts = [types.Part.from_bytes(data=img, mime_type="image/jpeg") for img in images] + [types.Part.from_text(text=prompt)]
    config = types.GenerateContentConfig(response_mime_type="application/json", response_schema=schema,
                                         temperature=cfg["temperature"])
    last_error = None
    for model in cfg["models"]:                     # the first model, then fallbacks (overloaded / quota)
        if model in _out_of_quota:
            continue
        for k, api_key in enumerate(keys):
            if calls_used() >= cfg["max_total_calls"]:
                raise BudgetExceeded(f"Gemini budget of {cfg['max_total_calls']} calls is used up")
            client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=cfg["timeout_s"] * 1000))
            started = time.time()
            try:
                response = client.models.generate_content(model=model, contents=[types.Content(role="user", parts=parts)],
                                                          config=config)
                seconds = time.time() - started
                _count_call(purpose, model, seconds, True)
                answer = json.loads(response.text)
                info = {"model": model, "key": k, "seconds": round(seconds, 2), "cached": False}
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps({"answer": answer, "info": info, "prompt": prompt}, indent=1))
                return answer, info
            except errors.APIError as error:
                _count_call(purpose, model, time.time() - started, False, f"HTTP {error.code}")
                last_error = f"{model} key#{k}: HTTP {error.code}"
                if error.code not in (429, 503):    # only quota / overload are worth another key or model
                    raise RuntimeError(last_error) from None
                if error.code == 429 and k == len(keys) - 1:
                    _out_of_quota.add(model)
    raise RuntimeError(f"Gemini unavailable ({last_error})")
