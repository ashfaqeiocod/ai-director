"""latency.py — how late every AI answer is in a live run (MVP 3 live test, step 3).

Each AI item has two times (both in match seconds):
  happened  when the thing happened on the pitch (an event's event_time, the end of the window the AI watched,
            a detection frame's time)
  known     when the live system KNEW it (logged_at: the live clock when the answer arrived)
delay = known - happened. A long delay = a slow AI; the dashboard draws each item as a bar happened -> known.

Sources:
  gemini    story looks (window end -> logged_at; model, key, request seconds from Gemini's call log) and the
            events Gemini reported (event_time -> logged_at)
  goal_ai   the goal watcher's looks at a goal camera (window end -> logged_at)
  colab     the Colab action model's events (event_time -> logged_at)
  detector  the football detector on Colab (frame time -> applied). 5 frames a second x 47 min = 14,000 items, so
            they are summed up per second (max / average delay, count, late ones); late frames are listed one by one.
Director cuts and map triggers have no delay (they happen in the loop); the page lists them from their own log.

Everything goes to latency.jsonl in the run folder and to the page as {"type": "latency", ...} messages.
Several threads report here (story workers, the goal watcher, the live loop), so one lock guards the file.
"""
from __future__ import annotations

import json
import threading
from collections.abc import Callable
from pathlib import Path

import numpy as np

SOURCES = ("gemini", "goal_ai", "colab", "detector")


class LatencyLog:
    def __init__(self, out: Path | None, publish: Callable[[dict], None]):
        self.file = open(out / "latency.jsonl", "w") if out else None
        self.publish = publish
        self.lock = threading.Lock()
        self.delays: dict[str, list[float]] = {s: [] for s in SOURCES}
        self.llm_calls: dict[float, dict] = {}       # Gemini story calls by window end (model, key, seconds)
        self.det_second: dict | None = None          # the detector's current one-second summary

    # ------------------------------------------------------------------ one item
    def add(self, source: str, happened: float, known: float, what: str, count: bool = True, **extra) -> dict:
        """count=False: shown, but not in the delay statistics (the detector counts every frame itself)."""
        item = {"source": source, "happened": round(happened, 2), "known": round(known, 2),
                "delay": round(known - happened, 2), "what": what, **extra}
        with self.lock:
            if count:
                self.delays[source].append(item["delay"])
            if self.file:
                self.file.write(json.dumps(item) + "\n")
                self.file.flush()
        self.publish({"type": "latency", **item})
        return item

    # ------------------------------------------------------------------ Gemini
    def note_llm_call(self, call: dict) -> None:
        """Gemini's call log (every request): remember model, key and request time per story window."""
        if call.get("kind") == "story" and not call.get("error") and call.get("window"):
            with self.lock:
                self.llm_calls[round(float(call["window"][1]), 2)] = call

    def story(self, update: dict) -> None:
        if update.get("source") != "gemini":
            return                                   # template lines (no Gemini) have no AI delay
        with self.lock:
            call = self.llm_calls.pop(round(float(update["window_end"]), 2), {})
        self.add("gemini", update["window_end"], update["logged_at"], f"story: {update.get('narration', '')}",
                 kind="look", model=call.get("model"), key=call.get("key"), request_s=call.get("latency_s"),
                 window_start=update.get("window_start"))

    def event(self, message: dict) -> None:
        """A NEW story event (Gemini and/or the Colab action model). Updates of an old event are not new news."""
        if not message.get("new") or message.get("rejected"):
            return
        sources = message.get("sources") or []
        kind = message.get("event_type") or message.get("type")
        if kind in (None, "none", "whistle"):
            return
        source = "colab" if sources == ["specialist"] else "gemini" if "gemini" in sources else None
        if source:
            self.add(source, message["event_time"], message["logged_at"], f"event: {kind.replace('_', ' ')}",
                     kind="event", event=kind, confidence=message.get("confidence"), sources=sources)

    # ------------------------------------------------------------------ goal watcher
    def goal_look(self, look: dict, model: str | None) -> None:
        verdict = "goal" if look.get("goal") else "save" if look.get("save") else "celebration" \
            if look.get("celebration") else "shot" if look.get("shot") else "no goal"
        self.add("goal_ai", look["t"], look["logged_at"], f"{look.get('side')} goal: {verdict}",
                 kind=look.get("kind"), model=model, verdict=verdict, trigger=look.get("trigger"))

    # ------------------------------------------------------------------ detector (summed up per second)
    def detection(self, frame_time: float, applied_at: float, source: str, rtt_ms: float | None) -> None:
        second = int(frame_time)
        with self.lock:
            cur = self.det_second
            if cur is not None and cur["second"] != second:
                self.det_second = None
                done = cur
            else:
                done = None
            if self.det_second is None:
                self.det_second = {"second": second, "delays": [], "rtts": [], "late": 0, "frames": 0}
            d = self.det_second
            d["frames"] += 1
            if source == "colab":
                d["delays"].append(applied_at - frame_time)
                self.delays["detector"].append(round(applied_at - frame_time, 3))
                if rtt_ms is not None:
                    d["rtts"].append(rtt_ms)
            else:
                d["late"] += 1
        if done is not None:
            self._flush_second(done)
        if source != "colab":                       # a late / missing answer: listed one by one
            self.add("detector", frame_time, applied_at, f"frame {frame_time:.1f}s: {source.replace('_', ' ')}",
                     count=False, kind="late", used=source)

    def _flush_second(self, d: dict) -> None:
        if not d["delays"]:
            return
        worst = max(d["delays"])
        self.add("detector", d["second"], d["second"] + worst, f"{len(d['delays'])}/{d['frames']} frames from the GPU",
                 count=False, kind="second", avg=round(float(np.mean(d["delays"])), 3), frames=d["frames"], late=d["late"],
                 rtt_ms=round(float(np.mean(d["rtts"])), 1) if d["rtts"] else None)

    # ------------------------------------------------------------------ end of the run
    def close(self) -> dict:
        with self.lock:
            last = self.det_second
            self.det_second = None
        if last is not None:
            self._flush_second(last)
        with self.lock:
            if self.file:
                self.file.close()
                self.file = None
            return summarize(self.delays)


def summarize(delays: dict[str, list[float]]) -> dict:
    """Average / 90% / max delay per source (seconds)."""
    out = {}
    for source, values in delays.items():
        if values:
            v = np.asarray(values)
            out[source] = {"n": len(v), "avg_s": round(float(v.mean()), 2), "p90_s": round(float(np.percentile(v, 90)), 2),
                           "max_s": round(float(v.max()), 2)}
    return out


def derive(updates: list[dict], events: list[dict], looks: list[dict]) -> list[dict]:
    """Latency items for runs recorded before latency.jsonl existed (from their story / goal-watch logs).
    No model names and no detector items there; the bars are the same."""
    out = []
    for u in updates:
        if u.get("source") == "gemini":
            out.append({"source": "gemini", "happened": u["window_end"], "known": u["logged_at"],
                        "delay": round(u["logged_at"] - u["window_end"], 2), "what": f"story: {u.get('narration', '')}",
                        "kind": "look"})
    for e in events:
        sources = e.get("sources") or []
        source = "colab" if sources == ["specialist"] else "gemini" if "gemini" in sources else None
        if source and e.get("type") not in ("none", "whistle"):
            out.append({"source": source, "happened": e["event_time"], "known": e["known_at"],
                        "delay": round(e["known_at"] - e["event_time"], 2), "what": f"event: {e['type'].replace('_', ' ')}",
                        "kind": "event"})
    for x in looks:
        verdict = "goal" if x.get("goal") else "save" if x.get("save") else "shot" if x.get("shot") else "no goal"
        out.append({"source": "goal_ai", "happened": x["t"], "known": x["logged_at"],
                    "delay": round(x["logged_at"] - x["t"], 2), "what": f"{x.get('side')} goal: {verdict}",
                    "kind": x.get("kind")})
    return sorted(out, key=lambda i: i["known"])
