"""Session and latency logs, one JSON object per line.

Every session records device model, OS and user agent, because LRA and ERM motors
feel different and Objective 5 has to be able to separate them. Fields are
whitelisted: nothing the browser sends beyond these is kept, and no IP address is
stored. Session ids are random and chosen by the page.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SESSION_FIELDS = ("session_id", "device_model", "os", "os_version", "user_agent",
                  "browser", "mobile", "vibration_support", "screen")
LATENCY_FIELDS = ("session_id", "slot", "scheme", "predicted_class", "autoplay",
                  "request_ms", "render_ms", "server_ms",
                  "select_to_ready_ms", "ready_to_play_ms", "vibrate_call_ms",
                  "machine_ms", "end_to_end_ms", "vibrate_returned")
MAX_FIELD_CHARS = 512


class SessionLog:
    def __init__(self, directory: str | Path):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _append(self, name: str, fields: tuple, data: dict) -> dict:
        if not data.get("session_id"):
            raise ValueError("session_id is required")
        row = {k: (v[:MAX_FIELD_CHARS] if isinstance(v, str) else v)
               for k, v in data.items() if k in fields}
        row["logged_at"] = datetime.now(timezone.utc).isoformat()
        with open(self.dir / name, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        return row

    def session(self, data: dict) -> dict:
        return self._append("sessions.jsonl", SESSION_FIELDS, data)

    def latency(self, data: dict) -> dict:
        return self._append("latency.jsonl", LATENCY_FIELDS, data)

    def _read(self, name: str) -> list[dict]:
        path = self.dir / name
        if not path.exists():
            return []
        return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]

    def latency_summary(self) -> dict:
        return summarise_latency(self._read("latency.jsonl"))


def summarise_latency(rows: list[dict]) -> dict:
    """Median and p95 per measure.

    `machine_ms` excludes the wait for a person to press play, so it is the number
    to report as system latency. `end_to_end_ms` is only comparable across runs
    when autoplay was on. Both end at the vibrate() call, the last moment a page
    can observe: motor spin-up afterwards is invisible to it.
    """
    out = {"n": len(rows)}
    for key in ("server_ms", "request_ms", "render_ms", "select_to_ready_ms",
                "vibrate_call_ms", "machine_ms"):
        out[key] = _stats([r.get(key) for r in rows])
    out["end_to_end_ms_autoplay"] = _stats(
        [r.get("end_to_end_ms") for r in rows if r.get("autoplay")])
    return out


def _stats(values: list) -> dict | None:
    v = np.array([x for x in values if isinstance(x, (int, float))], dtype=float)
    if v.size == 0:
        return None
    return {"n": int(v.size), "median": round(float(np.median(v)), 2),
            "p95": round(float(np.percentile(v, 95)), 2)}
