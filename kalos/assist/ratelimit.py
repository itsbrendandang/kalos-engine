"""Per-principal limit on `/api/assist/ask` (each question can call a model).

A sliding one-minute window per (tenant, subject), in process memory.
`KALOS_ASSIST_ASK_PER_MINUTE` (default 20; 0 turns the limit off). A single
engine process serves one deployment, so in-memory is enough here; the limit is
a brake on runaway clients, not an accounting record.
"""
from __future__ import annotations

import math
import os
import threading
import time
from collections import deque

_LOCK = threading.Lock()
_SEEN: dict[tuple[str, str], deque[float]] = {}
WINDOW_S = 60.0


def per_minute() -> int:
    raw = (os.environ.get("KALOS_ASSIST_ASK_PER_MINUTE") or "").strip()
    try:
        value = int(raw) if raw else 20
    except ValueError:
        value = 20
    return max(0, min(value, 10_000))


def check(tenant: str, subject: str | None, *, now: float | None = None) -> int:
    """0 when the call may go ahead (and counts it), else the seconds to wait."""
    limit = per_minute()
    if limit == 0:
        return 0
    t = time.monotonic() if now is None else now
    key = (tenant, subject or "")
    with _LOCK:
        seen = _SEEN.setdefault(key, deque())
        while seen and t - seen[0] >= WINDOW_S:
            seen.popleft()
        if len(seen) >= limit:
            return max(1, math.ceil(WINDOW_S - (t - seen[0])))
        seen.append(t)
        return 0


def reset() -> None:
    """Forget every window (tests)."""
    with _LOCK:
        _SEEN.clear()


__all__ = ["check", "per_minute", "reset"]
