"""Every row a campaign's model learns from, with where it came from
(`GET /api/campaigns/{id}/rows`, docs/CAMPAIGNS_API.md "Campaign rows").

A campaign's `base_rows` are the uploaded sheet followed by every measured run
folded in by a reanalyze, in order. The first `history[0].n_base` rows are the
upload; each later row matches, in order, an entry of `recorded_runs` (written
by `commit_fold`). Campaigns folded before `recorded_runs` existed still list
their folded rows, without a run id or round.

`_kind` is `measured` for every row today: the campaign store holds only
measured values. Simulated and synthetic rows arrive with the virtual-runs
branch and will carry their own `_kind`; this route never mixes them in
silently.
"""
from __future__ import annotations

from typing import Any

MAX_LIMIT = 10_000
DEFAULT_LIMIT = 2_000


def campaign_rows(state: dict[str, Any], *, offset: int = 0, limit: int = DEFAULT_LIMIT) -> dict[str, Any]:
    base: list[dict[str, Any]] = list(state.get("base_rows") or [])
    history = state.get("history") or []
    n_upload = history[0].get("n_base") if history and isinstance(history[0], dict) else None
    if not isinstance(n_upload, int) or not 0 <= n_upload <= len(base):
        n_upload = len(base)
    recorded = [r for r in state.get("recorded_runs") or [] if isinstance(r, dict)]
    columns: list[str] = []
    for row in base:
        for k in row:
            if k not in columns:
                columns.append(str(k))
    rows: list[dict[str, Any]] = []
    for i, row in enumerate(base):
        if i < n_upload:
            meta = {"_origin": "upload", "_round": 0, "_run_id": None, "_upload_row": i + 1}
        else:
            j = i - n_upload
            rec = recorded[j] if j < len(recorded) else {}
            rnd = rec.get("round")
            meta = {
                "_origin": "round",
                # 1-based like `stage.round`: a run started before the first
                # reanalyze is from round 1.
                "_round": int(rnd) + 1 if isinstance(rnd, int) else None,
                "_run_id": rec.get("run_id"),
                "_upload_row": None,
            }
        rows.append({**row, **meta, "_kind": "measured", "_index": i})
    start = max(0, int(offset))
    stop = start + max(0, min(int(limit), MAX_LIMIT))
    return {
        "target": state.get("target"),
        "columns": columns,
        "n_rows": len(rows),
        "n_upload": n_upload,
        "n_from_rounds": len(rows) - n_upload,
        "offset": start,
        "limit": stop - start,
        "rows": rows[start:stop],
        "kinds": ["measured"],
        "kinds_note": "Every row is a measured value. Simulated or synthetic rows are not stored here.",
    }


__all__ = ["DEFAULT_LIMIT", "MAX_LIMIT", "campaign_rows"]
