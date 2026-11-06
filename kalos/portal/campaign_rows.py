"""Every row a campaign's model learns from, with where it came from
(`GET /api/campaigns/{id}/rows`, docs/CAMPAIGNS_API.md "Campaign rows").

A campaign's `base_rows` are the uploaded sheet followed by every measured run
folded in by a reanalyze, in order. The first `history[0].n_base` rows are the
upload; each later row matches, in order, an entry of `recorded_runs` (written
by `commit_fold`). Campaigns folded before `recorded_runs` existed still list
their folded rows, without a run id or round.

`_kind` is `measured` for a row with a target value and `no_result` for an
upload row whose target cell is blank (the model does not learn from it). The
campaign store holds no simulated or synthetic rows. They arrive with the
virtual-runs branch and will carry their own `_kind`; this route never mixes
them in silently.

A sheet column whose name is one of the metadata names (`_origin`, `_round`,
`_run_id`, `_kind`, `_index`, `_upload_row`) is shown as `<name> (sheet)`, so
neither overwrites the other; `renamed_columns` lists each such rename.
"""
from __future__ import annotations

import math
from typing import Any

MAX_LIMIT = 10_000
DEFAULT_LIMIT = 2_000
META_KEYS: tuple[str, ...] = ("_origin", "_round", "_run_id", "_kind", "_index", "_upload_row")


def _blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return isinstance(value, str) and not value.strip()


def _renames(columns: list[str]) -> dict[str, str]:
    """Sheet column -> the name it is shown under, for columns that clash with
    a metadata name: `<name> (sheet)`, or `<name> (sheet 2)`, ... if taken."""
    taken = set(columns) | set(META_KEYS)
    out: dict[str, str] = {}
    for col in columns:
        if col not in META_KEYS:
            continue
        new, k = f"{col} (sheet)", 2
        while new in taken:
            new, k = f"{col} (sheet {k})", k + 1
        taken.add(new)
        out[col] = new
    return out


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
    renamed = _renames(columns)
    columns = [renamed.get(c, c) for c in columns]
    target = state.get("target")
    rows: list[dict[str, Any]] = []
    kinds: list[str] = []
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
        # Only an upload row can lack a result: a folded run always has one.
        kind = "no_result" if i < n_upload and target in row and _blank(row[target]) else "measured"
        if kind not in kinds:
            kinds.append(kind)
        cells = {renamed.get(str(k), str(k)): v for k, v in row.items()}
        rows.append({**cells, **meta, "_kind": kind, "_index": i})
    start = max(0, int(offset))
    stop = start + max(0, min(int(limit), MAX_LIMIT))
    return {
        "target": target,
        "columns": columns,
        "n_rows": len(rows),
        "n_upload": n_upload,
        "n_from_rounds": len(rows) - n_upload,
        "offset": start,
        "limit": stop - start,
        "rows": rows[start:stop],
        "kinds": kinds or ["measured"],
        "kinds_note": (
            "measured: the row has a result. no_result: the sheet row has no target value, "
            "so the model does not learn from it. Simulated or synthetic rows are not stored here."
        ),
        "renamed_columns": [{"column": old, "shown_as": new} for old, new in renamed.items()],
        "renamed_note": (
            "Sheet columns named like the row labels Kalos adds are shown with \"(sheet)\" after "
            "their name, so neither hides the other."
            if renamed else None
        ),
    }


__all__ = ["DEFAULT_LIMIT", "MAX_LIMIT", "campaign_rows"]
