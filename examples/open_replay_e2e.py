#!/usr/bin/env python3
"""End-to-end run of the portal's loop over HTTP on open published datasets.

    KALOS_OPEN_DATA=/path/to/open KALOS_AMII_DATA=/path/to/amii \\
    KALOS_STATE_DIR=/tmp/redesign/replay/state KALOS_PORT=8061 python -m kalos.portal &
    python examples/open_replay_e2e.py --base http://127.0.0.1:8061 --dataset all

For each dataset the driver does what the web app does:
  POST /api/run (seed rows + target + roles) -> proposals
  POST /api/campaign/start (proposed batch)
  "lab": reveal the held-out true value of each started recipe (see `match_recipe`)
  POST /api/campaign/result (per run)
  POST /api/campaign/reanalyze (fold + refit) -> next proposals
for `--rounds` rounds. Every call is logged with its HTTP status, wall time, the
server's resident memory (if `--server-pid` is known), and the response (or error
text). Full JSON responses are saved under `--out`.

The held-out "lab" is a nearest-neighbour lookup in the source data: a proposal
rarely sits exactly on a measured grid, so its revealed value is the closest
UNUSED measured row (same categorical levels, nearest in min-max normalised
continuous space). The match distance is recorded. This is a retrospective replay,
not validation.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kalos.bench.open_datasets import (  # noqa: E402
    AMII_SETS,
    OpenSheet,
    load_open_sheet,
)

# Seed size = the existing data the client uploads; batch = recipes per round.
PLAN = {
    "pichia_HSA": dict(seed=24, batch=4),
    "pichia_HerceptinA": dict(seed=24, batch=4),
    "pichia_RBDJ": dict(seed=24, batch=4),
    "trichoderma_ambr": dict(seed=8, batch=2),
    "trichoderma_ambr_condition": dict(seed=8, batch=2),
    "candida_rsm_nr1": dict(seed=6, batch=2),
    "DBO_rat_myocyte": dict(seed=24, batch=4),
    "MOBO_rat_myocyte": dict(seed=24, batch=4),
    "HeLa_regular": dict(seed=24, batch=4),
    "HeLa_timesaving": dict(seed=24, batch=4),
    "scfv_refolding_18270843": dict(seed=22, batch=4),   # seed = DoE1 history (seed_mask); lab = 25 BO runs
    "rudt_ib_refolding_16315733": dict(seed=8, batch=2),
    "cho_bioreactor_20829178": dict(seed=10, batch=2),
    "Tcell_expansion": dict(seed=24, batch=4),
    "TF1_expansion": dict(seed=24, batch=4),
}
SLOW_S = 60.0


class Log:
    def __init__(self, path: Path, server_pid: int | None):
        self.path = path
        self.server_pid = server_pid
        self.events: list[dict] = []
        path.parent.mkdir(parents=True, exist_ok=True)

    def rss_mb(self) -> float | None:
        if not self.server_pid:
            return None
        try:
            out = subprocess.run(["ps", "-o", "rss=", "-p", str(self.server_pid)],
                                 capture_output=True, text=True, timeout=5).stdout.strip()
            return round(int(out) / 1024, 1) if out else None
        except Exception:  # noqa: BLE001 - memory is best-effort
            return None

    def add(self, ev: dict) -> None:
        ev["rss_mb"] = self.rss_mb()
        self.events.append(ev)
        with self.path.open("a") as fh:
            fh.write(json.dumps(ev, default=str) + "\n")
        flag = " SLOW" if ev.get("wall_s", 0) > SLOW_S else ""
        err = f" err={ev['error']!r}" if ev.get("error") else ""
        print(f"[{ev['dataset']}] {ev['step']:<18} HTTP {ev.get('status')} "
              f"{ev.get('wall_s', 0):7.2f}s rss={ev['rss_mb']}MB{flag}{err}", flush=True)


def call(client: httpx.Client, log: Log, dataset: str, step: str, method: str, path: str,
         *, save: Path | None = None, **kw) -> tuple[int | None, dict | None]:
    t0 = time.perf_counter()
    status, body, error = None, None, None
    try:
        r = client.request(method, path, **kw)
        status = r.status_code
        try:
            body = r.json()
        except ValueError:
            body = {"raw": r.text[:500]}
        if status >= 400:
            error = json.dumps(body)[:800]
    except Exception as exc:  # noqa: BLE001 - recorded, the run continues to the next step
        error = f"client exception {type(exc).__name__}: {exc}"[:800]
    wall = time.perf_counter() - t0
    if save is not None and body is not None:
        save.parent.mkdir(parents=True, exist_ok=True)
        save.write_text(json.dumps(body, indent=1, default=str))
    log.add({"dataset": dataset, "step": step, "method": method, "path": path, "status": status,
             "wall_s": round(wall, 2), "error": error})
    return status, body


def match_recipe(recipe: dict, pool: pd.DataFrame, target: str, used: set[int], norm: dict) -> tuple[int | None, float]:
    """Nearest UNUSED measured row: categorical levels must match exactly, numeric
    features are compared in min-max normalised space. Returns (row label, distance)."""
    cand = pool.index[~pool.index.isin(list(used))]
    best, best_d = None, float("inf")
    for idx in cand:
        row = pool.loc[idx]
        d2, ok = 0.0, True
        for col, val in recipe.items():
            if col not in pool.columns or col == target:
                continue
            if col in norm["cat"]:
                if str(row[col]) != str(val):
                    ok = False
                    break
            else:
                span = norm["span"][col]
                try:
                    d2 += ((float(row[col]) - float(val)) / span) ** 2
                except (TypeError, ValueError):
                    ok = False
                    break
        if ok and d2 < best_d:
            best, best_d = idx, d2
    return (best, float(np.sqrt(best_d))) if best is not None else (None, float("inf"))


def started_recipe(pool: pd.DataFrame, label: int, keys) -> dict:
    """The recipe a lab would start for measured row `label`: the row's own values
    for the proposal's keys (JSON-safe scalars)."""
    out: dict = {}
    for k in keys:
        v = pool.loc[label, k]
        out[k] = float(v) if isinstance(v, (float, int, np.floating, np.integer)) else str(v)
    return out


def plan_batch(props: list[dict], oracle: pd.DataFrame, df: pd.DataFrame, target: str,
               used: set[int], norm: dict) -> dict:
    """Match each proposal to a distinct unused measured row. Returns
    `matched`: [(proposed recipe, row label, started recipe, distance)] and
    `unmatched`: [(proposed recipe, None)]. Does not mutate `used`."""
    taken = set(used)
    matched, unmatched = [], []
    for p in props:
        rec = p.get("recipe") or {}
        label, dist = match_recipe(rec, oracle, target, taken, norm)
        if label is None:
            unmatched.append((rec, None))
            continue
        taken.add(label)
        matched.append((rec, int(label), started_recipe(df, label, rec.keys()), dist))
    return {"matched": matched, "unmatched": unmatched}


def _matched_start_row(srec: dict, label: int, dist: float) -> dict:
    """A start entry for the MATCHED row. Prediction fields belong to the proposed
    recipe, not to this one, so they are not sent."""
    return {"recipe": srec, "pred": None, "std": None, "mode": None,
            "reason": f"replay: started held-out row {label} (proposal distance {dist:.3f})"}


def proposal_checks(props: list[dict], analysis: dict, base: pd.DataFrame, target: str) -> dict:
    """Sanity of a proposed batch: inside the engine's own search box, inside the
    observed range of the base rows, categorical levels known, and repeats."""
    box = analysis.get("feature_ranges") or {}
    out = {"n": len(props), "outside_search_box": 0, "outside_observed_range": 0,
           "unknown_levels": 0, "within_batch_repeats": 0, "keys_missing_in_base": []}
    seen = []
    for p in props:
        rec = p.get("recipe") or {}
        for k, v in rec.items():
            if isinstance(v, (int, float)) and k in box:
                lo, hi = box[k]
                if v < lo - 1e-9 or v > hi + 1e-9:
                    out["outside_search_box"] += 1
            if isinstance(v, (int, float)) and k in base.columns and pd.api.types.is_numeric_dtype(base[k]):
                if v < base[k].min() - 1e-9 or v > base[k].max() + 1e-9:
                    out["outside_observed_range"] += 1
            elif isinstance(v, str) and k in base.columns:
                if v not in set(base[k].astype(str)):
                    out["unknown_levels"] += 1
            elif k not in base.columns and k != target:
                out["keys_missing_in_base"].append(k)
        key = json.dumps(rec, sort_keys=True, default=str)
        if key in seen:
            out["within_batch_repeats"] += 1
        seen.append(key)
    out["keys_missing_in_base"] = sorted(set(out["keys_missing_in_base"]))
    return out


def run_dataset(client: httpx.Client, log: Log, name: str, sheet: OpenSheet, out: Path,
                *, seed: int, rounds: int, seed_n: int, batch: int, full_upload: bool = False) -> dict:
    df = sheet.df.reset_index(drop=True)
    target = sheet.target
    feats = [c for c in df.columns if c != target]
    cat_cols = set((sheet.roles or {}).get("categoricals", []))
    cat_cols |= {c for c in feats if not pd.api.types.is_numeric_dtype(df[c])}
    feats = [c for c in feats if c not in set((sheet.roles or {}).get("ids", []))]
    numeric = [c for c in feats if c not in cat_cols]
    norm = {"cat": cat_cols, "span": {c: float(max(df[c].max() - df[c].min(), 1e-12)) for c in numeric}}
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(df))
    if sheet.seed_mask is not None and not full_upload:
        # The dataset's own history is the seed; the rest is the held-out lab (no random draw).
        seed_idx = [i for i, m in enumerate(sheet.seed_mask) if m]
        oracle_idx = {i for i, m in enumerate(sheet.seed_mask) if not m}
    else:
        seed_idx = [int(i) for i in perm[:seed_n]] if not full_upload else list(range(len(df)))
        oracle_idx = set(int(i) for i in perm[seed_n:]) if not full_upload else set()
    base = df.loc[seed_idx]
    oracle_df = df.loc[sorted(oracle_idx)] if oracle_idx else df.iloc[0:0]
    summary: dict = {"dataset": name, "rows_in_source": len(df), "seed_rows": len(seed_idx),
                     "oracle_rows": len(oracle_df), "target": target, "batch": batch,
                     "notes": sheet.notes, "rounds": [], "pool_max": float(df[target].max())}

    csv = base.to_csv(index=False).encode()
    data = {"target": target, "replace": "true"}
    if sheet.roles:
        data["roles"] = json.dumps(sheet.roles)
    status, body = call(client, log, name, "upload", "POST", "/api/run",
                        files={"file": (f"{name}.csv", csv, "text/csv")}, data=data,
                        save=out / "upload.json")
    summary["upload_status"] = status
    if full_upload:
        summary["full_upload_status"] = status
        return summary
    if status != 200 or body is None:
        summary["stopped"] = f"upload returned {status}"
        return summary
    props = body.get("proposals") or []
    summary["upload_findings"] = _findings(body)
    summary["upload_checks"] = proposal_checks(props, body, base, target)
    summary["upload_gating"] = body.get("proposal_gating")
    summary["upload_validation"] = (body.get("validation") or {}).get("status")
    summary["upload_n_proposals"] = len(props)
    used: set[int] = set()
    best_seen = float(base[target].max())
    analysis = body

    for r in range(1, rounds + 1):
        rnd: dict = {"round": r, "proposals_in": len(props)}
        batch_props = props[:batch]
        # Match BEFORE starting. The run that is started carries the MATCHED row's own
        # recipe, and the revealed value belongs to that row. Attributing a row's value to
        # a different proposed recipe would fold imputed data into the fit as a measurement.
        plan = plan_batch(batch_props, oracle_df, df, target, used, norm)
        rnd["unmatched"] = [rec for rec, label in plan["unmatched"]]
        rnd["proposal_to_started"] = [
            {"proposed": rec, "started": srec, "row": int(label), "distance": round(dist, 4)}
            for rec, label, srec, dist in plan["matched"]
        ]
        if not plan["matched"]:
            rnd["stopped"] = "no proposed recipe matched an unused measured row; nothing to start"
            summary["rounds"].append(rnd)
            break
        status, started = call(client, log, name, f"start_r{r}", "POST", "/api/campaign/start",
                               json={"recipes": [_matched_start_row(srec, label, dist)
                                                 for _, label, srec, dist in plan["matched"]]},
                               save=out / f"start_r{r}.json")
        rnd["start_status"] = status
        if status != 200 or not started:
            rnd["stopped"] = f"start returned {status}"
            summary["rounds"].append(rnd)
            break
        runs = started.get("started", [])
        measured = []
        for run, (_, label, srec, dist) in zip(runs, plan["matched"]):
            used.add(label)
            value = float(df.loc[label, target])
            measured.append({"match_row": int(label), "distance": round(dist, 4), "value": value,
                             "started_recipe": srec, "run_recipe_ok": run.get("recipe") == srec})
            st, _ = call(client, log, name, f"result_r{r}", "POST", "/api/campaign/result",
                         json={"id": run["id"], "value": value})
            if st != 200:
                rnd.setdefault("result_errors", []).append(st)
        rnd["measured"] = measured
        rnd["match_distance_max"] = max((m["distance"] for m in measured), default=None)
        rnd["revealed_best_this_round"] = max((m["value"] for m in measured), default=None)
        best_seen = max([best_seen] + [m["value"] for m in measured])
        rnd["best_so_far"] = best_seen
        rnd["frac_of_pool_max"] = round(best_seen / summary["pool_max"], 4) if summary["pool_max"] else None
        status, rb = call(client, log, name, f"reanalyze_r{r}", "POST", "/api/campaign/reanalyze",
                          save=out / f"reanalyze_r{r}.json")
        rnd["reanalyze_status"] = status
        if status != 200 or rb is None:
            rnd["stopped"] = f"reanalyze returned {status}"
            summary["rounds"].append(rnd)
            break
        analysis = rb.get("analysis") or {}
        props = analysis.get("proposals") or []
        rnd["proposals_out"] = len(props)
        rnd["findings"] = _findings(analysis)
        rnd["checks"] = proposal_checks(props, analysis, _base_from(df, used, seed_idx, oracle_df), target)
        rnd["gating"] = analysis.get("proposal_gating")
        rnd["validation"] = (analysis.get("validation") or {}).get("status")
        rnd["promotion"] = analysis.get("promotion")
        rnd["reliability"] = analysis.get("reliability")
        rnd["proposal_notice"] = analysis.get("proposal_notice")
        rnd["design_box_exclusions"] = analysis.get("design_box_exclusions")
        rnd["n_units"] = (analysis.get("drivers") or {}).get("n_units") if isinstance(analysis.get("drivers"), dict) else None
        summary["rounds"].append(rnd)
        if not props:
            rnd["stopped"] = "no proposals returned"
            break
    summary["final_best"] = best_seen
    summary["final_frac_of_pool_max"] = round(best_seen / summary["pool_max"], 4) if summary["pool_max"] else None
    return summary


def _base_from(df: pd.DataFrame, used: set[int], seed_idx: list[int], oracle_df: pd.DataFrame) -> pd.DataFrame:
    """The rows the server now holds as base data: seed rows plus the revealed ones."""
    return df.loc[sorted(set(seed_idx) | set(used))]


def _findings(body: dict) -> list[str]:
    out = []
    for f in (body.get("validation") or {}).get("findings", []) or []:
        if isinstance(f, dict):
            out.append(f"{f.get('severity', '?')}:{f.get('code', f.get('check', '?'))}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://127.0.0.1:8061")
    ap.add_argument("--dataset", nargs="+", default=["all"])
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/tmp/redesign/replay/e2e")
    ap.add_argument("--server-pid", type=int, default=None)
    ap.add_argument("--full-upload-probe", action="store_true",
                    help="also upload the WHOLE sheet once per large dataset (fit-row cap probe)")
    a = ap.parse_args()

    names = list(PLAN) if a.dataset == ["all"] else a.dataset
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = Log(out / "events.jsonl", a.server_pid)
    results = []
    with httpx.Client(base_url=a.base, timeout=900.0) as client:
        for name in names:
            plan = PLAN.get(name, dict(seed=24, batch=4))
            try:
                sheet = load_open_sheet(name)
            except Exception as exc:  # noqa: BLE001
                log.add({"dataset": name, "step": "load", "status": None, "wall_s": 0,
                         "error": f"{type(exc).__name__}: {exc}"})
                continue
            s = run_dataset(client, log, name, sheet, out / name, seed=a.seed, rounds=a.rounds,
                            seed_n=plan["seed"], batch=plan["batch"])
            results.append(s)
            (out / name / "summary.json").write_text(json.dumps(s, indent=1, default=str))
            if a.full_upload_probe and name in AMII_SETS + ("pichia_RBDJ",):
                probe = run_dataset(client, log, name + "_FULL", sheet, out / (name + "_FULL"),
                                    seed=a.seed, rounds=0, seed_n=0, batch=0, full_upload=True)
                (out / (name + "_FULL") / "summary.json").write_text(json.dumps(probe, indent=1, default=str))
    (out / "all_summaries.json").write_text(json.dumps(results, indent=1, default=str))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
