"""Screening table: every arm directory under ROOT against a reference directory.

    python benchmarks/ml_v2/acq/screen_table.py ROOT REF_DIR [--suite closed_loop] [--md OUT.md]

Per arm: paired mean difference in FINAL regret (candidate - reference) with a
bootstrap 95% CI over instances (the protocol's C1a statistic), the same for the
mean regret over rounds 1..5 (a lower-variance screening signal, not a protocol
criterion), per-family mean differences, coverage at proposals at 90/95 % with
cluster CIs, the 90 % width ratio at proposals, median CPU seconds per analysis,
arm errors and filled slots. Tuning seeds only; never a held-out claim.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from kalos.bench.ml_v2.report import (
    _cov_clusters,
    boot_mean_ci,
    cluster_rate_ci,
    load_records,
)


def _final(rec):
    return (rec.get("rounds") or [{}])[-1].get("regret_obs")


def _auc(rec):
    v = [r.get("regret_obs") for r in (rec.get("rounds") or [])[1:]]
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else None


def _cov(recs, lv):
    cl, w = _cov_clusters(recs, lv, "at_proposals")
    h, t = sum(a for a, _ in cl), sum(b for _, b in cl)
    return (h / t if t else None), cluster_rate_ci(cl), (float(np.median(w)) if w else None)


def table(root: Path, ref: Path, suite: str, tasks: list[str] | None = None) -> list[dict]:
    R = {(r["suite"], r["task"], r["seed"]): r for r in load_records(ref) if r["suite"] == suite}
    rows = []
    dirs = [root] if (root / "run_meta.json").is_file() else sorted(p for p in root.iterdir() if (p / "raw").is_dir())
    for d in dirs:
        C = [c for c in load_records(d) if c["suite"] == suite and (not tasks or c["task"] in tasks)]
        pairs = [(c, R[(c["suite"], c["task"], c["seed"])]) for c in C if (c["suite"], c["task"], c["seed"]) in R]
        if not pairs:
            continue
        df = [_final(c) - _final(r) for c, r in pairs if _final(c) is not None and _final(r) is not None]
        da = [_auc(c) - _auc(r) for c, r in pairs if _auc(c) is not None and _auc(r) is not None]
        fam = defaultdict(list)
        for c, r in pairs:
            if _final(c) is not None and _final(r) is not None:
                fam[c["task"]].append(_final(c) - _final(r))
        cc = [c for c, _ in pairs]
        rr = [r for _, r in pairs]
        c90, ci90, w90 = _cov(cc, 0.9)
        c95, ci95, _ = _cov(cc, 0.95)
        _, _, w90r = _cov(rr, 0.9)
        cpu = [x["cpu_s"] for c in cc for x in c["rounds"][1:] if x.get("cpu_s") is not None]
        cpur = [x["cpu_s"] for c in rr for x in c["rounds"][1:] if x.get("cpu_s") is not None]
        err = sum(1 for c in cc for x in c["rounds"][1:] if x.get("error"))
        fill = sum(int(x.get("n_filled") or 0) for c in cc for x in c["rounds"][1:])
        rows.append({
            "arm": d.name, "n": len(df), "regret": float(np.mean([_final(c) for c, _ in pairs])),
            "ref_regret": float(np.mean([_final(r) for _, r in pairs])),
            "d_final": float(np.mean(df)), "ci_final": boot_mean_ci(df),
            "d_auc": float(np.mean(da)), "ci_auc": boot_mean_ci(da),
            "fam": {k: round(float(np.mean(v)), 3) for k, v in sorted(fam.items())},
            "cov90": c90, "cov90_ci": ci90, "cov95": c95, "cov95_ci": ci95,
            "w90_ratio": (w90 / w90r) if w90 and w90r else None,
            "cpu_med": float(np.median(cpu)) if cpu else None, "cpu_ratio": (np.median(cpu) / np.median(cpur)) if cpu and cpur else None,
            "errors": err, "filled": fill,
        })
    return sorted(rows, key=lambda r: r["d_final"])


def to_md(rows: list[dict]) -> str:
    fams = sorted({k for r in rows for k in r["fam"]})
    h = ("| arm | n | regret | d final [95% CI] | d mean-over-rounds [95% CI] | cov90 | cov95 | w90 ratio | cpu x ref | err/fill | "
         + " | ".join(fams) + " |")
    out = [h, "|" + "---|" * (10 + len(fams))]
    for r in rows:
        f = lambda ci: f"[{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else ""  # noqa: E731
        out.append(f"| {r['arm']} | {r['n']} | {r['regret']:.3f} | {r['d_final']:+.3f} {f(r['ci_final'])} | "
                   f"{r['d_auc']:+.3f} {f(r['ci_auc'])} | {r['cov90'] if r['cov90'] is None else round(r['cov90'], 3)} | "
                   f"{r['cov95'] if r['cov95'] is None else round(r['cov95'], 3)} | "
                   f"{'' if r['w90_ratio'] is None else round(r['w90_ratio'], 2)} | "
                   f"{'' if r['cpu_ratio'] is None else round(float(r['cpu_ratio']), 2)} | {r['errors']}/{r['filled']} | "
                   + " | ".join(f"{r['fam'].get(k, float('nan')):+.3f}" for k in fams) + " |")
    return "\n".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("ref", type=Path)
    ap.add_argument("--suite", default="closed_loop")
    ap.add_argument("--tasks", nargs="*")
    ap.add_argument("--md", type=Path)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    rows = table(a.root, a.ref, a.suite, a.tasks)
    md = to_md(rows)
    print(md)
    if a.md:
        a.md.write_text(md + "\n")
    if a.json:
        a.json.write_text(json.dumps(rows, indent=1, default=float) + "\n")
