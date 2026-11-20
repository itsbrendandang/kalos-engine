"""Summaries and the candidate-vs-reference comparison (PROTOCOL.md sections 7-8).

`summarize_dir(DIR)` writes DIR/summary.json and DIR/summary.md for one arm:
every metric per task (family or dataset), per regime, and pooled per group,
with 95% CIs. `compare_dirs(CAND, REF, ...)` pairs jobs by (suite, task, seed)
and evaluates the success criteria C1-C8 exactly as pre-registered.
Report-only groups (internal, info_only, user) are summarized but never enter
a criterion.
"""
from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import wilcoxon

from .protocol import (
    ALPHA,
    BOOT_SEED,
    CRITERIA,
    DECISION_GROUPS,
    LEVELS,
    N_BOOT,
    POOL_TOP_FRACTION,
    PROTOCOL_VERSION,
    REGRET_TARGETS,
)


# ------------------------------------------------------------------ stats
def boot_mean_ci(x: Sequence[float], n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> list[float] | None:
    v = np.asarray([a for a in x if a is not None and np.isfinite(a)], float)
    if len(v) < 2:
        return None
    rng = np.random.default_rng(seed)
    bs = v[rng.integers(0, len(v), (n_boot, len(v)))].mean(1)
    return [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]


def cluster_rate_ci(clusters: Sequence[tuple[int, int]], n_boot: int = N_BOOT,
                    seed: int = BOOT_SEED) -> list[float] | None:
    """(hits, total) per cluster (one instance); bootstrap over clusters."""
    c = [(h, t) for h, t in clusters if t > 0]
    if len(c) < 2:
        return None
    h = np.array([a for a, _ in c], float)
    t = np.array([b for _, b in c], float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(c), (n_boot, len(c)))
    bs = h[idx].sum(1) / t[idx].sum(1)
    return [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]


def wilson(k: int, n: int, z: float = 1.959964) -> list[float] | None:
    if n == 0:
        return None
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(float(mid - half), 4), round(float(mid + half), 4)]


def holm(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def _r(v: Any, nd: int = 4) -> Any:
    return None if v is None or (isinstance(v, float) and not np.isfinite(v)) else round(float(v), nd)


# ------------------------------------------------------------------ loading
def load_records(d: Path) -> list[dict[str, Any]]:
    out = []
    for f in sorted((Path(d) / "raw").rglob("s*.json")):
        rec = json.loads(f.read_text(encoding="utf-8"))
        if not rec.get("skipped"):
            out.append(rec)
    return out


def _cov_clusters(recs: Iterable[dict[str, Any]], level: float, where: str) -> tuple[list[tuple[int, int]], list[float]]:
    """(hits, total) per instance and the widths, at proposals or held-out runs."""
    clusters, widths = [], []
    for rec in recs:
        h = t = 0
        rounds = rec.get("rounds") or [rec]
        for r in rounds:
            block = r.get(where)
            if not block:
                continue
            b = block.get(str(level))
            if not b:
                continue
            if where == "heldout":
                h, t = h + b["inside"], t + b["n"]
                if b.get("median_width") is not None:
                    widths.append(b["median_width"])
            else:
                h, t = h + sum(b["inside"]), t + len(b["inside"])
                widths += b["widths"]
        clusters.append((h, t))
    return clusters, widths


def _cov_summary(recs: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for lv in LEVELS:
        for where in ("at_proposals", "heldout"):
            cl, w = _cov_clusters(recs, lv, where)
            hits, tot = sum(a for a, _ in cl), sum(b for _, b in cl)
            out[f"{where}_{lv}"] = {"coverage": _r(hits / tot) if tot else None, "n": tot,
                                    "ci95": cluster_rate_ci(cl), "median_width": _r(np.median(w)) if w else None}
    return out


def _runs_to_target(rec: dict[str, Any], key: str, thr: float) -> tuple[int, bool]:
    for r in rec["rounds"]:
        v = r.get(key)
        if v is not None and v <= thr:
            return int(r["n_total"]), True
    return int(rec["rounds"][-1]["n_total"]) + 1, False


def _wall(recs: list[dict[str, Any]]) -> dict[str, Any]:
    w = [r["wall_s"] for rec in recs for r in (rec.get("rounds") or rec.get("cuts") or rec.get("folds") or [rec])
         if r.get("wall_s") is not None]
    c = [r["cpu_s"] for rec in recs for r in (rec.get("rounds") or rec.get("cuts") or rec.get("folds") or [rec])
         if r.get("cpu_s") is not None]
    return {"median_wall_s": _r(np.median(w), 2) if w else None, "p90_wall_s": _r(np.percentile(w, 90), 2) if w else None,
            "median_cpu_s": _r(np.median(c), 2) if c else None, "n_calls": len(w),
            "arm_errors": sum(1 for rec in recs for r in (rec.get("rounds") or rec.get("cuts") or rec.get("folds") or [rec])
                              if r.get("error"))}


def _loop_task(recs: list[dict[str, Any]]) -> dict[str, Any]:
    final = [rec["rounds"][-1].get("regret_obs") for rec in recs]
    finb = [rec["rounds"][-1].get("regret_best") for rec in recs]
    out: dict[str, Any] = {"n_instances": len(recs),
                           "regret_obs_final": {"mean": _r(np.nanmean([v for v in final if v is not None])) if any(v is not None for v in final) else None,
                                                "ci95": boot_mean_ci(final)},
                           "regret_best_final": {"mean": _r(np.nanmean([v for v in finb if v is not None])) if any(v is not None for v in finb) else None,
                                                 "ci95": boot_mean_ci(finb)}}
    by_n0: dict[int, list[float]] = defaultdict(list)
    for rec in recs:
        v = rec["rounds"][-1].get("regret_obs")
        if v is not None:
            by_n0[int(rec["n0"])].append(v)
    out["regret_obs_final_by_n0"] = {str(k): {"mean": _r(np.mean(v)), "n": len(v), "ci95": boot_mean_ci(v)}
                                     for k, v in sorted(by_n0.items())}
    if recs and "top5_threshold" not in recs[0]:
        for thr in REGRET_TARGETS:
            vals = [_runs_to_target(rec, "regret_obs", thr) for rec in recs]
            out[f"runs_to_regret_{thr}"] = {"median": _r(float(np.median([v for v, _ in vals])), 1),
                                            "reached": sum(ok for _, ok in vals), "n": len(vals),
                                            "censored_at": "budget + 1"}
    else:
        vals = []
        for rec in recs:
            seq = rec.get("order_values") or []
            hit = next((i + 1 for i, v in enumerate(np.maximum.accumulate(seq)) if v >= rec["top5_threshold"]), None)
            vals.append((hit if hit is not None else len(seq) + 1, hit is not None))
        out[f"runs_to_pool_top{int(POOL_TOP_FRACTION * 100)}pct"] = {
            "median": _r(np.median([v for v, _ in vals]), 1), "reached": sum(ok for _, ok in vals), "n": len(vals)}
    rr = [r for rec in recs for r in rec["rounds"][1:]]
    out["trusted_rate_rounds"] = {"rate": _r(np.mean([r.get("trusted", False) for r in rr])) if rr else None,
                                  "ci95": wilson(sum(bool(r.get("trusted")) for r in rr), len(rr))}
    out["filled_proposals"] = int(sum(r.get("n_filled") or 0 for r in rr))
    out["infeasible_proposals"] = int(sum(r.get("n_infeasible") or 0 for r in rr))
    ps = [r.get("prospective_spearman") for r in rr if r.get("prospective_spearman") is not None]
    out["prospective_spearman_median"] = _r(float(np.median([float(p) for p in ps]))) if ps else None
    out.update(_cov_summary(recs))
    out.update(_wall(recs))
    return out


def _snap_task(recs: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    by_n: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for rec in recs:
        by_n[int(rec["n"])].append(rec)
    for n, rs in sorted(by_n.items()):
        k = sum(bool(r.get("trusted")) for r in rs)
        ps = [float(r["prospective_spearman"]) for r in rs if r.get("prospective_spearman") is not None]
        out[f"n{n}"] = {"n_instances": len(rs), "trusted_rate": _r(k / len(rs)), "trusted_ci95": wilson(k, len(rs)),
                        "withheld_rate": _r(np.mean([bool(r.get("withheld")) for r in rs])),
                        "prospective_spearman_median": _r(float(np.median([float(p) for p in ps]))) if ps else None,
                        **_cov_summary(rs)}
    k = sum(bool(r.get("trusted")) for r in recs)
    out["all"] = {"n_instances": len(recs), "trusted_rate": _r(k / len(recs)) if recs else None,
                  "trusted_ci95": wilson(k, len(recs)), "signal": bool(recs and recs[0].get("signal")),
                  **_cov_summary(recs), **_wall(recs)}
    return out


def _backtest_task(recs: list[dict[str, Any]]) -> dict[str, Any]:
    cuts = [c for rec in recs for c in rec.get("cuts", [])]
    out: dict[str, Any] = {"label": recs[0].get("label") if recs else None, "n_cuts": len(cuts),
                           "real_order": recs[0].get("real_order") if recs else None}
    for m in ("M1", "M2", "M3", "M4"):
        v = [c.get(m) for c in cuts if c.get(m) is not None]
        out[m] = {"mean": _r(np.mean(v)) if v else None, "ci95": boot_mean_ci(v), "n": len(v)}
    for base in ("random_mean", "ofat", "lab"):
        d = [c["M1"] - c[f"M1_{base}"] for c in cuts if c.get("M1") is not None and c.get(f"M1_{base}") is not None]
        ci = boot_mean_ci(d)
        out[f"M1_minus_{base}"] = {"mean": _r(np.mean(d)) if d else None, "ci95": ci, "n": len(d),
                                   "verdict": None if ci is None else ("win" if ci[1] < 0 else "loss" if ci[0] > 0 else "tie")}
    out["trusted_rate"] = _r(np.mean([bool(c.get("trusted")) for c in cuts])) if cuts else None
    out.update(_wall(recs))
    return out


def _folds_task(recs: list[dict[str, Any]]) -> dict[str, Any]:
    folds = [f for rec in recs for f in rec.get("folds", [])]
    out: dict[str, Any] = {"label": recs[0].get("label") if recs else None, "n_folds": len(folds)}
    keys = sorted({k for f in folds for k, v in f.items() if isinstance(v, (int, float)) and not isinstance(v, bool)
                   and k not in ("n_train", "n_test", "wall_s", "cpu_s", "held_out_scale")})
    for k in keys:
        v = [f[k] for f in folds if f.get(k) is not None]
        out[k] = {"mean": _r(np.mean(v)) if v else None, "ci95": boot_mean_ci(v), "n": len(v)}
    if any("direction" in f for f in folds):
        for dname in ("up", "down", "interior"):
            fs = [f for f in folds if f.get("direction") == dname]
            sp_ = [f["spearman"] for f in fs if f.get("spearman") is not None]
            out[f"direction_{dname}"] = {"n": len(fs), "spearman_mean": _r(np.mean(sp_)) if sp_ else None,
                                         "trusted": sum(bool(f.get("trusted")) for f in fs)}
    out["trusted_folds"] = sum(bool(f.get("trusted")) for f in folds)
    out["refused_or_withheld_folds"] = sum(bool(f.get("withheld")) for f in folds)
    for lv in LEVELS:
        h = sum(((f.get("coverage") or {}).get(str(lv), {}) and f["coverage"][str(lv)]["inside"]) or 0 for f in folds)
        t = sum(((f.get("coverage") or {}).get(str(lv), {}) and f["coverage"][str(lv)]["n"]) or 0 for f in folds)
        out[f"coverage_{lv}"] = {"coverage": _r(h / t) if t else None, "n": t, "ci95": wilson(h, t)}
    out.update(_wall(recs))
    return out


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    by: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        by[(r["suite"], r["task"])].append(r)
    tasks: dict[str, Any] = {}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (suite, task), recs in sorted(by.items()):
        recs.sort(key=lambda r: r["seed"])
        group = recs[0].get("group", "")
        if suite in ("closed_loop", "emulator", "pool"):
            body = _loop_task(recs)
            groups[f"{suite}|{group}"] += recs
        elif suite in ("snapshot", "snapshot_real"):
            body = _snap_task(recs)
        elif suite == "backtest":
            body = _backtest_task(recs)
        else:
            body = _folds_task(recs)
        tasks[f"{suite}|{task}"] = {"suite": suite, "task": task, "group": group,
                                    "decides": group in DECISION_GROUPS or group == "real_backtest", **body}
    pooled = {k: {"group": k.split("|")[1], **_loop_task(v)} for k, v in groups.items()}
    nulls = [r for r in records if r["suite"] in ("snapshot", "snapshot_real") and r.get("signal") is False
             and r.get("group") in (*DECISION_GROUPS, "synthetic")]
    k = sum(bool(r.get("trusted")) for r in nulls)
    pooled["false_trust_all_nulls"] = {"rate": _r(k / len(nulls)) if nulls else None, "n": len(nulls),
                                       "ci95": wilson(k, len(nulls))}
    return {"protocol": PROTOCOL_VERSION, "tasks": tasks, "pooled": pooled}


def _fmt_ci(ci: Any) -> str:
    return "" if not ci else f" [{ci[0]:.3g}, {ci[1]:.3g}]"


def _fmt(v: Any) -> str:
    return "n/a" if v is None else (f"{v:.3g}" if isinstance(v, float) else str(v))


def to_markdown(summary: dict[str, Any], meta: dict[str, Any]) -> str:
    L = [f"# ML benchmark v2: {meta.get('arm')} ({meta.get('split')} split)", "",
         f"Protocol {summary['protocol']}; spec `{meta.get('spec')}`; config hash `{meta.get('config_hash')}`; "
         f"commit `{(meta.get('git') or meta.get('guard') or {}).get('commit', '')[:10]}`; "
         f"started {meta.get('started')}, wall {meta.get('wall_seconds')} s with {meta.get('workers')} workers.", "",
         "Regret is normalized: (f* - f(best measured recipe)) / (f* - median f); 0 = optimum, 1 = a typical recipe. "
         "Coverage CIs are instance-cluster bootstraps; rate CIs are Wilson. Synthetic and emulator results are "
         "simulations; pool replay is retrospective. Report-only groups are marked.", ""]
    L += ["## Closed loop, emulator and pool replay (per task)", "",
          "| suite | task | group | n | regret final [95% CI] | by n0 (12/20/40) | cov@prop 90 / 95 | cov held-out 90 / 95 | width@prop 90 | trusted (rounds) | median s/analysis |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for t in summary["tasks"].values():
        if t["suite"] not in ("closed_loop", "emulator", "pool"):
            continue
        byn = "/".join(_fmt((t["regret_obs_final_by_n0"].get(str(n)) or {}).get("mean")) for n in (12, 20, 40))
        L.append(f"| {t['suite']} | {t['task']} | {t['group']}{'' if t['decides'] else ' (report-only)'} | "
                 f"{t['n_instances']} | {_fmt(t['regret_obs_final']['mean'])}{_fmt_ci(t['regret_obs_final']['ci95'])} | {byn} | "
                 f"{_fmt(t['at_proposals_0.9']['coverage'])} / {_fmt(t['at_proposals_0.95']['coverage'])} | "
                 f"{_fmt(t['heldout_0.9']['coverage'])} / {_fmt(t['heldout_0.95']['coverage'])} | "
                 f"{_fmt(t['at_proposals_0.9']['median_width'])} | {_fmt(t['trusted_rate_rounds']['rate'])} | "
                 f"{_fmt(t['median_wall_s'])} |")
    L += ["", "## Pooled per suite and group", "",
          "| suite / group | n | regret final [95% CI] | cov@prop 90 [CI] | cov@prop 95 [CI] | cov held-out 90 / 95 | width@prop 90 |",
          "|---|---|---|---|---|---|---|"]
    for key, t in summary["pooled"].items():
        if "n_instances" not in t:
            continue
        L.append(f"| {key} | {t['n_instances']} | {_fmt(t['regret_obs_final']['mean'])}{_fmt_ci(t['regret_obs_final']['ci95'])} | "
                 f"{_fmt(t['at_proposals_0.9']['coverage'])}{_fmt_ci(t['at_proposals_0.9']['ci95'])} | "
                 f"{_fmt(t['at_proposals_0.95']['coverage'])}{_fmt_ci(t['at_proposals_0.95']['ci95'])} | "
                 f"{_fmt(t['heldout_0.9']['coverage'])} / {_fmt(t['heldout_0.95']['coverage'])} | "
                 f"{_fmt(t['at_proposals_0.9']['median_width'])} |")
    ft = summary["pooled"].get("false_trust_all_nulls", {})
    L += ["", f"False trust over all null snapshots: {_fmt(ft.get('rate'))}{_fmt_ci(ft.get('ci95'))} (n={ft.get('n')}).", "",
          "## Snapshots (trust and calibration by data size)", "",
          "| suite | task | signal | n | trusted [Wilson CI] | cov@prop 90 | cov held-out 90 / 95 | width held-out 90 | prospective Spearman |",
          "|---|---|---|---|---|---|---|---|---|"]
    for t in summary["tasks"].values():
        if t["suite"] not in ("snapshot", "snapshot_real"):
            continue
        for nk, b in t.items():
            if not (isinstance(b, dict) and nk.startswith("n") and nk[1:].isdigit()):
                continue
            L.append(f"| {t['suite']} | {t['task']} | {t['all']['signal']} | {nk[1:]} | {_fmt(b['trusted_rate'])}{_fmt_ci(b['trusted_ci95'])} | "
                     f"{_fmt(b['at_proposals_0.9']['coverage'])} | {_fmt(b['heldout_0.9']['coverage'])} / {_fmt(b['heldout_0.95']['coverage'])} | "
                     f"{_fmt(b['heldout_0.9']['median_width'])} | {_fmt(b['prospective_spearman_median'])} |")
    L += ["", "## Retrospective backtest (pool-restricted; never a wet-lab win)", "",
          "| dataset | real order | cuts | M1 [CI] | M1 - random [CI] | M1 - OFAT | M1 - lab | M3 | M4 | trusted |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for t in summary["tasks"].values():
        if t["suite"] != "backtest":
            continue
        L.append(f"| {t['task']} | {t['real_order']} | {t['n_cuts']} | {_fmt(t['M1']['mean'])}{_fmt_ci(t['M1']['ci95'])} | "
                 f"{_fmt(t['M1_minus_random_mean']['mean'])}{_fmt_ci(t['M1_minus_random_mean']['ci95'])} {t['M1_minus_random_mean']['verdict'] or ''} | "
                 f"{_fmt(t['M1_minus_ofat']['mean'])} | {_fmt(t['M1_minus_lab']['mean'])} | {_fmt(t['M3']['mean'])} | "
                 f"{_fmt(t['M4']['mean'])} | {_fmt(t['trusted_rate'])} |")
    L += ["", "## Scale transfer, clone ranking (internal and fixture)", ""]
    for t in summary["tasks"].values():
        if t["suite"] not in ("scale", "internal"):
            continue
        keep = {k: (v.get("mean") if isinstance(v, dict) and "mean" in v else v) for k, v in t.items()
                if k not in ("suite", "task", "decides")}
        L.append(f"- **{t['suite']} / {t['task']}** ({t.get('label') or t['group']}): "
                 + ", ".join(f"{k}={_fmt(v) if not isinstance(v, dict) else json.dumps(v)}" for k, v in keep.items()))
    return "\n".join(L) + "\n"


def summarize_dir(d: Path) -> dict[str, Any]:
    d = Path(d)
    meta = json.loads((d / "run_meta.json").read_text(encoding="utf-8")) if (d / "run_meta.json").is_file() else {}
    summary = summarize(load_records(d))
    summary["meta"] = meta
    (d / "summary.json").write_text(json.dumps(summary, indent=1, default=str) + "\n", encoding="utf-8")
    (d / "summary.md").write_text(to_markdown(summary, meta), encoding="utf-8")
    print(f"wrote {d / 'summary.json'} and {d / 'summary.md'}")
    return summary


# --------------------------------------------------------------- compare
def _pair(cand: list[dict[str, Any]], ref: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    idx = {(r["suite"], r["task"], r["seed"]): r for r in ref}
    return [(c, idx[(c["suite"], c["task"], c["seed"])]) for c in cand if (c["suite"], c["task"], c["seed"]) in idx]


def _final(rec: dict[str, Any]) -> float | None:
    return (rec.get("rounds") or [{}])[-1].get("regret_obs")


def compare_records(cand: list[dict[str, Any]], ref: list[dict[str, Any]],
                    faults_problems: list[str] | None) -> dict[str, Any]:
    pairs = _pair(cand, ref)
    out: dict[str, Any] = {"protocol": PROTOCOL_VERSION, "n_pairs": len(pairs), "criteria": {}}
    loops = [(c, r) for c, r in pairs if c["suite"] in ("closed_loop", "emulator", "pool")]

    def diffs(sel: list[tuple[dict[str, Any], dict[str, Any]]]) -> list[float]:
        out_: list[float] = []
        for c, r in sel:
            a, b = _final(c), _final(r)
            if a is not None and b is not None:
                out_.append(a - b)
        return out_

    syn = [(c, r) for c, r in loops if c["suite"] == "closed_loop"]
    real = [(c, r) for c, r in loops if c["suite"] == "pool" and c.get("group") == "real_pool"]
    d_syn, d_real = diffs(syn), diffs(real)
    ci_syn = boot_mean_ci(d_syn)
    c1a = bool(ci_syn and ci_syn[1] < 0 and (not d_real or np.mean(d_real) <= 0))
    # width ratio at proposals (90%), synthetic closed loop, bootstrap over instances
    wc = [_cov_clusters([c], 0.9, "at_proposals")[1] for c, _ in syn]
    wr = [_cov_clusters([r], 0.9, "at_proposals")[1] for _, r in syn]
    ratio = ratio_ci = None
    if any(wc) and any(wr):
        ratio = float(np.median(np.concatenate([w for w in wc if w])) / np.median(np.concatenate([w for w in wr if w])))
        rng = np.random.default_rng(BOOT_SEED)
        bs = []
        for _ in range(1000):
            i = rng.integers(0, len(syn), len(syn))
            a = [w for k in i for w in wc[k]]
            b = [w for k in i for w in wr[k]]
            if a and b:
                bs.append(np.median(a) / np.median(b))
        ratio_ci = [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]
    c1b = bool(ratio_ci and ratio_ci[1] < CRITERIA["width_ratio_max"])
    out["criteria"]["C1a_regret"] = {"pass": c1a, "mean_diff_synthetic": _r(np.mean(d_syn)) if d_syn else None,
                                     "ci95": ci_syn, "n": len(d_syn),
                                     "mean_diff_real_pool": _r(np.mean(d_real)) if d_real else None, "n_real": len(d_real)}
    out["criteria"]["C1b_width"] = {"pass": c1b, "ratio": _r(ratio), "ci95": ratio_ci}
    # C2 coverage at proposals (synthetic closed loop + emulator): candidate
    cov_recs = [c for c, _ in loops if c["suite"] in ("closed_loop", "emulator")]
    c2 = True
    c2_detail = {}
    for lv in LEVELS:
        cl, _ = _cov_clusters(cov_recs, lv, "at_proposals")
        hits, tot = sum(a for a, _ in cl), sum(b for _, b in cl)
        pooled = hits / tot if tot else None
        worst = []
        cov_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for c in cov_recs:
            cov_by_task[f"{c['suite']}|{c['task']}"].append(c)
        for k, rs in cov_by_task.items():
            ci = cluster_rate_ci(_cov_clusters(rs, lv, "at_proposals")[0])
            if ci and ci[1] < lv - CRITERIA["coverage_tolerance"]:
                worst.append(k)
        ok = pooled is not None and pooled >= lv - CRITERIA["coverage_tolerance"] and not worst
        c2 &= ok
        c2_detail[str(lv)] = {"pooled": _r(pooled), "n": tot, "tasks_ci_below_nominal": worst, "pass": ok}
    out["criteria"]["C2_coverage_at_proposals"] = {"pass": c2, **c2_detail}
    # C3 held-out coverage guard
    c3 = True
    c3d = {}
    for lv in LEVELS:
        recs = cov_recs + [c for c, _ in pairs if c["suite"] == "snapshot" and c.get("signal")]
        cl, _ = _cov_clusters(recs, lv, "heldout")
        hits, tot = sum(a for a, _ in cl), sum(b for _, b in cl)
        pooled = hits / tot if tot else None
        ok = pooled is None or pooled >= lv - CRITERIA["heldout_coverage_slack"]
        c3 &= ok
        c3d[str(lv)] = {"pooled": _r(pooled), "n": tot, "pass": ok}
    out["criteria"]["C3_heldout_coverage"] = {"pass": c3, **c3d}
    # C4 false trust
    nulls = [(c, r) for c, r in pairs if c["suite"] in ("snapshot", "snapshot_real") and c.get("signal") is False
             and c.get("group") in (*DECISION_GROUPS, "synthetic")]
    rc = np.mean([bool(c.get("trusted")) for c, _ in nulls]) if nulls else 0.0
    rr = np.mean([bool(r.get("trusted")) for _, r in nulls]) if nulls else 0.0
    per: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
    for c, r in nulls:
        per[c["task"]].append((bool(c.get("trusted")), bool(r.get("trusted"))))
    worse = {k: [_r(np.mean([a for a, _ in v])), _r(np.mean([b for _, b in v]))] for k, v in per.items()
             if np.mean([a for a, _ in v]) > np.mean([b for _, b in v]) + CRITERIA["false_trust_slack_per_scenario"]}
    out["criteria"]["C4_false_trust"] = {"pass": bool(rc <= rr and not worse), "candidate": _r(rc), "reference": _r(rr),
                                         "n": len(nulls), "scenarios_worse": worse}
    # C5 planted faults
    out["criteria"]["C5_planted_faults"] = {"pass": faults_problems is not None and not faults_problems,
                                            "regressions": faults_problems,
                                            "note": None if faults_problems is not None else "fault scorecards not supplied"}
    # C6 no hidden loss (per family / dataset), Holm over all
    by_task: dict[str, list[float]] = defaultdict(list)
    for c, r in loops:
        fc, fr = _final(c), _final(r)
        if c.get("group") in DECISION_GROUPS and fc is not None and fr is not None:
            by_task[f"{c['suite']}|{c['task']}"].append(fc - fr)
    pv = {}
    for k, v in by_task.items():
        nz = [x for x in v if abs(x) > 1e-12]
        pv[k] = float(wilcoxon(nz).pvalue) if len(nz) >= 6 else 1.0
    adj = holm(pv) if pv else {}
    losses = {k: {"mean_diff": _r(np.mean(by_task[k])), "p_holm": _r(adj[k])} for k in adj
              if adj[k] < ALPHA and np.mean(by_task[k]) > CRITERIA["loss_margin_regret"]}
    out["criteria"]["C6_no_hidden_loss"] = {"pass": not losses, "losses": losses,
                                            "per_task": {k: {"mean_diff": _r(np.mean(v)), "n": len(v),
                                                             "p_holm": _r(adj.get(k))} for k, v in by_task.items()}}
    # C7 non-inferiority on the other axis
    c7 = True
    if c1a and not c1b and ratio_ci:
        c7 = ratio_ci[0] <= CRITERIA["width_ratio_noninferior"]
    if c1b and not c1a and ci_syn:
        c7 = ci_syn[0] <= 0
    out["criteria"]["C7_noninferior"] = {"pass": bool(c7)}
    # C8 backtest: no loss vs the reference on pooled real backtests (M1)
    bt = [(c, r) for c, r in pairs if c["suite"] == "backtest" and c.get("group") == "real_backtest"]
    d_bt = [cc["M1"] - rc_["M1"] for c, r in bt for cc, rc_ in zip(c.get("cuts", []), r.get("cuts", []))
            if cc.get("M1") is not None and rc_.get("M1") is not None]
    ci_bt = boot_mean_ci(d_bt)
    out["criteria"]["C8_backtest_no_loss"] = {"pass": not (ci_bt and ci_bt[0] > 0), "mean_diff": _r(np.mean(d_bt)) if d_bt else None,
                                              "ci95": ci_bt, "n": len(d_bt)}
    cr = out["criteria"]
    out["ships"] = bool((cr["C1a_regret"]["pass"] or cr["C1b_width"]["pass"]) and all(
        cr[k]["pass"] for k in ("C2_coverage_at_proposals", "C3_heldout_coverage", "C4_false_trust",
                                "C5_planted_faults", "C6_no_hidden_loss", "C7_noninferior", "C8_backtest_no_loss")))
    return out


def compare_dirs(cand: Path, ref: Path, baselines: Sequence[Path] = (), faults_cand: Path | None = None,
                 faults_ref: Path | None = None, out: Path | None = None) -> dict[str, Any]:
    cm = json.loads((Path(cand) / "run_meta.json").read_text(encoding="utf-8"))
    rm = json.loads((Path(ref) / "run_meta.json").read_text(encoding="utf-8"))
    if cm.get("split") != rm.get("split"):
        raise SystemExit(f"split mismatch: {cm.get('split')} vs {rm.get('split')}")
    problems = None
    if faults_cand and faults_ref:
        from kalos.bench.faults.score import gate_view, regressions

        problems = regressions(gate_view(json.loads(Path(faults_ref).read_text())),
                               gate_view(json.loads(Path(faults_cand).read_text())))
    res = compare_records(load_records(Path(cand)), load_records(Path(ref)), problems)
    res.update({"candidate": cm.get("arm"), "reference": rm.get("arm"), "split": cm.get("split"),
                "shippable_arm": cm.get("shippable"),
                "decision_valid": cm.get("split") == "heldout" and bool(cm.get("shippable"))})
    if not res["decision_valid"]:
        res["ships"] = False
        res["why_not_decisive"] = "only a held-out run of a shippable (engine-patch) arm can ship"
    res["baselines"] = {}
    for b in baselines:
        bm = json.loads((Path(b) / "run_meta.json").read_text(encoding="utf-8"))
        res["baselines"][bm.get("arm")] = compare_records(load_records(Path(b)), load_records(Path(ref)), None)["criteria"]["C1a_regret"]
    target = Path(out) if out else Path(cand) / f"compare_vs_{rm.get('arm')}.json"
    target.write_text(json.dumps(res, indent=1, default=str) + "\n", encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("candidate", "reference", "split", "ships")}),
          {k: v["pass"] for k, v in res["criteria"].items()})
    return res
