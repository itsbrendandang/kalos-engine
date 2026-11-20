"""Per-PR scoreboard: one table of the numbers a change must not make worse.

It measures nothing new. Every number comes from an existing, pre-registered
benchmark in this repo, run on a small fixed CI slice:

  faults     planted-fault suite (kalos.bench.faults, PROTOCOL v1), fast tier:
             caught / noticed / missed per fault and false alarms on clean
             cases, with Wilson 95% CIs. The full tier (model-family faults,
             ~40 analyses) is too slow for every PR; run it with --tier full.
  coverage   coverage of the shipped 90/95% proposal intervals AT the
             proposals, and median band width (kalos.bench.coverage_at_proposals),
             on CI seeds 500.. (disjoint from the method-selection seeds 0-39
             and the reported seeds 100-139), seed-cluster bootstrap CI.
  regret     normalised simple regret of the Kalos arm on the fast synthetic
             tasks of the pre-registered head-to-head (kalos.bench.vs_baselines,
             docs/BENCHMARK.md), smoke seeds only (never the confirmatory seeds).
             It will switch to the ML benchmark v2 CI split once that lands.
  walltime   wall time of one full analysis of the 160-row example
             (examples/synthetic_bioprocess) through the upload pipeline, with
             the engine's default thread settings.
  golden     drift of the full `_analyze` output on the eight trust-golden
             sheets (scripts/trust_golden.py): per-sheet digest and changed
             leaf paths against the base.

Usage (from a checkout; `--repo` points at the code to MEASURE, so the head's
scoreboard can measure the base branch too):

    python -m kalos.bench.scoreboard run --out /tmp/sb-head
    python kalos/bench/scoreboard.py run --repo ../base --out /tmp/sb-base
    python -m kalos.bench.scoreboard diff /tmp/sb-base /tmp/sb-head --md diff.md
    make scoreboard          # run on this checkout, write SCOREBOARD.md
    make scoreboard-diff     # also run origin/main in a temp worktree and diff

`diff` exits 1 when a metric regresses beyond its noise band (section NOISE in
SCOREBOARD.md) or the golden outputs drift; a PR that means to do either states
and justifies it (CI label `scoreboard-justified`).

Same-machine claim only: base and head must be measured on the same kind of
machine with the same locked stack; the diff refuses to judge golden drift and
wall time when the environments differ.

This file deliberately imports nothing from `kalos` at module level, so the
head's copy can drive an older or newer checkout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA = 1
SECTIONS: tuple[str, ...] = ("walltime", "faults", "golden", "coverage", "regret")

# ---- the fixed CI slice (changing any of these is a scoreboard amendment) -----------
COVERAGE_SCENARIOS: tuple[str, ...] = ("hartmann6_n30", "heteroscedastic_n40",
                                       "bioprocess_n40", "branin_n20_control")
COVERAGE_SEEDS: tuple[int, ...] = (500, 501, 502, 503, 504)
COVERAGE_LEVELS: tuple[str, ...] = ("0.9", "0.95")
REGRET_TASKS: tuple[tuple[str, float], ...] = (("gaussian_4d", 0.15), ("bioprocess_like_mixed", 0.15))
REGRET_SEEDS: tuple[int, ...] = (9002, 9003, 9004, 9005)
WALLTIME_SHEET = "examples/synthetic_bioprocess/synthetic_bioprocess.csv"
WALLTIME_TARGET = "titer_g_L"

# ---- noise bands (see SCOREBOARD.md "Noise bands") ---------------------------------
COVERAGE_ABS_TOL = 0.02     # coverage drop smaller than this is noise
WIDTH_REL_TOL = 0.10        # band width growth smaller than 10% is noise
REGRET_ABS_TOL = 0.01       # docs/BENCHMARK.md tie tolerance on normalised regret
WALLTIME_REL_TOL = 0.25     # CI runners vary; slower by < 25% is noise
Z95 = 1.959963984540054

_OUTCOME_RANK = {"caught": 3, "noticed": 2, "noticed (blind spot)": 2, "missed": 1,
                 "crashed": 0, "error": 0}


# ======================================================================== helpers
def wilson(k: int, n: int) -> list[float] | None:
    if n <= 0:
        return None
    p = k / n
    den = 1 + Z95 ** 2 / n
    c = (p + Z95 ** 2 / (2 * n)) / den
    h = Z95 * math.sqrt(p * (1 - p) / n + Z95 ** 2 / (4 * n * n)) / den
    return [round(max(0.0, c - h), 3), round(min(1.0, c + h), 3)]


def _cluster_ci(per_cluster: Sequence[Sequence[bool]], n_boot: int = 2000) -> list[float] | None:
    import numpy as np

    cl = [c for c in per_cluster if len(c)]
    if not cl:
        return None
    rng = np.random.default_rng(0)
    hits = np.array([sum(c) for c in cl], float)
    tot = np.array([len(c) for c in cl], float)
    idx = rng.integers(0, len(cl), (n_boot, len(cl)))
    bs = hits[idx].sum(1) / tot[idx].sum(1)
    return [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]


def _boot_mean_ci(x: Sequence[float], n_boot: int = 4000) -> list[float] | None:
    import numpy as np

    a = np.asarray(x, float)
    if a.size == 0:
        return None
    if a.size == 1:
        return [round(float(a[0]), 4)] * 2
    rng = np.random.default_rng(1)
    m = a[rng.integers(0, a.size, (n_boot, a.size))].mean(1)
    return [round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4)]


def _git(repo: Path, *args: str) -> str | None:
    try:
        return subprocess.run(  # noqa: S603 - fixed git argv, no shell
            ["git", "-C", str(repo), *args], capture_output=True, text=True,  # noqa: S607
            check=True, timeout=10).stdout.strip() or None
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def _env() -> dict[str, Any]:
    out: dict[str, Any] = {"platform": platform.platform(), "machine": platform.machine(),
                           "python": platform.python_version(), "cpus": os.cpu_count()}
    for mod in ("torch", "botorch", "gpytorch", "numpy", "scipy", "pandas", "sklearn"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001 - recorded as missing
            out[mod] = None
    return out


def _canon_digest(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    if isinstance(obj, dict):
        if not obj:
            flat[prefix] = {}
        for k, v in obj.items():
            flat.update(_flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, list):
        if not obj:
            flat[prefix] = []
        for i, v in enumerate(obj):
            flat.update(_flatten(v, f"{prefix}[{i}]"))
    else:
        flat[prefix] = obj
    return flat


# ======================================================================= sections
# Each section runs in its own interpreter (see `run`), imports the measured repo's
# code, and returns a JSON-able dict. A missing benchmark is reported, not faked.

def section_walltime(repo: Path, workers: int) -> dict[str, Any]:
    from kalos.portal.pipeline import compute_upload_analysis

    path = repo / WALLTIME_SHEET
    raw = path.read_bytes()
    t0 = time.perf_counter()
    _, res = compute_upload_analysis(raw, WALLTIME_TARGET, False, path.name)
    secs = time.perf_counter() - t0
    n_rows = max(0, len(raw.decode("utf-8", "replace").strip().splitlines()) - 1)
    return {"sheet": WALLTIME_SHEET, "rows": n_rows, "seconds": round(secs, 2),
            "n_proposals": len(res.get("proposals") or []),
            "torch_threads_env": os.environ.get("KALOS_TORCH_THREADS"),
            "note": "one run, engine default threads (KALOS_TORCH_THREADS unset)"}


def section_faults(repo: Path, workers: int, tier: str = "fast") -> dict[str, Any]:
    from kalos.bench.faults.cases import build_suite
    from kalos.bench.faults.runner import run_suite
    from kalos.bench.faults.score import score_suite

    out = Path(tempfile.mkdtemp(prefix=f"sb-faults-{tier}-"))
    # Repo-only bases (the optional Kaggle sheet is never needed in CI).
    build_suite(out, datasets_dir=Path("/nonexistent"))
    run_suite(out, tier, workers=workers, only=None, log=lambda m: None)
    card = score_suite(out, tier)
    per_fault = {r["id"]: r["outcome"] for r in card.get("faults", [])}
    per_clean = {r["id"]: bool(r["false_alarm"]) for r in card.get("clean", [])}
    t = card["totals"]
    return {"tier": tier, "protocol_version": card.get("protocol_version"),
            "suite_seed": card.get("suite_seed"), "skipped": t.get("skipped", 0), "crashed": t.get("crashed", 0),
            "caught": t["caught"], "noticed": t["noticed"], "missed": t["missed"],
            "faults_scored": t["faults_scored"], "clean_false_alarm": t["clean_false_alarm"],
            "clean_scored": t["clean_scored"],
            "caught_ci": wilson(t["caught"], t["faults_scored"]),
            "false_alarm_ci": wilson(t["clean_false_alarm"], t["clean_scored"]),
            "per_fault": per_fault, "per_clean": per_clean}


def section_golden(repo: Path, workers: int) -> dict[str, Any]:
    import importlib.util

    spec = importlib.util.spec_from_file_location("trust_golden", repo / "scripts" / "trust_golden.py")
    if spec is None or spec.loader is None:
        return {"available": False, "reason": "scripts/trust_golden.py not found"}
    tg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tg)
    current = tg.compute()
    return {"available": True, "sheets": {k: _canon_digest(v) for k, v in current.items()},
            "outputs": current}


def section_coverage(repo: Path, workers: int) -> dict[str, Any]:
    from multiprocessing import get_context

    from kalos.bench import coverage_at_proposals as cap

    jobs = [(s, seed) for s in COVERAGE_SCENARIOS if s in cap.SCENARIOS for seed in COVERAGE_SEEDS]
    with get_context("spawn").Pool(max(1, workers)) as pool:
        rows = pool.map(cap._worker, jobs, chunksize=1)
    out: dict[str, Any] = {"scenarios": list(COVERAGE_SCENARIOS), "seeds": list(COVERAGE_SEEDS),
                           "errors": [r for r in rows if "error" in r], "levels": {}}
    ok = [r for r in rows if "error" not in r]
    for lv in COVERAGE_LEVELS:
        clusters = [r["levels"][lv]["inside"] for r in ok if lv in r["levels"]]
        flat = [b for c in clusters for b in c]
        widths = [w for r in ok if lv in r["levels"] for w in r["levels"][lv]["width"]]
        avail = [b for r in ok if lv in r["levels"] for b in r["levels"][lv]["available"]]
        per_sc = {}
        for s in COVERAGE_SCENARIOS:
            fs = [b for r in ok if r["scenario"] == s for b in r["levels"][lv]["inside"]]
            per_sc[s] = round(sum(fs) / len(fs), 3) if fs else None
        out["levels"][lv] = {
            "coverage_at_proposals": round(sum(flat) / len(flat), 3) if flat else None,
            "ci95_seed_cluster": _cluster_ci(clusters),
            "n_measured": len(flat),
            "availability": round(sum(avail) / len(avail), 3) if avail else None,
            "median_width": round(float(sorted(widths)[len(widths) // 2]), 3) if widths else None,
            "per_scenario": per_sc,
        }
    out["rows"] = rows
    return out


def section_regret(repo: Path, workers: int) -> dict[str, Any]:
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context

    from kalos.bench import vs_baselines as vb

    units = [(t, "kalos", eta, s) for t, eta in REGRET_TASKS for s in REGRET_SEEDS]
    with ProcessPoolExecutor(max_workers=max(1, workers), mp_context=get_context("spawn")) as ex:
        runs = list(ex.map(vb._worker, units, chunksize=1))
    per = [{"task": r["task"], "eta": r["eta"], "seed": r["seed"],
            "regret_norm": round(float(r["regret_norm"]), 5),
            "obs_regret_norm": round(float(r["obs_regret_norm"]), 5),
            "fallback_steps": r["fallback_steps"], "errors": r["errors"][:3]} for r in runs]
    tasks = {}
    for t, _ in REGRET_TASKS:
        v = [p["regret_norm"] for p in per if p["task"] == t]
        tasks[t] = {"mean": round(sum(v) / len(v), 4) if v else None, "ci95": _boot_mean_ci(v)}
    allv = [p["regret_norm"] for p in per]
    return {"source": "kalos.bench.vs_baselines (docs/BENCHMARK.md), smoke seeds",
            "seeds": list(REGRET_SEEDS), "tasks": tasks,
            "mean_regret_norm": round(sum(allv) / len(allv), 4) if allv else None,
            "ci95": _boot_mean_ci(allv),
            "fallback_steps": sum(p["fallback_steps"] for p in per), "per_unit": per}


SECTION_FUNCS: dict[str, Callable[..., dict[str, Any]]] = {
    "walltime": section_walltime, "faults": section_faults, "golden": section_golden,
    "coverage": section_coverage, "regret": section_regret,
}


def _run_section(name: str, repo: Path, out: Path, workers: int, tier: str) -> None:
    """Child-process entry: import the measured repo first, run one section."""
    sys.path.insert(0, str(repo))
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    os.chdir(repo)
    t0 = time.perf_counter()
    try:
        fn = SECTION_FUNCS[name]
        data = fn(repo, workers, tier) if name == "faults" else fn(repo, workers)
        data["status"] = "ok"
    except Exception as exc:  # noqa: BLE001 - reported, never silently dropped
        import traceback

        data = {"status": "error", "error": f"{type(exc).__name__}: {str(exc)[:400]}",
                "trace": traceback.format_exc()[-2000:]}
    data["section_seconds"] = round(time.perf_counter() - t0, 1)
    (out / f"section-{name}.json").write_text(json.dumps(data, default=str) + "\n", encoding="utf-8")


# ============================================================================ run
def run(repo: Path, out: Path, *, workers: int, sections: Iterable[str], tier: str,
        timeout: float) -> dict[str, Any]:
    repo = repo.resolve()
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    me = Path(__file__).resolve()
    for name in sections:
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([str(repo), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
        env.setdefault("KALOS_STATE_DIR", tempfile.mkdtemp(prefix="sb-state-"))
        if name == "walltime":
            # Measure what a user gets: the engine's own thread defaults.
            for k in ("KALOS_TORCH_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
                env.pop(k, None)
        else:
            env.setdefault("KALOS_TORCH_THREADS", "1")
            env.setdefault("OMP_NUM_THREADS", "1")
        print(f"[scoreboard] {name} ...", flush=True)
        t0 = time.perf_counter()
        try:
            subprocess.run(  # noqa: S603 - our own script, no shell
                [sys.executable, str(me), "_section", name, "--repo", str(repo),
                            "--out", str(out), "--workers", str(workers), "--tier", tier],
                           env=env, cwd=str(repo), timeout=timeout, check=False)
        except subprocess.TimeoutExpired:
            (out / f"section-{name}.json").write_text(json.dumps(
                {"status": "error", "error": f"timed out after {timeout:.0f}s"}) + "\n")
        print(f"[scoreboard] {name} done in {time.perf_counter() - t0:.0f}s", flush=True)
    board: dict[str, Any] = {"schema": SCHEMA, "commit": _git(repo, "rev-parse", "HEAD"),
                             "branch": _git(repo, "rev-parse", "--abbrev-ref", "HEAD"),
                             "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                             "env": _env(), "workers": workers, "faults_tier": tier, "sections": {}}
    for name in sections:
        p = out / f"section-{name}.json"
        board["sections"][name] = (json.loads(p.read_text()) if p.exists()
                                   else {"status": "error", "error": "section wrote nothing"})
    golden = board["sections"].get("golden", {})
    if golden.get("outputs") is not None:
        (out / "golden-outputs.json").write_text(json.dumps(golden.pop("outputs"), sort_keys=True))
    (out / "scoreboard.json").write_text(json.dumps(board, indent=1, sort_keys=True, default=str) + "\n")
    (out / "SCOREBOARD.md").write_text(render(board), encoding="utf-8")
    return board


# ========================================================================= render
def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def _ci(ci: Sequence[float] | None) -> str:
    return "" if not ci else f" [{ci[0]:.3f}, {ci[1]:.3f}]"


def _why(sec: Mapping[str, Any]) -> str:
    return f"error: {sec['error']}" if sec.get("error") else "not run"


def headline(board: Mapping[str, Any]) -> list[tuple[str, str]]:
    """(metric, value) rows; each value carries its CI where there is one."""
    s = board.get("sections", {})
    rows: list[tuple[str, str]] = []
    f = s.get("faults", {})
    if f.get("status") == "ok":
        rows.append((f"Planted faults caught ({f['tier']} tier)",
                     f"{f['caught']}/{f['faults_scored']}{_ci(f['caught_ci'])}"))
        rows.append(("Planted faults caught or noticed",
                     f"{f['caught'] + f['noticed']}/{f['faults_scored']}"))
        rows.append(("Clean cases with a false alarm",
                     f"{f['clean_false_alarm']}/{f['clean_scored']}{_ci(f['false_alarm_ci'])}"))
    else:
        rows.append(("Planted faults", _why(f)))
    c = s.get("coverage", {})
    if c.get("status") == "ok":
        for lv in COVERAGE_LEVELS:
            L = c["levels"].get(lv, {})
            rows.append((f"Coverage at proposals, nominal {float(lv):.0%}",
                         f"{_pct(L.get('coverage_at_proposals'))}{_ci(L.get('ci95_seed_cluster'))}"
                         f" (n={L.get('n_measured')})"))
            rows.append((f"Median band width at proposals, {float(lv):.0%} (fraction of y range)",
                         _pct(L.get("median_width"))))
    else:
        rows.append(("Coverage at proposals", _why(c)))
    r = s.get("regret", {})
    if r.get("status") == "ok":
        rows.append(("Normalised regret, fast synthetic tasks (lower is better)",
                     f"{r['mean_regret_norm']:.4f}{_ci(r['ci95'])}"))
        for t, v in r["tasks"].items():
            rows.append((f"  regret: {t}", f"{v['mean']:.4f}{_ci(v['ci95'])}"))
    else:
        rows.append(("Regret", _why(r)))
    w = s.get("walltime", {})
    rows.append(("Wall time, 160-row example (s)",
                 f"{w['seconds']:.1f}" if w.get("status") == "ok" else _why(w)))
    g = s.get("golden", {})
    rows.append(("Golden sheets (digest recorded)",
                 str(len(g.get("sheets", {}))) if g.get("status") == "ok" else _why(g)))
    return rows


def render(board: Mapping[str, Any]) -> str:
    env = board.get("env", {})
    lines = [
        "# Kalos scoreboard",
        "",
        f"Commit `{(board.get('commit') or '?')[:10]}` ({board.get('branch')}), {board.get('created_at')}. "
        f"Machine: {env.get('platform')}, {env.get('cpus')} CPUs, Python {env.get('python')}, "
        f"torch {env.get('torch')}, botorch {env.get('botorch')}.",
        "",
        "Generated by `make scoreboard` (kalos/bench/scoreboard.py). Every number comes from an "
        "existing benchmark on a fixed CI slice; all inputs are synthetic. Wall time and golden "
        "digests are same-machine claims only.",
        "",
        "| Metric | Value (95% CI) |",
        "|---|---|",
        *[f"| {k} | {v} |" for k, v in headline(board)],
        "",
        "## Sources and slices",
        "",
        f"- Planted faults: `kalos.bench.faults`, PROTOCOL v1, {board.get('faults_tier')} tier, repo-only bases; Wilson 95% CIs.",
        f"- Coverage: `kalos.bench.coverage_at_proposals`, scenarios {', '.join(COVERAGE_SCENARIOS)}, "
        f"seeds {COVERAGE_SEEDS[0]}-{COVERAGE_SEEDS[-1]} (disjoint from selection seeds 0-39 and reported seeds 100-139); "
        "seed-cluster bootstrap CI. A smoke slice: the reported figures are in VALIDATION.md.",
        f"- Regret: `kalos.bench.vs_baselines` Kalos arm, tasks {', '.join(t for t, _ in REGRET_TASKS)} at noise 0.15, "
        f"smoke seeds {', '.join(map(str, REGRET_SEEDS))} (never the confirmatory seeds); bootstrap CI over units.",
        f"- Wall time: `compute_upload_analysis` on `{WALLTIME_SHEET}`, one run, engine default threads.",
        "- Golden: `scripts/trust_golden.py` sheets, full `_analyze` output minus wall-clock fields, sha256 per sheet.",
        "",
        "## Noise bands (what `diff` calls a regression)",
        "",
        "- Planted faults: deterministic per case. Any fault whose outcome gets worse, or any clean case that starts "
        "to false-alarm, is a regression.",
        f"- Coverage: a drop of more than max({COVERAGE_ABS_TOL}, half the base CI width) is a regression when the head "
        f"is below nominal; median width growth above {WIDTH_REL_TOL:.0%} is a regression.",
        f"- Regret: a mean increase above max({REGRET_ABS_TOL}, half the base CI width). Four seeds per task: a smoke "
        "signal, not a test of small differences.",
        f"- Wall time: slower by more than {WALLTIME_REL_TOL:.0%} (CI runners vary).",
        "- Golden: any changed sheet is drift. It must be explained in the PR (behaviour changes) or absent (refactors, speedups).",
        "",
    ]
    for name in SECTIONS:
        sec = board.get("sections", {}).get(name)
        if sec and sec.get("status") != "ok":
            lines += [f"Section `{name}` failed: {sec.get('error')}", ""]
    return "\n".join(lines)


# =========================================================================== diff
def _num(x: Any) -> float | None:
    return float(x) if isinstance(x, (int, float)) else None


def diff(base: Mapping[str, Any], head: Mapping[str, Any],
         base_golden: Mapping[str, Any] | None = None,
         head_golden: Mapping[str, Any] | None = None) -> tuple[str, list[str], list[str]]:
    """Return (markdown, regressions, notes)."""
    regs: list[str] = []
    notes: list[str] = []
    bs, hs = base.get("sections", {}), head.get("sections", {})
    same_env = {k: v for k, v in base.get("env", {}).items() if k != "platform"} == \
               {k: v for k, v in head.get("env", {}).items() if k != "platform"}

    # faults
    bf, hf = bs.get("faults", {}), hs.get("faults", {})
    if bf.get("status") == "ok" and hf.get("status") == "ok":
        for fault, b_out in sorted(bf["per_fault"].items()):
            h_out = hf["per_fault"].get(fault)
            if h_out is None:
                notes.append(f"faults: `{fault}` not scored on head")
            elif _OUTCOME_RANK.get(h_out, 0) < _OUTCOME_RANK.get(b_out, 0):
                regs.append(f"faults: `{fault}` {b_out} -> {h_out}")
            elif _OUTCOME_RANK.get(h_out, 0) > _OUTCOME_RANK.get(b_out, 0):
                notes.append(f"faults: `{fault}` improved {b_out} -> {h_out}")
        for clean, b_fa in sorted(bf["per_clean"].items()):
            h_fa = hf["per_clean"].get(clean)
            if h_fa and not b_fa:
                regs.append(f"faults: clean `{clean}` now false-alarms")
            elif b_fa and h_fa is False:
                notes.append(f"faults: clean `{clean}` no longer false-alarms")
        for fault in sorted(set(hf["per_fault"]) - set(bf["per_fault"])):
            notes.append(f"faults: new case `{fault}` -> {hf['per_fault'][fault]}")
    elif hf.get("status") != "ok":
        regs.append(f"faults: head section failed ({hf.get('error')})")

    # coverage
    bc, hc = bs.get("coverage", {}), hs.get("coverage", {})
    if bc.get("status") == "ok" and hc.get("status") == "ok":
        for lv in COVERAGE_LEVELS:
            b, h = bc["levels"].get(lv, {}), hc["levels"].get(lv, {})
            bcv, hcv = _num(b.get("coverage_at_proposals")), _num(h.get("coverage_at_proposals"))
            if bcv is not None and hcv is not None:
                ci = b.get("ci95_seed_cluster") or [bcv, bcv]
                band = max(COVERAGE_ABS_TOL, (ci[1] - ci[0]) / 2)
                if hcv < float(lv) and bcv - hcv > band:
                    regs.append(f"coverage {float(lv):.0%}: {bcv:.3f} -> {hcv:.3f} (band {band:.3f}, below nominal)")
            bw, hw = _num(b.get("median_width")), _num(h.get("median_width"))
            if bw and hw is not None and (hw - bw) / bw > WIDTH_REL_TOL:
                regs.append(f"band width {float(lv):.0%}: {bw:.3f} -> {hw:.3f} (+{(hw - bw) / bw:.0%})")
    elif hc.get("status") != "ok":
        regs.append(f"coverage: head section failed ({hc.get('error')})")

    # regret
    br, hr = bs.get("regret", {}), hs.get("regret", {})
    if br.get("status") == "ok" and hr.get("status") == "ok":
        b, h = _num(br.get("mean_regret_norm")), _num(hr.get("mean_regret_norm"))
        if b is not None and h is not None:
            ci = br.get("ci95") or [b, b]
            band = max(REGRET_ABS_TOL, (ci[1] - ci[0]) / 2)
            if h - b > band:
                regs.append(f"regret: {b:.4f} -> {h:.4f} (band {band:.4f})")
        if (hr.get("fallback_steps") or 0) > (br.get("fallback_steps") or 0):
            regs.append(f"regret: optimizer fallbacks {br.get('fallback_steps')} -> {hr.get('fallback_steps')}")
    elif hr.get("status") != "ok":
        regs.append(f"regret: head section failed ({hr.get('error')})")

    # wall time
    bw_, hw_ = bs.get("walltime", {}), hs.get("walltime", {})
    if bw_.get("status") == "ok" and hw_.get("status") == "ok":
        b, h = float(bw_["seconds"]), float(hw_["seconds"])
        if not same_env:
            notes.append("walltime: environments differ; not judged")
        elif b > 0 and (h - b) / b > WALLTIME_REL_TOL:
            regs.append(f"walltime: {b:.1f}s -> {h:.1f}s (+{(h - b) / b:.0%})")
    elif hw_.get("status") != "ok":
        regs.append(f"walltime: head section failed ({hw_.get('error')})")

    # golden
    bg, hg = bs.get("golden", {}), hs.get("golden", {})
    golden_lines: list[str] = []
    if bg.get("status") == "ok" and hg.get("status") == "ok":
        if not same_env:
            notes.append("golden: environments differ; drift not judged")
        else:
            for sheet in sorted(set(bg["sheets"]) | set(hg["sheets"])):
                if bg["sheets"].get(sheet) != hg["sheets"].get(sheet):
                    detail = ""
                    if base_golden is not None and head_golden is not None:
                        fb = _flatten(base_golden.get(sheet, {}))
                        fh = _flatten(head_golden.get(sheet, {}))
                        changed = sorted(k for k in set(fb) | set(fh) if fb.get(k, "<absent>") != fh.get(k, "<absent>"))
                        detail = f" ({len(changed)} leaf paths, e.g. {', '.join(changed[:4])})"
                        golden_lines += [f"- `{sheet}` `{k}`: {json.dumps(fb.get(k, '<absent>'), default=str)[:60]}"
                                         f" -> {json.dumps(fh.get(k, '<absent>'), default=str)[:60]}" for k in changed[:12]]
                    regs.append(f"golden drift: `{sheet}`{detail}")
    elif hg.get("status") != "ok":
        regs.append(f"golden: head section failed ({hg.get('error')})")

    hb, hh = dict(headline(base)), dict(headline(head))
    md = ["## Scoreboard: base vs head", "",
          f"Base `{(base.get('commit') or '?')[:10]}`, head `{(head.get('commit') or '?')[:10]}`."
          + ("" if same_env else " **Environments differ** (stack or CPU count): golden and wall time not judged."),
          "", "| Metric | Base | Head |", "|---|---|---|"]
    for k in dict.fromkeys([*hb, *hh]):
        mark = "" if hb.get(k) == hh.get(k) else " ◀"
        md.append(f"| {k} | {hb.get(k, 'n/a')} | {hh.get(k, 'n/a')}{mark} |")
    md += ["", f"**Verdict: {'REGRESSION or DRIFT (needs a stated reason)' if regs else 'no regression beyond the noise bands'}**", ""]
    if regs:
        md += ["Regressions / drift:", "", *[f"- {r}" for r in regs], ""]
    if notes:
        md += ["Notes:", "", *[f"- {n}" for n in notes], ""]
    if golden_lines:
        md += ["Golden changes (first paths):", "", *golden_lines, ""]
    return "\n".join(md), regs, notes


def _load_dir(d: Path) -> tuple[dict[str, Any], dict[str, Any] | None]:
    board = json.loads((d / "scoreboard.json").read_text())
    g = d / "golden-outputs.json"
    return board, (json.loads(g.read_text()) if g.exists() else None)


def merge(out: Path, parts: Sequence[Path]) -> dict[str, Any]:
    """Union of the sections of several `run` dirs of the SAME commit (CI runs the
    sections in parallel jobs); refuses to mix commits."""
    boards = [json.loads((p / "scoreboard.json").read_text()) for p in parts]
    commits = {b.get("commit") for b in boards}
    if len(commits) != 1:
        raise SystemExit(f"refusing to merge scoreboards of different commits: {sorted(map(str, commits))}")
    merged = dict(boards[0])
    merged["sections"] = {}
    merged["section_envs"] = {}
    for p, b in zip(parts, boards, strict=True):
        for name, sec in b["sections"].items():
            merged["sections"][name] = sec
            merged["section_envs"][name] = b.get("env")
        g = p / "golden-outputs.json"
        if g.exists():
            out.mkdir(parents=True, exist_ok=True)
            (out / "golden-outputs.json").write_text(g.read_text())
    out.mkdir(parents=True, exist_ok=True)
    (out / "scoreboard.json").write_text(json.dumps(merged, indent=1, sort_keys=True, default=str) + "\n")
    (out / "SCOREBOARD.md").write_text(render(merged), encoding="utf-8")
    return merged


# =========================================================================== main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="kalos.bench.scoreboard", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="measure a checkout")
    r.add_argument("--repo", type=Path, default=Path.cwd())
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    r.add_argument("--sections", default=",".join(SECTIONS))
    r.add_argument("--tier", choices=["fast", "full"], default="fast", help="planted-fault tier")
    r.add_argument("--timeout", type=float, default=3600, help="per-section timeout (s)")
    r.add_argument("--write", type=Path, default=None, help="also copy SCOREBOARD.md here")
    s = sub.add_parser("_section")
    s.add_argument("name", choices=SECTIONS)
    s.add_argument("--repo", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--workers", type=int, default=1)
    s.add_argument("--tier", default="fast")
    m = sub.add_parser("merge", help="merge `run` output dirs holding different sections into one")
    m.add_argument("out", type=Path)
    m.add_argument("parts", type=Path, nargs="+")
    d = sub.add_parser("diff", help="compare two `run` output dirs")
    d.add_argument("base", type=Path)
    d.add_argument("head", type=Path)
    d.add_argument("--md", type=Path, default=None)
    args = ap.parse_args(argv)

    if args.cmd == "_section":
        _run_section(args.name, args.repo.resolve(), args.out, args.workers, args.tier)
        return 0
    if args.cmd == "run":
        sections = [x for x in args.sections.split(",") if x]
        unknown = set(sections) - set(SECTIONS)
        if unknown:
            ap.error(f"unknown sections {sorted(unknown)}")
        board = run(args.repo, args.out, workers=args.workers, sections=sections,
                    tier=args.tier, timeout=args.timeout)
        text = (args.out / "SCOREBOARD.md").read_text()
        if args.write:
            args.write.write_text(text, encoding="utf-8")
        print(text)
        return 0 if all(v.get("status") == "ok" for v in board["sections"].values()) else 1
    if args.cmd == "merge":
        merged = merge(args.out, args.parts)
        print(f"merged {sorted(merged['sections'])} into {args.out}")
        return 0
    base, base_g = _load_dir(args.base)
    head, head_g = _load_dir(args.head)
    md, regs, _ = diff(base, head, base_g, head_g)
    if args.md:
        args.md.write_text(md + "\n", encoding="utf-8")
    print(md)
    return 1 if regs else 0


if __name__ == "__main__":
    raise SystemExit(main())
