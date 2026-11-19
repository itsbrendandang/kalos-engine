"""The blind runner: put each case folder through the engine, record signals.

It reads only `cases/` (never `evaluator/`, never the catalogue) and writes one
`observations/<tier>/case-NN.json` per case. PROTOCOL.md section 4 says what
each tier asks the engine:

- fast (torch-free): the upload parser's orientation pre-pass, the pre-fit
  refusals (`_analyze(preflight_only=True)`), the validation gate as the analysis
  path calls it, and result entry through `CampaignStore` plus the validation of
  the folded sheet. Scale cases are not run.
- full: the upload pipeline (`compute_upload_analysis`), the same result-entry
  path, and the scale readout (`build_readout`).

A refusal (`UploadRejected`) is an observation, not a failure. Any other
exception is recorded as `status: error` with its type name only.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
import traceback
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd

from . import signals as sig
from .cases import CASES_DIR

OBS_DIR = "observations"
TENANT = "faults"
Observation = dict[str, Any]


def _case_meta(folder: Path) -> dict[str, Any]:
    return dict(json.loads((folder / "case.json").read_text(encoding="utf-8")))


def _validation_signals(df: pd.DataFrame, target: str | None) -> list[sig.Signal]:
    """The validation gate exactly as `_analyze_body` calls it (default profile,
    the mode from the environment, the requested target, after the all-blank
    column drop and column de-duplication)."""
    from kalos.domains import BIOPROCESS_PROFILE
    from kalos.portal.analysis_parts.columns import _dedupe_columns
    from kalos.portal.analysis_parts.policy import validation_mode
    from kalos.validation import report_dict, validate_frame

    frame = _dedupe_columns(df.dropna(axis=1, how="all"))
    report = validate_frame(frame, target=target, profile=BIOPROCESS_PROFILE, mode=validation_mode())
    return sig.from_validation(report_dict(report))


def _sheet_fast(folder: Path, meta: dict[str, Any]) -> Observation:
    from kalos.portal.analysis import _analyze
    from kalos.portal.uploads import UploadRejected, parse_upload

    raw = (folder / meta["filename"]).read_bytes()
    target = meta.get("target")
    try:
        parsed = parse_upload(raw, filename=meta["filename"])
    except UploadRejected as rej:
        return {"status": "refused", "signals": [sig.from_refusal(rej.code, rej.message)]}
    signals = sig.from_orientation(parsed.df.attrs.get("kalos_orientation"))
    try:
        _analyze(parsed.df, target, preflight_only=True)
    except UploadRejected as rej:
        return {"status": "refused", "signals": [*signals, sig.from_refusal(rej.code, rej.message)]}
    signals += _validation_signals(parsed.df, target)
    return {"status": "ok", "signals": signals}


def _sheet_full(folder: Path, meta: dict[str, Any]) -> Observation:
    from kalos.portal.pipeline import compute_upload_analysis
    from kalos.portal.uploads import UploadRejected

    raw = (folder / meta["filename"]).read_bytes()
    try:
        df, result = compute_upload_analysis(raw, meta.get("target"), False, meta["filename"])
    except UploadRejected as rej:
        return {"status": "refused", "signals": [sig.from_refusal(rej.code, rej.message)]}
    rel = result.get("reliability") or {}
    summary = {
        "n": result.get("n"), "d": result.get("d"), "target": result.get("target"),
        "cv_spearman": result.get("cv_spearman"), "cv_ci95": result.get("cv_ci95"),
        "clears_floor": rel.get("clears_floor"), "group_col": result.get("group_col"),
        "features": result.get("features"), "categorical_features": result.get("categorical_features"),
        "n_proposals": len(result.get("proposals") or []),
        "promotion_passed": (result.get("promotion") or {}).get("passed"),
        "proposal_recipes": [p.get("recipe") for p in (result.get("proposals") or [])][:5],
    }
    return {"status": "ok", "signals": sig.from_analysis(result, df), "summary": summary}


def _result_case(folder: Path, meta: dict[str, Any]) -> Observation:
    """Seed a throwaway campaign with the sheet, start one run per entry, record
    each entry with `set_result`, then validate the folded sheet."""
    from kalos.portal.campaign import CampaignError, CampaignStore
    from kalos.portal.uploads import parse_upload

    raw = (folder / meta["filename"]).read_bytes()
    entries = json.loads((folder / meta["entries"]).read_text(encoding="utf-8"))
    target = meta["target"]
    df = parse_upload(raw, filename=meta["filename"]).df
    n_base = len(df)
    signals: list[sig.Signal] = []
    with tempfile.TemporaryDirectory(prefix="kalos-faults-") as tmp:
        store = CampaignStore(Path(tmp))
        try:
            features = [c for c in df.columns if c != target]
            store.seed(df, target, features, tenant=TENANT)
            runs = store.start([{"recipe": e["recipe"]} for e in entries], tenant=TENANT)
            for k, (run, entry) in enumerate(zip(runs, entries)):
                try:
                    out = store.set_result(run["id"], float(entry["value"]), tenant=TENANT)
                except CampaignError as exc:
                    signals.append(sig.signal("result_entry", "rejected", "error",
                                              rows=[n_base + k], message=str(exc)))
                    continue
                verdict = out.get("verdict")
                if isinstance(verdict, dict):
                    verdict = verdict.get("verdict")
                if verdict and verdict != "valid":
                    signals.append(sig.signal("result_entry", str(verdict), "warning",
                                              rows=[n_base + k], message=str(out.get("reason", ""))))
                if out.get("held"):
                    signals.append(sig.signal("result_entry", "needs_review", "warning", rows=[n_base + k]))
            folded, fold_target, _gen = store.plan_fold(tenant=TENANT)
        finally:
            store.close()
    signals += _validation_signals(folded, fold_target)
    return {"status": "ok", "signals": signals, "summary": {"n_base": n_base, "n_fold": len(folded)}}


def _scale_case(folder: Path, meta: dict[str, Any]) -> Observation:
    from kalos.portal.uploads import UploadRejected, parse_upload
    from kalos.scale.readout import TargetSpec, build_readout

    raw = (folder / meta["filename"]).read_bytes()
    req = json.loads((folder / meta["request"]).read_text(encoding="utf-8"))
    try:
        df = parse_upload(raw, filename=meta["filename"]).df
    except UploadRejected as rej:
        return {"status": "refused", "signals": [sig.from_refusal(rej.code, rej.message)]}
    t = req["target"]
    spec = TargetSpec(
        scale_L=float(t["scale_L"]), process_params=dict(t["process_params"]),
        agitation_rpm=t.get("agitation_rpm"), airflow_L_per_min=t.get("airflow_L_per_min"),
    )
    try:
        readout = build_readout(df, req["target_column"], req["process_columns"], spec, raw_bytes=raw)
    except ValueError as exc:  # the route's `invalid_target` refusal
        return {"status": "refused",
                "signals": [sig.signal("scale_readout", "refusal", "error", message=str(exc))]}
    summary = {k: readout.get(k) for k in ("decision", "reasons", "prediction", "interval",
                                           "requested_ratio", "reference_ratio", "max_trained_scale_L")}
    return {"status": "ok", "signals": sig.from_readout(readout), "summary": summary}


def observe_case(folder: Path, tier: str) -> Observation:
    """Run one case folder through the engine; never raises."""
    folder = Path(folder)
    meta = _case_meta(folder)
    surface = meta["surface"]
    obs: Observation = {"case": folder.name, "tier": tier, "surface": surface}
    fn: Callable[[Path, dict[str, Any]], Observation] | None
    if surface == "sheet":
        fn = _sheet_fast if tier == "fast" else _sheet_full
    elif surface == "result":
        fn = _result_case
    elif surface == "scale":
        fn = None if tier == "fast" else _scale_case
    else:
        fn = None
    if fn is None:
        return {**obs, "status": "not_run", "signals": [], "seconds": 0.0}
    t0 = time.perf_counter()
    try:
        body = fn(folder, meta)
    except Exception as exc:  # deliberate: a crash is an observation (scored as missed)
        tb = traceback.extract_tb(exc.__traceback__)
        where = f"{Path(tb[-1].filename).name}:{tb[-1].lineno}" if tb else ""
        body = {"status": "error", "signals": [], "error": f"{type(exc).__name__} at {where}"}
    return {**obs, **body, "seconds": round(time.perf_counter() - t0, 2)}


def _observe_to_file(args: tuple[str, str, str]) -> str:
    folder, tier, out_file = args
    obs = observe_case(Path(folder), tier)
    Path(out_file).write_text(json.dumps(obs, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return out_file


def run_suite(
    out: Path, tier: str, *, workers: int = 1, only: list[str] | None = None,
    log: Callable[[str], None] | None = None,
) -> list[Observation]:
    """Observe every case under `out/cases` and write `out/observations/<tier>/`."""
    if tier not in ("fast", "full"):
        raise ValueError("tier must be 'fast' or 'full'")
    out = Path(out)
    folders = sorted(p for p in (out / CASES_DIR).iterdir() if p.is_dir())
    if only:
        folders = [p for p in folders if p.name in set(only)]
    obs_dir = out / OBS_DIR / tier
    obs_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(str(p), tier, str(obs_dir / f"{p.name}.json")) for p in folders]
    say = log or (lambda _msg: None)
    if workers <= 1:
        for job in jobs:
            say(f"{Path(job[0]).name} ...")
            _observe_to_file(job)
    else:
        # Each worker is a fresh interpreter; give each a share of the cores.
        threads = max(1, (os.cpu_count() or 2) // workers)
        os.environ.setdefault("KALOS_TORCH_THREADS", str(threads))
        import multiprocessing as mp

        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
            for done in pool.map(_observe_to_file, jobs):
                say(f"{Path(done).stem} done")
    return [json.loads(Path(j[2]).read_text(encoding="utf-8")) for j in jobs]


def load_observations(out: Path, tier: str) -> dict[str, Observation]:
    obs_dir = Path(out) / OBS_DIR / tier
    if not obs_dir.is_dir():
        return {}
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(obs_dir.glob("case-*.json"))}


__all__ = ["OBS_DIR", "load_observations", "observe_case", "run_suite"]
