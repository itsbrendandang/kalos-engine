"""Per-clone (strain) statistics: labels in, strict-JSON numbers out.

This module answers the questions a process scientist asks about production
clones (STRAIN_VIEWS.md sections 3.2-3.6) without fitting a model of its own:

  * `measured_summary`    - what each clone measured (section 3.2).
  * `eb_clone_effects`    - process-adjusted clone effects with shrinkage, from
                            out-of-fold residuals the caller supplies (3.3).
  * `cramers_v` / `confounding` - is the clone separable from round, thaw,
                            scale or medium lot (3.4)?
  * `process_balance`     - did the clones run at the same process conditions (3.4)?
  * `interaction_screen`  - does the response to an input differ by clone (3.5)?
  * `by_round_best`       - best measured value per clone per round.

Every function is pure (numpy + scipy only, torch-free; see
`tests/test_core_torch_contract.py`) and every returned value is strict-JSON
safe: non-finite numbers become `None`, floats are rounded to `SIG_DIGITS`
significant digits, and labels are `str`. Each statistic that cannot be
supported by the data is returned as `None` (or status `insufficient`) with a
reason, never as a confident-looking number. The refusal codes follow the table
in STRAIN_VIEWS.md section 3.6: `{code, reason, what_would_fix_it}`.

Blank labels (`None`, `""`, whitespace, NaN) drop the run from whatever the
label is needed for, like other categoricals in `kalos/portal/analysis.py`.
"""
from __future__ import annotations

import math
import zlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats

from .drivers import benjamini_hochberg
from .replicates import aggregate_replicates

# --- Thresholds (STRAIN_VIEWS.md 3.2-3.6). Conventions, not derived bounds. ---

MIN_RUNS_CI = 3
# Runs before a clone's mean gets a t-interval. With n=2 the t quantile is 12.7
# and the interval is a formality; below 3 we print n and no interval.
MIN_RUNS_EFFECT = 3
# Runs before a clone gets a process-adjusted (empirical Bayes) effect.
THIN_MIN_RUNS = 6
THIN_MIN_LEVELS = 3
THIN_MIN_RANGE_FRAC = 0.5
# Thin-data rule for a (clone, input) interaction cell: a within-clone slope on
# fewer runs, fewer distinct settings, or a narrow slice of the range is an
# extrapolation, not a measurement of how that clone responds.
INTERACTION_P = 0.05
INTERACTION_I2 = 0.5
# Flag `clone_input_interaction` only when the heterogeneity is both detectable
# (p) and material (I^2: at least half the spread in slopes is beyond noise).
SMD_FLAG = 0.8
COVERAGE_FLAG = 0.3
# Process-imbalance flags: a standardized mean difference above 0.8 is "large"
# by Cohen's convention; a clone that covered under 30% of an input's observed
# range was effectively run at one setting.
BOOT_B = 500
# Bootstrap resamples for interaction-slope percentile intervals.
LOCO_MIN_CLONES = 3
LOCO_MIN_RUNS = 5
# Leave-one-clone-out transfer is reported only with >= 3 clones of >= 5 runs.

SIG_DIGITS = 6
# Significant digits of every float this module returns.

EB_VAR_FLOOR_FRAC = 1e-3
# A level whose residuals are all identical has s_k = 0, so v_k = 0 and inverse-
# variance weight 1/v_k = inf. Floor s_k^2 at this fraction of the pooled
# within-level variance so such a level is weighted heavily but finitely.

Z95 = float(stats.norm.ppf(0.975))
EB_CI_METHOD = "hull(eb_posterior, welch_t_unshrunk)"

__all__ = [
    "BOOT_B",
    "COVERAGE_FLAG",
    "INTERACTION_I2",
    "INTERACTION_P",
    "LOCO_MIN_CLONES",
    "LOCO_MIN_RUNS",
    "MIN_RUNS_CI",
    "MIN_RUNS_EFFECT",
    "SMD_FLAG",
    "THIN_MIN_LEVELS",
    "THIN_MIN_RANGE_FRAC",
    "THIN_MIN_RUNS",
    "bh_adjusted",
    "by_round_best",
    "confounding",
    "cramers_v",
    "eb_clone_effects",
    "interaction_screen",
    "measured_summary",
    "process_balance",
    "t_interval",
]


# --------------------------------------------------------------------------- #
# JSON-safety helpers
# --------------------------------------------------------------------------- #


def _r(x: object, digits: int | None = SIG_DIGITS) -> float | None:
    """Finite float rounded to `digits` significant digits, else None.

    `digits=None` returns the unrounded float (used by tests that compare to a
    scipy reference at 1e-9).
    """
    if x is None:
        return None
    try:
        v = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    if digits is None or v == 0.0:
        return v + 0.0  # normalize -0.0
    return float(f"{v:.{digits}g}")


def _label(x: object) -> str | None:
    """A category label as `str`, or None when blank.

    Integral floats (a numeric clone ID read through a float column) render
    without a trailing `.0`, so `329.0` and `329` are the same clone.
    """
    if x is None:
        return None
    if isinstance(x, float | np.floating):
        v = float(x)
        if not math.isfinite(v):
            return None
        if v.is_integer():
            return str(int(v))
        return repr(v)
    if isinstance(x, np.integer):
        return str(int(x))
    s = str(x).strip()
    return s or None


def _labels(seq: Sequence[object] | ArrayLike, n: int | None = None, what: str = "labels") -> list[str | None]:
    out = [_label(v) for v in np.asarray(seq, dtype=object).reshape(-1).tolist()]
    if n is not None and len(out) != n:
        raise ValueError(f"{what} must have one entry per run ({len(out)} != {n})")
    return out


def _sorted_labels(labs: Sequence[str | None]) -> list[str]:
    return sorted({lab for lab in labs if lab is not None})


def _refusal(code: str, reason: str, fix: str, **extra: object) -> dict[str, object]:
    out: dict[str, object] = {"code": code}
    out.update(extra)
    out["reason"] = reason
    out["what_would_fix_it"] = fix
    return out


def _fmt(x: float) -> str:
    return f"{x:.4g}"


# --------------------------------------------------------------------------- #
# 1. Measured summary
# --------------------------------------------------------------------------- #


def t_interval(values: ArrayLike, coverage: float = 0.95) -> tuple[float, float] | None:
    """Unrounded two-sided t-interval for the mean, or None when n < MIN_RUNS_CI."""
    v = np.asarray(values, dtype=float).reshape(-1)
    v = v[np.isfinite(v)]
    n = v.size
    if n < MIN_RUNS_CI:
        return None
    m = float(v.mean())
    half = float(stats.t.ppf(0.5 + coverage / 2.0, n - 1)) * float(v.std(ddof=1)) / math.sqrt(n)
    return m - half, m + half


def measured_summary(
    y: ArrayLike,
    *,
    recipe_key: Sequence[object] | None = None,
    run_ids: Sequence[object] | None = None,
    maximize: bool = True,
    round_digits: int | None = SIG_DIGITS,
) -> dict[str, Any]:
    """Measured summary of ONE clone's runs for ONE outcome (basis 'measured').

    Honesty rules:
      * No model is involved: every number is a plain statistic of measured runs.
      * `ci95` is a t-interval for the mean ONLY when `n_runs >= MIN_RUNS_CI`
        (3); otherwise `None` with `ci_reason: 'n<3'`. A zero-sd sample gives
        the degenerate interval `[mean, mean]`, which is what the data say.
      * `sd` needs n >= 2, else `None`.
      * `replicate_sd` is the df-weighted pooled within-recipe SD, reported only
        when at least 2 recipes have >= 2 runs (one replicated recipe is one
        variance estimate, not an assay-noise measurement); otherwise `None`
        with `replicate_sd_reason`.
      * NaN outcomes are ignored and counted in `n_missing`, never imputed.

    `recipe_key` (one key per run, e.g. `kalos.core.splits.row_hash_groups`)
    defines recipes; without it `n_recipes`/`n_replicated_recipes` are None.
    `run_ids` (aligned to `y`) names the best run. `round_digits=None` returns
    unrounded floats (for tests against a scipy reference).

    Returns `{basis, n_runs, n_missing, n_recipes, n_replicated_recipes, mean,
    sd, median, min, max, best, best_run_id, ci95, ci_reason, coverage,
    replicate_sd, replicate_sd_reason}`.
    """
    def r(x: object) -> float | None:
        return _r(x, round_digits)

    ya = np.asarray(y, dtype=float).reshape(-1)
    n_all = ya.size
    if recipe_key is not None and len(recipe_key) != n_all:
        raise ValueError("recipe_key must have one entry per run")
    if run_ids is not None and len(run_ids) != n_all:
        raise ValueError("run_ids must have one entry per run")
    ok = np.isfinite(ya)
    idx = np.nonzero(ok)[0]
    v = ya[ok]
    n = int(v.size)

    out: dict[str, Any] = {
        "basis": "measured",
        "n_runs": n,
        "n_missing": int(n_all - n),
        "n_recipes": None,
        "n_replicated_recipes": None,
        "mean": None,
        "sd": None,
        "median": None,
        "min": None,
        "max": None,
        "best": None,
        "best_run_id": None,
        "ci95": None,
        "ci_reason": None,
        "coverage": 0.95,
        "replicate_sd": None,
        "replicate_sd_reason": None,
    }
    if n > 0:
        out["mean"] = r(v.mean())
        out["sd"] = r(v.std(ddof=1)) if n >= 2 else None
        out["median"] = r(np.median(v))
        out["min"] = r(v.min())
        out["max"] = r(v.max())
        b = int(np.argmax(v)) if maximize else int(np.argmin(v))
        out["best"] = r(v[b])
        if run_ids is not None:
            out["best_run_id"] = _label(run_ids[int(idx[b])])
    ci = t_interval(v)
    if ci is None:
        out["ci_reason"] = f"n<{MIN_RUNS_CI}"
    else:
        out["ci95"] = [r(ci[0]), r(ci[1])]

    if recipe_key is None:
        out["replicate_sd_reason"] = "no recipe key supplied"
        return out
    keys = [recipe_key[int(i)] for i in idx]
    if n == 0:
        out["n_recipes"] = 0
        out["n_replicated_recipes"] = 0
        out["replicate_sd_reason"] = "no runs with a measured value"
        return out
    _, _, y_var, n_reps = aggregate_replicates(np.zeros((n, 1)), v, groups=keys)
    replicated = n_reps >= 2
    n_rep = int(replicated.sum())
    out["n_recipes"] = int(n_reps.size)
    out["n_replicated_recipes"] = n_rep
    if n_rep >= 2:
        dof = (n_reps[replicated] - 1).astype(float)
        out["replicate_sd"] = r(math.sqrt(float((dof * y_var[replicated]).sum() / dof.sum())))
    else:
        out["replicate_sd_reason"] = f"needs >=2 recipes with >=2 runs; this clone has {n_rep}"
    return out


# --------------------------------------------------------------------------- #
# 2. Empirical-Bayes clone effects
# --------------------------------------------------------------------------- #


def eb_clone_effects(
    resid: ArrayLike,
    labels: Sequence[object] | ArrayLike,
    *,
    reference: object,
    min_runs: int = MIN_RUNS_EFFECT,
) -> dict[str, Any]:
    """Process-adjusted clone effects by random-intercept empirical Bayes.

    `resid` are OUT-OF-FOLD residuals `y - yhat_oof` from a process-only model
    (no clone term) that the caller fits; using OOF residuals keeps the process
    model from absorbing the clone effect in-sample.

    Per qualifying level k (n_k >= `min_runs`): `rbar_k`, `v_k = s_k^2 / n_k`
    with s_k^2 floored at `EB_VAR_FLOOR_FRAC` x the pooled within-level
    variance (a level with identical residuals would otherwise get infinite
    weight). DerSimonian-Laird tau^2 (method of moments, clipped at 0), Cochran
    Q, `I^2 = max(0, (Q-(K-1))/Q)` (None when Q == 0 or K < 2), random-effects
    mean `mu`, shrinkage `w_k = tau2/(tau2+v_k)`, `theta_k = w_k rbar_k +
    (1-w_k) mu`. tau2 = 0 means full pooling: every theta equals mu.

    Honesty rules:
      * Effects are RELATIVE to `reference` (in target units, never "% better"):
        `estimate_k = theta_k - theta_ref`, the reference itself is 0.0 with
        `is_reference: True` and no interval.
      * `ci95` is APPROXIMATE (`approximate: True`) and is the HULL of two
        intervals: the empirical-Bayes interval
        `estimate +/- 1.96 sqrt(w_k v_k + w_ref v_ref)` (which ignores the
        uncertainty in tau^2 and mu) and the unshrunk Welch t-interval for
        `rbar_k - rbar_ref`. The EB interval alone is NOT honest with 3-4
        clones: tau^2 is then estimated from 2-3 degrees of freedom, is often
        0, and a tau^2 of 0 collapses the EB interval to `[0, 0]` - a confident
        "no difference". In a 200-dataset simulation (3-4 clones, 6-10 runs,
        true offsets 0/0.2/0.4/0.1, noise sd 1) the EB interval covered the
        true difference 46% of the time; the hull covers >= the Welch
        interval's ~94% by construction. Shrinkage therefore moves the point
        estimate but never narrows the interval below what the clone's own runs
        support. `ci_method` names this.
      * `mostly_pooled: True` when w_k < 0.5: the estimate is more prior than
        data, and the UI must show that.
      * A level with n < `min_runs` gets no estimate, only refusal
        `clone_too_few_runs`.
      * Fewer than 2 qualifying levels (`clone_effect_needs_two_clones`) or a
        non-qualifying reference (`reference_clone_too_few_runs`) gives NO
        estimates at all, with a top-level `refusal`.

    Returns `{method, reference, tau, tau2, i2, q, k, mu, levels: {label: {...}},
    refusal, refusals}`.
    """
    if min_runs < 2:
        raise ValueError("min_runs must be >= 2 (a within-level variance needs 2 runs)")
    r = np.asarray(resid, dtype=float).reshape(-1)
    labs = _labels(labels, r.size)
    ref = _label(reference)
    keep = [i for i in range(r.size) if labs[i] is not None and math.isfinite(r[i])]
    groups: dict[str, list[float]] = {}
    for i in keep:
        groups.setdefault(str(labs[i]), []).append(float(r[i]))
    names = sorted(groups)

    refusals: list[dict[str, object]] = []
    levels: dict[str, dict[str, Any]] = {}
    for lab in names:
        n_lab = len(groups[lab])
        levels[lab] = {
            "basis": "derived", "n": n_lab, "rbar": None, "v": None, "shrinkage_w": None,
            "theta": None, "estimate": None, "ci95": None, "coverage": 0.95,
            "approximate": True, "ci_method": EB_CI_METHOD, "mostly_pooled": None, "is_reference": lab == ref, "refusal": None,
        }
        if n_lab < min_runs:
            ref_ = _refusal(
                "clone_too_few_runs",
                f"{lab} has {n_lab} run(s) with a residual; an adjusted effect needs >={min_runs}.",
                f"run >={min_runs - n_lab} more recipe(s) on {lab}",
                level=lab,
            )
            levels[lab]["refusal"] = ref_
            refusals.append(ref_)

    qual = [lab for lab in names if len(groups[lab]) >= min_runs]
    out: dict[str, Any] = {
        "method": "oof_residual_empirical_bayes", "reference": ref, "tau": None, "tau2": None,
        "i2": None, "q": None, "k": len(qual), "mu": None, "levels": levels,
        "refusal": None, "refusals": refusals,
    }
    top: dict[str, object] | None = None
    if len(qual) < 2:
        top = _refusal(
            "clone_effect_needs_two_clones",
            f"{len(qual)} clone(s) have >={min_runs} runs; a clone effect needs at least 2.",
            f"run >={min_runs} recipes on at least two clones",
        )
    elif ref is None or ref not in qual:
        n_ref = len(groups.get(ref, [])) if ref is not None else 0
        top = _refusal(
            "reference_clone_too_few_runs",
            f"reference clone {ref} has {n_ref} run(s); effects relative to it need >={min_runs}.",
            f"run >={min_runs - n_ref} more recipe(s) on {ref}, or choose a reference with >={min_runs} runs",
        )
    if top is not None:
        out["refusal"] = top
        refusals.append(top)
        return out
    assert ref is not None

    vals = [np.asarray(groups[lab]) for lab in qual]
    nk = np.array([g.size for g in vals], dtype=float)
    rbar = np.array([g.mean() for g in vals])
    s2 = np.array([g.var(ddof=1) for g in vals])
    s2_pool = float(((nk - 1) * s2).sum() / (nk - 1).sum())
    floor = EB_VAR_FLOOR_FRAC * s2_pool if s2_pool > 0 else 1e-12 * max(1.0, float(np.max(rbar**2)))
    v = np.maximum(s2, floor) / nk

    K = len(qual)
    wi = 1.0 / v
    rbar_w = float((wi * rbar).sum() / wi.sum())
    Q = float((wi * (rbar - rbar_w) ** 2).sum())
    C = float(wi.sum() - (wi**2).sum() / wi.sum())
    tau2 = max(0.0, (Q - (K - 1)) / C) if C > 0 else 0.0
    i2 = None if Q <= 0 else max(0.0, (Q - (K - 1)) / Q)
    ws = 1.0 / (v + tau2)
    mu = float((ws * rbar).sum() / ws.sum())
    w = tau2 / (tau2 + v)
    theta = w * rbar + (1.0 - w) * mu
    j_ref = qual.index(ref)

    for j, lab in enumerate(qual):
        ent = levels[lab]
        ent.update(
            rbar=_r(rbar[j]), v=_r(v[j]), shrinkage_w=_r(w[j]), theta=_r(theta[j]),
            mostly_pooled=bool(w[j] < 0.5),
        )
        if j == j_ref:
            ent["estimate"] = 0.0
            continue
        est = float(theta[j] - theta[j_ref])
        half = Z95 * math.sqrt(float(w[j] * v[j] + w[j_ref] * v[j_ref]))
        raw = float(rbar[j] - rbar[j_ref])
        vd = float(v[j] + v[j_ref])
        welch_df = vd**2 / (v[j] ** 2 / (nk[j] - 1) + v[j_ref] ** 2 / (nk[j_ref] - 1))
        half_raw = float(stats.t.ppf(0.975, welch_df)) * math.sqrt(vd)
        ent["estimate"] = _r(est)
        ent["ci95"] = [_r(min(est - half, raw - half_raw)), _r(max(est + half, raw + half_raw))]
    out.update(tau=_r(math.sqrt(tau2)), tau2=_r(tau2), i2=_r(i2), q=_r(Q), mu=_r(mu))
    return out


# --------------------------------------------------------------------------- #
# 3. Confounding
# --------------------------------------------------------------------------- #


def _paired(a: Sequence[object] | ArrayLike, b: Sequence[object] | ArrayLike) -> tuple[list[str], list[str]]:
    la, lb = _labels(a), _labels(b)
    if len(la) != len(lb):
        raise ValueError("label columns must have the same length")
    pa: list[str] = []
    pb: list[str] = []
    for x, y in zip(la, lb, strict=True):
        if x is not None and y is not None:
            pa.append(x)
            pb.append(y)
    return pa, pb


def _cramers_v_raw(pa: list[str], pb: list[str]) -> float | None:
    ra, rb = sorted(set(pa)), sorted(set(pb))
    if len(ra) < 2 or len(rb) < 2:
        return None
    ia = {k: i for i, k in enumerate(ra)}
    ib = {k: i for i, k in enumerate(rb)}
    tab = np.zeros((len(ra), len(rb)))
    for x, y in zip(pa, pb, strict=True):
        tab[ia[x], ib[y]] += 1
    n = tab.sum()
    exp = np.outer(tab.sum(1), tab.sum(0)) / n
    chi2 = float(((tab - exp) ** 2 / exp).sum())
    return min(1.0, math.sqrt(chi2 / (n * (min(len(ra), len(rb)) - 1))))


def cramers_v(a: Sequence[object] | ArrayLike, b: Sequence[object] | ArrayLike) -> float | None:
    """Plain Cramer's V between two label columns (no bias correction).

    Rows where either label is blank are dropped pairwise. Returns None when
    either column has fewer than 2 levels: association with a constant is
    undefined, not zero.
    """
    return _r(_cramers_v_raw(*_paired(a, b)))


def confounding(
    clone_labels: Sequence[object] | ArrayLike,
    other_columns: Mapping[str, Sequence[object] | ArrayLike],
) -> dict[str, Any]:
    """Is each clone separable from round, thaw, scale, medium lot...?

    For every other label column with >= 2 levels (blank labels dropped
    pairwise): `{column, n, cramers_v, fully, confined, aliased}`.
      * `fully`: V >= 1 - 1e-9 (the clone label and the column carry the same
        partition along the smaller dimension).
      * `confined`: `{clone: level}` for clones whose runs all sit in ONE level.
        Confined alone is NOT a confound when that level also holds other clones:
        the clone comparison is then made inside the shared level.
      * `aliased`: clones whose occupied levels contain no other clone. For these
        the data cannot tell the clone from the column (refusal
        `clone_effect_not_separable`).

    Honesty rule: `not_separable: {clone: [columns]}` lists every clone aliased
    with some column; the caller withholds that clone's adjusted effect. Columns
    with < 2 levels (or only one clone) are listed in `skipped` with a reason,
    not reported as unconfounded.
    """
    cols: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    not_sep: dict[str, list[str]] = {}
    for col in other_columns:
        pa, pb = _paired(clone_labels, other_columns[col])
        if len(set(pb)) < 2 or len(set(pa)) < 2:
            skipped.append({
                "column": str(col),
                "reason": "fewer than 2 levels of the column or of the clone among runs with both labels",
            })
            continue
        occ: dict[str, set[str]] = {}
        members: dict[str, set[str]] = {}
        for c, lv in zip(pa, pb, strict=True):
            occ.setdefault(c, set()).add(lv)
            members.setdefault(lv, set()).add(c)
        confined = {c: next(iter(lvs)) for c, lvs in sorted(occ.items()) if len(lvs) == 1}
        aliased = [c for c, lvs in sorted(occ.items()) if all(members[lv] == {c} for lv in lvs)]
        v = _cramers_v_raw(pa, pb)
        cols.append({
            "column": str(col), "n": len(pa), "cramers_v": _r(v),
            "fully": bool(v is not None and v >= 1 - 1e-9),
            "confined": confined, "aliased": aliased,
        })
        for c in aliased:
            not_sep.setdefault(c, []).append(str(col))
    return {"columns": cols, "not_separable": not_sep, "skipped": skipped}


# --------------------------------------------------------------------------- #
# 4. Process balance
# --------------------------------------------------------------------------- #


def _as_2d(X: ArrayLike, n_names: int) -> np.ndarray:
    Xa = np.asarray(X, dtype=float)
    if Xa.ndim == 1:
        Xa = Xa.reshape(-1, 1)
    if Xa.ndim != 2 or Xa.shape[1] != n_names:
        raise ValueError(f"Xcont must be (n_runs, {n_names}) to match names")
    return Xa


def process_balance(
    Xcont: ArrayLike,
    labels: Sequence[object] | ArrayLike,
    names: Sequence[str],
    *,
    min_runs: int = MIN_RUNS_CI,
) -> dict[str, Any]:
    """Did the clones run at the same process conditions?

    Per continuous input (rows with NaN or a blank label dropped per input):
    per clone `coverage = (clone max - min) / (global max - min)` and
    `smd = (clone mean - pooled mean) / pooled sd` (pooled = all runs, ddof=1;
    None when the sd or the range is 0).

    Honesty rule: raw clone means are confounded with process conditions when
    clones ran at different settings. An input is flagged
    (`clone_process_imbalance`) when an ELIGIBLE clone (>= `min_runs` runs) has
    |SMD| > SMD_FLAG (0.8) or coverage < COVERAGE_FLAG (0.3); smaller clones are
    reported in `by_level` but never flagged (two runs always "cover" little).
    Each flagged (clone, input) gets a plain-language message naming the input.

    Returns `{flag, by_input: {name: {smd_max, coverage_min, flag, by_level:
    {label: {n, min, max, mean, smd, coverage, eligible, flagged}}}}, messages}`.
    """
    names = [str(n) for n in names]
    Xa = _as_2d(Xcont, len(names))
    labs = _labels(labels, Xa.shape[0])
    by_input: dict[str, Any] = {}
    messages: list[str] = []
    any_flag = False
    for j, name in enumerate(names):
        x = Xa[:, j]
        rows = [i for i in range(x.size) if labs[i] is not None and math.isfinite(x[i])]
        xs = x[rows]
        ls = [str(labs[i]) for i in rows]
        entry: dict[str, Any] = {"smd_max": None, "coverage_min": None, "flag": False, "by_level": {}}
        by_input[name] = entry
        if xs.size == 0:
            continue
        gmin, gmax = float(xs.min()), float(xs.max())
        grange = gmax - gmin
        gmean = float(xs.mean())
        gsd = float(xs.std(ddof=1)) if xs.size >= 2 else 0.0
        smds: list[float] = []
        covs: list[float] = []
        for lab in sorted(set(ls)):
            xk = xs[[i for i, lb in enumerate(ls) if lb == lab]]
            kmin, kmax, kmean = float(xk.min()), float(xk.max()), float(xk.mean())
            smd = (kmean - gmean) / gsd if gsd > 0 else None
            cov = (kmax - kmin) / grange if grange > 0 else None
            eligible = xk.size >= min_runs
            flagged = eligible and (
                (smd is not None and abs(smd) > SMD_FLAG) or (cov is not None and cov < COVERAGE_FLAG)
            )
            entry["by_level"][lab] = {
                "n": int(xk.size), "min": _r(kmin), "max": _r(kmax), "mean": _r(kmean),
                "smd": _r(smd), "coverage": _r(cov), "eligible": bool(eligible), "flagged": bool(flagged),
            }
            if eligible:
                if smd is not None:
                    smds.append(abs(smd))
                if cov is not None:
                    covs.append(cov)
            if flagged:
                entry["flag"] = True
                any_flag = True
                tail = f"its difference from other clones partly reflects {name}."
                if cov is not None and cov < COVERAGE_FLAG:
                    messages.append(
                        f"{lab} ran only at {name} {_fmt(kmin)}-{_fmt(kmax)} "
                        f"(coverage {cov:.2f} of the observed range); {tail}"
                    )
                else:
                    side = "higher" if kmean > gmean else "lower"
                    messages.append(
                        f"{lab} ran at {side} {name} than the pooled runs (mean {_fmt(kmean)} vs "
                        f"{_fmt(gmean)}, SMD {smd:.2f}); {tail}"
                    )
        entry["smd_max"] = _r(max(smds)) if smds else None
        entry["coverage_min"] = _r(min(covs)) if covs else None
    return {"flag": any_flag, "by_input": by_input, "messages": messages}


# --------------------------------------------------------------------------- #
# 5. Interaction screen
# --------------------------------------------------------------------------- #


def bh_adjusted(pvals: ArrayLike) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (step-up, monotone, capped at 1).

    `bh_adjusted(p) <= q` iff `kalos.core.drivers.benjamini_hochberg(p, q)`
    rejects; NaN is treated as 1.0 there and here.
    """
    p = np.asarray(pvals, dtype=float).reshape(-1)
    m = p.size
    if m == 0:
        return np.zeros(0)
    p = np.where(np.isnan(p), 1.0, p)
    order = np.argsort(p, kind="stable")
    scaled = p[order] * m / np.arange(1, m + 1)
    adj_sorted = np.minimum(1.0, np.minimum.accumulate(scaled[::-1])[::-1])
    adj = np.empty(m)
    adj[order] = adj_sorted
    return adj


def _slope_fit(x: np.ndarray, y: np.ndarray, Z: np.ndarray | None) -> tuple[float, float, int] | None:
    """OLS slope of y on x (optionally adjusting for columns of Z), its SE and residual df.

    Columns are standardized for a scale-free rank check; constant Z columns are
    dropped (absorbed by the intercept). None when x is constant or the design is
    rank-deficient or has no residual degrees of freedom.
    """
    sx = float(x.std())
    if not sx > 0:
        return None
    cols = [np.ones_like(x), (x - x.mean()) / sx]
    if Z is not None:
        for c in range(Z.shape[1]):
            sz = float(Z[:, c].std())
            if sz > 0:
                cols.append((Z[:, c] - Z[:, c].mean()) / sz)
    A = np.column_stack(cols)
    n, k = A.shape
    if n <= k:
        return None
    q, rr = np.linalg.qr(A)
    if np.min(np.abs(np.diag(rr))) < 1e-8 * math.sqrt(n):
        return None
    coef = np.linalg.solve(rr, q.T @ y)
    res = y - A @ coef
    sigma2 = float(res @ res) / (n - k)
    rinv = np.linalg.inv(rr)
    var1 = sigma2 * float((rinv[1] ** 2).sum())
    return float(coef[1]) / sx, math.sqrt(max(var1, 0.0)) / sx, n - k


def _welch_heterogeneity_p(Q: float, w: np.ndarray, dfs: np.ndarray) -> float:
    """p-value for slope heterogeneity that accounts for ESTIMATED slope variances.

    Cochran's Q referred to chi2(K-1) treats each 1/se^2 weight as known. With
    within-clone slopes fitted on 6-12 runs each se rests on 3-10 residual
    degrees of freedom, and the chi2 reference is anti-conservative: in a
    200-seed null simulation (3 clones x 10 runs, equal slopes) it rejected at
    ~9.5% for a nominal 5%. Welch's (1951) heterogeneous-variance test is the
    small-sample version of the same statistic:
        F = (Q / (K-1)) / (1 + 2 (K-2) L / (K^2 - 1)),
        L = sum_i (1 - w_i / sum w)^2 / df_i,
    referred to F(K-1, (K^2-1) / (3 L)). It reduces to chi2(K-1) as every df_i
    grows, so large clones get Cochran's answer.
    """
    K = w.size
    L = float((((1.0 - w / w.sum()) ** 2) / dfs).sum())
    if not L > 0:
        return float(stats.chi2.sf(Q, K - 1))
    F = (Q / (K - 1)) / (1.0 + 2.0 * (K - 2) * L / (K**2 - 1))
    return float(stats.f.sf(F, K - 1, (K**2 - 1) / (3.0 * L)))


def interaction_screen(
    Xcont: ArrayLike,
    y: ArrayLike,
    labels: Sequence[object] | ArrayLike,
    names: Sequence[str],
    *,
    seed: int,
    n_boot: int = BOOT_B,
) -> dict[str, Any]:
    """Does the response to each input differ by clone (clone x input screen)?

    Runs with a non-finite input or outcome, or a blank label, are dropped.

    Thin-data rule (honesty): a (clone, input) cell is `insufficient`, with a
    reason naming what is missing, unless the clone has >= THIN_MIN_RUNS (6)
    runs, >= THIN_MIN_LEVELS (3) distinct values of the input, and covers >=
    THIN_MIN_RANGE_FRAC (50%) of the input's global range.

    A qualifying cell is `estimated`: the within-clone OLS slope of y on x_j,
    adjusting for the other inputs only when n_k >= p + 4 (p = number of
    inputs; `adjusted_for_other_inputs` says which), its OLS `se`, and a 95%
    percentile bootstrap CI over `n_boot` resamples of the clone's runs
    (numpy Generator seeded with `[seed, crc32(label, input)]`, so the result is
    deterministic for a seed and independent of cell order; degenerate
    resamples are skipped, and `ci95` is None if fewer than half are usable).

    Per input row: fewer than 2 qualifying clones -> `insufficient` with a
    reason listing each clone. That means the data CANNOT TEST the interaction;
    it is never reported as "no interaction". Otherwise Cochran's Q over the
    qualifying slopes (weights 1/se^2), df = K-1, `p` from Welch's small-sample
    reference for Q (see `_welch_heterogeneity_p`; chi2.sf(Q, df) is reported as
    `p_chi2` for comparison, and is anti-conservative at these run counts),
    I^2 = max(0, (Q-df)/Q), status `tested`, and `flag` (clone_input_interaction)
    when p < INTERACTION_P and I^2 >= INTERACTION_I2. `p_bh` is the BH-adjusted
    p across tested rows, with `survives_fdr` from
    `drivers.benjamini_hochberg` at q = INTERACTION_P.

    Returns `{rows: {name: {status, reason, Q, df, p, p_chi2, p_bh, survives_fdr, i2,
    flag, n_qualifying}}, cells: {label: {name: {status, reason, slope, se,
    ci95, n, adjusted_for_other_inputs}}}, thin_rule, flags, note}`.
    """
    names = [str(n) for n in names]
    p = len(names)
    Xa = _as_2d(Xcont, p)
    ya = np.asarray(y, dtype=float).reshape(-1)
    if ya.size != Xa.shape[0]:
        raise ValueError("y must have one entry per run")
    labs = _labels(labels, ya.size)
    keep = np.array(
        [labs[i] is not None and math.isfinite(ya[i]) and bool(np.all(np.isfinite(Xa[i]))) for i in range(ya.size)],
        dtype=bool,
    )
    Xa, ya = Xa[keep], ya[keep]
    labs_k = [str(lab) for lab, k in zip(labs, keep, strict=True) if k]
    levels = sorted(set(labs_k))
    rows_of = {lab: np.array([i for i, lb in enumerate(labs_k) if lb == lab], dtype=int) for lab in levels}
    seed_base = int(seed) % (2**32)

    cells: dict[str, dict[str, Any]] = {lab: {} for lab in levels}
    rows: dict[str, dict[str, Any]] = {}
    tested: list[str] = []
    p_raw: list[float] = []
    for j, name in enumerate(names):
        xg = Xa[:, j]
        grange = float(xg.max() - xg.min()) if xg.size else 0.0
        slopes: list[float] = []
        ses: list[float] = []
        dfs: list[int] = []
        summary: list[str] = []
        for lab in levels:
            idx = rows_of[lab]
            nk = int(idx.size)
            xk, yk = xg[idx], ya[idx]
            n_lv = int(np.unique(np.round(xk, 10)).size)
            cov = float(xk.max() - xk.min()) / grange if grange > 0 else 0.0
            missing: list[str] = []
            if nk < THIN_MIN_RUNS:
                missing.append(f"n={nk} < {THIN_MIN_RUNS} runs")
            if n_lv < THIN_MIN_LEVELS:
                missing.append(f"{n_lv} distinct levels < {THIN_MIN_LEVELS}")
            if grange <= 0:
                missing.append(f"{name} is constant across all runs")
            elif cov < THIN_MIN_RANGE_FRAC:
                missing.append(f"covers {cov * 100:.0f}% < {THIN_MIN_RANGE_FRAC * 100:.0f}% of the range")
            cell: dict[str, Any] = {
                "status": "insufficient", "reason": None, "slope": None, "se": None, "ci95": None,
                "n": nk, "adjusted_for_other_inputs": None,
            }
            cells[lab][name] = cell
            fit = None
            adjusted = p > 1 and nk >= p + 4
            Zk = np.delete(Xa[idx], j, axis=1) if adjusted else None
            if not missing:
                fit = _slope_fit(xk, yk, Zk)
                if fit is None and adjusted:
                    adjusted, Zk = False, None
                    fit = _slope_fit(xk, yk, None)
                if fit is None:
                    missing.append(f"slope not estimable (the clone's {name} values are collinear)")
            if missing or fit is None:
                cell["reason"] = "; ".join(missing)
                summary.append(f"{lab} has {nk} runs ({cell['reason']})")
                continue
            slope, se, dfk = fit
            ci = None
            if n_boot > 0:
                rng = np.random.default_rng([seed_base, zlib.crc32(f"{lab}\x00{name}".encode())])
                boots = np.full(n_boot, np.nan)
                for b in range(n_boot):
                    bi = rng.integers(0, nk, size=nk)
                    fb = _slope_fit(xk[bi], yk[bi], Zk[bi] if Zk is not None else None)
                    if fb is not None:
                        boots[b] = fb[0]
                good = boots[np.isfinite(boots)]
                if good.size >= max(2, n_boot // 2):
                    ci = [_r(np.quantile(good, 0.025)), _r(np.quantile(good, 0.975))]
            cell.update(
                status="estimated", slope=_r(slope), se=_r(se), ci95=ci,
                adjusted_for_other_inputs=bool(adjusted),
            )
            summary.append(f"{lab} has {nk} runs (qualifies)")
            slopes.append(slope)
            dfs.append(dfk)
            # An exact within-clone fit (se = 0) would get infinite weight; floor it.
            ses.append(max(se, 1e-9 * (abs(slope) + 1.0)))

        K = len(slopes)
        row: dict[str, Any] = {
            "status": "insufficient", "reason": None, "Q": None, "df": None, "p": None, "p_chi2": None,
            "p_bh": None, "survives_fdr": None, "i2": None, "flag": False, "n_qualifying": K,
        }
        rows[name] = row
        if K < 2:
            row["reason"] = (
                f"need >=2 clones with >={THIN_MIN_RUNS} runs, >={THIN_MIN_LEVELS} distinct {name} levels "
                f"and >={THIN_MIN_RANGE_FRAC * 100:.0f}% of the {name} range; {K} of {len(levels)} "
                f"clones qualify: " + ", ".join(summary)
            )
            continue
        s = np.asarray(slopes)
        w = 1.0 / np.asarray(ses) ** 2
        sbar = float((w * s).sum() / w.sum())
        q_stat = float((w * (s - sbar) ** 2).sum())
        df = K - 1
        pval = _welch_heterogeneity_p(q_stat, w, np.asarray(dfs, dtype=float))
        i2 = max(0.0, (q_stat - df) / q_stat) if q_stat > df else 0.0
        row.update(
            status="tested", Q=_r(q_stat), df=df, p=_r(pval), p_chi2=_r(stats.chi2.sf(q_stat, df)), i2=_r(i2),
            flag=bool(pval < INTERACTION_P and i2 >= INTERACTION_I2),
        )
        tested.append(name)
        p_raw.append(pval)

    if tested:
        adj = bh_adjusted(p_raw)
        surv = benjamini_hochberg(p_raw, q=INTERACTION_P)
        for name, pa, sv in zip(tested, adj, surv, strict=True):
            rows[name]["p_bh"] = _r(pa)
            rows[name]["survives_fdr"] = bool(sv)
    return {
        "rows": rows,
        "cells": cells,
        "thin_rule": {
            "min_runs": THIN_MIN_RUNS, "min_levels": THIN_MIN_LEVELS, "min_range_frac": THIN_MIN_RANGE_FRAC,
        },
        "flags": [n for n in names if rows[n]["flag"]],
        "note": "insufficient means the data cannot test it, not that there is no interaction",
    }


# --------------------------------------------------------------------------- #
# 6. Best per round
# --------------------------------------------------------------------------- #


def by_round_best(
    y: ArrayLike,
    labels: Sequence[object] | ArrayLike,
    rounds: Sequence[object] | ArrayLike,
    *,
    maximize: bool = True,
) -> list[dict[str, Any]]:
    """Best MEASURED value per clone per round (no model, no smoothing).

    Runs with a NaN outcome or a blank clone/round label are dropped. Rounds are
    sorted numerically when every round label is numeric (and returned as int
    when integral), otherwise as strings. A clone absent from a round is absent
    from that round's `levels`, not reported as 0.

    Returns `[{round, levels: {label: {best, n}}}]`.
    """
    ya = np.asarray(y, dtype=float).reshape(-1)
    labs = _labels(labels, ya.size)
    rds = _labels(rounds, ya.size, what="rounds")
    cells: dict[str, dict[str, list[float]]] = {}
    for i in range(ya.size):
        lab, rd = labs[i], rds[i]
        if lab is None or rd is None or not math.isfinite(ya[i]):
            continue
        cells.setdefault(rd, {}).setdefault(lab, []).append(float(ya[i]))

    def num(s: str) -> float | None:
        try:
            v = float(s)
        except ValueError:
            return None
        return v if math.isfinite(v) else None

    keys = list(cells)
    numeric = all(num(k) is not None for k in keys)
    if numeric:
        keys.sort(key=lambda k: num(k) or 0.0)
    else:
        keys.sort()
    out: list[dict[str, Any]] = []
    for k in keys:
        rd_out: object = k
        if numeric:
            v = num(k)
            rd_out = int(v) if v is not None and v.is_integer() else _r(v)
        out.append({
            "round": rd_out,
            "levels": {
                lab: {"best": _r(max(vals) if maximize else min(vals)), "n": len(vals)}
                for lab, vals in sorted(cells[k].items())
            },
        })
    return out
