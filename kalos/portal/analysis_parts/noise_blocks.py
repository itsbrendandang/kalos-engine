"""The response blocks that describe noise and scale: the `noise` report and the
labeled `alternative_scale` evaluation pass (log-scale refit, evaluation only).
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np


def _num(x: object, nd: int = 4) -> float | None:
    """Round to `nd` places, or None for anything that is not a finite number.
    NaN and inf both become None so the response stays strict-JSON safe.

    Module-level (not local to `_noise_block`) so `_alternative_scale_block`
    can round `icc_log`/`log_offset` through the exact same function `_noise_block`
    uses for the raw `icc_raw`/`icc_log`/`log_offset` in `noise.scale` - the two
    blocks report overlapping numbers from the same `heteroscedasticity_report`
    call and must never disagree because of a rounding difference between two
    copies of the same helper."""
    if not isinstance(x, (int, float)) or isinstance(x, bool):
        return None
    if not np.isfinite(float(x)):
        return None
    return round(float(x), nd)


def _noise_block(
    nr: dict,
    best_single: float,
    best_reproducible: float | None,
    replicate_aware: bool,
    het: dict | None = None,
) -> dict:
    """The `noise` report for the analysis result: replicate structure + the
    honest signal-to-noise picture (BENCHMARK.md, "the real lever is assay noise").

    `icc` is the intraclass correlation - the fraction of titer variance that is
    real recipe-to-recipe signal rather than assay noise; a low ICC means most of
    the spread is noise and single-measurement "bests" are largely luck.
    `best_single` is the best individual measurement (rewards noise spikes);
    `best_reproducible` is the best replicate-averaged recipe - the titer a client
    would actually ship. `replicate_aware` is True when the proposed batch was fit
    on that reproducible objective with the measured noise floor fed to the GP.
    NaN stats (too few recipes / no replicates to estimate them) serialize as null.
    """
    noise_var = nr["noise_var"]
    signal_var = nr["signal_var"]
    return {
        "n_recipes": int(nr["n_recipes"]),
        "n_replicated": int(nr["n_replicated"]),
        "replicate_aware": bool(replicate_aware),
        "icc": _num(nr["icc"], 3),
        # Exact copies of a run (same factors AND same target) are not replicates
        # and are excluded from noise_sd / icc (review M1).
        "n_exact_duplicates": int(nr.get("n_exact_duplicates", 0)),
        "noise_sd": _num(np.sqrt(noise_var) if np.isfinite(noise_var) else None),
        "signal_sd": _num(np.sqrt(signal_var) if np.isfinite(signal_var) else None),
        "best_single": _num(best_single),
        "best_reproducible": _num(best_reproducible),
        # Whether the homoscedastic assumption behind `noise_sd` actually holds on
        # this sheet. A DIAGNOSTIC: nothing above is transformed, because changing
        # the target's scale would change every number in this response. When
        # `suggests_transform` is true, `icc` here is understating real signal.
        "scale": None
        if het is None
        else {
            "variance_mean_rho": _num(het.get("variance_mean_rho"), 3),
            "homoscedastic": het.get("homoscedastic"),
            "icc_raw": _num(het.get("icc_raw"), 3),
            "icc_log": _num(het.get("icc_log"), 3),
            "icc_gain": _num(het.get("icc_gain"), 3),
            "log_offset": _num(het.get("log_offset"), 6),
            "suggests_transform": bool(het.get("suggests_transform")),
            "reason": het.get("reason"),
        },
    }


# `alternative_scale` is a NULL-with-reason dict when not suggested (same
# pattern as `cv_logo` below: every key present, values `None`, `reason` states
# why), so a consumer never has to special-case "the key is missing."
_ALT_SCALE_NULL_KEYS = (
    "scale", "offset", "cv_spearman", "cv_ci95", "calibration", "icc", "interpretation",
)

_ALT_SCALE_INTERPRETATION = (
    "The log scale fits materially better here, which means the raw-scale assay "
    "noise floor is partly a scale artifact rather than uniform measurement "
    "noise, not that the underlying process changed. Proposals and every other "
    "number in this response remain on the raw scale; this block is "
    "evaluation-only."
)


def _alternative_scale_block(
    het: dict,
    X: np.ndarray,
    y: np.ndarray,
    *,
    groups,
    bounds,
    cat_dims,
    grouped_cv_report: Callable[..., dict],
    interval_calibration: Callable[..., dict],
) -> dict:
    """The `alternative_scale` report: an EXPLICIT, LABELED second evaluation
    pass on the log scale, run only when `heteroscedasticity_report` (`het`,
    already computed by the caller and reported in `noise.scale`) says a log
    transform would materially raise the ICC (`suggests_transform`).

    WHY THIS EXISTS. `heteroscedasticity_report` diagnoses the scale problem
    every analysis, `_noise_block` reports the diagnosis, and the engine then
    went on to fit the PROPOSAL surrogate on the raw scale regardless - the
    standing design decision (see `heteroscedasticity_report`'s docstring) is
    that the target's scale is never changed SILENTLY, because every reported
    number lives on it. Diagnosing and then quietly modeling on the wrong scale
    anyway is worse than either alone: it is the gap this block closes, by
    running the alternative openly instead of switching to it in secret.

    OFFSET SEMANTICS - READ BEFORE CHANGING. `het["log_offset"]` is the exact
    `c` `heteroscedasticity_report` used to compute `het["icc_log"]` there, via
    `np.log(y + c)` (see that function's docstring). It is NOT the argument to
    literal `np.log1p`, i.e. `log(1 + y + c)` - that transform was the first
    thing tried in `heteroscedasticity_report`'s own history and was WRONG at
    titer's scale (~0.005-0.02): the `+1` dominates the tiny offset, the
    transform is nearly the identity, and it moved the ICC by only 0.002 on a
    sheet with a real variance-mean coupling. `"log1p"` in this module's (and
    that function's) naming is an informal label for the transform FAMILY - log
    of the value plus a small, data-scaled offset so zeros survive - not a
    literal call to `np.log1p`. Fitting on `np.log1p(y + c)` here would
    silently reproduce that exact historical bug AND desynchronize this
    block's `icc` from `het["icc_log"]`, which defeats the consistency this
    block exists to guarantee. So the transform below is `np.log(y + c)`,
    byte-for-byte the same call `heteroscedasticity_report` made.

    WHY REFIT RATHER THAN RE-RANK. `cv_spearman` is rank-based: ranking the
    SAME held-out predictions after a log transform would return an identical
    number, which would be free but would prove nothing about the log scale
    specifically. The point of this block is that a NEW GP is fit and
    cross-validated on `log(y + c)` - different lengthscales, different
    inferred noise, a different posterior - so `cv_spearman` here can
    genuinely differ from the raw `cv_spearman`, and `calibration` here answers
    "are the log-scale bands honest" independently of whether the raw-scale
    bands were. Both are real information about which scale the process lives
    on; do not "optimize away" this second fit as redundant with the first.

    COST. `n_repeats=1` (not `CV_N_REPEATS`): this is a diagnostic comparison,
    not the headline reliability number, so the partition-variance treatment
    the headline `cv_spearman`/`cv_ci95` pays double for is not duplicated here.
    Runs at all only when `het["suggests_transform"]` is True, so an
    unaffected sheet pays nothing extra for this block.

    SCOPE. Evaluation-only: this function never touches `X`/`y`, the proposal
    path, the feasibility gate, or the constraint path, and returns a plain
    dict, not a fitted model. `calibration`'s `ece`/`z_std` are computed on
    LOG-SCALE residuals - comparable to the raw block's calibration in
    STRUCTURE (is coverage honest at each nominal level) but not in absolute
    magnitude, since the two live in different units.

    Returns the shape above when suggested; the same keys with `None` values
    plus a stated `reason` (the `cv_logo`-style null pattern) otherwise."""
    null_block: dict[str, object] = dict.fromkeys(_ALT_SCALE_NULL_KEYS)
    null_block["reason"] = het.get("reason") or "log transform not suggested on this sheet"
    if not het.get("suggests_transform") or het.get("log_offset") is None:
        return null_block

    c = float(het["log_offset"])
    # Byte-for-byte the same transform `heteroscedasticity_report` used for
    # `icc_log` - see the offset-semantics note above for why this must not be
    # `np.log1p(y + c)`.
    y_log = np.log(np.asarray(y, dtype=float) + c)
    rep_log = grouped_cv_report(
        X, y_log, groups=groups, n_splits=5, bounds=bounds, cat_dims=cat_dims, n_repeats=1,
    )
    rho_log = rep_log["spearman"]
    ci95_log = (
        None
        if rho_log != rho_log
        else [round(rep_log["ci95"][0], 3), round(rep_log["ci95"][1], 3)]
    )
    calibration_log = interval_calibration(rep_log["oof_actual"], rep_log["oof_pred"], rep_log["oof_std"])
    return {
        "scale": "log1p",
        "offset": _num(c, 6),
        "cv_spearman": None if rho_log != rho_log else round(float(rho_log), 3),
        "cv_ci95": ci95_log,
        # LOG-SCALE OOF: `ece`/`z_std` inside this dict are in log units, NOT
        # comparable in magnitude to the raw block's `reliability.calibration` -
        # only to its structure (is coverage honest at each nominal level).
        "calibration": calibration_log,
        "icc": _num(het.get("icc_log"), 3),
        "reason": het.get("reason"),
        "interpretation": _ALT_SCALE_INTERPRETATION,
    }
