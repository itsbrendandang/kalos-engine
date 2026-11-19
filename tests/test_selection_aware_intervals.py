"""F15: selection-aware per-proposal intervals (fast unit tests).

The coverage these bands reach at proposals is measured by
`python -m kalos.bench.coverage_at_proposals` (VALIDATION.md) and re-checked on
a few seeds in tests/test_trust_simulations.py (slow)."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from kalos.portal.analysis_parts.coverage_evidence import (
    BASIS_HELD_OUT,
    BASIS_SELECTION_AWARE,
)
from kalos.portal.analysis_parts.intervals import (
    METHOD_SELECTION_AWARE,
    band_from_summary,
    calibrate_normalized,
    calibrate_selection_aware,
    interval_method_summary,
    proposal_interval_fields,
)

LEVELS = (0.9, 0.95)


def _repeats(n: int = 60, folds: int = 5, n_rep: int = 2, seed: int = 0, optimism: float = 0.0):
    """OOF repeats where the model is optimistic about its top-ranked runs."""
    rng = np.random.default_rng(seed)
    out = []
    for r in range(n_rep):
        pred = rng.normal(0, 1, n)
        sd = rng.uniform(0.5, 1.5, n)
        top = pred >= np.quantile(pred, 0.7)
        actual = pred + rng.normal(0, sd) - optimism * top * sd
        fold = (np.arange(n) + r) % folds
        out.append({"pred": pred.tolist(), "std": sd.tolist(), "actual": actual.tolist(), "fold": fold.tolist()})
    return out


def test_selects_the_top_fraction_of_every_fold_and_floors_at_a_quantile():
    reps = _repeats(n=50, folds=5)
    cal = calibrate_selection_aware(reps, LEVELS)
    assert cal["method"] == METHOD_SELECTION_AWARE
    assert cal["n_pooled"] == 100
    assert cal["n_selected"] == 2 * 5 * math.ceil(0.3 * 10)
    all_sd = np.concatenate([r["std"] for r in reps])
    assert cal["_floor"] == pytest.approx(np.quantile(all_sd, 0.9))
    l90 = cal["levels"][0]
    assert l90["available"] and l90["rank"] == math.ceil((cal["n_selected"] + 1) * 0.9)
    # the quantile is the order statistic of the selected normalized scores
    scores = []
    for r in reps:
        p, s, a, f = (np.asarray(r[k]) for k in ("pred", "std", "actual", "fold"))
        for k in np.unique(f):
            idx = np.where(f == k)[0]
            top = idx[np.argsort(-p[idx], kind="stable")[: math.ceil(0.3 * len(idx))]]
            scores += (np.abs(a[top] - p[top]) / s[top]).tolist()
    assert l90["_q"] == pytest.approx(np.sort(scores)[l90["rank"] - 1])


def test_minimize_selects_the_lowest_predictions():
    reps = _repeats(n=50)
    hi = calibrate_selection_aware(reps, LEVELS, maximize=True)
    lo = calibrate_selection_aware(reps, LEVELS, maximize=False)
    assert hi["n_selected"] == lo["n_selected"]
    assert hi["levels"][0]["_q"] != lo["levels"][0]["_q"]


def test_optimism_at_the_top_widens_the_band():
    reps = _repeats(n=80, optimism=2.0, seed=3)
    sel = calibrate_selection_aware(reps, LEVELS)
    r0 = reps[0]
    plain = calibrate_normalized(r0["actual"], r0["pred"], r0["std"], LEVELS)
    assert sel["levels"][0]["_q"] > plain["levels"][0]["_q"]


def test_too_few_top_ranked_runs_refuses_the_level_with_the_count_needed():
    reps = _repeats(n=10, n_rep=1)
    cal = calibrate_selection_aware(reps, LEVELS)
    assert cal["n_selected"] == 5 * math.ceil(0.3 * 2)
    l90, l95 = cal["levels"]
    assert l90["available"] is False and l90["code"] == "too_few_top_ranked_runs" and l90["min_selected"] == 9
    assert l95["available"] is False and l95["min_selected"] == 19


def test_degenerate_sd_refuses():
    reps = _repeats(n=40)
    reps[0]["std"][3] = 0.0
    cal = calibrate_selection_aware(reps, LEVELS)
    assert all(lv["available"] is False and lv["code"] == "degenerate_sigma" for lv in cal["levels"])


def test_proposal_band_uses_the_floored_sd_and_labels_its_basis():
    reps = _repeats(n=60)
    sel = calibrate_selection_aware(reps, LEVELS)
    r0 = reps[0]
    plain = calibrate_normalized(r0["actual"], r0["pred"], r0["std"], LEVELS)
    floor = sel["_floor"]
    rows = proposal_interval_fields([2.0, 3.0], [0.1, 0.1], [floor / 2, floor * 2], plain, LEVELS, 1.0,
                                    "best_measured_single", selection=sel)
    for row, sd in zip(rows, [floor / 2, floor * 2]):
        for iv, lv in zip(row["intervals"], sel["levels"]):
            assert iv["method"] == METHOD_SELECTION_AWARE and iv["coverage_basis"] == BASIS_SELECTION_AWARE
            half = lv["_q"] * max(sd, floor)
            assert iv["hi"] - iv["lo"] == pytest.approx(2 * half, abs=2e-4)
            assert iv["sigma_used"] == round(max(sd, floor), 4)
            assert iv["caveat"]["coverage_at_proposals_estimate"] is None or \
                iv["caveat"]["coverage_at_proposals_estimate"]["nominal"] == iv["coverage"]
    summary = interval_method_summary(plain, "surrogate_fitted_noise", sel)
    assert summary["method"] == METHOD_SELECTION_AWARE
    assert summary["method_by_level"] == {"0.9": METHOD_SELECTION_AWARE, "0.95": METHOD_SELECTION_AWARE}
    for k, lv in enumerate(LEVELS):
        lo, hi = band_from_summary(np.array([3.0]), np.array([floor * 2]), summary, lv)
        assert lo is not None and hi is not None
        assert lo[0] == pytest.approx(rows[1]["intervals"][k]["lo"], abs=2e-3)
    json.dumps([rows, summary], allow_nan=False)


def test_falls_back_to_the_plain_band_per_level_and_says_why():
    reps = _repeats(n=40)                      # 2 x 5 folds x 3 = 30 selected: 90% ok, 95% needs 19 -> ok
    sel = calibrate_selection_aware(reps[:1], LEVELS)  # one repeat: 15 selected -> 95% refused
    r0 = reps[0]
    plain = calibrate_normalized(r0["actual"], r0["pred"], r0["std"], LEVELS)
    (row,) = proposal_interval_fields([1.0], [0.1], [0.8], plain, LEVELS, 0.0, "best_measured_single",
                                      selection=sel)
    i90, i95 = row["intervals"]
    assert i90["method"] == METHOD_SELECTION_AWARE
    assert i95["method"] == "conformal_normalized" and i95["coverage_basis"] == BASIS_HELD_OUT
    assert i95["selection_aware_unavailable"]["code"] == "too_few_top_ranked_runs"
    summary = interval_method_summary(plain, "surrogate_fitted_noise", sel)
    assert summary["method"] == "mixed"
    assert summary["method_by_level"]["0.95"] == "conformal_normalized"


def test_analysis_ships_selection_aware_bands():
    import pandas as pd

    from kalos.portal.analysis import _analyze

    rng = np.random.default_rng(0)
    m, ph = rng.uniform(0, 4, 40), rng.uniform(5, 7, 40)
    df = pd.DataFrame({"Methanol": m.round(3), "pH": ph.round(2),
                       "titer": (1.5 * m - 0.8 * (ph - 6) ** 2 + rng.normal(0, 0.1, 40)).round(3)})
    r = _analyze(df, target="titer")
    im = r["interval_method"]
    assert im["selection_aware"]["n_pooled"] == 2 * len(r["oof"])
    for p in r["proposals"]:
        for iv in p["intervals"]:
            assert iv["available"] and iv["lo"] <= p["pred"] <= iv["hi"]
            assert iv["method"] == im["method_by_level"][f"{iv['coverage']:g}"]
