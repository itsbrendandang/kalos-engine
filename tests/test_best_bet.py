"""Every batch carries the model's best bet (first-time-user study, Finding 1):
kalos/portal/analysis_parts/best_bet.py."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("botorch")

from kalos.portal.analysis import _analyze
from kalos.portal.analysis_parts.best_bet import place_best_bet, predicted_vs_best


def _sheet(n: int = 24, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ph = rng.uniform(6.0, 8.0, n).round(2)
    glu = rng.uniform(5.0, 20.0, n).round(1)
    titer = (50 - 20 * (ph - 7.1) ** 2 - 0.05 * (glu - 14) ** 2 + rng.normal(0, 0.5, n)).round(2)
    return pd.DataFrame({"pH": ph, "Glucose_g_L": glu, "Titer": titer})


def _check(out: dict) -> None:
    props = out["proposals"]
    opt = out["model_optimum"]
    assert props and opt is not None
    json.dumps(opt, allow_nan=False)
    sign = 1.0 if out.get("direction", "maximize") == "maximize" else -1.0
    if not opt["already_running"]:
        assert opt["in_batch"] == 0
        assert [p["pick"] for p in props].count("best_bet") == 1
        assert props[0]["pick"] == "best_bet" and props[0]["mode"] == "exploit"
        assert props[0]["pred"] == pytest.approx(opt["predicted"], abs=1e-3)
        assert any(p["mode"] != "explore" for p in props)  # never all "New ground"
    lo, hi = opt["likely_range"]
    assert lo <= opt["predicted"] <= hi
    for p in props:
        pvb = p["predicted_vs_best"]
        assert pvb["relation"] in {"better", "about_best", "worse"}
        assert pvb["plain"]
        if pvb["relation"] == "worse":
            assert "to learn, not to win" in pvb["plain"]
        # Invariant (security review B1): no proposal is predicted higher than the
        # best bet; only the 3-decimal rounding of `pred` separates them.
        assert sign * p["pred"] <= sign * opt["predicted"] + 1e-3, (p["pred"], opt["predicted"])


def test_best_bet_is_first_and_highest_predicted():
    out = _analyze(_sheet(), "Titer")
    _check(out)
    assert set(out["model_optimum"]["recipe"]) == {"pH", "Glucose_g_L"}


def test_minimize_best_bet_is_the_lowest_predicted():
    df = _sheet()
    df["Titer"] = -df["Titer"] + 100  # now lower is better near pH 7.1
    out = _analyze(df, "Titer", direction="minimize")
    _check(out)
    assert out["model_optimum"]["predicted"] <= min(p["pred"] for p in out["proposals"]) + 0.5


def test_best_bet_already_running_is_not_added_again():
    out = _analyze(_sheet(), "Titer")
    recipe = out["model_optimum"]["recipe"]
    again = _analyze(_sheet(), "Titer", pending=[recipe])
    assert again["model_optimum"]["already_running"] is True
    assert again["model_optimum"]["in_batch"] is None
    assert all(p["pick"] == "learn" for p in again["proposals"])


def test_place_best_bet_rules():
    bounds = np.array([[0.0, 0.0], [1.0, 1.0]])
    batch = np.array([[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]])
    mean = np.array([3.0, 1.0, 2.0])
    out, idx, running = place_best_bet(batch, np.array([0.7, 0.2]), mean, bounds, None, None, 3)
    assert idx == 0 and not running and out.shape == (3, 2)
    assert out[0].tolist() == [0.7, 0.2] and [0.5, 0.5] not in out.tolist()  # lowest replaced
    out, idx, _ = place_best_bet(batch, np.array([0.91, 0.9]), mean, bounds, None, None, 3)
    assert out[0].tolist() == [0.91, 0.9] and len(out) == 3  # near a proposal: it becomes P1
    out, idx, running = place_best_bet(batch, np.array([0.3, 0.3]), mean, bounds, None,
                                       np.array([[0.3, 0.3]]), 3)
    assert running and idx is None and out is batch
    out, idx, _ = place_best_bet(batch[:2], np.array([0.3, 0.8]), mean[:2], bounds, None, None, 3)
    assert len(out) == 3 and out[0].tolist() == [0.3, 0.8]  # short batch: appended


def test_predicted_vs_best_wording():
    worse = predicted_vs_best(40.0, 113.6, 27.5, maximize=True, target="Titer",
                              is_best_bet=False, mode="explore")
    assert worse["relation"] == "worse" and worse["difference"] == pytest.approx(-73.6)
    assert "to learn, not to win" in worse["plain"] and "least about" in worse["plain"]
    about = predicted_vs_best(110.0, 113.6, 27.5, maximize=True, target="Titer",
                              is_best_bet=True, mode="exploit")
    assert about["relation"] == "about_best" and about["plain"].startswith("Best bet")
    low = predicted_vs_best(5.0, 9.0, 1.0, maximize=False, target="Cost",
                            is_best_bet=False, mode="explore")
    assert low["relation"] == "better" and "below your best" in low["plain"]


SAMPLES = sorted(Path(os.path.expanduser("~/Desktop/kalos-samples")).glob("*.csv")) + [
    Path("/tmp/redesign/firsttime/samples/my_messy_sheet.csv"),
]
if len(SAMPLES) == 1:  # the sample folder is absent: keep one id so the test skips
    SAMPLES.insert(0, Path(os.path.expanduser("~/Desktop/kalos-samples/pichia_RBDJ.csv")))


@pytest.mark.parametrize("path", SAMPLES, ids=lambda p: p.stem)
def test_first_time_study_sheets_have_a_best_bet(path):
    """The four sheets of the first-time-user study (not in the repo; skipped
    when absent). Before: every proposal on A, B and C was below the best run
    and labelled New ground."""
    if not path.exists():
        pytest.skip(f"{path} not present")
    from kalos.portal.pipeline import compute_upload_analysis

    target = "Titer" if path.stem.startswith("pichia") or path.stem == "my_messy_sheet" else None
    _, out = compute_upload_analysis(path.read_bytes(), target, False, path.name)
    if out["verdict"]["state"] == "refused":
        # A refused verdict has no recipes and therefore no best bet.
        assert out["proposals"] == [] and out["model_optimum"] is None
        return
    _check(out)



def test_best_bet_is_the_argmax_over_all_candidates_not_a_local_optimum():
    """Planted (security review B1): the optimizer's answer is a local optimum
    at 1.0 while a batch row is predicted 5.0; the best bet must be that row."""
    from kalos.portal.analysis_parts.best_bet import (
        best_bet_candidates,
        choose_best_bet,
    )

    bounds = np.array([[0.0, 0.0], [1.0, 1.0]])
    optimizer_x = np.array([0.1, 0.1])
    batch = np.array([[0.9, 0.9], [0.5, 0.5]])
    X = np.array([[0.2, 0.3]])
    cands = best_bet_candidates(optimizer_x, batch, X, bounds, cat_dims=None,
                                cat_cardinalities=None, ge=None, eq=None, seed=0, n_samples=16)
    assert len(cands) == 1 + 2 + 1 + 16
    mean = np.zeros(len(cands))
    mean[0], mean[1] = 1.0, 5.0  # optimizer 1.0, first batch row 5.0
    idx, why = choose_best_bet(cands, mean)
    assert why is None and np.array_equal(cands[idx], batch[0])


def test_linear_constraints_filter_candidates():
    from kalos.portal.analysis_parts.best_bet import meets_linear

    C = np.array([[1.0, 2.0], [5.0, 5.0], [10.0, 0.0]])
    ge = [([0, 1], [-1.0, -1.0], -6.0)]  # x0 + x1 <= 6
    assert meets_linear(C, ge, None).tolist() == [True, False, False]
    eq = [([0, 1], [1.0, 1.0], 10.0)]
    assert meets_linear(C, None, eq).tolist() == [False, True, True]
