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
        # the best bet is the highest predicted value (within optimizer tolerance)
        assert sign * p["pred"] <= sign * opt["predicted"] + 0.02 * max(1.0, abs(opt["predicted"]))


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


SAMPLES = [
    Path(os.path.expanduser("~/Desktop/kalos-samples/pichia_RBDJ.csv")),
    Path(os.path.expanduser("~/Desktop/kalos-samples/cho_bioreactor_20829178.csv")),
    Path(os.path.expanduser("~/Desktop/kalos-samples/candida_rsm_nr1.csv")),
    Path("/tmp/redesign/firsttime/samples/my_messy_sheet.csv"),
]


@pytest.mark.parametrize("path", SAMPLES, ids=lambda p: p.stem)
def test_first_time_study_sheets_have_a_best_bet(path):
    """The four sheets of the first-time-user study (not in the repo; skipped
    when absent). Before: every proposal on A, B and C was below the best run
    and labelled New ground."""
    if not path.exists():
        pytest.skip(f"{path} not present")
    from kalos.portal.pipeline import compute_upload_analysis

    target = "Titer" if path.stem in {"pichia_RBDJ", "my_messy_sheet"} else None
    _, out = compute_upload_analysis(path.read_bytes(), target, False, path.name)
    _check(out)
