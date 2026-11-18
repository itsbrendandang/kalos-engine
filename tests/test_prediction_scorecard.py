"""Prediction scorecard: saved prediction and band vs measured result
(docs/CAMPAIGN_LOOP.md, "Prediction scorecard")."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kalos.portal.campaign import CampaignStore
from kalos.portal.scorecard import MIN_SCORED_RUNS, prediction_scorecard


def _df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "Methanol": rng.uniform(0, 4, 8).round(3),
        "pH": rng.uniform(5, 7, 8).round(2),
        "lipase_titer": [1.0, 2.0, 3.0, 4.0, 2.5, 1.5, 3.5, 0.5],
    })


def _run(pred, result, half=1.0, coverage=0.9, best=4.0, **kw):
    return {"id": kw.get("id", "r"), "round": 0, "mode": "explore", "pred": pred, "std": 0.3,
            "band_half_width": half, "band_coverage": coverage, "best_at_start": best,
            "result": result}


def test_empty_scorecard_is_not_judgeable():
    card = prediction_scorecard([])
    assert card["n_scored"] == 0 and card["n_banded"] == 0
    assert card["band_coverage_observed"] is None and card["spearman"] is None
    assert card["judgeable"] is False
    assert card["runs"] == []


def test_rows_score_error_band_and_gain():
    card = prediction_scorecard([_run(5.0, 5.5), _run(3.0, 4.5), _run(2.0, 1.8)])
    assert [r["in_band"] for r in card["runs"]] == [True, False, True]
    assert [r["beat_best"] for r in card["runs"]] == [True, True, False]
    assert card["runs"][1]["error"] == pytest.approx(1.5)
    assert card["band_hits"] == 2 and card["n_banded"] == 3
    assert card["band_coverage_observed"] == pytest.approx(2 / 3, abs=1e-4)
    assert card["band_coverage_stated"] == pytest.approx(0.9)
    assert card["mean_abs_error"] == pytest.approx((0.5 + 1.5 + 0.2) / 3)
    assert card["spearman"] == pytest.approx(1.0)
    assert card["n_beat_best"] == 2 and card["n_with_best"] == 3


def test_unmeasured_and_unpredicted_runs_are_skipped():
    card = prediction_scorecard([_run(5.0, None), _run(None, 3.0), _run(float("nan"), 3.0)])
    assert card["n_scored"] == 0


def test_runs_without_a_band_are_scored_but_not_banded():
    card = prediction_scorecard([_run(5.0, 5.2, half=None, coverage=None, best=None)])
    assert card["n_scored"] == 1 and card["n_banded"] == 0
    assert card["runs"][0]["in_band"] is None and card["runs"][0]["beat_best"] is None


def test_judgement_says_too_narrow_when_coverage_falls_short():
    runs = [_run(0.0, 0.5 if i < MIN_SCORED_RUNS - 3 else 2.0, id=str(i)) for i in range(MIN_SCORED_RUNS)]
    card = prediction_scorecard(runs)
    assert card["judgeable"] is True
    assert card["band_coverage_observed"] == pytest.approx(0.7)
    assert card["judgement"].startswith("band too narrow")


def test_judgement_says_held_when_coverage_meets_stated():
    runs = [_run(0.0, 0.5, id=str(i)) for i in range(MIN_SCORED_RUNS)]
    assert prediction_scorecard(runs)["judgement"].startswith("band held")


def test_start_saves_band_and_best_and_the_card_survives_the_fold(tmp_path):
    store = CampaignStore(tmp_path)
    store.seed(_df(), "lipase_titer", ["Methanol", "pH"], tenant="default")
    store.save_analysis(
        {"conformal_q": 0.75, "conformal_coverage": 0.9}, "upload", tenant="default",
    )
    started = store.start([
        {"recipe": {"Methanol": 2.0, "pH": 6.0}, "pred": 4.2, "std": 0.3, "mode": "exploit",
         "reason": "a"},
        {"recipe": {"Methanol": 3.0, "pH": 6.5}, "pred": 3.0, "std": 0.5, "mode": "explore",
         "reason": "b"},
    ], tenant="default")
    assert started[0]["band_half_width"] == 0.75
    assert started[0]["band_coverage"] == 0.9
    assert started[0]["best_at_start"] == 4.0

    store.set_result(started[0]["id"], 4.6, tenant="default")
    store.set_result(started[1]["id"], 5.0, tenant="default")
    before = store.summary(tenant="default")["scorecard"]
    assert before["n_scored"] == 2
    assert [r["in_band"] for r in before["runs"]] == [True, False]

    _, _, generation = store.plan_fold(tenant="default")
    store.commit_fold(generation, tenant="default")
    summary = store.summary(tenant="default")
    assert summary["pending"] == []
    assert summary["scorecard"] == before

    # A run started after the fold compares against the new best.
    later = store.start([{"recipe": {"Methanol": 1.0, "pH": 5.5}, "pred": 4.0, "std": 0.4,
                          "mode": "explore", "reason": "c"}], tenant="default")
    assert later[0]["best_at_start"] == 5.0


def test_start_without_an_analysis_saves_no_band(tmp_path):
    store = CampaignStore(tmp_path)
    store.seed(_df(), "lipase_titer", ["Methanol", "pH"], tenant="default")
    run = store.start([{"recipe": {"Methanol": 2.0, "pH": 6.0}, "pred": 4.2, "std": 0.3,
                        "mode": "exploit", "reason": "a"}], tenant="default")[0]
    assert run["band_half_width"] is None and run["band_coverage"] is None
    store.set_result(run["id"], 4.1, tenant="default")
    card = store.summary(tenant="default")["scorecard"]
    assert card["n_scored"] == 1 and card["n_banded"] == 0


def test_best_at_start_counts_measured_runs_not_yet_folded(tmp_path):
    store = CampaignStore(tmp_path)
    store.seed(_df(), "lipase_titer", ["Methanol", "pH"], tenant="default")
    first = store.start([{"recipe": {"Methanol": 2.0, "pH": 6.0}, "pred": 4.2, "std": 0.3,
                          "mode": "exploit", "reason": "a"}], tenant="default")[0]
    store.set_result(first["id"], 6.0, tenant="default")
    second = store.start([{"recipe": {"Methanol": 3.0, "pH": 6.0}, "pred": 5.0, "std": 0.3,
                           "mode": "exploit", "reason": "b"}], tenant="default")[0]
    assert second["best_at_start"] == 6.0
