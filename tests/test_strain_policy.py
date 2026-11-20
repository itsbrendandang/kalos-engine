"""Clone policy, refusals, confounding flags and anonymization (docs/CLONES.md).

Each test runs one analysis on a small synthetic sheet; the numbers are
synthetic and the assertions are on structure and refusals, not on fit quality.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from kalos.portal.analysis import _analyze
from kalos.portal.setup_settings import analysis_setup, normalize_setup

pytestmark = pytest.mark.slow


def sheet(
    per_clone: dict[str, int], *, offsets: dict[str, float] | None = None, seed: int = 1,
    temp_by_clone: dict[str, tuple[float, float]] | None = None,
    round_by_clone: dict[str, int] | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for lv, n in per_clone.items():
        lo, hi = (temp_by_clone or {}).get(lv, (33.0, 37.0))
        for _ in range(n):
            t = rng.uniform(lo, hi)
            feed = rng.uniform(2, 6)
            y = (offsets or {}).get(lv, 0.0) + 2 + 0.3 * (feed - 4) - 0.1 * (t - 35) ** 2 \
                + rng.normal(0, 0.15)
            row = {"clone": lv, "temp_C": round(t, 3), "feed_pct": round(feed, 3),
                   "titer_g_L": round(float(y), 4)}
            if round_by_clone is not None:
                row["round"] = round_by_clone[lv]
            rows.append(row)
    return pd.DataFrame(rows)


def setup(**doc) -> dict:
    return analysis_setup(normalize_setup(doc))


def test_fixed_policy_holds_every_proposal_at_the_named_clone():
    df = sheet({"A": 7, "B": 7, "C": 7}, offsets={"C": 1.0})
    r = _analyze(df, setup=setup(strain_column="clone", strain_level="B"))
    s = r["strain_summary"]
    assert s["policy"] == "fixed" and s["fixed_level"] == "B" and s["fixed_level_source"] == "setup"
    assert r["proposals"] and all(p["recipe"]["clone"] == "B" for p in r["proposals"])
    assert all(p["clone"] == {"column": "clone", "level": "B", "policy": "fixed", "n_runs": 7}
               for p in r["proposals"])
    assert all("extrapolated_for_clone" in p for p in r["proposals"])


def test_unknown_fixed_level_is_refused_and_proposals_withheld():
    df = sheet({"A": 6, "B": 6})
    r = _analyze(df, setup=setup(strain_column="clone", strain_level="Z9"))
    assert r["proposals"] == []
    assert r["proposal_notice"]["code"] == "clone_not_in_training"
    codes = {x["code"] for x in r["strain_summary"]["refusals"]}
    assert "clone_not_in_training" in codes
    assert r["cv_spearman"] is not None  # the rest of the analysis is still there


def test_registry_lead_without_runs_falls_back_to_most_runs():
    df = sheet({"A": 5, "B": 8})
    r = _analyze(df, setup={**setup(strain_column="clone"), "strain_lead": "Z9"})
    s = r["strain_summary"]
    assert s["fixed_level"] == "B" and s["fixed_level_source"] == "most_runs"
    assert "not a ranking" in s["fixed_level_note"]
    assert "clone_not_in_training" not in {x["code"] for x in s["refusals"]}


def test_thin_clone_is_refused_everywhere_it_would_be_a_claim():
    df = sheet({"A": 8, "B": 8, "D": 2}, offsets={"D": 2.0})
    r = _analyze(df, setup=setup(strain_column="clone", strain_policy="factor"))
    s = r["strain_summary"]
    d = next(lv for lv in s["levels"] if lv["label"] == "D")
    o = d["outcomes"]["titer_g_L"]
    assert d["n_runs"] == 2 and o["ci95"] is None and o["ci_reason"] == "n<3"
    assert "clone_too_few_runs" in d["flags"]
    assert d["adjusted_effect"]["estimate"] is None and d["adjusted_effect"]["refusal"]
    assert d["gp_at_reference"]["pred"] is None
    assert r["slices_by_level"]["clone"]["D"] == []
    assert any(x["level"] == "D" for x in r["slices_by_level_skipped"]["clone"])
    for p in r["proposals"]:
        if p["recipe"]["clone"] == "D":
            assert p["mode"] == "explore" and p["clone_refusal"]["code"] == "clone_too_few_runs"
    # interaction cells for a 2-run clone are never "no interaction"
    blob = json.dumps(s["interaction"]).lower()
    assert "no interaction" not in blob.replace("not that there is no interaction", "")


def test_clone_nested_in_round_withholds_the_adjusted_effect():
    df = sheet({"A": 6, "B": 6, "C": 6}, offsets={"B": 0.5, "C": 1.0},
               round_by_clone={"A": 0, "B": 0, "C": 1})
    r = _analyze(df, setup=setup(strain_column="clone"))
    s = r["strain_summary"]
    c = next(lv for lv in s["levels"] if lv["label"] == "C")
    assert c["adjusted_effect"]["estimate"] is None
    assert c["adjusted_effect"]["refusal"]["code"] == "clone_effect_not_separable"
    assert "clone_confounded_with:round" in c["flags"]
    b = next(lv for lv in s["levels"] if lv["label"] == "B")
    assert b["adjusted_effect"]["estimate"] is not None  # shares round 0 with A
    assert s["by_round"]["available"] and s["by_round"]["column"] == "round"


def test_clones_at_disjoint_temperatures_flag_process_imbalance():
    df = sheet({"A": 7, "B": 7}, temp_by_clone={"A": (33.0, 34.0), "B": (36.0, 37.0)})
    r = _analyze(df, setup=setup(strain_column="clone"))
    s = r["strain_summary"]
    assert s["process_balance"]["flag"] is True
    assert any(f.startswith("clone_process_imbalance:temp_C")
               for lv in s["levels"] for f in lv["flags"])


def test_anonymize_never_returns_a_real_clone_label():
    df = sheet({"CHO-K1-mAb7-C33": 6, "CHO-K1-mAb7-B07": 6})
    r = _analyze(df, anonymize=True, setup=setup(strain_column="clone"))
    blob = json.dumps(r)
    assert "CHO-K1-mAb7" not in blob
    labels = [lv["label"] for lv in r["strain_summary"]["levels"]]
    assert all(lab.startswith("level_") for lab in labels)
    assert set(r["oof_levels"]["clone"]) <= set(labels)
    assert {p["recipe"]["clone"] for p in r["proposals"]} <= set(labels)


def test_more_than_twelve_clones_are_summarized_not_modeled():
    df = sheet({f"K{i:02d}": 3 for i in range(13)})
    r = _analyze(df, setup=setup(strain_column="clone"))
    s = r["strain_summary"]
    assert "clone" not in r["categorical_features"]
    assert s["available"] and s["modeled"] is False and len(s["levels"]) == 13
    assert "too_many_clones" in {x["code"] for x in s["refusals"]}
    assert all("clone" not in p["recipe"] for p in r["proposals"])
