"""F05: cross-validation that cannot leak across replicates, batches, rounds or scales.

Fast unit tests on `kalos.portal.analysis_parts.cv_grouping` and the campaign
round index, plus integration checks through `_analyze` on the audit's two
leak scenarios (batch offsets with a text batch column; as-run replicates).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kalos.portal.analysis_parts.cv_grouping import (
    NEAR_DUPLICATE_TOL,
    is_batch_like,
    is_scale_like,
    leak_free_cv,
    replicate_clusters,
    round_index_from_attrs,
    same_partition,
    structure_candidates,
    union_groups,
)


def test_tolerance_matches_the_proposal_separation():
    from kalos.core.optimize import NEAR_DUPLICATE_TOL as OPT_TOL

    assert NEAR_DUPLICATE_TOL == OPT_TOL


@pytest.mark.parametrize("name", ["batch", "Batch_ID", "lot", "lot_no", "plate", "Plate 3", "run_date",
                                  "date", "week", "round", "block", "media_lot"])
def test_batch_like_names(name):
    assert is_batch_like(name)


@pytest.mark.parametrize("name", ["glucose_gL", "pH", "batch_volume_mL", "feed_day", "lotion", "around",
                                  "temperature_C", "Methanol"])
def test_not_batch_like(name):
    assert not is_batch_like(name)


def test_scale_like_names():
    assert is_scale_like("scale_L") and is_scale_like("working_volume") and is_scale_like("vessel")
    assert not is_scale_like("pH") and not is_scale_like("titer")


def test_jittered_replicates_cluster_and_distinct_rows_do_not():
    rng = np.random.default_rng(0)
    base = rng.uniform(0, 1, (15, 3))
    x = np.repeat(base, 4, axis=0) + rng.normal(0, 0.002, (60, 3))
    g = replicate_clusters(x, [1.0, 1.0, 1.0])
    assert len(set(g)) == 15
    true = np.repeat(np.arange(15), 4)
    assert same_partition(g, true)
    # canonical ids: first appearance order
    assert list(g[:8]) == [0, 0, 0, 0, 1, 1, 1, 1]


def test_widths_scale_the_tolerance():
    x = np.array([[0.0, 0.0], [3.0, 0.0]])
    assert len(set(replicate_clusters(x, [100.0, 1.0]))) == 1  # 3 of 100 = 0.03 < 0.05
    assert len(set(replicate_clusters(x, [10.0, 1.0]))) == 2   # 3 of 10 = 0.3


def test_complete_linkage_does_not_chain():
    # a chain of points 0.03 apart: single linkage would merge all of them
    x = np.arange(0, 0.31, 0.03).reshape(-1, 1)
    g = replicate_clusters(x, [1.0])
    assert len(set(g)) > 3


def test_missing_pattern_and_labels_keep_rows_apart():
    x = np.array([[1.0, np.nan], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    labels = pd.DataFrame({"medium": ["A", "A", "A", "B"]})
    g = replicate_clusters(x, [1.0, 1.0], labels)
    assert g[1] == g[2] and len({g[0], g[1], g[3]}) == 3


def test_union_and_same_partition():
    u = union_groups(["a", "a", "b", "c"], [0, 1, 1, 2])
    assert same_partition(u, [0, 0, 0, 1])
    assert same_partition(["x", "x", "y"], [5, 5, 7]) and not same_partition(["x", "y", "y"], [5, 5, 7])


def test_structure_candidates_from_columns_and_rounds():
    n = 24
    frame = pd.DataFrame({
        "pH": np.linspace(6, 7, n),
        "batch": [f"B{i // 4}" for i in range(n)],
        "run_id": [f"R{i}" for i in range(n)],          # an id: a new value every row -> not a batch
        "scale_L": [1.0] * 12 + [10.0] * 12,
        "titer": np.ones(n),
    })
    clusters = np.arange(n)
    c = structure_candidates(frame, exclude=["titer"], round_index=[i // 8 for i in range(n)], clusters=clusters)
    by = {(e["basis"], e["column"]): e for e in c}
    assert ("column", "batch") in by and by[("column", "batch")]["n_levels"] == 6
    assert ("scale_column", "scale_L") in by and by[("scale_column", "scale_L")]["n_levels"] == 2
    assert ("campaign_round", "round") in by and by[("campaign_round", "round")]["n_levels"] == 3
    assert not any(e["column"] == "run_id" for e in c)
    # merged with replicate clusters: a recipe measured in two batches joins them
    clusters2 = clusters.copy()
    clusters2[4] = clusters2[0]  # row 4 (batch B1) replicates row 0 (batch B0)
    c2 = structure_candidates(frame, exclude=["titer"], round_index=None, clusters=clusters2)
    batch = next(e for e in c2 if e["column"] == "batch")
    assert batch["merged_with_replicates"] and len(set(batch["groups"])) == 5


def test_round_index_aligns_to_the_fitted_rows():
    idx = pd.RangeIndex(6)
    assert round_index_from_attrs([0, 0, 0, 1, 1, 2], idx, pd.Index([0, 3, 5])) == [0, 1, 2]
    assert round_index_from_attrs([0, 1], idx, pd.Index([0])) is None   # wrong length
    assert round_index_from_attrs(None, idx, pd.Index([0])) is None


def _fake_rep(rho: float, n: int = 20, n_oof: int = 24) -> dict:
    return {"spearman": rho, "ci95": (rho - 0.2, rho + 0.2), "n_groups": n, "n_oof": n_oof}


def test_the_lowest_evaluable_score_binds():
    prim = _fake_rep(0.9)
    cands = [
        {"basis": "column", "column": "batch", "groups": np.repeat(np.arange(6), 4), "n_levels": 6,
         "merged_with_replicates": True},
        {"basis": "scale_column", "column": "scale_L", "groups": np.repeat(np.arange(2), 12), "n_levels": 2,
         "merged_with_replicates": True},
    ]
    seen = []

    def run_cv(g):
        seen.append(len(set(g)))
        return _fake_rep(-0.1, len(set(g)))

    block, rep, groups, basis = leak_free_cv(prim, np.arange(24), "replicate_clusters", None, cands, run_cv,
                                             n_exact_recipes=24, n_clusters=24)
    assert seen == [6]                      # the 2-level scale column is not cross-validated
    assert basis == "column" and rep["spearman"] == -0.1 and len(set(groups)) == 6
    assert block["used"] == {"basis": "column", "column": "batch", "n_groups": 6, "merged_with_replicates": True}
    codes = [w["code"] for w in block["warnings"]]
    assert "structure_not_checkable" in codes
    assert [c["used"] for c in block["candidates"]] == [False, True, False]


def test_too_few_groups_is_said_and_a_same_partition_is_not_rerun():
    prim = {"spearman": float("nan"), "ci95": (float("nan"), float("nan")), "n_groups": 2}
    same = [{"basis": "column", "column": "lot", "groups": np.array([0, 0, 1, 1]), "n_levels": 2,
             "merged_with_replicates": True}]

    def run_cv(g):  # pragma: no cover - must not be called
        raise AssertionError("re-ran an identical partition")

    block, _rep, _g, basis = leak_free_cv(prim, np.array([5, 5, 6, 6]), "replicate_clusters", None, same, run_cv,
                                          n_exact_recipes=4, n_clusters=2)
    assert basis == "replicate_clusters"
    codes = [w["code"] for w in block["warnings"]]
    assert "too_few_cv_groups" in codes and "near_duplicate_recipes_grouped" in codes
    assert block["candidates"][1]["same_partition_as"] == "replicate_clusters"


def test_a_candidate_whose_cv_skipped_a_fold_does_not_bind():
    # A young campaign: 30 upload rows (round 0), 2 rows in round 1, 1 in round 2.
    # Holding out round 0 leaves 3 training rows, so that fold is skipped and the
    # round candidate's score is a rank over 3 held-out rows with no interval.
    from kalos.core.evaluation import grouped_cv_report

    rng = np.random.default_rng(0)
    X = rng.uniform(size=(33, 2))
    y = X @ np.array([2.0, -1.0]) + 0.05 * rng.normal(size=33)
    y[30:] = y[30:].mean() - (y[30:] - y[30:].mean())  # the 3 new rows rank against the trend
    rounds = np.array([0] * 30 + [1] * 2 + [2])
    clusters = np.arange(33)

    def run_cv(g):
        return grouped_cv_report(X, y, groups=g, n_splits=5, n_boot=200)

    prim = run_cv(clusters)
    assert prim["n_oof"] == 33 and prim["spearman"] > 0.5
    cands = structure_candidates(pd.DataFrame(X, columns=["a", "b"]), exclude=[], round_index=rounds.tolist(),
                                 clusters=clusters)
    assert [c["basis"] for c in cands] == ["campaign_round"]

    block, rep, _g, basis = leak_free_cv(prim, clusters, "replicate_clusters", None, cands, run_cv,
                                         n_exact_recipes=33, n_clusters=33)
    assert basis == "replicate_clusters" and rep is prim
    rnd = block["candidates"][1]
    assert rnd["evaluable"] is False and rnd["used"] is False
    assert "only 3 of 33 runs were held out" in rnd["reason"]
    assert any(w["code"] == "structure_not_checkable" and w["column"] == "round" for w in block["warnings"])

    # The old rule, kept behind the flag: the 3-row negative score overrides the primary.
    _block, old_rep, _g, old_basis = leak_free_cv(prim, clusters, "replicate_clusters", None, cands, run_cv,
                                                  n_exact_recipes=33, n_clusters=33, require_full_coverage=False)
    assert old_basis == "campaign_round" and old_rep["n_oof"] == 3 and old_rep["spearman"] < 0


def test_a_full_coverage_candidate_without_an_interval_does_not_bind():
    prim = _fake_rep(0.6)
    cands = [{"basis": "column", "column": "lot", "groups": np.repeat(np.arange(3), 8), "n_levels": 3,
              "merged_with_replicates": True}]
    no_ci = {"spearman": -0.3, "ci95": (float("nan"), float("nan")), "n_groups": 2, "n_oof": 24}
    block, rep, _g, basis = leak_free_cv(prim, np.arange(24), "replicate_clusters", None, cands, lambda g: no_ci,
                                         n_exact_recipes=24, n_clusters=24)
    assert basis == "replicate_clusters" and rep is prim
    assert "too few for an interval" in block["candidates"][1]["reason"]


def test_campaign_fold_carries_the_round_index():
    from kalos.portal.campaign import base_row_rounds

    state = {"base_rows": [{}] * 9, "round": 2,
             "history": [{"round": 0, "n_base": 4}, {"round": 1, "n_base": 7}, {"round": 2, "n_base": 9}]}
    assert base_row_rounds(state) == [0, 0, 0, 0, 1, 1, 1, 2, 2]
    assert base_row_rounds({**state, "history": [{"round": 0, "n_base": 3}]}) is None   # rows do not match
    assert base_row_rounds({**state, "history": [{"round": 1, "n_base": 9}]}) is None   # does not start at 0
    assert base_row_rounds({"base_rows": [], "history": None}) is None


# --- integration through the analysis (the audit's c1/c2 sheets) -------------

def _batch_sheet(seed: int, *, numeric: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    xs, ys, bs = [], [], []
    for b in range(6):
        c = rng.uniform(0.15, 0.85, 3)
        xs.append(np.clip(c + rng.normal(0, 0.12, (8, 3)), 0, 1))
        ys.append(rng.normal(0, 1.0) + rng.normal(0, 0.2, 8))
        bs += [b + 1 if numeric else f"B{b + 1}"] * 8
    df = pd.DataFrame(np.vstack(xs).round(4), columns=["feed_a", "feed_b", "temp_offset"])
    df["batch"] = bs
    df["titer"] = np.concatenate(ys).round(4)
    return df


@pytest.mark.parametrize("numeric", [False, True])
def test_batch_column_is_a_cv_group_not_a_setpoint(numeric):
    from kalos.portal.analysis import _analyze

    r = _analyze(_batch_sheet(3, numeric=numeric), target="titer")
    assert "batch" not in r["features"]
    assert all("batch" not in p["recipe"] for p in r["proposals"])
    prov = {p["name"]: p["status"] for p in r["provenance"]}
    assert prov["batch"] == "dropped_cv_group"
    cg = r["cv_grouping"]
    cols = {c["column"] for c in cg["candidates"]}
    assert "batch" in cols
    assert cg["used"]["n_groups"] == r["cv_n_groups"]
    used = next(c for c in cg["candidates"] if c["used"])
    assert r["cv_spearman"] == used["spearman"]
    assert r["cv_spearman"] == min(c["spearman"] for c in cg["candidates"] if c["evaluable"])


def test_jittered_replicates_are_held_out_together():
    from kalos.portal.analysis import _analyze

    rng = np.random.default_rng(4)
    base = rng.uniform(0, 1, (15, 3))
    x = np.repeat(base, 4, axis=0) + rng.normal(0, 0.002, (60, 3))
    df = pd.DataFrame(x.round(5), columns=["feed_a", "feed_b", "temp_offset"])
    df["titer"] = (np.repeat(rng.normal(0, 1, 15), 4) + rng.normal(0, 0.2, 60)).round(4)
    r = _analyze(df, target="titer")
    cg = r["cv_grouping"]
    assert cg["n_exact_recipes"] == 60 and cg["n_replicate_clusters"] == 15
    assert r["cv_n_groups"] == 15
    assert "near_duplicate_recipes_grouped" in [w["code"] for w in cg["warnings"]]


def test_plan_fold_attaches_rounds_through_two_folds(tmp_path):
    from kalos.portal.campaign import CampaignStore

    store = CampaignStore(tmp_path)
    base = pd.DataFrame({"pH": [6.0, 6.5, 7.0, 7.5], "titer": [1.0, 2.0, 3.0, 2.5]})
    store.seed(base, "titer", ["pH"], tenant="t")
    for k in (1, 2):
        runs = store.start([{"recipe": {"pH": 6.1 + k / 10}}, {"recipe": {"pH": 6.9 + k / 10}}], tenant="t")
        for r in runs:
            store.set_result(r["id"], 2.0, tenant="t")
        df, _target, gen = store.plan_fold(tenant="t")
        expected = [0] * 4 + ([1, 1] if k == 2 else []) + [k, k]
        assert df.attrs["kalos_round_index"] == expected
        store.commit_fold(gen, tenant="t")
