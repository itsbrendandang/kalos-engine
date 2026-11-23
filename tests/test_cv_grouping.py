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


def _young_campaign(n_base: int, later: tuple[int, ...]):
    # Upload rows are round 0; each later round adds a couple of runs that rank
    # against the trend, so a round-0 fold fitted to them predicts round 0 badly.
    from kalos.core.evaluation import grouped_cv_report

    n = n_base + sum(later)
    rng = np.random.default_rng(0)
    X = rng.uniform(size=(n, 2))
    y = X @ np.array([2.0, -1.0]) + 0.05 * rng.normal(size=n)
    y[n_base:] = y[n_base:].mean() - (y[n_base:] - y[n_base:].mean())
    rounds = [0] * n_base + [r + 1 for r, k in enumerate(later) for _ in range(k)]
    clusters = np.arange(n)
    calls: list[int] = []

    def run_cv(g):
        calls.append(len(set(np.asarray(g).tolist())))
        return grouped_cv_report(X, y, groups=g, n_splits=5, n_boot=200)

    prim = run_cv(clusters)
    cands = structure_candidates(pd.DataFrame(X, columns=["a", "b"]), exclude=[], round_index=rounds,
                                 clusters=clusters)
    assert [c["basis"] for c in cands] == ["campaign_round"]
    return prim, clusters, cands, run_cv, n, calls


@pytest.mark.parametrize("n_base,later", [(30, (2, 1)), (30, (2, 2)), (160, (2, 1)), (160, (2, 2))])
def test_a_round_too_big_to_train_without_does_not_set_the_score(n_base, later):
    # QA: 162 -> 163 rows (rounds 160/2/1) moved rho 0.65 -> -0.50; at 160/2/2 every
    # run is held out and a 3-group interval exists, but round 0 is still predicted
    # from 4 rows.
    prim, clusters, cands, run_cv, n, calls = _young_campaign(n_base, later)
    assert prim["n_oof"] == n and prim["spearman"] > 0.5
    calls.clear()
    block, rep, _g, basis = leak_free_cv(prim, clusters, "replicate_clusters", None, cands, run_cv,
                                         n_exact_recipes=n, n_clusters=n, strict_candidates=True)
    assert basis == "replicate_clusters" and rep is prim
    rnd = block["candidates"][1]
    assert rnd["evaluable"] is False and rnd["used"] is False and rnd["spearman"] is None
    assert f"leaves {sum(later)} runs to train on" in rnd["reason"]
    assert any(w["code"] == "structure_not_checkable" and w["column"] == "round" for w in block["warnings"])
    assert calls == []  # not cross-validated at all

    # The old rule (flag off): the round-0 fold fitted to the few new runs sets the score.
    _block, old_rep, _g, old_basis = leak_free_cv(prim, clusters, "replicate_clusters", None, cands, run_cv,
                                                  n_exact_recipes=n, n_clusters=n, strict_candidates=False)
    assert old_basis == "campaign_round" and old_rep["spearman"] < prim["spearman"] - 0.3


@pytest.mark.parametrize("n_base,later", [(12, (6, 6)), (20, (5, 5, 5))])
def test_a_round_grouping_with_enough_runs_outside_round_0_still_binds(n_base, later):
    # 20/5/5/5: round 0 outweighs the rest but its fold still trains on 15 runs.
    prim, clusters, cands, run_cv, n, _calls = _young_campaign(n_base, later)
    block, rep, _g, basis = leak_free_cv(prim, clusters, "replicate_clusters", None, cands, run_cv,
                                         n_exact_recipes=n, n_clusters=n, strict_candidates=True)
    rnd = block["candidates"][1]
    assert rnd["evaluable"] is True and rnd["reason"] is None
    assert basis == "campaign_round" and rep["spearman"] < prim["spearman"]


def test_a_candidate_whose_cv_skipped_a_fold_does_not_bind():
    # 6 runs in lots of 3/2/1: no lot outweighs the rest, but holding out the
    # first lot leaves 3 training runs, so _oof skips that fold and only 3 runs
    # get a held-out prediction.
    prim = _fake_rep(0.6, n=6, n_oof=6)
    cands = [{"basis": "column", "column": "lot", "groups": np.array([0, 0, 0, 1, 1, 2]), "n_levels": 3,
              "merged_with_replicates": True}]
    partial = {"spearman": -0.5, "ci95": (float("nan"), float("nan")), "n_groups": 2, "n_oof": 3}
    block, rep, _g, basis = leak_free_cv(prim, np.arange(6), "replicate_clusters", None, cands, lambda g: partial,
                                         n_exact_recipes=6, n_clusters=6, strict_candidates=True)
    assert basis == "replicate_clusters" and rep is prim
    assert "only 3 of 6 runs were held out" in block["candidates"][1]["reason"]


def test_a_full_coverage_candidate_without_an_interval_does_not_bind():
    # Every resample degenerate (no variation to rank) leaves no interval even with 3+ groups.
    prim = _fake_rep(0.6)
    cands = [{"basis": "column", "column": "lot", "groups": np.repeat(np.arange(3), 8), "n_levels": 3,
              "merged_with_replicates": True}]
    no_ci = {"spearman": -0.3, "ci95": (float("nan"), float("nan")), "n_groups": 3, "n_oof": 24}
    block, rep, _g, basis = leak_free_cv(prim, np.arange(24), "replicate_clusters", None, cands, lambda g: no_ci,
                                         n_exact_recipes=24, n_clusters=24, strict_candidates=True)
    assert basis == "replicate_clusters" and rep is prim
    assert block["candidates"][1]["reason"].startswith("no 95% interval")


@pytest.mark.parametrize("raw,expected", [(None, True), ("", True), ("1", True), ("On", True), ("off", False),
                                          ("0", False), ("False", False)])
def test_strict_candidates_runtime_switch(monkeypatch, raw, expected):
    from kalos.portal.analysis_parts.cv_grouping import (
        STRICT_CANDIDATES_ENV,
        strict_candidates_default,
    )

    if raw is None:
        monkeypatch.delenv(STRICT_CANDIDATES_ENV, raising=False)
    else:
        monkeypatch.setenv(STRICT_CANDIDATES_ENV, raw)
    assert strict_candidates_default() is expected


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


def test_strict_candidates_typo_is_refused(monkeypatch):
    from kalos.envconfig import ConfigError
    from kalos.portal.analysis_parts.cv_grouping import (
        STRICT_CANDIDATES_ENV,
        strict_candidates_default,
    )
    from kalos.portal.config import check_env_settings

    monkeypatch.setenv(STRICT_CANDIDATES_ENV, "flase")
    with pytest.raises(ConfigError, match=STRICT_CANDIDATES_ENV):
        strict_candidates_default()
    with pytest.raises(ConfigError, match=STRICT_CANDIDATES_ENV):
        check_env_settings()


def test_a_shuffled_repeat_that_pairs_round_0_with_another_round_does_not_set_the_score():
    # 160 + six rounds of 2: the unshuffled partition holds out round 0 alone and
    # trains on 12, but the shuffled repeat (seed 1) holds out round 0 with one
    # more round and trains on 10.
    from kalos.core.evaluation import cv_partitions

    prim, clusters, cands, run_cv, n, calls = _young_campaign(160, (2,) * 6)
    g = np.asarray(cands[0]["groups"])
    parts = cv_partitions(np.zeros((n, 1)), np.zeros(n), g, n_splits=5, n_repeats=2)
    assert min(len(tr) for tr, _te in parts[0]) == 12
    assert min(len(tr) for tr, _te in parts[1]) == 10
    calls.clear()
    block, rep, _g, basis = leak_free_cv(prim, clusters, "replicate_clusters", None, cands, run_cv,
                                         n_exact_recipes=n, n_clusters=n, strict_candidates=True)
    rnd = block["candidates"][1]
    assert basis == "replicate_clusters" and rep is prim
    assert rnd["evaluable"] is False and "leaves 10 runs to train on" in rnd["reason"]
    assert calls == []
