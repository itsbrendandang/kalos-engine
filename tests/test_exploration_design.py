"""Exploration design when the model cannot rank recipes yet (QUEUE round 3
item 6): kalos/core/space_filling.py, kalos/portal/analysis_parts/exploration.py."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from kalos.core.space_filling import maximin_batch
from kalos.portal.analysis_parts.exploration import batch_strategy, exploration_reason

BOX = np.array([[0.0, 0.0], [1.0, 1.0]])


def test_maximin_fills_the_empty_part_of_the_box():
    X = np.array([[0.1, 0.1], [0.2, 0.15], [0.15, 0.2]])  # all in one corner
    out = maximin_batch(X, BOX, 3, seed=1)
    assert out.shape == (3, 2)
    assert np.all((out >= 0) & (out <= 1))
    d = np.sqrt(((out[:, None, :] - X[None]) ** 2).sum(-1)).min(1)
    assert d.min() > 0.4  # far from the tried corner
    again = maximin_batch(X, BOX, 3, seed=1)
    assert np.array_equal(out, again)  # seeded


def test_maximin_respects_pending_categoricals_and_a_filter():
    X = np.array([[0.5, 0.0], [0.5, 1.0]])
    box = np.array([[0.0, 0.0], [1.0, 1.0]])
    out = maximin_batch(X, box, 4, cat_dims=[1], cat_cardinalities=[2], seed=2,
                        pending=np.array([[0.0, 0.0]]))
    assert set(np.unique(out[:, 1])) <= {0.0, 1.0}
    assert not any(np.allclose(r, [0.0, 0.0]) for r in out)
    only_left = maximin_batch(X, box, 3, seed=2, feasible=lambda C: C[:, 0] < 0.3)
    assert np.all(only_left[:, 0] < 0.3)


def test_exploration_rule_table():
    ok = {"state": "reliable", "ordering": {"state": "ok"}}
    weak = {"state": "not_yet", "ordering": {"state": "weak"}}
    small = {"state": "not_yet", "ordering": {"state": "ok"}}
    refused = {"state": "refused", "ordering": {"state": "absent"}}
    assert exploration_reason(ok, n_distinct=40, n_inputs=4, setting=None) is None
    assert exploration_reason(ok, n_distinct=7, n_inputs=4, setting=None) == "few_recipes"
    assert exploration_reason(weak, n_distinct=40, n_inputs=4, setting=None) == "cannot_rank_yet"
    assert exploration_reason(small, n_distinct=40, n_inputs=4, setting="auto") is None
    assert exploration_reason(refused, n_distinct=3, n_inputs=4, setting=None) is None
    assert exploration_reason(weak, n_distinct=3, n_inputs=4, setting="off") is None
    s = batch_strategy("few_recipes", n_distinct=7, n_inputs=4)
    assert s["kind"] == "exploration_design" and "7 different recipes for 4 inputs" in s["plain"]
    assert batch_strategy(None, n_distinct=40, n_inputs=4)["kind"] == "model"


def _sheet(n: int, d: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 10, (n, d)).round(2)
    y = (X[:, 0] - 0.5 * (X[:, 1] - 5) ** 2 + rng.normal(0, 0.3, n)).round(3)
    df = pd.DataFrame(X, columns=[f"x{i}_g_L" for i in range(d)])
    df["Titer"] = y
    return df


def test_few_recipes_give_an_exploration_design_with_a_best_bet():
    pytest.importorskip("botorch")
    from kalos.domains import ColumnRoles
    from kalos.portal.analysis import _analyze

    df = _sheet(9, 5)  # 9 recipes, 5 inputs: fewer than 2 x 5
    roles = ColumnRoles.from_dict({"target": "Titer", "features": list(df.columns[:-1])})
    out = _analyze(df, roles=roles)
    json.dumps(out["batch_strategy"], allow_nan=False)
    if out["verdict"]["state"] == "refused":
        assert out["proposals"] == []
        return
    assert out["batch_strategy"]["kind"] == "exploration_design"
    assert out["batch_strategy"]["reason"] == "few_recipes"
    picks = [p["pick"] for p in out["proposals"]]
    assert picks[0] == "best_bet" and set(picks[1:]) == {"exploration_design"}
    assert all(p["mode"] == "explore" for p in out["proposals"][1:])

    off = ColumnRoles.from_dict({"target": "Titer", "features": list(df.columns[:-1]),
                                 "exploration_design": "off"})
    out_off = _analyze(df, roles=off)
    assert out_off["batch_strategy"]["kind"] == "model"


def test_bad_setting_is_a_coded_400():
    from kalos.domains import ColumnRoles, InvalidExplorationDesign

    with pytest.raises(InvalidExplorationDesign):
        ColumnRoles.from_dict({"target": "y", "exploration_design": "sometimes"})
