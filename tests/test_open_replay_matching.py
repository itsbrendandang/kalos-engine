"""Matching and start logic of examples/open_replay_e2e.py on a tiny frame.

The replay must start the MATCHED row's own recipe and attribute that row's value to it.
Attributing a different row's value to a proposed recipe would fold imputed data into the fit.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

_PATH = Path(__file__).resolve().parents[1] / "examples" / "open_replay_e2e.py"


@pytest.fixture(scope="module")
def drv():
    spec = importlib.util.spec_from_file_location("open_replay_e2e", _PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def pool():
    return pd.DataFrame({"a": [0.0, 4.0, 8.0, 4.0], "c": ["x", "x", "y", "y"], "y": [1.0, 2.0, 3.0, 4.0]})


NORM = {"cat": {"c"}, "span": {"a": 8.0}}


def test_match_is_nearest_unused_with_exact_categorical(drv, pool):
    label, dist = drv.match_recipe({"a": 4.2, "c": "x"}, pool, "y", set(), NORM)
    assert label == 1 and dist == pytest.approx(0.2 / 8.0)
    label, dist = drv.match_recipe({"a": 4.2, "c": "x"}, pool, "y", {1}, NORM)
    assert label == 0 and dist == pytest.approx(4.2 / 8.0)  # next nearest unused x-row
    assert drv.match_recipe({"a": 4.0, "c": "z"}, pool, "y", set(), NORM)[0] is None


def test_started_recipe_is_the_matched_rows_values_not_the_proposal(drv, pool):
    srec = drv.started_recipe(pool, 1, ["a", "c"])
    assert srec == {"a": 4.0, "c": "x"}  # the row's own values
    assert "y" not in srec  # the target is never part of the started recipe


def test_plan_batch_starts_matched_rows_and_keeps_proposal_distance(drv, pool):
    props = [
        {"recipe": {"a": 4.2, "c": "x"}},   # matches row 1 (x, a=4)
        {"recipe": {"a": 4.2, "c": "x"}},   # same proposal again: must take a DIFFERENT row
        {"recipe": {"a": 1.0, "c": "z"}},   # categorical level not in the data: unmatched
    ]
    used = {99}
    plan = drv.plan_batch(props, pool, pool, "y", used, NORM)
    assert used == {99}  # plan_batch must not mutate the caller's used set
    assert [m[1] for m in plan["matched"]] == [1, 0]  # second identical proposal takes the next x-row
    for rec, label, srec, dist in plan["matched"]:
        assert srec == {"a": float(pool.loc[label, "a"]), "c": pool.loc[label, "c"]}
        assert dist >= 0
        assert srec != rec or dist == 0  # started differs from the proposal unless it was exact
    assert plan["unmatched"] == [({"a": 1.0, "c": "z"}, None)]
