"""QUEUE round 3 item 12: a 30-row replicate sheet with a text `recipe` label
(10 levels, one per pH/temperature setting) took 77-88 s. The label was fitted
as a categorical input (a mixed GP whose acquisition enumerates every label)
and also ran leave-one-group-out over groups that are exactly the recipes.

The label now groups replicates and folds only. These tests pin the structure
(deterministic) and keep a loose wall-clock bound (docs/DESKTOP.md, "Fit time")."""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("botorch")

from kalos.portal.analysis import _analyze


def _replicate_sheet(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(10):
        ph, temp = round(rng.uniform(6.0, 8.0), 2), round(rng.uniform(30.0, 38.0), 1)
        mean = 40 - 8 * (ph - 7.0) ** 2 - 0.3 * (temp - 34.0) ** 2
        for _ in range(3):
            rows.append({"recipe": f"R{i + 1}", "pH": ph, "temp_C": temp,
                         "Titer": round(mean + rng.normal(0, 0.5), 2)})
    return pd.DataFrame(rows)


def test_recipe_label_groups_replicates_and_is_not_an_input(monkeypatch):
    from kalos.core import surrogate

    fits = {"n": 0}
    real_fit = surrogate.Surrogate.fit

    def counting_fit(self, *a, **k):
        fits["n"] += 1
        return real_fit(self, *a, **k)

    monkeypatch.setattr(surrogate.Surrogate, "fit", counting_fit)
    t0 = time.perf_counter()
    out = _analyze(_replicate_sheet(), "Titer")
    elapsed = time.perf_counter() - t0

    assert out["group_col"] == "recipe"
    assert out["features"] == ["pH", "temp_C"]
    assert out["categorical_features"] == []
    assert out["proposal_optimizer"] == "continuous"
    status = {p["name"]: p["status"] for p in out["provenance"]}
    assert status["recipe"] == "dropped_recipe_label"
    assert out["cv_logo"]["spearman"] is None
    assert "labels exactly the recipes" in out["cv_logo"]["reason"]
    assert all("recipe" not in p["recipe"] for p in out["proposals"])
    # 2 CV repeats x 5 folds + the proposal fit; the label used to add 10 LOGO fits.
    assert fits["n"] <= 12, fits["n"]
    if not os.environ.get("KALOS_SKIP_TIMING"):
        # Measured about 6 s on an M3 Pro under load (was 60-88 s). Loose on purpose.
        assert elapsed < 45, f"30-row replicate sheet took {elapsed:.1f} s"


def test_declared_categorical_label_is_kept():
    from kalos.domains import ColumnRoles
    from kalos.portal.analysis import _recipe_label_columns

    roles = ColumnRoles.from_dict({"target": "Titer", "categoricals": ["recipe"]})
    assert _recipe_label_columns(roles, "recipe", ["recipe"]) == []
    assert _recipe_label_columns(None, "recipe", ["recipe"]) == ["recipe"]
    assert _recipe_label_columns(None, None, ["recipe"]) == []
    assert _recipe_label_columns(None, "lot", ["medium"]) == []


def test_a_text_factor_that_is_not_a_label_stays_an_input():
    df = _replicate_sheet()
    # `medium` is crossed with the recipes: A and B both appear at several settings.
    df["medium"] = ["A", "B", "A"] * 10
    out = _analyze(df, "Titer")
    assert "medium" in out["categorical_features"]
