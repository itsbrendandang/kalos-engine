"""Medium, strain, batch, lot ... are process FACTORS, not identifiers.

Review finding H1: `id_hint` classed `medium|strain|recipe|batch|campaign|group|lot`
as identifiers and `group_hint` made the same column the CV group. On a sheet
where Medium in {A, B, C} carried a real effect on titer, Medium was dropped as
`dropped_id`, became the leakage group, and no proposal ever varied it.

The fix: those names are candidate categorical factors (reported through the
`auto_categorical` finding), and only become the CV group when every level is
confounded with a single recipe, or when declared in `roles.groups`. Strict
identifiers (id, uuid, sample id, operator, ...) are still identifiers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from kalos.domains import BIOPROCESS_PROFILE  # noqa: E402
from kalos.portal.analysis import _analyze, is_identifier_name  # noqa: E402


def _medium_sheet() -> pd.DataFrame:
    """24 runs, Medium A/B/C with 8 runs each and a real effect (A 0, B +2, C +4)."""
    rng = np.random.default_rng(1)
    medium = ["A"] * 8 + ["B"] * 8 + ["C"] * 8
    effect = np.array([{"A": 0.0, "B": 2.0, "C": 4.0}[m] for m in medium])
    ph = rng.uniform(6.0, 7.5, 24)
    glucose = rng.uniform(5.0, 20.0, 24)
    titer = 5.0 + effect + 0.5 * ph + rng.normal(0.0, 0.2, 24)
    return pd.DataFrame({"Medium": medium, "pH": ph, "Glucose_g_L": glucose, "titer": titer})


@pytest.mark.parametrize("name", ["Medium", "Strain", "Recipe", "Batch", "Campaign", "Group", "Lot"])
def test_factor_names_are_not_identifiers(name):
    assert not is_identifier_name(name, BIOPROCESS_PROFILE.id_hint)


@pytest.mark.parametrize("name", ["batch_id", "Lot Number", "campaign_id", "Sample ID", "uuid"])
def test_compound_ids_and_people_are_still_identifiers(name):
    assert is_identifier_name(name, BIOPROCESS_PROFILE.id_hint)


def test_medium_with_a_real_effect_is_a_categorical_factor_not_a_dropped_id():
    out = _analyze(_medium_sheet(), target="titer")
    prov = {row["name"]: row for row in out["provenance"]}
    assert prov["Medium"]["status"] != "dropped_id"
    assert "Medium" in out["categorical_features"]
    assert "Medium" in out["auto_categorical"]
    assert any(f["check"] == "auto_categorical" and f["column"] == "Medium"
               for f in out["validation"]["findings"])


def test_medium_is_not_the_cv_group_when_levels_span_many_recipes():
    """Each Medium level has 8 different pH/glucose settings, so grouping folds by
    it would hide a factor the model must learn. It is not the group."""
    out = _analyze(_medium_sheet(), target="titer")
    assert out["group_col"] is None


def test_a_level_confounded_with_one_recipe_is_the_group():
    """Batch whose every level is one recipe repeated (a true replicate block):
    grouping by it cannot hide a factor, so it is used as the CV group."""
    df = pd.DataFrame({
        "Batch": ["X"] * 4 + ["Y"] * 4 + ["Z"] * 4,
        "pH": [6.0] * 4 + [6.5] * 4 + [7.0] * 4,
        "Glucose_g_L": [5.0] * 4 + [10.0] * 4 + [15.0] * 4,
        "titer": [1.0, 1.1, 0.9, 1.0, 2.0, 2.1, 1.9, 2.0, 3.0, 3.1, 2.9, 3.0],
    })
    out = _analyze(df, target="titer")
    assert out["group_col"] == "Batch"
