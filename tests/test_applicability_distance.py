"""Distance-to-data per proposal (compmodel PLAN E11): a number, no verdict.

`proposals[].nearest_training_distance` is the Euclidean distance to the nearest
training row with each continuous feature divided by its design-box width and a
categorical mismatch counted as 1. `training_nn_distance_median` gives the same
metric between training rows, for context.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from kalos.portal.analysis_parts.proposals import (
    nearest_training_distance,
    training_nn_distance_median,
)

BOUNDS = np.array([[0.0, 10.0], [1.0, 20.0]])


def test_a_proposal_at_a_training_row_is_zero_and_far_is_large():
    X = np.array([[0.0, 10.0], [1.0, 20.0], [0.5, 15.0]])
    batch = np.array([[0.5, 15.0], [0.0, 20.0]])
    d = nearest_training_distance(batch, X, BOUNDS)
    assert d[0] == 0.0
    # nearest is (0.5, 15): half a box-width on each feature
    assert d[1] == pytest.approx(np.sqrt(0.5))


def test_units_are_box_widths():
    X = np.array([[0.0, 10.0]])
    d = nearest_training_distance(np.array([[0.1, 12.0]]), X, BOUNDS)
    assert d[0] == pytest.approx(np.hypot(0.1, 0.2))


def test_categorical_mismatch_counts_one():
    b = np.array([[0.0, 10.0, 0.0], [1.0, 20.0, 1.0]])
    X = np.array([[0.0, 10.0, 1.0]])
    d = nearest_training_distance(np.array([[0.0, 10.0, 0.0]]), X, b, cat_dims=[2])
    assert d[0] == 1.0


def test_training_median_handles_duplicates_and_chunks():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (300, 2)) * [1, 10] + [0, 10]
    m = training_nn_distance_median(X, BOUNDS)
    assert 0 < m < 0.2
    assert training_nn_distance_median(np.vstack([X[:1], X[:1]]), BOUNDS) == 0.0
    assert training_nn_distance_median(X[:1], BOUNDS) is None


@pytest.mark.parametrize("anonymize", [False, True])
def test_analysis_returns_the_field_without_verdict_words(anonymize):
    from kalos.portal.pipeline import compute_upload_analysis

    rng = np.random.default_rng(1)
    n = 24
    df = pd.DataFrame({"Methanol": rng.uniform(0, 4, n).round(3), "pH": rng.uniform(5, 7, n).round(2)})
    df["lipase_titer"] = (1.5 * df.Methanol - (df.pH - 6) ** 2 + rng.normal(0, 0.1, n)).round(3)
    _, r = compute_upload_analysis(df.to_csv(index=False).encode(), "lipase_titer", anonymize, "x.csv")
    assert r["proposals"]
    for p in r["proposals"]:
        assert isinstance(p["nearest_training_distance"], float) and p["nearest_training_distance"] >= 0
    assert isinstance(r["training_nn_distance_median"], float)
    text = json.dumps(r["proposals"]).lower()
    # The `applicability` flags (tests/test_applicability_flags.py) are factual
    # labels; verdict words stay out.
    for word in ("out of domain", "out-of-domain", "extrapolat", "unsafe", "trusted"):
        assert word not in text
