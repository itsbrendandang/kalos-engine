"""Exact copies are not replicates (review M1).

A replicate is a distinct run at the same factor settings, with its own measured
target. An exact copy (the same row pasted twice) repeats one measurement. Counted
as a replicate it gave zero within-recipe variance, so noise_sd collapsed to 0 and
the ICC to exactly 1.0: the sheet looked noise-free when it was not measured twice.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kalos.core.replicates import heteroscedasticity_report, noise_report


def test_exact_copies_alone_are_not_replicates():
    X = np.array([[1.0], [1.0], [2.0], [2.0], [3.0], [3.0]])
    y = np.array([10.0, 10.0, 20.0, 20.0, 30.0, 30.0])
    nr = noise_report(X, y)
    assert nr["n_replicated"] == 0
    assert np.isnan(nr["noise_var"])
    assert np.isnan(nr["icc"])
    assert nr["n_exact_duplicates"] == 3


def test_distinct_runs_at_the_same_settings_are_still_replicates():
    X = np.array([[1.0], [1.0], [2.0], [3.0]])
    y = np.array([10.0, 12.0, 20.0, 30.0])
    nr = noise_report(X, y)
    assert nr["n_replicated"] == 1
    assert nr["noise_var"] == pytest.approx(2.0)  # var of [10, 12], ddof=1
    assert nr["n_exact_duplicates"] == 0


def test_a_copy_inside_a_true_replicate_group_does_not_add_noise_information():
    X = np.array([[1.0], [1.0], [1.0], [2.0], [3.0]])
    y = np.array([10.0, 10.0, 12.0, 20.0, 30.0])
    nr = noise_report(X, y)
    assert nr["n_exact_duplicates"] == 1
    assert nr["n_replicated"] == 1
    assert nr["noise_var"] == pytest.approx(2.0)


def test_heteroscedasticity_report_ignores_exact_copies():
    X = np.array([[1.0], [1.0], [2.0], [2.0]])
    y = np.array([10.0, 10.0, 20.0, 20.0])
    het = heteroscedasticity_report(X, y)
    assert het["n_replicated"] == 0


def test_the_engine_reports_no_noise_for_a_sheet_of_pasted_rows():
    pytest.importorskip("torch")
    from kalos.portal.analysis import _analyze

    rng = np.random.default_rng(5)
    ph = rng.uniform(6.0, 7.5, 12).round(3)
    glu = rng.uniform(5.0, 20.0, 12).round(2)
    titer = (8 - 2 * (ph - 6.8) ** 2 + 0.05 * glu).round(4)
    base = pd.DataFrame({"pH": ph, "Glucose_g_L": glu, "titer": titer})
    df = pd.concat([base, base], ignore_index=True)  # every run pasted twice
    out = _analyze(df, target="titer")
    noise = out["noise"]
    assert noise["n_exact_duplicates"] == 12
    assert noise["icc"] != 1.0
    assert noise["noise_sd"] is None
