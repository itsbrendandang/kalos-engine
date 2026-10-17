"""Loader + trial tests for kalos.bench.amii, on a tiny fake of the paper's CSV layout
(the real datasets are not redistributed and are never read here)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kalos.bench.amii import (
    REGISTRY, amii_data_dir, benchmark_dataset, load_amii_pool, paired_comparison, run_trial, runs_to_top,
)


@pytest.fixture
def fake_dir(tmp_path):
    rng = np.random.default_rng(0)
    n = 40
    df = pd.DataFrame(rng.uniform(size=(n, 3)), columns=["Feature_0", "Feature_1", "Feature_2"])
    df["const"] = 1.0
    df["Label"] = -((df[["Feature_0", "Feature_1", "Feature_2"]] - 0.5) ** 2).sum(1)
    df["Label_var"] = 0.01
    df.insert(0, "Unnamed: 0", range(n))
    dup = df.iloc[:5].copy()
    dup["Label"] += 0.2  # replicate rows with identical settings
    dup["Fidelity"] = 0
    df["Fidelity"] = 3
    pd.concat([df, dup]).to_csv(tmp_path / "MOBO_dataset_rat_myocyte.csv", index=False)
    return tmp_path


def test_missing_data_raises_with_instructions(monkeypatch, tmp_path):
    monkeypatch.delenv("KALOS_AMII_DATA", raising=False)
    monkeypatch.delenv("KALOS_MEDIA_DATA", raising=False)
    with pytest.raises(FileNotFoundError, match="KALOS_AMII_DATA"):
        load_amii_pool("DBO_rat_myocyte")
    assert amii_data_dir(tmp_path / "nope") is None


def test_loader_uses_env_dir_top_fidelity_and_drops_constants(fake_dir, monkeypatch):
    monkeypatch.setenv("KALOS_AMII_DATA", str(fake_dir))
    X, y, info = load_amii_pool("MOBO_rat_myocyte")
    assert info["rows_in_file"] == 45 and info["rows_after_fidelity"] == 40
    assert X.shape == (40, 3) and "const" not in info["features"]
    assert np.isfinite(X).all() and np.isfinite(y).all()


def test_registry_covers_six_datasets():
    assert len(REGISTRY) == 6


def test_runs_to_top():
    assert runs_to_top([1, 2, 5, 5], 5) == 3 and runs_to_top([1, 2], 5) is None


def test_trial_shared_init_repeatable_and_labels_hidden(monkeypatch):
    from kalos.core.surrogate import Surrogate

    rng = np.random.default_rng(1)
    X = rng.uniform(size=(60, 3))
    y = -((X - 0.5) ** 2).sum(1)
    seen = []
    real_fit = Surrogate.fit

    def spy(self, Xf, yf, *a, **k):
        seen.append(len(yf))
        return real_fit(self, Xf, yf, *a, **k)

    monkeypatch.setattr(Surrogate, "fit", spy)
    a = run_trial(X, y, q=2, budget=8, n_init=6, seed=3)
    assert seen == [6, 8, 10, 12]  # fit sees only labels revealed so far
    monkeypatch.setattr(Surrogate, "fit", real_fit)
    b = run_trial(X, y, q=2, budget=8, n_init=6, seed=3)
    assert a == b
    assert {arm: len(c) for arm, c in a.items()} == {"bo": 14, "random": 14, "lhs": 14}
    assert a["bo"][5] == a["random"][5] == a["lhs"][5]  # same initial design
    for c in a.values():
        assert all(q2 >= q1 for q1, q2 in zip(c, c[1:]))


def test_benchmark_requires_five_seeds_and_reports_spread():
    rng = np.random.default_rng(2)
    X = rng.uniform(size=(50, 2))
    y = X.sum(1)
    with pytest.raises(ValueError, match="at least 5 seeds"):
        benchmark_dataset(X, y, q=1, budget=5, seeds=[0, 1])
    r = benchmark_dataset(X, y, q=4, budget=8, n_init=5, seeds=range(5))
    assert set(r["arms"]) == {"bo", "random", "lhs"}
    for v in r["arms"].values():
        assert v["q25"] <= v["median"] <= v["q75"] and v["n_seeds"] == 5


def test_paired_comparison_counts_and_sign_test():
    r = paired_comparison([5, 5, 5, 5, None, 9], [8, 7, 6, 5, 9, 9], cap=71)
    assert (r["bo_fewer"], r["bo_more"], r["tied"]) == (3, 1, 2)
    assert 0.0 < r["sign_test_p"] <= 1.0
    assert paired_comparison([4, 4], [4, 4], cap=9)["sign_test_p"] == 1.0  # no information


def test_lhs_breaks_grid_ties_at_random_not_by_file_order():
    # A 4-level grid has many exact nearest-neighbour ties; LHS picks must vary with the seed.
    g = np.array([[a, b] for a in range(4) for b in range(4) for _ in range(3)], float)
    y = np.arange(len(g), dtype=float)
    picks = {tuple(run_trial(g, y, q=1, budget=6, n_init=4, seed=s, arms=("lhs",))["lhs"]) for s in range(6)}
    assert len(picks) > 1


def test_trial_reports_fit_fallbacks(monkeypatch):
    from kalos.core.surrogate import FitError, Surrogate

    def boom(self, *a, **k):
        raise FitError("forced")

    monkeypatch.setattr(Surrogate, "fit", boom)
    rng = np.random.default_rng(1)
    X = rng.uniform(size=(40, 2))
    st: dict = {}
    run_trial(X, X.sum(1), q=2, budget=6, n_init=4, seed=0, arms=("bo",), stats=st)
    assert st["fit_fallbacks"] == 3  # 6 evaluations in batches of 2
