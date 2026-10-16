from __future__ import annotations

import pytest

from kalos.bench.mechanistic_benchmark import LEVELS, benchmark


def test_benchmark_structure_and_requires_five_seeds():
    with pytest.raises(ValueError, match="at least 5 seeds"):
        benchmark(seeds=[0, 1])
    r = benchmark(seeds=range(5), budget=4)
    assert set(r["summary"]) == {"plain", *LEVELS}
    assert set(r["hybrid_vs_plain_evals_to_95pct"]) == set(LEVELS)
    for m in r["summary"].values():
        assert m["evals_to_95pct"]["q25"] <= m["evals_to_95pct"]["median"] <= m["evals_to_95pct"]["q75"]


def test_a_correct_prior_never_hurts_and_an_oracle_helps():
    r = benchmark(seeds=range(5), budget=4)
    assert r["hybrid_vs_plain_evals_to_95pct"]["oracle"]["worse"] == 0
    assert r["summary"]["oracle"]["rmse_norm"]["median"] < r["summary"]["plain"]["rmse_norm"]["median"]
