from __future__ import annotations

import pytest

from kalos.bench.mechanistic_benchmark import LEVELS, benchmark


def test_benchmark_structure_and_requires_five_seeds():
    with pytest.raises(ValueError, match="at least 5 distinct seeds"):
        benchmark(seeds=[0, 1])
    r = benchmark(seeds=range(5), budget=4)
    assert set(r["summary"]) == {"plain", *LEVELS}
    assert set(r["hybrid_vs_plain_evals_to_95pct"]) == set(LEVELS)
    for m in r["summary"].values():
        t = m["evals_to_95pct"]
        assert 0.0 <= t["reach_rate"] <= 1.0
        if t["median"] is not None:
            assert t["q25"] <= t["median"] <= t["q75"]


def test_a_correct_prior_never_hurts_and_an_oracle_helps():
    r = benchmark(seeds=range(5), budget=4)
    assert r["hybrid_vs_plain_evals_to_95pct"]["oracle"]["worse"] == 0
    assert r["summary"]["oracle"]["rmse_norm"]["median"] < r["summary"]["plain"]["rmse_norm"]["median"]


def test_compare_reach_treats_both_missed_as_a_tie_and_one_sided_reach_as_a_win():
    from kalos.bench.mechanistic_benchmark import _compare_reach

    assert _compare_reach(None, None) == 0
    assert _compare_reach(5, None) == 1 and _compare_reach(None, 5) == -1
    assert _compare_reach(4, 6) == 1 and _compare_reach(6, 4) == -1 and _compare_reach(5, 5) == 0


def test_rmse_uses_one_fixed_split_shared_by_all_strategies(monkeypatch):
    import numpy as np

    from kalos.bench import mechanistic_benchmark as mb

    seen = []
    real = mb._fit_predict

    def spy(kind, Xtr, ytr, Xte, bounds):
        seen.append((kind, Xte.shape[0], Xte.copy()))
        return real(kind, Xtr, ytr, Xte, bounds)

    monkeypatch.setattr(mb, "_fit_predict", spy)
    X, y = mb.make_pool()
    mb.run_seed(0, X, y, budget=2)
    finals = {kind: Xte for kind, n, Xte in seen if n == int(0.3 * len(y))}
    assert set(finals) == {"plain", *mb.LEVELS}
    first = finals["plain"]
    assert all(np.array_equal(first, v) for v in finals.values())  # identical held-out rows for every strategy


def test_saved_result_file_matches_what_the_docs_claim():
    import json
    from pathlib import Path

    r = json.loads((Path(__file__).resolve().parents[1] / "examples/results/mechanistic_benchmark_10seeds.json").read_text())
    t = r["hybrid_vs_plain_evals_to_95pct"]
    assert len(r["seeds"]) == 10
    assert t["oracle"]["worse"] == 0 and t["oracle"]["better"] > t["mild"]["worse"]  # correct prior never hurts
    assert t["mild"]["better"] > t["mild"]["worse"]
    assert t["adversarial"]["better"] <= t["adversarial"]["worse"] + 1  # a wrong prior does not reliably help
    assert r["summary"]["plain"]["evals_to_95pct"]["reach_rate"] < 1.0  # right-censoring is reported, not hidden
