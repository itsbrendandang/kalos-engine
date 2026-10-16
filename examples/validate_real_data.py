"""Validate BO optimization on real bioprocess data against synthetic benchmarks.

Loads examples/datasets/dataset_0.csv, runs BO/LHS/random optimization, and
produces a comparison report with the synthetic benchmark (gaussian_bump +
ackley). Outputs a JSON report comparing final_simple_regret, final_best, and
experiment count across strategies on both real and synthetic data.

Run:  python examples/validate_real_data.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

# Ensure the package is importable from this repo root.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from kalos.bench.benchmark import run_benchmark, summarize as bench_summarize
from kalos.bench.objectives import Objective, gaussian_bump, ackley


# ---------------------------------------------------------------------------
# Helpers for loading and preparing a CSV dataset for BO
# ---------------------------------------------------------------------------

def _load_real_dataset(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Load dataset_0.csv and return (X, y, bounds, column_meta).

    Column roles from the CSV header:
      - Continuous features: temp_c, ph, min_do_pct, inoculum_vol_ml, rpm,
        lactose_feed_ml, duration_days, total_media_vol_ml
      - Target (maximize): yield_titer_fpu_ml (we normalize yield_titer to 0..1
        so the surface is comparable across objectives)
      - contamination, total_media_vol_ml are treated as continuous features
        too (total_media_vol_ml has a narrow range and may matter).

    We normalise each column to [0, 1] so the BO loop works in a standard
    unit hypercube.
    """
    import pandas as pd

    df = pd.read_csv(path)

    # Select feature columns (continuous, no metadata columns)
    feature_cols = [
        "temp_c", "ph", "min_do_pct", "inoculum_vol_ml",
        "rpm", "lactose_feed_ml", "duration_days",
        "total_media_vol_ml",
    ]
    target_col = "yield_titer_fpu_ml"

    X = df[feature_cols].values.astype(np.float64)
    y = df[target_col].values.astype(np.float64)

    # Normalise to [0, 1] per column (zero-variance cols get 0..1)
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0  # avoid division by zero
    X_norm = (X - mins) / ranges

    # Normalise target to [0, 1]
    y_min, y_max = y.min(), y.max()
    y_range = y_max - y_min if y_max > y_min else 1.0
    y_norm = (y - y_min) / y_range

    bounds = np.vstack([np.zeros(X_norm.shape[1]), np.ones(X_norm.shape[1])])

    column_meta = {
        "features": feature_cols,
        "target": target_col,
        "n_rows": len(df),
        "n_features": len(feature_cols),
        "y_min": float(y_min),
        "y_max": float(y_max),
        "y_mean": float(y.mean()),
    }
    return X_norm, y_norm, bounds, column_meta


# ---------------------------------------------------------------------------
# Real-data BO benchmark
# ---------------------------------------------------------------------------

def _real_objective(X_norm: np.ndarray, y_norm: np.ndarray) -> Objective:
    """Create a black-box Objective from real data.

    Uses a simple GP-like surface: the objective function is the *maximum
    observed value in the training set*, so the BO loop tries to find the
    region around the best observation.  This is a simplification — in a
    real setting you would fit a GP and query the posterior, but here we
    just use the training data as a black-box evaluation oracle.

    The optimum is the true max of y_norm (the best yield in the dataset).
    """
    optimum = float(y_norm.max())

    def f(X: np.ndarray) -> np.ndarray:
        """Return y_true for a set of design points.

        We use a nearest-neighbour approximation: for each candidate X, find
        the closest training point and return its y value + a small random
        perturbation to simulate measurement noise.  This creates a surface
        worth navigating.
        """
        X = np.atleast_2d(X)
        # Use a simple distance-weighted average of nearby points
        dists = np.sqrt(((X_norm - X[:, None, :]) ** 2).sum(axis=-1))  # (q, n)
        # Weight by inverse distance (with a small epsilon to avoid division by zero)
        weights = 1.0 / (dists + 1e-6)
        weights /= weights.sum(axis=1, keepdims=True)
        y_pred = weights @ y_norm
        return y_pred.reshape(-1)

    dim = X_norm.shape[1]
    bounds = np.vstack([np.zeros(dim), np.ones(dim)])
    return Objective("real_data", dim, bounds, optimum, optimum, f)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _run_real_benchmark(X_norm: np.ndarray, y_norm: np.ndarray, **kw: Any) -> Dict[str, Any]:
    """Run BO/LHS/random on the real-data objective."""
    obj = _real_objective(X_norm, y_norm)
    result = run_benchmark(obj, **kw)
    return {
        "summary": bench_summarize(result, obj),
        "strategy_count": len(result),
    }


def main() -> None:
    dataset_path = _PROJECT_ROOT / "examples" / "datasets" / "dataset_0.csv"

    print(f"Loading dataset: {dataset_path}")
    X_norm, y_norm, bounds, meta = _load_real_dataset(dataset_path)
    print(f"  Rows: {meta['n_rows']}, Features: {meta['n_features']}")
    print(f"  Target range: [{meta['y_min']:.3f}, {meta['y_max']:.3f}], "
          f"mean: {meta['y_mean']:.3f}")

    benchmark_kw = {
        "budget": min(15, meta["n_rows"]),
        "n_init": min(5, meta["n_rows"]),
        "noise": 0.05,
        "seeds": range(10),
    }

    # --- Real data benchmark ---
    print("\nRunning BO benchmark on real data ...")
    real_result = _run_real_benchmark(X_norm, y_norm, **benchmark_kw)
    print(f"  Strategies tested: {real_result['strategy_count']}")

    # --- Synthetic benchmarks ---
    print("\nRunning BO benchmark on synthetic data (gaussian_bump, ackley) ...")
    syn_results: Dict[str, Any] = {}

    for obj_fn, name in [(gaussian_bump, "gaussian_bump_3d"), (ackley, "ackley_3d")]:
        obj = obj_fn(dim=3)
        res = run_benchmark(obj, **benchmark_kw)
        syn_results[name] = {
            "summary": bench_summarize(res, obj),
            "strategy_count": len(res),
        }

    # --- Assemble the comparison report ---
    report: Dict[str, Any] = {
        "real_data": {
            "metadata": meta,
            "benchmark": real_result,
        },
        "synthetic_benchmarks": syn_results,
    }

    # --- Pretty-print a summary ---
    print("\n" + "=" * 70)
    print("COMPARISON REPORT")
    print("=" * 70)

    for label, src in [("Real data", report["real_data"]),
                       ("Gaussian bump", syn_results.get("gaussian_bump_3d")),
                       ("Ackley", syn_results.get("ackley_3d"))]:
        print(f"\n{label}:")
        summary = (
            report["real_data"]["benchmark"]["summary"]
            if label == "Real data"
            else src["summary"]
        )
        for strat, stats in summary["per_strategy"].items():
            print(
                f"  {strat:6s}: final_simple_regret={stats['final_simple_regret']:.4f}, "
                f"final_best={stats['final_best_mean']:.4f}"
            )
        speedup = summary.get("bo_fewer_experiments_than_lhs")
        if speedup is not None:
            print(f"  BO saves {speedup} experiments vs LHS")

    # --- Write JSON report ---
    report_path = _PROJECT_ROOT / "examples" / "validate_real_data_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nFull report written to: {report_path}")


if __name__ == "__main__":
    main()
