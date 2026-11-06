"""How often does the stopping advice say "gains within noise" too early?

A small closed-loop simulation for `kalos.core.stopping`. Synthetic only; no
real data. For each seed: a 2-input bump objective on [0, 10]^2 with Gaussian
assay noise (sd `NOISE_SD`), 6 random recipes run in triplicate, then `ROUNDS`
rounds of: fit the replicate-aware GP (pooled noise floor fed as sigma^2 / n),
read the stopping advice, run the model optimum and a batch of 2 more recipes in
triplicate. A FALSE STOP is a round where the advice says `gains_within_noise`
while the true optimum is still more than one noise sd above the true value of
the best measured recipe (by replicate mean).

Run: `python -c "from kalos.bench.stopping_sim import run; print(run())"`.
"""
from __future__ import annotations

import numpy as np

from kalos.core.optimize import posterior_mean_optimum, propose_batch
from kalos.core.replicates import aggregate_replicates, noise_report
from kalos.core.stopping import stopping_advice
from kalos.core.surrogate import Surrogate

NOISE_SD = 0.5
ROUNDS = 6
REPS = 3
BOUNDS = np.array([[0.0, 0.0], [10.0, 10.0]])
F_STAR = 10.0


def objective(X: np.ndarray) -> np.ndarray:
    X = np.atleast_2d(X)
    return F_STAR * np.exp(-((X[:, 0] - 7.0) ** 2 + (X[:, 1] - 3.0) ** 2) / 8.0)


def _measure(X: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    Xr = np.repeat(np.atleast_2d(X), REPS, axis=0)
    return Xr, objective(Xr) + rng.normal(0, NOISE_SD, Xr.shape[0])


def run_seed(seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    X, y = _measure(rng.uniform(0, 10, (6, 2)), rng)
    out = []
    for rnd in range(ROUNDS):
        Xu, ym, _v, n = aggregate_replicates(X, y)
        nr = noise_report(X, y)
        s = Surrogate().fit(Xu, ym, bounds=BOUNDS, noise=nr["noise_var"] / n)
        opt = posterior_mean_optimum(s, BOUNDS, seed=seed)
        batch = propose_batch(s, BOUNDS, q=2, seed=seed).X
        i = int(np.argmax(ym))
        st = stopping_advice(
            s, Xu[i], float(ym[i]), np.vstack([opt.reshape(1, -1), batch]),
            float(np.sqrt(nr["noise_var_pooled"])), seed=seed,
        )
        gap = F_STAR - float(objective(Xu[i])[0])
        out.append({"seed": seed, "round": rnd, "state": st["state"],
                    "p": st["p_gain_beyond_noise"], "true_gap": round(gap, 3)})
        Xn, yn = _measure(np.vstack([opt.reshape(1, -1), batch]), rng)
        X, y = np.vstack([X, Xn]), np.r_[y, yn]
    return out


def run(seeds: int = 10) -> dict:
    rows = [r for s in range(seeds) for r in run_seed(s)]
    stops = [r for r in rows if r["state"] == "gains_within_noise"]
    false = [r for r in stops if r["true_gap"] > NOISE_SD]
    return {
        "seeds": seeds,
        "rounds_per_seed": ROUNDS,
        "n_snapshots": len(rows),
        "n_gains_within_noise": len(stops),
        "n_false_stops": len(false),
        "false_stop_rate": None if not stops else round(len(false) / len(stops), 3),
        "rows": rows,
    }
