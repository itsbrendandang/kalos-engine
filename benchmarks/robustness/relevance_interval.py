"""Which interval for ARD relevance? Laplace at the MAP vs recipe subsampling.

Reproduces the table in the PR / docs/CAMPAIGNS_API.md "Driver second opinion":
a 5-input simulator where inputs 0 and 1 matter ("signal": sin(3 x0) + 0.8 x1;
"weak": 0.3 x0 + 0.3 x1) or nothing does ("noise"), noise sd 0.15, n = 20/40/80.
Each input gets a status from its lengthscale interval (relevant <= 1.0,
negligible >= 2.0, else unclear; kalos.core.gp_relevance thresholds). Counts are
printed per arm. Slow (GP fits); run on a quiet machine:

    python benchmarks/robustness/relevance_interval.py --seeds-laplace 30 --seeds-subsample 15
"""
from __future__ import annotations

import argparse
import copy
import warnings

import numpy as np

D = 5


def laplace_lengthscale_ci(surrogate, d: int, level: float = 0.95):
    """Laplace approximation of the log-lengthscale posterior at the MAP (the arm
    that was NOT shipped): central-difference Hessian of the negative log
    posterior over the raw parameters, mapped through the constraint transform."""
    import torch
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from scipy.stats import norm

    model = copy.deepcopy(surrogate.model)
    model.train()
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    sizes = [p.numel() for _n, p in params]
    theta0 = np.concatenate([p.detach().reshape(-1).numpy() for _n, p in params]).astype(float)
    n_data = int(model.train_targets.shape[-1])

    def grad(theta):
        off = 0
        with torch.no_grad():
            for (_n, p), k in zip(params, sizes):
                p.copy_(torch.as_tensor(theta[off:off + k], dtype=p.dtype).reshape(p.shape))
                off += k
        for _n, p in params:
            p.grad = None
        loss = -mll(model(*model.train_inputs), model.train_targets, *model.train_inputs).sum() * n_data
        loss.backward()
        return np.concatenate([p.grad.reshape(-1).numpy() for _n, p in params]).astype(float)

    H = np.zeros((theta0.size, theta0.size))
    for j in range(theta0.size):
        h = 1e-4 * max(1.0, abs(theta0[j]))
        tp, tm = theta0.copy(), theta0.copy()
        tp[j] += h
        tm[j] -= h
        H[:, j] = (grad(tp) - grad(tm)) / (2 * h)
    H = 0.5 * (H + H.T)
    try:
        cov = np.linalg.inv(H)
        np.linalg.cholesky(H)
    except np.linalg.LinAlgError:
        return None
    off, sl = 0, None
    for (n, _p), k in zip(params, sizes):
        if n.endswith("raw_lengthscale"):
            sl = np.arange(off, off + k)
        off += k
    raw = theta0[sl]  # botorch's lengthscale constraint has no transform: ls == raw
    mu, sd = np.log(raw), np.sqrt(np.diag(cov)[sl]) / raw
    z = norm.ppf(0.5 + level / 2)
    return np.column_stack([np.exp(mu - z * sd), np.exp(mu + z * sd)])


def _data(case: str, n: int, seed: int):
    rng = np.random.default_rng(1000 * n + seed)
    X = rng.uniform(0, 1, (n, D))
    if case == "signal":
        y = np.sin(3 * X[:, 0]) + 0.8 * X[:, 1] + rng.normal(0, 0.15, n)
    elif case == "weak":
        y = 0.3 * X[:, 0] + 0.3 * X[:, 1] + rng.normal(0, 0.15, n)
    else:
        y = rng.normal(0, 1, n)
    return X, y


def main() -> None:
    from kalos.core.gp_relevance import status_from_interval, subsample_relevance
    from kalos.core.surrogate import Surrogate

    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds-laplace", type=int, default=30)
    ap.add_argument("--seeds-subsample", type=int, default=15)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    box = np.array([[0.0] * D, [1.0] * D])
    for arm, n_seeds in (("laplace", args.seeds_laplace), ("subsample", args.seeds_subsample)):
        for case in ("signal", "weak", "noise"):
            for n in (20, 40, 80):
                rel = {"relevant": 0, "negligible": 0, "unclear": 0}
                irr = dict(rel)
                for seed in range(n_seeds):
                    X, y = _data(case, n, seed)
                    if arm == "laplace":
                        s = Surrogate().fit(X, y, bounds=box, restarts=3, restart_seed=seed)
                        ci = laplace_lengthscale_ci(s, D)
                        if ci is None:
                            continue
                        status = [status_from_interval(*ci[j]) for j in range(D)]
                    else:
                        status = subsample_relevance(X, y, bounds=box, seed=seed)["status"]
                    for j, st in enumerate(status):
                        (rel if (j < 2 and case != "noise") else irr)[st] += 1
                print(f"{arm:9s} {case:6s} n={n:2d} relevant inputs {rel}  irrelevant inputs {irr}", flush=True)


if __name__ == "__main__":
    main()
