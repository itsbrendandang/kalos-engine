"""Head-to-head benchmark: Kalos vs random, LHS, Optuna TPE, Ax and vanilla GP-EI.

The protocol is pre-registered in `docs/BENCHMARK.md`. Read that file first: it fixes
the tasks, seeds, budgets, batch and pending settings, endpoints and statistics.
This module implements it. It does not change it.

    python -m kalos.bench --vs-baselines           # confirmatory run (30 seeds)
    python -m kalos.bench --vs-baselines --pilot   # timing only, seed 9001, writes nothing
    python -m kalos.bench --vs-baselines --quick   # 2 seeds, smoke test, writes nothing

Optional comparators need the `bench` extra (`pip install -e ".[ml,bench]"`).
They are imported only when their arm runs.
"""
from __future__ import annotations

import csv
import importlib.metadata as _md
import json
import logging
import multiprocessing as mp
import os
import platform
import subprocess
import time
import warnings
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats
from scipy.stats import qmc

from kalos.core.optimize import propose
from kalos.core.surrogate import Surrogate

from .objectives import (
    MixedObjective,
    Objective,
    ackley,
    bioprocess_like,
    gaussian_bump,
)

# ---- Fixed design constants (docs/BENCHMARK.md sections 3-5) ---------------------

CONFIRMATORY_SEEDS: tuple[int, ...] = tuple(range(1000, 1030))
PILOT_SEED = 9001
QUICK_SEEDS: tuple[int, ...] = (9002, 9003)  # smoke test only; never written
BOOT_SEED = 20261018
N_BOOT = 10_000
TIE_TOL = 0.01  # normalised regret; |d| below this is a tie (section 7)
TARGETS = (0.90, 0.75)  # fractions of the way from f_med to f* (section 6, S2)
HEADLINE_TARGET = 0.90
N_REF_SAMPLES_LOG2 = 16
REF_SEED = 12345

METHODS: tuple[str, ...] = ("kalos", "random", "lhs", "optuna_tpe", "ax", "botorch_gp_ei")
BASELINES: tuple[str, ...] = tuple(m for m in METHODS if m != "kalos")


@dataclass(frozen=True)
class Task:
    name: str
    index: int  # stream id for seeding; fixed, do not renumber
    objective: Objective | MixedObjective
    n_init: int
    q: int
    rounds: int
    depth: int  # pending pipeline depth: points of each batch left in flight
    noises: tuple[float, ...]

    @property
    def n_total(self) -> int:
        return self.n_init + self.rounds * self.q

    @property
    def box(self) -> np.ndarray:
        obj = self.objective
        return obj.bounds() if isinstance(obj, MixedObjective) else obj.bounds

    @property
    def dim(self) -> int:
        return int(self.box.shape[1])

    @property
    def cat_dims(self) -> list[int]:
        obj = self.objective
        return obj.cat_dims if isinstance(obj, MixedObjective) else []

    @property
    def cat_cardinalities(self) -> list[int]:
        obj = self.objective
        return list(obj.cat_cardinalities) if isinstance(obj, MixedObjective) else []

    @property
    def scale(self) -> float:
        return float(self.objective.scale)

    @property
    def f_star(self) -> float:
        return float(self.objective.optimum)


TASKS: dict[str, Task] = {
    "gaussian_4d": Task("gaussian_4d", 0, gaussian_bump(4), 5, 1, 18, 0, (0.0, 0.15)),
    "ackley_4d": Task("ackley_4d", 1, ackley(4), 5, 1, 18, 0, (0.0, 0.15)),
    "bioprocess_like_mixed": Task(
        "bioprocess_like_mixed", 2, bioprocess_like(), 8, 4, 5, 2, (0.15,)
    ),
}


def get_task(name: str) -> Task:
    if name not in TASKS:
        raise ValueError(f"unknown task {name!r}; expected one of {sorted(TASKS)}")
    return TASKS[name]


# ---- Shared designs, noise and references ------------------------------------------


def _lhs_design(task: Task, n: int, rng: np.random.Generator) -> np.ndarray:
    """LHS over the numeric dims; uniform integer codes over the categorical dims."""
    if isinstance(task.objective, MixedObjective):
        cont = task.objective.cont_bounds
        n_cont = cont.shape[1]
        u = qmc.LatinHypercube(d=n_cont, seed=rng).random(n)
        numeric = cont[0] + u * (cont[1] - cont[0])
        cats = np.column_stack([
            rng.integers(0, k, size=n).astype(float) for k in task.cat_cardinalities
        ])
        return np.hstack([numeric, cats])
    lo, hi = task.box[0], task.box[1]
    u = qmc.LatinHypercube(d=task.dim, seed=rng).random(n)
    return lo + u * (hi - lo)


def shared_init(task: Task, seed: int) -> np.ndarray:
    """The initial design every method starts from. Depends only on (task, seed)."""
    return _lhs_design(task, task.n_init, np.random.default_rng([seed, task.index, 555]))


def noise_vector(task: Task, eta: float, seed: int) -> np.ndarray:
    """Common-random-number noise draws, one per measured experiment (unscaled)."""
    eta_id = int(round(eta * 100))
    rng = np.random.default_rng([seed, task.index, eta_id, 777])
    return rng.standard_normal(task.n_total) * (eta * task.scale)


def true_value(task: Task, X: np.ndarray) -> np.ndarray:
    y_true, _ = task.objective.evaluate(X, noise=0.0)
    return np.asarray(y_true, dtype=float).reshape(-1)


def _reference_median(task: Task) -> float:
    """Median of the noiseless objective under 2^16 scrambled Sobol (or mixed-uniform) draws."""
    n = 2 ** N_REF_SAMPLES_LOG2
    rng = np.random.default_rng(REF_SEED)
    if isinstance(task.objective, MixedObjective):
        cont = task.objective.cont_bounds
        u = qmc.Sobol(d=cont.shape[1], scramble=True, seed=REF_SEED).random(n)
        numeric = cont[0] + u * (cont[1] - cont[0])
        cats = np.column_stack([
            rng.integers(0, k, size=n).astype(float) for k in task.cat_cardinalities
        ])
        X = np.hstack([numeric, cats])
    else:
        u = qmc.Sobol(d=task.dim, scramble=True, seed=REF_SEED).random(n)
        X = task.box[0] + u * (task.box[1] - task.box[0])
    return float(np.median(true_value(task, X)))


_F_MED_CACHE: dict[str, float] = {}


def f_med(task: Task) -> float:
    if task.name not in _F_MED_CACHE:
        _F_MED_CACHE[task.name] = _reference_median(task)
    return _F_MED_CACHE[task.name]


# ---- Arms ------------------------------------------------------------------------


def _uniform(task: Task, n: int, rng: np.random.Generator) -> np.ndarray:
    lo, hi = task.box[0], task.box[1]
    X = lo + rng.random((n, task.dim)) * (hi - lo)
    for dim, k in zip(task.cat_dims, task.cat_cardinalities):
        X[:, dim] = rng.integers(0, k, size=n).astype(float)
    return X


class _Arm:
    """Interface: `ask(q, pending) -> (q, d)` and `tell(X, y)`.

    `pending` is the set of issued-but-unmeasured points. Arms that handle pending
    points themselves (Optuna, Ax) track them internally and ignore the argument.
    """

    def __init__(self, task: Task, seed: int) -> None:
        self.task = task
        self.seed = seed
        self.round = 0

    def tell(self, X: np.ndarray, y: np.ndarray) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def ask(self, q: int, pending: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


class _KalosArm(_Arm):
    """Production path: Surrogate().fit + propose, with inferred noise and in-flight points."""

    def __init__(self, task: Task, seed: int) -> None:
        super().__init__(task, seed)
        self.X: list[np.ndarray] = []
        self.y: list[float] = []

    def tell(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X.extend(list(np.asarray(X, float)))
        self.y.extend(list(np.asarray(y, float).reshape(-1)))

    def ask(self, q: int, pending: np.ndarray) -> np.ndarray:
        self.round += 1
        cat_dims = self.task.cat_dims or None
        cards = self.task.cat_cardinalities or None
        s = Surrogate().fit(
            np.vstack(self.X), np.asarray(self.y), bounds=self.task.box, cat_dims=cat_dims
        )
        P = propose(
            s,
            self.task.box,
            q=q,
            cat_dims=cat_dims,
            cat_cardinalities=cards,
            pending=pending if pending.size else None,
            seed=self.seed * 100 + self.round,
        )
        return np.asarray(P, float).reshape(q, self.task.dim)


class _RandomArm(_Arm):
    def __init__(self, task: Task, seed: int) -> None:
        super().__init__(task, seed)
        self.rng = np.random.default_rng([seed, task.index, 557])

    def tell(self, X: np.ndarray, y: np.ndarray) -> None:
        return None

    def ask(self, q: int, pending: np.ndarray) -> np.ndarray:
        return _uniform(self.task, q, self.rng)


class _LHSArm(_Arm):
    def __init__(self, task: Task, seed: int) -> None:
        super().__init__(task, seed)
        n_rest = task.rounds * task.q
        self.pool = _lhs_design(task, n_rest, np.random.default_rng([seed, task.index, 556]))
        self.pos = 0

    def tell(self, X: np.ndarray, y: np.ndarray) -> None:
        return None

    def ask(self, q: int, pending: np.ndarray) -> np.ndarray:
        out = self.pool[self.pos : self.pos + q]
        self.pos += q
        return out


def _match(open_rows: list[tuple[np.ndarray, Any]], x: np.ndarray) -> int:
    for i, (row, _) in enumerate(open_rows):
        if np.allclose(row, x, rtol=0.0, atol=1e-9):
            return i
    return -1


class _OptunaArm(_Arm):
    """optuna TPESampler, library defaults (constant_liar=True is the default in optuna 5.0)."""

    def __init__(self, task: Task, seed: int) -> None:
        super().__init__(task, seed)
        import optuna  # bench extra

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self._optuna = optuna
        self.study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
        )
        self.dists: dict[str, Any] = {}
        for j in range(task.dim):
            if j in task.cat_dims:
                k = task.cat_cardinalities[task.cat_dims.index(j)]
                self.dists[f"x{j}"] = optuna.distributions.CategoricalDistribution(list(range(k)))
            else:
                self.dists[f"x{j}"] = optuna.distributions.FloatDistribution(
                    float(task.box[0, j]), float(task.box[1, j])
                )
        self.open: list[tuple[np.ndarray, Any]] = []

    def _vec(self, params: dict[str, Any]) -> np.ndarray:
        return np.array([float(params[f"x{j}"]) for j in range(self.task.dim)])

    def tell(self, X: np.ndarray, y: np.ndarray) -> None:
        optuna = self._optuna
        for x, yi in zip(np.asarray(X, float), np.asarray(y, float).reshape(-1)):
            i = _match(self.open, x)
            if i >= 0:
                _, trial = self.open.pop(i)
                self.study.tell(trial, float(yi))
            else:  # shared init or fallback points: add as completed trials
                params = {f"x{j}": (int(round(x[j])) if j in self.task.cat_dims else float(x[j]))
                          for j in range(self.task.dim)}
                self.study.add_trial(
                    optuna.trial.create_trial(
                        params=params, distributions=self.dists, value=float(yi)
                    )
                )

    def ask(self, q: int, pending: np.ndarray) -> np.ndarray:
        rows = []
        for _ in range(q):
            trial = self.study.ask(fixed_distributions=self.dists)
            x = self._vec(trial.params)
            self.open.append((x, trial))
            rows.append(x)
        return np.vstack(rows)


class _AxArm(_Arm):
    """Ax Client with the default generation strategy; running trials are pending."""

    def __init__(self, task: Task, seed: int) -> None:
        super().__init__(task, seed)
        from ax.api.client import Client  # bench extra
        from ax.api.configs import ChoiceParameterConfig, RangeParameterConfig

        # Ax sets its own logger levels on import, so quiet them afterwards.
        for name in ("ax", "ax.api.client"):
            logging.getLogger(name).setLevel(logging.WARNING)

        params: list[Any] = []
        for j in range(task.dim):
            if j in task.cat_dims:
                k = task.cat_cardinalities[task.cat_dims.index(j)]
                params.append(ChoiceParameterConfig(
                    name=f"x{j}", values=list(range(k)), parameter_type="int", is_ordered=False
                ))
            else:
                params.append(RangeParameterConfig(
                    name=f"x{j}",
                    bounds=(float(task.box[0, j]), float(task.box[1, j])),
                    parameter_type="float",
                ))
        self.client = Client()
        self.client.configure_experiment(name=f"vsbase_{task.name}_{seed}", parameters=params)
        # A bare metric name is maximised in Ax 1.3.1. "-y" was tried first and
        # made Ax minimise y (checked on the pilot seed; it proposed the corners).
        self.client.configure_optimization(objective="y")
        # Same as the default strategy ("fast", what Client uses when unconfigured),
        # with its Sobol initialisation seeded. Without this the Ax arm was not
        # reproducible (see the deviation log, 2026-10-18).
        self.client.configure_generation_strategy(method="fast", initialization_random_seed=seed)
        self.open: list[tuple[np.ndarray, int]] = []

    def _vec(self, p: dict[str, Any]) -> np.ndarray:
        return np.array([float(p[f"x{j}"]) for j in range(self.task.dim)])

    def tell(self, X: np.ndarray, y: np.ndarray) -> None:
        for x, yi in zip(np.asarray(X, float), np.asarray(y, float).reshape(-1)):
            i = _match(self.open, x)
            if i >= 0:
                _, idx = self.open.pop(i)
            else:
                params = {f"x{j}": (int(round(x[j])) if j in self.task.cat_dims else float(x[j]))
                          for j in range(self.task.dim)}
                idx = self.client.attach_trial(parameters=params)
            self.client.complete_trial(trial_index=idx, raw_data={"y": float(yi)})

    def ask(self, q: int, pending: np.ndarray) -> np.ndarray:
        import torch  # core ml extra

        self.round += 1
        torch.manual_seed(self.seed * 100 + self.round)  # BoTorch's acquisition restarts
        rows: list[np.ndarray] = []
        guard = 0
        while len(rows) < q:
            out = self.client.get_next_trials(max_trials=q - len(rows))
            for idx in sorted(out):
                x = self._vec(dict(out[idx]))
                self.open.append((x, idx))
                rows.append(x)
            guard += 1
            if guard > 5:
                raise RuntimeError("Ax returned too few trials")
        return np.vstack(rows)


class _BoTorchGPEIArm(_Arm):
    """Plain BoTorch GP-EI: SingleTaskGP, Normalize + Standardize, qEI with X_pending.

    Categorical codes are treated as ordinal numbers and the output is rounded to the
    nearest level. There is no categorical kernel. This is deliberate (protocol section 2).
    """

    def __init__(self, task: Task, seed: int) -> None:
        super().__init__(task, seed)
        self.X: list[np.ndarray] = []
        self.y: list[float] = []

    def tell(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X.extend(list(np.asarray(X, float)))
        self.y.extend(list(np.asarray(y, float).reshape(-1)))

    def ask(self, q: int, pending: np.ndarray) -> np.ndarray:
        import torch  # core ml extra
        from botorch.acquisition import qExpectedImprovement
        from botorch.fit import fit_gpytorch_mll
        from botorch.models import SingleTaskGP
        from botorch.models.transforms import Normalize, Standardize
        from botorch.optim import optimize_acqf
        from gpytorch.mlls import ExactMarginalLogLikelihood

        self.round += 1
        torch.manual_seed(self.seed * 100 + self.round)
        dt = torch.double
        d = self.task.dim
        X = torch.as_tensor(np.vstack(self.X), dtype=dt)
        Y = torch.as_tensor(np.asarray(self.y), dtype=dt).unsqueeze(-1)
        B = torch.as_tensor(self.task.box, dtype=dt)
        model = SingleTaskGP(
            X, Y,
            input_transform=Normalize(d=d, bounds=B),
            outcome_transform=Standardize(m=1),
        )
        fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
        X_pend = torch.as_tensor(pending, dtype=dt) if pending.size else None
        # Plain qEI on purpose (the textbook form). BoTorch's deprecation notice that
        # qLogEI is preferred is silenced here so the logs stay readable.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*qLogExpectedImprovement.*")
            acq = qExpectedImprovement(model, best_f=float(Y.max()), X_pending=X_pend)
        cand, _ = optimize_acqf(acq, bounds=B, q=q, num_restarts=10, raw_samples=256)
        out = cand.detach().cpu().numpy().astype(float)
        for dim, k in zip(self.task.cat_dims, self.task.cat_cardinalities):
            out[:, dim] = np.clip(np.rint(out[:, dim]), 0, k - 1)
        return out.reshape(q, d)


def make_arm(method: str, task: Task, seed: int) -> _Arm:
    if method == "kalos":
        return _KalosArm(task, seed)
    if method == "random":
        return _RandomArm(task, seed)
    if method == "lhs":
        return _LHSArm(task, seed)
    if method == "optuna_tpe":
        return _OptunaArm(task, seed)
    if method == "ax":
        return _AxArm(task, seed)
    if method == "botorch_gp_ei":
        return _BoTorchGPEIArm(task, seed)
    raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")


# ---- Runner ------------------------------------------------------------------------


def _first_reach(curve: np.ndarray, target: float, n_total: int) -> int:
    hit = np.where(curve >= target)[0]
    return int(hit[0]) + 1 if hit.size else n_total + 1


def run_unit(task_name: str, method: str, eta: float, seed: int) -> dict[str, Any]:
    """One (task, method, noise, seed) run, following the protocol in docs/BENCHMARK.md."""
    task = get_task(task_name)
    N = task.n_total
    d = task.dim
    eps = noise_vector(task, eta, seed)
    arm = make_arm(method, task, seed)
    lo_ref = f_med(task)
    gap = task.f_star - lo_ref

    measured_true: list[float] = []
    measured_obs: list[float] = []
    measured_X: list[np.ndarray] = []
    errors: list[str] = []
    fallback_steps = 0
    ask_times: list[float] = []

    def measure(X: np.ndarray, k0: int) -> np.ndarray:
        yt = true_value(task, X)
        yo = yt + eps[k0 : k0 + len(yt)]
        measured_true.extend(yt.tolist())
        measured_obs.extend(yo.tolist())
        measured_X.extend(list(X))
        return yo

    X0 = shared_init(task, seed)
    y0 = measure(X0, 0)
    arm.tell(X0, y0)
    k = task.n_init
    inflight = np.empty((0, d))
    for r in range(task.rounds):
        t0 = time.perf_counter()
        try:
            Xq = arm.ask(task.q, inflight)
            Xq = np.asarray(Xq, float).reshape(task.q, d)
            if not np.isfinite(Xq).all():
                raise ValueError("non-finite proposal")
        except Exception as exc:  # protocol section 9: fallback is counted, never silent
            fallback_steps += 1
            errors.append(f"round {r + 1}: {type(exc).__name__}: {str(exc)[:160]}")
            Xq = _uniform(task, task.q, np.random.default_rng([seed, task.index, 558, r]))
        ask_times.append(time.perf_counter() - t0)
        now = Xq[: task.q - task.depth]
        done = np.vstack([inflight, now]) if inflight.size else now
        yo = measure(done, k)
        k += len(done)
        arm.tell(done, yo)
        inflight = Xq[task.q - task.depth :] if task.depth else np.empty((0, d))
    if inflight.size:
        yo = measure(inflight, k)
        k += len(inflight)
        arm.tell(inflight, yo)
    assert k == N, (k, N)

    true_curve = np.maximum.accumulate(np.asarray(measured_true))
    p_curve = (true_curve - lo_ref) / gap
    obs_best = int(np.argmax(np.asarray(measured_obs)))
    obs_regret = task.f_star - measured_true[obs_best]
    final_regret = task.f_star - float(true_curve[-1])

    times = np.asarray(ask_times)
    return {
        "task": task_name,
        "method": method,
        "eta": eta,
        "seed": seed,
        "n_init": task.n_init,
        "q": task.q,
        "rounds": task.rounds,
        "pending_depth": task.depth,
        "N": N,
        "f_star": task.f_star,
        "f_med": lo_ref,
        "final_best_true": float(true_curve[-1]),
        "regret_raw": final_regret,
        "regret_norm": 1.0 - float(p_curve[-1]),
        "obs_regret_raw": float(obs_regret),
        "obs_regret_norm": float((task.f_star - measured_true[obs_best]) / gap),
        "reach90": _first_reach(p_curve, HEADLINE_TARGET, N),
        "reach75": _first_reach(p_curve, TARGETS[1], N),
        "fallback_steps": fallback_steps,
        "ask_seconds_total": float(times.sum()),
        "ask_seconds_median": float(np.median(times)),
        "ask_seconds_p95": float(np.percentile(times, 95)),
        "errors": errors,
        "curve_p": [float(v) for v in p_curve],
    }


def _worker(args: tuple[str, str, float, int]) -> dict[str, Any]:
    return run_unit(*args)


def run_all(
    seeds: Sequence[int],
    *,
    tasks: Sequence[str] = tuple(TASKS),
    methods: Sequence[str] = METHODS,
    workers: int | None = None,
) -> list[dict[str, Any]]:
    units: list[tuple[str, str, float, int]] = []
    for t in tasks:
        for eta in get_task(t).noises:
            for m in methods:
                for s in seeds:
                    units.append((t, m, eta, int(s)))
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    n_workers = workers or max(1, (os.cpu_count() or 2) - 1)
    if n_workers == 1:
        return [_worker(u) for u in units]
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
        return list(ex.map(_worker, units, chunksize=1))


# ---- Analysis ----------------------------------------------------------------------


def bootstrap_mean_ci(x: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    x = np.asarray(x, float)
    idx = rng.integers(0, x.size, size=(N_BOOT, x.size))
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(x.mean()), float(lo), float(hi)


def _holm(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out: dict[str, float] = {}
    running = 0.0
    for rank, (name, p) in enumerate(items):
        running = max(running, min(1.0, (m - rank) * p))
        out[name] = running
    return out


def _by_seed(runs: list[dict[str, Any]], task: str, eta: float, method: str) -> dict[int, dict[str, Any]]:
    return {
        r["seed"]: r for r in runs
        if r["task"] == task and r["eta"] == eta and r["method"] == method
    }


def analyse(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Summaries per cell and method, plus paired comparisons with Kalos."""
    rng = np.random.default_rng(BOOT_SEED)
    cells: list[tuple[str, float]] = []
    for r in runs:
        key = (r["task"], r["eta"])
        if key not in cells:
            cells.append(key)
    out: dict[str, Any] = {"cells": []}
    for task_name, eta in cells:
        cell: dict[str, Any] = {"task": task_name, "eta": eta, "methods": {}, "paired_vs_kalos": {}}
        per_method: dict[str, dict[int, dict[str, Any]]] = {}
        for m in METHODS:
            per_method[m] = _by_seed(runs, task_name, eta, m)
        seeds = sorted(set(per_method["kalos"]))
        for m in METHODS:
            rs = per_method[m]
            ordered = [rs[s] for s in seeds if s in rs]
            if not ordered:
                continue
            reg_norm = np.array([r["regret_norm"] for r in ordered])
            reg_raw = np.array([r["regret_raw"] for r in ordered])
            obs_norm = np.array([r["obs_regret_norm"] for r in ordered])
            r90 = np.array([r["reach90"] for r in ordered], float)
            r75 = np.array([r["reach75"] for r in ordered], float)
            N = ordered[0]["N"]
            cell["methods"][m] = {
                "n_seeds": len(ordered),
                "N": N,
                "regret_raw": bootstrap_mean_ci(reg_raw, rng),
                "regret_norm": bootstrap_mean_ci(reg_norm, rng),
                "obs_regret_norm": bootstrap_mean_ci(obs_norm, rng),
                "reach90_rate": float(np.mean(r90 <= N)),
                "reach90_experiments": bootstrap_mean_ci(r90, rng),
                "reach75_rate": float(np.mean(r75 <= N)),
                "reach75_experiments": bootstrap_mean_ci(r75, rng),
                "fallback_steps_total": int(sum(r["fallback_steps"] for r in ordered)),
                "runs_with_fallback": int(sum(r["fallback_steps"] > 0 for r in ordered)),
                "ask_seconds_per_run": bootstrap_mean_ci(
                    np.array([r["ask_seconds_total"] for r in ordered]), rng
                ),
                "ask_seconds_median_per_proposal": float(np.median(
                    [r["ask_seconds_median"] for r in ordered]
                )),
                "ask_seconds_p95_per_proposal": float(np.percentile(
                    [r["ask_seconds_p95"] for r in ordered], 95
                )),
            }
        if per_method.get("kalos"):
            kal = np.array([per_method["kalos"][s]["regret_norm"] for s in seeds])
            pvals: dict[str, float] = {}
            comps: dict[str, dict[str, Any]] = {}
            for m in BASELINES:
                if m not in per_method or not all(s in per_method[m] for s in seeds):
                    continue
                other = np.array([per_method[m][s]["regret_norm"] for s in seeds])
                d = kal - other  # negative favours Kalos
                mean, lo, hi = bootstrap_mean_ci(d, rng)
                wtl = {
                    "kalos_wins": int(np.sum(d < -TIE_TOL)),
                    "ties": int(np.sum(np.abs(d) <= TIE_TOL)),
                    "kalos_losses": int(np.sum(d > TIE_TOL)),
                }
                if np.allclose(d, 0):
                    p = 1.0
                else:
                    p = float(stats.wilcoxon(d, zero_method="pratt").pvalue)
                pvals[m] = p
                if hi < 0:
                    call = "Kalos better"
                elif lo > 0:
                    call = "baseline better"
                else:
                    call = "unresolved (CI includes 0)"
                comps[m] = {"mean_diff_norm": mean, "ci": [lo, hi], "wtl": wtl,
                            "wilcoxon_p": p, "call": call}
            adj = _holm(pvals)
            for m, c in comps.items():
                c["wilcoxon_p_holm"] = adj[m]
            cell["paired_vs_kalos"] = comps
        out["cells"].append(cell)
    return out


# ---- Environment and output ------------------------------------------------------


def _pkg_version(name: str) -> str | None:
    try:
        return _md.version(name)
    except _md.PackageNotFoundError:
        return None


def environment() -> dict[str, Any]:
    cpu = platform.processor() or platform.machine()
    try:
        cpu = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=True  # noqa: S607 - fixed argv, executable resolved from PATH (ps/git)
        ).stdout.strip() or cpu
    except (OSError, subprocess.CalledProcessError):
        pass
    sha, dirty = None, None
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,  # noqa: S607 - fixed argv, executable resolved from PATH (ps/git)
                             check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "kalos", "docs"],  # noqa: S607 - fixed argv, executable resolved from PATH (ps/git)
                                    capture_output=True, text=True, check=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        pass
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": cpu,
        "cpu_count": os.cpu_count(),
        "git_sha": sha,
        "git_dirty_harness_paths": dirty,
        "packages": {p: _pkg_version(p) for p in (
            "torch", "botorch", "gpytorch", "numpy", "scipy", "pandas",
            "optuna", "ax-platform")},
    }


def _write_outputs(out_dir: Path, runs: list[dict[str, Any]], summary: dict[str, Any],
                   meta: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    flat_keys = [k for k in runs[0] if k != "curve_p" and k != "errors"]
    with open(out_dir / "vs_baselines_runs.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(flat_keys + ["errors"])
        for r in runs:
            w.writerow([r[k] for k in flat_keys] + [" | ".join(r["errors"])])
    with open(out_dir / "vs_baselines_curves.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        n = max(len(r["curve_p"]) for r in runs)
        w.writerow(["task", "eta", "method", "seed"] + [f"p{t + 1}" for t in range(n)])
        for r in runs:
            w.writerow([r["task"], r["eta"], r["method"], r["seed"]] + r["curve_p"])
    with open(out_dir / "vs_baselines.json", "w") as fh:
        json.dump({**meta, "summary": summary}, fh, indent=2, default=float)


def _fmt_ci(t: tuple[float, float, float], digits: int = 3) -> str:
    return f"{t[0]:.{digits}f} [{t[1]:.{digits}f}, {t[2]:.{digits}f}]"


def render_markdown(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    for cell in summary["cells"]:
        tname, eta = cell["task"], cell["eta"]
        lines.append(f"### {tname}, noise {int(round(eta * 100))}%\n")
        lines.append("| method | regret (raw) mean [95% CI] | regret (norm) | obs-best regret (norm) "
                     "| reach 90%: rate, mean experiments | ask s / run (mean) | vs Kalos |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for m, s in cell["methods"].items():
            comp = cell["paired_vs_kalos"].get(m)
            vs = "-" if comp is None else (
                f"{comp['call']}; W/T/L {comp['wtl']['kalos_wins']}/{comp['wtl']['ties']}/"
                f"{comp['wtl']['kalos_losses']}")
            lines.append(
                f"| {m} | {_fmt_ci(s['regret_raw'])} | {s['regret_norm'][0]:.3f} "
                f"| {s['obs_regret_norm'][0]:.3f} "
                f"| {s['reach90_rate']:.2f}, {s['reach90_experiments'][0]:.1f} (N={s['N']}) "
                f"| {s['ask_seconds_per_run'][0]:.1f} | {vs} |")
        lines.append("")
    return "\n".join(lines)


def run_vs_baselines(argv: Sequence[str], out_dir: Path) -> int:
    """Entry point used by `python -m kalos.bench --vs-baselines`."""
    pilot = "--pilot" in argv
    quick = "--quick" in argv
    if pilot:
        seeds: Sequence[int] = (PILOT_SEED,)
    elif quick:
        seeds = QUICK_SEEDS
    else:
        seeds = CONFIRMATORY_SEEDS
    t0 = time.perf_counter()
    runs = run_all(seeds)
    wall = time.perf_counter() - t0
    summary = analyse(runs)
    print(f"vs-baselines: {len(runs)} runs, seeds={len(seeds)}, wall {wall:.0f}s")
    print(render_markdown(summary))
    if pilot or quick:
        print("(pilot/quick: nothing written to benchmarks/results/)")
        return 0
    meta = {
        "protocol": "docs/BENCHMARK.md",
        "seeds": list(seeds),
        "wall_seconds": wall,
        "environment": environment(),
        "methods": list(METHODS),
    }
    _write_outputs(out_dir, runs, summary, meta)
    print(f"wrote {out_dir}/vs_baselines_runs.csv, vs_baselines_curves.csv, vs_baselines.json")
    return 0


__all__ = [
    "BASELINES",
    "CONFIRMATORY_SEEDS",
    "METHODS",
    "PILOT_SEED",
    "TASKS",
    "analyse",
    "render_markdown",
    "run_all",
    "run_unit",
    "run_vs_baselines",
]
