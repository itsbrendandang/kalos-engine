"""QC-trap campaign simulator: contaminated runs that read HIGH.

PROVENANCE. `FACTORS`, `BOUNDS`, `NOISE_SIGMA`, `N_RUNS`, `Campaign` and
`make_campaign` are ported verbatim from the owner's own `bioagent-eval`
(`src/bioagent_eval/sim.py`, commit e2db1b3, sole author Brendan Dang, no
proprietary data). The closed-loop wrapper below (`QCTrapObjective`,
`run_qc_trap`, `qc_trap_benchmark`) is new.

The surface is a separable Gaussian titer bump over four factors (pH, temp,
feed, DO) with ~5% lognormal assay noise. The trap: a fraction of runs come
back `qc="fail"` (contaminated) with an inflated reading of 1.1-1.4x the true
PEAK, i.e. better than any real run can ever be. An optimizer that fits on
those rows and reports the best observed row as its incumbent reports a
contaminated run; one that drops QC-failed rows (`kalos/validation/qc.py`, the
same rule the portal applies to uploads) is unaffected. A campaign can also
carry a temperature cap (an equipment limit) that binds the optimum, which is
the constrained case: the feasible optimum is the unconstrained one with temp
clipped, exact because the surface is separable.

A sanity surface, not a hard one: separable and unimodal, so a GP finds it
easily. Its value is the trap, not the difficulty. Made-up numbers, not
calibrated to any real process.
"""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import qmc

from kalos.validation.qc import exclude_qc_failed

# --- ported verbatim from bioagent-eval src/bioagent_eval/sim.py ------------- #

FACTORS: tuple[str, ...] = ("pH", "temp_C", "feed_g_L_day", "DO_pct")
BOUNDS: dict[str, tuple[float, float]] = {
    "pH": (6.4, 7.6),
    "temp_C": (30.0, 38.0),
    "feed_g_L_day": (0.0, 10.0),
    "DO_pct": (20.0, 60.0),
}
NOISE_SIGMA = 0.05
N_RUNS = 24


@dataclass(frozen=True)
class Campaign:
    campaign_id: str
    peak: float
    opt: dict[str, float]
    width: dict[str, float]
    runs: list[dict[str, float | str]] = field(compare=False)
    temp_cap: float | None = None

    def true_titer(self, x: dict[str, float]) -> float:
        z = sum(((x[f] - self.opt[f]) / self.width[f]) ** 2 for f in FACTORS)
        return float(self.peak * np.exp(-0.5 * z))

    def upper(self, f: str) -> float:
        hi = BOUNDS[f][1]
        return min(hi, self.temp_cap) if f == "temp_C" and self.temp_cap is not None else hi

    def best_feasible(self) -> dict[str, float]:
        """Feasible optimum: separable surface, so clip each optimum into its bounds."""
        return {f: min(max(self.opt[f], BOUNDS[f][0]), self.upper(f)) for f in FACTORS}

    def dominant_factor(self) -> str:
        # Narrowest width relative to range = largest effect on titer.
        return min(FACTORS, key=lambda f: self.width[f] / (BOUNDS[f][1] - BOUNDS[f][0]))

    def to_csv(self, rows: list[dict[str, float | str]] | None = None) -> str:
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=["run_id", *FACTORS, "titer_g_L", "qc"])
        w.writeheader()
        w.writerows(self.runs if rows is None else rows)
        return buf.getvalue()

    def constraints_text(self) -> str:
        lines = [f"{f}: [{BOUNDS[f][0]}, {self.upper(f)}]" for f in FACTORS]
        if self.temp_cap is not None:
            lines.append(f"Equipment limit: temp_C must be <= {self.temp_cap}")
        return "\n".join(lines)


def make_campaign(campaign_id: str, seed: int, temp_cap: bool = False) -> Campaign:
    rng = np.random.default_rng(seed)
    span = {f: BOUNDS[f][1] - BOUNDS[f][0] for f in FACTORS}
    # Optimum sits in the inner 60% of the box so it is reachable and not on an edge.
    opt = {f: round(float(BOUNDS[f][0] + span[f] * rng.uniform(0.2, 0.8)), 2) for f in FACTORS}
    # One clearly dominant factor (narrow), the rest broad, so "largest effect" is unambiguous.
    dominant = FACTORS[int(rng.integers(len(FACTORS)))]
    width = {f: span[f] * (rng.uniform(0.18, 0.22) if f == dominant else rng.uniform(0.5, 0.65)) for f in FACTORS}
    peak = round(float(rng.uniform(2.0, 8.0)), 2)
    cap = round(max(opt["temp_C"] - float(rng.uniform(1.5, 3.0)), 31.0), 1) if temp_cap else None

    c = Campaign(campaign_id, peak, opt, width, runs=[], temp_cap=cap)
    n_fail = int(rng.integers(1, 3))
    fail_idx = set(rng.choice(N_RUNS, size=n_fail, replace=False).tolist())
    for i in range(N_RUNS):
        x = {f: round(float(rng.uniform(*BOUNDS[f])), 2) for f in FACTORS}
        y = c.true_titer(x) * float(np.exp(rng.normal(0, NOISE_SIGMA)))
        qc = "pass"
        if i in fail_idx:
            # Contaminated run: inflated reading that beats every real run.
            y = peak * float(rng.uniform(1.1, 1.4))
            qc = "fail"
        c.runs.append({"run_id": f"{campaign_id}-R{i + 1:02d}", **x, "titer_g_L": float(f"{y:.4g}"), "qc": qc})
    return c


# --- closed-loop wrapper (new) ------------------------------------------------ #


@dataclass
class QCTrapObjective:
    """A `Campaign` as a closed-loop objective: each queried recipe returns a
    noisy titer and a QC verdict. With probability `p_fail` (independent of the
    recipe, as contamination is) the run fails QC and reads 1.1-1.4x the peak."""

    campaign: Campaign
    p_fail: float = 0.15

    @property
    def dim(self) -> int:
        return len(FACTORS)

    @property
    def bounds(self) -> np.ndarray:
        """`(2, d)` box with the temperature cap applied to the upper row."""
        lo = [BOUNDS[f][0] for f in FACTORS]
        hi = [self.campaign.upper(f) for f in FACTORS]
        return np.array([lo, hi], dtype=float)

    @property
    def optimum(self) -> float:
        return self.campaign.true_titer(self.campaign.best_feasible())

    @property
    def argmax(self) -> np.ndarray:
        best = self.campaign.best_feasible()
        return np.array([best[f] for f in FACTORS], dtype=float)

    def true_values(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, float))
        return np.array([self.campaign.true_titer(dict(zip(FACTORS, row))) for row in X])

    def evaluate(self, X, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """`(y_true, y_obs, qc)` for each row of `X`."""
        y_true = self.true_values(X)
        y_obs = y_true * np.exp(rng.normal(0.0, NOISE_SIGMA, size=y_true.shape))
        qc = ["pass"] * len(y_true)
        for i in range(len(y_true)):
            if rng.uniform() < self.p_fail:
                y_obs[i] = self.campaign.peak * float(rng.uniform(1.1, 1.4))
                qc[i] = "fail"
        return y_true, y_obs, qc


def qc_trap_objective(seed: int = 0, *, temp_cap: bool = False, p_fail: float = 0.15) -> QCTrapObjective:
    return QCTrapObjective(make_campaign(f"QC{seed:03d}", seed, temp_cap=temp_cap), p_fail=p_fail)


def _frame(X: np.ndarray, y: np.ndarray, qc: list[str]) -> pd.DataFrame:
    df = pd.DataFrame(X, columns=list(FACTORS))
    df["titer_g_L"] = y
    df["qc"] = qc
    return df


def run_qc_trap(
    obj: QCTrapObjective,
    *,
    filter_qc: bool = True,
    strategy: str = "bo",
    budget: int = 12,
    n_init: int = 8,
    seed: int = 0,
    num_restarts: int = 4,
    raw_samples: int = 128,
) -> dict:
    """One closed-loop trial. `filter_qc=True` drops QC-failed rows (via
    `exclude_qc_failed`, the portal's rule) before every fit and before picking
    the incumbent; `False` is the naive arm that trusts every reading.

    The INCUMBENT is what the loop would report as its best recipe: the row with
    the highest observed titer among the rows it trusts. Returns the incumbent's
    true titer, its regret against the feasible optimum, whether the incumbent
    is a QC-failed run, the best true value queried, and whether every queried
    point respects the temperature cap.
    """
    import torch

    from kalos.core.optimize import propose
    from kalos.core.surrogate import FitError, Surrogate

    if strategy not in ("bo", "random"):
        raise ValueError("strategy must be 'bo' or 'random'")
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    b = obj.bounds
    X = b[0] + qmc.LatinHypercube(d=obj.dim, seed=rng).random(n_init) * (b[1] - b[0])
    y_true, y_obs, qc = obj.evaluate(X, rng)
    qc = list(qc)

    def trusted() -> pd.DataFrame:
        df = _frame(X, y_obs, qc)
        if filter_qc:
            df, _ = exclude_qc_failed(df)
        return df

    for _ in range(budget):
        df = trusted()
        x_next = None
        if strategy == "bo" and len(df) >= 3:
            try:
                s = Surrogate().fit(
                    df[list(FACTORS)].to_numpy(float).copy(), df["titer_g_L"].to_numpy(float).copy(), bounds=b
                )
                x_next = propose(s, b, q=1, num_restarts=num_restarts, raw_samples=raw_samples)[0]
            except FitError:
                x_next = None
        if x_next is None:
            x_next = rng.uniform(b[0], b[1])
        yt, yo, q = obj.evaluate(x_next, rng)
        X = np.vstack([X, x_next])
        y_true = np.append(y_true, yt)
        y_obs = np.append(y_obs, yo)
        qc.extend(q)

    df = trusted()
    inc_label = df["titer_g_L"].idxmax()
    inc_pos = int(inc_label)  # _frame uses a RangeIndex, so the label is the position
    temp_j = FACTORS.index("temp_C")
    return {
        "incumbent_true": float(y_true[inc_pos]),
        "incumbent_regret": float(obj.optimum - y_true[inc_pos]),
        "incumbent_is_qc_fail": qc[inc_pos] == "fail",
        "best_true_queried": float(y_true.max()),
        "n_qc_fail": int(sum(v == "fail" for v in qc)),
        "temp_cap_honoured": bool((X[:, temp_j] <= b[1, temp_j] + 1e-9).all()),
        "optimum": obj.optimum,
    }


def qc_trap_benchmark(
    seeds=range(5), *, budget: int = 12, n_init: int = 8, temp_cap: bool = False, p_fail: float = 0.15
) -> dict:
    """Filtered vs unfiltered BO over `seeds` (one campaign per seed). Returns
    per-arm mean incumbent regret (normalised by the optimum) and the share of
    seeds whose reported incumbent is a contaminated run."""
    out: dict[str, dict] = {}
    for arm, filt in (("qc_filtered", True), ("unfiltered", False)):
        rows = [
            run_qc_trap(qc_trap_objective(s, temp_cap=temp_cap, p_fail=p_fail), filter_qc=filt,
                        budget=budget, n_init=n_init, seed=s)
            for s in seeds
        ]
        out[arm] = {
            "mean_norm_regret": float(np.mean([r["incumbent_regret"] / r["optimum"] for r in rows])),
            "incumbent_is_qc_fail_rate": float(np.mean([r["incumbent_is_qc_fail"] for r in rows])),
            "mean_norm_best_queried_regret": float(
                np.mean([(r["optimum"] - r["best_true_queried"]) / r["optimum"] for r in rows])
            ),
            "n_seeds": len(rows),
        }
    return out


__all__ = [
    "BOUNDS",
    "FACTORS",
    "Campaign",
    "QCTrapObjective",
    "make_campaign",
    "qc_trap_benchmark",
    "qc_trap_objective",
    "run_qc_trap",
]
