"""Advisory stopping signal: can more rounds still be told apart from assay noise?

Advice only. Nothing here removes a proposal or changes any other number; the
analysis reports the result in its `stopping` block and the scientist decides.

The rule follows the probabilistic regret bound of Wilson (2024,
arXiv:2402.16811) and Makarova et al. (2022, arXiv:2104.08166): stop being
confident about gains once the possible improvement is dominated by the
measurement error. Here the measurement error is the assay noise sd measured
from repeated recipes (`noise.floor.sigma`).

From the fitted proposal surrogate (fit on `sign * y`, so larger is better on
both directions) draw joint posterior samples of the latent response at the
best measured recipe and at every candidate (the model optimum and the
proposals). Per sample, the gain is the best candidate minus the best measured
recipe. `p_gain_beyond_noise` is the share of samples whose gain exceeds the
noise sd. `max_expected_gain` is the largest posterior-mean gain over the best
measured value, in the target's units, positive = better.
"""
from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["STOP_N_SAMPLES", "STOP_P_THRESHOLD", "stopping_advice"]

STOP_N_SAMPLES = 2000
# Joint posterior samples. The Monte Carlo error on a probability is at most
# about 0.011 (one standard error at p = 0.5).

STOP_P_THRESHOLD = 0.10
# `gains_within_noise` when P(gain > noise sd) is below this (delta = 0.1 in
# the regret-bound rule: the gain is within noise with probability >= 0.9).

_ADVICE = " This is advice, not a stop."


def _fmt(x: float) -> str:
    return f"{x:.2g}" if abs(x) < 10 else f"{x:.3g}"


def _joint_mean_cov(surrogate: Any, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Posterior mean and covariance of the latent response at `X`, in the
    units the surrogate was fit on."""
    import torch

    from kalos.core.surrogate import DEVICE, DTYPE

    model = surrogate.model
    assert model is not None, "fit() first"
    Xt = torch.as_tensor(np.asarray(X, float), dtype=DTYPE, device=DEVICE)
    model.eval()
    with torch.no_grad():
        post = model.posterior(Xt)
        mean = post.mean.squeeze(-1).cpu().numpy().astype(float)
        cov = post.mvn.covariance_matrix.cpu().numpy().astype(float)
    return mean.reshape(-1), cov.reshape(mean.size, mean.size)


def _sample(mean: np.ndarray, cov: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Joint normal samples with a seeded generator (never the global RNG)."""
    cov = 0.5 * (cov + cov.T)
    w, v = np.linalg.eigh(cov)
    root = v * np.sqrt(np.clip(w, 0.0, None))
    z = np.random.default_rng(seed).standard_normal((n, mean.size))
    return mean + z @ root.T


def unknown_advice(plain: str, *, noise_sd: float | None = None) -> dict:
    return {
        "advisory": True,
        "state": "unknown",
        "plain": plain + _ADVICE,
        "p_gain_beyond_noise": None,
        "max_expected_gain": None,
        "gain_over_noise": None,
        "noise_sd": None if noise_sd is None else round(float(noise_sd), 6),
        "threshold": STOP_P_THRESHOLD,
        "method": _METHOD,
        "n_samples": 0,
        "seed": None,
    }


_METHOD = (
    "joint posterior samples of the latent response at the best measured recipe, "
    "the model optimum and the proposals; gain = best candidate minus best measured "
    "recipe per sample; gains_within_noise when P(gain > assay noise sd) < threshold "
    "(probabilistic regret bound, Wilson 2024; Makarova et al. 2022)"
)

NO_REPLICATES = (
    "Kalos cannot say yet whether more rounds will help: it needs repeated runs of "
    "the same recipe to measure the run-to-run spread of your assay."
)


def stopping_advice(
    surrogate: Any,
    best_x: np.ndarray,
    best_value_fit: float,
    candidates: np.ndarray,
    noise_sd: float | None,
    *,
    n_samples: int = STOP_N_SAMPLES,
    seed: int = 0,
) -> dict:
    """The `stopping` block. `best_value_fit` is the best measured value on the
    fit scale (`sign * y`), `best_x` its recipe; `candidates` are encoded rows
    (model optimum and proposals). `noise_sd` is the assay noise sd of a single
    measurement, or None when there are no repeats (state `unknown`)."""
    if noise_sd is None or not np.isfinite(noise_sd) or noise_sd <= 0:
        return unknown_advice(NO_REPLICATES)
    cand = np.asarray(candidates, float).reshape(-1, np.asarray(best_x).size)
    if cand.shape[0] == 0:
        return unknown_advice(
            "Kalos cannot say whether more rounds will help: there is no candidate recipe to compare.",
            noise_sd=noise_sd,
        )
    pts = np.vstack([np.asarray(best_x, float).reshape(1, -1), cand])
    mean, cov = _joint_mean_cov(surrogate, pts)
    draws = _sample(mean, cov, int(n_samples), int(seed))
    gain = draws[:, 1:].max(axis=1) - draws[:, 0]
    p = float(np.mean(gain > noise_sd))
    max_gain = float(mean[1:].max() - best_value_fit)
    sd_txt = _fmt(float(noise_sd))
    if p < STOP_P_THRESHOLD:
        state = "gains_within_noise"
        plain = (
            "The model expects further gains smaller than the run-to-run spread of "
            f"your assay (about {sd_txt}); more rounds may not be told apart from noise."
        )
    else:
        state = "keep_going"
        plain = (
            "The model puts the chance of beating your best run by more than the "
            f"run-to-run spread of your assay (about {sd_txt}) at {p:.0%}. More rounds "
            "can still pay off."
        )
    return {
        "advisory": True,
        "state": state,
        "plain": plain + _ADVICE,
        "p_gain_beyond_noise": round(p, 4),
        "max_expected_gain": round(max_gain, 6),
        "gain_over_noise": round(max_gain / float(noise_sd), 3),
        "noise_sd": round(float(noise_sd), 6),
        "threshold": STOP_P_THRESHOLD,
        "method": _METHOD,
        "n_samples": int(n_samples),
        "seed": int(seed),
    }
