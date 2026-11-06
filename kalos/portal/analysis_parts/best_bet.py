"""The model's best bet in every batch, and an honest line on each proposal.

First-time-user study, Finding 1: on every sheet all five proposals were
predicted below the best measured run, labelled "New ground", and nothing said
why. The batch acquisition (noisy expected improvement) rewards uncertainty, so
on a sheet with a strong incumbent it spends the whole batch exploring.

Now every batch carries the model's best bet: the recipe with the highest
predicted value inside the search box (`posterior_mean_optimum`). If a proposal
is already that recipe (within the batch's own near-duplicate tolerance), it is
replaced by the exact best bet; otherwise the best bet replaces the proposal with the lowest predicted
value, so the batch size stays the same. If the best bet is already running (a
started, unmeasured run), nothing is added and `model_optimum.already_running`
says so.

Each proposal gets `pick` ("best_bet" or "learn") and `predicted_vs_best`:
the predicted difference from the best measured value, compared with the
likely range, and one plain sentence. A run predicted below the best says it is
there to learn, and what from.
"""
from __future__ import annotations

from typing import Any

import numpy as np

# Sobol points added to the best-bet candidates (inside the box and the
# inequality constraints), so a local optimum of the optimizer cannot hide a
# higher predicted recipe (security review B1, pichia_HSA: optimizer 74.9 while
# a batch row was predicted 110.0).
BEST_BET_SAMPLES = 2048


def meets_linear(C: np.ndarray, ge: list | None, eq: list | None, tol: float = 1e-6) -> np.ndarray:
    """Rows of C that meet every `>=` and `==` constraint triple."""
    C = np.asarray(C, float)
    ok = np.ones(len(C), dtype=bool)
    for idx, coef, rhs in ge or []:
        ok &= C[:, list(idx)] @ np.asarray(coef, float) >= float(rhs) - tol
    for idx, coef, rhs in eq or []:
        scale = 1.0 + float(np.abs(np.asarray(coef, float)).sum())
        ok &= np.abs(C[:, list(idx)] @ np.asarray(coef, float) - float(rhs)) <= tol * scale + 1e-6
    return ok


def best_bet_candidates(
    optimizer_x: np.ndarray, batch: np.ndarray, X_train: np.ndarray, bounds: np.ndarray, *,
    cat_dims: list[int] | None, cat_cardinalities: list[int] | None,
    ge: list | None, eq: list | None, seed: int, n_samples: int = BEST_BET_SAMPLES,
) -> np.ndarray:
    """Every recipe the best bet is chosen from: the optimizer's optimum, every
    batch row, every training row, and a Sobol sample of the box (skipped
    under an equality constraint, which a sample cannot meet). Only rows inside
    the box and the linear constraints are kept."""
    from scipy.stats import qmc

    b = np.asarray(bounds, float)
    lower, upper = b[0], b[1]
    d = lower.shape[0]
    parts = [np.asarray(optimizer_x, float).reshape(1, d)]
    if len(batch):
        parts.append(np.asarray(batch, float).reshape(-1, d))
    if len(X_train):
        parts.append(np.asarray(X_train, float).reshape(-1, d))
    if not eq and n_samples > 0:
        u = qmc.Sobol(d=d, scramble=True, seed=int(seed)).random(int(n_samples))
        S = lower + u * (upper - lower)
        for j, k in zip(cat_dims or [], cat_cardinalities or [], strict=False):
            S[:, j] = np.clip(np.floor(u[:, j] * int(k)), 0, int(k) - 1)
        parts.append(S)
    C = np.vstack(parts)
    inside = np.all((lower - 1e-9 <= C) & (upper + 1e-9 >= C), axis=1)
    return C[inside & meets_linear(C, ge, eq)]


def choose_best_bet(
    candidates: np.ndarray, model_mean: np.ndarray, *,
    constraint_mean: np.ndarray | None = None, constraint_floor: float | None = None,
    p_feasible: np.ndarray | None = None,
) -> tuple[int | None, str | None]:
    """Index of the candidate with the highest predicted value (model sign) that
    meets the same outcome rules as the batch: the constraint model's
    prediction at or above its floor, and P(feasible) at least 0.5 when the
    feasibility gate is on (security review S2). (None, plain reason) when no
    candidate qualifies."""
    ok = np.ones(len(candidates), dtype=bool)
    if constraint_mean is not None and constraint_floor is not None:
        ok &= np.asarray(constraint_mean, float) >= float(constraint_floor)
    if p_feasible is not None:
        ok &= np.asarray(p_feasible, float) >= 0.5
    if not ok.any():
        why = (
            "No recipe the model looked at is predicted to meet the limit you set, so there "
            "is no best bet this round."
            if constraint_mean is not None else
            "No recipe the model looked at is predicted to work, so there is no best bet this round."
        )
        return None, why
    scores = np.where(ok, np.asarray(model_mean, float), -np.inf)
    return int(np.argmax(scores)), None


def place_best_bet(
    batch: np.ndarray,
    optimum: np.ndarray,
    model_mean: np.ndarray,
    bounds: np.ndarray,
    cat_dims: list[int] | None,
    pending: np.ndarray | None,
    q: int,
) -> tuple[np.ndarray, int | None, bool]:
    """Return (batch with the best bet first, its index or None, already_running).

    `model_mean` is the predicted mean of `batch` in the model's own sign (higher
    is better). The best bet goes first so it is P1 on every page."""
    # Deferred: kalos.core.optimize imports torch, and this module is imported by
    # the portal at startup (tests/test_desktop_mode.py checks no torch there).
    from kalos.core.optimize import NEAR_DUPLICATE_TOL, proposal_separation

    lower, upper = np.asarray(bounds, float)[0], np.asarray(bounds, float)[1]
    cats = list(cat_dims or [])

    def near(a: np.ndarray, b: np.ndarray) -> bool:
        return proposal_separation(a, b, lower, upper, cats) < NEAR_DUPLICATE_TOL

    if pending is not None and len(pending) and any(near(optimum, p) for p in pending):
        return batch, None, True
    rows = [np.asarray(r, float) for r in batch]
    match = next((i for i, r in enumerate(rows) if near(optimum, r)), None)
    if match is not None:
        # A proposal within the batch tolerance of the best bet becomes the best
        # bet itself, so its prediction is exactly `model_optimum.predicted`.
        rows[match] = np.asarray(optimum, float)
    else:
        if len(rows) < q:
            rows.append(np.asarray(optimum, float))
            match = len(rows) - 1
        elif rows:
            match = int(np.argmin(np.asarray(model_mean, float)))
            rows[match] = np.asarray(optimum, float)
        else:
            rows = [np.asarray(optimum, float)]
            match = 0
    order = [match] + [i for i in range(len(rows)) if i != match]
    return np.vstack([rows[i] for i in order]), 0, False


def predicted_vs_best(
    pred: float, best: float, half_width: float | None, *, maximize: bool, target: str,
    is_best_bet: bool, mode: str,
) -> dict[str, Any]:
    """The predicted difference from the best measured value, in the target's
    units and in the direction that counts as better, with one plain sentence."""
    gain = (pred - best) if maximize else (best - pred)
    hw = float(half_width) if half_width is not None and half_width == half_width else None
    better, worse = ("above", "below") if maximize else ("below", "above")
    if hw is not None and abs(gain) <= hw:
        relation = "about_best"
        plain = (
            f"Predicted {abs(gain):.3g} {better if gain >= 0 else worse} your best {target} "
            f"({best:.4g}), inside the likely range (±{hw:.3g}): it may land "
            f"{better} or {worse} your best."
        )
    elif gain > 0:
        relation = "better"
        plain = (
            f"Predicted {gain:.3g} {better} your best {target} ({best:.4g}), more than "
            "the likely range."
        )
    elif is_best_bet:
        # Security review S3: the best bet is the model's top pick, never a run
        # "to learn": say plainly that nothing in the box beats the best run.
        relation = "worse"
        plain = (
            f"Predicted {abs(gain):.3g} {worse} your best {target} ({best:.4g}). The model "
            "does not expect any recipe in the search box to beat your best run; this is "
            "its highest prediction, and running it checks that."
        )
    else:
        relation = "worse"
        why = (
            "it tests a part of the range the model knows least about"
            if mode == "explore" else "it spreads the batch so the round teaches more"
        )
        plain = (
            f"Predicted {abs(gain):.3g} {worse} your best {target} ({best:.4g}). This run "
            f"is to learn, not to win: {why}."
        )
    if is_best_bet:
        plain = "Best bet: the highest predicted value the model found inside the search box. " + plain
    return {
        "difference": round(float(gain), 4),
        "likely_half_width": None if hw is None else round(hw, 4),
        "relation": relation,
        "plain": plain,
    }


def label_proposals(
    proposals: list[dict[str, Any]], best_idx: int | None, *, best: float,
    half_width: float | None, maximize: bool, target: str,
) -> list[dict[str, Any]]:
    for i, row in enumerate(proposals):
        is_bb = best_idx is not None and i == best_idx
        row["pick"] = "best_bet" if is_bb else "learn"
        if is_bb:
            row["mode"] = "exploit"
            row["reason"] = f"model's best bet: highest predicted {target} inside the search box"
        row["predicted_vs_best"] = predicted_vs_best(
            float(row["pred"]), best, half_width, maximize=maximize, target=target,
            is_best_bet=is_bb, mode=str(row.get("mode")),
        )
    return proposals


__all__ = [
    "BEST_BET_SAMPLES", "best_bet_candidates", "choose_best_bet", "label_proposals",
    "meets_linear", "place_best_bet", "predicted_vs_best",
]
