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


__all__ = ["label_proposals", "place_best_bet", "predicted_vs_best"]
