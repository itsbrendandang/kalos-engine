"""Joint feasibility over HTTP (`roles.specs`): P(all declared specs met).

A spec is a limit on an outcome column (`{column, min?, max?}`; the target may
be one of them). One model is fit per spec column, the same way the single
`constraint` model is fit (same design, rows where the value is present, at
least `CONSTRAINT_MIN_ROWS` of them). For every proposal and the best bet the
analysis reports `p_specs` (per spec) and `p_all_specs` (all at once), from
seeded draws of each model's predictive distribution for a new measurement
(model uncertainty plus fitted assay noise); kalos/core/feasibility_joint.py.

Advisory: computed after the batch is chosen, with the torch RNG forked, so
declaring specs never changes the proposals.
TODO(product): an opt-in flag to rank or gate the batch by `p_all_specs` (not built).

A spec column cannot be a model input: declared as a feature or categorical it
is a 400 `invalid_specs`; inferred as a feature it is removed from the inputs,
like the `constraint` column.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from kalos.portal.uploads import UploadRejected

PREDICTS = "a new measurement: model uncertainty plus the fitted assay noise"


def spec_columns(roles: Any) -> list[str]:
    return [sp.column for sp in (getattr(roles, "specs", None) or ())]


def guard_spec_columns(roles: Any, cont_feats: list, cat_feats: list) -> tuple[list, list]:
    """Refuse a spec on a declared input; drop spec columns from inferred inputs."""
    cols = set(spec_columns(roles))
    if not cols:
        return cont_feats, cat_feats
    declared = {str(c) for c in (*roles.features, *roles.categoricals)}
    for col in spec_columns(roles):
        if col in declared:
            raise UploadRejected(
                f"{col!r} is declared as an input, so it cannot also have a spec. "
                "A spec is a limit on a measured result.",
                code="invalid_specs",
            )
    return (
        [c for c in cont_feats if str(c) not in cols],
        [c for c in cat_feats if str(c) not in cols],
    )


def evaluate_specs(
    roles: Any,
    df: Any,
    keep_index: Any,
    X: np.ndarray,
    bounds: np.ndarray,
    cat_dims: list[int] | None,
    points: np.ndarray,
    *,
    seed: int,
    min_rows: int,
) -> dict[str, Any]:
    """Fit one model per spec and evaluate `points` (proposals, then the best
    bet). Returns `{"rows": [per-point {p_all_specs, p_specs}], "block": ...}`.
    Never raises for data reasons: a spec that cannot be modelled makes the
    block unavailable with a reason, and every probability null."""
    import pandas as pd
    import torch

    from kalos.core.feasibility_joint import METHOD, N_SAMPLES, joint_feasibility
    from kalos.core.surrogate import FitError, Surrogate

    specs = tuple(getattr(roles, "specs", None) or ())
    n_points = int(points.shape[0])
    empty_rows = [{"p_all_specs": None, "p_specs": None} for _ in range(n_points)]
    block: dict[str, Any] = {
        "specs": [], "n_samples": N_SAMPLES, "method": METHOD, "predicts": PREDICTS,
        "seed": int(seed), "available": False, "reason": "no specs declared",
    }
    if not specs:
        return {"rows": empty_rows, "block": block}

    means: list[np.ndarray] = []
    stds: list[np.ndarray] = []
    reasons: list[str] = []
    items: list[dict[str, Any]] = []
    # Forked: the spec fits draw from the torch RNG, and the advisory must not
    # move anything computed after it.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        for sp in specs:
            item: dict[str, Any] = {**sp.to_dict(), "n_rows_with_value": 0, "available": False,
                                    "reason": None}
            if sp.column not in df.columns:
                item["reason"] = f"column {sp.column!r} is not in the sheet"
            else:
                vals = pd.to_numeric(df.loc[keep_index, sp.column], errors="coerce")
                mask = vals.notna().to_numpy()
                n = int(mask.sum())
                item["n_rows_with_value"] = n
                if n < min_rows:
                    item["reason"] = (
                        f"only {n} runs have a {sp.column!r} value; need at least {min_rows}"
                    )
                else:
                    try:
                        model = Surrogate().fit(
                            X[mask], vals.to_numpy(float)[mask], bounds=bounds, cat_dims=cat_dims,
                        )
                        if n_points:
                            m, s = model.posterior(points, observation_noise=True)
                            means.append(np.asarray(m, float).reshape(-1))
                            stds.append(np.asarray(s, float).reshape(-1))
                        item["available"] = True
                        del model
                    except FitError as exc:
                        item["reason"] = f"the {sp.column!r} model could not be fit: {exc}"
            if item["reason"]:
                reasons.append(item["reason"])
            items.append(item)
    block["specs"] = items
    if reasons:
        block["reason"] = "; ".join(reasons)
        return {"rows": empty_rows, "block": block}
    block["available"], block["reason"] = True, None
    if not n_points:
        return {"rows": [], "block": block}
    limits = [(sp.min, sp.max) for sp in specs]
    p_specs, p_all = joint_feasibility(means, stds, limits, seed=seed)
    rows = [
        {
            "p_all_specs": round(float(p_all[j]), 4),
            "p_specs": {sp.column: round(float(p_specs[k, j]), 4) for k, sp in enumerate(specs)},
        }
        for j in range(n_points)
    ]
    return {"rows": rows, "block": block}
