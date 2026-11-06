"""Linear constraints on inputs over HTTP (`roles.linear_constraints`).

A lab states hard limits on its continuous inputs, for example a mixture that
must sum to 100 (`==`) or a sugar budget (`Glucose + Glycerol <= 30`). They are
known before any run, so they belong to the search, not to a model: the batch,
the repair pool and the best bet are all searched inside them (BoTorch
`inequality_constraints` / `equality_constraints`, raw units, the design's own
coordinate order; the GP's input scaling lives inside the model, so the
optimizer sees raw values).

Refusals, all before any fit:
- an input that is categorical or not an input of the analysis:
  `invalid_linear_constraint`;
- any categorical input in the design: `linear_constraints_with_categorical`
  (the mixed optimizers here do not take linear constraints);
- no recipe inside the search box meets them all: `infeasible_linear_constraints`.

An input the engine held fixed (near-constant on the sheet) enters at its fixed
value. Recipes are shown to 3 decimals, so each constraint is checked on the
shown recipe with a tolerance of 0.001 x the sum of its absolute coefficients.

Torch-free: the feasibility check uses the engine's LP check through a deferred
import only when there are constraints.
"""
from __future__ import annotations

from typing import Any

from kalos.portal.uploads import UploadRejected

# Shown recipes are rounded to 3 decimals (DesignSpace.decode_row).
ROUNDING_STEP = 0.001
TOLERANCE_RULE = (
    "Recipes are shown to 3 decimals, so each constraint may be off by up to "
    "0.001 x the sum of its absolute coefficients."
)


def check_linear_inputs(
    constraints: tuple, cont_feats: list, cat_feats: list, fixed_inputs: dict[str, dict],
) -> None:
    """Refuse a constraint on a categorical or unknown input, and any constraint
    on a design with categorical inputs."""
    if not constraints:
        return
    cont = {str(c) for c in cont_feats}
    cats = {str(c) for c in cat_feats}
    for k, con in enumerate(constraints, start=1):
        for name, _ in con.terms:
            if name in cats:
                raise UploadRejected(
                    f"Linear constraint {k} uses {name!r}, which is a categorical input. "
                    "Linear constraints apply to numeric inputs only.",
                    code="invalid_linear_constraint",
                )
            if name not in cont and name not in fixed_inputs:
                raise UploadRejected(
                    f"Linear constraint {k} uses {name!r}, which is not a numeric input "
                    "of this analysis.",
                    code="invalid_linear_constraint",
                )
    if cats:
        raise UploadRejected(
            "Linear constraints are not supported yet when the inputs include a "
            "categorical column. Remove the constraints or the categorical input.",
            code="linear_constraints_with_categorical",
        )


def encode_linear_constraints(
    constraints: tuple, names: list[str], fixed_inputs: dict[str, dict], bounds: Any,
) -> tuple[list, list]:
    """Map the constraints onto the design's coordinates as engine triples
    `(indices, coefficients, rhs)`: `(ge, eq)` with every `<=` flipped to `>=`.
    A fixed input moves to the right-hand side at its fixed value. Raises
    `infeasible_linear_constraints` when no recipe in the box meets them all."""
    if not constraints:
        return [], []
    import numpy as np

    pos = {n: i for i, n in enumerate(names)}
    ge: list = []
    eq: list = []
    for con in constraints:
        idx: list[int] = []
        coef: list[float] = []
        rhs = float(con.rhs)
        for name, c in con.terms:
            if name in pos:
                idx.append(pos[name])
                coef.append(float(c))
            else:
                rhs -= float(c) * float(fixed_inputs[name]["value"])
        if not idx or not any(coef):
            # Only fixed inputs: either it holds as the sheet stands or nothing can.
            lhs = 0.0
            slack = ROUNDING_STEP * sum(abs(c) for _, c in con.terms)
            ok = (
                abs(lhs - rhs) <= slack if con.op == "==" else
                lhs <= rhs + slack if con.op == "<=" else lhs >= rhs - slack
            )
            if not ok:
                raise _infeasible()
            continue
        if con.op == "==":
            eq.append((idx, coef, rhs))
        elif con.op == ">=":
            ge.append((idx, coef, rhs))
        else:
            ge.append((idx, [-c for c in coef], -rhs))
    from kalos.core.optimize import _validate_linear_sets

    b = np.asarray(bounds, float)
    try:
        _validate_linear_sets(ge, eq, b[0], b[1])
    except ValueError:
        raise _infeasible() from None
    return ge, eq


def _infeasible() -> UploadRejected:
    return UploadRejected(
        "No recipe inside the search box meets every linear constraint. Check the "
        "numbers, or let the search go past the tried range (allow_beyond).",
        code="infeasible_linear_constraints",
    )


def constraint_tolerance(con: Any) -> float:
    return ROUNDING_STEP * sum(abs(c) for _, c in con.terms)


def recipe_meets(con: Any, recipe: dict[str, Any]) -> bool | None:
    """Whether a shown recipe meets one constraint (None when an input is missing)."""
    lhs = 0.0
    for name, c in con.terms:
        v = recipe.get(name)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return None
        lhs += float(c) * float(v)
    tol = constraint_tolerance(con)
    if con.op == "==":
        return abs(lhs - con.rhs) <= tol
    if con.op == "<=":
        return lhs <= con.rhs + tol
    return lhs >= con.rhs - tol


def recipe_constraints_ok(constraints: tuple, recipe: dict[str, Any]) -> bool | None:
    """True when the recipe meets every constraint; None when none are declared."""
    if not constraints:
        return None
    return all(recipe_meets(con, recipe) is True for con in constraints)


def linear_constraints_block(constraints: tuple, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Top-level `linear_constraints`: the constraints as stated, the tolerance
    rule, and whether every row (proposals and the best bet) meets them."""
    items = []
    for con in constraints:
        d = con.to_dict()
        d["text"] = con.text()
        d["tolerance"] = round(constraint_tolerance(con), 9)
        items.append(d)
    oks = [r.get("constraints_ok") for r in rows]
    return {
        "constraints": items,
        "tolerance": TOLERANCE_RULE,
        "all_satisfied": all(ok is True for ok in oks) if constraints else True,
    }
