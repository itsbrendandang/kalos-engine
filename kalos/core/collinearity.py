"""Collinearity between process inputs: when two effects cannot be separated.

A media DoE often moves two inputs together by construction: acid and base
dosed to hold pH, a carbon source and its feed rate, a salt and the osmolality
it sets. When two inputs move together on the sheet, ANY model of the sheet
(Spearman, a GP, a regression) can only see their combined effect. The driver
panel then credits whichever of the pair happens to rank higher, and the GP's
ARD lengthscales split the relevance between them arbitrarily. Neither number
says which input matters; the data cannot say it.

Two checks, both on the same rows the driver panel uses (replicate means):

- PAIRS: Pearson correlation between every two continuous inputs. A pair with
  |r| >= `PAIR_R_THRESHOLD` is flagged.
- VIF (variance inflation factor) per input: 1 / (1 - R^2) of that input
  regressed on all the others. It also catches an input that is a combination of
  several others with no single strong pair. VIF >= `VIF_THRESHOLD` is flagged;
  an input that is an exact linear combination of the others has an infinite VIF,
  reported as `null` with `perfect: true` (JSON has no infinity).

Both are properties of the DESIGN (which recipes were run), not of the response.
Torch-free.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

PAIR_R_THRESHOLD = 0.9
"""|Pearson r| at or above which two inputs are flagged. At 0.9 the pair shares
81% of its variance, and a designed experiment that meant to vary them
independently would almost never produce it by chance at the sheet sizes Kalos
sees."""

VIF_THRESHOLD = 10.0
"""The conventional rule of thumb: the coefficient variance of that input is
inflated ten-fold by its overlap with the others."""

COLLINEAR_LABEL = "collinear: effects cannot be separated"

_PERFECT_R2 = 1.0 - 1e-10


def _vif(Z: np.ndarray) -> tuple[list[float | None], list[bool]]:
    n, d = Z.shape
    vifs: list[float | None] = []
    perfect: list[bool] = []
    for j in range(d):
        others = np.delete(Z, j, axis=1)
        A = np.column_stack([np.ones(n), others])
        coef, *_ = np.linalg.lstsq(A, Z[:, j], rcond=None)
        resid = Z[:, j] - A @ coef
        ss_tot = float(((Z[:, j] - Z[:, j].mean()) ** 2).sum())
        r2 = 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else 0.0
        if r2 >= _PERFECT_R2:
            vifs.append(None)
            perfect.append(True)
        else:
            vifs.append(round(1.0 / (1.0 - r2), 3))
            perfect.append(False)
    return vifs, perfect


def collinearity_report(
    X: np.ndarray,
    names: Sequence[str],
    *,
    pair_threshold: float = PAIR_R_THRESHOLD,
    vif_threshold: float = VIF_THRESHOLD,
) -> dict[str, Any]:
    """Pairs, VIFs and the flagged inputs. Never raises on a degenerate sheet."""
    Z = np.asarray(X, float)
    names = [str(n) for n in names]
    out: dict[str, Any] = {
        "method": "pearson_r_and_vif_on_recipe_means",
        "pair_threshold": pair_threshold,
        "vif_threshold": vif_threshold,
        "label": COLLINEAR_LABEL,
        "pairs": [],
        "vif": {},
        "vif_perfect": [],
        "vif_reason": None,
        "flagged": [],
        "groups": [],
    }
    if Z.ndim != 2 or Z.shape[1] < 2 or Z.shape[0] < 3:
        out["vif_reason"] = "fewer than two inputs or three rows"
        return out
    n, d = Z.shape
    sd = Z.std(axis=0)
    live = [j for j in range(d) if sd[j] > 0 and np.isfinite(Z[:, j]).all()]
    flagged: set[str] = set()
    adj: dict[str, set[str]] = {}
    if len(live) >= 2:
        R = np.corrcoef(Z[:, live], rowvar=False)
        for a in range(len(live)):
            for b in range(a + 1, len(live)):
                r = float(R[a, b])
                if np.isfinite(r) and abs(r) >= pair_threshold:
                    na, nb = names[live[a]], names[live[b]]
                    out["pairs"].append({"a": na, "b": nb, "r": round(r, 3)})
                    flagged.update((na, nb))
                    adj.setdefault(na, set()).add(nb)
                    adj.setdefault(nb, set()).add(na)
    out["pairs"].sort(key=lambda p: -abs(p["r"]))
    if len(live) >= 2 and n > len(live) + 1:
        vifs, perfect = _vif(Z[:, live])
        for k, j in enumerate(live):
            out["vif"][names[j]] = vifs[k]
            if perfect[k]:
                out["vif_perfect"].append(names[j])
                flagged.add(names[j])
            else:
                v = vifs[k]
                if v is not None and v >= vif_threshold:
                    flagged.add(names[j])
    else:
        out["vif_reason"] = (
            f"VIF needs more recipes ({n}) than inputs plus one ({len(live) + 1}); "
            "with this few rows every input is an exact combination of the others"
        )
    # Connected components of the pair graph: the sets whose members cannot be
    # told apart from each other.
    seen: set[str] = set()
    for start in sorted(adj):
        if start in seen:
            continue
        comp: set[str] = set()
        stack = [start]
        while stack:
            node = stack.pop()
            if node in comp:
                continue
            comp.add(node)
            stack.extend(adj[node] - comp)
        seen |= comp
        out["groups"].append(sorted(comp))
    out["flagged"] = [nm for nm in names if nm in flagged]
    return out


def partners(report: dict[str, Any], name: str) -> list[str]:
    """Inputs flagged as a collinear pair with `name`."""
    out = []
    for p in report.get("pairs", []):
        if p["a"] == name:
            out.append(p["b"])
        elif p["b"] == name:
            out.append(p["a"])
    return out


__all__ = [
    "COLLINEAR_LABEL", "PAIR_R_THRESHOLD", "VIF_THRESHOLD", "collinearity_report", "partners",
]
