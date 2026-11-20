"""The `driver_second_opinion` block: rank drivers next to GP relevance.

The driver panel (`drivers`) ranks each continuous input by its Spearman rho
with the target on recipe means, with a bootstrap CI and BH correction. This
block puts a second, independent reading next to it, the ARD relevance of the
proposal GP with a recipe-subsample interval (`kalos.core.gp_relevance`), plus
a collinearity check between inputs (`kalos.core.collinearity`), and lists the
places where they disagree:

- `rank_only`: significant by rank, but every subsample fit gives the input a
  long lengthscale ("strong by rank, irrelevant to the GP"). Usually a
  collinear partner carries the effect in the GP, or the rank association is
  carried by a few rows.
- `gp_only`: every subsample fit gives the input a short lengthscale, but the
  rank CI spans zero ("relevant to the GP, missed by rank"). Usually an
  interior optimum or an interaction, which a monotonic rank statistic cannot
  see (`gp_shapes` shows which).
- `collinear`: the input moves with another input on this sheet
  ("collinear: effects cannot be separated"); neither number can say which of
  them matters.

Nothing here is causal, and the block says so in its own fields.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from kalos.core.collinearity import COLLINEAR_LABEL, partners

RANK_ONLY_LABEL = "strong by rank, irrelevant to the GP"
GP_ONLY_LABEL = "relevant to the GP, missed by rank"

NOT_CAUSAL = (
    "Both readings are associations in observational data. Rank drivers measure a "
    "monotonic association with the target; GP relevance measures how fast the fitted "
    "response changes along an input. Neither is a causal effect, and lengthscales say "
    "nothing about what happens if an input is changed on its own."
)

WHAT_IT_MEASURES = {
    "rank": "Spearman rho with the target on recipe means, bootstrap 95% CI, BH-corrected "
            "significance (the `drivers` panel, all tested inputs)",
    "gp_relevance": "share of inverse ARD lengthscale of the proposal GP (inputs normalized to "
                    "the design box), point from the served fit, interval from refits on "
                    "recipe subsamples",
    "gp_status": "relevant: lengthscale interval at or below the relevant bound; negligible: at "
                 "or above the negligible bound; unclear: otherwise; not_judged: no GP reading",
}


def driver_second_opinion(
    drivers_all: Sequence[dict[str, Any]],
    names: Sequence[str],
    collinearity: dict[str, Any],
    gp: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the block. `drivers_all` are every tested input's rank entries
    (before the top-k cut), `names` the continuous inputs in GP column order,
    `gp` the `subsample_relevance` output or `{available: False, reason}`."""
    from kalos.core.gp_relevance import NEGLIGIBLE_MIN_LS, RELEVANT_MAX_LS

    by_name = {str(d["name"]): d for d in drivers_all}
    gp_ok = bool(gp.get("available"))
    features: list[dict[str, Any]] = []
    disagreements: list[dict[str, Any]] = []
    flagged = set(collinearity.get("flagged", []))
    for j, name in enumerate(names):
        d = by_name.get(str(name), {})
        ci = d.get("ci95")
        rank_sig = bool(d.get("significant", False))
        spans_zero = bool(ci is not None and ci[0] <= 0 <= ci[1])
        status = gp["status"][j] if gp_ok else "not_judged"
        flags: list[str] = []
        if gp_ok and rank_sig and status == "negligible":
            flags.append(RANK_ONLY_LABEL)
            hint = partners(collinearity, str(name))
            disagreements.append({
                "name": str(name), "kind": "rank_only", "label": RANK_ONLY_LABEL,
                "detail": (
                    f"rank rho {d.get('rho')} (CI {ci}) is significant, but the GP lengthscale "
                    f"interval {gp['lengthscale_ci'][j]} is at or above {NEGLIGIBLE_MIN_LS:g}"
                    + (f"; it is collinear with {', '.join(hint)}, which may carry the effect "
                       "in the GP" if hint else "")
                ),
            })
        if gp_ok and status == "relevant" and spans_zero:
            flags.append(GP_ONLY_LABEL)
            disagreements.append({
                "name": str(name), "kind": "gp_only", "label": GP_ONLY_LABEL,
                "detail": (
                    f"GP lengthscale interval {gp['lengthscale_ci'][j]} is at or below "
                    f"{RELEVANT_MAX_LS:g}, but the rank CI {ci} spans zero; a monotonic rank "
                    "statistic cannot see an interior optimum or an interaction (see gp_shapes)"
                ),
            })
        if str(name) in flagged:
            flags.append(COLLINEAR_LABEL)
            with_ = partners(collinearity, str(name))
            vif = collinearity.get("vif", {}).get(str(name))
            perfect = str(name) in collinearity.get("vif_perfect", [])
            disagreements.append({
                "name": str(name), "kind": "collinear", "label": COLLINEAR_LABEL,
                "detail": (
                    (f"moves with {', '.join(with_)} on this sheet" if with_ else
                     "is close to a linear combination of the other inputs")
                    + ("; VIF infinite (exact linear combination)" if perfect else
                       (f"; VIF {vif}" if vif is not None else ""))
                    + "; the data cannot say which of them matters"
                ),
            })
        features.append({
            "name": str(name),
            "rank_rho": d.get("rho"),
            "rank_ci95": ci,
            "rank_significant": rank_sig,
            "gp_relevance": gp["relevance"][j] if gp_ok else None,
            "gp_relevance_ci95": gp["relevance_ci"][j] if gp_ok else None,
            "gp_lengthscale": gp["lengthscale"][j] if gp_ok else None,
            "gp_lengthscale_ci95": gp["lengthscale_ci"][j] if gp_ok else None,
            "gp_status": status,
            "collinear_with": partners(collinearity, str(name)),
            "flags": flags,
        })
    features.sort(key=lambda f: -abs(float(f["rank_rho"] or 0.0)))
    gp_meta = {k: gp.get(k) for k in ("available", "reason", "method", "level", "n_subsamples",
                                      "n_fits_used", "fraction")}
    gp_meta.update({
        "source": "ard_lengthscales_of_the_proposal_gp",
        "relevant_max_lengthscale": RELEVANT_MAX_LS,
        "negligible_min_lengthscale": NEGLIGIBLE_MIN_LS,
        "units": "fraction of each input's design range",
    })
    return {
        "available": bool(names),
        "reason": None if names else "no continuous inputs to compare",
        "what_it_measures": WHAT_IT_MEASURES,
        "not_causal": NOT_CAUSAL,
        "gp": gp_meta,
        "features": features,
        "disagreements": disagreements,
        "collinearity": collinearity,
    }


__all__ = ["GP_ONLY_LABEL", "NOT_CAUSAL", "RANK_ONLY_LABEL", "driver_second_opinion"]
