"""F10: is the planned target run inside the mixing and aeration regime the
measured runs cover?

Every scale-up readout extrapolates on the scale axis (that is its job, and
`readout.decide` judges the step ratio against the ladder backtest). But the
model also reads the agitation- and aeration-derived physics features (power
per volume, superficial gas velocity, kLa), and a planned run can sit far
outside the range of those features even when the step ratio looks fine:
2,000 L at 240 rpm, from a ladder that stops at 300 L, puts power per volume
three times above anything measured. The number that comes back is then an
extrapolation in the physics the model leans on, with an interval calibrated
on runs that never saw that regime.

This module computes, at the target and over the measured runs, with the SAME
geometry, power-number and van't Riet assumptions as the readout's features:

- power per volume P/V (W/m3) and impeller tip speed (m/s), from agitation;
- superficial gas velocity (m/s) and gas flow per volume (vvm), from airflow;
- the kLa proxy (1/s), from both.

and compares each target value with the measured [min, max]:

- inside: nothing;
- outside (by more than `PHYSICS_EDGE_TOLERANCE`, 5 %): a warning naming the
  quantity, the values and the measured range;
- more than `PHYSICS_REFUSE_FACTOR` (2x) beyond the nearest measured value, for
  P/V, vvm or kLa (what the usual scale-up rules hold constant): a refusal.
  Tip speed and gas velocity rise with scale under those rules, so they only
  ever warn.

Each outside quantity that one setting controls carries the setting that
would bring it back inside the measured range (P/V scales with rpm cubed at a
fixed volume, tip speed with rpm, gas velocity and vvm with airflow), which is
the "what data or plan would fix it" line. These are proxies under stated
assumptions, not a vessel design; they are ranked against the measured runs
only, never against engineering limits, which depend on the vessel and the
cells.

Only the "physics" feature set has agitation and airflow; with "scale_only"
there is nothing to check here and the module says so.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .features import compute_scale_features, tank_geometry_proxy

PHYSICS_REFUSE_FACTOR = 2.0
# Within 5 % beyond the measured range counts as inside: the quantities are
# proxies from rounded settings (300 L/min at 7,500 L is 0.0400 vvm against a
# measured 0.0401), and a warning there would be noise.
PHYSICS_EDGE_TOLERANCE = 1.05


@dataclass(frozen=True)
class _Quantity:
    key: str
    label: str
    unit: str
    driver: str | None  # the planned setting that controls it at a fixed volume
    exponent: float  # quantity ~ driver ** exponent at a fixed volume
    can_refuse: bool  # see the note below


# Tip speed and superficial gas velocity rise with scale under the usual
# scale-up rules (constant P/V gives tip speed ~ V^(1/9); constant vvm gives gas
# velocity ~ V^(1/3)), so a target beyond the measured scales nearly always sits
# above their measured range. That is stated as a warning, never a refusal on
# its own. P/V, vvm and kLa are what those rules hold constant; far outside the
# measured range they can refuse.
_QUANTITIES = (
    _Quantity("specific_power_w_per_m3", "power per volume", "W/m3", "agitation_rpm", 3.0, True),
    _Quantity("tip_speed_m_per_s", "impeller tip speed", "m/s", "agitation_rpm", 1.0, False),
    _Quantity("superficial_gas_velocity_m_per_s", "superficial gas velocity", "m/s",
              "airflow_L_per_min", 1.0, False),
    _Quantity("vvm_per_min", "gas flow per liquid volume", "vvm", "airflow_L_per_min", 1.0, True),
    _Quantity("kla_proxy_per_s", "kLa (proxy)", "1/s", None, 1.0, True),
)


def physics_frame(df: pd.DataFrame, config: Any) -> pd.DataFrame:
    """The five checked quantities for each row of `df` (scale, agitation and
    airflow columns named by `config`, a `ScaleFeatureConfig`)."""
    feats = compute_scale_features(
        df, scale_column=config.scale_column, agitation_column=config.agitation_column,
        airflow_column=config.airflow_column, reference_scale_L=config.reference_scale_L,
        geometry=config.geometry, power=config.power, kla_params=config.kla_params,
    )
    scale = pd.to_numeric(df[config.scale_column], errors="coerce").to_numpy(float)
    rpm = pd.to_numeric(df[config.agitation_column], errors="coerce").to_numpy(float)
    air = pd.to_numeric(df[config.airflow_column], errors="coerce").to_numpy(float)
    impeller = tank_geometry_proxy(scale, config.geometry)["impeller_diameter_m"].to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        out = pd.DataFrame({
            "specific_power_w_per_m3": feats["specific_power_w_per_m3"].to_numpy(float),
            "tip_speed_m_per_s": np.pi * impeller * rpm / 60.0,
            "superficial_gas_velocity_m_per_s": feats["superficial_gas_velocity_m_per_s"].to_numpy(float),
            "vvm_per_min": air / np.where(scale > 0, scale, np.nan),
            "kla_proxy_per_s": feats["kla_proxy_per_s"].to_numpy(float),
        }, index=df.index)
    return out


def _fmt(v: float) -> str:
    return f"{v:,.3g}" if abs(v) < 1000 else f"{v:,.0f}"


def check_physics_range(clean_df: pd.DataFrame, target: Any, config: Any) -> dict[str, Any]:
    """Compare the target run's physics with the measured runs' (see module
    docstring). Returns `{checked, decision: none|warn|refuse, reasons,
    quantities: [...], refuse_factor}`; `checked` False (with a `reason`) when
    the sheet has no agitation/airflow."""
    if config.feature_set != "physics" or target.agitation_rpm is None or target.airflow_L_per_min is None:
        return {"checked": False, "decision": "none", "reasons": [], "quantities": [],
                "reason": "the sheet records no agitation and airflow, so mixing and aeration cannot be checked"}
    measured = physics_frame(clean_df, config)
    at = physics_frame(pd.DataFrame([{
        config.scale_column: target.scale_L, config.agitation_column: target.agitation_rpm,
        config.airflow_column: target.airflow_L_per_min,
    }]), config).iloc[0]
    max_scale = float(pd.to_numeric(clean_df[config.scale_column], errors="coerce").max())
    plan = {"agitation_rpm": float(target.agitation_rpm), "airflow_L_per_min": float(target.airflow_L_per_min)}
    quantities: list[dict[str, Any]] = []
    reasons: list[str] = []
    any_outside = refuse = False
    for q in _QUANTITIES:
        col = measured[q.key].to_numpy(float)
        col = col[np.isfinite(col) & (col > 0)]
        value = float(at[q.key])
        if len(col) == 0 or not np.isfinite(value) or value <= 0:
            continue
        lo, hi = float(col.min()), float(col.max())
        factor = value / hi if value > hi else (lo / value if value < lo else 1.0)
        status = "inside" if factor <= PHYSICS_EDGE_TOLERANCE else ("far_outside" if factor > PHYSICS_REFUSE_FACTOR else "outside")
        refuses = status == "far_outside" and q.can_refuse
        entry: dict[str, Any] = {
            "quantity": q.key, "label": q.label, "unit": q.unit, "target": value,
            "measured_min": lo, "measured_max": hi, "factor_outside": round(factor, 3), "status": status,
            "setting_to_stay_inside": None, "refuses": refuses,
        }
        if status != "inside":
            any_outside = True
            refuse = refuse or refuses
            side = "above the highest" if value > hi else "below the lowest"
            fix = ""
            if q.driver is not None:
                bound = hi if value > hi else lo
                setting = plan[q.driver] * (bound / value) ** (1.0 / q.exponent)
                entry["setting_to_stay_inside"] = {q.driver: round(setting, 3)}
                unit = "rpm" if q.driver == "agitation_rpm" else "L/min"
                fix = (f" Planning {q.driver} at about {_fmt(setting)} {unit} instead of "
                       f"{_fmt(plan[q.driver])} would keep it inside the measured range.")
            reasons.append(
                f"At {_fmt(target.scale_L)} L, {_fmt(target.agitation_rpm)} rpm and "
                f"{_fmt(target.airflow_L_per_min)} L/min the estimated {q.label} is {_fmt(value)} {q.unit}, "
                f"{factor:.1f}x {side} value in the measured runs ({_fmt(lo)}-{_fmt(hi)} {q.unit}, runs up to "
                f"{_fmt(max_scale)} L): the model has not seen this mixing or aeration regime.{fix}"
            )
        quantities.append(entry)
    decision = "refuse" if refuse else ("warn" if any_outside else "none")
    if decision == "refuse":
        reasons.append(
            f"With power per volume, gas flow per volume or kLa more than {PHYSICS_REFUSE_FACTOR:g}x outside "
            "the measured runs, a number would be a guess. "
            "Plan the target inside the measured range, or add runs (at the largest scale you can) at the "
            "planned agitation and airflow, then ask again."
        )
    return {"checked": True, "decision": decision, "reasons": reasons, "quantities": quantities,
            "refuse_factor": PHYSICS_REFUSE_FACTOR,
            "basis": "geometric proxies under the readout's stated physics assumptions; compared with the measured runs only"}


def extrapolation_statement(target_scale_L: float, max_scale_L: float | None) -> dict[str, Any] | None:
    """The plain fact every readout states: how far beyond the measured scales
    the target is."""
    if max_scale_L is None or not np.isfinite(max_scale_L) or max_scale_L <= 0:
        return None
    factor = float(target_scale_L) / float(max_scale_L)
    if factor <= 1.0:
        return {"target_scale_L": float(target_scale_L), "max_measured_scale_L": float(max_scale_L),
                "factor": round(factor, 3), "message": (
                    f"{_fmt(target_scale_L)} L is within the measured scales (up to {_fmt(max_scale_L)} L).")}
    return {
        "target_scale_L": float(target_scale_L), "max_measured_scale_L": float(max_scale_L),
        "factor": round(factor, 3),
        "message": (
            f"{_fmt(target_scale_L)} L is {factor:.1f}x the largest scale measured ({_fmt(max_scale_L)} L). "
            "No run on this sheet was made at the target scale, so any number here is an extrapolation; "
            "a run at or near the target scale is the only data that settles it."
        ),
    }


__all__ = ["PHYSICS_EDGE_TOLERANCE", "PHYSICS_REFUSE_FACTOR", "check_physics_range", "extrapolation_statement", "physics_frame"]
