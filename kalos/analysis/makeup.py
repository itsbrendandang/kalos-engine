"""Media make-up arithmetic: stock volumes and water top-up for one recipe.

Given the target concentration of each component in one run, the user's stock
solutions and the final volume, this module computes how much of each stock to
add and how much water tops the run up to volume, from C1*V1 = C2*V2:

    V_stock = (C_target / C_stock) * V_final      (both concentrations in one basis)
    V_water = V_final - sum V_stock

It is arithmetic from the user's declared stocks, not a model output; every
result carries that label. Nothing here reads files, talks to a database or
fits anything, and nothing is guessed: an impossible or ambiguous case is a
`Refusal` with a reason and `what_would_fix_it`, never a weaker answer.

Refusals (``Refusal.code``):

- ``negative_water``: the stock volumes add up to more than the final volume.
- ``stock_too_dilute``: a stock is less concentrated than its target, so it
  alone would need more than the final volume.
- ``unit_mismatch_no_mw``: target and stock are in mass and molar units (for
  example g/L and mM) and the stock declares no molecular weight.
- ``unknown_unit``, ``ambiguous_percent``, ``incompatible_units``: a unit this
  module cannot convert, a bare "%" (w/v, v/v or w/w is not said), or a pair that
  needs a density (% v/v or % w/w against a mass concentration).
- ``no_stock``, ``duplicate_component``, ``duplicate_stock``: a component with a
  non-zero target and no stock, a component listed twice in one recipe (it
  would be counted twice), or two stocks for one component (which one is used
  is a choice for the user, not for this module).
- ``invalid_final_volume``, ``invalid_target``, ``invalid_stock``,
  ``invalid_mw``: a number that is missing, not finite, negative or zero where
  a positive value is needed.

Units. Target and stock in the same unit need no conversion (the ratio is all
that matters), so any unit works when both sides use it, including "X" stocks.
Otherwise both sides are brought to one basis:

- mass concentration through the engine's one unit registry,
  `kalos.normalize.units` (g/L and mg/mL today; any mass-concentration unit
  added there later works here with no change);
- % w/v, defined as grams per 100 mL, so 1 % w/v = 10 g/L;
- molar concentration (M, mM, µM, nM, mol/L, mmol/L, µmol/L), by its SI
  prefix. Mass and molar meet only through the molecular weight declared on
  the stock (g/mol): g/L = mol/L * MW.

The registry does not carry molar or % w/v units yet; the two small tables
below hold only what make-up needs and should move into
`kalos.normalize.units` when that registry gains them.

The molecular weight must be that of the form the mass concentration refers to
(for example the heptahydrate if the stock was made by weighing the hydrate).

The water volume assumes volumes add. With concentrated stocks (50 % glycerol,
for example) they do not quite, and pH titrant is not counted, so at the bench
the water figure is a guide: bring the run to its final volume.

Arithmetic is exact: every number is read as the decimal it was written as
(``0.1`` is one tenth) and computed with `fractions.Fraction`; only the
returned volumes are rounded, once, to the nearest float. Volumes are not
snapped to a pipetting resolution here.

Component names are matched exactly (after trimming spaces); a near-miss
spelling is a ``no_stock`` refusal, never a fuzzy match.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Literal

from kalos.core.gates import _finite
from kalos.normalize.units import canonical_suffix, convert, is_known_unit

LABEL = "arithmetic from your stocks; not a model output"

_MICRO = str.maketrans({"µ": "u", "μ": "u"})  # MICRO SIGN, GREEK SMALL LETTER MU

# Molar units, to mol/L. Matched case-sensitively: "mM" is millimolar, "MM" is not a unit.
_MOLAR_CASE_SENSITIVE: dict[str, Fraction] = {
    "M": Fraction(1),
    "mM": Fraction(1, 10**3),
    "uM": Fraction(1, 10**6),
    "nM": Fraction(1, 10**9),
}
# The spelled-out forms are unambiguous in any case.
_MOLAR_ANY_CASE: dict[str, Fraction] = {
    "mol/l": Fraction(1),
    "mmol/l": Fraction(1, 10**3),
    "umol/l": Fraction(1, 10**6),
}
# % w/v is grams per 100 mL: 1 % w/v = 10 g/L.
_W_V_TOKENS = frozenset({"%w/v", "%(w/v)", "w/v%", "g/100ml"})
_W_V_G_PER_L = Fraction(10)
_BARE_PERCENT = frozenset({"%", "pct", "percent"})
_DENSITY_PERCENT = frozenset({"%v/v", "%(v/v)", "v/v%", "%w/w", "%(w/w)", "w/w%"})


@dataclass(frozen=True)
class Stock:
    """One stock solution: its concentration, unit and optional molecular weight (g/mol)."""

    component: str
    concentration: float
    unit: str
    mw_g_per_mol: float | None = None


@dataclass(frozen=True)
class Target:
    """The concentration one component should have in the finished run."""

    component: str
    concentration: float
    unit: str


@dataclass(frozen=True)
class Refusal:
    code: str
    component: str | None
    reason: str
    what_would_fix_it: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "component": self.component,
            "reason": self.reason,
            "what_would_fix_it": self.what_would_fix_it,
        }


@dataclass(frozen=True)
class ComponentVolume:
    """How much of one stock goes into the run. `volume_ml` is None when it could not be computed."""

    component: str
    target: float
    target_unit: str
    stock: float | None
    stock_unit: str | None
    volume_ml: float | None
    basis: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "component": self.component,
            "target": self.target,
            "target_unit": self.target_unit,
            "stock": self.stock,
            "stock_unit": self.stock_unit,
            "volume_ml": self.volume_ml,
            "basis": self.basis,
        }


@dataclass(frozen=True)
class MakeUp:
    """The make-up of one run: per-component stock volumes and the water top-up.

    `status` is "refused" when any refusal applies; `water_ml` is then None.
    Component lines that could be computed are still listed, so the reason
    for a `negative_water` refusal can be read off the volumes.
    """

    status: Literal["ok", "refused"]
    final_volume_ml: float
    components: tuple[ComponentVolume, ...]
    water_ml: float | None
    refusals: tuple[Refusal, ...] = field(default_factory=tuple)
    label: str = LABEL

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "label": self.label,
            "final_volume_ml": self.final_volume_ml,
            "components": [c.as_dict() for c in self.components],
            "water_ml": self.water_ml,
            "refusals": [r.as_dict() for r in self.refusals],
        }


@dataclass(frozen=True)
class _Basis:
    kind: Literal["mass", "molar"]
    factor: Fraction  # to g/L (mass) or mol/L (molar)


class _UnitProblem(Exception):
    def __init__(self, code: str, reason: str, fix: str) -> None:
        super().__init__(reason)
        self.code, self.reason, self.fix = code, reason, fix


def _exact(x: float) -> Fraction:
    """The decimal a person wrote, exactly: 0.1 -> 1/10 (not the binary float's value)."""
    return Fraction(repr(float(x)))


def _fold(unit: str) -> str:
    return "".join(unit.translate(_MICRO).split())


def _num(x: Fraction) -> str:
    return f"{float(x):g}"


_KNOWN_UNITS_TEXT = "g/L, mg/mL, % w/v, M, mM, µM, nM"


def _basis(unit: str) -> _Basis:
    folded = _fold(unit)
    low = folded.lower()
    if folded in _MOLAR_CASE_SENSITIVE:
        return _Basis("molar", _MOLAR_CASE_SENSITIVE[folded])
    if low in _MOLAR_ANY_CASE:
        return _Basis("molar", _MOLAR_ANY_CASE[low])
    if low in _W_V_TOKENS:
        return _Basis("mass", _W_V_G_PER_L)
    if low in _BARE_PERCENT:
        raise _UnitProblem(
            "ambiguous_percent",
            f"'{unit}' does not say whether it is % w/v, % v/v or % w/w",
            "write the unit as '% w/v' (grams per 100 mL), or give the stock and the target in the same unit",
        )
    if low in _DENSITY_PERCENT:
        raise _UnitProblem(
            "incompatible_units",
            f"'{unit}' cannot be converted to a mass concentration without the solution's density",
            "give the stock and the target in the same unit, or express both as a mass concentration (g/L)",
        )
    if is_known_unit(unit) and canonical_suffix(unit) == "_g_l":
        factor, _ = convert(1.0, unit)
        return _Basis("mass", _exact(factor))
    raise _UnitProblem(
        "unknown_unit",
        f"'{unit}' is not a concentration unit make-up can convert",
        f"give the stock and the target in the same unit, or use one of: {_KNOWN_UNITS_TEXT}",
    )


def _in_one_basis(target: Target, stock: Stock) -> tuple[Fraction, Fraction, str]:
    """Target and stock concentrations in one basis, and a note saying which. Raises _UnitProblem."""
    t, s = _exact(target.concentration), _exact(stock.concentration)
    if _fold(target.unit) == _fold(stock.unit):
        return t, s, f"same unit ({target.unit})"
    tb, sb = _basis(target.unit), _basis(stock.unit)
    if tb.kind == sb.kind:
        name = "g/L" if tb.kind == "mass" else "mol/L"
        return t * tb.factor, s * sb.factor, f"both converted to {name}"
    mw = stock.mw_g_per_mol
    if mw is None:
        raise _UnitProblem(
            "unit_mismatch_no_mw",
            f"the target is in {target.unit} and the stock in {stock.unit}; "
            "converting between mass and molar units needs the molecular weight",
            f"declare the molecular weight (g/mol) of {stock.component} on its stock, "
            "or give the stock and the target in the same unit",
        )
    mw_x = _exact(mw)
    t_mass = t * tb.factor * (mw_x if tb.kind == "molar" else 1)
    s_mass = s * sb.factor * (mw_x if sb.kind == "molar" else 1)
    return t_mass, s_mass, f"converted to g/L using MW {_num(mw_x)} g/mol (declared on the stock)"


def _invalid_number(what: str, value: object, positive: bool) -> str | None:
    """Why `value` is not a usable number, or None if it is."""
    if _finite(value):
        if positive and value <= 0:
            return f"{what} is {value:g}; it must be greater than 0"
        if value < 0:
            return f"{what} is {value:g}; it cannot be negative"
        return None
    return f"{what} is {value!r}, not a finite number"


def _line(target: Target, stock: Stock | None, volume: Fraction | None, basis: str) -> ComponentVolume:
    return ComponentVolume(
        target.component, target.concentration, target.unit,
        stock.concentration if stock else None, stock.unit if stock else None,
        float(volume) if volume is not None else None, basis,
    )


def make_up(
    targets: Sequence[Target], stocks: Sequence[Stock], final_volume_ml: float
) -> MakeUp:
    """Stock volumes and water top-up (mL) to make one run of `final_volume_ml` at `targets`.

    `stocks` may hold stocks for components this run does not use. A target of
    0 needs no stock and gets a 0 mL line (its stock, if any, is not checked).
    """
    bad_volume = _invalid_number("the final volume", final_volume_ml, positive=True)
    if bad_volume is not None:
        refusal = Refusal(
            "invalid_final_volume", None, bad_volume,
            "enter the final volume of the run in mL as a number greater than 0",
        )
        return MakeUp("refused", final_volume_ml, (), None, (refusal,))
    v_final = _exact(final_volume_ml)

    refusals: list[Refusal] = []
    by_name: dict[str, list[Stock]] = {}
    for st in stocks:
        by_name.setdefault(st.component.strip(), []).append(st)

    seen: set[str] = set()
    lines: list[ComponentVolume] = []
    volumes: dict[str, Fraction] = {}
    for tg in targets:
        name = tg.component.strip()

        if name in seen:
            refusals.append(Refusal(
                "duplicate_component", tg.component,
                f"{tg.component} appears more than once in this recipe; it would be counted twice",
                f"list {tg.component} once, with its total target concentration",
            ))
            continue
        seen.add(name)

        bad = _invalid_number(f"the target of {tg.component}", tg.concentration, positive=False)
        if bad is not None:
            refusals.append(Refusal(
                "invalid_target", tg.component, bad,
                f"enter the target concentration of {tg.component} as a number of 0 or more",
            ))
            lines.append(_line(tg, None, None, "not computed"))
            continue

        if tg.concentration == 0:
            volumes[name] = Fraction(0)
            lines.append(_line(tg, None, Fraction(0), "target is 0; nothing to add"))
            continue
        candidates = by_name.get(name, [])
        if not candidates:
            refusals.append(Refusal(
                "no_stock", tg.component,
                f"no stock is declared for {tg.component} (names must match exactly)",
                f"declare a stock solution for {tg.component}, or check its name matches the recipe",
            ))
            lines.append(_line(tg, None, None, "not computed"))
            continue
        if len(candidates) > 1:
            refusals.append(Refusal(
                "duplicate_stock", tg.component,
                f"{len(candidates)} stocks are declared for {tg.component}; which one to use is not stated",
                f"keep one stock for {tg.component} in the list passed to make-up",
            ))
            lines.append(_line(tg, None, None, "not computed"))
            continue
        stock = candidates[0]

        bad_stock = _invalid_number(f"the stock concentration of {tg.component}", stock.concentration, positive=True)
        bad_mw = (
            _invalid_number(f"the molecular weight of {tg.component}", stock.mw_g_per_mol, positive=True)
            if stock.mw_g_per_mol is not None else None
        )
        if bad_stock is not None or bad_mw is not None:
            if bad_stock is not None:
                refusals.append(Refusal(
                    "invalid_stock", tg.component, bad_stock,
                    f"enter the stock concentration of {tg.component} as a number greater than 0",
                ))
            if bad_mw is not None:
                refusals.append(Refusal(
                    "invalid_mw", tg.component, bad_mw,
                    f"enter the molecular weight of {tg.component} in g/mol as a number greater than 0",
                ))
            lines.append(_line(tg, stock, None, "not computed"))
            continue

        try:
            c_target, c_stock, basis = _in_one_basis(tg, stock)
        except _UnitProblem as p:
            refusals.append(Refusal(p.code, tg.component, p.reason, p.fix))
            lines.append(_line(tg, stock, None, "not computed"))
            continue

        volume = c_target / c_stock * v_final
        lines.append(_line(tg, stock, volume, basis))
        if c_stock < c_target:
            refusals.append(Refusal(
                "stock_too_dilute", tg.component,
                f"the {tg.component} stock ({stock.concentration:g} {stock.unit}) is less concentrated than "
                f"the target ({tg.concentration:g} {tg.unit}); it would need {_num(volume)} mL in a "
                f"{_num(v_final)} mL run",
                f"make a stock of {tg.component} more concentrated than the target, or lower the target",
            ))
            continue
        volumes[name] = volume

    if not refusals:
        total = sum(volumes.values(), Fraction(0))
        if total > v_final:
            biggest = sorted(volumes, key=lambda k: volumes[k], reverse=True)[:3]
            refusals.append(Refusal(
                "negative_water", None,
                f"the stocks add up to {_num(total)} mL, {_num(total - v_final)} mL more than the "
                f"{_num(v_final)} mL final volume",
                "use more concentrated stocks for the largest additions ("
                + ", ".join(f"{k}: {_num(volumes[k])} mL" for k in biggest)
                + "), lower the targets, or make a larger final volume",
            ))
        else:
            return MakeUp("ok", final_volume_ml, tuple(lines), float(v_final - total))
    return MakeUp("refused", final_volume_ml, tuple(lines), None, tuple(refusals))


__all__ = ["LABEL", "ComponentVolume", "MakeUp", "Refusal", "Stock", "Target", "make_up"]
