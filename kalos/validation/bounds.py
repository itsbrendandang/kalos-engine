"""Physically-possible-range registry for bioprocess measurement columns.

A run sheet's numeric columns are not free-form: "pH" cannot be 40 and a
titer cannot be negative, regardless of what a client's export script wrote
into the cell. This module is a small, data-driven table mapping a
measurement DIMENSION (ph, temperature_c, ...) to two nested ranges:

  - HARD range: values outside it are physically impossible for that
    dimension, full stop (a negative concentration, a pH above 14). A
    violation is a data bug, not a surprising-but-real measurement, so
    `checks.check_physical_bounds` reports it as an ERROR.
  - TYPICAL range: the inner band a value is expected to fall in during a
    normal bioprocess run. A value outside TYPICAL but still inside HARD is
    physically possible but operationally suspicious (pH 2.5 in a mammalian
    culture, a 5 C incubation) - worth a human's attention, so it is a
    WARNING, not an ERROR.

`DIMENSION_BOUNDS` is intentionally small and every number below is
justified in its own comment. Where a bound is a physical law (0-14 for pH,
non-negative concentration) that is stated. Where a bound is a convention or
heuristic chosen for lack of a sharper physical constraint, that is stated
too - this table does not manufacture false precision.

Column -> dimension inference is by header regex (`infer_dimension`), not by
looking at the data. A column whose header does not match any known pattern
gets no dimension and therefore no bounds check: silence, not a guess.

Base units for the "unbounded above" dimensions match `kalos.normalize.units`
exactly (concentration -> g/L, flow_rate -> mL/h, time -> hours) so
`checks.check_physical_bounds` can convert every cell to its base unit via
`units.convert` before comparing against these ranges - a Fahrenheit reading
must never be compared against a Celsius bound.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from kalos.normalize.header_units import parse_header_unit, unit_atoms

# --- range definitions ----------------------------------------------------- #


@dataclass(frozen=True)
class DimensionBounds:
    """Two nested ranges for one measurement dimension, in its base unit.

    `hard_lo`/`hard_hi` bound what is physically possible; `typical_lo`/
    `typical_hi` bound what is operationally expected. `typical` is always a
    subset of `hard` (`hard_lo <= typical_lo <= typical_hi <= hard_hi`).
    `unit` is the base unit these numbers are expressed in (matches
    `kalos.normalize.units`'s base unit for the same dimension, or "" for a
    dimensionless quantity like pH/OD).
    """

    dimension: str
    hard_lo: float
    hard_hi: float
    typical_lo: float
    typical_hi: float
    unit: str


_INF = float("inf")

DIMENSION_BOUNDS: dict[str, DimensionBounds] = {
    # pH: 0-14 is the textbook range for the aqueous [H+] scale used in every
    # bioprocess context here; going outside it is not "unusual", it is not
    # pH. Typical band 3-9 is a convention (not physics) wide enough to cover
    # microbial fermentation (E. coli ~6.8-7.4), mammalian culture (~6.8-7.4)
    # and the acidic fungal and yeast processes (Trichoderma and Aspergillus
    # fermentations and yeast ethanol runs routinely sit at pH 3-4; the earlier
    # 4.0 floor flagged them as atypical). A reading outside 3-9 but inside 0-14
    # is possible (a failed pH probe, an off-spec run, a citric-acid process) but
    # worth a human looking at it.
    "ph": DimensionBounds("ph", hard_lo=0.0, hard_hi=14.0, typical_lo=3.0, typical_hi=9.0, unit=""),
    # Temperature, base Celsius (per units.py). Hard range -20..150 C is the
    # spec's own choice: -20 C covers a freezer/cold-chain misread and 150 C
    # covers autoclave/CIP temperatures (typically ~121-134 C) with headroom;
    # neither a bioprocess run nor its supporting equipment operates outside
    # that band, so it is treated as a hard physical ceiling for this domain
    # rather than a law of physics. Typical band 15-45 C is a convention
    # covering ambient bench temperature through the hottest common microbial
    # incubation (~42 C) and mammalian culture (~37 C) with margin; a reading
    # outside that (e.g. 5 C) is possible (fridge storage, cold reagent) but
    # not a normal run temperature.
    "temperature_c": DimensionBounds(
        "temperature_c", hard_lo=-20.0, hard_hi=150.0, typical_lo=15.0, typical_hi=45.0, unit="_c"
    ),
    # Percent-scale readings (DO saturation %, viability %, purity %). Hard
    # range 0-100 is the definition of a percentage of a whole; this table
    # does not special-case supersaturated DO readings above 100% because the
    # spec's own dimension definition is 0..100. DO / viability / purity are
    # genuinely different metrics with different "normal" sub-ranges (a
    # healthy DO setpoint, a healthy viability, a target purity are not the
    # same numbers), and picking one typical band across all of them would be
    # a guess this table refuses to make - so `typical` equals `hard` here:
    # this dimension only ever produces ERROR (outside 0-100), never WARNING.
    "percent": DimensionBounds("percent", hard_lo=0.0, hard_hi=100.0, typical_lo=0.0, typical_hi=100.0, unit="_pct"),
    # Optical density (OD600-style biomass proxy), dimensionless. Hard upper
    # bound 200 and typical upper bound 100 are conventions, not physics: most
    # bench spectrophotometers read reliably up to roughly OD 1-4 undiluted
    # and require serial dilution beyond that, but well-run fed-batch cultures
    # are routinely reported well past that after dilution-corrected readout,
    # commonly into the tens. 100 is chosen as a generous typical ceiling
    # covering essentially all diluted readings; 200 as a hard ceiling beyond
    # which a transcription error or missing dilution factor is far more
    # likely than genuine biomass. Negative OD is physically impossible
    # (absorbance/turbidity cannot go negative), hence the 0 floor.
    "od": DimensionBounds("od", hard_lo=0.0, hard_hi=200.0, typical_lo=0.0, typical_hi=100.0, unit=""),
    # Concentration, base g/L (per units.py). Negative concentration is
    # physically impossible (hard floor 0), no hard ceiling exists - solute
    # solubility limits vary by orders of magnitude across the columns this
    # dimension covers (a dilute titer vs. a concentrated feed stock), so an
    # upper HARD bound would be a fiction. Typical upper bound 500 g/L is a
    # heuristic guard-rail, not physics: it sits above nearly all reported
    # protein titers (commonly single- to double-digit g/L) and above most
    # feed-stock concentrations (glucose feeds are often in the 300-700 g/L
    # range, so this is intentionally generous), flagging only genuinely
    # extreme values as worth a second look.
    "concentration": DimensionBounds(
        "concentration", hard_lo=0.0, hard_hi=_INF, typical_lo=0.0, typical_hi=500.0, unit="_g_l"
    ),
    # Flow / feed rate, base mL/h (per units.py). Negative flow is physically
    # impossible for a feed/flow rate as recorded on a run sheet (hard floor
    # 0); no hard ceiling exists. Typical upper bound 10,000 mL/h (10 L/h) is
    # a heuristic guard-rail sized to bench/pilot-scale bioreactors (roughly
    # 0.1 L to a few thousand L working volume); it is a convention about the
    # scale of equipment this domain profile targets, not a law of nature.
    "flow_rate": DimensionBounds(
        "flow_rate", hard_lo=0.0, hard_hi=_INF, typical_lo=0.0, typical_hi=10_000.0, unit="_ml_h"
    ),
    # Elapsed time, base hours (per units.py). Negative elapsed time is
    # impossible (hard floor 0); no hard ceiling exists. Typical upper bound
    # 720 h (30 days) is a heuristic guard-rail: most bioprocess runs
    # (fermentation or cell culture) complete well within a few weeks, so a
    # value past that is more often a unit slip (minutes logged as hours)
    # than a genuinely month-long run, but it is not physically impossible.
    "time_h": DimensionBounds("time_h", hard_lo=0.0, hard_hi=_INF, typical_lo=0.0, typical_hi=720.0, unit="_h"),
    # Viable / total cell density (VCD, TCD). Negative cell density is
    # impossible (hard floor 0). No TYPICAL band is offered and typical is set
    # equal to hard on purpose: the recorded unit for this quantity is not
    # standardized across run sheets (cells/mL, 1e5 cells/mL, and 1e6 cells/mL
    # are all common, and none of them appear in the unit registry), so the same
    # healthy culture legitimately reads 2e7, 200, or 20 depending only on the
    # column's unspoken scale factor. Any typical ceiling here would therefore
    # flag correct data roughly as often as wrong data. This dimension catches
    # the one thing that is unit-independent and genuinely impossible: a
    # negative count.
    "cell_density": DimensionBounds(
        "cell_density", hard_lo=0.0, hard_hi=_INF, typical_lo=0.0, typical_hi=_INF, unit=""
    ),
    # Osmolality, conventionally mOsm/kg. Negative is impossible (hard floor 0).
    # Typical band 200-600 mOsm/kg is a convention, not physics: mammalian cell
    # culture media sit near physiological osmolality (roughly 290 mOsm/kg) and
    # are commonly run up toward 400-450 in fed-batch as feeds accumulate, so
    # 200-600 covers normal operation with margin on both sides. A reading
    # outside it is possible (a deliberately hyperosmotic shift, a diluted
    # sample) but worth a human's attention.
    "osmolality": DimensionBounds(
        "osmolality", hard_lo=0.0, hard_hi=_INF, typical_lo=200.0, typical_hi=600.0, unit=""
    ),
    # Dissolved gas partial pressure (pCO2, pO2). Negative is impossible (hard
    # floor 0). typical equals hard deliberately, for the same reason as
    # cell_density: run sheets record this in mmHg, kPa, or as percent of
    # saturation interchangeably, and those scales differ by more than any
    # sensible band could straddle (a normal pCO2 is about 50 mmHg, about 7 kPa,
    # or about 7 percent). Only the non-negative floor is asserted.
    "partial_pressure": DimensionBounds(
        "partial_pressure", hard_lo=0.0, hard_hi=_INF, typical_lo=0.0, typical_hi=_INF, unit=""
    ),
    # Non-negative elapsed counters: culture day, elapsed generation number,
    # cumulative population doubling level. Each counts forward from the start
    # of a run and cannot be negative (hard floor 0). No upper bound of either
    # kind is asserted - these are open-ended counters whose plausible ceiling
    # depends entirely on the process, and the time_h dimension already guards
    # run duration where duration is what is actually recorded.
    "nonneg_count": DimensionBounds(
        "nonneg_count", hard_lo=0.0, hard_hi=_INF, typical_lo=0.0, typical_hi=_INF, unit=""
    ),
}

# --- header -> dimension inference ----------------------------------------- #

# Checked in this order; first match wins. Order matters where patterns could
# otherwise collide (e.g. "od" is checked before the broader "percent" group
# so an "OD600" header cannot be swallowed by a percent-ish pattern).
_DIMENSION_HINTS: tuple[tuple[str, re.Pattern[str]], ...] = (
    # "ph" must NOT be a bare substring match. `re.compile(r"ph")` also matches
    # phosphate, phenol, sulphate, morphology, phase and alpha - so a
    # "phosphate_g_L" feed column was being handed pH's hard 0-14 range and a
    # perfectly ordinary 200 g/L phosphate feed came back as a physically
    # impossible ERROR. In strict mode that rejects a valid client dataset, which
    # is a worse failure than the one the check exists to catch. Require the
    # token to stand alone instead: "pH", "pH_setpoint", "culture pH (units)"
    # match; "phosphate" and "morphology" do not. The lookarounds use [a-z]
    # under re.I, which also excludes A-Z.
    ("ph", re.compile(r"(?<![a-z])p\.?h(?![a-z])", re.I)),
    ("temperature_c", re.compile(r"temp(erature)?", re.I)),
    ("od", re.compile(r"\bod\d*\b|optical.?density", re.I)),
    (
        "percent",
        # BRE-117: `\bdo\b` never matched "DO_pct" (`_` is a word character), so
        # the shipped example's DO column had no bounds at all and DO = 140 %
        # passed silently. The token must stand alone between non-letters.
        re.compile(r"(?<![a-z])do(?![a-z])|dissolved.?oxygen|viability|purity|%|percent", re.I),
    ),
    # Concentration covers both the product titer and the named metabolites and
    # ions a real run sheet carries. The named-analyte vocabulary below is taken
    # from the 25 parameters recorded in the AstraZeneca CHO process dataset
    # (Gangadharan et al. 2021, 106 cultures, 5 L to 500 L, seven years of
    # industrial practice) rather than invented, so these are headers client
    # sheets genuinely use. Without them a "glucose" or "lactate" column matched
    # no dimension at all and a negative concentration passed the gate silently.
    (
        "concentration",
        re.compile(
            r"titer|titre|conc(entration)?|g[./]l|mg[./]ml"
            r"|glucose|glutamine|glutamate|ammoni(um|a)|lactate|bicarbonate"
            r"|sodium|potassium|phosphate|acetate|ethanol|methanol|glycerol",
            re.I,
        ),
    ),
    # Known ambiguity, called out rather than papered over: because
    # `concentration` is tested before `flow_rate`, a header naming both an
    # analyte and a rate ("sodium hydroxide feed rate") resolves to
    # concentration, not flow. Both dimensions share the non-negative hard floor
    # that catches the real defect class, so the only cost is comparing against
    # the wrong TYPICAL ceiling - a possible spurious warning, never a spurious
    # error. Reordering would simply move the ambiguity onto "glucose feed g/L".
    ("flow_rate", re.compile(r"feed|flow|rate", re.I)),
    ("time_h", re.compile(r"\btime\b|hour|duration|\bage\b", re.I)),
    ("cell_density", re.compile(r"cell.?density|\bvcd\b|\btcd\b|viable.?cell|total.?cell", re.I)),
    ("osmolality", re.compile(r"osmolality|osmolarity|\bosmo\b|mosm", re.I)),
    # Checked after flow_rate on purpose: "pCO2" reaches this pattern, while
    # "CO2 flow rate" is claimed by flow_rate first, which is the correct
    # reading of each header.
    ("partial_pressure", re.compile(r"\bp?co2\b|pco₂|\bpo2\b|partial.?pressure", re.I)),
    ("nonneg_count", re.compile(r"culture.?day|\bday\b|generation.?number|doubling", re.I)),
)


def infer_dimension(column_name: str) -> str | None:
    """Guess a column's measurement dimension from its header text alone.

    Returns a key into `DIMENSION_BOUNDS`, or `None` if nothing matches - a
    `None` result means "no bounds check for this column", not "assume some
    default dimension". Matching is deliberately loose (substring, case
    insensitive) to catch real-world header variants ("Temp (C)", "DO%",
    "feed_rate_mL_h"); the tradeoff, stated plainly, is that a broad pattern
    like "rate" for flow_rate can also match an unrelated "growth_rate"
    column and misclassify it. That is an accepted false-positive risk for a
    silent-skip-by-default design: false positives here cost one avoidable
    bounds check, false negatives cost a missed impossible value entirely.
    """
    for dim, pattern in _DIMENSION_HINTS:
        if pattern.search(column_name):
            return dim
    return None


# --- unit-aware column bounds (BRE-117) -------------------------------------- #
#
# Name matching alone missed `DO_pct` and cannot tell `antifoam_mgL` (mg/L) from
# a g/L column. `column_bounds` reads the UNIT the header states first
# (`kalos.normalize.header_units.parse_header_unit`: `titer_g_L`, `DO (%)`,
# `glutamine_mM`, `airflow_L_per_min`) and takes the bounds from the table below,
# expressed in that unit. Only when the header states no unit does it fall back
# to the name hints above. The hard floor of every non-negative quantity holds in
# any unit; the typical bands are given per unit where one is known.


@dataclass(frozen=True)
class ColumnBounds:
    """The bounds one column is judged against, in the unit the header states.

    `source` says, in plain words, where they came from (the header's unit or
    the column name). `note` is an optional sentence for the finding (why a
    value outside is not possible). `cell_dimension` is the `DIMENSION_BOUNDS`
    key used for a cell that carries its own unit token ("98.6 F"), which is
    converted to that dimension's base unit first.
    """

    dimension: str
    hard_lo: float
    hard_hi: float
    typical_lo: float
    typical_hi: float
    unit: str
    source: str
    note: str = ""
    cell_dimension: str | None = None


_NONNEG = (0.0, _INF, 0.0, _INF)

# unit (as `parse_header_unit` spells it, scale prefix removed) -> (dimension,
# hard_lo, hard_hi, typical_lo, typical_hi, cell_dimension). Typical bands follow
# the base-unit bands of DIMENSION_BOUNDS, converted.
_UNIT_BOUNDS: dict[str, tuple[str, float, float, float, float, str | None]] = {
    "°C": ("temperature", -20.0, 150.0, 15.0, 45.0, "temperature_c"),
    "K": ("temperature", 253.15, 423.15, 288.15, 318.15, "temperature_c"),
    "g/L": ("concentration", 0.0, _INF, 0.0, 500.0, "concentration"),
    "mg/mL": ("concentration", 0.0, _INF, 0.0, 500.0, "concentration"),
    "mg/L": ("concentration", 0.0, _INF, 0.0, 500_000.0, "concentration"),
    "µg/mL": ("concentration", 0.0, _INF, 0.0, 500_000.0, "concentration"),
    "mOsm": ("osmolality", 0.0, _INF, 200.0, 600.0, None),
    "mOsm/kg": ("osmolality", 0.0, _INF, 200.0, 600.0, None),
    "mOsm/L": ("osmolality", 0.0, _INF, 200.0, 600.0, None),
    "h": ("time", 0.0, _INF, 0.0, 720.0, "time_h"),
    "d": ("time", 0.0, _INF, 0.0, 30.0, "time_h"),
    "min": ("time", 0.0, _INF, 0.0, 43_200.0, "time_h"),
    "rpm": ("agitation", *_NONNEG, None),
    "vvm": ("gas flow per volume", *_NONNEG, None),
}
# Atoms that only ever measure a non-negative amount: a quantity made of them
# (a mass, a volume, an amount, a count, an activity, or a ratio of these such
# as g/L, mM, cells/mL, U/mL) cannot be negative. A ratio with a time in the
# denominator is a rate; only a volume per time (a flow) is non-negative by
# definition, while a net production or uptake rate (g/L/h) can be negative.
_AMOUNT_ATOMS = frozenset({
    "g", "mg", "µg", "ng", "kg", "L", "mL", "µL", "mol", "mmol", "µmol",
    "M", "mM", "µM", "nM", "mOsm", "cells", "U", "IU",
})
_VOLUME_ATOMS = frozenset({"L", "mL", "µL"})
_TIME_ATOMS = frozenset({"h", "min", "s", "d"})

# Percent columns. A share of a whole cannot leave 0-100 (DO as % of air
# saturation, viability, purity, aggregate or glycan fractions). A relative
# measure can (recovery, % of control, % change, yield vs theory, inhibition):
# no bound is asserted for those. Any other percent: 0-100 is the typical band
# (a warning outside), and only a negative value is impossible.
_PERCENT_OF_WHOLE = re.compile(
    r"(?<![a-z])do(?![a-z])|dissolved.?oxygen|o2.?sat|(?<![a-z])dot(?![a-z])|viab|purity|saturat"
    r"|aggregat|monomer|(?<![a-z])[hl]mw(?![a-z])|fragment|glycan|galactos|fucos|mannos|sialyl"
    r"|confluen|humidity|main.?peak|acidic|basic",
    re.I,
)
_PERCENT_RELATIVE = re.compile(
    r"change|delta|diff|increase|decrease|relative|(?<![a-z])rel(?![a-z])|(?<![a-z])vs(?![a-z])"
    r"|of.?control|recovery|yield|growth|inhibit|error|(?<![a-z])cv(?![a-z])|rsd|deviation|bias"
    r"|efficien",
    re.I,
)
_DO_NOTE = (
    "A probe calibrated to air reads above 100 % only with oxygen-enriched gas or overpressure; "
    "on a run sheet such a value is almost always a typo."
)


def _percent_bounds(name: str, unit: str, source: str) -> ColumnBounds | None:
    if _PERCENT_RELATIVE.search(name):
        return None
    if _PERCENT_OF_WHOLE.search(name):
        is_do = bool(re.search(r"(?<![a-z])do[t]?(?![a-z])|dissolved.?oxygen|o2.?sat", name, re.I))
        return ColumnBounds(
            "dissolved oxygen (% of air saturation)" if is_do else "a share of a whole",
            0.0, 100.0, 0.0, 100.0, unit, source,
            note=_DO_NOTE if is_do else "A share of a whole cannot be below 0 % or above 100 %.",
            cell_dimension="percent",
        )
    return ColumnBounds("percent", 0.0, _INF, 0.0, 100.0, unit, source, cell_dimension="percent")


def _unit_rule(name: str, unit: str) -> ColumnBounds | None:
    source = f"unit '{unit}' in the header"
    if unit == "%":
        return _percent_bounds(name, unit, source)
    atoms = unit_atoms(unit)
    key = "/".join(atoms)
    if key in _UNIT_BOUNDS:
        dim, hlo, hhi, tlo, thi, cell = _UNIT_BOUNDS[key]
        return ColumnBounds(dim, hlo, hhi, tlo, thi, unit, source, cell_dimension=cell)
    if not atoms:
        return None
    if all(a in _AMOUNT_ATOMS for a in atoms):
        kind = "amount" if len(atoms) == 1 else "concentration"
        return ColumnBounds(kind, *_NONNEG, unit, source)
    if len(atoms) == 2 and atoms[0] in _VOLUME_ATOMS and atoms[1] in _TIME_ATOMS:
        return ColumnBounds("flow rate", *_NONNEG, unit, source, cell_dimension="flow_rate")
    return None


def column_bounds(column_name: str) -> ColumnBounds | None:
    """The bounds a column is judged against, or None when neither its header's
    unit nor its name says what it measures (silence, not a guess).

    The header's unit wins (`DO_pct` is a percentage, `antifoam_mgL` is mg/L,
    `glutamine_mM` is a non-negative concentration); a header with no
    recognised unit falls back to the name hints (`temperature_C`, `pH`)."""
    name = str(column_name)
    unit = parse_header_unit(name)
    # pH is read from the name even when a unit-like token follows ("pH (units)").
    if unit is not None and infer_dimension(name) != "ph":
        rule = _unit_rule(name, unit)
        if rule is not None or unit == "%":
            return rule
    dim = infer_dimension(name)
    if dim is None:
        return None
    b = DIMENSION_BOUNDS[dim]
    label = _DIMENSION_WORDS.get(dim, dim.replace("_", " "))
    note = _DO_NOTE if dim == "percent" and re.search(r"(?<![a-z])do(?![a-z])|dissolved.?oxygen", name, re.I) else ""
    return ColumnBounds(
        label, b.hard_lo, b.hard_hi, b.typical_lo, b.typical_hi,
        _NAME_UNITS.get(dim, ""), f"read as {label} from the column name",
        note=note, cell_dimension=dim,
    )


# Plain words for the name-hint dimensions, and the unit the values are assumed
# to be in when only the name is known (pH and temperature conventions; for the
# others the bounds are a non-negative floor that holds in any unit).
_DIMENSION_WORDS = {
    "ph": "pH", "temperature_c": "temperature", "percent": "percent", "od": "optical density",
    "concentration": "concentration", "flow_rate": "flow rate", "time_h": "elapsed time",
    "cell_density": "cell density", "osmolality": "osmolality", "partial_pressure": "partial pressure",
    "nonneg_count": "count",
}
_NAME_UNITS = {"temperature_c": "°C", "percent": "%"}


def bounds_for_column(column_name: str) -> tuple[ColumnBounds, str] | None:
    """`column_bounds` with its source, for callers that print where the bounds
    came from (the result-entry checks)."""
    cb = column_bounds(column_name)
    return None if cb is None else (cb, cb.source)


__all__ = [
    "DIMENSION_BOUNDS", "ColumnBounds", "DimensionBounds", "bounds_for_column", "column_bounds",
    "infer_dimension",
]
