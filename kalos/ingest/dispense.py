"""Dispense volumes: find volume-to-add columns and convert them to concentrations.

A media sheet from a liquid handler often lists, per run, the VOLUME of each
stock to pipette ("Glycerol uL to add", "Trace mix_uL", "Volume (mL)", "added
as a liquid"), not the concentration each run ended up with. Using those
volumes as if they were concentrations is wrong as soon as two stocks differ in
strength, the final volume changes between runs, or a master mix delivers
several components at once. Two pieces handle this:

- `detect_dispense_columns` reads only the headers and returns one error
  finding (`volumes_not_concentrations`) per column that is a volume to add.
  Concentration headers (g/L, mg/mL, mM, % w/v, mL/L, ng/uL ...), rates
  (mL/h) and vessel volumes (working, culture, final, fill volume) are not
  flagged.
- `convert` turns those columns into final concentrations by C1 x V1 = C2 x V2,
  using only what the user declares: the stock concentration of each column
  (with its unit), the composition of each master mix (component ->
  concentration in the mix), and the final volume (a constant or a per-row
  column, with its unit). For every component

      c_final = sum over the sources that deliver it of c_in_source x V_source / V_final

  where a source is one column. A master-mix column is expanded into its
  components and is never also counted as an ingredient of its own; a
  component added directly and also present in a mix gets one term per
  column, never two terms for the same column.

Nothing is guessed: no stock, unit, final volume or molecular weight has a
default. A missing declaration is a refusal with the action that fixes it.
Volume units convert only among uL/mL/L; concentration units only within one
dimension (mass/volume, molar, volume/volume) through the table below, and
mass <-> molar only with a declared molecular weight. Rows whose volumes are
blank, not numbers or negative, whose final volume is missing, or whose added
volumes sum to more than the final volume get no value; they stay in the
table with a `dispense_status` saying why. The raw volume columns are kept
unchanged for the record and get no role suggestion; the converted
concentration columns, named "<component> (<unit>)", are suggested as
`setting` (they are the recipe inputs). Every converted column has its formula
in `meta["formulas"]` and `meta["conversion"]`, and every row has the
arithmetic with its own numbers in `dispense_provenance`.

Idea credit: the lab's finding that a media file imported dispense volumes as
if they were concentrations, and a component counted twice through a master
mix, from Bioqore's ad hoc analyses; see NOTICE. Re-implemented; no code copied.
"""
from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import pandas as pd

from kalos.ingest.findings import Finding, finding

PARSER_VERSION = "dispense/1"
_LIST_CAP = 50

STATUS_COLUMN = "dispense_status"
PROVENANCE_COLUMN = "dispense_provenance"
CONVERTED = "converted"

# Volume units -> litres. Only these convert into each other.
VOLUME_UNITS: dict[str, float] = {"uL": 1e-6, "mL": 1e-3, "L": 1.0}
_VOLUME_TOKENS: dict[str, str] = {
    "ul": "uL", "microliter": "uL", "microliters": "uL", "microlitre": "uL", "microlitres": "uL",
    "ml": "mL", "milliliter": "mL", "milliliters": "mL", "millilitre": "mL", "millilitres": "mL",
    "l": "L", "liter": "L", "liters": "L", "litre": "L", "litres": "L",
}

# Concentration units -> (dimension, factor to the dimension's base unit).
# Bases: g/L (mass/volume), mol/L (molar), mL/L (volume/volume).
CONCENTRATION_UNITS: dict[str, tuple[str, float]] = {
    "g/L": ("mass/volume", 1.0),
    "mg/mL": ("mass/volume", 1.0),
    "g/mL": ("mass/volume", 1e3),
    "mg/L": ("mass/volume", 1e-3),
    "ug/mL": ("mass/volume", 1e-3),
    "ng/uL": ("mass/volume", 1e-3),
    "ug/L": ("mass/volume", 1e-6),
    "ng/mL": ("mass/volume", 1e-6),
    "% w/v": ("mass/volume", 10.0),  # g per 100 mL
    "M": ("molar", 1.0),
    "mM": ("molar", 1e-3),
    "uM": ("molar", 1e-6),
    "nM": ("molar", 1e-9),
    "% v/v": ("volume/volume", 10.0),  # mL per 100 mL
    "mL/L": ("volume/volume", 1.0),
    "uL/mL": ("volume/volume", 1.0),
    "uL/L": ("volume/volume", 1e-3),
}
_MOLAR_ALIASES = {"mol/L": "M", "mmol/L": "mM", "umol/L": "uM", "nmol/L": "nM"}

# Header words that make a bare volume a vessel or process volume, not an addition.
_VESSEL_WORDS = frozenset({
    "working", "culture", "fill", "final", "total", "vessel", "reactor", "bioreactor", "flask",
    "plate", "well", "tube", "scale", "size", "capacity", "nominal", "max", "maximum", "min",
    "minimum", "headspace", "dead", "harvest", "sample", "sampled", "sampling", "removed",
    "removal", "bleed", "evaporation", "loss", "initial", "start", "starting", "broth",
    "inoculum", "inoculation", "titrant", "titration", "consumed", "consumption", "cumulative",
})
_RATIO_WORDS = frozenset({"/", "per"})
_AMOUNT_TOKENS = frozenset({
    "g", "mg", "ug", "ng", "pg", "kg", "mol", "mmol", "umol", "nmol", "iu", "u", "units", "cells",
})
_TIME_TOKENS = frozenset({"h", "hr", "hrs", "hour", "hours", "min", "mins", "minute", "s", "sec",
                          "d", "day", "days"})
_PHRASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("added_as_liquid", ("added", "as", "a", "liquid")),
    ("added_as_liquid", ("added", "as", "liquid")),
    ("to_add", ("to", "add")),
    ("dispense", ("dispense",)),
    ("dispense", ("dispensed",)),
    ("volume_word", ("volume",)),
    ("volume_word", ("vol",)),
)
_NUMBER_RE = re.compile(r"^\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*(.*?)\s*$")


def _ascii_micro(text: str) -> str:
    return text.replace("\u00b5", "u").replace("\u03bc", "u")


def _tokens(header: object) -> list[str]:
    """Lower-case word tokens of a header, keeping '/' and '%' as tokens."""
    return re.findall(r"[a-z0-9]+|/|%", _ascii_micro(str(header)).lower())


def name_key(name: object) -> str:
    """The case- and punctuation-insensitive key used to match names ("Citric.acid" == "citric acid")."""
    return " ".join(t for t in _tokens(name) if t not in ("/", "%"))


def _is_ratio_part(tokens: Sequence[str], i: int) -> bool:
    """True if the volume token at i is the numerator or denominator of a ratio (mg/mL, mL/h, mL.L)."""
    prev = tokens[i - 1] if i > 0 else ""
    nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
    if prev in _RATIO_WORDS or prev in _AMOUNT_TOKENS or prev in _VOLUME_TOKENS:
        return True
    return nxt in _RATIO_WORDS or nxt in _VOLUME_TOKENS or nxt in _TIME_TOKENS


def _volume_unit_positions(header: object) -> list[int]:
    """Indices in `_tokens(header)` of volume units that stand alone (not part of a ratio).

    A bare "L" counts only in brackets ("Feed (L)"): "Clone L" or "5L" are labels.
    """
    toks = _tokens(header)
    bracketed_l = re.search(r"[(\[]\s*l\s*[)\]]", str(header).lower()) is not None
    return [i for i, t in enumerate(toks) if t in _VOLUME_TOKENS and (t != "l" or bracketed_l)
            and not _is_ratio_part(toks, i)]


# ----------------------------------------------------------------------------
# Detection
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class DispenseColumn:
    """A column whose header says it is a volume to add."""

    column: str
    component: str | None  # name_key of what it dispenses; None if the header names nothing
    unit: str | None  # "uL", "mL" or "L"; None if the header has no volume unit
    pattern: str  # to_add, added_as_liquid, dispense, volume_word or unit_suffix


def parse_dispense_header(header: object) -> DispenseColumn | None:
    """A `DispenseColumn` if the header says it is a volume to add, else None.

    Flagged: a header with a volume unit that is not part of a ratio ("uL to
    add", "_uL", "Volume (mL)", "(mL)"), unless it names a vessel or process
    volume (working, culture, final, fill ... volume); and any header saying
    "added as (a) liquid". With "to add" or "dispense(d)" the vessel words do
    not apply. A header with no volume unit is flagged only for "added as a
    liquid" or for "volume" next to a component name ("Glycerol volume").
    """
    toks = _tokens(header)
    if not toks:
        return None
    drop: set[int] = set()
    kinds: list[str] = []
    for kind, phrase in _PHRASES:
        n = len(phrase)
        for i in range(len(toks) - n + 1):
            if tuple(toks[i:i + n]) == phrase and not drop.intersection(range(i, i + n)):
                drop.update(range(i, i + n))
                kinds.append(kind)
    units = _volume_unit_positions(header)
    unit_names = {_VOLUME_TOKENS[toks[i]] for i in units}
    strong = bool({"to_add", "dispense", "added_as_liquid"}.intersection(kinds))
    vessel = bool(_VESSEL_WORDS.intersection(toks))
    rest = [t for i, t in enumerate(toks) if i not in drop and i not in units and t not in ("/", "%")]
    component = " ".join(rest) or None
    if units:
        if vessel and not strong:
            return None
        pattern = next((k for k in ("to_add", "added_as_liquid", "dispense", "volume_word")
                        if k in kinds), "unit_suffix")
    elif "added_as_liquid" in kinds:
        pattern = "added_as_liquid"
    elif "volume_word" in kinds and component and not vessel:
        pattern = "volume_word"
    else:
        return None
    unit = next(iter(unit_names)) if len(unit_names) == 1 else None
    return DispenseColumn(str(header), component, unit, pattern)


def dispense_columns(frame: pd.DataFrame) -> list[DispenseColumn]:
    """Every column of `frame` whose header says it is a volume to add, in column order."""
    out = []
    for col in frame.columns:
        parsed = parse_dispense_header(col)
        if parsed is not None:
            out.append(parsed)
    return out


def detect_dispense_columns(frame: pd.DataFrame) -> tuple[Finding, ...]:
    """One `volumes_not_concentrations` error per dispense-volume column (headers only)."""
    found = []
    for dc in dispense_columns(frame):
        unit = f" in {dc.unit}" if dc.unit else ""
        found.append(finding(
            "volumes_not_concentrations", "error",
            f"Column '{dc.column}' is a volume to add{unit}, not the concentration each run ended "
            "up with. It is kept, but it is refused as a concentration input: the same volume gives "
            "a different concentration for a different stock or final volume, and a master mix "
            "delivers several components. Supply the stock concentration (with unit) for this "
            "column and the final volume so it can be converted, or upload the concentrations.",
            column=dc.column, component=dc.component, volume_unit=dc.unit, pattern=dc.pattern,
        ))
    return tuple(found)


# ----------------------------------------------------------------------------
# Declarations
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Quantity:
    """A number with the unit the user declared (e.g. a stock of 500 g/L, a final volume of 3 mL)."""

    value: float
    unit: str


@dataclass(frozen=True)
class FinalVolumeColumn:
    """The final volume of each run is in `column`, in `unit` (uL, mL or L)."""

    column: str
    unit: str


QuantitySpec = Quantity | str | tuple[float, str] | Mapping[str, object]


def parse_quantity(spec: object) -> Quantity | None:
    """Read "500 g/L", (500, "g/L"), {"value": 500, "unit": "g/L"} or a Quantity; None if not one."""
    if isinstance(spec, Quantity):
        value, unit = spec.value, spec.unit
    elif isinstance(spec, str):
        m = _NUMBER_RE.match(spec)
        if not m or not m.group(2):
            return None
        value, unit = float(m.group(1)), m.group(2)
    elif isinstance(spec, Mapping) and "value" in spec and "unit" in spec:
        value, unit = spec["value"], spec["unit"]  # type: ignore[assignment]
    elif isinstance(spec, tuple | list) and len(spec) == 2:
        value, unit = spec[0], spec[1]
    else:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float) or not isinstance(unit, str):
        return None
    if not unit.strip():
        return None
    return Quantity(float(value), unit.strip())


def volume_unit(unit: str) -> str | None:
    """Canonical volume unit ("uL", "mL", "L") or None."""
    return _VOLUME_TOKENS.get(_ascii_micro(unit).strip().lower())


def concentration_unit(unit: str) -> str | None:
    """Canonical concentration unit from `CONCENTRATION_UNITS`, or None if it is not in the table."""
    compact = re.sub(r"[\s()]", "", _ascii_micro(unit))
    for canon in (*CONCENTRATION_UNITS, *_MOLAR_ALIASES):
        if compact == canon.replace(" ", ""):
            return _MOLAR_ALIASES.get(canon, canon)
    lower = compact.lower()
    for canon, (dim, _f) in CONCENTRATION_UNITS.items():
        if dim != "molar" and lower == canon.replace(" ", "").lower():
            return canon
    for alias, canon in _MOLAR_ALIASES.items():
        if lower == alias.lower():
            return canon
    return None


def _fmt(x: float) -> str:
    return f"{x:.6g}"


def _round(x: float) -> float:
    """Drop float noise from a unit conversion (3.1349999999999993 -> 3.135) for display."""
    return float(f"{float(x):.10g}")


def _cell_text(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)) or str(value).strip() == "":
        return "(blank)"
    return str(value)


def _plain(q: Quantity) -> dict:
    return {"value": q.value, "unit": q.unit}


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Conversion
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class DispenseResult:
    """The sheet with converted concentration columns added, and what was found."""

    table: pd.DataFrame
    findings: tuple[Finding, ...]
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        records = self.table.astype(object).where(self.table.notna(), None).to_dict(orient="records")
        return {
            "table": records,
            "findings": [f.to_dict() for f in self.findings],
            "meta": dict(self.meta),
        }


@dataclass(frozen=True)
class _FinalVolume:
    unit: str  # canonical volume unit the user declared
    column: str | None = None  # per-row column, or
    litres: float | None = None  # one constant


@dataclass(frozen=True)
class _Source:
    column: str
    kind: str  # "stock" or "mix"
    via: str  # the declared stock or mix name
    concentration: Quantity  # value NaN if the declaration was not usable
    volume_unit: str


@dataclass(frozen=True)
class _Term:
    source: _Source
    unit: str  # canonical concentration unit of the source
    factor: float  # source unit -> output unit
    cross: bool  # mass <-> molar through a molecular weight


@dataclass(frozen=True)
class _Plan:
    component: str
    unit: str
    terms: tuple[_Term, ...]
    mw: float | None


def _usable(q: Quantity | None) -> bool:
    return q is not None and math.isfinite(q.value) and q.value > 0


def _refuse(frame: pd.DataFrame, found: list[Finding], meta: dict) -> DispenseResult:
    meta = {**meta, "converted": False, "role_suggestions": {}}
    return DispenseResult(frame.copy(), tuple(found), meta)


def _number_column(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """(numbers, bad) where bad marks blank, non-numeric and non-finite cells."""
    cleaned = series.map(lambda v: v.strip() if isinstance(v, str) else v)
    nums = pd.to_numeric(cleaned.map(lambda v: None if isinstance(v, bool) else v), errors="coerce")
    nums = nums.astype(float)
    bad = ~nums.map(math.isfinite).astype(bool)
    return nums, bad


def _final_volume(spec: object, columns: pd.Index) -> tuple[_FinalVolume | None, Finding | None]:
    if spec is None:
        return None, finding(
            "final_volume_missing", "error",
            "No final volume was declared, so no concentration can be computed. Declare the final "
            "volume as a constant with its unit (e.g. '3 mL') or as a per-row column with its unit."
            " Nothing was converted and nothing was changed.",
        )
    if isinstance(spec, FinalVolumeColumn):
        unit = volume_unit(spec.unit)
        if spec.column not in columns:
            return None, finding(
                "final_volume_missing", "error",
                f"The declared final-volume column '{spec.column}' is not in the sheet. Name an "
                "existing column or declare a constant final volume. Nothing was converted and "
                "nothing was changed.",
                column=spec.column,
            )
        if unit is None:
            return None, finding(
                "final_volume_unit_unknown", "error",
                f"The final-volume unit '{spec.unit}' is not a volume unit (uL, mL, L). Declare the "
                "column's unit. Nothing was converted and nothing was changed.", column=spec.column,
            )
        toks = _tokens(spec.column)
        in_header = sorted({_VOLUME_TOKENS[toks[i]] for i in _volume_unit_positions(spec.column)})
        if in_header and in_header != [unit]:
            return None, finding(
                "final_volume_unit_conflict", "error",
                f"The final-volume column '{spec.column}' says {in_header} in its header but was "
                f"declared in {unit}. Check which is right. Nothing was converted and nothing was "
                "changed.",
                column=spec.column, declared=unit, header=in_header,
            )
        return _FinalVolume(unit, column=spec.column), None
    q = parse_quantity(spec)
    unit = volume_unit(q.unit) if q else None
    if q is None or unit is None or not _usable(q):
        return None, finding(
            "final_volume_invalid", "error",
            f"The final volume {spec!r} is not a positive volume with a unit (uL, mL, L). Declare "
            "it like '3 mL'. Nothing was converted and nothing was changed.",
        )
    return _FinalVolume(unit, litres=q.value * VOLUME_UNITS[unit]), None


def _declarations(
    stocks: Mapping[str, object], mixes: Mapping[str, Mapping[str, object]],
    mws: Mapping[str, object],
) -> tuple[dict[str, tuple[str, str]], dict[str, float], list[Finding]]:
    """name_key -> (kind, declared name); name_key -> MW; and what is wrong with them."""
    found: list[Finding] = []
    decl: dict[str, tuple[str, str]] = {}
    for kind, names in (("stock", stocks), ("mix", mixes)):
        for name in names:
            key = name_key(name)
            if key in decl:
                found.append(finding(
                    "declaration_duplicated", "error",
                    f"'{name}' is declared more than once (as {decl[key][0]} '{decl[key][1]}' and as "
                    f"{kind} '{name}'; names are compared ignoring case and punctuation). Declare "
                    "each stock or mix once. Nothing was converted and nothing was changed.",
                    name=str(name),
                ))
            else:
                decl[key] = (kind, str(name))
    for mix, composition in mixes.items():
        seen: set[str] = set()
        for c in composition:
            key = name_key(c)
            if key in seen:
                found.append(finding(
                    "mix_component_duplicated", "error",
                    f"Master mix '{mix}' lists '{c}' more than once (names compared ignoring case and "
                    "punctuation). List each component of a mix once with its concentration in the "
                    "mix. Nothing was converted and nothing was changed.", mix=str(mix),
                    component=str(c),
                ))
            seen.add(key)
            if decl.get(key, ("", ""))[0] == "mix":
                found.append(finding(
                    "mix_nested", "error",
                    f"Master mix '{mix}' lists '{c}', which is itself a declared mix. Mixes are not "
                    "expanded inside mixes. Declare the mix's final composition. Nothing was "
                    "converted and nothing was changed.",
                    mix=str(mix), component=str(c),
                ))
        if not composition:
            found.append(finding(
                "mix_declared_empty", "info",
                f"Master mix '{mix}' was declared with no components: it is counted as delivering "
                "none of the components (a diluent). If it contains any of them, declare them.",
                mix=str(mix),
            ))
    mw_by_key: dict[str, float] = {}
    bad_mw: list[str] = []
    for name, mw in mws.items():
        if isinstance(mw, bool) or not isinstance(mw, int | float) or not math.isfinite(mw) or mw <= 0:
            bad_mw.append(str(name))
        else:
            mw_by_key[name_key(name)] = float(mw)
    if bad_mw:
        found.append(finding(
            "molecular_weight_invalid", "error",
            f"Molecular weights must be positive numbers in g/mol; {sorted(bad_mw)} are not. Fix "
            "them. Nothing was converted and nothing was changed.", components=sorted(bad_mw),
        ))
    return decl, mw_by_key, found


def _map_sources(
    detected: Sequence[DispenseColumn], decl: Mapping[str, tuple[str, str]],
) -> tuple[dict[str, tuple[str, str]], list[Finding]]:
    """Volume column -> (kind, declared name), refusing anything ambiguous or undeclared."""
    found: list[Finding] = []
    sources: dict[str, tuple[str, str]] = {}
    used: dict[str, str] = {}  # declaration key -> column
    for dc in detected:
        if dc.component is None:
            found.append(finding(
                "dispense_component_unknown", "error",
                f"Column '{dc.column}' is a volume to add but its header does not say what is "
                "added; an unknown addition may carry any component. Rename it '<stock or mix> "
                "(<unit>)', e.g. 'Glycerol (uL)'. Nothing was converted and nothing was changed.",
                column=dc.column,
            ))
            continue
        hit = decl.get(dc.component)
        if hit is None:
            found.append(finding(
                "dispense_column_no_stock", "error",
                f"Column '{dc.column}' is a volume to add of '{dc.component}', but no stock "
                "concentration or mix composition was declared for it. An undeclared addition may "
                "carry components that other columns also deliver, so every other value could be "
                "low. Declare its stock (e.g. '500 g/L'), its mix composition, or, for water or a "
                "diluent, an empty mix. Nothing was converted and nothing was changed.",
                column=dc.column, component=dc.component,
            ))
            continue
        if dc.component in used:
            found.append(finding(
                "dispense_source_duplicated", "error",
                f"Columns '{used[dc.component]}' and '{dc.column}' both dispense '{hit[1]}', so the "
                "stock would be counted twice. If one is a copy, remove it; if they are two real "
                "additions, add them into one column. Nothing was converted and nothing was changed.",
                column=dc.column, columns=[used[dc.component], dc.column],
            ))
            continue
        if dc.unit is None:
            found.append(finding(
                "volume_unit_missing", "error",
                f"Column '{dc.column}' is a volume to add but its header has no single volume unit "
                "(uL, mL, L). Put the unit in the header, e.g. 'Glycerol (uL)'. Nothing was "
                "converted and nothing was changed.", column=dc.column,
            ))
            continue
        used[dc.component] = dc.column
        sources[dc.column] = hit
    unused = [name for key, (_kind, name) in decl.items() if key not in used]
    if unused:
        found.append(finding(
            "declaration_not_used", "warning",
            f"Declared stock(s) or mix(es) {unused[:_LIST_CAP]} match no volume column. They were not "
            "used; check the spelling against the column headers.", names=unused[:_LIST_CAP],
        ))
    return sources, found


def _components(
    sources: Mapping[str, tuple[str, str]], units: Mapping[str, str],
    stocks: Mapping[str, object], mixes: Mapping[str, Mapping[str, object]],
) -> tuple[dict[str, tuple[str, list[_Source]]], list[Finding]]:
    """name_key -> (display name, one source per column that delivers it), in column order.

    A stock column delivers its own component; a mix column delivers each
    component of its composition and never the mix itself.
    """
    found: list[Finding] = []
    out: dict[str, tuple[str, list[_Source]]] = {}
    for col, (kind, name) in sources.items():
        if kind == "stock":
            delivered: list[tuple[str, object]] = [(name, stocks[name])]
        else:
            delivered = [(str(c), spec) for c, spec in mixes[name].items()]
        for component, spec in delivered:
            q = parse_quantity(spec)
            if q is None or not _usable(q):
                where = f"the stock for '{name}'" if kind == "stock" else (
                    f"the concentration of '{component}' in mix '{name}'")
                found.append(finding(
                    "stock_invalid", "error",
                    f"{where[0].upper()}{where[1:]} ({spec!r}) is not a positive concentration with a "
                    f"unit, so '{component}' gets no value; the other components are still "
                    "converted and the column's volume still counts toward the total. Declare it "
                    "like '500 g/L'.",
                    column=col, stock=name, component=component,
                ))
                q = Quantity(math.nan, "")
            out.setdefault(name_key(component), (component, []))[1].append(
                _Source(col, kind, name, q, units[col]))
    return out, found


def _plan(
    key: str, name: str, sources: Sequence[_Source], mw_by_key: Mapping[str, float],
    columns: pd.Index,
) -> tuple[_Plan | None, Finding | None, str | None]:
    """The output unit and the factor of each source, or the reason the component is refused."""
    if any(math.isnan(s.concentration.value) for s in sources):
        return None, None, "stock_invalid"  # already reported by _components
    canon = [concentration_unit(s.concentration.unit) for s in sources]
    unknown = sorted({s.concentration.unit for s, c in zip(sources, canon, strict=True) if c is None})
    if unknown:
        return None, finding(
            "concentration_unit_unknown", "error",
            f"'{name}' has a concentration unit the converter does not know ({unknown}); it gets "
            f"no value and the other components are still converted. Known units: {', '.join(CONCENTRATION_UNITS)}. Declare it in one of them (a "
            "fold stock such as 1000x can be declared as a mix with its composition).",
            component=name, units=unknown,
        ), "concentration_unit_unknown"
    out_unit = canon[0] or ""
    out_dim, out_f = CONCENTRATION_UNITS[out_unit]
    mw = mw_by_key.get(key)
    terms: list[_Term] = []
    incompatible: list[str] = []
    molar_mass = {"mass/volume", "molar"}
    need_mw = False
    for src, cu in zip(sources, canon, strict=True):
        dim, f = CONCENTRATION_UNITS[cu or ""]
        if dim == out_dim:
            terms.append(_Term(src, cu or "", f / out_f, False))
        elif {dim, out_dim} == molar_mass and mw is not None:
            factor = (f / mw if dim == "mass/volume" else f * mw) / out_f
            terms.append(_Term(src, cu or "", factor, True))
        else:
            incompatible.append(f"{src.concentration.unit} ({dim})")
            need_mw = need_mw or {dim, out_dim} == molar_mass
    if incompatible:
        need = " Declare its molecular weight (g/mol) to add mass and molar sources." if need_mw else ""
        return None, finding(
            "concentration_units_incompatible", "error",
            f"'{name}' is delivered in {out_unit} ({out_dim}) and in {incompatible}; these cannot be "
            f"added, so '{name}' gets no value; the other components are still converted.{need}",
            component=name, units=[out_unit, *incompatible],
        ), "concentration_units_incompatible"
    out_col = f"{name} ({out_unit})"
    if out_col in columns:
        return None, finding(
            "output_column_exists", "error",
            f"The sheet already has a column '{out_col}'; it was not overwritten and '{name}' was "
            "not converted. Rename one of them, or drop the volume column if the sheet already has "
            "the concentration.", column=out_col, component=name,
        ), "output_column_exists"
    return _Plan(name, out_unit, tuple(terms), mw), None, None


def _row_status(
    frame: pd.DataFrame, sources: Mapping[str, tuple[str, str]], units: Mapping[str, str],
    fv: _FinalVolume,
) -> tuple[dict[str, pd.Series], pd.Series, pd.Series, list[Finding]]:
    """Volumes in litres per column, the final volume in litres per row, the row status, findings."""
    found: list[Finding] = []
    idx = frame.index
    volumes_l: dict[str, pd.Series] = {}
    total_l = pd.Series(0.0, index=idx)
    bad_rows = pd.Series(False, index=idx)
    neg_rows = pd.Series(False, index=idx)
    for col in sources:
        nums, bad = _number_column(frame[col])
        neg = ~bad & (nums < 0)
        if bad.any():
            found.append(finding(
                "volume_not_numeric", "error",
                f"{int(bad.sum())} row(s) of '{col}' are blank or not a number; those rows are kept "
                "with no converted value (a blank is not read as 0). Enter the volume, or 0 if "
                "nothing was added.", column=col, rows=idx[bad],
                raw_values=sorted({_cell_text(v) for v in frame.loc[bad, col]})[:10],
            ))
        if neg.any():
            found.append(finding(
                "volume_negative", "error",
                f"{int(neg.sum())} row(s) of '{col}' have a negative volume; those rows are kept with "
                "no converted value. Fix the volumes.", column=col, rows=idx[neg],
            ))
        bad_rows |= bad
        neg_rows |= neg
        volumes_l[col] = nums * VOLUME_UNITS[units[col]]
        total_l = total_l + volumes_l[col].where(~bad, 0.0)
    if fv.column is not None:
        fv_nums, fv_bad = _number_column(frame[fv.column])
        fv_bad = fv_bad | (fv_nums <= 0)
        if fv_bad.any():
            found.append(finding(
                "final_volume_invalid", "error",
                f"{int(fv_bad.sum())} row(s) of the final-volume column '{fv.column}' are blank, not "
                "a number or not positive; those rows are kept with no converted value. Enter the "
                "final volume.", column=fv.column, rows=idx[fv_bad],
            ))
        final_l = fv_nums * VOLUME_UNITS[fv.unit]
    else:
        fv_bad = pd.Series(False, index=idx)
        final_l = pd.Series(float(fv.litres or 0.0), index=idx)
    checkable = ~bad_rows & ~neg_rows & ~fv_bad
    over = checkable & (total_l > final_l * (1 + 1e-9))
    if over.any():
        scale = VOLUME_UNITS[fv.unit]
        found.append(finding(
            "added_volume_exceeds_final", "error",
            f"In {int(over.sum())} row(s) the volumes added sum to more than the final volume, so "
            "the recipe cannot be made as written; those rows are kept with no converted value. "
            "Check the volumes (a unit or x1000 slip) and the final volume.",
            rows=idx[over],
            where=[{"row": int(i), f"total_added_{fv.unit}": _round(total_l[i] / scale),
                    f"final_{fv.unit}": _round(final_l[i] / scale)} for i in idx[over][:_LIST_CAP]],
        ))
    short = checkable & ~over & (total_l < final_l * (1 - 1e-9))
    if short.any():
        found.append(finding(
            "volume_remainder_not_declared", "info",
            f"In {int(short.sum())} row(s) the volumes added are less than the final volume. The "
            "remainder is taken to contain none of the converted components; if it is a medium "
            "that contains any of them, add it as a volume column with its composition.",
            rows=idx[short],
        ))
    status = pd.Series(CONVERTED, index=idx, dtype=object)
    status[fv_bad] = "refused_final_volume_invalid"
    status[neg_rows] = "refused_volume_negative"
    status[bad_rows] = "refused_volume_not_numeric"
    status[over] = "refused_added_volume_exceeds_final"
    return volumes_l, final_l, status, found


def _formula(plan: _Plan) -> str:
    terms = []
    for t in plan.terms:
        c = t.source.concentration
        shown = f"{_fmt(c.value)} {t.unit}"
        if t.factor != 1.0:
            mw = f", MW {_fmt(plan.mw)} g/mol" if t.cross and plan.mw else ""
            shown += f" (= {_fmt(c.value * t.factor)} {plan.unit}{mw})"
        where = f" in {t.source.via}" if t.source.kind == "mix" else ""
        terms.append(f"{shown}{where} x V({t.source.column}) / V_final")
    return "c = " + " + ".join(terms)


def convert(
    frame: pd.DataFrame,
    stocks: Mapping[str, object] | None,
    final_volume: object,
    master_mixes: Mapping[str, Mapping[str, object]] | None = None,
    *,
    molecular_weights: Mapping[str, float] | None = None,
) -> DispenseResult:
    """Convert dispense-volume columns to final concentrations by C1 x V1 = C2 x V2.

    `stocks`: stock name -> concentration with unit ("500 g/L", (500, "g/L") or
    `Quantity`) for every column that dispenses one stock. The name is matched
    to the component the header names, ignoring case and punctuation
    ("Citric.acid..uL.to.add" -> "Citric acid").
    `master_mixes`: mix name -> {component -> concentration in the mix} for every
    column that dispenses a mix. A diluent (water, or a base whose contents do
    not matter here) is declared as a mix with no components: {"Water": {}}.
    `final_volume`: a constant ("3 mL", (3, "mL")) or a `FinalVolumeColumn`.
    `molecular_weights`: component -> g/mol, needed only to add a mass and a
    molar source of the same component.

    Output columns are "<component> (<unit>)", the unit being that of the
    component's first source (in column order); other sources are converted
    to it with the factor shown in the formula. Column labels are read as text.
    """
    frame = frame.reset_index(drop=True)
    frame.columns = pd.Index([str(c) for c in frame.columns])
    stocks = dict(stocks or {})
    mixes = {m: dict(c) for m, c in (master_mixes or {}).items()}
    mws = dict(molecular_weights or {})

    def declared(spec: object) -> object:
        q = parse_quantity(spec)
        return _plain(q) if q else (None if spec is None else repr(spec))

    meta: dict = {
        "parser": PARSER_VERSION,
        "stocks": {str(k): declared(v) for k, v in stocks.items()},
        "master_mixes": {str(m): {str(k): declared(v) for k, v in c.items()} for m, c in mixes.items()},
        "final_volume": ({"column": final_volume.column, "unit": final_volume.unit}
                         if isinstance(final_volume, FinalVolumeColumn) else declared(final_volume)),
        "molecular_weights": {str(k): v for k, v in mws.items()},
        "volume_factors_to_L": dict(VOLUME_UNITS),
    }
    detected = dispense_columns(frame)
    meta["volume_columns"] = [
        {"column": d.column, "component": d.component, "unit": d.unit, "pattern": d.pattern}
        for d in detected
    ]
    if not detected:
        return _refuse(frame, [finding(
            "no_dispense_columns", "info",
            "No column header says it is a volume to add. Nothing was converted and nothing was "
            "changed.",
        )], meta)

    found: list[Finding] = []
    fv, fv_problem = _final_volume(final_volume, frame.columns)
    if fv_problem is not None:
        found.append(fv_problem)
    fv_col = fv.column if fv else None
    detected = [d for d in detected if d.column != fv_col]  # the final volume is never a source
    decl, mw_by_key, problems = _declarations(stocks, mixes, mws)
    found.extend(problems)
    sources, problems = _map_sources(detected, decl)
    found.extend(problems)
    if fv is None or any(f.severity == "error" for f in found):
        return _refuse(frame, [*detect_dispense_columns(frame[[d.column for d in detected]]), *found],
                       meta)

    units = {d.column: d.unit or "" for d in detected}
    components, problems = _components(sources, units, stocks, mixes)
    found.extend(problems)
    plans: dict[str, _Plan] = {}
    refused: dict[str, str] = {}
    for key, (name, srcs) in components.items():
        plan, problem, reason = _plan(key, name, srcs, mw_by_key, frame.columns)
        if problem is not None:
            found.append(problem)
        if plan is None:
            refused[name] = reason or ""
        else:
            plans[f"{name} ({plan.unit})"] = plan
    volumes_l, final_l, status, problems = _row_status(frame, sources, units, fv)
    found.extend(problems)
    ok = status == CONVERTED

    out = frame.copy()
    provenance: list[dict[str, str]] = [{} for _ in range(len(frame))]
    fv_scale = VOLUME_UNITS[fv.unit]
    fv_text = f"{fv.column} ({fv.unit})" if fv.column else f"{_fmt((fv.litres or 0) / fv_scale)} {fv.unit}"
    formulas: dict[str, str] = {}
    conversion: dict[str, dict] = {}
    for out_col, plan in plans.items():
        value = pd.Series(0.0, index=frame.index)
        for t in plan.terms:
            conc = t.source.concentration.value * t.factor
            value = value + conc * volumes_l[t.source.column] / final_l
        out[out_col] = value.where(ok)
        formulas[out_col] = _formula(plan)
        conversion[out_col] = {
            "component": plan.component, "unit": plan.unit, "formula": formulas[out_col],
            "final_volume": fv_text, "molecular_weight_g_per_mol": plan.mw,
            "note": "V and V_final are converted to litres before dividing.",
            "sources": [{
                "column": t.source.column, "kind": t.source.kind, "declared_as": t.source.via,
                "concentration": t.source.concentration.value, "concentration_unit": t.unit,
                "factor_to_output_unit": t.factor, "volume_unit": t.source.volume_unit,
                "volume_factor_to_L": VOLUME_UNITS[t.source.volume_unit],
            } for t in plan.terms],
        }
        for i in frame.index[ok]:
            parts = []
            for t in plan.terms:
                src = t.source
                in_mix = f" in {src.via}" if src.kind == "mix" else ""
                v_raw = volumes_l[src.column][i] / VOLUME_UNITS[src.volume_unit]
                parts.append(f"{_fmt(src.concentration.value * t.factor)} {plan.unit}{in_mix} x "
                             f"{_fmt(v_raw)} {src.volume_unit} / {_fmt(final_l[i] / fv_scale)} {fv.unit}")
            provenance[i][out_col] = " + ".join(parts) + f" = {_fmt(float(value[i]))} {plan.unit}"
    out[STATUS_COLUMN] = status.to_numpy()
    out[PROVENANCE_COLUMN] = pd.Series(provenance, index=frame.index, dtype=object)
    found.append(finding(
        "volume_columns_kept", "info",
        f"The {len(sources)} volume column(s) are kept unchanged for the record. Use the converted "
        "concentration columns as recipe inputs, not the volumes.",
        columns=list(sources)[:_LIST_CAP],
    ))
    meta.update({
        "converted": True,
        "formulas": formulas,
        "conversion": conversion,
        "refused_components": refused,
        "role_suggestions": dict.fromkeys(plans, "setting"),
    })
    return DispenseResult(out, tuple(found), meta)
