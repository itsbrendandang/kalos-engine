"""Plates: well labels, plate-reader grids and two-level plate maps.

Three deterministic pieces, each returning a long table plus findings
(`kalos.ingest.findings`), never dropping a row silently:

- `normalize_well` turns "A01", "a1" or " A 1" into "A1" and checks it against
  the plate format (24 = 4 x 6, 96 = 8 x 12, 384 = 16 x 24).
- `parse_grid_text` / `read_grid` read a plate laid out as a grid (a header of
  column numbers 1..n, then one line per row letter), the layout plate readers
  export. Every well becomes one row of the long table. Empty cells and cells
  that are not numbers (an overflow marker, a typo) stay in the table with
  `value` NaN, the raw text and a status, and are reported by well.
- `parse_plate_map` / `read_plate_map` read a plate map that places each sample
  at one or two levels (a culture plate and well, then the analytics plate and
  well the culture was read in). `join_by_position` joins measurements to the
  map **by plate and well, never by sample name**, and returns the mismatch
  report: map positions with no measurement, measurements at positions the map
  does not have, positions measured more than once, and positions where the
  instrument's sample name disagrees with the map (`well_name_mismatch`). All
  rows stay in the joined table with a `match` column saying which case each
  row is.

Values are never converted, rescaled or filled. A grid value carries the
measurement name and unit the user declares (e.g. "OD600", no unit); the parser
does not guess either. Plate ids are compared exactly (after trimming spaces).

Role suggestions use the setup-suggestion vocabulary (`setting`, `outcome`,
`covariate`, `batch_group`, `clone`, `id`, `derived_from_target`): plate id
columns are `batch_group` (wells on one plate share a run, so plate is a group,
not a factor), sample and well columns are `id`. Whether a measured value is an
outcome (harvest OD) or a covariate (inoculum OD) depends on the experiment, so
no role is suggested for it; the user declares it.

Idea credit: the lab's plate-map practice (join on well position, not on
sample names; normalize "A01" to "A1") from Bioqore's ad hoc analyses; see
NOTICE. Re-implemented; no code copied.
"""
from __future__ import annotations

import io
import math
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from kalos.ingest.findings import Finding, Severity, finding

PARSER_VERSION = "plates/1"

# plate format (number of wells) -> (rows, columns)
PLATE_FORMATS: dict[int, tuple[int, int]] = {24: (4, 6), 96: (8, 12), 384: (16, 24)}
_FORMAT_BY_COLUMNS = {cols: fmt for fmt, (_rows, cols) in PLATE_FORMATS.items()}
_ROW_LETTERS = "ABCDEFGHIJKLMNOP"  # 384 is the largest supported format (rows A..P)
_WELL_RE = re.compile(r"^\s*([A-Za-z])\s*0*([0-9]{1,3})\s*$")

MATCHED, MAP_ONLY, MEASUREMENT_ONLY = "matched", "map_only", "measurement_only"
_LIST_CAP = 50  # cap on lists of wells/positions inside a finding's detail


def normalize_well(label: object, plate_format: int | None = None) -> str | None:
    """Return the canonical well ("A1"), or None if `label` is not a well.

    With `plate_format`, a well outside that plate (e.g. "I1" or "A13" on a
    96-well plate) is also None. Without it, the bound is the 384-well plate.
    """
    if label is None or (isinstance(label, float) and math.isnan(label)):
        return None
    m = _WELL_RE.match(str(label))
    if not m:
        return None
    letter, number = m.group(1).upper(), int(m.group(2))
    rows, cols = PLATE_FORMATS.get(plate_format or 384, PLATE_FORMATS[384])
    if letter not in _ROW_LETTERS[:rows] or not 1 <= number <= cols:
        return None
    return f"{letter}{number}"


def well_row_col(well: str) -> tuple[str, int]:
    """("B", 7) for "B7". `well` must already be canonical."""
    return well[0], int(well[1:])


def wells(plate_format: int) -> list[str]:
    """Every well of a plate format in row-major order (A1, A2, ..., B1, ...)."""
    rows, cols = PLATE_FORMATS[plate_format]
    return [f"{r}{c}" for r in _ROW_LETTERS[:rows] for c in range(1, cols + 1)]


def read_text(path: str | os.PathLike[str]) -> tuple[str, str]:
    """Decode an instrument export: (text, encoding used).

    UTF-16 with a byte-order mark and UTF-8 (with or without BOM) are tried in
    that order; anything else is read as Latin-1, which never fails, and the
    encoding is returned so the caller can report it.
    """
    data = Path(path).read_bytes()
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return data.decode("utf-16"), "utf-16"
    try:
        return data.decode("utf-8-sig"), "utf-8"
    except UnicodeDecodeError:
        return data.decode("latin-1"), "latin-1"


# ----------------------------------------------------------------------------
# Grids
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class PlateTable:
    """A long table (one row per well or per map entry) and what was found."""

    table: pd.DataFrame
    findings: tuple[Finding, ...]
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "table": self.table.to_dict(orient="records"),
            "findings": [f.to_dict() for f in self.findings],
            "meta": dict(self.meta),
        }


def _sniff_delimiter(lines: Sequence[str]) -> str:
    sample = "\n".join(lines[:200])
    if "\t" in sample:
        return "\t"
    return ";" if sample.count(";") > sample.count(",") else ","


def _cells(line: str, delim: str) -> list[str]:
    cells = [c.strip().strip('"').strip() for c in line.rstrip("\r\n").split(delim)]
    while cells and cells[-1] == "":
        cells.pop()
    return cells


def _grid_header_columns(cells: Sequence[str]) -> int | None:
    """n if cells[1:] are exactly "1".."n" for a supported n, else None."""
    nums = list(cells[1:])
    n = len(nums)
    if n not in _FORMAT_BY_COLUMNS:
        return None
    return n if nums == [str(i) for i in range(1, n + 1)] else None


def _parse_number(raw: str, decimal: str) -> float | None:
    text = raw.strip()
    if decimal != ".":
        if "." in text:
            return None
        text = text.replace(decimal, ".")
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def parse_grid_text(
    text: str,
    *,
    measurement: str,
    unit: str | None = None,
    plate: str | None = None,
    plate_format: int | None = None,
    decimal: str = ".",
) -> PlateTable:
    """Read every plate grid in `text` into a long table, one row per well.

    `measurement` (e.g. "OD600") and `unit` are what the user says the numbers
    are; both are carried on every row, never inferred. `plate` is the plate id
    the grid belongs to (needed to join a map with several plates). With several
    grids in one file, each keeps its own `block` number.

    Columns: plate, block, well, row, column, measurement, unit, value, raw,
    status ("ok", "empty", "not_numeric"), plate_format, source_line (1-based).
    """
    lines = text.splitlines()
    delim = _sniff_delimiter(lines)
    records: list[dict] = []
    found: list[Finding] = []
    detected: list[int] = []
    i = 0
    block = 0
    while i < len(lines):
        header = _cells(lines[i], delim)
        ncols = _grid_header_columns(header)
        if ncols is None:
            i += 1
            continue
        block += 1
        fmt = _FORMAT_BY_COLUMNS[ncols]
        nrows = PLATE_FORMATS[fmt][0]
        detected.append(fmt)
        if plate_format is not None and plate_format != fmt:
            found.append(finding(
                "plate_format_mismatch", "error",
                f"Grid {block} has {ncols} columns, which is a {fmt}-well plate, but the plate was "
                f"declared as {plate_format}-well. Check the declared format or the export.",
                block=block, line=i + 1, declared=plate_format, detected=fmt,
            ))
        j = i + 1
        row_k = 0
        extra: list[str] = []
        while j < len(lines) and row_k < nrows:
            cells = _cells(lines[j], delim)
            if not cells or cells[0].upper() != _ROW_LETTERS[row_k]:
                break
            letter = _ROW_LETTERS[row_k]
            if len(cells) - 1 > ncols:
                extra.append(f"{letter} (line {j + 1})")
            for c in range(1, ncols + 1):
                raw = cells[c] if c < len(cells) else ""
                if raw == "":
                    value, status = math.nan, "empty"
                else:
                    num = _parse_number(raw, decimal)
                    value, status = (num, "ok") if num is not None else (math.nan, "not_numeric")
                records.append({
                    "plate": plate, "block": block, "well": f"{letter}{c}", "row": letter,
                    "column": c, "measurement": measurement, "unit": unit, "value": value,
                    "raw": raw, "status": status, "plate_format": fmt, "source_line": j + 1,
                })
            row_k += 1
            j += 1
        if row_k < nrows:
            found.append(finding(
                "grid_incomplete", "error",
                f"Grid {block} (line {i + 1}) has {ncols} columns, so {nrows} rows "
                f"({_ROW_LETTERS[0]}-{_ROW_LETTERS[nrows - 1]}) were expected, but only {row_k} "
                "were found. The rows that are there are kept; check the export for a cut-off plate.",
                block=block, line=i + 1, rows_expected=nrows, rows_found=row_k,
            ))
        if extra:
            found.append(finding(
                "grid_extra_cells", "warning",
                f"Grid {block} has values beyond column {ncols} on {len(extra)} row(s); they were "
                "not read. Check whether the export has a second plate side by side.",
                block=block, where=extra[:_LIST_CAP],
            ))
        i = j

    table = pd.DataFrame.from_records(records, columns=[
        "plate", "block", "well", "row", "column", "measurement", "unit", "value", "raw",
        "status", "plate_format", "source_line",
    ])
    if block == 0:
        found.append(finding(
            "no_grid_found", "error",
            "No plate grid was found: a grid starts with a header of column numbers 1-6 (24-well), "
            "1-12 (96-well) or 1-24 (384-well), followed by one line per row letter. Export the plate "
            "as a grid, or upload it as a table with a well column.",
        ))
    else:
        grid_checks: tuple[tuple[str, str, Severity, str], ...] = (
            ("empty", "grid_empty_wells", "info", "are empty"),
            ("not_numeric", "grid_value_not_numeric", "warning", "do not hold a number"),
        )
        for status, check, severity, what in grid_checks:
            hit = table.index[table["status"] == status]
            if len(hit):
                sub = table.loc[hit]
                where = [f"grid{b}:{w}" if block > 1 else str(w)
                         for b, w in zip(sub["block"], sub["well"], strict=True)]
                msg = f"{len(hit)} well(s) {what}; they are kept with no value (never filled)."
                if status == "not_numeric":
                    msg += (" The raw text is kept in `raw`. An overflow marker means the reading was "
                            "above the reader's range: dilute and re-read those wells.")
                    if decimal == "." and sub["raw"].str.fullmatch(r"-?\d+,\d+").all():
                        msg += " They look like decimal commas: read again with decimal=','."
                found.append(finding(
                    check, severity, msg, column="value", rows=hit,
                    wells=where[:_LIST_CAP],
                    raw_values=sorted(set(sub["raw"]))[:10] if status == "not_numeric" else [],
                ))
        if block > 1:
            found.append(finding(
                "multiple_grids", "info",
                f"{block} grids were found; each keeps its own `block` number. Give each grid its "
                "plate id before joining it to a plate map.",
                n_blocks=block,
            ))
    meta = {
        "parser": PARSER_VERSION, "delimiter": delim, "decimal": decimal, "n_grids": block,
        "plate_formats": detected, "measurement": measurement, "unit": unit,
        "role_suggestions": {"plate": "batch_group", "well": "id"},
    }
    return PlateTable(table, tuple(found), meta)


def read_grid(path: str | os.PathLike[str], **kwargs) -> PlateTable:
    """`parse_grid_text` on a file; the encoding used is added to `meta`."""
    text, encoding = read_text(path)
    result = parse_grid_text(text, **kwargs)
    return PlateTable(result.table, result.findings, {**result.meta, "encoding": encoding})


# ----------------------------------------------------------------------------
# Plate maps
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class MapLevel:
    """One placement level of a plate map: which columns hold the plate and the well."""

    name: str  # e.g. "culture" or "analytics"; output columns are f"{name}_plate", f"{name}_well"
    plate_column: str
    well_column: str
    plate_format: int | None = None


@dataclass(frozen=True)
class PlateMapColumns:
    sample: str | None
    levels: tuple[MapLevel, ...]


def _tokens(header: str) -> list[str]:
    toks = re.sub(r"[^0-9a-z]+", " ", str(header).lower()).split()
    return ["plate" if t == "plates" else ("well" if t == "wells" else t) for t in toks]


def _is_position(toks: Sequence[str]) -> bool:
    return bool(toks) and ("position" in toks or "pos" in toks or toks[-1] == "well")


def _level_name(key: set[str], index: int) -> str:
    if any(t.startswith("cultur") for t in key):
        return "culture"
    if any(t.startswith(("analyt", "assay", "read")) for t in key):
        return "analytics"
    return f"level{index + 1}"


def detect_plate_map_columns(headers: Sequence[str]) -> tuple[PlateMapColumns | None, str | None]:
    """Find the sample column and the (plate, well) column pairs from headers.

    Returns (columns, None) or (None, reason). A position column is one whose
    header says "position"/"pos" or ends in "well"; it pairs with the plate
    column whose header has the same remaining words ("24 well culture plates"
    with "24 well culture plate position"). A format number in the header (24,
    96, 384) becomes the level's plate format. Anything ambiguous is refused so
    the user declares the columns (`PlateMapColumns`) instead.
    """
    toks = {h: _tokens(h) for h in headers}
    sample_cols = [h for h in headers if "sample" in toks[h]]
    pos_cols = [h for h in headers if _is_position(toks[h])]
    plate_cols = [h for h in headers if "plate" in toks[h] and h not in pos_cols]
    if not pos_cols:
        return None, "no column header names a well position (e.g. 'culture plate position', 'well')"
    drop = {"position", "pos", "well", "plate"}
    used: set[str] = set()
    levels: list[MapLevel] = []
    for idx, pos in enumerate(pos_cols):
        key = set(toks[pos]) - drop
        exact = [p for p in plate_cols if set(toks[p]) - drop == key and p not in used]
        if len(exact) != 1:
            if len(plate_cols) == 1 and len(pos_cols) == 1:
                exact = plate_cols
            else:
                return None, (f"cannot tell which plate column goes with the position column "
                              f"'{pos}'")
        used.add(exact[0])
        fmt = next((int(t) for t in key if t.isdigit() and int(t) in PLATE_FORMATS), None)
        levels.append(MapLevel(_level_name(key, idx), exact[0], pos, fmt))
    names = [lv.name for lv in levels]
    if len(set(names)) != len(names):
        levels = [MapLevel(f"level{k + 1}", lv.plate_column, lv.well_column, lv.plate_format)
                  for k, lv in enumerate(levels)]
    if len(sample_cols) > 1:
        return None, f"more than one sample column: {sample_cols}"
    return PlateMapColumns(sample_cols[0] if sample_cols else None, tuple(levels)), None


def parse_plate_map(frame: pd.DataFrame, columns: PlateMapColumns | None = None) -> PlateTable:
    """Read a plate map into one row per entry with canonical plate and well columns.

    Output columns: `sample` (if the map has one), `<level>_plate` and
    `<level>_well` per level, every other map column unchanged (e.g. a media
    condition), and `map_row` (0-based row in the map). Cells are kept as text.
    """
    found: list[Finding] = []
    frame = frame.reset_index(drop=True)
    if columns is None:
        columns, reason = detect_plate_map_columns([str(c) for c in frame.columns])
        if columns is None:
            found.append(finding(
                "plate_map_columns_unclear", "error",
                f"The plate map's columns could not be read: {reason}. Declare the sample, plate "
                "and well columns for each level.",
                headers=[str(c) for c in frame.columns][:_LIST_CAP],
            ))
            return PlateTable(frame.iloc[0:0].copy(), tuple(found), {"parser": PARSER_VERSION})
    structural = {lv.plate_column for lv in columns.levels} | {lv.well_column for lv in columns.levels}
    if columns.sample:
        structural.add(columns.sample)
    out = pd.DataFrame(index=frame.index)
    if columns.sample:
        out["sample"] = frame[columns.sample].map(_text)
    for lv in columns.levels:
        out[f"{lv.name}_plate"] = frame[lv.plate_column].map(_text)
        out[f"{lv.name}_well"] = frame[lv.well_column].map(lambda w, f=lv.plate_format: normalize_well(w, f))
    for col in frame.columns:
        if col not in structural:
            out[col] = frame[col]
    out["map_row"] = range(len(frame))

    for lv in columns.levels:
        pcol, wcol = f"{lv.name}_plate", f"{lv.name}_well"
        raw_well = frame[lv.well_column].map(_text)
        missing = out.index[(out[pcol] == "") | (raw_well == "")]
        if len(missing):
            found.append(finding(
                "plate_map_missing_position", "error",
                f"{len(missing)} map row(s) have no {lv.name} plate or well; they cannot be joined "
                "by position and are kept unjoined.",
                column=lv.well_column, rows=missing, level=lv.name,
            ))
        bad = out.index[out[wcol].isna() & (raw_well != "")]
        if len(bad):
            fmt = f"{lv.plate_format}-well " if lv.plate_format else ""
            found.append(finding(
                "well_label_invalid", "error",
                f"{len(bad)} {lv.name} well label(s) are not wells of a {fmt}plate; they are kept "
                "unjoined. Fix the labels in the map.",
                column=lv.well_column, rows=bad, level=lv.name,
                labels=sorted(set(raw_well.loc[bad]))[:_LIST_CAP],
            ))
    for k, lv in enumerate(columns.levels):
        found.extend(_duplicate_positions(out, lv, last=k == len(columns.levels) - 1))
    meta = {
        "parser": PARSER_VERSION,
        "levels": [{"name": lv.name, "plate_column": lv.plate_column, "well_column": lv.well_column,
                    "plate_format": lv.plate_format} for lv in columns.levels],
        "sample_column": columns.sample,
        "role_suggestions": {
            **({"sample": "id"} if columns.sample else {}),
            **{f"{lv.name}_plate": "batch_group" for lv in columns.levels},
            **{f"{lv.name}_well": "id" for lv in columns.levels},
        },
    }
    return PlateTable(out, tuple(found), meta)


def _text(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return " ".join(str(value).split())


def _duplicate_positions(out: pd.DataFrame, lv: MapLevel, *, last: bool) -> list[Finding]:
    pcol, wcol = f"{lv.name}_plate", f"{lv.name}_well"
    ok = out[wcol].notna() & (out[pcol] != "")
    sub = out.loc[ok]
    found: list[Finding] = []
    conflicts: list[int] = []
    replicate_rows: list[int] = []
    conflict_where: list[str] = []
    replicate_where: list[str] = []
    for (plate, well), grp in sub.groupby([pcol, wcol], sort=True):
        if len(grp) < 2:
            continue
        same_sample = "sample" in grp and grp["sample"].nunique() == 1
        if same_sample and not last:
            replicate_rows.extend(grp.index)
            replicate_where.append(f"{plate}:{well}")
        else:
            conflicts.extend(grp.index)
            conflict_where.append(f"{plate}:{well}")
    if conflicts:
        found.append(finding(
            "plate_map_duplicate_position", "error",
            f"{len(conflict_where)} {lv.name} position(s) are listed more than once in the plate map. "
            "A measurement at such a position cannot be assigned to one sample; fix the map.",
            column=lv.well_column, rows=conflicts, level=lv.name, positions=conflict_where[:_LIST_CAP],
        ))
    if replicate_rows:
        found.append(finding(
            "technical_replicates", "info",
            f"{len(replicate_where)} {lv.name} well(s) are each read in more than one later well. These "
            "are repeat reads of one culture, not independent runs; treat them as one run.",
            column=lv.well_column, rows=replicate_rows, level=lv.name,
            positions=replicate_where[:_LIST_CAP],
        ))
    return found


def read_plate_map(path: str | os.PathLike[str], columns: PlateMapColumns | None = None) -> PlateTable:
    """`parse_plate_map` on a CSV/TSV file; every cell is read as text."""
    text, encoding = read_text(path)
    delim = _sniff_delimiter(text.splitlines())
    frame = pd.read_csv(io.StringIO(text), sep=delim, dtype=str, keep_default_na=False)
    result = parse_plate_map(frame, columns)
    return PlateTable(result.table, result.findings, {**result.meta, "encoding": encoding})


# ----------------------------------------------------------------------------
# Join by position
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class JoinResult:
    """Measurements joined to a plate map by position, with the mismatch report."""

    table: pd.DataFrame
    findings: tuple[Finding, ...]
    report: dict

    def to_dict(self) -> dict:
        return {
            "table": self.table.to_dict(orient="records"),
            "findings": [f.to_dict() for f in self.findings],
            "report": dict(self.report),
        }


def _refused(measurements: pd.DataFrame, f: Finding) -> JoinResult:
    table = measurements.reset_index(drop=True).copy()
    table["match"] = "not_joined"
    return JoinResult(table, (f,), {"joined": False, "reason": f.check})


def join_by_position(
    plate_map: PlateTable,
    measurements: pd.DataFrame,
    *,
    level: str,
    plate: str | None = None,
    sample_column: str | None = None,
) -> JoinResult:
    """Join a long measurement table to a plate map on (plate, well) at `level`.

    `measurements` needs a `well` column and either a `plate` column or the
    `plate` argument (which sets the plate for every measurement row). If the
    map has a single plate at that level and none is given, that plate is used
    and the report says so. `sample_column` names the instrument's own sample
    name column; where it disagrees with the map at the same position, the row
    is still joined by position and the disagreement is reported as
    `well_name_mismatch`.

    The table is an outer join: every map entry on the measured plate(s) and
    every measurement row appear, with `match` = matched / map_only /
    measurement_only. Map entries on plates that were not measured are left out
    of the table and listed in the report.
    """
    pcol, wcol = f"{level}_plate", f"{level}_well"
    mp = plate_map.table
    if pcol not in mp.columns or wcol not in mp.columns:
        levels = sorted(c[: -len("_plate")] for c in mp.columns if str(c).endswith("_plate"))
        return _refused(measurements, finding(
            "join_level_unknown", "error",
            f"The plate map has no level '{level}' (levels: {levels}). Choose the level the "
            "measurements were read at.", level=level, levels=levels,
        ))
    if "well" not in measurements.columns:
        return _refused(measurements, finding(
            "join_no_well_column", "error",
            "The measurements have no `well` column, so they cannot be joined by position.",
        ))
    meas = measurements.reset_index(drop=True).copy()
    meas["measurement_row"] = range(len(meas))
    found: list[Finding] = []
    map_plates = sorted(p for p in mp[pcol].dropna().unique() if p != "")
    if plate is not None:
        meas["plate"] = _text(plate)
    elif "plate" not in meas.columns or (meas["plate"].isna() | (meas["plate"].map(_text) == "")).all():
        if len(map_plates) != 1:
            return _refused(measurements, finding(
                "plate_not_declared", "error",
                f"The measurements carry no plate id and the map has {len(map_plates)} {level} plates "
                f"({map_plates[:_LIST_CAP]}). Say which plate was measured.",
                level=level, map_plates=map_plates[:_LIST_CAP],
            ))
        meas["plate"] = map_plates[0]
        found.append(finding(
            "plate_from_map", "info",
            f"The measurements carry no plate id; the map's only {level} plate "
            f"'{map_plates[0]}' was used.", level=level, plate=map_plates[0],
        ))
    meas["plate"] = meas["plate"].map(_text)
    fmt = next((lv.get("plate_format") for lv in plate_map.meta.get("levels", [])
                if lv.get("name") == level), None)
    meas["well_raw"] = meas["well"]
    meas["well"] = meas["well"].map(lambda w: normalize_well(w, fmt))

    measured_plates = sorted(p for p in meas["plate"].unique() if p != "")
    unknown_plates = [p for p in measured_plates if p not in map_plates]
    if unknown_plates:
        found.append(finding(
            "plate_not_in_map", "warning",
            f"Measured plate(s) {unknown_plates[:_LIST_CAP]} are not {level} plates in the map "
            f"({map_plates[:_LIST_CAP]}). Plate ids are compared exactly; their rows stay unjoined.",
            level=level, plates=unknown_plates[:_LIST_CAP], map_plates=map_plates[:_LIST_CAP],
        ))
    unmeasured = [p for p in map_plates if p not in measured_plates]

    # Map entries on the measured plates, plus entries with no plate at all (kept, unmatched).
    left = mp[mp[pcol].isin(measured_plates) | (mp[pcol] == "")].copy()
    left["_key"] = [f"{p}\x1f{w}" if p and isinstance(w, str) else f"\x00map{r}"
                    for p, w, r in zip(left[pcol], left[wcol], left["map_row"], strict=True)]
    right = meas.rename(columns={"plate": "_m_plate", "well": "_m_well"})
    right["_key"] = [f"{p}\x1f{w}" if p and isinstance(w, str) else f"\x00meas{r}"
                     for p, w, r in zip(right["_m_plate"], right["_m_well"], right["measurement_row"],
                                        strict=True)]
    joined = left.merge(right, how="outer", on="_key", indicator=True,
                        suffixes=("", "_measurement"), sort=False)
    joined["match"] = joined["_merge"].map(
        {"both": MATCHED, "left_only": MAP_ONLY, "right_only": MEASUREMENT_ONLY}).astype(str)
    joined[pcol] = joined[pcol].where(joined[pcol].notna(), joined["_m_plate"])
    joined[wcol] = joined[wcol].where(joined[wcol].notna(), joined["_m_well"])
    joined = joined.drop(columns=["_merge", "_key", "_m_plate", "_m_well"])
    joined = joined.sort_values(["map_row", "measurement_row"], na_position="last", kind="stable")
    joined = joined.reset_index(drop=True)
    bad_meas = meas.index[meas["well"].isna()]

    report: dict = {"joined": True, "level": level, "plates": measured_plates,
                    "map_plates_not_measured": unmeasured}
    if len(bad_meas):
        rows = joined.index[joined["measurement_row"].isin(bad_meas)]
        found.append(finding(
            "well_label_invalid", "error",
            f"{len(bad_meas)} measurement row(s) have a well label that is not a well"
            f"{f' of a {fmt}-well plate' if fmt else ''}; they are kept unjoined.",
            column="well_raw", rows=rows,
            labels=sorted({str(v) for v in meas.loc[bad_meas, "well_raw"]})[:_LIST_CAP],
        ))

    # mismatch report
    has_value = "value" in joined.columns
    map_only = joined.index[joined["match"] == MAP_ONLY]
    meas_only = joined.index[joined["match"] == MEASUREMENT_ONLY]
    report["matched"] = int((joined["match"] == MATCHED).sum())
    report["map_only"] = _positions(joined, map_only, pcol, wcol)
    report["measurement_only"] = _positions(joined, meas_only, pcol, wcol)
    if len(map_only):
        found.append(finding(
            "map_position_not_measured", "warning",
            f"{len(map_only)} map position(s) on the measured plate(s) have no measurement. They are "
            "kept with no value; check whether the wells were skipped or the plate id is wrong.",
            rows=map_only, level=level, positions=report["map_only"][:_LIST_CAP],
        ))
    if len(meas_only):
        if has_value:
            empty = joined.loc[meas_only, "value"].isna()
            with_value, without = meas_only[~empty.to_numpy()], meas_only[empty.to_numpy()]
        else:
            with_value, without = meas_only, meas_only[:0]
        if len(with_value):
            found.append(finding(
                "measurement_position_not_in_map", "warning",
                f"{len(with_value)} measured position(s) are not in the plate map, so they have no "
                "sample. They are kept unjoined; add them to the map or confirm they are empty wells "
                "or instrument standards.",
                rows=with_value, level=level,
                positions=_positions(joined, with_value, pcol, wcol)[:_LIST_CAP],
            ))
        if len(without):
            found.append(finding(
                "unmapped_empty_wells", "info",
                f"{len(without)} position(s) outside the plate map have no value (empty wells).",
                rows=without, level=level,
            ))
    if unmeasured:
        found.append(finding(
            "map_plates_not_measured", "info",
            f"{len(unmeasured)} {level} plate(s) in the map were not in these measurements: "
            f"{unmeasured[:_LIST_CAP]}.", level=level, plates=unmeasured[:_LIST_CAP],
        ))

    found.extend(_multiply_matched(joined, measurements.columns, pcol, wcol, report))
    if sample_column is not None:
        inst_col = f"{sample_column}_measurement" if sample_column in mp.columns else sample_column
        found.extend(_name_mismatches(joined, mp, inst_col, pcol, wcol, report))
    return JoinResult(joined, tuple(found), report)


def _positions(table: pd.DataFrame, idx: pd.Index, pcol: str, wcol: str) -> list[str]:
    return [f"{_text(p)}:{_text(w) or '?'}" for p, w in
            zip(table.loc[idx, pcol], table.loc[idx, wcol], strict=True)]


def _multiply_matched(joined: pd.DataFrame, meas_columns: pd.Index, pcol: str, wcol: str,
                      report: dict) -> list[Finding]:
    """Positions that got more than one row of the same measurement."""
    keys = [pcol, wcol] + [c for c in ("measurement", "block") if c in meas_columns]
    m = joined[(joined["match"] != MAP_ONLY) & joined[wcol].notna()]
    dup = m[m.duplicated(subset=keys, keep=False)]
    report["multiply_matched"] = sorted({f"{p}:{w}" for p, w in zip(dup[pcol], dup[wcol], strict=True)})
    if dup.empty:
        return []
    return [finding(
        "position_multiply_matched", "warning",
        f"{len(report['multiply_matched'])} position(s) have more than one row of the same "
        "measurement. All rows are kept; check for a duplicated export or a plate read twice. "
        "Repeat reads of one well are not independent runs.",
        rows=dup.index, positions=report["multiply_matched"][:_LIST_CAP],
    )]


def _name_mismatches(joined: pd.DataFrame, mp: pd.DataFrame, sample_column: str, pcol: str,
                     wcol: str, report: dict) -> list[Finding]:
    if sample_column not in joined.columns or "sample" not in joined.columns:
        report["name_mismatches"] = []
        return [finding(
            "sample_names_not_compared", "info",
            f"Sample names were not compared: the instrument column '{sample_column}' or the map's "
            "sample column is missing.", column=sample_column,
        )]
    matched = joined[joined["match"] == MATCHED]
    map_name = matched["sample"].map(_text)
    inst_name = matched[sample_column].map(_text)
    differ = matched.index[(map_name != inst_name).to_numpy()]
    where_in_map: dict[str, str] = {}
    for p, w, s in zip(mp[pcol], mp[wcol], mp["sample"], strict=True):
        where_in_map.setdefault(_text(s), f"{p}:{w}")
    items = []
    for i in differ:
        name = _text(joined.at[i, sample_column])
        items.append({
            "position": f"{_text(joined.at[i, pcol])}:{_text(joined.at[i, wcol])}",
            "map_sample": _text(joined.at[i, "sample"]),
            "instrument_sample": name,
            "instrument_name_is_at": where_in_map.get(name),
        })
    report["name_mismatches"] = items
    if not items:
        return []
    return [finding(
        "well_name_mismatch", "warning",
        f"At {len(items)} position(s) the instrument's sample name differs from the plate map. "
        "Rows were joined by plate and well, not by name; check that the map and the instrument "
        "run used the same layout.",
        column=sample_column, rows=differ, mismatches=items[:_LIST_CAP],
    )]
