"""S.NEST plate exports: DO and pH time series, as-run conditions, per-well summaries.

The S.NEST is a shaken microbioreactor plate (24- or 96-well) that reads
dissolved oxygen (DO) and pH in every well through optical sensor spots. Its CSV
export (`DO.csv`, `pH.csv`) is one file per channel:

1. a metadata block in `***Section***` blocks: `Key:,value[,unit]` lines
   (experiment name, start time, plate format, working volume, temperature,
   CO2, mixing, sensing), the plate layout (`Well ID,Group,Note`), the pauses
   (`Pause No.,from,to`) and the data-processing options (smoothing, baseline
   correction);
2. a data block under `***S.NEST DO data***` (or `pH data`): a header
   `Time,Running time(min),<well>_<group>_<note>,...` and one line per sampling
   time with one value per well.

`parse_snest_text` / `read_snest_csv` read one such file into a long table, one
row per well and sampling time, and never drop a row: a missing reading
(`N/A` or empty) or a value that is not a number stays in the table with `value`
NaN, its raw text and a `status`, and is reported by well and time.

Time: `time_h` is the instrument's `Running time` converted to hours, i.e.
elapsed time from the start time recorded in the export (`Start time`). With a
declared `start` (e.g. the inoculation time) it is shifted to elapsed time from
that start instead; `meta["time_basis"]` says which. The wall-clock `Time` is
kept as written in `timestamp`, and checked against `Start time` + running time.

Units are shown, never guessed: the export does not state the DO unit, so DO
rows carry the unit the user declares (e.g. "% air saturation") or none, with a
finding. pH rows carry "pH" (the channel is its own scale).

`read_snest` reads the DO and pH files of one or more plates and adds a per-well
summary table (`SnestResult.summary`). Every summary column has its formula and
the readings it was derived from in `meta["summary_columns"]`. Summaries need at
least `MIN_VALID_POINTS` valid readings of a channel in a well; a well with
fewer gets NaN for that channel's summaries and a finding, never an
extrapolated value. Time below a DO threshold needs a user-declared threshold;
without one that column is refused (NaN, error finding), never defaulted.

The metadata block is kept as recorded: `meta["conditions"]` holds every key
verbatim with its value and unit as written, `meta["as_run"]` the known plate
conditions (working volume, temperature, CO2, mixing) under stable names. They
are the conditions the export records for the run, not set points.

Roles (setup-suggestion vocabulary): plate is `batch_group` (wells on one plate
share a run), wells are `id`, and every summary is measured during the run, so
it is an `outcome` or a `covariate`, never a `setting`: pH start (read before
the culture has changed the medium much) is a `covariate`; the other summaries
are `outcome`s of the run.

The TXT exports (`DO.txt`, `pH.txt`: epoch milliseconds then one value per well)
are refused with a finding: they carry no well labels, start time or unit; in
the exports we checked, the DO TXT writes 0 where the CSV writes `N/A` (a
missing reading cannot be told from a true 0), and the pH TXT values are not
the CSV values (a processing stage the file does not name).

Idea credit: reading the S.NEST DO/pH exports per well beside the run's other
plate data, keeping the plate conditions the export records, from Bioqore's ad
hoc lab analyses (their run folders keep these exports; they have no parsing
code for them); see NOTICE. Re-implemented; no code copied.
"""
from __future__ import annotations

import csv
import math
import os
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from kalos.ingest.findings import Finding, finding

PARSER_VERSION = "snest/1"

# Fewer valid readings than this in one channel of one well and its summaries
# are NaN (with a finding). Two readings are a start and an end, not a curve:
# a min or a drift from them is one difference, so three is the floor.
MIN_VALID_POINTS = 3

# plate format (number of wells) -> (rows, columns)
PLATE_FORMATS: dict[int, tuple[int, int]] = {24: (4, 6), 96: (8, 12)}
_ROW_LETTERS = "ABCDEFGHIJKLMNOP"
_WELL_RE = re.compile(r"^\s*([A-Za-z])\s*0*([0-9]{1,3})\s*$")
_TITLE_RE = re.compile(r"^\*{3}\s*S\.?\s*NEST\s+(DO|pH)\s+data\s*(?:\(([^)]*)\))?\s*\*{3}$", re.I)
_SECTION_RE = re.compile(r"^\*{3}\s*(.*?)\s*\*{3}$")
_RUNNING_RE = re.compile(r"^running\s*time\s*\(\s*([a-z]+)\s*\)$", re.I)
_TXT_RE = re.compile(r"^\d{10,14}\s+[-+0-9.eE,]+$")
_MISSING_MARKERS = frozenset({"", "N/A", "NA", "n/a", "NaN", "nan"})
_TO_HOURS = {"s": 1 / 3600, "sec": 1 / 3600, "min": 1 / 60, "mins": 1 / 60, "h": 1.0, "hr": 1.0}
_CLOCK_FORMATS = ("%Y/%m/%d %H:%M:%S", "%Y/%m/%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M")
_CLOCK_TOLERANCE_MIN = 1.5  # the clock column shows whole minutes
_LIST_CAP = 50
_PH_RANGE = (0.0, 14.0)

OK, MISSING, NOT_NUMERIC, NO_TIME = "ok", "missing", "not_numeric", "no_time"

# known metadata keys (lower case, no trailing ":") -> stable name in meta["as_run"]
_AS_RUN_KEYS = {
    "working volume": "working_volume",
    "temperature": "temperature",
    "co2": "co2",
    "mixing": "mixing",
    "mixing rate": "mixing_rate",
    "mode": "mixing_mode",
    "suspension boost": "suspension_boost",
    "cycle": "mixing_cycle",
    "interval": "mixing_interval",
    "sampling rate": "sampling_interval",
}
_TABLE_SECTIONS = {"plate layout", "pause information"}

TABLE_COLUMNS = [
    "plate", "well", "well_canonical", "group", "note", "channel", "time_h", "timestamp",
    "running_time_raw", "value", "raw", "unit", "status", "in_pause", "source", "source_line",
]
SUMMARY_COLUMNS = [
    "plate", "well", "well_canonical", "group", "note", "baseline_reference_well",
    "n_valid_do", "n_valid_ph", "do_min", "do_final", "do_time_below_threshold_h",
    "do_time_unassessed_h", "ph_start", "ph_final", "ph_drift", "ph_min", "ph_max",
]


@dataclass(frozen=True)
class SnestResult:
    """The long time-series table, per-well summaries (if computed), findings and meta."""

    table: pd.DataFrame
    findings: tuple[Finding, ...]
    meta: dict = field(default_factory=dict)
    summary: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=SUMMARY_COLUMNS))

    def to_dict(self) -> dict:
        return {
            "table": _records(self.table),
            "summary": _records(self.summary),
            "findings": [f.to_dict() for f in self.findings],
            "meta": dict(self.meta),
        }


def _records(frame: pd.DataFrame) -> list[dict]:
    """Records with NaN as None and plain Python scalars (strict-JSON safe)."""
    out = []
    for rec in frame.to_dict(orient="records"):
        clean = {}
        for k, v in rec.items():
            if isinstance(v, np.generic):
                v = v.item()
            if isinstance(v, float) and not math.isfinite(v):
                v = None
            clean[str(k)] = v
        out.append(clean)
    return out


def canonical_well(label: object) -> str | None:
    """"A01" -> "A1"; None if `label` is not a well label (any plate up to 16 x 24)."""
    if label is None:
        return None
    m = _WELL_RE.match(str(label))
    if not m:
        return None
    letter, number = m.group(1).upper(), int(m.group(2))
    if letter not in _ROW_LETTERS or not 1 <= number <= 24:
        return None
    return f"{letter}{number}"


def in_plate_format(well: str, plate_format: int) -> bool:
    """True if canonical `well` is on a plate of `plate_format` wells (24 or 96)."""
    rows, cols = PLATE_FORMATS[plate_format]
    return well[0] in _ROW_LETTERS[:rows] and 1 <= int(well[1:]) <= cols


def read_text(path: str | os.PathLike[str]) -> tuple[str, str]:
    """Decode an export: (text, encoding). UTF-16 with BOM, UTF-8 (BOM or not), else Latin-1."""
    data = Path(path).read_bytes()
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return data.decode("utf-16"), "utf-16"
    try:
        return data.decode("utf-8-sig"), "utf-8"
    except UnicodeDecodeError:
        return data.decode("latin-1"), "latin-1"


def _cells(line: str) -> list[str]:
    cells = [c.strip() for c in next(csv.reader([line]))] if line.strip() else []
    while cells and cells[-1] == "":
        cells.pop()
    return cells


def _number(text: str) -> float | None:
    try:
        value = float(text.strip())
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _clock(text: str) -> datetime | None:
    for fmt in _CLOCK_FORMATS:
        try:
            return datetime.strptime(text.strip(), fmt)
        except ValueError:
            continue
    return None


def _empty_table() -> pd.DataFrame:
    return pd.DataFrame(columns=TABLE_COLUMNS)


def _where(sub: pd.DataFrame) -> list[str]:
    """'A1@2.50h' per row (well as exported, time in h), capped."""
    out = []
    for w, t in zip(sub["well"], sub["time_h"], strict=True):
        out.append(f"{w}@{t:.2f}h" if isinstance(t, float) and math.isfinite(t) else f"{w}@?")
    return out[:_LIST_CAP]


# ----------------------------------------------------------------------------
# One file
# ----------------------------------------------------------------------------


def parse_snest_text(
    text: str,
    *,
    plate: str | None = None,
    do_unit: str | None = None,
    start: datetime | None = None,
    plate_format: int | None = None,
    source: str | None = None,
) -> SnestResult:
    """Read one S.NEST CSV export (DO or pH) into a long table, one row per well and time.

    `plate` is the plate id (default: the export's `Experiment name`).
    `do_unit` is the DO unit the user declares (the export does not state it; ignored for pH).
    `start` shifts `time_h` to elapsed time from a declared start. A declared
    `plate_format` is checked against the export's.

    Columns: see `TABLE_COLUMNS`. `status` is "ok", "missing" (N/A or empty),
    "not_numeric" or "no_time" (the running time is not a number).
    """
    lines = text.splitlines()
    found: list[Finding] = []
    first = next((ln.strip() for ln in lines if ln.strip()), "")
    if _TXT_RE.match(first):
        found.append(finding(
            "snest_txt_not_supported", "error",
            "This looks like an S.NEST TXT export (epoch milliseconds, then one value per well). "
            "It has no well labels, start time or unit; its DO values use 0 where the CSV export "
            "says N/A (a gap cannot be told from a true 0); and its pH values are not the ones in "
            "the CSV export. Upload the CSV export (DO.csv, pH.csv) of the same plate instead. "
            "The file was not read, so the table is empty; nothing was changed.",
            source=source,
        ))
        return SnestResult(_empty_table(), tuple(found), {"parser": PARSER_VERSION, "source": source})

    title_idx = next((i for i, ln in enumerate(lines) if _TITLE_RE.match(ln.strip().strip(",").strip())),
                     None)
    meta_lines = lines if title_idx is None else lines[:title_idx]
    md = _parse_metadata(meta_lines)
    meta: dict = {"parser": PARSER_VERSION, "source": source, **md}
    if title_idx is None:
        found.append(finding(
            "no_data_section", "error",
            "No S.NEST data block was found: the CSV export has a line '***S.NEST DO data***' or "
            "'***S.NEST pH data***' followed by 'Time,Running time(min),<wells>'. Export the DO or pH "
            "data as CSV from the instrument software and upload that file. Nothing was read; "
            "nothing was changed.",
            source=source,
        ))
        return SnestResult(_empty_table(), tuple(found), meta)

    m = _TITLE_RE.match(lines[title_idx].strip().strip(",").strip())
    assert m is not None
    channel = "DO" if m.group(1).upper() == "DO" else "pH"
    file_unit = (m.group(2) or "").strip() or None
    meta["channel"] = channel

    # plate id
    exp_name = md["conditions"].get("Experiment name", {}).get("value") or None
    plate_id = plate if plate is not None else exp_name
    meta["plate"] = plate_id
    meta["plate_source"] = "declared" if plate is not None else ("Experiment name" if exp_name else None)
    if plate_id is None:
        found.append(finding(
            "plate_not_declared", "error",
            "The export has no 'Experiment name' and no plate id was declared, so its rows carry no "
            "plate. Declare the plate id when reading the file. All rows were kept.",
            source=source,
        ))

    # unit
    if channel == "pH":
        row_unit: str | None = "pH"
        meta["unit_source"] = "channel"
    elif file_unit is not None:
        row_unit = file_unit
        meta["unit_source"] = "file"
        if do_unit is not None and do_unit != file_unit:
            found.append(finding(
                "unit_conflict", "error",
                f"The export states the DO unit '{file_unit}' but '{do_unit}' was declared. Check "
                "which one is right. All rows were kept with the unit from the file.",
                declared=do_unit, in_file=file_unit, source=source,
            ))
    else:
        row_unit = do_unit
        meta["unit_source"] = "declared" if do_unit is not None else None
        if do_unit is None:
            found.append(finding(
                "unit_not_stated", "warning",
                "The export does not state the unit of the DO values (it is often % air "
                "saturation, but that is not assumed). Declare the unit (do_unit) when reading the "
                "file. All rows were kept with no unit; nothing was changed.",
                column="unit", source=source,
            ))
    meta["unit"] = row_unit

    # plate format
    fmt_text = md["conditions"].get("Plate format", {}).get("value") or ""
    fm = re.search(r"(\d+)\s*-?\s*well", fmt_text, re.I)
    file_fmt = int(fm.group(1)) if fm else None
    if plate_format is not None and file_fmt is not None and plate_format != file_fmt:
        found.append(finding(
            "plate_format_mismatch", "error",
            f"The export says '{fmt_text}' but a {plate_format}-well plate was declared. Check the "
            "declared format or the file. All rows were kept; wells were checked against the "
            "export's format.",
            declared=plate_format, in_file=file_fmt, source=source,
        ))
    fmt = file_fmt if file_fmt is not None else plate_format
    meta["plate_format"] = fmt
    if fmt is not None and fmt not in PLATE_FORMATS:
        found.append(finding(
            "plate_format_unknown", "warning",
            f"Plate format {fmt} is not a 24- or 96-well plate, so wells were not checked against "
            "it. Check the format line of the export. All rows were kept.",
            plate_format=fmt, source=source,
        ))
    elif fmt is None:
        found.append(finding(
            "plate_format_unknown", "warning",
            "The export has no 'Plate format' line and none was declared, so wells were not "
            "checked against the plate. Declare the plate format (24 or 96). All rows were kept.",
            source=source,
        ))

    # time basis
    recorded_start = _clock(md["conditions"].get("Start time", {}).get("value") or "")
    offset_h = 0.0
    if start is not None and recorded_start is not None:
        offset_h = (recorded_start - start).total_seconds() / 3600
        meta["time_basis"] = (f"elapsed h from the declared start {start.isoformat(sep=' ')} "
                              "(running time + recorded start - declared start)")
    else:
        meta["time_basis"] = "elapsed h from the start time recorded in the export (Running time converted to h)"
        if start is not None:
            found.append(finding(
                "start_time_missing", "error",
                "A start was declared but the export has no readable 'Start time', so time could "
                "not be shifted to it. Check the file's 'Start time' line. All rows were kept, with "
                "time from the instrument's running time.",
                source=source,
            ))
    meta["start_time_recorded"] = recorded_start.isoformat(sep=" ") if recorded_start else None
    meta["start_declared"] = start.isoformat(sep=" ") if start is not None else None

    table, data_findings, data_meta = _parse_data(
        lines, title_idx, channel=channel, unit=row_unit, plate=plate_id, fmt=fmt,
        offset_h=offset_h, recorded_start=recorded_start, layout=md["layout"],
        pauses=md["pauses"], source=source,
    )
    found.extend(data_findings)
    meta.update(data_meta)
    found.extend(_processing_findings(md, source))
    meta["role_suggestions"] = {"plate": "batch_group", "well": "id", "well_canonical": "id"}
    return SnestResult(table, tuple(found), meta)


def _parse_metadata(lines: Sequence[str]) -> dict:
    """Every metadata line kept verbatim; known plate conditions under stable names."""
    conditions: dict[str, dict] = {}
    layout: dict[str, dict] = {}
    pauses: list[dict] = []
    notes: list[str] = []
    section = ""
    table_header: list[str] | None = None
    title = None
    for i, line in enumerate(lines):
        cells = _cells(line)
        if not cells:
            table_header = None
            continue
        if set("".join(cells)) <= {"-"}:
            continue
        sm = _SECTION_RE.match(cells[0])
        if sm and len(cells) == 1:
            section = sm.group(1)
            table_header = None
            continue
        if title is None and not section:
            title = cells[0]
            continue
        if section.lower() in _TABLE_SECTIONS:
            if table_header is None:
                table_header = cells
                continue
            if section.lower() == "plate layout":
                label = cells[0]
                layout[canonical_well(label) or label] = {
                    "well": label,
                    "group": cells[1] if len(cells) > 1 else None,
                    "note": ",".join(cells[2:]) if len(cells) > 2 else None,
                }
            else:
                pauses.append({"name": cells[0], "from": cells[1] if len(cells) > 1 else None,
                               "to": cells[2] if len(cells) > 2 else None, "line": i + 1})
            continue
        if len(cells) == 1:
            notes.append(cells[0])
            continue
        key = cells[0][:-1].strip() if cells[0].endswith(":") else cells[0]
        name = key if key not in conditions else f"{section} / {key}"
        conditions[name] = {
            "section": section or None,
            "value": cells[1],
            "unit": cells[2] if len(cells) > 2 else None,
            "values": cells[1:],
            "line": i + 1,
        }
    as_run: dict[str, dict] = {}
    for key, entry in conditions.items():
        stable = _AS_RUN_KEYS.get(key.lower())
        if stable is None or stable in as_run:
            continue
        as_run[stable] = {"key": key, "value": entry["value"], "number": _number(entry["value"]),
                          "unit": entry["unit"]}
    return {
        "title": title,
        "conditions": conditions,
        "as_run": as_run,
        "as_run_note": ("Plate conditions as the export records them for this run, units as "
                        "written; recorded values, not set points."),
        "layout": layout,
        "pauses": pauses,
        "notes": notes,
    }


def _parse_data(
    lines: Sequence[str], title_idx: int, *, channel: str, unit: str | None, plate: str | None,
    fmt: int | None, offset_h: float, recorded_start: datetime | None, layout: dict,
    pauses: list[dict], source: str | None,
) -> tuple[pd.DataFrame, list[Finding], dict]:
    found: list[Finding] = []
    meta: dict = {}
    h = title_idx + 1
    while h < len(lines) and not lines[h].strip():
        h += 1
    header = _cells(lines[h]) if h < len(lines) else []
    rm = _RUNNING_RE.match(header[1]) if len(header) > 1 else None
    to_h = _TO_HOURS.get(rm.group(1).lower()) if rm else None
    if len(header) < 3 or header[0].lower() != "time" or to_h is None:
        found.append(finding(
            "data_header_unreadable", "error",
            "The data block's header is not 'Time,Running time(<min|s|h>),<wells>'. Export the data "
            "as CSV again, or check the file was not edited. Nothing was read; nothing was changed.",
            line=h + 1, header=header[:5], source=source,
        ))
        return _empty_table(), found, meta
    assert rm is not None
    meta["running_time_unit"] = rm.group(1)
    well_headers = header[2:]

    # data lines: until a blank line, a new section or the end
    data: list[tuple[int, list[str]]] = []
    j = h + 1
    while j < len(lines) and lines[j].strip() and not lines[j].lstrip().startswith("***"):
        data.append((j + 1, _cells(lines[j])))
        j += 1

    pause_windows = [(p, _clock(p["from"] or ""), _clock(p["to"] or "")) for p in pauses]
    times: list[float] = []
    clocks: list[datetime | None] = []
    for _line_no, cells in data:
        run = _number(cells[1]) if len(cells) > 1 else None
        times.append(run * to_h + offset_h if run is not None else math.nan)
        clocks.append(_clock(cells[0]) if cells else None)

    records = []
    for k, label in enumerate(well_headers):
        parts = label.split("_")
        well_label = parts[0].strip()
        canon = canonical_well(well_label)
        lay = layout.get(canon or well_label, {})
        group = lay.get("group", parts[1] if len(parts) > 1 else None)
        note = lay.get("note", "_".join(parts[2:]) if len(parts) > 2 else None)
        for (line_no, cells), t, clk in zip(data, times, clocks, strict=True):
            raw = cells[2 + k] if 2 + k < len(cells) else ""
            num = None if raw in _MISSING_MARKERS else _number(raw)
            if not math.isfinite(t):
                status = NO_TIME
            elif raw in _MISSING_MARKERS:
                status = MISSING
            else:
                status = OK if num is not None else NOT_NUMERIC
            in_pause = bool(clk is not None and any(
                a is not None and b is not None and a <= clk <= b for _p, a, b in pause_windows))
            records.append({
                "plate": plate, "well": well_label, "well_canonical": canon, "group": group,
                "note": note, "channel": channel, "time_h": t,
                "timestamp": cells[0] if cells else "", "running_time_raw": cells[1] if len(cells) > 1 else "",
                "value": num if num is not None else math.nan, "raw": raw, "unit": unit,
                "status": status, "in_pause": in_pause, "source": source, "source_line": line_no,
            })
    table = pd.DataFrame.from_records(records, columns=TABLE_COLUMNS)
    meta["wells"] = [w.split("_")[0].strip() for w in well_headers]
    meta["n_times"] = len(data)

    found.extend(_well_findings(table, well_headers, fmt, layout, source))
    found.extend(_time_findings(data, times, clocks, recorded_start, offset_h, to_h, source))
    found.extend(_value_findings(table, channel, source))
    if pauses:
        inside = table.index[table["in_pause"]]
        found.append(finding(
            "pauses_recorded", "info",
            f"The export records {len(pauses)} pause(s). time_h counts straight through them "
            f"(running time is elapsed time since the start). {len(inside)} reading(s) fall inside "
            "a pause, when shaking may have stopped; check them. They were kept as read, marked "
            "in_pause = True.",
            column="in_pause", rows=inside,
            pauses=[{k: p[k] for k in ("name", "from", "to")} for p in pauses][:_LIST_CAP],
            source=source,
        ))
    return table, found, meta


def _well_findings(table: pd.DataFrame, headers: Sequence[str], fmt: int | None, layout: dict,
                   source: str | None) -> list[Finding]:
    found: list[Finding] = []
    labels = [h.split("_")[0].strip() for h in headers]
    canon = [canonical_well(w) for w in labels]
    bad = [w for w, c in zip(labels, canon, strict=True) if c is None]
    if bad:
        rows = table.index[table["well_canonical"].isna()]
        found.append(finding(
            "well_label_invalid", "error",
            f"{len(bad)} data column(s) do not start with a well label. Check the column headers of "
            "the export. Their readings were kept with no well_canonical.",
            column="well", rows=rows, labels=bad[:_LIST_CAP], source=source,
        ))
    if fmt in PLATE_FORMATS:
        outside = sorted({c for c in canon if c is not None and not in_plate_format(c, fmt)})
        if outside:
            rows = table.index[table["well_canonical"].isin(outside)]
            nrows, ncols = PLATE_FORMATS[fmt]
            found.append(finding(
                "well_outside_plate_format", "error",
                f"{len(outside)} well(s) are not on a {fmt}-well plate (rows A-{_ROW_LETTERS[nrows - 1]}, "
                f"columns 1-{ncols}). Check the plate format or the column headers. Their readings "
                "were kept as read.",
                column="well", rows=rows, wells=outside[:_LIST_CAP], plate_format=fmt, source=source,
            ))
    seen: dict[str, int] = {}
    for c in canon:
        if c is not None:
            seen[c] = seen.get(c, 0) + 1
    dup = sorted(w for w, n in seen.items() if n > 1)
    if dup:
        rows = table.index[table["well_canonical"].isin(dup)]
        found.append(finding(
            "duplicate_well_column", "error",
            f"{len(dup)} well(s) have more than one data column, so a well's summary would mix two "
            "series. Check the export. All readings were kept.",
            column="well", rows=rows, wells=dup[:_LIST_CAP], source=source,
        ))
    if layout:
        data_wells = {c for c in canon if c is not None}
        layout_wells = set(layout)
        only_data = sorted(data_wells - layout_wells)
        only_layout = sorted(layout_wells - data_wells)
        if only_data or only_layout:
            found.append(finding(
                "layout_data_mismatch", "warning",
                "The plate layout block and the data columns list different wells "
                f"({len(only_data)} only in the data, {len(only_layout)} only in the layout). Check "
                "that the layout matches the plate. All readings were kept.",
                only_in_data=only_data[:_LIST_CAP], only_in_layout=only_layout[:_LIST_CAP],
                source=source,
            ))
    return found


def _time_findings(data: Sequence[tuple[int, list[str]]], times: Sequence[float],
                   clocks: Sequence[datetime | None], recorded_start: datetime | None,
                   offset_h: float, to_h: float, source: str | None) -> list[Finding]:
    found: list[Finding] = []
    lines = [ln for ln, _c in data]
    no_time = [ln for ln, t in zip(lines, times, strict=True) if not math.isfinite(t)]
    if no_time:
        found.append(finding(
            "time_not_numeric", "error",
            f"{len(no_time)} data line(s) have a running time that is not a number. Check those "
            "lines. Their readings were kept with status 'no_time' and left out of the summaries.",
            column="time_h", lines=no_time[:_LIST_CAP], source=source,
        ))
    dup_lines: list[int] = []
    back_lines: list[int] = []
    seen: set[float] = set()
    latest = -math.inf
    for ln, t in zip(lines, times, strict=True):
        if not math.isfinite(t):
            continue
        if t in seen:
            dup_lines.append(ln)
        elif t < latest:
            back_lines.append(ln)
        seen.add(t)
        latest = max(latest, t)
    if dup_lines:
        found.append(finding(
            "duplicate_timestamp", "warning",
            f"{len(dup_lines)} data line(s) repeat a running time already in the file. Check for a "
            "duplicated export or two joined files. All readings were kept; summaries use the "
            "readings sorted by time.",
            column="time_h", lines=dup_lines[:_LIST_CAP], source=source,
        ))
    if back_lines:
        found.append(finding(
            "time_not_increasing", "warning",
            f"{len(back_lines)} data line(s) have a running time earlier than a line above them. "
            "Check whether two runs were joined into one file. All readings were kept; summaries "
            "use the readings sorted by time.",
            column="time_h", lines=back_lines[:_LIST_CAP], source=source,
        ))
    if recorded_start is not None:
        off = []
        for ln, t, clk in zip(lines, times, clocks, strict=True):
            if clk is None or not math.isfinite(t):
                continue
            clock_min = (clk - recorded_start).total_seconds() / 60
            run_min = (t - offset_h) * 60
            if abs(clock_min - run_min) > _CLOCK_TOLERANCE_MIN:
                off.append(ln)
        if off:
            found.append(finding(
                "clock_time_disagrees", "warning",
                f"On {len(off)} data line(s) the clock time minus the recorded start differs from the "
                f"running time by more than {_CLOCK_TOLERANCE_MIN} min. Check the start time and the "
                "instrument clock. All readings were kept; time_h follows the running time.",
                column="timestamp", lines=off[:_LIST_CAP], source=source,
            ))
    return found


def _value_findings(table: pd.DataFrame, channel: str, source: str | None) -> list[Finding]:
    found: list[Finding] = []
    missing = table.index[table["status"] == MISSING]
    if len(missing):
        sub = table.loc[missing]
        found.append(finding(
            "reading_missing", "warning",
            f"{len(missing)} {channel} reading(s) in {sub['well'].nunique()} well(s) are missing (N/A "
            "or empty). Check the sensor spots of those wells. They were kept with no value (never "
            "filled) and left out of the summaries.",
            column="value", rows=missing, where=_where(sub), source=source,
        ))
    bad = table.index[table["status"] == NOT_NUMERIC]
    if len(bad):
        sub = table.loc[bad]
        found.append(finding(
            "reading_not_numeric", "warning",
            f"{len(bad)} {channel} reading(s) are not numbers. Check those cells in the export. They "
            "were kept with the raw text in `raw` and no value, and left out of the summaries.",
            column="value", rows=bad, where=_where(sub), raw_values=sorted(set(sub["raw"]))[:10],
            source=source,
        ))
    ok = table["status"] == OK
    if channel == "pH":
        lo, hi = _PH_RANGE
        out = table.index[ok & ((table["value"] < lo) | (table["value"] > hi))]
        if len(out):
            found.append(finding(
                "ph_out_of_range", "warning",
                f"{len(out)} pH reading(s) are outside {lo:g}-{hi:g}. Check the sensor spot and the "
                "calibration of those wells. They were kept as read; nothing was changed.",
                column="value", rows=out, where=_where(table.loc[out]), source=source,
            ))
    else:
        neg = table.index[ok & (table["value"] < 0)]
        if len(neg):
            found.append(finding(
                "do_negative", "warning",
                f"{len(neg)} DO reading(s) are below zero, which usually means a sensor offset. Check "
                "the calibration of those wells. They were kept as read; nothing was changed.",
                column="value", rows=neg, where=_where(table.loc[neg]), source=source,
            ))
    return found


def _processing_findings(md: dict, source: str | None) -> list[Finding]:
    found: list[Finding] = []
    conds = md["conditions"]
    on = [k for k in ("Moving Average Smoothing", "Baseline Correction")
          if str(conds.get(k, {}).get("value", "")).strip().upper() == "ON"]
    if on:
        found.append(finding(
            "instrument_processing_on", "info",
            f"The export says {' and '.join(on)} was ON: the values are as processed by the "
            "instrument software, not the sensor readings as taken. Keep this in mind when "
            "comparing plates. The values were kept as exported; nothing was changed.",
            options=on, source=source,
        ))
    cal = str(conds.get("Sensor calibration", {}).get("value", ""))
    if "after start" in cal.lower():
        found.append(finding(
            "sensor_calibration_after_start", "info",
            f"The sensor calibration setting reads '{cal}': readings before the calibration may be "
            "off, and pH start is the first valid reading as exported. Check the early readings. "
            "Nothing was changed.",
            setting=cal, source=source,
        ))
    return found


def read_snest_csv(path: str | os.PathLike[str], **kwargs) -> SnestResult:
    """`parse_snest_text` on a file; the encoding and file name are added."""
    text, encoding = read_text(path)
    kwargs.setdefault("source", Path(path).name)
    result = parse_snest_text(text, **kwargs)
    return SnestResult(result.table, result.findings, {**result.meta, "encoding": encoding})


# ----------------------------------------------------------------------------
# Several files and the per-well summary
# ----------------------------------------------------------------------------


def _summary_columns(do_unit: str | None, threshold: float | None, min_points: int) -> dict:
    ok_do = "readings of the well with channel == 'DO' and status == 'ok'"
    ok_ph = "readings of the well with channel == 'pH' and status == 'ok'"
    floor = f"NaN if fewer than {min_points} such readings"
    return {
        "n_valid_do": {"formula": f"count of {ok_do}", "derived_from": ["table.value", "table.status"],
                       "unit": None, "role": None},
        "n_valid_ph": {"formula": f"count of {ok_ph}", "derived_from": ["table.value", "table.status"],
                       "unit": None, "role": None},
        "do_min": {"formula": f"min(value) over {ok_do}; {floor}",
                   "derived_from": ["table.value[DO]"], "unit": do_unit, "role": "outcome"},
        "do_final": {"formula": f"value at the largest time_h among {ok_do} (last in file order on "
                                f"a tie); {floor}",
                     "derived_from": ["table.value[DO]", "table.time_h"], "unit": do_unit,
                     "role": "outcome"},
        "do_time_below_threshold_h": {
            "formula": ("sum over adjacent readings (consecutive in time_h, both status 'ok') of the "
                        "time with DO strictly below the declared threshold; within a pair the "
                        "crossing time is found by straight-line interpolation between the two "
                        "readings. A pair with a missing or non-numeric reading is not counted (its "
                        "duration goes to do_time_unassessed_h); nothing before the first or after "
                        f"the last reading is counted. NaN if no threshold was declared; {floor}"),
            "derived_from": ["table.value[DO]", "table.time_h", "declared do_threshold"],
            "threshold": threshold, "unit": "h", "role": "outcome"},
        "do_time_unassessed_h": {
            "formula": ("sum of time_h steps between adjacent DO readings where one or both are not "
                        f"status 'ok' (time that could not be assessed); {floor}"),
            "derived_from": ["table.status[DO]", "table.time_h"], "unit": "h", "role": None},
        "ph_start": {"formula": f"value at the smallest time_h among {ok_ph}; {floor}",
                     "derived_from": ["table.value[pH]", "table.time_h"], "unit": "pH",
                     "role": "covariate"},
        "ph_final": {"formula": f"value at the largest time_h among {ok_ph}; {floor}",
                     "derived_from": ["table.value[pH]", "table.time_h"], "unit": "pH",
                     "role": "outcome"},
        "ph_drift": {"formula": "ph_final - ph_start", "derived_from": ["ph_final", "ph_start"],
                     "unit": "pH", "role": "outcome"},
        "ph_min": {"formula": f"min(value) over {ok_ph}; {floor}", "derived_from": ["table.value[pH]"],
                   "unit": "pH", "role": "outcome"},
        "ph_max": {"formula": f"max(value) over {ok_ph}; {floor}", "derived_from": ["table.value[pH]"],
                   "unit": "pH", "role": "outcome"},
    }


def time_below(times: Sequence[float], values: Sequence[float], valid: Sequence[bool],
               threshold: float) -> tuple[float, float]:
    """(hours strictly below `threshold`, hours not assessed) over a time-sorted series.

    Only pairs of adjacent readings that are both valid count; the crossing
    inside a pair is found by straight-line interpolation. Nothing outside the
    first and last reading is counted.
    """
    below = 0.0
    unassessed = 0.0
    for i in range(len(times) - 1):
        t0, t1 = times[i], times[i + 1]
        dt = t1 - t0
        if dt <= 0:
            continue
        if not (valid[i] and valid[i + 1]):
            unassessed += dt
            continue
        v0, v1 = values[i], values[i + 1]
        if v0 < threshold and v1 < threshold:
            below += dt
        elif v0 < threshold or v1 < threshold:
            # one reading below, one at/above: the part of the step below the line
            lo, hi = (v0, v1) if v0 < threshold else (v1, v0)
            below += dt * (threshold - lo) / (hi - lo)
    return below, unassessed


def _unassessed(times: Sequence[float], valid: Sequence[bool]) -> float:
    return sum(times[i + 1] - times[i] for i in range(len(times) - 1)
               if times[i + 1] > times[i] and not (valid[i] and valid[i + 1]))


def summarize(
    table: pd.DataFrame,
    *,
    do_threshold: float | None = None,
    min_points: int = MIN_VALID_POINTS,
    baseline_wells: dict[str, set[str]] | None = None,
    files_per_channel: dict[tuple[str, str], int] | None = None,
) -> tuple[pd.DataFrame, list[Finding], dict]:
    """Per-well summary of a long S.NEST table: (summary, findings, column formulas).

    A plate and channel read from more than one file (by `source`, or as
    counted in `files_per_channel` when the same file was given twice) is not
    summarized: its summaries are NaN with an error finding.
    """
    if min_points < 2:
        raise ValueError(f"min_points must be at least 2, got {min_points}")
    found: list[Finding] = []
    threshold_ok = do_threshold is not None and math.isfinite(float(do_threshold))
    units = sorted({str(u) for u in table.loc[table["channel"] == "DO", "unit"].dropna()})
    do_unit = units[0] if len(units) == 1 else None
    if do_threshold is None:
        found.append(finding(
            "do_threshold_not_declared", "error",
            "Time below a DO threshold was not computed: no threshold was declared, and none is "
            "assumed. Declare do_threshold in the DO unit of the export (the DO level that matters "
            "for your culture) to compute it. The column is NaN for every well; the other summaries "
            "were computed.",
            column="do_time_below_threshold_h",
        ))
    elif not threshold_ok:
        found.append(finding(
            "do_threshold_invalid", "error",
            f"The declared DO threshold {do_threshold!r} is not a finite number, so time below it was "
            "not computed. Declare a number in the DO unit of the export. The column is NaN for "
            "every well; the other summaries were computed.",
            column="do_time_below_threshold_h",
        ))
    if len(units) > 1:
        found.append(finding(
            "do_unit_differs", "error",
            f"DO readings carry different units {units}. Do not compare DO summaries across these "
            "files until the units agree. Summaries were computed per well as read; nothing was "
            "converted.",
            column="unit", units=units,
        ))

    keyed = table.assign(_well=table["well_canonical"].where(table["well_canonical"].notna(),
                                                             table["well"]))
    keyed["_plate"] = keyed["plate"].fillna("").astype(str)
    n_files: dict[tuple[str, str], int] = {}
    for (p, c), sub in keyed.groupby(["_plate", "channel"]):
        n_files[(str(p), str(c))] = int(sub["source"].nunique(dropna=False))
    for key, n in (files_per_channel or {}).items():
        n_files[key] = max(n, n_files.get(key, 0))
    two_files = {key for key, n in n_files.items() if n > 1}
    for plate, channel in sorted(two_files):
        rows = keyed.index[(keyed["_plate"] == plate) & (keyed["channel"] == channel)]
        found.append(finding(
            "duplicate_channel_file", "error",
            f"Plate '{plate}' has {n_files[(plate, channel)]} {channel} files. Give one file per "
            f"channel and plate. All readings were kept; the {channel} summaries of that plate were "
            "not computed (NaN).",
            rows=rows, plate=plate, channel=channel,
            sources=sorted({str(x) for x in keyed.loc[rows, "source"]}),
        ))
    short: list[str] = []
    rows_out = []
    for (plate_key, well_key), grp in keyed.groupby(["_plate", "_well"], sort=True):
        plate, well = str(plate_key), str(well_key)
        rec: dict = {c: math.nan for c in SUMMARY_COLUMNS}
        first = grp.iloc[0]
        rec.update({
            "plate": first["plate"], "well": first["well"], "well_canonical": first["well_canonical"],
            "group": first["group"], "note": first["note"],
            "baseline_reference_well": well in (baseline_wells or {}).get(plate, set()),
        })
        for channel in ("DO", "pH"):
            ch = grp[(grp["channel"] == channel) & (grp["status"] != NO_TIME)]
            ch = ch.sort_values("time_h", kind="stable")
            valid = (ch["status"] == OK).to_numpy()
            n_valid = int(valid.sum())
            rec["n_valid_do" if channel == "DO" else "n_valid_ph"] = n_valid
            if ch.empty or (plate, channel) in two_files:
                continue
            if n_valid < min_points:
                short.append(f"{plate}:{well}:{channel} ({n_valid})")
                continue
            vals = ch["value"].to_numpy(dtype=float)
            times = ch["time_h"].to_numpy(dtype=float).tolist()
            ok_vals = vals[valid]
            if channel == "DO":
                rec["do_min"] = float(ok_vals.min())
                rec["do_final"] = float(ok_vals[-1])
                if threshold_ok:
                    assert do_threshold is not None
                    rec["do_time_below_threshold_h"], rec["do_time_unassessed_h"] = time_below(
                        times, vals.tolist(), valid.tolist(), float(do_threshold))
                else:
                    rec["do_time_unassessed_h"] = _unassessed(times, valid.tolist())
            else:
                rec["ph_start"] = float(ok_vals[0])
                rec["ph_final"] = float(ok_vals[-1])
                rec["ph_drift"] = rec["ph_final"] - rec["ph_start"]
                rec["ph_min"] = float(ok_vals.min())
                rec["ph_max"] = float(ok_vals.max())
        rows_out.append(rec)
    summary = pd.DataFrame.from_records(rows_out, columns=SUMMARY_COLUMNS)
    if short:
        found.append(finding(
            "too_few_points", "warning",
            f"{len(short)} well channel(s) have fewer than {min_points} valid readings. Check the "
            "sensor spots of those wells. Their summaries are NaN (never extrapolated); the readings "
            "were kept in the table.",
            wells=short[:_LIST_CAP], min_points=min_points,
        ))
    declared = float(do_threshold) if threshold_ok and do_threshold is not None else None
    return summary, found, _summary_columns(do_unit, declared, min_points)


def combine(
    results: Iterable[SnestResult],
    *,
    do_threshold: float | None = None,
    min_points: int = MIN_VALID_POINTS,
) -> SnestResult:
    """Join single-file results (DO and pH of one or more plates) and summarize per well."""
    results = list(results)
    found: list[Finding] = []
    frames = [r.table for r in results if not r.table.empty]
    for r in results:
        found.extend(r.findings)
    table = pd.concat(frames, ignore_index=True) if frames else _empty_table()

    files = [{k: r.meta.get(k) for k in ("source", "plate", "channel", "unit", "unit_source",
                                         "plate_format", "time_basis", "start_time_recorded",
                                         "encoding")} for r in results]
    by_plate: dict[str, list[SnestResult]] = {}
    for r in results:
        if r.meta.get("channel"):
            by_plate.setdefault(str(r.meta.get("plate") or ""), []).append(r)
    plates_meta: dict[str, dict] = {}
    baseline: dict[str, set[str]] = {}
    for plate, rs in sorted(by_plate.items()):
        channels = [r.meta["channel"] for r in rs]
        for ch in ("DO", "pH"):
            if ch not in channels:
                found.append(finding(
                    "channel_not_loaded", "info",
                    f"No {ch} file was read for plate '{plate}'. Add its {ch} export if there is "
                    f"one. Its {ch} summaries are empty; nothing was changed.",
                    plate=plate, channel=ch,
                ))
        ref = [r.meta for r in rs]
        differs = sorted({k for m in ref[1:] for k in ("as_run", "start_time_recorded", "plate_format")
                          if m.get(k) != ref[0].get(k)})
        if differs:
            found.append(finding(
                "metadata_differs_between_files", "warning",
                f"The files of plate '{plate}' record different {', '.join(differs)}. Check that the "
                "DO and pH files are from the same run. All readings were kept.",
                plate=plate, fields=differs,
            ))
        first = ref[0]
        plates_meta[plate] = {
            "as_run": first.get("as_run", {}),
            "as_run_note": first.get("as_run_note"),
            "conditions": first.get("conditions", {}),
            "start_time_recorded": first.get("start_time_recorded"),
            "plate_format": first.get("plate_format"),
            "pauses": first.get("pauses", []),
        }
        refs = str(first.get("conditions", {}).get("Baseline Reference Wells", {}).get("value") or "")
        baseline[plate] = {w for w in (canonical_well(x) for x in re.split(r"[/;\s]+", refs)) if w}

    per_channel: dict[tuple[str, str], int] = {}
    for plate, rs in by_plate.items():
        for r in rs:
            per_channel[(plate, r.meta["channel"])] = per_channel.get((plate, r.meta["channel"]), 0) + 1
    summary, sum_findings, columns = summarize(
        table, do_threshold=do_threshold, min_points=min_points, baseline_wells=baseline,
        files_per_channel=per_channel)
    found.extend(sum_findings)

    roles: dict[str, str] = {"plate": "batch_group", "well": "id", "well_canonical": "id"}
    roles.update({c: spec["role"] for c, spec in columns.items() if spec["role"]})
    meta = {
        "parser": PARSER_VERSION,
        "files": files,
        "plates": plates_meta,
        "time_basis": sorted({str(r.meta.get("time_basis")) for r in results if r.meta.get("time_basis")}),
        "do_threshold": float(do_threshold) if do_threshold is not None and math.isfinite(
            float(do_threshold)) else None,
        "do_threshold_unit": columns["do_min"]["unit"],
        "min_valid_points": min_points,
        "summary_columns": columns,
        "role_suggestions": roles,
    }
    return SnestResult(table, tuple(found), meta, summary)


def read_snest(
    paths: Sequence[str | os.PathLike[str]],
    *,
    plate: str | None = None,
    do_unit: str | None = None,
    start: datetime | None = None,
    plate_format: int | None = None,
    do_threshold: float | None = None,
    min_points: int = MIN_VALID_POINTS,
) -> SnestResult:
    """Read S.NEST CSV exports (DO and pH, one or more plates) and summarize per well.

    `plate` (if given) is the plate id of every file; otherwise each file's
    `Experiment name` is its plate id. `do_unit` is the declared DO unit.
    `do_threshold` (in the DO unit) enables time below threshold.
    """
    results = [read_snest_csv(p, plate=plate, do_unit=do_unit, start=start, plate_format=plate_format)
               for p in paths]
    return combine(results, do_threshold=do_threshold, min_points=min_points)
