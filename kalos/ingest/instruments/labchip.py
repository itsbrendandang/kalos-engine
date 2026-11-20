"""LabChip peak tables: the product peak by a declared size window, per well.

Reads the "PeakTable" CSV that Revvity LabChip GX software exports for the
Protein Express assay (headers: Well Label, Sample Name, Type, Size [kDa],
Conc. (ng/ul), % Purity, Migration Time(sec) - Start/End, Corr. Area,
Date_Time, Plate File, Height). Two tables come back, plus findings
(`kalos.validation.report.Finding`, built by `kalos.ingest.findings.finding`):

- `table`: the long peak table, one row per exported peak, every row kept with
  a `status` and a `reason`. Ladder wells and ladder peaks (Type "L", well
  labels such as "Ladder01"), markers ("LM", "UM"), system peaks ("Sys",
  "Xsys") and wells the user declares as standards are excluded from the
  results, with the reason, and stay in this table. Peaks of any other
  non-blank type (e.g. "?") are not quantified by the software and are not
  used as the product peak.
- `results`: one row per sample well with the product concentration.

The product peak is chosen ONLY by a size window `(lo_kda, hi_kda)` (both
ends included) that the user declares. The window is part of the assay
method: a different window is a different method, so it is written on every
result row (`window_lo_kda`, `window_hi_kda`) and in `meta`. With no window,
no product value is produced (error finding `size_window_not_declared`).

Per well:

- no sample peak in the window: `product_conc` is NaN, `status`
  "no_peak_in_window", `non_detect` True. This is a non-detect (below what the
  assay saw), never a zero; declare the assay's detection limit when modelling.
- one peak: its concentration, % purity and size.
- more than one peak: the well is refused (NaN, "multiple_peaks_in_window",
  the sizes in the finding) unless the user declares `multi_peak_rule`:
  "sum" (concentrations and % purity added; no single size) or "largest" (the
  peak with the highest concentration; a tie is refused). The rule is stored
  on every result row and in `meta`.
- a sample peak with no readable size anywhere in the well: refused
  ("peak_size_missing"), because it could be in the window.
- a non-sample peak (e.g. Type "?") in the window and no sample peak:
  "unquantified_peak_in_window", NaN, not a non-detect.

Sample peaks just outside the window are reported as info
(`peak_near_window`): within `near_margin_frac` of the bound (default 0.10, so
lo x 0.9 to lo and hi to hi x 1.1). The margin is an analyst's tolerance for
"close enough to look at", not an instrument specification, and it never
changes a result.

Units: `conc` keeps the exported number in the header's unit (`conc_unit`).
`product_conc_g_per_l` = ng/uL x 1e-3 (1 ng/uL = 1 mg/L), with the factor in
the `conversion` column and in `meta["conversion"]`. Any header unit other
than ng/ul (or ng/uL with a micro sign) is not guessed: the g/L column stays
empty and an error finding says so. % Purity is not converted. Date_Time and
Plate File are kept as text (provenance), never parsed.

Role suggestions (setup-suggestion vocabulary): the product concentration in
g/L and the product % purity are `outcome`; the raw ng/uL column is the same
measurement, so it is `derived_from_target` (never an input); sample name and
well are `id`. Size, migration time, area and height get no suggestion (a
peak's area is what the concentration is computed from, so it must never be a
factor). Nothing is ever suggested as a `setting`.

Results carry `well` (as exported), `well_canonical` ("A01" -> "A1") and
`sample_name`, so they join to a plate map by position (plate parser's
`join_by_position(..., sample_column="sample_name")`), which also checks the
sample names against the map.

Idea credit: the lab's LabChip analyses (ladder rows left out, product peak by
size, ng/uL to g/L) from Bioqore's ad hoc analyses; see NOTICE. Re-implemented;
no code copied.
"""
from __future__ import annotations

import csv
import io
import math
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from kalos.ingest.findings import Finding, Severity, finding

PARSER_VERSION = "labchip/1"

NG_PER_UL_TO_G_PER_L = 1e-3
CONVERSION = "g/L = ng/uL x 1e-3 (1 ng/uL = 1 mg/L)"
MULTI_PEAK_RULES = ("sum", "largest")
DEFAULT_NEAR_MARGIN_FRAC = 0.10

_LIST_CAP = 50
_WELL_RE = re.compile(r"^\s*([A-Pa-p])\s*0*([0-9]{1,2})\s*$")
_LADDER_WELL_RE = re.compile(r"^\s*ladder", re.IGNORECASE)
_STANDARD_NAME_RE = re.compile(
    r"\b(std|standard|bsa)\b|\d+(\.\d+)?\s*(ng|ug|\u00b5g|\u03bcg|mg)\s*/\s*(ul|\u00b5l|\u03bcl|ml)\b",
    re.IGNORECASE,
)

MARKER_TYPES = frozenset({"LM", "UM"})
LADDER_TYPES = frozenset({"L"})
SYSTEM_TYPES = frozenset({"SYS", "XSYS"})

# canonical column -> pattern on the normalized header (lower case, single spaces)
_HEADER_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("well", re.compile(r"^well label$")),
    ("sample_name", re.compile(r"^sample name$")),
    ("peak_type", re.compile(r"^type$")),
    ("size", re.compile(r"^size ?\[ ?(?P<unit>[^\]]*?) ?\]$")),
    ("conc", re.compile(r"^conc\.? ?\( ?(?P<unit>[^)]*?) ?\)$")),
    ("purity_pct", re.compile(r"^% ?purity$")),
    ("migration_start_s", re.compile(r"^migration time ?\( ?sec ?\) ?- ?start$")),
    ("migration_end_s", re.compile(r"^migration time ?\( ?sec ?\) ?- ?end$")),
    ("corr_area", re.compile(r"^corr\.? area$")),
    ("date_time", re.compile(r"^date_time$")),
    ("plate_file", re.compile(r"^plate file$")),
    ("height", re.compile(r"^height$")),
)
_REQUIRED = ("well", "sample_name", "peak_type", "size", "conc")
_NUMERIC = ("purity_pct", "migration_start_s", "migration_end_s", "corr_area", "height")

# long-table statuses that take a row's well out of the results
_EXCLUDED = {
    "ladder": "ladder well or ladder peak; used by the instrument for sizing",
    "marker": "marker peak (LM/UM); used by the instrument for alignment",
    "system_peak": "system peak (Sys/Xsys), not a protein from the sample",
    "standard": "well declared as a standard",
    "unassigned_type": "peak type is not a quantified sample peak",
}


@dataclass(frozen=True)
class LabChipResult:
    """The long peak table, the per-well results, what was found, and the method."""

    table: pd.DataFrame
    findings: tuple[Finding, ...]
    meta: dict = field(default_factory=dict)
    results: pd.DataFrame = field(default_factory=pd.DataFrame)

    def to_dict(self) -> dict:
        """JSON-safe: NaN becomes None."""
        return {
            "table": _records(self.table),
            "results": _records(self.results),
            "findings": [f.to_dict() for f in self.findings],
            "meta": dict(self.meta),
        }


def _records(frame: pd.DataFrame) -> list[dict]:
    out = []
    for rec in frame.to_dict(orient="records"):
        out.append({str(k): (None if isinstance(v, float) and math.isnan(v) else v)
                    for k, v in rec.items()})
    return out


def canonical_well(label: object) -> str | None:
    """"A01" -> "A1"; None if the label is not a well of a plate up to 384 wells."""
    if label is None:
        return None
    m = _WELL_RE.match(str(label))
    if not m:
        return None
    number = int(m.group(2))
    return f"{m.group(1).upper()}{number}" if 1 <= number <= 24 else None


def _norm_header(h: str) -> str:
    return " ".join(str(h).replace("\ufeff", "").strip().lower().split())


def _unit_is_ng_per_ul(unit: str | None) -> bool:
    if unit is None:
        return False
    u = unit.lower().replace("\u00b5", "u").replace("\u03bc", "u").replace(" ", "")
    return u == "ng/ul"


def _number(raw: str) -> float | None:
    try:
        value = float(raw.strip())
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _text(value: object) -> str:
    return " ".join(str(value).split()) if value is not None else ""


def read_text(path: str | os.PathLike[str]) -> tuple[str, str]:
    """Decode an export: UTF-16 with BOM, else UTF-8 (BOM optional), else Latin-1."""
    data = Path(path).read_bytes()
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return data.decode("utf-16"), "utf-16"
    try:
        return data.decode("utf-8-sig"), "utf-8"
    except UnicodeDecodeError:
        return data.decode("latin-1"), "latin-1"


def read_peak_table(path: str | os.PathLike[str], **kwargs) -> LabChipResult:
    """`parse_peak_table_text` on a file; the encoding used is added to `meta`."""
    text, encoding = read_text(path)
    res = parse_peak_table_text(text, **kwargs)
    return LabChipResult(res.table, res.findings, {**res.meta, "encoding": encoding}, res.results)


_RESULT_COLUMNS = [
    "plate_file", "well", "well_canonical", "sample_name", "product_conc", "conc_unit",
    "product_conc_g_per_l", "conversion", "product_purity_pct", "product_size_kda",
    "n_peaks_in_window", "sizes_in_window_kda", "rule_applied", "non_detect", "status",
    "window_lo_kda", "window_hi_kda", "multi_peak_rule", "date_time", "peak_rows",
]


def _window_problem(lo_kda: float | None, hi_kda: float | None) -> str | None:
    if lo_kda is None or hi_kda is None:
        return "missing"
    try:
        lo, hi = float(lo_kda), float(hi_kda)
    except (TypeError, ValueError):
        return "invalid"
    if not (math.isfinite(lo) and math.isfinite(hi)) or lo < 0 or lo >= hi:
        return "invalid"
    return None


def parse_peak_table_text(
    text: str,
    *,
    lo_kda: float | None = None,
    hi_kda: float | None = None,
    multi_peak_rule: str | None = None,
    standards: Sequence[str] = (),
    near_margin_frac: float = DEFAULT_NEAR_MARGIN_FRAC,
) -> LabChipResult:
    """Parse a LabChip PeakTable export; see the module docstring for the rules.

    `lo_kda`/`hi_kda`: the product size window in kDa (both required).
    `multi_peak_rule`: None (refuse wells with several peaks in the window),
    "sum" or "largest". `standards`: sample names or well labels of standard
    wells, excluded from the results. `near_margin_frac`: how far outside the
    window (as a fraction of the bound) a peak is reported as near it.
    """
    found: list[Finding] = []
    lines = text.splitlines()
    header_line = next((ln for ln in lines if ln.strip()), "")
    delim = "\t" if "\t" in header_line else (";" if header_line.count(";") > header_line.count(",")
                                                  else ",")
    reader = csv.reader(io.StringIO(text), delimiter=delim)
    rows: list[tuple[int, list[str]]] = []
    header: list[str] | None = None
    for cells in reader:
        if not any(c.strip() for c in cells):
            continue
        if header is None:
            header = [c.replace("\ufeff", "").strip() for c in cells]
        else:
            rows.append((reader.line_num, cells))
    header = header or []

    # -- columns -------------------------------------------------------------
    column_map: dict[str, str] = {}
    units: dict[str, str] = {}
    for h in header:
        norm = _norm_header(h)
        for name, pat in _HEADER_PATTERNS:
            m = pat.match(norm)
            if m and name not in column_map.values():
                column_map[h] = name
                if "unit" in pat.groupindex:
                    # unit as written in the header, case kept
                    raw = re.search(r"[\[(]\s*([^\])]*?)\s*[\])]\s*$", h)
                    units[name] = raw.group(1) if raw else m.group("unit")
                break
    missing = [c for c in _REQUIRED if c not in column_map.values()]
    window_issue = _window_problem(lo_kda, hi_kda)
    window = (float(lo_kda), float(hi_kda)) if window_issue is None else None  # type: ignore[arg-type]
    rule_ok = multi_peak_rule is None or multi_peak_rule in MULTI_PEAK_RULES
    margin_ok = isinstance(near_margin_frac, (int, float)) and 0 <= near_margin_frac < 1
    std_set = {_text(s) for s in standards}
    meta: dict = {
        "parser": PARSER_VERSION,
        "delimiter": delim,
        "column_map": dict(column_map),
        "size_unit": units.get("size"),
        "conc_unit": units.get("conc"),
        "size_window_kda": None if window is None else {"lo": window[0], "hi": window[1]},
        "window_includes_bounds": True,
        "multi_peak_rule": multi_peak_rule if rule_ok else None,
        "near_margin_frac": float(near_margin_frac) if margin_ok else None,
        "standards": sorted(std_set),
        "conversion": None,
        "role_suggestions": {},
    }

    if not header or missing:
        found.append(finding(
            "required_column_missing", "error",
            f"This file is missing the column(s) {missing} that a LabChip peak table has, so it "
            "was not read as one. Check that the export is the Peak Table (Well Label, Sample Name, "
            "Type, Size [kDa], Conc. (ng/ul), ...) and upload it again. The rows are kept as text; "
            "nothing was changed.",
            missing=missing, headers=header[:_LIST_CAP],
        ))
        unread = pd.DataFrame([dict(zip(header, c, strict=False)) for _, c in rows], columns=header)
        unread["status"] = "not_read"
        unread["source_line"] = [ln for ln, _ in rows]
        return LabChipResult(unread, tuple(found), meta, pd.DataFrame(columns=_RESULT_COLUMNS))

    # -- long table ----------------------------------------------------------
    by_name = {v: k for k, v in column_map.items()}
    extra = [h for h in header if h not in column_map]
    records: list[dict] = []
    ragged: list[int] = []
    for line_no, cells in rows:
        if len(cells) != len(header):
            ragged.append(len(records))
        cells = (cells + [""] * len(header))[: len(header)]
        cell = dict(zip(header, (c.strip() for c in cells), strict=True))
        val = {name: cell[h] for name, h in by_name.items()}
        well = val["well"]
        size_raw, conc_raw = val["size"], val["conc"]
        rec: dict = {
            "well": well,
            "well_canonical": canonical_well(well),
            "sample_name": _text(val["sample_name"]),
            "peak_type": val["peak_type"],
            "size_kda": _number(size_raw) if size_raw else None,
            "size_raw": size_raw,
            "conc": _number(conc_raw) if conc_raw else None,
            "conc_raw": conc_raw,
            "conc_unit": units.get("conc"),
        }
        for name in _NUMERIC:
            raw_v = val.get(name, "")
            rec[name] = _number(raw_v) if raw_v else None
        rec["date_time"] = val.get("date_time", "")
        rec["plate_file"] = val.get("plate_file", "")
        for h in extra:
            rec[h] = cell[h]
        rec["source_line"] = line_no
        records.append(rec)

    base_cols = ["well", "well_canonical", "sample_name", "peak_type", "size_kda", "size_raw",
                 "conc", "conc_raw", "conc_unit", *_NUMERIC, "date_time", "plate_file", *extra,
                 "source_line"]
    t = pd.DataFrame.from_records(records, columns=base_cols)
    for c in ("size_kda", "conc", *_NUMERIC):
        t[c] = pd.to_numeric(t[c], errors="coerce").astype(float)

    if not len(t):
        found.append(finding(
            "no_peaks", "error",
            "The peak table has a header but no peak rows. Check that the export finished and that "
            "the right plate was exported. Nothing was changed.",
        ))

    # -- classify rows -------------------------------------------------------
    ptype = t["peak_type"].str.strip().str.upper()
    ladder_well = t["well"].map(lambda w: bool(_LADDER_WELL_RE.match(str(w))))
    std_well = (t["sample_name"].isin(std_set) | t["well_canonical"].isin(
        {canonical_well(s) for s in std_set} - {None}) | t["well"].isin(std_set)) & ~ladder_well
    status = pd.Series("ok", index=t.index, dtype=object)
    status[ptype.isin(SYSTEM_TYPES)] = "system_peak"
    status[ptype.isin(MARKER_TYPES)] = "marker"
    unassigned = (ptype != "") & ~ptype.isin(SYSTEM_TYPES | MARKER_TYPES | LADDER_TYPES)
    status[unassigned] = "unassigned_type"
    status[std_well] = "standard"
    status[ptype.isin(LADDER_TYPES) | ladder_well] = "ladder"
    bad_label = (status != "ladder") & t["well_canonical"].isna()
    status[bad_label] = "well_label_invalid"
    sample = status == "ok"
    size_blank = sample & (t["size_raw"] == "")
    size_text = sample & (t["size_raw"] != "") & t["size_kda"].isna()
    conc_blank = sample & (t["conc_raw"] == "")
    conc_text = sample & (t["conc_raw"] != "") & t["conc"].isna()
    status[conc_text] = "conc_not_numeric"
    status[conc_blank] = "conc_missing"
    status[size_text] = "size_not_numeric"
    status[size_blank] = "size_missing"
    t["status"] = status
    reasons = {
        **_EXCLUDED,
        "ok": "",
        "well_label_invalid": "well label is not a plate well; cannot be placed",
        "size_missing": "size cell is blank",
        "size_not_numeric": "size cell is not a number",
        "conc_missing": "concentration cell is blank",
        "conc_not_numeric": "concentration cell is not a number",
    }
    t["reason"] = [reasons[s] if s != "unassigned_type" else f"peak type '{p}' is not a quantified "
                   "sample peak; not used as the product" for s, p in zip(t["status"], t["peak_type"],
                                                                         strict=True)]
    is_sample_peak = sample  # Type blank, in a sample well (numbers may still be missing)

    size_unit_bad = (units.get("size") or "").lower() != "kda"
    if window is None or size_unit_bad:
        t["in_window"] = None
    else:
        t["in_window"] = [None if math.isnan(s) else bool(window[0] <= s <= window[1])
                          for s in t["size_kda"]]

    # -- file-level findings -------------------------------------------------
    if ragged:
        found.append(finding(
            "row_length_mismatch", "warning",
            f"{len(ragged)} row(s) have a different number of cells than the header. Missing cells "
            "were read as blank and extra cells were not read. Check those lines in the export. The "
            "rows are kept; nothing was changed.",
            rows=ragged, lines=[int(x) for x in t.loc[ragged, "source_line"]][:_LIST_CAP],
        ))
    if size_unit_bad:
        found.append(finding(
            "size_unit_not_kda", "error",
            f"The size column is in '{units.get('size')}', not kDa, so the size window (in kDa) "
            "cannot be applied and no product peak was chosen. Export sizes in kDa. All rows are "
            "kept; nothing was changed.",
            column=by_name["size"], unit=units.get("size"),
        ))
    conversion_ok = _unit_is_ng_per_ul(units.get("conc"))
    if conversion_ok:
        meta["conversion"] = {
            "from_column": "product_conc", "from_unit": units.get("conc"),
            "to_column": "product_conc_g_per_l", "to_unit": "g/L",
            "factor": NG_PER_UL_TO_G_PER_L, "formula": CONVERSION,
        }
    else:
        found.append(finding(
            "conc_unit_not_ng_per_ul", "error",
            f"The concentration column is in '{units.get('conc')}', not ng/ul, so no g/L value was "
            "computed (the unit is not guessed). Export Conc. in ng/ul, or convert by hand with a "
            "stated factor. The numbers are kept as exported in `product_conc` with `conc_unit`; "
            "nothing was changed.", column=by_name["conc"], unit=units.get("conc"),
        ))
    if window_issue == "missing":
        found.append(finding(
            "size_window_not_declared", "error",
            "No product size window was declared, so no product peak was chosen and no product "
            "values were produced. Declare the window in kDa (lo_kda, hi_kda) that brackets the "
            "product band; it is stored with the results as part of the assay method. All peaks "
            "are kept; nothing was changed.",
        ))
    elif window_issue == "invalid":
        found.append(finding(
            "size_window_invalid", "error",
            f"The size window ({lo_kda!r}, {hi_kda!r}) cannot be used: both ends must be numbers "
            "in kDa, lo at least 0 and lo below hi. Declare a valid window. No product values were "
            "produced; all peaks are kept and nothing was changed.", lo_kda=_jsonable(lo_kda), hi_kda=_jsonable(hi_kda),
        ))
    if not rule_ok:
        found.append(finding(
            "multi_peak_rule_invalid", "error",
            f"multi_peak_rule {multi_peak_rule!r} is not 'sum', 'largest' or unset, so it was "
            "ignored and wells with several peaks in the window are refused. Declare 'sum' or "
            "'largest', or leave it unset. Nothing was changed.", rule=str(multi_peak_rule),
        ))
    if not margin_ok:
        found.append(finding(
            "near_margin_invalid", "warning",
            f"near_margin_frac {near_margin_frac!r} must be a number from 0 up to (not including) "
            "1, so peaks near the window were not reported. Results are not affected; nothing was "
            "changed.",
        ))
    if bad_label.any():
        bad_idx = t.index[bad_label]
        found.append(finding(
            "well_label_invalid", "error",
            f"{len(bad_idx)} row(s) have a well label that is neither a plate well (A1 to P24) nor a "
            "ladder well, so they cannot be placed on the plate and are left out of the results. "
            "Check the labels in the instrument's plate setup. The rows are kept in the peak table; "
            "nothing was changed.", column=by_name["well"], rows=bad_idx,
            labels=sorted(set(t.loc[bad_idx, "well"]))[:_LIST_CAP],
        ))

    excluded = t["status"].isin(list(_EXCLUDED))
    if excluded.any():
        counts = {k: int(v) for k, v in t.loc[excluded, "status"].value_counts().sort_index().items()}
        found.append(finding(
            "rows_excluded", "info",
            f"{int(excluded.sum())} peak row(s) are not sample peaks ({counts}) and are not used for "
            "the product. They are kept in the peak table with a status and reason; nothing was "
            "changed.",
            column="status", rows=t.index[excluded], counts=counts,
        ))
        blank_excl = excluded & ((t["size_raw"] == "") | (t["conc_raw"] == "")
                                 | t["size_kda"].isna() | t["conc"].isna())
        if blank_excl.any():
            found.append(finding(
                "blank_cells_in_excluded_rows", "info",
                f"{int(blank_excl.sum())} excluded row(s) (ladder, marker, system, standard or "
                "unassigned peaks) have a blank or non-numeric size or concentration. This is normal "
                "for peaks the software does not quantify. They are kept as exported; nothing was "
                "filled.",
                rows=t.index[blank_excl],
                lines=[int(x) for x in t.loc[blank_excl, "source_line"]][:_LIST_CAP],
            ))
    for col, blank, text_bad, check in (
        ("size", size_blank, size_text, "size_not_numeric"),
        ("conc", conc_blank, conc_text, "conc_not_numeric"),
    ):
        bad = blank | text_bad
        if bad.any():
            sub = t.loc[bad]
            where = [f"{w} line {int(ln)}: {r!r}" for w, ln, r in
                     zip(sub["well"], sub["source_line"], sub[f"{col}_raw"], strict=True)]
            found.append(finding(
                check, "warning",
                f"{int(bad.sum())} sample peak(s) have a blank ({int(blank.sum())}) or non-numeric "
                f"({int(text_bad.sum())}) {'size' if col == 'size' else 'concentration'} cell. A well "
                "whose product depends on such a peak is refused. Check those peaks in the instrument "
                f"software and re-export. The cells are kept as exported (text in `{col}_raw`); "
                "nothing was filled.",
                column=by_name[col], rows=sub.index, where=where[:_LIST_CAP],
            ))
    purity_bad = is_sample_peak & t["purity_pct"].isna()
    if purity_bad.any():
        found.append(finding(
            "purity_not_numeric", "warning",
            f"{int(purity_bad.sum())} sample peak(s) have a blank or non-numeric % Purity. Check "
            "those peaks in the instrument software. The cells are kept empty; nothing was filled.", column=by_name.get("purity_pct"), rows=t.index[purity_bad],
        ))
    key_ok = t["well"] != ""
    dup = key_ok & t["size_kda"].notna() & t.duplicated(["plate_file", "well", "size_kda"], keep=False)
    if dup.any():
        sub = t.loc[dup]
        pairs = sorted({f"{w}:{s:g} kDa" for w, s in zip(sub["well"], sub["size_kda"], strict=True)})
        found.append(finding(
            "duplicate_peak", "warning",
            f"{len(pairs)} (well, size) pair(s) appear more than once. Check for a doubled or "
            "pasted export. A duplicate inside the product window refuses that well. All rows are "
            "kept; nothing was changed.",
            rows=sub.index, pairs=pairs[:_LIST_CAP],
        ))
    plate_files = list(dict.fromkeys(p for p in t["plate_file"] if p))
    if len(plate_files) > 1:
        found.append(finding(
            "multiple_plate_files", "info",
            f"The table holds peaks from {len(plate_files)} plate files. Wells are kept apart per "
            "plate file; join each plate's results to its plate map separately. Nothing was "
            "changed.",
            n_plate_files=len(plate_files),
        ))
    found.extend(_standard_findings(t, std_set, ladder_well))

    # -- per-well results ----------------------------------------------------
    can_choose = window is not None and not size_unit_bad
    sample_wells = t[~t["status"].isin(["ladder", "standard", "well_label_invalid"])]
    results: list[dict] = []
    groups: dict[tuple[str, str], list[int]] = {}
    for i, pf, w in zip(sample_wells.index, sample_wells["plate_file"], sample_wells["well"],
                        strict=True):
        groups.setdefault((pf, w), []).append(int(i))
    no_marker: list[str] = []
    name_varies: list[str] = []
    for (pf, w), idx in groups.items():
        g = t.loc[idx]
        names = list(dict.fromkeys(g["sample_name"]))
        if len(names) > 1:
            name_varies.append(f"{w}: {names}")
        if not g["peak_type"].str.strip().str.upper().isin(["LM"]).any():
            no_marker.append(w)
        if window is not None and can_choose:
            results.append(_well_result(t, g, pf, w, names[0], window, multi_peak_rule if rule_ok
                                        else None, conversion_ok, units.get("conc")))
    res = pd.DataFrame.from_records(results, columns=_RESULT_COLUMNS)
    if len(res):
        res["product_conc"] = res["product_conc"].astype(float)
        res["product_conc_g_per_l"] = res["product_conc_g_per_l"].astype(float)
    if no_marker:
        found.append(finding(
            "lower_marker_missing", "warning",
            f"{len(no_marker)} sample well(s) have no lower marker (LM) peak, so sizes in those "
            "wells may be off and a non-detect there may be a failed injection. Check those wells' "
            "electropherograms and re-run them if the marker is missing. The rows are kept; "
            "nothing was changed.",
            wells=no_marker[:_LIST_CAP],
        ))
    if name_varies:
        found.append(finding(
            "sample_name_varies_in_well", "warning",
            f"{len(name_varies)} well(s) have more than one sample name across their peaks; the "
            "first name is used in the results. Check the sample names in the export. All rows are "
            "kept; nothing was changed.", wells=name_varies[:_LIST_CAP],
        ))
    if window is not None and can_choose:
        found.extend(_result_findings(t, res, window, near_margin_frac if margin_ok else None,
                                      is_sample_peak, multi_peak_rule if rule_ok else None))

    t = t[[*base_cols[:-1], "status", "reason", "in_window", "source_line"]]
    meta["role_suggestions"] = _roles(conversion_ok)
    meta["n_peaks"] = len(t)
    meta["n_result_wells"] = len(res)
    meta["plate_files"] = plate_files[:_LIST_CAP]
    return LabChipResult(t, tuple(found), meta, res)


def _jsonable(value: object) -> object:
    if isinstance(value, (int, float, str)) or value is None:
        return None if isinstance(value, float) and not math.isfinite(value) else value
    return repr(value)


def _roles(conversion_ok: bool) -> dict[str, str]:
    roles = {"sample_name": "id", "well": "id", "well_canonical": "id",
             "product_purity_pct": "outcome"}
    if conversion_ok:
        roles["product_conc_g_per_l"] = "outcome"
        roles["product_conc"] = "derived_from_target"
    else:
        roles["product_conc"] = "outcome"
    return roles


def _well_result(t: pd.DataFrame, g: pd.DataFrame, pf: str, w: str, name: str,
                 window: tuple[float, float], rule: str | None, conversion_ok: bool,
                 unit: str | None) -> dict:
    lo, hi = window
    peaks = g[g["status"].isin(["ok", "conc_missing", "conc_not_numeric"])]
    in_win = peaks[(peaks["size_kda"] >= lo) & (peaks["size_kda"] <= hi)]
    other = g[g["status"] == "unassigned_type"]
    other_in = other[(other["size_kda"] >= lo) & (other["size_kda"] <= hi)]
    sizes = [float(s) for s in in_win["size_kda"]]
    out: dict = {
        "plate_file": pf, "well": w, "well_canonical": canonical_well(w), "sample_name": name,
        "product_conc": math.nan, "conc_unit": unit, "product_conc_g_per_l": math.nan,
        "conversion": CONVERSION if conversion_ok else None, "product_purity_pct": math.nan,
        "product_size_kda": math.nan, "n_peaks_in_window": len(in_win),
        "sizes_in_window_kda": sizes, "rule_applied": None, "non_detect": False,
        "status": "ok", "window_lo_kda": lo, "window_hi_kda": hi, "multi_peak_rule": rule,
        "date_time": "; ".join(dict.fromkeys(d for d in g["date_time"] if d)),
        "peak_rows": [int(i) for i in in_win.index],
    }
    if g["status"].isin(["size_missing", "size_not_numeric"]).any():
        out["status"] = "peak_size_missing"
        return out
    if in_win.empty:
        if not other_in.empty:
            out["status"] = "unquantified_peak_in_window"
            out["peak_rows"] = [int(i) for i in other_in.index]
        else:
            out["status"] = "no_peak_in_window"
            out["non_detect"] = True
        return out
    if in_win["size_kda"].duplicated().any():
        out["status"] = "duplicate_peak_in_window"
        return out
    if len(in_win) > 1 and rule is None:
        out["status"] = "multiple_peaks_in_window"
        return out
    if in_win["conc"].isna().any():
        out["status"] = "product_conc_missing"
        return out
    if len(in_win) == 1:
        chosen, applied = in_win, "single_peak"
    elif rule == "sum":
        chosen, applied = in_win, "sum"
    else:
        top = in_win["conc"].max()
        chosen, applied = in_win[in_win["conc"] == top], "largest"
        if len(chosen) > 1:
            out["status"] = "largest_peak_tied"
            return out
    conc = float(chosen["conc"].sum())
    out.update({
        "product_conc": conc,
        "product_conc_g_per_l": conc * NG_PER_UL_TO_G_PER_L if conversion_ok else math.nan,
        "product_purity_pct": float(chosen["purity_pct"].sum(min_count=len(chosen))),
        "product_size_kda": float(chosen["size_kda"].iloc[0]) if len(chosen) == 1 else math.nan,
        "rule_applied": applied,
        "peak_rows": [int(i) for i in chosen.index],
    })
    return out


def _result_findings(t: pd.DataFrame, res: pd.DataFrame, window: tuple[float, float],
                     margin: float | None, is_sample_peak: pd.Series,
                     rule: str | None) -> list[Finding]:
    """Per-well findings. `rows` are the long-table rows of those wells' peaks;
    `detail["result_rows"]` are the rows of the results table."""
    lo, hi = window
    found: list[Finding] = []
    in_results = ~t["status"].isin(["ladder", "standard", "well_label_invalid"])
    keys = pd.Series(list(zip(t["plate_file"], t["well"], strict=True)), index=t.index)

    def well_finding(status: str, check: str, severity: Severity, message: str,
                     with_sizes: bool = False) -> None:
        hit = res[res["status"] == status] if status else res[res["rule_applied"] == rule]
        if hit.empty:
            return
        wanted = set(zip(hit["plate_file"], hit["well"], strict=True))
        rows = t.index[in_results & keys.isin(wanted)]
        found.append(finding(
            check, severity, f"{len(hit)} well(s) {message}", column="product_conc", rows=rows,
            result_rows=[int(i) for i in hit.index][:_LIST_CAP],
            wells=({w: s for w, s in zip(hit["well"], hit["sizes_in_window_kda"], strict=True)}
                   if with_sizes else list(hit["well"])[:_LIST_CAP]),
        ))

    span = f"between {lo:g} and {hi:g} kDa"
    well_finding(
        "no_peak_in_window", "no_peak_in_window", "warning",
        f"have no sample peak {span}. They are kept as non-detects (`product_conc` empty, "
        "`non_detect` True), not as 0. Check that the window brackets the product, and declare the "
        "assay's detection limit before modelling them. Nothing was filled.",
    )
    well_finding(
        "multiple_peaks_in_window", "multiple_peaks_in_window", "error",
        f"have more than one sample peak {span}, so no peak was picked and their product value is "
        "empty. Check the electropherograms, then narrow the window or declare "
        "multi_peak_rule='sum' (all of them are product) or 'largest'. All peaks are kept; nothing "
        "was changed.", with_sizes=True,
    )
    if rule is not None:
        well_finding(
            "", "multi_peak_rule_applied", "info",
            f"had more than one peak {span}; the declared rule '{rule}' was applied (stored in "
            "`multi_peak_rule` and meta). Check that the rule fits these wells. All peaks are kept.",
            with_sizes=True,
        )
    refusals: tuple[tuple[str, Severity, str], ...] = (
        ("duplicate_peak_in_window", "error",
         "have the same peak size listed twice in the window, so no product value was computed "
         "(a doubled row would double it). Check the export for a pasted or doubled row. All rows "
         "are kept; nothing was changed."),
        ("product_conc_missing", "error",
         "have a product peak whose concentration cell is blank or not a number, so the product "
         "value is empty. Check the peak integration in the instrument software and re-export. "
         "Nothing was filled."),
        ("peak_size_missing", "error",
         "have a sample peak with no readable size, which could be the product, so the well is "
         "refused and its value is empty. Check those peaks in the instrument software and "
         "re-export with sizes. Nothing was filled."),
        ("largest_peak_tied", "error",
         "have two peaks with the same highest concentration in the window, so 'largest' cannot "
         "choose and the value is empty. Narrow the window or use 'sum'. Nothing was changed."),
        ("unquantified_peak_in_window", "warning",
         "have no quantified sample peak in the window but a peak of another type (e.g. '?') "
         "there. They are not non-detects; the value is empty. Check the peak assignment in the "
         "instrument software. Nothing was filled."),
    )
    for status, sev, what in refusals:
        well_finding(status, status, sev, what)
    if margin is not None:
        s = t["size_kda"]
        near = is_sample_peak & (((s >= lo * (1 - margin)) & (s < lo))
                                 | ((s > hi) & (s <= hi * (1 + margin))))
        if near.any():
            sub = t.loc[near]
            found.append(finding(
                "peak_near_window", "info",
                f"{int(near.sum())} sample peak(s) lie just outside the window (within "
                f"{margin:.0%} of a bound: {lo * (1 - margin):g}-{lo:g} or {hi:g}-"
                f"{hi * (1 + margin):g} kDa) and were not used. If one is the product, widen the "
                "window and record it as a method change. Nothing was changed.",
                column="size_kda", rows=sub.index,
                peaks=[f"{w}:{x:g} kDa" for w, x in zip(sub["well"], sub["size_kda"],
                                                        strict=True)][:_LIST_CAP],
            ))
    return found


def _standard_findings(t: pd.DataFrame, std_set: set[str],
                       ladder_well: pd.Series) -> list[Finding]:
    found: list[Finding] = []
    if std_set:
        seen = set(t["sample_name"]) | set(t["well"]) | set(t["well_canonical"].dropna())
        unmatched = sorted(s for s in std_set
                           if s not in seen and canonical_well(s) not in seen)
        if unmatched:
            found.append(finding(
                "standard_not_found", "warning",
                f"Declared standard(s) {unmatched} match no sample name or well in the file, so "
                "nothing was excluded for them. Check the spelling. Nothing was changed.",
                standards=unmatched,
            ))
    looks = (~ladder_well) & (t["status"] != "standard") & t["sample_name"].map(
        lambda n: bool(_STANDARD_NAME_RE.search(n)))
    if looks.any():
        wells = list(dict.fromkeys(f"{w} ({n})" for w, n in zip(t.loc[looks, "well"],
                                                                  t.loc[looks, "sample_name"],
                                                                  strict=True)))
        found.append(finding(
            "possible_standard_not_declared", "warning",
            f"{len(wells)} well(s) have a sample name that looks like a standard (a concentration, "
            "'std', 'standard' or 'BSA'). They are treated as samples. If they are standards, "
            "declare them (standards=[...]) so they leave the results. Nothing was changed.",
            column="sample_name", rows=t.index[looks], wells=wells[:_LIST_CAP],
        ))
    return found
