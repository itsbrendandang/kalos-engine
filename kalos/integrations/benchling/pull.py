"""Pull runs from Benchling and map them into a Kalos run sheet.

Two sources:

- **Bioprocess study (v3).** A study's procedure runs, their conditions and
  condition replicates, the method execution instances of each replicate, and the
  planned and measured parameter values recorded on those instances. Outcomes come
  from results in one result schema, with the schema's field definitions (name,
  unit). One sheet row per condition replicate, i.e. per physical run.
- **Assay runs (v2).** Runs of one assay run schema (their numeric and dropdown
  fields are the inputs) and assay results of one result schema (the chosen field
  is the target). One sheet row per assay run.

Every column has a provenance entry (Benchling id -> column). Every run that did
not become a row is counted in the import report with its reason. The mapping never
guesses a value: a run with conflicting values for one parameter, no result, more
than one result, or a non-numeric result is skipped and counted.

Assumptions the specs do not settle (docs/BENCHLING.md, "Not verified"):
- a v3 result belongs to a run when one of its link fields points at the
  condition replicate (or its method execution instance; or the procedure run when
  that run has exactly one replicate);
- a planned or measured `value` reads as the TypedValue shape the plan input uses
  (`{"type": "ft_float", "floatValue": 1.5, "value": "1.5"}`), or a bare scalar.

This module is HTTP-only: no portal imports. `kalos.integrations.benchling.routes`
sends the sheet through the regular upload path, so every upload check applies.
"""
from __future__ import annotations

import csv
import io
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from kalos.integrations.benchling.client import (
    BenchlingClient,
    BenchlingError,
    api_id,
    chunks,
)

V3 = "/api/v3"
V2 = "/api/v2"
RUN_ID_COLUMN = "run_id"
MAX_STUDIES = 500
MAX_SCHEMA_SCAN = 50
MAX_EXAMPLES = 5
MAX_NAME_CHARS = 100
LINKED_IDS_PER_CALL = 200


class ImportRefused(BenchlingError):
    """The source cannot become a run sheet (no target field, no rows). The
    message names Benchling ids only, never values."""


# --- listing ------------------------------------------------------------------- #


def list_studies(client: BenchlingClient, *, limit: int = MAX_STUDIES) -> list[dict[str, Any]]:
    """Unarchived v3 studies: id, name, phase, modifiedAt."""
    out = []
    for item in client.paginate(f"{V3}/study/items", {"archived.anyOf": [False], "sort": "modifiedAt:desc"}):
        out.append({k: item.get(k) for k in ("id", "name", "phase", "modifiedAt", "displayId")})
        if len(out) >= limit:
            break
    return out


def list_projects(client: BenchlingClient, *, limit: int = MAX_STUDIES) -> list[dict[str, Any]]:
    """v2 projects (id, name), for tenants without Bioprocess."""
    out = []
    for item in client.paginate(f"{V2}/projects", {"archiveReason": "NOT_ARCHIVED"}, items_key="projects"):
        out.append({"id": item.get("id"), "name": item.get("name")})
        if len(out) >= limit:
            break
    return out


# --- sheet model ------------------------------------------------------------------ #


@dataclass
class ColumnSource:
    """Where one sheet column came from."""

    column: str
    role: str                      # "identifier" | "input" | "target"
    kind: str                      # planned_parameter | measured_parameter | result_field | run_field | replicate
    benchling_id: str              # parameter version id, field definition id, or v2 field system name
    name: str                      # Benchling display name
    unit: str | None = None        # unit symbol, when Benchling gave one
    unit_id: str | None = None
    data_type: str | None = None
    template_method_version_ids: list[str] = field(default_factory=list)
    rows_with_value: int = 0


@dataclass
class ImportReport:
    runs_seen: int = 0
    rows_kept: int = 0
    skipped: dict[str, int] = field(default_factory=dict)
    skipped_examples: dict[str, list[str]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def skip(self, reason: str, benchling_id: str) -> None:
        self.skipped[reason] = self.skipped.get(reason, 0) + 1
        ex = self.skipped_examples.setdefault(reason, [])
        if len(ex) < MAX_EXAMPLES:
            ex.append(benchling_id)


@dataclass
class RunSheet:
    source: str                    # "v3_study" | "v2_assay"
    source_id: str
    fetched_at: datetime
    columns: list[str]
    rows: list[dict[str, Any]]
    provenance: list[ColumnSource]
    report: ImportReport
    target_column: str
    target: dict[str, Any]         # schema_id, field_id, system_name, unit, project_id
    row_sources: list[dict[str, str]] = field(default_factory=list)
    flowchart_config_version_ids: list[str] = field(default_factory=list)

    @property
    def filename(self) -> str:
        """Records the source id and the fetch time (UTC)."""
        safe = re.sub(r"[^A-Za-z0-9_-]", "_", self.source_id)[:64]
        return f"benchling-{safe}-{self.fetched_at.strftime('%Y%m%dT%H%M%SZ')}.csv"

    def to_csv(self) -> bytes:
        buf = io.StringIO()
        writer = csv.writer(buf, lineterminator="\n")
        writer.writerow([csv_safe(c) for c in self.columns])
        for row in self.rows:
            writer.writerow([csv_safe(_cell(row.get(c))) for c in self.columns])
        return buf.getvalue().encode("utf-8")

    def provenance_table(self) -> list[dict[str, Any]]:
        return [asdict(p) for p in self.provenance]

    def summary(self) -> dict[str, Any]:
        return {
            "source": self.source, "source_id": self.source_id,
            "fetched_at": self.fetched_at.isoformat(), "filename": self.filename,
            "target_column": self.target_column, "target": self.target,
            "columns": self.columns, "provenance": self.provenance_table(),
            "report": asdict(self.report),
        }


def _cell(value: Any) -> str:
    """Floats as repr (exact round trip); text as is (csv_safe is applied by the caller)."""
    if value is None:
        return ""
    if isinstance(value, float):
        return repr(value)
    return str(value)


# A cell or header that a spreadsheet would run as a formula (CSV injection):
# text starting with one of these, unless it is a plain number like -2.5 or +1.
FORMULA_START = ("=", "+", "-", "@", "\t", "\r")
_PLAIN_NUMBER = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")


def csv_safe(text: str) -> str:
    """Prefix `'` to text a spreadsheet would treat as a formula. Idempotent
    (the result starts with `'`). Numbers written as text are left alone."""
    if text and text[0] in FORMULA_START and _PLAIN_NUMBER.fullmatch(text) is None:
        return "'" + text
    return text


def _clean_name(name: Any, fallback: str) -> str:
    text = name if isinstance(name, str) else ""
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C").strip()
    return csv_safe((text or fallback)[:MAX_NAME_CHARS])


def _safe_value(value: Any) -> Any:
    return csv_safe(value) if isinstance(value, str) else value


def _unique(name: str, taken: set[str]) -> str:
    candidate, n = name, 2
    while candidate.lower() in taken:
        candidate = f"{name} ({n})"
        n += 1
    taken.add(candidate.lower())
    return candidate


def _with_unit(name: str, unit: str | None) -> str:
    return f"{name} [{unit}]" if unit else name


# --- value parsing --------------------------------------------------------------- #


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        f = float(value)
        return f if math.isfinite(f) else None
    if isinstance(value, str):
        try:
            f = float(value.strip())
        except ValueError:
            return None
        return f if math.isfinite(f) else None
    return None


def parameter_value(raw: Any) -> float | str | None:
    """A planned/measured value as a number or a text level. TypedValue
    (`floatValue`/`value`), a `{"value": TypedValue}` wrapper, or a bare scalar."""
    if isinstance(raw, dict):
        if "value" in raw and isinstance(raw["value"], dict) and "type" not in raw:
            return parameter_value(raw["value"])
        kind = str(raw.get("type") or raw.get("__typename") or "")
        for key in ("floatValue", "numericValue", "intValue"):
            num = _number(raw.get(key))
            if num is not None:
                return num
        inner = raw.get("value")
        if "float" in kind or "decimal" in kind.lower() or "integer" in kind.lower():
            return _number(inner)
        if isinstance(inner, str):
            num = _number(inner) if "string" not in kind and "text" not in kind.lower() else None
            return num if num is not None else (inner.strip() or None)
        if isinstance(inner, (int, float)) and not isinstance(inner, bool):
            return _number(inner)
        if isinstance(inner, dict):  # a link: use its name or id as the level
            label = inner.get("name") or inner.get("id")
            return label if isinstance(label, str) and label else None
        return None
    if isinstance(raw, str):
        num = _number(raw)
        return num if num is not None else (raw.strip() or None)
    return _number(raw)


def _schema_field_number(sfv: dict[str, Any]) -> tuple[bool, float | None]:
    """(present, number) for a v3 SchemaFieldValue. Present but None = not numeric."""
    value = sfv.get("value")
    if value is None:
        return False, None
    if isinstance(value, dict):
        num = _number(value.get("value"))
        if num is None:
            num = _number(value.get("numericValue"))
        return True, num
    return True, _number(value)


def _linked_ids(sfv: dict[str, Any]) -> list[str]:
    value = sfv.get("value")
    if not isinstance(value, dict):
        return []
    inner = value.get("value")
    items = inner if isinstance(inner, list) else [inner]
    out = []
    for item in items:
        if isinstance(item, dict) and isinstance(item.get("id"), str):
            out.append(item["id"])
        elif isinstance(item, str):
            out.append(item)
    if isinstance(sfv.get("linkedEntityId"), str):
        out.append(sfv["linkedEntityId"])
    return out


# --- v3 study ----------------------------------------------------------------------- #


@dataclass
class StudyData:
    """The raw objects fetched for one study (kept separate from the mapping so the
    mapping can be tested on recorded fixtures)."""

    study_id: str
    runs: list[dict[str, Any]]
    conditions: list[dict[str, Any]]
    replicates: list[dict[str, Any]]
    meis: list[dict[str, Any]]
    planned: list[dict[str, Any]]
    measured: list[dict[str, Any]]
    parameter_versions: list[dict[str, Any]]
    units: dict[str, dict[str, Any]]
    result_schema_id: str
    field_definitions: list[dict[str, Any]]
    results: list[dict[str, Any]]
    fetched_at: datetime


def _ref_id(obj: Any, key: str) -> str | None:
    ref = obj.get(key) if isinstance(obj, dict) else None
    if isinstance(ref, dict) and isinstance(ref.get("id"), str):
        return ref["id"]
    return None


def find_result_schema(client: BenchlingClient, field_id: str) -> tuple[str, list[dict[str, Any]]]:
    """The result schema that holds field definition `field_id`, scanning at most
    MAX_SCHEMA_SCAN schemas."""
    for n, schema in enumerate(client.paginate(f"{V3}/result-schema/items", {"archived.anyOf": [False]})):
        if n >= MAX_SCHEMA_SCAN:
            break
        sid = schema.get("id")
        if not isinstance(sid, str):
            continue
        defs = field_definitions(client, sid)
        if any(d.get("id") == field_id for d in defs):
            return sid, defs
    raise ImportRefused("the target field was not found in the tenant's result schemas; pass result_schema_id")


def field_definitions(client: BenchlingClient, schema_id: str) -> list[dict[str, Any]]:
    body = client.get(f"{V3}/result-schema/{api_id(schema_id)}/field-definitions/items", {"archived.anyOf": [False]})
    items = body.get("items") if isinstance(body, dict) else None
    return [i for i in items or [] if isinstance(i, dict)]


def fetch_study(
    client: BenchlingClient, study_id: str, *, target_field_id: str, result_schema_id: str | None = None,
) -> StudyData:
    """Fetch everything `map_study` needs. Calls: about 2 per run, plus 1 per 100
    replicates, per 100 parameter values and per 100 results (docs/BENCHLING.md)."""
    fetched_at = datetime.now(UTC).replace(microsecond=0)
    runs = list(client.paginate(f"{V3}/procedure-run/items",
                                {"study.eq": study_id, "archived.anyOf": [False], "sort": "createdAt:asc"}))
    conditions: list[dict[str, Any]] = []
    meis: list[dict[str, Any]] = []
    for run in runs:
        rid = api_id(run.get("id"))
        body = client.get(f"{V3}/procedure-run/{rid}/conditions/items")
        conditions.extend(i for i in (body or {}).get("items", []) if isinstance(i, dict))
        body = client.get(f"{V3}/procedure-run/{rid}/method-execution-instances/items",
                          {"archived.anyOf": [False]})
        meis.extend(i for i in (body or {}).get("items", []) if isinstance(i, dict))
    rep_ids = sorted({r for m in meis if (r := _ref_id(m, "procedureRunConditionReplicate"))})
    replicates: list[dict[str, Any]] = []
    for ids in chunks(rep_ids):
        replicates.extend(client.paginate(f"{V3}/procedure-run-condition-replicate/items", {"id.anyOf": ids}))
    mei_ids = sorted({m["id"] for m in meis if isinstance(m.get("id"), str)})
    planned: list[dict[str, Any]] = []
    measured: list[dict[str, Any]] = []
    for ids in chunks(mei_ids):
        planned.extend(client.paginate(f"{V3}/procedure-parameter-planned-value/items",
                                       {"methodExecutionInstance.anyOf": ids}))
        measured.extend(client.paginate(f"{V3}/procedure-parameter-measured-value/items",
                                        {"methodExecutionInstance.anyOf": ids}))
    pv_ids = sorted({p for v in planned + measured if (p := _ref_id(v, "parameterVersion"))})
    versions: list[dict[str, Any]] = []
    for ids in chunks(pv_ids):
        versions.extend(client.paginate(f"{V3}/procedure-parameter-version/items", {"id.anyOf": ids}))
    if result_schema_id:
        defs = field_definitions(client, result_schema_id)
        schema_id = result_schema_id
    else:
        schema_id, defs = find_result_schema(client, target_field_id)
    # Only this study's results: filter on the link-field value server-side
    # (`schemaFieldLinkValue.linkedItemIds`, maxItems 1000; sent 200 at a time),
    # instead of listing every result the schema has across the tenant.
    link_ids = sorted(set(rep_ids) | set(mei_ids) | {r["id"] for r in runs if isinstance(r.get("id"), str)})
    results: list[dict[str, Any]] = []
    seen_results: set[str] = set()
    for ids in chunks(link_ids, LINKED_IDS_PER_CALL):
        for res in client.paginate(f"{V3}/result/items", {
            "schema.eq": schema_id, "archived.anyOf": [False], "schemaFieldLinkValue.linkedItemIds": ids,
        }):
            res_id = res.get("id")
            if isinstance(res_id, str) and res_id not in seen_results:
                seen_results.add(res_id)
                results.append(res)
    unit_ids = sorted({u for v in versions if (u := _ref_id(v.get("dataType") or {}, "unit"))}
                      | {u for d in defs if (u := _ref_id(d, "unit"))})
    units: dict[str, dict[str, Any]] = {}
    for ids in chunks(unit_ids):
        for unit in client.paginate(f"{V3}/unit/items", {"id.anyOf": ids}):
            if isinstance(unit.get("id"), str):
                units[unit["id"]] = unit
    return StudyData(study_id, runs, conditions, replicates, meis, planned, measured, versions, units,
                     schema_id, defs, results, fetched_at)


def _unit_symbol(units: dict[str, dict[str, Any]], unit_id: str | None) -> str | None:
    if unit_id is None:
        return None
    unit = units.get(unit_id) or {}
    sym = unit.get("symbol") or unit.get("name")
    return sym if isinstance(sym, str) and sym else None


def map_study(data: StudyData, *, target_field_id: str) -> RunSheet:
    """One row per condition replicate. See the module docstring for the rules."""
    report = ImportReport()
    target_def = next((d for d in data.field_definitions if d.get("id") == target_field_id), None)
    if target_def is None:
        raise ImportRefused("the target field is not a field of the result schema")
    typename = str(target_def.get("__typename") or target_def.get("type") or "")
    if typename and not any(t in typename.lower() for t in ("decimal", "integer", "float")):
        raise ImportRefused("the target field is not a numeric field")

    runs = {r["id"]: r for r in data.runs if isinstance(r.get("id"), str)}
    conditions = {c["id"]: c for c in data.conditions if isinstance(c.get("id"), str)}
    replicates = {r["id"]: r for r in data.replicates if isinstance(r.get("id"), str)}
    versions = {v["id"]: v for v in data.parameter_versions if isinstance(v.get("id"), str)}

    # Row unit = condition replicate (or the instance itself when it has none).
    row_meis: dict[str, list[dict[str, Any]]] = defaultdict(list)
    row_run: dict[str, str] = {}
    mei_row: dict[str, str] = {}
    for mei in data.meis:
        mid = mei.get("id")
        if not isinstance(mid, str) or mei.get("archived"):
            continue
        key = _ref_id(mei, "procedureRunConditionReplicate") or mid
        row_meis[key].append(mei)
        mei_row[mid] = key
        rid = _ref_id(mei, "procedureRun")
        if rid:
            row_run[key] = rid
    report.runs_seen = len(row_meis)
    runs_without_rows = [rid for rid in runs if rid not in set(row_run.values())]
    for rid in runs_without_rows:
        report.skip("run_has_no_method_execution_instances", rid)

    # values[row][pv_id] -> set of values; method versions per parameter
    planned_vals: dict[str, dict[str, list[Any]]] = defaultdict(lambda: defaultdict(list))
    measured_vals: dict[str, dict[str, list[Any]]] = defaultdict(lambda: defaultdict(list))
    methods: dict[str, set[str]] = defaultdict(set)
    mei_by_id = {m["id"]: m for m in data.meis if isinstance(m.get("id"), str)}
    for bucket, values in ((planned_vals, data.planned), (measured_vals, data.measured)):
        for v in values:
            mid = _ref_id(v, "methodExecutionInstance")
            pv = _ref_id(v, "parameterVersion")
            if mid is None or pv is None or mid not in mei_row:
                continue
            parsed = parameter_value(v.get("value"))
            if parsed is None:
                continue
            bucket[mei_row[mid]][pv].append(parsed)
            if bucket is planned_vals:
                mv = _ref_id(mei_by_id.get(mid, {}), "templateMethodVersion")
                if mv:
                    methods[pv].add(mv)

    planned_params = {pv for row in planned_vals.values() for pv in row}
    measured_params = {pv for row in measured_vals.values() for pv in row}
    measured_only = measured_params - planned_params
    if measured_params & planned_params:
        report.notes.append(
            f"{len(measured_params & planned_params)} parameter(s) have planned values; their measured "
            "values were not used as inputs (the planned value is the setpoint Kalos proposes)."
        )
    if measured_only:
        report.notes.append(
            f"{len(measured_only)} parameter(s) have measured values only and were added as inputs; check "
            "they are not a stand-in for the target."
        )

    # Columns
    taken: set[str] = {RUN_ID_COLUMN}
    provenance: list[ColumnSource] = [ColumnSource(
        RUN_ID_COLUMN, "identifier", "replicate", "procedureRunConditionReplicate.id", "Condition replicate")]
    param_col: dict[str, str] = {}
    for kind, params in (("planned_parameter", planned_params), ("measured_parameter", measured_only)):
        for pv in sorted(params, key=lambda p: (str((versions.get(p) or {}).get("name") or p).casefold(), p)):
            ver = versions.get(pv) or {}
            dtype = ver.get("dataType") or {}
            unit_id = _ref_id(dtype, "unit")
            unit = _unit_symbol(data.units, unit_id)
            name = _clean_name(ver.get("name"), pv)
            col = _unique(_with_unit(name, unit), taken)
            param_col[pv] = col
            provenance.append(ColumnSource(
                col, "input", kind, pv, name, unit, unit_id,
                dtype.get("type") if isinstance(dtype.get("type"), str) else None,
                sorted(methods.get(pv, ())),
            ))
    t_unit_id = _ref_id(target_def, "unit")
    t_unit = _unit_symbol(data.units, t_unit_id)
    t_name = _clean_name(target_def.get("name"), target_field_id)
    target_col = _unique(_with_unit(t_name, t_unit), taken)
    provenance.append(ColumnSource(target_col, "target", "result_field", target_field_id, t_name, t_unit,
                                   t_unit_id, typename or None))

    # Results -> rows
    run_rows: dict[str, list[str]] = defaultdict(list)
    for key, rid in row_run.items():
        run_rows[rid].append(key)
    cond_ids = set(conditions)
    target_by_row: dict[str, list[tuple[str, float | None]]] = defaultdict(list)
    project_ids: Counter[str] = Counter()
    for res in data.results:
        if res.get("archived"):
            continue
        fields = [f for f in res.get("schemaFields") or [] if isinstance(f, dict)]
        target_sfv = next((f for f in fields if _ref_id(f, "fieldDefinition") == target_field_id), None)
        if target_sfv is None:
            continue
        present, number = _schema_field_number(target_sfv)
        if not present:
            continue
        linked = {i for f in fields if f is not target_sfv for i in _linked_ids(f)}
        rows_hit = {k for k in linked if k in row_meis} | {mei_row[i] for i in linked if i in mei_row}
        if not rows_hit:
            run_hits = [r for r in linked if r in run_rows]
            if len(run_hits) == 1 and len(run_rows[run_hits[0]]) == 1:
                rows_hit = {run_rows[run_hits[0]][0]}
            elif run_hits or (linked & cond_ids):
                report.skip("result_linked_to_a_run_or_condition_not_one_replicate", str(res.get("id")))
                continue
            else:
                continue  # a result of another study
        if len(rows_hit) > 1:
            report.skip("result_linked_to_several_replicates", str(res.get("id")))
            continue
        target_by_row[next(iter(rows_hit))].append((str(res.get("id")), number))
        pid = _ref_id(res, "project")
        if pid:
            project_ids[pid] += 1

    rows: list[dict[str, Any]] = []
    row_sources: list[dict[str, str]] = []
    counts: Counter[str] = Counter()
    def _order(k: str) -> tuple[str, str, int, str]:
        rep = replicates.get(k)
        return (row_run.get(k, ""), _ref_id(rep or {}, "procedureRunCondition") or "", _rep_sort(rep), k)

    for key in sorted(row_meis, key=_order):
        hits = target_by_row.get(key, [])
        if not hits:
            report.skip("no_result_for_target", key)
            continue
        if len(hits) > 1:
            report.skip("several_results_for_target", key)
            continue
        if hits[0][1] is None:
            report.skip("target_not_numeric", key)
            continue
        row: dict[str, Any] = {RUN_ID_COLUMN: key, target_col: hits[0][1]}
        conflict = False
        for pv, col in param_col.items():
            src = planned_vals if pv in planned_params else measured_vals
            vals = src.get(key, {}).get(pv, [])
            distinct = {(type(v).__name__, v) for v in vals}
            if len(distinct) > 1:
                conflict = True
                break
            if vals:
                row[col] = _safe_value(vals[0])
                counts[col] += 1
        if conflict:
            report.skip("conflicting_values_for_one_parameter", key)
            continue
        if not any(c in row for c in param_col.values()):
            report.skip("no_parameter_values", key)
            continue
        rows.append(row)
        rep = replicates.get(key) or {}
        row_sources.append({
            "run_id": key, "procedure_run_id": row_run.get(key, ""),
            "condition_id": _ref_id(rep, "procedureRunCondition") or "",
            "result_id": hits[0][0],
        })
    for p in provenance:
        p.rows_with_value = counts.get(p.column, 0) if p.role == "input" else len(rows)
    report.rows_kept = len(rows)
    flowcharts = sorted({f for r in runs.values() if (f := _ref_id(r, "procedureFlowchartConfigVersion"))})
    target = {"schema_id": data.result_schema_id, "field_id": target_field_id,
              "system_name": target_def.get("systemName"), "unit": t_unit,
              "project_id": project_ids.most_common(1)[0][0] if project_ids else None,
              "schema_fields": sorted(str(d.get("systemName")) for d in data.field_definitions
                                      if d.get("systemName"))}
    if not rows:
        raise ImportRefused("no run in the study has both parameter values and a numeric result")
    return RunSheet("v3_study", data.study_id, data.fetched_at,
                    [RUN_ID_COLUMN, *param_col.values(), target_col], rows, provenance, report,
                    target_col, target, row_sources, flowcharts)


def _rep_sort(rep: dict[str, Any] | None) -> int:
    n = (rep or {}).get("replicateNumber")
    return n if isinstance(n, int) else 0


# --- v2 assay runs -------------------------------------------------------------------- #


_V2_INPUT_TYPES = frozenset({"float", "integer", "dropdown"})


@dataclass
class AssayData:
    run_schema_id: str
    result_schema_id: str
    runs: list[dict[str, Any]]
    results: list[dict[str, Any]]
    result_schema: dict[str, Any]
    fetched_at: datetime


def fetch_assay(client: BenchlingClient, run_schema_id: str, result_schema_id: str) -> AssayData:
    fetched_at = datetime.now(UTC).replace(microsecond=0)
    runs = list(client.paginate(f"{V2}/assay-runs", {"schemaId": run_schema_id}, items_key="assayRuns"))
    results = list(client.paginate(f"{V2}/assay-results",
                                   {"schemaId": result_schema_id, "archiveReason": "NOT_ARCHIVED"},
                                   items_key="assayResults"))
    schema = client.get(f"{V2}/assay-result-schemas/{api_id(result_schema_id)}") or {}
    return AssayData(run_schema_id, result_schema_id, runs, results, schema, fetched_at)


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def map_assay(data: AssayData, *, target_field: str) -> RunSheet:
    """One row per assay run. `target_field` is the result field's system name
    (the key in v2 `fields`)."""
    report = ImportReport()
    report.runs_seen = len(data.runs)
    defs = [d for d in data.result_schema.get("fieldDefinitions") or [] if isinstance(d, dict)]
    # v2 field definitions carry a display name, results are keyed by system name.
    tdef = next((d for d in defs if _norm(str(d.get("name") or "")) == _norm(target_field)), {})
    if tdef and tdef.get("type") not in (None, "float", "integer"):
        raise ImportRefused("the target field is not a numeric field")
    unit = tdef.get("unit") if isinstance(tdef.get("unit"), dict) else None
    t_unit = (unit or {}).get("symbol") or (unit or {}).get("name")
    run_ids = {r.get("id") for r in data.runs}

    by_run: dict[str, list[tuple[str, float | None]]] = defaultdict(list)
    project_ids: Counter[str] = Counter()
    seen_target = False
    for res in data.results:
        if res.get("archiveRecord"):
            continue
        fields = res.get("fields") or {}
        if target_field not in fields:
            continue
        seen_target = True
        links = [f.get("value") for f in fields.values()
                 if isinstance(f, dict) and f.get("type") == "assay_run_link"]
        linked = [v for v in links if isinstance(v, str) and v in run_ids]
        if len(linked) != 1:
            report.skip("result_not_linked_to_exactly_one_run", str(res.get("id")))
            continue
        by_run[linked[0]].append((str(res.get("id")), _number((fields[target_field] or {}).get("value"))))
        if isinstance(res.get("projectId"), str):
            project_ids[res["projectId"]] += 1
    if not seen_target:
        raise ImportRefused("no assay result carries the target field")

    # Input columns: numeric and dropdown run fields, in first-seen order.
    taken: set[str] = {RUN_ID_COLUMN}
    field_col: dict[str, str] = {}
    provenance = [ColumnSource(RUN_ID_COLUMN, "identifier", "run_field", "assayRun.id", "Assay run")]
    unused: set[str] = set()
    for run in data.runs:
        for key, f in (run.get("fields") or {}).items():
            if not isinstance(f, dict) or key in field_col:
                continue
            if f.get("type") in _V2_INPUT_TYPES and not f.get("isMulti"):
                col = _unique(_clean_name(key, key), taken)
                field_col[key] = col
                provenance.append(ColumnSource(col, "input", "run_field", key, key, None, None, f.get("type")))
            else:
                unused.add(key)
    if unused:
        report.notes.append(f"{len(unused)} run field(s) were not used (text, link, date or multi-value).")
    t_name = _clean_name(tdef.get("name") or target_field, target_field)
    target_col = _unique(_with_unit(t_name, t_unit if isinstance(t_unit, str) else None), taken)
    provenance.append(ColumnSource(target_col, "target", "result_field", target_field, t_name,
                                   t_unit if isinstance(t_unit, str) else None, None, tdef.get("type")))

    rows: list[dict[str, Any]] = []
    row_sources: list[dict[str, str]] = []
    counts: Counter[str] = Counter()
    for run in data.runs:
        rid = run.get("id")
        if not isinstance(rid, str) or run.get("archiveRecord"):
            continue
        hits = by_run.get(rid, [])
        if not hits:
            report.skip("no_result_for_target", rid)
            continue
        if len(hits) > 1:
            report.skip("several_results_for_target", rid)
            continue
        if hits[0][1] is None:
            report.skip("target_not_numeric", rid)
            continue
        row: dict[str, Any] = {RUN_ID_COLUMN: rid, target_col: hits[0][1]}
        for key, col in field_col.items():
            f = (run.get("fields") or {}).get(key) or {}
            value = parameter_value(f.get("textValue") if f.get("type") == "dropdown" else f.get("value"))
            if value is not None:
                row[col] = _safe_value(value)
                counts[col] += 1
        if len(row) == 2:
            report.skip("no_parameter_values", rid)
            continue
        rows.append(row)
        row_sources.append({"run_id": rid, "result_id": hits[0][0], "v3_id": str(run.get("v3Id") or "")})
    for p in provenance:
        p.rows_with_value = counts.get(p.column, 0) if p.role == "input" else len(rows)
    report.rows_kept = len(rows)
    if not rows:
        raise ImportRefused("no assay run has both input fields and a numeric result")
    target = {"schema_id": data.result_schema_id, "field_id": target_field, "system_name": target_field,
              "unit": t_unit, "project_id": project_ids.most_common(1)[0][0] if project_ids else None,
              "schema_fields": sorted({str(k) for r in data.results for k in (r.get("fields") or {})})}
    return RunSheet("v2_assay", data.run_schema_id, data.fetched_at,
                    [RUN_ID_COLUMN, *field_col.values(), target_col], rows, provenance, report,
                    target_col, target, row_sources)
