"""Typed argument/result models and thin adapters from skill tools to the
existing engine functions.

Each adapter takes a validated args model and a `ToolContext`, calls ONE engine
function (named in the tool table, kalos/skills/tools.py), and returns plain
JSON-safe data. Adapters never invent results: engine refusals propagate as the
engine raised them, and `invoke` records them as refusals.

Every engine import is inside the adapter, so importing this module never loads
torch (or the portal's FastAPI stack). Kalos-original code.
"""
from __future__ import annotations

import base64
import binascii
import dataclasses
import datetime as dt
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

# Size caps for inline data (no upload route in v1; these bound in-process calls).
MAX_ROWS = 20_000
MAX_COLUMNS = 500
MAX_INLINE_BYTES = 10 * 1024 * 1024
MAX_VECTOR = 100_000

Scalar = float | int | str | bool | None


class ToolRefusal(Exception):
    """A receipted non-result, never retried and never an error.

    `kind="blocked"`: the data cannot support the result (too few rows, a
    blocking finding, a failed gate). `kind="refused"`: the request is not
    allowed or not well-formed for this tool (unknown campaign, bad column
    names, policy). Never get past either by weakening a check."""

    def __init__(self, code: str, reason: str, *, kind: Literal["blocked", "refused"] = "refused",
                 rows: list[int] | None = None) -> None:
        super().__init__(reason)
        self.code = code
        self.reason = reason
        self.kind = kind
        self.rows = rows


@dataclass
class ToolContext:
    """Who is calling and with what. `store` is a `CampaignStore` when a tool
    needs campaign state; `runs` is a `RunStore` (None = the default root)."""

    tenant: str
    subject: str | None = None
    scopes: frozenset[str] = frozenset({"read"})
    store: Any = None
    runs: Any = None  # RunStore; None = the default root
    extras: dict[str, Any] = field(default_factory=dict)
    # Filled per call by the adapters: sha256 of every input actually read.
    input_sha256s: list[str] = field(default_factory=list)


# --- JSON safety --------------------------------------------------------------- #


def jsonable(obj: Any) -> Any:
    """Plain JSON data from engine output: numpy/pandas scalars and arrays,
    dataclasses, objects with `to_dict`, tuples/sets; non-finite floats -> None."""
    import numpy as np

    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, (int, np.integer)) and not isinstance(obj, bool):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, np.ndarray):
        return [jsonable(x) for x in obj.tolist()]
    if isinstance(obj, Mapping):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        seq = sorted(obj, key=str) if isinstance(obj, (set, frozenset)) else obj
        return [jsonable(x) for x in seq]
    if isinstance(obj, (dt.date, dt.datetime)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return jsonable(obj.to_dict())
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return jsonable({f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)})
    if hasattr(obj, "item"):
        return jsonable(obj.item())
    return str(obj)


def records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return [jsonable(r) for r in df.to_dict(orient="records")]


# --- argument models ----------------------------------------------------------- #


class Args(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TableArgs(Args):
    """A table given inline (`rows`, one dict per run) or read from one of the
    caller tenant's campaigns (`campaign_id`: base rows plus measured results)."""

    rows: list[dict[str, Scalar]] | None = Field(default=None, max_length=MAX_ROWS)
    campaign_id: str | None = Field(default=None, max_length=64)

    @model_validator(mode="after")
    def _one_source(self) -> TableArgs:
        if (self.rows is None) == (self.campaign_id is None):
            raise ValueError("give exactly one of 'rows' or 'campaign_id'")
        if self.rows is not None:
            if not self.rows:
                raise ValueError("'rows' is empty")
            cols = {k for r in self.rows for k in r}
            if len(cols) > MAX_COLUMNS:
                raise ValueError(f"more than {MAX_COLUMNS} columns")
        return self


class ModelArgs(TableArgs):
    """A table plus the response and factor columns. Taken from the campaign when
    `campaign_id` is given and they are omitted."""

    target: str | None = Field(default=None, max_length=256)
    features: list[str] | None = Field(default=None, max_length=MAX_COLUMNS)


class FileArgs(Args):
    """A run-sheet file given inline: `content_base64` (CSV/TSV/XLSX bytes) or
    `csv_text`."""

    content_base64: str | None = Field(default=None, max_length=MAX_INLINE_BYTES * 4 // 3 + 4)
    csv_text: str | None = Field(default=None, max_length=MAX_INLINE_BYTES)
    filename: str | None = Field(default=None, max_length=256)
    sheet: str | None = Field(default=None, max_length=256)

    @model_validator(mode="after")
    def _one_source(self) -> FileArgs:
        if (self.content_base64 is None) == (self.csv_text is None):
            raise ValueError("give exactly one of 'content_base64' or 'csv_text'")
        return self

    def raw(self, ctx: ToolContext) -> bytes:
        """The file bytes; their sha256 is recorded as an input hash."""
        import hashlib

        if self.csv_text is not None:
            data = self.csv_text.encode("utf-8")
        else:
            try:
                data = base64.b64decode(self.content_base64 or "", validate=True)
            except (binascii.Error, ValueError) as exc:
                raise ToolRefusal("bad_input", "content_base64 is not valid base64") from exc
        ctx.input_sha256s.append(hashlib.sha256(data).hexdigest())
        return data


class CampaignArgs(Args):
    campaign_id: str | None = Field(default=None, max_length=64)


Profile = Literal["bioprocess", "generic"]


def _profile(name: str):
    from kalos.domains import BIOPROCESS_PROFILE, GENERIC_PROFILE

    return BIOPROCESS_PROFILE if name == "bioprocess" else GENERIC_PROFILE


# --- table resolution ---------------------------------------------------------- #


def _campaign_state(ctx: ToolContext, campaign_id: str | None) -> dict[str, Any]:
    if ctx.store is None:
        raise ToolRefusal("no_campaign_store", "this call needs campaign storage")
    from kalos.portal.campaign import CampaignNotFound

    try:
        state = ctx.store.get(tenant=ctx.tenant, campaign_id=campaign_id)
    except CampaignNotFound as exc:
        raise ToolRefusal("not_found", "campaign not found") from exc
    if state is None:
        raise ToolRefusal("not_found", "campaign not found")
    return state


def resolve_table(args: TableArgs, ctx: ToolContext) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    """The frame to work on, and the campaign state it came from (if any).
    Campaign rows = base rows plus every measured pending result folded in
    (what the next analysis would see); awaiting runs are never included."""
    import pandas as pd

    from .receipts import frame_sha256

    if args.rows is not None:
        df = pd.DataFrame(list(args.rows))
        ctx.input_sha256s.append(frame_sha256(df))
        return df, None
    state = _campaign_state(ctx, args.campaign_id)
    target = state["target"]
    rows = list(state.get("base_rows") or [])
    for run in state.get("pending") or []:
        if run.get("result") is not None and isinstance(run.get("recipe"), dict):
            rows.append({**run["recipe"], target: run["result"]})
    if not rows:
        raise ToolRefusal("no_data", "the campaign has no measured rows", kind="blocked")
    df = pd.DataFrame(rows)
    ctx.input_sha256s.append(frame_sha256(df))
    return df, state


def resolve_model_table(args: ModelArgs, ctx: ToolContext) -> tuple[pd.DataFrame, str, list[str]]:
    df, state = resolve_table(args, ctx)
    target = args.target or (state or {}).get("target")
    features = list(args.features) if args.features else list((state or {}).get("features") or [])
    if not target:
        raise ToolRefusal("bad_input", "no target column given")
    if target not in df.columns:
        raise ToolRefusal("bad_input", "target column not in the table")
    if not features:
        features = [c for c in df.columns if c != target]
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ToolRefusal("bad_input", f"{len(missing)} feature column(s) not in the table")
    if target in features:
        raise ToolRefusal("bad_input", "the target cannot also be a feature")
    return df, target, features


def numeric_xy(df: pd.DataFrame, target: str, features: list[str]):
    """X (float, rows with a finite target and finite features only) and y."""
    import numpy as np
    import pandas as pd

    num = df[[*features, target]].apply(pd.to_numeric, errors="coerce")
    nonnum = [f for f in features if num[f].notna().sum() == 0 and df[f].notna().any()]
    if nonnum:
        raise ToolRefusal("bad_input", f"{len(nonnum)} feature column(s) are not numeric")
    num = num.replace([np.inf, -np.inf], np.nan).dropna()
    if len(num) < 3:
        raise ToolRefusal("too_few_rows", "fewer than 3 complete numeric rows", kind="blocked")
    return num[features].to_numpy(float), num[target].to_numpy(float)


# --- result models ------------------------------------------------------------- #


class Result(BaseModel):
    """Base for tool results. `data` is the engine output as plain JSON; each
    subclass types the keys the tool guarantees."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class Loose(BaseModel):
    """Typed guaranteed keys; the rest of the engine dict is kept as-is."""

    model_config = ConfigDict(extra="allow", frozen=True)


# column-role-inference ---------------------------------------------------------


class GuessRoleArgs(Args):
    header: str = Field(min_length=1, max_length=256)
    is_numeric: bool
    parse_rate: float = Field(ge=0.0, le=1.0)


class GuessRoleResult(Result):
    role: Literal["target", "feature", "group", "identity", "freetext", "metadata"]


def guess_role(a: GuessRoleArgs, ctx: ToolContext) -> dict:
    from kalos.normalize.synonyms import guess_role as fn

    return {"role": fn(a.header, is_numeric=a.is_numeric, parse_rate=a.parse_rate)}


class OfflinePlanArgs(TableArgs):
    target_column: str | None = Field(default=None, max_length=256)


class PlanColumn(Loose):
    raw_name: str
    canonical_name: str | None
    role: str


class OfflinePlanResult(Result):
    created_by: Literal["offline"]
    model: None = None
    columns: list[PlanColumn]


def offline_plan(a: OfflinePlanArgs, ctx: ToolContext) -> dict:
    from kalos.normalize.llm import offline_plan as fn

    df, _ = resolve_table(a, ctx)
    return jsonable(fn(df, target_column=a.target_column))


class ColumnProvenanceArgs(ModelArgs):
    profile: Profile = "bioprocess"


class ColumnProvenanceResult(Result):
    columns: list[Loose]


def column_provenance(a: ColumnProvenanceArgs, ctx: ToolContext) -> dict:
    import pandas as pd

    from kalos.portal.analysis_parts.identifiers import is_identifier_name
    from kalos.portal.validate import column_provenance as fn

    df, target, features = resolve_model_table(a, ctx)
    prof = _profile(a.profile)
    numeric = [str(c) for c in df.columns
               if pd.to_numeric(df[c], errors="coerce").notna().any()]
    out = fn(df, target=target, features=features, numeric_cols=numeric,
             is_identifier=lambda n: is_identifier_name(n, prof.id_hint),
             outcome_hint=prof.outcome_hint)
    return {"columns": jsonable(out)}


# replicate-noise-estimate ------------------------------------------------------


class NoiseArgs(ModelArgs):
    decimals: int = Field(default=6, ge=0, le=12)


class NoiseReportResult(Result):
    report: Loose


def noise_report(a: NoiseArgs, ctx: ToolContext) -> dict:
    from kalos.core.replicates import noise_report as fn

    df, target, features = resolve_model_table(a, ctx)
    X, y = numeric_xy(df, target, features)
    return {"report": jsonable(fn(X, y, decimals=a.decimals))}


class NoiseFloorResult(Result):
    noise_floor: float | None


def estimate_noise_floor(a: NoiseArgs, ctx: ToolContext) -> dict:
    from kalos.core.replicates import estimate_noise_floor as fn

    df, target, features = resolve_model_table(a, ctx)
    X, y = numeric_xy(df, target, features)
    return {"noise_floor": jsonable(fn(X, y, decimals=a.decimals))}


def heteroscedasticity_report(a: NoiseArgs, ctx: ToolContext) -> dict:
    from kalos.core.replicates import heteroscedasticity_report as fn

    df, target, features = resolve_model_table(a, ctx)
    X, y = numeric_xy(df, target, features)
    return {"report": jsonable(fn(X, y, decimals=a.decimals))}


# run-sheet-checks ----------------------------------------------------------------


class ParseUploadResult(Result):
    kind: str
    n_rows: int
    n_columns: int
    columns: list[str]
    sheets: list[str]
    sheet_used: str | None


def parse_upload(a: FileArgs, ctx: ToolContext) -> dict:
    from kalos.portal.uploads import parse_upload as fn

    p = fn(a.raw(ctx), filename=a.filename, sheet=a.sheet)
    return {"kind": p.kind, "n_rows": len(p.df), "n_columns": p.df.shape[1],
            "columns": [str(c) for c in p.df.columns], "sheets": list(p.sheets),
            "sheet_used": p.sheet_used}


class ValidateFrameArgs(TableArgs):
    target: str | None = Field(default=None, max_length=256)
    features: list[str] | None = Field(default=None, max_length=MAX_COLUMNS)
    mode: Literal["warn", "strict"] = "warn"
    profile: Profile = "bioprocess"


class CheckReportResult(Result):
    report: Loose


def validate_frame(a: ValidateFrameArgs, ctx: ToolContext) -> dict:
    from kalos.validation.runner import validate_frame as fn

    df, _ = resolve_table(a, ctx)
    report = fn(df, target=a.target, features=list(a.features or []),
                profile=_profile(a.profile), mode=a.mode)
    data = jsonable(report)
    if a.mode == "strict":
        blocking = [f for f in data.get("findings", []) if f.get("severity") == "error"]
        if blocking:
            checks = sorted({str(f.get("check")) for f in blocking})
            rows = sorted({int(r) for f in blocking for r in (f.get("rows") or [])})[:100]
            raise ToolRefusal("blocking_findings",
                              f"strict mode: {len(blocking)} blocking finding(s) from "
                              + ", ".join(checks), kind="blocked", rows=rows or None)
    return {"report": data}


# target-readiness-check --------------------------------------------------------


class TargetValuesArgs(Args):
    values: list[float | int | str | None] = Field(min_length=1, max_length=MAX_VECTOR)


class TargetVariationResult(Result):
    variation: Loose


def assess_target_variation(a: TargetValuesArgs, ctx: ToolContext) -> dict:
    from kalos.validation.target import assess_target_variation as fn

    return {"variation": jsonable(fn(list(a.values)))}


# unit-normalisation ------------------------------------------------------------


class ParseValueArgs(Args):
    raw: str | float | int = Field(union_mode="left_to_right")


class ParseValueResult(Result):
    value: float | None
    unit: str | None


def parse_value(a: ParseValueArgs, ctx: ToolContext) -> dict:
    from kalos.normalize.units import parse_value as fn

    raw = a.raw[:256] if isinstance(a.raw, str) else a.raw
    v, u = fn(raw)
    return {"value": jsonable(v), "unit": u}


class ConvertArgs(Args):
    value: float
    unit_token: str | None = Field(default=None, max_length=64)


class ConvertResult(Result):
    value: float | None
    base_unit: str


def convert_unit(a: ConvertArgs, ctx: ToolContext) -> dict:
    from kalos.normalize.units import convert as fn

    v, base = fn(a.value, a.unit_token)
    return {"value": jsonable(v), "base_unit": base}


class UnitConversionsResult(Result):
    conversions: list[Loose]
    findings: list[Loose]
    rows: list[dict[str, Any]]


def apply_unit_conversions(a: TableArgs, ctx: ToolContext) -> dict:
    from kalos.validation.checks import check_units_consistency
    from kalos.validation.runner import apply_unit_conversions as fn

    df, _ = resolve_table(a, ctx)
    findings, conversions = check_units_consistency(df)
    out = fn(df, conversions)
    return {"conversions": jsonable(conversions), "findings": jsonable(findings),
            "rows": records(out)}


# search-box-definition ---------------------------------------------------------


class SearchBoxArgs(ModelArgs):
    categoricals: list[str] = Field(default_factory=list, max_length=MAX_COLUMNS)
    bounds: dict[str, tuple[float, float]] | None = None


class Dim(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str
    kind: Literal["continuous", "categorical"]
    lower: float
    upper: float
    levels: list[str]


class SearchBoxResult(Result):
    dims: list[Dim]


def build_search_box(a: SearchBoxArgs, ctx: ToolContext) -> dict:
    from kalos.domains.profile import build_design_space as fn

    df, _target, features = resolve_model_table(a, ctx)
    cats = tuple(c for c in a.categoricals if c in features)
    space = fn(df, features, cats, bounds=dict(a.bounds) if a.bounds else None)
    return {"dims": jsonable(list(space.dims))}


class ConstraintArgs(Args):
    constraint: dict[str, Any] | None


class ConstraintResult(Result):
    ok: Literal[True]


def validate_constraint(a: ConstraintArgs, ctx: ToolContext) -> dict:
    from kalos.portal.analysis_parts.refusals import _validate_constraint as fn

    fn(a.constraint)
    return {"ok": True}


# driver-ranking ----------------------------------------------------------------


class DriverArgs(ModelArgs):
    n_boot: int = Field(default=200, ge=20, le=2000)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)


class BootstrapResult(Result):
    feature_names: list[str]
    mean: list[float | None]
    std: list[float | None]
    lo: list[float | None]
    hi: list[float | None]


def bootstrap_spearman(a: DriverArgs, ctx: ToolContext) -> dict:
    from kalos.core.drivers import bootstrap_spearman as fn

    df, target, features = resolve_model_table(a, ctx)
    X, y = numeric_xy(df, target, features)
    return jsonable(fn(X, y, B=a.n_boot, random_state=a.seed, feature_names=features))


class RankDriversArgs(DriverArgs):
    top_k: int = Field(default=5, ge=1, le=100)
    q: float = Field(default=0.05, gt=0.0, lt=1.0)


class Driver(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    factor: str
    rho_mean: float | None
    ci_lo: float | None
    ci_hi: float | None
    p_value: float | None
    significant_at_q: bool


class RankDriversResult(Result):
    drivers: list[Driver]
    q: float
    n_rows: int


def rank_drivers(a: RankDriversArgs, ctx: ToolContext) -> dict:
    import numpy as np

    from kalos.core.drivers import (
        benjamini_hochberg,
        bootstrap_spearman,
        spearman_driver_matrix,
    )
    from kalos.core.drivers import rank_drivers as fn

    df, target, features = resolve_model_table(a, ctx)
    X, y = numeric_xy(df, target, features)
    summary = bootstrap_spearman(X, y, B=a.n_boot, random_state=a.seed, feature_names=features)
    point = spearman_driver_matrix(X, y, feature_names=features)
    pv = np.asarray(point["pvals"], dtype=float)
    finite = np.isfinite(pv)
    sig = np.zeros(len(pv), dtype=bool)
    if finite.any():
        sig[finite] = benjamini_hochberg(pv[finite], q=a.q)
    idx = {n: i for i, n in enumerate(features)}
    out = []
    for name, _score in fn(summary, top_k=a.top_k):
        i = idx[name]
        out.append({"factor": name,
                    "rho_mean": jsonable(np.asarray(summary["mean"])[i]),
                    "ci_lo": jsonable(np.asarray(summary["lo"])[i]),
                    "ci_hi": jsonable(np.asarray(summary["hi"])[i]),
                    "p_value": jsonable(pv[i]), "significant_at_q": bool(sig[i])})
    return {"drivers": out, "q": a.q, "n_rows": len(y)}


# identifier-anonymisation ------------------------------------------------------


class AnonymiseArgs(TableArgs):
    pass


class AnonymiseResult(Result):
    rows: list[dict[str, Any]]
    dropped_columns: list[str]
    hashed_columns: list[str]
    salt_configured: bool


def anonymise_table(a: AnonymiseArgs, ctx: ToolContext) -> dict:
    """Row-wise `Anonymizer.anonymize_meta`: identity columns dropped, grouping
    ids (batch, lot, campaign, experiment) replaced by salted hashes, process
    values untouched. The report names columns, never the salt or alias key."""
    from kalos.data.anonymizer import Anonymizer, salt_configured
    from kalos.portal.config import is_production

    if is_production() and not salt_configured():
        raise ToolRefusal("no_salt", "no anonymisation salt is configured in production")
    df, _ = resolve_table(a, ctx)
    anon = Anonymizer()
    rows = []
    for rec in df.to_dict(orient="records"):
        clean = anon.anonymize_meta(rec)
        clean.pop("anonymized", None)
        rows.append(jsonable(clean))
    cols = [str(c) for c in df.columns]
    kept = set(rows[0]) if rows else set(cols)
    dropped = [c for c in cols if c not in kept]
    hashed = [c for c in cols if c in kept and rows and any(
        rows[i][c] != jsonable(v) for i, v in enumerate(df[c].tolist()))]
    return {"rows": rows, "dropped_columns": dropped, "hashed_columns": hashed,
            "salt_configured": salt_configured()}


class AnonymiseParsedResult(Result):
    kind: str
    sheets: list[str]
    sheet_used: str | None
    sheet_index: int | None


def anonymize_parsed(a: FileArgs, ctx: ToolContext) -> dict:
    from kalos.portal.analysis_parts.anonymize import anonymize_parsed as fn
    from kalos.portal.uploads import parse_upload

    p = fn(parse_upload(a.raw(ctx), filename=a.filename, sheet=a.sheet))
    return {"kind": p.kind, "sheets": list(p.sheets), "sheet_used": p.sheet_used,
            "sheet_index": p.sheet_index}


# trust-report -------------------------------------------------------------------


class CVArgs(ModelArgs):
    group_column: str | None = Field(default=None, max_length=256)
    n_splits: int = Field(default=5, ge=2, le=20)
    n_boot: int = Field(default=200, ge=0, le=5000)
    seed: int = Field(default=0, ge=0, le=2**32 - 1)


class CVResult(Result):
    report: Loose


def grouped_cv_report(a: CVArgs, ctx: ToolContext) -> dict:
    import numpy as np
    import pandas as pd

    from kalos.core.evaluation import grouped_cv_report as fn

    df, target, features = resolve_model_table(a, ctx)
    cols = [*features, target]
    num = df[cols].apply(pd.to_numeric, errors="coerce")
    keep = num.replace([np.inf, -np.inf], np.nan).dropna().index
    if len(keep) < 6:
        raise ToolRefusal("too_few_rows", "fewer than 6 complete numeric rows", kind="blocked")
    X = num.loc[keep, features].to_numpy(float)
    y = num.loc[keep, target].to_numpy(float)
    groups = None
    if a.group_column:
        if a.group_column not in df.columns:
            raise ToolRefusal("bad_input", "group column not in the table")
        groups = df.loc[keep, a.group_column].astype(str).to_numpy()
    rep = fn(X, y, groups=groups, n_splits=a.n_splits, n_boot=a.n_boot, random_state=a.seed)
    return {"report": jsonable(rep)}


class CalibrationArgs(Args):
    actual: list[float] = Field(min_length=2, max_length=MAX_VECTOR)
    predicted: list[float] = Field(min_length=2, max_length=MAX_VECTOR)
    std: list[float] = Field(min_length=2, max_length=MAX_VECTOR)

    @model_validator(mode="after")
    def _same_length(self) -> CalibrationArgs:
        if not (len(self.actual) == len(self.predicted) == len(self.std)):
            raise ValueError("actual, predicted and std must have the same length")
        return self


class CalibrationResult(Result):
    calibration: Loose


def interval_calibration(a: CalibrationArgs, ctx: ToolContext) -> dict:
    from kalos.core.evaluation import interval_calibration as fn

    return {"calibration": jsonable(fn(a.actual, a.predicted, a.std))}


class GatesArgs(Args):
    stats: dict[str, float | None]


class GatesResult(Result):
    passed: bool
    failures: list[str]
    metrics: dict[str, float | None]
    summary: str


def check_gates(a: GatesArgs, ctx: ToolContext) -> dict:
    from kalos.core.gates import check_gates as fn

    return jsonable(fn(dict(a.stats)))


class GPShapeArgs(ModelArgs):
    cv_spearman: float | None = Field(default=None, ge=-1.0, le=1.0)


class GPShapeResult(Result):
    report: Loose


def _fit_surrogate(X, y):
    import numpy as np

    from kalos.core.surrogate import Surrogate

    lo, hi = X.min(axis=0), X.max(axis=0)
    hi = np.where(hi > lo, hi, lo + 1.0)
    bounds = np.vstack([lo, hi])
    return Surrogate().fit(X, y, bounds=bounds), bounds


def gp_shape_report(a: GPShapeArgs, ctx: ToolContext) -> dict:
    from kalos.core.gp_shape import gp_shape_report as fn

    df, target, features = resolve_model_table(a, ctx)
    X, y = numeric_xy(df, target, features)
    s, _ = _fit_surrogate(X, y)
    return {"report": jsonable(fn(s, X, y, feature_names=features, cv_spearman=a.cv_spearman))}


# bo-next-batch ------------------------------------------------------------------


class SurrogateFitArgs(ModelArgs):
    pass


class FitPrediction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    mean: float | None
    std: float | None


class SurrogateFitResult(Result):
    n_rows: int
    features: list[str]
    in_sample: list[FitPrediction]


def surrogate_fit(a: SurrogateFitArgs, ctx: ToolContext) -> dict:
    import numpy as np

    df, target, features = resolve_model_table(a, ctx)
    X, y = numeric_xy(df, target, features)
    s, _ = _fit_surrogate(X, y)
    mu, sd = s.posterior(X)
    mu, sd = np.asarray(mu, float).reshape(-1), np.asarray(sd, float).reshape(-1)
    return {"n_rows": len(y), "features": features,
            "in_sample": [{"mean": jsonable(m), "std": jsonable(v)} for m, v in zip(mu, sd)]}


class ProposeArgs(ModelArgs):
    q: int = Field(default=4, ge=1, le=24)
    seed: int = Field(default=0, ge=0, le=2**32 - 1)
    objective: Literal["maximise", "minimise"] = "maximise"
    pending: list[dict[str, float]] = Field(default_factory=list, max_length=500)


class Proposal(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    recipe: dict[str, float]
    predicted_mean: float | None
    predicted_std: float | None


class ProposeResult(Result):
    proposals: list[Proposal]
    requested: int
    near_duplicates_replaced: int
    min_separation: float
    objective: Literal["maximise", "minimise"]
    pending_used: int


def propose_batch(a: ProposeArgs, ctx: ToolContext) -> dict:
    """Continuous factors only, bounds = the observed range of each factor (the
    search box; widening it is a listed gap of search-box-definition). For a
    campaign, its awaiting runs are passed as pending."""
    import numpy as np

    from kalos.core.optimize import propose_batch as fn

    df, target, features = resolve_model_table(a, ctx)
    X, y = numeric_xy(df, target, features)
    sign = 1.0 if a.objective == "maximise" else -1.0
    s, bounds = _fit_surrogate(X, sign * y)
    pend_rows = list(a.pending)
    if a.campaign_id is not None and not pend_rows and ctx.store is not None:
        pend_rows = ctx.store.awaiting_recipes(tenant=ctx.tenant, campaign_id=a.campaign_id)
    pend = [[float(r[f]) for f in features] for r in pend_rows
            if all(isinstance(r.get(f), (int, float)) for f in features)]
    pending = np.asarray(pend, float) if pend else None
    batch = fn(s, bounds, q=a.q, seed=a.seed, pending=pending)
    mu, sd = s.posterior(batch.X)
    mu = sign * np.asarray(mu, float).reshape(-1)
    sd = np.asarray(sd, float).reshape(-1)
    props = [{"recipe": {f: float(v) for f, v in zip(features, row)},
              "predicted_mean": jsonable(m), "predicted_std": jsonable(v)}
             for row, m, v in zip(np.asarray(batch.X), mu, sd)]
    return {"proposals": props, "requested": int(batch.requested),
            "near_duplicates_replaced": int(batch.near_duplicates_replaced),
            "min_separation": float(batch.min_separation), "objective": a.objective,
            "pending_used": len(pend)}


# scale-up-envelope-check ------------------------------------------------------


class ScaleTarget(Args):
    scale_L: float = Field(gt=0)
    process_params: dict[str, float]
    agitation_rpm: float | None = None
    airflow_L_per_min: float | None = None


class ScaleReadoutArgs(TableArgs):
    target_column: str = Field(max_length=256)
    process_columns: list[str] = Field(max_length=MAX_COLUMNS)
    target: ScaleTarget
    seed: int = Field(default=0, ge=0, le=2**32 - 1)


class ScaleReadoutResult(Result):
    readout: Loose


def scale_build_readout(a: ScaleReadoutArgs, ctx: ToolContext) -> dict:
    from kalos.scale.readout import TargetSpec
    from kalos.scale.readout import build_readout as fn

    df, _ = resolve_table(a, ctx)
    spec = TargetSpec(scale_L=a.target.scale_L, process_params=dict(a.target.process_params),
                      agitation_rpm=a.target.agitation_rpm,
                      airflow_L_per_min=a.target.airflow_L_per_min)
    return {"readout": jsonable(fn(df, a.target_column, list(a.process_columns), spec,
                                   seed=a.seed))}


# campaign-status / campaign-provenance-report -----------------------------------


class CampaignSummaryResult(Result):
    summary: Loose


def campaign_summary(a: CampaignArgs, ctx: ToolContext) -> dict:
    """The stored campaign summary with predictions stripped from runs: this
    output is labelled derived, and the best value it reports is measured only."""
    from kalos.portal.campaign import CampaignNotFound

    if ctx.store is None:
        raise ToolRefusal("no_campaign_store", "this call needs campaign storage")
    try:
        summary = ctx.store.summary(tenant=ctx.tenant, campaign_id=a.campaign_id)
    except CampaignNotFound as exc:
        raise ToolRefusal("not_found", "campaign not found") from exc
    summary = dict(summary)
    for key in ("pending", "runs"):
        if isinstance(summary.get(key), list):
            summary[key] = [{k: v for k, v in r.items() if k not in ("pred", "std")}
                            if isinstance(r, dict) else r for r in summary[key]]
    return {"summary": jsonable(summary)}


class StageResult(Result):
    stage: Loose


def derive_stage(a: CampaignArgs, ctx: ToolContext) -> dict:
    from kalos.portal.campaign import CampaignNotFound
    from kalos.portal.campaign_stage import stage_fields

    if ctx.store is None:
        raise ToolRefusal("no_campaign_store", "this call needs campaign storage")
    try:
        summary = ctx.store.summary(tenant=ctx.tenant, campaign_id=a.campaign_id)
        analysis = ctx.store.load_analysis(tenant=ctx.tenant, campaign_id=a.campaign_id)
    except CampaignNotFound as exc:
        raise ToolRefusal("not_found", "campaign not found") from exc
    return {"stage": jsonable(stage_fields(summary, analysis))}


class AuditArgs(Args):
    limit: int = Field(default=100, ge=1, le=500)


class AuditResult(Result):
    entries: list[Loose]


def audit_entries(a: AuditArgs, ctx: ToolContext) -> dict:
    if ctx.store is None:
        raise ToolRefusal("no_campaign_store", "this call needs campaign storage")
    return {"entries": jsonable(ctx.store.audit_entries(tenant=ctx.tenant, limit=a.limit))}


class RunRecordArgs(FileArgs):
    target: str | None = Field(default=None, max_length=256)
    seed: int = Field(default=0, ge=0, le=2**32 - 1)
    anonymize: bool = True
    profile: Profile = "bioprocess"


class RunRecordResult(Result):
    record: Loose


def run_record_build(a: RunRecordArgs, ctx: ToolContext) -> dict:
    from kalos.portal.run_record import build as fn
    from kalos.portal.uploads import parse_upload

    raw = a.raw(ctx)
    p = parse_upload(raw, filename=a.filename, sheet=a.sheet)
    n_features = max(p.df.shape[1] - (1 if a.target in p.df.columns else 0), 0)
    rec = fn(raw, a.filename, a.sheet, a.target, "", a.anonymize, a.profile, 1, a.seed,
             len(p.df), n_features, sheet_index=p.sheet_index, sheet_hash=p.sheet_hash)
    return {"record": jsonable(rec)}


# virtual-run-simulation ---------------------------------------------------------


class OdeArgs(Args):
    recipes: list[tuple[float, float, float]] = Field(min_length=1, max_length=200)
    t_end_h: float = Field(default=336.0, gt=0.0, le=10_000.0)


class SimulatedEndpoint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    mu_max: float
    ks: float
    s0: float
    final_biomass: float | None
    data_type: Literal["simulated"]


class OdeResult(Result):
    model: Literal["monod-batch-example"]
    t_end_h: float
    endpoints: list[SimulatedEndpoint]


def ode_model(a: OdeArgs, ctx: ToolContext) -> dict:
    import numpy as np

    from kalos.core.mechanistic import monod_example

    X = np.asarray([list(r) for r in a.recipes], float)
    if (X <= 0).any():
        raise ToolRefusal("bad_input", "mu_max, Ks and S0 must all be positive")
    y = monod_example(t_end=a.t_end_h).evaluate(X)
    return {"model": "monod-batch-example", "t_end_h": a.t_end_h,
            "endpoints": [{"mu_max": float(r[0]), "ks": float(r[1]), "s0": float(r[2]),
                           "final_biomass": jsonable(v), "data_type": "simulated"}
                          for r, v in zip(X, y)]}
