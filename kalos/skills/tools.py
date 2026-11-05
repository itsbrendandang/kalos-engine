"""The tool table and the one invocation path for skill tools.

Registration is an explicit, code-reviewed table (`TOOLS` below): no schema is
generated at runtime and nothing is injected into a global namespace. Each
`ToolSpec` names:

- `symbol`: the engine function or class the tool is bound to, checked
  statically (AST, no import) by `check_tools`;
- `adapter`: the function in `kalos.skills.bindings` that calls it;
- `args_model` / `result_model`: pydantic models that validate every call's
  arguments and every result;
- `label`: the data label of the result (measured, predicted, simulated,
  synthetic, derived, metadata);
- `scope`, `side_effects`, `needs_torch`, `timeout_s`.

`invoke(skill, tool, args, ctx)` is the only way to run a tool:

1. the skill must list the tool, must not be `planned`, and the caller must
   hold the skill's and the tool's scope;
2. `args_model.model_validate(args)`;
3. the adapter runs under a timeout;
4. `result_model.model_validate(data)`; the tool's label must be one of the
   skill's declared output labels (a predicted result can never come out of a
   skill that only declares derived outputs, and so on);
5. a `SkillRunReceipt` is written for every outcome: ok, refused (engine or
   adapter refusal, with the reason) or error.

Kalos-original code.
"""
from __future__ import annotations

import concurrent.futures as cf
import time
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from . import bindings as b
from .bindings import ToolContext, ToolRefusal
from .loader import Skill, check_symbol, default_engine_root
from .receipts import (
    OutputRecord,
    ReceiptStore,
    Refusal,
    SkillRunReceipt,
    engine_build_sha,
    new_receipt_id,
    sha256_json,
    table_sha256,
    utcnow,
)
from .schema import DataLabel

ADAPTER_MODULE = "kalos.skills.bindings"
# Exception class names (anywhere in the MRO) that mean "the engine refused this
# input", matched by name so this module never imports the classes (and torch).
REFUSAL_CLASS_NAMES = frozenset({"UploadRejected", "CampaignError", "FitError", "ToolRefusal"})


@dataclass(frozen=True)
class ToolSpec:
    name: str
    symbol: str
    adapter: str
    args_model: type[BaseModel]
    result_model: type[BaseModel]
    label: DataLabel
    scope: Literal["read", "write"] = "read"
    side_effects: Literal["none", "writes-campaign", "external"] = "none"
    needs_torch: bool = False
    timeout_s: int = 120
    description: str = ""
    output: str = ""  # which declared skill output this result feeds


def _t(name, symbol, adapter, args, result, label, *, torch=False, timeout=120, desc="",
       output="") -> ToolSpec:
    return ToolSpec(name=name, symbol=symbol, adapter=adapter, args_model=args,
                    result_model=result, label=label, needs_torch=torch, timeout_s=timeout,
                    description=desc, output=output)


TOOLS: dict[str, ToolSpec] = {t.name: t for t in [
    # column-role-inference
    _t("engine.guess_role", "kalos.normalize.synonyms:guess_role", "guess_role",
       b.GuessRoleArgs, b.GuessRoleResult, "derived",
       desc="Role guess for one header from its name, numeric-ness and parse rate."),
    _t("engine.offline_plan", "kalos.normalize.llm:offline_plan", "offline_plan",
       b.OfflinePlanArgs, b.OfflinePlanResult, "derived",
       desc="Deterministic per-column plan (no LLM)."),
    _t("engine.column_provenance", "kalos.portal.validate:column_provenance", "column_provenance",
       b.ColumnProvenanceArgs, b.ColumnProvenanceResult, "derived",
       desc="Why each column is a target, feature, identifier or ignored."),
    # replicate-noise-estimate
    _t("engine.noise_report", "kalos.core.replicates:noise_report", "noise_report",
       b.NoiseArgs, b.NoiseReportResult, "derived"),
    _t("engine.estimate_noise_floor", "kalos.core.replicates:estimate_noise_floor",
       "estimate_noise_floor", b.NoiseArgs, b.NoiseFloorResult, "derived"),
    _t("engine.heteroscedasticity_report", "kalos.core.replicates:heteroscedasticity_report",
       "heteroscedasticity_report", b.NoiseArgs, b.NoiseReportResult, "derived"),
    # run-sheet-checks
    _t("engine.parse_upload", "kalos.portal.uploads:parse_upload", "parse_upload",
       b.FileArgs, b.ParseUploadResult, "derived",
       desc="Parse a CSV/TSV/XLSX run sheet; returns its shape, never its values."),
    _t("engine.validate_frame", "kalos.validation.runner:validate_frame", "validate_frame",
       b.ValidateFrameArgs, b.CheckReportResult, "derived",
       desc="The eleven run-sheet checks; strict mode refuses on any error finding."),
    # target-readiness-check
    _t("engine.assess_target_variation", "kalos.validation.target:assess_target_variation",
       "assess_target_variation", b.TargetValuesArgs, b.TargetVariationResult, "derived"),
    # unit-normalisation
    _t("engine.parse_value", "kalos.normalize.units:parse_value", "parse_value",
       b.ParseValueArgs, b.ParseValueResult, "derived"),
    _t("engine.convert_unit", "kalos.normalize.units:convert", "convert_unit",
       b.ConvertArgs, b.ConvertResult, "derived"),
    _t("engine.apply_unit_conversions", "kalos.validation.runner:apply_unit_conversions",
       "apply_unit_conversions", b.TableArgs, b.UnitConversionsResult, "derived"),
    # search-box-definition
    _t("engine.build_search_box", "kalos.domains.profile:build_design_space", "build_search_box",
       b.SearchBoxArgs, b.SearchBoxResult, "derived"),
    _t("engine.validate_constraint", "kalos.portal.analysis_parts.refusals:_validate_constraint",
       "validate_constraint", b.ConstraintArgs, b.ConstraintResult, "derived"),
    # driver-ranking
    _t("engine.bootstrap_spearman", "kalos.core.drivers:bootstrap_spearman",
       "bootstrap_spearman", b.DriverArgs, b.BootstrapResult, "derived"),
    _t("engine.rank_drivers", "kalos.core.drivers:rank_drivers", "rank_drivers",
       b.RankDriversArgs, b.RankDriversResult, "derived"),
    # identifier-anonymisation
    _t("engine.anonymizer", "kalos.data.anonymizer:Anonymizer", "anonymise_table",
       b.AnonymiseArgs, b.AnonymiseResult, "derived"),
    _t("engine.anonymize_parsed", "kalos.portal.analysis_parts.anonymize:anonymize_parsed",
       "anonymize_parsed", b.FileArgs, b.AnonymiseParsedResult, "derived"),
    # trust-report
    _t("engine.grouped_cv_report", "kalos.core.evaluation:grouped_cv_report", "grouped_cv_report",
       b.CVArgs, b.CVResult, "derived", torch=True, timeout=600),
    _t("engine.interval_calibration", "kalos.core.evaluation:interval_calibration",
       "interval_calibration", b.CalibrationArgs, b.CalibrationResult, "derived", torch=True),
    _t("engine.check_gates", "kalos.core.gates:check_gates", "check_gates",
       b.GatesArgs, b.GatesResult, "derived"),
    _t("engine.gp_shape_report", "kalos.core.gp_shape:gp_shape_report", "gp_shape_report",
       b.GPShapeArgs, b.GPShapeResult, "derived", torch=True, timeout=300),
    # bo-next-batch
    _t("engine.surrogate_fit", "kalos.core.surrogate:Surrogate", "surrogate_fit",
       b.SurrogateFitArgs, b.SurrogateFitResult, "predicted", torch=True, timeout=300),
    _t("engine.propose_batch", "kalos.core.optimize:propose_batch", "propose_batch",
       b.ProposeArgs, b.ProposeResult, "predicted", torch=True, timeout=600),
    # scale-up-envelope-check
    _t("engine.scale_build_readout", "kalos.scale.readout:build_readout", "scale_build_readout",
       b.ScaleReadoutArgs, b.ScaleReadoutResult, "predicted", torch=True, timeout=600),
    # campaign-status / campaign-provenance-report
    _t("engine.campaign_summary", "kalos.portal.campaign:CampaignStore", "campaign_summary",
       b.CampaignArgs, b.CampaignSummaryResult, "derived"),
    _t("engine.derive_stage", "kalos.portal.campaign_stage:stage_fields", "derive_stage",
       b.CampaignArgs, b.StageResult, "derived"),
    _t("engine.audit_entries", "kalos.portal.campaign:CampaignStore", "audit_entries",
       b.AuditArgs, b.AuditResult, "derived"),
    _t("engine.run_record_build", "kalos.portal.run_record:build", "run_record_build",
       b.RunRecordArgs, b.RunRecordResult, "derived"),
    # virtual-run-simulation
    _t("engine.ode_model", "kalos.core.mechanistic:monod_example", "ode_model",
       b.OdeArgs, b.OdeResult, "simulated"),
]}


def get_tool(name: str) -> ToolSpec | None:
    return TOOLS.get(name)


def check_tools(skills: list[Skill] | None = None) -> list[str]:
    """Static checks, no imports: every tool's engine symbol resolves, its adapter
    exists, and every skill's `tools[]` names a registered tool whose label is one
    of the skill's declared output labels."""
    root = default_engine_root()
    problems: list[str] = []
    for t in TOOLS.values():
        problems += [f"{t.name}: {p}" for p in check_symbol(t.symbol, root)]
        if not callable(getattr(b, t.adapter, None)):
            problems.append(f"{t.name}: adapter {ADAPTER_MODULE}:{t.adapter} missing")
    for s in skills or []:
        for name in s.meta.tools:
            spec = TOOLS.get(name)
            if spec is None:
                problems.append(f"{s.slug}: tool {name} is not registered")
            elif spec.label not in s.meta.output_labels:
                problems.append(
                    f"{s.slug}: tool {name} emits {spec.label}, skill declares "
                    f"{sorted(s.meta.output_labels)}")
    return problems


# --- invocation --------------------------------------------------------------- #


class ToolCallError(Exception):
    """A call that could not run (unknown tool, scope, bad args) or failed inside
    the engine. `receipt` is set when the call got far enough to be receipted."""

    def __init__(self, code: str, message: str, receipt: SkillRunReceipt | None = None):
        super().__init__(message)
        self.code = code
        self.receipt = receipt


class ToolOutcome(BaseModel):
    outcome: Literal["ok", "refused"]
    label: DataLabel
    data: Any = None
    refusal: Refusal | None = None
    receipt: SkillRunReceipt


def _is_refusal(exc: BaseException) -> bool:
    return any(c.__name__ in REFUSAL_CLASS_NAMES for c in type(exc).__mro__)


def _refusal_code(exc: BaseException) -> str:
    code = getattr(exc, "code", None)
    return code if isinstance(code, str) and code else type(exc).__name__


def guard_measured_write(label: str) -> None:
    """Honesty invariant: only `measured` values may be written into a campaign's
    measured data. Called by any write path before it stores a skill result."""
    if label != "measured":
        raise ToolRefusal("label_mismatch",
                          f"a {label} result can never be written as measured data")


def _run_with_timeout(fn, args, ctx, timeout_s: int):
    # A thread, not a process: a timed-out engine call is abandoned (its result
    # discarded and receipted as a timeout), not killed. Long fits belong on the
    # job runner once a run route exists.
    ex = cf.ThreadPoolExecutor(max_workers=1, thread_name_prefix="kalos-skill-tool")
    try:
        fut = ex.submit(fn, args, ctx)
        return fut.result(timeout=timeout_s)
    finally:
        ex.shutdown(wait=False, cancel_futures=True)


def invoke(skill: Skill, tool: str, args: dict[str, Any], ctx: ToolContext, *,
           campaign_id: str | None = None, timeout_s: int | None = None) -> ToolOutcome:
    """Run one tool for one skill. Returns a receipted `ToolOutcome` for ok and
    refused calls; raises `ToolCallError` otherwise (receipted when it got past
    argument validation)."""
    m = skill.meta
    if m.status == "planned":
        raise ToolCallError("planned", f"skill {m.slug} is planned; it has no tools yet")
    if tool not in m.tools:
        raise ToolCallError("tool_not_in_skill", f"{tool} is not a tool of {m.slug}")
    spec = TOOLS.get(tool)
    if spec is None:
        raise ToolCallError("unknown_tool", f"{tool} is not registered")
    needed = {m.scope, spec.scope}
    if not needed <= set(ctx.scopes):
        raise ToolCallError("forbidden", f"requires scope(s) {sorted(needed)}")
    try:
        parsed = spec.args_model.model_validate(args)
    except ValidationError as exc:
        raise ToolCallError("invalid_args", f"invalid arguments for {tool}: "
                            f"{exc.error_count()} error(s)") from exc

    started, t0 = utcnow(), time.monotonic()
    dumped = parsed.model_dump(mode="json")
    rows = dumped.get("rows") if isinstance(dumped.get("rows"), list) else None
    input_sha = table_sha256(rows) if rows else None
    cid = campaign_id or dumped.get("campaign_id")
    seed = dumped.get("seed") if isinstance(dumped.get("seed"), int) else None

    def receipt(outcome, outputs=(), refusal=None) -> SkillRunReceipt:
        r = SkillRunReceipt(
            receipt_id=new_receipt_id(), skill_slug=m.slug, skill_version=m.version,
            skill_content_hash=skill.content_hash, tool=tool, symbol=spec.symbol,
            args_sha256=sha256_json(dumped), input_sha256=input_sha,
            input_rows=len(rows) if rows else None, seed=seed,
            build_sha=engine_build_sha(), outcome=outcome, outputs=list(outputs),
            refusal=refusal, tenant=ctx.tenant, subject=ctx.subject,
            campaign_id=cid if isinstance(cid, str) else None, started_at=started,
            finished_at=utcnow(), duration_ms=int((time.monotonic() - t0) * 1000))
        store = ctx.receipts if ctx.receipts is not None else ReceiptStore()
        store.write(r)
        return r

    adapter = getattr(b, spec.adapter)
    try:
        data = _run_with_timeout(adapter, parsed, ctx, timeout_s or spec.timeout_s)
    except cf.TimeoutError:
        r = receipt("error", refusal=Refusal(code="timeout", reason=f"no result within "
                                             f"{timeout_s or spec.timeout_s} s"))
        raise ToolCallError("timeout", f"{tool} timed out", r) from None
    except Exception as exc:  # noqa: BLE001 - classified below, always receipted
        if _is_refusal(exc):
            reason = getattr(exc, "reason", None) or getattr(exc, "message", None) or str(exc)
            ref = Refusal(code=_refusal_code(exc), reason=str(reason)[:2000])
            return ToolOutcome(outcome="refused", label=spec.label, refusal=ref,
                               receipt=receipt("refused", refusal=ref))
        r = receipt("error", refusal=Refusal(code="engine_error", reason=type(exc).__name__))
        raise ToolCallError("engine_error", f"{tool} failed: {type(exc).__name__}", r) from exc

    try:
        result = spec.result_model.model_validate(data)
    except ValidationError as exc:
        r = receipt("error", refusal=Refusal(code="invalid_result",
                                             reason=f"{exc.error_count()} error(s)"))
        raise ToolCallError("invalid_result", f"{tool} returned an invalid result", r) from exc
    if spec.label not in m.output_labels:
        r = receipt("error", refusal=Refusal(code="label_mismatch",
                                             reason=f"{spec.label} not declared by {m.slug}"))
        raise ToolCallError("label_mismatch", "result label contradicts the skill", r)
    out = result.model_dump(mode="json")
    output_name = spec.output or next(
        (o.name for o in m.outputs if o.label == spec.label), spec.name)
    rec = receipt("ok", outputs=[OutputRecord(name=output_name, label=spec.label,
                                              sha256=sha256_json(out))])
    return ToolOutcome(outcome="ok", label=spec.label, data=out, receipt=rec)


__all__ = [
    "TOOLS",
    "ToolCallError",
    "ToolContext",
    "ToolOutcome",
    "ToolRefusal",
    "ToolSpec",
    "check_tools",
    "get_tool",
    "guard_measured_write",
    "invoke",
]
