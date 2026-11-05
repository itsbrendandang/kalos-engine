"""The tool table: every skill tool, bound to one existing engine symbol.

Registration is an explicit, code-reviewed table (`TOOLS` below): no schema is
generated at runtime and nothing is injected into a global namespace. Each
`ToolSpec` names:

- `symbol`: the engine callable the tool is bound to (`pkg.mod:attr` or
  `pkg.mod:Class.method`), checked statically (AST, no import);
- `adapter`: the function in `kalos.skills.bindings` that calls it;
- `args_model` / `result_model`: pydantic models for every call's arguments and
  result;
- `label`: the data label of the result;
- `csv`: the result key (a list of row dicts) written as a CSV artifact for the
  skill's table output; everything else goes to one JSON artifact;
- `scope`, `side_effects`, `needs_torch`, `long_running`, `timeout_s`
  (default 1800 s) and the pyproject extras the tool needs.

The single runner that executes tools, writes artifacts and receipts is
`kalos.skills.runner`. Kalos-original code.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from . import bindings as b
from .loader import Skill, check_symbol, default_engine_root, defined_extras
from .schema import DataLabel

ADAPTER_MODULE = "kalos.skills.bindings"
DEFAULT_TIMEOUT_S = 1800


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
    # Fits a GP or similar. v1 runs every tool on an in-process worker thread,
    # which cannot be stopped; these should move to the killable job-worker
    # subprocess (kalos/portal/job_runner.py) when a run route exists.
    long_running: bool = False
    timeout_s: int = DEFAULT_TIMEOUT_S
    csv: str | None = None
    extra_extras: tuple[str, ...] = ()
    description: str = ""

    @property
    def requires_extras(self) -> frozenset[str]:
        out = set(self.extra_extras)
        if self.needs_torch:
            out.add("ml")
        if self.symbol.startswith("kalos.portal."):
            out.add("portal")
        return frozenset(out)

    @property
    def args_seed_default(self) -> int | None:
        f = self.args_model.model_fields.get("seed")
        return None if f is None else f.default


def _t(name, symbol, adapter, args, result, label, *, torch=False, long=False, timeout=None,
       csv=None, extras=(), desc="") -> ToolSpec:
    return ToolSpec(name=name, symbol=symbol, adapter=adapter, args_model=args,
                    result_model=result, label=label, needs_torch=torch, long_running=long,
                    timeout_s=timeout or DEFAULT_TIMEOUT_S, csv=csv, extra_extras=tuple(extras),
                    description=desc)


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
       "apply_unit_conversions", b.TableArgs, b.UnitConversionsResult, "derived", csv="rows"),
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
    _t("engine.anonymizer", "kalos.data.anonymizer:Anonymizer.anonymize_meta",
       "anonymise_table", b.AnonymiseArgs, b.AnonymiseResult, "derived", csv="rows",
       extras=("portal",)),
    _t("engine.anonymize_parsed", "kalos.portal.analysis_parts.anonymize:anonymize_parsed",
       "anonymize_parsed", b.FileArgs, b.AnonymiseParsedResult, "derived"),
    # trust-report
    _t("engine.grouped_cv_report", "kalos.core.evaluation:grouped_cv_report", "grouped_cv_report",
       b.CVArgs, b.CVResult, "derived", torch=True, long=True),
    _t("engine.interval_calibration", "kalos.core.evaluation:interval_calibration",
       "interval_calibration", b.CalibrationArgs, b.CalibrationResult, "derived", torch=True),
    _t("engine.check_gates", "kalos.core.gates:check_gates", "check_gates",
       b.GatesArgs, b.GatesResult, "derived"),
    _t("engine.gp_shape_report", "kalos.core.gp_shape:gp_shape_report", "gp_shape_report",
       b.GPShapeArgs, b.GPShapeResult, "derived", torch=True, long=True),
    # bo-next-batch
    _t("engine.surrogate_fit", "kalos.core.surrogate:Surrogate", "surrogate_fit",
       b.SurrogateFitArgs, b.SurrogateFitResult, "predicted", torch=True, long=True),
    _t("engine.propose_batch", "kalos.core.optimize:propose_batch", "propose_batch",
       b.ProposeArgs, b.ProposeResult, "predicted", torch=True, long=True),
    # scale-up-envelope-check
    _t("engine.scale_build_readout", "kalos.scale.readout:build_readout", "scale_build_readout",
       b.ScaleReadoutArgs, b.ScaleReadoutResult, "predicted", torch=True, long=True),
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


def artifact_targets(skill: Skill, spec: ToolSpec) -> tuple[str | None, str | None]:
    """(json output name, csv output name) this tool writes for this skill.
    JSON goes to the skill's one `json` output with the tool's label; the CSV
    (when `spec.csv`) to its one `table`/`csv` output with that label."""
    m = skill.meta
    js = [o.name for o in m.outputs if o.kind == "json" and o.label == spec.label]
    tb = [o.name for o in m.outputs if o.kind in ("table", "csv") and o.label == spec.label]
    return (js[0] if len(js) == 1 else None,
            (tb[0] if len(tb) == 1 else None) if spec.csv else None)


def check_tools(skills: list[Skill] | None = None, engine_root: Path | None = None) -> list[str]:
    """Static checks, no engine imports:

    - every tool's engine symbol resolves and its adapter exists;
    - every skill tool is registered, carries a label the skill declares, and
      maps to exactly one declared output per artifact;
    - a locked skill's entrypoint is the symbol of one of its own tools;
    - tool extras are declared in the skill's `requires_extras`, and every
      declared extra exists in pyproject.toml;
    - a tool that takes a `seed` needs a `seed` int parameter in the skill with
      the same default.
    """
    root = Path(engine_root) if engine_root is not None else default_engine_root()
    problems: list[str] = []
    for t in TOOLS.values():
        problems += [f"{t.name}: {p}" for p in check_symbol(t.symbol, root)]
        if not callable(getattr(b, t.adapter, None)):
            problems.append(f"{t.name}: adapter {ADAPTER_MODULE}:{t.adapter} missing")
    extras = defined_extras(root)
    for s in skills or []:
        m = s.meta
        for x in m.requires_extras:
            if x not in extras:
                problems.append(f"{s.slug}: requires_extras names {x!r}, not an extra in "
                                "pyproject.toml")
        symbols = set()
        for name in m.tools:
            spec = TOOLS.get(name)
            if spec is None:
                problems.append(f"{s.slug}: tool {name} is not registered")
                continue
            symbols.add(spec.symbol)
            if spec.label not in m.output_labels:
                problems.append(f"{s.slug}: tool {name} emits {spec.label}, skill declares "
                                f"{sorted(m.output_labels)}")
                continue
            js, tb = artifact_targets(s, spec)
            if js is None:
                problems.append(f"{s.slug}: tool {name} has no single json output labelled "
                                f"{spec.label}")
            if spec.csv and tb is None:
                problems.append(f"{s.slug}: tool {name} writes a table but the skill has no "
                                f"single table output labelled {spec.label}")
            missing = spec.requires_extras - set(m.requires_extras)
            if missing:
                problems.append(f"{s.slug}: tool {name} needs extras {sorted(missing)}")
            sd = spec.args_seed_default
            if sd is not None:
                param = next((p for p in m.parameters if p.name == "seed"), None)
                if param is None or param.type != "int":
                    problems.append(f"{s.slug}: tool {name} takes a seed; declare an int "
                                    "'seed' parameter")
                elif param.default != sd:
                    problems.append(f"{s.slug}: seed default {param.default} != tool default {sd}")
        if m.entrypoint is not None:
            problems += [f"{s.slug}: entrypoint {p}" for p in check_symbol(m.entrypoint, root)]
            if m.entrypoint not in symbols:
                problems.append(f"{s.slug}: entrypoint {m.entrypoint} is not the symbol of one "
                                "of its tools")
        if m.kind == "guided":
            for name in m.tools:
                spec = TOOLS.get(name)
                if spec is not None and (spec.scope != "read" or spec.side_effects != "none"):
                    problems.append(f"{s.slug}: guided skills may only use read-only tools")
    return problems


__all__ = [
    "DEFAULT_TIMEOUT_S",
    "TOOLS",
    "ToolSpec",
    "artifact_targets",
    "check_tools",
    "get_tool",
]
