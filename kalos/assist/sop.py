"""SOP drafts from a campaign's own records. Template-driven; no model writes v1.

A draft is a markdown procedure built from one source of recipes (started
runs, proposals in the published analysis, or every run of one round) and a
template (`batch_recipe`, `media_prep`, `custom`). Three rules hold for
every version of every draft:

1. Every number in the draft comes from the campaign record and is listed in
   `fields_from_record` with the source it was read from (`run:<id>`,
   `proposal:<round>.<index>`, `analysis:<section>`, `sheet:<sha256>`).
   Predicted values are labelled "Predicted" and never shown as measured;
   measured results are labelled "Measured".
2. Anything the lab has to decide (safety, equipment ids, who prepares and
   checks, limits, lots, quantities the record lacks) is a named placeholder
   written literally as `[[TO FILL: <name>]]` and listed in `placeholders`.
   Nothing here invents one of those.
3. The first line is always the header stating the draft is not reviewed and
   not an approved procedure. A revision (`revise`, the only step that uses
   a model) may change prose only: the header, the record values, the
   numbers, and the placeholder tokens are checked after the model answers
   and the revision is refused if any of them moved. `fill` replaces
   placeholder tokens with the lab's own text and nothing else.

Storage is append-only under `<assist tenant dir>/sop/<draft id>/v<N>.json`
(plus `v<N>.md`). A version file is never overwritten.
"""
from __future__ import annotations

import hashlib
import json
import re
import secrets
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from kalos.assist.llm import ChatModel
from kalos.assist.paths import tenant_dir
from kalos.assist.words import banned_in, unsupported_claims
from kalos.portal.campaign import CampaignStore

Template = Literal["batch_recipe", "media_prep", "custom"]
TEMPLATES: tuple[Template, ...] = ("batch_recipe", "media_prep", "custom")

DRAFT_ID_RE = re.compile(r"sop_[0-9a-f]{32}")
PLACEHOLDER_RE = re.compile(r"\[\[TO FILL: ([a-z][a-z0-9_]*)\]\]")
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
HEADER_PREFIX = "> Draft prepared with Kalos from campaign records on "
HEADER_SUFFIX = ". Not reviewed. Not an approved procedure."
MAX_FILL_CHARS = 500
MAX_RECIPES = 64

SECTIONS: tuple[str, ...] = (
    "purpose", "scope", "materials", "equipment", "parameters", "procedure",
    "in_process_checks", "acceptance_criteria", "deviations", "revision",
)
_SECTION_TITLES = {
    "purpose": "Purpose",
    "scope": "Scope",
    "materials": "Materials",
    "equipment": "Equipment",
    "parameters": "Settings",
    "procedure": "Procedure",
    "in_process_checks": "In-process checks",
    "acceptance_criteria": "Acceptance criteria",
    "deviations": "Deviations and notes",
    "revision": "Revision table",
}
_TEMPLATE_TITLES = {
    "batch_recipe": "Run recipe",
    "media_prep": "Media preparation",
    "custom": "Procedure",
}


class SopError(ValueError):
    """A request the draft code cannot act on. `code` names why."""

    code = "invalid_request"


class SourceError(SopError):
    """The source names no run or proposal this campaign holds."""

    code = "invalid_source"


class RevisionRefused(SopError):
    """The model's edit changed something a revision may not change."""

    code = "revision_refused"

    def __init__(self, reasons: list[str]) -> None:
        super().__init__("; ".join(reasons))
        self.reasons = list(reasons)


class UnknownPlaceholder(SopError):
    """`fill` named a placeholder the draft does not hold (any more)."""

    code = "unknown_placeholder"

    def __init__(self, names: list[str]) -> None:
        super().__init__("unknown placeholder")
        self.names = list(names)


class InvalidValue(SopError):
    code = "invalid_value"


class VersionConflict(SopError):
    """Two writers raced for the same version number; nothing was written."""

    code = "conflict"


@dataclass
class Draft:
    draft_id: str
    version: int
    title: str
    template: str
    campaign_id: str
    created_at: str
    created_by: str
    sha256: str
    model: str | None
    source_ids: list[str]
    markdown: str
    fields_from_record: list[dict[str, Any]] = field(default_factory=list)
    placeholders: list[dict[str, Any]] = field(default_factory=list)
    filled: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Draft:
        return cls(**{k: data[k] for k in cls.__dataclass_fields__})


# --- the record ---------------------------------------------------------------- #


@dataclass(frozen=True)
class _Record:
    """One recipe to write into the draft, with where it came from."""

    source_id: str
    label: str
    recipe: dict[str, Any]
    pred: Any
    std: Any
    result: Any  # None unless measured


def _label(i: int) -> str:
    """A, B, ..., Z, AA, AB, ...: recipe labels without digits, so a label is
    never mistaken for a value."""
    out = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        out = chr(ord("A") + r) + out
    return out


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _fmt(value: Any) -> str:
    """A record value as text. Numbers render as JSON does (6.0 stays 6.0), so
    the text in the draft is the text in `fields_from_record`."""
    if _is_number(value):
        return json.dumps(value)
    return str(value)


def _int_list(values: Any) -> list[int] | None:
    if not isinstance(values, list) or not values:
        return None
    if not all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in values):
        return None
    return list(values)


def resolve_source(
    source: dict[str, Any], state: dict[str, Any] | None, analysis: dict[str, Any] | None,
) -> list[_Record]:
    """The records `source` names. Exactly one key of `run_ids`,
    `proposal_index`, `round`. Raises `SourceError` when a name resolves to
    nothing the campaign holds (the message never echoes the input)."""
    if not isinstance(source, dict) or len(source) != 1:
        raise SourceError("source must hold exactly one of run_ids, proposal_index, round")
    key = next(iter(source))
    pending = [r for r in (state or {}).get("pending") or [] if isinstance(r, dict)]
    records: list[_Record] = []

    if key == "run_ids":
        ids = source[key]
        if not isinstance(ids, list) or not ids or not all(isinstance(i, str) for i in ids):
            raise SourceError("run_ids must be a non-empty list of run ids")
        if len(set(ids)) != len(ids):
            raise SourceError("run_ids holds a repeated id")
        by_id = {str(r.get("id")): r for r in pending}
        if any(i not in by_id for i in ids):
            raise SourceError("one or more run ids are not started runs of this campaign")
        runs = [by_id[i] for i in ids]
    elif key == "round":
        rnd = source[key]
        if not isinstance(rnd, int) or isinstance(rnd, bool) or rnd < 0:
            raise SourceError("round must be a whole number")
        runs = [r for r in pending if r.get("round") == rnd]
        if not runs:
            raise SourceError("no started run of this campaign belongs to that round")
    elif key == "proposal_index":
        idx = _int_list(source[key])
        if idx is None:
            raise SourceError("proposal_index must be a non-empty list of whole numbers")
        if len(set(idx)) != len(idx):
            raise SourceError("proposal_index holds a repeated index")
        proposals = (analysis or {}).get("proposals") or []
        if not isinstance(proposals, list) or any(i >= len(proposals) for i in idx):
            raise SourceError("one or more proposed-recipe positions are outside the published list")
        rnd = int((state or {}).get("round") or 0)
        for n, i in enumerate(idx):
            p = proposals[i]
            if not isinstance(p, dict) or not isinstance(p.get("recipe"), dict):
                raise SourceError("the published proposal holds no recipe")
            records.append(_Record(
                source_id=f"proposal:{rnd}.{i}", label=_label(n), recipe=dict(p["recipe"]),
                pred=p.get("pred"), std=p.get("std"), result=None,
            ))
        return _cap(records)
    else:
        raise SourceError("source must hold exactly one of run_ids, proposal_index, round")

    for n, run in enumerate(runs):
        recipe = run.get("recipe")
        if not isinstance(recipe, dict):
            raise SourceError("the run holds no recipe")
        records.append(_Record(
            source_id=f"run:{run['id']}", label=_label(n), recipe=dict(recipe),
            pred=run.get("pred"), std=run.get("std"), result=run.get("result"),
        ))
    return _cap(records)


def _cap(records: list[_Record]) -> list[_Record]:
    if len(records) > MAX_RECIPES:
        raise SourceError(f"a draft covers at most {MAX_RECIPES} recipes")
    return records


# --- building the draft -------------------------------------------------------- #


class _Builder:
    """Collects the markdown, the sourced fields, and the placeholders while
    the sections are written, so none of the three can disagree."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.fields: list[dict[str, Any]] = []
        self.placeholders: list[dict[str, Any]] = []
        self._seen: set[str] = set()

    def value(self, section: str, name: str, value: Any, source_id: str) -> str:
        text = _fmt(value)
        self.fields.append({"section": section, "field": name, "value": text, "source_id": source_id})
        return text

    def fill(self, section: str, name: str, why: str) -> str:
        if name not in self._seen:
            self._seen.add(name)
            self.placeholders.append({"name": name, "section": section, "why": why})
        return f"[[TO FILL: {name}]]"

    def section(self, key: str) -> None:
        self.lines.extend(["", f"## {_SECTION_TITLES[key]}", ""])

    def line(self, text: str = "") -> None:
        self.lines.append(text)


def _target_source(
    state: dict[str, Any] | None, analysis: dict[str, Any] | None, records: list[_Record],
) -> str:
    """Where the target name was read from: the sheet when its hash is known,
    else the analysis, else the first run (runs are measured in the target)."""
    from kalos.assist.index import input_sha256

    sha = input_sha256(analysis)
    if not sha:
        pending = [p for p in (state or {}).get("pending") or [] if isinstance(p, dict)]
        for p in pending:
            ap = p.get("analysis_provenance")
            if isinstance(ap, dict) and ap.get("input_sha256"):
                sha = ap["input_sha256"]
                break
    if isinstance(sha, str) and re.fullmatch(r"[0-9a-f]{64}", sha):
        return f"sheet:{sha}"
    if (analysis or {}).get("target"):
        return "analysis:target"
    return records[0].source_id


def _constraint(analysis: dict[str, Any] | None) -> tuple[str, Any] | None:
    block = (analysis or {}).get("constraint")
    if not isinstance(block, dict):
        return None
    column, floor = block.get("column"), block.get("floor")
    if isinstance(column, str) and column and _is_number(floor):
        return column, floor
    return None


def build_markdown(
    *, title: str, template: Template, records: list[_Record], target: str | None,
    target_source: str, analysis: dict[str, Any] | None, created_date: str,
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    """The v1 markdown and its two ledgers. Pure: same input, same text."""
    b = _Builder()
    b.line(f"{HEADER_PREFIX}{created_date}{HEADER_SUFFIX}")
    b.line()
    b.line(f"# {title}")

    features: list[str] = []
    for r in records:
        for k in r.recipe:
            if k not in features:
                features.append(k)
    ranges = (analysis or {}).get("feature_ranges")
    ranges = ranges if isinstance(ranges, dict) else {}
    target_text = b.value("purpose", "target", target, target_source) if target else None

    # Purpose
    b.section("purpose")
    noun = {"batch_recipe": "run", "media_prep": "medium", "custom": "procedure"}[template]
    if target_text:
        b.line(
            f"This draft describes how to prepare each {noun} listed below and record "
            f"`{target_text}` for it, using the settings held in the campaign record."
        )
    else:
        b.line(f"This draft describes how to prepare each {noun} listed below, using the "
               "settings held in the campaign record.")
    b.line("It is a starting point for the lab's own procedure, not a procedure in itself.")

    # Scope
    b.section("scope")
    b.line("Covers the recipes in the settings table, each identified by its record reference.")
    if template == "custom":
        b.line(b.fill("scope", "scope_statement", "the lab decides what this procedure covers and what it leaves out"))
    b.line("Safety: " + b.fill("scope", "safety_notes", "safety notes are not held in the campaign record"))
    b.line("Hazards: " + b.fill("scope", "hazard_flags", "hazard flags are not held in the campaign record"))

    # Materials
    b.section("materials")
    b.line("| Input | Quantity | Lot | Notes |")
    b.line("|---|---|---|---|")
    for name in features:
        b.line(
            f"| {name} | "
            + b.fill("materials", "material_quantities", "the record holds settings, not quantities to weigh or measure out")
            + " | "
            + b.fill("materials", "material_lot_numbers", "lot numbers are not held in the campaign record")
            + " | |"
        )
    b.line("| " + b.fill("materials", "other_materials", "materials beyond the recipe inputs (water, buffers, consumables) are not in the record") + " | | | |")

    # Equipment
    b.section("equipment")
    b.line("- " + b.fill("equipment", "equipment_list", "the campaign record names no equipment"))
    b.line("- Equipment ids: " + b.fill("equipment", "equipment_ids", "equipment ids are assigned by the lab"))

    # Parameters
    b.section("parameters")
    b.line("Record references:")
    b.line()
    for r in records:
        b.line(f"- Recipe {r.label}: `{r.source_id}`")
    b.line()
    b.line("| Recipe | Parameter | Value | Kind |")
    b.line("|---|---|---|---|")
    for r in records:
        for name, v in r.recipe.items():
            b.line(f"| {r.label} | {name} | {b.value('parameters', name, v, r.source_id)} | Setting |")
        if target_text and _is_number(r.pred):
            b.line(f"| {r.label} | {target_text} | {b.value('parameters', 'predicted', r.pred, r.source_id)} | Predicted |")
        if target_text and _is_number(r.std):
            b.line(f"| {r.label} | {target_text} | {b.value('parameters', 'predicted spread', r.std, r.source_id)} | Predicted spread |")
        if target_text and _is_number(r.result):
            b.line(f"| {r.label} | {target_text} | {b.value('parameters', 'measured', r.result, r.source_id)} | Measured |")
    for name in features:
        rng = ranges.get(name)
        if isinstance(rng, list) and len(rng) == 2 and all(_is_number(x) for x in rng):
            lo = b.value("parameters", f"{name} search range low", rng[0], "analysis:feature_ranges")
            hi = b.value("parameters", f"{name} search range high", rng[1], "analysis:feature_ranges")
            b.line(f"| all | {name} | {lo} | Search range, low |")
            b.line(f"| all | {name} | {hi} | Search range, high |")
    b.line()
    b.line("Predicted values are the model's estimate at the time of the analysis, not a measurement.")

    # Procedure
    b.section("procedure")
    steps = _procedure_steps(b, template, records, target_text)
    for n, step in enumerate(steps, 1):
        b.line(f"{n}. {step}")

    # In-process checks
    b.section("in_process_checks")
    b.line("- " + b.fill("in_process_checks", "in_process_checks", "the record does not say what is checked during the run"))
    b.line("- Record the value reached for each setting next to it in the settings table.")

    # Acceptance criteria
    b.section("acceptance_criteria")
    constraint = _constraint(analysis)
    if constraint:
        col, floor = constraint
        col_text = b.value("acceptance_criteria", "constraint column", col, "analysis:constraint")
        floor_text = b.value("acceptance_criteria", "constraint floor", floor, "analysis:constraint")
        b.line(f"- The analysis was asked to keep `{col_text}` at or above {floor_text}. "
               "The lab decides whether this is an acceptance limit for the run.")
    b.line("- " + b.fill("acceptance_criteria", "acceptance_limits", "the campaign record holds no acceptance limit"))

    # Deviations
    b.section("deviations")
    b.line("- " + b.fill("deviations", "deviation_handling", "how a deviation is recorded and who is told is the lab's rule"))
    b.line("- Note anything that departed from the written steps, with the step number.")

    # Revision table
    b.section("revision")
    b.line("| Version | Date | Prepared by | Checked by | Approved by | Change |")
    b.line("|---|---|---|---|---|---|")
    b.line(
        f"| 1 | {created_date} | "
        + b.fill("revision", "prepared_by", "the person preparing the procedure is not in the record")
        + " | "
        + b.fill("revision", "checked_by", "the person checking the procedure is not in the record")
        + " | "
        + b.fill("revision", "approved_by", "the person approving the procedure is not in the record")
        + " | Draft from campaign records |"
    )
    return "\n".join(b.lines) + "\n", b.fields, b.placeholders


def _recipe_text(b: _Builder, r: _Record) -> str:
    parts = [f"{k} {b.value('procedure', k, v, r.source_id)}" for k, v in r.recipe.items()]
    return f"Recipe {r.label}: " + "; ".join(parts)


def _procedure_steps(
    b: _Builder, template: Template, records: list[_Record], target_text: str | None,
) -> list[str]:
    settings = " ".join(_recipe_text(b, r) + "." for r in records)

    def measure() -> str:
        head = f"Measure `{target_text}` as described in " if target_text else "Measure the result as described in "
        return head + b.fill(
            "procedure", "measurement_method", "the record holds results, not how they were measured"
        ) + " and record it in the settings table as Measured."
    if template == "batch_recipe":
        return [
            "Check that every material and piece of equipment listed above is at hand, and write the lot numbers into the materials table.",
            "Prepare one run per recipe. Set each input to its setting: " + settings,
            "Record the value reached for each setting before going on.",
            "Carry out the run as described in " + b.fill("procedure", "run_method", "the record does not describe how a run is carried out") + ".",
            measure(),
        ]
    if template == "media_prep":
        return [
            "Check that every material and piece of equipment listed above is at hand, and write the lot numbers into the materials table.",
            "Prepare one medium per recipe. Set each component or condition to its setting: " + settings,
            "Dissolve and mix as described in " + b.fill("procedure", "mixing_method", "the record does not describe how the medium is mixed") + ".",
            "Adjust and confirm each condition, then record the value reached for each setting.",
            "Sterilise and store as described in " + b.fill("procedure", "sterilisation_and_storage", "the record does not describe sterilisation or storage") + ".",
            measure(),
        ]
    return [
        b.fill("procedure", "step_before_settings", "the lab writes what happens before the settings are applied"),
        "Apply the settings for each recipe: " + settings,
        "Record the value reached for each setting.",
        b.fill("procedure", "step_after_settings", "the lab writes what happens after the settings are applied"),
        measure(),
    ]


# --- storage ------------------------------------------------------------------- #


def _sop_dir(store: CampaignStore, tenant: str) -> Path:
    return tenant_dir(store.state_dir, tenant) / "sop"


def _draft_dir(store: CampaignStore, tenant: str, draft_id: str) -> Path | None:
    if not isinstance(draft_id, str) or DRAFT_ID_RE.fullmatch(draft_id) is None:
        return None
    return _sop_dir(store, tenant) / draft_id


def _versions(path: Path) -> list[int]:
    if not path.is_dir():
        return []
    out = []
    for p in path.iterdir():
        m = re.fullmatch(r"v(\d+)\.json", p.name)
        if m:
            out.append(int(m.group(1)))
    return sorted(out)


def _write_version(path: Path, draft: Draft) -> None:
    """Append one version. `open(..., "x")` so an existing file is never
    overwritten; a race for the same number raises `VersionConflict`."""
    path.mkdir(parents=True, exist_ok=True)
    target = path / f"v{draft.version}.json"
    try:
        with target.open("x", encoding="utf-8") as fh:
            json.dump(draft.to_dict(), fh, indent=2, sort_keys=True)
    except FileExistsError as exc:
        raise VersionConflict("that version already exists") from exc
    with (path / f"v{draft.version}.md").open("x", encoding="utf-8") as fh:
        fh.write(draft.markdown)


def load(store: CampaignStore, tenant: str, draft_id: str, version: int | None = None) -> Draft | None:
    """One version (latest by default) of the tenant's draft, or None for a
    malformed, missing or other tenant's id (the folder is per tenant)."""
    path = _draft_dir(store, tenant, draft_id)
    if path is None:
        return None
    versions = _versions(path)
    if not versions:
        return None
    if version is None:
        version = versions[-1]
    if version not in versions:
        return None
    with (path / f"v{version}.json").open(encoding="utf-8") as fh:
        data = json.load(fh)
    return Draft.from_dict(data)


def list_drafts(store: CampaignStore, tenant: str) -> list[dict[str, Any]]:
    """The tenant's drafts, newest first (by latest version's created_at)."""
    root = _sop_dir(store, tenant)
    out: list[dict[str, Any]] = []
    if not root.is_dir():
        return out
    for path in root.iterdir():
        if DRAFT_ID_RE.fullmatch(path.name) is None:
            continue
        first = load(store, tenant, path.name, version=1)
        latest = load(store, tenant, path.name)
        if first is None or latest is None:
            continue
        out.append({
            "draft_id": latest.draft_id,
            "title": latest.title,
            "template": latest.template,
            "campaign_id": latest.campaign_id,
            "latest_version": latest.version,
            "source_ids": list(latest.source_ids),
            "created_at": first.created_at,
            "updated_at": latest.created_at,
        })
    out.sort(key=lambda d: (d["updated_at"], d["draft_id"]), reverse=True)
    return out


def _now() -> datetime:
    return datetime.now(UTC)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --- the three operations ------------------------------------------------------ #


def draft(
    store: CampaignStore, tenant: str, campaign_id: str, source: dict[str, Any],
    template: Template, title: str | None, actor: str,
) -> Draft:
    """Version 1 of a new draft from the campaign record. No model."""
    if template not in TEMPLATES:
        raise SopError("unknown template")
    state = store.get(tenant=tenant, campaign_id=campaign_id)
    analysis = store.load_analysis(tenant=tenant, campaign_id=campaign_id)
    records = resolve_source(source, state, analysis)
    target = (state or {}).get("target") or (analysis or {}).get("target")
    target = str(target) if isinstance(target, str) and target else None
    now = _now()
    created_date = now.date().isoformat()
    title = (title or "").strip() or f"{_TEMPLATE_TITLES[template]} draft"
    markdown, fields, placeholders = build_markdown(
        title=title, template=template, records=records, target=target,
        target_source=_target_source(state, analysis, records), analysis=analysis,
        created_date=created_date,
    )
    out = Draft(
        draft_id="sop_" + secrets.token_hex(16),
        version=1,
        title=title,
        template=template,
        campaign_id=campaign_id,
        created_at=now.isoformat(),
        created_by=actor,
        sha256=_sha(markdown),
        model=None,
        source_ids=[r.source_id for r in records],
        markdown=markdown,
        fields_from_record=fields,
        placeholders=placeholders,
        filled=[],
    )
    path = _draft_dir(store, tenant, out.draft_id)
    assert path is not None
    _write_version(path, out)
    return out


def _next(store: CampaignStore, tenant: str, previous: Draft, **changes: Any) -> Draft:
    data = previous.to_dict()
    data.update(changes)
    data["version"] = previous.version + 1
    data["created_at"] = _now().isoformat()
    data["sha256"] = _sha(data["markdown"])
    out = Draft.from_dict(data)
    path = _draft_dir(store, tenant, out.draft_id)
    assert path is not None
    _write_version(path, out)
    return out


_REVISE_SYSTEM = (
    "You edit the wording of a draft laboratory procedure. Return the whole document, "
    "in markdown, and nothing else. Rules you must keep:\n"
    "- Keep the first line exactly as it is.\n"
    "- Do not add, remove or change any number, value, name or table cell.\n"
    "- Keep every token of the form [[TO FILL: name]] exactly as written; do not fill, "
    "rename, remove or add one.\n"
    "- Change prose only: sentence wording, order of words, clarity.\n"
    "- Do not promise outcomes or use the words will, guarantee, prove, cause, optimal."
)


def _strip_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else ""
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip("\n") + "\n"


def check_revision(previous: Draft, revised: str) -> list[str]:
    """Why `revised` may not replace `previous.markdown`; empty when it may."""
    reasons: list[str] = []
    old, new = previous.markdown, revised
    old_header = old.split("\n", 1)[0]
    new_header = new.split("\n", 1)[0] if new else ""
    if new_header != old_header:
        reasons.append("the header line was changed or is missing")
    for entry in previous.fields_from_record:
        value = str(entry["value"])
        if new.count(value) < old.count(value):
            reasons.append(f"record value for {entry['field']!s} no longer appears as written")
    for entry in previous.filled:
        value = str(entry["value"])
        if new.count(value) < old.count(value):
            reasons.append(f"filled value for {entry['name']!s} no longer appears as written")
    # Tables are record, not prose: every row stays as written (so a value can
    # never be relabelled, e.g. a Predicted row shown as Measured).
    old_rows = Counter(ln for ln in old.splitlines() if ln.startswith("|"))
    new_rows = Counter(ln for ln in new.splitlines() if ln.startswith("|"))
    if any(new_rows[row] < n for row, n in old_rows.items()):
        reasons.append("a table row was changed or removed")
    new_numbers = set(NUMBER_RE.findall(new)) - set(NUMBER_RE.findall(old))
    if new_numbers:
        reasons.append(f"{len(new_numbers)} number(s) appear that the previous version did not hold")
    old_tokens, new_tokens = Counter(PLACEHOLDER_RE.findall(old)), Counter(PLACEHOLDER_RE.findall(new))
    for name in sorted(set(old_tokens) | set(new_tokens)):
        if new_tokens[name] < old_tokens[name]:
            reasons.append(f"placeholder {name} was removed, renamed or filled")
        elif new_tokens[name] > old_tokens[name]:
            reasons.append(f"placeholder {name} appears more often than before")
    if "[[TO FILL:" in new and new.count("[[TO FILL:") != sum(new_tokens.values()):
        reasons.append("a malformed placeholder token appears")
    banned = sorted(set(banned_in(new)))
    if banned:
        reasons.append("words the assistant never uses appear: " + ", ".join(banned))
    claims = unsupported_claims(new, [old])
    if claims:
        reasons.append("claim words appear that the previous version did not hold: " + ", ".join(claims))
    return reasons


def revise(
    store: CampaignStore, tenant: str, draft_id: str, instruction: str,
    chat_model: ChatModel, actor: str,
) -> Draft:
    """Ask the model to edit the prose of the latest version; store N+1 when
    the post-check passes. Raises `RevisionRefused` (nothing stored),
    `ModelUnavailable` (from the model), `SopError` for a missing draft."""
    previous = load(store, tenant, draft_id)
    if previous is None:
        raise SopError("draft not found")
    prompt = (
        "Instruction from the lab:\n" + instruction.strip() + "\n\n"
        "The draft:\n\n" + previous.markdown
    )
    revised = _strip_fence(chat_model.complete(_REVISE_SYSTEM, prompt))
    reasons = check_revision(previous, revised)
    if reasons:
        raise RevisionRefused(reasons)
    return _next(store, tenant, previous, markdown=revised, model=chat_model.model, created_by=actor)


def check_fill_value(value: Any) -> str:
    if not isinstance(value, str):
        raise InvalidValue("a filled value must be text")
    text = value.strip()
    if not text or len(text) > MAX_FILL_CHARS:
        raise InvalidValue(f"a filled value must be 1-{MAX_FILL_CHARS} characters")
    if "\n" in text or "\r" in text or "[[" in text or "]]" in text:
        raise InvalidValue("a filled value must be one line without [[ or ]]")
    return text


def fill(
    store: CampaignStore, tenant: str, draft_id: str, values: dict[str, Any], actor: str,
) -> Draft:
    """Replace `[[TO FILL: name]]` tokens with the lab's text. Stores N+1 with
    no model. Raises `UnknownPlaceholder`, `InvalidValue`, `SopError`."""
    previous = load(store, tenant, draft_id)
    if previous is None:
        raise SopError("draft not found")
    if not isinstance(values, dict) or not values:
        raise InvalidValue("values must name at least one placeholder")
    clean = {str(k): check_fill_value(v) for k, v in values.items()}
    open_names = {p["name"] for p in previous.placeholders}
    unknown = sorted(k for k in clean if k not in open_names)
    if unknown:
        raise UnknownPlaceholder(unknown)
    markdown = previous.markdown
    for name, text in clean.items():
        markdown = markdown.replace(f"[[TO FILL: {name}]]", text)
    by_name = {p["name"]: p for p in previous.placeholders}
    filled = list(previous.filled) + [
        {"name": name, "section": by_name[name]["section"], "value": text, "filled_by": actor}
        for name, text in clean.items()
    ]
    placeholders = [p for p in previous.placeholders if p["name"] not in clean]
    return _next(
        store, tenant, previous, markdown=markdown, model=None, created_by=actor,
        placeholders=placeholders, filled=filled,
    )


__all__ = [
    "DRAFT_ID_RE",
    "HEADER_PREFIX",
    "HEADER_SUFFIX",
    "NUMBER_RE",
    "PLACEHOLDER_RE",
    "SECTIONS",
    "TEMPLATES",
    "Draft",
    "InvalidValue",
    "RevisionRefused",
    "SopError",
    "SourceError",
    "UnknownPlaceholder",
    "VersionConflict",
    "build_markdown",
    "check_fill_value",
    "check_revision",
    "draft",
    "fill",
    "list_drafts",
    "load",
    "resolve_source",
    "revise",
]
