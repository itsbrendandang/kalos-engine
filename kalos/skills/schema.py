"""The `kalos-skill/1` format: pydantic v2 models for a skill's YAML frontmatter.

A skill is a folder holding `SKILL.md` (YAML frontmatter + markdown body) and,
optionally, `references/`. This module only describes and checks the
frontmatter; `kalos.skills.loader` parses files and hashes folders.

Every model is `extra="forbid", frozen=True`: an unknown key is an error, not a
silently ignored typo, and a loaded skill cannot be mutated in place.

Cross-field rules (each has a failing and a passing case in
tests/test_skills_schema.py):

- an enum parameter declares `allowed`, and its default is one of them; a
  default must match the parameter's declared type;
- `implementation.status: implemented` needs >= 1 engine ref; `partial` and
  `needs-new-code` need >= 1 gap;
- lifecycle `status: planned` <=> `implementation.status: needs-new-code`, and a
  planned skill binds no tools (there is nothing to call yet);
- the summary states a negative boundary ("Not for ...");
- a locked skill declares `refuses_when`;
- side effects other than `none` need `scope: write`;
- every output carries a data label; `category: simulation` may only emit
  `simulated` or `synthetic`;
- `owner`/`visibility` agree (`system`/`system`, `tenant:<id>`/`tenant`);
- `updated >= created`.

Kalos-original code. The idea of a typed, versioned package with a hash identity
follows docs/SKILLS.md; no third-party code is used here.
"""
from __future__ import annotations

import datetime as dt
import re
from typing import Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_ID = "kalos-skill/1"
SLUG_RE = r"^[a-z0-9]+(?:-[a-z0-9]+)*$"
SEMVER_RE = r"^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$"
REF_RE = re.compile(r"^(?P<path>[\w./-]+\.py):(?P<line>\d+)$")
SYMBOL_RE = re.compile(r"^(?P<module>[A-Za-z_][\w.]*):(?P<attr>[A-Za-z_]\w*)$")
# An entrypoint or tool symbol may name a public method: `pkg.mod:Class.method`.
ENTRYPOINT_RE = re.compile(
    r"^(?P<module>[A-Za-z_][\w.]*):(?P<attr>[A-Za-z_]\w*)(?:\.(?P<method>[A-Za-z_]\w*))?$")
TOOL_NAME_RE = r"^(?:engine|skills)\.[a-z][a-z0-9_]*$"
OWNER_RE = re.compile(r"^(?:system|tenant:[A-Za-z0-9_.@-]{1,128}|user:[A-Za-z0-9_.@-]{1,256})$")
# "Not for ...", "not a ...", "never ...": a summary must say what the skill is NOT.
NEGATIVE_RE = re.compile(r"\b(?:not|never)\b", re.IGNORECASE)

Category = Literal[
    "intake", "data-quality", "design", "optimisation", "trust", "scale-up",
    "simulation", "analytics", "provenance", "governance",
]
CATEGORIES: tuple[str, ...] = get_args(Category)

# measured / simulated / synthetic use the same words as the `Origin` enum on the
# twin-virtual-runs branch (kalos/twin/origin.py:45 there), so receipts and
# virtual-run rows speak one vocabulary once that branch merges.
DataLabel = Literal["measured", "predicted", "simulated", "synthetic", "derived", "metadata"]
DATA_LABELS: tuple[str, ...] = get_args(DataLabel)
# Labels that are model or simulator output; never allowed into measured data.
NON_MEASURED_LABELS = frozenset({"predicted", "simulated", "synthetic"})

Lifecycle = Literal["active", "beta", "planned", "deprecated"]
ImplStatus = Literal["implemented", "partial", "needs-new-code"]

REQUIRED_SECTIONS = (
    "Purpose", "When to use", "When not to use", "Inputs", "Procedure",
    "Outputs", "Stop conditions", "Caveats", "Example",
)

_PARAM_PY_TYPES: dict[str, tuple[type, ...]] = {
    "int": (int,),
    "float": (int, float),
    "bool": (bool,),
    "str": (str,),
}


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class IOField(_Strict):
    """A declared input of a skill."""

    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    type: str = Field(min_length=2, max_length=40)
    required: bool = True
    description: str = Field(min_length=8)
    label: DataLabel | None = None


OutputKind = Literal["json", "csv", "table", "image", "file"]


class OutputField(_Strict):
    """A declared output: a typed artifact with a data label and a size floor
    (an artifact smaller than `min_bytes` is a runtime fault, not a result)."""

    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    kind: OutputKind
    description: str = Field(min_length=8)
    label: DataLabel
    min_bytes: int = Field(ge=0, le=1 << 30)


class Parameter(_Strict):
    """A bounded option, with the scientific effect of changing it."""

    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    type: Literal["int", "float", "bool", "str", "enum"]
    default: Any = None
    required: bool = False
    allowed: list[Any] | None = None
    effect: str = Field(min_length=8, description="Scientific effect of changing it")

    @model_validator(mode="after")
    def _consistent(self) -> Parameter:
        if self.type == "enum" and not self.allowed:
            raise ValueError(f"enum parameter {self.name!r} needs 'allowed'")
        if self.allowed and self.default is not None and self.default not in self.allowed:
            raise ValueError(f"default of {self.name!r} not in allowed")
        py = _PARAM_PY_TYPES.get(self.type)
        if py is not None and self.default is not None:
            # bool is an int subclass; a True default on an int/float parameter is a typo.
            if isinstance(self.default, bool) and self.type != "bool":
                raise ValueError(f"default of {self.name!r} is a bool, parameter is {self.type}")
            if not isinstance(self.default, py):
                raise ValueError(f"default of {self.name!r} does not match type {self.type}")
        return self


class EngineRef(_Strict):
    """A pointer to engine code: `pkg.mod:attr` plus `path.py:line`. The checker
    (`kalos.skills.loader.check_engine_refs`) resolves it statically."""

    symbol: str
    ref: str

    @field_validator("symbol")
    @classmethod
    def _sym(cls, v: str) -> str:
        if not SYMBOL_RE.match(v):
            raise ValueError("symbol must be 'package.module:attr'")
        return v

    @field_validator("ref")
    @classmethod
    def _ref(cls, v: str) -> str:
        if not REF_RE.match(v):
            raise ValueError("ref must be 'path/to/file.py:<line>'")
        return v


class Implementation(_Strict):
    status: ImplStatus
    engine: list[EngineRef] = []
    gaps: list[str] = []

    @model_validator(mode="after")
    def _consistent(self) -> Implementation:
        if self.status == "implemented" and not self.engine:
            raise ValueError("implemented skills must reference engine code")
        if self.status != "implemented" and not self.gaps:
            raise ValueError("partial / needs-new-code skills must list gaps")
        return self


class Verification(_Strict):
    deterministic: list[str] = Field(min_length=1)
    receipt: bool = True
    review: Literal["none", "optional", "required"] = "none"


CommercialStatus = Literal["allowed", "allowed-with-attribution", "needs-agreement", "blocked",
                          "unknown"]
# Only these count as allowed for commercial use. `unknown` never does.
COMMERCIAL_OK = frozenset({"allowed", "allowed-with-attribution"})


class DataSource(_Strict):
    """One software, dataset, API or reference a skill depends on. The
    frontmatter row is authoritative; DATA_SOURCES.md, if present, is rendered
    from these rows (`kalos.skills.loader.render_data_sources`)."""

    name: str = Field(min_length=1, max_length=200)
    kind: Literal["software", "dataset", "api", "reference"]
    version: str = Field(min_length=1, max_length=100)
    uri: str = Field(min_length=1, max_length=500)
    licence: str = Field(min_length=1, max_length=200)
    commercial_status: CommercialStatus
    bundled: bool
    verified_by: str = Field(min_length=1, max_length=500)
    checked_on: dt.date

    @property
    def commercial_use_allowed(self) -> bool:
        return self.commercial_status in COMMERCIAL_OK


class Licence(_Strict):
    spdx: str = Field(pattern=r"^(?:[A-Za-z0-9.+-]+|LicenseRef-[A-Za-z0-9.-]+)$")
    data_sources: list[DataSource] = []
    third_party_code: list[str] = []


class SkillMeta(_Strict):
    """The frontmatter of one `SKILL.md`."""

    # `schema` shadows a BaseModel attribute, so it is stored as `schema_id`.
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    schema_id: Literal["kalos-skill/1"] = Field(alias="schema")
    slug: str = Field(pattern=SLUG_RE, max_length=64)
    name: str = Field(min_length=3, max_length=80)
    version: str = Field(pattern=SEMVER_RE)
    kind: Literal["locked", "guided"]
    category: Category
    summary: str = Field(min_length=60, max_length=700)
    keywords: list[str] = []
    starting_prompt: str = Field(min_length=10)
    inputs: list[IOField] = Field(min_length=1)
    parameters: list[Parameter] = []
    outputs: list[OutputField] = Field(min_length=1)
    tools: list[str] = []
    # Locked skills: the one public engine callable that does the main
    # computation (`pkg.module:attr`); null while a skill is planned. Guided
    # skills have none.
    entrypoint: str | None = None
    # Optional dependency groups of pyproject.toml this skill's tools need.
    requires_extras: list[str] = []
    implementation: Implementation
    scope: Literal["read", "write"]
    side_effects: Literal["none", "writes-campaign", "external"]
    refuses_when: list[str] = []
    verification: Verification
    author: str = Field(min_length=1)
    owner: str = "system"
    visibility: Literal["system", "tenant", "private", "shared"] = "system"
    licence: Licence
    source: str | None = None
    status: Lifecycle = "active"
    created: dt.date
    updated: dt.date

    @field_validator("tools")
    @classmethod
    def _tool_names(cls, v: list[str]) -> list[str]:
        bad = [t for t in v if not re.match(TOOL_NAME_RE, t)]
        if bad:
            raise ValueError(f"tool names must look like 'engine.x' or 'skills.x': {bad}")
        if len(set(v)) != len(v):
            raise ValueError("duplicate tool name")
        return v

    @field_validator("owner")
    @classmethod
    def _owner(cls, v: str) -> str:
        if not OWNER_RE.match(v):
            raise ValueError("owner must be 'system', 'tenant:<id>' or 'user:<id>'")
        return v

    @model_validator(mode="after")
    def _rules(self) -> SkillMeta:
        if not NEGATIVE_RE.search(self.summary):
            raise ValueError("summary must state a negative boundary ('Not for ...')")
        if self.kind == "locked" and not self.refuses_when:
            raise ValueError("locked skills must declare refuses_when")
        if self.side_effects != "none" and self.scope != "write":
            raise ValueError("side effects require scope=write")
        if self.category == "simulation" and any(
            o.label not in ("simulated", "synthetic") for o in self.outputs
        ):
            raise ValueError("simulation skills may only emit label=simulated or synthetic")
        planned = self.status == "planned"
        if planned != (self.implementation.status == "needs-new-code"):
            raise ValueError("status=planned if and only if implementation.status=needs-new-code")
        if planned and self.tools:
            raise ValueError("a planned skill binds no tools until its code exists")
        if self.owner == "system" and self.visibility != "system":
            raise ValueError("owner=system requires visibility=system")
        if self.owner.startswith("tenant:") and self.visibility != "tenant":
            raise ValueError("owner=tenant:<id> requires visibility=tenant")
        if self.owner.startswith("user:") and self.visibility not in ("private", "shared"):
            raise ValueError("owner=user:<id> requires visibility=private or shared")
        if self.entrypoint is not None:
            em = ENTRYPOINT_RE.match(self.entrypoint)
            if em is None:
                raise ValueError("entrypoint must be 'package.module:attr[.method]'")
            if em["attr"].startswith("_") or (em["method"] or "").startswith("_"):
                raise ValueError("entrypoint must be public (no leading underscore)")
        if self.kind == "guided" and self.entrypoint is not None:
            raise ValueError("guided skills have no entrypoint")
        if self.kind == "locked" and not planned:
            if self.entrypoint is None:
                raise ValueError("a locked skill needs an entrypoint")
            if not self.verification.receipt:
                raise ValueError("a locked skill must write a receipt")
        if planned and self.entrypoint is not None:
            raise ValueError("a planned skill has no entrypoint until its code exists")
        if self.kind == "guided" and (self.scope != "read" or self.side_effects != "none"):
            raise ValueError("guided skills are read-only")
        if len(set(self.requires_extras)) != len(self.requires_extras):
            raise ValueError("duplicate extra in requires_extras")
        names = [o.name for o in self.outputs]
        if len(set(names)) != len(names):
            raise ValueError("duplicate output name")
        if self.updated < self.created:
            raise ValueError("updated before created")
        return self

    @property
    def output_labels(self) -> frozenset[str]:
        return frozenset(o.label for o in self.outputs)

    def output(self, name: str) -> OutputField | None:
        return next((o for o in self.outputs if o.name == name), None)

    @property
    def tenant(self) -> str | None:
        """The owning tenant for a Team skill, else None."""
        return self.owner.split(":", 1)[1] if self.owner.startswith("tenant:") else None


__all__ = [
    "CATEGORIES",
    "COMMERCIAL_OK",
    "DATA_LABELS",
    "ENTRYPOINT_RE",
    "NON_MEASURED_LABELS",
    "REF_RE",
    "REQUIRED_SECTIONS",
    "SCHEMA_ID",
    "SYMBOL_RE",
    "Category",
    "DataLabel",
    "DataSource",
    "EngineRef",
    "IOField",
    "Implementation",
    "Licence",
    "OutputField",
    "Parameter",
    "SkillMeta",
    "Verification",
]
