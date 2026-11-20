"""Request bodies of the records routes (projects, factors, lineage).

Adopted from Voyager's `validators/schemas.py` (NOTICE): one base model trims
surrounding whitespace from every string field before validation
(`TrimmedModel`, Voyager's `BaseValidation.trim_strings`), and every field
carries a `description`, so the OpenAPI document explains each input. Added for
Kalos: unknown keys are refused (`extra="forbid"`), strings and numbers are
strict (no "12" for 12, no NaN or infinity), and the domain rules run again in
`kalos.records` (these models only shape the request).

Validation failures use the engine's one error envelope (kalos/portal/app.py):
`422 {"error", "code": "validation_error", "detail": [...]}`, naming WHERE and
WHAT, never echoing the submitted value.
"""
from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import AllowInfNan, BaseModel, ConfigDict, Field, Strict, field_validator

from kalos.records.factors import MAX_DESCRIPTION_CHARS, MAX_FACTOR_NAME_CHARS
from kalos.records.lineage import MAX_NOTE_CHARS, MAX_REASON_CHARS
from kalos.records.projects import (
    MAX_NAME_CHARS,
    MAX_OBJECTIVE_CHARS,
    MAX_SPONSOR_CHARS,
)

_Num = Annotated[float, Strict(), AllowInfNan(False)]
ProjectStatus = Literal["drafting", "in_progress", "completed", "terminated"]
FactorRole = Literal["setting", "covariate"]
LinkRelation = Literal["next_round", "scale_up", "scale_down", "follow_up", "replicate", "derived"]


class TrimmedModel(BaseModel):
    """Strings are trimmed of surrounding whitespace before validation; unknown
    keys are refused."""

    model_config = ConfigDict(extra="forbid", strict=True)

    @field_validator("*", mode="before")
    @classmethod
    def _trim_strings(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value


class ProjectCreate(TrimmedModel):
    name: str = Field(min_length=1, max_length=MAX_NAME_CHARS,
                      description=f"Project name, unique within the tenant (1-{MAX_NAME_CHARS} characters).")
    objective: str = Field(min_length=1, max_length=MAX_OBJECTIVE_CHARS,
                           description=f"What the project is for, in plain words (up to {MAX_OBJECTIVE_CHARS} characters).")
    status: ProjectStatus = Field(default="drafting", description="Lifecycle status; see GET /api/project-statuses.")
    sponsor: str | None = Field(default=None, max_length=MAX_SPONSOR_CHARS,
                                description="Optional free-text sponsor or client label. Stored and shown, never interpreted.")


class ProjectUpdate(TrimmedModel):
    """A partial update: only the keys sent change. `sponsor: null` clears it."""

    name: str | None = Field(default=None, min_length=1, max_length=MAX_NAME_CHARS, description="New project name.")
    objective: str | None = Field(default=None, min_length=1, max_length=MAX_OBJECTIVE_CHARS, description="New objective.")
    status: ProjectStatus | None = Field(default=None, description="New lifecycle status.")
    sponsor: str | None = Field(default=None, max_length=MAX_SPONSOR_CHARS, description="New sponsor label, or null to clear it.")


class FactorPut(TrimmedModel):
    """One factor definition (a create or a whole replace, keyed by `name`)."""

    name: str = Field(min_length=1, max_length=MAX_FACTOR_NAME_CHARS,
                      description="The factor, as the run-sheet column it maps onto (e.g. glucose_g_L).")
    description: str | None = Field(default=None, max_length=MAX_DESCRIPTION_CHARS,
                                    description=f"What the factor is (up to {MAX_DESCRIPTION_CHARS} characters).")
    lower_bound: _Num = Field(description="Lowest value worth trying, in `unit`. Strictly below upper_bound.")
    upper_bound: _Num = Field(description="Highest value worth trying, in `unit`.")
    unit: str | None = Field(default=None, max_length=32,
                             description="A unit from GET /api/units (e.g. g/L, °C, %); empty for a dimensionless count.")
    role: FactorRole = Field(default="setting",
                             description="setting: the optimizer proposes it; covariate: measured and fitted, never proposed.")


class LinkCreate(TrimmedModel):
    """A parent -> child lineage link between two campaigns, or two of their runs."""

    parent_campaign_id: str = Field(min_length=36, max_length=36, description="The earlier campaign (cmp_...).")
    child_campaign_id: str = Field(min_length=36, max_length=36, description="The later campaign (cmp_...).")
    relation: LinkRelation = Field(description="How the child relates to the parent; see GET /api/lineage/relations.")
    parent_run_id: str | None = Field(default=None, max_length=128, description="Optional: the parent run (a run id of the parent campaign).")
    child_run_id: str | None = Field(default=None, max_length=128, description="Optional: the child run (a run id of the child campaign).")
    parent_stage: str | None = Field(default=None, max_length=17,
                                     description="Optional recorded stage key of the parent (e.g. S2); defaults to the campaign's current stage.")
    child_stage: str | None = Field(default=None, max_length=17,
                                    description="Optional recorded stage key of the child (e.g. S3); defaults to the campaign's current stage.")
    note: str | None = Field(default=None, max_length=MAX_NOTE_CHARS,
                             description="Optional note; required for relation 'derived'.")


class LinkRetract(TrimmedModel):
    reason: str = Field(min_length=1, max_length=MAX_REASON_CHARS,
                        description="Why the link is withdrawn. The link and the reason are both kept.")


__all__ = ["FactorPut", "LinkCreate", "LinkRetract", "ProjectCreate", "ProjectUpdate", "TrimmedModel"]
