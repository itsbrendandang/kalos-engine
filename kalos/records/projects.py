"""Projects: a grouping above campaigns (Voyager ``Project`` + ``ProjectStatus``).

A project has a name (unique per tenant), an objective, a status and an optional
free-text sponsor. It groups campaigns; a campaign belongs to at most one project.

What is NOT carried over from Voyager, on purpose:

- ``CustomerId`` and the ``Customer`` table: Kalos holds a lab's own data, not a
  services business's CRM. A sponsor or client name, when useful, is the optional
  ``sponsor`` label (stored, shown, never interpreted).
- ``MlConfig`` (a JSONB blob of model settings on the project): a Kalos campaign's
  setup (goal, bounds, constraints, roles) is the one place model inputs live.
- The ``ProjectStatus`` lookup TABLE: statuses are a fixed vocabulary in code
  (`STATUSES`), with Voyager's seeded names as labels.
"""
from __future__ import annotations

import unicodedata
import uuid
from dataclasses import dataclass
from typing import Any

from kalos.storage.migrations.v9 import PROJECT_STATUSES

STATUSES: tuple[str, ...] = PROJECT_STATUSES
# Voyager's seeded ProjectStatus names, as display labels.
STATUS_LABELS = {
    "drafting": "Drafting", "in_progress": "In progress", "completed": "Completed",
    "terminated": "Terminated",
}
STATUS_DESCRIPTIONS = {
    "drafting": "being planned; campaigns may be attached",
    "in_progress": "campaigns are running",
    "completed": "finished; its campaigns stay readable",
    "terminated": "stopped before completion; its campaigns stay readable",
}
MAX_NAME_CHARS = 80
MAX_OBJECTIVE_CHARS = 500
MAX_SPONSOR_CHARS = 120
MAX_PROJECTS_PER_TENANT = 1000
MAX_CAMPAIGNS_PER_PROJECT = 200
PROJECT_ID_PREFIX = "prj_"

_FORBIDDEN_CATEGORIES = {"Cc", "Cs", "Zl", "Zp"}


class RecordsValidationError(ValueError):
    """Input failed a records rule. `code` is stable; `reason` is authored here
    and never echoes the caller's input."""

    def __init__(self, code: str, reason: str) -> None:
        super().__init__(reason)
        self.code = code
        self.reason = reason


def has_control(value: str, *, allow_newlines: bool = False) -> bool:
    for ch in value:
        if allow_newlines and ch in "\n\r\t":
            continue
        if unicodedata.category(ch) in _FORBIDDEN_CATEGORIES:
            return True
    return False


def clean_text(
    value: object, what: str, max_chars: int, *, required: bool, allow_newlines: bool = False,
) -> str | None:
    """NFC, surrounding whitespace trimmed (Voyager's `trim_strings`), length and
    control-character checked. Blank optional text becomes None."""
    if value is None:
        if required:
            raise RecordsValidationError("invalid_body", f"{what} is required")
        return None
    if not isinstance(value, str):
        raise RecordsValidationError("invalid_body", f"{what} must be text")
    text = unicodedata.normalize("NFC", value).strip()
    if not text:
        if required:
            raise RecordsValidationError("invalid_body", f"{what} must not be blank")
        return None
    if len(text) > max_chars:
        raise RecordsValidationError("invalid_body", f"{what} must be at most {max_chars} characters")
    if has_control(text, allow_newlines=allow_newlines):
        raise RecordsValidationError("invalid_body", f"{what} must not contain control characters")
    return text


def new_project_id() -> str:
    return PROJECT_ID_PREFIX + uuid.uuid4().hex


def valid_project_id(value: str) -> bool:
    return (
        isinstance(value, str) and len(value) == len(PROJECT_ID_PREFIX) + 32
        and value.startswith(PROJECT_ID_PREFIX)
        and all(c in "0123456789abcdef" for c in value[len(PROJECT_ID_PREFIX):])
    )


@dataclass(frozen=True)
class ProjectFields:
    name: str
    objective: str
    status: str
    sponsor: str | None


def check_status(status: object) -> str:
    if status not in STATUSES:
        raise RecordsValidationError(
            "invalid_status", f"status must be one of: {', '.join(STATUSES)}"
        )
    return str(status)


def project_fields(
    *, name: object, objective: object, status: object, sponsor: object = None,
) -> ProjectFields:
    """The validated fields of a project (create, or a full replace)."""
    clean_name = clean_text(name, "name", MAX_NAME_CHARS, required=True)
    clean_objective = clean_text(
        objective, "objective", MAX_OBJECTIVE_CHARS, required=True, allow_newlines=True,
    )
    assert clean_name is not None and clean_objective is not None
    return ProjectFields(
        name=clean_name, objective=clean_objective, status=check_status(status),
        sponsor=clean_text(sponsor, "sponsor", MAX_SPONSOR_CHARS, required=False),
    )


def merge_update(current: dict[str, Any], patch: dict[str, Any]) -> ProjectFields:
    """A partial update applied over the stored project and validated whole.
    `sponsor: null` in the patch clears the sponsor."""
    return project_fields(
        name=patch.get("name", current["name"]),
        objective=patch.get("objective", current["objective"]),
        status=patch.get("status", current["status"]),
        sponsor=patch["sponsor"] if "sponsor" in patch else current.get("sponsor"),
    )


def status_catalogue() -> list[dict[str, str]]:
    """`GET /api/project-statuses`: the vocabulary, in lifecycle order."""
    return [
        {"status": s, "label": STATUS_LABELS[s], "description": STATUS_DESCRIPTIONS[s]}
        for s in STATUSES
    ]


__all__ = [
    "MAX_CAMPAIGNS_PER_PROJECT", "MAX_PROJECTS_PER_TENANT", "STATUSES", "ProjectFields",
    "RecordsValidationError", "check_status", "clean_text", "merge_update", "new_project_id",
    "project_fields", "status_catalogue", "valid_project_id",
]
