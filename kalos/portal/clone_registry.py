"""Label-only clone registry (schema v7 `campaign_clones` + `clone_titer_history`).

What a process scientist tracks about a production clone, stored as LABELS only
(docs/CLONES.md): a status (`lead`, `backup`, `retired`, `candidate`)
and a few allowlisted text fields (host, parent cell bank, construct id,
phenotype, product, passage at banking, note). Kalos stores, shows and filters
these values; it never parses them, never computes from them beyond counts and
equality, and never accepts a sequence, map or structure (the route refuses
those keys by construction).

`clone_titer_history` is the append-only record of MEASURED values per clone
(harvest titer, qP, space-time yield). A correction is a NEW row naming the row
it supersedes (`previous_id`, same tenant/campaign/label) and carrying a
non-empty `reason`; the database triggers reject UPDATE and DELETE.

Registry rows live outside the campaign state blob, so an edit never mints a
campaign generation (the `campaign_settings` rule). Every statement carries
``tenant = ?`` and the campaign is resolved with ``WHERE id = ? AND tenant = ?``
first, so another tenant's id raises `CampaignNotFound` exactly like a missing
one. Every write writes its audit row in the SAME transaction (keys and status
only, never field values).
"""
from __future__ import annotations

import datetime as _dt
import json
import math
import time
import unicodedata
from typing import Any

from kalos.portal.campaign import CampaignArchived, CampaignNotFound, CampaignStore

CLONE_STATUSES = ("lead", "backup", "retired", "candidate")
# Display / ordering rank of a status (never a computed score).
STATUS_ORDER = {"lead": 0, "backup": 1, "candidate": 2, "retired": 3}
CLONE_FIELD_KEYS = (
    "host", "parent_cell_bank", "construct_id", "phenotype_label", "product",
    "passage_at_banking", "note",
)
# Keys a caller might send that Kalos refuses by design: it stores labels only.
FORBIDDEN_KEYS = ("sequence", "fasta", "smiles", "pdb", "uniprot")
LABELS_ONLY_REASON = (
    "Kalos stores clone labels only; sequences, structures, database accessions and "
    "any key outside the allowlist are not accepted"
)

MAX_CLONES_PER_CAMPAIGN = 200
MAX_LABEL_CHARS = 64
MAX_FIELD_CHARS = 64
MAX_NOTE_CHARS = 500
MAX_SOURCE_NOTE_CHARS = 200
MAX_REASON_CHARS = 500
# A bound on the history so one campaign cannot grow the table without limit.
MAX_TITER_ROWS_PER_CAMPAIGN = 20_000

TITER_BASES = ("harvest_titer", "qP", "space_time_yield")
TITER_UNITS = ("g/L", "mg/L", "pg/cell/day", "g/L/day")
UNITS_BY_BASIS: dict[str, tuple[str, ...]] = {
    "harvest_titer": ("g/L", "mg/L"),
    "qP": ("pg/cell/day",),
    "space_time_yield": ("g/L/day",),
}
AGE_KINDS = ("passage", "generation", "pdl")

HISTORY_DEFAULT_LIMIT = 100
HISTORY_MAX_LIMIT = 500

_FORBIDDEN_CATEGORIES = {"Cc", "Cs", "Zl", "Zp"}
_NOTE_ALLOWED_CONTROL = {"\n", "\r", "\t"}
_TITER_COLUMNS = (
    "id", "label", "recorded_at", "measured_on", "basis", "value", "unit", "scale_label",
    "age_kind", "age_value", "source_note", "actor", "previous_id", "reason",
)


class CloneValidationError(ValueError):
    """Input failed validation. `reason` is authored here and never echoes input."""

    def __init__(self, code: str, reason: str) -> None:
        super().__init__(reason)
        self.code = code
        self.reason = reason


class CloneNotRegistered(LookupError):
    """The label has no row in this campaign's registry."""


class TooManyClones(RuntimeError):
    """The campaign already has MAX_CLONES_PER_CAMPAIGN registered clones."""


class TiterHistoryLimit(RuntimeError):
    """The campaign has reached MAX_TITER_ROWS_PER_CAMPAIGN history rows."""


# --- validation --------------------------------------------------------------- #


def _has_control(value: str, *, allow_newlines: bool = False) -> bool:
    for ch in value:
        if allow_newlines and ch in _NOTE_ALLOWED_CONTROL:
            continue
        if unicodedata.category(ch) in _FORBIDDEN_CATEGORIES:
            return True
    return False


def check_clone_label(label: object) -> str:
    """A clone label as it appears in the run sheet: 1..64 characters, not blank,
    no control characters. Not trimmed (it must match the sheet value exactly)."""
    if not isinstance(label, str) or not 1 <= len(label) <= MAX_LABEL_CHARS or not label.strip():
        raise CloneValidationError(
            "invalid_label", f"the clone label must be 1-{MAX_LABEL_CHARS} characters and not blank"
        )
    if _has_control(label):
        raise CloneValidationError("invalid_label", "the clone label must not contain control characters")
    return label


def _finite(value: object, what: str, code: str = "invalid_number") -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise CloneValidationError(code, f"{what} must be a finite number")
    return float(value)


def _text(value: object, what: str, max_chars: int, *, allow_newlines: bool = False) -> str | None:
    """A stripped short string, or None when absent/blank."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise CloneValidationError("invalid_field", f"{what} must be a string")
    text = value.strip()
    if len(text) > max_chars:
        raise CloneValidationError("invalid_field", f"{what} must be at most {max_chars} characters")
    if _has_control(text, allow_newlines=allow_newlines):
        raise CloneValidationError("invalid_field", f"{what} must not contain control characters")
    return text or None


def normalize_fields(fields: dict[str, Any]) -> dict[str, Any]:
    """The allowlisted label fields, stripped; absent or blank values dropped.
    Any key outside the allowlist (including FORBIDDEN_KEYS) is refused."""
    if not isinstance(fields, dict):
        raise CloneValidationError("invalid_body", LABELS_ONLY_REASON)
    if any(k not in CLONE_FIELD_KEYS for k in fields):
        raise CloneValidationError("invalid_body", LABELS_ONLY_REASON)
    out: dict[str, Any] = {}
    for key in CLONE_FIELD_KEYS:
        value = fields.get(key)
        if value is None:
            continue
        if key == "passage_at_banking" and not isinstance(value, str):
            out[key] = _finite(value, "passage_at_banking", code="invalid_field")
            continue
        limit = MAX_NOTE_CHARS if key == "note" else MAX_FIELD_CHARS
        text = _text(value, key, limit, allow_newlines=key == "note")
        if text is not None:
            out[key] = text
    return out


def _check_status(status: object) -> str:
    if status not in CLONE_STATUSES:
        raise CloneValidationError("invalid_status", "status must be lead, backup, retired or candidate")
    return str(status)


def _check_measured_on(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or len(value) != 10:
        raise CloneValidationError("invalid_date", "measured_on must be an ISO date (YYYY-MM-DD)")
    try:
        return _dt.date.fromisoformat(value).isoformat()
    except ValueError:
        raise CloneValidationError(
            "invalid_date", "measured_on must be an ISO date (YYYY-MM-DD)"
        ) from None


def normalize_titer_row(row: dict[str, Any]) -> dict[str, Any]:
    """Validate one titer-history submission. Raises `CloneValidationError`."""
    allowed = {"measured_on", "basis", "value", "unit", "scale_label", "age_kind", "age_value",
               "source_note", "previous_id", "reason"}
    if not isinstance(row, dict) or any(k not in allowed for k in row):
        raise CloneValidationError("invalid_body", LABELS_ONLY_REASON)
    basis = row.get("basis")
    if basis not in TITER_BASES:
        raise CloneValidationError("invalid_basis", "basis must be harvest_titer, qP or space_time_yield")
    unit = row.get("unit")
    if unit not in TITER_UNITS:
        raise CloneValidationError("invalid_unit", "unit must be one of g/L, mg/L, pg/cell/day, g/L/day")
    if unit not in UNITS_BY_BASIS[str(basis)]:
        raise CloneValidationError(
            "unit_basis_mismatch",
            "the unit does not match the basis: harvest_titer takes g/L or mg/L, qP takes "
            "pg/cell/day, space_time_yield takes g/L/day",
        )
    age_kind = row.get("age_kind")
    if age_kind is not None and age_kind not in AGE_KINDS:
        raise CloneValidationError("invalid_age_kind", "age_kind must be passage, generation or pdl")
    age_value = row.get("age_value")
    previous_id = row.get("previous_id")
    if previous_id is not None and (
        isinstance(previous_id, bool) or not isinstance(previous_id, int) or previous_id < 1
    ):
        raise CloneValidationError("invalid_previous_id", "previous_id must be a positive integer")
    reason = _text(row.get("reason"), "reason", MAX_REASON_CHARS, allow_newlines=True)
    if previous_id is not None and reason is None:
        raise CloneValidationError(
            "reason_required", "a correction (previous_id) needs a non-empty reason saying why"
        )
    if previous_id is None and reason is not None:
        raise CloneValidationError(
            "previous_id_required", "a reason is only recorded with a correction: send previous_id too"
        )
    return {
        "measured_on": _check_measured_on(row.get("measured_on")),
        "basis": str(basis),
        "value": _finite(row.get("value"), "value"),
        "unit": str(unit),
        "scale_label": _text(row.get("scale_label"), "scale_label", MAX_FIELD_CHARS),
        "age_kind": age_kind,
        "age_value": None if age_value is None else _finite(age_value, "age_value"),
        "source_note": _text(row.get("source_note"), "source_note", MAX_SOURCE_NOTE_CHARS,
                             allow_newlines=True),
        "previous_id": previous_id,
        "reason": reason,
    }


# --- reads ---------------------------------------------------------------------- #


def _resolve(conn: Any, store: CampaignStore, tenant: str, campaign_id: str, *, write: bool) -> None:
    row = store._row(conn, tenant, campaign_id)
    if row is None:
        raise CampaignNotFound()
    if write and row.archived_at is not None:
        raise CampaignArchived()


def _clone_view(row: tuple[Any, ...]) -> dict[str, Any]:
    label, status, fields, updated_at, updated_by = row
    try:
        parsed = json.loads(fields)
    except ValueError:
        parsed = {}
    return {
        "label": label, "status": status, "fields": parsed if isinstance(parsed, dict) else {},
        "updated_at": updated_at, "updated_by": updated_by,
    }


def list_clones(store: CampaignStore, *, tenant: str, campaign_id: str) -> list[dict[str, Any]]:
    """Every registered clone of the campaign, by label. `CampaignNotFound` for a
    missing or foreign campaign."""
    store._check(tenant, campaign_id)
    with store._txn(write=False) as conn:
        _resolve(conn, store, tenant, campaign_id, write=False)
        rows = conn.execute(
            "SELECT label, status, fields, updated_at, updated_by FROM campaign_clones "
            "WHERE tenant = ? AND campaign_id = ? ORDER BY label",
            (tenant, campaign_id),
        ).fetchall()
    return [_clone_view(r) for r in rows]


def get_clone(
    store: CampaignStore, *, tenant: str, campaign_id: str, label: str
) -> dict[str, Any] | None:
    """One registered clone, or None. `CampaignNotFound` for a missing/foreign campaign."""
    store._check(tenant, campaign_id)
    with store._txn(write=False) as conn:
        _resolve(conn, store, tenant, campaign_id, write=False)
        row = conn.execute(
            "SELECT label, status, fields, updated_at, updated_by FROM campaign_clones "
            "WHERE tenant = ? AND campaign_id = ? AND label = ?",
            (tenant, campaign_id, label),
        ).fetchone()
    return _clone_view(row) if row is not None else None


def lead_clone_label(store: CampaignStore, *, tenant: str, campaign_id: str) -> str | None:
    """The label of the campaign's single `status='lead'` clone; None when no
    clone, or more than one, is marked lead (the caller then has no default).
    `CampaignNotFound` for a missing or foreign campaign."""
    store._check(tenant, campaign_id)
    with store._txn(write=False) as conn:
        _resolve(conn, store, tenant, campaign_id, write=False)
        rows = conn.execute(
            "SELECT label FROM campaign_clones WHERE tenant = ? AND campaign_id = ? "
            "AND status = 'lead' ORDER BY label LIMIT 2",
            (tenant, campaign_id),
        ).fetchall()
    return str(rows[0][0]) if len(rows) == 1 else None


def list_titer_history(
    store: CampaignStore, *, tenant: str, campaign_id: str, label: str,
    before_id: int | None = None, limit: int = HISTORY_DEFAULT_LIMIT,
) -> dict[str, Any]:
    """One newest-first page of a registered clone's titer history, in the
    `exports_api._page` shape: `{items, has_more, next_before_id}`. Raises
    `CampaignNotFound` or `CloneNotRegistered`."""
    store._check(tenant, campaign_id)
    limit = max(1, min(int(limit), HISTORY_MAX_LIMIT))
    sql = (
        f"SELECT {', '.join(_TITER_COLUMNS)} FROM clone_titer_history "  # noqa: S608 - module-constant columns; values are bound
        "WHERE tenant = ? AND campaign_id = ? AND label = ?"
        + (" AND id < ?" if before_id is not None else "")
        + " ORDER BY id DESC LIMIT ?"
    )
    params: tuple[Any, ...] = (
        tenant, campaign_id, label, *((before_id,) if before_id is not None else ()), limit + 1,
    )
    with store._txn(write=False) as conn:
        _resolve(conn, store, tenant, campaign_id, write=False)
        if conn.execute(
            "SELECT 1 FROM campaign_clones WHERE tenant = ? AND campaign_id = ? AND label = ?",
            (tenant, campaign_id, label),
        ).fetchone() is None:
            raise CloneNotRegistered()
        fetched = conn.execute(sql, params).fetchall()
    items = [{**dict(zip(_TITER_COLUMNS, r)), "basis_kind": "measured",
              "is_correction": r[_TITER_COLUMNS.index("previous_id")] is not None}
             for r in fetched[:limit]]
    has_more = len(fetched) > limit
    return {
        "items": items,
        "has_more": has_more,
        "next_before_id": items[-1]["id"] if has_more and items else None,
    }


# --- writes ----------------------------------------------------------------------- #


def put_clone(
    store: CampaignStore, *, tenant: str, campaign_id: str, label: str, status: str,
    fields: dict[str, Any], updated_by: str | None, auth_method: str | None = None,
    audit_action: str = "campaign.clone.put",
) -> dict[str, Any]:
    """Create or replace (upsert) one clone's registry row and write its audit
    row in the same transaction. Never touches the campaign state blob, so no
    campaign generation is minted. Raises `CloneValidationError`,
    `CampaignNotFound`, `CampaignArchived`, `TooManyClones`."""
    store._check(tenant, campaign_id)
    label = check_clone_label(label)
    status = _check_status(status)
    clean = normalize_fields(fields)
    blob = json.dumps(clean, sort_keys=True, allow_nan=False)
    now = time.time()
    who = updated_by[:256] if isinstance(updated_by, str) else None
    with store._txn(write=True) as conn:
        _resolve(conn, store, tenant, campaign_id, write=True)
        exists = conn.execute(
            "SELECT 1 FROM campaign_clones WHERE tenant = ? AND campaign_id = ? AND label = ?",
            (tenant, campaign_id, label),
        ).fetchone() is not None
        if not exists:
            (count,) = conn.execute(
                "SELECT COUNT(*) FROM campaign_clones WHERE tenant = ? AND campaign_id = ?",
                (tenant, campaign_id),
            ).fetchone()
            if int(count) >= MAX_CLONES_PER_CAMPAIGN:
                raise TooManyClones()
        conn.execute(
            "INSERT INTO campaign_clones (tenant, campaign_id, label, fields, status, updated_at, "
            "updated_by) VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(tenant, campaign_id, label) DO UPDATE SET fields = excluded.fields, "
            "status = excluded.status, updated_at = excluded.updated_at, "
            "updated_by = excluded.updated_by WHERE campaign_clones.tenant = excluded.tenant",
            (tenant, campaign_id, label, blob, status, now, who),
        )
        store._insert_audit(
            conn, tenant=tenant, action=audit_action, outcome="ok", subject=updated_by,
            auth_method=auth_method, campaign_id=campaign_id, now=now,
            detail=(f"label={label};status={status};created={0 if exists else 1};"
                    f"keys={','.join(sorted(clean))}"),
        )
    return {"label": label, "status": status, "fields": clean, "updated_at": now,
            "updated_by": who, "created": not exists}


def append_titer(
    store: CampaignStore, *, tenant: str, campaign_id: str, label: str, row: dict[str, Any],
    actor: str | None, auth_method: str | None = None,
    audit_action: str = "campaign.clone.titer_append",
) -> dict[str, Any]:
    """Append one measured row to a registered clone's history (audit row in the
    same transaction). A correction needs `previous_id` (an existing row of the
    SAME tenant, campaign and label) and a non-empty `reason`. Raises
    `CloneValidationError`, `CampaignNotFound`, `CampaignArchived`,
    `CloneNotRegistered`, `TiterHistoryLimit`."""
    store._check(tenant, campaign_id)
    label = check_clone_label(label)
    clean = normalize_titer_row(row)
    who = actor[:256] if isinstance(actor, str) else None
    now = time.time()
    with store._txn(write=True) as conn:
        _resolve(conn, store, tenant, campaign_id, write=True)
        if conn.execute(
            "SELECT 1 FROM campaign_clones WHERE tenant = ? AND campaign_id = ? AND label = ?",
            (tenant, campaign_id, label),
        ).fetchone() is None:
            raise CloneNotRegistered()
        if clean["previous_id"] is not None and conn.execute(
            "SELECT 1 FROM clone_titer_history WHERE tenant = ? AND campaign_id = ? AND label = ? "
            "AND id = ?",
            (tenant, campaign_id, label, clean["previous_id"]),
        ).fetchone() is None:
            raise CloneValidationError(
                "previous_not_found",
                "previous_id must name an existing titer row of this clone in this campaign",
            )
        (count,) = conn.execute(
            "SELECT COUNT(*) FROM clone_titer_history WHERE tenant = ? AND campaign_id = ?",
            (tenant, campaign_id),
        ).fetchone()
        if int(count) >= MAX_TITER_ROWS_PER_CAMPAIGN:
            raise TiterHistoryLimit()
        cur = conn.execute(
            "INSERT INTO clone_titer_history (tenant, campaign_id, label, recorded_at, measured_on, "
            "basis, value, unit, scale_label, age_kind, age_value, source_note, actor, previous_id, "
            "reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (tenant, campaign_id, label, now, clean["measured_on"], clean["basis"], clean["value"],
             clean["unit"], clean["scale_label"], clean["age_kind"], clean["age_value"],
             clean["source_note"], who, clean["previous_id"], clean["reason"]),
        )
        new_id = int(cur.lastrowid or 0)
        store._insert_audit(
            conn, tenant=tenant, action=audit_action, outcome="ok", subject=actor,
            auth_method=auth_method, campaign_id=campaign_id, now=now,
            detail=(f"label={label};id={new_id};basis={clean['basis']};"
                    f"correction={0 if clean['previous_id'] is None else 1}"),
        )
    return {"id": new_id, "label": label, "recorded_at": now, **clean, "actor": who,
            "basis_kind": "measured", "is_correction": clean["previous_id"] is not None}
