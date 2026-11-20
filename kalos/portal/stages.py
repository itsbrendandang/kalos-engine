"""Per-campaign scale-up stage track and its go/no-go gate registry.

Two kinds of data, kept apart on purpose:

- STAGE DEFINITIONS (editable, `campaign_settings` kind ``stages``): an ordered
  list of stages, each with a volume range in liters, plus the RECORDED
  ``current_stage`` (what the team says it is working on; never computed). A
  campaign that never edited its stages reads `DEFAULT_STAGES`
  (``source: "default"``).
- GATE DECISIONS (append-only, `gate_decisions`): human go / no_go / pending
  decisions, each with a note and optional linked evidence ids. Every stage
  ends in a gate whose key is the stage key ("S1" = the gate at the end of
  S1). A gate's CURRENT decision is its newest row; older rows are history.
  The table rejects UPDATE and DELETE at the database level (schema v6).

Editing the stage definitions never touches recorded decisions: a gate whose
stage was removed or renamed keeps its history and is reported as an
"orphaned" gate (read-only; no new decision can be recorded against it until
a stage with that key exists again).

Every SQL statement here carries ``tenant = ?``, and the campaign is resolved
with ``WHERE id = ? AND tenant = ?`` first, so another tenant's campaign id
raises `CampaignNotFound` exactly like a missing one.
"""
from __future__ import annotations

import json
import math
import re
import time
import unicodedata
from typing import Any

from kalos.portal.campaign import CampaignArchived, CampaignNotFound, CampaignStore
from kalos.portal.campaign_settings import get_setting

STAGES_KIND = "stages"
STAGES_DOC_VERSION = 1

MIN_STAGES = 1
MAX_STAGES = 12
MAX_LABEL_CHARS = 80
MAX_NOMINAL_SCALES = 12
# 1,000,000 L (1,000 m3) is above any production bioreactor; larger is a typo.
MAX_VOLUME_L = 1_000_000.0
STAGE_KEY_PATTERN = re.compile(r"S?[A-Za-z0-9_-]{1,16}")

GATE_DECISIONS = ("go", "no_go", "pending")
MAX_NOTE_CHARS = 2000
MAX_EVIDENCE_ITEMS = 20
EVIDENCE_KINDS = ("analysis", "scale_readout", "job", "run")
EVIDENCE_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")
MAX_EVIDENCE_LABEL_CHARS = 120
# Bound on the append-only table per campaign, so one campaign cannot grow it
# without limit. Far above any real decision cadence.
MAX_DECISIONS_PER_CAMPAIGN = 1000
# `/gates` and `PUT /gates/{gate}` embed only the newest GATE_HISTORY_PREVIEW
# decisions of a gate; the rest is paged by `GET /gates/{gate}/history`.
GATE_HISTORY_PREVIEW = 20
HISTORY_DEFAULT_LIMIT = 50
HISTORY_MAX_LIMIT = 200
# Orphaned gates (decisions whose stage key was removed) listed in `/stages`
# and `/gates`: the ones with the most recent decisions, at most this many.
# Their full history stays readable via the history route.
MAX_ORPHANED_GATES_LISTED = 12

# REDESIGN_DIRECTION.md: Stage 1 HT media screen -> Stage 2 HT microbioreactor
# (100 uL to 3 mL) -> Stage 3 flask (50 to 250 mL) -> Stage 4 bench bioreactor
# (500 mL to 1 L) -> Stage 5 bioreactor (2 L, 10 L, 50 L) -> Stage 6 data
# package and handoff. Volumes in liters; null where the stage has no volume.
DEFAULT_STAGES: tuple[dict[str, Any], ...] = (
    {"key": "S1", "label": "HT media screen", "min_L": None, "max_L": None, "nominal_L": None},
    {"key": "S2", "label": "HT microbioreactor", "min_L": 0.0001, "max_L": 0.003, "nominal_L": None},
    {"key": "S3", "label": "Flask", "min_L": 0.05, "max_L": 0.25, "nominal_L": None},
    {"key": "S4", "label": "Bench bioreactor", "min_L": 0.5, "max_L": 1.0, "nominal_L": None},
    {"key": "S5", "label": "Bioreactor", "min_L": 2.0, "max_L": 50.0, "nominal_L": [2.0, 10.0, 50.0]},
    {"key": "S6", "label": "Data package and handoff", "min_L": None, "max_L": None, "nominal_L": None},
)

# Cc control, Cf format (bidi overrides U+202A-U+202E / U+2066-U+2069, zero
# width characters: text that displays differently from what it contains),
# Cs surrogates, Zl/Zp line/paragraph separators.
_LABEL_FORBIDDEN = {"Cc", "Cf", "Cs", "Zl", "Zp"}
# A note may span lines; every other control character is refused.
_NOTE_ALLOWED_CONTROL = {"\n", "\r", "\t"}


class StageValidationError(ValueError):
    """A stage document or gate decision failed validation. The message is
    authored here and never echoes caller input."""

    def __init__(self, code: str, reason: str) -> None:
        super().__init__(reason)
        self.code = code
        self.reason = reason


class GateNotFound(LookupError):
    """The gate key is not a stage in the campaign's current definitions."""


class DecisionLimitReached(RuntimeError):
    """The campaign already holds MAX_DECISIONS_PER_CAMPAIGN decisions."""


# --- validation --------------------------------------------------------------- #


def valid_stage_key(key: object) -> bool:
    return isinstance(key, str) and STAGE_KEY_PATTERN.fullmatch(key) is not None


def check_label(value: str, *, max_chars: int = MAX_LABEL_CHARS, what: str = "label") -> str:
    label = value.strip()
    if not 1 <= len(label) <= max_chars:
        raise StageValidationError("invalid_label", f"{what} must be 1-{max_chars} characters")
    if any(unicodedata.category(ch) in _LABEL_FORBIDDEN for ch in label):
        raise StageValidationError(
            "invalid_label", f"{what} must not contain control or format characters"
        )
    return label


def check_note(value: str) -> str:
    note = value.strip()
    if not note:
        raise StageValidationError("note_required", "a decision needs a non-empty note saying why")
    if len(note) > MAX_NOTE_CHARS:
        raise StageValidationError("invalid_note", f"note must be at most {MAX_NOTE_CHARS} characters")
    for ch in note:
        if ch in _NOTE_ALLOWED_CONTROL:
            continue
        if unicodedata.category(ch) in _LABEL_FORBIDDEN:
            raise StageValidationError(
                "invalid_note",
                "note must not contain control or format characters other than newline and tab",
            )
    return note


def _volume(value: Any, where: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StageValidationError("invalid_volume", f"{where} must be a number or null")
    v = float(value)
    if not math.isfinite(v) or v <= 0 or v > MAX_VOLUME_L:
        raise StageValidationError(
            "invalid_volume", f"{where} must be a positive finite volume in liters, at most {MAX_VOLUME_L:g}"
        )
    return v


def validate_stages(stages: list[dict[str, Any]], current_stage: str | None) -> dict[str, Any]:
    """Validate and normalize a stage document. Returns the document to store:
    `{"version", "stages": [...], "current_stage"}`.

    Rules: 1..12 stages; unique keys matching `STAGE_KEY_PATTERN`; labels 1..80
    characters, no control characters; `min_L` and `max_L` both null or both
    positive finite with `min_L <= max_L`; `nominal_L` null or 1..12 ascending
    unique values inside `[min_L, max_L]` (requires a range); stages with a
    volume are ORDERED: each one's `min_L` and `max_L` are >= the previous
    volumed stage's (ranges may touch or overlap, they may not go backwards);
    `current_stage` null or one of the keys."""
    if not isinstance(stages, list) or not MIN_STAGES <= len(stages) <= MAX_STAGES:
        raise StageValidationError("invalid_stages", f"provide {MIN_STAGES}-{MAX_STAGES} stages")
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    prev: tuple[float, float] | None = None
    for i, raw in enumerate(stages):
        where = f"stages[{i}]"
        key = raw.get("key")
        if not valid_stage_key(key):
            raise StageValidationError(
                "invalid_stage_key", f"{where}.key must match {STAGE_KEY_PATTERN.pattern}"
            )
        assert isinstance(key, str)
        if key in seen:
            raise StageValidationError("duplicate_stage_key", f"{where}.key repeats an earlier stage key")
        seen.add(key)
        label_raw = raw.get("label")
        if not isinstance(label_raw, str):
            raise StageValidationError("invalid_label", f"{where}.label must be a string")
        label = check_label(label_raw, what=f"{where}.label")
        lo = _volume(raw.get("min_L"), f"{where}.min_L")
        hi = _volume(raw.get("max_L"), f"{where}.max_L")
        if (lo is None) != (hi is None):
            raise StageValidationError(
                "invalid_volume", f"{where}: give both min_L and max_L, or neither"
            )
        nominal_raw = raw.get("nominal_L")
        nominal: list[float] | None = None
        if lo is not None and hi is not None:
            if lo > hi:
                raise StageValidationError("invalid_volume", f"{where}: min_L must be <= max_L")
            if prev is not None and (lo < prev[0] or hi < prev[1]):
                raise StageValidationError(
                    "stages_out_of_order",
                    f"{where}: volumes must not decrease from one stage to the next",
                )
            prev = (lo, hi)
        if nominal_raw is not None:
            if lo is None or hi is None:
                raise StageValidationError(
                    "invalid_volume", f"{where}: nominal_L needs a min_L/max_L range"
                )
            if not isinstance(nominal_raw, list) or not 1 <= len(nominal_raw) <= MAX_NOMINAL_SCALES:
                raise StageValidationError(
                    "invalid_volume", f"{where}.nominal_L must be null or 1-{MAX_NOMINAL_SCALES} volumes"
                )
            nominal = []
            for j, item in enumerate(nominal_raw):
                v = _volume(item, f"{where}.nominal_L[{j}]")
                assert v is not None  # null items are refused by the type check below
                if not lo <= v <= hi:
                    raise StageValidationError(
                        "invalid_volume", f"{where}.nominal_L[{j}] must lie within [min_L, max_L]"
                    )
                if nominal and v <= nominal[-1]:
                    raise StageValidationError(
                        "invalid_volume", f"{where}.nominal_L must be strictly ascending"
                    )
                nominal.append(v)
        out.append({"key": key, "label": label, "min_L": lo, "max_L": hi, "nominal_L": nominal})
    if current_stage is not None and current_stage not in seen:
        raise StageValidationError(
            "invalid_current_stage", "current_stage must be null or one of the stage keys"
        )
    return {"version": STAGES_DOC_VERSION, "stages": out, "current_stage": current_stage}


def default_document() -> dict[str, Any]:
    return {
        "version": STAGES_DOC_VERSION,
        "stages": [dict(s, nominal_L=list(s["nominal_L"]) if s["nominal_L"] else None)
                   for s in DEFAULT_STAGES],
        "current_stage": None,
    }


def _doc_from_value(value: Any) -> dict[str, Any] | None:
    """A stored document, re-validated on read. A document that no longer
    validates (hand-edited row, older rules) reads as None -> defaults."""
    if not isinstance(value, dict) or not isinstance(value.get("stages"), list):
        return None
    try:
        current = value.get("current_stage")
        return validate_stages(
            [s if isinstance(s, dict) else {} for s in value["stages"]],
            current if isinstance(current, str) else None,
        )
    except StageValidationError:
        return None


def load_stages(store: CampaignStore, *, tenant: str, campaign_id: str) -> dict[str, Any]:
    """The campaign's stage document plus `source` ('default' | 'campaign'),
    `updated_at`, `updated_by`. Raises `CampaignNotFound`."""
    stored = get_setting(store, tenant=tenant, campaign_id=campaign_id, kind=STAGES_KIND)
    doc = _doc_from_value(stored["value"]) if stored is not None else None
    if stored is None or doc is None:
        return {**default_document(), "source": "default", "updated_at": None, "updated_by": None}
    return {**doc, "source": "campaign", "updated_at": stored["updated_at"],
            "updated_by": stored["updated_by"]}


def _stage_keys_in_txn(conn: Any, tenant: str, campaign_id: str) -> set[str]:
    row = conn.execute(
        "SELECT value FROM campaign_settings WHERE campaign_id = ? AND tenant = ? AND kind = ?",
        (campaign_id, tenant, STAGES_KIND),
    ).fetchone()
    doc = None
    if row is not None:
        try:
            doc = _doc_from_value(json.loads(row[0]))
        except ValueError:
            doc = None
    if doc is None:
        doc = default_document()
    return {s["key"] for s in doc["stages"]}


# --- gate decisions (append-only) --------------------------------------------- #

_DECISION_COLUMNS = "id, gate, decision, note, evidence, decided_by, auth_method, decided_at"


def _decision_row(row: tuple[Any, ...]) -> dict[str, Any]:
    try:
        evidence = json.loads(row[4]) if row[4] else []
    except ValueError:
        evidence = []
    return {
        "id": int(row[0]),
        "gate": row[1],
        "decision": row[2],
        "note": row[3],
        "evidence_linked": evidence if isinstance(evidence, list) else [],
        "decided_by": row[5],
        "auth_method": row[6],
        "decided_at": row[7],
        "basis": "recorded",
    }


def _check_campaign(store: CampaignStore, conn: Any, tenant: str, campaign_id: str) -> None:
    if store._row(conn, tenant, campaign_id) is None:
        raise CampaignNotFound()


def _counts_in_txn(conn: Any, tenant: str, campaign_id: str) -> dict[str, tuple[int, int]]:
    """`{gate: (count, newest_id)}` counted in SQL; no decision row is loaded."""
    rows = conn.execute(
        "SELECT gate, COUNT(*), MAX(id) FROM gate_decisions "
        "WHERE tenant = ? AND campaign_id = ? GROUP BY gate",
        (tenant, campaign_id),
    ).fetchall()
    return {str(r[0]): (int(r[1]), int(r[2])) for r in rows}


def decision_counts(
    store: CampaignStore, *, tenant: str, campaign_id: str
) -> dict[str, tuple[int, int]]:
    """`{gate: (count, newest_id)}` (`SELECT gate, COUNT(*) ... GROUP BY
    gate`). Raises `CampaignNotFound` for a missing or foreign campaign."""
    store._check(tenant, campaign_id)
    with store._txn(write=False) as conn:
        _check_campaign(store, conn, tenant, campaign_id)
        return _counts_in_txn(conn, tenant, campaign_id)


def select_orphans(
    counts: dict[str, tuple[int, int]], stage_keys: list[str]
) -> tuple[list[str], int]:
    """Orphaned gate keys to list (at most MAX_ORPHANED_GATES_LISTED: those
    with the most recent decisions, returned sorted by key) and the total."""
    orphans = sorted((k for k in counts if k not in stage_keys),
                     key=lambda k: counts[k][1], reverse=True)
    return sorted(orphans[:MAX_ORPHANED_GATES_LISTED]), len(orphans)


def gate_summaries(
    store: CampaignStore, *, tenant: str, campaign_id: str, stage_keys: list[str],
    only_gate: str | None = None, per_gate: int = GATE_HISTORY_PREVIEW,
) -> tuple[dict[str, dict[str, Any]], int]:
    """Per gate: `{"count", "latest"}` where `latest` is the newest `per_gate`
    decisions, newest first. Gates listed: every key in `stage_keys` plus the
    orphaned gates chosen by `select_orphans` (or only `only_gate`). Returns
    the summaries and the TOTAL number of orphaned gates. Never loads more
    than `per_gate` rows per listed gate. Raises `CampaignNotFound`."""
    store._check(tenant, campaign_id)
    with store._txn(write=False) as conn:
        _check_campaign(store, conn, tenant, campaign_id)
        counts = _counts_in_txn(conn, tenant, campaign_id)
        orphans, n_orphans = select_orphans(counts, stage_keys)
        wanted = [only_gate] if only_gate is not None else [*stage_keys, *orphans]
        wanted = [g for g in dict.fromkeys(wanted) if g in counts]
        rows: list[tuple[Any, ...]] = []
        if wanted:
            marks = ", ".join("?" for _ in wanted)
            rows = conn.execute(
                f"SELECT {_DECISION_COLUMNS} FROM gate_decisions "  # noqa: S608 - column list is a module constant; placeholders only; values are bound parameters
                "WHERE tenant = ? AND id IN ("
                " SELECT id FROM ("
                "  SELECT id, ROW_NUMBER() OVER (PARTITION BY gate ORDER BY id DESC) AS rn"
                "  FROM gate_decisions WHERE tenant = ? AND campaign_id = ?"
                f"  AND gate IN ({marks})"
                " ) WHERE rn <= ?"
                ") ORDER BY id DESC",
                (tenant, tenant, campaign_id, *wanted, per_gate),
            ).fetchall()
    out: dict[str, dict[str, Any]] = {
        g: {"count": counts.get(g, (0, 0))[0], "latest": []}
        for g in ([only_gate] if only_gate is not None else [*stage_keys, *orphans])
    }
    for row in rows:
        entry = out.get(str(row[1]))
        if entry is not None:
            entry["latest"].append(_decision_row(row))
    return out, n_orphans


def gate_history(
    store: CampaignStore, *, tenant: str, campaign_id: str, gate: str,
    limit: int = HISTORY_DEFAULT_LIMIT, before_id: int | None = None,
) -> dict[str, Any]:
    """One newest-first page of a gate's decisions (keyset pagination on `id`).

    The gate must be a stage key of the current definitions or have recorded
    decisions (an orphaned gate keeps a readable history); otherwise
    `GateNotFound`. Raises `CampaignNotFound` for a missing or foreign
    campaign (checked first)."""
    limit = max(1, min(int(limit), HISTORY_MAX_LIMIT))
    store._check(tenant, campaign_id)
    with store._txn(write=False) as conn:
        _check_campaign(store, conn, tenant, campaign_id)
        if not valid_stage_key(gate):
            raise GateNotFound()
        if gate not in _stage_keys_in_txn(conn, tenant, campaign_id):
            known = conn.execute(
                "SELECT 1 FROM gate_decisions WHERE tenant = ? AND campaign_id = ? AND gate = ? LIMIT 1",
                (tenant, campaign_id, gate),
            ).fetchone()
            if known is None:
                raise GateNotFound()
        sql = (
            f"SELECT {_DECISION_COLUMNS} FROM gate_decisions "  # noqa: S608 - column list is a module constant; values are bound parameters
            "WHERE tenant = ? AND campaign_id = ? AND gate = ?"
            + (" AND id < ?" if before_id is not None else "")
            + " ORDER BY id DESC LIMIT ?"
        )
        params: tuple[Any, ...] = (
            tenant, campaign_id, gate, *((before_id,) if before_id is not None else ()), limit + 1,
        )
        fetched = conn.execute(sql, params).fetchall()
    items = [_decision_row(r) for r in fetched[:limit]]
    has_more = len(fetched) > limit
    return {
        "items": items,
        "has_more": has_more,
        "next_before_id": items[-1]["id"] if has_more and items else None,
    }


def record_decision(
    store: CampaignStore, *, tenant: str, campaign_id: str, gate: str, decision: str,
    note: str, evidence: list[dict[str, Any]], decided_by: str | None, auth_method: str | None,
) -> dict[str, Any]:
    """Append one decision for `gate`, with its audit row, in ONE transaction.

    Raises `CampaignNotFound` (missing/foreign), `CampaignArchived`,
    `GateNotFound` (the key is not a stage in the current definitions, read in
    the same transaction), or `DecisionLimitReached`."""
    if decision not in GATE_DECISIONS:
        raise StageValidationError("invalid_decision", "decision must be go, no_go or pending")
    store._check(tenant, campaign_id)
    note = check_note(note)
    blob = json.dumps(evidence, sort_keys=True, allow_nan=False)
    who = decided_by[:256] if isinstance(decided_by, str) else None
    method = auth_method[:64] if isinstance(auth_method, str) else None
    now = time.time()
    with store._txn(write=True) as conn:
        row = store._row(conn, tenant, campaign_id)
        if row is None:
            raise CampaignNotFound()
        if row.archived_at is not None:
            raise CampaignArchived()
        if gate not in _stage_keys_in_txn(conn, tenant, campaign_id):
            raise GateNotFound()
        (count,) = conn.execute(
            "SELECT COUNT(*) FROM gate_decisions WHERE tenant = ? AND campaign_id = ?",
            (tenant, campaign_id),
        ).fetchone()
        if int(count) >= MAX_DECISIONS_PER_CAMPAIGN:
            raise DecisionLimitReached()
        cur = conn.execute(
            "INSERT INTO gate_decisions (tenant, campaign_id, gate, decision, note, evidence, "
            "decided_by, auth_method, decided_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (tenant, campaign_id, gate, decision, note, blob, who, method, now),
        )
        new_id = int(cur.lastrowid or 0)
        store._insert_audit(
            conn, tenant=tenant, action="campaign.gate.decide", outcome="ok", subject=decided_by,
            auth_method=auth_method, campaign_id=campaign_id,
            detail=f"gate={gate};decision={decision};evidence={len(evidence)};id={new_id}", now=now,
        )
    return {
        "id": new_id, "gate": gate, "decision": decision, "note": note,
        "evidence_linked": evidence, "decided_by": who, "auth_method": method,
        "decided_at": now, "basis": "recorded",
    }


# --- computed evidence (read-only, never a verdict) --------------------------- #

_COMPUTED_DISCLAIMER = (
    "Computed from the campaign's latest stored analysis. It is campaign-wide, not "
    "specific to this stage, and no go/no-go is derived from it: the gate decision "
    "is the recorded human decision only."
)


def _num(value: Any) -> float | int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if math.isfinite(float(value)) else None


def _bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def computed_evidence(analysis: dict[str, Any] | None) -> dict[str, Any]:
    """A read-only summary of the stored analysis, labeled `basis: "computed"`.
    Fields are copied only when present with the right type; anything absent
    is null (older analyses lack newer fields - never backfilled)."""
    if not analysis:
        return {
            "available": False, "basis": "computed",
            "code": "no_analysis",
            "reason": "this campaign has no stored analysis",
            "what_would_fix_it": "upload a run sheet to the campaign (or reanalyze) to compute one",
        }
    rel = analysis.get("reliability") if isinstance(analysis.get("reliability"), dict) else None
    promo = analysis.get("promotion") if isinstance(analysis.get("promotion"), dict) else None
    ci = rel.get("ci95") if rel else None
    ci95 = [float(x) for x in ci] if (
        isinstance(ci, list) and len(ci) == 2 and all(_num(x) is not None for x in ci)
    ) else None
    failures = promo.get("failures") if promo else None
    target = analysis.get("target")
    return {
        "available": True,
        "basis": "computed",
        "source": "campaign_analysis",
        "scope": "campaign",
        "note": _COMPUTED_DISCLAIMER,
        "analysis_updated_at": _num(analysis.get("updated")),
        "campaign_generation": analysis.get("campaign_generation")
        if isinstance(analysis.get("campaign_generation"), str) else None,
        "target": target if isinstance(target, str) else None,
        # Rows the analysis was fit on (`n` of the stored analysis).
        "n_measured": _num(analysis.get("n")),
        # The analysis's `best`: the largest measured target value it saw.
        "best_measured": {"value": _num(analysis.get("best")), "basis": "measured"},
        "reliability": None if rel is None else {
            "basis": "derived",
            "method": "grouped_cv_spearman",
            "spearman": _num(rel.get("spearman")),
            "ci95": ci95,
            "spearman_floor": _num(rel.get("spearman_floor")),
            "clears_floor": _bool(rel.get("clears_floor")),
            "ci_excludes_zero": _bool(rel.get("ci_excludes_zero")),
        },
        "promotion": None if promo is None else {
            "basis": "derived",
            "passed": _bool(promo.get("passed")),
            "failures": [str(f)[:300] for f in failures][:20] if isinstance(failures, list) else None,
            "summary": str(promo["summary"])[:500] if isinstance(promo.get("summary"), str) else None,
        },
    }
