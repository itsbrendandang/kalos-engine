"""Virtual runs: generate, store, list and export simulated/synthetic rows
(schema v6 `virtual_runs`, docs/VIRTUAL_RUNS.md).

Kept out of `kalos.portal.campaign` on purpose: the campaign store holds lab
measurements; this module holds data that is never measured. It reuses the
store's connection and transaction (`CampaignStore._txn`) so a batch, and its
audit row, commit together or not at all.

Rules enforced here:

- Every statement on `virtual_runs` carries `tenant = ?`; the campaign is looked
  up with `WHERE id = ? AND tenant = ?`, so another tenant's campaign reads as
  missing (`CampaignNotFound`).
- An archived campaign refuses new batches (`CampaignArchived`).
- Rows are only ever inserted (the table rejects UPDATE and DELETE).
- Nothing here hands rows to an analysis. `rows_frame` returns a DataFrame with
  the `origin` column and `attrs["kalos_origin"]`, which every fit entry point
  refuses (kalos/twin/origin.py).
- An export holds ONE origin. A request that would mix origins, or mix virtual
  and measured rows, is refused (`MixedExportRefused`).

Torch-free (generation imports the wrapped bench modules only when it runs).
"""
from __future__ import annotations

import csv
import io
import json
import time
import uuid
from typing import Any

import pandas as pd

from kalos.portal.campaign import (
    CampaignArchived,
    CampaignError,
    CampaignNotFound,
    CampaignStore,
    validate_campaign_id,
)
from kalos.portal.tenancy import validate_tenant_id
from kalos.twin import generators as gen
from kalos.twin.origin import (
    ORIGIN_ATTR,
    ORIGIN_COLUMN,
    VIRTUAL_ORIGINS,
    Origin,
    parse_origin,
)

BATCH_PREFIX = "vjb_"
MAX_LIST_ROWS = 1000
MAX_EXPORT_ROWS = 50_000
MAX_ACTOR_CHARS = 512
OPEN_MODE_ACTOR = "anonymous"
AUDIT_CREATE = "campaign.virtual_runs.create"
AUDIT_EXPORT = "campaign.virtual_runs.export"

_COLUMNS = (
    "id, batch_id, k, origin, generator_id, generator_version, params_json, params_sha256, "
    "seed_entropy, pcg64_state, solver_json, library_versions, twin_version_id, "
    "design_json, design_sha256, outcome_json, requested_by, auth_method, created_at"
)
_JSON_COLUMNS = ("params_json", "seed_entropy", "pcg64_state", "solver_json",
                 "library_versions", "design_json", "outcome_json")
# Leading characters a spreadsheet may read as a formula (CSV injection).
_FORMULA_LEAD = ("=", "+", "-", "@", "\t", "\r")


class MixedExportRefused(CampaignError):
    """The export would hold more than one origin (or measured and virtual rows)."""

    code = "mixed_export_refused"


class NoVirtualRuns(CampaignError):
    """Nothing matches the export request."""

    code = "no_virtual_runs"


def _check(tenant: str, campaign_id: str) -> None:
    validate_tenant_id(tenant)
    try:
        validate_campaign_id(campaign_id)
    except CampaignError:
        raise CampaignNotFound() from None


def _campaign_row(conn: Any, tenant: str, campaign_id: str) -> tuple[Any, ...]:
    row = conn.execute(
        "SELECT id, archived_at FROM campaigns WHERE id = ? AND tenant = ?",
        (campaign_id, tenant),
    ).fetchone()
    if row is None:
        raise CampaignNotFound()
    return row


def _actor(requested_by: str | None) -> str:
    text = (requested_by or "").strip()
    return text[:MAX_ACTOR_CHARS] if text else OPEN_MODE_ACTOR


def new_batch_id() -> str:
    return BATCH_PREFIX + uuid.uuid4().hex


def create_batch(
    store: CampaignStore,
    *,
    tenant: str,
    campaign_id: str,
    generator_id: str,
    params: dict[str, Any] | None,
    n: int,
    requested_by: str | None,
    auth_method: str | None = None,
    twin_version_id: str | None = None,
    audit: bool = False,
    request_id: str | None = None,
) -> dict[str, Any]:
    """Generate `n` rows with `generator_id` and append them to `virtual_runs`.

    With `audit=True` the `campaign.virtual_runs.create` audit row is inserted in
    the SAME transaction as the rows, so a batch that cannot be audited is not
    stored. Its `detail` is a JSON object with `batch_id`, `origin`,
    `generator_id`, `generator_version`, `n`, `params_sha256` and
    `twin_version_id` (never the generated values). The route passes `audit=True`.

    The batch id is minted here and is the "job" component of the seed
    (`[tenant, campaign, twin_version, batch]`). Raises `CampaignNotFound`,
    `CampaignArchived`, or `GeneratorError` (bad generator, params, or n); nothing
    is written then. Returns the batch summary with its rows."""
    _check(tenant, campaign_id)
    # Refuse a missing/foreign/archived campaign before spending compute on it.
    with store._txn(write=False) as conn:
        if _campaign_row(conn, tenant, campaign_id)[1] is not None:
            raise CampaignArchived()
    batch_id = new_batch_id()
    batch = gen.generate(
        generator_id, params, n, tenant=tenant, campaign_id=campaign_id, job_id=batch_id,
        twin_version_id=twin_version_id,
    )
    actor = _actor(requested_by)
    now = time.time()
    common = (
        tenant, campaign_id, batch_id, str(batch.origin), batch.generator_id,
        batch.generator_version, gen.canonical_json(batch.params), batch.params_sha256,
        gen.canonical_json(batch.seed_entropy), gen.canonical_json(batch.pcg64_state),
        gen.canonical_json(batch.solver), gen.canonical_json(batch.library_versions),
        batch.twin_version_id,
    )
    with store._txn(write=True) as conn:
        if _campaign_row(conn, tenant, campaign_id)[1] is not None:
            raise CampaignArchived()  # archived between the check and the write
        conn.executemany(
            "INSERT INTO virtual_runs (tenant, campaign_id, batch_id, origin, generator_id, "
            "generator_version, params_json, params_sha256, seed_entropy, pcg64_state, "
            "solver_json, library_versions, twin_version_id, k, design_json, design_sha256, "
            "outcome_json, requested_by, auth_method, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (*common, r.k, gen.canonical_json(r.design), r.design_sha256,
                 gen.canonical_json(r.outcome), actor, auth_method, now)
                for r in batch.rows
            ],
        )
        if audit:
            detail = json.dumps({
                "batch_id": batch_id, "origin": str(batch.origin),
                "generator_id": batch.generator_id,
                "generator_version": batch.generator_version, "n": len(batch.rows),
                "params_sha256": batch.params_sha256,
                "twin_version_id": batch.twin_version_id,
            }, sort_keys=True)
            store._insert_audit(
                conn, tenant=tenant, action=AUDIT_CREATE, outcome="ok", subject=actor,
                auth_method=auth_method, campaign_id=campaign_id, job_id=batch_id,
                request_id=request_id, detail=detail, now=now,
            )
    return {
        "batch_id": batch_id,
        "campaign_id": campaign_id,
        "origin": str(batch.origin),
        "label": origin_label(batch.origin, batch.twin_version_id),
        "generator_id": batch.generator_id,
        "generator_version": batch.generator_version,
        "params": batch.params,
        "params_sha256": batch.params_sha256,
        "seed_entropy": [str(e) for e in batch.seed_entropy],
        "pcg64_state": batch.pcg64_state,
        "solver": batch.solver,
        "library_versions": batch.library_versions,
        "twin_version_id": batch.twin_version_id,
        "requested_by": actor,
        "created_at": now,
        "n": len(batch.rows),
        "runs": [
            {"k": r.k, "origin": str(batch.origin), "design": r.design, "outcome": r.outcome,
             "design_sha256": r.design_sha256}
            for r in batch.rows
        ],
    }


def origin_label(origin: Origin | str, twin_version_id: str | None = None) -> str:
    """The text shown next to virtual data: `SYNTHETIC - not measured` or
    `SIMULATED - twin <version> - not measured`."""
    o = parse_origin(origin)
    if o is Origin.SIMULATED:
        return f"SIMULATED - twin {twin_version_id or '?'} - not measured"
    return "SYNTHETIC - not measured"


def _decode(row: tuple[Any, ...], names: list[str]) -> dict[str, Any]:
    out = dict(zip(names, row))
    for col in _JSON_COLUMNS:
        out[col] = json.loads(out[col])
    out["seed_entropy"] = [str(e) for e in out["seed_entropy"]]
    return out


def list_runs(
    store: CampaignStore,
    *,
    tenant: str,
    campaign_id: str,
    origin: str | None = None,
    batch_id: str | None = None,
    after: int = 0,
    limit: int = 200,
) -> list[dict[str, Any]]:
    """`tenant`'s virtual rows for one campaign, oldest first, from `id > after`."""
    _check(tenant, campaign_id)
    sql = f"SELECT {_COLUMNS} FROM virtual_runs WHERE tenant = ? AND campaign_id = ? AND id > ?"  # noqa: S608 - column list is a module constant; values are bound parameters
    args: list[Any] = [tenant, campaign_id, int(after)]
    if origin is not None:
        sql += " AND origin = ?"
        args.append(str(parse_origin(origin)))
    if batch_id is not None:
        sql += " AND batch_id = ?"
        args.append(batch_id)
    sql += " ORDER BY id LIMIT ?"
    args.append(max(1, min(int(limit), MAX_LIST_ROWS)))
    with store._txn(write=False) as conn:
        _campaign_row(conn, tenant, campaign_id)
        cur = conn.execute(sql, tuple(args))
        names = [d[0] for d in cur.description]
        return [_decode(r, names) for r in cur.fetchall()]


def rows_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Rows as one DataFrame (design + outcome columns), marked with the origin
    column and `attrs`, so `check_origin` refuses it at any fit entry point."""
    records = [{ORIGIN_COLUMN: r["origin"], **r["design_json"], **r["outcome_json"]} for r in rows]
    df = pd.DataFrame(records)
    origins = {r["origin"] for r in rows}
    if origins:
        df.attrs[ORIGIN_ATTR] = max(origins, key=lambda o: list(Origin).index(Origin(o)))
    return df


def _safe_cell(value: Any) -> Any:
    if isinstance(value, str) and value.startswith(_FORMULA_LEAD):
        return "'" + value
    return value


def export_csv(
    store: CampaignStore,
    *,
    tenant: str,
    campaign_id: str,
    origin: str | None = None,
    batch_id: str | None = None,
    include_measured: bool = False,
) -> tuple[str, str, Origin]:
    """`(filename, csv_text, origin)` for one origin of one campaign's virtual rows.

    Refused (`MixedExportRefused`, nothing returned) when `include_measured` is
    asked for, or when no `origin` is given and the selection holds both
    simulated and synthetic rows. The file starts with `#` comment lines naming the
    origin, the generators, their versions and seed entropy; the first column is
    `origin` on every row; the filename carries the origin."""
    _check(tenant, campaign_id)
    if include_measured:
        raise MixedExportRefused(
            "an export holds one origin: measured and virtual rows are never exported "
            "together. Export the virtual rows on their own."
        )
    sql = f"SELECT {_COLUMNS} FROM virtual_runs WHERE tenant = ? AND campaign_id = ?"  # noqa: S608 - column list is a module constant; values are bound parameters
    args: list[Any] = [tenant, campaign_id]
    if origin is not None:
        sql += " AND origin = ?"
        args.append(str(parse_origin(origin)))
    if batch_id is not None:
        sql += " AND batch_id = ?"
        args.append(batch_id)
    sql += " ORDER BY id LIMIT ?"
    args.append(MAX_EXPORT_ROWS + 1)
    with store._txn(write=False) as conn:
        _campaign_row(conn, tenant, campaign_id)
        cur = conn.execute(sql, tuple(args))
        names = [d[0] for d in cur.description]
        rows = [_decode(r, names) for r in cur.fetchall()]
    if not rows:
        raise NoVirtualRuns("no virtual runs match this export")
    if len(rows) > MAX_EXPORT_ROWS:
        raise CampaignError(
            f"the export would exceed {MAX_EXPORT_ROWS} rows; narrow it with batch_id"
        )
    origins = {parse_origin(r["origin"]) for r in rows}
    if len(origins) > 1:
        raise MixedExportRefused(
            "this campaign has both simulated and synthetic runs; an export holds one "
            "origin. Pass origin=simulated or origin=synthetic."
        )
    (the_origin,) = origins
    assert the_origin in VIRTUAL_ORIGINS
    design_cols: list[str] = []
    outcome_cols: list[str] = []
    for r in rows:
        for c in r["design_json"]:
            if c not in design_cols:
                design_cols.append(c)
        for c in r["outcome_json"]:
            if c not in outcome_cols:
                outcome_cols.append(c)
    meta = ["virtual_batch_id", "virtual_k", "generator_id", "generator_version",
            "twin_version_id"]
    buf = io.StringIO()
    buf.write(f"# ORIGIN: {str(the_origin).upper()}. These rows are NOT lab measurements.\n")
    buf.write(f"# {origin_label(the_origin, rows[0]['twin_version_id'])}\n")
    buf.write(f"# campaign: {campaign_id}; rows: {len(rows)}; "
              "see docs/VIRTUAL_RUNS.md. Kalos refuses to analyse this file.\n")
    seen: set[str] = set()
    for r in rows:
        if r["batch_id"] in seen:
            continue
        seen.add(r["batch_id"])
        buf.write(
            f"# batch {r['batch_id']}: generator {r['generator_id']} {r['generator_version']}; "
            f"params_sha256 {r['params_sha256']}; seed_entropy {json.dumps(r['seed_entropy'])}; "
            f"twin_version {r['twin_version_id'] or 'none'}\n"
        )
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow([ORIGIN_COLUMN, *meta, *design_cols, *outcome_cols])
    for r in rows:
        writer.writerow([_safe_cell(v) for v in (
            r["origin"], r["batch_id"], r["k"], r["generator_id"], r["generator_version"],
            r["twin_version_id"] or "",
            *[r["design_json"].get(c, "") for c in design_cols],
            *[("" if r["outcome_json"].get(c) is None else r["outcome_json"][c])
              for c in outcome_cols],
        )])
    filename = f"kalos-{the_origin}-virtual-runs-{campaign_id}.csv"
    return filename, buf.getvalue(), the_origin


__all__ = [
    "MixedExportRefused",
    "NoVirtualRuns",
    "create_batch",
    "export_csv",
    "list_runs",
    "new_batch_id",
    "origin_label",
    "rows_frame",
]
