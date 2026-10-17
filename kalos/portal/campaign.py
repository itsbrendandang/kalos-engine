"""Kalos portal — the campaign store: closed optimization loops, several per tenant.

See docs/CAMPAIGN_LOOP.md for the loop contract this module implements. A
*campaign* is one optimization target plus a growing base dataset
(`base_rows`) and a list of `pending` runs — recipes started from a proposed
batch, each either *awaiting* a measured outcome (`result: null`) or
*measured* (`result` filled), waiting to be folded into `base_rows` on the
next re-analyze.

Deliberately kept torch-free: this module only persists data. `_analyze`
(`kalos.portal.analysis`) is the science; `kalos.portal.campaign_routes`
calls it, not this store, so importing `kalos.portal.campaign` never pays the
torch/botorch import tax and the store stays trivially unit-testable.

Persistence is `<KALOS_STATE_DIR>/portal.db` (default `~/.kalos`) at schema v2
(`kalos.portal.migrate`, docs/MIGRATIONS.md): a tenant owns any number of
campaigns (`cmp_<32 hex>` ids), each campaign owns at most one published
analysis (what `/api/latest` serves), and `tenant_defaults` names the campaign
the legacy single-campaign routes act on. The store migrates a v1 database on
first open, after taking a verified backup.

Isolation and concurrency rules, all enforced here:

- Every public method takes a REQUIRED keyword `tenant` (no default, so a call
  site cannot fall into the open-mode `default` tenant by omission), validated
  against `^[A-Za-z0-9_-]{1,64}$`. Every SQL statement on a tenant table
  carries `tenant = ?` (tests/test_campaign_schema_v2.py checks this), so a
  campaign id belonging to another tenant reads exactly like a missing one.
- An explicit `campaign_id` is validated against `^cmp_[0-9a-f]{32}$` before
  any SQL runs. `campaign_id=None` means "this tenant's default campaign",
  which is what the legacy routes use.
- Every read-modify-write runs in one `BEGIN IMMEDIATE` transaction (not only
  under the in-process lock), so a second process or connection cannot
  interleave. The write itself is a compare-and-swap on the `generation`
  column (`... AND generation IS ?`), and the analysis a fold or seed
  publishes is written in the same transaction as the campaign.
- An archived campaign refuses writes with `CampaignArchived` (HTTP 409).
"""
from __future__ import annotations

import json
import logging
import math
import os
import sqlite3
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Any

import pandas as pd

from kalos.portal.migrate import (
    CAMPAIGN_ID_PATTERN,
    DB_FILENAME,
    campaigns_table_is_v1_compatible as _campaigns_table_is_compatible,
    connect,
    ensure_schema,
)
from kalos.portal.tenancy import validate_tenant_id

log = logging.getLogger("kalos.portal")

DEFAULT_CAMPAIGN_NAME = "Default campaign"
LEGACY_CREATOR = "legacy-route"
MAX_CAMPAIGN_NAME_CHARS = 200
MAX_AUDIT_FIELD_CHARS = 512

__all__ = [
    "CampaignArchived",
    "CampaignConflict",
    "CampaignError",
    "CampaignNotFound",
    "CampaignStore",
    "TenantCampaigns",
    "get_campaign_store",
    "validate_campaign_id",
    "_campaigns_table_is_compatible",
]


class CampaignError(ValueError):
    """A client-facing campaign operation error: unknown run id, non-finite
    result value, or no active campaign. The message is authored here (never
    a raw parser/library error), so it is always safe for a route handler to
    echo back to the caller as-is."""


class CampaignNotFound(CampaignError):
    """No such campaign FOR THIS TENANT. A malformed id, a missing id and
    another tenant's id all raise this with the same message, so a caller can
    never probe for the existence of someone else's campaign."""

    def __init__(self) -> None:
        super().__init__("campaign not found")


class CampaignArchived(CampaignError):
    """The campaign is archived and refuses writes (HTTP 409)."""

    def __init__(self) -> None:
        super().__init__("the campaign is archived; it can be read but not changed")


class CampaignConflict(CampaignError):
    """The compare-and-swap failed: the campaign changed since it was read."""


def validate_campaign_id(campaign_id: object) -> str:
    """Return `campaign_id` if it is `cmp_` + 32 lowercase hex, else raise
    `CampaignNotFound` (the same error as a missing id; the input is not echoed)."""
    if not isinstance(campaign_id, str) or CAMPAIGN_ID_PATTERN.fullmatch(campaign_id) is None:
        raise CampaignNotFound()
    return campaign_id


def _finite_number(value: Any) -> float | None:
    """`value` as a float if it is a real, finite number (excluding bool,
    which is technically an `int` subclass but never a measured outcome),
    else None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    v = float(value)
    return v if math.isfinite(v) else None


def _best_measured(base_rows: list[dict[str, Any]], target: str) -> float | None:
    """The best (max) MEASURED target value over `base_rows`, or None when no
    row carries a finite numeric target — never a prediction (docs/CAMPAIGN_LOOP.md,
    "Honesty constraints"). Because base rows only ever grow, this is
    non-decreasing round over round, which is what the progress trajectory plots."""
    values = [v for row in base_rows if (v := _finite_number(row.get(target))) is not None]
    return max(values) if values else None


def _clip(value: str | None, limit: int = MAX_AUDIT_FIELD_CHARS) -> str | None:
    return None if value is None else str(value)[:limit]


class _Row:
    """One `campaigns` row as read inside a transaction."""

    __slots__ = ("id", "name", "state_text", "generation", "created_by",
                 "created_at", "updated_at", "archived_at")

    def __init__(self, row: tuple[Any, ...]) -> None:
        (self.id, self.name, self.state_text, self.generation, self.created_by,
         self.created_at, self.updated_at, self.archived_at) = row

    def state(self) -> dict[str, Any] | None:
        """The parsed campaign state, with `generation` taken from the column
        (the CAS source of truth). None for an unseeded or corrupt row: a corrupt
        row must not break the API, and is left in place, never overwritten by a
        read."""
        if self.state_text is None:
            return None
        try:
            data = json.loads(self.state_text)
        except ValueError:
            return None
        if not isinstance(data, dict):
            return None
        if self.generation is not None:
            data["generation"] = self.generation
        else:
            data.pop("generation", None)
        return data

    def info(self, *, is_default: bool) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "created_by": self.created_by,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "archived_at": self.archived_at,
            "archived": self.archived_at is not None,
            "seeded": self.state_text is not None,
            "is_default": is_default,
        }


_ROW_COLUMNS = "id, name, state, generation, created_by, created_at, updated_at, archived_at"


class CampaignStore:
    """Persists campaigns, their analyses, and the tenant default in
    `<state_dir>/portal.db` (schema v2). See the module docstring for the
    isolation and concurrency rules."""

    def __init__(self, state_dir: Path | None = None) -> None:
        self._dir = Path(state_dir) if state_dir is not None else Path(
            os.environ.get("KALOS_STATE_DIR", Path.home() / ".kalos")
        )
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / DB_FILENAME
        # Migrates a v1 database (backup first) or creates a fresh v2 one;
        # raises MigrationError / SchemaError rather than run on a bad schema.
        ensure_schema(self._path, latest_dir=self._dir / "latest")
        self._lock = threading.Lock()
        self._conn = connect(self._path, check_same_thread=False)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # --- transactions ------------------------------------------------------- #

    @contextmanager
    def _txn(self, *, write: bool) -> Iterator[sqlite3.Connection]:
        """One SQLite transaction. Writes use `BEGIN IMMEDIATE`, which takes the
        database write lock up front: a concurrent writer in another connection
        or process waits (busy_timeout) instead of interleaving its
        read-modify-write with ours. The threading lock only serializes this
        process's use of the one shared connection."""
        with self._lock:
            conn = self._conn
            conn.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            try:
                yield conn
            except BaseException:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                raise
            try:
                conn.execute("COMMIT")
            except BaseException:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                raise

    # --- in-transaction helpers (every statement carries tenant = ?) --------- #

    @staticmethod
    def _default_id(conn: sqlite3.Connection, tenant: str) -> str | None:
        row = conn.execute(
            "SELECT campaign_id FROM tenant_defaults WHERE tenant = ?", (tenant,)
        ).fetchone()
        return str(row[0]) if row is not None else None

    @staticmethod
    def _row(conn: sqlite3.Connection, tenant: str, campaign_id: str) -> _Row | None:
        row = conn.execute(
            f"SELECT {_ROW_COLUMNS} FROM campaigns WHERE id = ? AND tenant = ?",
            (campaign_id, tenant),
        ).fetchone()
        return _Row(row) if row is not None else None

    @staticmethod
    def _insert_campaign(
        conn: sqlite3.Connection, tenant: str, name: str, created_by: str | None,
        *, make_default: bool,
    ) -> str:
        cid = "cmp_" + uuid.uuid4().hex
        now = time.time()
        conn.execute(
            "INSERT INTO campaigns (id, tenant, name, state, generation, created_by, "
            "created_at, updated_at, archived_at) VALUES (?, ?, ?, NULL, NULL, ?, ?, ?, NULL)",
            (cid, tenant, name, created_by, now, now),
        )
        if make_default:
            conn.execute(
                "INSERT INTO tenant_defaults (tenant, campaign_id) VALUES (?, ?) "
                "ON CONFLICT(tenant) DO UPDATE SET campaign_id = excluded.campaign_id "
                "WHERE tenant_defaults.tenant = excluded.tenant",
                (tenant, cid),
            )
        return cid

    def _resolve(
        self, conn: sqlite3.Connection, tenant: str, campaign_id: str | None, *, create: bool
    ) -> _Row | None:
        """The row for an explicit `campaign_id` (CampaignNotFound if it is not
        this tenant's), or for the tenant default when `campaign_id` is None —
        created on demand when `create` (the legacy "first upload" path), else
        None if the tenant has no default yet."""
        if campaign_id is not None:
            row = self._row(conn, tenant, campaign_id)
            if row is None:
                raise CampaignNotFound()
            return row
        cid = self._default_id(conn, tenant)
        if cid is None:
            if not create:
                return None
            cid = self._insert_campaign(
                conn, tenant, DEFAULT_CAMPAIGN_NAME, LEGACY_CREATOR, make_default=True
            )
        return self._row(conn, tenant, cid)

    @staticmethod
    def _cas_write(
        conn: sqlite3.Connection, tenant: str, row: _Row, state: dict[str, Any],
        expected_generation: str | None,
    ) -> str:
        """Persist `state` iff the row still has `expected_generation` and is
        not archived, minting a fresh generation token. The token is how a
        re-analyze detects any intervening write (docs/CAMPAIGN_LOOP.md,
        "Re-analyze = the loop closing")."""
        new_generation = uuid.uuid4().hex
        state["generation"] = new_generation
        cur = conn.execute(
            "UPDATE campaigns SET state = ?, generation = ?, updated_at = ? "
            "WHERE id = ? AND tenant = ? AND archived_at IS NULL AND generation IS ?",
            (json.dumps(state), new_generation, time.time(), row.id, tenant, expected_generation),
        )
        if cur.rowcount != 1:
            raise CampaignConflict(
                "the campaign changed during this operation; nothing was written, please retry"
            )
        return new_generation

    @staticmethod
    def _write_analysis(
        conn: sqlite3.Connection, tenant: str, campaign_id: str, blob: str,
        dataset: str | None, campaign_generation: str | None, now: float,
    ) -> None:
        conn.execute(
            "INSERT INTO analyses (campaign_id, tenant, state, dataset, campaign_generation, "
            "updated_at) VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(campaign_id) DO UPDATE SET state = excluded.state, "
            "dataset = excluded.dataset, campaign_generation = excluded.campaign_generation, "
            "updated_at = excluded.updated_at WHERE analyses.tenant = excluded.tenant",
            (campaign_id, tenant, blob, dataset, campaign_generation, now),
        )

    @staticmethod
    def _analysis_blob(
        result: dict[str, Any], dataset: str, campaign_generation: str | None, now: float
    ) -> str | None:
        """The `/api/latest` payload as stored. None (logged) if it cannot be
        serialized: publishing an analysis is best-effort and must never fail
        the upload or the fold it accompanies."""
        state = {**result, "dataset": dataset, "updated": now,
                 "campaign_generation": campaign_generation}
        try:
            return json.dumps(state)
        except (TypeError, ValueError):
            log.exception("analysis could not be serialized; not published")
            return None

    @staticmethod
    def _check(tenant: str, campaign_id: str | None) -> None:
        """Validate ids BEFORE any SQL runs."""
        validate_tenant_id(tenant)
        if campaign_id is not None:
            validate_campaign_id(campaign_id)

    # --- campaign management (campaign-scoped: explicit ids) ----------------- #

    def create_campaign(
        self, *, tenant: str, name: str, created_by: str | None = None,
        make_default: bool = False,
    ) -> dict[str, Any]:
        """Create an empty (unseeded) campaign for `tenant`. Returns its info."""
        validate_tenant_id(tenant)
        if not isinstance(name, str) or not name.strip():
            raise CampaignError("campaign name must be a non-empty string")
        name = name.strip()[:MAX_CAMPAIGN_NAME_CHARS]
        with self._txn(write=True) as conn:
            cid = self._insert_campaign(
                conn, tenant, name, _clip(created_by), make_default=make_default
            )
            row = self._row(conn, tenant, cid)
            assert row is not None
            return row.info(is_default=self._default_id(conn, tenant) == cid)

    def campaign_info(self, *, tenant: str, campaign_id: str) -> dict[str, Any]:
        """Metadata for one of `tenant`'s campaigns; CampaignNotFound otherwise."""
        self._check(tenant, campaign_id)
        with self._txn(write=False) as conn:
            row = self._resolve(conn, tenant, campaign_id, create=False)
            assert row is not None
            return row.info(is_default=self._default_id(conn, tenant) == row.id)

    def list_campaigns(
        self, *, tenant: str, include_archived: bool = False
    ) -> list[dict[str, Any]]:
        """`tenant`'s campaigns, newest first."""
        validate_tenant_id(tenant)
        sql = f"SELECT {_ROW_COLUMNS} FROM campaigns WHERE tenant = ?"
        if not include_archived:
            sql += " AND archived_at IS NULL"
        sql += " ORDER BY updated_at DESC, id"
        with self._txn(write=False) as conn:
            default = self._default_id(conn, tenant)
            return [_Row(r).info(is_default=r[0] == default)
                    for r in conn.execute(sql, (tenant,)).fetchall()]

    def default_campaign_id(self, *, tenant: str) -> str | None:
        """The campaign the legacy single-campaign routes act on, if any."""
        validate_tenant_id(tenant)
        with self._txn(write=False) as conn:
            return self._default_id(conn, tenant)

    def archive_campaign(self, *, tenant: str, campaign_id: str) -> dict[str, Any]:
        """Archive a campaign: it stays readable, refuses writes (409). If it was
        the tenant default, the default is cleared, so the next legacy upload
        starts a fresh default campaign instead of failing. Idempotent."""
        self._check(tenant, campaign_id)
        with self._txn(write=True) as conn:
            row = self._resolve(conn, tenant, campaign_id, create=False)
            assert row is not None
            if row.archived_at is None:
                conn.execute(
                    "UPDATE campaigns SET archived_at = ? "
                    "WHERE id = ? AND tenant = ? AND archived_at IS NULL",
                    (time.time(), campaign_id, tenant),
                )
                conn.execute(
                    "DELETE FROM tenant_defaults WHERE tenant = ? AND campaign_id = ?",
                    (tenant, campaign_id),
                )
            row = self._row(conn, tenant, campaign_id)
            assert row is not None
            return row.info(is_default=False)

    def update_campaign(
        self, *, tenant: str, campaign_id: str, name: str | None = None,
        archived: bool | None = None,
    ) -> dict[str, Any]:
        """Rename and/or (un)archive one of `tenant`'s campaigns, in ONE transaction.

        Order inside the transaction: unarchive, rename, archive, so
        `{name, archived: false}` revives and renames an archived campaign and
        `{name, archived: true}` renames and then archives. Renaming an archived
        campaign (without unarchiving it) raises `CampaignArchived`. Archiving
        clears the tenant default exactly like `archive_campaign`; unarchiving
        does not restore it. CampaignNotFound for any id that is not this
        tenant's. Name validation (length, control characters) is the route's
        job; here it is only stripped and clipped like `create_campaign`.
        """
        self._check(tenant, campaign_id)
        if name is not None:
            if not isinstance(name, str) or not name.strip():
                raise CampaignError("campaign name must be a non-empty string")
            name = name.strip()[:MAX_CAMPAIGN_NAME_CHARS]
        with self._txn(write=True) as conn:
            row = self._resolve(conn, tenant, campaign_id, create=False)
            assert row is not None
            now = time.time()
            if archived is False and row.archived_at is not None:
                conn.execute(
                    "UPDATE campaigns SET archived_at = NULL, updated_at = ? "
                    "WHERE id = ? AND tenant = ?",
                    (now, campaign_id, tenant),
                )
                row.archived_at = None
            if name is not None:
                if row.archived_at is not None:
                    raise CampaignArchived()
                conn.execute(
                    "UPDATE campaigns SET name = ?, updated_at = ? "
                    "WHERE id = ? AND tenant = ? AND archived_at IS NULL",
                    (name, now, campaign_id, tenant),
                )
            if archived is True and row.archived_at is None:
                conn.execute(
                    "UPDATE campaigns SET archived_at = ? "
                    "WHERE id = ? AND tenant = ? AND archived_at IS NULL",
                    (now, campaign_id, tenant),
                )
                conn.execute(
                    "DELETE FROM tenant_defaults WHERE tenant = ? AND campaign_id = ?",
                    (tenant, campaign_id),
                )
            row = self._row(conn, tenant, campaign_id)
            assert row is not None
            return row.info(is_default=self._default_id(conn, tenant) == campaign_id)

    def for_tenant(self, tenant: str) -> TenantCampaigns:
        """A view of this store bound to one tenant (every call gets `tenant=`)."""
        return TenantCampaigns(self, tenant)

    # --- the loop (campaign_id=None => the tenant's default campaign) -------- #

    def seed(
        self, df: pd.DataFrame, target: str, features: list[str], *, tenant: str,
        campaign_id: str | None = None, analysis: dict[str, Any] | None = None,
        dataset: str | None = None,
    ) -> str:
        """Start a FRESH campaign state, overwriting the campaign's current one.
        Returns the `generation` token minted for it.

        Called after every successful `/api/run` upload: a new upload is a
        new base dataset, so any in-flight pending runs from a previous
        campaign are discarded rather than silently folded into the wrong
        dataset. `target`/`features` come from the analysis result
        (`result["target"]`, `result["proposal_features"]`), never guessed
        from column names — see docs/CAMPAIGN_LOOP.md, "Seeding". With
        `campaign_id=None` the tenant's default campaign is used, and created
        on the tenant's first upload.

        When `analysis` is given it is published as the campaign's analysis in
        the SAME transaction, stamped with the new generation, so `/api/latest`
        and `/api/campaign` can never describe different run sheets
        (docs/CAMPAIGN_LOOP.md, "Analysis/campaign coherence").
        """
        from kalos.portal.serialization import _json_safe_records  # local: avoids a cycle at import time

        self._check(tenant, campaign_id)
        base_rows = _json_safe_records(df)
        now = time.time()
        state: dict[str, Any] = {
            "target": target,
            "features": list(features),
            "base_rows": base_rows,
            "pending": [],
            "round": 0,
            # Progress trajectory: best-so-far at each round, starting at
            # round 0 (the uploaded base). Each reanalyze appends a point.
            "history": [{"round": 0, "best": _best_measured(base_rows, target), "n_base": len(base_rows)}],
            "updated_at": now,
        }
        with self._txn(write=True) as conn:
            row = self._resolve(conn, tenant, campaign_id, create=True)
            assert row is not None
            if row.archived_at is not None:
                raise CampaignArchived()
            generation = self._cas_write(conn, tenant, row, state, row.generation)
            if analysis is not None:
                blob = self._analysis_blob(analysis, dataset or "uploaded dataset", generation, now)
                if blob is not None:
                    self._write_analysis(conn, tenant, row.id, blob, dataset, generation, now)
            return generation

    def ping(self) -> None:
        """Readiness check: a read-only, tenant-agnostic query against the
        store's schema bookkeeping (never any tenant's data). Raises whatever
        sqlite3 raises when the database is missing, corrupt or locked."""
        with self._lock:
            self._conn.execute("SELECT MAX(version) FROM schema_migrations").fetchall()

    def get(self, *, tenant: str, campaign_id: str | None = None) -> dict[str, Any] | None:
        """The raw campaign state, or None if none seeded yet."""
        self._check(tenant, campaign_id)
        with self._txn(write=False) as conn:
            row = self._resolve(conn, tenant, campaign_id, create=False)
            return row.state() if row is not None else None

    def summary(
        self, *, tenant: str, campaign_id: str | None = None,
        analysis_generation: str | None = None,
    ) -> dict[str, Any]:
        """The `GET /api/campaign` response: `{"has_campaign": False}` before
        any upload has ever seeded a campaign, else the full summary
        (docs/CAMPAIGN_LOOP.md, "GET /api/campaign response").

        `best` is `max(target over base_rows)` — the best MEASURED value so
        far, never a prediction (see "Honesty constraints" in the design
        doc) — and is None when there is no base data yet or the target
        column carries no finite numeric value.

        `analysis_generation` is the stamp the caller read off `/api/latest`.
        `analysis_in_sync` is True only when it matches this campaign's live
        generation. A missing or stale stamp reads as False: an analysis that
        cannot be shown to match is reported as out of sync rather than assumed
        fine.
        """
        state = self.get(tenant=tenant, campaign_id=campaign_id)
        if state is None:
            return {"has_campaign": False}

        target = state["target"]
        base_rows: list[dict[str, Any]] = state["base_rows"]
        best = _best_measured(base_rows, target)

        pending = [{**run, "awaiting": run["result"] is None} for run in state["pending"]]
        n_awaiting = sum(1 for run in pending if run["awaiting"])
        return {
            "has_campaign": True,
            "target": target,
            "features": state["features"],
            "n_base": len(base_rows),
            "best": best,
            "round": state["round"],
            # Older campaigns (seeded before history existed) fall back to a
            # single current point so the frontend always has something to plot.
            "history": state.get("history")
            or [{"round": state["round"], "best": best, "n_base": len(base_rows)}],
            "pending": pending,
            "n_awaiting": n_awaiting,
            "n_measured": len(pending) - n_awaiting,
            "analysis_in_sync": (
                analysis_generation is not None
                and analysis_generation == state.get("generation")
            ),
        }

    def _writable_state(
        self, conn: sqlite3.Connection, tenant: str, campaign_id: str | None, missing: str
    ) -> tuple[_Row, dict[str, Any]]:
        row = self._resolve(conn, tenant, campaign_id, create=False)
        if row is not None and row.archived_at is not None:
            raise CampaignArchived()
        state = row.state() if row is not None else None
        if row is None or state is None:
            raise CampaignError(missing)
        return row, state

    def start(
        self, recipes: list[dict[str, Any]], *, tenant: str, campaign_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Append each proposed recipe as a pending, awaiting run. Returns
        the appended runs (each with its assigned `id`)."""
        self._check(tenant, campaign_id)
        with self._txn(write=True) as conn:
            row, state = self._writable_state(
                conn, tenant, campaign_id, "no active campaign; upload a run sheet first"
            )
            # Recipe keys are restricted to the campaign's own feature columns:
            # the dataset's columns minus the target. A recipe is folded back
            # into the base dataset as `{**recipe, target: result}`, so an
            # unknown key would invent a new column in the tenant's persisted
            # data. (`features` is only the displayed drivers; the engine's
            # proposals carry every design dimension, which are all columns of
            # the base dataset.)
            target = state["target"]
            allowed = set(state.get("features") or [])
            for base in state.get("base_rows") or []:
                allowed.update(base.keys())
            allowed.discard(target)
            now = time.time()
            appended: list[dict[str, Any]] = []
            # Each value must fit the kind of column it lands in (numeric vs
            # categorical level, read from base_rows); see kalos.portal.recipe_values.
            from kalos.portal.recipe_values import check_recipe_values, column_kinds

            kinds = column_kinds(state.get("base_rows") or [], allowed)
            for recipe in recipes:
                # Validate the recipe mapping HERE, at the point of the bad
                # input, so a malformed body 400s cleanly at /api/campaign/start
                # rather than surfacing as an unhandled 500 rounds later inside
                # the fold's `{**recipe, target: result}` expansion.
                recipe_map = recipe.get("recipe")
                if not isinstance(recipe_map, dict) or not recipe_map:
                    raise CampaignError(
                        "each recipe must include a non-empty 'recipe' mapping of feature -> value"
                    )
                unknown = [k for k in recipe_map if k not in allowed]
                if unknown:
                    # The offending names are not echoed: they are caller input.
                    raise CampaignError(
                        f"recipe has {len(unknown)} key(s) that are not feature columns of "
                        "this campaign's dataset (the target column is not a feature)"
                    )
                recipe_map, problem = check_recipe_values(recipe_map, kinds)
                if problem:
                    raise CampaignError(problem)
                run = {
                    "id": uuid.uuid4().hex,
                    "recipe": recipe_map,
                    "pred": recipe.get("pred"),
                    "std": recipe.get("std"),
                    "mode": recipe.get("mode"),
                    "reason": recipe.get("reason"),
                    "result": None,
                    "created_at": now,
                    "measured_at": None,
                }
                state["pending"].append(run)
                appended.append(run)
            state["updated_at"] = now
            self._cas_write(conn, tenant, row, state, row.generation)
            return appended

    def set_result(
        self, run_id: str, value: float, *, tenant: str, campaign_id: str | None = None
    ) -> dict[str, Any]:
        """Set the measured outcome on a pending run. Raises `CampaignError`
        (never silently ignored) if `run_id` is unknown or `value` is not a
        finite number — a NaN/inf result must never be foldable into the
        base dataset on the next re-analyze."""
        if not math.isfinite(value):
            raise CampaignError(f"result value must be a finite number; got {value!r}")
        self._check(tenant, campaign_id)
        with self._txn(write=True) as conn:
            row, state = self._writable_state(
                conn, tenant, campaign_id, f"no active campaign; unknown run id {run_id!r}"
            )
            for run in state["pending"]:
                if run["id"] == run_id:
                    now = time.time()
                    run["result"] = value
                    run["measured_at"] = now
                    state["updated_at"] = now
                    self._cas_write(conn, tenant, row, state, row.generation)
                    return dict(run)
            raise CampaignError(f"unknown pending run id {run_id!r}")

    def plan_fold(
        self, *, tenant: str, campaign_id: str | None = None
    ) -> tuple[pd.DataFrame, str, str | None]:
        """Compute the folded base dataset WITHOUT persisting anything.

        Returns `(DataFrame(folded_base_rows), target, generation)` where the
        DataFrame is `base_rows` plus every measured pending run folded in, and
        `generation` is the token of the campaign state this plan was built
        from. The caller (`kalos.portal.campaign_routes`) runs `_analyze` on
        the DataFrame and, only if that succeeds, calls `commit_fold(generation)`
        to make the fold durable. A campaign migrated from a state that never
        had a generation flows back as `None`, which `commit_fold` compares like
        any other token (`generation IS NULL`), and its write mints a real one.

        Nothing is mutated or written here. Raises `CampaignError` if there is
        no active campaign or no measured run to fold, and `CampaignArchived`
        for an archived campaign (the fold could never be committed).
        """
        self._check(tenant, campaign_id)
        with self._txn(write=False) as conn:
            _row, state = self._writable_state(
                conn, tenant, campaign_id, "no active campaign; upload a run sheet first"
            )
        target = state["target"]
        folded = list(state["base_rows"])
        measured = 0
        for run in state["pending"]:
            if run["result"] is not None:
                folded.append({**run["recipe"], target: run["result"]})
                measured += 1
        if measured == 0:
            raise CampaignError("no measured runs to fold; log at least one result first")
        return pd.DataFrame(folded), target, state.get("generation")

    def awaiting_recipes(
        self, *, tenant: str, campaign_id: str | None = None
    ) -> list[dict[str, Any]]:
        """The recipes that have been STARTED but not yet measured, as plain dicts.

        These are the runs currently in the incubator: proposed, dispensed, and
        waiting on an assay. The next proposal must know they are taken, or the
        loop spends part of its budget re-running an experiment already in
        progress. The re-analyze route hands these to `_analyze` as its
        `pending` block. Returns `[]` when there is no campaign or nothing is
        awaiting.
        """
        state = self.get(tenant=tenant, campaign_id=campaign_id)
        if state is None:
            return []
        return [
            dict(run["recipe"])
            for run in state["pending"]
            if run["result"] is None and isinstance(run.get("recipe"), dict)
        ]

    def commit_fold(
        self, generation: str | None, *, tenant: str, campaign_id: str | None = None,
        analysis: dict[str, Any] | None = None, dataset: str | None = None,
    ) -> dict[str, Any]:
        """Make the fold planned by `plan_fold` durable, but ONLY if the
        campaign has not changed since: the UPDATE is a compare-and-swap on
        `generation IS <the planned generation>`.

        Folds every measured pending run into `base_rows`, drops them from
        `pending` (awaiting runs stay), increments `round`, appends the
        progress-trajectory point, and persists. When `analysis` is given it is
        published in the same transaction, stamped with the new generation and
        labelled `dataset` (default `campaign round <n>`), so the fold and the
        analysis that describes it land together or not at all. Returns the
        updated state.

        Raises `CampaignError` if the campaign was reseeded or otherwise
        written underneath the in-flight re-analysis; nothing is written then.
        """
        self._check(tenant, campaign_id)
        changed = (
            "the campaign changed during re-analysis (a new upload or result "
            "landed); nothing was committed — please re-analyze again"
        )
        with self._txn(write=True) as conn:
            row, state = self._writable_state(
                conn, tenant, campaign_id, "no active campaign; nothing to commit"
            )
            if row.generation != generation:
                raise CampaignConflict(changed)
            target = state["target"]
            still_pending: list[dict[str, Any]] = []
            for run in state["pending"]:
                if run["result"] is not None:
                    state["base_rows"].append({**run["recipe"], target: run["result"]})
                else:
                    still_pending.append(run)
            state["pending"] = still_pending
            state["round"] += 1
            # Record the new best-so-far for the progress trajectory. base_rows
            # only grew, so this point is >= the previous one.
            history = state.setdefault("history", [])
            history.append(
                {
                    "round": state["round"],
                    "best": _best_measured(state["base_rows"], target),
                    "n_base": len(state["base_rows"]),
                }
            )
            now = time.time()
            state["updated_at"] = now
            try:
                new_generation = self._cas_write(conn, tenant, row, state, generation)
            except CampaignConflict:
                raise CampaignConflict(changed) from None
            if analysis is not None:
                label = dataset or f"campaign round {state['round']}"
                blob = self._analysis_blob(analysis, label, new_generation, now)
                if blob is not None:
                    self._write_analysis(conn, tenant, row.id, blob, label, new_generation, now)
            return state

    # --- analyses (what /api/latest serves) ---------------------------------- #

    def save_analysis(
        self, result: dict[str, Any], dataset: str, *, tenant: str,
        campaign_id: str | None = None, campaign_generation: str | None = None,
    ) -> None:
        """Publish `result` as the campaign's analysis, stamped with
        `campaign_generation` (None = not tied to any campaign state, which
        reads as out of sync). With `campaign_id=None` the tenant default is
        used, created (unseeded) if the tenant has none yet."""
        self._check(tenant, campaign_id)
        now = time.time()
        blob = self._analysis_blob(result, dataset, campaign_generation, now)
        if blob is None:
            return
        with self._txn(write=True) as conn:
            row = self._resolve(conn, tenant, campaign_id, create=True)
            assert row is not None
            if row.archived_at is not None:
                raise CampaignArchived()
            self._write_analysis(conn, tenant, row.id, blob, dataset, campaign_generation, now)

    def load_analysis(
        self, *, tenant: str, campaign_id: str | None = None
    ) -> dict[str, Any] | None:
        """The campaign's published analysis, or None. A corrupt row reads as None."""
        self._check(tenant, campaign_id)
        with self._txn(write=False) as conn:
            row = self._resolve(conn, tenant, campaign_id, create=False)
            if row is None:
                return None
            arow = conn.execute(
                "SELECT state FROM analyses WHERE campaign_id = ? AND tenant = ?",
                (row.id, tenant),
            ).fetchone()
        if arow is None:
            return None
        try:
            data = json.loads(arow[0])
        except ValueError:
            return None
        return data if isinstance(data, dict) else None

    # --- audit log ----------------------------------------------------------- #

    def record_audit(
        self, *, tenant: str, action: str, outcome: str, subject: str | None = None,
        auth_method: str | None = None, campaign_id: str | None = None,
        job_id: str | None = None, request_id: str | None = None, detail: str | None = None,
    ) -> None:
        """Append one audit entry. Callers pass identifiers and short codes only:
        never a token, a credential, or uploaded data. Every field is clipped."""
        validate_tenant_id(tenant)
        with self._txn(write=True) as conn:
            conn.execute(
                "INSERT INTO audit_log (ts, tenant, subject, auth_method, action, campaign_id, "
                "job_id, outcome, request_id, detail) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (time.time(), tenant, _clip(subject), _clip(auth_method), _clip(action, 64),
                 _clip(campaign_id, 64), _clip(job_id, 64), _clip(outcome, 64),
                 _clip(request_id, 128), _clip(detail)),
            )

    def audit_entries(self, *, tenant: str, limit: int = 100) -> list[dict[str, Any]]:
        """`tenant`'s most recent audit entries, newest first."""
        validate_tenant_id(tenant)
        with self._txn(write=False) as conn:
            cur = conn.execute(
                "SELECT id, ts, tenant, subject, auth_method, action, campaign_id, job_id, "
                "outcome, request_id, detail FROM audit_log WHERE tenant = ? "
                "ORDER BY id DESC LIMIT ?",
                (tenant, max(1, min(int(limit), 1000))),
            )
            names = [d[0] for d in cur.description]
            return [dict(zip(names, r)) for r in cur.fetchall()]


_TENANT_METHODS = frozenset({
    "create_campaign", "campaign_info", "list_campaigns", "default_campaign_id",
    "archive_campaign", "update_campaign", "seed", "get", "summary", "start", "set_result", "plan_fold",
    "awaiting_recipes", "commit_fold", "save_analysis", "load_analysis",
    "record_audit", "audit_entries",
})


class TenantCampaigns:
    """`store.for_tenant(t)`: every tenant-scoped store method with `tenant=t`
    pre-bound, so code holding the view cannot address another tenant."""

    def __init__(self, store: CampaignStore, tenant: str) -> None:
        self._store = store
        self.tenant = validate_tenant_id(tenant)

    def __getattr__(self, name: str) -> Any:
        if name not in _TENANT_METHODS:
            raise AttributeError(name)
        return partial(getattr(self._store, name), tenant=self.tenant)


# --- module-level singleton ------------------------------------------------- #
_STORE: CampaignStore | None = None


def get_campaign_store() -> CampaignStore:
    """The module-level `CampaignStore` at the default `~/.kalos/portal.db`
    (or `$KALOS_STATE_DIR`).

    Tests override this via `app.dependency_overrides[get_campaign_store]`
    and by pinning `_STORE`, pointing at a fresh `CampaignStore(tmp_path)` so
    they never touch the real `~/.kalos/portal.db`.
    """
    global _STORE
    if _STORE is None:
        _STORE = CampaignStore()
    return _STORE
