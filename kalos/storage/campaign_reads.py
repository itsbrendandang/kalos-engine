"""Read-only, batched SQL over the campaign tables for the records views.

One query for the names and archive state of many campaigns, and one for their
stored stage documents, instead of two store reads per campaign. Every
statement carries ``tenant = ?``: another tenant's id simply has no row. These
tables live in `portal.db` (moving behind `storage/campaign_repo.py` in
REORG_PLAN R12, when these queries move next to it).
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from kalos.storage.backend import Conn, json_value

CHUNK = 400  # ids per IN (...) list, far below SQLite's and PostgreSQL's limits


def _chunks(ids: list[str]) -> Iterable[list[str]]:
    for i in range(0, len(ids), CHUNK):
        yield ids[i : i + CHUNK]


def campaign_cards(conn: Conn, *, tenant: str, campaign_ids: Iterable[str]) -> dict[str, dict[str, Any]]:
    """`{campaign_id: {"name", "archived_at", "stages"}}` for the caller's
    campaigns among `campaign_ids` (others are absent). `stages` is the stored
    stage document (parsed JSON) or None when the campaign never set one."""
    ids = list(dict.fromkeys(campaign_ids))
    out: dict[str, dict[str, Any]] = {}
    for chunk in _chunks(ids):
        marks = ", ".join("?" for _ in chunk)
        for cid, name, archived_at in conn.execute(
            f"SELECT id, name, archived_at FROM campaigns WHERE tenant = ? AND id IN ({marks})",  # noqa: S608 - placeholders only
            (tenant, *chunk),
        ).fetchall():
            out[str(cid)] = {"name": name, "archived_at": archived_at, "stages": None}
        for cid, value in conn.execute(
            "SELECT campaign_id, value FROM campaign_settings WHERE tenant = ? AND kind = 'stages' "  # noqa: S608 - placeholders only
            f"AND campaign_id IN ({marks})",
            (tenant, *chunk),
        ).fetchall():
            if str(cid) in out:
                out[str(cid)]["stages"] = json_value(value)
    return out


__all__ = ["campaign_cards"]
