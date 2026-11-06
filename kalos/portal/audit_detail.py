"""The audit-row detail of an upload: which campaign settings (the declared
column roles, `settings.roles`) it set, changed or cleared.

Old -> new values are written only for the small scalar settings whose values
are a number, a boolean or one of a few fixed words; every other setting is
named by its key only. Column names, constraint terms, spec limits, labels and
anything else from the sheet or the roles object never reach the audit log.
"""
from __future__ import annotations

from typing import Any

from kalos.portal.campaign import CampaignStore

# The roles keys an audit row may name (ColumnRoles.from_dict). Any other key
# in a stored roles object is counted, never named.
ROLE_KEYS: tuple[str, ...] = (
    "target", "features", "groups", "ids", "categoricals", "sparse", "direction",
    "applicability_distance", "allow_beyond", "per_recipe_noise", "exploration_design",
    "linear_constraints", "specs",
)
# Scalar settings whose value may be written, and the words allowed for each
# word-valued one (numbers and booleans are always fine for these keys).
_SCALAR_WORDS: dict[str, frozenset[str]] = {
    "direction": frozenset({"maximize", "minimize"}),
    "exploration_design": frozenset({"auto", "off"}),
    "applicability_distance": frozenset(),
    "allow_beyond": frozenset(),
    "per_recipe_noise": frozenset(),
}


def _scalar(key: str, value: Any) -> str | None:
    """`value` as audit text when `key` is a small scalar setting and the value
    is safe to write, else None (then only the key is named)."""
    if key not in _SCALAR_WORDS:
        return None
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return f"{float(value):g}"
    if isinstance(value, str) and value.strip().lower() in _SCALAR_WORDS[key]:
        return value.strip().lower()
    return None


def roles_change_detail(old: Any, new: Any) -> str:
    """One short line naming what an upload did to the campaign's declared
    roles, e.g. "settings: allow_beyond 0.1 -> 0.2; features changed"."""
    old_d = old if isinstance(old, dict) else {}
    new_d = new if isinstance(new, dict) else {}
    if not old_d and not new_d:
        return "settings: no declared roles"
    if old_d and not new_d:
        return "settings: declared roles cleared (roles inferred from the column names)"
    parts: list[str] = []
    for key in ROLE_KEYS:
        if key not in old_d and key not in new_d:
            continue
        before, after = old_d.get(key), new_d.get(key)
        if key in old_d and key in new_d and before == after:
            continue
        b, a = _scalar(key, before), _scalar(key, after)
        if key not in old_d:
            parts.append(f"{key} set to {a}" if a is not None else f"{key} set")
        elif key not in new_d:
            parts.append(f"{key} removed")
        elif a is not None and b is not None:
            parts.append(f"{key} {b} -> {a}")
        else:
            parts.append(f"{key} changed")
    other = {k for k in (*old_d, *new_d) if k not in ROLE_KEYS}
    changed_other = sum(1 for k in other if old_d.get(k) != new_d.get(k) or (k in old_d) != (k in new_d))
    if changed_other:
        parts.append(f"{changed_other} other key(s) changed")
    return "settings: " + ("; ".join(parts) if parts else "declared roles unchanged")


def stored_roles(store: CampaignStore, *, tenant: str, campaign_id: str | None) -> Any:
    """The campaign's `settings.roles` (None when there is no campaign or none
    declared). Best effort: any read failure is None."""
    try:
        if campaign_id is None:
            campaign_id = store.default_campaign_id(tenant=tenant)
            if campaign_id is None:
                return None
        state = store.get(tenant=tenant, campaign_id=campaign_id) or {}
    except Exception:  # noqa: BLE001 - the audit detail must never fail the request
        return None
    settings = state.get("settings")
    return settings.get("roles") if isinstance(settings, dict) else None


__all__ = ["ROLE_KEYS", "roles_change_detail", "stored_roles"]
