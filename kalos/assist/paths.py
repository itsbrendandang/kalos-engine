"""Where the assistant keeps its own files: `<state dir>/assist/<tenant key>/`.

The tenant key is the lowercase hex of the tenant id's UTF-8 bytes (the same
injective encoding the audit archive uses), so two tenants never share a
folder and the folder name can never traverse a path. Nothing here is
campaign state: the index and SOP drafts can be deleted and rebuilt from the
campaign record (drafts are the lab's own text and are kept append-only).
"""
from __future__ import annotations

from pathlib import Path

from kalos.portal.tenancy import validate_tenant_id


def tenant_key(tenant: str) -> str:
    validate_tenant_id(tenant)
    return "t-" + tenant.encode("utf-8").hex()


def tenant_dir(state_dir: Path, tenant: str, *, create: bool = True) -> Path:
    """`<state_dir>/assist/<tenant key>/`, created on first use."""
    path = Path(state_dir) / "assist" / tenant_key(tenant)
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


__all__ = ["tenant_dir", "tenant_key"]
