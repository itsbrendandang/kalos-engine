"""Tenant id validation, shared by auth (config load) and every per-tenant
store. Dependency-free on purpose so storage modules can import it without
pulling in FastAPI."""
from __future__ import annotations

import re

# A tenant id is used as a storage key (SQLite rows, the per-tenant latest cache
# file) and in logs, so it is restricted to a conservative alphabet. A config
# entry that does not match is REJECTED, never sanitised: sanitising would map
# distinct ids (`acme/bio`, `acme_bio`) onto one key and silently merge two
# tenants' data.
TENANT_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,64}")

# The tenant open (anonymous) mode writes to. No credential - static token or
# engine token - may name it: it would inherit whatever anonymous callers stored
# there before auth was turned on.
RESERVED_ANON_TENANT = "default"


def is_valid_tenant_id(tenant: object) -> bool:
    """Whether `tenant` is a well-formed tenant id (`^[A-Za-z0-9_-]{1,64}$`)."""
    return isinstance(tenant, str) and TENANT_ID_PATTERN.fullmatch(tenant) is not None


def validate_tenant_id(tenant: object) -> str:
    """Return `tenant` unchanged if well-formed, else raise `ValueError`.
    Never rewrites the id."""
    if not isinstance(tenant, str) or TENANT_ID_PATTERN.fullmatch(tenant) is None:
        raise ValueError("invalid tenant id: must match ^[A-Za-z0-9_-]{1,64}$")
    return tenant
