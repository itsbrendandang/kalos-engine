"""TODO(E2/E4): cross-tenant release gate for JWT-authenticated tenants.

Placeholder ONLY. The E2 JWT verifier (`kalos/portal/jwt_auth.py`, branch
`feat/engine-jwt-verification`) is not on this branch's base
(`feat/campaign-schema-v2`), so every test here is skipped. The static-token
suite in tests/test_cross_tenant_isolation.py is the gate until then.

When E2 lands, port each test of tests/test_cross_tenant_isolation.py to two
JWT tenants (tokens minted by a test signing key), and add the JWT-specific
cases:

- tenant comes ONLY from the verified claim; a forged/unsigned token, a token
  for another audience/issuer, or an expired token never reaches a route (401);
- a JWT tenant vs a static-token tenant: foreign ids are the same 404 bytes;
- `audit_log.auth_method` reads "jwt" (Principal.auth_method) for JWT callers
  and "bearer" for static tokens;
- JWT scope claims map to read/write/admin exactly like static scopes
  (read cannot POST, write cannot archive, admin can).
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(
    reason="TODO(E2): JWT verifier not on this base; see module docstring"
)


def test_jwt_tenant_foreign_campaign_is_identical_404() -> None:  # pragma: no cover
    raise NotImplementedError


def test_jwt_tenant_list_shows_only_own_campaigns() -> None:  # pragma: no cover
    raise NotImplementedError


def test_jwt_scopes_map_to_read_write_admin() -> None:  # pragma: no cover
    raise NotImplementedError


def test_jwt_audit_rows_record_auth_method_jwt() -> None:  # pragma: no cover
    raise NotImplementedError


def test_forged_or_expired_jwt_never_reaches_a_campaign_route() -> None:  # pragma: no cover
    raise NotImplementedError
