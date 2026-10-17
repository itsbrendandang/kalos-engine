"""Auth configuration errors, shared by `kalos.portal.auth` and
`kalos.portal.jwt_auth` (kept separate so neither imports the other's
internals)."""
from __future__ import annotations


class AuthConfigError(RuntimeError):
    """The auth configuration is malformed. Surfaced as a 500 (an operator
    misconfiguration), never echoing token material."""


class JwtConfigError(AuthConfigError):
    """The engine-token (JWT) verification config is unusable: the public-key
    file is unreadable or malformed, or a `KALOS_JWT_*` knob is out of range.

    Fails CLOSED: every request answers 503 (and production refuses to start)
    rather than falling back to stale keys or to static tokens only."""
