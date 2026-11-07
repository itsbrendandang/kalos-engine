"""Auth configuration errors for `kalos.portal.auth`."""
from __future__ import annotations


class AuthConfigError(RuntimeError):
    """The auth configuration is malformed. Surfaced as a 500 (an operator
    misconfiguration), never echoing token material."""
