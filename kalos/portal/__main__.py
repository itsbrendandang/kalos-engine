"""python -m kalos.portal  ->  serve the portal, loopback-only by default.

Host and port come from `KALOS_HOST` / `KALOS_PORT` and default to
`127.0.0.1:8050`. They were previously hardcoded, which made this entrypoint
accidentally safe: nobody could expose the portal without editing the source.
Making them configurable is what a real deployment needs, so the bind is now
checked instead of merely being impossible.

Serving unauthenticated on a non-loopback interface fails fast with an actionable
message rather than starting and hoping the operator reads a warning. See
`kalos.portal.config.assert_safe_bind`. The peer-address guard in
`kalos.portal.app` covers the case where uvicorn is launched directly and this
check never runs at all.
"""
from __future__ import annotations

import os
import sys
from typing import IO, NoReturn

import uvicorn

from kalos.portal import engine_lock
from kalos.portal.auth import AuthConfigError, get_authenticator
from kalos.portal.config import (
    InsecureBindError,
    ProductionConfigError,
    assert_production_posture,
    assert_safe_exposure,
)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8050

# The state-directory lock (kalos/portal/engine_lock.py). Module-level so the
# file object, and with it the flock, lives as long as the server process.
_ENGINE_LOCK: IO[str] | None = None


def refuse(message: str) -> NoReturn:
    """Exit 2 with one line on stderr. A misconfiguration an operator has to fix
    is a refusal, not a crash: no traceback, and no setting value beyond what the
    message itself names."""
    print(f"kalos portal: {message}", file=sys.stderr)
    raise SystemExit(2)


def preflight(host: str) -> None:
    """Run every startup-time configuration check, and turn each failure into a
    refusal (`refuse`).

    Importing `kalos.portal.app` runs the import-time reads: the sizing knobs,
    `KALOS_ENV`, the token configuration and the production posture. Doing that
    here, inside the try, is what keeps a bad setting from surfacing as a
    traceback out of uvicorn's own import of the app. The remaining reads
    (job limits, request cap, free-space floor) are called explicitly so they are
    validated before the server starts rather than on the first request.
    """
    try:
        import kalos.portal.app  # noqa: F401 - import-time config checks run here
        from kalos.portal.config import ready_min_free_bytes
        from kalos.portal.jobs import job_settings
        from kalos.portal.limits import max_request_bytes

        job_settings()
        max_request_bytes()
        ready_min_free_bytes()
        auth_enforced = get_authenticator().enforces()
        # KALOS_ENV=production: tokenless open access is refused on ANY bind,
        # loopback included, and KALOS_ALLOW_OPEN_ACCESS does not override it.
        assert_production_posture(auth_enforced=auth_enforced)
        assert_safe_exposure(host, auth_configured=auth_enforced)
    except (ValueError, AuthConfigError, InsecureBindError, ProductionConfigError) as exc:
        # ConfigError and the job/busy parsers are ValueErrors that already name
        # the variable; the auth and production errors carry their own reason.
        refuse(str(exc))


def main() -> None:
    host = os.environ.get("KALOS_HOST", DEFAULT_HOST).strip() or DEFAULT_HOST
    raw_port = os.environ.get("KALOS_PORT", "").strip()
    try:
        port = int(raw_port) if raw_port else DEFAULT_PORT
    except ValueError:
        # A typo in a port must not silently fall back to a different one than the
        # operator asked for, since they would then look for the portal in the
        # wrong place.
        refuse(f"KALOS_PORT={raw_port[:40]!r} is not an integer")

    preflight(host)

    # One engine per state directory: a second one refuses to start rather than
    # share portal.db and the latest-analysis cache with the first.
    global _ENGINE_LOCK
    try:
        _ENGINE_LOCK = engine_lock.acquire_engine_lock()
    except engine_lock.EngineLockHeld as exc:
        print(f"kalos portal: {exc}", file=sys.stderr)
        raise SystemExit(2) from None

    uvicorn.run("kalos.portal.app:app", host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
