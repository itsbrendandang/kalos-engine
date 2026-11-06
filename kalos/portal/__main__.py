"""python -m kalos.portal  ->  serve the portal, loopback-only by default.

`--desktop` (or `KALOS_DESKTOP=1`) runs it as the desktop app's local engine:
token required, loopback only, desktop defaults, a startup self-check of the
state directory, and a bounded graceful shutdown (kalos/portal/desktop.py,
docs/DESKTOP.md).

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
import signal
import sys
from typing import IO

import uvicorn

from kalos.envconfig import ConfigError
from kalos.portal import desktop, engine_lock
from kalos.portal.auth import AuthConfigError, get_authenticator
from kalos.portal.config import (
    InsecureBindError,
    ProductionConfigError,
    assert_production_posture,
    assert_safe_exposure,
    check_env_settings,
)
from kalos.portal.job_runner import RunnerLocked, check_startup_config

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8050

# The state-directory lock (kalos/portal/engine_lock.py). Module-level so the
# file object, and with it the flock, lives as long as the server process.
_ENGINE_LOCK: IO[str] | None = None


def _refuse(message: str, code: int) -> None:
    print(f"kalos engine: {message}", file=sys.stderr)
    raise SystemExit(code)


def main(argv: list[str] | None = None) -> None:
    args = [] if argv is None else list(argv)
    unknown = [a for a in args if a != "--desktop"]
    if unknown:
        print(f"kalos portal: unknown argument {unknown[0]!r} (only --desktop is accepted)", file=sys.stderr)
        raise SystemExit(2)
    is_desktop = "--desktop" in args or desktop.is_desktop()
    applied: dict[str, str] = desktop.apply_defaults() if is_desktop else {}
    # The desktop shell shows stderr lines that start with "kalos engine: ".
    prefix = "kalos engine" if is_desktop else "kalos portal"
    host = os.environ.get("KALOS_HOST", DEFAULT_HOST).strip() or DEFAULT_HOST
    raw_port = os.environ.get("KALOS_PORT", "").strip()
    try:
        port = int(raw_port) if raw_port else DEFAULT_PORT
    except ValueError:
        # A typo in a port must not silently fall back to a different one than the
        # operator asked for, since they would then look for the portal in the
        # wrong place.
        print(f"{prefix}: KALOS_PORT={raw_port!r} is not an integer", file=sys.stderr)
        raise SystemExit(2) from None

    try:
        auth = get_authenticator()
        auth_enforced = auth.enforces()
        # KALOS_ENV=production: tokenless open access is refused on ANY bind,
        # loopback included, and KALOS_ALLOW_OPEN_ACCESS does not override it.
        # It needs a usable source: a JWT public key or a service-account token.
        assert_production_posture(auth_enforced=auth.has_auth_source())
        assert_safe_exposure(host, auth_configured=auth_enforced)
        if is_desktop:
            desktop.check_posture(host, auth_source=auth.has_auth_source())
    except desktop.DesktopConfigError as exc:
        _refuse(str(exc), desktop.EXIT_CONFIG)
    except AuthConfigError as exc:
        # A malformed token list or an unreadable/malformed JWT key file: fail
        # closed at startup. Messages never carry token or key material.
        print(f"{prefix}: authentication configuration is invalid: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
    except (InsecureBindError, ProductionConfigError) as exc:
        # Exit non-zero with the reason on stderr: this is a misconfiguration an
        # operator has to fix, not a crash, so it should read like a refusal.
        print(f"{prefix}: {exc}", file=sys.stderr)
        raise SystemExit(2) from None

    try:
        # Every enum-valued KALOS_* setting (KALOS_ENV, KALOS_VALIDATION_MODE,
        # KALOS_LLM_PROVIDER) and bounded one (KALOS_AUDIT_RETENTION_DAYS) is
        # checked before the socket opens: a typo refuses to start, one line, exit 2.
        check_env_settings()
        # Background-job settings that cannot run (kalos/portal/job_runner.py):
        # refuse to serve rather than come up with jobs accepted and never run.
        check_startup_config()
    except (ConfigError, ValueError, RunnerLocked) as exc:
        print(f"{prefix}: {exc}", file=sys.stderr)
        raise SystemExit(2) from None

    if is_desktop:
        # Refuse a damaged or unwritable state dir with one plain line before the
        # port opens (exit 3), so the desktop shell can show it as it is.
        try:
            summary = desktop.self_check(desktop.state_dir())
        except desktop.DesktopStartupError as exc:
            _refuse(str(exc), desktop.EXIT_DAMAGED_STATE)
        print(
            f"kalos engine: desktop mode on 127.0.0.1:{port}; {summary}; "
            f"defaults set: {', '.join(f'{k}={v}' for k, v in applied.items()) or 'none'}",
            file=sys.stderr,
        )

    # One engine per state directory: a second one refuses to start rather than
    # share portal.db and the latest-analysis cache with the first.
    global _ENGINE_LOCK
    try:
        _ENGINE_LOCK = engine_lock.acquire_engine_lock()
    except engine_lock.EngineLockError as exc:
        # Held by another engine, or not lockable on this filesystem: either way
        # this engine must not start on the directory.
        print(f"{prefix}: {exc}", file=sys.stderr)
        raise SystemExit(2) from None

    # uvicorn trusts X-Forwarded-For from loopback by default, which would make a
    # same-host reverse proxy turn every remote caller into a "loopback" peer and defeat
    # the open-mode remote guard. Ignore forwarded headers unless the operator names the
    # trusted proxy addresses explicitly.
    trusted = os.environ.get("KALOS_TRUSTED_PROXIES", "").strip()
    if is_desktop:
        # uvicorn restores the previous SIGTERM handler after its graceful shutdown
        # and raises the signal again; this handler is that previous one, so the
        # process ends with exit 0 and a last log line instead of dying by signal.
        signal.signal(signal.SIGTERM, _desktop_exit)
    uvicorn.run(
        "kalos.portal.app:app", host=host, port=port, log_level="warning",
        proxy_headers=bool(trusted), forwarded_allow_ips=trusted or None,
        # SIGTERM: stop accepting, give an in-flight analysis this long to finish,
        # then cancel it; the lifespan exit marks it abandoned so it commits nothing.
        timeout_graceful_shutdown=desktop.SHUTDOWN_GRACE_S if is_desktop else None,
    )
    if is_desktop:
        _desktop_exit()


def _desktop_exit(*_signal_args: object) -> None:
    """End a desktop engine after uvicorn's shutdown. A cancelled analysis keeps
    computing on its worker thread (threads cannot be stopped) and would hold the
    interpreter open past the shell's SIGKILL deadline. It can no longer commit
    (desktop.check_not_abandoned), so exit now."""
    print("kalos engine: stopped", file=sys.stderr)
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main(sys.argv[1:])
