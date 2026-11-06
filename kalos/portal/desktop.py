"""Desktop mode: the engine as a local process for one person on one Mac.

The desktop app (kalos-web `desktop/`, a Tauri shell) starts the engine with
`KALOS_DESKTOP=1` (or `python -m kalos.portal --desktop`). Desktop mode changes
defaults and adds refusals; it never loosens a server-mode guard. See
docs/DESKTOP.md for what the shell passes.

What it sets, unless the variable is already set:

- `KALOS_TORCH_THREADS`: half the cores, at most 4, at least 1, so a fit leaves
  the rest of the laptop usable.
- `KALOS_JOB_WORKERS=1`: one analysis at a time (sync routes and background
  jobs share this limit, kalos/portal/busy.py).
- `KALOS_LLM_PROVIDER`: `ollama` when something answers on the Ollama port
  (`KALOS_OLLAMA_URL`, default http://127.0.0.1:11434; one TCP connect, 0.2 s
  at most), else `none`. Never a hosted provider by default on a laptop.

What it refuses (exit 2, one line on stderr):

- a bind address other than 127.0.0.1 (`localhost`, `::1` and other loopback
  spellings included: the Host allowlist only names 127.0.0.1 and localhost);
- no token (`KALOS_AUTH_TOKENS` / `KALOS_AUTH_TOKENS_FILE` / JWT keys): the
  desktop engine never runs open, even on loopback;
- `KALOS_ALLOW_OPEN_ACCESS`;
- no `KALOS_CORS_ORIGINS` (or `*`): without it the server-mode default lets any
  localhost origin read responses, so the shell must name the app's own origin.

What it turns off: the interactive API docs and the audit-archive timer
(`KALOS_AUDIT_RETENTION_DAYS` is ignored with a warning; no audit row is
deleted either way).

Startup self-check (exit 3, one plain line on stderr, nothing written): the
state directory must be writable, and an existing `portal.db` must pass SQLite's
`quick_check` and the engine's schema checks (kalos/portal/migrate.py).

The server itself could not start (exit 4, one line naming the cause): the port
is in use, another engine schedules this folder's jobs, or the database upgrade
failed. uvicorn exits 3 for all of these; desktop mode maps them to 4 so the
shell never reads them as a damaged database.

Torch-free at import time on purpose: `__main__` imports this before the app.
"""
from __future__ import annotations

import os
import sqlite3
import threading
import time
from collections.abc import MutableMapping
from pathlib import Path

DESKTOP_ENV = "KALOS_DESKTOP"
# The only bind address desktop mode accepts, and the Host names it answers.
DESKTOP_HOST = "127.0.0.1"
ALLOWED_HOST_HEADERS = ("127.0.0.1", "localhost")
_TRUE = {"1", "true", "yes", "on"}

# Exit codes the shell reads (docs/DESKTOP.md).
EXIT_CONFIG = 2
EXIT_DAMAGED_STATE = 3
EXIT_SERVER_START = 4

# Seconds uvicorn waits for an in-flight request on SIGTERM before it cancels
# it. Below the shell's 15 s SIGKILL deadline, so the engine always gets to
# mark the abandoned work before it is killed.
SHUTDOWN_GRACE_S = 12

_STARTED_AT = time.monotonic()


class DesktopConfigError(RuntimeError):
    """Desktop mode was asked for with a setting it does not allow (exit 2)."""


class DesktopStartupError(RuntimeError):
    """The state directory or its database cannot be used (exit 3). The message
    is plain text for the person at the desk, not a stack trace."""


def is_desktop(environ: MutableMapping[str, str] | None = None) -> bool:
    env = os.environ if environ is None else environ
    return env.get(DESKTOP_ENV, "").strip().lower() in _TRUE


def default_torch_threads(cpu_count: int | None = None) -> int:
    """Half the cores, between 1 and 4."""
    cpus = cpu_count if cpu_count is not None else (os.cpu_count() or 1)
    return max(1, min(4, cpus // 2))


def apply_defaults(environ: MutableMapping[str, str] | None = None) -> dict[str, str]:
    """Set the desktop defaults that are not already set. Returns what was set.
    Must run before `kalos.portal.app` is imported (it reads some at import)."""
    env = os.environ if environ is None else environ
    env[DESKTOP_ENV] = "1"
    applied: dict[str, str] = {}
    for name, value in (
        ("KALOS_TORCH_THREADS", str(default_torch_threads())),
        ("KALOS_JOB_WORKERS", "1"),
    ):
        if not env.get(name, "").strip():
            env[name] = value
            applied[name] = value
    if not env.get("KALOS_LLM_PROVIDER", "").strip():
        provider = "ollama" if ollama_answers(env.get("KALOS_OLLAMA_URL", "")) else "none"
        env["KALOS_LLM_PROVIDER"] = provider
        applied["KALOS_LLM_PROVIDER"] = provider
    return applied


def ollama_answers(url: str = "", timeout: float = 0.2) -> bool:
    """Whether something accepts a TCP connection on the Ollama address. Only
    loopback addresses are probed; any other URL counts as not answering."""
    import socket
    from urllib.parse import urlsplit

    parts = urlsplit(url.strip() or "http://127.0.0.1:11434")
    host, port = parts.hostname or "", parts.port or 11434
    if host not in {"127.0.0.1", "localhost", "::1"}:
        return False
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def state_dir(environ: MutableMapping[str, str] | None = None) -> Path:
    env = os.environ if environ is None else environ
    raw = env.get("KALOS_STATE_DIR", "").strip()
    return Path(raw).expanduser() if raw else Path.home() / ".kalos"


def check_posture(host: str, *, auth_source: bool, environ: MutableMapping[str, str] | None = None) -> None:
    """Refuse a desktop start that would be reachable off the machine or open."""
    env = os.environ if environ is None else environ
    if host.strip() != DESKTOP_HOST:
        raise DesktopConfigError(
            f"desktop mode only listens on this machine (127.0.0.1); KALOS_HOST={host!r} is not allowed"
        )
    if env.get("KALOS_ALLOW_OPEN_ACCESS", "").strip():
        raise DesktopConfigError("desktop mode does not allow KALOS_ALLOW_OPEN_ACCESS")
    if not auth_source:
        raise DesktopConfigError(
            "desktop mode needs a token: set KALOS_AUTH_TOKENS (the desktop app does this)"
        )
    origins = [o.strip() for o in env.get("KALOS_CORS_ORIGINS", "").split(",") if o.strip()]
    if not origins or "*" in origins:
        raise DesktopConfigError(
            "desktop mode needs KALOS_CORS_ORIGINS set to the app's own origin, e.g. "
            "http://127.0.0.1:<web port> (the desktop app does this)"
        )


def self_check(directory: Path) -> str:
    """Refuse to start on a state directory that cannot be used. Returns a
    one-line summary for the log. Writes nothing to the database: a missing
    database is fine (the store creates it), a damaged one is left untouched."""
    from kalos.portal.migrate import DB_FILENAME, SCHEMA_VERSION, status_report

    try:
        directory.mkdir(parents=True, exist_ok=True)
        probe = directory / ".kalos-write-check"
        probe.write_text("ok")
        probe.unlink()
    except OSError as exc:
        raise DesktopStartupError(
            f"the data folder {directory} cannot be written ({exc.strerror or type(exc).__name__}). "
            "Check its permissions or free disk space."
        ) from None
    db = directory / DB_FILENAME
    if not db.exists():
        return f"state dir {directory}: no database yet (a new one is created)"
    _plain = (
        f"the database in {directory} is damaged ({{why}}). Your data was not changed. "
        "Restore a backup, or move the folder aside to start fresh."
    )
    try:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=5)
        try:
            rows = conn.execute("PRAGMA quick_check").fetchall()
        finally:
            conn.close()
    except sqlite3.DatabaseError as exc:
        raise DesktopStartupError(_plain.format(why=f"SQLite cannot read it: {exc}")) from None
    if [r[0] for r in rows] != ["ok"]:
        raise DesktopStartupError(_plain.format(why="SQLite integrity check failed"))
    try:
        report, healthy = status_report(db)
    except (sqlite3.DatabaseError, RuntimeError) as exc:
        raise DesktopStartupError(_plain.format(why=str(exc).splitlines()[0])) from None
    if not healthy:
        reason = next(
            (ln.strip() for ln in report.splitlines()[1:] if "SHAPE CHECK FAILED" not in ln
             and ln.strip() and not ln.strip().startswith(("schema version", "table "))),
            "its tables do not have the shape this engine expects",
        )
        if "newer than this code" in report:
            raise DesktopStartupError(
                f"the database in {directory} was written by a newer Kalos engine "
                f"(this one reads schema v{SCHEMA_VERSION}). Your data was not changed. "
                "Update the app, or use the newer version."
            )
        raise DesktopStartupError(_plain.format(why=reason))
    return f"state dir {directory}: database ok"


# --- server start failures (exit 4) ----------------------------------------- #
# The app's lifespan records why it refused to start (one plain line), because
# uvicorn only logs the exception and exits 3. `__main__` reads it back.
_STARTUP_FAILURE: dict[str, str | None] = {"cause": None}


def note_startup_failure(cause: str) -> None:
    _STARTUP_FAILURE["cause"] = cause


def startup_failure() -> str | None:
    return _STARTUP_FAILURE["cause"]


def port_in_use(host: str, port: int) -> bool:
    import socket

    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return True
    return False


def server_start_failure(host: str, port: int) -> str:
    """The `kalos engine:` line for a server that did not start (exit 4)."""
    cause = startup_failure()
    if cause:
        return f"the engine could not start: {cause}"
    if port_in_use(host, port):
        return (f"the engine could not start: port {port} on {host} is already in use "
                "(another program, or another Kalos engine)")
    return "the engine could not start: the server stopped during startup (see the lines above)"


# --- shutdown --------------------------------------------------------------- #
# Set when the server is shutting down and the grace period is over. Any
# analysis still running then must not commit: `check_not_abandoned` raises
# before a publish, so the store is left exactly as it was.
_ABANDON = threading.Event()


class AnalysisAbandoned(RuntimeError):
    """The engine is shutting down; the in-flight analysis was dropped and
    nothing was saved."""


def abandon_in_flight() -> None:
    _ABANDON.set()


def reset_abandon() -> None:
    """Tests only."""
    _ABANDON.clear()


def check_not_abandoned() -> None:
    if _ABANDON.is_set():
        raise AnalysisAbandoned("the engine is shutting down; this analysis was not saved")


# --- torch prewarm ---------------------------------------------------------- #
_PREWARM: dict[str, float | str | None] = {"started": None, "seconds": None, "error": None}


def prewarm_torch() -> threading.Thread:
    """Import the GP stack on a daemon thread after the port is open, so the
    first upload does not pay for it. Health reports `torch_loaded`."""

    def _run() -> None:
        t0 = time.perf_counter()
        _PREWARM["started"] = t0
        try:
            import kalos.core.evaluation  # noqa: F401 - imports surrogate, torch, gpytorch
            import kalos.core.optimize  # noqa: F401 - botorch acquisition
            from kalos.portal.pipeline import ensure_torch_threads

            ensure_torch_threads()
        except Exception as exc:  # noqa: BLE001 - the first fit will raise it properly
            _PREWARM["error"] = type(exc).__name__
        _PREWARM["seconds"] = round(time.perf_counter() - t0, 2)

    th = threading.Thread(target=_run, name="kalos-torch-prewarm", daemon=True)
    th.start()
    return th


def prewarm_status() -> dict[str, float | str | None]:
    return {"seconds": _PREWARM["seconds"], "error": _PREWARM["error"]}


def uptime_s() -> float:
    return round(time.monotonic() - _STARTED_AT, 1)
