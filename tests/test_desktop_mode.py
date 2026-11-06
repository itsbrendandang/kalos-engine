"""Desktop mode (kalos/portal/desktop.py, docs/DESKTOP.md): defaults, refusals,
the startup self-check, the shutdown guard, GET /api/health, and a torch-free
portal import. Every test uses a temp state dir; ~/.kalos is never touched."""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from kalos.portal import desktop
from kalos.portal.app import app, get_campaign_store
from kalos.portal.campaign import CampaignStore

ROOT = Path(__file__).resolve().parents[1]


def _tokens(secret: str = "desk-secret") -> str:
    return json.dumps([
        {"token_sha256": hashlib.sha256(secret.encode()).hexdigest(),
         "subject": "desktop-app", "tenant": "local", "scopes": ["read", "write"]},
        {"token_sha256": hashlib.sha256(b"admin-secret").hexdigest(),
         "subject": "operator", "tenant": "ops", "scopes": ["read", "admin"]},
    ])


def _env(tmp_path: Path, **extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("KALOS_")}
    env.update({"KALOS_STATE_DIR": str(tmp_path / "state"), "KALOS_JOBS_ENABLED": "0", **extra})
    return env


def _run_main(tmp_path: Path, *args: str, **env: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "kalos.portal", *args], cwd=ROOT, env=_env(tmp_path, **env),
        capture_output=True, text=True, timeout=120,
    )


# --- startup cost ------------------------------------------------------------ #


def test_portal_import_does_not_load_torch(tmp_path):
    """The GP stack is imported on first use (or by the desktop prewarm), never
    by importing the app: it is most of the cold-start time."""
    code = (
        "import sys, kalos.portal.app; "
        "print(sorted(m for m in ('torch', 'botorch', 'gpytorch') if m in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=_env(tmp_path),
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == "[]"


def test_scale_route_decision_labels_match_readout():
    from kalos.portal import scale_routes
    from kalos.scale import readout

    assert (scale_routes.NUMBER, scale_routes.NUMBER_WITH_WARNING, scale_routes.REFUSAL) == (
        readout.NUMBER, readout.NUMBER_WITH_WARNING, readout.REFUSAL)


# --- defaults and refusals ---------------------------------------------------- #


@pytest.mark.parametrize("cpus, threads", [(1, 1), (2, 1), (8, 4), (11, 4), (16, 4), (6, 3)])
def test_default_torch_threads_is_half_the_cores_capped(cpus, threads):
    assert desktop.default_torch_threads(cpus) == threads


def test_apply_defaults_never_overrides_an_explicit_setting():
    env = {"KALOS_TORCH_THREADS": "2"}
    applied = desktop.apply_defaults(env)
    assert env["KALOS_TORCH_THREADS"] == "2" and "KALOS_TORCH_THREADS" not in applied
    assert env["KALOS_JOB_WORKERS"] == "1" and env[desktop.DESKTOP_ENV] == "1"


def test_check_posture_refusals():
    with pytest.raises(desktop.DesktopConfigError, match="only listens"):
        desktop.check_posture("0.0.0.0", auth_source=True, environ={})  # noqa: S104 - refused
    with pytest.raises(desktop.DesktopConfigError, match="needs a token"):
        desktop.check_posture("127.0.0.1", auth_source=False, environ={})
    with pytest.raises(desktop.DesktopConfigError, match="KALOS_ALLOW_OPEN_ACCESS"):
        desktop.check_posture("127.0.0.1", auth_source=True, environ={"KALOS_ALLOW_OPEN_ACCESS": "1"})
    desktop.check_posture("127.0.0.1", auth_source=True, environ={})


def test_desktop_without_token_refuses_with_exit_2(tmp_path):
    out = _run_main(tmp_path, "--desktop")
    assert out.returncode == 2
    assert out.stderr.strip().splitlines()[-1].startswith("kalos engine: desktop mode needs a token")


def test_env_switch_is_the_same_as_the_flag(tmp_path):
    out = _run_main(tmp_path, KALOS_DESKTOP="1", KALOS_HOST="0.0.0.0")  # noqa: S104 - refused
    assert out.returncode == 2
    assert "kalos engine:" in out.stderr


def test_unknown_argument_is_refused(tmp_path):
    out = _run_main(tmp_path, "--nope")
    assert out.returncode == 2 and "unknown argument" in out.stderr


# --- startup self-check ------------------------------------------------------- #


def test_self_check_accepts_a_fresh_and_a_healthy_dir(tmp_path):
    assert "no database yet" in desktop.self_check(tmp_path / "new")
    CampaignStore(tmp_path / "ok")  # creates portal.db at the current schema
    assert "database ok" in desktop.self_check(tmp_path / "ok")


def test_self_check_refuses_a_damaged_file(tmp_path):
    (tmp_path / "portal.db").write_bytes(b"this is not a sqlite database" * 100)
    with pytest.raises(desktop.DesktopStartupError, match=r"is damaged.*Your data was not changed"):
        desktop.self_check(tmp_path)


def test_self_check_refuses_a_newer_schema(tmp_path):
    CampaignStore(tmp_path)
    conn = sqlite3.connect(tmp_path / "portal.db")
    conn.execute("INSERT INTO schema_migrations (version, name, applied_at) VALUES (99, 'future', 0)")
    conn.commit()
    conn.close()
    with pytest.raises(desktop.DesktopStartupError, match="newer Kalos engine"):
        desktop.self_check(tmp_path)


def test_self_check_refuses_a_wrong_shape(tmp_path):
    CampaignStore(tmp_path)
    conn = sqlite3.connect(tmp_path / "portal.db")
    conn.execute("DROP TABLE campaigns")
    conn.commit()
    conn.close()
    with pytest.raises(desktop.DesktopStartupError, match="is damaged"):
        desktop.self_check(tmp_path)


def test_damaged_state_dir_exits_3_before_binding(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    (state / "portal.db").write_bytes(b"garbage" * 500)
    before = (state / "portal.db").read_bytes()
    out = _run_main(tmp_path, "--desktop", KALOS_AUTH_TOKENS=_tokens(), KALOS_PORT="1")
    assert out.returncode == 3, out.stderr
    line = out.stderr.strip().splitlines()[-1]
    assert line.startswith("kalos engine: the database in ") and "damaged" in line
    assert (state / "portal.db").read_bytes() == before  # nothing written


# --- shutdown guard ------------------------------------------------------------ #


def test_abandoned_analysis_cannot_commit(tmp_path):
    store = CampaignStore(tmp_path)
    desktop.abandon_in_flight()
    try:
        with pytest.raises(desktop.AnalysisAbandoned):
            store.save_analysis({"n": 1}, "x.csv", tenant="local")
        assert store.load_analysis(tenant="local") is None
    finally:
        desktop.reset_abandon()
    store.save_analysis({"n": 1}, "x.csv", tenant="local")
    assert store.load_analysis(tenant="local") is not None


# --- GET /api/health ----------------------------------------------------------- #


@pytest.fixture
def client(tmp_path, monkeypatch):
    for var in ("KALOS_AUTH_TOKENS_FILE", "KALOS_ALLOW_OPEN_ACCESS"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("KALOS_AUTH_TOKENS", _tokens())
    store = CampaignStore(tmp_path)
    app.dependency_overrides[get_campaign_store] = lambda: store
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(get_campaign_store, None)


def test_health_needs_the_token(client):
    assert client.get("/api/health").status_code == 401


@pytest.fixture
def desktop_app(monkeypatch):
    from kalos.portal import app as app_module

    monkeypatch.setattr(app_module, "_DESKTOP", True)


def test_health_hides_host_facts_from_a_tenant_token_in_server_mode(client):
    body = client.get("/api/health", headers={"Authorization": "Bearer desk-secret"}).json()
    assert body["status"] == "ok" and body["mode"] == "server"
    assert not {"state_dir", "pid", "analysis_in_flight"} & set(body)
    admin = client.get("/api/health", headers={"Authorization": "Bearer admin-secret"}).json()
    assert {"state_dir", "pid", "analysis_in_flight"} <= set(admin)


def test_health_reports_versions_state_and_schema(client, tmp_path, desktop_app):
    from kalos.portal.migrate import SCHEMA_VERSION

    r = client.get("/api/health", headers={"Authorization": "Bearer desk-secret"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["state_dir"] == str(tmp_path)
    assert body["schema_version"] == body["schema_expected"] == SCHEMA_VERSION
    assert body["contract_version"] >= 2
    assert isinstance(body["torch_version"], str) and isinstance(body["torch_loaded"], bool)
    assert body["analysis_in_flight"] == 0 and body["pid"] == os.getpid()
    assert body["mode"] == "desktop"
    assert set(body) >= {"engine_version", "python_version", "botorch_version", "torch_threads",
                         "uptime_s"}


def test_health_is_503_with_a_plain_reason_when_the_store_fails(client):
    def _broken():
        raise sqlite3.DatabaseError("database disk image is malformed /secret/path")

    app.dependency_overrides[get_campaign_store] = _broken
    r = client.get("/api/health", headers={"Authorization": "Bearer desk-secret"})
    assert r.status_code == 503
    assert r.json()["code"] == "store_unavailable"
    assert "/secret/path" not in r.text


def test_sigterm_stops_a_desktop_engine_with_exit_0(tmp_path):
    """The desktop shell sends SIGTERM first. The engine answers health, stops,
    and exits 0 with a last plain line (not killed by the re-raised signal)."""
    import signal
    import socket
    import time
    import urllib.request

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    env = _env(tmp_path, KALOS_DESKTOP="1", KALOS_PORT=str(port), KALOS_AUTH_TOKENS=_tokens())
    proc = subprocess.Popen([sys.executable, "-m", "kalos.portal"], cwd=ROOT, env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 90
        while True:
            assert proc.poll() is None, proc.stderr.read() if proc.stderr else ""
            try:
                req = urllib.request.Request(f"http://127.0.0.1:{port}/api/health",
                                             headers={"Authorization": "Bearer desk-secret"})
                with urllib.request.urlopen(req, timeout=2) as r:  # noqa: S310 - loopback test URL
                    assert json.loads(r.read())["mode"] == "desktop"
                break
            except OSError:
                assert time.monotonic() < deadline
                time.sleep(0.1)
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=30) == 0
        assert proc.stderr is not None
        assert proc.stderr.read().strip().splitlines()[-1] == "kalos engine: stopped"
    finally:
        if proc.poll() is None:
            proc.kill()
