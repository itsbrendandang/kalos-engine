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
WEB_ORIGIN = "http://127.0.0.1:3360"


def _tokens(secret: str = "desk-secret") -> str:
    return json.dumps([
        {"token_sha256": hashlib.sha256(secret.encode()).hexdigest(),
         "subject": "desktop-app", "tenant": "local", "scopes": ["read", "write"]},
        {"token_sha256": hashlib.sha256(b"admin-secret").hexdigest(),
         "subject": "operator", "tenant": "ops", "scopes": ["read", "admin"]},
    ])


def _env(tmp_path: Path, **extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("KALOS_")}
    env.update({"KALOS_STATE_DIR": str(tmp_path / "state"), "KALOS_JOBS_ENABLED": "0",
                "KALOS_CORS_ORIGINS": WEB_ORIGIN, **extra})
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


def test_llm_provider_default_follows_a_local_ollama(monkeypatch):
    import socket

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    try:
        env = {"KALOS_OLLAMA_URL": f"http://127.0.0.1:{port}"}
        assert desktop.apply_defaults(env)["KALOS_LLM_PROVIDER"] == "ollama"
    finally:
        listener.close()
    env = {"KALOS_OLLAMA_URL": f"http://127.0.0.1:{port}"}
    assert desktop.apply_defaults(env)["KALOS_LLM_PROVIDER"] == "none"
    env = {"KALOS_LLM_PROVIDER": "anthropic"}
    assert "KALOS_LLM_PROVIDER" not in desktop.apply_defaults(env)
    assert not desktop.ollama_answers("http://example.com:11434")


def test_check_posture_refusals():
    ok = {"KALOS_CORS_ORIGINS": WEB_ORIGIN}
    with pytest.raises(desktop.DesktopConfigError, match="only listens"):
        desktop.check_posture("0.0.0.0", auth_source=True, environ=ok)  # noqa: S104 - refused
    with pytest.raises(desktop.DesktopConfigError, match="needs a token"):
        desktop.check_posture("127.0.0.1", auth_source=False, environ=ok)
    with pytest.raises(desktop.DesktopConfigError, match="KALOS_ALLOW_OPEN_ACCESS"):
        desktop.check_posture("127.0.0.1", auth_source=True, environ={**ok, "KALOS_ALLOW_OPEN_ACCESS": "1"})
    for cors in ({}, {"KALOS_CORS_ORIGINS": " "}, {"KALOS_CORS_ORIGINS": "*"}):
        with pytest.raises(desktop.DesktopConfigError, match="KALOS_CORS_ORIGINS"):
            desktop.check_posture("127.0.0.1", auth_source=True, environ=cors)
    desktop.check_posture("127.0.0.1", auth_source=True, environ=ok)


def test_desktop_without_cors_origins_refuses_with_exit_2(tmp_path):
    out = _run_main(tmp_path, "--desktop", KALOS_AUTH_TOKENS=_tokens(), KALOS_CORS_ORIGINS="")
    assert out.returncode == 2, out.stderr
    assert out.stderr.strip().splitlines()[-1].startswith(
        "kalos engine: desktop mode needs KALOS_CORS_ORIGINS")


def test_cors_in_desktop_mode_allows_only_the_app_origin(tmp_path):
    """The server-mode default (any localhost origin) is not used: only the
    origin the shell named gets CORS headers back."""
    code = (
        "import json\n"
        "from fastapi.testclient import TestClient\n"
        "c = TestClient(m.app, base_url='http://127.0.0.1:8050')\n"
        "out = {}\n"
        f"for o in ({WEB_ORIGIN!r}, 'http://localhost:5173', 'http://127.0.0.1:9999', 'https://evil.example'):\n"
        "    r = c.options('/api/health', headers={'Origin': o, 'Access-Control-Request-Method': 'GET',"
        " 'Access-Control-Request-Headers': 'authorization'})\n"
        "    out[o] = r.headers.get('access-control-allow-origin')\n"
        "print(json.dumps(out))\n"
    )
    out = _import_app(tmp_path, code, KALOS_DESKTOP="1", KALOS_AUTH_TOKENS=_tokens())
    assert out.returncode == 0, out.stderr
    got = json.loads(out.stdout.strip().splitlines()[-1])
    assert got == {WEB_ORIGIN: WEB_ORIGIN, "http://localhost:5173": None,
                   "http://127.0.0.1:9999": None, "https://evil.example": None}


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


def _import_app(tmp_path: Path, code: str = "", **env: str) -> subprocess.CompletedProcess[str]:
    """Import kalos.portal.app the way `uvicorn kalos.portal.app:app` does (no
    `__main__`), then run `code`."""
    return subprocess.run(
        [sys.executable, "-c", "import kalos.portal.app as m\n" + code], cwd=ROOT,
        env=_env(tmp_path, **env), capture_output=True, text=True, timeout=120,
    )


@pytest.mark.parametrize("env, needle", [
    ({}, "needs a token"),
    ({"KALOS_ALLOW_OPEN_ACCESS": "1"}, "KALOS_ALLOW_OPEN_ACCESS"),
    ({"KALOS_AUTH_TOKENS": _tokens(), "KALOS_HOST": "0.0.0.0"}, "only listens"),  # noqa: S104
])
def test_uvicorn_direct_import_runs_the_desktop_posture_check(tmp_path, env, needle):
    """`KALOS_DESKTOP=1 uvicorn kalos.portal.app:app` skips __main__; the import
    itself must refuse an open or exposed desktop engine."""
    out = _import_app(tmp_path, KALOS_DESKTOP="1", **env)
    assert out.returncode != 0
    assert "DesktopConfigError" in out.stderr and needle in out.stderr


def test_uvicorn_direct_import_with_a_token_loads(tmp_path):
    out = _import_app(tmp_path, "print(m._DESKTOP)", KALOS_DESKTOP="1",
                      KALOS_AUTH_TOKENS=_tokens())
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == "True"


@pytest.mark.parametrize("host", ["::1", "localhost", "127.0.0.2", "0.0.0.0"])  # noqa: S104
def test_desktop_binds_only_127_0_0_1(host):
    with pytest.raises(desktop.DesktopConfigError, match="only listens"):
        desktop.check_posture(host, auth_source=True, environ={"KALOS_CORS_ORIGINS": WEB_ORIGIN})


def test_desktop_refuses_kalos_host_ipv6_loopback_with_exit_2(tmp_path):
    out = _run_main(tmp_path, "--desktop", KALOS_AUTH_TOKENS=_tokens(), KALOS_HOST="::1")
    assert out.returncode == 2, out.stderr
    assert "KALOS_HOST='::1' is not allowed" in out.stderr.strip().splitlines()[-1]


def test_desktop_host_allowlist_and_no_api_docs(tmp_path):
    """DNS-rebinding defence: only Host 127.0.0.1 / localhost (any port) reach the
    app; a foreign, absent, uppercase, IPv6 or suffix-trick Host is 400 before
    the token check. /docs, /redoc and /openapi.json do not exist."""
    code = (
        "import json\n"
        "from fastapi.testclient import TestClient\n"
        "c = TestClient(m.app, base_url='http://127.0.0.1:8050')\n"
        "auth = {'Authorization': 'Bearer desk-secret'}\n"
        "hosts = ['127.0.0.1:8050', 'localhost:8050', '127.0.0.1', '127.0.0.1:1',\n"
        "         'evil.example', 'evil.example:8050', 'LOCALHOST:8050', 'Localhost',\n"
        "         '[::1]:8050', '127.0.0.1.evil.example', 'localhost.evil.example', '']\n"
        "out = {'hosts': {h: c.get('/api/health', headers={**auth, 'Host': h}).status_code for h in hosts}}\n"
        "import asyncio\n"
        "sent = []\n"
        "async def recv():\n"
        "    return {'type': 'http.request', 'body': b'', 'more_body': False}\n"
        "async def send(msg):\n"
        "    sent.append(msg)\n"
        "scope = {'type': 'http', 'asgi': {'version': '3.0'}, 'http_version': '1.1', 'method': 'GET',\n"
        "         'scheme': 'http', 'path': '/api/health', 'raw_path': b'/api/health', 'query_string': b'',\n"
        "         'root_path': '', 'headers': [(b'authorization', b'Bearer desk-secret')],\n"
        "         'client': ('127.0.0.1', 5000), 'server': ('127.0.0.1', 8050)}\n"
        "asyncio.run(m.app(scope, recv, send))\n"
        "out['absent'] = sent[0]['status']\n"
        "out['docs'] = {p: c.get(p, headers=auth).status_code for p in ('/docs', '/redoc', '/openapi.json')}\n"
        "print(json.dumps(out))\n"
    )
    out = _import_app(tmp_path, code, KALOS_DESKTOP="1", KALOS_AUTH_TOKENS=_tokens())
    assert out.returncode == 0, out.stderr
    got = json.loads(out.stdout.strip().splitlines()[-1])
    ok = {"127.0.0.1:8050", "localhost:8050", "127.0.0.1", "127.0.0.1:1"}
    assert {h for h, code in got["hosts"].items() if code == 200} == ok, got
    assert all(code == 400 for h, code in got["hosts"].items() if h not in ok), got
    assert got["absent"] == 400
    assert got["docs"] == {"/docs": 404, "/redoc": 404, "/openapi.json": 404}


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


def test_port_in_use_exits_4_not_3(tmp_path):
    """uvicorn exits 3 when the port is taken; the shell reads 3 as a damaged
    database. Desktop mode answers 4 with a line naming the port."""
    import socket

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    try:
        out = _run_main(tmp_path, "--desktop", KALOS_AUTH_TOKENS=_tokens(), KALOS_PORT=str(port))
    finally:
        listener.close()
    assert out.returncode == desktop.EXIT_SERVER_START == 4, out.stderr
    # The startup line names the address actually bound (KALOS_HOST).
    assert f"kalos engine: desktop mode on 127.0.0.1:{port}; " in out.stderr
    line = out.stderr.strip().splitlines()[-1]
    assert line == (f"kalos engine: the engine could not start: port {port} on 127.0.0.1 is "
                    "already in use (another program, or another Kalos engine)")


def test_runner_locked_in_the_lifespan_exits_4_naming_it(tmp_path):
    """Another process holds this folder's job-runner lock: the lifespan refuses
    (RunnerLocked) and the desktop engine exits 4, not uvicorn's 3."""
    import fcntl
    import socket

    from kalos.portal.job_runner import RUNNER_LOCK_NAME

    state = tmp_path / "state"
    state.mkdir()
    fd = os.open(state / RUNNER_LOCK_NAME, os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    try:
        out = _run_main(tmp_path, "--desktop", KALOS_AUTH_TOKENS=_tokens(), KALOS_PORT=str(port),
                        KALOS_JOBS_ENABLED="1")
    finally:
        os.close(fd)
    assert out.returncode == 4, out.stderr
    line = out.stderr.strip().splitlines()[-1]
    assert line.startswith("kalos engine: the engine could not start: another engine runs the jobs")


def test_lifespan_records_a_failed_migration_for_exit_4(tmp_path):
    from kalos.portal.migrate import MigrationError

    def _fails():
        raise MigrationError("backup verification failed (x)")

    app.dependency_overrides[get_campaign_store] = _fails
    try:
        with pytest.raises(MigrationError), TestClient(app):
            pass
        assert "database upgrade could not run safely" in (desktop.startup_failure() or "")
        assert "the database upgrade" in desktop.server_start_failure("127.0.0.1", 1)
    finally:
        app.dependency_overrides.pop(get_campaign_store, None)
        desktop.note_startup_failure("")


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
