"""Bad configuration is a one-line refusal (exit 2), never a traceback (D11/D13).

Each case runs `python -m kalos.portal` in a subprocess with a temp state dir.
Every case must exit before uvicorn starts, so no port is bound.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys

import pytest

pytest.importorskip("fastapi")

_TOKEN = json.dumps([{
    "token_sha256": hashlib.sha256(b"tok").hexdigest(),
    "subject": "s", "tenant": "acme", "scopes": ["read"],
}])


def _run(tmp_path, **env: str) -> subprocess.CompletedProcess:
    full = {k: v for k, v in os.environ.items() if not k.startswith("KALOS_")}
    full.update({"KALOS_STATE_DIR": str(tmp_path), "KALOS_HOST": "127.0.0.1", "KALOS_PORT": "8057"})
    full.update(env)
    return subprocess.run(
        [sys.executable, "-m", "kalos.portal"], env=full, capture_output=True, text=True, timeout=180,
    )


def _assert_clean_refusal(r: subprocess.CompletedProcess, needle: str) -> None:
    assert r.returncode == 2, (r.returncode, r.stderr[-500:])
    assert "Traceback" not in r.stderr, r.stderr[-800:]
    lines = [ln for ln in r.stderr.splitlines() if ln.startswith("kalos portal:")]
    assert len(lines) == 1 and needle in lines[0], r.stderr[-800:]


@pytest.mark.parametrize("name,value,needle", [
    ("KALOS_MAX_UPLOAD_MB", "lots", "KALOS_MAX_UPLOAD_MB must be a number"),
    ("KALOS_MAX_UPLOAD_MB", "nan", "KALOS_MAX_UPLOAD_MB must be a finite number"),
    ("KALOS_MAX_FIT_ROWS", "0", "KALOS_MAX_FIT_ROWS must be >= 1"),
    ("KALOS_MAX_FIT_ROWS", "2.5", "KALOS_MAX_FIT_ROWS must be a number"),
    ("KALOS_MAX_REQUEST_BYTES", "-5", "KALOS_MAX_REQUEST_BYTES must be >= 1"),
    ("KALOS_JOB_TIMEOUT_S", "soon", "KALOS_JOB_TIMEOUT_S must be a number"),
    ("KALOS_DEMO_TIMEOUT_S", "0", "KALOS_DEMO_TIMEOUT_S must be >= 1"),
    ("KALOS_TORCH_THREADS", "-1", "KALOS_TORCH_THREADS must be >= 0"),
])
def test_bad_sizing_knob_is_a_refusal(tmp_path, name, value, needle):
    _assert_clean_refusal(_run(tmp_path, **{name: value}), needle)


def test_bad_sizing_knob_error_does_not_echo_a_long_value(tmp_path):
    secret_ish = "x" * 200
    r = _run(tmp_path, KALOS_MAX_FIT_ROWS=secret_ish)
    _assert_clean_refusal(r, "KALOS_MAX_FIT_ROWS must be a number")
    assert secret_ish not in r.stderr


def test_malformed_token_json_is_a_refusal(tmp_path):
    _assert_clean_refusal(_run(tmp_path, KALOS_AUTH_TOKENS="[{not json"), "not valid JSON")


def test_default_tenant_token_is_a_refusal(tmp_path):
    bad = json.dumps([{"token_sha256": hashlib.sha256(b"t").hexdigest(),
                       "subject": "s", "tenant": "default", "scopes": ["read"]}])
    _assert_clean_refusal(_run(tmp_path, KALOS_AUTH_TOKENS=bad), "reserved tenant id 'default'")


def test_missing_token_file_is_a_refusal_without_the_path(tmp_path):
    missing = tmp_path / "no-such-tokens.json"
    r = _run(tmp_path, KALOS_AUTH_TOKENS_FILE=str(missing))
    _assert_clean_refusal(r, "cannot read KALOS_AUTH_TOKENS_FILE")
    assert str(missing) not in r.stderr


def test_production_on_loopback_without_anon_salt_is_a_refusal(tmp_path):
    r = _run(tmp_path, KALOS_ENV="production", KALOS_AUTH_TOKENS=_TOKEN)
    _assert_clean_refusal(r, "KALOS_ANON_SALT")


def test_production_loopback_with_salt_passes_the_preflight(tmp_path):
    """The salt rule is the only thing refusing here: with it set, the same
    config gets past preflight (it then starts uvicorn, so stop it by timeout)."""
    full = {k: v for k, v in os.environ.items() if not k.startswith("KALOS_")}
    full.update({"KALOS_STATE_DIR": str(tmp_path), "KALOS_HOST": "127.0.0.1", "KALOS_PORT": "8058",
                 "KALOS_ENV": "production", "KALOS_AUTH_TOKENS": _TOKEN,
                 "KALOS_ANON_SALT": "a-real-looking-salt"})
    try:
        r = subprocess.run([sys.executable, "-m", "kalos.portal"], env=full,
                           capture_output=True, text=True, timeout=25)
    except subprocess.TimeoutExpired as exc:
        out = (exc.stderr or b"").decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        assert "KALOS_ANON_SALT" not in out and "Traceback" not in out
        return
    assert "KALOS_ANON_SALT" not in r.stderr and "Traceback" not in r.stderr, r.stderr[-800:]
