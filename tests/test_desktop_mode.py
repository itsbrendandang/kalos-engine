"""Desktop mode (kalos/portal/desktop.py, docs/DESKTOP.md): defaults, refusals,
the startup self-check, the shutdown guard, GET /api/health, and a torch-free
portal import. Every test uses a temp state dir; ~/.kalos is never touched."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")


ROOT = Path(__file__).resolve().parents[1]


def _tokens(secret: str = "desk-secret") -> str:
    return json.dumps([{
        "token_sha256": hashlib.sha256(secret.encode()).hexdigest(),
        "subject": "desktop-app", "tenant": "local", "scopes": ["read", "write"],
    }])


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


