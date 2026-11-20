"""Trust-layer goldens (scripts/trust_golden.py): the full `_analyze` response on
eight synthetic sheets equals the recorded fixture, key for key.

Every flagged trust change (F02/F04/F05/F15) re-records the fixture with
`python scripts/trust_golden.py record --reason "..."`; the fixture's `reasons`
list is its change log and the commit carries the diff. Same-stack claim only
(tests/test_golden_determinism.py). The fixture is recorded on the CI stack
(linux x86_64, Python 3.11, uv.lock, CPU torch; `scripts/record_goldens.py`), so
it runs in CI. On another stack it is skipped with the reason, except when
`KALOS_REQUIRE_GOLDENS=1` (set in CI), where a stack mismatch FAILS: CI must
never skip the goldens silently.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "trust_golden.json"

pytestmark = [pytest.mark.slow, pytest.mark.serial]


def _harness():
    spec = importlib.util.spec_from_file_location("trust_golden", ROOT / "scripts" / "trust_golden.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_trust_goldens_unchanged():
    import torch

    tg = _harness()
    fixture = json.loads(FIXTURE.read_text())
    torch.set_num_threads(1)
    env = tg._env()
    if env != fixture["env"]:
        msg = f"trust goldens recorded on {fixture['env']}, running on {env}"
        if os.environ.get("KALOS_REQUIRE_GOLDENS") == "1":
            pytest.fail(msg + " (KALOS_REQUIRE_GOLDENS=1; re-record on this stack with "
                        "scripts/record_goldens.py and state the reason)")
        pytest.skip(msg)
    assert fixture["reasons"], "the fixture must say why it was last recorded"
    current = tg.compute()
    report = tg.diff(fixture["sheets"], current)
    assert all(line.endswith("unchanged") for line in report.splitlines()), report
