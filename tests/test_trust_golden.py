"""Trust-layer goldens (scripts/trust_golden.py): the full `_analyze` response on
eight synthetic sheets equals the recorded fixture, key for key.

Every flagged trust change (F02/F04/F05/F15) re-records the fixture with
`python scripts/trust_golden.py record --reason "..."`; the fixture's `reasons`
list is its change log and the commit carries the diff. Same-machine claim only
(tests/test_golden_determinism.py): skipped when the platform, the library
versions or the torch thread count differ from the recording.
"""
from __future__ import annotations

import importlib.util
import json
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
        pytest.skip(f"recorded on {fixture['env']}, running on {env}")
    assert fixture["reasons"], "the fixture must say why it was last recorded"
    current = tg.compute()
    report = tg.diff(fixture["sheets"], current)
    assert all(line.endswith("unchanged") for line in report.splitlines()), report
