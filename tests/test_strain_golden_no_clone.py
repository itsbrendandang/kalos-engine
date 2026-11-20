"""Golden outputs: campaigns WITHOUT a clone column are unchanged by the strain work.

The clone / covariate roles (docs/CLONES.md) must not move a single number for a
sheet that has no clone, strain, cell-line or covariate column. This pins the
full `_analyze` response of five such sheets, recorded with NO strain code on
`feat/trust-calibration` @ 773ec98 (the trust layer, F02/F04/F05/F15, merged
with main @ 8066c86; re-recorded because F05/F15 change the CV grouping and the
intervals for every campaign), earlier on main @ 8066c86 and first on
`feat/web-redesign-apis` @ 2692e7c, and asserts that every key recorded then
is byte-identical now. New keys may be added (they are additive); no
recorded key may change.

Excluded from the comparison, each for a stated reason:
  - `timestamp` and `model_diagnostics.fit_seconds`: wall clock.
  - `gp_shapes` on the MIXED sheet only: it was `available: false, "posterior
    sweep failed: RuntimeError"` at the base (the sweep passed d_cont columns to
    a d-column mixed GP). Fixing that bug is the one intended change; it is
    covered by tests/test_strain_bugfixes.py.

Same-stack claim only (see tests/test_golden_determinism.py and
docs/CAMPAIGNS_API.md "Determinism"): the fixture records the OS, architecture,
Python and library versions and the torch thread count it was produced with. It
is recorded on the CI stack (linux x86_64, Python 3.11, uv.lock, CPU torch;
`scripts/record_goldens.py`), so it runs in CI. On another stack the test skips
with the reason, except when `KALOS_REQUIRE_GOLDENS=1` (set in CI), where a
stack mismatch FAILS instead of skipping.

Re-recorded on the CI stack (VERIFY_MAIN SHOULD-FIX 3: the goldens never ran
in CI). Before that, on the original recording stack (macOS arm64, Python
3.14.7, numpy 2.5.1), main @ cf7105c reproduced the 773ec98 fixture key for key,
and the one later change (the coverage-at-proposals caveat) touched caveat
fields only, so the re-recorded fixture carries the same pre-strain baseline
forward.
"""
from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "strain_golden_no_clone.json"
EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "synthetic_bioprocess" / "synthetic_bioprocess.csv"


def _continuous() -> tuple[pd.DataFrame, str, dict]:
    rng = np.random.default_rng(0)
    m = rng.uniform(0, 4, 24)
    ph = rng.uniform(5, 7, 24)
    df = pd.DataFrame({
        "Methanol": m.round(3), "pH": ph.round(2),
        "lipase_titer": (1.5 * m - 0.8 * (ph - 6) ** 2 + rng.normal(0, 0.1, 24)).round(3),
    })
    return df, "lipase_titer", {}


def _bioprocess_subset() -> tuple[pd.DataFrame, str, dict]:
    df = pd.read_csv(EXAMPLE).head(30)[["pH", "temperature_C", "glucose_gL", "titer_g_L"]]
    return df, "titer_g_L", {}


def _mixed_medium() -> tuple[pd.DataFrame, str, dict]:
    rng = np.random.default_rng(1)
    n = 24
    medium = np.tile(["MedA", "MedB", "MedC"], 8)
    ph = rng.uniform(6.8, 7.2, n)
    t = rng.uniform(33, 37, n)
    off = {"MedA": 0.0, "MedB": 0.4, "MedC": 0.9}
    y = np.array([off[m] for m in medium]) + 2 - 0.3 * (t - 35) ** 2 / 4 + rng.normal(0, 0.15, n)
    df = pd.DataFrame({"medium": medium, "pH": ph.round(3), "temp_C": t.round(2),
                       "titer_g_L": y.round(3)})
    return df, "titer_g_L", {}


def _setup_constrained() -> tuple[pd.DataFrame, str, dict]:
    from kalos.portal.setup_settings import analysis_setup, normalize_setup

    rng = np.random.default_rng(2)
    n = 24
    a = rng.uniform(0, 5, n)
    b = rng.uniform(0, 5, n)
    c = rng.uniform(30, 37, n)
    y = a + 0.5 * b - 0.1 * (c - 34) ** 2 + rng.normal(0, 0.2, n)
    df = pd.DataFrame({"feed_a": a.round(3), "feed_b": b.round(3), "temp_C": c.round(2),
                       "titer_g_L": y.round(3)})
    setup = analysis_setup(normalize_setup({
        "goal": "maximize",
        "linear_constraints": [{"coefficients": {"feed_a": 1.0, "feed_b": 1.0}, "op": "<=", "rhs": 6.0}],
    }))
    return df, "titer_g_L", {"setup": setup}


def _replicated() -> tuple[pd.DataFrame, str, dict]:
    rng = np.random.default_rng(3)
    k = 12
    ph = rng.uniform(6.6, 7.4, k).round(2)
    feed = rng.uniform(1, 6, k).round(2)
    rows = []
    for i in range(k):
        for _ in range(2):
            y = 1 + 0.4 * feed[i] - 2 * (ph[i] - 7) ** 2 + rng.normal(0, 0.15)
            rows.append({"pH": ph[i], "feed_pct": feed[i], "titer_g_L": round(float(y), 3)})
    return pd.DataFrame(rows), "titer_g_L", {}


DATASETS = {
    "continuous": _continuous,
    "bioprocess_subset": _bioprocess_subset,
    "mixed_medium": _mixed_medium,
    "setup_constrained": _setup_constrained,
    "replicated": _replicated,
}
GOLDEN_THREADS = 1
EXCLUDED_KEYS = {"mixed_medium": {"gp_shapes"}}


def _environment() -> dict:
    import botorch
    import gpytorch
    import scipy
    import torch

    return {
        "machine": platform.machine(), "system": platform.system(),
        "python": ".".join(map(str, sys.version_info[:2])),
        "numpy": np.__version__, "scipy": scipy.__version__, "torch": torch.__version__,
        "botorch": botorch.__version__, "gpytorch": gpytorch.__version__,
        "torch_threads": GOLDEN_THREADS,
    }


def _run(name: str) -> dict:
    from kalos.portal.analysis import _analyze
    df, target, kwargs = DATASETS[name]()
    out = json.loads(json.dumps(_analyze(df, target, **kwargs), default=str))
    out.pop("timestamp", None)
    (out.get("model_diagnostics") or {}).pop("fit_seconds", None)
    return out


def _load() -> dict:
    if not FIXTURE.exists():
        pytest.skip("golden fixture missing")
    return json.loads(FIXTURE.read_text())


@pytest.mark.slow
@pytest.mark.serial
@pytest.mark.parametrize("name", sorted(DATASETS))
def test_no_clone_sheet_output_is_unchanged(name):
    golden = _load()
    if golden["environment"] != _environment():
        msg = f"golden recorded on {golden['environment']}, running on {_environment()}"
        if os.environ.get("KALOS_REQUIRE_GOLDENS") == "1":
            pytest.fail(msg + " (KALOS_REQUIRE_GOLDENS=1; re-record on this stack with "
                        "scripts/record_goldens.py and state the reason)")
        pytest.skip(msg)
    want = golden["results"][name]
    got = _run(name)
    changed = [
        k for k in want
        if k not in EXCLUDED_KEYS.get(name, set())
        and json.dumps(want[k], sort_keys=True) != json.dumps(got.get(k), sort_keys=True)
    ]
    assert not changed, f"{name}: keys changed vs the pre-strain golden: {changed}"


def record(recorded_on: str) -> dict:
    """The fixture payload on this stack (scripts/record_goldens.py writes it)."""
    import torch

    torch.set_num_threads(GOLDEN_THREADS)
    return {
        "recorded_on": recorded_on,
        "environment": _environment(),
        "results": {name: _run(name) for name in sorted(DATASETS)},
    }
