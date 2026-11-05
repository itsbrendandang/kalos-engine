"""Golden determinism test (arch review C-3 / T-1).

Claim under test: the same rows and the same seed, analysed twice on the same
machine, library stack and torch thread count, give identical proposals
(recipes, predicted mean and std), identical out-of-fold predictions and the
same `proposal_provenance.provenance_sha256`. Compared exactly (JSON equality of
the rounded output the API returns), not within a tolerance.

Not claimed: equality across machines, BLAS builds, thread counts or library
versions. No expected values are pinned in this file for that reason; a pinned
cross-platform golden needs a tolerance, which is a scientific decision
(docs/CAMPAIGNS_API.md, "Determinism").

Datasets: a synthetic 2-feature sheet (24 rows) and the first 30 rows / 3
features of the committed synthetic example (examples/synthetic_bioprocess).
Both are synthetic; no real data.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kalos.portal.pipeline import compute_upload_analysis

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "synthetic_bioprocess" / "synthetic_bioprocess.csv"


def _two_feature() -> tuple[bytes, str]:
    rng = np.random.default_rng(0)
    m = rng.uniform(0, 4, 24)
    ph = rng.uniform(5, 7, 24)
    df = pd.DataFrame({
        "Methanol": m.round(3), "pH": ph.round(2),
        "lipase_titer": (1.5 * m - 0.8 * (ph - 6) ** 2 + rng.normal(0, 0.1, 24)).round(3),
    })
    return df.to_csv(index=False).encode(), "lipase_titer"


def _bioprocess_subset() -> tuple[bytes, str]:
    df = pd.read_csv(EXAMPLE).head(30)[["pH", "temperature_C", "glucose_gL", "titer_g_L"]]
    return df.to_csv(index=False).encode(), "titer_g_L"


DATASETS = {"two_feature": _two_feature, "bioprocess_subset": _bioprocess_subset}


def _fingerprint(result: dict) -> dict:
    prov = dict(result["proposal_provenance"])
    prov.pop("timestamp")
    prov.pop("created_at")
    return {
        "proposals": result["proposals"],
        "oof": result["oof"],
        "proposal_features": result["proposal_features"],
        "run": result["run"],
        "provenance": prov,
    }


@pytest.mark.parametrize("name", sorted(DATASETS))
def test_same_rows_same_seed_twice_in_process_gives_identical_proposals(name):
    raw, target = DATASETS[name]()
    _, first = compute_upload_analysis(raw, target, False, "golden.csv")
    _, second = compute_upload_analysis(raw, target, False, "golden.csv")
    assert first["proposals"], "the fit returned no proposals; the test would be vacuous"
    a, b = _fingerprint(first), _fingerprint(second)
    # Exact: every recipe value, pred and std as returned by the API.
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    assert a["provenance"]["provenance_sha256"] == b["provenance"]["provenance_sha256"]


_CHILD = """
import json, sys
import pandas as pd
sys.path.insert(0, {tests!r})
from test_golden_determinism import DATASETS, _fingerprint
from kalos.portal.pipeline import compute_upload_analysis
raw, target = DATASETS[{name!r}]()
_, r = compute_upload_analysis(raw, target, False, "golden.csv")
print("GOLDEN" + json.dumps(_fingerprint(r), sort_keys=True))
"""


def test_a_fresh_process_gives_the_same_proposals():
    """Same machine, same stack, a new interpreter (as a background job runs):
    the RNG state of the parent process must not leak into the result."""
    name = "bioprocess_subset"
    raw, target = DATASETS[name]()
    _, here = compute_upload_analysis(raw, target, False, "golden.csv")
    out = subprocess.run(
        [sys.executable, "-c", _CHILD.format(tests=str(Path(__file__).parent), name=name)],
        capture_output=True, text=True, check=True, cwd=str(ROOT), timeout=600,
    ).stdout
    line = next(ln for ln in out.splitlines() if ln.startswith("GOLDEN"))
    assert line[len("GOLDEN"):] == json.dumps(_fingerprint(here), sort_keys=True)


def _bioprocess_80x6() -> tuple[bytes, str]:
    cols = ["pH", "temperature_C", "glucose_gL", "glutamine_mM", "feed_rate_mLh", "DO_pct", "titer_g_L"]
    return pd.read_csv(EXAMPLE).head(80)[cols].to_csv(index=False).encode(), "titer_g_L"


@pytest.mark.xfail(strict=False, reason=(
    "MEASURED NOT DETERMINISTIC (R2): 80 rows x 6 features of the committed synthetic "
    "example, analysed 4 times in one process, gave 4 different proposal batches at "
    "KALOS_TORCH_THREADS=4 and at 1, on a loaded machine; the same happens at 23d11df "
    "(before the R2 surrogate changes). The posterior agrees to ~1e-16 between runs; "
    "the acquisition optimum moves by whole units on differences that small. Cause "
    "not isolated (in-process state that _seeded_analysis does not reset is the lead)."
))
def test_80x6_subset_is_identical_three_times_in_process():
    raw, target = _bioprocess_80x6()
    outs = [json.dumps(compute_upload_analysis(raw, target, False, "g.csv")[1]["proposals"],
                       sort_keys=True) for _ in range(3)]
    assert outs[0] == outs[1] == outs[2]
