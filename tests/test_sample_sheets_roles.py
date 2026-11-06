"""On the open sample sheets used in the study (~/Desktop/kalos-samples, not in
the repo; skipped when absent), no input is dropped by its name: only true
outcome columns are. Owner decision: a concentration unit is not outcome
evidence (Candida's ammonium_g_L, scFv's Final Urea (M))."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from kalos.portal.analysis import _analyze
from kalos.portal.uploads import parse_upload

SAMPLES = Path(os.path.expanduser("~/Desktop/kalos-samples"))
# sheet -> (target to name, or None to infer; the inputs that must be used)
EXPECTED = {
    "candida_rsm_nr1": (None, {"molasses_g_L", "ammonium_g_L"}),
    "cho_bioreactor_20829178": (None, {"DO_pct_air", "agitation_rpm", "aeration_sL_h",
                                       "O2_inlet_pct", "CO2_inlet_pct", "feed_total_g"}),
    "pichia_HerceptinA": ("Titer", {"Carbon_conc", "Gly", "Met", "SP"}),
    "pichia_HSA": ("Titer", {"Carbon_conc", "Gly", "Met", "SP"}),
    "pichia_RBDJ": ("Titer", {"Carbon_conc", "Gly", "Met", "SP"}),
    "rudt_ib_refolding_16315733": (None, {"pH", "temperature_C", "feed_rate_coded"}),
    "scfv_refolding_18270843": (None, {"pH", "Final Urea (M)", "Dilution Factor",
                                       "GSSG (mM)", "DTT (mM)"}),
    "trichoderma_ambr": (None, {"strain", "condition"}),
}
NAME_RULES = {"outcome_like_name", "after_target_in_sheet"}


@pytest.mark.parametrize("stem", sorted(EXPECTED))
def test_no_input_is_dropped_by_its_name(stem):
    path = SAMPLES / f"{stem}.csv"
    if not path.exists():
        pytest.skip(f"{path} not present")
    target, inputs = EXPECTED[stem]
    df = parse_upload(path.read_bytes(), filename=path.name).df
    out = _analyze(df, target, preflight_only=True)
    cols = {c["name"]: c for c in out["columns"]}
    by_name = {n: c for n, c in cols.items() if c["reason"] in NAME_RULES}
    assert not by_name, f"dropped by name: {by_name}"
    for name in inputs:
        c = cols[name]
        used = c["used_in_fit"] or c["role"] == "fixed" or c["kind"] == "category"
        assert used, (name, c)
    if stem == "candida_rsm_nr1":
        assert cols["ammonium_g_L"]["used_in_fit"]
        assert cols["ammonium_g_L"]["reason"] == "possibly_measured"
        assert cols["ammonium_g_L"]["finding_code"] == "possibly_measured_after_run"
