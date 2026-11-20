"""Contract guard: the fields kalos-web reads off an /api/run response.

kalos-web hand-maintains its engine response types in `lib/api.ts` (the
`AnalyzeResult` family), with documentation this engine's OpenAPI cannot carry -
FastAPI types the request bodies but leaves the large analysis RESPONSE as an
untyped object, so there is nothing on the schema side to diff against. This
test closes that gap from the engine side: it drives a real `/api/run` through
the TestClient and asserts every top-level field the web depends on is actually
present in the response. If a refactor renames or drops one of these, this test
fails here - before the web silently reads `undefined` in production.

The field list below is copied from the response types in kalos-web
`lib/api.ts`. It is intentionally a top-level-only check: the web tolerates
absent *nested* fields on older engine builds (its types use `?`/`Partial<>`),
but these TOP-LEVEL keys are the backbone every analysis page reads. Keep this
list in sync with `lib/api.ts` when the web starts or stops depending on a field.
"""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from kalos.portal.app import app  # noqa: E402

# Top-level keys kalos-web's AnalyzeResult / PopulatedResult read. Derived from
# lib/api.ts; see that file for the per-field documentation and version rules.
_WEB_ANALYZE_FIELDS: frozenset[str] = frozenset({
    "n", "d", "target", "targets", "features", "categorical_features",
    "auto_categorical", "group_col",
    "cv_spearman", "cv_ci95", "cv_n_groups", "cv_n_repeats",
    "cv_spearman_per_repeat", "cv_logo", "cv_topk", "cv_group_mean_baseline",
    "conformal_q", "conformal_coverage",
    "driver_selection", "drivers",
    "promotion", "reliability",
    "proposal_gating", "proposal_features", "proposal_optimizer", "proposals",
    "constraint", "design_box_exclusions", "feature_ranges",
    "data_drops", "n_dropped_incomplete", "n_censored_excluded",
    "n_pending_considered",
    "best", "noise", "oof", "provenance",
    "embedding", "correlation", "response_surface",
    "model_diagnostics", "gp_shapes",
    "seed", "engine_version",
    # Web redesign APIs (docs: ENGINE_API_V2): R6 unit, R4 per-proposal intervals, R2 slices.
    "target_unit", "interval_levels", "interval_method", "slices", "slices_skipped",
    # Trust layer (F02/F04/F05/F15): the verdict and what it did to the batch,
    # the refusal (null unless refused), the CV grouping used, and the caveat on
    # the global conformal band.
    "verdict", "proposal_trust", "refusal", "cv_grouping", "conformal_caveat",
    # Clones / strains (docs/CLONES.md): present on every analysis, empty without a clone.
    "strain_summary", "slices_by_level", "slices_by_level_skipped", "oof_levels",
    "covariates", "drivers_skipped", "correlation_skipped", "categorical_effects",
    # VIZ_SPEC_V2 E3/E12 (kalos/analysis/viz_fields.py): heat-map fields.
    "categorical_levels", "response_surfaces",
})

# Per-proposal fields the redesign reads (R4): each proposal carries its own
# conformal intervals (or a structured refusal per level) plus model-based bands.
_WEB_PROPOSAL_FIELDS: frozenset[str] = frozenset({
    "recipe", "pred", "std", "mode", "predictive_std", "intervals", "gp_intervals",
    "quantiles", "expected_improvement", "p_improve",
})

# R3: the sigma grid and its labels on the 2D surface.
_WEB_SURFACE_FIELDS: frozenset[str] = frozenset({
    "x_feature", "y_feature", "x_vals", "y_vals", "z", "std", "std_kind", "std_basis", "z_basis",
    "fixed_recipe", "swept_at",
})

_RUN_CSV = (
    "glucose,temperature,titer\n"
    "10,30,2.1\n20,32,3.4\n15,31,2.8\n25,33,3.9\n"
    "12,29,2.3\n18,34,3.1\n22,30,3.5\n14,32,2.6\n"
    "16,31,2.9\n21,33,3.6\n13,30,2.4\n19,32,3.2\n"
)


@pytest.fixture(scope="module")
def analyze_response(tmp_path_factory) -> dict:
    """One /api/run shared by the module, against a PRIVATE store.

    Module-scoped fixtures run before conftest's function-scoped isolation, so
    this one pins its own `KALOS_STATE_DIR` and store singleton (restored on
    exit) instead of reaching whatever `get_campaign_store()` would build."""
    from kalos.portal import campaign

    state = tmp_path_factory.mktemp("web-contract-state")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("KALOS_STATE_DIR", str(state))
        store = campaign.CampaignStore(state)
        mp.setattr(campaign, "_STORE", store)
        try:
            resp = TestClient(app).post(
                "/api/run", files={"file": ("runs.csv", _RUN_CSV, "text/csv")})
        finally:
            store.close()
    assert resp.status_code == 200, resp.text
    return resp.json()


@pytest.mark.slow
def test_api_run_carries_every_field_the_web_reads(analyze_response: dict) -> None:
    """Every top-level key kalos-web's AnalyzeResult reads must be present.

    Marked slow: /api/run runs the full GP analysis pipeline. A missing field
    here means kalos-web will read `undefined` for it - so this is the test
    that catches an engine-side rename or removal before the web does.
    """
    present = set(analyze_response.keys())
    missing = _WEB_ANALYZE_FIELDS - present
    assert not missing, (
        "kalos-web reads these /api/run fields but the engine no longer returns "
        f"them (rename or removal?): {sorted(missing)}. If the web intentionally "
        "stopped using one, drop it from _WEB_ANALYZE_FIELDS and lib/api.ts together."
    )


@pytest.mark.slow
def test_providers_response_shape_matches_web(analyze_response: dict) -> None:
    """kalos-web's ProvidersResponse = { providers: Provider[] }, each Provider
    carrying name/available/required_env/missing_env/capability/fallback."""
    client = TestClient(app)
    resp = client.get("/api/providers")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "providers" in body and isinstance(body["providers"], list)
    web_provider_fields = {
        "name", "available", "required_env", "missing_env", "capability", "fallback",
    }
    for entry in body["providers"]:
        missing = web_provider_fields - set(entry)
        assert not missing, f"/api/providers entry missing web fields: {sorted(missing)}"


@pytest.mark.slow
def test_proposals_and_surface_carry_the_redesign_fields(analyze_response: dict) -> None:
    """R2-R4/R6: per-proposal intervals with their real coverage, the sigma grid,
    slices with a labeled model-based band, and the target unit block."""
    proposals = analyze_response["proposals"]
    assert proposals, "the contract sheet should produce proposals"
    for p in proposals:
        missing = _WEB_PROPOSAL_FIELDS - set(p)
        assert not missing, f"proposal missing web fields: {sorted(missing)}"
        for band in p["intervals"]:
            assert band["coverage"] in analyze_response["interval_levels"]
            # F15: selection-aware per level, the plain normalized band as the
            # per-level fallback; both carry their coverage basis and caveat (F02).
            assert band["method"] in {"conformal_selection_aware", "conformal_normalized"}
            assert band["coverage_basis"] in {"held_out_top_ranked_runs", "held_out_past_runs"}
            assert band["caveat"]["code"]
            if band["available"]:
                assert band["lo"] <= p["pred"] <= band["hi"]
            else:  # a refusal says why, never a guessed band
                assert band["lo"] is None and band["reason"]
        for band in p["gp_intervals"]:
            assert band["method"] == "gp_posterior" and "not held-out calibrated" in band["note"]
    surface = analyze_response["response_surface"]
    if surface is not None:
        assert not _WEB_SURFACE_FIELDS - set(surface)
    for surf in analyze_response["response_surfaces"]:  # E3: same shape per pair
        assert not _WEB_SURFACE_FIELDS - set(surf) and surf["pair_rank"] >= 1
    corr = analyze_response["correlation"]
    if corr is not None:  # E5: the label matches the computation, n per cell
        assert corr["unit"] == "row"
        assert len(corr["n_matrix"]) == len(corr["labels"])
    unit = analyze_response["target_unit"]
    assert set(unit) >= {"unit", "source", "header", "note"}
    if analyze_response["slices"] is not None:
        for sl in analyze_response["slices"]:
            assert sl["band_method"] == "gp_posterior" and sl["coverage"] == 0.9


@pytest.mark.slow
def test_trust_layer_fields_the_web_reads(analyze_response: dict) -> None:
    """The trust blocks the web renders: verdict -> proposal_trust (with every
    proposal labeled), refusal (null unless refused), cv_grouping (which
    grouping the CV used and why) and conformal_caveat (basis + coverage
    measured at proposals, never a bare 'calibrated')."""
    r = analyze_response
    verdict = r["verdict"]
    assert verdict["state"] in {"reliable", "not_yet", "refused"}
    assert verdict["code"] and verdict["headline"] and isinstance(verdict["reasons"], list)
    assert all(reason["code"] and reason["plain"] for reason in verdict["reasons"])

    trust = r["proposal_trust"]
    assert trust["verdict"] == verdict["state"]
    assert trust["status"] in {"trusted", "exploration_untrusted", "withheld"}
    assert trust["withheld"] is (trust["status"] == "withheld")
    assert trust["gated"] is (trust["status"] != "trusted")
    assert trust["n_returned"] == len(r["proposals"]) and trust["message"]
    assert isinstance(trust["reasons"], list)
    for p in r["proposals"]:
        assert p["trust"]["status"] == trust["status"] and p["trust"]["message"]

    refusal = r["refusal"]
    if verdict["state"] == "refused":
        assert refusal["code"] and refusal["plain"] and refusal["what_would_help"]
        assert r["proposals"] == [] and trust["withheld"] is True
    else:
        assert refusal is None

    grouping = r["cv_grouping"]
    assert grouping["used"]["basis"] and grouping["used"]["n_groups"] == r["cv_n_groups"]
    assert grouping["rule"] and isinstance(grouping["warnings"], list)
    assert any(c["used"] for c in grouping["candidates"])

    caveat = r["conformal_caveat"]
    assert caveat["code"] == "coverage_not_guaranteed"
    assert caveat["coverage_basis"] == "held_out_past_runs"
    assert caveat["message"] and "is calibrated" not in caveat["message"]
    est = caveat["coverage_at_proposals_estimate"]
    assert est["basis"] == "simulation" and 0 <= est["measured_min"] <= est["measured_max"] <= 1


_CAMPAIGN_CSV = "glucose,temperature,titer_g_L\n" + "".join(
    f"{10 + i},{29 + (i % 6)},{2.0 + 0.1 * i + 0.05 * (i % 3)}\n" for i in range(14)
)


@pytest.mark.slow
def test_campaign_routes_the_redesign_reads() -> None:
    """The new campaign-scoped routes answer with the top-level shapes the web reads."""
    client = TestClient(app)
    cid = client.post("/api/campaigns", json={"name": "contract"}).json()["id"]
    up = client.post(
        f"/api/campaigns/{cid}/upload",
        files={"file": ("runs.csv", _CAMPAIGN_CSV, "text/csv")},
    )
    assert up.status_code == 200, up.text

    listing = client.get("/api/campaigns?summary=true").json()
    card = next(c for c in listing["campaigns"] if c["id"] == cid)
    assert {"best_measured", "analysis", "goal", "n_awaiting", "current_stage",
            "latest_gate_decision", "active_job"} <= set(card)
    assert card["best_measured"]["basis"] == "measured"

    stages = client.get(f"/api/campaigns/{cid}/stages").json()
    assert [s["key"] for s in stages["stages"]] == ["S1", "S2", "S3", "S4", "S5", "S6"]
    gates = client.get(f"/api/campaigns/{cid}/gates").json()
    assert all(g["computed_evidence"]["basis"] == "computed" for g in gates["gates"])
    assert client.put(f"/api/campaigns/{cid}/gates/S1", json={"decision": "go"}).status_code == 400

    setup = client.get(f"/api/campaigns/{cid}/setup").json()
    assert setup["setup"]["goal"] == "maximize" and "settings_in_sync" in setup

    rows = client.get(f"/api/campaigns/{cid}/export/rows").json()
    assert rows["original_rows"]["n"] == 14
    history = client.get(f"/api/campaigns/{cid}/results/history").json()
    assert {"results", "audit"} <= set(history)
    ranges = client.post(f"/api/campaigns/{cid}/operating-ranges", json={"probability": 0.9}).json()
    assert ranges["status"] == "not_computed" and ranges["reason_code"] == "threshold_missing"


_CLONE_CSV = "clone,passage,temperature,feed,titer_g_L\n" + "".join(
    f"{c},{8 + i},{33 + (i % 5)},{2 + (i % 4)},"
    f"{2.0 + {'CHO-A12': 0.0, 'CHO-B07': 0.5, 'CHO-C33': 1.0}[c] + 0.2 * (i % 4) - 0.05 * (i % 5) ** 2}\n"
    for i, c in enumerate(["CHO-A12", "CHO-B07", "CHO-C33"] * 6)
)


@pytest.mark.slow
def test_clone_fields_and_routes_the_web_renders() -> None:
    """G1-G5 read only engine fields: strain_summary, slices_by_level (labeled
    model band with coverage, never calibrated), oof_levels aligned to oof, the
    held covariate, the clone chip on proposals, and the five registry routes."""
    client = TestClient(app)
    cid = client.post("/api/campaigns", json={"name": "clones"}).json()["id"]
    assert client.put(f"/api/campaigns/{cid}/clones/CHO-C33",
                      json={"status": "lead", "host": "CHO-K1"}).status_code == 200
    up = client.post(f"/api/campaigns/{cid}/upload",
                     files={"file": ("runs.csv", _CLONE_CSV, "text/csv")})
    assert up.status_code == 200, up.text
    a = client.get(f"/api/campaigns/{cid}/analysis").json()
    a = a.get("analysis", a)
    s = a["strain_summary"]
    assert s["available"] and s["column"] == "clone" and s["source"] == "detected"
    lv = s["levels"][0]
    assert {"label", "n_runs", "n_recipes", "outcomes", "adjusted_effect", "gp_at_reference",
            "covariates", "flags"} <= set(lv)
    o = lv["outcomes"]["titer_g_L"]
    assert o["basis"] == "measured" and {"mean", "ci95", "ci_reason", "n_runs"} <= set(o)
    assert lv["adjusted_effect"]["basis"] == "derived"
    assert lv["gp_at_reference"]["basis"] == "predicted"
    assert {"rows", "cells", "thin_rule", "note"} <= set(s["interaction"])
    for entries in a["slices_by_level"]["clone"].values():
        for sl in entries:
            assert sl["band_kind"] == "gp_posterior" and sl["calibrated"] is False
            assert isinstance(sl["coverage"], float) and sl["basis"] == "predicted"
    assert len(a["oof_levels"]["clone"]) == len(a["oof"])
    assert a["covariates"]["columns"] == ["passage"]
    for p in a["proposals"]:
        assert "passage" not in p["recipe"] and p["held"]["passage"]["proposed"] is False
        assert p["clone"]["level"] == p["recipe"]["clone"]

    # Declare the clone in the setup: fixed policy, the registry lead by default.
    r = client.put(f"/api/campaigns/{cid}/setup", json={"strain_column": "clone"})
    assert r.status_code == 200, r.text
    reg = client.get(f"/api/campaigns/{cid}/clones").json()
    assert reg["clones"][0]["label"] == "CHO-C33" and "unregistered_labels" in reg
    assert set(reg["seen_in_data"]) == {"CHO-A12", "CHO-B07", "CHO-C33"}
    summ = client.get(f"/api/campaigns/{cid}/clones/summary").json()
    assert summ["available"] and summ["registry_order"][0] == "CHO-C33"
    h = client.post(f"/api/campaigns/{cid}/clones/CHO-C33/titer-history",
                    json={"basis": "harvest_titer", "value": 3.1, "unit": "g/L"})
    assert h.status_code in (200, 201), h.text
    hist = client.get(f"/api/campaigns/{cid}/clones/CHO-C33/titer-history").json()
    assert hist["history"]["items"][0]["basis_kind"] == "measured"
    assert client.put(f"/api/campaigns/{cid}/clones/CHO-C33",
                      json={"status": "lead", "sequence": "MKT"}).status_code == 400
