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
})

_RUN_CSV = (
    "glucose,temperature,titer\n"
    "10,30,2.1\n20,32,3.4\n15,31,2.8\n25,33,3.9\n"
    "12,29,2.3\n18,34,3.1\n22,30,3.5\n14,32,2.6\n"
    "16,31,2.9\n21,33,3.6\n13,30,2.4\n19,32,3.2\n"
)


@pytest.fixture(scope="module")
def analyze_response() -> dict:
    client = TestClient(app)
    resp = client.post("/api/run", files={"file": ("runs.csv", _RUN_CSV, "text/csv")})
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
