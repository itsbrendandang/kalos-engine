"""Cold-start exploration designs: kalos.core.design (the generator),
kalos.portal.design_plan (campaign -> design, honest labels, CSV) and
`POST /api/campaigns/{id}/design` (kalos.portal.design_api).

Torch-free: no GP is fit. Seeded campaigns carry a hand-written analysis blob
(the fields the design reads), as in tests/test_setup_api.py.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import cdist

from kalos.core.design import (
    METHODS,
    CategoricalFactor,
    ContinuousFactor,
    DesignError,
    LinearConstraint,
    generate_design,
)
from kalos.portal.design_plan import FORBIDDEN_ROW_KEYS, LABEL, MAX_DESIGN_ROWS

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from kalos.portal import app as portal_module  # noqa: E402
from kalos.portal import campaign as campaign_module  # noqa: E402
from kalos.portal.app import app  # noqa: E402
from kalos.portal.campaign import CampaignStore, get_campaign_store  # noqa: E402

PH = ContinuousFactor("pH", 6.0, 7.5, step=0.05)
TEMP = ContinuousFactor("temperature_C", 30.0, 37.0, step=0.5)
GLC = ContinuousFactor("glucose_gL", 1.0, 40.0)
FEED = ContinuousFactor("feed_rate_mLh", 0.0, 5.0)
CLONE = CategoricalFactor("clone", ("A", "B", "C"))


def _forbidden_keys(obj, path=""):
    """Every key anywhere in `obj` that would carry a prediction."""
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in FORBIDDEN_ROW_KEYS:
                out.append(f"{path}.{k}")
            out += _forbidden_keys(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out += _forbidden_keys(v, f"{path}[{i}]")
    return out


# --------------------------------------------------------------------------- #
# generator
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("method", METHODS)
def test_every_method_is_seeded_deterministic_and_inside_bounds(method):
    a = generate_design([PH, TEMP, GLC, FEED], [CLONE], n=12, method=method, seed=7)
    b = generate_design([PH, TEMP, GLC, FEED], [CLONE], n=12, method=method, seed=7)
    c = generate_design([PH, TEMP, GLC, FEED], [CLONE], n=12, method=method, seed=8)
    assert a.rows == b.rows and a.quality == b.quality
    assert a.rows != c.rows
    assert len(a.rows) == 12 and a.method == method
    for r in a.rows:
        for f in (PH, TEMP, GLC, FEED):
            assert f.lo <= r[f.name] <= f.hi


def test_the_global_rngs_are_not_touched():
    state = np.random.get_state()
    generate_design([PH, GLC], n=8, method="lhs_maximin", seed=1)
    after = np.random.get_state()
    assert state[0] == after[0] and np.array_equal(state[1], after[1]) and state[2:] == after[2:]


def test_core_design_is_torch_free():
    code = "import sys, kalos.core.design; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_steps_and_levels_snap_to_real_setpoints():
    temp_levels = ContinuousFactor("temperature_C", 30.0, 37.0, levels=(30.0, 33.0, 37.0, 50.0))
    res = generate_design([PH, temp_levels, GLC], n=20, seed=3)
    for r in res.rows:
        k = (r["pH"] - 6.0) / 0.05
        assert abs(k - round(k)) < 1e-9
        assert r["pH"] == round(r["pH"], 2)  # no binary noise: 6.35, not 6.3500000000000005
        assert r["temperature_C"] in (30.0, 33.0, 37.0)  # 50 is outside the bounds
    assert res.snapping["factors"]["pH"] == {"step": 0.05}
    assert res.snapping["factors"]["temperature_C"] == {"levels": [30.0, 33.0, 37.0]}
    # the step grid is absolute (k * step), not anchored at lo
    off = generate_design([ContinuousFactor("pH", 6.02, 7.5, step=0.05)], n=10, seed=1)
    assert all(abs(r["pH"] / 0.05 - round(r["pH"] / 0.05)) < 1e-9 and r["pH"] >= 6.05 for r in off.rows)
    # a truly continuous factor is never given a step, only a 1e-4-of-range resolution
    assert res.snapping["factors"]["glucose_gL"] is None


def test_linear_constraints_are_respected_and_rejections_counted():
    cons = [LinearConstraint({"glucose_gL": 1.0, "feed_rate_mLh": 8.0}, "<=", 30.0)]
    res = generate_design([GLC, FEED], constraints=cons, n=16, method="sobol", seed=2)
    assert len(res.rows) == 16
    for r in res.rows:
        assert r["glucose_gL"] + 8.0 * r["feed_rate_mLh"] <= 30.0 + 1e-6
    c = res.constraints
    assert c["pool_candidates_rejected"] > 0 and c["design_candidates_drawn"] == 16
    assert 0 < c["pool_acceptance_rate"] < 1
    assert c["design_candidates_rejected"] >= 0
    ge = [LinearConstraint({"glucose_gL": 1.0}, ">=", 20.0)]
    assert all(r["glucose_gL"] >= 20.0 - 1e-6
               for r in generate_design([GLC, FEED], constraints=ge, n=8, seed=1).rows)


def test_an_infeasible_region_is_refused_with_counts():
    cons = [LinearConstraint({"glucose_gL": 1.0}, ">=", 100.0)]
    with pytest.raises(DesignError) as exc:
        generate_design([GLC, FEED], constraints=cons, n=4)
    assert exc.value.code == "no_feasible_candidates" and "rejected" in exc.value.reason


def test_categoricals_are_balanced_counting_existing_runs_and_held_levels_hold():
    res = generate_design([GLC, FEED], [CLONE], n=12, seed=0)
    counts = res.levels["clone"]["design_counts"]
    assert sorted(counts.values()) == [4, 4, 4]
    ex = np.array([[10.0, 1.0], [20.0, 2.0], [30.0, 3.0]])
    res = generate_design([GLC, FEED], [CLONE], n=6, seed=0, existing=ex,
                          existing_labels=[{"clone": "A"}] * 3)
    assert res.levels["clone"]["design_counts"] == {"A": 0, "B": 3, "C": 3}
    assert res.levels["clone"]["existing_counts"] == {"A": 3, "B": 0, "C": 0}
    held = generate_design([GLC], [CategoricalFactor("clone", ("A", "B"), fixed="B")], n=5)
    assert {r["clone"] for r in held.rows} == {"B"}
    assert held.levels["clone"]["policy"] == "held"
    two = generate_design([GLC], [CLONE, CategoricalFactor("medium", ("m1", "m2"))], n=12, seed=4)
    pairs = [(r["clone"], r["medium"]) for r in two.rows]
    assert len(set(pairs)) == 6 and all(pairs.count(p) == 2 for p in set(pairs))


def test_centre_points_and_replicate_slots():
    res = generate_design([PH, GLC], n=6, seed=1, centre_points=2, replicate_slots=3)
    assert res.roles == ["space_filling"] * 6 + ["centre"] * 2 + ["replicate"] * 3
    centre = res.rows[6]
    assert centre == res.rows[7]
    assert centre["pH"] == 6.75 and centre["glucose_gL"] == 20.5
    for i, src in enumerate(res.replicate_of):
        if src is not None:
            assert res.rows[i] == res.rows[src]
    # a centre that breaks a constraint moves to the feasible candidate nearest the centre
    cons = [LinearConstraint({"pH": 1.0}, "<=", 6.5)]
    res = generate_design([PH, GLC], constraints=cons, n=4, centre_points=1, seed=1)
    c = res.rows[res.roles.index("centre")]
    assert c["pH"] <= 6.5 and any("centre" in note for note in res.notes)


def test_existing_runs_are_augmented_not_ignored():
    rng = np.random.default_rng(0)
    facs = [ContinuousFactor(f"x{j}", 0.0, 1.0) for j in range(4)]
    wins = 0
    for s in range(10):
        ex = rng.random((5, 4))
        res = generate_design(facs, n=8, seed=s, existing=ex)
        assert res.method == "maximin_fill" and res.n_existing == 5
        new = np.array([[r[f"x{j}"] for j in range(4)] for r in res.rows])
        d_new = float(cdist(new, ex).min())
        assert res.quality["min_distance_to_existing"] == pytest.approx(d_new, abs=1e-5)
        wins += d_new > res.quality["random_reference"]["min_distance_to_existing"]
    assert wins >= 9  # gap filling beats random points at staying away from the existing runs
    with_nan = generate_design(facs, n=3, existing=np.array([[0.1, np.nan, 0.2, 0.3], [0.5] * 4]))
    assert with_nan.n_existing == 1 and with_nan.notes


def test_a_coarse_grid_reports_a_shortfall_instead_of_duplicating():
    a = ContinuousFactor("a", 0.0, 1.0, levels=(0.0, 1.0))
    b = ContinuousFactor("b", 0.0, 1.0, levels=(0.0, 1.0))
    res = generate_design([a, b], n=6, seed=0)
    assert len(res.rows) == 4 and res.shortfall["returned"] == 4
    assert len({(r["a"], r["b"]) for r in res.rows}) == 4


@pytest.mark.parametrize(("kw", "code"), [
    ({"step": 0.0}, "invalid_step"),
    ({"step": 5.0}, "invalid_step"),
    ({"levels": (9.0,)}, "invalid_levels"),
    ({"step": 0.1, "levels": (0.5,)}, "invalid_factor"),
])
def test_invalid_factor_declarations(kw, code):
    with pytest.raises(DesignError) as exc:
        ContinuousFactor("x", 0.0, 1.0, **kw)
    assert exc.value.code == code


def test_quality_metrics_are_reported():
    res = generate_design([PH, TEMP, GLC, FEED], n=12, method="lhs_maximin", seed=0)
    q = res.quality
    assert q["min_pairwise_distance"] > 0 and q["centered_l2_discrepancy"] > 0
    ref = q["random_reference"]
    assert ref["min_pairwise_distance"] > 0
    assert q["min_pairwise_distance"] > ref["min_pairwise_distance"]


# --------------------------------------------------------------------------- #
# the route
# --------------------------------------------------------------------------- #


def _sha(t: str) -> str:
    return hashlib.sha256(t.encode()).hexdigest()


TOKENS = [
    {"token_sha256": _sha("a-admin"), "subject": "alice@acme", "tenant": "acme",
     "scopes": ["read", "write", "admin"]},
    {"token_sha256": _sha("a-write"), "subject": "bob@acme", "tenant": "acme", "scopes": ["read", "write"]},
    {"token_sha256": _sha("a-read"), "subject": "carol@acme", "tenant": "acme", "scopes": ["read"]},
    {"token_sha256": _sha("b-admin"), "subject": "gina@globex", "tenant": "globex",
     "scopes": ["read", "write", "admin"]},
]


def H(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def env(tmp_path, monkeypatch):
    store = CampaignStore(tmp_path)
    monkeypatch.setenv("KALOS_AUTH_TOKENS", json.dumps(TOKENS))
    monkeypatch.delenv("KALOS_AUTH_TOKENS_FILE", raising=False)
    monkeypatch.setattr(portal_module, "_STATE_DIR", tmp_path)
    monkeypatch.setattr(portal_module, "_LATEST_DIR", tmp_path / "latest")
    monkeypatch.setattr(campaign_module, "_STORE", store)
    app.dependency_overrides[get_campaign_store] = lambda: store
    try:
        yield TestClient(app), store
    finally:
        app.dependency_overrides.pop(get_campaign_store, None)


def _empty(store: CampaignStore, tenant: str = "acme") -> str:
    return store.create_campaign(tenant=tenant, name="cold", created_by="t")["id"]


def _seeded(store: CampaignStore, *, analysis: dict | None = None, rows: int = 4) -> str:
    cid = _empty(store)
    df = pd.DataFrame({
        "sample": [f"S{i}" for i in range(rows)],
        "pH": [6.2, 6.8, 7.1, 6.5, 6.9, 7.3][:rows],
        "glucose_gL": [5.0, 20.0, 35.0, 12.0, 28.0, 8.0][:rows],
        "clone": ["A", "A", "B", "A", "B", "A"][:rows],
        "titer": [1.0, 2.0, 1.5, 2.2, 1.1, 0.7][:rows],
    })
    blob = analysis or {
        "target": "titer", "features": ["pH", "glucose_gL", "clone"], "categorical_features": ["clone"],
        "feature_ranges": {"pH": [6.2, 7.1], "glucose_gL": [5.0, 35.0]}, "proposals": [],
        "proposal_trust": {"status": "withheld"},
        "verdict": {"state": "refused", "reasons": [{"code": "too_few_groups"}]},
        "run": {"anonymize": False},
    }
    store.seed(df, "titer", ["pH", "glucose_gL", "clone"], tenant="acme", campaign_id=cid, analysis=blob)
    return cid


def _post(tc, cid, body=None, token="a-write"):
    return tc.post(f"/api/campaigns/{cid}/design", json=body or {}, headers=H(token))


COLD_BODY = {
    "n": 12, "seed": 3,
    "factors": {"pH": {"lo": 6.0, "hi": 7.5, "step": 0.05},
                "temperature_C": {"lo": 30, "hi": 37, "step": 0.5, "unit": "°C"},
                "glucose_gL": {"lo": 1, "hi": 40, "unit": "g/L"}},
    "categoricals": {"medium": ["CD-CHO", "ProCHO"]},
    "target": "titer_g_L",
}


def test_cold_start_plan_is_labelled_and_carries_no_prediction(env):
    tc, store = env
    cid = _empty(store)
    r = _post(tc, cid, COLD_BODY)
    assert r.status_code == 200, r.text
    d = r.json()["design"]
    assert d["label"] == LABEL and d["model_used"] is False and d["predictions"] is None
    assert d["model_not_used"]["code"] == "no_measured_runs"
    assert d["model_not_used"]["thresholds"]["min_runs_to_judge"] == 12
    assert d["n_rows"] == 12 and len(d["rows"]) == 12
    assert all(row["label"] == LABEL and row["model_used"] is False for row in d["rows"])
    assert _forbidden_keys(d) == []
    for row in d["rows"]:
        rec = row["recipe"]
        assert 6.0 <= rec["pH"] <= 7.5 and abs((rec["pH"] - 6.0) / 0.05 - round((rec["pH"] - 6.0) / 0.05)) < 1e-9
        assert rec["temperature_C"] * 2 == int(rec["temperature_C"] * 2)
        assert rec["medium"] in ("CD-CHO", "ProCHO")
    assert d["levels"]["medium"]["design_counts"] == {"CD-CHO": 6, "ProCHO": 6}
    q = d["quality"]
    assert q["min_pairwise_distance"] > 0 and q["centered_l2_discrepancy"] is not None
    assert d["existing"]["mode"] == "fresh design"
    assert d["template"]["columns"] == ["run_id", "pH", "temperature_C", "glucose_gL", "medium", "titer_g_L"]
    # the same request gives the same plan
    assert _post(tc, cid, COLD_BODY).json() == r.json()


def test_csv_is_the_upload_template(env):
    tc, store = env
    cid = _empty(store)
    r = _post(tc, cid, {**COLD_BODY, "format": "csv", "centre_points": 1, "replicate_slots": 1})
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("text/csv")
    assert "exploration-design-no-model" in r.headers["content-disposition"]
    assert r.headers["x-kalos-design-label"] == LABEL
    assert r.headers["x-kalos-content-sha256"] == hashlib.sha256(r.content).hexdigest()
    rows = list(csv.reader(io.StringIO(r.text)))
    assert rows[0] == ["run_id", "pH", "temperature_C", "glucose_gL", "medium", "titer_g_L"]
    assert len(rows) == 1 + 14
    assert rows[1][0] == "design-01" and all(row[-1] == "" for row in rows[1:])
    # the sheet parses as a run sheet: run_id is an id, the factors are columns
    from kalos.portal.analysis_parts.identifiers import _is_identifier_name

    assert _is_identifier_name(rows[0][0])


def test_formula_like_levels_are_neutralized_in_csv(env):
    tc, store = env
    cid = _empty(store)
    body = {"n": 2, "factors": {"x": {"lo": 0, "hi": 1}}, "categoricals": {"m": ["=HYPERLINK(1)", "ok"]},
            "format": "csv"}
    r = _post(tc, cid, body)
    assert r.status_code == 200
    assert "'=HYPERLINK(1)" in r.text and ",=HYPERLINK" not in r.text


def test_seeded_campaign_is_augmented_and_says_why_the_model_is_not_used(env):
    tc, store = env
    cid = _seeded(store)
    store.start([{"recipe": {"pH": 7.0, "glucose_gL": 10.0, "clone": "B"}}], tenant="acme", campaign_id=cid)
    r = _post(tc, cid, {"n": 6, "factors": {"pH": {"step": 0.1}}})
    assert r.status_code == 200, r.text
    d = r.json()["design"]
    assert d["method"] == "maximin_fill"
    assert d["existing"]["n_measured"] == 4 and d["existing"]["n_awaiting"] == 1
    assert d["existing"]["n_used_for_gap_filling"] == 5
    assert d["model_not_used"]["code"] == "n_below_trust_threshold"
    assert d["model_not_used"]["proposal_trust"] == "withheld"
    ph = next(f for f in d["factors"] if f["name"] == "pH")
    assert ph["bounds_source"] == "observed_range" and ph["step"] == 0.1
    assert any("inside the range already measured" in w for w in d["warnings"])
    assert d["levels"]["clone"]["policy"] == "balanced"
    assert "sample" in d["template"]["columns"] and "run_id" not in d["template"]["columns"]
    assert _forbidden_keys(d) == []
    # pending runs can be left out
    d2 = _post(tc, cid, {"n": 6, "include_pending": False}).json()["design"]
    assert d2["existing"]["n_used_for_gap_filling"] == 4


def test_setup_bounds_constraints_and_clone_policy_apply(env):
    tc, store = env
    cid = _seeded(store)
    setup = {"bounds": {"pH": {"lo": 6.0, "hi": 7.4}, "glucose_gL": {"lo": 2.0, "hi": 40.0}},
             "linear_constraints": [{"coefficients": {"pH": 1.0, "glucose_gL": 0.05}, "op": "<=", "rhs": 8.0}],
             "strain_column": "clone", "strain_policy": "factor"}
    assert tc.put(f"/api/campaigns/{cid}/setup", json=setup, headers=H("a-write")).status_code == 200
    d = _post(tc, cid, {"n": 9}).json()["design"]
    for row in d["rows"]:
        rec = row["recipe"]
        assert 6.0 <= rec["pH"] <= 7.4 and 2.0 <= rec["glucose_gL"] <= 40.0
        assert rec["pH"] + 0.05 * rec["glucose_gL"] <= 8.0 + 1e-6
    ph = next(f for f in d["factors"] if f["name"] == "pH")
    assert ph["bounds_source"] == "setup_bounds" and ph["beyond_observed_range"] is True
    assert d["constraints"]["n_constraints"] == 1
    clone = next(f for f in d["factors"] if f["name"] == "clone")
    assert clone["kind"] == "clone" and clone["policy"] == "balanced"
    # existing runs: A x3, B x1 -> the plan favours B
    counts = d["levels"]["clone"]["design_counts"]
    assert counts["B"] > counts["A"]
    # fixed policy holds the named clone
    setup["strain_policy"], setup["strain_level"] = "fixed", "B"
    assert tc.put(f"/api/campaigns/{cid}/setup", json=setup, headers=H("a-write")).status_code == 200
    d = _post(tc, cid, {"n": 4}).json()["design"]
    assert {row["recipe"]["clone"] for row in d["rows"]} == {"B"}


def test_fixed_clone_policy_without_a_level_is_refused_on_a_cold_campaign(env):
    tc, store = env
    cid = _empty(store)
    assert tc.put(f"/api/campaigns/{cid}/setup", json={"strain_column": "clone"}, headers=H("a-write")).status_code == 200
    body = {"n": 6, "factors": {"x": {"lo": 0, "hi": 1}}, "categoricals": {"clone": ["c1", "c2"]}}
    r = _post(tc, cid, body)
    assert r.status_code == 400 and r.json()["code"] == "clone_level_required"
    r = _post(tc, cid, {"n": 6, "factors": {"x": {"lo": 0, "hi": 1}}})
    assert r.status_code == 400 and r.json()["code"] == "clone_levels_required"


@pytest.mark.parametrize(("body", "code"), [
    ({"n": 4, "factors": {"nope": {"step": 1.0}}}, "unknown_factor"),
    ({"n": 4, "factors": {"pH": {"lo": 1.0, "hi": 15.0}}}, "bounds_conflict"),
])
def test_seeded_refusals(env, body, code):
    tc, store = env
    cid = _seeded(store)
    tc.put(f"/api/campaigns/{cid}/setup", json={"bounds": {"pH": {"lo": 6.0, "hi": 7.4}}}, headers=H("a-write"))
    r = _post(tc, cid, body)
    assert r.status_code == 400 and r.json()["code"] == code
    assert "nope" not in r.text  # caller input is never echoed


@pytest.mark.parametrize(("body", "code"), [
    ({"n": 4, "factors": {"pH": {"lo": 5.0, "hi": 15.0}}}, "physically_impossible_bounds"),
    ({"n": 4, "factors": {"x": {"lo": 0.0}}}, "factor_without_bounds"),
    ({"n": 4}, "no_factors"),
])
def test_cold_refusals(env, body, code):
    tc, store = env
    r = _post(tc, _empty(store), body)
    assert r.status_code == 400 and r.json()["code"] == code, r.text


@pytest.mark.parametrize("body", [
    {"n": MAX_DESIGN_ROWS + 1},
    {"n": 90, "centre_points": 4, "replicate_slots": 4},
    {"n": 0},
    {"n": 4, "method": "random"},
    {"n": 4, "unknown": 1},
    {"n": "4"},
    {"n": 4, "seed": -1},
    {"n": 4, "factors": {"x": {"lo": 0, "hi": 1, "step": -1}}},
])
def test_size_caps_and_body_validation(env, body):
    tc, store = env
    cid = _empty(store)
    r = _post(tc, cid, {"factors": {"x": {"lo": 0, "hi": 1}}, **body})
    assert r.status_code == 400, r.text


def test_scope_tenant_and_id_rules(env):
    tc, store = env
    cid = _empty(store)
    body = {"n": 4, "factors": {"x": {"lo": 0, "hi": 1}}}
    assert _post(tc, cid, body, token="a-read").status_code == 403
    assert tc.post(f"/api/campaigns/{cid}/design", json=body).status_code == 401
    foreign = _post(tc, cid, body, token="b-admin")
    malformed = _post(tc, "cmp_nope", body)
    missing = _post(tc, "cmp_" + "0" * 32, body)
    assert foreign.status_code == malformed.status_code == missing.status_code == 404
    assert foreign.content == malformed.content == missing.content
    store.update_campaign(tenant="acme", campaign_id=cid, archived=True)
    assert _post(tc, cid, body).status_code == 409


def test_every_call_is_audited_without_the_body(env):
    tc, store = env
    cid = _empty(store)
    _post(tc, cid, {"n": 3, "factors": {"SECRET_FACTOR": {"lo": 0, "hi": 1}}})
    _post(tc, cid, {"n": 3, "factors": {"SECRET_FACTOR": {"lo": 1, "hi": 0}}})
    rows = [e for e in store.audit_entries(tenant="acme") if e["action"] == "campaign.design"]
    assert [e["outcome"] for e in rows][:2] in (["http_400", "ok"], ["ok", "http_400"])
    assert "SECRET_FACTOR" not in json.dumps(rows)


def test_anonymized_campaign_aliases_clone_labels_and_refuses_the_template(env):
    tc, store = env
    blob = {
        "target": "titer", "features": ["pH", "glucose_gL", "clone"], "categorical_features": ["clone"],
        "feature_ranges": {"pH": [6.2, 7.1], "glucose_gL": [5.0, 35.0]}, "proposals": [],
        "strain_summary": {"column": "clone", "levels": [], "fixed_level": "A"},
        "run": {"anonymize": True},
    }
    cid = _seeded(store, analysis=blob)
    d = _post(tc, cid, {"n": 4}).json()["design"]
    assert d["anonymized"] is True
    assert all(row["recipe"]["clone"].startswith("level_") for row in d["rows"])
    assert "sample" not in json.dumps(d["template"])
    r = _post(tc, cid, {"n": 4, "format": "csv"})
    assert r.status_code == 409 and r.json()["code"] == "anonymized_template_unavailable"


def test_trusted_model_says_the_plan_ignores_it(env):
    tc, store = env
    blob = {
        "target": "titer", "features": ["pH", "glucose_gL", "clone"], "categorical_features": ["clone"],
        "feature_ranges": {"pH": [6.2, 7.1], "glucose_gL": [5.0, 35.0]}, "proposals": [],
        "proposal_trust": {"status": "trusted"}, "verdict": {"state": "reliable", "reasons": []},
        "run": {"anonymize": False},
    }
    cid = _seeded(store, analysis=blob)
    d = _post(tc, cid, {"n": 3}).json()["design"]
    assert d["model_not_used"]["code"] == "model_trusted_not_used"
    assert d["label"] == "exploration design, model not used"


def test_route_is_in_the_scope_manifest():
    entries = json.loads((portal_module.Path(__file__).resolve().parents[1] / "docs" / "route-scopes.json")
                         .read_text())
    hit = [e for e in entries if e["path"] == "/api/campaigns/{campaign_id}/design"]
    assert hit == [{"method": "POST", "path": "/api/campaigns/{campaign_id}/design",
                    "scopes": ["write"], "auth": "required"}]


def test_a_design_row_started_as_a_run_carries_no_engine_prediction(env):
    tc, store = env
    cid = _seeded(store)
    row = _post(tc, cid, {"n": 1}).json()["design"]["rows"][0]
    r = tc.post(f"/api/campaigns/{cid}/runs", headers=H("a-write"), json={"recipes": [
        {"recipe": row["recipe"], "mode": "exploration_design", "reason": row["label"]}]})
    assert r.status_code == 200, r.text
    run = r.json()["started"][0]
    assert run["pred"] is None and run["std"] is None and run["pred_source"] == "caller"
    assert run["prediction_at_start"]["available"] is False


@pytest.mark.slow
def test_the_csv_template_uploads_back_once_the_target_is_filled(env):
    tc, store = env
    cid = _empty(store)
    text = _post(tc, cid, {**COLD_BODY, "format": "csv"}).text
    df = pd.read_csv(io.StringIO(text))
    rng = np.random.default_rng(0)
    df["titer_g_L"] = (2.0 - (df["pH"] - 6.9) ** 2 + 0.02 * df["glucose_gL"] + rng.normal(0, 0.05, len(df))).round(3)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    r = tc.post(f"/api/campaigns/{cid}/upload", files={"file": ("plan.csv", buf.getvalue(), "text/csv")},
                data={"target": "titer_g_L"}, headers=H("a-write"))
    assert r.status_code == 200, r.text
    prov = {p["name"]: p["status"] for p in r.json()["provenance"]}
    assert prov["run_id"] == "dropped_id" and prov["titer_g_L"] == "target"
    assert prov["pH"] == "kept_feature" and prov["temperature_C"] == "kept_feature"


def test_anonymized_analysis_aliases_the_exploration_design():
    from kalos.portal.analysis_parts.anonymize import _anonymize_result
    from kalos.portal.design_plan import analysis_design

    X = np.array([[6.5, 0.0], [7.0, 1.0], [6.8, 0.0]])
    block = analysis_design(
        names=["pH", "clone"], categorical={"clone": ["CL-SECRET-1", "CL-SECRET-2"]},
        bounds=np.array([[6.0, 0.0], [7.5, 1.0]]), X=X, setup=None, held={}, n=4,
        analysis_view={"proposal_trust": {"status": "exploration_untrusted"}}, n_measured=3,
    )
    assert block["available"] is True and "CL-SECRET-1" in json.dumps(block)
    out = _anonymize_result({"exploration_design": block, "provenance": []},
                            categorical_levels={"clone": ["CL-SECRET-1", "CL-SECRET-2"]},
                            label_columns=["clone"])
    text = json.dumps(out["exploration_design"])
    assert "CL-SECRET" not in text and "level_" in text
