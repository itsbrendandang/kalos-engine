"""`GET/PUT /api/campaigns/{id}/setup` (kalos.portal.setup_api) and the
torch-free validation in kalos.portal.setup_settings.

No GP is fit here: campaigns are seeded with a hand-written analysis blob that
carries exactly the fields the setup validation reads (`target`, `features`,
`categorical_features`, `feature_ranges`, optional `setup.hash`).
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from kalos.portal import app as portal_module  # noqa: E402
from kalos.portal import campaign as campaign_module  # noqa: E402
from kalos.portal.app import app  # noqa: E402
from kalos.portal.campaign import CampaignStore, get_campaign_store  # noqa: E402
from kalos.portal.setup_settings import (  # noqa: E402
    DEFAULT_SETUP_HASH,
    SetupInvalid,
    SetupPlan,
    analysis_setup,
    anonymize_setup_block,
    check_feasible,
    normalize_setup,
    setup_for_analysis,
    setup_hash,
)

SECRET_FEATURE = "ZEUS-secret-feature"


def _sha(t: str) -> str:
    return hashlib.sha256(t.encode()).hexdigest()


TOKENS = [
    {"token_sha256": _sha("a-admin"), "subject": "alice@acme", "tenant": "acme",
     "scopes": ["read", "write", "admin"]},
    {"token_sha256": _sha("a-write"), "subject": "bob@acme", "tenant": "acme",
     "scopes": ["read", "write"]},
    {"token_sha256": _sha("a-read"), "subject": "carol@acme", "tenant": "acme",
     "scopes": ["read"]},
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


def _df() -> pd.DataFrame:
    return pd.DataFrame({
        "a": [0.0, 2.0, 4.0, 6.0, 8.0, 10.0],
        "b": [1.0, 3.0, 5.0, 2.0, 4.0, 6.0],
        "titer": [5.0, 3.0, 9.0, 1.5, 7.0, 4.0],
    })


def _analysis(*, cats: bool = False, setup_hash_value: str | None = None) -> dict:
    out = {
        "target": "titer",
        "features": ["a", "b", "medium"] if cats else ["a", "b"],
        "categorical_features": ["medium"] if cats else [],
        "feature_ranges": {"a": [0.0, 10.0], "b": [1.0, 6.0]},
        "proposals": [],
    }
    if setup_hash_value is not None:
        out["setup"] = {"hash": setup_hash_value, "status": "applied"}
    return out


def _campaign(store: CampaignStore, tenant: str = "acme", *, analysis: dict | None = None,
              seed: bool = True) -> str:
    cid = store.create_campaign(tenant=tenant, name="c", created_by="t")["id"]
    if seed:
        store.seed(_df(), "titer", ["a", "b"], tenant=tenant, campaign_id=cid,
                   analysis=analysis if analysis is not None else _analysis())
    return cid


GOOD = {
    "goal": "minimize",
    "bounds": {"a": {"lo": 1, "hi": 12.5}},
    "linear_constraints": [{"coefficients": {"a": 1, "b": 1}, "op": "<=", "rhs": 12}],
    "target_unit": " g/L ",
}


# --- round trip ------------------------------------------------------------------ #


def test_get_default_then_put_and_read_back(env):
    tc, store = env
    cid = _campaign(store)
    r = tc.get(f"/api/campaigns/{cid}/setup", headers=H("a-read"))
    assert r.status_code == 200
    body = r.json()
    assert body["source"] == "default"
    assert body["setup"] == {"goal": "maximize", "bounds": {}, "linear_constraints": [],
                             "target_unit": None}
    assert body["hash"] == DEFAULT_SETUP_HASH
    # The analysis has no setup block: it was computed with the default setup.
    assert body["settings_in_sync"] is True

    r = tc.put(f"/api/campaigns/{cid}/setup", json=GOOD, headers=H("a-write"))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["source"] == "stored"
    assert body["setup"]["goal"] == "minimize"
    assert body["setup"]["target_unit"] == "g/L"
    assert body["setup"]["bounds"] == {"a": {"lo": 1.0, "hi": 12.5}}
    assert body["updated_by"] == "bob@acme"
    assert body["hash"] == setup_hash(normalize_setup(GOOD))
    assert body["settings_in_sync"] is False  # goals changed; re-analyze to apply
    # hi=12.5 reaches past the observed 10.0: allowed, flagged, clipped for proposals.
    rep = body["bounds_report"]["a"]
    assert rep["extrapolates"] is True
    assert rep["observed"] == [0.0, 10.0]
    assert rep["effective"] == [1.0, 10.0]
    assert tc.get(f"/api/campaigns/{cid}/setup", headers=H("a-read")).json() == body
    # audit row written with the change (ids and counts only)
    rows = [e for e in store.audit_entries(tenant="acme") if e["action"] == "campaign.setup"]
    assert rows and rows[0]["outcome"] == "ok"
    assert "g/L" not in json.dumps(rows)


def test_settings_in_sync_follows_the_analysis_hash(env):
    tc, store = env
    cid = _campaign(store)
    tc.put(f"/api/campaigns/{cid}/setup", json=GOOD, headers=H("a-write"))
    h = setup_hash(normalize_setup(GOOD))
    store.save_analysis(_analysis(setup_hash_value=h), "x", tenant="acme", campaign_id=cid)
    body = tc.get(f"/api/campaigns/{cid}/setup", headers=H("a-read")).json()
    assert body["analysis_setup_hash"] == h
    assert body["settings_in_sync"] is True
    tc.put(f"/api/campaigns/{cid}/setup", json={**GOOD, "goal": "maximize"}, headers=H("a-write"))
    assert tc.get(f"/api/campaigns/{cid}/setup", headers=H("a-read")).json()["settings_in_sync"] is False


def test_no_analysis_means_sync_is_null_and_only_feasibility_is_checked(env):
    tc, store = env
    cid = _campaign(store, seed=False)
    r = tc.put(f"/api/campaigns/{cid}/setup",
               json={"bounds": {"anything": {"lo": 0, "hi": 1}}}, headers=H("a-write"))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["settings_in_sync"] is None
    assert body["bounds_report"]["anything"]["extrapolates"] is None
    bad = {"linear_constraints": [
        {"coefficients": {"x": 1}, "op": ">=", "rhs": 5},
        {"coefficients": {"x": 1}, "op": "<=", "rhs": 4},
    ]}
    r = tc.put(f"/api/campaigns/{cid}/setup", json=bad, headers=H("a-write"))
    assert r.status_code == 400 and r.json()["code"] == "infeasible_constraints"


# --- validation --------------------------------------------------------------------- #


@pytest.mark.parametrize("body,code", [
    ({"goal": "maximise"}, "invalid_body"),
    ({"bounds": {"a": {"lo": "1", "hi": 2}}}, "invalid_body"),  # strict: no numeric strings
    ({"bounds": {"a": {"lo": True, "hi": 2}}}, "invalid_body"),
    ({"bounds": {"a": {"lo": 1, "hi": 2, "x": 3}}}, "invalid_body"),
    ({"extra": 1}, "invalid_body"),
    ({"bounds": {f"f{i}": {"lo": 0, "hi": 1} for i in range(65)}}, "invalid_body"),
    ({"linear_constraints": [{"coefficients": {"a": 1}, "op": "<=", "rhs": 5}] * 17}, "invalid_body"),
    ({"linear_constraints": [{"coefficients": {f"f{i}": 1 for i in range(65)}, "op": "<=",
                              "rhs": 1}]}, "invalid_body"),
    ({"linear_constraints": [{"coefficients": {}, "op": "<=", "rhs": 1}]}, "invalid_body"),
    ({"linear_constraints": [{"coefficients": {"a": 1}, "op": "==", "rhs": 1}]}, "invalid_body"),
    ({"bounds": {"a": {"lo": 2, "hi": 2}}}, "invalid_bounds"),
    ({"bounds": {"a": {"lo": 3, "hi": 2}}}, "invalid_bounds"),
    ({"bounds": {"a": {"lo": -1e300, "hi": 2}}}, "invalid_setup"),
    ({"linear_constraints": [{"coefficients": {"a": 0}, "op": "<=", "rhs": 1}]}, "invalid_constraint"),
    ({"target_unit": "x" * 33}, "invalid_setup"),
    ({"target_unit": "g/L\u0000"}, "invalid_setup"),
    ({"target_unit": "g/\u200bL"}, "invalid_setup"),  # format char (Cf)
    ({"target_unit": "g/L\u202e"}, "invalid_setup"),  # bidi override (Cf)
    ({"target_unit": "\u2066mg\u2069"}, "invalid_setup"),  # bidi isolates (Cf)
    ({"target_unit": "\ufeffg/L"}, "invalid_setup"),  # BOM / zero width no-break (Cf)
    ({"bounds": {"titer": {"lo": 0, "hi": 1}}}, "target_not_a_feature"),
    ({"bounds": {SECRET_FEATURE: {"lo": 0, "hi": 1}}}, "unknown_feature"),
    ({"linear_constraints": [{"coefficients": {SECRET_FEATURE: 1}, "op": "<=", "rhs": 1}]},
     "unknown_feature"),
    ({"bounds": {"a": {"lo": 20, "hi": 30}}}, "bounds_outside_data"),
    ({"bounds": {"a": {"lo": 0, "hi": 2}}, "linear_constraints": [
        {"coefficients": {"a": 1, "b": 1}, "op": ">=", "rhs": 50}]}, "infeasible_constraints"),
    ({"linear_constraints": [
        {"coefficients": {"a": 1, "b": 1}, "op": ">=", "rhs": 5},
        {"coefficients": {"a": 1, "b": 1}, "op": "<=", "rhs": 5}]}, "degenerate_constraints"),
])
def test_put_refuses_invalid_setups(env, body, code):
    tc, store = env
    cid = _campaign(store)
    r = tc.put(f"/api/campaigns/{cid}/setup", json=body, headers=H("a-write"))
    assert r.status_code == 400, r.text
    out = r.json()
    assert out["code"] == code
    assert "error" in out
    # caller input is never echoed
    assert SECRET_FEATURE not in r.text
    # nothing stored
    assert tc.get(f"/api/campaigns/{cid}/setup", headers=H("a-read")).json()["source"] == "default"


def test_non_finite_numbers_are_refused(env):
    tc, store = env
    cid = _campaign(store)
    raw = '{"bounds": {"a": {"lo": 0, "hi": Infinity}}}'
    r = tc.put(f"/api/campaigns/{cid}/setup", content=raw,
               headers={**H("a-write"), "Content-Type": "application/json"})
    assert r.status_code == 400


def test_categoricals_refuse_constraints_and_categorical_bounds(env):
    tc, store = env
    cid = _campaign(store, analysis=_analysis(cats=True))
    r = tc.put(f"/api/campaigns/{cid}/setup", json={"linear_constraints": [
        {"coefficients": {"a": 1}, "op": "<=", "rhs": 5}]}, headers=H("a-write"))
    assert r.status_code == 400 and r.json()["code"] == "constraints_with_categoricals"
    assert r.json()["continuous_features"] == ["a", "b"]
    r = tc.put(f"/api/campaigns/{cid}/setup", json={"bounds": {"medium": {"lo": 0, "hi": 1}}},
               headers=H("a-write"))
    assert r.status_code == 400 and r.json()["code"] == "categorical_feature"
    r = tc.put(f"/api/campaigns/{cid}/setup", json={"linear_constraints": [
        {"coefficients": {"medium": 1}, "op": "<=", "rhs": 5}]}, headers=H("a-write"))
    assert r.status_code == 400 and r.json()["code"] == "categorical_feature"
    # bounds on continuous features are still fine with categoricals present
    r = tc.put(f"/api/campaigns/{cid}/setup", json={"bounds": {"a": {"lo": 1, "hi": 5}}},
               headers=H("a-write"))
    assert r.status_code == 200


# --- auth, tenancy, archive -------------------------------------------------------- #


def test_scopes(env):
    tc, store = env
    cid = _campaign(store)
    assert tc.get(f"/api/campaigns/{cid}/setup", headers=H("a-read")).status_code == 200
    r = tc.put(f"/api/campaigns/{cid}/setup", json=GOOD, headers=H("a-read"))
    assert r.status_code == 403
    # scope is checked before the id: a read token on a bad id is still 403
    assert tc.put("/api/campaigns/cmp_nope/setup", json=GOOD, headers=H("a-read")).status_code == 403
    assert tc.get(f"/api/campaigns/{cid}/setup").status_code == 401


def test_cross_tenant_ids_are_byte_identical_to_missing(env):
    tc, store = env
    b_cid = _campaign(store, tenant="globex")
    tc.put(f"/api/campaigns/{b_cid}/setup", json=GOOD, headers=H("b-admin"))
    before = tc.get(f"/api/campaigns/{b_cid}/setup", headers=H("b-admin")).json()
    missing = "cmp_" + "0" * 32
    for method, kw in (("GET", {}), ("PUT", {"json": {"goal": "maximize"}})):
        foreign = tc.request(method, f"/api/campaigns/{b_cid}/setup", headers=H("a-admin"), **kw)
        miss = tc.request(method, f"/api/campaigns/{missing}/setup", headers=H("a-admin"), **kw)
        bad = tc.request(method, "/api/campaigns/cmp_nope/setup", headers=H("a-admin"), **kw)
        assert foreign.status_code == 404
        assert foreign.json() == {"error": "campaign not found", "code": "not_found"}
        assert (foreign.status_code, foreign.content) == (miss.status_code, miss.content) \
            == (bad.status_code, bad.content)
        assert sorted(foreign.headers.items()) == sorted(miss.headers.items())
    assert tc.get(f"/api/campaigns/{b_cid}/setup", headers=H("b-admin")).json() == before
    # globex's setup does not leak into acme's default view of its own campaigns
    a_cid = _campaign(store)
    assert tc.get(f"/api/campaigns/{a_cid}/setup", headers=H("a-read")).json()["source"] == "default"


def test_archived_campaign_put_is_409_get_still_works(env):
    tc, store = env
    cid = _campaign(store)
    tc.put(f"/api/campaigns/{cid}/setup", json=GOOD, headers=H("a-write"))
    assert tc.patch(f"/api/campaigns/{cid}", json={"archived": True},
                    headers=H("a-admin")).status_code == 200
    r = tc.put(f"/api/campaigns/{cid}/setup", json={"goal": "maximize"}, headers=H("a-write"))
    assert r.status_code == 409 and r.json()["code"] == "campaign_archived"
    r = tc.get(f"/api/campaigns/{cid}/setup", headers=H("a-read"))
    assert r.status_code == 200 and r.json()["setup"]["goal"] == "minimize"


# --- campaign summary best is direction-aware -------------------------------------- #


def test_campaign_summary_best_follows_the_goal(env):
    tc, store = env
    cid = _campaign(store)
    s = tc.get(f"/api/campaigns/{cid}", headers=H("a-read")).json()["summary"]
    assert s["best"] == 9.0 and s["goal"] == "maximize"
    tc.put(f"/api/campaigns/{cid}/setup", json={"goal": "minimize"}, headers=H("a-write"))
    s = tc.get(f"/api/campaigns/{cid}", headers=H("a-read")).json()["summary"]
    assert s["best"] == 1.5 and s["goal"] == "minimize"
    assert s["history"][0]["best"] == 1.5
    # the legacy default-campaign view is unaffected for another campaign
    other = _campaign(store)
    assert tc.get(f"/api/campaigns/{other}", headers=H("a-read")).json()["summary"]["best"] == 9.0


def test_minimize_history_recomputes_each_round(tmp_path):
    from kalos.portal.campaign_settings import put_setting

    store = CampaignStore(tmp_path)
    cid = store.create_campaign(tenant="t", name="c", created_by="x")["id"]
    store.seed(_df(), "titer", ["a", "b"], tenant="t", campaign_id=cid)
    run = store.start([{"recipe": {"a": 1.0, "b": 1.0}}], tenant="t", campaign_id=cid)[0]
    store.set_result(run["id"], 0.5, tenant="t", campaign_id=cid)
    gen = store.generation_of(tenant="t", campaign_id=cid)
    store.commit_fold(gen, tenant="t", campaign_id=cid)
    put_setting(store, tenant="t", campaign_id=cid, kind="setup",
                value=normalize_setup({"goal": "minimize"}), updated_by="x")
    s = store.summary(tenant="t", campaign_id=cid)
    assert [p["best"] for p in s["history"]] == [1.5, 0.5]
    assert s["best"] == 0.5


# --- module-level pieces ---------------------------------------------------------- #


def test_setup_for_analysis_mapping(tmp_path):
    from kalos.portal.campaign_settings import put_setting

    store = CampaignStore(tmp_path)
    cid = store.create_campaign(tenant="t", name="c", created_by="x")["id"]
    assert setup_for_analysis(store, tenant="t", campaign_id=cid) is None
    doc = normalize_setup(GOOD)
    put_setting(store, tenant="t", campaign_id=cid, kind="setup", value=doc, updated_by="x")
    m = setup_for_analysis(store, tenant="t", campaign_id=cid)
    assert m == analysis_setup(doc)
    assert m["user_target_unit"] == "g/L" and m["hash"] == setup_hash(doc)
    # a stored doc that no longer normalizes fails closed
    put_setting(store, tenant="t", campaign_id=cid, kind="setup",
                value={"goal": "sideways"}, updated_by="x")
    assert setup_for_analysis(store, tenant="t", campaign_id=cid) == {"unreadable": True, "hash": None}


def test_hash_is_canonical():
    a = normalize_setup({"bounds": {"b": {"lo": 0, "hi": 1}, "a": {"lo": 0, "hi": 1.0}}})
    b = normalize_setup({"bounds": {"a": {"lo": 0.0, "hi": 1}, "b": {"lo": 0, "hi": 1}}})
    assert setup_hash(a) == setup_hash(b)
    assert setup_hash(normalize_setup({})) == DEFAULT_SETUP_HASH
    assert setup_hash(normalize_setup({"target_unit": "  "})) == DEFAULT_SETUP_HASH


def test_check_feasible():
    cons = [{"coefficients": {"a": 1.0, "b": 1.0}, "op": "<=", "rhs": 3.0}]
    check_feasible({"a": (0.0, 2.0), "b": (0.0, 2.0)}, cons)
    with pytest.raises(SetupInvalid) as e:
        check_feasible({"a": (2.0, 3.0), "b": (2.0, 3.0)}, cons)
    assert e.value.code == "infeasible_constraints"
    check_feasible({"a": (None, None), "b": (None, None)}, cons)  # free features


def test_post_check_drops_violators():
    plan = SetupPlan(goal="maximize", hash="h", bounds=np.array([[0.0, 0.0], [1.0, 1.0]]),
                     linear_constraints=[([0, 1], [-1.0, -1.0], -1.0)])  # a + b <= 1
    batch = np.array([[0.2, 0.3], [0.9, 0.9], [1.5, 0.0], [0.5, 0.5]])
    kept = plan.post_check(batch)
    assert kept.tolist() == [[0.2, 0.3], [0.5, 0.5]]
    assert plan.n_dropped == 2
    assert plan.block()["warnings"][0]["code"] == "proposals_dropped"


def test_anonymize_setup_block_aliases_feature_names():
    block = {"bounds_applied": {"operator_id": {}}, "constraints_applied": [
        {"coefficients": {"operator_id": 1.0}, "op": "<=", "rhs": 1.0}],
        "warnings": [{"code": "x", "message": "m", "feature": "operator_id"}, {"code": "y"}]}
    out = anonymize_setup_block(block, lambda n: "col_x" if n == "operator_id" else n)
    assert "operator_id" not in json.dumps(out)


def test_with_campaign_setup_binds_only_when_a_setup_is_stored(tmp_path):
    import functools

    from kalos.portal.campaign_settings import put_setting
    from kalos.portal.setup_settings import with_campaign_setup

    store = CampaignStore(tmp_path)
    cid = store.create_campaign(tenant="t", name="c", created_by="x")["id"]

    def stub(*a, **k):
        return k

    assert with_campaign_setup(stub, store, tenant="t", campaign_id=cid) is stub
    put_setting(store, tenant="t", campaign_id=cid, kind="setup",
                value=normalize_setup({"goal": "minimize"}), updated_by="x")
    bound = with_campaign_setup(stub, store, tenant="t", campaign_id=cid)
    assert isinstance(bound, functools.partial)
    assert bound()["setup"]["goal"] == "minimize"
