"""VIZ_SPEC R5: the engine's prediction + band is recorded when a run is started
(kalos/portal/run_prediction.py, CampaignStore.start), and returned with the
summary and result responses so predicted vs measured is auditable."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from kalos.portal import app as portal_module
from kalos.portal import busy
from kalos.portal import campaign as campaign_module
from kalos.portal.app import app
from kalos.portal.campaign import CampaignStore, get_campaign_store
from kalos.portal.run_prediction import (
    NO_ANALYSIS_REASON,
    NO_BAND_REASON,
    NOT_A_PROPOSAL_REASON,
    recipes_match,
    residual_fields,
)

TARGET = "lipase_titer"
PROV = {"provenance_sha256": "a" * 64, "input_sha256": "b" * 64, "config_sha256": "c" * 64,
        "engine_version": "test", "seed": 1, "timestamp": 1.0}


def _df(n: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    m = rng.uniform(0, 4, n)
    ph = rng.uniform(5, 7, n)
    return pd.DataFrame({
        "Methanol": m.round(3), "pH": ph.round(2),
        "strain": ["A", "B"] * (n // 2),
        TARGET: (1.5 * m - 0.8 * (ph - 6) ** 2).round(3),
    })


def _analysis(*, conformal: bool = True, extra: dict | None = None) -> dict:
    p0 = {"pred": 4.2, "std": 0.3, "mode": "exploit", "reason": "x",
          "recipe": {"Methanol": 2.5, "pH": 6.1, "strain": "A"}}
    p1 = {"pred": 3.1, "std": 0.6, "mode": "explore", "reason": "y",
          "recipe": {"Methanol": 1.234, "pH": 5.5, "strain": "B"}}
    if extra:
        p0.update(extra)
    out: dict = {"target": TARGET, "proposals": [p0, p1], "proposal_provenance": PROV,
                 "big_unrelated_block": list(range(500))}
    if conformal:
        out.update(conformal_q=0.5, conformal_coverage=0.8)
    return out


def _seed(store: CampaignStore, tenant: str = "acme", cid: str | None = None, **kw) -> None:
    store.seed(_df(), TARGET, ["Methanol", "pH", "strain"], tenant=tenant,
               campaign_id=cid, analysis=_analysis(**kw), dataset="d.csv")


# --- store level ------------------------------------------------------------ #


@pytest.fixture
def store(tmp_path):
    s = CampaignStore(tmp_path)
    yield s
    s.close()


def test_matched_recipe_stores_engine_band(store):
    _seed(store)
    run = store.start([{"recipe": {"Methanol": 2.5, "pH": 6.1, "strain": "A"},
                        "pred": 99.0, "std": 9.0}], tenant="acme")[0]
    pas = run["prediction_at_start"]
    assert pas["available"] is True and pas["basis"] == "predicted"
    assert pas["pred"] == 4.2 and pas["std"] == 0.3 and pas["std_method"] == "gp_posterior"
    assert pas["interval"] == {"lo": 3.7, "hi": 4.7, "half_width": 0.5, "coverage": 0.8,
                               "method": "conformal_global"}
    assert pas["interval_unavailable_reason"] is None
    assert pas["proposal_index"] == 0
    assert pas["analysis_provenance"] == PROV
    assert pas["analysis_generation"] == store.load_analysis(tenant="acme")["campaign_generation"]
    assert isinstance(pas["recorded_at"], float)
    # The caller's pred was not the engine's: the run carries the engine's values.
    assert run["pred_source"] == "engine" and run["pred"] == 4.2 and run["std"] == 0.3
    assert run["residual"] is None and run["inside_band"] is None
    # persisted, small (no analysis copied in)
    stored = store.get(tenant="acme")["pending"][0]
    assert stored["prediction_at_start"] == pas
    assert len(json.dumps(pas)) < 1024


def test_float_tolerance_is_the_proposal_precision(store):
    _seed(store)
    near = store.start([{"recipe": {"Methanol": 1.2341, "pH": 5.5, "strain": "B"}}],
                       tenant="acme")[0]
    assert near["prediction_at_start"]["proposal_index"] == 1
    far = store.start([{"recipe": {"Methanol": 1.236, "pH": 5.5, "strain": "B"}}],
                      tenant="acme")[0]
    assert far["prediction_at_start"]["available"] is False
    assert recipes_match({"a": 5}, {"a": 5.0})
    assert not recipes_match({"a": "5"}, {"a": 5})
    assert not recipes_match({"a": 1}, {"a": 1, "b": 2})


def test_edited_recipe_is_unavailable_and_caller_pred_not_promoted(store):
    _seed(store)
    run = store.start([{"recipe": {"Methanol": 2.5, "pH": 6.1, "strain": "B"},
                        "pred": 4.2, "std": 0.3}], tenant="acme")[0]
    assert run["prediction_at_start"] == {
        "available": False, "reason": NOT_A_PROPOSAL_REASON,
        "recorded_at": run["prediction_at_start"]["recorded_at"],
    }
    assert run["pred_source"] == "caller"
    assert run["pred"] == 4.2  # kept for compatibility, labelled as the caller's
    store.set_result(run["id"], 4.25, tenant="acme")
    pending = store.summary(tenant="acme")["pending"][0]
    assert pending["residual"] is None and pending["inside_band"] is None
    assert pending["pred_source"] == "caller"


def test_no_published_analysis(store):
    store.seed(_df(), TARGET, ["Methanol", "pH", "strain"], tenant="acme")
    run = store.start([{"recipe": {"Methanol": 2.5, "pH": 6.1, "strain": "A"}}],
                      tenant="acme")[0]
    assert run["prediction_at_start"]["available"] is False
    assert run["prediction_at_start"]["reason"] == NO_ANALYSIS_REASON


def test_no_conformal_band_records_reason(store):
    _seed(store, conformal=False)
    pas = store.start([{"recipe": {"Methanol": 2.5, "pH": 6.1, "strain": "A"}}],
                      tenant="acme")[0]["prediction_at_start"]
    assert pas["available"] is True and pas["interval"] is None
    assert pas["interval_unavailable_reason"] == NO_BAND_REASON


def test_per_proposal_intervals_copied_verbatim_when_small(store):
    iv = [{"lo": 3.9, "hi": 4.5, "coverage": 0.8, "method": "conformal_normalized"}]
    gp = [{"lo": 3.6, "hi": 4.8, "coverage": 0.95, "method": "gp_posterior"}]
    _seed(store, extra={"intervals": iv, "gp_intervals": gp})
    pas = store.start([{"recipe": {"Methanol": 2.5, "pH": 6.1, "strain": "A"}}],
                      tenant="acme")[0]["prediction_at_start"]
    assert pas["intervals"] == iv and pas["gp_intervals"] == gp


def test_oversized_proposal_block_is_omitted_not_truncated(store):
    _seed(store, extra={"intervals": [{"x": "y" * 5000}]})
    pas = store.start([{"recipe": {"Methanol": 2.5, "pH": 6.1, "strain": "A"}}],
                      tenant="acme")[0]["prediction_at_start"]
    assert "intervals" not in pas and pas["intervals_omitted"]


def test_residual_and_inside_band_after_result_and_correction(store):
    _seed(store)
    run = store.start([{"recipe": {"Methanol": 2.5, "pH": 6.1, "strain": "A"}}],
                      tenant="acme")[0]
    out = store.set_result(run["id"], 4.5, tenant="acme")
    assert out["residual"] == pytest.approx(0.3) and out["residual_basis"] == "derived"
    assert out["inside_band"] is True
    assert out["prediction_at_start"]["pred"] == 4.2
    out = store.set_result(run["id"], 5.0, tenant="acme", reason="re-assayed")
    assert out["residual"] == pytest.approx(0.8) and out["inside_band"] is False
    pending = store.summary(tenant="acme")["pending"][0]
    assert pending["residual"] == pytest.approx(0.8) and pending["inside_band"] is False


def test_legacy_run_without_block_reads_null_and_is_not_backfilled(store):
    _seed(store)
    store.start([{"recipe": {"Methanol": 2.5, "pH": 6.1, "strain": "A"}}], tenant="acme")
    # Simulate a run written before R5: drop the new keys from the stored state.
    with store._txn(write=True) as conn:
        row = store._resolve(conn, "acme", None, create=False)
        state = row.state()
        for run in state["pending"]:
            run.pop("prediction_at_start")
            run.pop("pred_source")
        store._cas_write(conn, "acme", row, state, row.generation)
    run_id = store.get(tenant="acme")["pending"][0]["id"]
    out = store.set_result(run_id, 4.5, tenant="acme")
    assert out["prediction_at_start"] is None and out["pred_source"] is None
    assert out["residual"] is None and out["inside_band"] is None
    stored = store.get(tenant="acme")["pending"][0]
    assert "prediction_at_start" not in stored  # never backfilled


def test_fold_does_not_add_columns(store):
    _seed(store)
    run = store.start([{"recipe": {"Methanol": 2.5, "pH": 6.1, "strain": "A"}}],
                      tenant="acme")[0]
    store.set_result(run["id"], 4.5, tenant="acme")
    df, _target, gen = store.plan_fold(tenant="acme")
    assert set(df.columns) == {"Methanol", "pH", "strain", TARGET}
    state = store.commit_fold(gen, tenant="acme")
    for row in state["base_rows"]:
        assert set(row) == {"Methanol", "pH", "strain", TARGET}


def test_residual_fields_unit():
    assert residual_fields({"result": 1.0}) == {
        "residual": None, "residual_basis": None, "inside_band": None}
    blk = {"available": True, "pred": 1.0, "interval": None}
    assert residual_fields({"result": 1.5, "prediction_at_start": blk})["inside_band"] is None
    assert residual_fields({"result": None, "prediction_at_start": blk})["residual"] is None


# --- HTTP: campaign-scoped + legacy routes, cross-tenant -------------------- #


def _sha(t: str) -> str:
    return hashlib.sha256(t.encode()).hexdigest()


TOKENS = [
    {"token_sha256": _sha("a-admin"), "subject": "alice@acme", "tenant": "acme",
     "scopes": ["read", "write", "admin"]},
    {"token_sha256": _sha("b-admin"), "subject": "gina@globex", "tenant": "globex",
     "scopes": ["read", "write", "admin"]},
]
A = {"Authorization": "Bearer a-admin"}
B = {"Authorization": "Bearer b-admin"}
MATCHED = {"recipe": {"Methanol": 2.5, "pH": 6.1, "strain": "A"}, "pred": 4.2, "std": 0.3,
           "mode": "exploit", "reason": "x"}


@pytest.fixture
def env(tmp_path, monkeypatch):
    s = CampaignStore(tmp_path)
    monkeypatch.setenv("KALOS_AUTH_TOKENS", json.dumps(TOKENS))
    monkeypatch.delenv("KALOS_AUTH_TOKENS_FILE", raising=False)
    monkeypatch.setattr(portal_module, "_STATE_DIR", tmp_path)
    monkeypatch.setattr(portal_module, "_LATEST_DIR", tmp_path / "latest")
    monkeypatch.setattr(campaign_module, "_STORE", s)
    app.dependency_overrides[get_campaign_store] = lambda: s
    busy.reset_limiter(total=2, per_tenant=1)
    try:
        yield TestClient(app), s
    finally:
        app.dependency_overrides.pop(get_campaign_store, None)
        s.close()


def test_campaign_routes_return_prediction_at_start(env):
    client, s = env
    cid = s.create_campaign(tenant="acme", name="c")["id"]
    _seed(s, cid=cid)
    r = client.post(f"/api/campaigns/{cid}/runs", json={"recipes": [MATCHED]}, headers=A)
    assert r.status_code == 200, r.text
    run = r.json()["started"][0]
    assert run["prediction_at_start"]["available"] is True and run["pred_source"] == "engine"
    r = client.post(f"/api/campaigns/{cid}/runs/{run['id']}/result", json={"value": 4.0},
                    headers=A)
    assert r.status_code == 200, r.text
    assert r.json()["residual"] == pytest.approx(-0.2) and r.json()["inside_band"] is True
    runs = client.get(f"/api/campaigns/{cid}", headers=A).json()["summary"]["pending"]
    assert runs[0]["prediction_at_start"]["interval"]["method"] == "conformal_global"
    assert runs[0]["inside_band"] is True


def test_legacy_routes_return_prediction_at_start(env):
    client, s = env
    _seed(s)
    r = client.post("/api/campaign/start", json={"recipes": [MATCHED]}, headers=A)
    assert r.status_code == 200, r.text
    run = r.json()["started"][0]
    assert run["prediction_at_start"]["proposal_index"] == 0
    r = client.post("/api/campaign/result", json={"id": run["id"], "value": 5.0}, headers=A)
    assert r.status_code == 200, r.text
    assert r.json()["inside_band"] is False and r.json()["residual"] == pytest.approx(0.8)
    body = client.get("/api/campaign", headers=A).json()
    assert body["pending"][0]["prediction_at_start"]["pred"] == 4.2
    assert body["pending"][0]["residual"] == pytest.approx(0.8)


def test_cross_tenant_prediction_uses_own_analysis_only(env):
    client, s = env
    # globex publishes an analysis proposing acme's recipe; acme has none.
    gcid = s.create_campaign(tenant="globex", name="g")["id"]
    _seed(s, tenant="globex", cid=gcid)
    acid = s.create_campaign(tenant="acme", name="a")["id"]
    s.seed(_df(), TARGET, ["Methanol", "pH", "strain"], tenant="acme", campaign_id=acid)
    r = client.post(f"/api/campaigns/{acid}/runs", json={"recipes": [MATCHED]}, headers=A)
    assert r.status_code == 200
    pas = r.json()["started"][0]["prediction_at_start"]
    assert pas["available"] is False and pas["reason"] == NO_ANALYSIS_REASON
    # acme cannot read or start against globex's campaign: byte-identical 404.
    foreign = client.get(f"/api/campaigns/{gcid}", headers=A)
    missing = client.get("/api/campaigns/cmp_" + "0" * 32, headers=A)
    assert foreign.status_code == 404 and foreign.content == missing.content
    r = client.post(f"/api/campaigns/{gcid}/runs", json={"recipes": [MATCHED]}, headers=A)
    assert r.status_code == 404
    assert s.get(tenant="globex", campaign_id=gcid)["pending"] == []
    # globex's own run matches globex's analysis.
    r = client.post(f"/api/campaigns/{gcid}/runs", json={"recipes": [MATCHED]}, headers=B)
    assert r.json()["started"][0]["prediction_at_start"]["available"] is True


def test_real_engine_proposal_round_trips(tmp_path):
    """A proposal produced by the real analysis, started verbatim, matches."""
    busy.reset_limiter(total=2, per_tenant=1)
    rng = np.random.default_rng(0)
    m, ph = rng.uniform(0, 4, 24), rng.uniform(5, 7, 24)
    df = pd.DataFrame({"Methanol": m.round(3), "pH": ph.round(2),
                       TARGET: (1.5 * m - 0.8 * (ph - 6) ** 2
                                + rng.normal(0, 0.1, 24)).round(3)})
    s = CampaignStore(tmp_path)
    try:
        result = portal_module._run_uploaded_sync(
            df.to_csv(index=False).encode(), TARGET, False, "p.csv", "", tenant="acme", store=s,
        )
        props = [p for p in result["proposals"] if isinstance(p.get("recipe"), dict)]
        assert props, "the engine produced no proposals with recipes"
        run = s.start([{"recipe": props[-1]["recipe"], "pred": 0.0}], tenant="acme")[0]
        pas = run["prediction_at_start"]
        assert pas["available"] is True and pas["proposal_index"] == len(props) - 1
        assert pas["pred"] == props[-1]["pred"] and run["pred_source"] == "engine"
        if result.get("conformal_q") is not None:
            assert pas["interval"]["half_width"] == result["conformal_q"]
            assert pas["interval"]["coverage"] == result["conformal_coverage"]
    finally:
        s.close()
