"""Report exports (kalos/portal/exports_api.py + kalos/portal/operating_ranges.py).

- row export: original vs folded split (verified fold ordering), unidentifiable
  history, anonymize (same aliases as the analysis, no leak, refusal when the
  mapping cannot be reproduced), CSV formula neutralization, provenance;
- results history: every version incl. corrections, the campaign's audit rows,
  pagination with per-section cursors, tenant scoping;
- operating ranges: every not_computed path, busy slot, body validation, the box
  construction, and (slow) a computed path on a clearly learnable sheet;
- cross-tenant 404 (byte-identical to missing/malformed) and scope on every route.
"""
from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import json
import uuid

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from kalos.portal import app as portal_module
from kalos.portal import busy
from kalos.portal import campaign as campaign_module
from kalos.portal.analysis_parts.anonymize import _anonymize_result
from kalos.portal.app import app
from kalos.portal.campaign import CampaignStore, get_campaign_store
from kalos.portal.campaign_settings import put_setting
from kalos.portal.exports_api import _column_alias, _level_alias
from kalos.portal.operating_ranges import maximal_box


def _sha(t: str) -> str:
    return hashlib.sha256(t.encode()).hexdigest()


TOKENS = [
    {"token_sha256": _sha("a-admin"), "subject": "alice@acme", "tenant": "acme",
     "scopes": ["read", "write", "admin"]},
    {"token_sha256": _sha("a-read"), "subject": "carol@acme", "tenant": "acme",
     "scopes": ["read"]},
    {"token_sha256": _sha("a-writeonly"), "subject": "wally@acme", "tenant": "acme",
     "scopes": ["write"]},
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


@contextlib.contextmanager
def _slots_exhausted(tenant: str = "acme"):
    limiter = busy.limiter()
    held = []
    while (lease := limiter.try_acquire(tenant, yield_to_queue=False)) is not None:
        held.append(lease)
    try:
        yield
    finally:
        for lease in held:
            lease.release()


def _df(n: int = 8, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    m = rng.uniform(0, 4, n)
    ph = rng.uniform(5, 7, n)
    return pd.DataFrame({
        "Methanol": m.round(3), "pH": ph.round(2),
        "lipase_titer": (1.5 * m - 0.8 * (ph - 6) ** 2 + rng.normal(0, 0.1, n)).round(3),
    })


def _plain_analysis(**extra) -> dict:
    return {"target": "lipase_titer", "run": {"anonymize": False, "kind": "upload",
                                              "input_sha256": "ab" * 32},
            "engine_version": "0.1.0", **extra}


def _recipe(m: float, ph: float) -> dict:
    return {"recipe": {"Methanol": m, "pH": ph}, "pred": 1.0, "std": 0.1, "mode": "explore",
            "reason": "x"}


def _seeded(store: CampaignStore, tenant: str = "acme", df: pd.DataFrame | None = None,
            analysis: dict | None = None) -> str:
    cid = store.create_campaign(tenant=tenant, name="c")["id"]
    df = _df() if df is None else df
    store.seed(df, "lipase_titer", ["Methanol", "pH"], tenant=tenant, campaign_id=cid,
               analysis=_plain_analysis() if analysis is None else analysis, dataset="runs.csv")
    return cid


def _fold(store: CampaignStore, cid: str, recipes: list[tuple[float, float, float]],
          tenant: str = "acme") -> None:
    runs = store.start([_recipe(m, p) for m, p, _ in recipes], tenant=tenant, campaign_id=cid)
    for run, (_m, _p, value) in zip(runs, recipes):
        store.set_result(run["id"], value, tenant=tenant, campaign_id=cid, actor="alice@acme")
    gen = store.plan_fold(tenant=tenant, campaign_id=cid)[2]
    store.commit_fold(gen, tenant=tenant, campaign_id=cid,
                      analysis=_plain_analysis(run={"anonymize": False, "kind": "reanalyze"}))


def _csv_rows(text: str) -> list[list[str]]:
    return list(csv.reader(io.StringIO(text)))


# --- (a) row export ------------------------------------------------------------------ #


def test_export_json_splits_original_rows_from_folded_rows(env):
    tc, store = env
    df = _df(8)
    cid = _seeded(store, df=df)
    _fold(store, cid, [(1.0, 6.0, 1.11), (2.0, 6.5, 2.22)])
    _fold(store, cid, [(3.0, 5.5, 3.33)])

    r = tc.get(f"/api/campaigns/{cid}/export/rows?format=json", headers=H("a-read"))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["original_rows_identifiable"] is True and body["original_rows_reason"] is None
    original = body["original_rows"]
    assert original["origin"] == "upload" and original["n"] == 8
    assert original["rows"] == json.loads(df.to_json(orient="records"))
    folded = body["folded_rows"]
    assert folded["origin"] == "measured_run" and folded["n"] == 3
    assert [(f["folded_in_round"], f["values"]) for f in folded["rows"]] == [
        (1, {"Methanol": 1.0, "pH": 6.0, "lipase_titer": 1.11}),
        (1, {"Methanol": 2.0, "pH": 6.5, "lipase_titer": 2.22}),
        (2, {"Methanol": 3.0, "pH": 5.5, "lipase_titer": 3.33}),
    ]
    assert body["unattributed_rows"] is None
    assert body["columns"] == ["Methanol", "pH", "lipase_titer"]
    prov = body["provenance"]
    assert prov["campaign_round"] == 2 and prov["analysis_in_sync"] is True
    assert prov["analysis_run_kind"] == "reanalyze" and prov["export_engine_version"]
    assert body["anonymization"] == {"applied": False}

    r = tc.get(f"/api/campaigns/{cid}/export/rows?format=csv", headers=H("a-read"))
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/csv")
    assert r.headers["x-kalos-content-sha256"] == hashlib.sha256(r.content).hexdigest()
    assert r.headers["x-kalos-original-rows-identifiable"] == "true"
    assert r.headers["content-disposition"] == f'attachment; filename="{cid}-rows.csv"'
    rows = _csv_rows(r.text)
    assert rows[0] == ["kalos_origin", "kalos_folded_in_round", "Methanol", "pH", "lipase_titer"]
    assert [row[:2] for row in rows[1:]] == [["upload", "0"]] * 8 + [
        ["measured_run", "1"], ["measured_run", "1"], ["measured_run", "2"]]
    assert rows[-1][2:] == ["3.0", "5.5", "3.33"]


def test_export_without_consistent_history_does_not_guess(env):
    tc, store = env
    for mutate, why in [
        (lambda s: s.pop("history"), "no recorded fold history"),
        (lambda s: s["history"].pop(0), "does not start at the upload"),
        (lambda s: s["base_rows"].append({"Methanol": 9}), "does not match its row count"),
    ]:
        cid = _seeded(store)
        _fold(store, cid, [(1.0, 6.0, 1.0)])
        with store._txn(write=True) as conn:
            state = json.loads(conn.execute("SELECT state FROM campaigns WHERE id = ?", (cid,)).fetchone()[0])
            mutate(state)
            conn.execute("UPDATE campaigns SET state = ? WHERE id = ?", (json.dumps(state), cid))
        body = tc.get(f"/api/campaigns/{cid}/export/rows", headers=H("a-read")).json()
        assert body["original_rows_identifiable"] is False
        assert why in body["original_rows_reason"]
        assert body["original_rows"] is None and body["folded_rows"] is None
        assert body["unattributed_rows"]["origin"] == "unknown"
        assert body["unattributed_rows"]["n"] == len(state["base_rows"])
        rows = _csv_rows(tc.get(f"/api/campaigns/{cid}/export/rows?format=csv", headers=H("a-read")).text)
        assert {row[0] for row in rows[1:]} == {"unknown"}


def test_export_unseeded_campaign(env):
    tc, store = env
    cid = store.create_campaign(tenant="acme", name="empty")["id"]
    assert tc.get(f"/api/campaigns/{cid}/export/rows", headers=H("a-read")).json() == {
        "campaign_id": cid, "has_data": False}
    r = tc.get(f"/api/campaigns/{cid}/export/rows?format=csv", headers=H("a-read"))
    assert r.status_code == 409 and r.json()["code"] == "no_data"


def test_export_rejects_unknown_format(env):
    tc, store = env
    cid = _seeded(store)
    r = tc.get(f"/api/campaigns/{cid}/export/rows?format=xlsx", headers=H("a-read"))
    assert r.status_code == 400 and "xlsx" not in r.text


def test_csv_neutralizes_formula_injection_but_keeps_numbers(env):
    tc, store = env
    df = _df(6)
    df["comment"] = ["=SUM(A1:A9)", "+1+2", "-cmd|' /C calc'!A0", "@x", "\tfoo", "safe"]
    df["=HYPERLINK"] = [1, 2, 3, 4, 5, 6]
    df.loc[0, "pH"] = -3.5
    cid = _seeded(store, df=df)
    rows = _csv_rows(tc.get(f"/api/campaigns/{cid}/export/rows?format=csv", headers=H("a-read")).text)
    header = rows[0]
    assert "'=HYPERLINK" in header and "=HYPERLINK" not in header
    col = header.index("comment")
    assert [r[col] for r in rows[1:]] == [
        "'=SUM(A1:A9)", "'+1+2", "'-cmd|' /C calc'!A0", "'@x", "'\tfoo", "safe"]
    assert rows[1][header.index("pH")] == "-3.5"  # a number is data, never prefixed
    # JSON returns values unmodified.
    body = tc.get(f"/api/campaigns/{cid}/export/rows", headers=H("a-read")).json()
    assert body["original_rows"]["rows"][0]["comment"] == "=SUM(A1:A9)"


SECRETS = ("PATIENT-JOHN-DOE", "Jane Smith", "sample_id", "operator", "Ward 7 notes")


def _anon_df() -> pd.DataFrame:
    df = _df(8)
    df["sample_id"] = [f"PATIENT-JOHN-DOE-{i}" for i in range(8)]
    df["operator"] = ["Jane Smith", "Bob Roe"] * 4
    df["Notes"] = ["Ward 7 notes"] * 8
    return df


def _anon_analysis(df: pd.DataFrame, *, with_operator_feature: bool = True) -> dict:
    """An analysis exactly as `_analyze(..., anonymize=True)` would publish it:
    built raw, then passed through the analysis's own `_anonymize_result`."""
    cats = ["operator"] if with_operator_feature else []
    raw = {
        "target": "lipase_titer", "features": ["Methanol", "pH", *cats],
        "categorical_features": cats,
        "provenance": [{"name": c, "status": "x"} for c in df.columns],
        "proposals": [{"recipe": {"Methanol": 1.0, "pH": 6.0,
                                  **({"operator": "Jane Smith"} if cats else {})}}],
    }
    out = _anonymize_result(raw, categorical_levels={"operator": ["Bob Roe", "Jane Smith"]} if cats else {})
    out["run"] = {"anonymize": True, "kind": "upload", "input_sha256": "cd" * 32}
    return out


def test_anonymized_export_uses_the_analysis_aliases_and_leaks_nothing(env):
    tc, store = env
    df = _anon_df()
    analysis = _anon_analysis(df)
    cid = _seeded(store, df=df, analysis=analysis)
    _fold(store, cid, [(1.0, 6.0, 1.5)])
    # keep the analysis anonymized after the fold (re-analysis carries the flag)
    store.save_analysis(analysis, "anonymized upload", tenant="acme", campaign_id=cid,
                        campaign_generation=store.generation_of(tenant="acme", campaign_id=cid))
    shown = {row["name"] for row in analysis["provenance"]}

    r = tc.get(f"/api/campaigns/{cid}/export/rows", headers=H("a-read"))
    assert r.status_code == 200, r.text
    body = r.json()
    for secret in SECRETS:
        assert secret not in r.text, secret
    assert set(body["columns"]) == shown
    assert body["anonymization"]["applied"] is True
    assert body["provenance"]["anonymized"] is True
    op_alias = _column_alias("operator")
    assert op_alias in shown and op_alias != "operator"
    # The analysis's own alias for the categorical level is reproduced exactly.
    analysis_level = analysis["proposals"][0]["recipe"][op_alias]
    assert analysis_level == _level_alias("operator", "Jane Smith")
    assert body["original_rows"]["rows"][0][op_alias] == analysis_level
    assert body["original_rows"]["rows"][0]["Methanol"] == df.loc[0, "Methanol"]
    assert body["folded_rows"]["rows"][0]["values"]["lipase_titer"] == 1.5

    r = tc.get(f"/api/campaigns/{cid}/export/rows?format=csv", headers=H("a-read"))
    assert r.status_code == 200 and r.headers["x-kalos-anonymized"] == "true"
    for secret in SECRETS:
        assert secret not in r.text, secret
    assert "runs.csv" not in r.text and "runs.csv" not in json.dumps(dict(r.headers))


def test_level_alias_matches_the_analysis_for_every_label():
    """Drift guard: the export's aliases are the analysis's (same salt, same scheme)."""
    levels = ["Jane Smith", " padded ", "x" * 40]
    out = _anonymize_result(
        {"categorical_features": ["operator"], "features": ["operator"],
         "proposals": [{"recipe": {"operator": lvl.strip()}} for lvl in levels]},
        categorical_levels={"operator": [lvl.strip() for lvl in levels]},
    )
    alias = _column_alias("operator")
    assert out["features"] == [alias]
    assert [p["recipe"][alias] for p in out["proposals"]] == [_level_alias("operator", lvl) for lvl in levels]


def test_anonymized_export_refused_when_aliases_cannot_be_reproduced(env):
    tc, store = env
    df = _anon_df()
    analysis = _anon_analysis(df)
    analysis["provenance"].append({"name": "col_deadbeef", "status": "x"})  # e.g. another salt
    cid = _seeded(store, df=df, analysis=analysis)
    for fmt in ("json", "csv"):
        r = tc.get(f"/api/campaigns/{cid}/export/rows?format={fmt}", headers=H("a-read"))
        assert r.status_code == 409 and r.json()["code"] == "anonymized_export_unavailable"
        for secret in SECRETS:
            assert secret not in r.text


def test_export_refused_when_the_anonymize_setting_is_not_recorded(env):
    tc, store = env
    df = _anon_df()
    analysis = _anon_analysis(df)
    analysis.pop("run")
    cid = _seeded(store, df=df, analysis=analysis)
    r = tc.get(f"/api/campaigns/{cid}/export/rows", headers=H("a-read"))
    assert r.status_code == 409 and r.json()["code"] == "anonymized_export_unavailable"
    assert "PATIENT" not in r.text


def test_export_is_audited(env):
    tc, store = env
    cid = _seeded(store)
    tc.get(f"/api/campaigns/{cid}/export/rows?format=csv", headers=H("a-read"))
    entries = [e for e in store.audit_entries(tenant="acme") if e["action"] == "campaign.export_rows"]
    assert entries and entries[0]["campaign_id"] == cid and entries[0]["detail"] == "format=csv"
    assert entries[0]["subject"] == "carol@acme"


# --- (b) results history ------------------------------------------------------------- #


def test_history_lists_every_version_including_corrections_and_audit(env):
    tc, store = env
    cid = _seeded(store)
    run = store.start([_recipe(1.0, 6.0)], tenant="acme", campaign_id=cid)[0]
    w = tc.post(f"/api/campaigns/{cid}/runs/{run['id']}/result", json={"value": 1.0}, headers=H("a-admin"))
    assert w.status_code == 200
    w = tc.post(f"/api/campaigns/{cid}/runs/{run['id']}/result",
                json={"value": 1.4, "reason": "pipetting error"}, headers=H("a-admin"))
    assert w.status_code == 200

    body = tc.get(f"/api/campaigns/{cid}/results/history", headers=H("a-read")).json()
    items = body["results"]["items"]
    assert [(i["run_id"], i["version"], i["value"], i["previous_value"], i["reason"], i["actor"],
             i["is_correction"]) for i in items] == [
        (run["id"], 2, 1.4, 1.0, "pipetting error", "alice@acme", True),
        (run["id"], 1, 1.0, None, None, "alice@acme", False),
    ]
    assert all(i["recorded_at"] > 0 and i["basis"] == "measured" for i in items)
    actions = [a["action"] for a in body["audit"]["items"]]
    assert actions.count("campaign.result") == 2
    assert all(a["campaign_id"] == cid for a in body["audit"]["items"])
    detail = json.loads(next(a for a in body["audit"]["items"] if a["action"] == "campaign.result")["detail"])
    assert detail["old_value"] == 1.0 and detail["new_value"] == 1.4


def test_history_paginates_with_per_section_cursors(env):
    tc, store = env
    cid = _seeded(store)
    runs = store.start([_recipe(1.0 + i / 10, 6.0) for i in range(5)], tenant="acme", campaign_id=cid)
    for i, run in enumerate(runs):
        store.set_result(run["id"], float(i), tenant="acme", campaign_id=cid, actor="a", audit=True)
    seen: list[int] = []
    cursor = None
    while True:
        q = "section=results&limit=2" + (f"&results_before_id={cursor}" if cursor else "")
        page = tc.get(f"/api/campaigns/{cid}/results/history?{q}", headers=H("a-read")).json()
        assert "audit" not in page
        seen += [i["id"] for i in page["results"]["items"]]
        if not page["results"]["has_more"]:
            assert page["results"]["next_before_id"] is None
            break
        cursor = page["results"]["next_before_id"]
    assert seen == sorted(seen, reverse=True) and len(seen) == len(set(seen)) == 5
    audit = tc.get(f"/api/campaigns/{cid}/results/history?section=audit&limit=3",
                   headers=H("a-read")).json()["audit"]
    assert len(audit["items"]) == 3 and audit["has_more"] is True
    for bad in ("limit=0", "limit=501", "results_before_id=0", "section=everything"):
        assert tc.get(f"/api/campaigns/{cid}/results/history?{bad}", headers=H("a-read")).status_code == 400


def test_history_never_shows_another_tenants_rows(env):
    tc, store = env
    a_cid = _seeded(store, "acme")
    b_cid = _seeded(store, "globex")
    b_run = store.start([_recipe(1.0, 6.0)], tenant="globex", campaign_id=b_cid)[0]
    store.set_result(b_run["id"], 7.7, tenant="globex", campaign_id=b_cid, actor="gina@globex", audit=True)
    body = tc.get(f"/api/campaigns/{a_cid}/results/history", headers=H("a-read"))
    assert body.json()["results"]["items"] == [] and body.json()["audit"]["items"] == []
    assert "gina" not in body.text and "7.7" not in body.text


# --- (c) operating ranges ------------------------------------------------------------- #


def _learnable(n: int = 30, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    a = rng.uniform(0, 10, n)
    b = rng.uniform(20, 40, n)
    y = 5 + 2 * a - 0.05 * (b - 30) ** 2 + rng.normal(0, 0.2, n)
    return pd.DataFrame({"glucose_gL": a.round(3), "temperature_C": b.round(3), "titer": y.round(3)})


def _ranges_analysis(df: pd.DataFrame, **override) -> dict:
    out = {
        "target": "titer", "features": ["glucose_gL", "temperature_C"], "categorical_features": [],
        "feature_ranges": {c: [float(df[c].min()), float(df[c].max())]
                           for c in ("glucose_gL", "temperature_C")},
        "n": len(df), "best": round(float(df["titer"].max()), 4),
        "reliability": {"spearman": 0.95, "ci95": [0.9, 0.98], "spearman_floor": 0.2,
                        "clears_floor": True, "ci_excludes_zero": True},
        "run": {"anonymize": False, "kind": "upload"},
        "promotion": {"passed": False},
    }
    out.update(override)
    return out


def _ranges_campaign(store: CampaignStore, df: pd.DataFrame | None = None, **override) -> str:
    df = _learnable() if df is None else df
    cid = store.create_campaign(tenant="acme", name="r")["id"]
    store.seed(df, "titer", ["glucose_gL", "temperature_C"], tenant="acme", campaign_id=cid,
               analysis=_ranges_analysis(df, **override), dataset="r.csv")
    return cid


def _ranges(tc, cid, body=None, token="a-read"):
    return tc.post(f"/api/campaigns/{cid}/operating-ranges",
                   json={"threshold": 15.0} if body is None else body, headers=H(token))


@pytest.mark.parametrize("override,code", [
    ({"reliability": {"clears_floor": False, "ci_excludes_zero": True, "spearman": 0.1}}, "model_not_reliable"),
    ({"reliability": {"clears_floor": True, "ci_excludes_zero": False}}, "model_not_reliable"),
    ({"reliability": None}, "reliability_unknown"),
    ({"categorical_features": ["glucose_gL", "temperature_C"]}, "categorical_only_design"),
    ({"n": 29}, "refit_data_mismatch"),
    ({"best": 999.0}, "refit_data_mismatch"),
    ({"features": ["glucose_gL", "not_a_column"]}, "refit_data_mismatch"),
    ({"feature_ranges": None}, "analysis_incomplete"),
    ({"run": {"kind": "upload"}}, "anonymized_mapping_unavailable"),
])
def test_operating_ranges_not_computed_paths(env, override, code):
    tc, store = env
    cid = _ranges_campaign(store, **override)
    r = _ranges(tc, cid)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "not_computed" and body["reason_code"] == code
    assert body["reason"] and body["what_would_fix_it"]
    assert "ranges" not in body


def test_operating_ranges_threshold_missing_no_data_no_analysis_stale_goal(env):
    tc, store = env
    cid = _ranges_campaign(store)
    assert _ranges(tc, cid, {}).json()["reason_code"] == "threshold_missing"
    assert tc.get(f"/api/campaigns/{cid}/operating-ranges", headers=H("a-read")).json()[
        "reason_code"] == "threshold_missing"
    empty = store.create_campaign(tenant="acme", name="e")["id"]
    assert _ranges(tc, empty).json()["reason_code"] == "no_data"
    bare = store.create_campaign(tenant="acme", name="b")["id"]
    store.seed(_learnable(), "titer", ["glucose_gL"], tenant="acme", campaign_id=bare)
    assert _ranges(tc, bare).json()["reason_code"] == "no_analysis"
    put_setting(store, tenant="acme", campaign_id=cid, kind="setup", value={"goal": "sideways"},
                updated_by="alice@acme")
    assert _ranges(tc, cid).json()["reason_code"] == "goal_unrecognized"
    run = store.start([{"recipe": {"glucose_gL": 1.0, "temperature_C": 30.0}}], tenant="acme",
                      campaign_id=cid)[0]
    store.set_result(run["id"], 3.0, tenant="acme", campaign_id=cid)
    assert _ranges(tc, cid).json()["reason_code"] == "analysis_stale"


@pytest.mark.parametrize("body", [
    {"threshold": 1.0, "probability": 0.4},
    {"threshold": 1.0, "probability": 1.0},
    {"threshold": 1.0, "basis": "measured"},
    {"threshold": "1.0"},
    {"threshold": 1.0, "extra": 1},
])
def test_operating_ranges_body_validation(env, body):
    tc, store = env
    cid = _ranges_campaign(store)
    assert _ranges(tc, cid, body).status_code == 400


def test_operating_ranges_engine_busy(env):
    tc, store = env
    cid = _ranges_campaign(store)
    with _slots_exhausted("acme"):
        r = _ranges(tc, cid)
    assert r.status_code == 503 and r.json()["code"] == "engine_busy" and r.headers["Retry-After"]


def test_maximal_box_holds_only_passing_samples():
    rng = np.random.default_rng(1)
    u = rng.uniform(size=(4000, 2))
    meets = (u[:, 0] > 0.5) & (u[:, 1] < 0.8)
    box = maximal_box(u, meets, np.array([0.9, 0.3]))
    assert box is not None
    lo, hi = box
    inside = np.all((u >= lo) & (u <= hi), axis=1)
    assert meets[inside].all() and inside.sum() > 500
    assert lo[0] > 0.5 and hi[1] < 0.8 and hi[0] > 0.99 and lo[1] < 0.01
    # an anchor that itself fails has no region
    assert maximal_box(u, np.zeros(len(u), bool), np.array([0.5, 0.5])) is None


@pytest.mark.slow
def test_operating_ranges_computed_on_a_learnable_sheet(env):
    tc, store = env
    df = _learnable(40)
    cid = store.create_campaign(tenant="acme", name="learn")["id"]
    up = tc.post(f"/api/campaigns/{cid}/upload", headers=H("a-admin"),
                 files={"file": ("learn.csv", df.to_csv(index=False).encode(), "text/csv")})
    assert up.status_code == 200, up.text
    rel = up.json()["reliability"]
    assert rel["clears_floor"] and rel["ci_excludes_zero"]
    r = tc.get(f"/api/campaigns/{cid}/operating-ranges?threshold=18&probability=0.9", headers=H("a-read"))
    body = r.json()
    assert body["status"] == "computed", body
    assert body["method"] == "gp_posterior_box" and body["model_based"] is True
    assert body["must_verify_experimentally"] is True and body["basis"] == "predicted"
    assert body["sampling"]["n_samples"] == 4096 and body["box"]["n_samples_inside"] >= 32
    assert body["box"]["all_samples_inside_meet_rule"] is True
    assert 0 < body["fraction_of_design_space_meeting_rule"] < 1
    assert body["anchor"]["measured_value"] == float(df["titer"].max())
    assert body["anchor"]["probability_meets_rule"] >= 0.9
    ranges = {x["feature"]: x for x in body["ranges"]}
    # titer >= 18 needs high glucose (2a >= 13 + curvature): the range sits at the top.
    assert ranges["glucose_gL"]["low"] > 6.0
    for x in body["ranges"]:
        assert x["design_low"] <= x["low"] <= x["high"] <= x["design_high"]
    # The true surface agrees on the box corners (synthetic sheet, known truth).
    g = ranges["glucose_gL"]
    t = ranges["temperature_C"]
    for a in (g["low"], g["high"]):
        for b in (t["low"], t["high"]):
            assert 5 + 2 * a - 0.05 * (b - 30) ** 2 > 18 - 1.0
    # An impossible threshold: no region.
    r = _ranges(tc, cid, {"threshold": 500.0})
    assert r.json()["reason_code"] == "no_region_found"
    assert r.json()["diagnostics"]["fraction_of_design_space_meeting_rule"] == 0.0


@pytest.mark.slow
def test_operating_ranges_minimize_goal_flips_direction(env):
    tc, store = env
    rng = np.random.default_rng(3)
    a = rng.uniform(0, 10, 30)
    b = rng.uniform(20, 40, 30)
    df = pd.DataFrame({"glucose_gL": a.round(3), "temperature_C": b.round(3),
                       "titer": (2 * a + 0.1 * b + rng.normal(0, 0.1, 30)).round(3)})
    cid = _ranges_campaign(store, df=df, best=round(float(df["titer"].max()), 4))
    put_setting(store, tenant="acme", campaign_id=cid, kind="setup", value={"goal": "minimize"},
                updated_by="alice@acme")
    body = _ranges(tc, cid, {"threshold": 8.0, "probability": 0.8}).json()
    assert body["status"] == "computed", body
    assert body["rule"]["direction"] == "at_most" and body["rule"]["goal_source"] == "setup"
    assert body["anchor"]["measured_value"] == float(df["titer"].min())
    assert {x["feature"]: x for x in body["ranges"]}["glucose_gL"]["high"] < 4.0


# --- cross-tenant + scope on every route ---------------------------------------------- #


ROUTES = [
    ("GET", "/api/campaigns/{cid}/export/rows", {}),
    ("GET", "/api/campaigns/{cid}/export/rows?format=csv", {}),
    ("GET", "/api/campaigns/{cid}/results/history", {}),
    ("GET", "/api/campaigns/{cid}/operating-ranges?threshold=1", {}),
    ("POST", "/api/campaigns/{cid}/operating-ranges", {"json": {"threshold": 1.0}}),
]


def _raw(resp) -> tuple:
    return resp.status_code, sorted(resp.headers.items()), resp.content


def test_foreign_campaign_is_byte_identical_to_missing_on_every_route(env):
    tc, store = env
    b_cid = _ranges_campaign_for(store, "globex")
    for method, path, kw in ROUTES:
        foreign = tc.request(method, path.format(cid=b_cid), headers=H("a-admin"), **kw)
        missing = tc.request(method, path.format(cid="cmp_" + uuid.uuid4().hex), headers=H("a-admin"), **kw)
        malformed = tc.request(method, path.format(cid="cmp_nope"), headers=H("a-admin"), **kw)
        assert foreign.status_code == 404, (method, path, foreign.text)
        assert _raw(foreign) == _raw(missing) == _raw(malformed), (method, path)
        assert foreign.json() == {"error": "campaign not found", "code": "not_found"}
        owner = tc.request(method, path.format(cid=b_cid), headers=H("b-admin"), **kw)
        assert owner.status_code == 200, (method, path, owner.text)


def _ranges_campaign_for(store: CampaignStore, tenant: str) -> str:
    df = _learnable()
    cid = store.create_campaign(tenant=tenant, name="secret")["id"]
    store.seed(df, "titer", ["glucose_gL"], tenant=tenant, campaign_id=cid,
               analysis=_ranges_analysis(df, reliability=None), dataset="secret.csv")
    return cid


def test_scope_is_checked_before_the_campaign_is_looked_at(env):
    tc, store = env
    cid = _ranges_campaign_for(store, "acme")
    for method, path, kw in ROUTES:
        for target in (cid, "cmp_" + uuid.uuid4().hex, "cmp_nope"):
            r = tc.request(method, path.format(cid=target), headers=H("a-writeonly"), **kw)
            assert r.status_code == 403, (method, path)
        ok = tc.request(method, path.format(cid=cid), headers=H("a-read"), **kw)
        assert ok.status_code == 200, (method, path, ok.text)
        assert tc.request(method, path.format(cid=cid), **kw).status_code == 401
