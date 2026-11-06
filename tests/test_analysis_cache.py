"""The same upload is not refit (kalos/portal/pipeline.py, `ANALYSIS_CACHE`)."""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from kalos.portal import pipeline

CSV = (
    b"pH,temp_C,Titer\n6.1,30,10\n6.5,31,12\n6.9,32,15\n7.2,33,14\n"
    b"7.6,34,11\n6.3,35,9\n7.0,30.5,13\n6.7,33.5,16\n"
)


@pytest.fixture(autouse=True)
def _cache_on(monkeypatch):
    monkeypatch.setenv("KALOS_ANALYSIS_CACHE", "4")
    pipeline.ANALYSIS_CACHE.clear()
    yield
    pipeline.ANALYSIS_CACHE.clear()


def _stub():
    calls = {"n": 0}

    def analyze(df, target, **kwargs):
        calls["n"] += 1
        return {"target": target or "Titer", "d": 2, "seed": 1234, "proposals": [{"x": calls["n"]}]}

    return analyze, calls


def test_identical_upload_is_analyzed_once():
    analyze, calls = _stub()
    _, first = pipeline.compute_upload_analysis(CSV, "Titer", False, "a.csv", analyze=analyze)
    _, second = pipeline.compute_upload_analysis(CSV, "Titer", False, "a.csv", analyze=analyze)
    assert calls["n"] == 1
    assert first["analysis_cache"] == {"hit": False} and second["analysis_cache"] == {"hit": True}
    assert second["proposals"] == first["proposals"]
    # The run record is rebuilt per request; the input hash is the same.
    assert second["run"]["input_sha256"] == first["run"]["input_sha256"]


def test_returned_results_are_independent_copies():
    analyze, _ = _stub()
    _, first = pipeline.compute_upload_analysis(CSV, "Titer", False, "a.csv", analyze=analyze)
    first["proposals"][0]["x"] = "mutated"
    _, second = pipeline.compute_upload_analysis(CSV, "Titer", False, "a.csv", analyze=analyze)
    assert second["proposals"][0]["x"] == 1


@pytest.mark.parametrize("change", [
    {"raw": CSV + b"6.8,31.5,14\n"},
    {"target": "pH"},
    {"filename": "a.tsv"},
    {"roles_json": '{"target": "Titer"}'},
    {"anonymize": True},
])
def test_any_input_change_refits(change):
    analyze, calls = _stub()
    base = {"raw": CSV, "target": "Titer", "anonymize": False, "filename": "a.csv", "roles_json": ""}
    pipeline.compute_upload_analysis(**base, analyze=analyze)
    pipeline.compute_upload_analysis(**{**base, **change}, analyze=analyze)
    assert calls["n"] == 2


def test_a_setting_change_refits(monkeypatch):
    analyze, calls = _stub()
    pipeline.compute_upload_analysis(CSV, "Titer", False, "a.csv", analyze=analyze)
    monkeypatch.setenv("KALOS_VALIDATION_MODE", "strict")
    pipeline.compute_upload_analysis(CSV, "Titer", False, "a.csv", analyze=analyze)
    assert calls["n"] == 2


def test_a_different_analyze_function_is_never_served_another_ones_result():
    a1, c1 = _stub()
    a2, c2 = _stub()
    pipeline.compute_upload_analysis(CSV, "Titer", False, "a.csv", analyze=a1)
    pipeline.compute_upload_analysis(CSV, "Titer", False, "a.csv", analyze=a2)
    assert (c1["n"], c2["n"]) == (1, 1)


def test_cache_off_and_refusals_not_kept(monkeypatch):
    analyze, calls = _stub()
    monkeypatch.setenv("KALOS_ANALYSIS_CACHE", "0")
    pipeline.compute_upload_analysis(CSV, "Titer", False, "a.csv", analyze=analyze)
    pipeline.compute_upload_analysis(CSV, "Titer", False, "a.csv", analyze=analyze)
    assert calls["n"] == 2
    monkeypatch.setenv("KALOS_ANALYSIS_CACHE", "4")
    n = {"n": 0}

    def refuse(df, target, **kwargs):
        n["n"] += 1
        raise ValueError("refused")

    for _ in range(2):
        with pytest.raises(ValueError):
            pipeline.compute_upload_analysis(CSV, "Titer", False, "a.csv", analyze=refuse)
    assert n["n"] == 2


def test_cache_keeps_the_last_n_entries(monkeypatch):
    monkeypatch.setenv("KALOS_ANALYSIS_CACHE", "1")
    analyze, calls = _stub()
    pipeline.compute_upload_analysis(CSV, "Titer", False, "a.csv", analyze=analyze)
    pipeline.compute_upload_analysis(CSV, "pH", False, "a.csv", analyze=analyze)
    pipeline.compute_upload_analysis(CSV, "Titer", False, "a.csv", analyze=analyze)
    assert calls["n"] == 3


def test_real_reupload_is_not_refit_over_http(tmp_path):
    """Two identical POST /api/run with replace=true: the second returns the
    same proposals without a second fit."""
    from fastapi.testclient import TestClient

    from kalos.portal import analysis
    from kalos.portal.app import app, get_campaign_store
    from kalos.portal.campaign import CampaignStore

    store = CampaignStore(tmp_path)
    app.dependency_overrides[get_campaign_store] = lambda: store
    real = analysis._analyze
    calls = {"n": 0}

    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    import kalos.portal.app as app_module

    orig = app_module._analyze
    app_module._analyze = counting
    try:
        client = TestClient(app)
        r1 = client.post("/api/run", files={"file": ("s.csv", CSV)}, data={"target": "Titer", "replace": "true"})
        r2 = client.post("/api/run", files={"file": ("s.csv", CSV)}, data={"target": "Titer", "replace": "true"})
    finally:
        app_module._analyze = orig
        app.dependency_overrides.pop(get_campaign_store, None)
    assert r1.status_code == r2.status_code == 200, (r1.text, r2.text)
    assert calls["n"] == 1
    assert r2.json()["analysis_cache"] == {"hit": True}
    assert r2.json()["proposals"] == r1.json()["proposals"]
    assert store.load_analysis(tenant="default")["analysis_cache"] == {"hit": True}
