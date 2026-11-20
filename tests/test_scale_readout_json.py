"""Tests for the JSON Scale-Up Readout (engine request R1):
`POST /api/scale/readout?format=json`, `Accept: application/json`, and
`POST /api/scale/readout.json`, plus `kalos.scale.readout_json` and the
additive chart evidence in `kalos.scale.readout` (scale_points, backtest,
divergence).

Real end-to-end readouts are a few GP fits each, so content-negotiation and
envelope tests stub `build_readout` with one cached real readout; the
numbers-level tests run the real pipeline.
"""
from __future__ import annotations

import functools
import hashlib
import json
import math

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from kalos.portal import scale_routes
from kalos.portal.app import UPLOAD_PATHS, app, get_campaign_store
from kalos.portal.auth import READ, WRITE
from kalos.portal.busy import AnalysisBusy
from kalos.portal.campaign import CampaignStore
from kalos.scale.readout import (
    ALPHA,
    DIVERGENCE_RULE,
    HARD_CAP_RATIO,
    MIN_RESIDUALS,
    REFUSE_RATIO_MULT,
    WARN_RATIO_MULT,
    LadderResult,
    RungResult,
    TargetSpec,
    build_divergence,
    build_readout,
)
from kalos.scale.readout_json import SCHEMA_VERSION, compute_readout_id, readout_to_json

_DEMO_CSV = "examples/synthetic_scaleup/synthetic_scaleup.csv"
_PROCESS = ["ph_setpoint", "temperature_C"]

_REQUIRED_KEYS = {
    "kind", "schema_version", "readout_id", "decision", "reasons", "prediction", "interval",
    "interval_coverage", "interval_label", "interval_method", "rungs", "skipped_rungs", "reference_ratio",
    "requested_ratio", "out_of_range_params", "baseline_comparison", "data_plan", "target_inputs",
    "target_column_unit", "provenance", "scale_points", "backtest", "max_trained_scale_L",
    "warn_ratio_mult", "refuse_ratio_mult", "divergence",
}


def _sha(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _demo_bytes() -> bytes:
    with open(_DEMO_CSV, "rb") as f:
        return f.read()


def _target_dict(**overrides) -> dict:
    target = {
        "scale_L": 7500.0,
        "agitation_rpm": 150.0,
        "airflow_L_per_min": 300.0,
        "target_column": "titer_g_per_L",
        "process_params": {"ph_setpoint": 7.2, "temperature_C": 37.0},
    }
    target.update(overrides)
    return target


def _post(tc, path="/api/scale/readout", *, target=None, csv_bytes=None, headers=None, params=None):
    return tc.post(
        path,
        params=params or {},
        data={"target": target if isinstance(target, str) else json.dumps(target or _target_dict())},
        files={"file": ("run_sheet.csv", csv_bytes if csv_bytes is not None else _demo_bytes(), "text/csv")},
        headers=headers or {},
    )


def _strict(r) -> dict:
    """Parse the body with NaN/Infinity REJECTED: the payload must be strict JSON."""
    def _no_constants(name):
        raise AssertionError(f"non-strict JSON constant {name}")
    return json.loads(r.text, parse_constant=_no_constants)


@pytest.fixture
def client(tmp_path):
    store = CampaignStore(tmp_path)
    app.dependency_overrides[get_campaign_store] = lambda: store
    try:
        yield TestClient(app), store
    finally:
        app.dependency_overrides.pop(get_campaign_store, None)


@functools.cache
def _demo_readout() -> dict:
    df = pd.read_csv(_DEMO_CSV)
    t = _target_dict()
    target = TargetSpec(
        scale_L=t["scale_L"], process_params=t["process_params"],
        agitation_rpm=t["agitation_rpm"], airflow_L_per_min=t["airflow_L_per_min"],
    )
    return build_readout(df, "titer_g_per_L", _PROCESS, target, raw_bytes=_demo_bytes())


@pytest.fixture
def stubbed(client, monkeypatch):
    """Route with `build_readout` replaced by one cached real readout."""
    readout = _demo_readout()
    monkeypatch.setattr(scale_routes, "build_readout", lambda *a, **k: readout)
    return client[0], readout


# --------------------------------------------------------------------------- #
# content negotiation; HTML stays the default and byte-identical
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "path, params, headers",
    [
        ("/api/scale/readout", {"format": "json"}, {}),
        ("/api/scale/readout", {}, {"Accept": "application/json"}),
        ("/api/scale/readout", {}, {"Accept": "text/html;q=0.5, application/json"}),
        ("/api/scale/readout.json", {}, {}),
        ("/api/scale/readout.json", {}, {"Accept": "text/html"}),
        ("/api/scale/readout", {"format": "json"}, {"Accept": "text/html"}),
    ],
)
def test_json_variants(stubbed, path, params, headers):
    tc, _readout = stubbed
    r = _post(tc, path, params=params, headers=headers)
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("application/json")
    body = _strict(r)
    assert body["kind"] == "scale_readout"
    assert body["schema_version"] == SCHEMA_VERSION


@pytest.mark.parametrize(
    "params, headers",
    [
        ({}, {}),
        ({}, {"Accept": "*/*"}),
        ({}, {"Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"}),
        ({}, {"Accept": "application/json, text/html"}),  # tie keeps the HTML default
        ({}, {"Accept": "application/json;q=0"}),
        ({"format": "html"}, {"Accept": "application/json"}),
    ],
)
def test_html_stays_default_and_byte_identical(stubbed, params, headers):
    tc, readout = stubbed
    r = _post(tc, params=params, headers=headers)
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("text/html")
    assert r.text == scale_routes.render_readout_html(readout)
    assert "accept" in r.headers.get("vary", "").lower()


def test_html_page_ignores_the_new_evidence_keys():
    readout = _demo_readout()
    new = {"scale_points", "max_trained_scale_L", "backtest", "divergence", "interval_half_width"}
    assert new <= set(readout)
    stripped = {k: v for k, v in readout.items() if k not in new}
    assert scale_routes.render_readout_html(readout) == scale_routes.render_readout_html(stripped)


def test_unknown_format_is_400(stubbed):
    tc, _ = stubbed
    r = _post(tc, params={"format": "xml"})
    assert r.status_code == 400
    assert r.json() == {"error": "unsupported format; use 'json' or 'html'", "code": "invalid_format"}


def test_wants_json_parser():
    w = scale_routes._wants_json
    assert w("application/json")
    assert w("application/json;q=0.9, text/html;q=0.8")
    assert not w(None) and not w("") and not w("*/*") and not w("application/*")
    assert not w("text/html, application/json")
    assert not w("application/json;q=bogus")


def test_json_path_gets_the_upload_body_cap():
    assert "/api/scale/readout.json" in UPLOAD_PATHS


# --------------------------------------------------------------------------- #
# payload contents (real pipeline, cached)
# --------------------------------------------------------------------------- #


def test_number_payload_contract(stubbed):
    tc, readout = stubbed
    body = _strict(_post(tc, params={"format": "json"}))
    assert set(body) >= _REQUIRED_KEYS
    assert body["decision"] == "NUMBER"
    lo, hi = body["interval"]
    assert lo < body["prediction"] < hi
    assert body["interval_coverage"] == pytest.approx(1 - ALPHA)
    assert body["interval_method"] == "split_conformal_ladder_residuals"
    assert "approximate" in body["interval_label"] and "not calibrated" in body["interval_label"]
    assert body["interval_half_width"] == pytest.approx((hi - lo) / 2)
    assert body["warn_ratio_mult"] == WARN_RATIO_MULT
    assert body["refuse_ratio_mult"] == REFUSE_RATIO_MULT
    assert body["hard_cap_ratio"] == HARD_CAP_RATIO
    assert body["max_trained_scale_L"] == 5000.0
    assert body["rungs"] == readout["rungs"]
    assert body["provenance"]["raw_upload_sha256"] == hashlib.sha256(_demo_bytes()).hexdigest()

    zones = body["zones"]
    ref = body["reference_ratio"]
    assert zones["basis"] == "derived"
    assert zones["licensed_max_L"] == pytest.approx(min(5000 * WARN_RATIO_MULT * ref, 5000 * HARD_CAP_RATIO))
    assert zones["warn_max_L"] == pytest.approx(min(5000 * REFUSE_RATIO_MULT * ref, 5000 * HARD_CAP_RATIO))


def test_scale_points_pair_measured_rows_with_backtest_predictions():
    readout = _demo_readout()
    df = pd.read_csv(_DEMO_CSV)
    points = readout["scale_points"]
    assert len(points) == len(df)
    assert all(p["basis"] == "measured" for p in points)
    # every measured value is a real row of the sheet (multiset equality)
    assert sorted(p["actual"] for p in points) == pytest.approx(sorted(df["titer_g_per_L"].tolist()))
    held_out = {r["scale_L"] for r in readout["rungs"]}
    for p in points:
        if p["scale_L"] in held_out:
            assert isinstance(p["oof_pred"], float) and p["oof_pred_basis"] == "predicted"
        else:
            assert p["oof_pred"] is None and p["oof_pred_basis"] is None
    # the smallest two scales can never be held out
    smallest = sorted(df["scale_L"].unique())[:2]
    assert all(p["oof_pred"] is None for p in points if p["scale_L"] in smallest)
    # pooled residuals recomputed from scale_points match the engine's pooled MAE
    res = [abs(p["actual"] - p["oof_pred"]) for p in points if p["oof_pred"] is not None]
    assert len(res) == readout["baseline_comparison"]["n_residuals"]
    assert float(np.mean(res)) == pytest.approx(readout["baseline_comparison"]["pooled_mae"])


def test_backtest_block_matches_rungs():
    readout = _demo_readout()
    bt = readout["backtest"]
    assert bt["method"] == "nested_leave_last_scale_out"
    assert [r["scale_L"] for r in bt["per_scale"]] == [r["scale_L"] for r in readout["rungs"]]
    for row, rung in zip(bt["per_scale"], readout["rungs"], strict=True):
        for k in ("n", "mae", "naive_mean_mae", "naive_nn_mae", "beats_both", "too_few_to_judge"):
            assert row[k] == rung[k]
        pts = [p for p in readout["scale_points"] if p["scale_L"] == rung["scale_L"]]
        assert row["measured_mean"] == pytest.approx(np.mean([p["actual"] for p in pts]))
        assert row["backtest_pred_mean"] == pytest.approx(np.mean([p["oof_pred"] for p in pts]))
    assert bt["skipped"] == readout["skipped_rungs"]
    assert bt["pooled"]["n_residuals"] == readout["baseline_comparison"]["n_residuals"]


def test_divergence_on_demo_is_computed_with_a_stated_rule():
    div = _demo_readout()["divergence"]
    assert div["computed"] is True
    assert div["rule"] == DIVERGENCE_RULE
    first = div["per_scale"][0]
    # the first rung has no smaller held-out scales: no band is invented
    assert first["band_half_width"] is None and first["outside_band"] is None
    assert first["band_unavailable_reason"]
    banded = [e for e in div["per_scale"] if e["band_half_width"] is not None]
    assert banded and all(e["n_prior_residuals"] >= MIN_RESIDUALS for e in banded)
    for e in div["per_scale"]:
        assert e["diverges"] == bool(e["codes"])
    assert [f["scale_L"] for f in div["flags"]] == [e["scale_L"] for e in div["per_scale"] if e["diverges"]]


# --------------------------------------------------------------------------- #
# divergence unit tests (no GP)
# --------------------------------------------------------------------------- #


def _rung(scale, *, beats=True, n=5):
    return RungResult(
        scale_L=scale, step_ratio=2.0, n=n, mae=0.1, naive_mean_mae=0.5, naive_nn_mae=0.5,
        beats_both=beats, too_few_to_judge=n < 3,
    )


def _ladder(rungs, rows):
    res = np.asarray([abs(r["actual"] - r["pred"]) for r in rows])
    return LadderResult(
        rungs=rungs, skipped=[], residuals=res, pooled_mae=float(res.mean()) if len(res) else math.nan,
        pooled_naive_mean_mae=1.0, pooled_naive_nn_mae=1.0, pooled_beats_both=True, oof_rows=rows,
    )


def _rows(scale, actuals, preds):
    return [
        {"scale_L": scale, "actual": a, "pred": p, "naive_mean_pred": 0.0, "naive_nn_pred": 0.0}
        for a, p in zip(actuals, preds, strict=True)
    ]


def test_divergence_flags_a_scale_outside_the_smaller_scale_band():
    rows = _rows(4.0, [1.0] * 10, [1.05] * 10) + _rows(8.0, [3.0] * 5, [1.0] * 5) + _rows(16.0, [1.0] * 5, [1.02] * 5)
    out = build_divergence(_ladder([_rung(4.0), _rung(8.0), _rung(16.0)], rows))
    by = {e["scale_L"]: e for e in out["per_scale"]}
    assert by[4.0]["outside_band"] is None  # no smaller residuals
    assert by[8.0]["outside_band"] is True and by[8.0]["band_half_width"] == pytest.approx(0.05)
    assert "outside_backtest_band" in by[8.0]["codes"]
    # 16 L's band uses the 15 residuals from 4 L and 8 L (wide, because 8 L missed)
    assert by[16.0]["n_prior_residuals"] == 15 and by[16.0]["outside_band"] is False
    assert out["flags"] == [{"scale_L": 8.0, "codes": ["outside_backtest_band"]}]


def test_divergence_flags_did_not_beat_baselines_but_not_on_thin_rungs():
    rows = _rows(4.0, [1.0] * 3, [1.0] * 3) + _rows(8.0, [1.0], [1.0])
    out = build_divergence(_ladder([_rung(4.0, beats=False, n=3), _rung(8.0, beats=False, n=1)], rows))
    assert out["flags"] == [{"scale_L": 4.0, "codes": ["did_not_beat_baselines"]}]


def test_divergence_not_computed_without_rungs():
    out = build_divergence(_ladder([], []))
    assert out["computed"] is False and out["reason"]


# --------------------------------------------------------------------------- #
# refusal, gate 422s, busy, scope - shared envelopes
# --------------------------------------------------------------------------- #


def test_refusal_json_never_carries_a_number(client):
    tc, _ = client
    r = _post(tc, "/api/scale/readout.json", target=_target_dict(scale_L=5000.0 * 17.0))
    assert r.status_code == 200, r.text
    body = _strict(r)
    assert body["decision"] == "REFUSAL"
    assert body["prediction"] is None and body["interval"] is None
    assert body["interval_coverage"] is None and body["interval_method"] is None
    assert body["reasons"] and body["data_plan"]
    assert body["rungs"] and body["scale_points"]
    assert body["divergence"]["computed"] is True


def test_sparse_sheet_refusal_is_strict_json(client):
    from test_scale_readout import _sparse_sheet  # pytest puts tests/ on sys.path

    tc, _ = client
    csv_bytes = _sparse_sheet().to_csv(index=False).encode("utf-8")
    target = _target_dict(scale_L=42000.0, airflow_L_per_min=1500.0, process_params={"ph_setpoint": 7.0, "temperature_C": 36.7})
    body = _strict(_post(tc, params={"format": "json"}, target=target, csv_bytes=csv_bytes))
    assert body["decision"] == "REFUSAL"
    assert body["prediction"] is None
    assert any("more run(s) at any scale" in line for line in body["data_plan"])
    assert all(e["band_half_width"] is None for e in body["divergence"]["per_scale"])


@pytest.mark.parametrize("path, params", [("/api/scale/readout", {"format": "json"}), ("/api/scale/readout.json", {})])
def test_gate_and_target_refusals_keep_the_422_envelope(client, path, params):
    tc, _ = client
    r = _post(tc, path, params=params, target=_target_dict(scale_L=100.0))
    assert r.status_code == 422
    assert r.json() == {
        "error": "interpolation is out of scope for this readout",
        "code": "target_not_interpolation",
        "failed_check": "target_not_interpolation",
        "detail": "interpolation is out of scope for this readout",
    }
    r = _post(tc, path, params=params, target="not json")
    assert r.status_code == 422
    assert set(r.json()) == {"error", "code", "failed_check", "detail"}
    assert r.json()["code"] == "invalid_target"


@pytest.mark.parametrize("path, params", [("/api/scale/readout", {"format": "json"}), ("/api/scale/readout.json", {})])
def test_busy_slot_is_503(client, monkeypatch, path, params):
    tc, _ = client

    async def _busy(*_a, **_k):
        raise AnalysisBusy()

    monkeypatch.setattr(scale_routes, "run_in_slot", _busy)
    r = _post(tc, path, params=params)
    assert r.status_code == 503
    assert r.json()["code"] == "engine_busy"
    assert "Retry-After" in r.headers


@pytest.mark.parametrize("path, params", [("/api/scale/readout", {"format": "json"}), ("/api/scale/readout.json", {})])
def test_scope_is_write(stubbed, monkeypatch, path, params):
    tc, _ = stubbed
    monkeypatch.setenv(
        "KALOS_AUTH_TOKENS",
        json.dumps([
            {"token_sha256": _sha("rtok"), "subject": "r", "tenant": "t", "scopes": [READ]},
            {"token_sha256": _sha("wtok"), "subject": "w", "tenant": "t", "scopes": [WRITE]},
        ]),
    )
    monkeypatch.delenv("KALOS_AUTH_TOKENS_FILE", raising=False)
    assert _post(tc, path, params=params).status_code == 401
    assert _post(tc, path, params=params, headers={"Authorization": "Bearer rtok"}).status_code == 403
    assert _post(tc, path, params=params, headers={"Authorization": "Bearer wtok"}).status_code == 200


def test_json_readout_is_stateless(stubbed, client, tmp_path):
    tc, _ = stubbed
    _, store = client
    before = set(tmp_path.rglob("*"))
    assert _post(tc, "/api/scale/readout.json").status_code == 200
    assert set(tmp_path.rglob("*")) == before
    assert store.get(tenant="default") is None


# --------------------------------------------------------------------------- #
# readout_id
# --------------------------------------------------------------------------- #


def _spec(**kw) -> TargetSpec:
    base = {"scale_L": 7500.0, "process_params": {"ph_setpoint": 7.2, "temperature_C": 37.0}}
    base.update(kw)
    return TargetSpec(**base)


def test_readout_id_is_deterministic_and_sensitive():
    rid = compute_readout_id(b"sheet", "y", _spec(), None)
    assert len(rid) == 64 and int(rid, 16) >= 0
    assert rid == compute_readout_id(b"sheet", "y", _spec(), {})
    reordered = _spec(process_params={"temperature_C": 37.0, "ph_setpoint": 7.2})
    assert rid == compute_readout_id(b"sheet", "y", reordered, None)
    assert rid != compute_readout_id(b"sheet2", "y", _spec(), None)
    assert rid != compute_readout_id(b"sheet", "z", _spec(), None)
    assert rid != compute_readout_id(b"sheet", "y", _spec(scale_L=7501.0), None)
    assert rid != compute_readout_id(b"sheet", "y", _spec(), {"power_number": 6.0})
    assert rid != compute_readout_id(b"sheet", "y", _spec(), None, engine_version="0.0.0-other")


def test_route_readout_id_ignores_target_json_whitespace(stubbed):
    tc, _ = stubbed
    compact = json.dumps(_target_dict(), separators=(",", ":"))
    spaced = json.dumps(_target_dict(), indent=4, sort_keys=True)
    a = _strict(_post(tc, "/api/scale/readout.json", target=compact))["readout_id"]
    b = _strict(_post(tc, "/api/scale/readout.json", target=spaced))["readout_id"]
    t = _target_dict()
    expected = compute_readout_id(
        _demo_bytes(), "titer_g_per_L",
        TargetSpec(t["scale_L"], t["process_params"], t["agitation_rpm"], t["airflow_L_per_min"]), None,
    )
    assert a == b == expected


def test_readout_to_json_replaces_non_finite_with_null():
    readout = dict(_demo_readout())
    readout["baseline_comparison"] = {**readout["baseline_comparison"], "pooled_naive_nn_mae": math.nan}
    out = readout_to_json(readout, readout_id="x")
    assert out["baseline_comparison"]["pooled_naive_nn_mae"] is None
    json.dumps(out, allow_nan=False)
