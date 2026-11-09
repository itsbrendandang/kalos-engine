"""Regressions for the four replay defects reported on PR 123.

Synthetic sheets only. Each case was reproduced on main before the fix:

- two CV groups leave a NaN interval, and JSONResponse (allow_nan=False) raises
- a standard label + measurement sheet whose labels contain "pH" is transposed
  and then refused as too few rows
- a -1/0/1 column named like a physical measurement fails physical bounds
- a large-magnitude column that only jitters by float noise is still searched
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from starlette.responses import JSONResponse

from kalos.normalize.orientation import detect_orientation
from kalos.portal.analysis_parts.ranges import _physical_range
from kalos.portal.uploads import parse_upload
from kalos.validation.bounds import is_coded_factor
from kalos.validation.checks import check_constant_columns, check_physical_bounds
from kalos.validation.runner import validate_frame
from kalos.validation.target import (
    numeric_span_is_constant,
    numeric_span_is_near_constant,
)

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from kalos.portal import app as portal_module  # noqa: E402
from kalos.portal import campaign as campaign_module  # noqa: E402
from kalos.portal.analysis import _analyze  # noqa: E402
from kalos.portal.campaign import CampaignStore, get_campaign_store  # noqa: E402


def _walk_nonfinite(obj):
    found = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            found.extend(f"{key}.{item}" for item in _walk_nonfinite(value))
    elif isinstance(obj, list):
        for value in obj:
            found.extend(_walk_nonfinite(value))
    elif isinstance(obj, (float, np.floating)) and not np.isfinite(float(obj)):
        found.append(repr(obj))
    return found


# --- 1. two CV groups: finite JSON, coded interval absence ------------------- #


def _two_group_sheet() -> pd.DataFrame:
    """Two recipes, four replicates each. The recipe is the CV group, so there are two."""
    rows = []
    for methanol, ph, base in ((0.4, 6.2, 1.0), (3.2, 7.2, 3.0)):
        for k in range(4):
            rows.append({
                "Methanol": methanol,
                "pH": ph,
                "titer_g_L": round(base + 0.05 * k, 4),
            })
    return pd.DataFrame(rows)


def test_two_groups_return_a_finite_interval_or_a_coded_absence():
    out = _analyze(_two_group_sheet(), target="titer_g_L")
    assert out["cv_n_groups"] == 2
    assert out["cv_spearman"] is not None
    assert out["cv_ci95"] is None
    assert out["cv_ci95_code"] == "cv_ci_unavailable"
    assert out["reliability"]["ci95"] is None
    assert out["reliability"]["spearman"] == out["cv_spearman"]
    assert _walk_nonfinite(out) == []
    JSONResponse(out)  # allow_nan=False; a NaN here used to raise


@pytest.fixture
def client(tmp_path, monkeypatch):
    store = CampaignStore(tmp_path)
    monkeypatch.setenv("KALOS_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(portal_module, "_STATE_DIR", tmp_path)
    monkeypatch.setattr(portal_module, "_LATEST_DIR", tmp_path / "latest")
    monkeypatch.setattr(campaign_module, "_STORE", store)
    portal_module.app.dependency_overrides[get_campaign_store] = lambda: store
    try:
        yield TestClient(portal_module.app)
    finally:
        portal_module.app.dependency_overrides.pop(get_campaign_store, None)


def test_two_group_upload_and_reanalyze_are_finite_json(client):
    raw = _two_group_sheet().to_csv(index=False).encode()
    uploaded = client.post(
        "/api/run",
        files={"file": ("runs.csv", raw, "text/csv")},
        data={"target": "titer_g_L"},
    )
    assert uploaded.status_code == 200, uploaded.text[:400]
    body = uploaded.json()
    assert body["cv_ci95"] is None
    assert body["cv_ci95_code"] == "cv_ci_unavailable"
    # Promotion text says "missing/NaN" in words. The payload itself must still
    # be strict JSON: no bare NaN/Infinity tokens.
    json.dumps(body, allow_nan=False)

    # Fold one more replicate of an existing recipe so reanalyze still has two
    # groups, then the same payload has to come back as strict JSON. Before the
    # fix this response was HTTP 500: reanalyze builds JSONResponse outside the
    # upload route's error handler.
    started = client.post("/api/campaign/start", json={"recipes": [
        {"recipe": {"Methanol": 0.4, "pH": 6.2}, "pred": 1.1, "std": 0.2,
         "mode": "explore", "reason": "replicate"},
    ]})
    assert started.status_code == 200, started.text[:300]
    recorded = client.post(
        "/api/campaign/result",
        json={"id": started.json()["started"][0]["id"], "value": 1.15},
    )
    assert recorded.status_code == 200, recorded.text[:300]
    again = client.post("/api/campaign/reanalyze")
    assert again.status_code == 200, again.text[:400]
    payload = again.json()
    analysis = payload["analysis"]
    assert analysis["cv_ci95"] is None
    assert analysis["cv_ci95_code"] == "cv_ci_unavailable"
    json.dumps(payload, allow_nan=False)


# --- 2. transposed parse must not refuse a standard narrow sheet ------------- #


def _label_sheet() -> pd.DataFrame:
    """One text factor whose levels name a pH, plus the measurement. Standard."""
    rows = []
    for i, ph in enumerate([4, 5, 6, 7, 4, 5, 6, 7]):
        rows.append({"condition": f"T2 pH {ph}", "mucic_acid_g_L": round(1.0 + 0.15 * i, 3)})
    return pd.DataFrame(rows)


def test_repeated_ph_labels_are_not_detected_as_transposed():
    report = detect_orientation(_label_sheet())
    assert report.orientation == "standard"
    assert report.normalized_frame is None


def test_label_sheet_parses_and_analyzes_instead_of_too_few_rows():
    text = _label_sheet().to_csv(index=False).encode()
    parsed = parse_upload(text, filename="runs.csv")
    assert parsed.df.shape == (8, 2)
    assert parsed.df.attrs["kalos_orientation"]["orientation"] == "standard"
    out = _analyze(parsed.df)
    assert out["n"] == 8
    assert "condition" in out["categorical_features"]
    assert out["target"] == "mucic_acid_g_L"


def test_wider_sheet_with_repeated_ph_labels_is_not_transposed():
    """The same label token on every row must not flip a sheet that has real columns."""
    df = _label_sheet()
    df.insert(1, "feed_g_L", np.linspace(10, 40, len(df)))
    report = detect_orientation(df)
    assert report.orientation != "transposed"
    assert report.normalized_frame is None


# --- 3. coded columns are not physical-bounds failures ----------------------- #


def _coded_sheet() -> pd.DataFrame:
    return pd.DataFrame({
        "pH": [-1, 0, 1, -1, 0, 1, -1, 0],
        "temperature": [-1, 0, 1, -1, 0, 1, -1, 0],
        "feed_rate": [-1, -0.5, 0, 0.5, 1, -1, 0, 1],
        "feed_rate_coded": [-0.8, -0.2, 0.1, 0.4, -0.5, 0.0, 0.7, -1.0],
        "glucose_g_L": [10, 20, 30, 40, 15, 25, 35, 45],
        "titer_g_L": [1.0, 1.2, 1.5, 1.7, 1.1, 1.4, 1.6, 1.8],
    })


def test_coded_factors_skip_physical_bounds_and_real_violations_do_not():
    findings = check_physical_bounds(_coded_sheet())
    errors = [f for f in findings if f.severity == "error"]
    assert errors == []
    coded = {f.column for f in findings if f.check == "coded_column"}
    assert coded == {"pH", "temperature", "feed_rate", "feed_rate_coded"}
    assert "glucose_g_L" not in coded

    impossible = pd.DataFrame({
        "pH": [7.0, 7.1, 40.0, 6.9, 7.0, 7.2, 6.8, 7.0],
        "feed_mL_h": [0.2, 0.3, 0.4, -0.6, 0.5, 0.55, 0.6, 0.7],
        "temp_C": [37, 37.1, 36.9, 37, -999, 37.2, 36.8, 37],
        "titer_g_L": [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7],
    })
    flagged = {
        f.column for f in check_physical_bounds(impossible) if f.severity == "error"
    }
    assert flagged == {"pH", "feed_mL_h", "temp_C"}


def test_one_negative_sentinel_on_a_lattice_is_still_an_error():
    """A single -1 among otherwise non-negative lattice points is a typo, not a level."""
    df = pd.DataFrame({
        "feed_rate": [0.0, 0.5, 1.0, 0.0, 0.5, 1.0, 0.0, -1.0],
        "titer_g_L": [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7],
    })
    assert not is_coded_factor("feed_rate", df["feed_rate"].to_numpy(float))
    errors = [f for f in check_physical_bounds(df) if f.severity == "error"]
    assert [f.column for f in errors] == ["feed_rate"]


def test_coded_factor_keeps_its_code_box_and_strict_mode_does_not_fail():
    coded = np.array([-1.0, 0.0, 1.0, -1.0, 0.0, 1.0])
    lo, hi, n_excluded = _physical_range("pH", coded)
    assert (lo, hi, n_excluded) == (-1.0, 1.0, 0)
    # The sentinel that used to stretch a temperature box is still excluded.
    temps = np.array([37.0, 37.2, 36.8, -999.0, 37.1])
    lo_t, hi_t, n_t = _physical_range("temp_C", temps)
    assert n_t == 1
    assert lo_t == pytest.approx(36.8)
    assert hi_t == pytest.approx(37.2)

    report = validate_frame(_coded_sheet(), target="titer_g_L")
    assert report.status != "fail"
    assert report.counts["error"] == 0


# --- 4. near-constant large factors are held fixed --------------------------- #


def test_float_dust_on_a_large_factor_is_near_constant_and_a_real_span_is_not():
    dust = np.array([5000.0] * 11 + [5000.0 + 1e-6])
    assert numeric_span_is_near_constant(dust)
    assert numeric_span_is_constant(dust)
    # The clean-sheet temperature span (36.5 to 36.85) is a real searched range.
    temps = [36.5 + 0.05 * (i % 8) for i in range(24)]
    assert not numeric_span_is_constant(temps)
    assert not numeric_span_is_near_constant(temps)
    # Relative 1e-6 of 5000 is above the float-noise bar and stays a factor.
    resolvable = np.array([5000.0] * 11 + [5000.0 * (1 + 1e-6)])
    assert not numeric_span_is_constant(resolvable)


def test_near_constant_volume_is_flagged_and_not_searched():
    n = 12
    methanol = np.linspace(0.2, 3.5, n)
    volume = np.full(n, 5000.0)
    volume[-1] = 5000.0 + 1e-6
    df = pd.DataFrame({
        "Methanol": np.round(methanol, 4),
        "Culture_Volume": volume,
        "titer_g_L": np.round(1.1 * methanol + 0.2, 4),
    })
    flagged = [f for f in check_constant_columns(df) if f.check == "near_constant_input"]
    assert [f.column for f in flagged] == ["Culture_Volume"]

    out = _analyze(df, target="titer_g_L")
    assert "Culture_Volume" not in out["features"]
    assert "Culture_Volume" not in out["feature_ranges"]
    assert "Methanol" in out["features"]
    for proposal in out["proposals"]:
        assert "Culture_Volume" not in proposal["recipe"]
    status = {row["name"]: row["status"] for row in out["provenance"]}
    assert status["Culture_Volume"] == "dropped_constant"
    findings = [
        f for f in out["validation"]["findings"]
        if f["check"] == "near_constant_input" and f["column"] == "Culture_Volume"
    ]
    assert findings and findings[0]["severity"] == "info"
