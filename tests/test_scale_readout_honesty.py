"""Readout honesty fixes (VOYAGER_PORT WS3, branch fix/scale-readout-honesty).

1. assay-method confound check (`kalos.scale.assay_confound`) and its effect on
   the readout's decision (the LOSO refusal is in test_scale_evaluation_refusals.py);
2. plate/flask working volumes are labelled assumptions
   (`kalos.scale.working_volume`), never silent.

Every sheet here is synthetic: the same SHAPE as the real sheets that exposed
the gaps (a 5-scale, one-run-per-scale ladder whose assay changes with scale; a
plate/flask funnel), with invented values. No partner data is used.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from kalos.scale import readout as ro
from kalos.scale.assay_confound import (
    FINDING_CODE,
    METHOD_NAME_RE,
    assay_method_confound,
    candidate_columns,
)
from kalos.scale.readout_json import compute_readout_id, readout_to_json
from kalos.scale.working_volume import (
    NOMINAL_WORKING_VOLUME_L,
    SOURCE_DEFAULT,
    SOURCE_USER,
    format_key,
    resolve_scale_labels,
)

_DEMO_CSV = "examples/synthetic_scaleup/synthetic_scaleup.csv"



@pytest.fixture(autouse=True)
def _isolate_checks_setting(monkeypatch):
    # these tests are the checks-on state (the default); see
    # test_scale_readout_checks_setting.py for both states
    monkeypatch.delenv("KALOS_SCALE_READOUT_CHECKS", raising=False)


def _method_ladder() -> pd.DataFrame:
    """5 scales, one run each; the titer assay switches between a gel method and
    HPLC from step to step, one method cell is blank, and the titer follows the
    method (invented values)."""
    return pd.DataFrame(
        {
            "scale_L": [1.5, 4.0, 20.0, 2500.0, 30000.0],
            "induction_ph": [6.1, 6.9, 6.8, 6.2, 6.9],
            "titer_g_L": [0.72, 0.70, 1.84, 0.69, 1.52],
            "titer_method": ["SDS-PAGE", None, "RP-HPLC", "SDS-PAGE", "RP-HPLC"],
        }
    )


# --------------------------------------------------------------------------- #
# 1. assay-method confound: the check itself
# --------------------------------------------------------------------------- #


def test_method_ladder_is_a_refusal_level_confound():
    out = assay_method_confound(_method_ladder(), "titer_g_L", "scale_L")
    assert out["severity"] == "refuse"
    (col,) = out["columns"]
    assert col["column"] == "titer_method" and col["source"] == "name"
    assert col["code"] == FINDING_CODE
    assert col["cramers_v_scale"] == 1.0
    assert col["every_step_changes_method"] is True
    # the blank-method run (0.70) sits inside the gel range, so no complete
    # separation; the between-method share carries the finding
    assert col["target_separates_completely"] is False
    assert col["target_share"] >= 0.5
    assert col["levels"] == {"RP-HPLC": 2, "SDS-PAGE": 2, "not recorded": 1}
    assert col["target_by_method"]["RP-HPLC"]["basis"] == "measured"
    assert out["reasons"][0].startswith("outcome tracks assay method; effects confounded")
    assert out["what_would_fix_it"]


def test_aliased_method_without_a_change_at_every_step_warns():
    df = pd.DataFrame(
        {
            "scale_L": [1.0] * 3 + [10.0] * 3 + [100.0] * 3 + [1000.0] * 3,
            "titer": [1.0, 1.1, 0.9, 1.05, 0.95, 1.0, 2.0, 2.1, 1.9, 2.05, 1.95, 2.0],
            "assay": ["gel"] * 6 + ["hplc"] * 6,
        }
    )
    out = assay_method_confound(df, "titer", "scale_L")
    assert out["severity"] == "warn"
    col = out["columns"][0]
    assert col["cramers_v_scale"] == 1.0
    assert col["every_step_changes_method"] is False
    assert "effects confounded" in col["message"]


def test_method_balanced_across_scales_and_unrelated_to_outcome_is_silent():
    rng = np.random.default_rng(3)
    df = pd.DataFrame(
        {
            "scale_L": np.repeat([1.0, 10.0, 100.0, 1000.0], 4),
            "titer": rng.normal(2.0, 0.1, 16),
            "quantification_method": ["gel", "hplc"] * 8,
        }
    )
    out = assay_method_confound(df, "titer", "scale_L")
    assert out["severity"] is None
    assert out["reasons"] == []
    assert out["columns"][0]["cramers_v_scale"] == 0.0


def test_outcome_tracking_a_method_that_is_not_tied_to_scale_still_warns():
    rng = np.random.default_rng(4)
    method = ["gel", "hplc"] * 8
    titer = np.array([1.0 if m == "gel" else 2.0 for m in method]) + rng.normal(0, 0.05, 16)
    df = pd.DataFrame({"scale_L": np.repeat([1.0, 10.0, 100.0, 1000.0], 4), "titer": titer, "assay": method})
    col = assay_method_confound(df, "titer", "scale_L")["columns"][0]
    assert col["severity"] == "warn"
    assert col["target_share_p"] <= 0.05
    assert col["message"].startswith("outcome tracks assay method:")
    assert "effects confounded" not in col["message"]


def test_exact_permutation_p_on_a_tiny_complete_separation():
    # 2 + 3 runs: 10 distinct relabellings, only the observed one separates.
    df = pd.DataFrame({"scale_L": [1.0, 2.0, 3.0, 4.0, 5.0], "y": [1, 1.1, 3, 3.1, 3.2], "method": list("aabbb")})
    col = assay_method_confound(df, "y", "scale_L")["columns"][0]
    assert col["target_share_p_method"] == "exact"
    assert col["target_share_p"] == pytest.approx(0.1)


def test_single_method_is_skipped_with_a_reason():
    df = _method_ladder().assign(titer_method="RP-HPLC")
    out = assay_method_confound(df, "titer_g_L", "scale_L")
    assert out["severity"] is None
    assert "one method level" in out["columns"][0]["skipped"]


def test_candidate_columns_by_name_by_declaration_and_not_numeric_measurements():
    df = _method_ladder().assign(
        assay_cv_pct=[3.1, 4.2, 2.9, 5.5, 3.3],  # a numeric measurement, not a method
        instrument="A",
        lab_kit=["k1", "k1", "k2", "k1", "k2"],
    )
    cands, skipped = candidate_columns(
        df, exclude=["titer_g_L", "scale_L"], declared=["lab_kit", "not_a_column"]
    )
    assert ("lab_kit", "declared") in cands
    assert ("titer_method", "name") in cands and ("instrument", "name") in cands
    assert all(c != "assay_cv_pct" for c, _ in cands)
    assert skipped == [{"column": "not_a_column", "reason": "declared as an assay method column but not in the sheet"}]
    assert METHOD_NAME_RE.search("lipase_titer_method") and not METHOD_NAME_RE.search("methanol_feed")


def test_confound_check_is_deterministic():
    rng = np.random.default_rng(5)
    df = pd.DataFrame(
        {"scale_L": np.repeat([1.0, 10.0, 100.0], 10), "y": rng.normal(size=30), "method": list("ab") * 15}
    )
    assert assay_method_confound(df, "y", "scale_L") == assay_method_confound(df, "y", "scale_L")


# --------------------------------------------------------------------------- #
# 1b. the readout downgrades on the finding
# --------------------------------------------------------------------------- #


def test_readout_on_the_method_ladder_refuses_with_every_reason():
    out = ro.build_readout(_method_ladder(), "titer_g_L", [], ro.TargetSpec(scale_L=90000.0, process_params={}))
    assert out["decision"] == ro.REFUSAL
    reasons = out["reasons"]
    assert reasons[0] == "only 1 pooled ladder residual(s); need at least 10"
    assert reasons[1].startswith("no rung with at least 3 runs")
    assert reasons[2].startswith("outcome tracks assay method; effects confounded: 'titer_method'")
    assert out["assay_confound"]["severity"] == "refuse"
    assert any("one assay method" in step for step in out["data_plan"])
    assert out["prediction"] is None and out["interval"] is None


def _decision(decision: str) -> ro.DecisionResult:
    return ro.DecisionResult(
        decision=decision, reasons=["ladder"], reference_ratio=2.0, requested_ratio=1.5, out_of_range_params=["ph"]
    )


@pytest.mark.parametrize(
    ("ladder", "severity", "expected"),
    [
        (ro.NUMBER, "warn", ro.NUMBER_WITH_WARNING),
        (ro.NUMBER_WITH_WARNING, "warn", ro.NUMBER_WITH_WARNING),
        (ro.NUMBER, "refuse", ro.REFUSAL),
        (ro.NUMBER_WITH_WARNING, "refuse", ro.REFUSAL),
        (ro.REFUSAL, "warn", ro.REFUSAL),  # never an upgrade
    ],
)
def test_measured_checks_only_ever_downgrade(ladder, severity, expected):
    confound = {"severity": severity}
    out = ro._apply_measured_checks(_decision(ladder), confound, None, ["finding"])
    assert out.decision == expected
    assert out.reasons == ["ladder", "finding"]


def test_number_becomes_a_warning_when_the_outcome_tracks_a_method():
    df = pd.read_csv(_DEMO_CSV)
    # a planted method column: HPLC for the upper half of the titers
    df["titer_method"] = np.where(df["titer_g_per_L"] > df["titer_g_per_L"].median(), "hplc", "gel")
    target = ro.TargetSpec(
        scale_L=7500.0,
        agitation_rpm=150.0,
        airflow_L_per_min=10.0,
        process_params={"ph_setpoint": 7.0, "temperature_C": 36.7},
    )
    clean = ro.build_readout(pd.read_csv(_DEMO_CSV), "titer_g_per_L", ["ph_setpoint", "temperature_C"], target)
    assert clean["decision"] == ro.NUMBER  # the same sheet without the column
    out = ro.build_readout(df, "titer_g_per_L", ["ph_setpoint", "temperature_C"], target)
    assert out["decision"] == ro.NUMBER_WITH_WARNING
    assert any(r.startswith("outcome tracks assay method") for r in out["reasons"])
    assert out["prediction"] == clean["prediction"]  # the warning does not move the number
    payload = readout_to_json(out, readout_id="x")
    assert payload["assay_confound"]["severity"] == "warn"


# --------------------------------------------------------------------------- #
# 3. plate and flask working volumes: explicit, labelled assumptions
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("label", "key"),
    [
        ("384w", "384w"), ("384-well", "384w"), ("384 well plate", "384w"),
        ("96w", "96w"), ("96-well", "96w"),
        ("24w", "24w"), ("24 deep well", "24w"), ("24DWP", "24w"),
        ("SF125", "SF125"), ("sf-125", "SF125"), ("125 mL shake flask", "SF125"),
        ("96 deep well", None), ("SF250", None), ("6-well", None), ("bioreactor", None),
    ],
)
def test_format_labels(label, key):
    assert format_key(label) == key


def test_resolution_labels_every_volume_and_never_guesses():
    res = resolve_scale_labels(["384w", "SF125", 2.0, None, "96 deep well", "SF125"])
    assert res.values[:4] == [NOMINAL_WORKING_VOLUME_L["384w"], 0.03, 2.0, None]
    assert res.values[4] == "96 deep well"
    assert res.unresolved == ["96 deep well"]
    assert res.assumptions["SF125"]["n_rows"] == 2
    assert res.assumptions["SF125"]["source"] == SOURCE_DEFAULT
    assert res.assumptions["SF125"]["basis"] == "assumption"
    assert res.used_defaults


def test_user_overrides_by_key_or_by_label_win():
    res = resolve_scale_labels(["SF125", "96 deep well"], {"SF125": 0.025, "96 deep well": 0.0012})
    assert res.values == [0.025, 0.0012]
    assert {a["source"] for a in res.assumptions.values()} == {SOURCE_USER}
    assert not res.used_defaults and res.unresolved == []
    with pytest.raises(ValueError, match="positive"):
        resolve_scale_labels(["SF125"], {"SF125": 0.0})


def _funnel_sheet(labels: list[str]) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    n = 6 * len(labels)
    return pd.DataFrame(
        {
            "scale_L": [lbl for lbl in labels for _ in range(6)],
            "ph": np.round(rng.uniform(6.8, 7.2, n), 2),
            "titer": np.round(rng.uniform(1.0, 3.0, n), 2),
        }
    )


def test_readout_prints_every_assumed_volume():
    df = _funnel_sheet(["384w", "96w", "24w", "SF125"])
    out = ro.build_readout(df, "titer", ["ph"], ro.TargetSpec(scale_L=0.25, process_params={"ph": 7.0}))
    assert out["gate"]["passed"]
    pa = out["physics_assumptions"]
    for key in ("384w", "96w", "24w", "SF125"):
        assert pa[f"nominal_working_volume_L[{key}]"] == {
            "value": NOMINAL_WORKING_VOLUME_L[key],
            "source": SOURCE_DEFAULT,
        }
    wv = out["working_volumes"]
    assert wv["applied"] and wv["basis"] == "assumption"
    assert any(r.startswith("scale volumes for plate/flask formats are nominal") for r in out["reasons"])
    payload = readout_to_json(out, readout_id="x")
    assert payload["working_volumes"]["assumptions"]["SF125"]["value"] == 0.03


def test_readout_refuses_an_unknown_vessel_label_by_name():
    df = _funnel_sheet(["384w", "96 deep well", "SF125"])
    out = ro.build_readout(df, "titer", ["ph"], ro.TargetSpec(scale_L=0.25, process_params={"ph": 7.0}))
    assert out["decision"] == ro.REFUSAL
    assert out["gate"]["failed_check"] == "scale_labels_unresolved"
    assert "'96 deep well'" in out["gate"]["detail"]
    assert out["working_volumes"]["unresolved"] == ["96 deep well"]


def test_numeric_scale_sheets_are_untouched():
    df = pd.read_csv(_DEMO_CSV)
    frame, res = ro._resolve_working_volumes(df, None)
    assert frame is df and res is None


def test_volume_number_with_defaults_is_downgraded_to_a_warning():
    from kalos.scale.working_volume import VolumeResolution

    res = VolumeResolution(values=[], assumptions={"SF125": {"source": SOURCE_DEFAULT}}, unresolved=[])
    out = ro._apply_measured_checks(_decision(ro.NUMBER), {"severity": None}, res, ["volumes"])
    assert out.decision == ro.NUMBER_WITH_WARNING
    user = VolumeResolution(values=[], assumptions={"SF125": {"source": SOURCE_USER}}, unresolved=[])
    out = ro._apply_measured_checks(_decision(ro.NUMBER), {"severity": None}, user, ["volumes"])
    assert out.decision == ro.NUMBER


def test_readout_id_unchanged_without_checks_and_sensitive_with_them():
    spec = ro.TargetSpec(scale_L=7500.0, process_params={"ph": 7.0})
    base = compute_readout_id(b"s", "y", spec, None)
    assert base == compute_readout_id(b"s", "y", spec, None, checks={})
    assert base == compute_readout_id(b"s", "y", spec, None, checks={"assay_columns": []})
    assert base != compute_readout_id(b"s", "y", spec, None, checks={"assay_columns": ["m"]})
    assert base != compute_readout_id(b"s", "y", spec, None, checks={"working_volume_overrides": {"SF125": 0.025}})


# --------------------------------------------------------------------------- #
# route: the optional target keys
# --------------------------------------------------------------------------- #


@pytest.fixture
def client(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from kalos.portal.app import app, get_campaign_store
    from kalos.portal.campaign import CampaignStore

    store = CampaignStore(tmp_path)
    app.dependency_overrides[get_campaign_store] = lambda: store
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(get_campaign_store, None)


def _post(tc, target: dict, df: pd.DataFrame):
    return tc.post(
        "/api/scale/readout.json",
        data={"target": json.dumps(target)},
        files={"file": ("sheet.csv", df.to_csv(index=False).encode(), "text/csv")},
    )


def test_route_passes_volume_overrides_and_assay_columns(client):
    df = _funnel_sheet(["384w", "96w", "24w", "SF125"]).assign(kit=lambda d: np.where(d.index % 2, "a", "b"))
    target = {
        "scale_L": 0.25, "target_column": "titer", "process_params": {"ph": 7.0},
        "assay_columns": ["kit"], "working_volume_overrides": {"SF125": 0.025},
    }
    r = _post(client, target, df)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["working_volumes"]["assumptions"]["SF125"]["source"] == SOURCE_USER
    assert body["physics_assumptions"]["nominal_working_volume_L[SF125]"]["value"] == 0.025
    assert [c["column"] for c in body["assay_confound"]["columns"]] == ["kit"]
    assert body["provenance"]["assay_columns_declared"] == ["kit"]


@pytest.mark.parametrize(
    "bad",
    [{"assay_columns": "kit"}, {"assay_columns": [1]}, {"working_volume_overrides": ["SF125"]},
     {"working_volume_overrides": {"SF125": "big"}}],
)
def test_route_rejects_malformed_optional_keys(client, bad):
    target = {"scale_L": 0.25, "target_column": "titer", "process_params": {"ph": 7.0}, **bad}
    r = _post(client, target, _funnel_sheet(["384w", "96w", "24w"]))
    assert r.status_code == 422
    assert r.json()["code"] == "invalid_target"


def test_route_unknown_vessel_label_is_a_named_422(client):
    target = {"scale_L": 0.25, "target_column": "titer", "process_params": {"ph": 7.0}}
    r = _post(client, target, _funnel_sheet(["384w", "96 deep well", "SF125"]))
    assert r.status_code == 422
    assert r.json()["code"] == "scale_labels_unresolved"
    assert "96 deep well" in r.json()["detail"]
