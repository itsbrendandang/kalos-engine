"""Result-entry checks (kalos/portal/result_referee.py; Crux #1, warn-only).

A x10 / x1000 result, an impossible result and a value pasted into a second run of
the same round are flagged when recorded, with the reason and the suspected correct
value. Nothing is held or changed: the value is recorded and folded as entered. The
checks never see the prediction. On upload the sheet check reports the same problems.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kalos.portal import result_referee as rr
from kalos.portal.campaign import CampaignStore
from kalos.validation import report_dict, validate_frame
from kalos.validation.checks import check_result_duplicates, significant_digits

ROOT = Path(__file__).resolve().parents[1]
BIO = ROOT / "examples" / "synthetic_bioprocess" / "synthetic_bioprocess.csv"
TARGET = "titer_g_L"


def _history() -> tuple[float, ...]:
    return tuple(pd.read_csv(BIO)[TARGET].astype(float))


def _judge(value: float, same_round: tuple[rr.SameRoundResult, ...] = ()) -> dict:
    return rr.judge(rr.RefereeInput(value=value, target=TARGET, measured=_history(), same_round=same_round))


def test_plain_result_is_valid() -> None:
    out = _judge(0.543)
    assert out["verdict"] == "valid"
    assert out["checks"] == [] and out["held"] is False and out["mode"] == "report"


def test_surprising_but_plausible_new_best_is_valid() -> None:
    best = max(_history())
    assert _judge(round(best * 1.10, 4))["verdict"] == "valid"


@pytest.mark.parametrize("true_value,factor", [(0.445, 10), (0.306, 1000), (0.942, 1000)])
def test_x10_and_x1000_results_are_flagged_with_the_suspected_value(true_value: float, factor: int) -> None:
    out = _judge(true_value * factor)
    assert out["verdict"] == "result_magnitude_jump"
    assert out["severity"] == "warning" and out["held"] is False
    assert out["suspected_value"] == pytest.approx(true_value)
    assert f"x{factor}" in out["reason"]
    assert "kept as entered" in out["reason"].lower()


def test_negative_titer_is_physically_impossible() -> None:
    out = _judge(-0.543)
    assert out["verdict"] == "result_physically_impossible"
    assert out["suspected_value"] == pytest.approx(0.543)


def test_value_pasted_into_a_second_run_of_the_round() -> None:
    other = rr.SameRoundResult("a" * 32, 0.942, {"pH": 7.0})
    out = _judge(0.942, (other,))
    assert out["verdict"] == "result_duplicate_entry"
    assert out["suspected_value"] is None
    assert out["checks"][0]["detail"]["same_as_run"] == "a" * 32


@pytest.mark.parametrize("value", [0.0, 0.5, 1.2])  # 0, or too few significant digits
def test_imprecise_or_zero_repeats_are_not_called_pastes(value: float) -> None:
    other = rr.SameRoundResult("b" * 32, value)
    assert _judge(value, (other,))["verdict"] == "valid"


def test_a_commonly_recorded_value_is_not_called_a_paste() -> None:
    measured = (0.125,) * 10 + tuple(np.linspace(0.1, 1.0, 40).round(3))
    inp = rr.RefereeInput(0.125, TARGET, measured, (rr.SameRoundResult("c" * 32, 0.125),))
    assert rr.judge(inp)["verdict"] == "valid"


def test_too_little_history_judges_no_magnitude() -> None:
    inp = rr.RefereeInput(50.0, TARGET, (0.4, 0.5, 0.6))
    assert rr.judge(inp)["verdict"] == "valid"


def test_the_checks_are_blind_to_the_prediction() -> None:
    fields = set(rr.RefereeInput.__dataclass_fields__)
    assert not fields & {"pred", "std", "prediction", "band", "analysis"}
    tree = ast.parse(Path(rr.__file__).read_text(encoding="utf-8"))
    imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert not any(m and ("run_prediction" in m or "analysis" in m) for m in imported)


def test_significant_digits() -> None:
    assert significant_digits(0.942) == 3
    assert significant_digits(5.0) == 1
    assert significant_digits(1200.0) == 2
    assert significant_digits(1.5642) == 5
    assert significant_digits(0.0) == 0


# --- recorded through the campaign store ------------------------------------------ #


@pytest.fixture
def store(tmp_path):
    s = CampaignStore(tmp_path)
    df = pd.read_csv(BIO)
    s.seed(df, TARGET, [c for c in df.columns if c != TARGET], tenant="acme")
    yield s
    s.close()


def _recipe(i: int) -> dict:
    row = pd.read_csv(BIO).drop(columns=["run_id", TARGET]).iloc[i].to_dict()
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()}


def test_set_result_flags_a_x1000_slip_but_records_it_as_entered(store) -> None:
    (run,) = store.start([{"recipe": _recipe(3)}], tenant="acme")
    out = store.set_result(run["id"], 497.0, tenant="acme")
    assert out["result"] == 497.0  # recorded exactly as entered
    assert out["held"] is False
    assert out["verdict"]["verdict"] == "result_magnitude_jump"
    assert out["verdict"]["suspected_value"] == pytest.approx(0.497)
    state = store.get(tenant="acme")
    assert state is not None
    assert state["pending"][0]["entry_check"]["verdict"] == "result_magnitude_jump"
    folded, _target, _gen = store.plan_fold(tenant="acme")
    assert float(folded[TARGET].iloc[-1]) == 497.0  # warn-only: folded as usual


def test_set_result_flags_a_value_pasted_into_another_run_of_the_round(store) -> None:
    runs = store.start([{"recipe": _recipe(3)}, {"recipe": _recipe(5)}], tenant="acme")
    first = store.set_result(runs[0]["id"], 0.943, tenant="acme")
    second = store.set_result(runs[1]["id"], 0.943, tenant="acme")
    assert first["verdict"]["verdict"] == "valid"
    assert second["verdict"]["verdict"] == "result_duplicate_entry"
    # a correction with a reason is judged again
    fixed = store.set_result(runs[1]["id"], 0.611, tenant="acme", reason="pasted from run 1")
    assert fixed["verdict"]["verdict"] == "valid"


# --- the same problems on upload ---------------------------------------------------- #


def test_upload_flags_a_result_pasted_into_the_next_row() -> None:
    df = pd.read_csv(BIO)
    i = int(df[TARGET].idxmax())
    df.loc[i + 1, TARGET] = df.loc[i, TARGET]
    findings = check_result_duplicates(df, TARGET)
    assert len(findings) == 1
    assert set(findings[0].rows) == {i, i + 1}
    assert findings[0].detail["suspected_value"] is None


def test_upload_does_not_call_repeats_on_a_long_imprecise_sheet_pastes() -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.uniform(size=600), "y": rng.uniform(1, 3, 600).round(3)})
    df.loc[11, "y"] = df.loc[10, "y"]
    assert check_result_duplicates(df, "y") == []


def test_upload_sheet_check_is_wired_into_validation() -> None:
    df = pd.read_csv(BIO)
    i = int(df[TARGET].idxmax())
    df.loc[i + 1, TARGET] = df.loc[i, TARGET]
    report = report_dict(validate_frame(df, target=TARGET))
    assert any(f["check"] == "result_duplicate_entry" for f in report["findings"])
