"""Every finding code and every refusal code the engine emits has a plain
why-line in the engine (kalos/portal/why_lines.py), and every finding in an
analysis carries it as `why`."""
from __future__ import annotations

import io
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kalos.portal.why_lines import WHY, why_for

KALOS = Path(__file__).resolve().parents[1] / "kalos"
# `code="x"`, `check="x"`, `code = "x"` (class attribute) and `"code": "x"`.
_CODE = re.compile(r"""(?:\b(?:code|check|failed_check)\s*=\s*|["'](?:code|check)["']\s*:\s*)["']([a-z][a-z0-9_]*)["']""")
# `coded_body(message, "x")` (kalos/portal/why_lines.py).
_CODED_BODY = re.compile(r"""coded_body\((?:[^()]|\([^()]*\))*?,\s*["']([a-z][a-z0-9_]*)["'],?\s*\)""", re.S)
# `_refusal("x", ...)` (kalos/portal/scale_routes.py) and the generic envelope codes
# by HTTP status (`_STATUS_CODES` in kalos/portal/app.py: `400: "bad_request"`).
_REFUSAL = re.compile(r"""_refusal\(\s*["']([a-z][a-z0-9_]*)["']""")
_STATUS = re.compile(r"""^\s*\d{3}:\s*["']([a-z][a-z0-9_]*)["'],""", re.M)
# Codes built from tables rather than literals: the verdict states and codes
# (kalos/portal/verdict.py) and the role-safety finding codes.
_TABLE_CODES = {
    "reliable", "not_yet", "refused", "insufficient", "no_ordering",
    "after_target_in_sheet", "row_number_column", "headerless_column",
    "outcome_like_name", "possibly_measured_after_run", "category_not_modeled",
}


def _static_codes() -> set[str]:
    codes: set[str] = set()
    for path in KALOS.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        codes |= set(_CODE.findall(text)) | set(_CODED_BODY.findall(text))
        codes |= set(_REFUSAL.findall(text)) | set(_STATUS.findall(text))
    return codes | _TABLE_CODES


def test_every_code_in_the_engine_source_has_a_why_line():
    codes = _static_codes()
    # The scan must see the codes it is meant to cover.
    for expected in ("invalid_linear_constraint", "roles_not_applied", "units_consistency",
                     "search_box", "too_few_groups", "target_constant", "inputs_changed",
                     "forbidden", "target_ratio_hard_cap", "invalid_target"):
        assert expected in codes
    missing = sorted(c for c in codes if why_for(c) is None)
    assert missing == []


def test_the_listed_codes_have_lines():
    for code in (
        "invalid_linear_constraint", "infeasible_linear_constraints",
        "linear_constraints_with_categorical", "invalid_specs", "invalid_allow_beyond",
        "invalid_exploration_design", "invalid_per_recipe_noise", "row_number_column",
        "after_target_in_sheet", "outcome_like_name", "possibly_measured_after_run",
        "headerless_column", "category_not_modeled", "search_box", "inputs_changed",
        "cv_ci_unavailable", "near_constant_input", "coded_column",
        "concentration_before_target", "feature_looks_outcome_like",
        "after_target_high_correlation", "roles_not_applied", "model_refused", "no_ordering",
        "too_few_groups", "too_few_rows", "stopping_advice_only", "specs_treated_independently",
    ):
        assert why_for(code), code


def test_lines_are_plain_sentences():
    banned = re.compile(r"\b(spearman|p-value|regression|gaussian|kernel|json|api|http|"
                        r"null|exception|endpoint|parameter)\b", re.I)
    for code, line in WHY.items():
        assert line.strip().endswith((".", "!")) and line[0].isupper(), code
        assert not banned.search(line), code


def _messy_sheet(seed: int) -> pd.DataFrame:
    """A sheet that trips many sheet checks and column findings at once."""
    rng = np.random.default_rng(seed)
    n = 14
    df = pd.DataFrame({
        "Row": np.arange(1, n + 1),
        "Glucose_g_L": rng.uniform(5, 30, n).round(2),
        "Temp_C": rng.uniform(28, 34, n).round(1),
        "pH": np.r_[np.full(n - 1, 7.0), 7.001],
        "OD600": rng.uniform(1, 9, n).round(2),
        "Media": [f"lot{i}" for i in range(n)],
        "Titer": rng.uniform(10, 60, n).round(2),
        "Final_yield": rng.uniform(0, 1, n).round(3),
        "Lactate_g_L": rng.uniform(0, 5, n).round(2),
    })
    df.loc[3, "Glucose_g_L"] = np.nan
    df.loc[5, "Temp_C"] = 900.0
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    return df


def _assert_findings_have_why(result: dict) -> set[str]:
    seen: set[str] = set()
    for f in result["validation"]["findings"]:
        assert f.get("why"), f
        seen.add(f["code"])
    for f in result.get("campaign_findings") or []:
        assert f.get("why"), f
        seen.add(f["code"])
    return seen


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_preflight_findings_carry_their_why(seed):
    from kalos.portal.analysis import _analyze

    result = _analyze(_messy_sheet(seed), "Titer", preflight_only=True)
    seen = _assert_findings_have_why(result)
    assert len(seen) >= 4


def test_full_analysis_and_campaign_findings_carry_their_why(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from kalos.portal import app as portal_module
    from kalos.portal import campaign as campaign_module
    from kalos.portal.app import app
    from kalos.portal.campaign import CampaignStore, get_campaign_store

    monkeypatch.setenv("KALOS_STATE_DIR", str(tmp_path))
    store = CampaignStore(tmp_path)
    monkeypatch.setattr(portal_module, "_STATE_DIR", tmp_path)
    monkeypatch.setattr(portal_module, "_LATEST_DIR", tmp_path / "latest")
    monkeypatch.setattr(campaign_module, "_STORE", store)
    app.dependency_overrides[get_campaign_store] = lambda: store
    rng = np.random.default_rng(7)
    n = 12
    df = pd.DataFrame({"A": rng.uniform(0, 1, n), "B": rng.uniform(0, 1, n),
                       "C": rng.uniform(0, 1, n)})
    df["Titer"] = 3 * df["A"] - df["B"] + rng.normal(0, 0.1, n)
    csv = df.round(4).to_csv(index=False).encode()
    try:
        c = TestClient(app)

        def run(features):
            r = c.post("/api/run", files={"file": ("s.csv", io.BytesIO(csv), "text/csv")},
                       data={"target": "Titer",
                             "roles": json.dumps({"target": "Titer", "features": features})})
            assert r.status_code == 200, r.text
            return r.json()

        first = run(["A", "B"])
        seen = _assert_findings_have_why(first)
        assert "search_box" in seen
        second = run(["A", "B", "C"])
        assert "inputs_changed" in _assert_findings_have_why(second)
        # A coded 400 refusal carries the line next to its code.
        bad = c.post("/api/run", files={"file": ("s.csv", io.BytesIO(csv), "text/csv")},
                     data={"target": "Titer",
                           "roles": json.dumps({"target": "Titer", "allow_beyond": 5})})
        assert bad.status_code == 400
        assert bad.json()["code"] == "invalid_allow_beyond"
        assert bad.json()["why"] == WHY["invalid_allow_beyond"]
    finally:
        app.dependency_overrides.pop(get_campaign_store, None)
