"""Every code the guide can emit has a why-line (kalos/assist/why.py).

The codes are read from the engine source itself (every `code="..."` an upload
refusal raises, every `check="..."` a finding carries, every model check), so a
new code without a why-line fails here instead of reaching the user as a
fallback sentence.
"""
from __future__ import annotations

import re
from pathlib import Path

from kalos.assist.guide import _GATE_TEXT
from kalos.assist.why import ENGINE_ONLY, GLOSSARY, MODEL_CHECKS, WHY, why_for
from kalos.assist.words import banned_in, claims_in

ROOT = Path(__file__).resolve().parents[1] / "kalos"
# Codes that are HTTP-layer refusals, never a sheet or model problem.
NOT_GUIDE_CODES = {"unauthenticated"}


def _grep(pattern: str, *dirs: str) -> set[str]:
    found: set[str] = set()
    for d in dirs:
        for path in (ROOT / d).rglob("*.py"):
            found.update(re.findall(pattern, path.read_text(encoding="utf-8")))
    return found


def test_every_refusal_code_has_a_why_line():
    codes = _grep(r'code="([a-z_]+)"', "portal", "validation") - NOT_GUIDE_CODES
    codes |= {"fit_failed", "parse_error"}  # set as dict literals in app.py
    missing = sorted(codes - set(WHY))
    assert not missing, missing


def test_every_finding_check_has_a_why_line():
    checks = _grep(r'check="([a-z_]+)"', "portal", "validation")
    missing = sorted(checks - set(WHY))
    assert not missing, missing


def test_every_model_check_has_a_why_line():
    assert set(_GATE_TEXT) <= set(MODEL_CHECKS)


def test_every_guide_code_has_a_why_line():
    for code in ("run_waiting", "record_partial", "learn_ready", "setup_no_sheet", "plan_ready",
                 "empty_batch", "not-yet", "insufficient", "unmeasured", "model_check",
                 "fewer_distinct_candidates"):
        assert code in WHY, code


# Glossary lines that still use a claim word ("will", "caused"). Mirrored
# verbatim, so they are pinned here and sent to the glossary owner; the list
# may only shrink.
KNOWN_GLOSSARY_CLAIMS = {
    "group_mean_margin", "feasibility_auc", "brier", "provenance_metadata", "target_constant",
    "target_constant_upload", "run_waiting", "column_unique", "user_excluded",
}


def test_why_lines_are_plain():
    for code, line in WHY.items():
        assert line.endswith("."), code
        assert banned_in(line) == [], code
        assert "_" not in line, code
    for code, line in ENGINE_ONLY.items():
        assert claims_in(line) == [], code
    with_claims = {code for code, line in GLOSSARY.items() if claims_in(line)}
    assert with_claims <= KNOWN_GLOSSARY_CLAIMS, with_claims - KNOWN_GLOSSARY_CLAIMS


def test_engine_only_does_not_shadow_the_glossary():
    assert not set(ENGINE_ONLY) & set(GLOSSARY)


def test_unknown_code_falls_back():
    assert why_for("does_not_exist") == why_for(None) == GLOSSARY["generic"]
    assert why_for("too_many_fit_rows") == GLOSSARY["too_many_fit_rows"]
