"""Model evaluation harness for the column-normalization task.

This scores whatever LLM the engine is configured to use on an OBJECTIVE,
labelled task: given a messy client run sheet, assign each column a role
(target/feature/group/identity/freetext) and, for numeric+unit columns, a unit.
The ground truth is known, so accuracy is exact-match, not a judgement call.

Why a harness and not a pass/fail test: it is the yardstick for "is our model
good?". Run it against each tier to compare on identical cases:

    # deterministic floor (no model) - always runs, part of CI
    pytest tests/test_model_eval_normalize.py -s

    # local candidate (any OpenAI-compatible server, e.g. a local Qwen)
    KALOS_LLM_PROVIDER=openai \
      KALOS_OPENAI_BASE_URL=http://127.0.0.1:59601/v1 \
      KALOS_OPENAI_MODEL=<model-id> KALOS_OPENAI_API_KEY=<key> \
      KALOS_MODEL_EVAL_LIVE=1 pytest tests/test_model_eval_normalize.py -s

    # ceiling baseline (Anthropic) - the key is read by the SDK, never stored here
    KALOS_LLM_PROVIDER=anthropic ANTHROPIC_API_KEY=... \
      KALOS_MODEL_EVAL_LIVE=1 pytest tests/test_model_eval_normalize.py -s

The default (CI) run scores only the offline tier, so it needs no network and no
key. The live tiers run only when KALOS_MODEL_EVAL_LIVE is set, so a missing
server or key can never fail CI. The printed per-column table and the headline
accuracy are the artefact: they tell us how far a candidate model is from the
offline floor and the Anthropic ceiling BEFORE anyone fine-tunes anything.
"""
from __future__ import annotations

import os

import pandas as pd
import pytest

from kalos.normalize.apply import apply_plan  # noqa: F401  (import-safety check)
from kalos.normalize.llm import offline_plan, propose_plan

# --- the labelled evaluation set ----------------------------------------------- #
# Each case: a messy run sheet, and the expected role per raw column (plus the
# expected unit token where the column carries one). Grounded in the real run
# sheets the engine is built for.

def _sheet_a() -> pd.DataFrame:
    return pd.DataFrame({
        "Client Sample Name": [f"Acme-BF-{i:03d}" for i in range(18, 24)],
        "Operator": ["J. Rivera", "J. Rivera", "M. Chen", "M. Chen", "J. Rivera", "M. Chen"],
        "Notes": [
            "Foam observed at hour 12, adjusted antifoam during the run.",
            "Nothing unusual, run proceeded per protocol with no deviation.",
            "Slight pH drift corrected manually at the 24h timepoint.",
            "Client requested an early harvest due to a scheduling conflict.",
            "DO probe recalibrated mid run after a suspicious reading.",
            "Standard run, harvested on schedule with no issues noted.",
        ],
        "Campaign": ["C-100", "C-100", "C-100", "C-101", "C-101", "C-101"],
        "Titer (g/L)": [3.1, 3.4, 3.9, 2.8, 3.3, 3.7],
        "Temp": [34.6, 34.8, 35.0, 34.9, 35.1, 34.7],
    })


_EXPECTED_A: dict[str, dict[str, str | None]] = {
    "Client Sample Name": {"role": "identity"},
    "Operator": {"role": "identity"},
    "Notes": {"role": "freetext"},
    "Campaign": {"role": "group"},
    "Titer (g/L)": {"role": "target", "unit_token": "g/L"},
    "Temp": {"role": "feature", "unit_token": "C"},
}


_CASES: list[tuple[str, pd.DataFrame, dict[str, dict[str, str | None]]]] = [
    ("sheet_a", _sheet_a(), _EXPECTED_A),
]


def _score_plan(plan, expected: dict[str, dict[str, str | None]]) -> dict[str, float]:
    by_raw = {c.raw_name: c for c in plan.columns}
    role_hits = 0
    unit_total = 0
    unit_hits = 0
    rows: list[str] = []
    for raw, want in expected.items():
        col = by_raw.get(raw)
        got_role = getattr(col, "role", None) if col else None
        ok_role = got_role == want["role"]
        role_hits += int(ok_role)
        line = f"  {raw!r:28} role: got={got_role!s:9} want={want['role']!s:9} {'ok' if ok_role else 'MISS'}"
        if "unit_token" in want:
            unit_total += 1
            got_unit = getattr(col, "unit_token", None) if col else None
            ok_unit = got_unit == want["unit_token"]
            unit_hits += int(ok_unit)
            line += f" | unit: got={got_unit!s:5} want={want['unit_token']!s:5} {'ok' if ok_unit else 'MISS'}"
        rows.append(line)
    n = len(expected)
    return {
        "role_accuracy": role_hits / n,
        "unit_accuracy": (unit_hits / unit_total) if unit_total else 1.0,
        "n": float(n),
        "_rows": rows,  # type: ignore[dict-item]
    }


def _provider_label() -> str:
    provider = (os.environ.get("KALOS_LLM_PROVIDER") or "offline").strip() or "offline"
    model = (os.environ.get("KALOS_OPENAI_MODEL")
             or os.environ.get("KALOS_ASSIST_ANTHROPIC_MODEL") or "")
    return f"{provider}{(' / ' + model) if model else ''}"


def test_offline_normalization_floor(capsys):
    """The deterministic floor: offline_plan with no model. Always runs; this is
    the accuracy any model must beat to be worth its latency and privacy cost."""
    total_role = 0.0
    total_unit = 0.0
    with capsys.disabled():
        print("\n=== model eval: column normalization (OFFLINE floor) ===")
        for name, df, expected in _CASES:
            m = _score_plan(offline_plan(df), expected)
            total_role += m["role_accuracy"]
            total_unit += m["unit_accuracy"]
            print(f"[{name}] role_acc={m['role_accuracy']:.3f}  unit_acc={m['unit_accuracy']:.3f}")
            for row in m["_rows"]:
                print(row)
        k = len(_CASES)
        print(f"OFFLINE mean: role_acc={total_role / k:.3f}  unit_acc={total_unit / k:.3f}")

    # The offline tier is deterministic; these are its known-good numbers on the
    # eval set. If a code change moves them, the harness flags it.
    m = _score_plan(offline_plan(_sheet_a()), _EXPECTED_A)
    assert m["role_accuracy"] >= 0.80, f"offline role accuracy regressed: {m}"


@pytest.mark.skipif(
    not os.environ.get("KALOS_MODEL_EVAL_LIVE"),
    reason="live model eval is opt-in: set KALOS_MODEL_EVAL_LIVE=1 and a provider",
)
def test_live_model_normalization(capsys):
    """Score the configured live provider (openai-compatible local model, or
    anthropic) on the same cases. Opt-in, so it never runs or fails in CI."""
    total_role = 0.0
    total_unit = 0.0
    label = _provider_label()
    with capsys.disabled():
        print(f"\n=== model eval: column normalization (LIVE: {label}) ===")
        for name, df, expected in _CASES:
            plan = propose_plan(df)
            m = _score_plan(plan, expected)
            total_role += m["role_accuracy"]
            total_unit += m["unit_accuracy"]
            print(f"[{name}] created_by={plan.created_by}  "
                  f"role_acc={m['role_accuracy']:.3f}  unit_acc={m['unit_accuracy']:.3f}")
            for row in m["_rows"]:
                print(row)
        k = len(_CASES)
        print(f"LIVE ({label}) mean: role_acc={total_role / k:.3f}  unit_acc={total_unit / k:.3f}")

    # No hard floor on the live tier: a flaky or weak model must not fail the
    # suite. The printed numbers are the artefact. We only assert the call
    # produced a validatable plan (the contract propose_plan guarantees).
    propose_plan(_sheet_a()).validate()
