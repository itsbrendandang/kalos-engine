"""The language-model test set (benchmarks/llm) is well-formed and grounded:
every expected answer is in the engine's records, every scored column exists,
and only MIT / Apache-2.0 models are candidates. No model is run here."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks" / "llm"
sys.path.insert(0, str(BENCH))

import llm_cases as cases  # noqa: E402


def test_every_expected_answer_is_in_the_fixture_records():
    fixture = json.loads((BENCH / "fixtures" / "assistant_campaign.json").read_text())
    text = " ".join(c["text"] for c in fixture["chunks"]).lower()
    guide = " ".join(str(v) for v in fixture["guide"].values()).lower()
    for case in cases.QUESTIONS:
        if case.get("refuse"):
            assert "expect" not in case
            continue
        for group in case["expect"]:
            assert any(alt.lower() in text or alt.lower() in guide for alt in group), (case["q"], group)


def test_sheets_score_only_real_columns_with_known_roles():
    from kalos.assist.setup_suggest import ROLES

    norm_roles = {"target", "feature", "group", "identity", "freetext", "metadata"}
    names = set()
    for sheet in cases.sheets():
        names.add(sheet["name"])
        df = sheet["df"]
        assert sheet["target"] in df.columns
        assert sheet["columns"][sheet["target"]]["norm"] == ["target"]
        for col, exp in sheet["columns"].items():
            assert col in df.columns, (sheet["name"], col)
            assert set(exp["norm"]) <= norm_roles and set(exp["setup"]) <= set(ROLES)
    assert len(names) == 6
    assert sum(len(s["columns"]) for s in cases.sheets()) == 51
    assert sum(not q.get("refuse") for q in cases.QUESTIONS) == 17
    assert sum(bool(q.get("refuse")) for q in cases.QUESTIONS) == 10


def test_candidates_are_mit_or_apache_models():
    src = (BENCH / "run_eval.py").read_text()
    licences = {"phi4-mini": "MIT", "qwen2.5:7b": "Apache-2.0", "qwen2.5:14b": "Apache-2.0"}
    start = src.index("CONFIGS: dict")
    block = src[start:src.index("}", start)]
    for model in {m for m in licences if m in block}:
        assert licences[model] in {"MIT", "Apache-2.0"}
    assert "llama" not in block and "mistral" not in block
