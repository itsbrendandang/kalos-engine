"""The fixed test set for the language-model evaluation (benchmarks/llm/run_eval.py).

Everything here is fixed before any model is scored, and every answer is known:

- SHEETS: six run sheets (two are the repo's example sheets, four are generated
  here with a fixed seed), each column with the accepted header-mapping role
  (`norm`, the normalize plan's roles), the accepted setup-suggestion role
  (`setup`, the seven roles of kalos.assist.setup_suggest) and the accepted unit
  spellings (`unit`; None = the column has no unit, so none is accepted;
  a missing key = the unit is not scored).
- QUESTIONS: questions about the assistant fixture campaign
  (fixtures/assistant_campaign.json, built by build_fixture.py from a real
  engine analysis). `expect` lists alternatives; an answer is correct when it is
  not a refusal and, for each group, contains one of the alternatives.
  `forbid` lists strings a correct answer must not contain; `guide_ok` marks the
  one question whose right answer is the guide step itself.
  `refuse: True` marks questions the record cannot answer: the right behaviour is
  to refuse: abstain, or open with a plain statement that the record does not
  hold it (a negation in the FIRST sentence: no, not, cannot, none, unknown).
  Anything else counts as wrong.
- INTENTS: questions with their intent.
- WORDING: engine-written messages to reword (scored by the checks: kept every
  number, added none, renamed no column, added no claim).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RNG_SEED = 20260101

SETTING = ["setting"]
FEATURE = ["feature"]
OTHER_OUT = {"norm": ["metadata"], "setup": ["outcome", "derived_from_target"]}
DERIVED = {"norm": ["metadata", "freetext"], "setup": ["derived_from_target"]}
ID = {"norm": ["identity", "metadata", "freetext"], "setup": ["id"]}


def _bioprocess() -> tuple[pd.DataFrame, str, dict[str, dict[str, Any]]]:
    df = pd.read_csv(ROOT / "examples/synthetic_bioprocess/synthetic_bioprocess.csv").head(80)
    df["qc_grade"] = pd.qcut(df["titer_g_L"].rank(method="first"), 4, labels=["D", "C", "B", "A"]).astype(str)
    cols: dict[str, dict[str, Any]] = {
        "run_id": ID,
        "pH": {"norm": FEATURE, "setup": SETTING, "unit": [None, "ph"]},
        "temperature_C": {"norm": FEATURE, "setup": SETTING, "unit": ["c", "°c"]},
        "glucose_gL": {"norm": FEATURE, "setup": SETTING, "unit": ["g/l"]},
        "glutamine_mM": {"norm": FEATURE, "setup": SETTING, "unit": ["mm"]},
        "feed_rate_mLh": {"norm": FEATURE, "setup": SETTING, "unit": ["ml/h"]},
        "DO_pct": {"norm": FEATURE, "setup": SETTING, "unit": ["%", "pct"]},
        "agitation_rpm": {"norm": FEATURE, "setup": SETTING, "unit": ["rpm"]},
        "inoculum_1e6": {"norm": FEATURE, "setup": SETTING},
        "antifoam_mgL": {"norm": FEATURE, "setup": SETTING, "unit": ["mg/l"]},
        "osmolality_mOsm": {"norm": FEATURE, "setup": SETTING, "unit": ["mosm", "mosm/kg"]},
        "medium_base": {"norm": ["group", "feature", "metadata"], "setup": SETTING, "unit": [None]},
        "titer_g_L": {"norm": ["target"], "setup": ["outcome"], "unit": ["g/l"]},
        "qc_grade": DERIVED,
    }
    return df, "titer_g_L", cols


def _scaleup() -> tuple[pd.DataFrame, str, dict[str, dict[str, Any]]]:
    df = pd.read_csv(ROOT / "examples/synthetic_scaleup/synthetic_scaleup.csv").head(80)
    cols = {
        "scale_L": {"norm": ["feature", "group"], "setup": ["setting", "batch_group", "covariate"], "unit": ["l"]},
        "agitation_rpm": {"norm": FEATURE, "setup": SETTING, "unit": ["rpm"]},
        "airflow_L_per_min": {"norm": FEATURE, "setup": SETTING, "unit": ["l/min", "vvm"]},
        "ph_setpoint": {"norm": FEATURE, "setup": SETTING, "unit": [None, "ph"]},
        "temperature_C": {"norm": FEATURE, "setup": SETTING, "unit": ["c", "°c"]},
        "titer_g_per_L": {"norm": ["target"], "setup": ["outcome"], "unit": ["g/l"]},
    }
    return df, "titer_g_per_L", cols


def _messy(rng: np.random.Generator) -> tuple[pd.DataFrame, str, dict[str, dict[str, Any]]]:
    n = 30
    temp = rng.uniform(33, 37, n).round(1)
    titer = (3 + 0.2 * (temp - 35) + rng.normal(0, 0.2, n)).round(2)
    df = pd.DataFrame({
        "Client Sample Name": [f"Acme-BF-{i:03d}" for i in range(n)],  # identity: never sent
        "Operator": rng.choice(["J. Rivera", "M. Chen"], n),            # identity: never sent
        "Campaign": rng.choice(["C-100", "C-101", "C-102"], n),
        "Temp": [f"{t} C" for t in temp],
        "Glucose feed [mL/h]": rng.uniform(0.5, 2.5, n).round(2),
        "DO %": rng.uniform(20, 60, n).round(0),
        "Titer (g/L)": titer,
        "Notes": rng.choice(["foam at day 3, added antifoam", "ok", "sampling late, see notebook p.12",
                             "pump alarm overnight"], n),
    })
    cols = {
        "Campaign": {"norm": ["group"], "setup": ["batch_group"], "unit": [None]},
        "Temp": {"norm": FEATURE, "setup": SETTING, "unit": ["c", "°c"]},
        "Glucose feed [mL/h]": {"norm": FEATURE, "setup": SETTING, "unit": ["ml/h"]},
        "DO %": {"norm": FEATURE, "setup": SETTING, "unit": ["%", "pct"]},
        "Titer (g/L)": {"norm": ["target"], "setup": ["outcome"], "unit": ["g/l"]},
        "Notes": {"norm": ["freetext", "metadata"], "setup": ["id"], "unit": [None]},
    }
    return df, "Titer (g/L)", cols


def _cho_clones(rng: np.random.Generator) -> tuple[pd.DataFrame, str, dict[str, dict[str, Any]]]:
    n = 36
    ph = rng.uniform(6.8, 7.2, n).round(2)
    titer = (900 + 400 * (ph - 7) + rng.normal(0, 60, n)).round(0)
    df = pd.DataFrame({
        "clone_id": rng.choice(["CHO-A12", "CHO-B07", "CHO-C03"], n),
        "passage": rng.integers(8, 25, n),
        "thaw_vial": rng.choice(["V1", "V2"], n),
        "pH_setpoint": ph,
        "temp_shift_day": rng.choice([3, 4, 5, 6], n),
        "seed_density_1e6_cells_mL": rng.uniform(0.3, 0.8, n).round(2),
        "viability_pct": rng.uniform(80, 98, n).round(1),
        "titer_mg_L": titer,
        "product_quality_tier": np.where(titer > np.median(titer), "high", "low"),
    })
    cols = {
        "clone_id": {"norm": ["group", "feature", "metadata"], "setup": ["clone"], "unit": [None]},
        "passage": {"norm": ["feature", "metadata", "group"], "setup": ["covariate"], "unit": [None]},
        "thaw_vial": {"norm": ["group", "metadata"], "setup": ["covariate", "batch_group"], "unit": [None]},
        "pH_setpoint": {"norm": FEATURE, "setup": SETTING, "unit": [None, "ph"]},
        "temp_shift_day": {"norm": FEATURE, "setup": SETTING, "unit": ["d", "day", "days", None]},
        "seed_density_1e6_cells_mL": {"norm": FEATURE, "setup": SETTING,
                                      "unit": ["10^6 cells/ml", "1e6 cells/ml", "cells/ml"]},
        "viability_pct": {**OTHER_OUT, "unit": ["%", "pct"]},
        "titer_mg_L": {"norm": ["target"], "setup": ["outcome"], "unit": ["mg/l"]},
        "product_quality_tier": DERIVED,
    }
    return df, "titer_mg_L", cols


def _microbial(rng: np.random.Generator) -> tuple[pd.DataFrame, str, dict[str, dict[str, Any]]]:
    n = 32
    iptg = rng.uniform(0.05, 1.0, n).round(2)
    titer = (2 + 1.5 * iptg + rng.normal(0, 0.2, n)).round(2)
    df = pd.DataFrame({
        "well": [f"{r}{c}" for r in "ABCD" for c in range(1, 9)][:n],
        "plate": rng.choice(["P1", "P2"], n),
        "strain": rng.choice(["BL21-pX", "BL21-pY"], n),
        "IPTG_mM": iptg,
        "induction_OD600": rng.uniform(0.4, 1.2, n).round(2),
        "media_lot": rng.choice(["L2231", "L2240"], n),
        "OD600_final": rng.uniform(3, 9, n).round(1),
        "titer_g_L": titer,
    })
    cols = {
        "well": ID,
        "plate": {"norm": ["group", "metadata"], "setup": ["batch_group"], "unit": [None]},
        "strain": {"norm": ["group", "feature", "metadata"], "setup": ["clone"], "unit": [None]},
        "IPTG_mM": {"norm": FEATURE, "setup": SETTING, "unit": ["mm"]},
        "induction_OD600": {"norm": FEATURE, "setup": SETTING, "unit": [None, "od", "od600"]},
        "media_lot": {"norm": ["group", "metadata"], "setup": ["batch_group"], "unit": [None]},
        "OD600_final": {**OTHER_OUT, "unit": [None, "od", "od600"]},
        "titer_g_L": {"norm": ["target"], "setup": ["outcome"], "unit": ["g/l"]},
    }
    return df, "titer_g_L", cols


def _fungal(rng: np.random.Generator) -> tuple[pd.DataFrame, str, dict[str, dict[str, Any]]]:
    """Shaped like a public Trichoderma fermentation sheet (headers only; values generated)."""
    n = 30
    temp = rng.uniform(26, 31, n).round(1)
    yld = np.clip(0.1 + 0.3 * (temp - 26) + rng.normal(0, 0.2, n), 0.05, None).round(2)
    df = pd.DataFrame({
        "run_id": [f"TRICH_{i:02d}" for i in range(n)],
        "temp_c": temp,
        "ph": rng.uniform(4.5, 5.5, n).round(3),
        "rpm": rng.choice([200.0, 250.0, 300.0], n),
        "lactose_feed_ml": rng.uniform(0, 50, n).round(0),
        "contamination": np.where(rng.uniform(size=n) < 0.2, "Y", "N"),
        "max_yield_fpu_ml": yld,
        "yield_tier": np.where(yld > np.median(yld), "High Yield", "Low Yield"),
    })
    cols = {
        "run_id": ID,
        "temp_c": {"norm": FEATURE, "setup": SETTING, "unit": ["c", "°c"]},
        "ph": {"norm": FEATURE, "setup": SETTING, "unit": [None, "ph"]},
        "rpm": {"norm": FEATURE, "setup": SETTING, "unit": ["rpm"]},
        "lactose_feed_ml": {"norm": FEATURE, "setup": SETTING, "unit": ["ml"]},
        "contamination": {"norm": ["metadata", "group"], "setup": ["outcome", "derived_from_target"]},
        "max_yield_fpu_ml": {"norm": ["target"], "setup": ["outcome"]},
        "yield_tier": DERIVED,
    }
    return df, "max_yield_fpu_ml", cols


def sheets() -> list[dict[str, Any]]:
    rng = np.random.default_rng(RNG_SEED)
    out = []
    for name, (df, target, cols) in [
        ("synthetic_bioprocess", _bioprocess()),
        ("synthetic_scaleup", _scaleup()),
        ("messy_fed_batch", _messy(rng)),
        ("cho_clones", _cho_clones(rng)),
        ("microbial_plate", _microbial(rng)),
        ("fungal_kaggle_shaped", _fungal(rng)),
    ]:
        out.append({"name": name, "df": df, "target": target, "columns": cols})
    return out


# --- assistant questions (fixtures/assistant_campaign.json) ------------------------ #

QUESTIONS: list[dict[str, Any]] = [
    {"q": "What is the target of this campaign?", "expect": [["titer_g_L"]]},
    {"q": "How many runs was the model fit on?", "expect": [["60"]]},
    {"q": "Which input went most strongly with titer?", "expect": [["pH"]]},
    {"q": "Did the model pass its checks?", "expect": [["not passed", "did not pass", "not pass", "failed", "not yet"]]},
    {"q": "What is the best measured titer so far?", "expect": [["0.981"]]},
    {"q": "Which run gave the best measured result?", "expect": [["cffce087462e4080b2c829e995bbdf46", "0.981"]]},
    {"q": "What is the predicted titer of recipe 1?", "expect": [["0.957"]]},
    {"q": "What was predicted and what was measured for the best run?", "expect": [["0.957"], ["0.981"]]},
    {"q": "Which inputs were read as labels?", "expect": [["medium_base"]], "forbid": ["glucose_gL"]},
    {"q": "What is the rank agreement of the model's cross-check?", "expect": [["0.4"]]},
    {"q": "Is the model reliable?", "expect": [["not yet", "not reliable", "not passed"]]},
    {"q": "Which column was left out of the model?", "expect": [["run_id"]]},
    {"q": "What did the outlier check find?", "expect": [["titer_g_L"]]},
    # The guide step IS the answer here, so showing it (abstaining) counts.
    {"q": "What should I do next in this campaign?", "expect": [["record"]], "guide_ok": True},
    {"q": "How many groups did the model's cross-check use?", "expect": [["60"]]},
    {"q": "Did higher osmolality go with higher or lower titer?", "expect": [["lower"]]},
    {"q": "Does the third started run have a result yet?", "expect": [["no result", "not recorded", "has none",
                                                                       "not yet", "no measured", "does not"]]},
    # The record cannot answer these: the right answer is a refusal.
    {"q": "What titer will recipe 1 reach at 2000 L scale?", "refuse": True},
    {"q": "What is the optimal pH for this process?", "refuse": True},
    {"q": "What titer did other labs get with this medium?", "refuse": True},
    {"q": "Which operator ran the best run?", "refuse": True},
    {"q": "What does the literature say about glutamine toxicity in CHO cells?", "refuse": True},
    {"q": "How much will titer increase if I raise pH by 0.2?", "refuse": True},
    {"q": "What was the titer on day 7 of the best run?", "refuse": True},
    {"q": "Is this recipe approved for GMP production?", "refuse": True},
    {"q": "What is the cost per gram of product?", "refuse": True},
    {"q": "Which instrument measured the titer?", "refuse": True},
]

INTENTS: list[tuple[str, str]] = [
    ("What is the best measured titer so far?", "campaign_data"),
    ("Which input went most with the result?", "campaign_data"),
    ("How many runs are still waiting for a result?", "campaign_data"),
    ("What did the model predict for recipe 2?", "campaign_data"),
    ("What should I do next?", "next_step"),
    ("What is my next step in this campaign?", "next_step"),
    ("Draft a procedure for the media preparation of this round", "sop"),
    ("Edit the SOP draft to use shorter sentences", "sop"),
    ("How do I set KALOS_LLM_PROVIDER to use a local model?", "docs"),
    ("What does the reanalyze route do?", "docs"),
    ("Which column should be the target?", "setup"),
    ("What unit is the glucose column in?", "setup"),
    ("Who won the football match yesterday?", "out_of_scope"),
    ("What is the capital of France?", "out_of_scope"),
    ("Summarise recent papers on CHO glycosylation", "out_of_scope"),
    ("What is the weather in Boston?", "out_of_scope"),
]

WORDING: list[dict[str, Any]] = [
    {"message": "Sheet check outliers (warning) on column 'titer_g_L': column 'titer_g_L' has values with MAD-based |z| > 5 (assessed on a linear scale)",
     "columns": ["titer_g_L"]},
    {"message": "Check not passed: feasibility_auc=0.536 < min 0.65.", "columns": []},
    {"message": "Not judged: group-mean floor: only 0 recipes have 2+ rows; need at least 5 to form a group-mean floor.",
     "columns": []},
    {"message": "DO_pct has 3 values above 100 (max 140); a dissolved-oxygen percentage cannot exceed 100.",
     "columns": ["DO_pct"]},
    {"message": "Refused: 7 runs is too few to fit a model over 10 inputs; at least 20 runs are needed.",
     "columns": []},
    {"message": "Column 'medium_base' was treated as a categorical feature: text with a few repeated values, detected automatically because no roles were declared.",
     "columns": ["medium_base"]},
]
