"""The skill tool table and its single invocation path (kalos/skills/tools.py,
kalos/skills/bindings.py): every tool is bound to a real engine symbol, every
call validates its arguments and result, carries a data label the skill
declares, and writes a receipt for ok, refused and failed outcomes."""
from __future__ import annotations

import base64
import dataclasses
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pydantic")
pytest.importorskip("yaml")

from kalos.skills import bindings, load_builtin
from kalos.skills.receipts import ReceiptStore, sha256_json, table_sha256
from kalos.skills.tools import (
    TOOLS,
    ToolCallError,
    ToolContext,
    ToolRefusal,
    check_tools,
    guard_measured_write,
    invoke,
)

TARGET = "titer_g_L"
FEATS = ["temp_C", "pH"]


def _df(n: int = 16, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "batch_id": [f"B{i // 2}" for i in range(n)],
        "temp_C": np.repeat(rng.uniform(30, 38, n // 2), 2).round(2),
        "pH": np.repeat(rng.uniform(6, 7.5, n // 2), 2).round(2),
    })
    df[TARGET] = (0.3 * df.temp_C - 2 * (df.pH - 6.8) ** 2 + rng.normal(0, 0.2, n)).round(3)
    return df


ROWS = _df().to_dict(orient="records")
MODEL = {"rows": ROWS, "target": TARGET, "features": FEATS}


@pytest.fixture
def reg():
    return load_builtin()


@pytest.fixture
def ctx(tmp_path):
    pytest.importorskip("fastapi")
    from kalos.portal.campaign import CampaignStore

    store = CampaignStore(tmp_path / "state")
    cid = store.create_campaign(tenant="acme", name="c")["id"]
    store.seed(_df(), TARGET, FEATS, tenant="acme", campaign_id=cid)
    c = ToolContext(tenant="acme", subject="alice@acme", scopes=frozenset({"read", "write"}),
                    store=store, receipts=ReceiptStore(tmp_path / "receipts"))
    c.extras["cid"] = cid
    try:
        yield c
    finally:
        store.close()


def _receipts(ctx):
    return ctx.receipts.list_for(ctx.tenant)


# --- the table ------------------------------------------------------------------ #


def test_tool_table_resolves_statically_and_covers_every_skill(reg):
    assert check_tools(reg.all()) == []
    used = {t for s in reg.all() for t in s.meta.tools}
    assert used == set(TOOLS), "every registered tool is used by a skill, and vice versa"
    assert len(TOOLS) == 30


def test_tool_labels_match_skill_outputs(reg):
    for s in reg.all():
        for name in s.meta.tools:
            assert TOOLS[name].label in s.meta.output_labels, (s.slug, name)
    assert TOOLS["engine.propose_batch"].label == "predicted"
    assert TOOLS["engine.scale_build_readout"].label == "predicted"
    assert TOOLS["engine.ode_model"].label == "simulated"


def test_check_tools_reports_unregistered_and_mismatched(reg):
    sk = reg.get("driver-ranking", "t0")
    bad = dataclasses.replace(sk, meta=sk.meta.model_copy(update={"tools": ["engine.nope"]}))
    assert any("not registered" in p for p in check_tools([bad]))
    bad2 = dataclasses.replace(sk, meta=sk.meta.model_copy(update={"tools": ["engine.ode_model"]}))
    assert any("emits simulated" in p for p in check_tools([bad2]))


def test_importing_tools_does_not_load_torch():
    out = subprocess.run(
        [sys.executable, "-c",
         "import sys; import kalos.skills.tools, kalos.skills.bindings, kalos.skills.receipts;"
         "from kalos.skills import load_builtin; from kalos.skills.tools import check_tools;"
         "assert check_tools(load_builtin().all()) == [];"
         "print('torch' in sys.modules)"],
        capture_output=True, text=True, check=True).stdout.strip()
    assert out == "False"


# --- every torch-free tool runs end to end ---------------------------------------- #

CSV = _df().to_csv(index=False)
TORCH_FREE_CALLS = [
    ("column-role-inference", "engine.guess_role",
     {"header": "Titer (g/L)", "is_numeric": True, "parse_rate": 1.0}),
    ("column-role-inference", "engine.offline_plan", {"rows": ROWS, "target_column": TARGET}),
    ("column-role-inference", "engine.column_provenance", MODEL),
    ("replicate-noise-estimate", "engine.noise_report", MODEL),
    ("replicate-noise-estimate", "engine.estimate_noise_floor", MODEL),
    ("replicate-noise-estimate", "engine.heteroscedasticity_report", MODEL),
    ("run-sheet-checks", "engine.parse_upload", {"csv_text": CSV, "filename": "x.csv"}),
    ("run-sheet-checks", "engine.validate_frame", {"rows": ROWS, "target": TARGET}),
    ("target-readiness-check", "engine.assess_target_variation",
     {"values": [r[TARGET] for r in ROWS]}),
    ("unit-normalisation", "engine.parse_value", {"raw": "12 mg/mL"}),
    ("unit-normalisation", "engine.convert_unit", {"value": 12, "unit_token": "mg/mL"}),
    ("unit-normalisation", "engine.apply_unit_conversions",
     {"rows": [{"t": "1 g/L"}, {"t": "2 g/L"}, {"t": "3 g/L"}]}),
    ("search-box-definition", "engine.build_search_box", MODEL),
    ("search-box-definition", "engine.validate_constraint", {"constraint": None}),
    ("driver-ranking", "engine.bootstrap_spearman", {**MODEL, "n_boot": 50}),
    ("driver-ranking", "engine.rank_drivers", {**MODEL, "n_boot": 50}),
    ("identifier-anonymisation", "engine.anonymizer", {"rows": ROWS}),
    ("identifier-anonymisation", "engine.anonymize_parsed",
     {"content_base64": base64.b64encode(CSV.encode()).decode(), "filename": "x.csv"}),
    ("trust-report", "engine.check_gates", {"stats": {"surrogate_spearman": 0.5}}),
    ("campaign-provenance-report", "engine.run_record_build", {"csv_text": CSV, "target": TARGET}),
    ("campaign-provenance-report", "engine.audit_entries", {"limit": 10}),
    ("virtual-run-simulation", "engine.ode_model", {"recipes": [[0.3, 0.5, 10.0]], "t_end_h": 48}),
]


@pytest.mark.parametrize("slug,tool,args", TORCH_FREE_CALLS, ids=[c[1] for c in TORCH_FREE_CALLS])
def test_tool_runs_and_is_receipted(reg, ctx, slug, tool, args):
    sk = reg.get(slug, "acme")
    out = invoke(sk, tool, args, ctx)
    assert out.outcome == "ok", out.refusal
    r = out.receipt
    assert r.outcome == "ok" and r.tool == tool and r.symbol == TOOLS[tool].symbol
    assert r.skill_slug == slug and r.skill_content_hash == sk.content_hash
    assert r.tenant == "acme" and r.subject == "alice@acme"
    assert [o.label for o in r.outputs] == [TOOLS[tool].label]
    assert r.outputs[0].sha256 == sha256_json(out.data)
    assert r.args_sha256 == sha256_json(TOOLS[tool].args_model.model_validate(args)
                                        .model_dump(mode="json"))
    if "rows" in args:
        assert r.input_sha256 == table_sha256(args["rows"]) and r.input_rows == len(args["rows"])
    assert ctx.receipts.get("acme", r.receipt_id) == r


def test_campaign_tools_read_the_callers_campaign(reg, ctx):
    cid = ctx.extras["cid"]
    s = invoke(reg.get("campaign-status", "acme"), "engine.campaign_summary",
               {"campaign_id": cid}, ctx)
    assert s.outcome == "ok" and s.data["summary"]["target"] == TARGET
    assert s.receipt.campaign_id == cid
    st = invoke(reg.get("campaign-status", "acme"), "engine.derive_stage", {"campaign_id": cid}, ctx)
    assert st.outcome == "ok" and "stage" in st.data["stage"]
    n = invoke(reg.get("replicate-noise-estimate", "acme"), "engine.noise_report",
               {"campaign_id": cid}, ctx)
    assert n.outcome == "ok" and n.data["report"]["n_rows"] == 16


def test_campaign_summary_strips_predictions(reg, ctx):
    cid = ctx.extras["cid"]
    ctx.store.start([{"recipe": {"temp_C": 33.0, "pH": 6.5}, "pred": 9.9, "std": 0.4,
                      "mode": "explore", "reason": "x"}], tenant="acme", campaign_id=cid)
    s = invoke(reg.get("campaign-status", "acme"), "engine.campaign_summary",
               {"campaign_id": cid}, ctx)
    for run in s.data["summary"]["pending"]:
        assert "pred" not in run and "std" not in run


def test_other_tenants_campaign_is_a_receipted_refusal(reg, ctx):
    other = ctx.store.create_campaign(tenant="globex", name="g")["id"]
    out = invoke(reg.get("campaign-status", "acme"), "engine.campaign_summary",
                 {"campaign_id": other}, ctx)
    assert out.outcome == "refused" and out.refusal.code == "not_found"
    assert out.receipt.outcome == "refused" and out.receipt.outputs == []


def test_engine_refusal_is_receipted(reg, ctx):
    rows = [{"a": "x"}]
    out = invoke(reg.get("run-sheet-checks", "acme"), "engine.parse_upload",
                 {"csv_text": "\n\n"}, ctx)
    assert out.outcome == "refused" and out.receipt.refusal is not None
    out2 = invoke(reg.get("replicate-noise-estimate", "acme"), "engine.noise_report",
                  {"rows": rows, "target": "a"}, ctx)
    assert out2.outcome == "refused"


def test_strict_mode_refuses_on_error_findings(reg, ctx):
    rows = [dict(r) for r in ROWS]
    rows[0]["pH"] = 25.0  # outside the physical range of pH
    sk = reg.get("run-sheet-checks", "acme")
    warn = invoke(sk, "engine.validate_frame", {"rows": rows, "mode": "warn"}, ctx)
    assert warn.outcome == "ok"
    has_error = warn.data["report"]["counts"]["error"] > 0
    strict = invoke(sk, "engine.validate_frame", {"rows": rows, "mode": "strict"}, ctx)
    assert strict.outcome == ("refused" if has_error else "ok")
    assert has_error, "pH 25 should be a blocking physical-bounds finding"
    assert strict.refusal.code == "blocking_findings"


def test_simulated_output_is_labelled_per_row(reg, ctx):
    out = invoke(reg.get("virtual-run-simulation", "acme"), "engine.ode_model",
                 {"recipes": [[0.3, 0.5, 10.0], [0.2, 1.0, 5.0]]}, ctx)
    assert out.label == "simulated"
    assert all(e["data_type"] == "simulated" for e in out.data["endpoints"])
    assert out.data["endpoints"][0]["final_biomass"] > 0.1


def test_deterministic_tools_repeat_bit_for_bit(reg, ctx):
    sk = reg.get("driver-ranking", "acme")
    a = invoke(sk, "engine.rank_drivers", {**MODEL, "n_boot": 50, "seed": 3}, ctx)
    b = invoke(sk, "engine.rank_drivers", {**MODEL, "n_boot": 50, "seed": 3}, ctx)
    assert a.receipt.outputs[0].sha256 == b.receipt.outputs[0].sha256
    assert a.receipt.args_sha256 == b.receipt.args_sha256
    assert a.receipt.receipt_id != b.receipt.receipt_id
    assert a.receipt.seed == 3


# --- gatekeeping ------------------------------------------------------------------ #


def test_planned_skill_has_nothing_to_run(reg, ctx):
    with pytest.raises(ToolCallError) as e:
        invoke(reg.get("hplc-peak-summary", "acme"), "engine.noise_report", MODEL, ctx)
    assert e.value.code == "planned"


def test_tool_must_belong_to_skill(reg, ctx):
    with pytest.raises(ToolCallError) as e:
        invoke(reg.get("driver-ranking", "acme"), "engine.noise_report", MODEL, ctx)
    assert e.value.code == "tool_not_in_skill"
    assert _receipts(ctx) == []


def test_scope_is_enforced(reg, ctx):
    ro = dataclasses.replace(ctx, scopes=frozenset({"read"}))
    with pytest.raises(ToolCallError) as e:
        invoke(reg.get("bo-next-batch", "acme"), "engine.surrogate_fit", MODEL, ro)
    assert e.value.code == "forbidden"


@pytest.mark.parametrize("args", [
    {"rows": ROWS, "campaign_id": "x", "target": TARGET},   # two sources
    {"target": TARGET},                                      # no source
    {**MODEL, "decimals": 99},                               # out of range
    {**MODEL, "surprise": 1},                                # unknown key
])
def test_invalid_args_rejected_before_running(reg, ctx, args):
    with pytest.raises(ToolCallError) as e:
        invoke(reg.get("replicate-noise-estimate", "acme"), "engine.noise_report", args, ctx)
    assert e.value.code == "invalid_args" and e.value.receipt is None
    assert _receipts(ctx) == []


def test_invalid_result_is_receipted_error(reg, ctx, monkeypatch):
    monkeypatch.setattr(bindings, "noise_report", lambda a, c: {"report": "not a dict"})
    with pytest.raises(ToolCallError) as e:
        invoke(reg.get("replicate-noise-estimate", "acme"), "engine.noise_report", MODEL, ctx)
    assert e.value.code == "invalid_result" and e.value.receipt.outcome == "error"


def test_label_contradiction_is_refused(reg, ctx):
    sk = reg.get("virtual-run-simulation", "acme")
    outs = [o.model_copy(update={"label": "synthetic"}) for o in sk.meta.outputs]
    lying = dataclasses.replace(sk, meta=sk.meta.model_copy(update={"outputs": outs}))
    with pytest.raises(ToolCallError) as e:
        invoke(lying, "engine.ode_model", {"recipes": [[0.3, 0.5, 10.0]]}, ctx)
    assert e.value.code == "label_mismatch" and e.value.receipt.outcome == "error"


def test_engine_crash_is_receipted_error(reg, ctx, monkeypatch):
    def boom(a, c):
        raise RuntimeError("secret cell value 123")
    monkeypatch.setattr(bindings, "noise_report", boom)
    with pytest.raises(ToolCallError) as e:
        invoke(reg.get("replicate-noise-estimate", "acme"), "engine.noise_report", MODEL, ctx)
    assert e.value.code == "engine_error"
    assert e.value.receipt.refusal.reason == "RuntimeError"  # no message text stored


def test_timeout_is_receipted(reg, ctx, monkeypatch):
    def slow(a, c):
        time.sleep(1.0)
        return {"noise_floor": 0.0}
    monkeypatch.setattr(bindings, "estimate_noise_floor", slow)
    with pytest.raises(ToolCallError) as e:
        invoke(reg.get("replicate-noise-estimate", "acme"), "engine.estimate_noise_floor", MODEL,
               ctx, timeout_s=0.05)
    assert e.value.code == "timeout" and e.value.receipt.refusal.code == "timeout"


def test_adapter_refusal_class(reg, ctx, monkeypatch):
    def refuse(a, c):
        raise ToolRefusal("policy", "nope")
    monkeypatch.setattr(bindings, "noise_report", refuse)
    out = invoke(reg.get("replicate-noise-estimate", "acme"), "engine.noise_report", MODEL, ctx)
    assert out.outcome == "refused" and out.refusal.code == "policy"


@pytest.mark.parametrize("label", ["predicted", "simulated", "synthetic", "derived", "metadata"])
def test_only_measured_may_be_written_as_measured(label):
    with pytest.raises(ToolRefusal):
        guard_measured_write(label)
    guard_measured_write("measured")


# --- torch-backed tools (fit a small GP) ------------------------------------------ #


def test_bo_next_batch_from_campaign_is_predicted(reg, ctx):
    pytest.importorskip("torch")
    cid = ctx.extras["cid"]
    out = invoke(reg.get("bo-next-batch", "acme"), "engine.propose_batch",
                 {"campaign_id": cid, "q": 2, "seed": 1}, ctx)
    assert out.outcome == "ok" and out.label == "predicted"
    assert 1 <= len(out.data["proposals"]) <= 2
    for p in out.data["proposals"]:
        assert set(p["recipe"]) == set(FEATS)
        assert 30 <= p["recipe"]["temp_C"] <= 38
    assert out.receipt.outputs[0].label == "predicted" and out.receipt.seed == 1


def test_minimise_flips_the_sign(reg, ctx):
    pytest.importorskip("torch")
    sk = reg.get("bo-next-batch", "acme")
    hi = invoke(sk, "engine.surrogate_fit", MODEL, ctx)
    assert hi.outcome == "ok" and len(hi.data["in_sample"]) == 16
    lo = invoke(sk, "engine.propose_batch", {**MODEL, "q": 1, "objective": "minimise"}, ctx)
    assert lo.outcome == "ok"
    mean = lo.data["proposals"][0]["predicted_mean"]
    assert mean < np.mean([r[TARGET] for r in ROWS])


def test_trust_tools(reg, ctx):
    pytest.importorskip("torch")
    sk = reg.get("trust-report", "acme")
    cv = invoke(sk, "engine.grouped_cv_report",
                {**MODEL, "group_column": "batch_id", "n_splits": 3, "n_boot": 20}, ctx)
    assert cv.outcome == "ok" and "spearman" in cv.data["report"]
    cal = invoke(sk, "engine.interval_calibration",
                 {"actual": [1, 2, 3, 4], "predicted": [1.1, 2.2, 2.9, 4.2],
                  "std": [0.2, 0.2, 0.2, 0.2]}, ctx)
    assert cal.outcome == "ok"
    shape = invoke(sk, "engine.gp_shape_report", MODEL, ctx)
    assert shape.outcome == "ok"


def test_scale_readout_is_predicted_or_refused(reg, ctx):
    pytest.importorskip("torch")
    rng = np.random.default_rng(7)
    rows = []
    for s in (1.0, 10.0, 100.0, 1000.0):
        for _ in range(6):
            rows.append({"scale_L": s, "ph_setpoint": float(rng.uniform(6.9, 7.1)),
                         "temperature_C": float(rng.uniform(36.5, 37.0)),
                         "titer_g_per_L": 5.0 - 0.2 * np.log10(s) + float(rng.normal(0, 0.05))})
    out = invoke(reg.get("scale-up-envelope-check", "acme"), "engine.scale_build_readout",
                 {"rows": rows, "target_column": "titer_g_per_L",
                  "process_columns": ["ph_setpoint", "temperature_C"],
                  "target": {"scale_L": 2000.0,
                             "process_params": {"ph_setpoint": 7.0, "temperature_C": 36.7}}},
                 ctx)
    assert out.receipt.outcome == out.outcome
    if out.outcome == "ok":
        assert out.label == "predicted" and isinstance(out.data["readout"], dict)
