"""The single skill runner (kalos/skills/runner.py): validated calls, small
status JSON, artifacts in a fresh per-run directory, receipts through
running -> analysis_complete -> terminal, blocked/refused/error outcomes,
timeouts, verify_receipt, finalize, and one run-twice determinism test per
locked skill that has tools."""
from __future__ import annotations

import base64
import dataclasses
import json
import threading
import time

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pydantic")
pytest.importorskip("yaml")
pytest.importorskip("fastapi")

from kalos.skills import bindings, load_builtin
from kalos.skills.receipts import RECEIPT_FILE, RunStore, frame_sha256, sha256_hex
from kalos.skills.runner import (
    ToolCallError,
    ToolContext,
    ToolRefusal,
    _Fault,
    _write_artifact,
    finalize,
    guard_measured_write,
    invoke,
    verify_receipt,
)
from kalos.skills.tools import TOOLS

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
CSV = _df().to_csv(index=False)


@pytest.fixture
def reg():
    return load_builtin()


@pytest.fixture
def ctx(tmp_path):
    from kalos.portal.campaign import CampaignStore

    store = CampaignStore(tmp_path / "state")
    cid = store.create_campaign(tenant="acme", name="c")["id"]
    store.seed(_df(), TARGET, FEATS, tenant="acme", campaign_id=cid)
    c = ToolContext(tenant="acme", subject="alice@acme", scopes=frozenset({"read", "write"}),
                    store=store, runs=RunStore(tmp_path / "runs"))
    c.extras["cid"] = cid
    try:
        yield c
    finally:
        store.close()


def run(reg, ctx, slug, tool, args, **kw):
    return invoke(reg.get(slug, "acme"), tool, args, ctx, **kw)


# --- every tool runs end to end -------------------------------------------------- #

TORCH_FREE = [
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


@pytest.mark.parametrize("slug,tool,args", TORCH_FREE, ids=[c[1] for c in TORCH_FREE])
def test_tool_runs_writes_artifacts_and_receipt(reg, ctx, slug, tool, args):
    sk = reg.get(slug, "acme")
    res = invoke(sk, tool, args, ctx)
    st = res.status
    assert st.status == "completed", (st.blockers, st.error)
    assert res.run_dir == ctx.runs.root / "acme" / st.receipt_id
    # small status: only artifact descriptors, never data
    assert set(st.model_dump()) == {"status", "summary", "artifacts", "blockers", "error",
                                    "receipt_id"}
    for a in st.artifacts:
        blob = (res.run_dir / a.path).read_bytes()
        assert sha256_hex(blob) == a.sha256 and len(blob) == a.bytes
        assert a.label == TOOLS[tool].label == sk.meta.output(a.name).label
        assert len(blob) >= sk.meta.output(a.name).min_bytes
    r = res.receipt
    assert (r.state, r.result, r.status) == ("terminal", "pass", "completed")
    assert r.skill_content_hash == sk.content_hash and r.symbol == TOOLS[tool].symbol
    assert r.tenant == "acme" and r.subject == "alice@acme" and r.library_versions
    assert json.loads(r.args_json) == TOOLS[tool].args_model.model_validate(args).model_dump(
        mode="json")
    assert r.args_sha256 == sha256_hex(r.args_json.encode())
    if "rows" in args:
        assert r.input_sha256s == [frame_sha256(pd.DataFrame(args["rows"]))]
    assert ctx.runs.get("acme", st.receipt_id) == r
    assert verify_receipt(res.run_dir, sk).ok


def test_table_outputs_go_to_csv_files(reg, ctx):
    res = run(reg, ctx, "identifier-anonymisation", "engine.anonymizer", {"rows": ROWS})
    names = {a.name: a.path for a in res.status.artifacts}
    assert names == {"anonymised_table": "anonymised_table.csv", "alias_report": "alias_report.json"}
    table = res.load("anonymised_table")
    assert len(table) == 16 and "B0" not in {r["batch_id"] for r in table}
    assert "rows" not in res.load("alias_report")
    # inputs are hashed, never copied: no raw identifier anywhere in the run dir
    for p in res.run_dir.iterdir():
        assert "B0," not in p.read_text()


def test_campaign_tools_read_the_callers_campaign(reg, ctx):
    cid = ctx.extras["cid"]
    s = run(reg, ctx, "campaign-status", "engine.campaign_summary", {"campaign_id": cid})
    assert s.load("status")["summary"]["target"] == TARGET and s.receipt.campaign_id == cid
    n = run(reg, ctx, "replicate-noise-estimate", "engine.noise_report", {"campaign_id": cid})
    assert n.load("noise_report")["report"]["n_rows"] == 16
    assert len(n.receipt.input_sha256s) == 1


def test_campaign_summary_strips_predictions(reg, ctx):
    cid = ctx.extras["cid"]
    ctx.store.start([{"recipe": {"temp_C": 33.0, "pH": 6.5}, "pred": 9.9, "std": 0.4,
                      "mode": "explore", "reason": "x"}], tenant="acme", campaign_id=cid)
    s = run(reg, ctx, "campaign-status", "engine.campaign_summary", {"campaign_id": cid})
    for r in s.load("status")["summary"]["pending"]:
        assert "pred" not in r and "std" not in r


# --- outcomes ------------------------------------------------------------------------ #


def _terminal_fail(res, status):
    assert res.status.status == status and res.status.artifacts == []
    assert (res.receipt.state, res.receipt.result) == ("terminal", "fail")
    assert not verify_receipt(res.run_dir, None).ok


def test_other_tenants_campaign_is_refused(reg, ctx):
    other = ctx.store.create_campaign(tenant="globex", name="g")["id"]
    res = run(reg, ctx, "campaign-status", "engine.campaign_summary", {"campaign_id": other})
    _terminal_fail(res, "refused")
    assert res.status.blockers[0].code == "not_found"


def test_too_few_rows_is_blocked(reg, ctx):
    res = run(reg, ctx, "replicate-noise-estimate", "engine.noise_report",
              {"rows": ROWS[:2], "target": TARGET, "features": FEATS})
    _terminal_fail(res, "blocked")
    assert res.status.blockers[0].code == "too_few_rows"


def test_unparseable_upload_is_blocked_not_error(reg, ctx):
    res = run(reg, ctx, "run-sheet-checks", "engine.parse_upload", {"csv_text": "\n\n"})
    assert res.status.status in ("blocked", "refused")
    assert res.receipt.input_sha256s  # the bytes were hashed even though refused


def test_strict_mode_blocks_with_rows(reg, ctx):
    rows = [dict(r) for r in ROWS]
    rows[3]["pH"] = 25.0
    warn = run(reg, ctx, "run-sheet-checks", "engine.validate_frame", {"rows": rows})
    assert warn.status.status == "completed"
    assert warn.load("check_report")["report"]["counts"]["error"] > 0
    strict = run(reg, ctx, "run-sheet-checks", "engine.validate_frame",
                 {"rows": rows, "mode": "strict"})
    _terminal_fail(strict, "blocked")
    b = strict.status.blockers[0]
    assert b.code == "blocking_findings" and b.rows == [3]


def test_engine_crash_stores_code_not_text(reg, ctx, monkeypatch):
    def boom(a, c):
        raise RuntimeError("secret cell value 123")
    monkeypatch.setattr(bindings, "noise_report", boom)
    res = run(reg, ctx, "replicate-noise-estimate", "engine.noise_report", MODEL)
    _terminal_fail(res, "error")
    e = res.status.error
    assert e.error_code == "internal" and e.exception_type == "RuntimeError"
    assert e.receipt_id == res.status.receipt_id
    assert "secret" not in (res.run_dir / RECEIPT_FILE).read_text()


def test_unreadable_input_error_code(reg, ctx, monkeypatch):
    class ParserError(ValueError):
        pass

    def bad(a, c):
        raise ParserError("row 3: 'Zeus' cannot parse")
    monkeypatch.setattr(bindings, "parse_upload", bad)
    res = run(reg, ctx, "run-sheet-checks", "engine.parse_upload", {"csv_text": "a\n1\n"})
    assert res.status.error.error_code == "input_unreadable"
    assert "Zeus" not in (res.run_dir / RECEIPT_FILE).read_text()


def test_fit_failure_is_blocked(reg, ctx, monkeypatch):
    class FitError(RuntimeError):
        pass

    def fail(a, c):
        raise FitError("cholesky jitter 1e-3 failed")
    monkeypatch.setattr(bindings, "surrogate_fit", fail)
    res = run(reg, ctx, "bo-next-batch", "engine.surrogate_fit", MODEL)
    _terminal_fail(res, "blocked")
    assert res.status.blockers[0].code == "fit_failed"
    assert "cholesky" not in res.status.blockers[0].message


def test_invalid_result_and_small_output(reg, ctx, monkeypatch):
    monkeypatch.setattr(bindings, "noise_report", lambda a, c: {"report": "not a dict"})
    res = run(reg, ctx, "replicate-noise-estimate", "engine.noise_report", MODEL)
    assert res.status.error.error_code == "invalid_result"
    sk = reg.get("replicate-noise-estimate", "acme")
    outs = [o.model_copy(update={"min_bytes": 10**6}) for o in sk.meta.outputs]
    big = dataclasses.replace(sk, meta=sk.meta.model_copy(update={"outputs": outs}))
    monkeypatch.undo()
    res2 = invoke(big, "engine.noise_report", MODEL, ctx)
    assert res2.status.error.error_code == "output_too_small"
    assert not list(res2.run_dir.glob("*.json*"))[1:]  # only receipt.json


def test_label_contradiction_is_an_error(reg, ctx):
    sk = reg.get("virtual-run-simulation", "acme")
    outs = [o.model_copy(update={"label": "synthetic"}) for o in sk.meta.outputs]
    lying = dataclasses.replace(sk, meta=sk.meta.model_copy(update={"outputs": outs}))
    res = invoke(lying, "engine.ode_model", {"recipes": [[0.3, 0.5, 10.0]]}, ctx)
    assert res.status.error.error_code == "label_mismatch"


def test_timeout_discards_late_result(reg, ctx, monkeypatch):
    released = threading.Event()

    def slow(a, c):
        released.wait(5)
        return {"noise_floor": 0.0}
    monkeypatch.setattr(bindings, "estimate_noise_floor", slow)
    res = run(reg, ctx, "replicate-noise-estimate", "engine.estimate_noise_floor", MODEL,
              timeout_s=0.05)
    assert res.status.error.error_code == "timeout"
    released.set()
    time.sleep(0.2)
    after = ctx.runs.get("acme", res.status.receipt_id)
    assert after.state == "terminal" and after.status == "error" and after.outputs == []
    assert sorted(p.name for p in res.run_dir.iterdir()) == [RECEIPT_FILE]


def test_adapter_refusal_kinds(reg, ctx, monkeypatch):
    monkeypatch.setattr(bindings, "noise_report",
                        lambda a, c: (_ for _ in ()).throw(ToolRefusal("policy", "nope")))
    assert run(reg, ctx, "replicate-noise-estimate", "engine.noise_report",
               MODEL).status.status == "refused"
    monkeypatch.setattr(bindings, "noise_report", lambda a, c: (_ for _ in ()).throw(
        ToolRefusal("gate", "failed gate", kind="blocked")))
    assert run(reg, ctx, "replicate-noise-estimate", "engine.noise_report",
               MODEL).status.status == "blocked"


# --- calls that never start -------------------------------------------------------- #


def test_planned_skill_has_nothing_to_run(reg, ctx):
    with pytest.raises(ToolCallError) as e:
        run(reg, ctx, "hplc-peak-summary", "engine.noise_report", MODEL)
    assert e.value.code == "planned"


@pytest.mark.parametrize("slug,tool,args,code", [
    ("driver-ranking", "engine.noise_report", MODEL, "tool_not_in_skill"),
    ("replicate-noise-estimate", "engine.noise_report",
     {"rows": ROWS, "campaign_id": "x", "target": TARGET}, "invalid_args"),
    ("replicate-noise-estimate", "engine.noise_report", {"target": TARGET}, "invalid_args"),
    ("replicate-noise-estimate", "engine.noise_report", {**MODEL, "decimals": 99}, "invalid_args"),
    ("replicate-noise-estimate", "engine.noise_report", {**MODEL, "surprise": 1}, "invalid_args"),
])
def test_rejected_before_any_run_dir(reg, ctx, slug, tool, args, code):
    with pytest.raises(ToolCallError) as e:
        run(reg, ctx, slug, tool, args)
    assert e.value.code == code
    assert not (ctx.runs.root / "acme").exists()


def test_scope_is_enforced(reg, ctx):
    ro = dataclasses.replace(ctx, scopes=frozenset({"read"}))
    with pytest.raises(ToolCallError) as e:
        invoke(reg.get("bo-next-batch", "acme"), "engine.surrogate_fit", MODEL, ro)
    assert e.value.code == "forbidden"


@pytest.mark.parametrize("label", ["predicted", "simulated", "synthetic", "derived", "metadata"])
def test_only_measured_may_be_written_as_measured(label):
    with pytest.raises(ToolRefusal):
        guard_measured_write(label)
    guard_measured_write("measured")


# --- state machine, finalize, verify ----------------------------------------------- #


def test_hold_then_finalize(reg, ctx):
    sk = reg.get("driver-ranking", "acme")
    res = invoke(sk, "engine.rank_drivers", {**MODEL, "n_boot": 20}, ctx, hold=True)
    assert res.receipt.state == "analysis_complete" and res.receipt.result is None
    assert not verify_receipt(res.run_dir, sk).ok
    done = finalize(res, present=lambda d, r: [("drivers", "drivers_report.md",
                                                 b"# Drivers\n", "derived")])
    assert done.receipt.state == "terminal" and done.receipt.result == "pass"
    assert {a.path for a in done.receipt.outputs} == {"drivers.json", "drivers_report.md"}
    assert verify_receipt(res.run_dir, sk).ok
    with pytest.raises(ToolCallError):
        finalize(done)  # only from analysis_complete


def test_verify_detects_changes(reg, ctx):
    sk = reg.get("driver-ranking", "acme")
    res = invoke(sk, "engine.rank_drivers", {**MODEL, "n_boot": 20}, ctx)
    assert verify_receipt(res.run_dir, sk).ok
    changed = dataclasses.replace(sk, content_hash="sha256:" + "1" * 64)
    assert "skill content hash changed" in verify_receipt(res.run_dir, changed).problems[0]
    (res.run_dir / "drivers.json").write_text("{}")
    assert any("changed since the run" in p for p in verify_receipt(res.run_dir, sk).problems)
    (res.run_dir / "drivers.json").unlink()
    assert any("missing" in p for p in verify_receipt(res.run_dir, sk).problems)


def test_artifact_paths_cannot_escape(tmp_path):
    d = tmp_path / "run"
    d.mkdir()
    for bad in ["../x.json", "/etc/x.json", "a/../../x.json"]:
        with pytest.raises(_Fault):
            _write_artifact(d, bad, b"{}")
    _write_artifact(d, "ok.json", b"{}")
    with pytest.raises(_Fault):
        _write_artifact(d, "ok.json", b"{}")  # never overwrite


# --- determinism: one per locked skill with tools ---------------------------------- #

DETERMINISM = {
    "bo-next-batch": ("engine.propose_batch", {**MODEL, "q": 2, "seed": 3}),
    "campaign-provenance-report": ("engine.run_record_build", {"csv_text": CSV, "target": TARGET}),
    "campaign-status": ("engine.derive_stage", {"campaign_id": "<cid>"}),
    "column-role-inference": ("engine.offline_plan", {"rows": ROWS, "target_column": TARGET}),
    "driver-ranking": ("engine.rank_drivers", {**MODEL, "n_boot": 50, "seed": 3}),
    "identifier-anonymisation": ("engine.anonymizer", {"rows": ROWS}),
    "replicate-noise-estimate": ("engine.noise_report", MODEL),
    "run-sheet-checks": ("engine.validate_frame", {"rows": ROWS, "target": TARGET}),
    "scale-up-envelope-check": ("engine.scale_build_readout", "<scale>"),
    "search-box-definition": ("engine.build_search_box", MODEL),
    "target-readiness-check": ("engine.assess_target_variation",
                               {"values": [r[TARGET] for r in ROWS]}),
    "trust-report": ("engine.grouped_cv_report",
                     {**MODEL, "group_column": "batch_id", "n_splits": 3, "n_boot": 20, "seed": 1}),
    "unit-normalisation": ("engine.apply_unit_conversions",
                           {"rows": [{"t": "1 g/L"}, {"t": "2 g/L"}, {"t": "3 g/L"}]}),
    "virtual-run-simulation": ("engine.ode_model", {"recipes": [[0.3, 0.5, 10.0]]}),
}


def _scale_args():
    rng = np.random.default_rng(7)
    rows = [{"scale_L": s, "ph_setpoint": float(rng.uniform(6.9, 7.1)),
             "temperature_C": float(rng.uniform(36.5, 37.0)),
             "titer_g_per_L": 5.0 - 0.2 * np.log10(s) + float(rng.normal(0, 0.05))}
            for s in (1.0, 10.0, 100.0, 1000.0) for _ in range(6)]
    return {"rows": rows, "target_column": "titer_g_per_L",
            "process_columns": ["ph_setpoint", "temperature_C"],
            "target": {"scale_L": 2000.0,
                       "process_params": {"ph_setpoint": 7.0, "temperature_C": 36.7}}}


def test_determinism_table_covers_every_locked_skill_with_tools(reg):
    assert set(DETERMINISM) == {s.slug for s in reg.all()
                                if s.meta.kind == "locked" and s.meta.tools}


@pytest.mark.parametrize("slug", sorted(DETERMINISM))
def test_same_args_seed_build_and_skill_give_same_output_hashes(reg, ctx, slug):
    tool, args = DETERMINISM[slug]
    if TOOLS[tool].needs_torch:
        pytest.importorskip("torch")
    if args == "<scale>":
        args = _scale_args()
    elif args.get("campaign_id") == "<cid>":
        args = {"campaign_id": ctx.extras["cid"]}
    a = run(reg, ctx, slug, tool, args)
    b = run(reg, ctx, slug, tool, args)
    assert a.status.status == b.status.status
    assert a.receipt.args_sha256 == b.receipt.args_sha256
    assert a.receipt.input_sha256s == b.receipt.input_sha256s
    assert a.receipt.skill_content_hash == b.receipt.skill_content_hash
    assert a.receipt.build_sha == b.receipt.build_sha
    assert [(x.name, x.sha256) for x in a.status.artifacts] == \
        [(x.name, x.sha256) for x in b.status.artifacts]
    assert a.status.receipt_id != b.status.receipt_id
    if a.status.status == "completed":
        assert a.status.artifacts
