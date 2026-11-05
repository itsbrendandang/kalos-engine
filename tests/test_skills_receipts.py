"""Receipts and run directories (kalos/skills/receipts.py): strict model,
hashing in the run-record form, fresh per-run directories, and the forward-only
state machine running -> analysis_complete -> terminal."""
from __future__ import annotations

import json

import pandas as pd
import pytest

pytest.importorskip("pydantic")
from pydantic import ValidationError

from kalos.portal.run_record import fold_bytes
from kalos.skills.receipts import (
    RECEIPT_FILE,
    RunStore,
    SkillRunReceipt,
    TransitionError,
    canonical_json,
    default_root,
    engine_build_sha,
    library_versions,
    new_receipt_id,
    sha256_hex,
    sha256_json,
    table_sha256,
    utcnow,
)


def make(**over) -> SkillRunReceipt:
    now = utcnow()
    d = dict(receipt_id=new_receipt_id(), skill_slug="driver-ranking", skill_version="0.1.0",
             skill_content_hash="sha256:" + "0" * 64, tool="engine.rank_drivers",
             symbol="kalos.core.drivers:rank_drivers", args_json="{}",
             args_sha256=sha256_json({}), build_sha="unknown", state="running",
             tenant="acme", started_at=now, updated_at=now)
    d.update(over)
    return SkillRunReceipt.model_validate(d)


def test_receipt_model_strict():
    r = make()
    assert r.schema_id == "kalos-skill-receipt/1"
    with pytest.raises(ValidationError):
        make(extra_field=1)
    with pytest.raises(ValidationError):
        make(outputs=[{"name": "x", "path": "x.json", "sha256": "a" * 64, "bytes": 1,
                       "label": "guess"}])
    with pytest.raises(ValidationError):
        make(receipt_id="nope")
    with pytest.raises(ValidationError):
        make(state="done")
    with pytest.raises(ValidationError):
        make(error={"error_code": "weird", "message": "m", "receipt_id": "x"})


def test_canonical_json_is_order_free_and_maps_nan_to_null():
    assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})
    assert canonical_json({"x": [float("nan"), float("inf")]}) == canonical_json({"x": [None, None]})


def test_table_hash_matches_run_record_form():
    rows = [{"a": 1.0, "b": "x"}, {"a": 2.5, "b": "y"}]
    assert table_sha256(rows) == sha256_hex(fold_bytes(pd.DataFrame(rows)))


def test_build_sha_and_library_versions(monkeypatch):
    monkeypatch.setenv("KALOS_BUILD_SHA", "abc123")
    assert engine_build_sha() == "abc123"
    v = library_versions()
    assert v["numpy"] and "torch" in v and "kalos" in v


def test_fresh_run_dir_and_forward_only_states(tmp_path):
    store = RunStore(tmp_path)
    r = make()
    d = store.create_run(r)
    assert d == tmp_path / "acme" / r.receipt_id and (d / RECEIPT_FILE).is_file()
    with pytest.raises(FileExistsError):
        store.create_run(r)  # never reuse a run directory
    with pytest.raises(TransitionError):
        store.create_run(make(state="terminal", result="pass"))
    mid = r.model_copy(update={"state": "analysis_complete", "status": "completed"})
    store.update(mid)
    with pytest.raises(TransitionError):
        store.update(r)  # backwards
    end = mid.model_copy(update={"state": "terminal", "result": "pass"})
    store.update(end)
    with pytest.raises(TransitionError):
        store.update(end)  # terminal receipts are never written again
    assert store.get("acme", r.receipt_id) == end
    assert json.loads((d / RECEIPT_FILE).read_text())["state"] == "terminal"
    assert not list(d.glob(".tmp-*"))


def test_running_may_go_straight_to_terminal_fail(tmp_path):
    store = RunStore(tmp_path)
    r = make()
    store.create_run(r)
    store.update(r.model_copy(update={"state": "terminal", "result": "fail", "status": "blocked"}))


def test_tenant_scoping(tmp_path):
    store = RunStore(tmp_path)
    r = make()
    store.create_run(r)
    assert store.get("globex", r.receipt_id) is None
    assert store.get("acme", "../../etc/passwd") is None
    store.create_run(make(skill_slug="trust-report"))
    assert len(store.list_for("acme")) == 2
    assert [x.skill_slug for x in store.list_for("acme", skill="trust-report")] == ["trust-report"]
    assert store.list_for("globex") == []
    with pytest.raises(ValueError):
        store.create_run(make(tenant=".."))


def test_default_root_follows_state_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("KALOS_STATE_DIR", str(tmp_path))
    assert default_root() == tmp_path / "skill_runs"
