"""Skill run receipts (kalos/skills/receipts.py): model shape, hashing in the
run-record form, append-only per-tenant storage."""
from __future__ import annotations

import json

import pandas as pd
import pytest

pytest.importorskip("pydantic")
from pydantic import ValidationError

from kalos.portal.run_record import fold_bytes
from kalos.skills.receipts import (
    ReceiptStore,
    SkillRunReceipt,
    canonical_json,
    default_root,
    engine_build_sha,
    new_receipt_id,
    sha256_hex,
    sha256_json,
    table_sha256,
    utcnow,
)


def make(**over) -> SkillRunReceipt:
    d = dict(receipt_id=new_receipt_id(), skill_slug="driver-ranking", skill_version="0.1.0",
             skill_content_hash="sha256:" + "0" * 64, tool="engine.rank_drivers",
             symbol="kalos.core.drivers:rank_drivers", args_sha256=sha256_json({"a": 1}),
             build_sha="unknown", outcome="ok",
             outputs=[{"name": "drivers", "label": "derived", "sha256": "a" * 64}],
             tenant="acme", started_at=utcnow(), finished_at=utcnow(), duration_ms=3)
    d.update(over)
    return SkillRunReceipt.model_validate(d)


def test_receipt_model_strict():
    r = make()
    assert r.schema_id == "kalos-skill-receipt/1"
    with pytest.raises(ValidationError):
        make(extra_field=1)
    with pytest.raises(ValidationError):
        make(outputs=[{"name": "x", "label": "guess", "sha256": "a" * 64}])
    with pytest.raises(ValidationError):
        make(receipt_id="nope")


def test_canonical_json_is_order_free_and_maps_nan_to_null():
    assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})
    assert canonical_json({"x": [float("nan"), float("inf")]}) == canonical_json({"x": [None, None]})


def test_table_hash_matches_run_record_form():
    rows = [{"a": 1.0, "b": "x"}, {"a": 2.5, "b": "y"}]
    assert table_sha256(rows) == sha256_hex(fold_bytes(pd.DataFrame(rows)))


def test_build_sha_from_env(monkeypatch):
    monkeypatch.setenv("KALOS_BUILD_SHA", "abc123")
    assert engine_build_sha() == "abc123"


def test_store_append_only_and_tenant_scoped(tmp_path):
    store = ReceiptStore(tmp_path)
    r = make()
    p = store.write(r)
    assert p.parent.name == "acme" and json.loads(p.read_text())["receipt_id"] == r.receipt_id
    assert not list(p.parent.glob(".tmp-*"))
    with pytest.raises(FileExistsError):
        store.write(r)
    assert store.get("acme", r.receipt_id) == r
    assert store.get("globex", r.receipt_id) is None
    assert store.get("acme", "../../etc/passwd") is None
    r2 = make(skill_slug="trust-report")
    store.write(r2)
    assert [x.receipt_id for x in store.list_for("acme", skill="trust-report")] == [r2.receipt_id]
    assert len(store.list_for("acme")) == 2
    assert store.list_for("globex") == []


def test_store_rejects_bad_tenant(tmp_path):
    with pytest.raises(ValueError):
        ReceiptStore(tmp_path).write(make(tenant="../x"))


def test_default_root_follows_state_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("KALOS_STATE_DIR", str(tmp_path))
    assert default_root() == tmp_path / "skill_receipts"
