"""E5-lite proposal provenance (kalos/portal/run_record.py `proposal_provenance`).

Each analysis carries a `proposal_provenance` block (engine version, library
versions for torch/botorch/gpytorch, seed, input SHA-256, config hash,
timestamp), stored with the analysis and returned by the analysis routes; each
started run carries `analysis_provenance`, a reference to the published
analysis it was started against. No model artefact is stored.
"""
from __future__ import annotations

import hashlib
import io
import json
import re

import numpy as np
import pandas as pd
import pytest

from kalos import __version__ as ENGINE_VERSION
from kalos.portal import app as portal_app
from kalos.portal import busy, run_record
from kalos.portal.campaign import CampaignStore
from kalos.portal.pipeline import attach_reanalyze_run, compute_upload_analysis

FILENAME = "plant.csv"
_KEYS = {
    "schema_version", "engine_version", "build_sha", "library_versions", "seed",
    "input_sha256", "config_sha256", "config_fields", "model_artifact_stored",
    "provenance_sha256", "timestamp", "created_at", "pin_drift",
}
_REF_KEYS = {"provenance_sha256", "input_sha256", "config_sha256", "engine_version",
             "seed", "timestamp"}


def _csv(n: int = 24, seed: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    m = rng.uniform(0, 4, n)
    ph = rng.uniform(5, 7, n)
    df = pd.DataFrame({
        "Methanol": m.round(3), "pH": ph.round(2),
        "lipase_titer": (1.5 * m - 0.8 * (ph - 6) ** 2 + rng.normal(0, 0.1, n)).round(3),
    })
    return df.to_csv(index=False).encode()


def _stub(df, target, *, timestamp=1_700_000_000, **_):
    return {"target": target or "lipase_titer", "proposal_features": ["Methanol", "pH"],
            "d": 2, "n": len(df), "proposals": [], "seed": 1234,
            "timestamp": timestamp, "engine_version": ENGINE_VERSION,
            "promotion": {"status": "stub"}, "proposal_gating": {"gated": False},
            "validation": {"status": "pass"}}


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for name in ("KALOS_BUILD_SHA", "KALOS_VALIDATION_MODE"):
        monkeypatch.delenv(name, raising=False)
    busy.reset_limiter(total=2, per_tenant=1)
    yield
    busy.reset_limiter()


def test_block_fields_and_types():
    raw = _csv()
    _, result = compute_upload_analysis(raw, "lipase_titer", False, FILENAME, analyze=_stub)
    p = result["proposal_provenance"]
    assert set(p) == _KEYS
    assert p["engine_version"] == ENGINE_VERSION
    assert p["input_sha256"] == hashlib.sha256(raw).hexdigest() == result["run"]["input_sha256"]
    assert p["seed"] == 1234
    assert set(p["library_versions"]) >= {"torch", "botorch", "gpytorch"}
    assert all(isinstance(p["library_versions"][k], str) for k in ("torch", "botorch", "gpytorch"))
    assert re.fullmatch(r"[0-9a-f]{64}", p["config_sha256"])
    assert re.fullmatch(r"[0-9a-f]{64}", p["provenance_sha256"])
    assert p["timestamp"] == 1_700_000_000
    assert p["created_at"] == "2023-11-14T22:13:20Z"
    assert p["model_artifact_stored"] is False
    assert p["build_sha"] == "unknown"
    assert p["config_fields"] == [*run_record.CONFIG_FIELDS, "proposal_fit_restarts"]


def test_config_hash_is_over_the_named_run_fields():
    _, result = compute_upload_analysis(_csv(), "lipase_titer", False, FILENAME, analyze=_stub)
    run = result["run"]
    cfg = {k: run.get(k) for k in run_record.CONFIG_FIELDS}
    cfg["proposal_fit_restarts"] = None  # the stub analysis has no model_diagnostics
    expected = hashlib.sha256(
        json.dumps(cfg, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()
    assert result["proposal_provenance"]["config_sha256"] == expected


def test_provenance_sha_is_stable_across_time_and_moves_with_inputs():
    raw = _csv()
    _, a = compute_upload_analysis(raw, "lipase_titer", False, FILENAME, analyze=_stub)
    _, b = compute_upload_analysis(
        raw, "lipase_titer", False, FILENAME,
        analyze=lambda df, t, **kw: _stub(df, t, timestamp=1_800_000_000, **kw),
    )
    pa, pb = a["proposal_provenance"], b["proposal_provenance"]
    assert pa["timestamp"] != pb["timestamp"]
    assert pa["provenance_sha256"] == pb["provenance_sha256"]  # timestamp excluded
    # different bytes -> different input hash and key, same config
    _, c = compute_upload_analysis(_csv(seed=1), "lipase_titer", False, FILENAME, analyze=_stub)
    pc = c["proposal_provenance"]
    assert pc["input_sha256"] != pa["input_sha256"]
    assert pc["config_sha256"] == pa["config_sha256"]
    assert pc["provenance_sha256"] != pa["provenance_sha256"]
    # a different config (anonymize) -> different config hash
    _, d = compute_upload_analysis(raw, "lipase_titer", True, FILENAME, analyze=_stub)
    assert d["proposal_provenance"]["config_sha256"] != pa["config_sha256"]


def test_anonymized_block_carries_no_filename():
    _, result = compute_upload_analysis(_csv(), "lipase_titer", True, FILENAME, analyze=_stub)
    assert FILENAME not in json.dumps(result["proposal_provenance"])


def test_reanalyze_path_attaches_the_block():
    df = pd.read_csv(io.BytesIO(_csv()))
    out = attach_reanalyze_run(df, _stub(df, "lipase_titer"), anonymize=False, pending_runs=2)
    p = out["proposal_provenance"]
    assert set(p) == _KEYS
    assert p["input_sha256"] == hashlib.sha256(run_record.fold_bytes(df)).hexdigest()


def test_persisted_and_returned_by_the_sync_route_and_latest(tmp_path):
    store = CampaignStore(tmp_path)
    try:
        result = portal_app._run_uploaded_sync(
            _csv(), "lipase_titer", False, FILENAME, "", tenant="acme", store=store,
        )
        assert set(result["proposal_provenance"]) == _KEYS
        stored = store.load_analysis(tenant="acme")
        assert stored["proposal_provenance"] == result["proposal_provenance"]
    finally:
        store.close()


def test_started_runs_reference_the_published_analysis(tmp_path):
    store = CampaignStore(tmp_path)
    try:
        result = portal_app._run_uploaded_sync(
            _csv(), "lipase_titer", False, FILENAME, "", tenant="acme", store=store,
        )
        runs = store.start([{"recipe": {"Methanol": 1.0, "pH": 6.0}}], tenant="acme")
        ref = runs[0]["analysis_provenance"]
        assert set(ref) == _REF_KEYS
        prov = result["proposal_provenance"]
        assert ref == {k: prov[k] for k in _REF_KEYS}
        # persisted in the campaign state, not just echoed
        pending = store.summary(tenant="acme")["pending"]
        assert pending[0]["analysis_provenance"] == ref
    finally:
        store.close()


def test_started_run_without_a_block_records_null(tmp_path):
    """An analysis written before the block existed (or none at all) -> null."""
    store = CampaignStore(tmp_path)
    try:
        portal_app._run_uploaded_sync(
            _csv(), "lipase_titer", False, FILENAME, "", tenant="acme", store=store,
        )
        legacy = store.load_analysis(tenant="acme")
        legacy.pop("proposal_provenance")
        gen = legacy.get("campaign_generation")
        store.save_analysis(legacy, "legacy", tenant="acme", campaign_generation=gen)
        runs = store.start([{"recipe": {"Methanol": 1.0, "pH": 6.0}}], tenant="acme")
        assert runs[0]["analysis_provenance"] is None
    finally:
        store.close()


def test_provenance_ref_ignores_non_dict_blocks():
    assert run_record.provenance_ref(None) is None
    assert run_record.provenance_ref({}) is None
    assert run_record.provenance_ref({"proposal_provenance": "x"}) is None
