"""The provenance `run` block (kalos/portal/run_record.py, docs/CAMPAIGNS_API.md).

Sync and job paths must attach an identical block for the same bytes; the
input hash is over the raw bytes; anonymize keeps the filename and sheet out of
the stored blob; the key set and types are pinned; an old blob without `run`
still loads; and the same bytes on the same stack give the same proposals.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from kalos.portal import app as portal_app
from kalos.portal import busy, job_worker, jobs, run_record
from kalos.portal.campaign import CampaignStore
from kalos.portal.pipeline import compute_upload_analysis, stored_dataset_label

FILENAME = "secret-plant-7.csv"


def _df(n: int = 24, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    m = rng.uniform(0, 4, n)
    ph = rng.uniform(5, 7, n)
    return pd.DataFrame({
        "Methanol": m.round(3), "pH": ph.round(2),
        "lipase_titer": (1.5 * m - 0.8 * (ph - 6) ** 2 + rng.normal(0, 0.1, n)).round(3),
    })


def _csv(df: pd.DataFrame | None = None) -> bytes:
    return (df if df is not None else _df()).to_csv(index=False).encode()


def _stub_analyze(df, target, *, profile=None, anonymize=False, progress=None, roles=None, **_):
    """Stands in for the GP fit (the same stub the job tests use): the result
    keys the publish path reads, plus the ones the run record reads. `d` is the
    fitted feature count; `proposal_features` is only the top-4 drivers."""
    return {"target": target or "lipase_titer", "proposal_features": ["Methanol", "pH"],
            "d": 2, "n": len(df),
            "proposals": [], "marker": "stub-result", "seed": 1234,
            "promotion": {"status": "stub"}, "proposal_gating": {"gated": False},
            "validation": {"status": "pass"}}


@pytest.fixture(autouse=True)
def _fresh_limiter():
    busy.reset_limiter(total=2, per_tenant=1)
    yield
    busy.reset_limiter()


@pytest.fixture
def store(tmp_path, monkeypatch):
    for name in ("KALOS_JOB_QUEUE_DEPTH", "KALOS_JOB_MAX_UPLOAD_BYTES", "KALOS_JOB_TIMEOUT_S",
                 "KALOS_JOBS_ENABLED", "KALOS_BUILD_SHA", "KALOS_VALIDATION_MODE"):
        monkeypatch.delenv(name, raising=False)
    s = CampaignStore(tmp_path)
    yield s
    s.close()


def _sync_result(store, raw, *, anonymize=False, filename=FILENAME, roles=""):
    return portal_app._run_uploaded_sync(
        raw, "lipase_titer", anonymize, filename, roles, tenant="acme", store=store,
    )


def _job_result(store, raw, *, anonymize=False, filename=FILENAME, roles=""):
    cid = store.create_campaign(tenant="acme", name="c")["id"]
    job = jobs.enqueue(
        store, tenant="acme", campaign_id=cid, raw=raw, filename=filename,
        target="lipase_titer", anonymize=anonymize, roles=roles, sheet=None,
        requested_by="tester",
    )
    assert jobs.claim(store, tenant="acme", runner_id="test")["id"] == job["id"]
    assert job_worker.run_job(store, job["id"], analyze=_stub_analyze) == "succeeded"
    return store.load_analysis(tenant="acme", campaign_id=cid)


# --- the block itself -------------------------------------------------------- #

def test_input_sha256_is_hashlib_over_the_raw_bytes():
    raw = _csv()
    df, result = compute_upload_analysis(raw, "lipase_titer", False, FILENAME,
                                         analyze=_stub_analyze)
    run = result["run"]
    assert run["input_sha256"] == hashlib.sha256(raw).hexdigest()
    assert run["input_bytes"] == len(raw)
    assert run["n_rows"] == len(df) == 24
    assert run["n_features"] == 2
    assert run["target"] == "lipase_titer"
    assert run["seed"] == 1234
    assert run["schema_version"] == run_record.RUN_SCHEMA_VERSION == 1


def test_n_features_is_the_fitted_feature_count_not_the_top_drivers():
    """Regression: `n_features` used to be len(proposal_features), which is the
    top-4 drivers (analysis.py drv[:4]), so a 6-feature fit reported 4. This stub
    fits 7 features and names only 4 drivers; the record must say 7."""
    def stub(df, target, **kwargs):
        out = _stub_analyze(df, target, **kwargs)
        out.update(d=7, proposal_features=["a", "b", "c", "d"])
        return out

    _, result = compute_upload_analysis(_csv(), "lipase_titer", False, FILENAME, analyze=stub)
    assert result["run"]["n_features"] == 7


def test_n_features_matches_a_real_six_feature_fit():
    """The same claim against the real fit: six inputs, four drivers shown."""
    rng = np.random.default_rng(3)
    n = 40
    cols = {f"x{i}": rng.uniform(0, 1, n).round(4) for i in range(6)}
    cols["lipase_titer"] = (2 * cols["x0"] - cols["x1"] + rng.normal(0, 0.05, n)).round(4)
    _, result = compute_upload_analysis(
        pd.DataFrame(cols).to_csv(index=False).encode(), "lipase_titer", False, FILENAME,
    )
    assert result["d"] == 6 and len(result["proposal_features"]) == 4
    assert result["run"]["n_features"] == result["d"] == 6


def test_build_sha_comes_from_env_and_defaults_to_unknown(monkeypatch):
    monkeypatch.delenv("KALOS_BUILD_SHA", raising=False)
    assert run_record.build_sha() == "unknown"
    monkeypatch.setenv("KALOS_BUILD_SHA", "abc123")
    _, result = compute_upload_analysis(_csv(), "lipase_titer", False, FILENAME,
                                        analyze=_stub_analyze)
    assert result["run"]["build_sha"] == "abc123"


def test_run_record_module_is_torch_free():
    code = ("import sys, kalos.portal.run_record; "
            "print(any(m.split('.')[0] in ('torch', 'botorch', 'gpytorch') for m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False"


def test_gate_decisions_are_referenced_by_key_not_copied():
    _, result = compute_upload_analysis(_csv(), "lipase_titer", False, FILENAME,
                                        analyze=_stub_analyze)
    gates = result["run"]["gates"]
    assert gates == {"promotion": "promotion", "proposal_gating": "proposal_gating",
                     "validation": "validation"}
    # the decision bodies stay where they were; the record holds no copy of them
    assert "status" not in json.dumps(result["run"])


# --- sync and job paths ------------------------------------------------------- #

def test_sync_and_job_paths_attach_an_identical_run_block(store, monkeypatch):
    monkeypatch.setattr(portal_app, "_analyze", _stub_analyze)
    raw = _csv()
    sync = _sync_result(store, raw)
    job = _job_result(store, raw)
    assert sync["run"] == job["run"]
    assert sync["run"]["input_sha256"] == hashlib.sha256(raw).hexdigest()


def test_stored_blob_carries_the_run_block(store, monkeypatch):
    monkeypatch.setattr(portal_app, "_analyze", _stub_analyze)
    _sync_result(store, _csv())
    stored = store.load_analysis(tenant="acme")
    assert stored["run"]["filename"] == FILENAME
    assert stored["dataset"] == FILENAME


# --- privacy ------------------------------------------------------------------ #

def test_anonymize_leaves_no_filename_in_the_stored_blob(store, monkeypatch):
    monkeypatch.setattr(portal_app, "_analyze", _stub_analyze)
    _sync_result(store, _csv(), anonymize=True)
    stored = store.load_analysis(tenant="acme")
    text = json.dumps(stored)
    assert "secret-plant-7" not in text
    assert "filename" not in stored["run"] and "sheet" not in stored["run"]
    assert stored["dataset"] == "anonymized upload " + stored["run"]["input_sha256"][:12]


def test_anonymize_leaves_no_filename_in_the_job_blob(store):
    stored = _job_result(store, _csv(), anonymize=True)
    assert "secret-plant-7" not in json.dumps(stored)
    assert "filename" not in stored["run"]


def test_stored_dataset_label_only_hides_under_anonymize():
    assert stored_dataset_label({"run": {"anonymize": False}}, FILENAME) == FILENAME
    assert stored_dataset_label({}, FILENAME) == FILENAME
    label = stored_dataset_label({"run": {"anonymize": True, "input_sha256": "ab" * 32}}, FILENAME)
    assert FILENAME not in label and label == "anonymized upload " + "ab" * 6


# --- golden: key set and types ------------------------------------------------ #

_TYPES = {
    "schema_version": int, "input_sha256": str, "input_bytes": int, "target": str,
    "profile": str, "roles_declared": (dict, type(None)), "anonymize": bool, "seed": int,
    "torch_threads": (int, type(None)), "torch_threads_actual": (int, type(None)),
    "max_fit_rows": int, "sheet_index": (int, type(None)), "sheet_hash": (str, type(None)),
    "build_sha": str, "library_versions": dict,
    "validation_mode": str, "n_rows": int, "n_features": int, "kind": str,
    "pending_runs": int, "gates": dict,
}
_KEYS_PLAIN = set(_TYPES) | {"filename", "sheet"}


def test_key_set_and_types_are_pinned(monkeypatch):
    # Independent of the environment: a deployment's validation mode (or thread
    # pin, build sha) must not change which keys exist or their types.
    monkeypatch.setenv("KALOS_VALIDATION_MODE", "warn")
    _, plain = compute_upload_analysis(_csv(), "lipase_titer", False, FILENAME,
                                       analyze=_stub_analyze)
    run = plain["run"]
    assert set(run) == _KEYS_PLAIN
    for key, typ in _TYPES.items():
        assert isinstance(run[key], typ), key
    assert isinstance(run["filename"], str) and isinstance(run["sheet"], (str, type(None)))
    assert set(run["library_versions"]) == {"numpy", "pandas", "torch", "botorch"}
    assert all(v is None or isinstance(v, str) for v in run["library_versions"].values())
    assert run["profile"] == "bioprocess" and run["roles_declared"] is None
    assert run["validation_mode"] == "warn"  # pinned by the monkeypatch above


def test_anonymize_key_set_is_pinned_without_filename_and_sheet():
    _, anon = compute_upload_analysis(_csv(), "lipase_titer", True, FILENAME,
                                      analyze=_stub_analyze)
    assert set(anon["run"]) == set(_TYPES)
    assert anon["run"]["anonymize"] is True


def test_declared_roles_are_recorded_and_switch_the_profile():
    roles = json.dumps({"target": "lipase_titer", "features": ["Methanol", "pH"]})
    _, result = compute_upload_analysis(_csv(), "lipase_titer", False, FILENAME, roles,
                                        analyze=_stub_analyze)
    assert result["run"]["roles_declared"] == {"target": "lipase_titer",
                                               "features": ["Methanol", "pH"]}
    assert result["run"]["profile"] == "generic"


# --- back-compat -------------------------------------------------------------- #

def test_old_blob_without_run_still_loads(store):
    old = {"target": "lipase_titer", "proposal_features": ["Methanol"], "proposals": [],
           "marker": "pre-run-record", "seed": 1234, "engine_version": "0.0.0"}
    store.save_analysis(old, "old.csv", tenant="acme")
    loaded = store.load_analysis(tenant="acme")
    assert loaded is not None and loaded["marker"] == "pre-run-record"
    assert "run" not in loaded  # absent (no key), not null
    assert loaded["seed"] == 1234 and loaded["engine_version"] == "0.0.0"


# --- determinism (real fit) --------------------------------------------------- #

def test_same_bytes_same_stack_same_proposals():
    """The claim holds for the same library stack and the same torch thread
    count (`run.torch_threads`, `run.library_versions`). Across different
    versions or thread counts the proposals may legitimately differ."""
    raw = _csv()
    _, first = compute_upload_analysis(raw, "lipase_titer", False, FILENAME)
    _, second = compute_upload_analysis(raw, "lipase_titer", False, FILENAME)
    assert first["run"] == second["run"]
    assert first["proposals"] == second["proposals"]
    assert first["proposal_features"] == second["proposal_features"]
    # the existing audit keys stay at the top level for back-compat
    assert first["seed"] == second["seed"] == first["run"]["seed"]
    assert first["engine_version"] and first["run"]["torch_threads_actual"] >= 1
