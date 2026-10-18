"""What a background job keeps about its upload (docs/JOBS.md, "Privacy").

`jobs.params` is the form fields the worker needs. Before this fix it held the
raw filename under every mode and kept it after the job finished. The rules
pinned here: under `anonymize` the filename is never stored (only its
extension, which the parser uses for format detection); every terminal state
clears `params` in the same transaction that sets the state.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from kalos.portal import busy, job_worker, jobs
from kalos.portal.campaign import CampaignStore

FILENAME = "secret-plant-7.csv"
TARGET = "lipase_titer"
ROLES = json.dumps({"target": TARGET, "features": ["Methanol", "pH"]})


def _csv() -> bytes:
    rng = np.random.default_rng(0)
    m = rng.uniform(0, 4, 24)
    ph = rng.uniform(5, 7, 24)
    df = pd.DataFrame({"Methanol": m.round(3), "pH": ph.round(2),
                       TARGET: (1.5 * m + rng.normal(0, 0.1, 24)).round(3)})
    return df.to_csv(index=False).encode()


def _stub(df, target, **kwargs):
    return {"target": target, "proposal_features": ["Methanol"], "d": 2, "proposals": [],
            "marker": "stub", "seed": 1, "promotion": {}, "proposal_gating": {},
            "validation": {"status": "pass"}}


@pytest.fixture(autouse=True)
def _limiter():
    busy.reset_limiter(total=2, per_tenant=1)
    yield
    busy.reset_limiter()


@pytest.fixture
def store(tmp_path, monkeypatch):
    for name in ("KALOS_JOB_QUEUE_DEPTH", "KALOS_JOB_MAX_UPLOAD_BYTES", "KALOS_JOB_TIMEOUT_S",
                 "KALOS_JOBS_ENABLED"):
        monkeypatch.delenv(name, raising=False)
    s = CampaignStore(tmp_path)
    yield s
    s.close()


def _enqueue(store, *, anonymize: bool, filename: str = FILENAME, raw: bytes | None = None):
    cid = store.create_campaign(tenant="acme", name="c")["id"]
    job = jobs.enqueue(
        store, tenant="acme", campaign_id=cid, raw=raw if raw is not None else _csv(),
        filename=filename, target=TARGET, anonymize=anonymize, roles=ROLES, sheet="Runs",
        requested_by="tester",
    )
    return cid, job["id"]


def _stored_params(store, job_id: str) -> str:
    """The raw `params` column, exactly as it sits in portal.db."""
    with store._txn(write=False) as conn:
        return conn.execute("SELECT params FROM jobs WHERE id = ?", (job_id,)).fetchone()[0]


def _run(store, job_id: str, raw_claim: bool = True) -> str:
    if raw_claim:
        assert jobs.claim(store, tenant="acme", runner_id="test")["id"] == job_id
    return job_worker.run_job(store, job_id, analyze=_stub)


# --- while queued --------------------------------------------------------------- #


def test_anonymize_never_stores_the_filename(store):
    _, job_id = _enqueue(store, anonymize=True, filename="secret-plant-7.xlsx")
    stored = _stored_params(store, job_id)
    assert "secret-plant-7" not in stored
    params = json.loads(stored)
    assert params["ext"] == ".xlsx" and "filename" not in params
    assert params["anonymize"] is True


def test_without_anonymize_the_filename_is_stored_while_the_job_is_active(store):
    _, job_id = _enqueue(store, anonymize=False)
    assert json.loads(_stored_params(store, job_id))["filename"] == FILENAME


# --- after a terminal state ------------------------------------------------------ #


@pytest.mark.parametrize("anonymize", [True, False])
def test_params_are_cleared_when_the_job_succeeds(store, anonymize):
    _, job_id = _enqueue(store, anonymize=anonymize)
    assert _run(store, job_id) == "succeeded"
    stored = _stored_params(store, job_id)
    assert stored == "{}"
    assert "secret-plant-7" not in stored and TARGET not in stored and "Runs" not in stored


def test_params_are_cleared_when_the_job_fails(store):
    _, job_id = _enqueue(store, anonymize=True, raw=b"\x00\x01 not a table")
    assert _run(store, job_id) == "failed"
    assert _stored_params(store, job_id) == "{}"


def test_params_are_cleared_when_a_queued_job_is_cancelled(store):
    _, job_id = _enqueue(store, anonymize=False)
    jobs.request_cancel(store, tenant="acme", job_id=job_id)
    assert _stored_params(store, job_id) == "{}"


# --- the worker still parses by extension ---------------------------------------- #


def test_anonymized_job_still_uses_the_extension_for_format_detection(store):
    """A `.json` name is refused as a non-tabular type even though the bytes are
    a CSV: the anonymized job must keep the extension to make the same decision
    the sync route makes."""
    _, job_id = _enqueue(store, anonymize=True, filename="export.json")
    assert _run(store, job_id) == "failed"
    assert jobs.get_job(store, tenant="acme", job_id=job_id)["error"]["code"] == "unsupported_file_type"


def test_anonymized_job_without_extension_is_parsed_as_text(store):
    _, job_id = _enqueue(store, anonymize=True, filename="export")
    assert _run(store, job_id) == "succeeded"
