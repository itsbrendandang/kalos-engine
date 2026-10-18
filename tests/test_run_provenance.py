"""What the run record says about the fit it describes (docs/CAMPAIGNS_API.md,
"Run record"): the fit-row cap, the torch thread count (configured and actual,
with "unpinned" distinct from 0), and which sheet was analysed.
"""
from __future__ import annotations

import io
import json

import numpy as np
import pandas as pd
import pytest

from kalos.data.anonymizer import _hash
from kalos.portal import pipeline, uploads

USED = "Patient 12 log"
OTHER = "Tank 3 raw"


def _rows(n: int = 24, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    m = rng.uniform(0, 4, n)
    ph = rng.uniform(5, 7, n)
    return pd.DataFrame({"Methanol": m.round(3), "pH": ph.round(2),
                         "lipase_titer": (1.5 * m - 0.8 * (ph - 6) ** 2
                                          + rng.normal(0, 0.1, n)).round(3)})


def _stub(df, target, **kwargs):
    return {"target": target, "proposal_features": ["Methanol"], "d": 2, "proposals": [],
            "marker": "stub", "seed": 1, "promotion": {}, "proposal_gating": {},
            "validation": {"status": "pass"}}


def _xlsx(sheets: dict[str, pd.DataFrame]) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name, index=False)
    return buf.getvalue()


# --- the fit-row cap -------------------------------------------------------- #


def test_max_fit_rows_is_recorded_and_follows_the_engine_setting(monkeypatch):
    monkeypatch.setattr(uploads, "MAX_FIT_ROWS", 777)
    _, result = pipeline.compute_upload_analysis(_rows().to_csv(index=False).encode(),
                                                 "lipase_titer", False, "a.csv", analyze=_stub)
    assert result["run"]["max_fit_rows"] == 777


# --- torch threads: configured (None = unpinned) and actual ------------------- #


def test_unpinned_torch_threads_is_null_not_zero(monkeypatch):
    monkeypatch.setattr(pipeline, "TORCH_THREADS", 0)  # KALOS_TORCH_THREADS=0: do not pin
    _, result = pipeline.compute_upload_analysis(_rows().to_csv(index=False).encode(),
                                                 "lipase_titer", False, "a.csv", analyze=_stub)
    assert result["run"]["torch_threads"] is None


def test_pinned_torch_threads_is_the_configured_count(monkeypatch):
    monkeypatch.setattr(pipeline, "TORCH_THREADS", 3)
    _, result = pipeline.compute_upload_analysis(_rows().to_csv(index=False).encode(),
                                                 "lipase_titer", False, "a.csv", analyze=_stub)
    assert result["run"]["torch_threads"] == 3


def test_torch_threads_actual_is_what_torch_reports_after_a_real_fit():
    torch = pytest.importorskip("torch")
    _, result = pipeline.compute_upload_analysis(_rows().to_csv(index=False).encode(),
                                                 "lipase_titer", False, "a.csv")
    assert result["run"]["torch_threads_actual"] == torch.get_num_threads()
    assert result["run"]["torch_threads_actual"] >= 1


def test_torch_threads_actual_is_null_when_torch_was_never_loaded(monkeypatch):
    import sys

    monkeypatch.delitem(sys.modules, "torch", raising=False)
    assert pipeline.torch_threads_actual() is None


# --- the analysed sheet: index and salted hash, never the name under anonymize --- #


def _two_sheets() -> bytes:
    return _xlsx({OTHER: _rows(seed=1), USED: _rows(seed=2)})


def test_sheet_index_and_hash_identify_the_analysed_sheet():
    _, result = pipeline.compute_upload_analysis(_two_sheets(), "lipase_titer", False,
                                                 "book.xlsx", sheet=USED, analyze=_stub)
    assert result["run"]["sheet_index"] == 1
    assert result["run"]["sheet_hash"] == _hash(USED)
    assert result["run"]["sheet"] == USED


def test_under_anonymize_the_sheet_is_a_hash_and_an_index_only():
    _, result = pipeline.compute_upload_analysis(_two_sheets(), "lipase_titer", True,
                                                 "book.xlsx", sheet=USED, analyze=_stub)
    run = result["run"]
    assert run["sheet_index"] == 1 and run["sheet_hash"] == _hash(USED)
    assert "sheet" not in run
    assert USED not in json.dumps(result)


def test_a_csv_has_no_sheet_identity():
    _, result = pipeline.compute_upload_analysis(_rows().to_csv(index=False).encode(),
                                                 "lipase_titer", False, "a.csv", analyze=_stub)
    assert result["run"]["sheet_index"] is None and result["run"]["sheet_hash"] is None
