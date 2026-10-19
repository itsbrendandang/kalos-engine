"""Sheet names are the uploader's words ("Patient 12 log"). Under anonymize
they must not reach any response, error body, stored field or log-facing
context: a refused sheet, a no-usable-sheet list, the analyzed sheet, and the
annotation an analysis error carries. Before this, the parse step ran outside
the anonymizing wrapper, so `sheet_not_found` listed every real sheet name.
"""
from __future__ import annotations

import io
import json

import numpy as np
import pandas as pd
import pytest

from kalos.portal import pipeline
from kalos.portal.uploads import UploadRejected

USED = "Patient 12 log"
OTHER = "Tank 3 raw"


def _good(n: int = 12, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    m = rng.uniform(0, 4, n)
    ph = rng.uniform(5, 7, n)
    return pd.DataFrame({"Methanol": m.round(3), "pH": ph.round(2),
                         "lipase_titer": (1.5 * m + rng.normal(0, 0.1, n)).round(3)})


def _xlsx(sheets: dict[str, pd.DataFrame]) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name, index=False)
    return buf.getvalue()


def _stub(df, target, **kwargs):
    return {"target": target, "proposal_features": ["Methanol"], "d": 2, "proposals": [],
            "marker": "stub", "seed": 1, "promotion": {}, "proposal_gating": {},
            "validation": {"status": "pass"}}


def _raw() -> bytes:
    return _xlsx({OTHER: _good(seed=1), USED: _good(seed=2)})


def test_sheet_not_found_lists_no_sheet_name_under_anonymize():
    with pytest.raises(UploadRejected) as exc:
        pipeline.preflight_upload(_raw(), "lipase_titer", True, "book.xlsx", sheet="Patient 1")
    rej = exc.value
    assert rej.code == "sheet_not_found"
    blob = rej.message + json.dumps(rej.context)
    assert USED not in blob and OTHER not in blob
    assert all(name.startswith("sheet_") for name in rej.context["sheets"])
    assert len(rej.context["sheets"]) == 2


def test_sheet_not_found_still_lists_names_without_anonymize():
    """The non-anonymous refusal is unchanged: the names are the uploader's own."""
    with pytest.raises(UploadRejected) as exc:
        pipeline.preflight_upload(_raw(), "lipase_titer", False, "book.xlsx", sheet="Patient 1")
    assert exc.value.context["sheets"] == [OTHER, USED]


def test_no_usable_sheet_message_names_no_sheet_under_anonymize():
    raw = _xlsx({USED: pd.DataFrame({"note": ["x"]}), OTHER: pd.DataFrame({"note": ["y"]})})
    with pytest.raises(UploadRejected) as exc:
        pipeline.compute_upload_analysis(raw, "lipase_titer", True, "book.xlsx",
                                         analyze=_stub)
    assert exc.value.code == "no_usable_sheet"
    blob = exc.value.message + json.dumps(exc.value.context)
    assert USED not in blob and OTHER not in blob


def test_analyzed_sheet_is_aliased_in_the_response_and_run_record_under_anonymize():
    _, result = pipeline.compute_upload_analysis(_raw(), "lipase_titer", True, "book.xlsx",
                                                 sheet=USED, analyze=_stub)
    assert result["sheet_used"].startswith("sheet_")
    assert len(result["sheets"]) == 2 and all(s.startswith("sheet_") for s in result["sheets"])
    blob = json.dumps(result)
    assert USED not in blob and OTHER not in blob
    assert "sheet" not in result["run"]


def test_analyzed_sheet_is_named_as_before_without_anonymize():
    _, result = pipeline.compute_upload_analysis(_raw(), "lipase_titer", False, "book.xlsx",
                                                 sheet=USED, analyze=_stub)
    assert result["sheet_used"] == USED and result["sheets"] == [OTHER, USED]
    assert result["run"]["sheet"] == USED


def test_analysis_error_annotation_names_no_sheet_under_anonymize():
    """The sentence an analysis error gets ("This was sheet X; the workbook also
    has Y") is built from the parsed sheet names, so it must be aliased too."""
    def refuse(df, target, **kwargs):
        raise UploadRejected("the data is unusable", code="target_constant")

    with pytest.raises(UploadRejected) as exc:
        pipeline.compute_upload_analysis(_raw(), "lipase_titer", True, "book.xlsx",
                                         sheet=USED, analyze=refuse)
    blob = exc.value.message + json.dumps(exc.value.context)
    assert USED not in blob and OTHER not in blob
    assert "This was sheet" in exc.value.message
