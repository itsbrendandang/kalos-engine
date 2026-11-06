"""A `condition,y` sheet (one text factor + one numeric target) and its 3-column
variant were read as transposed because labels such as "A pH 4" hit the "ph"
keyword, then refused as "1 data row". They parse as normal sheets now. When a
transposed reading still leaves too few runs, the refusal names the sheet's
real shape.
"""
from __future__ import annotations

import pytest

from kalos.normalize.orientation import detect_orientation
from kalos.portal.uploads import UploadRejected, _parse_upload

_LABELS = ["A pH 4", "B pH 4", "A pH 3", "B pH 3", "A YE 1"] * 2
_Y = [10.0, 11.7, 13.4, 15.1, 16.8, 10.4, 11.9, 13.0, 15.6, 16.2]


def _two_col() -> bytes:
    rows = ["condition,y", *[f"{c},{y}" for c, y in zip(_LABELS, _Y)]]
    return ("\n".join(rows) + "\n").encode()


def _three_col() -> bytes:
    rows = ["condition,x,y", *[f"{c},{i * 0.5},{y}" for i, (c, y) in enumerate(zip(_LABELS, _Y))]]
    return ("\n".join(rows) + "\n").encode()


def test_two_column_text_factor_sheet_parses_as_standard():
    df = _parse_upload(_two_col())
    assert df.shape == (10, 2)
    assert list(df.columns) == ["condition", "y"]
    assert df.attrs["kalos_orientation"]["orientation"] == "standard"


def test_three_column_variant_parses_as_standard():
    df = _parse_upload(_three_col())
    assert df.shape == (10, 3)
    assert list(df.columns) == ["condition", "x", "y"]


def test_unique_labels_two_columns_tall_sheet_is_standard():
    import pandas as pd

    df = pd.DataFrame({
        "condition": [f"pH {i} feed" for i in range(8)], "y": [float(i) for i in range(8)],
    })
    assert detect_orientation(df).orientation == "standard"


def test_two_column_sheet_analyzes():
    pytest.importorskip("torch")
    from kalos.portal.analysis import _analyze

    out = _analyze(_parse_upload(_two_col()), target="y")
    assert out["n"] == 10
    assert out["categorical_features"] == ["condition"]


def test_transposed_refusal_names_the_sheet_as_written():
    from kalos.portal.analysis import _analyze

    text = (
        "parameter,exp1,exp2,exp3\n"
        "temperature,30,32,35\n"
        "ph_setpoint,6.5,7.0,7.2\n"
        "induction_duration,4,5,4.5\n"
        "titer_output,8.5,9.2,10.1\n"
    )
    df = _parse_upload(text.encode())
    assert df.attrs["kalos_orientation"]["orientation"] == "transposed"
    with pytest.raises(UploadRejected) as exc:
        _analyze(df, target="titer_output", preflight_only=True)
    assert exc.value.code == "too_few_rows"
    assert "4 data rows and 4 columns" in exc.value.message
    assert "gives 3 runs" in exc.value.message
    assert exc.value.context["rows_as_written"] == 4
