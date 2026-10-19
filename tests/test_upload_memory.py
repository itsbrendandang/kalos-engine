"""Upload memory amplification (review M2).

A 24 MB CSV with 4.2 M rows used to peak near 527 MB RSS: the whole file was
parsed by pandas BEFORE the row cap was checked, and header detection parsed it
again. The row count is now taken by a streaming pass before any pandas parse,
and pandas reads the file once with the header row already decided.

These tests use a call-count bound (pandas is never asked to parse an over-cap
file) and an RSS bound measured in a fresh process. No wall-clock assertions.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import pandas as pd
import pytest

from kalos.portal import uploads
from kalos.portal.uploads import UploadRejected, parse_upload

_HEADER = "run_id,temp_C,ph,titer_g_L\n"


def _csv(rows: int, *, start: int = 0) -> bytes:
    body = "".join(f"{i},{30 + (i % 7)},{6.5 + (i % 3) / 10},{1.0 + (i % 11)}\n"
                   for i in range(start, start + rows))
    return (_HEADER + body).encode()


@pytest.fixture
def read_csv_calls(monkeypatch):
    """Count pandas parses of an upload (the expensive step) without changing them."""
    calls: list[int] = []
    real = pd.read_csv

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(uploads.pd, "read_csv", counting)
    return calls


def test_over_cap_csv_is_refused_before_pandas_parses_it(read_csv_calls):
    raw = _csv(uploads.MAX_CSV_ROWS + 1)
    with pytest.raises(UploadRejected) as exc:
        parse_upload(raw, filename="runs.csv")
    assert exc.value.code == "too_many_rows"
    assert read_csv_calls == []  # the refusal happens on the streaming count alone


def test_at_the_row_cap_is_accepted_and_read_once(read_csv_calls):
    parsed = parse_upload(_csv(uploads.MAX_CSV_ROWS), filename="runs.csv")
    assert len(parsed.df) == uploads.MAX_CSV_ROWS
    assert read_csv_calls == [1]  # one pandas parse, header detection did not re-read


def test_row_cap_boundary_with_a_group_label_header(monkeypatch):
    """With the header on row 2 the data has one row fewer, so the cap is exact."""
    monkeypatch.setattr(uploads, "MAX_CSV_ROWS", 20)
    group = "Process,,,Analytics\n"
    body = "".join(f"{i},{i * 2}\n" for i in range(20))  # 20 data rows
    ok = (group + "temp,ph\n" + "".join(f"{i},{i * 2}\n" for i in range(20))).encode()
    assert len(parse_upload(ok, filename="g.csv").df) == 20
    over = (group + "temp,ph\n" + body + "1,2\n").encode()  # 21 data rows
    with pytest.raises(UploadRejected) as exc:
        parse_upload(over, filename="g.csv")
    assert exc.value.code == "too_many_rows"


def test_blank_lines_are_not_counted_as_rows(monkeypatch):
    monkeypatch.setattr(uploads, "MAX_CSV_ROWS", 5)
    raw = ("a,b\n\n1,2\n\n3,4\n\n5,6\n\n7,8\n\n9,10\n\n").encode()
    assert len(parse_upload(raw, filename="b.csv").df) == 5


def test_a_quoted_newline_is_one_row(monkeypatch):
    monkeypatch.setattr(uploads, "MAX_CSV_ROWS", 5)
    raw = ('a,b\n"multi\nline",1\n2,2\n3,3\n4,4\n5,5\n').encode()
    assert len(parse_upload(raw, filename="q.csv").df) == 5


def test_a_group_header_file_is_read_once(read_csv_calls):
    raw = (
        "Process,,,Analytics\n" + "temp,ph,flow,titer\n"
        + "".join(f"{30 + i},{6.5 + i / 10},{i},{1.0 + i}\n" for i in range(12))
    ).encode()
    parsed = parse_upload(raw, filename="g.csv")
    assert list(parsed.df.columns) == ["temp", "ph", "flow", "titer"]
    assert read_csv_calls == [1]


def test_over_cap_csv_rss_growth_is_bounded():
    """A fresh process: the 4.2 M-row file (about 24 MB) must not grow the
    resident set by pandas-sized amounts before it is refused."""
    script = textwrap.dedent("""
        import json, resource, sys
        from kalos.portal.uploads import UploadRejected, parse_upload
        rows = 4_200_000
        raw = b"a,b\\n" + b"1,2\\n" * rows
        assert len(raw) < 25 * 1024 * 1024
        def rss():
            value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return value if sys.platform == "darwin" else value * 1024
        before = rss()
        try:
            parse_upload(raw, filename="big.csv")
            code = None
        except UploadRejected as rej:
            code = rej.code
        print(json.dumps({"code": code, "growth": rss() - before}))
    """)
    env = {k: v for k, v in os.environ.items() if not k.startswith("KALOS_")}
    out = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True,
                         text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    result = json.loads(out.stdout.strip().splitlines()[-1])
    assert result["code"] == "too_many_rows"
    # Old code: ~500 MB of growth. Refused early: the decoded text (about 17 MB)
    # is the only large allocation. The bound leaves room for allocator noise.
    assert result["growth"] < 120 * 1024 * 1024, result


@pytest.mark.parametrize("eol", ["\r\n", "\r", "\n"])
def test_every_line_ending_counts_one_row(monkeypatch, eol):
    """CRLF, bare CR and LF all end a record, as they do for pandas."""
    monkeypatch.setattr(uploads, "MAX_CSV_ROWS", 4)
    raw = eol.join(["a,b", "1,2", "3,4", "5,6", "7,8", ""]).encode()
    assert len(parse_upload(raw, filename="eol.csv").df) == 4
    over = eol.join(["a,b", "1,2", "3,4", "5,6", "7,8", "9,10", ""]).encode()
    with pytest.raises(UploadRejected) as exc:
        parse_upload(over, filename="eol.csv")
    assert exc.value.code == "too_many_rows"
