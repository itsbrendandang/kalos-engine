"""`python -m kalos.bench --help` prints help and exits; it must not start the
full sweep (VERIFY_MAIN NIT 12). An unknown flag is an error, not a silent sweep."""
from __future__ import annotations

import pytest

from kalos.bench.__main__ import main


def test_help_prints_usage_and_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "usage: python -m kalos.bench" in out and "--vs-baselines" in out


def test_unknown_flag_is_an_error(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--no-such-flag"])
    assert exc.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err
