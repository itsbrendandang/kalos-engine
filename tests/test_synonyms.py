"""Tests for `kalos.normalize.synonyms`: `snake_canonical` edge cases, the
`SYNONYMS` prior's invariants, `guess_role` decision precedence and the
`parse_rate` boundary, and the (previously untested) `preferred_target`.

Complements the happy-path cases in `tests/test_normalize_core.py`.
"""
from __future__ import annotations

import pytest

from kalos.normalize.synonyms import (
    SYNONYMS,
    guess_role,
    preferred_target,
    snake_canonical,
)

# --- snake_canonical: edge cases ------------------------------------------- #


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("", ""),
        ("   ", ""),
        ("(g/L)", ""),  # a header that is only a unit annotation collapses to nothing
        ("--Run ID--", "run_id"),  # leading/trailing punctuation runs are stripped
        ("OD600", "od600"),  # digits survive, letters are lower-cased
        ("Feed  Rate (mL/h) ", "feed_rate"),  # multiple separators collapse to one "_"
    ],
)
def test_snake_canonical_edge_cases(raw: str, expected: str) -> None:
    assert snake_canonical(raw) == expected


def test_snake_canonical_is_idempotent() -> None:
    once = snake_canonical("Titer (g/L)")
    assert snake_canonical(once) == once


# --- SYNONYMS prior invariants --------------------------------------------- #


def test_synonyms_aliases_are_canonical_and_include_their_key() -> None:
    for canonical, aliases in SYNONYMS.items():
        assert canonical in aliases
        for alias in aliases:
            assert snake_canonical(alias) == alias


def test_synonyms_aliases_do_not_collide_across_keys() -> None:
    seen: dict[str, str] = {}
    for canonical, aliases in SYNONYMS.items():
        for alias in aliases:
            assert alias not in seen, f"{alias!r} listed under both {seen.get(alias)!r} and {canonical!r}"
            seen[alias] = canonical


# --- guess_role: precedence and boundaries --------------------------------- #


def test_identity_wins_over_group_hint() -> None:
    # "strain" is an anonymizer DROP substring; "batch" is a HASH substring.
    # Identity is checked first, so the column is dropped, not hashed.
    assert guess_role("Strain Batch", is_numeric=False, parse_rate=0.0) == "identity"


def test_group_wins_over_outcome_even_when_numeric() -> None:
    # "experiment" is an anonymizer HASH substring; numeric-ness is irrelevant.
    assert guess_role("Experiment Day", is_numeric=True, parse_rate=1.0) == "group"
    # Bare _GROUP_HINT match, no anonymizer rule involved.
    assert guess_role("Medium", is_numeric=True, parse_rate=1.0) == "group"


def test_outcome_header_requires_numeric_to_be_target() -> None:
    assert guess_role("Titer", is_numeric=True, parse_rate=1.0) == "target"
    # Same header, non-numeric column: falls through to the text branches.
    assert guess_role("Titer", is_numeric=False, parse_rate=0.0) == "freetext"


def test_parse_rate_boundary_at_0_8() -> None:
    assert guess_role("Notes", is_numeric=False, parse_rate=0.79) == "freetext"
    assert guess_role("Notes", is_numeric=False, parse_rate=0.8) == "metadata"


# --- preferred_target ------------------------------------------------------ #


def test_preferred_target_picks_first_titer_like_candidate() -> None:
    assert preferred_target(["OD600", "Yield (%)", "Titer"]) == "Yield (%)"


def test_preferred_target_falls_back_to_first_candidate() -> None:
    assert preferred_target(["OD600", "Biomass"]) == "OD600"
    assert preferred_target(["Product Conc"]) == "Product Conc"


def test_preferred_target_is_case_insensitive_and_deterministic() -> None:
    candidates = ["biomass", "TITRE", "titer"]
    assert preferred_target(candidates) == "TITRE"
    assert preferred_target(candidates) == preferred_target(list(candidates))
