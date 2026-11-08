"""Tests for `kalos.features.protein.clean_sequence`: the amino-acid filter.

This is a pure, deterministic function that strips whitespace, uppercases,
and removes any character that is not a valid one-letter amino-acid code.

It is called by both `KmerEmbedder` and `ESM2Embedder`, so its contract
(TypesError on non-str, ValueError on empty-after-cleaning, correct filtering)
must be stable.
"""
from __future__ import annotations

import pytest

from kalos.features import clean_sequence


class TestCleanSequenceBasic:
    """Happy-path: clean sequences pass through with only casing/whitespace fixes."""

    def test_uppercase_stays_unchanged(self) -> None:
        assert clean_sequence("MVLSPADKTNVKAAWGKVGA") == "MVLSPADKTNVKAAWGKVGA"

    def test_lowercase_uppercased(self) -> None:
        assert clean_sequence("mv lsp adk") == "MVLSPADK"

    def test_mixed_case_normalised(self) -> None:
        assert clean_sequence("MvLsPaDk") == "MVLSPADK"

    def test_whitespace_stripped(self) -> None:
        assert clean_sequence("  MVLSPADK  ") == "MVLSPADK"

    def test_newlines_stripped(self) -> None:
        assert clean_sequence("MV\nLS\nPA\nDK") == "MVLSPADK"

    def test_tab_stripped(self) -> None:
        assert clean_sequence("MV\tLS") == "MVLS"


class TestCleanSequenceFiltering:
    """Non-amino-acid characters are removed."""

    def test_numeric_digits_removed(self) -> None:
        assert clean_sequence("MVL123SPADK") == "MVLSPADK"

    def test_special_chars_removed(self) -> None:
        assert clean_sequence("MV!@#$%LSPADK") == "MVLSPADK"

    def test_spaces_removed(self) -> None:
        assert clean_sequence("M V L S P A D K") == "MVLSPADK"

    def test_only_non_aa_raises(self) -> None:
        with pytest.raises(ValueError, match="no valid amino-acid residues"):
            clean_sequence("123!@#")

    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValueError, match="no valid amino-acid residues"):
            clean_sequence("")

    def test_whitespace_only_raises(self) -> None:
        with pytest.raises(ValueError, match="no valid amino-acid residues"):
            clean_sequence("   ")

    def test_none_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="protein sequence must be a string"):
            clean_sequence(None)  # type: ignore[arg-type]

    def test_int_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="protein sequence must be a string"):
            clean_sequence(42)  # type: ignore[arg-type]


class TestCleanSequenceAminoAcidSet:
    """Only the standard 20 one-letter codes are kept."""

    def test_all_20_amino_acids(self) -> None:
        # The full IUPAC one-letter alphabet
        aa = "ACDEFGHIKLMNPQRSTVWY"
        assert clean_sequence(aa) == aa

    def test_invalid_letter_removed(self) -> None:
        # B, J, O, U, X, Z are non-standard and should be stripped
        assert clean_sequence("ABCDEFBOGHIJKL") == "ACDEFGHIKL"

    def test_sequence_with_trailing_garbage(self) -> None:
        # 123 and @#$ are not amino-acid letters
        assert clean_sequence("MVLSPADK123@#$") == "MVLSPADK"
