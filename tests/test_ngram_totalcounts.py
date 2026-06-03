"""Tests for the Google Books v3 totalcounts-5 parser."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.data_prep.ngram_totalcounts import (
    parse_totalcounts_text,
    load_totalcounts,
)


class TestParseTotalcountsText:
    def test_parses_year_to_match_count(self):
        text = "1500,117,6,1\t1501,200,8,2\t1502,50,3,1"
        result = parse_totalcounts_text(text)
        assert result == {1500: 117, 1501: 200, 1502: 50}

    def test_empty_input_returns_empty_dict(self):
        assert parse_totalcounts_text("") == {}
        assert parse_totalcounts_text("\n") == {}

    def test_malformed_cells_are_skipped(self):
        # First cell malformed (missing words), second cell good, third cell non-int year.
        text = "1500\t1501,200,8,2\tABCD,300,10,3"
        result = parse_totalcounts_text(text)
        assert result == {1501: 200}

    def test_trailing_whitespace_and_newlines_handled(self):
        text = "1500,117,6,1\t1501,200,8,2\n"
        assert parse_totalcounts_text(text) == {1500: 117, 1501: 200}


class TestLoadTotalcounts:
    def test_reads_file_from_disk(self, tmp_path: Path):
        p = tmp_path / "totalcounts-5"
        p.write_text("1500,117,6,1\t1501,200,8,2\n", encoding="utf-8")
        assert load_totalcounts(p) == {1500: 117, 1501: 200}

    def test_missing_file_raises_file_not_found(self, tmp_path: Path):
        p = tmp_path / "does-not-exist"
        with pytest.raises(FileNotFoundError):
            load_totalcounts(p)
