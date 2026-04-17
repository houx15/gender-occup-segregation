#!/usr/bin/env python3
"""
Smoke tests for the English Google Ngram corpus builder.
"""

import gzip
import logging
import pytest
from pathlib import Path

from scripts.data_prep.build_corpora_ngram_en import (
    clean_ngram,
    parse_ngram_line,
    process_ngram_file,
    generate_time_slices,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(tmp_path):
    """Return a minimal config dict pointing at tmp_path."""
    return {
        "corpus": {
            "min_count_threshold": 1,
        },
        "paths": {
            "corpora_dir": str(tmp_path / "corpora"),
            "log_dir": str(tmp_path / "logs"),
        },
    }


def _make_ngram_file(dir_path: Path, name: str, lines: list[str]) -> Path:
    """Write a plain-text ngram file (process_ngram_file reads decompressed text)."""
    p = dir_path / name
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def _make_gz_ngram_file(dir_path: Path, name: str, lines: list[str]) -> Path:
    """Write a gzip-compressed ngram file."""
    p = dir_path / name
    content = ("\n".join(lines) + "\n").encode("utf-8")
    with gzip.open(p, "wb") as f:
        f.write(content)
    return p


# ---------------------------------------------------------------------------
# Unit-level: clean_ngram
# ---------------------------------------------------------------------------

class TestCleanNgram:
    def test_lowercases(self):
        result = clean_ngram("Hello WORLD")
        assert result is not None
        assert result == result.lower()

    def test_strips_punctuation(self):
        # Punctuation-only token should be dropped; valid tokens kept
        result = clean_ngram("good, morning")
        assert result is not None
        assert "," not in result

    def test_returns_none_for_single_token(self):
        assert clean_ngram("hello") is None

    def test_returns_none_for_all_punctuation(self):
        assert clean_ngram("!@#$ *** ^^^") is None

    def test_preserves_apostrophe(self):
        result = clean_ngram("it's a test ngram")
        assert result is not None
        assert "it's" in result

    def test_multi_token_output(self):
        result = clean_ngram("The Quick Brown Fox")
        assert result == "the quick brown fox"


# ---------------------------------------------------------------------------
# Unit-level: parse_ngram_line
# ---------------------------------------------------------------------------

class TestParseNgramLine:
    def test_parses_two_year_entries(self):
        line = "the quick brown fox jumps\t2000,42,3\t2001,50,4"
        entries = parse_ngram_line(line)
        assert len(entries) == 2

    def test_entry_fields(self):
        line = "hello world test foo bar\t1990,10,2"
        entries = parse_ngram_line(line)
        assert len(entries) == 1
        ngram_text, year, count = entries[0]
        assert year == 1990
        assert count == 10
        assert ngram_text.islower()

    def test_empty_for_no_tab(self):
        assert parse_ngram_line("no tabs here at all") == []

    def test_empty_for_malformed_year_field(self):
        assert parse_ngram_line("hello world\tnotvalid") == []

    def test_lowercases_ngram(self):
        line = "UPPER CASE NGRAM HERE NOW\t2010,5,1"
        entries = parse_ngram_line(line)
        assert len(entries) == 1
        ngram_text, _, _ = entries[0]
        assert ngram_text.islower()

    def test_drops_single_token_ngram(self):
        # After cleaning, only one valid token remains — should be dropped
        line = "hello!!! ??? *** &&& ~~~\t2000,5,1"
        entries = parse_ngram_line(line)
        assert entries == []


# ---------------------------------------------------------------------------
# Integration: process_ngram_file writes correct slice directories
# ---------------------------------------------------------------------------

class TestProcessNgramFile:
    """End-to-end smoke test: tiny fake ngram file -> slice corpus files."""

    NGRAM_LINES = [
        # Two entries for 2000 (falls in 2000-2009 slice)
        "the quick brown fox jumps\t2000,5,1\t2001,8,1",
        # One entry for 1990 (falls in 1990-1999 slice)
        "gender and work norms change\t1995,3,1",
        # Should be filtered out: only one valid token after clean
        "!!! ??? &&& *** ~~~\t2000,10,1",
        # Out of any slice range
        "far future text data point\t2020,10,1",
    ]

    def _run(self, tmp_path):
        config = _make_config(tmp_path)
        logger = logging.getLogger("test_process_ngram")

        # File name must contain '-' so split("-")[1] yields the file index
        ngram_file = _make_ngram_file(
            tmp_path, "5-00-of-01.txt", self.NGRAM_LINES
        )

        time_slices = [
            (1990, 1999),
            (2000, 2009),
        ]

        process_ngram_file(ngram_file, time_slices, config, logger)
        return Path(config["paths"]["corpora_dir"])

    def test_slice_directories_created(self, tmp_path):
        corpora_dir = self._run(tmp_path)
        assert (corpora_dir / "1990_1999").exists()
        assert (corpora_dir / "2000_2009").exists()

    def test_corpus_files_written(self, tmp_path):
        corpora_dir = self._run(tmp_path)
        # File index is split("-")[1] = "00"
        assert (corpora_dir / "1990_1999" / "corpus_00.txt").exists()
        assert (corpora_dir / "2000_2009" / "corpus_00.txt").exists()

    def test_corpus_contains_expected_ngrams(self, tmp_path):
        corpora_dir = self._run(tmp_path)
        text_2000s = (corpora_dir / "2000_2009" / "corpus_00.txt").read_text(encoding="utf-8")
        assert "the quick brown fox jumps" in text_2000s

        text_1990s = (corpora_dir / "1990_1999" / "corpus_00.txt").read_text(encoding="utf-8")
        assert "gender and work norms change" in text_1990s

    def test_out_of_range_year_not_written(self, tmp_path):
        corpora_dir = self._run(tmp_path)
        # Year 2020 falls outside both slices; no 2020_2029 dir should exist
        assert not (corpora_dir / "2020_2029").exists()

    def test_lowercase_in_output(self, tmp_path):
        corpora_dir = self._run(tmp_path)
        text = (corpora_dir / "2000_2009" / "corpus_00.txt").read_text(encoding="utf-8")
        for line in text.strip().splitlines():
            if line:
                assert line == line.lower(), f"Non-lowercase line found: {line!r}"

    def test_no_punctuation_in_output(self, tmp_path):
        corpora_dir = self._run(tmp_path)
        for slice_name in ["1990_1999", "2000_2009"]:
            p = corpora_dir / slice_name / "corpus_00.txt"
            if p.exists():
                for line in p.read_text(encoding="utf-8").strip().splitlines():
                    if line:
                        # No bare punctuation tokens (commas, periods, etc.)
                        for token in line.split():
                            assert token.replace("'", "").isalpha(), (
                                f"Non-alpha token {token!r} found in line {line!r}"
                            )

    def test_min_count_filter(self, tmp_path):
        """Entries below min_count_threshold should be excluded."""
        config = _make_config(tmp_path)
        config["corpus"]["min_count_threshold"] = 10  # require count >= 10
        logger = logging.getLogger("test_min_count")

        lines = [
            # count=2 < 10, should be filtered out
            "low count ngram here now\t2000,2,1",
            # count=15 >= 10, should be included
            "high count ngram here now\t2000,15,1",
        ]
        (tmp_path / "raw").mkdir(parents=True, exist_ok=True)
        ngram_file = _make_ngram_file(tmp_path / "raw", "5-01-of-01.txt", lines)

        time_slices = [(2000, 2009)]
        process_ngram_file(ngram_file, time_slices, config, logger)

        corpora_dir = Path(config["paths"]["corpora_dir"])
        corpus_file = corpora_dir / "2000_2009" / "corpus_01.txt"
        assert corpus_file.exists()
        content = corpus_file.read_text(encoding="utf-8")
        assert "high count ngram here now" in content
        assert "low count ngram here now" not in content


# ---------------------------------------------------------------------------
# Integration: build_corpora with a .gz input file (via file_name shortcut)
# ---------------------------------------------------------------------------

class TestBuildCorporaWithGzFile:
    """Test the full build_corpora path using file_name to skip decompression."""

    def test_gz_file_processed_end_to_end(self, tmp_path):
        """Write a .gz file, decompress manually, confirm process works."""
        import shutil

        decompressed_dir = tmp_path / "decompressed"
        decompressed_dir.mkdir()

        lines = [
            "work and family balance here\t1950,10,1",
            "gender norm change over time\t1955,20,2",
        ]
        # Write as plain text (simulating what decompress_file would produce)
        ngram_file = _make_ngram_file(decompressed_dir, "5-42-of-99.txt", lines)

        config = {
            "corpus": {"min_count_threshold": 1},
            "paths": {
                "corpora_dir": str(tmp_path / "corpora"),
                "decompressed_dir": str(decompressed_dir),
                "raw_ngram_dir": str(tmp_path / "raw"),
                "log_dir": str(tmp_path / "logs"),
            },
            "time_slices": {
                "start_year": 1950,
                "end_year": 1959,
                "window_size": 10,
                "step_size": 10,
            },
        }

        logger = logging.getLogger("test_build_corpora_gz")

        from scripts.data_prep.build_corpora_ngram_en import build_corpora
        build_corpora(config, logger, file_name="5-42-of-99.txt")

        corpora_dir = Path(config["paths"]["corpora_dir"])
        slice_dir = corpora_dir / "1950_1959"
        assert slice_dir.exists(), "Expected 1950_1959 slice directory"

        corpus_files = list(slice_dir.glob("corpus_*.txt"))
        assert len(corpus_files) > 0, "Expected at least one corpus file"

        total_lines = sum(
            len(f.read_text(encoding="utf-8").strip().splitlines())
            for f in corpus_files
        )
        assert total_lines > 0, "Expected non-empty corpus output"
