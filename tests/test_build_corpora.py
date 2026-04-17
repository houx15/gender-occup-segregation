#!/usr/bin/env python3
"""
Unit tests for corpus building utilities.
"""

import pytest
from pathlib import Path

from scripts.data_prep.build_corpora_ngram import (
    generate_time_slices,
    parse_ngram_line_v3,
)


class TestGenerateTimeSlices:
    """Test cases for generate_time_slices."""

    def test_basic(self):
        """Test basic time slice generation."""
        slices = generate_time_slices(1940, 2015, 10, 5)
        assert len(slices) > 0

        # First slice should start at 1940
        assert slices[0][0] == 1940

        # Check window size
        assert slices[0][1] - slices[0][0] + 1 == 10

        # Check step size
        if len(slices) > 1:
            assert slices[1][0] - slices[0][0] == 5

    def test_exact(self):
        """Test specific time slice outputs."""
        slices = generate_time_slices(1940, 1960, 10, 5)

        expected = [
            (1940, 1949),
            (1945, 1954),
            (1950, 1959),
            (1955, 1960),
            (1960, 1960),
        ]

        assert slices == expected


class TestParseNgramLineV3:
    """Tests for parse_ngram_line_v3."""

    def test_valid_line(self):
        """Parse a well-formed v3 line: ngram<TAB>year,c1,c2 ..."""
        # clean_ngram requires space-separated multi-token ngrams
        line = "工人 阶级\t1950,100,10\t1951,200,20"
        entries = parse_ngram_line_v3(line)
        assert len(entries) == 2
        ngram, year, count = entries[0]
        assert ngram == "工人 阶级"
        assert year == 1950
        assert count == 100

    def test_invalid_line(self):
        """Malformed line (no tab) returns empty list."""
        line = "invalid line without tabs"
        entries = parse_ngram_line_v3(line)
        assert entries == []

    def test_missing_year_count(self):
        """Line with tab but no parseable year,count returns empty list."""
        line = "你 好\tnotanumber"
        entries = parse_ngram_line_v3(line)
        assert entries == []


class TestCorpusIntegration:
    """Integration tests using corpus builder helpers."""

    def test_corpus_file_creation(self, tmp_path):
        """Manually build a filtered corpus and verify output."""
        ngram_file = tmp_path / "test_ngrams.txt"
        with open(ngram_file, "w", encoding="utf-8") as f:
            # v3 format: space-separated tokens <TAB> year,c1,c2
            # clean_ngram requires at least 2 space-separated tokens
            f.write("这 是 一 个 测试\t1945,10,5\n")
            f.write("另 一 个 测试 句\t1948,5,3\n")
            f.write("超 出 范围 的 句\t1960,3,2\n")

        corpus_file = tmp_path / "corpus.txt"

        with open(corpus_file, "w", encoding="utf-8") as out_f:
            with open(ngram_file, "r", encoding="utf-8") as in_f:
                for line in in_f:
                    entries = parse_ngram_line_v3(line)
                    for ngram, year, _count in entries:
                        if 1940 <= year <= 1949:
                            out_f.write(ngram + "\n")

        with open(corpus_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        assert len(lines) == 2
        assert lines[0].strip() == "这 是 一 个 测试"
        assert lines[1].strip() == "另 一 个 测试 句"


class TestRmrbBuilder:
    """Tests for the Renminribao corpus builder."""

    def test_rmrb_builder_produces_corpus(self, tmp_path):
        pytest.importorskip("jieba")

        import logging
        from scripts.data_prep.build_corpora_rmrb import build_corpora

        # Create fixture directory structure matching rmrb path convention:
        # {decade}s/{year}/报刊/人民日报/rmrb_{year}_{month}.txt
        raw = tmp_path / "raw" / "1940s" / "1945" / "报刊" / "人民日报"
        raw.mkdir(parents=True)
        (raw / "rmrb_1945_05.txt").write_text(
            "新华社北京电 我爱北京天安门 http://example.com 工人阶级伟大\n" * 20,
            encoding="gb18030",
        )

        corpora_dir = tmp_path / "data" / "corpora"

        config = {
            "language": "zh",
            "data_source": "renminribao",
            "analysis_unit": "longitudinal",
            "analysis_mode": "prestige",
            "corpus": {
                "tokenizer": "jieba",
                "stopwords": "zh_newspaper",
                "lowercase": False,
                "min_words": 2,
            },
            "paths": {
                "base_dir": str(tmp_path),
                "raw_rmrb_dir": str(tmp_path / "raw"),
                "corpora_dir": str(corpora_dir),
                "models_dir": str(tmp_path / "data" / "models"),
                "results_dir": str(tmp_path / "data" / "results"),
                "log_dir": str(tmp_path / "logs"),
            },
            "time_slices": {
                "start_year": 1940,
                "end_year": 1949,
                "window_size": 10,
                "step_size": 10,
            },
        }

        logger = logging.getLogger("test_rmrb")
        build_corpora(config, logger, data_dir=str(tmp_path / "raw"))

        # The slice dir should be 1940_1949
        slice_dir = corpora_dir / "1940_1949"
        assert slice_dir.exists(), f"Expected slice dir {slice_dir} to be created"

        corpus_files = list(slice_dir.glob("corpus_*.txt"))
        assert len(corpus_files) > 0, "Expected at least one corpus file"

        total_lines = sum(
            len(cf.read_text(encoding="utf-8").strip().splitlines())
            for cf in corpus_files
        )
        assert total_lines > 0, "Expected non-empty corpus output"
