"""Tests for the Chinese Google Ngram corpus builder — weight_mode dispatch."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from scripts.data_prep.build_corpora_ngram import process_ngram_file

logger = logging.getLogger("test")


def _make_ngram_file(dir_path: Path, name: str, lines: list[str]) -> Path:
    """Write a plain-text v3 ngram file (process_ngram_file reads decompressed text)."""
    p = dir_path / name
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def _line(ngram_text: str, year: int, match_count: int, volume_count: int = 1) -> str:
    """One v3 ngram line: ngram\\tyear,match,volume."""
    return f"{ngram_text}\t{year},{match_count},{volume_count}"


def _config(tmp_path: Path, **overrides) -> dict:
    """Minimal config dict for process_ngram_file."""
    corpus = {"min_count_threshold": 1}
    corpus.update(overrides.pop("corpus", {}))
    return {
        "corpus": corpus,
        "paths": {"corpora_dir": str(tmp_path / "corpora")},
        **overrides,
    }


def _read_corpus(corpora_dir: Path, slice_name: str) -> list[str]:
    """Read all corpus_*.txt lines for a slice (sorted by filename), return list of lines."""
    slice_dir = corpora_dir / slice_name
    if not slice_dir.exists():
        return []
    out: list[str] = []
    for p in sorted(slice_dir.glob("corpus_*.txt")):
        out.extend(line for line in p.read_text(encoding="utf-8").splitlines() if line)
    return out


# ngram_text needs ≥ 2 Chinese tokens to survive clean_ngram (line 69 of build_corpora_ngram.py)
NGRAM = "中国 经济 发展 政策 改革"
NGRAM_CLEAN = "中国 经济 发展 政策 改革"  # already pure-Chinese, clean_ngram preserves


class TestPresenceModeDefault:
    def test_no_weight_mode_key_preserves_existing_behavior(self, tmp_path):
        # Two identical-ngram entries in different years that both fall in 1940_1949.
        # Presence mode = set dedup → ONE corpus line, regardless of count.
        f = _make_ngram_file(tmp_path, "5-00000-of-00105", [
            _line(NGRAM, 1942, match_count=5),
            _line(NGRAM, 1948, match_count=200),
        ])
        cfg = _config(tmp_path)  # no weight_mode → defaults to "presence"
        time_slices = [(1940, 1949)]
        process_ngram_file(f, time_slices, cfg, logger)
        lines = _read_corpus(Path(cfg["paths"]["corpora_dir"]), "1940_1949")
        assert lines == [NGRAM_CLEAN]


class TestInvalidWeightMode:
    def test_raises_value_error(self, tmp_path):
        f = _make_ngram_file(tmp_path, "5-00000-of-00105", [
            _line(NGRAM, 1942, match_count=1),
        ])
        cfg = _config(tmp_path, corpus={"weight_mode": "bogus"})
        with pytest.raises(ValueError, match="weight_mode"):
            process_ngram_file(f, [(1940, 1949)], cfg, logger)
