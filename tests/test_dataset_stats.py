"""Tests for scripts.common.dataset_stats."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from scripts.common.dataset_stats import (
    CorpusStats, RawVolumeEntry, write_cache, read_cache, cache_is_fresh,
)

logger = logging.getLogger("test")


def _make_corpus_files(unit_dir: Path, contents: list[str]) -> list[Path]:
    """Write corpus_NNNNNN files; return their paths."""
    unit_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, body in enumerate(contents):
        p = unit_dir / f"corpus_{i:06d}"
        p.write_text(body, encoding="utf-8")
        paths.append(p)
    return paths


class TestCacheIO:
    def test_write_then_read_roundtrip(self, tmp_path):
        unit_dir = tmp_path / "1940_1949"
        files = _make_corpus_files(unit_dir, ["a b c\n", "a b\n"])
        stats = CorpusStats(
            unit_name="1940_1949", n_docs=2, n_tokens=5, n_vocab_raw=3,
            n_corpus_files=1, scanned_at="2026-06-02T00:00:00", from_cache=False,
        )
        write_cache(unit_dir, stats, files)
        loaded = read_cache(unit_dir)
        assert loaded is not None
        assert loaded.unit_name == "1940_1949"
        assert loaded.n_docs == 2
        assert loaded.n_tokens == 5
        assert loaded.n_vocab_raw == 3
        assert loaded.from_cache is True  # read_cache sets this

    def test_cache_missing_returns_none(self, tmp_path):
        unit_dir = tmp_path / "1940_1949"
        unit_dir.mkdir()
        assert read_cache(unit_dir) is None

    def test_cache_corrupt_returns_none(self, tmp_path):
        unit_dir = tmp_path / "1940_1949"
        unit_dir.mkdir()
        (unit_dir / ".dataset_stats.json").write_text("{not json", encoding="utf-8")
        assert read_cache(unit_dir) is None  # logged + ignored, not raised

    def test_cache_schema_version_mismatch_returns_none(self, tmp_path):
        unit_dir = tmp_path / "1940_1949"
        unit_dir.mkdir()
        (unit_dir / ".dataset_stats.json").write_text(
            json.dumps({"schema_version": 99, "n_docs": 1, "n_tokens": 1,
                        "n_vocab_raw": 1, "scanned_at": "x", "corpus_files": []}),
            encoding="utf-8",
        )
        assert read_cache(unit_dir) is None


class TestCacheFreshness:
    def test_fresh_when_files_unchanged(self, tmp_path):
        unit_dir = tmp_path / "u"
        files = _make_corpus_files(unit_dir, ["x\n"])
        stats = CorpusStats(unit_name="u", n_docs=1, n_tokens=1, n_vocab_raw=1,
                            n_corpus_files=1, scanned_at="t", from_cache=False)
        write_cache(unit_dir, stats, files)
        # Same files, same mtimes → fresh
        assert cache_is_fresh(unit_dir, files) is True

    def test_stale_when_file_mtime_changes(self, tmp_path):
        unit_dir = tmp_path / "u"
        files = _make_corpus_files(unit_dir, ["x\n"])
        stats = CorpusStats(unit_name="u", n_docs=1, n_tokens=1, n_vocab_raw=1,
                            n_corpus_files=1, scanned_at="t", from_cache=False)
        write_cache(unit_dir, stats, files)
        # Bump mtime → stale
        import os, time
        new_mtime = files[0].stat().st_mtime + 100
        os.utime(files[0], (new_mtime, new_mtime))
        assert cache_is_fresh(unit_dir, files) is False

    def test_stale_when_file_added(self, tmp_path):
        unit_dir = tmp_path / "u"
        files = _make_corpus_files(unit_dir, ["x\n"])
        stats = CorpusStats(unit_name="u", n_docs=1, n_tokens=1, n_vocab_raw=1,
                            n_corpus_files=1, scanned_at="t", from_cache=False)
        write_cache(unit_dir, stats, files)
        new_file = unit_dir / "corpus_000001"
        new_file.write_text("y\n", encoding="utf-8")
        assert cache_is_fresh(unit_dir, files + [new_file]) is False

    def test_stale_when_no_cache(self, tmp_path):
        unit_dir = tmp_path / "u"
        files = _make_corpus_files(unit_dir, ["x\n"])
        assert cache_is_fresh(unit_dir, files) is False
