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


from scripts.common.dataset_stats import scan_corpus_unit


class TestScanCorpusUnit:
    def test_counts_docs_tokens_vocab(self, tmp_path):
        unit_dir = tmp_path / "u"
        _make_corpus_files(unit_dir, ["a b c\n", "a b\n"])
        stats = scan_corpus_unit(unit_dir, logger)
        assert stats.n_docs == 2
        assert stats.n_tokens == 5   # 3 + 2
        assert stats.n_vocab_raw == 3  # {a, b, c}
        assert stats.n_corpus_files == 2
        assert stats.from_cache is False

    def test_multi_file_aggregation(self, tmp_path):
        unit_dir = tmp_path / "u"
        _make_corpus_files(unit_dir, ["a b\n", "c d\n", "a c\n"])
        stats = scan_corpus_unit(unit_dir, logger)
        assert stats.n_docs == 3
        assert stats.n_tokens == 6
        assert stats.n_vocab_raw == 4  # {a, b, c, d}
        assert stats.n_corpus_files == 3

    def test_empty_unit_dir(self, tmp_path):
        unit_dir = tmp_path / "u"
        unit_dir.mkdir()
        stats = scan_corpus_unit(unit_dir, logger)
        assert stats.n_docs == 0
        assert stats.n_tokens == 0
        assert stats.n_vocab_raw == 0
        assert stats.n_corpus_files == 0

    def test_empty_file(self, tmp_path):
        unit_dir = tmp_path / "u"
        _make_corpus_files(unit_dir, [""])
        stats = scan_corpus_unit(unit_dir, logger)
        assert stats.n_docs == 0
        assert stats.n_tokens == 0
        assert stats.n_vocab_raw == 0

    def test_cache_written_after_scan(self, tmp_path):
        unit_dir = tmp_path / "u"
        _make_corpus_files(unit_dir, ["a b\n"])
        scan_corpus_unit(unit_dir, logger)
        assert (unit_dir / ".dataset_stats.json").exists()

    def test_cache_hit_skips_scan(self, tmp_path):
        unit_dir = tmp_path / "u"
        _make_corpus_files(unit_dir, ["a b c\n"])
        # First scan: writes cache.
        scan_corpus_unit(unit_dir, logger)
        # Mutate cache to a sentinel value so we can detect a re-scan.
        cache_path = unit_dir / ".dataset_stats.json"
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        payload["n_tokens"] = 99999
        cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        # Second scan: must use cache, not rescan.
        stats = scan_corpus_unit(unit_dir, logger)
        assert stats.n_tokens == 99999
        assert stats.from_cache is True

    def test_force_recomputes_even_with_cache(self, tmp_path):
        unit_dir = tmp_path / "u"
        _make_corpus_files(unit_dir, ["a b c\n"])
        scan_corpus_unit(unit_dir, logger)
        cache_path = unit_dir / ".dataset_stats.json"
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        payload["n_tokens"] = 99999
        cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        stats = scan_corpus_unit(unit_dir, logger, force=True)
        assert stats.n_tokens == 3  # rescanned, sentinel overwritten

    def test_returns_vocab_set_when_requested(self, tmp_path):
        unit_dir = tmp_path / "u"
        _make_corpus_files(unit_dir, ["a b c\n", "a b\n"])
        stats, vocab = scan_corpus_unit(unit_dir, logger, return_vocab=True)
        assert vocab == {"a", "b", "c"}


import sys
import types


def _install_fake_gensim():
    """Match the pattern in tests/test_analyze_cohens_d_singlelist.py."""
    if "gensim" in sys.modules and not getattr(sys.modules["gensim"], "_fake", False):
        try:
            from gensim.models import KeyedVectors  # noqa: F401
            return
        except Exception:
            pass
    fake_gensim = types.ModuleType("gensim")
    fake_gensim._fake = True  # type: ignore[attr-defined]
    fake_models = types.ModuleType("gensim.models")

    class _FakeKV:
        @staticmethod
        def load(path):
            class _Stub:
                index_to_key = ["a", "b", "c"]
            return _Stub()

    fake_models.KeyedVectors = _FakeKV  # type: ignore[attr-defined]
    fake_gensim.models = fake_models  # type: ignore[attr-defined]
    sys.modules["gensim"] = fake_gensim
    sys.modules["gensim.models"] = fake_models


_install_fake_gensim()

from scripts.common.dataset_stats import discover_units, model_vocab_size


class TestDiscoverUnits:
    def test_returns_sorted_subdirs(self, tmp_path):
        corpora = tmp_path / "corpora"
        (corpora / "1950_1959").mkdir(parents=True)
        (corpora / "1940_1949").mkdir(parents=True)
        (corpora / "_skip_file.txt").parent.mkdir(exist_ok=True)
        (corpora / "_skip_file.txt").write_text("x")
        config = {"paths": {"corpora_dir": str(corpora)}}
        units = discover_units(config)
        assert units == ["1940_1949", "1950_1959"]

    def test_missing_corpora_dir_returns_empty(self, tmp_path):
        config = {"paths": {"corpora_dir": str(tmp_path / "nope")}}
        assert discover_units(config) == []


class TestModelVocabSize:
    def test_returns_vocab_count(self, tmp_path):
        model_path = tmp_path / "m.model"
        model_path.write_text("stub")  # contents irrelevant — fake gensim ignores
        assert model_vocab_size(model_path, logger) == 3

    def test_missing_file_returns_none(self, tmp_path):
        assert model_vocab_size(tmp_path / "nope.model", logger) is None
