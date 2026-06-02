"""Tests for Chinese + English Google Ngram raw-data walkers."""

from __future__ import annotations

import logging
from pathlib import Path

from scripts.data_prep.raw_volume import WALKERS
from scripts.data_prep.raw_volume.ngram_zh import walk as walk_zh
from scripts.data_prep.raw_volume.ngram_en import walk as walk_en

logger = logging.getLogger("test")


def _make_gz(path: Path, n_bytes: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x1f\x8b" + b"\x00" * (n_bytes - 2))


def test_zh_walker_registered_under_ngram_zh_data_source():
    # The zh walker is dispatched when language=='zh' and data_source=='ngram'.
    assert "ngram_zh" in WALKERS
    assert WALKERS["ngram_zh"] is walk_zh


def test_en_walker_registered_under_ngram_en_data_source():
    assert "ngram_en" in WALKERS
    assert WALKERS["ngram_en"] is walk_en


def test_zh_groups_by_inferred_year(tmp_path):
    # Filename pattern carries the year (e.g. 5gm-1945.gz).
    _make_gz(tmp_path / "5gm-1942.gz", n_bytes=300)
    _make_gz(tmp_path / "5gm-1948.gz", n_bytes=500)
    _make_gz(tmp_path / "5gm-1955.gz", n_bytes=200)
    config = {"time_slices": {"start_year": 1940, "end_year": 1959,
                              "window_size": 10, "step_size": 5}}
    result = walk_zh(tmp_path, ["1940_1949", "1945_1954", "1950_1959"], config, logger)
    assert result["1940_1949"].n_files == 2  # 1942 + 1948
    assert result["1945_1954"].n_files == 1  # 1948
    assert result["1950_1959"].n_files == 1  # 1955
    assert result["1940_1949"].n_bytes == 800


def test_en_groups_by_inferred_year(tmp_path):
    _make_gz(tmp_path / "googlebooks-eng-all-5gram-20120701-aa-1990.gz", n_bytes=400)
    _make_gz(tmp_path / "googlebooks-eng-all-5gram-20120701-bb-1995.gz", n_bytes=600)
    config = {"time_slices": {"start_year": 1990, "end_year": 1999,
                              "window_size": 10, "step_size": 10}}
    result = walk_en(tmp_path, ["1990_1999"], config, logger)
    assert result["1990_1999"].n_files == 2
    assert result["1990_1999"].n_bytes == 1000


def test_missing_raw_dir(tmp_path):
    config = {"time_slices": {"start_year": 1940, "end_year": 1949,
                              "window_size": 10, "step_size": 10}}
    result = walk_zh(tmp_path / "nope", ["1940_1949"], config, logger)
    assert result["1940_1949"].n_files == 0
