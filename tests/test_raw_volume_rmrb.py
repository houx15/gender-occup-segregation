"""Tests for the RMRB raw-data walker."""

from __future__ import annotations

import logging
from pathlib import Path

from scripts.data_prep.raw_volume import WALKERS
from scripts.data_prep.raw_volume.rmrb import walk

logger = logging.getLogger("test")


def _make_rmrb_tree(root: Path, files_by_year: dict[int, int]) -> None:
    """Create rmrb_YYYY_MM.txt files under {decade}/{year}/报刊/人民日报/."""
    for year, n_months in files_by_year.items():
        decade = f"{(year // 10) * 10}s"
        ymd = root / decade / str(year) / "报刊" / "人民日报"
        ymd.mkdir(parents=True, exist_ok=True)
        for m in range(1, n_months + 1):
            (ymd / f"rmrb_{year}_{m:02d}.txt").write_text("x" * 100, encoding="utf-8")


def test_walker_registered():
    assert "renminribao" in WALKERS
    assert WALKERS["renminribao"] is walk


def test_walks_overlapping_slices(tmp_path):
    _make_rmrb_tree(tmp_path, {1942: 12, 1948: 6, 1955: 3})
    # Default RMRB profile: 10-year window, 5-year step → 1940_1949, 1945_1954, …
    config = {
        "time_slices": {"start_year": 1940, "end_year": 1959,
                        "window_size": 10, "step_size": 5},
    }
    units = ["1940_1949", "1945_1954", "1950_1959"]
    result = walk(tmp_path, units, config, logger)
    # 1940_1949 contains years 1942 + 1948 = 12 + 6 = 18 files
    assert result["1940_1949"].n_files == 18
    # 1945_1954 contains years 1948 + (no 1950–1954 here) = 6
    assert result["1945_1954"].n_files == 6
    # 1950_1959 contains year 1955 = 3
    assert result["1950_1959"].n_files == 3
    # Bytes are 100 per file.
    assert result["1940_1949"].n_bytes == 1800


def test_returns_zero_entry_for_unit_with_no_files(tmp_path):
    _make_rmrb_tree(tmp_path, {1942: 1})
    config = {"time_slices": {"start_year": 1940, "end_year": 1959,
                              "window_size": 10, "step_size": 5}}
    result = walk(tmp_path, ["1970_1979"], config, logger)
    assert result["1970_1979"].n_files == 0
    assert result["1970_1979"].n_bytes == 0


def test_missing_raw_dir_yields_zero_entries(tmp_path):
    config = {"time_slices": {"start_year": 1940, "end_year": 1959,
                              "window_size": 10, "step_size": 5}}
    result = walk(tmp_path / "nope", ["1940_1949"], config, logger)
    assert result["1940_1949"].n_files == 0
