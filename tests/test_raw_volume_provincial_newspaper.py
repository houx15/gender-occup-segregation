"""Tests for the provincial-newspaper raw-data walker."""

from __future__ import annotations

import logging
from pathlib import Path

from scripts.data_prep.raw_volume import WALKERS
from scripts.data_prep.raw_volume.provincial_newspaper import walk

logger = logging.getLogger("test")


def _make_tree(root: Path, layout: dict[str, dict[str, int]]) -> None:
    """layout = {folder_name: {year: n_files_in_year}}."""
    for folder, years in layout.items():
        for year, n in years.items():
            d = root / folder / str(year)
            d.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                (d / f"{year}-{i:03d}.txt").write_text("y" * 50, encoding="utf-8")


def test_walker_registered():
    assert "newspaper" in WALKERS
    assert WALKERS["newspaper"] is walk


def test_groups_by_province_year(tmp_path):
    _make_tree(tmp_path, {
        "北京日报":  {"2020": 3, "2021": 5},
        "天津日报":  {"2020": 2},
        "广东日报":  {"2022": 4},
    })
    units = ["北京_2020", "北京_2021", "天津_2020", "广东_2022", "上海_2020"]
    result = walk(tmp_path, units, {}, logger)
    assert result["北京_2020"].n_files == 3
    assert result["北京_2021"].n_files == 5
    assert result["天津_2020"].n_files == 2
    assert result["广东_2022"].n_files == 4
    assert result["上海_2020"].n_files == 0  # nothing under that folder
    assert result["北京_2020"].n_bytes == 3 * 50


def test_unknown_province_folder_ignored(tmp_path):
    _make_tree(tmp_path, {"未知日报": {"2020": 5}})
    result = walk(tmp_path, ["北京_2020"], {}, logger)
    assert result["北京_2020"].n_files == 0


def test_missing_raw_dir_yields_zero_entries(tmp_path):
    result = walk(tmp_path / "nope", ["北京_2020"], {}, logger)
    assert result["北京_2020"].n_files == 0
