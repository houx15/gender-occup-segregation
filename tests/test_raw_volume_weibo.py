"""Tests for the Weibo raw-data walker (parquet shards)."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pytest

from scripts.data_prep.raw_volume import WALKERS
from scripts.data_prep.raw_volume.weibo import walk

logger = logging.getLogger("test")

pa = pytest.importorskip("pyarrow")


def _write_parquet(path: Path, n_rows: int, province_code: str = "11") -> None:
    df = pd.DataFrame({
        "text": ["t"] * n_rows,
        "user_province": [province_code] * n_rows,
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def test_walker_registered():
    assert "weibo" in WALKERS
    assert WALKERS["weibo"] is walk


def test_counts_files_and_bytes_per_province(tmp_path):
    # Province code 11 = 北京, 31 = 上海 (per PROVINCE_CODE_TO_NAME).
    _write_parquet(tmp_path / "2020" / "a.parquet", n_rows=10, province_code="11")
    _write_parquet(tmp_path / "2020" / "b.parquet", n_rows=20, province_code="11")
    _write_parquet(tmp_path / "2020" / "c.parquet", n_rows=5,  province_code="31")
    result = walk(tmp_path, ["北京", "上海", "天津"], {}, logger)
    # 北京: 2 files; 上海: 1 file; 天津: none.
    assert result["北京"].n_files == 2
    assert result["上海"].n_files == 1
    assert result["天津"].n_files == 0
    # n_source_docs from parquet row counts.
    assert result["北京"].n_source_docs == 30
    assert result["上海"].n_source_docs == 5
    assert result["天津"].n_source_docs == 0
    assert result["北京"].n_bytes > 0


def test_missing_raw_dir_yields_zero_entries(tmp_path):
    result = walk(tmp_path / "nope", ["北京"], {}, logger)
    assert result["北京"].n_files == 0
    assert result["北京"].n_source_docs == 0
