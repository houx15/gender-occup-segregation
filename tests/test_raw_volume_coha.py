"""Tests for the COHA raw-data walker."""

from __future__ import annotations

import logging
from pathlib import Path

from scripts.data_prep.raw_volume import WALKERS
from scripts.data_prep.raw_volume.coha import walk

logger = logging.getLogger("test")


def _make_coha_tree(root: Path, files_by_decade: dict[str, int]) -> None:
    """Mimic COHA's text_NNNNs/ layout used by build_corpora_coha."""
    for decade, n in files_by_decade.items():
        d = root / f"text_{decade}"
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            (d / f"doc_{i:04d}.txt").write_text("z" * 80, encoding="utf-8")


def test_walker_registered():
    assert "coha" in WALKERS
    assert WALKERS["coha"] is walk


def test_groups_by_decade(tmp_path):
    _make_coha_tree(tmp_path, {"1940": 3, "1950": 5, "1960": 2})
    result = walk(tmp_path, ["1940s", "1950s", "1960s", "1970s"], {}, logger)
    assert result["1940s"].n_files == 3
    assert result["1950s"].n_files == 5
    assert result["1960s"].n_files == 2
    assert result["1970s"].n_files == 0
    assert result["1940s"].n_bytes == 3 * 80


def test_missing_raw_dir(tmp_path):
    result = walk(tmp_path / "nope", ["1940s"], {}, logger)
    assert result["1940s"].n_files == 0
