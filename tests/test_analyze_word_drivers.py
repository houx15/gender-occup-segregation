"""Tests for scripts.analyze_word_drivers — per-word ideation decomposition."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("MPLBACKEND", "Agg")

from scripts.analyze_word_drivers import (
    _slice_start_year,
    build_long_table,
    build_summary_table,
)


def _rnd_long_fixture() -> pd.DataFrame:
    """science sign +1, family sign -1. 'c' churns (only middle slice) -> it is
    NOT in the science consistent set and must be dropped from both tables."""
    data = {
        ("science", "a"): {1990: -0.4, 2000: -0.2, 2010: 0.0},
        ("science", "b"): {1990: 0.1, 2000: 0.2, 2010: 0.5},
        ("science", "c"): {2000: 0.3},   # churn
        ("family", "d"): {1990: 0.2, 2000: 0.1, 2010: -0.1},
    }
    rows = []
    for (cat, occ), yv in data.items():
        for yr, v in yv.items():
            rows.append({
                "unit_name": f"{yr}s", "category": cat,
                "occupation": occ, "rnd": v, "in_vocab": True,
            })
    return pd.DataFrame(rows)


class _Log:
    def info(self, *a, **k):
        pass

    def warning(self, *a, **k):
        pass


def test_slice_start_year_formats():
    assert _slice_start_year("1990s") == 1990
    assert _slice_start_year("1940_1949") == 1940
    assert _slice_start_year("北京") is None
    # province-year: dropped (mirrors visualize._decade_start_year, which
    # returns None because int("北京") raises)
    assert _slice_start_year("北京_2020") is None


def test_long_table_consistent_set_mean_deviation_sign():
    df = build_long_table(_rnd_long_fixture(), {"science": 1, "family": -1}, _Log())
    # 'c' churns -> excluded from the consistent set -> absent from the table
    assert "c" not in set(df[df.category == "science"]["occupation"])
    # science 1990 consistent set {a,b}: signed mean = (-0.4 + 0.1)/2 = -0.15
    sci90 = df[(df.category == "science") & (df.year == 1990)]
    assert sci90["cat_mean_signed"].round(6).eq(-0.15).all()
    a90 = sci90[sci90.occupation == "a"].iloc[0]
    assert a90["signed_rnd"] == pytest.approx(-0.4)   # sign +1
    assert a90["deviation"] == pytest.approx(-0.4 - (-0.15))
    # science 2000 mean over {a,b} = 0.0 (NOT 0.1 — 'c' excluded)
    sci00 = df[(df.category == "science") & (df.year == 2000)]
    assert sci00["cat_mean_signed"].round(6).eq(0.0).all()
    # family sign flip: d rnd 0.2 -> signed -0.2
    d90 = df[(df.category == "family") & (df.year == 1990)].iloc[0]
    assert d90["signed_rnd"] == pytest.approx(-0.2)
