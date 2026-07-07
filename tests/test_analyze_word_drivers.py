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


def test_summary_contribution_sums_to_delta_of_mean():
    long_df = build_long_table(_rnd_long_fixture(), {"science": 1, "family": -1}, _Log())
    summ = build_summary_table(long_df, _Log())
    sci = summ[summ.category == "science"]
    # consistent set is {a, b}; 'c' dropped upstream
    assert set(sci.occupation) == {"a", "b"}
    # per-word: a delta = 0.0-(-0.4)=0.4, b delta = 0.5-0.1=0.4; N=2 -> contrib 0.2 each
    a = sci[sci.occupation == "a"].iloc[0]
    assert a["delta"] == pytest.approx(0.4)
    assert a["contribution"] == pytest.approx(0.2)
    # Σ contribution == Δ cat_mean_signed = 0.25 - (-0.15) = 0.4
    assert sci["contribution"].sum() == pytest.approx(0.4)


def test_summary_slope_is_ols():
    long_df = build_long_table(_rnd_long_fixture(), {"science": 1, "family": -1}, _Log())
    summ = build_summary_table(long_df, _Log())
    # a: years [1990,2000,2010], signed [-0.4,-0.2,0.0] -> slope 0.02 / yr
    a_slope = summ[(summ.category == "science") & (summ.occupation == "a")]["slope"].iloc[0]
    assert a_slope == pytest.approx(0.02)


def test_main_writes_four_files(tmp_path, monkeypatch):
    import scripts.analyze_word_drivers as awd

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    _rnd_long_fixture().to_parquet(results_dir / "garg_weat_rnd_long.parquet", index=False)

    cfg = {
        "paths": {"results_dir": str(results_dir), "log_dir": str(tmp_path / "logs")},
        "analysis": {"ideation_sign": {"science": 1, "family": -1}},
    }
    monkeypatch.setattr(awd, "load_config", lambda _p: cfg)
    monkeypatch.setattr(awd, "setup_logging", lambda *_a, **_k: _Log())

    awd.main(config="ignored.yml")

    for name in ("word_drivers_long", "word_drivers_summary"):
        assert (results_dir / f"{name}.parquet").exists()
        assert (results_dir / f"{name}.csv").exists()
    long_df = pd.read_parquet(results_dir / "word_drivers_long.parquet")
    assert {"signed_rnd", "cat_mean_signed", "deviation"} <= set(long_df.columns)


def test_main_errors_without_input(tmp_path, monkeypatch):
    import scripts.analyze_word_drivers as awd

    cfg = {"paths": {"results_dir": str(tmp_path), "log_dir": str(tmp_path)},
           "analysis": {}}
    monkeypatch.setattr(awd, "load_config", lambda _p: cfg)
    monkeypatch.setattr(awd, "setup_logging", lambda *_a, **_k: _Log())
    with pytest.raises(FileNotFoundError):
        awd.main(config="ignored.yml")


def test_empty_when_no_longitudinal_units():
    # all-provincial unit names -> everything dropped -> empty tables, no crash
    prov = pd.DataFrame([
        {"unit_name": "北京", "category": "science", "occupation": "a",
         "rnd": 0.1, "in_vocab": True},
        {"unit_name": "上海", "category": "science", "occupation": "a",
         "rnd": 0.2, "in_vocab": True},
    ])
    long_df = build_long_table(prov, {"science": 1}, _Log())
    assert long_df.empty
    assert list(long_df.columns) == [
        "category", "year", "unit_name", "occupation",
        "rnd", "signed_rnd", "cat_mean_signed", "deviation",
    ]
    summ = build_summary_table(long_df, _Log())
    assert summ.empty
    assert "contribution" in summ.columns


def test_main_raises_valueerror_on_missing_columns(tmp_path, monkeypatch):
    import scripts.analyze_word_drivers as awd
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    # missing 'in_vocab' column
    pd.DataFrame([{"unit_name": "1990s", "category": "science",
                   "occupation": "a", "rnd": 0.1}]).to_parquet(
        results_dir / "garg_weat_rnd_long.parquet", index=False)
    cfg = {"paths": {"results_dir": str(results_dir), "log_dir": str(tmp_path)},
           "analysis": {}}
    monkeypatch.setattr(awd, "load_config", lambda _p: cfg)
    monkeypatch.setattr(awd, "setup_logging", lambda *_a, **_k: _Log())
    with pytest.raises(ValueError):
        awd.main(config="ignored.yml")
