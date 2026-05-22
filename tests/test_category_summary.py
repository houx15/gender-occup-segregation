"""Tests for scripts.common.category_summary (metric-agnostic summaries)."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from scripts.common.category_summary import (
    build_summary, compute_consistent_set, subsample_bands_from_lookup,
)

logger = logging.getLogger("test")


def test_consistent_set_intersects_across_units():
    # 'a','b' in both units; 'c' only in 1990s → consistent set drops 'c'.
    rows = [
        {"unit_name": "1990s", "category": "lead", "occupation": "a", "in_vocab": True},
        {"unit_name": "1990s", "category": "lead", "occupation": "b", "in_vocab": True},
        {"unit_name": "1990s", "category": "lead", "occupation": "c", "in_vocab": True},
        {"unit_name": "2000s", "category": "lead", "occupation": "a", "in_vocab": True},
        {"unit_name": "2000s", "category": "lead", "occupation": "b", "in_vocab": True},
        {"unit_name": "2000s", "category": "lead", "occupation": "c", "in_vocab": False},
    ]
    consistent = compute_consistent_set(
        pd.DataFrame(rows), {"lead": ["a", "b", "c"]}, ["1990s", "2000s"], logger,
    )
    assert consistent["lead"] == ["a", "b"]


def test_consistent_set_empty_when_long_df_empty():
    empty = pd.DataFrame(
        columns=["unit_name", "category", "occupation", "in_vocab"]
    )
    consistent = compute_consistent_set(empty, {"lead": ["a"]}, [], logger)
    assert consistent == {"lead": []}


def test_subsample_band_empty_consistent_set_is_nan():
    bands = subsample_bands_from_lookup(
        {}, ["u"], {"lead": []},
        fraction=0.8, n_rounds=10, ci=0.95, seed=1,
    )
    lo, hi, mean = bands[("u", "lead")]
    assert np.isnan(lo) and np.isnan(hi) and np.isnan(mean)


def _long(value_col="value"):
    """Two units × one category 'lead' × three occupations, all in vocab.
    1990s values: [-1, -2, 3] (2/3 male-leaning); 2000s: [1, 2, -3] (1/3)."""
    rows = []
    data = {
        "1990s": {"a": -1.0, "b": -2.0, "c": 3.0},
        "2000s": {"a": 1.0, "b": 2.0, "c": -3.0},
    }
    for unit, occs in data.items():
        for occ, val in occs.items():
            rows.append({
                "unit_name": unit, "category": "lead", "occupation": occ,
                value_col: val, "in_vocab": True,
            })
    return pd.DataFrame(rows)


def test_subsample_band_default_mean_statistic():
    occs = ["a", "b", "c", "d", "e"]
    lookup = {("u", "lead", w): float(i) for i, w in enumerate(occs)}  # 0..4
    bands = subsample_bands_from_lookup(
        lookup, ["u"], {"lead": occs},
        fraction=0.8, n_rounds=200, ci=0.95, seed=42,
    )
    lo, hi, mean = bands[("u", "lead")]
    assert lo < hi
    assert lo <= mean <= hi


def test_subsample_band_proportion_statistic():
    from scripts.common.metrics import proportion_below
    occs = ["a", "b", "c", "d", "e"]
    # three negative, two positive → proportion below 0 around 0.6
    lookup = {
        ("u", "lead", "a"): -1.0, ("u", "lead", "b"): -2.0,
        ("u", "lead", "c"): -3.0, ("u", "lead", "d"): 1.0,
        ("u", "lead", "e"): 2.0,
    }
    bands = subsample_bands_from_lookup(
        lookup, ["u"], {"lead": occs},
        fraction=0.8, n_rounds=200, ci=0.95, seed=7,
        statistic=lambda x: proportion_below(x, 0.0),
    )
    lo, hi, mean = bands[("u", "lead")]
    assert 0.0 <= lo <= mean <= hi <= 1.0


def test_build_summary_has_mean_and_proportion_columns():
    long_df = _long()
    consistent = {"lead": ["a", "b", "c"]}
    summary = build_summary(
        long_df, ["1990s", "2000s"], consistent, logger,
        value_col="value", boot_n_iter=200, boot_ci=0.68,
        sub_fraction=0.8, sub_rounds=50, sub_ci=0.95, seed=42,
    )
    assert {
        "mean_value", "mean_ci_low", "mean_ci_high",
        "mean_sub_low", "mean_sub_high", "mean_sub_mean",
        "prop_male", "prop_ci_low", "prop_ci_high",
        "prop_sub_low", "prop_sub_high", "prop_sub_mean",
        "n_occupations", "n_consistent",
    } <= set(summary.columns)
    # 1990s: 2 of 3 male-leaning; 2000s: 1 of 3
    p90 = summary[(summary.unit_name == "1990s")]["prop_male"].iloc[0]
    p00 = summary[(summary.unit_name == "2000s")]["prop_male"].iloc[0]
    assert p90 == 2 / 3
    assert p00 == 1 / 3


def test_build_summary_legacy_rnd_aliases():
    long_df = _long()
    consistent = {"lead": ["a", "b", "c"]}
    summary = build_summary(
        long_df, ["1990s", "2000s"], consistent, logger,
        value_col="value", legacy_rnd_aliases=True,
        boot_n_iter=100, boot_ci=0.68, sub_fraction=0.8,
        sub_rounds=50, sub_ci=0.95, seed=42,
    )
    assert {"mean_rnd", "ci_low", "ci_high", "sub_low", "sub_high", "sub_mean"} <= set(summary.columns)
    # Aliases equal their generic counterparts row-by-row.
    assert (summary["mean_rnd"] == summary["mean_value"]).all()
    assert (summary["ci_low"] == summary["mean_ci_low"]).all()
    assert (summary["sub_mean"] == summary["mean_sub_mean"]).all()
