#!/usr/bin/env python3
"""Decompose each Garg-WEAT dimension's plotted ideation line into per-word RND.

Reads the per-word RND long table already written by analyze_category_bias
(garg_weat_rnd_long.parquet) and produces two per-corpus driver tables,
restricted to the per-category GLOBAL CONSISTENT SET (words in vocab in all
slices) so they reproduce the published mean_rnd line exactly:

  word_drivers_long.{parquet,csv}     one row per (category, year, word):
      rnd, signed_rnd, cat_mean_signed, deviation
  word_drivers_summary.{parquet,csv}  one row per (category, word):
      first/last year, signed_first/last, delta, contribution, slope

No model loading — pure pandas over an existing parquet.

Usage:
  python -m scripts.analyze_word_drivers --config=config/profiles/garg_weat_renminribao.yml
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import fire

from scripts.common.config_loader import load_config
from scripts.common.logging_utils import setup_logging


def _slice_start_year(unit_name) -> Optional[int]:
    """Start year from a longitudinal unit label. Mirrors
    scripts.visualize._decade_start_year (kept local so this data script needs
    no matplotlib import): '1990s' -> 1990, '1940_1949' -> 1940. Province and
    province-year units ('北京', '北京_2020') don't parse and return None."""
    s = str(unit_name)
    if len(s) == 5 and s.endswith("s") and s[:4].isdigit():
        return int(s[:4])
    try:
        return int(s.split("_")[0])
    except (ValueError, IndexError):
        return None


def _consistent_words_per_category(df: pd.DataFrame) -> Dict[str, set]:
    """Words in vocab in EVERY slice of their category (df already in_vocab-only,
    year-filtered). Mirrors category_summary.compute_consistent_set."""
    out: Dict[str, set] = {}
    for cat, g in df.groupby("category"):
        n_slices = g["year"].nunique()
        counts = g.groupby("occupation")["year"].nunique()
        out[cat] = set(counts[counts == n_slices].index)
    return out


def build_long_table(
    rnd_long: pd.DataFrame,
    ideation_sign: Dict[str, int],
    logger,
) -> pd.DataFrame:
    """One row per (category, year, word) over the consistent set: rnd,
    signed_rnd, cat_mean_signed, deviation.

    cat_mean_signed is the plotted line's value for that (category, slice): the
    mean signed_rnd over the per-category global consistent set (words in vocab
    in ALL slices), matching how analyze_category_bias builds mean_rnd.
    """
    df = rnd_long[rnd_long["in_vocab"]].copy()
    df["year"] = df["unit_name"].map(_slice_start_year)
    dropped = int(df["year"].isna().sum())
    if dropped:
        logger.info(
            f"  word_drivers: dropping {dropped} rows with non-longitudinal "
            f"unit_name (provincial units are out of scope)"
        )
    df = df[df["year"].notna()].copy()
    df["year"] = df["year"].astype(int)
    df["signed_rnd"] = df["rnd"] * df["category"].map(
        lambda c: ideation_sign.get(c, 1)
    )

    keep = _consistent_words_per_category(df)
    mask = df.apply(
        lambda r: r["occupation"] in keep.get(r["category"], set()), axis=1
    )
    df = df[mask].copy()

    df["cat_mean_signed"] = df.groupby(["category", "year"])["signed_rnd"].transform(
        "mean"
    )
    df["deviation"] = df["signed_rnd"] - df["cat_mean_signed"]
    return (
        df[[
            "category", "year", "unit_name", "occupation",
            "rnd", "signed_rnd", "cat_mean_signed", "deviation",
        ]]
        .sort_values(["category", "year", "occupation"])
        .reset_index(drop=True)
    )


def build_summary_table(long_df, logger):  # implemented in Task 2
    raise NotImplementedError
