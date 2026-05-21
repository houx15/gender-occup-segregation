"""
Regression tests for scripts.visualize.plot_garg_weat_categories_trend.
Locks the per-category trend rendering on decade-format units and the
empty / all-NaN safeguards.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("MPLBACKEND", "Agg")


def _make_summary(
    units: Iterable[str],
    categories: Iterable[str],
    mean_rnd: float = 0.5,
) -> pd.DataFrame:
    rows = []
    for u in units:
        for i, c in enumerate(categories):
            rows.append({
                "unit_name": u,
                "category": c,
                "mean_rnd": mean_rnd - 0.1 * i,
                "ci_low": mean_rnd - 0.2 - 0.1 * i,
                "ci_high": mean_rnd + 0.2 - 0.1 * i,
                "sub_low": mean_rnd - 0.3 - 0.1 * i,
                "sub_high": mean_rnd + 0.3 - 0.1 * i,
                "sub_mean": mean_rnd - 0.1 * i,
                "n_occupations": 10,
                "n_consistent": 10,
            })
    return pd.DataFrame(rows)


def _pdfs(d: Path) -> List[str]:
    return sorted(p.name for p in d.glob("*.pdf"))


def test_writes_pdf_for_decade_units(tmp_path):
    from scripts.visualize import plot_garg_weat_categories_trend

    df = _make_summary(
        units=["1910s", "1950s", "1990s"],
        categories=["leadership", "family", "science"],
    )
    plot_garg_weat_categories_trend(
        df, tmp_path, logging.getLogger("test"), embedding_source="trained_coha",
    )
    names = _pdfs(tmp_path)
    assert any("fig2_garg_weat_categories__trained_coha" in n for n in names), (
        f"expected PDF with source suffix; got {names}"
    )


def test_apply_ideation_sign_reverses_only_marked_categories():
    from scripts.visualize import apply_ideation_sign

    df = pd.DataFrame({
        "category": ["leadership", "family", "science"],
        "mean_rnd": [0.4, 0.5, 0.3],
        "ci_low": [0.2, 0.3, 0.1],
        "ci_high": [0.6, 0.7, 0.5],
    })
    out = apply_ideation_sign(
        df, {"leadership": 1, "family": -1, "science": 1},
        ["mean_rnd", "ci_low", "ci_high"],
    )
    # leadership/science untouched
    assert out.loc[out["category"] == "leadership", "mean_rnd"].iloc[0] == 0.4
    assert out.loc[out["category"] == "science", "mean_rnd"].iloc[0] == 0.3
    # family negated (band bounds negated too — order swap is fine for fill)
    fam = out[out["category"] == "family"].iloc[0]
    assert fam["mean_rnd"] == -0.5
    assert fam["ci_low"] == -0.3
    assert fam["ci_high"] == -0.7


def test_apply_ideation_sign_none_is_identity():
    from scripts.visualize import apply_ideation_sign

    df = pd.DataFrame({"category": ["family"], "mean_rnd": [0.5]})
    out = apply_ideation_sign(df, None, ["mean_rnd"])
    assert out["mean_rnd"].iloc[0] == 0.5


def test_reversed_category_legend_and_pdf(tmp_path):
    from scripts.visualize import plot_garg_weat_categories_trend

    df = _make_summary(
        units=["1910s", "1950s", "1990s"],
        categories=["leadership", "family", "science"],
    )
    plot_garg_weat_categories_trend(
        df, tmp_path, logging.getLogger("test"), embedding_source="trained_coha",
        category_sign={"leadership": 1, "family": -1, "science": 1},
    )
    assert any(
        "fig2_garg_weat_categories__trained_coha" in n for n in _pdfs(tmp_path)
    )


def test_subsample_band_writes_tagged_pdf(tmp_path):
    from scripts.visualize import plot_garg_weat_categories_trend

    df = _make_summary(
        units=["1910s", "1950s", "1990s"],
        categories=["leadership", "family", "science"],
    )
    plot_garg_weat_categories_trend(
        df, tmp_path, logging.getLogger("test"), embedding_source="trained_coha",
        band_cols=("sub_low", "sub_high"), band_tag="subsample",
    )
    names = _pdfs(tmp_path)
    assert any(
        "fig2_garg_weat_categories__trained_coha__subsample" in n for n in names
    ), f"expected subsample-tagged PDF; got {names}"


def test_empty_dataframe_skips_write(tmp_path, caplog):
    from scripts.visualize import plot_garg_weat_categories_trend
    with caplog.at_level(logging.WARNING):
        plot_garg_weat_categories_trend(
            pd.DataFrame(), tmp_path, logging.getLogger("test"),
        )
    assert _pdfs(tmp_path) == []


def test_all_nan_skips_write_with_error(tmp_path, caplog):
    from scripts.visualize import plot_garg_weat_categories_trend

    df = _make_summary(units=["1990s"], categories=["leadership"])
    df["mean_rnd"] = np.nan
    df["ci_low"] = np.nan
    df["ci_high"] = np.nan

    with caplog.at_level(logging.ERROR):
        plot_garg_weat_categories_trend(
            df, tmp_path, logging.getLogger("test"),
        )
    assert _pdfs(tmp_path) == []
    assert any(
        "every mean_rnd is NaN" in r.message for r in caplog.records
    ), f"expected NaN refusal log; got {[r.message for r in caplog.records]}"


def test_unparseable_units_skip_with_warning(tmp_path, caplog):
    from scripts.visualize import plot_garg_weat_categories_trend
    df = _make_summary(
        units=["alpha", "beta", "gamma"],
        categories=["leadership"],
    )
    with caplog.at_level(logging.WARNING):
        plot_garg_weat_categories_trend(
            df, tmp_path, logging.getLogger("test"),
        )
    assert _pdfs(tmp_path) == []
