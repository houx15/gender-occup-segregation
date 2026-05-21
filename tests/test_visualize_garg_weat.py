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


def test_window_format_units_render_trend(tmp_path):
    """RMRB / china-ngram slices like '1940_1949' must parse as start-year
    so the Chinese longitudinal RND trend renders (not just COHA 'YYYYs')."""
    from scripts.visualize import plot_garg_weat_categories_trend

    df = _make_summary(
        units=["1940_1949", "1950_1959", "1960_1969"],
        categories=["leadership", "family", "science"],
    )
    plot_garg_weat_categories_trend(
        df, tmp_path, logging.getLogger("test"), embedding_source="renminribao",
    )
    assert any(
        "fig2_garg_weat_categories__renminribao" in n for n in _pdfs(tmp_path)
    ), f"expected window-format trend PDF; got {_pdfs(tmp_path)}"


def test_province_units_skip_trend(tmp_path, caplog):
    """Province units (e.g. '北京') must NOT render as a trend — they belong to
    the provincial RND plots. The trend plot should skip with a warning."""
    from scripts.visualize import plot_garg_weat_categories_trend

    df = _make_summary(
        units=["北京", "上海", "广东"],
        categories=["leadership"],
    )
    with caplog.at_level(logging.WARNING):
        plot_garg_weat_categories_trend(
            df, tmp_path, logging.getLogger("test"),
        )
    assert _pdfs(tmp_path) == []


# --- Provincial RND plots + unit-kind detection -----------------------------

def test_unit_kind_classifier():
    from scripts.visualize import _garg_weat_unit_kind
    assert _garg_weat_unit_kind(["1990s", "2000s"]) == "longitudinal"
    assert _garg_weat_unit_kind(["1940_1949", "1990_1999"]) == "longitudinal"
    assert _garg_weat_unit_kind(["北京_2020", "广东_2023"]) == "province_year"
    assert _garg_weat_unit_kind(["北京", "上海", "广东"]) == "province"


def test_provincial_rankings_and_heatmap(tmp_path):
    from scripts.visualize import (
        plot_garg_weat_provincial_rankings, plot_garg_weat_provincial_heatmap,
    )
    df = _make_summary(
        units=["北京", "上海", "广东", "四川"],
        categories=["leadership", "family", "science"],
    )
    sign = {"leadership": 1, "family": -1, "science": 1}
    plot_garg_weat_provincial_rankings(
        df, tmp_path, logging.getLogger("test"),
        category_sign=sign, data_source="weibo",
    )
    plot_garg_weat_provincial_heatmap(
        df, tmp_path, logging.getLogger("test"),
        category_sign=sign, data_source="weibo",
    )
    names = _pdfs(tmp_path)
    assert any("garg_weat_provincial_rankings" in n for n in names), names
    assert any("garg_weat_provincial_heatmap" in n for n in names), names


def test_provincial_frame_averages_over_years_and_orients():
    """province-year units collapse to province-level means, and family is
    sign-flipped onto the ideation axis."""
    from scripts.visualize import _garg_weat_provincial_frame

    rows = [
        {"unit_name": "北京_2020", "category": "family", "mean_rnd": 0.4},
        {"unit_name": "北京_2021", "category": "family", "mean_rnd": 0.6},
        {"unit_name": "北京_2020", "category": "leadership", "mean_rnd": 0.2},
    ]
    df = pd.DataFrame(rows)
    out, reversed_cats = _garg_weat_provincial_frame(
        df, {"leadership": 1, "family": -1, "science": 1}
    )
    assert reversed_cats == ["family"]
    fam = out[(out["province"] == "北京") & (out["category"] == "family")]
    # mean(0.4, 0.6) = 0.5, flipped -> -0.5
    assert abs(fam["value"].iloc[0] - (-0.5)) < 1e-9
    lead = out[(out["province"] == "北京") & (out["category"] == "leadership")]
    assert abs(lead["value"].iloc[0] - 0.2) < 1e-9


def test_provincial_choropleth_skips_without_shapefile(tmp_path, caplog):
    """No shapefile / no geopandas → graceful skip, no crash, no PDF."""
    from scripts.visualize import plot_garg_weat_provincial_choropleth

    df = _make_summary(units=["北京", "上海"], categories=["leadership"])
    with caplog.at_level(logging.INFO):
        plot_garg_weat_provincial_choropleth(
            df, tmp_path, logging.getLogger("test"),
            category_sign=None, shapefile="/nonexistent/path.shp",
        )
    assert not any(n.startswith("garg_weat_choropleth") for n in _pdfs(tmp_path))
