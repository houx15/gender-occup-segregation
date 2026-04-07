#!/usr/bin/env python3
"""
Unified visualization for both prestige and WEAT analysis modes.

Usage:
    python -m scripts.visualize --config=config/config.yml
    python -m scripts.visualize --config=config/config.yml --mode=prestige
    python -m scripts.visualize --config=config/config.yml --mode=weat
"""

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import fire

from scripts.common.config_loader import load_config, get_analysis_unit, get_wordlist_dir
from scripts.common.logging_utils import setup_logging


# Chinese font setup
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "SimHei", "DejaVu Sans", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# Human-readable data source labels for figure titles
DATA_SOURCE_LABELS = {
    "ngram": "Google Ngram",
    "renminribao": "People's Daily",
    "weibo": "Weibo",
    "newspaper": "Provincial Newspapers",
}


def get_figure_path(filename: str, figures_dir: Path) -> Path:
    """Get dated figure path."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    if not filename.endswith(".pdf"):
        filename += ".pdf"
    return figures_dir / f"{date_str}_{filename}"


# =============================================================================
# Prestige mode plots
# =============================================================================

def plot_prestige_by_gender_over_time(df, figures_dir, logger):
    """Plot prestige scores for most male vs most female occupations over time."""
    if df.empty or "gender_score" not in df.columns or "time_slice" not in df.columns:
        return

    prestige_cols = [c for c in df.columns if c.endswith("_score") and c != "gender_score"]
    if not prestige_cols:
        return

    n_dims = len(prestige_cols)
    n_rows = (n_dims + 1) // 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 6 * n_rows))
    axes = np.array(axes).flatten() if n_dims > 1 else [axes]

    time_slices = df.sort_values("start_year")["time_slice"].unique()

    for idx, col in enumerate(prestige_cols):
        ax = axes[idx]
        male_scores, female_scores, labels = [], [], []
        for ts in time_slices:
            sd = df[df["time_slice"] == ts]
            if len(sd) < 10:
                continue
            n_top = max(1, len(sd) // 10)
            female_scores.append(sd.nlargest(n_top, "gender_score")[col].mean())
            male_scores.append(sd.nsmallest(n_top, "gender_score")[col].mean())
            labels.append(ts)

        ax.plot(range(len(labels)), male_scores, "o-", label="Most Male (10%)")
        ax.plot(range(len(labels)), female_scores, "s--", label="Most Female (10%)")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        dim_name = col.replace("_score", "").replace("_", " ").title()
        ax.set_title(f"{dim_name} by Gender Typing")
        ax.legend()
        ax.grid(True, alpha=0.3)

    for idx in range(n_dims, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    path = get_figure_path("prestige_by_gender_over_time", figures_dir)
    plt.savefig(path, format="pdf")
    plt.close()
    logger.info(f"  Saved: {path.name}")


def plot_gender_prestige_correlation(df, figures_dir, logger):
    """Plot correlation between gender and prestige scores over time."""
    if df.empty or "gender_score" not in df.columns or "time_slice" not in df.columns:
        return

    prestige_cols = [c for c in df.columns if c.endswith("_score") and c != "gender_score"]
    if not prestige_cols:
        return

    n_dims = len(prestige_cols)
    n_rows = (n_dims + 1) // 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 6 * n_rows))
    axes = np.array(axes).flatten() if n_dims > 1 else [axes]

    time_slices = df.sort_values("start_year")["time_slice"].unique()

    for idx, col in enumerate(prestige_cols):
        ax = axes[idx]
        corrs, labels = [], []
        for ts in time_slices:
            sd = df[df["time_slice"] == ts]
            if sd["gender_score"].notna().sum() > 1 and sd[col].notna().sum() > 1:
                corrs.append(sd["gender_score"].corr(sd[col]))
            else:
                corrs.append(np.nan)
            labels.append(ts)

        ax.plot(range(len(labels)), corrs, "o-")
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        dim_name = col.replace("_score", "").replace("_", " ").title()
        ax.set_title(f"Gender-Prestige Correlation: {dim_name}")
        ax.set_ylabel("Correlation")
        ax.grid(True, alpha=0.3)

    for idx in range(n_dims, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    path = get_figure_path("gender_prestige_correlation_over_time", figures_dir)
    plt.savefig(path, format="pdf")
    plt.close()
    logger.info(f"  Saved: {path.name}")


def plot_prestige_by_category(df, config, figures_dir, logger):
    """Plot prestige by occupation category over time."""
    if df.empty or "time_slice" not in df.columns:
        return

    wl_dir = get_wordlist_dir(config)
    cat_file = wl_dir / "occup_category.json"
    if not cat_file.exists():
        cat_file = wl_dir / "occup_category_zh.json"
    if not cat_file.exists():
        return

    with open(cat_file, "r", encoding="utf-8") as f:
        categories = json.load(f)

    prestige_cols = [c for c in df.columns if c.endswith("_score") and c != "gender_score"]
    if not prestige_cols:
        return

    time_slices = df.sort_values("start_year")["time_slice"].unique()

    # Sort categories by gender score
    cat_gender = {}
    for cat, occs in categories.items():
        cat_data = df[df["occupation"].isin(occs)]
        if not cat_data.empty:
            cat_gender[cat] = cat_data["gender_score"].mean()

    sorted_cats = sorted(cat_gender.items(), key=lambda x: x[1], reverse=True)
    n_cats = len(sorted_cats)
    n_cols_plot = 4
    n_rows = (n_cats + n_cols_plot - 1) // n_cols_plot

    fig, axes = plt.subplots(n_rows, n_cols_plot, figsize=(16, 4 * n_rows))
    axes = np.array(axes).flatten()

    for idx, (cat, avg_g) in enumerate(sorted_cats):
        ax = axes[idx]
        cat_data = df[df["occupation"].isin(categories[cat])]
        if cat_data.empty:
            ax.set_title(f"{cat}\n(no data)")
            continue

        for col in prestige_cols:
            vals = [cat_data[cat_data["time_slice"] == ts][col].mean() for ts in time_slices]
            dim_name = col.replace("_score", "").replace("_", " ").title()
            ax.plot(range(len(time_slices)), vals, "o-", label=dim_name, alpha=0.7)

        ax.set_xticks(range(len(time_slices)))
        ax.set_xticklabels(time_slices, rotation=45, ha="right", fontsize=8)
        ax.set_title(f"{cat}\n(Avg Gender: {avg_g:.3f})", fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    for idx in range(n_cats, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    path = get_figure_path("prestige_by_category_over_time", figures_dir)
    plt.savefig(path, format="pdf")
    plt.close()
    logger.info(f"  Saved: {path.name}")


# =============================================================================
# WEAT mode plots
# =============================================================================

def plot_weat_heatmap(weat_df, figures_dir, logger, data_source=None):
    """Plot WEAT Cohen's d heatmap across units and dimensions."""
    if weat_df.empty:
        return

    pivot = weat_df.pivot_table(index="unit", columns="dimension", values="cohens_d")
    if pivot.empty:
        return

    source_label = DATA_SOURCE_LABELS.get(data_source, "")
    source_suffix = f"  [Data: {source_label}]" if source_label else ""

    fig, ax = plt.subplots(figsize=(10, max(6, len(pivot) * 0.4)))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
    ax.set_title(f"WEAT Cohen's d by Unit and Dimension{source_suffix}")
    plt.tight_layout()
    path = get_figure_path("weat_cohens_d_heatmap", figures_dir)
    plt.savefig(path, format="pdf")
    plt.close()
    logger.info(f"  Saved: {path.name}")


def plot_weat_rankings(weat_df, figures_dir, logger, data_source=None):
    """Plot Cohen's d rankings for each dimension."""
    if weat_df.empty:
        return

    source_label = DATA_SOURCE_LABELS.get(data_source, "")
    source_suffix = f"  [Data: {source_label}]" if source_label else ""

    dimensions = weat_df["dimension"].unique()
    fig, axes = plt.subplots(1, len(dimensions), figsize=(6 * len(dimensions), max(6, len(weat_df["unit"].unique()) * 0.3)))
    if len(dimensions) == 1:
        axes = [axes]

    for ax, dim in zip(axes, dimensions):
        dim_data = weat_df[weat_df["dimension"] == dim].sort_values("cohens_d")
        colors = ["red" if d > 0 else "blue" for d in dim_data["cohens_d"]]
        ax.barh(dim_data["unit"], dim_data["cohens_d"], color=colors, alpha=0.7)
        ax.axvline(x=0, color="black", linewidth=0.5)
        ax.set_title(f"{dim} (Cohen's d){source_suffix}")
        ax.set_xlabel("Cohen's d")

    plt.tight_layout()
    path = get_figure_path("weat_rankings", figures_dir)
    plt.savefig(path, format="pdf")
    plt.close()
    logger.info(f"  Saved: {path.name}")


def plot_weat_longitudinal_trend(weat_df, figures_dir, logger, data_source=None):
    """Plot Cohen's d trend over time for longitudinal WEAT analysis.

    Units are expected to be time slices like '1940_1949'. If units don't
    parse as year ranges, this plot is skipped.
    """
    if weat_df.empty:
        return

    # Try to parse units as time slices (start_year_end_year)
    def parse_year(unit_name):
        try:
            parts = str(unit_name).split("_")
            return int(parts[0])
        except (ValueError, IndexError):
            return None

    weat_df = weat_df.copy()
    weat_df["start_year"] = weat_df["unit"].apply(parse_year)

    # Only proceed if most units parse as years
    if weat_df["start_year"].notna().sum() < 3:
        return

    weat_df = weat_df.dropna(subset=["start_year"]).sort_values("start_year")
    dimensions = weat_df["dimension"].unique()

    # Human-readable data source label for titles
    source_label = DATA_SOURCE_LABELS.get(data_source, data_source or "")
    source_suffix = f"  [Data: {source_label}]" if source_label else ""

    # Plot 1: All dimensions on one plot
    fig, ax = plt.subplots(figsize=(12, 6))
    markers = ["o", "s", "^", "D", "v"]
    for i, dim in enumerate(dimensions):
        dim_data = weat_df[weat_df["dimension"] == dim].sort_values("start_year")
        ax.plot(dim_data["start_year"], dim_data["cohens_d"],
                marker=markers[i % len(markers)], linewidth=2, label=dim.replace("_", " ").title())

    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Start Year of Time Slice")
    ax.set_ylabel("Cohen's d")
    ax.set_title(f"WEAT Gender Norm Indices Over Time{source_suffix}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = get_figure_path("weat_longitudinal_trend", figures_dir)
    plt.savefig(path, format="pdf")
    plt.close()
    logger.info(f"  Saved: {path.name}")

    # Plot 2: One standalone figure per dimension
    dim_labels = {
        "work_family": {
            "title": "Work-Family Gender Norm Over Time",
            "group1": "Family words", "group2": "Work words",
            "interpret": "Positive d = family more female-associated",
        },
        "leadership": {
            "title": "Leadership Gender Norm Over Time",
            "group1": "Non-leadership words", "group2": "Leadership words",
            "interpret": "Positive d = leadership more male-associated",
        },
        "stem": {
            "title": "STEM Gender Norm Over Time",
            "group1": "Non-STEM words", "group2": "STEM words",
            "interpret": "Positive d = STEM more male-associated",
        },
    }

    for dim in dimensions:
        dim_data = weat_df[weat_df["dimension"] == dim].sort_values("start_year")
        labels = dim_labels.get(dim, {})

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), height_ratios=[2, 1],
                                        sharex=True, gridspec_kw={"hspace": 0.08},
                                        layout="constrained")

        # Top panel: Cohen's d timeline
        ax1.plot(dim_data["start_year"], dim_data["cohens_d"], "o-",
                 linewidth=2.5, color="#2c3e50", markersize=8, zorder=5)
        ax1.fill_between(dim_data["start_year"], 0, dim_data["cohens_d"],
                         alpha=0.12, color="#2c3e50")
        ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        for threshold in [0.2, 0.5, 0.8]:
            ax1.axhline(y=threshold, color="lightcoral", linestyle=":", alpha=0.3)
            ax1.axhline(y=-threshold, color="lightblue", linestyle=":", alpha=0.3)
        ax1.set_ylabel("Cohen's d", fontsize=12)
        base_title = labels.get("title", f"{dim.replace('_', ' ').title()} Over Time")
        ax1.set_title(f"{base_title}{source_suffix}",
                      fontsize=14, fontweight="bold")
        ax1.grid(True, alpha=0.3)

        # Annotate interpretation
        interpret = labels.get("interpret", "")
        if interpret:
            ax1.text(0.02, 0.95, interpret, transform=ax1.transAxes,
                     fontsize=9, verticalalignment="top", color="gray",
                     fontstyle="italic")

        # Bottom panel: Group mean projections (if available)
        if "group1_mean" in dim_data.columns and "group2_mean" in dim_data.columns:
            g1_label = labels.get("group1", "Group 1")
            g2_label = labels.get("group2", "Group 2")
            ax2.plot(dim_data["start_year"], dim_data["group1_mean"], "s--",
                     linewidth=1.5, color="#e74c3c", markersize=6, label=g1_label, alpha=0.8)
            ax2.plot(dim_data["start_year"], dim_data["group2_mean"], "^--",
                     linewidth=1.5, color="#3498db", markersize=6, label=g2_label, alpha=0.8)
            ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
            ax2.set_ylabel("Mean projection\n(cosine sim)", fontsize=10)
            # Indicate axis direction: positive = female, negative = male
            # (axis is constructed as female - male)
            ax2.text(1.01, 0.95, "Female +", transform=ax2.transAxes,
                     fontsize=8, color="#e74c3c", verticalalignment="top", fontstyle="italic")
            ax2.text(1.01, 0.05, "Male +", transform=ax2.transAxes,
                     fontsize=8, color="#3498db", verticalalignment="bottom", fontstyle="italic")
            ax2.legend(fontsize=9, loc="best")
            ax2.grid(True, alpha=0.3)
        else:
            ax2.set_visible(False)

        # X-axis: time slice labels
        ax2.set_xlabel("Time Slice Start Year", fontsize=12)
        x_ticks = dim_data["start_year"].values
        ax2.set_xticks(x_ticks)
        ax2.set_xticklabels([str(int(y)) for y in x_ticks], rotation=45, ha="right")

        path = get_figure_path(f"weat_timeline_{dim}", figures_dir)
        plt.savefig(path, format="pdf")
        plt.close()
        logger.info(f"  Saved: {path.name}")


def plot_weat_projection_boxplots(proj_df, figures_dir, logger):
    """Plot boxplots of projection values per unit, grouped by concept category.

    Diagnostic plot to check cross-unit comparability (from reference visualizer Step 3).
    """
    if proj_df.empty or "unit" not in proj_df.columns:
        return

    categories = proj_df["category"].unique()
    n_cats = len(categories)
    fig, axes = plt.subplots(1, n_cats, figsize=(6 * n_cats, max(6, len(proj_df["unit"].unique()) * 0.3)))
    if n_cats == 1:
        axes = [axes]

    value_col = "projection_zscore" if "projection_zscore" in proj_df.columns else "cosine_sim"

    for ax, cat in zip(axes, sorted(categories)):
        cat_data = proj_df[proj_df["category"] == cat]
        units = sorted(cat_data["unit"].unique())
        data_per_unit = [cat_data[cat_data["unit"] == u][value_col].values for u in units]
        ax.boxplot(data_per_unit, tick_labels=units, vert=True)
        ax.set_xticklabels(units, rotation=45, ha="right", fontsize=8)
        ax.set_title(cat.replace("_", " ").title())
        ax.set_ylabel(value_col)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = get_figure_path("weat_projection_boxplots", figures_dir)
    plt.savefig(path, format="pdf")
    plt.close()
    logger.info(f"  Saved: {path.name}")


# Short province name → shapefile full name mapping (ADM1_ZH column)
SHORT_TO_FULL_PROVINCE = {
    "北京": "北京市", "天津": "天津市", "河北": "河北省", "山西": "山西省",
    "内蒙古": "内蒙古自治区", "辽宁": "辽宁省", "吉林": "吉林省",
    "黑龙江": "黑龙江省", "上海": "上海市", "江苏": "江苏省", "浙江": "浙江省",
    "安徽": "安徽省", "福建": "福建省", "江西": "江西省", "山东": "山东省",
    "河南": "河南省", "湖北": "湖北省", "湖南": "湖南省", "广东": "广东省",
    "广西": "广西壮族自治区", "海南": "海南省", "重庆": "重庆市", "四川": "四川省",
    "贵州": "贵州省", "云南": "云南省", "西藏": "西藏自治区", "陕西": "陕西省",
    "甘肃": "甘肃省", "青海": "青海省", "宁夏": "宁夏回族自治区",
    "新疆": "新疆维吾尔自治区",
}


def _match_province_in_shapefile(dim_data, china):
    """Match province full names (e.g. 北京市) to shapefile ADM1_ZH column."""
    # dim_data['province'] should already contain full names via SHORT_TO_FULL_PROVINCE
    match_col = "ADM1_ZH"
    for col in ["ADM1_ZH", "NAME", "name", "省", "name_zh"]:
        if col in china.columns:
            match_col = col
            break
    matched = china.merge(dim_data, left_on=match_col, right_on="province", how="left")
    return matched


def _plot_single_choropleth(merged, title, filename, figures_dir, logger):
    """Render and save a single choropleth map."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    merged.plot(
        column="cohens_d", ax=ax, legend=True, cmap="RdBu_r",
        missing_kwds={"color": "lightgrey", "label": "No data"},
        edgecolor="black", linewidth=0.3,
    )
    ax.set_title(title)
    ax.set_axis_off()
    plt.tight_layout()
    path = get_figure_path(filename, figures_dir)
    plt.savefig(path, format="pdf")
    plt.close()
    logger.info(f"  Saved: {path.name}")


def _parse_province_year(unit_name):
    """Parse province-year unit names like '北京_2020' → ('北京', 2020).

    Returns (province, year) or (None, None) if not parseable.
    """
    parts = str(unit_name).rsplit("_", 1)
    if len(parts) == 2:
        try:
            year = int(parts[1])
            if 1990 <= year <= 2030:
                return parts[0], year
        except ValueError:
            pass
    return None, None


def plot_weat_choropleth(weat_df, figures_dir, logger):
    """Plot choropleth maps of Cohen's d per province using geopandas.

    Detects province-year unit names (e.g. '北京_2020') and generates:
      - One map per dimension per year (per-year maps)
      - One averaged map per dimension (overall, averaged across years)
    Provinces without data are shown in grey.

    Falls back to simple province matching if units are not province-year format.

    Requires geopandas and a China shapefile. Skips gracefully if unavailable.
    """
    try:
        import geopandas as gpd
    except ImportError:
        logger.info("  Skipping choropleth maps (geopandas not installed)")
        return

    if weat_df.empty or "unit" not in weat_df.columns:
        return

    shapefile_paths = [
        Path("/lustre/home/2401111059/youth-analysis/configs/china_shp/chn_admbnda_adm1_ocha_2020.shp"),
        Path("data/shapefiles/china_provinces.shp"),
        Path("provincial/data/china_provinces.shp"),
        Path("shapefiles/china_provinces.shp"),
    ]
    shapefile = None
    for sp in shapefile_paths:
        if sp.exists():
            shapefile = sp
            break

    if shapefile is None:
        logger.info("  Skipping choropleth maps (no China shapefile found)")
        return

    china = gpd.read_file(shapefile)
    dimensions = weat_df["dimension"].unique()

    # Detect whether units are province-year format
    sample_units = weat_df["unit"].unique()[:5]
    parsed = [_parse_province_year(u) for u in sample_units]
    is_province_year = sum(1 for p, y in parsed if p is not None) >= len(sample_units) // 2 + 1

    for dim in dimensions:
        dim_raw = weat_df[weat_df["dimension"] == dim][["unit", "cohens_d"]].copy()

        if is_province_year:
            # Parse province-year units
            dim_raw[["province_short", "year"]] = dim_raw["unit"].apply(
                lambda u: pd.Series(_parse_province_year(u))
            )
            dim_raw = dim_raw.dropna(subset=["province_short"])
            if dim_raw.empty:
                continue

            # Map short names → full names for shapefile matching
            dim_raw["province"] = dim_raw["province_short"].map(SHORT_TO_FULL_PROVINCE)
            dim_raw = dim_raw.dropna(subset=["province"])

            # --- Per-year maps ---
            years = sorted(dim_raw["year"].unique())
            for year in years:
                year_data = dim_raw[dim_raw["year"] == year][["province", "cohens_d"]]
                if year_data.empty or year_data["cohens_d"].isna().all():
                    logger.info(f"  Skipping {dim} year={year}: no data")
                    continue
                merged = _match_province_in_shapefile(year_data, china)
                n_matched = merged["cohens_d"].notna().sum()
                if n_matched == 0:
                    logger.info(f"  Skipping {dim} year={year}: no shapefile matches")
                    continue
                title = f"{dim.replace('_', ' ').title()} {int(year)} (n={n_matched})"
                filename = f"weat_choropleth_{dim}_{int(year)}"
                _plot_single_choropleth(merged, title, filename, figures_dir, logger)

            # --- Overall averaged map ---
            avg_data = dim_raw.groupby("province")["cohens_d"].mean().reset_index()
            merged = _match_province_in_shapefile(avg_data, china)
            n_matched = merged["cohens_d"].notna().sum()
            if n_matched > 0:
                title = f"{dim.replace('_', ' ').title()} Average (n={n_matched})"
                filename = f"weat_choropleth_{dim}_overall"
                _plot_single_choropleth(merged, title, filename, figures_dir, logger)
        else:
            # Simple province matching (non province-year units)
            dim_data = dim_raw.rename(columns={"unit": "province"})
            merged = _match_province_in_shapefile(dim_data, china)
            if merged["cohens_d"].notna().sum() == 0:
                logger.info(f"  Skipping choropleth for {dim}: no matches")
                continue
            title = f"{dim.replace('_', ' ').title()} - Cohen's d by Province"
            _plot_single_choropleth(merged, title, f"weat_choropleth_{dim}",
                                    figures_dir, logger)


def plot_weat_choropleth_by_year(weat_df, figures_dir, logger):
    """Plot per-year choropleth maps for province-year WEAT analysis.

    This is a standalone function that generates separate choropleth maps
    for each year when units are in province_year format (e.g., '北京_2020').
    Provinces without data are shown in grey.

    Args:
        weat_df: DataFrame with columns [unit, dimension, cohens_d]
        figures_dir: Directory to save figures
        logger: Logger instance
    """
    try:
        import geopandas as gpd
    except ImportError:
        logger.info("  Skipping choropleth maps (geopandas not installed)")
        return

    if weat_df.empty or "unit" not in weat_df.columns:
        return

    # Detect whether units are province-year format
    sample_units = weat_df["unit"].unique()[:5]
    parsed = [_parse_province_year(u) for u in sample_units]
    is_province_year = sum(1 for p, y in parsed if p is not None) >= len(sample_units) // 2 + 1

    if not is_province_year:
        logger.info("  Units not in province-year format, skipping per-year choropleth")
        return

    shapefile_paths = [
        Path("/lustre/home/2401111059/youth-analysis/configs/china_shp/chn_admbnda_adm1_ocha_2020.shp"),
        Path("data/shapefiles/china_provinces.shp"),
        Path("provincial/data/china_provinces.shp"),
        Path("shapefiles/china_provinces.shp"),
    ]
    shapefile = None
    for sp in shapefile_paths:
        if sp.exists():
            shapefile = sp
            break

    if shapefile is None:
        logger.info("  Skipping choropleth maps (no China shapefile found)")
        return

    china = gpd.read_file(shapefile)
    dimensions = weat_df["dimension"].unique()

    for dim in dimensions:
        dim_raw = weat_df[weat_df["dimension"] == dim][["unit", "cohens_d"]].copy()

        # Parse province-year units
        dim_raw[["province_short", "year"]] = dim_raw["unit"].apply(
            lambda u: pd.Series(_parse_province_year(u))
        )
        dim_raw = dim_raw.dropna(subset=["province_short"])
        if dim_raw.empty:
            continue

        # Map short names → full names for shapefile matching
        dim_raw["province"] = dim_raw["province_short"].map(SHORT_TO_FULL_PROVINCE)
        dim_raw = dim_raw.dropna(subset=["province"])

        # Generate per-year maps
        years = sorted(dim_raw["year"].unique())
        for year in years:
            year_data = dim_raw[dim_raw["year"] == year][["province", "cohens_d"]]
            if year_data.empty or year_data["cohens_d"].isna().all():
                continue

            merged = _match_province_in_shapefile(year_data, china)
            n_matched = merged["cohens_d"].notna().sum()
            if n_matched == 0:
                continue

            title = f"{dim.replace('_', ' ').title()} {int(year)} (n={n_matched})"
            filename = f"weat_choropleth_{dim}_{int(year)}"
            _plot_single_choropleth(merged, title, filename, figures_dir, logger)


def plot_weat_year_comparison(weat_df, figures_dir, logger):
    """Plot side-by-side bar charts comparing provinces across years.

    Creates horizontal bar charts showing Cohen's d values for each province,
    with different colors/bars for each year. Only works for province-year format.

    Args:
        weat_df: DataFrame with columns [unit, dimension, cohens_d]
        figures_dir: Directory to save figures
        logger: Logger instance
    """
    if weat_df.empty or "unit" not in weat_df.columns:
        return

    # Detect whether units are province-year format
    sample_units = weat_df["unit"].unique()[:5]
    parsed = [_parse_province_year(u) for u in sample_units]
    is_province_year = sum(1 for p, y in parsed if p is not None) >= len(sample_units) // 2 + 1

    if not is_province_year:
        logger.info("  Units not in province-year format, skipping year comparison")
        return

    # Parse all units
    weat_df = weat_df.copy()
    weat_df[["province", "year"]] = weat_df["unit"].apply(
        lambda u: pd.Series(_parse_province_year(u))
    )
    weat_df = weat_df.dropna(subset=["province", "year"])

    if weat_df.empty:
        return

    dimensions = weat_df["dimension"].unique()

    for dim in dimensions:
        dim_data = weat_df[weat_df["dimension"] == dim].copy()

        # Create pivot for bar chart
        pivot_data = dim_data.pivot_table(
            index="province", columns="year", values="cohens_d", aggfunc="first"
        )

        if pivot_data.empty:
            continue

        fig, ax = plt.subplots(figsize=(10, max(8, len(pivot_data) * 0.5)))

        # Plot grouped horizontal bars
        pivot_data.plot(kind="barh", ax=ax, width=0.8)

        ax.axvline(x=0, color="black", linewidth=0.5)
        ax.set_xlabel("Cohen's d")
        ax.set_ylabel("Province")
        ax.set_title(f"{dim.replace('_', ' ').title()} - Province Comparison by Year")
        ax.legend(title="Year", loc="lower right")
        ax.grid(True, alpha=0.3, axis="x")

        plt.tight_layout()
        path = get_figure_path(f"weat_year_comparison_{dim}", figures_dir)
        plt.savefig(path, format="pdf")
        plt.close()
        logger.info(f"  Saved: {path.name}")


# =============================================================================
# Main
# =============================================================================

def main(config="config/config.yml", mode=None):
    """
    Create visualizations for analysis results.

    Args:
        config: Path to configuration file
        mode: "prestige", "weat", or None (auto-detect from config)
    """
    config_data = load_config(config)
    logger = setup_logging(Path(config_data["paths"]["log_dir"]), "visualize.log")

    logger.info("=" * 80)
    logger.info("Starting visualization")
    logger.info("=" * 80)

    sns.set_style("whitegrid")
    figures_dir = Path(config_data["paths"].get("figures_dir", config_data["paths"]["results_dir"] + "/figures"))
    results_dir = Path(config_data["paths"]["results_dir"])
    analysis_mode = mode or config_data.get("analysis_mode", "prestige")

    if analysis_mode == "prestige":
        # Load prestige results
        for fname in ("occupation_scores_by_slice.parquet", "occupation_scores_by_province.parquet"):
            fpath = results_dir / fname
            if fpath.exists():
                df = pd.read_parquet(fpath)
                logger.info(f"Loaded {fpath}: {len(df)} rows")
                if "time_slice" in df.columns:
                    plot_prestige_by_gender_over_time(df, figures_dir, logger)
                    plot_gender_prestige_correlation(df, figures_dir, logger)
                    plot_prestige_by_category(df, config_data, figures_dir, logger)
                break

    elif analysis_mode == "weat":
        weat_path = results_dir / "weat_results.csv"
        if weat_path.exists():
            weat_df = pd.read_csv(weat_path)
            logger.info(f"Loaded {weat_path}: {len(weat_df)} rows")

            # Check if data is in province-year format
            if not weat_df.empty and "unit" in weat_df.columns:
                sample_units = weat_df["unit"].unique()[:5]
                parsed = [_parse_province_year(u) for u in sample_units]
                is_province_year = sum(1 for p, y in parsed if p is not None) >= len(sample_units) // 2 + 1
            else:
                is_province_year = False

            ds = config_data.get("data_source")
            plot_weat_heatmap(weat_df, figures_dir, logger, data_source=ds)
            plot_weat_rankings(weat_df, figures_dir, logger, data_source=ds)
            plot_weat_longitudinal_trend(weat_df, figures_dir, logger, data_source=ds)

            if is_province_year:
                # For province-year data, use per-year choropleth and year comparison
                plot_weat_choropleth_by_year(weat_df, figures_dir, logger)
                plot_weat_year_comparison(weat_df, figures_dir, logger)
            else:
                # For other formats, use the general fallback
                plot_weat_choropleth(weat_df, figures_dir, logger)

        # Projection boxplots (diagnostic)
        proj_path = results_dir / "word_projections.csv"
        if proj_path.exists():
            proj_df = pd.read_csv(proj_path)
            logger.info(f"Loaded {proj_path}: {len(proj_df)} rows")
            plot_weat_projection_boxplots(proj_df, figures_dir, logger)

    logger.info("=" * 80)
    logger.info("Visualization completed!")
    logger.info("=" * 80)


if __name__ == "__main__":
    fire.Fire(main)
