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
import matplotlib.colors as mcolors
import matplotlib.font_manager as _fm
from scipy.stats import pearsonr

from scripts.common.config_loader import load_config, get_analysis_unit, get_wordlist_dir
from scripts.common.logging_utils import setup_logging


_DEFAULT_CJK_FONT_PATH = "/usr/share/fonts/google-droid/DroidSansFallback.ttf"


def _configure_fonts(config: dict) -> None:
    """Register language-appropriate fonts with matplotlib."""
    language = config["language"]
    if language != "zh":
        return  # matplotlib defaults are fine for English

    cjk_path = config.get("fonts", {}).get("cjk_path", _DEFAULT_CJK_FONT_PATH)
    try:
        _fm.fontManager.addfont(cjk_path)
        family = _fm.FontProperties(fname=cjk_path).get_name()
        plt.rcParams["font.sans-serif"] = [family] + plt.rcParams["font.sans-serif"]
        plt.rcParams["axes.unicode_minus"] = False
    except FileNotFoundError:
        pass


LABELS = {
    "zh": {
        "year": "年份",
        "start_year": "起始年份",
        "province": "省份",
        "state": "州",
        "gender_norm": "性别规范指数",
        "cohens_d": "Cohen's d 效应量",
        "cohens_d_abs": "|Cohen's d|",
        "prestige": "声望",
        "evaluation": "评价",
        "potency": "力量",
        "activity": "活动",
        "gender_axis": "性别轴投影",
        "work_family": "工作-家庭",
        "leadership": "领导力",
        "stem": "STEM",
        "male": "男性",
        "female": "女性",
        "occupation": "职业",
        "correlation": "皮尔逊相关系数",
        "slice": "时间窗",
        "value": "值",
    },
    "en": {
        "year": "Year",
        "start_year": "Start year",
        "province": "State",
        "state": "State",
        "gender_norm": "Gender norm index",
        "cohens_d": "Cohen's d",
        "cohens_d_abs": "|Cohen's d|",
        "prestige": "Prestige",
        "evaluation": "Evaluation",
        "potency": "Potency",
        "activity": "Activity",
        "gender_axis": "Gender-axis projection",
        "work_family": "Work–Family",
        "leadership": "Leadership",
        "stem": "STEM",
        "male": "Male",
        "female": "Female",
        "occupation": "Occupation",
        "correlation": "Pearson r",
        "slice": "Time window",
        "value": "Value",
    },
}


def L(config: dict, key: str) -> str:
    """Look up a user-facing label in the current language. Unknown keys fall back to the key itself."""
    return LABELS.get(config["language"], {}).get(key, key)

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
# Garg (2018) replication mode plots
# =============================================================================

def plot_garg_trend(df, figures_dir, logger, embedding_source=None):
    """Plot Garg (2018) Fig 2 replication: mean relative norm distance by decade.

    Args:
        df: summary DataFrame from analyze_garg.build_summary, with at least
            columns unit_name / mean_rnd / ci_low / ci_high. If a non-null
            mean_pct_diff column is present, an "Avg. Women Occupation %
            Difference" overlay is rendered on a twin y-axis (matching Garg's
            plot_creation.plot_averagebias_over_time_consistentoccupations).
        figures_dir: directory to write the figure into.
        logger: logger for status messages.
        embedding_source: optional short label like "trained_coha" or
            "histwords_sgns". Flows into the figure filename suffix
            (fig2_garg_replication__{source}.pdf) and title so runs against
            different embedding sources do not overwrite each other.

    Empty-DataFrame behavior: logs a WARNING and returns early without writing
    a figure. This avoids producing misleading empty plots when the upstream
    analyzer wrote a zero-row summary (e.g. all decades skipped).
    """
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"__{embedding_source}" if embedding_source else ""
    out_path = figures_dir / f"fig2_garg_replication{suffix}.pdf"

    if df is None or df.empty:
        logger.warning(
            "plot_garg_trend called with empty DataFrame; skipping write of "
            f"{out_path}"
        )
        return

    # Same shape as empty for the eye, but technically different — catch it
    # explicitly so the user gets a loud error instead of an empty PDF.
    if "mean_rnd" in df.columns and df["mean_rnd"].isna().all():
        logger.error(
            "plot_garg_trend: every mean_rnd is NaN — refusing to write a "
            "blank figure. Check the analyze_garg log for vocab/OOV "
            f"diagnostics. (Would have written {out_path}.)"
        )
        return

    df_sorted = df.sort_values("unit_name").reset_index(drop=True)

    has_pct_overlay = (
        "mean_pct_diff" in df_sorted.columns
        and df_sorted["mean_pct_diff"].notna().any()
    )

    fig, ax1 = plt.subplots(figsize=(10, 6))

    bias_color = "#1f4e79"  # blue, matches Garg's ax1
    pct_color = "#2e7d32"   # green, matches Garg's ax2

    ax1.plot(
        df_sorted["unit_name"],
        df_sorted["mean_rnd"],
        marker="o",
        color=bias_color,
        linewidth=1.8,
        label="Avg. women bias (embedding)",
    )
    ax1.fill_between(
        df_sorted["unit_name"],
        df_sorted["ci_low"],
        df_sorted["ci_high"],
        color=bias_color,
        alpha=0.18,
        label="95% CI",
    )
    ax1.axhline(y=0, color="lightgrey", linestyle="--", linewidth=1)
    ax1.set_xlabel("Decade")
    ax1.set_ylabel(
        "Avg. women bias  (||v − c_male|| − ||v − c_female||)",
        color=bias_color,
    )
    ax1.tick_params(axis="y", labelcolor=bias_color)
    ax1.tick_params(axis="x", rotation=45)

    if has_pct_overlay:
        ax2 = ax1.twinx()
        ax2.plot(
            df_sorted["unit_name"],
            df_sorted["mean_pct_diff"],
            marker="s",
            color=pct_color,
            linewidth=1.8,
            label="Avg. women occupation % difference (census)",
        )
        ax2.set_ylabel(
            "Avg. women occupation % difference  (2·female_share − 1)·100",
            color=pct_color,
        )
        ax2.tick_params(axis="y", labelcolor=pct_color)

        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="best", framealpha=0.85)
    else:
        ax1.legend(loc="best", framealpha=0.85)

    title_src = f" — {embedding_source}" if embedding_source else ""
    ax1.set_title(
        f"Garg (2018) Fig 2 replication: average gender bias of occupations{title_src}"
    )

    plt.tight_layout()
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close()
    logger.info(f"Saved Garg trend figure: {out_path}")


# =============================================================================
# Garg-WEAT (per-category RND) plot
# =============================================================================

def apply_ideation_sign(df, category_sign, value_cols):
    """Re-orient RND values onto a gender-ideation axis.

    For leadership/science, more female-leaning (higher RND) already means
    less-traditional ideation; for family it's the reverse — a more
    female-leaning 'family' is the *traditional* view. Multiplying family's
    values by -1 (via category_sign) puts every line on one axis where
    higher = less traditional. Negating a band swaps its low/high, but
    fill_between handles unordered bounds, so callers don't need to re-sort.
    """
    if not category_sign:
        return df
    df = df.copy()
    signs = df["category"].map(lambda c: category_sign.get(c, 1))
    for col in value_cols:
        if col in df.columns:
            df[col] = df[col] * signs
    return df


def plot_garg_weat_categories_trend(
    df, figures_dir, logger, embedding_source=None,
    band_cols=("ci_low", "ci_high"), band_tag="", line_col="mean_rnd",
    band_label=None, category_sign=None,
):
    """Plot per-category mean RND over time. One line per category (leadership,
    family, science) on a single axis, with a shaded uncertainty band.

    Companion to plot_garg_trend — same metric (relative norm distance,
    Garg sign convention), same gender axis, but partitioned into named
    occupation buckets. The expected DataFrame is
    analyze_garg_weat.build_summary's output.

    ``band_cols`` selects which (low, high) column pair shades the ribbon, and
    ``band_tag`` is appended to the filename — so the same summary renders both
    the Garg-style with-replacement bootstrap band (``ci_low``/``ci_high``) and
    the 80%-subsample band (``sub_low``/``sub_high``) as separate figures.

    Empty / all-NaN safeguards mirror plot_garg_trend: log loudly, refuse
    to write a blank PDF.
    """
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"__{embedding_source}" if embedding_source else ""
    tag = f"__{band_tag}" if band_tag else ""
    out_path = figures_dir / f"fig2_garg_weat_categories{suffix}{tag}.pdf"

    if df is None or df.empty:
        logger.warning(
            "plot_garg_weat_categories_trend called with empty DataFrame; "
            f"skipping write of {out_path}"
        )
        return

    if line_col in df.columns and df[line_col].isna().all():
        logger.error(
            f"plot_garg_weat_categories_trend: every {line_col} is NaN — "
            "refusing to write a blank figure. Check the analyze_garg_weat log "
            f"for vocab/consistent-set diagnostics. (Would have written "
            f"{out_path}.)"
        )
        return

    df = df.copy()

    def _parse_decade(unit_name):
        """Start year from a longitudinal unit name. Handles decade labels
        ('1990s' -> 1990, COHA/HistWords) and rolling-window slices
        ('1940_1949' -> 1940, the ngram / renminribao pipelines), matching
        plot_weat_longitudinal_trend.parse_year. Province and province-year
        units (e.g. '北京', '北京_2020') don't parse here — those route to the
        provincial RND plots instead of this trend.
        """
        s = str(unit_name)
        # Decade label: '1990s' -> 1990
        if len(s) == 5 and s.endswith("s") and s[:4].isdigit():
            return int(s[:4])
        # Window slice: '1940_1949' -> 1940 (non-numeric prefixes -> None)
        try:
            return int(s.split("_")[0])
        except (ValueError, IndexError):
            return None

    df["start_year"] = df["unit_name"].apply(_parse_decade)
    n_parsed = int(df["start_year"].notna().sum())
    if n_parsed == 0:
        logger.warning(
            "plot_garg_weat_categories_trend: no unit_name parsed as a "
            "'YYYYs' decade or 'YYYY_YYYY' window (sample: "
            f"{list(df['unit_name'].unique()[:5])}); skipping. Province / "
            "province-year units should use the provincial RND plots instead."
        )
        return

    df = df.dropna(subset=["start_year"]).sort_values("start_year")

    low_col, high_col = band_cols
    df = apply_ideation_sign(df, category_sign, [line_col, low_col, high_col])
    reversed_cats = (
        [c for c, s in category_sign.items() if s < 0] if category_sign else []
    )

    fig, ax = plt.subplots(figsize=(11, 6))

    # Distinct, accessible colours per category — falls back to matplotlib
    # default cycle for any extra/renamed category the user adds later.
    palette = {
        "leadership": "#1f4e79",
        "family":     "#c0392b",
        "science":    "#2e7d32",
    }
    markers = {"leadership": "o", "family": "s", "science": "^"}
    default_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])

    for i, (cat_name, group) in enumerate(df.groupby("category")):
        group = group.sort_values("start_year")
        color = palette.get(cat_name) or default_cycle[i % len(default_cycle)]
        marker = markers.get(cat_name, "o")

        label = cat_name.title() + (" (rev.)" if cat_name in reversed_cats else "")
        ax.plot(
            group["start_year"],
            group[line_col],
            marker=marker,
            color=color,
            linewidth=1.8,
            label=label,
        )
        if low_col in group.columns and high_col in group.columns:
            ax.fill_between(
                group["start_year"],
                group[low_col],
                group[high_col],
                color=color,
                alpha=0.15,
            )

    ax.axhline(y=0, color="lightgrey", linestyle="--", linewidth=1)
    ax.set_xlabel("Decade")
    if reversed_cats:
        ax.set_ylabel(
            "Gender ideation  (oriented RND)\n"
            "higher = less traditional"
        )
    else:
        ax.set_ylabel(
            "Mean RND  (||v − c_male|| − ||v − c_female||)\n"
            "larger = more female-leaning"
        )
    title_src = f" — {embedding_source}" if embedding_source else ""
    band_note = f"  [{band_label}]" if band_label else ""
    ax.set_title(
        f"Per-category gender-ideation over time{title_src}{band_note}"
    )
    if reversed_cats:
        fig.text(
            0.5, -0.02,
            f"{', '.join(sorted(reversed_cats))} reversed (RND × −1) so all "
            "lines share a 'less traditional = up' axis",
            ha="center", fontsize=8, color="grey",
        )
    ax.legend(title="Category", loc="best", framealpha=0.85)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close()
    logger.info(f"Saved Garg-WEAT categories figure: {out_path}")


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

    Recognised unit formats:
      - '1940_1949' (rolling-window slices used by the ngram pipeline)
      - '1990s'      (decade labels used by the COHA + HistWords pipelines)
    If units don't parse as either, this plot is skipped.
    """
    if weat_df.empty:
        return

    def parse_year(unit_name):
        s = str(unit_name)
        # Decade label: '1990s' -> 1990
        if len(s) == 5 and s.endswith("s") and s[:4].isdigit():
            return int(s[:4])
        # Window slice: '1940_1949' -> 1940
        try:
            parts = s.split("_")
            return int(parts[0])
        except (ValueError, IndexError):
            return None

    weat_df = weat_df.copy()
    weat_df["start_year"] = weat_df["unit"].apply(parse_year)

    # Only proceed if most units parse as years
    n_parsed = int(weat_df["start_year"].notna().sum())
    if n_parsed < 3:
        logger.warning(
            f"plot_weat_longitudinal_trend: only {n_parsed} of "
            f"{len(weat_df)} units parsed as years (sample units: "
            f"{list(weat_df['unit'].unique()[:5])}); skipping. Add the "
            "format to parse_year() above if this is a new unit naming "
            "convention."
        )
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
# Reverse lookup: full shapefile name → short display name
FULL_TO_SHORT_PROVINCE = {v: k for k, v in SHORT_TO_FULL_PROVINCE.items()}

# English province name → Chinese short name (for merging with survey data)
ENGLISH_TO_CHINESE_PROVINCE = {
    "Beijing": "北京", "Tianjin": "天津", "Hebei": "河北", "Shanxi": "山西",
    "Inner Mongolia": "内蒙古", "Liaoning": "辽宁", "Jilin": "吉林",
    "Heilongjiang": "黑龙江", "Shanghai": "上海", "Jiangsu": "江苏",
    "Zhejiang": "浙江", "Anhui": "安徽", "Fujian": "福建", "Jiangxi": "江西",
    "Shandong": "山东", "Henan": "河南", "Hubei": "湖北", "Hunan": "湖南",
    "Guangdong": "广东", "Guangxi": "广西", "Hainan": "海南", "Chongqing": "重庆",
    "Sichuan": "四川", "Guizhou": "贵州", "Yunnan": "云南", "Tibet": "西藏",
    "Shaanxi": "陕西", "Gansu": "甘肃", "Qinghai": "青海", "Ningxia": "宁夏",
    "Xinjiang": "新疆",
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


def _plot_single_choropleth(merged, title, filename, figures_dir, logger,
                            value_col="cohens_d", norm=None):
    """Render and save a single choropleth map with province name labels.

    ``value_col`` selects the column to colour by (default ``cohens_d`` for the
    WEAT path; the Garg-WEAT provincial path passes its oriented RND column).
    ``norm`` (e.g. a ``TwoSlopeNorm``) fixes a symmetric, dataset-wide colour
    range so the maps are directionally readable and comparable; default None
    keeps matplotlib's per-map autoscaling (the WEAT behaviour).
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    plot_kwargs = dict(
        column=value_col, ax=ax, legend=True, cmap="RdBu_r",
        missing_kwds={"color": "lightgrey", "label": "No data"},
        edgecolor="black", linewidth=0.3,
    )
    if norm is not None:
        plot_kwargs["norm"] = norm
    merged.plot(**plot_kwargs)

    # Label each province with its short name at the polygon centroid
    label_col = None
    for col in ["ADM1_ZH", "NAME", "name", "省", "name_zh"]:
        if col in merged.columns:
            label_col = col
            break

    if label_col is not None:
        for _, row in merged.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            full_name = row[label_col]
            short_name = FULL_TO_SHORT_PROVINCE.get(full_name, full_name)
            # Use representative_point (guaranteed inside polygon)
            try:
                pt = geom.representative_point()
            except Exception:
                continue
            # Provinces with data get dark label; no-data provinces get lighter
            has_data = pd.notna(row.get(value_col))
            ax.annotate(
                short_name,
                xy=(pt.x, pt.y),
                ha="center", va="center",
                fontsize=5.5,
                color="black" if has_data else "gray",
                alpha=0.85 if has_data else 0.45,
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


def _merge_weat_survey(weat_df, survey_csv_path):
    """Merge WEAT results with survey data on (province, year).

    Parses WEAT units (e.g. '北京_2020'), maps survey English province names
    to Chinese via ENGLISH_TO_CHINESE_PROVINCE, and performs an inner merge
    so only rows with both WEAT and survey data remain.

    Returns:
        DataFrame with columns: province_short, year, dimension, cohens_d,
        dataset, gender_ideation_mean, etc.
    """
    # Parse WEAT units into province_short and year
    weat = weat_df.copy()
    weat[["province_short", "year"]] = weat["unit"].apply(
        lambda u: pd.Series(_parse_province_year(u))
    )
    weat = weat.dropna(subset=["province_short"])
    weat["year"] = weat["year"].astype(int)

    # Load and map survey province names to Chinese
    survey = pd.read_csv(survey_csv_path)
    survey["province_short"] = survey["province"].map(ENGLISH_TO_CHINESE_PROVINCE)
    survey = survey.dropna(subset=["province_short"])

    # Inner merge: keep only rows where BOTH data sources exist
    merged = weat.merge(
        survey,
        on=["province_short", "year"],
        how="inner",
        suffixes=("_weat", "_survey"),
    )
    return merged


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
            # Simple province matching (non province-year units, e.g. "北京")
            dim_data = dim_raw.copy()
            dim_data["province"] = dim_data["unit"].map(SHORT_TO_FULL_PROVINCE)
            dim_data = dim_data.dropna(subset=["province"])
            if dim_data.empty:
                logger.info(f"  Skipping choropleth for {dim}: no province name matches")
                continue
            merged = _match_province_in_shapefile(dim_data, china)
            if merged["cohens_d"].notna().sum() == 0:
                logger.info(f"  Skipping choropleth for {dim}: no shapefile matches")
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
# Garg-WEAT provincial plots (cross-province RND — the per-province analogue of
# the longitudinal trend, replacing the Cohen's d provincial view). Reuses the
# metric-agnostic helpers (_match_province_in_shapefile, _plot_single_choropleth,
# _parse_province_year, province name maps); the Cohen's d functions above are
# left untouched.
# =============================================================================

def _find_china_shapefile(extra=None):
    """Return the first existing China province shapefile, or None.

    ``extra`` (e.g. config['paths']['shapefile']) is tried first so a profile
    can point at its own shapefile.
    """
    candidates = []
    if extra:
        candidates.append(Path(extra))
    candidates += [
        Path("/lustre/home/2401111059/youth-analysis/configs/china_shp/chn_admbnda_adm1_ocha_2020.shp"),
        Path("data/shapefiles/china_provinces.shp"),
        Path("provincial/data/china_provinces.shp"),
        Path("shapefiles/china_provinces.shp"),
    ]
    for sp in candidates:
        if sp and sp.exists():
            return sp
    return None


def _symmetric_vmax(values):
    """A 'nice' symmetric colour limit for diverging maps: max |value| rounded
    up to 1/2/2.5/5 × 10^k. Used so every map in one dataset shares the same
    0-centred, comparable range (e.g. -0.2..0.2), letting colour encode
    direction (and the family flip) consistently.
    """
    import math
    arr = np.asarray(list(values), dtype=float)
    arr = arr[~np.isnan(arr)]
    v = float(np.max(np.abs(arr))) if arr.size else 1.0
    if v <= 0:
        return 1.0
    exp = math.floor(math.log10(v))
    base = 10.0 ** exp
    for m in (1.0, 2.0, 2.5, 5.0, 10.0):
        if v <= m * base:
            return m * base
    return 10.0 * base


def _garg_weat_unit_kind(unit_names):
    """Classify garg_weat summary units to choose the right plots.

    Returns 'longitudinal' (decade '1990s' or window '1940_1949'),
    'province_year' ('北京_2020'), or 'province' ('北京'), by majority vote over
    a sample. A window like '1990_1999' is longitudinal (both sides 4-digit
    years), not province-year — the head must be non-numeric for province-year.
    """
    from collections import Counter

    def classify(u):
        s = str(u)
        if len(s) == 5 and s.endswith("s") and s[:4].isdigit():
            return "longitudinal"
        if "_" in s:
            head, tail = s.rsplit("_", 1)
            head_year = head.isdigit() and len(head) == 4
            tail_year = tail.isdigit() and len(tail) == 4
            if head_year and tail_year:
                return "longitudinal"
            if (not head_year) and tail_year:
                return "province_year"
        if s.isdigit():
            return "longitudinal"
        return "province"

    sample = list(pd.unique(list(unit_names)))[:12]
    if not sample:
        return "longitudinal"
    return Counter(classify(u) for u in sample).most_common(1)[0][0]


def _garg_weat_provincial_frame(summary_df, category_sign=None):
    """Shape build_summary output into a province-level RND frame.

    Returns (df, reversed_cats) where df has columns [province, category, value]
    — value is the (sign-oriented) mean RND, averaged across years when the
    units are province-year (e.g. '北京_2020'). reversed_cats lists the
    categories flipped by ideation_sign (for axis labelling).
    """
    df = summary_df.copy()
    df = df[df["mean_rnd"].notna()]
    if df.empty:
        return df.assign(province=[], value=[]), []

    # Orient onto the gender-ideation axis (higher = less traditional).
    df = apply_ideation_sign(df, category_sign, ["mean_rnd"])
    reversed_cats = (
        [c for c, s in category_sign.items() if s < 0] if category_sign else []
    )

    # province / province-year -> province (drop the year, average over it).
    def to_province(u):
        prov, _year = _parse_province_year(u)
        return prov if prov is not None else str(u)

    df["province"] = df["unit_name"].apply(to_province)
    out = (
        df.groupby(["province", "category"], as_index=False)["mean_rnd"]
        .mean()
        .rename(columns={"mean_rnd": "value"})
    )
    return out, reversed_cats


def _rnd_axis_label(reversed_cats):
    return (
        "Gender ideation (oriented RND)\nhigher = less traditional"
        if reversed_cats
        else "Mean RND  (larger = more female-leaning)"
    )


def plot_garg_weat_provincial_rankings(summary_df, figures_dir, logger,
                                       category_sign=None, data_source=None):
    """Horizontal bar rankings of per-province RND, one panel per category.

    The RND analogue of plot_weat_rankings; reads build_summary output.
    """
    df, reversed_cats = _garg_weat_provincial_frame(summary_df, category_sign)
    if df.empty:
        logger.warning("plot_garg_weat_provincial_rankings: no in-vocab rows; skipping")
        return

    source_label = DATA_SOURCE_LABELS.get(data_source, "")
    source_suffix = f"  [Data: {source_label}]" if source_label else ""
    categories = sorted(df["category"].unique())
    n_prov = df["province"].nunique()

    fig, axes = plt.subplots(
        1, len(categories),
        figsize=(6 * len(categories), max(6, n_prov * 0.3)),
    )
    if len(categories) == 1:
        axes = [axes]

    for ax, cat in zip(axes, categories):
        cat_data = df[df["category"] == cat].sort_values("value")
        colors = ["#c0392b" if v > 0 else "#1f4e79" for v in cat_data["value"]]
        ax.barh(cat_data["province"], cat_data["value"], color=colors, alpha=0.8)
        ax.axvline(x=0, color="black", linewidth=0.5)
        label = cat.title() + (" (rev.)" if cat in reversed_cats else "")
        ax.set_title(f"{label}{source_suffix}")
        ax.set_xlabel(_rnd_axis_label(reversed_cats))

    plt.tight_layout()
    path = get_figure_path("garg_weat_provincial_rankings", figures_dir)
    plt.savefig(path, format="pdf")
    plt.close()
    logger.info(f"  Saved: {path.name}")


def plot_garg_weat_provincial_heatmap(summary_df, figures_dir, logger,
                                      category_sign=None, data_source=None):
    """Province × category heatmap of per-province RND.

    The RND analogue of plot_weat_heatmap.
    """
    df, reversed_cats = _garg_weat_provincial_frame(summary_df, category_sign)
    if df.empty:
        logger.warning("plot_garg_weat_provincial_heatmap: no in-vocab rows; skipping")
        return

    pivot = df.pivot_table(index="province", columns="category", values="value")
    if pivot.empty:
        return

    source_label = DATA_SOURCE_LABELS.get(data_source, "")
    source_suffix = f"  [Data: {source_label}]" if source_label else ""
    rev_note = " — (rev.): " + ", ".join(sorted(reversed_cats)) if reversed_cats else ""

    fig, ax = plt.subplots(figsize=(10, max(6, len(pivot) * 0.4)))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
    ax.set_title(f"Per-province gender-ideation RND{source_suffix}{rev_note}")
    plt.tight_layout()
    path = get_figure_path("garg_weat_provincial_heatmap", figures_dir)
    plt.savefig(path, format="pdf")
    plt.close()
    logger.info(f"  Saved: {path.name}")


def plot_garg_weat_provincial_choropleth(summary_df, figures_dir, logger,
                                         category_sign=None, shapefile=None):
    """Choropleth maps of per-province RND, one per category.

    The RND analogue of plot_weat_choropleth. For province-year units it also
    draws per-year maps. Skips gracefully if geopandas or a shapefile are
    unavailable. Reuses the WEAT choropleth leaf helpers.
    """
    try:
        import geopandas as gpd
    except ImportError:
        logger.info("  Skipping RND choropleth maps (geopandas not installed)")
        return

    if summary_df is None or summary_df.empty or "unit_name" not in summary_df.columns:
        return

    sp = _find_china_shapefile(shapefile)
    if sp is None:
        logger.info("  Skipping RND choropleth maps (no China shapefile found)")
        return
    china = gpd.read_file(sp)

    df = summary_df.copy()
    df = df[df["mean_rnd"].notna()]
    if df.empty:
        return
    df = apply_ideation_sign(df, category_sign, ["mean_rnd"])

    # One symmetric, 0-centred colour range for EVERY map in this dataset, so
    # colour encodes direction (and the family flip) and maps are comparable.
    vmax = _symmetric_vmax(df["mean_rnd"])
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    sample_units = df["unit_name"].unique()[:5]
    parsed = [_parse_province_year(u) for u in sample_units]
    is_province_year = (
        sum(1 for p, _y in parsed if p is not None) >= len(sample_units) // 2 + 1
    )

    for cat in sorted(df["category"].unique()):
        cat_raw = df[df["category"] == cat][["unit_name", "mean_rnd"]].rename(
            columns={"mean_rnd": "value"}
        )

        if is_province_year:
            cat_raw[["province_short", "year"]] = cat_raw["unit_name"].apply(
                lambda u: pd.Series(_parse_province_year(u))
            )
            cat_raw = cat_raw.dropna(subset=["province_short"])
            if cat_raw.empty:
                continue
            cat_raw["province"] = cat_raw["province_short"].map(SHORT_TO_FULL_PROVINCE)
            cat_raw = cat_raw.dropna(subset=["province"])

            for year in sorted(cat_raw["year"].unique()):
                year_data = cat_raw[cat_raw["year"] == year][["province", "value"]]
                if year_data.empty or year_data["value"].isna().all():
                    continue
                merged = _match_province_in_shapefile(year_data, china)
                if merged["value"].notna().sum() == 0:
                    continue
                _plot_single_choropleth(
                    merged, f"{cat.title()} {int(year)}",
                    f"garg_weat_choropleth_{cat}_{int(year)}",
                    figures_dir, logger, value_col="value", norm=norm,
                )

            avg_data = cat_raw.groupby("province")["value"].mean().reset_index()
            merged = _match_province_in_shapefile(avg_data, china)
            if merged["value"].notna().sum() > 0:
                _plot_single_choropleth(
                    merged, f"{cat.title()} Average",
                    f"garg_weat_choropleth_{cat}_overall",
                    figures_dir, logger, value_col="value", norm=norm,
                )
        else:
            # Bare-province units (Weibo: '北京') are SHORT names; the shapefile
            # ADM1_ZH column holds FULL names ('北京市'). Map short -> full
            # before matching, mirroring plot_weat_choropleth's province branch.
            cat_data = cat_raw.rename(columns={"unit_name": "province_short"})
            cat_data["province"] = cat_data["province_short"].map(SHORT_TO_FULL_PROVINCE)
            cat_data = cat_data.dropna(subset=["province"])
            merged = _match_province_in_shapefile(cat_data[["province", "value"]], china)
            if merged["value"].notna().sum() == 0:
                logger.info(f"  Skipping RND choropleth for {cat}: no matches")
                continue
            _plot_single_choropleth(
                merged, f"{cat.title()} — RND by Province",
                f"garg_weat_choropleth_{cat}", figures_dir, logger,
                value_col="value", norm=norm,
            )


# =============================================================================
# Garg-WEAT provincial survey-correlation plots (RND analogues of the WEAT
# aggregated grid / CFPS-CGSS scatters / province longitudinal trends). All
# operate on the oriented RND (higher = less traditional), which shares the
# survey's direction, so the expected correlation with survey ideation is
# positive. Reuse the WEAT helpers; the WEAT functions are left untouched.
# =============================================================================

def _garg_weat_oriented_units(summary_df, category_sign=None):
    """Per-unit oriented RND long frame with columns [unit, category, value].

    Keeps the raw unit name (renamed 'unit') so the province-year helpers and
    _merge_weat_survey work unchanged.
    """
    df = summary_df.copy()
    df = df[df["mean_rnd"].notna()]
    if df.empty:
        return pd.DataFrame(columns=["unit", "category", "value"])
    df = apply_ideation_sign(df, category_sign, ["mean_rnd"])
    return df.rename(
        columns={"unit_name": "unit", "mean_rnd": "value"}
    )[["unit", "category", "value"]]


def _draw_fit(ax, x, y):
    """Overlay an OLS line + r / p / equation annotation on a scatter axis."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < 3:
        return
    xv, yv = x[m], y[m]
    slope, intercept = np.polyfit(xv, yv, 1)
    xl = np.linspace(xv.min(), xv.max(), 100)
    ax.plot(xl, slope * xl + intercept, "--", color="gray", linewidth=1.5, zorder=3)
    r, p = pearsonr(xv, yv)
    p_str = f"p = {p:.3f}" if p >= 0.001 else "p < 0.001"
    ax.text(
        0.05, 0.95, f"r = {r:.3f}\n{p_str}\ny = {slope:.2f}x + {intercept:.2f}",
        transform=ax.transAxes, fontsize=9, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )


def plot_garg_weat_choropleth_grid(summary_df, figures_dir, logger,
                                   category_sign=None, shapefile=None):
    """Category × recent-years grid of oriented-RND choropleths (province-year).
    RND analogue of plot_choropleth_aggregated_grid.
    """
    try:
        import geopandas as gpd
    except ImportError:
        logger.info("  Skipping RND choropleth grid (geopandas not installed)")
        return
    sp = _find_china_shapefile(shapefile)
    if sp is None:
        logger.info("  Skipping RND choropleth grid (no China shapefile)")
        return

    long = _garg_weat_oriented_units(summary_df, category_sign)
    if long.empty:
        return
    long[["province_short", "year"]] = long["unit"].apply(
        lambda u: pd.Series(_parse_province_year(u))
    )
    long = long.dropna(subset=["province_short"])
    if long.empty:
        logger.info("  Skipping RND grid (units are not province-year)")
        return
    long["year"] = long["year"].astype(int)

    china = gpd.read_file(sp)
    categories = sorted(long["category"].unique())
    years = sorted(long["year"].unique())[-4:]  # most recent up to 4
    # Same nice symmetric range as the individual maps -> comparable within the
    # dataset, 0-centred so colour encodes direction.
    vmax = _symmetric_vmax(long["value"])
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    fig, axes = plt.subplots(
        len(categories), len(years),
        figsize=(5 * len(years), 5 * len(categories)), squeeze=False,
    )
    fig.suptitle(
        "Gender-ideation RND by province (newspapers)",
        fontsize=16, fontweight="bold", y=0.98,
    )
    for r, cat in enumerate(categories):
        for c, year in enumerate(years):
            ax = axes[r][c]
            sub = long[(long["category"] == cat) & (long["year"] == year)][
                ["province_short", "value"]
            ].copy()
            if sub.empty:
                ax.set_axis_off(); ax.set_title("n=0", fontsize=9); continue
            sub["province"] = sub["province_short"].map(SHORT_TO_FULL_PROVINCE)
            sub = sub.dropna(subset=["province"])
            merged = _match_province_in_shapefile(sub[["province", "value"]], china)
            n = int(merged["value"].notna().sum())
            merged.plot(
                column="value", ax=ax, cmap="RdBu_r", norm=norm,
                missing_kwds={"color": "lightgrey"}, edgecolor="black", linewidth=0.15,
            )
            ax.set_axis_off(); ax.set_title(f"n={n}", fontsize=9)
        axes[r][0].text(
            -0.05, 0.5, cat.title(), transform=axes[r][0].transAxes,
            rotation=90, va="center", ha="right", fontsize=12, fontweight="bold",
        )
    for c, year in enumerate(years):
        axes[0][c].text(
            0.5, 1.12, str(year), transform=axes[0][c].transAxes,
            ha="center", va="bottom", fontsize=12, fontweight="bold",
        )
    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.70])
    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=norm); sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax).set_label("RND (oriented)", fontsize=10)
    plt.tight_layout(rect=[0, 0, 0.92, 0.96])
    path = get_figure_path("garg_weat_choropleth_grid", figures_dir)
    plt.savefig(path, format="pdf"); plt.close()
    logger.info(f"  Saved: {path.name}")


def plot_garg_weat_survey_scatter(summary_df, survey_csv_path, figures_dir, logger,
                                  category_sign=None):
    """Per-category oriented RND vs survey gender ideation (province-year).
    Writes garg_weat_scatter_{cfps,cgss,combined}. RND analogue of
    plot_survey_embedding_scatter.
    """
    long = _garg_weat_oriented_units(summary_df, category_sign)
    if long.empty:
        return
    merged = _merge_weat_survey(long, survey_csv_path)  # carries category, value
    if merged.empty:
        logger.info("  Skipping RND survey scatter: no merged data")
        return
    merged = merged[merged["dataset"].isin(["CFPS", "CGSS"])]
    if merged.empty:
        logger.info("  Skipping RND survey scatter: no CFPS/CGSS rows")
        return

    categories = sorted(long["category"].unique())
    period_bins = [(2007, 2009), (2010, 2014), (2015, 2019), (2020, 2024)]
    period_colors = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]
    period_labels = ["2007–2009", "2010–2014", "2015–2019", "2020–2024"]
    markers = {"CFPS": "P", "CGSS": "X"}

    def _period(y):
        for i, (lo, hi) in enumerate(period_bins):
            if lo <= y <= hi:
                return i
        return None
    merged["period"] = merged["year"].apply(_period)

    def _scatter(data, suffix):
        fig, axes = plt.subplots(
            1, len(categories), figsize=(6 * len(categories), 5.5), squeeze=False,
        )
        axes = axes[0]
        for ax, cat in zip(axes, categories):
            cd = data[data["category"] == cat].dropna(
                subset=["gender_ideation_mean", "value"]
            )
            for pi in range(len(period_bins)):
                sub = cd[cd["period"] == pi]
                if sub.empty:
                    continue
                if suffix == "combined":
                    for ds, mk in markers.items():
                        s2 = sub[sub["dataset"] == ds]
                        if not s2.empty:
                            ax.scatter(s2["gender_ideation_mean"], s2["value"],
                                       c=period_colors[pi], marker=mk, s=70,
                                       alpha=0.8, edgecolors="white", linewidths=0.5)
                else:
                    ax.scatter(sub["gender_ideation_mean"], sub["value"],
                               c=period_colors[pi], marker="o", s=70, alpha=0.8,
                               edgecolors="white", linewidths=0.5,
                               label=period_labels[pi])
            _draw_fit(ax, cd["gender_ideation_mean"].values, cd["value"].values)
            ax.set_title(cat.title(), fontsize=12, fontweight="bold")
            ax.set_xlabel("Survey gender ideation", fontsize=10)
            ax.set_ylabel("RND (oriented, higher = less traditional)", fontsize=9)
            ax.grid(True, alpha=0.3)
        axes[0].legend(fontsize=7, loc="lower right", framealpha=0.9)
        plt.tight_layout()
        path = get_figure_path(f"garg_weat_scatter_{suffix}", figures_dir)
        plt.savefig(path, format="pdf"); plt.close()
        logger.info(f"  Saved: {path.name}")

    cfps = merged[merged["dataset"] == "CFPS"]
    if not cfps.empty:
        _scatter(cfps, "cfps")
    cgss = merged[merged["dataset"] == "CGSS"]
    if not cgss.empty:
        _scatter(cgss, "cgss")
    _scatter(merged, "combined")


def plot_garg_weat_province_trends(summary_df, survey_csv_path, figures_dir, logger,
                                   category_sign=None):
    """Per-province dual-axis trends: oriented RND per category (left) vs survey
    ideation (right) over years. RND analogue of plot_province_longitudinal_trends.
    """
    long = _garg_weat_oriented_units(summary_df, category_sign)
    if long.empty:
        return
    long[["province_short", "year"]] = long["unit"].apply(
        lambda u: pd.Series(_parse_province_year(u))
    )
    long = long.dropna(subset=["province_short"])
    if long.empty:
        logger.info("  Skipping RND province trends (units not province-year)")
        return
    long["year"] = long["year"].astype(int)
    merged = _merge_weat_survey(
        _garg_weat_oriented_units(summary_df, category_sign), survey_csv_path
    )

    target = ["河南", "浙江", "内蒙古", "辽宁"]
    categories = sorted(long["category"].unique())
    palette = {"leadership": "#1f4e79", "family": "#c0392b", "science": "#2e7d32"}

    fig, axes = plt.subplots(1, len(target), figsize=(5 * len(target), 5), squeeze=False)
    axes = axes[0]
    fig.suptitle(
        "Province longitudinal trends: gender-ideation RND vs survey",
        fontsize=14, fontweight="bold",
    )
    for ax1, prov in zip(axes, target):
        pv = long[long["province_short"] == prov]
        for cat in categories:
            cd = pv[pv["category"] == cat].sort_values("year")
            if cd.empty:
                continue
            ax1.plot(cd["year"], cd["value"], "o-", color=palette.get(cat),
                     markersize=4, linewidth=1.4, label=cat.title())
        ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
        ax1.set_ylabel("RND (oriented)", fontsize=9)
        ax1.set_title(prov, fontsize=12, fontweight="bold")
        ax1.grid(True, alpha=0.2)
        ax2 = ax1.twinx()
        if not merged.empty:
            ps = merged[merged["province_short"] == prov]
            for ds, grp in ps.groupby("dataset"):
                grp = grp.drop_duplicates(subset=["year"]).sort_values("year")
                ax2.plot(grp["year"], grp["gender_ideation_mean"], "--", marker="X",
                         markersize=6, linewidth=1.3, alpha=0.85, color="#8e44ad",
                         label=f"{ds}")
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("Survey ideation", fontsize=9, color="#8e44ad")
        ax2.tick_params(axis="y", labelcolor="#8e44ad", labelsize=8)
    axes[0].legend(fontsize=7, loc="best")
    plt.tight_layout()
    path = get_figure_path("garg_weat_province_longitudinal_trends", figures_dir)
    plt.savefig(path, format="pdf"); plt.close()
    logger.info(f"  Saved: {path.name}")


# Province-level correlates from provincial_cleaned.csv. Each entry is
# (axis label, builder over the survey df, required columns) — mirrors the
# previous Weibo Cohen's d correlation panels (log GDP / education / employment
# / log income / education gap M-F / employment gap M-F / CFPS ideation).
_WEIBO_CORRELATES = [
    ("log(GDP)", lambda d: np.log(d["gdp_2024"]), ["gdp_2024"]),
    ("Average Years of Education Attainment",
     lambda d: d["eduy_gt25_2020"], ["eduy_gt25_2020"]),
    ("Employment Rate", lambda d: d["emp_2020"], ["emp_2020"]),
    ("log(Average Income)", lambda d: np.log(d["avg_income_2024"]), ["avg_income_2024"]),
    ("Difference in Education Years (Male - Female)",
     lambda d: d["eduy_m_gt25_2020"] - d["eduy_f_gt25_2020"],
     ["eduy_m_gt25_2020", "eduy_f_gt25_2020"]),
    ("Difference in Employment Rate (Male - Female)",
     lambda d: d["emp_m_2020"] - d["emp_f_2020"], ["emp_m_2020", "emp_f_2020"]),
    ("Gender Ideation in Survey (CFPS)",
     lambda d: d["cfps_ideation_2020"], ["cfps_ideation_2020"]),
]


def plot_garg_weat_weibo_correlation(summary_df, provincial_csv, figures_dir, logger,
                                     category_sign=None):
    """Per-category oriented RND vs province-level socioeconomic + survey
    variables (Weibo cross-section): one multi-panel figure per category,
    reproducing the previous Weibo Cohen's d correlation panels. Variables whose
    columns are absent from the CSV are skipped, so it degrades gracefully.
    """
    p = Path(provincial_csv)
    if not p.exists():
        logger.info(f"  Skipping Weibo correlation: {provincial_csv} not found")
        return
    long = _garg_weat_oriented_units(summary_df, category_sign)  # unit = province (zh)
    if long.empty:
        return
    surv = pd.read_csv(p)
    if "province" not in surv.columns:
        logger.info("  Skipping Weibo correlation: no 'province' column")
        return

    avail = [(lab, fn) for lab, fn, cols in _WEIBO_CORRELATES
             if all(c in surv.columns for c in cols)]
    if not avail:
        logger.info("  Skipping Weibo correlation: no usable provincial variables")
        return
    xvars = surv[["province"]].copy()
    for lab, fn in avail:
        xvars[lab] = fn(surv)

    ncols = 3
    nrows = (len(avail) + ncols - 1) // ncols
    for cat in sorted(long["category"].unique()):
        cd = long[long["category"] == cat][["unit", "value"]].rename(
            columns={"unit": "province"}
        )
        merged = cd.merge(xvars, on="province", how="inner")
        if merged.empty:
            logger.info(f"  Weibo correlation {cat}: no province overlap")
            continue
        ylabel = f"{cat.title()} gender norm from Weibo (RND, oriented)"
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows),
                                 squeeze=False)
        for i, (lab, _fn) in enumerate(avail):
            ax = axes[i // ncols][i % ncols]
            sub = merged[[lab, "value"]].dropna()
            if len(sub) >= 2:
                sns.regplot(
                    x=lab, y="value", data=sub, ax=ax,
                    scatter_kws={"s": 45, "alpha": 0.75, "color": "#4878a8"},
                    line_kws={"color": "red", "linewidth": 1.5},
                )
                title = lab
                if len(sub) >= 3:
                    r, pval = pearsonr(sub[lab].values, sub["value"].values)
                    pstr = f"p={pval:.3f}" if pval >= 0.001 else "p<0.001"
                    title = f"{lab}\nr={r:.3f}, {pstr}"
                ax.set_title(title, fontsize=10)
            else:
                ax.set_title(lab, fontsize=10)
            ax.set_xlabel(lab, fontsize=9)
            ax.set_ylabel(ylabel, fontsize=8)
        for j in range(len(avail), nrows * ncols):
            axes[j // ncols][j % ncols].set_axis_off()
        plt.tight_layout()
        path = get_figure_path(f"garg_weat_weibo_correlation_{cat}", figures_dir)
        plt.savefig(path, format="pdf"); plt.close()
        logger.info(f"  Saved: {path.name}")


# =============================================================================
# Composite: WEAT + Survey overlay
# =============================================================================

def plot_weat_survey_composite(weat_df, survey_df, figures_dir, logger, data_source=None):
    """Plot WEAT Cohen's d trends with survey gender ideation overlaid.

    Creates one figure per data source with:
      - Left y-axis: 3 WEAT dimension lines (work-family, leadership, STEM)
      - Right y-axis: survey scores (ACWF + CFPS) as scatter points

    Args:
        weat_df: WEAT results with columns [unit, dimension, cohens_d]
        survey_df: Survey data with columns [year, dataset, gender_ideation_mean]
        figures_dir: Output directory
        logger: Logger instance
        data_source: e.g. "ngram", "renminribao"
    """
    if weat_df.empty:
        return

    # Parse time slices to midpoint years
    def parse_midpoint(unit_name):
        try:
            parts = str(unit_name).split("_")
            return (int(parts[0]) + int(parts[1])) / 2
        except (ValueError, IndexError):
            return None

    weat_df = weat_df.copy()
    weat_df["mid_year"] = weat_df["unit"].apply(parse_midpoint)
    weat_df = weat_df.dropna(subset=["mid_year"]).sort_values("mid_year")

    if len(weat_df) < 3:
        return

    source_label = DATA_SOURCE_LABELS.get(data_source, data_source or "")

    # WEAT dimension styling: solid lines, markers, thick + translucent
    dim_styles = {
        "work_family": {"color": "#e74c3c", "marker": "o", "label": "WEAT: Work-Family"},
        "leadership":  {"color": "#2c3e50", "marker": "s", "label": "WEAT: Leadership"},
        "stem":        {"color": "#27ae60", "marker": "^", "label": "WEAT: STEM"},
    }

    # Survey dataset styling: solid lines, large markers, bold + opaque
    survey_styles = {
        "ACWF": {"color": "#8e44ad", "marker": "D", "label": "Survey: ACWF"},
        "CFPS": {"color": "#f39c12", "marker": "P", "label": "Survey: CFPS"},
        "CGSS": {"color": "#16a085", "marker": "X", "label": "Survey: CGSS"},
    }

    fig, ax1 = plt.subplots(figsize=(14, 7))

    # Left axis: WEAT Cohen's d — dashed lines, thick, semi-transparent
    dimensions = weat_df["dimension"].unique()
    for dim in dimensions:
        style = dim_styles.get(dim, {"color": "gray", "marker": "x", "label": dim})
        dim_data = weat_df[weat_df["dimension"] == dim].sort_values("mid_year")
        ax1.plot(dim_data["mid_year"], dim_data["cohens_d"],
                 linestyle="--", color=style["color"], marker=style["marker"],
                 linewidth=3, markersize=5, label=style["label"], alpha=0.45)

    ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.4)
    ax1.set_xlabel("Year", fontsize=12)
    ax1.set_ylabel("WEAT Cohen's d (text embedding)", fontsize=12, color="#2c3e50")
    ax1.tick_params(axis="y", labelcolor="#2c3e50")
    ax1.grid(True, alpha=0.2)

    # Direction annotations for left y-axis
    ax1.text(-0.01, 0.97, "More gender-\nstereotyped", transform=ax1.transAxes,
             fontsize=8, color="#2c3e50", alpha=0.7, verticalalignment="top",
             horizontalalignment="right", fontstyle="italic")
    ax1.text(-0.01, 0.03, "Less gender-\nstereotyped", transform=ax1.transAxes,
             fontsize=8, color="#2c3e50", alpha=0.7, verticalalignment="bottom",
             horizontalalignment="right", fontstyle="italic")

    # Right axis: survey scores — dashed lines, large markers, bold
    ax2 = ax1.twinx()
    if survey_df is not None and not survey_df.empty:
        for dataset_name, grp in survey_df.groupby("dataset"):
            style = survey_styles.get(dataset_name,
                                      {"color": "gray", "marker": "o", "label": dataset_name})
            grp = grp.sort_values("year")
            # Main line + markers — solid, opaque, prominent
            ax2.plot(grp["year"], grp["gender_ideation_mean"],
                     linestyle="-", color=style["color"], marker=style["marker"],
                     linewidth=3, markersize=12, label=style["label"], alpha=1.0,
                     zorder=10, markeredgecolor="white", markeredgewidth=1.5)

    ax2.set_ylabel("Survey gender ideation", fontsize=12, color="#8e44ad")
    ax2.tick_params(axis="y", labelcolor="#8e44ad")
    ax2.set_ylim(0, 1)

    # Direction annotations for right y-axis
    ax2.text(1.01, 0.97, "More\ntraditional", transform=ax2.transAxes,
             fontsize=8, color="#8e44ad", alpha=0.7, verticalalignment="top",
             horizontalalignment="left", fontstyle="italic")
    ax2.text(1.01, 0.03, "More\nprogressive", transform=ax2.transAxes,
             fontsize=8, color="#8e44ad", alpha=0.7, verticalalignment="bottom",
             horizontalalignment="left", fontstyle="italic")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9,
               framealpha=0.9)

    title = "Gender Norms: Text Embeddings vs. Survey Attitudes"
    if source_label:
        title += f"  [Text: {source_label}]"
    ax1.set_title(title, fontsize=14, fontweight="bold")

    path = get_figure_path("weat_survey_composite", figures_dir)
    plt.savefig(path, format="pdf", bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {path.name}")


def main_composite(weat_results_csv, data_source=None, survey_csv="data/surveys/processed/gender_ideation_by_year.csv",
                   figures_dir=None):
    """Standalone entry point for composite WEAT + survey figure.

    Usage:
        python -m scripts.visualize main_composite \\
            --weat_results_csv=results_weat/weat_results.csv \\
            --data_source=ngram \\
            --figures_dir=figures_weat
    """
    import logging
    logger = logging.getLogger("composite")
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    weat_df = pd.read_csv(weat_results_csv)
    survey_df = pd.read_csv(survey_csv) if Path(survey_csv).exists() else pd.DataFrame()

    if figures_dir is None:
        figures_dir = Path(weat_results_csv).parent / "figures"
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    sns.set_style("whitegrid")
    plot_weat_survey_composite(weat_df, survey_df, figures_dir, logger, data_source=data_source)


# =============================================================================
# Provincial survey correlation visualizations
# =============================================================================


def plot_choropleth_aggregated_grid(weat_df, figures_dir, logger, config=None):
    """Plot 3×4 grid of choropleth maps: 3 WEAT dimensions × 4 years.

    Creates a single figure with 12 small maps showing provincial Cohen's d
    values, with a shared diverging colorbar centered at 0.
    """
    if config is not None and config.get("language", "zh") != "zh":
        logger.info("Skipping survey comparison: zh-only")
        return

    try:
        import geopandas as gpd
    except ImportError:
        logger.info("  Skipping choropleth grid (geopandas not installed)")
        return

    if weat_df.empty:
        return

    shapefile = Path(
        "/lustre/home/2401111059/youth-analysis/configs/china_shp/"
        "chn_admbnda_adm1_ocha_2020.shp"
    )
    if not shapefile.exists():
        logger.info("  Skipping choropleth grid (shapefile not found)")
        return

    china = gpd.read_file(shapefile)

    # Parse WEAT units into province_short and year
    weat = weat_df.copy()
    weat[["province_short", "year"]] = weat["unit"].apply(
        lambda u: pd.Series(_parse_province_year(u))
    )
    weat = weat.dropna(subset=["province_short"])
    weat["year"] = weat["year"].astype(int)

    dimensions = ["work_family", "leadership", "stem"]
    years = [2018, 2020, 2022, 2024]
    dim_labels = {
        "work_family": "Work-Family",
        "leadership": "Leadership",
        "stem": "STEM",
    }

    # Compute global symmetric color range
    all_d = weat["cohens_d"]
    vmax = max(abs(all_d.min()), abs(all_d.max())) if len(all_d) > 0 else 1.0
    vmin = -vmax
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    fig, axes = plt.subplots(3, 4, figsize=(20, 16))
    fig.suptitle(
        "Gender Norm Index by Province (Provincial Newspapers)",
        fontsize=16, fontweight="bold", y=0.98,
    )

    for row_idx, dim in enumerate(dimensions):
        for col_idx, year in enumerate(years):
            ax = axes[row_idx, col_idx]
            dim_year = weat[(weat["dimension"] == dim) & (weat["year"] == year)]
            if dim_year.empty:
                ax.set_axis_off()
                ax.set_title("n=0", fontsize=9)
                continue

            plot_data = dim_year[["province_short", "cohens_d"]].copy()
            plot_data["province"] = plot_data["province_short"].map(
                SHORT_TO_FULL_PROVINCE
            )
            plot_data = plot_data.dropna(subset=["province"])

            merged = _match_province_in_shapefile(plot_data, china)
            n_data = merged["cohens_d"].notna().sum()

            norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
            merged.plot(
                column="cohens_d", ax=ax, cmap="RdBu_r", norm=norm,
                missing_kwds={"color": "lightgrey"},
                edgecolor="black", linewidth=0.15,
            )

            # Province name labels (same logic as _plot_single_choropleth)
            label_col = None
            for col in ["ADM1_ZH", "NAME", "name", "省", "name_zh"]:
                if col in merged.columns:
                    label_col = col
                    break
            if label_col is not None:
                for _, row in merged.iterrows():
                    geom = row.geometry
                    if geom is None or geom.is_empty:
                        continue
                    full_name = row[label_col]
                    short_name = FULL_TO_SHORT_PROVINCE.get(full_name, full_name)
                    try:
                        pt = geom.representative_point()
                    except Exception:
                        continue
                    has_data = pd.notna(row.get("cohens_d"))
                    ax.annotate(
                        short_name,
                        xy=(pt.x, pt.y),
                        ha="center", va="center",
                        fontsize=4,
                        color="black" if has_data else "gray",
                        alpha=0.7 if has_data else 0.3,
                    )

            ax.set_axis_off()
            ax.set_title(f"n={n_data}", fontsize=9)

        # Row label on left edge
        axes[row_idx, 0].text(
            -0.05, 0.5, dim_labels[dim],
            transform=axes[row_idx, 0].transAxes,
            rotation=90, va="center", ha="right",
            fontsize=12, fontweight="bold",
        )

    # Column labels on top
    for col_idx, year in enumerate(years):
        axes[0, col_idx].text(
            0.5, 1.12, str(year),
            transform=axes[0, col_idx].transAxes,
            ha="center", va="bottom",
            fontsize=12, fontweight="bold",
        )

    # Shared colorbar — use a dedicated axis on the right to avoid stealing
    # space from the map subplots (which pushes the colorbar into the middle)
    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.70])  # [left, bottom, width, height]
    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Cohen's d", fontsize=10)

    plt.tight_layout(rect=[0, 0, 0.92, 0.96])
    path = get_figure_path("choropleth_aggregated_grid", figures_dir)
    plt.savefig(path, format="pdf")
    plt.close()
    logger.info(f"  Saved: {path.name}")


def plot_survey_embedding_scatter(weat_df, survey_csv_path, figures_dir, logger, config=None):
    """Plot correlation scatter plots between WEAT embeddings and survey data.

    Creates three PDFs:
      - CFPS-only scatter
      - CGSS-only scatter
      - Combined scatter (both datasets, different markers)
    Each with 3 subplots (one per WEAT dimension).
    """
    if config is not None and config.get("language", "zh") != "zh":
        logger.info("Skipping survey comparison: zh-only")
        return

    merged = _merge_weat_survey(weat_df, survey_csv_path)
    if merged.empty:
        logger.info("  Skipping scatter plots: no merged data")
        return

    # Only use CGSS and CFPS
    merged = merged[merged["dataset"].isin(["CGSS", "CFPS"])]
    if merged.empty:
        logger.info("  Skipping scatter plots: no CGSS/CFPS data after merge")
        return

    dim_labels = {
        "work_family": "Work-Family",
        "leadership": "Leadership",
        "stem": "STEM",
    }
    dimensions = ["work_family", "leadership", "stem"]

    # Time period bins and colors
    period_bins = [(2007, 2009), (2010, 2014), (2015, 2019), (2020, 2024)]
    period_colors = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]
    period_labels = ["2007–2009", "2010–2014", "2015–2019", "2020–2024"]

    def _assign_period(year):
        for i, (lo, hi) in enumerate(period_bins):
            if lo <= year <= hi:
                return i
        return None

    merged["period"] = merged["year"].apply(_assign_period)

    survey_markers = {"CFPS": "P", "CGSS": "X"}

    def _make_scatter(data, filename_suffix):
        """Generate one scatter figure with 1×3 subplots."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=False)

        for idx, dim in enumerate(dimensions):
            ax = axes[idx]
            dim_data = data[
                data["dimension"] == dim
            ].dropna(subset=["gender_ideation_mean", "cohens_d"])

            if dim_data.empty:
                ax.set_title(dim_labels[dim], fontsize=12)
                ax.set_xlabel("Survey gender ideation")
                ax.set_ylabel("WEAT Cohen's d")
                continue

            # Plot points colored by time period
            for p_idx, (lo, hi) in enumerate(period_bins):
                mask = dim_data["period"] == p_idx
                subset = dim_data[mask]
                if subset.empty:
                    continue

                if filename_suffix == "combined":
                    # Different markers per dataset
                    for ds, marker in survey_markers.items():
                        ds_sub = subset[subset["dataset"] == ds]
                        if ds_sub.empty:
                            continue
                        ax.scatter(
                            ds_sub["gender_ideation_mean"],
                            ds_sub["cohens_d"],
                            c=period_colors[p_idx],
                            marker=marker,
                            s=80, alpha=0.8,
                            edgecolors="white", linewidths=0.5,
                            zorder=5,
                        )
                else:
                    ax.scatter(
                        subset["gender_ideation_mean"],
                        subset["cohens_d"],
                        c=period_colors[p_idx],
                        marker="o",
                        s=80, alpha=0.8,
                        edgecolors="white", linewidths=0.5,
                        label=period_labels[p_idx],
                        zorder=5,
                    )

            # OLS regression line across all points
            x = dim_data["gender_ideation_mean"].values
            y = dim_data["cohens_d"].values
            mask_valid = ~(np.isnan(x) | np.isnan(y))
            if mask_valid.sum() >= 3:
                x_valid, y_valid = x[mask_valid], y[mask_valid]
                slope, intercept = np.polyfit(x_valid, y_valid, 1)
                x_line = np.linspace(x_valid.min(), x_valid.max(), 100)
                ax.plot(
                    x_line, slope * x_line + intercept,
                    "--", color="gray", linewidth=1.5, zorder=3,
                )
                r, p = pearsonr(x_valid, y_valid)
                eq = f"y = {slope:.2f}x + {intercept:.2f}"
                p_str = f"p = {p:.3f}" if p >= 0.001 else "p < 0.001"
                ax.text(
                    0.05, 0.95,
                    f"r = {r:.3f}\n{p_str}\n{eq}",
                    transform=ax.transAxes, fontsize=9,
                    verticalalignment="top",
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        facecolor="white", alpha=0.8,
                    ),
                )

            ax.set_title(dim_labels[dim], fontsize=12, fontweight="bold")
            ax.set_xlabel("Survey gender ideation", fontsize=10)
            if idx == 0:
                ax.set_ylabel("WEAT Cohen's d", fontsize=10)
            ax.grid(True, alpha=0.3)

        # Build legend: period colors + dataset markers
        # Period color legend
        from matplotlib.lines import Line2D
        period_handles = [
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=period_colors[i], markersize=8,
                   label=period_labels[i])
            for i in range(len(period_bins))
        ]

        if filename_suffix == "combined":
            ds_handles = [
                Line2D([0], [0], marker=survey_markers[ds], color="w",
                       markerfacecolor="gray", markersize=8,
                       label=f"Survey: {ds}")
                for ds in ["CFPS", "CGSS"]
            ]
            all_handles = period_handles + ds_handles
        else:
            all_handles = period_handles

        axes[0].legend(
            handles=all_handles, fontsize=7,
            loc="lower right", framealpha=0.9,
        )

        plt.tight_layout()
        path = get_figure_path(f"scatter_{filename_suffix}", figures_dir)
        plt.savefig(path, format="pdf")
        plt.close()
        logger.info(f"  Saved: {path.name}")

    # CFPS only
    cfps_data = merged[merged["dataset"] == "CFPS"]
    if not cfps_data.empty:
        _make_scatter(cfps_data, "cfps")
    else:
        logger.info("  Skipping CFPS scatter: no data")

    # CGSS only
    cgss_data = merged[merged["dataset"] == "CGSS"]
    if not cgss_data.empty:
        _make_scatter(cgss_data, "cgss")
    else:
        logger.info("  Skipping CGSS scatter: no data")

    # Combined
    _make_scatter(merged, "combined")


def plot_province_longitudinal_trends(weat_df, survey_csv_path, figures_dir, logger, config=None):
    """Plot longitudinal trends for 4 provinces with dual WEAT/survey axes.

    Creates a 3×4 grid (3 dimensions × 4 provinces) with dual y-axes
    showing WEAT Cohen's d (left) and survey gender ideation (right).
    """
    if config is not None and config.get("language", "zh") != "zh":
        logger.info("Skipping survey comparison: zh-only")
        return

    merged = _merge_weat_survey(weat_df, survey_csv_path)

    # Parse all WEAT data for province-year lines (not just merged with survey)
    weat = weat_df.copy()
    weat[["province_short", "year"]] = weat["unit"].apply(
        lambda u: pd.Series(_parse_province_year(u))
    )
    weat = weat.dropna(subset=["province_short"])
    weat["year"] = weat["year"].astype(int)

    target_provinces = {
        "河南": "Henan",
        "浙江": "Zhejiang",
        "内蒙古": "Inner Mongolia",
        "辽宁": "Liaoning",
    }

    dimensions = ["work_family", "leadership", "stem"]
    dim_labels = {
        "work_family": "Work-Family",
        "leadership": "Leadership",
        "stem": "STEM",
    }
    survey_styles = {
        "CFPS": {"color": "#f39c12", "marker": "P", "label": "CFPS Survey"},
        "CGSS": {"color": "#16a085", "marker": "X", "label": "CGSS Survey"},
    }

    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    fig.suptitle(
        "Province Longitudinal Trends: WEAT vs. Survey",
        fontsize=14, fontweight="bold", y=0.98,
    )

    for row_idx, dim in enumerate(dimensions):
        weat_dim = weat[weat["dimension"] == dim]
        merged_dim = (
            merged[merged["dimension"] == dim] if not merged.empty
            else pd.DataFrame()
        )

        for col_idx, (prov_cn, _prov_en) in enumerate(
            target_provinces.items()
        ):
            ax1 = axes[row_idx, col_idx]

            # WEAT embedding line (left axis)
            prov_weat = weat_dim[
                weat_dim["province_short"] == prov_cn
            ].sort_values("year")
            if not prov_weat.empty:
                ax1.plot(
                    prov_weat["year"], prov_weat["cohens_d"], "o-",
                    color="#2c3e50", markersize=5, linewidth=1.5,
                    label="WEAT Cohen's d", zorder=5,
                )
                ax1.fill_between(
                    prov_weat["year"], 0, prov_weat["cohens_d"],
                    alpha=0.1, color="#2c3e50",
                )

            ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
            ax1.set_ylabel("Cohen's d", fontsize=9, color="#2c3e50")
            ax1.tick_params(axis="y", labelcolor="#2c3e50", labelsize=8)
            ax1.tick_params(axis="x", labelsize=8)
            ax1.grid(True, alpha=0.2)

            # Survey data (right axis)
            ax2 = ax1.twinx()
            prov_survey = (
                merged_dim[merged_dim["province_short"] == prov_cn]
                if not merged_dim.empty
                else pd.DataFrame()
            )
            if not prov_survey.empty:
                for ds_name, grp in prov_survey.groupby("dataset"):
                    if ds_name not in survey_styles:
                        continue
                    style = survey_styles[ds_name]
                    grp = grp.sort_values("year")
                    ax2.plot(
                        grp["year"], grp["gender_ideation_mean"],
                        linestyle="--", color=style["color"],
                        marker=style["marker"], markersize=7,
                        linewidth=1.5, label=style["label"],
                        alpha=0.9, zorder=4,
                    )

            ax2.set_ylim(0, 1)
            ax2.set_ylabel("Survey ideation", fontsize=9, color="#8e44ad")
            ax2.tick_params(axis="y", labelcolor="#8e44ad", labelsize=8)

            # Province title on top row only
            if row_idx == 0:
                ax1.set_title(prov_cn, fontsize=12, fontweight="bold")

            # Legend on first subplot only
            if row_idx == 0 and col_idx == 0:
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(
                    lines1 + lines2, labels1 + labels2,
                    fontsize=7, loc="upper right", framealpha=0.8,
                )

            # Row labels on left edge
            if col_idx == 0:
                ax1.text(
                    -0.15, 0.5, dim_labels[dim],
                    transform=ax1.transAxes,
                    rotation=90, va="center", ha="right",
                    fontsize=10, fontweight="bold",
                )

            # X label on bottom row only
            if row_idx == 2:
                ax1.set_xlabel("Year", fontsize=9)

    plt.tight_layout(rect=[0.03, 0, 1, 0.96])
    path = get_figure_path("province_longitudinal_trends", figures_dir)
    plt.savefig(path, format="pdf")
    plt.close()
    logger.info(f"  Saved: {path.name}")


def main(config="config/config.yml", mode=None):
    """
    Create visualizations for analysis results.

    Args:
        config: Path to configuration file
        mode: "prestige", "weat", "garg", or None (auto-detect from config)
    """
    config_data = load_config(config)
    logger = setup_logging(Path(config_data["paths"]["log_dir"]), "visualize.log")

    logger.info("=" * 80)
    logger.info("Starting visualization")
    logger.info("=" * 80)

    sns.set_style("whitegrid")
    _configure_fonts(config_data)  # must run after sns.set_style resets rcParams
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

            # Composite: WEAT + survey overlay (for longitudinal data)
            if not is_province_year:
                survey_path = Path("data/surveys/processed/gender_ideation_by_year.csv")
                if survey_path.exists():
                    survey_df = pd.read_csv(survey_path)
                    logger.info(f"Loaded survey data: {len(survey_df)} rows")
                    plot_weat_survey_composite(weat_df, survey_df, figures_dir, logger, data_source=ds)

            # Provincial survey correlation analysis
            if is_province_year:
                survey_csv = Path("data/surveys/processed/gender_ideation_by_province_year.csv")
                if survey_csv.exists():
                    survey_df = pd.read_csv(survey_csv)
                    logger.info(f"Loaded provincial survey data: {len(survey_df)} rows")
                    plot_choropleth_aggregated_grid(weat_df, figures_dir, logger, config=config_data)
                    plot_survey_embedding_scatter(weat_df, str(survey_csv), figures_dir, logger, config=config_data)
                    plot_province_longitudinal_trends(weat_df, str(survey_csv), figures_dir, logger, config=config_data)

        # Projection boxplots (diagnostic)
        proj_path = results_dir / "word_projections.csv"
        if proj_path.exists():
            proj_df = pd.read_csv(proj_path)
            logger.info(f"Loaded {proj_path}: {len(proj_df)} rows")
            plot_weat_projection_boxplots(proj_df, figures_dir, logger)

    elif analysis_mode == "garg":
        summary_path = results_dir / "garg_average_bias_by_decade.parquet"
        if summary_path.exists():
            df = pd.read_parquet(summary_path)
            logger.info(f"Loaded {summary_path}: {len(df)} rows")
            embedding_source = config_data.get("embedding_source")
            plot_garg_trend(df, figures_dir, logger, embedding_source=embedding_source)
        else:
            logger.error(f"Garg summary not found at {summary_path}")

    elif analysis_mode == "garg_weat":
        summary_path = results_dir / "garg_weat_summary_by_category.parquet"
        if summary_path.exists():
            df = pd.read_parquet(summary_path)
            logger.info(f"Loaded {summary_path}: {len(df)} rows")
            embedding_source = config_data.get("embedding_source")
            # Per-category orientation onto a gender-ideation axis (e.g.
            # family reversed): config analysis.ideation_sign maps
            # category -> +1/-1. Absent -> raw RND, no flip.
            category_sign = config_data.get("analysis", {}).get("ideation_sign")
            ds = config_data.get("data_source")
            kind = (
                _garg_weat_unit_kind(df["unit_name"]) if not df.empty
                else "longitudinal"
            )
            logger.info(f"garg_weat units classified as: {kind}")

            if kind == "longitudinal":
                # Two versions of the band on the same trend:
                #   bootstrap — Garg's with-replacement occupation bootstrap
                #   subsample — keep 80% of words without replacement, N rounds
                plot_garg_weat_categories_trend(
                    df, figures_dir, logger, embedding_source=embedding_source,
                    band_cols=("ci_low", "ci_high"), band_tag="bootstrap",
                    band_label="bootstrap CI (Garg)", category_sign=category_sign,
                )
                plot_garg_weat_categories_trend(
                    df, figures_dir, logger, embedding_source=embedding_source,
                    band_cols=("sub_low", "sub_high"), band_tag="subsample",
                    band_label="80% word-subsample band", category_sign=category_sign,
                )
            else:
                # Provincial cross-section (province or province-year): RND
                # replaces the Cohen's d provincial view (rankings / heatmap /
                # choropleth). Cohen's d plots stay available via analyze_weat.
                shapefile = config_data.get("paths", {}).get("shapefile")
                plot_garg_weat_provincial_rankings(
                    df, figures_dir, logger,
                    category_sign=category_sign, data_source=ds,
                )
                plot_garg_weat_provincial_heatmap(
                    df, figures_dir, logger,
                    category_sign=category_sign, data_source=ds,
                )
                plot_garg_weat_provincial_choropleth(
                    df, figures_dir, logger,
                    category_sign=category_sign, shapefile=shapefile,
                )
                # Survey-correlation extras (RND analogues of the WEAT views).
                if kind == "province_year":
                    # Newspaper: province-year survey (CFPS/CGSS by year).
                    survey_csv = Path(
                        "data/surveys/processed/gender_ideation_by_province_year.csv"
                    )
                    if survey_csv.exists():
                        plot_garg_weat_choropleth_grid(
                            df, figures_dir, logger,
                            category_sign=category_sign, shapefile=shapefile,
                        )
                        plot_garg_weat_survey_scatter(
                            df, str(survey_csv), figures_dir, logger,
                            category_sign=category_sign,
                        )
                        plot_garg_weat_province_trends(
                            df, str(survey_csv), figures_dir, logger,
                            category_sign=category_sign,
                        )
                    else:
                        logger.info(
                            f"  No province-year survey at {survey_csv}; "
                            "skipping grid / scatter / trends"
                        )
                else:
                    # Weibo (province cross-section): most-recent provincial
                    # survey value per province.
                    prov_csv = Path("data/surveys/provincial/provincial_cleaned.csv")
                    if prov_csv.exists():
                        plot_garg_weat_weibo_correlation(
                            df, str(prov_csv), figures_dir, logger,
                            category_sign=category_sign,
                        )
                    else:
                        logger.info(
                            f"  No provincial survey at {prov_csv}; "
                            "skipping Weibo correlation"
                        )
        else:
            logger.error(f"Garg-WEAT summary not found at {summary_path}")

    logger.info("=" * 80)
    logger.info("Visualization completed!")
    logger.info("=" * 80)


if __name__ == "__main__":
    fire.Fire({"main": main, "composite": main_composite})
