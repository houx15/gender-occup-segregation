#!/usr/bin/env python3
"""
Provincial correlation analysis: correlate WEAT indices with socioeconomic data.

Migrated from reference/gender_norms/correlation_analysis_2024.py.
Only relevant for provincial analysis (weibo, newspaper).

Usage:
    python -m scripts.analyze_correlation --config=config/config.yml
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import fire

try:
    from scipy.stats import pearsonr
except ImportError:
    pearsonr = None

from scripts.common.config_loader import load_config
from scripts.common.logging_utils import setup_logging

# Chinese font setup — same fix as visualize.py
import matplotlib.font_manager as _fm
_CJK_FONT_PATH = "/usr/share/fonts/google-droid/DroidSansFallback.ttf"
_fm.fontManager.addfont(_CJK_FONT_PATH)
_CJK_FAMILY = _fm.FontProperties(fname=_CJK_FONT_PATH).get_name()
plt.rcParams["font.sans-serif"] = [_CJK_FAMILY] + plt.rcParams["font.sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

PROVINCE_NAME_MAPPING = {
    "北京": "北京市", "天津": "天津市", "上海": "上海市", "重庆": "重庆市",
    "河北": "河北省", "山西": "山西省", "辽宁": "辽宁省", "吉林": "吉林省",
    "黑龙江": "黑龙江省", "江苏": "江苏省", "浙江": "浙江省", "安徽": "安徽省",
    "福建": "福建省", "江西": "江西省", "山东": "山东省", "河南": "河南省",
    "湖北": "湖北省", "湖南": "湖南省", "广东": "广东省", "海南": "海南省",
    "四川": "四川省", "贵州": "贵州省", "云南": "云南省", "陕西": "陕西省",
    "甘肃": "甘肃省", "青海": "青海省", "台湾": "台湾省",
    "内蒙古": "内蒙古自治区", "广西": "广西壮族自治区", "西藏": "西藏自治区",
    "宁夏": "宁夏回族自治区", "新疆": "新疆维吾尔自治区",
    "香港": "香港特别行政区", "澳门": "澳门特别行政区",
}
PROVINCE_REVERSE = {v: k for k, v in PROVINCE_NAME_MAPPING.items()}


def normalize_province(name) -> str | None:
    text = "" if name is None else str(name).replace(" ", "").replace("\u3000", "").strip()
    if not text or text.lower() == "nan":
        return None
    if text.startswith("中国"):
        text = text.replace("中国", "", 1)
    if text in PROVINCE_NAME_MAPPING:
        return text
    if text in PROVINCE_REVERSE:
        return PROVINCE_REVERSE[text]
    for suffix in ("省", "市", "自治区", "特别行政区", "壮族自治区", "回族自治区", "维吾尔自治区"):
        if text.endswith(suffix):
            stripped = text[:-len(suffix)]
            if stripped in PROVINCE_NAME_MAPPING:
                return stripped
            if stripped in PROVINCE_REVERSE:
                return PROVINCE_REVERSE[stripped]
    return None


def compute_corr(x, y):
    data = pd.concat([x, y], axis=1).dropna()
    if data.empty or len(data) < 3:
        return np.nan, np.nan
    if pearsonr is None:
        return float(data.corr().iloc[0, 1]), np.nan
    r, p = pearsonr(data.iloc[:, 0], data.iloc[:, 1])
    return float(r), float(p)


def auto_discover_variables(df, exclude_cols):
    """Auto-discover all numeric columns in a DataFrame as correlation variables."""
    var_map = []
    for col in df.columns:
        if col in exclude_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            # Use column name as label, prettified
            label = col.replace("_", " ").title()
            var_map.append((col, label))
    return var_map


def apply_transforms(df, transforms, logger):
    """Apply column transforms (e.g., log, diff) defined in config."""
    for transform in transforms:
        name = transform["name"]
        op = transform["op"]
        if op == "log":
            source = transform["source"]
            if source in df.columns:
                df[name] = df[source].apply(
                    lambda v: np.log(v) if pd.notna(v) and v > 0 else np.nan
                )
                logger.info(f"  Transform: {name} = log({source})")
        elif op == "diff":
            a, b = transform["a"], transform["b"]
            if a in df.columns and b in df.columns:
                df[name] = df[a] - df[b]
                logger.info(f"  Transform: {name} = {a} - {b}")
    return df


def merge_longitudinal(norms, external_df, logger):
    """
    Merge WEAT results with external time-series data for longitudinal correlation.

    The external CSV must have a "year" or "start_year" column. Units in the norms
    file are parsed as time slices (e.g., "1940_1949") and matched by start_year.
    """
    # Parse start_year from unit names
    norms = norms.copy()

    def parse_start_year(unit):
        try:
            return int(str(unit).split("_")[0])
        except (ValueError, IndexError):
            return None

    norms["start_year"] = norms["unit"].apply(parse_start_year)
    norms = norms.dropna(subset=["start_year"])
    norms["start_year"] = norms["start_year"].astype(int)

    # Determine join column in external data
    if "start_year" in external_df.columns:
        join_col = "start_year"
    elif "year" in external_df.columns:
        external_df = external_df.rename(columns={"year": "start_year"})
        join_col = "start_year"
    else:
        logger.error("External data must have a 'year' or 'start_year' column")
        return None

    external_df[join_col] = pd.to_numeric(external_df[join_col], errors="coerce").astype("Int64")

    merged = norms.merge(external_df, on=join_col, how="inner")
    logger.info(f"  Merged: {len(merged)} time points")
    return merged


def merge_provincial(norms, external_df, logger):
    """
    Merge WEAT results with external provincial data.

    The external CSV must have a "province" column with Chinese province names.
    """
    norms = norms.copy()
    external_df = external_df.copy()

    unit_col = "unit" if "unit" in norms.columns else "province"
    norms["province"] = norms[unit_col].map(normalize_province)
    external_df["province"] = external_df["province"].map(normalize_province)

    merged = norms.merge(external_df, on="province", how="inner")
    logger.info(f"  Merged: {len(merged)} provinces")
    return merged


def run_correlation(merged, corr_config, figures_dir, logger):
    """Core correlation logic shared by both provincial and longitudinal modes."""
    # Apply transforms
    transforms = corr_config.get("transforms", [])
    if transforms:
        merged = apply_transforms(merged, transforms, logger)

    # Discover Cohen's d columns
    cohens_cols = []
    for col in merged.columns:
        if col.endswith("_cohens_d"):
            label = col.replace("_cohens_d", "").replace("_", " ").title() + " (Cohen's d)"
            cohens_cols.append((col, label))

    if not cohens_cols:
        logger.warning("No Cohen's d columns found in norms file")
        return

    # Discover external variables
    configured_vars = corr_config.get("variables")
    exclude_cols = {"province", "unit", "start_year", "end_year", "year"} | {c for c, _ in cohens_cols}
    # Also exclude any *_n columns from WEAT output
    exclude_cols |= {c for c in merged.columns if c.endswith("_group1_n") or c.endswith("_group2_n")}

    if configured_vars:
        var_map = []
        for v in configured_vars:
            if isinstance(v, dict):
                col, label = v["column"], v.get("label", v["column"])
            else:
                col, label = v, v.replace("_", " ").title()
            if col in merged.columns:
                var_map.append((col, label))
            else:
                logger.warning(f"  Variable '{col}' not found in merged data, skipping")
    else:
        var_map = auto_discover_variables(merged, exclude_cols)

    if not var_map:
        logger.warning("No external variables found for correlation")
        return

    logger.info(f"  Cohen's d columns: {[c for c, _ in cohens_cols]}")
    logger.info(f"  External variables: {[c for c, _ in var_map]}")

    # Compute correlations
    rows = []
    for coh_col, _ in cohens_cols:
        for var_col, _ in var_map:
            r, p = compute_corr(merged[coh_col], merged[var_col])
            n = merged[[coh_col, var_col]].dropna().shape[0]
            rows.append({"cohens_d": coh_col, "variable": var_col, "n": n, "pearson_r": r, "p_value": p})
    result_df = pd.DataFrame(rows)
    result_df.to_csv(figures_dir / "correlation_results.csv", index=False)
    logger.info(f"  Saved correlation results: {len(rows)} pairs")

    # Scatter plots
    date_prefix = datetime.now().strftime("%Y%m%d")
    for coh_col, coh_title in cohens_cols:
        nrows = (len(var_map) + 2) // 3
        fig, axes = plt.subplots(nrows, 3, figsize=(16, 4 * nrows))
        axes = axes.flatten()
        for ax, (var_col, var_label) in zip(axes, var_map):
            plot_df = merged[[coh_col, var_col]].dropna()
            if plot_df.empty:
                ax.set_axis_off()
                continue
            sns.regplot(data=plot_df, x=var_col, y=coh_col, ax=ax,
                       scatter_kws={"s": 30, "alpha": 0.8}, line_kws={"color": "red"})
            r, p = compute_corr(plot_df[var_col], plot_df[coh_col])
            ax.set_title(f"{var_label}\nr={r:.3f}, p={p:.3f}")
            ax.set_xlabel(var_label)
            ax.set_ylabel(coh_title)
        for ax in axes[len(var_map):]:
            ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(figures_dir / f"{date_prefix}_{coh_col}_correlation.pdf", format="pdf")
        plt.close(fig)


def main(config="config/config.yml", norms_file=None, external_data=None):
    """
    Correlate WEAT indices with external socioeconomic data.

    Supports two modes (auto-detected):
      - Provincial: external CSV has a "province" column
      - Longitudinal: external CSV has a "year" or "start_year" column

    The external CSV format:
      Provincial mode:
        province, gdp, income, education_years, ...
        北京, 43760, 85000, 13.2, ...

      Longitudinal mode:
        year, female_labor_participation, gdp_per_capita, ...
        1940, 0.45, 1200, ...
        1945, 0.48, 1350, ...
        (or use "start_year" to match time slice starts like 1940, 1945, ...)

    All numeric columns are auto-discovered for correlation.

    Args:
        config: Path to configuration file
        norms_file: Override path to gender_norm_index.csv
        external_data: Override path to external CSV file
    """
    config_data = load_config(config)
    logger = setup_logging(Path(config_data["paths"]["log_dir"]), "analyze_correlation.log")

    logger.info("=" * 80)
    logger.info("Starting correlation analysis")
    logger.info("=" * 80)

    results_dir = Path(config_data["paths"]["results_dir"])
    figures_dir = Path(config_data["paths"].get("figures_dir", str(results_dir / "correlation")))
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load WEAT results
    norms_path = norms_file or str(results_dir / "gender_norm_index.csv")
    corr_config = config_data.get("correlation", {})
    ext_path = external_data or corr_config.get("external_data") or corr_config.get("provincial_data") or str(
        Path(config_data["paths"]["base_dir"]) / "provincial" / "data" / "provincial_cleaned.csv"
    )

    logger.info(f"  Norms file: {norms_path}")
    logger.info(f"  External data: {ext_path}")

    norms = pd.read_csv(norms_path)
    ext_df = pd.read_csv(ext_path)

    # Auto-detect mode: provincial if "province" column exists, longitudinal if "year"/"start_year"
    if "province" in ext_df.columns:
        logger.info("  Mode: provincial (detected 'province' column)")
        merged = merge_provincial(norms, ext_df, logger)
    elif "year" in ext_df.columns or "start_year" in ext_df.columns:
        logger.info("  Mode: longitudinal (detected 'year'/'start_year' column)")
        merged = merge_longitudinal(norms, ext_df, logger)
    else:
        logger.error("External CSV must have either a 'province' column or a 'year'/'start_year' column")
        return

    if merged is None or len(merged) < 3:
        logger.error("Too few data points after merge (need >= 3)")
        if "province" in ext_df.columns:
            logger.info("  Tip: province names should be Chinese (e.g., 北京, 广东, 内蒙古)")
        else:
            logger.info("  Tip: year values should match time slice start years (e.g., 1940, 1945)")
        return

    run_correlation(merged, corr_config, figures_dir, logger)

    logger.info("=" * 80)
    logger.info("Correlation analysis completed!")
    logger.info("=" * 80)


if __name__ == "__main__":
    fire.Fire(main)
