"""Metric-agnostic per-category summaries for single-wordlist bias analyses.

Shared by analyze_garg_weat (RND) and analyze_cohens_d_singlelist (cosine
projection). The only thing that varies between metrics is the per-word value;
everything here (categories, consistent set, bootstrap + subsample bands, the
mean and proportion-male-leaned statistics) operates on a generic value column.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

from scripts.common.config_loader import get_wordlist_dir
from scripts.common.metrics import bootstrap_ci, proportion_below


def _mean(x: np.ndarray) -> float:
    return float(np.mean(x))


def _prop_male(x: np.ndarray) -> float:
    return proportion_below(x, 0.0)


def load_categories(config: dict, logger) -> Dict[str, List[str]]:
    """Read category occupation files from config['wordlists']['categories']."""
    wl_dir = get_wordlist_dir(config)
    cats_cfg = config.get("wordlists", {}).get("categories")
    if not cats_cfg:
        raise ValueError(
            "config must define wordlists.categories "
            "(mapping category-name -> filename)"
        )
    out: Dict[str, List[str]] = {}
    for cat_name, fname in cats_cfg.items():
        path = wl_dir / fname
        if not path.exists():
            raise FileNotFoundError(f"Category file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            words = [line.strip() for line in f if line.strip()]
        out[cat_name] = words
        logger.info(f"  {cat_name}: loaded {len(words)} candidates from {path.name}")
    return out


def compute_consistent_set(
    long_df: pd.DataFrame,
    categories: Dict[str, List[str]],
    units: List[str],
    logger,
) -> Dict[str, List[str]]:
    """Per category, find occupations in vocab in ALL units."""
    consistent: Dict[str, List[str]] = {}
    for cat_name in categories:
        cat_df = long_df[long_df["category"] == cat_name]
        in_all = set(cat_df.loc[cat_df["in_vocab"], "occupation"].unique())
        for u in units:
            unit_in = set(
                cat_df.loc[(cat_df["unit_name"] == u) & cat_df["in_vocab"], "occupation"]
            )
            in_all &= unit_in
        consistent[cat_name] = sorted(in_all)
        logger.info(
            f"consistent-set {cat_name}: {len(consistent[cat_name])}/"
            f"{len(categories[cat_name])} occupations across {len(units)} units"
        )
    return consistent


def subsample_bands_from_lookup(
    value_lookup: Dict[Tuple[str, str, str], float],
    units: List[str],
    consistent_sets: Dict[str, List[str]],
    fraction: float,
    n_rounds: int,
    ci: float,
    seed: int,
    statistic: Callable[[np.ndarray], float] = _mean,
) -> Dict[Tuple[str, str], Tuple[float, float, float]]:
    """Word-subsample robustness band for an arbitrary per-round ``statistic``.

    Each round keeps ``fraction`` of a category's consistent set (without
    replacement); the SAME subset is held across every unit so the band
    isolates word-choice sensitivity. ``value_lookup`` maps
    (unit, category, occupation) -> value (in-vocab only). Returns
    dict[(unit, category)] -> (low, high, mean) where low/high are the ``ci``
    percentile interval of the n_rounds round-statistics and mean is their mean.
    """
    rng = np.random.default_rng(seed)
    alpha = (1.0 - ci) / 2.0
    out: Dict[Tuple[str, str], Tuple[float, float, float]] = {}

    for cat_name, consistent in consistent_sets.items():
        consistent = list(consistent)
        n = len(consistent)
        if n == 0:
            for u in units:
                out[(u, cat_name)] = (float("nan"), float("nan"), float("nan"))
            continue
        k = max(1, int(round(fraction * n)))

        round_stats: Dict[str, List[float]] = {u: [] for u in units}
        for _ in range(n_rounds):
            subset = consistent if k >= n else rng.choice(consistent, size=k, replace=False)
            for u in units:
                vals = [
                    value_lookup[(u, cat_name, w)]
                    for w in subset
                    if (u, cat_name, w) in value_lookup
                ]
                round_stats[u].append(
                    float(statistic(np.asarray(vals, dtype=float))) if vals else np.nan
                )

        for u in units:
            arr = np.asarray(round_stats[u], dtype=float)
            if np.isnan(arr).all():
                out[(u, cat_name)] = (float("nan"), float("nan"), float("nan"))
                continue
            lo = float(np.nanpercentile(arr, 100.0 * alpha))
            hi = float(np.nanpercentile(arr, 100.0 * (1.0 - alpha)))
            out[(u, cat_name)] = (lo, hi, float(np.nanmean(arr)))
    return out


def build_summary(
    long_df: pd.DataFrame,
    units: List[str],
    consistent_sets: Dict[str, List[str]],
    logger,
    value_col: str = "value",
    boot_n_iter: int = 5000,
    boot_ci: float = 0.68,
    sub_fraction: float = 0.8,
    sub_rounds: int = 100,
    sub_ci: float = 0.95,
    seed: int = 42,
    legacy_rnd_aliases: bool = False,
) -> pd.DataFrame:
    """Per-(unit, category) summary carrying TWO statistics, each with TWO bands.

    Statistics: ``mean_value`` (category mean of the metric) and ``prop_male``
    (share of occupations with value < 0). Each carries a with-replacement
    bootstrap CI and a word-subsample band.
    """
    in_vocab = long_df[long_df["in_vocab"]]
    value_lookup: Dict[Tuple[str, str, str], float] = {
        (r.unit_name, r.category, r.occupation): float(getattr(r, value_col))
        for r in in_vocab.itertuples(index=False)
    }
    mean_bands = subsample_bands_from_lookup(
        value_lookup, units, consistent_sets,
        fraction=sub_fraction, n_rounds=sub_rounds, ci=sub_ci, seed=seed,
        statistic=_mean,
    )
    prop_bands = subsample_bands_from_lookup(
        value_lookup, units, consistent_sets,
        fraction=sub_fraction, n_rounds=sub_rounds, ci=sub_ci, seed=seed,
        statistic=_prop_male,
    )

    rows: List[dict] = []
    for u in units:
        unit_long = long_df[long_df["unit_name"] == u]
        for cat_name, consistent in consistent_sets.items():
            m_lo, m_hi, m_mean = mean_bands.get((u, cat_name), (np.nan, np.nan, np.nan))
            p_lo, p_hi, p_mean = prop_bands.get((u, cat_name), (np.nan, np.nan, np.nan))
            sub = unit_long[
                (unit_long["category"] == cat_name)
                & unit_long["occupation"].isin(consistent)
                & unit_long["in_vocab"]
            ]
            if sub.empty:
                rows.append({
                    "unit_name": u, "category": cat_name,
                    "mean_value": np.nan, "mean_ci_low": np.nan, "mean_ci_high": np.nan,
                    "mean_sub_low": m_lo, "mean_sub_high": m_hi, "mean_sub_mean": m_mean,
                    "prop_male": np.nan, "prop_ci_low": np.nan, "prop_ci_high": np.nan,
                    "prop_sub_low": p_lo, "prop_sub_high": p_hi, "prop_sub_mean": p_mean,
                    "n_occupations": 0, "n_consistent": len(consistent),
                })
                continue
            arr = sub[value_col].to_numpy(dtype=float)
            mean_ci_low, mean_ci_high = bootstrap_ci(
                arr, n_iter=boot_n_iter, ci=boot_ci, seed=seed
            )
            prop_ci_low, prop_ci_high = bootstrap_ci(
                arr, n_iter=boot_n_iter, ci=boot_ci, seed=seed,
                statistic=_prop_male,
            )
            rows.append({
                "unit_name": u, "category": cat_name,
                "mean_value": float(arr.mean()),
                "mean_ci_low": float(mean_ci_low), "mean_ci_high": float(mean_ci_high),
                "mean_sub_low": m_lo, "mean_sub_high": m_hi, "mean_sub_mean": m_mean,
                "prop_male": _prop_male(arr),
                "prop_ci_low": float(prop_ci_low), "prop_ci_high": float(prop_ci_high),
                "prop_sub_low": p_lo, "prop_sub_high": p_hi, "prop_sub_mean": p_mean,
                "n_occupations": int(arr.size),
                "n_consistent": len(consistent),
            })

    summary_df = pd.DataFrame(rows)

    if legacy_rnd_aliases and not summary_df.empty:
        summary_df["mean_rnd"] = summary_df["mean_value"]
        summary_df["ci_low"] = summary_df["mean_ci_low"]
        summary_df["ci_high"] = summary_df["mean_ci_high"]
        summary_df["sub_low"] = summary_df["mean_sub_low"]
        summary_df["sub_high"] = summary_df["mean_sub_high"]
        summary_df["sub_mean"] = summary_df["mean_sub_mean"]

    n_rows = len(summary_df)
    n_valid = int(summary_df["mean_value"].notna().sum()) if n_rows else 0
    if n_rows > 0 and n_valid == 0:
        logger.error(
            f"build_summary: ALL {n_rows} (unit, category) rows have "
            "mean_value=NaN — downstream plot will be empty. Likely causes: "
            "(a) consistent-set empty for every category, (b) gender "
            "representation couldn't be built. Scroll up for per-unit diagnostics."
        )
    return summary_df
