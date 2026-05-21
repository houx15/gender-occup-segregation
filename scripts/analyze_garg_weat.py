#!/usr/bin/env python3
"""
Garg-WEAT mode: per-category RND analysis.

Same metric as analyze_garg (relative norm distance against male/female
centroids, Garg sign convention `||v - c_male|| - ||v - c_female||`,
L2-normalized vectors), but the occupations are partitioned into named
buckets (leadership / family / science) and the per-decade mean is
computed separately per bucket. The output is one trend line per
category, all sharing the same gender axis and decade scaffolding.

Composes existing helpers from analyze_garg: discover_models,
load_model_for_unit, decade_to_census_year, load_gender_words. The
inner per-unit loop is reimplemented here because we need to load the
model once and iterate categories, rather than analyze_garg's
single-occupation-list contract.

Outputs:
  garg_weat_rnd_long.parquet
    columns: unit_name, category, occupation, rnd, in_vocab
  garg_weat_summary_by_category.parquet
    columns: unit_name, category, mean_rnd, ci_low, ci_high,
             n_occupations, n_consistent

Usage:
  python -m scripts.analyze_garg_weat --config=config/profiles/garg_weat_coha_trained.yml
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import fire

from scripts.common.config_loader import load_config, get_wordlist_dir
from scripts.common.logging_utils import setup_logging
from scripts.common.metrics import (
    relative_norm_distance, bootstrap_ci, l2_normalize,
)
from scripts.analyze_garg import (
    discover_models, load_model_for_unit, decade_to_census_year,
    load_gender_words,
)


def load_categories(config: dict, logger) -> Dict[str, List[str]]:
    """Read category occupation files from config['wordlists']['categories'].

    Config structure expected:
      wordlists:
        dir: wordlists/en/garg_weat
        gender_words_file: gender_words.json
        categories:
          leadership: cleaned_leadership.txt
          family:     cleaned_family.txt
          science:    cleaned_science.txt
    """
    wl_dir = get_wordlist_dir(config)
    cats_cfg = config.get("wordlists", {}).get("categories")
    if not cats_cfg:
        raise ValueError(
            "garg_weat config must define wordlists.categories "
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


def analyze_unit(
    model_path: Path,
    unit_name: str,
    categories: Dict[str, List[str]],
    gender_words: dict,
    logger,
    config: dict,
) -> Optional[pd.DataFrame]:
    """Per-unit RND across all categories. Loads model + builds centroids once,
    then iterates categories.

    Returns long_df with columns (unit_name, category, occupation, rnd,
    in_vocab), or None if gender centroids are unobtainable.
    """
    model = load_model_for_unit(model_path, config)

    try:
        vocab_size = len(model.key_to_index)
        logger.info(f"  {unit_name}: vocab_size={vocab_size}")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"  {unit_name}: could not introspect vocab ({e!r})")

    # L2-normalized centroids — match analyze_garg's normalization regime.
    male_vecs = [l2_normalize(model[w]) for w in gender_words["male"] if w in model.key_to_index]
    female_vecs = [l2_normalize(model[w]) for w in gender_words["female"] if w in model.key_to_index]
    male_found = [w for w in gender_words["male"] if w in model.key_to_index]
    female_found = [w for w in gender_words["female"] if w in model.key_to_index]

    logger.info(
        f"  {unit_name}: gender words — "
        f"male {len(male_found)}/{len(gender_words['male'])} found, "
        f"female {len(female_found)}/{len(gender_words['female'])} found"
    )

    if not male_vecs or not female_vecs:
        logger.warning(
            f"  {unit_name}: skipping — gender centroids unobtainable "
            f"(male={len(male_found)}, female={len(female_found)})"
        )
        return None

    c_male = np.mean(np.asarray(male_vecs), axis=0)
    c_female = np.mean(np.asarray(female_vecs), axis=0)

    long_rows: List[dict] = []
    for cat_name, words in categories.items():
        n_in = 0
        for w in words:
            if w in model.key_to_index:
                vec = l2_normalize(model[w])
                rnd = relative_norm_distance(vec, c_male, c_female)
                long_rows.append({
                    "unit_name": unit_name,
                    "category": cat_name,
                    "occupation": w,
                    "rnd": float(rnd),
                    "in_vocab": True,
                })
                n_in += 1
            else:
                long_rows.append({
                    "unit_name": unit_name,
                    "category": cat_name,
                    "occupation": w,
                    "rnd": np.nan,
                    "in_vocab": False,
                })
        logger.info(f"    {cat_name}: {n_in}/{len(words)} in vocab")

    return pd.DataFrame(long_rows)


def compute_consistent_set(
    long_df: pd.DataFrame,
    categories: Dict[str, List[str]],
    units: List[str],
    logger,
) -> Dict[str, List[str]]:
    """Per category, find occupations in vocab in ALL units. Mirrors
    analyze_garg.compute_consistent_set but partitioned by category.
    """
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


def build_summary(
    long_df: pd.DataFrame,
    units: List[str],
    consistent_sets: Dict[str, List[str]],
    logger,
) -> pd.DataFrame:
    """Per-(unit, category) summary row using each category's consistent set."""
    rows: List[dict] = []
    for u in units:
        unit_long = long_df[long_df["unit_name"] == u]
        for cat_name, consistent in consistent_sets.items():
            sub = unit_long[
                (unit_long["category"] == cat_name)
                & unit_long["occupation"].isin(consistent)
                & unit_long["in_vocab"]
            ]
            if sub.empty:
                rows.append({
                    "unit_name": u, "category": cat_name,
                    "mean_rnd": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                    "n_occupations": 0, "n_consistent": len(consistent),
                })
                continue
            rnd_arr = sub["rnd"].to_numpy(dtype=float)
            mean_rnd = float(rnd_arr.mean())
            ci_low, ci_high = bootstrap_ci(rnd_arr, n_iter=1000, ci=0.95, seed=42)
            rows.append({
                "unit_name": u, "category": cat_name,
                "mean_rnd": mean_rnd,
                "ci_low": float(ci_low), "ci_high": float(ci_high),
                "n_occupations": int(rnd_arr.size),
                "n_consistent": len(consistent),
            })

    summary_df = pd.DataFrame(
        rows,
        columns=[
            "unit_name", "category", "mean_rnd", "ci_low", "ci_high",
            "n_occupations", "n_consistent",
        ],
    )

    # Loud failure mode: if every row is NaN, the downstream plot is blank.
    n_rows = len(summary_df)
    n_valid = int(summary_df["mean_rnd"].notna().sum())
    if n_rows > 0 and n_valid == 0:
        logger.error(
            f"build_summary: ALL {n_rows} (unit, category) rows have "
            "mean_rnd=NaN — downstream plot will be empty. Likely causes: "
            "(a) consistent-set is empty for every category, (b) gender "
            "centroids couldn't be built. Scroll up for per-unit diagnostics."
        )
    return summary_df


def main(config: str = "config/config.yml", unit: Optional[str] = None) -> None:
    """Run Garg-WEAT per-category RND analysis.

    Args:
        config: path to config YAML
        unit: optional unit-name prefix filter (passed through like analyze_garg)
    """
    config_data = load_config(config)
    logger = setup_logging(
        Path(config_data["paths"]["log_dir"]), "analyze_garg_weat.log"
    )

    logger.info("=" * 80)
    logger.info("Starting Garg-WEAT (per-category RND) analysis")
    logger.info("=" * 80)

    wl_dir = get_wordlist_dir(config_data)
    wl_config = config_data.get("wordlists", {})
    gender_words = load_gender_words(
        wl_dir / wl_config.get("gender_words_file", "gender_words.json"), logger
    )
    categories = load_categories(config_data, logger)

    models = discover_models(config_data)
    if unit:
        models = [(p, n) for p, n in models if n.startswith(str(unit))]

    # decade_range clip — mirrors analyze_garg behaviour.
    decade_range = config_data.get("analysis", {}).get("decade_range")
    if decade_range:
        try:
            start, end = int(decade_range[0]), int(decade_range[1])
        except (TypeError, ValueError, IndexError):
            raise ValueError(
                f"analysis.decade_range must be [start, end] integers, "
                f"got {decade_range!r}"
            )
        before = len(models)
        filtered: List[Tuple[Path, str]] = []
        for path, unit_name in models:
            year = decade_to_census_year(unit_name)
            if year is None:
                logger.warning(
                    f"  decade_range filter: cannot parse decade from "
                    f"unit_name {unit_name!r}; keeping it"
                )
                filtered.append((path, unit_name))
            elif start <= year <= end:
                filtered.append((path, unit_name))
        models = filtered
        logger.info(
            f"decade_range filter [{start}, {end}]: {before} -> {len(models)} models"
        )

    if not models:
        logger.error(
            f"No models found in {config_data['paths']['models_dir']}"
            + (f" matching unit prefix '{unit}'" if unit else "")
            + (f" within decade_range {decade_range}" if decade_range else "")
        )
        return

    logger.info(f"Found {len(models)} models")

    long_frames: List[pd.DataFrame] = []
    units: List[str] = []
    for model_path, unit_name in models:
        long_df = analyze_unit(
            model_path, unit_name, categories, gender_words, logger, config_data,
        )
        if long_df is None:
            continue
        long_frames.append(long_df)
        units.append(unit_name)

    if not long_frames:
        logger.error("No units produced results — nothing written")
        return

    long_combined = pd.concat(long_frames, ignore_index=True)
    consistent_sets = compute_consistent_set(long_combined, categories, units, logger)
    summary_df = build_summary(long_combined, units, consistent_sets, logger)

    results_dir = Path(config_data["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    long_path = results_dir / "garg_weat_rnd_long.parquet"
    summary_path = results_dir / "garg_weat_summary_by_category.parquet"
    long_combined.to_parquet(long_path, index=False)
    summary_df.to_parquet(summary_path, index=False)

    logger.info(f"Saved: {long_path}")
    logger.info(f"Saved: {summary_path}")
    logger.info("=" * 80)
    logger.info("Garg-WEAT analysis completed!")
    logger.info("=" * 80)


if __name__ == "__main__":
    fire.Fire(main)
