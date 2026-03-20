#!/usr/bin/env python3
"""
Prestige mode analysis: compute gender typing and prestige scores for occupations.

Uses 4 prestige dimension projections + Pearson correlations.
Works for both longitudinal (time slices) and provincial units.

Usage:
    python -m scripts.analyze_prestige --config=config/config.yml
    python -m scripts.analyze_prestige --config=config/config.yml --unit=1940_1949
"""

import json
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import fire

from scripts.common.config_loader import (
    load_config, get_analysis_unit, get_model_name, get_wordlist_dir
)
from scripts.common.embedding_utils import (
    load_model, get_word_vector, construct_semantic_axis
)
from scripts.common.logging_utils import setup_logging


def load_occupations(file_path: Path, logger) -> List[str]:
    """Load occupation list from file."""
    logger.info(f"Loading occupations from {file_path}")
    occupations = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            occ = line.strip()
            if occ:
                occupations.append(occ)
    logger.info(f"  Loaded {len(occupations)} occupations")
    return occupations


def analyze_model(
    model_path: Path, unit_name: str, occupations: List[str],
    gender_words: dict, prestige_axes: dict, logger,
) -> pd.DataFrame:
    """Analyze a single model for gender and prestige scores."""
    logger.info(f"\nAnalyzing {unit_name}...")
    model = load_model(str(model_path))

    # Gender axis (positive = female)
    gender_axis, n_female, n_male = construct_semantic_axis(
        gender_words["female_terms"], gender_words["male_terms"], model
    )
    if gender_axis is None:
        logger.warning(f"  Could not construct gender axis for {unit_name}")
    else:
        logger.info(f"  Gender axis: {n_male} male, {n_female} female terms found")

    # Prestige axes
    prestige_axis_vectors = {}
    for dimension, terms in prestige_axes.items():
        axis, n_pos, n_neg = construct_semantic_axis(
            terms["positive"], terms["negative"], model
        )
        if axis is not None:
            prestige_axis_vectors[dimension] = axis
            logger.info(f"  {dimension}: {n_pos} positive, {n_neg} negative")

    # Score occupations
    results = []
    for occ in occupations:
        occ_vec = get_word_vector(model, occ)
        if occ_vec is None:
            continue

        row = {"occupation": occ}
        row["gender_score"] = float(np.dot(occ_vec, gender_axis)) if gender_axis is not None else np.nan
        for dim, axis in prestige_axis_vectors.items():
            row[f"{dim}_score"] = float(np.dot(occ_vec, axis))
        results.append(row)

    logger.info(f"  Scored {len(results)} occupations")
    return pd.DataFrame(results)


def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute summary statistics and gender-prestige correlations per unit."""
    results = []
    for unit_name in df["unit_name"].unique():
        data = df[df["unit_name"] == unit_name]
        summary = {
            "unit_name": unit_name,
            "num_occupations": len(data),
            "mean_gender_score": data["gender_score"].mean(),
            "std_gender_score": data["gender_score"].std(),
        }
        dim_cols = [c for c in data.columns if c.endswith("_score") and c != "gender_score"]
        for col in dim_cols:
            dim = col.replace("_score", "")
            if data["gender_score"].notna().sum() > 1 and data[col].notna().sum() > 1:
                summary[f"corr_gender_{dim}"] = data["gender_score"].corr(data[col])
            else:
                summary[f"corr_gender_{dim}"] = np.nan
        results.append(summary)
    return pd.DataFrame(results)


def discover_models(config: dict) -> List[tuple]:
    """Discover all model files and their unit names."""
    models_dir = Path(config["paths"]["models_dir"])
    template = config.get("embedding", {}).get("model_name_template", "{unit_name}.model")

    # Extract prefix and suffix from template
    parts = template.split("{unit_name}")
    prefix = parts[0] if len(parts) > 0 else ""
    suffix = parts[1] if len(parts) > 1 else ".model"

    models = []
    for model_file in sorted(models_dir.glob("*.model")):
        name = model_file.name
        if name.startswith(prefix) and name.endswith(suffix):
            unit_name = name[len(prefix):-len(suffix)] if suffix else name[len(prefix):]
            models.append((model_file, unit_name))
    return models


def main(config="config/config.yml", unit=None):
    """
    Run prestige analysis on trained embedding models.

    Args:
        config: Path to configuration file
        unit: Analyze only a specific unit (time slice or province name)
    """
    config_data = load_config(config)
    logger = setup_logging(Path(config_data["paths"]["log_dir"]), "analyze_prestige.log")

    logger.info("=" * 80)
    logger.info("Starting prestige analysis")
    logger.info("=" * 80)

    # Load wordlists
    wl_dir = get_wordlist_dir(config_data)
    wl_config = config_data.get("wordlists", {})

    occupations = load_occupations(wl_dir / wl_config.get("occupations_file", "occupations_zh.txt"), logger)

    with open(wl_dir / wl_config.get("gender_words_file", "gender_words_zh.json"), "r", encoding="utf-8") as f:
        gender_words = json.load(f)

    with open(wl_dir / wl_config.get("prestige_axes_file", "prestige_axes_zh.json"), "r", encoding="utf-8") as f:
        prestige_axes = json.load(f)

    # Discover and filter models
    models = discover_models(config_data)
    if not models:
        logger.error(f"No models found in {config_data['paths']['models_dir']}")
        return

    if unit:
        models = [(p, n) for p, n in models if n.startswith(str(unit))]

    logger.info(f"Found {len(models)} models")

    # Analyze each model
    all_results = []
    analysis_unit = get_analysis_unit(config_data)

    for model_path, unit_name in models:
        df = analyze_model(model_path, unit_name, occupations, gender_words, prestige_axes, logger)
        df["unit_name"] = unit_name

        # For longitudinal mode, add year info
        if analysis_unit == "longitudinal":
            try:
                start_year, end_year = map(int, unit_name.split("_"))
                df["start_year"] = start_year
                df["end_year"] = end_year
                df["time_slice"] = unit_name
            except ValueError:
                pass

        all_results.append(df)

    if not all_results:
        logger.error("No results produced")
        return

    combined = pd.concat(all_results, ignore_index=True)

    # Save results
    results_dir = Path(config_data["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    if analysis_unit == "longitudinal":
        output_file = results_dir / "occupation_scores_by_slice.parquet"
    else:
        output_file = results_dir / "occupation_scores_by_province.parquet"

    combined.to_parquet(output_file, index=False)
    logger.info(f"Saved: {output_file}")

    # Summary statistics
    summary = compute_summary(combined)
    summary_file = results_dir / "summary_statistics.parquet"
    summary.to_parquet(summary_file, index=False)
    logger.info(f"Saved: {summary_file}")

    logger.info("=" * 80)
    logger.info("Prestige analysis completed!")
    logger.info("=" * 80)


if __name__ == "__main__":
    fire.Fire(main)
