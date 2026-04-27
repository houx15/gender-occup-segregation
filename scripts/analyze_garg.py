#!/usr/bin/env python3
"""
Garg-mode analysis: compute relative norm distance (RND) between occupation
vectors and gender centroids, per time/unit.

Implements Garg et al. (2018) "Word embeddings quantify 100 years of gender
and ethnic stereotypes". Per unit:
  RND(occ) = ||v_occ - c_female|| - ||v_occ - c_male||
  mean_rnd over in-vocab occupations + 95% bootstrap CI

Outputs:
  garg_relative_norm_by_decade.parquet
    columns: unit_name, occupation, rnd, in_vocab
  garg_average_bias_by_decade.parquet
    columns: unit_name, mean_rnd, ci_low, ci_high, n_occupations

Usage:
    python -m scripts.analyze_garg --config=config/config.yml
    python -m scripts.analyze_garg --config=config/config.yml --unit=1990s
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import fire

from scripts.common.config_loader import (
    load_config, get_wordlist_dir, _parse_model_template,
)
from scripts.common.embedding_utils import (
    load_model, compute_centroid, check_oov,
)
from scripts.common.logging_utils import setup_logging
from scripts.common.metrics import relative_norm_distance, bootstrap_ci


def discover_models(config: dict) -> List[Tuple[Path, str]]:
    """
    Discover all model files and their unit names.

    Mirrors `scripts.analyze_prestige.discover_models` to keep blast radius
    small (per WI-3 plan: do not hoist a shared helper).
    """
    models_dir = Path(config["paths"]["models_dir"])
    prefix, suffix = _parse_model_template(config)

    ext = suffix.lstrip(".")
    models: List[Tuple[Path, str]] = []
    for model_file in sorted(models_dir.glob(f"*.{ext}")):
        name = model_file.name
        if name.startswith(prefix) and name.endswith(suffix):
            unit_name = name[len(prefix):-len(suffix)] if suffix else name[len(prefix):]
            models.append((model_file, unit_name))
    return models


def load_occupations(file_path: Path, logger) -> List[str]:
    """Load occupation list (one per line, blanks dropped)."""
    logger.info(f"Loading occupations from {file_path}")
    occupations: List[str] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            occ = line.strip()
            if occ:
                occupations.append(occ)
    logger.info(f"  Loaded {len(occupations)} occupations")
    return occupations


def load_gender_words(file_path: Path, logger) -> dict:
    """Load gender word JSON with 'male' and 'female' lists."""
    logger.info(f"Loading gender words from {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "male" not in data or "female" not in data:
        raise ValueError(
            f"gender_words file {file_path} must contain 'male' and 'female' keys"
        )
    logger.info(
        f"  Loaded male={len(data['male'])}, female={len(data['female'])}"
    )
    return data


def analyze_unit(
    model_path: Path,
    unit_name: str,
    occupations: List[str],
    gender_words: dict,
    logger,
) -> Optional[Tuple[pd.DataFrame, dict]]:
    """
    Compute RND for every occupation in a single unit.

    Returns:
        (long_df, summary_row) where
          long_df has columns: unit_name, occupation, rnd, in_vocab
          summary_row has keys: unit_name, mean_rnd, ci_low, ci_high, n_occupations
        or None if both gender centroids are unobtainable (unit is skipped).
    """
    model = load_model(str(model_path))

    c_male, male_found = compute_centroid(model, gender_words["male"])
    c_female, female_found = compute_centroid(model, gender_words["female"])

    if c_male is None or c_female is None:
        logger.warning(
            f"  {unit_name}: skipping — could not compute gender centroids "
            f"(male_found={len(male_found)}, female_found={len(female_found)})"
        )
        return None

    long_rows = []
    in_vocab_rnds: List[float] = []

    for occ in occupations:
        found, _oov = check_oov(model, [occ])
        if found:
            rnd = relative_norm_distance(model[occ], c_male, c_female)
            long_rows.append({
                "unit_name": unit_name,
                "occupation": occ,
                "rnd": float(rnd),
                "in_vocab": True,
            })
            in_vocab_rnds.append(float(rnd))
        else:
            long_rows.append({
                "unit_name": unit_name,
                "occupation": occ,
                "rnd": np.nan,
                "in_vocab": False,
            })

    long_df = pd.DataFrame(long_rows)

    n_in_vocab = len(in_vocab_rnds)
    n_total = len(occupations)
    pct = (n_in_vocab / n_total) if n_total else 0.0

    if n_in_vocab == 0:
        logger.warning(
            f"  {unit_name}: coverage=0/{n_total} (0.0%) — no in-vocab occupations"
        )
        summary = {
            "unit_name": unit_name,
            "mean_rnd": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "n_occupations": 0,
        }
    else:
        rnd_arr = np.asarray(in_vocab_rnds, dtype=float)
        mean_rnd = float(rnd_arr.mean())
        ci_low, ci_high = bootstrap_ci(rnd_arr, n_iter=1000, ci=0.95, seed=42)
        logger.info(
            f"{unit_name}: coverage={n_in_vocab}/{n_total} ({pct:.1%}), "
            f"mean_rnd={mean_rnd:.4f}"
        )
        if pct < 0.5:
            logger.warning(
                f"  {unit_name}: low coverage {pct:.1%} (< 50%) — "
                f"mean_rnd may be unreliable"
            )
        summary = {
            "unit_name": unit_name,
            "mean_rnd": mean_rnd,
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            "n_occupations": n_in_vocab,
        }

    return long_df, summary


def main(config: str = "config/config.yml", unit: Optional[str] = None) -> None:
    """
    Run Garg-style RND analysis on trained embedding models.

    Args:
        config: Path to configuration file
        unit: If set, only analyze units whose name starts with this string
    """
    config_data = load_config(config)
    logger = setup_logging(
        Path(config_data["paths"]["log_dir"]), "analyze_garg.log"
    )

    logger.info("=" * 80)
    logger.info("Starting Garg (RND) analysis")
    logger.info("=" * 80)

    wl_dir = get_wordlist_dir(config_data)
    wl_config = config_data.get("wordlists", {})

    occupations = load_occupations(
        wl_dir / wl_config.get("occupations_file", "occupations.txt"), logger
    )
    gender_words = load_gender_words(
        wl_dir / wl_config.get("gender_words_file", "gender_words.json"), logger
    )

    models = discover_models(config_data)
    if unit:
        models = [(p, n) for p, n in models if n.startswith(str(unit))]

    if not models:
        logger.error(
            f"No models found in {config_data['paths']['models_dir']}"
            + (f" matching unit prefix '{unit}'" if unit else "")
        )
        return

    logger.info(f"Found {len(models)} models")

    long_frames: List[pd.DataFrame] = []
    summary_rows: List[dict] = []

    for model_path, unit_name in models:
        result = analyze_unit(
            model_path, unit_name, occupations, gender_words, logger
        )
        if result is None:
            continue
        long_df, summary = result
        long_frames.append(long_df)
        summary_rows.append(summary)

    if not long_frames:
        logger.error("No units produced results — nothing written")
        return

    long_combined = pd.concat(long_frames, ignore_index=True)
    summary_df = pd.DataFrame(
        summary_rows,
        columns=["unit_name", "mean_rnd", "ci_low", "ci_high", "n_occupations"],
    )

    results_dir = Path(config_data["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    long_path = results_dir / "garg_relative_norm_by_decade.parquet"
    summary_path = results_dir / "garg_average_bias_by_decade.parquet"

    long_combined.to_parquet(long_path, index=False)
    summary_df.to_parquet(summary_path, index=False)

    logger.info(f"Saved: {long_path}")
    logger.info(f"Saved: {summary_path}")
    logger.info("=" * 80)
    logger.info("Garg analysis completed!")
    logger.info("=" * 80)


if __name__ == "__main__":
    fire.Fire(main)
