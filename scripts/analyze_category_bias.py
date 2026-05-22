#!/usr/bin/env python3
"""Orchestrator: compute one or more single-wordlist gender-bias metrics.

Driven by config ``analysis.metrics`` (a list of {"rnd", "cohens_d"}). Each
model is loaded ONCE; every listed metric's per-word producer runs on it, then
the shared summary (mean + male-leaned proportion, each with bootstrap and
subsample bands) is built and written per metric.

Usage:
  python -m scripts.analyze_category_bias --config=config/profiles/garg_weat_coha_trained.yml
  python -m scripts.analyze_category_bias --config=... --metrics=rnd,cohens_d
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import fire

from scripts.common.config_loader import load_config, get_wordlist_dir
from scripts.common.logging_utils import setup_logging
from scripts.common.category_summary import (
    load_categories, compute_consistent_set, build_summary,
)
from scripts.analyze_garg import (
    discover_models, load_model_for_unit, decade_to_census_year, load_gender_words,
)
from scripts.analyze_garg_weat import rnd_values
from scripts.analyze_cohens_d_singlelist import projection_values

# metric -> (producer, long_stem, summary_stem, long_value_name, legacy_aliases)
METRIC_SPECS = {
    "rnd": (rnd_values, "garg_weat_rnd_long",
            "garg_weat_summary_by_category", "rnd", True),
    "cohens_d": (projection_values, "cohens_d_singlelist_long",
                 "cohens_d_singlelist_summary_by_category", "projection", False),
}


def _resolve_metrics(config_data: dict, override: Optional[List[str]]) -> List[str]:
    if override is not None:
        metrics = override
    else:
        metrics = config_data.get("analysis", {}).get("metrics")
    if not metrics:
        raise ValueError(
            "config analysis.metrics is required (a non-empty list of "
            f"{sorted(METRIC_SPECS)}); none found. Prefer an explicit list."
        )
    unknown = [m for m in metrics if m not in METRIC_SPECS]
    if unknown:
        raise ValueError(
            f"Unknown metric(s) {unknown}; valid: {sorted(METRIC_SPECS)}"
        )
    return list(metrics)


def _filter_models(models, unit, decade_range, logger):
    if unit:
        models = [(p, n) for p, n in models if n.startswith(str(unit))]
    if decade_range:
        try:
            start, end = int(decade_range[0]), int(decade_range[1])
        except (TypeError, ValueError, IndexError):
            raise ValueError(
                f"analysis.decade_range must be [start, end] integers, got {decade_range!r}"
            )
        kept = []
        for path, unit_name in models:
            year = decade_to_census_year(unit_name)
            if year is None or start <= year <= end:
                kept.append((path, unit_name))
        logger.info(f"decade_range [{start}, {end}]: {len(models)} -> {len(kept)} models")
        models = kept
    return models


def run(config_data: dict, metrics: Optional[List[str]], unit: Optional[str] = None) -> None:
    logger = setup_logging(Path(config_data["paths"]["log_dir"]), "analyze_category_bias.log")
    metrics = _resolve_metrics(config_data, metrics)
    logger.info("=" * 80)
    logger.info(f"Single-wordlist bias analysis — metrics={metrics}")
    logger.info("=" * 80)

    wl_dir = get_wordlist_dir(config_data)
    wl_cfg = config_data.get("wordlists", {})
    gender_words = load_gender_words(
        wl_dir / wl_cfg.get("gender_words_file", "gender_words.json"), logger
    )
    categories = load_categories(config_data, logger)

    models = discover_models(config_data)
    decade_range = config_data.get("analysis", {}).get("decade_range")
    models = _filter_models(models, unit, decade_range, logger)
    if not models:
        logger.error("No models found after filtering — nothing written")
        return
    logger.info(f"Found {len(models)} models")

    collected: Dict[str, Tuple[List[pd.DataFrame], List[str]]] = {
        m: ([], []) for m in metrics
    }
    for model_path, unit_name in models:
        model = load_model_for_unit(model_path, config_data)  # loaded ONCE
        for m in metrics:
            producer = METRIC_SPECS[m][0]
            long_df = producer(model, unit_name, categories, gender_words, logger)
            if long_df is None:
                continue
            collected[m][0].append(long_df)
            collected[m][1].append(unit_name)

    analysis_cfg = config_data.get("analysis", {})
    boot = analysis_cfg.get("bootstrap", {})
    sub = analysis_cfg.get("subsample", {})
    seed = int(analysis_cfg.get("seed", 42))
    results_dir = Path(config_data["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    for m in metrics:
        frames, units = collected[m]
        _, long_stem, summary_stem, value_name, legacy = METRIC_SPECS[m]
        if not frames:
            logger.error(f"[{m}] No units produced results — skipping outputs")
            continue
        long_combined = pd.concat(frames, ignore_index=True)
        consistent = compute_consistent_set(long_combined, categories, units, logger)
        summary = build_summary(
            long_combined, units, consistent, logger,
            value_col="value",
            boot_n_iter=int(boot.get("n_iter", 5000)),
            boot_ci=float(boot.get("ci", 0.68)),
            sub_fraction=float(sub.get("fraction", 0.8)),
            sub_rounds=int(sub.get("n_rounds", 100)),
            sub_ci=float(sub.get("ci", 0.95)),
            seed=seed,
            legacy_rnd_aliases=legacy,
        )
        long_out = long_combined.rename(columns={"value": value_name})
        long_path = results_dir / f"{long_stem}.parquet"
        summary_path = results_dir / f"{summary_stem}.parquet"
        long_out.to_parquet(long_path, index=False)
        summary.to_parquet(summary_path, index=False)
        logger.info(f"[{m}] Saved: {long_path}")
        logger.info(f"[{m}] Saved: {summary_path}")

    logger.info("=" * 80)
    logger.info("Single-wordlist bias analysis completed!")
    logger.info("=" * 80)


def main(config: str = "config/config.yml", unit: Optional[str] = None,
         metrics: Optional[str] = None) -> None:
    """CLI. ``metrics`` (comma-separated) overrides config analysis.metrics."""
    config_data = load_config(config)
    override = None
    if metrics is not None:
        override = [m.strip() for m in str(metrics).split(",") if m.strip()]
    run(config_data, metrics=override, unit=unit)


if __name__ == "__main__":
    fire.Fire(main)
