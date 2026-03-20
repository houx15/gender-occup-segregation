#!/usr/bin/env python3
"""
WEAT mode analysis: compute Cohen's d effect sizes for gender norm dimensions.

Implements the 5-step pipeline:
  Step 0: OOV check
  Step 1: Build gender axes per unit
  Step 2: Compute concept word projections onto gender axes
  Step 3: Check cross-unit comparability
  Step 4: Compute WEAT Cohen's d
  Step 5: Save results

Works for both provincial and longitudinal units.

Usage:
    python -m scripts.analyze_weat --config=config/config.yml
    python -m scripts.analyze_weat --config=config/config.yml --skip_oov
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import fire

from scripts.common.config_loader import (
    load_config, get_analysis_unit, get_model_name, get_wordlist_dir,
    _parse_model_template,
)
from scripts.common.embedding_utils import (
    load_model, get_word_vector, compute_centroid, construct_semantic_axis,
    compute_projection, compute_cohens_d, check_oov,
)
from scripts.common.logging_utils import setup_logging


# =============================================================================
# Data classes
# =============================================================================

@dataclass
class Wordlists:
    male: List[str] = field(default_factory=list)
    female: List[str] = field(default_factory=list)
    family: List[str] = field(default_factory=list)
    work: List[str] = field(default_factory=list)
    leadership: List[str] = field(default_factory=list)
    non_leadership: List[str] = field(default_factory=list)
    stem: List[str] = field(default_factory=list)
    non_stem: List[str] = field(default_factory=list)

    def get_all_concept_words(self) -> Dict[str, List[str]]:
        return {
            "family": self.family, "work": self.work,
            "leadership": self.leadership, "non_leadership": self.non_leadership,
            "stem": self.stem, "non_stem": self.non_stem,
        }

    def get_all_categories(self) -> Dict[str, List[str]]:
        return {"male": self.male, "female": self.female, **self.get_all_concept_words()}


# =============================================================================
# Wordlist loading
# =============================================================================

def load_weat_wordlists(config: dict) -> Wordlists:
    """Load WEAT wordlists from config-specified directory."""
    wl_dir = get_wordlist_dir(config)
    wl_config = config.get("wordlists", {})

    def load_json(filename):
        path = wl_dir / filename
        if not path.exists():
            print(f"  Warning: {path} not found")
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    gender = load_json(wl_config.get("weat_gender_file", "gender_words.json"))
    domestic = load_json(wl_config.get("weat_domestic_work_file", "domestic_work_words.json"))
    leadership = load_json(wl_config.get("weat_leadership_file", "leadership_words.json"))
    stem = load_json(wl_config.get("weat_stem_file", "stem_words.json"))

    wl = Wordlists(
        male=gender.get("male", []),
        female=gender.get("female", []),
        family=domestic.get("family", []),
        work=domestic.get("work", []),
        leadership=leadership.get("leadership", []),
        non_leadership=leadership.get("non_leadership", []),
        stem=stem.get("stem", []),
        non_stem=stem.get("non_stem", []),
    )
    print(f"Wordlists: male={len(wl.male)}, female={len(wl.female)}, "
          f"family={len(wl.family)}, work={len(wl.work)}, "
          f"leadership={len(wl.leadership)}, non_leadership={len(wl.non_leadership)}, "
          f"stem={len(wl.stem)}, non_stem={len(wl.non_stem)}")
    return wl


# =============================================================================
# Model discovery
# =============================================================================

def discover_units(config: dict) -> List[Tuple[str, Path]]:
    """Discover available model files and their unit names."""
    models_dir = Path(config["paths"]["models_dir"])
    prefix, suffix = _parse_model_template(config)

    # Glob for the correct extension from the template suffix
    ext = suffix.lstrip(".")  # e.g. "model" or "kv"
    units = []
    for model_file in sorted(models_dir.glob(f"*.{ext}")):
        name = model_file.name
        if name.startswith(prefix) and name.endswith(suffix):
            unit_name = name[len(prefix):-len(suffix)] if suffix else name[len(prefix):]
            units.append((unit_name, model_file))
    return units


# =============================================================================
# Step 0: OOV check
# =============================================================================

def run_oov_check(units, wordlists, logger):
    """Check OOV across all units."""
    logger.info("\n" + "=" * 70)
    logger.info("Step 0: OOV Check")
    logger.info("=" * 70)

    categories = wordlists.get_all_categories()
    coverage_rows = []
    word_found = {w: {"category": cat, "found_in": []}
                  for cat, words in categories.items() for w in words}

    for unit_name, model_path in units:
        model = load_model(str(model_path))
        for cat, words in categories.items():
            found, oov = check_oov(model, words)
            coverage_rows.append({
                "unit": unit_name, "category": cat,
                "total": len(words), "found": len(found),
                "coverage": len(found) / len(words) if words else 0,
            })
            for w in found:
                word_found[w]["found_in"].append(unit_name)
        del model

    coverage_df = pd.DataFrame(coverage_rows)

    # Summary
    for cat in categories:
        avg = coverage_df[coverage_df["category"] == cat]["coverage"].mean()
        logger.info(f"  {cat}: avg coverage {avg:.1%}")

    word_rows = [
        {"word": w, "category": info["category"],
         "n_units_found": len(info["found_in"]),
         "coverage": len(info["found_in"]) / len(units) if units else 0}
        for w, info in word_found.items()
    ]
    word_df = pd.DataFrame(word_rows)
    if not word_df.empty:
        word_df = word_df.sort_values(["category", "coverage"])

    return coverage_df, word_df


# =============================================================================
# Steps 1-4: Gender axes, projections, comparability, WEAT
# =============================================================================

def run_analysis(units, wordlists, logger, standardize=None):
    """Run steps 1-4 of the WEAT pipeline."""
    concept_words = wordlists.get_all_concept_words()

    # Step 1: Build gender axes
    logger.info("\n" + "=" * 70)
    logger.info("Step 1: Build gender axes")
    logger.info("=" * 70)

    gender_axes = {}
    for unit_name, model_path in units:
        model = load_model(str(model_path))
        axis, n_pos, n_neg = construct_semantic_axis(
            wordlists.female, wordlists.male, model
        )
        if axis is not None:
            gender_axes[unit_name] = {"axis": axis, "model_path": model_path,
                                       "n_female": n_pos, "n_male": n_neg}
            logger.info(f"  {unit_name}: male={n_neg}, female={n_pos}")
        del model

    logger.info(f"  Built {len(gender_axes)} gender axes")

    # Step 2: Compute projections
    logger.info("\n" + "=" * 70)
    logger.info("Step 2: Compute projections")
    logger.info("=" * 70)

    projection_rows = []
    for unit_name, info in gender_axes.items():
        model = load_model(str(info["model_path"]))
        for category, words in concept_words.items():
            for word in words:
                vec = get_word_vector(model, word)
                if vec is not None:
                    proj, cos = compute_projection(vec, info["axis"])
                    projection_rows.append({
                        "unit": unit_name, "word": word,
                        "category": category, "projection": proj, "cosine_sim": cos,
                    })
        del model

    logger.info(f"  {len(projection_rows)} projection records")
    proj_df = pd.DataFrame(projection_rows)

    # Step 3: Comparability check
    logger.info("\n" + "=" * 70)
    logger.info("Step 3: Comparability check")
    logger.info("=" * 70)

    stats_rows = []
    for unit_name in proj_df["unit"].unique():
        vals = proj_df[proj_df["unit"] == unit_name]["cosine_sim"].values
        stats_rows.append({
            "unit": unit_name, "mean": float(np.mean(vals)),
            "std": float(np.std(vals)), "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        })
    stats_df = pd.DataFrame(stats_rows)

    means = stats_df["mean"].values
    stds = stats_df["std"].values
    cv_means = np.std(means) / abs(np.mean(means)) if abs(np.mean(means)) > 1e-10 else 0
    cv_stds = np.std(stds) / np.mean(stds) if np.mean(stds) > 1e-10 else 0
    logger.info(f"  CV of means: {cv_means:.4f}, CV of stds: {cv_stds:.4f}")

    needs_std = cv_means > 0.3 or cv_stds > 0.3
    use_zscore = standardize if standardize is not None else needs_std

    if use_zscore:
        logger.info("  Applying z-score standardization")
        stats_map = {r["unit"]: r for _, r in stats_df.iterrows()}
        def zscore(row):
            s = stats_map.get(row["unit"])
            if s is None or s["std"] < 1e-10:
                return 0.0
            return (row["cosine_sim"] - s["mean"]) / s["std"]
        proj_df["projection_zscore"] = proj_df.apply(zscore, axis=1)

    # Step 4: WEAT Cohen's d
    logger.info("\n" + "=" * 70)
    logger.info("Step 4: Compute WEAT Cohen's d")
    logger.info("=" * 70)

    value_col = "projection_zscore" if use_zscore else "cosine_sim"
    dimensions = [
        ("work_family", "family", "work"),
        ("leadership", "non_leadership", "leadership"),
        ("stem", "non_stem", "stem"),
    ]

    weat_rows = []
    for unit_name in proj_df["unit"].unique():
        udf = proj_df[proj_df["unit"] == unit_name]
        for dim_name, g1_cat, g2_cat in dimensions:
            g1 = udf[udf["category"] == g1_cat][value_col].values
            g2 = udf[udf["category"] == g2_cat][value_col].values
            if len(g1) == 0 or len(g2) == 0:
                continue
            d, ps = compute_cohens_d(g1, g2)
            weat_rows.append({
                "unit": unit_name, "dimension": dim_name,
                "cohens_d": d, "group1_mean": float(np.mean(g1)),
                "group2_mean": float(np.mean(g2)),
                "group1_std": float(np.std(g1, ddof=1)) if len(g1) > 1 else 0.0,
                "group2_std": float(np.std(g2, ddof=1)) if len(g2) > 1 else 0.0,
                "group1_n": len(g1), "group2_n": len(g2),
                "pooled_std": ps,
            })

    weat_df = pd.DataFrame(weat_rows)

    for dim_name, _, _ in dimensions:
        dim_d = weat_df[weat_df["dimension"] == dim_name]
        if len(dim_d) > 0:
            logger.info(f"  {dim_name}: mean d={dim_d['cohens_d'].mean():.3f} "
                       f"(SD={dim_d['cohens_d'].std():.3f})")

    return gender_axes, proj_df, stats_df, weat_df, use_zscore


# =============================================================================
# Step 5: Save results
# =============================================================================

def save_results(config, coverage_df, word_df, gender_axes, proj_df,
                 stats_df, weat_df, use_zscore, logger):
    """Save all analysis results."""
    logger.info("\n" + "=" * 70)
    logger.info("Step 5: Save results")
    logger.info("=" * 70)

    results_dir = Path(config["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    if coverage_df is not None:
        coverage_df.to_csv(results_dir / "oov_unit_coverage.csv", index=False, encoding="utf-8-sig")
    if word_df is not None:
        word_df.to_csv(results_dir / "oov_word_coverage.csv", index=False, encoding="utf-8-sig")

    # Gender axes info
    axes_data = [
        {"unit": u, "n_male": info["n_male"], "n_female": info["n_female"]}
        for u, info in gender_axes.items()
    ]
    pd.DataFrame(axes_data).to_csv(results_dir / "gender_axes.csv", index=False, encoding="utf-8-sig")

    proj_df.to_csv(results_dir / "word_projections.csv", index=False, encoding="utf-8-sig")
    stats_df.to_csv(results_dir / "unit_projection_stats.csv", index=False, encoding="utf-8-sig")
    weat_df.to_csv(results_dir / "weat_results.csv", index=False, encoding="utf-8-sig")

    # Wide format: gender norm index
    if not weat_df.empty:
        wide = weat_df.pivot_table(
            index="unit", columns="dimension",
            values=["cohens_d", "group1_n", "group2_n"], aggfunc="first",
        )
        wide.columns = [f"{col[1]}_{col[0]}" for col in wide.columns]
        wide = wide.reset_index()
        wide.to_csv(results_dir / "gender_norm_index.csv", index=False, encoding="utf-8-sig")

    logger.info(f"  Results saved to {results_dir}")


# =============================================================================
# Main
# =============================================================================

def main(config="config/config.yml", standardize=None, skip_oov=False):
    """
    Run WEAT analysis on trained embedding models.

    Args:
        config: Path to configuration file
        standardize: Force standardization (True/False/None=auto)
        skip_oov: Skip OOV check step
    """
    config_data = load_config(config)
    logger = setup_logging(Path(config_data["paths"]["log_dir"]), "analyze_weat.log")

    logger.info("=" * 80)
    logger.info("Starting WEAT analysis")
    logger.info("=" * 80)

    wordlists = load_weat_wordlists(config_data)
    units = discover_units(config_data)

    if not units:
        logger.error("No models found")
        return

    logger.info(f"Found {len(units)} model units")

    # Step 0
    coverage_df = word_df = None
    if not skip_oov:
        coverage_df, word_df = run_oov_check(units, wordlists, logger)

    # Steps 1-4
    gender_axes, proj_df, stats_df, weat_df, use_zscore = run_analysis(
        units, wordlists, logger, standardize=standardize
    )

    # Step 5
    save_results(config_data, coverage_df, word_df, gender_axes,
                 proj_df, stats_df, weat_df, use_zscore, logger)

    logger.info("=" * 80)
    logger.info("WEAT analysis completed!")
    logger.info("=" * 80)


if __name__ == "__main__":
    fire.Fire(main)
