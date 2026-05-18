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

Works for both provincial and longitudinal units. Embedding format is
dispatched via ``embedding.format`` (default ``gensim_kv`` = our trained
models; ``histwords`` = Hamilton et al. 2016 .npy + vocab.pkl pairs).

Usage:
    python -m scripts.analyze_weat --config=config/config.yml
    python -m scripts.analyze_weat --config=config/config.yml --skip_oov
"""

import json
import os
import re
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
    load_model as _load_gensim_kv, get_word_vector, compute_centroid,
    construct_semantic_axis, compute_projection, compute_cohens_d, check_oov,
)

# Module-level alias preserved so callers / tests can monkeypatch
# ``analyze_weat.load_model`` to inject a stub KeyedVectors. The
# format-aware dispatch (``load_model_for_unit``) only falls back here for
# the default gensim_kv path.
load_model = _load_gensim_kv
from scripts.common.logging_utils import setup_logging


_DECADE_RE = re.compile(r"^(\d{4})s$")


def _embedding_format(config: dict) -> str:
    """Return the embedding format key. Defaults to 'gensim_kv' (our trained
    models). 'histwords' selects the Hamilton et al. 2016 .npy + vocab.pkl
    layout used by the published HistWords COHA SGNS / SVD vectors.
    """
    return config.get("embedding", {}).get("format", "gensim_kv")


def _decade_to_year(unit_name: str) -> Optional[int]:
    """'1990s' -> 1990. Returns None if the unit name doesn't match."""
    m = _DECADE_RE.match(unit_name)
    if not m:
        return None
    return int(m.group(1))


def load_model_for_unit(model_path: Path, config: dict):
    """Format-aware loader for a single unit's model file.

    Wraps the legacy ``embedding_utils.load_model`` for trained gensim
    models and adds a HistWords path for pre-trained .npy + vocab.pkl
    pairs. The default (gensim_kv) path goes through this module's
    ``load_model`` alias so tests can still monkeypatch
    ``analyze_weat.load_model``.
    """
    fmt = _embedding_format(config)
    if fmt == "histwords":
        from scripts.common.embedding_loaders import load_histwords_decade
        return load_histwords_decade(Path(model_path))
    return load_model(str(model_path))


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
    """Discover available model files and their unit names.

    Dispatches on ``embedding.format``:
      - "gensim_kv" (default): scans ``models_dir`` for files matching the
        ``model_name_template`` (legacy behavior).
      - "histwords": scans ``models_dir`` for ``{YYYY}-w.npy`` pairs and
        yields ``(unit_name="YYYYs", npy_path)``.
    """
    fmt = _embedding_format(config)
    models_dir = Path(config["paths"]["models_dir"])

    if fmt == "histwords":
        # Local import to avoid a hard gensim dependency at module-load time
        # for non-histwords paths (esp. in test envs with stubbed gensim).
        from scripts.common.embedding_loaders import discover_histwords_decades
        # discover_histwords_decades yields (npy_path, unit_name); flip to
        # match this module's historical (unit_name, path) order so
        # downstream callers don't need to care which format ran.
        return [(unit_name, path) for path, unit_name in discover_histwords_decades(models_dir)]

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

def run_oov_check(units, wordlists, logger, config=None):
    """Check OOV across all units.

    ``config`` is used to dispatch on ``embedding.format`` (gensim_kv default
    vs histwords). Optional for backwards compatibility with callers that
    monkeypatch ``load_model`` directly.
    """
    logger.info("\n" + "=" * 70)
    logger.info("Step 0: OOV Check")
    logger.info("=" * 70)

    categories = wordlists.get_all_categories()
    coverage_rows = []
    word_found = {w: {"category": cat, "found_in": []}
                  for cat, words in categories.items() for w in words}

    for unit_name, model_path in units:
        model = (load_model_for_unit(model_path, config)
                 if config is not None else load_model(str(model_path)))
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

def run_analysis(units, wordlists, logger, standardize=None, config=None):
    """Run steps 1-4 of the WEAT pipeline.

    ``config`` is used to dispatch on ``embedding.format`` (gensim_kv
    default vs histwords). Optional for backwards compatibility.
    """
    concept_words = wordlists.get_all_concept_words()

    def _load(path):
        return (load_model_for_unit(path, config)
                if config is not None else load_model(str(path)))

    # Step 1: Build gender axes
    logger.info("\n" + "=" * 70)
    logger.info("Step 1: Build gender axes")
    logger.info("=" * 70)

    gender_axes = {}
    for unit_name, model_path in units:
        model = _load(model_path)
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
        model = _load(info["model_path"])
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

    # Optional [start, end] decade clip — matches the Garg pipeline so
    # WEAT runs on HistWords (1810s-2000s) and on our trained_coha
    # (1910s-1990s) operate on the same comparable decade window.
    decade_range = config_data.get("analysis", {}).get("decade_range")
    if decade_range:
        try:
            start, end = int(decade_range[0]), int(decade_range[1])
        except (TypeError, ValueError, IndexError):
            raise ValueError(
                f"analysis.decade_range must be [start, end] integers, "
                f"got {decade_range!r}"
            )
        before = len(units)
        filtered: List[Tuple[str, Path]] = []
        for unit_name, path in units:
            year = _decade_to_year(unit_name)
            if year is None:
                logger.warning(
                    f"  decade_range filter: cannot parse decade from "
                    f"unit_name {unit_name!r}; keeping it"
                )
                filtered.append((unit_name, path))
            elif start <= year <= end:
                filtered.append((unit_name, path))
        units = filtered
        logger.info(
            f"decade_range filter [{start}, {end}]: {before} -> {len(units)} units"
        )

    if not units:
        logger.error("No models found")
        return

    logger.info(f"Found {len(units)} model units")

    # Step 0
    coverage_df = word_df = None
    if not skip_oov:
        coverage_df, word_df = run_oov_check(units, wordlists, logger, config=config_data)

    # Steps 1-4
    gender_axes, proj_df, stats_df, weat_df, use_zscore = run_analysis(
        units, wordlists, logger, standardize=standardize, config=config_data,
    )

    # Step 5
    save_results(config_data, coverage_df, word_df, gender_axes,
                 proj_df, stats_df, weat_df, use_zscore, logger)

    logger.info("=" * 80)
    logger.info("WEAT analysis completed!")
    logger.info("=" * 80)


if __name__ == "__main__":
    fire.Fire(main)
