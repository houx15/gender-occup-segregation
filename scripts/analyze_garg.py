#!/usr/bin/env python3
"""
Garg-mode analysis: compute relative norm distance (RND) between occupation
vectors and gender centroids, per time/unit.

Implements Garg et al. (2018) "Word embeddings quantify 100 years of gender
and ethnic stereotypes". Per unit:
  RND(occ) = ||v_occ - c_male|| - ||v_occ - c_female||  (Garg sign convention)
  mean_rnd over in-vocab occupations + 95% bootstrap CI

Optional: when `wordlists.occupation_percentages_file` is set, also computes
`mean_pct_diff` per decade — the average of (2*female_share - 1)*100 over the
SAME set of occupations used for mean_rnd. Mirrors Garg's overlay axis in
plot_creation.plot_averagebias_over_time_consistentoccupations.

Optional: when `analysis.consistent_occupations: true`, restricts every
decade's mean to occupations that are in vocab in ALL decades (and that have
valid census data in all decades when the percent CSV is configured). This
matches Garg's per-figure consistent-set semantics and removes spurious
cliffs caused by per-decade vocab churn.

Outputs:
  garg_relative_norm_by_decade.parquet
    columns: unit_name, occupation, rnd, in_vocab
  garg_average_bias_by_decade.parquet
    columns: unit_name, mean_rnd, ci_low, ci_high, n_occupations,
             mean_pct_diff (NaN when no percent CSV), n_consistent

Usage:
    python -m scripts.analyze_garg --config=config/config.yml
    python -m scripts.analyze_garg --config=config/config.yml --unit=1990s
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import fire

from scripts.common.config_loader import (
    load_config, get_wordlist_dir, _parse_model_template,
)
from scripts.common.embedding_utils import (
    load_model, check_oov,
)
from scripts.common.logging_utils import setup_logging
from scripts.common.metrics import (
    relative_norm_distance, bootstrap_ci, l2_normalize,
)


_DECADE_RE = re.compile(r"^(\d{4})s$")


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


def load_occupation_percent_data(file_path: Path) -> Dict[str, Dict[int, float]]:
    """Load Garg's occupation_percentages_gender CSV.

    CSV columns: Census year, Occupation, Total Weight, Female, Male.
    Returns {occupation: {census_year: pct_diff}} where
      pct_diff = (2 * female_share - 1) * 100
    matching utilities.occupation_func_female_percent.
    """
    out: Dict[str, Dict[int, float]] = {}
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                year = int(row["Census year"])
                female = float(row["Female"])
            except (ValueError, KeyError, TypeError):
                continue
            occ = row["Occupation"].strip()
            pct_diff = (2.0 * female - 1.0) * 100.0
            out.setdefault(occ, {})[year] = pct_diff
    return out


def decade_to_census_year(unit_name: str) -> Optional[int]:
    """'1990s' -> 1990. Returns None if the unit name doesn't match."""
    m = _DECADE_RE.match(unit_name)
    if not m:
        return None
    return int(m.group(1))


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

    # L2-normalize every fetched vector before centroid + distance, matching
    # Garg et al. 2018's dataset_utilities/normalize_vectors.py. Without this,
    # decade-to-decade vector-norm drift inflates RND magnitudes and breaks
    # direct comparison to Garg's Fig 2 numbers.
    male_unit = [l2_normalize(model[w]) for w in gender_words["male"] if w in model.key_to_index]
    female_unit = [l2_normalize(model[w]) for w in gender_words["female"] if w in model.key_to_index]
    male_found = [w for w in gender_words["male"] if w in model.key_to_index]
    female_found = [w for w in gender_words["female"] if w in model.key_to_index]

    if not male_unit or not female_unit:
        logger.warning(
            f"  {unit_name}: skipping — could not compute gender centroids "
            f"(male_found={len(male_found)}, female_found={len(female_found)})"
        )
        return None

    c_male = np.mean(np.asarray(male_unit), axis=0)
    c_female = np.mean(np.asarray(female_unit), axis=0)

    long_rows = []
    in_vocab_rnds: List[float] = []

    for occ in occupations:
        found, _oov = check_oov(model, [occ])
        if found:
            occ_vec = l2_normalize(model[occ])
            rnd = relative_norm_distance(occ_vec, c_male, c_female)
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


def compute_consistent_set(
    long_df: pd.DataFrame,
    occ_percents: Optional[Dict[str, Dict[int, float]]],
    units: List[str],
    logger,
) -> List[str]:
    """Occupations in vocab in ALL units, plus (if provided) with valid pct
    data in every census year corresponding to a unit. Mirrors Garg's
    "consistentoccupations" filter.
    """
    in_all_units = set(long_df.loc[long_df["in_vocab"], "occupation"].unique())
    for u in units:
        unit_in_vocab = set(
            long_df.loc[(long_df["unit_name"] == u) & long_df["in_vocab"], "occupation"]
        )
        in_all_units &= unit_in_vocab

    if occ_percents is not None:
        years = [y for y in (decade_to_census_year(u) for u in units) if y is not None]
        with_pct = set()
        for occ, year_to_pct in occ_percents.items():
            if all(y in year_to_pct and not np.isnan(year_to_pct[y]) for y in years):
                with_pct.add(occ)
        consistent = sorted(in_all_units & with_pct)
        logger.info(
            f"Consistent-set filter: {len(in_all_units)} in vocab in all units, "
            f"{len(with_pct)} with full census coverage, "
            f"{len(consistent)} surviving."
        )
    else:
        consistent = sorted(in_all_units)
        logger.info(
            f"Consistent-set filter (no pct): {len(consistent)} occupations in "
            f"vocab in all {len(units)} units."
        )
    return consistent


def build_summary(
    long_df: pd.DataFrame,
    units: List[str],
    occ_percents: Optional[Dict[str, Dict[int, float]]],
    consistent_set: Optional[List[str]],
    logger,
) -> pd.DataFrame:
    """Per-unit summary row. If consistent_set is given, mean_rnd / mean_pct_diff
    are computed only over those occupations; otherwise per-unit independent.
    """
    rows: List[dict] = []
    for u in units:
        unit_long = long_df[long_df["unit_name"] == u]

        if consistent_set is not None:
            sub = unit_long[unit_long["occupation"].isin(consistent_set) & unit_long["in_vocab"]]
            n_consistent = len(consistent_set)
        else:
            sub = unit_long[unit_long["in_vocab"]]
            n_consistent = int(sub.shape[0])

        if sub.empty:
            mean_rnd = np.nan
            ci_low = ci_high = np.nan
            n_occ = 0
        else:
            rnd_arr = sub["rnd"].to_numpy(dtype=float)
            mean_rnd = float(rnd_arr.mean())
            ci_low, ci_high = bootstrap_ci(rnd_arr, n_iter=1000, ci=0.95, seed=42)
            n_occ = int(rnd_arr.size)

        mean_pct_diff = np.nan
        if occ_percents is not None and not sub.empty:
            year = decade_to_census_year(u)
            if year is not None:
                pcts = []
                for occ in sub["occupation"]:
                    yr_to_pct = occ_percents.get(occ)
                    if yr_to_pct is not None and year in yr_to_pct:
                        pcts.append(yr_to_pct[year])
                if pcts:
                    mean_pct_diff = float(np.mean(pcts))

        rows.append({
            "unit_name": u,
            "mean_rnd": mean_rnd,
            "ci_low": float(ci_low) if not np.isnan(ci_low) else np.nan,
            "ci_high": float(ci_high) if not np.isnan(ci_high) else np.nan,
            "n_occupations": n_occ,
            "mean_pct_diff": mean_pct_diff,
            "n_consistent": n_consistent,
        })

    return pd.DataFrame(
        rows,
        columns=[
            "unit_name", "mean_rnd", "ci_low", "ci_high",
            "n_occupations", "mean_pct_diff", "n_consistent",
        ],
    )


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

    occ_pct_filename = wl_config.get("occupation_percentages_file")
    occ_percents: Optional[Dict[str, Dict[int, float]]] = None
    if occ_pct_filename:
        pct_path = wl_dir / occ_pct_filename
        if pct_path.exists():
            occ_percents = load_occupation_percent_data(pct_path)
            logger.info(
                f"Loaded occupation percentages: {len(occ_percents)} occupations"
            )
        else:
            logger.warning(
                f"occupation_percentages_file set to {pct_path} but file not "
                f"found — second-axis overlay will be skipped."
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
    units: List[str] = []
    for model_path, unit_name in models:
        result = analyze_unit(
            model_path, unit_name, occupations, gender_words, logger
        )
        if result is None:
            continue
        long_df, _ = result
        long_frames.append(long_df)
        units.append(unit_name)

    if not long_frames:
        logger.error("No units produced results — nothing written")
        return

    long_combined = pd.concat(long_frames, ignore_index=True)

    use_consistent = bool(
        config_data.get("analysis", {}).get("consistent_occupations", False)
    )
    if use_consistent:
        consistent_set = compute_consistent_set(
            long_combined, occ_percents, units, logger
        )
    else:
        consistent_set = None

    summary_df = build_summary(
        long_combined, units, occ_percents, consistent_set, logger
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
