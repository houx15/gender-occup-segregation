#!/usr/bin/env python3
"""Garg-WEAT mode: per-category RND producer + thin CLI.

The RND metric (||v - c_male|| - ||v - c_female||, Garg sign convention,
L2-normalized vectors) over one occupation list per category. The summary
machinery now lives in scripts.common.category_summary and is shared with the
single-list Cohen's d analysis; this module keeps the RND-specific per-word
producer and re-exports the shared helpers for backward compatibility.

Outputs (written by the orchestrator, names unchanged):
  garg_weat_rnd_long.parquet            cols: unit_name, category, occupation, rnd, in_vocab
  garg_weat_summary_by_category.parquet mean_rnd/ci_*/sub_* (+ new prop_* columns)

Usage:
  python -m scripts.analyze_garg_weat --config=config/profiles/garg_weat_coha_trained.yml
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import fire

from scripts.common.config_loader import load_config
from scripts.common.metrics import relative_norm_distance, l2_normalize
# Re-export shared helpers so existing imports/tests keep working.
from scripts.common.category_summary import (  # noqa: F401
    load_categories, compute_consistent_set,
    subsample_bands_from_lookup, build_summary,
)


def rnd_values(
    model,
    unit_name: str,
    categories: Dict[str, List[str]],
    gender_words: dict,
    logger,
) -> Optional[pd.DataFrame]:
    """Per-unit RND across all categories, on an ALREADY-LOADED model.

    Returns a long DataFrame (unit_name, category, occupation, value, in_vocab)
    or None if gender centroids are unobtainable. ``value`` is the RND.
    """
    male_vecs = [l2_normalize(model[w]) for w in gender_words["male"] if w in model.key_to_index]
    female_vecs = [l2_normalize(model[w]) for w in gender_words["female"] if w in model.key_to_index]
    logger.info(
        f"  {unit_name}: gender words — "
        f"male {len(male_vecs)}/{len(gender_words['male'])} found, "
        f"female {len(female_vecs)}/{len(gender_words['female'])} found"
    )
    if not male_vecs or not female_vecs:
        logger.warning(
            f"  {unit_name}: skipping RND — gender centroids unobtainable "
            f"(male={len(male_vecs)}, female={len(female_vecs)})"
        )
        return None

    c_male = np.mean(np.asarray(male_vecs), axis=0)
    c_female = np.mean(np.asarray(female_vecs), axis=0)

    rows: List[dict] = []
    for cat_name, words in categories.items():
        n_in = 0
        for w in words:
            if w in model.key_to_index:
                vec = l2_normalize(model[w])
                rows.append({
                    "unit_name": unit_name, "category": cat_name, "occupation": w,
                    "value": float(relative_norm_distance(vec, c_male, c_female)),
                    "in_vocab": True,
                })
                n_in += 1
            else:
                rows.append({
                    "unit_name": unit_name, "category": cat_name, "occupation": w,
                    "value": np.nan, "in_vocab": False,
                })
        logger.info(f"    {cat_name}: {n_in}/{len(words)} in vocab")
    return pd.DataFrame(rows)


def main(config: str = "config/config.yml", unit: Optional[str] = None) -> None:
    """Run the per-category RND analysis (delegates to the orchestrator)."""
    from scripts.analyze_category_bias import run  # local import avoids cycle
    config_data = load_config(config)
    run(config_data, metrics=["rnd"], unit=unit)


if __name__ == "__main__":
    fire.Fire(main)
