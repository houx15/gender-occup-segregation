#!/usr/bin/env python3
"""
OOV probe: for a given config + a directory of candidate occupation files,
report per-decade per-word vocabulary membership.

Dry-run diagnostic used BEFORE committing to a final wordlist for the
analyze_garg_weat experiment. Workflow:
  1. Build oversized candidate lists in wordlists/en/garg_weat/.
  2. Run this probe across all three embedding sources via
     slurm/oov_probe.slurm.
  3. Inspect the per-word coverage table and prune candidates that OOV in
     too many decades.

Reuses discover_models + load_model_for_unit from analyze_garg so model
discovery + format dispatch (gensim_kv vs histwords) stays consistent
with the real analyzer.

Usage:
    python -m scripts.oov_probe \
        --config=config/profiles/coha_garg.yml \
        --wordlist_dir=wordlists/en/garg_weat \
        --candidates=candidates_leadership.txt,candidates_family.txt,candidates_science.txt \
        --out=/tmp/oov_trained_coha.csv

Output CSV columns: source, decade, category, word, in_vocab
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd
import fire

from scripts.common.config_loader import load_config
from scripts.common.logging_utils import setup_logging
from scripts.analyze_garg import (
    discover_models, load_model_for_unit, decade_to_census_year,
)


def _category_from_filename(fname: str) -> str:
    """'candidates_leadership.txt' -> 'leadership'."""
    stem = Path(fname).stem
    if stem.startswith("candidates_"):
        return stem[len("candidates_"):]
    return stem


def _load_words(wl_dir: Path, fname: str) -> List[str]:
    with open(wl_dir / fname, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def main(
    config: str,
    wordlist_dir: str,
    candidates: str,
    out: str = "oov_probe.csv",
) -> None:
    cfg = load_config(config)
    logger = setup_logging(Path(cfg["paths"]["log_dir"]), "oov_probe.log")

    wl_dir = Path(wordlist_dir)
    candidate_files = [c.strip() for c in candidates.split(",") if c.strip()]

    categories = {
        _category_from_filename(f): _load_words(wl_dir, f)
        for f in candidate_files
    }

    source = cfg.get("embedding_source", "unknown")
    logger.info("=" * 70)
    logger.info(f"OOV probe — source={source}")
    logger.info("=" * 70)
    for cat, words in categories.items():
        logger.info(f"  {cat}: {len(words)} candidates")

    models = discover_models(cfg)

    # Mirror analyze_garg's decade_range clip so the HistWords runs only
    # probe inside the comparable [1910, 1990] window when configured.
    decade_range = cfg.get("analysis", {}).get("decade_range")
    if decade_range:
        start, end = int(decade_range[0]), int(decade_range[1])
        filtered = []
        for path, unit in models:
            year = decade_to_census_year(unit)
            if year is None or start <= year <= end:
                filtered.append((path, unit))
        logger.info(
            f"decade_range [{start}, {end}]: {len(models)} -> {len(filtered)} models"
        )
        models = filtered

    if not models:
        logger.error("No models discovered")
        return

    rows = []
    for model_path, unit_name in models:
        model = load_model_for_unit(model_path, cfg)
        vocab_size = len(model.key_to_index)
        logger.info(f"  {unit_name}: vocab_size={vocab_size}")
        for cat, words in categories.items():
            n_in = 0
            for w in words:
                in_vocab = w in model.key_to_index
                if in_vocab:
                    n_in += 1
                rows.append({
                    "source": source,
                    "decade": unit_name,
                    "category": cat,
                    "word": w,
                    "in_vocab": in_vocab,
                })
            logger.info(f"    {cat}: {n_in}/{len(words)} in vocab")
        del model

    df = pd.DataFrame(rows)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info(f"Saved: {out_path}")

    # Per-word coverage across decades for this source — sorted ascending
    # so the worst candidates show at the top.
    per_word = (
        df.groupby(["category", "word"])["in_vocab"]
          .agg(["sum", "count"])
          .rename(columns={"sum": "n_in_vocab", "count": "n_decades"})
    )
    per_word["coverage"] = per_word["n_in_vocab"] / per_word["n_decades"]
    per_word = per_word.sort_values(["category", "coverage"])
    print(f"\nPer-word coverage for source={source} (ascending — prune from top):")
    print(per_word.to_string())


if __name__ == "__main__":
    fire.Fire(main)
