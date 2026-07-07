#!/usr/bin/env python3
"""Precompute the data-derived per-year totalcounts for per_year_capped builds.

The shipped ``totalcounts-5`` under-counts relative to the summed match_counts we
emit, so ``scale = cap / year_total`` never bites and a 10-year slice blew past
``10·cap`` by ~6.7×. ``build_data_totalcounts`` derives the correct per-year
denominator (Σ kept match_count) from the data and caches it as
``raw_ngram_dir/totalcounts-5.derived``; every per-slice build then reuses it.

Run this ONCE before the slice sweep — and especially before splitting the sweep
into concurrent sbatch invocations, so they don't each redo the full pass (and
race on writing the cache).

Usage:
    python -m scripts.data_prep.precompute_ngram_totalcounts \
        --config=config/profiles/garg_weat_china_ngram_subsampled.yml
"""

from pathlib import Path

import fire

from scripts.common.config_loader import load_config
from scripts.common.logging_utils import setup_logging
from scripts.data_prep.build_corpora_ngram import build_data_totalcounts


def main(config: str = "config/profiles/garg_weat_china_ngram_subsampled.yml"):
    cfg = load_config(config)
    logger = setup_logging(
        Path(cfg["paths"]["log_dir"]), "precompute_ngram_totalcounts.log"
    )
    year_total = build_data_totalcounts(cfg, logger)

    # Surface the real per-year totals against the cap so the cap can be
    # sanity-checked (the old cap=1e8 was calibrated against the wrong totals).
    cap = int(cfg["corpus"].get("per_year_token_cap", 1_000_000_000))
    capped = sum(1 for t in year_total.values() if t > cap)
    logger.info(
        f"per_year_token_cap={cap:,}: {capped}/{len(year_total)} years will be CAPPED"
    )
    for y in sorted(year_total):
        t = year_total[y]
        logger.info(f"  {y}: {t:,}  {'CAPPED' if t > cap else 'pass-through'}")


if __name__ == "__main__":
    fire.Fire(main)
