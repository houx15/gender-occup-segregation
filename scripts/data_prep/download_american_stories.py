#!/usr/bin/env python3
"""Download American Stories article text for the configured years.

Writes raw_data_dir/american_stories_{year}.jsonl (one article per line).
Network step — run where the node has internet (login/internet node).
Idempotent: skips a year whose output already exists and is non-empty.

Usage:
  python -m scripts.data_prep.download_american_stories --config=config/profiles/garg_weat_american_stories.yml
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import fire

from scripts.common.config_loader import load_config
from scripts.common.logging_utils import setup_logging

_KEEP = ("article_id", "newspaper_name", "date", "headline", "byline", "article")


def _download_year(year: int, out_path: str, logger) -> int:
    from datasets import load_dataset
    ds = load_dataset(
        "dell-research-harvard/AmericanStories",
        "subset_years",
        year_list=[str(year)],
        trust_remote_code=True,
    )
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for split in ds:
            for row in ds[split]:
                rec = {k: row.get(k, "") for k in _KEEP}
                if not rec.get("article"):
                    continue
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
    logger.info(f"  {year}: wrote {n} articles -> {out_path}")
    return n


def main(config: str = "config/config.yml") -> None:
    """Download American Stories articles for configured years to raw_data_dir."""
    cfg = load_config(config)
    logger = setup_logging(Path(cfg["paths"]["log_dir"]), "download_american_stories.log")
    raw_dir = cfg["paths"]["raw_data_dir"]
    os.makedirs(raw_dir, exist_ok=True)
    for year in cfg["us_states"]["years"]:
        out = os.path.join(raw_dir, f"american_stories_{year}.jsonl")
        if os.path.exists(out) and os.path.getsize(out) > 0:
            logger.info(f"  {year}: exists, skipping")
            continue
        logger.info(f"Downloading American Stories {year}...")
        _download_year(year, out, logger)


if __name__ == "__main__":
    fire.Fire(main)
