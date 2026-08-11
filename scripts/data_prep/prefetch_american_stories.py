#!/usr/bin/env python3
"""Prefetch American Stories raw tarballs — the NETWORK step, kept LIGHT.

Downloads only the raw ``faro_{year}.tar.gz`` archives to raw_data_dir on a
login/internet node. This is pure I/O (~100s/year) and stays well under a login
node's wall-clock/CPU limits — unlike the article extraction, which is CPU/
memory-heavy and belongs on a compute node. The compute-node build step
(build_corpora_us) then parses these tarballs OFFLINE and directly, with no
intermediate JSONL, so nothing large is written twice to scratch.

Idempotent: a year whose tarball already exists (non-empty) is skipped.

Usage (login node, after `pip install huggingface_hub`):
  python -m scripts.data_prep.prefetch_american_stories --config=config/profiles/garg_weat_american_stories.yml
"""

from __future__ import annotations

from pathlib import Path

import fire

from scripts.common.config_loader import load_config
from scripts.common.logging_utils import setup_logging

_REPO = "dell-research-harvard/AmericanStories"


def main(config: str = "config/config.yml") -> None:
    """Download the configured years' faro_{year}.tar.gz to raw_data_dir."""
    cfg = load_config(config)
    logger = setup_logging(Path(cfg["paths"]["log_dir"]), "prefetch_american_stories.log")
    raw_dir = Path(cfg["paths"]["raw_data_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import hf_hub_download

    for year in cfg["us_states"]["years"]:
        fname = f"faro_{year}.tar.gz"
        dest = raw_dir / fname
        if dest.exists() and dest.stat().st_size > 0:
            logger.info(f"  {year}: {fname} exists, skipping")
            continue
        logger.info(f"Prefetching {fname} ...")
        path = hf_hub_download(
            repo_id=_REPO, filename=fname, repo_type="dataset",
            local_dir=str(raw_dir),
        )
        logger.info(f"  {year}: cached -> {path}")

    logger.info("Prefetch done. Build corpora offline on a compute node: "
                "python -m scripts.data_prep.build_corpora_us --config=<config>")


if __name__ == "__main__":
    fire.Fire(main)
