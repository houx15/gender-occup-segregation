#!/usr/bin/env python3
"""Download American Stories article text for the configured years.

Writes raw_data_dir/american_stories_{year}.jsonl (one article per line).
Network step — run where the node has internet (login/internet node).
Idempotent: skips a year whose output already exists and is non-empty.
Streams from HuggingFace (memory-flat) — a decade-year is millions of articles,
so we never materialize a year into RAM/Arrow cache.

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
    import inspect

    from datasets import load_dataset

    # streaming=True is essential: a single decade-year is millions of articles
    # (1900 alone is ~6.6M). The non-streaming path first MATERIALIZES the whole
    # year into an Arrow cache before we write a line — that memory/disk spike is
    # what the login-node cgroup SIGKILLs ("Killed"). Streaming pulls the tar and
    # yields rows one at a time, so memory stays flat and nothing is cached to
    # scratch twice.
    #
    # `trust_remote_code` is only a load_dataset() parameter on datasets>=2.16.
    # Older versions run the dataset script without it AND reject the kwarg
    # (it gets forwarded to the builder config). Pass it only if supported.
    kwargs = {"year_list": [str(year)], "streaming": True}
    if "trust_remote_code" in inspect.signature(load_dataset).parameters:
        kwargs["trust_remote_code"] = True
    ds = load_dataset(
        "dell-research-harvard/AmericanStories",
        "subset_years",
        **kwargs,
    )
    n = 0
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            for split in ds:
                for row in ds[split]:
                    rec = {k: row.get(k, "") for k in _KEEP}
                    if not rec.get("article"):
                        continue
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n += 1
                    if n % 500_000 == 0:
                        logger.info(f"  {year}: streamed {n:,} articles...")
    except Exception:
        # Don't leave a truncated file — a non-empty partial would be silently
        # skipped by main()'s idempotency check on the next run.
        if os.path.exists(out_path):
            os.remove(out_path)
        raise
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
