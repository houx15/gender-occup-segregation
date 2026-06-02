#!/usr/bin/env python3
"""Dataset & training summary reporter — one Markdown per source.

Produces ``<results_dir>/dataset_summary.md`` with corpus totals, raw-data
footprint, training hyperparameters, and a per-unit breakdown. Reuses the
existing per-source profiles in ``config/profiles/``.

Usage:
    python -m scripts.describe_dataset --config=config/profiles/garg_weat_renminribao.yml
    python -m scripts.describe_dataset --config=…  --force
    python -m scripts.describe_dataset --config=…  --units=1940_1949,1950_1959
    python -m scripts.describe_dataset --config=…  --no-raw
    python -m scripts.describe_dataset --config=…  --no-model
    python -m scripts.describe_dataset --config=…  --vocab-union   # opt-in; bypasses cache
"""

from __future__ import annotations

from pathlib import Path

import fire

from scripts.common.config_loader import load_config
from scripts.common.dataset_stats import run as describe
from scripts.common.logging_utils import setup_logging


def main(
    config: str,
    force: bool = False,
    units: str = "",
    no_raw: bool = False,
    no_model: bool = False,
    vocab_union: bool = False,
) -> None:
    cfg = load_config(config)
    log_dir = Path(cfg["paths"].get("log_dir", cfg["paths"]["results_dir"]))
    logger = setup_logging(log_dir, "describe_dataset.log")
    describe(
        cfg,
        config_path=config,
        force=force,
        units=units or None,
        no_raw=no_raw,
        no_model=no_model,
        vocab_union=vocab_union,
        logger=logger,
    )


if __name__ == "__main__":
    fire.Fire(main)
