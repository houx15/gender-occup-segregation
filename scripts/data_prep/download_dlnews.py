#!/usr/bin/env python3
"""Transfer 3DLNews2 newspaper slices via Globus for the configured years.

3DLNews2 is distributed through Globus. Auth is a one-time interactive
`globus login` on the login node; transfers then run headless from this script.
Globus transfer is free for individual researchers; Princeton runs a managed
endpoint (dest).

Config (dlnews block):
  source_endpoint: <3DLNews2 collection UUID>
  dest_endpoint:   <Princeton endpoint UUID>
  source_root:     /.../preprocessed_state
  dest_root:       <raw_data_dir on the dest endpoint's namespace>
  states:          optional; default = all 50 states + DC

Fallback if OAuth can't run headless: this prints the equivalent
`globus transfer --batch` command; run it manually, then proceed to the builder.

Usage:
  python -m scripts.data_prep.download_dlnews --config=config/profiles/garg_weat_dlnews.yml
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple

import fire

from scripts.common.config_loader import load_config
from scripts.common.logging_utils import setup_logging
from scripts.data_prep import us_state_mapper as usm


def build_transfer_batch(source_root: str, dest_root: str, years: List[int],
                         states: List[str]) -> List[Tuple[str, str]]:
    """(src, dst) path pairs for each state x year newspaper slice."""
    pairs: List[Tuple[str, str]] = []
    for state in states:
        for year in years:
            fname = f"preprocessed_google_newspaper_{state}_{year}.jsonl.gz"
            src = f"{source_root}/{state}/{fname}"
            dst = f"{dest_root}/{fname}"
            pairs.append((src, dst))
    return pairs


def main(config: str = "config/config.yml", dry_run: bool = False) -> None:
    cfg = load_config(config)
    logger = setup_logging(Path(cfg["paths"]["log_dir"]), "download_dlnews.log")
    d = cfg["dlnews"]
    states = d.get("states") or list(usm._STATE_NAME_TO_USPS.keys())
    years = cfg["us_states"]["years"]
    pairs = build_transfer_batch(d["source_root"], d["dest_root"], years, states)

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False,
                                     encoding="utf-8") as bf:
        for src, dst in pairs:
            bf.write(f'"{src}" "{dst}"\n')
        batch_file = bf.name

    try:
        cmd = ["globus", "transfer", "--batch", batch_file,
               d["source_endpoint"], d["dest_endpoint"], "--label", "3dlnews2-us-arm"]
        logger.info(f"Prepared {len(pairs)} transfer pairs; batch file: {batch_file}")
        if dry_run:
            logger.info("dry_run: " + " ".join(cmd))
            return
        logger.info("Submitting Globus transfer (requires prior `globus login`)...")
        out = subprocess.run(cmd, capture_output=True, text=True)
        logger.info(out.stdout.strip())
        if out.returncode != 0:
            logger.error(out.stderr.strip())
            logger.error("If OAuth cannot run here, run the batch manually:\n  "
                         + " ".join(cmd))
            raise SystemExit(out.returncode)
        task_id = out.stdout.strip().split()[-1]
        subprocess.run(["globus", "task", "wait", task_id], check=False)
    finally:
        if os.path.exists(batch_file):
            os.remove(batch_file)


if __name__ == "__main__":
    fire.Fire(main)
