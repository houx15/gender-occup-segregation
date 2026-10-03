#!/usr/bin/env python3
"""Transfer 3DLNews2 collections (all years, all states) via Globus.

3DLNews2 is distributed through Globus. Auth is a one-time interactive
`globus login` on the login node; transfers then run headless from this script.

Each configured collection (one platform x media type, e.g. Google newspaper,
Google TV) is copied RECURSIVELY from its ``preprocessed_state`` dir into its
own subdir of dest_root, so every year/state comes along without guessing
per-media filenames, and media types stay separate on disk:

  {dest_root}/google_newspaper/{USPS}/preprocessed_newspaper_articles_{USPS}_{YEAR}.jsonl.gz
  {dest_root}/google_tv/{USPS}/...

Which collections feed a given corpus is chosen at build time
(``dlnews.corpus_collections``), so newspaper and TV can be analysed apart or
together. The live Globus layout differs from the repo README (verified for
Google newspaper 2026-08-11: ``/1-Google/1-Newspaper/preprocessed_state``), so
each source dir is checked with ``globus ls`` before transfer and a wrong path
fails loudly instead of transferring nothing.

Config (dlnews block):
  source_endpoint: <3DLNews2 no-HTML collection UUID>
  dest_endpoint:   <Princeton endpoint UUID>
  dest_root:       <raw_data_dir on the dest endpoint's namespace>
  collections:     {name: source preprocessed_state dir}
  states:          optional 2-letter USPS allow-list; default = whole dirs

Usage:
  python -m scripts.data_prep.download_dlnews --config=config/profiles/garg_weat_dlnews.yml
  python -m scripts.data_prep.download_dlnews --config=... --dry_run
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fire

from scripts.common.config_loader import load_config
from scripts.common.logging_utils import setup_logging


def build_transfer_batch(collections: Dict[str, str], dest_root: str,
                         states: Optional[List[str]] = None) -> List[Tuple[str, str]]:
    """Recursive (src_dir, dst_dir) pairs: one per collection, or per collection x state."""
    if not collections:
        raise ValueError("dlnews.collections is empty — nothing to transfer")
    pairs: List[Tuple[str, str]] = []
    for name, src_root in collections.items():
        src_root = src_root.rstrip("/")
        dst_root = f"{dest_root.rstrip('/')}/{name}"
        if states:
            pairs.extend((f"{src_root}/{s}", f"{dst_root}/{s}") for s in states)
        else:
            pairs.append((src_root, dst_root))
    return pairs


def write_batch_file(pairs: List[Tuple[str, str]], tmp_dir: Optional[str] = None) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False,
                                     dir=tmp_dir, encoding="utf-8") as bf:
        for src, dst in pairs:
            bf.write(f'--recursive "{src}" "{dst}"\n')
        return bf.name


def check_sources(endpoint: str, collections: Dict[str, str], logger) -> None:
    """`globus ls` each source dir; abort listing every path that doesn't exist."""
    missing = []
    for name, src in collections.items():
        out = subprocess.run(["globus", "ls", f"{endpoint}:{src}"],
                             capture_output=True, text=True)
        if out.returncode != 0:
            missing.append(f"  {name}: {src}\n    {out.stderr.strip()}")
        else:
            logger.info(f"source ok: {name} -> {src} "
                        f"({len(out.stdout.split())} entries)")
    if missing:
        raise SystemExit(
            "3DLNews2 source dirs not found on the collection:\n" + "\n".join(missing)
            + f"\nBrowse the real layout with `globus ls {endpoint}:/` and fix "
              "dlnews.collections in the config.")


def main(config: str = "config/config.yml", dry_run: bool = False) -> None:
    cfg = load_config(config)
    logger = setup_logging(Path(cfg["paths"]["log_dir"]), "download_dlnews.log")
    d = cfg["dlnews"]
    pairs = build_transfer_batch(d["collections"], d["dest_root"], d.get("states"))
    batch_file = write_batch_file(pairs)

    try:
        # --sync-level checksum makes re-runs idempotent: Globus skips files
        # already present at the dest with a matching checksum, so an interrupted
        # transfer resumes instead of re-copying everything.
        cmd = ["globus", "transfer", "--batch", batch_file,
               d["source_endpoint"], d["dest_endpoint"],
               "--label", "3dlnews2-us-arm",
               "--sync-level", d.get("sync_level", "checksum")]
        logger.info(f"Prepared {len(pairs)} recursive transfer pairs: "
                    + ", ".join(sorted(d["collections"])))
        if dry_run:
            logger.info("dry_run: " + " ".join(cmd))
            return
        check_sources(d["source_endpoint"], d["collections"], logger)
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
