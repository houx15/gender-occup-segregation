#!/usr/bin/env python3
"""
Build decade-partitioned corpora from COHA free n-gram data.

Expected input: TSV files under config['paths']['coha_decompressed_dir'],
one or more files per decade, each line:
    word1<TAB>word2<TAB>...<TAB>wordN<TAB>freq

Filenames are expected to embed the decade (e.g., "w4_1940.txt",
"coha_4grams_1950s.txt"). Any filename containing a 4-digit year or a
"<year>s" decade marker is accepted.

Output:
    {corpora_dir}/{decade}s/corpus_{shard_idx}.txt

Each output line is a single n-gram with tokens joined by single spaces.
train_embeddings.py treats each line as a mini-document, matching the
existing Chinese ngram flow.

Usage:
    python -m scripts.data_prep.build_corpora_coha --config=config/config.yml
"""

import os
import re
from pathlib import Path
from typing import Optional, Tuple

import fire

from scripts.common.config_loader import load_config
from scripts.common.logging_utils import setup_logging


ENGLISH_TOKEN_RE = re.compile(r"[a-z']+")
DECADE_RE = re.compile(r"(1[89]\d{2}|20\d{2})s?")


def decade_from_filename(path: Path) -> Optional[str]:
    """Extract a '1940s'-style decade label from a COHA filename."""
    m = DECADE_RE.search(path.stem)
    if not m:
        return None
    year = int(m.group(1))
    return f"{(year // 10) * 10}s"


def parse_coha_line(line: str, n: int) -> Optional[Tuple[str, int]]:
    """
    Parse a single TSV line: <ngram_text><TAB><freq>

    The ngram_text is a space-joined sequence of n tokens.  The n parameter
    is used only to validate that the text contains the right number of
    space-separated tokens (after cleaning).

    Returns (cleaned_ngram_text, freq) or None if the line should be skipped.
    """
    parts = line.rstrip("\n").split("\t")
    if len(parts) < 2:
        return None
    try:
        freq = int(parts[-1])
    except ValueError:
        return None
    raw_text = "\t".join(parts[:-1])
    tokens = ENGLISH_TOKEN_RE.findall(raw_text.lower())
    if len(tokens) < 2:
        return None
    return " ".join(tokens), freq


def build_corpora(config, logger):
    n = config.get("coha", {}).get("ngram_order", 4)
    min_freq = config.get("coha", {}).get("min_freq", 1)
    decomp_dir = Path(config["paths"]["coha_decompressed_dir"])
    corpora_dir = Path(config["paths"]["corpora_dir"])

    tsv_files = sorted(decomp_dir.rglob("*.txt")) + sorted(decomp_dir.rglob("*.tsv"))
    logger.info(f"Found {len(tsv_files)} TSV files under {decomp_dir}")

    for idx, tsv_path in enumerate(tsv_files):
        decade = decade_from_filename(tsv_path)
        if decade is None:
            logger.warning(f"Could not parse decade from {tsv_path.name}; skipping")
            continue

        out_dir = corpora_dir / decade
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"corpus_{idx:03d}.txt"

        logger.info(f"Processing {tsv_path.name} -> {out_path}")
        lines_written = 0
        with open(tsv_path, "r", encoding="utf-8", errors="ignore") as fin, \
             open(out_path, "w", encoding="utf-8") as fout:
            for line in fin:
                result = parse_coha_line(line, n)
                if result is None:
                    continue
                text, freq = result
                if freq < min_freq:
                    continue
                fout.write(text + "\n")
                lines_written += 1
        logger.info(f"  Wrote {lines_written:,} n-grams to {out_path.name}")


def main(config: str = "config/config.yml"):
    """Build decade-partitioned corpora from COHA free n-gram data."""
    cfg = load_config(config)
    if cfg["data_source"] != "coha":
        raise ValueError("build_corpora_coha requires data_source='coha' in config")
    logger = setup_logging(Path(cfg["paths"]["log_dir"]), "build_corpora_coha.log")

    logger.info("=" * 80)
    logger.info("Starting COHA corpus building")
    logger.info("=" * 80)

    build_corpora(cfg, logger)

    logger.info("=" * 80)
    logger.info("COHA corpus building completed!")
    logger.info("=" * 80)


if __name__ == "__main__":
    fire.Fire(main)
