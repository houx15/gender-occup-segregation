#!/usr/bin/env python3
"""
Build decade-partitioned corpora from COHA data.

Two input formats are supported, selected via config['coha']['format']:

  * "ngram"    — COHA free n-gram release (default; preserves prior behavior).
                 Each input file is TSV: <word1>\t<word2>\t...\t<wordN>\t<freq>.
                 Filenames embed the decade (e.g., "w4_1940.txt").

  * "fulltext" — COHA paid full-text release (one document per file, prose).
                 Each file starts with a "@@<docID>" header, then space-tokenized
                 prose with "@ @ @ @ @ @ @ @ @ @" (10-@) markers inserted at
                 random intervals as copyright redactions. Filenames look like
                 "fic_1900_1076.txt" / "mag_1955_99.txt" / "news_2005_42.txt".
                 This is the format HistWords / Garg actually trained on.

Output (same shape for both formats):
    {corpora_dir}/{decade}s/corpus_{shard_idx}.txt

Each output line is one mini-document (an n-gram, or one inter-redaction prose
chunk). train_embeddings.CorpusIterator reads each line as a sentence.

Usage:
    python -m scripts.data_prep.build_corpora_coha --config=config/config.yml
"""

import re
from pathlib import Path
from typing import List, Optional, Tuple

import fire

from scripts.common.config_loader import load_config
from scripts.common.logging_utils import setup_logging


ENGLISH_TOKEN_RE = re.compile(r"[a-z']+")
DECADE_RE = re.compile(r"(1[89]\d{2}|20\d{2})s?")

# COHA full-text artefacts.
# Header line of the form "@@1076" (any leading/trailing whitespace).
COHA_HEADER_RE = re.compile(r"^\s*@@\d+\s*$", re.MULTILINE)
# Redaction marker: 10+ "@" tokens separated by whitespace. Some files emit
# slightly more or fewer @s; we accept anything ≥ 5 in a row.
COHA_REDACTION_RE = re.compile(r"(?:@\s+){4,}@")


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

    The ngram_text is a space-joined sequence of n tokens. The n parameter
    is currently unused but kept for API stability.

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


def parse_coha_fulltext(text: str, min_tokens: int = 5) -> List[str]:
    """
    Parse a COHA full-text document into a list of clean prose chunks.

    Steps:
      1. Strip "@@<docID>" header line(s).
      2. Split on the "@ @ @ @ @ @ @ @ @ @" copyright redaction marker so
         that SGNS context windows do not span the (semantically broken)
         redaction gap.
      3. For each segment, lowercase + tokenize with [a-z']+ (drops
         punctuation tokens, keeps apostrophes).
      4. Drop chunks shorter than min_tokens.

    Returns one space-joined token string per surviving chunk.
    """
    # Drop header lines like "@@1076".
    cleaned = COHA_HEADER_RE.sub("", text)
    # Split on redaction markers; an empty / pure-whitespace segment naturally
    # tokenizes to [] and is filtered out below.
    segments = COHA_REDACTION_RE.split(cleaned)

    chunks: List[str] = []
    for seg in segments:
        tokens = ENGLISH_TOKEN_RE.findall(seg.lower())
        if len(tokens) < min_tokens:
            continue
        chunks.append(" ".join(tokens))
    return chunks


def _build_corpora_ngram(config, logger):
    """Original n-gram release path (TSV with frequency column)."""
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


def _build_corpora_fulltext(config, logger):
    """Full-text release path: one document per file, prose with @-redactions."""
    min_tokens = config.get("coha", {}).get("min_chunk_tokens", 5)
    decomp_dir = Path(config["paths"]["coha_decompressed_dir"])
    corpora_dir = Path(config["paths"]["corpora_dir"])

    txt_files = sorted(decomp_dir.rglob("*.txt"))
    logger.info(f"Found {len(txt_files)} text files under {decomp_dir}")

    # One aggregated shard per decade. Open file handles lazily; close at the end.
    decade_handles: dict[str, "object"] = {}
    decade_counts: dict[str, int] = {}
    skipped_no_decade = 0

    try:
        for txt_path in txt_files:
            decade = decade_from_filename(txt_path)
            if decade is None:
                skipped_no_decade += 1
                continue

            if decade not in decade_handles:
                out_dir = corpora_dir / decade
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / "corpus_aggregate.txt"
                decade_handles[decade] = open(out_path, "w", encoding="utf-8")
                decade_counts[decade] = 0
                logger.info(f"  Opened {out_path}")

            try:
                with open(txt_path, "r", encoding="utf-8", errors="ignore") as fin:
                    text = fin.read()
            except OSError as e:
                logger.warning(f"  Could not read {txt_path.name}: {e}")
                continue

            chunks = parse_coha_fulltext(text, min_tokens=min_tokens)
            fout = decade_handles[decade]
            for chunk in chunks:
                fout.write(chunk + "\n")
            decade_counts[decade] += len(chunks)
    finally:
        for fh in decade_handles.values():
            fh.close()

    if skipped_no_decade:
        logger.warning(
            f"Skipped {skipped_no_decade} files with no parseable decade in filename"
        )
    for decade in sorted(decade_counts):
        logger.info(f"  {decade}: {decade_counts[decade]:,} chunks")


def build_corpora(config, logger):
    """Dispatch to the correct format-specific builder."""
    fmt = config.get("coha", {}).get("format", "ngram")
    if fmt == "ngram":
        _build_corpora_ngram(config, logger)
    elif fmt == "fulltext":
        _build_corpora_fulltext(config, logger)
    else:
        raise ValueError(
            f"Unknown coha.format: {fmt!r}. Expected 'ngram' or 'fulltext'."
        )


def main(config: str = "config/config.yml"):
    """Build decade-partitioned corpora from COHA data."""
    cfg = load_config(config)
    if cfg["data_source"] != "coha":
        raise ValueError("build_corpora_coha requires data_source='coha' in config")
    logger = setup_logging(Path(cfg["paths"]["log_dir"]), "build_corpora_coha.log")

    fmt = cfg.get("coha", {}).get("format", "ngram")
    logger.info("=" * 80)
    logger.info(f"Starting COHA corpus building (format={fmt})")
    logger.info("=" * 80)

    build_corpora(cfg, logger)

    logger.info("=" * 80)
    logger.info("COHA corpus building completed!")
    logger.info("=" * 80)


if __name__ == "__main__":
    fire.Fire(main)
