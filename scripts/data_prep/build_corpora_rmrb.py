#!/usr/bin/env python3
"""
Build time-sliced corpora from Renminribao (People's Daily) text data.

Data path format:
    {decade}/{year}/报刊/人民日报/rmrb_{year}_{month}.txt

Output format:
    {corpora_dir}/{start_year}_{end_year}/corpus_*.txt

Usage:
    python -m scripts.data_prep.build_corpora_rmrb --config=config/config.yml
    python -m scripts.data_prep.build_corpora_rmrb --config=config/config.yml --slice=1940_1949
"""

import os
import re
from pathlib import Path
from typing import List, Tuple
from collections import defaultdict

import fire

from scripts.common.config_loader import load_config
from scripts.common.logging_utils import setup_logging
from scripts.common.preprocessing import preprocess


def generate_time_slices(start_year, end_year, window_size, step_size):
    slices = []
    current_start = start_year
    while current_start <= end_year:
        current_end = min(current_start + window_size - 1, end_year)
        slices.append((current_start, current_end))
        current_start += step_size
        if current_start > end_year:
            break
    return slices


def extract_year_from_filename(filename):
    match = re.search(r'rmrb_(\d{4})_\d{2}\.txt', filename)
    return int(match.group(1)) if match else None


def find_renminribao_files(data_dir):
    files = []
    for decade_dir in sorted(data_dir.glob("*s")):
        if not decade_dir.is_dir():
            continue
        for year_dir in sorted(decade_dir.glob("*")):
            if not year_dir.is_dir():
                continue
            rmr_dir = year_dir / "报刊" / "人民日报"
            if not rmr_dir.exists():
                continue
            for txt_file in sorted(rmr_dir.glob("rmrb_*.txt")):
                year = extract_year_from_filename(txt_file.name)
                if year:
                    files.append((txt_file, year))
    return files


def process_file(file_path, year, time_slices, corpora_dir, logger, config, buffer_size=10000):
    stats = defaultdict(int)
    matched_slices = [
        f"{s}_{e}" for s, e in time_slices if s <= year <= e
    ]
    if not matched_slices:
        return stats

    file_index = file_path.stem.replace("rmrb_", "").replace("_", "")
    buffers = {s: [] for s in matched_slices}
    corpus_files = {}
    for s in matched_slices:
        slice_dir = corpora_dir / s
        slice_dir.mkdir(parents=True, exist_ok=True)
        corpus_files[s] = slice_dir / f"corpus_{file_index}.txt"

    try:
        min_words = config.get("corpus", {}).get("min_words", 5)
        lines_processed = 0
        with open(file_path, 'r', encoding='gb18030', errors='ignore') as f:
            for line in f:
                lines_processed += 1
                tokens = preprocess(
                    line,
                    language=config["language"],
                    tokenizer=config["corpus"]["tokenizer"],
                    stopwords_key=config["corpus"].get("stopwords"),
                    lowercase=config["corpus"].get("lowercase", False),
                    min_words=min_words,
                )
                if tokens is None:
                    continue
                segmented = " ".join(tokens)
                for s in matched_slices:
                    buffers[s].append(segmented)
                    stats[f"{s}_lines"] += 1
                if len(buffers[matched_slices[0]]) >= buffer_size:
                    for s in matched_slices:
                        with open(corpus_files[s], 'a', encoding='utf-8') as out_f:
                            out_f.write("\n".join(buffers[s]) + "\n")
                        buffers[s] = []

        for s in matched_slices:
            if buffers[s]:
                with open(corpus_files[s], 'a', encoding='utf-8') as out_f:
                    out_f.write("\n".join(buffers[s]) + "\n")

        stats['files_processed'] = 1
        stats['lines_read'] = lines_processed
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        stats['errors'] = 1

    return stats


def build_corpora(config, logger, specific_slice=None, data_dir=None):
    ts_config = config['time_slices']
    time_slices = generate_time_slices(
        ts_config['start_year'], ts_config['end_year'],
        ts_config['window_size'], ts_config['step_size']
    )
    logger.info(f"Generated {len(time_slices)} time slices")

    if specific_slice:
        start, end = map(int, specific_slice.split('_'))
        time_slices = [(start, end)]

    if data_dir:
        data_path = Path(data_dir)
    elif 'raw_data_dir' in config.get('paths', {}):
        data_path = Path(config['paths']['raw_data_dir'])
    else:
        data_path = Path("/scratch/network/yh6580/chinese_corpus/xiandai_20250422/xiandai")
        logger.warning(f"Using default data path: {data_path}")

    if not data_path.exists():
        logger.error(f"Data directory not found: {data_path}")
        return

    corpora_dir = Path(config['paths']['corpora_dir'])
    corpora_dir.mkdir(parents=True, exist_ok=True)

    files = find_renminribao_files(data_path)
    logger.info(f"Found {len(files)} Renminribao files")

    if not files:
        logger.error("No Renminribao files found")
        return

    total_stats = defaultdict(int)
    for idx, (file_path, year) in enumerate(files, 1):
        logger.info(f"Processing [{idx}/{len(files)}]: {file_path.name} (year: {year})")
        stats = process_file(file_path, year, time_slices, corpora_dir, logger, config)
        for key, value in stats.items():
            total_stats[key] += value

    logger.info("=" * 80)
    logger.info(f"Completed: {total_stats['files_processed']} files, {total_stats['errors']} errors")


def main(config='config/config.yml', slice=None, data_dir=None):
    """Build time-sliced corpora from Renminribao data."""
    config_data = load_config(config)
    logger = setup_logging(Path(config_data['paths']['log_dir']), "build_corpora_rmrb.log")

    logger.info("=" * 80)
    logger.info("Starting Renminribao corpus building")
    logger.info("=" * 80)

    build_corpora(config_data, logger, specific_slice=slice, data_dir=data_dir)

    logger.info("=" * 80)
    logger.info("Renminribao corpus building completed!")
    logger.info("=" * 80)


if __name__ == "__main__":
    fire.Fire(main)
