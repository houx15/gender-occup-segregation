#!/usr/bin/env python3
"""
Build time-sliced corpora from Chinese Google 5-gram data.

Usage:
    python build_corpora.py --config=config/config.yml
    python build_corpora.py --config=config/config.yml --slice=1940_1949
    python build_corpora.py --config=config/config.yml --overwrite
"""

import logging
import sys
from pathlib import Path
from typing import List, Tuple
from collections import defaultdict
import yaml
import fire
import re


def setup_logging(log_dir: Path) -> logging.Logger:
    """Configure logging to both file and console."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "build_corpora.log"

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def generate_time_slices(start_year: int, end_year: int, window_size: int, step_size: int) -> List[Tuple[int, int]]:
    """Generate time slice windows based on configuration."""
    slices = []
    current_start = start_year

    while current_start <= end_year:
        current_end = min(current_start + window_size - 1, end_year)
        slices.append((current_start, current_end))
        current_start += step_size

        if current_start > end_year:
            break

    return slices


CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")

def clean_ngram(ngram: str):
    tokens = ngram.split()
    clean_tokens = []
    for t in tokens:
        t = "".join(CHINESE_RE.findall(t))
        if t:
            clean_tokens.append(t)
    if len(clean_tokens) <= 1:
        return None
    return " ".join(clean_tokens) if len(clean_tokens) > 0 else None


def parse_ngram_line_v3(line: str):
    """
    Parse a v3 Chinese syntactic ngram line.

    Return list of tuples: (ngram_string, year, match_count)
    """
    parts = line.strip().split('\t')
    if len(parts) < 2:
        return []
    ngram = parts[0]                # e.g., "改革_VERB 开放_NOUN 促进_VERB"
    ngram = clean_ngram(ngram)
    if not ngram:
        return []
    year_triplets = parts[1:]       # e.g., ["1980,12,12", "1981,14,14"]
    result = []
    for yc in year_triplets:
        try:
            year, count1, count2 = yc.split(',')
            result.append((ngram, int(year), int(count1)))
        except:
            continue
    return result


def process_ngram_file(
    file_path: Path,
    time_slices,
    config,
    logger,
):
    min_count = config['corpus']['min_count_threshold']
    use_counts = config['corpus']['use_counts']
    corpora_dir = Path(config['paths']['corpora_dir'])

    logger.info(f"Processing {file_path.name}...")

    lines_processed = 0
    lines_included = defaultdict(int)
    file_index = file_path.name.split("-")[1]

    write_buffer = defaultdict(set)
    largest_buffer = 10000

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            lines_processed += 1
            entries = parse_ngram_line_v3(line)  # NEW
            if not entries:
                continue
            for ngram_text, year, match_count in entries:
                if match_count < min_count:
                    continue
                matched_slices = set()
                for start_year, end_year in time_slices:
                    if start_year <= year <= end_year:
                        matched_slices.add(f"{start_year}_{end_year}")
                
                for slice_name in matched_slices:
                    write_buffer[slice_name].add(ngram_text)
                    if len(write_buffer[slice_name]) > largest_buffer:
                        with open(corpora_dir / slice_name / f"corpus_{file_index}.txt", 'a', encoding='utf-8') as f:
                            f.write("\n".join(list(write_buffer[slice_name])) + "\n")
                        write_buffer[slice_name] = set()
                    lines_included[slice_name] += 1
            
            if lines_processed % 1000000 == 0:
                logger.info(f"  Processed {lines_processed:,} lines from {file_path.name}")
    
    for slice_name, buffer in write_buffer.items():
        if len(buffer) > 0:
            with open(corpora_dir / slice_name / f"corpus_{file_index}.txt", 'a', encoding='utf-8') as f:
                f.write("\n".join(list(buffer)) + "\n")

    logger.info(f"Completed {file_path.name}: {lines_processed:,} lines processed")
    for slice_name, count in lines_included.items():
        logger.info(f"  {slice_name}: {count:,} n-grams included")


def build_corpora(config_data: dict, logger: logging.Logger, specific_slice: str = None, overwrite: bool = False, file_name: str = None) -> None:
    """Build all time-sliced corpora."""
    time_slices_config = config_data['time_slices']
    time_slices = generate_time_slices(
        time_slices_config['start_year'],
        time_slices_config['end_year'],
        time_slices_config['window_size'],
        time_slices_config['step_size']
    )

    logger.info(f"Generated {len(time_slices)} time slices:")
    for start, end in time_slices:
        logger.info(f"  {start}-{end}")

    if specific_slice:
        start, end = map(int, specific_slice.split('_'))
        time_slices = [(start, end)]
        logger.info(f"\nBuilding only slice: {start}-{end}")

    
    decompressed_dir = Path(config_data['paths']['decompressed_dir'])
    if file_name:
        ngram_files = [decompressed_dir / file_name]
    else:
        ngram_files = sorted(decompressed_dir.glob("googlebooks-chi-sim-all-5gram-*"))

    logger.info(f"\nFound {len(ngram_files)} n-gram files to process")

    for ngram_file in ngram_files:
        process_ngram_file(ngram_file, time_slices, config_data, logger)

def main(file_name=None, config='config/config.yml', slice=None, overwrite=False):
    """
    Build time-sliced corpora from Chinese Google 5-gram data.

    Args:
        config: Path to configuration file
        slice: Build only a specific slice (format: 1940_1949)
        overwrite: Overwrite existing corpora
    """
    config_data = load_config(config)
    log_dir = Path(config_data['paths']['log_dir'])
    logger = setup_logging(log_dir)

    logger.info("="*80)
    logger.info("Starting corpus building")
    logger.info("="*80)

    build_corpora(config_data, logger, specific_slice=slice, overwrite=overwrite, file_name=file_name)

    logger.info("\n" + "="*80)
    logger.info("Corpus building completed!")
    logger.info("="*80)

    return 0



if __name__ == "__main__":
    fire.Fire(main)
