#!/usr/bin/env python3
"""
Build time-sliced corpora from Chinese Google 5-gram data.

Usage:
    python build_corpora_renminribao.py --config=config/renminribao.yml
    python build_corpora_renminribao.py --config=config/renminribao.yml --slice=1940_1949
    python build_corpora_renminribao.py --config=config/renminribao.yml --overwrite
"""

import os
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
    decompress = True
    raw_ngram_dir = Path(config_data['paths']['raw_ngram_dir'])
    if file_name:
        ngram_zips = [decompressed_dir / file_name]
        decompress = False
    else:
        ngram_zips = sorted(raw_ngram_dir.glob("5-*-of-00105.gz"))

    logger.info(f"\nFound {len(ngram_zips)} n-gram files to process")

    for single_zip in ngram_zips:
        if decompress:
            ngram_file = decompressed_dir / single_zip.stem
            decompress_file(single_zip, ngram_file, logger)
        else:
            ngram_file = single_zip
        process_ngram_file(ngram_file, time_slices, config_data, logger)

        # delete the ngram file
        if decompress:
            os.remove(ngram_file)

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
