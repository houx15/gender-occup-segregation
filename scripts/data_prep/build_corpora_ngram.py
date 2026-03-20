#!/usr/bin/env python3
"""
Build time-sliced corpora from Chinese Google 5-gram data.

Usage:
    python -m scripts.data_prep.build_corpora_ngram --config=config/config.yml
    python -m scripts.data_prep.build_corpora_ngram --config=config/config.yml --slice=1940_1949
"""

import os
import re
import gzip
import shutil
from pathlib import Path
from typing import List, Tuple
from collections import defaultdict

import fire

from scripts.common.config_loader import load_config
from scripts.common.logging_utils import setup_logging


CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")


def decompress_file(gz_path: Path, output_path: Path, logger) -> Tuple[bool, str]:
    """Decompress a gzip file."""
    filename = gz_path.name
    if output_path.exists() and output_path.stat().st_size > 0:
        logger.info(f"Skipping decompression of {filename} (already exists)")
        return True, f"Already decompressed: {filename}"
    try:
        logger.info(f"Decompressing {filename}...")
        with gzip.open(gz_path, "rb") as f_in:
            with open(output_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        output_size = output_path.stat().st_size
        logger.info(f"Completed decompressing {filename} ({output_size:,} bytes)")
        return True, f"Decompressed: {filename}"
    except Exception as e:
        logger.error(f"Failed to decompress {filename}: {e}")
        if output_path.exists():
            output_path.unlink()
        return False, f"Failed: {filename} - {e}"


def generate_time_slices(start_year, end_year, window_size, step_size):
    """Generate time slice windows."""
    slices = []
    current_start = start_year
    while current_start <= end_year:
        current_end = min(current_start + window_size - 1, end_year)
        slices.append((current_start, current_end))
        current_start += step_size
        if current_start > end_year:
            break
    return slices


def clean_ngram(ngram: str):
    """Clean an ngram by extracting only Chinese characters."""
    tokens = ngram.split()
    clean_tokens = []
    for t in tokens:
        t = "".join(CHINESE_RE.findall(t))
        if t:
            clean_tokens.append(t)
    if len(clean_tokens) <= 1:
        return None
    return " ".join(clean_tokens)


def parse_ngram_line_v3(line: str):
    """Parse a v3 Chinese syntactic ngram line."""
    parts = line.strip().split('\t')
    if len(parts) < 2:
        return []
    ngram = clean_ngram(parts[0])
    if not ngram:
        return []
    result = []
    for yc in parts[1:]:
        try:
            year, count1, count2 = yc.split(',')
            result.append((ngram, int(year), int(count1)))
        except Exception:
            continue
    return result


def process_ngram_file(file_path, time_slices, config, logger):
    """Process a single ngram file and write to time-slice corpus files."""
    min_count = config['corpus']['min_count_threshold']
    corpora_dir = Path(config['paths']['corpora_dir'])
    os.makedirs(corpora_dir, exist_ok=True)

    logger.info(f"Processing {file_path.name}...")
    lines_processed = 0
    lines_included = defaultdict(int)
    file_index = file_path.name.split("-")[1]
    write_buffer = defaultdict(set)
    largest_buffer = 10000

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            lines_processed += 1
            entries = parse_ngram_line_v3(line)
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
                        os.makedirs(corpora_dir / slice_name, exist_ok=True)
                        with open(corpora_dir / slice_name / f"corpus_{file_index}.txt", 'a', encoding='utf-8') as out:
                            out.write("\n".join(list(write_buffer[slice_name])) + "\n")
                        write_buffer[slice_name] = set()
                    lines_included[slice_name] += 1
            if lines_processed % 1000000 == 0:
                logger.info(f"  Processed {lines_processed:,} lines from {file_path.name}")

    for slice_name, buffer in write_buffer.items():
        if buffer:
            os.makedirs(corpora_dir / slice_name, exist_ok=True)
            with open(corpora_dir / slice_name / f"corpus_{file_index}.txt", 'a', encoding='utf-8') as out:
                out.write("\n".join(list(buffer)) + "\n")

    logger.info(f"Completed {file_path.name}: {lines_processed:,} lines processed")
    for slice_name, count in lines_included.items():
        logger.info(f"  {slice_name}: {count:,} n-grams included")


def build_corpora(config, logger, specific_slice=None, file_name=None):
    """Build all time-sliced corpora from ngram data."""
    ts_config = config['time_slices']
    time_slices = generate_time_slices(
        ts_config['start_year'], ts_config['end_year'],
        ts_config['window_size'], ts_config['step_size']
    )
    logger.info(f"Generated {len(time_slices)} time slices")

    if specific_slice:
        start, end = map(int, specific_slice.split('_'))
        time_slices = [(start, end)]

    decompressed_dir = Path(config['paths']['decompressed_dir'])
    raw_ngram_dir = Path(config['paths']['raw_ngram_dir'])
    decompress = True

    if file_name:
        ngram_zips = [decompressed_dir / file_name]
        decompress = False
    else:
        ngram_zips = sorted(raw_ngram_dir.glob("5-*-of-00105.gz"))

    logger.info(f"Found {len(ngram_zips)} n-gram files to process")

    for single_zip in ngram_zips:
        if decompress:
            ngram_file = decompressed_dir / single_zip.stem
            decompress_file(single_zip, ngram_file, logger)
        else:
            ngram_file = single_zip
        process_ngram_file(ngram_file, time_slices, config, logger)
        if decompress:
            os.remove(ngram_file)


def main(file_name=None, config='config/config.yml', slice=None):
    """Build time-sliced corpora from Chinese Google 5-gram data."""
    config_data = load_config(config)
    logger = setup_logging(Path(config_data['paths']['log_dir']), "build_corpora_ngram.log")

    logger.info("=" * 80)
    logger.info("Starting ngram corpus building")
    logger.info("=" * 80)

    build_corpora(config_data, logger, specific_slice=slice, file_name=file_name)

    logger.info("=" * 80)
    logger.info("Ngram corpus building completed!")
    logger.info("=" * 80)


if __name__ == "__main__":
    fire.Fire(main)
