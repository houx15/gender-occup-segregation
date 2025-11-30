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


def parse_ngram_line(line: str, delimiter: str, year_col: int, count_col: int) -> Tuple[str, int, int]:
    """Parse a single line from an n-gram file."""
    try:
        parts = line.strip().split(delimiter)
        if len(parts) < max(year_col, count_col) + 1:
            return None, None, None

        ngram_part = parts[0]
        year = int(parts[year_col])
        match_count = int(parts[count_col])

        return ngram_part, year, match_count

    except (ValueError, IndexError):
        return None, None, None


def process_ngram_file(
    file_path: Path,
    time_slices: List[Tuple[int, int]],
    output_files: dict,
    config: dict,
    logger: logging.Logger,
    stats: dict
) -> None:
    """Process a single n-gram file and write to appropriate time slice corpora."""
    delimiter = config['ngram']['delimiter']
    year_col = config['ngram']['year_column']
    count_col = config['ngram']['match_count_column']
    use_counts = config['corpus']['use_counts']
    min_count = config['corpus']['min_count_threshold']

    logger.info(f"Processing {file_path.name}...")

    lines_processed = 0
    lines_included = defaultdict(int)

    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                lines_processed += 1

                ngram_text, year, match_count = parse_ngram_line(line, delimiter, year_col, count_col)

                if ngram_text is None or year is None:
                    continue

                if match_count < min_count:
                    continue

                for start_year, end_year in time_slices:
                    if start_year <= year <= end_year:
                        slice_name = f"{start_year}_{end_year}"

                        if use_counts:
                            repetitions = min(match_count, 1000)
                            for _ in range(repetitions):
                                output_files[slice_name].write(ngram_text + '\n')
                                stats[slice_name]['tokens'] += len(ngram_text.split())
                        else:
                            output_files[slice_name].write(ngram_text + '\n')
                            stats[slice_name]['tokens'] += len(ngram_text.split())

                        lines_included[slice_name] += 1

                if lines_processed % 1000000 == 0:
                    logger.info(f"  Processed {lines_processed:,} lines from {file_path.name}")

    except Exception as e:
        logger.error(f"Error processing {file_path.name}: {e}")
        raise

    logger.info(f"Completed {file_path.name}: {lines_processed:,} lines processed")
    for slice_name, count in lines_included.items():
        logger.info(f"  {slice_name}: {count:,} n-grams included")


def build_corpora(config_data: dict, logger: logging.Logger, specific_slice: str = None, overwrite: bool = False) -> None:
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

    corpora_dir = Path(config_data['paths']['corpora_dir'])
    decompressed_dir = Path(config_data['paths']['decompressed_dir'])

    output_files = {}
    stats = {}

    for start_year, end_year in time_slices:
        slice_name = f"{start_year}_{end_year}"
        slice_dir = corpora_dir / slice_name
        slice_dir.mkdir(parents=True, exist_ok=True)

        corpus_file = slice_dir / "corpus.txt"

        if corpus_file.exists() and not overwrite:
            logger.warning(f"Corpus {slice_name} already exists, skipping (use --overwrite to regenerate)")
            continue

        output_files[slice_name] = open(corpus_file, 'w', encoding='utf-8')
        stats[slice_name] = {'lines': 0, 'tokens': 0}

    if not output_files:
        logger.info("No corpora to build (all exist or no slices selected)")
        return

    ngram_files = sorted(decompressed_dir.glob("*.txt"))
    if not ngram_files:
        ngram_files = sorted(decompressed_dir.glob("googlebooks-chi-sim-all-5gram-*"))

    if not ngram_files:
        logger.error(f"No n-gram files found in {decompressed_dir}")
        for f in output_files.values():
            f.close()
        return

    logger.info(f"\nFound {len(ngram_files)} n-gram files to process")

    for ngram_file in ngram_files:
        process_ngram_file(ngram_file, time_slices, output_files, config_data, logger, stats)

    for slice_name, f in output_files.items():
        f.close()
        stats[slice_name]['lines'] = sum(1 for _ in open(f.name, 'r', encoding='utf-8'))

    logger.info("\n" + "="*80)
    logger.info("Corpus Building Summary:")
    logger.info("="*80)

    for slice_name in sorted(stats.keys()):
        logger.info(f"\n{slice_name}:")
        logger.info(f"  Lines: {stats[slice_name]['lines']:,}")
        logger.info(f"  Approximate tokens: {stats[slice_name]['tokens']:,}")


def main(config='config/config.yml', slice=None, overwrite=False):
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

    build_corpora(config_data, logger, specific_slice=slice, overwrite=overwrite)

    logger.info("\n" + "="*80)
    logger.info("Corpus building completed!")
    logger.info("="*80)

    return 0


if __name__ == "__main__":
    fire.Fire(main)
