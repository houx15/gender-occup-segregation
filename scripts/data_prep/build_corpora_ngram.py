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

import numpy as np
import fire

from scripts.common.config_loader import load_config
from scripts.common.logging_utils import setup_logging
from scripts.data_prep.ngram_totalcounts import load_totalcounts


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


VALID_WEIGHT_MODES = {"presence", "per_year_capped"}


def process_ngram_file(file_path, time_slices, config, logger, year_total=None):
    """Process a single ngram file and write to time-slice corpus files.

    Dispatches on ``config['corpus']['weight_mode']`` (default ``"presence"``):
      - ``"presence"``: dedup-per-slice via set (one line per unique ngram per slice).
      - ``"per_year_capped"``: HistWords-style (Hamilton et al. 2016, Appendix A).
        For each (ngram, year, match_count) row, scale = min(1, cap / year_total[year]);
        n_emit = floor(match_count * scale) + Bernoulli(frac(match_count * scale)).
        Requires ``year_total: dict[int, int]`` mapping year → total match_count.
    """
    corpus_cfg = config['corpus']
    min_count = corpus_cfg['min_count_threshold']
    weight_mode = corpus_cfg.get('weight_mode', 'presence')
    if weight_mode not in VALID_WEIGHT_MODES:
        raise ValueError(
            f"Invalid corpus.weight_mode={weight_mode!r}; "
            f"expected one of {sorted(VALID_WEIGHT_MODES)}"
        )

    if weight_mode == "per_year_capped":
        if year_total is None:
            raise ValueError(
                "per_year_capped weight_mode requires year_total argument "
                "(mapping year -> total match_count from totalcounts-5)"
            )
        if 'per_year_token_cap' not in corpus_cfg:
            logger.warning(
                "corpus.per_year_token_cap missing; defaulting to 1_000_000_000 (HistWords default)"
            )
        if 'rng_seed' not in corpus_cfg:
            logger.info("corpus.rng_seed missing; defaulting to 0")
        cap = int(corpus_cfg.get('per_year_token_cap', 1_000_000_000))
        seed = int(corpus_cfg.get('rng_seed', 0))
        rng = np.random.default_rng(seed)

    corpora_dir = Path(config['paths']['corpora_dir'])
    os.makedirs(corpora_dir, exist_ok=True)

    logger.info(f"Processing {file_path.name} (weight_mode={weight_mode})...")
    lines_processed = 0
    lines_emitted = defaultdict(int)
    file_index = file_path.name.split("-")[1]
    write_buffer: dict = defaultdict(set) if weight_mode == "presence" else defaultdict(list)
    largest_buffer = 10000

    def _flush(slice_name: str):
        buf = write_buffer[slice_name]
        if not buf:
            return
        os.makedirs(corpora_dir / slice_name, exist_ok=True)
        with open(corpora_dir / slice_name / f"corpus_{file_index}.txt", 'a', encoding='utf-8') as out:
            out.write("\n".join(list(buf)) + "\n")
        write_buffer[slice_name] = set() if weight_mode == "presence" else []

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            lines_processed += 1
            entries = parse_ngram_line_v3(line)
            if not entries:
                continue
            for ngram_text, year, match_count in entries:
                if match_count < min_count:
                    continue
                if weight_mode == "per_year_capped":
                    if year not in year_total:
                        raise KeyError(
                            f"Year {year} missing from totalcounts-5 (raw_ngram_dir/totalcounts-5)"
                        )
                    if year_total[year] == 0:
                        raise ValueError(
                            f"Year {year} has year_total=0 in totalcounts-5; cannot compute scale"
                        )
                    scale = min(1.0, cap / year_total[year])
                    expected = match_count * scale
                    n_floor = int(expected)
                    frac = expected - n_floor
                    n_emit = n_floor + (1 if rng.random() < frac else 0)
                    if n_emit <= 0:
                        continue
                matched_slices = set()
                for start_year, end_year in time_slices:
                    if start_year <= year <= end_year:
                        matched_slices.add(f"{start_year}_{end_year}")
                for slice_name in matched_slices:
                    if weight_mode == "presence":
                        write_buffer[slice_name].add(ngram_text)
                        lines_emitted[slice_name] += 1
                    else:  # per_year_capped
                        write_buffer[slice_name].extend([ngram_text] * n_emit)
                        lines_emitted[slice_name] += n_emit
                    if len(write_buffer[slice_name]) > largest_buffer:
                        _flush(slice_name)
            if lines_processed % 1000000 == 0:
                logger.info(f"  Processed {lines_processed:,} lines from {file_path.name}")

    for slice_name in list(write_buffer.keys()):
        _flush(slice_name)

    logger.info(f"Completed {file_path.name}: {lines_processed:,} lines processed")
    for slice_name, count in lines_emitted.items():
        logger.info(f"  {slice_name}: {count:,} n-gram emissions")


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

    year_total = None
    if config['corpus'].get('weight_mode') == 'per_year_capped':
        totalcounts_path = raw_ngram_dir / 'totalcounts-5'
        year_total = load_totalcounts(totalcounts_path)
        logger.info(f"Loaded totalcounts-5 ({len(year_total)} years)")

    # Ensure decompression target dir exists — without this, decompress_file's
    # open(output_path, 'wb') raises FileNotFoundError on the first run of a
    # fresh profile (e.g. *_weighted dirs).
    os.makedirs(decompressed_dir, exist_ok=True)

    if file_name:
        ngram_zips = [decompressed_dir / file_name]
        decompress = False
    else:
        ngram_zips = sorted(raw_ngram_dir.glob("5-*-of-00105.gz"))

    logger.info(f"Found {len(ngram_zips)} n-gram files to process")

    for single_zip in ngram_zips:
        if decompress:
            ngram_file = decompressed_dir / single_zip.stem
            ok, msg = decompress_file(single_zip, ngram_file, logger)
            if not ok:
                # Skip processing — decompression failed (returned False).
                # Without this guard, process_ngram_file silently reads 0 lines
                # from the missing file, then os.remove raises FileNotFoundError.
                logger.error(f"Skipping {single_zip.name}: {msg}")
                continue
        else:
            ngram_file = single_zip
        process_ngram_file(ngram_file, time_slices, config, logger, year_total=year_total)
        if decompress:
            os.remove(ngram_file)


def main(file_name: str = None, config: str = 'config/config.yml', slice: str = None):
    """Build time-sliced corpora from Chinese Google 5-gram data.

    The ``: str`` annotations are load-bearing: Fire auto-coerces CLI args
    using ``ast.literal_eval``, and PEP 515 makes ``1940_1949`` a valid int
    literal (=> ``19401949``). Without the annotation, ``--slice=1940_1949``
    would arrive as an int and crash ``specific_slice.split('_')`` later.
    """
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
