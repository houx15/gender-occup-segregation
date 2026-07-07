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


def corpus_signature(config) -> str:
    """A string fingerprint of the corpus params that determine slice content.

    Stored inside each shard's ``.done`` marker so a marker is trusted only
    when the *current* config would produce the same corpus. Switching
    weight_mode (e.g. presence → per_year_capped) or changing the cap/seed
    invalidates old markers, forcing a rebuild rather than silent reuse of
    stale data.
    """
    c = config['corpus']
    weight_mode = c.get('weight_mode', 'presence')
    parts = [f"weight_mode={weight_mode}", f"min_count={c['min_count_threshold']}"]
    if weight_mode == "per_year_capped":
        parts.append(f"cap={int(c.get('per_year_token_cap', 1_000_000_000))}")
        parts.append(f"rng_seed={int(c.get('rng_seed', 0))}")
        # year_total now comes from data-derived totalcounts, not the shipped
        # totalcounts-5. Tag the signature so any marker written under the old
        # (under-capped) denominator is distrusted → forces a clean rebuild.
        parts.append("ytsrc=derived")
        # Storage is now COMPACT (ngram<TAB>count) rather than expanded (one
        # physical line per emitted copy). Tag it so a marker from the old
        # expanded build is distrusted → the 85G slice is rebuilt, not reused.
        parts.append("fmt=compact")
    return ";".join(parts)


def _shard_marker(corpora_dir: Path, slice_name: str, file_index: str) -> Path:
    """Completion sentinel for one shard's contribution to one slice.

    Leading dot keeps it out of the ``corpus_*`` / ``corpus_*.txt`` globs used
    by train_embeddings.py and dataset_stats.py.
    """
    return corpora_dir / slice_name / f".corpus_{file_index}.done"


def shard_file_index(shard_path: Path) -> str:
    """Derive the shard index used in ``corpus_{index}.txt`` from a shard name.

    Mirrors process_ngram_file: ``5-00042-of-00105[.gz]`` → ``"00042"``.
    """
    return shard_path.name.split("-")[1]


DERIVED_TOTALCOUNTS_NAME = "totalcounts-5.derived"


def accumulate_year_total(file_path, min_count, year_total):
    """Sum kept-ngram ``match_count`` per year from one decompressed shard.

    Uses the SAME keep filter as emission — ``parse_ngram_line_v3`` (which applies
    ``clean_ngram``: ≥2 Chinese tokens) then ``match_count >= min_count`` — so the
    per-year totals are a *self-consistent* denominator for the cap: with
    ``scale = min(1, cap/year_total)`` the emitted lines for a year become
    ``min(Σ_kept match_count, cap)``. Mutates and returns ``year_total``.
    """
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            for _ngram, year, match_count in parse_ngram_line_v3(line):
                if match_count < min_count:
                    continue
                year_total[year] = year_total.get(year, 0) + match_count
    return year_total


def build_data_totalcounts(config, logger):
    """Compute per-year kept-match totals across ALL shards and cache them.

    The shipped ``totalcounts-5`` undercounts relative to what we emit — its
    per-year values sit far below ``Σ match_count`` of the kept 5-grams, so
    ``scale = cap / year_total ≈ 1`` and the cap never bites (a 10-year slice
    blew past ``10·cap`` by ~6.7×). We instead derive the denominator from the
    data itself and cache it as ``totalcounts-5.derived`` (one tab-separated line
    of ``year,count`` cells, readable by ``load_totalcounts``). Computed once;
    reused by every per-slice build.
    """
    raw_ngram_dir = Path(config['paths']['raw_ngram_dir'])
    decompressed_dir = Path(config['paths']['decompressed_dir'])
    min_count = config['corpus']['min_count_threshold']
    os.makedirs(decompressed_dir, exist_ok=True)

    shards = sorted(raw_ngram_dir.glob("5-*-of-00105.gz"))
    if not shards:
        raise FileNotFoundError(
            f"No 5-*-of-00105.gz shards in {raw_ngram_dir}; cannot derive totalcounts"
        )
    logger.info(f"Deriving per-year totalcounts from {len(shards)} shards (one-time)...")
    year_total: dict = {}
    for gz in shards:
        ngram_file = decompressed_dir / gz.stem
        ok, msg = decompress_file(gz, ngram_file, logger)
        if not ok:
            raise RuntimeError(f"totalcounts derive failed on {gz.name}: {msg}")
        accumulate_year_total(ngram_file, min_count, year_total)
        os.remove(ngram_file)

    derived = raw_ngram_dir / DERIVED_TOTALCOUNTS_NAME
    cells = "\t".join(f"{y},{c}" for y, c in sorted(year_total.items()))
    tmp = derived.with_name(derived.name + ".tmp")
    tmp.write_text(cells + "\n", encoding="utf-8")
    tmp.replace(derived)  # atomic — a concurrent reader never sees a partial file
    logger.info(f"Wrote {derived} ({len(year_total)} years)")
    return year_total


def load_data_totalcounts(config, logger):
    """Load the derived per-year totals, computing+caching them if absent."""
    derived = Path(config['paths']['raw_ngram_dir']) / DERIVED_TOTALCOUNTS_NAME
    if derived.exists():
        year_total = load_totalcounts(derived)
        logger.info(
            f"Loaded derived totalcounts ({len(year_total)} years) from {derived}"
        )
        return year_total
    return build_data_totalcounts(config, logger)


def process_ngram_file(file_path, time_slices, config, logger, year_total=None):
    """Process a single ngram file and write to time-slice corpus files.

    Dispatches on ``config['corpus']['weight_mode']`` (default ``"presence"``):
      - ``"presence"``: dedup-per-slice via set (one bare-ngram line per unique
        ngram per slice).
      - ``"per_year_capped"``: HistWords-style (Hamilton et al. 2016, Appendix A).
        For each (ngram, year, match_count) row, scale = min(1, cap / year_total[year]);
        n_emit = floor(match_count * scale) + Bernoulli(frac(match_count * scale)).
        Requires ``year_total: dict[int, int]`` mapping year → total match_count.
        Output is COMPACT: one ``ngram<TAB>count`` line per unique ngram per slice
        (count = Σ_year n_emit), NOT ``count`` repeated lines — so disk stays at
        type-size. train_embeddings' CorpusIterator(expand_counts=True) re-expands
        each line to ``count`` sentences at read time.
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
    file_index = shard_file_index(file_path)
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
            if weight_mode == "presence":
                for ngram_text, year, match_count in entries:
                    if match_count < min_count:
                        continue
                    for start_year, end_year in time_slices:
                        if start_year <= year <= end_year:
                            slice_name = f"{start_year}_{end_year}"
                            write_buffer[slice_name].add(ngram_text)
                            lines_emitted[slice_name] += 1
                            if len(write_buffer[slice_name]) > largest_buffer:
                                _flush(slice_name)
            else:  # per_year_capped — COMPACT: one "ngram<TAB>count" line per
                   # unique ngram per slice (count = Σ_year n_emit). Storing the
                   # count instead of `count` repeated lines keeps disk at
                   # type-size; train_embeddings expands it at read time.
                ngram_text = entries[0][0]  # v3: one ngram per line, all entries share it
                slice_count: dict = defaultdict(int)
                for _ng, year, match_count in entries:
                    if match_count < min_count:
                        continue
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
                    for start_year, end_year in time_slices:
                        if start_year <= year <= end_year:
                            slice_count[f"{start_year}_{end_year}"] += n_emit
                for slice_name, cnt in slice_count.items():
                    write_buffer[slice_name].append(f"{ngram_text}\t{cnt}")
                    lines_emitted[slice_name] += cnt
                    if len(write_buffer[slice_name]) > largest_buffer:
                        _flush(slice_name)
            if lines_processed % 1000000 == 0:
                logger.info(f"  Processed {lines_processed:,} lines from {file_path.name}")

    for slice_name in list(write_buffer.keys()):
        _flush(slice_name)

    logger.info(f"Completed {file_path.name}: {lines_processed:,} lines processed")
    for slice_name, count in lines_emitted.items():
        logger.info(f"  {slice_name}: {count:,} n-gram emissions")


def resolve_specific_slice(specific_slice, time_slices):
    """Look up the canonical (start, end) pair for a --slice CLI argument.

    Accepts either a 4-digit start year (int — the CLI-friendly form that
    dodges Fire's PEP 515 coercion of ``"1940_1949"`` → int 19401949) or a
    ``"YYYY_YYYY"`` string. The slice list from the profile is
    authoritative — a start year that isn't in it raises rather than
    silently building a bespoke window.

    Returns a single-element list ``[(start, end)]`` to slot back into the
    caller's ``time_slices`` variable.
    """
    if isinstance(specific_slice, int):
        start_year = specific_slice
    elif isinstance(specific_slice, str):
        start_year = int(specific_slice.split('_')[0])
    else:
        raise TypeError(
            f"--slice must be int or str, got {type(specific_slice).__name__}"
        )
    matching = [(s, e) for (s, e) in time_slices if s == start_year]
    if not matching:
        raise ValueError(
            f"--slice={specific_slice!r} does not match any slice start year "
            f"in this profile's time_slices. Valid start years: "
            f"{[s for (s, _) in time_slices]}"
        )
    return matching


def build_corpora(config, logger, specific_slice=None, file_name=None):
    """Build all time-sliced corpora from ngram data."""
    ts_config = config['time_slices']
    time_slices = generate_time_slices(
        ts_config['start_year'], ts_config['end_year'],
        ts_config['window_size'], ts_config['step_size']
    )
    logger.info(f"Generated {len(time_slices)} time slices")

    if specific_slice is not None:
        time_slices = resolve_specific_slice(specific_slice, time_slices)

    decompressed_dir = Path(config['paths']['decompressed_dir'])
    raw_ngram_dir = Path(config['paths']['raw_ngram_dir'])
    corpora_dir = Path(config['paths']['corpora_dir'])
    signature = corpus_signature(config)
    decompress = True

    year_total = None
    if config['corpus'].get('weight_mode') == 'per_year_capped':
        year_total = load_data_totalcounts(config, logger)

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

    all_slices = [f"{s}_{e}" for (s, e) in time_slices]

    for single_zip in ngram_zips:
        file_index = shard_file_index(single_zip)

        # Per-shard resume: a slice is "done" for this shard only when its
        # marker exists AND was written under the current corpus_signature.
        # Slices already done are reused; the rest are (re)built. A corpus file
        # without a trusted marker is unverified (partial crash, or stale config)
        # and gets deleted before rebuild so we never append onto old data.
        pending = [
            sl for sl in all_slices
            if not (
                _shard_marker(corpora_dir, sl, file_index).exists()
                and _shard_marker(corpora_dir, sl, file_index)
                    .read_text(encoding="utf-8").strip() == signature
            )
        ]
        if not pending:
            logger.info(
                f"Shard {file_index}: complete for all {len(all_slices)} "
                f"active slice(s); reusing existing corpora"
            )
            continue

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

        # Drop any unverified/partial output for pending slices so the rebuild
        # writes clean files instead of appending onto leftovers.
        for sl in pending:
            partial = corpora_dir / sl / f"corpus_{file_index}.txt"
            if partial.exists():
                logger.info(
                    f"Shard {file_index} slice {sl}: removing unverified corpus "
                    f"file before rebuild"
                )
                partial.unlink()

        pending_slices = [(s, e) for (s, e) in time_slices if f"{s}_{e}" in pending]
        process_ngram_file(ngram_file, pending_slices, config, logger, year_total=year_total)

        # Mark each rebuilt slice done (after a clean pass). Written even when a
        # slice received zero lines from this shard, so we don't reprocess it.
        for sl in pending:
            os.makedirs(corpora_dir / sl, exist_ok=True)
            _shard_marker(corpora_dir, sl, file_index).write_text(
                signature, encoding="utf-8"
            )

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
