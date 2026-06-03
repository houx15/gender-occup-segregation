# China Ngram HistWords-Style Per-Year Subsampling — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Spec:** [`docs/superpowers/specs/2026-06-03-china-ngram-histwords-style-subsampling-design.md`](../specs/2026-06-03-china-ngram-histwords-style-subsampling-design.md)

**Goal:** Replace the methods-wrong `capped_repetition` weight_mode with HistWords-style `per_year_capped` subsampling, backed by a new `ngram_totalcounts.py` parser; swap the config profile and SLURM defaults accordingly.

**Architecture:** New isolated module `scripts/data_prep/ngram_totalcounts.py` parses `raw_ngrams/totalcounts-5` into `dict[int, int]`. `scripts/data_prep/build_corpora_ngram.py` loses its `capped_repetition` branch and gains a `per_year_capped` branch that consumes a `year_total` dict passed in by `build_corpora`. Bernoulli-on-fractional-part sampling via a seeded `numpy.random.Generator`. New profile + dirs ship under the `*_subsampled` name; old `_weighted` artifacts are deleted (operational cleanup on Princeton is out-of-scope for the code plan).

**Tech Stack:** Python 3, pytest, numpy (already in repo dependencies via gensim). No new packages.

---

## File structure

```
scripts/data_prep/ngram_totalcounts.py                              — CREATE
scripts/data_prep/build_corpora_ngram.py                            — MODIFY (drop capped_repetition; add per_year_capped; wire totalcounts loading)
config/profiles/garg_weat_china_ngram_subsampled.yml                — CREATE
config/profiles/garg_weat_china_ngram_weighted.yml                  — DELETE
slurm/build_train_china_ngram_weighted_per_slice.slurm              — RENAME → ..._subsampled_per_slice.slurm + retarget config
slurm/garg_weat_zh.slurm                                            — MODIFY (DEFAULT_CONFIGS flip)
slurm/describe_dataset_zh.slurm                                     — MODIFY (DEFAULT_CONFIGS flip)
tests/test_ngram_totalcounts.py                                     — CREATE
tests/test_build_corpora_ngram.py                                   — MODIFY (drop capped_repetition tests; add per_year_capped tests)
```

---

## Task 1: `ngram_totalcounts.py` module + tests

**Files:**
- Create: `scripts/data_prep/ngram_totalcounts.py`
- Create: `tests/test_ngram_totalcounts.py`

The Google Books v3 `totalcounts-5` file is a single line, tab-separated cells, each cell is `year,match_count,page_count,volume_count`. The parser ignores everything but year and match_count.

- [ ] **Step 1: Write the failing tests**

Write to `tests/test_ngram_totalcounts.py`:

```python
"""Tests for the Google Books v3 totalcounts-5 parser."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.data_prep.ngram_totalcounts import (
    parse_totalcounts_text,
    load_totalcounts,
)


class TestParseTotalcountsText:
    def test_parses_year_to_match_count(self):
        text = "1500,117,6,1\t1501,200,8,2\t1502,50,3,1"
        result = parse_totalcounts_text(text)
        assert result == {1500: 117, 1501: 200, 1502: 50}

    def test_empty_input_returns_empty_dict(self):
        assert parse_totalcounts_text("") == {}
        assert parse_totalcounts_text("\n") == {}

    def test_malformed_cells_are_skipped(self):
        # First cell malformed (missing words), second cell good, third cell non-int year.
        text = "1500\t1501,200,8,2\tABCD,300,10,3"
        result = parse_totalcounts_text(text)
        assert result == {1501: 200}

    def test_trailing_whitespace_and_newlines_handled(self):
        text = "1500,117,6,1\t1501,200,8,2\n"
        assert parse_totalcounts_text(text) == {1500: 117, 1501: 200}


class TestLoadTotalcounts:
    def test_reads_file_from_disk(self, tmp_path: Path):
        p = tmp_path / "totalcounts-5"
        p.write_text("1500,117,6,1\t1501,200,8,2\n", encoding="utf-8")
        assert load_totalcounts(p) == {1500: 117, 1501: 200}

    def test_missing_file_raises_file_not_found(self, tmp_path: Path):
        p = tmp_path / "does-not-exist"
        with pytest.raises(FileNotFoundError):
            load_totalcounts(p)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_ngram_totalcounts.py -v`
Expected: ImportError / ModuleNotFoundError on `scripts.data_prep.ngram_totalcounts` (module doesn't exist yet).

- [ ] **Step 3: Write minimal implementation**

Write to `scripts/data_prep/ngram_totalcounts.py`:

```python
"""Parse the Google Books v3 totalcounts-5 file into a per-year word-count dict.

The file is a single line of tab-separated cells; each cell is the comma-separated
tuple `year,match_count,page_count,volume_count`. We keep only year → match_count.
"""

from __future__ import annotations

from pathlib import Path


def parse_totalcounts_text(text: str) -> dict[int, int]:
    """Parse the raw text of a v3 totalcounts file. Malformed cells are skipped."""
    result: dict[int, int] = {}
    for cell in text.strip().split("\t"):
        parts = cell.split(",")
        if len(parts) < 2:
            continue
        try:
            year = int(parts[0])
            match_count = int(parts[1])
        except ValueError:
            continue
        result[year] = match_count
    return result


def load_totalcounts(path: Path) -> dict[int, int]:
    """Read a totalcounts-5 file from disk and parse it."""
    return parse_totalcounts_text(Path(path).read_text(encoding="utf-8"))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_ngram_totalcounts.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/data_prep/ngram_totalcounts.py tests/test_ngram_totalcounts.py
git commit -m "feat(ngram): parse google v3 totalcounts-5 → {year: match_count}"
```

---

## Task 2: Delete `capped_repetition` weight_mode

**Files:**
- Modify: `scripts/data_prep/build_corpora_ngram.py`
- Modify: `tests/test_build_corpora_ngram.py`

Clean removal first; Task 3 adds the new mode against a clean baseline.

- [ ] **Step 1: Delete the `capped_repetition` tests from the test file**

In `tests/test_build_corpora_ngram.py`, delete two whole classes — `TestCappedRepetition` (lines roughly 69–112) and `TestRepeatCapEdge` (lines roughly 125–133). Keep `TestPresenceModeDefault` and `TestInvalidWeightMode`. The file should now have exactly two test classes.

- [ ] **Step 2: Run remaining tests to verify they pass against the unchanged source**

Run: `pytest tests/test_build_corpora_ngram.py -v`
Expected: 2 passed (`test_no_weight_mode_key_preserves_existing_behavior`, `test_raises_value_error`).

- [ ] **Step 3: Remove `capped_repetition` branch from the source**

In `scripts/data_prep/build_corpora_ngram.py`:

Replace the constant on line 92:
```python
VALID_WEIGHT_MODES = {"presence", "capped_repetition"}
```
with:
```python
VALID_WEIGHT_MODES = {"presence"}
```

Replace the `process_ngram_file` function body (lines 95–164) with a presence-only version. The new function:

```python
def process_ngram_file(file_path, time_slices, config, logger):
    """Process a single ngram file in presence-only mode.

    Dedup-per-slice via set: one corpus line per unique ngram per slice,
    regardless of match_count (above min_count_threshold).
    """
    corpus_cfg = config['corpus']
    min_count = corpus_cfg['min_count_threshold']
    weight_mode = corpus_cfg.get('weight_mode', 'presence')
    if weight_mode not in VALID_WEIGHT_MODES:
        raise ValueError(
            f"Invalid corpus.weight_mode={weight_mode!r}; "
            f"expected one of {sorted(VALID_WEIGHT_MODES)}"
        )

    corpora_dir = Path(config['paths']['corpora_dir'])
    os.makedirs(corpora_dir, exist_ok=True)

    logger.info(f"Processing {file_path.name} (weight_mode={weight_mode})...")
    lines_processed = 0
    lines_emitted = defaultdict(int)
    file_index = file_path.name.split("-")[1]
    write_buffer: dict = defaultdict(set)
    largest_buffer = 10000

    def _flush(slice_name: str):
        buf = write_buffer[slice_name]
        if not buf:
            return
        os.makedirs(corpora_dir / slice_name, exist_ok=True)
        with open(corpora_dir / slice_name / f"corpus_{file_index}.txt", 'a', encoding='utf-8') as out:
            out.write("\n".join(list(buf)) + "\n")
        write_buffer[slice_name] = set()

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
                    lines_emitted[slice_name] += 1
                    if len(write_buffer[slice_name]) > largest_buffer:
                        _flush(slice_name)
            if lines_processed % 1000000 == 0:
                logger.info(f"  Processed {lines_processed:,} lines from {file_path.name}")

    for slice_name in list(write_buffer.keys()):
        _flush(slice_name)

    logger.info(f"Completed {file_path.name}: {lines_processed:,} lines processed")
    for slice_name, count in lines_emitted.items():
        logger.info(f"  {slice_name}: {count:,} n-gram emissions")
```

- [ ] **Step 4: Run tests to verify presence-only still works and bad weight_mode still raises**

Run: `pytest tests/test_build_corpora_ngram.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/data_prep/build_corpora_ngram.py tests/test_build_corpora_ngram.py
git commit -m "refactor(ngram): remove capped_repetition weight_mode (methods-wrong, superseded by per_year_capped)"
```

---

## Task 3: Add `per_year_capped` weight_mode end-to-end

**Files:**
- Modify: `scripts/data_prep/build_corpora_ngram.py`
- Modify: `tests/test_build_corpora_ngram.py`

The new mode requires a `year_total` argument to `process_ngram_file`. `build_corpora` loads it once before the shard loop.

- [ ] **Step 1: Write the failing tests**

Append the following to `tests/test_build_corpora_ngram.py`:

```python
class TestPerYearCapped:
    def test_pass_through_when_year_total_below_cap(self, tmp_path):
        # year_total = 5e7 (below cap 1e8) → scale = 1.0 → emit exactly match_count copies.
        f = _make_ngram_file(tmp_path, "5-00000-of-00105", [
            _line(NGRAM, 1942, match_count=7),
        ])
        cfg = _config(tmp_path, corpus={
            "weight_mode": "per_year_capped",
            "per_year_token_cap": 100_000_000,
            "rng_seed": 42,
        })
        process_ngram_file(f, [(1940, 1949)], cfg, logger, year_total={1942: 50_000_000})
        lines = _read_corpus(Path(cfg["paths"]["corpora_dir"]), "1940_1949")
        assert lines == [NGRAM_CLEAN] * 7

    def test_scale_down_when_year_total_above_cap_is_deterministic(self, tmp_path):
        # year_total = 1e9, cap = 1e8 → scale = 0.1. match_count = 100 → expected = 10 (integer; no Bernoulli).
        f = _make_ngram_file(tmp_path, "5-00000-of-00105", [
            _line(NGRAM, 1942, match_count=100),
        ])
        cfg = _config(tmp_path, corpus={
            "weight_mode": "per_year_capped",
            "per_year_token_cap": 100_000_000,
            "rng_seed": 42,
        })
        process_ngram_file(f, [(1940, 1949)], cfg, logger, year_total={1942: 1_000_000_000})
        lines = _read_corpus(Path(cfg["paths"]["corpora_dir"]), "1940_1949")
        assert lines == [NGRAM_CLEAN] * 10

    def test_bernoulli_fractional_part_is_unbiased(self, tmp_path):
        # match_count=15, scale=0.1 → expected = 1.5 → n_emit ∈ {1, 2} per trial.
        # Across 200 seeds, empirical mean should land within 3σ of 1.5.
        # Var(Bernoulli(0.5)) = 0.25; SE over 200 trials ≈ sqrt(0.25/200) ≈ 0.0354. 3σ ≈ 0.106.
        n_emits = []
        for seed in range(200):
            sub = tmp_path / f"trial_{seed}"
            sub.mkdir()
            f = _make_ngram_file(sub, "5-00000-of-00105", [
                _line(NGRAM, 1942, match_count=15),
            ])
            cfg = _config(sub, corpus={
                "weight_mode": "per_year_capped",
                "per_year_token_cap": 100_000_000,
                "rng_seed": seed,
            })
            process_ngram_file(f, [(1940, 1949)], cfg, logger, year_total={1942: 1_500_000_000})
            n_emits.append(len(_read_corpus(Path(cfg["paths"]["corpora_dir"]), "1940_1949")))
        mean = sum(n_emits) / len(n_emits)
        assert 1.39 < mean < 1.61, f"Bernoulli looks biased: mean={mean:.3f} over 200 trials"
        # Also confirm both outcomes occur — guards against a stuck RNG.
        assert set(n_emits) == {1, 2}

    def test_missing_year_in_totalcounts_raises_key_error(self, tmp_path):
        f = _make_ngram_file(tmp_path, "5-00000-of-00105", [
            _line(NGRAM, 1942, match_count=10),
        ])
        cfg = _config(tmp_path, corpus={
            "weight_mode": "per_year_capped",
            "per_year_token_cap": 100_000_000,
            "rng_seed": 42,
        })
        with pytest.raises(KeyError, match="1942"):
            process_ngram_file(f, [(1940, 1949)], cfg, logger, year_total={1943: 1_000_000})

    def test_per_year_capped_without_year_total_raises_value_error(self, tmp_path):
        f = _make_ngram_file(tmp_path, "5-00000-of-00105", [
            _line(NGRAM, 1942, match_count=10),
        ])
        cfg = _config(tmp_path, corpus={
            "weight_mode": "per_year_capped",
            "per_year_token_cap": 100_000_000,
            "rng_seed": 42,
        })
        with pytest.raises(ValueError, match="year_total"):
            process_ngram_file(f, [(1940, 1949)], cfg, logger)  # year_total omitted

    def test_presence_mode_does_not_require_year_total(self, tmp_path):
        # Regression: presence mode should keep working with no year_total argument.
        f = _make_ngram_file(tmp_path, "5-00000-of-00105", [
            _line(NGRAM, 1942, match_count=5),
        ])
        cfg = _config(tmp_path)  # default → presence
        process_ngram_file(f, [(1940, 1949)], cfg, logger)  # no year_total kwarg
        assert _read_corpus(Path(cfg["paths"]["corpora_dir"]), "1940_1949") == [NGRAM_CLEAN]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_build_corpora_ngram.py::TestPerYearCapped -v`
Expected: All 6 fail — TypeError on unexpected `year_total` kwarg, ValueError on invalid weight_mode, etc.

- [ ] **Step 3: Implement the new mode in `process_ngram_file`**

In `scripts/data_prep/build_corpora_ngram.py`:

Add a numpy import at the top of the imports block (just after `from collections import defaultdict`):
```python
import numpy as np
```

Update the constant:
```python
VALID_WEIGHT_MODES = {"presence", "per_year_capped"}
```

Replace the `process_ngram_file` signature and body with:

```python
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
```

- [ ] **Step 4: Run new tests to verify they pass**

Run: `pytest tests/test_build_corpora_ngram.py -v`
Expected: 8 passed (2 from before + 6 new).

- [ ] **Step 5: Wire `year_total` loading into `build_corpora`**

In `scripts/data_prep/build_corpora_ngram.py`, add an import near the top:

```python
from scripts.data_prep.ngram_totalcounts import load_totalcounts
```

In the `build_corpora` function, after `decompress = True` (around line 182) and before the `os.makedirs(decompressed_dir, exist_ok=True)` line, add:

```python
    year_total = None
    if config['corpus'].get('weight_mode') == 'per_year_capped':
        totalcounts_path = raw_ngram_dir / 'totalcounts-5'
        year_total = load_totalcounts(totalcounts_path)
        logger.info(f"Loaded totalcounts-5 ({len(year_total)} years)")
```

Then change the `process_ngram_file(...)` call inside the shard loop to pass `year_total`:

```python
        process_ngram_file(ngram_file, time_slices, config, logger, year_total=year_total)
```

- [ ] **Step 6: Verify the full test suite still passes**

Run: `pytest tests/test_build_corpora_ngram.py tests/test_ngram_totalcounts.py -v`
Expected: 14 passed (8 + 6).

- [ ] **Step 7: Commit**

```bash
git add scripts/data_prep/build_corpora_ngram.py tests/test_build_corpora_ngram.py
git commit -m "feat(ngram): add per_year_capped weight_mode (HistWords-style subsampling)"
```

---

## Task 4: Profile swap — new `_subsampled` profile, delete `_weighted`

**Files:**
- Create: `config/profiles/garg_weat_china_ngram_subsampled.yml`
- Delete: `config/profiles/garg_weat_china_ngram_weighted.yml`

- [ ] **Step 1: Create the new profile**

Write to `config/profiles/garg_weat_china_ngram_subsampled.yml`:

```yaml
# config/profiles/garg_weat_china_ngram_subsampled.yml
# HistWords-style per-year-capped subsampling pipeline (Hamilton et al. 2016,
# Appendix A). For each (ngram, year, match_count) row, scale by
# min(1, per_year_token_cap / year_total[year]) and emit
# floor(match_count * scale) + Bernoulli(frac) copies via a seeded RNG.
# Replaces the 2026-06-02 garg_weat_china_ngram_weighted.yml, which used a
# methods-wrong per-row min(match_count, cap) (deleted).
language: "zh"
data_source: "ngram"
analysis_mode: "garg_weat"

paths:
  base_dir: "/scratch/network/yh6580/gender-occup"
  raw_ngram_dir: "/scratch/network/yh6580/gender-occup/raw_ngrams"
  decompressed_dir: "/scratch/network/yh6580/gender-occup/raw_ngrams_decompressed_subsampled"
  corpora_dir: "/scratch/network/yh6580/gender-occup/corpora_subsampled"
  models_dir: "/scratch/network/yh6580/gender-occup/models_subsampled"
  results_dir: "/scratch/network/yh6580/gender-occup/results_garg_weat_china_ngram_subsampled"
  log_dir: "/scratch/network/yh6580/gender-occup/logs_subsampled"
  figures_dir: "/scratch/network/yh6580/gender-occup/figures_garg_weat_china_ngram_subsampled"

wordlists:
  dir: "wordlists/zh/garg_weat_longitudinal"
  gender_words_file: "gender_words.json"
  categories:
    leadership: "cleaned_leadership.txt"
    family: "cleaned_family.txt"
    science: "cleaned_science.txt"

ngram:
  language: "chi_sim"
  n: 5
  min_year: 1940
  max_year: 2020

time_slices:
  window_size: 10
  step_size: 5
  start_year: 1940
  end_year: 2020

embedding:
  vector_size: 300
  window: 4
  min_count: 50
  sg: 1
  negative: 15
  workers: 16
  epochs: 5
  seed: 42
  model_name_template: "chi_sim_5gram_{slice_name}.model"

embedding_source: "china_ngram_subsampled"

corpus:
  # HistWords-style per-year-capped subsampling. cap=1e8 chosen as a
  # proportional analogue to the paper's 1e9 (English ngram is ~10x larger
  # than chi_sim per year). rng_seed makes the build byte-exact reproducible.
  weight_mode: "per_year_capped"
  per_year_token_cap: 100000000
  rng_seed: 42
  min_count_threshold: 1
  tokenizer: "whitespace"

analysis:
  metrics: [rnd, cohens_d]
  seed: 42
  ideation_sign:
    leadership: 1
    science: 1
    family: -1
  bootstrap:
    n_iter: 5000
    ci: 0.68
  subsample:
    fraction: 0.8
    n_rounds: 100
    ci: 0.95
```

- [ ] **Step 2: Delete the old `_weighted` profile**

```bash
git rm config/profiles/garg_weat_china_ngram_weighted.yml
```

- [ ] **Step 3: Verify the new profile parses**

Run: `python3 -c "import yaml; print(list(yaml.safe_load(open('config/profiles/garg_weat_china_ngram_subsampled.yml'))))"`
Expected: `['language', 'data_source', 'analysis_mode', 'paths', 'wordlists', 'ngram', 'time_slices', 'embedding', 'embedding_source', 'corpus', 'analysis']`

- [ ] **Step 4: Commit**

```bash
git add config/profiles/garg_weat_china_ngram_subsampled.yml
git commit -m "config(ngram): new subsampled profile replaces methods-wrong _weighted profile"
```

---

## Task 5: SLURM updates — rename driver + flip `*_zh.slurm` defaults

**Files:**
- Rename: `slurm/build_train_china_ngram_weighted_per_slice.slurm` → `slurm/build_train_china_ngram_subsampled_per_slice.slurm`
- Modify: the renamed file's CONFIG constant + leading comment
- Modify: `slurm/garg_weat_zh.slurm` (DEFAULT_CONFIGS entry)
- Modify: `slurm/describe_dataset_zh.slurm` (DEFAULT_CONFIGS entry)

- [ ] **Step 1: Rename the per-slice driver**

```bash
git mv slurm/build_train_china_ngram_weighted_per_slice.slurm slurm/build_train_china_ngram_subsampled_per_slice.slurm
```

- [ ] **Step 2: Retarget the config inside the driver**

In `slurm/build_train_china_ngram_subsampled_per_slice.slurm`, replace the `CONFIG=` line:

```bash
CONFIG="config/profiles/garg_weat_china_ngram_weighted.yml"
```
becomes:
```bash
CONFIG="config/profiles/garg_weat_china_ngram_subsampled.yml"
```

Also update the SBATCH job name on line 2:
```bash
#SBATCH --job-name=cnw_per_slice
```
becomes:
```bash
#SBATCH --job-name=cns_per_slice
```

And update the leading comment block — change every occurrence of `corpora_weighted/`, `corpus_weighted`, `_weighted` in comments to `corpora_subsampled/`, `corpus_subsampled`, `_subsampled` respectively. (Multiple lines in the comment block — apply consistently.)

- [ ] **Step 3: Flip the default config in `garg_weat_zh.slurm`**

In `slurm/garg_weat_zh.slurm` line 33:
```bash
    "config/profiles/garg_weat_china_ngram_weighted.yml"
```
becomes:
```bash
    "config/profiles/garg_weat_china_ngram_subsampled.yml"
```

- [ ] **Step 4: Flip the default config in `describe_dataset_zh.slurm`**

In `slurm/describe_dataset_zh.slurm` line 37:
```bash
    "config/profiles/garg_weat_china_ngram_weighted.yml"
```
becomes:
```bash
    "config/profiles/garg_weat_china_ngram_subsampled.yml"
```

- [ ] **Step 5: Update `_weighted` mentions in `build_corpus_zh.slurm` and `train_zh.slurm` comments**

In `slurm/build_corpus_zh.slurm`, find the comment line:
```bash
#   sbatch slurm/build_corpus_zh.slurm config/profiles/garg_weat_china_ngram_weighted.yml
```
and change it to:
```bash
#   sbatch slurm/build_corpus_zh.slurm config/profiles/garg_weat_china_ngram_subsampled.yml
```

In `slurm/train_zh.slurm`, find the comment line:
```bash
#   sbatch slurm/train_zh.slurm config/profiles/garg_weat_china_ngram_weighted.yml
```
and change it to:
```bash
#   sbatch slurm/train_zh.slurm config/profiles/garg_weat_china_ngram_subsampled.yml
```

- [ ] **Step 6: Verify SLURM scripts pass bash syntax check**

Run: `bash -n slurm/build_train_china_ngram_subsampled_per_slice.slurm && bash -n slurm/garg_weat_zh.slurm && bash -n slurm/describe_dataset_zh.slurm && bash -n slurm/build_corpus_zh.slurm && bash -n slurm/train_zh.slurm && echo OK`
Expected: `OK`

- [ ] **Step 7: Verify no stale `_weighted` references remain in the slurm tree**

Run: `grep -rn "_weighted" slurm/ config/ scripts/ tests/ 2>/dev/null || echo "no matches"`
Expected: `no matches` (or only matches inside the 06-02 spec / plan / runbook docs, which is fine — they're historical).

If anything outside `docs/` matches, fix it before committing.

- [ ] **Step 8: Commit**

```bash
git add slurm/
git commit -m "slurm: rename cnw_per_slice driver to subsampled + flip zh.slurm defaults"
```

---

## Task 6: Final sweep — full test suite + summary

**Files:** none modified

- [ ] **Step 1: Run the targeted test suite for this work**

Run: `pytest tests/test_ngram_totalcounts.py tests/test_build_corpora_ngram.py -v`
Expected: 14 passed.

- [ ] **Step 2: Run the full test suite to check for collateral damage**

Run: `pytest tests/ -x --tb=short`
Expected: All previously-passing tests still pass. If any test broke that has nothing to do with this work, investigate before declaring done.

- [ ] **Step 3: Confirm clean working tree**

Run: `git status --short`
Expected: only untracked items unrelated to this work (e.g., `TASK.md`, `data/surveys/...`).

- [ ] **Step 4: Verify the supersession arrow lands correctly**

Run: `head -10 docs/superpowers/specs/2026-06-02-china-ngram-weighting-and-summary-enhancements-design.md`
Expected: top includes "Superseded for weighting (Q1) by ..." line pointing at the 06-03 spec.

Run: `head -10 docs/superpowers/plans/2026-06-02-china-ngram-weighting-and-summary-enhancements.md`
Expected: top includes the "2026-06-03 — Superseded for the weighting tasks" callout.

- [ ] **Step 5: Print final summary**

Run: `git log --oneline -10`
Expected to see, top-down (newest first):
- slurm: rename cnw_per_slice driver to subsampled + flip zh.slurm defaults
- config(ngram): new subsampled profile replaces methods-wrong _weighted profile
- feat(ngram): add per_year_capped weight_mode (HistWords-style subsampling)
- refactor(ngram): remove capped_repetition weight_mode (methods-wrong, superseded by per_year_capped)
- feat(ngram): parse google v3 totalcounts-5 → {year: match_count}
- docs: 2026-06-03 spec replaces china ngram capped_repetition with HistWords-style per-year subsampling

No new commit needed for Task 6 — it's a verification gate.

---

## Operational steps (the user runs these on Princeton, after the plan completes)

These are **not** implementer tasks — recording here so the runbook stays current.

1. `scancel` any active `cnw_per_slice` jobs against the deleted `_weighted` profile.
2. `rm -rf /scratch/network/yh6580/gender-occup/corpora_weighted` (reclaims ~114 GB partial build).
3. `rm -rf /scratch/network/yh6580/gender-occup/raw_ngrams_decompressed_weighted /scratch/network/yh6580/gender-occup/models_weighted /scratch/network/yh6580/gender-occup/logs_weighted /scratch/network/yh6580/gender-occup/results_garg_weat_china_ngram_weighted /scratch/network/yh6580/gender-occup/figures_garg_weat_china_ngram_weighted` if any exist.
4. `git pull` on Princeton checkout.
5. `sbatch slurm/build_train_china_ngram_subsampled_per_slice.slurm` for the full 17-slice sweep at the corrected methodology. Expected per-slice peak ~3 GB, total ~50 GB; wall-clock dominated by gensim training.
6. After build+train completes: `sbatch slurm/garg_weat_zh.slurm` (analyze + visualize against the new subsampled models) and `sbatch slurm/describe_dataset_zh.slurm` (refresh methods summary).
