# China Ngram Count-Weighting + dataset_summary Enhancements — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **2026-06-03 — Superseded for the weighting tasks.** The `capped_repetition` weighting this plan implemented was methods-wrong (compresses the count signal's dynamic range). It has been replaced by HistWords-style per-year token-budget subsampling. See the corrected design at [`docs/superpowers/specs/2026-06-03-china-ngram-histwords-style-subsampling-design.md`](../specs/2026-06-03-china-ngram-histwords-style-subsampling-design.md) and its implementation plan (to be written next). The Q2 dataset_summary tasks in this plan (year range, tokens/doc, model-vocab range) remain correct and have shipped.

**Spec:** [`docs/superpowers/specs/2026-06-02-china-ngram-weighting-and-summary-enhancements-design.md`](../specs/2026-06-02-china-ngram-weighting-and-summary-enhancements-design.md)

**Goal:** Make the Chinese Google 5-gram pipeline count-weighted (capped raw repetition) via a new profile + fresh dirs, and add Year range + Tokens/doc + per-source model-vocab range to `dataset_summary.md`.

**Architecture:** Add `corpus.weight_mode` (default `"presence"`, new `"capped_repetition"`) + `corpus.repeat_cap` (default 100) to `build_corpora_ngram.py::process_ngram_file`. New profile `garg_weat_china_ngram_weighted.yml` clones the old china_ngram profile with `*_weighted` paths and the new weight mode. Both Princeton `*_zh.slurm` scripts default to the new profile. `scripts/common/dataset_stats.py` gains a `_year_range` helper, two new per-unit table columns, two new `SourceTotals` fields, and a Training-section model-vocab-range bullet.

**Tech Stack:** Python 3, pytest. No new dependencies.

---

## File structure

```
scripts/data_prep/build_corpora_ngram.py                            — MODIFY (process_ngram_file dispatches on weight_mode)
scripts/common/dataset_stats.py                                     — MODIFY (Year range, Tokens/doc, model_vocab_min/max, Training bullet)
config/profiles/garg_weat_china_ngram_weighted.yml                  — CREATE
slurm/garg_weat_zh.slurm                                            — MODIFY (DEFAULT_CONFIGS swap)
slurm/describe_dataset_zh.slurm                                     — MODIFY (DEFAULT_CONFIGS swap)
tests/test_build_corpora_ngram.py                                   — CREATE (6 weighting tests)
tests/test_dataset_stats.py                                         — MODIFY (8 new tests across Tasks 4–5)
```

---

## Task 1: Builder — `weight_mode` + `repeat_cap` + `capped_repetition` branch

**Files:**
- Modify: `scripts/data_prep/build_corpora_ngram.py`
- Create: `tests/test_build_corpora_ngram.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_build_corpora_ngram.py
"""Tests for the Chinese Google Ngram corpus builder — weight_mode dispatch."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from scripts.data_prep.build_corpora_ngram import process_ngram_file

logger = logging.getLogger("test")


def _make_ngram_file(dir_path: Path, name: str, lines: list[str]) -> Path:
    """Write a plain-text v3 ngram file (process_ngram_file reads decompressed text)."""
    p = dir_path / name
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def _line(ngram_text: str, year: int, match_count: int, volume_count: int = 1) -> str:
    """One v3 ngram line: ngram\\tyear,match,volume."""
    return f"{ngram_text}\t{year},{match_count},{volume_count}"


def _config(tmp_path: Path, **overrides) -> dict:
    """Minimal config dict for process_ngram_file."""
    corpus = {"min_count_threshold": 1}
    corpus.update(overrides.pop("corpus", {}))
    return {
        "corpus": corpus,
        "paths": {"corpora_dir": str(tmp_path / "corpora")},
        **overrides,
    }


def _read_corpus(corpora_dir: Path, slice_name: str) -> list[str]:
    """Read all corpus_*.txt lines for a slice (sorted by filename), return list of lines."""
    slice_dir = corpora_dir / slice_name
    if not slice_dir.exists():
        return []
    out: list[str] = []
    for p in sorted(slice_dir.glob("corpus_*.txt")):
        out.extend(line for line in p.read_text(encoding="utf-8").splitlines() if line)
    return out


# ngram_text needs ≥ 2 Chinese tokens to survive clean_ngram (line 69 of build_corpora_ngram.py)
NGRAM = "中国 经济 发展 政策 改革"
NGRAM_CLEAN = "中国 经济 发展 政策 改革"  # already pure-Chinese, clean_ngram preserves


class TestPresenceModeDefault:
    def test_no_weight_mode_key_preserves_existing_behavior(self, tmp_path):
        # Two identical-ngram entries in different years that both fall in 1940_1949.
        # Presence mode = set dedup → ONE corpus line, regardless of count.
        f = _make_ngram_file(tmp_path, "5-00000-of-00105", [
            _line(NGRAM, 1942, match_count=5),
            _line(NGRAM, 1948, match_count=200),
        ])
        cfg = _config(tmp_path)  # no weight_mode → defaults to "presence"
        time_slices = [(1940, 1949)]
        process_ngram_file(f, time_slices, cfg, logger)
        lines = _read_corpus(Path(cfg["paths"]["corpora_dir"]), "1940_1949")
        assert lines == [NGRAM_CLEAN]


class TestCappedRepetition:
    def test_count_below_cap_emits_count_copies(self, tmp_path):
        f = _make_ngram_file(tmp_path, "5-00000-of-00105", [
            _line(NGRAM, 1942, match_count=3),
        ])
        cfg = _config(tmp_path, corpus={"weight_mode": "capped_repetition", "repeat_cap": 100})
        process_ngram_file(f, [(1940, 1949)], cfg, logger)
        lines = _read_corpus(Path(cfg["paths"]["corpora_dir"]), "1940_1949")
        assert lines == [NGRAM_CLEAN] * 3

    def test_count_above_cap_is_clamped(self, tmp_path):
        f = _make_ngram_file(tmp_path, "5-00000-of-00105", [
            _line(NGRAM, 1942, match_count=5000),
        ])
        cfg = _config(tmp_path, corpus={"weight_mode": "capped_repetition", "repeat_cap": 100})
        process_ngram_file(f, [(1940, 1949)], cfg, logger)
        lines = _read_corpus(Path(cfg["paths"]["corpora_dir"]), "1940_1949")
        assert lines == [NGRAM_CLEAN] * 100

    def test_multi_year_in_same_slice_sums_capped_contributions(self, tmp_path):
        # 1942: 80 (below cap → 80 copies). 1948: 200 (clamped to 100 copies). Both in 1940_1949.
        # Total: 180 lines in the slice corpus.
        f = _make_ngram_file(tmp_path, "5-00000-of-00105", [
            _line(NGRAM, 1942, match_count=80),
            _line(NGRAM, 1948, match_count=200),
        ])
        cfg = _config(tmp_path, corpus={"weight_mode": "capped_repetition", "repeat_cap": 100})
        process_ngram_file(f, [(1940, 1949)], cfg, logger)
        lines = _read_corpus(Path(cfg["paths"]["corpora_dir"]), "1940_1949")
        assert len(lines) == 180
        assert set(lines) == {NGRAM_CLEAN}

    def test_min_count_threshold_filter_runs_before_cap(self, tmp_path):
        # Below min_count_threshold: dropped entirely (no copies emitted).
        f = _make_ngram_file(tmp_path, "5-00000-of-00105", [
            _line(NGRAM, 1942, match_count=5),
        ])
        cfg = _config(
            tmp_path,
            corpus={"weight_mode": "capped_repetition", "repeat_cap": 100, "min_count_threshold": 10},
        )
        process_ngram_file(f, [(1940, 1949)], cfg, logger)
        lines = _read_corpus(Path(cfg["paths"]["corpora_dir"]), "1940_1949")
        assert lines == []


class TestInvalidWeightMode:
    def test_raises_value_error(self, tmp_path):
        f = _make_ngram_file(tmp_path, "5-00000-of-00105", [
            _line(NGRAM, 1942, match_count=1),
        ])
        cfg = _config(tmp_path, corpus={"weight_mode": "bogus"})
        with pytest.raises(ValueError, match="weight_mode"):
            process_ngram_file(f, [(1940, 1949)], cfg, logger)


class TestRepeatCapEdge:
    def test_repeat_cap_one_emits_one_copy(self, tmp_path):
        f = _make_ngram_file(tmp_path, "5-00000-of-00105", [
            _line(NGRAM, 1942, match_count=100),
        ])
        cfg = _config(tmp_path, corpus={"weight_mode": "capped_repetition", "repeat_cap": 1})
        process_ngram_file(f, [(1940, 1949)], cfg, logger)
        lines = _read_corpus(Path(cfg["paths"]["corpora_dir"]), "1940_1949")
        assert lines == [NGRAM_CLEAN]
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /Users/houyuxin/08Coding/gender-occup-segregation && python -m pytest tests/test_build_corpora_ngram.py -v
```

Expected: tests run but most fail — `TestPresenceModeDefault` passes (existing behavior), but `TestCappedRepetition`, `TestInvalidWeightMode`, and `TestRepeatCapEdge` fail because the `weight_mode` key is ignored by the current builder (it always uses `set.add` dedup, so capped_repetition emits 1 line, not N).

- [ ] **Step 3: Modify `process_ngram_file` to dispatch on `weight_mode`**

Replace the existing `process_ngram_file` in `scripts/data_prep/build_corpora_ngram.py` (lines 92–137) with:

```python
VALID_WEIGHT_MODES = {"presence", "capped_repetition"}


def process_ngram_file(file_path, time_slices, config, logger):
    """Process a single ngram file and write to time-slice corpus files.

    Dispatches on ``config['corpus']['weight_mode']`` (default ``"presence"``):
      - ``"presence"``: dedup-per-slice via set (one line per unique ngram per slice).
      - ``"capped_repetition"``: emit ``min(match_count, repeat_cap)`` copies of each
        (ngram, year) entry into every matching slice; cross-year contributions sum.
    """
    corpus_cfg = config['corpus']
    min_count = corpus_cfg['min_count_threshold']
    weight_mode = corpus_cfg.get('weight_mode', 'presence')
    if weight_mode not in VALID_WEIGHT_MODES:
        raise ValueError(
            f"Invalid corpus.weight_mode={weight_mode!r}; "
            f"expected one of {sorted(VALID_WEIGHT_MODES)}"
        )
    repeat_cap = max(int(corpus_cfg.get('repeat_cap', 100)), 1)

    corpora_dir = Path(config['paths']['corpora_dir'])
    os.makedirs(corpora_dir, exist_ok=True)

    logger.info(f"Processing {file_path.name} (weight_mode={weight_mode}, repeat_cap={repeat_cap})...")
    lines_processed = 0
    lines_emitted = defaultdict(int)
    file_index = file_path.name.split("-")[1]
    # Buffer type depends on mode: set for dedup (presence), list for repetition.
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
                matched_slices = set()
                for start_year, end_year in time_slices:
                    if start_year <= year <= end_year:
                        matched_slices.add(f"{start_year}_{end_year}")
                for slice_name in matched_slices:
                    if weight_mode == "presence":
                        write_buffer[slice_name].add(ngram_text)
                        lines_emitted[slice_name] += 1
                    else:  # capped_repetition
                        n_repeats = min(match_count, repeat_cap)
                        write_buffer[slice_name].extend([ngram_text] * n_repeats)
                        lines_emitted[slice_name] += n_repeats
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

Key changes vs the original:
- `weight_mode` is validated up front; bad values raise `ValueError`.
- `repeat_cap` defaulted to 100 and clamped to ≥ 1 (so `repeat_cap=0` or negative still emits one copy).
- Buffer is `set` in presence mode (existing behavior) and `list` in capped_repetition mode.
- Flush logic extracted into `_flush(slice_name)` since the inner-loop and post-loop flush both need it; buffer is reset to the correct empty type.
- `lines_included` renamed to `lines_emitted` and now counts ACTUAL lines written (after repetition), not entries matched.

- [ ] **Step 4: Run tests to confirm they pass**

```bash
cd /Users/houyuxin/08Coding/gender-occup-segregation && python -m pytest tests/test_build_corpora_ngram.py -v
```

Expected: 7 passed (TestPresenceModeDefault: 1, TestCappedRepetition: 4, TestInvalidWeightMode: 1, TestRepeatCapEdge: 1).

- [ ] **Step 5: Commit**

```bash
cd /Users/houyuxin/08Coding/gender-occup-segregation && git add scripts/data_prep/build_corpora_ngram.py tests/test_build_corpora_ngram.py && git commit -m "build_corpora_ngram: add weight_mode dispatch (presence default, capped_repetition new)"
```

---

## Task 2: New profile — `garg_weat_china_ngram_weighted.yml`

**Files:**
- Create: `config/profiles/garg_weat_china_ngram_weighted.yml`

- [ ] **Step 1: Confirm the existing china_ngram profile**

```bash
cd /Users/houyuxin/08Coding/gender-occup-segregation && cat config/profiles/garg_weat_china_ngram.yml
```

Expected: shows the existing profile (no weight_mode key; corpus has `use_counts: false`, `min_count_threshold: 1`, `tokenizer: whitespace`).

- [ ] **Step 2: Create the new weighted profile**

```yaml
# config/profiles/garg_weat_china_ngram_weighted.yml
# Count-weighted Chinese Google 5-gram pipeline. Built via the new
# capped_repetition weight_mode in scripts/data_prep/build_corpora_ngram.py
# (emit min(match_count, repeat_cap) copies per (ngram, year), summing
# across years within each slice). Coexists with the presence-only
# garg_weat_china_ngram.yml — fresh dirs (*_weighted) so both sets of
# corpora and models live side-by-side for A/B comparison.
language: "zh"
data_source: "ngram"
analysis_mode: "garg_weat"

paths:
  base_dir: "/scratch/network/yh6580/gender-occup"
  # Source ngrams: same .gz shards as the presence-only build.
  raw_ngram_dir: "/scratch/network/yh6580/gender-occup/raw_ngrams"
  # Fresh dirs — _weighted suffix throughout — so nothing overlaps the
  # existing presence-only china_ngram corpora/models/results/figures.
  decompressed_dir: "/scratch/network/yh6580/gender-occup/raw_ngrams_decompressed_weighted"
  corpora_dir: "/scratch/network/yh6580/gender-occup/corpora_weighted"
  models_dir: "/scratch/network/yh6580/gender-occup/models_weighted"
  results_dir: "/scratch/network/yh6580/gender-occup/results_garg_weat_china_ngram_weighted"
  log_dir: "/scratch/network/yh6580/gender-occup/logs_weighted"
  figures_dir: "/scratch/network/yh6580/gender-occup/figures_garg_weat_china_ngram_weighted"

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

embedding_source: "china_ngram_weighted"

corpus:
  # New count-weighted mode — emit min(match_count, repeat_cap) copies of
  # each (ngram, year) entry; cross-year contributions within a slice sum.
  weight_mode: "capped_repetition"
  repeat_cap: 100
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

- [ ] **Step 3: Sanity-check the YAML parses and config_loader accepts it**

```bash
cd /Users/houyuxin/08Coding/gender-occup-segregation && python -c "
from scripts.common.config_loader import load_config
cfg = load_config('config/profiles/garg_weat_china_ngram_weighted.yml')
print('weight_mode:', cfg['corpus']['weight_mode'])
print('repeat_cap:', cfg['corpus']['repeat_cap'])
print('embedding_source:', cfg['embedding_source'])
print('analysis.metrics:', cfg['analysis']['metrics'])
"
```

Expected:
```
weight_mode: capped_repetition
repeat_cap: 100
embedding_source: china_ngram_weighted
analysis.metrics: ['rnd', 'cohens_d']
```

- [ ] **Step 4: Commit**

```bash
cd /Users/houyuxin/08Coding/gender-occup-segregation && git add config/profiles/garg_weat_china_ngram_weighted.yml && git commit -m "profile: garg_weat_china_ngram_weighted (capped_repetition, repeat_cap=100, fresh dirs)"
```

---

## Task 3: SLURM defaults — both `*_zh.slurm` scripts use the new profile

**Files:**
- Modify: `slurm/garg_weat_zh.slurm`
- Modify: `slurm/describe_dataset_zh.slurm`

- [ ] **Step 1: Modify `slurm/garg_weat_zh.slurm` DEFAULT_CONFIGS**

Replace the existing `DEFAULT_CONFIGS` block (lines 31–34) so the second entry points at the weighted profile:

```bash
DEFAULT_CONFIGS=(
    "config/profiles/garg_weat_renminribao.yml"
    "config/profiles/garg_weat_china_ngram_weighted.yml"
)
```

- [ ] **Step 2: Modify `slurm/describe_dataset_zh.slurm` DEFAULT_CONFIGS**

Replace the existing `DEFAULT_CONFIGS` block (lines 35–38) so the second entry points at the weighted profile:

```bash
DEFAULT_CONFIGS=(
    "config/profiles/garg_weat_renminribao.yml"
    "config/profiles/garg_weat_china_ngram_weighted.yml"
)
```

- [ ] **Step 3: Verify the swap and shell syntax**

```bash
cd /Users/houyuxin/08Coding/gender-occup-segregation && for f in slurm/garg_weat_zh.slurm slurm/describe_dataset_zh.slurm; do
  echo "=== $f ==="
  bash -n "$f" && echo "syntax OK"
  grep -c "garg_weat_china_ngram_weighted.yml" "$f" | xargs -I {} echo "weighted profile references: {}"
  grep -c "garg_weat_china_ngram.yml" "$f" | xargs -I {} echo "old profile references: {}"
done
```

Expected: both files pass `bash -n`. Weighted-profile references = 1 each (the new DEFAULT_CONFIGS entry). Old-profile references = 0 each (the old DEFAULT_CONFIGS entry was removed; `garg_weat_china_ngram.yml` is NOT a substring of `garg_weat_china_ngram_weighted.yml` because the boundary character is `_` not `.`).

- [ ] **Step 4: Commit**

```bash
cd /Users/houyuxin/08Coding/gender-occup-segregation && git add slurm/garg_weat_zh.slurm slurm/describe_dataset_zh.slurm && git commit -m "slurm: zh scripts default to garg_weat_china_ngram_weighted profile"
```

---

## Task 4: Q2 part 1 — `_year_range` helper + Year range / Tokens-per-doc columns

**Files:**
- Modify: `scripts/common/dataset_stats.py`
- Modify: `tests/test_dataset_stats.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_dataset_stats.py`:

```python
from scripts.common.dataset_stats import _year_range


class TestYearRange:
    def test_slice_format(self):
        assert _year_range("1940_1949") == "1940–1949"

    def test_coha_decade(self):
        assert _year_range("1940s") == "1940–1949"

    def test_province_year(self):
        assert _year_range("北京_2020") == "2020"

    def test_unknown_returns_dash(self):
        assert _year_range("北京") == "—"


class TestRenderMarkdownNewColumns:
    def _minimal_config(self):
        return {
            "language": "zh", "data_source": "renminribao",
            "embedding": {"vector_size": 300, "window": 4, "min_count": 50,
                          "sg": 1, "negative": 15, "epochs": 5, "seed": 42,
                          "workers": 16, "model_name_template": "rmrb_{slice_name}.model"},
            "paths": {"models_dir": "/data/models", "raw_data_dir": "/data/raw"},
        }

    def test_per_unit_table_includes_year_range_and_tokens_per_doc(self):
        # 100 docs, 1000 tokens → 10.0 tokens/doc.
        per_unit = {
            "1940_1949": (_stats("1940_1949", 100, 1000, 250), 80, _raw("1940_1949", 12, 5000)),
        }
        totals = aggregate_source(per_unit, vocab_union_count=None)
        md = render_markdown(
            totals=totals, per_unit=per_unit, config=self._minimal_config(),
            config_path="x.yml", generated_at="2026-06-02",
        )
        # New header columns:
        assert "| Unit | Year range | Documents | Tokens | Tokens/doc |" in md
        # Year-range value for 1940_1949:
        assert "1940–1949" in md
        # Tokens/doc value (10.0 formatted with thousands-sep):
        assert "10.0" in md
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /Users/houyuxin/08Coding/gender-occup-segregation && python -m pytest tests/test_dataset_stats.py::TestYearRange tests/test_dataset_stats.py::TestRenderMarkdownNewColumns -v
```

Expected: `ImportError: cannot import name '_year_range'` (helper doesn't exist yet).

- [ ] **Step 3: Add `_year_range` helper to `scripts/common/dataset_stats.py`**

Add after `_fmt` (just before `render_markdown`):

```python
import re


def _year_range(unit_name: str) -> str:
    """Parse year info from common unit-name shapes; returns en-dash range or '—'.

    Examples:
      '1940_1949'  -> '1940–1949'   (longitudinal slice)
      '1940s'      -> '1940–1949'   (COHA decade)
      '北京_2020'  -> '2020'        (province-year)
      '北京'       -> '—'           (bare province, no year info)
    """
    m = re.match(r"^(\d{4})_(\d{4})$", unit_name)
    if m:
        return f"{m.group(1)}–{m.group(2)}"
    m = re.match(r"^(\d{4})s$", unit_name)
    if m:
        start = int(m.group(1))
        return f"{start}–{start + 9}"
    m = re.match(r"^.+_(\d{4})$", unit_name)
    if m:
        return m.group(1)
    return "—"
```

- [ ] **Step 4: Update `render_markdown` per-unit table header + rows**

In `scripts/common/dataset_stats.py`, replace the existing per-unit-table block in `render_markdown` (currently lines ~322–335) with:

```python
    # Per-unit breakdown
    lines.append("## Per-unit breakdown")
    lines.append("")
    lines.append(
        f"| Unit | Year range | {docs_header} | Tokens | Tokens/doc | "
        f"Raw vocab | Model vocab | Raw files | Raw bytes |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for unit_name in sorted(per_unit):
        stats, mv, raw = per_unit[unit_name]
        raw_files = _fmt(raw.n_files) if raw is not None else "n/a"
        raw_bytes = _human_bytes(raw.n_bytes) if raw is not None else "n/a"
        tokens_per_doc = (
            f"{stats.n_tokens / stats.n_docs:,.1f}" if stats.n_docs > 0 else "—"
        )
        lines.append(
            f"| {unit_name} | {_year_range(unit_name)} | {_fmt(stats.n_docs)} | "
            f"{_fmt(stats.n_tokens)} | {tokens_per_doc} | "
            f"{_fmt(stats.n_vocab_raw)} | {_fmt(mv)} | {raw_files} | {raw_bytes} |"
        )
    lines.append("")
    return "\n".join(lines)
```

Header now has 9 columns; separator has 9 cells; rows have 9 cells.

- [ ] **Step 5: Run tests to confirm they pass**

```bash
cd /Users/houyuxin/08Coding/gender-occup-segregation && python -m pytest tests/test_dataset_stats.py -v
```

Expected: 30 passed (25 from prior tasks + 5 new: 4 TestYearRange + 1 TestRenderMarkdownNewColumns). Pre-existing `TestRenderMarkdown::test_renders_all_sections` still passes — its assertions don't depend on the per-unit column count.

- [ ] **Step 6: Commit**

```bash
cd /Users/houyuxin/08Coding/gender-occup-segregation && git add scripts/common/dataset_stats.py tests/test_dataset_stats.py && git commit -m "dataset_stats: _year_range helper + Year range / Tokens-per-doc per-unit columns"
```

---

## Task 5: Q2 part 2 — `model_vocab_min/max` in `SourceTotals` + Training-section bullet

**Files:**
- Modify: `scripts/common/dataset_stats.py`
- Modify: `tests/test_dataset_stats.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_dataset_stats.py`:

```python
class TestAggregateSourceModelVocabRange:
    def test_min_and_max_with_some_none(self):
        per_unit = {
            "u1": (_stats("u1", 1, 1, 1), 100, None),
            "u2": (_stats("u2", 1, 1, 1), 200, None),
            "u3": (_stats("u3", 1, 1, 1), None, None),  # missing model
            "u4": (_stats("u4", 1, 1, 1), 300, None),
        }
        totals = aggregate_source(per_unit, vocab_union_count=None)
        assert totals.model_vocab_min == 100
        assert totals.model_vocab_max == 300

    def test_all_none_yields_none(self):
        per_unit = {
            "u1": (_stats("u1", 1, 1, 1), None, None),
            "u2": (_stats("u2", 1, 1, 1), None, None),
        }
        totals = aggregate_source(per_unit, vocab_union_count=None)
        assert totals.model_vocab_min is None
        assert totals.model_vocab_max is None


class TestRenderMarkdownTrainingVocabRange:
    def _minimal_config(self):
        return {
            "language": "zh", "data_source": "renminribao",
            "embedding": {"vector_size": 300, "window": 4, "min_count": 50,
                          "sg": 1, "negative": 15, "epochs": 5, "seed": 42,
                          "workers": 16, "model_name_template": "rmrb_{slice_name}.model"},
            "paths": {"models_dir": "/data/models", "raw_data_dir": "/data/raw"},
        }

    def test_training_section_shows_vocab_range_when_any_model_present(self):
        per_unit = {
            "1940_1949": (_stats("1940_1949", 1, 1, 1), 100, None),
            "1950_1959": (_stats("1950_1959", 1, 1, 1), 300, None),
        }
        totals = aggregate_source(per_unit, vocab_union_count=None)
        md = render_markdown(
            totals=totals, per_unit=per_unit, config=self._minimal_config(),
            config_path="x.yml", generated_at="2026-06-02",
        )
        # Sentence shape from spec: "Model vocab across N units: min — max (mean: …)"
        assert "Model vocab across 2 units:" in md
        assert "100" in md
        assert "300" in md
        assert "mean: 200" in md  # (100+300)/2

    def test_training_section_suppresses_bullet_when_all_models_missing(self):
        per_unit = {
            "u1": (_stats("u1", 1, 1, 1), None, None),
        }
        totals = aggregate_source(per_unit, vocab_union_count=None)
        md = render_markdown(
            totals=totals, per_unit=per_unit, config=self._minimal_config(),
            config_path="x.yml", generated_at="2026-06-02",
        )
        assert "Model vocab across" not in md
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /Users/houyuxin/08Coding/gender-occup-segregation && python -m pytest tests/test_dataset_stats.py::TestAggregateSourceModelVocabRange tests/test_dataset_stats.py::TestRenderMarkdownTrainingVocabRange -v
```

Expected: `AttributeError: 'SourceTotals' object has no attribute 'model_vocab_min'`.

- [ ] **Step 3: Extend `SourceTotals` and `aggregate_source`**

In `scripts/common/dataset_stats.py`, modify the `SourceTotals` dataclass (currently ~lines 197–208) to add two trailing fields with defaults:

```python
@dataclass
class SourceTotals:
    n_units: int
    n_docs: int
    n_tokens: int
    vocab_raw_min: int
    vocab_raw_max: int
    vocab_raw_mean: float
    n_model_vocab_sum: int             # sum of per-unit model vocabs (None entries skipped)
    n_raw_files: int
    n_raw_bytes: int
    vocab_union_count: Optional[int]   # None unless --vocab-union was passed
    model_vocab_min: Optional[int] = None   # min across units with non-None model vocab
    model_vocab_max: Optional[int] = None   # max across units with non-None model vocab
```

In `aggregate_source` (currently ~lines 215–234), compute and pass the two new fields. Replace the function body with:

```python
def aggregate_source(
    per_unit: Dict[str, PerUnitTriple],
    vocab_union_count: Optional[int],
) -> SourceTotals:
    """Reduce per-unit triples to one source-level summary row."""
    if not per_unit:
        return SourceTotals(0, 0, 0, 0, 0, 0.0, 0, 0, 0, vocab_union_count)
    vocab_raws = [s.n_vocab_raw for s, _, _ in per_unit.values()]
    model_vocabs = [v for _, v, _ in per_unit.values() if v is not None]
    model_vocab_min = min(model_vocabs) if model_vocabs else None
    model_vocab_max = max(model_vocabs) if model_vocabs else None
    return SourceTotals(
        n_units=len(per_unit),
        n_docs=sum(s.n_docs for s, _, _ in per_unit.values()),
        n_tokens=sum(s.n_tokens for s, _, _ in per_unit.values()),
        vocab_raw_min=min(vocab_raws),
        vocab_raw_max=max(vocab_raws),
        vocab_raw_mean=sum(vocab_raws) / len(vocab_raws),
        n_model_vocab_sum=sum(model_vocabs),
        n_raw_files=sum(r.n_files for _, _, r in per_unit.values() if r is not None),
        n_raw_bytes=sum(r.n_bytes for _, _, r in per_unit.values() if r is not None),
        vocab_union_count=vocab_union_count,
        model_vocab_min=model_vocab_min,
        model_vocab_max=model_vocab_max,
    )
```

(Note: `n_model_vocab_sum` is now computed once via `sum(model_vocabs)` for clarity, identical result to before.)

- [ ] **Step 4: Add the Training-section bullet to `render_markdown`**

In `render_markdown`, after the "Model files" line and before the trailing `lines.append("")` of the Training section (currently ~line 320), insert:

```python
    if totals.model_vocab_min is not None:
        n_with_vocab = sum(1 for _, v, _ in per_unit.values() if v is not None)
        mean_mv = totals.n_model_vocab_sum / n_with_vocab if n_with_vocab else 0
        lines.append(
            f"- Model vocab across {totals.n_units} units: "
            f"{_fmt(totals.model_vocab_min)} — {_fmt(totals.model_vocab_max)} "
            f"(mean: {mean_mv:,.0f})"
        )
```

Place it directly after `lines.append(f"- Model files: …")` and before the existing `lines.append("")` that closes the Training section.

- [ ] **Step 5: Run tests to confirm they pass**

```bash
cd /Users/houyuxin/08Coding/gender-occup-segregation && python -m pytest tests/test_dataset_stats.py -v
```

Expected: 34 passed (30 from after Task 4 + 4 new: 2 TestAggregateSourceModelVocabRange + 2 TestRenderMarkdownTrainingVocabRange).

- [ ] **Step 6: Commit**

```bash
cd /Users/houyuxin/08Coding/gender-occup-segregation && git add scripts/common/dataset_stats.py tests/test_dataset_stats.py && git commit -m "dataset_stats: SourceTotals.model_vocab_min/max + Training-section vocab-range bullet"
```

---

## Task 6: Full-suite regression sweep

**Files:** none (verification only)

- [ ] **Step 1: Run the new and modified test surfaces**

```bash
cd /Users/houyuxin/08Coding/gender-occup-segregation && python -m pytest tests/test_build_corpora_ngram.py tests/test_dataset_stats.py -v
```

Expected: 41 passed (7 from Task 1 + 34 from Tasks 4 & 5).

- [ ] **Step 2: Run the broader test surface to confirm no regression**

```bash
cd /Users/houyuxin/08Coding/gender-occup-segregation && python -m pytest tests/ 2>&1 | tail -10
```

Expected: all 325 pre-existing tests still pass (5 skipped), plus the new ones, all skipped/pass totals unchanged or grown. Zero failures.

- [ ] **Step 3: Sanity-check the new profile via `describe_dataset` --help and dry-run config loading**

```bash
cd /Users/houyuxin/08Coding/gender-occup-segregation && python -c "
from scripts.common.config_loader import load_config
cfg = load_config('config/profiles/garg_weat_china_ngram_weighted.yml')
print('weight_mode:', cfg['corpus']['weight_mode'])
print('repeat_cap:', cfg['corpus']['repeat_cap'])
"
```

Expected: `weight_mode: capped_repetition`, `repeat_cap: 100`.

- [ ] **Step 4: Confirm both SLURM scripts default to the weighted profile**

```bash
cd /Users/houyuxin/08Coding/gender-occup-segregation && for f in slurm/garg_weat_zh.slurm slurm/describe_dataset_zh.slurm; do
  echo "=== $f ==="
  grep -A 3 "DEFAULT_CONFIGS=(" "$f"
done
```

Expected: both scripts show `garg_weat_renminribao.yml` and `garg_weat_china_ngram_weighted.yml` as the two default configs.

- [ ] **Step 5: No-op commit unless steps 1–4 produced edits**

If anything needed to be fixed during the sweep, commit the fix; otherwise skip.

---

## Done condition

After Task 6:

1. `python -m pytest tests/test_build_corpora_ngram.py tests/test_dataset_stats.py -v` shows 40 passed.
2. `python -m pytest tests/` shows the full pre-existing suite (325 + 5 skipped) plus the new tests, all green.
3. `python -m scripts.data_prep.build_corpora_ngram --config=config/profiles/garg_weat_china_ngram_weighted.yml --slice=<one>` on the server produces a count-weighted corpus (line count materially higher than presence-only equivalent).
4. `sbatch slurm/garg_weat_zh.slurm` with no args invokes the weighted profile.
5. `dataset_summary.md` for any source shows `Year range` and `Tokens/doc` columns and (when models exist) a `Model vocab across N units` bullet in the Training section.
6. Old presence-only profile still works when passed explicitly — backwards compat preserved.
