# China Ngram Count-Weighting + dataset_summary Enhancements — Design Spec

**Date:** 2026-06-02
**Status:** Superseded for weighting (Q1) by [`2026-06-03-china-ngram-histwords-style-subsampling-design.md`](2026-06-03-china-ngram-histwords-style-subsampling-design.md). Dataset-summary enhancements (Q2) are unaffected and shipped.

> **2026-06-03 update:** The `capped_repetition` weighting introduced in this spec turned out to be methods-wrong — `min(match_count, cap)` compresses the dynamic range of the count signal, which is the very thing weighting was meant to preserve. HistWords (Hamilton et al. 2016, Appendix A) applies a **per-year** token-budget subsample on the count-proportional stream instead. The corrected design is in the 06-03 spec linked above. This file is kept as historical record of how we got there; the Q2 dataset_summary changes it describes (year range, tokens/doc, model-vocab range) remain correct and shipped.

**Purpose:**
- (Q1) Make the China Google 5-gram pipeline count-weighted instead of presence-only, so trained embeddings reflect actual corpus frequency rather than ngram-type co-occurrence. Capped raw repetition (`min(match_count, repeat_cap)` per (ngram, year)) to bound disk and training cost. Coexists with the existing presence-only china ngram models via a new profile + fresh directories.
- (Q2) Surface vocabulary and per-model details in `dataset_summary.md` that a methods section will quote: per-unit year range, mean tokens per training doc, and per-source model-vocab range.

Both changes ship in one spec because Q2 will be needed to describe Q1 in writing.

---

## Goal

- A new config profile `config/profiles/garg_weat_china_ngram_weighted.yml` that, when fed through the existing `build_corpora_ngram` → `train_embeddings` flow, produces count-weighted China Ngram corpora and models — independent of the existing presence-only ones.
- The corpus builder honors a `corpus.weight_mode` key (`"presence"` default, `"capped_repetition"` new) and a `corpus.repeat_cap` (default 100) when the new mode is active.
- The two Princeton `*_zh.slurm` scripts (`garg_weat_zh.slurm`, `describe_dataset_zh.slurm`) default to the new weighted profile, with the old one still runnable by passing it explicitly.
- `dataset_summary.md` per-unit table gains `Year range` and `Tokens/doc` columns; the Training section reports a per-source model-vocab range.

## Non-goals

- Not changing the English Google ngram pipeline (`build_corpora_ngram_en.py`). EN switched to pre-trained HistWords; that builder is dormant.
- Not changing RMRB / Weibo / provincial newspaper / COHA builders. None of them have a `weight_mode` concept; they read real running text where token frequency is already the natural signal.
- Not normalizing by `totalcounts-5`. The user's pointer to that file is noted; per-year frequency normalization is a future variant (Option B/C in brainstorming). Capped raw repetition (Option A) is what ships.
- Not orchestrating the rebuild + retrain itself. The existing `build_corpus.slurm` / `train.slurm` accept `--config=`; the user runs them against the new profile on the server.
- Not computing OOV rate (fraction of corpus tokens dropped by Word2Vec's `min_count`). Useful but expensive; deferred.

---

## Decisions table

| # | Decision | Choice | Why |
|---|---|---|---|
| 1 | Weighting scheme | Capped raw repetition: emit `min(match_count, repeat_cap)` copies per (ngram, year) | Simple, exact for low-frequency ngrams, prevents disk blowup from super-frequent function-word 5-grams. Sliding control via `repeat_cap`. |
| 2 | Cap granularity | Per-(ngram, year) | Each year's match_count independently capped; cross-year contributions sum. Matches the row-level semantics of Google's v3 line format. |
| 3 | `repeat_cap` default | 100 | Most 5-gram per-year counts fall under 100, so they contribute their true frequency. Super-frequent ngrams get clamped. ~20–50× disk vs presence-only. |
| 4 | Coexistence | New profile + fresh dirs (`*_weighted`) | Existing presence-only china-ngram models stay around for direct A/B comparison in the methods. |
| 5 | English ngram pipeline | Untouched | EN uses pre-trained HistWords; the EN builder is dormant. |
| 6 | totalcounts-5 use | Not used for weighting | Capped raw repetition needs only per-row `match_count`. Normalization-style weighting (Option B/C) would need totalcounts; deferred. |
| 7 | SLURM defaults | Both `*_zh.slurm` scripts default to the new profile | Going-forward "the china ngram run" means the weighted one; old profile still works if passed explicitly. |
| 8 | Q2 additions | Year range + Tokens/doc columns; Training-section model-vocab range | Smallest set that gives a methods paragraph everything it needs to cite per-model coverage and corpus shape. |

---

## Architecture

### Q1 — Count-weighting in `build_corpora_ngram.py`

The hot loop in `process_ngram_file` changes from set-based dedup to mode-dispatched accumulation:

**Current (presence):**
```python
for ngram_text, year, match_count in entries:
    if match_count < min_count:
        continue
    for s in matched_slices:
        write_buffer[s].add(ngram_text)        # set: dedup per slice
```

**New (mode-dispatched):**
```python
weight_mode = config['corpus'].get('weight_mode', 'presence')
repeat_cap  = config['corpus'].get('repeat_cap', 100)

for ngram_text, year, match_count in entries:
    if match_count < min_count:
        continue
    for s in matched_slices:
        if weight_mode == 'presence':
            write_buffer[s].add(ngram_text)
        else:  # capped_repetition
            n_repeats = min(match_count, repeat_cap)
            write_buffer[s].extend([ngram_text] * n_repeats)
```

The buffer type changes between modes (`set` for presence, `list` for capped_repetition). At flush time, both flush via `"\n".join(buffer)` — `set` iteration order is stable for the same content, so the existing flush logic works either way.

The `largest_buffer = 10000` threshold flushes by element count, which in capped_repetition will hit ~100× more often than in presence mode — that's fine, just more file appends. Could grow the buffer to 1,000,000 for capped mode to amortize, but YAGNI for v1.

### Q1 — New config profile

`config/profiles/garg_weat_china_ngram_weighted.yml` is a near-copy of `garg_weat_china_ngram.yml` with:

```yaml
paths:
  corpora_dir: "/scratch/network/yh6580/china_ngram_weighted/corpora"
  models_dir:  "/scratch/network/yh6580/china_ngram_weighted/models"
  results_dir: "/scratch/network/yh6580/gender-occup/results_garg_weat_china_ngram_weighted"
  log_dir:     "/scratch/network/yh6580/china_ngram_weighted/logs"
  figures_dir: "/scratch/network/yh6580/gender-occup/figures_garg_weat_china_ngram_weighted"
  raw_data_dir: <same as old>

corpus:
  weight_mode: capped_repetition
  repeat_cap: 100
  min_count_threshold: 1   # unchanged from old
  tokenizer: jieba
  min_words: 5
```

`embedding.model_name_template`, `time_slices`, `wordlists`, `analysis.metrics`, `embedding_source` all match the old profile so visualization and category-bias analysis flow through unchanged.

### Q1 — SLURM script updates

Two scripts get a one-line DEFAULT_CONFIGS swap:

`slurm/garg_weat_zh.slurm`:
```bash
DEFAULT_CONFIGS=(
    "config/profiles/garg_weat_renminribao.yml"
    "config/profiles/garg_weat_china_ngram_weighted.yml"   # was: garg_weat_china_ngram.yml
)
```

`slurm/describe_dataset_zh.slurm` gets the same swap.

Old presence-only profile can still be run explicitly:
```bash
sbatch slurm/garg_weat_zh.slurm config/profiles/garg_weat_china_ngram.yml
```

### Q2 — dataset_summary additions

Three changes to `scripts/common/dataset_stats.py`:

**(a) `SourceTotals` gains two fields:**

```python
@dataclass
class SourceTotals:
    # ...existing 10 fields...
    model_vocab_min: Optional[int] = None   # min across units with non-None model vocab
    model_vocab_max: Optional[int] = None   # max  "
```

`aggregate_source` populates them when at least one unit has a non-None model vocab; otherwise both remain `None`.

**(b) New private helper `_year_range(unit_name)`:**

```python
def _year_range(unit_name: str) -> str:
    """Parse year info from common unit-name shapes; returns en-dash range or '—'."""
    m = re.match(r"^(\d{4})_(\d{4})$", unit_name)           # 1940_1949
    if m:
        return f"{m.group(1)}–{m.group(2)}"
    m = re.match(r"^(\d{4})s$", unit_name)                   # 1940s (COHA)
    if m:
        start = int(m.group(1))
        return f"{start}–{start + 9}"
    m = re.match(r"^.+_(\d{4})$", unit_name)                 # 北京_2020
    if m:
        return m.group(1)
    return "—"
```

**(c) `render_markdown` adds:**

- Per-unit table: `Year range` inserted between `Unit` and `Documents`; `Tokens/doc` inserted between `Tokens` and `Raw vocab`:
  ```
  | Unit | Year range | Documents | Tokens | Tokens/doc | Raw vocab | Model vocab | Raw files | Raw bytes |
  ```
  `Tokens/doc` cell = `f"{stats.n_tokens / max(stats.n_docs, 1):,.1f}"` (or `"—"` if `n_docs == 0`).

- Training section: one new bullet after the model-files line:
  ```
  - Model vocab across {n_units} units: {model_vocab_min:,} — {model_vocab_max:,} (mean: {n_model_vocab_sum / n_units_with_vocab:,.0f})
  ```
  Suppressed entirely when all model vocabs are None (e.g. `--no-model`).

---

## Schema

### Config schema delta

| Key | Type | Default | Meaning |
|---|---|---|---|
| `corpus.weight_mode` | str | `"presence"` | One of `"presence"` (existing behavior, dedup-per-slice set) or `"capped_repetition"` (emit per (ngram, year) `min(match_count, repeat_cap)` copies) |
| `corpus.repeat_cap` | int | `100` | Per-(ngram, year) cap on repetition count. Only consulted when `weight_mode == "capped_repetition"`. |

Existing keys (`min_count_threshold`, `tokenizer`, `min_words`) unchanged.

### Per-unit table column order

```
| Unit | Year range | Documents | Tokens | Tokens/doc | Raw vocab | Model vocab | Raw files | Raw bytes |
```

For ngram sources, `Documents` is still renamed to `N-gram entries` (Task 4 behavior preserved).

### Training section bullets (Q2 addition shown last)

```
- Algorithm: Word2Vec skip-gram with negative sampling (gensim)
- `vector_size=300 · window=4 · min_count=50 · sg=1 · negative=15 · epochs=5 · seed=42`
- Model files: `<models_dir>/rmrb_{slice_name}.model` (one per time slice)
- Model vocab across 15 units: 187,432 — 432,109 (mean: 318,704)   ← NEW
```

---

## Performance & failure modes

### Disk + train cost (Q1)

For Chinese 5-gram, the per-year `match_count` distribution is roughly power-law: a small head of super-frequent 5-grams (millions of counts) and a long tail with counts in the single digits. With `repeat_cap = 100`:
- Tail (count ≤ 100): emitted exactly `count` times — no change in fidelity vs unclamped.
- Head (count > 100): emitted 100 times. Information lost: relative ordering above 100.

Disk size roughly scales as `sum_over_(ngram, year) of min(count, 100)` vs presence-only's `|unique ngrams per slice|`. Empirically this is 20–50× for English Books 5-gram; expect a similar order for Chinese.

Training time grows in proportion to emitted tokens. A presence-only Chinese 5-gram training run that takes ~hours per slice will become tens-of-hours per slice. The user runs these on the server overnight; not a blocker.

### Failure modes

- **`weight_mode` typo in config:** Hard error. Validate in builder: if value not in `{"presence", "capped_repetition"}`, raise `ValueError`. Memory note: user prefers breaks over silent defaults.
- **`repeat_cap` set to 0 or negative:** Treat as 1 (one copy emitted). Documented in builder.
- **Corpora dir collision** (someone runs the weighted builder pointing at an existing presence-only `corpora_dir`): builder appends to existing files, mixing presence + repetition records. Prevention: the new profile uses fresh dirs (`*_weighted` suffix). Document in spec; no enforcement code.
- **dataset_summary `Year range` for unrecognized unit shapes:** Returns `"—"`. Reported as a per-unit cell, doesn't break the render.
- **Backwards compat:** Existing presence-only china-ngram models stay loadable. Old profile still works (yields presence corpora as before). Old SLURM behavior preserved when old profile passed explicitly.

---

## Testing plan

### Q1 — `tests/test_build_corpora_ngram.py` (NEW)

The repo currently has `tests/test_build_corpora.py` covering generic build helpers and `tests/test_build_corpora_ngram_en.py` covering English ngram specifics. Add a peer file for the Chinese ngram, focused on weighting.

1. **`test_presence_mode_unchanged_default`** — config without `weight_mode`, fixture with one ngram with `match_count=5`. Expect one line in the output (set dedup). Lock in backwards compat.
2. **`test_capped_repetition_emits_count_copies`** — `weight_mode=capped_repetition, repeat_cap=100`. One ngram, count=3. Expect 3 lines.
3. **`test_capped_repetition_clamps_at_cap`** — count=5000, cap=100. Expect 100 lines.
4. **`test_capped_repetition_sums_across_years_in_slice`** — same ngram in 1942 (count=80) and 1948 (count=200) both in `1940_1949` slice. Expect `80 + 100 = 180` lines.
5. **`test_min_count_threshold_filter_runs_before_cap`** — `min_count_threshold=10, repeat_cap=100`, ngram with count=5. Expect 0 lines (filtered).
6. **`test_invalid_weight_mode_raises`** — `weight_mode="bogus"` → `ValueError`.

Tests use a tmpdir + a synthetic ngram file (one line, v3 tab-separated). No live download; no pyarrow.

### Q2 — appended to `tests/test_dataset_stats.py`

7. **`test_year_range_slice_format`** — `_year_range("1940_1949")` → `"1940–1949"`.
8. **`test_year_range_coha_decade`** — `_year_range("1940s")` → `"1940–1949"`.
9. **`test_year_range_province_year`** — `_year_range("北京_2020")` → `"2020"`.
10. **`test_year_range_unknown_returns_dash`** — `_year_range("北京")` → `"—"`.
11. **`test_aggregate_source_model_vocab_range`** — per_unit with vocabs `[100, 200, None, 300]` → `model_vocab_min=100, model_vocab_max=300`.
12. **`test_aggregate_source_model_vocab_range_all_none`** — all None → both `None`.
13. **`test_render_markdown_includes_year_range_and_tokens_per_doc_columns`** — render output contains `"Year range"` header and a value like `"1940–1949"`; contains `"Tokens/doc"` header.
14. **`test_render_markdown_training_section_shows_model_vocab_range`** — render output contains `"Model vocab across"`.

Run with `python -m pytest tests/test_build_corpora_ngram.py tests/test_dataset_stats.py -v` — expect ~14 new passing on top of the existing surface.

---

## CLI / runbook

After this ships, the user's workflow on the Princeton server to produce count-weighted china ngram models is:

```bash
# 1. Build the weighted corpora (uses the new profile)
sbatch slurm/build_corpus.slurm config/profiles/garg_weat_china_ngram_weighted.yml

# 2. Train (uses the new profile's corpora_dir + models_dir)
sbatch slurm/train.slurm config/profiles/garg_weat_china_ngram_weighted.yml

# 3. Analyze + visualize (default zh script now defaults to weighted)
sbatch slurm/garg_weat_zh.slurm

# 4. Generate methods-section summary
sbatch slurm/describe_dataset_zh.slurm
```

For comparison runs against the old presence-only models:
```bash
sbatch slurm/garg_weat_zh.slurm config/profiles/garg_weat_china_ngram.yml
```

---

## Out of scope (future work)

- **Per-year frequency normalization** using `totalcounts-5`. Would require reading the totalcounts file once per source and dividing per-year `match_count` by `totalcounts[year]` before applying the cap. Adds a knob; uses the file the user mentioned. Defer until we see the capped-repetition results.
- **OOV rate per unit.** Compute the fraction of corpus tokens dropped by Word2Vec's `min_count` filter — requires either a corpus re-scan against model vocab or persisting Word2Vec's `corpus_total_words` and computing `1 - sum(vocab counts) / corpus_total_words`. Useful methods detail; defer.
- **Per-source weighting metadata in dataset_summary.** Q1's `weight_mode` could surface in the Training section ("Token weighting: capped repetition, cap=100") so the methods paragraph cites the scheme directly. Three-source taxonomy (`raw running text` / `presence-only ngram types` / `count-weighted ngram (cap=N)`). Adds ~15 LOC. Useful; defer.
- **Builder + train SLURM orchestration for the new profile.** A single `slurm/china_ngram_weighted_rebuild.slurm` that runs build → train end-to-end would be convenient but is a workflow convenience, not a correctness requirement. Defer.

---

## Acceptance criteria

1. `python -m pytest tests/test_build_corpora_ngram.py tests/test_dataset_stats.py -v` passes (all 14+ new tests).
2. Running `python -m scripts.data_prep.build_corpora_ngram --config=config/profiles/garg_weat_china_ngram_weighted.yml --slice=<one>` on the server produces a non-empty `corpora_dir/<slice>/corpus_*.txt`; line count is materially larger than the equivalent presence-only build (≥ 10× per slice for any non-trivial year).
3. `sbatch slurm/garg_weat_zh.slurm` with no args invokes the weighted profile (verified in the SLURM log header listing `DEFAULT_CONFIGS`).
4. `dataset_summary.md` for any source shows `Year range` and `Tokens/doc` columns and (for sources with at least one trained model) a `Model vocab across N units` line in the Training section.
5. Old presence-only profile still works when passed explicitly — backwards compat preserved.
