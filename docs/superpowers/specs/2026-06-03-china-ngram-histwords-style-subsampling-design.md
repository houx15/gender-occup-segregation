# China Ngram HistWords-Style Per-Year Subsampling — Design Spec

**Date:** 2026-06-03
**Status:** Approved (pending user review of this file)
**Supersedes:** [`2026-06-02-china-ngram-weighting-and-summary-enhancements-design.md`](2026-06-02-china-ngram-weighting-and-summary-enhancements-design.md) — weighting portion only. The `dataset_summary.md` enhancements in that spec (year range, tokens/doc, model-vocab range) are unaffected and have already shipped.

**Purpose:** Replace the methodologically-wrong `capped_repetition` weighting mode (which compresses the dynamic range of count signal via `min(match_count, repeat_cap)`) with HistWords-style per-year token-budget subsampling on the count-proportional stream.

---

## Problem with the previous approach

The 2026-06-02 spec introduced `weight_mode: capped_repetition`, which emits `min(match_count, repeat_cap)` copies of each `(ngram, year)` row. This is wrong on methods grounds:

- It **compresses the dynamic range** of the count signal. A 5-gram with `match_count=50,000` and one with `match_count=100` (or `30`) end up with identical contribution. The relative-frequency structure we wanted from count-weighting is destroyed at the top of the distribution where most of the corpus signal lives.
- It is **not what HistWords does** for Google N-Gram SGNS. Re-reading Hamilton et al. (2016) Appendix A: for SGNS on Google N-Gram data, they materialize a per-year token stream (each n-gram emitted `match_count` times), then downsample any year whose stream exceeds 10⁹ tokens to exactly 10⁹ via uniform subsampling. Per-year cap, not per-row cap.
- The 114 GB of partial `corpora_weighted/` built before disk-quota crash is unusable under this corrected methodology and must be reclaimed.

The PPMI+SVD path in HistWords is a different beast (operates on the full co-occurrence matrix, no stream) and is not relevant — our SGNS arm is the one being corrected.

---

## Goal

- A new config profile `config/profiles/garg_weat_china_ngram_subsampled.yml` that, when fed through the existing `build_corpora_ngram` → `train_embeddings` flow, produces per-year-capped subsampled China Ngram corpora and models — independent of and comparable to the existing presence-only ones.
- The corpus builder honors a new `corpus.weight_mode: per_year_capped` value plus `corpus.per_year_token_cap` and `corpus.rng_seed`. The `presence` mode is preserved unchanged.
- The `capped_repetition` mode (introduced 2026-06-02) is removed entirely — code, tests, profile, constants.
- The 2026-06-02 `garg_weat_china_ngram_weighted.yml` profile is deleted along with its on-disk artifacts (`corpora_weighted/`, `models_weighted/`).
- The existing per-slice SLURM driver is renamed and re-pointed at the new profile.
- The two Princeton `*_zh.slurm` scripts (`garg_weat_zh.slurm`, `describe_dataset_zh.slurm`) default to the new subsampled profile.

## Non-goals

- Not touching the English ngram pipeline. English embeddings come from pre-trained HistWords (Hamilton et al. 2016); the EN builder is dormant.
- Not touching RMRB / Weibo / provincial newspaper / COHA. They train on real running text — token frequency is already the natural signal; no `weight_mode` decision applies.
- Not implementing true two-pass reservoir subsampling. The unbiased one-pass Bernoulli-on-fractional-part has identical expectation and is much cheaper. See Approaches.
- Not changing analyze / visualize / describe pipelines. They read models, not corpora — orthogonal to this work.
- Not re-running training under the (deleted) `capped_repetition` mode to compare. It's wrong on methods grounds and the partial build crashed before producing models anyway. No artifacts to compare against.

---

## Methodological foundation

Per Hamilton et al. (2016) Appendix A: for SGNS on Google N-Gram data, the count-weighted training stream is materialized by emitting each n-gram `match_count` times into the year `y`'s stream, then any year whose total stream exceeds 10⁹ tokens is **uniformly subsampled down to 10⁹**. Years below the cap pass through in full. The reason given in the paper: "SGNS works with text streams and not co-occurrence counts" — the count signal has to be expressed as a token stream for SGNS, but bounded somehow or modern years (with ~10¹¹+ tokens) blow out memory and dominate training.

For our China 5-gram case we adopt the same algorithm with two adjustments:
1. **Cap value 10⁸** (one order of magnitude below the paper's 10⁹). Justified because chi_sim Google Books v3 is materially smaller than English — proportional compression is comparable to English at 10⁹. Disk envelope: ~3 GB per slice, ~50 GB total across 17 slices. Comparable footprint to the existing presence-only build (57 GB).
2. **Unbiased one-pass Bernoulli sampling** rather than true uniform subsample of the materialized stream. Identical expectation; one pass instead of two; no per-year buffer in RAM. See Approaches A vs B.

---

## Decisions

| # | Decision | Choice | Rationale |
|---|---|---|---|
| 1 | Per-year token cap value | `per_year_token_cap = 100_000_000` (10⁸) | Chi_sim Google ngram is ~10× smaller than English per Hamilton et al. Proportional compression at 10⁸ here ≈ 10⁹ on English. Disk ≈ 50 GB total / 3 GB per slice. |
| 2 | Sampling primitive | Floor + Bernoulli(fractional part), seeded | `n_emit = floor(match_count × scale) + Bernoulli(frac(match_count × scale))`. Unbiased (`E[n_emit] = match_count × scale`), one-pass, deterministic with `rng_seed`. |
| 3 | Old `capped_repetition` mode | **Removed entirely.** Code, tests, profile, constant. | Methods-wrong, not just suboptimal. Keeping it invites future-you to use it. Saves test maintenance burden. |
| 4 | Coexistence with presence-only | New profile + fresh dirs (`*_subsampled`) | Existing presence-only china-ngram models stay untouched for direct A/B comparison in the methods section. |
| 5 | `totalcounts-5` use | **Required input.** Single pre-pass to parse `dict[year, total_words]`. Cached in memory for entire build. | Authoritative per-year totals ship with the corpus. Computing on-the-fly from shards (Approach C) is an extra full pass for no statistical gain. |
| 6 | RNG seed location | `corpus.rng_seed: 42` in profile YAML | Reproducibility. Anyone re-running the build with the same profile + same input shards gets byte-exact corpus files. |
| 7 | Naming | profile `garg_weat_china_ngram_subsampled.yml`; `weight_mode: per_year_capped`; dirs `corpora_subsampled/`, `models_subsampled/`, `results_garg_weat_china_ngram_subsampled/`; `embedding_source: china_ngram_subsampled` | `subsampled` is the user-facing name (short, methods can spell out). `per_year_capped` is the internal mechanism (precise for code-reading). |
| 8 | Cleanup of 2026-06-02 artifacts | Delete `config/profiles/garg_weat_china_ngram_weighted.yml`, `/scratch/network/yh6580/gender-occup/corpora_weighted/` (114 GB partial), any `models_weighted/`. Mark 2026-06-02 spec superseded. | Reclaims disk, removes methods-wrong code path, leaves clean A (presence) vs B (subsampled) story. |
| 9 | SLURM driver | Rename `slurm/build_train_china_ngram_weighted_per_slice.slurm` → `..._subsampled_per_slice.slurm`. Default-config flip. | Mechanism (per-slice build → train → delete) unchanged; just retargeted. |
| 10 | `*_zh.slurm` defaults | `garg_weat_zh.slurm` and `describe_dataset_zh.slurm` swap their second default config from `..._weighted.yml` to `..._subsampled.yml` | Going-forward "the china ngram run" means the subsampled one. |

---

## Approaches considered

### A. Single-pass with `totalcounts-5` pre-load — CHOSEN

```
read totalcounts-5 → year_total : dict[year, total_words]   (one-time, tiny)
for each shard:
    decompress
    for each (ngram, year, match_count) row:
        if match_count < min_count_threshold: skip
        if year not in year_total: KeyError                  (fail fast)
        scale = min(1.0, per_year_token_cap / year_total[year])
        expected = match_count * scale
        n_emit = floor(expected) + Bernoulli(frac(expected))
        for each slice covering year:
            buffer[slice].extend([ngram] * n_emit)
            if len(buffer[slice]) > 10000: flush
```

One pass over all shards. Constant memory (the totalcounts dict is ~100 lines). Bernoulli step uses a single `numpy.random.Generator` seeded from config.

### B. Two-pass true-uniform-subsample (paper-strict)

Pass 1: materialize the full count-weighted stream per year to temp files. Pass 2: for each year whose temp file exceeds 10⁸ lines, uniform-subsample down to exactly 10⁸; otherwise keep all.

Statistically equivalent to A in expectation. Costs ~5+ TB of temp storage (the unweighted materialized stream is huge) or holding one year at a time in RAM (tens of GB). No upside given A's Bernoulli has the same expectation.

### C. Compute per-year totals from shards themselves (no `totalcounts-5`)

Pass 1: walk all shards summing `match_count` per year to build `year_total[year]` from scratch. Pass 2: same as A's main loop.

Robust if we don't trust `totalcounts-5` (e.g., if chi_sim shards have been pre-filtered for something the totalcounts file doesn't reflect). Costs a full extra pass over ~150 GB of decompressed shards just for counting. Overkill — `totalcounts-5` ships with the corpus and is authoritative.

**Trade-off summary:** A is paper-faithful in expectation, one-pass, constant memory, trivially seedable. B and C buy us no statistical gain at significant runtime/storage cost. **A is chosen.**

---

## Architecture

A new `per_year_capped` `weight_mode` replaces `capped_repetition` inside `process_ngram_file`. A new module `scripts/data_prep/ngram_totalcounts.py` is responsible for parsing `raw_ngrams/totalcounts-5` into a `dict[year, total_words]` and is called once at the top of `build_corpora` before the shard loop. Each shard is processed in a single pass; per-row `n_emit` is computed via a seeded NumPy `Generator` using the unbiased floor-plus-Bernoulli formula. The existing buffer-flush-by-slice and per-shard corpus-file mechanics stay identical. The new profile `garg_weat_china_ngram_subsampled.yml` clones presence-only paths to `*_subsampled/` directories. The existing per-slice SLURM driver retains its build→train→delete structure; only the profile pointer and filename change.

The new module is small (~30 LOC) and isolated: it takes a file path, returns a `dict[int, int]`, and has no dependencies on the rest of the pipeline. This makes it independently testable and makes `build_corpora_ngram.py` easier to reason about.

---

## Components

```
scripts/
├── data_prep/
│   ├── ngram_totalcounts.py            # NEW (~30 LOC): parse totalcounts-5 → {year: total_words}
│   └── build_corpora_ngram.py          # MODIFY: delete capped_repetition branch; add per_year_capped branch
config/
└── profiles/
    ├── garg_weat_china_ngram_subsampled.yml   # NEW (replaces _weighted)
    └── garg_weat_china_ngram_weighted.yml     # DELETE
slurm/
├── build_train_china_ngram_weighted_per_slice.slurm
│                                       # RENAME → ..._subsampled_per_slice.slurm
│                                       # CONFIG path: ..._weighted.yml → ..._subsampled.yml
├── garg_weat_zh.slurm                  # MODIFY: default config arg #2 → ..._subsampled.yml
└── describe_dataset_zh.slurm           # MODIFY: default config arg #2 → ..._subsampled.yml
tests/
├── test_ngram_totalcounts.py           # NEW: parse + KeyError + malformed-line handling
└── test_build_corpora_ngram.py         # MODIFY: drop capped_repetition tests; add per_year_capped tests
```

---

## Data flow

```
raw_ngrams/totalcounts-5  ──→  parse_totalcounts()  ──→  {1940: 1.2e9, 1941: 9.5e8, ...}
                                                                  │
                                                                  ▼
                                                    (passed into process_ngram_file
                                                     once per build, kept in scope)
                                                                  │
raw_ngrams/5-NNNNN-of-00105.gz  ──→  decompress()                 │
              │                                                   │
              ▼                                                   │
       parse_ngram_line_v3()                                      │
              │                                                   │
              ▼                                                   │
       (ngram, year, match_count) ───────┬───────────────────────┤
                                          │                       │
                                          ▼                       │
                                  match_count < min_count?  ──→ skip
                                          │                       │
                                          ▼                       │
                               year not in year_total?  ──→ raise KeyError
                                          │                       │
                                          ▼                       │
                                  scale = min(1.0,                │
                                              cap / year_total[year])
                                  expected = match_count × scale
                                  n_emit  = floor(expected)
                                          + Bernoulli(frac(expected))
                                          │
                                          ▼
                            for slice covering year:
                                buffer[slice].extend([ngram] × n_emit)
                                if len(buffer[slice]) > 10_000: flush
                                          │
                                          ▼
                       corpora_subsampled/<slice>/corpus_NNNNN.txt
```

---

## Error handling

- **Missing year in `totalcounts-5`** (shard contains a year totalcounts didn't list): raise `KeyError(year)` — fail fast. Silent emit-at-scale-1.0 would skew the cap.
- **Unknown `weight_mode`** in config: `ValueError`, same pattern as today.
- **`per_year_token_cap` missing from profile**: use default `1_000_000_000` (10⁹) and log a warning. The shipped profile sets it explicitly to `100_000_000`.
- **`rng_seed` missing from profile**: default to `0` and log info. Shipped profile sets it to `42`.
- **`totalcounts-5` file missing**: raised only when `weight_mode == per_year_capped`. The `presence` mode never reads it.
- **Existing `capped_repetition` profile/config still in user's tree**: build fails with `ValueError` from the unknown weight_mode check. The 2026-06-02 profile is being deleted as part of this work, so no live consumer expected.

---

## Testing strategy

TDD with subagent-driven-development (the same pattern used for the 2026-06-02 work). One module test file (new) and one builder test file (modified).

**`tests/test_ngram_totalcounts.py`** (~4 tests):
1. Parse a v3-format totalcounts string (tab-separated `year,words,pages,books` cells) — assert dict shape + a couple of known values.
2. Empty input → empty dict, no error.
3. Malformed cell (missing comma, non-int year) → skipped, other cells parsed normally.
4. File-path overload (read from disk) — sanity that the I/O wrapper works.

**`tests/test_build_corpora_ngram.py`** (delete `capped_repetition` cases; add ~6 `per_year_capped` cases):
1. **Pass-through when year_total ≤ cap**: a year with `total_words = 5e7` and cap = 1e8 → scale = 1.0 → every row emitted exactly `match_count` times.
2. **Scale-down when year_total > cap**: a year with `total_words = 1e9` and cap = 1e8 → scale = 0.1 → for an input row with `match_count = 100`, deterministic seeded RNG produces a specific output count; assert byte-exact corpus file against snapshot.
3. **Unbiasedness property**: synthetic year with `total_words = 1e10`, scale = 0.01, single ngram with `match_count = 10000`, run 1000 trials with different seeds, assert empirical mean `n_emit` within 3σ of expected = 100.
4. **Missing year in totalcounts raises KeyError** with the year in the message.
5. **Unknown `weight_mode` raises ValueError** (regression for existing behavior).
6. **`weight_mode == presence` does not require totalcounts-5** (file path can be absent / file can be missing).

---

## Operational changes (one-time, on Princeton)

1. `scancel` any active `cnw_per_slice` or `build_corpus_zh` job operating on the weighted profile.
2. `rm -rf /scratch/network/yh6580/gender-occup/corpora_weighted` — reclaims 114 GB.
3. `rm -rf /scratch/network/yh6580/gender-occup/models_weighted` if present (likely empty or absent — training never completed).
4. `git pull` to land this work, then `sbatch slurm/build_train_china_ngram_subsampled_per_slice.slurm` to do the full 17-slice sweep at the new methodology. Expected peak ~3 GB per slice, total ~50 GB, wall-clock dominated by training (gensim) not corpus building.

The 2026-06-02 spec and plan stay on disk as historical records with supersession notes added at the top of each.

---

## Out of scope (deliberately)

- **PPMI+SVD path.** Not the embedding family we're using here; HistWords mentions it as the count-based representation but our SGNS arm is the one being corrected. **Future robustness-check work:** the user flagged 2026-06-03 that PPMI+SVD will be added later as a second training method to compare against the SGNS arm — not as a replacement, not as alternative weighting. Structurally different (operates on a weighted co-occurrence matrix rather than a token stream), so it needs a separate training module (`scripts/train_embeddings_ppmi_svd.py`), an `embedding.algorithm` config field, and either a separate analysis arm or an algorithm-aware orchestrator. The subsampled stream this spec produces is the right input format for both methods — PPMI co-occurrence counting reads the same `corpora_subsampled/<slice>/corpus_*.txt` files, just differently. Tracked in memory `project_garg_weat_expansion.md`.
- **English ngram pipeline.** English embeddings come from pre-trained HistWords vectors (Hamilton et al. 2016 release). The local `build_corpora_ngram_en.py` is dormant.
- **Word-subsampling parameter (`sample`) in gensim's Word2Vec.** Gensim's SGNS already applies word-frequency subsampling at training time by default — that's the second mechanism HistWords mentions. We keep gensim's default behavior; no config-level knob.
- **Re-running analysis / visualize / describe.** They consume models, not corpora — re-run them after Step 4 completes against the new `..._subsampled` profile per the runbook pattern.

---

## References

- Hamilton, W. L., Leskovec, J., & Jurafsky, D. (2016). Diachronic Word Embeddings Reveal Statistical Laws of Semantic Change. *ACL 2016*. Appendix A documents the per-year 10⁹-token cap for SGNS on Google N-Gram data.
- Google Books N-Gram Viewer dataset v3 format: <https://storage.googleapis.com/books/ngrams/books/datasetsv3.html>
- 2026-06-02 predecessor spec (superseded for weighting): [`2026-06-02-china-ngram-weighting-and-summary-enhancements-design.md`](2026-06-02-china-ngram-weighting-and-summary-enhancements-design.md)
