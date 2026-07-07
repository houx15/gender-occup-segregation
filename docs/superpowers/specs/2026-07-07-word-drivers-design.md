# Word-level ideation drivers — design

**Date:** 2026-07-07
**Status:** Approved (brainstorm), pending implementation plan

## Problem

Our Garg-WEAT pipeline plots a per-dimension "gender ideation" trend for four
corpora (Chinese: china-ngram, Renminribao; English: COHA, Google en-ngram),
across three dimensions (`leadership`, `science`, `family`). Each plotted point
is an aggregate — the mean of one relative-norm-distance (RND) score per
occupation word. Readers currently cannot see **which words** produce that
level, or **which words drive its change** over time.

This feature surfaces and decomposes that aggregate: a "show your work" table
(year × word scores) plus figures that rank the words driving each dimension.

## Key fact: the per-word data already exists

The plotted ideation point for a `(category, time-slice)` is exactly:

```
ideation(category, slice) = mean over in-vocab words of [ sign(category) * rnd(word, slice) ]
```

Every ingredient already lives in `garg_weat_rnd_long.parquet`, written by
`analyze_category_bias` for all four corpora (each profile has
`metrics: [rnd, cohens_d]`, so the `rnd` long file is always produced):

| column | meaning |
|---|---|
| `unit_name` | time-slice label (`1990s`, `1940_1949`, …) |
| `category` | dimension (`leadership` / `science` / `family`) |
| `occupation` | the word |
| `rnd` | `‖v_word − c_male‖ − ‖v_word − c_female‖` (Garg sign) |
| `in_vocab` | whether the word was in that slice's vocab |

Per-dimension sign lives in config `analysis.ideation_sign`
(`leadership: 1, science: 1, family: -1`). So this feature is **pure pandas
over an existing parquet** — no re-training, no model loading.

## Non-goals (YAGNI)

- Provincial / weibo arms (cross-sectional, no time axis).
- Per-(occupation × individual-gender-word) distance breakdown.
- Any new embedding computation.
- Wiring into `run_pipeline.sh` — `garg_weat` mode does not run through it
  (its Step 4 only knows `prestige/weat/garg`). Production path is the two
  garg_weat slurm loops.

## Architecture

Two separate scripts, mirroring the repo's `analyze_* → visualize` split, both
invoked per-config inside the existing garg_weat slurm loops.

### Component 1 — `scripts/analyze_word_drivers.py` (tables)

CLI: `python -m scripts.analyze_word_drivers --config=<profile>.yml`

Reads `results_dir/garg_weat_rnd_long.parquet` and config
`analysis.ideation_sign`. Writes to `results_dir` (both parquet **and** CSV):

**Preprocessing**
1. Keep `in_vocab == True` rows.
2. `year = _slice_start_year(unit_name)` (mirrors `visualize._decade_start_year`;
   handles `1990s` and `1940_1949`). Drop rows where `year is None` (provincial
   units) with an INFO log — those corpora are out of scope.
3. `signed_rnd = rnd * ideation_sign[category]` (default sign 1). All lenses
   operate on `signed_rnd` so direction matches the plotted trend.
4. **Restrict to the per-category global consistent set** — words in-vocab in
   ALL retained slices of the category. This is not optional and is *not* keyed
   off any config flag: it exactly mirrors `category_summary.compute_consistent_set`,
   which `analyze_category_bias.py` applies **unconditionally** when building the
   plotted `mean_rnd`. Operating on the same set is what makes `cat_mean_signed`
   reproduce the published line (and see "Why the consistent set" below).

**(a) `word_drivers_long.{parquet,csv}`** — one row per `(category, year, word)`:

| category | year | unit_name | occupation | rnd | signed_rnd | cat_mean_signed | deviation |
|---|---|---|---|---|---|---|---|

- Rows are only for consistent-set words (step 4). `cat_mean_signed` = mean
  `signed_rnd` over that set for the `(category, year)` — i.e. the plotted
  line's value. Averaging `signed_rnd` within any `(category, year)` reproduces
  the figure point exactly.
- `deviation = signed_rnd − cat_mean_signed` → **lens 3** (who pulls the level
  male/female that year).

**(b) `word_drivers_summary.{parquet,csv}`** — one row per `(category, word)`:

| category | occupation | first_year | last_year | signed_first | signed_last | delta | contribution | slope |
|---|---|---|---|---|---|---|---|---|

- `first_year` / `last_year` = min/max slice of the category.
- `delta = signed_last − signed_first` → **lens 2** (raw per-word change).
- `contribution = delta / N`, `N` = size of the consistent set → **lens 1**;
  `Σ contribution` equals the change in the plotted line exactly.
- `slope` = OLS of `signed_rnd` on `year` over the category's slices — a
  companion column (no figure).
- Rows ranked within category by `|contribution|`.

### Why the consistent set (and why there is no churn handling)

The plotted `mean_rnd` is built by `category_summary.build_summary` over
`compute_consistent_set(...)` — words in-vocab in ALL units, per category —
and `analyze_category_bias.py:155` calls that **unconditionally**. There is no
`consistent_occupations` toggle in the garg_weat path (that key is only read by
the separate single-wordlist `analyze_garg.py`, and no garg_weat profile sets
it). So to reproduce the published line, the word-drivers tables must use the
same set — always.

A happy consequence: the consistent set is churn-free by construction (every
word is in-vocab in every slice), so every word has a well-defined value at both
endpoints. There is no `present_both` / NaN-contribution bookkeeping — every
consistent-set word contributes, and `Σ contribution` equals `Δ cat_mean_signed`
exactly. The word-drivers script recomputes this set itself from the long
parquet (same units, same rows → same set as `build_summary` used).

### Component 2 — `scripts/visualize_word_drivers.py` (figures)

CLI: `python -m scripts.visualize_word_drivers --config=<profile>.yml`

Reads the two tables; reuses `visualize.py` helpers (`_configure_fonts`,
`get_figure_path`, `L`, font handling for Chinese). **One file per
(dimension × form)** → 12 files per corpus, into `figures_dir`:

| form | file stem | lens | content |
|---|---|---|---|
| contribution bars | `word_drivers_contribution_<cat>` | 1 | diverging horizontal bars, top-N by `|contribution|`, sorted; annotated Σ = Δideation |
| slope / dumbbell | `word_drivers_slope_<cat>` | 2 | one row/word, `signed_first`→`signed_last`, sorted by `|delta|`, colored by direction |
| word × year heatmap | `word_drivers_heatmap_<cat>` | 3 | rows=words (sorted by `delta`), cols=slices, color=`signed_rnd` (diverging, symmetric vmax) |
| trajectory small multiples | `word_drivers_trajectory_<cat>` | overview | all words faint, top-N movers bold+labelled, `cat_mean_signed` overlaid |

- `top_n` configurable via `analysis.word_drivers.top_n` (default 20).
- All figures draw from the consistent-set words in the tables (bars/slope/
  trajectory cap to `top_n` by |contribution| or |delta|; heatmap shows all
  consistent-set words in the dimension).
- File extension follows existing figure convention (PDF).

## Wiring

Add to **both** `slurm/garg_weat_all_sources.slurm` and
`slurm/garg_weat_zh.slurm`, at the **ok-tail** of each config iteration — after
the primary figures are validated and `STATUSES+=("ok")` is recorded.
**Non-fatal**: a driver failure logs a WARN and lets the loop advance; it must
NOT append to any status array or `continue` (that would desync the per-config
arrays).

```bash
if python -m scripts.analyze_word_drivers --config="$CONFIG" \
    && python -m scripts.visualize_word_drivers --config="$CONFIG"; then
    echo "  word_drivers: ok ($RESULTS_DIR/word_drivers_*.{parquet,csv})"
else
    echo "  WARN: word_drivers step failed (primary figures unaffected)"
fi
```

The two scripts are also runnable standalone for ad-hoc single-corpus inspection.

## Testing — `tests/test_analyze_word_drivers.py`

On a small synthetic `garg_weat_rnd_long`-shaped fixture (include a churning
word — in-vocab in only some slices — to exercise the consistent-set filter):

1. `mean(signed_rnd)` within `(category, year)` equals `cat_mean_signed`.
2. `Σ contribution` within a category equals `cat_mean_signed(last) −
   cat_mean_signed(first)` (holds exactly on the churn-free consistent set).
3. `ideation_sign` flips `family` (sign −1 applied to `signed_rnd`).
4. The churning word is excluded from BOTH tables (not in the consistent set);
   the consistent-set mean at each slice ignores it.
5. `deviation = signed_rnd − cat_mean_signed`.
6. `_slice_start_year` drops province / province-year units (returns None).

Visualization is smoke-tested (figures written, no exception) following the
pattern in existing `tests/test_visualize_*` files.

## Files touched

- **new** `scripts/analyze_word_drivers.py`
- **new** `scripts/visualize_word_drivers.py`
- **new** `tests/test_analyze_word_drivers.py`
- **edit** `slurm/garg_weat_all_sources.slurm` (+2 calls, non-fatal)
- **edit** `slurm/garg_weat_zh.slurm` (+2 calls, non-fatal)
