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
`analysis.ideation_sign` (+ `analysis.consistent_occupations`). Writes to
`results_dir` (both parquet **and** CSV):

**Preprocessing**
1. Keep `in_vocab == True` rows.
2. `year = _decade_start_year(unit_name)` (reused from `visualize.py`; handles
   `1990s` and `1940_1949`). Drop rows where `year is None` (provincial units)
   with an INFO log — those corpora are out of scope.
3. `signed_rnd = rnd * ideation_sign[category]` (reuse
   `visualize.apply_ideation_sign`). All lenses operate on `signed_rnd` so
   direction matches the plotted trend.

**(a) `word_drivers_long.{parquet,csv}`** — one row per `(category, year, word)`:

| category | year | unit_name | occupation | rnd | signed_rnd | cat_mean_signed | deviation |
|---|---|---|---|---|---|---|---|

- `cat_mean_signed` = mean `signed_rnd` over the word set used for that
  `(category, slice)` — i.e. the plotted line's value.
- `deviation = signed_rnd − cat_mean_signed` → **lens 3** (who pulls the level
  male/female that year). Averaging `signed_rnd` within any `(category, year)`
  reproduces the figure point exactly.

**(b) `word_drivers_summary.{parquet,csv}`** — one row per `(category, word)`:

| category | occupation | first_year | last_year | signed_first | signed_last | delta | contribution | slope | present_both | n_slices |
|---|---|---|---|---|---|---|---|---|---|---|

- `delta = signed_last − signed_first` → **lens 2** (raw per-word change).
- `contribution = delta / N` → **lens 1**; sums to the change in the plotted
  line over the endpoint-consistent set.
- `slope` = OLS of `signed_rnd` on `year` over all slices the word appears in —
  a churn-robust companion column (no figure).
- Rows ranked within category by `|contribution|` (NaN-contribution words last).

### Endpoint / churn handling (the one subtlety)

For contributions to sum exactly to Δ(plotted mean), the word set must be
identical at both endpoints; vocab churn breaks this.

- `first_year` / `last_year` are **per category** = min/max slice where that
  category has ≥1 in-vocab word.
- **Endpoint-consistent set** = words in-vocab at *both* endpoints.
  `delta` / `contribution` are computed only for those (`present_both = True`);
  `N` = size of that set. Then `Σ contribution` = change in the
  consistent-set mean.
- Words present in only some slices still appear in the long table and the
  trajectory figure, but get `present_both = False`, `contribution = NaN`, and
  are excluded from the decomposition, contribution bars, and slope chart.
- **`analysis.consistent_occupations` honored**: when `true`, `cat_mean_signed`
  and the decomposition both restrict to the global consistent set (words
  in-vocab in *all* slices), matching how the summary/plot is built, so the
  table's `cat_mean_signed` reproduces the figure. When `false` (default),
  `cat_mean_signed` is per-slice in-vocab mean and the decomposition uses the
  two-endpoint set.

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
- Bars/slope/trajectory restrict to `present_both` words; heatmap shows all
  in-vocab words in the dimension.
- File extension follows existing figure convention (PDF).

## Wiring

Add to **both** `slurm/garg_weat_all_sources.slurm` and
`slurm/garg_weat_zh.slurm`, immediately after the existing
`visualize main` block succeeds for a config (so primary figures are already
safe). **Non-fatal**: a driver failure logs a WARN and a distinct status, then
`continue`s — it never regresses the primary deliverable.

```bash
if ! python -m scripts.analyze_word_drivers --config="$CONFIG"; then
    echo "  WARN: analyze_word_drivers failed (primary figures unaffected)"
    STATUSES+=("word_drivers_analyze_warn"); continue
fi
if ! python -m scripts.visualize_word_drivers --config="$CONFIG"; then
    echo "  WARN: visualize_word_drivers failed (primary figures unaffected)"
    STATUSES+=("word_drivers_visualize_warn"); continue
fi
```

(Exact placement adapts to each loop's status-array style; the two scripts are
also runnable standalone for ad-hoc single-corpus inspection.)

## Testing — `tests/test_analyze_word_drivers.py`

On a small synthetic `garg_weat_rnd_long`-shaped fixture:

1. `mean(signed_rnd)` within `(category, year)` equals `cat_mean_signed`.
2. `Σ contribution` over `present_both` words equals
   `cat_mean_signed(last) − cat_mean_signed(first)` on the consistent set.
3. `ideation_sign` flips `family` (sign −1 applied to `signed_rnd`).
4. Endpoint churn: a word OOV at an endpoint gets `present_both=False`,
   `contribution=NaN`, and is excluded from the decomposition sum.
5. `deviation = signed_rnd − cat_mean_signed`.
6. `consistent_occupations=True` restricts `cat_mean_signed` to the global
   consistent set.

Visualization is smoke-tested (figures written, no exception) following the
pattern in existing `tests/test_visualize_*` files.

## Files touched

- **new** `scripts/analyze_word_drivers.py`
- **new** `scripts/visualize_word_drivers.py`
- **new** `tests/test_analyze_word_drivers.py`
- **edit** `slurm/garg_weat_all_sources.slurm` (+2 calls, non-fatal)
- **edit** `slurm/garg_weat_zh.slurm` (+2 calls, non-fatal)
