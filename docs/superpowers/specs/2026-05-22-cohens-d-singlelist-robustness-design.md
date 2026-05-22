# Cohen's-d (single-list) robustness check for Garg-WEAT

**Date:** 2026-05-22
**Status:** Design — approved for planning

## Motivation

Our gender-norm bias measurement currently exists in two methodological
lineages:

1. **Original Cohen's d (WEAT) — two contrasting target wordlists.**
   `analyze_weat.py` builds a female−male gender axis and, per dimension,
   projects **two** opposing target sets (e.g. `family` vs. `work`,
   `leadership` vs. `non_leadership`, `stem` vs. `non_stem`) onto it, then
   reports the standardized difference (Cohen's d) between the two sets.

2. **Garg RND — one target wordlist per category.**
   `analyze_garg_weat.py` uses **one** occupation list per category
   (leadership / family / science) and measures its average relative norm
   distance `||v − c_male|| − ||v − c_female||`.

A reviewer-facing robustness question follows: how much of the result depends
on the *measurement* (RND vs. projection) versus the *design* (one wordlist vs.
two)? To isolate the measurement, we want a **Cohen's-d-style projection metric
applied to Garg's single-wordlist design** — same inputs as RND, different
metric. This is the "single-list Cohen's d."

This spec adds that metric as a config-selectable robustness companion to RND,
plus a second summary statistic (proportion of male-leaned words) for **both**
metrics, and extends the existing visualization to render the new outputs.

## Scope

In scope:
- A new per-word metric: cosine projection of each occupation onto the
  female−male gender axis, averaged over Garg's single wordlist per category.
- A second summary statistic for both RND and projection: the **proportion of
  male-leaned occupations** (fraction with metric < 0).
- Config-driven dispatch: one config declares `analysis.metrics: [rnd,
  cohens_d]` (any subset); a single orchestrator computes the requested metrics
  in one pass, loading each model only once.
- Both statistics carry **both** uncertainty bands already used by Garg-WEAT:
  with-replacement bootstrap CI and word-subsample robustness band.
- Run across all 9 existing model configs (no new data, no new wordlists).
- Visualization of mean-projection trends and proportion-male-leaned trends,
  reusing the existing plotting code.

Out of scope:
- The two-wordlist WEAT Cohen's d (`analyze_weat.py`) is untouched.
- No new corpora, embeddings, or wordlist files.
- No change to the RND formula or its existing outputs' columns (additive only).

## Key decisions (locked during brainstorming)

| Decision | Choice |
|---|---|
| Projection scalar | **Cosine to axis**: L2-normalize each occupation vector (as Garg does), project onto the unit-norm female−male axis. Equals cosine similarity to the axis, bounded [−1, 1]. No cross-unit z-scoring — keeps a clean structural parallel to RND. |
| Sign convention | Positive = female-leaning, negative = male-leaning (matches Garg's RND sign). |
| Proportion definition | Male-leaned = metric **< 0** (natural neutral point). Reported for **both** RND and projection. |
| Bands | Each statistic (mean, proportion) gets both a bootstrap CI and a subsample band — same machinery and config as garg_weat. |
| Config | **Reuse** the existing 9 `garg_weat_*.yml` profiles. Add `analysis.metrics: [rnd, cohens_d]`. No new profiles, no new wordlist dirs. |
| Dispatch | **One orchestrator, single pass.** Reads `analysis.metrics`, loads each model once, computes the listed metric(s), writes per-metric outputs. |
| New script name | `analyze_cohens_d_singlelist.py` (the "singlelist" suffix distinguishes it from the two-wordlist WEAT Cohen's d). |
| Visualization | In scope; reuse existing parameterized plot code. |

## The 9 models

- **American — COHA (3):** trained, HistWords SGNS, HistWords SVD.
- **American — HistWords Google Ngram (2):** eng-all, eng-fiction.
- **Chinese longitudinal (2):** Renminribao, China Ngram.
- **Chinese provincial (2):** provincial newspaper, Weibo.

Each is an existing `config/profiles/garg_weat_*.yml` profile.

## Architecture

### Module layout

The metric-agnostic summary machinery currently lives inside
`analyze_garg_weat.py`. To let one orchestrator drive multiple metrics without
duplication, that machinery moves to a shared module; each metric becomes a
small per-word value producer.

```
scripts/
  analyze_category_bias.py          # NEW orchestrator + CLI entry point
  analyze_cohens_d_singlelist.py    # NEW: projection_values(model, ...) -> long_df
  analyze_garg_weat.py              # RND metric: rnd_values(model, ...) -> long_df
                                    #   (kept runnable for back-compat; delegates)
  common/
    category_summary.py             # NEW shared: load_categories,
                                    #   compute_consistent_set,
                                    #   subsample_bands_from_lookup (generalized),
                                    #   build_summary (generalized)
    metrics.py                      # generalized bootstrap + proportion helper
    embedding_utils.py              # REUSED: construct_semantic_axis, compute_projection
    embedding_loaders.py            # REUSED: histwords loaders
    config_loader.py                # REUSED: load_config, get_wordlist_dir
```

### Per-word metric contract

Both metrics expose a function with the same signature, operating on an
**already-loaded** model (so the orchestrator loads each model once):

```python
def <metric>_values(
    model, unit_name, categories, gender_words, logger
) -> pd.DataFrame:
    # returns long rows: unit_name, category, occupation, value, in_vocab
```

- `rnd_values` (in `analyze_garg_weat.py`): builds L2-normalized male/female
  centroids, computes `relative_norm_distance` per occupation.
- `projection_values` (in `analyze_cohens_d_singlelist.py`): builds the
  female−male axis via `construct_semantic_axis(female, male, model)`,
  L2-normalizes each occupation vector, computes `compute_projection(vec,
  axis)` (cosine to axis) per occupation.

Both return a **generic `value` column** (not `rnd`/`projection`), so the
shared summary code is metric-name-free. The orchestrator tags outputs with the
metric when naming files.

### Orchestrator flow (`analyze_category_bias.py`)

```
load_config(config)
metrics = config["analysis"]["metrics"]      # e.g. ["rnd", "cohens_d"]; required
gender_words, categories = load (shared, once)
models = discover_models(config)             # reused from analyze_garg
for each (model_path, unit_name):
    model = load_model_for_unit(...)          # loaded ONCE per model
    for metric in metrics:
        long_df[metric].append(<metric>_values(model, unit_name, ...))
for metric in metrics:
    long = concat(long_df[metric])
    consistent = compute_consistent_set(long, categories, units)
    summary = build_summary(long, units, consistent, value_col="value", ...)
    write long  -> results_dir / f"{file_stem[metric]}_long.parquet"
    write summary -> results_dir / f"{file_stem[metric]}_summary_by_category.parquet"
```

`metrics` is **required** in config; absence is a loud error (prefer breaks over
silent defaults). The orchestrator validates each entry is in `{rnd,
cohens_d}`.

File stems:
- `rnd` → `garg_weat_rnd_long`, `garg_weat_summary_by_category` (unchanged names).
- `cohens_d` → `cohens_d_singlelist_long`, `cohens_d_singlelist_summary_by_category`.

### Backward compatibility

- `analyze_garg_weat.py` keeps its CLI `main` and its public function names so
  existing tests and any direct invocations still work; its summary helpers are
  re-exported from / delegate to `common/category_summary.py`. Running it
  behaves as `metrics=[rnd]`.
- Existing `garg_weat_*.parquet` column names are preserved. New columns are
  **added**, not renamed.

## The two summary statistics and their bands

`build_summary` is generalized to take `value_col` (default `"rnd"` for
back-compat) and to compute, per (unit, category) over the category's
consistent set:

1. **mean** of the metric.
2. **proportion male-leaned** = mean(value < 0).

Each statistic carries:
- a **bootstrap CI** (with-replacement, `analysis.bootstrap.{n_iter, ci}`), and
- a **subsample band** (word-drop without replacement, hold subset across units,
  `analysis.subsample.{fraction, n_rounds, ci}`).

To support a non-mean statistic, `metrics.bootstrap_ci` is generalized to accept
a `statistic` callable (default `np.mean`); a `proportion_below(values,
threshold=0.0)` helper is added. `subsample_bands_from_lookup` likewise takes a
`statistic` so the subsample band can be computed for the proportion using the
identical resampling logic (same held-out word subsets).

### Summary schema (per unit × category)

Existing RND columns kept; new proportion columns added. Generic naming so both
metrics share the schema (the RND file keeps `mean_rnd` as an alias of the mean
for back-compat; see below).

| Column | Meaning |
|---|---|
| `unit_name`, `category` | keys |
| `mean_value` | mean of the metric (mean RND or mean projection) |
| `mean_ci_low`, `mean_ci_high` | bootstrap CI of the mean |
| `mean_sub_low`, `mean_sub_high`, `mean_sub_mean` | subsample band of the mean |
| `prop_male` | proportion of occupations with value < 0 |
| `prop_ci_low`, `prop_ci_high` | bootstrap CI of the proportion |
| `prop_sub_low`, `prop_sub_high`, `prop_sub_mean` | subsample band of the proportion |
| `n_occupations`, `n_consistent` | counts |

**RND back-compat:** the `garg_weat_summary_by_category.parquet` additionally
retains the original column names (`mean_rnd`, `ci_low`, `ci_high`, `sub_low`,
`sub_high`, `sub_mean`) as aliases of the corresponding `mean_*` columns, so
existing visualization and any downstream consumers keep working unchanged. New
`prop_*` columns are appended.

## Configuration changes

Add to each of the 9 `garg_weat_*.yml` profiles:

```yaml
analysis:
  metrics: [rnd, cohens_d]   # which measures to compute this run
  # existing seed / ideation_sign / bootstrap / subsample unchanged
```

No `config_loader` schema change is required to *run* (the orchestrator reads
`analysis.metrics` directly), but `config_loader` validation should accept and
sanity-check the `metrics` list (entries ∈ `{rnd, cohens_d}`) to fail fast.

## Visualization

Extend `scripts/visualize.py`, reusing its already-parameterized plotters:

- **Category trends:** `plot_garg_weat_categories_trend` already accepts
  `line_col` and `band_cols`. Drive it for the projection summary with
  `line_col="mean_value"` and the projection bands, producing
  `fig2_cohens_d_singlelist_categories*.pdf` alongside the existing RND figure.
- **Proportion trends:** render `prop_male` over time (with its bootstrap and
  subsample bands) for both metrics — a new thin call into the same trend
  plotter with `line_col="prop_male"` and `prop_*` band columns, y-axis labeled
  as "share of male-leaning occupations".
- **Provincial views** (`plot_garg_weat_provincial_rankings/heatmap/choropleth`)
  currently hardcode `mean_rnd`. Parameterize their value column so they can
  render the projection summary and the proportion for the provincial
  (newspaper, Weibo) configs.
- The visualization dispatcher detects which summary parquets exist in
  `results_dir` and plots whichever are present (RND only, projection only, or
  both), so it works regardless of the `metrics` list used.

Exact dispatch wiring in `visualize.py main` to be finalized in the plan after
reading its entry point; the changes are additive and parameterizing, not a
rewrite.

## Testing

- Unit: `proportion_below` (sign threshold, empty guard); generalized
  `bootstrap_ci`/`subsample` with `statistic=np.mean` reproduce current numbers
  (regression guard); `projection_values` sign correctness on a tiny synthetic
  model (a clearly female-leaning and a clearly male-leaning token).
- Integration: orchestrator with `metrics=[rnd]` reproduces the current
  `garg_weat_*` outputs byte-for-byte on a fixture model set (back-compat);
  `metrics=[rnd, cohens_d]` produces both output pairs from a single model load.
- Existing `analyze_garg_weat` tests must continue to pass after the
  helper extraction (mechanical move + parameterization, defaults preserved).

## Risks / notes

- Moving summary helpers into `common/category_summary.py` touches working
  garg_weat code; mitigated by preserving public names, default `value_col`,
  alias columns, and the regression/integration tests above.
- Cosine-to-axis without z-scoring assumes the gender axis is comparably scaled
  across units; this is the same assumption Garg's un-standardized RND makes, so
  the two metrics stay on equal footing for the robustness comparison. If
  cross-unit axis drift proves large, z-scoring can be added later as a separate,
  explicitly-flagged variant (not in this spec).
