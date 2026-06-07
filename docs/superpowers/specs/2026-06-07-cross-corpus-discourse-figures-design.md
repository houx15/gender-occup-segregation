# Cross-corpus discourse figures (制度话语 × 公众话语)

**Date:** 2026-06-07
**Status:** Design approved, pending spec review
**Scope:** Visualization only. No re-computation of metrics.

## Motivation

We have Garg-style Relative Norm Distance (RND) results, with bootstrap and
subsample uncertainty bands, computed per (corpus, category, time-window) for
two Chinese corpora:

- **Renmin Ribao / 人民日报 (RMRB)** — *制度话语* (institutional discourse)
- **China Ngram / ngram_zh** — *公众话语* (public discourse)

The existing `garg_weat` figure (`plot_garg_weat_categories_trend`) draws one
corpus at a time, with the three categories (leadership / family / science) as
colored lines. This spec adds the **orthogonal cut**: one category-topic per
figure, the two corpora overlaid, so institutional vs. public discourse can be
read against each other over time.

## Inputs (already computed, on `/scratch`)

Both files share columns `unit_name, category, mean_rnd, ci_low, ci_high`
(bootstrap 68%), `sub_low, sub_high` (80% word-subsample), `prop_male`, and the
same rolling windows `1940_1949 … 2015_2024`.

| Role | Profile | results_dir leaf | Title label |
|------|---------|------------------|-------------|
| 制度话语 (institutional) | `config/profiles/garg_weat_renminribao.yml` | `results_garg_weat_renminribao` | People's Daily |
| 公众话语 (public) | `config/profiles/garg_weat_china_ngram.yml` | `results_garg_weat_china_ngram` | Google Ngram |

`data_source` → title label comes from the existing `DATA_SOURCE_LABELS` map
(`renminribao` → "People's Daily", `ngram` → "Google Ngram"). Both profiles
already carry `analysis.ideation_sign = {leadership: 1, science: 1, family: -1}`.

## Figures (8 PDFs)

Two topics × {absolute, Δ-from-2000} × {bootstrap band, subsample band}.

| Topic | Categories | Lines | Y-axis |
|-------|-----------|-------|--------|
| Family | family | family × {RMRB, ngram} = 2 | oriented RND (family reversed → "higher = less traditional") |
| Work & Science | leadership, science | {leadership, science} × {RMRB, ngram} = 4 | oriented RND (both +1) |

- **Absolute** version: raw (ideation-oriented) `mean_rnd`.
- **Δ-from-2000** version: `mean_rnd_t − mean_rnd` at the baseline window
  `1995_2004` (centered on 2000). Every line passes through 0 at 2000.

Filenames (undated, matching the sibling `plot_garg_weat_categories_trend`):

```
fig_crosscorpus_family__bootstrap.pdf
fig_crosscorpus_family__subsample.pdf
fig_crosscorpus_family_rel2000__bootstrap.pdf
fig_crosscorpus_family_rel2000__subsample.pdf
fig_crosscorpus_work_science__bootstrap.pdf
fig_crosscorpus_work_science__subsample.pdf
fig_crosscorpus_work_science_rel2000__bootstrap.pdf
fig_crosscorpus_work_science_rel2000__subsample.pdf
```

## Encoding (consistent across all 8)

- **Color = category** — reuse existing palette: family `#c0392b`,
  leadership `#1f4e79`, science `#2e7d32`.
- **Line style = corpus** — RMRB = solid + filled marker; ngram = dashed + open
  marker.
- **Two legends** — a color legend (category) and a linestyle legend
  (制度 / 公众 source). Family figures have one color but still two linestyles.
- `axhline(0)` reference line, grid `alpha=0.3`, `figsize=(11, 6)`, seaborn
  whitegrid — matching `plot_garg_weat_categories_trend`.

## Methodological decisions (baked in)

1. **Comparability caveat (absolute figs).** Cross-corpus RND *levels* are not
   directly comparable (a function of each corpus's embedding geometry — vocab,
   frequency, training). The absolute figures carry a footnote saying so: read
   direction and temporal shape, not the size of the gap between corpora.

2. **Δ is a difference, not a ratio.** RND straddles and crosses zero, so a
   ratio/percent index is unstable. We subtract the `1995_2004` value. Bands are
   shifted by the same baseline constant (baseline treated as a fixed reference
   point), so band width is preserved across the series including at 2000.

3. **No silent baseline default.** If a (corpus, category) is missing the
   `1995_2004` window, log loudly and skip that line — never substitute a
   neighboring window silently. (User preference: prefer breaks over silent
   defaults.)

4. **Ideation orientation applied** via `apply_ideation_sign` (family × −1), so
   the new figures match the existing fig2 convention. Family appears alone in
   its figure, so the reversal only flips that axis's interpretation.

## Code shape

### 1. `plot_cross_corpus_category_trend(...)` — new function in `visualize.py`

Signature (mirrors the conventions of `plot_garg_weat_categories_trend`):

```python
def plot_cross_corpus_category_trend(
    df, figures_dir, logger, *,
    categories,               # e.g. ["family"] or ["leadership", "science"]
    source_labels,            # {source_key: "People's Daily", ...}
    source_styles=None,       # {source_key: {"linestyle": "-", "fillstyle": "full"}}
    band_cols=("ci_low", "ci_high"),
    band_tag="bootstrap",
    band_label=None,
    line_col="mean_rnd",
    category_sign=None,
    normalize_to=None,        # baseline unit_name, e.g. "1995_2004"; None = absolute
    fig_stem="fig_crosscorpus",
    ylabel=None,
):
```

- Expects a combined long DataFrame with an added `source` column.
- Filters to `categories`; parses `start_year` via the existing `_parse_decade`
  helper (refactor it module-level if still nested, otherwise reuse).
- Applies `apply_ideation_sign(df, category_sign, [line_col, low, high])`.
- If `normalize_to` is set: for each (source, category), subtract that group's
  `line_col` at `normalize_to` from `line_col`, `low_col`, `high_col`. Missing
  baseline → `logger.error` + skip that (source, category).
- Draws one line per (category, source): color from palette by category,
  linestyle/marker by source; shaded band via `fill_between`.
- Builds two legends (category colors, source linestyles).
- Writes `{fig_stem}{_rel2000?}{__band_tag}.pdf` (undated, like the sibling
  trend plot). Empty / all-NaN guards mirror the existing function (refuse blank
  PDFs, log loudly).

### 2. `cross_corpus(...)` — new Fire entry point in `visualize.py`

```python
def cross_corpus(
    institutional_config="config/profiles/garg_weat_renminribao.yml",
    public_config="config/profiles/garg_weat_china_ngram.yml",
    figures_dir=None,
    baseline_unit="1995_2004",
):
```

- Loads both profiles via `load_config`; calls `_configure_fonts` (zh).
- Reads each `results_dir/garg_weat_summary_by_category.parquet`; tags each with
  `source` = its `data_source`; concatenates.
- Resolves `source_labels` from `DATA_SOURCE_LABELS`, `category_sign` from the
  institutional profile's `analysis.ideation_sign` (both profiles agree).
- `figures_dir` default → `…/gender-occup/figures_garg_weat_cross_corpus_zh`
  (sibling of the other figure dirs); overridable.
- Emits the 8 figures by looping over:
  - topics: `("family", ["family"], "fig_crosscorpus_family")`,
    `("work_science", ["leadership", "science"], "fig_crosscorpus_work_science")`
  - normalization: `(None, "")` and `(baseline_unit, "_rel2000")`
  - bands: `(("ci_low","ci_high"), "bootstrap")` and
    `(("sub_low","sub_high"), "subsample")`
- Missing-parquet → loud error, non-zero-ish behavior (raise), no silent skip.

Config still drives dispatch — we pass two existing profiles; nothing about the
metric computation changes.

### 3. `slurm/garg_weat_cross_corpus_zh.slurm` — new slurm script

Mirrors `garg_weat_zh.slurm` structure:

- Same `#SBATCH` header style (1 node, ~4 cpus, ~16G, short walltime — this is
  pure plotting), `logs/garg_weat_cross_corpus_zh_%j.{out,err}`.
- `module load anaconda3/...`; `conda activate llm`.
- Reuse the `read_config_value` YAML helper to resolve `figures_dir` for
  validation.
- Single stage (no analysis):
  ```bash
  python -m scripts.visualize cross_corpus \
      --institutional_config=config/profiles/garg_weat_renminribao.yml \
      --public_config=config/profiles/garg_weat_china_ngram.yml
  ```
- Validation: assert exactly 8 `*crosscorpus*.pdf` files in `figures_dir`; exit
  non-zero with a clear message otherwise.

## Out of scope

- No metric re-computation, no new analysis profiles.
- No `prop_male` or Cohen's d cross-corpus variant (could follow the same
  pattern later if wanted).
- No English / COHA corpora.
- No changes to the existing single-corpus `garg_weat` figures.

## Testing / verification

- Local smoke test with tiny synthetic parquets (two sources, the three
  categories, a handful of windows incl. `1995_2004`) to confirm: 8 PDFs
  written, Δ lines pass through 0 at baseline, missing-baseline path logs+skips,
  empty/all-NaN guards fire.
- On cluster: run the slurm script; confirm 8 PDFs land in
  `figures_garg_weat_cross_corpus_zh` and CJK glyphs render (font registered).
