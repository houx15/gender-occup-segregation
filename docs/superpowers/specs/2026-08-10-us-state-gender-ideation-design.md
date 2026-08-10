# US state-level gender-ideation pipeline — design

**Date:** 2026-08-10
**Status:** approved design, pre-plan
**Author:** houyuxin (with Claude)

## Goal

Build embedding-based gender-ideology indicators at the US **state × year** level,
measured via the project's existing Garg relative-norm-distance (RND) and single-list
Cohen's d machinery, and render the result as per-year US choropleths. Two textual
sources, kept as **independent arms** so historical newspaper OCR and modern web news
are never mixed in one embedding space:

| Arm | Source | Years | Unit | State assignment |
|---|---|---|---:|---|
| A | American Stories (`dell-research-harvard/AmericanStories`, HuggingFace) | 1900, 1910, 1920, 1930, 1940, 1950, 1960 | `{state}_{year}` | LCCN → LoC US Newspaper Directory lookup |
| B | 3DLNews2 (`wm-newslab/3DLNews2`, Globus bulk) | 1996, 2000, 2010, 2020 | `{state}_{year}` | `location.state` (already in the data) |

Only textual data partitionable by state is in scope. Each arm produces per-`state_year`
corpora → per-`state_year` embedding model → RND / Cohen's d per unit → longitudinal
trend figures + per-year state choropleth.

Unit naming (`california_1940`) is chosen so the existing `_decade_start_year` parser and
trend plots in `scripts/visualize.py` read it with no change.

## Non-goals

- No merging of the two arms into one timeline (deliberately rejected — OCR vs. web
  register and OCR-quality differences would confound a shared embedding space).
- No 1964–1994 state-archive backfill (Georgia/Oregon Open ONI etc.) — out of scope for
  this iteration; may follow as a later arm.
- No new training or bias-metric code. Training (`train_embeddings.py`) and analysis
  (`analyze_category_bias.py`) are reused unchanged, driven by config.

## Architecture

The design mirrors the existing `provincial_newspaper` arm end to end:
per-unit corpora → per-unit `.kv` model → `analyze_category_bias` (RND + Cohen's d) →
`garg_weat_summary_by_category.parquet` → visualize (trend + choropleth).

### New modules

1. **`scripts/data_prep/us_state_mapper.py`**
   - Builds an LCCN → publisher-state table once from the LoC *US Newspaper Directory*
     JSON API, caching to `raw_data_dir`. `resolve_state(lccn) -> state | None`.
   - State-name normalizer: full name ↔ USPS 2-letter ↔ shapefile `NAME` column.
     Single source of truth for state identity across both arms and the choropleth.
   - Arm B uses only the normalizer (its `location.state` is authoritative); Arm A uses
     `resolve_state`.

2. **`scripts/data_prep/download_american_stories.py`**
   - `load_dataset("dell-research-harvard/AmericanStories", "subset_years", year_list=[...])`
     for the configured years; writes raw article JSONL (`article_id`, `newspaper_name`,
     `date`, `headline`, `byline`, `article`, LCCN) to `raw_data_dir`. Runs where the node
     has internet (login node or internet-enabled node). Idempotent: skips years already
     materialized.

3. **`scripts/data_prep/download_dlnews.py`**
   - Pulls `Google/1-Newspapers/preprocessed_state/{STATE}/preprocessed_google_newspaper_{STATE}_{YEAR}.jsonl.gz`
     via the **Globus CLI** for the configured years, from the 3DLNews2 source collection
     UUID to the Princeton destination endpoint UUID (both in config), then
     `globus task wait`.
   - **Auth model:** a one-time interactive `globus login` on the login node (browser
     OAuth, like `gcloud auth login`); after that, cached tokens let the script issue
     transfers headless. Globus transfer/use is free for individual researchers; Princeton
     runs a managed endpoint.
   - **Fallback (documented, not default):** if OAuth cannot be completed from the node,
     the user runs the printed `globus transfer` command manually and the builder reads the
     resulting local files. The builder never assumes network.

4. **`scripts/data_prep/build_corpora_us.py`**
   - Reads raw JSONL, assigns state (mapper for A, inline for B), filters to valid news
     content (`is_news_article` + non-empty body for B; non-empty `article` for A).
   - Cleans via existing
     `preprocess(language="en", tokenizer="nltk_en", stopwords_key="en_default", lowercase=True, min_words=...)`.
   - Applies **configurable wire-copy dedup** (see below).
   - Writes per-state rolling corpus files under `corpora_dir/{state}/` using a generalized
     `UnitCorpusWriter` (lifted from `ProvinceCorpusWriter` in
     `build_corpora_newspaper.py`), keyed by `{state}_{year}` unit.
   - Emits a **state × year coverage report** (document counts). Units below
     `us_states.min_documents` are dropped from the trainable set, with the drop logged —
     never silently trained on thin data.

5. **`scripts/visualize.py`** (additions only)
   - `_match_state_in_shapefile(dim_data, states_gdf)` and `plot_us_choropleth(...)`,
     direct analogs of `_match_province_in_shapefile` / `plot_weat_choropleth`, drawing on
     a US-states shapefile. Existing trend plots need no change.

### Wire-copy dedup

Syndicated/wire copy is the same story printed across many states; left in, it distorts
per-state signal (flagged in `docs/2026-08-10-us-data.md`). Config:

```yaml
corpus:
  dedup:
    enabled: true          # default on
    method: shingle        # exact | shingle
    shingle_k: 8           # k-token shingles for near-dup signature
    scope: within_year     # de-duplicate across states within the same year
```

- `exact`: drop repeats sharing a normalized title+body hash.
- `shingle`: MinHash/k-shingle signature catches near-identical wire copy with minor edits.
- Scope is **within-year across states** — the AP story reprinted in 20 states collapses
  to one, but genuine year-over-year repetition is preserved.

### Reused unchanged

- `scripts/train_embeddings.py` — per-unit training, `model_name_template: "model_{unit_name}.kv"`.
- `scripts/analyze_category_bias.py` — RND + single-list Cohen's d → `garg_weat_summary_by_category.parquet`.
- `scripts/common/preprocessing.py` — `en` cleaner, `nltk_en` tokenizer, `en_default` stopwords.
- Wordlists `wordlists/en/garg_weat/` — `gender_words.json` + `cleaned_{leadership,family,science}.txt`.
- Trend plots in `scripts/visualize.py` (parse `_YYYY` units already).

## Configs

Two profiles, both `language: en`, `analysis_mode: garg_weat`, `metrics: [rnd, cohens_d]`:

- `config/profiles/garg_weat_american_stories.yml`
- `config/profiles/garg_weat_dlnews.yml`

Each adds a `us_states` block:

```yaml
us_states:
  shapefile: "data/shapefiles/us_states.shp"
  min_documents: 500        # per state_year unit; below → dropped
  years: [1900, 1910, 1920, 1930, 1940, 1950, 1960]   # arm-specific
  # states: optional explicit allow-list; default = all resolved states
```

`ideation_sign`: leadership +1, science +1, family −1 (matching the existing English garg
configs). `embedding` block copies the English defaults (vector_size 300, window 5,
min_count 20, sg 1, negative 10, epochs 10, seed 42).

Base dir follows the existing English arm convention (`/scratch/network/yh6580/gender-occup`).

## Slurm (full end-to-end)

- `slurm/prepare_us_data.slurm` — download (Arm A HF / Arm B Globus) + `build_corpora_us` +
  coverage report. Structured so the network-touching download can run on a login/internet
  node and the CPU build on a compute node.
- `slurm/train_us.slurm` — per-unit training over the trainable `state_year` set.
- `slurm/garg_weat_us.slurm` — `analyze_category_bias` + `visualize` (trend + choropleth),
  mirroring `slurm/garg_weat_zh.slurm` (per-config loop, status summary, non-fatal
  word-drivers step optional).

## Shapefile

US-states boundaries: Census cartographic-boundary `cb_20m` (public zip), fetched into
`data/shapefiles/us_states.shp` by the prepare step (or documented for manual placement).
`_match_state_in_shapefile` joins on the normalized state `NAME`.

## Data flow

```
[Arm A] HF subset_years ──> raw article JSONL ─┐
                                               ├─> build_corpora_us ─> corpora/{state}/corpus_* ─> train ─> models/model_{state}_{year}.kv ─> analyze_category_bias ─> summary.parquet ─> visualize (trend + choropleth)
[Arm B] Globus transfer ──> raw {STATE}_{YEAR}.jsonl.gz ─┘
        (state via LCCN→LoC for A; location.state for B; dedup within-year)
```

## Coverage & honesty constraints

- American Stories state coverage is uneven by decade; early decades (1900–1920) will cover
  far fewer states than 1940–1960. The coverage report makes this explicit; sparse units are
  dropped, not silently trained.
- Raw modern news text (Arm B) is not redistributed; only derived artifacts
  (`state_year`, RND/Cohen's d, coverage metrics, embeddings/aggregates) leave controlled
  storage — consistent with the licensing notes in `docs/2026-08-10-us-data.md`.
- Every drop/skip (below-threshold unit, unresolved LCCN, empty year) is logged; no silent
  truncation.

## Testing

- `us_state_mapper`: unit tests for name normalization (full ↔ USPS ↔ shapefile) and
  `resolve_state` on known + unknown LCCNs (fixture table, no network).
- `build_corpora_us`: fixture JSONL for both arms → assert per-state routing, en
  preprocessing, dedup collapses a duplicated wire story within-year, below-threshold unit
  dropped, coverage counts correct.
- `dedup`: exact and shingle methods on crafted near-duplicates.
- Choropleth: smoke test that `_match_state_in_shapefile` joins a small synthetic frame to a
  states shapefile without dropping matched states (skips gracefully if geopandas/shapefile
  absent, like the provincial choropleth).

## Open items deferred to the plan

- Exact LoC US Newspaper Directory endpoint/params and LCCN field location within the
  American Stories `article_id` / scan metadata (resolve during implementation against the
  live API).
- 3DLNews2 source collection UUID + Princeton destination endpoint UUID (user-provided
  config values).
- Final dedup default parameters (`shingle_k`, exact vs. shingle) after inspecting real
  duplication rates on a sample slice.
