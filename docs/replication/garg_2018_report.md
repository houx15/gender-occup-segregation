# Replication report — Garg et al. (2018), Fig 2

A short summary of what we attempted, what worked, and the bugs we fixed along the way. The methodology + run instructions live in [`garg_2018.md`](garg_2018.md); this file is the results read-out.

## What we replicated

**Garg, N., Schiebinger, L., Jurafsky, D., & Zou, J. (2018).** *Word embeddings quantify 100 years of gender and ethnic stereotypes.* PNAS, 115(16), E3635–E3644.

Specifically Fig 2: the **average gender bias of occupations** (relative norm distance from male vs. female centroids) computed per decade across COHA, with the IPUMS census female-share-of-occupation as a twin-axis overlay. Garg's plot covers 1910s–1990s and shows occupational embeddings drifting from male-leaning toward gender-neutral as census female participation rises.

## Embedding sources we ran

We computed the same metric on three different sets of decade-aligned embeddings, all backed by COHA:

| Source key | Corpus | Algorithm | Trained by | Vocab (1910s) |
|---|---|---|---|---|
| `trained_coha` | COHA full-text (paid release) | SGNS, gensim word2vec | Us, on Princeton Slurm | per-decade |
| `histwords_sgns` | COHA (Davies BYU release ~2014–2016) | SGNS | Hamilton et al. 2016 | 50,000 |
| `histwords_svd` | Same COHA | Truncated SVD over PPMI | Hamilton et al. 2016 | ~12,000 |

All three use **identical hyperparameters** where comparable (vector_size=300, window=4, min_count=100, sg=1, negative=15, epochs=5). The HistWords vectors are the exact pretrained ones Garg consumed.

## Findings

### `trained_coha` — replicated the qualitative trend

Our from-scratch trained vectors produce the expected downward trend in average women-bias over 1910s–1990s. Figure: `figures_garg/fig2_garg_replication__trained_coha.pdf`.

This validates the full pipeline: COHA download → fulltext-chunk extraction → SGNS training → centroid + RND metric → consistent-occupation aggregation → plot.

### `histwords_sgns` — matches Garg's published vectors

Same qualitative trend, slightly different absolute values per decade (different gensim version + no Procrustes alignment in our pipeline; HistWords applies orthogonal Procrustes across decades). This is the closest direct comparison to Garg's Fig 2. Figure: `figures_histwords_sgns/fig2_garg_replication__histwords_sgns.pdf`.

Per-decade coverage is **100% (76/76 occupations in vocab, 20/20 gender words found in every decade)** — the published HistWords SGNS vocab is large and consistent enough that no filtering tradeoffs come into play.

### `histwords_svd` — much sparser vocab, higher variance

The SVD/PPMI vectors are released with aggressive frequency truncation (1810s vocab is only ~1,200 words; even 1990s is ~14,500). Coverage in our 76-occupation list ranges from 7% (1810s, mostly unusable) to 83% (2000s). Pre-1910 decades especially are too sparse to interpret. Figure: `figures_histwords_svd/fig2_garg_replication__histwords_svd.pdf`.

Garg himself reports SVD only as a robustness check in the SI Appendix, not as primary evidence — this matches our experience.

## Method notes that turned out to matter

Three things we got wrong on the first pass, surfaced by careful comparison with Garg's reference code (`reference/GARG/`):

1. **L2-normalize every fetched vector before centroid + distance.** Garg's `dataset_utilities/normalize_vectors.py` does this; without it, decade-to-decade vector-norm drift inflates RND magnitudes and breaks comparability to published numbers. Implemented in `scripts/common/metrics.py:l2_normalize`.

2. **Sign convention.** Garg's RND is `||v − c_male|| − ||v − c_female||` (positive = female-leaning). We had this flipped originally, which inverted the entire plot. Fixed in `scripts/common/metrics.py:relative_norm_distance`.

3. **Consistent-occupation filter.** Garg's Fig 2 restricts every decade's mean to the same set of occupations — specifically those that are in vocab in *every* decade AND have valid census data in every decade. Without this filter, per-decade vocab churn produces spurious cliffs in the trend. Implemented in `scripts/analyze_garg.py:compute_consistent_set`.

## Bugs caught during the multi-source rollout

The HistWords runs surfaced two non-obvious failure modes that we now guard against:

- **HistWords vocab files are Python-2 pickles.** Loading them with `pickle.load(f)` under Py3 returned `bytes` objects, so KeyedVectors keys became `b'doctor'` and every occupation OOV'd. Fix: pass `encoding="latin1"` in `scripts/common/embedding_loaders.py:_load_vocab_pkl` and add a regression test.

- **Census coverage doesn't span the HistWords range.** Garg's vendored census CSV starts in 1850, but HistWords ships 1810s–2000s. The consistent-set filter requires census data in every analyzed decade, so any pre-1850 decade zeroes the intersection → all-NaN summary → blank PDF. Fix: added `analysis.decade_range: [start, end]` config option, set to `[1910, 1990]` for both HistWords profiles to match Garg's Fig 2 scope. The trained_coha pipeline already only produces 1910s–1990s on disk, so it's unaffected.

The analyzer now emits per-unit diagnostics (vocab sample, gender-word OOV list, occupation OOV preview, per-unit contribution to the consistent set) so these classes of bug fail loudly rather than silently producing empty figures.

## How to reproduce

```bash
# On Princeton, after pulling main and downloading the HistWords archives
# via scripts/data_prep/download_pretrained_embeddings.py:
sbatch slurm/fig2_all_sources.slurm
```

This runs all three configs (`coha_garg.yml`, `coha_histwords_sgns.yml`, `coha_histwords_svd.yml`), continues past per-source failures, and prints a SUMMARY block at the end listing the resolved PDF for each.

Outputs (under `/scratch/network/yh6580/gender-occup/`):

```
figures_garg/fig2_garg_replication__trained_coha.pdf
figures_histwords_sgns/fig2_garg_replication__histwords_sgns.pdf
figures_histwords_svd/fig2_garg_replication__histwords_svd.pdf
```

Plus per-source parquet files at `data/results_*/garg_average_bias_by_decade.parquet` and `data/results_*/garg_relative_norm_by_decade.parquet`.

## What's next

The Fig 2 replication confirms the pipeline is sound. Natural follow-ups that re-use the same machinery:

- **Other figures from Garg (2018)**: Fig 1 (census scatter) and Fig 6 (NYT fine-grained) need different embedding sources (Google News word2vec, NYT vectors) — both already wired into the pretrained-embedding downloader.
- **State-level extension** of the same RND metric on COHA-style state corpora (the original goal of this codebase) — currently deferred per `docs/superpowers/bilingual_refactor/`.
- **Cross-embedding comparison plot** layering the three Fig 2 trend lines onto one axis, to visualize where our trained vectors and HistWords diverge.
