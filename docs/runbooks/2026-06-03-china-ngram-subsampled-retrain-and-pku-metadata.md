# Runbook — China Ngram Subsampled Retrain (Princeton) + Weibo / Provincial Newspaper Metadata Export (PKU)

**Date:** 2026-06-03
**Supersedes:** [`2026-06-02-china-ngram-weighted-retrain-and-pku-metadata.md`](2026-06-02-china-ngram-weighted-retrain-and-pku-metadata.md) (methods-wrong `capped_repetition` weighting).
**Audience:** anyone (incl. future you) coming back to run or repeat the HistWords-style per-year-capped China-Ngram retrain on Princeton scratch, and / or to regenerate the methods-section dataset summaries for Weibo + provincial newspaper on the PKU `/lustre` server.

Two independent workflows. They can run concurrently (different servers / different SLURM queues / different output dirs — no conflict).

---

## Princeton — HistWords-style per-year-capped China Ngram retrain

**What you get when this finishes:** a second set of china-ngram models trained on per-year-capped subsampled 5-gram corpora (per Hamilton et al. 2016, Appendix A — count-proportional emission with `min(1, 1e8 / year_total[year])` scaling, unbiased floor+Bernoulli sampling via a seeded RNG), living next to the existing presence-only models. Plus analyze/visualize outputs and a methods summary.

**Prereqs:**
- Princeton checkout at `/home/yh6580/gender-occup-segregation` (or wherever) pulled to a SHA at or after `1f28aa9`.
- `module load anaconda3/2023.3 && conda activate llm` works from a login node.
- `/scratch/network/yh6580/gender-occup/raw_ngrams/` has the `5-*-of-00105.gz` shards AND the `totalcounts-5` file (the file lives next to the shards; it ships with the Google Books v3 corpus).
- ≥ ~60 GB free scratch space (full sweep peaks at ~50 GB total; per-slice driver keeps peak at ~3 GB per slice).
- Princeton SLURM 48h wall-clock limit on jobs — the driver is now configured for 48h and the full 17-slice run must be split into 2–3 sbatch invocations (see Step 2).

### Step 0 — One-time cleanup of stale 06-02 artifacts (skip if already done)

```bash
# Cancel any still-queued cnw_per_slice jobs from the 06-02 attempt:
squeue -u $USER --name=cnw_per_slice -o "%i" --noheader | xargs -r scancel

# Reclaim the 114 GB partial weighted corpora:
rm -rf /scratch/network/yh6580/gender-occup/corpora_weighted
rm -rf /scratch/network/yh6580/gender-occup/raw_ngrams_decompressed_weighted
rm -rf /scratch/network/yh6580/gender-occup/models_weighted
rm -rf /scratch/network/yh6580/gender-occup/logs_weighted
rm -rf /scratch/network/yh6580/gender-occup/results_garg_weat_china_ngram_weighted
rm -rf /scratch/network/yh6580/gender-occup/figures_garg_weat_china_ngram_weighted
```

The 06-02 `garg_weat_china_ngram_weighted.yml` profile is already removed from the repo (commit `ebd893a`), so there's no live config consumer of those paths.

### Step 1 — Pull the corrected code

```bash
cd /home/yh6580/gender-occup-segregation
git pull
git log --oneline | head -10   # confirm 1f28aa9 or later is in main
```

Key commits this runbook depends on:
- `b59f519` totalcounts-5 parser module
- `39168cd` + `f5af2a3` per_year_capped weight_mode
- `ebd893a` new subsampled profile
- `7b6ddda` SLURM driver renamed + zh.slurm defaults flipped
- `1f28aa9` 48h time cap on driver

### Step 2 — Build + train all 17 slices, split into 2–3 jobs (48h-safe)

The per-slice driver does **build → train → delete-corpus** per slice, keeping peak disk to one slice's footprint (~3 GB at cap=10⁸). The whole 17-slice sweep is too long for a single 48h job, so split it. Three roughly-equal subsets:

```bash
# Subset 1 — 1940s through mid-1960s (6 slices, smaller corpora):
sbatch slurm/build_train_china_ngram_subsampled_per_slice.slurm \
  "1940_1949 1945_1954 1950_1959 1955_1964 1960_1969 1965_1974"

# Subset 2 — late 1960s through mid-2000s (6 slices, mid-size corpora):
sbatch slurm/build_train_china_ngram_subsampled_per_slice.slurm \
  "1970_1979 1975_1984 1980_1989 1985_1994 1990_1999 1995_2004"

# Subset 3 — 2000s onward (5 slices, densest corpora):
sbatch slurm/build_train_china_ngram_subsampled_per_slice.slurm \
  "2000_2009 2005_2014 2010_2019 2015_2020 2020_2020"
```

Each job:
1. Builds `corpora_subsampled/<slice>/corpus_NNNNN.txt` for one slice (decompresses needed shards, applies per-year scaling, deletes decompressed shards after).
2. Trains `models_subsampled/chi_sim_5gram_<slice>.model` from that slice's corpus.
3. `rm -rf corpora_subsampled/<slice>/` to free disk before the next slice.

Jobs queue independently; SLURM will run them concurrently if compute allows. Peak across all three running in parallel is still bounded by `3 × per-slice` ≈ 9 GB (well within scratch). If you want strict serial execution to be conservative:

```bash
# Capture the first job's id, then chain:
JID1=$(sbatch --parsable slurm/build_train_china_ngram_subsampled_per_slice.slurm \
  "1940_1949 1945_1954 1950_1959 1955_1964 1960_1969 1965_1974")
JID2=$(sbatch --parsable --dependency=afterok:$JID1 \
  slurm/build_train_china_ngram_subsampled_per_slice.slurm \
  "1970_1979 1975_1984 1980_1989 1985_1994 1990_1999 1995_2004")
sbatch --dependency=afterok:$JID2 slurm/build_train_china_ngram_subsampled_per_slice.slurm \
  "2000_2009 2005_2014 2010_2019 2015_2020 2020_2020"
```

**Monitoring.** Logs land at `logs/cns_per_slice_<jobid>.out` and `.err`. Each iteration prints `[N] SLICE=...`, the `du -sh` of the built corpus, and `[N] reclaimed:` after the delete. The final SUMMARY block lists each slice's status (`ok`, `build_failed`, `no_corpus_dir`, `train_failed`).

**Heads-up on memory.** Job header is `--mem=64G`. If gensim OOMs in 2000s-decade slices, bump the relevant subset's job to 128G.

### Step 3 — Analyze + visualize

After all three Step 2 jobs finish:

```bash
sbatch slurm/garg_weat_zh.slurm
```

The `_zh.slurm` default-runs **both** Renminribao and the new China-Ngram-subsampled profile. Both arms run through `analyze_category_bias` → both metrics (`rnd` + `cohens_d`) → `visualize`. Outputs:

- `/scratch/network/yh6580/renminribao/results/` + `figures/` — Renminribao
- `/scratch/network/yh6580/gender-occup/results_garg_weat_china_ngram_subsampled/` + `figures_garg_weat_china_ngram_subsampled/` — China-Ngram subsampled

### Step 4 — Methods-section summary

```bash
sbatch slurm/describe_dataset_zh.slurm
```

Same default-config pair (Renminribao + China-Ngram subsampled). Writes `dataset_summary.md` per source under each profile's `results_dir`. Content: corpus totals (docs, tokens, raw vocab, model vocab range), raw-data footprint, training hyperparameters, per-unit table with Year range + Tokens/doc columns.

### Step 5 — A/B vs the existing presence-only china-ngram models

The old `garg_weat_china_ngram.yml` profile and its trained models are untouched on disk — they're the "presence" arm for direct A/B comparison in methods. Regenerate analyze + summary outputs against them:

```bash
sbatch slurm/garg_weat_zh.slurm        config/profiles/garg_weat_china_ngram.yml
sbatch slurm/describe_dataset_zh.slurm config/profiles/garg_weat_china_ngram.yml
```

No build / train needed — these reuse the existing `corpora/` and `models/` directories. The two `dataset_summary.md` files (presence-only vs subsampled) make the side-by-side methods comparison straightforward.

---

## PKU — Weibo + provincial-newspaper metadata export

**What you get:** two `dataset_summary.md` files, one per source, ready to paste into the methods section.

**No retrain needed.** Weibo and provincial newspaper were trained on real running text from the start; the count-weighting question is a 5-gram artifact and doesn't apply here. Pure analyze-existing-corpora.

**Prereqs:**
- Repo checked out at `/lustre/home/2401111059/gender-occup-segregation`, pulled to a SHA at or after `425ee7e` (the `describe_dataset` enhancements from the 06-02 work — unaffected by the 06-03 weighting correction).
- `~/miniconda3` + `conda activate opinion` works from a login node.
- Existing corpora and models present at the paths in `garg_weat_weibo.yml` and `garg_weat_provincial_newspaper.yml`.

### Run both

```bash
cd /lustre/home/2401111059/gender-occup-segregation
git pull
sbatch slurm/describe_dataset_pku.slurm
```

Default configs: `garg_weat_weibo.yml` + `garg_weat_provincial_newspaper.yml`.

Outputs:
- `<weibo profile's results_dir>/dataset_summary.md` — Weibo
- `/lustre/.../data/provincial_newspaper/results_garg_weat/dataset_summary.md` — provincial newspaper

### Run one of the two explicitly

```bash
sbatch slurm/describe_dataset_pku.slurm config/profiles/garg_weat_provincial_newspaper.yml
sbatch slurm/describe_dataset_pku.slurm config/profiles/garg_weat_weibo.yml
```

### Performance

- **Cache-cold first run.** Weibo opens parquet metadata to bucket files by province; provincial newspaper rglobs every province-year directory for file counts. Minutes per source.
- **Cache-warm reruns.** `.dataset_stats.json` sidecars in each `corpora_dir/<unit>/` directory mean second-pass runs are seconds. `--force` recomputes if a corpus has been rebuilt under the same directory.

---

## Concurrent metadata export — run NOW while the build/train jobs queue

The describe jobs read existing models only — zero conflict with Step 2's build/train. Run them in parallel on either server:

```bash
# Princeton — export presence-only china-ngram + renminribao metadata RIGHT NOW
# (explicit configs avoid the default which now includes the not-yet-trained
# subsampled profile; the script's guard would skip it gracefully, but being
# explicit is cleaner):
sbatch slurm/describe_dataset_zh.slurm \
  config/profiles/garg_weat_china_ngram.yml \
  config/profiles/garg_weat_renminribao.yml

# PKU — separate server, same parallel option:
sbatch slurm/describe_dataset_pku.slurm
```

These finish in minutes. The Step 2 build/train jobs queue independently and don't notice.

After Step 2 + 3 + 4 complete, the new `models_subsampled/` exists and `sbatch slurm/describe_dataset_zh.slurm` (no args) will pick it up automatically alongside Renminribao.

---

## Reference — full Princeton flow at a glance

| Step | Command | Time (approx) |
|---|---|---|
| 0. Cleanup 06-02 partial weighted (one-time) | `rm -rf corpora_weighted ...` (see Step 0) | seconds |
| 1. Pull corrected code | `git pull` | seconds |
| 2a. Build+train subsampled (subset 1: 6 slices) | `sbatch ..._subsampled_per_slice.slurm "1940_..._1965_1974"` | ≤48h (≈10–18h typical) |
| 2b. Build+train subsampled (subset 2: 6 slices) | `sbatch ..._subsampled_per_slice.slurm "1970_..._1995_2004"` | ≤48h |
| 2c. Build+train subsampled (subset 3: 5 slices) | `sbatch ..._subsampled_per_slice.slurm "2000_..._2020_2020"` | ≤48h |
| 3. Analyze + visualize (Renminribao + subsampled) | `sbatch slurm/garg_weat_zh.slurm` | minutes per slice |
| 4. Methods summary | `sbatch slurm/describe_dataset_zh.slurm` | minutes cold / seconds warm |
| 5. A/B vs presence-only china-ngram | `sbatch slurm/garg_weat_zh.slurm config/profiles/garg_weat_china_ngram.yml` + describe sibling | same as 3+4 |

| Server | Concurrent task | Command | Time |
|---|---|---|---|
| Princeton llm | Existing presence-only + RMRB metadata export (anytime) | `sbatch slurm/describe_dataset_zh.slurm config/profiles/garg_weat_china_ngram.yml config/profiles/garg_weat_renminribao.yml` | minutes cold / seconds warm |
| PKU opinion | Weibo + provincial newspaper metadata export | `sbatch slurm/describe_dataset_pku.slurm` | minutes cold / seconds warm |

---

## Related artifacts

- **Current spec:** [`docs/superpowers/specs/2026-06-03-china-ngram-histwords-style-subsampling-design.md`](../superpowers/specs/2026-06-03-china-ngram-histwords-style-subsampling-design.md)
- **Current plan:** [`docs/superpowers/plans/2026-06-03-china-ngram-histwords-style-subsampling.md`](../superpowers/plans/2026-06-03-china-ngram-histwords-style-subsampling.md)
- **Predecessor runbook (superseded):** [`2026-06-02-china-ngram-weighted-retrain-and-pku-metadata.md`](2026-06-02-china-ngram-weighted-retrain-and-pku-metadata.md)
- **SLURM scripts in this flow:**
  - `slurm/build_train_china_ngram_subsampled_per_slice.slurm` (per-slice driver, 48h-capped)
  - `slurm/garg_weat_zh.slurm` (analyze + visualize, default = Renminribao + subsampled)
  - `slurm/describe_dataset_zh.slurm` (methods summary, default = Renminribao + subsampled)
  - `slurm/describe_dataset_pku.slurm` (PKU side, weibo + provincial newspaper)
- **Last code commit this runbook depends on:** `1f28aa9` (48h time cap + split documentation)
