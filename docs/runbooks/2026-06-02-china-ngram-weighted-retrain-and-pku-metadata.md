# Runbook — China Ngram Count-Weighted Retrain (Princeton) + Weibo / Provincial Newspaper Metadata Export (PKU)

**Date:** 2026-06-02
**Status:** Superseded by [`2026-06-03-china-ngram-subsampled-retrain-and-pku-metadata.md`](2026-06-03-china-ngram-subsampled-retrain-and-pku-metadata.md). The Princeton workflow described here used the methods-wrong `capped_repetition` weighting (`min(match_count, cap)`); the corrected HistWords-style per-year-capped subsampling lives in the 06-03 runbook. The PKU workflow is unchanged — copied forward verbatim in the new runbook.

> **2026-06-03 update:** Do NOT follow the Princeton steps below. The profile `garg_weat_china_ngram_weighted.yml` and the driver `build_train_china_ngram_weighted_per_slice.slurm` have been deleted; the partial 114 GB `corpora_weighted/` on Princeton scratch should be deleted. See the 06-03 runbook for the corrected procedure.

**Audience:** historical record only. For current procedure, use the 06-03 runbook.

This runbook covers two independent workflows. Run either or both — they don't depend on each other.

---

## Princeton — count-weighted China Ngram retrain

**What you get when this finishes:** a second set of china-ngram models trained on count-weighted 5-gram corpora (each `(ngram, year)` row contributes `min(match_count, 100)` copies), living next to the existing presence-only models. Plus all the analyze/visualize outputs and a methods summary.

**Prereqs:**
- Repo checked out at `/home/yh6580/gender-occup-segregation` (or wherever your Princeton checkout lives) and pulled to a SHA at or after `4240886`.
- `module load anaconda3/2023.3 && conda activate llm` works from a login node.
- `/scratch/network/yh6580/gender-occup/raw_ngrams/` has the `5-*-of-00105.gz` shards (same source as the existing presence-only build).
- `/scratch/network/yh6580/` has enough free space for `~30-50×` the presence-only `corpora/` directory under `corpora_weighted/`.

### Step 1 — Build the count-weighted corpora

```bash
sbatch slurm/build_corpus_zh.slurm config/profiles/garg_weat_china_ngram_weighted.yml
```

What this does: decompresses each of 105 `.gz` shards in turn, parses every `(ngram, year, match_count, volume_count)` triple, emits `min(match_count, repeat_cap=100)` copies of the ngram into every matching time-slice's `corpus_*.txt` (17 slices total, 10-year window, 5-year step). Decompressed shards are deleted after processing — disk only holds one shard at a time on top of the corpora.

**Watch for timing.** First shard's processing time is your wall-clock signal. The job header is `--time=24:00:00`; if your projected total exceeds that, kill the job (`scancel <jobid>`) and either:
- Bump the time line in `slurm/build_corpus_zh.slurm` (`#SBATCH --time=48:00:00`)
- Split via `--file_name=` arg and submit shards in parallel jobs

Progress lives in `logs/build_corpus_zh_<jobid>.out` — the `Processed N,000,000 lines from <shard>` log lines every million tell you where you are within a shard.

### Step 2 — Train the embeddings

After Step 1 finishes:

```bash
sbatch slurm/train_zh.slurm config/profiles/garg_weat_china_ngram_weighted.yml
```

What this does: walks `corpora_weighted/*/`, trains one Word2Vec model per slice, saves under `models_weighted/chi_sim_5gram_<slice>.model`. Hyperparameters (`vector_size=300, window=4, min_count=50, sg=1, negative=15, epochs=5`) match the presence-only profile so the two model sets are directly comparable.

**Heads-up on memory.** Job header is `--mem=64G` (bumped from the PKU sibling because gensim's vocab table grows when training on 20–50× more tokens). If you OOM, bump to 128G or higher.

**Heads-up on incomplete builds.** The trainer walks whatever slice directories exist under `corpora_weighted/`. If Step 1 was killed before all 17 slices finished, Step 2 will only train the slices that exist. That's fine — re-run Step 1 to fill in the missing slices later, then re-run Step 2 with `--retrain` if you want to overwrite the partial models.

### Step 3 — Analyze + visualize

After Step 2 finishes:

```bash
sbatch slurm/garg_weat_zh.slurm
```

The `_zh.slurm` default-runs **both** the Renminribao profile and the China-Ngram-weighted profile (since the swap in commit `040c3e2`). Both arms run through the `analyze_category_bias` orchestrator → both metrics (`rnd` + `cohens_d`) → `visualize`. Outputs land at:

- `/scratch/network/yh6580/renminribao/results/` + `figures/` — Renminribao
- `/scratch/network/yh6580/gender-occup/results_garg_weat_china_ngram_weighted/` + `figures_garg_weat_china_ngram_weighted/` — China-Ngram weighted

### Step 4 — Methods-section summary

```bash
sbatch slurm/describe_dataset_zh.slurm
```

Same default-config pair as Step 3 (Renminribao + China-Ngram weighted). Writes `dataset_summary.md` per source under each profile's `results_dir`. Content: corpus totals (docs, tokens, raw vocab, model vocab range), raw-data footprint (files + bytes), training hyperparameters, per-unit breakdown table with Year range + Tokens/doc columns.

### Optional — A/B against the existing presence-only china-ngram models

The old `garg_weat_china_ngram.yml` profile and its trained models are untouched on disk. To regenerate analyze + summary outputs against them:

```bash
sbatch slurm/garg_weat_zh.slurm        config/profiles/garg_weat_china_ngram.yml
sbatch slurm/describe_dataset_zh.slurm config/profiles/garg_weat_china_ngram.yml
```

No build / train needed — these reuse the existing `corpora/` and `models/` directories. The two `dataset_summary.md` files (presence-only vs weighted) make the side-by-side methods-section comparison straightforward.

---

## PKU — Weibo + provincial-newspaper metadata export

**What you get:** two `dataset_summary.md` files, one per source, ready to paste into the methods section.

**No retrain needed.** Weibo and provincial newspaper were trained on real running text from the start; the count-weighting question is a 5-gram artifact and doesn't apply here. This is purely an analyze-existing-corpora step.

**Prereqs:**
- Repo checked out at `/lustre/home/2401111059/gender-occup-segregation` and pulled to a SHA at or after `425ee7e` (the `describe_dataset` enhancements).
- `~/miniconda3` + `conda activate opinion` works from a login node.
- Existing corpora and models present at the paths in `garg_weat_weibo.yml` and `garg_weat_provincial_newspaper.yml`.

### Run both

```bash
cd /lustre/home/2401111059/gender-occup-segregation
git pull
sbatch slurm/describe_dataset_pku.slurm
```

The default configs are `garg_weat_weibo.yml` + `garg_weat_provincial_newspaper.yml`. Outputs:

- `<weibo profile's results_dir>/dataset_summary.md` — Weibo methods table
- `/lustre/.../data/provincial_newspaper/results_garg_weat/dataset_summary.md` — provincial newspaper methods table

### Run one of the two

```bash
sbatch slurm/describe_dataset_pku.slurm config/profiles/garg_weat_provincial_newspaper.yml
sbatch slurm/describe_dataset_pku.slurm config/profiles/garg_weat_weibo.yml
```

### Performance expectations

- **Cache-cold first run.** Weibo opens parquet metadata to bucket files by province (one schema-read + one row-count per parquet shard) — expect minutes per source. Provincial newspaper rglobs every province-year directory for file counts — also minutes.
- **Cache-warm re-runs.** `.dataset_stats.json` sidecars in each `corpora_dir/<unit>/` directory mean second-pass runs are seconds. `--force` recomputes if a corpus has been rebuilt under the same directory.

---

## Reference — full workflow at a glance

| Server | Step | Command | Time (approx) |
|---|---|---|---|
| Princeton llm | 1. Build weighted corpora | `sbatch slurm/build_corpus_zh.slurm config/profiles/garg_weat_china_ngram_weighted.yml` | Hours–day per cold run; 105 shards × 17 slices |
| Princeton llm | 2. Train weighted embeddings | `sbatch slurm/train_zh.slurm config/profiles/garg_weat_china_ngram_weighted.yml` | Tens of hours per slice; 17 slices |
| Princeton llm | 3. Analyze + visualize (Renminribao + weighted China-Ngram) | `sbatch slurm/garg_weat_zh.slurm` | Minutes per slice |
| Princeton llm | 4. Methods summary | `sbatch slurm/describe_dataset_zh.slurm` | Minutes cold; seconds warm |
| Princeton llm | (Optional) A/B vs presence-only | `sbatch slurm/garg_weat_zh.slurm config/profiles/garg_weat_china_ngram.yml` + describe sibling | Same as steps 3+4 |
| PKU opinion | Weibo + provincial newspaper metadata | `sbatch slurm/describe_dataset_pku.slurm` | Minutes cold; seconds warm |

## Related artifacts

- **Spec:** `docs/superpowers/specs/2026-06-02-china-ngram-weighting-and-summary-enhancements-design.md`
- **Plan:** `docs/superpowers/plans/2026-06-02-china-ngram-weighting-and-summary-enhancements.md`
- **Earlier dataset_summary spec:** `docs/superpowers/specs/2026-06-02-dataset-summary-design.md`
- **Earlier dataset_summary plan:** `docs/superpowers/plans/2026-06-02-dataset-summary.md`
- **SLURM scripts touched in this work:** `slurm/build_corpus_zh.slurm`, `slurm/train_zh.slurm`, `slurm/build_corpus.slurm` (PKU, typo fix), `slurm/garg_weat_zh.slurm`, `slurm/describe_dataset_zh.slurm`, `slurm/describe_dataset_pku.slurm`
- **Last relevant commit on `main`:** `4240886` (build_corpora_ngram decompressed_dir auto-create + skip-on-failure)
