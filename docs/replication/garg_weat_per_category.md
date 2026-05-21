# Garg-WEAT — per-category relative norm distance

The `garg_weat` analysis mode extends Garg's relative norm distance (RND) from a
single occupation list (the `garg` mode, see [`garg_2018.md`](garg_2018.md)) to
**named concept categories** — `leadership`, `family`, `science` — and orients
them onto a single **gender-ideation axis**. It runs on any per-unit embeddings,
so the same code serves English (COHA, Google Books Ngram) and Chinese
(Renminribao, China Ngram, Weibo, provincial newspapers), longitudinal *and*
provincial.

This is the metric we use as the RND counterpart to the Cohen's d (`weat`) view
of the Chinese corpora. **No model is retrained for it** — every arm reuses
embeddings produced elsewhere (trained or pre-trained); only the analysis and
plots are new. The Cohen's d (`weat`) scripts and configs are left untouched.

---

## Metric

For each category word `w`, with L2-normalized vectors and male/female centroids
`c_male`, `c_female` (means of the L2-normalized gender word vectors):

```
RND(w) = ||v_w − c_male|| − ||v_w − c_female||      (positive = female-leaning)
```

Per `(unit, category)` we report the mean RND over the category's
**consistent set** — words in vocabulary in *every* unit, computed per category —
mirroring Garg's consistent-occupation restriction.

Implemented in `scripts/analyze_garg_weat.py` (metric in
`scripts/common/metrics.py:relative_norm_distance`).

### Gender-ideation axis (the family flip)

The three categories don't share a "less traditional" direction in raw RND:
liberalization pushes leadership/science *toward* women (RND ↑) but family
*toward* men (RND ↓). To put them on one axis where **higher = less
traditional**, `analysis.ideation_sign` multiplies each category's RND by ±1:

```yaml
analysis:
  ideation_sign:
    leadership: 1
    science: 1
    family: -1     # reverse: family female-leaning is the traditional view
```

Applied at plot time (`scripts/visualize.py:apply_ideation_sign`). With the flip,
all three lines rising together means society-wide de-traditionalization; family
diverging is a real finding (the domestic sphere not de-gendering in step), not
a bug.

### Two uncertainty bands

Computed in one pass (`build_summary`), rendered as two figures:

| Band | Method | Default | Question it answers |
|---|---|---|---|
| `bootstrap` | with-replacement resample of the consistent set (Garg's `sns.tsplot`) | 5000 iters, 68% (≈ ±1 SE) | sampling noise over occupations |
| `subsample` | keep a fraction of the words without replacement, fixed across units | 80%, 100 rounds, 95% | sensitivity to *which* words were chosen |

Both configurable under `analysis.bootstrap` / `analysis.subsample`.

---

## Longitudinal vs provincial

`scripts/visualize.py` auto-detects the unit naming and routes the plots
(`_garg_weat_unit_kind`):

- **longitudinal** — decade labels `1990s` or rolling windows `1940_1949` →
  per-category trend over time (`fig2_garg_weat_categories__<src>__{bootstrap,subsample}.pdf`).
- **province** (`北京`) / **province-year** (`北京_2020`) → the cross-province
  RND view, replacing the Cohen's d provincial plots:
  - `garg_weat_provincial_rankings` — barh per category across provinces
  - `garg_weat_provincial_heatmap` — province × category
  - `garg_weat_provincial_choropleth` — per-category maps (per-year + overall
    for province-year; needs geopandas + a China shapefile, skipped gracefully
    otherwise)

A window like `1990_1999` stays longitudinal (both sides are 4-digit years);
province-year requires a non-numeric head.

---

## Arms

Every arm is a config profile pointing at **already-existing** models, plus the
shared `analysis` block above.

| Arm | Profile | Models | Units |
|---|---|---|---|
| COHA (trained) | `garg_weat_coha_trained.yml` | from-scratch SGNS | decades |
| COHA HistWords SGNS/SVD | `garg_weat_coha_histwords_{sgns,svd}.yml` | pre-trained | decades |
| **Google Ngram English (All)** | `garg_weat_google_ngram_eng_all.yml` | pre-trained HistWords | decades |
| **Google Ngram English Fiction** | `garg_weat_google_ngram_eng_fiction_all.yml` | pre-trained HistWords | decades |
| **Renminribao** | `garg_weat_renminribao.yml` | Princeton `/scratch` | `1940_1949` windows |
| **China Ngram** | `garg_weat_china_ngram.yml` | Princeton `/scratch` | `1940_1949` windows |
| **Weibo** | `garg_weat_weibo.yml` | PKU `/gpfs` | provinces |
| **Provincial newspaper** | `garg_weat_provincial_newspaper.yml` | PKU `/lustre` | province-year |

The Chinese arms reuse the exact `models_dir` + `model_name_template` of their
`*_weat.yml` / `*_server.yml` Cohen's d siblings (verified identical) — only
`results_dir` / `figures_dir` differ.

> **English caveat.** HistWords `eng-all` is Google Books **"All English"**
> (British + American + everything) and `eng-fiction-all` is English fiction —
> **neither is US-specific**. COHA remains the American corpus; the Google Books
> arms are a broader-English, larger-scale comparison, not American per se.

---

## Wordlists

| Language | Dir | Notes |
|---|---|---|
| English | `wordlists/en/garg_weat/` | curated `candidates_*.txt` |
| Chinese (formal) | `wordlists/zh/garg_weat_formal/` | for RMRB, China Ngram, newspaper |
| Chinese (informal) | `wordlists/zh/garg_weat_informal/` | for Weibo |

The Chinese category seeds were derived from the existing WEAT *target* lists
(`domestic_work_words.json["family"]`, `leadership_words.json["leadership"]`,
`stem_words.json["stem"]`) plus the register's `gender_words.json`. Each
`candidates_<cat>.txt` is dedup + OOV-pruned into `cleaned_<cat>.txt` by
`scripts/prepare_wordlists.py`; the configs consume the `cleaned_*` lists. The
coverage threshold is **≥80% for English**, **≥70% for Chinese** (the Chinese
seeds include lower-frequency terms that dip below 0.8 in the sparser early
decades). Review/replace the seeds freely.

---

## Running an arm

```bash
# 0. (English pre-trained only) fetch the vectors on a login node
python -m scripts.data_prep.download_pretrained_embeddings \
    --source=google_ngram_eng_all \
    --target_dir=/scratch/network/yh6580/gender-occup/data/pretrained_embeddings
#    → confirm the resolved models_dir in MANIFEST.json matches the profile

# 1. (Chinese arms) build cleaned_*.txt from the seed candidates (needs models)
python -m scripts.prepare_wordlists --config config/profiles/garg_weat_renminribao.yml

# 2. RND analysis on the existing models
python -m scripts.analyze_garg_weat --config config/profiles/<arm>.yml

# 3. figures (trend or provincial, auto-routed by unit naming)
python -m scripts.visualize main --config config/profiles/<arm>.yml
```

### Slurm wrappers (English and Chinese kept separate)

| Script | Server / style | Covers | Default arms |
|---|---|---|---|
| `slurm/prepare_wordlists.slurm` | Princeton | English precheck | COHA configs → `wordlists/en/garg_weat` |
| `slurm/garg_weat_all_sources.slurm` | Princeton | **English** analyze + visualize | COHA trained / HistWords SGNS / SVD + google_ngram eng-all / eng-fiction-all |
| `slurm/prepare_wordlists_zh.slurm` | Princeton | **Chinese** precheck (formal) | RMRB + China-Ngram (pooled) → `wordlists/zh/garg_weat_formal` |
| `slurm/garg_weat_zh.slurm` | Princeton | **Chinese** analyze + visualize | RMRB + China-Ngram (longitudinal) |
| `slurm/prepare_wordlists_pku.slurm` | **PKU** | **Chinese** provincial precheck | Weibo → informal, newspaper → formal (threshold 0.7) |
| `slurm/garg_weat_pku.slurm` | **PKU** | **Chinese** provincial analyze + visualize | Weibo + provincial newspaper |

PKU scripts use that server's conventions (`conda activate opinion`, no `module
load`, PKU mail, `cd` to the `/lustre` checkout) — kept separate from the
Princeton scripts on purpose.

Each multi-source script also accepts an explicit config list as positional
args, and skips any arm whose `models_dir` is missing or empty.

Outputs land in the profile's `results_dir` / `figures_dir`:

```
results_dir/
  garg_weat_rnd_long.parquet               # unit_name, category, occupation, rnd, in_vocab
  garg_weat_summary_by_category.parquet     # + mean_rnd, ci_low/high, sub_low/high/mean, n_*
figures_dir/
  fig2_garg_weat_categories__<src>__bootstrap.pdf   # longitudinal
  fig2_garg_weat_categories__<src>__subsample.pdf
  garg_weat_provincial_{rankings,heatmap}.pdf       # provincial
  garg_weat_choropleth_<category>[_<year>|_overall].pdf
```

---

## Files involved

| File | Role |
|------|------|
| `scripts/analyze_garg_weat.py` | per-category RND analyzer + two bands |
| `scripts/visualize.py` | trend plot, provincial plots, `_garg_weat_unit_kind` dispatch, `apply_ideation_sign` |
| `scripts/data_prep/download_pretrained_embeddings.py` | `google_ngram_eng_{all,fiction_all}` sources |
| `config/profiles/garg_weat_*.yml` | one profile per arm |
| `wordlists/{en,zh}/garg_weat*/` | category + gender wordlists |
| `tests/test_visualize_garg_weat.py`, `tests/test_analyze_garg_weat.py` | coverage |
