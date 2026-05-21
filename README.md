# Gender-Occupation Segregation Analysis

Unified framework for analyzing **gender norms and occupational segregation in Chinese text** using word embeddings. Supports four data sources and two analysis modes across servers.

## Overview

This project measures how gender associations with occupations and social concepts evolve over time and vary across regions, using Word2Vec embeddings trained on large-scale Chinese text corpora.

**Data sources:**
- **Google N-grams** — Chinese simplified 5-grams (1940-2015), national longitudinal
- **Renminribao (People's Daily)** — newspaper text (1940s-2010s), national longitudinal
- **Weibo** — social media posts by province, provincial cross-sectional
- **Newspaper** — regional newspapers mapped to provinces, provincial cross-sectional

**Analysis modes:**
- **Prestige** — Projects occupations onto gender + 4 prestige axes (evaluation, potency, activity, general prestige). Computes Pearson correlations between gender typing and prestige over time.
- **WEAT** — Computes Cohen's d effect sizes for 3 gender norm dimensions: work-family, leadership, and STEM. Supports cross-provincial comparison and correlation with socioeconomic indicators.
- **Garg-WEAT** — Garg's relative norm distance (RND) per concept category (leadership / family / science), oriented onto a single gender-ideation axis (higher = less traditional), with bootstrap + word-subsample bands. The RND counterpart to WEAT's Cohen's d; runs on the same models, longitudinal or provincial. See [`docs/replication/garg_weat_per_category.md`](docs/replication/garg_weat_per_category.md).

## Quick Start

```bash
# 1. Clone and configure
git clone <repo-url>
cd gender-occup-segregation
cp config/config.example.yml config/config.yml
# Edit config/config.yml with your server paths and data source

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download raw data on the login node (Slurm compute nodes have no internet)
python -m scripts.data_prep.download_ngrams --config config/config.yml

# 4. Submit corpus building + training + analysis to Slurm
sbatch slurm/full_pipeline.slurm
# (run_pipeline.sh auto-skips the download step when raw files already exist)
```

## Repository Structure

```
gender-occup-segregation/
├── config/
│   ├── config.example.yml          # Full config template (all data sources)
│   └── profiles/                   # Ready-to-use configs per server
│       ├── ngram_server.yml
│       ├── renminribao_server.yml
│       ├── weibo_server.yml
│       └── newspaper_server.yml
├── scripts/
│   ├── common/                     # Shared utilities
│   │   ├── config_loader.py        # Unified config loading + validation
│   │   ├── embedding_utils.py      # Axis construction, projection, Cohen's d
│   │   └── logging_utils.py        # Logging setup
│   ├── data_prep/                  # Data source-specific corpus building
│   │   ├── download_ngrams.py      # Google n-gram download
│   │   ├── build_corpora_ngram.py  # N-gram -> time-sliced corpora
│   │   ├── build_corpora_rmrb.py   # Renminribao -> time-sliced corpora
│   │   ├── build_corpora_weibo.py  # Weibo -> province-level corpora
│   │   ├── build_corpora_newspaper.py  # Newspaper -> province-level corpora
│   │   └── province_mapper.py      # Newspaper name -> province mapping pipeline
│   ├── train_embeddings.py         # Unified Word2Vec trainer (time slices or provinces)
│   ├── analyze_prestige.py         # Prestige mode: gender + prestige dimension scores
│   ├── analyze_weat.py             # WEAT mode: 5-step Cohen's d pipeline
│   ├── analyze_correlation.py      # Correlate indices with external socioeconomic data
│   └── visualize.py                # Unified visualization (prestige + WEAT plots)
├── wordlists/
│   ├── prestige/                   # Prestige mode wordlists
│   │   ├── occupations_zh.txt      # Occupation terms
│   │   ├── gender_words_zh.json    # Male/female term pairs
│   │   ├── prestige_axes_zh.json   # 4 prestige dimension definitions
│   │   └── occup_category.json     # Occupation category groupings
│   ├── weat_formal/                # WEAT wordlists for formal text (newspaper, ngram)
│   │   ├── gender_words.json
│   │   ├── domestic_work_words.json
│   │   ├── leadership_words.json
│   │   └── stem_words.json
│   └── weat_informal/              # WEAT wordlists for informal text (Weibo)
│       ├── gender_words.json       # Includes colloquial terms (帅哥, 闺蜜, etc.)
│       ├── domestic_work_words.json
│       ├── leadership_words.json
│       └── stem_words.json
├── provincial/                     # Provincial socioeconomic data processing
│   ├── clean_provincial_data.py    # Clean survey/census data into standard format
│   └── data/                       # CSV/Excel data files (not in git)
├── slurm/                          # Slurm job templates
│   ├── build_corpus.slurm
│   ├── train.slurm
│   ├── analyze.slurm
│   └── full_pipeline.slurm
├── tests/
├── run_pipeline.sh                 # Full pipeline with auto-skip
├── requirements.txt
└── reference/                      # Original project code (archive, not in git)
```

## Configuration

Copy the example config and edit for your server:

```bash
cp config/config.example.yml config/config.yml
```

Or start from a profile:

```bash
cp config/profiles/weibo_server.yml config/config.yml
```

Key fields:

```yaml
data_source: "ngram"          # "ngram", "renminribao", "weibo", "newspaper"
analysis_mode: "prestige"     # "prestige" or "weat"

paths:
  base_dir: "/path/to/project"
  corpora_dir: "/path/to/corpora"
  models_dir: "/path/to/models"
  results_dir: "/path/to/results"
  log_dir: "/path/to/logs"

wordlists:
  dir: "wordlists/prestige"  # or "wordlists/weat_formal", "wordlists/weat_informal"

embedding:
  model_name_template: "chi_sim_5gram_{unit_name}.model"  # must match your model filenames
```

The `model_name_template` determines how scripts discover and name models. `{unit_name}` is replaced with the time slice (e.g., `1940_1949`) or province name (e.g., `北京`).

## Usage

### Full Pipeline (Recommended)

```bash
./run_pipeline.sh --config config/config.yml
```

Each stage **auto-skips** if its output already exists. Use `--force-*` to re-run:

```bash
./run_pipeline.sh --force-train     # Re-train even if models exist
./run_pipeline.sh --force-all       # Re-run everything
```

### Step by Step

**1. Build corpora** (data source-specific):

```bash
# N-gram
python -m scripts.data_prep.build_corpora_ngram --config=config/config.yml

# Renminribao
python -m scripts.data_prep.build_corpora_rmrb --config=config/config.yml

# Weibo (by province group for parallel execution)
python -m scripts.data_prep.build_corpora_weibo --config=config/config.yml --year=2024 --group=0

# Newspaper (requires province mapping first, see below)
python -m scripts.data_prep.build_corpora_newspaper --config=config/config.yml
```

**2. Train embeddings:**

```bash
python -m scripts.train_embeddings --config=config/config.yml

# Train specific unit
python -m scripts.train_embeddings --config=config/config.yml --unit=1940_1949

# Train specific province group (for parallel Slurm jobs)
python -m scripts.train_embeddings --config=config/config.yml --group=0
```

**3. Analyze:**

```bash
# Prestige mode
python -m scripts.analyze_prestige --config=config/config.yml

# WEAT mode
python -m scripts.analyze_weat --config=config/config.yml
```

**4. Visualize:**

```bash
python -m scripts.visualize --config=config/config.yml
```

**5. Correlate with external data** (optional):

```bash
python -m scripts.analyze_correlation --config=config/config.yml
```

### Using Existing Models

If you already have trained models and just want to analyze + visualize:

```bash
# Point config to your models directory, then:
python -m scripts.analyze_weat --config=config/config.yml
python -m scripts.visualize --config=config/config.yml
```

The pipeline auto-skips download, corpus building, and training when output files exist.

### Cross-Mode Analysis

Any data source can use either analysis mode. For example, to run WEAT on n-gram models:

```yaml
data_source: "ngram"
analysis_mode: "weat"
wordlists:
  dir: "wordlists/weat_formal"
embedding:
  model_name_template: "chi_sim_5gram_{unit_name}.model"
```

## Newspaper Pipeline

For newspaper data, an additional step maps newspaper names to provinces:

```bash
# 1. Extract newspaper names from raw JSONL files
python -m scripts.data_prep.province_mapper extract --config=config/config.yml

# 2. Auto-map newspapers to provinces
python -m scripts.data_prep.province_mapper map --config=config/config.yml

# 3. Review unknowns, add manual fixes to MANUAL_MAPPINGS in province_mapper.py, then:
python -m scripts.data_prep.province_mapper add --config=config/config.yml

# 4. Build corpora (uses the mapping)
python -m scripts.data_prep.build_corpora_newspaper --config=config/config.yml
```

## Correlation Analysis

Correlates WEAT Cohen's d indices with external socioeconomic data. Supports both provincial and longitudinal modes, auto-detected from your CSV.

**Provincial CSV format** (for Weibo/newspaper analysis):

```csv
province,gdp,income,education_years,employment_rate
北京,43760,85000,13.2,0.78
广东,13500,55000,10.8,0.72
```

Province names can be short (`北京`) or full (`北京市`, `内蒙古自治区`).

**Longitudinal CSV format** (for n-gram/Renminribao analysis):

```csv
year,female_labor_rate,gdp_per_capita,literacy_rate
1940,0.45,1200,0.32
1950,0.52,1500,0.45
```

The `year` column is matched to time slice start years.

All numeric columns are auto-discovered for correlation. To select specific columns or add derived variables (log, diff), configure in `config.yml`:

```yaml
correlation:
  external_data: "path/to/your_data.csv"
  variables:
    - column: "gdp"
      label: "GDP"
    - "employment_rate"
  transforms:
    - name: "log_gdp"
      op: "log"
      source: "gdp"
```

## Running on Slurm

Slurm compute nodes have no internet access. Always download raw data on a login node first:

```bash
# Login node — download raw n-gram or COHA files
python -m scripts.data_prep.download_ngrams --config config/config.yml   # Chinese/English ngram
python -m scripts.data_prep.download_coha   --config config/profiles/coha_server.yml  # COHA
```

Then submit the rest to Slurm (`run_pipeline.sh` auto-skips the download step when files exist):

```bash
# Full pipeline (corpus → train → analyze → visualize)
sbatch slurm/full_pipeline.slurm
sbatch slurm/full_pipeline_en.slurm config/profiles/ngram_en_server.yml  # English

# Individual stages
sbatch slurm/build_corpus.slurm
sbatch slurm/train.slurm
sbatch slurm/analyze.slurm

# Train specific province group (parallel jobs)
sbatch slurm/train.slurm config/config.yml --group=0
sbatch slurm/train.slurm config/config.yml --group=1
```

Edit the `#SBATCH` headers in the slurm scripts to adjust resources and email.

## Output Files

### Prestige Mode

| File | Description |
|------|-------------|
| `occupation_scores_by_slice.parquet` | Gender + prestige scores per occupation per time slice |
| `occupation_scores_by_province.parquet` | Same, for provincial analysis |
| `summary_statistics.parquet` | Gender-prestige correlations per unit |

### WEAT Mode

| File | Description |
|------|-------------|
| `weat_results.csv` | Cohen's d per unit per dimension (long format) |
| `gender_norm_index.csv` | Cohen's d per unit (wide format, main result) |
| `word_projections.csv` | All concept word projections onto gender axes |
| `oov_unit_coverage.csv` | OOV diagnostics per unit |
| `gender_axes.csv` | Gender axis metadata |

### Garg-WEAT Mode

| File | Description |
|------|-------------|
| `garg_weat_rnd_long.parquet` | RND per category word per unit (long: unit_name, category, occupation, rnd, in_vocab) |
| `garg_weat_summary_by_category.parquet` | Per (unit, category) mean RND + bootstrap and subsample bands (main result) |

### Visualizations

| Plot | Mode | Description |
|------|------|-------------|
| `prestige_by_gender_over_time` | Prestige | Top 10% male vs female occupations' prestige scores |
| `gender_prestige_correlation_over_time` | Prestige | Pearson r between gender and prestige per time slice |
| `prestige_by_category_over_time` | Prestige | Prestige trends by occupation category |
| `weat_cohens_d_heatmap` | WEAT | Heatmap of Cohen's d across units and dimensions |
| `weat_rankings` | WEAT | Bar chart ranking units by Cohen's d |
| `weat_longitudinal_trend` | WEAT | Line chart of Cohen's d over time (longitudinal) |
| `weat_longitudinal_by_dimension` | WEAT | Per-dimension trend with effect size reference lines |
| `weat_projection_boxplots` | WEAT | Projection distributions per unit (diagnostic) |
| `weat_choropleth_*` | WEAT | Provincial maps (requires geopandas + shapefile) |
| `fig2_garg_weat_categories__<src>__{bootstrap,subsample}` | Garg-WEAT | Per-category RND trend on the gender-ideation axis (longitudinal) |
| `garg_weat_provincial_{rankings,heatmap}` | Garg-WEAT | Cross-province RND per category |
| `garg_weat_choropleth_*` | Garg-WEAT | Provincial RND maps (requires geopandas + shapefile) |
| `*_correlation.pdf` | Correlation | Scatter plots with regression lines |

## English Pipeline

The framework supports English-language corpora alongside the Chinese pipeline.

**Supported English sources:**
- **Google Books Ngram (English)** — 5-gram files from the Google Ngram v3 corpus (1800–2019), producing time-sliced corpora identical in structure to the Chinese n-gram pipeline.
- **COHA (Corpus of Historical American English)** — decade-level 4-gram files for 1810s–2010s (support is implemented; large-scale runs deferred).

English wordlists live under `wordlists/en/` (subdirectories `prestige/` and `weat_formal/`) and follow the same JSON/TXT conventions as the Chinese wordlists. NLTK punkt\_tab and stopwords data are required before the first run:

```bash
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords')"
```

**Quick start (English Ngram on Princeton Slurm):**

Slurm compute nodes have no internet access. Download on a login node first, then submit the rest as a batch job.

```bash
# Step 1 — login node: download raw n-gram files (~300 GB)
python -m scripts.data_prep.download_ngrams --config config/profiles/ngram_en_server.yml

# Step 2 — submit corpus building + training + analysis to Slurm
sbatch slurm/full_pipeline_en.slurm config/profiles/ngram_en_server.yml
```

`run_pipeline.sh` auto-skips the download step when the raw files already exist, so the Slurm job picks up cleanly from corpus building.

The English builder lowercases all tokens, strips punctuation (apostrophes are preserved), and writes one `corpus_{index}.txt` per ngram file into the standard `corpora_dir/{start}_{end}/` slice directories — the same layout consumed by `train_embeddings.py`.

**Quick start (COHA on Princeton Slurm):**

COHA archives are email-gated. You must sign up at [corpusdata.org](https://www.corpusdata.org/) first — they email you a page with download URLs for the free n-gram archives (1-gram through 5-gram, 1810s–2010s).

```bash
# Step 1 — paste the URLs into config/profiles/coha_server.yml under coha.source_archive_urls
#          and set coha.n to match the n-gram size you downloaded (e.g. n: 5 for 5-grams)

# Step 2 — login node: download + decompress
python -m scripts.data_prep.download_coha --config config/profiles/coha_server.yml

# Step 3 — submit corpus building + training + analysis to Slurm
sbatch slurm/full_pipeline_en.slurm config/profiles/coha_server.yml
```

Notes on COHA data layout:
- Free archives are bundled by n-gram size (`coha-5-grams.zip`, etc.), **not** by decade. After decompression, each archive expands into per-decade TSV files that the builder auto-buckets into decade slice dirs (`1940s/`, `1950s/`, …).
- The builder reads decade from filenames via regex — verify one decompressed filename contains a decade marker (e.g. `..._1940s_...`) before running the pipeline. If not, adjust `decade_from_filename` in `scripts/data_prep/build_corpora_coha.py`.
- Do not mix n-gram sizes in one run. Set `coha.n` to the size you downloaded and keep one archive set per config.

### Garg (2018) replication

A third analysis mode, `garg`, replicates Fig 2 of Garg et al. 2018 (gender-occupation bias trend across COHA decades) using the relative norm distance metric. The full methodology, training-parameter rationale, end-to-end run instructions (including Dropbox-shared COHA archives), and acceptance criteria live in [`docs/replication/garg_2018.md`](docs/replication/garg_2018.md).

```bash
# Same flow as COHA above, but with the Garg-pinned profile
python -m scripts.data_prep.download_coha --config config/profiles/coha_garg.yml
sbatch slurm/full_pipeline_en.slurm config/profiles/coha_garg.yml
# → figures_garg/fig2_garg_replication.pdf
```

### Garg-WEAT (per-category RND)

The `garg_weat` mode extends Garg's RND to concept categories (leadership /
family / science) on a gender-ideation axis, and runs on every corpus —
including pre-trained HistWords **Google Books Ngram** English vectors that need
no training:

```bash
# Pre-trained Google Books "English (All)" + "English Fiction" vectors
python -m scripts.data_prep.download_pretrained_embeddings \
    --source=google_ngram_eng_all \
    --target_dir=/scratch/network/yh6580/gender-occup/data/pretrained_embeddings
python -m scripts.analyze_garg_weat --config config/profiles/garg_weat_google_ngram_eng_all.yml
python -m scripts.visualize main      --config config/profiles/garg_weat_google_ngram_eng_all.yml
```

The same mode powers the Chinese RND arms (Renminribao, China Ngram, Weibo,
provincial newspapers) on their existing models. Full methodology, the family
sign-flip, the two uncertainty bands, every arm, and run commands are in
[`docs/replication/garg_weat_per_category.md`](docs/replication/garg_weat_per_category.md).

## Methodology

### Word Embeddings

- **Algorithm**: Word2Vec skip-gram with negative sampling (gensim)
- **Vector size**: 300 dimensions
- **Training**: One model per analysis unit (time slice or province)
- **Tokenization**: Whitespace (n-grams), jieba (Chinese text)

### Prestige Analysis

Semantic axes are constructed as normalized centroid differences:

```
gender_axis = normalize(centroid(female_terms) - centroid(male_terms))
prestige_axis = normalize(centroid(high_terms) - centroid(low_terms))

gender_score(occupation) = dot(occupation_vector, gender_axis)
prestige_score(occupation) = dot(occupation_vector, prestige_axis)
```

### WEAT Analysis

Follows the Word Embedding Association Test framework:

1. **Gender axis**: `female_centroid - male_centroid` (normalized)
2. **Projections**: Each concept word projected onto the gender axis (cosine similarity)
3. **Cohen's d**: Effect size comparing two concept groups on the gender axis
   - Work-Family: `d = (family_mean - work_mean) / pooled_std`
   - Leadership: `d = (non_leadership_mean - leadership_mean) / pooled_std`
   - STEM: `d = (non_stem_mean - stem_mean) / pooled_std`
   - Positive d = traditional gender norm direction

### Wordlist Design

- **`prestige/`**: Occupation terms + prestige dimension poles from Osgood's semantic differential
- **`weat_formal/`**: Gender/concept terms suited for formal text (newspaper, books)
- **`weat_informal/`**: Adds colloquial terms (帅哥, 闺蜜, 带娃) for social media text

## Requirements

- Python 3.8+
- `gensim`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `PyYAML`, `fire`
- `jieba` (for Chinese text tokenization; not needed for n-gram data)
- `scipy` (for Pearson correlation p-values)
- Optional: `geopandas` (for choropleth maps)

```bash
pip install -r requirements.txt
```

## What's in Git vs Not

**Tracked:**
- All scripts, wordlists, configs (except `config.yml`), slurm templates, tests

**Not tracked:**
- `config/config.yml` (server-specific paths)
- `data/` (corpora, models, results — too large)
- `provincial/data/` (survey CSVs, `.dta` files)
- `logs/`, `figures/`
- `reference/` (archived original project code)

## Citation

```
[Your paper citation here]
```

This project adapts methods from:
- Jiang, Wenhao. 2025. "Cultural Symbolic Biases and Wage Inequality."
- Caliskan, Aylin, Joanna J. Bryson, and Arvind Narayanan. 2017. "Semantics derived automatically from language corpora contain human-like biases." Science 356(6334): 183-186.

## Data Sources

- Google Books Ngram (Chinese simplified, 5-grams, v20200217) — CC BY 3.0
- Renminribao corpus
- Weibo social media data
- Chinese regional newspaper corpus
