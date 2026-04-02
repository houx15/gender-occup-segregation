# Agent Guidelines for Gender-Occupation Segregation Project

## Project Overview

Research project measuring gender norms and occupational segregation in Chinese text using Word2Vec embeddings. Supports four data sources (Google N-grams, Renminribao, Weibo, Newspaper) and two analysis modes (Prestige, WEAT).

## ⚠️ CRITICAL OPERATIONAL RULES

### 1. SLURM Usage (MANDATORY)
**This is a login node of a computational server (HPC cluster).** CPU-intensive or long-running tasks MUST be submitted via SLURM, never run directly on the login node:

```bash
# Submit job to SLURM
sbatch slurm/script_name.slurm

# NEVER run directly on login node:
# python train_embeddings.py ...  ← BAD
# python build_corpora.py ...    ← BAD
```

Quick commands (< 30 seconds, minimal CPU) are acceptable on login node (e.g., `python -c "import yaml; ..."` for config checks).

### 2. Data Handling
- **Data files are large** (12GB+ for provincial newspaper data). NEVER read entire datasets into memory.
- Always use streaming, chunked processing, or `usecols`/`nrows` for exploration.
- When exploring data format, read only first few lines/records.
- Corpus builders use rolling file writers to bound memory usage.

### 3. Project Organization
- **Main work folder**: `/lustre/home/2401111059/gender-occup-segregation/`
- **Docs, notes, plans**: Save here (not in youth-analysis/)
- **Data lives externally**: Raw newspaper data at `/lustre/home/2401111059/newspaper_data/`
- Weibo data, ngram data, etc. each have their own external locations

### 4. Province Name Mapping for Provincial Newspapers
The provincial newspaper folder names map to province names as follows:
- Most are direct: `北京日报` → `北京`, `四川日报` → `四川`
- Special cases:
  - `广东日报` → `广东` (actual paper is 南方日报)
  - `上海日报（解放日报）` → `上海`
  - `山东日报（大众日报）` → `山东`
  - `江苏日报` folder does NOT exist (no 江苏日报 data; 新华日报 not in dataset)

### 5. Data Format: Provincial Newspaper Text Files
- Location: `/lustre/home/2401111059/newspaper_data/provincial_newspaper/{省日报}/YYYY/MM/YYYY-MM-DD.txt`
- Each `.txt` file: single line (no line breaks), tab-separated articles
- Content: raw newspaper text in Chinese (UTF-8)
- Total: ~12GB, 30 provinces, years vary (2007-2024 depending on province)

## Build/Run Commands

### Config-driven pipeline
```bash
# Full pipeline (auto-skips existing outputs)
./run_pipeline.sh --config config/config.yml

# Or step by step via SLURM
sbatch slurm/build_corpus.slurm
sbatch slurm/train.slurm
sbatch slurm/analyze.slurm
```

### Testing
```bash
python -m pytest tests/
```

## Code Style
- Python with Fire CLI (`python -m scripts.module --arg=value`)
- YAML configuration (single source of truth)
- Chinese comments and documentation where appropriate
- Type hints recommended
- Imports: stdlib → third-party → local

## Pipeline Architecture

```
Raw Data → Corpus Builder → Embedding Trainer → WEAT/Prestige Analyzer → Visualizer
   ↓            ↓                   ↓                    ↓                  ↓
.txt/.jsonl  corpus_* files    .kv/.model files    .csv results       .pdf figures
```

Each data source has its own corpus builder but shares the same trainer/analyzer/visualizer.
