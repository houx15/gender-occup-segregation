# Chinese Google 5-gram Occupation-Gender-Prestige Project

This repository supports research on **gendered occupational prestige in Chinese culture** from 1940-2015, using Chinese Google Books 5-gram data and decade-specific word embeddings. The design parallels Wenhao Jiang (2025) for U.S. English, adapted to Chinese.

## Quick Start

```bash
# 1. Clone and setup
git clone <your-repo>
cd gender-occup-segregation
./setup_server.sh  # Interactive setup

# 2. Run pipeline
screen -S gender-analysis
source venv/bin/activate
./run_pipeline.sh 2>&1 | tee pipeline.log
# Ctrl+A, D to detach
```

## Project Overview

The project analyzes how gender associations and prestige perceptions of occupations have evolved in Chinese written culture using word embeddings trained on time-sliced corpora.

**What it computes:**
1. **Gender typing scores**: How occupations associate with male vs. female concepts
2. **Prestige dimensions**: Evaluation (moral worth), potency (power), activity (liveliness), and general prestige

**Pipeline stages:**
1. Download & decompress Chinese Google 5-gram data
2. Build time-sliced corpora (overlapping 10-year windows, 5-year steps)
3. Train Word2Vec models (skip-gram) for each time slice
4. Analyze occupation positions along gender and prestige axes

## Repository Structure

```
.
├── config/
│   ├── config.example.yml     # Template (copy to config.yml)
│   └── config.yml             # Your config (NOT in git)
├── scripts/                   # Four main pipeline scripts
│   ├── download_ngrams.py
│   ├── build_corpora.py
│   ├── train_embeddings.py
│   └── analyze_embeddings.py
├── wordlists/                 # Input word lists
│   ├── occupations_zh.txt
│   ├── gender_words_zh.json
│   └── prestige_axes_zh.json
├── tests/                     # Unit tests
├── data/                      # Generated data (NOT in git)
│   ├── raw_ngrams/
│   ├── corpora/
│   ├── models/
│   └── results/              # Your final outputs!
└── logs/                      # Execution logs (NOT in git)
```

## Requirements

### System
- Python 3.8+
- 100+ GB disk space
- 16+ GB RAM (reduce workers if less)
- Multiple CPU cores (recommended)

### Python Dependencies

```bash
pip install -r requirements.txt
```

Key packages: `gensim`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `PyYAML`, `fire`, `requests`, `beautifulsoup4`

## Installation

### Local Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp config/config.example.yml config/config.yml
# Edit config/config.yml with your paths
```

### Server Setup

```bash
# On your server
git clone <your-repo>
cd gender-occup-segregation
./setup_server.sh  # Automated interactive setup
```

The setup script will:
- Create virtual environment
- Install dependencies
- Create `config/config.yml` from template
- Ask for your server paths
- Set CPU worker count
- Create directory structure
- Run tests

## Configuration

Edit `config/config.yml` (copy from `config.example.yml`):

```yaml
paths:
  base_dir: "/path/to/project"
  raw_ngram_dir: "/path/to/data/raw_ngrams"
  # ... other paths

embedding:
  vector_size: 300
  window: 4
  min_count: 50
  workers: 16  # Match your CPU cores
  epochs: 5

time_slices:
  window_size: 10   # years per corpus
  step_size: 5      # overlap
  start_year: 1940
  end_year: 2015
```

## Usage

### All Scripts Use Fire CLI

All scripts use [Python Fire](https://github.com/google/python-fire) for CLI:

```bash
# General pattern
python script.py --param=value --flag

# Get help
python script.py -- --help
```

### Option 1: Full Pipeline

```bash
./run_pipeline.sh

# With options
./run_pipeline.sh --skip-download --export-plots
```

### Option 2: Step by Step

**Step 1: Download Data**
```bash
python scripts/download_ngrams.py --config=config/config.yml

# Options
python scripts/download_ngrams.py --config=config/config.yml --skip_decompress
python scripts/download_ngrams.py --max_workers=8
```

**Step 2: Build Corpora**
```bash
python scripts/build_corpora.py --config=config/config.yml

# Options
python scripts/build_corpora.py --slice=1940_1949  # Single slice
python scripts/build_corpora.py --overwrite
```

**Step 3: Train Embeddings**
```bash
python scripts/train_embeddings.py --config=config/config.yml

# Options
python scripts/train_embeddings.py --slice=1940_1949
python scripts/train_embeddings.py --retrain
```

**Step 4: Analyze**
```bash
python scripts/analyze_embeddings.py --config=config/config.yml

# Options
python scripts/analyze_embeddings.py --export_plots
python scripts/analyze_embeddings.py --slice=1940_1949
```

## Running on Server

### Using Screen (Recommended)

```bash
# Start session
screen -S gender-analysis

# Activate and run
source venv/bin/activate
./run_pipeline.sh 2>&1 | tee pipeline.log

# Detach: Ctrl+A, then D
# Reattach: screen -r gender-analysis
```

### Using tmux

```bash
tmux new -s gender-analysis
source venv/bin/activate
./run_pipeline.sh 2>&1 | tee pipeline.log
# Detach: Ctrl+B, then D
# Reattach: tmux attach -t gender-analysis
```

### Using nohup

```bash
nohup ./run_pipeline.sh > pipeline.log 2>&1 &
echo $! > pipeline.pid
```

### Parallel Training (Speed Up)

Train different time slices simultaneously:

```bash
# Manual approach
screen -S train_1940
source venv/bin/activate
python scripts/train_embeddings.py --slice=1940_1949

# Automated approach
for slice in 1940_1949 1945_1954 1950_1959 1955_1964 1960_1969 1965_1974 1970_1979 1975_1984 1980_1989 1985_1994 1990_1999 1995_2004 2000_2009 2005_2014 2010_2015; do
  screen -dmS train_${slice} bash -c "source venv/bin/activate && python scripts/train_embeddings.py --slice=${slice} 2>&1 | tee logs/train_${slice}.log"
  sleep 5
done
```

## Monitoring Progress

```bash
# Check logs
tail -f logs/*.log

# Reattach to screen
screen -r gender-analysis

# Check running processes
ps aux | grep python

# Disk usage
df -h
du -sh data/*
```

## Output Files

After analysis completes, find results in `data/results/`:

- **`occupation_gender_typing_by_decade.csv`**: Gender scores over time
  - Columns: occupation, time_slice, start_year, end_year, gender_score, coverage
  - Negative score = male-typed, Positive = female-typed

- **`occupation_prestige_by_decade.csv`**: Prestige dimension scores over time
  - Columns: occupation, time_slice, prestige_evaluation, prestige_potency, prestige_activity, prestige_general_prestige, coverage

- **`occupation_gender_prestige_joint.csv`**: Combined dataset

- **`summary_statistics.csv`**: Time-series correlations and summary metrics

- **`plots/`** (if `--export_plots` used): Diagnostic visualizations

## Downloading Results from Server

```bash
# Compress on server
cd data/results
tar -czf results_$(date +%Y%m%d).tar.gz *.csv plots/

# Download to local machine
scp server:/path/to/results_*.tar.gz ./

# Or use rsync (resumable)
rsync -avzP server:/path/to/data/results/ ./local_results/
```

## Customization

### Add Occupations

Edit `wordlists/occupations_zh.txt`:
```
新职业1
新职业2
```

Then re-run analysis step.

### Modify Gender/Prestige Words

Edit JSON files in `wordlists/`:
- `gender_words_zh.json` - Gender-associated terms
- `prestige_axes_zh.json` - Prestige dimension definitions

### Change Time Windows

Edit `config/config.yml`:
```yaml
time_slices:
  window_size: 15  # Change from 10 to 15 years
  step_size: 5
```

Then re-run corpus building, training, and analysis.

## Methodology

### Word Embeddings
- **Algorithm**: Skip-gram with negative sampling (Word2Vec)
- **Vector size**: 300 dimensions
- **Training**: One model per 10-year time slice
- **Vocabulary**: ~50K-500K tokens per model

### Semantic Axes

Axes are constructed as normalized differences between centroids:

```python
# Gender axis
male_centroid = mean([v_男, v_男人, v_他, ...])
female_centroid = mean([v_女, v_女人, v_她, ...])
gender_axis = normalize(male_centroid - female_centroid)

# Occupation score
gender_score = dot(occupation_vector, gender_axis)
```

### Multi-Character Occupations

Chinese occupations are multi-character (e.g., 工程师). We use a hybrid strategy:
1. Try whole-token embedding if available
2. Fall back to averaging character embeddings
3. Filter out occupations with low character coverage

### Prestige Dimensions

Following Osgood's semantic differential:
1. **Evaluation**: Moral worth (高尚-卑鄙)
2. **Potency**: Power/competence (强大-弱小)
3. **Activity**: Liveliness (活跃-沉闷)
4. **General Prestige**: Social status (高贵-低贱)

## Performance & Resource Usage

### Disk Space
- Raw data: ~50 GB compressed, ~200 GB decompressed
- Corpora: ~100 GB
- Models: ~10 GB
- Results: <1 GB
- **Total: ~360 GB**

### Time Estimates
- Download: 2-6 hours (network dependent)
- Corpus building: 1-2 hours
- Training: 30-60 hours (sequential) or 3-6 hours (parallel)
- Analysis: 30 minutes
- **Total: 36-72 hours sequential, 6-12 hours optimized**

### Memory
- Reduce `workers` in config if < 16GB RAM
- Each model training: 2-8 GB RAM

## Troubleshooting

### Out of Memory
```yaml
# In config/config.yml
embedding:
  workers: 4  # Reduce from 16
```

### Out of Disk Space
```bash
# After corpus building, can remove decompressed files
rm -rf data/raw_ngrams_decompressed/*
```

### Download Failures
Rerun the script - it automatically skips completed files:
```bash
python scripts/download_ngrams.py --config=config/config.yml
```

### Process Died
Check logs and restart - scripts skip completed work:
```bash
tail -100 logs/*.log
./run_pipeline.sh --skip-download  # Skip done steps
```

### Low Occupation Coverage
Adjust in `config/config.yml`:
```yaml
analysis:
  min_coverage: 0.3  # Lower from 0.5
  occupation_strategy: "average_chars"  # Or "hybrid"
```

## Testing

```bash
# Run all tests
python -m unittest discover tests/

# Specific test
python tests/test_build_corpora.py
```

## File Locations Summary

**Tracked in Git:**
- Scripts, word lists, tests
- Documentation (*.md)
- `config/config.example.yml` (template)
- `requirements.txt`, `.gitignore`

**NOT in Git (Generated):**
- `config/config.yml` (your server paths)
- `data/` (360+ GB)
- `logs/` (execution logs)
- `venv/` (Python environment)

## Technical Details

### Code Statistics
- 4 main scripts: ~1,500 lines Python
- Word lists: 80+ occupations, 14+ gender terms, 40+ prestige terms
- Tests: ~235 lines

### Dependencies
See `requirements.txt`. Core:
- gensim (embeddings)
- pandas (data manipulation)
- numpy (linear algebra)
- fire (CLI)

### Reproducibility
- Fixed random seeds (seed: 42)
- Deterministic training (where possible)
- Version-controlled configuration
- Metadata saved with each model

## Citation

If you use this code, please cite:

```
[Your paper citation here]
```

This project adapts methods from:
```
Jiang, Wenhao. 2025. "Cultural Symbolic Biases and Wage Inequality."
```

## Data Source

- Google Books Ngram (Chinese simplified, 5-grams, version 20200217)
- Licensed under CC BY 3.0
- URL: https://storage.googleapis.com/books/ngrams/books/datasetsv3.html

## License

[Specify your license]

## Contact

[Your contact information]

## Acknowledgments

- Google Books Ngram data
- Gensim library for Word2Vec
- Methods adapted from Jiang (2025)
