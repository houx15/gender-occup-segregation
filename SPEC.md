# Chinese Google 5-gram Occupation–Gender–Prestige Project

This repository is intended to support a research project on **gendered occupational prestige in Chinese culture over the last ~century**, using **Chinese Google Books 5-gram data** and **decade-specific word embeddings**. The overall design closely parallels Wenhao Jiang (2025) for U.S. English, but adapted to Chinese. 

The codebase is organized into four main components:

1. **Data download & unpacking**: fetch and decompress all Chinese 5-gram files.
2. **Corpus construction**: build overlapping 10-year sub-corpora (sliding by 5 years) between 1940 and 2015.
3. **Embedding training**: train one word2vec model per time slice with shared hyperparameters.
4. **Analysis**: compute gender-typing and prestige scores for a predefined list of occupations across decades.

The goal of this document is to specify **requirements and interfaces** so that multiple coders can implement the project consistently.

---

## 1. Repository Structure

Suggested high-level layout:

```text
.
├── README.md                     # High-level description & usage
├── config/
│   └── config.yml                # Master configuration file
├── scripts/
│   ├── download_ngrams.py        # Download & decompress Chinese 5-gram data
│   ├── build_corpora.py          # Build time-sliced corpora from raw 5-grams
│   ├── train_embeddings.py       # Train one embedding per time slice
│   └── analyze_embeddings.py     # Compute gender & prestige measures
├── wordlists/
│   ├── occupations_zh.txt        # List of Chinese occupation titles
│   ├── gender_words_zh.json      # Gender word pairs / sets
│   └── prestige_axes_zh.json     # Prestige-related antonym pairs
├── data/
│   ├── raw_ngrams/               # Downloaded .gz files (and/or decompressed raw)
│   ├── corpora/                  # Time-sliced training corpora (text)
│   ├── models/                   # Trained embedding models (one per slice)
│   └── results/                  # CSVs & derived metrics
├── notebooks/
│   └── exploration.ipynb         # Optional exploratory analysis / sanity checks
└── tests/
    └── ...                       # Unit tests for each module

Coders do not have to follow this exactly, but any deviations should be documented in README.md and config.yml.

⸻

2. Configuration (config/config.yml)

Use a single YAML file to centralize settings. It should be the only place that hard-codes paths and hyperparameters.

2.1 Paths

paths:
  base_dir: "/path/to/project"
  raw_ngram_dir: "/path/to/project/data/raw_ngrams"
  decompressed_dir: "/path/to/project/data/raw_ngrams_decompressed"
  corpora_dir: "/path/to/project/data/corpora"
  models_dir: "/path/to/project/data/models"
  results_dir: "/path/to/project/data/results"
  log_dir: "/path/to/project/logs"

2.2 Google Ngram settings

These are specific to Chinese simplified 5-grams as hosted by Google.

ngram:
  language: "chi_sim"
  n: 5
  index_url: "https://storage.googleapis.com/books/ngrams/books/20200217/chi_sim/chi_sim-5-ngrams_exports.html"
  # File name pattern can be used to validate/parsing the index
  file_pattern: "googlebooks-chi-sim-all-5gram-*.gz"
  min_year: 1940
  max_year: 2015

The parser must be robust to the exact 5-gram file format (columns, separators, etc.) and should allow easy adjustment if Google’s schema differs slightly from prior documentation.

2.3 Time slicing configuration

We want 10-year windows with 5-year step, covering 1940–2015:

time_slices:
  window_size: 10        # years per corpus (inclusive)
  step_size: 5           # years between the starts of two slices
  start_year: 1940
  end_year: 2015         # inclusive; last window may be truncated if needed
  # Derived slices might look like: [1940–1949], [1945–1954], ..., [2005–2014]

The code should compute the actual [start, end] pairs from these parameters rather than hard-coding them.

2.4 Word2Vec / embedding settings

Mirror Jiang’s general approach: skip-gram, ~300 dimensions, with frequency threshold.  ￼

embedding:
  vector_size: 300
  window: 4               # or 5; must be consistent across time slices
  min_count: 50           # discard tokens with total freq < 50 in that slice
  sg: 1                   # 1 = skip-gram, 0 = CBOW
  negative: 15            # negative sampling size
  workers: 16             # number of CPU cores to use
  epochs: 5               # passes over corpus; configurable
  seed: 42                # reproducibility
  # Implementation details (e.g., "gensim") can be fixed or configurable:
  implementation: "gensim"

Note: all scripts must ensure UTF-8 and full support for Chinese characters.

2.5 Word lists (gender, occupations, prestige)

The config should point to external word list files but also optionally embed some default lists.

wordlists:
  occupations_file: "wordlists/occupations_zh.txt"
  gender_words_file: "wordlists/gender_words_zh.json"
  prestige_axes_file: "wordlists/prestige_axes_zh.json"

Example content formats:

wordlists/occupations_zh.txt
One occupation per line, UTF-8, no extra columns. Multi-character titles are allowed (e.g. “工程师”, “护士”, “程序员”).

wordlists/gender_words_zh.json

{
  "male_terms": ["男", "男人", "男性", "他", "父亲", "丈夫"],
  "female_terms": ["女", "女人", "女性", "她", "母亲", "妻子"],
  "antonym_pairs": [
    ["男", "女"],
    ["先生", "女士"]
  ]
}

These will be used to define a gender axis (e.g., centroid difference between male and female terms) similar to Jiang’s construction of a gender subspace.  ￼

wordlists/prestige_axes_zh.json
Use separate axes for the four symbolic dimensions (general prestige + three affective components) as in Jiang: evaluation (moral), potency (power/competence), activity (liveliness), plus residual/general prestige.  ￼

Example:

{
  "evaluation": {
    "positive": ["高尚", "正直", "体面", "有道德"],
    "negative": ["卑鄙", "不道德", "可耻"]
  },
  "potency": {
    "positive": ["强大", "有权力", "有影响力", "重要"],
    "negative": ["弱小", "无力", "边缘"]
  },
  "activity": {
    "positive": ["活跃", "忙碌", "快速"],
    "negative": ["沉闷", "缓慢", "静止"]
  },
  "general_prestige": {
    "positive": ["高贵", "有地位", "体面"],
    "negative": ["低贱", "卑微", "下层"]
  }
}

Scripts should not assume any specific words; they must read these JSONs and construct axes generically in the same way for all dimensions.

⸻

3. Script 1 — download_ngrams.py

Purpose
	•	Parse the index page for Chinese simplified 5-gram files.
	•	Download all relevant .gz files to raw_ngram_dir.
	•	Optionally verify checksums if available.
	•	Decompress each archive into decompressed_dir.

Requirements
	1.	CLI interface
Example usage:

python scripts/download_ngrams.py --config config/config.yml

Optional flags:
	•	--skip-decompress to only download.
	•	--max-workers to control parallel downloads.

	2.	Index parsing
	•	Fetch ngram.index_url (HTML page).
	•	Extract all download URLs matching file_pattern.
	•	Support retries on network failures.
	•	Log the list of URLs and file sizes.
	3.	Downloading
	•	Download each file into raw_ngram_dir.
	•	Support resumable downloads where possible (e.g. check existing partial files; at minimum, skip files that are already fully present).
	•	Provide progress logging (per file + summary).
	4.	Decompression
	•	For each .gz file, decompress into decompressed_dir with a clear naming convention:
	•	Input: googlebooks-chi-sim-all-5gram-20120701-0.gz
	•	Output: googlebooks-chi-sim-all-5gram-20120701-0.txt
	•	Use streaming decompression to avoid loading the whole file into memory.
	•	Respect --skip-decompress.
	5.	Logging & errors
	•	All actions should be logged to log_dir/download_ngrams.log.
	•	Fail fast on unrecoverable errors, but continue past individual file failures when possible.

⸻

4. Script 2 — build_corpora.py

Purpose
	•	Convert decompressed Chinese 5-gram files into time-sliced corpora for embedding training.
	•	Each corpus corresponds to one time slice (e.g., 1940–1949, 1945–1954, …).
	•	Each line in the output corpus is a “sentence” of five Chinese tokens.

Input assumptions

Each decompressed 5-gram line will have fields like:

token1 token2 token3 token4 token5  year  match_count  volume_count

The exact schema should be configurable (e.g. via config: which column is year, which is match_count).

Key design decisions
	•	Treat each character token (or Google’s tokenization unit) as one token in word2vec. No additional segmentation is required; we mirror Google’s units for training and later reconstruct multi-character occupation words when analyzing.
	•	Use match_count as a frequency; the simplest implementation can either:
	•	Option A: Write each 5-gram only once per line (ignoring counts).
	•	Option B (preferred, but more expensive): Up-weight by counts (e.g., write multiple times or use sampling).

This choice should be controlled by config, e.g.:

corpus:
  use_counts: false

Requirements
	1.	CLI interface

python scripts/build_corpora.py --config config/config.yml

Optional flags:
	•	--slice 1940-1949 to build a single slice.
	•	--overwrite to regenerate existing corpora.

	2.	Time slice generation
	•	Use time_slices settings from config to compute all [start_year, end_year] windows.
	•	For each slice, create a subdirectory in corpora_dir, e.g.:

data/corpora/1940_1949/corpus.txt
data/corpora/1945_1954/corpus.txt


	3.	Filtering by year
	•	For each decompressed file:
	•	Read line by line.
	•	Parse tokens, year, and counts.
	•	For each time slice, include the line if the year falls within [start_year, end_year].
	4.	Writing corpora
	•	For each slice, write lines like:

字1 字2 字3 字4 字5


	•	No additional punctuation or markers by default.
	•	Ensure that output is valid UTF-8.
	•	Allow optional subsampling / down-scaling for extremely frequent n-grams (future extension; can be controlled via config).

	5.	Performance
	•	Files are large; code should be streaming-based and may use simple multi-processing (e.g. per-file workers) if needed.
	•	Memory usage must be bounded.
	6.	Logging
	•	Log number of lines processed and included per slice.
	•	Log total tokens per slice (approximate corpus size).

⸻

5. Script 3 — train_embeddings.py

Purpose
	•	Train one word2vec / SGNS model per time slice using the corpora generated above.
	•	Save models and relevant metadata for later analysis.

Requirements
	1.	CLI interface

python scripts/train_embeddings.py --config config/config.yml

Optional flags:
	•	--slice 1940_1949 to train only a single slice.
	•	--retrain to overwrite models.

	2.	Model training
	•	For each time slice directory in corpora_dir:
	•	Load corpus.txt as an iterator of tokenized sentences.
	•	Train a word2vec (skip-gram, negative sampling) model using parameters from embedding section of config:
	•	vector_size
	•	window
	•	min_count
	•	sg
	•	negative
	•	workers
	•	epochs
	•	seed
	•	Normalize resulting vectors to unit length after training (or rely on library defaults and ensure later code handles normalization consistently).
	3.	Model saving
	•	Save each model in models_dir under a consistent naming convention:

data/models/chi_sim_5gram_1940_1949.model
data/models/chi_sim_5gram_1945_1954.model
...


	•	Additionally save a small JSON metadata file per model containing:

{
  "time_slice": "1940_1949",
  "start_year": 1940,
  "end_year": 1949,
  "vector_size": 300,
  "window": 4,
  "min_count": 50,
  "num_tokens": 123456789,
  "vocab_size": 45678
}


	4.	Determinism
	•	Set random seeds and disable multi-thread nondeterminism as much as feasible (depending on library capabilities).
	•	Document any sources of non-determinism.
	5.	(Optional, future) Alignment
	•	The initial version can treat each decade model independently.
	•	A later extension can add Procrustes alignment across decades, but this is not required for the initial implementation.

⸻

6. Script 4 — analyze_embeddings.py

Purpose
	•	Using the trained embeddings and configured word lists, compute:
	•	Gender typing of occupations over time.
	•	Prestige dimensions of occupations over time (evaluation, potency, activity, general prestige).
	•	Relationships between gender typing and prestige, and differences between male-typed vs. female-typed occupations.

This mirrors Jiang’s approach: occupations are mapped into a semantic space and their positions relative to gender- and prestige-signaling phrases summarize cultural schemata.  ￼

Requirements
	1.	CLI interface

python scripts/analyze_embeddings.py --config config/config.yml

Optional flags:
	•	--slice 1940_1949 to analyze a single time slice.
	•	--export-plots to generate simple diagnostic visualizations.

	2.	Loading models & word lists
	•	For each model in models_dir:
	•	Load the word2vec model.
	•	Load occupation list from occupations_zh.txt.
	•	Load gender words / antonym pairs from gender_words_zh.json.
	•	Load prestige axes from prestige_axes_zh.json.
	3.	Handling multi-character occupation titles
	•	Occupation titles are multi-character (e.g. “工程师”).
	•	The embeddings are trained at the unit level used in the 5-grams (likely single characters or Google’s tokenization).
	•	There are two viable strategies (repo should pick one and document it):
	•	Strategy A: Use whole-token vectors
If occupation titles appear as single tokens in the vocabulary, simply use model[word].
	•	Strategy B: Average over characters
If not, represent each occupation as the average of embeddings of its characters:

v_occ = mean(model[char] for char in "工程师" if char in vocab)


	•	The implementation must:
	•	Check coverage for each occupation (how many characters are in the vocabulary).
	•	Optionally drop occupations with insufficient coverage (configurable threshold).

	4.	Constructing semantic axes
For each dimension (gender, evaluation, potency, activity, general prestige):
	•	Load the positive and negative term lists from JSON.
	•	For each model:

v_pos = mean(embedding[w] for w in positive_terms if w in vocab)
v_neg = mean(embedding[w] for w in negative_terms if w in vocab)
axis = normalize(v_pos - v_neg)


	•	Gender typing of occupation occ is then:
	•	gender_score = cosine(embedding[occ], axis), or equivalently dot product if vectors are normalized.

	5.	Outputs
At minimum, generate the following CSVs in results_dir:
	•	occupation_gender_typing_by_decade.csv
Columns:
	•	occupation
	•	time_slice
	•	start_year
	•	end_year
	•	gender_score (projection on gender axis)
	•	coverage (e.g., fraction of characters or tokens present)
	•	occupation_prestige_by_decade.csv
Columns:
	•	occupation
	•	time_slice
	•	prestige_evaluation
	•	prestige_potency
	•	prestige_activity
	•	prestige_general
	•	coverage
	•	occupation_gender_prestige_joint.csv (optional merge of the above two).
	6.	Derived analyses
The script should also compute some basic summary statistics and write them to separate CSVs or JSON:
	•	Correlation between gender typing and each prestige dimension within each time slice.
	•	Average prestige trajectories for:
	•	female-typed vs. male-typed occupations;
	•	specific subsets (e.g., “female-typed occupations in middle prestige band”).
High-level regression modeling (e.g., fixed-effects estimation) is out of scope for this repository and can be done separately in R/Stata/Python using the generated tables.
	7.	(Optional) Simple plots
If --export-plots is set, generate basic visualizations (e.g., using matplotlib):
	•	Time series of mean gender typing for all occupations.
	•	Scatterplots of gender typing vs. general prestige for selected decades.
Plots are saved into results_dir/plots/.

⸻

7. Testing & Validation

Coders should add minimal tests in tests/:
	1.	Unit tests
	•	test_download_ngrams.py: mock the index page and test URL extraction logic.
	•	test_build_corpora.py: given a tiny synthetic 5-gram file, verify:
	•	Year filtering,
	•	Sentence formatting,
	•	Slice creation.
	•	test_train_embeddings.py: train on a tiny corpus and verify:
	•	Model files are created,
	•	Vocab size > 0.
	•	test_analyze_embeddings.py: using a small fake model & wordlists, verify:
	•	Axes construction,
	•	Occupation projections,
	•	Output files format.
	2.	Integration test
	•	A script or test that runs the full pipeline on a toy dataset (e.g., a reduced 5-gram file with only a few lines) to check that all stages connect.

⸻

8. Logging & Reproducibility
	•	All scripts should use a consistent logging framework (e.g., Python logging) with log files under log_dir.
	•	Critical configuration (config.yml) must be archived with each run (e.g., copy it into results_dir/run_YYYYMMDDHHMM/).
	•	Training script should log:
	•	Effective corpus size,
	•	Vocabulary size,
	•	Hyperparameters actually used.

⸻

9. Extensibility (Future-proofing)

The implementation should be modular so that future extensions are easy:
	•	Replace or add additional corpora (e.g., COHA/COCA equivalent for Chinese).
	•	Implement cross-decade alignment of embeddings.
	•	Add R / Stata scripts under analysis/ for panel models (e.g., fixed effects or mediation analysis linking cultural bias to wage data, analogous to Jiang’s work).  ￼

⸻

10. Summary for Coders

In short, coders should:
	1.	Implement four main scripts as specified.
	2.	Use config.yml as the single source of truth for paths and hyperparameters.
	3.	Ensure UTF-8 and Chinese compatibility throughout.
	4.	Generate clean, well-documented intermediate outputs (corpora, models, CSVs) to support downstream statistical analysis.
