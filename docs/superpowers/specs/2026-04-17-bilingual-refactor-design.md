# Bilingual Refactor: Chinese-Only → Chinese + English Pipeline

**Date:** 2026-04-17
**Status:** Design approved; pending spec review before implementation planning
**Scope:** Refactor the existing Chinese-only gender-norms text-embedding pipeline into a bilingual (Chinese + English) pipeline. Add Google Ngram English and COHA (free n-gram subset) as the first two English data sources. Leave room for more English sources later.

---

## 1. Goals and non-goals

### Goals

- Introduce `language: zh | en` as a first-class, required top-level config key.
- Add two new English data sources (`data_source: ngram` with `language: en`, and `data_source: coha`). Both run as longitudinal analyses in MVP.
- Keep training, analysis (prestige + WEAT), and the math in `embedding_utils.py` **unchanged**; they are already language-agnostic.
- Factor per-source text-preprocessing duplication into a single shared module with a tokenizer registry, a stopword registry, and one unified `clean_text` / `preprocess` entry point.
- Seed curated English wordlists from published sources (Caliskan 2017, Bolukbasi 2016, O*NET, Osgood 1957 / Nakao-Treas 1994).
- Migrate the 7 existing Chinese profiles to explicitly set `language: zh` (breaking).
- Never break the Chinese pipeline at any step of the migration.

### Non-goals (Phase 1)

- State-level COHA analysis. Requires full-text COHA (paid). Deferred to Phase 2.
- A US survey equivalent to CFPS/CGSS for correlation. Deferred.
- Legacy script cleanup (`scripts/build_corpora.py`, `scripts/build_corpora_renminribao.py`, top-level `scripts/download_ngrams.py`).
- Renaming `provincial/` to something language-neutral.
- Full language-plugin architecture (rejected as premature at current scope).

---

## 2. Architectural decisions (locked)

| Decision | Choice | Rationale |
|---|---|---|
| Language in config | Top-level `language: zh \| en` key, independent of `data_source` | Clean separation; ngram exists in both languages with same machinery |
| MVP English sources | Google Ngram English + COHA free n-grams | Both freely downloadable; both longitudinal; minimal new infra |
| COHA analysis unit (Phase 1) | Longitudinal (decade-level) | Free COHA 4-grams have no source/state metadata |
| COHA analysis unit (Phase 2) | State-level via source→state mapping | Requires paid full-text COHA |
| Wordlist layout | `wordlists/{zh,en}/{prestige,weat_formal,weat_informal}/*.json` | Config `language` naturally picks subdir; drop `_zh` suffix from files |
| English wordlists | Seed from Caliskan 2017 / Bolukbasi 2016 / O*NET / Osgood 1957 | Well-cited, reproducible; user will review/tune |
| English tokenizer (registry) | Whitespace (Phase 1 MVP; both English sources are pre-tokenized n-grams); NLTK `word_tokenize` registered and tested, reserved for Phase 2 full-text COHA | Matches the lightness of the current jieba-only stack; NLTK is the standard choice in sociolinguistics when real word-splitting is needed |
| Backward compatibility | Breaking: all configs must declare `language` | 7 profiles, trivial edit each; avoids silent-default mistakes |
| Code organization | Language-aware config + shared `preprocessing.py` module; flat per-source builders | Matches user's mandate of "share what can be shared; split parsers; config drives dispatch" |

---

## 3. Config schema changes

### 3.1 New / changed keys

```yaml
# NEW — required; no default. Loader rejects configs that omit it.
language: "zh"                 # "zh" | "en"

data_source: "ngram"           # Extended enum (see §3.2)

ngram:
  language: "chi_sim"           # Extended: "chi_sim" | "eng". Used only for download URL.
  n: 5                          # Only 5 supported (Google Ngram)
  min_year: 1940
  max_year: 2015
  delimiter: "\t"
  year_column: 1
  match_count_column: 2
  volume_count_column: 3

# NEW — required when data_source == "coha"
coha:
  ngram_order: 4                # 1, 2, 3, or 4. MVP uses 4.
  source_archive_urls: [ ... ]  # URLs from corpusdata.org email-gated signup
  decade_min: 1810
  decade_max: 2000

corpus:
  tokenizer: "whitespace"        # Extended: "whitespace" | "jieba" | "nltk_en"
  stopwords: "en_default"        # NEW optional. Defaults from (language, data_source).
  lowercase: true                # NEW optional. Default true for en, false for zh.
  use_counts: false
  min_count_threshold: 1
  min_words: 5

paths:
  base_dir: ...
  raw_ngram_dir: ...
  decompressed_dir: ...
  raw_coha_dir: "data/raw_coha"                     # NEW, required when data_source == "coha"
  coha_decompressed_dir: "data/raw_coha_decompressed"  # NEW
  raw_data_dir: ...
  corpora_dir: ...
  models_dir: ...
  results_dir: ...
  log_dir: ...
  figures_dir: ...

# NEW optional. Defaults to the existing hardcoded path. zh only.
fonts:
  cjk_path: "/usr/share/fonts/google-droid/DroidSansFallback.ttf"

wordlists:
  dir: "wordlists/en/weat_formal"   # Now under language subdir
  # Filename keys unchanged:
  occupations_file: "occupations.txt"      # Note: `_zh` suffix dropped
  gender_words_file: "gender_words.json"
  prestige_axes_file: "prestige_axes.json"
  weat_gender_file: "gender_words.json"
  weat_domestic_work_file: "domestic_work_words.json"
  weat_leadership_file: "leadership_words.json"
  weat_stem_file: "stem_words.json"
```

### 3.2 `(data_source, language)` compatibility matrix

```python
DATA_SOURCE_LANGUAGE_COMPAT = {
    "ngram":       {"zh", "en"},
    "renminribao": {"zh"},
    "weibo":       {"zh"},
    "newspaper":   {"zh"},
    "coha":        {"en"},
}
```

Enforced in `_validate_config()` with a clear error message.

### 3.3 `config_loader.py` changes

- `VALID_LANGUAGES = {"zh", "en"}`
- `_validate_config()` rejects missing `language` and incompatible `(data_source, language)` pairs.
- `_set_defaults()` uses the `(language, data_source) → defaults` table below.
- `get_wordlist_dir()` resolves to `wordlists/{language}/{mode}/...`. No legacy fallback; we migrate in lockstep (see §7).

**Default table:**

| language | data_source | tokenizer     | stopwords      | lowercase |
|----------|-------------|---------------|----------------|-----------|
| zh       | ngram       | `whitespace`  | *(none)*       | `false`   |
| zh       | renminribao | `jieba`       | `zh_default`   | `false`   |
| zh       | weibo       | `jieba`       | `zh_weibo`     | `false`   |
| zh       | newspaper   | `jieba`       | `zh_newspaper` | `false`   |
| en       | ngram       | `whitespace`  | *(none)*       | `true`    |
| en       | coha        | `whitespace`  | *(none)*       | `true`    |

Notes: ngram-style sources skip stopword filtering at corpus-build time (`min_count` during training does the same work, matching existing Chinese behavior). When `corpus.stopwords` is explicitly set in a config, that value overrides the default.

### 3.4 Migration of existing 7 profiles

Every existing profile gets:
- `language: zh` added at the top.
- `wordlists.dir` updated: `wordlists/weat_formal` → `wordlists/zh/weat_formal` (etc.).
- Existing `occupations_zh.txt` etc. referenced by new names (see §5).

One-line edits × 7 profiles.

---

## 4. Shared preprocessing module

New file: **`scripts/common/preprocessing.py`** (~250 lines).

### 4.1 Public API

```python
# Tokenizer registry (lazy imports)
def tokenize_whitespace(text: str) -> list[str]: ...
def tokenize_jieba(text: str) -> list[str]: ...
def tokenize_nltk_en(text: str) -> list[str]: ...
TOKENIZERS: dict[str, Callable] = {"whitespace": ..., "jieba": ..., "nltk_en": ...}

# Stopword registry
STOPWORDS: dict[str, frozenset[str]] = {
    "zh_default":   frozenset({...}),   # 27-word set (from current rmrb builder)
    "zh_weibo":     frozenset({...}),   # zh_default ∪ social media slang
    "zh_newspaper": frozenset({...}),   # zh_default ∪ journalism terms
    "en_default":   frozenset(...),     # from nltk.corpus.stopwords.words("english")
    "en_newspaper": frozenset({...}),   # en_default ∪ journalism terms (reserved for Phase 2)
}

# Character-keep regex by language
KEEP_PATTERNS: dict[str, re.Pattern] = {
    "zh": re.compile(r"[\u4e00-\u9fff]+"),
    "en": re.compile(r"[a-z']+"),        # applied post-lowercase
}

def clean_text(
    text: str,
    language: str,
    *,
    lowercase: bool = False,
    strip_urls: bool = True,
    strip_mentions: bool = False,    # weibo-style @
    strip_parens: bool = False,      # newspaper-style (...)
    strip_zero_width: bool = True,
) -> str: ...

def preprocess(
    text: str,
    *,
    language: str,
    tokenizer: str,                  # key into TOKENIZERS
    stopwords_key: str,              # key into STOPWORDS
    lowercase: bool,
    min_words: int,
    cleaner_opts: dict | None = None,
) -> list[str] | None:               # None ⇒ filter this document out
    ...
```

### 4.2 What it replaces

Inline definitions of `clean_text()`, `segment_text()`, and `STOPWORDS` in:
- `scripts/data_prep/build_corpora_rmrb.py`
- `scripts/data_prep/build_corpora_weibo.py`
- `scripts/data_prep/build_corpora_newspaper.py`

Each builder shrinks by ~80 lines. Per-source differences (weibo mentions, newspaper parens, rmrb `gb18030` encoding) become either flags on `preprocess()` or stay local (encoding is a file-read concern, not a text concern).

Ngram builders (Chinese and English) don't need `preprocess()` — Google already pre-tokenized. They use `clean_text()` + `KEEP_PATTERNS[language]`.

### 4.3 Lazy imports

`jieba` imported only when `tokenize_jieba` is called; `nltk` similarly. Importing `preprocessing.py` on a Chinese-only machine never requires NLTK, and vice versa.

### 4.4 Phase 1 tokenizer usage

Both MVP English sources (Google Ngram EN, COHA free n-grams) are pre-tokenized, so `tokenize_nltk_en` is registered and unit-tested but not called by any Phase 1 builder. It becomes live in Phase 2 when the full-text COHA parser runs.

---

## 5. Data pipeline changes

### 5.1 `scripts/data_prep/download_ngrams.py` — parameterize by language

- Base URL built from `ngram.language`:
  - `chi_sim` → `http://storage.googleapis.com/books/ngrams/books/20200217/chi_sim/`
  - `eng` → `http://storage.googleapis.com/books/ngrams/books/20200217/eng/`
- Shard count scraped from the index page (English has different count than Chinese).
- Rest unchanged.

### 5.2 `scripts/data_prep/build_corpora_ngram_en.py` — new

~150 lines, structural near-copy of `build_corpora_ngram.py`. Differences:
- `clean_text(..., language="en", lowercase=True)`.
- `KEEP_PATTERNS["en"]` retains `[a-z']+` tokens.
- Tokenizer is `whitespace` (Google pre-tokenized).
- No explicit stopword step — `min_count` at training handles it (same as Chinese ngram today).

Not a refactor of the Chinese version. Character class and tokenization policy differ; two readable files beat one branchy file.

### 5.3 `scripts/data_prep/download_coha.py` — new

~120 lines, mirrors `download_ngrams.py`:
- Reads `coha.source_archive_urls` from config (user pastes URLs from the email-gated corpusdata.org signup).
- Downloads in parallel; thread pool size configurable.
- Decompresses ZIPs into `paths.coha_decompressed_dir`.
- Logs to `log_dir/download_coha.log`.

### 5.4 `scripts/data_prep/build_corpora_coha.py` — new

~200 lines. Parses COHA n-gram TSVs (format: `word1<TAB>word2<TAB>...<TAB>freq`). For each ngram:
- Respects COHA's pre-existing decade bucketing (files are by decade).
- Treats each n-gram as a mini-document (same trick the Chinese ngram builder uses with 5-grams).
- Applies `clean_text(..., language="en", lowercase=True)` per token; drops n-grams where fewer than 2 tokens survive cleaning.
- Writes `corpora_dir/{decade}/corpus_{shard_idx}.txt` — exact same layout as Chinese ngram, so `train_embeddings.py` needs zero changes.

Phase 1 COHA ignores `time_slices` config (decades are fixed in the source).

### 5.5 Phase 2 stubs (documented, not built)

- `scripts/data_prep/source_state_mapper.py` — will mirror `province_mapper.py` for COHA publication → US state.
- `build_corpora_coha_fulltext.py` — parses paid full-text COHA, outputs state-partitioned corpora.

### 5.6 `run_pipeline.sh` dispatch

```
Stage 1 (download):
    data_source == ngram → download_ngrams.py (language drives URL)
    data_source == coha  → download_coha.py
    other                → skip

Stage 2 (corpus):
    data_source == ngram       → build_corpora_ngram.py OR build_corpora_ngram_en.py (on language)
    data_source == coha        → build_corpora_coha.py
    data_source == renminribao → build_corpora_rmrb.py
    data_source == weibo       → build_corpora_weibo.py
    data_source == newspaper   → build_corpora_newspaper.py
```

Stages 3–6 (train, analyze, visualize, correlate) are language-agnostic at the dispatch level.

---

## 6. Wordlists

### 6.1 Directory reorganization

```
wordlists/
├── zh/
│   ├── prestige/
│   │   ├── occupations.txt          # was occupations_zh.txt
│   │   ├── gender_words.json        # was gender_words_zh.json
│   │   ├── prestige_axes.json       # was prestige_axes_zh.json
│   │   ├── occup_category.json
│   │   └── occup_category_zh.json
│   ├── weat_formal/
│   │   ├── gender_words.json
│   │   ├── domestic_work_words.json
│   │   ├── leadership_words.json
│   │   └── stem_words.json
│   └── weat_informal/ (same 4 files)
└── en/
    ├── prestige/
    │   ├── occupations.txt           # from O*NET 2020 (~200 terms)
    │   ├── gender_words.json         # from Bolukbasi 2016 gender pairs
    │   └── prestige_axes.json        # Osgood 1957 EPA + Nakao-Treas 1994 prestige
    └── weat_formal/
        ├── gender_words.json         # Caliskan 2017 gender stimuli
        ├── domestic_work_words.json  # Caliskan 2017 career/family stimuli
        ├── leadership_words.json     # Garg 2018 replication list
        └── stem_words.json           # Caliskan 2017 STEM/arts stimuli
```

### 6.2 Cleanup

- `wordlists/test.py` (stray file) — deleted.
- `wordlists/gender_words_zh_backup.json` — moved into `wordlists/zh/prestige/` for consistency.
- File renames done via `git mv` to preserve history.

### 6.3 Phase 1 omits

- `wordlists/en/weat_informal/` — no English analog to Weibo colloquial register in MVP sources. Directory not created.

---

## 7. Visualization and correlation changes

### 7.1 `scripts/visualize.py`

- `_configure_fonts(language)` extracted from the current import-time side effect. Called once at the top of the entry point, after config load:
  - `language == "zh"` → existing CJK font registration (now reads path from `config["fonts"]["cjk_path"]` if present, else current hardcode).
  - `language == "en"` → no-op.
- Label lookup: `LABELS = { "zh": {...}, "en": {...} }` — ~20 keys for axis labels, titles, legends. All user-facing strings pulled from `LABELS[config["language"]][key]`.
- Survey-comparison plots (prestige-mode, merges CFPS/CGSS) guarded with `if config["language"] == "zh"`: single early-return, cleanly skipped for English.
- Choropleth plots (`plot_weat_choropleth_*`) unchanged for MVP — already guarded by `analysis_unit == "provincial"`, which is never true for MVP English sources. Phase 2 adds a language branch for US states shapefile.

### 7.2 `scripts/analyze_correlation.py`

- Same font-block extraction as visualize.
- `PROVINCE_NAME_MAPPING` (short↔long Chinese names) moves inside a `zh`-only branch.
- For English, merges on `year` column only (MVP has no English provincial analysis).

### 7.3 Lines touched

- `visualize.py`: ~60 lines refactored; no function deletions.
- `analyze_correlation.py`: ~30 lines refactored.

---

## 8. Slurm and orchestration

### 8.1 New templates

- `slurm/download_ngrams_en.slurm` — thin wrapper: `python -m scripts.data_prep.download_ngrams --config=$CONFIG` with an EN profile.
- `slurm/download_coha.slurm`
- `slurm/build_corpus_coha.slurm`
- `slurm/full_pipeline_en.slurm` — wraps `run_pipeline.sh` with an EN config.

### 8.2 No changes

- `slurm/analyze_adroit.slurm`, `slurm/visualize.slurm`, `slurm/train.slurm`, etc. already config-driven; a new EN profile path is all they need.
- Existing province-group train scripts (`train_prov_group*.slurm`) are Chinese-weibo specific and unused for EN MVP.

---

## 9. Dependencies

Add to `requirements.txt`:
```
nltk>=3.8
```

Add to `setup_server.sh`:
```bash
python -m nltk.downloader -d "${NLTK_DATA:-$HOME/nltk_data}" punkt stopwords
```

No new heavy dependencies. No spaCy. No new shapefile dependencies in Phase 1.

---

## 10. Testing

### 10.1 New test files

- **`tests/test_preprocessing.py`** (~80 lines) — round-trip each tokenizer (whitespace / jieba / nltk_en) and each stopword set on small fixtures.
- **`tests/test_config_loader.py`** (~100 lines) — full `(language, data_source)` validation matrix; wordlist dir resolution; error paths for missing / invalid `language`.

### 10.2 Extended tests

- `tests/test_build_corpora.py` — keep existing Chinese fixtures; add one English Ngram fixture, one COHA fixture (tiny hand-crafted input).
- `tests/test_analyze_embeddings.py` — add one English wordlist fixture to confirm math functions work on English tokens.

### 10.3 Smoke tests

One per language: `run_pipeline.sh` on a 10-document fixture corpus, verifying exit codes and output file existence. No full-pipeline integration test (too expensive; fixture smoke catches the cheap failures).

### 10.4 Goal

Every refactored Chinese path keeps passing its existing test; every new English path has at least one sanity check.

---

## 11. Migration plan (ordered commits)

Each commit keeps the Chinese pipeline runnable.

1. Add `language` validation stubs to `config_loader.py`; add empty `scripts/common/preprocessing.py`. Existing tests pass.
2. Populate `preprocessing.py` with tokenizer + stopword registries (Chinese and English). New tests in `test_preprocessing.py` pass. No callers yet.
3. `git mv` wordlists into `wordlists/zh/...`; update all 7 profiles to add `language: zh` and new `wordlists.dir`. Drop `_zh` suffixes. Chinese pipeline tests pass.
4. Refactor 3 Chinese text builders to call `preprocessing.preprocess()`. Tests pass.
5. Refactor `visualize.py` + `analyze_correlation.py` (font extraction, label dict, language guards). Tests pass.
6. Parameterize `download_ngrams.py` by `ngram.language`. Chinese pipeline tests pass.
7. Add `wordlists/en/` seeded lists. Add `build_corpora_ngram_en.py` + `ngram_en_server.yml` + `ngram_en_weat.yml` profiles. Run EN pipeline on a tiny fixture.
8. Add `download_coha.py` + `build_corpora_coha.py` + `coha_server.yml` profile. Run COHA pipeline on a tiny fixture.
9. Add Slurm templates for EN + COHA.
10. README update: new section "English pipeline" with usage examples.

---

## 12. Phase 2 (documented, not built)

- **Full-text COHA:** `download_coha_fulltext.py`, `build_corpora_coha_fulltext.py`, `source_state_mapper.py`.
- **US state-level analysis:** `analysis_unit: provincial` for COHA; US states shapefile; state choropleth in `visualize.py`.
- **US survey correlation:** integration of GSS / ANES / similar.
- **Additional English sources:** each future source gets its own builder + profile + wordlist (if needed); shared preprocessing absorbs them.

---

## 13. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Refactoring Chinese builders regresses the working pipeline | High | Steps 1–5 preserve all existing tests; add a fixture-based smoke test before step 4 lands |
| COHA download URLs are email-gated | Low | `download_coha.py` reads URLs from config; no scraping |
| NLTK data missing on cluster | Medium | `setup_server.sh` installs it; tokenizer raises clear error at runtime if missing |
| Google Ngram English shard count/URL scheme drift | Low | Scrape the index page at runtime (same as current Chinese download) |
| English wordlists need tuning after first run | Medium | Seeded from published stimuli; user reviews and iterates as ordinary tuning work |

---

## 14. Open items for implementation planning

- COHA archive URLs — user pastes into `coha.source_archive_urls` at download time.
- English NLTK data install path on the Princeton cluster — handled in `setup_server.sh`.
- No other blockers.
