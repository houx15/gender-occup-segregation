# Dataset & Training Summary Reporter — Design Spec

**Date:** 2026-06-02
**Status:** Approved (pending user review of this file)
**Purpose:** Generate a methods-section-ready summary of one data source — corpus volume, vocabulary, raw-data footprint, and training hyperparameters — so we can quote concrete numbers when writing about Renminribao, provincial newspaper, Weibo, China Ngram, English Ngram, and COHA.

---

## Goal

A single command:

```
python -m scripts.describe_dataset --config=config/profiles/garg_weat_renminribao.yml
```

…produces `<results_dir>/dataset_summary.md` containing:

- Source totals (units, documents, tokens, raw vocab, model vocab)
- Raw-data footprint (source files, bytes)
- Training hyperparameters (read from the config profile)
- Per-unit breakdown table (one row per time slice / province-year / province)

The same command works against any of the 6 source profiles, with source-specific raw-data layouts handled by small adapter modules.

## Non-goals

- Not a corpus-quality report (no OOV diagnostics, no example sentences, no per-document length distributions).
- Not a model-quality report (no nearest-neighbor probes, no analogy benchmarks).
- Not a survey-side summary (CFPS / CGSS / 妇女地位调查 are out of scope here).
- Not a multi-source comparison view (one config in, one Markdown out — comparing across sources is a future composer that consumes these per-source files).

---

## Decisions table

| # | Decision | Choice | Why |
|---|---|---|---|
| 1 | Which stats? | docs + tokens (from corpus), vocab (corpus-raw + model-post-min_count), raw-data files + bytes, training hyperparameters | All four are useful for a methods section; corpus walk gives the first two cheaply; vocab needs both flavors so the reader sees how much `min_count` filters out. |
| 2 | Granularity | Summary + per-unit breakdown | The paragraph quotes totals; the appendix uses the breakdown. One scan produces both. |
| 3 | Output format | Markdown only | Direct paste into methods/appendix. No CSV/parquet — if we ever need machine-readable later, the sidecar JSON cache already holds the numbers. |
| 4 | Caching | Per-unit JSON sidecar | `<corpora_dir>/<unit>/.dataset_stats.json` keyed by per-file `(name, mtime, size)`. Re-runs are instant unless corpus changed. `--force` recomputes. |
| 5 | Source dispatch | Per-source raw-data walker plugins | One adapter per source in `scripts/data_prep/raw_volume/`, each with the same `walk(...)` signature; everything else (corpus scan, model vocab, render) is shared. |
| 6 | Source coverage | All 6: rmrb, newspaper, weibo, ngram_zh, ngram_en, coha | One adapter each; if you can run the source through `train_embeddings`, you can describe it here. |
| 7 | Cache location | Hidden sidecar in corpora_dir | Co-located with the data it describes; cleaned up automatically if corpora are deleted; doesn't clutter `ls` output. |

---

## Architecture

```
scripts/describe_dataset.py            — CLI shim (~80 lines): load config, orchestrate, write Markdown
scripts/common/dataset_stats.py        — shared logic: corpus scan, model vocab, sidecar cache, Markdown render
scripts/data_prep/raw_volume/
    __init__.py                        — registry: data_source → walker function
    rmrb.py                            — {decade}/{year}/报刊/人民日报/rmrb_YYYY_MM.txt
    provincial_newspaper.py            — {province_folder}/{year}/...
    weibo.py                           — *.parquet shards (per-province; reads parquet row counts)
    ngram_zh.py                        — Chinese Google Ngram .gz shards
    ngram_en.py                        — English Google Ngram .gz shards
    coha.py                            — COHA decade layout
tests/test_dataset_stats.py            — corpus scan, cache hit/miss, vocab, render
tests/test_raw_volume_<source>.py      — one fixture tree per source adapter
```

### File responsibilities

**`scripts/describe_dataset.py`** — CLI entry point. Fire-based, mirrors existing `scripts/analyze_*.py` shape. Loads the YAML profile via `scripts.common.config_loader.load_config`, calls `dataset_stats.run(config, ...)`, exits.

**`scripts/common/dataset_stats.py`** — does almost everything:

- `discover_units(config) -> List[str]` — walks `corpora_dir` for sub-directories matching `model_name_template`; returns sorted unit names. Reuses the same logic `train_embeddings` already uses (extract into a shared helper if not already shared).
- `scan_corpus_unit(unit_dir, logger, force=False) -> CorpusStats` — counts `n_docs`, `n_tokens`, `n_vocab_raw` over `corpus_*` files. Checks sidecar cache first. Returns dataclass.
- `model_vocab_size(model_path, logger) -> Optional[int]` — opens with gensim `KeyedVectors.load` (or `Word2Vec.load` then `.wv`), returns `len(index_to_key)`. Logs and returns `None` on missing/corrupt.
- `aggregate_source(per_unit_stats, raw_volume, config) -> SourceTotals` — sums docs/tokens/files/bytes; vocab totals are NOT summed (would double-count) — instead, "raw vocab total" is recomputed as one final scan-and-union pass over all units, or omitted if `--no-vocab-union` is set (since the union scan is the only reason an O(corpus) pass can't be fully avoided).
- `render_markdown(source_totals, per_unit_rows, config) -> str` — produces the Markdown shown below.
- `run(config, force=False, units=None, no_raw=False, no_model=False) -> None` — orchestrator that wires everything together and writes `<results_dir>/dataset_summary.md`.

**`scripts/data_prep/raw_volume/__init__.py`** — exports `WALKERS: Dict[str, Callable]` mapping `data_source` strings (`"renminribao"`, `"newspaper"`, …) to walker functions. Each walker has signature `walk(raw_data_dir: Path, units: List[str], logger) -> Dict[str, RawVolumeEntry]`.

**`scripts/data_prep/raw_volume/<source>.py`** — each adapter knows that source's directory layout:
- `rmrb.py`: walks `{decade}/{year}/报刊/人民日报/rmrb_YYYY_MM.txt`, groups by time-slice membership (one source file can belong to multiple overlapping slices — same logic as `build_corpora_rmrb.find_renminribao_files` + slice membership).
- `provincial_newspaper.py`: walks `{province_folder}/{year}/...`, groups by `{province}_{year}` unit using the `FOLDER_TO_PROVINCE` mapping from `build_corpora_provincial_newspaper.py`.
- `weibo.py`: walks `*.parquet`, opens each (pyarrow `dataset.scanner`) to get `row_count` per file, groups by province. Bytes are file size.
- `ngram_zh.py` / `ngram_en.py`: walks `.gz` shards, groups by year inferred from filename / path. `n_docs` is undefined for ngram (the corpus is already n-grams, not documents) — set to `null` and note this in the Markdown.
- `coha.py`: walks COHA's decade-organized text files. `n_docs` = file count or paragraph count (whichever matches how the corpus builder defines a document — see `build_corpora_coha.py`).

### Data flow per invocation

```
load_config(profile.yml)
  │
  ├─ discover_units(config)                     ← walk corpora_dir
  │
  ├─ for unit in units:
  │     scan_corpus_unit(unit_dir)              ← cache hit OR recompute
  │     model_vocab_size(model_path)            ← gensim load
  │
  ├─ raw_volume = WALKERS[data_source].walk(raw_data_dir, units)
  │
  ├─ aggregate_source(...)
  │
  └─ write <results_dir>/dataset_summary.md
```

The walker is called once per source (not once per unit) so each adapter can do one efficient pass.

---

## Schema

### `CorpusStats` (dataclass, per unit)

```python
@dataclass
class CorpusStats:
    unit_name: str
    n_docs: int               # lines across corpus_* files
    n_tokens: int             # sum of whitespace-split token counts
    n_vocab_raw: int          # |set(types)| in this unit, pre-min_count
    n_corpus_files: int       # how many corpus_* files
    scanned_at: str           # ISO timestamp
    from_cache: bool          # True if loaded from sidecar
```

### `RawVolumeEntry` (dataclass, per unit)

```python
@dataclass
class RawVolumeEntry:
    unit_name: str
    n_files: int              # source files (rmrb_*.txt / *.parquet / *.gz / ...)
    n_bytes: int              # sum of file sizes
    layout_hint: str          # one-line glob describing where these came from
    # n_docs may be set by walker if it's known cheaply (e.g., parquet row count)
    n_source_docs: Optional[int] = None
```

### Sidecar cache JSON schema

`<corpora_dir>/<unit>/.dataset_stats.json`:

```json
{
  "schema_version": 1,
  "n_docs": 12345,
  "n_tokens": 9876543,
  "n_vocab_raw": 54321,
  "scanned_at": "2026-06-02T14:30:00",
  "corpus_files": [
    {"name": "corpus_000000", "size": 12345678, "mtime": 1717000000.0}
  ]
}
```

Cache hit when every file's `(name, size, mtime)` matches exactly AND the set of `corpus_*` files in the directory matches the cached list. Anything else → miss → recompute → overwrite.

### Markdown output

```markdown
# Dataset Summary — renminribao (zh)

Generated 2026-06-02 from `config/profiles/garg_weat_renminribao.yml`.

## Corpus totals

| Units | Documents | Tokens | Raw vocab (union) | Model vocab (sum across units, min_count=50) |
|---|---|---|---|---|
| 15 | 12,345,678 | 1,234,567,890 | 9,876,543 | 6,481,635 |

## Raw data

- Layout: `{raw_data_dir}/{decade}/{year}/报刊/人民日报/rmrb_YYYY_MM.txt`
- Source files: 4,821
- Bytes: 5.2 GB

## Training

- Algorithm: Word2Vec skip-gram, negative sampling (gensim)
- `vector_size=300 · window=4 · min_count=50 · sg=1 · negative=15 · epochs=5 · seed=42 · workers=16`
- Model files: `<models_dir>/renminribao_{slice}.model` (one per time slice)

## Per-unit breakdown

| Unit | Documents | Tokens | Raw vocab | Model vocab | Raw files | Raw bytes |
|---|---|---|---|---|---|---|
| 1940_1949 | 521,032 | 41,235,891 | 583,201 | 312,440 | 240 | 254 MB |
| 1945_1954 | … | … | … | … | … | … |
| …
```

For provincial sources, rows are sorted by province then year (`北京_2020`, `北京_2021`, …, `天津_2020`, …) and the "Corpus totals" row says `N province-years (P provinces × Y years)`.

For ngram sources, the `Documents` column is renamed `N-gram entries` (in both the totals row and the per-unit table) and gets the same line-count value — each corpus line IS an n-gram. No `n/a` cells; the rename is the only difference.

The training-section sub-header reads `min_count=<config['embedding']['min_count']>` — pulled per source rather than hard-coded — so each source's actual threshold is shown.

---

## CLI

```
python -m scripts.describe_dataset --config=<profile.yml>            # default: cache, scan everything
python -m scripts.describe_dataset --config=<…> --force              # ignore cache, recompute
python -m scripts.describe_dataset --config=<…> --units=1940_1949,1950_1959
python -m scripts.describe_dataset --config=<…> --no-raw             # skip raw-data walk
python -m scripts.describe_dataset --config=<…> --no-model           # skip model vocab
python -m scripts.describe_dataset --config=<…> --no-vocab-union     # skip cross-unit vocab union (the only O(corpus) totalling pass)
```

Output always lands at `<config['paths']['results_dir']>/dataset_summary.md`. The `paths.results_dir` is already in every profile — no new config keys needed.

---

## Performance & failure modes

**Cache warm:** Each unit is one JSON read + one `KeyedVectors.load`. ~1 second per unit. 15 RMRB slices → ~15 s.

**Cache cold (first run):** Scan dominates. For RMRB longitudinal (~12M docs across slices), expect ~3–10 minutes on a login node depending on disk. Per-unit, so resumable: if killed mid-run, completed units stay cached.

**Cache cold + `--no-vocab-union`:** Per-unit scans only collect `n_vocab_raw` for that unit; the cross-unit union is skipped. Cache writes happen as soon as a unit is done — partial progress preserved.

**Source-file walker performance:** RMRB / provincial-newspaper walks are O(n_files) `stat()` calls — negligible. Weibo walker opens each parquet for row count — this is O(n_parquets) and could be slow if there are tens of thousands of shards; the walker logs progress per 100 files.

**Failure modes:**
- Missing model file → log warning, `model_vocab` cell is `n/a`, run continues.
- Missing `raw_data_dir` → log warning, raw-data section says "raw_data_dir not present on this host" (e.g., running describe on a server without the source data), per-unit raw cells are `n/a`. Useful when you want to re-render the methods table on a machine that only has corpora + models.
- Missing `corpora_dir` → hard error: there's nothing to describe.
- Cache schema mismatch (`schema_version`) → ignore cache, recompute.
- Corrupt cache JSON → log warning, recompute, overwrite.

---

## Testing plan

`tests/test_dataset_stats.py`:

1. **Corpus scan basic.** Write tmp `corpus_000000` with `["a b c\n", "a b\n"]`. Assert `n_docs=2, n_tokens=5, n_vocab_raw=3`.
2. **Multi-file scan.** Two corpus files. Assert sums and union vocab.
3. **Empty file.** Assert `n_docs=0, n_tokens=0, n_vocab_raw=0` (no crash).
4. **Cache miss → write.** First call writes sidecar. Assert file exists with expected keys.
5. **Cache hit → no rescan.** Second call returns cached `CorpusStats` with `from_cache=True`. Mutate cache to a sentinel value → second call returns the sentinel (proves the scan was skipped).
6. **Cache invalidation: mtime change.** Touch a corpus file → second call rescans.
7. **Cache invalidation: file added.** Add `corpus_000001` → second call rescans.
8. **Cache invalidation: schema_version mismatch.** Cache says `"schema_version": 99` → rescan.
9. **Corrupt cache.** Write malformed JSON → log + rescan, don't raise.
10. **`--force` recomputes.** Cache present → still rescans.
11. **Markdown render snapshot.** Build one minimal `SourceTotals` + 2 per-unit rows, assert exact Markdown string.
12. **Model vocab missing model file.** Returns `None`, doesn't raise.
13. **`render_markdown` ngram variant.** `data_source="ngram"` → both totals and per-unit `Documents` columns become `N-gram entries` and carry the same count values that `Documents` would have for other sources.

`tests/test_raw_volume_<source>.py` — one per adapter. Each:
- Fixture tree mimicking that source's layout (a handful of files in `tmp_path`).
- Call `walk(tmp_path, units, logger)`.
- Assert grouping is correct (each file goes to the right unit) and `n_files` / `n_bytes` match expectations.

Run all with `python -m pytest tests/test_dataset_stats.py tests/test_raw_volume_*.py`.

---

## Out of scope (future work)

- **Comparison composer.** A second script that reads multiple `dataset_summary.md` files (one per source) and emits a side-by-side comparison table for a single methods-section paragraph. Easy to add later; the per-source files are the building block.
- **CSV/Parquet export.** If a downstream consumer needs machine-readable numbers, add a flag — the cache already holds them.
- **OOV / coverage diagnostics.** Belongs in `oov_probe.py` (already exists).
- **Per-document length distribution.** Useful for some corpora-comparison papers; out of scope for a methods-section reporter.

---

## Acceptance criteria

A run finishes when:

1. `python -m scripts.describe_dataset --config=config/profiles/garg_weat_renminribao.yml` exits 0 on a host with corpora + models present.
2. `<results_dir>/dataset_summary.md` exists, contains the four sections (Corpus totals, Raw data, Training, Per-unit breakdown), and per-unit numbers add to the totals.
3. Re-running with no changes uses the cache — second invocation finishes in < 5 s for RMRB / Chinese ngram, < 30 s for provincial newspaper.
4. Running against any of the 6 source profiles produces a valid Markdown file with no exceptions.
5. All tests in `tests/test_dataset_stats.py` and `tests/test_raw_volume_*.py` pass.
