# Dataset & Training Summary Reporter — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Spec:** [`docs/superpowers/specs/2026-06-02-dataset-summary-design.md`](../specs/2026-06-02-dataset-summary-design.md)

**Goal:** A `python -m scripts.describe_dataset --config=<profile.yml>` command that emits `<results_dir>/dataset_summary.md` with corpus totals, raw-data footprint, training hyperparameters, and a per-unit breakdown, for any of 6 sources (rmrb / newspaper / weibo / ngram_zh / ngram_en / coha).

**Architecture:** One CLI shim (`scripts/describe_dataset.py`) drives a shared module (`scripts/common/dataset_stats.py`) plus six per-source raw-data walkers under `scripts/data_prep/raw_volume/`. Per-unit JSON sidecars in `corpora_dir/<unit>/.dataset_stats.json` cache scans so re-runs are near-instant.

**Tech Stack:** Python 3, gensim (KeyedVectors / Word2Vec), pyarrow (weibo parquet), Fire (CLI), pytest. No new dependencies.

**Spec correction folded in:** Cross-unit raw-vocab *union* total becomes opt-in via `--vocab-union` (default off), because computing it requires bypassing the cache. Default Corpus-totals row shows per-unit vocab as `min / mean / max` across units — which is what most methods sections actually quote.

---

## File structure

```
scripts/describe_dataset.py                    — CREATE: CLI shim (~80 lines)
scripts/common/dataset_stats.py                — CREATE: dataclasses, cache I/O, scan, render
scripts/data_prep/raw_volume/__init__.py       — CREATE: WALKERS registry
scripts/data_prep/raw_volume/rmrb.py           — CREATE
scripts/data_prep/raw_volume/provincial_newspaper.py — CREATE
scripts/data_prep/raw_volume/weibo.py          — CREATE
scripts/data_prep/raw_volume/ngram_zh.py       — CREATE
scripts/data_prep/raw_volume/ngram_en.py       — CREATE
scripts/data_prep/raw_volume/coha.py           — CREATE

tests/test_dataset_stats.py                    — CREATE
tests/test_raw_volume_rmrb.py                  — CREATE
tests/test_raw_volume_provincial_newspaper.py  — CREATE
tests/test_raw_volume_weibo.py                 — CREATE
tests/test_raw_volume_ngram.py                 — CREATE (covers zh + en)
tests/test_raw_volume_coha.py                  — CREATE
tests/test_describe_dataset.py                 — CREATE: end-to-end smoke
```

---

## Task 1: Dataclasses + sidecar cache I/O

**Files:**
- Create: `scripts/common/dataset_stats.py`
- Create: `tests/test_dataset_stats.py`

- [ ] **Step 1: Write the failing tests for cache I/O**

```python
# tests/test_dataset_stats.py
"""Tests for scripts.common.dataset_stats."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from scripts.common.dataset_stats import (
    CorpusStats, RawVolumeEntry, write_cache, read_cache, cache_is_fresh,
)

logger = logging.getLogger("test")


def _make_corpus_files(unit_dir: Path, contents: list[str]) -> list[Path]:
    """Write corpus_NNNNNN files; return their paths."""
    unit_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, body in enumerate(contents):
        p = unit_dir / f"corpus_{i:06d}"
        p.write_text(body, encoding="utf-8")
        paths.append(p)
    return paths


class TestCacheIO:
    def test_write_then_read_roundtrip(self, tmp_path):
        unit_dir = tmp_path / "1940_1949"
        files = _make_corpus_files(unit_dir, ["a b c\n", "a b\n"])
        stats = CorpusStats(
            unit_name="1940_1949", n_docs=2, n_tokens=5, n_vocab_raw=3,
            n_corpus_files=1, scanned_at="2026-06-02T00:00:00", from_cache=False,
        )
        write_cache(unit_dir, stats, files)
        loaded = read_cache(unit_dir)
        assert loaded is not None
        assert loaded.unit_name == "1940_1949"
        assert loaded.n_docs == 2
        assert loaded.n_tokens == 5
        assert loaded.n_vocab_raw == 3
        assert loaded.from_cache is True  # read_cache sets this

    def test_cache_missing_returns_none(self, tmp_path):
        unit_dir = tmp_path / "1940_1949"
        unit_dir.mkdir()
        assert read_cache(unit_dir) is None

    def test_cache_corrupt_returns_none(self, tmp_path):
        unit_dir = tmp_path / "1940_1949"
        unit_dir.mkdir()
        (unit_dir / ".dataset_stats.json").write_text("{not json", encoding="utf-8")
        assert read_cache(unit_dir) is None  # logged + ignored, not raised

    def test_cache_schema_version_mismatch_returns_none(self, tmp_path):
        unit_dir = tmp_path / "1940_1949"
        unit_dir.mkdir()
        (unit_dir / ".dataset_stats.json").write_text(
            json.dumps({"schema_version": 99, "n_docs": 1, "n_tokens": 1,
                        "n_vocab_raw": 1, "scanned_at": "x", "corpus_files": []}),
            encoding="utf-8",
        )
        assert read_cache(unit_dir) is None


class TestCacheFreshness:
    def test_fresh_when_files_unchanged(self, tmp_path):
        unit_dir = tmp_path / "u"
        files = _make_corpus_files(unit_dir, ["x\n"])
        stats = CorpusStats(unit_name="u", n_docs=1, n_tokens=1, n_vocab_raw=1,
                            n_corpus_files=1, scanned_at="t", from_cache=False)
        write_cache(unit_dir, stats, files)
        # Same files, same mtimes → fresh
        assert cache_is_fresh(unit_dir, files) is True

    def test_stale_when_file_mtime_changes(self, tmp_path):
        unit_dir = tmp_path / "u"
        files = _make_corpus_files(unit_dir, ["x\n"])
        stats = CorpusStats(unit_name="u", n_docs=1, n_tokens=1, n_vocab_raw=1,
                            n_corpus_files=1, scanned_at="t", from_cache=False)
        write_cache(unit_dir, stats, files)
        # Bump mtime → stale
        import os, time
        new_mtime = files[0].stat().st_mtime + 100
        os.utime(files[0], (new_mtime, new_mtime))
        assert cache_is_fresh(unit_dir, files) is False

    def test_stale_when_file_added(self, tmp_path):
        unit_dir = tmp_path / "u"
        files = _make_corpus_files(unit_dir, ["x\n"])
        stats = CorpusStats(unit_name="u", n_docs=1, n_tokens=1, n_vocab_raw=1,
                            n_corpus_files=1, scanned_at="t", from_cache=False)
        write_cache(unit_dir, stats, files)
        new_file = unit_dir / "corpus_000001"
        new_file.write_text("y\n", encoding="utf-8")
        assert cache_is_fresh(unit_dir, files + [new_file]) is False

    def test_stale_when_no_cache(self, tmp_path):
        unit_dir = tmp_path / "u"
        files = _make_corpus_files(unit_dir, ["x\n"])
        assert cache_is_fresh(unit_dir, files) is False
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_dataset_stats.py -v
```

Expected: `ImportError: cannot import name 'CorpusStats' from 'scripts.common.dataset_stats'` (module doesn't exist yet).

- [ ] **Step 3: Create the module with dataclasses and cache I/O**

```python
# scripts/common/dataset_stats.py
"""Dataset & training summary helpers.

Pure functions plus dataclasses; the CLI shim (scripts/describe_dataset.py)
wires them together. Per-unit corpus scans are cached as JSON sidecars
(.dataset_stats.json) next to the corpus_* files.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

CACHE_FILENAME = ".dataset_stats.json"
CACHE_SCHEMA_VERSION = 1


@dataclass
class CorpusStats:
    unit_name: str
    n_docs: int
    n_tokens: int
    n_vocab_raw: int
    n_corpus_files: int
    scanned_at: str
    from_cache: bool


@dataclass
class RawVolumeEntry:
    unit_name: str
    n_files: int
    n_bytes: int
    layout_hint: str
    n_source_docs: Optional[int] = None  # set by walkers that know it cheaply (e.g. weibo)


def _file_fingerprint(p: Path) -> dict:
    st = p.stat()
    return {"name": p.name, "size": st.st_size, "mtime": st.st_mtime}


def write_cache(unit_dir: Path, stats: CorpusStats, corpus_files: List[Path]) -> None:
    """Persist per-unit scan results to a JSON sidecar."""
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "n_docs": stats.n_docs,
        "n_tokens": stats.n_tokens,
        "n_vocab_raw": stats.n_vocab_raw,
        "scanned_at": stats.scanned_at,
        "corpus_files": [_file_fingerprint(p) for p in sorted(corpus_files)],
    }
    (unit_dir / CACHE_FILENAME).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def read_cache(unit_dir: Path, logger: Optional[logging.Logger] = None) -> Optional[CorpusStats]:
    """Read sidecar; return None if missing, corrupt, or schema-incompatible."""
    cache_path = unit_dir / CACHE_FILENAME
    if not cache_path.exists():
        return None
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        if logger:
            logger.warning(f"Corrupt cache at {cache_path}: {e!r}; will recompute")
        return None
    if payload.get("schema_version") != CACHE_SCHEMA_VERSION:
        return None
    return CorpusStats(
        unit_name=unit_dir.name,
        n_docs=int(payload["n_docs"]),
        n_tokens=int(payload["n_tokens"]),
        n_vocab_raw=int(payload["n_vocab_raw"]),
        n_corpus_files=len(payload.get("corpus_files", [])),
        scanned_at=str(payload.get("scanned_at", "")),
        from_cache=True,
    )


def cache_is_fresh(unit_dir: Path, corpus_files: List[Path]) -> bool:
    """True iff the sidecar's recorded fingerprints exactly match the live files."""
    cache_path = unit_dir / CACHE_FILENAME
    if not cache_path.exists():
        return False
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    if payload.get("schema_version") != CACHE_SCHEMA_VERSION:
        return False
    cached = {f["name"]: (f["size"], f["mtime"]) for f in payload.get("corpus_files", [])}
    live = {p.name: (p.stat().st_size, p.stat().st_mtime) for p in corpus_files}
    return cached == live
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
python -m pytest tests/test_dataset_stats.py -v
```

Expected: 8 passed (4 TestCacheIO + 4 TestCacheFreshness).

- [ ] **Step 5: Commit**

```bash
git add scripts/common/dataset_stats.py tests/test_dataset_stats.py
git commit -m "dataset_stats: dataclasses + per-unit sidecar cache I/O"
```

---

## Task 2: Corpus scan with cache integration

**Files:**
- Modify: `scripts/common/dataset_stats.py`
- Modify: `tests/test_dataset_stats.py`

- [ ] **Step 1: Write the failing tests for corpus scan**

Append to `tests/test_dataset_stats.py`:

```python
from scripts.common.dataset_stats import scan_corpus_unit


class TestScanCorpusUnit:
    def test_counts_docs_tokens_vocab(self, tmp_path):
        unit_dir = tmp_path / "u"
        _make_corpus_files(unit_dir, ["a b c\n", "a b\n"])
        stats = scan_corpus_unit(unit_dir, logger)
        assert stats.n_docs == 2
        assert stats.n_tokens == 5   # 3 + 2
        assert stats.n_vocab_raw == 3  # {a, b, c}
        assert stats.n_corpus_files == 1
        assert stats.from_cache is False

    def test_multi_file_aggregation(self, tmp_path):
        unit_dir = tmp_path / "u"
        _make_corpus_files(unit_dir, ["a b\n", "c d\n", "a c\n"])
        stats = scan_corpus_unit(unit_dir, logger)
        assert stats.n_docs == 3
        assert stats.n_tokens == 6
        assert stats.n_vocab_raw == 4  # {a, b, c, d}
        assert stats.n_corpus_files == 3

    def test_empty_unit_dir(self, tmp_path):
        unit_dir = tmp_path / "u"
        unit_dir.mkdir()
        stats = scan_corpus_unit(unit_dir, logger)
        assert stats.n_docs == 0
        assert stats.n_tokens == 0
        assert stats.n_vocab_raw == 0
        assert stats.n_corpus_files == 0

    def test_empty_file(self, tmp_path):
        unit_dir = tmp_path / "u"
        _make_corpus_files(unit_dir, [""])
        stats = scan_corpus_unit(unit_dir, logger)
        assert stats.n_docs == 0
        assert stats.n_tokens == 0
        assert stats.n_vocab_raw == 0

    def test_cache_written_after_scan(self, tmp_path):
        unit_dir = tmp_path / "u"
        _make_corpus_files(unit_dir, ["a b\n"])
        scan_corpus_unit(unit_dir, logger)
        assert (unit_dir / ".dataset_stats.json").exists()

    def test_cache_hit_skips_scan(self, tmp_path):
        unit_dir = tmp_path / "u"
        _make_corpus_files(unit_dir, ["a b c\n"])
        # First scan: writes cache.
        scan_corpus_unit(unit_dir, logger)
        # Mutate cache to a sentinel value so we can detect a re-scan.
        cache_path = unit_dir / ".dataset_stats.json"
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        payload["n_tokens"] = 99999
        cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        # Second scan: must use cache, not rescan.
        stats = scan_corpus_unit(unit_dir, logger)
        assert stats.n_tokens == 99999
        assert stats.from_cache is True

    def test_force_recomputes_even_with_cache(self, tmp_path):
        unit_dir = tmp_path / "u"
        _make_corpus_files(unit_dir, ["a b c\n"])
        scan_corpus_unit(unit_dir, logger)
        cache_path = unit_dir / ".dataset_stats.json"
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        payload["n_tokens"] = 99999
        cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        stats = scan_corpus_unit(unit_dir, logger, force=True)
        assert stats.n_tokens == 3  # rescanned, sentinel overwritten

    def test_returns_vocab_set_when_requested(self, tmp_path):
        unit_dir = tmp_path / "u"
        _make_corpus_files(unit_dir, ["a b c\n", "a b\n"])
        stats, vocab = scan_corpus_unit(unit_dir, logger, return_vocab=True)
        assert vocab == {"a", "b", "c"}
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_dataset_stats.py::TestScanCorpusUnit -v
```

Expected: `ImportError: cannot import name 'scan_corpus_unit'`.

- [ ] **Step 3: Add scan_corpus_unit to the module**

Append to `scripts/common/dataset_stats.py`:

```python
import datetime
from typing import Set, Tuple, Union


def _list_corpus_files(unit_dir: Path) -> List[Path]:
    """Sorted list of corpus_* files in a unit dir (excludes the .dataset_stats.json sidecar)."""
    return sorted(p for p in unit_dir.glob("corpus_*") if p.is_file())


def scan_corpus_unit(
    unit_dir: Path,
    logger: logging.Logger,
    force: bool = False,
    return_vocab: bool = False,
) -> Union[CorpusStats, Tuple[CorpusStats, Set[str]]]:
    """Count documents, tokens, and unique types in a unit's corpus_* files.

    With ``return_vocab=False`` (default) returns CorpusStats; cache is used
    when fresh. With ``return_vocab=True`` returns ``(CorpusStats, set[str])``;
    always rescans (the cache stores counts, not the vocab set).
    """
    corpus_files = _list_corpus_files(unit_dir)

    # Cache fast path (only when caller doesn't need the actual vocab set).
    if not return_vocab and not force and cache_is_fresh(unit_dir, corpus_files):
        cached = read_cache(unit_dir, logger)
        if cached is not None:
            return cached

    # Scan.
    n_docs = 0
    n_tokens = 0
    vocab: Set[str] = set()
    for path in corpus_files:
        with path.open("r", encoding="utf-8", buffering=8 * 1024 * 1024) as f:
            for line in f:
                tokens = line.split()
                if not tokens:
                    continue
                n_docs += 1
                n_tokens += len(tokens)
                vocab.update(tokens)

    stats = CorpusStats(
        unit_name=unit_dir.name,
        n_docs=n_docs,
        n_tokens=n_tokens,
        n_vocab_raw=len(vocab),
        n_corpus_files=len(corpus_files),
        scanned_at=datetime.datetime.now().isoformat(timespec="seconds"),
        from_cache=False,
    )
    if corpus_files:
        write_cache(unit_dir, stats, corpus_files)

    if return_vocab:
        return stats, vocab
    return stats
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
python -m pytest tests/test_dataset_stats.py -v
```

Expected: 16 passed (8 from Task 1 + 8 new).

- [ ] **Step 5: Commit**

```bash
git add scripts/common/dataset_stats.py tests/test_dataset_stats.py
git commit -m "dataset_stats: scan_corpus_unit with cache + opt-in vocab set"
```

---

## Task 3: Unit discovery + model vocab lookup

**Files:**
- Modify: `scripts/common/dataset_stats.py`
- Modify: `tests/test_dataset_stats.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_dataset_stats.py`:

```python
import sys
import types


def _install_fake_gensim():
    """Match the pattern in tests/test_analyze_cohens_d_singlelist.py."""
    if "gensim" in sys.modules and not getattr(sys.modules["gensim"], "_fake", False):
        try:
            from gensim.models import KeyedVectors  # noqa: F401
            return
        except Exception:
            pass
    fake_gensim = types.ModuleType("gensim")
    fake_gensim._fake = True  # type: ignore[attr-defined]
    fake_models = types.ModuleType("gensim.models")

    class _FakeKV:
        @staticmethod
        def load(path):
            class _Stub:
                index_to_key = ["a", "b", "c"]
            return _Stub()

    fake_models.KeyedVectors = _FakeKV  # type: ignore[attr-defined]
    fake_gensim.models = fake_models  # type: ignore[attr-defined]
    sys.modules["gensim"] = fake_gensim
    sys.modules["gensim.models"] = fake_models


_install_fake_gensim()

from scripts.common.dataset_stats import discover_units, model_vocab_size


class TestDiscoverUnits:
    def test_returns_sorted_subdirs(self, tmp_path):
        corpora = tmp_path / "corpora"
        (corpora / "1950_1959").mkdir(parents=True)
        (corpora / "1940_1949").mkdir(parents=True)
        (corpora / "_skip_file.txt").parent.mkdir(exist_ok=True)
        (corpora / "_skip_file.txt").write_text("x")
        config = {"paths": {"corpora_dir": str(corpora)}}
        units = discover_units(config)
        assert units == ["1940_1949", "1950_1959"]

    def test_missing_corpora_dir_returns_empty(self, tmp_path):
        config = {"paths": {"corpora_dir": str(tmp_path / "nope")}}
        assert discover_units(config) == []


class TestModelVocabSize:
    def test_returns_vocab_count(self, tmp_path):
        model_path = tmp_path / "m.model"
        model_path.write_text("stub")  # contents irrelevant — fake gensim ignores
        assert model_vocab_size(model_path, logger) == 3

    def test_missing_file_returns_none(self, tmp_path):
        assert model_vocab_size(tmp_path / "nope.model", logger) is None
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_dataset_stats.py::TestDiscoverUnits tests/test_dataset_stats.py::TestModelVocabSize -v
```

Expected: `ImportError: cannot import name 'discover_units'`.

- [ ] **Step 3: Add discover_units and model_vocab_size**

Append to `scripts/common/dataset_stats.py`:

```python
def discover_units(config: dict) -> List[str]:
    """Return sorted unit directory names under corpora_dir.

    Mirrors scripts.train_embeddings.discover_units; lifted here so the
    describe-dataset tool doesn't import the trainer.
    """
    corpora_dir = Path(config["paths"]["corpora_dir"])
    if not corpora_dir.exists():
        return []
    return sorted(d.name for d in corpora_dir.iterdir() if d.is_dir())


def model_vocab_size(model_path: Path, logger: logging.Logger) -> Optional[int]:
    """Open a gensim model and return its vocab size; None on missing/corrupt.

    Uses KeyedVectors.load — works for .kv files and for Word2Vec.save_word2vec_format
    output. For full Word2Vec.save() output, gensim falls back through its loader
    chain; if that fails the function logs and returns None.
    """
    if not model_path.exists():
        logger.warning(f"Model file missing: {model_path}")
        return None
    try:
        # Local import: gensim is broken on some test envs (scipy.linalg.triu);
        # only this function actually needs it.
        from gensim.models import KeyedVectors
        kv = KeyedVectors.load(str(model_path))
        return len(kv.index_to_key)
    except Exception as e:  # noqa: BLE001 — gensim raises many distinct types
        logger.warning(f"Could not introspect {model_path}: {e!r}")
        return None
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
python -m pytest tests/test_dataset_stats.py -v
```

Expected: 20 passed (16 + 4).

- [ ] **Step 5: Commit**

```bash
git add scripts/common/dataset_stats.py tests/test_dataset_stats.py
git commit -m "dataset_stats: unit discovery + model vocab introspection"
```

---

## Task 4: Source totals + Markdown renderer

**Files:**
- Modify: `scripts/common/dataset_stats.py`
- Modify: `tests/test_dataset_stats.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_dataset_stats.py`:

```python
from scripts.common.dataset_stats import (
    SourceTotals, aggregate_source, render_markdown,
)


def _stats(unit, docs, tokens, vocab):
    return CorpusStats(
        unit_name=unit, n_docs=docs, n_tokens=tokens, n_vocab_raw=vocab,
        n_corpus_files=1, scanned_at="t", from_cache=False,
    )


def _raw(unit, files, nbytes):
    return RawVolumeEntry(unit_name=unit, n_files=files, n_bytes=nbytes,
                          layout_hint="hint")


class TestAggregateSource:
    def test_sums_and_per_unit_stats(self):
        per_unit = {
            "u1": (_stats("u1", 10, 100, 50), 20, _raw("u1", 5, 1000)),
            "u2": (_stats("u2", 20, 300, 80), None, _raw("u2", 7, 2000)),
        }
        totals = aggregate_source(per_unit, vocab_union_count=None)
        assert totals.n_units == 2
        assert totals.n_docs == 30
        assert totals.n_tokens == 400
        assert totals.vocab_raw_min == 50
        assert totals.vocab_raw_max == 80
        assert totals.vocab_raw_mean == 65.0
        assert totals.n_model_vocab_sum == 20  # one None skipped
        assert totals.n_raw_files == 12
        assert totals.n_raw_bytes == 3000
        assert totals.vocab_union_count is None

    def test_vocab_union_passed_through(self):
        per_unit = {"u1": (_stats("u1", 1, 1, 1), None, None)}
        totals = aggregate_source(per_unit, vocab_union_count=999)
        assert totals.vocab_union_count == 999


class TestRenderMarkdown:
    def _minimal_config(self):
        return {
            "language": "zh", "data_source": "renminribao",
            "embedding": {"vector_size": 300, "window": 4, "min_count": 50,
                          "sg": 1, "negative": 15, "epochs": 5, "seed": 42,
                          "workers": 16, "model_name_template": "rmrb_{slice_name}.model"},
            "paths": {"models_dir": "/data/models", "raw_data_dir": "/data/raw"},
        }

    def test_renders_all_sections(self):
        per_unit = {
            "1940_1949": (_stats("1940_1949", 100, 1000, 250), 80, _raw("1940_1949", 12, 5000)),
        }
        totals = aggregate_source(per_unit, vocab_union_count=None)
        md = render_markdown(
            totals=totals, per_unit=per_unit, config=self._minimal_config(),
            config_path="config/profiles/renminribao.yml", generated_at="2026-06-02",
        )
        assert "# Dataset Summary — renminribao (zh)" in md
        assert "## Corpus totals" in md
        assert "## Raw data" in md
        assert "## Training" in md
        assert "## Per-unit breakdown" in md
        assert "1940_1949" in md
        assert "min_count=50" in md  # pulled from config, not hard-coded
        assert "vector_size=300" in md
        assert "config/profiles/renminribao.yml" in md
        assert "2026-06-02" in md

    def test_ngram_renames_documents_column(self):
        per_unit = {
            "1940_1949": (_stats("1940_1949", 7, 21, 3), None, _raw("1940_1949", 1, 10)),
        }
        totals = aggregate_source(per_unit, vocab_union_count=None)
        cfg = self._minimal_config()
        cfg["data_source"] = "ngram"
        md = render_markdown(
            totals=totals, per_unit=per_unit, config=cfg,
            config_path="x.yml", generated_at="2026-06-02",
        )
        assert "N-gram entries" in md
        # The renamed column carries the same count as Documents would.
        assert "7" in md

    def test_vocab_union_shown_when_present(self):
        per_unit = {"u1": (_stats("u1", 1, 1, 1), None, None)}
        totals = aggregate_source(per_unit, vocab_union_count=12345)
        md = render_markdown(
            totals=totals, per_unit=per_unit, config=self._minimal_config(),
            config_path="x.yml", generated_at="2026-06-02",
        )
        assert "12,345" in md or "12345" in md
        assert "union" in md.lower()
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_dataset_stats.py::TestAggregateSource tests/test_dataset_stats.py::TestRenderMarkdown -v
```

Expected: `ImportError: cannot import name 'SourceTotals'`.

- [ ] **Step 3: Add SourceTotals, aggregate_source, render_markdown**

Append to `scripts/common/dataset_stats.py`:

```python
from typing import Dict, Tuple


@dataclass
class SourceTotals:
    n_units: int
    n_docs: int
    n_tokens: int
    vocab_raw_min: int
    vocab_raw_max: int
    vocab_raw_mean: float
    n_model_vocab_sum: int             # sum of per-unit model vocabs (None entries skipped)
    n_raw_files: int
    n_raw_bytes: int
    vocab_union_count: Optional[int]   # None unless --vocab-union was passed


PerUnitTriple = Tuple[CorpusStats, Optional[int], Optional[RawVolumeEntry]]
# (corpus_stats, model_vocab_size_or_None, raw_volume_entry_or_None)


def aggregate_source(
    per_unit: Dict[str, PerUnitTriple],
    vocab_union_count: Optional[int],
) -> SourceTotals:
    """Reduce per-unit triples to one source-level summary row."""
    if not per_unit:
        return SourceTotals(0, 0, 0, 0, 0, 0.0, 0, 0, 0, vocab_union_count)
    vocab_raws = [s.n_vocab_raw for s, _, _ in per_unit.values()]
    return SourceTotals(
        n_units=len(per_unit),
        n_docs=sum(s.n_docs for s, _, _ in per_unit.values()),
        n_tokens=sum(s.n_tokens for s, _, _ in per_unit.values()),
        vocab_raw_min=min(vocab_raws),
        vocab_raw_max=max(vocab_raws),
        vocab_raw_mean=sum(vocab_raws) / len(vocab_raws),
        n_model_vocab_sum=sum(v for _, v, _ in per_unit.values() if v is not None),
        n_raw_files=sum(r.n_files for _, _, r in per_unit.values() if r is not None),
        n_raw_bytes=sum(r.n_bytes for _, _, r in per_unit.values() if r is not None),
        vocab_union_count=vocab_union_count,
    )


def _human_bytes(n: int) -> str:
    """1024-based, two-significant-figures, KB/MB/GB/TB."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return f"{n:.1f} TB"


def _fmt(n) -> str:
    if n is None:
        return "n/a"
    if isinstance(n, float):
        return f"{n:,.1f}"
    return f"{n:,}"


def render_markdown(
    totals: SourceTotals,
    per_unit: Dict[str, PerUnitTriple],
    config: dict,
    config_path: str,
    generated_at: str,
) -> str:
    """Render the per-source Markdown summary."""
    src = config.get("data_source", "?")
    lang = config.get("language", "?")
    emb = config.get("embedding", {})
    is_ngram = src == "ngram"
    docs_header = "N-gram entries" if is_ngram else "Documents"

    lines: List[str] = []
    lines.append(f"# Dataset Summary — {src} ({lang})")
    lines.append("")
    lines.append(f"Generated {generated_at} from `{config_path}`.")
    lines.append("")

    # Corpus totals
    lines.append("## Corpus totals")
    lines.append("")
    union_cell = f" {_fmt(totals.vocab_union_count)} (union)" if totals.vocab_union_count is not None else ""
    lines.append(f"| Units | {docs_header} | Tokens | Raw vocab per unit (min / mean / max){' | Raw vocab union' if totals.vocab_union_count is not None else ''} | Trained-model vocab (sum, min_count={emb.get('min_count', '?')}) |")
    sep_extra = "|---" if totals.vocab_union_count is not None else ""
    lines.append(f"|---|---|---|---{sep_extra}|---|")
    vocab_cells = f"{_fmt(totals.vocab_raw_min)} / {_fmt(totals.vocab_raw_mean)} / {_fmt(totals.vocab_raw_max)}"
    union_col = f" | {_fmt(totals.vocab_union_count)}" if totals.vocab_union_count is not None else ""
    lines.append(
        f"| {_fmt(totals.n_units)} | {_fmt(totals.n_docs)} | {_fmt(totals.n_tokens)} | "
        f"{vocab_cells}{union_col} | {_fmt(totals.n_model_vocab_sum)} |"
    )
    lines.append("")

    # Raw data
    lines.append("## Raw data")
    lines.append("")
    layout_hint = next(
        (r.layout_hint for _, _, r in per_unit.values() if r is not None), None
    )
    raw_dir = config.get("paths", {}).get("raw_data_dir", "?")
    if layout_hint:
        lines.append(f"- Layout: `{layout_hint}`")
    lines.append(f"- raw_data_dir: `{raw_dir}`")
    lines.append(f"- Source files: {_fmt(totals.n_raw_files)}")
    lines.append(f"- Bytes: {_human_bytes(totals.n_raw_bytes)}")
    lines.append("")

    # Training
    lines.append("## Training")
    lines.append("")
    algo = "Word2Vec skip-gram with negative sampling (gensim)" if emb.get("sg") == 1 else "Word2Vec CBOW (gensim)"
    lines.append(f"- Algorithm: {algo}")
    params = (
        f"vector_size={emb.get('vector_size', '?')} · "
        f"window={emb.get('window', '?')} · "
        f"min_count={emb.get('min_count', '?')} · "
        f"sg={emb.get('sg', '?')} · "
        f"negative={emb.get('negative', '?')} · "
        f"epochs={emb.get('epochs', '?')} · "
        f"seed={emb.get('seed', '?')}"
    )
    lines.append(f"- `{params}`")
    template = emb.get("model_name_template", "?")
    models_dir = config.get("paths", {}).get("models_dir", "?")
    lines.append(f"- Model files: `{models_dir}/{template}`")
    lines.append("")

    # Per-unit breakdown
    lines.append("## Per-unit breakdown")
    lines.append("")
    lines.append(f"| Unit | {docs_header} | Tokens | Raw vocab | Model vocab | Raw files | Raw bytes |")
    lines.append("|---|---|---|---|---|---|---|")
    for unit_name in sorted(per_unit):
        stats, mv, raw = per_unit[unit_name]
        raw_files = _fmt(raw.n_files) if raw is not None else "n/a"
        raw_bytes = _human_bytes(raw.n_bytes) if raw is not None else "n/a"
        lines.append(
            f"| {unit_name} | {_fmt(stats.n_docs)} | {_fmt(stats.n_tokens)} | "
            f"{_fmt(stats.n_vocab_raw)} | {_fmt(mv)} | {raw_files} | {raw_bytes} |"
        )
    lines.append("")
    return "\n".join(lines)
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
python -m pytest tests/test_dataset_stats.py -v
```

Expected: 25 passed (20 + 5 new: 2 TestAggregateSource + 3 TestRenderMarkdown).

- [ ] **Step 5: Commit**

```bash
git add scripts/common/dataset_stats.py tests/test_dataset_stats.py
git commit -m "dataset_stats: aggregate_source + Markdown renderer (ngram-aware, vocab-union-aware)"
```

---

## Task 5: Raw-volume registry + RMRB walker

**Files:**
- Create: `scripts/data_prep/raw_volume/__init__.py`
- Create: `scripts/data_prep/raw_volume/rmrb.py`
- Create: `tests/test_raw_volume_rmrb.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_raw_volume_rmrb.py
"""Tests for the RMRB raw-data walker."""

from __future__ import annotations

import logging
from pathlib import Path

from scripts.data_prep.raw_volume import WALKERS
from scripts.data_prep.raw_volume.rmrb import walk

logger = logging.getLogger("test")


def _make_rmrb_tree(root: Path, files_by_year: dict[int, int]) -> None:
    """Create rmrb_YYYY_MM.txt files under {decade}/{year}/报刊/人民日报/."""
    for year, n_months in files_by_year.items():
        decade = f"{(year // 10) * 10}s"
        ymd = root / decade / str(year) / "报刊" / "人民日报"
        ymd.mkdir(parents=True, exist_ok=True)
        for m in range(1, n_months + 1):
            (ymd / f"rmrb_{year}_{m:02d}.txt").write_text("x" * 100, encoding="utf-8")


def test_walker_registered():
    assert "renminribao" in WALKERS
    assert WALKERS["renminribao"] is walk


def test_walks_overlapping_slices(tmp_path):
    _make_rmrb_tree(tmp_path, {1942: 12, 1948: 6, 1955: 3})
    # Default RMRB profile: 10-year window, 5-year step → 1940_1949, 1945_1954, …
    config = {
        "time_slices": {"start_year": 1940, "end_year": 1959,
                        "window_size": 10, "step_size": 5},
    }
    units = ["1940_1949", "1945_1954", "1950_1959"]
    result = walk(tmp_path, units, config, logger)
    # 1940_1949 contains years 1942 + 1948 = 12 + 6 = 18 files
    assert result["1940_1949"].n_files == 18
    # 1945_1954 contains years 1948 + (no 1950–1954 here) = 6
    assert result["1945_1954"].n_files == 6
    # 1950_1959 contains year 1955 = 3
    assert result["1950_1959"].n_files == 3
    # Bytes are 100 per file.
    assert result["1940_1949"].n_bytes == 1800


def test_returns_zero_entry_for_unit_with_no_files(tmp_path):
    _make_rmrb_tree(tmp_path, {1942: 1})
    config = {"time_slices": {"start_year": 1940, "end_year": 1959,
                              "window_size": 10, "step_size": 5}}
    result = walk(tmp_path, ["1970_1979"], config, logger)
    assert result["1970_1979"].n_files == 0
    assert result["1970_1979"].n_bytes == 0


def test_missing_raw_dir_yields_zero_entries(tmp_path):
    config = {"time_slices": {"start_year": 1940, "end_year": 1959,
                              "window_size": 10, "step_size": 5}}
    result = walk(tmp_path / "nope", ["1940_1949"], config, logger)
    assert result["1940_1949"].n_files == 0
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_raw_volume_rmrb.py -v
```

Expected: `ModuleNotFoundError: scripts.data_prep.raw_volume`.

- [ ] **Step 3: Create the registry and RMRB walker**

```python
# scripts/data_prep/raw_volume/__init__.py
"""Per-source raw-data walkers for the dataset summary reporter.

Each walker exports a ``walk(raw_data_dir, units, config, logger)`` function
returning ``Dict[unit_name, RawVolumeEntry]``. The WALKERS registry maps
config ``data_source`` strings to walkers; the reporter dispatches on it.
"""

from scripts.data_prep.raw_volume.rmrb import walk as _walk_rmrb

WALKERS = {
    "renminribao": _walk_rmrb,
}
```

```python
# scripts/data_prep/raw_volume/rmrb.py
"""RMRB (People's Daily) raw-data walker.

Layout: {raw_data_dir}/{decade}/{year}/报刊/人民日报/rmrb_YYYY_MM.txt

A source file can belong to multiple overlapping time slices, so the walker
groups by membership rather than by partition.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from scripts.common.dataset_stats import RawVolumeEntry

LAYOUT = "{raw_data_dir}/{decade}/{year}/报刊/人民日报/rmrb_YYYY_MM.txt"
_YEAR_RE = re.compile(r"rmrb_(\d{4})_\d{2}\.txt")


def _slice_bounds(unit_name: str) -> Tuple[int, int] | None:
    m = re.match(r"^(\d{4})_(\d{4})$", unit_name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def _iter_source_files(raw_data_dir: Path):
    """Yield (path, year) for every rmrb_YYYY_MM.txt under the layout."""
    if not raw_data_dir.exists():
        return
    for decade_dir in sorted(raw_data_dir.glob("*s")):
        if not decade_dir.is_dir():
            continue
        for year_dir in sorted(decade_dir.iterdir()):
            if not year_dir.is_dir():
                continue
            rmr_dir = year_dir / "报刊" / "人民日报"
            if not rmr_dir.exists():
                continue
            for txt_file in sorted(rmr_dir.glob("rmrb_*.txt")):
                m = _YEAR_RE.search(txt_file.name)
                if m:
                    yield txt_file, int(m.group(1))


def walk(
    raw_data_dir: Path,
    units: List[str],
    config: dict,
    logger,
) -> Dict[str, RawVolumeEntry]:
    raw_data_dir = Path(raw_data_dir)
    if not raw_data_dir.exists():
        logger.warning(f"raw_data_dir not present: {raw_data_dir}")
        return {u: RawVolumeEntry(u, 0, 0, LAYOUT) for u in units}

    files_by_unit: Dict[str, List[Path]] = defaultdict(list)
    for path, year in _iter_source_files(raw_data_dir):
        for unit in units:
            bounds = _slice_bounds(unit)
            if bounds is None:
                continue
            start, end = bounds
            if start <= year <= end:
                files_by_unit[unit].append(path)

    out: Dict[str, RawVolumeEntry] = {}
    for u in units:
        files = files_by_unit.get(u, [])
        out[u] = RawVolumeEntry(
            unit_name=u,
            n_files=len(files),
            n_bytes=sum(p.stat().st_size for p in files),
            layout_hint=LAYOUT,
        )
    return out
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
python -m pytest tests/test_raw_volume_rmrb.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/data_prep/raw_volume/__init__.py scripts/data_prep/raw_volume/rmrb.py tests/test_raw_volume_rmrb.py
git commit -m "raw_volume: registry + RMRB walker (overlapping time-slice membership)"
```

---

## Task 6: Provincial newspaper walker

**Files:**
- Create: `scripts/data_prep/raw_volume/provincial_newspaper.py`
- Create: `tests/test_raw_volume_provincial_newspaper.py`
- Modify: `scripts/data_prep/raw_volume/__init__.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_raw_volume_provincial_newspaper.py
"""Tests for the provincial-newspaper raw-data walker."""

from __future__ import annotations

import logging
from pathlib import Path

from scripts.data_prep.raw_volume import WALKERS
from scripts.data_prep.raw_volume.provincial_newspaper import walk

logger = logging.getLogger("test")


def _make_tree(root: Path, layout: dict[str, dict[str, int]]) -> None:
    """layout = {folder_name: {year: n_files_in_year}}."""
    for folder, years in layout.items():
        for year, n in years.items():
            d = root / folder / str(year)
            d.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                (d / f"{year}-{i:03d}.txt").write_text("y" * 50, encoding="utf-8")


def test_walker_registered():
    assert "newspaper" in WALKERS
    assert WALKERS["newspaper"] is walk


def test_groups_by_province_year(tmp_path):
    _make_tree(tmp_path, {
        "北京日报":  {"2020": 3, "2021": 5},
        "天津日报":  {"2020": 2},
        "广东日报":  {"2022": 4},
    })
    units = ["北京_2020", "北京_2021", "天津_2020", "广东_2022", "上海_2020"]
    result = walk(tmp_path, units, {}, logger)
    assert result["北京_2020"].n_files == 3
    assert result["北京_2021"].n_files == 5
    assert result["天津_2020"].n_files == 2
    assert result["广东_2022"].n_files == 4
    assert result["上海_2020"].n_files == 0  # nothing under that folder
    assert result["北京_2020"].n_bytes == 3 * 50


def test_unknown_province_folder_ignored(tmp_path):
    _make_tree(tmp_path, {"未知日报": {"2020": 5}})
    result = walk(tmp_path, ["北京_2020"], {}, logger)
    assert result["北京_2020"].n_files == 0


def test_missing_raw_dir_yields_zero_entries(tmp_path):
    result = walk(tmp_path / "nope", ["北京_2020"], {}, logger)
    assert result["北京_2020"].n_files == 0
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_raw_volume_provincial_newspaper.py -v
```

Expected: `ModuleNotFoundError: scripts.data_prep.raw_volume.provincial_newspaper`.

- [ ] **Step 3: Create the walker**

```python
# scripts/data_prep/raw_volume/provincial_newspaper.py
"""Provincial newspaper raw-data walker.

Layout: {raw_data_dir}/{province_folder}/{year}/...

Province folder names follow the mapping in
scripts.data_prep.build_corpora_provincial_newspaper.FOLDER_TO_PROVINCE
(reused so the two stay in sync).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from scripts.common.dataset_stats import RawVolumeEntry
from scripts.data_prep.build_corpora_provincial_newspaper import FOLDER_TO_PROVINCE

LAYOUT = "{raw_data_dir}/{province_folder}/{year}/..."


def _parse_unit(unit_name: str):
    """'北京_2020' -> ('北京', '2020'); returns None on bad shapes."""
    if "_" not in unit_name:
        return None
    province, year = unit_name.rsplit("_", 1)
    return province, year


def walk(
    raw_data_dir: Path,
    units: List[str],
    config: dict,
    logger,
) -> Dict[str, RawVolumeEntry]:
    raw_data_dir = Path(raw_data_dir)
    if not raw_data_dir.exists():
        logger.warning(f"raw_data_dir not present: {raw_data_dir}")
        return {u: RawVolumeEntry(u, 0, 0, LAYOUT) for u in units}

    # Per (province, year) → list of paths.
    files_by_pv: Dict[tuple, List[Path]] = defaultdict(list)
    for folder in sorted(raw_data_dir.iterdir()):
        if not folder.is_dir():
            continue
        province = FOLDER_TO_PROVINCE.get(folder.name)
        if province is None:
            continue
        for year_dir in sorted(folder.iterdir()):
            if not year_dir.is_dir():
                continue
            year = year_dir.name
            for f in year_dir.rglob("*"):
                if f.is_file():
                    files_by_pv[(province, year)].append(f)

    out: Dict[str, RawVolumeEntry] = {}
    for u in units:
        parsed = _parse_unit(u)
        if parsed is None:
            out[u] = RawVolumeEntry(u, 0, 0, LAYOUT)
            continue
        files = files_by_pv.get(parsed, [])
        out[u] = RawVolumeEntry(
            unit_name=u,
            n_files=len(files),
            n_bytes=sum(p.stat().st_size for p in files),
            layout_hint=LAYOUT,
        )
    return out
```

- [ ] **Step 4: Register the walker**

Modify `scripts/data_prep/raw_volume/__init__.py`:

```python
"""Per-source raw-data walkers for the dataset summary reporter.

Each walker exports a ``walk(raw_data_dir, units, config, logger)`` function
returning ``Dict[unit_name, RawVolumeEntry]``. The WALKERS registry maps
config ``data_source`` strings to walkers; the reporter dispatches on it.
"""

from scripts.data_prep.raw_volume.rmrb import walk as _walk_rmrb
from scripts.data_prep.raw_volume.provincial_newspaper import walk as _walk_newspaper

WALKERS = {
    "renminribao": _walk_rmrb,
    "newspaper": _walk_newspaper,
}
```

- [ ] **Step 5: Run tests to confirm they pass**

```bash
python -m pytest tests/test_raw_volume_provincial_newspaper.py tests/test_raw_volume_rmrb.py -v
```

Expected: 8 passed.

- [ ] **Step 6: Commit**

```bash
git add scripts/data_prep/raw_volume/provincial_newspaper.py scripts/data_prep/raw_volume/__init__.py tests/test_raw_volume_provincial_newspaper.py
git commit -m "raw_volume: provincial-newspaper walker (reuses FOLDER_TO_PROVINCE)"
```

---

## Task 7: Weibo walker (parquet)

**Files:**
- Create: `scripts/data_prep/raw_volume/weibo.py`
- Create: `tests/test_raw_volume_weibo.py`
- Modify: `scripts/data_prep/raw_volume/__init__.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_raw_volume_weibo.py
"""Tests for the Weibo raw-data walker (parquet shards)."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pytest

from scripts.data_prep.raw_volume import WALKERS
from scripts.data_prep.raw_volume.weibo import walk

logger = logging.getLogger("test")

pa = pytest.importorskip("pyarrow")


def _write_parquet(path: Path, n_rows: int, province_code: str = "11") -> None:
    df = pd.DataFrame({
        "text": ["t"] * n_rows,
        "user_province": [province_code] * n_rows,
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def test_walker_registered():
    assert "weibo" in WALKERS
    assert WALKERS["weibo"] is walk


def test_counts_files_and_bytes_per_province(tmp_path):
    # Province code 11 = 北京, 31 = 上海 (per PROVINCE_CODE_TO_NAME).
    _write_parquet(tmp_path / "2020" / "a.parquet", n_rows=10, province_code="11")
    _write_parquet(tmp_path / "2020" / "b.parquet", n_rows=20, province_code="11")
    _write_parquet(tmp_path / "2020" / "c.parquet", n_rows=5,  province_code="31")
    result = walk(tmp_path, ["北京", "上海", "天津"], {}, logger)
    # 北京: 2 files; 上海: 1 file; 天津: none.
    assert result["北京"].n_files == 2
    assert result["上海"].n_files == 1
    assert result["天津"].n_files == 0
    # n_source_docs from parquet row counts.
    assert result["北京"].n_source_docs == 30
    assert result["上海"].n_source_docs == 5
    assert result["天津"].n_source_docs == 0
    assert result["北京"].n_bytes > 0


def test_missing_raw_dir_yields_zero_entries(tmp_path):
    result = walk(tmp_path / "nope", ["北京"], {}, logger)
    assert result["北京"].n_files == 0
    assert result["北京"].n_source_docs == 0
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_raw_volume_weibo.py -v
```

Expected: `ModuleNotFoundError: scripts.data_prep.raw_volume.weibo`.

- [ ] **Step 3: Create the Weibo walker**

```python
# scripts/data_prep/raw_volume/weibo.py
"""Weibo raw-data walker.

Layout: {raw_data_dir}/.../*.parquet, with a ``user_province`` (GB/T 2260 code)
or ``region_name`` column. Per-province grouping; row counts are exact (parquet
metadata is cheap to read).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from scripts.common.dataset_stats import RawVolumeEntry
from scripts.data_prep.build_corpora_weibo import (
    PROVINCE_CODE_TO_NAME, PROVINCE_NAME_TO_CODE,
)

LAYOUT = "{raw_data_dir}/**/*.parquet (user_province GB/T 2260 codes)"


def _row_count(parquet_path: Path) -> int:
    """Cheap row count via parquet metadata."""
    import pyarrow.parquet as pq
    return pq.ParquetFile(parquet_path).metadata.num_rows


def _province_of(path: Path, logger) -> Optional[str]:
    """Read just enough of the parquet to find the dominant province."""
    import pyarrow.parquet as pq
    try:
        # Read only the province column.
        for col in ("user_province", "region_name"):
            schema = pq.read_schema(path)
            if col in schema.names:
                table = pq.read_table(path, columns=[col])
                values = table[col].to_pylist()
                if not values:
                    return None
                # Most-common entry.
                from collections import Counter
                top = Counter(v for v in values if v is not None).most_common(1)
                if not top:
                    return None
                v = top[0][0]
                if col == "user_province":
                    return PROVINCE_CODE_TO_NAME.get(str(v))
                return v  # region_name is already a province name
        return None
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Could not read province from {path.name}: {e!r}")
        return None


def walk(
    raw_data_dir: Path,
    units: List[str],
    config: dict,
    logger,
) -> Dict[str, RawVolumeEntry]:
    raw_data_dir = Path(raw_data_dir)
    if not raw_data_dir.exists():
        logger.warning(f"raw_data_dir not present: {raw_data_dir}")
        return {
            u: RawVolumeEntry(u, 0, 0, LAYOUT, n_source_docs=0) for u in units
        }

    by_prov: Dict[str, List[Path]] = defaultdict(list)
    rows_by_prov: Dict[str, int] = defaultdict(int)

    parquets = sorted(raw_data_dir.rglob("*.parquet"))
    for i, p in enumerate(parquets, 1):
        if i % 100 == 0:
            logger.info(f"  Weibo walker: scanned {i}/{len(parquets)} parquets")
        province = _province_of(p, logger)
        if province is None:
            continue
        by_prov[province].append(p)
        rows_by_prov[province] += _row_count(p)

    out: Dict[str, RawVolumeEntry] = {}
    for u in units:
        # Unit may be bare province name or province_year — extract province.
        province = u.split("_", 1)[0]
        files = by_prov.get(province, [])
        out[u] = RawVolumeEntry(
            unit_name=u,
            n_files=len(files),
            n_bytes=sum(p.stat().st_size for p in files),
            layout_hint=LAYOUT,
            n_source_docs=rows_by_prov.get(province, 0),
        )
    return out
```

- [ ] **Step 4: Register the walker**

Modify `scripts/data_prep/raw_volume/__init__.py`:

```python
from scripts.data_prep.raw_volume.rmrb import walk as _walk_rmrb
from scripts.data_prep.raw_volume.provincial_newspaper import walk as _walk_newspaper
from scripts.data_prep.raw_volume.weibo import walk as _walk_weibo

WALKERS = {
    "renminribao": _walk_rmrb,
    "newspaper": _walk_newspaper,
    "weibo": _walk_weibo,
}
```

- [ ] **Step 5: Run tests to confirm they pass**

```bash
python -m pytest tests/test_raw_volume_weibo.py tests/test_raw_volume_provincial_newspaper.py tests/test_raw_volume_rmrb.py -v
```

Expected: 11 passed (3 + 4 + 4).

- [ ] **Step 6: Commit**

```bash
git add scripts/data_prep/raw_volume/weibo.py scripts/data_prep/raw_volume/__init__.py tests/test_raw_volume_weibo.py
git commit -m "raw_volume: weibo walker (parquet row counts via pyarrow metadata)"
```

---

## Task 8: Ngram walkers (zh + en)

**Files:**
- Create: `scripts/data_prep/raw_volume/ngram_zh.py`
- Create: `scripts/data_prep/raw_volume/ngram_en.py`
- Create: `tests/test_raw_volume_ngram.py`
- Modify: `scripts/data_prep/raw_volume/__init__.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_raw_volume_ngram.py
"""Tests for Chinese + English Google Ngram raw-data walkers."""

from __future__ import annotations

import logging
from pathlib import Path

from scripts.data_prep.raw_volume import WALKERS
from scripts.data_prep.raw_volume.ngram_zh import walk as walk_zh
from scripts.data_prep.raw_volume.ngram_en import walk as walk_en

logger = logging.getLogger("test")


def _make_gz(path: Path, n_bytes: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x1f\x8b" + b"\x00" * (n_bytes - 2))


def test_zh_walker_registered_under_ngram_zh_data_source():
    # The zh walker is dispatched when language=='zh' and data_source=='ngram'.
    assert "ngram_zh" in WALKERS
    assert WALKERS["ngram_zh"] is walk_zh


def test_en_walker_registered_under_ngram_en_data_source():
    assert "ngram_en" in WALKERS
    assert WALKERS["ngram_en"] is walk_en


def test_zh_groups_by_inferred_year(tmp_path):
    # Filename pattern carries the year (e.g. 5gm-1945.gz).
    _make_gz(tmp_path / "5gm-1942.gz", n_bytes=300)
    _make_gz(tmp_path / "5gm-1948.gz", n_bytes=500)
    _make_gz(tmp_path / "5gm-1955.gz", n_bytes=200)
    config = {"time_slices": {"start_year": 1940, "end_year": 1959,
                              "window_size": 10, "step_size": 5}}
    result = walk_zh(tmp_path, ["1940_1949", "1945_1954", "1950_1959"], config, logger)
    assert result["1940_1949"].n_files == 2  # 1942 + 1948
    assert result["1945_1954"].n_files == 1  # 1948
    assert result["1950_1959"].n_files == 1  # 1955
    assert result["1940_1949"].n_bytes == 800


def test_en_groups_by_inferred_year(tmp_path):
    _make_gz(tmp_path / "googlebooks-eng-all-5gram-20120701-aa-1990.gz", n_bytes=400)
    _make_gz(tmp_path / "googlebooks-eng-all-5gram-20120701-bb-1995.gz", n_bytes=600)
    config = {"time_slices": {"start_year": 1990, "end_year": 1999,
                              "window_size": 10, "step_size": 10}}
    result = walk_en(tmp_path, ["1990_1999"], config, logger)
    assert result["1990_1999"].n_files == 2
    assert result["1990_1999"].n_bytes == 1000


def test_missing_raw_dir(tmp_path):
    config = {"time_slices": {"start_year": 1940, "end_year": 1949,
                              "window_size": 10, "step_size": 10}}
    result = walk_zh(tmp_path / "nope", ["1940_1949"], config, logger)
    assert result["1940_1949"].n_files == 0
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_raw_volume_ngram.py -v
```

Expected: `ModuleNotFoundError: scripts.data_prep.raw_volume.ngram_zh`.

- [ ] **Step 3: Create the ngram walkers + shared helper**

```python
# scripts/data_prep/raw_volume/ngram_zh.py
"""Chinese Google Ngram raw-data walker.

Layout: {raw_data_dir}/**/*.gz, with the year extractable from the filename
(typically `5gm-YYYY.gz`).
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from scripts.common.dataset_stats import RawVolumeEntry

LAYOUT = "{raw_data_dir}/**/*.gz  (year from filename, e.g. 5gm-YYYY.gz)"
_YEAR_RE = re.compile(r"(\d{4})")


def _slice_bounds(unit: str) -> Tuple[int, int] | None:
    m = re.match(r"^(\d{4})_(\d{4})$", unit)
    return (int(m.group(1)), int(m.group(2))) if m else None


def walk(
    raw_data_dir: Path,
    units: List[str],
    config: dict,
    logger,
) -> Dict[str, RawVolumeEntry]:
    raw_data_dir = Path(raw_data_dir)
    if not raw_data_dir.exists():
        logger.warning(f"raw_data_dir not present: {raw_data_dir}")
        return {u: RawVolumeEntry(u, 0, 0, LAYOUT) for u in units}

    files_by_unit: Dict[str, List[Path]] = defaultdict(list)
    for path in sorted(raw_data_dir.rglob("*.gz")):
        m = _YEAR_RE.search(path.name)
        if not m:
            continue
        year = int(m.group(1))
        for u in units:
            b = _slice_bounds(u)
            if b is not None and b[0] <= year <= b[1]:
                files_by_unit[u].append(path)

    out: Dict[str, RawVolumeEntry] = {}
    for u in units:
        files = files_by_unit.get(u, [])
        out[u] = RawVolumeEntry(
            unit_name=u,
            n_files=len(files),
            n_bytes=sum(p.stat().st_size for p in files),
            layout_hint=LAYOUT,
        )
    return out
```

```python
# scripts/data_prep/raw_volume/ngram_en.py
"""English Google Ngram raw-data walker.

Layout: {raw_data_dir}/googlebooks-eng-*-5gram-*-YYYY.gz (year is the LAST
4-digit run in the filename).
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from scripts.common.dataset_stats import RawVolumeEntry

LAYOUT = "{raw_data_dir}/googlebooks-eng-*-5gram-*-YYYY.gz"
_YEAR_RE = re.compile(r"(\d{4})")


def _slice_bounds(unit: str) -> Tuple[int, int] | None:
    m = re.match(r"^(\d{4})_(\d{4})$", unit)
    return (int(m.group(1)), int(m.group(2))) if m else None


def walk(
    raw_data_dir: Path,
    units: List[str],
    config: dict,
    logger,
) -> Dict[str, RawVolumeEntry]:
    raw_data_dir = Path(raw_data_dir)
    if not raw_data_dir.exists():
        logger.warning(f"raw_data_dir not present: {raw_data_dir}")
        return {u: RawVolumeEntry(u, 0, 0, LAYOUT) for u in units}

    files_by_unit: Dict[str, List[Path]] = defaultdict(list)
    for path in sorted(raw_data_dir.rglob("*.gz")):
        # En filenames have a date stamp too (e.g. 20120701-…-1990); take the LAST
        # 4-digit run so we land on the year, not the export date.
        years = _YEAR_RE.findall(path.name)
        if not years:
            continue
        year = int(years[-1])
        for u in units:
            b = _slice_bounds(u)
            if b is not None and b[0] <= year <= b[1]:
                files_by_unit[u].append(path)

    out: Dict[str, RawVolumeEntry] = {}
    for u in units:
        files = files_by_unit.get(u, [])
        out[u] = RawVolumeEntry(
            unit_name=u,
            n_files=len(files),
            n_bytes=sum(p.stat().st_size for p in files),
            layout_hint=LAYOUT,
        )
    return out
```

- [ ] **Step 4: Register the walkers**

Modify `scripts/data_prep/raw_volume/__init__.py`:

```python
from scripts.data_prep.raw_volume.rmrb import walk as _walk_rmrb
from scripts.data_prep.raw_volume.provincial_newspaper import walk as _walk_newspaper
from scripts.data_prep.raw_volume.weibo import walk as _walk_weibo
from scripts.data_prep.raw_volume.ngram_zh import walk as _walk_ngram_zh
from scripts.data_prep.raw_volume.ngram_en import walk as _walk_ngram_en

WALKERS = {
    "renminribao": _walk_rmrb,
    "newspaper": _walk_newspaper,
    "weibo": _walk_weibo,
    # ngram is dispatched by (language, data_source) — see resolve_walker()
    # in scripts.common.dataset_stats. The registry uses synthetic keys.
    "ngram_zh": _walk_ngram_zh,
    "ngram_en": _walk_ngram_en,
}
```

- [ ] **Step 5: Run tests to confirm they pass**

```bash
python -m pytest tests/test_raw_volume_ngram.py -v
```

Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add scripts/data_prep/raw_volume/ngram_zh.py scripts/data_prep/raw_volume/ngram_en.py scripts/data_prep/raw_volume/__init__.py tests/test_raw_volume_ngram.py
git commit -m "raw_volume: ngram walkers (zh + en), year inferred from filename"
```

---

## Task 9: COHA walker

**Files:**
- Create: `scripts/data_prep/raw_volume/coha.py`
- Create: `tests/test_raw_volume_coha.py`
- Modify: `scripts/data_prep/raw_volume/__init__.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_raw_volume_coha.py
"""Tests for the COHA raw-data walker."""

from __future__ import annotations

import logging
from pathlib import Path

from scripts.data_prep.raw_volume import WALKERS
from scripts.data_prep.raw_volume.coha import walk

logger = logging.getLogger("test")


def _make_coha_tree(root: Path, files_by_decade: dict[str, int]) -> None:
    """Mimic COHA's text_NNNNs/ layout used by build_corpora_coha."""
    for decade, n in files_by_decade.items():
        d = root / f"text_{decade}"
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            (d / f"doc_{i:04d}.txt").write_text("z" * 80, encoding="utf-8")


def test_walker_registered():
    assert "coha" in WALKERS
    assert WALKERS["coha"] is walk


def test_groups_by_decade(tmp_path):
    _make_coha_tree(tmp_path, {"1940": 3, "1950": 5, "1960": 2})
    result = walk(tmp_path, ["1940s", "1950s", "1960s", "1970s"], {}, logger)
    assert result["1940s"].n_files == 3
    assert result["1950s"].n_files == 5
    assert result["1960s"].n_files == 2
    assert result["1970s"].n_files == 0
    assert result["1940s"].n_bytes == 3 * 80


def test_missing_raw_dir(tmp_path):
    result = walk(tmp_path / "nope", ["1940s"], {}, logger)
    assert result["1940s"].n_files == 0
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_raw_volume_coha.py -v
```

Expected: `ModuleNotFoundError: scripts.data_prep.raw_volume.coha`.

- [ ] **Step 3: Create the COHA walker**

```python
# scripts/data_prep/raw_volume/coha.py
"""COHA raw-data walker.

Layout: {raw_data_dir}/text_{decade}/*.txt
Units are decades (e.g. '1940s').
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

from scripts.common.dataset_stats import RawVolumeEntry

LAYOUT = "{raw_data_dir}/text_{decade}/*.txt"


def walk(
    raw_data_dir: Path,
    units: List[str],
    config: dict,
    logger,
) -> Dict[str, RawVolumeEntry]:
    raw_data_dir = Path(raw_data_dir)
    if not raw_data_dir.exists():
        logger.warning(f"raw_data_dir not present: {raw_data_dir}")
        return {u: RawVolumeEntry(u, 0, 0, LAYOUT) for u in units}

    out: Dict[str, RawVolumeEntry] = {}
    for u in units:
        m = re.match(r"^(\d{4})s$", u)
        if not m:
            out[u] = RawVolumeEntry(u, 0, 0, LAYOUT)
            continue
        decade = m.group(1)
        decade_dir = raw_data_dir / f"text_{decade}"
        files = sorted(decade_dir.glob("*.txt")) if decade_dir.exists() else []
        out[u] = RawVolumeEntry(
            unit_name=u,
            n_files=len(files),
            n_bytes=sum(p.stat().st_size for p in files),
            layout_hint=LAYOUT,
        )
    return out
```

- [ ] **Step 4: Register the walker**

Modify `scripts/data_prep/raw_volume/__init__.py`:

```python
from scripts.data_prep.raw_volume.rmrb import walk as _walk_rmrb
from scripts.data_prep.raw_volume.provincial_newspaper import walk as _walk_newspaper
from scripts.data_prep.raw_volume.weibo import walk as _walk_weibo
from scripts.data_prep.raw_volume.ngram_zh import walk as _walk_ngram_zh
from scripts.data_prep.raw_volume.ngram_en import walk as _walk_ngram_en
from scripts.data_prep.raw_volume.coha import walk as _walk_coha

WALKERS = {
    "renminribao": _walk_rmrb,
    "newspaper": _walk_newspaper,
    "weibo": _walk_weibo,
    "ngram_zh": _walk_ngram_zh,
    "ngram_en": _walk_ngram_en,
    "coha": _walk_coha,
}
```

- [ ] **Step 5: Run tests to confirm they pass**

```bash
python -m pytest tests/test_raw_volume_coha.py -v
```

Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add scripts/data_prep/raw_volume/coha.py scripts/data_prep/raw_volume/__init__.py tests/test_raw_volume_coha.py
git commit -m "raw_volume: COHA walker (decade dirs)"
```

---

## Task 10: CLI shim + `run()` orchestrator

**Files:**
- Create: `scripts/describe_dataset.py`
- Modify: `scripts/common/dataset_stats.py` (add `resolve_walker` + `run`)
- Create: `tests/test_describe_dataset.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_describe_dataset.py
"""End-to-end smoke test for scripts.describe_dataset.

Builds a tiny corpus + raw-data tree, runs ``run(config, ...)``, and asserts
the Markdown lands at the right path with all four sections.
"""

from __future__ import annotations

import logging
import sys
import types
from pathlib import Path

import pytest

logger = logging.getLogger("test")


def _install_fake_gensim():
    if "gensim" in sys.modules and not getattr(sys.modules["gensim"], "_fake", False):
        try:
            from gensim.models import KeyedVectors  # noqa: F401
            return
        except Exception:
            pass
    fake_gensim = types.ModuleType("gensim")
    fake_gensim._fake = True  # type: ignore[attr-defined]
    fake_models = types.ModuleType("gensim.models")

    class _FakeKV:
        @staticmethod
        def load(path):
            # Same vocab list as the shim in tests/test_dataset_stats.py.
            # Both files install module-level fakes; pytest collects every
            # test module before running, so whichever shim is installed last
            # wins. Keep these in sync so per-test-file order doesn't matter.
            class _Stub:
                index_to_key = ["a", "b", "c"]
            return _Stub()

    fake_models.KeyedVectors = _FakeKV  # type: ignore[attr-defined]
    fake_gensim.models = fake_models  # type: ignore[attr-defined]
    sys.modules["gensim"] = fake_gensim
    sys.modules["gensim.models"] = fake_models


_install_fake_gensim()

from scripts.common.dataset_stats import run, resolve_walker


def _make_rmrb_setup(tmp_path: Path) -> tuple[Path, dict]:
    """Set up corpora + models + raw layout for a 2-slice RMRB run."""
    corpora = tmp_path / "corpora"
    models = tmp_path / "models"
    raw = tmp_path / "raw"
    results = tmp_path / "results"
    for d in (corpora, models, raw, results):
        d.mkdir(parents=True, exist_ok=True)

    # Two units' corpus files.
    (corpora / "1940_1949").mkdir()
    (corpora / "1940_1949" / "corpus_000000").write_text("a b c\na b\n", encoding="utf-8")
    (corpora / "1950_1959").mkdir()
    (corpora / "1950_1959" / "corpus_000000").write_text("c d\n", encoding="utf-8")

    # Two stub model files.
    (models / "rmrb_1940_1949.model").write_text("stub")
    (models / "rmrb_1950_1959.model").write_text("stub")

    # RMRB raw layout.
    rmrb_dir = raw / "1940s" / "1942" / "报刊" / "人民日报"
    rmrb_dir.mkdir(parents=True)
    (rmrb_dir / "rmrb_1942_01.txt").write_text("x" * 100, encoding="utf-8")

    config = {
        "language": "zh",
        "data_source": "renminribao",
        "embedding": {"vector_size": 300, "window": 4, "min_count": 50, "sg": 1,
                      "negative": 15, "epochs": 5, "seed": 42, "workers": 16,
                      "model_name_template": "rmrb_{slice_name}.model"},
        "time_slices": {"start_year": 1940, "end_year": 1959,
                        "window_size": 10, "step_size": 10},
        "paths": {
            "corpora_dir": str(corpora),
            "models_dir": str(models),
            "raw_data_dir": str(raw),
            "results_dir": str(results),
        },
    }
    return results, config


class TestResolveWalker:
    def test_renminribao(self):
        assert resolve_walker({"data_source": "renminribao"}) is not None

    def test_ngram_zh(self):
        w = resolve_walker({"data_source": "ngram", "language": "zh"})
        assert w is not None
        from scripts.data_prep.raw_volume.ngram_zh import walk
        assert w is walk

    def test_ngram_en(self):
        w = resolve_walker({"data_source": "ngram", "language": "en"})
        assert w is not None
        from scripts.data_prep.raw_volume.ngram_en import walk
        assert w is walk

    def test_unknown_source(self):
        assert resolve_walker({"data_source": "what"}) is None


class TestRun:
    def test_writes_markdown_with_all_sections(self, tmp_path):
        results, config = _make_rmrb_setup(tmp_path)
        run(config, config_path="x.yml")
        md_path = results / "dataset_summary.md"
        assert md_path.exists()
        md = md_path.read_text(encoding="utf-8")
        assert "# Dataset Summary — renminribao (zh)" in md
        assert "## Corpus totals" in md
        assert "## Raw data" in md
        assert "## Training" in md
        assert "## Per-unit breakdown" in md
        assert "1940_1949" in md
        assert "1950_1959" in md

    def test_force_recomputes(self, tmp_path):
        results, config = _make_rmrb_setup(tmp_path)
        run(config, config_path="x.yml")
        cache = Path(config["paths"]["corpora_dir"]) / "1940_1949" / ".dataset_stats.json"
        assert cache.exists()
        # Run again with force — should re-write (mtime bumps).
        first_mtime = cache.stat().st_mtime
        import time; time.sleep(0.01)
        run(config, config_path="x.yml", force=True)
        assert cache.stat().st_mtime >= first_mtime

    def test_no_raw_skips_walker(self, tmp_path):
        results, config = _make_rmrb_setup(tmp_path)
        run(config, config_path="x.yml", no_raw=True)
        md = (results / "dataset_summary.md").read_text(encoding="utf-8")
        # Raw bytes cell becomes n/a when no_raw skipped.
        assert "n/a" in md

    def test_unit_filter(self, tmp_path):
        results, config = _make_rmrb_setup(tmp_path)
        run(config, config_path="x.yml", units="1940_1949")
        md = (results / "dataset_summary.md").read_text(encoding="utf-8")
        assert "1940_1949" in md
        # 1950_1959 should be absent from per-unit table.
        assert "1950_1959" not in md
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_describe_dataset.py -v
```

Expected: `ImportError: cannot import name 'run' from 'scripts.common.dataset_stats'`.

- [ ] **Step 3: Add `resolve_walker` and `run` to `scripts/common/dataset_stats.py`**

Append to `scripts/common/dataset_stats.py`:

```python
from scripts.common.config_loader import get_model_name


def resolve_walker(config: dict):
    """Pick the right raw-data walker for this config's data_source.

    Ngram dispatches on language too (ngram_zh vs ngram_en).
    Returns None if no walker is registered for this source.
    """
    from scripts.data_prep.raw_volume import WALKERS

    src = config.get("data_source")
    if src == "ngram":
        key = f"ngram_{config.get('language', '')}"
        return WALKERS.get(key)
    return WALKERS.get(src)


def run(
    config: dict,
    config_path: str,
    force: bool = False,
    units: Optional[str] = None,
    no_raw: bool = False,
    no_model: bool = False,
    vocab_union: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Path:
    """Orchestrate one source's dataset summary; write Markdown; return its path.

    Args:
        config: parsed config dict (already validated).
        config_path: path string for provenance line in the Markdown header.
        force: ignore sidecar caches, rescan every unit.
        units: optional comma-separated subset of unit names to include.
        no_raw: skip the raw-data walker (per-unit raw cells become n/a).
        no_model: skip per-unit model vocab introspection.
        vocab_union: opt-in cross-unit raw-vocab union (bypasses cache).
        logger: optional pre-built logger; default uses a console-only one.
    """
    if logger is None:
        logger = logging.getLogger("describe_dataset")
        if not logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
            logger.addHandler(h)
            logger.setLevel(logging.INFO)

    unit_list = discover_units(config)
    if units:
        wanted = set(u.strip() for u in units.split(","))
        unit_list = [u for u in unit_list if u in wanted]
    if not unit_list:
        logger.error(f"No units to describe under {config['paths']['corpora_dir']}")
        raise SystemExit(1)
    logger.info(f"Describing {len(unit_list)} unit(s): {', '.join(unit_list)}")

    corpora_dir = Path(config["paths"]["corpora_dir"])
    models_dir = Path(config["paths"]["models_dir"])

    # Raw-data walk (once per source).
    raw_by_unit: Dict[str, Optional[RawVolumeEntry]] = {u: None for u in unit_list}
    if not no_raw:
        walker = resolve_walker(config)
        if walker is None:
            logger.warning(f"No raw-data walker for data_source={config.get('data_source')!r}; "
                           "raw cells will be n/a")
        else:
            raw_dir = Path(config["paths"].get("raw_data_dir", ""))
            raw_by_unit = {u: e for u, e in walker(raw_dir, unit_list, config, logger).items()}

    # Per-unit corpus + model.
    per_unit: Dict[str, PerUnitTriple] = {}
    vocab_union_set: Optional[Set[str]] = set() if vocab_union else None
    for u in unit_list:
        unit_dir = corpora_dir / u
        if vocab_union:
            stats, vocab = scan_corpus_unit(unit_dir, logger, force=force, return_vocab=True)
            vocab_union_set.update(vocab)
        else:
            stats = scan_corpus_unit(unit_dir, logger, force=force)
        mv = None if no_model else model_vocab_size(models_dir / get_model_name(u, config), logger)
        per_unit[u] = (stats, mv, raw_by_unit.get(u))

    totals = aggregate_source(
        per_unit,
        vocab_union_count=len(vocab_union_set) if vocab_union_set is not None else None,
    )

    generated_at = datetime.datetime.now().strftime("%Y-%m-%d")
    md = render_markdown(totals, per_unit, config, config_path, generated_at)
    results_dir = Path(config["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "dataset_summary.md"
    out_path.write_text(md, encoding="utf-8")
    logger.info(f"Wrote {out_path}")
    return out_path
```

- [ ] **Step 4: Create the CLI shim**

```python
# scripts/describe_dataset.py
#!/usr/bin/env python3
"""Dataset & training summary reporter — one Markdown per source.

Produces ``<results_dir>/dataset_summary.md`` with corpus totals, raw-data
footprint, training hyperparameters, and a per-unit breakdown. Reuses the
existing per-source profiles in ``config/profiles/``.

Usage:
    python -m scripts.describe_dataset --config=config/profiles/garg_weat_renminribao.yml
    python -m scripts.describe_dataset --config=…  --force
    python -m scripts.describe_dataset --config=…  --units=1940_1949,1950_1959
    python -m scripts.describe_dataset --config=…  --no-raw
    python -m scripts.describe_dataset --config=…  --no-model
    python -m scripts.describe_dataset --config=…  --vocab-union   # opt-in; bypasses cache
"""

from __future__ import annotations

from pathlib import Path

import fire

from scripts.common.config_loader import load_config
from scripts.common.dataset_stats import run as describe
from scripts.common.logging_utils import setup_logging


def main(
    config: str,
    force: bool = False,
    units: str = "",
    no_raw: bool = False,
    no_model: bool = False,
    vocab_union: bool = False,
) -> None:
    cfg = load_config(config)
    log_dir = Path(cfg["paths"].get("log_dir", cfg["paths"]["results_dir"]))
    logger = setup_logging(log_dir, "describe_dataset.log")
    describe(
        cfg,
        config_path=config,
        force=force,
        units=units or None,
        no_raw=no_raw,
        no_model=no_model,
        vocab_union=vocab_union,
        logger=logger,
    )


if __name__ == "__main__":
    fire.Fire(main)
```

- [ ] **Step 5: Run tests to confirm they pass**

```bash
python -m pytest tests/test_describe_dataset.py -v
```

Expected: 8 passed (4 TestResolveWalker + 4 TestRun).

- [ ] **Step 6: Commit**

```bash
git add scripts/describe_dataset.py scripts/common/dataset_stats.py tests/test_describe_dataset.py
git commit -m "describe_dataset: CLI + run() orchestrator (cache, --force, --units, --no-raw, --no-model, --vocab-union)"
```

---

## Task 11: Full-suite regression sweep

**Files:** none (verification only)

- [ ] **Step 1: Run the full new test surface**

```bash
python -m pytest tests/test_dataset_stats.py tests/test_raw_volume_rmrb.py tests/test_raw_volume_provincial_newspaper.py tests/test_raw_volume_weibo.py tests/test_raw_volume_ngram.py tests/test_raw_volume_coha.py tests/test_describe_dataset.py -v
```

Expected: 44 passed (25 + 4 + 4 + 3 + 5 + 3 + 8 = 52 — recount if numbers shift after final implementation; key check is ZERO failures).

- [ ] **Step 2: Run the broader test surface to confirm no regression**

```bash
python -m pytest tests/ -v
```

Expected: all pre-existing tests still pass; only the new ones added.

- [ ] **Step 3: Sanity-check the CLI with a `--help`-style invocation (Fire prints arg signature)**

```bash
python -m scripts.describe_dataset --help 2>&1 | head -20
```

Expected: Fire prints a usage block listing `--config`, `--force`, `--units`, `--no-raw`, `--no-model`, `--vocab-union`.

- [ ] **Step 4: Commit (no-op marker if nothing changed)**

If Steps 1–3 produced no edits, skip. Otherwise:

```bash
git add -A
git commit -m "describe_dataset: regression sweep + CLI sanity check"
```

---

## Done condition

After Task 11:

- `python -m scripts.describe_dataset --config=config/profiles/garg_weat_renminribao.yml` on a real host produces `/scratch/network/yh6580/gender-occup/results_garg_weat_renminribao/dataset_summary.md` with all four sections.
- Same against `garg_weat_provincial_newspaper.yml` on the PKU server produces the equivalent Markdown for the provincial newspaper arm.
- Both invocations are near-instant on a second run (cache warm).
- 52 unit tests pass; no regression in the rest of the suite.
