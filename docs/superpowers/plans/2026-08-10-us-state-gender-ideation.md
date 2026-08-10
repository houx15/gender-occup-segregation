# US State-Level Gender-Ideation Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce embedding-based gender-ideology indicators at US state × year granularity from two sources (American Stories 1900–1960 by decade; 3DLNews2 1996–2020), and render them as per-year US choropleths.

**Architecture:** Two independent arms mirroring the existing `provincial_newspaper` arm. New code covers only acquisition, state assignment, corpus building (with wire-copy dedup), and a US choropleth; training (`train_embeddings.py`) and bias analysis (`analyze_category_bias.py`) are reused unchanged. Each corpus is written to `corpora_dir/{state}_{year}/`, so a unit directory name *is* the `{state}_{year}` unit that `discover_units` / `discover_models` already parse.

**Tech Stack:** Python 3.12, `fire` CLIs, `gensim` Word2Vec (existing), `nltk` en tokenizer (existing), HuggingFace `datasets`, `globus-cli`, `geopandas` + matplotlib, `pytest`.

## Global Constraints

- Language: `en` throughout. Preprocess via `scripts.common.preprocessing.preprocess(language="en", tokenizer="nltk_en", stopwords_key="en_default", lowercase=True)`.
- Unit naming: `{state}_{year}`, state as canonical shapefile name lowercased with spaces→`_` (e.g. `new_york_1940`, `california_1996`). One writer directory per unit.
- Arm years: American Stories `[1900,1910,1920,1930,1940,1950,1960]`; 3DLNews2 `[1996,2000,2010,2020]`.
- `ideation_sign`: leadership `+1`, science `+1`, family `-1` (matches existing English garg configs).
- `metrics: [rnd, cohens_d]`; `model_name_template: "model_{unit_name}.kv"`.
- Wordlists: `wordlists/en/garg_weat/` (`gender_words.json`, `cleaned_{leadership,family,science}.txt`).
- Base dir convention: `/scratch/network/yh6580/gender-occup` (US arms live under it).
- No raw modern-news redistribution — only derived artifacts leave controlled storage.
- Every drop/skip (below-threshold unit, unresolved LCCN, empty year) is logged; no silent truncation.
- Follow existing patterns: `fire.Fire(main)` entrypoints, `load_config`, `setup_logging`.
- Commit after each task when tests are green; stage only that task's files; stay on `main`; do not push.

---

## File Structure

**Create:**
- `scripts/data_prep/us_state_mapper.py` — state-name normalizer + LCCN→state table + `resolve_state`.
- `scripts/data_prep/dedup.py` — wire-copy deduper (exact + MinHash/LSH shingle).
- `scripts/data_prep/build_corpora_us.py` — raw JSONL → per-`state_year` corpora + coverage report.
- `scripts/data_prep/download_american_stories.py` — HF `subset_years` → raw article JSONL.
- `scripts/data_prep/download_dlnews.py` — Globus transfer of 3DLNews2 newspaper slices.
- `scripts/data_prep/fetch_us_shapefile.py` — fetch Census `cb_20m` states shapefile.
- `config/profiles/garg_weat_american_stories.yml`, `config/profiles/garg_weat_dlnews.yml`.
- `slurm/prepare_us_data.slurm`, `slurm/train_us.slurm`, `slurm/garg_weat_us.slurm`.
- Tests: `tests/test_us_state_mapper.py`, `tests/test_dedup.py`, `tests/test_build_corpora_us.py`, `tests/test_us_choropleth.py`.

**Modify:**
- `scripts/visualize.py` — add `_state_year_parse`, `_match_state_in_shapefile`, `plot_us_choropleth`.

---

## Task 1: State mapper (normalizer + LCCN→state + resolve)

**Files:**
- Create: `scripts/data_prep/us_state_mapper.py`
- Test: `tests/test_us_state_mapper.py`

**Interfaces:**
- Produces:
  - `normalize_state(raw: str) -> str | None` — any of full name / USPS 2-letter / mixed case → canonical Title-Case name (`"california"`,`"CA"`,`"California "` → `"California"`); unknown → `None`.
  - `unit_state(state_name: str) -> str` — canonical name → unit token (`"New York"` → `"new_york"`).
  - `lccn_from_article_id(article_id: str) -> str | None` — extract the LCCN embedded in an American Stories article/scan id.
  - `build_lccn_state_table(directory_records: list[dict]) -> dict[str, str]` — LoC directory records → `{lccn: canonical_state}`.
  - `load_lccn_state_table(path: str) -> dict[str, str]`, `save_lccn_state_table(table, path)`.
  - `resolve_state(lccn: str, table: dict[str, str]) -> str | None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_us_state_mapper.py
import json
from scripts.data_prep import us_state_mapper as m


def test_normalize_state_accepts_name_usps_and_messy_case():
    assert m.normalize_state("California") == "California"
    assert m.normalize_state("california") == "California"
    assert m.normalize_state("  CA ") == "California"
    assert m.normalize_state("New york") == "New York"
    assert m.normalize_state("District of Columbia") == "District of Columbia"
    assert m.normalize_state("DC") == "District of Columbia"
    assert m.normalize_state("Freedonia") is None
    assert m.normalize_state("") is None


def test_unit_state_tokenizes():
    assert m.unit_state("California") == "california"
    assert m.unit_state("New York") == "new_york"
    assert m.unit_state("District of Columbia") == "district_of_columbia"


def test_lccn_from_article_id():
    # American Stories ids embed the LCCN of the source title.
    assert m.lccn_from_article_id("sn83030214_1940-01-02_p1_a3") == "sn83030214"
    assert m.lccn_from_article_id("2012271201-1950-06-05-seq1-1") == "2012271201"
    assert m.lccn_from_article_id("no-lccn-here") is None


def test_build_and_resolve_state_table():
    records = [
        {"lccn": "sn83030214", "state": "New York"},
        {"lccn": "sn84020000", "state": "CA"},
        {"lccn": "sn00000001", "state": "Freedonia"},  # unknown -> dropped
    ]
    table = m.build_lccn_state_table(records)
    assert table == {"sn83030214": "New York", "sn84020000": "California"}
    assert m.resolve_state("sn83030214", table) == "New York"
    assert m.resolve_state("sn99999999", table) is None


def test_table_roundtrip(tmp_path):
    table = {"sn83030214": "New York"}
    p = tmp_path / "lccn_state.json"
    m.save_lccn_state_table(table, str(p))
    assert m.load_lccn_state_table(str(p)) == table
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_us_state_mapper.py -v`
Expected: FAIL with `ModuleNotFoundError` / `AttributeError`.

- [ ] **Step 3: Write the implementation**

```python
# scripts/data_prep/us_state_mapper.py
#!/usr/bin/env python3
"""US state identity: normalize names, extract LCCNs, map LCCN -> state.

Single source of truth for state identity across both US arms and the
choropleth. Arm B (3DLNews2) uses only ``normalize_state`` (its ``location.state``
is authoritative); Arm A (American Stories) additionally maps each article's
LCCN to a publisher state via a table built from the LoC US Newspaper Directory.
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional

import fire

from scripts.common.config_loader import load_config
from scripts.common.logging_utils import setup_logging

# Canonical name spelling matches the Census cb_20m shapefile NAME column.
_STATE_NAME_TO_USPS: Dict[str, str] = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "District of Columbia": "DC", "Florida": "FL", "Georgia": "GA", "Hawaii": "HI",
    "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
    "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI",
    "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX",
    "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
    "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
}
_USPS_TO_STATE_NAME: Dict[str, str] = {v: k for k, v in _STATE_NAME_TO_USPS.items()}
_LOWER_NAME_TO_CANON: Dict[str, str] = {k.lower(): k for k in _STATE_NAME_TO_USPS}

_LCCN_RE = re.compile(r"(sn\d{8}|\d{10})")


def normalize_state(raw: str) -> Optional[str]:
    """Full name / USPS 2-letter / messy case -> canonical Title-Case name."""
    if not raw or not isinstance(raw, str):
        return None
    s = raw.strip()
    if not s:
        return None
    up = s.upper()
    if up in _USPS_TO_STATE_NAME:
        return _USPS_TO_STATE_NAME[up]
    return _LOWER_NAME_TO_CANON.get(s.lower())


def unit_state(state_name: str) -> str:
    """Canonical name -> unit token: 'New York' -> 'new_york'."""
    return state_name.strip().lower().replace(" ", "_")


def lccn_from_article_id(article_id: str) -> Optional[str]:
    """Extract the LCCN (sn######## or a 10-digit id) embedded in an id string."""
    if not article_id:
        return None
    m = _LCCN_RE.search(str(article_id))
    return m.group(1) if m else None


def build_lccn_state_table(directory_records: List[dict]) -> Dict[str, str]:
    """LoC directory records -> {lccn: canonical_state}. Drops unknown states."""
    table: Dict[str, str] = {}
    for rec in directory_records:
        lccn = (rec.get("lccn") or "").strip()
        state = normalize_state(rec.get("state") or "")
        if lccn and state:
            table[lccn] = state
    return table


def resolve_state(lccn: str, table: Dict[str, str]) -> Optional[str]:
    return table.get(lccn) if lccn else None


def save_lccn_state_table(table: Dict[str, str], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(table, f, ensure_ascii=False, indent=0, sort_keys=True)


def load_lccn_state_table(path: str) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _fetch_loc_directory_records() -> List[dict]:
    """Fetch LoC US Newspaper Directory records: [{lccn, state}, ...].

    Paginates the loc.gov directory JSON API. Network step — run where the node
    has internet. Kept out of the pure-logic functions above so they stay
    unit-testable without network.
    """
    import requests

    records: List[dict] = []
    base = "https://www.loc.gov/collections/directory-of-us-newspapers-in-american-libraries/"
    page = 1
    while True:
        resp = requests.get(base, params={"fo": "json", "c": 500, "sp": page}, timeout=60)
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if not results:
            break
        for r in results:
            lccn = r.get("number_lccn", [None])
            lccn = lccn[0] if isinstance(lccn, list) else lccn
            loc = r.get("location_state", [None])
            loc = loc[0] if isinstance(loc, list) else loc
            if lccn and loc:
                records.append({"lccn": lccn, "state": loc})
        page += 1
    return records


def build(config: str = "config/config.yml") -> None:
    """Fetch the LoC directory and write the LCCN->state table to raw_data_dir."""
    cfg = load_config(config)
    logger = setup_logging(cfg["paths"]["log_dir"], "us_state_mapper.log")
    out = f"{cfg['paths']['raw_data_dir']}/lccn_state_table.json"
    logger.info("Fetching LoC US Newspaper Directory records...")
    records = _fetch_loc_directory_records()
    table = build_lccn_state_table(records)
    save_lccn_state_table(table, out)
    logger.info(f"Wrote {len(table)} LCCN->state entries to {out}")


if __name__ == "__main__":
    fire.Fire({"build": build})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_us_state_mapper.py -v`
Expected: PASS (all 5 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/data_prep/us_state_mapper.py tests/test_us_state_mapper.py
git commit -m "feat(us): state-name normalizer + LCCN->state mapper"
```

---

## Task 2: Wire-copy deduper (exact + MinHash/LSH shingle)

**Files:**
- Create: `scripts/data_prep/dedup.py`
- Test: `tests/test_dedup.py`

**Interfaces:**
- Produces:
  - `normalize_for_hash(text: str) -> str` — lowercase, collapse whitespace, strip non-alphanumerics.
  - `Deduper(method="shingle", shingle_k=8, n_perm=64, bands=16, seed=42)` with:
    - `is_duplicate(text: str) -> bool` — True if a near/exact duplicate of a previously-seen text; records the text either way.
    - `reset() -> None` — clear seen state (called per year slice when `scope="within_year"`).
- Consumes: nothing (pure, dependency-free).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dedup.py
from scripts.data_prep.dedup import Deduper, normalize_for_hash


def test_normalize_for_hash():
    assert normalize_for_hash("The  QUICK, brown fox!") == "the quick brown fox"


def test_exact_dedup_catches_identical_only():
    d = Deduper(method="exact")
    a = "Senate passes the new farm bill today in Washington."
    assert d.is_duplicate(a) is False          # first sighting
    assert d.is_duplicate(a) is True           # exact repeat
    assert d.is_duplicate(a + " Extra clause.") is False  # exact = not a dup


def test_shingle_dedup_catches_near_duplicate_wire_copy():
    d = Deduper(method="shingle", shingle_k=4, n_perm=64, bands=16)
    base = ("washington the senate approved a sweeping new farm bill on tuesday "
            "sending the measure to the house for final consideration next week")
    near = ("washington the senate approved a sweeping new farm bill on tuesday "
            "sending the measure to the house for a final vote next week")  # minor edit
    far = ("local high school students won the regional science fair with a "
           "project on solar powered water purification for rural communities")
    assert d.is_duplicate(base) is False
    assert d.is_duplicate(near) is True   # near-dup wire copy -> caught
    assert d.is_duplicate(far) is False   # unrelated -> kept


def test_reset_clears_state():
    d = Deduper(method="exact")
    a = "same story"
    assert d.is_duplicate(a) is False
    d.reset()
    assert d.is_duplicate(a) is False  # after reset, first sighting again
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_dedup.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

```python
# scripts/data_prep/dedup.py
#!/usr/bin/env python3
"""Wire-copy deduplication for US news corpora.

Syndicated copy (the same AP story printed in many states) inflates per-state
signal. This module removes it. Two methods:
  * ``exact``   — drop repeats sharing a normalized-text hash. O(n), robust.
  * ``shingle`` — MinHash + LSH near-duplicate detection: catches wire copy with
                  minor edits. Dependency-free (hashlib-based permutations).

Scope is controlled by the caller: for ``within_year`` scope, call ``reset()``
between year slices so only same-year cross-state duplicates collapse.
"""

from __future__ import annotations

import hashlib
import re
from typing import Dict, List, Set

_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]+")
_WS_RE = re.compile(r"\s+")
_MERSENNE = (1 << 61) - 1  # large prime for hash permutations


def normalize_for_hash(text: str) -> str:
    """Lowercase, drop non-alphanumerics, collapse whitespace."""
    s = (text or "").lower()
    s = _NON_ALNUM_RE.sub(" ", s)
    return _WS_RE.sub(" ", s).strip()


def _stable_hash(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "big")


class Deduper:
    """Track seen documents; report whether a new one duplicates an old one."""

    def __init__(self, method: str = "shingle", shingle_k: int = 8,
                 n_perm: int = 64, bands: int = 16, seed: int = 42):
        if method not in ("exact", "shingle"):
            raise ValueError(f"unknown dedup method: {method!r}")
        if method == "shingle" and n_perm % bands != 0:
            raise ValueError("n_perm must be divisible by bands")
        self.method = method
        self.shingle_k = shingle_k
        self.n_perm = n_perm
        self.bands = bands
        self.rows_per_band = n_perm // bands
        # deterministic (a, b) permutation coefficients
        rng = _stable_hash(f"seed:{seed}")
        self._ab = []
        for i in range(n_perm):
            a = (_stable_hash(f"a:{seed}:{i}") % (_MERSENNE - 1)) + 1
            b = _stable_hash(f"b:{seed}:{i}") % _MERSENNE
            self._ab.append((a, b))
        self.reset()

    def reset(self) -> None:
        self._exact_seen: Set[str] = set()
        self._lsh_buckets: Dict[int, Set[int]] = {}
        self._next_id = 0

    def _minhash(self, text: str) -> List[int]:
        toks = normalize_for_hash(text).split()
        if len(toks) < self.shingle_k:
            shingles = {" ".join(toks)} if toks else {""}
        else:
            shingles = {
                " ".join(toks[i:i + self.shingle_k])
                for i in range(len(toks) - self.shingle_k + 1)
            }
        hvals = [_stable_hash(s) for s in shingles]
        sig = []
        for a, b in self._ab:
            sig.append(min(((a * h + b) % _MERSENNE) for h in hvals))
        return sig

    def _band_keys(self, sig: List[int]) -> List[int]:
        keys = []
        for band in range(self.bands):
            chunk = tuple(sig[band * self.rows_per_band:(band + 1) * self.rows_per_band])
            keys.append(_stable_hash(f"{band}:{chunk}"))
        return keys

    def is_duplicate(self, text: str) -> bool:
        if self.method == "exact":
            key = hashlib.blake2b(
                normalize_for_hash(text).encode("utf-8"), digest_size=16
            ).hexdigest()
            if key in self._exact_seen:
                return True
            self._exact_seen.add(key)
            return False

        # shingle: LSH banding — a collision in any band => candidate duplicate.
        sig = self._minhash(text)
        keys = self._band_keys(sig)
        for k in keys:
            if k in self._lsh_buckets and self._lsh_buckets[k]:
                # any prior doc shares a band -> treat as near-duplicate
                return True
        for k in keys:
            self._lsh_buckets.setdefault(k, set()).add(self._next_id)
        self._next_id += 1
        return False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_dedup.py -v`
Expected: PASS. If `test_shingle_dedup_catches_near_duplicate_wire_copy` is flaky at the boundary, lower `bands` to 8 (higher recall) — bands/rows trade precision for recall; document the chosen value in the config comment.

- [ ] **Step 5: Commit**

```bash
git add scripts/data_prep/dedup.py tests/test_dedup.py
git commit -m "feat(us): wire-copy deduper (exact + MinHash/LSH shingle)"
```

---

## Task 3: US corpus builder (routing + preprocess + dedup + coverage)

**Files:**
- Create: `scripts/data_prep/build_corpora_us.py`
- Test: `tests/test_build_corpora_us.py`

**Interfaces:**
- Consumes: `us_state_mapper.{normalize_state, unit_state, lccn_from_article_id, resolve_state, load_lccn_state_table}`, `dedup.Deduper`, `preprocessing.preprocess`.
- Produces:
  - `UnitCorpusWriter(unit_name, output_dir, max_bytes=1GB)` — rolling `corpus_%06d` writer under `output_dir/{unit_name}/`.
  - `iter_records(arm, raw_dir, year) -> Iterator[dict]` — yields `{"text":..., "state":..., "title":...}` per source arm.
  - `build_corpus(config, logger, arm, max_files=None) -> dict` — writes corpora, returns coverage `{unit_name: n_docs}`.
  - `write_coverage_report(coverage, path)` — CSV `state,year,unit_name,n_docs,kept`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_build_corpora_us.py
import gzip
import json
from pathlib import Path

import pytest

from scripts.data_prep import build_corpora_us as b


def _cfg(tmp_path, arm, min_docs=2, dedup_method="exact"):
    return {
        "language": "en",
        "paths": {
            "raw_data_dir": str(tmp_path / "raw"),
            "corpora_dir": str(tmp_path / "corpora"),
            "log_dir": str(tmp_path / "logs"),
            "results_dir": str(tmp_path / "results"),
        },
        "corpus": {
            "tokenizer": "nltk_en", "stopwords": "en_default",
            "lowercase": True, "min_words": 3,
            "dedup": {"enabled": True, "method": dedup_method,
                      "shingle_k": 4, "scope": "within_year"},
        },
        "us_states": {"years": [1940], "min_documents": min_docs},
        "_arm": arm,
    }


def test_dlnews_records_route_by_inline_state(tmp_path):
    raw = tmp_path / "raw"; raw.mkdir(parents=True)
    rows = [
        {"content": "the senate approved the farm policy reform bill today",
         "location": {"state": "New York"}, "is_news_article": True, "title": "t1"},
        {"content": "governor signed the education funding measure this morning",
         "location": {"state": "NY"}, "is_news_article": True, "title": "t2"},
        {"content": "ignored ad content", "location": {"state": "Freedonia"},
         "is_news_article": True, "title": "t3"},
    ]
    with gzip.open(raw / "preprocessed_google_newspaper_New York_1940.jsonl.gz", "wt",
                   encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    recs = list(b.iter_records("dlnews", str(raw), 1940))
    states = sorted(r["state"] for r in recs)
    assert states == ["New York", "New York"]  # NY normalized; Freedonia dropped


def test_build_corpus_writes_units_and_drops_below_threshold(tmp_path):
    cfg = _cfg(tmp_path, "dlnews", min_docs=2)
    raw = Path(cfg["paths"]["raw_data_dir"]); raw.mkdir(parents=True)
    rows = [
        {"content": "the senate approved the farm policy reform bill today",
         "location": {"state": "New York"}, "is_news_article": True, "title": "a"},
        {"content": "governor signed the education funding measure this morning",
         "location": {"state": "New York"}, "is_news_article": True, "title": "b"},
        {"content": "small county fair drew a modest crowd over the weekend",
         "location": {"state": "Nevada"}, "is_news_article": True, "title": "c"},
    ]
    with gzip.open(raw / "preprocessed_google_newspaper_x_1940.jsonl.gz", "wt",
                   encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    import logging
    coverage = b.build_corpus(cfg, logging.getLogger("t"), arm="dlnews")
    assert coverage["new_york_1940"] == 2
    assert coverage["nevada_1940"] == 1
    # New York unit dir written (>=min_docs); Nevada below threshold -> not trained
    assert (Path(cfg["paths"]["corpora_dir"]) / "new_york_1940").exists()


def test_dedup_collapses_wire_copy_within_year(tmp_path):
    cfg = _cfg(tmp_path, "dlnews", min_docs=1, dedup_method="exact")
    raw = Path(cfg["paths"]["raw_data_dir"]); raw.mkdir(parents=True)
    wire = "the senate approved a sweeping farm bill on tuesday afternoon"
    rows = [
        {"content": wire, "location": {"state": "New York"}, "is_news_article": True, "title": "w"},
        {"content": wire, "location": {"state": "Texas"}, "is_news_article": True, "title": "w"},
    ]
    with gzip.open(raw / "preprocessed_google_newspaper_x_1940.jsonl.gz", "wt",
                   encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    import logging
    coverage = b.build_corpus(cfg, logging.getLogger("t"), arm="dlnews")
    total = sum(coverage.values())
    assert total == 1  # identical wire story counted once across states
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_build_corpora_us.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

```python
# scripts/data_prep/build_corpora_us.py
#!/usr/bin/env python3
"""Build per-(state, year) English corpora for the US arms.

Arm 'american_stories': raw article JSONL; state via LCCN -> LoC table.
Arm 'dlnews' (3DLNews2): preprocessed_google_newspaper_{STATE}_{YEAR}.jsonl.gz;
state via inline ``location.state``.

Each unit is written to corpora_dir/{state}_{year}/corpus_%06d, so training
and analysis discover units with no changes. Wire-copy dedup runs within-year
across states by default. A coverage report records per-unit doc counts and
which units clear us_states.min_documents.

Usage:
  python -m scripts.data_prep.build_corpora_us --config=config/profiles/garg_weat_dlnews.yml
  python -m scripts.data_prep.build_corpora_us --config=... --arm=american_stories
"""

from __future__ import annotations

import glob
import gzip
import json
import os
from pathlib import Path
from typing import Dict, Iterator, Optional

import fire

from scripts.common.config_loader import load_config
from scripts.common.logging_utils import setup_logging
from scripts.common.preprocessing import preprocess
from scripts.data_prep import us_state_mapper as usm
from scripts.data_prep.dedup import Deduper


class UnitCorpusWriter:
    """Rolling file writer for one {state}_{year} unit (from ProvinceCorpusWriter)."""

    def __init__(self, unit_name: str, output_dir: str, max_bytes: int = 1024 ** 3):
        self.unit_name = unit_name
        self.max_bytes = max_bytes
        self.unit_dir = os.path.join(output_dir, unit_name)
        os.makedirs(self.unit_dir, exist_ok=True)
        self.index = 0
        self.total_lines = 0
        self._open_next()

    def _open_next(self):
        while True:
            fp = os.path.join(self.unit_dir, f"corpus_{self.index:06d}")
            if not os.path.exists(fp):
                break
            self.index += 1
        self.file = open(fp, "w", buffering=8 * 1024 * 1024, encoding="utf-8")
        self.bytes_written = 0

    def write(self, words):
        if not words or len(words) < 5:
            return
        line = " ".join(words) + "\n"
        if self.bytes_written + len(line) > self.max_bytes:
            self.file.close()
            self.index += 1
            self._open_next()
        self.file.write(line)
        self.bytes_written += len(line)
        self.total_lines += 1

    def close(self):
        self.file.close()


def _open_maybe_gzip(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return open(path, "r", encoding="utf-8", errors="ignore")


def iter_records(arm: str, raw_dir: str, year: int,
                 lccn_table: Optional[Dict[str, str]] = None) -> Iterator[dict]:
    """Yield {'text','state','title'} for one arm+year. Unknown/absent state dropped."""
    if arm == "dlnews":
        pattern = os.path.join(raw_dir, f"*_{year}.jsonl.gz")
        files = sorted(glob.glob(pattern)) or sorted(
            glob.glob(os.path.join(raw_dir, f"*_{year}.jsonl")))
        for fp in files:
            with _open_maybe_gzip(fp) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not r.get("is_news_article", True):
                        continue
                    loc = r.get("location") or {}
                    state = usm.normalize_state(loc.get("state") or "")
                    text = r.get("content") or ""
                    if state and text:
                        yield {"text": text, "state": state, "title": r.get("title", "")}
    elif arm == "american_stories":
        if lccn_table is None:
            lccn_table = {}
        pattern = os.path.join(raw_dir, f"american_stories_{year}*.jsonl")
        for fp in sorted(glob.glob(pattern)):
            with _open_maybe_gzip(fp) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    lccn = usm.lccn_from_article_id(r.get("article_id", ""))
                    state = usm.resolve_state(lccn, lccn_table) if lccn else None
                    text = r.get("article") or ""
                    if state and text:
                        yield {"text": text, "state": state, "title": r.get("headline", "")}
    else:
        raise ValueError(f"unknown arm: {arm!r}")


def build_corpus(config: dict, logger, arm: str, max_files: Optional[int] = None) -> Dict[str, int]:
    raw_dir = config["paths"]["raw_data_dir"]
    corpora_dir = config["paths"]["corpora_dir"]
    years = config["us_states"]["years"]
    min_docs = int(config["us_states"].get("min_documents", 500))
    dcfg = config.get("corpus", {}).get("dedup", {"enabled": False})

    lccn_table = None
    if arm == "american_stories":
        table_path = os.path.join(raw_dir, "lccn_state_table.json")
        lccn_table = usm.load_lccn_state_table(table_path) if os.path.exists(table_path) else {}
        logger.info(f"Loaded LCCN->state table: {len(lccn_table)} entries")

    coverage: Dict[str, int] = {}
    writers: Dict[str, UnitCorpusWriter] = {}
    for year in years:
        deduper = Deduper(
            method=dcfg.get("method", "shingle"),
            shingle_k=int(dcfg.get("shingle_k", 8)),
        ) if dcfg.get("enabled") else None
        n_seen = n_dup = 0
        for rec in iter_records(arm, raw_dir, year, lccn_table):
            n_seen += 1
            if deduper is not None and deduper.is_duplicate(rec["text"]):
                n_dup += 1
                continue
            tokens = preprocess(
                rec["text"],
                language=config["language"],
                tokenizer=config["corpus"]["tokenizer"],
                stopwords_key=config["corpus"].get("stopwords"),
                lowercase=config["corpus"].get("lowercase", True),
                min_words=config["corpus"].get("min_words", 5),
            )
            if tokens is None:
                continue
            unit = f"{usm.unit_state(rec['state'])}_{year}"
            if unit not in writers:
                writers[unit] = UnitCorpusWriter(unit, corpora_dir)
            writers[unit].write(tokens)
            coverage[unit] = coverage.get(unit, 0) + 1
        logger.info(f"year={year}: seen={n_seen}, wire-dups dropped={n_dup}")
    for w in writers.values():
        w.close()

    kept = {u: n for u, n in coverage.items() if n >= min_docs}
    dropped = {u: n for u, n in coverage.items() if n < min_docs}
    if dropped:
        logger.warning(f"{len(dropped)} units below min_documents={min_docs} "
                       f"(kept out of training): {sorted(dropped)}")
    report_path = os.path.join(config["paths"]["results_dir"], f"coverage_{arm}.csv")
    os.makedirs(config["paths"]["results_dir"], exist_ok=True)
    write_coverage_report(coverage, min_docs, report_path)
    logger.info(f"Coverage report -> {report_path}. Trainable units: {len(kept)}")
    return coverage


def write_coverage_report(coverage: Dict[str, int], min_docs: int, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("unit_name,state,year,n_docs,kept\n")
        for unit in sorted(coverage):
            state, _, year = unit.rpartition("_")
            kept = 1 if coverage[unit] >= min_docs else 0
            f.write(f"{unit},{state},{year},{coverage[unit]},{kept}\n")


def main(config: str = "config/config.yml", arm: Optional[str] = None,
         max_files: Optional[int] = None) -> None:
    cfg = load_config(config)
    logger = setup_logging(Path(cfg["paths"]["log_dir"]), "build_corpora_us.log")
    arm = arm or cfg.get("_arm") or cfg.get("embedding_source")
    logger.info(f"Building US corpora: arm={arm}")
    build_corpus(cfg, logger, arm=arm, max_files=max_files)


if __name__ == "__main__":
    fire.Fire(main)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_build_corpora_us.py -v`
Expected: PASS (3 tests). Requires `nltk` punkt/stopwords data available (already a project dep; tests use short sentences that tokenize fine).

- [ ] **Step 5: Commit**

```bash
git add scripts/data_prep/build_corpora_us.py tests/test_build_corpora_us.py
git commit -m "feat(us): per-state_year corpus builder with dedup + coverage report"
```

---

## Task 4: American Stories downloader

**Files:**
- Create: `scripts/data_prep/download_american_stories.py`

**Interfaces:**
- Consumes: `us_states.years` from config.
- Produces: raw `american_stories_{year}.jsonl` under `raw_data_dir`, one JSON object per article with keys `article_id, newspaper_name, date, headline, byline, article`.

- [ ] **Step 1: Write the implementation**

```python
# scripts/data_prep/download_american_stories.py
#!/usr/bin/env python3
"""Download American Stories article text for the configured years.

Writes raw_data_dir/american_stories_{year}.jsonl (one article per line).
Network step — run where the node has internet (login/internet node).
Idempotent: skips a year whose output already exists and is non-empty.

Usage:
  python -m scripts.data_prep.download_american_stories --config=config/profiles/garg_weat_american_stories.yml
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import fire

from scripts.common.config_loader import load_config
from scripts.common.logging_utils import setup_logging

_KEEP = ("article_id", "newspaper_name", "date", "headline", "byline", "article")


def _download_year(year: int, out_path: str, logger) -> int:
    from datasets import load_dataset
    ds = load_dataset(
        "dell-research-harvard/AmericanStories",
        "subset_years",
        year_list=[str(year)],
        trust_remote_code=True,
    )
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for split in ds:
            for row in ds[split]:
                rec = {k: row.get(k, "") for k in _KEEP}
                if not rec.get("article"):
                    continue
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
    logger.info(f"  {year}: wrote {n} articles -> {out_path}")
    return n


def main(config: str = "config/config.yml") -> None:
    cfg = load_config(config)
    logger = setup_logging(Path(cfg["paths"]["log_dir"]), "download_american_stories.log")
    raw_dir = cfg["paths"]["raw_data_dir"]
    os.makedirs(raw_dir, exist_ok=True)
    for year in cfg["us_states"]["years"]:
        out = os.path.join(raw_dir, f"american_stories_{year}.jsonl")
        if os.path.exists(out) and os.path.getsize(out) > 0:
            logger.info(f"  {year}: exists, skipping")
            continue
        logger.info(f"Downloading American Stories {year}...")
        _download_year(year, out, logger)


if __name__ == "__main__":
    fire.Fire(main)
```

- [ ] **Step 2: Smoke-test the module imports and CLI wiring (no network)**

Run: `python -c "import scripts.data_prep.download_american_stories as d; print(d.main.__doc__ is not None)"`
Expected: prints `True` (module imports cleanly; `datasets` import is deferred into `_download_year`).

- [ ] **Step 3: Commit**

```bash
git add scripts/data_prep/download_american_stories.py
git commit -m "feat(us): American Stories downloader (HF subset_years -> raw jsonl)"
```

---

## Task 5: 3DLNews2 Globus downloader

**Files:**
- Create: `scripts/data_prep/download_dlnews.py`

**Interfaces:**
- Consumes: config `dlnews.{source_endpoint, dest_endpoint, source_root, dest_root}`, `us_states.years`.
- Produces:
  - `build_transfer_batch(source_root, dest_root, years, states) -> list[tuple[str,str]]` — (src_path, dst_path) pairs (pure, testable).
  - `main(config)` — issues `globus transfer --batch` then `globus task wait`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_download_dlnews.py
from scripts.data_prep.download_dlnews import build_transfer_batch


def test_build_transfer_batch_pairs_paths():
    pairs = build_transfer_batch(
        "/3dlnews2/Google/1-Newspapers/preprocessed_state",
        "/scratch/network/yh6580/gender-occup/data/american_state/dlnews/raw",
        years=[2000, 2020], states=["New York"])
    assert (
        "/3dlnews2/Google/1-Newspapers/preprocessed_state/New York/preprocessed_google_newspaper_New York_2000.jsonl.gz",
        "/scratch/network/yh6580/gender-occup/data/american_state/dlnews/raw/preprocessed_google_newspaper_New York_2000.jsonl.gz",
    ) in pairs
    assert len(pairs) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_download_dlnews.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

```python
# scripts/data_prep/download_dlnews.py
#!/usr/bin/env python3
"""Transfer 3DLNews2 newspaper slices via Globus for the configured years.

3DLNews2 is distributed through Globus. Auth is a one-time interactive
`globus login` on the login node; transfers then run headless from this script.
Globus transfer is free for individual researchers; Princeton runs a managed
endpoint (dest).

Config (dlnews block):
  source_endpoint: <3DLNews2 collection UUID>
  dest_endpoint:   <Princeton endpoint UUID>
  source_root:     /.../preprocessed_state
  dest_root:       <raw_data_dir on the dest endpoint's namespace>
  states:          optional; default = all 50 states + DC

Fallback if OAuth can't run headless: this prints the equivalent
`globus transfer --batch` command; run it manually, then proceed to the builder.

Usage:
  python -m scripts.data_prep.download_dlnews --config=config/profiles/garg_weat_dlnews.yml
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import fire

from scripts.common.config_loader import load_config
from scripts.common.logging_utils import setup_logging
from scripts.data_prep import us_state_mapper as usm


def build_transfer_batch(source_root: str, dest_root: str, years: List[int],
                         states: List[str]) -> List[Tuple[str, str]]:
    """(src, dst) path pairs for each state x year newspaper slice."""
    pairs: List[Tuple[str, str]] = []
    for state in states:
        for year in years:
            fname = f"preprocessed_google_newspaper_{state}_{year}.jsonl.gz"
            src = f"{source_root}/{state}/{fname}"
            dst = f"{dest_root}/{fname}"
            pairs.append((src, dst))
    return pairs


def main(config: str = "config/config.yml", dry_run: bool = False) -> None:
    cfg = load_config(config)
    logger = setup_logging(Path(cfg["paths"]["log_dir"]), "download_dlnews.log")
    d = cfg["dlnews"]
    states = d.get("states") or list(usm._STATE_NAME_TO_USPS.keys())
    years = cfg["us_states"]["years"]
    pairs = build_transfer_batch(d["source_root"], d["dest_root"], years, states)

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False,
                                     encoding="utf-8") as bf:
        for src, dst in pairs:
            bf.write(f'"{src}" "{dst}"\n')
        batch_file = bf.name

    cmd = ["globus", "transfer", "--batch", batch_file,
           d["source_endpoint"], d["dest_endpoint"], "--label", "3dlnews2-us-arm"]
    logger.info(f"Prepared {len(pairs)} transfer pairs; batch file: {batch_file}")
    if dry_run:
        logger.info("dry_run: " + " ".join(cmd))
        return
    logger.info("Submitting Globus transfer (requires prior `globus login`)...")
    out = subprocess.run(cmd, capture_output=True, text=True)
    logger.info(out.stdout.strip())
    if out.returncode != 0:
        logger.error(out.stderr.strip())
        logger.error("If OAuth cannot run here, run the batch manually:\n  "
                     + " ".join(cmd))
        raise SystemExit(out.returncode)
    task_id = out.stdout.strip().split()[-1]
    subprocess.run(["globus", "task", "wait", task_id], check=False)


if __name__ == "__main__":
    fire.Fire(main)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_download_dlnews.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/data_prep/download_dlnews.py tests/test_download_dlnews.py
git commit -m "feat(us): 3DLNews2 Globus downloader (batch build + transfer)"
```

---

## Task 6: US states shapefile fetcher + config profiles

**Files:**
- Create: `scripts/data_prep/fetch_us_shapefile.py`, `config/profiles/garg_weat_american_stories.yml`, `config/profiles/garg_weat_dlnews.yml`

**Interfaces:**
- Produces: `data/shapefiles/us_states.shp` (+ sidecars); two config profiles consumed by all later stages.

- [ ] **Step 1: Write the shapefile fetcher**

```python
# scripts/data_prep/fetch_us_shapefile.py
#!/usr/bin/env python3
"""Fetch the US Census cartographic-boundary states shapefile (cb 20m).

Downloads the public zip and extracts it to data/shapefiles/, renaming the
layer to us_states.*. Public-domain Census data. Network step.

Usage:
  python -m scripts.data_prep.fetch_us_shapefile
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import fire

_URL = "https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_state_20m.zip"


def main(out_dir: str = "data/shapefiles") -> None:
    import requests

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    resp = requests.get(_URL, timeout=120)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        z.extractall(out)
    for src in out.glob("cb_2018_us_state_20m.*"):
        dst = out / ("us_states" + src.suffix)
        src.replace(dst)
    print(f"US states shapefile -> {out/'us_states.shp'}")


if __name__ == "__main__":
    fire.Fire(main)
```

- [ ] **Step 2: Write `config/profiles/garg_weat_american_stories.yml`**

```yaml
# Garg-WEAT per-(state, year) gender ideation on American Stories (1900-1960).
# State via LCCN -> LoC US Newspaper Directory. Units are state-year
# (e.g. 'california_1940'); per-year US choropleths + trend.
language: "en"
data_source: "american_stories"
analysis_mode: "garg_weat"
embedding_source: "american_stories"
_arm: "american_stories"

paths:
  base_dir: "/scratch/network/yh6580/gender-occup"
  raw_data_dir: "/scratch/network/yh6580/gender-occup/data/american_state/american_stories/raw"
  corpora_dir: "/scratch/network/yh6580/gender-occup/data/american_state/american_stories/corpora"
  models_dir: "/scratch/network/yh6580/gender-occup/data/american_state/american_stories/models"
  results_dir: "/scratch/network/yh6580/gender-occup/data/american_state/american_stories/results_garg_weat"
  log_dir: "/scratch/network/yh6580/gender-occup/logs/american_stories"
  figures_dir: "/scratch/network/yh6580/gender-occup/figures/american_stories_garg_weat"

us_states:
  shapefile: "data/shapefiles/us_states.shp"
  min_documents: 500
  years: [1900, 1910, 1920, 1930, 1940, 1950, 1960]

wordlists:
  dir: "wordlists/en/garg_weat"
  gender_words_file: "gender_words.json"
  categories:
    leadership: "cleaned_leadership.txt"
    family: "cleaned_family.txt"
    science: "cleaned_science.txt"

corpus:
  tokenizer: "nltk_en"
  stopwords: "en_default"
  lowercase: true
  min_words: 5
  dedup:
    enabled: true
    method: "shingle"
    shingle_k: 8
    scope: "within_year"

embedding:
  vector_size: 300
  window: 5
  min_count: 20
  sg: 1
  negative: 10
  workers: 8
  epochs: 10
  seed: 42
  model_name_template: "model_{unit_name}.kv"

analysis:
  metrics: [rnd, cohens_d]
  seed: 42
  ideation_sign:
    leadership: 1
    science: 1
    family: -1
  bootstrap:
    n_iter: 5000
    ci: 0.68
  subsample:
    fraction: 0.8
    n_rounds: 100
    ci: 0.95
```

- [ ] **Step 3: Write `config/profiles/garg_weat_dlnews.yml`**

Same as Task 6 Step 2 but with these differences (repeat the full file with these values):
- `data_source`, `embedding_source`, `_arm`: `"dlnews"`
- `paths.*`: replace `american_stories` path segment with `dlnews`; `log_dir` → `.../logs/dlnews`; `figures_dir` → `.../figures/dlnews_garg_weat`
- `us_states.years: [1996, 2000, 2010, 2020]`
- Add a `dlnews` block:

```yaml
dlnews:
  source_endpoint: "REPLACE_WITH_3DLNEWS2_COLLECTION_UUID"
  dest_endpoint: "REPLACE_WITH_PRINCETON_ENDPOINT_UUID"
  source_root: "/REPLACE/preprocessed_state"
  dest_root: "/scratch/network/yh6580/gender-occup/data/american_state/dlnews/raw"
  # states: optional allow-list; default = all 50 + DC
```

All other blocks (`wordlists`, `corpus` incl. dedup, `embedding`, `analysis`) are identical to the American Stories profile.

- [ ] **Step 4: Verify both configs load**

Run: `python -c "from scripts.common.config_loader import load_config; [load_config(f'config/profiles/garg_weat_{a}.yml') for a in ('american_stories','dlnews')]; print('ok')"`
Expected: prints `ok` (no schema/parse error).

- [ ] **Step 5: Commit**

```bash
git add scripts/data_prep/fetch_us_shapefile.py config/profiles/garg_weat_american_stories.yml config/profiles/garg_weat_dlnews.yml
git commit -m "feat(us): shapefile fetcher + American Stories/3DLNews2 garg_weat configs"
```

---

## Task 7: US choropleth in visualize.py

**Files:**
- Modify: `scripts/visualize.py` (add functions; do not alter existing ones)
- Test: `tests/test_us_choropleth.py`

**Interfaces:**
- Consumes: `_plot_single_choropleth` (existing), config `analysis.ideation_sign`, `us_states.shapefile`.
- Produces:
  - `_state_year_parse(unit_name: str) -> tuple[str, int] | None` — `"new_york_1940"` → `("New York", 1940)`.
  - `_match_state_in_shapefile(dim_data, states_gdf)` — merge on shapefile `NAME` (dim_data has canonical `state` names).
  - `plot_us_choropleth(summary_df, figures_dir, logger, config)` — per-year oriented-RND maps.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_us_choropleth.py
import pandas as pd
from scripts import visualize as v


def test_state_year_parse():
    assert v._state_year_parse("new_york_1940") == ("New York", 1940)
    assert v._state_year_parse("california_1996") == ("California", 1996)
    assert v._state_year_parse("district_of_columbia_2000") == ("District of Columbia", 2000)
    assert v._state_year_parse("1990s") is None
    assert v._state_year_parse("not_a_unit") is None


def test_match_state_in_shapefile_joins_on_name():
    gpd = __import__("importlib").import_module("geopandas") if _has_gpd() else None
    if gpd is None:
        import pytest; pytest.skip("geopandas not installed")
    from shapely.geometry import Point
    states = gpd.GeoDataFrame(
        {"NAME": ["California", "Nevada"], "geometry": [Point(0, 0), Point(1, 1)]})
    dim = pd.DataFrame({"state": ["California"], "oriented_rnd": [0.12]})
    merged = v._match_state_in_shapefile(dim, states)
    row = merged[merged["NAME"] == "California"].iloc[0]
    assert abs(row["oriented_rnd"] - 0.12) < 1e-9


def _has_gpd():
    import importlib.util
    return importlib.util.find_spec("geopandas") is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_us_choropleth.py -v`
Expected: FAIL with `AttributeError: module 'scripts.visualize' has no attribute '_state_year_parse'`.

- [ ] **Step 3: Add the implementation to `scripts/visualize.py`**

Add near the provincial choropleth helpers (after `_match_province_in_shapefile`):

```python
def _state_year_parse(unit_name):
    """'new_york_1940' -> ('New York', 1940). Returns None if it doesn't parse."""
    from scripts.data_prep.us_state_mapper import normalize_state
    s = str(unit_name)
    head, _, tail = s.rpartition("_")
    if not head or not tail.isdigit():
        return None
    state = normalize_state(head.replace("_", " "))
    if state is None:
        return None
    return state, int(tail)


def _match_state_in_shapefile(dim_data, states_gdf):
    """Merge per-state values onto a US states shapefile on its name column."""
    match_col = "NAME"
    for col in ["NAME", "name", "STATE_NAME", "STUSPS"]:
        if col in states_gdf.columns:
            match_col = col
            break
    return states_gdf.merge(dim_data, left_on=match_col, right_on="state", how="left")


def plot_us_choropleth(summary_df, figures_dir, logger, config):
    """Per-year US choropleths of oriented gender-ideation RND by state.

    summary_df: rows with columns category, unit_name, mean_rnd (from
    garg_weat_summary_by_category.parquet). Orientation applies
    analysis.ideation_sign so higher = less traditional across categories.
    Skips gracefully if geopandas / the shapefile is unavailable.
    """
    try:
        import geopandas as gpd
    except ImportError:
        logger.info("  Skipping US choropleth (geopandas not installed)")
        return
    shp = Path(config.get("us_states", {}).get("shapefile", "data/shapefiles/us_states.shp"))
    if not shp.exists():
        logger.info(f"  Skipping US choropleth (no shapefile at {shp})")
        return
    states_gdf = gpd.read_file(shp)
    # Continental view: drop AK/HI/territories for legibility (still computed).
    states_gdf = states_gdf[~states_gdf["NAME"].isin(
        ["Alaska", "Hawaii", "Puerto Rico"])]

    signs = config.get("analysis", {}).get("ideation_sign", {})
    df = summary_df.copy()
    parsed = df["unit_name"].apply(_state_year_parse)
    df = df[parsed.notna()].copy()
    if df.empty:
        logger.info("  Skipping US choropleth (no state_year units parsed)")
        return
    df["state"] = [p[0] for p in parsed[parsed.notna()]]
    df["year"] = [p[1] for p in parsed[parsed.notna()]]
    df["oriented_rnd"] = df.apply(
        lambda r: r["mean_rnd"] * signs.get(r["category"], 1), axis=1)

    # Average orientation across categories -> one ideation value per state-year.
    agg = (df.groupby(["state", "year"])["oriented_rnd"].mean().reset_index())
    vmax = float(agg["oriented_rnd"].abs().max()) or 1.0
    from matplotlib.colors import TwoSlopeNorm
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    for year in sorted(agg["year"].unique()):
        year_data = agg[agg["year"] == year][["state", "oriented_rnd"]]
        merged = _match_state_in_shapefile(year_data, states_gdf)
        _plot_single_choropleth(
            merged, f"US gender ideation by state — {year}",
            f"us_gender_ideation_{year}.pdf", figures_dir, logger,
            value_col="oriented_rnd", norm=norm, cmap="RdBu_r")
```

- [ ] **Step 4: Wire it into `main` (garg_weat path) of `scripts/visualize.py`**

Find where the provincial choropleth is dispatched in `main` (search `plot_weat_choropleth`) and add, guarded on the US arms:

```python
    if config.get("data_source") in ("american_stories", "dlnews"):
        plot_us_choropleth(summary_df, figures_dir, logger, config)
```

(Use the same `summary_df` / `figures_dir` variables already in scope there; if the garg path names them differently, match the local names.)

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_us_choropleth.py -v`
Expected: PASS (`_state_year_parse` tests pass; the shapefile join test passes if geopandas present, else skips).

- [ ] **Step 6: Commit**

```bash
git add scripts/visualize.py tests/test_us_choropleth.py
git commit -m "feat(us): per-year US state choropleth of gender ideation"
```

---

## Task 8: Slurm orchestration (prepare / train / analyze)

**Files:**
- Create: `slurm/prepare_us_data.slurm`, `slurm/train_us.slurm`, `slurm/garg_weat_us.slurm`

**Interfaces:** consumes the two config profiles; drives the CLIs from Tasks 1–7. No new Python.

- [ ] **Step 1: Write `slurm/prepare_us_data.slurm`**

```bash
#!/bin/bash
#SBATCH --job-name=prepare_us_data
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/prepare_us_data_%j.out
#SBATCH --error=logs/prepare_us_data_%j.err

# Acquire + build US corpora. The download steps need internet — run on an
# internet-enabled node/partition. 3DLNews2 needs a one-time `globus login`
# on the login node BEFORE submitting. Pass the arm's config as $1.
#   sbatch slurm/prepare_us_data.slurm config/profiles/garg_weat_american_stories.yml
#   sbatch slurm/prepare_us_data.slurm config/profiles/garg_weat_dlnews.yml

module load anaconda3/2023.3
conda activate llm
set -u

CONFIG="${1:?pass a config profile}"
ARM=$(python3 -c "import yaml;print(yaml.safe_load(open('$CONFIG'))['_arm'])")

# Shapefile (once; harmless if it already exists)
python -m scripts.data_prep.fetch_us_shapefile || echo "WARN: shapefile fetch failed"

if [ "$ARM" = "american_stories" ]; then
    python -m scripts.data_prep.us_state_mapper build --config="$CONFIG"
    python -m scripts.data_prep.download_american_stories --config="$CONFIG"
else
    python -m scripts.data_prep.download_dlnews --config="$CONFIG"
fi

python -m scripts.data_prep.build_corpora_us --config="$CONFIG" --arm="$ARM"
echo "DONE prepare_us_data: $ARM"
```

- [ ] **Step 2: Write `slurm/train_us.slurm`**

```bash
#!/bin/bash
#SBATCH --job-name=train_us
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_us_%j.out
#SBATCH --error=logs/train_us_%j.err

# Train one Word2Vec model per {state}_{year} unit discovered in corpora_dir.
# Reuses scripts.train_embeddings unchanged.
#   sbatch slurm/train_us.slurm config/profiles/garg_weat_american_stories.yml

module load anaconda3/2023.3
conda activate llm
set -u
CONFIG="${1:?pass a config profile}"
python -m scripts.train_embeddings --config="$CONFIG"
echo "DONE train_us"
```

- [ ] **Step 3: Write `slurm/garg_weat_us.slurm`** (mirror `slurm/garg_weat_zh.slurm`)

```bash
#!/bin/bash
#SBATCH --job-name=garg_weat_us
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/garg_weat_us_%j.out
#SBATCH --error=logs/garg_weat_us_%j.err

# US Garg-WEAT: analyze_category_bias (RND + Cohen's d) + visualize (trend +
# US choropleth) for the two US arms. Mirrors slurm/garg_weat_zh.slurm.
module load anaconda3/2023.3
conda activate llm
set -u

DEFAULT_CONFIGS=(
    "config/profiles/garg_weat_american_stories.yml"
    "config/profiles/garg_weat_dlnews.yml"
)
if [ "$#" -gt 0 ]; then CONFIGS=("$@"); else CONFIGS=("${DEFAULT_CONFIGS[@]}"); fi

read_config_value() { python3 -c "import yaml; c=yaml.safe_load(open('$1')); print($2)"; }
declare -a STATUSES

for CONFIG in "${CONFIGS[@]}"; do
    echo "----- $CONFIG -----"
    if [ ! -f "$CONFIG" ]; then STATUSES+=("missing_config"); continue; fi
    MODELS_DIR=$(read_config_value "$CONFIG" "c['paths']['models_dir']")
    RESULTS_DIR=$(read_config_value "$CONFIG" "c['paths']['results_dir']")
    if [ ! -d "$MODELS_DIR" ]; then echo "SKIP: no models_dir"; STATUSES+=("no_models_dir"); continue; fi
    NHIT=$(find "$MODELS_DIR" -maxdepth 2 -name '*.kv' | head -1 | wc -l | tr -d ' ')
    if [ "$NHIT" -eq 0 ]; then echo "SKIP: no .kv"; STATUSES+=("empty_models_dir"); continue; fi

    if ! python -m scripts.analyze_category_bias --config="$CONFIG"; then
        STATUSES+=("analyze_failed"); continue; fi
    SUMMARY="$RESULTS_DIR/garg_weat_summary_by_category.parquet"
    if [ ! -f "$SUMMARY" ]; then STATUSES+=("no_results_written"); continue; fi
    if ! python -m scripts.visualize main --config="$CONFIG"; then
        STATUSES+=("visualize_failed"); continue; fi
    STATUSES+=("ok")
done

echo "===== SUMMARY ====="
for j in "${!CONFIGS[@]}"; do printf "  %-56s %s\n" "${CONFIGS[$j]}" "${STATUSES[$j]}"; done
```

- [ ] **Step 4: Shellcheck / syntax check**

Run: `bash -n slurm/prepare_us_data.slurm && bash -n slurm/train_us.slurm && bash -n slurm/garg_weat_us.slurm && echo ok`
Expected: prints `ok`.

- [ ] **Step 5: Commit**

```bash
git add slurm/prepare_us_data.slurm slurm/train_us.slurm slurm/garg_weat_us.slurm
git commit -m "slurm(us): prepare + train + garg_weat orchestration for US arms"
```

---

## Self-Review

**Spec coverage:**
- Two independent arms → Tasks 3/6 (`_arm`, separate configs/paths). ✓
- LCCN→state from LoC directory → Task 1. ✓
- 3DLNews2 inline state → Task 3 `iter_records("dlnews")`. ✓
- Scripts fetch over network → Tasks 1 (`build`), 4, 5, 6. ✓
- `{state}_{year}` units, reuse train/analyze unchanged → Task 3 writer dir = unit; verified against `discover_units`/`discover_models`/`get_model_name`. ✓
- Configurable within-year wire dedup, default on → Tasks 2/3 + config `corpus.dedup`. ✓
- Coverage report + drop-below-threshold, logged → Task 3. ✓
- US choropleth mirroring provincial → Task 7. ✓
- Full end-to-end slurm → Task 8. ✓
- English preprocessing reuse → Task 3 `preprocess(language="en", ...)`. ✓
- Wordlists `wordlists/en/garg_weat/` → Task 6 configs. ✓

**Placeholder scan:** The only intentional `REPLACE_WITH_*` values are the 3DLNews2/Princeton Globus UUIDs and source_root — user-supplied config secrets, flagged as such in the spec's "Open items." No `TBD`/`TODO` in code steps.

**Type consistency:** `normalize_state`/`unit_state`/`resolve_state`/`lccn_from_article_id` signatures match across Tasks 1, 3, 7. `Deduper(method, shingle_k, ...).is_duplicate/reset` match Tasks 2, 3. `iter_records`/`build_corpus`/`UnitCorpusWriter` names match test and impl in Task 3. `_state_year_parse` returns `(canonical_state, int)` consumed by `plot_us_choropleth` in Task 7. Choropleth value column `oriented_rnd` consistent across `_match_state_in_shapefile` test and `plot_us_choropleth`.

**Deferred (validate during implementation, per spec):**
- Exact LoC directory JSON param names (`number_lccn`/`location_state`) and pagination — Task 1 `_fetch_loc_directory_records`.
- LCCN location inside American Stories `article_id` — Task 1 `lccn_from_article_id` regex; adjust once a real id is inspected.
- Dedup `bands`/`shingle_k` defaults — Task 2 Step 4 note; tune on a sample slice.
