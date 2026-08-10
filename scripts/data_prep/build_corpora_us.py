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
        if not words:
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


def build_corpus(config: dict, logger, arm: str) -> Dict[str, int]:
    raw_dir = config["paths"]["raw_data_dir"]
    corpora_dir = config["paths"]["corpora_dir"]
    years = config["us_states"]["years"]
    min_docs = int(config["us_states"].get("min_documents", 500))
    dcfg = config.get("corpus", {}).get("dedup", {"enabled": False})
    _scope = dcfg.get("scope", "within_year")
    if dcfg.get("enabled") and _scope != "within_year":
        raise ValueError(
            f"build_corpora_us only supports dedup scope 'within_year', got {_scope!r}. "
            "Within-year-across-states scoping is structural (fresh Deduper per year)."
        )

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


def main(config: str = "config/config.yml", arm: Optional[str] = None) -> None:
    cfg = load_config(config)
    logger = setup_logging(Path(cfg["paths"]["log_dir"]), "build_corpora_us.log")
    arm = arm or cfg.get("_arm") or cfg.get("embedding_source")
    logger.info(f"Building US corpora: arm={arm}")
    build_corpus(cfg, logger, arm=arm)


if __name__ == "__main__":
    fire.Fire(main)
