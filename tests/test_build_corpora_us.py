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


def test_american_stories_reads_tarball_matches_state_and_drops(tmp_path):
    # Parse raw faro_{year}.tar.gz directly (the login node only downloads tars;
    # extraction + state matching happen here). article_id is rebuilt exactly as
    # the HF builder does: full_article_id + "_" + scan_id, where scan_id is the
    # member basename and embeds the LCCN (verified real format from the card:
    # "1_1870-01-01_p1_sn82014899_...").
    import io
    import tarfile

    raw = tmp_path / "raw"; raw.mkdir(parents=True)
    scans = {
        # member basename -> scan JSON. First member's LCCN resolves; second's
        # LCCN is not in the table (its articles must be dropped).
        "faro_1940/1940-01-02_p1_sn83030214_00211105483_1940010201_0773.json": {
            "lccn": {"title": "The Example Times"},
            "full articles": [
                {"full_article_id": 1, "headline": "H1", "byline": "",
                 "article": "he leads the council meeting on the new policy"},
                {"full_article_id": 2, "headline": "H2", "byline": "",
                 "article": "she chairs the science board this spring"},
            ],
        },
        "faro_1940/1940-03-05_p2_sn99999999_00211105483_1940030502_0773.json": {
            "lccn": {"title": "Unknown Gazette"},
            "full articles": [
                {"full_article_id": 1, "headline": "X", "byline": "",
                 "article": "content whose lccn is not in the table"},
            ],
        },
    }
    tar_path = raw / "faro_1940.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        for name, obj in scans.items():
            payload = json.dumps(obj).encode("utf-8")
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))

    table = {"sn83030214": "New York"}  # sn99999999 deliberately absent
    stats: dict = {}
    recs = list(b.iter_records("american_stories", str(raw), 1940, table, stats=stats))
    assert [r["state"] for r in recs] == ["New York", "New York"]  # both resolved articles
    assert [r["title"] for r in recs] == ["H1", "H2"]
    assert stats["read"] == 3           # 2 resolved + 1 unresolved
    assert stats["kept"] == 2
    assert stats["dropped_unresolved_lccn"] == 1


def test_american_stories_stats_report_resolution_drops(tmp_path):
    # stats mutates with drop reasons so the caller can log the resolution rate.
    raw = tmp_path / "raw"; raw.mkdir(parents=True)
    rows = [
        {"article_id": "sn83030214_1940-01-02_p1_a1", "article": "he leads the town"},   # kept
        {"article_id": "sn99999999_1940-01-02_p1_a2", "article": "unknown lccn here"},   # unresolved
        {"article_id": "no-lccn-at-all", "article": "no id text"},                       # no lccn
        {"article_id": "sn83030214_1940-01-02_p1_a3", "article": ""},                    # empty text
    ]
    with open(raw / "american_stories_1940.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    table = {"sn83030214": "New York"}
    stats: dict = {}
    recs = list(b.iter_records("american_stories", str(raw), 1940, table, stats=stats))
    assert [r["state"] for r in recs] == ["New York"]
    assert stats["read"] == 4
    assert stats["kept"] == 1
    assert stats["dropped_unresolved_lccn"] == 1
    assert stats["dropped_no_lccn"] == 1
    assert stats["dropped_empty_text"] == 1


def test_dlnews_state_from_usps_filename_is_authoritative(tmp_path):
    # 3DLNews2 partitions files by USPS code; the filename is authoritative even
    # when a record has no inline location (or a different one).
    raw = tmp_path / "raw"; raw.mkdir(parents=True)
    rows = [
        {"content": "the senate approved the farm policy reform bill today",
         "is_news_article": True, "title": "a"},  # no location field at all
        {"content": "governor signed the education funding measure this morning",
         "location": {"state": "California"}, "is_news_article": True, "title": "b"},
    ]
    with gzip.open(raw / "preprocessed_google_newspaper_NY_2000.jsonl.gz", "wt",
                   encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    recs = list(b.iter_records("dlnews", str(raw), 2000))
    assert [r["state"] for r in recs] == ["New York", "New York"]  # filename wins


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


def test_build_corpus_skips_already_built_year(tmp_path):
    # Re-running build must not re-append (duplicate) an already-built year; the
    # per-(arm, year) marker makes it idempotent unless rebuild=True.
    import logging
    cfg = _cfg(tmp_path, "dlnews", min_docs=1)
    raw = Path(cfg["paths"]["raw_data_dir"]); raw.mkdir(parents=True)
    rows = [
        {"content": "the senate approved the farm policy reform bill today",
         "is_news_article": True, "title": "a"},
        {"content": "governor signed the education funding measure this morning",
         "is_news_article": True, "title": "b"},
    ]
    with gzip.open(raw / "preprocessed_google_newspaper_NY_1940.jsonl.gz", "wt",
                   encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    log = logging.getLogger("t")
    cov1 = b.build_corpus(cfg, log, arm="dlnews")
    assert cov1["new_york_1940"] == 2
    unit_dir = Path(cfg["paths"]["corpora_dir"]) / "new_york_1940"
    lines_before = sum(len(p.read_text().splitlines()) for p in unit_dir.glob("corpus_*"))

    cov2 = b.build_corpus(cfg, log, arm="dlnews")  # marker present -> skipped
    assert cov2 == {}  # nothing recounted
    lines_after = sum(len(p.read_text().splitlines()) for p in unit_dir.glob("corpus_*"))
    assert lines_after == lines_before  # no duplicate append

    cov3 = b.build_corpus(cfg, log, arm="dlnews", rebuild=True)  # forced redo
    assert cov3["new_york_1940"] == 2
    lines_rebuilt = sum(len(p.read_text().splitlines()) for p in unit_dir.glob("corpus_*"))
    assert lines_rebuilt == lines_before  # rebuilt clean, not doubled


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
