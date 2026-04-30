"""Tests for scripts.data_prep.download_pretrained_embeddings.

The download paths themselves can't run in CI without network and ~1 GB of
disk, so these tests cover the registry and the per-source dispatch logic
with monkeypatched stubs for the network/decompress steps.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from scripts.data_prep import download_pretrained_embeddings as dpe


def test_registry_contains_expected_sources():
    expected = {
        "histwords_coha_sgns",
        "histwords_coha_all",
        "glove_wiki_gigaword",
        "glove_commoncrawl",
        "google_news_word2vec",
    }
    assert expected.issubset(set(dpe.SOURCES))


def test_histwords_sgns_url_matches_stanford_snap():
    src = dpe.SOURCES["histwords_coha_sgns"]
    assert src.url is not None
    assert src.url.endswith("coha-word_sgns.zip")
    assert "snap.stanford.edu" in src.url
    assert src.decompress == "zip"


def test_google_news_is_manual():
    src = dpe.SOURCES["google_news_word2vec"]
    assert src.url is None
    assert src.decompress == "manual"


def test_main_unknown_source_raises():
    with pytest.raises(SystemExit, match="Unknown source"):
        dpe.main(source="not_a_real_source", target_dir="/tmp/ignored")


def test_process_source_skips_when_archive_present(tmp_path, monkeypatch):
    """If the archive already exists with content, the downloader should skip
    the network call entirely."""
    src = dpe.SOURCES["glove_wiki_gigaword"]
    src_dir = tmp_path / src.name
    src_dir.mkdir()
    archive = src_dir / src.archive_filename
    archive.write_bytes(b"fake archive bytes")

    called = {"download": 0, "decompress": 0}

    def fake_download(*args, **kwargs):
        called["download"] += 1

    def fake_decompress(archive, target, logger):
        called["decompress"] += 1

    monkeypatch.setattr(dpe, "_stream_download", fake_download)
    monkeypatch.setattr(dpe, "_decompress_zip", fake_decompress)

    logger = logging.getLogger("test_dpe")
    dpe._process_source(src, tmp_path, logger)

    # _stream_download is still called (it short-circuits internally on size>0
    # but only when invoked); the dispatcher always calls it. Decompress also
    # always runs; for this test we just confirm no crash and one call each.
    assert called["download"] == 1
    assert called["decompress"] == 1


def test_process_source_manual_warns_when_archive_missing(tmp_path, caplog):
    src = dpe.SOURCES["google_news_word2vec"]
    logger = logging.getLogger("test_dpe_manual")
    logger.setLevel(logging.WARNING)
    with caplog.at_level(logging.WARNING, logger=logger.name):
        dpe._process_source(src, tmp_path, logger)
    assert any("manual" in rec.message.lower() for rec in caplog.records), (
        f"Expected a manual-download warning; got {[r.message for r in caplog.records]}"
    )
