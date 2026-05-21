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


def test_google_news_uses_huggingface_mirror_with_sidecar():
    """The Google Drive original isn't downloadable headlessly. We use the
    fse/word2vec-google-news-300 HF mirror, which ships in gensim
    split-save format: a small .model file + a ~3.35 GB .vectors.npy sidecar.
    Both files must land in the source dir for KeyedVectors.load to work.
    """
    src = dpe.SOURCES["google_news_word2vec"]
    assert src.url is not None
    assert "huggingface.co" in src.url
    assert src.decompress == "none"
    extra_filenames = {fname for _u, fname in src.extra_files}
    assert "word2vec-google-news-300.model.vectors.npy" in extra_filenames


def test_process_source_pulls_extra_files(tmp_path, monkeypatch):
    """For sources with extra_files, _process_source must download every
    one — not just the primary archive."""
    src = dpe.SOURCES["google_news_word2vec"]
    downloaded: list[str] = []

    def fake_download(url, dest, logger):
        downloaded.append(url)
        # Touch a fake file so detect_models_dir has something to look at.
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(b"\x00")

    monkeypatch.setattr(dpe, "_stream_download", fake_download)

    logger = logging.getLogger("test_dpe_extra_files")
    entry = dpe._process_source(src, tmp_path, logger)

    assert entry["status"] == "ok"
    # Main URL + every extra URL should be hit exactly once.
    expected = [src.url] + [u for u, _f in src.extra_files]
    assert downloaded == expected


def test_main_unknown_source_raises():
    with pytest.raises(SystemExit, match="Unknown source"):
        dpe.main(source="not_a_real_source", target_dir="/tmp/ignored")


def test_process_source_returns_manifest_entry(tmp_path, monkeypatch):
    """_process_source returns a dict describing what happened (status,
    archive path, resolved models_dir)."""
    src = dpe.SOURCES["glove_wiki_gigaword"]
    src_dir = tmp_path / src.name
    src_dir.mkdir()
    archive = src_dir / src.archive_filename
    archive.write_bytes(b"fake archive bytes")

    monkeypatch.setattr(dpe, "_stream_download", lambda *a, **k: None)
    monkeypatch.setattr(dpe, "_decompress_zip", lambda *a, **k: None)

    logger = logging.getLogger("test_dpe_manifest")
    entry = dpe._process_source(src, tmp_path, logger)
    assert entry["name"] == src.name
    assert entry["status"] == "ok"
    assert entry["archive"].endswith(src.archive_filename)


def test_manual_source_status_when_archive_missing(tmp_path, caplog, monkeypatch):
    """Construct a synthetic manual-mode source so we test the manual-pending
    branch independently of the registry (which now points google_news at
    the HF mirror)."""
    src = dpe.Source(
        name="manual_test",
        url=None,
        archive_filename="manual_test.bin.gz",
        decompress="manual",
        note="synthetic",
    )
    logger = logging.getLogger("test_dpe_manual")
    logger.setLevel(logging.WARNING)
    with caplog.at_level(logging.WARNING, logger=logger.name):
        entry = dpe._process_source(src, tmp_path, logger)
    assert entry["status"] == "manual_pending"
    assert any("manual" in rec.message.lower() for rec in caplog.records), (
        f"Expected a manual-download warning; got {[r.message for r in caplog.records]}"
    )


def test_detect_models_dir_picks_directory_with_most_npy(tmp_path):
    """For HistWords sources, _detect_models_dir picks the directory with
    the most -w.npy files in the recursive scan."""
    nested = tmp_path / "extracted_root" / "sgns"
    nested.mkdir(parents=True)
    for year in (1810, 1820, 1830):
        (nested / f"{year}-w.npy").write_bytes(b"\x00")
    # A noise file in another sibling dir — only one match, should not win.
    other = tmp_path / "extracted_root" / "other"
    other.mkdir()
    (other / "1990-w.npy").write_bytes(b"\x00")

    logger = logging.getLogger("test_dpe_detect")
    out = dpe._detect_models_dir(tmp_path, "histwords_coha_sgns", logger)
    assert out == nested, f"Expected {nested}, got {out}"


def test_detect_models_dir_returns_none_when_nothing_matches(tmp_path):
    logger = logging.getLogger("test_dpe_detect_none")
    out = dpe._detect_models_dir(tmp_path, "histwords_coha_sgns", logger)
    assert out is None


def test_main_writes_manifest_and_processes_all_sources(tmp_path, monkeypatch):
    """End-to-end of main() with no real network: every source ends up in
    MANIFEST.json with a status entry."""
    monkeypatch.setattr(dpe, "_stream_download", lambda *a, **k: None)
    monkeypatch.setattr(dpe, "_decompress_zip", lambda *a, **k: None)
    monkeypatch.setattr(dpe, "_decompress_gz", lambda *a, **k: None)

    dpe.main(target_dir=str(tmp_path))

    import json
    manifest_path = tmp_path / "MANIFEST.json"
    assert manifest_path.exists()
    data = json.loads(manifest_path.read_text())
    names = {e["name"] for e in data["entries"]}
    assert names == set(dpe.SOURCES)
