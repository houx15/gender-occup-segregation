#!/usr/bin/env python3
"""
Download pre-trained embedding archives used by Garg et al. (2018).

Sources supported:
  histwords_coha_sgns   HistWords COHA, SGNS, word-forms.   ~120 MB zip.
                        Per-decade .npy + vocab.pkl files (1810s–1990s).
                        Used for Fig 2 replication on the published vectors.
  histwords_coha_all    HistWords COHA bundle (SGNS + SVD + PPMI). ~1 GB zip.
                        Use this if you also want SVD for SI Appendix.
  glove_wiki_gigaword   GloVe vectors trained on Wikipedia 2014 + Gigaword 5.
                        ~822 MB zip → glove.6B.{50,100,200,300}d.txt.
  glove_commoncrawl     GloVe 840B/300d Common Crawl. ~2 GB zip.
  google_news_word2vec  Google News word2vec, 300d. ~1.5 GB gz. Hosted on
                        Google Drive — direct download is unreliable; this
                        source prints the canonical URL and asks you to
                        place the file manually.

Usage:
    # Download a single source
    python -m scripts.data_prep.download_pretrained_embeddings \\
        --source=histwords_coha_sgns \\
        --target_dir=data/pretrained_embeddings

    # Download every supported source
    python -m scripts.data_prep.download_pretrained_embeddings \\
        --source=all \\
        --target_dir=data/pretrained_embeddings

    # List supported sources without downloading
    python -m scripts.data_prep.download_pretrained_embeddings --list

Each source lands under {target_dir}/{source}/ and is decompressed in place
(if applicable). Re-running with the archive already present is a no-op.
"""

from __future__ import annotations

import gzip
import logging
import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import requests
import fire


# -----------------------------------------------------------------------------
# Source registry
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Source:
    name: str
    url: Optional[str]
    archive_filename: str
    decompress: str  # "zip", "gz", or "manual"
    note: str = ""


SOURCES: Dict[str, Source] = {
    "histwords_coha_sgns": Source(
        name="histwords_coha_sgns",
        url="http://snap.stanford.edu/historical_embeddings/coha-word_sgns.zip",
        archive_filename="coha-word_sgns.zip",
        decompress="zip",
        note=(
            "Hamilton et al. 2016 SGNS COHA vectors per decade — the "
            "embeddings Garg (2018) Fig 2 actually used."
        ),
    ),
    "histwords_coha_all": Source(
        name="histwords_coha_all",
        url="http://snap.stanford.edu/historical_embeddings/coha-word.zip",
        archive_filename="coha-word.zip",
        decompress="zip",
        note=(
            "Full HistWords COHA bundle (SGNS + SVD + PPMI). Use this if "
            "you also want the SVD vectors for the SI Appendix figures."
        ),
    ),
    "glove_wiki_gigaword": Source(
        name="glove_wiki_gigaword",
        url="https://nlp.stanford.edu/data/glove.6B.zip",
        archive_filename="glove.6B.zip",
        decompress="zip",
        note="GloVe Wikipedia 2014 + Gigaword 5 (50/100/200/300d).",
    ),
    "glove_commoncrawl": Source(
        name="glove_commoncrawl",
        url="https://nlp.stanford.edu/data/glove.840B.300d.zip",
        archive_filename="glove.840B.300d.zip",
        decompress="zip",
        note="GloVe Common Crawl 840B / 300d. ~2 GB compressed.",
    ),
    "google_news_word2vec": Source(
        name="google_news_word2vec",
        url=None,
        archive_filename="GoogleNews-vectors-negative300.bin.gz",
        decompress="manual",
        note=(
            "Google News word2vec is on Google Drive and not directly "
            "downloadable via plain HTTP. Get the file from "
            "https://code.google.com/archive/p/word2vec/ "
            "(or any well-known mirror), place it in the target_dir, and "
            "this script will decompress it."
        ),
    ),
}


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

def _setup_logger() -> logging.Logger:
    logger = logging.getLogger("download_pretrained")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(sh)
    return logger


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _stream_download(url: str, dest: Path, logger: logging.Logger) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        logger.info(f"  archive already at {dest} ({dest.stat().st_size:,} bytes), skipping download")
        return
    logger.info(f"  downloading {url} -> {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        written = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if not chunk:
                    continue
                f.write(chunk)
                written += len(chunk)
        logger.info(
            f"  downloaded {written:,} bytes"
            + (f" (Content-Length was {total:,})" if total else "")
        )


def _decompress_zip(archive: Path, target_dir: Path, logger: logging.Logger) -> None:
    logger.info(f"  unzipping {archive.name} into {target_dir}")
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(target_dir)


def _decompress_gz(archive: Path, target_dir: Path, logger: logging.Logger) -> None:
    out_path = target_dir / archive.name[:-3]  # strip .gz
    if out_path.exists() and out_path.stat().st_size > 0:
        logger.info(f"  already decompressed at {out_path}, skipping")
        return
    logger.info(f"  gunzipping {archive.name} -> {out_path}")
    target_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(archive, "rb") as src, open(out_path, "wb") as dst:
        shutil.copyfileobj(src, dst)


# -----------------------------------------------------------------------------
# Per-source flow
# -----------------------------------------------------------------------------

def _process_source(src: Source, target_dir: Path, logger: logging.Logger) -> None:
    src_dir = target_dir / src.name
    src_dir.mkdir(parents=True, exist_ok=True)
    archive = src_dir / src.archive_filename

    logger.info(f"=== {src.name} ===")
    if src.note:
        logger.info(f"  {src.note}")

    if src.decompress == "manual":
        if archive.exists():
            logger.info(f"  found pre-staged archive {archive}, decompressing")
            if src.archive_filename.endswith(".gz"):
                _decompress_gz(archive, src_dir, logger)
            elif src.archive_filename.endswith(".zip"):
                _decompress_zip(archive, src_dir, logger)
            else:
                logger.info("  no decompression rule for this filename, leaving as-is")
        else:
            logger.warning(
                f"  {src.name} requires manual download; place "
                f"{src.archive_filename} into {src_dir} and re-run."
            )
        return

    if src.url is None:
        logger.error(f"  {src.name}: no URL configured and decompress != 'manual'; skipping")
        return

    _stream_download(src.url, archive, logger)

    if src.decompress == "zip":
        _decompress_zip(archive, src_dir, logger)
    elif src.decompress == "gz":
        _decompress_gz(archive, src_dir, logger)
    else:
        logger.info(f"  no decompression for {src.name}")


# -----------------------------------------------------------------------------
# Entry points
# -----------------------------------------------------------------------------

def list_sources() -> None:
    """Print every supported source key with its note and archive size hint."""
    for name, src in SOURCES.items():
        url = src.url or "(manual)"
        print(f"{name:24s}  {url}")
        if src.note:
            print(f"  ↳ {src.note}")


def main(
    source: str = "all",
    target_dir: str = "data/pretrained_embeddings",
    list_only: bool = False,
) -> None:
    """Download one or all pretrained embedding archives.

    Args:
        source: One of the keys in SOURCES, or "all".
        target_dir: Root directory; each source lands under {target_dir}/{source}/.
        list_only: If True, just print the registry and exit.
    """
    if list_only or source == "list":
        list_sources()
        return

    logger = _setup_logger()
    target = Path(target_dir).resolve()
    target.mkdir(parents=True, exist_ok=True)
    logger.info(f"target_dir = {target}")

    if source == "all":
        names: List[str] = list(SOURCES)
    else:
        if source not in SOURCES:
            raise SystemExit(
                f"Unknown source '{source}'. Known: {sorted(SOURCES)} or 'all'."
            )
        names = [source]

    for name in names:
        try:
            _process_source(SOURCES[name], target, logger)
        except requests.RequestException as e:
            logger.error(f"  {name}: download failed — {e}")
        except zipfile.BadZipFile as e:
            logger.error(f"  {name}: archive corrupted — {e}")

    logger.info("done.")


if __name__ == "__main__":
    fire.Fire({"main": main, "list": list_sources})
