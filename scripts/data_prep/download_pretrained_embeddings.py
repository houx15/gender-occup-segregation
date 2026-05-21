#!/usr/bin/env python3
"""
One-shot downloader + preparer for the pre-trained embedding archives Garg
et al. (2018) reference. Designed to be invoked with no arguments — it'll
fetch every source it knows how to fetch, decompress what's compressible,
and write a MANIFEST.json describing exactly where each source's model
files ended up so you can pin profiles to those paths.

Sources supported:
  histwords_coha_sgns   HistWords COHA, SGNS, word-forms.   ~120 MB zip.
                        Per-decade .npy + vocab.pkl files (1810s–1990s).
                        Used for Fig 2 replication on the published vectors.
  histwords_coha_all    HistWords COHA bundle (SGNS + SVD + PPMI). ~1 GB zip.
                        Use this if you also want SVD for SI Appendix.
  google_ngram_eng_all  HistWords Google Books Ngram "English (All)" SGNS, per
                        decade 1800s–1990s. Same .npy + vocab.pkl layout as
                        COHA. NOT US-specific — broader-English scale check.
  google_ngram_eng_fiction_all
                        HistWords Google Books Ngram "English Fiction" SGNS,
                        per decade 1800s–1990s. Fiction register companion.
  glove_wiki_gigaword   GloVe Wikipedia 2014 + Gigaword 5. ~822 MB zip →
                        glove.6B.{50,100,200,300}d.txt.
  glove_commoncrawl     GloVe 840B/300d Common Crawl. ~2 GB zip.
  google_news_word2vec  Google News word2vec, 300d. ~1.5 GB gz. Hosted on
                        Google Drive; direct download is unreliable, so
                        the script just notes it as manual and continues.

Usage:
    # The lazy invocation: fetch + unzip + record everything we know about.
    python -m scripts.data_prep.download_pretrained_embeddings

    # Pick a target dir (e.g. on /scratch on Slurm clusters)
    python -m scripts.data_prep.download_pretrained_embeddings \\
        --target_dir=/scratch/network/yh6580/gender-occup/data/pretrained_embeddings

    # Single source
    python -m scripts.data_prep.download_pretrained_embeddings \\
        --source=histwords_coha_sgns

    # List sources without downloading
    python -m scripts.data_prep.download_pretrained_embeddings list

Each source lands under {target_dir}/{source}/ and is decompressed in place.
After unzipping the script scans for the actual model files (HistWords
*-w.npy, GloVe glove.*.txt, word2vec *.bin) and records the resolved
"models_dir" path in MANIFEST.json — that's the value to plug into the
matching config profile.

Re-running is safe: archives that already exist with non-zero size are
skipped, decompressed contents are not re-extracted if already present.
"""

from __future__ import annotations

import gzip
import json
import logging
import shutil
import sys
import time
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
    decompress: str  # "zip", "gz", "none", or "manual"
    note: str = ""
    # Additional companion files to fetch alongside the main archive (used for
    # multi-file save formats like gensim's KeyedVectors split into
    # .model + .model.vectors.npy). Each entry is (url, dest_filename).
    extra_files: tuple = ()


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
    "google_ngram_eng_all": Source(
        name="google_ngram_eng_all",
        url="http://snap.stanford.edu/historical_embeddings/eng-all_sgns.zip",
        archive_filename="eng-all_sgns.zip",
        decompress="zip",
        note=(
            "HistWords Google Books Ngram 'English (All)' SGNS vectors per "
            "decade (1800s–1990s). Same .npy + vocab.pkl layout as the COHA "
            "vectors. Note: 'All English', NOT US-specific — a broader-English, "
            "larger-scale companion to the COHA arms, not American per se."
        ),
    ),
    "google_ngram_eng_fiction_all": Source(
        name="google_ngram_eng_fiction_all",
        url="http://snap.stanford.edu/historical_embeddings/eng-fiction-all_sgns.zip",
        archive_filename="eng-fiction-all_sgns.zip",
        decompress="zip",
        note=(
            "HistWords Google Books Ngram 'English Fiction (All)' SGNS vectors "
            "per decade (1800s–1990s). Same layout as eng-all; fiction register."
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
        url=(
            "https://huggingface.co/fse/word2vec-google-news-300/resolve/main/"
            "word2vec-google-news-300.model"
        ),
        archive_filename="word2vec-google-news-300.model",
        decompress="none",
        extra_files=(
            (
                "https://huggingface.co/fse/word2vec-google-news-300/resolve/main/"
                "word2vec-google-news-300.model.vectors.npy",
                "word2vec-google-news-300.model.vectors.npy",
            ),
        ),
        note=(
            "Google News word2vec, 300d, via the fse/word2vec-google-news-300 "
            "Hugging Face mirror (gensim KeyedVectors split-save format: a "
            "small .model file + a ~3.35 GB sidecar .vectors.npy). Load with "
            "gensim.models.KeyedVectors.load(<.model path>); the sidecar is "
            "auto-discovered. Total download is ~3.5 GB."
        ),
    ),
}


# Heuristics for detecting the actual model directory after extraction.
# Each tuple is (glob_pattern, description_for_logs).
_DETECT_PATTERNS: Dict[str, List[tuple[str, str]]] = {
    "histwords_coha_sgns": [("*-w.npy", "HistWords {YYYY}-w.npy files")],
    "histwords_coha_all":  [("*-w.npy", "HistWords {YYYY}-w.npy files")],
    "google_ngram_eng_all":          [("*-w.npy", "HistWords {YYYY}-w.npy files")],
    "google_ngram_eng_fiction_all":  [("*-w.npy", "HistWords {YYYY}-w.npy files")],
    "glove_wiki_gigaword": [("glove.*.txt", "GloVe text-format vectors")],
    "glove_commoncrawl":   [("glove.*.txt", "GloVe text-format vectors")],
    "google_news_word2vec": [("*.model", "gensim KeyedVectors split-save"),
                             ("*.bin", "word2vec binary")],
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
# Download with progress
# -----------------------------------------------------------------------------

def _format_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _stream_download(url: str, dest: Path, logger: logging.Logger) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        logger.info(f"  archive already at {dest} ({_format_bytes(dest.stat().st_size)}), skipping download")
        return
    logger.info(f"  downloading {url} -> {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        written = 0
        last_log = time.time()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):  # 1 MB chunks
                if not chunk:
                    continue
                f.write(chunk)
                written += len(chunk)
                # Log progress every 5s (avoids spamming for fast/slow links).
                now = time.time()
                if now - last_log >= 5.0:
                    if total:
                        pct = 100.0 * written / total
                        logger.info(
                            f"    progress {_format_bytes(written)} / "
                            f"{_format_bytes(total)} ({pct:.1f}%)"
                        )
                    else:
                        logger.info(f"    progress {_format_bytes(written)}")
                    last_log = now
        logger.info(
            f"  download complete: {_format_bytes(written)}"
            + (f" (Content-Length was {_format_bytes(total)})" if total else "")
        )


def _decompress_zip(archive: Path, target_dir: Path, logger: logging.Logger) -> None:
    # Skip extraction if any non-archive file already lives under target_dir.
    if target_dir.exists():
        existing = [
            p for p in target_dir.rglob("*")
            if p.is_file() and p != archive and p.name != "MANIFEST.json"
        ]
        if existing:
            logger.info(
                f"  zip looks already extracted ({len(existing)} non-archive "
                f"files under {target_dir}), skipping unzip"
            )
            return
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
# Post-extraction detection
# -----------------------------------------------------------------------------

def _detect_models_dir(src_dir: Path, source_name: str, logger: logging.Logger) -> Optional[Path]:
    """Walk ``src_dir`` and find the directory containing the most files
    matching this source's recognised pattern. Returns None if nothing matches.

    For HistWords this picks the directory containing the most -w.npy files;
    for GloVe, the directory containing glove.*.txt; etc.
    """
    patterns = _DETECT_PATTERNS.get(source_name, [])
    if not patterns or not src_dir.exists():
        return None

    best_dir: Optional[Path] = None
    best_count = 0
    for pattern, _label in patterns:
        # Group matches by parent directory; pick the parent with the most hits.
        by_parent: Dict[Path, int] = {}
        for hit in src_dir.rglob(pattern):
            if hit.is_file():
                by_parent[hit.parent] = by_parent.get(hit.parent, 0) + 1
        if not by_parent:
            continue
        parent, count = max(by_parent.items(), key=lambda kv: kv[1])
        if count > best_count:
            best_dir = parent
            best_count = count

    if best_dir is not None:
        logger.info(f"  detected models_dir = {best_dir}  ({best_count} matched files)")
    else:
        logger.info(f"  could not detect a models_dir for {source_name} under {src_dir}")
    return best_dir


# -----------------------------------------------------------------------------
# Per-source flow
# -----------------------------------------------------------------------------

def _process_source(src: Source, target_dir: Path, logger: logging.Logger) -> dict:
    """Run download → decompress → detect for one source. Returns a manifest
    entry describing the outcome (status, archive path, resolved models_dir)."""
    src_dir = target_dir / src.name
    src_dir.mkdir(parents=True, exist_ok=True)
    archive = src_dir / src.archive_filename

    entry = {
        "name": src.name,
        "url": src.url,
        "archive": str(archive),
        "decompress": src.decompress,
        "status": "pending",
        "models_dir": None,
        "note": src.note,
    }

    logger.info(f"=== {src.name} ===")
    if src.note:
        logger.info(f"  {src.note}")

    if src.decompress == "manual":
        if archive.exists():
            logger.info(f"  found pre-staged archive {archive}, decompressing")
            try:
                if src.archive_filename.endswith(".gz"):
                    _decompress_gz(archive, src_dir, logger)
                elif src.archive_filename.endswith(".zip"):
                    _decompress_zip(archive, src_dir, logger)
                entry["status"] = "ok"
            except (OSError, zipfile.BadZipFile, EOFError) as e:
                logger.error(f"  decompress failed for {src.name}: {e}")
                entry["status"] = f"decompress_failed: {e}"
        else:
            logger.warning(
                f"  manual source — drop {src.archive_filename} into {src_dir} "
                "and re-run to decompress."
            )
            entry["status"] = "manual_pending"
    else:
        if src.url is None:
            logger.error(f"  {src.name}: no URL configured; skipping")
            entry["status"] = "no_url"
        else:
            try:
                _stream_download(src.url, archive, logger)
                # Pull any companion files (e.g. gensim split-save sidecars).
                for extra_url, extra_fname in src.extra_files:
                    extra_dest = src_dir / extra_fname
                    _stream_download(extra_url, extra_dest, logger)
                if src.decompress == "zip":
                    _decompress_zip(archive, src_dir, logger)
                elif src.decompress == "gz":
                    _decompress_gz(archive, src_dir, logger)
                # decompress == "none": no further work; main + sidecars are
                # the final files (e.g. gensim .model + .model.vectors.npy).
                entry["status"] = "ok"
            except requests.RequestException as e:
                logger.error(f"  download failed for {src.name}: {e}")
                entry["status"] = f"download_failed: {e}"
            except zipfile.BadZipFile as e:
                logger.error(f"  zip corrupted for {src.name}: {e}")
                entry["status"] = f"bad_zip: {e}"

    if entry["status"] == "ok":
        models_dir = _detect_models_dir(src_dir, src.name, logger)
        if models_dir is not None:
            entry["models_dir"] = str(models_dir)

    return entry


# -----------------------------------------------------------------------------
# Manifest
# -----------------------------------------------------------------------------

def _write_manifest(target_dir: Path, entries: List[dict], logger: logging.Logger) -> Path:
    manifest_path = target_dir / "MANIFEST.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"entries": entries}, f, indent=2)
    logger.info(f"manifest written to {manifest_path}")
    return manifest_path


def _print_summary(entries: List[dict], logger: logging.Logger) -> None:
    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    for e in entries:
        status = e["status"]
        models_dir = e["models_dir"] or "(not detected)"
        logger.info(f"  {e['name']:24s}  status={status}")
        logger.info(f"    models_dir: {models_dir}")
    logger.info("=" * 80)


# -----------------------------------------------------------------------------
# Entry points
# -----------------------------------------------------------------------------

def list_sources() -> None:
    """Print every supported source with its URL and one-line description."""
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
    """Download + decompress + record locations for one or all pretrained
    embedding archives. Default is "all" — running this with no args fetches
    every source we know about into data/pretrained_embeddings/.

    Args:
        source: Source key from SOURCES, or "all".
        target_dir: Root directory; each source lands under
                    {target_dir}/{source}/ and the manifest at
                    {target_dir}/MANIFEST.json.
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

    entries: List[dict] = []
    for name in names:
        entry = _process_source(SOURCES[name], target, logger)
        entries.append(entry)

    _write_manifest(target, entries, logger)
    _print_summary(entries, logger)
    logger.info("done.")


if __name__ == "__main__":
    # Single-entry CLI: invoking with no args runs main(source="all") into the
    # default target_dir. Pass --source=list to print the registry, or
    # --source=<name> for a single source.
    fire.Fire(main)
