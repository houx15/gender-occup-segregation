#!/usr/bin/env python3
"""
Download and decompress COHA n-gram archives.

Expects a list of ZIP archive URLs in config['coha']['source_archive_urls'].
Each URL is typically a decade-level archive that the user obtained via the
corpusdata.org email-gated signup.

Usage:
    python -m scripts.data_prep.download_coha --config=config/config.yml
    python -m scripts.data_prep.download_coha --config=config/config.yml --max_workers=4
"""

import logging
import sys
import zipfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple

import requests
import fire

from scripts.common.config_loader import load_config


def _setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("download_coha")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_dir / "download_coha.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
    logger.addHandler(sh)
    return logger


def download_one(url: str, out_path: Path, logger: logging.Logger) -> Tuple[bool, str]:
    if out_path.exists() and out_path.stat().st_size > 0:
        logger.info(f"Skipping {out_path.name} (already exists)")
        return True, "skipped"
    try:
        logger.info(f"Downloading {url} -> {out_path.name}")
        r = requests.get(url, stream=True, timeout=120)
        r.raise_for_status()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                if chunk:
                    f.write(chunk)
        logger.info(f"Downloaded {out_path.name} ({out_path.stat().st_size:,} bytes)")
        return True, "downloaded"
    except requests.RequestException as e:
        logger.error(f"Failed to download {url}: {e}")
        if out_path.exists():
            out_path.unlink()
        return False, str(e)


def decompress_one(zip_path: Path, out_dir: Path, logger: logging.Logger) -> Tuple[bool, str]:
    stem_dir = out_dir / zip_path.stem
    if stem_dir.exists() and any(stem_dir.iterdir()):
        logger.info(f"Skipping {zip_path.name} (already decompressed)")
        return True, "skipped"
    try:
        logger.info(f"Decompressing {zip_path.name} -> {out_dir}")
        out_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(out_dir)
        logger.info(f"Decompressed {zip_path.name}")
        return True, "decompressed"
    except Exception as e:
        logger.error(f"Failed to decompress {zip_path.name}: {e}")
        return False, str(e)


def main(config: str = "config/config.yml", max_workers: int = 4, skip_decompress: bool = False):
    """Download COHA n-gram archives to raw_coha_dir and decompress to coha_decompressed_dir."""
    cfg = load_config(config)
    if cfg["data_source"] != "coha":
        raise ValueError("download_coha requires data_source='coha' in config")

    urls: List[str] = cfg.get("coha", {}).get("source_archive_urls", [])
    if not urls:
        raise ValueError(
            "config.coha.source_archive_urls is empty. "
            "Paste the download URLs from your corpusdata.org signup email."
        )

    raw_dir = Path(cfg["paths"]["raw_coha_dir"])
    decomp_dir = Path(cfg["paths"]["coha_decompressed_dir"])
    log_dir = Path(cfg["paths"]["log_dir"])
    logger = _setup_logging(log_dir)

    logger.info("=" * 80)
    logger.info(f"Starting COHA download ({len(urls)} URLs)")
    logger.info("=" * 80)

    raw_dir.mkdir(parents=True, exist_ok=True)

    download_tasks = []
    for url in urls:
        filename = url.rstrip("/").split("/")[-1]
        out_path = raw_dir / filename
        download_tasks.append((url, out_path))

    results = {"ok": 0, "fail": 0}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(download_one, u, p, logger): (u, p) for u, p in download_tasks}
        for f in as_completed(futs):
            ok, _ = f.result()
            results["ok" if ok else "fail"] += 1
    logger.info(f"Downloads: {results}")

    if skip_decompress:
        logger.info("Skipping decompression as requested")
        return
    decomp_dir.mkdir(parents=True, exist_ok=True)
    zips = sorted(raw_dir.glob("*.zip"))
    dresults = {"ok": 0, "fail": 0}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(decompress_one, z, decomp_dir, logger): z for z in zips}
        for f in as_completed(futs):
            ok, _ = f.result()
            dresults["ok" if ok else "fail"] += 1
    logger.info(f"Decompressions: {dresults}")

    logger.info("=" * 80)
    logger.info("COHA download completed")
    logger.info("=" * 80)


if __name__ == "__main__":
    fire.Fire(main)
