#!/usr/bin/env python3
"""
Download and decompress Chinese Google 5-gram data.

Usage:
    python download_ngrams.py --config=config/config.yml
    python download_ngrams.py --config=config/config.yml --skip_decompress
    python download_ngrams.py --config=config/config.yml --max_workers=8
"""

import logging
import sys
import gzip
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple
import requests
from bs4 import BeautifulSoup
import yaml
import fire


def setup_logging(log_dir: Path) -> logging.Logger:
    """Configure logging to both file and console."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "download_ngrams.log"

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def parse_index_page(index_url: str, file_pattern: str, logger: logging.Logger) -> List[str]:
    """Parse the index page and extract all download URLs matching the pattern."""
    logger.info(f"Fetching index page: {index_url}")

    try:
        response = requests.get(index_url, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"Failed to fetch index page: {e}")
        raise

    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract all links
    download_urls = []
    base_url = "https://storage.googleapis.com/books/ngrams/books/20200217/chi_sim/"

    for link in soup.find_all('a'):
        href = link.get('href')
        if href and href.endswith('.gz') and '5gram' in href:
            if not href.startswith('http'):
                href = base_url + href
            download_urls.append(href)

    logger.info(f"Found {len(download_urls)} files matching pattern")

    for i, url in enumerate(download_urls[:5]):
        logger.info(f"  Sample URL {i+1}: {url}")

    return download_urls


def download_file(url: str, output_path: Path, logger: logging.Logger) -> Tuple[bool, str]:
    """Download a single file with progress logging."""
    filename = output_path.name

    # Check if file already exists
    if output_path.exists():
        file_size = output_path.stat().st_size
        if file_size > 0:
            logger.info(f"Skipping {filename} (already exists, {file_size:,} bytes)")
            return True, f"Already exists: {filename}"

    try:
        logger.info(f"Downloading {filename}...")
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()

        with open(output_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)

        logger.info(f"Completed {filename} ({downloaded:,} bytes)")
        return True, f"Downloaded: {filename}"

    except requests.RequestException as e:
        logger.error(f"Failed to download {filename}: {e}")
        if output_path.exists():
            output_path.unlink()
        return False, f"Failed: {filename} - {str(e)}"


def decompress_file(gz_path: Path, output_path: Path, logger: logging.Logger) -> Tuple[bool, str]:
    """Decompress a gzip file."""
    filename = gz_path.name

    # Check if already decompressed
    if output_path.exists():
        file_size = output_path.stat().st_size
        if file_size > 0:
            logger.info(f"Skipping decompression of {filename} (already exists)")
            return True, f"Already decompressed: {filename}"

    try:
        logger.info(f"Decompressing {filename}...")

        with gzip.open(gz_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        output_size = output_path.stat().st_size
        logger.info(f"Completed decompressing {filename} ({output_size:,} bytes)")
        return True, f"Decompressed: {filename}"

    except Exception as e:
        logger.error(f"Failed to decompress {filename}: {e}")
        if output_path.exists():
            output_path.unlink()
        return False, f"Failed to decompress: {filename} - {str(e)}"


def main(config='config/config.yml', skip_decompress=False, max_workers=4):
    """
    Download and decompress Chinese Google 5-gram data.

    Args:
        config: Path to configuration file
        skip_decompress: Skip decompression step
        max_workers: Maximum number of parallel downloads/decompressions
    """
    # Load configuration
    config_data = load_config(config)

    # Setup paths
    raw_dir = Path(config_data['paths']['raw_ngram_dir'])
    decompressed_dir = Path(config_data['paths']['decompressed_dir'])
    log_dir = Path(config_data['paths']['log_dir'])

    raw_dir.mkdir(parents=True, exist_ok=True)
    if not skip_decompress:
        decompressed_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(log_dir)

    logger.info("="*80)
    logger.info("Starting Chinese Google 5-gram download")
    logger.info("="*80)

    # Parse index page
    index_url = config_data['ngram']['index_url']
    file_pattern = config_data['ngram']['file_pattern']

    download_urls = parse_index_page(index_url, file_pattern, logger)

    if not download_urls:
        logger.error("No download URLs found!")
        return 1

    # Download files
    logger.info(f"\nStarting download of {len(download_urls)} files with {max_workers} workers...")

    download_tasks = []
    for url in download_urls:
        filename = url.split('/')[-1]
        output_path = raw_dir / filename
        download_tasks.append((url, output_path))

    download_results = {'success': 0, 'failed': 0, 'skipped': 0}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(download_file, url, path, logger): (url, path)
            for url, path in download_tasks
        }

        for future in as_completed(futures):
            success, message = future.result()
            if success:
                if "Already exists" in message:
                    download_results['skipped'] += 1
                else:
                    download_results['success'] += 1
            else:
                download_results['failed'] += 1

    logger.info("\nDownload Summary:")
    logger.info(f"  Successful: {download_results['success']}")
    logger.info(f"  Skipped (already exists): {download_results['skipped']}")
    logger.info(f"  Failed: {download_results['failed']}")

    # Decompress files
    if not skip_decompress:
        logger.info(f"\nStarting decompression with {max_workers} workers...")

        decompress_tasks = []
        for gz_file in raw_dir.glob("*.gz"):
            output_name = gz_file.stem
            output_path = decompressed_dir / output_name
            decompress_tasks.append((gz_file, output_path))

        decompress_results = {'success': 0, 'failed': 0, 'skipped': 0}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(decompress_file, gz_path, out_path, logger): (gz_path, out_path)
                for gz_path, out_path in decompress_tasks
            }

            for future in as_completed(futures):
                success, message = future.result()
                if success:
                    if "Already decompressed" in message:
                        decompress_results['skipped'] += 1
                    else:
                        decompress_results['success'] += 1
                else:
                    decompress_results['failed'] += 1

        logger.info("\nDecompression Summary:")
        logger.info(f"  Successful: {decompress_results['success']}")
        logger.info(f"  Skipped (already exists): {decompress_results['skipped']}")
        logger.info(f"  Failed: {decompress_results['failed']}")

    logger.info("\n" + "="*80)
    logger.info("Download process completed!")
    logger.info("="*80)

    return 0


if __name__ == "__main__":
    fire.Fire(main)
