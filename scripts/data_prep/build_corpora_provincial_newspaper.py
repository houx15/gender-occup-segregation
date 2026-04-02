#!/usr/bin/env python3
"""
Build province-year corpora from provincial newspaper text data.

Reads plain text files organized by province folder / year / month / day,
cleans and segments them with jieba, and writes corpus files per province-year unit
(e.g., 北京_2020/corpus_000000).

Usage:
    python -m scripts.data_prep.build_corpora_provincial_newspaper --config=config/config.yml
    python -m scripts.data_prep.build_corpora_provincial_newspaper --config=config/config.yml --province=北京
    python -m scripts.data_prep.build_corpora_provincial_newspaper --config=config/config.yml --group=0
    python -m scripts.data_prep.build_corpora_provincial_newspaper --config=config/config.yml --resume=True
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Set

import fire
import jieba

from scripts.common.config_loader import load_config
from scripts.common.logging_utils import setup_logging


# Province folder name → province name mapping
FOLDER_TO_PROVINCE = {
    "北京日报": "北京",
    "天津日报": "天津",
    "河北日报": "河北",
    "山西日报": "山西",
    "内蒙古日报": "内蒙古",
    "辽宁日报": "辽宁",
    "吉林日报": "吉林",
    "黑龙江日报": "黑龙江",
    "上海日报（解放日报）": "上海",
    "江苏日报": "江苏",
    "浙江日报": "浙江",
    "安徽日报": "安徽",
    "福建日报": "福建",
    "江西日报": "江西",
    "山东日报（大众日报）": "山东",
    "河南日报": "河南",
    "湖北日报": "湖北",
    "湖南日报": "湖南",
    "广东日报": "广东",
    "广西日报": "广西",
    "海南日报": "海南",
    "重庆日报": "重庆",
    "四川日报": "四川",
    "贵州日报": "贵州",
    "云南日报": "云南",
    "西藏日报": "西藏",
    "陕西日报": "陕西",
    "甘肃日报": "甘肃",
    "青海日报": "青海",
    "宁夏日报": "宁夏",
    "新疆日报": "新疆",
}

STOPWORDS = {
    "的", "是", "了", "在", "有", "和", "就", "不", "人", "都",
    "一", "一个", "上", "也", "很", "到", "说", "要", "去", "你",
    "会", "着", "没有", "看", "好", "自己", "这", "那", "我", "他",
    "她", "我们", "你们", "他们", "她们", "什么", "怎么", "这个", "那个",
    "可以", "因为", "所以", "但是", "而且", "或者", "如果", "虽然",
    "已经", "可能", "应该", "需要", "通过", "进行", "提出", "以及",
    "本报", "记者", "报道", "日前", "近日", "今天", "昨天", "今年",
}


def clean_text(text: str) -> str:
    """Clean raw newspaper text."""
    if not text:
        return ""
    text = str(text)
    text = re.sub(r'[　\s]+', ' ', text)
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'（[^）]*）', '', text)
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', text)
    return text.strip()


def segment_text(text: str) -> list:
    """Segment text with jieba and filter stopwords."""
    if not text or len(text) < 20:
        return []
    words = jieba.lcut(text, HMM=True)
    filtered = [
        w.strip() for w in words
        if w.strip() and w.strip() not in STOPWORDS
        and len(w.strip()) > 1 and not w.strip().isdigit()
    ]
    return filtered


class ProvinceYearCorpusWriter:
    """Rolling file writer per province-year unit."""

    def __init__(self, unit_name: str, output_dir: str,
                 max_bytes: int = 1024 * 1024 * 1024):
        """
        Args:
            unit_name: Province-year identifier, e.g. '北京_2020'
            output_dir: Root corpora directory
            max_bytes: Max size per corpus file (default 1GB)
        """
        self.unit_name = unit_name
        self.max_bytes = max_bytes
        self.unit_dir = os.path.join(output_dir, unit_name)
        os.makedirs(self.unit_dir, exist_ok=True)
        self.index = 0
        self.bytes_written = 0
        self.total_lines = 0
        self._open_next()

    def _open_next(self):
        while True:
            filepath = os.path.join(self.unit_dir, f"corpus_{self.index:06d}")
            if not os.path.exists(filepath):
                break
            self.index += 1
        self.file = open(filepath, 'w', buffering=8 * 1024 * 1024,
                         encoding='utf-8')
        self.bytes_written = 0

    def write(self, words: list):
        """Write a segmented line if it has >= 5 tokens."""
        if not words or len(words) < 5:
            return
        line = ' '.join(words) + '\n'
        if self.bytes_written + len(line) > self.max_bytes:
            self.file.close()
            self.index += 1
            self._open_next()
        self.file.write(line)
        self.bytes_written += len(line)
        self.total_lines += 1

    def close(self):
        self.file.close()


def _discover_province_years(raw_data_dir: str,
                             allowed_provinces: Optional[Set[str]] = None,
                             allowed_year: Optional[str] = None
                             ) -> List[tuple]:
    """
    Discover all (province, year, folder_path) tuples from the directory structure.

    Returns:
        List of (province_name, year_str, province_folder_path)
    """
    results = []
    if not os.path.isdir(raw_data_dir):
        return results

    for folder_name in sorted(os.listdir(raw_data_dir)):
        folder_path = os.path.join(raw_data_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        province = FOLDER_TO_PROVINCE.get(folder_name)
        if province is None:
            continue
        if allowed_provinces and province not in allowed_provinces:
            continue

        # Iterate year subdirectories
        try:
            entries = sorted(os.listdir(folder_path))
        except PermissionError:
            continue

        for entry in entries:
            entry_path = os.path.join(folder_path, entry)
            if not os.path.isdir(entry_path):
                continue
            # Year directories should be numeric (e.g., '2020')
            if not entry.isdigit():
                continue
            if allowed_year and entry != allowed_year:
                continue
            results.append((province, entry, folder_path))

    return results


def _collect_txt_files(province_folder_path: str, year: str) -> List[str]:
    """
    Collect all .txt files for a given province folder and year.
    Walks through year/MM/ subdirectories.
    """
    txt_files = []
    year_path = os.path.join(province_folder_path, year)
    if not os.path.isdir(year_path):
        return txt_files

    for month_name in sorted(os.listdir(year_path)):
        month_path = os.path.join(year_path, month_name)
        if not os.path.isdir(month_path):
            continue
        try:
            for fname in sorted(os.listdir(month_path)):
                if fname.endswith('.txt'):
                    txt_files.append(os.path.join(month_path, fname))
        except PermissionError:
            continue

    return txt_files


def build_corpus(config: dict, logger, province: Optional[str] = None,
                 group: Optional[int] = None, year: Optional[str] = None,
                 resume: bool = True):
    """
    Build province-year corpora from provincial newspaper text files.

    Args:
        config: Loaded configuration dictionary
        logger: Logger instance
        province: Process only this province (e.g., '北京')
        group: Process only this province group index (for SLURM parallelism)
        year: Process only this year (e.g., '2020')
        resume: Skip already-processed province-year units
    """
    raw_data_dir = config['paths']['raw_data_dir']
    corpora_dir = config['paths']['corpora_dir']
    log_dir = config['paths']['log_dir']

    # Determine which provinces to process
    allowed_provinces = None
    if province:
        allowed_provinces = {province}
    elif group is not None:
        prov_config = config.get('provincial', {})
        province_groups = prov_config.get('province_groups', [])
        if group < 0 or group >= len(province_groups):
            logger.error(f"Invalid group index {group}, "
                         f"only {len(province_groups)} groups available")
            return
        allowed_provinces = set(province_groups[group])
        logger.info(f"Processing group {group}: {allowed_provinces}")

    # Discover province-year combinations
    province_years = _discover_province_years(raw_data_dir, allowed_provinces,
                                              year)
    logger.info(f"Discovered {len(province_years)} province-year combinations")

    if not province_years:
        logger.warning("No province-year combinations found. "
                       "Check raw_data_dir and province names.")
        return

    # Resume support: load checkpoint
    processed_units: Set[str] = set()
    checkpoint_file = os.path.join(log_dir,
                                   'processed_provincial_newspaper_units.json')
    if resume and os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            processed_units = set(json.load(f))
        logger.info(f"Resume mode: {len(processed_units)} units already "
                    f"processed")

    # Filter out already-processed units
    if resume and processed_units:
        province_years = [
            (p, y, fp) for p, y, fp in province_years
            if f"{p}_{y}" not in processed_units
        ]
        logger.info(f"After resume filter: {len(province_years)} units "
                    f"remaining")

    # Process each province-year unit
    writers: Dict[str, ProvinceYearCorpusWriter] = {}
    stats = {
        'total_segments': 0,
        'skipped': 0,
        'errors': 0,
        'provinces': set(),
        'years': set(),
    }

    for prov, yr, folder_path in province_years:
        unit_name = f"{prov}_{yr}"
        stats['provinces'].add(prov)
        stats['years'].add(yr)

        logger.info(f"Processing {unit_name} ...")

        txt_files = _collect_txt_files(folder_path, yr)
        logger.info(f"  Found {len(txt_files)} daily files for {unit_name}")

        if not txt_files:
            processed_units.add(unit_name)
            continue

        # Create writer if needed
        if unit_name not in writers:
            writers[unit_name] = ProvinceYearCorpusWriter(unit_name,
                                                          corpora_dir)

        unit_segments = 0
        unit_errors = 0

        for txt_file in txt_files:
            try:
                # Skip empty files
                if os.path.getsize(txt_file) == 0:
                    continue

                with open(txt_file, 'r', encoding='utf-8',
                          errors='ignore') as f:
                    content = f.read()

                if not content.strip():
                    continue

                # Each file is a single line, tab-separated segments
                segments = content.split('\t')

                for seg in segments:
                    seg = seg.strip()
                    if not seg:
                        continue
                    cleaned = clean_text(seg)
                    if len(cleaned) < 20:
                        stats['skipped'] += 1
                        continue
                    words = segment_text(cleaned)
                    if len(words) < 5:
                        stats['skipped'] += 1
                        continue
                    writers[unit_name].write(words)
                    unit_segments += 1
                    stats['total_segments'] += 1

            except Exception as e:
                unit_errors += 1
                stats['errors'] += 1
                # Only log first few errors per unit to avoid spam
                if unit_errors <= 3:
                    logger.warning(f"  Error reading {txt_file}: {e}")

        if unit_errors > 3:
            logger.warning(f"  ... {unit_errors - 3} more errors suppressed")

        logger.info(f"  {unit_name}: {unit_segments:,} segments, "
                    f"{unit_errors} errors")

        processed_units.add(unit_name)

        # Periodic checkpoint save (every unit)
        if resume:
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(sorted(processed_units), f, ensure_ascii=False)

    # Close all writers
    for writer in writers.values():
        writer.close()

    # Log summary
    logger.info(f"Completed: {stats['total_segments']:,} segments, "
                f"{stats['skipped']:,} skipped, {stats['errors']:,} errors, "
                f"{len(stats['provinces'])} provinces, "
                f"{len(stats['years'])} years")

    # Final checkpoint save
    if resume:
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(sorted(processed_units), f, ensure_ascii=False)
        logger.info(f"Checkpoint saved: {len(processed_units)} units")


def main(config: str = 'config/config.yml', province: Optional[str] = None,
         group: Optional[int] = None, year: Optional[str] = None,
         resume: bool = True):
    """
    Build province-year corpora from provincial newspaper text data.

    Args:
        config: Path to YAML config file
        province: Process only this province (e.g., '北京')
        group: Process only this province group index (for SLURM)
        year: Process only this year (e.g., '2020')
        resume: Skip already-processed units via checkpoint
    """
    config_data = load_config(config)
    logger = setup_logging(
        Path(config_data['paths']['log_dir']),
        "build_corpora_provincial_newspaper.log"
    )

    logger.info("=" * 80)
    logger.info("Starting provincial newspaper corpus building")
    logger.info(f"  province={province}, group={group}, year={year}, "
                f"resume={resume}")
    logger.info("=" * 80)

    build_corpus(config_data, logger, province=province, group=group,
                 year=year, resume=resume)

    logger.info("=" * 80)
    logger.info("Provincial newspaper corpus building completed!")
    logger.info("=" * 80)


if __name__ == "__main__":
    fire.Fire(main)
