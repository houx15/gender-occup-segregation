#!/usr/bin/env python3
"""
Build province-level corpora from newspaper JSONL data.

Migrated from reference/gender_norms/newspaper_corpus_builder.py.

Usage:
    python -m scripts.data_prep.build_corpora_newspaper --config=config/config.yml
    python -m scripts.data_prep.build_corpora_newspaper --config=config/config.yml --max_files=10
"""

import json
import os
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict

import fire
import jieba

from scripts.common.config_loader import load_config
from scripts.common.logging_utils import setup_logging


STOPWORDS = {
    "的", "是", "了", "在", "有", "和", "就", "不", "人", "都",
    "一", "一个", "上", "也", "很", "到", "说", "要", "去", "你",
    "会", "着", "没有", "看", "好", "自己", "这", "那", "我", "他",
    "她", "我们", "你们", "他们", "她们", "什么", "怎么", "这个", "那个",
    "可以", "因为", "所以", "但是", "而且", "或者", "如果", "虽然",
    "已经", "可能", "应该", "需要", "通过", "进行", "提出", "以及",
    "本报", "记者", "报道", "日前", "近日", "今天", "昨天", "今年",
}


def load_mapping(mapping_file: str) -> Dict[str, str]:
    """Load newspaper -> province mapping."""
    with open(mapping_file, 'r', encoding='utf-8') as f:
        mapping_data = json.load(f)

    mapping = {}
    for newspaper, info in mapping_data.items():
        province = info['province']
        if province not in ('全国', '行业', '未知'):
            mapping[newspaper] = province
    return mapping


def clean_text(text: str) -> str:
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
    if not text or len(text) < 20:
        return []
    words = jieba.lcut(text, HMM=True)
    filtered = [
        w.strip() for w in words
        if w.strip() and w.strip() not in STOPWORDS
        and len(w.strip()) > 1 and not w.strip().isdigit()
    ]
    return filtered


class ProvinceCorpusWriter:
    """Rolling file writer per province."""
    def __init__(self, province, output_dir, max_bytes=1024 * 1024 * 1024):
        self.province = province
        self.max_bytes = max_bytes
        self.province_dir = os.path.join(output_dir, province)
        os.makedirs(self.province_dir, exist_ok=True)
        self.index = 0
        self.bytes_written = 0
        self.total_lines = 0
        self._open_next()

    def _open_next(self):
        while True:
            filepath = os.path.join(self.province_dir, f"corpus_{self.index:06d}")
            if not os.path.exists(filepath):
                break
            self.index += 1
        self.file = open(filepath, 'w', buffering=8 * 1024 * 1024, encoding='utf-8')
        self.bytes_written = 0

    def write(self, words):
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


def build_corpus(config, logger, max_files=None, min_article_length=50,
                 resume=True, file_list=None):
    """Build province-level newspaper corpora."""
    raw_data_dir = config['paths']['raw_data_dir']
    corpora_dir = config['paths']['corpora_dir']

    # Load newspaper -> province mapping
    prov_config = config.get('provincial', {})
    mapping_file = prov_config.get(
        'newspaper_mapping_file',
        os.path.join(Path(config['paths']['base_dir']), 'provincial', 'data', 'newspaper_province_mapping.json')
    )
    mapping = load_mapping(mapping_file)
    logger.info(f"Loaded mapping: {len(mapping)} newspapers -> provinces")

    # Get files to process
    log_dir = config['paths']['log_dir']
    if file_list and os.path.exists(file_list):
        with open(file_list, 'r', encoding='utf-8') as f:
            all_files = [line.strip() for line in f if line.strip()]
    else:
        all_files = sorted([f for f in os.listdir(raw_data_dir) if f.endswith('.json')])

    # Resume support
    processed_files = set()
    if resume:
        checkpoint_file = os.path.join(log_dir, 'processed_newspaper_files.json')
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                processed_files = set(json.load(f))
            logger.info(f"Resume mode: {len(processed_files)} files already processed")

    if resume and processed_files:
        all_files = [f for f in all_files if f not in processed_files]

    if max_files:
        all_files = all_files[:max_files]

    logger.info(f"Processing {len(all_files)} files")

    province_writers: Dict[str, ProvinceCorpusWriter] = {}
    stats = {'total_articles': 0, 'skipped': 0, 'errors': 0}

    for filename in all_files:
        filepath = os.path.join(raw_data_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        article = json.loads(line)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        stats['errors'] += 1
                        continue
                    try:
                        source = article.get('source', '').strip()
                        text = article.get('pdf_txt', '')
                        if source not in mapping:
                            stats['skipped'] += 1
                            continue
                        province = mapping[source]
                        cleaned = clean_text(text)
                        if len(cleaned) < min_article_length:
                            stats['skipped'] += 1
                            continue
                        words = segment_text(cleaned)
                        if len(words) < 5:
                            stats['skipped'] += 1
                            continue
                        if province not in province_writers:
                            province_writers[province] = ProvinceCorpusWriter(province, corpora_dir)
                        province_writers[province].write(words)
                        stats['total_articles'] += 1
                    except Exception:
                        stats['errors'] += 1
        except Exception as e:
            logger.error(f"Failed to read {filename}: {e}")
            stats['errors'] += 1

        processed_files.add(filename)
        if len(processed_files) % 100 == 0 and resume:
            checkpoint_file = os.path.join(log_dir, 'processed_newspaper_files.json')
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(list(processed_files), f, ensure_ascii=False)

    for writer in province_writers.values():
        writer.close()

    logger.info(f"Completed: {stats['total_articles']:,} articles, "
                f"{stats['skipped']:,} skipped, {stats['errors']:,} errors, "
                f"{len(province_writers)} provinces")

    if resume:
        checkpoint_file = os.path.join(log_dir, 'processed_newspaper_files.json')
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(list(processed_files), f, ensure_ascii=False)


def main(config='config/config.yml', max_files=None, resume=True, file_list=None):
    """Build province-level corpora from newspaper data."""
    config_data = load_config(config)
    logger = setup_logging(Path(config_data['paths']['log_dir']), "build_corpora_newspaper.log")

    logger.info("=" * 80)
    logger.info("Starting newspaper corpus building")
    logger.info("=" * 80)

    build_corpus(config_data, logger, max_files=max_files, resume=resume, file_list=file_list)

    logger.info("=" * 80)
    logger.info("Newspaper corpus building completed!")
    logger.info("=" * 80)


if __name__ == "__main__":
    fire.Fire(main)
