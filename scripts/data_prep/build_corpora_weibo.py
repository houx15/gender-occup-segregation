#!/usr/bin/env python3
"""
Build province-level corpora from Weibo data.

Migrated from reference/gender_norms/gender_embedding_trainer.py.

Supports two data formats:
- 2020: cleaned_weibo_cov/{year}/ parquet files with province codes
- 2024: cleaned_weibo_data/ parquet files with region_name

Usage:
    python -m scripts.data_prep.build_corpora_weibo --config=config/config.yml --year=2024 --group=0
"""

import os
import gc
import glob
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timedelta

import pandas as pd
import fire

from scripts.common.config_loader import load_config
from scripts.common.logging_utils import setup_logging
from scripts.common.preprocessing import preprocess


# Province code mapping (GB/T 2260)
PROVINCE_CODE_TO_NAME = {
    "11": "北京", "12": "天津", "13": "河北", "14": "山西", "15": "内蒙古",
    "21": "辽宁", "22": "吉林", "23": "黑龙江",
    "31": "上海", "32": "江苏", "33": "浙江", "34": "安徽", "35": "福建",
    "36": "江西", "37": "山东",
    "41": "河南", "42": "湖北", "43": "湖南", "44": "广东", "45": "广西",
    "46": "海南",
    "50": "重庆", "51": "四川", "52": "贵州", "53": "云南", "54": "西藏",
    "61": "陕西", "62": "甘肃", "63": "青海", "64": "宁夏", "65": "新疆",
    "71": "中国台湾", "81": "中国香港", "82": "中国澳门",
}
PROVINCE_NAME_TO_CODE = {v: k for k, v in PROVINCE_CODE_TO_NAME.items()}

# Default province groups for parallel execution
PROVINCE_GROUPS = [
    ["北京", "天津", "河北", "山西"],
    ["内蒙古", "辽宁", "吉林", "黑龙江"],
    ["上海", "江苏"],
    ["福建", "江西", "山东"],
    ["河南", "湖北", "湖南"],
    ["广西", "海南"],
    ["重庆", "四川", "贵州", "云南"],
    ["西藏", "陕西", "甘肃", "青海", "宁夏", "新疆"],
    ["中国台湾", "中国香港", "中国澳门"],
    ["浙江", "安徽"],
    ["广东"],
]


def clean_region_name(region):
    if pd.isna(region) or region == "":
        return None
    region = str(region).strip()
    if region.startswith("发布于 "):
        region = region[4:].strip()
    return region if region else None


def convert_province_code(code):
    if pd.isna(code):
        return None
    code_str = str(int(code)) if isinstance(code, (int, float)) else str(code).strip()
    return PROVINCE_CODE_TO_NAME.get(code_str)


class RollingFile:
    """Rolling file writer that splits at max_bytes."""
    def __init__(self, base_dir, prefix="corpus", max_bytes=1024 * 1024 * 1024):
        self.base_dir = base_dir
        self.prefix = prefix
        self.max_bytes = max_bytes
        self.index = 0
        self.bytes_written = 0
        self._open_next()

    def _open_next(self):
        while True:
            path = os.path.join(self.base_dir, f"{self.prefix}_{self.index:06d}")
            if not os.path.exists(path):
                break
            self.index += 1
        self.f = open(path, "w", buffering=8 * 1024 * 1024, encoding="utf-8")
        self.bytes_written = 0

    def write(self, text):
        if self.bytes_written + len(text) > self.max_bytes:
            self.f.close()
            self.index += 1
            self._open_next()
        self.f.write(text + "\n")
        self.bytes_written += len(text)

    def close(self):
        self.f.close()


def prepare_2024(config, logger, year, start_month, end_month):
    """Prepare 2024 Weibo data by province and date."""
    raw_data_dir = config['paths']['raw_data_dir']
    corpora_dir = Path(config['paths']['corpora_dir'])
    allowed_provinces = list(PROVINCE_CODE_TO_NAME.values())

    start_date = datetime(year, start_month, 1)
    if end_month == 12:
        end_date = datetime(year, 12, 31)
    else:
        end_date = datetime(year, end_month + 1, 1) - timedelta(days=1)

    date_range = [start_date + timedelta(days=n) for n in range((end_date - start_date).days + 1)]

    for current_date in date_range:
        date_str = current_date.strftime("%Y-%m-%d")
        logger.info(f"Processing date: {date_str}")

        pattern = os.path.join(raw_data_dir, f"{date_str}.parquet.temp_*")
        temp_files = sorted(glob.glob(pattern))
        if not temp_files:
            continue

        province_data = defaultdict(list)
        for temp_file in temp_files:
            try:
                df = pd.read_parquet(temp_file)
                if "weibo_content" not in df.columns or "region_name" not in df.columns:
                    continue
                df = df.dropna(subset=["weibo_id", "weibo_content", "region_name"])
                # Strip retweet prefix (text after "//" is the quoted original)
                df["weibo_content"] = df["weibo_content"].astype(str).str.split("//").str[0].str.strip()
                df = df[df["weibo_content"].str.len() > 5]
                df["province"] = df["region_name"].apply(clean_region_name)
                df = df.dropna(subset=["province"])
                df = df[df["province"].isin(allowed_provinces)]
                df = df[["weibo_id", "weibo_content", "province"]]
                for province in df["province"].unique():
                    province_data[province].append(df[df["province"] == province].copy())
                del df
            except Exception as e:
                logger.error(f"Failed to read {temp_file}: {e}")

        for province, data_list in province_data.items():
            combined = pd.concat(data_list, ignore_index=True).drop_duplicates(subset=["weibo_id"])
            province_dir = corpora_dir / province
            province_dir.mkdir(parents=True, exist_ok=True)
            output_file = province_dir / f"{date_str}.parquet"
            combined.to_parquet(output_file, index=False)
            logger.info(f"  {province}: {len(combined):,} posts -> {output_file}")
            del combined
        gc.collect()


def clean_for_word2vec(config, logger, provinces):
    """Convert prepared parquet data to tokenized corpus files."""
    corpora_dir = Path(config['paths']['corpora_dir'])
    corpus_cfg = config.get("corpus", {})
    for province in provinces:
        province_dir = corpora_dir / province
        if not province_dir.exists():
            continue
        parquet_files = sorted(province_dir.glob("*.parquet"))
        if not parquet_files:
            continue
        roller = RollingFile(str(province_dir))
        total = 0
        for pq_path in parquet_files:
            try:
                df = pd.read_parquet(pq_path, columns=["weibo_content"])
                lines = []
                for raw_text in df["weibo_content"].dropna():
                    tokens = preprocess(
                        str(raw_text),
                        language=config.get("language", "zh"),
                        tokenizer=corpus_cfg.get("tokenizer", "jieba"),
                        stopwords_key=corpus_cfg.get("stopwords", "zh_weibo"),
                        lowercase=corpus_cfg.get("lowercase", False),
                        min_words=corpus_cfg.get("min_words", 5),
                        cleaner_opts={"strip_mentions": True, "strip_parens": True},
                    )
                    if tokens is None:
                        continue
                    lines.append(" ".join(tokens))
                if lines:
                    roller.write("\n".join(lines))
                    total += len(lines)
                os.remove(pq_path)
            except Exception as e:
                logger.error(f"Error processing {pq_path}: {e}")
        roller.close()
        logger.info(f"  {province}: {total} lines")


def build_2020(config, logger, year, province_filter=None):
    """Build corpora from 2020 Weibo data (province codes in parquet)."""
    raw_data_dir = config['paths']['raw_data_dir']
    corpora_dir = Path(config['paths']['corpora_dir'])
    year_dir = os.path.join(raw_data_dir, str(year))

    if not os.path.exists(year_dir):
        logger.error(f"Data directory not found: {year_dir}")
        return

    parquet_files = sorted(glob.glob(os.path.join(year_dir, "*.parquet")))
    logger.info(f"Found {len(parquet_files)} parquet files for {year}")

    corpus_cfg = config.get("corpus", {})
    province_writers = {}
    for file_path in parquet_files:
        try:
            df = pd.read_parquet(file_path, columns=["province", "weibo_content"])
            df = df.dropna(subset=["province", "weibo_content"])
            df["province"] = df["province"].apply(convert_province_code)
            df = df.dropna(subset=["province"])

            if province_filter:
                df = df[df["province"] == province_filter]

            for province in df["province"].unique():
                prov_data = df[df["province"] == province]
                lines = []
                for raw_text in prov_data["weibo_content"].dropna():
                    tokens = preprocess(
                        str(raw_text),
                        language=config.get("language", "zh"),
                        tokenizer=corpus_cfg.get("tokenizer", "jieba"),
                        stopwords_key=corpus_cfg.get("stopwords", "zh_weibo"),
                        lowercase=corpus_cfg.get("lowercase", False),
                        min_words=corpus_cfg.get("min_words", 5),
                        cleaner_opts={"strip_mentions": True, "strip_parens": True},
                    )
                    if tokens is None:
                        continue
                    lines.append(" ".join(tokens))
                if not lines:
                    continue
                if province not in province_writers:
                    prov_dir = corpora_dir / province
                    prov_dir.mkdir(parents=True, exist_ok=True)
                    province_writers[province] = RollingFile(str(prov_dir))
                province_writers[province].write("\n".join(lines))
            del df
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")

    for writer in province_writers.values():
        writer.close()


def main(config='config/config.yml', year=2024, group=None, province=None,
         start_month=None, end_month=None, step="all"):
    """
    Build province-level corpora from Weibo data.

    Args:
        config: Path to configuration file
        year: Data year (2020 or 2024)
        group: Province group index for parallel execution
        province: Single province to process
        start_month: Start month for 2024 prepare step
        end_month: End month for 2024 prepare step
        step: "prepare", "clean", or "all"
    """
    config_data = load_config(config)
    logger = setup_logging(Path(config_data['paths']['log_dir']), "build_corpora_weibo.log")

    logger.info("=" * 80)
    logger.info(f"Starting Weibo corpus building (year={year})")
    logger.info("=" * 80)

    prov_config = config_data.get("provincial", {})
    groups = prov_config.get("province_groups", PROVINCE_GROUPS)

    if year == 2020:
        build_2020(config_data, logger, year, province_filter=province)
    elif year == 2024:
        if step in ("prepare", "all") and start_month and end_month:
            prepare_2024(config_data, logger, year, start_month, end_month)
        if step in ("clean", "all"):
            if group is not None:
                provinces = groups[group]
            elif province:
                provinces = [province]
            else:
                provinces = [p for g in groups for p in g]
            clean_for_word2vec(config_data, logger, provinces)

    logger.info("=" * 80)
    logger.info("Weibo corpus building completed!")
    logger.info("=" * 80)


if __name__ == "__main__":
    fire.Fire(main)
