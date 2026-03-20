#!/usr/bin/env python3
"""
Newspaper-to-province mapping utility.

Full pipeline for recovering newspaper→province mapping from raw data:
  1. extract:  Scan raw JSONL files → newspaper_stats.csv (names + article counts)
  2. map:      Apply auto-mapping rules → newspaper_province_mapping.json
  3. add:      Manually add/fix mappings for newspapers the rules missed

Usage:
    # Step 1: Extract newspaper names from raw JSONL data
    python -m scripts.data_prep.province_mapper extract --config=config/config.yml

    # Step 2: Auto-map newspapers to provinces
    python -m scripts.data_prep.province_mapper map --config=config/config.yml

    # Step 3: Add manual fixes (edit MANUAL_MAPPINGS in this file, then run)
    python -m scripts.data_prep.province_mapper add --config=config/config.yml
"""

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict

import fire

from scripts.common.config_loader import load_config
from scripts.common.logging_utils import setup_logging


PROVINCES = {
    "北京", "天津", "河北", "山西", "内蒙古",
    "辽宁", "吉林", "黑龙江",
    "上海", "江苏", "浙江", "安徽", "福建", "江西", "山东",
    "河南", "湖北", "湖南", "广东", "广西", "海南",
    "重庆", "四川", "贵州", "云南", "西藏",
    "陕西", "甘肃", "青海", "宁夏", "新疆",
}

NATIONAL_NEWSPAPERS = {
    "人民日报", "人民日报海外版", "光明日报", "经济日报", "解放军报",
    "工人日报", "农民日报", "科技日报", "法制日报", "中国青年报",
    "中国证券报", "上海证券报", "证券时报", "证券日报",
    "金融时报", "21世纪经济报道", "第一财经日报", "经济参考报",
    "新华每日电讯", "南方周末",
}

CITY_TO_PROVINCE = {
    "北京": "北京", "上海": "上海", "天津": "天津", "重庆": "重庆",
    "广州": "广东", "深圳": "广东", "珠海": "广东", "佛山": "广东",
    "南京": "江苏", "苏州": "江苏", "无锡": "江苏", "常州": "江苏",
    "杭州": "浙江", "宁波": "浙江", "温州": "浙江",
    "济南": "山东", "青岛": "山东", "烟台": "山东",
    "成都": "四川", "绵阳": "四川",
    "武汉": "湖北", "宜昌": "湖北",
    "长沙": "湖南", "株洲": "湖南",
    "郑州": "河南", "洛阳": "河南",
    "西安": "陕西", "宝鸡": "陕西",
    "沈阳": "辽宁", "大连": "辽宁",
    "哈尔滨": "黑龙江", "齐齐哈尔": "黑龙江",
    "长春": "吉林",
    "石家庄": "河北", "唐山": "河北",
    "太原": "山西", "大同": "山西",
    "呼和浩特": "内蒙古", "包头": "内蒙古",
    "南宁": "广西", "柳州": "广西", "桂林": "广西",
    "昆明": "云南", "大理": "云南",
    "贵阳": "贵州", "遵义": "贵州",
    "兰州": "甘肃", "天水": "甘肃",
    "西宁": "青海",
    "银川": "宁夏",
    "乌鲁木齐": "新疆",
    "拉萨": "西藏",
    "海口": "海南", "三亚": "海南",
    "厦门": "福建", "福州": "福建", "泉州": "福建",
    "南昌": "江西", "九江": "江西",
    "合肥": "安徽", "芜湖": "安徽",
}

SPECIAL_CASES = {
    "南方日报": "广东",
    "文汇报": "上海",
    "新华日报": "江苏",
    "内蒙古日报（汉）": "内蒙古",
    "人民日报(海外版)": "全国",
    "人民日报（海外版）": "全国",
    "光明日报?": "全国",
    "甘孜日报": "四川",
    "乌鲁木齐晚报（汉）": "新疆",
    "阿勒泰日报（汉）": "新疆",
    "日喀则报（汉）": "西藏",
    "格尔木报": "青海",
}

# Add manual mappings here for newspapers that auto-rules can't resolve.
# Run `python -m scripts.data_prep.province_mapper add` to apply them.
MANUAL_MAPPINGS = {
    "建筑报": "行业",
    "广西政法报": "广西",
    "河北科技报(农村版)": "河北",
    "中药报": "行业",
    "中国轻工报": "全国",
    "医药导报(中药报)": "行业",
    "北京电子报": "北京",
    "经济日报农村版": "全国",
    "卫生与生活": "行业",
    "中药事业报": "行业",
    "中国国土资源报(地矿版)": "全国",
    "法治快报(广西政法报)": "广西",
    "广西法制报": "广西",
    "大众科技报.科学奥秘周刊": "全国",
    "经济参政报": "全国",
    "检察日报明镜周刊": "全国",
    "海南特区科技报": "海南",
    "-中国机电日报": "全国",
}


# =============================================================================
# Core mapping logic
# =============================================================================

def map_newspaper_to_province(name: str) -> str:
    """Map a newspaper name to its province. Returns province, '全国', '行业', or '未知'."""
    if name in NATIONAL_NEWSPAPERS:
        return "全国"
    if name in SPECIAL_CASES:
        return SPECIAL_CASES[name]
    if name in MANUAL_MAPPINGS:
        return MANUAL_MAPPINGS[name]

    # Provincial dailies
    for province in PROVINCES:
        if name == f"{province}日报":
            return province

    # City dailies / newspapers containing city name
    for city, province in CITY_TO_PROVINCE.items():
        if city in name:
            return province

    # Province name in newspaper name
    for province in PROVINCES:
        if province in name:
            return province

    if any(kw in name for kw in ["报", "刊", "通讯"]):
        return "行业"

    return "未知"


def _get_output_dir(config):
    """Get output directory for mapping files."""
    base_dir = Path(config["paths"]["base_dir"])
    return base_dir / "provincial" / "data"


def _get_raw_data_dir(config):
    """Get raw newspaper data directory."""
    return config["paths"]["raw_data_dir"]


# =============================================================================
# Step 1: Extract newspaper names from raw JSONL
# =============================================================================

def extract(config="config/config.yml", sample_size=None):
    """
    Scan raw newspaper JSONL files and extract newspaper names + article counts.

    Outputs:
      - newspaper_names.txt (one name per line)
      - newspaper_stats.csv (name, article count)

    Args:
        config: Path to configuration file
        sample_size: Only process first N lines per file (for testing)
    """
    config_data = load_config(config)
    logger = setup_logging(Path(config_data["paths"]["log_dir"]), "province_mapper.log")

    data_dir = _get_raw_data_dir(config_data)
    output_dir = _get_output_dir(config_data)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Extracting newspaper names from raw data")
    logger.info("=" * 60)

    all_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.json')])
    logger.info(f"Found {len(all_files)} JSONL files")

    newspaper_counts: Dict[str, int] = defaultdict(int)
    total_articles = 0
    errors = 0

    for filename in all_files:
        filepath = os.path.join(data_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                line_count = 0
                for line in f:
                    if sample_size and line_count >= sample_size:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        article = json.loads(line)
                        source = article.get('source', '').strip()
                        if source:
                            newspaper_counts[source] += 1
                            total_articles += 1
                        line_count += 1
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        errors += 1
        except Exception as e:
            logger.error(f"Failed to read {filename}: {e}")
            errors += 1

    sorted_newspapers = sorted(newspaper_counts.items(), key=lambda x: x[1], reverse=True)

    # Save newspaper_names.txt
    names_file = output_dir / "newspaper_names.txt"
    with open(names_file, 'w', encoding='utf-8') as f:
        for name, _ in sorted_newspapers:
            f.write(f"{name}\n")

    # Save newspaper_stats.csv
    stats_file = output_dir / "newspaper_stats.csv"
    with open(stats_file, 'w', encoding='utf-8-sig') as f:
        f.write("报纸名称,文章数量\n")
        for name, count in sorted_newspapers:
            escaped = f'"{name}"' if ',' in name or '"' in name else name
            f.write(f"{escaped},{count}\n")

    logger.info(f"Extracted {len(sorted_newspapers)} newspapers, {total_articles:,} articles, {errors} errors")
    logger.info(f"Saved: {names_file}")
    logger.info(f"Saved: {stats_file}")

    # Preview top 20
    for i, (name, count) in enumerate(sorted_newspapers[:20], 1):
        logger.info(f"  {i:3d}. {name:<30s} {count:>10,}")


# =============================================================================
# Step 2: Auto-map newspapers to provinces
# =============================================================================

def map(config="config/config.yml"):
    """
    Auto-map newspaper names to provinces using rule-based matching.

    Reads newspaper_stats.csv (from extract step) and produces
    newspaper_province_mapping.json.

    Args:
        config: Path to configuration file
    """
    config_data = load_config(config)
    logger = setup_logging(Path(config_data["paths"]["log_dir"]), "province_mapper.log")

    output_dir = _get_output_dir(config_data)
    stats_file = output_dir / "newspaper_stats.csv"

    if not stats_file.exists():
        logger.error(f"Stats file not found: {stats_file}")
        logger.info("Run 'extract' first to generate newspaper_stats.csv")
        return

    import pandas as pd
    stats_df = pd.read_csv(stats_file, encoding='utf-8-sig')

    mapping = {}
    for _, row in stats_df.iterrows():
        newspaper = row['报纸名称']
        article_count = row['文章数量']
        province = map_newspaper_to_province(newspaper)
        mapping[newspaper] = {
            "province": province,
            "article_count": int(article_count),
        }

    output_file = output_dir / "newspaper_province_mapping.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    # Summary
    counts = {"province": 0, "national": 0, "industry": 0, "unknown": 0}
    for info in mapping.values():
        p = info["province"]
        if p in PROVINCES:
            counts["province"] += 1
        elif p == "全国":
            counts["national"] += 1
        elif p == "行业":
            counts["industry"] += 1
        else:
            counts["unknown"] += 1

    logger.info(f"Mapping: {len(mapping)} newspapers -> "
                f"Provincial: {counts['province']}, National: {counts['national']}, "
                f"Industry: {counts['industry']}, Unknown: {counts['unknown']}")
    logger.info(f"Saved: {output_file}")

    if counts["unknown"] > 0:
        unknown = [n for n, info in mapping.items() if info["province"] == "未知"]
        logger.info(f"Unknown newspapers ({len(unknown)}):")
        for name in unknown[:20]:
            logger.info(f"  - {name}")
        if len(unknown) > 20:
            logger.info(f"  ... and {len(unknown) - 20} more")
        logger.info("Add them to MANUAL_MAPPINGS in province_mapper.py, then run 'add'")


# =============================================================================
# Step 3: Add manual mapping fixes
# =============================================================================

def add(config="config/config.yml"):
    """
    Apply MANUAL_MAPPINGS to an existing newspaper_province_mapping.json.

    Edit the MANUAL_MAPPINGS dict in this file for newspapers that auto-rules
    can't resolve, then run this command to merge them in.

    Args:
        config: Path to configuration file
    """
    config_data = load_config(config)
    logger = setup_logging(Path(config_data["paths"]["log_dir"]), "province_mapper.log")

    output_dir = _get_output_dir(config_data)
    mapping_file = output_dir / "newspaper_province_mapping.json"

    if not mapping_file.exists():
        logger.error(f"Mapping file not found: {mapping_file}")
        logger.info("Run 'map' first to generate the initial mapping")
        return

    with open(mapping_file, 'r', encoding='utf-8') as f:
        mapping = json.load(f)

    added = 0
    updated = 0
    for newspaper, province in MANUAL_MAPPINGS.items():
        if newspaper not in mapping:
            mapping[newspaper] = {"province": province, "article_count": 0}
            logger.info(f"  Added: {newspaper} -> {province}")
            added += 1
        elif mapping[newspaper]["province"] != province:
            old = mapping[newspaper]["province"]
            mapping[newspaper]["province"] = province
            logger.info(f"  Updated: {newspaper}: {old} -> {province}")
            updated += 1

    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    logger.info(f"Done: {added} added, {updated} updated")


# =============================================================================
# Preview utility
# =============================================================================

def preview(config="config/config.yml", filename=None, n_lines=5):
    """
    Preview a raw newspaper JSONL file.

    Args:
        config: Path to configuration file
        filename: Specific file to preview (default: first file)
        n_lines: Number of articles to show
    """
    config_data = load_config(config)
    data_dir = _get_raw_data_dir(config_data)

    if filename is None:
        all_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.json')])
        if not all_files:
            print("No JSONL files found")
            return
        filename = all_files[0]

    filepath = os.path.join(data_dir, filename)
    print(f"\nPreview: {filename}\n")

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f):
            if i >= n_lines:
                break
            try:
                article = json.loads(line.strip())
                print(f"--- Article {i + 1} ---")
                print(f"  source:  {article.get('source', 'N/A')}")
                print(f"  title:   {article.get('title', 'N/A')}")
                print(f"  length:  {len(article.get('pdf_txt', ''))} chars")
                print()
            except json.JSONDecodeError as e:
                print(f"  JSON error: {e}\n")


if __name__ == "__main__":
    fire.Fire({
        "extract": extract,
        "map": map,
        "add": add,
        "preview": preview,
    })
