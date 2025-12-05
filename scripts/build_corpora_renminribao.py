#!/usr/bin/env python3
"""
构建人民日报时间切片语料库

从人民日报文本文件中提取、清洗、分词，并按时间切片组织输出。

数据路径格式：
    {decade}/{year}/报刊/人民日报/rmrb_{year}_{month}.txt
    例如：1940s/1946/报刊/人民日报/rmrb_1946_05.txt

输出格式：
    {corpora_dir}/{start_year}_{end_year}/corpus_*.txt
    每行是一个清洗、分词后的文本（空格分隔的词）

Usage:
    python build_corpora_renminribao.py --config=config/renminribao.yml
    python build_corpora_renminribao.py --config=config/renminribao.yml --slice=1940_1949
    python build_corpora_renminribao.py --config=config/renminribao.yml --overwrite
"""

import os
import logging
import sys
from pathlib import Path
from typing import List, Tuple, Set
from collections import defaultdict
import yaml
import fire
import re
import jieba

# 停用词集合（可以根据需要扩展）
STOPWORDS = {
    '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', 
    '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', 
    '自己', '这', '那', '他', '她', '它', '们', '为', '而', '以', '与', '及', '或',
    '等', '等', '等', '等', '等', '等', '等', '等', '等', '等', '等', '等', '等',
    ' ', '\t', '\n', '\r'
}

CHINESE_RE = re.compile(r"[\u4e00-\u9fff]+")


def setup_logging(log_dir: Path) -> logging.Logger:
    """配置日志记录到文件和控制台"""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "build_corpora_renminribao.log"

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def load_config(config_path: str) -> dict:
    """从YAML文件加载配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def generate_time_slices(start_year: int, end_year: int, window_size: int, step_size: int) -> List[Tuple[int, int]]:
    """根据配置生成时间切片窗口"""
    slices = []
    current_start = start_year

    while current_start <= end_year:
        current_end = min(current_start + window_size - 1, end_year)
        slices.append((current_start, current_end))
        current_start += step_size

        if current_start > end_year:
            break

    return slices


def clean_renminribao_text(text: str) -> str:
    """
    清洗人民日报文本
    
    移除：
    - 特殊符号和标点
    - URL链接
    - 多余空白
    - 非中文字符（保留中文）
    """
    if not text or not isinstance(text, str):
        return ""
    
    # 移除URL
    text = re.sub(r"http[s]?://\S+", "", text)
    text = re.sub(r"www\.\S+", "", text)
    
    # 移除方括号内容（如[图片]、[链接]等）
    text = re.sub(r"\[.*?\]", "", text)
    
    # 移除特殊符号
    text = text.replace("\u200b", "")  # 零宽空格
    text = text.replace("…", "")
    
    # 只保留中文字符
    chinese_text = "".join(CHINESE_RE.findall(text))
    
    return chinese_text.strip()


def segment_text(text: str, min_words: int = 5) -> str:
    """
    对文本进行分词
    
    Args:
        text: 清洗后的中文文本
        min_words: 最小词数，少于该数量的文本返回空字符串
    
    Returns:
        空格分隔的词序列，如果词数不足则返回空字符串
    """
    if not text or len(text.strip()) == 0:
        return ""
    
    # 使用jieba分词
    words = jieba.lcut(text, HMM=True)
    
    # 过滤停用词和空白
    filtered_words = [w.strip() for w in words if w.strip() and w.strip() not in STOPWORDS]
    
    # 如果词数不足，返回空字符串
    if len(filtered_words) < min_words:
        return ""
    
    return " ".join(filtered_words)


def extract_year_from_filename(filename: str) -> int:
    """
    从文件名中提取年份
    
    文件名格式：rmrb_{year}_{month}.txt
    例如：rmrb_1946_05.txt -> 1946
    """
    match = re.search(r'rmrb_(\d{4})_\d{2}\.txt', filename)
    if match:
        return int(match.group(1))
    return None


def find_renminribao_files(data_dir: Path) -> List[Tuple[Path, int]]:
    """
    查找所有人民日报文件并提取年份
    
    Returns:
        List of (file_path, year) tuples
    """
    files = []
    
    # 遍历目录结构：{decade}/{year}/报刊/人民日报/rmrb_{year}_{month}.txt
    for decade_dir in sorted(data_dir.glob("*s")):  # 1940s, 1950s, etc.
        if not decade_dir.is_dir():
            continue
        
        for year_dir in sorted(decade_dir.glob("*")):
            if not year_dir.is_dir():
                continue
            
            rmr_dir = year_dir / "报刊" / "人民日报"
            if not rmr_dir.exists():
                continue
            
            for txt_file in sorted(rmr_dir.glob("rmrb_*.txt")):
                year = extract_year_from_filename(txt_file.name)
                if year:
                    files.append((txt_file, year))
    
    return files


def process_file(
    file_path: Path,
    year: int,
    time_slices: List[Tuple[int, int]],
    corpora_dir: Path,
    logger: logging.Logger,
    min_words: int = 5,
    buffer_size: int = 10000
) -> dict:
    """
    处理单个人民日报文件
    
    Args:
        file_path: 文件路径
        year: 文件年份
        time_slices: 时间切片列表
        corpora_dir: 语料库输出目录
        logger: 日志记录器
        min_words: 最小词数
        buffer_size: 缓冲区大小（行数）
    
    Returns:
        统计信息字典
    """
    stats = defaultdict(int)
    
    # 确定该文件属于哪些时间切片
    matched_slices = []
    for start_year, end_year in time_slices:
        if start_year <= year <= end_year:
            slice_name = f"{start_year}_{end_year}"
            matched_slices.append(slice_name)
    
    if not matched_slices:
        logger.debug(f"文件 {file_path.name} (年份: {year}) 不在任何时间切片范围内")
        return stats
    
    # 生成唯一的文件名（基于原文件名，在函数开始时计算一次）
    # rmrb_1946_05.txt -> 194605
    file_index = file_path.stem.replace("rmrb_", "").replace("_", "")
    
    # 为每个匹配的时间切片创建缓冲区和输出文件路径
    buffers = {slice_name: [] for slice_name in matched_slices}
    corpus_files = {}
    for slice_name in matched_slices:
        slice_dir = corpora_dir / slice_name
        slice_dir.mkdir(parents=True, exist_ok=True)
        corpus_files[slice_name] = slice_dir / f"corpus_{file_index}.txt"
    
    try:
        lines_processed = 0
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(f, 1):
                lines_processed += 1
                
                # 清洗文本
                cleaned = clean_renminribao_text(line)
                if not cleaned:
                    continue
                
                # 分词
                segmented = segment_text(cleaned, min_words=min_words)
                if not segmented:
                    continue
                
                # 添加到所有匹配的时间切片缓冲区
                for slice_name in matched_slices:
                    buffers[slice_name].append(segmented)
                    stats[f"{slice_name}_lines"] += 1
                
                # 如果缓冲区满了，写入文件
                if len(buffers[matched_slices[0]]) >= buffer_size:
                    for slice_name in matched_slices:
                        with open(corpus_files[slice_name], 'a', encoding='utf-8') as out_f:
                            out_f.write("\n".join(buffers[slice_name]) + "\n")
                        buffers[slice_name] = []
        
        # 写入剩余的缓冲区内容
        for slice_name in matched_slices:
            if buffers[slice_name]:
                with open(corpus_files[slice_name], 'a', encoding='utf-8') as out_f:
                    out_f.write("\n".join(buffers[slice_name]) + "\n")
        
        stats['files_processed'] = 1
        stats['lines_read'] = lines_processed
        
    except Exception as e:
        logger.error(f"处理文件 {file_path} 时出错: {e}")
        stats['errors'] = 1
    
    return stats


def build_corpora(
    config_data: dict,
    logger: logging.Logger,
    specific_slice: str = None,
    overwrite: bool = False,
    data_dir: str = None
) -> None:
    """
    构建所有时间切片语料库
    
    Args:
        config_data: 配置数据
        logger: 日志记录器
        specific_slice: 只处理特定切片（格式: 1940_1949）
        overwrite: 是否覆盖已存在的语料库
        data_dir: 人民日报数据目录（如果未在配置中指定）
    """
    # 生成时间切片
    time_slices_config = config_data['time_slices']
    time_slices = generate_time_slices(
        time_slices_config['start_year'],
        time_slices_config['end_year'],
        time_slices_config['window_size'],
        time_slices_config['step_size']
    )
    
    logger.info(f"生成 {len(time_slices)} 个时间切片:")
    for start, end in time_slices:
        logger.info(f"  {start}-{end}")
    
    if specific_slice:
        start, end = map(int, specific_slice.split('_'))
        time_slices = [(start, end)]
        logger.info(f"\n只构建切片: {start}-{end}")
    
    # 确定数据目录
    if data_dir:
        data_path = Path(data_dir)
    elif 'paths' in config_data and 'data_dir' in config_data['paths']:
        data_path = Path(config_data['paths']['data_dir'])
    else:
        # 默认路径（根据注释中的路径）
        data_path = Path("/scratch/network/yh6580/chinese_corpus/xiandai_20250422/xiandai")
        logger.warning(f"未在配置中找到data_dir，使用默认路径: {data_path}")
    
    if not data_path.exists():
        logger.error(f"数据目录不存在: {data_path}")
        return
    
    # 语料库输出目录
    corpora_dir = Path(config_data['paths']['corpora_dir'])
    corpora_dir.mkdir(parents=True, exist_ok=True)
    
    # 如果overwrite，清理已存在的语料库
    if overwrite and specific_slice:
        slice_dir = corpora_dir / specific_slice
        if slice_dir.exists():
            logger.info(f"删除已存在的切片目录: {slice_dir}")
            import shutil
            shutil.rmtree(slice_dir)
    elif overwrite:
        logger.warning("overwrite=True 但未指定specific_slice，跳过清理")
    
    # 查找所有人民日报文件
    logger.info(f"\n在 {data_path} 中查找人民日报文件...")
    files = find_renminribao_files(data_path)
    logger.info(f"找到 {len(files)} 个文件")
    
    if not files:
        logger.error("未找到任何人民日报文件")
        return
    
    # 统计信息
    total_stats = defaultdict(int)
    
    # 处理每个文件
    for file_idx, (file_path, year) in enumerate(files, 1):
        logger.info(f"\n处理文件 [{file_idx}/{len(files)}]: {file_path.name} (年份: {year})")
        
        stats = process_file(
            file_path,
            year,
            time_slices,
            corpora_dir,
            logger,
            min_words=5
        )
        
        # 更新总统计
        for key, value in stats.items():
            total_stats[key] += value
        
        if file_idx % 100 == 0:
            logger.info(f"已处理 {file_idx}/{len(files)} 个文件")
    
    # 输出统计信息
    logger.info("\n" + "="*80)
    logger.info("处理完成！统计信息:")
    logger.info("="*80)
    logger.info(f"总文件数: {total_stats['files_processed']}")
    logger.info(f"错误数: {total_stats['errors']}")
    
    for slice_name in sorted(set([f"{s}_{e}" for s, e in time_slices])):
        lines_key = f"{slice_name}_lines"
        if lines_key in total_stats:
            logger.info(f"  {slice_name}: {total_stats[lines_key]:,} 行")


def main(
    config: str = 'config/renminribao.yml',
    slice: str = None,
    overwrite: bool = False,
    data_dir: str = None
):
    """
    构建人民日报时间切片语料库
    
    Args:
        config: 配置文件路径
        slice: 只构建特定切片（格式: 1940_1949）
        overwrite: 是否覆盖已存在的语料库
        data_dir: 人民日报数据目录（可选，如果未在配置中指定）
    """
    config_data = load_config(config)
    log_dir = Path(config_data['paths']['log_dir'])
    logger = setup_logging(log_dir)
    
    logger.info("="*80)
    logger.info("开始构建人民日报语料库")
    logger.info("="*80)
    
    build_corpora(
        config_data,
        logger,
        specific_slice=slice,
        overwrite=overwrite,
        data_dir=data_dir
    )
    
    logger.info("\n" + "="*80)
    logger.info("语料库构建完成！")
    logger.info("="*80)
    
    return 0


if __name__ == "__main__":
    fire.Fire(main)
