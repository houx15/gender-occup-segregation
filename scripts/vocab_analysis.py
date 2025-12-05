#!/usr/bin/env python3
"""
Vocabulary analysis script for word2vec models.

Function 1: Load vocabularies from all models, deduplicate using set,
            print total vocab size, and save the combined vocabulary.

Function 2: Use LLMs to analyze the vocabulary and select:
            - Occupation-related words
            - Prestige adjectives
            - Gender-related words (male and female)
            These words may help analyze the gender and prestige of occupations
            based on the word2vec models we trained.

Usage:
    python vocab_analysis.py --config=config/config.yml
    python vocab_analysis.py --config=config/config.yml --analyze_with_llm
"""

import logging
import sys
import json
from pathlib import Path
from typing import Set, List, Dict
import yaml
from gensim.models import KeyedVectors
import fire
import os

from openai import OpenAI


def setup_logging(log_dir: Path) -> logging.Logger:
    """Configure logging to both file and console."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "vocab_analysis.log"

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # File handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_all_vocabularies(models_dir: Path, logger: logging.Logger) -> Set[str]:
    """
    Load vocabularies from all models and return deduplicated set.

    Args:
        models_dir: Directory containing model files
        logger: Logger instance

    Returns:
        Set of all unique vocabulary words
    """
    logger.info("Loading vocabularies from all models...")

    # Find all model files
    model_files = sorted(models_dir.glob("*.model"))

    if not model_files:
        logger.warning(f"No model files found in {models_dir}")
        return set()

    logger.info(f"Found {len(model_files)} model files")

    all_vocab = set()

    for model_path in model_files:
        logger.info(f"  Loading vocabulary from {model_path.name}...")
        try:
            model = KeyedVectors.load(str(model_path))
            # Get vocabulary from model
            vocab = set(model.index_to_key)
            logger.info(f"    Loaded {len(vocab)} words (total unique: {len(all_vocab)})")
            all_vocab.update(vocab)
        except Exception as e:
            logger.error(f"    Error loading {model_path.name}: {e}")
            continue

    logger.info(f"\nTotal unique vocabulary size: {len(all_vocab)}")

    return all_vocab


def save_vocabulary(vocab: Set[str], output_dir: Path, logger: logging.Logger) -> None:
    """
    Save vocabulary to a text file (one word per line).

    Args:
        vocab: Set of vocabulary words
        output_path: Path to save the vocabulary file
        logger: Logger instance
    """
    logger.info(f"Saving vocabulary to {output_dir}...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sort vocabulary for consistent output
    sorted_vocab = sorted(vocab)

    possible_occupations = []
    others = []
    for word in sorted_vocab:
        if word.endswith("家") or word.endswith("师") or word.endswith("士") or word.endswith("员") or word.endswith("官") or word.endswith("长") or word.endswith("人") or word.endswith("工"):
            possible_occupations.append(word)
        else:
            others.append(word)

    with open(output_dir / "combined_vocabulary.txt", "w", encoding="utf-8") as f:
        for word in others:
            f.write(word + "\n")
        
    with open(output_dir / "possible_occupations.txt", "w", encoding="utf-8") as f:
        for word in possible_occupations:
            f.write(word + "\n")

    logger.info(f"  Saved {len(others)} words to {output_dir / 'combined_vocabulary.txt'}")
    logger.info(f"  Saved {len(possible_occupations)} words to {output_dir / 'possible_occupations.txt'}")



def analyze_vocab_with_llm(
    vocab: Set[str],
    logger: logging.Logger,
    api_key: str = None,
    model: str = "gpt-5-nano",
) -> Dict[str, List[str]]:

    # 初始化 OpenAI client
    client = OpenAI(api_key=api_key)

    system_prompt = """
你是一个严格的中文职业分类器。
你的任务：判断给定词条是否是1940年以来中国语境中的“职业 / 工作岗位 / 职务”。

你的输出必须是一个大写字母，不要解释：
A = 一定是职业
B = 一定不是职业
C = 不确定（身份/头衔/角色/群体/机构/行为，而不是明确职业）
"""

    results = {"A": [], "B": [], "C": []}

    for term in vocab:
        try:
            logger.info(f"Processing term: {term}")

            user_prompt = f"{system_prompt}\n\n词条：{term}\n请回答 A, B, 或 C："

            response = client.responses.create(
                model=model,
                # system=system_prompt,
                input=user_prompt,
                reasoning={"effort": "minimal"},
                # max_output_tokens=16,
            )

            reply = response.output_text.strip().upper()
            if reply not in ["A", "B", "C"]:
                logger.warning(f"Unexpected reply for '{term}': {reply}, fallback to C")
                reply = "C"

            results[reply].append(term)

        except Exception as e:
            logger.error(f"Error processing {term}: {e}")
            results["C"].append(term)

    return results

def save_llm_analysis_results(
    results: Dict[str, List[str]], output_dir: Path, logger: logging.Logger
) -> None:
    """
    Save LLM analysis results to JSON file.

    Args:
        results: Dictionary with analysis results
        output_path: Path to save the results
        logger: Logger instance
    """
    logger.info(f"Saving LLM analysis results to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"  Saved results to {output_path}")


def main(
    config="config/config.yml",
    analyze_with_llm=False,
    api_key=None,
    llm_model="gpt-5-nano",
    batch_size=500,
):
    """
    Analyze vocabularies from all trained models.

    Args:
        config: Path to configuration file
        analyze_with_llm: Whether to use LLM to analyze vocabulary
        api_key: OpenAI API key (if None, tries to get from environment)
        llm_model: Model name to use for LLM analysis
        batch_size: Number of words to analyze per API call
    """
    # Load configuration
    config_data = load_config(config)

    # Setup logging
    log_dir = Path(config_data["paths"]["log_dir"])
    logger = setup_logging(log_dir)

    logger.info("=" * 80)
    logger.info("Starting vocabulary analysis")
    logger.info("=" * 80)

    # Function 1: Load and save all vocabularies
    models_dir = Path(config_data["paths"]["models_dir"])
    all_vocab = load_all_vocabularies(models_dir, logger)

    if not all_vocab:
        logger.error("No vocabulary found. Please train models first.")
        return 1

    # Save combined vocabulary
    results_dir = Path(config_data["paths"]["results_dir"])
    save_vocabulary(all_vocab, results_dir, logger)

    logger.info("\n" + "=" * 80)
    logger.info("Vocabulary analysis completed!")
    logger.info("=" * 80)

    return 0


def llm(config="config/config.yml"):
    config_data = load_config(config)
    # Setup logging

    api_key = config_data["api_key"]
    log_dir = Path(config_data["paths"]["log_dir"])
    logger = setup_logging(log_dir)

    logger.info("=" * 80)
    logger.info("Starting vocabulary analysis")
    logger.info("=" * 80)

    with open("combined_vocabulary.txt", "r", encoding="utf-8") as f:
        combined_vocabulary = f.readlines()
    
    results = analyze_vocab_with_llm(combined_vocabulary, logger, api_key=api_key, model="gpt-5-nano")

    with open("must_occupations.txt", "w", encoding="utf-8") as f:
        for word in results["A"]:
            f.write(word + "\n")
    with open("may_occupations.txt", "w", encoding="utf-8") as f:
        for word in results["C"]:
            f.write(word + "\n")


if __name__ == "__main__":
    fire.Fire({"main": main, "llm": llm})
