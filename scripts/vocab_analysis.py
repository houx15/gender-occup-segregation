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
from gensim.models import Word2Vec
import fire
import os

# Try to import OpenAI for LLM analysis
try:
    import openai

    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


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
            model = Word2Vec.load(str(model_path))
            # Get vocabulary from model
            vocab = set(model.wv.key_to_index.keys())
            all_vocab.update(vocab)
            logger.info(
                f"    Loaded {len(vocab)} words (total unique: {len(all_vocab)})"
            )
        except Exception as e:
            logger.error(f"    Error loading {model_path.name}: {e}")
            continue

    logger.info(f"\nTotal unique vocabulary size: {len(all_vocab)}")

    return all_vocab


def save_vocabulary(vocab: Set[str], output_path: Path, logger: logging.Logger) -> None:
    """
    Save vocabulary to a text file (one word per line).

    Args:
        vocab: Set of vocabulary words
        output_path: Path to save the vocabulary file
        logger: Logger instance
    """
    logger.info(f"Saving vocabulary to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Sort vocabulary for consistent output
    sorted_vocab = sorted(vocab)

    with open(output_path, "w", encoding="utf-8") as f:
        for word in sorted_vocab:
            f.write(word + "\n")

    logger.info(f"  Saved {len(sorted_vocab)} words to {output_path}")


def analyze_vocab_with_llm(
    vocab: Set[str],
    logger: logging.Logger,
    api_key: str = None,
    model: str = "gpt-4o-mini",
    batch_size: int = 500,
) -> Dict[str, List[str]]:
    """
    Use LLM to analyze vocabulary and categorize words.

    Args:
        vocab: Set of vocabulary words to analyze
        logger: Logger instance
        api_key: OpenAI API key (if None, tries to get from environment)
        model: Model name to use
        batch_size: Number of words to analyze per API call

    Returns:
        Dictionary with keys: 'occupations', 'prestige_adjectives',
        'male_words', 'female_words'
    """
    if not HAS_OPENAI:
        logger.error("OpenAI library not installed. Install with: pip install openai")
        return {}

    # Get API key
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable or pass --api_key"
            )
            return {}

    logger.info("Analyzing vocabulary with LLM...")
    logger.info(f"  Using model: {model}")
    logger.info(f"  Total words to analyze: {len(vocab)}")

    # Initialize OpenAI client
    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    # Convert set to sorted list for consistent processing
    vocab_list = sorted(list(vocab))

    # Process in batches
    results = {
        "occupations": [],
        "prestige_adjectives": [],
        "male_words": [],
        "female_words": [],
    }

    total_batches = (len(vocab_list) + batch_size - 1) // batch_size

    for i in range(0, len(vocab_list), batch_size):
        batch = vocab_list[i : i + batch_size]
        batch_num = i // batch_size + 1

        logger.info(
            f"  Processing batch {batch_num}/{total_batches} ({len(batch)} words)..."
        )

        try:
            # Create prompt
            prompt = f"""请分析以下中文词汇列表，并将它们分类到以下四个类别中：

1. 职业相关词 (occupations): 与职业、工作、职位相关的词汇
2. 声望形容词 (prestige_adjectives): 描述声望、地位、品质的形容词
3. 男性相关词 (male_words): 与男性、男性特征相关的词汇
4. 女性相关词 (female_words): 与女性、女性特征相关的词汇

词汇列表：
{', '.join(batch)}

请以JSON格式返回结果，格式如下：
{{
  "occupations": ["词1", "词2", ...],
  "prestige_adjectives": ["词1", "词2", ...],
  "male_words": ["词1", "词2", ...],
  "female_words": ["词1", "词2", ...]
}}

只返回JSON，不要其他文字说明。如果一个词可能属于多个类别，请选择最合适的类别。如果某个类别没有匹配的词，返回空数组。"""

            # Call OpenAI API
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "你是一个专业的中文词汇分析助手。请严格按照JSON格式返回结果。",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
            )

            # Parse response
            response_text = response.choices[0].message.content.strip()

            # Try to extract JSON from response
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()

            batch_results = json.loads(response_text)

            # Merge results
            for category in results:
                if category in batch_results:
                    results[category].extend(batch_results[category])

            logger.info(
                f"    Found: {len(batch_results.get('occupations', []))} occupations, "
                f"{len(batch_results.get('prestige_adjectives', []))} prestige adjectives, "
                f"{len(batch_results.get('male_words', []))} male words, "
                f"{len(batch_results.get('female_words', []))} female words"
            )

        except json.JSONDecodeError as e:
            logger.warning(f"    Failed to parse JSON response: {e}")
            logger.warning(f"    Response: {response_text[:200]}...")
            continue
        except Exception as e:
            logger.error(f"    Error processing batch {batch_num}: {e}")
            continue

    # Deduplicate results
    for category in results:
        results[category] = sorted(list(set(results[category])))

    logger.info("\nAnalysis complete!")
    logger.info(f"  Total occupations found: {len(results['occupations'])}")
    logger.info(
        f"  Total prestige adjectives found: {len(results['prestige_adjectives'])}"
    )
    logger.info(f"  Total male words found: {len(results['male_words'])}")
    logger.info(f"  Total female words found: {len(results['female_words'])}")

    return results


def save_llm_analysis_results(
    results: Dict[str, List[str]], output_path: Path, logger: logging.Logger
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
    llm_model="gpt-4o-mini",
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
    vocab_output = results_dir / "combined_vocabulary.txt"
    save_vocabulary(all_vocab, vocab_output, logger)

    # Function 2: Analyze with LLM if requested
    if analyze_with_llm:
        logger.info("\n" + "=" * 80)
        logger.info("Starting LLM-based vocabulary analysis")
        logger.info("=" * 80)

        llm_results = analyze_vocab_with_llm(
            all_vocab, logger, api_key=api_key, model=llm_model, batch_size=batch_size
        )

        if llm_results:
            llm_output = results_dir / "llm_vocab_analysis.json"
            save_llm_analysis_results(llm_results, llm_output, logger)
        else:
            logger.warning("LLM analysis returned no results")
    else:
        logger.info("\nSkipping LLM analysis (use --analyze_with_llm to enable)")

    logger.info("\n" + "=" * 80)
    logger.info("Vocabulary analysis completed!")
    logger.info("=" * 80)

    return 0


if __name__ == "__main__":
    fire.Fire(main)
