#!/usr/bin/env python3
"""
Analyze embeddings to compute gender typing and prestige scores for occupations.

Usage:
    python analyze_embeddings.py --config=config/config.yml
    python analyze_embeddings.py --config=config/config.yml --slice=1940_1949
    python analyze_embeddings.py --config=config/config.yml --export_plots
"""

import logging
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import yaml
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
import seaborn as sns
import fire


def setup_logging(log_dir: Path) -> logging.Logger:
    """Configure logging to both file and console."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "analyze_embeddings.log"

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


def load_occupations(file_path: Path, logger: logging.Logger) -> List[str]:
    """Load occupation list from file."""
    logger.info(f"Loading occupations from {file_path}")
    occupations = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            occ = line.strip()
            if occ:
                occupations.append(occ)
    logger.info(f"  Loaded {len(occupations)} occupations")
    return occupations


def load_word_lists(file_path: Path, logger: logging.Logger) -> dict:
    """Load word lists from JSON file."""
    logger.info(f"Loading word lists from {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_occupation_vector(
    occupation: str,
    model: Word2Vec,
    strategy: str,
    logger: logging.Logger
) -> Tuple[Optional[np.ndarray], float]:
    """
    Get vector representation for a multi-character occupation.

    Args:
        occupation: Occupation string (may be multi-character)
        model: Word2Vec model
        strategy: Strategy to use ("whole_token", "average_chars", or "hybrid")
        logger: Logger instance

    Returns:
        Tuple of (vector, coverage)
        - vector: numpy array or None if insufficient coverage
        - coverage: fraction of characters found in vocabulary
    """
    vocab = model.wv

    # Strategy 1: Try whole token
    if strategy in ["whole_token", "hybrid"]:
        if occupation in vocab:
            return vocab[occupation], 1.0

    # Strategy 2: Average character vectors
    if strategy in ["average_chars", "hybrid"]:
        char_vectors = []
        for char in occupation:
            if char in vocab:
                char_vectors.append(vocab[char])

        if len(char_vectors) == 0:
            return None, 0.0

        coverage = len(char_vectors) / len(occupation)
        avg_vector = np.mean(char_vectors, axis=0)

        # Normalize to unit length
        norm = np.linalg.norm(avg_vector)
        if norm > 0:
            avg_vector = avg_vector / norm

        return avg_vector, coverage

    return None, 0.0


def construct_semantic_axis(
    positive_terms: List[str],
    negative_terms: List[str],
    model: Word2Vec,
    logger: logging.Logger
) -> Tuple[Optional[np.ndarray], int, int]:
    """
    Construct a semantic axis from positive and negative terms.

    Args:
        positive_terms: List of positive pole terms
        negative_terms: List of negative pole terms
        model: Word2Vec model
        logger: Logger instance

    Returns:
        Tuple of (axis_vector, num_positive_found, num_negative_found)
    """
    vocab = model.wv

    # Get vectors for positive terms
    positive_vectors = []
    for term in positive_terms:
        # Try whole term first
        if term in vocab:
            positive_vectors.append(vocab[term])
        else:
            # Try averaging characters
            char_vecs = [vocab[c] for c in term if c in vocab]
            if char_vecs:
                positive_vectors.append(np.mean(char_vecs, axis=0))

    # Get vectors for negative terms
    negative_vectors = []
    for term in negative_terms:
        if term in vocab:
            negative_vectors.append(vocab[term])
        else:
            char_vecs = [vocab[c] for c in term if c in vocab]
            if char_vecs:
                negative_vectors.append(np.mean(char_vecs, axis=0))

    if not positive_vectors or not negative_vectors:
        return None, len(positive_vectors), len(negative_vectors)

    # Compute centroids
    v_pos = np.mean(positive_vectors, axis=0)
    v_neg = np.mean(negative_vectors, axis=0)

    # Axis is the difference
    axis = v_pos - v_neg

    # Normalize to unit length
    norm = np.linalg.norm(axis)
    if norm > 0:
        axis = axis / norm

    return axis, len(positive_vectors), len(negative_vectors)


def analyze_model(
    model_path: Path,
    slice_name: str,
    start_year: int,
    end_year: int,
    occupations: List[str],
    gender_words: dict,
    prestige_axes: dict,
    config: dict,
    logger: logging.Logger
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analyze a single model to compute gender and prestige scores.

    Args:
        model_path: Path to the model file
        slice_name: Name of the time slice
        start_year: Start year
        end_year: End year
        occupations: List of occupations to analyze
        gender_words: Gender word lists
        prestige_axes: Prestige axis definitions
        config: Configuration dictionary
        logger: Logger instance

    Returns:
        Tuple of (gender_df, prestige_df)
    """
    logger.info(f"\nAnalyzing {slice_name}...")

    # Load model
    logger.info(f"  Loading model from {model_path}")
    model = Word2Vec.load(str(model_path))

    # Get configuration
    strategy = config['analysis']['occupation_strategy']
    min_coverage = config['analysis']['min_coverage']

    # Construct gender axis
    logger.info(f"  Constructing gender axis...")
    gender_axis, num_male, num_female = construct_semantic_axis(
        gender_words['male_terms'],
        gender_words['female_terms'],
        model,
        logger
    )

    if gender_axis is None:
        logger.warning(f"  Could not construct gender axis for {slice_name}")
        logger.warning(f"    Male terms found: {num_male}, Female terms found: {num_female}")
    else:
        logger.info(f"    Male terms found: {num_male}/{len(gender_words['male_terms'])}")
        logger.info(f"    Female terms found: {num_female}/{len(gender_words['female_terms'])}")

    # Construct prestige axes
    logger.info(f"  Constructing prestige axes...")
    prestige_axis_vectors = {}

    for dimension, terms in prestige_axes.items():
        axis, num_pos, num_neg = construct_semantic_axis(
            terms['positive'],
            terms['negative'],
            model,
            logger
        )

        if axis is None:
            logger.warning(f"    Could not construct {dimension} axis")
            logger.warning(f"      Positive: {num_pos}, Negative: {num_neg}")
        else:
            prestige_axis_vectors[dimension] = axis
            logger.info(f"    {dimension}: {num_pos} positive, {num_neg} negative terms found")

    # Analyze occupations
    logger.info(f"  Analyzing {len(occupations)} occupations...")

    gender_results = []
    prestige_results = []

    for occupation in occupations:
        # Get occupation vector
        occ_vector, coverage = get_occupation_vector(occupation, model, strategy, logger)

        if occ_vector is None or coverage < min_coverage:
            continue

        # Compute gender score
        if gender_axis is not None:
            gender_score = float(np.dot(occ_vector, gender_axis))
        else:
            gender_score = np.nan

        gender_results.append({
            'occupation': occupation,
            'time_slice': slice_name,
            'start_year': start_year,
            'end_year': end_year,
            'gender_score': gender_score,
            'coverage': coverage
        })

        # Compute prestige scores
        prestige_scores = {
            'occupation': occupation,
            'time_slice': slice_name,
            'start_year': start_year,
            'end_year': end_year,
            'coverage': coverage
        }

        for dimension, axis in prestige_axis_vectors.items():
            score = float(np.dot(occ_vector, axis))
            prestige_scores[f'prestige_{dimension}'] = score

        prestige_results.append(prestige_scores)

    logger.info(f"  Analyzed {len(gender_results)} occupations with sufficient coverage")

    # Convert to DataFrames
    gender_df = pd.DataFrame(gender_results)
    prestige_df = pd.DataFrame(prestige_results)

    return gender_df, prestige_df


def compute_summary_statistics(
    gender_df: pd.DataFrame,
    prestige_df: pd.DataFrame,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Compute summary statistics and correlations.

    Args:
        gender_df: Gender typing DataFrame
        prestige_df: Prestige DataFrame
        logger: Logger instance

    Returns:
        Summary statistics DataFrame
    """
    logger.info("\nComputing summary statistics...")

    # Merge gender and prestige data
    merged_df = pd.merge(
        gender_df,
        prestige_df,
        on=['occupation', 'time_slice', 'start_year', 'end_year'],
        suffixes=('_gender', '_prestige')
    )

    summary_results = []

    # Group by time slice
    for slice_name in merged_df['time_slice'].unique():
        slice_data = merged_df[merged_df['time_slice'] == slice_name]

        summary = {
            'time_slice': slice_name,
            'start_year': slice_data['start_year'].iloc[0],
            'end_year': slice_data['end_year'].iloc[0],
            'num_occupations': len(slice_data),
            'mean_gender_score': slice_data['gender_score'].mean(),
            'std_gender_score': slice_data['gender_score'].std()
        }

        # Compute correlations between gender and each prestige dimension
        prestige_cols = [col for col in slice_data.columns if col.startswith('prestige_')]

        for col in prestige_cols:
            dimension = col.replace('prestige_', '')
            if slice_data['gender_score'].notna().sum() > 1 and slice_data[col].notna().sum() > 1:
                corr = slice_data['gender_score'].corr(slice_data[col])
                summary[f'corr_gender_{dimension}'] = corr
            else:
                summary[f'corr_gender_{dimension}'] = np.nan

        summary_results.append(summary)

    summary_df = pd.DataFrame(summary_results)

    return summary_df


def create_plots(
    gender_df: pd.DataFrame,
    prestige_df: pd.DataFrame,
    output_dir: Path,
    logger: logging.Logger
) -> None:
    """
    Create diagnostic visualizations.

    Args:
        gender_df: Gender typing DataFrame
        prestige_df: Prestige DataFrame
        output_dir: Directory to save plots
        logger: Logger instance
    """
    logger.info("\nCreating diagnostic plots...")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    sns.set_style("whitegrid")

    # Plot 1: Mean gender score over time
    if not gender_df.empty and 'gender_score' in gender_df.columns:
        time_series = gender_df.groupby('start_year')['gender_score'].mean().reset_index()

        plt.figure(figsize=(10, 6))
        plt.plot(time_series['start_year'], time_series['gender_score'], marker='o', linewidth=2)
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Mean Gender Score', fontsize=12)
        plt.title('Mean Gender Typing of Occupations Over Time', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'gender_score_over_time.png', dpi=300)
        plt.close()
        logger.info(f"  Saved: gender_score_over_time.png")

    # Plot 2: Gender vs General Prestige scatter (selected decades)
    merged_df = pd.merge(
        gender_df,
        prestige_df,
        on=['occupation', 'time_slice', 'start_year', 'end_year'],
        suffixes=('_gender', '_prestige')
    )

    if 'prestige_general_prestige' in merged_df.columns:
        # Select a few representative decades
        years_to_plot = sorted(merged_df['start_year'].unique())
        if len(years_to_plot) > 4:
            years_to_plot = years_to_plot[::len(years_to_plot)//4][:4]

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()

        for idx, year in enumerate(years_to_plot):
            if idx >= 4:
                break

            data = merged_df[merged_df['start_year'] == year]

            axes[idx].scatter(
                data['gender_score'],
                data['prestige_general_prestige'],
                alpha=0.6,
                s=50
            )
            axes[idx].set_xlabel('Gender Score (Male ← → Female)', fontsize=10)
            axes[idx].set_ylabel('General Prestige', fontsize=10)
            axes[idx].set_title(f'{year}-{year+9}', fontsize=12)
            axes[idx].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'gender_prestige_scatter.png', dpi=300)
        plt.close()
        logger.info(f"  Saved: gender_prestige_scatter.png")


def analyze_all_models(
    config: dict,
    logger: logging.Logger,
    specific_slice: str = None,
    export_plots: bool = False
) -> None:
    """
    Analyze all trained models.

    Args:
        config: Configuration dictionary
        logger: Logger instance
        specific_slice: If provided, only analyze this slice
        export_plots: Whether to export diagnostic plots
    """
    # Load word lists
    base_dir = Path(config['paths']['base_dir'])
    occupations = load_occupations(base_dir / config['wordlists']['occupations_file'], logger)
    gender_words = load_word_lists(base_dir / config['wordlists']['gender_words_file'], logger)
    prestige_axes = load_word_lists(base_dir / config['wordlists']['prestige_axes_file'], logger)

    # Find all models
    models_dir = Path(config['paths']['models_dir'])
    model_files = sorted(models_dir.glob("*.model"))

    if not model_files:
        logger.error(f"No model files found in {models_dir}")
        return

    logger.info(f"\nFound {len(model_files)} model files")

    # Filter to specific slice if requested
    if specific_slice:
        model_files = [f for f in model_files if specific_slice in f.name]
        if not model_files:
            logger.error(f"Model for slice {specific_slice} not found")
            return

    # Analyze each model
    all_gender_results = []
    all_prestige_results = []

    for model_path in model_files:
        # Parse slice name from filename
        # Expected format: chi_sim_5gram_1940_1949.model
        slice_name = model_path.stem.replace('chi_sim_5gram_', '')
        try:
            start_year, end_year = map(int, slice_name.split('_'))
        except ValueError:
            logger.warning(f"Could not parse years from {model_path.name}, skipping")
            continue

        # Analyze model
        gender_df, prestige_df = analyze_model(
            model_path,
            slice_name,
            start_year,
            end_year,
            occupations,
            gender_words,
            prestige_axes,
            config,
            logger
        )

        all_gender_results.append(gender_df)
        all_prestige_results.append(prestige_df)

    # Combine results
    combined_gender = pd.concat(all_gender_results, ignore_index=True)
    combined_prestige = pd.concat(all_prestige_results, ignore_index=True)

    # Save results
    results_dir = Path(config['paths']['results_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)

    gender_output = results_dir / 'occupation_gender_typing_by_decade.csv'
    prestige_output = results_dir / 'occupation_prestige_by_decade.csv'

    logger.info(f"\nSaving results...")
    combined_gender.to_csv(gender_output, index=False, encoding='utf-8')
    logger.info(f"  Saved: {gender_output}")

    combined_prestige.to_csv(prestige_output, index=False, encoding='utf-8')
    logger.info(f"  Saved: {prestige_output}")

    # Create joint file
    joint_df = pd.merge(
        combined_gender,
        combined_prestige,
        on=['occupation', 'time_slice', 'start_year', 'end_year'],
        suffixes=('_gender', '_prestige')
    )
    joint_output = results_dir / 'occupation_gender_prestige_joint.csv'
    joint_df.to_csv(joint_output, index=False, encoding='utf-8')
    logger.info(f"  Saved: {joint_output}")

    # Compute summary statistics
    summary_df = compute_summary_statistics(combined_gender, combined_prestige, logger)
    summary_output = results_dir / 'summary_statistics.csv'
    summary_df.to_csv(summary_output, index=False, encoding='utf-8')
    logger.info(f"  Saved: {summary_output}")

    # Create plots if requested
    if export_plots:
        plots_dir = results_dir / 'plots'
        create_plots(combined_gender, combined_prestige, plots_dir, logger)


def main(config='config/config.yml', slice=None, export_plots=False):
    """
    Analyze embeddings to compute gender typing and prestige scores.

    Args:
        config: Path to configuration file
        slice: Analyze only a specific slice (format: 1940_1949)
        export_plots: Export diagnostic visualizations
    """
    # Load configuration
    config_data = load_config(config)

    # Setup logging
    log_dir = Path(config_data['paths']['log_dir'])
    logger = setup_logging(log_dir)

    logger.info("="*80)
    logger.info("Starting embedding analysis")
    logger.info("="*80)

    # Analyze models
    analyze_all_models(config_data, logger, specific_slice=slice, export_plots=export_plots)

    logger.info("\n" + "="*80)
    logger.info("Analysis completed!")
    logger.info("="*80)

    return 0
if __name__ == "__main__":
    fire.Fire(main)
