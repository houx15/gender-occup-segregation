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
from datetime import datetime
import yaml
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
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


def load_occupations(file_path: Path, logger: logging.Logger) -> List[str]:
    """Load occupation list from file."""
    logger.info(f"Loading occupations from {file_path}")
    occupations = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            occ = line.strip()
            if occ:
                occupations.append(occ)
    logger.info(f"  Loaded {len(occupations)} occupations")
    return occupations


def load_word_lists(file_path: Path, logger: logging.Logger) -> dict:
    """Load word lists from JSON file."""
    logger.info(f"Loading word lists from {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_occupation_vector(
    occupation: str, model: KeyedVectors,
) -> Optional[np.ndarray]:
    """
    Get vector representation for an occupation.

    Args:
        occupation: Occupation string
        model: KeyedVectors model

    Returns:
        numpy array or None if not found in vocabulary
    """
    vocab = model.key_to_index

    # Only use whole token strategy
    if occupation in vocab:
        return model[occupation]

    return None


def construct_semantic_axis(
    positive_terms: List[str],
    negative_terms: List[str],
    model: KeyedVectors,
    logger: logging.Logger,
) -> Tuple[Optional[np.ndarray], int, int]:
    """
    Construct a semantic axis from positive and negative terms.

    Args:
        positive_terms: List of positive pole terms
        negative_terms: List of negative pole terms
        model: KeyedVectors model
        logger: Logger instance

    Returns:
        Tuple of (axis_vector, num_positive_found, num_negative_found)
    """
    vocab = model.key_to_index

    # Get vectors for positive terms (only whole terms)
    positive_vectors = []
    for term in positive_terms:
        if term in vocab:
            positive_vectors.append(model[term])

    # Get vectors for negative terms (only whole terms)
    negative_vectors = []
    for term in negative_terms:
        if term in vocab:
            negative_vectors.append(model[term])

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
    logger: logging.Logger,
) -> pd.DataFrame:
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
        DataFrame with columns: occupation, gender_score, {dimension}_score for each dimension
    """
    logger.info(f"\nAnalyzing {slice_name}...")

    # Load model
    logger.info(f"  Loading model from {model_path}")
    model = KeyedVectors.load(str(model_path))

    # Construct gender axis
    logger.info(f"  Constructing gender axis...")
    gender_axis, num_female, num_male = construct_semantic_axis(
        gender_words["female_terms"], gender_words["male_terms"], model, logger
    )

    if gender_axis is None:
        logger.warning(f"  Could not construct gender axis for {slice_name}")
        logger.warning(
            f"    Male terms found: {num_male}, Female terms found: {num_female}"
        )
    else:
        logger.info(
            f"    Male terms found: {num_male}/{len(gender_words['male_terms'])}"
        )
        logger.info(
            f"    Female terms found: {num_female}/{len(gender_words['female_terms'])}"
        )

    # Construct prestige axes
    logger.info(f"  Constructing prestige axes...")
    prestige_axis_vectors = {}

    for dimension, terms in prestige_axes.items():
        axis, num_pos, num_neg = construct_semantic_axis(
            terms["positive"], terms["negative"], model, logger
        )

        if axis is None:
            logger.warning(f"    Could not construct {dimension} axis")
            logger.warning(f"      Positive: {num_pos}, Negative: {num_neg}")
        else:
            prestige_axis_vectors[dimension] = axis
            logger.info(
                f"    {dimension}: {num_pos} positive, {num_neg} negative terms found"
            )

    # Analyze occupations
    logger.info(f"  Analyzing {len(occupations)} occupations...")

    results = []

    for occupation in occupations:
        # Get occupation vector
        occ_vector = get_occupation_vector(
            occupation, model
        )

        if occ_vector is None:
            continue

        # Initialize result row for this occupation
        row = {
            "occupation": occupation,
        }

        # Compute gender score, positive -> female
        if gender_axis is not None:
            gender_score = float(np.dot(occ_vector, gender_axis))
        else:
            gender_score = np.nan
        row["gender_score"] = gender_score

        # Compute prestige scores for each dimension
        for dimension, axis in prestige_axis_vectors.items():
            score = float(np.dot(occ_vector, axis))
            row[f"{dimension}_score"] = score

        results.append(row)

    logger.info(
        f"  Analyzed {len(results)} occupations with sufficient coverage"
    )

    # Convert to DataFrame
    df = pd.DataFrame(results)

    return df


def compute_summary_statistics(
    df: pd.DataFrame, logger: logging.Logger
) -> pd.DataFrame:
    """
    Compute summary statistics and correlations.

    Args:
        df: Combined DataFrame with occupation scores
        logger: Logger instance

    Returns:
        Summary statistics DataFrame
    """
    logger.info("\nComputing summary statistics...")

    summary_results = []

    # Group by time slice
    for slice_name in df["time_slice"].unique():
        slice_data = df[df["time_slice"] == slice_name]

        summary = {
            "time_slice": slice_name,
            "start_year": slice_data["start_year"].iloc[0],
            "end_year": slice_data["end_year"].iloc[0],
            "num_occupations": len(slice_data),
            "mean_gender_score": slice_data["gender_score"].mean(),
            "std_gender_score": slice_data["gender_score"].std(),
        }

        # Compute correlations between gender and each dimension score
        dimension_cols = [
            col for col in slice_data.columns if col.endswith("_score") and col != "gender_score"
        ]

        for col in dimension_cols:
            dimension = col.replace("_score", "")
            if (
                slice_data["gender_score"].notna().sum() > 1
                and slice_data[col].notna().sum() > 1
            ):
                corr = slice_data["gender_score"].corr(slice_data[col])
                summary[f"corr_gender_{dimension}"] = corr
            else:
                summary[f"corr_gender_{dimension}"] = np.nan

        summary_results.append(summary)

    summary_df = pd.DataFrame(summary_results)

    return summary_df


def get_figure_path(filename: str) -> Path:
    """
    Get the full path for saving a figure with date prefix.

    Args:
        filename: Base filename (without extension, will be saved as PDF)

    Returns:
        Path object for the figure file
    """
    figures_dir = Path("figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Get current date in yyyymmdd format
    date_str = datetime.now().strftime("%Y%m%d")
    
    # Add date prefix to filename
    dated_filename = f"{date_str}_{filename}"
    if not dated_filename.endswith(".pdf"):
        dated_filename += ".pdf"
    
    return figures_dir / dated_filename


def create_plots(
    df: pd.DataFrame,
    logger: logging.Logger,
) -> None:
    """
    Create diagnostic visualizations.

    Args:
        df: Combined DataFrame with occupation scores
        logger: Logger instance
    """
    logger.info("\nCreating diagnostic plots...")

    # Set style
    sns.set_style("whitegrid")

    # Plot 1: Prestige scores for most male vs most female occupations over time
    if not df.empty and "gender_score" in df.columns:
        # Find all prestige dimension columns
        prestige_cols = [
            col for col in df.columns if col.endswith("_score") and col != "gender_score"
        ]
        
        if prestige_cols:
            # Determine subplot layout (2x2 for up to 4 dimensions)
            n_dims = len(prestige_cols)
            n_cols = 2
            n_rows = (n_dims + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 6 * n_rows))
            if n_dims == 1:
                axes = [axes]
            else:
                axes = axes.flatten()
            
            # Get unique time slices sorted by start_year
            time_slices = df.sort_values("start_year")["time_slice"].unique()
            
            for idx, prestige_col in enumerate(prestige_cols):
                ax = axes[idx]
                
                # Store data for plotting
                male_scores = []
                female_scores = []
                x_labels = []
                
                for time_slice in time_slices:
                    slice_data = df[df["time_slice"] == time_slice].copy()
                    
                    if len(slice_data) < 10:
                        # Not enough data for 10% calculation
                        continue
                    
                    # Calculate 10% threshold
                    n_top = int(max(1, len(slice_data) // 10))
                    
                    # Most female (highest gender_score) - 10%
                    most_female = slice_data.nlargest(n_top, "gender_score")
                    female_avg = most_female[prestige_col].mean()
                    
                    # Most male (lowest gender_score) - 10%
                    most_male = slice_data.nsmallest(n_top, "gender_score")
                    male_avg = most_male[prestige_col].mean()
                    
                    male_scores.append(male_avg)
                    female_scores.append(female_avg)
                    x_labels.append(time_slice)
                
                # Plot lines
                x_positions = range(len(x_labels))
                ax.plot(
                    x_positions, male_scores,
                    marker="o", linewidth=2, linestyle="-", label="Most Male (10%)"
                )
                ax.plot(
                    x_positions, female_scores,
                    marker="s", linewidth=2, linestyle="--", label="Most Female (10%)"
                )
                
                # Set labels and title
                ax.set_xticks(x_positions)
                ax.set_xticklabels(x_labels, rotation=45, ha="right")
                ax.set_xlabel("Time Slice", fontsize=10)
                dimension_name = prestige_col.replace("_score", "").replace("_", " ").title()
                ax.set_ylabel(f"Mean {dimension_name} Score", fontsize=10)
                ax.set_title(f"{dimension_name} by Gender Typing", fontsize=12)
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3)
            
            # Hide unused subplots
            for idx in range(n_dims, len(axes)):
                axes[idx].set_visible(False)
            
            plt.tight_layout()
            fig_path = get_figure_path("prestige_by_gender_over_time")
            plt.savefig(fig_path, format="pdf")
            plt.close()
            logger.info(f"  Saved: {fig_path.name}")

    # Plot 2: Gender vs dimension scores scatter (selected decades)
    # Find dimension score columns (excluding gender_score)
    dimension_cols = [
        col for col in df.columns if col.endswith("_score") and col != "gender_score"
    ]

    if dimension_cols:
        # Use the first dimension for plotting (or look for a specific one)
        target_dimension = None
        for dim_name in ["general_prestige", "prestige"]:
            if f"{dim_name}_score" in df.columns:
                target_dimension = f"{dim_name}_score"
                break
        if target_dimension is None:
            target_dimension = dimension_cols[0]

        if target_dimension in df.columns:
            # Select a few representative decades
            years_to_plot = sorted(df["start_year"].unique())
            if len(years_to_plot) > 4:
                years_to_plot = years_to_plot[:: len(years_to_plot) // 4][:4]

            fig, axes = plt.subplots(2, 2, figsize=(14, 12))
            axes = axes.flatten()

            for idx, year in enumerate(years_to_plot):
                if idx >= 4:
                    break

                data = df[df["start_year"] == year]

                axes[idx].scatter(
                    data["gender_score"], data[target_dimension], alpha=0.6, s=50
                )
                axes[idx].set_xlabel("Gender Score (Male ← → Female)", fontsize=10)
                dimension_name = target_dimension.replace("_score", "").replace("_", " ").title()
                axes[idx].set_ylabel(dimension_name, fontsize=10)
                axes[idx].set_title(f"{year}-{year+9}", fontsize=12)
                axes[idx].grid(True, alpha=0.3)

            plt.tight_layout()
            plot_name = f"gender_{target_dimension.replace('_score', '')}_scatter"
            fig_path = get_figure_path(plot_name)
            plt.savefig(fig_path, format="pdf")
            plt.close()
            logger.info(f"  Saved: {fig_path.name}")

    # Plot 3: Correlation between gender_score and prestige scores over time
    if not df.empty and "gender_score" in df.columns:
        prestige_cols = [
            col for col in df.columns if col.endswith("_score") and col != "gender_score"
        ]
        
        if prestige_cols:
            # Determine subplot layout
            n_dims = len(prestige_cols)
            n_cols = 2
            n_rows = (n_dims + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 6 * n_rows))
            if n_dims == 1:
                axes = [axes]
            else:
                axes = axes.flatten()
            
            # Get unique time slices sorted by start_year
            time_slices = df.sort_values("start_year")["time_slice"].unique()
            
            for idx, prestige_col in enumerate(prestige_cols):
                ax = axes[idx]
                
                # Store correlation data
                correlations = []
                x_labels = []
                
                for time_slice in time_slices:
                    slice_data = df[df["time_slice"] == time_slice].copy()
                    
                    # Calculate correlation between gender_score and prestige_score
                    if (
                        slice_data["gender_score"].notna().sum() > 1
                        and slice_data[prestige_col].notna().sum() > 1
                    ):
                        corr = slice_data["gender_score"].corr(slice_data[prestige_col])
                        correlations.append(corr)
                        x_labels.append(time_slice)
                    else:
                        correlations.append(np.nan)
                        x_labels.append(time_slice)
                
                # Plot correlation line
                x_positions = range(len(x_labels))
                ax.plot(
                    x_positions, correlations,
                    marker="o", linewidth=2, linestyle="-"
                )
                
                # Add horizontal line at y=0 for reference
                ax.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
                
                # Set labels and title
                ax.set_xticks(x_positions)
                ax.set_xticklabels(x_labels, rotation=45, ha="right")
                ax.set_xlabel("Time Slice", fontsize=10)
                ax.set_ylabel(
                    "Correlation\n(Positive: Female ↔ High Prestige)",
                    fontsize=10
                )
                dimension_name = prestige_col.replace("_score", "").replace("_", " ").title()
                ax.set_title(
                    f"Gender-Prestige Correlation: {dimension_name}",
                    fontsize=12
                )
                ax.grid(True, alpha=0.3)
                
                # Set y-axis limits to show full range
                if len(correlations) > 0 and not all(np.isnan(correlations)):
                    valid_corrs = [c for c in correlations if not np.isnan(c)]
                    if valid_corrs:
                        y_min = min(valid_corrs)
                        y_max = max(valid_corrs)
                        y_range = y_max - y_min
                        if y_range > 0:
                            ax.set_ylim(
                                y_min - 0.1 * y_range,
                                y_max + 0.1 * y_range
                            )
            
            # Hide unused subplots
            for idx in range(n_dims, len(axes)):
                axes[idx].set_visible(False)
            
            plt.tight_layout()
            fig_path = get_figure_path("gender_prestige_correlation_over_time")
            plt.savefig(fig_path, format="pdf")
            plt.close()
            logger.info(f"  Saved: {fig_path.name}")

    # Plot 4: Prestige scores by occupation category over time
    if not df.empty and "gender_score" in df.columns:
        # Load occupation categories
        base_dir = Path(__file__).parent.parent
        occup_category_path = base_dir / "wordlists" / "occup_category.json"
        
        try:
            with open(occup_category_path, "r", encoding="utf-8") as f:
                occup_category = json.load(f)
            
            prestige_cols = [
                col for col in df.columns if col.endswith("_score") and col != "gender_score"
            ]
            
            if prestige_cols:
                # Get unique time slices sorted by start_year
                time_slices = df.sort_values("start_year")["time_slice"].unique()
                
                # Calculate average gender score for each category (across all time slices)
                category_avg_gender = {}
                for category, occupations in occup_category.items():
                    category_data = df[df["occupation"].isin(occupations)]
                    if not category_data.empty:
                        category_avg_gender[category] = category_data["gender_score"].mean()
                
                # Sort categories by average gender score (highest = most female)
                sorted_categories = sorted(
                    category_avg_gender.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                
                # Calculate number of rows needed (4 categories per row)
                n_categories = len(sorted_categories)
                n_cols = 4
                n_rows = (n_categories + n_cols - 1) // n_cols
                
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
                # Ensure axes is always a flat array for indexing
                if not isinstance(axes, np.ndarray):
                    axes = np.array([axes])
                elif axes.ndim > 1:
                    axes = axes.flatten()
                else:
                    axes = axes
                
                for row_idx, (category, avg_gender) in enumerate(sorted_categories):
                    ax = axes[row_idx]
                    occupations = occup_category[category]
                    
                    # Filter data for this category
                    category_data = df[df["occupation"].isin(occupations)].copy()
                    
                    if category_data.empty:
                        ax.text(0.5, 0.5, f"No data for\n{category}", 
                               ha="center", va="center", transform=ax.transAxes)
                        ax.set_title(f"{category}\n(Avg Gender: {avg_gender:.3f})", fontsize=10)
                        continue
                    
                    # Calculate mean scores for each time slice
                    time_slice_data = []
                    for time_slice in time_slices:
                        slice_data = category_data[category_data["time_slice"] == time_slice]
                        if not slice_data.empty:
                            row = {"time_slice": time_slice}
                            # Calculate mean for each prestige dimension
                            for prestige_col in prestige_cols:
                                row[prestige_col] = slice_data[prestige_col].mean()
                            # Calculate overall mean prestige
                            row["mean_prestige"] = np.mean([slice_data[col].mean() for col in prestige_cols])
                            time_slice_data.append(row)
                    
                    if not time_slice_data:
                        ax.text(0.5, 0.5, f"No data for\n{category}", 
                               ha="center", va="center", transform=ax.transAxes)
                        ax.set_title(f"{category}\n(Avg Gender: {avg_gender:.3f})", fontsize=10)
                        continue
                    
                    time_slice_df = pd.DataFrame(time_slice_data)
                    x_positions = range(len(time_slices))
                    
                    # Plot each prestige dimension
                    colors = plt.cm.tab10(np.linspace(0, 1, len(prestige_cols)))
                    for idx, prestige_col in enumerate(prestige_cols):
                        values = []
                        for time_slice in time_slices:
                            row = time_slice_df[time_slice_df["time_slice"] == time_slice]
                            if not row.empty:
                                values.append(row[prestige_col].iloc[0])
                            else:
                                values.append(np.nan)
                        
                        dimension_name = prestige_col.replace("_score", "").replace("_", " ").title()
                        ax.plot(
                            x_positions, values,
                            marker="o", linewidth=1.5, linestyle="-",
                            label=dimension_name, color=colors[idx], alpha=0.7
                        )
                    
                    # Plot mean prestige
                    mean_values = []
                    for time_slice in time_slices:
                        row = time_slice_df[time_slice_df["time_slice"] == time_slice]
                        if not row.empty:
                            mean_values.append(row["mean_prestige"].iloc[0])
                        else:
                            mean_values.append(np.nan)
                    
                    ax.plot(
                        x_positions, mean_values,
                        marker="s", linewidth=2, linestyle="--",
                        label="Mean", color="black", alpha=0.8
                    )
                    
                    # Set labels and title
                    ax.set_xticks(x_positions)
                    ax.set_xticklabels(time_slices, rotation=45, ha="right", fontsize=8)
                    ax.set_xlabel("Time Slice", fontsize=9)
                    ax.set_ylabel("Prestige Score", fontsize=9)
                    ax.set_title(f"{category}\n(Avg Gender: {avg_gender:.3f})", fontsize=10)
                    ax.legend(fontsize=7, loc="best", ncol=2)
                    ax.grid(True, alpha=0.3)
                
                # Hide unused subplots
                for idx in range(n_categories, len(axes)):
                    axes[idx].set_visible(False)
                
                plt.tight_layout()
                fig_path = get_figure_path("prestige_by_category_over_time")
                plt.savefig(fig_path, format="pdf")
                plt.close()
                logger.info(f"  Saved: {fig_path.name}")
        
        except FileNotFoundError:
            logger.warning(f"Could not find occup_category.json at {occup_category_path}, skipping category plot")
        except Exception as e:
            logger.warning(f"Error creating category plot: {e}")


def analyze_all_models(
    config: dict,
    logger: logging.Logger,
    specific_slice: str = None,
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
    occupations = load_occupations(config["wordlists"]["occupations_file"], logger)
    gender_words = load_word_lists(config["wordlists"]["gender_words_file"], logger)
    prestige_axes = load_word_lists(config["wordlists"]["prestige_axes_file"], logger)

    # Find all models
    models_dir = Path(config["paths"]["models_dir"])
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
    all_results = []

    for model_path in model_files:
        # Parse slice name from filename
        # Expected format: chi_sim_5gram_1940_1949.model
        slice_name = model_path.stem.replace("chi_sim_5gram_", "")
        try:
            start_year, end_year = map(int, slice_name.split("_"))
        except ValueError:
            logger.warning(f"Could not parse years from {model_path.name}, skipping")
            continue

        # Analyze model
        df = analyze_model(
            model_path,
            slice_name,
            start_year,
            end_year,
            occupations,
            gender_words,
            prestige_axes,
            config,
            logger,
        )

        # Add slice information to each row
        df["time_slice"] = slice_name
        df["start_year"] = start_year
        df["end_year"] = end_year

        all_results.append(df)

    # Combine results
    combined_df = pd.concat(all_results, ignore_index=True)

    # Reorder columns: occupation, time_slice, start_year, end_year, gender_score, then dimension scores
    dimension_cols = [col for col in combined_df.columns if col.endswith("_score")]
    other_cols = ["occupation", "time_slice", "start_year", "end_year"]
    column_order = other_cols + sorted(dimension_cols)
    combined_df = combined_df[column_order]

    # Save results
    results_dir = Path(config["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    output_file = results_dir / "occupation_scores_by_slice.parquet"

    logger.info(f"\nSaving results...")
    combined_df.to_parquet(output_file, index=False)
    logger.info(f"  Saved: {output_file}")

    # Compute summary statistics
    summary_df = compute_summary_statistics(combined_df, logger)
    summary_output = results_dir / "summary_statistics.parquet"
    summary_df.to_parquet(summary_output, index=False)
    logger.info(f"  Saved: {summary_output}")

    return combined_df


def main(config="config/config.yml", slice=None):
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
    log_dir = Path(config_data["paths"]["log_dir"])
    logger = setup_logging(log_dir)

    logger.info("=" * 80)
    logger.info("Starting embedding analysis")
    logger.info("=" * 80)

    # Analyze models
    combined_df = analyze_all_models(
        config_data, logger, specific_slice=slice
    )

    create_plots(combined_df, logger)

    logger.info("\n" + "=" * 80)
    logger.info("Analysis completed!")
    logger.info("=" * 80)

    return 0


if __name__ == "__main__":
    fire.Fire(main)
