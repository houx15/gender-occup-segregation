#!/usr/bin/env python3
"""
Train word2vec embeddings for each time slice.

Usage:
    python train_embeddings.py --config=config/config.yml
    python train_embeddings.py --config=config/config.yml --slice=1940_1949
    python train_embeddings.py --config=config/config.yml --retrain
"""

import logging
import sys
import json
from pathlib import Path
from typing import Iterator, List
import yaml
from gensim.models import Word2Vec
import fire
from gensim.models.callbacks import CallbackAny2Vec


class EpochLogger(CallbackAny2Vec):
    """Callback to log training progress."""

    def __init__(self, logger, slice_name):
        self.epoch = 0
        self.logger = logger
        self.slice_name = slice_name

    def on_epoch_end(self, model):
        self.epoch += 1
        self.logger.info(f"  {self.slice_name} - Completed epoch {self.epoch}")


def setup_logging(log_dir: Path) -> logging.Logger:
    """Configure logging to both file and console."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "train_embeddings.log"

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


class CorpusIterator:
    """Iterator for reading corpus files line by line."""

    def __init__(self, corpora_dir: str, slice_name: str):
        self.corpus_files = glob.glob(f"{corpora_dir}/{slice_name}/corpus_*.txt")
        self.corpus_files.sort()

    def __iter__(self) -> Iterator[List[str]]:
        """Yield tokenized sentences from the corpus."""
        for corpus_file in self.corpus_files:
            with open(
                corpus_file, "r", encoding="utf-8", buffering=8 * 1024 * 1024
            ) as f:
                for line in f:
                    # Split on whitespace to get tokens
                    tokens = line.strip().split()
                    if tokens:  # Skip empty lines
                        yield tokens


def count_corpus_lines(corpus_file: Path) -> int:
    """Count the number of lines in a corpus file."""
    count = 0
    with open(corpus_file, "r", encoding="utf-8") as f:
        for _ in f:
            count += 1
    return count


def train_model(slice_name: str, config: dict, logger: logging.Logger) -> Word2Vec:
    """
    Train a word2vec model for a single time slice.

    Args:
        slice_name: Name of the time slice (e.g., "1940_1949")
        config: Configuration dictionary
        logger: Logger instance

    Returns:
        Trained Word2Vec model
    """
    logger.info(f"\nTraining model for {slice_name}...")

    # Get embedding configuration
    emb_config = config["embedding"]

    # Create corpus iterator
    corpora_dir = Path(config["paths"]["corpora_dir"])
    corpus = CorpusIterator(corpora_dir, slice_name)

    # Initialize model
    logger.info(f"  Initializing Word2Vec model...")
    logger.info(f"    Vector size: {emb_config['vector_size']}")
    logger.info(f"    Window: {emb_config['window']}")
    logger.info(f"    Min count: {emb_config['min_count']}")
    logger.info(f"    Skip-gram: {emb_config['sg']}")
    logger.info(f"    Negative sampling: {emb_config['negative']}")
    logger.info(f"    Workers: {emb_config['workers']}")
    logger.info(f"    Epochs: {emb_config['epochs']}")

    # Train model
    logger.info(f"  Training for {emb_config['epochs']} epochs...")
    epoch_logger = EpochLogger(logger, slice_name)

    import multiprocessing

    workers = multiprocessing.cpu_count() - 1
    workers = min(workers, emb_config["workers"])

    model = Word2Vec(
        sentences=corpus,
        vector_size=emb_config["vector_size"],
        window=emb_config["window"],
        min_count=emb_config["min_count"],
        sg=emb_config["sg"],
        negative=emb_config["negative"],
        workers=workers,
        epochs=emb_config["epochs"],
        seed=emb_config["seed"],
        compute_loss=True,
        callbacks=[epoch_logger],
    )

    # Normalize vectors to unit length
    model.wv.fill_norms()

    logger.info(f"  Training completed for {slice_name}")
    logger.info(f"    Final loss: {model.get_latest_training_loss():.2f}")

    return model


def save_model_and_metadata(
    model: Word2Vec,
    slice_name: str,
    start_year: int,
    end_year: int,
    config: dict,
    logger: logging.Logger,
) -> None:
    """
    Save the trained model and its metadata.

    Args:
        model: Trained Word2Vec model
        slice_name: Name of the time slice
        start_year: Start year of the slice
        end_year: End year of the slice
        config: Configuration dictionary
        logger: Logger instance
    """
    models_dir = Path(config["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = models_dir / f"chi_sim_5gram_{slice_name}.model"
    logger.info(f"  Saving model to {model_path}")
    model.wv.save(str(model_path))

    # Save metadata
    metadata = {
        "time_slice": slice_name,
        "start_year": start_year,
        "end_year": end_year,
        "vector_size": config["embedding"]["vector_size"],
        "window": config["embedding"]["window"],
        "min_count": config["embedding"]["min_count"],
        "sg": config["embedding"]["sg"],
        "negative": config["embedding"]["negative"],
        "epochs": config["embedding"]["epochs"],
        "seed": config["embedding"]["seed"],
        "vocab_size": len(model.wv),
    }

    metadata_path = models_dir / f"chi_sim_5gram_{slice_name}.meta.json"
    logger.info(f"  Saving metadata to {metadata_path}")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def train_all_models(
    config: dict,
    logger: logging.Logger,
    specific_slice: str = None,
    retrain: bool = False,
) -> None:
    """
    Train models for all time slices.

    Args:
        config: Configuration dictionary
        logger: Logger instance
        specific_slice: If provided, only train this slice
        retrain: Whether to retrain existing models
    """
    corpora_dir = Path(config["paths"]["corpora_dir"])
    models_dir = Path(config["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)

    # Find all time slice directories
    slice_dirs = sorted([d for d in corpora_dir.iterdir() if d.is_dir()])

    if not slice_dirs:
        logger.error(f"No time slice directories found in {corpora_dir}")
        return

    logger.info(f"Found {len(slice_dirs)} time slice directories")

    # Filter to specific slice if requested
    if specific_slice:
        # found dir with name started by specific_slice
        slice_dirs = [d for d in slice_dirs if d.name.startswith(str(specific_slice))]
        if not slice_dirs:
            logger.error(f"Time slice {specific_slice} not found")
            return
        logger.info(f"Training only slice: {specific_slice}")

    # Train each model
    for slice_dir in slice_dirs:
        slice_name = slice_dir.name
        # Check if model already exists
        model_path = models_dir / f"chi_sim_5gram_{slice_name}.model"
        if model_path.exists() and not retrain:
            logger.info(
                f"Model for {slice_name} already exists, skipping (use --retrain to overwrite)"
            )
            continue
        # Parse start and end years from slice name
        try:
            start_year, end_year = map(int, slice_name.split("_"))
        except ValueError:
            logger.warning(f"Could not parse years from {slice_name}, skipping")
            continue

        # Train model
        model = train_model(slice_name, config, logger)

        # Save model and metadata
        save_model_and_metadata(model, slice_name, start_year, end_year, config, logger)


def main(config="config/config.yml", time_slice=None, retrain=False):
    """
    Train word2vec embeddings for time-sliced Chinese corpora.

    Args:
        config: Path to configuration file
        time_slice: Train only a specific slice (format: 1940)
        retrain: Retrain existing models
    """
    # Load configuration
    config_data = load_config(config)

    # Setup logging
    log_dir = Path(config_data["paths"]["log_dir"])
    logger = setup_logging(log_dir)

    logger.info("=" * 80)
    logger.info("Starting embedding training")
    logger.info("=" * 80)

    # Train models
    train_all_models(config_data, logger, specific_slice=time_slice, retrain=retrain)

    logger.info("=" * 80)
    logger.info("Embedding training completed!")
    logger.info("=" * 80)

    return 0


if __name__ == "__main__":
    fire.Fire(main)
