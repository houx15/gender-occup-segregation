#!/usr/bin/env python3
"""
Unified Word2Vec embedding trainer.

Handles both modes:
- Longitudinal (ngram, renminribao): one model per time slice
- Provincial (weibo, newspaper): one model per province

Usage:
    python -m scripts.train_embeddings --config=config/config.yml
    python -m scripts.train_embeddings --config=config/config.yml --unit=1940_1949
    python -m scripts.train_embeddings --config=config/config.yml --group=0
    python -m scripts.train_embeddings --config=config/config.yml --retrain
"""

from __future__ import annotations

import gc
import glob
import json
import multiprocessing
from pathlib import Path
from typing import Iterator, List

import fire

from scripts.common.config_loader import load_config, get_analysis_unit, get_model_name
from scripts.common.logging_utils import setup_logging

# gensim is imported lazily inside train_model — it pulls in scipy/BLAS at
# import time, which we don't want (or need) for the pure-Python CorpusIterator,
# unit discovery, or their tests.


class CorpusIterator:
    """Iterator for reading corpus files from a unit directory.

    With ``expand_counts=True`` each line is treated as ``ngram<TAB>count`` and
    yielded ``count`` times — the per_year_capped subsampled corpus is stored
    COMPACT (one line per unique ngram per slice) so disk stays at type-size
    while word2vec still trains on ``count`` copies (Kozlowski et al. 2019). A
    line with no TAB count column counts as one copy. With the default
    ``expand_counts=False`` each line is one sentence (presence corpora, plain
    sentence corpora), unchanged.
    """
    def __init__(self, corpora_dir: str, unit_name: str, expand_counts: bool = False):
        unit_dir = Path(corpora_dir) / unit_name
        self.corpus_files = sorted(glob.glob(str(unit_dir / "corpus_*")))
        self.expand_counts = expand_counts

    def __iter__(self) -> Iterator[List[str]]:
        for corpus_file in self.corpus_files:
            with open(corpus_file, "r", encoding="utf-8", buffering=8 * 1024 * 1024) as f:
                if self.expand_counts:
                    for line in f:
                        line = line.rstrip("\n")
                        if not line:
                            continue
                        ngram, _, count = line.partition("\t")
                        tokens = ngram.split()
                        if not tokens:
                            continue
                        n = int(count) if count else 1
                        for _ in range(n):
                            yield tokens
                else:
                    for line in f:
                        tokens = line.strip().split()
                        if tokens:
                            yield tokens


def train_model(unit_name: str, config: dict, logger) -> "Word2Vec | None":
    """Train a Word2Vec model for a single unit (time slice or province).

    Returns ``None`` (instead of crashing) when the unit has too little text to
    form a vocabulary at ``min_count`` — e.g. a sparse {state}_{year}. gensim's
    one-shot constructor would raise "you must first build vocabulary" mid-run
    and abort the whole batch; we build vocab first, check it, and skip.
    """
    from gensim.models import Word2Vec
    from gensim.models.callbacks import CallbackAny2Vec

    class EpochLogger(CallbackAny2Vec):
        """Callback to log training progress."""
        def __init__(self, logger, unit_name):
            self.epoch = 0
            self.logger = logger
            self.unit_name = unit_name

        def on_epoch_end(self, model):
            self.epoch += 1
            self.logger.info(f"  {self.unit_name} - Completed epoch {self.epoch}")

    logger.info(f"\nTraining model for {unit_name}...")
    emb_config = config["embedding"]

    # per_year_capped corpora are stored compact (ngram<TAB>count); expand at
    # read time so word2vec sees the repeated sentences. Any other corpus is
    # one-sentence-per-line.
    expand_counts = config.get("corpus", {}).get("weight_mode") == "per_year_capped"
    corpus = CorpusIterator(config["paths"]["corpora_dir"], unit_name,
                            expand_counts=expand_counts)

    workers = min(multiprocessing.cpu_count() - 1, emb_config.get("workers", 16))
    epoch_logger = EpochLogger(logger, unit_name)

    logger.info(f"  vector_size={emb_config['vector_size']}, window={emb_config['window']}, "
                f"min_count={emb_config['min_count']}, sg={emb_config['sg']}, "
                f"epochs={emb_config['epochs']}, workers={workers}")

    # Build the vocabulary first so we can detect (and skip) units too sparse to
    # train, rather than let gensim raise mid-run and abort the whole batch.
    model = Word2Vec(
        vector_size=emb_config["vector_size"],
        window=emb_config["window"],
        min_count=emb_config["min_count"],
        sg=emb_config["sg"],
        negative=emb_config.get("negative", 15),
        workers=workers,
        epochs=emb_config["epochs"],
        seed=emb_config.get("seed", 42),
        compute_loss=True,
    )
    model.build_vocab(corpus)
    if len(model.wv) == 0:
        logger.warning(
            f"  {unit_name}: empty vocabulary — no token reaches "
            f"min_count={emb_config['min_count']}; skipping (too little data)"
        )
        return None

    model.train(
        corpus,
        total_examples=model.corpus_count,
        epochs=emb_config["epochs"],
        compute_loss=True,
        callbacks=[epoch_logger],
    )
    model.wv.fill_norms()

    logger.info(f"  Completed {unit_name}: vocab={len(model.wv):,}, loss={model.get_latest_training_loss():.2f}")
    return model


def save_model(model: Word2Vec, unit_name: str, config: dict, logger, **extra_meta):
    """Save trained model and metadata."""
    models_dir = Path(config["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)

    model_filename = get_model_name(unit_name, config)
    model_path = models_dir / model_filename
    logger.info(f"  Saving model to {model_path}")
    model.wv.save(str(model_path))

    metadata = {
        "unit_name": unit_name,
        "vector_size": config["embedding"]["vector_size"],
        "window": config["embedding"]["window"],
        "min_count": config["embedding"]["min_count"],
        "sg": config["embedding"]["sg"],
        "negative": config["embedding"].get("negative", 15),
        "epochs": config["embedding"]["epochs"],
        "seed": config["embedding"].get("seed", 42),
        "vocab_size": len(model.wv),
        **extra_meta,
    }
    # Strip extension and add .meta.json
    meta_filename = str(Path(model_filename).stem) + ".meta.json"
    meta_path = models_dir / meta_filename
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def discover_units(config: dict) -> List[str]:
    """Discover available units (time slice dirs or province dirs) in corpora_dir."""
    corpora_dir = Path(config["paths"]["corpora_dir"])
    if not corpora_dir.exists():
        return []
    return sorted([d.name for d in corpora_dir.iterdir() if d.is_dir()])


def train_all(config: dict, logger, specific_unit=None, group=None, retrain=False):
    """Train models for all units."""
    models_dir = Path(config["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)

    units = discover_units(config)
    if not units:
        logger.error(f"No unit directories found in {config['paths']['corpora_dir']}")
        return

    # Filter by specific unit
    if specific_unit:
        units = [u for u in units if u.startswith(str(specific_unit))]
        if not units:
            logger.error(f"Unit {specific_unit} not found")
            return

    # Filter by group (provincial mode)
    if group is not None:
        prov_config = config.get("provincial", {})
        groups = prov_config.get("province_groups", [])
        if 0 <= group < len(groups):
            target_provinces = set(groups[group])
            # Support province-year units (e.g. 北京_2020 matches province 北京)
            units = [u for u in units
                     if u in target_provinces
                     or any(u.startswith(p + "_") for p in target_provinces)]
            logger.info(f"Training group {group}: provinces={target_provinces}, {len(units)} units")
        else:
            logger.error(f"Invalid group index: {group}")
            return

    logger.info(f"Training {len(units)} models")

    for unit_name in units:
        model_filename = get_model_name(unit_name, config)
        model_path = models_dir / model_filename
        if model_path.exists() and not retrain:
            logger.info(f"Skipping {unit_name} (model exists, use --retrain to overwrite)")
            continue

        extra_meta = {}
        # For longitudinal units, parse years
        analysis_unit = get_analysis_unit(config)
        if analysis_unit == "longitudinal":
            try:
                start_year, end_year = map(int, unit_name.split("_"))
                extra_meta = {"start_year": start_year, "end_year": end_year}
            except ValueError:
                pass

        model = train_model(unit_name, config, logger)
        if model is None:
            continue  # too sparse to train; already warned. Next unit.
        save_model(model, unit_name, config, logger, **extra_meta)

        del model
        gc.collect()


def main(config: str = "config/config.yml", unit: str = None, group: str = None, retrain: bool = False):
    """
    Train Word2Vec embeddings.

    Args:
        config: Path to configuration file
        unit: Train only a specific unit (time slice name or province name)
        group: Train only a specific province group (index)
        retrain: Retrain existing models

    The ``: str`` annotations on ``unit`` / ``group`` are load-bearing: Fire
    auto-coerces CLI args via ``ast.literal_eval``, and PEP 515 makes
    ``1940_1949`` a valid int literal (=> ``19401949``). Without the
    annotation, ``--unit=1940_1949`` would arrive as int and silently fail
    the ``units[i].startswith(...)`` filter.
    """
    config_data = load_config(config)
    logger = setup_logging(Path(config_data["paths"]["log_dir"]), "train_embeddings.log")

    logger.info("=" * 80)
    logger.info(f"Starting embedding training (source={config_data['data_source']}, "
                f"unit={get_analysis_unit(config_data)})")
    logger.info("=" * 80)

    train_all(config_data, logger, specific_unit=unit, group=group, retrain=retrain)

    logger.info("=" * 80)
    logger.info("Embedding training completed!")
    logger.info("=" * 80)


if __name__ == "__main__":
    fire.Fire(main)
