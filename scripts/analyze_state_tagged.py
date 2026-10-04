#!/usr/bin/env python3
"""Approach A: state scores from ONE shared model per period with state-tagged anchors.

The corpus builder (us_states.tag_gender_by_state) pools every state's articles
into one corpus per period and tags only the gender anchor words with the state
('she__ohio'). List words (occupations, family words) are untagged, so their
vectors are learned from all of the period's text; each state contributes only
its own gender centroids. A state's score is the RND of the shared list words
against that state's tagged male / female centroids — differences between states
come from how each state's text uses gendered words, not from separate training
runs on small corpora.

Outputs use the same unit names ('ohio_2005') and files as analyze_category_bias
(long + summary parquets, word_coverage.csv), so check_state_benchmarks,
diagnose_unit_stability and the US plots work unchanged.

States are kept when (a) they meet us_states.min_documents in that period
(coverage_<arm>.csv) and (b) each pole has >= analysis.min_tagged_anchors
tagged anchors in vocab.

Usage:
  python -m scripts.analyze_state_tagged --config=config/profiles/garg_weat_dlnews_tagged.yml
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import fire
import pandas as pd

from scripts.analyze_category_bias import METRIC_SPECS, _filter_models, summarize_and_write
from scripts.analyze_garg import discover_models, load_gender_words, load_model_for_unit
from scripts.common.category_summary import load_categories
from scripts.common.config_loader import get_wordlist_dir, load_config
from scripts.common.logging_utils import setup_logging

SEP = "__"


def states_in_vocab(keys: Iterable[str], anchors: Set[str]) -> Set[str]:
    out = set()
    for k in keys:
        base, sep, slug = k.partition(SEP)
        if sep and slug and base in anchors:
            out.add(slug)
    return out


def tagged_gender_words(gender_words: dict, slug: str) -> dict:
    return {side: [f"{w}{SEP}{slug}" for w in gender_words[side]] for side in ("male", "female")}


def analyze_period_model(model, period: int, gender_words: dict,
                         categories: Dict[str, List[str]], metrics: List[str],
                         min_anchors: int, allowed_states: Optional[Set[str]], logger
                         ) -> Tuple[Dict[str, List[pd.DataFrame]], Dict[str, List[str]]]:
    """Per-state long frames for one pooled period model."""
    anchors = set(gender_words["male"]) | set(gender_words["female"])
    vocab = model.key_to_index
    frames = {m: [] for m in metrics}
    units = {m: [] for m in metrics}
    for slug in sorted(states_in_vocab(vocab, anchors)):
        if allowed_states is not None and slug not in allowed_states:
            continue
        tagged = tagged_gender_words(gender_words, slug)
        n_m = sum(w in vocab for w in tagged["male"])
        n_f = sum(w in vocab for w in tagged["female"])
        if min(n_m, n_f) < min_anchors:
            logger.info(f"  {slug}_{period}: skipped (tagged anchors male {n_m}, female {n_f} "
                        f"< {min_anchors})")
            continue
        unit = f"{slug}_{period}"
        for m in metrics:
            long_df = METRIC_SPECS[m][0](model, unit, categories, tagged, logger)
            if long_df is not None:
                frames[m].append(long_df)
                units[m].append(unit)
    return frames, units


def main(config: str) -> None:
    cfg = load_config(config)
    logger = setup_logging(Path(cfg["paths"]["log_dir"]), "analyze_state_tagged.log")
    analysis = cfg.get("analysis", {})
    metrics = analysis.get("metrics", ["rnd"])
    min_anchors = int(analysis.get("min_tagged_anchors", 3))
    wl = cfg.get("wordlists", {})
    gender_words = load_gender_words(
        get_wordlist_dir(cfg) / wl.get("gender_words_file", "gender_words.json"), logger)
    categories = load_categories(cfg, logger)

    arm = cfg.get("_arm") or cfg.get("embedding_source")
    cov = pd.read_csv(Path(cfg["paths"]["results_dir"]) / f"coverage_{arm}.csv")
    min_docs = int(cfg["us_states"].get("min_documents", 500))
    ok = cov[cov["n_docs"] >= min_docs]

    models = _filter_models(discover_models(cfg), None, analysis.get("decade_range"), logger)
    if not models:
        raise SystemExit(f"No pooled period models in {cfg['paths']['models_dir']}")
    collected = {m: ([], []) for m in metrics}
    for path, unit in models:
        period = int(unit.rsplit("_", 1)[1])
        allowed = set(ok[ok["year"] == period]["state"])
        model = load_model_for_unit(path, cfg)
        logger.info(f"{unit}: vocab {len(model.key_to_index)}, "
                    f"{len(allowed)} states with >= {min_docs} documents")
        frames, units = analyze_period_model(model, period, gender_words, categories,
                                             metrics, min_anchors, allowed, logger)
        for m in metrics:
            collected[m][0].extend(frames[m])
            collected[m][1].extend(units[m])
        logger.info(f"{unit}: scored {len(units[metrics[0]])} states")
        del model
    summarize_and_write(collected, metrics, categories, cfg, logger)


if __name__ == "__main__":
    fire.Fire(main)
