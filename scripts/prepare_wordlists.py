#!/usr/bin/env python3
"""
Wordlist precheck for the analyze_garg_weat experiment.

Turns the raw, manually-curated ``candidates_<category>.txt`` lists into the
``cleaned_<category>.txt`` lists the pipeline actually consumes. Two passes:

  1. Dedup (per file): keep only the FIRST occurrence of each word within a
     category file. A word may legitimately appear in two different categories
     (e.g. "logical" in both leadership and science) — cross-category
     duplicates are left alone.
  2. OOV prune: probe every word against the embedding vocabularies and drop
     any word whose coverage is below ``--threshold`` (default 0.8). Coverage
     is the fraction of models the word is in-vocab for, pooled across every
     ``(config, model)`` cell the run sees. With 11 models, a word in vocab
     for 9 of them scores 9/11 = 0.818 and survives.

The raw ``candidates_*.txt`` files are NEVER modified. Outputs:

  - ``cleaned_<category>.txt`` (one per category) — the pruned, deduped list
    the garg_weat configs point at.
  - ``<out_dir>/wordlist_precheck_report.csv`` — every word with its coverage
    and disposition (kept / dropped_dup / dropped_oov), for manual review
    before running the full pipeline.

Reuses discover_models + load_model_for_unit + decade_to_census_year from
analyze_garg so model discovery and format dispatch (gensim_kv vs histwords)
stay identical to the real analyzer.

Usage (on the cluster, where the embeddings live):
    python -m scripts.prepare_wordlists \
        --configs=config/profiles/garg_weat_coha_trained.yml,config/profiles/garg_weat_coha_histwords_sgns.yml,config/profiles/garg_weat_coha_histwords_svd.yml \
        --wordlist_dir=wordlists/en/garg_weat \
        --candidates=candidates_leadership.txt,candidates_family.txt,candidates_science.txt \
        --threshold=0.8
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import fire

from scripts.common.config_loader import load_config
from scripts.common.logging_utils import setup_logging
from scripts.analyze_garg import (
    discover_models, load_model_for_unit, decade_to_census_year,
)


# --------------------------------------------------------------------------
# Pure helpers (unit-tested without any embeddings)
# --------------------------------------------------------------------------

def category_from_filename(fname: str) -> str:
    """'candidates_leadership.txt' -> 'leadership'."""
    stem = Path(fname).stem
    if stem.startswith("candidates_"):
        return stem[len("candidates_"):]
    if stem.startswith("cleaned_"):
        return stem[len("cleaned_"):]
    return stem


def dedup_within_files(
    categories: Dict[str, List[str]]
) -> Tuple[Dict[str, List[str]], List[dict]]:
    """Per-file dedup: keep the first occurrence of each word within each
    category. Cross-category duplicates are preserved (a word may belong to
    two buckets).

    Returns (deduped_categories, dropped_rows) where each dropped row is
    {category, word, reason='duplicate_in_file'}.
    """
    deduped: Dict[str, List[str]] = {}
    dropped: List[dict] = []
    for cat, words in categories.items():
        seen = set()
        kept: List[str] = []
        for w in words:
            if w in seen:
                dropped.append(
                    {"category": cat, "word": w, "reason": "duplicate_in_file"}
                )
                continue
            seen.add(w)
            kept.append(w)
        deduped[cat] = kept
    return deduped, dropped


def prune_by_coverage(
    categories: Dict[str, List[str]],
    coverage: Dict[Tuple[str, str], float],
    threshold: float,
) -> Tuple[Dict[str, List[str]], List[dict]]:
    """Drop words whose pooled coverage is below ``threshold``.

    ``coverage`` maps (category, word) -> fraction in [0, 1]. A word missing
    from ``coverage`` is treated as 0.0 (probed but never in vocab).

    Returns (kept_categories, pruned_rows) where each pruned row is
    {category, word, coverage, reason='below_threshold'}.
    """
    kept: Dict[str, List[str]] = {}
    pruned: List[dict] = []
    for cat, words in categories.items():
        survivors: List[str] = []
        for w in words:
            cov = coverage.get((cat, w), 0.0)
            if cov >= threshold:
                survivors.append(w)
            else:
                pruned.append({
                    "category": cat, "word": w,
                    "coverage": cov, "reason": "below_threshold",
                })
        kept[cat] = survivors
    return kept, pruned


# --------------------------------------------------------------------------
# Coverage probe (needs the embeddings)
# --------------------------------------------------------------------------

def probe_coverage(
    config_paths: List[str],
    categories: Dict[str, List[str]],
    logger,
) -> Tuple[Dict[Tuple[str, str], float], int]:
    """Probe every word against every model, pooled across all configs.

    Returns (coverage, n_models) where coverage maps (category, word) ->
    fraction of models the word is in-vocab for.
    """
    in_vocab_counts: Dict[Tuple[str, str], int] = {
        (cat, w): 0 for cat, words in categories.items() for w in words
    }
    n_models = 0

    for cfg_path in config_paths:
        cfg = load_config(cfg_path)
        source = cfg.get("embedding_source", "unknown")
        models = discover_models(cfg)

        # Mirror analyze_garg_weat's decade_range clip so we probe inside the
        # same comparable window the analyzer will use.
        decade_range = cfg.get("analysis", {}).get("decade_range")
        if decade_range:
            start, end = int(decade_range[0]), int(decade_range[1])
            kept = []
            for path, unit in models:
                year = decade_to_census_year(unit)
                if year is None or start <= year <= end:
                    kept.append((path, unit))
            logger.info(
                f"[{source}] decade_range [{start}, {end}]: "
                f"{len(models)} -> {len(kept)} models"
            )
            models = kept

        logger.info(f"[{source}] probing {len(models)} models")
        for model_path, unit_name in models:
            model = load_model_for_unit(model_path, cfg)
            n_models += 1
            vocab = model.key_to_index
            for cat, words in categories.items():
                n_in = 0
                for w in words:
                    if w in vocab:
                        in_vocab_counts[(cat, w)] += 1
                        n_in += 1
                logger.info(
                    f"  [{source}/{unit_name}] {cat}: {n_in}/{len(words)} in vocab"
                )
            del model

    if n_models == 0:
        logger.error("No models discovered across the given configs")
        return {}, 0

    coverage = {
        key: count / n_models for key, count in in_vocab_counts.items()
    }
    return coverage, n_models


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def _load_words(wl_dir: Path, fname: str) -> List[str]:
    with open(wl_dir / fname, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def main(
    configs: str,
    wordlist_dir: str,
    candidates: str,
    threshold: float = 0.8,
    out_prefix: str = "cleaned_",
) -> None:
    """Dedup + OOV-prune candidate wordlists into cleaned lists.

    Args:
        configs: comma-separated config paths whose models define coverage.
        wordlist_dir: directory holding candidates_*.txt (also where
            cleaned_*.txt are written).
        candidates: comma-separated candidate filenames.
        threshold: minimum pooled coverage (fraction of models) to keep a word.
        out_prefix: prefix for the cleaned output files.
    """
    config_paths = [c.strip() for c in configs.split(",") if c.strip()]
    candidate_files = [c.strip() for c in candidates.split(",") if c.strip()]
    wl_dir = Path(wordlist_dir)

    # Logging: park the log next to the first config's log_dir.
    first_cfg = load_config(config_paths[0])
    logger = setup_logging(
        Path(first_cfg["paths"]["log_dir"]), "prepare_wordlists.log"
    )

    logger.info("=" * 72)
    logger.info("Wordlist precheck — dedup + OOV prune")
    logger.info(f"  configs:   {config_paths}")
    logger.info(f"  candidates:{candidate_files}")
    logger.info(f"  threshold: {threshold}")
    logger.info("=" * 72)

    # --- Load raw candidates -------------------------------------------------
    raw: Dict[str, List[str]] = {}
    for fname in candidate_files:
        cat = category_from_filename(fname)
        raw[cat] = _load_words(wl_dir, fname)
        logger.info(f"  {cat}: {len(raw[cat])} raw candidates from {fname}")

    # --- Pass 1: per-file dedup ---------------------------------------------
    deduped, dup_rows = dedup_within_files(raw)
    for cat in deduped:
        logger.info(
            f"  {cat}: {len(raw[cat])} -> {len(deduped[cat])} after dedup "
            f"({len(raw[cat]) - len(deduped[cat])} duplicates dropped)"
        )

    # --- Pass 2: OOV coverage prune -----------------------------------------
    coverage, n_models = probe_coverage(config_paths, deduped, logger)
    if n_models == 0:
        logger.error("Aborting: no models to probe against.")
        return

    kept, oov_rows = prune_by_coverage(deduped, coverage, threshold)

    logger.info("-" * 72)
    logger.info(f"Pooled over {n_models} models; threshold={threshold}")
    for cat in kept:
        logger.info(
            f"  {cat}: {len(deduped[cat])} -> {len(kept[cat])} after OOV prune "
            f"({len(deduped[cat]) - len(kept[cat])} below {threshold})"
        )

    # --- Write cleaned lists -------------------------------------------------
    for cat, words in kept.items():
        out_path = wl_dir / f"{out_prefix}{cat}.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(words) + ("\n" if words else ""))
        logger.info(f"  wrote {out_path} ({len(words)} words)")

    # --- Report CSV for manual review ---------------------------------------
    report_rows: List[dict] = []
    dup_set = {(r["category"], r["word"]) for r in dup_rows}
    pruned_set = {(r["category"], r["word"]) for r in oov_rows}
    for cat, words in raw.items():
        for w in words:
            if (cat, w) in dup_set:
                disposition = "dropped_dup"
            elif (cat, w) in pruned_set:
                disposition = "dropped_oov"
            else:
                disposition = "kept"
            report_rows.append({
                "category": cat,
                "word": w,
                "coverage": coverage.get((cat, w), float("nan")),
                "n_models": n_models,
                "disposition": disposition,
            })
    report_df = pd.DataFrame(report_rows)

    results_dir = Path(first_cfg["paths"].get("results_dir", "."))
    results_dir.mkdir(parents=True, exist_ok=True)
    report_path = results_dir / "wordlist_precheck_report.csv"
    report_df.to_csv(report_path, index=False)
    logger.info(f"  wrote report {report_path}")

    # Echo the dropped-by-OOV words (ascending coverage) so the job log shows
    # exactly what was cut and how close it was to the bar.
    print(f"\nDropped by OOV prune (coverage < {threshold}, pooled over {n_models} models):")
    dropped_df = report_df[report_df["disposition"] == "dropped_oov"].sort_values(
        ["category", "coverage"]
    )
    if dropped_df.empty:
        print("  (none)")
    else:
        print(dropped_df[["category", "word", "coverage"]].to_string(index=False))

    logger.info("=" * 72)
    logger.info("Wordlist precheck complete. Review cleaned_*.txt, then run "
                "the full garg_weat pipeline.")
    logger.info("=" * 72)


if __name__ == "__main__":
    fire.Fire(main)
