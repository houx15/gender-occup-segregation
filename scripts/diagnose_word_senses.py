#!/usr/bin/env python3
"""Nearest neighbours of wordlist entries in the largest units — a sense check.

A list word can be dominated by a non-occupational sense in news text (the
surname Weaver, a TV pilot). Its nearest neighbours show which sense the
model learned: first names / surnames point to a name, trade words to the job.

Units are the ``n_units`` largest analyzed units by article count
(coverage_<arm>.csv), restricted to analysis.decade_range.

Writes <results_dir>/word_senses.csv (word, unit, rank, neighbor, similarity).

Usage:
  python -m scripts.diagnose_word_senses --config=config/profiles/garg_weat_dlnews.yml \
      --words=weaver,carpenter,painter --n_units=5 --topn=12
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import fire
import pandas as pd

from scripts.analyze_category_bias import _filter_models
from scripts.analyze_garg import discover_models, load_model_for_unit
from scripts.common.config_loader import load_config
from scripts.common.logging_utils import setup_logging
from scripts.common.metrics import entry_vector


def nearest_neighbors(model, entry: str, topn: int = 12) -> List[Tuple[str, float]]:
    """Top-``topn`` neighbours of an entry ('a|b' pooled), excluding its own forms."""
    vec = entry_vector(model, entry)
    if vec is None:
        return []
    forms = {f.strip() for f in entry.split("|")}
    hits = model.similar_by_vector(vec, topn=topn + len(forms))
    return [(w, float(s)) for w, s in hits if w not in forms][:topn]


def main(config: str, words: str, n_units: int = 5, topn: int = 12) -> None:
    cfg = load_config(config)
    logger = setup_logging(Path(cfg["paths"]["log_dir"]), "diagnose_word_senses.log")
    entries = [w.strip() for w in (words.split(",") if isinstance(words, str) else words)]
    results = Path(cfg["paths"]["results_dir"])
    arm = cfg.get("_arm") or cfg.get("embedding_source")
    cov = pd.read_csv(results / f"coverage_{arm}.csv")

    models = _filter_models(discover_models(cfg), None,
                            cfg.get("analysis", {}).get("decade_range"), logger)
    by_unit = dict((u, p) for p, u in models)
    largest = [u for u in cov.sort_values("n_docs", ascending=False)["unit_name"]
               if u in by_unit][:n_units]

    rows = []
    for unit in largest:
        model = load_model_for_unit(by_unit[unit], cfg)
        for entry in entries:
            for rank, (w, s) in enumerate(nearest_neighbors(model, entry, topn), 1):
                rows.append({"word": entry, "unit": unit, "rank": rank,
                             "neighbor": w, "similarity": round(s, 3)})
        del model
    out = pd.DataFrame(rows)
    out.to_csv(results / "word_senses.csv", index=False)
    for (entry, unit), g in out.groupby(["word", "unit"], sort=False):
        print(f"{entry:>14} @ {unit:<22} {' '.join(g['neighbor'])}")


if __name__ == "__main__":
    fire.Fire(main)
