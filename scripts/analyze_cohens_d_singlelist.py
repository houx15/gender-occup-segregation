#!/usr/bin/env python3
"""Single-list Cohen's d producer: cosine projection onto the gender axis.

Robustness companion to Garg's RND. Same single-wordlist-per-category design,
different measurement: build the female−male gender axis, then project each
occupation (L2-normalized, as Garg does) onto it. The per-word value is the
cosine to the axis; sign matches RND (positive = female-leaning).

Distinct from the two-wordlist WEAT Cohen's d in analyze_weat.py (hence the
"singlelist" name): here each category is ONE occupation list, not a contrast
between two target sets.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from scripts.common.metrics import l2_normalize
from scripts.common.embedding_utils import construct_semantic_axis, compute_projection


def projection_values(
    model,
    unit_name: str,
    categories: Dict[str, List[str]],
    gender_words: dict,
    logger,
) -> Optional[pd.DataFrame]:
    """Per-unit cosine projection across all categories, on a LOADED model.

    Returns a long DataFrame (unit_name, category, occupation, value, in_vocab)
    or None if the gender axis is unobtainable. ``value`` is the projection of
    the L2-normalized occupation vector onto the unit-norm female−male axis,
    i.e. the cosine similarity to the axis.
    """
    axis, n_pos, n_neg = construct_semantic_axis(
        gender_words["female"], gender_words["male"], model
    )
    logger.info(
        f"  {unit_name}: gender axis — "
        f"female {n_pos}/{len(gender_words['female'])} found, "
        f"male {n_neg}/{len(gender_words['male'])} found"
    )
    if axis is None:
        logger.warning(
            f"  {unit_name}: skipping projection — gender axis unobtainable "
            f"(female={n_pos}, male={n_neg})"
        )
        return None

    rows: List[dict] = []
    for cat_name, words in categories.items():
        n_in = 0
        for w in words:
            if w in model.key_to_index:
                vec = l2_normalize(model[w])
                projection, _cosine = compute_projection(vec, axis)
                rows.append({
                    "unit_name": unit_name, "category": cat_name, "occupation": w,
                    "value": float(projection), "in_vocab": True,
                })
                n_in += 1
            else:
                rows.append({
                    "unit_name": unit_name, "category": cat_name, "occupation": w,
                    "value": np.nan, "in_vocab": False,
                })
        logger.info(f"    {cat_name}: {n_in}/{len(words)} in vocab")
    return pd.DataFrame(rows)
