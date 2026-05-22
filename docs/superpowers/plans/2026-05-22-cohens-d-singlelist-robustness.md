# Single-list Cohen's d Robustness Check — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a config-selectable "single-list Cohen's d" metric (mean cosine projection of Garg's one-wordlist-per-category occupations onto the female−male gender axis) alongside the existing RND, plus a proportion-male-leaned statistic for both metrics, each with bootstrap + subsample bands, computed in one pass per model and visualized by reusing the existing plot code.

**Architecture:** Extract the metric-agnostic summary machinery out of `analyze_garg_weat.py` into a shared `scripts/common/category_summary.py`. Each metric becomes a small per-word value producer taking an already-loaded model. A new orchestrator `scripts/analyze_category_bias.py` reads `analysis.metrics: [rnd, cohens_d]`, loads each model once, runs the listed producers, and writes per-metric `*_long` + `*_summary_by_category` parquets. Existing RND outputs keep their column names via legacy aliases.

**Tech Stack:** Python, numpy, pandas, gensim KeyedVectors, fire CLI, pytest. Spec: `docs/superpowers/specs/2026-05-22-cohens-d-singlelist-robustness-design.md`.

---

## File Structure

- **Modify** `scripts/common/metrics.py` — generalize `bootstrap_ci` to accept a `statistic` callable; add `proportion_below`.
- **Create** `scripts/common/category_summary.py` — shared `load_categories`, `compute_consistent_set`, `subsample_bands_from_lookup` (generalized), `build_summary` (mean + proportion, both bands, optional legacy RND aliases).
- **Modify** `scripts/analyze_garg_weat.py` — extract `rnd_values(model, ...)`; re-export shared helpers (back-compat for tests); `main` delegates to the orchestrator with `metrics=["rnd"]`.
- **Create** `scripts/analyze_cohens_d_singlelist.py` — `projection_values(model, ...)` producer.
- **Create** `scripts/analyze_category_bias.py` — orchestrator (`run` + `main` CLI).
- **Modify** `scripts/common/config_loader.py` — validate `analysis.metrics` entries.
- **Modify** the 9 `config/profiles/garg_weat_*.yml` — add `analysis.metrics: [rnd, cohens_d]`.
- **Modify** `scripts/visualize.py` — parameterize provincial plots by value column; plot the cohens_d summary and the proportion trend when present.
- **Create/Modify** tests: `tests/test_metrics.py`, `tests/test_category_summary.py` (new), `tests/test_analyze_cohens_d_singlelist.py` (new), `tests/test_analyze_category_bias.py` (new), `tests/test_analyze_garg_weat.py` (unchanged — must stay green).

Convention notes (match existing code):
- Tests install a fake `gensim` and a `StubKV` (see `tests/test_analyze_garg_weat.py:28-101`); reuse that exact pattern.
- Run tests with `python -m pytest`. The repo treats `scripts` as a package (imports like `from scripts.common.metrics import ...`).

---

## Task 1: Generalize `bootstrap_ci` and add `proportion_below`

**Files:**
- Modify: `scripts/common/metrics.py:46-64`
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_metrics.py`:

```python
from scripts.common.metrics import proportion_below


class TestProportionBelow:
    def test_counts_strictly_below_threshold(self):
        vals = np.array([-2.0, -0.5, 0.0, 1.0])
        # < 0 → two of four
        assert proportion_below(vals, 0.0) == pytest.approx(0.5)

    def test_all_below(self):
        assert proportion_below(np.array([-1.0, -2.0]), 0.0) == pytest.approx(1.0)

    def test_none_below(self):
        assert proportion_below(np.array([1.0, 2.0]), 0.0) == pytest.approx(0.0)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            proportion_below(np.array([]), 0.0)


class TestBootstrapCIStatistic:
    def test_default_statistic_is_mean(self):
        rng = np.random.default_rng(5)
        values = rng.standard_normal(100)
        a = bootstrap_ci(values, n_iter=300, ci=0.9, seed=11)
        b = bootstrap_ci(values, n_iter=300, ci=0.9, seed=11, statistic=np.mean)
        assert a == b

    def test_proportion_statistic_in_unit_interval(self):
        rng = np.random.default_rng(0)
        values = rng.standard_normal(200)  # ~half below 0
        lo, hi = bootstrap_ci(
            values, n_iter=500, ci=0.95, seed=42,
            statistic=lambda x: proportion_below(x, 0.0),
        )
        assert 0.0 <= lo <= hi <= 1.0
        assert 0.3 < (lo + hi) / 2 < 0.7
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_metrics.py::TestProportionBelow -v`
Expected: FAIL — `cannot import name 'proportion_below'`.

- [ ] **Step 3: Implement**

In `scripts/common/metrics.py`, add after `relative_norm_distance` (before `bootstrap_ci`):

```python
def proportion_below(values: np.ndarray, threshold: float = 0.0) -> float:
    """Fraction of values strictly less than ``threshold``.

    With the project's sign convention (positive = female-leaning), a
    threshold of 0.0 yields the share of occupations that are male-leaning.
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        raise ValueError("proportion_below: values must be non-empty")
    return float(np.mean(values < threshold))
```

Replace `bootstrap_ci` (lines 46-64) with a `statistic`-aware version:

```python
def bootstrap_ci(
    values: np.ndarray,
    n_iter: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
    statistic=None,
) -> tuple[float, float]:
    """Percentile bootstrap CI for a statistic of ``values``.

    ``statistic`` is a callable mapping a 1-D resample to a scalar; it
    defaults to the mean (computed vectorized for speed). Pass e.g.
    ``lambda x: proportion_below(x, 0.0)`` for the male-leaned proportion.
    """
    values = np.asarray(values)
    if values.size == 0:
        raise ValueError("bootstrap_ci: values must be non-empty")
    rng = np.random.default_rng(seed)
    n = values.shape[0]
    idx = rng.integers(0, n, size=(n_iter, n))
    if statistic is None or statistic is np.mean:
        stats = values[idx].mean(axis=1)
    else:
        stats = np.array([float(statistic(values[row])) for row in idx])
    alpha = (1.0 - ci) / 2.0
    lo = float(np.percentile(stats, 100.0 * alpha))
    hi = float(np.percentile(stats, 100.0 * (1.0 - alpha)))
    if lo > hi:  # numerical guard; percentile should already enforce this
        lo, hi = hi, lo
    return lo, hi
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_metrics.py -v`
Expected: PASS (all existing + new).

- [ ] **Step 5: Commit**

```bash
git add scripts/common/metrics.py tests/test_metrics.py
git commit -m "feat(metrics): statistic-aware bootstrap_ci + proportion_below"
```

---

## Task 2: Create shared `category_summary.py`

Move the metric-agnostic helpers out of `analyze_garg_weat.py`, generalized to any value column and to compute both the mean and the proportion-male-leaned statistic with both uncertainty bands.

**Files:**
- Create: `scripts/common/category_summary.py`
- Test: `tests/test_category_summary.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_category_summary.py`:

```python
"""Tests for scripts.common.category_summary (metric-agnostic summaries)."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from scripts.common.category_summary import (
    build_summary, subsample_bands_from_lookup,
)

logger = logging.getLogger("test")


def _long(value_col="value"):
    """Two units × one category 'lead' × three occupations, all in vocab.
    1990s values: [-1, -2, 3] (2/3 male-leaning); 2000s: [1, 2, -3] (1/3)."""
    rows = []
    data = {
        "1990s": {"a": -1.0, "b": -2.0, "c": 3.0},
        "2000s": {"a": 1.0, "b": 2.0, "c": -3.0},
    }
    for unit, occs in data.items():
        for occ, val in occs.items():
            rows.append({
                "unit_name": unit, "category": "lead", "occupation": occ,
                value_col: val, "in_vocab": True,
            })
    return pd.DataFrame(rows)


def test_subsample_band_default_mean_statistic():
    occs = ["a", "b", "c", "d", "e"]
    lookup = {("u", "lead", w): float(i) for i, w in enumerate(occs)}  # 0..4
    bands = subsample_bands_from_lookup(
        lookup, ["u"], {"lead": occs},
        fraction=0.8, n_rounds=200, ci=0.95, seed=42,
    )
    lo, hi, mean = bands[("u", "lead")]
    assert lo < hi
    assert lo <= mean <= hi


def test_subsample_band_proportion_statistic():
    from scripts.common.metrics import proportion_below
    occs = ["a", "b", "c", "d", "e"]
    # three negative, two positive → proportion below 0 around 0.6
    lookup = {
        ("u", "lead", "a"): -1.0, ("u", "lead", "b"): -2.0,
        ("u", "lead", "c"): -3.0, ("u", "lead", "d"): 1.0,
        ("u", "lead", "e"): 2.0,
    }
    bands = subsample_bands_from_lookup(
        lookup, ["u"], {"lead": occs},
        fraction=0.8, n_rounds=200, ci=0.95, seed=7,
        statistic=lambda x: proportion_below(x, 0.0),
    )
    lo, hi, mean = bands[("u", "lead")]
    assert 0.0 <= lo <= mean <= hi <= 1.0


def test_build_summary_has_mean_and_proportion_columns():
    long_df = _long()
    consistent = {"lead": ["a", "b", "c"]}
    summary = build_summary(
        long_df, ["1990s", "2000s"], consistent, logger,
        value_col="value", boot_n_iter=200, boot_ci=0.68,
        sub_fraction=0.8, sub_rounds=50, sub_ci=0.95, seed=42,
    )
    assert {
        "mean_value", "mean_ci_low", "mean_ci_high",
        "mean_sub_low", "mean_sub_high", "mean_sub_mean",
        "prop_male", "prop_ci_low", "prop_ci_high",
        "prop_sub_low", "prop_sub_high", "prop_sub_mean",
        "n_occupations", "n_consistent",
    } <= set(summary.columns)
    # 1990s: 2 of 3 male-leaning; 2000s: 1 of 3
    p90 = summary[(summary.unit_name == "1990s")]["prop_male"].iloc[0]
    p00 = summary[(summary.unit_name == "2000s")]["prop_male"].iloc[0]
    assert p90 == 2 / 3
    assert p00 == 1 / 3


def test_build_summary_legacy_rnd_aliases():
    long_df = _long()
    consistent = {"lead": ["a", "b", "c"]}
    summary = build_summary(
        long_df, ["1990s", "2000s"], consistent, logger,
        value_col="value", legacy_rnd_aliases=True,
        boot_n_iter=100, boot_ci=0.68, sub_fraction=0.8,
        sub_rounds=50, sub_ci=0.95, seed=42,
    )
    assert {"mean_rnd", "ci_low", "ci_high", "sub_low", "sub_high", "sub_mean"} <= set(summary.columns)
    # Aliases equal their generic counterparts row-by-row.
    assert (summary["mean_rnd"] == summary["mean_value"]).all()
    assert (summary["ci_low"] == summary["mean_ci_low"]).all()
    assert (summary["sub_mean"] == summary["mean_sub_mean"]).all()
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_category_summary.py -v`
Expected: FAIL — `No module named 'scripts.common.category_summary'`.

- [ ] **Step 3: Implement**

Create `scripts/common/category_summary.py`. The `load_categories`, `compute_consistent_set`, and the subsample structure are moved from `analyze_garg_weat.py` (lines 49-77, 152-175, 178-232) and generalized.

```python
"""Metric-agnostic per-category summaries for single-wordlist bias analyses.

Shared by analyze_garg_weat (RND) and analyze_cohens_d_singlelist (cosine
projection). The only thing that varies between metrics is the per-word value;
everything here (categories, consistent set, bootstrap + subsample bands, the
mean and proportion-male-leaned statistics) operates on a generic value column.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

from scripts.common.config_loader import get_wordlist_dir
from scripts.common.metrics import bootstrap_ci, proportion_below


def _mean(x: np.ndarray) -> float:
    return float(np.mean(x))


def _prop_male(x: np.ndarray) -> float:
    return proportion_below(x, 0.0)


def load_categories(config: dict, logger) -> Dict[str, List[str]]:
    """Read category occupation files from config['wordlists']['categories']."""
    wl_dir = get_wordlist_dir(config)
    cats_cfg = config.get("wordlists", {}).get("categories")
    if not cats_cfg:
        raise ValueError(
            "config must define wordlists.categories "
            "(mapping category-name -> filename)"
        )
    out: Dict[str, List[str]] = {}
    for cat_name, fname in cats_cfg.items():
        path = wl_dir / fname
        if not path.exists():
            raise FileNotFoundError(f"Category file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            words = [line.strip() for line in f if line.strip()]
        out[cat_name] = words
        logger.info(f"  {cat_name}: loaded {len(words)} candidates from {path.name}")
    return out


def compute_consistent_set(
    long_df: pd.DataFrame,
    categories: Dict[str, List[str]],
    units: List[str],
    logger,
) -> Dict[str, List[str]]:
    """Per category, find occupations in vocab in ALL units."""
    consistent: Dict[str, List[str]] = {}
    for cat_name in categories:
        cat_df = long_df[long_df["category"] == cat_name]
        in_all = set(cat_df.loc[cat_df["in_vocab"], "occupation"].unique())
        for u in units:
            unit_in = set(
                cat_df.loc[(cat_df["unit_name"] == u) & cat_df["in_vocab"], "occupation"]
            )
            in_all &= unit_in
        consistent[cat_name] = sorted(in_all)
        logger.info(
            f"consistent-set {cat_name}: {len(consistent[cat_name])}/"
            f"{len(categories[cat_name])} occupations across {len(units)} units"
        )
    return consistent


def subsample_bands_from_lookup(
    value_lookup: Dict[Tuple[str, str, str], float],
    units: List[str],
    consistent_sets: Dict[str, List[str]],
    fraction: float,
    n_rounds: int,
    ci: float,
    seed: int,
    statistic: Callable[[np.ndarray], float] = _mean,
) -> Dict[Tuple[str, str], Tuple[float, float, float]]:
    """Word-subsample robustness band for an arbitrary per-round ``statistic``.

    Each round keeps ``fraction`` of a category's consistent set (without
    replacement); the SAME subset is held across every unit so the band
    isolates word-choice sensitivity. ``value_lookup`` maps
    (unit, category, occupation) -> value (in-vocab only). Returns
    dict[(unit, category)] -> (low, high, mean) where low/high are the ``ci``
    percentile interval of the n_rounds round-statistics and mean is their mean.
    """
    rng = np.random.default_rng(seed)
    alpha = (1.0 - ci) / 2.0
    out: Dict[Tuple[str, str], Tuple[float, float, float]] = {}

    for cat_name, consistent in consistent_sets.items():
        consistent = list(consistent)
        n = len(consistent)
        if n == 0:
            for u in units:
                out[(u, cat_name)] = (float("nan"), float("nan"), float("nan"))
            continue
        k = max(1, int(round(fraction * n)))

        round_stats: Dict[str, List[float]] = {u: [] for u in units}
        for _ in range(n_rounds):
            subset = consistent if k >= n else rng.choice(consistent, size=k, replace=False)
            for u in units:
                vals = [
                    value_lookup[(u, cat_name, w)]
                    for w in subset
                    if (u, cat_name, w) in value_lookup
                ]
                round_stats[u].append(
                    float(statistic(np.asarray(vals, dtype=float))) if vals else np.nan
                )

        for u in units:
            arr = np.asarray(round_stats[u], dtype=float)
            if np.isnan(arr).all():
                out[(u, cat_name)] = (float("nan"), float("nan"), float("nan"))
                continue
            lo = float(np.nanpercentile(arr, 100.0 * alpha))
            hi = float(np.nanpercentile(arr, 100.0 * (1.0 - alpha)))
            out[(u, cat_name)] = (lo, hi, float(np.nanmean(arr)))
    return out


def build_summary(
    long_df: pd.DataFrame,
    units: List[str],
    consistent_sets: Dict[str, List[str]],
    logger,
    value_col: str = "value",
    boot_n_iter: int = 5000,
    boot_ci: float = 0.68,
    sub_fraction: float = 0.8,
    sub_rounds: int = 100,
    sub_ci: float = 0.95,
    seed: int = 42,
    legacy_rnd_aliases: bool = False,
) -> pd.DataFrame:
    """Per-(unit, category) summary carrying TWO statistics, each with TWO bands.

    Statistics: ``mean_value`` (category mean of the metric) and ``prop_male``
    (share of occupations with value < 0). Each carries a with-replacement
    bootstrap CI and a word-subsample band.
    """
    in_vocab = long_df[long_df["in_vocab"]]
    value_lookup: Dict[Tuple[str, str, str], float] = {
        (r.unit_name, r.category, r.occupation): float(getattr(r, value_col))
        for r in in_vocab.itertuples(index=False)
    }
    mean_bands = subsample_bands_from_lookup(
        value_lookup, units, consistent_sets,
        fraction=sub_fraction, n_rounds=sub_rounds, ci=sub_ci, seed=seed,
        statistic=_mean,
    )
    prop_bands = subsample_bands_from_lookup(
        value_lookup, units, consistent_sets,
        fraction=sub_fraction, n_rounds=sub_rounds, ci=sub_ci, seed=seed,
        statistic=_prop_male,
    )

    rows: List[dict] = []
    for u in units:
        unit_long = long_df[long_df["unit_name"] == u]
        for cat_name, consistent in consistent_sets.items():
            m_lo, m_hi, m_mean = mean_bands.get((u, cat_name), (np.nan, np.nan, np.nan))
            p_lo, p_hi, p_mean = prop_bands.get((u, cat_name), (np.nan, np.nan, np.nan))
            sub = unit_long[
                (unit_long["category"] == cat_name)
                & unit_long["occupation"].isin(consistent)
                & unit_long["in_vocab"]
            ]
            if sub.empty:
                rows.append({
                    "unit_name": u, "category": cat_name,
                    "mean_value": np.nan, "mean_ci_low": np.nan, "mean_ci_high": np.nan,
                    "mean_sub_low": m_lo, "mean_sub_high": m_hi, "mean_sub_mean": m_mean,
                    "prop_male": np.nan, "prop_ci_low": np.nan, "prop_ci_high": np.nan,
                    "prop_sub_low": p_lo, "prop_sub_high": p_hi, "prop_sub_mean": p_mean,
                    "n_occupations": 0, "n_consistent": len(consistent),
                })
                continue
            arr = sub[value_col].to_numpy(dtype=float)
            mean_ci_low, mean_ci_high = bootstrap_ci(
                arr, n_iter=boot_n_iter, ci=boot_ci, seed=seed
            )
            prop_ci_low, prop_ci_high = bootstrap_ci(
                arr, n_iter=boot_n_iter, ci=boot_ci, seed=seed,
                statistic=_prop_male,
            )
            rows.append({
                "unit_name": u, "category": cat_name,
                "mean_value": float(arr.mean()),
                "mean_ci_low": float(mean_ci_low), "mean_ci_high": float(mean_ci_high),
                "mean_sub_low": m_lo, "mean_sub_high": m_hi, "mean_sub_mean": m_mean,
                "prop_male": _prop_male(arr),
                "prop_ci_low": float(prop_ci_low), "prop_ci_high": float(prop_ci_high),
                "prop_sub_low": p_lo, "prop_sub_high": p_hi, "prop_sub_mean": p_mean,
                "n_occupations": int(arr.size),
                "n_consistent": len(consistent),
            })

    summary_df = pd.DataFrame(rows)

    if legacy_rnd_aliases and not summary_df.empty:
        summary_df["mean_rnd"] = summary_df["mean_value"]
        summary_df["ci_low"] = summary_df["mean_ci_low"]
        summary_df["ci_high"] = summary_df["mean_ci_high"]
        summary_df["sub_low"] = summary_df["mean_sub_low"]
        summary_df["sub_high"] = summary_df["mean_sub_high"]
        summary_df["sub_mean"] = summary_df["mean_sub_mean"]

    n_rows = len(summary_df)
    n_valid = int(summary_df["mean_value"].notna().sum()) if n_rows else 0
    if n_rows > 0 and n_valid == 0:
        logger.error(
            f"build_summary: ALL {n_rows} (unit, category) rows have "
            "mean_value=NaN — downstream plot will be empty. Likely causes: "
            "(a) consistent-set empty for every category, (b) gender "
            "representation couldn't be built. Scroll up for per-unit diagnostics."
        )
    return summary_df
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_category_summary.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/common/category_summary.py tests/test_category_summary.py
git commit -m "feat: shared metric-agnostic category_summary (mean + male-leaned proportion)"
```

---

## Task 3: Refactor `analyze_garg_weat.py` onto the shared module

Extract the RND per-word producer to take an already-loaded model, re-export the shared helpers (so `tests/test_analyze_garg_weat.py` keeps passing), and slim `main` to delegate to the orchestrator. **The existing `tests/test_analyze_garg_weat.py` must stay green unchanged.**

**Files:**
- Modify: `scripts/analyze_garg_weat.py`
- Test (existing, must pass): `tests/test_analyze_garg_weat.py`

- [ ] **Step 1: Replace the module body**

Rewrite `scripts/analyze_garg_weat.py` to:

```python
#!/usr/bin/env python3
"""Garg-WEAT mode: per-category RND producer + thin CLI.

The RND metric (||v - c_male|| - ||v - c_female||, Garg sign convention,
L2-normalized vectors) over one occupation list per category. The summary
machinery now lives in scripts.common.category_summary and is shared with the
single-list Cohen's d analysis; this module keeps the RND-specific per-word
producer and re-exports the shared helpers for backward compatibility.

Outputs (written by the orchestrator, names unchanged):
  garg_weat_rnd_long.parquet            cols: unit_name, category, occupation, rnd, in_vocab
  garg_weat_summary_by_category.parquet mean_rnd/ci_*/sub_* (+ new prop_* columns)

Usage:
  python -m scripts.analyze_garg_weat --config=config/profiles/garg_weat_coha_trained.yml
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import fire

from scripts.common.config_loader import load_config
from scripts.common.metrics import relative_norm_distance, l2_normalize
# Re-export shared helpers so existing imports/tests keep working.
from scripts.common.category_summary import (  # noqa: F401
    load_categories, compute_consistent_set,
    subsample_bands_from_lookup, build_summary,
)


def rnd_values(
    model,
    unit_name: str,
    categories: Dict[str, List[str]],
    gender_words: dict,
    logger,
) -> Optional[pd.DataFrame]:
    """Per-unit RND across all categories, on an ALREADY-LOADED model.

    Returns a long DataFrame (unit_name, category, occupation, value, in_vocab)
    or None if gender centroids are unobtainable. ``value`` is the RND.
    """
    male_vecs = [l2_normalize(model[w]) for w in gender_words["male"] if w in model.key_to_index]
    female_vecs = [l2_normalize(model[w]) for w in gender_words["female"] if w in model.key_to_index]
    logger.info(
        f"  {unit_name}: gender words — "
        f"male {len(male_vecs)}/{len(gender_words['male'])} found, "
        f"female {len(female_vecs)}/{len(gender_words['female'])} found"
    )
    if not male_vecs or not female_vecs:
        logger.warning(
            f"  {unit_name}: skipping RND — gender centroids unobtainable "
            f"(male={len(male_vecs)}, female={len(female_vecs)})"
        )
        return None

    c_male = np.mean(np.asarray(male_vecs), axis=0)
    c_female = np.mean(np.asarray(female_vecs), axis=0)

    rows: List[dict] = []
    for cat_name, words in categories.items():
        n_in = 0
        for w in words:
            if w in model.key_to_index:
                vec = l2_normalize(model[w])
                rows.append({
                    "unit_name": unit_name, "category": cat_name, "occupation": w,
                    "value": float(relative_norm_distance(vec, c_male, c_female)),
                    "in_vocab": True,
                })
                n_in += 1
            else:
                rows.append({
                    "unit_name": unit_name, "category": cat_name, "occupation": w,
                    "value": np.nan, "in_vocab": False,
                })
        logger.info(f"    {cat_name}: {n_in}/{len(words)} in vocab")
    return pd.DataFrame(rows)


def main(config: str = "config/config.yml", unit: Optional[str] = None) -> None:
    """Run the per-category RND analysis (delegates to the orchestrator)."""
    from scripts.analyze_category_bias import run  # local import avoids cycle
    config_data = load_config(config)
    run(config_data, metrics=["rnd"], unit=unit)


if __name__ == "__main__":
    fire.Fire(main)
```

- [ ] **Step 2: Run the existing garg_weat tests (will fail until Task 5 lands)**

Run: `python -m pytest tests/test_analyze_garg_weat.py -v`
Expected: FAIL — `cannot import name 'run' from 'scripts.analyze_category_bias'` (module not created yet). This is expected; Task 5 makes them pass. Note the dependency.

> Implementation order note: Tasks 3, 4, 5 form one unit — `analyze_garg_weat.main` and the orchestrator are mutually referenced. Land them together; run `tests/test_analyze_garg_weat.py` green at the END of Task 5. Commit Task 3's edit now without claiming tests pass.

- [ ] **Step 3: Commit**

```bash
git add scripts/analyze_garg_weat.py
git commit -m "refactor(garg_weat): RND producer on loaded model + shared summary; main delegates"
```

---

## Task 4: `projection_values` producer (single-list Cohen's d)

**Files:**
- Create: `scripts/analyze_cohens_d_singlelist.py`
- Test: `tests/test_analyze_cohens_d_singlelist.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_analyze_cohens_d_singlelist.py`:

```python
"""Tests for projection_values (single-list Cohen's d per-word producer)."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger("test")


class StubKV:
    def __init__(self, vectors):
        self._v = {w: np.asarray(v, dtype=float) for w, v in vectors.items()}
        self.key_to_index = {w: i for i, w in enumerate(self._v)}

    def __getitem__(self, k):
        return self._v[k]

    def __contains__(self, k):
        return k in self._v


# After L2-normalize, c_male=(1,0), c_female=(-1,0); axis female-male = (-1,0).
_MALE = {"he": [1.0, 0.0], "man": [1.0, 0.0]}
_FEMALE = {"she": [-1.0, 0.0], "woman": [-1.0, 0.0]}


def _model():
    return StubKV({
        **_MALE, **_FEMALE,
        "president": [0.6, 0.0], "manager": [0.5, 0.0],   # +x → male-leaning
        "cooking": [-0.5, 0.0], "cleaning": [-0.6, 0.0],  # -x → female-leaning
    })


def _categories():
    return {"leadership": ["president", "manager"], "family": ["cooking", "cleaning"]}


def test_long_schema_and_sign_convention():
    from scripts.analyze_cohens_d_singlelist import projection_values
    gw = {"male": list(_MALE), "female": list(_FEMALE)}
    df = projection_values(_model(), "1990s", _categories(), gw, logger)
    assert set(df.columns) >= {"unit_name", "category", "occupation", "value", "in_vocab"}
    lead = df[df.category == "leadership"]["value"]
    fam = df[df.category == "family"]["value"]
    # axis points male→female (female - male), so male-leaning occupations
    # (+x, near c_male) project NEGATIVE; family (−x) projects POSITIVE.
    assert (lead < 0).all()
    assert (fam > 0).all()


def test_returns_none_when_gender_unobtainable():
    from scripts.analyze_cohens_d_singlelist import projection_values
    model = StubKV({"president": [0.6, 0.0]})  # no gender words
    gw = {"male": list(_MALE), "female": list(_FEMALE)}
    assert projection_values(model, "x", {"leadership": ["president"]}, gw, logger) is None


def test_oov_occupation_marked_not_in_vocab():
    from scripts.analyze_cohens_d_singlelist import projection_values
    gw = {"male": list(_MALE), "female": list(_FEMALE)}
    cats = {"leadership": ["president", "ceo_missing"]}
    df = projection_values(_model(), "1990s", cats, gw, logger)
    missing = df[df.occupation == "ceo_missing"].iloc[0]
    assert missing["in_vocab"] == False  # noqa: E712
    assert np.isnan(missing["value"])
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_analyze_cohens_d_singlelist.py -v`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement**

Create `scripts/analyze_cohens_d_singlelist.py`:

```python
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
```

> Note: because `vec` is L2-normalized, `compute_projection`'s `projection`
> (dot with the unit-norm axis) already equals the cosine to the axis — the
> spec's "cosine to axis" choice. We use `projection` (not the second return
> value, which re-normalizes by the word norm and would here be identical).

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_analyze_cohens_d_singlelist.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/analyze_cohens_d_singlelist.py tests/test_analyze_cohens_d_singlelist.py
git commit -m "feat: single-list Cohen's d projection producer"
```

---

## Task 5: Orchestrator `analyze_category_bias.py`

Reads `analysis.metrics`, loads each model once, runs the listed producers, writes per-metric outputs. Makes both Task 3's `analyze_garg_weat.main` and the new multi-metric path work.

**Files:**
- Create: `scripts/analyze_category_bias.py`
- Test: `tests/test_analyze_category_bias.py`
- Verify green: `tests/test_analyze_garg_weat.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_analyze_category_bias.py` (reuses the fake-gensim + StubKV pattern from `tests/test_analyze_garg_weat.py`):

```python
"""Integration tests for the analyze_category_bias orchestrator."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml


class StubKV:
    def __init__(self, vectors):
        self._v = {w: np.asarray(v, dtype=float) for w, v in vectors.items()}
        self.key_to_index = {w: i for i, w in enumerate(self._v)}

    def __getitem__(self, k):
        return self._v[k]

    def __contains__(self, k):
        return k in self._v


def _install_fake_gensim():
    if "gensim" in sys.modules and not getattr(sys.modules["gensim"], "_fake", False):
        try:
            from gensim.models import KeyedVectors  # noqa: F401
            return
        except Exception:
            pass
    fake = types.ModuleType("gensim")
    fake._fake = True
    models = types.ModuleType("gensim.models")
    models.KeyedVectors = StubKV
    fake.models = models
    sys.modules["gensim"] = fake
    sys.modules["gensim.models"] = models


_install_fake_gensim()

_MALE = {"he": [1.0, 0.0], "man": [1.0, 0.0]}
_FEMALE = {"she": [-1.0, 0.0], "woman": [-1.0, 0.0]}
_CATEGORIES = {"leadership": ["president", "manager"], "family": ["cooking", "cleaning"]}


def _kvs():
    occs_a = {"president": [0.6, 0.0], "manager": [0.5, 0.0],
              "cooking": [-0.5, 0.0], "cleaning": [-0.6, 0.0]}
    occs_b = {"president": [0.5, 0.0], "manager": [0.4, 0.0],
              "cooking": [-0.3, 0.0], "cleaning": [-0.4, 0.0]}
    return {"1990s": StubKV({**_MALE, **_FEMALE, **occs_a}),
            "2000s": StubKV({**_MALE, **_FEMALE, **occs_b})}


def _loader(unit_to_kv):
    def _load(model_path):
        name = Path(str(model_path)).name
        for unit, kv in unit_to_kv.items():
            if unit in name:
                return kv
        raise KeyError(name)
    return _load


def _write_config(tmp_path, metrics):
    base = tmp_path / "proj"
    for sub in ("data/models", "data/results"):
        (base / sub).mkdir(parents=True)
    (base / "logs").mkdir(parents=True)
    wl = tmp_path / "wordlists" / "en" / "garg_weat"
    wl.mkdir(parents=True)
    (wl / "gender_words.json").write_text(
        json.dumps({"male": list(_MALE), "female": list(_FEMALE)}), encoding="utf-8")
    cats_block = {}
    for cat, words in _CATEGORIES.items():
        (wl / f"candidates_{cat}.txt").write_text("\n".join(words) + "\n", encoding="utf-8")
        cats_block[cat] = f"candidates_{cat}.txt"
    cfg = {
        "language": "en", "data_source": "coha", "analysis_mode": "garg_weat",
        "paths": {
            "base_dir": str(base),
            "raw_coha_dir": str(base / "data/raw_coha"),
            "coha_decompressed_dir": str(base / "data/coha_dec"),
            "corpora_dir": str(base / "data/corpora"),
            "models_dir": str(base / "data/models"),
            "results_dir": str(base / "data/results"),
            "log_dir": str(base / "logs"),
        },
        "coha": {"source_archive_urls": []},
        "embedding": {"model_name_template": "coha_{unit_name}.kv"},
        "wordlists": {"dir": str(wl), "gender_words_file": "gender_words.json",
                      "categories": cats_block},
        "analysis": {"metrics": metrics},
    }
    p = tmp_path / "config.yml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return p


def _touch(models_dir, units):
    for u in units:
        (models_dir / f"coha_{u}.kv").touch()


def test_both_metrics_single_pass(tmp_path, monkeypatch):
    cfg = _write_config(tmp_path, ["rnd", "cohens_d"])
    kvs = _kvs()
    rdir = Path(yaml.safe_load(cfg.read_text())["paths"]["results_dir"])
    _touch(Path(yaml.safe_load(cfg.read_text())["paths"]["models_dir"]), list(kvs))

    import scripts.analyze_garg as ag
    monkeypatch.setattr(ag, "load_model", _loader(kvs))

    import scripts.analyze_category_bias as acb
    acb.main(config=str(cfg))

    assert (rdir / "garg_weat_rnd_long.parquet").exists()
    assert (rdir / "garg_weat_summary_by_category.parquet").exists()
    assert (rdir / "cohens_d_singlelist_long.parquet").exists()
    assert (rdir / "cohens_d_singlelist_summary_by_category.parquet").exists()

    # RND long keeps the legacy column name.
    rnd_long = pd.read_parquet(rdir / "garg_weat_rnd_long.parquet")
    assert "rnd" in rnd_long.columns
    proj_long = pd.read_parquet(rdir / "cohens_d_singlelist_long.parquet")
    assert "projection" in proj_long.columns

    # Both summaries carry the proportion statistic.
    for f in ("garg_weat_summary_by_category.parquet",
              "cohens_d_singlelist_summary_by_category.parquet"):
        s = pd.read_parquet(rdir / f)
        assert {"prop_male", "prop_ci_low", "prop_sub_low"} <= set(s.columns)


def test_rnd_only_skips_cohens_d(tmp_path, monkeypatch):
    cfg = _write_config(tmp_path, ["rnd"])
    kvs = _kvs()
    rdir = Path(yaml.safe_load(cfg.read_text())["paths"]["results_dir"])
    _touch(Path(yaml.safe_load(cfg.read_text())["paths"]["models_dir"]), list(kvs))
    import scripts.analyze_garg as ag
    monkeypatch.setattr(ag, "load_model", _loader(kvs))
    import scripts.analyze_category_bias as acb
    acb.main(config=str(cfg))
    assert (rdir / "garg_weat_summary_by_category.parquet").exists()
    assert not (rdir / "cohens_d_singlelist_summary_by_category.parquet").exists()


def test_missing_metrics_key_raises(tmp_path, monkeypatch):
    import yaml as _yaml
    cfg = _write_config(tmp_path, ["rnd"])
    data = _yaml.safe_load(cfg.read_text())
    del data["analysis"]["metrics"]
    cfg.write_text(_yaml.safe_dump(data), encoding="utf-8")
    kvs = _kvs()
    _touch(Path(data["paths"]["models_dir"]), list(kvs))
    import scripts.analyze_garg as ag
    monkeypatch.setattr(ag, "load_model", _loader(kvs))
    import scripts.analyze_category_bias as acb
    import pytest
    with pytest.raises((ValueError, KeyError)):
        acb.main(config=str(cfg))
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_analyze_category_bias.py -v`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement**

Create `scripts/analyze_category_bias.py`:

```python
#!/usr/bin/env python3
"""Orchestrator: compute one or more single-wordlist gender-bias metrics.

Driven by config ``analysis.metrics`` (a list of {"rnd", "cohens_d"}). Each
model is loaded ONCE; every listed metric's per-word producer runs on it, then
the shared summary (mean + male-leaned proportion, each with bootstrap and
subsample bands) is built and written per metric.

Usage:
  python -m scripts.analyze_category_bias --config=config/profiles/garg_weat_coha_trained.yml
  python -m scripts.analyze_category_bias --config=... --metrics=rnd,cohens_d
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import fire

from scripts.common.config_loader import load_config, get_wordlist_dir
from scripts.common.logging_utils import setup_logging
from scripts.common.category_summary import (
    load_categories, compute_consistent_set, build_summary,
)
from scripts.analyze_garg import (
    discover_models, load_model_for_unit, decade_to_census_year, load_gender_words,
)
from scripts.analyze_garg_weat import rnd_values
from scripts.analyze_cohens_d_singlelist import projection_values

# metric -> (producer, long_stem, summary_stem, long_value_name, legacy_aliases)
METRIC_SPECS = {
    "rnd": (rnd_values, "garg_weat_rnd_long",
            "garg_weat_summary_by_category", "rnd", True),
    "cohens_d": (projection_values, "cohens_d_singlelist_long",
                 "cohens_d_singlelist_summary_by_category", "projection", False),
}


def _resolve_metrics(config_data: dict, override: Optional[List[str]]) -> List[str]:
    if override is not None:
        metrics = override
    else:
        metrics = config_data.get("analysis", {}).get("metrics")
    if not metrics:
        raise ValueError(
            "config analysis.metrics is required (a non-empty list of "
            f"{sorted(METRIC_SPECS)}); none found. Prefer an explicit list."
        )
    unknown = [m for m in metrics if m not in METRIC_SPECS]
    if unknown:
        raise ValueError(
            f"Unknown metric(s) {unknown}; valid: {sorted(METRIC_SPECS)}"
        )
    return list(metrics)


def _filter_models(models, unit, decade_range, logger):
    if unit:
        models = [(p, n) for p, n in models if n.startswith(str(unit))]
    if decade_range:
        try:
            start, end = int(decade_range[0]), int(decade_range[1])
        except (TypeError, ValueError, IndexError):
            raise ValueError(
                f"analysis.decade_range must be [start, end] integers, got {decade_range!r}"
            )
        kept = []
        for path, unit_name in models:
            year = decade_to_census_year(unit_name)
            if year is None or start <= year <= end:
                kept.append((path, unit_name))
        logger.info(f"decade_range [{start}, {end}]: {len(models)} -> {len(kept)} models")
        models = kept
    return models


def run(config_data: dict, metrics: List[str], unit: Optional[str] = None) -> None:
    logger = setup_logging(Path(config_data["paths"]["log_dir"]), "analyze_category_bias.log")
    metrics = _resolve_metrics(config_data, metrics)
    logger.info("=" * 80)
    logger.info(f"Single-wordlist bias analysis — metrics={metrics}")
    logger.info("=" * 80)

    wl_dir = get_wordlist_dir(config_data)
    wl_cfg = config_data.get("wordlists", {})
    gender_words = load_gender_words(
        wl_dir / wl_cfg.get("gender_words_file", "gender_words.json"), logger
    )
    categories = load_categories(config_data, logger)

    models = discover_models(config_data)
    decade_range = config_data.get("analysis", {}).get("decade_range")
    models = _filter_models(models, unit, decade_range, logger)
    if not models:
        logger.error("No models found after filtering — nothing written")
        return
    logger.info(f"Found {len(models)} models")

    # metric -> (list of long frames, list of unit names that produced output)
    collected: Dict[str, Tuple[List[pd.DataFrame], List[str]]] = {
        m: ([], []) for m in metrics
    }
    for model_path, unit_name in models:
        model = load_model_for_unit(model_path, config_data)  # loaded ONCE
        for m in metrics:
            producer = METRIC_SPECS[m][0]
            long_df = producer(model, unit_name, categories, gender_words, logger)
            if long_df is None:
                continue
            collected[m][0].append(long_df)
            collected[m][1].append(unit_name)

    analysis_cfg = config_data.get("analysis", {})
    boot = analysis_cfg.get("bootstrap", {})
    sub = analysis_cfg.get("subsample", {})
    seed = int(analysis_cfg.get("seed", 42))
    results_dir = Path(config_data["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    for m in metrics:
        frames, units = collected[m]
        _, long_stem, summary_stem, value_name, legacy = METRIC_SPECS[m]
        if not frames:
            logger.error(f"[{m}] No units produced results — skipping outputs")
            continue
        long_combined = pd.concat(frames, ignore_index=True)
        consistent = compute_consistent_set(long_combined, categories, units, logger)
        summary = build_summary(
            long_combined, units, consistent, logger,
            value_col="value",
            boot_n_iter=int(boot.get("n_iter", 5000)),
            boot_ci=float(boot.get("ci", 0.68)),
            sub_fraction=float(sub.get("fraction", 0.8)),
            sub_rounds=int(sub.get("n_rounds", 100)),
            sub_ci=float(sub.get("ci", 0.95)),
            seed=seed,
            legacy_rnd_aliases=legacy,
        )
        # Friendly value name in the long file (rnd / projection) for back-compat.
        long_out = long_combined.rename(columns={"value": value_name})
        long_path = results_dir / f"{long_stem}.parquet"
        summary_path = results_dir / f"{summary_stem}.parquet"
        long_out.to_parquet(long_path, index=False)
        summary.to_parquet(summary_path, index=False)
        logger.info(f"[{m}] Saved: {long_path}")
        logger.info(f"[{m}] Saved: {summary_path}")

    logger.info("=" * 80)
    logger.info("Single-wordlist bias analysis completed!")
    logger.info("=" * 80)


def main(config: str = "config/config.yml", unit: Optional[str] = None,
         metrics: Optional[str] = None) -> None:
    """CLI. ``metrics`` (comma-separated) overrides config analysis.metrics."""
    config_data = load_config(config)
    override = None
    if metrics is not None:
        override = [m.strip() for m in str(metrics).split(",") if m.strip()]
    run(config_data, metrics=override, unit=unit)


if __name__ == "__main__":
    fire.Fire(main)
```

> `run` is called with `metrics=["rnd"]` by `analyze_garg_weat.main` and with
> `metrics=None` (→ read from config) by this module's own `main`. The
> `_resolve_metrics` `override` path handles both.

- [ ] **Step 4: Run the new + existing garg_weat tests together**

Run: `python -m pytest tests/test_analyze_category_bias.py tests/test_analyze_garg_weat.py -v`
Expected: PASS for both files. (The existing garg_weat tests now flow through the orchestrator and must still see `garg_weat_rnd_long.parquet` with an `rnd` column, `garg_weat_summary_by_category.parquet` with `mean_rnd/ci_*/sub_*` + new `prop_*`, 12 long rows, 6 summary rows, correct signs.)

- [ ] **Step 5: Commit**

```bash
git add scripts/analyze_category_bias.py tests/test_analyze_category_bias.py
git commit -m "feat: single-pass orchestrator dispatching rnd/cohens_d metrics from config"
```

---

## Task 6: Validate `analysis.metrics` in config_loader

**Files:**
- Modify: `scripts/common/config_loader.py:53-82` (inside `_validate_config`)
- Test: `tests/test_config_loader.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_config_loader.py` (match its existing style; it builds minimal valid config dicts and calls `load_config` on a written file, or calls `_validate_config` directly — inspect the file head and reuse whichever helper exists). Minimal direct-call version:

```python
def test_invalid_metrics_entry_rejected():
    import pytest
    from scripts.common.config_loader import _validate_config
    cfg = {
        "language": "en", "data_source": "coha", "analysis_mode": "garg_weat",
        "paths": {
            "base_dir": "b", "models_dir": "m", "results_dir": "r", "log_dir": "l",
            "raw_coha_dir": "rc", "coha_decompressed_dir": "cd", "corpora_dir": "co",
        },
        "coha": {"source_archive_urls": []},
        "analysis": {"metrics": ["rnd", "bogus"]},
    }
    with pytest.raises(ValueError):
        _validate_config(cfg)


def test_valid_metrics_accepted():
    from scripts.common.config_loader import _validate_config
    cfg = {
        "language": "en", "data_source": "coha", "analysis_mode": "garg_weat",
        "paths": {
            "base_dir": "b", "models_dir": "m", "results_dir": "r", "log_dir": "l",
            "raw_coha_dir": "rc", "coha_decompressed_dir": "cd", "corpora_dir": "co",
        },
        "coha": {"source_archive_urls": []},
        "analysis": {"metrics": ["rnd", "cohens_d"]},
    }
    _validate_config(cfg)  # should not raise
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_config_loader.py::test_invalid_metrics_entry_rejected -v`
Expected: FAIL — no validation yet (no error raised).

- [ ] **Step 3: Implement**

In `scripts/common/config_loader.py`, at module scope near the other `VALID_*` constants add:

```python
VALID_BIAS_METRICS = {"rnd", "cohens_d"}
```

Inside `_validate_config`, after the `analysis_mode` block (after line 82), add:

```python
    metrics = config.get("analysis", {}).get("metrics")
    if metrics is not None:
        if not isinstance(metrics, (list, tuple)) or not metrics:
            raise ValueError(
                "analysis.metrics must be a non-empty list, "
                f"got {metrics!r}"
            )
        unknown = [m for m in metrics if m not in VALID_BIAS_METRICS]
        if unknown:
            raise ValueError(
                f"Invalid analysis.metrics entries {unknown}. "
                f"Must be from: {sorted(VALID_BIAS_METRICS)}"
            )
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_config_loader.py -v`
Expected: PASS (existing + new two).

- [ ] **Step 5: Commit**

```bash
git add scripts/common/config_loader.py tests/test_config_loader.py
git commit -m "feat(config): validate analysis.metrics entries"
```

---

## Task 7: Add `metrics: [rnd, cohens_d]` to the 9 garg_weat profiles

**Files (modify):**
- `config/profiles/garg_weat_coha_trained.yml`
- `config/profiles/garg_weat_coha_histwords_sgns.yml`
- `config/profiles/garg_weat_coha_histwords_svd.yml`
- `config/profiles/garg_weat_google_ngram_eng_all.yml`
- `config/profiles/garg_weat_google_ngram_eng_fiction_all.yml`
- `config/profiles/garg_weat_renminribao.yml`
- `config/profiles/garg_weat_china_ngram.yml`
- `config/profiles/garg_weat_provincial_newspaper.yml`
- `config/profiles/garg_weat_weibo.yml`

- [ ] **Step 1: Inspect one profile's `analysis:` block**

Run: `grep -nA8 "^analysis:" config/profiles/garg_weat_coha_trained.yml`
Expected: shows `seed`, `ideation_sign`, `bootstrap`, `subsample` keys.

- [ ] **Step 2: Add the `metrics` key under `analysis:` in each file**

For each profile, add this as the first child of the existing `analysis:` mapping (keep existing keys intact):

```yaml
analysis:
  # Which single-wordlist bias metrics to compute this run.
  #   rnd      = Garg relative norm distance
  #   cohens_d = single-list Cohen's d (mean cosine projection onto gender axis)
  metrics: [rnd, cohens_d]
  # ... existing seed / ideation_sign / bootstrap / subsample unchanged ...
```

Edit all 9 files. Do not change any other key.

- [ ] **Step 3: Verify each file still parses and carries metrics**

Run:
```bash
python -c "import yaml,glob; [print(f, yaml.safe_load(open(f))['analysis']['metrics']) for f in sorted(glob.glob('config/profiles/garg_weat_*.yml'))]"
```
Expected: 9 lines, each printing `['rnd', 'cohens_d']`.

- [ ] **Step 4: Commit**

```bash
git add config/profiles/garg_weat_*.yml
git commit -m "config: enable rnd + cohens_d metrics on all 9 garg_weat profiles"
```

---

## Task 8: Visualization — plot the projection summary and the male-leaned proportion

Reuse the already-parameterized `plot_garg_weat_categories_trend` (it takes `line_col`/`band_cols`); parameterize the provincial plotters by value column; in the `garg_weat` branch of `visualize.main`, after the RND plots, also render the cohens_d summary (if present) and the proportion trend for whatever summaries exist.

**Files:**
- Modify: `scripts/visualize.py` (provincial frame/plotters ~1284-1420; `main` garg_weat branch 2602-2695)
- Test: `tests/test_visualize_garg_weat.py` (existing, must stay green) + new assertions

- [ ] **Step 1: Read the pieces to change**

Run:
```bash
sed -n '1284,1320p;2602,2696p' scripts/visualize.py
```
Confirm `_garg_weat_provincial_frame` hardcodes `mean_rnd` (1293/1298/1310-1312) and the `garg_weat` branch calls the longitudinal trend twice and the three provincial plotters.

- [ ] **Step 2: Write a failing test for proportion + projection trend dispatch**

Append to `tests/test_visualize_garg_weat.py` (reuse its fixture style — it builds a summary DataFrame and a tmp figures dir; mirror the existing test that calls `plot_garg_weat_categories_trend`). Add:

```python
def test_categories_trend_plots_proportion_column(tmp_path):
    import pandas as pd
    import scripts.visualize as viz
    import logging
    df = pd.DataFrame({
        "unit_name": ["1990s", "2000s"] * 3,
        "category": ["leadership"] * 2 + ["family"] * 2 + ["science"] * 2,
        "mean_value": [-0.3, -0.2, 0.3, 0.2, -0.1, -0.05],
        "prop_male": [0.9, 0.7, 0.1, 0.2, 0.6, 0.5],
        "prop_ci_low": [0.8, 0.6, 0.05, 0.1, 0.5, 0.4],
        "prop_ci_high": [0.95, 0.8, 0.2, 0.3, 0.7, 0.6],
        "prop_sub_low": [0.82, 0.62, 0.06, 0.12, 0.52, 0.42],
        "prop_sub_high": [0.93, 0.78, 0.18, 0.28, 0.68, 0.58],
    })
    figdir = tmp_path / "figs"
    figdir.mkdir()
    # line_col=prop_male, no ideation flip, bounded band
    viz.plot_garg_weat_categories_trend(
        df, figdir, logging.getLogger("t"),
        band_cols=("prop_ci_low", "prop_ci_high"), band_tag="proportion_bootstrap",
        line_col="prop_male", category_sign=None,
    )
    assert any(figdir.glob("*proportion_bootstrap*"))
```

Run: `python -m pytest tests/test_visualize_garg_weat.py::test_categories_trend_plots_proportion_column -v`
Expected: FAIL only if `plot_garg_weat_categories_trend` can't accept these args. Inspect its signature (line 421-432) — it already takes `line_col`, `band_cols`, `band_tag`, `category_sign`. If it derives the output filename solely from `band_tag` it should pass; if it hardcodes a `mean_rnd`-only guard (line 458 checks `every {line_col} is NaN`), confirm it uses `line_col`. If the test fails on a hardcoded column, fix in Step 3.

- [ ] **Step 3: Parameterize provincial value column**

Change `_garg_weat_provincial_frame` (line 1284) signature and body to accept `value_col="mean_rnd"`:

```python
def _garg_weat_provincial_frame(summary_df, category_sign=None, value_col="mean_rnd"):
    df = summary_df.copy()
    df = df[df[value_col].notna()]
    # ... existing reversed-category logic unchanged ...
    df = apply_ideation_sign(df, category_sign, [value_col])
    # ... in the groupby/rename, replace "mean_rnd" with value_col:
    grouped = (
        df.groupby(["province", "category"], as_index=False)[value_col]
        .mean()
        .rename(columns={value_col: "value"})
    )
    return grouped, reversed_cats
```

Thread `value_col` through `plot_garg_weat_provincial_rankings/heatmap/choropleth` (add `value_col="mean_rnd"` param, pass to `_garg_weat_provincial_frame`). Existing callers keep the default, so existing tests are unaffected.

- [ ] **Step 4: Add cohens_d + proportion plotting to the garg_weat branch**

In `main` (after the existing RND block that ends ~line 2695, still inside `elif analysis_mode == "garg_weat":`), refactor the longitudinal/provincial plotting into a local helper and call it for each present summary. Replace the existing longitudinal/provincial body with:

```python
        def _plot_category_summary(df, *, value_col, tag, prop=False):
            kind = _garg_weat_unit_kind(df["unit_name"]) if not df.empty else "longitudinal"
            csign = None if prop else category_sign
            if kind == "longitudinal":
                plot_garg_weat_categories_trend(
                    df, figures_dir, logger, embedding_source=embedding_source,
                    band_cols=(f"{value_col}_ci_low" if prop else "ci_low",
                               f"{value_col}_ci_high" if prop else "ci_high"),
                    band_tag=f"{tag}_bootstrap", band_label="bootstrap CI",
                    line_col=value_col, category_sign=csign,
                )
                plot_garg_weat_categories_trend(
                    df, figures_dir, logger, embedding_source=embedding_source,
                    band_cols=(f"{value_col}_sub_low" if prop else "sub_low",
                               f"{value_col}_sub_high" if prop else "sub_high"),
                    band_tag=f"{tag}_subsample", band_label="80% word-subsample band",
                    line_col=value_col, category_sign=csign,
                )
            else:
                shapefile = config_data.get("paths", {}).get("shapefile")
                plot_garg_weat_provincial_rankings(
                    df, figures_dir, logger, category_sign=csign,
                    data_source=ds, value_col=value_col)
                plot_garg_weat_provincial_heatmap(
                    df, figures_dir, logger, category_sign=csign,
                    data_source=ds, value_col=value_col)
                plot_garg_weat_provincial_choropleth(
                    df, figures_dir, logger, category_sign=csign,
                    shapefile=shapefile, value_col=value_col)

        # RND (unchanged outputs) — mean trend + male-leaned proportion.
        _plot_category_summary(df, value_col="mean_rnd", tag="garg_weat")
        if "prop_male" in df.columns:
            _plot_category_summary(df, value_col="prop_male", tag="garg_weat_propmale", prop=True)

        # Single-list Cohen's d, if its summary exists.
        proj_path = results_dir / "cohens_d_singlelist_summary_by_category.parquet"
        if proj_path.exists():
            pdf = pd.read_parquet(proj_path)
            logger.info(f"Loaded {proj_path}: {len(pdf)} rows")
            _plot_category_summary(pdf, value_col="mean_value", tag="cohens_d_singlelist")
            if "prop_male" in pdf.columns:
                _plot_category_summary(
                    pdf, value_col="prop_male", tag="cohens_d_singlelist_propmale", prop=True)
```

> The proportion band columns are `prop_ci_low/high` and `prop_sub_low/high`
> (see Task 2 schema). For `prop_male` the helper builds those names via the
> `prop=True` branch. For `mean_value`/`mean_rnd` it uses the existing
> `ci_low/ci_high/sub_low/sub_high` aliases (present on both summaries).

> Keep the existing survey-correlation extras (scatter/choropleth-grid/trends,
> lines 2650-2693) for the RND provincial case as-is; they are out of scope for
> the projection metric in this plan.

- [ ] **Step 5: Run viz tests**

Run: `python -m pytest tests/test_visualize_garg_weat.py tests/test_visualize_garg.py tests/test_visualize_weat.py -v`
Expected: PASS (existing unchanged + new proportion test).

- [ ] **Step 6: Commit**

```bash
git add scripts/visualize.py tests/test_visualize_garg_weat.py
git commit -m "feat(viz): plot single-list Cohen's d + male-leaned proportion, parameterize provincial value column"
```

---

## Task 9: Full suite + smoke run + memory note

**Files:** none (verification) + memory update.

- [ ] **Step 1: Run the entire test suite**

Run: `python -m pytest -q`
Expected: all pass. If any pre-existing unrelated failures exist, note them but ensure no NEW failures from this work.

- [ ] **Step 2: Smoke-run the orchestrator on one real config (if models are present locally)**

Run (only if `paths.models_dir` for this profile has models on disk):
```bash
python -m scripts.analyze_category_bias --config=config/profiles/garg_weat_coha_trained.yml --metrics=rnd,cohens_d
```
Expected: writes `garg_weat_*` and `cohens_d_singlelist_*` parquets to the profile's `results_dir`; log shows each model loaded once and both metrics computed. If models are not available locally, skip and note that this runs on Slurm.

- [ ] **Step 3: Verify the visualization runs on the produced summaries**

Run (if Step 2 ran): `python -m scripts.visualize main --config=config/profiles/garg_weat_coha_trained.yml`
Expected: figures for RND trend, projection trend, and both proportion trends appear in `figures_dir`.

- [ ] **Step 4: Update project memory**

Append a one-line pointer to `~/.claude/projects/-Users-houyuxin-08Coding-gender-occup-segregation/memory/MEMORY.md` and write `project_singlelist_cohens_d.md` recording: the orchestrator entry point, the `analysis.metrics` config switch, the two output file families, and that RND columns are aliased for back-compat. (Follow the memory format in the repo's MEMORY.md.)

- [ ] **Step 5: Commit any verification fixups**

```bash
git add -A
git commit -m "test: full-suite verification for single-list Cohen's d robustness check"
```

---

## Self-Review

**Spec coverage:**
- New projection metric (cosine to axis, single wordlist) → Task 4. ✓
- Proportion male-leaned for BOTH metrics → Task 2 (`build_summary`) + Task 5 (both summaries). ✓
- Config-driven `metrics: [...]`, single-pass, one model load → Task 5 + Task 7. ✓
- Both statistics × both bands (bootstrap + subsample) → Task 1 (statistic-aware bootstrap) + Task 2 (build_summary). ✓
- Reuse 9 garg_weat profiles, no new wordlists → Task 7. ✓
- RND output back-compat (aliased columns, `rnd` long column, filenames) → Task 2 (`legacy_rnd_aliases`) + Task 3 + Task 5 + existing tests in Task 5 Step 4. ✓
- Visualization of projection + proportion, reusing parameterized plotters → Task 8. ✓
- config_loader validation of metrics → Task 6. ✓
- Sign convention (positive = female-leaning) consistent across metrics → Task 4 test + RND unchanged. ✓

**Placeholder scan:** No TBD/TODO; every code step has concrete code; every test step has runnable assertions and expected output. Task 8 Step 2/3 ask the implementer to inspect `plot_garg_weat_categories_trend`'s existing guard against a hardcoded column — that's a verification-and-fix instruction with the concrete fix (use `line_col`), not a placeholder.

**Type/name consistency:** Producers (`rnd_values`, `projection_values`) share the signature `(model, unit_name, categories, gender_words, logger) -> Optional[DataFrame]` with a `value` column; orchestrator `METRIC_SPECS` maps to `(producer, long_stem, summary_stem, long_value_name, legacy)`; `build_summary(value_col=..., legacy_rnd_aliases=...)` matches its definition and call sites; `subsample_bands_from_lookup(..., statistic=...)` matches Task 2 definition and the re-export used by `tests/test_analyze_garg_weat.py`. Output filenames (`garg_weat_rnd_long`, `garg_weat_summary_by_category`, `cohens_d_singlelist_long`, `cohens_d_singlelist_summary_by_category`) are consistent across Tasks 5, 8, and the spec.
