"""Word fixed-effects summaries for unbalanced (unit x word) bias panels.

The default summary averages a category's words over the *consistent set*
(words in vocab in every unit), which collapses when many units are small.
Here every word in vocab in >= ``min_coverage`` of the units is used, and each
unit's level is estimated from the additive model

    value[u, w] = a[u] + b[w] + e,     with mean over words of b = 0,

so a unit missing a strongly female- or male-leaning word is not shifted by
its absence. On a balanced panel a[u] equals the plain unit mean, i.e. the
default statistic. Uncertainty comes from resampling WORDS (with replacement
for the CI, without for the subsample band) and refitting, mirroring the
word-level bootstrap / subsample bands of category_summary.build_summary.
Output columns match build_summary, so plots and diagnostics are unchanged.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from scripts.common.category_summary import finalize_summary


def fit_unit_effects(u: np.ndarray, w: np.ndarray, y: np.ndarray, n_units: int,
                     n_words: int, max_iter: int = 500, tol: float = 1e-10) -> np.ndarray:
    """Two-way FE by alternating projections; returns a[u] (NaN if unit unobserved)."""
    cnt_u = np.bincount(u, minlength=n_units).astype(float)
    cnt_w = np.bincount(w, minlength=n_words).astype(float)
    seen_w = cnt_w > 0
    a = np.zeros(n_units)
    b = np.zeros(n_words)
    for _ in range(max_iter):
        a_new = np.bincount(u, y - b[w], minlength=n_units) / np.maximum(cnt_u, 1)
        b = np.bincount(w, y - a_new[u], minlength=n_words) / np.maximum(cnt_w, 1)
        shift = b[seen_w].mean()
        b[seen_w] -= shift
        a_new += shift
        done = np.max(np.abs(a_new - a)) < tol
        a = a_new
        if done:
            break
    a[cnt_u == 0] = np.nan
    return a


def coverage_word_sets(long_df: pd.DataFrame, units: List[str], min_coverage: float,
                       by_category: Optional[Dict[str, float]] = None) -> Dict[str, List[str]]:
    """Per category, words in vocab in >= min_coverage of ``units``
    (``by_category`` overrides the bar for named categories)."""
    by_category = by_category or {}
    sets: Dict[str, List[str]] = {}
    sub = long_df[long_df["unit_name"].isin(units)]
    for cat, g in sub.groupby("category", sort=False):
        cov = g[g["in_vocab"]].groupby("occupation")["unit_name"].nunique() / len(units)
        order = list(dict.fromkeys(g["occupation"]))  # wordlist order
        bar = float(by_category.get(cat, min_coverage))
        sets[cat] = [w for w in order if cov.get(w, 0.0) >= bar]
    return sets


def word_coverage_table(long_df: pd.DataFrame, units: List[str],
                        word_sets: Dict[str, List[str]]) -> pd.DataFrame:
    """category, occupation, n_units_in_vocab, coverage, used — for the research log."""
    sub = long_df[long_df["unit_name"].isin(units)]
    n_in = (sub[sub["in_vocab"]].groupby(["category", "occupation"])["unit_name"]
            .nunique())
    rows = []
    for (cat, occ), _ in sub.groupby(["category", "occupation"], sort=False):
        k = int(n_in.get((cat, occ), 0))
        rows.append({"category": cat, "occupation": occ, "n_units_in_vocab": k,
                     "coverage": k / len(units), "used": occ in word_sets.get(cat, [])})
    return (pd.DataFrame(rows)
            .sort_values(["category", "coverage"], ascending=[True, False])
            .reset_index(drop=True))


def _resample_fit(u, w, y, by_word: List[np.ndarray], draw: np.ndarray,
                  n_units: int) -> Tuple[np.ndarray, np.ndarray]:
    """Refit on resampled words (each draw becomes its own word); also prop_male."""
    rows = [by_word[j] for j in draw]
    idx = np.concatenate(rows)
    new_w = np.repeat(np.arange(len(draw)), [len(r) for r in rows])
    a = fit_unit_effects(u[idx], new_w, y[idx], n_units, len(draw))
    cnt = np.bincount(u[idx], minlength=n_units)
    male = np.bincount(u[idx], (y[idx] < 0).astype(float), minlength=n_units)
    with np.errstate(invalid="ignore", divide="ignore"):
        prop = np.where(cnt > 0, male / cnt, np.nan)
    return a, prop


def _bands(stats: np.ndarray, ci: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Percentile band + mean over resamples (axis 0); all-NaN columns stay NaN."""
    alpha = (1.0 - ci) / 2.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return (np.nanpercentile(stats, 100 * alpha, axis=0),
                np.nanpercentile(stats, 100 * (1 - alpha), axis=0),
                np.nanmean(stats, axis=0))


def build_fe_summary(
    long_df: pd.DataFrame,
    units: List[str],
    word_sets: Dict[str, List[str]],
    logger,
    value_col: str = "value",
    boot_n_iter: int = 1000,
    boot_ci: float = 0.68,
    sub_fraction: float = 0.8,
    sub_rounds: int = 100,
    sub_ci: float = 0.95,
    seed: int = 42,
    legacy_rnd_aliases: bool = False,
) -> pd.DataFrame:
    """Per-(unit, category) FE level + prop_male, each with word-bootstrap CI and
    word-subsample band. Same columns as category_summary.build_summary."""
    rng = np.random.default_rng(seed)
    unit_ix = {un: i for i, un in enumerate(units)}
    rows: List[dict] = []
    for cat, words in word_sets.items():
        g = long_df[(long_df["category"] == cat) & long_df["in_vocab"]
                    & long_df["occupation"].isin(words)
                    & long_df["unit_name"].isin(units)]
        word_ix = {wd: j for j, wd in enumerate(words)}
        u = g["unit_name"].map(unit_ix).to_numpy()
        w = g["occupation"].map(word_ix).to_numpy()
        y = g[value_col].to_numpy(dtype=float)
        n_u, n_w = len(units), len(words)
        nan = np.full(n_u, np.nan)
        if n_w == 0:
            logger.warning(f"[fixed_effects] {cat}: no words reach the coverage bar")
            est = prop = ci_lo = ci_hi = p_lo = p_hi = nan
            s_lo = s_hi = s_mean = sp_lo = sp_hi = sp_mean = nan
            n_obs = np.zeros(n_u, dtype=int)
        else:
            est = fit_unit_effects(u, w, y, n_u, n_w)
            n_obs = np.bincount(u, minlength=n_u)
            male = np.bincount(u, (y < 0).astype(float), minlength=n_u)
            with np.errstate(invalid="ignore", divide="ignore"):
                prop = np.where(n_obs > 0, male / n_obs, np.nan)
            by_word = [np.flatnonzero(w == j) for j in range(n_w)]
            boot = [_resample_fit(u, w, y, by_word, rng.integers(0, n_w, n_w), n_u)
                    for _ in range(boot_n_iter)]
            ci_lo, ci_hi, _ = _bands(np.array([b[0] for b in boot]), boot_ci)
            p_lo, p_hi, _ = _bands(np.array([b[1] for b in boot]), boot_ci)
            k = max(1, int(round(sub_fraction * n_w)))
            subs = [_resample_fit(u, w, y, by_word,
                                  rng.choice(n_w, size=k, replace=False), n_u)
                    for _ in range(sub_rounds)]
            s_lo, s_hi, s_mean = _bands(np.array([s[0] for s in subs]), sub_ci)
            sp_lo, sp_hi, sp_mean = _bands(np.array([s[1] for s in subs]), sub_ci)
            logger.info(f"[fixed_effects] {cat}: {n_w} words, {len(y)} unit-word cells")
        for i, un in enumerate(units):
            rows.append({
                "unit_name": un, "category": cat,
                "mean_value": est[i], "mean_ci_low": ci_lo[i], "mean_ci_high": ci_hi[i],
                "mean_sub_low": s_lo[i], "mean_sub_high": s_hi[i], "mean_sub_mean": s_mean[i],
                "prop_male": prop[i], "prop_ci_low": p_lo[i], "prop_ci_high": p_hi[i],
                "prop_sub_low": sp_lo[i], "prop_sub_high": sp_hi[i], "prop_sub_mean": sp_mean[i],
                "n_occupations": int(n_obs[i]), "n_consistent": n_w,
            })
    return finalize_summary(rows, legacy_rnd_aliases, logger)
