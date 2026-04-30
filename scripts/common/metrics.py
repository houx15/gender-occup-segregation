"""Embedding bias metrics (Garg 2018 RND) and bootstrap CI helpers."""

from __future__ import annotations

import numpy as np


def l2_normalize(v: np.ndarray) -> np.ndarray:
    """Return v scaled to unit L2-norm. Zero vectors are returned unchanged.

    Mirrors Garg et al. 2018's `dataset_utilities/normalize_vectors.py`, which
    divides every embedding row by its L2 norm before any distance is computed.
    Without this step, distances and centroids inherit per-decade vector-norm
    drift and the relative-norm-distance metric is not directly comparable to
    Garg's published numbers.
    """
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return v
    return v / n


def relative_norm_distance(
    v_w: np.ndarray, c_male: np.ndarray, c_female: np.ndarray
) -> float:
    """Garg (2018) Eq. 3: ||v_w - c_male|| - ||v_w - c_female||.

    Sign convention matches Garg's plots ("Avg. Women Bias" axis):
      positive  → v_w is closer to the female centroid (female-leaning)
      negative  → v_w is closer to the male centroid (male-leaning)

    Caller is responsible for L2-normalizing v_w / c_male / c_female before
    invoking this if Garg-comparable magnitudes are wanted; see `l2_normalize`.
    """
    arrays = (v_w, c_male, c_female)
    if any(a.size == 0 for a in arrays):
        raise ValueError("relative_norm_distance: inputs must be non-empty")
    if not (v_w.shape == c_male.shape == c_female.shape):
        raise ValueError(
            "relative_norm_distance: shape mismatch "
            f"(v_w={v_w.shape}, c_male={c_male.shape}, c_female={c_female.shape})"
        )
    return float(np.linalg.norm(v_w - c_male) - np.linalg.norm(v_w - c_female))


def bootstrap_ci(
    values: np.ndarray,
    n_iter: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    values = np.asarray(values)
    if values.size == 0:
        raise ValueError("bootstrap_ci: values must be non-empty")
    rng = np.random.default_rng(seed)
    n = values.shape[0]
    idx = rng.integers(0, n, size=(n_iter, n))
    means = values[idx].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    lo = float(np.percentile(means, 100.0 * alpha))
    hi = float(np.percentile(means, 100.0 * (1.0 - alpha)))
    if lo > hi:  # numerical guard; percentile should already enforce this
        lo, hi = hi, lo
    return lo, hi
