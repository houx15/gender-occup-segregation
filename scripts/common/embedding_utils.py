"""
Shared embedding utilities for axis construction, projection, and vector operations.

Used by both prestige and WEAT analysis modes.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from gensim.models import KeyedVectors


def load_model(model_path: str) -> KeyedVectors:
    """Load a KeyedVectors model from disk."""
    return KeyedVectors.load(model_path)


def get_word_vector(model: KeyedVectors, word: str) -> Optional[np.ndarray]:
    """Get vector for a word, or None if not in vocabulary."""
    if word in model.key_to_index:
        return model[word]
    return None


def compute_centroid(
    model: KeyedVectors, words: List[str]
) -> Tuple[Optional[np.ndarray], List[str]]:
    """
    Compute centroid vector for a list of words.

    Returns:
        Tuple of (centroid_vector, list_of_found_words).
        centroid_vector is None if no words are found.
    """
    vectors = []
    found_words = []

    for word in words:
        vec = get_word_vector(model, word)
        if vec is not None:
            vectors.append(vec)
            found_words.append(word)

    if not vectors:
        return None, []

    centroid = np.mean(vectors, axis=0)
    return centroid, found_words


def construct_semantic_axis(
    positive_terms: List[str],
    negative_terms: List[str],
    model: KeyedVectors,
) -> Tuple[Optional[np.ndarray], int, int]:
    """
    Construct a normalized semantic axis from positive and negative term sets.

    The axis points from negative centroid toward positive centroid.

    Args:
        positive_terms: Terms defining the positive pole
        negative_terms: Terms defining the negative pole
        model: KeyedVectors model

    Returns:
        Tuple of (normalized_axis_vector, num_positive_found, num_negative_found).
        axis_vector is None if either pole has no terms in vocabulary.
    """
    pos_centroid, pos_found = compute_centroid(model, positive_terms)
    neg_centroid, neg_found = compute_centroid(model, negative_terms)

    if pos_centroid is None or neg_centroid is None:
        return None, len(pos_found), len(neg_found)

    axis = pos_centroid - neg_centroid

    norm = np.linalg.norm(axis)
    if norm < 1e-10:
        return None, len(pos_found), len(neg_found)

    axis = axis / norm
    return axis, len(pos_found), len(neg_found)


def compute_projection(
    word_vec: np.ndarray, axis: np.ndarray
) -> Tuple[float, float]:
    """
    Compute a word vector's projection onto an axis.

    Args:
        word_vec: The word's embedding vector
        axis: A normalized axis vector

    Returns:
        Tuple of (dot_product, cosine_similarity).
        dot_product is the raw projection (axis is unit-length).
        cosine_similarity normalizes by the word vector's length.
    """
    projection = float(np.dot(word_vec, axis))

    word_norm = np.linalg.norm(word_vec)
    if word_norm < 1e-10:
        cosine_sim = 0.0
    else:
        cosine_sim = projection / word_norm

    return projection, cosine_sim


def compute_pooled_std(values1: np.ndarray, values2: np.ndarray) -> float:
    """Compute pooled standard deviation for two groups."""
    n1, n2 = len(values1), len(values2)
    if n1 < 2 or n2 < 2:
        return 0.0

    s1 = np.std(values1, ddof=1)
    s2 = np.std(values2, ddof=1)

    pooled_var = ((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2)
    return float(np.sqrt(pooled_var))


def compute_cohens_d(
    group1_values: np.ndarray, group2_values: np.ndarray
) -> Tuple[float, float]:
    """
    Compute Cohen's d effect size.

    d = (mean1 - mean2) / pooled_std

    Returns:
        Tuple of (cohens_d, pooled_std).
    """
    mean1 = float(np.mean(group1_values))
    mean2 = float(np.mean(group2_values))

    pooled_std = compute_pooled_std(group1_values, group2_values)
    if pooled_std < 1e-10:
        return 0.0, 0.0

    cohens_d = (mean1 - mean2) / pooled_std
    return float(cohens_d), pooled_std


def check_oov(model: KeyedVectors, words: List[str]) -> Tuple[List[str], List[str]]:
    """
    Check which words are in/out of vocabulary.

    Returns:
        Tuple of (found_words, oov_words).
    """
    found = []
    oov = []
    for word in words:
        if word in model.key_to_index:
            found.append(word)
        else:
            oov.append(word)
    return found, oov
