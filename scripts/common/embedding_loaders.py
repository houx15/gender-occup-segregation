"""Format-specific loaders for pre-trained embedding archives.

Each loader returns a gensim KeyedVectors, so downstream analysis code
(``analyze_garg``, etc.) can treat every embedding source the same way.

Currently implemented:
  * ``load_histwords_decade``  — Hamilton et al. 2016 HistWords format
                                 ({decade}-w.npy + {decade}-vocab.pkl)

To add later, when Fig 1 (Google News / GloVe scatter) is in scope:
  * ``load_word2vec_bin``      — Google News word2vec .bin / .bin.gz
  * ``load_glove_text``        — GloVe text format (word v1 v2 ... vN)
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterable, List

import numpy as np


def _load_vocab_pkl(vocab_path: Path) -> List[str]:
    """Load a HistWords vocab pickle. The file is a Python pickle of a list
    of word strings (one entry per row of the parallel -w.npy matrix).

    HistWords (Hamilton et al. 2016) was released as Python 2 code, so the
    vocab pickles contain Py2 ``str`` objects. Under Py3 those unpickle as
    ``bytes`` by default — making every key look like ``b'doctor'`` and
    every downstream lookup OOV. We pass ``encoding="latin1"`` so the legacy
    pickles deserialize as proper str, and decode any straggler bytes for
    safety. Modern Py3 pickles (used by our tests) are unaffected.
    """
    with open(vocab_path, "rb") as f:
        try:
            words = pickle.load(f, encoding="latin1")
        except TypeError:
            # encoding= was added in Py3; if we're somehow on Py2, fall back.
            f.seek(0)
            words = pickle.load(f)
    if isinstance(words, np.ndarray):
        words = words.tolist()
    if not isinstance(words, (list, tuple)):
        raise ValueError(
            f"Expected list/tuple in {vocab_path}, got {type(words).__name__}"
        )
    out: List[str] = []
    for w in words:
        if isinstance(w, bytes):
            out.append(w.decode("utf-8", errors="replace"))
        else:
            out.append(str(w))
    return out


def load_histwords_decade(npy_path: Path, vocab_path: "Path | None" = None):
    """Build a KeyedVectors from a HistWords decade pair (-w.npy + -vocab.pkl).

    Args:
        npy_path:   Path to the {decade}-w.npy vector matrix.
        vocab_path: Path to the matching {decade}-vocab.pkl file. If omitted,
                    derived by replacing "-w.npy" with "-vocab.pkl".

    Returns:
        gensim.models.KeyedVectors with vectors and vocab populated.

    Note: gensim is imported lazily so this module loads cleanly in test
    environments where the gensim install is broken (e.g. host scipy missing
    triu); only this function actually requires it.
    """
    from gensim.models import KeyedVectors  # local import (see docstring)

    npy_path = Path(npy_path)
    if vocab_path is None:
        if not npy_path.name.endswith("-w.npy"):
            raise ValueError(
                f"Cannot infer vocab path from {npy_path}: expected -w.npy suffix"
            )
        vocab_path = npy_path.with_name(npy_path.name.replace("-w.npy", "-vocab.pkl"))
    else:
        vocab_path = Path(vocab_path)

    if not npy_path.exists():
        raise FileNotFoundError(f"HistWords vector file not found: {npy_path}")
    if not vocab_path.exists():
        raise FileNotFoundError(f"HistWords vocab file not found: {vocab_path}")

    vectors = np.load(npy_path)
    if vectors.ndim != 2:
        raise ValueError(
            f"Expected 2-D vector matrix in {npy_path}, got shape {vectors.shape}"
        )

    words = _load_vocab_pkl(vocab_path)
    if len(words) != vectors.shape[0]:
        raise ValueError(
            f"vocab length {len(words)} != vectors rows {vectors.shape[0]} "
            f"for {npy_path}"
        )

    kv = KeyedVectors(vector_size=vectors.shape[1])
    kv.add_vectors(words, vectors.astype(np.float32, copy=False))
    return kv


def histwords_unit_name_from_npy(npy_path: Path) -> str | None:
    """'1990-w.npy' -> '1990s'. Returns None if the filename doesn't match."""
    npy_path = Path(npy_path)
    if not npy_path.name.endswith("-w.npy"):
        return None
    stem = npy_path.name[: -len("-w.npy")]
    try:
        year = int(stem)
    except ValueError:
        return None
    if not (1500 <= year <= 2100):  # sanity guard
        return None
    return f"{year}s"


def discover_histwords_decades(models_dir: Path) -> Iterable[tuple[Path, str]]:
    """Yield (npy_path, unit_name) for every {YYYY}-w.npy under ``models_dir``,
    sorted by decade ascending. Pairs with missing -vocab.pkl are skipped.
    """
    models_dir = Path(models_dir)
    pairs: list[tuple[Path, str]] = []
    for npy in models_dir.rglob("*-w.npy"):
        unit_name = histwords_unit_name_from_npy(npy)
        if unit_name is None:
            continue
        vocab = npy.with_name(npy.name.replace("-w.npy", "-vocab.pkl"))
        if not vocab.exists():
            continue
        pairs.append((npy, unit_name))
    pairs.sort(key=lambda pair: pair[1])
    return pairs
