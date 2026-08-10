#!/usr/bin/env python3
"""Wire-copy deduplication for US news corpora.

Syndicated copy (the same AP story printed in many states) inflates per-state
signal. This module removes it. Two methods:
  * ``exact``   — drop repeats sharing a normalized-text hash. O(n), robust.
  * ``shingle`` — MinHash + LSH near-duplicate detection: catches wire copy with
                  minor edits. Dependency-free (hashlib-based permutations).

Scope is controlled by the caller: for ``within_year`` scope, call ``reset()``
between year slices so only same-year cross-state duplicates collapse.
"""

from __future__ import annotations

import hashlib
import re
from typing import Dict, List, Set

_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]+")
_WS_RE = re.compile(r"\s+")
_MERSENNE = (1 << 61) - 1  # large prime for hash permutations


def normalize_for_hash(text: str) -> str:
    """Lowercase, drop non-alphanumerics, collapse whitespace."""
    s = (text or "").lower()
    s = _NON_ALNUM_RE.sub(" ", s)
    return _WS_RE.sub(" ", s).strip()


def _stable_hash(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "big")


class Deduper:
    """Track seen documents; report whether a new one duplicates an old one."""

    def __init__(self, method: str = "shingle", shingle_k: int = 8,
                 n_perm: int = 64, bands: int = 16, seed: int = 42):
        if method not in ("exact", "shingle"):
            raise ValueError(f"unknown dedup method: {method!r}")
        if method == "shingle" and n_perm % bands != 0:
            raise ValueError("n_perm must be divisible by bands")
        self.method = method
        self.shingle_k = shingle_k
        self.n_perm = n_perm
        self.bands = bands
        self.rows_per_band = n_perm // bands
        # deterministic (a, b) permutation coefficients
        rng = _stable_hash(f"seed:{seed}")
        self._ab = []
        for i in range(n_perm):
            a = (_stable_hash(f"a:{seed}:{i}") % (_MERSENNE - 1)) + 1
            b = _stable_hash(f"b:{seed}:{i}") % _MERSENNE
            self._ab.append((a, b))
        self.reset()

    def reset(self) -> None:
        self._exact_seen: Set[str] = set()
        self._lsh_buckets: Dict[int, Set[int]] = {}
        self._next_id = 0

    def _minhash(self, text: str) -> List[int]:
        toks = normalize_for_hash(text).split()
        if len(toks) < self.shingle_k:
            shingles = {" ".join(toks)} if toks else {""}
        else:
            shingles = {
                " ".join(toks[i:i + self.shingle_k])
                for i in range(len(toks) - self.shingle_k + 1)
            }
        hvals = [_stable_hash(s) for s in shingles]
        sig = []
        for a, b in self._ab:
            sig.append(min(((a * h + b) % _MERSENNE) for h in hvals))
        return sig

    def _band_keys(self, sig: List[int]) -> List[int]:
        keys = []
        for band in range(self.bands):
            chunk = tuple(sig[band * self.rows_per_band:(band + 1) * self.rows_per_band])
            keys.append(_stable_hash(f"{band}:{chunk}"))
        return keys

    def is_duplicate(self, text: str) -> bool:
        if self.method == "exact":
            key = hashlib.blake2b(
                normalize_for_hash(text).encode("utf-8"), digest_size=16
            ).hexdigest()
            if key in self._exact_seen:
                return True
            self._exact_seen.add(key)
            return False

        # shingle: LSH banding — a collision in any band => candidate duplicate.
        sig = self._minhash(text)
        keys = self._band_keys(sig)
        for k in keys:
            if k in self._lsh_buckets and self._lsh_buckets[k]:
                # any prior doc shares a band -> treat as near-duplicate
                return True
        for k in keys:
            self._lsh_buckets.setdefault(k, set()).add(self._next_id)
        self._next_id += 1
        return False
