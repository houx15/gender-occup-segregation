"""Dataset & training summary helpers.

Pure functions plus dataclasses; the CLI shim (scripts/describe_dataset.py)
wires them together. Per-unit corpus scans are cached as JSON sidecars
(.dataset_stats.json) next to the corpus_* files.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

CACHE_FILENAME = ".dataset_stats.json"
CACHE_SCHEMA_VERSION = 1


@dataclass
class CorpusStats:
    unit_name: str
    n_docs: int
    n_tokens: int
    n_vocab_raw: int
    n_corpus_files: int
    scanned_at: str
    from_cache: bool


@dataclass
class RawVolumeEntry:
    unit_name: str
    n_files: int
    n_bytes: int
    layout_hint: str
    n_source_docs: Optional[int] = None  # set by walkers that know it cheaply (e.g. weibo)


def _file_fingerprint(p: Path) -> dict:
    st = p.stat()
    return {"name": p.name, "size": st.st_size, "mtime": st.st_mtime}


def write_cache(unit_dir: Path, stats: CorpusStats, corpus_files: List[Path]) -> None:
    """Persist per-unit scan results to a JSON sidecar."""
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "n_docs": stats.n_docs,
        "n_tokens": stats.n_tokens,
        "n_vocab_raw": stats.n_vocab_raw,
        "scanned_at": stats.scanned_at,
        "corpus_files": [_file_fingerprint(p) for p in sorted(corpus_files)],
    }
    (unit_dir / CACHE_FILENAME).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def read_cache(unit_dir: Path, logger: Optional[logging.Logger] = None) -> Optional[CorpusStats]:
    """Read sidecar; return None if missing, corrupt, or schema-incompatible."""
    cache_path = unit_dir / CACHE_FILENAME
    if not cache_path.exists():
        return None
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        if logger:
            logger.warning(f"Corrupt cache at {cache_path}: {e!r}; will recompute")
        return None
    if payload.get("schema_version") != CACHE_SCHEMA_VERSION:
        return None
    return CorpusStats(
        unit_name=unit_dir.name,
        n_docs=int(payload["n_docs"]),
        n_tokens=int(payload["n_tokens"]),
        n_vocab_raw=int(payload["n_vocab_raw"]),
        n_corpus_files=len(payload.get("corpus_files", [])),
        scanned_at=str(payload.get("scanned_at", "")),
        from_cache=True,
    )


def cache_is_fresh(unit_dir: Path, corpus_files: List[Path]) -> bool:
    """True iff the sidecar's recorded fingerprints exactly match the live files."""
    cache_path = unit_dir / CACHE_FILENAME
    if not cache_path.exists():
        return False
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    if payload.get("schema_version") != CACHE_SCHEMA_VERSION:
        return False
    cached = {f["name"]: (f["size"], f["mtime"]) for f in payload.get("corpus_files", [])}
    live = {p.name: (p.stat().st_size, p.stat().st_mtime) for p in corpus_files}
    return cached == live


import datetime
from typing import Set, Tuple, Union


def _list_corpus_files(unit_dir: Path) -> List[Path]:
    """Sorted list of corpus_* files in a unit dir (excludes the .dataset_stats.json sidecar)."""
    return sorted(p for p in unit_dir.glob("corpus_*") if p.is_file())


def scan_corpus_unit(
    unit_dir: Path,
    logger: logging.Logger,
    force: bool = False,
    return_vocab: bool = False,
) -> Union[CorpusStats, Tuple[CorpusStats, Set[str]]]:
    """Count documents, tokens, and unique types in a unit's corpus_* files.

    With ``return_vocab=False`` (default) returns CorpusStats; cache is used
    when fresh. With ``return_vocab=True`` returns ``(CorpusStats, set[str])``;
    always rescans (the cache stores counts, not the vocab set).
    """
    corpus_files = _list_corpus_files(unit_dir)

    # Cache fast path (only when caller doesn't need the actual vocab set).
    if not return_vocab and not force and cache_is_fresh(unit_dir, corpus_files):
        cached = read_cache(unit_dir, logger)
        if cached is not None:
            return cached

    # Scan.
    n_docs = 0
    n_tokens = 0
    vocab: Set[str] = set()
    for path in corpus_files:
        with path.open("r", encoding="utf-8", buffering=8 * 1024 * 1024) as f:
            for line in f:
                tokens = line.split()
                if not tokens:
                    continue
                n_docs += 1
                n_tokens += len(tokens)
                vocab.update(tokens)

    stats = CorpusStats(
        unit_name=unit_dir.name,
        n_docs=n_docs,
        n_tokens=n_tokens,
        n_vocab_raw=len(vocab),
        n_corpus_files=len(corpus_files),
        scanned_at=datetime.datetime.now().isoformat(timespec="seconds"),
        from_cache=False,
    )
    if corpus_files:
        write_cache(unit_dir, stats, corpus_files)

    if return_vocab:
        return stats, vocab
    return stats
