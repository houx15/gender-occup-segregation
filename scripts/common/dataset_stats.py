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
