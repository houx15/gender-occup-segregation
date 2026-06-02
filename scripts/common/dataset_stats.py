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


def discover_units(config: dict) -> List[str]:
    """Return sorted unit directory names under corpora_dir.

    Mirrors scripts.train_embeddings.discover_units; lifted here so the
    describe-dataset tool doesn't import the trainer.
    """
    corpora_dir = Path(config["paths"]["corpora_dir"])
    if not corpora_dir.exists():
        return []
    return sorted(d.name for d in corpora_dir.iterdir() if d.is_dir())


def model_vocab_size(model_path: Path, logger: logging.Logger) -> Optional[int]:
    """Open a gensim model and return its vocab size; None on missing/corrupt.

    Uses KeyedVectors.load — works for .kv files and for Word2Vec.save_word2vec_format
    output. For full Word2Vec.save() output, gensim falls back through its loader
    chain; if that fails the function logs and returns None.
    """
    if not model_path.exists():
        logger.warning(f"Model file missing: {model_path}")
        return None
    try:
        # Local import: gensim is broken on some test envs (scipy.linalg.triu);
        # only this function actually needs it.
        from gensim.models import KeyedVectors
        kv = KeyedVectors.load(str(model_path))
        return len(kv.index_to_key)
    except Exception as e:  # noqa: BLE001 — gensim raises many distinct types
        logger.warning(f"Could not introspect {model_path}: {e!r}")
        return None


from typing import Dict, Tuple


@dataclass
class SourceTotals:
    n_units: int
    n_docs: int
    n_tokens: int
    vocab_raw_min: int
    vocab_raw_max: int
    vocab_raw_mean: float
    n_model_vocab_sum: int             # sum of per-unit model vocabs (None entries skipped)
    n_raw_files: int
    n_raw_bytes: int
    vocab_union_count: Optional[int]   # None unless --vocab-union was passed


PerUnitTriple = Tuple[CorpusStats, Optional[int], Optional[RawVolumeEntry]]
# (corpus_stats, model_vocab_size_or_None, raw_volume_entry_or_None)


def aggregate_source(
    per_unit: Dict[str, PerUnitTriple],
    vocab_union_count: Optional[int],
) -> SourceTotals:
    """Reduce per-unit triples to one source-level summary row."""
    if not per_unit:
        return SourceTotals(0, 0, 0, 0, 0, 0.0, 0, 0, 0, vocab_union_count)
    vocab_raws = [s.n_vocab_raw for s, _, _ in per_unit.values()]
    return SourceTotals(
        n_units=len(per_unit),
        n_docs=sum(s.n_docs for s, _, _ in per_unit.values()),
        n_tokens=sum(s.n_tokens for s, _, _ in per_unit.values()),
        vocab_raw_min=min(vocab_raws),
        vocab_raw_max=max(vocab_raws),
        vocab_raw_mean=sum(vocab_raws) / len(vocab_raws),
        n_model_vocab_sum=sum(v for _, v, _ in per_unit.values() if v is not None),
        n_raw_files=sum(r.n_files for _, _, r in per_unit.values() if r is not None),
        n_raw_bytes=sum(r.n_bytes for _, _, r in per_unit.values() if r is not None),
        vocab_union_count=vocab_union_count,
    )


def _human_bytes(n: int) -> str:
    """1024-based byte sizes; one decimal place above bytes (e.g. 1.5 KB)."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return f"{n:.1f} TB"  # unreachable, kept to satisfy type-checkers


def _fmt(n: Optional[Union[int, float]]) -> str:
    if n is None:
        return "n/a"
    if isinstance(n, float):
        return f"{n:,.1f}"
    return f"{n:,}"


import re


def _year_range(unit_name: str) -> str:
    """Parse year info from common unit-name shapes; returns en-dash range or '—'.

    Examples:
      '1940_1949'  -> '1940–1949'   (longitudinal slice)
      '1940s'      -> '1940–1949'   (COHA decade)
      '北京_2020'  -> '2020'        (province-year)
      '北京'       -> '—'           (bare province, no year info)
    """
    m = re.match(r"^(\d{4})_(\d{4})$", unit_name)
    if m:
        return f"{m.group(1)}–{m.group(2)}"
    m = re.match(r"^(\d{4})s$", unit_name)
    if m:
        start = int(m.group(1))
        return f"{start}–{start + 9}"
    m = re.match(r"^.+_(\d{4})$", unit_name)
    if m:
        return m.group(1)
    return "—"


def render_markdown(
    totals: SourceTotals,
    per_unit: Dict[str, PerUnitTriple],
    config: dict,
    config_path: str,
    generated_at: str,
) -> str:
    """Render the per-source Markdown summary."""
    src = config.get("data_source", "?")
    lang = config.get("language", "?")
    emb = config.get("embedding", {})
    is_ngram = src == "ngram"
    docs_header = "N-gram entries" if is_ngram else "Documents"

    lines: List[str] = []
    lines.append(f"# Dataset Summary — {src} ({lang})")
    lines.append("")
    lines.append(f"Generated {generated_at} from `{config_path}`.")
    lines.append("")

    # Corpus totals
    lines.append("## Corpus totals")
    lines.append("")
    lines.append(f"| Units | {docs_header} | Tokens | Raw vocab per unit (min / mean / max){' | Raw vocab union' if totals.vocab_union_count is not None else ''} | Trained-model vocab (sum, min_count={emb.get('min_count', '?')}) |")
    sep_extra = "|---" if totals.vocab_union_count is not None else ""
    lines.append(f"|---|---|---|---{sep_extra}|---|")
    vocab_cells = f"{_fmt(totals.vocab_raw_min)} / {_fmt(totals.vocab_raw_mean)} / {_fmt(totals.vocab_raw_max)}"
    union_col = f" | {_fmt(totals.vocab_union_count)}" if totals.vocab_union_count is not None else ""
    lines.append(
        f"| {_fmt(totals.n_units)} | {_fmt(totals.n_docs)} | {_fmt(totals.n_tokens)} | "
        f"{vocab_cells}{union_col} | {_fmt(totals.n_model_vocab_sum)} |"
    )
    lines.append("")

    # Raw data
    lines.append("## Raw data")
    lines.append("")
    layout_hint = next(
        (r.layout_hint for _, _, r in per_unit.values() if r is not None), None
    )
    raw_dir = config.get("paths", {}).get("raw_data_dir", "?")
    if layout_hint:
        lines.append(f"- Layout: `{layout_hint}`")
    lines.append(f"- raw_data_dir: `{raw_dir}`")
    lines.append(f"- Source files: {_fmt(totals.n_raw_files)}")
    lines.append(f"- Bytes: {_human_bytes(totals.n_raw_bytes)}")
    lines.append("")

    # Training
    lines.append("## Training")
    lines.append("")
    algo = "Word2Vec skip-gram with negative sampling (gensim)" if emb.get("sg") == 1 else "Word2Vec CBOW (gensim)"
    lines.append(f"- Algorithm: {algo}")
    params = (
        f"vector_size={emb.get('vector_size', '?')} · "
        f"window={emb.get('window', '?')} · "
        f"min_count={emb.get('min_count', '?')} · "
        f"sg={emb.get('sg', '?')} · "
        f"negative={emb.get('negative', '?')} · "
        f"epochs={emb.get('epochs', '?')} · "
        f"seed={emb.get('seed', '?')}"
    )
    lines.append(f"- `{params}`")
    template = emb.get("model_name_template", "?")
    models_dir = config.get("paths", {}).get("models_dir", "?")
    lines.append(f"- Model files: `{models_dir}/{template}`")
    lines.append("")

    # Per-unit breakdown
    lines.append("## Per-unit breakdown")
    lines.append("")
    lines.append(
        f"| Unit | Year range | {docs_header} | Tokens | Tokens/doc | "
        f"Raw vocab | Model vocab | Raw files | Raw bytes |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for unit_name in sorted(per_unit):
        stats, mv, raw = per_unit[unit_name]
        raw_files = _fmt(raw.n_files) if raw is not None else "n/a"
        raw_bytes = _human_bytes(raw.n_bytes) if raw is not None else "n/a"
        tokens_per_doc = (
            f"{stats.n_tokens / stats.n_docs:,.1f}" if stats.n_docs > 0 else "—"
        )
        lines.append(
            f"| {unit_name} | {_year_range(unit_name)} | {_fmt(stats.n_docs)} | "
            f"{_fmt(stats.n_tokens)} | {tokens_per_doc} | "
            f"{_fmt(stats.n_vocab_raw)} | {_fmt(mv)} | {raw_files} | {raw_bytes} |"
        )
    lines.append("")
    return "\n".join(lines)


from scripts.common.config_loader import get_model_name


def resolve_walker(config: dict):
    """Pick the right raw-data walker for this config's data_source.

    Ngram dispatches on language too (ngram_zh vs ngram_en).
    Returns None if no walker is registered for this source.
    """
    from scripts.data_prep.raw_volume import WALKERS

    src = config.get("data_source")
    if src == "ngram":
        key = f"ngram_{config.get('language', '')}"
        return WALKERS.get(key)
    return WALKERS.get(src)


def run(
    config: dict,
    config_path: str,
    force: bool = False,
    units: Optional[str] = None,
    no_raw: bool = False,
    no_model: bool = False,
    vocab_union: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Path:
    """Orchestrate one source's dataset summary; write Markdown; return its path.

    Args:
        config: parsed config dict (already validated).
        config_path: path string for provenance line in the Markdown header.
        force: ignore sidecar caches, rescan every unit.
        units: optional comma-separated subset of unit names to include.
        no_raw: skip the raw-data walker (per-unit raw cells become n/a).
        no_model: skip per-unit model vocab introspection.
        vocab_union: opt-in cross-unit raw-vocab union (bypasses cache).
        logger: optional pre-built logger; default uses a console-only one.
    """
    if logger is None:
        logger = logging.getLogger("describe_dataset")
        if not logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
            logger.addHandler(h)
            logger.setLevel(logging.INFO)

    unit_list = discover_units(config)
    if units:
        wanted = set(u.strip() for u in units.split(","))
        unit_list = [u for u in unit_list if u in wanted]
    if not unit_list:
        logger.error(f"No units to describe under {config['paths']['corpora_dir']}")
        raise SystemExit(1)
    logger.info(f"Describing {len(unit_list)} unit(s): {', '.join(unit_list)}")

    corpora_dir = Path(config["paths"]["corpora_dir"])
    models_dir = Path(config["paths"]["models_dir"])

    # Raw-data walk (once per source).
    raw_by_unit: Dict[str, Optional[RawVolumeEntry]] = {u: None for u in unit_list}
    if not no_raw:
        walker = resolve_walker(config)
        if walker is None:
            logger.warning(f"No raw-data walker for data_source={config.get('data_source')!r}; "
                           "raw cells will be n/a")
        else:
            raw_dir = Path(config["paths"].get("raw_data_dir", ""))
            raw_by_unit = {u: e for u, e in walker(raw_dir, unit_list, config, logger).items()}

    # Per-unit corpus + model.
    per_unit: Dict[str, PerUnitTriple] = {}
    vocab_union_set: Optional[Set[str]] = set() if vocab_union else None
    for u in unit_list:
        unit_dir = corpora_dir / u
        if vocab_union:
            stats, vocab = scan_corpus_unit(unit_dir, logger, force=force, return_vocab=True)
            vocab_union_set.update(vocab)
        else:
            stats = scan_corpus_unit(unit_dir, logger, force=force)
        mv = None if no_model else model_vocab_size(models_dir / get_model_name(u, config), logger)
        per_unit[u] = (stats, mv, raw_by_unit.get(u))

    totals = aggregate_source(
        per_unit,
        vocab_union_count=len(vocab_union_set) if vocab_union_set is not None else None,
    )

    generated_at = datetime.datetime.now().strftime("%Y-%m-%d")
    md = render_markdown(totals, per_unit, config, config_path, generated_at)
    results_dir = Path(config["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "dataset_summary.md"
    out_path.write_text(md, encoding="utf-8")
    logger.info(f"Wrote {out_path}")
    return out_path
