#!/usr/bin/env python3
"""Diagnose why subsampled slice corpora differ in size (read-only).

Answers: is a slice's size the *expected* result of per-year capping over real
data, or a symptom of the cap silently failing / data being duplicated?

For each slice dir present under ``corpora_dir`` it reports the emitted line
count against the theoretical ceiling the cap allows:

    bound(slice) = sum_{y in slice} min(year_total[y], per_year_token_cap)

Under a correct per-year cap, emitted lines can never *materially* exceed this
bound: clean_ngram only drops ngrams (sum of kept match_count <= year_total),
and per-year emissions <= scale * year_total = min(year_total, cap). So:

    lines / bound  ~<= 1   -> size is real data under the cap (expected)
    lines / bound   >> 1   -> cap did not bite / data duplicated (a bug)

(The only legitimate way to slightly exceed the bound is Bernoulli rounding on
*capped* years, which adds <= 1 line per row; a ratio near 2, 3, 4... is not
that.)

It also prints the per-year ``year_total`` vs cap table, which is the root-cause
view: years with ``year_total < cap`` are PASS-THROUGH (no subsampling at all),
so slices spanning sparse eras (e.g. 1970-74, Cultural Revolution) are small and
slices spanning booms (1980-84, post-reform) are large — the cap does NOT
equalize them because it is per-year, not per-slice.

Usage (from repo root):
    python -m scripts.diagnose_subsampled_sizes
    python -m scripts.diagnose_subsampled_sizes --config=config/profiles/garg_weat_china_ngram_subsampled.yml
"""

from pathlib import Path

import fire

from scripts.common.config_loader import load_config
from scripts.data_prep.ngram_totalcounts import load_totalcounts
from scripts.data_prep.build_corpora_ngram import generate_time_slices


def count_newlines(path: Path) -> int:
    """Count lines in a (possibly huge) file without loading it into memory."""
    n = 0
    with open(path, "rb") as f:
        while True:
            chunk = f.read(16 * 1024 * 1024)
            if not chunk:
                break
            n += chunk.count(b"\n")
    return n


def sum_counts(path: Path) -> int:
    """Sum the COMPACT count column (Σ tokens) from a corpus file.

    Capped corpora store one ``ngram<TAB>count`` line per unique ngram, so the
    emitted TOKEN total is Σ of column 2 — not the physical line count (which is
    now the number of unique ngrams). A line without a TAB count column counts
    as 1 (presence/plain)."""
    total = 0
    with open(path, "r", encoding="utf-8", errors="ignore",
              buffering=16 * 1024 * 1024) as f:
        for line in f:
            _ngram, tab, count = line.rstrip("\n").partition("\t")
            total += int(count) if tab else 1
    return total


def human(nbytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if nbytes < 1024:
            return f"{nbytes:.1f}{unit}"
        nbytes /= 1024
    return f"{nbytes:.1f}PB"


def main(config: str = "config/profiles/garg_weat_china_ngram_subsampled.yml"):
    cfg = load_config(config)
    corpus_cfg = cfg["corpus"]
    cap = int(corpus_cfg.get("per_year_token_cap", 1_000_000_000))
    weight_mode = corpus_cfg.get("weight_mode", "presence")

    corpora_dir = Path(cfg["paths"]["corpora_dir"])
    raw_dir = Path(cfg["paths"]["raw_ngram_dir"])

    print("=" * 78)
    print(f"config           = {config}")
    print(f"weight_mode      = {weight_mode}")
    print(f"per_year_token_cap = {cap:,}")
    print(f"corpora_dir      = {corpora_dir}")
    print("=" * 78)

    # Prefer the data-derived totalcounts (the correct per-year denominator);
    # fall back to the shipped file only if the derived cache isn't built yet.
    derived = raw_dir / "totalcounts-5.derived"
    totalcounts_path = derived if derived.exists() else raw_dir / "totalcounts-5"
    print(f"totalcounts source = {totalcounts_path.name}")
    year_total = load_totalcounts(totalcounts_path)

    ts = cfg["time_slices"]
    slices = generate_time_slices(
        ts["start_year"], ts["end_year"], ts["window_size"], ts["step_size"]
    )

    print("\n=== per-year totalcounts vs cap ===")
    print("  (PASS-THROUGH = year_total < cap => scale 1.0 => NO subsampling)")
    for y in sorted(year_total):
        t = year_total[y]
        scale = min(1.0, cap / t) if t else 0.0
        tag = "CAPPED" if t > cap else "pass-through"
        print(f"  {y}: year_total={t:>16,}  scale={scale:.4f}  {tag}")

    print("\n=== per-slice: emitted TOKENS vs cap bound ===")
    print("  Corpora are COMPACT: one 'ngram<TAB>count' line per unique ngram.")
    print("  tokens = Σ count column  (what word2vec trains on after expansion)")
    print("  ngrams = physical lines  (unique types on disk; drives disk size)")
    print("  bound = sum_y min(year_total[y], cap)  (hard ceiling under correct capping)")
    print("  ratio = tokens / bound   (~<=1 expected; >>1 means cap failed)")
    print("-" * 78)
    for (s, e) in slices:
        name = f"{s}_{e}"
        bound = sum(min(year_total.get(y, 0), cap) for y in range(s, e + 1))
        d = corpora_dir / name
        if not d.is_dir():
            print(f"  {name}: (absent)                                   bound={bound:>15,}")
            continue
        files = sorted(d.glob("corpus_*.txt"))
        ngrams = sum(count_newlines(p) for p in files)
        tokens = sum(sum_counts(p) for p in files)
        nbytes = sum(p.stat().st_size for p in files)
        ratio = tokens / bound if bound else float("inf")
        verdict = "OK" if tokens <= bound * 1.05 else ">>> EXCEEDS CAP BOUND — investigate"
        print(
            f"  {name}: files={len(files):>3}  ngrams={ngrams:>13,}  tokens={tokens:>15,}  "
            f"size={human(nbytes):>9}  bound={bound:>15,}  ratio={ratio:5.2f}  {verdict}"
        )
    print("-" * 78)


if __name__ == "__main__":
    fire.Fire(main)
