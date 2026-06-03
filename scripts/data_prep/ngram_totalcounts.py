"""Parse the Google Books v3 totalcounts-5 file into a per-year word-count dict.

The file is a single line of tab-separated cells; each cell is the comma-separated
tuple `year,match_count,page_count,volume_count`. We keep only year → match_count.
"""

from __future__ import annotations

from pathlib import Path


def parse_totalcounts_text(text: str) -> dict[int, int]:
    """Parse the raw text of a v3 totalcounts file. Malformed cells are skipped."""
    result: dict[int, int] = {}
    for cell in text.strip().split("\t"):
        parts = cell.split(",")
        if len(parts) < 2:
            continue
        try:
            year = int(parts[0])
            match_count = int(parts[1])
        except ValueError:
            continue
        result[year] = match_count
    return result


def load_totalcounts(path: Path) -> dict[int, int]:
    """Read a totalcounts-5 file from disk and parse it."""
    return parse_totalcounts_text(Path(path).read_text(encoding="utf-8"))
