"""Chinese Google Ngram raw-data walker.

Layout: {raw_data_dir}/**/*.gz, with the year extractable from the filename
(typically `5gm-YYYY.gz`).
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from scripts.common.dataset_stats import RawVolumeEntry

LAYOUT = "{raw_data_dir}/**/*.gz  (year from filename, e.g. 5gm-YYYY.gz)"
_YEAR_RE = re.compile(r"(\d{4})")


def _slice_bounds(unit: str) -> Tuple[int, int] | None:
    m = re.match(r"^(\d{4})_(\d{4})$", unit)
    return (int(m.group(1)), int(m.group(2))) if m else None


def walk(
    raw_data_dir: Path,
    units: List[str],
    config: dict,
    logger,
) -> Dict[str, RawVolumeEntry]:
    raw_data_dir = Path(raw_data_dir)
    if not raw_data_dir.exists():
        logger.warning(f"raw_data_dir not present: {raw_data_dir}")
        return {u: RawVolumeEntry(u, 0, 0, LAYOUT) for u in units}

    files_by_unit: Dict[str, List[Path]] = defaultdict(list)
    for path in sorted(raw_data_dir.rglob("*.gz")):
        m = _YEAR_RE.search(path.name)
        if not m:
            continue
        year = int(m.group(1))
        for u in units:
            b = _slice_bounds(u)
            if b is not None and b[0] <= year <= b[1]:
                files_by_unit[u].append(path)

    out: Dict[str, RawVolumeEntry] = {}
    for u in units:
        files = files_by_unit.get(u, [])
        out[u] = RawVolumeEntry(
            unit_name=u,
            n_files=len(files),
            n_bytes=sum(p.stat().st_size for p in files),
            layout_hint=LAYOUT,
        )
    return out
