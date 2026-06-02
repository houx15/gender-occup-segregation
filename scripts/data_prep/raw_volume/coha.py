"""COHA raw-data walker.

Layout: {raw_data_dir}/text_{decade}/*.txt
Units are decades (e.g. '1940s').
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

from scripts.common.dataset_stats import RawVolumeEntry

LAYOUT = "{raw_data_dir}/text_{decade}/*.txt"


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

    out: Dict[str, RawVolumeEntry] = {}
    for u in units:
        m = re.match(r"^(\d{4})s$", u)
        if not m:
            out[u] = RawVolumeEntry(u, 0, 0, LAYOUT)
            continue
        decade = m.group(1)
        decade_dir = raw_data_dir / f"text_{decade}"
        files = sorted(decade_dir.glob("*.txt")) if decade_dir.exists() else []
        out[u] = RawVolumeEntry(
            unit_name=u,
            n_files=len(files),
            n_bytes=sum(p.stat().st_size for p in files),
            layout_hint=LAYOUT,
        )
    return out
