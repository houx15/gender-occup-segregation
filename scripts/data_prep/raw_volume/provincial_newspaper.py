"""Provincial newspaper raw-data walker.

Layout: {raw_data_dir}/{province_folder}/{year}/...

Province folder names follow the mapping in
scripts.data_prep.build_corpora_provincial_newspaper.FOLDER_TO_PROVINCE
(reused so the two stay in sync).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from scripts.common.dataset_stats import RawVolumeEntry
from scripts.data_prep.build_corpora_provincial_newspaper import FOLDER_TO_PROVINCE

LAYOUT = "{raw_data_dir}/{province_folder}/{year}/..."


def _parse_unit(unit_name: str):
    """'北京_2020' -> ('北京', '2020'); returns None on bad shapes."""
    if "_" not in unit_name:
        return None
    province, year = unit_name.rsplit("_", 1)
    return province, year


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

    # Per (province, year) → list of paths.
    files_by_pv: Dict[tuple, List[Path]] = defaultdict(list)
    for folder in sorted(raw_data_dir.iterdir()):
        if not folder.is_dir():
            continue
        province = FOLDER_TO_PROVINCE.get(folder.name)
        if province is None:
            continue
        for year_dir in sorted(folder.iterdir()):
            if not year_dir.is_dir():
                continue
            year = year_dir.name
            for f in year_dir.rglob("*"):
                if f.is_file():
                    files_by_pv[(province, year)].append(f)

    out: Dict[str, RawVolumeEntry] = {}
    for u in units:
        parsed = _parse_unit(u)
        if parsed is None:
            out[u] = RawVolumeEntry(u, 0, 0, LAYOUT)
            continue
        files = files_by_pv.get(parsed, [])
        out[u] = RawVolumeEntry(
            unit_name=u,
            n_files=len(files),
            n_bytes=sum(p.stat().st_size for p in files),
            layout_hint=LAYOUT,
        )
    return out
