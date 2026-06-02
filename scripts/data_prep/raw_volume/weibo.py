"""Weibo raw-data walker.

Layout: {raw_data_dir}/.../*.parquet, with a ``user_province`` (GB/T 2260 code)
or ``region_name`` column. Per-province grouping; row counts are exact (parquet
metadata is cheap to read).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from scripts.common.dataset_stats import RawVolumeEntry
from scripts.data_prep.build_corpora_weibo import (
    PROVINCE_CODE_TO_NAME, PROVINCE_NAME_TO_CODE,
)

LAYOUT = "{raw_data_dir}/**/*.parquet (user_province GB/T 2260 codes)"


def _row_count(parquet_path: Path) -> int:
    """Cheap row count via parquet metadata."""
    import pyarrow.parquet as pq
    return pq.ParquetFile(parquet_path).metadata.num_rows


def _province_of(path: Path, logger) -> Optional[str]:
    """Read just enough of the parquet to find the dominant province."""
    import pyarrow.parquet as pq
    try:
        # Read only the province column.
        for col in ("user_province", "region_name"):
            schema = pq.read_schema(path)
            if col in schema.names:
                table = pq.read_table(path, columns=[col])
                values = table[col].to_pylist()
                if not values:
                    return None
                # Most-common entry.
                from collections import Counter
                top = Counter(v for v in values if v is not None).most_common(1)
                if not top:
                    return None
                v = top[0][0]
                if col == "user_province":
                    return PROVINCE_CODE_TO_NAME.get(str(v))
                return v  # region_name is already a province name
        return None
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Could not read province from {path.name}: {e!r}")
        return None


def walk(
    raw_data_dir: Path,
    units: List[str],
    config: dict,
    logger,
) -> Dict[str, RawVolumeEntry]:
    raw_data_dir = Path(raw_data_dir)
    if not raw_data_dir.exists():
        logger.warning(f"raw_data_dir not present: {raw_data_dir}")
        return {
            u: RawVolumeEntry(u, 0, 0, LAYOUT, n_source_docs=0) for u in units
        }

    by_prov: Dict[str, List[Path]] = defaultdict(list)
    rows_by_prov: Dict[str, int] = defaultdict(int)

    parquets = sorted(raw_data_dir.rglob("*.parquet"))
    for i, p in enumerate(parquets, 1):
        if i % 100 == 0:
            logger.info(f"  Weibo walker: scanned {i}/{len(parquets)} parquets")
        province = _province_of(p, logger)
        if province is None:
            continue
        by_prov[province].append(p)
        rows_by_prov[province] += _row_count(p)

    out: Dict[str, RawVolumeEntry] = {}
    for u in units:
        # Unit may be bare province name or province_year — extract province.
        province = u.split("_", 1)[0]
        files = by_prov.get(province, [])
        out[u] = RawVolumeEntry(
            unit_name=u,
            n_files=len(files),
            n_bytes=sum(p.stat().st_size for p in files),
            layout_hint=LAYOUT,
            n_source_docs=rows_by_prov.get(province, 0),
        )
    return out
