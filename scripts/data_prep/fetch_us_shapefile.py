#!/usr/bin/env python3
"""Fetch the US Census cartographic-boundary states shapefile (cb 20m).

Downloads the public zip and extracts it to data/shapefiles/, renaming the
layer to us_states.*. Public-domain Census data. Network step.

Usage:
  python -m scripts.data_prep.fetch_us_shapefile
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import fire

_URL = "https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_state_20m.zip"


def main(out_dir: str = "data/shapefiles") -> None:
    import requests

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    resp = requests.get(_URL, timeout=120)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        z.extractall(out)
    for src in out.glob("cb_2018_us_state_20m.*"):
        dst = out / ("us_states" + src.suffix)
        src.replace(dst)
    print(f"US states shapefile -> {out/'us_states.shp'}")


if __name__ == "__main__":
    fire.Fire(main)
