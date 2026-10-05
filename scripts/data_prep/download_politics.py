#!/usr/bin/env python3
"""Download state presidential returns 1976-2024 (MIT Election Data and Science
Lab, Harvard Dataverse doi:10.7910/DVN/42MVDX, file 1976-2024-president.csv).

NETWORK STEP (login node):
  python -m scripts.data_prep.download_politics \
      --out_dir=/scratch/network/yh6580/gender-occup/data/politics
"""

from pathlib import Path

import fire
import requests

URL = "https://dataverse.harvard.edu/api/access/datafile/13887042"


def main(out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / "1976-2024-president.csv"
    r = requests.get(URL, timeout=120)
    r.raise_for_status()
    dest.write_bytes(r.content)
    print(f"wrote {dest} ({len(r.content) / 1e6:.1f} MB)")


if __name__ == "__main__":
    fire.Fire(main)
