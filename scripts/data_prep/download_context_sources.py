#!/usr/bin/env python3
"""Download external state-context sources for Part II-B.

  BEA regional bulk files   SAGDP.zip (state GDP), SAINC.zip (personal income
                            incl. population)  https://apps.bea.gov/regional/zip/
  Correlates of State Policy Project v2.6 (IPPSR, Michigan State University)
                            correlates2-6.csv + codebook_2-6.csv
  Berry, Ringquist, Fording & Hanson citizen / government ideology, v2018
                            stateideology_v2018.dta (zip)

NETWORK STEP (login node; files are small):
  python -m scripts.data_prep.download_context_sources \
      --out_dir=/scratch/network/yh6580/gender-occup/data/context_sources
"""

from __future__ import annotations

from pathlib import Path

import fire
import requests

FILES = {
    "SAGDP.zip": "https://apps.bea.gov/regional/zip/SAGDP.zip",
    "SAINC.zip": "https://apps.bea.gov/regional/zip/SAINC.zip",
    "correlates2-6.csv": "https://ippsr.msu.edu/sites/default/files/cspp/correlates2-6.csv",
    "codebook_2-6.csv": "https://ippsr.msu.edu/sites/default/files/cspp/codebook_2-6.csv",
    "stateideology_v2018.dta.zip": "https://rcfording.files.wordpress.com/2020/02/stateideology_v2018.dta_.zip",
}


def main(out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        dest = out / name
        if dest.exists():
            print(f"  have {name}")
            continue
        r = requests.get(url, timeout=300, headers={"User-Agent": "Mozilla/5.0 (research download)"})
        r.raise_for_status()
        dest.write_bytes(r.content)
        print(f"  downloaded {name} ({len(r.content) / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
