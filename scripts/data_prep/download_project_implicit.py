#!/usr/bin/env python3
"""Download Project Implicit Gender-Career IAT yearly files from OSF.

Subjective (attitude) benchmark for the state-window comparison: implicit
career-family stereotype (IAT D score) and explicit career / family gender
associations, with the respondent's US state. Data: Gender-Career IAT
2005-2025, OSF project abxq7, files in component gmewy.

Picks one file per year: the CSV zip when published, else the SPSS zip (some
years, e.g. 2011 and 2022-2024, have no CSV). Multi-year bundles are skipped.

NETWORK STEP — run on the Adroit login node (tmux for the ~850 MB):
  python -m scripts.data_prep.download_project_implicit --config=config/project_implicit.yml
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

import fire
import requests
import yaml

OSF_API = "https://api.osf.io/v2/nodes/{node}/files/osfstorage/"


def select_year_files(names: List[str], years: List[int]) -> Dict[int, str]:
    """year -> file name; CSV zip preferred over the SPSS zip."""
    out = {}
    for y in years:
        single = [n for n in names
                  if re.search(rf"public\.{y}(\b|[-.])", n) and n.endswith(".zip")
                  and not re.search(rf"public\.\d{{4}}-\d{{4}}", n)]
        csv = [n for n in single if "csv" in n.lower()]
        pick = sorted(csv)[:1] or sorted(n for n in single if "sav" not in n.lower())[:1] \
            or sorted(single)[:1]
        if not pick:
            raise SystemExit(f"no Gender-Career IAT file for {y} on OSF")
        out[y] = pick[0]
    return out


def _list_files(node: str) -> Dict[str, str]:
    url, files = OSF_API.format(node=node), {}
    while url:
        d = requests.get(url, timeout=60).json()
        for f in d["data"]:
            files[f["attributes"]["name"]] = f["links"]["download"]
        url = d["links"].get("next")
    return files


def _download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  have {dest.name}")
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=8 << 20):
                f.write(chunk)
    tmp.rename(dest)
    print(f"  downloaded {dest.name} ({dest.stat().st_size / 1e6:.0f} MB)", flush=True)


def main(config: str = "config/project_implicit.yml") -> None:
    cfg = yaml.safe_load(open(config))
    raw = Path(cfg["out_dir"]) / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    files = _list_files(cfg["osf_node"])
    picks = select_year_files(list(files), cfg["years"])
    for name in [cfg["codebook"]] + [picks[y] for y in sorted(picks)]:
        _download(files[name], raw / name)
    print(f"Done. {len(picks)} yearly files in {raw}")


if __name__ == "__main__":
    fire.Fire(main)
