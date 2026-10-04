#!/usr/bin/env python3
"""Request and download an IPUMS USA ACS extract defined by a small YAML config.

config/ipums_acs.yml: employed persons, occupation x sex x state (occupation side).
config/ipums_acs_family.yml: adults 25-54 with children, marital status,
employment, hours and own + spouse earnings (family side).

Source for census female shares by occupation (national and state, every
year), replacing Garg's file, which stops at 2015 and is national only.

NETWORK STEP — run on the Adroit login node (compute nodes have no internet),
inside tmux since IPUMS can take a while to build the extract:
  tmux new -s ipums
  python -m scripts.data_prep.download_ipums_acs --config=config/ipums_acs.yml
  python -m scripts.data_prep.download_ipums_acs --config=config/ipums_acs_family.yml

Resumable: the submitted extract number is saved to <out_dir>/extract.json and
re-used on re-runs (no duplicate submissions); finished files are not
re-downloaded. API key: env IPUMS_API_KEY, else ~/.config/ipums/api_key.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional

import fire
import requests
import yaml

API = "https://api.ipums.org/extracts"
PARAMS = {"collection": "usa", "version": 2}
KEY_FILE = Path.home() / ".config" / "ipums" / "api_key"


def read_api_key(env: Mapping[str, str] = os.environ, key_file: Path = KEY_FILE) -> str:
    if env.get("IPUMS_API_KEY"):
        return env["IPUMS_API_KEY"].strip()
    if Path(key_file).is_file():
        return Path(key_file).read_text().strip()
    raise SystemExit(f"No IPUMS API key: set IPUMS_API_KEY or put it in {key_file} (chmod 600)")


def extract_body(years: List[int], variables: List[str],
                 case_selections: Dict[str, List[str]], description: str,
                 attached: Optional[Dict[str, List[str]]] = None) -> dict:
    """IPUMS API v2 request body. ``attached`` adds household members'
    values, e.g. {"INCWAGE": ["spouse"]} -> INCWAGE_SP."""
    attached = attached or {}

    def _spec(v):
        spec = {}
        if v in case_selections:
            spec["caseSelections"] = {"general": case_selections[v]}
        if v in attached:
            spec["attachedCharacteristics"] = attached[v]
        return spec

    return {
        "description": description,
        "dataStructure": {"rectangular": {"on": "P"}},
        "dataFormat": "csv",
        "samples": {f"us{y}a": {} for y in years},
        "variables": {v: _spec(v) for v in variables},
    }


def _download(url: str, dest: Path, headers: dict) -> None:
    if dest.exists():
        print(f"  have {dest.name}")
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, headers=headers, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=8 << 20):
                f.write(chunk)
    tmp.rename(dest)
    print(f"  downloaded {dest} ({dest.stat().st_size / 1e9:.2f} GB)")


def main(config: str = "config/ipums_acs.yml", poll_seconds: int = 60) -> None:
    cfg = yaml.safe_load(open(config))
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    headers = {"Authorization": read_api_key(), "Content-Type": "application/json"}
    state_file = out / "extract.json"

    if state_file.exists():
        number = json.loads(state_file.read_text())["number"]
        print(f"Resuming extract {number}")
    else:
        body = extract_body(cfg["years"], cfg["variables"], cfg.get("case_selections", {}),
                            cfg["description"], cfg.get("attached"))
        r = requests.post(API, params=PARAMS, headers=headers, json=body, timeout=120)
        if r.status_code >= 400:
            raise SystemExit(f"Extract submission failed ({r.status_code}): {r.text}")
        number = r.json()["number"]
        state_file.write_text(json.dumps({"number": number, "body": body}, indent=2))
        print(f"Submitted extract {number}")

    while True:
        info = requests.get(f"{API}/{number}", params=PARAMS, headers=headers, timeout=120).json()
        status = info.get("status")
        print(f"  {time.strftime('%H:%M:%S')} status={status}", flush=True)
        if status == "completed":
            break
        if status in ("failed", "canceled"):
            raise SystemExit(f"Extract {number} {status}: {info.get('errors')}")
        time.sleep(poll_seconds)

    links = info["downloadLinks"]
    for kind in ("ddiCodebook", "data"):
        url = links[kind]["url"]
        _download(url, out / Path(url).name, headers)
    print(f"Done. Files in {out}")


if __name__ == "__main__":
    fire.Fire(main)
