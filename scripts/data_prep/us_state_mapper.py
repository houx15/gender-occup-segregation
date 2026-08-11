#!/usr/bin/env python3
"""US state identity: normalize names, extract LCCNs, map LCCN -> state.

Single source of truth for state identity across both US arms and the
choropleth. Arm B (3DLNews2) uses only ``normalize_state`` (its ``location.state``
is authoritative); Arm A (American Stories) additionally maps each article's
LCCN to a publisher state via a table built from the LoC US Newspaper Directory.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import fire

from scripts.common.config_loader import load_config
from scripts.common.logging_utils import setup_logging

# Canonical name spelling matches the Census cb_20m shapefile NAME column.
_STATE_NAME_TO_USPS: Dict[str, str] = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "District of Columbia": "DC", "Florida": "FL", "Georgia": "GA", "Hawaii": "HI",
    "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
    "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI",
    "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX",
    "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
    "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
}
_USPS_TO_STATE_NAME: Dict[str, str] = {v: k for k, v in _STATE_NAME_TO_USPS.items()}
_LOWER_NAME_TO_CANON: Dict[str, str] = {k.lower(): k for k in _STATE_NAME_TO_USPS}

_LCCN_RE = re.compile(r"(sn\d{8}|\d{10})")


def normalize_state(raw: str) -> Optional[str]:
    """Full name / USPS 2-letter / messy case -> canonical Title-Case name."""
    if not raw or not isinstance(raw, str):
        return None
    s = raw.strip()
    if not s:
        return None
    up = s.upper()
    if up in _USPS_TO_STATE_NAME:
        return _USPS_TO_STATE_NAME[up]
    return _LOWER_NAME_TO_CANON.get(s.lower())


def unit_state(state_name: str) -> str:
    """Canonical name -> unit token: 'New York' -> 'new_york'."""
    return state_name.strip().lower().replace(" ", "_")


def lccn_from_article_id(article_id: str) -> Optional[str]:
    """Extract the LCCN (sn######## or a 10-digit id) embedded in an id string."""
    if not article_id:
        return None
    m = _LCCN_RE.search(str(article_id))
    return m.group(1) if m else None


def build_lccn_state_table(directory_records: List[dict]) -> Dict[str, str]:
    """LoC directory records -> {lccn: canonical_state}. Drops unknown states."""
    table: Dict[str, str] = {}
    for rec in directory_records:
        lccn = (rec.get("lccn") or "").strip()
        state = normalize_state(rec.get("state") or "")
        if lccn and state:
            table[lccn] = state
    return table


def resolve_state(lccn: str, table: Dict[str, str]) -> Optional[str]:
    return table.get(lccn) if lccn else None


def save_lccn_state_table(table: Dict[str, str], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(table, f, ensure_ascii=False, indent=0, sort_keys=True)


def load_lccn_state_table(path: str) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _fetch_loc_directory_records(retries: int = 4) -> List[dict]:
    """Fetch LCCN->state records: [{lccn, state}, ...] from Chronicling America.

    Uses https://chroniclingamerica.loc.gov/newspapers.json — a single,
    well-formed listing of every digitized title with its ``lccn`` and ``state``.
    American Stories is derived from Chronicling America scans, so this covers
    exactly the titles the article ids reference, and it avoids the flaky,
    truncation-prone pagination of the loc.gov directory collection API.

    Network step — run where the node has internet. Kept out of the pure-logic
    functions above so they stay unit-testable without network.
    """
    import time

    import requests

    url = "https://chroniclingamerica.loc.gov/newspapers.json"
    last_err: "Exception | None" = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, timeout=180)
            resp.raise_for_status()
            data = resp.json()
            break
        except (requests.RequestException, ValueError) as e:
            last_err = e
            if attempt == retries:
                raise
            time.sleep(2 * attempt)  # simple backoff on transient/truncated responses
    else:  # pragma: no cover - loop always breaks or raises
        raise last_err  # type: ignore[misc]

    records: List[dict] = []
    for n in data.get("newspapers", []):
        lccn = (n.get("lccn") or "").strip()
        state = (n.get("state") or "").strip()
        if lccn and state:
            records.append({"lccn": lccn, "state": state})
    return records


def build(config: str = "config/config.yml") -> None:
    """Fetch the LoC directory and write the LCCN->state table to raw_data_dir."""
    cfg = load_config(config)
    logger = setup_logging(Path(cfg["paths"]["log_dir"]), "us_state_mapper.log")
    out = f"{cfg['paths']['raw_data_dir']}/lccn_state_table.json"
    logger.info("Fetching Chronicling America newspapers.json (LCCN->state)...")
    records = _fetch_loc_directory_records()
    logger.info(f"Fetched {len(records)} title records")
    table = build_lccn_state_table(records)
    save_lccn_state_table(table, out)
    logger.info(f"Wrote {len(table)} LCCN->state entries to {out}")


if __name__ == "__main__":
    fire.Fire({"build": build})
