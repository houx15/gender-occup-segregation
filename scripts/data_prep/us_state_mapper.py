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


_LOC_NEWSPAPERS_TXT = "https://www.loc.gov/chroniclingamerica/newspapers.txt"


def _parse_newspapers_txt(text: str) -> List[dict]:
    """Parse the Chronicling America bulk title list into [{lccn, state}, ...].

    Pipe-delimited, verified header (2026-08-11):
        Newspapers|LCCN|OCLC|ISSN|State|County|City|Geo Location|...
    LCCN is column index 1, State is column index 4. The header line and any
    row with fewer than 5 columns are skipped.
    """
    records: List[dict] = []
    for line in text.splitlines():
        if not line or line.startswith("Newspapers|"):
            continue
        cols = line.split("|")
        if len(cols) < 5:
            continue
        lccn = cols[1].strip()
        state = cols[4].strip()
        if lccn and state:
            records.append({"lccn": lccn, "state": state})
    return records


def _fetch_loc_directory_records(retries: int = 4) -> List[dict]:
    """Fetch LCCN->state records from Chronicling America's bulk title list.

    Downloads the single pipe-delimited file at ``_LOC_NEWSPAPERS_TXT`` (one
    request for every digitized title) and parses it via ``_parse_newspapers_txt``.
    American Stories is derived from Chronicling America scans, so this covers
    exactly the titles the article ids reference; one flat file avoids the
    paginated, truncation-prone JSON API.

    Network step — run where the node has internet. The pure parser is factored
    out so it stays unit-testable without network.
    """
    import time

    import requests

    last_err: "Exception | None" = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(_LOC_NEWSPAPERS_TXT, timeout=180)
            resp.raise_for_status()
            return _parse_newspapers_txt(resp.text)
        except requests.RequestException as e:
            last_err = e
            if attempt == retries:
                raise
            time.sleep(2 * attempt)  # backoff on transient failures
    raise last_err  # pragma: no cover - loop returns or raises


def build(config: str = "config/config.yml", titles_file: Optional[str] = None) -> None:
    """Build the LCCN->state table from Chronicling America's title list.

    Source precedence:
      1. --titles_file (CLI), or config us_states.loc_titles_file: a LOCAL copy
         of the pipe-delimited title list (download it in a browser and place it
         on the server — loc.gov blocks programmatic user-agents).
      2. otherwise, fetch _LOC_NEWSPAPERS_TXT over the network.
    """
    cfg = load_config(config)
    logger = setup_logging(Path(cfg["paths"]["log_dir"]), "us_state_mapper.log")
    raw_dir = Path(cfg["paths"]["raw_data_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)  # ensure raw dir exists before save
    out = str(raw_dir / "lccn_state_table.json")

    local = titles_file or cfg.get("us_states", {}).get("loc_titles_file")
    if local:
        p = Path(local)
        if not p.exists():
            raise FileNotFoundError(f"loc_titles_file not found: {p}")
        logger.info(f"Reading local Chronicling America title list: {p}")
        records = _parse_newspapers_txt(p.read_text(encoding="utf-8", errors="replace"))
    else:
        logger.info(f"Fetching Chronicling America title list: {_LOC_NEWSPAPERS_TXT}")
        records = _fetch_loc_directory_records()

    logger.info(f"Parsed {len(records)} title records")
    if not records:
        raise RuntimeError(
            "No LCCN->state records parsed. If reading a local file, check it is "
            "the pipe-delimited list with header 'Newspapers|LCCN|OCLC|ISSN|State|...' "
            f"(LCCN col 2, State col 5). Source: {local or _LOC_NEWSPAPERS_TXT}"
        )
    table = build_lccn_state_table(records)
    save_lccn_state_table(table, out)
    logger.info(f"Wrote {len(table)} LCCN->state entries to {out}")


if __name__ == "__main__":
    fire.Fire({"build": build})
