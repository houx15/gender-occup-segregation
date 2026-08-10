#!/usr/bin/env python3
"""Sanity-check ``lccn_from_article_id`` against real American Stories article ids.

Samples article ids from one year — an already-downloaded raw jsonl if present,
otherwise a fresh HuggingFace pull — applies ``lccn_from_article_id``, and
reports the extraction rate with examples of both hits and misses. If the LoC
LCCN->state table has been built, it also reports how many extracted LCCNs
resolve to a state (the end-to-end check that actually matters for the arm).

This is a diagnostic, run once before committing to the full American Stories
build. If the miss rate is high or the resolved rate is low, the regex in
``us_state_mapper.lccn_from_article_id`` (or the LoC table build) needs a look.

Usage (login/internet node, or after download_american_stories):
  python -m scripts.data_prep.check_lccn_regex \
      --config=config/profiles/garg_weat_american_stories.yml --year=1940 --n=500
"""

from __future__ import annotations

import json
from pathlib import Path

import fire

from scripts.common.config_loader import load_config
from scripts.data_prep import us_state_mapper as usm


def _ids_from_raw(path: str, n: int) -> list[str]:
    ids: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            aid = rec.get("article_id", "")
            if aid:
                ids.append(aid)
            if len(ids) >= n:
                break
    return ids


def _ids_from_hf(year: int, n: int) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset(
        "dell-research-harvard/AmericanStories",
        "subset_years",
        year_list=[str(year)],
        trust_remote_code=True,
    )
    ids: list[str] = []
    for split in ds:
        for row in ds[split]:
            aid = row.get("article_id", "")
            if aid:
                ids.append(aid)
            if len(ids) >= n:
                return ids
    return ids


def main(
    config: str = "config/profiles/garg_weat_american_stories.yml",
    year: int = 1940,
    n: int = 500,
) -> None:
    """Report LCCN extraction + state-resolution rates for one year's ids."""
    cfg = load_config(config)
    raw_dir = Path(cfg["paths"]["raw_data_dir"])
    raw = raw_dir / f"american_stories_{year}.jsonl"

    if raw.exists() and raw.stat().st_size > 0:
        print(f"[source] existing raw file: {raw}")
        ids = _ids_from_raw(str(raw), n)
    else:
        print(f"[source] no raw file at {raw}")
        print(f"[source] pulling year {year} from HuggingFace (NEEDS INTERNET)...")
        ids = _ids_from_hf(year, n)

    total = len(ids)
    if not total:
        print("no article ids sampled — nothing to check")
        return

    matched = [(aid, usm.lccn_from_article_id(aid)) for aid in ids]
    hits = [(a, l) for a, l in matched if l]
    misses = [a for a, l in matched if not l]

    print(f"\nsampled {total} article ids from year {year}")
    print(f"  LCCN extracted: {len(hits)} ({100 * len(hits) / total:.1f}%)")
    print(f"  no LCCN:        {len(misses)} ({100 * len(misses) / total:.1f}%)")

    print("\n-- examples: LCCN extracted --")
    for aid, lccn in hits[:10]:
        print(f"   {aid!r:48} -> {lccn}")
    if misses:
        print("\n-- examples: NO LCCN (regex may need adjusting) --")
        for aid in misses[:10]:
            print(f"   {aid!r}")

    table_path = raw_dir / "lccn_state_table.json"
    if table_path.exists():
        table = usm.load_lccn_state_table(str(table_path))
        resolved = sum(1 for _, lccn in hits if usm.resolve_state(lccn, table))
        pct = 100 * resolved / len(hits) if hits else 0.0
        print(
            f"\nLoC table present ({len(table)} entries): "
            f"{resolved}/{len(hits)} extracted LCCNs resolve to a state ({pct:.1f}%)"
        )
    else:
        print(
            f"\n(no LoC table at {table_path} — run "
            f"`python -m scripts.data_prep.us_state_mapper build --config={config}` "
            "first to also test state resolution)"
        )


if __name__ == "__main__":
    fire.Fire(main)
