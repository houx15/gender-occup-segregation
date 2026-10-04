#!/usr/bin/env python3
"""Census female shares by occupation word, national and by state, per period.

OFFLINE (Slurm) step after download_ipums_acs. Two stages:

1. aggregate: stream the IPUMS ACS extract (employed persons) in chunks and
   sum person weights by (YEAR, STATEFIP, OCC2010) for women and in total
   -> <out_dir>/occ2010_sex_state_year.parquet (small; built once).
2. shares: map each list word to its OCC2010 codes (mapping CSV: word,
   occ2010 = '3255; 3256') and sum over codes and over the years of each
   period (us_states.year_bins wide, aligned to period_start)
   -> <out_dir>/occupation_female_share_{national,state}.csv

Usage:
  python -m scripts.data_prep.build_occupation_shares --config=config/ipums_acs.yml
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import fire
import pandas as pd
import yaml

USECOLS = ["YEAR", "STATEFIP", "SEX", "OCC2010", "PERWT"]
FEMALE = 2  # IPUMS SEX code


def aggregate_extract(path: Path, chunksize: int = 2_000_000) -> pd.DataFrame:
    """YEAR, STATEFIP, OCC2010, female, total (sums of PERWT)."""
    parts = []
    for chunk in pd.read_csv(path, usecols=USECOLS, chunksize=chunksize):
        chunk["female"] = chunk["PERWT"].where(chunk["SEX"] == FEMALE, 0)
        parts.append(chunk.groupby(["YEAR", "STATEFIP", "OCC2010"])[["female", "PERWT"]].sum())
    agg = pd.concat(parts).groupby(level=[0, 1, 2]).sum()
    return agg.rename(columns={"PERWT": "total"}).reset_index()


def parse_occ_codes(cell) -> List[int]:
    return [int(c) for c in str(cell).replace(",", ";").split(";") if c.strip()] \
        if pd.notna(cell) else []


def word_female_shares(agg: pd.DataFrame, mapping: pd.DataFrame, period_start: int,
                       width: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """National (word, period) and state (word, STATEFIP, period) female shares."""
    pairs = [(row.word, code) for row in mapping.itertuples()
             for code in parse_occ_codes(row.occ2010)]
    link = pd.DataFrame(pairs, columns=["word", "OCC2010"])
    d = agg.merge(link, on="OCC2010")
    d["period"] = period_start + (d["YEAR"] - period_start) // width * width
    d = d[d["YEAR"] >= period_start]

    def _share(keys):
        g = d.groupby(keys)[["female", "total"]].sum().reset_index()
        g["female_share"] = g["female"] / g["total"]
        return g.rename(columns={"total": "weighted_n"}).drop(columns="female")

    return _share(["word", "period"]), _share(["word", "STATEFIP", "period"])


def main(config: str = "config/ipums_acs.yml", rebuild: bool = False) -> None:
    cfg = yaml.safe_load(open(config))
    out = Path(cfg["out_dir"])
    agg_path = out / "occ2010_sex_state_year.parquet"
    if agg_path.exists() and not rebuild:
        agg = pd.read_parquet(agg_path)
    else:
        extracts = sorted(out.glob("usa_*.csv.gz"))
        if len(extracts) != 1:
            raise SystemExit(f"expected one usa_*.csv.gz in {out}, found {extracts}")
        agg = aggregate_extract(extracts[0])
        agg.to_parquet(agg_path, index=False)
    print(f"aggregate: {len(agg)} (year, state, occ2010) cells, years "
          f"{agg['YEAR'].min()}-{agg['YEAR'].max()}")

    mapping = pd.read_csv(cfg["occupation_mapping"])
    nat, state = word_female_shares(agg, mapping, int(cfg["period_start"]),
                                    int(cfg["period_width"]))
    nat.to_csv(out / "occupation_female_share_national.csv", index=False)
    state.to_csv(out / "occupation_female_share_state.csv", index=False)
    print(f"wrote shares for {nat['word'].nunique()} words: "
          f"{len(nat)} national, {len(state)} state rows")


if __name__ == "__main__":
    fire.Fire(main)
