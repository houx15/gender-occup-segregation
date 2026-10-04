#!/usr/bin/env python3
"""Subjective gender-attitude measures per state-window from Project Implicit.

OFFLINE (Slurm) step after download_project_implicit. Gender-Career IAT,
2005-2024, US respondents with a valid state (STATE, derived from ZIP code).
Per (state, window):

  iat_*       implicit association of men with career and women with family
              (IAT D score, D_biep.Male_Career_all; > 0 = stereotypic)
  explicit_*  self-reported stereotype: assocareer - assofamily (each 1 =
              strongly female ... 7 = strongly male; > 0 = career male, family female)

Each comes raw (respondent mean) and sex-balanced (mean of the women's and
men's means), because volunteers are mostly women. Respondents are
self-selected, not a probability sample; fuller post-stratification (age x
education to ACS) is a possible refinement.

Sex codings over the years: sex 'f'/'m' (to 2016), birthsex 1 male / 2 female
(2016-2023), genderIdentity_0002 1 man / 2 woman (2023-).

Usage:
  python -m scripts.data_prep.build_attitude_measures --config=config/project_implicit.yml --width=10 --step=5
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import Optional

import fire
import numpy as np
import pandas as pd
import yaml

from scripts.common.periods import window_label_map

COLUMNS = ["year", "STATE", "sex", "birthsex", "genderIdentity_0002",
           "D_biep.Male_Career_all", "assocareer", "assofamily"]
US_STATES = set(
    "AL AK AZ AR CA CO CT DE DC FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE "
    "NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY".split())


def read_year_file(path: Path, tmp_dir: Path) -> pd.DataFrame:
    """Needed columns from one yearly zip (CSV, or SPSS .sav)."""
    with zipfile.ZipFile(path) as z:
        members = [m for m in z.namelist()
                   if not m.startswith("__MACOSX") and m.lower().endswith((".csv", ".sav"))]
        if len(members) != 1:
            raise SystemExit(f"{path.name}: expected one data file, found {members}")
        member = members[0]
        if member.lower().endswith(".csv"):
            with z.open(member) as f:
                header = pd.read_csv(f, nrows=0, encoding="utf-8-sig").columns
            cols = [c for c in COLUMNS if c in header]
            with z.open(member) as f:
                return pd.read_csv(f, usecols=cols, encoding="utf-8-sig", low_memory=False)[cols]
        import pyreadstat
        sav = Path(z.extract(member, tmp_dir))
    try:
        _, meta = pyreadstat.read_sav(str(sav), metadataonly=True)
        cols = [c for c in COLUMNS if c in meta.column_names]
        df, _ = pyreadstat.read_sav(str(sav), usecols=cols)
        return df[cols]
    finally:
        sav.unlink()


def harmonize(raw: pd.DataFrame, file_year: int) -> pd.DataFrame:
    """year, state, female (1/0/NaN), iat, explicit — US respondents only."""
    d = pd.DataFrame(index=raw.index)
    d["year"] = pd.to_numeric(raw.get("year", file_year), errors="coerce").fillna(file_year)
    d["state"] = raw["STATE"].astype(str).str.strip().str.upper()
    female = pd.Series(np.nan, index=raw.index)
    if "sex" in raw:
        female = female.fillna(raw["sex"].map({"f": 1.0, "m": 0.0}))
    if "birthsex" in raw:
        female = female.fillna(pd.to_numeric(raw["birthsex"], errors="coerce").map({2: 1.0, 1: 0.0}))
    if "genderIdentity_0002" in raw:
        gi = pd.to_numeric(raw["genderIdentity_0002"], errors="coerce")
        female = female.fillna(gi.map({2: 1.0, 1: 0.0}))
    d["female"] = female
    d["iat"] = pd.to_numeric(raw.get("D_biep.Male_Career_all"), errors="coerce")
    d["explicit"] = (pd.to_numeric(raw.get("assocareer"), errors="coerce")
                     - pd.to_numeric(raw.get("assofamily"), errors="coerce"))
    d = d[d["state"].isin(US_STATES)]
    return d.astype({"year": int}).reset_index(drop=True)


def attitude_measures(h: pd.DataFrame, period_start: int, width: int,
                      step: Optional[int] = None) -> pd.DataFrame:
    labels = window_label_map(list(range(period_start, int(h["year"].max()) + 1)), width, step)
    d = h[h["year"].isin(labels)].copy()
    d["period"] = d["year"].map(labels)
    d = d.explode("period").astype({"period": int})
    rows = []
    for (state, period), g in d.groupby(["state", "period"]):
        row = {"state": state, "period": period, "n": len(g),
               "share_female": g["female"].mean()}
        for v in ("iat", "explicit"):
            row[f"{v}_mean"] = g[v].mean()
            by_sex = g.groupby("female")[v].mean()
            row[f"{v}_sex_balanced"] = (by_sex.get(1.0, np.nan) + by_sex.get(0.0, np.nan)) / 2
            row[f"n_{v}"] = int(g[v].notna().sum())
        rows.append(row)
    return pd.DataFrame(rows)


def main(config: str = "config/project_implicit.yml", width: int = 10, step: int = 5,
         period_start: int = 2005) -> None:
    cfg = yaml.safe_load(open(config))
    out = Path(cfg["out_dir"])
    cache = out / "respondents_us.parquet"
    if cache.exists():
        h = pd.read_parquet(cache)
    else:
        parts = []
        for path in sorted((out / "raw").glob("*.zip")):
            m = re.search(r"public\.(\d{4})", path.name)
            if not m or re.search(r"public\.\d{4}-\d{4}", path.name):
                continue
            part = harmonize(read_year_file(path, out), int(m.group(1)))
            print(f"  {path.name}: {len(part)} US respondents", flush=True)
            parts.append(part)
        h = pd.concat(parts, ignore_index=True)
        h.to_parquet(cache, index=False)
    m = attitude_measures(h, period_start, width, step)
    dest = out / f"attitude_measures_{width}y_step{step}.csv"
    m.to_csv(dest, index=False)
    print(f"{dest.name}: {len(m)} state-window rows; respondents per row median "
          f"{int(m['n'].median())}, min {int(m['n'].min())}")
    print(m.groupby("period")[["n", "iat_sex_balanced", "explicit_sex_balanced"]]
          .agg(["median", "mean"]).round(3).to_string())


if __name__ == "__main__":
    fire.Fire(main)
