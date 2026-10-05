#!/usr/bin/env python3
"""Housework and care split per state-window from IPUMS ATUS (objective family).

OFFLINE (Slurm) step after download_ipums_acs --config=config/ipums_atus.yml.
ATUS respondents aged 25-54, one diary day each; minutes from the BLS time-use
categories; final weights WT06, except WT20 for 2020 (BLS guidance for the
pandemic year). Per (state, window):

  women_share_household          women's / (women's + men's) mean minutes of
                                 household activities (BLS_HHACT)   [main]
  women_share_housework          same for core housework (BLS_HHACT_HWORK)
  women_share_childcare_parents  same for caring for household children
                                 (BLS_CAREHH_KID), respondents with own child < 18
  n_respondents                  unweighted respondents (state cells are small)

Higher = more traditional (women do a larger share).

Usage:
  python -m scripts.data_prep.build_housework_measures --width=10 --step=5 --period_start=1995
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import fire
import numpy as np
import pandas as pd
import yaml

from scripts.common.periods import window_label_map
from scripts.data_prep.build_occupation_shares import ddi_labels
from scripts.data_prep.us_state_mapper import normalize_state, unit_state

ACTIVITIES = {"household": "BLS_HHACT", "housework": "BLS_HHACT_HWORK",
              "childcare_parents": "BLS_CAREHH_KID"}


def _weight(c: pd.DataFrame) -> pd.Series:
    use20 = (c["YEAR"] == 2020) & (c.get("WT20", 0) > 0)
    return c["WT06"].where(~use20, c.get("WT20", 0)).clip(lower=0)


def year_state_stats(path: Path) -> pd.DataFrame:
    c = pd.read_csv(path)
    c = c[c["AGE"].between(25, 54)].copy()
    c["w"] = _weight(c)
    woman, man = c["SEX"] == 2, c["SEX"] == 1
    parent = c["HH_CHILD"] == 1
    cols = {"n": pd.Series(1, index=c.index)}
    for name, var in ACTIVITIES.items():
        sel = parent if name == "childcare_parents" else pd.Series(True, index=c.index)
        mins = c[var].where(c[var] < 1440, np.nan).fillna(0)   # minutes in a day
        for sex, mask in (("women", woman), ("men", man)):
            cols[f"w_{sex}_{name}"] = c["w"] * (mask & sel)
            cols[f"m_{sex}_{name}"] = c["w"] * (mask & sel) * mins
    out = pd.DataFrame(cols)
    out[["YEAR", "STATEFIP"]] = c[["YEAR", "STATEFIP"]]
    return out.groupby(["YEAR", "STATEFIP"]).sum().reset_index()


def housework_measures(stats: pd.DataFrame, period_start: int, width: int,
                       step: Optional[int] = None) -> pd.DataFrame:
    labels = window_label_map(list(range(period_start, int(stats["YEAR"].max()) + 1)), width, step)
    d = stats[stats["YEAR"].isin(labels)].copy()
    d["period"] = d["YEAR"].map(labels)
    s = (d.explode("period").astype({"period": int}).drop(columns="YEAR")
         .groupby(["STATEFIP", "period"]).sum())
    out = pd.DataFrame(index=s.index)
    with np.errstate(divide="ignore", invalid="ignore"):
        for name in ACTIVITIES:
            women = s[f"m_women_{name}"] / s[f"w_women_{name}"]
            men = s[f"m_men_{name}"] / s[f"w_men_{name}"]
            out[f"women_share_{name}"] = women / (women + men)
            out[f"women_minutes_{name}"] = women
            out[f"men_minutes_{name}"] = men
    out["n_respondents"] = s["n"]
    return out.reset_index()


def main(config: str = "config/ipums_atus.yml", width: int = 10, step: int = 5,
         period_start: int = 1995) -> None:
    cfg = yaml.safe_load(open(config))
    out = Path(cfg["out_dir"])
    extracts = sorted(out.glob("atus_*.csv.gz"))
    if len(extracts) != 1:
        raise SystemExit(f"expected one atus_*.csv.gz in {out}, found {extracts}")
    stats = year_state_stats(extracts[0])
    m = housework_measures(stats, period_start, width, step)
    labels = ddi_labels(sorted(out.glob("atus_*.xml"))[0], "STATEFIP")
    m.insert(1, "state", m["STATEFIP"].map(lambda f: unit_state(normalize_state(labels[f]))))
    dest = out / f"housework_measures_{width}y_step{step}.csv"
    m.to_csv(dest, index=False)
    print(f"{dest.name}: {len(m)} state-window rows")
    print(m.groupby("period")[["women_share_household", "women_share_housework",
                               "women_share_childcare_parents", "n_respondents"]]
          .agg(["mean", "min"]).round(3).to_string())


if __name__ == "__main__":
    fire.Fire(main)
