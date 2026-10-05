#!/usr/bin/env python3
"""Family-side survey measures per state-window from the IPUMS ACS family extract.

OFFLINE (Slurm) step after download_ipums_acs --config=config/ipums_acs_family.yml.
Adults 25-54, person weights (PERWT) throughout. Per (state, window):

  motherhood_emp_gap    employment rate, childless women - mothers of children < 5
  motherhood_hours_gap  usual weekly hours, employed childless women - employed
                        mothers of children < 5
  married_women_nilf    share of married women (spouse present) not in the labor force
  wife_earnings_share   mean wife's share of couple wage income (INCWAGE /
                        (INCWAGE + INCWAGE_SP), couples with positive income)
  wife_earns_more       share of those couples where the wife earns more
  gender_emp_gap        employment rate, men - women

Higher = more traditional, except wife_earnings_share / wife_earns_more.
Spouse sex is not in the extract, so same-sex couples are not excluded from
the couple measures (a small share of married couples).

Stage 1 sums weighted counts per (YEAR, STATEFIP) in chunks (cached as
year_state_stats.parquet); stage 2 sums them over the same time windows as the
text units (scripts/common/periods.py) -> family_measures_WIDTHy_stepSTEP.csv.

Usage:
  python -m scripts.data_prep.build_family_measures --config=config/ipums_acs_family.yml --width=10 --step=5
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import fire
import numpy as np
import pandas as pd
import yaml

from scripts.common.periods import window_label_map
from scripts.data_prep.build_occupation_shares import ddi_labels, find_extracts, sources_changed

USECOLS = ["YEAR", "STATEFIP", "PERWT", "NCHILD", "NCHLT5", "SEX", "MARST",
           "EMPSTAT", "LABFORCE", "UHRSWORK", "INCWAGE", "INCWAGE_SP"]
MALE, FEMALE = 1, 2
EMPLOYED = 1          # EMPSTAT
NOT_IN_LF = 1         # LABFORCE: 1 = no, 2 = yes
MARRIED_SP = 1        # MARST: married, spouse present
WAGE_NA = 999998      # INCWAGE >= this is missing / N/A


def _stats(c: pd.DataFrame) -> pd.DataFrame:
    w = c["PERWT"]
    woman, man = c["SEX"] == FEMALE, c["SEX"] == MALE
    emp = c["EMPSTAT"] == EMPLOYED
    mom5 = woman & (c["NCHLT5"] > 0)
    childless = woman & (c["NCHILD"] == 0)
    wife = woman & (c["MARST"] == MARRIED_SP)
    own, sp = c["INCWAGE"], c["INCWAGE_SP"]
    valid = wife & own.notna() & sp.notna() & (own < WAGE_NA) & (sp < WAGE_NA) & ((own + sp) > 0)
    share = (own / (own + sp)).where(valid, 0.0)
    cols = {
        "w_mom5": w * mom5, "w_mom5_emp": w * (mom5 & emp),
        "w_childless": w * childless, "w_childless_emp": w * (childless & emp),
        "w_mom5_emp_hours": w * (mom5 & emp) * c["UHRSWORK"],
        "w_childless_emp_hours": w * (childless & emp) * c["UHRSWORK"],
        "w_wife": w * wife, "w_wife_nilf": w * (wife & (c["LABFORCE"] == NOT_IN_LF)),
        "w_couple": w * valid, "w_couple_share": w * share,
        "w_couple_wife_more": w * (valid & (own > sp)),
        "w_men": w * man, "w_men_emp": w * (man & emp),
        "w_women": w * woman, "w_women_emp": w * (woman & emp),
        "n_persons": pd.Series(1, index=c.index),  # unweighted respondents
    }
    out = pd.DataFrame(cols)
    out[["YEAR", "STATEFIP"]] = c[["YEAR", "STATEFIP"]]
    return out.groupby(["YEAR", "STATEFIP"]).sum()


def year_state_stats(path: Path, chunksize: int = 2_000_000) -> pd.DataFrame:
    parts = [_stats(c) for c in pd.read_csv(path, usecols=USECOLS, chunksize=chunksize)]
    return pd.concat(parts).groupby(level=[0, 1]).sum().reset_index()


def family_measures(stats: pd.DataFrame, period_start: int, width: int,
                    step: Optional[int] = None) -> pd.DataFrame:
    last = int(stats["YEAR"].max())
    labels = window_label_map(list(range(period_start, last + 1)), width, step)
    d = stats[stats["YEAR"].isin(labels)].copy()
    d["period"] = d["YEAR"].map(labels)
    s = (d.explode("period").astype({"period": int}).drop(columns="YEAR")
         .groupby(["STATEFIP", "period"]).sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        out = pd.DataFrame({
            "motherhood_emp_gap": s.w_childless_emp / s.w_childless - s.w_mom5_emp / s.w_mom5,
            "motherhood_hours_gap": s.w_childless_emp_hours / s.w_childless_emp
                                    - s.w_mom5_emp_hours / s.w_mom5_emp,
            "married_women_nilf": s.w_wife_nilf / s.w_wife,
            "wife_earnings_share": s.w_couple_share / s.w_couple,
            "wife_earns_more": s.w_couple_wife_more / s.w_couple,
            "gender_emp_gap": s.w_men_emp / s.w_men - s.w_women_emp / s.w_women,
            "weighted_n_women": s.w_women,
            "n_persons": s.n_persons,
        })
    return out.reset_index()


def main(config: str = "config/ipums_acs_family.yml", width: int = 10, step: int = 5,
         period_start: int = 2005, rebuild: bool = False) -> None:
    cfg = yaml.safe_load(open(config))
    out = Path(cfg["out_dir"])
    cache = out / "year_state_stats.parquet"
    extracts = find_extracts(out)
    if not extracts:
        raise SystemExit(f"no usa_*.csv.gz under {out}")
    if cache.exists() and not rebuild and not sources_changed(out, extracts):
        stats = pd.read_parquet(cache)
    else:
        # every extract (2005-2024 + early_2000_2004/); rebuilt when the set changes
        stats = (pd.concat([year_state_stats(e) for e in extracts])
                 .groupby(["YEAR", "STATEFIP"]).sum().reset_index())
        stats.to_parquet(cache, index=False)
        (out / "aggregate_sources.txt").write_text("\n".join(str(e) for e in extracts))
    m = family_measures(stats, period_start, width, step)
    states = ddi_labels(sorted(out.glob("usa_*.xml"))[0], "STATEFIP")
    m.insert(1, "state", m["STATEFIP"].map(states))
    dest = out / f"family_measures_{width}y_step{step}.csv"
    m.to_csv(dest, index=False)
    print(f"{dest.name}: {len(m)} state-window rows, periods {sorted(m['period'].unique())}")
    print(m.groupby("period").mean(numeric_only=True).round(3).drop(columns="STATEFIP").to_string())


if __name__ == "__main__":
    fire.Fire(main)
