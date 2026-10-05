#!/usr/bin/env python3
"""State context per state-window for Part II-B (explaining state differences).

OFFLINE (Slurm). Sources and blocks (analysis plan 2.4-2.7):

  socioeconomic   (IPUMS context extract, adults 25-64, PERWT)
    real_income_pc       mean INCTOT x CPI99 (1999 dollars)
    ba_share             EDUC >= 10 (4+ years of college)
    metro_share          METRO 2-4 among METRO 1-4 (0 = not identifiable, excluded)
    unemployment_rate    EMPSTAT unemployed / labour force
    manufacturing_share  IND1990 100-392 among the employed
    service_share        IND1990 400-932 among the employed
  gendered labour market
    women_lfp            women in the labour force / women
    women_emp_rate       employed women / women
    gender_wage_gap      1 - mean full-time real wage, women / men
                         (UHRSWORK >= 35, wage income > 0)
    female_share_managers / _professionals   (occupation extract; OCC2010
                         0010-0430 / 0500-3540)
  policy
    pfl_share            share of the window's years with state paid family
                         leave benefits in effect (config/policy/paid_family_leave.csv)
  political
    gop_two_party_share  Republican / (Republican + Democrat) presidential
                         vote, mean over elections in the window (MIT Election Lab)
  external (download_context_sources; window mean over observed years, NaN
  unless >= half the window's years are observed — no carrying forward)
    log_real_gdp_pc      BEA SAGDP9 real GDP (chained 2017 $) / SAINC1 population
    universal_prek, equal_pay_law, so_employment_law, evangelical_lds_share,
    abortion_restrictions (count of 7 restrictions)   CSPP v2.6
    citizen_ideology     Berry et al. citi6016 (higher = more liberal)

Windows match the text units (scripts/common/periods.py).

Usage:
  python -m scripts.data_prep.build_context_measures --width=10 --step=5 --period_start=1995
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import fire
import numpy as np
import pandas as pd
import yaml

from scripts.common.periods import window_label_map, year_windows
from scripts.data_prep.build_occupation_shares import (
    ddi_labels, find_extracts, load_aggregate, sources_changed,
)
from scripts.data_prep.us_state_mapper import normalize_state, unit_state

USECOLS = ["YEAR", "STATEFIP", "PERWT", "SEX", "EDUC", "METRO", "IND1990", "INCTOT",
           "INCWAGE", "CPI99", "UHRSWORK", "EMPSTAT", "LABFORCE"]
INC_NA = 9999998


def _stats(c: pd.DataFrame) -> pd.DataFrame:
    w = c["PERWT"]
    woman, man = c["SEX"] == 2, c["SEX"] == 1
    emp, unemp, inlf = c["EMPSTAT"] == 1, c["EMPSTAT"] == 2, c["LABFORCE"] == 2
    metro_known = c["METRO"].between(1, 4)
    inc_ok = c["INCTOT"] < INC_NA
    real_inc = (c["INCTOT"] * c["CPI99"]).where(inc_ok, 0)
    ft = (c["UHRSWORK"] >= 35) & (c["INCWAGE"] > 0) & (c["INCWAGE"] < 999998)
    real_wage = (c["INCWAGE"] * c["CPI99"]).where(ft, 0)
    cols = {
        "w_all": w, "w_ba": w * (c["EDUC"] >= 10),
        "w_metro_known": w * metro_known, "w_metro": w * (metro_known & c["METRO"].between(2, 4)),
        "w_lf": w * inlf, "w_unemp": w * unemp, "w_emp": w * emp,
        "w_emp_manuf": w * (emp & c["IND1990"].between(100, 392)),
        "w_emp_serv": w * (emp & c["IND1990"].between(400, 932)),
        "w_inc_ok": w * inc_ok, "w_inc": w * real_inc,
        "w_women": w * woman, "w_women_lf": w * (woman & inlf), "w_women_emp": w * (woman & emp),
        "w_ft_women": w * (woman & ft), "w_ft_women_wage": w * real_wage * woman,
        "w_ft_men": w * (man & ft), "w_ft_men_wage": w * real_wage * man,
        "n_persons": pd.Series(1, index=c.index),
    }
    out = pd.DataFrame(cols)
    out[["YEAR", "STATEFIP"]] = c[["YEAR", "STATEFIP"]]
    return out.groupby(["YEAR", "STATEFIP"]).sum()


def year_state_stats(path: Path, chunksize: int = 2_000_000) -> pd.DataFrame:
    parts = [_stats(c) for c in pd.read_csv(path, usecols=USECOLS, chunksize=chunksize)]
    return pd.concat(parts).groupby(level=[0, 1]).sum().reset_index()


def context_measures(stats: pd.DataFrame, period_start: int, width: int,
                     step: Optional[int] = None) -> pd.DataFrame:
    labels = window_label_map(list(range(period_start, int(stats["YEAR"].max()) + 1)), width, step)
    d = stats[stats["YEAR"].isin(labels)].copy()
    d["period"] = d["YEAR"].map(labels)
    s = (d.explode("period").astype({"period": int}).drop(columns="YEAR")
         .groupby(["STATEFIP", "period"]).sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        out = pd.DataFrame({
            "real_income_pc": s.w_inc / s.w_inc_ok,
            "ba_share": s.w_ba / s.w_all,
            "metro_share": s.w_metro / s.w_metro_known,
            "unemployment_rate": s.w_unemp / s.w_lf,
            "manufacturing_share": s.w_emp_manuf / s.w_emp,
            "service_share": s.w_emp_serv / s.w_emp,
            "women_lfp": s.w_women_lf / s.w_women,
            "women_emp_rate": s.w_women_emp / s.w_women,
            "gender_wage_gap": 1 - (s.w_ft_women_wage / s.w_ft_women) / (s.w_ft_men_wage / s.w_ft_men),
            "n_persons_context": s.n_persons,
        })
    return out.reset_index()


def occupation_group_shares(agg: pd.DataFrame, period_start: int, width: int,
                            step: Optional[int]) -> pd.DataFrame:
    """Women's share of managers (OCC2010 10-430) and professionals (500-3540)."""
    labels = window_label_map(list(range(period_start, int(agg["YEAR"].max()) + 1)), width, step)
    d = agg[agg["YEAR"].isin(labels)].copy()
    d["period"] = d["YEAR"].map(labels)
    d = d.explode("period").astype({"period": int})
    rows = {}
    for name, lo, hi in (("female_share_managers", 10, 430),
                         ("female_share_professionals", 500, 3540)):
        g = d[d["OCC2010"].between(lo, hi)].groupby(["STATEFIP", "period"])[["female", "total"]].sum()
        rows[name] = g["female"] / g["total"]
    return pd.DataFrame(rows).reset_index()


# ------------------------------------------------------------------ external sources
ABORTION_RESTRICTIONS = ["fundslife", "infconsent", "gagrule", "medicalrest", "insprivate",
                         "inspublic", "inswaiver"]
CSPP_POLICY = {"universalprek": "universal_prek", "equalpay": "equal_pay_law",
               "solaw": "so_employment_law", "evangldsper": "evangelical_lds_share"}
MIN_STATES_CODED = 40


def _observed(col: pd.Series, year: pd.Series) -> pd.Series:
    """CSPP coding rules: a year counts only if >= 40 states are coded; a variable
    that never takes 0 is a '1 or blank' dummy, so blank = 0 within coded years."""
    coded = col.notna().groupby(year).transform("sum") >= MIN_STATES_CODED
    if col.dropna().isin([0, 1]).all() and not (col == 0).any():
        years = year[col.notna()]
        in_range = year.between(years.min(), years.max()) if len(years) else year.isna()
        return col.fillna(0).where(in_range)
    return col.where(coded)


def cspp_yearly(c: pd.DataFrame, state_key) -> pd.DataFrame:
    out = pd.DataFrame({"state": c["st"].map(state_key), "year": c["year"].astype(int)})
    for src, name in CSPP_POLICY.items():
        out[name] = _observed(c[src], c["year"])
    parts = [_observed(c[v], c["year"]) for v in ABORTION_RESTRICTIONS]
    out["abortion_restrictions"] = sum(parts)   # NaN if any item unobserved that year
    return out


def window_means(yearly: pd.DataFrame, cols: List[str], periods: List[int], width: int) -> pd.DataFrame:
    """Mean over the window's observed years; NaN unless >= half the years are observed."""
    rows = []
    for p in periods:
        g = yearly[(yearly["year"] >= p) & (yearly["year"] < p + width)]
        for st, gg in g.groupby("state"):
            row = {"state": st, "period": p}
            for c in cols:
                v = gg[c].dropna()
                row[c] = v.mean() if len(v) >= width / 2 else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def external_measures(sources: Path, periods: List[int], width: int) -> pd.DataFrame:
    """BEA real GDP per capita, CSPP policies / religion, Berry citizen ideology."""
    import zipfile
    slug = lambda s: unit_state(normalize_state(str(s).strip()))  # noqa: E731

    def bea(zipname, prefix, line):
        zf = zipfile.ZipFile(sources / zipname)
        name = next(n for n in zf.namelist() if n.startswith(prefix))
        d = pd.read_csv(zf.open(name), encoding="latin-1", dtype=str)
        d = d[(d["LineCode"].str.strip() == str(line)) & (d["GeoFIPS"].str.strip('" ').str.endswith("000"))
              & (d["GeoFIPS"].str.strip('" ') != "00000")]
        long = d.melt(id_vars=["GeoName"], value_vars=[c for c in d.columns if c.isdigit()],
                      var_name="year", value_name="v")
        long["v"] = pd.to_numeric(long["v"], errors="coerce")
        long["state"] = long["GeoName"].str.replace("*", "", regex=False).map(slug)
        return long.dropna(subset=["state"]).assign(year=lambda x: x["year"].astype(int))[["state", "year", "v"]]

    gdp = bea("SAGDP.zip", "SAGDP9__ALL_AREAS", 1).rename(columns={"v": "gdp"})
    pop = bea("SAINC.zip", "SAINC1__ALL_AREAS", 2).rename(columns={"v": "pop"})
    econ = gdp.merge(pop, on=["state", "year"])
    econ["log_real_gdp_pc"] = np.log(econ["gdp"] * 1e6 / econ["pop"])

    cspp = pd.read_csv(sources / "correlates2-6.csv", encoding="latin-1",
                       usecols=lambda x: x in ["st", "year"] + list(CSPP_POLICY) + ABORTION_RESTRICTIONS)
    pol = cspp_yearly(cspp, slug)

    zf = zipfile.ZipFile(sources / "stateideology_v2018.dta.zip")
    berry = pd.read_stata(zf.open(next(n for n in zf.namelist() if n.endswith(".dta"))))
    berry = pd.DataFrame({"state": berry["statename"].map(slug), "year": berry["year"].astype(int),
                          "citizen_ideology": berry["citi6016"]})

    yearly = (econ[["state", "year", "log_real_gdp_pc"]]
              .merge(pol, on=["state", "year"], how="outer")
              .merge(berry, on=["state", "year"], how="outer"))
    cols = ["log_real_gdp_pc", "universal_prek", "equal_pay_law", "so_employment_law",
            "abortion_restrictions", "evangelical_lds_share", "citizen_ideology"]
    return window_means(yearly, cols, periods, width)


def pfl_share_by_window(pfl: pd.DataFrame, states: List[str], periods: List[int],
                        width: int) -> pd.DataFrame:
    start = dict(zip(pfl["state"], pfl["benefits_start_year"]))
    rows = []
    for st in states:
        for p in periods:
            yrs = range(p, p + width)
            s0 = start.get(st)
            rows.append({"state": st, "period": p,
                         "pfl_share": 0.0 if s0 is None else sum(y >= s0 for y in yrs) / width})
    return pd.DataFrame(rows)


def gop_share_by_window(votes: pd.DataFrame, periods: List[int], width: int) -> pd.DataFrame:
    v = votes[votes["party_simplified"].isin(["REPUBLICAN", "DEMOCRAT"])]
    t = v.pivot_table(index=["year", "state"], columns="party_simplified",
                      values="candidatevotes", aggfunc="sum").reset_index()
    t["share"] = t["REPUBLICAN"] / (t["REPUBLICAN"] + t["DEMOCRAT"])
    t["state"] = t["state"].map(lambda s: unit_state(normalize_state(s.title())))
    rows = []
    for p in periods:
        g = t[(t["year"] >= p) & (t["year"] < p + width)]
        for st, gg in g.groupby("state"):
            rows.append({"state": st, "period": p, "gop_two_party_share": gg["share"].mean(),
                         "n_elections": len(gg)})
    return pd.DataFrame(rows)


def main(config: str = "config/ipums_acs_context.yml", width: int = 10, step: int = 5,
         period_start: int = 1995, rebuild: bool = False,
         occupation_config: str = "config/ipums_acs.yml",
         pfl_file: str = "config/policy/paid_family_leave.csv",
         votes_file: str = "/scratch/network/yh6580/gender-occup/data/politics/1976-2024-president.csv",
         sources_dir: str = "/scratch/network/yh6580/gender-occup/data/context_sources",
         ) -> None:
    cfg = yaml.safe_load(open(config))
    out = Path(cfg["out_dir"])
    cache = out / "year_state_stats.parquet"
    extracts = find_extracts(out)
    if cache.exists() and not rebuild and not sources_changed(out, extracts):
        stats = pd.read_parquet(cache)
    else:
        stats = (pd.concat([year_state_stats(e) for e in extracts])
                 .groupby(["YEAR", "STATEFIP"]).sum().reset_index())
        stats.to_parquet(cache, index=False)
        (out / "aggregate_sources.txt").write_text("\n".join(str(e) for e in extracts))

    m = context_measures(stats, period_start, width, step)
    occ_out = Path(yaml.safe_load(open(occupation_config))["out_dir"])
    m = m.merge(occupation_group_shares(load_aggregate(occ_out), period_start, width, step),
                on=["STATEFIP", "period"], how="left")
    labels = ddi_labels(sorted(out.glob("usa_*.xml"))[0], "STATEFIP")
    m.insert(1, "state", m["STATEFIP"].map(lambda f: unit_state(normalize_state(labels[f]))))

    periods = [p for p, _ in year_windows(list(range(period_start, 2025)), width, step)]
    m = (m.merge(pfl_share_by_window(pd.read_csv(pfl_file), sorted(m["state"].unique()),
                                     periods, width), on=["state", "period"], how="left")
         .merge(gop_share_by_window(pd.read_csv(votes_file), periods, width)
                [["state", "period", "gop_two_party_share"]], on=["state", "period"], how="left"))
    sources = Path(sources_dir)
    if sources.exists():
        m = m.merge(external_measures(sources, periods, width), on=["state", "period"], how="left")
    dest = out / f"context_measures_{width}y_step{step}.csv"
    m.to_csv(dest, index=False)
    print(f"{dest.name}: {len(m)} state-window rows")
    print(m.groupby("period").mean(numeric_only=True).round(3).drop(columns="STATEFIP").to_string())


if __name__ == "__main__":
    fire.Fire(main)
