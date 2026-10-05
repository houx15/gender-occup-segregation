#!/usr/bin/env python3
"""Compare our state-window gender scores with survey benchmarks.

Descriptive check. One row per analyzed state-window unit with:
  ours_<category>   our word-fixed-effects score per category (occupation,
                    family_sphere, household; oriented RND, > 0 = female-leaning)
and survey benchmarks for the same state and window:
  objective, occupation  matched_female_share, duncan, female_emp_share (ACS)
  objective, family      motherhood_emp_gap, motherhood_hours_gap,
                         married_women_nilf, wife_earnings_share,
                         wife_earns_more, gender_emp_gap (ACS)
  subjective             iat_*, explicit_* (Project Implicit Gender-Career IAT)

Correlations (Pearson, Spearman) of every ours x benchmark pair: pooled over all
units, across states within each window, and across states for the change from
the first to the last window. ``agreement_r`` re-signs Pearson r via
TRADITIONAL_SIGN so that > 0 always means the two agree. Plus a national trend table (means over a
balanced panel of states).

Config (census_check block): shares_dir, family_file, attitude_file (each for
the profile's time windows; missing files are skipped with a note).

Writes to results_dir: state_benchmark_table.csv,
state_benchmark_correlations.csv, state_benchmark_trend.csv.

Usage:
  python -m scripts.check_state_benchmarks --config=config/profiles/garg_weat_dlnews_w10.yml
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import fire
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from scripts.check_occupation_census import state_period_table
from scripts.common.config_loader import load_config
from scripts.data_prep.us_state_mapper import normalize_state, unit_state

OCCUPATION = ["matched_female_share", "duncan", "female_emp_share"]
FAMILY = ["motherhood_emp_gap", "motherhood_hours_gap", "married_women_nilf",
          "wife_earnings_share", "wife_earns_more", "gender_emp_gap"]
ATTITUDE = ["iat_sex_balanced", "explicit_sex_balanced", "iat_mean", "explicit_mean"]

# +1 if a higher value means MORE traditional gender norms, -1 if less.
# ours_*: raw mean RND (> 0 = closer to women); female-leaning occupations are
# less traditional, female-leaning family / household words more traditional.
TRADITIONAL_SIGN = {
    "ours_occupation": -1, "ours_family_sphere": 1, "ours_household": 1,
    "matched_female_share": -1, "duncan": 1, "female_emp_share": -1,
    "motherhood_emp_gap": 1, "motherhood_hours_gap": 1, "married_women_nilf": 1,
    "wife_earnings_share": -1, "wife_earns_more": -1, "gender_emp_gap": 1,
    "iat_sex_balanced": 1, "explicit_sex_balanced": 1, "iat_mean": 1, "explicit_mean": 1,
    "family_index_acs": 1,
}


def add_family_index(t: pd.DataFrame) -> pd.DataFrame:
    """family_index_acs: the main objective family measure. Each ACS family
    measure is z-scored across all state-windows, oriented with
    TRADITIONAL_SIGN (higher = more traditional), and averaged over the
    measures available for the unit."""
    cols = [c for c in FAMILY if c in t.columns]
    if not cols:
        return t
    z = pd.DataFrame({c: TRADITIONAL_SIGN[c] * (t[c] - t[c].mean()) / t[c].std(ddof=0)
                      for c in cols})
    return t.assign(family_index_acs=z.mean(axis=1, skipna=True))


def with_unit_name(d: pd.DataFrame) -> pd.DataFrame:
    """Add unit_name ('ohio_2005') from a state name or USPS code + period."""
    d = d.copy()
    d["unit_name"] = (d["state"].map(lambda s: unit_state(normalize_state(str(s))))
                      + "_" + d["period"].astype(int).astype(str))
    return d


def our_scores(summary: pd.DataFrame) -> pd.DataFrame:
    w = summary.pivot_table(index="unit_name", columns="category", values="mean_value")
    w.columns = [f"ours_{c}" for c in w.columns]
    return w.reset_index()


def benchmark_table(ours: pd.DataFrame, sources: List[pd.DataFrame]) -> pd.DataFrame:
    t = ours.copy()
    for src in sources:
        s = with_unit_name(src).drop(columns=[c for c in ("state", "period", "STATEFIP")
                                              if c in src.columns])
        t = t.merge(s, on="unit_name", how="left")
    parts = t["unit_name"].str.rsplit("_", n=1)
    t["state"], t["period"] = parts.str[0], parts.str[1].astype(int)
    return t


def _corr(scope, ours, survey, x, y):
    ok = x.notna() & y.notna()
    n = int(ok.sum())
    return {"ours": ours, "survey": survey, "scope": scope, "n": n,
            "pearson_r": pearsonr(x[ok], y[ok])[0] if n > 2 else np.nan,
            "spearman_r": spearmanr(x[ok], y[ok])[0] if n > 2 else np.nan}


def correlate(t: pd.DataFrame, ours_cols: List[str], survey_cols: List[str]) -> pd.DataFrame:
    rows = []
    periods = sorted(t["period"].unique())
    for o in ours_cols:
        for s in survey_cols:
            rows.append(_corr("pooled", o, s, t[o], t[s]))
            for p in periods:
                g = t[t["period"] == p]
                rows.append(_corr(f"period {p}", o, s, g[o], g[s]))
            # change: first -> last window in which BOTH measures have data
            # (e.g. Project Implicit starts in 2005, after the first text window)
            both = [p for p in periods
                    if t.loc[t["period"] == p, [o, s]].notna().all(axis=1).any()]
            if len(both) > 1:
                a, b = both[0], both[-1]
                w = t.pivot_table(index="state", columns="period", values=[o, s])
                rows.append(_corr(f"change {a}->{b}", o, s,
                                  w[(o, b)] - w[(o, a)], w[(s, b)] - w[(s, a)]))
    return pd.DataFrame(rows)


def add_agreement(corr: pd.DataFrame) -> pd.DataFrame:
    """agreement_r: pearson_r re-signed so that > 0 = ours and survey agree
    (both point to more, or both to less, traditional norms)."""
    sign = (corr["ours"].map(TRADITIONAL_SIGN) * corr["survey"].map(TRADITIONAL_SIGN))
    return corr.assign(agreement_r=corr["pearson_r"] * sign)


def national_trend(t: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    periods = t["period"].nunique()
    full = t.groupby("state")["period"].nunique()
    b = t[t["state"].isin(full[full == periods].index)]
    out = b.groupby("period")[cols].mean()
    out.insert(0, "n_states", b.groupby("period")["state"].nunique())
    return out.reset_index()


def main(config: str) -> None:
    cfg = load_config(config)
    cc = cfg["census_check"]
    results = Path(cfg["paths"]["results_dir"])
    shares = Path(cc["shares_dir"])

    summary = pd.read_parquet(results / "garg_weat_summary_by_category.parquet")
    cov = pd.read_csv(results / "word_coverage.csv")
    used = set(cov[(cov["category"] == "occupation") & cov["used"]]["occupation"])
    occ = state_period_table(summary, pd.read_csv(shares / "occupation_female_share_state.csv"),
                             pd.read_csv(shares / "state_labor_indicators.csv"), used)
    occ = occ[["unit_name"] + OCCUPATION]

    sources, survey_cols = [], list(OCCUPATION)
    for key, cols in (("family_file", FAMILY), ("attitude_file", ATTITUDE)):
        path = cc.get(key)
        if path and Path(path).exists():
            sources.append(pd.read_csv(path))
            survey_cols += cols
        else:
            print(f"  (skipping {key}: {path} not found)")

    t = add_family_index(
        benchmark_table(our_scores(summary).merge(occ, on="unit_name", how="left"), sources))
    if "family_index_acs" in t.columns:
        survey_cols.append("family_index_acs")
    ours_cols = [c for c in t.columns if c.startswith("ours_")]
    corr = add_agreement(correlate(t, ours_cols, survey_cols))
    trend = national_trend(t, ours_cols + survey_cols)
    t.to_csv(results / "state_benchmark_table.csv", index=False)
    corr.to_csv(results / "state_benchmark_correlations.csv", index=False)
    trend.to_csv(results / "state_benchmark_trend.csv", index=False)

    pd.set_option("display.width", 250)
    print(f"== {config}: {len(t)} units, benchmarks: {survey_cols}")
    print("\nNational trend (balanced states):")
    print(trend.round(4).to_string(index=False))
    print("\nAgreement correlations (> 0 = ours and survey agree on more/less traditional):")
    print(corr.pivot_table(index=["ours", "survey"], columns="scope",
                           values="agreement_r").round(2).to_string())


if __name__ == "__main__":
    fire.Fire(main)
