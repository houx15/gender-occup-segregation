#!/usr/bin/env python3
"""Check occupation gender scores against ACS census data (IPUMS).

A descriptive check, not a validation. Two comparisons:

1. National, by occupation (Garg et al. 2018 style): per period, each used
   occupation's mean RND over the state-period units vs its national female
   share in the same period.
2. By state-period: our occupation gender norm (the unit's word-fixed-effects
   occupation score) vs three survey measures for the same state and period:
     - matched_female_share: equal-weight mean female share of OUR used
       occupations in that state-period (same occupations on both sides);
     - duncan: occupational segregation index over all occupations;
     - female_emp_share: women's share of employment.
   Correlations: pooled over all state-periods, across states within each
   period, and across states for the change first -> last period.

Inputs (config census_check block): shares_dir with the outputs of
scripts.data_prep.build_occupation_shares; results_dir with the analysis
outputs (garg_weat_rnd_long.parquet, garg_weat_summary_by_category.parquet,
word_coverage.csv).

Writes to results_dir: occupation_census_scores.csv,
occupation_census_correlations.csv, state_survey_table.csv,
state_survey_correlations.csv; to figures_dir: occupation_census_check.pdf,
state_survey_check.pdf.

Usage:
  python -m scripts.check_occupation_census --config=config/profiles/garg_weat_dlnews.yml
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import fire
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402

from scripts.common.config_loader import load_config  # noqa: E402
from scripts.data_prep.us_state_mapper import unit_state  # noqa: E402

SURVEY_MEASURES = ["matched_female_share", "duncan", "female_emp_share"]


def _period_of(unit_names: pd.Series) -> pd.Series:
    return unit_names.str.rsplit("_", n=1).str[1].astype(int)


def occupation_period_scores(long_df: pd.DataFrame, used: set) -> pd.DataFrame:
    """period, occupation, rnd (mean over units with the word), n_units."""
    d = long_df[(long_df["category"] == "occupation") & long_df["in_vocab"]
                & long_df["occupation"].isin(used)].copy()
    d["period"] = _period_of(d["unit_name"])
    return (d.groupby(["period", "occupation"])["rnd"]
            .agg(rnd="mean", n_units="count").reset_index())


def census_correlations(table: pd.DataFrame) -> pd.DataFrame:
    """Per period: n, Pearson r (raw and logit share) and Spearman r of RND vs share."""
    rows = []
    for period, g in table.dropna(subset=["rnd", "female_share"]).groupby("period"):
        p = g["female_share"].clip(0.005, 0.995)
        logit = np.log(p / (1 - p))
        ok = len(g) > 2
        rows.append({
            "period": period, "n_occupations": len(g),
            "pearson_r": pearsonr(g["rnd"], g["female_share"])[0] if ok else np.nan,
            "pearson_r_logit": pearsonr(g["rnd"], logit)[0] if ok else np.nan,
            "spearman_r": spearmanr(g["rnd"], g["female_share"])[0] if ok else np.nan,
        })
    return pd.DataFrame(rows)


def state_period_table(summary: pd.DataFrame, shares_state: pd.DataFrame,
                       labor: pd.DataFrame, used: set) -> pd.DataFrame:
    """One row per analyzed state-period unit: our score + survey measures."""
    ours = summary[summary["category"] == "occupation"][["unit_name", "mean_value"]]
    ours = ours.rename(columns={"mean_value": "our_score"})
    s = shares_state[shares_state["word"].isin(used)].copy()
    s["unit_name"] = s["state"].map(unit_state) + "_" + s["period"].astype(str)
    matched = (s.groupby("unit_name")["female_share"].agg(["mean", "count"])
               .rename(columns={"mean": "matched_female_share", "count": "n_matched_words"}))
    lab = labor.copy()
    lab["unit_name"] = lab["state"].map(unit_state) + "_" + lab["period"].astype(str)
    t = (ours.merge(matched, on="unit_name", how="left")
         .merge(lab[["unit_name", "state", "duncan", "female_emp_share"]],
                on="unit_name", how="left"))
    t["period"] = _period_of(t["unit_name"])
    return t


def _corr_row(scope: str, measure: str, x: pd.Series, y: pd.Series) -> dict:
    ok = x.notna() & y.notna()
    n = int(ok.sum())
    return {"scope": scope, "measure": measure, "n": n,
            "pearson_r": pearsonr(x[ok], y[ok])[0] if n > 2 else np.nan,
            "spearman_r": spearmanr(x[ok], y[ok])[0] if n > 2 else np.nan}


def state_correlations(table: pd.DataFrame, measures: List[str]) -> pd.DataFrame:
    rows = []
    periods = sorted(table["period"].unique())
    for m in measures:
        rows.append(_corr_row("pooled", m, table["our_score"], table[m]))
        for p in periods:
            g = table[table["period"] == p]
            rows.append(_corr_row(f"period {p}", m, g["our_score"], g[m]))
        if len(periods) > 1:
            first, last = periods[0], periods[-1]
            w = table.pivot_table(index="state", columns="period", values=["our_score", m])
            d_ours = w[("our_score", last)] - w[("our_score", first)]
            d_m = w[(m, last)] - w[(m, first)]
            rows.append(_corr_row(f"change {first}->{last}", m, d_ours, d_m))
    return pd.DataFrame(rows)


def _plot_national(table, corr, width, path):
    periods = sorted(table["period"].unique())
    fig, axes = plt.subplots(1, len(periods), figsize=(4.2 * len(periods), 4.2),
                             squeeze=False, sharey=True)
    r = corr.set_index("period")
    for ax, p in zip(axes[0], periods):
        g = table[table["period"] == p].dropna(subset=["female_share"])
        ax.scatter(g["female_share"], g["rnd"], s=12)
        for _, row in g.iterrows():
            ax.annotate(row["occupation"], (row["female_share"], row["rnd"]), fontsize=5)
        ax.axhline(0, color="grey", lw=0.6, ls="--")
        label = f"{p}–{(p + width - 1) % 100:02d}" if width > 1 else str(p)
        ax.set_title(f"{label}: r={r.loc[p, 'pearson_r']:.2f} (n={int(r.loc[p, 'n_occupations'])})")
        ax.set_xlabel("ACS female share")
    axes[0][0].set_ylabel("Occupation RND (> 0 = female-leaning)")
    plt.tight_layout()
    plt.savefig(path, format="pdf")
    plt.close()


def _plot_state(table, corr, path):
    fig, axes = plt.subplots(1, len(SURVEY_MEASURES), figsize=(4.4 * len(SURVEY_MEASURES), 4.2),
                             squeeze=False)
    pooled = corr[corr["scope"] == "pooled"].set_index("measure")
    for ax, m in zip(axes[0], SURVEY_MEASURES):
        for p, g in table.groupby("period"):
            ax.scatter(g[m], g["our_score"], s=10, label=str(p))
        ax.set_xlabel(m)
        ax.set_title(f"{m}: pooled r={pooled.loc[m, 'pearson_r']:.2f} (n={int(pooled.loc[m, 'n'])})")
    axes[0][0].set_ylabel("Our occupation gender score (state-period)")
    axes[0][0].legend(title="period", fontsize=7)
    plt.tight_layout()
    plt.savefig(path, format="pdf")
    plt.close()


def main(config: str) -> None:
    cfg = load_config(config)
    shares_dir = Path(cfg["census_check"]["shares_dir"])
    results = Path(cfg["paths"]["results_dir"])
    figures = Path(cfg["paths"]["figures_dir"])
    figures.mkdir(parents=True, exist_ok=True)
    width = int(cfg.get("us_states", {}).get("year_bins") or 1)

    cov = pd.read_csv(results / "word_coverage.csv")
    used = set(cov[(cov["category"] == "occupation") & cov["used"]]["occupation"])

    # 1. national, by occupation
    long_df = pd.read_parquet(results / "garg_weat_rnd_long.parquet")
    nat = pd.read_csv(shares_dir / "occupation_female_share_national.csv")
    scores = occupation_period_scores(long_df, used)
    table = scores.merge(nat.rename(columns={"word": "occupation"}),
                         on=["period", "occupation"], how="left")
    corr = census_correlations(table)
    table.to_csv(results / "occupation_census_scores.csv", index=False)
    corr.to_csv(results / "occupation_census_correlations.csv", index=False)
    _plot_national(table, corr, width, figures / "occupation_census_check.pdf")
    n_match = table.dropna(subset=["female_share"])["occupation"].nunique()
    print(f"== national, by occupation: {len(used)} used, {n_match} with ACS shares")
    print(corr.round(3).to_string(index=False))

    # 2. by state-period
    summary = pd.read_parquet(results / "garg_weat_summary_by_category.parquet")
    st = state_period_table(summary,
                            pd.read_csv(shares_dir / "occupation_female_share_state.csv"),
                            pd.read_csv(shares_dir / "state_labor_indicators.csv"), used)
    scorr = state_correlations(st, SURVEY_MEASURES)
    st.to_csv(results / "state_survey_table.csv", index=False)
    scorr.to_csv(results / "state_survey_correlations.csv", index=False)
    _plot_state(st, scorr, figures / "state_survey_check.pdf")
    print(f"\n== by state-period: {len(st)} units")
    print(scorr.round(3).to_string(index=False))


if __name__ == "__main__":
    fire.Fire(main)
