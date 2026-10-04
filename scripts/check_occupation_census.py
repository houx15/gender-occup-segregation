#!/usr/bin/env python3
"""Check occupation scores against census female shares (Garg et al. 2018 style).

A descriptive check, not a validation: how closely does each occupation's
embedding gender score (RND, > 0 = female-leaning) track the share of women
actually working in it? For each analyzed period, an occupation's score is its
mean RND over the units where it is in vocab, and its female share is the mean
of the census years inside the period from Garg's
``occupation_percentages_gender.csv`` (1850-2015). Periods after 2015 fall back
to the latest census year, flagged in ``share_years``.

Only occupations the analysis actually used (``word_coverage.csv``) and that
have a Garg counterpart (``garg_word`` in the grounding table) enter.

Config (census_check block):
  grounding_file:     wordlists/en/occupation_family/occupation_grounding.csv
  female_share_file:  wordlists/en/garg/occupation_percentages_gender.csv

Writes to results_dir: occupation_census_scores.csv, occupation_census_correlations.csv;
to figures_dir: occupation_census_check.pdf.

Usage:
  python -m scripts.check_occupation_census --config=config/profiles/garg_weat_dlnews.yml
"""

from __future__ import annotations

from pathlib import Path

import fire
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402

from scripts.common.config_loader import load_config  # noqa: E402


def occupation_period_scores(long_df: pd.DataFrame, grounding: pd.DataFrame,
                             used: set) -> pd.DataFrame:
    """period, occupation, garg_word, rnd (mean over units), n_units."""
    d = long_df[(long_df["category"] == "occupation") & long_df["in_vocab"]
                & long_df["occupation"].isin(used)].copy()
    d["period"] = d["unit_name"].str.rsplit("_", n=1).str[1].astype(int)
    out = (d.groupby(["period", "occupation"])["rnd"]
           .agg(rnd="mean", n_units="count").reset_index())
    garg = grounding.set_index("word")["garg_word"]
    out["garg_word"] = out["occupation"].map(garg)
    return out


def period_female_share(garg: pd.DataFrame, period: int, width: int) -> pd.DataFrame:
    """garg_word, female_share, share_years for one period [period, period+width-1]."""
    rows = []
    for occ, g in garg.groupby("Occupation"):
        inside = g[(g["Census year"] >= period) & (g["Census year"] <= period + width - 1)]
        if not inside.empty:
            ys = inside["Census year"]
            rows.append({"garg_word": occ, "female_share": inside["Female"].mean(),
                         "share_years": f"{ys.min()}-{ys.max()}" if ys.nunique() > 1
                         else str(ys.min())})
            continue
        before = g[g["Census year"] < period]
        if not before.empty:
            last = before.loc[before["Census year"].idxmax()]
            rows.append({"garg_word": occ, "female_share": float(last["Female"]),
                         "share_years": f"{int(last['Census year'])} (latest)"})
    return pd.DataFrame(rows, columns=["garg_word", "female_share", "share_years"])


def census_correlations(table: pd.DataFrame) -> pd.DataFrame:
    """Per period: n, Pearson r (raw and logit share) and Spearman r of RND vs share."""
    rows = []
    for period, g in table.dropna(subset=["rnd", "female_share"]).groupby("period"):
        p = g["female_share"].clip(0.005, 0.995)
        logit = np.log(p / (1 - p))
        rows.append({
            "period": period, "n_occupations": len(g),
            "pearson_r": pearsonr(g["rnd"], g["female_share"])[0] if len(g) > 2 else np.nan,
            "pearson_r_logit": pearsonr(g["rnd"], logit)[0] if len(g) > 2 else np.nan,
            "spearman_r": spearmanr(g["rnd"], g["female_share"])[0] if len(g) > 2 else np.nan,
        })
    return pd.DataFrame(rows)


def _plot(table: pd.DataFrame, corr: pd.DataFrame, width: int, path: Path) -> None:
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
        ax.set_xlabel("Census female share")
    axes[0][0].set_ylabel("Occupation RND (> 0 = female-leaning)")
    plt.tight_layout()
    plt.savefig(path, format="pdf")
    plt.close()


def main(config: str) -> None:
    cfg = load_config(config)
    cc = cfg["census_check"]
    results = Path(cfg["paths"]["results_dir"])
    width = int(cfg.get("us_states", {}).get("year_bins") or 1)

    long_df = pd.read_parquet(results / "garg_weat_rnd_long.parquet")
    cov = pd.read_csv(results / "word_coverage.csv")
    used = set(cov[(cov["category"] == "occupation") & cov["used"]]["occupation"])
    grounding = pd.read_csv(cc["grounding_file"])
    garg = pd.read_csv(cc["female_share_file"])

    scores = occupation_period_scores(long_df, grounding, used)
    shares = pd.concat([period_female_share(garg, p, width).assign(period=p)
                        for p in sorted(scores["period"].unique())], ignore_index=True)
    table = scores.merge(shares, on=["period", "garg_word"], how="left")
    corr = census_correlations(table)

    table.to_csv(results / "occupation_census_scores.csv", index=False)
    corr.to_csv(results / "occupation_census_correlations.csv", index=False)
    figures = Path(cfg["paths"]["figures_dir"])
    figures.mkdir(parents=True, exist_ok=True)
    _plot(table, corr, width, figures / "occupation_census_check.pdf")

    n_match = table.dropna(subset=["female_share"])["occupation"].nunique()
    print(f"== {config}: {len(used)} occupations used, {n_match} with census shares")
    print(corr.round(3).to_string(index=False))


if __name__ == "__main__":
    fire.Fire(main)
