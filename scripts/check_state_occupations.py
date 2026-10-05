#!/usr/bin/env python3
"""Garg-style occupation checks at the state level: by occupation, by time, both.

Cells are state x occupation x window: the occupation's RND in that state's
model (> 0 = female-leaning) and its ACS female share in that state and window
(IPUMS, person-weighted; national share alongside for comparison). Only
occupations used by the analysis (word_coverage.csv) and present in the state
model enter.

1. by occupation   per state-window: r across occupations, RND vs the state's
                   female share — and vs the national share, to see whether the
                   state model carries state-specific information.
2. by occupation & time   per state: r across occupation x window points
                   (Garg's pooled design, one state at a time).
3. by time         per state x occupation: change first -> last window in RND
                   vs change in female share; pooled r, plus r of each state's
                   mean change.

Small ACS cells are noisy (few respondents for an occupation in a small
state); every analysis is also run on cells with weighted_n >= --min_weighted_n.

Writes to --out_dir: tables/state_occ_*.csv, figures/08-10_*.png, and
state_occupation_tables.md.

Usage:
  python -m scripts.check_state_occupations --config=config/profiles/garg_weat_dlnews_w10.yml \
      --out_dir=/scratch/network/yh6580/gender-occup/results/report_dlnews_w10
"""

from __future__ import annotations

from pathlib import Path

import fire
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import pearsonr  # noqa: E402

from scripts.common.config_loader import load_config  # noqa: E402
from scripts.data_prep.us_state_mapper import normalize_state, unit_state  # noqa: E402


def _r(x, y):
    ok = x.notna() & y.notna()
    return pearsonr(x[ok], y[ok])[0] if ok.sum() > 2 and x[ok].std() > 0 and y[ok].std() > 0 \
        else np.nan


def merge_cells(long_df, state_shares, national_shares, used) -> pd.DataFrame:
    d = long_df[(long_df["category"] == "occupation") & long_df["in_vocab"]
                & long_df["occupation"].isin(used)].copy()
    parts = d["unit_name"].str.rsplit("_", n=1)
    d["state"], d["period"] = parts.str[0], parts.str[1].astype(int)
    s = state_shares.copy()
    s["state"] = s["state"].map(lambda x: unit_state(normalize_state(str(x))))
    s = s.rename(columns={"word": "occupation"})[["state", "period", "occupation",
                                                  "female_share", "weighted_n"]]
    n = national_shares.rename(columns={"word": "occupation", "female_share": "national_share"})
    out = (d[["state", "period", "occupation", "rnd"]]
           .merge(s, on=["state", "period", "occupation"], how="inner")
           .merge(n[["occupation", "period", "national_share"]],
                  on=["occupation", "period"], how="left"))
    return out


def by_occupation(cells, min_occupations=10) -> pd.DataFrame:
    rows = []
    for (st, p), g in cells.groupby(["state", "period"]):
        if len(g) < min_occupations:
            continue
        rows.append({"state": st, "period": p, "n_occupations": len(g),
                     "r_state_share": _r(g["rnd"], g["female_share"]),
                     "r_national_share": _r(g["rnd"], g["national_share"])})
    return pd.DataFrame(rows)


def by_occupation_and_time(cells, min_points=20) -> pd.DataFrame:
    rows = []
    for st, g in cells.groupby("state"):
        if len(g) >= min_points:
            rows.append({"state": st, "n_points": len(g), "n_windows": g["period"].nunique(),
                         "r": _r(g["rnd"], g["female_share"])})
    return pd.DataFrame(rows)


def by_time(cells):
    a, b = cells["period"].min(), cells["period"].max()
    w = cells.pivot_table(index=["state", "occupation"], columns="period",
                          values=["rnd", "female_share"])
    pairs = pd.DataFrame({"d_rnd": w[("rnd", b)] - w[("rnd", a)],
                          "d_share": w[("female_share", b)] - w[("female_share", a)]}).dropna()
    pairs = pairs.reset_index()
    means = pairs.groupby("state")[["d_rnd", "d_share"]].mean()
    summary = pd.DataFrame([
        {"scope": f"state x occupation, change {a}->{b}", "n": len(pairs),
         "pearson_r": _r(pairs["d_rnd"], pairs["d_share"])},
        {"scope": f"state mean change {a}->{b}", "n": len(means),
         "pearson_r": _r(means["d_rnd"], means["d_share"])},
    ])
    return pairs, summary


def _md(df):
    d = df.round(3)
    return "\n".join(["| " + " | ".join(map(str, d.columns)) + " |",
                      "|" + "|".join("---" for _ in d.columns) + "|"]
                     + ["| " + " | ".join(str(v) for v in r) + " |" for r in d.itertuples(index=False)])


def main(config: str, out_dir: str, min_weighted_n: float = 2000.0) -> None:
    cfg = load_config(config)
    res = Path(cfg["paths"]["results_dir"])
    shares = Path(cfg["census_check"]["shares_dir"])
    width = int(cfg["us_states"].get("year_bins") or 1)
    lab = lambda p: f"{p}–{(p + width - 1) % 100:02d}"  # noqa: E731
    out = Path(out_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)

    cov = pd.read_csv(res / "word_coverage.csv")
    used = set(cov[(cov["category"] == "occupation") & cov["used"]]["occupation"])
    all_cells = merge_cells(pd.read_parquet(res / "garg_weat_rnd_long.parquet"),
                            pd.read_csv(shares / "occupation_female_share_state.csv"),
                            pd.read_csv(shares / "occupation_female_share_national.csv"), used)
    md, results = [], {}
    for name, cells in (("all cells", all_cells),
                        (f"cells with ACS weighted_n >= {int(min_weighted_n)}",
                         all_cells[all_cells["weighted_n"] >= min_weighted_n])):
        occ = by_occupation(cells)
        occ_time = by_occupation_and_time(cells)
        pairs, time_summary = by_time(cells)
        results[name] = (occ, occ_time, pairs)
        occ_sum = (occ.assign(window=occ["period"].map(lab)).groupby("window")
                   .agg(state_windows=("state", "count"),
                        median_occupations=("n_occupations", "median"),
                        median_r_state_share=("r_state_share", "median"),
                        median_r_national_share=("r_national_share", "median"),
                        share_positive=("r_state_share", lambda x: (x > 0).mean()))
                   .reset_index())
        ot_sum = pd.DataFrame([{"states": len(occ_time),
                                "median_points": occ_time["n_points"].median(),
                                "median_r": occ_time["r"].median(),
                                "q25_r": occ_time["r"].quantile(0.25),
                                "q75_r": occ_time["r"].quantile(0.75),
                                "share_positive": (occ_time["r"] > 0).mean()}])
        tag = "all" if name == "all cells" else "large"
        occ.to_csv(out / "tables" / f"state_occ_by_occupation_{tag}.csv", index=False)
        occ_time.to_csv(out / "tables" / f"state_occ_by_occupation_time_{tag}.csv", index=False)
        pairs.to_csv(out / "tables" / f"state_occ_by_time_pairs_{tag}.csv", index=False)
        time_summary.to_csv(out / "tables" / f"state_occ_by_time_{tag}.csv", index=False)
        md += [f"## {name} (n = {len(cells)} state x occupation x window cells)\n",
               "### By occupation (per state-window)\n", _md(occ_sum) + "\n",
               "### By occupation & time (per state)\n", _md(ot_sum) + "\n",
               "### By time (change, first -> last window)\n", _md(time_summary) + "\n"]
    (out / "state_occupation_tables.md").write_text("\n".join(md), encoding="utf-8")

    occ, occ_time, pairs = results["all cells"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    bins = np.linspace(-0.6, 1, 33)
    axes[0].hist(occ["r_state_share"], bins=bins, alpha=0.6, label="vs state's own female share")
    axes[0].hist(occ["r_national_share"], bins=bins, alpha=0.6, label="vs national female share")
    axes[0].axvline(0, color="grey", lw=0.6)
    axes[0].set_xlabel("r across occupations, within one state-window")
    axes[0].set_ylabel("state-windows")
    axes[0].legend(fontsize=8)
    axes[0].set_title("By occupation: each state model vs ACS")
    axes[1].scatter(occ["r_national_share"], occ["r_state_share"], s=10)
    lim = [-0.6, 1]
    axes[1].plot(lim, lim, color="grey", lw=0.6)
    axes[1].set_xlabel("r with national share")
    axes[1].set_ylabel("r with state's own share")
    axes[1].set_title("State vs national shares (same state-window)")
    plt.tight_layout()
    plt.savefig(out / "figures" / "08_state_by_occupation.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.hist(occ_time["r"], bins=bins)
    ax.axvline(0, color="grey", lw=0.6)
    ax.set_xlabel("r across occupation x window points, within one state")
    ax.set_ylabel("states")
    ax.set_title("By occupation & time (Garg's pooled design, per state)")
    plt.tight_layout()
    plt.savefig(out / "figures" / "09_state_by_occupation_time.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 4.6))
    ax.scatter(pairs["d_share"], pairs["d_rnd"], s=5, alpha=0.4)
    ax.axhline(0, color="grey", lw=0.6)
    ax.axvline(0, color="grey", lw=0.6)
    ax.set_xlabel("Change in ACS female share (state x occupation)")
    ax.set_ylabel("Change in RND")
    ax.set_title(f"By time: change, first -> last window "
                 f"(r = {_r(pairs['d_rnd'], pairs['d_share']):.2f}, n = {len(pairs)})")
    plt.tight_layout()
    plt.savefig(out / "figures" / "10_state_by_time.png", dpi=150)
    plt.close()
    print((out / "state_occupation_tables.md").read_text())


if __name__ == "__main__":
    fire.Fire(main)
