#!/usr/bin/env python3
"""Garg et al. (2018) Fig. 2 for 3DLNews2: average occupation bias over time vs
the average women's occupation % difference, nationally and per state.

For each window: blue = mean RND over a fixed set of occupations (present in
every window), green = mean (women % - men %) of the same occupations from ACS
(person-weighted). Shaded bands = bootstrap SE (resampling occupations), as in
Garg's figure. Two y axes, like the original.

National: occupation RND averaged over the state models (no national model)
vs national ACS shares. States: each state's own models vs its own ACS shares,
one small panel per state.

Writes to --out_dir: figures/11_garg_fig2_national[_TAG].png,
figures/12_garg_fig2_states[_TAG].png, tables/garg_fig2_{national,states}[_TAG].csv.

Usage:
  python -m scripts.plot_garg_fig2 --config=config/profiles/garg_weat_dlnews_w10.yml \
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

from scripts.check_state_occupations import merge_cells  # noqa: E402
from scripts.common.config_loader import load_config  # noqa: E402

BLUE, GREEN = "#1f3fd1", "#1b8a2a"


def pct_difference(share: pd.Series) -> pd.Series:
    """Women % minus men % (Garg's 'occupation % difference')."""
    return 100 * (2 * share - 1)


def garg_series(cells: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """cells: period, occupation, rnd, female_share. One row per period."""
    periods = sorted(cells["period"].unique())
    present = cells.dropna(subset=["rnd", "female_share"]).groupby("occupation")["period"].nunique()
    occs = present[present == len(periods)].index
    d = cells[cells["occupation"].isin(occs)].copy()
    d["pct_diff"] = pct_difference(d["female_share"])
    rnd = d.pivot_table(index="occupation", columns="period", values="rnd")
    pct = d.pivot_table(index="occupation", columns="period", values="pct_diff")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(occs), size=(n_boot, len(occs)))
    rows = []
    for p in periods:
        r, c = rnd[p].to_numpy(), pct[p].to_numpy()
        rows.append({"period": p, "n_occupations": len(occs),
                     "avg_bias": r.mean(), "bias_se": r[idx].mean(axis=1).std(),
                     "avg_pct_diff": c.mean(), "pct_diff_se": c[idx].mean(axis=1).std()})
    return pd.DataFrame(rows)


def _dual_axis(ax, s: pd.DataFrame, labels, title: str, small: bool = False):
    x = np.arange(len(s))
    ax.plot(x, s["avg_bias"], color=BLUE, marker="o", ms=3 if small else 6)
    ax.fill_between(x, s["avg_bias"] - s["bias_se"], s["avg_bias"] + s["bias_se"],
                    color=BLUE, alpha=0.15)
    ax2 = ax.twinx()
    ax2.plot(x, s["avg_pct_diff"], color=GREEN, marker="o", ms=3 if small else 6)
    ax2.fill_between(x, s["avg_pct_diff"] - s["pct_diff_se"],
                     s["avg_pct_diff"] + s["pct_diff_se"], color=GREEN, alpha=0.15)
    ax.set_xticks(x, labels, fontsize=5 if small else 9)
    ax.set_title(title, fontsize=7 if small else 11)
    if small:
        ax.tick_params(axis="y", labelsize=5, colors=BLUE)
        ax2.tick_params(axis="y", labelsize=5, colors=GREEN)
    else:
        ax.set_ylabel("Avg. women bias (RND)", color=BLUE)
        ax2.set_ylabel("Avg. women occup. % difference (ACS)", color=GREEN)
    return ax2


def main(config: str, out_dir: str, tag: str = "", n_boot: int = 1000) -> None:
    cfg = load_config(config)
    res = Path(cfg["paths"]["results_dir"])
    shares = Path(cfg["census_check"]["shares_dir"])
    width = int(cfg["us_states"].get("year_bins") or 1)
    lab = lambda p: f"{p}–{(p + width - 1) % 100:02d}"  # noqa: E731
    out = Path(out_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    sfx = f"_{tag}" if tag else ""

    # national: per-occupation mean RND over state models vs national ACS share
    nat_cells = pd.read_csv(res / "occupation_census_scores.csv")
    nat = garg_series(nat_cells, n_boot=n_boot)
    nat.to_csv(out / "tables" / f"garg_fig2_national{sfx}.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4.6))
    _dual_axis(ax, nat, [lab(p) for p in nat["period"]],
               f"National: occupations' gender bias vs ACS (n = {nat['n_occupations'][0]} occupations)")
    ax.set_xlabel("Window")
    plt.tight_layout()
    plt.savefig(out / "figures" / f"11_garg_fig2_national{sfx}.png", dpi=150)
    plt.close()

    # states: each state's models vs its own ACS shares
    cov = pd.read_csv(res / "word_coverage.csv")
    used = set(cov[(cov["category"] == "occupation") & cov["used"]]["occupation"])
    cells = merge_cells(pd.read_parquet(res / "garg_weat_rnd_long.parquet"),
                        pd.read_csv(shares / "occupation_female_share_state.csv"),
                        pd.read_csv(shares / "occupation_female_share_national.csv"), used)
    n_periods = cells["period"].nunique()
    series = []
    for st, g in cells.groupby("state"):
        if g["period"].nunique() < n_periods:
            continue
        s = garg_series(g, n_boot=n_boot)
        if s["n_occupations"].iloc[0] >= 10:
            series.append(s.assign(state=st))
    states = pd.concat(series, ignore_index=True)
    states.to_csv(out / "tables" / f"garg_fig2_states{sfx}.csv", index=False)

    names = sorted(states["state"].unique())
    ncol = 8
    nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.3 * ncol, 1.9 * nrow))
    for ax in axes.flat[len(names):]:
        ax.set_axis_off()
    for ax, st in zip(axes.flat, names):
        s = states[states["state"] == st].sort_values("period")
        _dual_axis(ax, s, [str(p)[2:] for p in s["period"]],
                   f"{st.replace('_', ' ').title()} (n={s['n_occupations'].iloc[0]})", small=True)
    fig.suptitle("Per state: avg. women bias (blue, RND) vs avg. women occupation % difference "
                 "(green, ACS), same occupations; bands = bootstrap SE", fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.savefig(out / "figures" / f"12_garg_fig2_states{sfx}.png", dpi=130)
    plt.close()

    # how often do the two lines move together, per state (sign of change first -> last)
    first, last = states["period"].min(), states["period"].max()
    w = states.pivot_table(index="state", columns="period", values=["avg_bias", "avg_pct_diff"])
    d_bias = w[("avg_bias", last)] - w[("avg_bias", first)]
    d_pct = w[("avg_pct_diff", last)] - w[("avg_pct_diff", first)]
    print(nat.round(4).to_string(index=False))
    print(f"\nstates: {len(names)}; both lines up {first}->{last}: "
          f"{int(((d_bias > 0) & (d_pct > 0)).sum())}; bias up: {int((d_bias > 0).sum())}; "
          f"ACS % difference up: {int((d_pct > 0).sum())}; "
          f"r(d_bias, d_pct) across states = {d_bias.corr(d_pct):.2f}")


if __name__ == "__main__":
    fire.Fire(main)
