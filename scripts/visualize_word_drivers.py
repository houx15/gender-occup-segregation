#!/usr/bin/env python3
"""Render word-level ideation-driver figures from word_drivers_* tables.

One PDF per (dimension × form): contribution bars, slope/dumbbell, word×year
heatmap, trajectory small-multiples. Reads the consistent-set tables produced
by scripts.analyze_word_drivers.

Usage:
  python -m scripts.visualize_word_drivers --config=config/profiles/garg_weat_renminribao.yml
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import fire

from scripts.common.config_loader import load_config
from scripts.common.logging_utils import setup_logging
from scripts.visualize import _configure_fonts

_NEG = "#c0392b"   # negative signed value / contribution (toward more-traditional)
_POS = "#2c7fb8"   # positive signed value / contribution (toward less-traditional)


def _top_n(cfg: dict) -> int:
    return int(cfg.get("analysis", {}).get("word_drivers", {}).get("top_n", 20))


def plot_contribution(summary_df, category, figures_dir, top_n, logger):
    sub = summary_df[summary_df["category"] == category].copy()
    if sub.empty:
        logger.warning(f"  contribution[{category}]: no words — skipped")
        return
    sub["_absc"] = sub["contribution"].abs()
    sub = sub.sort_values("_absc", ascending=False).head(top_n).sort_values("contribution")
    colors = [_NEG if v < 0 else _POS for v in sub["contribution"]]
    fig, ax = plt.subplots(figsize=(8, max(3.0, 0.35 * len(sub))))
    ax.barh(sub["occupation"].astype(str), sub["contribution"], color=colors)
    ax.axvline(0, color="k", lw=0.8)
    total = float(sub["contribution"].sum())
    ax.set_xlabel("contribution to Δ ideation (signed RND / N)")
    ax.set_title(
        f"{category}: word contributions to change "
        f"({int(sub['first_year'].iloc[0])}→{int(sub['last_year'].iloc[0])}); "
        f"shown Σ={total:+.3f}"
    )
    fig.tight_layout()
    out = Path(figures_dir) / f"word_drivers_contribution_{category}.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  saved {out}")


def plot_slope(summary_df, category, figures_dir, top_n, logger):
    sub = summary_df[summary_df["category"] == category].copy()
    if sub.empty:
        logger.warning(f"  slope[{category}]: no words — skipped")
        return
    sub["_absd"] = sub["delta"].abs()
    sub = sub.sort_values("_absd", ascending=False).head(top_n).sort_values("delta")
    fig, ax = plt.subplots(figsize=(8, max(3.0, 0.35 * len(sub))))
    y = np.arange(len(sub))
    for yi, (_, r) in zip(y, sub.iterrows()):
        color = _POS if r["delta"] >= 0 else _NEG
        ax.plot([r["signed_first"], r["signed_last"]], [yi, yi], color=color, lw=1.5, zorder=1)
        ax.scatter([r["signed_first"]], [yi], color="#bbbbbb", s=30, zorder=2)
        ax.scatter([r["signed_last"]], [yi], color=color, s=30, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(sub["occupation"].astype(str))
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("signed RND (start ○ → end ●)")
    ax.set_title(
        f"{category}: per-word change "
        f"{int(sub['first_year'].iloc[0])}→{int(sub['last_year'].iloc[0])}"
    )
    fig.tight_layout()
    out = Path(figures_dir) / f"word_drivers_slope_{category}.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  saved {out}")


def plot_heatmap(long_df, summary_df, category, figures_dir, logger):
    sub = long_df[long_df["category"] == category]
    if sub.empty:
        logger.warning(f"  heatmap[{category}]: no rows — skipped")
        return
    pivot = sub.pivot_table(index="occupation", columns="year", values="signed_rnd")
    order = summary_df[summary_df["category"] == category].set_index("occupation")["delta"]
    order = order.sort_values(ascending=False, na_position="last").index
    pivot = pivot.reindex([o for o in order if o in pivot.index]).dropna(how="all")
    if pivot.empty:
        logger.warning(f"  heatmap[{category}]: empty pivot — skipped")
        return
    arr = pivot.to_numpy()
    finite = arr[np.isfinite(arr)]
    vmax = float(np.abs(finite).max()) if finite.size else 1.0
    if vmax == 0.0:
        vmax = 1.0
    fig, ax = plt.subplots(
        figsize=(max(6.0, 0.5 * pivot.shape[1]), max(4.0, 0.3 * pivot.shape[0]))
    )
    sns.heatmap(
        pivot, cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax,
        cbar_kws={"label": "signed RND"}, ax=ax,
    )
    ax.set_title(f"{category}: signed RND by word × year")
    ax.set_xlabel("year")
    ax.set_ylabel("")
    fig.tight_layout()
    out = Path(figures_dir) / f"word_drivers_heatmap_{category}.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  saved {out}")


def plot_trajectory(long_df, summary_df, category, figures_dir, top_n, logger):
    sub = long_df[long_df["category"] == category]
    if sub.empty:
        logger.warning(f"  trajectory[{category}]: no rows — skipped")
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    for _, g in sub.groupby("occupation"):
        g = g.sort_values("year")
        ax.plot(g["year"], g["signed_rnd"], color="#cccccc", lw=0.7, zorder=1)
    movers = summary_df[summary_df["category"] == category].copy()
    movers["_absd"] = movers["delta"].abs()
    movers = movers.sort_values("_absd", ascending=False).head(top_n)
    cmap = matplotlib.colormaps["tab10"]
    for i, occ in enumerate(movers["occupation"]):
        g = sub[sub["occupation"] == occ].sort_values("year")
        ax.plot(g["year"], g["signed_rnd"], color=cmap(i % 10), lw=1.8, zorder=3, label=str(occ))
        ax.annotate(
            str(occ), (g["year"].iloc[-1], g["signed_rnd"].iloc[-1]),
            fontsize=7, xytext=(3, 0), textcoords="offset points",
        )
    mean_line = sub.groupby("year")["cat_mean_signed"].first().sort_index()
    ax.plot(
        mean_line.index, mean_line.values, color="k", lw=2.2, ls="--",
        zorder=4, label="dimension mean",
    )
    ax.set_xlabel("year")
    ax.set_ylabel("signed RND")
    ax.set_title(f"{category}: per-word trajectories (top {top_n} movers bold)")
    ax.legend(fontsize=6, ncol=2, loc="best")
    fig.tight_layout()
    out = Path(figures_dir) / f"word_drivers_trajectory_{category}.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  saved {out}")


def main(config: str = "config/config.yml") -> None:
    """Render all four driver figures per dimension from the driver tables."""
    cfg = load_config(config)
    logger = setup_logging(
        Path(cfg["paths"]["log_dir"]), "visualize_word_drivers.log"
    )
    sns.set_style("whitegrid")
    _configure_fonts(cfg)  # must run after sns.set_style resets rcParams
    results_dir = Path(cfg["paths"]["results_dir"])
    figures_dir = Path(
        cfg["paths"].get("figures_dir", cfg["paths"]["results_dir"] + "/figures")
    )
    figures_dir.mkdir(parents=True, exist_ok=True)

    long_path = results_dir / "word_drivers_long.parquet"
    summ_path = results_dir / "word_drivers_summary.parquet"
    if not long_path.exists() or not summ_path.exists():
        raise FileNotFoundError(
            f"visualize_word_drivers: expected {long_path} and {summ_path}. "
            f"Run scripts.analyze_word_drivers first."
        )
    long_df = pd.read_parquet(long_path)
    summary_df = pd.read_parquet(summ_path)
    top_n = _top_n(cfg)

    categories = sorted(long_df["category"].unique())
    logger.info(f"word_drivers figures: categories={categories}, top_n={top_n}")
    for cat in categories:
        plot_contribution(summary_df, cat, figures_dir, top_n, logger)
        plot_slope(summary_df, cat, figures_dir, top_n, logger)
        plot_heatmap(long_df, summary_df, cat, figures_dir, logger)
        plot_trajectory(long_df, summary_df, cat, figures_dir, top_n, logger)


if __name__ == "__main__":
    fire.Fire(main)
