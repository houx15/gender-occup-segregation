#!/usr/bin/env python3
"""State-period stability diagnostic for garg_weat results on {state}_{period} units.

A real regional gender norm should change slowly, so a state's category score
should correlate strongly between adjacent periods (rough bar: r > 0.6).
Near-zero correlations mean between-state differences are mostly estimation
noise (too few / too rare wordlist words, or too-small per-unit corpora).

Reads ``garg_weat_summary_by_category.parquet`` from results_dir and writes:
  - unit_stability_pairs.csv:   category, period_a, period_b, n_states, r
  - unit_stability_summary.csv: per category — words used, units, mean/min
                                adjacent corr, share of units whose CI excludes 0

Usage:
  python -m scripts.diagnose_unit_stability --config=config/profiles/garg_weat_dlnews.yml
"""

from __future__ import annotations

from pathlib import Path

import fire
import pandas as pd

from scripts.common.config_loader import load_config


def _with_state_period(summary: pd.DataFrame) -> pd.DataFrame:
    df = summary.copy()
    parts = df["unit_name"].str.rsplit("_", n=1)
    df["state"] = parts.str[0]
    df["period"] = parts.str[1].astype(int)
    return df


def adjacent_stability(summary: pd.DataFrame) -> pd.DataFrame:
    """Correlation of state scores between each pair of adjacent periods, per category."""
    df = _with_state_period(summary)
    rows = []
    for cat, g in df.groupby("category"):
        wide = g.pivot_table(index="state", columns="period", values="mean_value")
        periods = sorted(wide.columns)
        for a, b in zip(periods[:-1], periods[1:]):
            both = wide[[a, b]].dropna()
            corr = both[a].corr(both[b]) if len(both) >= 3 else float("nan")
            rows.append({"category": cat, "period_a": a, "period_b": b,
                         "n_states": len(both), "r": corr})
    return pd.DataFrame(rows, columns=["category", "period_a", "period_b", "n_states", "r"])


def summarize_stability(summary: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    df = _with_state_period(summary)
    sig = (df["mean_ci_low"] > 0) | (df["mean_ci_high"] < 0)
    out = df.assign(sig=sig).groupby("category").agg(
        n_words=("n_consistent", "max"),
        n_units=("unit_name", "nunique"),
        share_sig=("sig", "mean"),
    )
    corr = pairs.groupby("category")["r"].agg(mean_adjacent_corr="mean",
                                                 min_adjacent_corr="min")
    return out.join(corr).reset_index()


def main(config: str) -> None:
    cfg = load_config(config)
    results = Path(cfg["paths"]["results_dir"])
    summary = pd.read_parquet(results / "garg_weat_summary_by_category.parquet")
    pairs = adjacent_stability(summary)
    out = summarize_stability(summary, pairs)
    pairs.to_csv(results / "unit_stability_pairs.csv", index=False)
    out.to_csv(results / "unit_stability_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(f"== {config}")
    print(out.round(3).to_string(index=False))
    print(pairs.round(3).to_string(index=False))


if __name__ == "__main__":
    fire.Fire(main)
