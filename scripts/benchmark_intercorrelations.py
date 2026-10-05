#!/usr/bin/env python3
"""How strongly do the survey benchmarks agree with EACH OTHER across states?

Sets the yardstick for judging our text measure: if two survey measures of
gender norms barely correlate across states, our measure cannot be expected
to correlate strongly with either.

All measures (ours and survey) are oriented with
check_state_benchmarks.TRADITIONAL_SIGN so that > 0 always means agreement on
more / less traditional. For each period, Pearson r across states; the matrix
reported is the mean over periods. Also each measure's stability across
non-overlapping windows (first vs last period).

Reads <results_dir>/state_benchmark_table.csv (from check_state_benchmarks).
Writes <results_dir>/benchmark_intercorrelations.csv and _stability.csv.

Usage:
  python -m scripts.benchmark_intercorrelations --config=config/profiles/garg_weat_dlnews.yml
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import fire
import pandas as pd

from scripts.check_state_benchmarks import TRADITIONAL_SIGN
from scripts.common.config_loader import load_config


def oriented_within_period_corr(t: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    o = t[["period"] + cols].copy()
    for c in cols:
        o[c] = o[c] * TRADITIONAL_SIGN.get(c, 1)
    mats = [g[cols].corr() for _, g in o.groupby("period")]
    return sum(mats) / len(mats)


def first_last_stability(t: pd.DataFrame, cols: List[str]) -> pd.Series:
    a, b = t["period"].min(), t["period"].max()
    w = t.pivot_table(index="state", columns="period", values=cols)
    return pd.Series({c: w[(c, a)].corr(w[(c, b)]) for c in cols}, name=f"r_{a}_vs_{b}")


def main(config: str) -> None:
    cfg = load_config(config)
    res = Path(cfg["paths"]["results_dir"])
    t = pd.read_csv(res / "state_benchmark_table.csv")
    cols = [c for c in TRADITIONAL_SIGN if c in t.columns and t[c].notna().any()]
    m = oriented_within_period_corr(t, cols)
    stab = first_last_stability(t, cols)
    m.to_csv(res / "benchmark_intercorrelations.csv")
    stab.to_csv(res / "benchmark_stability.csv")
    pd.set_option("display.width", 250)
    short = {c: c.replace("ours_", "O:").replace("_sex_balanced", "_sb")[:14] for c in cols}
    print(f"== {config}: mean within-period r across states (oriented; > 0 = agree)")
    print(m.rename(index=short, columns=short).round(2).to_string())
    print(f"\n== stability of each measure across states ({stab.name})")
    print(stab.round(2).to_string())


if __name__ == "__main__":
    fire.Fire(main)
