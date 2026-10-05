#!/usr/bin/env python3
"""Side-by-side comparison of US state arms (per-state 5y, per-state 10y, approach A ...).

Reads each profile's results_dir outputs (run analyze/check jobs first):
  unit_stability_summary.csv       words used, adjacent-period stability per category
  state_benchmark_correlations.csv agreement_r per ours x benchmark (> 0 = agree)
  state_benchmark_trend.csv        national means over a balanced state panel

Prints and writes <out_dir>/arm_comparison_{stability,agreement,trend}.csv.
diagnose_unit_stability is (re)run for any arm missing its summary.

Usage:
  python -m scripts.compare_arms \
    --configs=config/profiles/garg_weat_dlnews.yml,config/profiles/garg_weat_dlnews_w10.yml,config/profiles/garg_weat_dlnews_tagged.yml \
    --out_dir=/scratch/network/yh6580/gender-occup/results/us_arm_comparison
"""

from __future__ import annotations

from pathlib import Path

import fire
import pandas as pd

from scripts.check_state_benchmarks import add_agreement
from scripts.common.config_loader import load_config


def agreement_overview(corr: pd.DataFrame, arm: str) -> pd.DataFrame:
    """Per ours x survey: mean within-period agreement_r, change, pooled."""
    within = (corr[corr["scope"].str.startswith("period")]
              .groupby(["ours", "survey"])["agreement_r"].mean().rename("within_period_mean"))
    change = (corr[corr["scope"].str.startswith("change")]
              .set_index(["ours", "survey"])["agreement_r"].rename("change"))
    pooled = (corr[corr["scope"] == "pooled"]
              .set_index(["ours", "survey"])["agreement_r"].rename("pooled"))
    out = pd.concat([within, change, pooled], axis=1).reset_index()
    out.insert(0, "arm", arm)
    return out


def trend_overview(trend: pd.DataFrame, arm: str) -> pd.DataFrame:
    rows = []
    for col in [c for c in trend.columns if c not in ("period", "n_states")]:
        s = trend.sort_values("period")[col].reset_index(drop=True)
        diffs = s.diff().dropna()
        rows.append({"arm": arm, "measure": col, "first": s.iloc[0], "last": s.iloc[-1],
                     "monotone": bool((diffs > 0).all() or (diffs < 0).all()),
                     "n_states": int(trend["n_states"].iloc[0])})
    return pd.DataFrame(rows)


def main(configs: str, out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stab, agree, trend = [], [], []
    for cfg_path in [c.strip() for c in configs.split(",") if c.strip()]:
        cfg = load_config(cfg_path)
        arm = Path(cfg_path).stem.replace("garg_weat_", "")
        res = Path(cfg["paths"]["results_dir"])
        if not (res / "unit_stability_summary.csv").exists():
            from scripts.diagnose_unit_stability import main as stability_main
            stability_main(cfg_path)
        s = pd.read_csv(res / "unit_stability_summary.csv")
        s.insert(0, "arm", arm)
        stab.append(s)
        # recompute agreement_r from pearson_r (older outputs predate the column)
        corr = add_agreement(pd.read_csv(res / "state_benchmark_correlations.csv")
                             .drop(columns="agreement_r", errors="ignore"))
        agree.append(agreement_overview(corr, arm))
        trend.append(trend_overview(pd.read_csv(res / "state_benchmark_trend.csv"), arm))

    stab, agree, trend = (pd.concat(x, ignore_index=True) for x in (stab, agree, trend))
    stab.to_csv(out / "arm_comparison_stability.csv", index=False)
    agree.to_csv(out / "arm_comparison_agreement.csv", index=False)
    trend.to_csv(out / "arm_comparison_trend.csv", index=False)

    pd.set_option("display.width", 250)
    print("== Stability (adjacent-period correlation of state scores)")
    print(stab.pivot_table(index="category", columns="arm",
                           values=["n_words", "mean_adjacent_corr"]).round(3).to_string())
    print("\n== Agreement with benchmarks (> 0 = agree), mean over windows / change")
    print(agree.pivot_table(index=["ours", "survey"], columns="arm",
                            values=["within_period_mean", "change"]).round(2).to_string())
    print("\n== National trend of our scores")
    print(trend[trend["measure"].str.startswith("ours_")].round(4).to_string(index=False))


if __name__ == "__main__":
    fire.Fire(main)
