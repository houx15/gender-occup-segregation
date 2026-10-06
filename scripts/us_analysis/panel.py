"""Canonical datasets for the US state-window analysis (plan, workflow steps 1-2).

Writes to <out_dir>/data/:
  state_window_panel.csv  one row per state-window: text scores (raw RND, CI
                          half-widths), every survey measure, text volume
  occupation_cells.csv    state x occupation x window: RND, ACS state and
                          national female share
  household_terms.csv     state x term x window: RND for household-work terms
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.check_state_occupations import merge_cells
from scripts.common.config_loader import load_config
from scripts.report_us_dlnews import count_tokens


def build_panel(config: str, out_dir: str) -> pd.DataFrame:
    cfg = load_config(config)
    res = Path(cfg["paths"]["results_dir"])
    start, end = cfg["analysis"]["decade_range"]
    data = Path(out_dir) / "data"
    data.mkdir(parents=True, exist_ok=True)

    # text scores + interval half-widths (68% bootstrap over words ~ 1 SE)
    s = pd.read_parquet(res / "garg_weat_summary_by_category.parquet")
    s["hw"] = (s["mean_ci_high"] - s["mean_ci_low"]) / 2
    hw = s.pivot_table(index="unit_name", columns="category", values="hw")
    hw.columns = [f"se_ours_{c}" for c in hw.columns]
    nw = s.pivot_table(index="unit_name", columns="category", values="n_occupations")
    nw.columns = [f"nwords_ours_{c}" for c in nw.columns]

    # survey measures (state_benchmark_table already has ours_* and every benchmark)
    t = pd.read_csv(res / "state_benchmark_table.csv")
    t = t.merge(hw.reset_index(), on="unit_name", how="left").merge(nw.reset_index(),
                                                                    on="unit_name", how="left")

    # text volume
    cov = pd.read_csv(res / "coverage_dlnews.csv")[["unit_name", "n_docs"]]
    corpora = Path(cfg["paths"]["corpora_dir"])
    cov["tokens"] = [count_tokens(corpora / u) if (corpora / u).is_dir() else float("nan")
                     for u in cov["unit_name"]]
    t = t.merge(cov, on="unit_name", how="left")
    # state context (Part II-B predictors), if built for these windows
    ctx_path = cfg.get("census_check", {}).get("context_file")
    if ctx_path and Path(ctx_path).exists():
        ctx = pd.read_csv(ctx_path).drop(columns=["STATEFIP"], errors="ignore")
        t = t.merge(ctx, on=["state", "period"], how="left")
    t = t[(t["period"] >= start) & (t["period"] <= end)].sort_values(["state", "period"])
    t.to_csv(data / "state_window_panel.csv", index=False)

    # word-level datasets
    long_df = pd.read_parquet(res / "garg_weat_rnd_long.parquet")
    cc = load_config(config)["census_check"]
    wc = pd.read_csv(res / "word_coverage.csv")
    used = wc[wc["used"]]
    occ = merge_cells(long_df,
                      pd.read_csv(Path(cc["shares_dir"]) / "occupation_female_share_state.csv"),
                      pd.read_csv(Path(cc["shares_dir"]) / "occupation_female_share_national.csv"),
                      set(used[used["category"] == "occupation"]["occupation"]))
    occ = occ[(occ["period"] >= start) & (occ["period"] <= end)]
    occ.to_csv(data / "occupation_cells.csv", index=False)

    fam = long_df[(long_df["category"] == "household") & long_df["in_vocab"]]
    fam = fam.merge(used[["category", "occupation"]], on=["category", "occupation"])
    parts = fam["unit_name"].str.rsplit("_", n=1)
    fam = fam.assign(state=parts.str[0], period=parts.str[1].astype(int)).rename(
        columns={"occupation": "term"})
    fam = fam[(fam["period"] >= start) & (fam["period"] <= end)]
    fam[["state", "period", "category", "term", "rnd"]].to_csv(data / "household_terms.csv", index=False)
    print(f"panel: {len(t)} state-windows; {len(occ)} occupation cells; {len(fam)} household-term cells")
    return t
