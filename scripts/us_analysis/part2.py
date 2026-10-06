"""Part II — geography and dynamics of text gender norms (analysis plan 2.1-2.3).

Scores: higher = less traditional (common.py); 0 = gender-neutral RND.
Colour: diverging RdBu, blue = less traditional, red = more traditional,
centred at 0, one fixed range per domain across all windows.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

from scripts.common.config_loader import load_config
from scripts.data_prep.us_state_mapper import normalize_state
from scripts.us_analysis.common import (
    CMAP, LESS_TRAD_COLOR, MORE_TRAD_COLOR, TEXT_LABEL, change_legend, md_table, save, text_egal, window_label,
)

TEXT_COLS = ["ours_occupation", "ours_family_sphere"]
REGION = {  # US Census regions
    "Northeast": ["connecticut", "maine", "massachusetts", "new_hampshire", "rhode_island",
                  "vermont", "new_jersey", "new_york", "pennsylvania"],
    "Midwest": ["illinois", "indiana", "michigan", "ohio", "wisconsin", "iowa", "kansas",
                "minnesota", "missouri", "nebraska", "north_dakota", "south_dakota"],
    "South": ["delaware", "district_of_columbia", "florida", "georgia", "maryland",
              "north_carolina", "south_carolina", "virginia", "west_virginia", "alabama",
              "kentucky", "mississippi", "tennessee", "arkansas", "louisiana", "oklahoma", "texas"],
    "West": ["arizona", "colorado", "idaho", "montana", "nevada", "new_mexico", "utah", "wyoming",
             "alaska", "california", "hawaii", "oregon", "washington"],
}
STATE_REGION = {s: r for r, ss in REGION.items() for s in ss}


def _scores(panel: pd.DataFrame, col: str) -> pd.DataFrame:
    return panel.assign(y=text_egal(panel, col), se=panel[f"se_{col}"])[
        ["state", "period", "y", "se"]].dropna(subset=["y"])


def maps(panel: pd.DataFrame, shapefile: Path, out: Path) -> str:
    import geopandas as gpd
    states = gpd.read_file(shapefile)
    states = states[~states["NAME"].isin(["Alaska", "Hawaii", "Puerto Rico"])]
    periods = sorted(panel["period"].unique())
    for col in TEXT_COLS:
        d = _scores(panel, col)
        d["NAME"] = d["state"].str.replace("_", " ").map(normalize_state)
        vmax = float(np.nanquantile(d["y"].abs(), 0.98))
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        ncol = 3
        nrow = int(np.ceil((len(periods) + 1) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.6 * nrow))
        axes = list(axes.flat)
        for ax, p in zip(axes, periods):
            g = states.merge(d[d["period"] == p][["NAME", "y"]], on="NAME", how="left")
            g.plot(column="y", ax=ax, cmap=CMAP, norm=norm, edgecolor="black", linewidth=0.2,
                   missing_kwds={"color": "#dddddd"})
            ax.set_title(window_label(p), fontsize=9)
            ax.set_axis_off()
        rest = axes[len(periods):]
        for ax in rest:
            ax.set_axis_off()
        sm = plt.cm.ScalarMappable(cmap=CMAP, norm=norm)
        fig.colorbar(sm, ax=rest[0], orientation="horizontal", fraction=0.4,
                     label="higher = less traditional (0 = neutral)")
        fig.suptitle(f"2.1 {TEXT_LABEL[col]} by state and window (common scale; grey = no "
                     "model; AK, HI not shown)", fontsize=10)
        fig.tight_layout()
        fig.savefig(out / "figures" / f"2_1_maps_{col.replace('ours_', '')}.pdf")
        plt.close(fig)
    return "**2.1 Maps.** One map per window, common scale per domain (2_1_maps_*.pdf).\n"


def heatmaps(panel: pd.DataFrame, out: Path) -> str:
    md = []
    for col in TEXT_COLS:
        d = _scores(panel, col)
        w = d.pivot_table(index="state", columns="period", values="y")
        first = w.apply(lambda r: r.dropna().iloc[0] if r.notna().any() else np.nan, axis=1)
        last = w.apply(lambda r: r.dropna().iloc[-1] if r.notna().any() else np.nan, axis=1)
        orders = {
            "average": w.mean(axis=1).sort_values().index,
            "baseline": first.sort_values().index,
            "final": last.sort_values().index,
            "region": (pd.DataFrame({"region": [STATE_REGION.get(s, "?") for s in w.index],
                                     "avg": w.mean(axis=1)}, index=w.index)
                       .sort_values(["region", "avg"]).index),
        }
        vmax = float(np.nanquantile(np.abs(w.values), 0.98))
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        for name, order in orders.items():
            m = w.loc[order]
            fig, ax = plt.subplots(figsize=(5.2, 10))
            im = ax.imshow(m.values, cmap=CMAP, norm=norm, aspect="auto")
            ax.set_xticks(range(m.shape[1]), [window_label(p) for p in m.columns], fontsize=7)
            labels = [s.replace("_", " ").title() + (f" ({STATE_REGION.get(s, '?')[:2]})"
                                                     if name == "region" else "") for s in m.index]
            ax.set_yticks(range(m.shape[0]), labels, fontsize=6)
            fig.colorbar(im, ax=ax, shrink=0.5, label="blue = less, red = more traditional (0 = neutral)")
            ax.set_title(f"2.2 {TEXT_LABEL[col]}: state x window\n(ordered by {name}; "
                         "white = no model)", fontsize=9)
            save(fig, out / "figures" / f"2_2_heatmap_{col.replace('ours_', '')}_{name}.pdf")
        w.reset_index().to_csv(out / "tables" / f"2_2_state_window_{col.replace('ours_', '')}.csv",
                               index=False)
        md.append(f"{TEXT_LABEL[col]}: orderings average (main), baseline, final, region")
    return "**2.2 State x window heatmaps.** " + "; ".join(md) + ".\n"


def change_ranking(panel: pd.DataFrame, out: Path) -> str:
    periods = sorted(panel["period"].unique())
    last = periods[-1]
    md = []
    for col in TEXT_COLS:
        d = _scores(panel, col)
        for base in periods[:-1]:
            a = d[d["period"] == base].set_index("state")
            b = d[d["period"] == last].set_index("state")
            ch = pd.DataFrame({"change": b["y"] - a["y"],
                               "se": np.sqrt(b["se"] ** 2 + a["se"] ** 2)}).dropna().sort_values("change")
            ch["lo"], ch["hi"] = ch["change"] - 1.96 * ch["se"], ch["change"] + 1.96 * ch["se"]
            tag = f"{col.replace('ours_', '')}_{base}_{last}"
            ch.reset_index().to_csv(out / "tables" / f"2_3_change_{tag}.csv", index=False)
            if base != periods[1]:   # main figure: second window (2000-09) -> last
                continue
            fig, ax = plt.subplots(figsize=(5.5, 9.5))
            y = np.arange(len(ch))
            colors = np.where(ch["lo"] > 0, LESS_TRAD_COLOR, np.where(ch["hi"] < 0, MORE_TRAD_COLOR, "#999999"))
            ax.errorbar(ch["change"], y, xerr=1.96 * ch["se"], fmt="none", ecolor=colors, elinewidth=0.8)
            ax.scatter(ch["change"], y, c=colors, s=12, zorder=3)
            ax.axvline(0, color="black", lw=0.6)
            ax.set_yticks(y, ch.index.str.replace("_", " ").str.title(), fontsize=6)
            ax.set_xlabel(f"Change {window_label(base)} → {window_label(last)}\n"
                          "(> 0 = less traditional; 95% interval)", fontsize=8)
            ax.set_title(f"2.3 {TEXT_LABEL[col]}: change by state", fontsize=9)
            change_legend(ax)
            save(fig, out / "figures" / f"2_3_change_{col.replace('ours_', '')}.pdf")
            md.append(f"{TEXT_LABEL[col]} {window_label(base)}→{window_label(last)}: "
                      f"{len(ch)} states, {int((ch['lo'] > 0).sum())} significantly less "
                      f"traditional, {int((ch['hi'] < 0).sum())} significantly more; median "
                      f"change {ch['change'].median():+.4f}")
    return "**2.3 Change ranking.** " + "; ".join(md) + ".\n"


def run_part2(panel: pd.DataFrame, config: str, out_root: Path) -> str:
    cfg = load_config(config)
    out = out_root / "main"
    for sub in ("figures", "tables"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    shp = Path(cfg["us_states"].get("shapefile", "data/shapefiles/us_states.shp"))
    md = ["# Part II — geography and dynamics\n",
          "Orientation: higher = less traditional; 0 = gender-neutral RND.\n",
          maps(panel, shp, out), heatmaps(panel, out), change_ranking(panel, out)]
    text = "\n".join(md)
    (out_root / "part2_summary.md").write_text(text, encoding="utf-8")
    return text
