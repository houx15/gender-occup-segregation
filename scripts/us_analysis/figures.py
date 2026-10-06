"""Combined figures following the plan's "Recommended Figure Structure".

Built from the saved per-step tables and the canonical panel, so they always
match the step figures. Written to <out_dir>/figures_combined/:
  figure1_validation.pdf     1.1 occupations, 1.3 state means; 1.2 trends (occupation | domestic and care work)
  figure2_validation_DOMAIN.pdf  1.4 every direct benchmark: pooled, state and time
                             dimensions, national trend (copied)
  figure3_reliability.pdf    1.6 mismatch vs volume, 1.7 alignment slope, every benchmark
  figure4_maps_<domain>.pdf  2.1 (copied)
  figure5_dynamics_<domain>.pdf  2.2 heatmap + 2.3 change ranking
  figure6_explanatory.pdf    II-B coefficients: between states (state averages) and
                             within states (state + window FE)
Orientation: higher = less traditional (state level).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

from scripts.us_analysis.common import (
    CMAP, DOMAIN_LABEL, DOMAINS, LESS_TRAD_COLOR, MORE_TRAD_COLOR, SURVEY_LABEL, TEXT_COL, TEXT_LABEL,
    change_legend, save, scatter_fit, window_label,
)
from scripts.us_analysis.part2b import BLOCK_COLOR

DOMS = list(DOMAINS)


def figure1(panel, main, out):
    occ = pd.read_csv(main / "tables" / "1_1_occupation_validity.csv")
    trend = pd.read_csv(main / "tables" / "1_2_temporal.csv")
    geo = pd.read_csv(main / "tables" / "1_3_geography.csv")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    scatter_fit(axes[0][0], occ["female_share"], occ["rnd"], "ACS female share of the occupation",
                "Text RND (> 0 = closer to female words)", "A. Occupations", labels=occ["occupation"])
    g = geo[geo["domain"] == "occupation"].sort_values("mean")
    y = np.arange(len(g))
    axes[0][1].errorbar(g["mean"], y, xerr=1.96 * g["se"], fmt="o", ms=2, color="#4c72b0",
                        ecolor="#9ab", elinewidth=0.7)
    axes[0][1].set_yticks(y, g["state"].str.replace("_", " ").str.title(), fontsize=4)
    axes[0][1].set_xlabel("Occupation text score, state mean over windows "
                          "(higher = less traditional)", fontsize=8)
    axes[0][1].set_title("B. State means (occupation; 95% interval)", fontsize=9)
    for ax, col, lab in ((axes[1][0], "ours_occupation", "C. Occupation trend"),
                         (axes[1][1], TEXT_COL["household"], "D. Domestic and care work trend")):
        g = trend[trend["text"] == col]
        x = np.arange(len(g))
        ax.errorbar(x, g["mean"], yerr=g["ci95"], fmt="o", capsize=3, color="#4c72b0")
        ax.plot(x, g["mean"], ls=":", lw=0.8, color="#4c72b0")
        ax.set_xticks(x, g["window"], fontsize=7)
        ax.set_ylabel("Mean text score (higher = less traditional)", fontsize=8)
        ax.set_title(f"{lab} (95% CI over states)", fontsize=9)
    fig.suptitle("Figure 1. External and descriptive validation", fontsize=11)
    save(fig, out / "figure1_validation.pdf")


def figure3(out_root, out):
    """Reliability across every direct benchmark: A. does the text-survey
    mismatch shrink with text volume (1.6)? B. average alignment slope and how
    much it varies across states (1.7)."""
    rows = []
    for f in sorted(out_root.glob("validation-*/tables/1_6_volume_error.csv")):
        survey = f.parts[-3].replace("validation-", "")
        v = pd.read_csv(f)
        h = pd.read_csv(f.parent / "1_7_hierarchical.csv")
        for dm in v["domain"]:
            r6, r7 = v[v["domain"] == dm].iloc[0], h[h["domain"] == dm].iloc[0]
            rows.append({"domain": dm, "survey": survey, "r6": r6["r_abs_error_log_tokens"], "n": r6["n"],
                         "slope": r7["global_slope"], "se": r7["global_se"], "sd": r7["slope_sd"]})
    if not rows:
        return
    t = pd.DataFrame(rows)
    t["label"] = [f"{DOMAIN_LABEL[d]}: {SURVEY_LABEL.get(s, s)}" for d, s in zip(t["domain"], t["survey"])]
    t = t.sort_values(["domain", "survey"]).reset_index(drop=True)
    y = np.arange(len(t))
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.45 * len(t) + 2.2), sharey=True)
    z, zse = np.arctanh(t["r6"]), 1 / np.sqrt(t["n"] - 3)        # Fisher CI for r
    axes[0].errorbar(t["r6"], y, xerr=[t["r6"] - np.tanh(z - 1.96 * zse), np.tanh(z + 1.96 * zse) - t["r6"]],
                     fmt="o", color="#4c72b0")
    axes[0].axvline(0, color="grey", lw=0.6)
    axes[0].set_yticks(y, t["label"], fontsize=8)
    axes[0].set_xlabel("r(|survey - survey predicted from text|, log10 tokens), 95% CI\n"
                       "< 0: less mismatch where there is more text", fontsize=8)
    axes[0].set_title("A. Mismatch vs text volume (1.6)", fontsize=9)
    axes[1].errorbar(t["slope"], y, xerr=1.96 * t["se"], fmt="o", color="#4c72b0")
    for yi, (sl, sd) in enumerate(zip(t["slope"], t["sd"])):
        axes[1].annotate(f"SD across states {sd:.2f}", (sl, yi), xytext=(0, 6), textcoords="offset points",
                         fontsize=6, ha="center")
    axes[1].axvline(0, color="grey", lw=0.6)
    axes[1].set_xlabel("Average slope of survey on text (z), 95% CI\n"
                       "label = SD of the state-specific slopes", fontsize=8)
    axes[1].set_title("B. Hierarchical alignment slope (1.7)", fontsize=9)
    fig.suptitle("Figure 3. Measurement reliability, every direct benchmark "
                 "(state-level slopes: validation-*/figures/1_7_state_slopes.pdf)", fontsize=10)
    save(fig, out / "figure3_reliability.pdf")


def figure5(panel, main, out):
    for dom, col in TEXT_COL.items():
        tag = col.replace("ours_", "")
        w = pd.read_csv(main / "tables" / f"2_2_state_window_{tag}.csv").set_index("state")
        w = w.loc[w.mean(axis=1).sort_values().index]
        changes = sorted((main / "tables").glob(f"2_3_change_{tag}_2000_*.csv"))
        fig, axes = plt.subplots(1, 2, figsize=(11, 10))
        vmax = float(np.nanquantile(np.abs(w.values), 0.98))
        im = axes[0].imshow(w.values, cmap=CMAP, norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax), aspect="auto")
        axes[0].set_xticks(range(w.shape[1]), [window_label(int(p)) for p in w.columns], fontsize=6)
        axes[0].set_yticks(range(w.shape[0]), w.index.str.replace("_", " ").str.title(), fontsize=5)
        fig.colorbar(im, ax=axes[0], shrink=0.5,
                     label="text score: blue = less, red = more traditional (0 = gender-neutral)")
        axes[0].set_title("A. Level: state x window (ordered by average; white = no model)", fontsize=9)
        if changes:
            ch = pd.read_csv(changes[0]).sort_values("change")
            y = np.arange(len(ch))
            colors = np.where(ch["lo"] > 0, LESS_TRAD_COLOR, np.where(ch["hi"] < 0, MORE_TRAD_COLOR, "#999999"))
            axes[1].errorbar(ch["change"], y, xerr=1.96 * ch["se"], fmt="none", ecolor=colors, elinewidth=0.7)
            axes[1].scatter(ch["change"], y, c=colors, s=8, zorder=3)
            axes[1].axvline(0, color="black", lw=0.6)
            axes[1].set_yticks(y, ch["state"].str.replace("_", " ").str.title(), fontsize=5)
            axes[1].set_xlabel("Change in text score, 2015–24 minus 2000–09 (95% CI)", fontsize=8)
            axes[1].set_title("B. Change 2000–09 → 2015–24 (> 0 = toward less traditional)", fontsize=9)
            change_legend(axes[1])
        fig.suptitle(f"Figure 5. State-level dynamics: {TEXT_LABEL[col]}", fontsize=11)
        save(fig, out / f"figure5_dynamics_{dom}.pdf")


SPEC_TITLE = {"between states": "between states: state averages over windows, one point per state",
              "within states (state + window FE)": "within states: state-windows with state and "
                                                   "window fixed effects"}


def figure6(main, out):
    f = main / "tables" / "2b_models.csv"
    if not f.exists():
        return
    res = pd.read_csv(f)
    res = res[res["model"] != "all blocks"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharey="row")
    for i, spec in enumerate(SPEC_TITLE):
        for ax, dom in zip(axes[i], DOMS):
            g = res[(res["domain"] == dom) & (res["spec"] == spec)].reset_index(drop=True)
            for k, r in g.iterrows():
                ax.errorbar(r["coef"], k, xerr=1.96 * r["se"], fmt="o", ms=4, color=BLOCK_COLOR[r["block"]])
            ax.axvline(0, color="grey", lw=0.6)
            ax.set_yticks(range(len(g)), g["term"], fontsize=7)
            ax.set_title(f"{DOMAIN_LABEL[dom]} text score — {SPEC_TITLE[spec]}\n(n = {g['n'].min()}–{g['n'].max()})",
                         fontsize=8)
            ax.set_xlabel("Standardized coefficient (95% CI), one model per block", fontsize=8)
    handles = [plt.Line2D([], [], color=c, marker="o", ls="", label=b) for b, c in BLOCK_COLOR.items()]
    axes[0][1].legend(handles=handles, fontsize=7)
    fig.suptitle("Figure 6. State characteristics and text gender norms (associational; "
                 "higher = less traditional)", fontsize=11)
    save(fig, out / "figure6_explanatory.pdf")


def run_figures(panel: pd.DataFrame, out_root: Path) -> str:
    main = out_root / "main"
    out = out_root / "figures_combined"
    out.mkdir(parents=True, exist_ok=True)
    figure1(panel, main, out)
    for dom in DOMS:   # Figure 2: validation against every direct benchmark (step 1.4)
        src = main / "figures" / f"1_4_validation_{dom}.pdf"
        if src.exists():
            shutil.copy(src, out / f"figure2_validation_{dom}.pdf")
    figure3(out_root, out)
    for dom, col in TEXT_COL.items():
        src = main / "figures" / f"2_1_maps_{col.replace('ours_', '')}.pdf"
        if src.exists():
            shutil.copy(src, out / f"figure4_maps_{dom}.pdf")
    figure5(panel, main, out)
    figure6(main, out)
    return f"Combined figures: {', '.join(sorted(p.name for p in out.glob('*.pdf')))}\n"
