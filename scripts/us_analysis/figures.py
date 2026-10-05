"""Combined figures following the plan's "Recommended Figure Structure".

Built from the saved per-step tables and the canonical panel, so they always
match the step figures. Written to <out_dir>/figures_combined/:
  figure1_validation.pdf     1.1 occupations, 1.2 trends, 1.3 state heterogeneity
  figure2_survey.pdf         1.4 state-window, 1.5 between, 1.5 within (occupation | family)
  figure3_reliability.pdf    1.6 volume vs error, 1.7 state slopes
  figure4_maps_<domain>.pdf  2.1 (copied)
  figure5_dynamics_<domain>.pdf  2.2 heatmap + 2.3 change ranking
  figure6_explanatory.pdf    II-B between-state coefficients
Orientation: higher = less traditional (state level).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from matplotlib.colors import TwoSlopeNorm

from scripts.us_analysis.common import (
    CMAP, LESS_TRAD_COLOR, MAIN, MORE_TRAD_COLOR, SURVEY_LABEL, TEXT_LABEL, save, scatter_fit, survey_egal,
    text_egal, window_label, zscore,
)
from scripts.us_analysis.part2b import BLOCK_COLOR

DOMS = list(MAIN.pairs)


def _frame(panel, dom):
    tcol, scol = MAIN.pairs[dom]
    d = pd.DataFrame({"state": panel["state"], "period": panel["period"], "tokens": panel["tokens"],
                      "text": text_egal(panel, tcol), "survey": survey_egal(panel, scol)}).dropna(
        subset=["text", "survey"])
    return d, tcol, scol


def figure1(panel, main, out):
    occ = pd.read_csv(main / "tables" / "1_1_occupation_validity.csv")
    trend = pd.read_csv(main / "tables" / "1_2_temporal.csv")
    geo = pd.read_csv(main / "tables" / "1_3_geography.csv")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    scatter_fit(axes[0][0], occ["female_share"], occ["rnd"], "ACS female share of the occupation",
                "Text RND (> 0 = closer to female words)", "A. Occupations", labels=occ["occupation"])
    for ax, col, lab in ((axes[0][1], "ours_occupation", "B. Occupation trend"),
                         (axes[1][0], "ours_family_sphere", "C. Family trend")):
        g = trend[trend["text"] == col]
        x = np.arange(len(g))
        ax.errorbar(x, g["mean"], yerr=g["ci95"], fmt="o", capsize=3, color="#4c72b0")
        ax.plot(x, g["mean"], ls=":", lw=0.8, color="#4c72b0")
        ax.set_xticks(x, g["window"], fontsize=7)
        ax.set_ylabel("Mean text score (higher = less traditional)", fontsize=8)
        ax.set_title(f"{lab} (95% CI over states)", fontsize=9)
    g = geo[geo["domain"] == "occupation"].sort_values("mean")
    y = np.arange(len(g))
    axes[1][1].errorbar(g["mean"], y, xerr=1.96 * g["se"], fmt="o", ms=2, color="#4c72b0",
                        ecolor="#9ab", elinewidth=0.7)
    axes[1][1].set_yticks(y, g["state"].str.replace("_", " ").str.title(), fontsize=4)
    axes[1][1].set_xlabel("Mean occupation text score over windows", fontsize=8)
    axes[1][1].set_title("D. State heterogeneity (occupation)", fontsize=9)
    fig.suptitle("Figure 1. External and descriptive validation", fontsize=11)
    save(fig, out / "figure1_validation.pdf")


def figure2(panel, out):
    fig, axes = plt.subplots(3, 2, figsize=(11, 13))
    for j, dom in enumerate(DOMS):
        d, tcol, scol = _frame(panel, dom)
        between = d.groupby("state")[["text", "survey"]].mean()
        within = d[["text", "survey"]] - d.groupby("state")[["text", "survey"]].transform("mean")
        scatter_fit(axes[0][j], d["survey"], d["text"], SURVEY_LABEL[scol], TEXT_LABEL[tcol],
                    f"{dom}: state-windows")
        scatter_fit(axes[1][j], between["survey"], between["text"], "state mean", "state mean",
                    f"{dom}: between states")
        scatter_fit(axes[2][j], within["survey"], within["text"], "within-state deviation",
                    "within-state deviation", f"{dom}: within states")
    fig.suptitle("Figure 2. Survey validation (main measures; higher = less traditional)", fontsize=11)
    save(fig, out / "figure2_survey.pdf")


def figure3(panel, main, out):
    fig, axes = plt.subplots(2, 2, figsize=(11, 12), gridspec_kw={"height_ratios": [1, 2.2]})
    for j, dom in enumerate(DOMS):
        d, _, _ = _frame(panel, dom)
        d = d.dropna(subset=["tokens"]).copy()
        d["sz"], d["tz"] = zscore(d["survey"]), zscore(d["text"])
        fit = smf.ols("sz ~ tz", data=d).fit()
        scatter_fit(axes[0][j], np.log10(d["tokens"]), (d["sz"] - fit.fittedvalues).abs(),
                    "log10 tokens", "|survey − predicted| (SD)", f"{dom}: discrepancy vs text volume")
        f = main / "tables" / f"1_7_state_slopes_{dom}.csv"
        if f.exists():
            h = pd.read_csv(f).sort_values("slope")
            y = np.arange(len(h))
            axes[1][j].errorbar(h["slope"], y, xerr=[h["slope"] - h["lo"], h["hi"] - h["slope"]],
                                fmt="o", ms=2, color="#4c72b0", ecolor="#9ab", elinewidth=0.7)
            axes[1][j].axvline(0, color="grey", lw=0.6)
            axes[1][j].set_yticks(y, h["state"].str.replace("_", " ").str.title(), fontsize=4)
            axes[1][j].set_title(f"{dom}: partially pooled state slopes", fontsize=9)
    fig.suptitle("Figure 3. Measurement reliability", fontsize=11)
    save(fig, out / "figure3_reliability.pdf")


def figure5(panel, main, out):
    for dom, col in (("occupation", "ours_occupation"), ("family", "ours_family_sphere")):
        tag = col.replace("ours_", "")
        w = pd.read_csv(main / "tables" / f"2_2_state_window_{tag}.csv").set_index("state")
        w = w.loc[w.mean(axis=1).sort_values().index]
        changes = sorted((main / "tables").glob(f"2_3_change_{tag}_2000_*.csv"))
        fig, axes = plt.subplots(1, 2, figsize=(11, 10))
        vmax = float(np.nanquantile(np.abs(w.values), 0.98))
        im = axes[0].imshow(w.values, cmap=CMAP, norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax), aspect="auto")
        axes[0].set_xticks(range(w.shape[1]), [window_label(int(p)) for p in w.columns], fontsize=6)
        axes[0].set_yticks(range(w.shape[0]), w.index.str.replace("_", " ").str.title(), fontsize=5)
        fig.colorbar(im, ax=axes[0], shrink=0.5, label="higher = less traditional")
        axes[0].set_title("A. State x window (ordered by average)", fontsize=9)
        if changes:
            ch = pd.read_csv(changes[0]).sort_values("change")
            y = np.arange(len(ch))
            colors = np.where(ch["lo"] > 0, LESS_TRAD_COLOR, np.where(ch["hi"] < 0, MORE_TRAD_COLOR, "#999999"))
            axes[1].errorbar(ch["change"], y, xerr=1.96 * ch["se"], fmt="none", ecolor=colors, elinewidth=0.7)
            axes[1].scatter(ch["change"], y, c=colors, s=8, zorder=3)
            axes[1].axvline(0, color="black", lw=0.6)
            axes[1].set_yticks(y, ch["state"].str.replace("_", " ").str.title(), fontsize=5)
            axes[1].set_title("B. Change 2000–09 → 2015–24 (> 0 = less traditional)", fontsize=9)
        fig.suptitle(f"Figure 5. State-level dynamics: {TEXT_LABEL[col]}", fontsize=11)
        save(fig, out / f"figure5_dynamics_{dom}.pdf")


def figure6(main, out):
    f = main / "tables" / "2b_models.csv"
    if not f.exists():
        return
    res = pd.read_csv(f)
    res = res[(res["spec"] == "between states") & (res["model"] != "all blocks")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for ax, dom in zip(axes, DOMS):
        g = res[res["domain"] == dom].reset_index(drop=True)
        for i, r in g.iterrows():
            ax.errorbar(r["coef"], i, xerr=1.96 * r["se"], fmt="o", ms=4, color=BLOCK_COLOR[r["block"]])
        ax.axvline(0, color="grey", lw=0.6)
        ax.set_yticks(range(len(g)), g["term"], fontsize=7)
        ax.set_title(f"{dom} text score", fontsize=9)
        ax.set_xlabel("Standardized coefficient (95% CI), between states, one model per block", fontsize=8)
    handles = [plt.Line2D([], [], color=c, marker="o", ls="", label=b) for b, c in BLOCK_COLOR.items()]
    axes[1].legend(handles=handles, fontsize=7)
    fig.suptitle("Figure 6. State characteristics and text gender norms (associational)", fontsize=11)
    save(fig, out / "figure6_explanatory.pdf")


def run_figures(panel: pd.DataFrame, out_root: Path) -> str:
    main = out_root / "main"
    out = out_root / "figures_combined"
    out.mkdir(parents=True, exist_ok=True)
    figure1(panel, main, out)
    figure2(panel, out)
    figure3(panel, main, out)
    for dom, tag in (("occupation", "occupation"), ("family", "family_sphere")):
        src = main / "figures" / f"2_1_maps_{tag}.pdf"
        if src.exists():
            shutil.copy(src, out / f"figure4_maps_{dom}.pdf")
    figure5(panel, main, out)
    figure6(main, out)
    return f"Combined figures: {', '.join(sorted(p.name for p in out.glob('*.pdf')))}\n"
