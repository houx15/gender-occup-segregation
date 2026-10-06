"""Combined figures following the plan's "Recommended Figure Structure".

Built from the saved per-step tables and the canonical panel, so they always
match the step figures. Written to <out_dir>/figures_combined/:
  figure1_validation.pdf     1.1 occupations, 1.3 state means; 1.2 trends (occupation | domestic and care work)
  figure2_survey.pdf         1.4 state-window, 1.5 between, 1.5 within (occupation | domestic and care work)
  figure3_reliability.pdf    1.6 volume vs error, 1.7 state slopes
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
import statsmodels.formula.api as smf
from matplotlib.colors import TwoSlopeNorm

from scripts.us_analysis.common import (
    BETWEEN_NOTE, CMAP, DOMAIN_LABEL, LESS_TRAD_COLOR, MAIN, MISMATCH_LABEL, MORE_TRAD_COLOR,
    SLOPE_LABEL, SURVEY_LABEL, TEXT_COL, TEXT_LABEL, VOLUME_LABEL, WITHIN_NOTE, change_legend, save,
    scatter_fit, survey_egal, text_egal, window_label, zscore,
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


def figure2(panel, out):
    fig, axes = plt.subplots(3, 2, figsize=(11, 14))
    for j, (dom, letters) in enumerate(zip(DOMS, ("ACE", "BDF"))):
        d, tcol, scol = _frame(panel, dom)
        between = d.groupby("state")[["text", "survey"]].mean()
        within = d[["text", "survey"]] - d.groupby("state")[["text", "survey"]].transform("mean")
        tlab, slab = TEXT_LABEL[tcol], SURVEY_LABEL[scol]
        scatter_fit(axes[0][j], d["survey"], d["text"], slab, tlab,
                    f"{letters[0]}. {DOMAIN_LABEL[dom]}: one point = one state in one window")
        scatter_fit(axes[1][j], between["survey"], between["text"], f"{slab}\n({BETWEEN_NOTE})",
                    f"{tlab}\n({BETWEEN_NOTE})",
                    f"{letters[1]}. {DOMAIN_LABEL[dom]}, between states: one point = one state")
        scatter_fit(axes[2][j], within["survey"], within["text"], f"{slab}\n({WITHIN_NOTE})",
                    f"{tlab}\n({WITHIN_NOTE})",
                    f"{letters[2]}. {DOMAIN_LABEL[dom]}, within states: one point = one state-window")
    fig.suptitle("Figure 2. Does the text score agree with the survey? (higher = less traditional)",
                 fontsize=11)
    fig.text(0.5, 0.004, "C, D: are states that are less traditional in the survey also less "
             "traditional in text?\nE, F: when a state moves between windows in the survey, does its "
             "text move the same way?  r > 0 = agreement.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(out / "figure2_survey.pdf")
    plt.close(fig)


def figure3(panel, main, out):
    fig, axes = plt.subplots(2, 2, figsize=(11, 12.5), gridspec_kw={"height_ratios": [1, 2.2]})
    for j, (dom, letters) in enumerate(zip(DOMS, ("AC", "BD"))):
        d, _, _ = _frame(panel, dom)
        d = d.dropna(subset=["tokens"]).copy()
        d["sz"], d["tz"] = zscore(d["survey"]), zscore(d["text"])
        fit = smf.ols("sz ~ tz", data=d).fit()
        scatter_fit(axes[0][j], np.log10(d["tokens"]), (d["sz"] - fit.fittedvalues).abs(),
                    VOLUME_LABEL, MISMATCH_LABEL,
                    f"{letters[0]}. {DOMAIN_LABEL[dom]}: is the mismatch larger where there is less text?")
        f = main / "tables" / f"1_7_state_slopes_{dom}.csv"
        if f.exists():
            h = pd.read_csv(f).sort_values("slope")
            glob = pd.read_csv(main / "tables" / "1_7_hierarchical.csv").set_index("domain")
            y = np.arange(len(h))
            colors = np.where(h["lo"] > 0, LESS_TRAD_COLOR, np.where(h["hi"] < 0, MORE_TRAD_COLOR, "#888888"))
            axes[1][j].errorbar(h["slope"], y, xerr=[h["slope"] - h["lo"], h["hi"] - h["slope"]],
                                fmt="none", ecolor=colors, elinewidth=0.7)
            axes[1][j].scatter(h["slope"], y, c=colors, s=6, zorder=3)
            axes[1][j].axvline(0, color="grey", lw=0.6, label="0 = text unrelated to survey")
            axes[1][j].axvline(glob.loc[dom, "global_slope"], color="black", lw=0.8, ls="--",
                               label="average slope over all states")
            axes[1][j].set_yticks(y, h["state"].str.replace("_", " ").str.title(), fontsize=4)
            axes[1][j].set_xlabel(SLOPE_LABEL, fontsize=8)
            axes[1][j].legend(fontsize=6, loc="lower right")
            axes[1][j].set_title(f"{letters[1]}. {DOMAIN_LABEL[dom]}: how closely does text track the survey "
                                 "in each state?", fontsize=9)
    fig.suptitle("Figure 3. Measurement reliability", fontsize=11)
    fig.text(0.5, 0.004, "C, D: one hierarchical model, slopes partially pooled toward the average; "
             "bars = 95% intervals (blue > 0, red < 0, grey includes 0).\nEach state has at most 5 "
             "windows, so most intervals include 0.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(out / "figure3_reliability.pdf")
    plt.close(fig)


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
    figure2(panel, out)
    figure3(panel, main, out)
    for dom, col in TEXT_COL.items():
        src = main / "figures" / f"2_1_maps_{col.replace('ours_', '')}.pdf"
        if src.exists():
            shutil.copy(src, out / f"figure4_maps_{dom}.pdf")
    figure5(panel, main, out)
    figure6(main, out)
    return f"Combined figures: {', '.join(sorted(p.name for p in out.glob('*.pdf')))}\n"
