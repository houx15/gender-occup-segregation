"""Part I — measurement validation (analysis plan 1.1-1.8).

Orientation: state-level scores are "higher = less traditional" (common.py).
1.1-1.3 use only the text measure and are written to main/. 1.4-1.7 are run
per Spec (main + robustness-*); 1.8 (subjective) is the robustness-iat and
robustness-explicit specs.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from scripts.us_analysis.common import (
    BETWEEN_NOTE, DOMAINS, DOMAIN_LABEL, LESS_TRAD_COLOR, MISMATCH_LABEL, MORE_TRAD_COLOR,
    SLOPE_LABEL, SURVEY_LABEL, Spec, TEXT_COL, TEXT_LABEL, VOLUME_LABEL, WITHIN_NOTE, corr,
    md_table, save, scatter_fit, survey_egal, text_egal, window_label, zscore,
)

# ---------------------------------------------------------------- 1.1 - 1.3 --
def occupation_validity(cells: pd.DataFrame, out: Path) -> str:
    """1.1 One point per occupation, pooled over states and windows."""
    occ = (cells.groupby("occupation")
           .agg(rnd=("rnd", "mean"), female_share=("female_share", "mean"),
                n_cells=("rnd", "count")).reset_index())
    occ.to_csv(out / "tables" / "1_1_occupation_validity.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 5.2))
    st = scatter_fit(ax, occ["female_share"], occ["rnd"], "ACS female share of the occupation "
                     "(pooled over states and windows)", "Text RND (> 0 = closer to female words)",
                     "1.1 Occupations: text gender association vs female share",
                     labels=occ["occupation"])
    save(fig, out / "figures" / "1_1_occupation_validity.pdf")
    return (f"**1.1 Occupation-level validity.** {st['n']} occupations, pooled over states and "
            f"windows: r = {st['r']:.2f} (p = {st['p']:.3g}). No domestic-work "
            "analogue: there is no per-term external benchmark for household words.\n")


def temporal(panel: pd.DataFrame, out: Path) -> str:
    """1.2 Mean text score per window (discrete waves), 95% CI over states."""
    rows = []
    cols = ["ours_occupation", "ours_household"]
    for col in cols:
        v = panel.assign(y=text_egal(panel, col))
        for p, g in v.groupby("period"):
            y = g["y"].dropna()
            rows.append({"text": col, "window": window_label(p), "period": p, "states": len(y),
                         "mean": y.mean(), "ci95": 1.96 * y.std() / np.sqrt(len(y))})
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "tables" / "1_2_temporal.csv", index=False)
    fig, axes = plt.subplots(1, len(cols), figsize=(4.4 * len(cols), 3.8))
    for ax, col in zip(axes, cols):
        g = tab[tab["text"] == col]
        x = np.arange(len(g))
        ax.errorbar(x, g["mean"], yerr=g["ci95"], fmt="o", color="#4c72b0", capsize=3)
        ax.plot(x, g["mean"], ls=":", color="#4c72b0", lw=0.8)
        ax.axhline(0, color="grey", lw=0.6)
        ax.set_xticks(x, g["window"], fontsize=7)
        ax.set_title(f"{TEXT_LABEL[col]}\n(n states: {', '.join(map(str, g['states']))})", fontsize=9)
        ax.set_ylabel("Text score (higher = less traditional)", fontsize=8)
    fig.suptitle("1.2 Mean text score per window (95% CI over states)", fontsize=10)
    save(fig, out / "figures" / "1_2_temporal.pdf")
    lines = []
    for col in cols:
        g = tab[tab["text"] == col]
        lines.append(f"{TEXT_LABEL[col]}: " + ", ".join(
            f"{w} {m:+.4f}" for w, m in zip(g["window"], g["mean"])))
    return "**1.2 Temporal variation** (higher = less traditional). " + "; ".join(lines) + ".\n"


def geography(panel: pd.DataFrame, out: Path) -> str:
    """1.3 State means over windows, sorted, with 95% intervals."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 9.5))
    rows = []
    for ax, dom in zip(axes, DOMAINS):
        col = TEXT_COL[dom]
        v = panel.assign(y=text_egal(panel, col), se=panel[f"se_{col}"])
        g = (v.groupby("state").agg(mean=("y", "mean"), k=("y", "count"),
                                    se=("se", lambda x: np.sqrt((x ** 2).sum()) / len(x)))
             .reset_index().sort_values("mean"))
        g["domain"] = dom
        rows.append(g)
        y = np.arange(len(g))
        ax.errorbar(g["mean"], y, xerr=1.96 * g["se"], fmt="o", ms=3, color="#4c72b0",
                    ecolor="#9ab", elinewidth=0.8)
        ax.axvline(g["mean"].mean(), color="black", lw=0.6, ls="--")
        ax.set_yticks(y, g["state"].str.replace("_", " ").str.title(), fontsize=6)
        ax.set_xlabel("Mean text score over windows (higher = less traditional)", fontsize=8)
        ax.set_title(TEXT_LABEL[col], fontsize=9)
    fig.suptitle("1.3 States sorted by average text score (95% interval from word bootstrap)",
                 fontsize=10)
    save(fig, out / "figures" / "1_3_geography.pdf")
    tab = pd.concat(rows)
    tab.to_csv(out / "tables" / "1_3_geography.csv", index=False)
    parts = []
    for dom in DOMAINS:
        g = tab[tab["domain"] == dom]
        parts.append(f"{DOMAIN_LABEL[dom]}: SD across states {g['mean'].std():.4f}, median 95% half-width "
                     f"{(1.96 * g['se']).median():.4f}")
    return "**1.3 Geographic heterogeneity.** " + "; ".join(parts) + ".\n"


# ---------------------------------------------------------------- 1.4 - 1.7 --
def _domain_frame(panel: pd.DataFrame, text_col: str, survey_col: str) -> pd.DataFrame:
    d = pd.DataFrame({"state": panel["state"], "period": panel["period"],
                      "text": text_egal(panel, text_col), "survey": survey_egal(panel, survey_col),
                      "tokens": panel["tokens"]}).dropna(subset=["text", "survey"])
    d["text_z"], d["survey_z"] = zscore(d["text"]), zscore(d["survey"])
    return d


def _models(d: pd.DataFrame) -> pd.DataFrame:
    """Survey = a + b Text (pooled) and with state + window FE; standardized,
    SEs clustered by state."""
    rows = []
    for name, formula in (("pooled", "survey_z ~ text_z"),
                          ("state + window FE", "survey_z ~ text_z + C(state) + C(period)")):
        fit = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["state"]})
        rows.append({"model": name, "beta_std": fit.params["text_z"], "se": fit.bse["text_z"],
                     "p": fit.pvalues["text_z"], "n": int(fit.nobs), "r2": fit.rsquared})
    return pd.DataFrame(rows)


def _hierarchical(d: pd.DataFrame) -> pd.DataFrame:
    """1.7 Survey_z = a_s + b_s Text_z, b_s ~ N(b, s^2): random intercepts and
    slopes by state (statsmodels MixedLM). State slope = global + BLUP; interval
    from the BLUP's conditional variance plus the global slope's variance."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = smf.mixedlm("survey_z ~ text_z", d, groups=d["state"],
                          re_formula="~text_z").fit(method=["lbfgs", "powell"], reml=True)
    b, b_se = fit.fe_params["text_z"], fit.bse_fe["text_z"]
    rows = []
    for st, re in fit.random_effects.items():
        cov = fit.random_effects_cov[st]
        var = float(cov.loc["text_z", "text_z"]) if "text_z" in cov.index else 0.0
        slope = b + float(re.get("text_z", 0.0))
        half = 1.96 * np.sqrt(var + b_se ** 2)
        rows.append({"state": st, "slope": slope, "lo": slope - half, "hi": slope + half,
                     "n_windows": int((d["state"] == st).sum())})
    out = pd.DataFrame(rows).sort_values("slope")
    out.attrs.update(global_slope=b, global_se=b_se,
                     slope_sd=float(np.sqrt(fit.cov_re.loc["text_z", "text_z"]))
                     if "text_z" in fit.cov_re.index else np.nan)
    return out


def survey_validation(panel: pd.DataFrame, spec: Spec, out: Path) -> str:
    out_f, out_t = out / "figures", out / "tables"
    doms = [dm for dm in DOMAINS if dm in spec.pairs]
    frames = {dm: _domain_frame(panel, *spec.pairs[dm]) for dm in doms}
    md = [f"## {spec.name}\n"]

    # 1.4 state-window scatter + models
    fig, axes = plt.subplots(1, len(doms), figsize=(5.2 * len(doms), 4.4), squeeze=False)
    model_rows = []
    for ax, dm, panel_lab in zip(axes[0], doms, "AB"):
        tcol, scol = spec.pairs[dm]
        d = frames[dm]
        scatter_fit(ax, d["survey"], d["text"], f"{SURVEY_LABEL[scol]}\n(higher = less traditional)",
                    f"{TEXT_LABEL[tcol]} (higher = less traditional)",
                    f"{panel_lab}. {DOMAIN_LABEL[dm]}: one point = one state in one 10-year window")
        model_rows.append(_models(d).assign(domain=dm, text=tcol, survey=scol))
    fig.suptitle("1.4 State-window alignment: text vs survey", fontsize=10)
    save(fig, out_f / "1_4_state_window.pdf")
    models = pd.concat(model_rows)[["domain", "text", "survey", "model", "beta_std", "se", "p",
                                    "n", "r2"]]
    models.to_csv(out_t / "1_4_models.csv", index=False)
    md += ["### 1.4 State-window alignment (standardized; SE clustered by state)\n",
           md_table(models) + "\n"]

    # 1.5 between / within
    fig, axes = plt.subplots(2, len(doms), figsize=(5.2 * len(doms), 8.4), squeeze=False)
    bw_rows = []
    for j, dm in enumerate(doms):
        tcol, scol = spec.pairs[dm]
        d = frames[dm]
        between = d.groupby("state")[["text", "survey"]].mean()
        within = d[["text", "survey"]] - d.groupby("state")[["text", "survey"]].transform("mean")
        b = scatter_fit(axes[0][j], between["survey"], between["text"],
                        f"{SURVEY_LABEL[scol]}\n({BETWEEN_NOTE})", f"{TEXT_LABEL[tcol]}\n({BETWEEN_NOTE})",
                        f"Between states — {DOMAIN_LABEL[dm]}: one point = one state")
        w = scatter_fit(axes[1][j], within["survey"], within["text"],
                        f"{SURVEY_LABEL[scol]}\n({WITHIN_NOTE})",
                        f"{TEXT_LABEL[tcol]}\n({WITHIN_NOTE})",
                        f"Within states — {DOMAIN_LABEL[dm]}: one point = one state-window")
        bw_rows += [{"domain": dm, "component": "between states", **b},
                    {"domain": dm, "component": "within states", **w}]
    fig.suptitle("1.5 Between- and within-state alignment (higher = less traditional)", fontsize=10)
    save(fig, out_f / "1_5_between_within.pdf")
    bw = pd.DataFrame(bw_rows)[["domain", "component", "r", "p", "n", "slope"]]
    bw.to_csv(out_t / "1_5_between_within.csv", index=False)
    md += ["### 1.5 Between- and within-state alignment\n", md_table(bw) + "\n"]

    # 1.6 measurement error vs text volume
    fig, axes = plt.subplots(2, len(doms), figsize=(5.2 * len(doms), 8.2), squeeze=False)
    vol_rows = []
    for j, dm in enumerate(doms):
        d = frames[dm].dropna(subset=["tokens"]).copy()
        fit = smf.ols("survey_z ~ text_z", data=d).fit()
        d["abs_error"] = (d["survey_z"] - fit.fittedvalues).abs()
        d["log_tokens"] = np.log10(d["tokens"])
        st = scatter_fit(axes[0][j], d["log_tokens"], d["abs_error"],
                         VOLUME_LABEL, MISMATCH_LABEL, f"{DOMAIN_LABEL[dm]}: is the mismatch larger where there is less text?")
        d["quintile"] = pd.qcut(d["log_tokens"], 5, labels=[1, 2, 3, 4, 5])
        q = d.groupby("quintile", observed=True)["abs_error"].agg(["mean", "std", "count"])
        axes[1][j].errorbar(q.index.astype(int), q["mean"], yerr=1.96 * q["std"] / np.sqrt(q["count"]),
                            fmt="o-", color="#4c72b0", capsize=3)
        axes[1][j].set_xlabel("Text-volume quintile (1 = least text)", fontsize=8)
        axes[1][j].set_ylabel("Mean |error| (95% CI)", fontsize=8)
        vol_rows.append({"domain": dm, "r_abs_error_log_tokens": st["r"], "p": st["p"],
                         "slope_per_log10_tokens": st["slope"], "n": st["n"],
                         "mean_abs_error_q1": q["mean"].iloc[0], "mean_abs_error_q5": q["mean"].iloc[-1]})
    fig.suptitle("1.6 Text-survey discrepancy vs text volume", fontsize=10)
    save(fig, out_f / "1_6_volume_error.pdf")
    vol = pd.DataFrame(vol_rows)
    vol.to_csv(out_t / "1_6_volume_error.csv", index=False)
    md += ["### 1.6 Discrepancy vs text volume\n", md_table(vol) + "\n"]

    # 1.7 hierarchical state-specific slopes
    fig, axes = plt.subplots(1, len(doms), figsize=(5.2 * len(doms), 9.5), squeeze=False)
    hier_rows = []
    for ax, dm in zip(axes[0], doms):
        try:
            h = _hierarchical(frames[dm])
        except Exception as e:  # noqa: BLE001
            ax.set_title(f"{DOMAIN_LABEL[dm]}: model did not converge ({type(e).__name__})", fontsize=8)
            hier_rows.append({"domain": dm, "global_slope": np.nan, "global_se": np.nan,
                              "slope_sd": np.nan, "states_ci_excl_0": np.nan})
            continue
        h.assign(domain=dm).to_csv(out_t / f"1_7_state_slopes_{dm}.csv", index=False)
        y = np.arange(len(h))
        colors = np.where(h["lo"] > 0, LESS_TRAD_COLOR, np.where(h["hi"] < 0, MORE_TRAD_COLOR, "#888888"))
        ax.errorbar(h["slope"], y, xerr=[h["slope"] - h["lo"], h["hi"] - h["slope"]], fmt="none",
                    ecolor=colors, elinewidth=0.8)
        ax.scatter(h["slope"], y, c=colors, s=9, zorder=3)
        ax.axvline(0, color="grey", lw=0.6, label="0 = text unrelated to survey")
        ax.axvline(h.attrs["global_slope"], color="black", lw=0.8, ls="--",
                   label="average slope over all states")
        ax.set_yticks(y, h["state"].str.replace("_", " ").str.title(), fontsize=6)
        ax.set_xlabel(SLOPE_LABEL + "\npartial pooling; 95% interval: blue > 0, red < 0, grey includes 0",
                      fontsize=8)
        ax.legend(fontsize=6, loc="lower right")
        ax.set_title(f"{DOMAIN_LABEL[dm]}: global slope {h.attrs['global_slope']:.2f} "
                     f"(SE {h.attrs['global_se']:.2f})", fontsize=9)
        hier_rows.append({"domain": dm, "global_slope": h.attrs["global_slope"],
                          "global_se": h.attrs["global_se"], "slope_sd": h.attrs["slope_sd"],
                          "states_ci_excl_0": int(((h["lo"] > 0) | (h["hi"] < 0)).sum())})
    fig.suptitle("1.7 How closely does text track the survey in each state? (exploratory)", fontsize=10)
    save(fig, out_f / "1_7_state_slopes.pdf")
    hier = pd.DataFrame(hier_rows)
    hier.to_csv(out_t / "1_7_hierarchical.csv", index=False)
    md += ["### 1.7 Hierarchical state-specific slopes\n", md_table(hier) + "\n"]
    return "\n".join(md)


def run_part1(panel: pd.DataFrame, cells: pd.DataFrame, specs: List[Spec], out_root: Path) -> str:
    main = out_root / "main"
    for sub in ("figures", "tables"):
        (main / sub).mkdir(parents=True, exist_ok=True)
    md = ["# Part I — measurement validation\n",
          "Orientation: state-level scores are higher = less traditional.\n",
          occupation_validity(cells, main), temporal(panel, main), geography(panel, main)]
    for spec in specs:
        folder = out_root / spec.name
        for sub in ("figures", "tables"):
            (folder / sub).mkdir(parents=True, exist_ok=True)
        md.append(survey_validation(panel, spec, folder))
    text = "\n".join(md)
    (out_root / "part1_summary.md").write_text(text, encoding="utf-8")
    return text
