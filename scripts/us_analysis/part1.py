"""Part I — measurement validation (analysis plan 1.1-1.8).

Orientation: state-level scores are "higher = less traditional" (common.py).
1.1-1.3 use only the text measure (main/). 1.4 validates each text score
against every direct benchmark (common.VALIDATION, including the subjective
IAT / explicit measures of plan 1.8) in three dimensions: pooled
state-windows, between states, within states (state + window FE). 1.9 reports
the related survey measures (common.CORRELATES) as a coefficient table.
1.6-1.7 (reliability) run per direct benchmark in validation-<measure>/.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from scripts.us_analysis.common import (
    BETWEEN_NOTE, CORRELATES, DOMAINS, DOMAIN_LABEL, MISMATCH_LABEL, SURVEY_LABEL, Spec, TEXT_COL,
    TEXT_LABEL, VALIDATION, VOLUME_LABEL, md_table, plot_state_slopes, save, scatter_fit,
    survey_egal, text_egal, validation_specs, window_label, zscore,
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


def _two_way_resid(d: pd.DataFrame, col: str) -> pd.Series:
    """col net of state and window means (residual on state + window FE)."""
    return smf.ols(f"{col} ~ C(state) + C(period)", data=d).fit().resid


def three_dimensions(d: pd.DataFrame) -> List[dict]:
    """Standardized slope of survey on text in the three dimensions (each is
    the slope of the matching scatter panel):
      pooled    state-windows, OLS, SE clustered by state
      between   state means over windows (re-standardized), OLS, HC1 SE
      within    state + window FE, SE clustered by state (= slope on the
                two-way demeaned values)"""
    out = []
    fit = smf.ols("survey_z ~ text_z", data=d).fit(cov_type="cluster", cov_kwds={"groups": d["state"]})
    out.append(("pooled", fit, len(d)))
    m = d.groupby("state")[["text_z", "survey_z"]].mean().apply(zscore)
    out.append(("between", smf.ols("survey_z ~ text_z", data=m).fit(cov_type="HC1"), len(m)))
    fit = smf.ols("survey_z ~ text_z + C(state) + C(period)", data=d).fit(
        cov_type="cluster", cov_kwds={"groups": d["state"]})
    out.append(("within", fit, len(d)))
    return [{"spec": name, "beta": f.params["text_z"], "se": f.bse["text_z"], "p": f.pvalues["text_z"],
             "n": n, "states": d["state"].nunique()} for name, f, n in out]


SPEC_NOTE = {"pooled": "Pooled", "between": "Between states", "within": "Within states"}
FE_ROWS = {"State FE": {"pooled": "No", "between": "—", "within": "Yes"},
           "Window FE": {"pooled": "No", "between": "—", "within": "Yes"}}


def _stars(p: float) -> str:
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def journal_table(res: pd.DataFrame, row_order: List[str], title: str, note: str) -> Tuple[str, str]:
    """Coefficient table in journal style: rows = survey measures, columns =
    domain x dimension; cells 'beta*** (SE)'. Returns (markdown, LaTeX)."""
    cols = [(dm, sp) for dm in DOMAINS for sp in SPEC_NOTE if ((res["domain"] == dm) & (res["spec"] == sp)).any()]
    cell = {(r.survey, r.domain, r.spec): (f"{r.beta:.2f}{_stars(r.p)}", f"({r.se:.2f})")
            for r in res.itertuples()}
    rows = [v for v in row_order if v in set(res["survey"])]
    head = ["Survey measure"] + [f"{DOMAIN_LABEL[dm].capitalize()}: {SPEC_NOTE[sp]}" for dm, sp in cols]
    md = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for v in rows:
        md.append("| " + " | ".join([SURVEY_LABEL.get(v, v)] + [
            " ".join(cell.get((v, dm, sp), ("", ""))) for dm, sp in cols]) + " |")
    n = res.groupby(["domain", "spec"])["n"].agg(["min", "max"])
    md.append("| Observations | " + " | ".join(
        f"{n.loc[(dm, sp), 'min']}" + (f"–{n.loc[(dm, sp), 'max']}" if n.loc[(dm, sp), 'max'] != n.loc[(dm, sp), 'min'] else "")
        for dm, sp in cols) + " |")
    for lab, m in FE_ROWS.items():
        md.append(f"| {lab} | " + " | ".join(m[sp] for _, sp in cols) + " |")
    md_text = f"**{title}**\n\n" + "\n".join(md) + f"\n\n{note}\n"
    tex = ["\\begin{table}[ht]\\centering\\small", f"\\caption{{{title}}}",
           "\\begin{tabular}{l" + "c" * len(cols) + "}", "\\toprule",
           " & ".join([""] + [f"\\multicolumn{{{sum(1 for d, _ in cols if d == dm)}}}{{c}}{{{DOMAIN_LABEL[dm].capitalize()}}}"
                              for dm in dict.fromkeys(d for d, _ in cols)]) + " \\\\",
           " & ".join(["Survey measure"] + [SPEC_NOTE[sp] for _, sp in cols]) + " \\\\", "\\midrule"]
    for v in rows:
        b = [cell.get((v, dm, sp), ("", ""))[0] for dm, sp in cols]
        se = [cell.get((v, dm, sp), ("", ""))[1] for dm, sp in cols]
        tex += [" & ".join([SURVEY_LABEL.get(v, v).replace("&", "\\&")] + [re.sub(r"(\*+)$", r"$^{\1}$", x) for x in b])
                + " \\\\", " & ".join([""] + se) + " \\\\"]
    tex += ["\\midrule", " & ".join(["Observations"] + [str(n.loc[(dm, sp), "max"]) for dm, sp in cols]) + " \\\\"]
    tex += [" & ".join([lab] + [m[sp] for _, sp in cols]) + " \\\\" for lab, m in FE_ROWS.items()]
    tex += ["\\bottomrule", "\\end{tabular}", f"\\par\\footnotesize {note}", "\\end{table}"]
    return md_text, "\n".join(tex)


TABLE_NOTE = ("Standardized coefficients of the survey measure on the text score (both oriented "
              "higher = less traditional; > 0 = agreement). Pooled: state-windows, SE clustered by "
              "state. Between: state means over windows, HC1 SE. Within: state and window fixed "
              "effects, SE clustered by state. * p < 0.05, ** p < 0.01, *** p < 0.001.")


def validation(panel: pd.DataFrame, out: Path) -> str:
    """Direct benchmarks: per domain one figure, one row per benchmark, panels
    A pooled, B state dimension, C time dimension (two-way demeaned), D the
    national trend (window means; 5 points, descriptive)."""
    rows = []
    for dm in DOMAINS:
        bench = [s for s in VALIDATION[dm] if s in panel.columns and panel[s].notna().any()]
        tcol = TEXT_COL[dm]
        fig, axes = plt.subplots(len(bench), 4, figsize=(19, 3.9 * len(bench)), squeeze=False)
        for i, s in enumerate(bench):
            d = _domain_frame(panel, tcol, s)
            slab, tlab = SURVEY_LABEL[s], TEXT_LABEL[tcol]
            ax = axes[i]
            scatter_fit(ax[0], d["survey_z"], d["text_z"], f"{slab} (z)", f"{tlab} (z)",
                        f"A. Pooled: one point = one state-window", s=8)
            m = d.groupby("state")[["text_z", "survey_z"]].mean()
            scatter_fit(ax[1], m["survey_z"], m["text_z"], f"{slab}\n({BETWEEN_NOTE})",
                        f"{tlab}\n({BETWEEN_NOTE})", "B. State dimension: one point = one state", s=10)
            scatter_fit(ax[2], _two_way_resid(d, "survey_z"), _two_way_resid(d, "text_z"),
                        f"{slab}\n(net of state and window means)", f"{tlab}\n(net of state and window means)",
                        "C. Time dimension: change beyond the national trend", s=8)
            g = d.groupby("period")[["text_z", "survey_z"]].mean()
            x = [window_label(int(p_)) for p_ in g.index]
            ax[3].plot(x, g["text_z"], "-o", color="#4c72b0", label="text")
            ax[3].plot(x, g["survey_z"], "--s", color="#dd8452", label="survey")
            r = g["text_z"].corr(g["survey_z"]) if len(g) > 2 else np.nan
            ax[3].set_title(f"D. National trend: mean over states per window\nr over {len(g)} windows = "
                            f"{r:.2f} (descriptive)", fontsize=9)
            ax[3].set_ylabel("mean z (higher = less traditional)", fontsize=8)
            ax[3].legend(fontsize=7)
            ax[3].tick_params(labelsize=7)
            ax[0].annotate(slab, xy=(-0.32, 0.5), xycoords="axes fraction", rotation=90, va="center",
                           ha="center", fontsize=9, fontweight="bold")
            rows += [dict(domain=dm, survey=s, **r_) for r_ in three_dimensions(d)]
        fig.suptitle(f"1.4 Validation against direct benchmarks: {DOMAIN_LABEL[dm]} "
                     "(both sides z, higher = less traditional; slope > 0 = agreement)", fontsize=11)
        save(fig, out / "figures" / f"1_4_validation_{dm}.pdf")
    res = pd.DataFrame(rows)
    res.to_csv(out / "tables" / "1_4_validation.csv", index=False)
    order = list(dict.fromkeys(s for ms in VALIDATION.values() for s in ms))
    md, tex = journal_table(res, order, "Validation: text scores against direct survey benchmarks",
                            TABLE_NOTE)
    (out / "tables" / "1_4_validation_table.md").write_text(md, encoding="utf-8")
    (out / "tables" / "1_4_validation_table.tex").write_text(tex, encoding="utf-8")
    return "### 1.4 Validation (direct benchmarks; pooled, state and time dimensions)\n\n" + md


def correlates(panel: pd.DataFrame, out: Path, drop_states: Tuple[str, ...] = (), tag: str = "") -> str:
    """Indirect survey measures (family roles, labour-market structure) against
    each text score: same three dimensions; Benjamini-Hochberg q over the
    correlates within each domain x dimension."""
    from statsmodels.stats.multitest import multipletests
    pnl = panel[~panel["state"].isin(drop_states)]
    rows = []
    for dm in DOMAINS:
        for s in CORRELATES:
            if s not in pnl.columns or pnl[s].notna().sum() < 10:
                continue
            rows += [dict(domain=dm, survey=s, **r_) for r_ in three_dimensions(_domain_frame(pnl, TEXT_COL[dm], s))]
    res = pd.DataFrame(rows)
    res["q_bh"] = res.groupby(["domain", "spec"])["p"].transform(lambda p_: multipletests(p_, method="fdr_bh")[1])
    res.to_csv(out / "tables" / f"1_9_correlates{tag}.csv", index=False)
    note = (TABLE_NOTE + " Benjamini–Hochberg q-values over the correlates within each column: "
            f"1_9_correlates{tag}.csv." + (f" Excluding: {', '.join(drop_states)}." if drop_states else ""))
    md, tex = journal_table(res, CORRELATES, "Correlates: text scores and related survey measures"
                            + (" (excluding DC)" if drop_states else ""), note)
    (out / "tables" / f"1_9_correlates_table{tag}.md").write_text(md, encoding="utf-8")
    (out / "tables" / f"1_9_correlates_table{tag}.tex").write_text(tex, encoding="utf-8")
    return md


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


def reliability(panel: pd.DataFrame, spec: Spec, out: Path) -> str:
    """1.6 discrepancy vs text volume and 1.7 state-specific slopes, for one
    direct benchmark (every domain it validates)."""
    out_f, out_t = out / "figures", out / "tables"
    doms = [dm for dm in DOMAINS if dm in spec.pairs]
    frames = {dm: _domain_frame(panel, *spec.pairs[dm]) for dm in doms}
    md = [f"## {spec.name}\n"]

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
                              "slope_sd": np.nan, "slopes_pooled": np.nan, "states_ci_excl_0": np.nan})
            continue
        h.assign(domain=dm).to_csv(out_t / f"1_7_state_slopes_{dm}.csv", index=False)
        pooled = plot_state_slopes(ax, h, h.attrs["global_slope"], h.attrs["global_se"],
                                   h.attrs["slope_sd"])
        ax.set_title(f"{DOMAIN_LABEL[dm]}: global slope {h.attrs['global_slope']:.2f} "
                     f"(SE {h.attrs['global_se']:.2f})", fontsize=9)
        hier_rows.append({"domain": dm, "global_slope": h.attrs["global_slope"],
                          "global_se": h.attrs["global_se"], "slope_sd": h.attrs["slope_sd"],
                          "slopes_pooled": pooled,
                          "states_ci_excl_0": int(((h["lo"] > 0) | (h["hi"] < 0)).sum())})
    fig.suptitle("1.7 How closely does text track the survey in each state? (exploratory)", fontsize=10)
    save(fig, out_f / "1_7_state_slopes.pdf")
    hier = pd.DataFrame(hier_rows)
    hier.to_csv(out_t / "1_7_hierarchical.csv", index=False)
    md += ["### 1.7 Hierarchical state-specific slopes\n", md_table(hier) + "\n"]
    return "\n".join(md)


def run_part1(panel: pd.DataFrame, cells: pd.DataFrame, out_root: Path) -> str:
    main = out_root / "main"
    for sub in ("figures", "tables"):
        (main / sub).mkdir(parents=True, exist_ok=True)
    md = ["# Part I — measurement validation\n",
          "Orientation: state-level scores are higher = less traditional.\n",
          occupation_validity(cells, main), temporal(panel, main), geography(panel, main),
          validation(panel, main),
          "### 1.9 Correlates (related survey measures; associations, not validation)\n\n"
          + correlates(panel, main),
          "Sensitivity without DC (an outlier on Duncan):\n\n"
          + correlates(panel, main, drop_states=("district_of_columbia",), tag="_no_dc"),
          "## Reliability per direct benchmark (1.6 – 1.7)\n"]
    for spec in validation_specs():
        folder = out_root / spec.name
        for sub in ("figures", "tables"):
            (folder / sub).mkdir(parents=True, exist_ok=True)
        md.append(reliability(panel, spec, folder))
    text = "\n".join(md)
    (out_root / "part1_summary.md").write_text(text, encoding="utf-8")
    return text
