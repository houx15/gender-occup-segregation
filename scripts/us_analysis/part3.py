"""Part III — programmatic case selection (plan 3.1-3.3, workflow step 8).

Every rule is explicit and its picks are saved to tables/3_case_selection.csv
before any interpretation.

Occupations (national RND = mean over state models, occupations in every window):
  stereotyping s = RND x sign(female share - 0.5)  (> 0: text leans to the
  occupation's majority gender). Rules: largest decline / increase in s
  (first -> last window), most stable RND (smallest SD over windows),
  reversal (RND changes sign, both |RND| > 0.005), largest |residual| from
  RND ~ female share.
Family terms: largest |change|, most stable, reversal, largest SD across states.
States (text score, higher = less traditional; baseline = second window,
  final = last): similar baseline / divergent final (pairs), similar
  socioeconomic structure / divergent change (pairs), largest move to less
  traditional, least change, largest |text - survey gap|, weakest / strongest
  partially pooled alignment slope (Part I.7).
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.us_analysis.common import MAIN, md_table, save, survey_egal, text_egal, window_label, zscore

K = 5
SOCIO = ["log_real_gdp_pc", "log_real_income_pc", "ba_share", "metro_share", "unemployment_rate",
         "manufacturing_share", "service_share"]


def _top(df, col, k=K, ascending=False):
    return df.sort_values(col, ascending=ascending).head(k)


def occupation_cases(cells: pd.DataFrame) -> pd.DataFrame:
    periods = sorted(cells["period"].unique())
    nat = cells.groupby(["occupation", "period"]).agg(rnd=("rnd", "mean"),
                                                     share=("female_share", "mean")).reset_index()
    full = nat.groupby("occupation")["period"].nunique()
    nat = nat[nat["occupation"].isin(full[full == len(periods)].index)]
    w = nat.pivot(index="occupation", columns="period", values="rnd")
    share = nat.groupby("occupation")["share"].mean()
    a, b = periods[0], periods[-1]
    sign = np.sign(share - 0.5)
    o = pd.DataFrame({"rnd_first": w[a], "rnd_last": w[b], "sd": w.std(axis=1),
                      "share": share, "stereo_change": (w[b] - w[a]) * sign})
    coef = np.polyfit(o["share"], w.mean(axis=1), 1)
    o["resid"] = w.mean(axis=1) - np.polyval(coef, o["share"])
    rows = []
    for crit, sel in (
            ("largest decline in stereotyping", _top(o, "stereo_change", ascending=True)),
            ("largest increase in stereotyping", _top(o, "stereo_change")),
            ("most stable", _top(o, "sd", ascending=True)),
            ("reversal in gender association",
             o[(np.sign(o["rnd_first"]) != np.sign(o["rnd_last"]))
               & (o["rnd_first"].abs() > 0.005) & (o["rnd_last"].abs() > 0.005)]),
            ("largest text vs female-share discrepancy", o.reindex(o["resid"].abs().sort_values(ascending=False).index).head(K))):
        for occ, r in sel.iterrows():
            rows.append({"domain": "occupation", "level": "occupation", "criterion": crit, "case": occ,
                         "value": r["stereo_change"] if "stereo" in crit else (
                             r["sd"] if crit == "most stable" else (r["resid"] if "discrepancy" in crit
                                                                    else r["rnd_last"])),
                         "details": f"RND {window_label(a)} {r['rnd_first']:+.4f} -> {window_label(b)} "
                                    f"{r['rnd_last']:+.4f}; female share {r['share']:.2f}"})
    return pd.DataFrame(rows), w


def family_cases(terms: pd.DataFrame) -> pd.DataFrame:
    periods = sorted(terms["period"].unique())
    nat = terms.groupby(["category", "term", "period"])["rnd"].mean().reset_index()
    w = nat.pivot_table(index=["category", "term"], columns="period", values="rnd")
    a, b = periods[0], periods[-1]
    het = terms.groupby(["category", "term"])["rnd"].std()
    o = pd.DataFrame({"first": w[a], "last": w[b], "change": w[b] - w[a], "sd": w.std(axis=1),
                      "state_sd": het}).dropna(subset=["first", "last"])
    rows = []
    for crit, sel, val in (
            ("largest temporal change", o.reindex(o["change"].abs().sort_values(ascending=False).index).head(3), "change"),
            ("most stable", _top(o, "sd", 3, ascending=True), "sd"),
            ("reversal", o[np.sign(o["first"]) != np.sign(o["last"])], "change"),
            ("strongest state heterogeneity", _top(o, "state_sd", 3), "state_sd")):
        for (cat, term), r in sel.iterrows():
            rows.append({"domain": "family", "level": f"term ({cat})", "criterion": crit, "case": term,
                         "value": r[val], "details": f"RND {window_label(a)} {r['first']:+.4f} -> "
                                                     f"{window_label(b)} {r['last']:+.4f}"})
    return pd.DataFrame(rows), w


def state_cases(panel: pd.DataFrame, out_root: Path) -> pd.DataFrame:
    periods = sorted(panel["period"].unique())
    base, final = periods[1], periods[-1]
    rows = []
    for dom, (tcol, scol) in MAIN.pairs.items():
        d = panel.assign(y=text_egal(panel, tcol), s=survey_egal(panel, scol))
        w = d.pivot_table(index="state", columns="period", values="y")
        sd = d["y"].std()
        both = w[[base, final]].dropna()
        ch = (both[final] - both[base]).rename("change")
        # (1) similar baseline, divergent final
        pairs = [(a_, b_, abs(both.loc[a_, base] - both.loc[b_, base]),
                  abs(both.loc[a_, final] - both.loc[b_, final])) for a_, b_ in combinations(both.index, 2)]
        p = pd.DataFrame(pairs, columns=["a", "b", "d_base", "d_final"])
        p = p[p["d_base"] < 0.25 * sd].sort_values("d_final", ascending=False).head(K)
        for r in p.itertuples():
            rows.append({"domain": dom, "level": "state pair", "criterion": "similar baseline, divergent final",
                         "case": f"{r.a} / {r.b}", "value": r.d_final,
                         "details": f"baseline gap {r.d_base:.4f}, final gap {r.d_final:.4f}"})
        # (2) similar socioeconomic structure, divergent change
        socio = [c for c in SOCIO if c in d.columns]
        if socio:
            dd = d.copy()
            if "real_income_pc" in dd and "log_real_income_pc" in socio:
                dd["log_real_income_pc"] = np.log(dd["real_income_pc"])
            x = dd[dd["period"] == base].set_index("state")[socio].dropna().apply(zscore)
            x = x.loc[x.index.intersection(ch.index)]
            pr = [(a_, b_, float(np.linalg.norm(x.loc[a_] - x.loc[b_])), abs(ch[a_] - ch[b_]))
                  for a_, b_ in combinations(x.index, 2)]
            pr = pd.DataFrame(pr, columns=["a", "b", "dist", "d_change"])
            pr = pr[pr["dist"] <= pr["dist"].quantile(0.1)].sort_values("d_change", ascending=False).head(K)
            for r in pr.itertuples():
                rows.append({"domain": dom, "level": "state pair",
                             "criterion": "similar socioeconomic structure, divergent change",
                             "case": f"{r.a} / {r.b}", "value": r.d_change,
                             "details": f"socioeconomic distance {r.dist:.2f} (z units), change gap {r.d_change:.4f}"})
        # (3) largest move toward less traditional, (4) little change
        for crit, sel in (("largest move toward less traditional", ch.sort_values(ascending=False).head(K)),
                          ("little change", ch.reindex(ch.abs().sort_values().index).head(K))):
            for st, v in sel.items():
                rows.append({"domain": dom, "level": "state", "criterion": crit, "case": st, "value": v,
                             "details": f"{window_label(base)} -> {window_label(final)}"})
        # (5) largest text-survey discrepancy
        ok = d["y"].notna() & d["s"].notna()
        d.loc[ok, "gap"] = zscore(d.loc[ok, "y"]) - zscore(d.loc[ok, "s"])
        g = d.groupby("state")["gap"].mean().dropna()
        for st, v in g.reindex(g.abs().sort_values(ascending=False).index).head(K).items():
            rows.append({"domain": dom, "level": "state", "criterion": "largest text-survey discrepancy",
                         "case": st, "value": v, "details": "mean z(text) - z(survey)"})
        # (6) weakest / strongest alignment (Part I.7)
        sl = out_root / "main" / "tables" / f"1_7_state_slopes_{dom}.csv"
        if sl.exists():
            s7 = pd.read_csv(sl).sort_values("slope")
            for crit, sel in (("weakest alignment (I.7)", s7.head(3)), ("strongest alignment (I.7)", s7.tail(3))):
                for r in sel.itertuples():
                    rows.append({"domain": dom, "level": "state", "criterion": crit, "case": r.state,
                                 "value": r.slope, "details": f"95% [{r.lo:.2f}, {r.hi:.2f}]"})
    return pd.DataFrame(rows)


def _trajectories(panel: pd.DataFrame, cases: pd.DataFrame, out: Path) -> None:
    for dom, (tcol, scol) in MAIN.pairs.items():
        d = panel.assign(y=zscore(text_egal(panel, tcol)), s=zscore(survey_egal(panel, scol)))
        sel = cases[(cases["domain"] == dom) & (cases["level"] == "state")]
        crits = list(dict.fromkeys(sel["criterion"]))
        fig, axes = plt.subplots(len(crits), 1, figsize=(9, 2.6 * len(crits)), squeeze=False)
        for ax, crit in zip(axes[:, 0], crits):
            for i, st in enumerate(sel[sel["criterion"] == crit]["case"]):
                g = d[d["state"] == st].sort_values("period")
                color = plt.cm.tab10(i)
                ax.plot(g["period"], g["y"], "-o", color=color, ms=3, label=st.replace("_", " ").title())
                ax.plot(g["period"], g["s"], "--", color=color, lw=0.8)
            ax.set_title(f"{crit} (solid = text, dashed = survey; z, higher = less traditional)", fontsize=8)
            ax.legend(fontsize=6, ncol=3)
            ax.tick_params(labelsize=7)
        fig.suptitle(f"3.3 State cases: {dom}", fontsize=10)
        save(fig, out / "figures" / f"3_3_state_cases_{dom}.pdf")


PROFILE = {
    "text volume": ["tokens"],
    "economy": ["log_real_gdp_pc", "ba_share", "metro_share", "unemployment_rate", "manufacturing_share"],
    "labour-market gender structure": ["women_lfp", "duncan", "gender_wage_gap", "female_share_professionals"],
    "policy": ["pfl_share", "universal_prek", "abortion_restrictions"],
    "political & cultural": ["gop_two_party_share", "citizen_ideology"],
}


def state_profiles(panel: pd.DataFrame, cases: pd.DataFrame) -> pd.DataFrame:
    """Plan 3.3: for each selected state, text and survey trajectory, text volume,
    economic, labour-market, policy and political context — first vs last window
    the state is observed in (missing where a source does not cover the window)."""
    states = sorted({st for c in cases[cases["level"].str.startswith("state")]["case"]
                     for st in str(c).split(" / ")})
    rows = []
    for st in states:
        g = panel[panel["state"] == st].sort_values("period")
        if g.empty:
            continue
        a, b = g.iloc[0], g.iloc[-1]
        row = {"state": st, "first_window": window_label(int(a["period"])),
               "last_window": window_label(int(b["period"])),
               "criteria": "; ".join(sorted(set(cases[cases["case"].str.contains(st, regex=False)]["criterion"])))}
        for dom, (tcol, scol) in MAIN.pairs.items():
            row[f"text_{dom}_first"], row[f"text_{dom}_last"] = (text_egal(g, tcol).iloc[0],
                                                                 text_egal(g, tcol).iloc[-1])
            row[f"survey_{dom}_first"], row[f"survey_{dom}_last"] = (survey_egal(g, scol).iloc[0],
                                                                     survey_egal(g, scol).iloc[-1])
        for vs in PROFILE.values():
            for v in vs:
                if v in g.columns:
                    row[f"{v}_first"], row[f"{v}_last"] = a[v], b[v]
        rows.append(row)
    return pd.DataFrame(rows)


def run_part3(panel: pd.DataFrame, cells: pd.DataFrame, terms: pd.DataFrame, out_root: Path) -> str:
    out = out_root / "main"
    for sub in ("figures", "tables"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    occ, w_occ = occupation_cases(cells)
    fam, w_fam = family_cases(terms)
    st = state_cases(panel, out_root)
    cases = pd.concat([occ, fam, st], ignore_index=True)
    cases.to_csv(out / "tables" / "3_case_selection.csv", index=False)

    # occupation trajectories of the selected occupations
    sel = list(dict.fromkeys(occ["case"]))
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, o_ in enumerate(sel):
        ax.plot([window_label(p) for p in w_occ.columns], w_occ.loc[o_], "-o", ms=3,
                color=plt.cm.tab20(i % 20), label=o_)
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_ylabel("National RND (> 0 = closer to female words)")
    ax.legend(fontsize=6, ncol=3)
    ax.set_title("3.1 Selected occupations: national RND by window", fontsize=10)
    save(fig, out / "figures" / "3_1_occupation_cases.pdf")
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (key, row) in enumerate(w_fam.iterrows()):
        ax.plot([window_label(p) for p in w_fam.columns], row, "-o", ms=3, color=plt.cm.tab20(i % 20),
                label=f"{key[1]} ({key[0]})")
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_ylabel("National RND (> 0 = closer to female words)")
    ax.legend(fontsize=6, ncol=3)
    ax.set_title("3.2 Family and household terms: national RND by window", fontsize=10)
    save(fig, out / "figures" / "3_2_family_terms.pdf")
    _trajectories(panel, st, out)
    prof = state_profiles(panel, cases)
    prof.to_csv(out / "tables" / "3_3_state_profiles.csv", index=False)

    show = ["state", "first_window", "last_window", "text_occupation_first", "text_occupation_last",
            "survey_occupation_first", "survey_occupation_last", "tokens_first", "tokens_last",
            "log_real_gdp_pc_first", "log_real_gdp_pc_last", "women_lfp_first", "women_lfp_last",
            "pfl_share_last", "gop_two_party_share_first", "gop_two_party_share_last"]
    text = "\n".join(["# Part III — case selection\n",
                      "Rules in scripts/us_analysis/part3.py; full table tables/3_case_selection.csv.\n",
                      md_table(cases) + "\n",
                      "### 3.3 State profiles (first vs last observed window; full table "
                      "tables/3_3_state_profiles.csv)\n",
                      md_table(prof[[c for c in show if c in prof.columns]]) + "\n"])
    (out_root / "part3_summary.md").write_text(text, encoding="utf-8")
    return text
