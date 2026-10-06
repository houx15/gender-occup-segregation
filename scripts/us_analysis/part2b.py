"""Part II-B — explaining state differences in text gender norms (plan 2.4-2.7).

Outcome: text score per domain (higher = less traditional), standardized.
Predictors in theoretical blocks (standardized; built by
scripts/data_prep/build_context_measures.py, plus the ACS family measures and
the Duncan index). Shared blocks enter both domains; the gender blocks are
domain-specific (occupation-related for occupation, family-related for
domestic and care work):

  socioeconomic       (both) log real GDP per capita, log real income per adult, BA share,
                      metro share, unemployment, manufacturing share, service share
  gendered labour     (occupation) women's LFP, occupational segregation (Duncan),
  market              gender wage gap, women's share of managers / professionals
  workplace policy    (occupation) equal pay law, sexual-orientation employment protection
  family and care     (household) motherhood employment and hours gaps, married
                      women not in the labour force, wife's earnings share
  family policy       (household) paid family leave in effect (share of window
                      years), universal pre-K, abortion restrictions (count)
  political &         (both) Republican two-party presidential vote share, citizen
  cultural            ideology (Berry et al.), evangelical + LDS share
Several external predictors end before 2015-24 (see the notes); models use
complete cases, so blocks containing them have fewer state-windows.

Two specifications per block and for all blocks together:
  between   state means over windows, OLS (HC1 SEs)                 n = states
  within    state + window fixed effects, SEs clustered by state    n = state-windows
Associational: few windows per state, no causal design (plan 2.6).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from scripts.us_analysis.common import TEXT_COL, TEXT_LABEL, md_table, save, text_egal, zscore

SOCIOECONOMIC = ["log_real_gdp_pc", "log_real_income_pc", "ba_share", "metro_share",
                 "unemployment_rate", "manufacturing_share", "service_share"]
POLITICAL = ["gop_two_party_share", "citizen_ideology", "evangelical_lds_share"]
BLOCKS: Dict[str, Dict[str, List[str]]] = {
    "occupation": {
        "socioeconomic": SOCIOECONOMIC,
        "gendered labour market": ["women_lfp", "duncan", "gender_wage_gap",
                                   "female_share_managers", "female_share_professionals"],
        "workplace policy": ["equal_pay_law", "so_employment_law"],
        "political & cultural": POLITICAL,
    },
    "household": {
        "socioeconomic": SOCIOECONOMIC,
        "family and care": ["motherhood_emp_gap", "motherhood_hours_gap", "married_women_nilf",
                            "wife_earnings_share"],
        "family policy": ["pfl_share", "universal_prek", "abortion_restrictions"],
        "political & cultural": POLITICAL,
    },
}
BLOCK_COLOR = {"socioeconomic": "#4c72b0", "gendered labour market": "#dd8452",
               "workplace policy": "#55a868", "family and care": "#dd8452",
               "family policy": "#55a868", "political & cultural": "#c44e52"}
OUTCOMES = TEXT_COL


def prepare(panel: pd.DataFrame) -> pd.DataFrame:
    d = panel.copy()
    if "real_income_pc" in d:
        d["log_real_income_pc"] = np.log(d["real_income_pc"])
    return d


def available_blocks(d: pd.DataFrame, dom: str) -> Dict[str, List[str]]:
    """The domain's predictor blocks, restricted to variables present in the panel."""
    blocks = {b: [v for v in vs if v in d.columns and d[v].notna().any()]
              for b, vs in BLOCKS[dom].items()}
    return {b: vs for b, vs in blocks.items() if vs}


def block_legend(ax, blocks) -> None:
    ax.legend(handles=[plt.Line2D([], [], color=BLOCK_COLOR[b], marker="o", ls="", label=b)
                       for b in blocks], fontsize=7, loc="lower right")


def _fit(d: pd.DataFrame, y: str, xs: List[str], fe: bool):
    data = d[[y, "state", "period"] + xs].dropna()
    if fe:
        data = data.copy()
        data[[y] + xs] = data[[y] + xs].apply(zscore)
        rhs = " + ".join(xs) + " + C(state) + C(period)"
        fit = smf.ols(f"{y} ~ {rhs}", data=data).fit(cov_type="cluster",
                                                   cov_kwds={"groups": data["state"]})
        base = smf.ols(f"{y} ~ C(state) + C(period)", data=data).fit()
        return fit, fit.rsquared - base.rsquared, len(data)
    m = data.groupby("state")[[y] + xs].mean().apply(zscore)
    fit = smf.ols(f"{y} ~ " + " + ".join(xs), data=m).fit(cov_type="HC1")
    return fit, fit.rsquared, len(m)


def models(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dom, col in OUTCOMES.items():
        blocks = available_blocks(d, dom)
        if not blocks:
            continue
        specs = dict(blocks)
        specs["all blocks"] = [v for vs in blocks.values() for v in vs]
        dd = d.assign(y=text_egal(d, col))
        for spec_name, fe in (("between states", False), ("within states (state + window FE)", True)):
            for block, xs in specs.items():
                fit, r2, n = _fit(dd, "y", xs, fe)
                for v in xs:
                    rows.append({"domain": dom, "spec": spec_name, "model": block,
                                 "block": next(b for b, vs in blocks.items() if v in vs),
                                 "term": v, "coef": fit.params[v], "se": fit.bse[v],
                                 "p": fit.pvalues[v], "n": n,
                                 "r2_or_added_r2": r2})
    return pd.DataFrame(rows)


def coefficient_plot(res: pd.DataFrame, out: Path) -> None:
    single = res[res["model"] != "all blocks"]
    for dom in OUTCOMES:
        fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
        for ax, spec in zip(axes, ["between states", "within states (state + window FE)"]):
            g = single[(single["domain"] == dom) & (single["spec"] == spec)].reset_index(drop=True)
            y = np.arange(len(g))
            for i, r in g.iterrows():
                ax.errorbar(r["coef"], i, xerr=1.96 * r["se"], fmt="o", color=BLOCK_COLOR[r["block"]],
                            ms=4, elinewidth=1)
            ax.axvline(0, color="grey", lw=0.6)
            ax.set_yticks(y, g["term"], fontsize=7)
            ax.set_title(spec, fontsize=9)
            ax.set_xlabel("Standardized coefficient (95% CI), one model per block", fontsize=8)
        block_legend(axes[1], dict.fromkeys(single[single["domain"] == dom]["block"]))
        fig.suptitle(f"2B {TEXT_LABEL[OUTCOMES[dom]]} (higher = less traditional): state predictors",
                     fontsize=10)
        save(fig, out / "figures" / f"2b_predictors_{dom}.pdf")


def run_part2b(panel: pd.DataFrame, config: str, out_root: Path) -> str:
    out = out_root / "main"
    for sub in ("figures", "tables"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    d = prepare(panel)
    if not any(available_blocks(d, dom) for dom in OUTCOMES):
        return "# Part II-B\n\nNo context predictors in the panel (build_context_measures first).\n"
    res = models(d)
    res.to_csv(out / "tables" / "2b_models.csv", index=False)
    coefficient_plot(res, out)
    fit = (res.groupby(["domain", "spec", "model"])
           .agg(n=("n", "first"), r2_or_added_r2=("r2_or_added_r2", "first"),
                significant_terms=("p", lambda p: int((p < 0.05).sum())), terms=("p", "count"))
           .reset_index())
    fit.to_csv(out / "tables" / "2b_block_fit.csv", index=False)
    sig = res[(res["p"] < 0.05) & (res["model"] != "all blocks")][
        ["domain", "spec", "term", "coef", "se", "p"]]
    from scripts.us_analysis.policy_timing import policy_timing
    timing = policy_timing(panel, "config/policy/paid_family_leave.csv", out)
    text = "\n".join([
        "# Part II-B — explaining state differences\n",
        timing,
        "Outcome: text score (higher = less traditional); all variables standardized. "
        "Predictors: socioeconomic and political & cultural blocks for both domains; "
        "gendered labour market and workplace policy for occupation; family and care and "
        "family policy for domestic and care work. "
        "Between: state means, OLS (HC1). Within: state + window FE, SE clustered by state; "
        "fit column = added R2 over the FE-only model. Associational only.\n",
        "### Block fit\n", md_table(fit) + "\n",
        "### Terms with p < 0.05 (one model per block)\n",
        (md_table(sig) if len(sig) else "None.") + "\n"])
    (out_root / "part2b_summary.md").write_text(text, encoding="utf-8")
    return text
