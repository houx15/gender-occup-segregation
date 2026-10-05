"""Part II-C — text-survey discrepancy as an outcome (plan II-C).

gap_st = z(text score) - z(main survey measure), both oriented "higher = more
traditional" and standardized over state-windows. gap > 0: the state's news
text is more traditional than its survey measure suggests. Also the residual
from the survey-on-text calibration (survey_z ~ text_z) as an alternative.

Predictors: the Part II-B blocks plus log text volume, (a) pooled with window
FE (SE clustered by state) and (b) between states (state means, HC1).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from scripts.us_analysis.common import MAIN, md_table, save, survey_trad, text_trad, zscore
from scripts.us_analysis.part2b import BLOCK_COLOR, available_blocks, prepare


def add_gaps(panel: pd.DataFrame) -> pd.DataFrame:
    d = prepare(panel)
    for dom, (tcol, scol) in MAIN.pairs.items():
        t, s = text_trad(d, tcol), survey_trad(d, scol)
        ok = t.notna() & s.notna()
        tz, sz = zscore(t[ok]), zscore(s[ok])
        d.loc[ok, f"gap_{dom}"] = tz - sz
        fit = smf.ols("s ~ t", data=pd.DataFrame({"s": sz, "t": tz})).fit()
        d.loc[ok, f"resid_{dom}"] = fit.resid
    d["log_tokens"] = np.log10(d["tokens"])
    return d


def models(d: pd.DataFrame) -> pd.DataFrame:
    blocks = available_blocks(d)
    blocks["text volume"] = ["log_tokens"]
    xs_all = [v for vs in blocks.values() for v in vs]
    rows = []
    for dom in MAIN.pairs:
        y = f"gap_{dom}"
        data = d[["state", "period", y] + xs_all].dropna()
        z = data.copy()
        z[[y] + xs_all] = z[[y] + xs_all].apply(zscore)
        pooled = smf.ols(f"{y} ~ " + " + ".join(xs_all) + " + C(period)", data=z).fit(
            cov_type="cluster", cov_kwds={"groups": z["state"]})
        m = data.groupby("state")[[y] + xs_all].mean().apply(zscore)
        between = smf.ols(f"{y} ~ " + " + ".join(xs_all), data=m).fit(cov_type="HC1")
        for spec, fit, n in (("pooled + window FE", pooled, len(z)), ("between states", between, len(m))):
            for v in xs_all:
                rows.append({"domain": dom, "spec": spec, "block": next(b for b, vs in blocks.items() if v in vs),
                             "term": v, "coef": fit.params[v], "se": fit.bse[v], "p": fit.pvalues[v],
                             "n": n, "r2": fit.rsquared})
    return pd.DataFrame(rows)


def run_part2c(panel: pd.DataFrame, config: str, out_root: Path) -> str:
    out = out_root / "main"
    for sub in ("figures", "tables"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    d = add_gaps(panel)
    d[["unit_name", "state", "period"] + [c for c in d.columns if c.startswith(("gap_", "resid_"))]
      ].to_csv(out / "tables" / "2c_gaps.csv", index=False)

    # state ranking of the mean gap
    fig, axes = plt.subplots(1, 2, figsize=(10, 9.5))
    for ax, dom in zip(axes, MAIN.pairs):
        g = d.groupby("state")[f"gap_{dom}"].agg(["mean", "std", "count"]).dropna().sort_values("mean")
        y = np.arange(len(g))
        ax.errorbar(g["mean"], y, xerr=1.96 * g["std"] / np.sqrt(g["count"]), fmt="o", ms=3,
                    color="#4c72b0", ecolor="#9ab", elinewidth=0.8)
        ax.axvline(0, color="grey", lw=0.6)
        ax.set_yticks(y, g.index.str.replace("_", " ").str.title(), fontsize=6)
        ax.set_xlabel("Mean gap: z(text) − z(survey)\n(> 0 = text more traditional than survey)", fontsize=8)
        ax.set_title(dom, fontsize=9)
    fig.suptitle("2C Text-survey discrepancy by state (main survey measures)", fontsize=10)
    save(fig, out / "figures" / "2c_gap_states.pdf")

    if not available_blocks(prepare(panel)):
        return "# Part II-C\n\nGaps computed; no context predictors yet.\n"
    res = models(d)
    res.to_csv(out / "tables" / "2c_models.csv", index=False)
    for dom in MAIN.pairs:
        fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
        for ax, spec in zip(axes, ["pooled + window FE", "between states"]):
            g = res[(res["domain"] == dom) & (res["spec"] == spec)].reset_index(drop=True)
            for i, r in g.iterrows():
                ax.errorbar(r["coef"], i, xerr=1.96 * r["se"], fmt="o", ms=4,
                            color=BLOCK_COLOR.get(r["block"], "#8172b3"))
            ax.axvline(0, color="grey", lw=0.6)
            ax.set_yticks(range(len(g)), g["term"], fontsize=7)
            ax.set_title(f"{spec} (R² = {g['r2'].iloc[0]:.2f}, n = {g['n'].iloc[0]})", fontsize=9)
            ax.set_xlabel("Standardized coefficient (95% CI), all predictors jointly", fontsize=8)
        fig.suptitle(f"2C Predictors of the text-survey gap: {dom}", fontsize=10)
        save(fig, out / "figures" / f"2c_predictors_{dom}.pdf")
    sig = res[res["p"] < 0.05][["domain", "spec", "term", "coef", "se", "p"]]
    fit = res.groupby(["domain", "spec"]).agg(n=("n", "first"), r2=("r2", "first")).reset_index()
    text = "\n".join([
        "# Part II-C — text-survey discrepancy\n",
        "gap = z(text) − z(survey), main measures, higher = text more traditional than survey. "
        "All predictors jointly, standardized.\n",
        "### Model fit\n", md_table(fit) + "\n",
        "### Terms with p < 0.05\n", (md_table(sig) if len(sig) else "None.") + "\n"])
    (out_root / "part2c_summary.md").write_text(text, encoding="utf-8")
    return text
