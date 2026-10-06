"""Shared definitions for the US state-window analysis.

Canonical orientation (used in every figure and model):

    higher = LESS TRADITIONAL / less gender-stereotypical

- Text, occupation: +mean RND over occupations (occupation words closer to
  female words = less traditional).
- Text, domestic and care work (code key "household"): -mean RND over the
  domestic- and care-work words (closer to female words = more traditional).
- Survey measures: oriented with -check_state_benchmarks.TRADITIONAL_SIGN.

Occupation-level validation (Part I.1) keeps raw RND (> 0 = female-leaning)
against the female share, where that reading is the intuitive one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402

from scripts.check_state_benchmarks import TRADITIONAL_SIGN  # noqa: E402

# text category column -> sign that makes it "higher = less traditional"
TEXT_SIGN = {"ours_occupation": 1, "ours_household": -1}
# diverging colours: red = more traditional, blue = less traditional
CMAP = "RdBu"
LESS_TRAD_COLOR, MORE_TRAD_COLOR = "#2166ac", "#b2182b"
TEXT_LABEL = {"ours_occupation": "Text: occupation", "ours_household": "Text: domestic and care work"}
SURVEY_LABEL = {
    "matched_female_share": "ACS: female share of our occupations",
    "duncan": "ACS: occupational segregation (Duncan)",
    "female_emp_share": "ACS: women's share of employment",
    "family_index_acs": "ACS: family index",
    "motherhood_emp_gap": "ACS: motherhood employment gap",
    "motherhood_hours_gap": "ACS: motherhood hours gap",
    "married_women_nilf": "ACS: married women not in LF",
    "wife_earnings_share": "ACS: wife's earnings share",
    "wife_earns_more": "ACS: wife earns more",
    "gender_emp_gap": "ACS: gender employment gap",
    "iat_sex_balanced": "IAT: implicit career-family stereotype",
    "explicit_sex_balanced": "Project Implicit: explicit stereotype",
    "women_share_household": "ATUS: women's share of household activities",
    "women_share_housework": "ATUS: women's share of housework",
    "women_share_childcare_parents": "ATUS: women's share of childcare (parents)",
}
DOMAINS = ("occupation", "household")
# main text measure per domain
TEXT_COL = {"occupation": "ours_occupation", "household": "ours_household"}
DOMAIN_LABEL = {"occupation": "occupation", "household": "domestic and care work"}

# axis wording shared by the step figures (1.5-1.7) and the combined figures
BETWEEN_NOTE = "state average over windows"
MISMATCH_LABEL = "Text-survey mismatch: |survey - survey predicted from text| (SD)"
VOLUME_LABEL = "Text volume of the state-window model (log10 tokens)"
SLOPE_LABEL = ("State-specific slope of survey on text (z units)\n"
               "> 0: where text is less traditional, the survey is too")


# Direct benchmarks (same concept as the text measure): validation. Every one
# is reported; none is singled out as "main".
VALIDATION: Dict[str, List[str]] = {
    "occupation": ["matched_female_share", "iat_sex_balanced", "explicit_sex_balanced"],
    "household": ["women_share_housework", "women_share_household", "women_share_childcare_parents",
                  "iat_sex_balanced", "explicit_sex_balanced"],
}
# Related but different concepts (family roles, labour-market structure):
# correlates / mechanisms, reported as a coefficient table for both domains.
CORRELATES: List[str] = ["family_index_acs", "motherhood_emp_gap", "motherhood_hours_gap",
                         "married_women_nilf", "wife_earnings_share", "wife_earns_more",
                         "gender_emp_gap", "female_emp_share", "duncan"]


@dataclass
class Spec:
    """One reliability run (1.6-1.7): folder name + (text column, survey column) per domain."""
    name: str
    pairs: Dict[str, Tuple[str, str]] = field(default_factory=dict)


def validation_specs() -> List[Spec]:
    """One Spec per direct benchmark, covering every domain it validates."""
    surveys = list(dict.fromkeys(m for ms in VALIDATION.values() for m in ms))
    return [Spec(f"validation-{m}", {d: (TEXT_COL[d], m) for d in DOMAINS if m in VALIDATION[d]})
            for m in surveys]


def text_egal(panel: pd.DataFrame, col: str) -> pd.Series:
    """Text score oriented higher = less traditional."""
    return TEXT_SIGN[col] * panel[col]


def survey_egal(panel: pd.DataFrame, col: str) -> pd.Series:
    """Survey measure oriented higher = less traditional."""
    return -TRADITIONAL_SIGN.get(col, 1) * panel[col]


def survey_composite(panel: pd.DataFrame, dom: str) -> pd.Series:
    """Mean of the domain's direct benchmarks, each oriented (higher = less
    traditional) and z-scored over state-windows; mean of those available.
    Used where one survey value per state-window is needed (II-C, Part III)."""
    cols = [c for c in VALIDATION[dom] if c in panel.columns]
    return pd.concat([zscore(survey_egal(panel, c)) for c in cols], axis=1).mean(axis=1, skipna=True)


def zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std()


def window_label(p: int, width: int = 10) -> str:
    return f"{p}–{(p + width - 1) % 100:02d}"


def corr(x: pd.Series, y: pd.Series) -> Tuple[float, float, int]:
    ok = x.notna() & y.notna()
    n = int(ok.sum())
    if n < 3 or x[ok].std() == 0 or y[ok].std() == 0:
        return np.nan, np.nan, n
    r, p = pearsonr(x[ok], y[ok])
    return float(r), float(p), n


def scatter_fit(ax, x: pd.Series, y: pd.Series, xlabel: str, ylabel: str, title: str = "",
                s: int = 12, color: str = "#4c72b0", labels: Optional[pd.Series] = None) -> dict:
    """Scatter + OLS line with 95% CI band; annotates r, p and n. Returns the stats."""
    ok = x.notna() & y.notna()
    x, y = x[ok].astype(float), y[ok].astype(float)
    ax.scatter(x, y, s=s, color=color, alpha=0.75)
    if labels is not None:
        for xi, yi, lab in zip(x, y, labels[ok]):
            ax.annotate(str(lab), (xi, yi), fontsize=5, alpha=0.8)
    r, p, n = corr(x, y)
    rho, rho_p = spearmanr(x, y) if n >= 3 else (np.nan, np.nan)
    out = {"r": r, "p": p, "rho": float(rho), "rho_p": float(rho_p), "n": n, "slope": np.nan}
    if n >= 3:
        fit = sm.OLS(y, sm.add_constant(x)).fit()
        xs = np.linspace(x.min(), x.max(), 100)
        pred = fit.get_prediction(sm.add_constant(xs)).summary_frame(alpha=0.05)
        ax.plot(xs, pred["mean"], color="black", lw=1)
        ax.fill_between(xs, pred["mean_ci_lower"], pred["mean_ci_upper"], color="grey", alpha=0.25)
        out["slope"] = float(fit.params.iloc[1])
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title((title + "\n" if title else "") + f"r = {r:.2f} (p = {p:.3f}), Spearman ρ = {rho:.2f} (p = {rho_p:.3f}), n = {n}",
                 fontsize=9)
    ax.tick_params(labelsize=7)
    return out


def change_legend(ax) -> None:
    """Legend for change plots coloured by whether the 95% CI excludes 0."""
    ax.legend(handles=[plt.Line2D([], [], color=c, marker="o", ls="-", lw=0.7, ms=4, label=lab) for c, lab in (
        (LESS_TRAD_COLOR, "less traditional (95% CI above 0)"),
        (MORE_TRAD_COLOR, "more traditional (95% CI below 0)"),
        ("#999999", "no clear change (CI includes 0)"))], fontsize=6, loc="lower right")


def plot_state_slopes(ax, h: pd.DataFrame, global_slope: float, global_se: float,
                      slope_sd: float, fontsize: int = 6) -> bool:
    """Partially pooled state slopes (1.7), one row per state. When the state
    slopes barely vary (SD of state slopes < SE of the common slope), every
    state's interval is about the common interval, so per-state bars are not
    drawn: the common 95% interval is shown once as a band. Returns that flag."""
    pooled = bool(slope_sd < global_se)
    y = np.arange(len(h))
    ax.axvline(0, color="grey", lw=0.6, label="0 = text unrelated to survey")
    ax.axvline(global_slope, color="black", lw=0.8, ls="--", label="average slope over all states")
    if pooled:
        ax.axvspan(global_slope - 1.96 * global_se, global_slope + 1.96 * global_se, color="#4c72b0",
                   alpha=0.12, label="95% interval of the average slope")
        ax.scatter(h["slope"], y, color="#333333", s=9, zorder=3)
        note = (f"state slopes pooled to the average (SD across states {slope_sd:.3f} < SE "
                f"{global_se:.3f});\nper-state intervals equal the band and are not drawn")
    else:
        colors = np.where(h["lo"] > 0, LESS_TRAD_COLOR, np.where(h["hi"] < 0, MORE_TRAD_COLOR, "#888888"))
        ax.errorbar(h["slope"], y, xerr=[h["slope"] - h["lo"], h["hi"] - h["slope"]], fmt="none",
                    ecolor=colors, elinewidth=0.8)
        ax.scatter(h["slope"], y, c=colors, s=9, zorder=3)
        note = "partial pooling; 95% interval: blue > 0, red < 0, grey includes 0"
    ax.set_yticks(y, h["state"].str.replace("_", " ").str.title(), fontsize=fontsize)
    ax.set_xlabel(SLOPE_LABEL + "\n" + note, fontsize=8)
    ax.legend(fontsize=6, loc="lower right")
    return pooled


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def md_table(df: pd.DataFrame, digits: int = 3) -> str:
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].round(digits)
    lines = ["| " + " | ".join(map(str, d.columns)) + " |",
             "|" + "|".join("---" for _ in d.columns) + "|"]
    lines += ["| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |"
              for r in d.itertuples(index=False)]
    return "\n".join(lines)
