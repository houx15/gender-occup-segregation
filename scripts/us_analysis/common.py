"""Shared definitions for the US state-window analysis.

Canonical orientation (used in every figure and model):

    higher = MORE TRADITIONAL / more gender-stereotypical

- Text, occupation: -mean RND over occupations (occupation words closer to
  male words = more traditional).
- Text, family: +mean RND over family words (family words closer to female
  words = more traditional).
- Survey measures: oriented with check_state_benchmarks.TRADITIONAL_SIGN.

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
from scipy.stats import pearsonr  # noqa: E402

from scripts.check_state_benchmarks import TRADITIONAL_SIGN  # noqa: E402

# text category column -> sign that makes it "higher = more traditional"
TEXT_SIGN = {"ours_occupation": -1, "ours_family_sphere": 1, "ours_household": 1}
TEXT_LABEL = {"ours_occupation": "Text: occupation", "ours_family_sphere": "Text: family sphere",
              "ours_household": "Text: household work"}
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
}
DOMAINS = ("occupation", "family")


@dataclass
class Spec:
    """One validation run: folder name + (text column, survey column) per domain."""
    name: str
    pairs: Dict[str, Tuple[str, str]] = field(default_factory=dict)


MAIN = Spec("main", {"occupation": ("ours_occupation", "matched_female_share"),
                     "family": ("ours_family_sphere", "family_index_acs")})
ROBUSTNESS: List[Spec] = (
    [Spec(f"robustness-{m}", {"occupation": ("ours_occupation", m)})
     for m in ("duncan", "female_emp_share")]
    + [Spec(f"robustness-{m}", {"family": ("ours_family_sphere", m)})
       for m in ("motherhood_emp_gap", "motherhood_hours_gap", "married_women_nilf",
                 "wife_earnings_share", "wife_earns_more", "gender_emp_gap")]
    + [Spec("robustness-iat", {"occupation": ("ours_occupation", "iat_sex_balanced"),
                               "family": ("ours_family_sphere", "iat_sex_balanced")}),
       Spec("robustness-explicit", {"occupation": ("ours_occupation", "explicit_sex_balanced"),
                                    "family": ("ours_family_sphere", "explicit_sex_balanced")}),
       Spec("robustness-household-text", {"family": ("ours_household", "family_index_acs")})]
)


def text_trad(panel: pd.DataFrame, col: str) -> pd.Series:
    return TEXT_SIGN[col] * panel[col]


def survey_trad(panel: pd.DataFrame, col: str) -> pd.Series:
    return TRADITIONAL_SIGN.get(col, 1) * panel[col]


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
    out = {"r": r, "p": p, "n": n, "slope": np.nan}
    if n >= 3:
        fit = sm.OLS(y, sm.add_constant(x)).fit()
        xs = np.linspace(x.min(), x.max(), 100)
        pred = fit.get_prediction(sm.add_constant(xs)).summary_frame(alpha=0.05)
        ax.plot(xs, pred["mean"], color="black", lw=1)
        ax.fill_between(xs, pred["mean_ci_lower"], pred["mean_ci_upper"], color="grey", alpha=0.25)
        out["slope"] = float(fit.params.iloc[1])
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title((title + "\n" if title else "") + f"r = {r:.2f} (p = {p:.3f}), n = {n}", fontsize=9)
    ax.tick_params(labelsize=7)
    return out


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
