"""Robustness: national-average time trends on a balanced panel.

A national mean per window (mean over states) can move only because the set
of states changes between windows (e.g. a state first modelled in 2005-14).
This check keeps windows from ``start`` on (default 2000-09; 1995-2004 has
too few states) and only units observed in every one of them:

  text scores (1.2 trend)            states with the score in every window
  word trajectories (3.1, 3.2)       per word, states where the word is in
                                     vocab in every window

and compares each national mean with the all-states mean over the same
windows. The case picks are the main ones (main/tables/3_case_selection.csv).
Writes <out_root>/robustness-balanced-2000/ and returns a markdown summary.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.us_analysis.common import TEXT_COL, TEXT_LABEL, md_table, save, text_egal, window_label
from scripts.us_analysis.part3 import _rule_panels


def complete_units(d: pd.DataFrame, unit: str, value: str, periods) -> pd.DataFrame:
    """Rows of ``d`` whose ``unit`` has a non-missing ``value`` in every period."""
    ok = d.dropna(subset=[value]).groupby(unit)["period"].nunique()
    return d[d[unit].isin(ok[ok == len(periods)].index) & d["period"].isin(periods)]


def trend_table(panel: pd.DataFrame, periods) -> pd.DataFrame:
    rows = []
    for col in TEXT_COL.values():
        v = panel[panel["period"].isin(periods)].assign(y=lambda x: text_egal(x, col))
        for sample, g0 in (("all states", v), ("balanced", complete_units(v, "state", "y", periods))):
            for p, g in g0.groupby("period"):
                y = g["y"].dropna()
                rows.append({"text": col, "sample": sample, "window": window_label(p), "period": p,
                             "states": len(y), "mean": y.mean(), "ci95": 1.96 * y.std() / np.sqrt(len(y))})
    return pd.DataFrame(rows)


def trend_figure(tab: pd.DataFrame, path: Path) -> None:
    cols = list(TEXT_COL.values())
    fig, axes = plt.subplots(1, len(cols), figsize=(4.6 * len(cols), 3.9))
    for ax, col in zip(axes, cols):
        for (sample, style), dx in zip((("all states", dict(color="#999999", ls="--")),
                                        ("balanced", dict(color="#4c72b0", ls="-"))), (-0.06, 0.06)):
            g = tab[(tab["text"] == col) & (tab["sample"] == sample)]
            x = np.arange(len(g)) + dx
            ax.errorbar(x, g["mean"], yerr=g["ci95"], fmt="o", capsize=3, color=style["color"],
                        label=f"{sample} (n = {', '.join(map(str, g['states']))})")
            ax.plot(x, g["mean"], lw=0.9, **style)
        ax.set_xticks(np.arange(len(g)), g["window"], fontsize=7)
        ax.set_ylabel("Mean text score (higher = less traditional)", fontsize=8)
        ax.set_title(TEXT_LABEL[col], fontsize=9)
        ax.legend(fontsize=6)
    fig.suptitle("1.2 robustness: national mean per window, all states vs states observed in "
                 "every window (95% CI over states)", fontsize=9)
    save(fig, path)


def word_trajectories(cells: pd.DataFrame, word: str, periods) -> pd.DataFrame:
    """National mean RND per word and window: all states vs, per word, states
    with the word in every window. Index = word, columns = (sample, period)."""
    d = cells[cells["period"].isin(periods)]
    out = {}
    for sample, g0 in (("all states", d), ("balanced", None)):
        if g0 is None:
            g0 = pd.concat([complete_units(g, "state", "rnd", periods) for _, g in d.groupby(word)])
        w = g0.groupby([word, "period"])["rnd"].mean().unstack("period").reindex(columns=periods)
        n = g0.dropna(subset=["rnd"]).groupby(word)["state"].nunique()
        out[sample] = w.assign(states=n)
    return pd.concat(out, axis=1)


def compare_words(t: pd.DataFrame, periods) -> dict:
    a, b = t["all states"][periods], t["balanced"][periods]
    ch_a, ch_b = a[periods[-1]] - a[periods[0]], b[periods[-1]] - b[periods[0]]
    ok = ch_a.notna() & ch_b.notna()
    return {"n_words": int(ok.sum()), "r_change": float(np.corrcoef(ch_a[ok], ch_b[ok])[0, 1]),
            "same_sign_change": float((np.sign(ch_a[ok]) == np.sign(ch_b[ok])).mean()),
            "median_abs_diff": float((a - b).abs().stack().median()),
            "median_states_balanced": float(t["balanced"]["states"].median())}


def run_balanced(panel: pd.DataFrame, cells: pd.DataFrame, terms: pd.DataFrame, out_root: Path,
                 start: int = 2000) -> str:
    out = out_root / f"robustness-balanced-{start}"
    for sub in ("figures", "tables"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    periods = sorted(p for p in panel["period"].unique() if p >= start)

    tab = trend_table(panel, periods)
    tab.to_csv(out / "tables" / "1_2_temporal_balanced.csv", index=False)
    trend_figure(tab, out / "figures" / "1_2_temporal_balanced.pdf")
    wide = tab.pivot_table(index=["text", "sample"], columns="window", values="mean").reset_index()
    wide["change"] = wide[window_label(periods[-1])] - wide[window_label(periods[0])]
    wide["text"] = wide["text"].map(TEXT_LABEL)

    cases = pd.read_csv(out_root / "main" / "tables" / "3_case_selection.csv")
    hh = TEXT_COL["household"].replace("ours_", "")
    md_words, rows = [], []
    for name, d, word, level, fig_name, title in (
            ("occupations", cells, "occupation", "occupation", "3_1_occupation_cases_balanced.pdf",
             "3.1 robustness: selected occupations, national RND, states with the word in every window"),
            ("domestic- and care-work terms", terms[terms["category"] == hh], "term", f"term ({hh})",
             "3_2_household_terms_balanced.pdf",
             "3.2 robustness: domestic and care work terms, national RND, states with the word in every window")):
        t = word_trajectories(d, word, periods)
        flat = t.copy()
        flat.columns = [f"{s}_{window_label(p)}" if isinstance(p, (int, np.integer)) else f"{s}_{p}"
                        for s, p in flat.columns]
        flat.reset_index().to_csv(out / "tables" / f"{fig_name.replace('.pdf', '.csv')}", index=False)
        c = compare_words(t, periods)
        rows.append({"word_list": name, **c})
        sel = cases[cases["level"] == level].copy()
        lines = t["balanced"][periods].copy()
        if level.startswith("term"):
            sel["case"] = [f"{x} ({hh})" for x in sel["case"]]
            lines.index = [f"{x} ({hh})" for x in lines.index]
        _rule_panels(lines.dropna(how="all"), sel, title, out / "figures" / fig_name)
    comp = pd.DataFrame(rows)
    comp.to_csv(out / "tables" / "3_word_trajectories_balanced_vs_all.csv", index=False)

    text = "\n".join([
        f"# Robustness: balanced panel, {window_label(periods[0])} to {window_label(periods[-1])}\n",
        "National means per window recomputed on units observed in every window (text "
        "scores: states with the score in all windows; word trajectories: per word, states with "
        "the word in all windows), compared with all states over the same windows.\n",
        "### 1.2 National mean text score (higher = less traditional)\n",
        md_table(wide, 4) + "\n",
        "### 3.1 / 3.2 Word trajectories: balanced vs all states\n",
        "r_change: correlation across words of the first-to-last-window change; "
        "same_sign_change: share of words whose change has the same sign; median_abs_diff: "
        "median |all - balanced| over word-windows.\n",
        md_table(comp, 3) + "\n"])
    (out_root / "part_balanced_summary.md").write_text(text, encoding="utf-8")
    return text
