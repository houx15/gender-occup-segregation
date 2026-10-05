"""Plan 2.6 — policy timing assessment before any DID / event-study design.

For each policy (currently: state paid family leave): adoption year, treated
states, which measurement windows each adoption precedes / falls inside, and
whether enough treated states change status between two observed windows.
Writes main/tables/2_6_policy_timing.csv and returns a markdown summary.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.us_analysis.common import md_table, window_label


def policy_timing(panel: pd.DataFrame, pfl_file: str, out: Path, width: int = 10) -> str:
    pfl = pd.read_csv(pfl_file)
    periods = sorted(panel["period"].unique())
    rows = []
    for r in pfl.itertuples():
        obs = set(panel[panel["state"] == r.state]["period"])
        status = {}
        for p in periods:
            end = p + width - 1
            status[window_label(p)] = ("before" if r.benefits_start_year > end else
                                       "after" if r.benefits_start_year <= p else "during")
        pre = [p for p in periods if p in obs and r.benefits_start_year > p + width - 1]
        post = [p for p in periods if p in obs and r.benefits_start_year <= p]
        rows.append({"state": r.state, "benefits_start": r.benefits_start_year, **status,
                     "observed_fully_before": len(pre), "observed_fully_after": len(post),
                     "usable_before_after": bool(pre and post)})
    t = pd.DataFrame(rows)
    t.to_csv(out / "tables" / "2_6_policy_timing.csv", index=False)
    n_usable = int(t["usable_before_after"].sum())
    verdict = ("too few treated states with a clean before and after window for a DID/event "
               "study; paid family leave enters Part II-B as a state-window exposure (share of "
               "window years with benefits), interpreted associationally"
               if n_usable < 5 else "enough treated states for a before/after comparison")
    return ("**2.6 Policy timing (paid family leave).** "
            f"{len(t)} treated states; {n_usable} have both a window entirely before and one "
            f"entirely after benefits began: {verdict}. Other policy domains in the plan "
            "(childcare, equal pay, discrimination protections, reproductive policy) are not "
            "yet collected.\n\n" + md_table(t) + "\n")
