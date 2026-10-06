"""Plan 3.1-3.2 — how the semantic neighbourhood of selected cases changes.

For each occupation / domestic- and care-work term selected in Part III and each window: the
`n_states` largest state models (most text), the entry's `topn` nearest
neighbours in each (singular/plural forms pooled, the entry's own forms
excluded), and neighbours that recur across those states.
Writes main/tables/3_semantic_neighbours.csv and part3s_summary.md.
Loads models: run on Slurm.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict

import pandas as pd

from scripts.analyze_garg import discover_models, load_model_for_unit
from scripts.common.config_loader import get_wordlist_dir, load_config
from scripts.diagnose_word_senses import nearest_neighbors
from scripts.us_analysis.common import window_label


def entry_forms(cfg: dict) -> Dict[str, str]:
    """label -> wordlist entry ('nurse' -> 'nurse|nurses') for every category."""
    wl = get_wordlist_dir(cfg)
    out = {}
    for fname in cfg["wordlists"]["categories"].values():
        for line in (Path(wl) / fname).read_text().splitlines():
            if line.strip():
                out[line.split("|")[0].strip()] = line.strip()
    return out


def run_semantics(config: str, out_root: Path, n_states: int = 5, topn: int = 15,
                  keep: int = 8) -> str:
    cfg = load_config(config)
    out = out_root / "main"
    cases = pd.read_csv(out / "tables" / "3_case_selection.csv")
    words = sorted(set(cases[cases["level"].isin(["occupation"]) |
                             cases["level"].str.startswith("term")]["case"]))
    forms = entry_forms(cfg)
    panel = pd.read_csv(out_root / "data" / "state_window_panel.csv")
    models = {u: p for p, u in discover_models(cfg)}
    rows = []
    for period, g in panel.groupby("period"):
        biggest = g.sort_values("tokens", ascending=False)["unit_name"].head(n_states)
        loaded = {u: load_model_for_unit(models[u], cfg) for u in biggest if u in models}
        for w in words:
            entry = forms.get(w, w)
            counts, sims, n_have = Counter(), {}, 0
            for u, m in loaded.items():
                nn = nearest_neighbors(m, entry, topn)
                if nn:
                    n_have += 1
                for nb, s in nn:
                    counts[nb] += 1
                    sims.setdefault(nb, []).append(s)
            for nb, c in counts.most_common():
                rows.append({"word": w, "period": period, "window": window_label(period),
                             "neighbour": nb, "n_states": c, "of_states": n_have,
                             "mean_similarity": sum(sims[nb]) / len(sims[nb])})
        del loaded
    t = pd.DataFrame(rows)
    t.to_csv(out / "tables" / "3_semantic_neighbours.csv", index=False)

    lines = ["# Part III (semantics) — nearest neighbours of selected cases\n",
             f"{n_states} largest state models per window; top {topn} neighbours each; listed: the "
             f"{keep} neighbours recurring in most states (count in brackets).\n"]
    for w in words:
        lines.append(f"**{w}**\n")
        for win, g in t[t["word"] == w].groupby("window", sort=False):
            g = g.sort_values(["n_states", "mean_similarity"], ascending=False).head(keep)
            lines.append(f"- {win}: " + ", ".join(f"{r.neighbour} ({r.n_states})"
                                                   for r in g.itertuples()))
        lines.append("")
    text = "\n".join(lines)
    (out_root / "part3s_summary.md").write_text(text, encoding="utf-8")
    return text
