# US 3DLNews2 report — where to start

Gender norms in US local news, by state and 10-year window (1995–2004,
2000–09, 2005–14, 2010–19, 2015–24), measured with word embeddings (Garg RND)
and compared with surveys (ACS, ATUS, Project Implicit).

**Orientation.** Everything in `analysis/` reads **higher = less traditional**
(maps: blue = less, red = more traditional). The methods tables (`tables.md`,
notes §3.3) show **raw RND**: > 0 = closer to female words.

## Reading order

1. **Methods and data** — [`../2026-10-05-us-3dlnews-notes.md`](../2026-10-05-us-3dlnews-notes.md):
   units and windows, corpus, word lists, how the score is computed, survey
   measures, weights, missing data, sample sizes, context predictors.
2. **Headline figures** — `analysis/figures_combined/` (Figures 1–6, below).
3. **Results text** — [`analysis/analysis_summary.md`](analysis/analysis_summary.md):
   all parts in one file, numbers and tables. The per-part files
   (`analysis/part*_summary.md`) are the same text split by part.
4. **Detail** — step-by-step figures and tables in `analysis/main/`, then the
   robustness checks in `analysis/robustness-*/` as needed.
5. **The plan** these results follow — [`2026-10-05-analysis-plan.md`](2026-10-05-analysis-plan.md).

## Headline figures (`analysis/figures_combined/`)

| File | Content | Plan step |
|---|---|---|
| `figure1_validation.pdf` | top: occupation RND vs ACS female share, state means; bottom: national trends (occupation, household work) | 1.1–1.3 |
| `figure2_survey.pdf` | text vs survey: state-window, between states, within states | 1.4–1.5 |
| `figure3_reliability.pdf` | text volume vs uncertainty; state-specific alignment slopes | 1.6–1.7 |
| `figure4_maps_{occupation,household}.pdf` | state maps, one per window | 2.1 |
| `figure5_dynamics_{occupation,household}.pdf` | state × window heatmap; change 2000–09 → 2015–24 by state | 2.2–2.3 |
| `figure6_explanatory.pdf` | state predictors of the text score: between states (state averages) and within states (state + window FE) | II-B |

## Folder structure

```
us_3dlnews_report/
├── README.md                    this file
├── 2026-10-05-analysis-plan.md  analysis plan (PI)
├── tables.md, tables/, figures/ methods statistics: data volume, training set-up,
│                                word-list coverage, survey measures, sample sizes,
│                                text scores per window (scripts/report_us_dlnews.py)
├── analysis/                    results (scripts/us_analysis/run.py)
│   ├── analysis_summary.md      all results in one file
│   ├── part1_summary.md         I    measurement validation (1.1–1.8)
│   ├── part2_summary.md         II   geography and dynamics (2.1–2.3, 2.6 policy timing)
│   ├── part2b_summary.md        II-B state predictors (socioeconomic, labour market,
│   │                                 policy, political & cultural)
│   ├── part2c_summary.md        II-C text–survey gap as an outcome
│   ├── part3_summary.md         III  case selection + state profiles (3.1–3.3)
│   ├── part3s_summary.md        III  nearest-neighbour words of selected terms
│   ├── figures_combined/        Figures 1–6
│   ├── main/figures, main/tables  every step, main measures; file names start
│   │                              with the plan step (1_4_..., 2b_..., 3_3_...)
│   ├── robustness-<measure>/    Part I steps 1.4–1.7 rerun with another survey
│   │                            measure or text category (list below)
│   └── data/                    analysis inputs: state × window panel,
│                                occupation cells, household-work terms
└── archive/                     deprecated tables and outputs (first report version;
                                 family-sphere-as-main analysis); do not cite
                                 (see archive/README.md)
```

## Main and robustness measures

| Domain | Text (main) | Survey (main) | Robustness folders |
|---|---|---|---|
| Occupation | occupation words (65 used) | ACS female share of our occupations | `duncan`, `female_emp_share`; `iat`, `explicit` (both domains) |
| Household work | household-work words (21 used) | ACS family index (6 measures) | `motherhood_emp_gap`, `motherhood_hours_gap`, `married_women_nilf`, `wife_earnings_share`, `wife_earns_more`, `gender_emp_gap`, `atus-housework`, `atus-household`, `atus-childcare`; `iat`, `explicit` |

`atus-*` use ATUS time-use shares (women's share of housework, household
activities, childcare) as the survey measure; `iat` / `explicit` are Project
Implicit (subjective) measures. The study has two text domains, occupation and
household work; the earlier family-sphere list (home, kids, marriage, ...) is
no longer used.

## Related documents (outside this folder)

- Word-list decisions and evidence: [`../research-log-wordlists.md`](../research-log-wordlists.md)
- Word lists and screening tables: `wordlists/en/occupation_family/`
  (`occupation_grounding.csv`, `household_screening.csv`)
