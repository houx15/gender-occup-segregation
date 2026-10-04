# Research log: wordlists for the US state-period arm (3DLNews2)

What we decided about the words we measure, why, and what the data said.
Newest entries at the bottom. Each entry gives the date, the decision, the
evidence (with the job / commit that produced it), and open questions.

Setup common to all entries unless stated otherwise:

- Corpus: 3DLNews2, Google-platform local **newspaper** articles (TV kept as a
  separate, thin robustness arm). Twitter downloaded but not used.
- Units: state x 5-year period (`us_states.year_bins: 5`), labelled by the
  period start (`ohio_2010` = 2010-2014). One word2vec model per unit
  (300d, window 5, `min_count` 20); units with < 500 articles are not trained.
- Gender anchors: Garg et al. (2018) male / female word lists
  (`gender_words.json`, 20 + 20 words).
- Score: relative norm distance (RND, Garg et al. 2018) of each list word to
  the female vs male centroid; > 0 = female-leaning. `ideation_sign` flips
  family-type categories so that higher = less traditional.
- Research focus: **change** in gender ideation over time, by state. Low
  correlation of state rankings between adjacent periods is acceptable.

---

## 2026-10-04 — Baseline: COHA-derived lists (leadership / family / science)

**Lists.** `wordlists/en/garg_weat/cleaned_{leadership,family,science}.txt`
(70 / 21 / 54 words), originally pruned against COHA models.

**Rule.** Consistent set: a word is used only if it is in vocab in *every*
analyzed unit.

**Evidence** (237 newspaper models, 1995-2024):

| Category   | words used | adjacent-period state corr. |
|------------|-----------:|----------------------------:|
| family     | 2 / 21     | 0.36 |
| leadership | 12 / 71    | 0.52 |
| science    | 0 / 55     | — |

The thinnest units (`*_1995`, ~100 articles) veto most words.

## 2026-10-04 — Re-prune the large candidate pools against 3DLNews2

**Change.** `prepare_wordlists` on `candidates_*.txt` (445 / 213 / 442 words),
keeping words in vocab in all 237 newspaper models -> `dlnews_cleaned_*`.

**Evidence** (job 3391352): leadership 27, family 3, science 7 words.
Stability did not improve (leadership 0.42, family -0.01, science 0.05).
Words like *research, science, teacher, technology* were in 236/237 models
and dropped for missing in one thin 1995 unit.

**Takeaway.** Coverage is the binding constraint; the family candidates are
mostly rare chores/objects that local news seldom uses.

## 2026-10-04 — Add `family_sphere` (common family words)

**Change.** New category `family_sphere`: Caliskan et al. (2017) WEAT-6
family words (home, parents, children, family, cousins, marriage, wedding,
relatives) + household, kitchen. Gender-neutral, frequent in news. Sign -1.

**Evidence** (job 3391366): only 4/10 survived the every-unit rule
(children, family, home, parents); still better stability (0.35) than the
chore-based family list.

## 2026-10-04 — Refocus: occupation (primary) + family

**Decision (PI).** Drop leadership / science / chore-family as headline
measures. Focus on **occupation** (most important; start from Garg et al.
2018) and **family**. Words must be grounded and must have good coverage in
this corpus.

**Occupation list** — `wordlists/en/occupation_family/occupation_grounding.csv`
documents every candidate (word, plural, include, 2018 Census occupation title,
Garg source word, note):

- Start: Garg's 76-word list plus the 124 occupations in his census
  female-share file (`wordlists/en/garg/occupation_percentages_gender.csv`,
  1850-2015). Keeping Garg's word in `garg_word` lets us reuse his female
  shares to validate the embedding scores.
- Replaced Garg's artificial neutral forms with the words news actually uses:
  fireperson -> firefighter, newsperson -> journalist, bankteller -> teller.
- Excluded (with reason in the CSV): obsolete (bootblack, milliner,
  typesetter...), common surnames (smith, mason, porter, baker, cook),
  ambiguous (pilot, guard, conductor, collector, server, developer),
  gendered pairs (actor, waiter), not occupations (student, retired, sales,
  official, clerical, unemployed).
- Added common modern occupations (programmer, paramedic, receptionist,
  paralegal, nanny, caregiver, realtor, reporter, prosecutor, ...).
- Result: 134 candidates. Kept with a caveat: *secretary* (Secretary of
  State sense), *supervisor* (elected county supervisor) — Garg comparability.

**Household list** — `candidates_household.txt`: 20 unpaid-work / care words
from ATUS activity categories (housework, laundry, cooking, childcare,
daycare, caregiving, ...).

**Window.** Analyze 2005-09 .. 2020-24 only (`analysis.decade_range`); earlier
periods cover 8-32 states. `decade_range` was extended to parse
`{state}_{period}` units, and `prepare_wordlists` now uses the same clip.

**Evidence** (job 3391382, every-unit rule over 197 models):

| Category      | kept |
|---------------|-----:|
| occupation    | 10 / 134 (attorney, coach, driver, editor, executive, judge, manager, police, secretary, writer) |
| household     | 0 / 20 |
| family_sphere | 4 / 10 |

Singular coverage (share of models with the word): teacher 0.99,
reporter 0.99, nurse 0.91, engineer 0.92, firefighter 0.83, cashier 0.16,
plumber 0.05; household: cooking 0.89, cleaning 0.86, laundry 0.62,
housework / homemaking 0.00.

**Takeaway.** The every-unit rule is unworkable with small state-period
models; the surviving occupations are generic titles that cannot measure
segregation.

## 2026-10-04 — Plural pooling + word fixed effects

**Change 1 — plural pooling.** Wordlist entries may list surface forms,
`nurse|nurses`. The entry vector is the mean of the L2-normalized in-vocab
forms (re-normalized); the entry counts as in vocab if any form is. Local
news often uses the plural (*nurses, teachers, plumbers*). Plurals are in the
`plural` column of `occupation_grounding.csv`; family/household lists carry
their own variants. (`scripts/common/metrics.py: entry_vector`)

**Change 2 — word fixed effects** (`analysis.word_set: fixed_effects`,
`min_word_coverage: 0.5`). Use every word in vocab in >= 50% of analyzed
units. Each unit's level comes from `value[u, w] = a[u] + b[w] + e` with
mean(b) = 0, so a unit is not shifted by which words it happens to have; on a
balanced panel `a[u]` is exactly the plain mean used before. CIs resample
words and refit. Per-word coverage is written to
`<results_dir>/word_coverage.csv`. (`scripts/common/fixed_effects.py`)

**Evidence** (job 3391395, commit 3023aef; 197 newspaper models 2005-2024):

| Category      | words used (cov. >= 0.5) | before (every-unit rule) | adjacent-period corr. | states w/ significant change 2005-09 -> 2020-24 |
|---------------|---------------:|-----:|-----:|--------:|
| occupation    | 56 / 134 | 10 | 0.22 | 21 / 46 |
| family_sphere | 10 / 10  | 4  | 0.29 | 5 / 46  |
| household     | 6 / 20   | 0  | -0.02 | 9 / 37 |

- Occupation words used: attorney coach driver editor executive judge manager
  police secretary writer teacher reporter artist author doctor lawyer
  professor prosecutor administrator sheriff athlete soldier farmer
  firefighter pastor engineer nurse instructor clerk detective photographer
  operator supervisor journalist musician physician contractor chef scientist
  trooper consultant counselor carpenter architect designer dancer technician
  weaver inspector mechanic therapist entrepreneur surgeon veterinarian
  painter paramedic.
- Just below the bar (coverage): biologist 0.50, miner 0.50, librarian 0.49,
  dispatcher 0.49, sailor 0.43, dentist 0.42, broker 0.42, caregiver 0.38,
  realtor 0.38.
- Household words used: groceries, cooking, cleaning, dishes, gardening,
  laundry (care words — parenting 0.37, daycare 0.32, childcare 0.26 — miss).
- National trend (balanced panel of 46 states, oriented RND): occupation
  rises steadily, -0.0122 -> -0.0091 -> -0.0067 -> -0.0043 (occupations
  remain male-leaning but less so); family_sphere stays female-leaning
  (about -0.009 to -0.012), no clear trend; household is noisy.
- TV arm: occupation 21 words, family_sphere 4, household 0 — confirms TV is
  too thin for these lists.

**Follow-ups flagged.** *weaver* (coverage 0.69 vs ~0.05 for comparable
crafts) is almost certainly the surname Weaver; *carpenter* and *painter*
may carry surname / artist senses too. Check their per-word RND and likely
exclude weaver.

---

## Open questions

- Validate occupation scores against census female shares (Garg's file covers
  the words with a `garg_word`; ACS state x occupation shares would allow a
  state-level validation).
- Do *secretary* / *supervisor* behave as outliers? If so, drop them.
- TV arm: keep as robustness only (≤ 8 states present in every period).
