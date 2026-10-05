### Corpus per state-window (3DLNews2, Google newspaper articles)

| window | states | states_trained | articles_median | articles_total | tokens_median_millions | tokens_min_millions | tokens_max_millions |
|---|---|---|---|---|---|---|---|
| 2005–14 | 51 | 49 | 4259.0 | 342228 | 1.79 | 0.415 | 15.88 |
| 2010–19 | 51 | 51 | 7841.0 | 588326 | 3.104 | 0.408 | 24.408 |
| 2015–24 | 51 | 51 | 11559.0 | 780648 | 4.104 | 0.412 | 27.456 |

### Word lists: candidates and words used (in vocab in >= 50% of state-window models)

| category | n_candidates | n_used | words |
|---|---|---|---|
| family_sphere | 10 | 10 | home, parents, children, family, kitchen, cousins, marriage, relatives, wedding, household |
| household | 20 | 9 | cleaning, cooking, groceries, dishes, gardening, laundry, parenting, daycare, chores |
| occupation | 133 | 75 | artist, attorney, author, coach, doctor, driver, editor, executive, judge, lawyer, manager, police, professor, prosecutor, reporter, secretary, teacher, writer, administrator, sheriff, clerk, engineer, nurse, soldier, athlete, contractor, detective, farmer, firefighter, journalist, pastor, photographer, instructor, musician, supervisor, operator, chef, consultant, physician, scientist, counselor, technician, trooper, dancer, designer, architect, carpenter, entrepreneur, therapist, inspector, surgeon, mechanic, painter, veterinarian, paramedic, biologist, dispatcher, librarian, broker, sailor, dentist, accountant, miner, psychologist, caregiver, realtor, banker, attendant, auditor, gardener, bartender, pharmacist, clergy, custodian, teller |

### National, by occupation: occupation RND vs ACS female share

| window | n_occupations | pearson_r | spearman_r | pearson_r_logit |
|---|---|---|---|---|
| 2005–14 | 72 | 0.627 | 0.608 | 0.609 |
| 2010–19 | 72 | 0.67 | 0.624 | 0.646 |
| 2015–24 | 72 | 0.695 | 0.64 | 0.666 |

### National trend: mean over a balanced panel of states (RND for ours; ACS shares / gaps)

| window | n_states | ours_occupation | ours_family_sphere | ours_household | matched_female_share | duncan | female_emp_share | motherhood_emp_gap | motherhood_hours_gap | married_women_nilf | wife_earnings_share | wife_earns_more | gender_emp_gap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2005–14 | 49 | -0.009 | 0.01 | 0.003 | 0.413 | 0.522 | 0.473 | 0.124 | 3.552 | 0.247 | 0.362 | 0.276 | 0.098 |
| 2010–19 | 49 | -0.005 | 0.011 | 0.008 | 0.422 | 0.509 | 0.476 | 0.111 | 3.04 | 0.241 | 0.365 | 0.285 | 0.088 |
| 2015–24 | 49 | -0.004 | 0.011 | 0.009 | 0.434 | 0.493 | 0.475 | 0.108 | 2.806 | 0.229 | 0.372 | 0.293 | 0.081 |

### State level: agreement r between our scores and ACS measures (> 0 = both point to more, or both to less, traditional)

| ACS measure | ours_family_sphere (within window) | ours_household (within window) | ours_occupation (within window) | ours_family_sphere (change) | ours_household (change) | ours_occupation (change) |
|---|---|---|---|---|---|---|
| matched_female_share | -0.12 | 0.047 | 0.027 | -0.159 | 0.258 | -0.116 |
| duncan | 0.133 | -0.067 | -0.242 | -0.126 | 0.129 | 0.017 |
| female_emp_share | 0.059 | 0.059 | -0.17 | 0.139 | 0.153 | 0.101 |
| motherhood_emp_gap | 0.012 | 0.107 | 0.001 | 0.019 | 0.013 | 0.361 |
| motherhood_hours_gap | 0.131 | 0.106 | -0.215 | -0.1 | -0.189 | 0.275 |
| married_women_nilf | -0.056 | 0.058 | 0.015 | 0.069 | -0.149 | 0.232 |
| wife_earnings_share | 0.015 | 0.102 | -0.092 | 0.065 | 0.076 | 0.267 |
| wife_earns_more | 0.057 | 0.082 | -0.177 | 0.098 | 0.117 | 0.23 |
| gender_emp_gap | 0.033 | 0.084 | -0.066 | 0.188 | 0.155 | 0.148 |

### Reliability: correlation of state values, window 2005–14 vs 2015–24 (no shared years)

| measure | r_2005_vs_2015 |
|---|---|
| ours_occupation | 0.16 |
| ours_family_sphere | 0.209 |
| ours_household | -0.172 |
| matched_female_share | 0.703 |
| duncan | 0.984 |
| female_emp_share | 0.933 |
| motherhood_emp_gap | 0.912 |
| motherhood_hours_gap | 0.862 |
| married_women_nilf | 0.962 |
| wife_earnings_share | 0.939 |
| wife_earns_more | 0.92 |
| gender_emp_gap | 0.933 |
