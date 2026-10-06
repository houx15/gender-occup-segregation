# Part II-B — explaining state differences

**2.6 Policy timing (paid family leave).** 10 treated states; 2 have both a window entirely before and one entirely after benefits began: too few treated states with a clean before and after window for a DID/event study; paid family leave enters Part II-B as a state-window exposure (share of window years with benefits), interpreted associationally. The other policy domains (universal pre-K, equal pay, sexual-orientation employment protection, abortion restrictions; CSPP) enter Part II-B as window means; their coverage ends 2013-2017, so no timing design.

| state | benefits_start | 1995–04 | 2000–09 | 2005–14 | 2010–19 | 2015–24 | observed_fully_before | observed_fully_after | usable_before_after |
|---|---|---|---|---|---|---|---|---|---|
| california | 2004 | during | during | after | after | after | 0 | 3 | False |
| new_jersey | 2009 | before | during | during | after | after | 1 | 2 | True |
| rhode_island | 2014 | before | before | during | during | after | 1 | 1 | True |
| new_york | 2018 | before | before | before | during | during | 3 | 0 | False |
| washington | 2020 | before | before | before | before | during | 4 | 0 | False |
| district_of_columbia | 2020 | before | before | before | before | during | 4 | 0 | False |
| massachusetts | 2021 | before | before | before | before | during | 4 | 0 | False |
| connecticut | 2022 | before | before | before | before | during | 4 | 0 | False |
| oregon | 2023 | before | before | before | before | during | 4 | 0 | False |
| colorado | 2024 | before | before | before | before | during | 4 | 0 | False |

Outcome: text score (higher = less traditional); all variables standardized. Between: state means, OLS (HC1). Within: state + window FE, SE clustered by state; fit column = added R2 over the FE-only model. Associational only.

### Block fit

| domain | spec | model | n | r2_or_added_r2 | significant_terms | terms |
|---|---|---|---|---|---|---|
| household | between states | all blocks | 46 | 0.512 | 3 | 20 |
| household | between states | gendered labour market | 51 | 0.125 | 0 | 5 |
| household | between states | policy | 46 | 0.042 | 0 | 5 |
| household | between states | political & cultural | 50 | 0.114 | 1 | 3 |
| household | between states | socioeconomic | 50 | 0.142 | 0 | 7 |
| household | within states (state + window FE) | all blocks | 75 | 0.2 | 0 | 20 |
| household | within states (state + window FE) | gendered labour market | 228 | 0.041 | 1 | 5 |
| household | within states (state + window FE) | policy | 75 | 0.015 | 0 | 5 |
| household | within states (state + window FE) | political & cultural | 173 | 0.003 | 0 | 3 |
| household | within states (state + window FE) | socioeconomic | 223 | 0.043 | 0 | 7 |
| occupation | between states | all blocks | 47 | 0.49 | 0 | 20 |
| occupation | between states | gendered labour market | 51 | 0.265 | 2 | 5 |
| occupation | between states | policy | 47 | 0.102 | 1 | 5 |
| occupation | between states | political & cultural | 50 | 0.018 | 0 | 3 |
| occupation | between states | socioeconomic | 50 | 0.282 | 1 | 7 |
| occupation | within states (state + window FE) | all blocks | 79 | 0.095 | 0 | 20 |
| occupation | within states (state + window FE) | gendered labour market | 232 | 0.016 | 0 | 5 |
| occupation | within states (state + window FE) | policy | 79 | 0.032 | 1 | 5 |
| occupation | within states (state + window FE) | political & cultural | 177 | 0.024 | 0 | 3 |
| occupation | within states (state + window FE) | socioeconomic | 227 | 0.039 | 0 | 7 |

### Terms with p < 0.05 (one model per block)

| domain | spec | term | coef | se | p |
|---|---|---|---|---|---|
| occupation | between states | metro_share | -0.448 | 0.153 | 0.003 |
| occupation | between states | duncan | 0.915 | 0.214 | 0.0 |
| occupation | between states | female_share_professionals | -0.446 | 0.181 | 0.014 |
| occupation | between states | abortion_restrictions | -0.332 | 0.163 | 0.042 |
| occupation | within states (state + window FE) | pfl_share | 0.149 | 0.076 | 0.049 |
| household | between states | citizen_ideology | 0.567 | 0.269 | 0.035 |
| household | within states (state + window FE) | female_share_managers | 1.152 | 0.567 | 0.042 |
