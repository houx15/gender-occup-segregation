# Part II-B — explaining state differences

**2.6 Policy timing (paid family leave).** 10 treated states; 2 have both a window entirely before and one entirely after benefits began: too few treated states with a clean before and after window for a DID/event study; paid family leave enters Part II-B as a state-window exposure (share of window years with benefits), interpreted associationally. Other policy domains in the plan (childcare, equal pay, discrimination protections, reproductive policy) are not yet collected.

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

Outcome: text score (higher = more traditional); all variables standardized. Between: state means, OLS (HC1). Within: state + window FE, SE clustered by state; fit column = added R2 over the FE-only model. Associational only.

### Block fit

| domain | spec | model | n | r2_or_added_r2 | significant_terms | terms |
|---|---|---|---|---|---|---|
| family | between states | all blocks | 50 | 0.156 | 0 | 13 |
| family | between states | gendered labour market | 51 | 0.079 | 0 | 5 |
| family | between states | policy | 51 | 0.003 | 0 | 1 |
| family | between states | political | 51 | 0.054 | 1 | 1 |
| family | between states | socioeconomic | 50 | 0.045 | 0 | 6 |
| family | within states (state + window FE) | all blocks | 227 | 0.084 | 2 | 13 |
| family | within states (state + window FE) | gendered labour market | 232 | 0.019 | 0 | 5 |
| family | within states (state + window FE) | policy | 232 | 0.0 | 0 | 1 |
| family | within states (state + window FE) | political | 232 | 0.007 | 0 | 1 |
| family | within states (state + window FE) | socioeconomic | 227 | 0.026 | 1 | 6 |
| occupation | between states | all blocks | 50 | 0.429 | 2 | 13 |
| occupation | between states | gendered labour market | 51 | 0.265 | 2 | 5 |
| occupation | between states | policy | 51 | 0.004 | 0 | 1 |
| occupation | between states | political | 51 | 0.081 | 1 | 1 |
| occupation | between states | socioeconomic | 50 | 0.268 | 1 | 6 |
| occupation | within states (state + window FE) | all blocks | 227 | 0.06 | 0 | 13 |
| occupation | within states (state + window FE) | gendered labour market | 232 | 0.016 | 0 | 5 |
| occupation | within states (state + window FE) | policy | 232 | 0.004 | 0 | 1 |
| occupation | within states (state + window FE) | political | 232 | 0.0 | 0 | 1 |
| occupation | within states (state + window FE) | socioeconomic | 227 | 0.033 | 0 | 6 |

### Terms with p < 0.05 (one model per block)

| domain | spec | term | coef | se | p |
|---|---|---|---|---|---|
| occupation | between states | metro_share | 0.44 | 0.153 | 0.004 |
| occupation | between states | duncan | -0.915 | 0.214 | 0.0 |
| occupation | between states | female_share_professionals | 0.446 | 0.181 | 0.014 |
| occupation | between states | gop_two_party_share | -0.285 | 0.128 | 0.027 |
| family | between states | gop_two_party_share | 0.232 | 0.104 | 0.025 |
| family | within states (state + window FE) | unemployment_rate | -0.522 | 0.208 | 0.012 |
