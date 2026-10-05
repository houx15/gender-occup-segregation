# Part II-C — text-survey discrepancy

gap = z(text) − z(survey), main measures, higher = text more traditional than survey. All predictors jointly, standardized; predictors observed in < 80% of state-windows left out: abortion_restrictions, citizen_ideology, equal_pay_law, evangelical_lds_share, so_employment_law, universal_prek.

### Model fit

| domain | spec | n | r2 |
|---|---|---|---|
| family | between states | 50 | 0.684 |
| family | pooled + window FE | 227 | 0.502 |
| occupation | between states | 50 | 0.605 |
| occupation | pooled + window FE | 227 | 0.399 |

### Terms with p < 0.05

| domain | spec | term | coef | se | p |
|---|---|---|---|---|---|
| occupation | pooled + window FE | ba_share | 0.395 | 0.184 | 0.032 |
| occupation | pooled + window FE | metro_share | 0.266 | 0.097 | 0.006 |
| occupation | pooled + window FE | manufacturing_share | -0.571 | 0.194 | 0.003 |
| occupation | pooled + window FE | service_share | -0.612 | 0.245 | 0.012 |
| occupation | pooled + window FE | duncan | -0.489 | 0.238 | 0.04 |
| occupation | pooled + window FE | female_share_professionals | 0.622 | 0.161 | 0.0 |
| occupation | between states | metro_share | 0.421 | 0.188 | 0.025 |
| occupation | between states | manufacturing_share | -0.873 | 0.333 | 0.009 |
| occupation | between states | service_share | -0.882 | 0.417 | 0.035 |
| occupation | between states | female_share_professionals | 0.941 | 0.285 | 0.001 |
| family | pooled + window FE | log_real_gdp_pc | -0.263 | 0.113 | 0.02 |
| family | pooled + window FE | female_share_professionals | 0.533 | 0.164 | 0.001 |
| family | between states | female_share_professionals | 0.781 | 0.253 | 0.002 |
