# Part II-C — text-survey discrepancy

gap = z(text) − z(survey composite: mean of the z-scored direct benchmarks), higher = text less traditional than survey. All predictors jointly, standardized; predictors observed in < 80% of state-windows left out: abortion_restrictions, citizen_ideology, equal_pay_law, evangelical_lds_share, so_employment_law, universal_prek.

### Model fit

| domain | spec | n | r2 |
|---|---|---|---|
| household | between states | 50 | 0.381 |
| household | pooled + window FE | 194 | 0.529 |
| occupation | between states | 50 | 0.568 |
| occupation | pooled + window FE | 227 | 0.441 |

### Terms with p < 0.05

| domain | spec | term | coef | se | p |
|---|---|---|---|---|---|
| occupation | pooled + window FE | manufacturing_share | 0.491 | 0.185 | 0.008 |
| occupation | pooled + window FE | female_share_professionals | -0.393 | 0.168 | 0.019 |
| occupation | pooled + window FE | log_tokens | -0.262 | 0.107 | 0.015 |
| occupation | between states | manufacturing_share | 0.959 | 0.34 | 0.005 |
| occupation | between states | female_share_professionals | -0.813 | 0.303 | 0.007 |
| occupation | between states | log_tokens | -0.309 | 0.153 | 0.044 |
| household | pooled + window FE | service_share | 0.393 | 0.196 | 0.045 |
| household | pooled + window FE | pfl_share | 0.151 | 0.049 | 0.002 |
| household | between states | women_lfp | -0.693 | 0.34 | 0.041 |
