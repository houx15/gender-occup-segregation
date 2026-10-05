# State-Level Gender Norm Analysis Plan

## 1. Project Goal

We have a state-level longitudinal dataset with approximately 3–5 time points per U.S. state.

For each state-year, we have:

- Text-based gender bias measures based on RND-style embedding methods.
- Two semantic domains:
  - Occupation
  - Family
- Survey-based gender ideology measures.
  - Main analysis: objective survey measures.
  - Robustness analysis: subjective survey measures.
- ACS and other state-level contextual variables.
- Text volume varies substantially across states and years.

The analysis should be organized around three major goals:

1. Validate the text-based gender norm measure.
2. Use the measure to study state-level geographic and temporal variation in gender norms.
3. Use selected occupations, family concepts, and states as interpretable case studies.

The main analysis should use the objective survey measure. The full set of survey-validation analyses should later be replicated using the subjective measure as a robustness check.

---

# Part I. Measurement Validation

The first section should establish that the text-based measure captures meaningful variation in gender norms.

Occupation and family should follow parallel analytical structures whenever possible.

## 1.1 Occupation-Level External Validity

### Research Question

Do occupations that are more female-dominated in reality also appear more female-associated in the text-based embedding measure?

### Unit of Analysis

Occupation.

Pool observations across:

- states
- years

For each occupation, calculate:

- average text-based gender bias
- actual female share in that occupation

### Visualization

Scatter plot:

- x-axis: actual female share in occupation
- y-axis: text-based gender bias
- each point: one occupation

Add:

- fitted regression line
- confidence interval
- correlation coefficient
- number of occupations

The sign should be oriented so that interpretation is intuitive and consistent across all figures.

This figure serves as a simple external/convergent validity check.

For the family domain, only construct an analogous figure if there is a substantively meaningful external benchmark. Do not force artificial symmetry.

---

## 1.2 Aggregate Temporal Variation

### Research Question

Does the text-based gender norm measure exhibit meaningful temporal variation?

### Occupation

Pool across:

- occupations
- states

For each available year, estimate the average occupation gender bias.

### Family

Pool across:

- family terms
- states

For each year, estimate average family gender bias.

### Visualization

Use a simple temporal plot showing:

- mean bias by year
- uncertainty interval if appropriate

Because there are only approximately 3–5 time points, avoid visually overstating smooth trends.

Treat the observations as discrete measurement waves rather than a dense continuous time series.

---

## 1.3 Geographic Heterogeneity Across States

### Research Question

Does the measure capture meaningful geographic variation across states?

For each state, calculate average gender bias pooled across:

- years
- occupations, for the occupation measure

or:

- years
- family terms, for the family measure

### Visualization

Use a sorted state-level dot plot or caterpillar plot.

Preferred design:

- y-axis: states
- x-axis: average gender bias
- states sorted by bias
- uncertainty intervals if available

Do not use a map for this particular validation question because the goal is precise state comparison rather than geography.

---

## 1.4 Core Survey Validation: State-Year Alignment

This should be the central validation figure.

### Research Question

Do state-year text-based gender norms align with state-year gender ideology measured using survey data?

### Unit of Analysis

State-year.

Construct separately for:

- occupation gender bias
- family gender bias

### Main Survey Variable

Use the objective survey measure.

### Visualization

Scatter plot:

- x-axis: survey-based gender ideology
- y-axis: text-based gender bias
- each point: one state-year

Produce two main panels:

- Panel A: Occupation
- Panel B: Family

Add:

- fitted regression line
- confidence interval
- correlation coefficient
- sample size

State and year labels do not need to be shown for every point, but the underlying dataset should preserve them for diagnostics and later case selection.

### Statistical Models

At minimum, estimate:

```text
Survey_st = alpha + beta * TextBias_st + error_st
```

Also estimate specifications including:

```text
Survey_st = alpha_s + lambda_t + beta * TextBias_st + error_st
```

where:

- alpha_s = state fixed effects
- lambda_t = year fixed effects

The purpose is to distinguish pooled cross-sectional alignment from within-state temporal alignment.

Report standardized coefficients when useful for comparison between occupation and family measures.

---

## 1.5 Separate Between-State and Within-State Alignment

The pooled state-year relationship combines two distinct sources of variation:

1. Persistent differences across states.
2. Changes within the same state over time.

These should be explicitly separated.

### Between-State Analysis

For each state, calculate:

```text
mean_text_bias_s
mean_survey_ideology_s
```

Then visualize:

- x-axis: mean survey ideology
- y-axis: mean text bias
- each point: one state

This answers:

> Are states that are more gender-traditional in survey data also more gender-stereotypical in textual representations?

### Within-State Analysis

Demean both variables within state:

```text
text_within_st = text_bias_st - mean(text_bias_s)
survey_within_st = survey_st - mean(survey_s)
```

Then visualize:

- x-axis: within-state survey deviation
- y-axis: within-state text-bias deviation
- each point: one state-year

This answers:

> When a state's gender ideology changes over time, does the text-based gender norm measure move in the same direction?

Run these analyses separately for occupation and family.

---

## 1.6 Measurement Reliability and Text Volume

Text volume differs across state-year observations and may affect measurement precision.

Instead of relying primarily on state-specific raw correlations, evaluate whether low-volume observations produce larger measurement discrepancies.

### Step 1: Construct Measurement Error

Using the survey-validation model, calculate prediction residuals:

```text
residual_st = survey_st - predicted_survey_st
```

Then define:

```text
absolute_error_st = abs(residual_st)
```

Potentially also calculate standardized residuals.

### Step 2: Relate Error to Text Volume

Estimate:

```text
absolute_error_st ~ log(text_volume_st)
```

### Visualization

Scatter plot:

- x-axis: log text volume
- y-axis: absolute survey-text discrepancy

Add:

- fitted line
- confidence interval

Also consider a binned version:

- divide observations into text-volume quintiles
- calculate mean absolute error within each quintile
- plot error by volume quintile

Expected interpretation:

```text
Higher text volume -> lower measurement error
```

This provides evidence about measurement reliability.

---

## 1.7 State-Specific Measurement Alignment

We are interested in whether the text measure performs differently across states.

However, each state only has approximately 3–5 time points.

Therefore:

**Do not use raw state-specific Pearson correlations as the primary estimator.**

These correlations will be extremely unstable with such small within-state sample sizes.

Instead, use a hierarchical / partial-pooling approach.

### Suggested Model

Conceptually:

```text
Survey_st = alpha_s + beta_s * TextBias_st + error_st
```

with:

```text
beta_s ~ Normal(beta_global, sigma_beta)
```

The exact specification can be adjusted depending on the available modeling framework.

### Output

Estimate state-specific alignment slopes using partial pooling.

### Visualization

Caterpillar plot:

- y-axis: state
- x-axis: estimated state-specific alignment coefficient
- uncertainty interval for each state
- vertical reference line at zero
- optionally also show the global average slope

This identifies states where text-survey alignment appears unusually strong or weak while accounting for the very small number of observations per state.

Do this separately for:

- occupation
- family

Treat this as exploratory heterogeneity analysis rather than definitive state ranking.

---

## 1.8 Subjective Survey Measure as Robustness Check

After completing all major survey-validation analyses using the objective survey measure, replicate them using the subjective survey measure.

At minimum replicate:

1. state-year survey/text scatter
2. between-state alignment
3. within-state alignment
4. text-volume versus measurement-error relationship
5. hierarchical state-specific alignment model

These should primarily appear in supplementary materials unless the subjective/objective comparison produces a substantively important result.

---

# Part II. State-Level Dynamics of Gender Norms

After validating the measure, use it to study geographic and temporal variation in gender norms.

The main substantive questions are:

1. How do gender norms differ across states?
2. How do states change over time?
3. Which states change the most?
4. Are states converging or diverging?
5. What state-level factors predict these differences and changes?

Run analyses separately for occupation and family where appropriate.

---

## 2.1 Geographic Maps

Create one U.S. state map per measurement year.

### Color Scale

Use a diverging scale centered at zero.

Interpretation should be consistent across all maps:

- one side = more stereotypical
- the other side = more counter-stereotypical
- zero = neutral/reference point

Use the exact same color limits across years so maps are directly comparable.

Avoid automatically rescaling each year independently.

Although red/green is possible, prefer a more neutral and colorblind-accessible diverging palette unless there is a strong substantive reason to use red and green.

### Goal

Maps should answer:

> Where are gender norms more stereotypical or counter-stereotypical at each time point?

---

## 2.2 State-by-Year Heatmap

Maps are useful for geography but poor for tracking temporal change.

Therefore also construct a state × year heatmap.

### Structure

Rows:

- states

Columns:

- measurement years

Cells:

- gender norm score

Use the same diverging scale centered at zero.

### State Ordering

Do not alphabetically order states by default.

Test several meaningful ordering strategies:

1. average gender norm over all years
2. baseline gender norm
3. final-year gender norm
4. region first, then average ideology within region

The main version should use whichever ordering makes temporal and cross-state structure easiest to interpret.

### Goal

The heatmap should make it easy to see:

- persistent state differences
- common national shifts
- state-specific changes
- convergence
- divergence
- possible clusters of states

---

## 2.3 State Change Ranking

Construct a direct measure of long-run state change:

```text
delta_bias_s = bias_s,last - bias_s,first
```

### Visualization

Sorted horizontal dot plot or interval-style plot.

- y-axis: state
- x-axis: change in gender norm
- vertical reference line at zero

Interpretation:

- negative direction = movement toward less stereotypical norms
- positive direction = movement toward more stereotypical norms

Adjust sign conventions depending on how the RND score is coded.

### Goal

Answer directly:

> Which states changed the most?

This is easier to interpret than asking readers to compare multiple maps manually.

---

# Part II-B. Explaining State-Level Change

The analysis should go beyond describing state differences.

The next question is:

> Why do gender norms change differently across states?

Candidate explanations should be theoretically organized rather than entered into one large atheoretical regression.

---

## 2.4 Economic and Structural Conditions

Potential predictors include:

- income per capita
- state GDP per capita
- unemployment
- educational attainment
- urbanization
- industrial structure
- manufacturing share
- service-sector share

However, these broad modernization indicators are secondary to variables with a more direct theoretical connection to gender norms.

---

## 2.5 Gendered Economic Structure

This is likely to be especially important for the occupation measure.

Potential predictors:

- female labor force participation
- female employment rate
- occupational gender segregation
- female share in professional occupations
- female share in managerial occupations
- gender wage gap
- state occupational structure

A theoretically meaningful model could test whether changes in actual gendered labor-market structure predict changes in textual occupational gender norms.

For example:

```text
OccupationGenderNorm_st
    ~ FemaleLaborForceParticipation_st
    + state_FE
    + year_FE
```

or analogous specifications.

The purpose is to test whether cultural representations track actual structural gender integration.

---

## 2.6 Policy Environment

Investigate whether major state-level policy changes are associated with changes in gender norms.

Potential policy domains include:

- paid family leave
- childcare policy
- equal-pay legislation
- workplace discrimination protections
- reproductive policy
- family policy
- other gender-related employment or social policies

For each policy:

1. identify adoption date
2. identify treated states
3. determine whether policy timing overlaps meaningfully with our available measurement waves
4. assess whether enough treated and comparison states exist

Because we only have approximately 3–5 measurement waves, do not automatically implement a conventional event-study design.

Use DID/event-study approaches only if the timing structure supports credible identification.

Otherwise, treat policy exposure as a state-year explanatory variable and interpret results associationally.

---

## 2.7 Political and Cultural Context

Potential predictors include broader state-level political or cultural environment.

Examples may include:

- state political ideology
- electoral environment
- measures of social conservatism/liberalism
- other relevant cultural indicators

The substantive question is not simply whether political ideology correlates with gender norms.

A more interesting question is:

> Are geographic differences in gender norms better explained by economic modernization, gendered economic structure, policy environment, or broader political-cultural context?

This can become a major substantive contribution if the measures support it.

---

# Part II-C. Text-Survey Discrepancy as a Substantive Outcome

The difference between text-based gender norms and survey-based gender ideology may itself be theoretically meaningful.

Construct a state-year discrepancy measure.

For example, after standardization:

```text
gap_st = standardized_text_bias_st - standardized_survey_ideology_st
```

Or use residuals from a survey-on-text calibration model.

### Research Question

In what kinds of states does public textual discourse appear more or less gender-stereotypical than survey-reported attitudes?

Potential predictors:

- political environment
- economic structure
- demographic composition
- urbanization
- education
- text volume
- policy environment

This analysis could connect the measurement contribution to the substantive contribution.

---

# Part III. Case Studies and Interpretation

The final section should make the aggregate results interpretable.

Cases should be selected systematically from the preceding quantitative results rather than chosen subjectively in advance.

---

## 3.1 Occupation Case Studies

Identify occupations showing patterns such as:

1. largest decline in gender stereotyping
2. largest increase in gender stereotyping
3. high stability
4. reversal in gender association
5. particularly large discrepancy between textual bias and actual female share

For selected occupations, inspect how their semantic gender associations change over time.

Potential examples may include traditionally gendered occupations, but the final selection should be data-driven.

The purpose is to understand what the aggregate occupation measure actually represents.

---

## 3.2 Family Concept Case Studies

Apply a similar strategy to family-related concepts.

Identify terms or semantic relationships showing:

- large temporal change
- stability
- reversal
- unusually strong state heterogeneity

Use these cases to interpret what "family gender norm change" means semantically.

---

## 3.3 State Case Studies

Select states based on transparent empirical criteria.

Potential selection strategies:

1. states with similar baseline values but strongly divergent later trajectories
2. states with similar economic structures but different gender-norm trajectories
3. states with the largest movement toward less stereotypical norms
4. states with little change
5. states with the largest text-survey discrepancy
6. states with unusually weak or strong hierarchical text-survey alignment

For each selected state, examine:

- gender norm trajectory
- survey trajectory
- text volume
- relevant economic changes
- labor-market gender structure
- policy changes
- political/cultural context

The goal is not causal proof from individual cases, but substantive interpretation of the quantitative patterns.

---

# Recommended Figure Structure

A possible final paper structure is:

## Figure 1. External and Descriptive Validation

Possible panels:

- Occupation bias vs actual female share
- Aggregate occupation trend
- Aggregate family trend
- State-level heterogeneity

This figure should establish that the measure behaves meaningfully across occupations, time, and geography.

---

## Figure 2. Survey Validation

Main figure.

Possible panels:

- Occupation: state-year text bias vs objective survey ideology
- Family: state-year text bias vs objective survey ideology
- Occupation: between-state alignment
- Family: between-state alignment
- Occupation: within-state alignment
- Family: within-state alignment

If this becomes too dense, split it into two figures.

---

## Figure 3. Measurement Reliability

Possible panels:

- text volume vs absolute survey-text error, occupation
- text volume vs absolute survey-text error, family
- state-specific partially pooled alignment coefficients, occupation
- state-specific partially pooled alignment coefficients, family

---

## Figure 4. Geographic Evolution

Maps for each available year.

Create separate versions for occupation and family if necessary.

Use common scales across time.

---

## Figure 5. State-Level Dynamics

Possible panels:

- state × year heatmap
- sorted first-to-last state change
- potentially a distribution of state-level changes

Again, occupation and family may require separate figures if combining them reduces readability.

---

## Figure 6+. Explanatory Analyses

Figures should depend on the strongest empirical findings.

Potential examples:

- female labor force participation vs occupation gender norm
- policy adoption and gender norm change
- economic modernization and gender norm change
- text-survey discrepancy by political/cultural context

Do not commit to all of these before evaluating the data.

---

## Final Figures / Supplementary Case Studies

Occupation, family, and state-level interpretive cases.

---

# Statistical and Visualization Principles

## Avoid Overinterpreting Sparse Time Series

Each state has only approximately 3–5 time points.

Therefore:

- do not emphasize smooth state-specific time trends
- do not treat state-specific raw Pearson correlations as reliable estimates
- avoid conventional event studies unless policy timing and measurement timing support them

Use partial pooling where state-specific estimates are necessary.

---

## Keep Scales Consistent

For comparable figures:

- maintain consistent score direction
- maintain consistent axis limits where meaningful
- maintain common map and heatmap color scales
- clearly document whether higher values mean more stereotypical or less stereotypical gender norms

Create a single canonical orientation early in the pipeline and use it everywhere.

---

## Preserve Observation-Level Metadata

Every analytical state-year record should retain:

- state
- year
- text bias
- survey objective measure
- survey subjective measure
- text volume
- relevant ACS variables
- relevant policy variables
- relevant political/contextual variables

This will make later diagnostics, labeling, and case selection much easier.

---

# Suggested Coding Workflow

## Step 1. Build Canonical State-Year Datasets

Create clean state-year datasets separately for:

- occupation
- family

Each should contain all measurement, survey, volume, and contextual variables.

---

## Step 2. Build Occupation-Level and Family-Term-Level Validation Datasets

Create lower-level datasets preserving:

- occupation/family term
- state
- year
- text bias
- external benchmark where available

---

## Step 3. Generate Validation Statistics

Calculate:

- pooled correlations
- regression coefficients
- between-state correlations
- within-state coefficients
- survey prediction residuals
- absolute residuals
- text-volume relationships

---

## Step 4. Fit Hierarchical State-Specific Alignment Models

Do not use raw state correlations as the main state-specific statistic.

Store:

- posterior/estimated state-specific slope
- uncertainty interval
- global slope

---

## Step 5. Generate Geographic and Temporal Measures

Calculate:

- state-year gender norm scores
- state average
- first-year value
- last-year value
- first-to-last change
- within-state standardized change where useful

---

## Step 6. Merge Contextual Predictors

Potential sources include:

- ACS
- economic indicators
- labor-force indicators
- policy datasets
- political/contextual indicators

Document source and temporal coverage for every variable.

---

## Step 7. Run Explanatory Models

Begin with theoretically motivated models rather than automatically including all predictors.

Organize predictors into blocks:

1. socioeconomic modernization
2. gendered labor-market structure
3. policy environment
4. political/cultural environment

Compare explanatory power where useful.

---

## Step 8. Select Cases Programmatically

Create reproducible rules for identifying:

- largest changes
- strongest reversals
- largest residuals
- strongest/weakest alignment
- matched states with divergent trajectories

Save a case-selection table before conducting qualitative interpretation.

---

# Main Narrative

The analysis should ultimately support the following sequence:

1. The text-based gender norm measure corresponds to meaningful real-world gender structure.
2. It aligns with established survey measures of gender ideology.
3. Measurement quality varies partly with text availability and potentially across states.
4. The measure reveals substantial geographic variation and uneven temporal change in gender norms across U.S. states.
5. These differences may be systematically related to gendered economic structure, socioeconomic development, policy environment, and political-cultural context.
6. Specific occupations, family concepts, and states help reveal what these aggregate changes mean substantively.

The coding and visualization pipeline should prioritize this analytical narrative rather than producing every possible descriptive figure.
