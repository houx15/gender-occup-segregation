# Methodology: Provincial Gender Norm Index from Social Media and Newspaper Text

## 1. Overview

This document describes the construction and validation of province-level gender norm indices derived from two Chinese text data sources: Weibo (social media) and provincial-level official newspapers. Unlike the longitudinal analysis based on Google Ngram and People's Daily (see *methodology_longitudinal.md*), the present analysis is fundamentally **cross-sectional in nature**: we train one Word2Vec embedding model per province (Weibo) or per province-year unit (newspapers), yielding spatial rather than temporal variation as the primary dimension of analysis.

We apply the same WEAT (Word Embedding Association Test) framework to measure gender stereotyping along three dimensions—Work–Family, Leadership, and STEM—and attempt validation against province-level survey aggregates from the CFPS and CGSS programs. As we discuss in Section 8, the limited temporal overlap between text data availability and survey coverage poses a fundamental constraint on the validation exercise.

---

## 2. Data Sources

### 2.1 Weibo (Social Media)

We use geotagged Weibo posts collected from public timelines. Posts are assigned to provinces based on user-reported location fields or geolocation metadata. The data covers 31 provincial-level administrative units, with substantial variation in volume across provinces (larger provinces and more urbanized regions contributing more posts). Because the Weibo data was collected in a single cross-sectional snapshot, there is no temporal stratification; all posts from a given province are pooled into a single corpus.

**Key characteristics:**
- Informal register (colloquial Chinese, internet slang)
- Cross-sectional: one time period, no longitudinal variation
- Province-level granularity
- Word lists: `wordlists/weat_informal/` (adapted for social media register)

### 2.2 Provincial Official Newspapers

We use the full-text archives of provincial-level official newspapers (省日报) from 30 provinces (Jiangsu excluded—no data available). Source files are organized as `{省日报}/YYYY/MM/YYYY-MM-DD.txt`, where each file contains a single line of tab-separated articles. The raw dataset totals approximately 12 GB.

**Key characteristics:**
- Formal register (official media language)
- Longitudinal: coverage spans 2007–2024, though with **highly uneven provincial coverage over time** (see Section 7)
- Province-year granularity: one embedding model per province × year combination
- Word lists: `wordlists/weat_formal/` (standard formal register)

| Property | Weibo | Provincial Newspapers |
|----------|-------|----------------------|
| Register | Informal | Formal (official media) |
| Temporal design | Cross-sectional | Longitudinal (2007–2024) |
| Analysis unit | Province | Province × Year |
| Province coverage | 31 | 30 (no 江苏) |
| Word list variant | `weat_informal` | `weat_formal` |

---

## 3. Corpus Construction

### 3.1 Weibo

Posts are segmented using jieba with HMM enabled, filtered for minimum document length (≥5 tokens), and pooled into a single corpus per province. A minimum document threshold of 1,000 per province is enforced; provinces falling below this threshold are excluded from analysis.

### 3.2 Provincial Newspapers

For each province-year unit, all qualifying text files are processed:

1. **Text extraction**: Read `.txt` files from the province's directory tree, split on tabs to recover individual articles.
2. **Cleaning**: Remove URLs, HTML artifacts, non-Chinese characters (retaining CJK range `\u4e00`–`\u9fff`), and excessive whitespace.
3. **Segmentation**: Word segmentation via jieba with HMM enabled; removal of 35+ high-frequency stopwords.
4. **Filtering**: Exclude documents with fewer than 5 tokens after segmentation.
5. **Aggregation**: All qualifying documents from a given province and year are pooled into a single corpus.

A minimum document threshold of 500 per province-year unit is enforced. After filtering, the pipeline produces 177 province-year corpora spanning 30 provinces and 18 years (2007–2024), totaling approximately 9.4 GB of segmented text.

---

## 4. Word Embedding Training

We train skip-gram Word2Vec models independently on each corpus using the gensim implementation. Hyperparameters are held constant across all units and data sources:

| Parameter | Value |
|-----------|-------|
| Architecture | Skip-gram (`sg=1`) |
| Vector dimensionality | 300 |
| Context window | 5 |
| Minimum word frequency | 20 |
| Negative samples | 10 |
| Training epochs | 10 |
| Random seed | 42 |

The minimum frequency threshold of 20 is lower than the threshold of 50 used in the longitudinal analysis, reflecting the smaller per-unit corpus sizes in the provincial setting.

For the newspaper pipeline, 177 models are trained. To manage computational cost, province-year units are grouped into 8 balanced batches by raw data volume (~1.38 GB each) and trained in parallel via SLURM array jobs.

---

## 5. Gender Norm Index Construction

The gender axis, concept word lists, projection, and effect size computation follow the identical procedure described in *methodology_longitudinal.md*, Sections 5.1–5.4. The three WEAT dimensions are:

- **Work–Family**: Family/domestic words vs. career/work words
- **Leadership**: Non-leadership positions vs. leadership positions
- **STEM**: Non-STEM fields vs. STEM fields

For each province (Weibo) or province-year unit (newspapers), we compute Cohen's $d$ as the effect size. Positive values indicate that the first concept group (e.g., family words) is more female-associated than the second group (e.g., work words).

As a complementary measure on the same models, we also compute Garg's relative norm distance (RND) per category (leadership, family, science) for each province / province-year, oriented onto a gender-ideation axis (family reversed) and reported via cross-province rankings, heatmaps, and choropleths. This RND view is reported alongside — not in place of — the Cohen's $d$ index. Full procedure: [`replication/garg_weat_per_category.md`](replication/garg_weat_per_category.md).

---

## 6. Visualization

### 6.1 Choropleth Maps

For the newspaper analysis, we generate choropleth maps of China using a GeoDataFrame of provincial boundaries (shapefile: `chn_admbnda_adm1_ocha_2020.shp`, matched on `ADM1_ZH` column). Province-level Cohen's $d$ values are mapped to a diverging `RdBu_r` colormap centered at 0. Provinces without data for a given year are rendered in grey.

An aggregated grid figure presents all 12 combinations of 3 dimensions × 4 selected years (2018, 2020, 2022, 2024) in a single 3×4 layout with a shared colorbar. This enables visual comparison of both spatial patterns (across provinces) and temporal evolution (across years) within one figure.

Each province polygon is labeled with its short name (e.g., 北京, 广东) at the geometric representative point. Provinces with data receive dark labels; provinces without data receive light grey labels.

### 6.2 Correlation Scatter Plots

We generate scatter plots of embedding-based Cohen's $d$ (y-axis) against survey-based gender ideation scores (x-axis) for province-year units where both measures are available. Points are colored by time period (2007–2009, 2010–2014, 2015–2019, 2020–2024). Separate figures are produced for CFPS-only, CGSS-only, and combined samples. Each subplot displays an OLS regression line with Pearson's $r$, $p$-value, and the regression equation.

### 6.3 Province Longitudinal Trends

For four selected provinces (河南, 浙江, 内蒙古, 辽宁), we plot dual-axis time series showing WEAT Cohen's $d$ (left axis) and survey gender ideation (right axis) from 2007 to 2024, with separate lines for CFPS and CGSS survey points.

---

## 7. Data Coverage and Its Consequences

### 7.1 Uneven Provincial Coverage Over Time

The newspaper data does not provide uniform coverage across provinces and years. Data availability is heavily concentrated in recent years:

| Year | Provinces with data |
|------|-------------------|
| 2007 | 1 (河南) |
| 2008–2009 | 2 |
| 2010 | 3 |
| 2011–2016 | 4 (河南, 海南, 浙江, 山东) |
| 2017 | 6 |
| 2018 | 10 |
| 2019 | 11 |
| 2020 | 13 |
| 2021 | 18 |
| 2022 | 27 |
| 2023 | 29 |
| 2024 | 30 |

This means that the early years of the newspaper analysis effectively capture only 4 provinces (河南, 海南, 浙江, 山东), while cross-provincial analysis at scale is feasible only from 2021 onward. We note that the four "rich" provinces are not randomly selected—they are provinces where the official newspaper archive is most complete, which may correlate with media ecosystem characteristics.

### 7.2 Implications for Longitudinal Analysis

The severely unbalanced panel structure means that:

1. **Temporal trends** can be meaningfully estimated for only the 4 rich provinces (2007–2024). For the remaining 26 provinces, the time series is truncated to 1–8 years.
2. **Cross-sectional comparisons** across many provinces are only available for 2022–2024, eliminating the possibility of observing change over time for most of the country.
3. **Panel regression** with province and year fixed effects—which would be the ideal identification strategy—is infeasible for most provinces due to insufficient temporal variation.

---

## 8. Survey Validation: A Fundamental Data Constraint

### 8.1 The Overlap Problem

Validating the text-based indices against survey data requires province-year units where both a WEAT model and a survey estimate exist. The three survey programs provide province-level gender ideation scores for the following years:

| Survey | Province-years available | Years |
|--------|------------------------|-------|
| ACWF | 72 | 1990, 2000, 2010 |
| CFPS | 59 | 2014, 2020 |
| CGSS | 85 | 2010, 2012, 2013, 2015, 2017, 2018, 2021, 2023 |

The CGSS appears to offer rich temporal coverage (8 waves), making it the most promising validation source. However, **a critical mismatch arises**: the CGSS province sampling does not cover all provinces in every wave. In particular:

- **CGSS 2010–2017 waves** surveyed only 6–9 provinces, predominantly large urbanized provinces (北京, 上海, 天津, 河北, 辽宁, 内蒙古, 吉林, 山西, 黑龙江). These are *exactly* the provinces that lack newspaper data in those early years.
- **WEAT 2010–2016** covers only 河南, 海南, 浙江, 山东—none of which appear in the CGSS sample for those years.

The result is **zero overlap** for the first four CGSS waves:

| CGSS Year | CGSS Provinces | WEAT Provinces | **Overlap** |
|-----------|---------------|----------------|-------------|
| 2010 | 9 (北京, 上海, 天津, ...) | 3 (河南, 海南, 浙江) | **0** |
| 2012 | 9 | 4 (+山东) | **0** |
| 2013 | 8 | 4 | **0** |
| 2015 | 8 | 4 | **0** |
| 2017 | 8 | 6 (+河北, 内蒙古) | **2** |
| 2018 | 28 (nationwide) | 10 | **8** |
| 2021 | 6 | 18 | **6** |
| 2023 | 9 | 29 | **9** |

For CFPS, the overlap is larger because both waves (2014, 2020) surveyed nearly all provinces. However, CFPS provides only two time points, yielding 17 province-year matches per WEAT dimension.

### 8.2 Consequences for Validation

The total merged dataset (WEAT × CGSS + CFPS, strict year match) contains only **135 province-year observations** across 17 provinces and 3 WEAT dimensions:

- CFPS: 51 observations (17 provinces × 1 year × 3 dimensions × ~17 provinces)
- CGSS: 75 observations (from 2017, 2018, 2021, 2023 waves only)
- ACWF: 9 observations (2010 wave, 3 provinces)

The resulting correlation estimates are weak and statistically non-significant (all $p > 0.05$), though the sample is too small to draw firm conclusions.

### 8.3 Why Year Tolerance Cannot Resolve the Mismatch

A natural remedy would be to relax the strict year-matching requirement—e.g., matching a WEAT model from 2015 to a CGSS survey from 2017 (±2 years). However, this is methodologically inappropriate because:

1. **Gender norms change over time.** The very purpose of longitudinal analysis is to capture temporal variation. Smearing observations across years would attenuate the correlation we are trying to measure.
2. **The mismatch is structural, not incidental.** The provinces without newspaper data (北京, 上海, 天津, etc.) are systematically different from those with data (河南, 海南, etc.) in urbanization, economic development, and media environment. Year tolerance would not fix this—the same provinces lack data across multiple adjacent years.
3. **The WEAT estimates for 2010–2016 come from only 4 provinces.** Even if we matched with tolerance, we would be correlating survey scores from predominantly urban provinces with WEAT scores from predominantly rural/mid-income provinces—a compositional mismatch that would bias the correlation in unknowable directions.

### 8.4 What This Means for the Analysis

Given these constraints, we acknowledge that:

1. **The survey validation for the provincial newspaper analysis is severely underpowered.** We cannot draw strong conclusions about whether the text-based indices track survey-measured gender attitudes.
2. **The cross-sectional Weibo analysis faces a related but different problem.** Because Weibo provides only a single cross-section, validation reduces to a single correlation between 31 provincial WEAT scores and 31 survey scores—a sample size that offers minimal statistical power.
3. **The four selected longitudinal provinces (河南, 浙江, 内蒙古, 辽宁)** are chosen precisely because they have the richest WEAT coverage, but only 内蒙古 and 辽宁 also appear frequently in the CGSS sample. 河南 and 浙江 each have only 1 CGSS match (2018) and 2 CFPS matches (2014, 2020).

---

## 9. Recommendations for Future Work

Given the data constraints documented above, we suggest the following directions for discussion:

1. **Expand newspaper coverage.** If additional provincial newspaper archives can be obtained for the early period (2010–2017), particularly for the CGSS-sampled provinces (北京, 上海, 天津, etc.), the overlap with survey data would increase dramatically.
2. **Use the 2022–2024 cross-section for validation.** These years provide WEAT estimates for 27–30 provinces, which can be correlated with the most recent CGSS/CFPS waves in a pure cross-sectional design with reasonable power.
3. **Supplement with alternative validation targets.** Rather than relying solely on survey self-reports, consider correlating WEAT indices with behavioral or structural gender-equality indicators (e.g., female labor force participation rates, gender wage gap estimates, sex ratios at birth) that are available annually at the provincial level from statistical yearbooks.
4. **Frequency artifact controls.** If survey validation proceeds, all regression specifications must control for the relative frequency of gendered target words in each province-year corpus, following van Loon et al. (2022), who demonstrated that WEAT–survey correlations can be entirely spurious when frequency confounds are not addressed.
5. **Multi-level modeling framework.** When sufficient overlap is achieved, use hierarchical models with province random intercepts, year random intercepts, and frequency controls to properly partition the variance.

---

## References

- Bolukbasi, T., Chang, K.-W., Zou, J., Saligrama, V., & Kalai, A. (2016). Man is to computer programmer as woman is to homemaker? Debiasing word embeddings. *Advances in Neural Information Processing Systems*, 29.
- Caliskan, A., Bryson, J. J., & Narayanan, A. (2017). Semantics derived automatically from language corpora contain human-like biases. *Science*, 356(6334), 183–186.
- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Lawrence Erlbaum Associates.
- Garg, N., Schiebinger, L., Jurafsky, D., & Zou, J. (2018). Word embeddings quantify 100 years of gender and ethnic stereotypes. *Proceedings of the National Academy of Sciences*, 115(16), E3635–E3644.
- Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., & Dean, J. (2013). Distributed representations of words and phrases and their compositionality. *Advances in Neural Information Processing Systems*, 26.
- van Loon, A., Kessler, J. B., van der Wielen, N., & van Beuningen, D. (2022). Negative associations in word embeddings predict anti-Black bias across regions—but only via name frequency. *Proceedings of the National Academy of Sciences*, 120(18), e2217972120.
