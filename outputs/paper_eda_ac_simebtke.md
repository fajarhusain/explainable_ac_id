# Exploratory Data Analysis of Air Conditioner Energy Efficiency Patterns in Indonesia: Insights from the SIMEBTKE Database

---

**Authors:** [Author Name], [Co-author Name]  
**Affiliation:** [Institution]  
**Corresponding email:** [email]  

---

## Abstract

Air conditioning (AC) accounts for a significant and rapidly growing share of household electricity consumption in Indonesia. The Sistem Informasi Manajemen Efisiensi dan Konservasi Energi (SIMEBTKE), managed by the Directorate General of New, Renewable Energy and Energy Conservation (DJEBTKE) under the Ministry of Energy and Mineral Resources (ESDM), provides a publicly accessible database of energy-labelled AC products certified through the Sertifikat Hemat Energi (SHE) programme. This study presents a comprehensive exploratory data analysis (EDA) of 1,923 AC product records (623 inverter and 1,300 non-inverter models) retrieved from the SIMEBTKE database to identify patterns of energy efficiency across the Indonesian AC market. The analysis encompasses data quality assessment, missing value identification, duplicate detection, numerical format validation, outlier analysis, and multivariate correlation modelling. Key findings reveal that: (1) 55.7% of registered products carry a 4-star energy rating, while only 8.6% achieve the maximum 5-star rating; (2) all 5-star-rated products are inverter-type, whereas all 1-star products are non-inverter, indicating a clear technological divide; (3) a strong positive correlation exists between Energy Efficiency Ratio (EER) and star rating for non-inverter ACs (r = 0.828), whereas the correlation is negative for inverter ACs (r = -0.444), reflecting the different metric basis (CSPF vs. EER); (4) annual electricity cost is nearly perfectly derived from energy consumption (r = 0.997 for non-inverter), posing a data leakage risk for predictive modelling; and (5) significant data quality issues exist, including 75.82% missing values in certification dates and 541 duplicate registration numbers. These findings provide a data-driven foundation for understanding AC energy efficiency patterns in Indonesia and highlight critical preprocessing requirements for subsequent machine learning applications.

**Keywords:** air conditioner; energy efficiency; EER; CSPF; energy labeling; SIMEBTKE; exploratory data analysis; machine learning; clustering; classification; Indonesia

---

## 1. Introduction

Indonesia's energy demand has grown substantially over the past decade, driven by rapid urbanisation, rising living standards, and increasing electrification rates (Government of Indonesia, 2014; IEA, 2018). Among household appliances, air conditioning (AC) units represent one of the most significant contributors to peak electricity load, particularly in tropical climates where cooling demand is year-round (Pérez-Lombard et al., 2008; Santamouris & Kolokotsa, 2013). The International Energy Agency (IEA, 2018) projects that global cooling energy demand will triple by 2050, with Southeast Asia accounting for a disproportionate share of this growth. In Indonesia, household AC ownership has increased from less than 10% in 2010 to an estimated 20–25% in 2023 (BPS, 2023), and this trajectory is expected to accelerate.

To address the environmental and economic implications of this growth, the Government of Indonesia has established a regulatory framework for energy conservation. Government Regulation No. 70 of 2009 on Energy Conservation mandates energy efficiency standards for energy-consuming equipment, including household appliances (Government of Indonesia, 2009). This regulation was further strengthened by Government Regulation No. 79 of 2014 on the National Energy Policy, which set targets for reducing energy intensity and increasing the share of renewable energy (Government of Indonesia, 2014). A central instrument in this framework is the Sertifikat Hemat Energi (SHE), an energy efficiency certificate issued to products that meet minimum efficiency standards, accompanied by a star rating system from 1 (least efficient) to 5 (most efficient).

The Sistem Informasi Manajemen Efisiensi dan Konservasi Energi (SIMEBTKE) serves as the digital platform through which the DJEBTKE manages and disseminates data on energy-labelled products. For AC products, the database records key technical specifications including brand (merek), family (famili), model, type (inverter or non-inverter), power input (watt), cooling capacity (BTU/h), energy efficiency value (EER for non-inverter or CSPF for inverter), star rating, annual energy consumption (kWh), annual electricity cost (Rp), registration/SHE number, certification dates, and the Product Certification Body (LSPro) responsible for testing.

Despite the availability of this rich dataset, there has been limited published research that systematically analyses the SIMEBTKE database to extract actionable insights about AC energy efficiency patterns in Indonesia. Most existing studies on AC energy efficiency focus on either laboratory-based performance testing (ISO, 2017; ISO, 2013) or aggregate national-level energy consumption modelling (Ürge-Vorsatz et al., 2015; Hu et al., 2020). Few studies adopt a data-driven, exploratory approach to understand the structure, quality, and patterns embedded in national energy labelling databases, particularly in the context of developing countries with rapidly growing cooling demand.

Exploratory Data Analysis (EDA), as formalised by Tukey (1977), provides a principled framework for understanding data before applying formal statistical modelling or machine learning. In the context of energy efficiency databases, EDA serves multiple critical purposes: it reveals data quality issues that could compromise subsequent analysis (Wickham, 2014; McKinney, 2017), identifies distributional characteristics and outliers that may require special handling, uncovers correlations and potential data leakage pathways, and generates hypotheses for confirmatory analysis (Field, 2018). This is particularly important for energy labelling data, where derived variables (e.g., annual electricity cost computed from energy consumption and a fixed tariff) can introduce subtle data leakage if not properly identified before predictive modelling (Kuhn & Johnson, 2013).

This study addresses this gap by conducting a comprehensive EDA and machine learning analysis of the SIMEBTKE AC product database. The specific objectives are: (1) to assess the quality of the SIMEBTKE database, including missing values, duplicates, and format inconsistencies; (2) to characterise the distribution of key energy efficiency variables across AC types and brands; (3) to identify potential data leakage pathways that could affect subsequent predictive modelling; (4) to perform statistical hypothesis testing to confirm observed patterns; and (5) to develop and evaluate machine learning models—clustering, classification, and regression—with explicit leakage prevention. The remainder of this paper is organised as follows: Section 2 describes the data source and analytical methods; Section 3 presents the EDA results; Section 4 presents the statistical analysis and machine learning results; and Section 5 concludes with implications and recommendations.

---

## 2. Method

### 2.1 Data Source

The dataset was retrieved from the publicly accessible SIMEBTKE consumer portal at `https://simebtke.esdm.go.id/sinergi/skem-label/konsumen/pengondisi-udara-ac` (DJEBTKE, 2024). The web interface provides two product categories—inverter and non-inverter—each accessible through separate AJAX endpoints using Bootstrap Table server-side pagination. A custom Python script was developed to systematically retrieve all records via HTTP requests to the AJAX API endpoints, with pagination parameters (limit, offset, page, search). A total of 1,923 records were retrieved: 623 inverter-type and 1,300 non-inverter-type AC products. The raw data were preserved in their original format (JSON and CSV) in the `data/raw/` directory to ensure reproducibility, in accordance with established data management practices (Wilkinson et al., 2016).

### 2.2 Variables

The dataset comprises 15 variables, which can be grouped into four categories:

**Identity variables:** NO. (record number), Merek (brand), Famili (product family), Model, Tipe (type: inverter/non-inverter), No. Registrasi/No. SHE (registration/certificate number), LSPro (Product Certification Body).

**Technical specification variables:** Daya (power input in watts), Kapasitas Pendinginan (cooling capacity in BTU/h), Nilai Efisiensi (efficiency value: EER for non-inverter, CSPF for inverter).

**Energy performance variables:** Rating Bintang (star rating, 1–5), Konsumsi Energi Tahunan (annual energy consumption in kWh), Biaya Listrik Tahunan (annual electricity cost in Rp).

**Certification temporal variables:** Tanggal Terbit SHE (SHE issue date), SHE Berlaku Sampai Dengan Tanggal (SHE validity expiry date).

### 2.3 Analytical Approach

The EDA followed a systematic 13-step protocol (labelled A through M):

**(A–D) Data loading and overview:** The dataset was loaded with `dtype=str` to preserve raw format. Shape, column types, and the first 10 rows were inspected.

**(E) Missing value identification:** Missing values were identified by checking for empty strings, null, NA, N/A, nan, None, and dash characters, following the approach described by Wickham (2014).

**(F) Duplicate detection:** Three types of duplication were checked: full-row duplicates, duplicates excluding the record number column, and duplicates in the registration number field.

**(G) Categorical variable profiling:** Unique value counts and frequency distributions were computed for all categorical variables (Merek, Famili, Model, Tipe, Rating Bintang, LSPro).

**(H) Numerical format validation:** Each numeric column was inspected for non-numeric characters (commas, currency prefixes, unit suffixes) and tested for direct parsing via `pd.to_numeric()`.

**(I) Date format validation:** Date columns were validated against the ISO 8601 format (YYYY-MM-DD).

**(J) Anomaly identification:** Outliers were identified using the interquartile range (IQR) method (Tukey, 1977), and cross-validation checks were performed: (a) EER was compared against the theoretical calculation of cooling capacity divided by power input, and (b) the implied electricity tariff was computed as annual cost divided by annual consumption.

**(K) Statistical summary:** Descriptive statistics (count, mean, standard deviation, minimum, maximum, percentiles at 1%, 5%, 25%, 50%, 75%, 95%, 99%) were computed for all numeric variables, along with skewness and kurtosis.

**(L) Visualisation:** Eight visualisations were generated: histograms of all numeric variables, boxplots stratified by AC type, bar charts of top brands, rating distributions by type, scatter plots of power–capacity–efficiency relationships, a Pearson correlation heatmap, boxplots of consumption and cost by rating, and a bar chart of LSPro distribution.

**(M) Machine learning exclusion:** No machine learning was performed at this stage, consistent with the principle that EDA should precede predictive modelling (Tukey, 1977; Kuhn & Johnson, 2013).

### 2.4 Tools and Environment

All analyses were conducted in Python 3 using the following libraries: pandas (v3.0) for data manipulation (McKinney, 2017), numpy (v2.5) for numerical computation (Harris et al., 2020), matplotlib (v3.11) and seaborn (v0.13) for visualisation (Hunter, 2007; Waskom, 2021), scipy (v1.18) for statistical functions (Virtanen et al., 2020), and scikit-learn (v1.9) for machine learning (Pedregosa et al., 2011). The analysis was structured as a reproducible pipeline, with all transformations documented and intermediate outputs saved to designated directories (`data/processed/`, `outputs/figures/`, `outputs/tables/`).

### 2.5 Machine Learning Methodology

Three machine learning tasks were formulated with explicit data leakage prevention:

**Clustering (K-Means):** Unsupervised clustering was performed on three standardised features (Daya, Kapasitas Pendinginan, EER/CSPF) to identify natural product groupings. The optimal number of clusters (k) was determined using the elbow method (inertia/WCSS) and silhouette score (Rousseeuw, 1987) for k = 2–10. Cluster profiles were characterised by median values of the input features and modal star rating and AC type.

**Classification (Rating Bintang):** A multi-class classification task was formulated to predict the 5-level star rating from power input (Daya), cooling capacity (Kapasitas), AC type (Tipe), and PK category—**excluding EER/CSPF, annual energy consumption, and annual electricity cost** to prevent data leakage (Kuhn & Johnson, 2013). Four algorithms were evaluated: Random Forest (Breiman, 2001), K-Nearest Neighbours (Cover & Hart, 1967), Decision Tree, and Logistic Regression. Models were evaluated using 5-fold stratified cross-validation (F1-weighted) and a held-out test set (20%), with metrics including accuracy, F1-score, and confusion matrix analysis.

**Regression (EER/CSPF):** A regression task was formulated to predict the continuous efficiency value from the same leakage-safe predictor set (excluding annual consumption, cost, and star rating). Four algorithms were evaluated: Random Forest, Ridge Regression, Gradient Boosting (Friedman, 2001), and Linear Regression. Models were evaluated using 5×3 repeated cross-validation (R²) and a held-out test set (20%), with metrics including R², RMSE, and MAE.

All features were standardised using z-score normalisation. Categorical variables were one-hot encoded. The random seed was fixed at 42 for reproducibility.

---

## 3. Results and Discussion

### 3.1 Dataset Overview

The SIMEBTKE AC database contains 1,923 product records across 15 variables (Table 1). The dataset is dominated by non-inverter ACs (n = 1,300, 67.6%), with inverter-type ACs comprising the remaining 32.4% (n = 623). This imbalance reflects the historical prevalence of non-inverter technology in the Indonesian market, though the proportion of inverter products is expected to grow as manufacturers shift toward more efficient technologies (Daikin Industries, 2023; Samsung Electronics, 2024).

**Table 1.** Dataset overview

| Characteristic | Value |
|---|---|
| Total records | 1,923 |
| Total variables | 15 |
| Inverter records | 623 (32.4%) |
| Non-inverter records | 1,300 (67.6%) |
| Unique brands | 98 |
| Unique product families | 1,781 |
| Unique models | 1,789 |

*Note.* Data retrieved from SIMEBTKE portal, accessed September 2024.

### 3.2 Missing Values

Three variables exhibit substantial missing values (Table 2). The certification date columns (Tanggal Terbit SHE and SHE Berlaku Sampai Dengan Tanggal) each have 1,458 missing entries (75.82%), while LSPro has 1,113 missing entries (57.88%). The high proportion of missing certification dates likely reflects two factors: (1) older product registrations may have been recorded without digital date fields, and (2) some products may have been certified through older paper-based processes that predated the SIMEBTKE digital system. All technical and energy performance variables are complete (0% missing), which is encouraging for subsequent quantitative analysis.

**Table 2.** Missing value summary

| Variable | Missing count | Percentage (%) |
|---|---|---|
| Tanggal Terbit SHE | 1,458 | 75.82 |
| SHE Berlaku Sampai Dengan Tanggal | 1,458 | 75.82 |
| LSPro | 1,113 | 57.88 |
| All other variables | 0 | 0.00 |

### 3.3 Duplicate Analysis

While no full-row duplicates were found, 10 duplicate records were identified when excluding the NO. column, and—more concerningly—541 duplicate entries were found in the No. Registrasi/No. SHE field (Table 3). Examination of these duplicates reveals that they primarily stem from batch registration practices, where a single SHE certificate number is assigned to multiple product models from the same manufacturer (e.g., Mitsubishi Heavy Industries registered models SRK13YYP-W3 and SRK18YYM-W3 under the same certificate number 16/06.06.06/1A/24/LSPro/V/2024; Bestlife registered over 60 model variants under certificate 004/LSP/QI/06.1-I/2024). This finding has implications for data uniqueness: while the records are not true duplicates (they represent distinct physical products), the shared registration number means that certificate-level analyses should aggregate by registration number rather than treating each row as an independent certificate.

**Table 3.** Duplicate row summary

| Duplicate type | Count |
|---|---|
| Full-row duplicates | 0 |
| Duplicates (excluding NO.) | 10 |
| Duplicates (No. Registrasi/No. SHE) | 541 |

### 3.4 Numerical Format Assessment

All numeric variables were stored as strings in the raw data. Five of six numeric columns (Daya, Kapasitas Pendinginan, Nilai Efisiensi, Rating Bintang, Konsumsi Energi Tahunan) were stored in clean decimal format and could be parsed directly (100% success rate). The Biaya Listrik Tahunan (annual electricity cost) column, however, used comma-based thousand separators (e.g., "4,797,676.80"), resulting in only 27/1,923 (1.4%) records being parseable without preprocessing. This format inconsistency must be addressed in the data cleaning stage by removing comma separators before numeric conversion.

Date columns were stored in ISO 8601 format (YYYY-MM-DD), with 464/465 non-null values in Tanggal Terbit SHE successfully parsed. One anomalous value ("0000-00-00") was identified and should be treated as missing in subsequent processing.

### 3.5 Anomaly and Outlier Identification

Several anomalous values were identified across the numeric variables (Table 4). The IQR-based outlier method (Tukey, 1977) identified relatively low outlier rates for most variables (0.3–1.8%), suggesting that the majority of the data is within expected ranges.

**Table 4.** Outlier summary (IQR method, 1.5×IQR fence)

| Variable | IQR | Outlier range | Outliers | % |
|---|---|---|---|---|
| Daya (watt) | [593.4–1,374.8] | <−578.8 or >2,547.0 | 19 | 1.0 |
| Kapasitas Pendinginan (BTU/h) | [6,842.7–14,688.9] | <−4,926.6 or >26,458.3 | 6 | 0.3 |
| Konsumsi Energi Tahunan (kWh) | [1,165.5–3,358.7] | <−2,124.3 or >6,648.5 | 32 | 1.7 |
| Biaya Listrik Tahunan (Rp) | [1,683,856–4,823,164] | <−3,025,105 or >9,532,126 | 34 | 1.8 |

Extreme outliers were identified that warrant investigation:

- **Daya (power input):** One record (BEKO BIVOE 120) has a power input of 1.16 W, which is physically implausible for a room AC unit and likely represents a data entry error. At the upper end, 10 records exceed 5,000 W, with the maximum being 20,400 W (Midea MSBE-24CRN1). These high values are attributable to multi-model registrations where the power input represents the combined consumption of multiple indoor/outdoor unit pairs.

- **Konsumsi Energi Tahunan:** The maximum value of 5,040,796 kWh/year is approximately 2,200 times the median (2,271 kWh), corresponding to the same multi-model registration issue. When restricted to single-model registrations, the distribution is more compact.

- **Biaya Listrik Tahunan:** Twenty-seven records have a value of exactly Rp 0.00, and the maximum value is Rp 99,999,999.99 (a suspiciously round number, possibly a placeholder or error).

Cross-validation checks provided important insights:

1. **EER cross-check:** For non-inverter ACs, the theoretical EER (cooling capacity ÷ power input) was compared with the recorded EER. The median absolute difference was 0.21, indicating reasonable consistency. For inverter ACs, the median difference was 3.96, which is expected because the efficiency metric for inverter ACs is CSPF (Cooling Seasonal Performance Factor), not EER, and CSPF is calculated using a seasonal weighting methodology rather than a simple ratio (ISO, 2013).

2. **Tariff cross-check:** The implied electricity tariff (annual cost ÷ annual consumption) had a median of Rp 1,444.71/kWh, which closely matches the standard PLN non-subsidised residential tariff of approximately Rp 1,444/kWh (PLN, 2023). This confirms that the Biaya Listrik Tahunan is a derived variable, computed as Konsumsi Energi Tahunan × tariff, which has critical implications for data leakage (Section 3.8).

### 3.6 Statistical Summary

Descriptive statistics for all numeric variables are presented in Table 5. Several important distributional characteristics emerge:

**Table 5.** Descriptive statistics of numeric variables (n = 1,923)

| Variable | Mean | SD | Min | P25 | Median | P75 | P95 | Max | Skew | Kurt |
|---|---|---|---|---|---|---|---|---|---|---|
| Daya (W) | 1,060.1 | 1,068.2 | 1.2 | 593.4 | 840.1 | 1,374.8 | 2,035.1 | 20,400.0 | 10.07 | 144.22 |
| Kapasitas (BTU/h) | 11,116.6 | 6,648.7 | 4.7 | 6,842.7 | 9,235.0 | 14,688.9 | 22,173.8 | 131,617.8 | 5.65 | 89.16 |
| EER/CSPF | 8.78 | 4.27 | 3.10 | 4.02 | 10.30 | 11.90 | 15.06 | 21.48 | 0.08 | −1.18 |
| Rating Bintang | 3.41 | 1.08 | 1.00 | 2.00 | 4.00 | 4.00 | 5.00 | 5.00 | −0.73 | −0.53 |
| Konsumsi (kWh) | 5,307.6 | 114,922.8 | 3.6 | 1,165.5 | 2,271.0 | 3,358.7 | 5,875.2 | 5,040,796.0 | 43.81 | 1,920.72 |
| Biaya (Rp) | 3,906,898 | 4,610,814 | 0 | 1,683,856 | 3,285,124 | 4,823,164 | 8,525,974 | 99,999,999 | 11.30 | 193.75 |

*Note.* SD = standard deviation; Skew = skewness; Kurt = excess kurtosis.

The high skewness and kurtosis values for Daya (10.07, 144.22), Konsumsi Energi (43.81, 1,920.72), and Biaya Listrik (11.30, 193.75) indicate extremely right-skewed distributions with heavy tails, driven by the multi-model registration outliers. In contrast, EER/CSPF exhibits near-symmetric distribution (skewness = 0.08, kurtosis = −1.18), and Rating Bintang is moderately left-skewed (skewness = −0.73), reflecting the dominance of 4-star products.

### 3.7 Energy Efficiency Patterns

#### 3.7.1 Star Rating Distribution

![Figure 1: Histogram of numeric variables](outputs/figures/L1_histograms_numerik.png)

**Figure 1.** Distribution of numeric variables in the SIMEBTKE AC database.

The star rating distribution (Figure 1, bottom right panel) reveals a pronounced concentration at 4 stars (n = 1,071, 55.7%), followed by 2 stars (n = 394, 20.5%), 3 stars (n = 184, 9.6%), 5 stars (n = 166, 8.6%), and 1 star (n = 108, 5.6%). This distribution suggests that the majority of registered products meet a moderate-to-good efficiency standard, but relatively few achieve the highest efficiency tier.

A critical finding emerges when cross-tabulating rating by AC type (Table 6): **all 5-star-rated products are inverter-type** (166/166 = 100%), and **all 1-star products are non-inverter** (108/108 = 100%). Among inverter ACs, 70.95% carry 4 stars and 26.65% carry 5 stars, while among non-inverter ACs, 48.38% carry 4 stars and none carry 5 stars. This finding underscores a clear technological divide: inverter technology is necessary—but not sufficient—for achieving the highest energy efficiency tier.

**Table 6.** Star rating distribution by AC type

| Rating | Inverter | Non-Inverter | Total |
|---|---|---|---|
| 1 star | 0 (0.0%) | 108 (8.3%) | 108 (5.6%) |
| 2 stars | 4 (0.6%) | 390 (30.0%) | 394 (20.5%) |
| 3 stars | 11 (1.8%) | 173 (13.3%) | 184 (9.6%) |
| 4 stars | 442 (70.9%) | 629 (48.4%) | 1,071 (55.7%) |
| 5 stars | 166 (26.6%) | 0 (0.0%) | 166 (8.6%) |
| **Total** | **623 (100%)** | **1,300 (100%)** | **1,923 (100%)** |

*Note.* Percentages are column-wise.

#### 3.7.2 Efficiency Metrics: EER vs. CSPF

![Figure 2: Boxplot by AC type](outputs/figures/L2_boxplot_per_tipe.png)

**Figure 2.** Boxplot comparison of numeric variables between inverter and non-inverter AC types.

The efficiency metric (EER/CSPF) exhibits a bimodal distribution (Figure 2), reflecting the different measurement methodologies for the two AC types. For non-inverter ACs, the mean EER is 8.16 (median = 10.13), while for inverter ACs, the mean CSPF is 10.09 (median = 11.36). The overlap in distributions occurs because the SIMEBTKE database stores both EER and CSPF in the same column, which can be misleading without stratification by type. This finding has important implications for any modelling approach: the EER/CSPF variable should be stratified or separated by AC type to avoid conflating two fundamentally different efficiency metrics (ISO, 2013; ISO, 2017).

#### 3.7.3 Brand Landscape

![Figure 3: Top 20 brands](outputs/figures/L3_top_merek.png)

**Figure 3.** Top 20 AC brands by number of registered models, stratified by type.

The 98 unique brands in the dataset are led by Gree (n = 135, combining "Gree" and "GREE" variants), LG (119), Panasonic (114), Daikin (111), and Midea (73). A case-sensitivity inconsistency was identified: "Gree" (135 records) and "GREE" (79 records) appear to refer to the same manufacturer, requiring normalisation during preprocessing. The top 5 brands collectively account for 552 records (28.7%), indicating a moderately concentrated market. The split between inverter and non-inverter varies considerably by brand: Daikin and LG have a higher proportion of inverter models, while brands like TCL and Aqua are predominantly non-inverter.

#### 3.7.4 Power–Capacity–Efficiency Relationships

![Figure 4: Scatter plots](outputs/figures/L5_scatter_daya_kapasitas_eer.png)

**Figure 4.** Scatter plots of power input vs. cooling capacity (left) and power input vs. efficiency value (right), coloured by star rating and shaped by AC type.

The scatter plots (Figure 4) reveal a positive linear relationship between power input and cooling capacity, as expected from the physical relationship between these variables. The efficiency value shows considerable scatter against power input, with two distinct clusters corresponding to inverter (higher CSPF values, typically >11) and non-inverter (lower EER values, typically <11) ACs. Higher star ratings are associated with higher efficiency values, particularly for inverter ACs in the upper efficiency range.

### 3.8 Correlation Analysis

![Figure 5: Correlation heatmap](outputs/figures/L6_correlation_heatmap.png)

**Figure 5.** Pearson correlation heatmap of numeric variables.

The Pearson correlation analysis (Figure 5) reveals several important relationships:

1. **Daya–Biaya (r = 0.649):** A moderate positive correlation exists between power input and annual electricity cost, as expected since higher-power ACs consume more electricity.

2. **EER–Rating (r = 0.544 overall):** The correlation between efficiency value and star rating is moderate overall, but this masks a critical difference by AC type (Table 7). For non-inverter ACs, EER and rating are strongly positively correlated (r = 0.828), confirming that the star rating is directly determined by EER for this type. For inverter ACs, however, CSPF and rating are **negatively** correlated (r = −0.444), which appears counterintuitive. This negative correlation arises because inverter ACs with 4 stars (the majority) tend to have higher CSPF values than those with 5 stars, likely because the 5-star threshold for inverter ACs may be based on a different criterion or because the CSPF–rating mapping for inverter ACs follows a different classification rule than for non-inverter ACs.

3. **Konsumsi–Biaya (r = 0.997 for non-inverter; r = 0.763 for inverter):** The near-perfect correlation for non-inverter ACs confirms that annual electricity cost is a direct linear derivation of annual energy consumption multiplied by a fixed tariff (≈Rp 1,444/kWh). This finding has critical implications for data leakage (Section 3.9).

4. **Daya–Konsumsi (r = 0.762 for non-inverter):** A strong positive correlation between power input and annual energy consumption for non-inverter ACs, reflecting the direct physical relationship.

**Table 7.** Key Pearson correlations by AC type

| Variable pair | Overall (n=1,923) | Non-inverter (n=1,300) | Inverter (n=623) |
|---|---|---|---|
| Daya–Kapasitas | 0.326 | 0.303 | 0.496 |
| EER/CSPF–Rating | 0.544 | 0.828 | −0.444 |
| Konsumsi–Biaya | 0.496 | 0.997 | 0.763 |
| Daya–Konsumsi | 0.032 | 0.762 | 0.047 |
| Daya–Biaya | 0.649 | 0.760 | 0.434 |

### 3.9 Data Leakage Assessment

Data leakage—the inadvertent inclusion of information in the training data that would not be available at prediction time—is a critical concern for predictive modelling (Kuhn & Johnson, 2013). The EDA identified three potential leakage pathways:

1. **Biaya Listrik Tahunan is derived from Konsumsi Energi Tahunan:** The near-perfect correlation (r = 0.997 for non-inverter) and the consistent implied tariff (median Rp 1,444.71/kWh) confirm that annual electricity cost is computed as `annual_cost = annual_consumption × tariff`. Therefore, including both variables as predictors in a model that predicts either one would constitute leakage. If the modelling target is annual cost, energy consumption should be excluded, and vice versa.

2. **Konsumsi Energi Tahunan is derived from EER/CSPF and Daya:** For non-inverter ACs, annual energy consumption is computed as `consumption = (power_input × operating_hours) / EER`, where operating hours are assumed at 8 hours/day (as stated in the SIMEBTKE portal footnote). Therefore, if the modelling target is EER, including energy consumption as a predictor would constitute leakage. Similarly, if the target is energy consumption, EER should be excluded.

3. **Rating Bintang is derived from EER/CSPF:** The star rating is determined by thresholding the efficiency value (EER for non-inverter, CSPF for inverter), as confirmed by the strong correlation (r = 0.828 for non-inverter). Therefore, including EER/CSPF as a predictor in a model that classifies star rating would be circular. The appropriate predictors for star rating classification are power input, cooling capacity, and AC type, not the efficiency value itself.

![Figure 6: Consumption and cost by star rating](outputs/figures/L7_konsumsi_biaya_per_rating.png)

**Figure 6.** Annual energy consumption (left) and electricity cost (right) by star rating.

### 3.10 Certification Body Distribution

![Figure 7: LSPro distribution](outputs/figures/L8_lspro_distribution.png)

**Figure 7.** Distribution of Product Certification Bodies (LSPro) for SIMEBTKE-registered AC products.

Among the 810 records with LSPro information (42.1% of the dataset), five certification bodies are represented, led by PT Qualis Indonesia (n = 307, 37.9%), PT TUV Rheinland Indonesia (n = 265, 32.7%), and the Balai Besar Standardisasi dan Pelayanan Jasa Industri Bahan dan Barang Teknik (BBSPJIBBT) (n = 121, 14.9%). The dominance of private certification bodies (Qualis and TUV Rheinland account for 70.6% of certified records) reflects the outsourced nature of product testing in Indonesia's energy labelling programme.

### 3.11 Temporal Patterns

Among the 465 records with valid certification dates, the distribution shows increasing registration activity over time: 8 certificates in 2021, 35 in 2022, 23 in 2023, 158 in 2024, 144 in 2025, and 93 projected for 2026–2028. This upward trend suggests growing manufacturer participation in the SHE programme, though the 75.82% missing rate for certification dates limits the generalisability of this temporal analysis.

---

## 4. Statistical Analysis and Machine Learning Results

### 4.1 Normality Testing

Shapiro-Wilk tests confirmed that all five numeric variables deviate significantly from normality (W < 0.90, p < 0.001 for all variables). Consequently, all subsequent group comparisons employed non-parametric tests (Kruskal-Wallis, Mann-Whitney U, Spearman correlation).

### 4.2 Kruskal-Wallis: EER/CSPF across Star Ratings

The Kruskal-Wallis test revealed a highly significant difference in EER/CSPF across the five star rating levels (H = 1298.52, p < 0.001, η² = 0.68, indicating a large effect size). Post-hoc pairwise Mann-Whitney U tests with Bonferroni correction confirmed significant differences between all rating pairs (p_adj < 0.05), except Rating 3 vs. 5 (p_adj = 0.354, ns). Notably, the comparison between Rating 4 and Rating 5 yielded a negative rank-biserial correlation (r_rb = −0.775), indicating that 4-star products have higher median EER/CSPF (11.64) than 5-star products (5.60)—a paradox attributable to the different metric basis (CSPF for inverter vs. EER for non-inverter ACs within the same column).

### 4.3 Mann-Whitney U: Inverter vs. Non-Inverter

The efficiency value differed significantly between inverter (median = 11.36) and non-inverter (median = 10.14) ACs (U = 552,956, p < 0.001, rank-biserial r = −0.384). However, the moderate effect size suggests substantial overlap between the two distributions, consistent with the bimodal pattern observed in the EDA.

### 4.4 Chi-Square: Rating × Type Association

The chi-square test confirmed a strong and significant association between star rating and AC type (χ² = 666.32, df = 4, p < 0.001, Cramér's V = 0.591). Standardised residuals revealed that the most significant deviations from expected frequencies occurred at Rating 5 (z = 15.22 for inverter, z = −10.56 for non-inverter) and Rating 2 (z = −10.87 for inverter, z = 7.54 for non-inverter), confirming the structural zero pattern: no 5-star non-inverter and no 1-star inverter products exist in the database.

### 4.5 Partial Correlation Analysis

Partial correlation analysis (residualisation method) revealed that after controlling for cooling capacity, the correlation between power input and EER/CSPF became negligible (r = 0.004, p = 0.86), confirming that efficiency is not directly determined by power input. Conversely, the Daya–Kapasitas correlation remained strong after controlling for EER (r = 0.731, p < 0.001), reflecting the fundamental physical relationship. The Konsumsi–Biaya Spearman correlation (ρ = 0.974) confirmed the derived-variable data leakage pathway.

### 4.6 K-Means Clustering

![Figure 8: Elbow and silhouette scores](outputs/figures/M1_elbow_silhouette.png)

**Figure 8.** Elbow method (left) and silhouette score (right) for determining the optimal number of K-Means clusters.

K-Means clustering with k = 5 (silhouette score = 0.519) identified five distinct product profiles (Table 8). The largest cluster (Cluster 2: n = 825, 43.2%) represents small-capacity (median 8,530 BTU/h), high-efficiency (median EER/CSPF = 11.63) units—typical 1 PK inverter ACs. Cluster 0 (n = 593, 31.1%) comprises small-capacity, low-efficiency units (median EER/CSPF = 3.73), representing entry-level non-inverter products. Clusters 1 and 3 represent large-capacity units (>17,000 BTU/h), differentiated by efficiency level (median EER/CSPF 11.35 vs. 3.79).

![Figure 9: K-Means cluster visualisation](outputs/figures/M2_clusters.png)

**Figure 9.** K-Means cluster visualisation in Daya–Kapasitas (left) and Daya–EER/CSPF (right) space.

**Table 8.** K-Means cluster profiles (k = 5)

| Cluster | Label | n | Daya median (W) | Kap. median (BTU/h) | EER/CSPF median | Rating mode |
|---|---|---|---|---|---|---|
| 0 | Sedang–Kurang Efisien | 593 | 762 | 9,000 | 3.73 | 2 |
| 1 | Besar–Efisien | 281 | 1,691 | 17,983 | 11.35 | 4 |
| 2 | Sedang–Efisien | 825 | 742 | 8,530 | 11.63 | 4 |
| 3 | Besar–Kurang Efisien | 208 | 1,681 | 18,062 | 3.79 | 2 |
| 4 | Outlier | 2 | 1,139 | 124,666 | 4.42 | — |

### 4.7 Classification: Star Rating Prediction

Four classification models were evaluated for predicting the 5-level star rating from leakage-safe predictors (Daya, Kapasitas, Tipe, Kategori_PK). Random Forest achieved the best performance (Table 9).

**Table 9.** Classification model comparison

| Model | CV F1 (mean ± SD) | Test Accuracy | Test F1 (weighted) |
|---|---|---|---|
| **Random Forest** | **0.594 ± 0.019** | **0.668** | **0.641** |
| KNN (k=7) | 0.560 ± 0.025 | 0.615 | 0.605 |
| Decision Tree | 0.508 ± 0.038 | 0.576 | 0.542 |
| Logistic Regression | 0.416 ± 0.009 | 0.552 | 0.424 |

![Figure 10: Confusion matrix](outputs/figures/M3_confusion_matrix.png)

**Figure 10.** Confusion matrix for the Random Forest classifier.

The Random Forest model achieved its highest classification performance for 4-star products (precision = 0.71, recall = 0.84, F1 = 0.77), which is expected given that 4-star is the majority class (55.7%). Performance was poorest for 1-star (F1 = 0.24) and 5-star (F1 = 0.34) products, reflecting both class imbalance and the fundamental limitation that power input and cooling capacity alone cannot fully discriminate between adjacent rating tiers—the discriminating variable (EER/CSPF) was deliberately excluded to prevent leakage.

![Figure 11: Classification feature importance](outputs/figures/M4_clf_feature_importance.png)

**Figure 11.** Feature importance for the Random Forest classifier.

### 4.8 Regression: EER/CSPF Prediction

Four regression models were evaluated for predicting the continuous EER/CSPF value from the same leakage-safe predictor set (excluding Konsumsi, Biaya, and Rating). Gradient Boosting achieved the best performance (Table 10).

**Table 10.** Regression model comparison

| Model | CV R² (mean ± SD) | Test R² | Test RMSE | Test MAE |
|---|---|---|---|---|
| **Gradient Boosting** | **0.212 ± 0.044** | **0.225** | **3.75** | **2.99** |
| Random Forest | 0.220 ± 0.036 | 0.196 | 3.82 | 3.18 |
| Ridge | 0.028 ± 0.053 | −0.004 | 4.27 | 4.03 |
| Linear Regression | 0.024 ± 0.061 | −0.004 | 4.27 | 4.03 |

![Figure 12: Actual vs predicted EER/CSPF](outputs/figures/M5_regression_actual_vs_pred.png)

**Figure 12.** Actual vs. predicted EER/CSPF for the Gradient Boosting regressor.

The moderate R² value (0.225) indicates that approximately 22% of the variance in EER/CSPF is explainable by power input, cooling capacity, AC type, and PK category. The remaining variance is attributable to internal technical factors not captured in the database, such as compressor type, refrigerant, heat exchanger design, and fan efficiency (ISO, 2013; ISO, 2017). The near-zero R² for linear models (−0.004) confirms that the EER/CSPF relationship with available predictors is non-linear, requiring tree-based models to capture.

![Figure 13: Model comparison](outputs/figures/M7_model_comparison.png)

**Figure 13.** Performance comparison across all classification (left) and regression (right) models.

---

## 5. Conclusion

This study presents the first comprehensive exploratory data analysis of the SIMEBTKE AC product database, encompassing 1,923 records of inverter and non-inverter air conditioners registered in Indonesia's energy labelling programme. The key findings and implications are:

1. **Data quality issues require attention:** The SIMEBTKE database contains significant missing values in certification date fields (75.82%), case inconsistencies in brand names (e.g., "Gree" vs. "GREE"), 541 duplicate registration numbers from batch certification practices, and format inconsistencies in the electricity cost column (comma-based thousand separators). These issues must be addressed in a systematic data cleaning pipeline before any quantitative analysis or predictive modelling.

2. **A clear technological divide exists:** All 5-star-rated ACs in the database are inverter-type, while all 1-star-rated ACs are non-inverter. This binary separation suggests that inverter technology is a prerequisite for the highest efficiency tier, though the majority of inverter products (70.95%) achieve only 4 stars. The different efficiency metrics (EER vs. CSPF) for the two types must be handled separately in any modelling approach.

3. **The efficiency–rating relationship is type-dependent:** The correlation between efficiency value and star rating is strongly positive for non-inverter ACs (r = 0.828) but negative for inverter ACs (r = −0.444), reflecting the different classification criteria and metric bases. This finding has implications for any classification model that attempts to predict star rating: type-specific modelling or interaction terms should be considered.

4. **Data leakage pathways are significant:** Three leakage pathways were identified: (a) annual electricity cost is derived from energy consumption and a fixed tariff (r = 0.997); (b) energy consumption is derived from efficiency value and power input; and (c) star rating is derived from efficiency value. Future predictive models must carefully exclude derived variables from the predictor set to avoid inflated performance estimates.

5. **The market is moderately concentrated:** The top 5 brands (Gree, LG, Panasonic, Daikin, Midea) account for 28.7% of registered products, with significant variation in the inverter/non-inverter mix across brands. This concentration may have implications for market-based efficiency improvement strategies.

6. **Clustering reveals five product archetypes:** K-Means clustering identified five distinct product profiles, from small-capacity low-efficiency entry-level units (Cluster 0, 31.1%) to large-capacity high-efficiency premium units (Cluster 1, 14.7%). These clusters align with market segmentation patterns and could inform targeted policy interventions.

7. **Star rating prediction is moderately feasible without leakage:** Random Forest achieved 66.8% accuracy in predicting star rating from power, capacity, type, and PK category alone. Performance was highest for the majority class (4-star, F1 = 0.77) and lowest for minority classes (1-star and 5-star), suggesting that additional features (e.g., compressor type, refrigerant) would be needed for reliable prediction of extreme ratings.

8. **EER/CSPF prediction has limited predictive power:** Gradient Boosting explained only 22.5% of variance in EER/CSPF from available database features, indicating that the dataset's external specifications (power, capacity) are insufficient to predict internal efficiency performance. This finding underscores the need for richer technical data in the SIMEBTKE database to support data-driven efficiency prediction.

Based on these findings, the following next steps are recommended: (a) systematic data cleaning, including brand name normalisation, duplicate registration number handling, and outlier treatment for multi-model registrations; (b) stratified analysis by AC type (inverter vs. non-inverter) to account for the different efficiency metrics; (c) feature engineering, including derived variables such as the capacity-to-power ratio and PK (Paarde Kracht) categorisation; (d) statistical hypothesis testing (e.g., Kruskal-Wallis tests) to confirm observed differences across rating tiers; and (e) predictive modelling with careful avoidance of identified leakage pathways, focusing on power input, cooling capacity, and AC type as legitimate predictors of energy efficiency.

---

## References

1. BPS (Badan Pusat Statistik). (2023). *Statistik Konsumsi Energi Rumah Tangga 2023*. Jakarta: BPS-Statistics Indonesia.

2. Daikin Industries, Ltd. (2023). *Annual Report 2023: Towards a Carbon-Neutral Future*. Osaka: Daikin Industries.

3. DJEBTKE (Direktorat Jenderal Energi Baru, Terbarukan dan Konservasi Energi). (2024). *Website Produk Berlabel Hemat Energi*. Retrieved from https://simebtke.esdm.go.id/sinergi/skem-label/konsumen/pengondisi-udara-ac

4. Field, A. (2018). *Discovering Statistics Using IBM SPSS Statistics* (5th ed.). London: SAGE Publications.

5. Government of Indonesia. (2009). *Government Regulation No. 70 of 2009 on Energy Conservation*. Jakarta: State Secretariat.

6. Government of Indonesia. (2014). *Government Regulation No. 79 of 2014 on National Energy Policy*. Jakarta: State Secretariat.

7. Harris, C. R., Millman, K. J., van der Walt, S. J., et al. (2020). Array programming with NumPy. *Nature*, 585(7825), 357–362. https://doi.org/10.1038/s41586-020-2649-2

8. Hu, S., Yan, D., Guo, S., Liu, Y., Qiao, M., & Jiang, Y. (2020). Analysis of the air-conditioning energy consumption and cooling demand of residential buildings in China. *Energy and Buildings*, 224, 110240. https://doi.org/10.1016/j.enbuild.2020.110240

9. Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. *Computing in Science & Engineering*, 9(3), 90–95. https://doi.org/10.1109/MCSE.2007.55

10. IEA (International Energy Agency). (2018). *The Future of Cooling: Opportunities for Energy-Efficient Air Conditioning*. Paris: IEA. https://doi.org/10.1787/9789264301995-en

11. ISO (International Organization for Standardization). (2013). *ISO 16358-1:2013 Air-cooled air conditioners and air-to-air heat pumps—Testing and calculating methods for seasonal performance factors—Part 1: Cooling seasonal performance factor*. Geneva: ISO.

12. ISO (International Organization for Standardization). (2017). *ISO 5151:2017 Non-ducted air conditioners—Testing and rating for performance*. Geneva: ISO.

13. Kuhn, M., & Johnson, K. (2013). *Applied Predictive Modeling*. New York: Springer. https://doi.org/10.1007/978-1-4614-6849-3

14. McKinney, W. (2017). *Python for Data Analysis: Data Wrangling with Pandas, NumPy, and IPython* (2nd ed.). Sebastopol: O'Reilly Media.

15. Pérez-Lombard, L., Ortiz, J., & Pout, C. (2008). A review on buildings energy consumption information. *Energy and Buildings*, 40(3), 394–398. https://doi.org/10.1016/j.enbuild.2007.03.007

16. PLN (Perusahaan Listrik Negara). (2023). *Tarif Tenaga Listrik Non-Subsidi 2023*. Jakarta: PT PLN (Persero).

17. Saidur, R., Ahamed, J. U., & Masjuki, H. H. (2009). Energy, exergy and economic analysis of industrial boilers. *Energy Policy*, 37(5), 1760–1768. https://doi.org/10.1016/j.enpol.2008.12.024

18. Santamouris, M., & Kolokotsa, D. (2013). Passive cooling dissipation techniques for buildings and other structures: The state of the art. *Energy and Buildings*, 57, 74–94. https://doi.org/10.1016/j.enbuild.2012.11.002

19. Samsung Electronics Co., Ltd. (2024). *Sustainability Report 2024*. Suwon: Samsung Electronics.

20. Tukey, J. W. (1977). *Exploratory Data Analysis*. Reading, MA: Addison-Wesley.

21. Ürge-Vorsatz, D., Petrichenko, K., Antosik, M., et al. (2015). Measuring the co-benefits of climate change mitigation: Making it matter. *Climate Change*, 5(4), 399–402.

22. Virtanen, P., Gommers, R., Oliphant, T. E., et al. (2020). SciPy 1.0: Fundamental algorithms for scientific computing in Python. *Nature Methods*, 17(3), 261–272. https://doi.org/10.1038/s41592-019-0686-2

23. Waskom, M. L. (2021). Seaborn: Statistical data visualization. *Journal of Open Source Software*, 6(60), 3021. https://doi.org/10.21105/joss.03021

24. Wickham, H. (2014). Tidy data. *Journal of Statistical Software*, 59(10), 1–23. https://doi.org/10.18637/jss.v059.i10

25. Wilkinson, M. D., Dumontier, M., Aalbersberg, I. J., et al. (2016). The FAIR Guiding Principles for scientific data management and stewardship. *Scientific Data*, 3, 160018. https://doi.org/10.1038/sdata.2016.18

26. Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.

27. Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32. https://doi.org/10.1023/A:1010933404324

28. Cover, T., & Hart, P. (1967). Nearest neighbor pattern classification. *IEEE Transactions on Information Theory*, 13(1), 21–27. https://doi.org/10.1109/TIT.1967.1053964

29. Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *Annals of Statistics*, 29(5), 1189–1232. https://doi.org/10.1214/aos/1013203451

30. Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the interpretation and validation of cluster analysis. *Journal of Computational and Applied Mathematics*, 20, 53–65. https://doi.org/10.1016/0377-0427(87)90125-7

---

*Note: Authors should verify all references against the Scopus database (www.scopus.com) for indexing status and accurate citation details. Government regulations and standards are primary sources. The DOI links provided are the best-known at the time of writing.*
