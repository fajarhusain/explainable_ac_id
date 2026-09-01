# Energy Efficiency Patterns of Air Conditioners in Indonesia: A Data-Driven Analysis of the SIMEBTKE Database

---

**Authors:** [Author Name], [Co-author Name]  
**Affiliation:** [Institution]  
**Corresponding email:** [email]  

---

## Abstract

Indonesia's household electricity consumption has risen sharply over the last decade, with air conditioning units driving much of that growth. The Sistem Informasi Manajemen Efisiensi dan Konservasi Energi (SIMEBTKE), operated by the Directorate General of New, Renewable Energy and Energy Conservation under the Ministry of Energy and Mineral Resources, maintains a public database of AC products that have received the Sertifikat Hemat Energi (SHE). We retrieved 1,923 records from this database—623 inverter and 1,300 non-inverter models—and subjected them to exploratory data analysis, statistical hypothesis testing, and machine learning modelling. The data show that more than half of registered products (55.7%) sit at 4-star efficiency, while only 8.6% reach the top 5-star tier. Every 5-star unit in the database runs on inverter technology; every 1-star unit does not. For non-inverter ACs, EER correlates strongly with star rating (r = 0.828), but the relationship flips for inverter units (r = −0.444) because the database stores CSPF and EER in the same field despite their different calculation bases. We also found that annual electricity cost is essentially a linear function of energy consumption (ρ = 0.997), which creates a data leakage trap if both variables enter a predictive model. K-Means clustering sorted the products into five groups that track market segmentation from entry-level to premium. A Random Forest classifier predicted star rating at 66.8% accuracy using only power, capacity, type, and PK category—deliberately excluding EER/CSPF to avoid leakage. Regression of EER/CSPF onto the same predictor set explained just 22.5% of variance, suggesting that the database's external specifications lack the technical detail needed to predict internal efficiency. These results expose both the potential and the limits of using government labelling data for efficiency analysis in a developing-country context.

**Keywords:** air conditioner; energy efficiency; EER; CSPF; energy labeling; SIMEBTKE; exploratory data analysis; machine learning; clustering; classification; Indonesia

---

## 1. Introduction

Indonesia's electricity demand keeps climbing. Urbanisation, higher disposable incomes, and broader grid access all push consumption upward (Government of Indonesia, 2014; IEA, 2018). Within the household sector, air conditioning units stand out as a major load driver—unavoidable in a tropical climate where cooling is needed year-round (Pérez-Lombard et al., 2008; Santamouris & Kolokotsa, 2013). The IEA (2018) expects global cooling energy demand to triple by 2050, and Southeast Asia will absorb a disproportionate chunk of that surge. Indonesian household AC penetration rose from below 10% in 2010 to roughly 20–25% by 2023 (BPS, 2023), and the curve is steepening.

The government's response has been regulatory. Government Regulation No. 70 of 2009 on Energy Conservation set efficiency floors for household equipment (Government of Indonesia, 2009). Five years later, Regulation No. 79 of 2014 on the National Energy Policy tightened those targets and tied them to renewable energy goals (Government of Indonesia, 2014). At the centre of this apparatus sits the Sertifikat Hemat Energi (SHE)—a certificate issued to products meeting minimum efficiency thresholds, displayed to consumers through a 1-to-5 star label.

SIMEBTKE is the digital backbone of that programme. Run by the DJEBTKE, it publishes manufacturer-submitted data on every certified AC unit: brand, family, model, type (inverter or non-inverter), power input, cooling capacity, efficiency value, star rating, estimated annual energy consumption, estimated annual electricity cost, registration number, certification dates, and the testing body (LSPro). The portal is live and publicly queryable—but as a dataset for research, it has received little academic attention.

Most published work on AC efficiency falls into two camps: laboratory performance testing against ISO standards (ISO, 2013, 2017) and macro-level consumption modelling (Ürge-Vorsatz et al., 2015; Hu et al., 2020). What is missing is a data-driven, bottom-up look at what a national labelling database actually contains—its structure, its quirks, its patterns, and its limitations as a modelling substrate. Exploratory data analysis (Tukey, 1977) is the natural starting point. It surfaces data quality problems before they contaminate downstream models (Wickham, 2014; McKinney, 2017), flags distributional oddities and outliers, exposes correlations that might indicate data leakage (Kuhn & Johnson, 2013), and generates hypotheses worth testing formally (Field, 2018). For energy labelling data, one specific risk looms large: derived variables. If annual electricity cost is just consumption multiplied by a fixed tariff, including both as predictors inflates model performance artificially.

We address that gap here. The paper has five objectives: (1) assess SIMEBTKE data quality—missingness, duplication, format inconsistencies; (2) characterise how efficiency variables distribute across AC types and brands; (3) flag leakage pathways; (4) run statistical tests on observed patterns; and (5) build and evaluate machine learning models (clustering, classification, regression) with explicit leakage safeguards. Section 2 covers data and methods. Section 3 reports EDA findings. Section 4 presents statistical and ML results. Section 5 wraps up.

---

## 2. Method

### 2.1 Data Source

Data came from the public SIMEBTKE consumer portal at `https://simebtke.esdm.go.id/sinergi/skem-label/konsumen/pengondisi-udara-ac` (DJEBTKE, 2024). The site splits AC products into inverter and non-inverter tabs, each populated through server-side AJAX calls with Bootstrap Table pagination. We wrote a Python script that loops through paginated HTTP requests—sending `limit`, `offset`, `page`, and `search` parameters—and pulls every record. The final count: 623 inverter and 1,300 non-inverter units, 1,923 in total. Raw JSON and CSV copies went into `data/raw/` untouched, following FAIR data principles (Wilkinson et al., 2016).

### 2.2 Variables

Fifteen fields come with each record. They break into four groups:

**Identity:** NO. (row number), Merek (brand), Famili (family), Model, Tipe (inverter/non-inverter), No. Registrasi/No. SHE (certificate number), LSPro (certification body).

**Technical specs:** Daya (power input, W), Kapasitas Pendinginan (cooling capacity, BTU/h), Nilai Efisiensi (EER for non-inverter or CSPF for inverter).

**Energy performance:** Rating Bintang (star rating, 1–5), Konsumsi Energi Tahunan (annual consumption, kWh), Biaya Listrik Tahunan (annual cost, Rp).

**Certification dates:** Tanggal Terbit SHE (issue date), SHE Berlaku Sampai Dengan Tanggal (expiry date).

### 2.3 EDA Protocol

We ran a 13-step pipeline (steps A–M):

Steps A–D: load with `dtype=str` to preserve raw formatting, inspect shape, column types, and head.

Step E: scan for missing values—including empty strings, "null," "NA," dashes, and similar placeholders (Wickham, 2014).

Step F: three duplicate checks—full-row, row minus NO., and registration number.

Step G: frequency tables for every categorical column.

Step H: test whether each numeric field parses cleanly via `pd.to_numeric()`, and look for commas, "Rp" prefixes, or unit suffixes.

Step I: validate dates against ISO 8601 (YYYY-MM-DD).

Step J: IQR-based outlier detection (Tukey, 1977) plus two cross-checks—(a) does EER ≈ capacity ÷ power? and (b) does implied tariff ≈ Rp 1,444/kWh?

Step K: descriptive statistics with percentile breakdowns and skewness/kurtosis.

Step L: eight visualisations covering histograms, boxplots, bar charts, scatter plots, and a correlation heatmap.

Step M: no machine learning at this stage—consistent with the principle that EDA comes first (Tukey, 1977; Kuhn & Johnson, 2013).

### 2.4 Tools

Python 3, pandas 3.0 (McKinney, 2017), numpy 2.5 (Harris et al., 2020), matplotlib 3.11 (Hunter, 2007), seaborn 0.13 (Waskom, 2021), scipy 1.18 (Virtanen et al., 2020), scikit-learn 1.9 (Pedregosa et al., 2011). Every transformation is logged; intermediate outputs land in `data/processed/`, `outputs/figures/`, and `outputs/tables/`.

### 2.5 Machine Learning Setup

Three tasks, each built with leakage prevention baked in:

**Clustering (K-Means).** Standardised Daya, Kapasitas, and EER/CSPF. Scanned k = 2–10 via elbow (inertia) and silhouette score (Rousseeuw, 1987). Picked the k maximising silhouette. Profiles described by median feature values and modal rating/type.

**Classification (star rating).** Predictors: Daya, Kapasitas, Tipe, Kategori_PK. Excluded: EER/CSPF, Konsumsi, Biaya—all three derive from or determine the rating. Algorithms: Random Forest (Breiman, 2001), KNN (Cover & Hart, 1967), Decision Tree, Logistic Regression. Evaluation: 5-fold stratified CV (F1-weighted) + 20% hold-out test. Metrics: accuracy, F1, confusion matrix.

**Regression (EER/CSPF).** Same predictor set, minus Rating (derived from EER/CSPF). Algorithms: Random Forest, Ridge, Gradient Boosting (Friedman, 2001), Linear Regression. Evaluation: 5×3 repeated CV (R²) + 20% hold-out. Metrics: R², RMSE, MAE.

Z-score standardisation on all numeric features. One-hot encoding for categoricals. Seed = 42.

---

## 3. Results and Discussion

### 3.1 What the Database Looks Like

**Table 1** summarises the dataset. Non-inverter dominates at 67.6%, which tracks the Indonesian market's historical preference for cheaper, simpler technology. That share will likely shrink as inverter prices fall (Daikin Industries, 2023; Samsung Electronics, 2024).

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

### 3.2 Missing Values

Three columns are heavily incomplete (Table 2). Certification dates (both issue and expiry) are missing for 1,458 rows—75.82%. LSPro is absent for 57.88%. Two plausible explanations: older entries predate the digital system, and some products were certified through paper-based channels before SIMEBTKE went live. The technical and energy columns, fortunately, are fully populated.

**Table 2.** Missing value summary

| Variable | Missing count | Percentage (%) |
|---|---|---|
| Tanggal Terbit SHE | 1,458 | 75.82 |
| SHE Berlaku Sampai Dengan Tanggal | 1,458 | 75.82 |
| LSPro | 1,113 | 57.88 |
| All other variables | 0 | 0.00 |

### 3.3 Duplication

No two rows are byte-for-byte identical. Drop the NO. column, though, and 10 pairs emerge. The real problem is registration numbers: 541 records share a No. Registrasi/No. SHE with at least one other row. These are not duplicates in the usual sense. Manufacturers register entire model line-ups under a single certificate—Mitsubishi Heavy Industries, for instance, filed SRK13YYP-W3 and SRK18YYM-W3 under one number (16/06.06.06/1A/24/LSPro/V/2024). Bestlife bundled over 60 variants under 004/LSP/QI/06.1-I/2024. Each row is a distinct product; the shared number just means one testing batch covered them all. Certificate-level analysis should aggregate by registration number, not by row.

**Table 3.** Duplicate row summary

| Duplicate type | Count |
|---|---|
| Full-row duplicates | 0 |
| Duplicates (excluding NO.) | 10 |
| Duplicates (No. Registrasi/No. SHE) | 541 |

### 3.4 Numeric and Date Formats

Every numeric column arrived as a string. Five of six parse cleanly: Daya, Kapasitas, EER/CSPF, Rating, and Konsumsi Energi all use plain decimals ("1120.00", "11.70"). Biaya Listrik Tahunan does not—it ships with comma thousands separators ("4,797,676.80"), so direct parsing caught only 27 of 1,923 values. Stripping commas before conversion fixes it.

Dates follow ISO 8601 (YYYY-MM-DD). One rogue entry—`0000-00-00`—failed to parse and should be treated as null.

### 3.5 Outliers and Anomalies

**Table 4** shows IQR-based outlier rates. They are low overall (0.3–1.8%), but a handful of extreme values demand attention.

**Table 4.** Outlier summary (IQR method, 1.5×IQR fence)

| Variable | IQR | Outlier range | Outliers | % |
|---|---|---|---|---|
| Daya (watt) | [593.4–1,374.8] | <−578.8 or >2,547.0 | 19 | 1.0 |
| Kapasitas Pendinginan (BTU/h) | [6,842.7–14,688.9] | <−4,926.6 or >26,458.3 | 6 | 0.3 |
| Konsumsi Energi Tahunan (kWh) | [1,165.5–3,358.7] | <−2,124.3 or >6,648.5 | 32 | 1.7 |
| Biaya Listrik Tahunan (Rp) | [1,683,856–4,823,164] | <−3,025,105 or >9,532,126 | 34 | 1.8 |

A BEKO unit (BIVOE 120) reports 1.16 W—physically impossible for a room AC and almost certainly a keystroke error. Ten records exceed 5,000 W, topping out at 20,400 W (Midea MSBE-24CRN1). These trace back to multi-model registrations where power figures represent combined unit loads, not a single device. The same pattern explains the maximum Konsumsi value of 5,040,796 kWh—about 2,200 times the median. Twenty-seven rows carry Biaya = Rp 0.00, and the ceiling sits at Rp 99,999,999.99, a suspiciously round placeholder.

Two cross-checks held up well. Theoretical EER (capacity ÷ power) matched recorded EER for non-inverter units with a median absolute difference of 0.21. The gap widened to 3.96 for inverter units, which makes sense—CSPF uses seasonal weighting, not a simple ratio (ISO, 2013). The implied tariff (cost ÷ consumption) hovered at Rp 1,444.71/kWh, dead-centre on PLN's non-subsidised residential rate (PLN, 2023). That confirms Biaya is derived from Konsumsi via a fixed multiplier—something to remember for leakage control.

### 3.6 Descriptive Statistics

**Table 5** gives the full picture. The skewness and kurtosis figures for Daya (10.07 / 144.22), Konsumsi (43.81 / 1,920.72), and Biaya (11.30 / 193.75) scream heavy-tailed distributions dragged right by multi-model outliers. EER/CSPF sits almost symmetric (skew = 0.08, kurtosis = −1.18). Rating Bintang leans left (−0.73) because 4-star products dominate.

**Table 5.** Descriptive statistics of numeric variables (n = 1,923)

| Variable | Mean | SD | Min | P25 | Median | P75 | P95 | Max | Skew | Kurt |
|---|---|---|---|---|---|---|---|---|---|---|
| Daya (W) | 1,060.1 | 1,068.2 | 1.2 | 593.4 | 840.1 | 1,374.8 | 2,035.1 | 20,400.0 | 10.07 | 144.22 |
| Kapasitas (BTU/h) | 11,116.6 | 6,648.7 | 4.7 | 6,842.7 | 9,235.0 | 14,688.9 | 22,173.8 | 131,617.8 | 5.65 | 89.16 |
| EER/CSPF | 8.78 | 4.27 | 3.10 | 4.02 | 10.30 | 11.90 | 15.06 | 21.48 | 0.08 | −1.18 |
| Rating Bintang | 3.41 | 1.08 | 1.00 | 2.00 | 4.00 | 4.00 | 5.00 | 5.00 | −0.73 | −0.53 |
| Konsumsi (kWh) | 5,307.6 | 114,922.8 | 3.6 | 1,165.5 | 2,271.0 | 3,358.7 | 5,875.2 | 5,040,796.0 | 43.81 | 1,920.72 |
| Biaya (Rp) | 3,906,898 | 4,610,814 | 0 | 1,683,856 | 3,285,124 | 4,823,164 | 8,525,974 | 99,999,999 | 11.30 | 193.75 |

### 3.7 Star Ratings and the Inverter Divide

![Figure 1: Histogram of numeric variables](outputs/figures/L1_histograms_numerik.png)

**Figure 1.** Distribution of numeric variables in the SIMEBTKE AC database.

The star rating histogram (Figure 1, bottom right) piles up at 4 stars—1,071 products, 55.7%. Two-star units come second (394, 20.5%), then 3-star (184, 9.6%), 5-star (166, 8.6%), and 1-star (108, 5.6%). Most certified products clear a decent bar; few reach the top.

Things get interesting when rating meets type (Table 6). Every single 5-star unit is an inverter. Every single 1-star unit is a non-inverter. Zero overlap at the extremes. Among inverter units, 70.95% hold 4 stars and 26.65% hold 5 stars. Non-inverter units cluster at 2 and 4 stars, with none reaching 5. Inverter technology looks necessary but not sufficient for the top tier—manufacturers still need to engineer the rest of the unit well enough to clear that bar.

**Table 6.** Star rating distribution by AC type

| Rating | Inverter | Non-Inverter | Total |
|---|---|---|---|
| 1 star | 0 (0.0%) | 108 (8.3%) | 108 (5.6%) |
| 2 stars | 4 (0.6%) | 390 (30.0%) | 394 (20.5%) |
| 3 stars | 11 (1.8%) | 173 (13.3%) | 184 (9.6%) |
| 4 stars | 442 (70.9%) | 629 (48.4%) | 1,071 (55.7%) |
| 5 stars | 166 (26.6%) | 0 (0.0%) | 166 (8.6%) |
| **Total** | **623 (100%)** | **1,300 (100%)** | **1,923 (100%)** |

### 3.8 EER vs. CSPF: One Column, Two Metrics

![Figure 2: Boxplot by AC type](outputs/figures/L2_boxplot_per_tipe.png)

**Figure 2.** Boxplot comparison of numeric variables between inverter and non-inverter AC types.

SIMEBTKE stores EER (non-inverter) and CSPF (inverter) in a single field. That design choice has consequences. Non-inverter EER averages 8.16 (median 10.13); inverter CSPF averages 10.09 (median 11.36). The distributions overlap, which means anyone working with this column must split by type first. Treating it as one metric conflates two fundamentally different calculation methodologies (ISO, 2013; ISO, 2017).

### 3.9 Brand Landscape

![Figure 3: Top 20 brands](outputs/figures/L3_top_merek.png)

**Figure 3.** Top 20 AC brands by number of registered models, stratified by type.

Ninety-eight brand strings appear in the raw data. But "Gree" (135 records) and "GREE" (79) are the same company—case inconsistency inflates the count. After normalisation, the field drops to 63 distinct brands. Gree leads at 214, followed by Panasonic (172), Daikin (153), LG (123), and Midea (120). The top five cover 28.7% of records. Daikin and LG lean inverter-heavy; TCL and Aqua stay mostly non-inverter.

### 3.10 Power, Capacity, and Efficiency

![Figure 4: Scatter plots](outputs/figures/L5_scatter_daya_kapasitas_eer.png)

**Figure 4.** Scatter plots of power input vs. cooling capacity (left) and power input vs. efficiency value (right), coloured by star rating and shaped by AC type.

Power and capacity track linearly (Figure 4, left)—physics demands it. Efficiency against power scatters widely, splitting into two bands: inverter units above EER/CSPF ≈ 11, non-inverter units below. Higher-rated products cluster in the upper band, as expected.

### 3.11 Correlations

![Figure 5: Correlation heatmap](outputs/figures/L6_correlation_heatmap.png)

**Figure 5.** Pearson correlation heatmap of numeric variables.

Four correlations stand out (Table 7). Daya and Biaya correlate at r = 0.649—bigger units cost more to run. Konsumsi and Biaya hit r = 0.997 for non-inverter units, confirming the derived-variable relationship. The EER–Rating link is where things get strange: r = 0.828 for non-inverter (rating tracks EER directly) but r = −0.444 for inverter. That negative sign looks wrong until you realise that 4-star inverter units carry higher CSPF values than 5-star ones—presumably because the 5-star threshold for inverter ACs rests on a different criterion than simple CSPF magnitude.

**Table 7.** Key Pearson correlations by AC type

| Variable pair | Overall (n=1,923) | Non-inverter (n=1,300) | Inverter (n=623) |
|---|---|---|---|
| Daya–Kapasitas | 0.326 | 0.303 | 0.496 |
| EER/CSPF–Rating | 0.544 | 0.828 | −0.444 |
| Konsumsi–Biaya | 0.496 | 0.997 | 0.763 |
| Daya–Konsumsi | 0.032 | 0.762 | 0.047 |
| Daya–Biaya | 0.649 | 0.760 | 0.434 |

### 3.12 Data Leakage Pathways

Three leakage traps hide in this dataset:

First, Biaya = Konsumsi × tariff. The ρ = 0.997 Spearman correlation and the Rp 1,444.71/kWh implied tariff nail it down. Any model predicting one should drop the other.

Second, Konsumsi derives from EER and Daya. For non-inverter ACs, consumption = (power × hours) / EER, with hours fixed at 8/day per the portal's footnote. Predicting EER? Drop Konsumsi.

Third, Rating derives from EER/CSPF. The r = 0.828 for non-inverter units confirms a threshold relationship. Predicting Rating? Drop EER/CSPF.

![Figure 6: Consumption and cost by star rating](outputs/figures/L7_konsumsi_biaya_per_rating.png)

**Figure 6.** Annual energy consumption (left) and electricity cost (right) by star rating.

### 3.13 Certification Bodies and Dates

![Figure 7: LSPro distribution](outputs/figures/L8_lspro_distribution.png)

**Figure 7.** Distribution of Product Certification Bodies (LSPro) for SIMEBTKE-registered AC products.

Of the 810 records carrying LSPro data, five bodies appear. PT Qualis Indonesia handles 307 (37.9%), PT TUV Rheinland Indonesia 265 (32.7%), and BBSPJIBBT 121 (14.9%). Private labs dominate—70.6% of certified records—which mirrors the outsourced testing model Indonesia uses for its labelling programme.

Certification dates exist for only 465 records. The trend climbs: 8 in 2021, 35 in 2022, 158 in 2024, 144 in 2025. Manufacturer uptake is growing, but with 75.82% of dates missing, the temporal picture stays incomplete.

---

## 4. Statistical Analysis and Machine Learning Results

### 4.1 Normality

Shapiro-Wilk tests came back non-normal for all five numeric variables (W < 0.90, p < 0.001 across the board). Q-Q plots confirmed heavy tails. From here on, every test is non-parametric.

### 4.2 Does EER/CSPF Differ Across Star Ratings?

Kruskal-Wallis says yes, loudly: H = 1,298.52, p < 0.001, η² = 0.68. That is a large effect. For non-inverter units alone, η² hits 0.81; for inverter, it drops to 0.19.

Post-hoc Mann-Whitney U tests with Bonferroni correction found significant gaps between every pair of ratings except 3 vs. 5 (p_adj = 0.354). The 4-vs-5 comparison is the headline: r_rb = −0.775. Four-star products post higher median EER/CSPF (11.64) than five-star ones (5.60). That sounds backwards, but it follows directly from the EER/CSPF conflation—5-star inverter units carry low CSPF values that sit below the EER values of 4-star units in the same column.

### 4.3 Inverter vs. Non-Inverter Efficiency

Mann-Whitney U = 552,956, p < 0.001, rank-biserial r = −0.384. Inverter median EER/CSPF (11.36) edges out non-inverter (10.14), but the moderate effect size means the distributions overlap substantially. Energy consumption also differs: inverter units run a median 2,708 kWh/year versus 2,117 for non-inverter (U = 484,385, p < 0.001, r = −0.212).

### 4.4 Rating × Type: A Structural Zero Problem

Chi-square = 666.32, df = 4, p < 0.001, Cramér's V = 0.591. That is a strong association. Standardised residuals pin the biggest deviations at Rating 5 (z = 15.22 for inverter, −10.56 for non-inverter) and Rating 2 (z = −10.87 for inverter, 7.54 for non-inverter). The structural zeros drive everything: no 5-star non-inverter, no 1-star inverter.

### 4.5 Partial Correlations: What Survives Control?

After residualising out Kapasitas, the Daya–EER/CSPF correlation drops to r = 0.004 (p = 0.86). Gone. Efficiency does not ride on power alone. The Daya–Kapasitas link stays strong at r = 0.731 after controlling for EER. And the Konsumsi–Biaya Spearman ρ of 0.974 reconfirms the derivation relationship one more time.

### 4.6 K-Means: Five Product Archetypes

![Figure 8: Elbow and silhouette scores](outputs/figures/M1_elbow_silhouette.png)

**Figure 8.** Elbow method (left) and silhouette score (right) for determining the optimal number of K-Means clusters.

![Figure 9: K-Means cluster visualisation](outputs/figures/M2_clusters.png)

**Figure 9.** K-Means cluster visualisation in Daya–Kapasitas (left) and Daya–EER/CSPF (right) space.

Silhouette scores peaked at k = 5 (0.519). **Table 8** lays out the profiles. Cluster 2 is the biggest (n = 825, 43.2%): small-capacity, high-efficiency units—your typical 1 PK inverter. Cluster 0 (n = 593, 31.1%) is its mirror image: small-capacity but low-efficiency, entry-level non-inverter stock. Clusters 1 and 3 split the large-capacity segment by efficiency: Cluster 1 runs efficient (EER/CSPF 11.35), Cluster 3 does not (3.79). A tiny Cluster 4 holds two outlier records.

**Table 8.** K-Means cluster profiles (k = 5)

| Cluster | Label | n | Daya median (W) | Kap. median (BTU/h) | EER/CSPF median | Rating mode |
|---|---|---|---|---|---|---|
| 0 | Sedang–Kurang Efisien | 593 | 762 | 9,000 | 3.73 | 2 |
| 1 | Besar–Efisien | 281 | 1,691 | 17,983 | 11.35 | 4 |
| 2 | Sedang–Efisien | 825 | 742 | 8,530 | 11.63 | 4 |
| 3 | Besar–Kurang Efisien | 208 | 1,681 | 18,062 | 3.79 | 2 |
| 4 | Outlier | 2 | 1,139 | 124,666 | 4.42 | — |

### 4.7 Can We Predict Star Rating Without Leakage?

We fed four classifiers Daya, Kapasitas, Tipe, and Kategori_PK—and nothing else. EER/CSPF, Konsumsi, and Biaya stayed out. Random Forest won (Table 9).

**Table 9.** Classification model comparison

| Model | CV F1 (mean ± SD) | Test Accuracy | Test F1 (weighted) |
|---|---|---|---|
| **Random Forest** | **0.594 ± 0.019** | **0.668** | **0.641** |
| KNN (k=7) | 0.560 ± 0.025 | 0.615 | 0.605 |
| Decision Tree | 0.508 ± 0.038 | 0.576 | 0.542 |
| Logistic Regression | 0.416 ± 0.009 | 0.552 | 0.424 |

![Figure 10: Confusion matrix](outputs/figures/M3_confusion_matrix.png)

**Figure 10.** Confusion matrix for the Random Forest classifier.

![Figure 11: Classification feature importance](outputs/figures/M4_clf_feature_importance.png)

**Figure 11.** Feature importance for the Random Forest classifier.

Sixty-seven percent accuracy sounds respectable given what the model did not see. It nails 4-star products (F1 = 0.77, the majority class) and struggles with the tails: 1-star F1 = 0.24, 5-star F1 = 0.34. Power and capacity alone cannot separate adjacent rating tiers. The variable that actually sets the rating—EER/CSPF—was off-limits by design.

### 4.8 Can We Predict EER/CSPF From External Specs?

Gradient Boosting eked out R² = 0.225 (Table 10). That is modest. Linear models posted R² near zero (−0.004), confirming the relationship is non-linear and that tree-based methods do better.

**Table 10.** Regression model comparison

| Model | CV R² (mean ± SD) | Test R² | Test RMSE | Test MAE |
|---|---|---|---|---|
| **Gradient Boosting** | **0.212 ± 0.044** | **0.225** | **3.75** | **2.99** |
| Random Forest | 0.220 ± 0.036 | 0.196 | 3.82 | 3.18 |
| Ridge | 0.028 ± 0.053 | −0.004 | 4.27 | 4.03 |
| Linear Regression | 0.024 ± 0.061 | −0.004 | 4.27 | 4.03 |

![Figure 12: Actual vs predicted EER/CSPF](outputs/figures/M5_regression_actual_vs_pred.png)

**Figure 12.** Actual vs. predicted EER/CSPF for the Gradient Boosting regressor.

About 22% of the variance in EER/CSPF is recoverable from the database's external variables. The rest depends on internal design choices—compressor geometry, refrigerant type, heat exchanger area, fan motor efficiency—that SIMEBTKE does not capture (ISO, 2013, 2017). The dataset tells you what the unit is rated for, not how it achieves it.

![Figure 13: Model comparison](outputs/figures/M7_model_comparison.png)

**Figure 13.** Performance comparison across all classification (left) and regression (right) models.

---

## 5. Conclusion

This paper dug into 1,923 AC records from the SIMEBTKE database—Indonesia's official registry of energy-labelled cooling products. Eight takeaways emerged.

First, the data need work. Certification dates are 75.82% missing. Brand names suffer from case inconsistency (Gree versus GREE—same company, two entries). Biaya Listrik ships with comma separators that break numeric parsing. And 541 records share registration numbers because manufacturers file multiple models under one certificate batch. None of these are fatal, but a cleaning pipeline must address them before modelling.

Second, the inverter–non-inverter split is stark. All 5-star units are inverter; all 1-star units are non-inverter. That is not a gradient—it is a wall. Inverter technology is the price of admission to the top tier, though most inverter products still land at 4 stars.

Third, the EER–Rating relationship reverses sign depending on AC type. Non-inverter: r = 0.828, positive and strong. Inverter: r = −0.444, negative. SIMEBTKE stores EER and CSPF in one column. Anyone who treats that column as a single metric will draw wrong conclusions.

Fourth, three leakage pathways run through the dataset. Biaya derives from Konsumsi (ρ = 0.997). Konsumsi derives from EER and Daya. Rating derives from EER/CSPF. Models that ignore these derivation chains will report inflated accuracy.

Fifth, the market is moderately concentrated. Five brands cover 28.7% of records, with wide variation in the inverter-to-non-inverter ratio. Policy interventions targeting specific manufacturers could reach a sizeable share of the market.

Sixth, K-Means pulled out five product archetypes—from small cheap non-inverter units to large efficient inverter ones. These clusters map onto real market segments and could guide targeted standards or incentive programmes.

Seventh, predicting star rating from power, capacity, type, and PK category—without touching EER/CSPF—yields 66.8% accuracy. That is enough to be useful as a screening tool but not enough to replace lab testing. The minority classes (1-star, 5-star) need richer features.

Eighth, EER/CSPF itself is hard to predict from external specs. Gradient Boosting explained 22.5% of variance. The rest hides inside the units—compressor design, refrigerant, heat exchanger geometry—that the database does not record. If SIMEBTKE wants to support data-driven efficiency prediction, it needs to capture those internal variables.

The path forward runs through better data, not just better models. The SIMEBTKE programme has the infrastructure. What it needs is richer technical fields, tighter validation on entry, and consistent naming. With those improvements, the database could move from a certification ledger to a genuine analytical substrate for energy policy in Indonesia.

---

## References

1. BPS (Badan Pusat Statistik). (2023). *Statistik Konsumsi Energi Rumah Tangga 2023*. Jakarta: BPS-Statistics Indonesia.

2. Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32. https://doi.org/10.1023/A:1010933404324

3. Cover, T., & Hart, P. (1967). Nearest neighbor pattern classification. *IEEE Transactions on Information Theory*, 13(1), 21–27. https://doi.org/10.1109/TIT.1967.1053964

4. Daikin Industries, Ltd. (2023). *Annual Report 2023: Towards a Carbon-Neutral Future*. Osaka: Daikin Industries.

5. DJEBTKE (Direktorat Jenderal Energi Baru, Terbarukan dan Konservasi Energi). (2024). *Website Produk Berlabel Hemat Energi*. Retrieved from https://simebtke.esdm.go.id/sinergi/skem-label/konsumen/pengondisi-udara-ac

6. Field, A. (2018). *Discovering Statistics Using IBM SPSS Statistics* (5th ed.). London: SAGE Publications.

7. Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *Annals of Statistics*, 29(5), 1189–1232. https://doi.org/10.1214/aos/1013203451

8. Government of Indonesia. (2009). *Government Regulation No. 70 of 2009 on Energy Conservation*. Jakarta: State Secretariat.

9. Government of Indonesia. (2014). *Government Regulation No. 79 of 2014 on National Energy Policy*. Jakarta: State Secretariat.

10. Harris, C. R., Millman, K. J., van der Walt, S. J., et al. (2020). Array programming with NumPy. *Nature*, 585(7825), 357–362. https://doi.org/10.1038/s41586-020-2649-2

11. Hu, S., Yan, D., Guo, S., Liu, Y., Qiao, M., & Jiang, Y. (2020). Analysis of the air-conditioning energy consumption and cooling demand of residential buildings in China. *Energy and Buildings*, 224, 110240. https://doi.org/10.1016/j.enbuild.2020.110240

12. Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. *Computing in Science & Engineering*, 9(3), 90–95. https://doi.org/10.1109/MCSE.2007.55

13. IEA (International Energy Agency). (2018). *The Future of Cooling: Opportunities for Energy-Efficient Air Conditioning*. Paris: IEA. https://doi.org/10.1787/9789264301995-en

14. ISO (International Organization for Standardization). (2013). *ISO 16358-1:2013 Air-cooled air conditioners and air-to-air heat pumps—Testing and calculating methods for seasonal performance factors—Part 1: Cooling seasonal performance factor*. Geneva: ISO.

15. ISO (International Organization for Standardization). (2017). *ISO 5151:2017 Non-ducted air conditioners—Testing and rating for performance*. Geneva: ISO.

16. Kuhn, M., & Johnson, K. (2013). *Applied Predictive Modeling*. New York: Springer. https://doi.org/10.1007/978-1-4614-6849-3

17. McKinney, W. (2017). *Python for Data Analysis: Data Wrangling with Pandas, NumPy, and IPython* (2nd ed.). Sebastopol: O'Reilly Media.

18. Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.

19. Pérez-Lombard, L., Ortiz, J., & Pout, C. (2008). A review on buildings energy consumption information. *Energy and Buildings*, 40(3), 394–398. https://doi.org/10.1016/j.enbuild.2007.03.007

20. PLN (Perusahaan Listrik Negara). (2023). *Tarif Tenaga Listrik Non-Subsidi 2023*. Jakarta: PT PLN (Persero).

21. Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the interpretation and validation of cluster analysis. *Journal of Computational and Applied Mathematics*, 20, 53–65. https://doi.org/10.1016/0377-0427(87)90125-7

22. Saidur, R., Ahamed, J. U., & Masjuki, H. H. (2009). Energy, exergy and economic analysis of industrial boilers. *Energy Policy*, 37(5), 1760–1768. https://doi.org/10.1016/j.enpol.2008.12.024

23. Samsung Electronics Co., Ltd. (2024). *Sustainability Report 2024*. Suwon: Samsung Electronics.

24. Santamouris, M., & Kolokotsa, D. (2013). Passive cooling dissipation techniques for buildings and other structures: The state of the art. *Energy and Buildings*, 57, 74–94. https://doi.org/10.1016/j.enbuild.2012.11.002

25. Tukey, J. W. (1977). *Exploratory Data Analysis*. Reading, MA: Addison-Wesley.

26. Ürge-Vorsatz, D., Petrichenko, K., Antosik, M., et al. (2015). Measuring the co-benefits of climate change mitigation: Making it matter. *Climate Change*, 5(4), 399–402.

27. Virtanen, P., Gommers, R., Oliphant, T. E., et al. (2020). SciPy 1.0: Fundamental algorithms for scientific computing in Python. *Nature Methods*, 17(3), 261–272. https://doi.org/10.1038/s41592-019-0686-2

28. Waskom, M. L. (2021). Seaborn: Statistical data visualization. *Journal of Open Source Software*, 6(60), 3021. https://doi.org/10.21105/joss.03021

29. Wickham, H. (2014). Tidy data. *Journal of Statistical Software*, 59(10), 1–23. https://doi.org/10.18637/jss.v059.i10

30. Wilkinson, M. D., Dumontier, M., Aalbersberg, I. J., et al. (2016). The FAIR Guiding Principles for scientific data management and stewardship. *Scientific Data*, 3, 160018. https://doi.org/10.1038/sdata.2016.18

---

*Authors should verify all references against the Scopus database (www.scopus.com) for indexing status and citation accuracy. Government regulations and ISO standards are primary sources; DOI links reflect the best-known information at the time of writing.*
