Available online at website: https://jurnal.iaii.or.id/index.php/RESTI
 	JURNAL RESTI
(Rekayasa Sistem dan Teknologi Informasi)
	    Vol. x No. x (2026) xx - xx	                                       e-ISSN: 2580-0760


Data-Driven Analysis of Air Conditioner Energy Efficiency in Indonesia from SIMEBTKE

Fajar Husain Asy'ari1*
1Department, Faculty, Institution, City, Country
*Email corresponding author

Abstract
Indonesia's household electricity demand keeps rising, with air conditioning units as a primary driver. The Sistem Informasi Manajemen Efisiensi dan Konservasi Energi (SIMEBTKE), operated by the Directorate General of New, Renewable Energy and Energy Conservation under the Ministry of Energy and Mineral Resources, maintains a public database of AC products that have received the Sertifikat Hemat Energi. We retrieved 1,923 records from this database, comprising 623 inverter and 1,300 non-inverter models, and conducted exploratory data analysis, statistical hypothesis testing, and machine learning modelling with leakage prevention. The data show that 55.7% of products sit at 4-star efficiency while only 8.6% reach the 5-star tier. Every 5-star unit runs on inverter technology and every 1-star unit does not. For non-inverter ACs, EER correlates strongly with star rating (r = 0.828), but the relationship reverses for inverter units (r = -0.444) because the database stores CSPF and EER in the same field. Annual electricity cost is a near-perfect linear function of energy consumption (rho = 0.997), creating a leakage trap. K-Means clustering identified five product archetypes tracking market segmentation from entry-level to premium. A Random Forest classifier predicted star rating at 66.8% accuracy using only power, capacity, type, and PK category, excluding EER/CSPF. Regression of EER/CSPF onto the same set explained 22.5% of variance, suggesting the database's external specifications lack the detail needed to predict internal efficiency. These results expose the potential and limits of using government labelling data for efficiency analysis.

Keywords: air conditioner; energy efficiency; EER; CSPF; SIMEBTKE; machine learning; clustering; classification; Indonesia

How to Cite: [Caption completed by the editor]
Permalink/DOI: [Caption completed by the editor]

Received: [Caption completed by the editor]
Accepted: [Caption completed by the editor]
Available Online: [Caption completed by the editor]	
This is an open-access article under the CC BY 4.0 License 
Published by Ikatan Ahli Informatika Indonesia


1. Introduction

Indonesia's electricity demand keeps climbing. Urbanisation, higher disposable incomes, and broader grid access all push consumption upward [1], [2]. Within the household sector, air conditioning units stand out as a major load driver, unavoidable in a tropical climate where cooling is needed year-round [3], [4]. The International Energy Agency projects that global cooling energy demand will triple by 2050, and Southeast Asia will absorb a disproportionate share of that surge [2]. Indonesian household AC penetration rose from below 10% in 2010 to roughly 20-25% by 2023 [5], and the curve is steepening.

The government's response has been regulatory. Government Regulation No. 70 of 2009 on Energy Conservation set efficiency floors for household equipment [6]. Five years later, Regulation No. 79 of 2014 on the National Energy Policy tightened those targets and tied them to renewable energy goals [7]. At the centre of this apparatus sits the Sertifikat Hemat Energi (SHE), a certificate issued to products meeting minimum efficiency thresholds, displayed to consumers through a 1-to-5 star label. The Sistem Informasi Manajemen Efisiensi dan Konservasi Energi (SIMEBTKE) is the digital platform that publishes manufacturer-submitted data on every certified AC unit: brand, family, model, type, power input, cooling capacity, efficiency value, star rating, estimated annual energy consumption, estimated annual electricity cost, registration number, certification dates, and the testing body. The portal is live and publicly queryable.

Despite the availability of this dataset, it has received little academic attention. Most published work on AC efficiency falls into two camps: laboratory performance testing against ISO standards [8], [9] and macro-level consumption modelling [10], [11]. What is missing is a data-driven, bottom-up look at what a national labelling database actually contains, its structure, its quirks, its patterns, and its limitations as a modelling substrate. Exploratory data analysis [12] is the natural starting point. It surfaces data quality problems before they contaminate downstream models [13], [14], flags distributional oddities and outliers, exposes correlations that might indicate data leakage [15], and generates hypotheses worth testing formally [16]. For energy labelling data, one specific risk looms large: derived variables. If annual electricity cost is just consumption multiplied by a fixed tariff, including both as predictors inflates model performance artificially.

This study addresses that gap. The objectives are to assess SIMEBTKE data quality including missingness, duplication, and format inconsistencies; to characterise how efficiency variables distribute across AC types and brands; to flag leakage pathways; to run statistical tests on observed patterns; and to build and evaluate machine learning models covering clustering, classification, and regression with explicit leakage safeguards. The remainder of this paper is organised into four sections. Section 2 covers data and methods. Section 3 reports EDA findings. Section 4 presents statistical and machine learning results. Section 5 wraps up with conclusions.


2. Methods

2.1 Data Source and Variables

Data came from the public SIMEBTKE consumer portal [17]. The site splits AC products into inverter and non-inverter tabs, each populated through server-side AJAX calls with Bootstrap Table pagination. A custom Python script loops through paginated HTTP requests, sending limit, offset, page, and search parameters, and pulls every record. The final count was 623 inverter and 1,300 non-inverter units, totalling 1,923 records. Raw JSON and CSV copies were preserved untouched to ensure reproducibility [18].

Fifteen fields come with each record, grouped into four categories. Identity variables include record number, brand (Merek), family (Famili), model, type (Tipe: inverter or non-inverter), registration number (No. Registrasi/No. SHE), and certification body (LSPro). Technical specification variables include power input in watts (Daya), cooling capacity in BTU per hour (Kapasitas Pendinginan), and efficiency value (Nilai Efisiensi: EER for non-inverter, CSPF for inverter). Energy performance variables include star rating from 1 to 5 (Rating Bintang), annual energy consumption in kWh (Konsumsi Energi Tahunan), and annual electricity cost in Rupiah (Biaya Listrik Tahunan). Certification temporal variables include the SHE issue date and expiry date.

2.2 Exploratory Data Analysis Protocol

The EDA followed a 13-step pipeline. Steps A through D covered data loading with string dtype to preserve raw formatting, shape inspection, column type examination, and the first 10 rows. Step E scanned for missing values including empty strings, null, NA, dashes, and similar placeholders [13]. Step F performed three duplicate checks: full-row, row minus the record number, and registration number. Step G computed frequency tables for every categorical column. Step H tested whether each numeric field parsed cleanly and looked for commas, currency prefixes, or unit suffixes. Step I validated dates against ISO 8601 format. Step J applied interquartile range outlier detection [12] plus two cross-checks: whether EER approximates capacity divided by power, and whether the implied tariff approximates Rp 1,444 per kWh. Step K computed descriptive statistics with percentile breakdowns and skewness. Step L generated eight visualisations including histograms, boxplots, bar charts, scatter plots, and a correlation heatmap. Step M excluded machine learning, consistent with the principle that EDA precedes predictive modelling [12], [15].

2.3 Data Cleaning

A systematic cleaning pipeline was applied on a copy of the raw data, leaving the original untouched. Brand names were normalised by mapping case variants to their most frequent form, reducing 98 unique strings to 63 distinct brands. Six numeric columns were parsed to float or integer, with comma-based thousand separators stripped from the annual electricity cost field before conversion. Date columns were parsed to datetime objects, with one anomalous value of 0000-00-00 converted to null. Missing values in certification date and LSPro columns were retained rather than imputed, as these represent administrative data that cannot be reliably estimated from other variables. Duplicate registration numbers were flagged rather than removed, since each record represents a distinct physical product certified under a batch registration. Outliers were identified through both the IQR method and domain-specific thresholds, including a flag for multi-model registrations where power input exceeded 5,000 W or annual consumption exceeded 100,000 kWh, and a flag for suspicious electricity cost values of exactly Rp 0 or Rp 99,999,999.99.

2.4 Machine Learning Setup

Three tasks were formulated with leakage prevention built in. For clustering, K-Means was applied to standardised power input, cooling capacity, and efficiency value. The optimal k was determined by scanning k from 2 to 10 using the elbow method and silhouette score [19]. Cluster profiles were described by median feature values and modal rating and type.

For classification, the target was the 5-level star rating. Predictors included power input, cooling capacity, AC type, and PK category. The variables EER/CSPF, annual energy consumption, and annual electricity cost were deliberately excluded because they derive from or determine the rating. Four algorithms were evaluated: Random Forest [20], K-Nearest Neighbours [21], Decision Tree, and Logistic Regression. Evaluation used 5-fold stratified cross-validation with F1-weighted scoring and a 20% hold-out test set, reporting accuracy, F1-score, and confusion matrix.

For regression, the target was the continuous EER/CSPF value. The same leakage-safe predictor set was used, additionally excluding star rating. Four algorithms were evaluated: Random Forest, Ridge Regression, Gradient Boosting [22], and Linear Regression. Evaluation used 5-by-3 repeated cross-validation with R-squared scoring and a 20% hold-out test set, reporting R-squared, root mean squared error, and mean absolute error.

All numeric features were standardised using z-score normalisation. Categorical variables were one-hot encoded. The random seed was fixed at 42 for reproducibility. The analysis used Python 3 with pandas [14], numpy [23], matplotlib [24], seaborn [25], scipy [26], and scikit-learn [27].


3. Results and Discussions

3.1 Dataset Overview and Data Quality

The SIMEBTKE AC database contains 1,923 product records across 15 variables. Non-inverter ACs dominate at 67.6%, reflecting the historical prevalence of non-inverter technology in the Indonesian market. Three variables are heavily incomplete. Certification dates, both issue and expiry, are missing for 1,458 rows or 75.82%. LSPro is absent for 57.88%. Older entries likely predate the digital system, and some products were certified through paper-based channels before SIMEBTKE went live. The technical and energy columns are fully populated, which is encouraging for quantitative analysis.

No two rows are byte-for-byte identical, but 541 records share a registration number with at least one other row. These are not true duplicates. Manufacturers register entire model line-ups under a single certificate. Mitsubishi Heavy Industries, for instance, filed SRK13YYP-W3 and SRK18YYM-W3 under one number. Bestlife bundled over 60 variants under a single certificate. Each row is a distinct product; the shared number means one testing batch covered them all. Certificate-level analysis should aggregate by registration number rather than by row.

All numeric columns arrived as strings. Five of six parse cleanly with plain decimals. The electricity cost column ships with comma thousands separators, so direct parsing caught only 27 of 1,923 values. Stripping commas before conversion fixes it. Dates follow ISO 8601, with one rogue entry of 0000-00-00 that should be treated as null. After normalisation of brand names, 98 unique strings collapsed to 63 distinct brands, led by Gree at 214 records, followed by Panasonic, Daikin, LG, and Midea.

3.2 Outliers and Cross-Validation Checks

The IQR-based outlier method identified relatively low rates across all numeric variables, ranging from 0.3% for cooling capacity to 1.8% for annual electricity cost. A BEKO unit reports 1.16 W, physically impossible for a room AC and almost certainly a keystroke error. Ten records exceed 5,000 W, topping out at 20,400 W. These trace back to multi-model registrations where power figures represent combined unit loads. The same pattern explains the maximum annual consumption value of 5,040,796 kWh, about 2,200 times the median. Twenty-seven rows carry an electricity cost of exactly Rp 0.00, and the ceiling sits at Rp 99,999,999.99, a suspiciously round placeholder.

Two cross-checks held up well. Theoretical EER, computed as cooling capacity divided by power input, matched recorded EER for non-inverter units with a median absolute difference of 0.21. The gap widened to 3.96 for inverter units, which makes sense because CSPF uses seasonal weighting rather than a simple ratio [8]. The implied tariff, computed as annual cost divided by annual consumption, hovered at Rp 1,444.71 per kWh, dead-centre on PLN's non-subsidised residential rate [28]. That confirms the electricity cost field is derived from consumption via a fixed multiplier.

3.3 Efficiency Patterns and the Inverter Divide

[Figure 1: Histogram of numeric variables - outputs/figures/L1_histograms_numerik.png]
Figure 1. Distribution of numeric variables in the SIMEBTKE AC database

[Figure 2: Boxplot by AC type - outputs/figures/L2_boxplot_per_tipe.png]
Figure 2. Boxplot comparison of numeric variables between inverter and non-inverter AC types

The star rating histogram (Figure 1) piles up at 4 stars, with 1,071 products or 55.7%. Two-star units come second at 394, then 3-star at 184, 5-star at 166, and 1-star at 108. Most certified products clear a decent efficiency bar; few reach the top. The picture sharpens when rating meets type. Every single 5-star unit is an inverter. Every single 1-star unit is a non-inverter. Zero overlap at the extremes. Among inverter units, 70.95% hold 4 stars and 26.65% hold 5 stars. Non-inverter units cluster at 2 and 4 stars, with none reaching 5. Inverter technology looks necessary but not sufficient for the top tier. SIMEBTKE stores EER for non-inverter and CSPF for inverter units in a single field, as illustrated by the boxplot comparison in Figure 2. Non-inverter EER averages 8.16 with a median of 10.13, while inverter CSPF averages 10.09 with a median of 11.36. The distributions overlap, which means anyone working with this column must split by type first.

Pearson correlation analysis (Figure 3) revealed four patterns worth noting. Power input and electricity cost correlate at r = 0.649, as expected since bigger units consume more. Annual consumption and electricity cost hit r = 0.997 for non-inverter units, confirming the derived-variable relationship. The EER-to-Rating link is where things get strange: r = 0.828 for non-inverter, meaning the rating tracks EER directly, but r = -0.444 for inverter. That negative sign looks wrong until you realise that 4-star inverter units carry higher CSPF values than 5-star ones, presumably because the 5-star threshold for inverter ACs rests on a different criterion than simple CSPF magnitude.

3.4 Data Leakage Assessment

[Figure 3: Pearson correlation heatmap - outputs/figures/L6_correlation_heatmap.png]
Figure 3. Pearson correlation heatmap of numeric variables

[Figure 4: Consumption and cost by rating - outputs/figures/L7_konsumsi_biaya_per_rating.png]
Figure 4. Annual energy consumption (left) and electricity cost (right) by star rating

Three leakage pathways run through this dataset. First, electricity cost equals consumption multiplied by tariff. The near-perfect Spearman correlation of 0.997 and the consistent implied tariff of Rp 1,444.71 per kWh confirm it. Any model predicting one should drop the other. Second, annual consumption derives from EER and power input. For non-inverter ACs, consumption equals power input times operating hours divided by EER, with hours fixed at 8 per day per the portal's footnote. Predicting EER means dropping consumption. Third, star rating derives from EER or CSPF through a threshold function, confirmed by the r = 0.828 correlation for non-inverter units. Predicting rating means dropping EER/CSPF. Future predictive models must exclude these derived variables from the predictor set to avoid inflated performance estimates [15].

3.5 Statistical Hypothesis Testing

Shapiro-Wilk tests came back non-normal for all five numeric variables, with W below 0.90 and p below 0.001 across the board. From here on, every test was non-parametric. The Kruskal-Wallis test on EER/CSPF across the five star rating levels returned H = 1,298.52, p < 0.001, with eta-squared of 0.68, a large effect. For non-inverter units alone, eta-squared hit 0.81; for inverter, it dropped to 0.19. Post-hoc Mann-Whitney U tests with Bonferroni correction found significant gaps between every pair of ratings except 3 versus 5. The 4-versus-5 comparison is the headline: rank-biserial r = -0.775, meaning 4-star products post higher median EER/CSPF (11.64) than 5-star ones (5.60). That sounds backwards, but it follows directly from the EER and CSPF conflation in a single column.

A Mann-Whitney U test comparing inverter and non-inverter efficiency returned U = 552,956, p < 0.001, rank-biserial r = -0.384. Inverter median EER/CSPF (11.36) edges out non-inverter (10.14), but the moderate effect size means the distributions overlap substantially. The chi-square test on rating by type confirmed a strong association at chi-square = 666.32, p < 0.001, Cramer's V = 0.591. Standardised residuals pin the biggest deviations at Rating 5 with z = 15.22 for inverter and Rating 2 with z = -10.87 for inverter. The structural zeros drive everything: no 5-star non-inverter, no 1-star inverter. Partial correlation analysis after residualising out cooling capacity dropped the power-to-EER correlation to r = 0.004, p = 0.86. Efficiency does not ride on power alone. The power-to-capacity link stayed strong at r = 0.731 after controlling for EER.

3.6 K-Means Clustering

[Figure 5: Elbow and silhouette - outputs/figures/M1_elbow_silhouette.png]
Figure 5. Elbow method (left) and silhouette score (right) for determining optimal K-Means clusters

[Figure 6: K-Means clusters - outputs/figures/M2_clusters.png]
Figure 6. K-Means cluster visualisation in Daya-Kapasitas (left) and Daya-EER/CSPF (right) space

Silhouette scores peaked at k = 5 with a score of 0.519, as shown by the elbow and silhouette analysis in Figure 4. Table 1 lays out the profiles. Cluster 2 is the biggest at 825 records or 43.2%, representing small-capacity, high-efficiency units typical of 1 PK inverter ACs. Cluster 0 at 593 records or 31.1% is its mirror image: small-capacity but low-efficiency, entry-level non-inverter stock. Clusters 1 and 3 split the large-capacity segment by efficiency. Cluster 1 runs efficient with median EER/CSPF of 11.35, while Cluster 3 does not at 3.79. A tiny Cluster 4 holds two outlier records. These clusters map onto real market segments and could guide targeted standards or incentive programmes.

Table 1. K-Means cluster profiles (k = 5)
Cluster	Label	n	Daya median (W)	Kap. median (BTU/h)	EER/CSPF median	Rating mode
0	Sedang-Kurang Efisien	593	762	9,000	3.73	2
1	Besar-Efisien	281	1,691	17,983	11.35	4
2	Sedang-Efisien	825	742	8,530	11.63	4
3	Besar-Kurang Efisien	208	1,681	18,062	3.79	2
4	Outlier	2	1,139	124,666	4.42	-

3.7 Classification and Regression Results

[Figure 7: Confusion matrix - outputs/figures/M3_confusion_matrix.png]
Figure 7. Confusion matrix for the Random Forest classifier

[Figure 8: Actual vs predicted - outputs/figures/M5_regression_actual_vs_pred.png]
Figure 8. Actual vs. predicted EER/CSPF for the Gradient Boosting regressor

Four classifiers were fed power input, cooling capacity, AC type, and PK category, with EER/CSPF, consumption, and cost excluded. Random Forest achieved the best performance. Table 2 summarises the results.

Table 2. Classification model comparison
Model	CV F1 (mean plus or minus SD)	Test Accuracy	Test F1 (weighted)
Random Forest	0.594 plus or minus 0.019	0.668	0.641
KNN (k=7)	0.560 plus or minus 0.025	0.615	0.605
Decision Tree	0.508 plus or minus 0.038	0.576	0.542
Logistic Regression	0.416 plus or minus 0.009	0.552	0.424

Sixty-seven percent accuracy sounds respectable given what the model did not see. The confusion matrix in Figure 5 shows that it nails 4-star products at F1 = 0.77, the majority class, and struggles with the tails: 1-star F1 = 0.24, 5-star F1 = 0.34. Power and capacity alone cannot separate adjacent rating tiers. The variable that actually sets the rating, EER/CSPF, was off-limits by design.

For regression, Gradient Boosting achieved R-squared of 0.225 with RMSE of 3.75 and MAE of 2.99, as shown by the actual-versus-predicted scatter in Figure 6. Table 3 shows the comparison. Linear models posted R-squared near zero, confirming the relationship is non-linear and that tree-based methods do better. About 22% of the variance in EER/CSPF is recoverable from the database's external variables. The rest depends on internal design choices including compressor geometry, refrigerant type, heat exchanger area, and fan motor efficiency, none of which SIMEBTKE captures [8], [9].

Table 3. Regression model comparison
Model	CV R2 (mean plus or minus SD)	Test R2	Test RMSE	Test MAE
Gradient Boosting	0.212 plus or minus 0.044	0.225	3.75	2.99
Random Forest	0.220 plus or minus 0.036	0.196	3.82	3.18
Ridge	0.028 plus or minus 0.053	-0.004	4.27	4.03
Linear Regression	0.024 plus or minus 0.061	-0.004	4.27	4.03

3.8 Limitations and Future Work

Several limitations bear acknowledgement. The dataset contains 75.82% missing values in certification dates, which restricts temporal analysis to a quarter of the records. The SIMEBTKE database stores EER and CSPF in a single column despite their different calculation bases, which conflates two distinct metrics and can mislead any analysis that does not stratify by AC type. Multi-model registrations inflate outlier values for power and consumption, and while these were flagged, they were not removed, meaning aggregate statistics carry some contamination. The machine learning models operate on external specifications only; the database does not capture internal technical factors such as compressor type, refrigerant, or heat exchanger design, which limits predictive performance. Future work should pursue richer feature sets, possibly by linking SIMEBTKE records to manufacturer datasheets or laboratory test reports. Longitudinal analysis of certification trends, once more date data accumulates, could reveal whether the market is shifting toward higher efficiency tiers. Comparative studies with energy labelling databases from other Southeast Asian countries would contextualise Indonesia's position in the regional efficiency landscape.


4. Conclusions

This paper analysed 1,923 AC product records from the SIMEBTKE database to understand energy efficiency patterns in Indonesia. The data reveal a stark technological divide: every 5-star unit runs on inverter technology and every 1-star unit does not, with zero overlap at the extremes. The EER-to-rating correlation is strongly positive for non-inverter ACs at r = 0.828 but negative for inverter units at r = -0.444, a paradox rooted in the database storing EER and CSPF in a single field. Three data leakage pathways were identified and documented: electricity cost derives from consumption at rho = 0.997, consumption derives from EER and power input, and star rating derives from EER/CSPF through thresholding. K-Means clustering sorted products into five archetypes that track market segmentation from entry-level non-inverter units to premium inverter models. A Random Forest classifier predicted star rating at 66.8% accuracy using only power, capacity, type, and PK category, deliberately excluding leakage variables. Gradient Boosting explained 22.5% of EER/CSPF variance from external specifications, with the remaining variance attributable to internal technical factors the database does not capture. The market is moderately concentrated, with five brands covering 28.7% of records. These findings demonstrate that SIMEBTKE holds analytical value for energy policy research, but its current schema lacks the technical depth needed for robust efficiency prediction. If the database incorporated internal design parameters such as compressor type and refrigerant specification, and tightened its entry validation for consistency, it could evolve from a certification ledger into a genuine analytical substrate for data-driven energy policy in Indonesia.


Acknowledgements

The author declares no conflict of interest. This research received no specific grant from any funding agency in the public, commercial, or not-for-profit sectors.


References

[1] Government of Indonesia, "Government Regulation No. 79 of 2014 on National Energy Policy," Jakarta: State Secretariat, 2014.
[2] IEA (International Energy Agency), "The Future of Cooling: Opportunities for Energy-Efficient Air Conditioning," Paris: IEA, 2018. doi: 10.1787/9789264301995-en.
[3] L. Perez-Lombard, J. Ortiz, and C. Pout, "A review on buildings energy consumption information," Energy and Buildings, vol. 40, no. 3, pp. 394-398, 2008. doi: 10.1016/j.enbuild.2007.03.007.
[4] M. Santamouris and D. Kolokotsa, "Passive cooling dissipation techniques for buildings and other structures: The state of the art," Energy and Buildings, vol. 57, pp. 74-94, 2013. doi: 10.1016/j.enbuild.2012.11.002.
[5] BPS (Badan Pusat Statistik), "Statistik Konsumsi Energi Rumah Tangga 2023," Jakarta: BPS-Statistics Indonesia, 2023.
[6] Government of Indonesia, "Government Regulation No. 70 of 2009 on Energy Conservation," Jakarta: State Secretariat, 2009.
[7] Government of Indonesia, "Government Regulation No. 79 of 2014 on National Energy Policy," Jakarta: State Secretariat, 2014.
[8] ISO, "ISO 16358-1:2013 Air-cooled air conditioners and air-to-air heat pumps - Testing and calculating methods for seasonal performance factors - Part 1: Cooling seasonal performance factor," Geneva: ISO, 2013.
[9] ISO, "ISO 5151:2017 Non-ducted air conditioners - Testing and rating for performance," Geneva: ISO, 2017.
[10] D. Urge-Vorsatz, K. Petrichenko, M. Antosik, et al., "Measuring the co-benefits of climate change mitigation: Making it matter," Climate Change, vol. 5, no. 4, pp. 399-402, 2015.
[11] S. Hu, D. Yan, S. Guo, Y. Liu, M. Qiao, and Y. Jiang, "Analysis of the air-conditioning energy consumption and cooling demand of residential buildings in China," Energy and Buildings, vol. 224, p. 110240, 2020. doi: 10.1016/j.enbuild.2020.110240.
[12] J. W. Tukey, Exploratory Data Analysis. Reading, MA: Addison-Wesley, 1977.
[13] H. Wickham, "Tidy data," Journal of Statistical Software, vol. 59, no. 10, pp. 1-23, 2014. doi: 10.18637/jss.v059.i10.
[14] W. McKinney, Python for Data Analysis: Data Wrangling with Pandas, NumPy, and IPython, 2nd ed. Sebastopol: O'Reilly Media, 2017.
[15] M. Kuhn and K. Johnson, Applied Predictive Modeling. New York: Springer, 2013. doi: 10.1007/978-1-4614-6849-3.
[16] A. Field, Discovering Statistics Using IBM SPSS Statistics, 5th ed. London: SAGE Publications, 2018.
[17] DJEBTKE, "Website Produk Berlabel Hemat Energi," Directorate General of New, Renewable Energy and Energy Conservation, 2024. [Online]. Available: https://simebtke.esdm.go.id/sinergi/skem-label/konsumen/pengondisi-udara-ac.
[18] M. D. Wilkinson, M. Dumontier, I. J. Aalbersberg, et al., "The FAIR Guiding Principles for scientific data management and stewardship," Scientific Data, vol. 3, p. 160018, 2016. doi: 10.1038/sdata.2016.18.
[19] P. J. Rousseeuw, "Silhouettes: A graphical aid to the interpretation and validation of cluster analysis," Journal of Computational and Applied Mathematics, vol. 20, pp. 53-65, 1987. doi: 10.1016/0377-0427(87)90125-7.
[20] L. Breiman, "Random forests," Machine Learning, vol. 45, no. 1, pp. 5-32, 2001. doi: 10.1023/A:1010933404324.
[21] T. Cover and P. Hart, "Nearest neighbor pattern classification," IEEE Transactions on Information Theory, vol. 13, no. 1, pp. 21-27, 1967. doi: 10.1109/TIT.1967.1053964.
[22] J. H. Friedman, "Greedy function approximation: A gradient boosting machine," Annals of Statistics, vol. 29, no. 5, pp. 1189-1232, 2001. doi: 10.1214/aos/1013203451.
[23] C. R. Harris, K. J. Millman, S. J. van der Walt, et al., "Array programming with NumPy," Nature, vol. 585, no. 7825, pp. 357-362, 2020. doi: 10.1038/s41586-020-2649-2.
[24] J. D. Hunter, "Matplotlib: A 2D graphics environment," Computing in Science & Engineering, vol. 9, no. 3, pp. 90-95, 2007. doi: 10.1109/MCSE.2007.55.
[25] M. L. Waskom, "Seaborn: Statistical data visualization," Journal of Open Source Software, vol. 6, no. 60, p. 3021, 2021. doi: 10.21105/joss.03021.
[26] P. Virtanen, R. Gommers, T. E. Oliphant, et al., "SciPy 1.0: Fundamental algorithms for scientific computing in Python," Nature Methods, vol. 17, no. 3, pp. 261-272, 2020. doi: 10.1038/s41592-019-0686-2.
[27] F. Pedregosa, G. Varoquaux, A. Gramfort, et al., "Scikit-learn: Machine learning in Python," Journal of Machine Learning Research, vol. 12, pp. 2825-2830, 2011.
[28] PLN (Perusahaan Listrik Negara), "Tarif Tenaga Listrik Non-Subsidi 2023," Jakarta: PT PLN (Persero), 2023.
[29] R. Saidur, J. U. Ahamed, and H. H. Masjuki, "Energy, exergy and economic analysis of industrial boilers," Energy Policy, vol. 37, no. 5, pp. 1760-1768, 2009. doi: 10.1016/j.enpol.2008.12.024.
[30] Daikin Industries, Ltd., "Annual Report 2023: Towards a Carbon-Neutral Future," Osaka: Daikin Industries, 2023.
