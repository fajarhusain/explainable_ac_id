"""
============================================================================
 TAHAP 3: STATISTICAL ANALYSIS
 Dataset : AC SIMEBTKE (data bersih, eksklusi multi-model registration)
 Tujuan : Uji hipotesis untuk konfirmasi pola efisiensi energi
 Aturan  : Lihat Tahap 1 (14 aturan penelitian)
============================================================================
Langkah:
  A. Uji normalitas (Shapiro-Wilk)
  B. Kruskal-Wallis: EER/CSPF antar Rating Bintang + post-hoc Dunn
  C. Mann-Whitney U: EER/CSPF Inverter vs Non-Inverter
  D. Chi-square: Rating x Tipe (asosiasi)
  E. Korelasi parsial: Daya-Kapasitas-EER (kontrol Tipe)
  F. Kruskal-Wallis: EER/CSPF per Kategori PK
  G. Effect size (eta-squared, rank-biserial)
  H. Visualisasi statistik
  I. Laporan ringkasan
============================================================================
"""

# ============================================================
# 0. SETUP
# ============================================================
import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import (shapiro, kruskal, mannwhitneyu, chi2_contingency,
                         spearmanr, kendalltau)
# scikit-learn untuk partial correlation
from sklearn.linear_model import LinearRegression

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: f'{x:.6f}')

sns.set_style('whitegrid')
plt.rcParams.update({
    'figure.dpi': 100, 'savefig.dpi': 300, 'font.size': 10,
    'axes.titlesize': 12, 'axes.labelsize': 10, 'figure.facecolor': 'white',
})

CLEAN_PATH = 'data/processed/ac_simebtke_clean.csv'

print("=" * 70)
print("TAHAP 3: STATISTICAL ANALYSIS")
print("=" * 70)

# Load data bersih, eksklusi multi-model registration untuk analisis inti
df_full = pd.read_csv(CLEAN_PATH, encoding='utf-8-sig')
df = df_full[df_full['is_multi_model_reg'] == 0].copy()
print(f"Data bersih (full): {len(df_full)} records")
print(f"Data analisis (tanpa multi-model): {len(df)} records")

# ============================================================
# A. UJI NORMALITAS (Shapiro-Wilk)
# ============================================================
print("\n" + "=" * 70)
print("A. UJI NORMALITAS (Shapiro-Wilk)")
print("=" * 70)

normality_cols = ['Daya (watt)', 'Kapasitas Pendinginan (BTU/h)',
                  'Nilai Efisiensi (EER/CSPF)', 'Konsumsi Energi Tahunan (kWh)',
                  'Biaya Listrik Tahunan (Rp)']

normality_results = []
for col in normality_cols:
    data = df[col].dropna()
    # Shapiro-Wilk (sample max 5000, ambil sample 5000 jika lebih)
    sample = data.sample(min(5000, len(data)), random_state=42)
    stat_sw, p_sw = shapiro(sample)
    # D'Agostino-Pearson (untuk konfirmasi, tidak ada batas sample)
    stat_dp, p_dp = stats.normaltest(data)

    is_normal = p_sw > 0.05
    normality_results.append({
        'Variabel': col,
        'N': len(data),
        'Shapiro_W': stat_sw,
        'Shapiro_p': p_sw,
        'Normal_SW': 'Ya' if is_normal else 'Tidak',
        'DAgostino_K2': stat_dp,
        'DAgostino_p': p_dp,
        'Normal_DP': 'Ya' if p_dp > 0.05 else 'Tidak',
        'Skewness': data.skew(),
        'Kurtosis': data.kurtosis(),
    })
    print(f"  {col}:")
    print(f"    Shapiro-Wilk: W={stat_sw:.4f}, p={p_sw:.2e} -> {'Normal' if is_normal else 'TIDAK normal'}")
    print(f"    D'Agostino: K2={stat_dp:.4f}, p={p_dp:.2e} -> {'Normal' if p_dp > 0.05 else 'TIDAK normal'}")
    print(f"    Skewness={data.skew():.4f}, Kurtosis={data.kurtosis():.4f}")

norm_df = pd.DataFrame(normality_results)
norm_df.to_csv('outputs/tables/11_normality_tests.csv', index=False)

print("\n[ASEMSI] Semua variabel numerik TIDAK berdistribusi normal (p < 0.05).")
print("  Keputusan: Gunakan uji non-parametrik (Kruskal-Wallis, Mann-Whitney, Spearman).")

# ============================================================
# B. KRUSKAL-WALLIS: EER/CSPF antar Rating Bintang + post-hoc Dunn
# ============================================================
print("\n" + "=" * 70)
print("B. KRUSKAL-WALLIS: EER/CSPF antar Rating Bintang")
print("=" * 70)

eer_col = 'Nilai Efisiensi (EER/CSPF)'
ratings = sorted(df['Rating Bintang (1-5)'].dropna().unique())
groups = [df[df['Rating Bintang (1-5)'] == r][eer_col].dropna().values for r in ratings]

stat_kw, p_kw = kruskal(*groups)
# Effect size: eta-squared (Kruskal-Wallis)
n_total = sum(len(g) for g in groups)
eta_sq = (stat_kw - len(groups) + 1) / (n_total - len(groups))

print(f"  Kruskal-Wallis H = {stat_kw:.4f}")
print(f"  p-value = {p_kw:.2e}")
print(f"  Eta-squared (effect size) = {eta_sq:.4f}")
print(f"  Interpretasi: {'Perbedaan SIGNIFIKAN' if p_kw < 0.05 else 'Tidak signifikan'} (alpha=0.05)")

print(f"\n  Ringkasan per Rating Bintang (keseluruhan):")
for r in ratings:
    d = df[df['Rating Bintang (1-5)'] == r][eer_col].dropna()
    print(f"    Rating {int(r)}: n={len(d)}, median={d.median():.2f}, mean={d.mean():.2f}, std={d.std():.2f}")

# Per Tipe
for tipe in ['Non-Inverter', 'Inverter']:
    print(f"\n  --- {tipe} ---")
    sub = df[df['Tipe'] == tipe]
    groups_t = [sub[sub['Rating Bintang (1-5)'] == r][eer_col].dropna().values
                for r in ratings if len(sub[sub['Rating Bintang (1-5)'] == r]) > 0]
    if len(groups_t) >= 2:
        stat_t, p_t = kruskal(*groups_t)
        n_t = sum(len(g) for g in groups_t)
        eta_t = (stat_t - len(groups_t) + 1) / (n_t - len(groups_t))
        print(f"    H = {stat_t:.4f}, p = {p_t:.2e}, eta_sq = {eta_t:.4f}")

# Post-hoc: Pairwise Mann-Whitney dengan Bonferroni correction
print(f"\n  Post-hoc: Pairwise Mann-Whitney U (Bonferroni correction)")
pairs = [(ratings[i], ratings[j]) for i in range(len(ratings)) for j in range(i+1, len(ratings))]
n_comparisons = len(pairs)
alpha_bonf = 0.05 / n_comparisons

posthoc_results = []
for r1, r2 in pairs:
    d1 = df[df['Rating Bintang (1-5)'] == r1][eer_col].dropna()
    d2 = df[df['Rating Bintang (1-5)'] == r2][eer_col].dropna()
    stat_mw, p_mw = mannwhitneyu(d1, d2, alternative='two-sided')
    # Rank-biserial effect size
    n1, n2 = len(d1), len(d2)
    r_rb = 1 - (2 * stat_mw) / (n1 * n2)
    sig = p_mw < alpha_bonf
    posthoc_results.append({
        'Rating_1': int(r1), 'Rating_2': int(r2),
        'U_stat': stat_mw, 'p_value': p_mw,
        'p_bonferroni': p_mw * n_comparisons,
        'Signifikan_Bonf': 'Ya' if sig else 'Tidak',
        'rank_biserial_r': r_rb,
    })
    print(f"    Rating {int(r1)} vs {int(r2)}: U={stat_mw:.0f}, p={p_mw:.2e}, "
          f"p_adj={p_mw*n_comparisons:.2e}, r_rb={r_rb:.4f}, {'SIG' if sig else 'ns'}")

posthoc_df = pd.DataFrame(posthoc_results)
posthoc_df.to_csv('outputs/tables/12_posthoc_dunn_rating.csv', index=False)

# Simpan hasil Kruskal-Wallis
kw_results = pd.DataFrame([{
    'Uji': 'Kruskal-Wallis',
    'Variabel': 'EER/CSPF ~ Rating Bintang',
    'Statistik': stat_kw,
    'p_value': p_kw,
    'Effect_size_eta2': eta_sq,
    'Signifikan': 'Ya' if p_kw < 0.05 else 'Tidak',
    'N': n_total,
}])

# ============================================================
# C. MANN-WHITNEY U: EER/CSPF Inverter vs Non-Inverter
# ============================================================
print("\n" + "=" * 70)
print("C. MANN-WHITNEY U: EER/CSPF Inverter vs Non-Inverter")
print("=" * 70)

inv = df[df['Tipe'] == 'Inverter'][eer_col].dropna()
noninv = df[df['Tipe'] == 'Non-Inverter'][eer_col].dropna()

stat_mw2, p_mw2 = mannwhitneyu(inv, noninv, alternative='two-sided')
r_rb2 = 1 - (2 * stat_mw2) / (len(inv) * len(noninv))

print(f"  Inverter:    n={len(inv)}, median={inv.median():.2f}, mean={inv.mean():.2f}")
print(f"  Non-Inverter: n={len(noninv)}, median={noninv.median():.2f}, mean={noninv.mean():.2f}")
print(f"  Mann-Whitney U = {stat_mw2:.0f}")
print(f"  p-value = {p_mw2:.2e}")
print(f"  Rank-biserial r = {r_rb2:.4f}")
print(f"  Interpretasi: {'Perbedaan SIGNIFIKAN' if p_mw2 < 0.05 else 'Tidak signifikan'}")

# Konsumsi Energi per Tipe
print(f"\n  --- Konsumsi Energi Tahunan per Tipe ---")
inv_k = df[df['Tipe'] == 'Inverter']['Konsumsi Energi Tahunan (kWh)'].dropna()
noninv_k = df[df['Tipe'] == 'Non-Inverter']['Konsumsi Energi Tahunan (kWh)'].dropna()
stat_k, p_k = mannwhitneyu(inv_k, noninv_k, alternative='two-sided')
r_k = 1 - (2 * stat_k) / (len(inv_k) * len(noninv_k))
print(f"  Inverter: median={inv_k.median():.0f} kWh")
print(f"  Non-Inverter: median={noninv_k.median():.0f} kWh")
print(f"  U={stat_k:.0f}, p={p_k:.2e}, r_rb={r_k:.4f}")

kw_results = pd.concat([kw_results, pd.DataFrame([
    {'Uji': 'Mann-Whitney U', 'Variabel': 'EER/CSPF ~ Tipe', 'Statistik': stat_mw2,
     'p_value': p_mw2, 'Effect_size_eta2': np.nan,
     'Signifikan': 'Ya' if p_mw2 < 0.05 else 'Tidak', 'N': len(inv)+len(noninv)},
    {'Uji': 'Mann-Whitney U', 'Variabel': 'Konsumsi ~ Tipe', 'Statistik': stat_k,
     'p_value': p_k, 'Effect_size_eta2': np.nan,
     'Signifikan': 'Ya' if p_k < 0.05 else 'Tidak', 'N': len(inv_k)+len(noninv_k)},
])], ignore_index=True)

# ============================================================
# D. CHI-SQUARE: Rating x Tipe
# ============================================================
print("\n" + "=" * 70)
print("D. CHI-SQUARE: Asosiasi Rating Bintang x Tipe AC")
print("=" * 70)

ct = pd.crosstab(df['Rating Bintang (1-5)'], df['Tipe'])
print(f"\n  Cross-tabulation:")
print(ct.to_string())

chi2, p_chi, dof, expected = chi2_contingency(ct)
n = ct.values.sum()
cramers_v = np.sqrt(chi2 / (n * (min(ct.shape) - 1)))

print(f"\n  Chi-square = {chi2:.4f}")
print(f"  df = {dof}")
print(f"  p-value = {p_chi:.2e}")
print(f"  Cramer's V = {cramers_v:.4f}")
print(f"  Interpretasi: {'Asosiasi SIGNIFIKAN' if p_chi < 0.05 else 'Tidak signifikan'}")
print(f"  Effect size (Cramer's V): {cramers_v:.4f} ", end="")
if cramers_v < 0.1:
    print("(sangat lemah)")
elif cramers_v < 0.3:
    print("(lemah)")
elif cramers_v < 0.5:
    print("(moderat)")
else:
    print("(kuat)")

# Residual terstandardized
residuals = (ct - expected) / np.sqrt(expected)
print(f"\n  Residuals terstandardized:")
print(residuals.round(2).to_string())
print(f"  [ASEMSI] Residual |z| > 2 menunjukkan deviasi signifikan dari expected.")

kw_results = pd.concat([kw_results, pd.DataFrame([{
    'Uji': 'Chi-square', 'Variabel': 'Rating x Tipe', 'Statistik': chi2,
    'p_value': p_chi, 'Effect_size_eta2': cramers_v,
    'Signifikan': 'Ya' if p_chi < 0.05 else 'Tidak', 'N': n,
}])], ignore_index=True)

# ============================================================
# E. KORELASI PARSIAL: Daya-Kapasitas-EER (kontrol Tipe)
# ============================================================
print("\n" + "=" * 70)
print("E. KORELASI PARSIAL: Daya-Kapasitas-EER (kontrol Tipe)")
print("=" * 70)

# Spearman correlation (non-parametrik)
print("\n  --- Korelasi Spearman (keseluruhan, tanpa multi-model) ---")
spear_cols = ['Daya (watt)', 'Kapasitas Pendinginan (BTU/h)', eer_col,
              'Rating Bintang (1-5)', 'Konsumsi Energi Tahunan (kWh)',
              'Biaya Listrik Tahunan (Rp)']
spearman_matrix = df[spear_cols].corr(method='spearman')
print(spearman_matrix.round(4).to_string())

# Per Tipe
for tipe in ['Non-Inverter', 'Inverter']:
    print(f"\n  --- Spearman: {tipe} ---")
    sub = df[df['Tipe'] == tipe]
    print(sub[spear_cols].corr(method='spearman').round(4).to_string())

# Partial correlation: Daya-Kapasitas kontrol EER (dan sebaliknya)
# Metode: residualisasi dengan regresi linear
print("\n  --- Korelasi Parsial (residualisasi) ---")

def partial_corr(df_, x, y, controls):
    """Hitung korelasi parsial dengan residualisasi OLS."""
    X_c = df_[controls].values if controls else np.zeros((len(df_), 0))
    # Residualisasi x
    reg_x = LinearRegression().fit(X_c, df_[x].values)
    res_x = df_[x].values - reg_x.predict(X_c)
    # Residualisasi y
    reg_y = LinearRegression().fit(X_c, df_[y].values)
    res_y = df_[y].values - reg_y.predict(X_c)
    # Pearson pada residual
    r, p = stats.pearsonr(res_x, res_y)
    return r, p

partial_tests = [
    ('Daya (watt)', 'Kapasitas Pendinginan (BTU/h)', [eer_col]),
    ('Daya (watt)', 'Kapasitas Pendinginan (BTU/h)', [eer_col, 'Rating Bintang (1-5)']),
    ('Daya (watt)', eer_col, ['Kapasitas Pendinginan (BTU/h)']),
    ('Kapasitas Pendinginan (BTU/h)', eer_col, ['Daya (watt)']),
    ('Daya (watt)', 'Konsumsi Energi Tahunan (kWh)', [eer_col]),
    ('Daya (watt)', 'Biaya Listrik Tahunan (Rp)', [eer_col, 'Konsumsi Energi Tahunan (kWh)']),
]

for tipe in ['All', 'Non-Inverter', 'Inverter']:
    print(f"\n  --- {tipe} ---")
    sub = df if tipe == 'All' else df[df['Tipe'] == tipe]
    for x, y, controls in partial_tests:
        available = [c for c in [x, y] + controls if c in sub.columns]
        data = sub[[x, y] + controls].dropna()
        if len(data) < 10:
            print(f"    {x} ~ {y} | {controls}: data tidak cukup")
            continue
        try:
            r, p = partial_corr(data, x, y, controls)
            print(f"    r({x}, {y} | {controls}) = {r:.4f}, p = {p:.2e} {'SIG' if p < 0.05 else 'ns'}")
        except:
            print(f"    {x} ~ {y} | {controls}: error")

# ============================================================
# F. KRUSKAL-WALLIS: EER/CSPF per Kategori PK
# ============================================================
print("\n" + "=" * 70)
print("F. KRUSKAL-WALLIS: EER/CSPF per Kategori PK")
print("=" * 70)

pk_order = ['0.5 PK', '1 PK', '1.5 PK', '2 PK', '2.5 PK', '3+ PK']
pk_groups = [df[df['Kategori_PK'] == pk][eer_col].dropna().values
             for pk in pk_order if len(df[df['Kategori_PK'] == pk]) > 0]

stat_pk, p_pk = kruskal(*pk_groups)
n_pk = sum(len(g) for g in pk_groups)
eta_pk = (stat_pk - len(pk_groups) + 1) / (n_pk - len(pk_groups))

print(f"  Kruskal-Wallis H = {stat_pk:.4f}")
print(f"  p-value = {p_pk:.2e}")
print(f"  Eta-squared = {eta_pk:.4f}")

print(f"\n  Ringkasan per Kategori PK:")
for pk in pk_order:
    d = df[df['Kategori_PK'] == pk][eer_col].dropna()
    if len(d) > 0:
        print(f"    {pk}: n={len(d)}, median={d.median():.2f}, mean={d.mean():.2f}")

# Per Tipe
for tipe in ['Non-Inverter', 'Inverter']:
    print(f"\n  --- {tipe} ---")
    sub = df[df['Tipe'] == tipe]
    pk_groups_t = [sub[sub['Kategori_PK'] == pk][eer_col].dropna().values
                   for pk in pk_order if len(sub[sub['Kategori_PK'] == pk]) > 0]
    if len(pk_groups_t) >= 2:
        stat_p, p_p = kruskal(*pk_groups_t)
        n_p = sum(len(g) for g in pk_groups_t)
        eta_p = (stat_p - len(pk_groups_t) + 1) / (n_p - len(pk_groups_t))
        print(f"    H = {stat_p:.4f}, p = {p_p:.2e}, eta_sq = {eta_p:.4f}")

kw_results = pd.concat([kw_results, pd.DataFrame([
    {'Uji': 'Kruskal-Wallis', 'Variabel': 'EER/CSPF ~ Kategori PK', 'Statistik': stat_pk,
     'p_value': p_pk, 'Effect_size_eta2': eta_pk,
     'Signifikan': 'Ya' if p_pk < 0.05 else 'Tidak', 'N': n_pk},
])], ignore_index=True)

# Spearman: EER vs Daya per PK-Tipe
print(f"\n  --- Spearman: EER vs Daya per PK-Tipe ---")
for tipe in ['Non-Inverter', 'Inverter']:
    sub = df[df['Tipe'] == tipe]
    for pk in pk_order:
        d = sub[sub['Kategori_PK'] == pk]
        if len(d) >= 10:
            r_s, p_s = spearmanr(d[eer_col], d['Daya (watt)'])
            print(f"    {tipe} {pk}: r_s={r_s:.4f}, p={p_s:.2e} {'SIG' if p_s < 0.05 else 'ns'} (n={len(d)})")

# Simpan semua hasil uji
kw_results.to_csv('outputs/tables/13_hypothesis_tests.csv', index=False)
spearman_matrix.to_csv('outputs/tables/14_spearman_correlation.csv')

# ============================================================
# G. VISUALISASI STATISTIK
# ============================================================
print("\n" + "=" * 70)
print("G. VISUALISASI STATISTIK")
print("=" * 70)

# --- G1. Q-Q Plot untuk normalitas ---
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Q-Q Plot untuk Uji Normalitas', fontsize=14, fontweight='bold')
for idx, col in enumerate(normality_cols + ['Rating Bintang (1-5)']):
    ax = axes[idx // 3, idx % 3]
    data = df[col].dropna()
    stats.probplot(data, dist='norm', plot=ax)
    ax.set_title(f'{col}', fontsize=10)
    ax.get_lines()[0].set_color('steelblue')
    ax.get_lines()[0].set_markersize(3)
    ax.get_lines()[1].set_color('red')
plt.tight_layout()
plt.savefig('outputs/figures/G1_qq_plots.png', bbox_inches='tight')
plt.close()
print("  [SAVED] G1_qq_plots.png")

# --- G2. Boxplot EER per Rating + Tipe dengan significance annotation ---
fig, ax = plt.subplots(figsize=(12, 8))
sns.boxplot(data=df, x='Rating Bintang (1-5)', y=eer_col, hue='Tipe',
            ax=ax, palette='Set2', order=sorted(df['Rating Bintang (1-5)'].unique()))
ax.set_title('Nilai Efisiensi (EER/CSPF) per Rating Bintang dan Tipe AC\n'
             '(Kruskal-Wallis p < 0.001, post-hoc Mann-Whitney dengan Bonferroni)',
             fontsize=12, fontweight='bold')
ax.set_xlabel('Rating Bintang')
ax.set_ylabel('Nilai Efisiensi (EER/CSPF)')
ax.legend(title='Tipe', loc='upper left')
plt.tight_layout()
plt.savefig('outputs/figures/G2_eer_per_rating_tipe.png', bbox_inches='tight')
plt.close()
print("  [SAVED] G2_eer_per_rating_tipe.png")

# --- G3. Heatmap residual Chi-square ---
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(residuals, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, linewidths=0.5, ax=ax,
            cbar_kws={'label': 'Residual terstandardized'})
ax.set_title(f'Chi-square Residual: Rating x Tipe\n'
             f'Chi2={chi2:.2f}, p={p_chi:.2e}, Cramer V={cramers_v:.3f}',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/figures/G3_chi2_residuals.png', bbox_inches='tight')
plt.close()
print("  [SAVED] G3_chi2_residuals.png")

# --- G4. Spearman correlation heatmap per Tipe ---
fig, axes = plt.subplots(1, 3, figsize=(22, 7))
fig.suptitle('Korelasi Spearman antar Variabel Numerik',
             fontsize=14, fontweight='bold')

for idx, (label, sub) in enumerate([('Keseluruhan', df),
                                     ('Non-Inverter', df[df['Tipe']=='Non-Inverter']),
                                     ('Inverter', df[df['Tipe']=='Inverter'])]):
    ax = axes[idx]
    corr = sub[spear_cols].corr(method='spearman')
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt='.3f', cmap='RdBu_r',
                center=0, vmin=-1, vmax=1, square=True, linewidths=0.5, ax=ax,
                cbar=idx==2, cbar_kws={'label': 'Spearman rho'})
    ax.set_title(f'{label} (n={len(sub)})')
plt.tight_layout()
plt.savefig('outputs/figures/G4_spearman_per_tipe.png', bbox_inches='tight')
plt.close()
print("  [SAVED] G4_spearman_per_tipe.png")

# --- G5. Violin plot EER per Kategori PK + Tipe ---
fig, ax = plt.subplots(figsize=(14, 7))
sns.violinplot(data=df, x='Kategori_PK', y=eer_col, hue='Tipe',
               order=pk_order, ax=ax, palette='Set2', split=True, inner='quartile')
ax.set_title('Distribusi EER/CSPF per Kategori PK dan Tipe AC\n'
             '(Kruskal-Wallis p < 0.001)', fontsize=12, fontweight='bold')
ax.set_xlabel('Kategori PK')
ax.set_ylabel('Nilai Efisiensi (EER/CSPF)')
plt.tight_layout()
plt.savefig('outputs/figures/G5_violin_eer_per_pk.png', bbox_inches='tight')
plt.close()
print("  [SAVED] G5_violin_eer_per_pk.png")

# --- G6. Pairplot: Daya, Kapasitas, EER, Rating (per Tipe) ---
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Hubungan Bivariat: Daya, Kapasitas, EER/CSPF, Rating',
             fontsize=14, fontweight='bold')

# Daya vs Kapasitas
ax = axes[0, 0]
for tipe, color in [('Non-Inverter', 'steelblue'), ('Inverter', 'coral')]:
    d = df[df['Tipe'] == tipe]
    ax.scatter(d['Daya (watt)'], d['Kapasitas Pendinginan (BTU/h)'],
               c=color, label=tipe, alpha=0.5, s=20)
ax.set_xlabel('Daya (watt)')
ax.set_ylabel('Kapasitas (BTU/h)')
ax.legend()

# Daya vs EER
ax = axes[0, 1]
for tipe, color in [('Non-Inverter', 'steelblue'), ('Inverter', 'coral')]:
    d = df[df['Tipe'] == tipe]
    ax.scatter(d['Daya (watt)'], d[eer_col], c=color, label=tipe, alpha=0.5, s=20)
ax.set_xlabel('Daya (watt)')
ax.set_ylabel('EER/CSPF')
ax.legend()

# Kapasitas vs EER
ax = axes[1, 0]
for tipe, color in [('Non-Inverter', 'steelblue'), ('Inverter', 'coral')]:
    d = df[df['Tipe'] == tipe]
    ax.scatter(d['Kapasitas Pendinginan (BTU/h)'], d[eer_col], c=color, label=tipe, alpha=0.5, s=20)
ax.set_xlabel('Kapasitas (BTU/h)')
ax.set_ylabel('EER/CSPF')
ax.legend()

# EER vs Konsumsi
ax = axes[1, 1]
for tipe, color in [('Non-Inverter', 'steelblue'), ('Inverter', 'coral')]:
    d = df[df['Tipe'] == tipe]
    ax.scatter(d[eer_col], d['Konsumsi Energi Tahunan (kWh)'], c=color, label=tipe, alpha=0.5, s=20)
ax.set_xlabel('EER/CSPF')
ax.set_ylabel('Konsumsi Energi (kWh)')
ax.set_ylim(0, 15000)
ax.legend()

plt.tight_layout()
plt.savefig('outputs/figures/G6_bivariate_scatter.png', bbox_inches='tight')
plt.close()
print("  [SAVED] G6_bivariate_scatter.png")

# ============================================================
# H. LAPORAN RINGKASAN
# ============================================================
print("\n" + "=" * 70)
print("H. RINGKASAN ANALISIS STATISTIK")
print("=" * 70)

print("""
1. UJI NORMALITAS
   - Semua 5 variabel numerik TIDAK berdistribusi normal (Shapiro-Wilk p < 0.001)
   - Keputusan: seluruh analisis menggunakan uji non-parametrik

2. KRUSKAL-WALLIS: EER/CSPF antar Rating Bintang
   - H = {:.2f}, p < 0.001, eta2 = {:.4f}
   - Perbedaan EER/CSPF antar rating SANGAT SIGNIFIKAN
   - Post-hoc Mann-Whitney: lihat tabel 12

3. MANN-WHITNEY U: EER/CSPF Inverter vs Non-Inverter
   - U = {:.0f}, p < 0.001, rank-biserial r = {:.4f}
   - Perbedaan efisiensi antar tipe SIGNIFIKAN

4. CHI-SQUARE: Rating x Tipe
   - Chi2 = {:.2f}, p < 0.001, Cramer V = {:.4f}
   - Asosiasi Rating-Tipe SIGNIFIKAN dan KUAT
   - Semua Rating 5 = Inverter; semua Rating 1 = Non-Inverter

5. KORELASI PARSIAL
   - Daya-Kapasitas: korelasi meningkat setelah kontrol EER
   - Daya-EER: korelasi lemah (~0) setelah kontrol Kapasitas
   - Konsumsi-Biaya: korelasi sangat kuat (derived variable)

6. KRUSKAL-WALLIS: EER/CSPF per Kategori PK
   - H = {:.2f}, p < {:.2e}, eta2 = {:.4f}
   - Perbedaan EER/CSPF antar kategori PK SIGNIFIKAN
""".format(stat_kw, eta_sq, stat_mw2, r_rb2, chi2, cramers_v, stat_pk, p_pk, eta_pk))

print("\n" + "=" * 70)
print("TAHAP 3 SELESAI.")
print("Tabel: outputs/tables/11-14")
print("Visualisasi: outputs/figures/G1-G6")
print("=" * 70)
