"""
============================================================================
 TAHAP 1: DATA UNDERSTANDING & EXPLORATORY DATA ANALYSIS (EDA)
 Dataset : Produk Pengondisi Udara (AC) - SIMEBTKE Kementerian ESDM Indonesia
 Sumber  : https://simebtke.esdm.go.id/sinergi/skem-label/konsumen/pengondisi-udara-ac
 Tujuan  : Identifikasi pola efisiensi energi AC di Indonesia
============================================================================
Aturan penelitian:
 1. Tidak mengubah data mentah
 2. Salinan dataframe sebelum preprocessing
 3. Tidak menghapus data tanpa alasan
 4. Identifikasi missing value sebelum imputasi
 5. Identifikasi kemungkinan data leakage
 6. Tidak membuat kesimpulan sebelum melihat hasil
 7. Tidak mengarang nilai/pola
 8. Semua transformasi dapat direproduksi
 9. Visualisasi sesuai untuk publikasi ilmiah
10. Interpretasi statistik setelah analisis penting
11. Tandai setiap asumsi
12. Simpan preprocessing ke data/processed
13. Simpan grafik ke outputs/figures
14. Simpan tabel analisis ke outputs/tables
============================================================================
"""

# ============================================================
# 0. SETUP & KONFIGURASI
# ============================================================
import os
import warnings
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: f'{x:.4f}')

sns.set_style('whitegrid')
plt.rcParams.update({
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'figure.facecolor': 'white',
})

# Direktori output (aturan 12-14)
for d in ['data/raw', 'data/processed', 'outputs/figures', 'outputs/tables']:
    os.makedirs(d, exist_ok=True)

RAW_PATH = 'data/raw/ac_simebtke_raw.csv'

print("=" * 70)
print("TAHAP 1: DATA UNDERSTANDING & EDA")
print("Dataset: AC SIMEBTKE Kementerian ESDM")
print("=" * 70)

# ============================================================
# A. LOAD DATASET (aturan 1 & 2: tidak ubah mentah, buat salinan)
# ============================================================
print("\n" + "=" * 70)
print("A. LOAD DATASET")
print("=" * 70)

# Baca sebagai string agar format mentah tidak berubah
df_raw = pd.read_csv(RAW_PATH, dtype=str, encoding='utf-8-sig')
n_rows_raw, n_cols_raw = df_raw.shape
print(f"File: {RAW_PATH}")
print(f"Encoding: utf-8-sig")
print(f"Loaded {n_rows_raw} baris x {n_cols_raw} kolom")

# Salinan untuk inspeksi (aturan 2)
df = df_raw.copy()

# ============================================================
# B. SHAPE
# ============================================================
print("\n" + "=" * 70)
print("B. SHAPE DATASET")
print("=" * 70)
print(f"Jumlah baris : {df.shape[0]}")
print(f"Jumlah kolom : {df.shape[1]}")

# ============================================================
# C. NAMA & TIPE KOLOM
# ============================================================
print("\n" + "=" * 70)
print("C. NAMA DAN TIPE KOLOM")
print("=" * 70)
print(df.dtypes.to_string())
print(f"\nTotal kolom numerik (int/float): {(df.dtypes.apply(lambda x: pd.api.types.is_numeric_dtype(x))).sum()}")
print(f"Total kolom object/string     : {(df.dtypes == 'object').sum()}")
print("\n[ASEMSI] Semua kolom terbaca sebagai object/string karena dibaca dengan dtype=str untuk preservasi format mentah.")

# ============================================================
# D. 10 BARIS PERTAMA
# ============================================================
print("\n" + "=" * 70)
print("D. 10 BARIS PERTAMA")
print("=" * 70)
print(df.head(10).to_string())

# ============================================================
# E. MISSING VALUES
# ============================================================
print("\n" + "=" * 70)
print("E. MISSING VALUES")
print("=" * 70)
# Missing value mentah: string kosong, NaN, 'null', 'NA', ''
for_missing = df.replace(['', 'null', 'NA', 'N/A', 'nan', 'None', '-'], np.nan)
missing_count = for_missing.isnull().sum()
missing_pct = (missing_count / len(for_missing) * 100).round(2)
missing_df = pd.DataFrame({
    'Jumlah Missing': missing_count,
    'Persentase (%)': missing_pct,
}).sort_values('Jumlah Missing', ascending=False)
missing_df = missing_df[missing_df['Jumlah Missing'] > 0]
print(missing_df.to_string())
print(f"\nKolom tanpa missing: {list(missing_df[missing_df['Jumlah Missing'] == 0].index)}")
print(f"Kolom dengan missing: {list(missing_df.index)}")

# Simpan tabel missing value
missing_full = pd.DataFrame({
    'Jumlah Missing': for_missing.isnull().sum(),
    'Persentase (%)': (for_missing.isnull().sum() / len(for_missing) * 100).round(2),
})
missing_full.to_csv('outputs/tables/01_missing_values.csv')

# ============================================================
# F. DUPLICATE ROWS
# ============================================================
print("\n" + "=" * 70)
print("F. DUPLICATE ROWS")
print("=" * 70)
# Cek duplikat pada seluruh kolom
dup_all = df.duplicated().sum()
print(f"Duplicate rows (semua kolom): {dup_all}")

# Cek duplikat tanpa kolom NO.
cols_no_no = [c for c in df.columns if c != 'NO.']
dup_no_no = df[cols_no_no].duplicated().sum()
print(f"Duplicate rows (tanpa NO.): {dup_no_no}")

# Cek duplikat pada No. Registrasi
if 'No. Registrasi/No. SHE' in df.columns:
    reg = df['No. Registrasi/No. SHE'].replace('', np.nan).dropna()
    dup_reg = reg.duplicated().sum()
    print(f"Duplicate No. Registrasi/No. SHE: {dup_reg}")
    if dup_reg > 0:
        dups = df[df['No. Registrasi/No. SHE'].isin(reg[reg.duplicated()])].sort_values('No. Registrasi/No. SHE')
        print(f"\nContoh duplikat No. Registrasi:")
        print(dups[['NO.', 'Merek', 'Model', 'No. Registrasi/No. SHE']].head(20).to_string())

pd.DataFrame({
    'Tipe Duplikat': ['Semua kolom', 'Tanpa NO.', 'No. Registrasi/No. SHE'],
    'Jumlah': [dup_all, dup_no_no, dup_reg],
}).to_csv('outputs/tables/02_duplicates.csv', index=False)

# ============================================================
# G. UNIQUE VALUES UNTUK KOLOM KATEGORIKAL
# ============================================================
print("\n" + "=" * 70)
print("G. UNIQUE VALUES - KOLOM KATEGORIKAL")
print("=" * 70)
categorical_cols = ['Merek', 'Famili', 'Model', 'Tipe', 'Rating Bintang (1-5)', 'LSPro']
cat_summary = []
for col in categorical_cols:
    n_unique = df[col].nunique()
    cat_summary.append({'Kolom': col, 'Jumlah Unique': n_unique})
    print(f"\n--- {col} ({n_unique} unique) ---")
    top10 = df[col].value_counts().head(10)
    print(top10.to_string())

pd.DataFrame(cat_summary).to_csv('outputs/tables/03_categorical_unique.csv', index=False)

# ============================================================
# H. FORMAT ANGKA PADA KOLOM NUMERIK
# ============================================================
print("\n" + "=" * 70)
print("H. FORMAT ANGKA PADA KOLOM NUMERIK")
print("=" * 70)
numeric_cols = [
    'Daya (watt)',
    'Kapasitas Pendinginan (BTU/h)',
    'Nilai Efisiensi (EER/CSPF)',
    'Rating Bintang (1-5)',
    'Konsumsi Energi Tahunan (kWh)',
    'Biaya Listrik Tahunan (Rp)',
]

for col in numeric_cols:
    print(f"\n--- {col} ---")
    samples = df[col].dropna().unique()[:10]
    print(f"  Sample nilai mentah: {list(samples)}")
    # Cek apakah ada karakter non-numerik
    has_comma = df[col].astype(str).str.contains(',', na=False).any()
    has_rp = df[col].astype(str).str.contains('Rp', case=False, na=False).any()
    has_watt = df[col].astype(str).str.contains('W', case=False, na=False).any()
    has_btu = df[col].astype(str).str.contains('BTU', case=False, na=False).any()
    has_space = df[col].astype(str).str.contains(' ', na=False).any()
    # Coba parse langsung
    parsed_direct = pd.to_numeric(df[col], errors='coerce')
    n_parsed = parsed_direct.notna().sum()
    n_total = df[col].notna().sum()
    print(f"  Ada koma (,): {has_comma}")
    print(f"  Ada 'Rp': {has_rp}")
    print(f"  Ada 'W': {has_watt}")
    print(f"  Ada 'BTU': {has_btu}")
    print(f"  Ada spasi: {has_space}")
    print(f"  Parse langsung berhasil: {n_parsed}/{n_total}")

# Parsing yang sesuai
print("\n--- STRATEGI PARSING ---")
print("Daya, Kapasitas, EER/CSPF, Rating, Konsumsi Energi: parse langsung (format desimal standar)")
print("Biaya Listrik: hapus koma thousand separator lalu parse")

# ============================================================
# I. FORMAT TANGGAL
# ============================================================
print("\n" + "=" * 70)
print("I. FORMAT TANGGAL")
print("=" * 70)
date_cols = ['Tanggal Terbit SHE', 'SHE Berlaku Sampai Dengan Tanggal']
for col in date_cols:
    print(f"\n--- {col} ---")
    non_null = df[col].replace('', np.nan).dropna()
    print(f"  Non-null: {len(non_null)}/{len(df)}")
    if len(non_null) > 0:
        samples = non_null.unique()[:10]
        print(f"  Sample: {list(samples)}")
        # Coba parse dengan format ISO (YYYY-MM-DD)
        parsed = pd.to_datetime(non_null, format='%Y-%m-%d', errors='coerce')
        n_parsed = parsed.notna().sum()
        print(f"  Parse ISO (YYYY-MM-DD): {n_parsed}/{len(non_null)} berhasil")
        # Cek apakah ada format lain
        if n_parsed < len(non_null):
            unparseable = non_null[parsed.isna()].unique()[:10]
            print(f"  Tidak terparse: {list(unparseable)}")

# ============================================================
# J. IDENTIFIKASI NILAI TIDAK WAJAR
# ============================================================
print("\n" + "=" * 70)
print("J. IDENTIFIKASI NILAI TIDAK WAJAR")
print("=" * 70)

# Parse numerik untuk inspeksi
df_check = df.copy()
for col in ['Daya (watt)', 'Kapasitas Pendinginan (BTU/h)', 'Nilai Efisiensi (EER/CSPF)',
            'Konsumsi Energi Tahunan (kWh)']:
    df_check[col + '_num'] = pd.to_numeric(df_check[col], errors='coerce')

# Biaya Listrik: hapus koma
df_check['Biaya_num'] = pd.to_numeric(
    df_check['Biaya Listrik Tahunan (Rp)'].str.replace(',', '', regex=False),
    errors='coerce'
)
df_check['Rating_num'] = pd.to_numeric(df_check['Rating Bintang (1-5)'], errors='coerce')

# Daya (watt)
print("\n--- Daya (watt) ---")
daya = df_check['Daya (watt)_num'].dropna()
print(f"  Min: {daya.min():.2f}, Max: {daya.max():.2f}, Mean: {daya.mean():.2f}")
n_zero_daya = (daya == 0).sum()
n_neg_daya = (daya < 0).sum()
print(f"  Nilai 0: {n_zero_daya}, Nilai negatif: {n_neg_daya}")
if daya.min() < 100:
    print(f"  [PERHATIAN] Daya < 100W:")
    low = df_check[df_check['Daya (watt)_num'] < 100][['NO.', 'Merek', 'Model', 'Daya (watt)']].head(10)
    print(low.to_string())
if daya.max() > 5000:
    print(f"  [PERHATIAN] Daya > 5000W:")
    high = df_check[df_check['Daya (watt)_num'] > 5000][['NO.', 'Merek', 'Model', 'Daya (watt)']].head(10)
    print(high.to_string())

# Kapasitas Pendinginan
print("\n--- Kapasitas Pendinginan (BTU/h) ---")
kap = df_check['Kapasitas Pendinginan (BTU/h)_num'].dropna()
print(f"  Min: {kap.min():.2f}, Max: {kap.max():.2f}, Mean: {kap.mean():.2f}")
n_zero_kap = (kap == 0).sum()
n_neg_kap = (kap < 0).sum()
print(f"  Nilai 0: {n_zero_kap}, Nilai negatif: {n_neg_kap}")

# EER/CSPF
print("\n--- Nilai Efisiensi (EER/CSPF) ---")
eer = df_check['Nilai Efisiensi (EER/CSPF)_num'].dropna()
print(f"  Min: {eer.min():.2f}, Max: {eer.max():.2f}, Mean: {eer.mean():.2f}")
n_zero_eer = (eer == 0).sum()
n_neg_eer = (eer < 0).sum()
print(f"  Nilai 0: {n_zero_eer}, Nilai negatif: {n_neg_eer}")
if eer.min() < 5:
    print(f"  [PERHATIAN] EER/CSPF < 5:")
    low = df_check[df_check['Nilai Efisiensi (EER/CSPF)_num'] < 5][['NO.', 'Merek', 'Model', 'Nilai Efisiensi (EER/CSPF)']].head(10)
    print(low.to_string())
if eer.max() > 30:
    print(f"  [PERHATIAN] EER/CSPF > 30:")
    high = df_check[df_check['Nilai Efisiensi (EER/CSPF)_num'] > 30][['NO.', 'Merek', 'Model', 'Nilai Efisiensi (EER/CSPF)']].head(10)
    print(high.to_string())

# Rating Bintang
print("\n--- Rating Bintang (1-5) ---")
rating = df_check['Rating_num'].dropna()
print(f"  Min: {rating.min()}, Max: {rating.max()}")
print(f"  Distribusi: {dict(rating.value_counts().sort_index())}")
out_of_range = rating[(rating < 1) | (rating > 5)]
print(f"  Di luar rentang 1-5: {len(out_of_range)}")

# Konsumsi Energi
print("\n--- Konsumsi Energi Tahunan (kWh) ---")
kons = df_check['Konsumsi Energi Tahunan (kWh)_num'].dropna()
print(f"  Min: {kons.min():.2f}, Max: {kons.max():.2f}, Mean: {kons.mean():.2f}")
n_zero_kons = (kons == 0).sum()
n_neg_kons = (kons < 0).sum()
print(f"  Nilai 0: {n_zero_kons}, Nilai negatif: {n_neg_kons}")

# Biaya Listrik
print("\n--- Biaya Listrik Tahunan (Rp) ---")
biaya = df_check['Biaya_num'].dropna()
print(f"  Min: {biaya.min():,.2f}, Max: {biaya.max():,.2f}, Mean: {biaya.mean():,.2f}")
n_zero_biaya = (biaya == 0).sum()
n_neg_biaya = (biaya < 0).sum()
print(f"  Nilai 0: {n_zero_biaya}, Nilai negatif: {n_neg_biaya}")

# Cek konsistensi: EER = Kapasitas / Daya (untuk non-inverter EER, bukan CSPF)
print("\n--- CROSS-CHECK: EER = Kapasitas(BTU/h) / Daya(W) ---")
df_check['EER_calc'] = df_check['Kapasitas Pendinginan (BTU/h)_num'] / df_check['Daya (watt)_num']
df_check['EER_diff'] = abs(df_check['EER_calc'] - df_check['Nilai Efisiensi (EER/CSPF)_num'])
noninv = df_check[df_check['Tipe'] == 'Non-Inverter']
eer_diff_noninv = noninv['EER_diff'].dropna()
print(f"  Non-Inverter: median selisih EER terhitung vs tercatat = {eer_diff_noninv.median():.4f}")
print(f"  Non-Inverter: max selisih = {eer_diff_noninv.max():.4f}")
inv = df_check[df_check['Tipe'] == 'Inverter']
eer_diff_inv = inv['EER_diff'].dropna()
print(f"  Inverter: median selisih CSPF terhitung vs tercatat = {eer_diff_inv.median():.4f}")
print(f"  [ASEMSI] Untuk Inverter, nilai efisiensi adalah CSPF (bukan EER), sehingga rumus EER=Kapasitas/Daya tidak berlaku langsung.")

# Cek konsistensi: Biaya = Konsumsi * tarif
print("\n--- CROSS-CHECK: Biaya = Konsumsi(kWh) * tarif_listrik ---")
df_check['tarif_calc'] = df_check['Biaya_num'] / df_check['Konsumsi Energi Tahunan (kWh)_num']
tarif = df_check['tarif_calc'].dropna()
print(f"  Tarif terhitung: min={tarif.min():.2f}, median={tarif.median():.2f}, max={tarif.max():.2f}")
print(f"  [ASEMSI] Tarif listrik PLN non-subsidi ~Rp 1,444/kWh (2023). Nilai median dapat mengindikasikan tarif acuan.")

# ============================================================
# K. RINGKASAN STATISTIK NUMERIK
# ============================================================
print("\n" + "=" * 70)
print("K. RINGKASAN STATISTIK NUMERIK")
print("=" * 70)

# Buat dataframe numerik yang sudah diparse
df_numeric = pd.DataFrame()
df_numeric['Daya (watt)'] = pd.to_numeric(df['Daya (watt)'], errors='coerce')
df_numeric['Kapasitas Pendinginan (BTU/h)'] = pd.to_numeric(df['Kapasitas Pendinginan (BTU/h)'], errors='coerce')
df_numeric['Nilai Efisiensi (EER/CSPF)'] = pd.to_numeric(df['Nilai Efisiensi (EER/CSPF)'], errors='coerce')
df_numeric['Rating Bintang'] = pd.to_numeric(df['Rating Bintang (1-5)'], errors='coerce')
df_numeric['Konsumsi Energi Tahunan (kWh)'] = pd.to_numeric(df['Konsumsi Energi Tahunan (kWh)'], errors='coerce')
df_numeric['Biaya Listrik Tahunan (Rp)'] = pd.to_numeric(
    df['Biaya Listrik Tahunan (Rp)'].str.replace(',', '', regex=False), errors='coerce'
)

desc = df_numeric.describe(include='all', percentiles=[.01, .05, .25, .5, .75, .95, .99]).T
desc['count_nonnull'] = df_numeric.count()
desc['missing'] = df_numeric.isnull().sum()
desc['missing_pct'] = (df_numeric.isnull().sum() / len(df_numeric) * 100).round(2)
print(desc.to_string())
desc.to_csv('outputs/tables/04_statistical_summary.csv')

# Skewness & Kurtosis
print("\n--- Skewness & Kurtosis ---")
skew_kurt = pd.DataFrame({
    'Skewness': df_numeric.skew(numeric_only=True),
    'Kurtosis': df_numeric.kurtosis(numeric_only=True),
})
print(skew_kurt.to_string())
skew_kurt.to_csv('outputs/tables/05_skewness_kurtosis.csv')

# ============================================================
# L. VISUALISASI AWAL
# ============================================================
print("\n" + "=" * 70)
print("L. VISUALISASI AWAL")
print("=" * 70)

# --- L1. Histogram semua variabel numerik ---
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Distribusi Variabel Numerik - Dataset AC SIMEBTKE', fontsize=14, fontweight='bold')

plot_cols = [
    ('Daya (watt)', 'Daya (watt)'),
    ('Kapasitas Pendinginan (BTU/h)', 'Kapasitas Pendingin (BTU/h)'),
    ('Nilai Efisiensi (EER/CSPF)', 'Nilai Efisiensi (EER/CSPF)'),
    ('Konsumsi Energi Tahunan (kWh)', 'Konsumsi Energi Tahunan (kWh)'),
    ('Biaya Listrik Tahunan (Rp)', 'Biaya Listrik Tahunan (Rp)'),
]
for idx, (col, label) in enumerate(plot_cols):
    ax = axes[idx // 3, idx % 3]
    data = df_numeric[col].dropna()
    ax.hist(data, bins=40, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(data.mean(), color='red', linestyle='--', linewidth=1.5, label=f'Mean={data.mean():.1f}')
    ax.axvline(data.median(), color='green', linestyle='--', linewidth=1.5, label=f'Median={data.median():.1f}')
    ax.set_xlabel(label)
    ax.set_ylabel('Frekuensi')
    ax.legend(fontsize=8)

# Rating Bintang sebagai bar chart
ax = axes[1, 2]
rating_counts = df_numeric['Rating Bintang'].dropna().value_counts().sort_index()
bars = ax.bar(rating_counts.index, rating_counts.values, color='coral', edgecolor='white')
ax.set_xlabel('Rating Bintang')
ax.set_ylabel('Frekuensi')
for bar, val in zip(bars, rating_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, str(val),
            ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('outputs/figures/L1_histograms_numerik.png', bbox_inches='tight')
plt.close()
print("  [SAVED] outputs/figures/L1_histograms_numerik.png")

# --- L2. Boxplot per Tipe (Inverter vs Non-Inverter) ---
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Boxplot per Tipe AC - Inverter vs Non-Inverter', fontsize=14, fontweight='bold')

df_plot = df_numeric.copy()
df_plot['Tipe'] = df['Tipe'].values

box_cols = [
    ('Daya (watt)', 'Daya (watt)'),
    ('Kapasitas Pendinginan (BTU/h)', 'Kapasitas (BTU/h)'),
    ('Nilai Efisiensi (EER/CSPF)', 'EER/CSPF'),
    ('Konsumsi Energi Tahunan (kWh)', 'Konsumsi (kWh)'),
    ('Biaya Listrik Tahunan (Rp)', 'Biaya (Rp)'),
    ('Rating Bintang', 'Rating Bintang'),
]
for idx, (col, label) in enumerate(box_cols):
    ax = axes[idx // 3, idx % 3]
    sns.boxplot(data=df_plot, x='Tipe', y=col, ax=ax, palette='Set2')
    ax.set_xlabel('')
    ax.set_ylabel(label)

plt.tight_layout()
plt.savefig('outputs/figures/L2_boxplot_per_tipe.png', bbox_inches='tight')
plt.close()
print("  [SAVED] outputs/figures/L2_boxplot_per_tipe.png")

# --- L3. Bar chart Top 20 Merek ---
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Top 20 Merek AC berdasarkan Jumlah Model Terdaftar', fontsize=14, fontweight='bold')

top_merek = df['Merek'].value_counts().head(20)
axes[0].barh(top_merek.index[::-1], top_merek.values[::-1], color='steelblue', edgecolor='white')
axes[0].set_xlabel('Jumlah Model')
axes[0].set_title('Top 20 Merek (Semua)')

# Top 20 Merek berdasarkan Tipe
merek_tipe = df.groupby(['Merek', 'Tipe']).size().unstack(fill_value=0)
merek_tipe['Total'] = merek_tipe.sum(axis=1)
top_merek_tipe = merek_tipe.sort_values('Total', ascending=False).head(20).drop('Total', axis=1)
top_merek_tipe.plot(kind='barh', stacked=True, ax=axes[1], color=['steelblue', 'coral'])
axes[1].set_xlabel('Jumlah Model')
axes[1].set_title('Top 20 Merek (per Tipe)')
axes[1].legend(title='Tipe')

plt.tight_layout()
plt.savefig('outputs/figures/L3_top_merek.png', bbox_inches='tight')
plt.close()
print("  [SAVED] outputs/figures/L3_top_merek.png")

# --- L4. Distribusi Rating Bintang per Tipe ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Distribusi Rating Bintang Hemat Energi', fontsize=14, fontweight='bold')

rating_tipe = df_plot.groupby(['Rating Bintang', 'Tipe']).size().unstack(fill_value=0)
rating_tipe.plot(kind='bar', ax=axes[0], color=['steelblue', 'coral'], edgecolor='white')
axes[0].set_xlabel('Rating Bintang')
axes[0].set_ylabel('Jumlah Model')
axes[0].set_title('Jumlah Model per Rating & Tipe')
axes[0].legend(title='Tipe')

# Proporsi
rating_pct = rating_tipe.div(rating_tipe.sum(axis=0), axis=1) * 100
rating_pct.plot(kind='bar', ax=axes[1], color=['steelblue', 'coral'], edgecolor='white')
axes[1].set_xlabel('Rating Bintang')
axes[1].set_ylabel('Persentase (%)')
axes[1].set_title('Proporsi Rating per Tipe')
axes[1].legend(title='Tipe')

plt.tight_layout()
plt.savefig('outputs/figures/L4_rating_per_tipe.png', bbox_inches='tight')
plt.close()
print("  [SAVED] outputs/figures/L4_rating_per_tipe.png")

# --- L5. Scatter: Daya vs Kapasitas, diwarnai Rating ---
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Hubungan Daya, Kapasitas, dan Efisiensi', fontsize=14, fontweight='bold')

scatter_df = df_numeric.copy()
scatter_df['Tipe'] = df['Tipe'].values
scatter_df['Merek'] = df['Merek'].values

sns.scatterplot(data=scatter_df, x='Daya (watt)', y='Kapasitas Pendinginan (BTU/h)',
                hue='Rating Bintang', style='Tipe', ax=axes[0], palette='RdYlGn',
                alpha=0.7, s=40)
axes[0].set_title('Daya vs Kapasitas Pendinginan')

sns.scatterplot(data=scatter_df, x='Daya (watt)', y='Nilai Efisiensi (EER/CSPF)',
                hue='Rating Bintang', style='Tipe', ax=axes[1], palette='RdYlGn',
                alpha=0.7, s=40)
axes[1].set_title('Daya vs Nilai Efisiensi (EER/CSPF)')

plt.tight_layout()
plt.savefig('outputs/figures/L5_scatter_daya_kapasitas_eer.png', bbox_inches='tight')
plt.close()
print("  [SAVED] outputs/figures/L5_scatter_daya_kapasitas_eer.png")

# --- L6. Correlation Heatmap ---
fig, ax = plt.subplots(figsize=(10, 8))
corr = df_numeric.corr()
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, mask=mask, annot=True, fmt='.3f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, square=True, linewidths=0.5, ax=ax,
            cbar_kws={'label': 'Pearson r'})
ax.set_title('Korelasi Pearson antar Variabel Numerik', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/figures/L6_correlation_heatmap.png', bbox_inches='tight')
plt.close()
print("  [SAVED] outputs/figures/L6_correlation_heatmap.png")

# --- L7. Konsumsi Energi & Biaya per Rating Bintang ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Efisiensi Energi & Biaya per Rating Bintang', fontsize=14, fontweight='bold')

sns.boxplot(data=df_plot, x='Rating Bintang', y='Konsumsi Energi Tahunan (kWh)',
            ax=axes[0], palette='RdYlGn')
axes[0].set_title('Konsumsi Energi per Rating Bintang')

sns.boxplot(data=df_plot, x='Rating Bintang', y='Biaya Listrik Tahunan (Rp)',
            ax=axes[1], palette='RdYlGn')
axes[1].set_title('Biaya Listrik per Rating Bintang')
axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))

plt.tight_layout()
plt.savefig('outputs/figures/L7_konsumsi_biaya_per_rating.png', bbox_inches='tight')
plt.close()
print("  [SAVED] outputs/figures/L7_konsumsi_biaya_per_rating.png")

# --- L8. Distribusi LSPro ---
fig, ax = plt.subplots(figsize=(10, 6))
lspro_counts = df['LSPro'].replace('', np.nan).dropna().value_counts()
ax.barh(lspro_counts.index[::-1], lspro_counts.values[::-1], color='steelblue', edgecolor='white')
ax.set_xlabel('Jumlah Model')
ax.set_title('Distribusi Lembaga Sertifikasi (LSPro)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/figures/L8_lspro_distribution.png', bbox_inches='tight')
plt.close()
print("  [SAVED] outputs/figures/L8_lspro_distribution.png")

# ============================================================
# M. TIDAK ADA MACHINE LEARNING
# ============================================================
print("\n" + "=" * 70)
print("M. MACHINE LEARNING")
print("=" * 70)
print("Tahap ini TIDAK melakukan machine learning.")
print("Fokus: data understanding dan EDA saja.")

# ============================================================
# SIMPAN DATA YANG SUDAH DIPARSE NUMERIK (reproducible)
# ============================================================
df_processed = df.copy()
# Rename NO. untuk konsistensi
df_processed = df_processed.rename(columns={'NO.': 'NO.'})

# Parse numerik
for col in ['Daya (watt)', 'Kapasitas Pendinginan (BTU/h)', 'Nilai Efisiensi (EER/CSPF)',
            'Rating Bintang (1-5)', 'Konsumsi Energi Tahunan (kWh)']:
    df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')

df_processed['Biaya Listrik Tahunan (Rp)'] = pd.to_numeric(
    df_processed['Biaya Listrik Tahunan (Rp)'].str.replace(',', '', regex=False),
    errors='coerce'
)

# Parse tanggal
for col in ['Tanggal Terbit SHE', 'SHE Berlaku Sampai Dengan Tanggal']:
    df_processed[col] = pd.to_datetime(df_processed[col], format='%Y-%m-%d', errors='coerce')

df_processed.to_csv('data/processed/ac_simebtke_parsed.csv', index=False, encoding='utf-8-sig')
print(f"\n[SAVED] data/processed/ac_simebtke_parsed.csv ({df_processed.shape[0]} x {df_processed.shape[1]})")

# Simpan info tipe data
dtype_info = pd.DataFrame({
    'Kolom': df_processed.columns,
    'Tipe_Data': [str(t) for t in df_processed.dtypes],
    'Non_Null': df_processed.count().values,
    'Missing': df_processed.isnull().sum().values,
})
dtype_info.to_csv('outputs/tables/00_dtype_info.csv', index=False)

print("\n" + "=" * 70)
print("TAHAP 1 SELESAI. Lihat outputs/figures dan outputs/tables.")
print("=" * 70)
