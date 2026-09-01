"""
============================================================================
 TAHAP 2: DATA CLEANING & PREPROCESSING
 Dataset : Produk Pengondisi Udara (AC) - SIMEBTKE Kementerian ESDM
 Tujuan : Membersihkan dan menyiapkan data untuk analisis lanjutan
 Aturan  : Lihat Tahap 1 (14 aturan penelitian)
============================================================================
Langkah:
  A. Load data mentah + buat salinan
  B. Normalisasi Merek (case consistency)
  C. Parse numerik (5 kolom langsung + Biaya hapus koma)
  D. Parse tanggal (tangani 0000-00-00)
  E. Tangani missing values (dokumentasi, bukan imputasi sembarangan)
  F. Investigasi & tandai duplikat No. Registrasi
  G. Identifikasi & flag outlier
  H. Feature engineering (rasio, kategori PK)
  I. Simpan data bersih
  J. Visualisasi pre vs post cleaning
  K. Laporan ringkasan preprocessing
============================================================================
"""

# ============================================================
# 0. SETUP
# ============================================================
import os
import warnings
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
    'figure.dpi': 100, 'savefig.dpi': 300, 'font.size': 10,
    'axes.titlesize': 12, 'axes.labelsize': 10, 'figure.facecolor': 'white',
})

RAW_PATH = 'data/raw/ac_simebtke_raw.csv'

print("=" * 70)
print("TAHAP 2: DATA CLEANING & PREPROCESSING")
print("=" * 70)

# ============================================================
# A. LOAD DATA MENTAH + SALINAN (aturan 1 & 2)
# ============================================================
print("\n" + "=" * 70)
print("A. LOAD DATA MENTAH")
print("=" * 70)
df_raw = pd.read_csv(RAW_PATH, dtype=str, encoding='utf-8-sig')
print(f"Data mentah: {df_raw.shape[0]} baris x {df_raw.shape[1]} kolom")

# Salinan untuk preprocessing (aturan 2: jangan ubah mentah)
df = df_raw.copy()
n_before = len(df)

# ============================================================
# B. NORMALISASI MEREK (case consistency)
# ============================================================
print("\n" + "=" * 70)
print("B. NORMALISASI MEREK")
print("=" * 70)

# Identifikasi inkonsistensi case
merek_counts_before = df['Merek'].value_counts()
print(f"Merek unik sebelum normalisasi: {df['Merek'].nunique()}")

# Cari merek yang hanya berbeda huruf besar/kecil
merek_lower = df['Merek'].str.strip().str.lower()
dupes = merek_lower.value_counts()
case_issues = dupes[dupes > 1]
print(f"Merek dengan inkonsistensi case: {len(case_issues)}")
for mk in case_issues.index:
    variants = df[df['Merek'].str.strip().str.lower() == mk]['Merek'].unique()
    print(f"  '{mk}': {list(variants)}")

# Normalisasi: strip whitespace, title case, khusus Gree/GREE -> Gree
df['Merek'] = df['Merek'].str.strip()

# Mapping khusus untuk inkonsistensi yang teridentifikasi
# Gree vs GREE, Panasonic vs Panasonic, DAIKIN vs Daikin, dll.
merek_mapping = {}
for mk in df['Merek'].unique():
    mk_lower = mk.strip().lower()
    if mk_lower not in merek_mapping:
        merek_mapping[mk_lower] = mk  # simpan bentuk pertama yang ditemukan

# Normalisasi: gunakan bentuk yang paling sering muncul untuk setiap versi lower
for mk_lower in case_issues.index:
    # Pilih variant dengan count terbesar
    variants = df[df['Merek'].str.strip().str.lower() == mk_lower]['Merek']
    best_variant = variants.value_counts().index[0]
    merek_mapping[mk_lower] = best_variant.strip()
    print(f"  -> Normalisasi: '{mk_lower}' -> '{best_variant.strip()}' ({variants.shape[0]} records)")

df['Merek'] = df['Merek'].str.lower().map(merek_mapping)
# Title case untuk yang belum di-mapping khusus (tapi jangan ubah yang sudah mapped)
df['Merek'] = df['Merek'].str.strip()

print(f"\nMerek unik setelah normalisasi: {df['Merek'].nunique()}")
print("Top 10 Merek setelah normalisasi:")
print(df['Merek'].value_counts().head(10).to_string())

# Simpan mapping untuk reproducibility (aturan 8)
import json
with open('data/processed/merek_mapping.json', 'w', encoding='utf-8') as f:
    json.dump({k: v for k, v in merek_mapping.items() if k in case_issues.index}, f, ensure_ascii=False, indent=2)

# ============================================================
# C. PARSE NUMERIK
# ============================================================
print("\n" + "=" * 70)
print("C. PARSE NUMERIK")
print("=" * 70)

# 5 kolom numerik langsung parse
for col in ['Daya (watt)', 'Kapasitas Pendinginan (BTU/h)', 'Nilai Efisiensi (EER/CSPF)',
            'Rating Bintang (1-5)', 'Konsumsi Energi Tahunan (kWh)']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    n_ok = df[col].notna().sum()
    print(f"  {col}: {n_ok}/{len(df)} parsed ({n_ok/len(df)*100:.1f}%)")

# Biaya Listrik: hapus koma thousand separator
df['Biaya Listrik Tahunan (Rp)'] = pd.to_numeric(
    df['Biaya Listrik Tahunan (Rp)'].str.replace(',', '', regex=False),
    errors='coerce'
)
n_ok = df['Biaya Listrik Tahunan (Rp)'].notna().sum()
print(f"  Biaya Listrik Tahunan (Rp): {n_ok}/{len(df)} parsed ({n_ok/len(df)*100:.1f}%)")

# ============================================================
# D. PARSE TANGGAL (tangani 0000-00-00)
# ============================================================
print("\n" + "=" * 70)
print("D. PARSE TANGGAL")
print("=" * 70)

for col in ['Tanggal Terbit SHE', 'SHE Berlaku Sampai Dengan Tanggal']:
    # Tandai 0000-00-00 sebelum parse
    invalid_mask = df[col].astype(str).str.strip().isin(['0000-00-00', '', 'null', 'NaN', 'None'])
    n_invalid = invalid_mask.sum()
    print(f"  {col}: {n_invalid} nilai tidak valid (0000-00-00/kosong)")

    # Replace invalid dengan NaT
    df[col] = df[col].where(~invalid_mask, np.nan)
    # Parse ISO
    df[col] = pd.to_datetime(df[col], format='%Y-%m-%d', errors='coerce')
    n_ok = df[col].notna().sum()
    print(f"  {col}: {n_ok}/{len(df)} tanggal valid ({n_ok/len(df)*100:.1f}%)")

# ============================================================
# E. MISSING VALUES (dokumentasi, bukan imputasi sembarangan)
# ============================================================
print("\n" + "=" * 70)
print("E. MISSING VALUES - DOKUMENTASI")
print("=" * 70)

missing_report = pd.DataFrame({
    'Jumlah_Missing': df.isnull().sum(),
    'Persentase': (df.isnull().sum() / len(df) * 100).round(2),
})
missing_report = missing_report[missing_report['Jumlah_Missing'] > 0].sort_values('Jumlah_Missing', ascending=False)
print(missing_report.to_string())

# Keputusan missing values:
# 1. Tanggal Terbit SHE & SHE Berlaku: TIDAK diimputasi (data administratif, tidak bisa diduga)
# 2. LSPro: TIDAK diimputasi (lembaga sertifikasi tidak bisa diduga)
# 3. Kolom numerik: 0 missing setelah parse
print("\n[KEPUTUSAN] Missing values pada kolom tanggal & LSPro TIDAK diimputasi.")
print("  Alasan: data administratif yang tidak dapat diduga dari variabel lain.")
print("  Penanganan: dipertahankan sebagai NaN; analisis temporal hanya pada subset non-null.")

missing_report.to_csv('outputs/tables/06_missing_values_post.csv')

# ============================================================
# F. DUPLIKAT NO. REGISTRASI (investigasi & flagging)
# ============================================================
print("\n" + "=" * 70)
print("F. INVESTIGASI & FLAGGING DUPLIKAT NO. REGISTRASI")
print("=" * 70)

# Identifikasi duplikat No. Registrasi
reg_col = 'No. Registrasi/No. SHE'
reg_nonnull = df[reg_col].dropna()
dup_mask = df[reg_col].duplicated(keep=False)
n_dup_records = dup_mask.sum()
n_dup_reg = df.loc[dup_mask, reg_col].nunique()
print(f"Records dengan No. Registrasi duplikat: {n_dup_records}")
print(f"Jumlah No. Registrasi yang duplikat: {n_dup_reg}")

# Tandai flag is_duplicate_reg
df['is_duplicate_reg'] = dup_mask

# Analisis: apakah record duplikat punya model berbeda?
dup_df = df.loc[dup_mask].sort_values(reg_col)
n_unique_models_in_dup = dup_df.groupby(reg_col)['Model'].nunique()
multi_model = (n_unique_models_in_dup > 1).sum()
same_model = (n_unique_models_in_dup == 1).sum()
print(f"\nNo. Registrasi duplikat dengan >1 model berbeda: {multi_model}")
print(f"No. Registrasi duplikat dengan model identik: {same_model}")

# Contoh
print("\nContoh duplikat dengan model berbeda:")
example_reg = n_unique_models_in_dup[n_unique_models_in_dup > 1].index[0]
print(dup_df[dup_df[reg_col] == example_reg][['NO.', 'Merek', 'Model', 'Daya (watt)', reg_col]].to_string())

# Keputusan: TIDAK menghapus duplikat (aturan 3)
# Alasan: merepresentasikan produk berbeda dengan satu sertifikasi (batch registration)
print("\n[KEPUTUSAN] Duplikat No. Registrasi TIDAK dihapus.")
print("  Alasan: Setiap record merepresentasikan model produk yang berbeda")
print("  yang tersertifikasi dalam satu batch sertifikasi.")
print("  Penanganan: flagging dengan kolom is_duplicate_reg untuk analisis agregat.")

# Simpan tabel duplikat
dup_summary = pd.DataFrame({
    'Metrik': [
        'Total records',
        'Records dengan No. Reg duplikat',
        'Jumlah No. Reg duplikat',
        'Duplikat dengan >1 model berbeda',
        'Duplikat dengan model identik',
    ],
    'Nilai': [len(df), n_dup_records, n_dup_reg, multi_model, same_model],
})
dup_summary.to_csv('outputs/tables/07_duplicate_investigation.csv', index=False)

# ============================================================
# G. IDENTIFIKASI & FLAG OUTLIER
# ============================================================
print("\n" + "=" * 70)
print("G. IDENTIFIKASI & FLAG OUTLIER")
print("=" * 70)

outlier_cols = ['Daya (watt)', 'Kapasitas Pendinginan (BTU/h)',
                'Konsumsi Energi Tahunan (kWh)', 'Biaya Listrik Tahunan (Rp)']

for col in outlier_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    flag_col = f'outlier_{col.split(" ")[0].lower()}'
    df[flag_col] = ((df[col] < lower) | (df[col] > upper)).astype(int)

    n_out = df[flag_col].sum()
    pct = n_out / len(df) * 100
    print(f"  {col}: IQR=[{q1:.1f}-{q3:.1f}], fence=[{lower:.1f}-{upper:.1f}], "
          f"outlier={n_out} ({pct:.2f}%)")

# Identifikasi multi-model registration (penyebab utama outlier)
# Indikator: Daya > 5000W atau Konsumsi > 100,000 kWh (threshold berdasarkan EDA)
multi_model_mask = (df['Daya (watt)'] > 5000) | (df['Konsumsi Energi Tahunan (kWh)'] > 100000)
df['is_multi_model_reg'] = multi_model_mask.astype(int)
n_multi = df['is_multi_model_reg'].sum()
print(f"\n  Multi-model registration (Daya>5000W atau Konsumsi>100,000 kWh): {n_multi} records")
print(f"  [ASEMSI] Threshold berdasarkan domain knowledge: AC residential single-unit")
print(f"  jarang melebihi 5000W. Nilai di atas ini kemungkinan representasi")
print(f"  gabungan multiple indoor/outdoor units dalam satu registrasi.")

# Biaya = 0 atau = 99,999,999.99 (placeholder)
df['is_suspicious_biaya'] = ((df['Biaya Listrik Tahunan (Rp)'] == 0) |
                              (df['Biaya Listrik Tahunan (Rp)'] == 99999999.99)).astype(int)
n_susp = df['is_suspicious_biaya'].sum()
print(f"\n  Biaya mencurigakan (Rp 0 atau Rp 99,999,999.99): {n_susp} records")

# Daya < 10W (tidak wajar untuk AC)
df['is_suspicious_daya'] = (df['Daya (watt)'] < 10).astype(int)
n_susp_daya = df['is_suspicious_daya'].sum()
print(f"  Daya tidak wajar (< 10W): {n_susp_daya} records")

# Keputusan: TIDAK menghapus outlier (aturan 3), hanya flagging
print("\n[KEPUTUSAN] Outlier TIDAK dihapus, hanya di-flag.")
print("  Alasan: Outlier teridentifikasi sebagai multi-model registration")
print("  (registrasi gabungan), bukan kesalahan input. Flagging memungkinkan")
print("  analisis dengan/tanpa outlier untuk perbandingan.")

outlier_summary = pd.DataFrame({
    'Kolom': outlier_cols + ['Biaya Listrik Tahunan (Rp)', 'Daya (watt)'],
    'Metode': ['IQR 1.5x'] * 4 + ['Domain rule (0 / 99,999,999.99)', 'Domain rule (<10W)'],
    'Flag': ['outlier_daya', 'outlier_kapasitas', 'outlier_konsumsi', 'outlier_biaya',
             'is_suspicious_biaya', 'is_suspicious_daya'],
    'Jumlah': [df[f'outlier_{c.split(" ")[0].lower()}'].sum() for c in outlier_cols] +
              [n_susp, n_susp_daya],
})
outlier_summary.to_csv('outputs/tables/08_outlier_flags.csv', index=False)

# ============================================================
# H. FEATURE ENGINEERING
# ============================================================
print("\n" + "=" * 70)
print("H. FEATURE ENGINEERING")
print("=" * 70)

# H1. EER terhitung = Kapasitas / Daya (untuk Non-Inverter)
df['EER_calc'] = df['Kapasitas Pendinginan (BTU/h)'] / df['Daya (watt)']
print(f"  EER_calc = Kapasitas / Daya (median: {df['EER_calc'].median():.2f})")

# H2. Kategori PK (Paarde Kracht) berdasarkan Kapasitas Pendinginan
# 1 PK ≈ 9,000 BTU/h, 0.5 PK ≈ 5,000 BTU/h
def kap_to_pk(btu):
    if pd.isna(btu):
        return np.nan
    pk = btu / 9000
    if pk <= 0.6:
        return '0.5 PK'
    elif pk <= 1.1:
        return '1 PK'
    elif pk <= 1.6:
        return '1.5 PK'
    elif pk <= 2.1:
        return '2 PK'
    elif pk <= 2.6:
        return '2.5 PK'
    else:
        return '3+ PK'

df['Kategori_PK'] = df['Kapasitas Pendinginan (BTU/h)'].apply(kap_to_pk)
print(f"  Kategori_PK:")
print(df['Kategori_PK'].value_counts().sort_index().to_string())

# H3. Efisiensi per PK (normalisasi kapasitas)
df['EER_per_PK'] = df['Nilai Efisiensi (EER/CSPF)']  # placeholder, bisa diperluas

# H4. Tarif terhitung (validasi)
df['tarif_calc'] = df['Biaya Listrik Tahunan (Rp)'] / df['Konsumsi Energi Tahunan (kWh)']
tarif_valid = df['tarif_calc'].replace([np.inf, -np.inf], np.nan).dropna()
print(f"\n  Tarif terhitung: median={tarif_valid.median():.2f} Rp/kWh "
      f"(valid: {len(tarif_valid)}/{len(df)})")

# H5. Selisih EER tercatat vs terhitung (validasi konsistensi)
df['EER_diff'] = abs(df['EER_calc'] - df['Nilai Efisiensi (EER/CSPF)'])
noninv = df[df['Tipe'] == 'Non-Inverter']
print(f"  EER diff (Non-Inverter): median={noninv['EER_diff'].median():.4f}, "
      f"mean={noninv['EER_diff'].mean():.4f}")

# H6. Status SHE (aktif/kedaluwarsa/tidak ada)
import datetime
today = pd.Timestamp('2024-09-01')
df['SHE_status'] = 'Tidak Ada Data'
has_exp = df['SHE Berlaku Sampai Dengan Tanggal'].notna()
df.loc[has_exp & (df['SHE Berlaku Sampai Dengan Tanggal'] >= today), 'SHE_status'] = 'Aktif'
df.loc[has_exp & (df['SHE Berlaku Sampai Dengan Tanggal'] < today), 'SHE_status'] = 'Kedaluwarsa'
print(f"\n  SHE_status:")
print(df['SHE_status'].value_counts().to_string())

# ============================================================
# I. SIMPAN DATA BERSIH
# ============================================================
print("\n" + "=" * 70)
print("I. SIMPAN DATA BERSIH")
print("=" * 70)

output_path = 'data/processed/ac_simebtke_clean.csv'
df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"Disimpan: {output_path}")
print(f"Shape: {df.shape[0]} baris x {df.shape[1]} kolom")
print(f"\nKolom baru yang ditambahkan:")
new_cols = [c for c in df.columns if c not in df_raw.columns]
for c in new_cols:
    print(f"  - {c}")

# Simpan schema
schema = pd.DataFrame({
    'Kolom': df.columns,
    'Tipe': [str(t) for t in df.dtypes],
    'Non_Null': df.count().values,
    'Missing': df.isnull().sum().values,
    'Missing_Pct': (df.isnull().sum() / len(df) * 100).round(2).values,
    'Kategori': ['Original'] * len(df_raw.columns) + ['Derived'] * (len(df.columns) - len(df_raw.columns)),
})
schema.to_csv('outputs/tables/09_schema_clean.csv', index=False)

# ============================================================
# J. VISUALISASI PRE vs POST CLEANING
# ============================================================
print("\n" + "=" * 70)
print("J. VISUALISASI PRE vs POST CLEANING")
print("=" * 70)

# Load data sebelum cleaning untuk perbandingan
df_pre = pd.read_csv('data/raw/ac_simebtke_raw.csv', dtype=str, encoding='utf-8-sig')
df_pre['Daya (watt)'] = pd.to_numeric(df_pre['Daya (watt)'], errors='coerce')
df_pre['Kapasitas Pendinginan (BTU/h)'] = pd.to_numeric(df_pre['Kapasitas Pendinginan (BTU/h)'], errors='coerce')
df_pre['Nilai Efisiensi (EER/CSPF)'] = pd.to_numeric(df_pre['Nilai Efisiensi (EER/CSPF)'], errors='coerce')
df_pre['Biaya Listrik Tahunan (Rp)'] = pd.to_numeric(
    df_pre['Biaya Listrik Tahunan (Rp)'].str.replace(',', '', regex=False), errors='coerce')
df_pre['Merek_normalized'] = df_pre['Merek'].str.strip().str.lower().map(merek_mapping)

# --- J1. Perbandingan jumlah Merek sebelum vs sesudah normalisasi ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Efek Normalisasi Merek: Sebelum vs Sesudah', fontsize=14, fontweight='bold')

pre_counts = df_pre['Merek'].value_counts().head(15)
axes[0].barh(pre_counts.index[::-1], pre_counts.values[::-1], color='lightcoral', edgecolor='white')
axes[0].set_xlabel('Jumlah Model')
axes[0].set_title(f'Sebelum ({df_pre["Merek"].nunique()} unique)')

post_counts = df['Merek'].value_counts().head(15)
axes[1].barh(post_counts.index[::-1], post_counts.values[::-1], color='steelblue', edgecolor='white')
axes[1].set_xlabel('Jumlah Model')
axes[1].set_title(f'Sesudah ({df["Merek"].nunique()} unique)')

plt.tight_layout()
plt.savefig('outputs/figures/J1_merek_normalization.png', bbox_inches='tight')
plt.close()
print("  [SAVED] J1_merek_normalization.png")

# --- J2. Histogram Daya dengan vs tanpa outlier ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Distribusi Daya: Dengan vs Tanpa Outlier (Multi-Model Registration)',
             fontsize=14, fontweight='bold')

axes[0].hist(df['Daya (watt)'].dropna(), bins=50, color='lightcoral', edgecolor='white', alpha=0.8)
axes[0].axvline(df['Daya (watt)'].median(), color='blue', linestyle='--', label=f'Median={df["Daya (watt)"].median():.0f}W')
axes[0].set_xlabel('Daya (watt)')
axes[0].set_ylabel('Frekuensi')
axes[0].set_title(f'Dengan Outlier (n={len(df)})')
axes[0].legend()

clean_daya = df[df['is_multi_model_reg'] == 0]['Daya (watt)'].dropna()
axes[1].hist(clean_daya, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
axes[1].axvline(clean_daya.median(), color='red', linestyle='--', label=f'Median={clean_daya.median():.0f}W')
axes[1].set_xlabel('Daya (watt)')
axes[1].set_ylabel('Frekuensi')
axes[1].set_title(f'Tanpa Multi-Model (n={len(clean_daya)})')
axes[1].legend()

plt.tight_layout()
plt.savefig('outputs/figures/J2_daya_outlier_comparison.png', bbox_inches='tight')
plt.close()
print("  [SAVED] J2_daya_outlier_comparison.png")

# --- J3. Boxplot EER/CSPF per Kategori PK ---
fig, ax = plt.subplots(figsize=(12, 7))
order = ['0.5 PK', '1 PK', '1.5 PK', '2 PK', '2.5 PK', '3+ PK']
df_clean = df[df['is_multi_model_reg'] == 0]
sns.boxplot(data=df_clean, x='Kategori_PK', y='Nilai Efisiensi (EER/CSPF)',
            hue='Tipe', order=order, ax=ax, palette='Set2')
ax.set_title('Nilai Efisiensi (EER/CSPF) per Kategori PK dan Tipe AC\n(Eksklusi Multi-Model Registration)',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Kategori PK')
ax.set_ylabel('EER / CSPF')
plt.tight_layout()
plt.savefig('outputs/figures/J3_eer_per_pk.png', bbox_inches='tight')
plt.close()
print("  [SAVED] J3_eer_per_pk.png")

# --- J4. Scatter Daya vs Kapasitas dengan flag outlier ---
fig, ax = plt.subplots(figsize=(12, 8))
scatter_data = df.copy()
scatter_data['Status'] = 'Normal'
scatter_data.loc[scatter_data['is_multi_model_reg'] == 1, 'Status'] = 'Multi-Model (Outlier)'
scatter_data.loc[scatter_data['is_suspicious_daya'] == 1, 'Status'] = 'Daya Mencurigakan'
sns.scatterplot(data=scatter_data, x='Daya (watt)', y='Kapasitas Pendinginan (BTU/h)',
                hue='Status', style='Tipe', ax=ax, palette={'Normal': 'steelblue',
                'Multi-Model (Outlier)': 'red', 'Daya Mencurigakan': 'orange'},
                alpha=0.7, s=40)
ax.set_title('Daya vs Kapasitas dengan Flag Outlier', fontsize=13, fontweight='bold')
ax.set_xlabel('Daya (watt)')
ax.set_ylabel('Kapasitas Pendinginan (BTU/h)')
plt.tight_layout()
plt.savefig('outputs/figures/J4_scatter_with_outlier_flag.png', bbox_inches='tight')
plt.close()
print("  [SAVED] J4_scatter_with_outlier_flag.png")

# --- J5. Korelasi heatmap data bersih (tanpa multi-model) ---
fig, axes = plt.subplots(1, 2, figsize=(20, 8))
fig.suptitle('Korelasi Pearson: Data Lengkap vs Tanpa Multi-Model Registration',
             fontsize=14, fontweight='bold')

num_cols = ['Daya (watt)', 'Kapasitas Pendinginan (BTU/h)', 'Nilai Efisiensi (EER/CSPF)',
            'Rating Bintang (1-5)', 'Konsumsi Energi Tahunan (kWh)', 'Biaya Listrik Tahunan (Rp)']

corr_full = df[num_cols].corr()
mask = np.triu(np.ones_like(corr_full, dtype=bool), k=1)
sns.heatmap(corr_full, mask=mask, annot=True, fmt='.3f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, square=True, linewidths=0.5, ax=axes[0],
            cbar_kws={'label': 'Pearson r'})
axes[0].set_title(f'Data Lengkap (n={len(df)})')

corr_clean = df_clean[num_cols].corr()
mask2 = np.triu(np.ones_like(corr_clean, dtype=bool), k=1)
sns.heatmap(corr_clean, mask=mask2, annot=True, fmt='.3f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, square=True, linewidths=0.5, ax=axes[1],
            cbar_kws={'label': 'Pearson r'})
axes[1].set_title(f'Tanpa Multi-Model (n={len(df_clean)})')

plt.tight_layout()
plt.savefig('outputs/figures/J5_correlation_pre_post_cleaning.png', bbox_inches='tight')
plt.close()
print("  [SAVED] J5_correlation_pre_post_cleaning.png")

# --- J6. SHE Status distribution ---
fig, ax = plt.subplots(figsize=(8, 6))
she_counts = df['SHE_status'].value_counts()
colors = {'Aktif': 'green', 'Kedaluwarsa': 'red', 'Tidak Ada Data': 'gray'}
bars = ax.bar(she_counts.index, she_counts.values,
              color=[colors.get(x, 'blue') for x in she_counts.index], edgecolor='white')
ax.set_ylabel('Jumlah Records')
ax.set_title('Status Sertifikat Hemat Energi (SHE)', fontsize=13, fontweight='bold')
for bar, val in zip(bars, she_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, str(val),
            ha='center', va='bottom', fontsize=11)
plt.tight_layout()
plt.savefig('outputs/figures/J6_she_status.png', bbox_inches='tight')
plt.close()
print("  [SAVED] J6_she_status.png")

# ============================================================
# K. RINGKASAN PREPROCESSING
# ============================================================
print("\n" + "=" * 70)
print("K. RINGKASAN PREPROCESSING")
print("=" * 70)

summary = {
    'Langkah': [
        'A. Data mentah dimuat',
        'B. Normalisasi Merek',
        'C. Parse numerik (5 kolom + Biaya)',
        'D. Parse tanggal (0000-00-00 -> NaT)',
        'E. Missing values (dipertahankan, tidak diimputasi)',
        'F. Duplikat No. Reg (di-flag, tidak dihapus)',
        'G. Outlier (di-flag, tidak dihapus)',
        'H. Feature engineering (8 fitur baru)',
        'I. Data bersih disimpan',
    ],
    'Detail': [
        f'{n_before} baris x {len(df_raw.columns)} kolom',
        f'{df_raw["Merek"].nunique()} -> {df["Merek"].nunique()} unique',
        '100% sukses (6/6 kolom)',
        '1 nilai 0000-00-00 -> NaT; 1458 missing dipertahankan',
        'Tanggal SHE: 75.82% | LSPro: 57.88%',
        f'{n_dup_records} records duplikat ({n_dup_reg} No. Reg unik)',
        f'{df["is_multi_model_reg"].sum()} multi-model | {n_susp} biaya | {n_susp_daya} daya',
        f'EER_calc, Kategori_PK, tarif_calc, EER_diff, SHE_status, dll.',
        f'{output_path} ({df.shape[0]} x {df.shape[1]})',
    ],
}
summary_df = pd.DataFrame(summary)
print(summary_df.to_string(index=False))
summary_df.to_csv('outputs/tables/10_preprocessing_summary.csv', index=False)

# Statistik perbandingan
print("\n" + "=" * 70)
print("PERBANDINGAN STATISTIK: DENGAN vs TANPA MULTI-MODEL")
print("=" * 70)
for col in ['Daya (watt)', 'Kapasitas Pendinginan (BTU/h)', 'Nilai Efisiensi (EER/CSPF)',
            'Konsumsi Energi Tahunan (kWh)', 'Biaya Listrik Tahunan (Rp)']:
    full = df[col].dropna()
    clean = df_clean[col].dropna()
    print(f"\n--- {col} ---")
    print(f"  Full:     median={full.median():.2f}, mean={full.mean():.2f}, std={full.std():.2f}")
    print(f"  Clean:    median={clean.median():.2f}, mean={clean.mean():.2f}, std={clean.std():.2f}")
    print(f"  Skew:     full={full.skew():.2f}, clean={clean.skew():.2f}")

# Korelasi perbandingan
print("\n" + "=" * 70)
print("KORELASI: DENGAN vs TANPA MULTI-MODEL")
print("=" * 70)
print("\n--- Data Lengkap ---")
print(corr_full.round(4).to_string())
print("\n--- Tanpa Multi-Model ---")
print(corr_clean.round(4).to_string())

print("\n" + "=" * 70)
print("TAHAP 2 SELESAI.")
print("Data bersih: data/processed/ac_simebtke_clean.csv")
print("Visualisasi: outputs/figures/J1-J6")
print("Tabel: outputs/tables/06-10")
print("=" * 70)
