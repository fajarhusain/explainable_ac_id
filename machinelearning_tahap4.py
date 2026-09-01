"""
============================================================================
 TAHAP 4: MACHINE LEARNING
 Dataset : AC SIMEBTKE (data bersih, eksklusi multi-model registration)
 Tujuan : Clustering, Klasifikasi Rating, Regresi EER/CSPF
 Aturan  : Lihat Tahap 1 (14 aturan penelitian)
 Leakage prevention:
   - Target klasifikasi Rating -> prediktor: Daya, Kapasitas, Tipe (BUKAN EER/CSPF, Konsumsi, Biaya)
   - Target regresi EER/CSPF -> prediktor: Daya, Kapasitas, Tipe (BUKAN Konsumsi, Biaya, Rating)
============================================================================
Langkah:
  A. Clustering (K-Means) — profil Daya-Kapasitas-EER
  B. Klasifikasi Rating Bintang (RF, KNN, DT, LR)
  C. Regresi EER/CSPF (RF, Ridge, GBM, LR)
  D. Evaluasi komprehensif (CV, confusion matrix, metrics)
  E. Feature importance & interpretasi
  F. Visualisasi ML
  G. Laporan ringkasan
============================================================================
"""

# ============================================================
# 0. SETUP
# ============================================================
import os, warnings, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import (train_test_split, cross_val_score, 
                                      StratifiedKFold, RepeatedKFold)
from sklearn.metrics import (classification_report, confusion_matrix, 
                              accuracy_score, f1_score, mean_squared_error,
                              mean_absolute_error, r2_score, silhouette_score,
                              silhouette_samples)
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression
from sklearn.pipeline import Pipeline

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

RANDOM_STATE = 42
CLEAN_PATH = 'data/processed/ac_simebtke_clean.csv'

print("=" * 70)
print("TAHAP 4: MACHINE LEARNING")
print("=" * 70)

# Load data bersih, eksklusi multi-model registration
df_full = pd.read_csv(CLEAN_PATH, encoding='utf-8-sig')
df = df_full[df_full['is_multi_model_reg'] == 0].copy()
print(f"Data analisis (tanpa multi-model): {len(df)} records")

# ============================================================
# A. CLUSTERING (K-Means)
# ============================================================
print("\n" + "=" * 70)
print("A. CLUSTERING (K-Means)")
print("=" * 70)

# Features untuk clustering: Daya, Kapasitas, EER/CSPF
cluster_features = ['Daya (watt)', 'Kapasitas Pendinginan (BTU/h)', 'Nilai Efisiensi (EER/CSPF)']
X_cluster = df[cluster_features].dropna()

# Standardize
scaler_cluster = StandardScaler()
X_scaled = scaler_cluster.fit_transform(X_cluster)

# Elbow method + Silhouette score
k_range = range(2, 11)
inertias = []
silhouettes = []
for k in k_range:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    sil = silhouette_score(X_scaled, km.labels_)
    silhouettes.append(sil)
    print(f"  k={k}: inertia={km.inertia_:.2f}, silhouette={sil:.4f}")

# Plot elbow + silhouette
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Penentuan Jumlah Cluster Optimal (K-Means)', fontsize=14, fontweight='bold')

axes[0].plot(list(k_range), inertias, 'bo-', markersize=8)
axes[0].set_xlabel('Jumlah Cluster (k)')
axes[0].set_ylabel('Inertia (WCSS)')
axes[0].set_title('Elbow Method')

axes[1].plot(list(k_range), silhouettes, 'rs-', markersize=8)
axes[1].set_xlabel('Jumlah Cluster (k)')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Score')
axes[1].axhline(y=max(silhouettes), color='green', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('outputs/figures/M1_elbow_silhouette.png', bbox_inches='tight')
plt.close()
print("  [SAVED] M1_elbow_silhouette.png")

# Pilih k optimal (silhouette tertinggi)
k_optimal = list(k_range)[np.argmax(silhouettes)]
print(f"\n  k optimal (silhouette tertinggi): k={k_optimal}")

# Fit K-Means dengan k optimal
kmeans = KMeans(n_clusters=k_optimal, random_state=RANDOM_STATE, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Profil cluster
print(f"\n  Profil Cluster (k={k_optimal}):")
cluster_profile = df.groupby('cluster')[cluster_features + ['Rating Bintang (1-5)']].agg(['mean', 'median', 'std', 'count'])
print(cluster_profile.round(2).to_string())

# Cluster summary
cluster_summary = df.groupby('cluster').agg({
    'Daya (watt)': ['mean', 'median'],
    'Kapasitas Pendinginan (BTU/h)': ['mean', 'median'],
    'Nilai Efisiensi (EER/CSPF)': ['mean', 'median'],
    'Rating Bintang (1-5)': lambda x: x.mode().iloc[0] if len(x) > 0 else np.nan,
    'Tipe': lambda x: x.value_counts().index[0],
    'Merek': 'count',
}).round(2)
cluster_summary.columns = ['Daya_mean', 'Daya_median', 'Kap_mean', 'Kap_median',
                            'EER_mean', 'EER_median', 'Rating_mode', 'Tipe_mode', 'Count']
print(f"\n  Ringkasan Cluster:")
print(cluster_summary.to_string())
cluster_summary.to_csv('outputs/tables/15_cluster_profile.csv')

# Label interpretasi cluster
cluster_labels = {}
for c in sorted(df['cluster'].unique()):
    sub = df[df['cluster'] == c]
    med_daya = sub['Daya (watt)'].median()
    med_kap = sub['Kapasitas Pendinginan (BTU/h)'].median()
    med_eer = sub['Nilai Efisiensi (EER/CSPF)'].median()
    
    # Kategorikan berdasarkan median
    if med_eer >= 11:
        eff = 'Efisien'
    elif med_eer >= 8:
        eff = 'Sedang'
    else:
        eff = 'Kurang Efisien'
    
    if med_kap < 7000:
        cap = 'Kecil'
    elif med_kap < 14000:
        cap = 'Sedang'
    else:
        cap = 'Besar'
    
    cluster_labels[c] = f'{cap}-{eff}'

df['cluster_label'] = df['cluster'].map(cluster_labels)
print(f"\n  Label Cluster:")
for c, label in cluster_labels.items():
    n = (df['cluster'] == c).sum()
    print(f"    Cluster {c}: {label} (n={n})")

# Visualisasi cluster
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle(f'Hasil Clustering K-Means (k={k_optimal})', fontsize=14, fontweight='bold')

scatter1 = axes[0].scatter(df['Daya (watt)'], df['Kapasitas Pendinginan (BTU/h)'], 
                           c=df['cluster'], cmap='Set1', alpha=0.6, s=20)
axes[0].set_xlabel('Daya (watt)')
axes[0].set_ylabel('Kapasitas Pendinginan (BTU/h)')
axes[0].set_title('Daya vs Kapasitas')
plt.colorbar(scatter1, ax=axes[0], label='Cluster')

scatter2 = axes[1].scatter(df['Daya (watt)'], df['Nilai Efisiensi (EER/CSPF)'], 
                           c=df['cluster'], cmap='Set1', alpha=0.6, s=20)
axes[1].set_xlabel('Daya (watt)')
axes[1].set_ylabel('EER/CSPF')
axes[1].set_title('Daya vs EER/CSPF')
plt.colorbar(scatter2, ax=axes[1], label='Cluster')

plt.tight_layout()
plt.savefig('outputs/figures/M2_clusters.png', bbox_inches='tight')
plt.close()
print("  [SAVED] M2_clusters.png")

# 3D scatter
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(df['Daya (watt)'], df['Kapasitas Pendinginan (BTU/h)'], 
                     df['Nilai Efisiensi (EER/CSPF)'], c=df['cluster'], cmap='Set1', alpha=0.6, s=20)
ax.set_xlabel('Daya (watt)')
ax.set_ylabel('Kapasitas (BTU/h)')
ax.set_zlabel('EER/CSPF')
ax.set_title(f'3D Cluster Visualization (k={k_optimal})', fontsize=13, fontweight='bold')
plt.colorbar(scatter, ax=ax, label='Cluster', shrink=0.6)
plt.tight_layout()
plt.savefig('outputs/figures/M2b_clusters_3d.png', bbox_inches='tight')
plt.close()
print("  [SAVED] M2b_clusters_3d.png")

# ============================================================
# B. KLASIFIKASI RATING BINTANG
# ============================================================
print("\n" + "=" * 70)
print("B. KLASIFIKASI RATING BINTANG")
print("=" * 70)
print("  [LEAKAGE PREVENTION] Prediktor: Daya, Kapasitas, Tipe, Kategori_PK")
print("  DILARANG: EER/CSPF, Konsumsi, Biaya (derived dari Rating)")

# Features (tanpa leakage)
clf_features_num = ['Daya (watt)', 'Kapasitas Pendinginan (BTU/h)']
clf_features_cat = ['Tipe', 'Kategori_PK']

# One-hot encode kategorikal
X_clf = df[clf_features_num + clf_features_cat].copy()
X_clf = pd.get_dummies(X_clf, columns=clf_features_cat, drop_first=True)
y_clf = df['Rating Bintang (1-5)'].astype(int)

print(f"  Features: {list(X_clf.columns)}")
print(f"  Target: Rating Bintang (1-5)")
print(f"  Class distribution:")
print(y_clf.value_counts().sort_index().to_string())

# Train/test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=RANDOM_STATE, stratify=y_clf
)
print(f"\n  Train: {len(X_train)}, Test: {len(X_test)}")

# Standardize
scaler_clf = StandardScaler()
X_train_scaled = scaler_clf.fit_transform(X_train)
X_test_scaled = scaler_clf.transform(X_test)

# Models
models_clf = {
    'Random Forest': RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, max_depth=10),
    'KNN': KNeighborsClassifier(n_neighbors=7),
    'Decision Tree': DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=8),
    'Logistic Regression': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
clf_results = []

for name, model in models_clf.items():
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='f1_weighted')
    
    # Fit on full train
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    clf_results.append({
        'Model': name,
        'CV_F1_mean': cv_scores.mean(),
        'CV_F1_std': cv_scores.std(),
        'Test_Accuracy': acc,
        'Test_F1': f1,
    })
    print(f"\n  {name}:")
    print(f"    CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"    Test Accuracy: {acc:.4f}")
    print(f"    Test F1 (weighted): {f1:.4f}")

clf_results_df = pd.DataFrame(clf_results).sort_values('Test_F1', ascending=False)
clf_results_df.to_csv('outputs/tables/16_classification_results.csv', index=False)
print(f"\n  Ranking model:")
print(clf_results_df.to_string(index=False))

# Best model
best_clf_name = clf_results_df.iloc[0]['Model']
best_clf_model = models_clf[best_clf_name]
print(f"\n  Best model: {best_clf_name}")

# Confusion matrix
y_pred_best = best_clf_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred_best, labels=sorted(y_clf.unique()))

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=sorted(y_clf.unique()), yticklabels=sorted(y_clf.unique()),
            ax=ax, cbar_kws={'label': 'Count'})
ax.set_xlabel('Predicted Rating')
ax.set_ylabel('Actual Rating')
ax.set_title(f'Confusion Matrix — {best_clf_name}\n(Test Accuracy={clf_results_df.iloc[0]["Test_Accuracy"]:.4f})',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/figures/M3_confusion_matrix.png', bbox_inches='tight')
plt.close()
print("  [SAVED] M3_confusion_matrix.png")

# Classification report
clf_report = classification_report(y_test, y_pred_best, output_dict=True)
clf_report_df = pd.DataFrame(clf_report).T
clf_report_df.to_csv('outputs/tables/17_classification_report.csv')
print(f"\n  Classification Report ({best_clf_name}):")
print(classification_report(y_test, y_pred_best))

# Feature importance (Random Forest)
if hasattr(best_clf_model, 'feature_importances_'):
    importances = best_clf_model.feature_importances_
    feat_df = pd.DataFrame({'Feature': X_clf.columns, 'Importance': importances})
    feat_df = feat_df.sort_values('Importance', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(feat_df['Feature'], feat_df['Importance'], color='steelblue', edgecolor='white')
    ax.set_xlabel('Feature Importance')
    ax.set_title(f'Feature Importance — {best_clf_name}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/figures/M4_clf_feature_importance.png', bbox_inches='tight')
    plt.close()
    print("  [SAVED] M4_clf_feature_importance.png")
    
    feat_df.to_csv('outputs/tables/18_clf_feature_importance.csv', index=False)

# ============================================================
# C. REGRESI EER/CSPF
# ============================================================
print("\n" + "=" * 70)
print("C. REGRESI EER/CSPF")
print("=" * 70)
print("  [LEAKAGE PREVENTION] Prediktor: Daya, Kapasitas, Tipe, Kategori_PK")
print("  DILARANG: Konsumsi, Biaya, Rating (derived dari EER/CSPF)")

reg_features_num = ['Daya (watt)', 'Kapasitas Pendinginan (BTU/h)']
reg_features_cat = ['Tipe', 'Kategori_PK']

X_reg = df[reg_features_num + reg_features_cat].copy()
X_reg = pd.get_dummies(X_reg, columns=reg_features_cat, drop_first=True)
y_reg = df['Nilai Efisiensi (EER/CSPF)'].dropna()
X_reg = X_reg.loc[y_reg.index]

print(f"  Features: {list(X_reg.columns)}")
print(f"  Target: EER/CSPF")
print(f"  N samples: {len(y_reg)}")

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=RANDOM_STATE
)

scaler_reg = StandardScaler()
X_train_r_scaled = scaler_reg.fit_transform(X_train_r)
X_test_r_scaled = scaler_reg.transform(X_test_r)

models_reg = {
    'Random Forest': RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, max_depth=10),
    'Ridge': Ridge(alpha=1.0, random_state=RANDOM_STATE),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, random_state=RANDOM_STATE, max_depth=5),
    'Linear Regression': LinearRegression(),
}

cv_reg = RepeatedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE)
reg_results = []

for name, model in models_reg.items():
    cv_scores = cross_val_score(model, X_train_r_scaled, y_train_r, cv=cv_reg, scoring='r2')
    
    model.fit(X_train_r_scaled, y_train_r)
    y_pred_r = model.predict(X_test_r_scaled)
    
    r2 = r2_score(y_test_r, y_pred_r)
    rmse = np.sqrt(mean_squared_error(y_test_r, y_pred_r))
    mae = mean_absolute_error(y_test_r, y_pred_r)
    
    reg_results.append({
        'Model': name,
        'CV_R2_mean': cv_scores.mean(),
        'CV_R2_std': cv_scores.std(),
        'Test_R2': r2,
        'Test_RMSE': rmse,
        'Test_MAE': mae,
    })
    print(f"\n  {name}:")
    print(f"    CV R2: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"    Test R2: {r2:.4f}")
    print(f"    Test RMSE: {rmse:.4f}")
    print(f"    Test MAE: {mae:.4f}")

reg_results_df = pd.DataFrame(reg_results).sort_values('Test_R2', ascending=False)
reg_results_df.to_csv('outputs/tables/19_regression_results.csv', index=False)
print(f"\n  Ranking model:")
print(reg_results_df.to_string(index=False))

# Best regression model
best_reg_name = reg_results_df.iloc[0]['Model']
best_reg_model = models_reg[best_reg_name]
y_pred_best_r = best_reg_model.predict(X_test_r_scaled)
print(f"\n  Best model: {best_reg_name}")

# Actual vs Predicted scatter
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(y_test_r, y_pred_best_r, alpha=0.4, s=20, color='steelblue')
ax.plot([y_test_r.min(), y_test_r.max()], [y_test_r.min(), y_test_r.max()], 'r--', lw=2)
ax.set_xlabel('Actual EER/CSPF')
ax.set_ylabel('Predicted EER/CSPF')
ax.set_title(f'Actual vs Predicted — {best_reg_name}\n'
             f'R2={reg_results_df.iloc[0]["Test_R2"]:.4f}, '
             f'RMSE={reg_results_df.iloc[0]["Test_RMSE"]:.4f}',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/figures/M5_regression_actual_vs_pred.png', bbox_inches='tight')
plt.close()
print("  [SAVED] M5_regression_actual_vs_pred.png")

# Residual plot
residuals = y_test_r.values - y_pred_best_r
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y_pred_best_r, residuals, alpha=0.4, s=20, color='steelblue')
ax.axhline(y=0, color='red', linestyle='--')
ax.set_xlabel('Predicted EER/CSPF')
ax.set_ylabel('Residual')
ax.set_title(f'Residual Plot — {best_reg_name}', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/figures/M5b_residuals.png', bbox_inches='tight')
plt.close()
print("  [SAVED] M5b_residuals.png")

# Feature importance (RF)
if hasattr(best_reg_model, 'feature_importances_'):
    importances_r = best_reg_model.feature_importances_
    feat_df_r = pd.DataFrame({'Feature': X_reg.columns, 'Importance': importances_r})
    feat_df_r = feat_df_r.sort_values('Importance', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(feat_df_r['Feature'], feat_df_r['Importance'], color='coral', edgecolor='white')
    ax.set_xlabel('Feature Importance')
    ax.set_title(f'Feature Importance (Regression) — {best_reg_name}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/figures/M6_reg_feature_importance.png', bbox_inches='tight')
    plt.close()
    print("  [SAVED] M6_reg_feature_importance.png")
    feat_df_r.to_csv('outputs/tables/20_reg_feature_importance.csv', index=False)

# ============================================================
# D. PERBANDINGAN MODEL (BAR CHART)
# ============================================================
print("\n" + "=" * 70)
print("D. PERBANDINGAN MODEL")
print("=" * 70)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Perbandingan Performa Model Machine Learning', fontsize=14, fontweight='bold')

# Classification
clf_plot = clf_results_df.sort_values('Test_F1')
axes[0].barh(clf_plot['Model'], clf_plot['Test_F1'], color='steelblue', edgecolor='white')
axes[0].set_xlabel('Test F1 Score (weighted)')
axes[0].set_title('Klasifikasi Rating Bintang')
for i, v in enumerate(clf_plot['Test_F1']):
    axes[0].text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=9)

# Regression
reg_plot = reg_results_df.sort_values('Test_R2')
axes[1].barh(reg_plot['Model'], reg_plot['Test_R2'], color='coral', edgecolor='white')
axes[1].set_xlabel('Test R² Score')
axes[1].set_title('Regresi EER/CSPF')
for i, v in enumerate(reg_plot['Test_R2']):
    axes[1].text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('outputs/figures/M7_model_comparison.png', bbox_inches='tight')
plt.close()
print("  [SAVED] M7_model_comparison.png")

# ============================================================
# E. CLUSTER vs RATING ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("E. CLUSTER vs RATING ANALYSIS")
print("=" * 70)

ct_cluster = pd.crosstab(df['cluster'], df['Rating Bintang (1-5)'])
print(f"\n  Cross-tab Cluster x Rating:")
print(ct_cluster.to_string())

ct_cluster_tipe = pd.crosstab(df['cluster'], df['Tipe'])
print(f"\n  Cross-tab Cluster x Tipe:")
print(ct_cluster_tipe.to_string())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Distribusi Rating dan Tipe per Cluster', fontsize=14, fontweight='bold')

ct_cluster.plot(kind='bar', stacked=True, ax=axes[0], colormap='RdYlGn', edgecolor='white')
axes[0].set_xlabel('Cluster')
axes[0].set_ylabel('Count')
axes[0].set_title('Cluster x Rating')
axes[0].legend(title='Rating', bbox_to_anchor=(1.05, 1), loc='upper left')

ct_cluster_tipe.plot(kind='bar', stacked=True, ax=axes[1], color=['steelblue', 'coral'], edgecolor='white')
axes[1].set_xlabel('Cluster')
axes[1].set_ylabel('Count')
axes[1].set_title('Cluster x Tipe')
axes[1].legend(title='Tipe', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig('outputs/figures/M8_cluster_vs_rating.png', bbox_inches='tight')
plt.close()
print("  [SAVED] M8_cluster_vs_rating.png")

# ============================================================
# F. SIMPAN DATA DENGAN CLUSTER
# ============================================================
print("\n" + "=" * 70)
print("F. SIMPAN DATA DENGAN CLUSTER")
print("=" * 70)

output_ml = 'data/processed/ac_simebtke_ml.csv'
df.to_csv(output_ml, index=False, encoding='utf-8-sig')
print(f"  Disimpan: {output_ml} ({df.shape[0]} x {df.shape[1]})")

# ============================================================
# G. RINGKASAN MACHINE LEARNING
# ============================================================
print("\n" + "=" * 70)
print("G. RINGKASAN MACHINE LEARNING")
print("=" * 70)

ml_summary = f"""
1. CLUSTERING (K-Means, k={k_optimal})
   - Silhouette score: {max(silhouettes):.4f}
   - {k_optimal} cluster terbentuk dengan profil:
"""
for c, label in cluster_labels.items():
    n = (df['cluster'] == c).sum()
    ml_summary += f"     Cluster {c}: {label} (n={n})\n"

ml_summary += f"""
2. KLASIFIKASI RATING BINTANG
   - Best model: {best_clf_name}
   - Test Accuracy: {clf_results_df.iloc[0]['Test_Accuracy']:.4f}
   - Test F1 (weighted): {clf_results_df.iloc[0]['Test_F1']:.4f}
   - CV F1: {clf_results_df.iloc[0]['CV_F1_mean']:.4f} ± {clf_results_df.iloc[0]['CV_F1_std']:.4f}
   - [ASEMSI] Akurasi moderat karena prediktor (Daya, Kapasitas, Tipe) tidak
     sepenuhnya menentukan Rating. EER/CSPF dikecualikan untuk mencegah leakage.

3. REGRESI EER/CSPF
   - Best model: {best_reg_name}
   - Test R2: {reg_results_df.iloc[0]['Test_R2']:.4f}
   - Test RMSE: {reg_results_df.iloc[0]['Test_RMSE']:.4f}
   - Test MAE: {reg_results_df.iloc[0]['Test_MAE']:.4f}
   - [ASEMSI] R2 rendah karena EER/CSPF ditentukan oleh faktor teknis internal
     (kompresor, refrigeran, desain) yang tidak tersedia dalam dataset.

4. DATA LEAKAGE PREVENTION
   - Klasifikasi: prediktor = Daya, Kapasitas, Tipe, Kategori_PK (tanpa EER/CSPF, Konsumsi, Biaya)
   - Regresi: prediktor = Daya, Kapasitas, Tipe, Kategori_PK (tanpa Konsumsi, Biaya, Rating)
   - Semua derived variables dikecualikan dari prediktor
"""
print(ml_summary)

# Simpan summary
with open('outputs/tables/21_ml_summary.txt', 'w') as f:
    f.write(ml_summary)

print("\n" + "=" * 70)
print("TAHAP 4 SELESAI.")
print("Data ML: data/processed/ac_simebtke_ml.csv")
print("Visualisasi: outputs/figures/M1-M8")
print("Tabel: outputs/tables/15-21")
print("=" * 70)
