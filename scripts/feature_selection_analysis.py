"""
Analiza istotności cech — 4 metody + porównanie modeli przed/po usunięciu.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import json
import joblib
from pathlib import Path
from datetime import datetime
from scipy.stats import wilcoxon

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.inspection import permutation_importance
from scipy.stats import pearsonr
from sklearn.feature_selection import chi2
from sklearn.metrics import roc_auc_score

import xgboost as xgb
import lightgbm as lgbm

# ============================================================================
# Wczytanie danych
# ============================================================================
print("=" * 70)
print("ANALIZA ISTOTNOŚCI CECH — SYSTEM VASCULITIS XAI")
print("=" * 70)

X_raw = joblib.load('models/saved/X_train.joblib')
y_raw = joblib.load('models/saved/y_train.joblib')
with open('models/saved/feature_names.json') as f:
    features = json.load(f)

df = pd.DataFrame(X_raw, columns=features)
y = pd.Series(y_raw)

print(f"\nDane: {df.shape[0]} pacjentów, {df.shape[1]} cech")
print(f"Klasa 0 (przeżyli): {(y == 0).sum()}, Klasa 1 (zgon): {(y == 1).sum()}")

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

numeric_feats = ['Wiek_rozpoznania', 'Opoznienie_Rozpoznia', 'Liczba_Zajetych_Narzadow',
                 'Kreatynina', 'Czas_Sterydow', 'Eozynofilia_Krwi_Obwodowej_Wartosc']
binary_feats = [f for f in features if f not in numeric_feats]

# ============================================================================
# METODA 1 — Filter (Pearson + Chi²)
# ============================================================================
print("\n" + "=" * 70)
print("METODA 1 — FILTER (Pearson + Chi²)")
print("=" * 70)

pearson_scores, pearson_pvals = {}, {}
for f in numeric_feats:
    corr, pval = pearsonr(X_train[f], y_train)
    pearson_scores[f] = abs(corr)
    pearson_pvals[f] = pval

chi2_vals, chi2_pvals = chi2(X_train[binary_feats], y_train)
chi2_scores = dict(zip(binary_feats, chi2_vals))
chi2_pval_dict = dict(zip(binary_feats, chi2_pvals))

print("\n--- Numeric (Pearson) ---")
for f, v in sorted(pearson_scores.items(), key=lambda x: -x[1]):
    sig = "✓" if pearson_pvals[f] < 0.05 else "✗"
    print(f"  {f:<42} |r|={v:.4f}  p={pearson_pvals[f]:.4f}  {sig}")

print("\n--- Binary (Chi²) ---")
for f, v, p in sorted(zip(binary_feats, chi2_vals, chi2_pvals), key=lambda x: -x[1]):
    sig = "✓" if p < 0.05 else "✗"
    print(f"  {f:<42} chi²={v:>7.2f}  p={p:.4f}  {sig}")

insignificant_filter = [f for f in numeric_feats if pearson_pvals[f] > 0.05] + \
                       [f for f in binary_feats if chi2_pval_dict[f] > 0.05]
print(f"\nNiesignifikantne (p > 0.05): {insignificant_filter}")

# ============================================================================
# METODA 2 — RFECV (szybszy: 3-fold)
# ============================================================================
print("\n" + "=" * 70)
print("METODA 2 — RFECV (3-fold, RandomForest)")
print("=" * 70)

rf_rfe = RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced', n_jobs=-1)
rfecv = RFECV(rf_rfe, step=1, cv=StratifiedKFold(3, shuffle=True, random_state=42),
              scoring='roc_auc', min_features_to_select=3)
rfecv.fit(X_train, y_train)

print(f"\nOptymalna liczba cech: {rfecv.n_features_}")
ranking = pd.Series(rfecv.ranking_, index=features).sort_values()
for feat, rank in ranking.items():
    status = "✓ ZOSTAJE" if rank == 1 else f"  usuń (rank {rank})"
    print(f"  {feat:<42} {status}")

rfe_removed = [f for f, s in zip(features, rfecv.support_) if not s]

# ============================================================================
# METODA 3 — Feature Importance (3 modele)
# ============================================================================
print("\n" + "=" * 70)
print("METODA 3 — FEATURE IMPORTANCE (XGBoost, RF, LightGBM)")
print("=" * 70)

model_specs = {
    'XGBoost': xgb.XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=6,
                                  scale_pos_weight=5, random_state=42, eval_metric='auc',
                                  use_label_encoder=False, verbosity=0),
    'RF': RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced',
                                  random_state=42, n_jobs=-1),
    'LightGBM': lgbm.LGBMClassifier(n_estimators=200, learning_rate=0.1, num_leaves=31,
                                     is_unbalance=True, random_state=42, verbose=-1),
}

importance_dfs = {}
for name, model in model_specs.items():
    model.fit(X_train, y_train)
    fi = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    importance_dfs[name] = fi
    print(f"\n--- {name} ---")
    for feat, val in fi.items():
        bar = "█" * int(val / fi.max() * 30)
        print(f"  {feat:<42} {val:.4f}  {bar}")

# ============================================================================
# METODA 4 — Permutation Importance
# ============================================================================
print("\n" + "=" * 70)
print("METODA 4 — PERMUTATION IMPORTANCE")
print("=" * 70)

rf_perm = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced',
                                  random_state=42, n_jobs=-1)
rf_perm.fit(X_train, y_train)

perm = permutation_importance(rf_perm, X_test, y_test, n_repeats=20,
                               random_state=42, scoring='roc_auc')

perm_df = pd.DataFrame({
    'feature': features,
    'mean_drop': perm.importances_mean,
    'std': perm.importances_std
}).sort_values('mean_drop', ascending=False)

for _, row in perm_df.iterrows():
    status = "⚠ ZBĘNA" if row['mean_drop'] <= 0 else ""
    print(f"  {row['feature']:<42} {row['mean_drop']:+.4f} ± {row['std']:.4f}  {status}")

perm_redundant = perm_df[perm_df['mean_drop'] <= 0]['feature'].tolist()

# ============================================================================
# AGREGACJA
# ============================================================================
print("\n" + "=" * 70)
print("AGREGACJA — KOŃCOWY RANKING")
print("=" * 70)

results = pd.DataFrame({'feature': features})

# Filter norm
filt = []
for f in features:
    filt.append(pearson_scores.get(f, chi2_scores.get(f, 0)))
results['filter'] = filt
results['filter_n'] = results['filter'] / results['filter'].max()

# RFECV
results['rfe'] = rfecv.support_.astype(int)

# FI average (normalized per model, then averaged)
for name in model_specs:
    fi = importance_dfs[name]
    results[f'fi_{name}_n'] = [fi[f] / fi.max() for f in features]
results['fi_avg_n'] = results[[f'fi_{n}_n' for n in model_specs]].mean(axis=1)

# Permutation norm
results['perm'] = [perm_df.loc[perm_df['feature'] == f, 'mean_drop'].values[0] for f in features]
results['perm_n'] = results['perm'].clip(0) / results['perm'].clip(0).max().clip(1e-9)

# Final average
results['avg'] = results[['filter_n', 'rfe', 'fi_avg_n', 'perm_n']].mean(axis=1)
results = results.sort_values('avg', ascending=False).reset_index(drop=True)

# Votes for removal (≥3 out of 4 methods agree)
removal_votes = {}
for f in features:
    votes = 0
    if f in insignificant_filter: votes += 1
    if not rfecv.support_[features.index(f)]: votes += 1
    if all(importance_dfs[n][f] < importance_dfs[n].median() for n in model_specs): votes += 1
    if f in perm_redundant: votes += 1
    removal_votes[f] = votes

features_to_remove = [f for f, v in removal_votes.items() if v >= 3]
features_to_keep = [f for f in features if f not in features_to_remove]

print(f"\nGłosy za usunięciem (max 4):")
for f in sorted(removal_votes, key=removal_votes.get, reverse=True):
    if removal_votes[f] >= 2:
        marker = "⚠ USUŃ" if removal_votes[f] >= 3 else "? rozważ"
        print(f"  {f:<42} {removal_votes[f]}/4  {marker}")

print(f"\n{'─'*70}")
print(f"USUNIĘTE ({len(features_to_remove)}):")
for f in features_to_remove:
    print(f"  ✗ {f}  (głosy: {removal_votes[f]}/4)")
print(f"\nZACHOWANE ({len(features_to_keep)}):")
for _, row in results.iterrows():
    if row['feature'] in features_to_keep:
        print(f"  ✓ {row['feature']:<42} avg={row['avg']:.3f}")

# ============================================================================
# TEST — 100 bootstrapów przed/po
# ============================================================================
print("\n" + "=" * 70)
print(f"TEST: {len(features)} cech vs {len(features_to_keep)} cech — 100 bootstrapów")
print("=" * 70)

test_model_specs = {
    'XGBoost': lambda: xgb.XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=6,
                                          scale_pos_weight=5, random_state=42, eval_metric='auc',
                                          use_label_encoder=False, verbosity=0),
    'Random Forest': lambda: RandomForestClassifier(n_estimators=200, max_depth=10,
                                                      class_weight='balanced', random_state=42, n_jobs=-1),
    'LightGBM': lambda: lgbm.LGBMClassifier(n_estimators=200, learning_rate=0.1, num_leaves=31,
                                              is_unbalance=True, random_state=42, verbose=-1),
}

# Pre-train all models on full + reduced features
trained_full = {}
trained_red = {}
for name, fn in test_model_specs.items():
    m = fn()
    m.fit(X_train, y_train)
    trained_full[name] = m
    m2 = fn()
    m2.fit(X_train[features_to_keep], y_train)
    trained_red[name] = m2

N_BOOTSTRAP = 100
rng = np.random.RandomState(42)

for name in test_model_specs:
    aucs_full, aucs_red = [], []
    for _ in range(N_BOOTSTRAP):
        idx = rng.choice(len(X_test), size=len(X_test), replace=True)
        X_bs, y_bs = X_test.iloc[idx], y_test.iloc[idx]
        if y_bs.nunique() < 2:
            continue

        p_full = trained_full[name].predict_proba(X_bs)[:, 1]
        p_red = trained_red[name].predict_proba(X_bs[features_to_keep])[:, 1]
        aucs_full.append(roc_auc_score(y_bs, p_full))
        aucs_red.append(roc_auc_score(y_bs, p_red))

    mf, sf = np.mean(aucs_full), np.std(aucs_full)
    mr, sr = np.mean(aucs_red), np.std(aucs_red)
    diff = mr - mf
    
    try:
        _, pval = wilcoxon(aucs_full, aucs_red)
        sig = "ISTOTNA" if pval < 0.05 else "nieistotna"
        pval_str = f"p={pval:.4f} ({sig})"
    except:
        pval_str = "—"

    verdict = "✓ OK" if abs(diff) < 0.01 else ("↑ LEPSZY" if diff > 0 else "↓ GORSZY")
    print(f"\n  {name}:")
    print(f"    Pełny     ({len(features):>2} cech): AUC = {mf:.4f} ± {sf:.4f}")
    print(f"    Zredukowany ({len(features_to_keep):>2} cech): AUC = {mr:.4f} ± {sr:.4f}")
    print(f"    Różnica: {diff:+.4f}  {verdict}  | Wilcoxon {pval_str}")

# ============================================================================
# ZAPISZ
# ============================================================================
output = {
    'timestamp': datetime.now().isoformat(),
    'n_patients': int(df.shape[0]),
    'n_features_original': len(features),
    'n_features_kept': len(features_to_keep),
    'features_removed': features_to_remove,
    'features_kept': features_to_keep,
    'removal_votes': removal_votes,
    'ranking': [{ 'feature': r['feature'], 'avg_importance': round(r['avg'], 4) }
                for _, r in results.iterrows()],
}
with open('models/saved/feature_selection_results.json', 'w') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"\nZapisano: models/saved/feature_selection_results.json")
