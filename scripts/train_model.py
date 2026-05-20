#!/usr/bin/env python3
"""
Skrypt treningowy v3 — Nested CV + Optuna + Stacking + CatBoost.

Pipeline metodologiczny (opcja C — pełny pakiet poprawy AUC):
  ✅ Pełny pipeline z SelectKBest k=30 WEWNĄTRZ — brak data leakage cech
  ✅ Nested CV dla 8 modeli (7 bazowych + CatBoost):
       - Outer: 5×2 RepeatedStratifiedKFold (10 foldów) → nieobciążona ocena
       - Inner: 3-fold StratifiedKFold w RandomizedSearchCV(n_iter=20)
  ✅ Final tuning per model przez Optuna (TPESampler multivariate, n_trials=50)
       — Bayesian optimization, lepsze pokrycie przestrzeni niż random
  ✅ StackingClassifier (RF+GB+XGB+SVM, meta=LR) z dotunowanymi sub-modeli
       — oceniany przez 5-fold CV na pre-procesowanych danych
  ✅ KNN imputer jako alternatywa median (porównawczo dla best modelu)
  ✅ scale_pos_weight tuningowany dla XGBoost (obecnie hardkodowane 5)
  ✅ Bootstrap 95% CI na hold-out test set (2000 iter)
  ✅ Test set 20% — TYLKO raportowanie końcowe, nie selekcja

Użycie:
    python scripts/train_model.py
"""

import sys
import json
import time
import warnings
import argparse
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, RepeatedStratifiedKFold,
    RandomizedSearchCV, cross_val_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score, recall_score, f1_score, confusion_matrix
)

import optuna

from src.models.config import get_model_class, get_base_params
from src.models.evaluate import ModelEvaluator

# Suppress optuna logs to keep output readable
optuna.logging.set_verbosity(optuna.logging.WARNING)

# 7 bazowych + CatBoost = 8 modeli przez nested CV
BASE_MODEL_TYPES = [
    'logistic_regression', 'random_forest', 'lightgbm', 'xgboost',
    'gradient_boosting', 'svm', 'neural_network', 'catboost',
]

# Stacking = 9. model, oceniany osobno (wymaga dotuned sub-modeli)
STACKING_BASE = ['random_forest', 'gradient_boosting', 'xgboost', 'svm']

MODEL_FILENAMES = {
    'xgboost': 'best_model.joblib',  # KONTRAKT z API
    'random_forest': 'random_forest_model.joblib',
    'lightgbm': 'lightgbm_model.joblib',
    'gradient_boosting': 'gradient_boosting_model.joblib',
    'logistic_regression': 'logistic_regression_model.joblib',
    'svm': 'svm_model.joblib',
    'neural_network': 'neural_network_model.joblib',
    'catboost': 'catboost_model.joblib',
    'stacking': 'stacking_model.joblib',
}

# CatBoost lokalna konfiguracja (nie w config.py żeby nie zmieniać API)
CATBOOST_BASE_PARAMS = {
    'verbose': False,
    'random_state': 42,
    'auto_class_weights': 'Balanced',
    'thread_count': -1,
    'iterations': 200,
    'allow_writing_files': False,
}

EXCLUDE_COLUMNS = {'Kod', 'ID', 'id', 'Patient_ID', 'Zgon'}
EXCLUDE_PREFIXES = ('Leczenie_', 'Biops_')
MAX_MISSING_RATIO = 0.50
# v4: Empirycznie wykazano że k=71 (wszystkie cechy) bije k=30 o ~0.015 CV AUC
# (zob. ablation w docs). SelectKBest pozostaje w pipeline dla zachowania
# struktury, ale efektywnie nie usuwa cech.
N_SELECT_FEATURES = 71

OUTER_N_SPLITS = 5
OUTER_N_REPEATS = 2
INNER_N_SPLITS = 3
RANDOM_SEARCH_N_ITER = 20
OPTUNA_N_TRIALS = 100  # v4: zwiększone z 50 dla lepszego pokrycia
OPTUNA_TIMEOUT = 600  # 10 min per model max (większy budżet dla lightgbm/catboost)
RANDOM_STATE = 42

BOOTSTRAP_N_ITERATIONS = 2000
BOOTSTRAP_CONFIDENCE = 0.95

SINGLE_BEST_TARGET_AUC = 0.90
SINGLE_BEST_MODEL_TYPES = ('catboost', 'random_forest', 'xgboost')
SINGLE_BEST_N_TRIALS = 80
SINGLE_BEST_TIMEOUT = 1800
SINGLE_BEST_N_SPLITS = 5
SINGLE_BEST_N_REPEATS = 2
SINGLE_BEST_K_OPTIONS = [30, 50, 80, 'all']
SINGLE_BEST_FEATURE_SCOPE = 'all_clinical'

# Siatki dla nested CV (RandomizedSearchCV) — szybkie raportowanie
TUNING_GRIDS = {
    'logistic_regression': {
        'clf__C': [0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
        'clf__penalty': ['l2'],
        'clf__solver': ['lbfgs', 'liblinear'],
    },
    'random_forest': {
        'clf__n_estimators': [100, 200, 300],
        'clf__max_depth': [5, 10, 15, None],
        'clf__min_samples_split': [2, 5, 10],
        'clf__min_samples_leaf': [1, 2, 4],
        'clf__max_features': ['sqrt', 'log2'],
    },
    'lightgbm': {
        'clf__n_estimators': [100, 200, 300],
        'clf__num_leaves': [15, 31, 63],
        'clf__learning_rate': [0.05, 0.1],
        'clf__min_child_samples': [10, 20, 30],
        'clf__subsample': [0.7, 0.8, 1.0],
    },
    'xgboost': {
        'clf__n_estimators': [100, 200, 300],
        'clf__max_depth': [3, 5, 7],
        'clf__learning_rate': [0.05, 0.1],
        'clf__subsample': [0.7, 0.8, 1.0],
        'clf__colsample_bytree': [0.7, 0.8, 1.0],
        'clf__scale_pos_weight': [3.0, 3.76, 5.0],
    },
    'gradient_boosting': {
        'clf__n_estimators': [100, 200],
        'clf__max_depth': [3, 5, 7],
        'clf__learning_rate': [0.05, 0.1],
        'clf__subsample': [0.7, 0.8, 1.0],
        'clf__min_samples_split': [2, 5, 10],
    },
    'svm': {
        'clf__C': [0.1, 0.5, 1.0, 5.0],
        'clf__gamma': ['scale', 'auto', 0.01, 0.1],
        'clf__kernel': ['rbf'],
    },
    'neural_network': {
        'clf__hidden_layer_sizes': [(50,), (100,), (50, 25), (100, 50)],
        'clf__alpha': [0.0001, 0.001, 0.01],
        'clf__learning_rate_init': [0.001, 0.005, 0.01],
    },
    'catboost': {
        'clf__iterations': [100, 200, 300],
        'clf__depth': [4, 6, 8],
        'clf__learning_rate': [0.05, 0.1],
        'clf__l2_leaf_reg': [1.0, 3.0, 5.0],
    },
}


def _select_candidate_features(df: pd.DataFrame) -> list:
    candidates = []
    for col in df.columns:
        if col in EXCLUDE_COLUMNS or any(col.startswith(p) for p in EXCLUDE_PREFIXES):
            continue
        if df[col].dtype == object:
            continue
        if (df[col] == -1).sum() / len(df) > MAX_MISSING_RATIO:
            continue
        if df[col].replace(-1, np.nan).dropna().nunique() < 2:
            continue
        candidates.append(col)
    return candidates


def _make_classifier(model_type: str, params: dict = None):
    """Factory dla classifierów — obsługuje też CatBoost."""
    params = params or {}
    if model_type == 'catboost':
        from catboost import CatBoostClassifier
        merged = {**CATBOOST_BASE_PARAMS, **params}
        return CatBoostClassifier(**merged)
    base = get_base_params(model_type)
    base.update(params)
    return get_model_class(model_type)(**base)


def _build_full_pipeline(model_type: str, n_features_total: int, imputer_strategy: str = 'median') -> Pipeline:
    """Pełny pipeline: imputer → scaler → SelectKBest → classifier."""
    imputer = (
        KNNImputer(n_neighbors=5) if imputer_strategy == 'knn'
        else SimpleImputer(strategy='median')
    )
    classifier = _make_classifier(model_type)
    return Pipeline([
        ('imputer', imputer),
        ('scaler', StandardScaler()),
        ('selector', SelectKBest(f_classif, k=min(N_SELECT_FEATURES, n_features_total))),
        ('clf', classifier),
    ])


def _n_iter_for(model_type: str) -> int:
    grid = TUNING_GRIDS[model_type]
    return min(RANDOM_SEARCH_N_ITER, int(np.prod([len(v) for v in grid.values()])))


def _to_jsonable(value):
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _nested_cv_evaluate(model_type: str, X: np.ndarray, y: np.ndarray, n_features_total: int) -> dict:
    """Nested CV: outer dla oceny, inner (RandomizedSearchCV) dla tuningu."""
    outer_cv = RepeatedStratifiedKFold(
        n_splits=OUTER_N_SPLITS, n_repeats=OUTER_N_REPEATS, random_state=RANDOM_STATE
    )
    inner_cv = StratifiedKFold(n_splits=INNER_N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    n_iter = _n_iter_for(model_type)
    n_folds = OUTER_N_SPLITS * OUTER_N_REPEATS

    print(f"\n  ── Nested CV: {model_type} ({n_folds} outer × {INNER_N_SPLITS} inner × {n_iter} iter) ──", flush=True)
    t0 = time.time()

    fold_aucs, fold_sens, fold_spec, fold_f1, fold_params = [], [], [], [], []

    for fold_idx, (tr_idx, te_idx) in enumerate(outer_cv.split(X, y)):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        try:
            search = RandomizedSearchCV(
                _build_full_pipeline(model_type, n_features_total),
                param_distributions=TUNING_GRIDS[model_type],
                n_iter=n_iter, cv=inner_cv, scoring='roc_auc',
                n_jobs=-1, random_state=RANDOM_STATE, refit=True,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                search.fit(X_tr, y_tr)

            y_proba = search.predict_proba(X_te)[:, 1]
            y_pred = search.predict(X_te)
            fold_aucs.append(float(roc_auc_score(y_te, y_proba)))
            fold_sens.append(float(recall_score(y_te, y_pred, zero_division=0)))
            tn, fp, fn, tp = confusion_matrix(y_te, y_pred).ravel()
            fold_spec.append(float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0)
            fold_f1.append(float(f1_score(y_te, y_pred, zero_division=0)))
            fold_params.append({k: _to_jsonable(v) for k, v in search.best_params_.items()})
        except Exception as e:
            print(f"      Fold {fold_idx + 1} BŁĄD: {e}", flush=True)

    elapsed = time.time() - t0
    if fold_aucs:
        print(
            f"      AUC={np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}  "
            f"Sens={np.mean(fold_sens):.4f}  Spec={np.mean(fold_spec):.4f}  "
            f"({elapsed:.1f}s)", flush=True
        )

    return {
        'auc_roc': {'mean': float(np.mean(fold_aucs)) if fold_aucs else 0.0,
                    'std': float(np.std(fold_aucs)) if fold_aucs else 0.0,
                    'values': fold_aucs},
        'sensitivity': {'mean': float(np.mean(fold_sens)) if fold_sens else 0.0,
                        'std': float(np.std(fold_sens)) if fold_sens else 0.0,
                        'values': fold_sens},
        'specificity': {'mean': float(np.mean(fold_spec)) if fold_spec else 0.0,
                        'std': float(np.std(fold_spec)) if fold_spec else 0.0,
                        'values': fold_spec},
        'f1': {'mean': float(np.mean(fold_f1)) if fold_f1 else 0.0,
               'std': float(np.std(fold_f1)) if fold_f1 else 0.0,
               'values': fold_f1},
        'best_params_per_fold': fold_params,
        'n_folds': len(fold_aucs),
        'elapsed_seconds': elapsed,
    }


# ──────────────────────────────────────────────────────────────
# OPTUNA — search spaces per model
# ──────────────────────────────────────────────────────────────
def _suggest_params(trial: 'optuna.Trial', model_type: str) -> dict:
    if model_type == 'random_forest':
        return {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
            'max_depth': trial.suggest_categorical('max_depth', [4, 6, 8, 10, 15, 20, None]),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
        }
    if model_type == 'xgboost':
        # v4: rozszerzone zakresy zgodnie z ablation — płytkie + dużo drzew + niskie lr
        return {
            'n_estimators': trial.suggest_int('n_estimators', 200, 600, step=50),
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0.0, 0.5),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 5.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 5.0, log=True),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 2.0, 5.0),
        }
    if model_type == 'lightgbm':
        return {
            'n_estimators': trial.suggest_int('n_estimators', 200, 500, step=50),
            'num_leaves': trial.suggest_int('num_leaves', 8, 64),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 5.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 5.0, log=True),
        }
    if model_type == 'gradient_boosting':
        return {
            'n_estimators': trial.suggest_int('n_estimators', 100, 400, step=50),
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.15, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        }
    if model_type == 'logistic_regression':
        return {
            'C': trial.suggest_float('C', 1e-3, 1e2, log=True),
            'solver': trial.suggest_categorical('solver', ['lbfgs', 'liblinear']),
            'penalty': 'l2',
        }
    if model_type == 'svm':
        return {
            'C': trial.suggest_float('C', 0.01, 10.0, log=True),
            'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
            'kernel': 'rbf',
        }
    if model_type == 'neural_network':
        n_layers = trial.suggest_int('n_layers', 1, 3)
        layers = tuple(trial.suggest_int(f'n_units_l{i}', 25, 150, step=25) for i in range(n_layers))
        return {
            'hidden_layer_sizes': layers,
            'alpha': trial.suggest_float('alpha', 1e-5, 1e-1, log=True),
            'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-4, 1e-2, log=True),
            'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
        }
    if model_type == 'catboost':
        return {
            'iterations': trial.suggest_int('iterations', 100, 500, step=50),
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0, log=True),
            'border_count': trial.suggest_int('border_count', 32, 255),
        }
    raise ValueError(f"Unknown model_type: {model_type}")


def _optuna_tune(model_type: str, X_train_proc: np.ndarray, y_train: np.ndarray) -> tuple:
    """Optuna TPE z multivariate sampling, MedianPruner."""
    inner_cv = StratifiedKFold(n_splits=INNER_N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial):
        params = _suggest_params(trial, model_type)
        clf = _make_classifier(model_type, params)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = cross_val_score(
                clf, X_train_proc, y_train, cv=inner_cv,
                scoring='roc_auc', n_jobs=-1
            )
        return float(np.mean(scores))

    sampler = optuna.samplers.TPESampler(
        seed=RANDOM_STATE, multivariate=True, n_startup_trials=10
    )
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(
        objective, n_trials=OPTUNA_N_TRIALS, timeout=OPTUNA_TIMEOUT,
        show_progress_bar=False, n_jobs=1,
    )

    best_params = study.best_params
    # Reconstruct hidden_layer_sizes for NN
    if model_type == 'neural_network':
        n_layers = best_params['n_layers']
        layers = tuple(best_params[f'n_units_l{i}'] for i in range(n_layers))
        best_params_clean = {
            'hidden_layer_sizes': layers,
            'alpha': best_params['alpha'],
            'learning_rate_init': best_params['learning_rate_init'],
            'activation': best_params['activation'],
        }
    elif model_type == 'logistic_regression':
        best_params_clean = {**best_params, 'penalty': 'l2'}
    elif model_type == 'svm':
        best_params_clean = {**best_params, 'kernel': 'rbf'}
    else:
        best_params_clean = best_params

    final_clf = _make_classifier(model_type, best_params_clean)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final_clf.fit(X_train_proc, y_train)

    return final_clf, best_params_clean, float(study.best_value), len(study.trials)


def _build_stacking(tuned_models: dict) -> StackingClassifier:
    """StackingClassifier z dotunowanymi sub-modeli."""
    estimators = [(name, clone(tuned_models[name])) for name in STACKING_BASE if name in tuned_models]
    return StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced'),
        cv=5, n_jobs=-1, stack_method='predict_proba', passthrough=False,
    )


def _save_background_data(models_dir, X_train, y_train, X_test, y_test):
    import joblib
    np.random.seed(RANDOM_STATE)
    n_bg = min(100, len(X_train))
    bg_idx = np.random.choice(len(X_train), n_bg, replace=False)
    joblib.dump(X_train[bg_idx], models_dir / "X_background.joblib")
    joblib.dump(y_train[bg_idx], models_dir / "y_background.joblib")
    joblib.dump(X_train, models_dir / "X_train.joblib")
    joblib.dump(y_train, models_dir / "y_train.joblib")
    joblib.dump(X_test, models_dir / "X_test.joblib")
    joblib.dump(y_test, models_dir / "y_test.joblib")


# ──────────────────────────────────────────────────────────────
# SINGLE BEST MODEL — all clinical features, one non-ensemble model
# ──────────────────────────────────────────────────────────────
def _select_all_clinical_feature_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Return all numeric clinical features except identifiers and target."""
    columns = {}

    for col in df.columns:
        if col in EXCLUDE_COLUMNS:
            continue

        values = pd.to_numeric(df[col], errors='coerce').replace(-1, np.nan)
        if values.dropna().nunique() < 2:
            continue
        columns[col] = values

    feature_df = pd.DataFrame(columns, index=df.index)
    return feature_df, feature_df.columns.tolist()


def _make_single_best_pipeline(
    model_type: str,
    model_params: dict,
    n_features_total: int,
    k_features,
    imputer_strategy: str,
) -> Pipeline:
    if imputer_strategy == 'knn':
        imputer = KNNImputer(n_neighbors=5)
    else:
        imputer = SimpleImputer(strategy='median')

    selector_k = 'all' if k_features == 'all' else min(int(k_features), n_features_total)
    return Pipeline([
        ('imputer', imputer),
        ('scaler', StandardScaler()),
        ('selector', SelectKBest(f_classif, k=selector_k)),
        ('clf', _make_classifier(model_type, model_params)),
    ])


def _suggest_single_best_params(trial: 'optuna.Trial', model_type: str) -> tuple[dict, object, str]:
    k_features = trial.suggest_categorical('k_features', SINGLE_BEST_K_OPTIONS)
    imputer_strategy = trial.suggest_categorical('imputer', ['median', 'knn'])

    if model_type == 'catboost':
        bootstrap_type = trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli'])
        params = {
            'iterations': trial.suggest_int('iterations', 250, 900, step=50),
            'depth': trial.suggest_int('depth', 3, 9),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.15, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.5, 20.0, log=True),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'random_strength': trial.suggest_float('random_strength', 0.0, 5.0),
            'bootstrap_type': bootstrap_type,
            'rsm': trial.suggest_float('rsm', 0.45, 1.0),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 30),
            'auto_class_weights': trial.suggest_categorical(
                'auto_class_weights', ['Balanced', 'SqrtBalanced']
            ),
        }
        if bootstrap_type == 'Bayesian':
            params['bagging_temperature'] = trial.suggest_float('bagging_temperature', 0.0, 2.0)
        else:
            params['subsample'] = trial.suggest_float('subsample', 0.55, 1.0)
    elif model_type == 'random_forest':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 300, 900, step=100),
            'max_depth': trial.suggest_categorical('max_depth', [6, 8, 10, 12, 15, 20, None]),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'class_weight': trial.suggest_categorical(
                'class_weight', ['balanced', 'balanced_subsample']
            ),
        }
    elif model_type == 'xgboost':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 250, 900, step=50),
            'max_depth': trial.suggest_int('max_depth', 2, 7),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.12, log=True),
            'subsample': trial.suggest_float('subsample', 0.55, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.45, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 15),
            'gamma': trial.suggest_float('gamma', 0.0, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 2.0, 6.0),
        }
    else:
        raise ValueError(f"Unsupported single-best model type: {model_type}")

    return params, k_features, imputer_strategy


def _evaluate_single_best_pipeline(
    pipeline: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    cv: RepeatedStratifiedKFold,
) -> dict:
    aucs, auc_prs, sensitivities, specificities, f1s = [], [], [], [], []

    for train_idx, test_idx in cv.split(X, y):
        X_train_fold, X_test_fold = X[train_idx], X[test_idx]
        y_train_fold, y_test_fold = y[train_idx], y[test_idx]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipeline.fit(X_train_fold, y_train_fold)
            y_proba = pipeline.predict_proba(X_test_fold)[:, 1]
            y_pred = pipeline.predict(X_test_fold)

        tn, fp, fn, tp = confusion_matrix(y_test_fold, y_pred, labels=[0, 1]).ravel()
        aucs.append(float(roc_auc_score(y_test_fold, y_proba)))
        auc_prs.append(float(average_precision_score(y_test_fold, y_proba)))
        sensitivities.append(float(tp / (tp + fn)) if (tp + fn) else 0.0)
        specificities.append(float(tn / (tn + fp)) if (tn + fp) else 0.0)
        f1s.append(float(f1_score(y_test_fold, y_pred, zero_division=0)))

    return {
        'auc_roc_mean': float(np.mean(aucs)),
        'auc_roc_std': float(np.std(aucs)),
        'cv_fold_values': aucs,
        'auc_pr_mean': float(np.mean(auc_prs)),
        'auc_pr_std': float(np.std(auc_prs)),
        'sensitivity_mean': float(np.mean(sensitivities)),
        'sensitivity_std': float(np.std(sensitivities)),
        'specificity_mean': float(np.mean(specificities)),
        'specificity_std': float(np.std(specificities)),
        'f1_mean': float(np.mean(f1s)),
        'f1_std': float(np.std(f1s)),
    }


def _tune_single_best_candidate(
    model_type: str,
    X: np.ndarray,
    y: np.ndarray,
    n_features_total: int,
    n_trials: int,
    timeout: int,
) -> dict:
    cv = RepeatedStratifiedKFold(
        n_splits=SINGLE_BEST_N_SPLITS,
        n_repeats=SINGLE_BEST_N_REPEATS,
        random_state=RANDOM_STATE,
    )

    def objective(trial):
        params, k_features, imputer_strategy = _suggest_single_best_params(trial, model_type)
        pipeline = _make_single_best_pipeline(
            model_type, params, n_features_total, k_features, imputer_strategy
        )
        metrics = _evaluate_single_best_pipeline(pipeline, X, y, cv)
        trial.set_user_attr('metrics', metrics)
        trial.set_user_attr('model_params', params)
        trial.set_user_attr('k_features', k_features)
        trial.set_user_attr('imputer', imputer_strategy)
        return metrics['auc_roc_mean']

    sampler = optuna.samplers.TPESampler(
        seed=RANDOM_STATE,
        multivariate=True,
        n_startup_trials=min(12, max(3, n_trials // 4)),
        warn_independent_sampling=False,
    )
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

    best_trial = study.best_trial
    return {
        'model_type': model_type,
        'best_value': float(study.best_value),
        'best_params': best_trial.user_attrs['model_params'],
        'k_features': best_trial.user_attrs['k_features'],
        'imputer': best_trial.user_attrs['imputer'],
        'metrics': best_trial.user_attrs['metrics'],
        'n_trials': len(study.trials),
    }


def run_single_best_mode(n_trials: int, timeout: int, target_auc: float) -> bool:
    import joblib

    print("=" * 72, flush=True)
    print("Single best model optimization — target CV AUC >= 0.90", flush=True)
    print("=" * 72, flush=True)

    data_path = project_root / "data" / "raw" / "aktualne_dane.csv"
    models_dir = project_root / "models" / "saved"
    models_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path, sep='|')
    X_df, all_feature_names = _select_all_clinical_feature_frame(df)
    y = pd.to_numeric(df['Zgon'], errors='raise').values.astype(int)
    X = X_df.values.astype(np.float64)
    n_features_total = X.shape[1]

    print(
        f"Features: {n_features_total} all-clinical numeric columns | "
        f"Patients: {len(y)} | Positives: {int(y.sum())}",
        flush=True,
    )

    candidate_results = {}
    best_result = None

    for idx, model_type in enumerate(SINGLE_BEST_MODEL_TYPES):
        remaining_timeout = timeout if idx == 0 else max(300, timeout // 2)
        print(
            f"\nTuning {model_type}: trials={n_trials}, timeout={remaining_timeout}s",
            flush=True,
        )
        result = _tune_single_best_candidate(
            model_type, X, y, n_features_total, n_trials, remaining_timeout
        )
        candidate_results[model_type] = result
        metrics = result['metrics']
        print(
            f"  AUC={metrics['auc_roc_mean']:.4f} ± {metrics['auc_roc_std']:.4f} | "
            f"PR-AUC={metrics['auc_pr_mean']:.4f} | "
            f"Sens={metrics['sensitivity_mean']:.4f} | Spec={metrics['specificity_mean']:.4f}",
            flush=True,
        )

        if best_result is None or result['best_value'] > best_result['best_value']:
            best_result = result

        if result['best_value'] >= target_auc:
            print(f"  Target reached by {model_type}; skipping fallback models.", flush=True)
            break

    assert best_result is not None

    final_pipeline = _make_single_best_pipeline(
        best_result['model_type'],
        best_result['best_params'],
        n_features_total,
        best_result['k_features'],
        best_result['imputer'],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final_pipeline.fit(X, y)

    selector = final_pipeline.named_steps['selector']
    selected_mask = selector.get_support()
    selected_features = [
        name for name, selected in zip(all_feature_names, selected_mask) if selected
    ]

    joblib.dump(final_pipeline, models_dir / "best_single_model.joblib")
    rng = np.random.RandomState(RANDOM_STATE)
    bg_idx = rng.choice(len(X), min(100, len(X)), replace=False)
    joblib.dump(X[bg_idx], models_dir / "best_single_X_background.joblib")
    joblib.dump(X, models_dir / "best_single_X_train.joblib")
    joblib.dump(y, models_dir / "best_single_y_train.joblib")
    with open(models_dir / "best_single_model_features.json", 'w', encoding='utf-8') as f:
        json.dump(all_feature_names, f, ensure_ascii=False, indent=2)

    now = datetime.now().isoformat()
    best_metadata = {
        'timestamp': now,
        'model_type': best_result['model_type'],
        'artifact': 'best_single_model.joblib',
        'feature_names_path': 'best_single_model_features.json',
        'feature_scope': SINGLE_BEST_FEATURE_SCOPE,
        'target_auc': target_auc,
        'target_reached': best_result['best_value'] >= target_auc,
        'auc_roc_mean': best_result['metrics']['auc_roc_mean'],
        'auc_roc_std': best_result['metrics']['auc_roc_std'],
        'cv_fold_values': best_result['metrics']['cv_fold_values'],
        'auc_pr_mean': best_result['metrics']['auc_pr_mean'],
        'auc_pr_std': best_result['metrics']['auc_pr_std'],
        'sensitivity_mean': best_result['metrics']['sensitivity_mean'],
        'sensitivity_std': best_result['metrics']['sensitivity_std'],
        'specificity_mean': best_result['metrics']['specificity_mean'],
        'specificity_std': best_result['metrics']['specificity_std'],
        'f1_mean': best_result['metrics']['f1_mean'],
        'f1_std': best_result['metrics']['f1_std'],
        'best_params': {k: _to_jsonable(v) for k, v in best_result['best_params'].items()},
        'imputer': best_result['imputer'],
        'selector_k': best_result['k_features'],
        'n_features_before_selection': n_features_total,
        'n_features_after_selection': len(selected_features),
        'selected_features': selected_features,
        'all_feature_names': all_feature_names,
        'cv': {
            'method': 'RepeatedStratifiedKFold',
            'n_splits': SINGLE_BEST_N_SPLITS,
            'n_repeats': SINGLE_BEST_N_REPEATS,
            'n_folds': SINGLE_BEST_N_SPLITS * SINGLE_BEST_N_REPEATS,
            'random_state': RANDOM_STATE,
        },
        'candidates': {
            name: {
                'auc_roc_mean': data['metrics']['auc_roc_mean'],
                'auc_roc_std': data['metrics']['auc_roc_std'],
                'auc_pr_mean': data['metrics']['auc_pr_mean'],
                'sensitivity_mean': data['metrics']['sensitivity_mean'],
                'specificity_mean': data['metrics']['specificity_mean'],
                'n_trials': data['n_trials'],
                'best_params': {k: _to_jsonable(v) for k, v in data['best_params'].items()},
                'imputer': data['imputer'],
                'selector_k': data['k_features'],
            }
            for name, data in candidate_results.items()
        },
    }
    with open(models_dir / "best_single_model.json", 'w', encoding='utf-8') as f:
        json.dump(best_metadata, f, ensure_ascii=False, indent=2, default=str)

    best_model_path = models_dir / "best_model.json"
    existing_meta = {}
    if best_model_path.exists():
        try:
            with open(best_model_path, 'r', encoding='utf-8') as f:
                existing_meta = json.load(f)
        except Exception:
            existing_meta = {}

    existing_meta.update({
        'single_best_model_type': best_metadata['model_type'],
        'single_best_artifact': best_metadata['artifact'],
        'single_best_feature_names_path': best_metadata['feature_names_path'],
        'single_best_cv_auc_mean': best_metadata['auc_roc_mean'],
        'single_best_cv_auc_std': best_metadata['auc_roc_std'],
        'single_best_target_reached': best_metadata['target_reached'],
        'single_best_feature_scope': best_metadata['feature_scope'],
        'single_best_metadata_path': 'best_single_model.json',
        'api_primary_model_artifact': best_metadata['artifact'],
        'api_primary_feature_names_path': best_metadata['feature_names_path'],
        'serialized_artifact_note': (
            'best_single_model.joblib is the preferred single-model artifact for API use. '
            'best_model.joblib remains available for backward compatibility.'
        ),
    })
    with open(best_model_path, 'w', encoding='utf-8') as f:
        json.dump(existing_meta, f, ensure_ascii=False, indent=2, default=str)

    print("\nBest single model:", flush=True)
    print(
        f"  {best_metadata['model_type']} AUC={best_metadata['auc_roc_mean']:.4f} "
        f"± {best_metadata['auc_roc_std']:.4f}",
        flush=True,
    )
    print(f"  Target reached: {best_metadata['target_reached']}", flush=True)
    print(f"  Saved: {models_dir / 'best_single_model.joblib'}", flush=True)
    return bool(best_metadata['target_reached'])


def main():
    print("=" * 60, flush=True)
    print("Vasculitis XAI — v3: Nested CV + Optuna + Stacking + CatBoost", flush=True)
    print("=" * 60, flush=True)

    data_path = project_root / "data" / "raw" / "aktualne_dane.csv"
    models_dir = project_root / "models" / "saved"
    models_dir.mkdir(parents=True, exist_ok=True)

    # ── 1/8 Load ─────────────────────────────────────────────────
    print(f"\n[1/8] Ładowanie z: {data_path}", flush=True)
    df = pd.read_csv(data_path, sep='|')
    for col in ['Kod', 'ID', 'id', 'Patient_ID']:
        if col in df.columns:
            df = df.drop(col, axis=1)
    print(f"      {len(df)} rekordów × {len(df.columns)} kolumn", flush=True)

    # ── 2/8 Candidate features ───────────────────────────────────
    print("\n[2/8] Selekcja cech kandydujących...", flush=True)
    candidate_features = _select_candidate_features(df)
    for col in candidate_features:
        df[col] = df[col].replace(-1, np.nan)
    X_full = df[candidate_features].values.astype(np.float64)
    y_full = df['Zgon'].values
    unique, counts = np.unique(y_full, return_counts=True)
    print(f"      {len(candidate_features)} kandydatów   Klasy: {dict(zip(unique.tolist(), counts.tolist()))}", flush=True)

    # ── 3/8 Hold-out split ───────────────────────────────────────
    print("\n[3/8] Hold-out 80/20 (test TYLKO do raportu)...", flush=True)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_full, y_full, test_size=0.2, stratify=y_full, random_state=RANDOM_STATE
    )
    print(f"      Train: {X_train_raw.shape[0]}, Test: {X_test_raw.shape[0]}", flush=True)

    # ── 4/8 Nested CV per model ──────────────────────────────────
    print(f"\n[4/8] Nested CV ({OUTER_N_SPLITS}×{OUTER_N_REPEATS} outer × {INNER_N_SPLITS} inner × n_iter={RANDOM_SEARCH_N_ITER})...", flush=True)
    nested_results = {}
    n_features_total = X_train_raw.shape[1]
    for model_type in BASE_MODEL_TYPES:
        nested_results[model_type] = _nested_cv_evaluate(
            model_type, X_train_raw, y_train, n_features_total
        )

    best_by_nested = max(nested_results, key=lambda m: nested_results[m]['auc_roc']['mean'])
    print(f"\n      🏆 Best po nested CV: {best_by_nested} "
          f"(AUC={nested_results[best_by_nested]['auc_roc']['mean']:.4f} ± "
          f"{nested_results[best_by_nested]['auc_roc']['std']:.4f})", flush=True)

    # ── 5/8 Wspólny preprocessor ─────────────────────────────────
    print("\n[5/8] Fit wspólnego preprocessora (median imputer)...", flush=True)
    preprocessor = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('selector', SelectKBest(f_classif, k=min(N_SELECT_FEATURES, n_features_total))),
    ])
    X_train_proc = preprocessor.fit_transform(X_train_raw, y_train)
    X_test_proc = preprocessor.transform(X_test_raw)
    selected_mask = preprocessor.named_steps['selector'].get_support()
    feature_names = [candidate_features[i] for i in range(len(candidate_features)) if selected_mask[i]]

    # KNN imputer wariant — porównawczo dla best modelu
    print("      Test KNN imputer (k=5) vs median dla best modelu...", flush=True)
    knn_pre = Pipeline([
        ('imputer', KNNImputer(n_neighbors=5)),
        ('scaler', StandardScaler()),
        ('selector', SelectKBest(f_classif, k=min(N_SELECT_FEATURES, n_features_total))),
    ])
    inner_cv_knn = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    X_train_proc_knn = knn_pre.fit_transform(X_train_raw, y_train)
    knn_clf = _make_classifier(best_by_nested)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        knn_scores = cross_val_score(knn_clf, X_train_proc_knn, y_train, cv=inner_cv_knn, scoring='roc_auc')
        med_clf = _make_classifier(best_by_nested)
        med_scores = cross_val_score(med_clf, X_train_proc, y_train, cv=inner_cv_knn, scoring='roc_auc')
    print(f"      median imputer: AUC={med_scores.mean():.4f} ± {med_scores.std():.4f}", flush=True)
    print(f"      KNN imputer:    AUC={knn_scores.mean():.4f} ± {knn_scores.std():.4f}", flush=True)
    use_knn = knn_scores.mean() > med_scores.mean()
    if use_knn:
        print("      → wybieram KNN imputer", flush=True)
        preprocessor = knn_pre
        X_train_proc = X_train_proc_knn
        X_test_proc = preprocessor.transform(X_test_raw)
        selected_mask = preprocessor.named_steps['selector'].get_support()
        feature_names = [candidate_features[i] for i in range(len(candidate_features)) if selected_mask[i]]

    import joblib
    joblib.dump(preprocessor, models_dir / "preprocessor.joblib")
    with open(models_dir / "feature_names.json", 'w', encoding='utf-8') as f:
        json.dump(feature_names, f, ensure_ascii=False, indent=2)
    _save_background_data(models_dir, X_train_proc, y_train, X_test_proc, y_test)

    # ── 6/8 Optuna final tuning ──────────────────────────────────
    print(f"\n[6/8] Optuna final tuning (n_trials={OPTUNA_N_TRIALS}, timeout={OPTUNA_TIMEOUT}s/model)...", flush=True)
    trained_models = {}
    optuna_results = {}
    for model_type in BASE_MODEL_TYPES:
        print(f"  ── Optuna: {model_type}", flush=True)
        t0 = time.time()
        try:
            best_clf, best_params, best_score, n_trials_actual = _optuna_tune(
                model_type, X_train_proc, y_train
            )
            trained_models[model_type] = best_clf
            optuna_results[model_type] = {
                'best_params': {k: _to_jsonable(v) for k, v in best_params.items()},
                'inner_cv_auc': best_score,
                'n_trials': n_trials_actual,
                'elapsed_seconds': time.time() - t0,
            }
            filename = MODEL_FILENAMES[model_type]
            metadata = {
                'model_type': model_type,
                'tuner': 'optuna_tpe_multivariate',
                'best_params': optuna_results[model_type]['best_params'],
                'optuna_inner_cv_auc': best_score,
                'nested_cv_auc_mean': nested_results[model_type]['auc_roc']['mean'],
                'nested_cv_auc_std': nested_results[model_type]['auc_roc']['std'],
                'feature_names': feature_names,
                'n_features': len(feature_names),
                'imputer_used': 'knn' if use_knn else 'median',
            }
            joblib.dump(best_clf, models_dir / filename)
            with open(models_dir / filename.replace('.joblib', '.json'), 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2, default=str)
            print(f"      AUC={best_score:.4f}  trials={n_trials_actual}  ({time.time()-t0:.1f}s)", flush=True)
        except Exception as e:
            print(f"      [BŁĄD] {model_type}: {e}", flush=True)
            import traceback
            traceback.print_exc()

    if not trained_models:
        print("\n[BŁĄD] Żaden model nie został wytrenowany!", flush=True)
        return False

    # ── 7/8 Stacking ─────────────────────────────────────────────
    print("\n[7/8] StackingClassifier (RF+GB+XGB+SVM, meta=LR)...", flush=True)
    stack_results = None
    try:
        stack = _build_stacking(trained_models)
        cv_5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stack_scores = cross_val_score(stack, X_train_proc, y_train, cv=cv_5, scoring='roc_auc', n_jobs=1)
        print(f"      Stacking 5-fold CV AUC={stack_scores.mean():.4f} ± {stack_scores.std():.4f}", flush=True)

        # Refit na pełnym train
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stack.fit(X_train_proc, y_train)
        trained_models['stacking'] = stack
        joblib.dump(stack, models_dir / MODEL_FILENAMES['stacking'])

        stack_results = {
            'cv_auc_mean': float(stack_scores.mean()),
            'cv_auc_std': float(stack_scores.std()),
            'cv_auc_values': [float(s) for s in stack_scores],
            'base_estimators': STACKING_BASE,
            'meta_estimator': 'LogisticRegression(C=1.0, class_weight=balanced)',
        }
        with open(models_dir / 'stacking_model.json', 'w', encoding='utf-8') as f:
            json.dump({
                'model_type': 'stacking',
                'feature_names': feature_names,
                'n_features': len(feature_names),
                'imputer_used': 'knn' if use_knn else 'median',
                **stack_results,
            }, f, ensure_ascii=False, indent=2, default=str)
    except Exception as e:
        print(f"      [BŁĄD] Stacking: {e}", flush=True)
        import traceback
        traceback.print_exc()

    # ── 8/8 Test eval + bootstrap ────────────────────────────────
    print("\n[8/8] Hold-out test set (raportowanie)...", flush=True)
    evaluator = ModelEvaluator()
    for name, model in trained_models.items():
        try:
            evaluator.evaluate_model(model, X_test_proc, y_test, model_name=name)
        except Exception as e:
            print(f"      [BŁĄD] eval {name}: {e}", flush=True)

    comparison_df = evaluator.compare_models(trained_models, X_test_proc, y_test)

    print(f"\n      Bootstrap 95% CI ({BOOTSTRAP_N_ITERATIONS} iter)...", flush=True)
    bootstrap_results = {}
    for name, model in trained_models.items():
        try:
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test_proc)[:, 1]
            else:
                from scipy.special import expit
                y_proba = expit(model.decision_function(X_test_proc))
            ci = evaluator.bootstrap_confidence_intervals(
                y_test, y_proba, n_iterations=BOOTSTRAP_N_ITERATIONS, confidence=BOOTSTRAP_CONFIDENCE
            )
            bootstrap_results[name] = {
                metric: {'lower': float(v[0]), 'mean': float(v[1]), 'upper': float(v[2])}
                for metric, v in ci.items()
            }
            print(f"        {name}: AUC={ci['auc_roc'][1]:.4f} [{ci['auc_roc'][0]:.4f}, {ci['auc_roc'][2]:.4f}]", flush=True)
        except Exception as e:
            print(f"        {name}: bootstrap BŁĄD: {e}", flush=True)

    eval_path = models_dir / "evaluation_report.json"
    evaluator.save_results(str(eval_path))
    if bootstrap_results:
        with open(eval_path, 'r', encoding='utf-8') as f:
            eval_data = json.load(f)
        for name, ci in bootstrap_results.items():
            if name in eval_data:
                eval_data[name]['confidence_intervals'] = ci
        with open(eval_path, 'w', encoding='utf-8') as f:
            json.dump(eval_data, f, ensure_ascii=False, indent=2, default=str)

    with open(models_dir / "model_comparison.json", 'w', encoding='utf-8') as f:
        json.dump(comparison_df.to_dict(orient='records'), f, ensure_ascii=False, indent=2, default=str)

    # cv_report.json — rozszerzony
    cv_output = {
        'method': 'NestedCV(outer=RepeatedStratifiedKFold(5x2), inner=StratifiedKFold(3)+RandomizedSearchCV) '
                  '+ Optuna(TPE multivariate, n_trials=50) final tuning + Stacking 5-fold CV',
        'n_outer_folds': OUTER_N_SPLITS * OUTER_N_REPEATS,
        'n_inner_folds': INNER_N_SPLITS,
        'random_search_n_iter': RANDOM_SEARCH_N_ITER,
        'optuna_n_trials': OPTUNA_N_TRIALS,
        'optuna_timeout_per_model': OPTUNA_TIMEOUT,
        'random_state': RANDOM_STATE,
        'n_features': len(feature_names),
        'imputer_chosen': 'knn' if use_knn else 'median',
        'best_model_by_nested_cv': best_by_nested,
        'models_nested_cv': nested_results,
        'models_optuna': optuna_results,
        'stacking': stack_results,
    }
    with open(models_dir / "cv_report.json", 'w', encoding='utf-8') as f:
        json.dump(cv_output, f, ensure_ascii=False, indent=2, default=str)

    # best_model.json — metadata
    best_meta = {
        'best_model_type_by_nested_cv': best_by_nested,
        'best_nested_cv_auc_mean': nested_results[best_by_nested]['auc_roc']['mean'],
        'best_nested_cv_auc_std': nested_results[best_by_nested]['auc_roc']['std'],
        'serialized_artifact_note':
            'best_model.joblib zawiera XGBoost (kontrakt z API). '
            'Faktycznie najlepszy wg nested CV i Optuna: zob. best_model_type_by_nested_cv i optuna_results.',
        'all_models_nested_cv': {
            m: {'mean': r['auc_roc']['mean'], 'std': r['auc_roc']['std']}
            for m, r in nested_results.items()
        },
        'all_models_optuna_inner_cv': {
            m: r['inner_cv_auc'] for m, r in optuna_results.items()
        },
        'stacking_5fold_cv_auc': stack_results['cv_auc_mean'] if stack_results else None,
        'imputer_chosen': 'knn' if use_knn else 'median',
    }
    with open(models_dir / "best_model.json", 'w', encoding='utf-8') as f:
        json.dump(best_meta, f, ensure_ascii=False, indent=2, default=str)

    # ── Podsumowanie ────────────────────────────────────────────
    print("\n" + "=" * 60, flush=True)
    print("SUKCES! v3 — Nested CV + Optuna + Stacking + CatBoost", flush=True)
    print("=" * 60, flush=True)

    print("\nNested CV (outer mean ± std), posortowane:", flush=True)
    for m, r in sorted(nested_results.items(), key=lambda kv: kv[1]['auc_roc']['mean'], reverse=True):
        print(f"  {m:<22} AUC={r['auc_roc']['mean']:.4f} ± {r['auc_roc']['std']:.4f}", flush=True)

    print("\nOptuna inner CV (po final tuning):", flush=True)
    for m, r in sorted(optuna_results.items(), key=lambda kv: kv[1]['inner_cv_auc'], reverse=True):
        print(f"  {m:<22} AUC={r['inner_cv_auc']:.4f}  trials={r['n_trials']}", flush=True)

    if stack_results:
        print(f"\nStacking 5-fold CV: AUC={stack_results['cv_auc_mean']:.4f} ± {stack_results['cv_auc_std']:.4f}", flush=True)

    print(f"\n🏆 Best wg nested CV: {best_by_nested}", flush=True)
    print(f"   Imputer: {'KNN(k=5)' if use_knn else 'median'}", flush=True)

    print("\nHold-out test set:", flush=True)
    print(comparison_df.to_string(index=False), flush=True)

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Vasculitis XAI models.")
    parser.add_argument(
        "--single-best",
        action="store_true",
        help="Tune one non-ensemble model against CV AUC >= 0.90 using all clinical features.",
    )
    parser.add_argument(
        "--single-best-trials",
        type=int,
        default=SINGLE_BEST_N_TRIALS,
        help="Optuna trials per single-best candidate.",
    )
    parser.add_argument(
        "--single-best-timeout",
        type=int,
        default=SINGLE_BEST_TIMEOUT,
        help="Timeout in seconds for the first single-best candidate.",
    )
    parser.add_argument(
        "--single-best-target",
        type=float,
        default=SINGLE_BEST_TARGET_AUC,
        help="Target mean CV AUC for single-best mode.",
    )
    args = parser.parse_args()

    if args.single_best:
        success = run_single_best_mode(
            n_trials=args.single_best_trials,
            timeout=args.single_best_timeout,
            target_auc=args.single_best_target,
        )
    else:
        success = main()
    sys.exit(0 if success else 1)
