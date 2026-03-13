#!/usr/bin/env python3
"""
Skrypt do trenowania modeli predykcji potrzeby dializy w zapaleniu naczyń.

Trenuje 4 modele: Logistic Regression, Random Forest, SVM, Naive Bayes.
Tworzy dialysis_model_registry.json z metrykami i domyślnym modelem (najlepszy AUC).

Target: Dializa (0/1), wartości -1 traktowane jako 0 (brak dializy).
Dane: 900 próbek, 171 pozytywnych (19%).

Użycie:
    python scripts/train_dialysis_model.py

Wymagania:
    - Dane w data/raw/aktualne_dane.csv (separator: |)
    - Kolumna target: Dializa (0/1/-1)
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# Dodaj ścieżkę projektu
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.preprocessing import DataPreprocessor, DIALYSIS_FEATURES
from src.models.train import ModelTrainer

# Modele do wytrenowania
MODEL_TYPES = ['logistic_regression', 'random_forest', 'xgboost', 'svm', 'naive_bayes']

MODEL_DISPLAY_NAMES = {
    'logistic_regression': 'Logistic Regression',
    'random_forest': 'Random Forest',
    'xgboost': 'XGBoost',
    'svm': 'SVM',
    'naive_bayes': 'Naive Bayes',
}


def main():
    """Główna funkcja trenowania modeli dializy."""

    print("=" * 60)
    print("Vasculitis XAI - Trenowanie modeli predykcji dializy")
    print("=" * 60)

    # Ścieżki
    data_path = project_root / "data" / "raw" / "aktualne_dane.csv"
    models_dir = project_root / "models" / "saved_dialysis"
    features_output_path = models_dir / "dialysis_feature_names.json"
    registry_path = models_dir / "dialysis_model_registry.json"

    # Sprawdź czy dane istnieją
    if not data_path.exists():
        print(f"\n[BŁĄD] Nie znaleziono pliku danych: {data_path}")
        print("\nUpewnij się, że plik CSV z danymi pacjentów znajduje się w:")
        print(f"  {data_path}")
        return False

    print(f"\n[1/5] Ładowanie danych z: {data_path}")

    # Przygotowanie preprocessora
    preprocessor = DataPreprocessor()

    try:
        df = preprocessor.load_data(str(data_path), separator='|')
        print(f"      Załadowano {len(df)} rekordów z {len(df.columns)} kolumnami")
    except Exception as e:
        print(f"\n[BŁĄD] Nie udało się załadować danych: {e}")
        return False

    # Sprawdź czy kolumna target istnieje
    if 'Dializa' not in df.columns:
        print(f"\n[BŁĄD] Brak kolumny 'Dializa' w danych!")
        print(f"       Dostępne kolumny: {list(df.columns)[:10]}...")
        return False

    # Obsługa wartości -1 PRZED pipeline (kluczowe!)
    print(f"      Rozkład Dializa przed obsługą: {df['Dializa'].value_counts().to_dict()}")
    df = preprocessor.prepare_dialysis_target(df, treat_minus_one_as_zero=True)
    print(f"      Rozkład Dializa po obsłudze: {df['Dializa'].value_counts().to_dict()}")

    # Filtruj do DIALYSIS_FEATURES + target
    available_features = [f for f in DIALYSIS_FEATURES if f in df.columns]
    missing_features = [f for f in DIALYSIS_FEATURES if f not in df.columns]
    if missing_features:
        print(f"      Uwaga: brak kolumn w danych: {missing_features}")

    columns_to_keep = available_features + ['Dializa']
    df = df[columns_to_keep]
    print(f"      Wybrano {len(available_features)} cech z {len(DIALYSIS_FEATURES)} zdefiniowanych")

    print(f"\n[2/5] Przygotowanie pipeline'u...")

    try:
        X, y, feature_names = preprocessor.prepare_pipeline(
            df,
            target_col='Dializa',
            n_features=len(available_features),
            scale=True,
            handle_missing=True
        )
        print(f"      Przygotowano {X.shape[0]} próbek z {X.shape[1]} cechami")
        print(f"      Rozkład klas: {dict(zip(*np.unique(y, return_counts=True)))}")
    except Exception as e:
        print(f"\n[BŁĄD] Pipeline preprocessing: {e}")
        return False

    print(f"\n[3/5] Podział danych (80/20, stratified)...")

    try:
        X_train, X_test, y_train, y_test = preprocessor.get_train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        print(f"      Zbiór treningowy: {X_train.shape[0]} próbek")
        print(f"      Zbiór testowy: {X_test.shape[0]} próbek")
        print(f"      Klasa pozytywna (train): {int(y_train.sum())} ({y_train.mean()*100:.1f}%)")
        print(f"      Klasa pozytywna (test): {int(y_test.sum())} ({y_test.mean()*100:.1f}%)")
    except Exception as e:
        print(f"\n[BŁĄD] Podział danych: {e}")
        return False

    # Zapisz artefakty XAI (background data, train data)
    import joblib
    models_dir.mkdir(parents=True, exist_ok=True)

    bg_indices = np.random.RandomState(42).choice(len(X_train), size=min(100, len(X_train)), replace=False)
    X_background = X_train[bg_indices]
    y_background = y_train[bg_indices]

    joblib.dump(X_background, models_dir / "X_background.joblib")
    joblib.dump(y_background, models_dir / "y_background.joblib")
    joblib.dump(X_train, models_dir / "X_train.joblib")
    joblib.dump(y_train, models_dir / "y_train.joblib")
    print(f"      Zapisano artefakty XAI: X_background ({X_background.shape}), X_train ({X_train.shape})")

    # Weryfikacja zapisu artefaktów XAI
    for name in ["X_background.joblib", "y_background.joblib", "X_train.joblib", "y_train.joblib"]:
        path = models_dir / name
        if path.exists():
            print(f"      [OK] {name} ({path.stat().st_size} B)")
        else:
            print(f"      [BŁĄD] {name} nie zapisany!")

    print(f"\n[4/5] Trenowanie {len(MODEL_TYPES)} modeli...")

    from sklearn.metrics import roc_auc_score, accuracy_score, recall_score

    # Utwórz katalog
    models_dir.mkdir(parents=True, exist_ok=True)

    # Rejestr modeli
    registry = {
        "task": "dialysis_prediction",
        "target_column": "Dializa",
        "created_at": datetime.now().isoformat(),
        "n_train_samples": int(X_train.shape[0]),
        "n_test_samples": int(X_test.shape[0]),
        "n_features": int(X_train.shape[1]),
        "feature_names_file": "dialysis_feature_names.json",
        "default_model": None,
        "models": {}
    }

    best_auc = -1.0
    best_model_type = None

    for i, model_type in enumerate(MODEL_TYPES, 1):
        display_name = MODEL_DISPLAY_NAMES.get(model_type, model_type)
        print(f"\n  [{i}/{len(MODEL_TYPES)}] Trenowanie: {display_name}...")

        try:
            trainer = ModelTrainer(model_type)
            trainer.fit(X_train, y_train, feature_names=feature_names)

            # Ewaluacja
            y_pred_proba = trainer.predict_proba(X_test)[:, 1]
            y_pred = trainer.predict(X_test)

            auc = roc_auc_score(y_test, y_pred_proba)
            acc = accuracy_score(y_test, y_pred)
            sens = recall_score(y_test, y_pred)

            print(f"         AUC-ROC: {auc:.3f}")
            print(f"         Accuracy: {acc:.3f}")
            print(f"         Sensitivity: {sens:.3f}")

            # Zapisz model
            model_path = models_dir / f"{model_type}.joblib"
            trainer.save_model(str(model_path), include_metadata=True)
            print(f"         Zapisano: {model_path}")

            # Dodaj do rejestru
            registry["models"][model_type] = {
                "display_name": display_name,
                "file": f"{model_type}.joblib",
                "metadata_file": f"{model_type}.json",
                "metrics": {
                    "auc_roc": round(auc, 4),
                    "accuracy": round(acc, 4),
                    "sensitivity": round(sens, 4)
                },
                "trained_at": datetime.now().isoformat()
            }

            # Sprawdź najlepszy AUC
            if auc > best_auc:
                best_auc = auc
                best_model_type = model_type

        except Exception as e:
            print(f"         [BŁĄD] {e}")
            import traceback
            traceback.print_exc()
            registry["models"][model_type] = {
                "display_name": display_name,
                "error": str(e),
                "trained_at": datetime.now().isoformat()
            }

    # EBM — Explainable Boosting Machine
    print(f"\n  [EBM] Trenowanie Explainable Boosting Machine...")
    try:
        from src.xai import EBMExplainer
        ebm = EBMExplainer(feature_names=feature_names)
        ebm.fit(X_train, y_train, feature_names=feature_names)
        ebm.save_model(str(models_dir / "dialysis_ebm_model.joblib"))
        print(f"         Zapisano: {models_dir / 'dialysis_ebm_model.joblib'}")
    except Exception as e:
        print(f"         [OSTRZEŻENIE] EBM: {e}")

    # Ustaw domyślny model (najlepszy AUC)
    registry["default_model"] = best_model_type

    print(f"\n[5/5] Zapisywanie rejestru i cech...")

    # Zapisz feature_names
    with open(features_output_path, 'w', encoding='utf-8') as f:
        json.dump(feature_names, f, ensure_ascii=False, indent=2)
    print(f"      Cechy zapisane: {features_output_path}")

    # Zapisz rejestr
    with open(registry_path, 'w', encoding='utf-8') as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)
    print(f"      Rejestr zapisany: {registry_path}")

    # Podsumowanie
    print("\n" + "=" * 60)
    print("SUKCES! Modele dializy zostały wytrenowane i zapisane.")
    print("=" * 60)

    print(f"\nDomyślny model: {best_model_type} (AUC: {best_auc:.3f})")
    print(f"\nWytrenowane modele:")
    for mt, info in registry["models"].items():
        if "metrics" in info:
            metrics = info["metrics"]
            default_marker = " [DOMYŚLNY]" if mt == best_model_type else ""
            print(f"  - {info['display_name']}: AUC={metrics['auc_roc']:.3f}, "
                  f"Acc={metrics['accuracy']:.3f}, Sens={metrics['sensitivity']:.3f}{default_marker}")
        else:
            print(f"  - {info['display_name']}: BŁĄD - {info.get('error', 'nieznany')}")

    print(f"\nCechy modelu ({len(feature_names)}):")
    for i, name in enumerate(feature_names, 1):
        print(f"  {i:2}. {name}")

    print(f"\nPliki wyjściowe:")
    print(f"  - Katalog modeli: {models_dir}")
    print(f"  - Rejestr: {registry_path}")
    print(f"  - Cechy: {features_output_path}")

    print("\nMożesz teraz uruchomić API z modelem dializy:")
    print("  python -m src.api.main")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
