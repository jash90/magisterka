#!/usr/bin/env python3
"""
Skrypt do trenowania modelu predykcji śmiertelności w zapaleniu naczyń.

Użycie:
    python scripts/train_model.py

Wymagania:
    - Dane w data/raw/aktualne_dane.csv (separator: |)
    - Kolumna target: Zgon (0/1)
"""

import sys
import json
from pathlib import Path

# Dodaj ścieżkę projektu
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.preprocessing import DataPreprocessor
from src.models.train import ModelTrainer


def main():
    """Główna funkcja trenowania modelu."""

    print("=" * 60)
    print("Vasculitis XAI - Trenowanie modelu")
    print("=" * 60)

    # Ścieżki
    data_path = project_root / "data" / "raw" / "aktualne_dane.csv"
    model_output_path = project_root / "models" / "saved" / "best_model.joblib"
    features_output_path = project_root / "models" / "saved" / "feature_names.json"

    # Sprawdź czy dane istnieją
    if not data_path.exists():
        print(f"\n[BŁĄD] Nie znaleziono pliku danych: {data_path}")
        print("\nUpewnij się, że plik CSV z danymi pacjentów znajduje się w:")
        print(f"  {data_path}")
        print("\nFormat pliku:")
        print("  - Separator: | (pipe)")
        print("  - Kolumna target: Zgon (0=przeżycie, 1=zgon)")
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
    if 'Zgon' not in df.columns:
        print(f"\n[BŁĄD] Brak kolumny 'Zgon' w danych!")
        print(f"       Dostępne kolumny: {list(df.columns)[:10]}...")
        return False

    print(f"\n[2/5] Przygotowanie pipeline'u...")

    try:
        X, y, feature_names = preprocessor.prepare_pipeline(
            df,
            target_col='Zgon',
            n_features=20,  # Top 20 cech
            scale=True,
            handle_missing=True
        )
        print(f"      Przygotowano {X.shape[0]} próbek z {X.shape[1]} cechami")
        print(f"      Rozkład klas: {dict(zip(*np.unique(y, return_counts=True)))}")
    except Exception as e:
        print(f"\n[BŁĄD] Pipeline preprocessing: {e}")
        return False

    print(f"\n[3/5] Podział danych (80/20)...")

    try:
        X_train, X_test, y_train, y_test = preprocessor.get_train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        print(f"      Zbiór treningowy: {X_train.shape[0]} próbek")
        print(f"      Zbiór testowy: {X_test.shape[0]} próbek")
    except Exception as e:
        print(f"\n[BŁĄD] Podział danych: {e}")
        return False

    print(f"\n[4/5] Trenowanie modelu XGBoost...")

    try:
        trainer = ModelTrainer('xgboost')
        trainer.fit(X_train, y_train, feature_names=feature_names)

        # Ewaluacja
        from sklearn.metrics import roc_auc_score, accuracy_score, recall_score

        y_pred_proba = trainer.predict_proba(X_test)[:, 1]
        y_pred = trainer.predict(X_test)

        auc = roc_auc_score(y_test, y_pred_proba)
        acc = accuracy_score(y_test, y_pred)
        sens = recall_score(y_test, y_pred)

        print(f"      AUC-ROC: {auc:.3f}")
        print(f"      Accuracy: {acc:.3f}")
        print(f"      Sensitivity: {sens:.3f}")

    except Exception as e:
        print(f"\n[BŁĄD] Trenowanie modelu: {e}")
        import traceback
        traceback.print_exc()
        return False

    print(f"\n[5/5] Zapisywanie modelu...")

    try:
        # Utwórz katalog jeśli nie istnieje
        model_output_path.parent.mkdir(parents=True, exist_ok=True)

        # Zapisz model
        trainer.save_model(str(model_output_path), include_metadata=True)
        print(f"      Model zapisany: {model_output_path}")

        # Zapisz nazwy cech
        with open(features_output_path, 'w', encoding='utf-8') as f:
            json.dump(feature_names, f, ensure_ascii=False, indent=2)
        print(f"      Cechy zapisane: {features_output_path}")

    except Exception as e:
        print(f"\n[BŁĄD] Zapisywanie modelu: {e}")
        return False

    print("\n" + "=" * 60)
    print("SUKCES! Model został wytrenowany i zapisany.")
    print("=" * 60)
    print(f"\nPliki wyjściowe:")
    print(f"  - Model: {model_output_path}")
    print(f"  - Cechy: {features_output_path}")
    print(f"\nCechy modelu ({len(feature_names)}):")
    for i, name in enumerate(feature_names, 1):
        print(f"  {i:2}. {name}")

    print("\nMożesz teraz uruchomić API z pełnym modelem:")
    print("  cd /Users/bartlomiejzimny/Projects/magisterka")
    print("  python -m src.api.main")

    return True


if __name__ == "__main__":
    import numpy as np  # Import tutaj dla dostępności w main()
    success = main()
    sys.exit(0 if success else 1)
