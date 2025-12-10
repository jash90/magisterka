"""
Moduł trenowania modeli ML.

Zawiera klasę ModelTrainer do trenowania, tuningu hiperparametrów
i zarządzania modelami klasyfikacji.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_val_score
)
from sklearn.metrics import make_scorer, roc_auc_score
import joblib
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime
import logging

from .config import (
    MODEL_CONFIGS, get_model_class, get_base_params,
    get_grid_search_params, get_random_search_params
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Klasa do trenowania i zarządzania modelami ML.

    Attributes:
        model_type: Typ modelu (np. 'xgboost', 'random_forest')
        model: Wytrenowany model
        best_params: Najlepsze znalezione parametry
        cv_results: Wyniki cross-validation
        training_history: Historia trenowania
    """

    def __init__(
        self,
        model_type: str,
        custom_params: Optional[Dict[str, Any]] = None
    ):
        """
        Inicjalizacja trainera.

        Args:
            model_type: Typ modelu
            custom_params: Opcjonalne niestandardowe parametry
        """
        if model_type not in MODEL_CONFIGS:
            raise ValueError(f"Nieznany model: {model_type}. Dostępne: {list(MODEL_CONFIGS.keys())}")

        self.model_type = model_type
        self.custom_params = custom_params or {}

        # Pobierz klasę i parametry bazowe
        self.model_class = get_model_class(model_type)
        self.base_params = get_base_params(model_type)
        self.base_params.update(self.custom_params)

        # Zainicjalizuj model
        self.model = self.model_class(**self.base_params)

        # Wyniki trenowania
        self.best_params: Optional[Dict[str, Any]] = None
        self.cv_results: Optional[Dict[str, Any]] = None
        self.training_history: List[Dict[str, Any]] = []
        self.feature_names: Optional[List[str]] = None

        logger.info(f"Zainicjalizowano ModelTrainer dla {model_type}")

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: Optional[List[str]] = None,
        eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> 'ModelTrainer':
        """
        Wytrenuj model.

        Args:
            X_train: Dane treningowe
            y_train: Etykiety treningowe
            feature_names: Opcjonalne nazwy cech
            eval_set: Opcjonalny zbiór walidacyjny (X_val, y_val)

        Returns:
            self
        """
        self.feature_names = feature_names

        start_time = datetime.now()
        logger.info(f"Rozpoczęcie trenowania {self.model_type}...")

        # Specjalne parametry dla XGBoost/LightGBM z eval_set
        fit_params = {}
        if eval_set is not None and self.model_type in ['xgboost', 'lightgbm']:
            fit_params['eval_set'] = [eval_set]
            if self.model_type == 'xgboost':
                fit_params['verbose'] = False
            elif self.model_type == 'lightgbm':
                fit_params['callbacks'] = None

        self.model.fit(X_train, y_train, **fit_params)

        training_time = (datetime.now() - start_time).total_seconds()

        # Zapisz historię
        self.training_history.append({
            'timestamp': datetime.now().isoformat(),
            'action': 'fit',
            'training_time': training_time,
            'n_samples': len(X_train),
            'n_features': X_train.shape[1]
        })

        logger.info(f"Trenowanie zakończone w {training_time:.2f}s")
        return self

    def tune_hyperparameters(
        self,
        X: np.ndarray,
        y: np.ndarray,
        method: str = 'grid',
        cv: int = 5,
        scoring: str = 'roc_auc',
        n_iter: int = 50,
        n_jobs: int = -1
    ) -> 'ModelTrainer':
        """
        Tuning hiperparametrów.

        Args:
            X: Dane
            y: Etykiety
            method: Metoda tuningu ('grid' lub 'random')
            cv: Liczba foldów CV
            scoring: Metryka do optymalizacji
            n_iter: Liczba iteracji dla RandomizedSearchCV
            n_jobs: Liczba procesów

        Returns:
            self z zaktualizowanym modelem
        """
        logger.info(f"Rozpoczęcie tuningu hiperparametrów ({method})...")
        start_time = datetime.now()

        # Pobierz siatkę parametrów
        if method == 'grid':
            param_space = get_grid_search_params(self.model_type)
        else:
            param_space = get_random_search_params(self.model_type)

        if not param_space:
            logger.warning(f"Brak siatki parametrów dla {self.model_type}")
            return self

        # Stratified K-Fold
        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

        # Utwórz nowy model z bazowymi parametrami
        base_model = self.model_class(**self.base_params)

        # Grid lub Random Search
        if method == 'grid':
            search = GridSearchCV(
                estimator=base_model,
                param_grid=param_space,
                scoring=scoring,
                cv=cv_splitter,
                n_jobs=n_jobs,
                verbose=1,
                refit=True
            )
        else:
            search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=param_space,
                n_iter=n_iter,
                scoring=scoring,
                cv=cv_splitter,
                n_jobs=n_jobs,
                verbose=1,
                refit=True,
                random_state=42
            )

        search.fit(X, y)

        # Aktualizuj model i parametry
        self.model = search.best_estimator_
        self.best_params = search.best_params_
        self.cv_results = {
            'best_score': search.best_score_,
            'best_params': search.best_params_,
            'cv_results': pd.DataFrame(search.cv_results_).to_dict()
        }

        tuning_time = (datetime.now() - start_time).total_seconds()

        # Zapisz historię
        self.training_history.append({
            'timestamp': datetime.now().isoformat(),
            'action': f'tune_{method}',
            'tuning_time': tuning_time,
            'best_score': search.best_score_,
            'best_params': search.best_params_
        })

        logger.info(f"Tuning zakończony w {tuning_time:.2f}s")
        logger.info(f"Najlepszy wynik {scoring}: {search.best_score_:.4f}")
        logger.info(f"Najlepsze parametry: {search.best_params_}")

        return self

    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5,
        scoring: Union[str, List[str]] = 'roc_auc'
    ) -> Dict[str, np.ndarray]:
        """
        Walidacja krzyżowa.

        Args:
            X: Dane
            y: Etykiety
            cv: Liczba foldów
            scoring: Metryka(i) do ewaluacji

        Returns:
            Słownik z wynikami CV
        """
        logger.info(f"Rozpoczęcie {cv}-fold cross-validation...")

        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

        if isinstance(scoring, str):
            scores = cross_val_score(
                self.model, X, y, cv=cv_splitter, scoring=scoring
            )
            results = {
                scoring: scores,
                f'{scoring}_mean': np.mean(scores),
                f'{scoring}_std': np.std(scores)
            }
        else:
            results = {}
            for metric in scoring:
                scores = cross_val_score(
                    self.model, X, y, cv=cv_splitter, scoring=metric
                )
                results[metric] = scores
                results[f'{metric}_mean'] = np.mean(scores)
                results[f'{metric}_std'] = np.std(scores)

        logger.info(f"CV zakończone")
        for key, value in results.items():
            if '_mean' in key:
                logger.info(f"  {key}: {value:.4f}")

        return results

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predykcja klas.

        Args:
            X: Dane

        Returns:
            Przewidywane klasy
        """
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predykcja prawdopodobieństw.

        Args:
            X: Dane

        Returns:
            Macierz prawdopodobieństw
        """
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            # Dla modeli bez predict_proba użyj decision_function
            decision = self.model.decision_function(X)
            # Przekształć na pseudo-prawdopodobieństwa
            from scipy.special import expit
            proba = expit(decision)
            return np.column_stack([1 - proba, proba])

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Pobierz ważność cech z modelu.

        Returns:
            Słownik {nazwa_cechy: ważność} lub None
        """
        if not hasattr(self.model, 'feature_importances_'):
            logger.warning(f"Model {self.model_type} nie ma feature_importances_")
            return None

        importances = self.model.feature_importances_

        if self.feature_names:
            return dict(zip(self.feature_names, importances))
        else:
            return {f'feature_{i}': imp for i, imp in enumerate(importances)}

    def save_model(self, path: str, include_metadata: bool = True) -> None:
        """
        Zapisz model do pliku.

        Args:
            path: Ścieżka do pliku
            include_metadata: Czy zapisać metadane
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Zapisz model
        joblib.dump(self.model, path)
        logger.info(f"Model zapisany do {path}")

        # Zapisz metadane
        if include_metadata:
            metadata = {
                'model_type': self.model_type,
                'best_params': self.best_params,
                'feature_names': self.feature_names,
                'training_history': self.training_history,
                'base_params': self.base_params,
                'timestamp': datetime.now().isoformat()
            }

            metadata_path = path.with_suffix('.json')
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2, default=str)
            logger.info(f"Metadane zapisane do {metadata_path}")

    def load_model(self, path: str) -> 'ModelTrainer':
        """
        Wczytaj model z pliku.

        Args:
            path: Ścieżka do pliku

        Returns:
            self
        """
        path = Path(path)

        self.model = joblib.load(path)
        logger.info(f"Model wczytany z {path}")

        # Wczytaj metadane jeśli istnieją
        metadata_path = path.with_suffix('.json')
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            self.best_params = metadata.get('best_params')
            self.feature_names = metadata.get('feature_names')
            self.training_history = metadata.get('training_history', [])
            logger.info(f"Metadane wczytane z {metadata_path}")

        return self

    def get_model_summary(self) -> Dict[str, Any]:
        """
        Pobierz podsumowanie modelu.

        Returns:
            Słownik z informacjami o modelu
        """
        summary = {
            'model_type': self.model_type,
            'model_class': str(self.model_class),
            'is_fitted': hasattr(self.model, 'classes_') or hasattr(self.model, 'n_features_in_'),
            'n_features': getattr(self.model, 'n_features_in_', None),
            'feature_names': self.feature_names,
            'best_params': self.best_params,
            'n_training_runs': len(self.training_history)
        }

        # Dodaj feature importance jeśli dostępne
        importance = self.get_feature_importance()
        if importance:
            sorted_importance = dict(
                sorted(importance.items(), key=lambda x: x[1], reverse=True)
            )
            summary['top_10_features'] = dict(list(sorted_importance.items())[:10])

        return summary


def train_multiple_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    model_types: Optional[List[str]] = None,
    feature_names: Optional[List[str]] = None,
    tune: bool = False,
    tune_method: str = 'random',
    n_iter: int = 30
) -> Dict[str, ModelTrainer]:
    """
    Wytrenuj wiele modeli.

    Args:
        X_train: Dane treningowe
        y_train: Etykiety treningowe
        X_val: Dane walidacyjne
        y_val: Etykiety walidacyjne
        model_types: Lista typów modeli (None = wszystkie)
        feature_names: Nazwy cech
        tune: Czy wykonać tuning hiperparametrów
        tune_method: Metoda tuningu
        n_iter: Liczba iteracji dla random search

    Returns:
        Słownik {model_type: ModelTrainer}
    """
    if model_types is None:
        model_types = ['random_forest', 'xgboost', 'lightgbm', 'logistic_regression']

    eval_set = (X_val, y_val) if X_val is not None and y_val is not None else None

    trained_models = {}

    for model_type in model_types:
        logger.info(f"\n{'='*50}")
        logger.info(f"Trenowanie modelu: {model_type}")
        logger.info('='*50)

        trainer = ModelTrainer(model_type)

        if tune:
            # Tuning na połączonych danych train+val
            if eval_set:
                X_tune = np.vstack([X_train, X_val])
                y_tune = np.concatenate([y_train, y_val])
            else:
                X_tune, y_tune = X_train, y_train

            trainer.tune_hyperparameters(
                X_tune, y_tune,
                method=tune_method,
                n_iter=n_iter
            )

        trainer.fit(X_train, y_train, feature_names=feature_names, eval_set=eval_set)

        trained_models[model_type] = trainer

    logger.info(f"\nWytrenowano {len(trained_models)} modeli")
    return trained_models
