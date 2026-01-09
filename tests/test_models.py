"""
Testy jednostkowe dla modułu modeli ML.
"""

import pytest
import numpy as np
import sys
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.config import (
    MODEL_CONFIGS, MEDICAL_METRICS, TRAINING_ORDER,
    get_model_class, get_base_params, get_grid_search_params
)
from src.models.evaluate import ModelEvaluator


class TestModelConfig:
    """Testy dla konfiguracji modeli."""

    def test_model_configs_not_empty(self):
        """Test że konfiguracje nie są puste."""
        assert len(MODEL_CONFIGS) > 0

    def test_required_models_exist(self):
        """Test że wymagane modele istnieją."""
        required = ['random_forest', 'xgboost', 'lightgbm']
        for model in required:
            assert model in MODEL_CONFIGS

    def test_model_config_structure(self):
        """Test struktury konfiguracji modelu."""
        for model_name, config in MODEL_CONFIGS.items():
            assert 'class' in config, f"Brak 'class' w {model_name}"
            assert 'base_params' in config, f"Brak 'base_params' w {model_name}"
            assert 'grid_search' in config, f"Brak 'grid_search' w {model_name}"

    def test_medical_metrics(self):
        """Test metryk medycznych."""
        expected = ['roc_auc', 'average_precision', 'recall', 'precision', 'f1', 'brier_score']
        for metric in expected:
            assert metric in MEDICAL_METRICS

    def test_get_model_class_valid(self):
        """Test pobierania klasy modelu."""
        cls = get_model_class('random_forest')
        assert cls is not None
        assert 'RandomForestClassifier' in str(cls)

    def test_get_model_class_invalid(self):
        """Test błędu dla nieistniejącego modelu."""
        with pytest.raises(ValueError):
            get_model_class('nieistniejacy_model')

    def test_get_base_params(self):
        """Test pobierania bazowych parametrów."""
        params = get_base_params('xgboost')
        assert 'n_estimators' in params
        assert 'learning_rate' in params
        assert 'random_state' in params

    def test_get_grid_search_params(self):
        """Test pobierania parametrów grid search."""
        params = get_grid_search_params('random_forest')
        assert 'n_estimators' in params
        assert 'max_depth' in params
        assert isinstance(params['n_estimators'], list)


class TestModelTrainer:
    """Testy dla klasy ModelTrainer."""

    @pytest.fixture
    def sample_data(self):
        """Fixture dla przykładowych danych."""
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)
        return X, y

    @pytest.fixture
    def feature_names(self):
        """Fixture dla nazw cech."""
        return [f'feature_{i}' for i in range(10)]

    def test_import_model_trainer(self):
        """Test importu ModelTrainer."""
        from src.models.train import ModelTrainer
        trainer = ModelTrainer('random_forest')
        assert trainer is not None
        assert trainer.model_type == 'random_forest'

    def test_trainer_invalid_model(self):
        """Test błędu dla nieistniejącego modelu."""
        from src.models.train import ModelTrainer
        with pytest.raises(ValueError):
            ModelTrainer('nieistniejacy_model')


class TestModelEvaluator:
    """Testy dla klasy ModelEvaluator."""

    @pytest.fixture
    def evaluator(self):
        """Fixture dla evaluatora."""
        return ModelEvaluator()

    @pytest.fixture
    def sample_predictions(self):
        """Fixture dla przykładowych predykcji."""
        np.random.seed(42)
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0])
        y_pred = np.array([0, 0, 1, 0, 0, 1, 1, 1, 1, 0])
        y_proba = np.array([0.1, 0.2, 0.8, 0.4, 0.3, 0.9, 0.6, 0.85, 0.75, 0.15])
        return y_true, y_pred, y_proba

    def test_calculate_medical_metrics(self, evaluator, sample_predictions):
        """Test obliczania metryk medycznych."""
        y_true, y_pred, y_proba = sample_predictions

        metrics = evaluator.calculate_medical_metrics(y_true, y_pred, y_proba)

        # Sprawdź obecność metryk
        assert 'accuracy' in metrics
        assert 'sensitivity' in metrics
        assert 'specificity' in metrics
        assert 'ppv' in metrics
        assert 'npv' in metrics
        assert 'auc_roc' in metrics
        assert 'brier_score' in metrics

        # Sprawdź zakresy wartości
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['sensitivity'] <= 1
        assert 0 <= metrics['specificity'] <= 1
        assert 0 <= metrics['auc_roc'] <= 1

    def test_confusion_matrix_values(self, evaluator, sample_predictions):
        """Test wartości macierzy konfuzji."""
        y_true, y_pred, y_proba = sample_predictions

        metrics = evaluator.calculate_medical_metrics(y_true, y_pred, y_proba)

        # Suma TP + TN + FP + FN = total
        total = (metrics['true_positives'] + metrics['true_negatives'] +
                 metrics['false_positives'] + metrics['false_negatives'])
        assert total == len(y_true)

    def test_sensitivity_calculation(self, evaluator):
        """Test obliczania czułości."""
        y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 0, 0, 0, 0, 1])  # 3 TP, 1 FN

        metrics = evaluator.calculate_medical_metrics(y_true, y_pred)

        assert metrics['sensitivity'] == 0.75  # 3/4

    def test_specificity_calculation(self, evaluator):
        """Test obliczania swoistości."""
        y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 0, 0, 0, 0, 1])  # 3 TN, 1 FP

        metrics = evaluator.calculate_medical_metrics(y_true, y_pred)

        assert metrics['specificity'] == 0.75  # 3/4

    def test_find_optimal_threshold(self, evaluator, sample_predictions):
        """Test znajdowania optymalnego progu."""
        y_true, y_pred, y_proba = sample_predictions

        threshold, metrics = evaluator.find_optimal_threshold(
            y_true, y_proba,
            metric='youden',
            min_sensitivity=0.5
        )

        assert 0 < threshold < 1
        assert 'sensitivity' in metrics
        assert metrics['sensitivity'] >= 0.5

    def test_bootstrap_ci(self, evaluator, sample_predictions):
        """Test przedziałów ufności bootstrap."""
        y_true, y_pred, y_proba = sample_predictions

        ci_results = evaluator.bootstrap_confidence_intervals(
            y_true, y_proba,
            n_iterations=100  # Mała liczba dla szybkości
        )

        assert 'auc_roc' in ci_results
        lower, mean, upper = ci_results['auc_roc']
        assert lower <= mean <= upper


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
