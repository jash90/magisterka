"""
Testy jednostkowe dla modułu XAI.
"""

import pytest
import numpy as np
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestLIMEExplainer:
    """Testy dla klasy LIMEExplainer."""

    @pytest.fixture
    def sample_data(self):
        """Fixture dla przykładowych danych."""
        np.random.seed(42)
        X_train = np.random.randn(50, 5)
        return X_train

    @pytest.fixture
    def feature_names(self):
        """Fixture dla nazw cech."""
        return ['Wiek', 'Kreatynina', 'CRP', 'Nerki', 'OIT']

    @pytest.fixture
    def mock_model(self):
        """Fixture dla mock modelu."""
        model = Mock()
        model.predict.return_value = np.array([0])
        model.predict_proba.return_value = np.array([[0.7, 0.3]])
        return model

    def test_import_lime_explainer(self):
        """Test importu LIMEExplainer."""
        from src.xai.lime_explainer import LIMEExplainer
        assert LIMEExplainer is not None

    def test_lime_explainer_init(self, mock_model, sample_data, feature_names):
        """Test inicjalizacji LIME Explainer."""
        from src.xai.lime_explainer import LIMEExplainer

        explainer = LIMEExplainer(
            model=mock_model,
            X_train=sample_data,
            feature_names=feature_names
        )

        assert explainer.model == mock_model
        assert explainer.feature_names == feature_names
        assert explainer.explainer is not None

    def test_to_json(self, mock_model, sample_data, feature_names):
        """Test serializacji do JSON."""
        from src.xai.lime_explainer import LIMEExplainer
        import json

        explainer = LIMEExplainer(
            model=mock_model,
            X_train=sample_data,
            feature_names=feature_names
        )

        # Mock explanation
        mock_exp = {
            'prediction': 0,
            'prediction_label': 'Przeżycie',
            'probability': {'Przeżycie': 0.7, 'Zgon': 0.3},
            'feature_weights': [('Wiek > 50', 0.1), ('CRP < 10', -0.05)],
            'risk_factors': [('Wiek > 50', 0.1)],
            'protective_factors': [('CRP < 10', -0.05)],
            'intercept': 0.15
        }

        json_str = explainer.to_json(mock_exp)
        parsed = json.loads(json_str)

        assert 'prediction' in parsed
        assert 'feature_contributions' in parsed


class TestSHAPExplainer:
    """Testy dla klasy SHAPExplainer."""

    @pytest.fixture
    def sample_data(self):
        """Fixture dla przykładowych danych."""
        np.random.seed(42)
        return np.random.randn(50, 5)

    @pytest.fixture
    def feature_names(self):
        """Fixture dla nazw cech."""
        return ['Wiek', 'Kreatynina', 'CRP', 'Nerki', 'OIT']

    def test_import_shap_explainer(self):
        """Test importu SHAPExplainer."""
        from src.xai.shap_explainer import SHAPExplainer
        assert SHAPExplainer is not None

    def test_detect_explainer_type(self, sample_data, feature_names):
        """Test wykrywania typu explainera."""
        from src.xai.shap_explainer import SHAPExplainer

        # Mock dla RandomForest
        rf_model = Mock()
        rf_model.__class__.__name__ = 'RandomForestClassifier'
        rf_model.feature_importances_ = np.ones(5)

        explainer = SHAPExplainer.__new__(SHAPExplainer)
        detected = explainer._detect_explainer_type(rf_model)
        assert detected == 'tree'

        # Mock dla LogisticRegression
        lr_model = Mock()
        lr_model.__class__.__name__ = 'LogisticRegression'

        detected = explainer._detect_explainer_type(lr_model)
        assert detected == 'linear'


class TestDALEXWrapper:
    """Testy dla klasy DALEXWrapper."""

    def test_import_dalex_wrapper(self):
        """Test importu DALEXWrapper."""
        from src.xai.dalex_wrapper import DALEXWrapper
        assert DALEXWrapper is not None


class TestEBMExplainer:
    """Testy dla klasy EBMExplainer."""

    def test_import_ebm_explainer(self):
        """Test importu EBMExplainer."""
        from src.xai.ebm_explainer import EBMExplainer
        assert EBMExplainer is not None

    def test_ebm_init(self):
        """Test inicjalizacji EBM."""
        from src.xai.ebm_explainer import EBMExplainer

        explainer = EBMExplainer(
            feature_names=['A', 'B', 'C'],
            class_names=['Neg', 'Pos']
        )

        assert explainer.feature_names == ['A', 'B', 'C']
        assert explainer.class_names == ['Neg', 'Pos']
        assert explainer._is_fitted is False

    def test_ebm_not_fitted_error(self):
        """Test błędu gdy model nie jest dopasowany."""
        from src.xai.ebm_explainer import EBMExplainer

        explainer = EBMExplainer()

        with pytest.raises(RuntimeError):
            explainer.predict(np.array([[1, 2, 3]]))


class TestXAIComparison:
    """Testy dla klasy XAIComparison."""

    @pytest.fixture
    def feature_names(self):
        """Fixture dla nazw cech."""
        return ['Wiek', 'Kreatynina', 'CRP', 'Nerki', 'OIT']

    def test_import_xai_comparison(self):
        """Test importu XAIComparison."""
        from src.xai.comparison import XAIComparison
        assert XAIComparison is not None

    def test_comparison_init(self, feature_names):
        """Test inicjalizacji porównywarki."""
        from src.xai.comparison import XAIComparison

        comparison = XAIComparison(feature_names=feature_names)

        assert comparison.feature_names == feature_names
        assert comparison.comparison_results == {}

    def test_compare_feature_rankings(self, feature_names):
        """Test porównania rankingów."""
        from src.xai.comparison import XAIComparison

        comparison = XAIComparison(feature_names=feature_names)

        # Mock explanations
        explanations = {
            'SHAP': {
                'feature_impacts': [
                    {'feature': 'Wiek', 'shap_value': 0.3},
                    {'feature': 'Kreatynina', 'shap_value': 0.2},
                    {'feature': 'CRP', 'shap_value': 0.1}
                ]
            },
            'LIME': {
                'feature_weights': [
                    ('Wiek > 50', 0.25),
                    ('CRP < 10', 0.15),
                    ('Kreatynina > 100', 0.1)
                ]
            }
        }

        result = comparison.compare_feature_rankings(explanations, top_n=5)

        assert 'rankings' in result
        assert 'agreement_matrix' in result
        assert 'common_top_features' in result

    def test_calculate_agreement(self, feature_names):
        """Test obliczania zgodności."""
        from src.xai.comparison import XAIComparison

        comparison = XAIComparison(feature_names=feature_names)

        explanations = {
            'SHAP': {
                'feature_impacts': [
                    {'feature': 'Wiek', 'shap_value': 0.3},
                    {'feature': 'Kreatynina', 'shap_value': 0.2}
                ]
            },
            'LIME': {
                'feature_weights': [
                    ('Wiek > 50', 0.25),
                    ('Kreatynina > 100', 0.15)
                ]
            }
        }

        agreement = comparison.calculate_agreement(explanations)

        assert 'mean_ranking_agreement' in agreement
        assert 0 <= agreement['mean_ranking_agreement'] <= 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
