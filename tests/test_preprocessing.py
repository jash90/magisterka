"""
Testy jednostkowe dla modułu preprocessingu.
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Dodaj ścieżkę projektu
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocessing import DataPreprocessor, KLUCZOWE_CECHY
from src.data.imbalance import ImbalanceHandler


class TestDataPreprocessor:
    """Testy dla klasy DataPreprocessor."""

    @pytest.fixture
    def preprocessor(self):
        """Fixture dla preprocessora."""
        return DataPreprocessor()

    @pytest.fixture
    def sample_df(self):
        """Fixture dla przykładowego DataFrame."""
        np.random.seed(42)
        n_samples = 100

        return pd.DataFrame({
            'Wiek': np.random.randint(20, 80, n_samples),
            'Plec': np.random.randint(0, 2, n_samples),
            'Kreatynina': np.random.uniform(50, 200, n_samples),
            'Max_CRP': np.random.uniform(0, 100, n_samples),
            'Liczba_Zajetych_Narzadow': np.random.randint(0, 5, n_samples),
            'Manifestacja_Nerki': np.random.randint(0, 2, n_samples),
            'Zaostrz_Wymagajace_OIT': np.random.randint(0, 2, n_samples),
            'Zgon': np.random.randint(0, 2, n_samples)
        })

    def test_init(self, preprocessor):
        """Test inicjalizacji."""
        assert preprocessor.scaler is None
        assert preprocessor.encoders == {}
        assert preprocessor.selected_features is None
        assert preprocessor._is_fitted is False

    def test_get_data_summary(self, preprocessor, sample_df):
        """Test generowania podsumowania danych."""
        summary = preprocessor.get_data_summary(sample_df)

        assert 'shape' in summary
        assert 'dtypes' in summary
        assert 'missing_values' in summary
        assert 'numeric_columns' in summary
        assert summary['shape'] == (100, 8)

    def test_handle_missing_values_median(self, preprocessor):
        """Test uzupełniania brakujących wartości medianą."""
        df = pd.DataFrame({
            'A': [1, 2, np.nan, 4, 5],
            'B': [10, np.nan, 30, 40, 50]
        })

        result = preprocessor.handle_missing_values(df, strategy='median')

        assert result['A'].isna().sum() == 0
        assert result['B'].isna().sum() == 0
        assert result['A'].iloc[2] == 3.0  # Mediana [1,2,4,5]

    def test_handle_missing_values_treats_minus_one(self, preprocessor):
        """Test zamiany -1 na NaN."""
        df = pd.DataFrame({
            'A': [1, -1, 3, 4, 5],
            'B': [10, 20, -1, 40, 50]
        })

        result = preprocessor.handle_missing_values(df, treat_minus_one_as_missing=True)

        assert -1 not in result['A'].values
        assert -1 not in result['B'].values

    def test_scale_features_standard(self, preprocessor):
        """Test standaryzacji."""
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])

        X_scaled = preprocessor.scale_features(X, method='standard')

        assert X_scaled.shape == X.shape
        assert np.abs(X_scaled.mean(axis=0)).max() < 0.1  # Średnia bliska 0
        assert np.abs(X_scaled.std(axis=0) - 1).max() < 0.2  # Std bliska 1

    def test_scale_features_minmax(self, preprocessor):
        """Test skalowania MinMax."""
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])

        X_scaled = preprocessor.scale_features(X, method='minmax')

        assert X_scaled.min() >= 0
        assert X_scaled.max() <= 1

    def test_prepare_pipeline(self, preprocessor, sample_df):
        """Test pełnego pipeline'a."""
        X, y, feature_names = preprocessor.prepare_pipeline(
            sample_df,
            target_col='Zgon',
            n_features=5
        )

        assert X.shape[0] == len(sample_df)
        assert X.shape[1] == 5
        assert len(y) == len(sample_df)
        assert len(feature_names) == 5
        assert preprocessor._is_fitted is True

    def test_get_train_test_split(self, preprocessor, sample_df):
        """Test podziału danych."""
        X = sample_df.drop('Zgon', axis=1).values
        y = sample_df['Zgon'].values

        X_train, X_test, y_train, y_test = preprocessor.get_train_test_split(
            X, y, test_size=0.2, stratify=True
        )

        assert len(X_train) == 80
        assert len(X_test) == 20
        assert len(y_train) == 80
        assert len(y_test) == 20


class TestImbalanceHandler:
    """Testy dla klasy ImbalanceHandler."""

    @pytest.fixture
    def handler(self):
        """Fixture dla handlera."""
        return ImbalanceHandler(random_state=42)

    @pytest.fixture
    def imbalanced_data(self):
        """Fixture dla niezbalansowanych danych."""
        np.random.seed(42)
        n_majority = 90
        n_minority = 10

        X = np.vstack([
            np.random.randn(n_majority, 5),
            np.random.randn(n_minority, 5) + 2
        ])
        y = np.array([0] * n_majority + [1] * n_minority)

        return X, y

    def test_get_class_distribution(self, handler, imbalanced_data):
        """Test rozkładu klas."""
        X, y = imbalanced_data

        dist = handler.get_class_distribution(y)

        assert dist['minority_count'] == 10
        assert dist['majority_count'] == 90
        assert dist['imbalance_ratio'] == 9.0

    def test_calculate_class_weights(self, handler, imbalanced_data):
        """Test obliczania wag klas."""
        X, y = imbalanced_data

        weights = handler.calculate_class_weights(y)

        assert 0 in weights
        assert 1 in weights
        assert weights[1] > weights[0]  # Klasa mniejszościowa ma większą wagę

    def test_get_scale_pos_weight(self, handler, imbalanced_data):
        """Test obliczania scale_pos_weight."""
        X, y = imbalanced_data

        spw = handler.get_scale_pos_weight(y)

        assert spw == 9.0  # 90/10

    def test_apply_smote(self, handler, imbalanced_data):
        """Test SMOTE."""
        X, y = imbalanced_data

        X_resampled, y_resampled = handler.apply_smote(X, y)

        # Po SMOTE powinno być więcej próbek
        assert len(y_resampled) > len(y)

        # Klasy powinny być bardziej zbalansowane
        unique, counts = np.unique(y_resampled, return_counts=True)
        assert counts[0] == counts[1]  # Domyślnie auto = równe klasy

    def test_apply_undersampling(self, handler, imbalanced_data):
        """Test undersamplingui."""
        X, y = imbalanced_data

        X_resampled, y_resampled = handler.apply_undersampling(X, y)

        # Po undersampling powinno być mniej próbek
        assert len(y_resampled) < len(y)

    def test_recommend_strategy(self, handler, imbalanced_data):
        """Test rekomendacji strategii."""
        X, y = imbalanced_data

        strategy = handler.recommend_strategy(y, len(y))

        assert strategy in [
            'smote', 'smote_tomek', 'combined', 'class_weights',
            'random_oversampling', 'adasyn'
        ]

    def test_apply_strategy(self, handler, imbalanced_data):
        """Test zastosowania strategii."""
        X, y = imbalanced_data

        X_resampled, y_resampled = handler.apply_strategy(X, y, strategy='smote')

        assert len(y_resampled) >= len(y)


class TestFeatureEngineer:
    """Testy dla klasy FeatureEngineer."""

    @pytest.fixture
    def sample_df(self):
        """Fixture dla przykładowego DataFrame."""
        return pd.DataFrame({
            'Wiek': [30, 55, 70, 45, 60],
            'Wiek_rozpoznania': [25, 50, 65, 40, 55],
            'Max_CRP': [5, 50, 150, 20, 80],
            'Kreatynina': [80, 100, 150, 90, 130],
            'Manifestacja_Nerki': [0, 1, 1, 0, 1],
            'Manifestacja_Sercowo-Naczyniowy': [0, 0, 1, 0, 1],
            'Zaostrz_Wymagajace_OIT': [0, 0, 1, 0, 1]
        })

    def test_import_feature_engineer(self):
        """Test importu FeatureEngineer."""
        from src.data.feature_engineering import FeatureEngineer
        fe = FeatureEngineer()
        assert fe is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
