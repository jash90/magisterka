"""
Moduł EBM Explainer (Explainable Boosting Machine).

Implementuje inherentnie interpretowalny model EBM z biblioteki interpret.
"""

from interpret.glassbox import ExplainableBoostingClassifier
from interpret import show
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Union, Tuple
import json
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EBMExplainer:
    """
    Klasa do trenowania i wyjaśniania modeli EBM.

    EBM (Explainable Boosting Machine) to inherentnie interpretowalny
    model, który łączy dokładność gradient boosting z interpretowalnością
    generalizowanych modeli addytywnych (GAM).

    Attributes:
        model: Wytrenowany model EBM
        feature_names: Lista nazw cech
        class_names: Nazwy klas
    """

    def __init__(
        self,
        feature_names: Optional[List[str]] = None,
        class_names: List[str] = None,
        max_bins: int = 256,
        max_interaction_bins: int = 32,
        interactions: int = 10,
        outer_bags: int = 8,
        inner_bags: int = 0,
        learning_rate: float = 0.01,
        validation_size: float = 0.15,
        early_stopping_rounds: int = 50,
        max_rounds: int = 5000,
        random_state: int = 42
    ):
        """
        Inicjalizacja EBM Explainera.

        Args:
            feature_names: Lista nazw cech
            class_names: Nazwy klas
            max_bins: Maksymalna liczba binów dla cech ciągłych
            max_interaction_bins: Maksymalna liczba binów dla interakcji
            interactions: Liczba interakcji do wykrycia
            outer_bags: Liczba zewnętrznych bagów
            inner_bags: Liczba wewnętrznych bagów
            learning_rate: Współczynnik uczenia
            validation_size: Rozmiar zbioru walidacyjnego
            early_stopping_rounds: Rundy do early stopping
            max_rounds: Maksymalna liczba rund
            random_state: Ziarno losowości
        """
        self.feature_names = feature_names
        self.class_names = class_names or ['Przeżycie', 'Zgon']

        # Inicjalizacja modelu EBM
        self.model = ExplainableBoostingClassifier(
            max_bins=max_bins,
            max_interaction_bins=max_interaction_bins,
            interactions=interactions,
            outer_bags=outer_bags,
            inner_bags=inner_bags,
            learning_rate=learning_rate,
            validation_size=validation_size,
            early_stopping_rounds=early_stopping_rounds,
            max_rounds=max_rounds,
            random_state=random_state
        )

        self._is_fitted = False

        logger.info("EBMExplainer zainicjalizowany")

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> 'EBMExplainer':
        """
        Wytrenuj model EBM.

        Args:
            X_train: Dane treningowe
            y_train: Etykiety treningowe
            feature_names: Opcjonalne nazwy cech

        Returns:
            self
        """
        if feature_names:
            self.feature_names = feature_names
        elif self.feature_names is None:
            self.feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]

        logger.info(f"Trenowanie EBM na {len(X_train)} próbkach...")

        self.model.fit(X_train, y_train)
        self._is_fitted = True

        logger.info("Trenowanie EBM zakończone")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predykcja klas.

        Args:
            X: Dane

        Returns:
            Przewidywane klasy
        """
        self._check_fitted()
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predykcja prawdopodobieństw.

        Args:
            X: Dane

        Returns:
            Macierz prawdopodobieństw
        """
        self._check_fitted()
        return self.model.predict_proba(X)

    def _check_fitted(self):
        """Sprawdź czy model jest wytrenowany."""
        if not self._is_fitted:
            raise RuntimeError("Model nie jest wytrenowany. Użyj fit() najpierw.")

    def explain_global(self) -> Dict[str, Any]:
        """
        Globalne wyjaśnienie modelu.

        Returns:
            Słownik z globalnym wyjaśnieniem
        """
        self._check_fitted()

        # Pobierz globalne wyjaśnienie
        global_exp = self.model.explain_global()

        # Wyodrębnij dane
        feature_names = global_exp.data()['names']
        feature_scores = global_exp.data()['scores']

        # Posortuj wg ważności
        importance = dict(zip(feature_names, feature_scores))
        importance_sorted = dict(
            sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)
        )

        # Wykryj interakcje
        interactions = []
        for name in feature_names:
            if ' x ' in name:
                interactions.append(name)

        explanation = {
            'feature_importance': importance_sorted,
            'feature_names': list(importance_sorted.keys()),
            'feature_scores': list(importance_sorted.values()),
            'interactions_detected': interactions,
            'n_features': len([f for f in feature_names if ' x ' not in f]),
            'n_interactions': len(interactions)
        }

        return explanation

    def explain_local(
        self,
        instance: np.ndarray
    ) -> Dict[str, Any]:
        """
        Lokalne wyjaśnienie dla pojedynczej instancji.

        Args:
            instance: Wektor cech

        Returns:
            Słownik z lokalnym wyjaśnieniem
        """
        self._check_fitted()

        # Upewnij się że instance jest 2D
        if instance.ndim == 1:
            instance = instance.reshape(1, -1)

        # Lokalne wyjaśnienie
        local_exp = self.model.explain_local(instance)

        # Wyodrębnij dane
        exp_data = local_exp.data(0)  # Pierwsza instancja

        names = exp_data['names']
        scores = exp_data['scores']
        values = exp_data['values']

        # Predykcja
        proba = self.predict_proba(instance)[0]
        prediction = int(np.argmax(proba))

        # Przygotuj contributions
        contributions = []
        for name, score, value in zip(names, scores, values):
            if name != 'intercept':
                contributions.append({
                    'feature': name,
                    'score': float(score),
                    'value': value,
                    'direction': 'zwiększa ryzyko' if score > 0 else 'zmniejsza ryzyko'
                })

        # Posortuj wg bezwzględnej wartości
        contributions_sorted = sorted(
            contributions,
            key=lambda x: abs(x['score']),
            reverse=True
        )

        # Intercept
        intercept_idx = names.index('intercept') if 'intercept' in names else -1
        intercept = float(scores[intercept_idx]) if intercept_idx >= 0 else 0

        explanation = {
            'prediction': prediction,
            'prediction_label': self.class_names[prediction],
            'probability': {
                self.class_names[0]: float(proba[0]),
                self.class_names[1]: float(proba[1])
            },
            'probability_positive': float(proba[1]),
            'intercept': intercept,
            'contributions': contributions_sorted,
            'risk_factors': [c for c in contributions_sorted if c['score'] > 0],
            'protective_factors': [c for c in contributions_sorted if c['score'] < 0]
        }

        return explanation

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Pobierz ważność cech z modelu.

        Returns:
            Słownik {nazwa_cechy: ważność}
        """
        global_exp = self.explain_global()
        return global_exp['feature_importance']

    def get_feature_function(
        self,
        feature: str
    ) -> Dict[str, Any]:
        """
        Pobierz funkcję kształtu dla cechy.

        Args:
            feature: Nazwa cechy

        Returns:
            Słownik z danymi funkcji kształtu
        """
        self._check_fitted()

        global_exp = self.model.explain_global()

        # Znajdź indeks cechy
        feature_names = global_exp.data()['names']
        if feature not in feature_names:
            raise ValueError(f"Cecha '{feature}' nie istnieje. Dostępne: {feature_names}")

        feature_idx = feature_names.index(feature)

        # Pobierz dane specyficzne dla cechy
        specific_data = global_exp.data(feature_idx)

        return {
            'feature': feature,
            'type': specific_data.get('type', 'unknown'),
            'names': specific_data.get('names', []),
            'scores': specific_data.get('scores', []),
            'density': specific_data.get('density', {})
        }

    def plot_global_importance(
        self,
        max_features: int = 15,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Wykres globalnej ważności cech.

        Args:
            max_features: Maksymalna liczba cech
            save_path: Ścieżka do zapisu

        Returns:
            Figura matplotlib
        """
        importance = self.get_feature_importance()

        # Filtruj tylko cechy (bez interakcji) i weź top N
        features_only = {
            k: v for k, v in importance.items()
            if ' x ' not in k
        }
        top_features = dict(list(features_only.items())[:max_features])

        fig, ax = plt.subplots(figsize=(10, 8))

        names = list(top_features.keys())
        scores = list(top_features.values())

        y_pos = np.arange(len(names))
        colors = ['#d73027' if s > 0 else '#1a9850' for s in scores]

        ax.barh(y_pos, scores, color=colors, alpha=0.8)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=9)
        ax.invert_yaxis()

        ax.axvline(x=0, color='black', linewidth=0.5)
        ax.set_xlabel('Średni wpływ na predykcję')
        ax.set_title('EBM - Globalna ważność cech')
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Wykres globalny EBM zapisany do {save_path}")

        return fig

    def plot_local_explanation(
        self,
        explanation: Dict[str, Any],
        max_features: int = 10,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Wykres lokalnego wyjaśnienia.

        Args:
            explanation: Wyjaśnienie z explain_local
            max_features: Maksymalna liczba cech
            save_path: Ścieżka do zapisu

        Returns:
            Figura matplotlib
        """
        contributions = explanation['contributions'][:max_features]

        fig, ax = plt.subplots(figsize=(10, 6))

        names = [c['feature'] for c in contributions]
        scores = [c['score'] for c in contributions]
        colors = ['#d73027' if s > 0 else '#1a9850' for s in scores]

        y_pos = np.arange(len(names))
        ax.barh(y_pos, scores, color=colors, alpha=0.8)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=9)
        ax.invert_yaxis()

        ax.axvline(x=0, color='black', linewidth=0.5)
        ax.set_xlabel('Wkład do predykcji')
        ax.set_title(
            f'EBM - Lokalne wyjaśnienie\n'
            f'Predykcja: {explanation["prediction_label"]} '
            f'({explanation["probability_positive"]:.1%})'
        )

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Wykres lokalny EBM zapisany do {save_path}")

        return fig

    def plot_feature_function(
        self,
        feature: str,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Wykres funkcji kształtu dla cechy.

        Args:
            feature: Nazwa cechy
            save_path: Ścieżka do zapisu

        Returns:
            Figura matplotlib
        """
        func_data = self.get_feature_function(feature)

        fig, ax = plt.subplots(figsize=(10, 6))

        names = func_data['names']
        scores = func_data['scores']

        # Jeśli numeryczna - wykres liniowy
        try:
            x_vals = [float(n) for n in names]
            ax.plot(x_vals, scores, linewidth=2, color='steelblue')
            ax.fill_between(x_vals, scores, alpha=0.3)
        except (ValueError, TypeError):
            # Kategoryczna - wykres słupkowy
            x_pos = np.arange(len(names))
            ax.bar(x_pos, scores, color='steelblue', alpha=0.8)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(names, rotation=45, ha='right')

        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.set_xlabel(feature)
        ax.set_ylabel('Wpływ na predykcję')
        ax.set_title(f'EBM - Funkcja kształtu dla {feature}')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Wykres funkcji kształtu zapisany do {save_path}")

        return fig

    def show_dashboard(self):
        """
        Wyświetl interaktywny dashboard (w Jupyter).
        """
        self._check_fitted()

        global_exp = self.model.explain_global()
        show(global_exp)

    def to_json(self, explanation: Dict[str, Any]) -> str:
        """
        Serializuj wyjaśnienie do JSON.

        Args:
            explanation: Wyjaśnienie

        Returns:
            String JSON
        """
        # Konwertuj numpy types
        def convert(obj):
            if isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        json_exp = json.loads(json.dumps(explanation, default=convert))
        return json.dumps(json_exp, ensure_ascii=False, indent=2)

    def save_model(self, path: str) -> None:
        """
        Zapisz model do pliku.

        Args:
            path: Ścieżka do pliku
        """
        self._check_fitted()

        save_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'class_names': self.class_names
        }

        joblib.dump(save_data, path)
        logger.info(f"Model EBM zapisany do {path}")

    def load_model(self, path: str) -> 'EBMExplainer':
        """
        Wczytaj model z pliku.

        Args:
            path: Ścieżka do pliku

        Returns:
            self
        """
        save_data = joblib.load(path)

        self.model = save_data['model']
        self.feature_names = save_data['feature_names']
        self.class_names = save_data['class_names']
        self._is_fitted = True

        logger.info(f"Model EBM wczytany z {path}")

        return self

    def get_model_summary(self) -> Dict[str, Any]:
        """
        Pobierz podsumowanie modelu.

        Returns:
            Słownik z informacjami o modelu
        """
        self._check_fitted()

        global_exp = self.explain_global()

        return {
            'model_type': 'ExplainableBoostingClassifier',
            'n_features': global_exp['n_features'],
            'n_interactions': global_exp['n_interactions'],
            'top_5_features': list(global_exp['feature_importance'].keys())[:5],
            'interactions_detected': global_exp['interactions_detected'],
            'is_fitted': self._is_fitted
        }
