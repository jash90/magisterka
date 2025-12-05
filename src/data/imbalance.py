"""
Moduł do obsługi niezbalansowanych danych.

Zawiera funkcje do resamplingu i obliczania wag klas
dla problemów z niezrównoważonymi klasami.
"""

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, NearMiss
from imblearn.combine import SMOTETomek, SMOTEENN
from typing import Tuple, Dict, Optional, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImbalanceHandler:
    """
    Klasa do obsługi niezbalansowanych danych.

    Zapewnia różne metody resamplingu i obliczania wag klas
    dla danych medycznych z niezrównoważonymi klasami.
    """

    def __init__(self, random_state: int = 42):
        """
        Inicjalizacja handlera.

        Args:
            random_state: Ziarno losowości dla reprodukowalności
        """
        self.random_state = random_state

    def get_class_distribution(self, y: np.ndarray) -> Dict[str, Union[int, float]]:
        """
        Sprawdź rozkład klas w danych.

        Args:
            y: Wektor etykiet

        Returns:
            Słownik z informacjami o rozkładzie klas
        """
        unique, counts = np.unique(y, return_counts=True)
        total = len(y)

        distribution = {
            'classes': unique.tolist(),
            'counts': counts.tolist(),
            'percentages': (counts / total * 100).tolist(),
            'imbalance_ratio': max(counts) / min(counts),
            'minority_class': unique[np.argmin(counts)],
            'majority_class': unique[np.argmax(counts)],
            'minority_count': int(min(counts)),
            'majority_count': int(max(counts))
        }

        logger.info(f"Rozkład klas: {dict(zip(unique, counts))}")
        logger.info(f"Współczynnik niezbalansowania: {distribution['imbalance_ratio']:.2f}")

        return distribution

    def calculate_class_weights(
        self,
        y: np.ndarray,
        method: str = 'balanced'
    ) -> Dict[int, float]:
        """
        Oblicz wagi klas.

        Args:
            y: Wektor etykiet
            method: Metoda obliczania wag ('balanced', 'inverse', 'sqrt_inverse')

        Returns:
            Słownik z wagami klas
        """
        classes = np.unique(y)

        if method == 'balanced':
            weights = compute_class_weight('balanced', classes=classes, y=y)
        elif method == 'inverse':
            # Odwrotność częstości
            counts = np.bincount(y)
            weights = len(y) / (len(classes) * counts)
        elif method == 'sqrt_inverse':
            # Pierwiastek z odwrotności częstości (łagodniejsze)
            counts = np.bincount(y)
            weights = np.sqrt(len(y) / (len(classes) * counts))
        else:
            raise ValueError(f"Nieznana metoda: {method}")

        class_weights = dict(zip(classes.astype(int), weights))

        logger.info(f"Wagi klas ({method}): {class_weights}")
        return class_weights

    def get_scale_pos_weight(self, y: np.ndarray) -> float:
        """
        Oblicz scale_pos_weight dla XGBoost/LightGBM.

        Args:
            y: Wektor etykiet (0/1)

        Returns:
            Wartość scale_pos_weight
        """
        n_negative = np.sum(y == 0)
        n_positive = np.sum(y == 1)

        scale_pos_weight = n_negative / n_positive

        logger.info(f"scale_pos_weight: {scale_pos_weight:.2f}")
        return scale_pos_weight

    def apply_smote(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sampling_strategy: Union[str, float, Dict] = 'auto',
        k_neighbors: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Zastosuj SMOTE (Synthetic Minority Over-sampling Technique).

        Args:
            X: Macierz cech
            y: Wektor etykiet
            sampling_strategy: Strategia samplowania
            k_neighbors: Liczba sąsiadów dla SMOTE

        Returns:
            Tuple (X_resampled, y_resampled)
        """
        # Dostosuj k_neighbors do liczby próbek klasy mniejszościowej
        minority_count = np.sum(y == 1)  # Zakładamy że zgon (1) jest klasą mniejszościową
        k_neighbors = min(k_neighbors, minority_count - 1)

        if k_neighbors < 1:
            logger.warning("Za mało próbek klasy mniejszościowej dla SMOTE. Używam RandomOverSampler.")
            return self.apply_random_oversampling(X, y, sampling_strategy)

        smote = SMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=k_neighbors,
            random_state=self.random_state
        )

        X_resampled, y_resampled = smote.fit_resample(X, y)

        logger.info(f"SMOTE: {len(y)} -> {len(y_resampled)} próbek")
        return X_resampled, y_resampled

    def apply_adasyn(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sampling_strategy: Union[str, float, Dict] = 'auto',
        n_neighbors: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Zastosuj ADASYN (Adaptive Synthetic Sampling).

        Args:
            X: Macierz cech
            y: Wektor etykiet
            sampling_strategy: Strategia samplowania
            n_neighbors: Liczba sąsiadów

        Returns:
            Tuple (X_resampled, y_resampled)
        """
        minority_count = np.sum(y == 1)
        n_neighbors = min(n_neighbors, minority_count - 1)

        if n_neighbors < 1:
            logger.warning("Za mało próbek dla ADASYN. Używam RandomOverSampler.")
            return self.apply_random_oversampling(X, y, sampling_strategy)

        adasyn = ADASYN(
            sampling_strategy=sampling_strategy,
            n_neighbors=n_neighbors,
            random_state=self.random_state
        )

        X_resampled, y_resampled = adasyn.fit_resample(X, y)

        logger.info(f"ADASYN: {len(y)} -> {len(y_resampled)} próbek")
        return X_resampled, y_resampled

    def apply_random_oversampling(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sampling_strategy: Union[str, float, Dict] = 'auto'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Zastosuj losowy oversampling.

        Args:
            X: Macierz cech
            y: Wektor etykiet
            sampling_strategy: Strategia samplowania

        Returns:
            Tuple (X_resampled, y_resampled)
        """
        ros = RandomOverSampler(
            sampling_strategy=sampling_strategy,
            random_state=self.random_state
        )

        X_resampled, y_resampled = ros.fit_resample(X, y)

        logger.info(f"RandomOverSampler: {len(y)} -> {len(y_resampled)} próbek")
        return X_resampled, y_resampled

    def apply_undersampling(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sampling_strategy: Union[str, float, Dict] = 'auto'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Zastosuj losowy undersampling klasy większościowej.

        Args:
            X: Macierz cech
            y: Wektor etykiet
            sampling_strategy: Strategia samplowania

        Returns:
            Tuple (X_resampled, y_resampled)
        """
        rus = RandomUnderSampler(
            sampling_strategy=sampling_strategy,
            random_state=self.random_state
        )

        X_resampled, y_resampled = rus.fit_resample(X, y)

        logger.info(f"RandomUnderSampler: {len(y)} -> {len(y_resampled)} próbek")
        return X_resampled, y_resampled

    def apply_tomek_links(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Zastosuj Tomek Links do usunięcia granicznych próbek.

        Args:
            X: Macierz cech
            y: Wektor etykiet

        Returns:
            Tuple (X_resampled, y_resampled)
        """
        tomek = TomekLinks()

        X_resampled, y_resampled = tomek.fit_resample(X, y)

        logger.info(f"TomekLinks: {len(y)} -> {len(y_resampled)} próbek")
        return X_resampled, y_resampled

    def apply_nearmiss(
        self,
        X: np.ndarray,
        y: np.ndarray,
        version: int = 1,
        n_neighbors: int = 3
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Zastosuj NearMiss undersampling.

        Args:
            X: Macierz cech
            y: Wektor etykiet
            version: Wersja algorytmu NearMiss (1, 2, lub 3)
            n_neighbors: Liczba sąsiadów

        Returns:
            Tuple (X_resampled, y_resampled)
        """
        nearmiss = NearMiss(
            version=version,
            n_neighbors=n_neighbors
        )

        X_resampled, y_resampled = nearmiss.fit_resample(X, y)

        logger.info(f"NearMiss v{version}: {len(y)} -> {len(y_resampled)} próbek")
        return X_resampled, y_resampled

    def apply_smote_tomek(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sampling_strategy: Union[str, float, Dict] = 'auto'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Zastosuj kombinację SMOTE + Tomek Links.

        Args:
            X: Macierz cech
            y: Wektor etykiet
            sampling_strategy: Strategia samplowania

        Returns:
            Tuple (X_resampled, y_resampled)
        """
        minority_count = np.sum(y == 1)
        k_neighbors = min(5, minority_count - 1)

        if k_neighbors < 1:
            logger.warning("Za mało próbek dla SMOTETomek. Używam RandomOverSampler + TomekLinks.")
            X_over, y_over = self.apply_random_oversampling(X, y, sampling_strategy)
            return self.apply_tomek_links(X_over, y_over)

        smote_tomek = SMOTETomek(
            sampling_strategy=sampling_strategy,
            random_state=self.random_state,
            smote=SMOTE(k_neighbors=k_neighbors, random_state=self.random_state)
        )

        X_resampled, y_resampled = smote_tomek.fit_resample(X, y)

        logger.info(f"SMOTETomek: {len(y)} -> {len(y_resampled)} próbek")
        return X_resampled, y_resampled

    def apply_smote_enn(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sampling_strategy: Union[str, float, Dict] = 'auto'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Zastosuj kombinację SMOTE + ENN (Edited Nearest Neighbors).

        Args:
            X: Macierz cech
            y: Wektor etykiet
            sampling_strategy: Strategia samplowania

        Returns:
            Tuple (X_resampled, y_resampled)
        """
        minority_count = np.sum(y == 1)
        k_neighbors = min(5, minority_count - 1)

        if k_neighbors < 1:
            logger.warning("Za mało próbek dla SMOTEENN. Używam RandomOverSampler.")
            return self.apply_random_oversampling(X, y, sampling_strategy)

        smote_enn = SMOTEENN(
            sampling_strategy=sampling_strategy,
            random_state=self.random_state,
            smote=SMOTE(k_neighbors=k_neighbors, random_state=self.random_state)
        )

        X_resampled, y_resampled = smote_enn.fit_resample(X, y)

        logger.info(f"SMOTEENN: {len(y)} -> {len(y_resampled)} próbek")
        return X_resampled, y_resampled

    def apply_combined(
        self,
        X: np.ndarray,
        y: np.ndarray,
        over_sampling_strategy: float = 0.5,
        under_sampling_strategy: float = 0.8
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Zastosuj kombinację oversampling + undersampling.

        Args:
            X: Macierz cech
            y: Wektor etykiet
            over_sampling_strategy: Proporcja oversamplingu
            under_sampling_strategy: Proporcja undersamplingu

        Returns:
            Tuple (X_resampled, y_resampled)
        """
        # Najpierw oversampling klasy mniejszościowej
        X_over, y_over = self.apply_smote(X, y, sampling_strategy=over_sampling_strategy)

        # Potem undersampling klasy większościowej
        X_final, y_final = self.apply_undersampling(X_over, y_over, sampling_strategy=under_sampling_strategy)

        logger.info(f"Combined: {len(y)} -> {len(y_over)} (after SMOTE) -> {len(y_final)} (after undersampling)")
        return X_final, y_final

    def recommend_strategy(
        self,
        y: np.ndarray,
        dataset_size: int
    ) -> str:
        """
        Zalecaj strategię obsługi niezbalansowania.

        Args:
            y: Wektor etykiet
            dataset_size: Rozmiar zbioru danych

        Returns:
            Nazwa zalecanej strategii
        """
        dist = self.get_class_distribution(y)
        imbalance_ratio = dist['imbalance_ratio']
        minority_count = dist['minority_count']

        # Logika rekomendacji
        if minority_count < 10:
            recommendation = 'random_oversampling'
            reason = "Za mało próbek mniejszościowych dla SMOTE"
        elif imbalance_ratio < 2:
            recommendation = 'class_weights'
            reason = "Niski poziom niezbalansowania - wagi klas wystarczą"
        elif imbalance_ratio < 5:
            recommendation = 'smote'
            reason = "Umiarkowane niezbalansowanie - SMOTE powinien wystarczyć"
        elif imbalance_ratio < 10:
            recommendation = 'smote_tomek'
            reason = "Wysokie niezbalansowanie - SMOTETomek dla lepszego rozdziału klas"
        else:
            recommendation = 'combined'
            reason = "Bardzo wysokie niezbalansowanie - kombinacja metod"

        if dataset_size < 500 and recommendation in ['smote_tomek', 'combined']:
            recommendation = 'smote'
            reason += " (mały dataset - uproszczono do SMOTE)"

        logger.info(f"Zalecana strategia: {recommendation}")
        logger.info(f"Powód: {reason}")

        return recommendation

    def apply_strategy(
        self,
        X: np.ndarray,
        y: np.ndarray,
        strategy: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Zastosuj wybraną lub automatycznie zalecaną strategię.

        Args:
            X: Macierz cech
            y: Wektor etykiet
            strategy: Nazwa strategii (None = automatyczna rekomendacja)

        Returns:
            Tuple (X_resampled, y_resampled)
        """
        if strategy is None:
            strategy = self.recommend_strategy(y, len(y))

        strategy_map = {
            'smote': self.apply_smote,
            'adasyn': self.apply_adasyn,
            'random_oversampling': self.apply_random_oversampling,
            'undersampling': self.apply_undersampling,
            'tomek_links': self.apply_tomek_links,
            'smote_tomek': self.apply_smote_tomek,
            'smote_enn': self.apply_smote_enn,
            'combined': self.apply_combined,
            'class_weights': lambda X, y: (X, y),  # Brak resamplingu
        }

        if strategy not in strategy_map:
            raise ValueError(f"Nieznana strategia: {strategy}. Dostępne: {list(strategy_map.keys())}")

        if strategy == 'class_weights':
            logger.info("Strategia 'class_weights' - bez resamplingu, użyj calculate_class_weights()")
            return X, y

        return strategy_map[strategy](X, y)
