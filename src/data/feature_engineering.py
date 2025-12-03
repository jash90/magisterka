"""
Moduł inżynierii cech dla danych medycznych.

Zawiera funkcje do tworzenia nowych cech, agregacji
i transformacji danych pacjentów z zapaleniem naczyń.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Tuple
from sklearn.preprocessing import PolynomialFeatures
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Klasa do inżynierii cech medycznych.

    Tworzy nowe cechy z istniejących danych klinicznych,
    uwzględniając domenową wiedzę medyczną.
    """

    def __init__(self):
        """Inicjalizacja feature engineera."""
        self.created_features: List[str] = []
        self.feature_descriptions: Dict[str, str] = {}

    def create_age_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Utwórz cechy związane z wiekiem.

        Args:
            df: DataFrame z danymi

        Returns:
            DataFrame z nowymi cechami
        """
        df = df.copy()

        # Kategorie wiekowe
        if 'Wiek' in df.columns:
            df['Wiek_Kategoria'] = pd.cut(
                df['Wiek'],
                bins=[0, 40, 55, 65, 75, 100],
                labels=[0, 1, 2, 3, 4]  # młody, średni, starszy, senior, sędziwy
            ).astype(float)
            self.created_features.append('Wiek_Kategoria')
            self.feature_descriptions['Wiek_Kategoria'] = 'Kategoria wiekowa (0-4)'

        # Czas trwania choroby (jeśli są dostępne dane)
        if 'Wiek' in df.columns and 'Wiek_rozpoznania' in df.columns:
            df['Czas_Choroby'] = df['Wiek'] - df['Wiek_rozpoznania']
            df['Czas_Choroby'] = df['Czas_Choroby'].clip(lower=0)  # Usuń ujemne wartości
            self.created_features.append('Czas_Choroby')
            self.feature_descriptions['Czas_Choroby'] = 'Czas trwania choroby (lata)'

        # Czy choroba rozpoznana wcześnie (<50 lat)
        if 'Wiek_rozpoznania' in df.columns:
            df['Wczesne_Rozpoznanie'] = (df['Wiek_rozpoznania'] < 50).astype(int)
            self.created_features.append('Wczesne_Rozpoznanie')
            self.feature_descriptions['Wczesne_Rozpoznanie'] = 'Rozpoznanie przed 50. rokiem życia'

        logger.info(f"Utworzono {len(self.created_features)} cech związanych z wiekiem")
        return df

    def create_organ_involvement_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Utwórz cechy związane z zajęciem narządów.

        Args:
            df: DataFrame z danymi

        Returns:
            DataFrame z nowymi cechami
        """
        df = df.copy()

        # Kolumny manifestacji narządowych
        manifestation_cols = [col for col in df.columns if col.startswith('Manifestacja_')]

        if manifestation_cols:
            # Suma zajętych narządów (jeśli nie ma już takiej kolumny)
            if 'Liczba_Zajetych_Narzadow' not in df.columns:
                df['Liczba_Zajetych_Narzadow_Calc'] = df[manifestation_cols].sum(axis=1)
                self.created_features.append('Liczba_Zajetych_Narzadow_Calc')
                self.feature_descriptions['Liczba_Zajetych_Narzadow_Calc'] = 'Obliczona liczba zajętych narządów'

            # Zajęcie narządów krytycznych (serce, nerki, CNS)
            critical_organs = [
                'Manifestacja_Sercowo-Naczyniowy',
                'Manifestacja_Nerki',
                'Manifestacja_Zajecie_CSN'
            ]
            available_critical = [col for col in critical_organs if col in df.columns]

            if available_critical:
                df['Zajecie_Narzadow_Krytycznych'] = df[available_critical].max(axis=1)
                df['Liczba_Narzadow_Krytycznych'] = df[available_critical].sum(axis=1)
                self.created_features.extend(['Zajecie_Narzadow_Krytycznych', 'Liczba_Narzadow_Krytycznych'])
                self.feature_descriptions['Zajecie_Narzadow_Krytycznych'] = 'Czy zajęty jakikolwiek narząd krytyczny'
                self.feature_descriptions['Liczba_Narzadow_Krytycznych'] = 'Liczba zajętych narządów krytycznych'

        logger.info(f"Utworzono cechy zajęcia narządów: {len([f for f in self.created_features if 'Narzad' in f or 'Manifestacja' in f])}")
        return df

    def create_laboratory_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Utwórz cechy związane z wynikami laboratoryjnymi.

        Args:
            df: DataFrame z danymi

        Returns:
            DataFrame z nowymi cechami
        """
        df = df.copy()

        # CRP - marker zapalny
        if 'Max_CRP' in df.columns:
            # Kategorie CRP
            df['CRP_Kategoria'] = pd.cut(
                df['Max_CRP'],
                bins=[-1, 10, 50, 100, float('inf')],
                labels=[0, 1, 2, 3]  # normalny, podwyższony, wysoki, bardzo wysoki
            ).astype(float)
            self.created_features.append('CRP_Kategoria')
            self.feature_descriptions['CRP_Kategoria'] = 'Kategoria poziomu CRP'

            # Log CRP (dla normalizacji rozkładu)
            df['Log_CRP'] = np.log1p(df['Max_CRP'].clip(lower=0))
            self.created_features.append('Log_CRP')
            self.feature_descriptions['Log_CRP'] = 'Logarytm CRP'

        # Kreatynina - funkcja nerek
        if 'Kreatynina' in df.columns:
            # eGFR przybliżone (uproszczone, bez pełnego wzoru MDRD/CKD-EPI)
            if 'Wiek' in df.columns:
                # Uproszczony wzór - w praktyce należy użyć pełnego wzoru z płcią i rasą
                df['eGFR_Est'] = 175 * (df['Kreatynina'] / 88.4) ** (-1.154) * df['Wiek'] ** (-0.203)
                df['eGFR_Est'] = df['eGFR_Est'].clip(lower=0, upper=200)  # Rozsądne granice
                self.created_features.append('eGFR_Est')
                self.feature_descriptions['eGFR_Est'] = 'Szacowane eGFR'

            # Kategorie funkcji nerek
            df['Kreatynina_Wysoka'] = (df['Kreatynina'] > 120).astype(int)  # >120 μmol/L
            self.created_features.append('Kreatynina_Wysoka')
            self.feature_descriptions['Kreatynina_Wysoka'] = 'Podwyższona kreatynina (>120)'

        logger.info(f"Utworzono cechy laboratoryjne")
        return df

    def create_treatment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Utwórz cechy związane z leczeniem.

        Args:
            df: DataFrame z danymi

        Returns:
            DataFrame z nowymi cechami
        """
        df = df.copy()

        # Intensywność leczenia sterydami
        if 'Sterydy_Dawka_g' in df.columns:
            df['Wysokie_Sterydy'] = (df['Sterydy_Dawka_g'] > 1).astype(int)
            self.created_features.append('Wysokie_Sterydy')
            self.feature_descriptions['Wysokie_Sterydy'] = 'Wysokie dawki sterydów (>1g)'

        if 'Czas_Sterydow' in df.columns:
            df['Dlugie_Sterydy'] = (df['Czas_Sterydow'] > 12).astype(int)  # >12 miesięcy
            self.created_features.append('Dlugie_Sterydy')
            self.feature_descriptions['Dlugie_Sterydy'] = 'Długotrwałe leczenie sterydami (>12 mies.)'

        # Intensywne terapie
        intensive_therapies = ['Plazmaferezy', 'Dializa']
        available_therapies = [col for col in intensive_therapies if col in df.columns]

        if available_therapies:
            df['Intensywne_Leczenie'] = df[available_therapies].max(axis=1)
            self.created_features.append('Intensywne_Leczenie')
            self.feature_descriptions['Intensywne_Leczenie'] = 'Zastosowano intensywne terapie'

        logger.info(f"Utworzono cechy leczenia")
        return df

    def create_complication_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Utwórz cechy związane z powikłaniami.

        Args:
            df: DataFrame z danymi

        Returns:
            DataFrame z nowymi cechami
        """
        df = df.copy()

        # Kolumny powikłań
        complication_cols = [col for col in df.columns if col.startswith('Powiklania_')]

        if complication_cols:
            # Suma powikłań
            df['Liczba_Powiklan'] = df[complication_cols].sum(axis=1)
            self.created_features.append('Liczba_Powiklan')
            self.feature_descriptions['Liczba_Powiklan'] = 'Liczba powikłań'

            # Ciężkie powikłania
            severe_complications = [
                'Powiklania_Serce/pluca',
                'Powiklania_Infekcja'
            ]
            available_severe = [col for col in severe_complications if col in df.columns]

            if available_severe:
                df['Ciezkie_Powiklania'] = df[available_severe].max(axis=1)
                self.created_features.append('Ciezkie_Powiklania')
                self.feature_descriptions['Ciezkie_Powiklania'] = 'Wystąpiły ciężkie powikłania'

        # OIT jako marker ciężkości
        if 'Zaostrz_Wymagajace_OIT' in df.columns:
            # Już jest binarna
            pass

        logger.info(f"Utworzono cechy powikłań")
        return df

    def create_interaction_features(
        self,
        df: pd.DataFrame,
        features: Optional[List[str]] = None,
        degree: int = 2,
        include_bias: bool = False
    ) -> pd.DataFrame:
        """
        Utwórz cechy interakcyjne (iloczyny cech).

        Args:
            df: DataFrame z danymi
            features: Lista cech do interakcji (None = wszystkie numeryczne)
            degree: Stopień wielomianu
            include_bias: Czy dodać kolumnę bias

        Returns:
            DataFrame z nowymi cechami
        """
        df = df.copy()

        if features is None:
            # Wybierz kluczowe cechy do interakcji
            features = [col for col in df.columns if col in [
                'Wiek', 'Liczba_Zajetych_Narzadow', 'Kreatynina', 'Max_CRP',
                'Zaostrz_Wymagajace_OIT', 'Zajecie_Narzadow_Krytycznych'
            ]]

        if len(features) < 2:
            logger.warning("Za mało cech do utworzenia interakcji")
            return df

        # Filtruj cechy które istnieją
        features = [f for f in features if f in df.columns]

        if len(features) < 2:
            logger.warning("Za mało istniejących cech do utworzenia interakcji")
            return df

        # Utwórz interakcje
        X_interaction = df[features].values

        poly = PolynomialFeatures(
            degree=degree,
            include_bias=include_bias,
            interaction_only=True
        )

        X_poly = poly.fit_transform(X_interaction)

        # Nazwy nowych cech
        poly_feature_names = poly.get_feature_names_out(features)

        # Dodaj tylko nowe cechy (nie oryginalne)
        for i, name in enumerate(poly_feature_names):
            if name not in features and ' ' in name:  # Tylko interakcje
                clean_name = name.replace(' ', '_x_')
                df[clean_name] = X_poly[:, i]
                self.created_features.append(clean_name)
                self.feature_descriptions[clean_name] = f'Interakcja: {name}'

        logger.info(f"Utworzono {len([f for f in self.created_features if '_x_' in f])} cech interakcyjnych")
        return df

    def create_risk_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Utwórz uproszczony score ryzyka oparty na cechach klinicznych.

        Args:
            df: DataFrame z danymi

        Returns:
            DataFrame z nowym score'em
        """
        df = df.copy()

        risk_score = np.zeros(len(df))

        # Wiek (1 punkt za każde 10 lat powyżej 50)
        if 'Wiek' in df.columns:
            risk_score += np.maximum(0, (df['Wiek'] - 50) / 10)

        # Liczba zajętych narządów
        if 'Liczba_Zajetych_Narzadow' in df.columns:
            risk_score += df['Liczba_Zajetych_Narzadow']

        # Zajęcie krytyczne
        if 'Zajecie_Narzadow_Krytycznych' in df.columns:
            risk_score += df['Zajecie_Narzadow_Krytycznych'] * 2

        # OIT
        if 'Zaostrz_Wymagajace_OIT' in df.columns:
            risk_score += df['Zaostrz_Wymagajace_OIT'] * 3

        # Wysoka kreatynina
        if 'Kreatynina_Wysoka' in df.columns:
            risk_score += df['Kreatynina_Wysoka'] * 2
        elif 'Kreatynina' in df.columns:
            risk_score += (df['Kreatynina'] > 120).astype(int) * 2

        # Wysokie CRP
        if 'Max_CRP' in df.columns:
            risk_score += (df['Max_CRP'] > 100).astype(int) * 1.5

        # Dializa
        if 'Dializa' in df.columns:
            risk_score += df['Dializa'] * 3

        df['Risk_Score'] = risk_score
        df['Risk_Category'] = pd.cut(
            df['Risk_Score'],
            bins=[-float('inf'), 3, 6, 10, float('inf')],
            labels=[0, 1, 2, 3]  # niskie, średnie, wysokie, bardzo wysokie
        ).astype(float)

        self.created_features.extend(['Risk_Score', 'Risk_Category'])
        self.feature_descriptions['Risk_Score'] = 'Obliczony score ryzyka'
        self.feature_descriptions['Risk_Category'] = 'Kategoria ryzyka (0-3)'

        logger.info("Utworzono Risk Score i Risk Category")
        return df

    def engineer_all_features(
        self,
        df: pd.DataFrame,
        include_interactions: bool = True
    ) -> pd.DataFrame:
        """
        Zastosuj wszystkie transformacje cech.

        Args:
            df: DataFrame z danymi
            include_interactions: Czy tworzyć cechy interakcyjne

        Returns:
            DataFrame z wszystkimi nowymi cechami
        """
        self.created_features = []
        self.feature_descriptions = {}

        df = self.create_age_features(df)
        df = self.create_organ_involvement_features(df)
        df = self.create_laboratory_features(df)
        df = self.create_treatment_features(df)
        df = self.create_complication_features(df)
        df = self.create_risk_score(df)

        if include_interactions:
            df = self.create_interaction_features(df)

        logger.info(f"Łącznie utworzono {len(self.created_features)} nowych cech")
        return df

    def get_feature_info(self) -> pd.DataFrame:
        """
        Zwróć informacje o utworzonych cechach.

        Returns:
            DataFrame z opisami cech
        """
        return pd.DataFrame({
            'Cecha': self.created_features,
            'Opis': [self.feature_descriptions.get(f, '') for f in self.created_features]
        })
