"""
Moduł porównania metod XAI.

Zawiera funkcje do porównywania wyjaśnień generowanych przez
różne metody XAI (LIME, SHAP, DALEX, EBM).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import List, Dict, Any, Optional, Tuple
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class XAIComparison:
    """
    Klasa do porównywania metod XAI.

    Umożliwia porównanie rankingów cech, stabilności wyjaśnień
    i zgodności między różnymi metodami XAI.
    """

    def __init__(
        self,
        feature_names: List[str],
        class_names: List[str] = None
    ):
        """
        Inicjalizacja porównywarki XAI.

        Args:
            feature_names: Lista nazw cech
            class_names: Nazwy klas
        """
        self.feature_names = feature_names
        self.class_names = class_names or ['Przeżycie', 'Zgon']
        self.comparison_results: Dict[str, Any] = {}

    def compare_feature_rankings(
        self,
        explanations: Dict[str, Dict[str, Any]],
        top_n: int = 10
    ) -> Dict[str, Any]:
        """
        Porównaj rankingi cech między metodami XAI.

        Args:
            explanations: Słownik {nazwa_metody: wyjaśnienie}
            top_n: Liczba top cech do porównania

        Returns:
            Słownik z wynikami porównania
        """
        rankings = {}
        importance_scores = {}

        for method_name, exp in explanations.items():
            # Wyodrębnij ranking cech
            ranking = self._extract_feature_ranking(exp, method_name)
            rankings[method_name] = ranking[:top_n]

            # Wyodrębnij scores
            scores = self._extract_importance_scores(exp, method_name)
            importance_scores[method_name] = scores

        # Oblicz zgodność rankingów
        agreement_matrix = self._calculate_ranking_agreement(rankings)

        # Znajdź wspólne top cechy
        common_top_features = self._find_common_top_features(rankings, threshold=0.5)

        # Korelacja Spearmana między rankingami
        spearman_correlations = self._calculate_spearman_correlations(importance_scores)

        result = {
            'rankings': rankings,
            'importance_scores': importance_scores,
            'agreement_matrix': agreement_matrix,
            'common_top_features': common_top_features,
            'spearman_correlations': spearman_correlations,
            'top_n': top_n
        }

        self.comparison_results['rankings'] = result
        return result

    def _extract_feature_ranking(
        self,
        explanation: Dict[str, Any],
        method: str
    ) -> List[str]:
        """Wyodrębnij ranking cech z wyjaśnienia."""
        if 'feature_impacts' in explanation:
            # SHAP format
            return [fi['feature'] for fi in explanation['feature_impacts']]
        elif 'feature_weights' in explanation:
            # LIME format
            # Wyodrębnij nazwy cech z opisów
            ranking = []
            for feat_desc, _ in explanation['feature_weights']:
                for fn in self.feature_names:
                    if fn in feat_desc:
                        if fn not in ranking:
                            ranking.append(fn)
                        break
            return ranking
        elif 'contributions' in explanation:
            # DALEX/EBM format
            return [c['feature'] if 'feature' in c else c['variable']
                    for c in explanation['contributions']]
        elif 'feature_importance' in explanation:
            # Global importance format
            return list(explanation['feature_importance'].keys())
        else:
            logger.warning(f"Nieznany format wyjaśnienia dla metody {method}")
            return []

    def _extract_importance_scores(
        self,
        explanation: Dict[str, Any],
        method: str
    ) -> Dict[str, float]:
        """Wyodrębnij scores ważności z wyjaśnienia."""
        scores = {}

        if 'feature_impacts' in explanation:
            # SHAP format
            for fi in explanation['feature_impacts']:
                scores[fi['feature']] = abs(fi['shap_value'])
        elif 'feature_weights' in explanation:
            # LIME format
            for feat_desc, weight in explanation['feature_weights']:
                for fn in self.feature_names:
                    if fn in feat_desc:
                        scores[fn] = abs(weight)
                        break
        elif 'contributions' in explanation:
            # DALEX/EBM format
            for c in explanation['contributions']:
                feat = c.get('feature', c.get('variable', ''))
                score = c.get('score', c.get('contribution', 0))
                scores[feat] = abs(score)
        elif 'feature_importance' in explanation:
            scores = {k: abs(v) for k, v in explanation['feature_importance'].items()}

        return scores

    def _calculate_ranking_agreement(
        self,
        rankings: Dict[str, List[str]]
    ) -> pd.DataFrame:
        """Oblicz macierz zgodności rankingów."""
        methods = list(rankings.keys())
        n_methods = len(methods)

        agreement = np.zeros((n_methods, n_methods))

        for i, m1 in enumerate(methods):
            for j, m2 in enumerate(methods):
                if i == j:
                    agreement[i, j] = 1.0
                else:
                    # Jaccard similarity for top features
                    set1 = set(rankings[m1])
                    set2 = set(rankings[m2])
                    if set1 or set2:
                        agreement[i, j] = len(set1 & set2) / len(set1 | set2)

        return pd.DataFrame(agreement, index=methods, columns=methods)

    def _find_common_top_features(
        self,
        rankings: Dict[str, List[str]],
        threshold: float = 0.5
    ) -> List[str]:
        """Znajdź cechy wspólne dla większości metod."""
        if not rankings:
            return []

        all_features = set()
        for ranking in rankings.values():
            all_features.update(ranking)

        n_methods = len(rankings)
        common = []

        for feature in all_features:
            count = sum(1 for ranking in rankings.values() if feature in ranking)
            if count / n_methods >= threshold:
                common.append(feature)

        return common

    def _calculate_spearman_correlations(
        self,
        importance_scores: Dict[str, Dict[str, float]]
    ) -> pd.DataFrame:
        """Oblicz korelacje Spearmana między metodami."""
        methods = list(importance_scores.keys())

        # Znajdź wspólne cechy
        all_features = set()
        for scores in importance_scores.values():
            all_features.update(scores.keys())

        common_features = list(all_features)

        # Utwórz macierz scores
        scores_matrix = {}
        for method, scores in importance_scores.items():
            scores_matrix[method] = [scores.get(f, 0) for f in common_features]

        # Oblicz korelacje
        n_methods = len(methods)
        correlations = np.zeros((n_methods, n_methods))

        for i, m1 in enumerate(methods):
            for j, m2 in enumerate(methods):
                if i == j:
                    correlations[i, j] = 1.0
                else:
                    corr, _ = stats.spearmanr(
                        scores_matrix[m1],
                        scores_matrix[m2]
                    )
                    correlations[i, j] = corr if not np.isnan(corr) else 0

        return pd.DataFrame(correlations, index=methods, columns=methods)

    def calculate_stability(
        self,
        explainer,
        instance: np.ndarray,
        method_name: str,
        n_runs: int = 100
    ) -> Dict[str, Any]:
        """
        Zbadaj stabilność wyjaśnień.

        Args:
            explainer: Obiekt explainera (LIME lub inny)
            instance: Instancja do wyjaśnienia
            method_name: Nazwa metody
            n_runs: Liczba powtórzeń

        Returns:
            Metryki stabilności
        """
        all_rankings = []
        all_scores = {f: [] for f in self.feature_names}

        for i in range(n_runs):
            try:
                exp = explainer.explain_instance(instance)
                scores = self._extract_importance_scores(exp, method_name)

                for feat in self.feature_names:
                    all_scores[feat].append(scores.get(feat, 0))

                ranking = self._extract_feature_ranking(exp, method_name)
                all_rankings.append(ranking[:5])  # Top 5

            except Exception as e:
                logger.warning(f"Błąd w iteracji {i}: {e}")
                continue

        if not all_rankings:
            return {'error': 'Brak poprawnych wyjaśnień'}

        # Metryki stabilności
        stability_metrics = {
            'method': method_name,
            'n_runs': n_runs,
            'n_successful': len(all_rankings),
            'feature_std': {
                feat: np.std(scores) for feat, scores in all_scores.items()
            },
            'feature_mean': {
                feat: np.mean(scores) for feat, scores in all_scores.items()
            },
            'top_feature_consistency': self._calculate_top_consistency(all_rankings),
            'ranking_stability': self._calculate_ranking_stability(all_rankings)
        }

        # Współczynnik zmienności
        cv_values = []
        for feat in self.feature_names:
            mean = stability_metrics['feature_mean'][feat]
            std = stability_metrics['feature_std'][feat]
            if abs(mean) > 0.001:
                cv_values.append(std / abs(mean))

        stability_metrics['mean_cv'] = np.mean(cv_values) if cv_values else 0

        self.comparison_results[f'stability_{method_name}'] = stability_metrics
        return stability_metrics

    def _calculate_top_consistency(
        self,
        rankings: List[List[str]]
    ) -> float:
        """Oblicz zgodność top cechy."""
        if not rankings:
            return 0.0

        top_features = [r[0] if r else None for r in rankings]
        top_features = [t for t in top_features if t is not None]

        if not top_features:
            return 0.0

        most_common = max(set(top_features), key=top_features.count)
        return top_features.count(most_common) / len(top_features)

    def _calculate_ranking_stability(
        self,
        rankings: List[List[str]]
    ) -> float:
        """Oblicz stabilność rankingu (średnia zgodność Jaccard)."""
        if len(rankings) < 2:
            return 1.0

        agreements = []
        for i in range(len(rankings)):
            for j in range(i + 1, len(rankings)):
                set1 = set(rankings[i])
                set2 = set(rankings[j])
                if set1 or set2:
                    jaccard = len(set1 & set2) / len(set1 | set2)
                    agreements.append(jaccard)

        return np.mean(agreements) if agreements else 0.0

    def calculate_agreement(
        self,
        explanations: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Oblicz ogólną zgodność między metodami XAI.

        Args:
            explanations: Słownik {nazwa_metody: wyjaśnienie}

        Returns:
            Metryki zgodności
        """
        # Porównaj rankingi
        ranking_comparison = self.compare_feature_rankings(explanations)

        # Oblicz średnią zgodność
        agreement_matrix = ranking_comparison['agreement_matrix']
        n = len(agreement_matrix)
        if n > 1:
            # Średnia z górnego trójkąta (bez diagonali)
            upper_triangle = agreement_matrix.values[np.triu_indices(n, k=1)]
            mean_agreement = np.mean(upper_triangle)
        else:
            mean_agreement = 1.0

        # Zgodność kierunku (czy wszystkie metody zgadzają się co do kierunku wpływu)
        direction_agreement = self._calculate_direction_agreement(explanations)

        result = {
            'mean_ranking_agreement': mean_agreement,
            'direction_agreement': direction_agreement,
            'spearman_mean': ranking_comparison['spearman_correlations'].values[
                np.triu_indices(n, k=1)
            ].mean() if n > 1 else 1.0,
            'common_features': ranking_comparison['common_top_features'],
            'n_methods': len(explanations)
        }

        self.comparison_results['agreement'] = result
        return result

    def _calculate_direction_agreement(
        self,
        explanations: Dict[str, Dict[str, Any]]
    ) -> float:
        """Oblicz zgodność kierunku wpływu cech."""
        if len(explanations) < 2:
            return 1.0

        # Zbierz kierunki dla każdej cechy
        feature_directions = {f: [] for f in self.feature_names}

        for method, exp in explanations.items():
            scores = self._extract_importance_scores(exp, method)

            # Określ kierunek z oryginalnego wyjaśnienia
            if 'feature_impacts' in exp:
                for fi in exp['feature_impacts']:
                    feat = fi['feature']
                    direction = 1 if fi['shap_value'] > 0 else -1
                    if feat in feature_directions:
                        feature_directions[feat].append(direction)
            elif 'contributions' in exp:
                for c in exp['contributions']:
                    feat = c.get('feature', c.get('variable', ''))
                    score = c.get('score', c.get('contribution', 0))
                    direction = 1 if score > 0 else -1
                    if feat in feature_directions:
                        feature_directions[feat].append(direction)

        # Oblicz zgodność
        agreements = []
        for feat, directions in feature_directions.items():
            if len(directions) >= 2:
                # Czy wszystkie kierunki są takie same?
                agreement = 1.0 if len(set(directions)) == 1 else 0.0
                agreements.append(agreement)

        return np.mean(agreements) if agreements else 1.0

    def plot_ranking_comparison(
        self,
        rankings: Optional[Dict[str, List[str]]] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Wizualizacja porównania rankingów.

        Args:
            rankings: Słownik z rankingami (lub użyj zapisanych)
            save_path: Ścieżka do zapisu

        Returns:
            Figura matplotlib
        """
        if rankings is None:
            if 'rankings' not in self.comparison_results:
                raise ValueError("Brak danych do wizualizacji. Wykonaj compare_feature_rankings()")
            rankings = self.comparison_results['rankings']['rankings']

        methods = list(rankings.keys())
        n_methods = len(methods)

        # Przygotuj dane do wykresu
        all_features = set()
        for ranking in rankings.values():
            all_features.update(ranking[:10])

        features = list(all_features)

        # Macierz pozycji
        position_matrix = np.zeros((len(features), n_methods))

        for j, method in enumerate(methods):
            for i, feature in enumerate(features):
                if feature in rankings[method]:
                    pos = rankings[method].index(feature) + 1
                    position_matrix[i, j] = pos
                else:
                    position_matrix[i, j] = np.nan

        fig, ax = plt.subplots(figsize=(12, 8))

        # Heatmapa pozycji
        im = ax.imshow(position_matrix, cmap='RdYlGn_r', aspect='auto')

        ax.set_xticks(np.arange(n_methods))
        ax.set_yticks(np.arange(len(features)))
        ax.set_xticklabels(methods, fontsize=10)
        ax.set_yticklabels(features, fontsize=9)

        # Dodaj wartości
        for i in range(len(features)):
            for j in range(n_methods):
                val = position_matrix[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f'{int(val)}', ha='center', va='center', fontsize=9)

        plt.colorbar(im, label='Pozycja w rankingu')
        ax.set_title('Porównanie rankingów cech między metodami XAI')
        ax.set_xlabel('Metoda XAI')
        ax.set_ylabel('Cecha')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Wykres porównania zapisany do {save_path}")

        return fig

    def plot_agreement_heatmap(
        self,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Wykres heatmapy zgodności między metodami.

        Args:
            save_path: Ścieżka do zapisu

        Returns:
            Figura matplotlib
        """
        if 'rankings' not in self.comparison_results:
            raise ValueError("Brak danych. Wykonaj compare_feature_rankings()")

        agreement_matrix = self.comparison_results['rankings']['agreement_matrix']

        fig, ax = plt.subplots(figsize=(8, 6))

        sns.heatmap(
            agreement_matrix,
            annot=True,
            fmt='.2f',
            cmap='YlGnBu',
            ax=ax,
            vmin=0,
            vmax=1
        )

        ax.set_title('Zgodność rankingów między metodami XAI\n(Jaccard Similarity)')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Heatmapa zgodności zapisana do {save_path}")

        return fig

    def generate_comparison_report(
        self,
        explanations: Dict[str, Dict[str, Any]],
        output_path: Optional[str] = None
    ) -> str:
        """
        Wygeneruj raport tekstowy porównania metod XAI.

        Args:
            explanations: Słownik z wyjaśnieniami
            output_path: Opcjonalna ścieżka do zapisu

        Returns:
            Raport jako string
        """
        # Wykonaj porównania
        ranking_comp = self.compare_feature_rankings(explanations)
        agreement = self.calculate_agreement(explanations)

        report_lines = [
            "=" * 60,
            "RAPORT PORÓWNANIA METOD XAI",
            "=" * 60,
            "",
            f"Liczba porównywanych metod: {agreement['n_methods']}",
            f"Metody: {', '.join(explanations.keys())}",
            "",
            "-" * 40,
            "ZGODNOŚĆ METOD",
            "-" * 40,
            f"Średnia zgodność rankingów (Jaccard): {agreement['mean_ranking_agreement']:.3f}",
            f"Średnia korelacja Spearmana: {agreement['spearman_mean']:.3f}",
            f"Zgodność kierunku wpływu: {agreement['direction_agreement']:.1%}",
            "",
            "-" * 40,
            "WSPÓLNE NAJWAŻNIEJSZE CECHY",
            "-" * 40,
        ]

        for feat in agreement['common_features'][:10]:
            report_lines.append(f"  • {feat}")

        report_lines.extend([
            "",
            "-" * 40,
            "RANKINGI POSZCZEGÓLNYCH METOD",
            "-" * 40,
        ])

        for method, ranking in ranking_comp['rankings'].items():
            report_lines.append(f"\n{method}:")
            for i, feat in enumerate(ranking[:5], 1):
                report_lines.append(f"  {i}. {feat}")

        report_lines.extend([
            "",
            "-" * 40,
            "MACIERZ ZGODNOŚCI (Jaccard)",
            "-" * 40,
            ranking_comp['agreement_matrix'].to_string(),
            "",
            "-" * 40,
            "KORELACJE SPEARMANA",
            "-" * 40,
            ranking_comp['spearman_correlations'].to_string(),
            "",
            "=" * 60,
            "KONIEC RAPORTU",
            "=" * 60
        ])

        report = "\n".join(report_lines)

        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"Raport zapisany do {output_path}")

        return report

    def to_json(self) -> str:
        """
        Serializuj wyniki porównania do JSON.

        Returns:
            String JSON
        """
        # Konwertuj DataFrames do dict
        results = {}
        for key, value in self.comparison_results.items():
            if isinstance(value, dict):
                converted = {}
                for k, v in value.items():
                    if isinstance(v, pd.DataFrame):
                        converted[k] = v.to_dict()
                    elif isinstance(v, np.ndarray):
                        converted[k] = v.tolist()
                    else:
                        converted[k] = v
                results[key] = converted
            else:
                results[key] = value

        return json.dumps(results, ensure_ascii=False, indent=2, default=str)
