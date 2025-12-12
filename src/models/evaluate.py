"""
Moduł ewaluacji modeli ML z metrykami medycznymi.

Zawiera funkcje do oceny modeli klasyfikacji z uwzględnieniem
specyfiki zastosowań medycznych (sensitivity, specificity, NPV, PPV).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score, recall_score,
    f1_score, accuracy_score, brier_score_loss, confusion_matrix,
    roc_curve, precision_recall_curve, classification_report,
    log_loss, matthews_corrcoef, cohen_kappa_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, List, Tuple, Union
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Klasa do ewaluacji modeli z metrykami medycznymi.

    Zapewnia kompleksową ocenę modeli klasyfikacji binarnej
    z uwzględnieniem metryk istotnych w kontekście medycznym.
    """

    def __init__(self):
        """Inicjalizacja ewaluatora."""
        self.results: Dict[str, Dict[str, Any]] = {}
        self.comparison_df: Optional[pd.DataFrame] = None

    def calculate_medical_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Oblicz metryki istotne klinicznie.

        Args:
            y_true: Prawdziwe etykiety
            y_pred: Przewidywane etykiety
            y_proba: Prawdopodobieństwa klasy pozytywnej

        Returns:
            Słownik z metrykami
        """
        # Macierz konfuzji
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        metrics = {
            # Podstawowe metryki
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),  # = Sensitivity
            'f1': f1_score(y_true, y_pred, zero_division=0),

            # Metryki medyczne
            'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,  # True Positive Rate
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,  # True Negative Rate
            'ppv': tp / (tp + fp) if (tp + fp) > 0 else 0,  # Positive Predictive Value
            'npv': tn / (tn + fn) if (tn + fn) > 0 else 0,  # Negative Predictive Value

            # Dodatkowe metryki
            'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'false_negative_rate': fn / (fn + tp) if (fn + tp) > 0 else 0,
            'balanced_accuracy': (tp / (tp + fn) + tn / (tn + fp)) / 2 if (tp + fn) > 0 and (tn + fp) > 0 else 0,
            'mcc': matthews_corrcoef(y_true, y_pred),  # Matthews Correlation Coefficient
            'kappa': cohen_kappa_score(y_true, y_pred),

            # Confusion matrix values
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn)
        }

        # Metryki wymagające prawdopodobieństw
        if y_proba is not None:
            metrics['auc_roc'] = roc_auc_score(y_true, y_proba)
            metrics['auc_pr'] = average_precision_score(y_true, y_proba)
            metrics['brier_score'] = brier_score_loss(y_true, y_proba)
            metrics['log_loss'] = log_loss(y_true, y_proba)

        return metrics

    def evaluate_model(
        self,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str = 'model',
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Pełna ewaluacja modelu.

        Args:
            model: Wytrenowany model
            X_test: Dane testowe
            y_test: Etykiety testowe
            model_name: Nazwa modelu (do identyfikacji)
            threshold: Próg klasyfikacji

        Returns:
            Słownik z pełną ewaluacją
        """
        logger.info(f"Ewaluacja modelu: {model_name}")

        # Predykcje
        y_pred = model.predict(X_test)

        # Prawdopodobieństwa
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = None

        # Predykcje z custom threshold
        if y_proba is not None and threshold != 0.5:
            y_pred_threshold = (y_proba >= threshold).astype(int)
        else:
            y_pred_threshold = y_pred

        # Oblicz metryki
        metrics = self.calculate_medical_metrics(y_test, y_pred_threshold, y_proba)

        # Raport klasyfikacji
        class_report = classification_report(
            y_test, y_pred_threshold,
            target_names=['Przeżycie', 'Zgon'],
            output_dict=True
        )

        result = {
            'model_name': model_name,
            'metrics': metrics,
            'classification_report': class_report,
            'threshold': threshold,
            'n_samples': len(y_test),
            'class_distribution': {
                'actual_positive': int(sum(y_test)),
                'actual_negative': int(len(y_test) - sum(y_test)),
                'predicted_positive': int(sum(y_pred_threshold)),
                'predicted_negative': int(len(y_test) - sum(y_pred_threshold))
            }
        }

        # Zapisz wyniki
        self.results[model_name] = result

        # Loguj kluczowe metryki
        logger.info(f"  AUC-ROC: {metrics.get('auc_roc', 'N/A'):.4f}" if y_proba is not None else "  AUC-ROC: N/A")
        logger.info(f"  Sensitivity: {metrics['sensitivity']:.4f}")
        logger.info(f"  Specificity: {metrics['specificity']:.4f}")
        logger.info(f"  PPV: {metrics['ppv']:.4f}")
        logger.info(f"  NPV: {metrics['npv']:.4f}")

        return result

    def find_optimal_threshold(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        metric: str = 'youden',
        min_sensitivity: float = 0.8
    ) -> Tuple[float, Dict[str, float]]:
        """
        Znajdź optymalny próg klasyfikacji.

        Args:
            y_true: Prawdziwe etykiety
            y_proba: Prawdopodobieństwa
            metric: Metoda optymalizacji ('youden', 'f1', 'balanced')
            min_sensitivity: Minimalna wymagana czułość

        Returns:
            Tuple (optymalny_próg, metryki przy tym progu)
        """
        thresholds = np.linspace(0.1, 0.9, 81)
        best_threshold = 0.5
        best_score = -np.inf
        best_metrics = {}

        for thresh in thresholds:
            y_pred = (y_proba >= thresh).astype(int)
            metrics = self.calculate_medical_metrics(y_true, y_pred, y_proba)

            # Sprawdź minimalną czułość
            if metrics['sensitivity'] < min_sensitivity:
                continue

            # Oblicz score wg wybranej metody
            if metric == 'youden':
                score = metrics['sensitivity'] + metrics['specificity'] - 1
            elif metric == 'f1':
                score = metrics['f1']
            elif metric == 'balanced':
                score = metrics['balanced_accuracy']
            else:
                score = metrics.get(metric, 0)

            if score > best_score:
                best_score = score
                best_threshold = thresh
                best_metrics = metrics

        logger.info(f"Optymalny próg ({metric}): {best_threshold:.2f}")
        logger.info(f"  Sensitivity: {best_metrics.get('sensitivity', 0):.4f}")
        logger.info(f"  Specificity: {best_metrics.get('specificity', 0):.4f}")

        return best_threshold, best_metrics

    def bootstrap_confidence_intervals(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        n_iterations: int = 1000,
        confidence: float = 0.95,
        random_state: int = 42
    ) -> Dict[str, Tuple[float, float, float]]:
        """
        Oblicz przedziały ufności dla metryk metodą bootstrap.

        Args:
            y_true: Prawdziwe etykiety
            y_proba: Prawdopodobieństwa
            n_iterations: Liczba iteracji bootstrap
            confidence: Poziom ufności
            random_state: Ziarno losowości

        Returns:
            Słownik {metryka: (dolna_granica, średnia, górna_granica)}
        """
        np.random.seed(random_state)

        metrics_bootstrap = {
            'auc_roc': [],
            'auc_pr': [],
            'sensitivity': [],
            'specificity': [],
            'ppv': [],
            'npv': []
        }

        n_samples = len(y_true)

        for _ in range(n_iterations):
            # Losuj indeksy z powtórzeniami
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            y_true_boot = y_true[indices]
            y_proba_boot = y_proba[indices]
            y_pred_boot = (y_proba_boot >= 0.5).astype(int)

            # Oblicz metryki
            try:
                metrics = self.calculate_medical_metrics(y_true_boot, y_pred_boot, y_proba_boot)
                for key in metrics_bootstrap:
                    if key in metrics:
                        metrics_bootstrap[key].append(metrics[key])
            except Exception:
                continue

        # Oblicz przedziały ufności
        alpha = (1 - confidence) / 2
        ci_results = {}

        for metric, values in metrics_bootstrap.items():
            if values:
                values = np.array(values)
                lower = np.percentile(values, alpha * 100)
                mean = np.mean(values)
                upper = np.percentile(values, (1 - alpha) * 100)
                ci_results[metric] = (lower, mean, upper)

        return ci_results

    def compare_models(
        self,
        models: Dict[str, Any],
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> pd.DataFrame:
        """
        Porównaj wiele modeli.

        Args:
            models: Słownik {nazwa: model}
            X_test: Dane testowe
            y_test: Etykiety testowe

        Returns:
            DataFrame z porównaniem
        """
        comparison_data = []

        for name, model in models.items():
            result = self.evaluate_model(model, X_test, y_test, model_name=name)
            metrics = result['metrics']

            comparison_data.append({
                'Model': name,
                'AUC-ROC': metrics.get('auc_roc', np.nan),
                'AUC-PR': metrics.get('auc_pr', np.nan),
                'Sensitivity': metrics['sensitivity'],
                'Specificity': metrics['specificity'],
                'PPV': metrics['ppv'],
                'NPV': metrics['npv'],
                'F1': metrics['f1'],
                'Accuracy': metrics['accuracy'],
                'Brier Score': metrics.get('brier_score', np.nan)
            })

        self.comparison_df = pd.DataFrame(comparison_data)
        self.comparison_df = self.comparison_df.sort_values('AUC-ROC', ascending=False)

        return self.comparison_df

    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        model_name: str = 'Model',
        ax: Optional[plt.Axes] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Wykres krzywej ROC.

        Args:
            y_true: Prawdziwe etykiety
            y_proba: Prawdopodobieństwa
            model_name: Nazwa modelu
            ax: Opcjonalny subplot
            save_path: Ścieżka do zapisu

        Returns:
            Figura matplotlib
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.figure

        ax.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})', linewidth=2)
        ax.plot([0, 1], [0, 1], 'k--', label='Losowy klasyfikator')
        ax.set_xlabel('False Positive Rate (1 - Specificity)')
        ax.set_ylabel('True Positive Rate (Sensitivity)')
        ax.set_title('Krzywa ROC')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Wykres ROC zapisany do {save_path}")

        return fig

    def plot_precision_recall_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        model_name: str = 'Model',
        ax: Optional[plt.Axes] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Wykres krzywej Precision-Recall.

        Args:
            y_true: Prawdziwe etykiety
            y_proba: Prawdopodobieństwa
            model_name: Nazwa modelu
            ax: Opcjonalny subplot
            save_path: Ścieżka do zapisu

        Returns:
            Figura matplotlib
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        ap = average_precision_score(y_true, y_proba)

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.figure

        ax.plot(recall, precision, label=f'{model_name} (AP = {ap:.3f})', linewidth=2)

        # Linia bazowa (proporcja klasy pozytywnej)
        baseline = sum(y_true) / len(y_true)
        ax.axhline(y=baseline, color='k', linestyle='--', label=f'Bazowa ({baseline:.2f})')

        ax.set_xlabel('Recall (Sensitivity)')
        ax.set_ylabel('Precision (PPV)')
        ax.set_title('Krzywa Precision-Recall')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Wykres PR zapisany do {save_path}")

        return fig

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = 'Model',
        normalize: bool = True,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Wykres macierzy konfuzji.

        Args:
            y_true: Prawdziwe etykiety
            y_pred: Przewidywane etykiety
            model_name: Nazwa modelu
            normalize: Czy normalizować
            save_path: Ścieżka do zapisu

        Returns:
            Figura matplotlib
        """
        cm = confusion_matrix(y_true, y_pred)

        if normalize:
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_display = cm_normalized
            fmt = '.2%'
        else:
            cm_display = cm
            fmt = 'd'

        fig, ax = plt.subplots(figsize=(8, 6))

        sns.heatmap(
            cm_display,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=['Przeżycie', 'Zgon'],
            yticklabels=['Przeżycie', 'Zgon'],
            ax=ax
        )

        # Dodaj wartości bezwzględne
        if normalize:
            for i in range(2):
                for j in range(2):
                    ax.text(
                        j + 0.5, i + 0.7, f'(n={cm[i, j]})',
                        ha='center', va='center', fontsize=10, color='gray'
                    )

        ax.set_xlabel('Predykcja')
        ax.set_ylabel('Rzeczywistość')
        ax.set_title(f'Macierz konfuzji - {model_name}')

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Macierz konfuzji zapisana do {save_path}")

        return fig

    def plot_model_comparison(
        self,
        metrics: List[str] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Wykres porównania modeli.

        Args:
            metrics: Lista metryk do porównania
            save_path: Ścieżka do zapisu

        Returns:
            Figura matplotlib
        """
        if self.comparison_df is None:
            raise ValueError("Najpierw wykonaj compare_models()")

        if metrics is None:
            metrics = ['AUC-ROC', 'Sensitivity', 'Specificity', 'PPV', 'NPV']

        # Filtruj dostępne metryki
        available_metrics = [m for m in metrics if m in self.comparison_df.columns]

        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(self.comparison_df))
        width = 0.15
        multiplier = 0

        for metric in available_metrics:
            offset = width * multiplier
            bars = ax.bar(
                x + offset,
                self.comparison_df[metric],
                width,
                label=metric
            )
            multiplier += 1

        ax.set_xlabel('Model')
        ax.set_ylabel('Wartość metryki')
        ax.set_title('Porównanie modeli')
        ax.set_xticks(x + width * (len(available_metrics) - 1) / 2)
        ax.set_xticklabels(self.comparison_df['Model'], rotation=45, ha='right')
        ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        ax.set_ylim(0, 1.1)
        ax.axhline(y=0.8, color='r', linestyle='--', alpha=0.5, label='Próg 0.8')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Porównanie modeli zapisane do {save_path}")

        return fig

    def generate_report(self, output_path: Optional[str] = None) -> str:
        """
        Wygeneruj raport tekstowy.

        Args:
            output_path: Opcjonalna ścieżka do zapisu

        Returns:
            Raport jako string
        """
        report_lines = [
            "=" * 60,
            "RAPORT EWALUACJI MODELI",
            "=" * 60,
            ""
        ]

        for model_name, result in self.results.items():
            metrics = result['metrics']
            report_lines.extend([
                f"\n{'='*40}",
                f"Model: {model_name}",
                f"{'='*40}",
                "",
                "METRYKI MEDYCZNE:",
                f"  AUC-ROC:     {metrics.get('auc_roc', 'N/A'):.4f}" if 'auc_roc' in metrics else "  AUC-ROC:     N/A",
                f"  AUC-PR:      {metrics.get('auc_pr', 'N/A'):.4f}" if 'auc_pr' in metrics else "  AUC-PR:      N/A",
                f"  Sensitivity: {metrics['sensitivity']:.4f}",
                f"  Specificity: {metrics['specificity']:.4f}",
                f"  PPV:         {metrics['ppv']:.4f}",
                f"  NPV:         {metrics['npv']:.4f}",
                "",
                "POZOSTAŁE METRYKI:",
                f"  Accuracy:    {metrics['accuracy']:.4f}",
                f"  F1-Score:    {metrics['f1']:.4f}",
                f"  MCC:         {metrics['mcc']:.4f}",
                f"  Brier Score: {metrics.get('brier_score', 'N/A'):.4f}" if 'brier_score' in metrics else "  Brier Score: N/A",
                "",
                "MACIERZ KONFUZJI:",
                f"  TP: {metrics['true_positives']}  FP: {metrics['false_positives']}",
                f"  FN: {metrics['false_negatives']}  TN: {metrics['true_negatives']}"
            ])

        report = "\n".join(report_lines)

        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"Raport zapisany do {output_path}")

        return report

    def save_results(self, filepath: str) -> None:
        """
        Zapisz wyniki do JSON.

        Args:
            filepath: Ścieżka do pliku
        """
        # Konwertuj numpy arrays do list
        results_serializable = {}
        for model_name, result in self.results.items():
            results_serializable[model_name] = {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in result.items()
            }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results_serializable, f, ensure_ascii=False, indent=2)

        logger.info(f"Wyniki zapisane do {filepath}")
