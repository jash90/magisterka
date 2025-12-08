"""
Konfiguracja modeli ML dla systemu XAI.

Zawiera hiperparametry bazowe i siatki do GridSearchCV
dla wszystkich obsługiwanych modeli.
"""

from typing import Dict, Any, List

# Metryki medyczne do ewaluacji
MEDICAL_METRICS = [
    'roc_auc',
    'average_precision',
    'recall',  # Sensitivity - kluczowe w medycynie
    'precision',
    'f1',
    'brier_score'
]

# Konfiguracje modeli
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    'random_forest': {
        'class': 'sklearn.ensemble.RandomForestClassifier',
        'base_params': {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'class_weight': 'balanced',
            'random_state': 42,
            'n_jobs': -1
        },
        'grid_search': {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [5, 10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        },
        'random_search': {
            'n_estimators': [50, 100, 150, 200, 250, 300],
            'max_depth': [3, 5, 7, 10, 15, 20, None],
            'min_samples_split': [2, 3, 5, 7, 10],
            'min_samples_leaf': [1, 2, 3, 4, 5],
            'max_features': ['sqrt', 'log2', None]
        }
    },

    'xgboost': {
        'class': 'xgboost.XGBClassifier',
        'base_params': {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6,
            'min_child_weight': 1,
            'gamma': 0,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'scale_pos_weight': 5,  # Dostosować do proporcji klas
            'random_state': 42,
            'eval_metric': 'auc',
            'use_label_encoder': False
        },
        'grid_search': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7, 9],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0]
        },
        'random_search': {
            'n_estimators': [50, 75, 100, 150, 200, 250],
            'learning_rate': [0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2],
            'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
            'min_child_weight': [1, 2, 3, 4, 5],
            'gamma': [0, 0.1, 0.2, 0.3],
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0]
        }
    },

    'lightgbm': {
        'class': 'lightgbm.LGBMClassifier',
        'base_params': {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'num_leaves': 31,
            'max_depth': -1,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'is_unbalance': True,
            'random_state': 42,
            'verbose': -1
        },
        'grid_search': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'num_leaves': [15, 31, 63],
            'max_depth': [-1, 5, 10, 15],
            'min_child_samples': [10, 20, 30]
        },
        'random_search': {
            'n_estimators': [50, 75, 100, 150, 200],
            'learning_rate': [0.01, 0.02, 0.05, 0.08, 0.1, 0.15],
            'num_leaves': [15, 20, 25, 31, 40, 50, 63],
            'max_depth': [-1, 5, 7, 10, 12, 15],
            'min_child_samples': [5, 10, 15, 20, 25, 30],
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0]
        }
    },

    'neural_network': {
        'class': 'sklearn.neural_network.MLPClassifier',
        'base_params': {
            'hidden_layer_sizes': (100, 50),
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.0001,
            'learning_rate': 'adaptive',
            'learning_rate_init': 0.001,
            'max_iter': 500,
            'early_stopping': True,
            'validation_fraction': 0.1,
            'random_state': 42
        },
        'grid_search': {
            'hidden_layer_sizes': [(50,), (100,), (50, 25), (100, 50), (100, 50, 25)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate_init': [0.001, 0.01]
        },
        'random_search': {
            'hidden_layer_sizes': [
                (25,), (50,), (75,), (100,),
                (25, 10), (50, 25), (75, 25), (100, 50),
                (50, 25, 10), (100, 50, 25)
            ],
            'activation': ['relu', 'tanh', 'logistic'],
            'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1],
            'learning_rate_init': [0.0001, 0.001, 0.005, 0.01]
        }
    },

    'logistic_regression': {
        'class': 'sklearn.linear_model.LogisticRegression',
        'base_params': {
            'penalty': 'l2',
            'C': 1.0,
            'class_weight': 'balanced',
            'solver': 'lbfgs',
            'max_iter': 1000,
            'random_state': 42
        },
        'grid_search': {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        },
        'random_search': {
            'C': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100],
            'penalty': ['l1', 'l2', 'elasticnet'],
            'solver': ['liblinear', 'saga', 'newton-cg', 'lbfgs']
        }
    },

    'svm': {
        'class': 'sklearn.svm.SVC',
        'base_params': {
            'kernel': 'rbf',
            'C': 1.0,
            'gamma': 'scale',
            'class_weight': 'balanced',
            'probability': True,
            'random_state': 42
        },
        'grid_search': {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
            'kernel': ['rbf', 'poly', 'sigmoid']
        },
        'random_search': {
            'C': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],
            'gamma': ['scale', 'auto', 0.0001, 0.001, 0.01, 0.1, 1],
            'kernel': ['rbf', 'poly', 'sigmoid', 'linear']
        }
    },

    'gradient_boosting': {
        'class': 'sklearn.ensemble.GradientBoostingClassifier',
        'base_params': {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 5,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'subsample': 0.8,
            'random_state': 42
        },
        'grid_search': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10],
            'subsample': [0.6, 0.8, 1.0]
        },
        'random_search': {
            'n_estimators': [50, 75, 100, 150, 200],
            'learning_rate': [0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2],
            'max_depth': [3, 4, 5, 6, 7, 8],
            'min_samples_split': [2, 3, 5, 7, 10],
            'min_samples_leaf': [1, 2, 3, 4],
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0]
        }
    }
}

# Optymalna kolejność trenowania (od szybszych do wolniejszych)
TRAINING_ORDER: List[str] = [
    'logistic_regression',
    'random_forest',
    'lightgbm',
    'xgboost',
    'gradient_boosting',
    'svm',
    'neural_network'
]

# Modele kompatybilne z TreeSHAP (szybkie SHAP)
TREE_SHAP_COMPATIBLE: List[str] = [
    'random_forest',
    'xgboost',
    'lightgbm',
    'gradient_boosting'
]

# Progi dla metryk medycznych
MEDICAL_THRESHOLDS = {
    'auc_roc_min': 0.75,  # Minimalny akceptowalny AUC-ROC
    'sensitivity_min': 0.80,  # Minimalna czułość (ważne w medycynie!)
    'specificity_min': 0.60,  # Minimalna swoistość
    'ppv_min': 0.50,  # Minimum positive predictive value
    'npv_min': 0.90  # Minimum negative predictive value
}


def get_model_class(model_type: str):
    """
    Pobierz klasę modelu na podstawie nazwy.

    Args:
        model_type: Nazwa typu modelu

    Returns:
        Klasa modelu
    """
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Nieznany model: {model_type}. Dostępne: {list(MODEL_CONFIGS.keys())}")

    class_path = MODEL_CONFIGS[model_type]['class']
    module_name, class_name = class_path.rsplit('.', 1)

    import importlib
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def get_base_params(model_type: str, scale_pos_weight: float = None) -> Dict[str, Any]:
    """
    Pobierz bazowe parametry dla modelu.

    Args:
        model_type: Nazwa typu modelu
        scale_pos_weight: Opcjonalna waga dla niezbalansowanych klas

    Returns:
        Słownik parametrów
    """
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Nieznany model: {model_type}")

    params = MODEL_CONFIGS[model_type]['base_params'].copy()

    # Ustaw scale_pos_weight dla XGBoost jeśli podano
    if scale_pos_weight is not None and model_type == 'xgboost':
        params['scale_pos_weight'] = scale_pos_weight

    return params


def get_grid_search_params(model_type: str) -> Dict[str, List]:
    """
    Pobierz siatkę parametrów do GridSearchCV.

    Args:
        model_type: Nazwa typu modelu

    Returns:
        Słownik z siatką parametrów
    """
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Nieznany model: {model_type}")

    return MODEL_CONFIGS[model_type].get('grid_search', {})


def get_random_search_params(model_type: str) -> Dict[str, List]:
    """
    Pobierz parametry do RandomizedSearchCV.

    Args:
        model_type: Nazwa typu modelu

    Returns:
        Słownik z parametrami
    """
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Nieznany model: {model_type}")

    return MODEL_CONFIGS[model_type].get('random_search', {})
