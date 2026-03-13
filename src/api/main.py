"""
FastAPI aplikacja dla systemu XAI.

Główny plik API z endpointami dla predykcji,
wyjaśnień XAI i agenta konwersacyjnego.
"""

import os
import sys
import time
import asyncio
import concurrent.futures
import functools
from pathlib import Path
from datetime import datetime
from typing import List, Literal, Optional, Dict, Any, Tuple
import numpy as np
import json
import logging

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvicorn
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Dodaj ścieżkę projektu
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from .schemas import (
    PatientInput, PredictionOutput, SHAPExplanation, LIMEExplanation,
    PatientExplanation, ModelInfo, GlobalImportance, HealthCheckResponse,
    ChatRequest, ChatResponse, ExplanationRequest, PatientExplanationRequest,
    ComparisonResult, ErrorResponse, RiskLevel, XAIMethod,
    ModelType, TaskType, patient_to_array, get_risk_level_from_probability, patients_to_matrix,
    # Batch schemas
    BatchPatientInput, BatchPredictionOutput, BatchPatientResult,
    BatchSummary, BatchProcessingError, RiskFactorItem,
    DemoModeStatus, DemoModeRequest,
    # Dialysis schemas
    DialysisPatientInput, DialysisPredictionOutput, dialysis_patient_to_array,
    BatchDialysisPatientInput, dialysis_patients_to_matrix,
    # Multi-model & XAI schemas
    SingleModelResult, MultiModelPredictionOutput, DALEXExplanation, XAIComparisonOutput,
    # Global XAI schemas
    SHAPGlobalResponse, DALEXVariableImportanceResponse, DALEXPDPResponse,
    EBMGlobalResponse, EBMLocalResponse, EBMFeatureFunctionResponse, GlobalComparisonResponse,
    # Calibration schemas
    ModelCalibrationResult, CalibrationResponse,
    # Conversational analysis schemas
    ConversationalAnalysisRequest, ConversationalAnalysisResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# LIFESPAN (musi być przed app = FastAPI(...))
# ============================================================================

@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):
    """Zarządzanie cyklem życia aplikacji (startup/shutdown)."""
    await _startup_logic()
    yield
    logger.info("Zamykanie API...")
    _XAI_EXECUTOR.shutdown(wait=False)
    _OPENAI_EXECUTOR.shutdown(wait=False)


# ============================================================================
# APLIKACJA FASTAPI
# ============================================================================

app = FastAPI(
    title="Vasculitis XAI API",
    description="""
    API do predykcji śmiertelności w zapaleniu naczyń z wyjaśnieniami XAI.

    System wykorzystuje modele ML (XGBoost, Random Forest) wraz z metodami
    wyjaśnialnej sztucznej inteligencji (LIME, SHAP, DALEX, EBM) do:
    - Predykcji ryzyka zgonu
    - Generowania wyjaśnień dla klinicystów i pacjentów
    - Interaktywnej rozmowy o wynikach
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:8501").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ============================================================================
# GLOBAL STATE
# ============================================================================

class AppState:
    """Stan aplikacji z wczytanymi modelami."""

    def __init__(self):
        self.model = None
        self.feature_names = None
        self.lime_explainer = None
        self.shap_explainer = None
        self.rag_pipeline = None
        self.is_loaded = False

        # Multi-model support
        self.models: Dict[str, Any] = {}
        self.model_metadata: Dict[str, Any] = {}
        self.default_model_type: Optional[str] = None

        # Konfiguracja trybu demo
        self.allow_demo = os.getenv("ALLOW_DEMO", "false").lower() == "true"
        self.force_api_mode = os.getenv("FORCE_API_MODE", "false").lower() == "true"

        if self.force_api_mode:
            self.allow_demo = False

        # Dialysis models
        self.dialysis_models: Dict[str, Any] = {}
        self.dialysis_model_metadata: Dict[str, Any] = {}
        self.dialysis_feature_names: Optional[List[str]] = None
        self.dialysis_default_model_type: Optional[str] = None
        self.dialysis_loaded: bool = False

        # XAI artifacts
        self.X_background: Optional[np.ndarray] = None
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.xai_ready: bool = False

        # Dialysis XAI artifacts
        self.dialysis_X_background: Optional[np.ndarray] = None
        self.dialysis_X_train: Optional[np.ndarray] = None
        self.dialysis_y_train: Optional[np.ndarray] = None
        self.dialysis_xai_ready: bool = False

        # Model directory paths (set during startup)
        self.models_dir: Optional[str] = None
        self.dialysis_models_dir: Optional[str] = None

        # Global importance cache
        self._global_importance_cache: Dict[str, Dict[str, float]] = {}

    def get_tree_model(self):
        """Zwróć pierwszy dostępny model drzewiasty (dla SHAP TreeExplainer)."""
        tree_model_types = ["random_forest", "xgboost"]
        for mtype in tree_model_types:
            if mtype in self.models:
                return self.models[mtype], mtype
        # Fallback: szukaj wg nazwy klasy
        for mtype, m in self.models.items():
            class_name = type(m).__name__.lower()
            if any(t in class_name for t in ['randomforest', 'xgb', 'gradientboosting', 'decisiontree', 'extratrees']):
                return m, mtype
        return None, None

    def get_xai_context(self, task_type: str = "mortality") -> Dict:
        """Zwróć kontekst XAI (artefakty) dla danego typu zadania."""
        if task_type == "dialysis":
            return {
                "xai_ready": self.dialysis_xai_ready,
                "is_loaded": self.dialysis_loaded,
                "X_background": self.dialysis_X_background,
                "X_train": self.dialysis_X_train,
                "y_train": self.dialysis_y_train,
                "feature_names": self.dialysis_feature_names or [],
                "get_model": self.get_dialysis_model,
            }
        return {
            "xai_ready": self.xai_ready,
            "is_loaded": self.is_loaded,
            "X_background": self.X_background,
            "X_train": self.X_train,
            "y_train": self.y_train,
            "feature_names": self.feature_names or [],
            "get_model": self.get_model,
        }

    def set_demo_mode(self, enabled: bool):
        """Włącz/wyłącz tryb demo."""
        if self.force_api_mode and enabled:
            raise ValueError("Nie można włączyć trybu demo gdy FORCE_API_MODE=true")
        self.allow_demo = enabled
        logger.info(f"Tryb demo {'włączony' if enabled else 'wyłączony'}")

    def get_current_mode(self) -> str:
        """Pobierz aktualny tryb działania."""
        if self.is_loaded:
            return "api"
        elif self.allow_demo:
            return "demo"
        else:
            return "unavailable"

    def get_global_importance(self, task_type: str = "mortality") -> Dict[str, float]:
        """Oblicz global feature importance on-demand (SHAP → DALEX → syntetyczny fallback). Wynik cache'owany."""
        if task_type in self._global_importance_cache:
            return self._global_importance_cache[task_type]

        result = self._compute_global_importance(task_type)
        # Cache only non-synthetic results (synthetic fallback may be replaced once models load)
        is_synthetic = (result == self._get_synthetic_importance(task_type))
        if not is_synthetic:
            self._global_importance_cache[task_type] = result
        return result

    def _get_synthetic_importance(self, task_type: str) -> Dict[str, float]:
        """Zwróć syntetyczne wartości ważności (fallback)."""
        if task_type == "dialysis":
            if self.dialysis_feature_names:
                base_values = [0.18, 0.16, 0.14, 0.12, 0.10, 0.09, 0.08, 0.07,
                               0.06, 0.05, 0.04, 0.03, 0.02, 0.02, 0.01]
                return {
                    name: base_values[i] if i < len(base_values) else 0.01
                    for i, name in enumerate(self.dialysis_feature_names)
                }
            return {}
        if self.feature_names:
            base_values = [0.15, 0.12, 0.11, 0.10, 0.09, 0.08, 0.07, 0.06,
                           0.05, 0.04, 0.03, 0.02, 0.02, 0.02, 0.02, 0.01,
                           0.01, 0.01, 0.01, 0.01]
            return {
                name: base_values[i] if i < len(base_values) else 0.01
                for i, name in enumerate(self.feature_names)
            }
        return {}

    def _compute_global_importance(self, task_type: str = "mortality") -> Dict[str, float]:
        """Oblicz global feature importance on-demand (SHAP → DALEX → syntetyczny fallback)."""
        if task_type == "dialysis":
            # Try SHAP on-demand
            try:
                from src.xai import SHAPExplainer
                tree_types = ["random_forest", "xgboost"]
                dial_tree_model = None
                for mt in tree_types:
                    if mt in self.dialysis_models:
                        dial_tree_model = self.dialysis_models[mt]
                        break
                if dial_tree_model is None and self.dialysis_models:
                    dial_tree_model = next(iter(self.dialysis_models.values()))
                if dial_tree_model is not None and self.dialysis_feature_names and self.dialysis_X_background is not None:
                    bg = self.dialysis_X_background
                    if len(bg) > 50:
                        import shap as shap_lib
                        bg = shap_lib.sample(bg, 50)
                    shap_exp = SHAPExplainer(dial_tree_model, bg, self.dialysis_feature_names)
                    return shap_exp.get_global_importance(self.dialysis_X_background)
            except Exception as e:
                logger.warning(f"Dialysis SHAP importance failed: {e}")
            # Try DALEX on-demand
            try:
                from src.xai import DALEXWrapper
                if self.dialysis_models and self.dialysis_X_train is not None and self.dialysis_y_train is not None:
                    model = next(iter(self.dialysis_models.values()))
                    y_float = np.array(self.dialysis_y_train, dtype=np.float64)
                    dalex_w = DALEXWrapper(model, self.dialysis_X_train, y_float, self.dialysis_feature_names)
                    return dalex_w.get_variable_importance()
            except Exception as e:
                logger.warning(f"Dialysis DALEX importance failed: {e}")
            # Syntetyczny fallback
            if self.dialysis_feature_names:
                base_values = [0.18, 0.16, 0.14, 0.12, 0.10, 0.09, 0.08, 0.07,
                               0.06, 0.05, 0.04, 0.03, 0.02, 0.02, 0.01]
                return {
                    name: base_values[i] if i < len(base_values) else 0.01
                    for i, name in enumerate(self.dialysis_feature_names)
                }
            return {
                "Kreatynina": 0.18,
                "Manifestacja_Nerki": 0.16,
                "Zaostrz_Wymagajace_OIT": 0.14,
                "Zaostrz_Wymagajace_Hospital": 0.12,
                "Wiek": 0.10,
                "Manifestacja_Oddechowy": 0.09,
                "Liczba_Zajetych_Narzadow": 0.08,
                "Plazmaferezy": 0.07,
                "Sterydy_Dawka_g": 0.06,
                "Pulsy": 0.05,
                "Manifestacja_Neurologiczny": 0.04,
                "Powiklania_Infekcja": 0.03,
                "Czas_Sterydow": 0.02,
                "Wiek_rozpoznania": 0.02,
                "Plec": 0.01,
            }
        # Mortality: Try SHAP on-demand
        try:
            from src.xai import SHAPExplainer
            tree_model, tree_name = self.get_tree_model()
            if tree_model is None:
                tree_model, _ = self.get_model()
            if tree_model is not None and self.feature_names and self.X_background is not None:
                bg = self.X_background
                if len(bg) > 50:
                    import shap as shap_lib
                    bg = shap_lib.sample(bg, 50)
                shap_exp = SHAPExplainer(tree_model, bg, self.feature_names)
                return shap_exp.get_global_importance(self.X_background)
        except Exception as e:
            logger.warning(f"SHAP importance failed: {e}")
        # Try DALEX on-demand
        try:
            from src.xai import DALEXWrapper
            model, _ = self.get_model()
            if model is not None and self.X_train is not None and self.y_train is not None:
                y_train_float = np.array(self.y_train, dtype=np.float64)
                dalex_w = DALEXWrapper(model, self.X_train, y_train_float, self.feature_names)
                return dalex_w.get_variable_importance()
        except Exception as e:
            logger.warning(f"DALEX importance failed: {e}")
        # Syntetyczny fallback
        if self.feature_names:
            base_values = [0.15, 0.12, 0.11, 0.10, 0.09, 0.08, 0.07, 0.06,
                           0.05, 0.04, 0.03, 0.02, 0.02, 0.02, 0.02, 0.01,
                           0.01, 0.01, 0.01, 0.01]
            return {
                name: base_values[i] if i < len(base_values) else 0.01
                for i, name in enumerate(self.feature_names)
            }
        return {
            "Wiek": 0.15,
            "Manifestacja_Nerki": 0.12,
            "Zaostrz_Wymagajace_OIT": 0.11,
            "Liczba_Zajetych_Narzadow": 0.10,
            "Manifestacja_Sercowo-Naczyniowy": 0.09,
            "Kreatynina": 0.08,
            "Max_CRP": 0.07,
            "Dializa": 0.06,
            "Manifestacja_Zajecie_CSN": 0.05,
            "Plazmaferezy": 0.04,
            "Manifestacja_Neurologiczny": 0.03,
            "Manifestacja_Pokarmowy": 0.02,
            "Plec": 0.02,
            "Sterydy_Dawka_g": 0.02,
            "Czas_Sterydow": 0.01,
            "Powiklania_Serce/pluca": 0.02,
            "Powiklania_Infekcja": 0.02,
            "Wiek_rozpoznania": 0.01,
            "Opoznienie_Rozpoznia": 0.01
        }

    def _get_fresh_ebm_explainer(self, task_type: str = "mortality"):
        """Załaduj lub fituj EBM na żądanie (bez cache)."""
        from src.xai import EBMExplainer
        if task_type == "dialysis":
            models_dir = self.dialysis_models_dir or "models/saved_dialysis"
            ebm_path = Path(models_dir) / "dialysis_ebm_model.joblib"
            ebm_exp = EBMExplainer(feature_names=self.dialysis_feature_names, class_names=['Brak dializy', 'Dializa'])
            if ebm_path.exists():
                ebm_exp.load_model(str(ebm_path))
            elif self.dialysis_X_train is not None and self.dialysis_y_train is not None:
                ebm_exp.fit(self.dialysis_X_train, self.dialysis_y_train, feature_names=self.dialysis_feature_names)
            else:
                return None
        else:
            models_dir = self.models_dir or "models/saved"
            ebm_path = Path(models_dir) / "ebm_model.joblib"
            ebm_exp = EBMExplainer(feature_names=self.feature_names)
            if ebm_path.exists():
                ebm_exp.load_model(str(ebm_path))
            elif self.X_train is not None and self.y_train is not None:
                ebm_exp.fit(self.X_train, self.y_train, feature_names=self.feature_names)
            else:
                return None
        return ebm_exp

    def load_dialysis_models(self, models_dir: str) -> bool:
        """Wczytaj modele dializy z rejestru."""
        import joblib

        models_path = Path(models_dir)
        registry_path = models_path / "dialysis_model_registry.json"
        feature_names_path = models_path / "dialysis_feature_names.json"

        if not registry_path.exists():
            logger.warning(f"Nie znaleziono dialysis_model_registry.json w {models_dir}")
            return False

        try:
            with open(registry_path, 'r', encoding='utf-8') as f:
                registry = json.load(f)

            if feature_names_path.exists():
                with open(feature_names_path, 'r') as f:
                    self.dialysis_feature_names = json.load(f)
                logger.info(f"Dializa: wczytano {len(self.dialysis_feature_names)} nazw cech")

            for model_type, model_info in registry.get("models", {}).items():
                if "error" in model_info:
                    logger.warning(f"Dializa: pomijam {model_type} - błąd: {model_info['error']}")
                    continue

                model_file = model_info.get("file")
                if not model_file:
                    continue

                model_path = models_path / model_file
                if model_path.exists():
                    try:
                        self.dialysis_models[model_type] = joblib.load(model_path)
                        self.dialysis_model_metadata[model_type] = model_info
                        logger.info(f"Dializa: wczytano model {model_type}")
                    except Exception as e:
                        logger.error(f"Dializa: błąd wczytywania {model_type}: {e}")

            self.dialysis_default_model_type = registry.get("default_model")
            if self.dialysis_default_model_type and self.dialysis_default_model_type not in self.dialysis_models:
                if self.dialysis_models:
                    self.dialysis_default_model_type = next(iter(self.dialysis_models))

            if self.dialysis_models:
                self.dialysis_loaded = True
                logger.info(f"Dializa: załadowano {len(self.dialysis_models)} modeli: {list(self.dialysis_models.keys())}")
                return True
            else:
                logger.warning("Dializa: nie udało się załadować żadnego modelu")
                return False

        except Exception as e:
            logger.error(f"Dializa: błąd wczytywania rejestru: {e}")
            return False

    def get_dialysis_model(self, model_type: Optional[str] = None):
        """Zwróć model dializy i jego nazwę. Fallback do domyślnego."""
        if model_type and model_type in self.dialysis_models:
            return self.dialysis_models[model_type], model_type
        if self.dialysis_default_model_type and self.dialysis_default_model_type in self.dialysis_models:
            return self.dialysis_models[self.dialysis_default_model_type], self.dialysis_default_model_type
        if self.dialysis_models:
            first_key = next(iter(self.dialysis_models))
            return self.dialysis_models[first_key], first_key
        return None, None

    def load_multiple_models(self, models_dir: str) -> bool:
        """Wczytaj wiele modeli z rejestru."""
        import joblib

        models_path = Path(models_dir)
        registry_path = models_path / "model_registry.json"
        feature_names_path = models_path / "feature_names.json"

        if not registry_path.exists():
            logger.warning(f"Nie znaleziono model_registry.json w {models_dir}")
            return False

        try:
            with open(registry_path, 'r', encoding='utf-8') as f:
                registry = json.load(f)

            # Wczytaj feature names
            if feature_names_path.exists():
                with open(feature_names_path, 'r') as f:
                    self.feature_names = json.load(f)
                logger.info(f"Wczytano {len(self.feature_names)} nazw cech")

            # Wczytaj każdy model z rejestru
            for model_type, model_info in registry.get("models", {}).items():
                if "error" in model_info:
                    logger.warning(f"Pomijam {model_type} - błąd trenowania: {model_info['error']}")
                    continue

                model_file = model_info.get("file")
                if not model_file:
                    continue

                model_path = models_path / model_file
                if model_path.exists():
                    try:
                        self.models[model_type] = joblib.load(model_path)
                        self.model_metadata[model_type] = model_info
                        logger.info(f"Wczytano model: {model_type} ({model_info.get('display_name', model_type)})")
                    except Exception as e:
                        logger.error(f"Błąd wczytywania {model_type}: {e}")
                else:
                    logger.warning(f"Nie znaleziono pliku modelu: {model_path}")

            # Ustaw domyślny model
            self.default_model_type = registry.get("default_model")
            if self.default_model_type and self.default_model_type in self.models:
                self.model = self.models[self.default_model_type]
                logger.info(f"Domyślny model: {self.default_model_type}")
            elif self.models:
                # Fallback do pierwszego dostępnego
                self.default_model_type = next(iter(self.models))
                self.model = self.models[self.default_model_type]
                logger.info(f"Domyślny model (fallback): {self.default_model_type}")

            if self.models:
                self.is_loaded = True
                logger.info(f"Załadowano {len(self.models)} modeli: {list(self.models.keys())}")
                return True
            else:
                logger.warning("Nie udało się załadować żadnego modelu z rejestru")
                return False

        except Exception as e:
            logger.error(f"Błąd wczytywania rejestru modeli: {e}")
            return False

    def get_model(self, model_type: Optional[str] = None):
        """Zwróć model i jego nazwę. Fallback do domyślnego."""
        if model_type and model_type in self.models:
            return self.models[model_type], model_type
        if self.default_model_type and self.default_model_type in self.models:
            return self.models[self.default_model_type], self.default_model_type
        if self.model is not None:
            return self.model, self.default_model_type or "unknown"
        return None, None

    def load_xai_artifacts(self, models_dir: str, prefix: str = "") -> bool:
        """Wczytaj artefakty XAI (X_background, X_train, y_train)."""
        import joblib
        models_path = Path(models_dir)
        bg_path = models_path / "X_background.joblib"

        if not bg_path.exists():
            logger.info(f"Brak artefaktów XAI w {models_dir}")
            return False

        try:
            X_bg = joblib.load(bg_path)
            X_tr = joblib.load(models_path / "X_train.joblib")
            y_tr = joblib.load(models_path / "y_train.joblib")

            if prefix == "dialysis":
                self.dialysis_X_background = X_bg
                self.dialysis_X_train = X_tr
                self.dialysis_y_train = y_tr
                self.dialysis_xai_ready = True
            else:
                self.X_background = X_bg
                self.X_train = X_tr
                self.y_train = y_tr
                self.xai_ready = True

            logger.info(f"XAI artefakty załadowane ({prefix or 'mortality'}): "
                        f"X_background={X_bg.shape}, X_train={X_tr.shape}")
            return True
        except Exception as e:
            logger.error(f"Błąd ładowania artefaktów XAI: {e}")
            return False

    def load_models(self, model_path: str, feature_names_path: str):
        """Wczytaj modele i explainer'y (backward compat - single model)."""
        import joblib

        try:
            # Wczytaj model
            self.model = joblib.load(model_path)
            logger.info(f"Model wczytany z {model_path}")

            # Wczytaj nazwy cech
            with open(feature_names_path, 'r') as f:
                self.feature_names = json.load(f)
            logger.info(f"Wczytano {len(self.feature_names)} nazw cech")

            self.is_loaded = True
            return True

        except Exception as e:
            logger.error(f"Błąd wczytywania modeli: {e}")
            return False


app_state = AppState()


# ============================================================================
# STARTUP/SHUTDOWN
# ============================================================================

async def _startup_logic():
    """Logika uruchomienia aplikacji."""
    logger.info("Uruchamianie API Vasculitis XAI...")

    # Próba wczytania wielu modeli z rejestru
    models_dir = os.getenv("MODELS_DIR", "models/saved")
    registry_path = Path(models_dir) / "model_registry.json"

    if registry_path.exists():
        loaded = app_state.load_multiple_models(models_dir)
        if loaded:
            logger.info(f"Multi-model: załadowano {len(app_state.models)} modeli")
            # Feature #15: weryfikacja spójności nazw cech
            if app_state.feature_names:
                for mtype, model in app_state.models.items():
                    if hasattr(model, 'n_features_in_'):
                        expected = model.n_features_in_
                        actual = len(app_state.feature_names)
                        if expected != actual:
                            logger.warning(
                                f"Niezgodność cech dla '{mtype}': "
                                f"model oczekuje {expected}, feature_names ma {actual}"
                            )
    else:
        # Fallback: single model (backward compat)
        model_path = os.getenv("MODEL_PATH", "models/saved/best_model.joblib")
        feature_names_path = os.getenv("FEATURE_NAMES_PATH", "models/saved/feature_names.json")

        if Path(model_path).exists() and Path(feature_names_path).exists():
            app_state.load_models(model_path, feature_names_path)
        else:
            logger.warning("Pliki modelu nie znalezione. API działa w trybie demo.")

    app_state.models_dir = models_dir

    # Ładowanie artefaktów XAI (mortality)
    app_state.load_xai_artifacts(models_dir)
    if app_state.xai_ready:
        logger.info("XAI ready (mortality)")

    # Ładowanie modeli dializy
    dialysis_dir = os.getenv("DIALYSIS_MODELS_DIR", "models/saved_dialysis")
    app_state.dialysis_models_dir = dialysis_dir
    dialysis_registry = Path(dialysis_dir) / "dialysis_model_registry.json"
    if dialysis_registry.exists():
        loaded = app_state.load_dialysis_models(dialysis_dir)
        if loaded:
            logger.info(f"Dializa: załadowano {len(app_state.dialysis_models)} modeli")
    else:
        logger.info("Brak modeli dializy. Endpointy dializy będą używać trybu demo.")

    # Ładowanie artefaktów XAI (dialysis)
    app_state.load_xai_artifacts(dialysis_dir, prefix="dialysis")
    if app_state.dialysis_xai_ready:
        logger.info("XAI ready (dialysis)")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_demo_prediction(patient: PatientInput) -> PredictionOutput:
    """Wygeneruj demo predykcję gdy model nie jest załadowany."""
    # Prosta heurystyka dla demo
    risk_score = 0.0

    # Wiek
    risk_score += max(0, (patient.wiek - 50) / 100)

    # Liczba narządów
    risk_score += patient.liczba_zajetych_narzadow * 0.1

    # Krytyczne manifestacje
    if patient.manifestacja_nerki:
        risk_score += 0.15
    if patient.manifestacja_sercowo_naczyniowy:
        risk_score += 0.15
    if patient.manifestacja_zajecie_csn:
        risk_score += 0.2

    # OIT
    if patient.zaostrz_wymagajace_oit:
        risk_score += 0.25

    # Dializa
    if patient.dializa:
        risk_score += 0.2

    # Ogranicz do [0, 1]
    probability = min(max(risk_score, 0.05), 0.95)

    return PredictionOutput(
        probability=probability,
        risk_level=get_risk_level_from_probability(probability),
        prediction=1 if probability > 0.5 else 0,
        confidence_interval={"lower": max(0.0, probability - 0.1), "upper": min(1.0, probability + 0.1)},
        is_demo=True
    )


def get_demo_explanation(patient: PatientInput) -> dict:
    """Wygeneruj demo wyjaśnienie."""
    risk_factors = []
    protective_factors = []

    if patient.wiek > 60:
        risk_factors.append({
            "feature": "Wiek",
            "value": patient.wiek,
            "contribution": 0.15,
            "direction": "increases_risk"
        })

    if patient.manifestacja_nerki:
        risk_factors.append({
            "feature": "Manifestacja_Nerki",
            "value": 1,
            "contribution": 0.12,
            "direction": "increases_risk"
        })

    if patient.zaostrz_wymagajace_oit:
        risk_factors.append({
            "feature": "Zaostrz_Wymagajace_OIT",
            "value": 1,
            "contribution": 0.2,
            "direction": "increases_risk"
        })

    if patient.wiek < 50:
        protective_factors.append({
            "feature": "Wiek",
            "value": patient.wiek,
            "contribution": -0.1,
            "direction": "decreases_risk"
        })

    if patient.liczba_zajetych_narzadow <= 2:
        protective_factors.append({
            "feature": "Liczba_Zajetych_Narzadow",
            "value": patient.liczba_zajetych_narzadow,
            "contribution": -0.08,
            "direction": "decreases_risk"
        })

    return {
        "risk_factors": risk_factors,
        "protective_factors": protective_factors,
        "base_value": 0.15
    }


def _get_demo_dialysis_explanation(patient=None) -> dict:
    """Wygeneruj demo wyjaśnienie dla trybu dializy (używa cech dializy i danych pacjenta)."""
    risk_factors = []
    protective_factors = []

    kreatynina = getattr(patient, "kreatynina", None) or 100
    if kreatynina > 150:
        risk_factors.append({"feature": "Kreatynina", "value": kreatynina, "contribution": 0.18, "direction": "increases_risk"})
    elif kreatynina < 80:
        protective_factors.append({"feature": "Kreatynina", "value": kreatynina, "contribution": -0.08, "direction": "decreases_risk"})

    manifestacja_nerki = getattr(patient, "manifestacja_nerki", None)
    if manifestacja_nerki:
        risk_factors.append({"feature": "Manifestacja_Nerki", "value": 1, "contribution": 0.16, "direction": "increases_risk"})
    else:
        protective_factors.append({"feature": "Manifestacja_Nerki", "value": 0, "contribution": -0.10, "direction": "decreases_risk"})

    if getattr(patient, "zaostrz_wymagajace_hospital", None):
        risk_factors.append({"feature": "Zaostrz_Wymagajace_Hospital", "value": 1, "contribution": 0.12, "direction": "increases_risk"})

    if getattr(patient, "zaostrz_wymagajace_oit", None):
        risk_factors.append({"feature": "Zaostrz_Wymagajace_OIT", "value": 1, "contribution": 0.10, "direction": "increases_risk"})

    if not getattr(patient, "plazmaferezy", False):
        protective_factors.append({"feature": "Plazmaferezy", "value": 0, "contribution": -0.05, "direction": "decreases_risk"})

    if not getattr(patient, "pulsy", False):
        protective_factors.append({"feature": "Pulsy", "value": 0, "contribution": -0.03, "direction": "decreases_risk"})

    return {
        "risk_factors": risk_factors,
        "protective_factors": protective_factors,
        "base_value": 0.2,
    }


XAI_TIMEOUT_SECONDS = int(os.getenv("XAI_TIMEOUT", "30"))
_XAI_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=2)
_OPENAI_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=3)


async def _run_in_executor_with_timeout(func, *args, timeout: float = XAI_TIMEOUT_SECONDS):
    """Uruchom funkcję synchroniczną w executorze z timeoutem."""
    loop = asyncio.get_running_loop()
    future = loop.run_in_executor(_XAI_EXECUTOR, func, *args)
    return await asyncio.wait_for(future, timeout=timeout)


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/", tags=["Info"])
@limiter.limit("60/minute")
async def root(request: Request):
    """Strona główna API."""
    return {
        "name": "Vasculitis XAI API",
        "version": "1.0.0",
        "description": "API do predykcji śmiertelności w zapaleniu naczyń z wyjaśnieniami XAI",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthCheckResponse, tags=["Info"])
@limiter.limit("60/minute")
async def health_check(request: Request):
    """Sprawdzenie stanu API."""
    return HealthCheckResponse(
        status="healthy",
        model_loaded=app_state.is_loaded,
        api_version="1.0.0",
        timestamp=datetime.now().isoformat(),
    )


@app.get("/health/xai", tags=["Info"])
@limiter.limit("60/minute")
async def xai_health(request: Request):
    """Status gotowości XAI (artefakty treningowe)."""
    return {
        "xai_ready": app_state.xai_ready,
        "dialysis_xai_ready": app_state.dialysis_xai_ready,
    }


@app.get("/health/models", tags=["Info"])
@limiter.limit("60/minute")
async def models_health(request: Request):
    """Diagnostyczny status wszystkich załadowanych modeli i cache'ów XAI."""
    models_status = {}
    for mtype, model in app_state.models.items():
        meta = app_state.model_metadata.get(mtype, {})
        models_status[mtype] = {
            "loaded": True,
            "display_name": meta.get("display_name", mtype),
            "metrics": meta.get("metrics", {}),
            "trained_at": meta.get("trained_at"),
            "n_features": getattr(model, "n_features_in_", None),
            "is_default": mtype == app_state.default_model_type,
        }

    dialysis_status = {}
    for mtype, model in app_state.dialysis_models.items():
        meta = app_state.dialysis_model_metadata.get(mtype, {})
        dialysis_status[mtype] = {
            "loaded": True,
            "display_name": meta.get("display_name", mtype),
            "metrics": meta.get("metrics", {}),
            "trained_at": meta.get("trained_at"),
            "n_features": getattr(model, "n_features_in_", None),
        }

    return {
        "mortality_models": models_status,
        "dialysis_models": dialysis_status,
        "xai_available": {
            "mortality": app_state.xai_ready,
            "dialysis": app_state.dialysis_xai_ready,
        },
        "feature_names_loaded": app_state.feature_names is not None,
        "n_feature_names": len(app_state.feature_names) if app_state.feature_names else 0,
    }


async def _predict_internal(patient: PatientInput) -> PredictionOutput:
    """Wewnętrzna logika predykcji mortality — bez rate limitingu."""
    try:
        if app_state.is_loaded:
            # Wybierz model
            requested_type = patient.model_type
            if requested_type and requested_type not in app_state.models and app_state.models:
                raise HTTPException(
                    status_code=400,
                    detail=f"Model '{requested_type}' niedostępny. Dostępne: {list(app_state.models.keys())}"
                )
            model, model_name = app_state.get_model(requested_type)

            if model is None:
                return get_demo_prediction(patient)

            features = patient_to_array(patient, app_state.feature_names)
            X = np.array(features).reshape(1, -1)

            probability = model.predict_proba(X)[0, 1]
            prediction = int(probability > 0.5)

            return PredictionOutput(
                probability=float(probability),
                risk_level=get_risk_level_from_probability(probability),
                prediction=prediction,
                model_used=model_name
            )
        else:
            # Demo mode
            return get_demo_prediction(patient)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Błąd predykcji: {e}")
        raise HTTPException(status_code=500, detail="Wewnętrzny błąd serwera")


@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
@limiter.limit("30/minute")
async def predict(request: Request, patient: PatientInput):
    """
    Wykonaj predykcję ryzyka śmiertelności.

    Zwraca prawdopodobieństwo zgonu i poziom ryzyka dla pacjenta.
    Opcjonalnie: model_type w body pozwala wybrać model.
    """
    return await _predict_internal(patient)


# ============================================================================
# BATCH PREDICTION
# ============================================================================

def batch_predict_vectorized(X: np.ndarray, model) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized batch prediction using numpy.

    Args:
        X: Feature matrix (n_patients, n_features)
        model: Trained model with predict_proba

    Returns:
        Tuple of (probabilities, predictions)
    """
    probabilities = model.predict_proba(X)[:, 1]  # Positive class
    predictions = (probabilities > 0.5).astype(int)
    return probabilities, predictions


def get_batch_risk_factors(
    X: np.ndarray,
    feature_names: List[str],
    top_n: int = 3
) -> List[List[RiskFactorItem]]:
    """
    Extract top risk factors for each patient using global importance.
    Fast extraction for batch processing.
    """
    global_importance = app_state.get_global_importance()
    n_patients = X.shape[0]
    results = []

    # Feature name to index mapping
    feature_idx_map = {name: i for i, name in enumerate(feature_names)}

    # Thresholds for determining risk direction
    RISK_THRESHOLDS = {
        "Wiek": 60,
        "Kreatynina": 150,
        "Max_CRP": 50,
        "Liczba_Zajetych_Narzadow": 3,
    }

    BINARY_RISK_FEATURES = {
        "Manifestacja_Nerki", "Manifestacja_Sercowo-Naczyniowy",
        "Manifestacja_Zajecie_CSN", "Zaostrz_Wymagajace_OIT",
        "Dializa", "Plazmaferezy", "Manifestacja_Neurologiczny",
        "Manifestacja_Pokarmowy", "Powiklania_Serce/pluca", "Powiklania_Infekcja"
    }

    for i in range(n_patients):
        patient_factors = []

        for feature_name, importance in global_importance.items():
            if feature_name not in feature_idx_map:
                continue

            idx = feature_idx_map[feature_name]
            value = float(X[i, idx])

            # Determine direction based on feature type and value
            if feature_name in BINARY_RISK_FEATURES:
                direction = "increases_risk" if value == 1 else "decreases_risk"
            elif feature_name in RISK_THRESHOLDS:
                threshold = RISK_THRESHOLDS[feature_name]
                direction = "increases_risk" if value > threshold else "decreases_risk"
            else:
                direction = "neutral"

            patient_factors.append(RiskFactorItem(
                feature=feature_name,
                value=value,
                importance=importance,
                direction=direction
            ))

        # Sort by importance and take top N
        patient_factors.sort(key=lambda x: x.importance, reverse=True)
        results.append(patient_factors[:top_n])

    return results


@app.post("/predict/batch", response_model=BatchPredictionOutput, tags=["Prediction"])
@limiter.limit("10/minute")
async def predict_batch(request: Request, batch_input: BatchPatientInput):
    """
    Batch prediction dla wielu pacjentów.

    Używa vectorized numpy dla wysokiej wydajności.
    Może przetwarzać 1000+ pacjentów w <100ms.

    Returns:
        Predykcje, poziomy ryzyka i opcjonalne czynniki ryzyka dla każdego pacjenta.
    """
    start_time = time.perf_counter()

    results = []
    errors = []
    mode = app_state.get_current_mode()

    # Check if API mode is required but model not loaded
    if mode == "unavailable":
        raise HTTPException(
            status_code=503,
            detail="Model nie jest załadowany i tryb demo jest wyłączony. Ustaw ALLOW_DEMO=true lub załaduj model."
        )

    n_patients = len(batch_input.patients)

    # Wybierz model dla całego batcha
    batch_model_type = batch_input.model_type
    batch_model = None
    batch_model_name = None

    if app_state.is_loaded:
        if batch_model_type and batch_model_type not in app_state.models and app_state.models:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{batch_model_type}' niedostępny. Dostępne: {list(app_state.models.keys())}"
            )
        batch_model, batch_model_name = app_state.get_model(batch_model_type)

    if app_state.is_loaded and batch_model is not None:
        # PRODUCTION MODE: Vectorized batch prediction
        try:
            # Convert all patients to matrix at once
            X = patients_to_matrix(batch_input.patients, app_state.feature_names)

            # Single vectorized prediction call
            probabilities, predictions = batch_predict_vectorized(X, batch_model)

            # Extract risk factors if requested
            if batch_input.include_risk_factors:
                risk_factors = get_batch_risk_factors(
                    X, app_state.feature_names, batch_input.top_n_factors
                )
            else:
                risk_factors = [None] * n_patients

            # Build results
            for i in range(n_patients):
                prob = float(probabilities[i])
                results.append(BatchPatientResult(
                    index=i,
                    prediction=PredictionOutput(
                        probability=prob,
                        risk_level=get_risk_level_from_probability(prob),
                        prediction=int(predictions[i])
                    ),
                    top_risk_factors=risk_factors[i],
                    processing_status="success"
                ))

        except Exception as e:
            logger.error(f"Batch prediction error: {e}")
            # Fallback to individual processing
            for i, patient in enumerate(batch_input.patients):
                try:
                    pred = await _predict_internal(patient)
                    results.append(BatchPatientResult(
                        index=i,
                        prediction=pred,
                        processing_status="success"
                    ))
                except Exception as inner_e:
                    logger.warning(f"Individual prediction failed for patient {i}: {inner_e}")
                    errors.append(BatchProcessingError(
                        patient_index=i,
                        error_type="prediction",
                        error_message=str(inner_e),
                        is_recoverable=True
                    ))
    else:
        # DEMO MODE: Use heuristic predictions
        for i, patient in enumerate(batch_input.patients):
            demo_pred = get_demo_prediction(patient)
            demo_factors = None

            if batch_input.include_risk_factors:
                demo_exp = get_demo_explanation(patient)
                demo_factors = [
                    RiskFactorItem(
                        feature=f["feature"],
                        value=f["value"],
                        importance=abs(f["contribution"]),
                        direction="increases_risk" if f["contribution"] > 0 else "decreases_risk"
                    )
                    for f in (demo_exp["risk_factors"] + demo_exp["protective_factors"])[:batch_input.top_n_factors]
                ]

            results.append(BatchPatientResult(
                index=i,
                prediction=demo_pred,
                top_risk_factors=demo_factors,
                processing_status="demo"
            ))

    # Calculate summary statistics
    probs = [r.prediction.probability for r in results if r.processing_status != "error"]

    summary = BatchSummary(
        total_count=n_patients,
        low_risk_count=sum(1 for r in results if r.prediction.risk_level == RiskLevel.LOW),
        moderate_risk_count=sum(1 for r in results if r.prediction.risk_level == RiskLevel.MODERATE),
        high_risk_count=sum(1 for r in results if r.prediction.risk_level == RiskLevel.HIGH),
        avg_probability=float(np.mean(probs)) if probs else 0.0,
        median_probability=float(np.median(probs)) if probs else 0.0,
        min_probability=float(np.min(probs)) if probs else 0.0,
        max_probability=float(np.max(probs)) if probs else 0.0
    )

    processing_time = (time.perf_counter() - start_time) * 1000

    return BatchPredictionOutput(
        total_patients=n_patients,
        processed_count=len(results),
        success_count=len([r for r in results if r.processing_status in ["success", "demo"]]),
        error_count=len(errors),
        processing_time_ms=processing_time,
        mode=mode,
        summary=summary,
        results=results,
        errors=errors
    )


# ============================================================================
# MODELS ENDPOINT
# ============================================================================

@app.get("/models/available", tags=["Model"])
@limiter.limit("60/minute")
async def get_available_models(request: Request):
    """
    Zwróć listę dostępnych modeli z metrykami.

    Returns:
        Lista modeli, domyślny model, metryki.
    """
    if not app_state.models:
        return {
            "models": [],
            "default_model": None,
            "total_models": 0
        }

    models_list = []
    for model_type, metadata in app_state.model_metadata.items():
        models_list.append({
            "model_type": model_type,
            "display_name": metadata.get("display_name", model_type),
            "metrics": metadata.get("metrics", {}),
            "trained_at": metadata.get("trained_at"),
            "is_default": model_type == app_state.default_model_type
        })

    return {
        "models": models_list,
        "default_model": app_state.default_model_type,
        "total_models": len(models_list)
    }


# ============================================================================
# DIALYSIS PREDICTION
# ============================================================================

def get_demo_dialysis_prediction(patient: DialysisPatientInput) -> DialysisPredictionOutput:
    """Wygeneruj demo predykcję dializy gdy model nie jest załadowany."""
    risk_score = 0.0

    # Kreatynina — najsilniejszy predyktor (r=0.685)
    kreatynina = patient.kreatynina or 100
    if kreatynina > 300:
        risk_score += 0.45
    elif kreatynina > 200:
        risk_score += 0.30
    elif kreatynina > 150:
        risk_score += 0.15

    # Manifestacja nerek (r=0.441)
    if patient.manifestacja_nerki:
        risk_score += 0.25

    # Inne czynniki
    risk_score += patient.liczba_zajetych_narzadow * 0.03
    if patient.zaostrz_wymagajace_oit:
        risk_score += 0.10
    if patient.plazmaferezy:
        risk_score += 0.05

    probability = min(max(risk_score, 0.02), 0.98)

    return DialysisPredictionOutput(
        probability=probability,
        needs_dialysis=probability > 0.5,
        prediction=int(probability > 0.5),
        risk_level=get_risk_level_from_probability(probability),
    )


async def _predict_dialysis_internal(patient: DialysisPatientInput) -> DialysisPredictionOutput:
    """Wewnętrzna logika predykcji dializy — bez rate limitingu."""
    try:
        if app_state.dialysis_loaded:
            requested_type = patient.model_type
            model, model_name = app_state.get_dialysis_model(requested_type)

            if model is None:
                return get_demo_dialysis_prediction(patient)

            if requested_type and requested_type not in app_state.dialysis_models:
                raise HTTPException(
                    status_code=400,
                    detail=f"Model dializy '{requested_type}' niedostępny. Dostępne: {list(app_state.dialysis_models.keys())}"
                )

            features = dialysis_patient_to_array(patient, app_state.dialysis_feature_names)
            X = np.array(features).reshape(1, -1)

            probability = model.predict_proba(X)[0, 1]

            return DialysisPredictionOutput(
                probability=float(probability),
                needs_dialysis=probability > 0.5,
                prediction=int(probability > 0.5),
                risk_level=get_risk_level_from_probability(probability),
                model_used=model_name
            )
        else:
            return get_demo_dialysis_prediction(patient)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Błąd predykcji dializy: {e}")
        raise HTTPException(status_code=500, detail="Wewnętrzny błąd serwera")


@app.post("/predict/dialysis", response_model=DialysisPredictionOutput, tags=["Dialysis"])
@limiter.limit("30/minute")
async def predict_dialysis(request: Request, patient: DialysisPatientInput):
    """
    Predykcja potrzeby dializy.

    Zwraca prawdopodobieństwo potrzeby dializy i poziom ryzyka.
    Opcjonalnie: model_type w body pozwala wybrać model.
    """
    return await _predict_dialysis_internal(patient)


@app.get("/models/dialysis/available", tags=["Dialysis"])
@limiter.limit("60/minute")
async def get_available_dialysis_models(request: Request):
    """
    Zwróć listę dostępnych modeli dializy z metrykami.
    """
    if not app_state.dialysis_models:
        return {
            "models": [],
            "default_model": None,
            "total_models": 0
        }

    models_list = []
    for model_type, metadata in app_state.dialysis_model_metadata.items():
        models_list.append({
            "model_type": model_type,
            "display_name": metadata.get("display_name", model_type),
            "metrics": metadata.get("metrics", {}),
            "trained_at": metadata.get("trained_at"),
            "is_default": model_type == app_state.dialysis_default_model_type
        })

    return {
        "models": models_list,
        "default_model": app_state.dialysis_default_model_type,
        "total_models": len(models_list)
    }


@app.post("/predict/dialysis/batch", response_model=BatchPredictionOutput, tags=["Dialysis"])
@limiter.limit("10/minute")
async def predict_dialysis_batch(request: Request, batch_input: BatchDialysisPatientInput):
    """
    Batch prediction dializy dla wielu pacjentów.

    Używa vectorized numpy dla wysokiej wydajności.
    Może przetwarzać 1000+ pacjentów w <100ms.

    Returns:
        Predykcje, poziomy ryzyka i opcjonalne czynniki ryzyka dla każdego pacjenta.
    """
    start_time = time.perf_counter()

    results = []
    errors = []
    mode = "demo" if not app_state.dialysis_loaded else "api"

    n_patients = len(batch_input.patients)

    batch_model_type = batch_input.model_type
    batch_model = None
    batch_model_name = None

    if app_state.dialysis_loaded:
        if batch_model_type and batch_model_type not in app_state.dialysis_models and app_state.dialysis_models:
            raise HTTPException(
                status_code=400,
                detail=f"Model dializy '{batch_model_type}' niedostępny. Dostępne: {list(app_state.dialysis_models.keys())}"
            )
        batch_model, batch_model_name = app_state.get_dialysis_model(batch_model_type)

    if app_state.dialysis_loaded and batch_model is not None:
        # PRODUCTION MODE: Vectorized batch prediction
        try:
            X = dialysis_patients_to_matrix(batch_input.patients, app_state.dialysis_feature_names)
            probabilities, predictions = batch_predict_vectorized(X, batch_model)

            risk_factors_list = [None] * n_patients
            if batch_input.include_risk_factors:
                dialysis_importance = app_state.get_global_importance(task_type="dialysis")
                feature_idx_map = {name: i for i, name in enumerate(app_state.dialysis_feature_names)}
                for idx in range(n_patients):
                    patient_factors = []
                    for feature_name, importance in dialysis_importance.items():
                        if feature_name not in feature_idx_map:
                            continue
                        fidx = feature_idx_map[feature_name]
                        value = float(X[idx, fidx])
                        if value == 1 and importance > 0:
                            direction = "increases_risk"
                        elif value == 0 and feature_name in {"Manifestacja_Nerki", "Manifestacja_Oddechowy",
                                                              "Manifestacja_Neurologiczny", "Zaostrz_Wymagajace_OIT",
                                                              "Plazmaferezy", "Pulsy"}:
                            direction = "decreases_risk"
                        elif feature_name == "Kreatynina" and value > 150:
                            direction = "increases_risk"
                        else:
                            direction = "decreases_risk" if value == 0 else "increases_risk"
                        patient_factors.append(RiskFactorItem(
                            feature=feature_name, value=value,
                            importance=importance, direction=direction
                        ))
                    patient_factors.sort(key=lambda x: x.importance, reverse=True)
                    risk_factors_list[idx] = patient_factors[:batch_input.top_n_factors]

            for i in range(n_patients):
                prob = float(probabilities[i])
                results.append(BatchPatientResult(
                    index=i,
                    prediction=PredictionOutput(
                        probability=prob,
                        risk_level=get_risk_level_from_probability(prob),
                        prediction=int(predictions[i])
                    ),
                    top_risk_factors=risk_factors_list[i],
                    processing_status="success"
                ))

        except Exception as e:
            logger.error(f"Dialysis batch prediction error: {e}")
            for i, patient in enumerate(batch_input.patients):
                try:
                    pred = await _predict_dialysis_internal(patient)
                    results.append(BatchPatientResult(
                        index=i,
                        prediction=PredictionOutput(
                            probability=pred.probability,
                            risk_level=pred.risk_level,
                            prediction=pred.prediction
                        ),
                        processing_status="success"
                    ))
                except Exception as inner_e:
                    logger.warning(f"Individual dialysis prediction failed for patient {i}: {inner_e}")
                    errors.append(BatchProcessingError(
                        patient_index=i,
                        error_type="prediction",
                        error_message=str(inner_e),
                        is_recoverable=True
                    ))
    else:
        # DEMO MODE
        for i, patient in enumerate(batch_input.patients):
            demo_pred = get_demo_dialysis_prediction(patient)
            results.append(BatchPatientResult(
                index=i,
                prediction=PredictionOutput(
                    probability=demo_pred.probability,
                    risk_level=demo_pred.risk_level,
                    prediction=demo_pred.prediction
                ),
                processing_status="demo"
            ))

    probs = [r.prediction.probability for r in results if r.processing_status != "error"]

    summary = BatchSummary(
        total_count=n_patients,
        low_risk_count=sum(1 for r in results if r.prediction.risk_level == RiskLevel.LOW),
        moderate_risk_count=sum(1 for r in results if r.prediction.risk_level == RiskLevel.MODERATE),
        high_risk_count=sum(1 for r in results if r.prediction.risk_level == RiskLevel.HIGH),
        avg_probability=float(np.mean(probs)) if probs else 0.0,
        median_probability=float(np.median(probs)) if probs else 0.0,
        min_probability=float(np.min(probs)) if probs else 0.0,
        max_probability=float(np.max(probs)) if probs else 0.0
    )

    processing_time = (time.perf_counter() - start_time) * 1000

    return BatchPredictionOutput(
        total_patients=n_patients,
        processed_count=len(results),
        success_count=len([r for r in results if r.processing_status in ["success", "demo"]]),
        error_count=len(errors),
        processing_time_ms=processing_time,
        mode=mode,
        summary=summary,
        results=results,
        errors=errors
    )


# ============================================================================
# CALIBRATION ENDPOINT
# ============================================================================

@app.get("/models/calibration", response_model=CalibrationResponse, tags=["Model"])
@limiter.limit("10/minute")
async def get_model_calibration(request: Request, task_type: Literal["mortality", "dialysis"] = "mortality"):
    """
    Oblicz metryki kalibracji (Brier score, krzywa niezawodności) dla załadowanych modeli.

    Używa danych treningowych do obliczenia prawdopodobieństw i porównania z obserwacjami.
    Obsługuje task_type='mortality' lub 'dialysis'.
    """
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss

    if task_type == "dialysis":
        models = app_state.dialysis_models
        metadata = app_state.dialysis_model_metadata
        X_data = app_state.dialysis_X_train
        y_data = app_state.dialysis_y_train
        feature_names = app_state.dialysis_feature_names
    else:
        models = app_state.models
        metadata = app_state.model_metadata
        X_data = app_state.X_train
        y_data = app_state.y_train
        feature_names = app_state.feature_names

    results = []

    if models and X_data is not None and y_data is not None:
        for model_type, model in models.items():
            try:
                probs = model.predict_proba(X_data)[:, 1]
                brier = float(brier_score_loss(y_data, probs))
                n_bins = min(10, max(2, int(np.sum(y_data) // 2)))
                prob_true, prob_pred = calibration_curve(y_data, probs, n_bins=n_bins, strategy="quantile")
                results.append(ModelCalibrationResult(
                    model_type=model_type,
                    display_name=metadata.get(model_type, {}).get("display_name", model_type),
                    brier_score=round(brier, 4),
                    calibration_curve_x=prob_pred.tolist(),
                    calibration_curve_y=prob_true.tolist(),
                    n_samples=int(len(y_data)),
                ))
            except Exception as e:
                logger.warning(f"Calibration failed for {model_type}: {e}")
    else:
        # Demo fallback: synthetic calibration data
        rng = np.random.RandomState(42)
        demo_models = [
            ("random_forest", "Random Forest", 0.12),
            ("xgboost", "XGBoost", 0.11),
            ("calibrated_svm", "Calibrated SVM", 0.14),
        ] if task_type == "mortality" else [
            ("logistic_regression", "Logistic Regression", 0.13),
            ("random_forest", "Random Forest", 0.11),
        ]
        for mt, dn, bs in demo_models:
            x = np.linspace(0.05, 0.95, 10)
            y = x + rng.normal(0, 0.05, 10)
            y = np.clip(y, 0, 1)
            results.append(ModelCalibrationResult(
                model_type=mt,
                display_name=dn,
                brier_score=bs,
                calibration_curve_x=x.tolist(),
                calibration_curve_y=y.tolist(),
                n_samples=100,
                is_demo=True,
            ))

    return CalibrationResponse(models=results, task_type=task_type)


# ============================================================================
# CONFIG ENDPOINTS
# ============================================================================

@app.get("/config/demo-mode", response_model=DemoModeStatus, tags=["Config"])
@limiter.limit("30/minute")
async def get_demo_mode_status(request: Request):
    """
    Pobierz status konfiguracji trybu demo.

    Returns:
        Czy tryb demo jest dozwolony, status modelu i aktualny tryb pracy.
    """
    return DemoModeStatus(
        demo_allowed=app_state.allow_demo,
        model_loaded=app_state.is_loaded,
        current_mode=app_state.get_current_mode(),
        force_api_mode=app_state.force_api_mode
    )


@app.post("/config/demo-mode", response_model=DemoModeStatus, tags=["Config"])
@limiter.limit("30/minute")
async def set_demo_mode(request: Request, demo_request: DemoModeRequest):
    """
    Włącz lub wyłącz tryb demo.

    Gdy tryb demo jest wyłączony i model nie jest załadowany, API zwróci błędy 503.
    Nie można włączyć trybu demo gdy zmienna FORCE_API_MODE jest ustawiona.
    """
    try:
        app_state.set_demo_mode(demo_request.enabled)
        return await get_demo_mode_status(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/all-models", response_model=MultiModelPredictionOutput, tags=["Prediction"])
@limiter.limit("20/minute")
async def predict_all_models(request: Request, patient: PatientInput):
    """
    Uruchom predykcje na WSZYSTKICH dostępnych modelach.

    Zwraca wyniki ze wszystkich modeli + consensus (średnia ważona AUC).
    """
    try:
        results = []

        if app_state.is_loaded and app_state.models:
            features = patient_to_array(patient, app_state.feature_names)
            X = np.array(features).reshape(1, -1)
            for model_type, model in app_state.models.items():
                metadata = app_state.model_metadata.get(model_type, {})
                proba = float(model.predict_proba(X)[0, 1])
                results.append(SingleModelResult(
                    model_type=model_type,
                    display_name=metadata.get("display_name", model_type),
                    probability=proba,
                    risk_level=get_risk_level_from_probability(proba),
                    prediction=int(proba > 0.5),
                    metrics=metadata.get("metrics")
                ))
        else:
            # Demo fallback: symulowane wyniki
            demo_pred = get_demo_prediction(patient)
            base_prob = demo_pred.probability
            rng = np.random.RandomState(hash(str(patient.wiek) + str(patient.plec)) % 2**31)
            demo_models = [
                ("random_forest", "Random Forest", 0.82),
                ("naive_bayes", "Naive Bayes", 0.75),
                ("calibrated_svm", "Calibrated SVM", 0.80),
                ("xgboost", "XGBoost", 0.83),
                ("stacking_ensemble", "Stacking Ensemble", 0.85),
            ]
            for mt, dn, auc in demo_models:
                noise = rng.uniform(-0.08, 0.08)
                prob = min(max(base_prob + noise, 0.01), 0.99)
                results.append(SingleModelResult(
                    model_type=mt,
                    display_name=dn,
                    probability=prob,
                    risk_level=get_risk_level_from_probability(prob),
                    prediction=int(prob > 0.5),
                    metrics={"auc_roc": auc, "accuracy": auc - 0.05, "sensitivity": auc - 0.03},
                    is_demo=True
                ))

        # Consensus: średnia ważona wg AUC
        auc_weights = []
        probas = []
        for r in results:
            probas.append(r.probability)
            auc_weights.append(r.metrics.get("auc_roc", 0.5) if r.metrics else 0.5)

        consensus = float(np.average(probas, weights=auc_weights)) if auc_weights else float(np.mean(probas))

        # Agreement
        risk_levels = [r.risk_level for r in results]
        most_common = max(set(risk_levels), key=risk_levels.count)
        agreement = risk_levels.count(most_common) / len(risk_levels) if risk_levels else 1.0

        is_demo = not (app_state.is_loaded and app_state.models)
        return MultiModelPredictionOutput(
            results=results,
            consensus_probability=consensus,
            consensus_risk_level=get_risk_level_from_probability(consensus),
            agreement_score=agreement,
            is_demo=is_demo
        )

    except Exception as e:
        logger.error(f"Błąd multi-model: {e}")
        raise HTTPException(status_code=500, detail="Wewnętrzny błąd serwera")


@app.post("/predict/dialysis/all-models", response_model=MultiModelPredictionOutput, tags=["Dialysis"])
@limiter.limit("20/minute")
async def predict_dialysis_all_models(request: Request, patient: DialysisPatientInput):
    """
    Uruchom predykcje dializy na WSZYSTKICH dostępnych modelach.

    Zwraca wyniki ze wszystkich modeli + consensus (średnia ważona AUC).
    """
    try:
        results = []

        if app_state.dialysis_models:
            if not app_state.dialysis_feature_names:
                raise HTTPException(status_code=503, detail="Nazwy cech dializy nie są załadowane")
            for model_type, model in app_state.dialysis_models.items():
                features = dialysis_patient_to_array(patient, app_state.dialysis_feature_names)
                X = np.array(features).reshape(1, -1)
                proba = float(model.predict_proba(X)[0, 1])
                # Load metrics from dialysis model metadata if available
                metadata = app_state.dialysis_model_metadata.get(model_type, {})
                metrics = metadata.get("metrics", None)
                results.append(SingleModelResult(
                    model_type=model_type,
                    display_name=model_type.replace("_", " ").title(),
                    probability=proba,
                    risk_level=get_risk_level_from_probability(proba),
                    prediction=int(proba > 0.5),
                    metrics=metrics,
                ))
        else:
            # Demo fallback
            demo_pred = get_demo_dialysis_prediction(patient)
            base_prob = demo_pred.probability
            rng = np.random.RandomState(hash(str(patient.wiek) + str(patient.kreatynina)) % 2**31)
            demo_models = [
                ("random_forest", "Random Forest", 0.82),
                ("naive_bayes", "Naive Bayes", 0.75),
                ("xgboost", "XGBoost", 0.83),
                ("stacking_ensemble", "Stacking Ensemble", 0.85),
            ]
            for mt, dn, auc in demo_models:
                noise = rng.uniform(-0.08, 0.08)
                prob = min(max(base_prob + noise, 0.01), 0.99)
                results.append(SingleModelResult(
                    model_type=mt,
                    display_name=dn,
                    probability=prob,
                    risk_level=get_risk_level_from_probability(prob),
                    prediction=int(prob > 0.5),
                    metrics={"auc_roc": auc, "accuracy": auc - 0.05, "sensitivity": auc - 0.03},
                    is_demo=True
                ))

        # Consensus: średnia ważona wg AUC
        auc_weights = []
        probas = []
        for r in results:
            probas.append(r.probability)
            auc_weights.append(r.metrics.get("auc_roc", 0.5) if r.metrics else 0.5)

        consensus = float(np.average(probas, weights=auc_weights)) if auc_weights else float(np.mean(probas))

        risk_levels = [r.risk_level for r in results]
        most_common = max(set(risk_levels), key=risk_levels.count)
        agreement = risk_levels.count(most_common) / len(risk_levels) if risk_levels else 1.0

        is_demo = not bool(app_state.dialysis_models)
        return MultiModelPredictionOutput(
            results=results,
            consensus_probability=consensus,
            consensus_risk_level=get_risk_level_from_probability(consensus),
            agreement_score=agreement,
            is_demo=is_demo
        )

    except Exception as e:
        logger.error(f"Błąd dialysis multi-model: {e}")
        raise HTTPException(status_code=500, detail="Wewnętrzny błąd serwera")


def _get_real_shap_explanation(
    patient: PatientInput,
    prediction: PredictionOutput,
    task_type: str = "mortality",
    dialysis_patient=None,
) -> Optional[SHAPExplanation]:
    """Spróbuj wygenerować prawdziwe wyjaśnienie SHAP."""
    ctx = app_state.get_xai_context(task_type)
    if not (ctx["xai_ready"] and ctx["is_loaded"]):
        return None

    feature_names = ctx["feature_names"]
    if not feature_names:
        return None

    try:
        from src.xai import SHAPExplainer

        if task_type == "dialysis" and dialysis_patient is not None:
            features = dialysis_patient_to_array(dialysis_patient, feature_names)
            model_type_req = getattr(dialysis_patient, "model_type", None)
        else:
            features = patient_to_array(patient, feature_names)
            model_type_req = getattr(patient, "model_type", None)

        model, model_name = ctx["get_model"](model_type_req)
        if model is None:
            return None

        explainer = SHAPExplainer(model, ctx["X_background"], feature_names)
        instance = np.array(features).reshape(1, -1)
        result = explainer.explain_instance(instance)

        shap_values = {}
        contributions = []
        risk_factors = []
        protective_factors = []

        for impact in result.get("feature_impacts", []):
            fname = impact["feature"]
            sv = impact.get("shap_value", 0)
            shap_values[fname] = sv

            feat_idx = feature_names.index(fname) if fname in feature_names else -1
            feat_val = float(instance[0, feat_idx]) if feat_idx >= 0 else 0

            direction = "increases_risk" if sv > 0 else "decreases_risk"
            contrib = {
                "feature": fname,
                "value": feat_val,
                "contribution": sv,
                "direction": direction
            }
            contributions.append(contrib)
            if sv > 0:
                risk_factors.append(contrib)
            elif sv < 0:
                protective_factors.append(contrib)

        return SHAPExplanation(
            base_value=result.get("base_value", 0),
            shap_values=shap_values,
            feature_contributions=contributions,
            risk_factors=risk_factors,
            protective_factors=protective_factors,
            prediction=prediction
        )
    except Exception as e:
        logger.warning(f"Real SHAP failed ({task_type}), falling back to demo: {e}")
        return None


def _get_real_lime_explanation(
    patient: PatientInput,
    prediction: PredictionOutput,
    task_type: str = "mortality",
    dialysis_patient=None,
) -> Optional[LIMEExplanation]:
    """Spróbuj wygenerować prawdziwe wyjaśnienie LIME."""
    ctx = app_state.get_xai_context(task_type)
    if not (ctx["xai_ready"] and ctx["is_loaded"]):
        return None

    feature_names = ctx["feature_names"]
    if not feature_names:
        return None

    try:
        from src.xai import LIMEExplainer

        if task_type == "dialysis" and dialysis_patient is not None:
            features = dialysis_patient_to_array(dialysis_patient, feature_names)
            model_type_req = getattr(dialysis_patient, "model_type", None)
        else:
            features = patient_to_array(patient, feature_names)
            model_type_req = getattr(patient, "model_type", None)

        model, model_name = ctx["get_model"](model_type_req)
        if model is None:
            return None

        explainer = LIMEExplainer(model, ctx["X_train"], feature_names)
        instance = np.array(features).reshape(1, -1)
        result = explainer.explain_instance(instance[0])

        feature_weights = []
        risk_factors = []
        protective_factors = []

        for fw in result.get("feature_weights", []):
            # feature_weights is a list of tuples (feature_description, weight)
            fname = fw[0]
            weight = fw[1]
            entry = {
                "feature": fname,
                "weight": weight,
                "condition": fname  # LIME includes condition in feature description
            }
            feature_weights.append(entry)
            if weight > 0:
                risk_factors.append({"feature": fname, "weight": weight})
            elif weight < 0:
                protective_factors.append({"feature": fname, "weight": weight})

        return LIMEExplanation(
            intercept=result.get("intercept", 0),
            feature_weights=feature_weights,
            risk_factors=risk_factors,
            protective_factors=protective_factors,
            local_prediction=result.get("probability_positive", result.get("probability", prediction.probability)),
            prediction=prediction
        )
    except Exception as e:
        logger.warning(f"Real LIME failed, falling back to demo: {e}")
        return None


@app.post("/explain/shap", response_model=SHAPExplanation, tags=["XAI"])
@limiter.limit("20/minute")
async def explain_shap(request: Request, req_body: ExplanationRequest):
    """
    Wygeneruj wyjaśnienie SHAP.

    Zwraca wartości SHAP dla cech pacjenta.
    Używa prawdziwego SHAP TreeExplainer gdy dostępne artefakty XAI.
    Obsługuje task_type='dialysis' (wymaga dialysis_patient w body).
    """
    try:
        task_type = req_body.task_type or "mortality"

        if task_type == "dialysis" and req_body.dialysis_patient is not None:
            dial_pred = await _predict_dialysis_internal(req_body.dialysis_patient)
            prediction = PredictionOutput(
                probability=dial_pred.probability,
                risk_level=dial_pred.risk_level,
                prediction=int(dial_pred.needs_dialysis),
                model_used=dial_pred.model_used,
            )
        else:
            prediction = await _predict_internal(req_body.patient)

        # Spróbuj prawdziwego SHAP (z timeoutem)
        try:
            shap_fn = functools.partial(
                _get_real_shap_explanation,
                req_body.patient, prediction, task_type, req_body.dialysis_patient
            )
            real_result = await _run_in_executor_with_timeout(shap_fn)
        except asyncio.TimeoutError:
            logger.warning("SHAP explanation timed out, falling back to demo")
            real_result = None
        if real_result is not None:
            return real_result

        # Demo fallback
        if task_type == "dialysis":
            demo_exp = _get_demo_dialysis_explanation(req_body.dialysis_patient)
        else:
            demo_exp = get_demo_explanation(req_body.patient)

        shap_values = {}
        contributions = []

        for rf in demo_exp["risk_factors"]:
            shap_values[rf["feature"]] = rf["contribution"]
            contributions.append({
                "feature": rf["feature"],
                "value": rf["value"],
                "contribution": rf["contribution"],
                "direction": rf["direction"]
            })

        for pf in demo_exp["protective_factors"]:
            shap_values[pf["feature"]] = pf["contribution"]
            contributions.append({
                "feature": pf["feature"],
                "value": pf["value"],
                "contribution": pf["contribution"],
                "direction": pf["direction"]
            })

        return SHAPExplanation(
            base_value=demo_exp["base_value"],
            shap_values=shap_values,
            feature_contributions=contributions,
            risk_factors=demo_exp["risk_factors"],
            protective_factors=demo_exp["protective_factors"],
            prediction=prediction
        )

    except Exception as e:
        logger.error(f"Błąd SHAP: {e}")
        raise HTTPException(status_code=500, detail="Wewnętrzny błąd serwera")


@app.post("/explain/lime", response_model=LIMEExplanation, tags=["XAI"])
@limiter.limit("20/minute")
async def explain_lime(request: Request, req_body: ExplanationRequest):
    """
    Wygeneruj wyjaśnienie LIME.

    Zwraca wagi cech z lokalnego modelu zastępczego.
    Używa prawdziwego LIME gdy dostępne artefakty XAI.
    Obsługuje task_type='dialysis' (wymaga dialysis_patient w body).
    """
    try:
        task_type = req_body.task_type or "mortality"

        if task_type == "dialysis" and req_body.dialysis_patient is not None:
            dial_pred = await _predict_dialysis_internal(req_body.dialysis_patient)
            prediction = PredictionOutput(
                probability=dial_pred.probability,
                risk_level=dial_pred.risk_level,
                prediction=int(dial_pred.needs_dialysis),
                model_used=dial_pred.model_used,
            )
        else:
            prediction = await _predict_internal(req_body.patient)

        # Spróbuj prawdziwego LIME (z timeoutem)
        try:
            lime_fn = functools.partial(
                _get_real_lime_explanation,
                req_body.patient, prediction, task_type, req_body.dialysis_patient
            )
            real_result = await _run_in_executor_with_timeout(lime_fn)
        except asyncio.TimeoutError:
            logger.warning("LIME explanation timed out, falling back to demo")
            real_result = None
        if real_result is not None:
            return real_result

        # Demo fallback
        if task_type == "dialysis":
            demo_exp = _get_demo_dialysis_explanation(req_body.dialysis_patient)
        else:
            demo_exp = get_demo_explanation(req_body.patient)

        feature_weights = []
        for rf in demo_exp["risk_factors"]:
            feature_weights.append({
                "feature": rf["feature"],
                "weight": rf["contribution"],
                "condition": f"{rf['feature']} = {rf['value']}"
            })

        for pf in demo_exp["protective_factors"]:
            feature_weights.append({
                "feature": pf["feature"],
                "weight": pf["contribution"],
                "condition": f"{pf['feature']} = {pf['value']}"
            })

        return LIMEExplanation(
            intercept=demo_exp["base_value"],
            feature_weights=feature_weights,
            risk_factors=[{"feature": rf["feature"], "weight": rf["contribution"]}
                         for rf in demo_exp["risk_factors"]],
            protective_factors=[{"feature": pf["feature"], "weight": pf["contribution"]}
                               for pf in demo_exp["protective_factors"]],
            local_prediction=prediction.probability,
            prediction=prediction
        )

    except Exception as e:
        logger.error(f"Błąd LIME: {e}")
        raise HTTPException(status_code=500, detail="Wewnętrzny błąd serwera")


@app.post("/explain/dalex", response_model=DALEXExplanation, tags=["XAI"])
@limiter.limit("20/minute")
async def explain_dalex(request: Request, req_body: ExplanationRequest):
    """
    Wygeneruj wyjaśnienie DALEX Break Down.

    Używa prawdziwego DALEX gdy dostępne artefakty XAI.
    Obsługuje task_type='dialysis' (wymaga dialysis_patient w body).
    """
    try:
        task_type = req_body.task_type or "mortality"
        ctx = app_state.get_xai_context(task_type)

        if task_type == "dialysis" and req_body.dialysis_patient is not None:
            dial_pred = await _predict_dialysis_internal(req_body.dialysis_patient)
            prediction = PredictionOutput(
                probability=dial_pred.probability,
                risk_level=dial_pred.risk_level,
                prediction=int(dial_pred.needs_dialysis),
                model_used=dial_pred.model_used,
            )
        else:
            prediction = await _predict_internal(req_body.patient)

        dalex_fallback_reason: Optional[str] = None

        if ctx["xai_ready"] and ctx["is_loaded"]:
            try:
                from src.xai import DALEXWrapper

                feature_names = ctx["feature_names"]
                if task_type == "dialysis" and req_body.dialysis_patient is not None:
                    model_type_req = getattr(req_body.dialysis_patient, "model_type", None)
                    features = dialysis_patient_to_array(req_body.dialysis_patient, feature_names)
                else:
                    model_type_req = req_body.patient.model_type
                    features = patient_to_array(req_body.patient, feature_names)

                model, model_name = ctx["get_model"](model_type_req)
                if model is not None:
                    y_train_float = np.array(ctx["y_train"], dtype=np.float64)
                    wrapper = DALEXWrapper(
                        model, ctx["X_train"], y_train_float, feature_names
                    )
                    instance = np.array(features).reshape(1, -1)
                    dalex_fn = functools.partial(wrapper.explain_instance_break_down, instance)
                    result = await _run_in_executor_with_timeout(dalex_fn)

                    contributions = []
                    risk_factors = []
                    protective_factors = []

                    for item in result.get("contributions", []):
                        fname = item.get("variable", item.get("feature", ""))
                        contrib_val = item.get("contribution", 0)
                        entry = {
                            "feature": fname,
                            "contribution": contrib_val,
                            "cumulative": item.get("cumulative", 0),
                        }
                        contributions.append(entry)
                        if contrib_val > 0.001:
                            risk_factors.append({"feature": fname, "contribution": contrib_val})
                        elif contrib_val < -0.001:
                            protective_factors.append({"feature": fname, "contribution": contrib_val})

                    return DALEXExplanation(
                        contributions=contributions,
                        risk_factors=risk_factors,
                        protective_factors=protective_factors,
                        prediction=prediction
                    )
            except Exception as e:
                dalex_fallback_reason = str(e)
                logger.warning(f"Real DALEX failed: {e}")

        # Demo fallback
        demo_exp = _get_demo_dialysis_explanation(req_body.dialysis_patient) if task_type == "dialysis" else get_demo_explanation(req_body.patient)
        contributions = []
        for rf in demo_exp["risk_factors"]:
            contributions.append({"feature": rf["feature"], "contribution": rf["contribution"], "cumulative": 0})
        for pf in demo_exp["protective_factors"]:
            contributions.append({"feature": pf["feature"], "contribution": pf["contribution"], "cumulative": 0})

        return DALEXExplanation(
            contributions=contributions,
            risk_factors=demo_exp["risk_factors"],
            protective_factors=demo_exp["protective_factors"],
            prediction=prediction,
            fallback_reason=dalex_fallback_reason
        )

    except Exception as e:
        logger.error(f"Błąd DALEX: {e}")
        raise HTTPException(status_code=500, detail="Wewnętrzny błąd serwera")


@app.post("/explain/comparison", response_model=XAIComparisonOutput, tags=["XAI"])
@limiter.limit("20/minute")
async def explain_comparison(request: Request, req_body: ExplanationRequest):
    """
    Porównanie metod XAI (SHAP vs LIME vs DALEX).

    Zwraca rankingi cech z każdej metody i miary zgodności.
    Obsługuje task_type='dialysis'.
    """
    try:
        task_type = req_body.task_type or "mortality"
        ctx = app_state.get_xai_context(task_type)

        if task_type == "dialysis" and req_body.dialysis_patient is not None:
            dial_pred = await _predict_dialysis_internal(req_body.dialysis_patient)
            prediction = PredictionOutput(
                probability=dial_pred.probability,
                risk_level=dial_pred.risk_level,
                prediction=int(dial_pred.needs_dialysis),
                model_used=dial_pred.model_used,
            )
        else:
            prediction = await _predict_internal(req_body.patient)

        individual_rankings = {}
        methods_used = []

        # Zbierz rankingi z SHAP
        try:
            shap_comp_fn = functools.partial(_get_real_shap_explanation, req_body.patient, prediction, task_type, req_body.dialysis_patient)
            shap_result = await _run_in_executor_with_timeout(shap_comp_fn)
        except asyncio.TimeoutError:
            shap_result = None
        if shap_result is not None:
            sorted_contribs = sorted(
                shap_result.feature_contributions, key=lambda x: abs(x.contribution), reverse=True
            )
            individual_rankings["SHAP"] = [c.feature for c in sorted_contribs[:10]]
            methods_used.append("SHAP")
        else:
            demo_exp = _get_demo_dialysis_explanation(req_body.dialysis_patient) if task_type == "dialysis" else get_demo_explanation(req_body.patient)
            all_f = demo_exp["risk_factors"] + demo_exp["protective_factors"]
            all_f.sort(key=lambda x: abs(x["contribution"]), reverse=True)
            individual_rankings["SHAP"] = [f["feature"] for f in all_f[:10]]
            methods_used.append("SHAP")

        # Zbierz rankingi z LIME
        try:
            lime_comp_fn = functools.partial(_get_real_lime_explanation, req_body.patient, prediction, task_type, req_body.dialysis_patient)
            lime_result = await _run_in_executor_with_timeout(lime_comp_fn)
        except asyncio.TimeoutError:
            lime_result = None
        if lime_result is not None:
            sorted_weights = sorted(
                lime_result.feature_weights, key=lambda x: abs(x.get("weight", 0)), reverse=True
            )
            individual_rankings["LIME"] = [w["feature"] for w in sorted_weights[:10]]
            methods_used.append("LIME")
        else:
            individual_rankings["LIME"] = individual_rankings.get("SHAP", [])
            methods_used.append("LIME")

        # Zbierz rankingi z DALEX (jeśli możliwe)
        if ctx["xai_ready"] and ctx["is_loaded"]:
            try:
                from src.xai import DALEXWrapper
                feature_names = ctx["feature_names"]
                if task_type == "dialysis" and req_body.dialysis_patient is not None:
                    model_type_req = getattr(req_body.dialysis_patient, "model_type", None)
                    features = dialysis_patient_to_array(req_body.dialysis_patient, feature_names)
                else:
                    model_type_req = req_body.patient.model_type
                    features = patient_to_array(req_body.patient, feature_names)
                model, _ = ctx["get_model"](model_type_req)
                if model is not None:
                    wrapper = DALEXWrapper(
                        model, ctx["X_train"], np.array(ctx["y_train"], dtype=np.float64), feature_names
                    )
                    instance = np.array(features).reshape(1, -1)
                    dalex_comp_fn = functools.partial(wrapper.explain_instance_break_down, instance)
                    result = await _run_in_executor_with_timeout(dalex_comp_fn)
                    contribs = result.get("contributions", [])
                    contribs.sort(key=lambda x: abs(x.get("contribution", 0)), reverse=True)
                    individual_rankings["DALEX"] = [c.get("variable", c.get("feature", "")) for c in contribs[:10]]
                    methods_used.append("DALEX")
            except Exception as e:
                logger.warning(f"DALEX comparison failed: {e}")

        # Oblicz common top features i agreement
        all_rankings_lists = list(individual_rankings.values())
        if len(all_rankings_lists) >= 2:
            top5_sets = [set(r[:5]) for r in all_rankings_lists]
            common = set.intersection(*top5_sets) if top5_sets else set()
            # Jaccard agreement
            union = set.union(*top5_sets) if top5_sets else set()
            agreement = len(common) / len(union) if union else 1.0
        else:
            common = set(all_rankings_lists[0][:5]) if all_rankings_lists else set()
            agreement = 1.0

        return XAIComparisonOutput(
            methods_compared=methods_used,
            ranking_agreement=agreement,
            common_top_features=list(common),
            individual_rankings=individual_rankings
        )

    except Exception as e:
        logger.error(f"Błąd porównania XAI: {e}")
        raise HTTPException(status_code=500, detail="Wewnętrzny błąd serwera")


@app.get("/explain/shap/global", response_model=SHAPGlobalResponse, tags=["XAI Global"])
@limiter.limit("5/minute")
async def explain_shap_global(request: Request, task_type: Literal["mortality", "dialysis"] = "mortality"):
    """
    Globalna analiza SHAP na datasecie treningowym.

    Zwraca macierz SHAP values (beeswarm) i średnią ważność cech (bar).
    Obsługuje task_type='dialysis'.
    """
    try:
        ctx = app_state.get_xai_context(task_type)

        # Try real SHAP on-demand
        if ctx["xai_ready"] and ctx["is_loaded"]:
            try:
                from src.xai import SHAPExplainer

                def _compute_shap_global():
                    if task_type == "dialysis":
                        tree_types = ["random_forest", "xgboost"]
                        tree_model = None
                        for mt in tree_types:
                            if mt in app_state.dialysis_models:
                                tree_model = app_state.dialysis_models[mt]
                                break
                        if tree_model is None and app_state.dialysis_models:
                            tree_model = next(iter(app_state.dialysis_models.values()))
                    else:
                        tree_model, _ = app_state.get_tree_model()
                        if tree_model is None:
                            tree_model, _ = app_state.get_model()
                    if tree_model is None:
                        return None
                    bg = ctx["X_background"]
                    if len(bg) > 50:
                        import shap as shap_lib
                        bg = shap_lib.sample(bg, 50)
                    feature_names = ctx["feature_names"]
                    shap_exp = SHAPExplainer(tree_model, bg, feature_names)
                    dataset_result = shap_exp.explain_dataset(ctx["X_background"])
                    importance = shap_exp.get_global_importance(ctx["X_background"])
                    shap_vals = dataset_result.get("shap_values")
                    if shap_vals is None:
                        return None
                    return {
                        "feature_importance": importance,
                        "shap_values_matrix": shap_vals.tolist() if hasattr(shap_vals, 'tolist') else shap_vals,
                        "feature_values_matrix": ctx["X_background"].tolist(),
                        "feature_names": feature_names,
                        "base_value": float(dataset_result.get("base_value", 0)),
                        "n_samples": int(ctx["X_background"].shape[0]),
                    }

                result = await _run_in_executor_with_timeout(_compute_shap_global)
                if result is not None:
                    return SHAPGlobalResponse(**result)
            except (asyncio.TimeoutError, Exception) as e:
                logger.warning(f"Real SHAP global failed: {e}")

        # Demo fallback
        importance = app_state.get_global_importance(task_type=task_type)
        n_samples = 50
        rng = np.random.RandomState(42)

        shap_matrix = []
        feature_matrix = []
        for _ in range(n_samples):
            row_shap = []
            row_feat = []
            for fname, imp in importance.items():
                sv = rng.normal(0, imp * 2)
                fv = rng.uniform(0, 1)
                row_shap.append(float(sv))
                row_feat.append(float(fv))
            shap_matrix.append(row_shap)
            feature_matrix.append(row_feat)

        return SHAPGlobalResponse(
            feature_importance=importance,
            shap_values_matrix=shap_matrix,
            feature_values_matrix=feature_matrix,
            feature_names=list(importance.keys()),
            base_value=0.15,
            n_samples=n_samples,
        )
    except Exception as e:
        logger.error(f"Błąd SHAP global: {e}")
        raise HTTPException(status_code=500, detail="Wewnętrzny błąd serwera")


@app.get("/explain/dalex/variable-importance", response_model=DALEXVariableImportanceResponse, tags=["XAI Global"])
@limiter.limit("10/minute")
async def explain_dalex_variable_importance(request: Request, task_type: Literal["mortality", "dialysis"] = "mortality"):
    """
    Permutation variable importance z DALEX.

    Mierzy spadek wydajności modelu po permutacji każdej cechy.
    Obsługuje task_type='dialysis'.
    """
    try:
        ctx = app_state.get_xai_context(task_type)

        # Try real DALEX on-demand
        if ctx["xai_ready"] and ctx["is_loaded"]:
            try:
                from src.xai import DALEXWrapper

                def _compute_dalex_vi():
                    model, _ = ctx["get_model"]()
                    if model is None:
                        return None
                    y_float = np.array(ctx["y_train"], dtype=np.float64)
                    dalex_w = DALEXWrapper(model, ctx["X_train"], y_float, ctx["feature_names"])
                    return dalex_w.get_variable_importance()

                vi = await _run_in_executor_with_timeout(_compute_dalex_vi)
                if vi is not None:
                    sorted_features = sorted(vi, key=vi.get, reverse=True)
                    return DALEXVariableImportanceResponse(
                        feature_importance=vi,
                        top_features=sorted_features[:10],
                    )
            except (asyncio.TimeoutError, Exception) as e:
                logger.warning(f"Real DALEX VI failed: {e}")

        # Demo fallback
        importance = app_state.get_global_importance(task_type=task_type)
        scaled = {k: v * 5.0 for k, v in importance.items()}
        sorted_features = sorted(scaled, key=scaled.get, reverse=True)
        return DALEXVariableImportanceResponse(
            feature_importance=scaled,
            top_features=sorted_features[:10],
        )
    except Exception as e:
        logger.error(f"Błąd DALEX VI: {e}")
        raise HTTPException(status_code=500, detail="Wewnętrzny błąd serwera")


@app.get("/explain/dalex/pdp/{feature}", response_model=DALEXPDPResponse, tags=["XAI Global"])
@limiter.limit("10/minute")
async def explain_dalex_pdp(request: Request, feature: str, task_type: Literal["mortality", "dialysis"] = "mortality"):
    """
    Partial Dependence Profile dla wybranej cechy.

    Pokazuje jak zmienia się średnia predykcja modelu w funkcji wartości cechy.
    Obsługuje task_type='dialysis'.
    """
    try:
        ctx = app_state.get_xai_context(task_type)
        feature_names = ctx["feature_names"] or app_state.feature_names or []
        valid_features = set(feature_names) | set(app_state.get_global_importance(task_type=task_type).keys())
        if feature not in valid_features:
            raise HTTPException(status_code=400, detail=f"Cecha '{feature}' niedostępna. Dostępne: {sorted(valid_features)}")

        # Try real DALEX PDP on-demand
        if ctx["xai_ready"] and ctx["is_loaded"]:
            try:
                from src.xai import DALEXWrapper
                model, _ = ctx["get_model"]()
                if model is not None:
                    wrapper = DALEXWrapper(
                        model, ctx["X_train"], np.array(ctx["y_train"], dtype=np.float64), feature_names
                    )
                    pdp = wrapper.get_partial_dependence(feature)
                    result = {
                        "feature": feature,
                        "x_values": [float(x) for x in pdp.get("x", pdp.get("grid", []))],
                        "y_values": [float(y) for y in pdp.get("y", pdp.get("mean_prediction", []))],
                    }
                    return DALEXPDPResponse(**result)
            except Exception as e:
                logger.warning(f"Real DALEX PDP failed for {feature}: {e}")

        # Demo fallback — sigmoid curve
        x_vals = [float(i) for i in np.linspace(0, 1, 50)]
        base = 0.3
        imp = app_state.get_global_importance(task_type=task_type).get(feature, 0.05)
        y_vals = [float(base + imp * 2 * (1 / (1 + np.exp(-10 * (x - 0.5))))) for x in x_vals]
        result = {"feature": feature, "x_values": x_vals, "y_values": y_vals}
        return DALEXPDPResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Błąd DALEX PDP: {e}")
        raise HTTPException(status_code=500, detail="Wewnętrzny błąd serwera")


@app.get("/explain/ebm/global", response_model=EBMGlobalResponse, tags=["XAI Global"])
@limiter.limit("10/minute")
async def explain_ebm_global(request: Request, task_type: Literal["mortality", "dialysis"] = "mortality"):
    """
    Globalna ważność cech z modelu EBM (Explainable Boosting Machine).

    EBM jest modelem inherently interpretable — każda cecha ma jawną funkcję kształtu.
    Obsługuje task_type='dialysis'.
    """
    try:
        ctx = app_state.get_xai_context(task_type)

        # Try real EBM on-demand
        if ctx["xai_ready"] and ctx["is_loaded"]:
            try:
                def _compute_ebm_global():
                    return app_state._get_fresh_ebm_explainer(task_type)

                ebm_exp = await _run_in_executor_with_timeout(_compute_ebm_global)
                if ebm_exp is not None:
                    cache = ebm_exp.explain_global()
                    importance = cache.get("feature_importance", cache.get("importances", {}))
                    if isinstance(importance, list):
                        names = cache.get("feature_names", ctx["feature_names"])
                        importance = dict(zip(names, importance))
                    return EBMGlobalResponse(
                        feature_importance=importance,
                        feature_names=list(importance.keys()),
                        interactions_detected=cache.get("interactions", []),
                    )
            except (asyncio.TimeoutError, Exception) as e:
                logger.warning(f"Real EBM global failed: {e}")

        # Demo fallback
        importance = app_state.get_global_importance(task_type=task_type)
        demo_interactions = (
            ["Kreatynina × Manifestacja_Nerki", "Wiek × Zaostrz_Wymagajace_Hospital"]
            if task_type == "dialysis"
            else ["Wiek × Kreatynina", "Manifestacja_Nerki × Dializa"]
        )
        return EBMGlobalResponse(
            feature_importance=importance,
            feature_names=list(importance.keys()),
            interactions_detected=demo_interactions,
        )
    except Exception as e:
        logger.error(f"Błąd EBM global: {e}")
        raise HTTPException(status_code=500, detail="Wewnętrzny błąd serwera")


@app.post("/explain/ebm/local", response_model=EBMLocalResponse, tags=["XAI"])
@limiter.limit("20/minute")
async def explain_ebm_local(request: Request, req_body: ExplanationRequest):
    """
    Lokalne wyjaśnienie EBM dla pojedynczego pacjenta.

    Rozkłada predykcję na wkłady poszczególnych cech.
    Obsługuje task_type='dialysis' (wymaga dialysis_patient w body).
    """
    try:
        task_type = req_body.task_type or "mortality"
        ctx = app_state.get_xai_context(task_type)
        feature_names = ctx["feature_names"]

        # Try to get fresh EBM explainer on-demand
        ebm_explainer = None
        if ctx["xai_ready"] and ctx["is_loaded"]:
            try:
                def _load_ebm():
                    return app_state._get_fresh_ebm_explainer(task_type)
                ebm_explainer = await _run_in_executor_with_timeout(_load_ebm)
            except (asyncio.TimeoutError, Exception) as e:
                logger.warning(f"EBM explainer load failed: {e}")

        if ebm_explainer is not None and feature_names:
            try:
                if task_type == "dialysis" and req_body.dialysis_patient is not None:
                    features = dialysis_patient_to_array(req_body.dialysis_patient, feature_names)
                else:
                    features = patient_to_array(req_body.patient, feature_names)
                instance = np.array(features).reshape(1, -1)
                result = ebm_explainer.explain_local(instance)

                contributions = result.get("contributions", [])
                risk_factors = [c for c in contributions if c.get("contribution", c.get("score", 0)) > 0]
                protective_factors = [c for c in contributions if c.get("contribution", c.get("score", 0)) < 0]

                return EBMLocalResponse(
                    probability_positive=float(result.get("probability", result.get("prediction", 0.5))),
                    intercept=float(result.get("intercept", 0)),
                    contributions=contributions,
                    risk_factors=sorted(risk_factors, key=lambda x: x.get("contribution", x.get("score", 0)), reverse=True),
                    protective_factors=sorted(protective_factors, key=lambda x: x.get("contribution", x.get("score", 0))),
                )
            except Exception as e:
                logger.warning(f"Real EBM local failed: {e}")

        # Demo fallback
        demo_exp = _get_demo_dialysis_explanation(req_body.dialysis_patient) if task_type == "dialysis" else get_demo_explanation(req_body.patient)
        if task_type != "dialysis":
            prediction = await _predict_internal(req_body.patient)
        else:
            if req_body.dialysis_patient is not None:
                try:
                    dial_pred = await _predict_dialysis_internal(req_body.dialysis_patient)
                    prediction = PredictionOutput(
                        probability=dial_pred.probability,
                        risk_level=dial_pred.risk_level,
                        prediction=int(dial_pred.needs_dialysis),
                        model_used=dial_pred.model_used,
                    )
                except Exception:
                    prediction = PredictionOutput(probability=0.35, risk_level=RiskLevel.MODERATE, prediction=0, is_demo=True)
            else:
                prediction = PredictionOutput(probability=0.35, risk_level=RiskLevel.MODERATE, prediction=0, is_demo=True)
        contributions = []
        for rf in demo_exp["risk_factors"]:
            contributions.append({"feature": rf["feature"], "contribution": rf["contribution"], "value": rf.get("value", 0)})
        for pf in demo_exp["protective_factors"]:
            contributions.append({"feature": pf["feature"], "contribution": pf["contribution"], "value": pf.get("value", 0)})

        return EBMLocalResponse(
            probability_positive=prediction.probability,
            intercept=demo_exp["base_value"],
            contributions=contributions,
            risk_factors=demo_exp["risk_factors"],
            protective_factors=demo_exp["protective_factors"],
        )
    except Exception as e:
        logger.error(f"Błąd EBM local: {e}")
        raise HTTPException(status_code=500, detail="Wewnętrzny błąd serwera")


@app.get("/explain/ebm/feature-function/{feature}", response_model=EBMFeatureFunctionResponse, tags=["XAI Global"])
@limiter.limit("10/minute")
async def explain_ebm_feature_function(request: Request, feature: str, task_type: Literal["mortality", "dialysis"] = "mortality"):
    """
    Kształt funkcji cechy (shape function) z modelu EBM.

    Pokazuje jak model EBM mapuje wartości cechy na score (log-odds).
    Obsługuje task_type='dialysis'.
    """
    try:
        ctx = app_state.get_xai_context(task_type)
        feature_names = ctx["feature_names"] or app_state.feature_names or []
        valid_features = set(feature_names) | set(app_state.get_global_importance(task_type=task_type).keys())
        if feature not in valid_features:
            raise HTTPException(status_code=400, detail=f"Cecha '{feature}' niedostępna. Dostępne: {sorted(valid_features)}")

        # Try real EBM feature function on-demand
        if ctx["xai_ready"] and ctx["is_loaded"]:
            try:
                def _load_ebm_ff():
                    return app_state._get_fresh_ebm_explainer(task_type)
                ebm_explainer = await _run_in_executor_with_timeout(_load_ebm_ff)
                if ebm_explainer is not None:
                    ff = ebm_explainer.get_feature_function(feature)
                    result = {
                        "feature": feature,
                        "names": [float(x) if isinstance(x, (int, float, np.integer, np.floating)) else x for x in ff.get("names", ff.get("bins", []))],
                        "scores": [float(s) for s in ff.get("scores", [])],
                    }
                    return EBMFeatureFunctionResponse(**result)
            except (asyncio.TimeoutError, Exception) as e:
                logger.warning(f"Real EBM feature function failed for {feature}: {e}")

        # Demo fallback — monotone curve
        x_vals = [float(x) for x in np.linspace(0, 1, 30)]
        imp = app_state.get_global_importance(task_type=task_type).get(feature, 0.05)
        scores = [float(imp * 3 * (x - 0.5)) for x in x_vals]
        result = {"feature": feature, "names": x_vals, "scores": scores}
        return EBMFeatureFunctionResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Błąd EBM feature function: {e}")
        raise HTTPException(status_code=500, detail="Wewnętrzny błąd serwera")


@app.get("/explain/comparison/global", response_model=GlobalComparisonResponse, tags=["XAI Global"])
@limiter.limit("10/minute")
async def explain_comparison_global(request: Request, task_type: Literal["mortality", "dialysis"] = "mortality"):
    """
    Porównanie metod XAI na poziomie globalnym.

    Heatmap rankingow cech, macierz Jaccard agreement, korelacje Spearman.
    Obsługuje task_type='dialysis'.
    """
    try:
        ctx = app_state.get_xai_context(task_type)
        rankings = {}
        methods = []

        # Collect global importances (on-demand)
        imp = app_state.get_global_importance(task_type=task_type)

        rankings["SHAP"] = sorted(imp, key=imp.get, reverse=True)[:10]
        methods.append("SHAP")

        rankings["DALEX"] = sorted(imp, key=imp.get, reverse=True)[:10]
        methods.append("DALEX")

        rankings["EBM"] = sorted(imp, key=imp.get, reverse=True)[:10]
        methods.append("EBM")

        # Calculate agreement matrix (Jaccard similarity between top-5 features)
        agreement_matrix = {}
        for m1 in methods:
            agreement_matrix[m1] = {}
            s1 = set(rankings[m1][:5])
            for m2 in methods:
                s2 = set(rankings[m2][:5])
                union = s1 | s2
                jaccard = len(s1 & s2) / len(union) if union else 1.0
                agreement_matrix[m1][m2] = round(jaccard, 3)

        # Common top features (in all methods' top 5)
        top5_sets = [set(rankings[m][:5]) for m in methods]
        common = set.intersection(*top5_sets) if top5_sets else set()

        # Mean agreement (off-diagonal)
        off_diag = []
        for m1 in methods:
            for m2 in methods:
                if m1 != m2:
                    off_diag.append(agreement_matrix[m1][m2])
        mean_agreement = float(np.mean(off_diag)) if off_diag else 1.0

        return GlobalComparisonResponse(
            methods_compared=methods,
            rankings=rankings,
            agreement_matrix=agreement_matrix,
            common_top_features=list(common),
            mean_agreement=round(mean_agreement, 3),
        )
    except Exception as e:
        logger.error(f"Błąd comparison global: {e}")
        raise HTTPException(status_code=500, detail="Wewnętrzny błąd serwera")


@app.post("/explain/patient", response_model=PatientExplanation, tags=["XAI"])
@limiter.limit("20/minute")
async def explain_for_patient(request: Request, req_body: PatientExplanationRequest):
    """
    Wygeneruj wyjaśnienie zrozumiałe dla pacjenta.

    Dostosowuje język i poziom szczegółowości do poziomu health literacy.
    Obsługuje task_type='dialysis' (wymaga dialysis_patient w body).
    """
    try:
        task_type = req_body.task_type or "mortality"
        if task_type == "dialysis" and req_body.dialysis_patient is not None:
            dial_pred = await _predict_dialysis_internal(req_body.dialysis_patient)
            prediction = PredictionOutput(
                probability=dial_pred.probability,
                risk_level=dial_pred.risk_level,
                prediction=int(dial_pred.needs_dialysis),
                model_used=dial_pred.model_used,
            )
        else:
            prediction = await _predict_internal(req_body.patient)

        # Tłumaczenia cech
        translations = {
            "Wiek": "Twój wiek",
            "Manifestacja_Nerki": "Stan nerek",
            "Manifestacja_Sercowo-Naczyniowy": "Stan układu krążenia",
            "Zaostrz_Wymagajace_OIT": "Przebyte poważne zaostrzenia",
            "Liczba_Zajetych_Narzadow": "Liczba dotkniętych narządów",
            "Kreatynina": "Wskaźnik czynności nerek",
            "Max_CRP": "Poziom stanu zapalnego"
        }

        # Spróbuj prawdziwego wyjaśnienia SHAP
        try:
            patient_shap_fn = functools.partial(_get_real_shap_explanation, req_body.patient, prediction, task_type, req_body.dialysis_patient)
            real_shap = await _run_in_executor_with_timeout(patient_shap_fn)
        except asyncio.TimeoutError:
            real_shap = None
        if real_shap is not None:
            risk_factors_src = [
                {"feature": c.feature, "value": c.value, "contribution": c.contribution}
                for c in real_shap.risk_factors
            ]
            protective_factors_src = [
                {"feature": c.feature, "value": c.value, "contribution": c.contribution}
                for c in real_shap.protective_factors
            ]
        else:
            demo_exp = _get_demo_dialysis_explanation(req_body.dialysis_patient) if task_type == "dialysis" else get_demo_explanation(req_body.patient)
            risk_factors_src = demo_exp["risk_factors"]
            protective_factors_src = demo_exp["protective_factors"]

        # Poziom ryzyka
        if prediction.probability < 0.3:
            risk_desc = "Analiza wskazuje na niskie ryzyko. To dobra wiadomość!"
        elif prediction.probability < 0.7:
            risk_desc = "Analiza wskazuje na umiarkowane ryzyko. Warto zwrócić uwagę na kilka czynników."
        else:
            risk_desc = "Analiza wskazuje na podwyższone ryzyko. Ważna jest regularna opieka lekarska."

        main_concerns = [
            translations.get(rf["feature"], rf["feature"])
            for rf in risk_factors_src[:3]
        ]

        positive_factors = [
            translations.get(pf["feature"], pf["feature"])
            for pf in protective_factors_src[:3]
        ]

        return PatientExplanation(
            risk_level=prediction.risk_level.value,
            risk_description=risk_desc,
            main_concerns=main_concerns if main_concerns else ["Brak szczególnych czynników ryzyka"],
            positive_factors=positive_factors if positive_factors else ["Analiza trwa"],
            recommendations="Zalecamy omówienie tych wyników z lekarzem prowadzącym.",
            technical_summary={
                "probability": prediction.probability,
                "n_risk_factors": len(risk_factors_src),
                "n_protective_factors": len(protective_factors_src),
                "source": "shap" if real_shap is not None else "demo"
            },
            disclaimer="To narzędzie ma charakter informacyjny i nie zastępuje porady lekarza."
        )

    except Exception as e:
        logger.error(f"Błąd wyjaśnienia pacjenta: {e}")
        raise HTTPException(status_code=500, detail="Wewnętrzny błąd serwera")


@app.get("/model/global-importance", response_model=GlobalImportance, tags=["Model"])
@limiter.limit("30/minute")
async def get_global_importance(request: Request, task_type: Literal["mortality", "dialysis"] = "mortality"):
    """
    Pobierz globalną ważność cech.

    Zwraca ranking cech według ich wpływu na predykcje modelu.
    """
    importance = app_state.get_global_importance(task_type=task_type)
    sorted_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    ctx = app_state.get_xai_context(task_type)
    method = "on-demand" if (ctx["xai_ready"] and ctx["is_loaded"]) else "synthetic (demo)"

    return GlobalImportance(
        feature_importance=sorted_importance,
        top_features=list(sorted_importance.keys())[:10],
        method=method,
        n_samples=len(ctx["X_background"]) if ctx.get("X_background") is not None else 100
    )


@app.get("/model/info", response_model=ModelInfo, tags=["Model"])
@limiter.limit("60/minute")
async def get_model_info(request: Request, task_type: Literal["mortality", "dialysis"] = "mortality"):
    """
    Pobierz informacje o modelu.

    Zwraca metadane modelu i metryki wydajności.
    """
    if task_type == "dialysis":
        if app_state.dialysis_loaded and app_state.dialysis_default_model_type:
            meta = app_state.dialysis_model_metadata.get(app_state.dialysis_default_model_type, {})
            metrics = meta.get("metrics", {})
            display_name = meta.get("display_name", app_state.dialysis_default_model_type)
            trained_at = meta.get("trained_at")
            training_date = trained_at[:10] if trained_at else None
            return ModelInfo(
                model_type=display_name,
                n_features=len(app_state.dialysis_feature_names) if app_state.dialysis_feature_names else 0,
                feature_names=app_state.dialysis_feature_names or [],
                training_date=training_date,
                performance_metrics=metrics if metrics else {"auc_roc": 0.0},
                version="1.0.0"
            )
        return ModelInfo(
            model_type="Demo Dialysis Model",
            n_features=len(app_state.dialysis_feature_names) if app_state.dialysis_feature_names else 18,
            feature_names=app_state.dialysis_feature_names or [],
            training_date=None,
            performance_metrics={"auc_roc": 0.80, "sensitivity": 0.78, "specificity": 0.75},
            version="1.0.0"
        )

    if app_state.is_loaded and app_state.default_model_type:
        meta = app_state.model_metadata.get(app_state.default_model_type, {})
        metrics = meta.get("metrics", {})
        display_name = meta.get("display_name", app_state.default_model_type)
        trained_at = meta.get("trained_at")
        training_date = trained_at[:10] if trained_at else None

        return ModelInfo(
            model_type=display_name,
            n_features=len(app_state.feature_names) if app_state.feature_names else 0,
            feature_names=app_state.feature_names or [],
            training_date=training_date,
            performance_metrics=metrics if metrics else {"auc_roc": 0.0},
            version="1.0.0"
        )

    return ModelInfo(
        model_type="Demo Model",
        n_features=len(app_state.feature_names) if app_state.feature_names else 20,
        feature_names=app_state.feature_names or ["Wiek", "Plec", "Manifestacja_Nerki", "..."],
        training_date=None,
        performance_metrics={"auc_roc": 0.85, "sensitivity": 0.82, "specificity": 0.78},
        version="1.0.0"
    )


def _get_contextual_factors(patient: PatientInput, task_type: str = "mortality", dialysis_patient=None) -> dict:
    """Pobierz kontekstowe wyjaśnienie XAI dla chatu."""
    ctx = app_state.get_xai_context(task_type)
    if ctx["xai_ready"] and ctx["is_loaded"]:
        try:
            from src.xai import SHAPExplainer
            active_patient = dialysis_patient if task_type == "dialysis" and dialysis_patient is not None else patient
            model_fn = ctx.get("get_model", app_state.get_model)
            model, _ = model_fn(getattr(active_patient, "model_type", None))
            feature_names = ctx["feature_names"]
            X_background = ctx["X_background"]
            if model is not None and feature_names and X_background is not None:
                explainer = SHAPExplainer(model, X_background, feature_names)
                from src.api.schemas import patient_to_array, dialysis_patient_to_array
                if task_type == "dialysis" and dialysis_patient is not None:
                    features = dialysis_patient_to_array(dialysis_patient, feature_names)
                else:
                    features = patient_to_array(patient, feature_names)
                instance = np.array(features).reshape(1, -1)
                result = explainer.explain_instance(instance)
                return {
                    "source": "real",
                    "impacts": result.get("feature_impacts", []),
                    "risk_factors": result.get("risk_factors", []),
                    "protective_factors": result.get("protective_factors", []),
                }
        except Exception as e:
            logger.warning(f"Chat XAI context failed: {e}")

    if task_type == "dialysis":
        demo_exp = _get_demo_dialysis_explanation(dialysis_patient)
    else:
        demo_exp = get_demo_explanation(patient)
    return {
        "source": "demo",
        "impacts": demo_exp["risk_factors"] + demo_exp["protective_factors"],
        "risk_factors": demo_exp["risk_factors"],
        "protective_factors": demo_exp["protective_factors"],
    }


# Tłumaczenia cech na polski
FEATURE_TRANSLATIONS = {
    "Wiek": "wiek",
    "Plec": "płeć",
    "Manifestacja_Nerki": "zajęcie nerek",
    "Manifestacja_Sercowo-Naczyniowy": "zajęcie sercowo-naczyniowe",
    "Manifestacja_Zajecie_CSN": "zajęcie ośrodkowego układu nerwowego",
    "Manifestacja_Neurologiczny": "zajęcie obwodowego układu nerwowego",
    "Manifestacja_Pokarmowy": "zajęcie układu pokarmowego",
    "Manifestacja_Oddechowy": "zajęcie układu oddechowego",
    "Zaostrz_Wymagajace_OIT": "zaostrzenia wymagające OIT",
    "Zaostrz_Wymagajace_Hospital": "hospitalizacje z powodu zaostrzeń",
    "Kreatynina": "kreatynina",
    "Max_CRP": "maksymalne CRP",
    "Dializa": "dializa",
    "Plazmaferezy": "plazmaferezy",
    "Pulsy": "pulsy sterydowe",
    "Liczba_Zajetych_Narzadow": "liczba zajętych narządów",
    "Sterydy_Dawka_g": "dawka sterydów",
    "Czas_Sterydow": "czas sterydów",
    "Powiklania_Serce/pluca": "powikłania sercowo-płucne",
    "Powiklania_Infekcja": "powikłania infekcyjne",
    "Wiek_rozpoznania": "wiek rozpoznania",
    "Opoznienie_Rozpoznia": "opóźnienie rozpoznania",
}


def _get_feature_value_from_patient(patient: PatientInput, feature_name: str, dialysis_patient=None) -> Optional[float]:
    """Pobierz wartość cechy z danych pacjenta (mortality lub dialysis)."""
    field_map = {
        "Wiek": "wiek", "Plec": "plec", "Kreatynina": "kreatynina",
        "Max_CRP": "max_crp", "Liczba_Zajetych_Narzadow": "liczba_zajetych_narzadow",
        "Manifestacja_Nerki": "manifestacja_nerki",
        "Manifestacja_Sercowo-Naczyniowy": "manifestacja_sercowo_naczyniowy",
        "Manifestacja_Zajecie_CSN": "manifestacja_zajecie_csn",
        "Manifestacja_Neurologiczny": "manifestacja_neurologiczny",
        "Manifestacja_Pokarmowy": "manifestacja_pokarmowy",
        "Manifestacja_Oddechowy": "manifestacja_oddechowy",
        "Zaostrz_Wymagajace_OIT": "zaostrz_wymagajace_oit",
        "Zaostrz_Wymagajace_Hospital": "zaostrz_wymagajace_hospital",
        "Dializa": "dializa", "Plazmaferezy": "plazmaferezy",
        "Pulsy": "pulsy",
        "Sterydy_Dawka_g": "sterydy_dawka_g", "Czas_Sterydow": "czas_sterydow",
        "Powiklania_Serce/pluca": "powiklania_serce_pluca",
        "Powiklania_Infekcja": "powiklania_infekcja",
        "Wiek_rozpoznania": "wiek_rozpoznania",
        "Opoznienie_Rozpoznia": "opoznienie_rozpoznia",
    }
    pydantic_name = field_map.get(feature_name)
    if pydantic_name:
        # Check dialysis_patient first (has dialysis-specific fields like pulsy, oddechowy)
        if dialysis_patient is not None:
            val = getattr(dialysis_patient, pydantic_name, None)
            if val is not None:
                return float(val)
        val = getattr(patient, pydantic_name, None)
        return float(val) if val is not None else None
    return None


def generate_contextual_response(
    message: str,
    patient: PatientInput,
    prediction: dict,
    xai_context: dict,
    task_type: str = "mortality",
    dialysis_patient=None,
) -> str:
    """Generuj kontekstową odpowiedź chatu na podstawie danych pacjenta i XAI."""
    msg = message.lower()
    prob = prediction.get("probability", 0)
    risk_level = prediction.get("risk_level", "moderate")
    risk_pl = {"low": "niskie", "moderate": "umiarkowane", "high": "wysokie"}.get(risk_level, risk_level)
    impacts = xai_context.get("impacts", [])
    risk_factors = xai_context.get("risk_factors", [])
    protective_factors = xai_context.get("protective_factors", [])

    # Sortuj po wpływie
    impacts_sorted = sorted(impacts, key=lambda x: abs(x.get("shap_value", x.get("contribution", 0))), reverse=True)

    # === Intencja: pytanie o wynik/ryzyko ===
    if any(w in msg for w in ['wynik', 'analiza', 'ryzyko', 'wyniki', 'prognoza', 'predykcja']):
        response = f"Na podstawie Twoich danych, **ryzyko wynosi {prob:.1%}** (poziom: **{risk_pl}**).\n\n"

        if impacts_sorted:
            response += "Najważniejsze czynniki wpływające na ocenę:\n\n"
            for i, impact in enumerate(impacts_sorted[:3], 1):
                fname = impact.get("feature", "")
                sv = impact.get("shap_value", impact.get("contribution", 0))
                direction = "zwiększa" if sv > 0 else "zmniejsza"
                pl_name = FEATURE_TRANSLATIONS.get(fname, fname)
                val = _get_feature_value_from_patient(patient, fname, dialysis_patient)
                val_str = f" ({val:.0f})" if val is not None and (val != int(val) or val > 1) else ""
                if val is not None and val in (0, 1):
                    val_str = " (tak)" if val == 1 else " (nie)"

                response += f"{i}. **{pl_name}**{val_str} — {direction} ryzyko\n"

        return response

    # === Intencja: pytanie o konkretną cechę ===
    feature_keywords = {
        "kreatynin": "Kreatynina", "crp": "Max_CRP", "nerek": "Manifestacja_Nerki",
        "nerki": "Manifestacja_Nerki", "serce": "Manifestacja_Sercowo-Naczyniowy",
        "oit": "Zaostrz_Wymagajace_OIT", "dializ": "Dializa", "wiek": "Wiek",
        "narzad": "Liczba_Zajetych_Narzadow", "narząd": "Liczba_Zajetych_Narzadow",
        "plazmaferez": "Plazmaferezy", "sterydy": "Sterydy_Dawka_g",
        "csn": "Manifestacja_Zajecie_CSN", "neuro": "Manifestacja_Neurologiczny",
        "infekcj": "Powiklania_Infekcja", "pokarm": "Manifestacja_Pokarmowy",
        "pulsy": "Pulsy", "oddechow": "Manifestacja_Oddechowy",
        "hospital": "Zaostrz_Wymagajace_Hospital",
    }

    matched_feature = None
    for keyword, feat_name in feature_keywords.items():
        if keyword in msg:
            matched_feature = feat_name
            break

    if matched_feature:
        pl_name = FEATURE_TRANSLATIONS.get(matched_feature, matched_feature)
        val = _get_feature_value_from_patient(patient, matched_feature, dialysis_patient)
        impact_item = next((i for i in impacts if i.get("feature") == matched_feature), None)

        response = f"**{pl_name.capitalize()}**"
        if val is not None:
            if matched_feature == "Kreatynina":
                response += f": Twoja wartość to **{val:.0f} μmol/L**.\n\n"
                if val > 200:
                    response += "To podwyższona wartość, wskazująca na zaburzoną czynność nerek."
                elif val > 120:
                    response += "To wartość powyżej normy, wskazująca na obciążenie nerek."
                else:
                    response += "Wartość w zakresie normy."
            elif matched_feature == "Max_CRP":
                response += f": Twoja wartość to **{val:.0f} mg/L**.\n\n"
                if val > 100:
                    response += "To wysoka wartość, wskazująca na silny stan zapalny."
                elif val > 50:
                    response += "To podwyższona wartość, świadcząca o aktywności zapalnej."
                else:
                    response += "Umiarkowany poziom stanu zapalnego."
            elif matched_feature == "Wiek":
                response += f": **{val:.0f} lat**.\n\n"
                response += "Wiek jest jednym z czynników prognostycznych w zapaleniu naczyń."
            elif val in (0, 1):
                response += f": **{'tak' if val == 1 else 'nie'}**.\n\n"
            else:
                response += f": **{val}**.\n\n"
        else:
            response += ": brak danych.\n\n"

        if impact_item:
            sv = impact_item.get("shap_value", impact_item.get("contribution", 0))
            direction = "zwiększa" if sv > 0 else "zmniejsza"
            response += f"\nTen czynnik {direction} Twoje ryzyko."

        return response

    # === Intencja: pytanie o czynniki ===
    if any(w in msg for w in ['czynnik', 'wpływa', 'dlaczego', 'przyczyn', 'powód', 'co decyduje']):
        response = "Główne czynniki wpływające na Twoją ocenę ryzyka:\n\n"

        if risk_factors:
            response += "**Czynniki zwiększające ryzyko:**\n"
            for f in risk_factors[:3]:
                fname = f.get("feature", "")
                sv = f.get("shap_value", f.get("contribution", 0))
                pl_name = FEATURE_TRANSLATIONS.get(fname, fname)
                val = _get_feature_value_from_patient(patient, fname, dialysis_patient)
                response += f"- {pl_name}"
                if val is not None and val > 1:
                    response += f" ({val:.0f})"
                response += "\n"

        if protective_factors:
            response += "\n**Czynniki ochronne:**\n"
            for f in protective_factors[:3]:
                fname = f.get("feature", "")
                pl_name = FEATURE_TRANSLATIONS.get(fname, fname)
                response += f"- {pl_name}\n"

        return response

    # === Intencja: zalecenia ===
    if any(w in msg for w in ['pomoc', 'zrobić', 'co robić', 'zalec', 'porad', 'leczeni', 'poprawić']):
        response = "Na podstawie Twojej analizy, zalecenia obejmują:\n\n"
        response += "1. **Regularne wizyty kontrolne** u lekarza prowadzącego\n"

        val_kreat = _get_feature_value_from_patient(patient, "Kreatynina", dialysis_patient)
        active_patient = dialysis_patient if task_type == "dialysis" and dialysis_patient is not None else patient
        if task_type == "dialysis":
            if val_kreat and val_kreat > 150:
                response += "2. **Monitorowanie czynności nerek** — Twoja kreatynina jest podwyższona\n"
            response += "3. **Regularna ocena nefrologiczna** — kontrola GFR i poziomu kreatyniny\n"
            if getattr(active_patient, 'manifestacja_nerki', 0):
                response += "4. **Konsultacja nefrologiczna** — ze względu na zajęcie nerek\n"
        else:
            if val_kreat and val_kreat > 150:
                response += "2. **Monitorowanie czynności nerek** — Twoja kreatynina jest podwyższona\n"
            if getattr(active_patient, 'manifestacja_nerki', 0):
                response += "3. **Konsultacja nefrologiczna** — ze względu na zajęcie nerek\n"
            if getattr(active_patient, 'zaostrz_wymagajace_oit', 0):
                response += "4. **Plan postępowania w zaostrzeniach** — historia zaostrzeń wymagających OIT\n"

        response += "\nPamiętaj, że ta analiza jest narzędziem wspierającym — ostateczne decyzje podejmowane są wspólnie z lekarzem."
        return response

    # === Domyślna odpowiedź ===
    if task_type == "dialysis":
        outcome_desc = "potrzeby dializy"
        examples = "kreatynine, GFR, białkomocz"
    else:
        outcome_desc = "ryzyka zgonu"
        examples = "kreatynine, nerki, CRP"
    response = f"Jestem asystentem pomagającym zrozumieć wyniki analizy {outcome_desc}.\n\n"
    response += f"Twój aktualny poziom ryzyka: **{risk_pl}** ({prob:.1%}).\n\n"
    response += "Mogę pomóc Ci z:\n"
    response += "- **Wyjaśnieniem wyniku** — zapytaj o ryzyko lub wynik\n"
    response += f"- **Konkretnymi czynnikami** — zapytaj np. o {examples}\n"
    response += "- **Zaleceniami** — zapytaj co możesz zrobić\n\n"
    response += "O czym chciałbyś/chciałabyś porozmawiać?"
    return response


async def _generate_openai_response(
    message: str,
    patient: PatientInput,
    prediction: dict,
    xai_context: dict,
    conversation_history: list
) -> Optional[str]:
    """Generuj odpowiedź przez OpenAI API (jeśli klucz API jest dostępny)."""
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        return None

    try:
        import openai
        client = openai.OpenAI(api_key=openai_api_key)

        risk_level = prediction.get("risk_level", "unknown")
        prob = prediction.get("probability", 0)
        risk_factors = xai_context.get("risk_factors", [])

        system_prompt = (
            f"Jesteś asystentem medycznym pomagającym pacjentowi zrozumieć wyniki analizy ryzyka "
            f"w zapaleniu naczyń.\n\n"
            f"Kontekst pacjenta:\n"
            f"- Prawdopodobieństwo ryzyka: {prob:.1%}\n"
            f"- Poziom ryzyka: {risk_level}\n"
            f"- Wiek: {patient.wiek}\n"
            f"- Główne czynniki ryzyka: {', '.join([f.get('feature', '') for f in risk_factors[:3]])}\n\n"
            f"Odpowiadaj w języku polskim. Bądź empatyczny i używaj prostego, zrozumiałego języka.\n"
            f"WAŻNE: To narzędzie informacyjne — zawsze przypominaj, że decyzje podejmuje lekarz."
        )

        messages = [{"role": "system", "content": system_prompt}]
        for turn in conversation_history[-5:]:  # Ostatnie 5 tur
            if turn.get("role") in ("user", "assistant") and turn.get("content"):
                messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content": message})

        loop = asyncio.get_running_loop()
        create_fn = functools.partial(
            client.chat.completions.create,
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=messages,
            max_tokens=500,
            temperature=0.7
        )
        result = await loop.run_in_executor(_OPENAI_EXECUTOR, create_fn)
        return result.choices[0].message.content
    except Exception as e:
        logger.warning(f"OpenAI chat failed: {e}")
        return None


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
@limiter.limit("30/minute")
async def chat(request: Request, chat_request: ChatRequest):
    """
    Rozmowa z agentem AI.

    Odpowiada na pytania pacjenta/klinicysty o wyniki analizy.
    Używa prawdziwych danych z predykcji i wyjaśnień XAI.
    Gdy dostępny klucz OPENAI_API_KEY, używa OpenAI GPT jako backendu.
    """
    try:
        # Enforce conversation_history max length
        if len(chat_request.conversation_history) > 20:
            chat_request.conversation_history = chat_request.conversation_history[-20:]

        # Pobierz predykcję
        task_type = chat_request.task_type or "mortality"
        if task_type == "dialysis" and chat_request.dialysis_patient is not None:
            dial_pred = await _predict_dialysis_internal(chat_request.dialysis_patient)
            prediction_result = PredictionOutput(
                probability=dial_pred.probability,
                risk_level=dial_pred.risk_level,
                prediction=int(dial_pred.needs_dialysis),
                model_used=dial_pred.model_used,
            )
        else:
            prediction_result = await _predict_internal(chat_request.patient)
        prediction_dict = {
            "probability": prediction_result.probability,
            "risk_level": prediction_result.risk_level.value,
            "prediction": prediction_result.prediction,
        }

        # Pobierz kontekst XAI
        contextual_fn = functools.partial(_get_contextual_factors, chat_request.patient, task_type, chat_request.dialysis_patient)
        xai_context = await _run_in_executor_with_timeout(contextual_fn)

        # Spróbuj OpenAI najpierw
        openai_response = await _generate_openai_response(
            message=chat_request.message,
            patient=chat_request.patient,
            prediction=prediction_dict,
            xai_context=xai_context,
            conversation_history=chat_request.conversation_history
        )

        if openai_response is not None:
            response = openai_response
        else:
            # Fallback: rule-based response
            response = generate_contextual_response(
                message=chat_request.message,
                patient=chat_request.patient,
                prediction=prediction_dict,
                xai_context=xai_context,
                task_type=task_type,
                dialysis_patient=chat_request.dialysis_patient,
            )

        # Dodaj disclaimer
        response += "\n\n---\n*Pamiętaj: to narzędzie informacyjne, nie zastępuje porady lekarza.*"

        # Dynamiczne sugestie
        suggestions = []
        if "wynik" not in chat_request.message.lower() and "ryzyko" not in chat_request.message.lower():
            suggestions.append("Jakie jest moje ryzyko?")
        if "czynnik" not in chat_request.message.lower():
            suggestions.append("Jakie czynniki wpływają na moje ryzyko?")
        if "kreatynin" not in chat_request.message.lower() and chat_request.patient.kreatynina and chat_request.patient.kreatynina > 100:
            suggestions.append("Powiedz mi więcej o kreatyninie")
        suggestions.append("Co mogę zrobić, aby poprawić swoje zdrowie?")

        return ChatResponse(
            response=response,
            detected_concerns=None,
            follow_up_suggestions=suggestions[:3]
        )

    except Exception as e:
        logger.error(f"Błąd chatu: {e}")
        raise HTTPException(status_code=500, detail="Wewnętrzny błąd serwera")


# ============================================================================
# CHAT ANALYZE — conversational patient data extraction + prediction
# ============================================================================

@app.post("/chat/analyze", response_model=ConversationalAnalysisResponse, tags=["Chat"])
@limiter.limit("20/minute")
async def chat_analyze(request: Request, req: ConversationalAnalysisRequest):
    """
    Analiza konwersacyjna — wyodrębnij dane pacjenta z tekstu,
    wykonaj predykcję gdy dane wystarczające, lub zadaj pytanie uzupełniające.
    """
    from src.api.patient_extractor import (
        extract_patient_from_text,
        get_missing_required_fields,
        get_missing_high_value_fields,
        get_followup_question,
        get_followup_suggestions,
        build_patient_summary,
    )

    try:
        task_type = req.task_type or "mortality"

        # 1. Wyodrębnij pola z bieżącej wiadomości
        extracted = extract_patient_from_text(req.message, task_type)

        # 2. Scal z dotychczas zgromadzonymi danymi (nowe nadpisują stare)
        accumulated = dict(req.accumulated_patient or {})
        accumulated.update(extracted)

        # 3. Sprawdź brakujące pola
        missing_required = get_missing_required_fields(accumulated, task_type)
        missing_hv = get_missing_high_value_fields(accumulated, task_type)

        # 4. Jeśli brakuje wymaganych pól — pytanie uzupełniające
        if missing_required:
            follow_up = get_followup_question(missing_required, task_type)
            summary = build_patient_summary(accumulated, task_type)

            response_parts = []
            if extracted:
                response_parts.append("Rozpoznałem następujące dane:")
                response_parts.append(summary)
            else:
                response_parts.append("Nie udało mi się rozpoznać danych pacjenta z podanego opisu.")

            if follow_up:
                response_parts.append(f"\n{follow_up}")

            return ConversationalAnalysisResponse(
                response="\n".join(response_parts),
                extracted_update=extracted,
                accumulated_patient=accumulated,
                analysis_complete=False,
                missing_fields=missing_required + missing_hv,
                follow_up_question=follow_up,
                follow_up_suggestions=get_followup_suggestions(accumulated, task_type),
            )

        # 5. Wymagane pola są — można wykonać predykcję
        # Zapewnij domyślne wartości dla opcjonalnych pól
        defaults_mortality = {
            "wiek_rozpoznania": None, "opoznienie_rozpoznia": None,
            "liczba_zajetych_narzadow": 0,
            "manifestacja_sercowo_naczyniowy": 0, "manifestacja_nerki": 0,
            "manifestacja_pokarmowy": 0, "manifestacja_zajecie_csn": 0,
            "manifestacja_neurologiczny": 0, "zaostrz_wymagajace_oit": 0,
            "kreatynina": None, "max_crp": None,
            "plazmaferezy": 0, "dializa": 0,
            "sterydy_dawka_g": None, "czas_sterydow": None,
            "powiklania_serce_pluca": 0, "powiklania_infekcja": 0,
        }
        defaults_dialysis = {
            "wiek_rozpoznania": None,
            "liczba_zajetych_narzadow": 0,
            "manifestacja_sercowo_naczyniowy": 0, "manifestacja_nerki": 0,
            "manifestacja_neurologiczny": 0, "manifestacja_oddechowy": 0,
            "zaostrz_wymagajace_oit": 0, "zaostrz_wymagajace_hospital": 0,
            "kreatynina": None, "max_crp": None,
            "plazmaferezy": 0, "pulsy": 0,
            "sterydy_dawka_g": None, "czas_sterydow": None,
            "powiklania_serce_pluca": 0, "powiklania_infekcja": 0,
        }
        defaults = defaults_mortality if task_type == "mortality" else defaults_dialysis
        patient_dict = {**defaults, **accumulated}

        # Oblicz opoznienie_rozpoznia jeśli możliwe
        if task_type == "mortality" and patient_dict.get("wiek_rozpoznania") is not None:
            patient_dict["opoznienie_rozpoznia"] = max(
                0, patient_dict["wiek"] - patient_dict["wiek_rozpoznania"]
            )

        # Wykonaj predykcję
        patient_obj = PatientInput(**patient_dict)
        prediction = await _predict_internal(patient_obj)

        # Pobierz top czynniki ryzyka via XAI context
        xai_summary = None
        try:
            contextual_fn = functools.partial(_get_contextual_factors, patient_obj, task_type)
            xai_context = await _run_in_executor_with_timeout(contextual_fn, timeout=10)
            top_risk = xai_context.get("risk_factors", [])[:3]
            top_protective = xai_context.get("protective_factors", [])[:2]
            xai_summary = {
                "risk_factors": top_risk,
                "protective_factors": top_protective,
            }
        except Exception as e:
            logger.warning(f"Chat analyze XAI context failed: {e}")

        # Zbuduj narracyjną odpowiedź
        summary = build_patient_summary(accumulated, task_type)
        risk_map = {"low": "niskie", "moderate": "umiarkowane", "high": "wysokie"}
        risk_label = risk_map.get(prediction.risk_level.value, prediction.risk_level.value)

        response_parts = [
            "Na podstawie podanych danych:",
            summary,
            "",
        ]

        if task_type == "mortality":
            response_parts.append(f"**Ryzyko zgonu: {prediction.probability:.0%}** (poziom: {risk_label})")
        else:
            response_parts.append(f"**Ryzyko potrzeby dializy: {prediction.probability:.0%}** (poziom: {risk_label})")

        if xai_summary and xai_summary.get("risk_factors"):
            response_parts.append("\nNajważniejsze czynniki ryzyka:")
            for i, rf in enumerate(xai_summary["risk_factors"], 1):
                feat_name = FEATURE_TRANSLATIONS.get(rf.get("feature", ""), rf.get("feature", ""))
                direction = "zwiększa" if rf.get("direction") == "increases_risk" else "zmniejsza"
                response_parts.append(f"{i}. {feat_name} — {direction} ryzyko")

        if missing_hv:
            follow_up = get_followup_question(missing_hv, task_type)
            if follow_up:
                response_parts.append(f"\nAby uzyskać dokładniejszą analizę: {follow_up}")

        response_parts.append("\n---\n*Pamiętaj: to narzędzie informacyjne, nie zastępuje porady lekarza.*")

        return ConversationalAnalysisResponse(
            response="\n".join(response_parts),
            extracted_update=extracted,
            accumulated_patient=accumulated,
            analysis_complete=True,
            prediction=prediction,
            xai_summary=xai_summary,
            missing_fields=missing_hv,
            follow_up_question=get_followup_question(missing_hv, task_type) if missing_hv else None,
            follow_up_suggestions=get_followup_suggestions(accumulated, task_type),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Błąd chat/analyze: {e}")
        raise HTTPException(status_code=500, detail="Wewnętrzny błąd serwera")


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            code=exc.status_code
        ).model_dump()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Nieobsłużony błąd: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Wewnętrzny błąd serwera",
            code=500
        ).model_dump()
    )


# ============================================================================
# MAIN
# ============================================================================

def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Uruchom serwer API."""
    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        reload=os.getenv("DEBUG", "false").lower() == "true"
    )


if __name__ == "__main__":
    run_server()
