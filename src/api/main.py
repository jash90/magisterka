"""
FastAPI aplikacja dla systemu XAI.

Główny plik API z endpointami dla predykcji,
wyjaśnień XAI i agenta konwersacyjnego.
"""

import os
import sys
import time
import asyncio
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd
import json
import logging
import joblib

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Dodaj ścieżkę projektu
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from .schemas import (
    PatientInput, PredictionOutput, SHAPExplanation, LIMEExplanation,
    PatientExplanation, ModelInfo, GlobalImportance, HealthCheckResponse,
    ChatRequest, ChatResponse, ExplanationRequest, PatientExplanationRequest,
    ComparisonResult, KrishnaComparisonResult,
    CalibrationResponse, ModelCalibration, CalibrationCurvePoint,
    DecisionCurveResponse, ModelDCA, NetBenefitPoint,
    CounterfactualResponse, CounterfactualExample, CounterfactualChange, CounterfactualMetrics,
    ErrorResponse, RiskLevel, XAIMethod, HealthLiteracyLevel,
    patient_to_array, get_risk_level_from_probability, patients_to_matrix,
    # Batch schemas
    BatchPatientInput, BatchPredictionOutput, BatchPatientResult,
    BatchSummary, BatchProcessingError, RiskFactorItem,
    DemoModeStatus, DemoModeRequest,
    # Agent schemas
    AgentPredictionData, AgentPredictionFactor,
    AgentConversationRequest, AgentConversationResponse,
    # Multi-model schemas
    MultiModelPredictionOutput, ModelPrediction,
    # DALEX/EBM schemas
    DALEXExplanation, EBMExplanation,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# GLOBAL STATE
# ============================================================================

class AppState:
    """Stan aplikacji z wczytanymi modelami."""

    def __init__(self):
        self.model = None
        self.feature_names = None
        self.model_path: Optional[Path] = None
        self.model_metadata: Dict[str, Any] = {}
        self.lime_explainer = None
        self.shap_explainer = None
        self.rag_pipeline = None
        self.is_loaded = False

        # Konfiguracja trybu demo
        self.allow_demo = os.getenv("ALLOW_DEMO", "true").lower() == "true"
        self.force_api_mode = os.getenv("FORCE_API_MODE", "false").lower() == "true"

        if self.force_api_mode:
            self.allow_demo = False

        # Global feature importance cache
        self._global_importance_cache: Optional[Dict[str, float]] = None

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

    def get_global_importance(self) -> Dict[str, float]:
        """Pobierz global feature importance — obliczane dynamicznie z SHAP."""
        if self._global_importance_cache is None:
            if self.shap_explainer is not None:
                try:
                    X_bg = joblib.load(Path("models/saved/X_background.joblib"))
                    self._global_importance_cache = self.shap_explainer.get_global_importance(X_bg)
                    logger.info("Global importance obliczona z SHAP")
                except Exception as e:
                    logger.warning(f"Nie udało się obliczyć SHAP importance: {e}")
            # Fallback: feature importances z modelu (dla drzewiastych)
            if self._global_importance_cache is None and self.model is not None:
                if hasattr(self.model, 'feature_importances_') and self.feature_names:
                    self._global_importance_cache = dict(zip(
                        self.feature_names,
                        self.model.feature_importances_.tolist()
                    ))
                    logger.info("Global importance z feature_importances_")
                else:
                    self._global_importance_cache = {}
            if self._global_importance_cache is None:
                self._global_importance_cache = {}
        return self._global_importance_cache

    def load_models(self, model_path: str, feature_names_path: str):
        """Wczytaj modele i explainer'y."""
        import joblib

        try:
            self.model_path = Path(model_path)
            # Wczytaj model
            self.model = joblib.load(model_path)
            logger.info(f"Model wczytany z {model_path}")

            # Wczytaj nazwy cech
            with open(feature_names_path, 'r') as f:
                self.feature_names = json.load(f)
            logger.info(f"Wczytano {len(self.feature_names)} nazw cech")
            self.model_metadata = _load_primary_model_metadata()

            self.is_loaded = True

            # Inicjalizuj XAI explainery
            models_dir = Path(model_path).parent
            self._init_xai_explainers(models_dir)

            return True

        except Exception as e:
            logger.error(f"Błąd wczytywania modeli: {e}")
            return False

    def _init_xai_explainers(self, models_dir: Path):
        """Inicjalizuj SHAP i LIME explainery z prawdziwymi danymi."""
        import joblib
        from src.xai.shap_explainer import SHAPExplainer
        from src.xai.lime_explainer import LIMEExplainer

        try:
            X_background_path = models_dir / "X_background.joblib"
            X_train_path = models_dir / "X_train.joblib"
            if self.model_path and self.model_path.name == "best_single_model.joblib":
                X_background_path = models_dir / "best_single_X_background.joblib"
                X_train_path = models_dir / "best_single_X_train.joblib"

            if X_background_path.exists():
                X_background = joblib.load(X_background_path)
                if X_background.shape[1] == len(self.feature_names):
                    self.shap_explainer = SHAPExplainer(
                        model=self.model,
                        X_background=X_background,
                        feature_names=self.feature_names
                    )
                    logger.info("SHAPExplainer zainicjalizowany z prawdziwymi danymi")
                else:
                    logger.warning("Pominięto SHAP: X_background nie pasuje do liczby cech modelu")
            else:
                logger.warning("X_background.joblib nie znaleziony — SHAP w trybie demo")

            if X_train_path.exists():
                X_train = joblib.load(X_train_path)
                if X_train.shape[1] == len(self.feature_names):
                    self.lime_explainer = LIMEExplainer(
                        model=self.model,
                        X_train=X_train,
                        feature_names=self.feature_names
                    )
                    logger.info("LIMEExplainer zainicjalizowany z prawdziwymi danymi")
                else:
                    logger.warning("Pominięto LIME: X_train nie pasuje do liczby cech modelu")
            else:
                logger.warning("X_train.joblib nie znaleziony — LIME w trybie demo")

        except Exception as e:
            logger.error(f"Błąd inicjalizacji XAI explainerów: {e}")
            self.shap_explainer = None
            self.lime_explainer = None


app_state = AppState()


def _load_primary_model_metadata() -> Dict[str, Any]:
    """Load metadata describing the preferred production model."""
    best_single_path = Path("models/saved/best_single_model.json")
    if best_single_path.exists():
        try:
            with open(best_single_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            metadata.setdefault("metadata_path", str(best_single_path))
            return metadata
        except Exception as e:
            logger.warning(f"Nie udało się wczytać best_single_model.json: {e}")

    best_model_path = Path("models/saved/best_model.json")
    if best_model_path.exists():
        try:
            with open(best_model_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            metadata.setdefault("metadata_path", str(best_model_path))
            return metadata
        except Exception as e:
            logger.warning(f"Nie udało się wczytać best_model.json: {e}")

    return {}


def _resolve_primary_model_paths() -> Tuple[str, str]:
    """Resolve startup model paths from explicit env vars or training metadata."""
    env_model_path = os.getenv("MODEL_PATH")
    env_features_path = os.getenv("FEATURE_NAMES_PATH")
    if env_model_path:
        return env_model_path, env_features_path or "models/saved/feature_names.json"

    metadata = _load_primary_model_metadata()
    artifact = metadata.get("api_primary_model_artifact") or metadata.get("artifact")
    features = (
        metadata.get("api_primary_feature_names_path")
        or metadata.get("feature_names_path")
        or "feature_names.json"
    )

    if artifact:
        return str(Path("models/saved") / artifact), str(Path("models/saved") / features)

    return "models/saved/best_model.joblib", env_features_path or "models/saved/feature_names.json"


def _load_model_by_key(model_key: str):
    """Załaduj model z dysku na podstawie klucza (xgboost/random_forest/lightgbm)."""
    if model_key in {"primary", None}:
        if app_state.model is None:
            raise HTTPException(status_code=503, detail="Model nie jest wczytany")
        return app_state.model
    filename = _MODEL_FILES.get(model_key)
    if filename is None:
        raise HTTPException(status_code=400, detail=f"Nieznany model: {model_key}. Dostępne: xgboost, random_forest, lightgbm")
    model_path = Path("models/saved") / filename
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model {model_key} nie istnieje: {model_path}")
    try:
        return joblib.load(model_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Błąd wczytywania modelu {model_key}: {e}")


# ============================================================================
# STARTUP/SHUTDOWN
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Zdarzenie uruchomienia aplikacji."""
    logger.info("Uruchamianie API Vasculitis XAI...")

    # Próba wczytania modeli. Jeśli env nie nadpisuje ścieżki, preferuj
    # najlepszy pojedynczy model zapisany przez scripts/train_model.py --single-best.
    model_path, feature_names_path = _resolve_primary_model_paths()

    if Path(model_path).exists() and Path(feature_names_path).exists():
        app_state.load_models(model_path, feature_names_path)
    else:
        logger.warning("Pliki modelu nie znalezione. API działa w trybie demo.")


@app.on_event("shutdown")
async def shutdown_event():
    """Zdarzenie zamknięcia aplikacji."""
    logger.info("Zamykanie API...")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_demo_prediction(patient: PatientInput) -> PredictionOutput:
    """Wygeneruj demo predykcję gdy model nie jest załadowany."""
    risk_score = 0.0

    risk_score += patient.liczba_zajetych_narzadow * 0.1

    if patient.manifestacja_nerki:
        risk_score += 0.15
    if patient.manifestacja_sercowo_naczyniowy:
        risk_score += 0.15
    if patient.manifestacja_zajecie_csn:
        risk_score += 0.2
    if patient.zaostrz_wymagajace_oit:
        risk_score += 0.25
    if patient.zaostrz_wymagajace_hospital:
        risk_score += 0.1
    if patient.plazmaferezy:
        risk_score += 0.1

    probability = min(max(risk_score, 0.05), 0.95)

    return PredictionOutput(
        probability=probability,
        risk_level=get_risk_level_from_probability(probability),
        prediction=1 if probability > 0.5 else 0,
        confidence_interval={"lower": max(0, probability - 0.1), "upper": min(1, probability + 0.1)}
    )


def get_demo_explanation(patient: PatientInput) -> dict:
    """Wygeneruj demo wyjaśnienie."""
    risk_factors = []
    protective_factors = []

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


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/", tags=["Info"])
async def root():
    """Strona główna API."""
    return {
        "name": "Vasculitis XAI API",
        "version": "1.0.0",
        "description": "API do predykcji śmiertelności w zapaleniu naczyń z wyjaśnieniami XAI",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthCheckResponse, tags=["Info"])
async def health_check():
    """Sprawdzenie stanu API."""
    return HealthCheckResponse(
        status="healthy",
        model_loaded=app_state.is_loaded,
        api_version="1.0.0",
        timestamp=datetime.now().isoformat()
    )


@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
async def predict(patient: PatientInput):
    """
    Wykonaj predykcję ryzyka śmiertelności.

    Zwraca prawdopodobieństwo zgonu i poziom ryzyka dla pacjenta.
    """
    try:
        if app_state.is_loaded and app_state.model is not None:
            # Użyj prawdziwego modelu
            features = patient_to_array(patient, app_state.feature_names)
            X = np.array(features).reshape(1, -1)

            probability = app_state.model.predict_proba(X)[0, 1]
            prediction = int(probability > 0.5)

            return PredictionOutput(
                probability=float(probability),
                risk_level=get_risk_level_from_probability(probability),
                prediction=prediction
            )
        else:
            # Demo mode
            return get_demo_prediction(patient)

    except Exception as e:
        logger.error(f"Błąd predykcji: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# MULTI-MODEL PREDICTION
# ============================================================================

_MODEL_FILES = {
    "best_single": "best_single_model.joblib",
    "xgboost": "best_model.joblib",
    "random_forest": "random_forest_model.joblib",
    "lightgbm": "lightgbm_model.joblib",
    "gradient_boosting": "gradient_boosting_model.joblib",
    "logistic_regression": "logistic_regression_model.joblib",
    "svm": "svm_model.joblib",
    "neural_network": "neural_network_model.joblib",
    "catboost": "catboost_model.joblib",
}

_MODEL_LABELS = {
    "best_single": "Best Single Model",
    "xgboost": "XGBoost",
    "random_forest": "Random Forest",
    "lightgbm": "LightGBM",
    "gradient_boosting": "Gradient Boosting",
    "logistic_regression": "Logistic Regression",
    "svm": "SVM",
    "neural_network": "Neural Network (MLP)",
    "catboost": "CatBoost",
}


@app.post("/predict/all", response_model=MultiModelPredictionOutput, tags=["Prediction"])
async def predict_all_models(patient: PatientInput):
    """
    Predykcja ryzyka ze wszystkich dostępnych modeli.

    Zwraca wyniki XGBoost, Random Forest i LightGBM oraz średnią ensemble.
    """

    models_dir = Path("models/saved")
    features = patient_to_array(patient, app_state.feature_names)
    X = np.array(features).reshape(1, -1)

    results: List[ModelPrediction] = []
    probabilities: List[float] = []

    for model_key, filename in _MODEL_FILES.items():
        model_path = models_dir / filename
        if not model_path.exists():
            continue
        try:
            model = joblib.load(model_path)
            prob = float(model.predict_proba(X)[0, 1])
            pred = int(prob > 0.5)
            results.append(ModelPrediction(
                model_name=_MODEL_LABELS[model_key],
                probability=prob,
                risk_level=get_risk_level_from_probability(prob),
                prediction=pred,
            ))
            probabilities.append(prob)
        except Exception as e:
            logger.warning(f"Nie udało się wczytać modelu {model_key}: {e}")
            continue

    if not probabilities:
        raise HTTPException(status_code=500, detail="Brak dostępnych modeli")

    ensemble_prob = sum(probabilities) / len(probabilities)

    return MultiModelPredictionOutput(
        models=results,
        ensemble_probability=ensemble_prob,
        ensemble_risk_level=get_risk_level_from_probability(ensemble_prob),
        primary_model=app_state.model_metadata.get("model_type", "primary"),
    )


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
async def predict_batch(request: BatchPatientInput):
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

    n_patients = len(request.patients)

    if app_state.is_loaded and app_state.model is not None:
        # PRODUCTION MODE: Vectorized batch prediction
        try:
            # Convert all patients to matrix at once
            X = patients_to_matrix(request.patients, app_state.feature_names)

            # Single vectorized prediction call
            probabilities, predictions = batch_predict_vectorized(X, app_state.model)

            # Extract risk factors if requested
            if request.include_risk_factors:
                risk_factors = get_batch_risk_factors(
                    X, app_state.feature_names, request.top_n_factors
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
            for i, patient in enumerate(request.patients):
                try:
                    pred = await predict(patient)
                    results.append(BatchPatientResult(
                        index=i,
                        prediction=pred,
                        processing_status="success"
                    ))
                except Exception as inner_e:
                    errors.append(BatchProcessingError(
                        patient_index=i,
                        error_type="prediction",
                        error_message=str(inner_e),
                        is_recoverable=True
                    ))
    else:
        # DEMO MODE: Use heuristic predictions
        for i, patient in enumerate(request.patients):
            demo_pred = get_demo_prediction(patient)
            demo_factors = None

            if request.include_risk_factors:
                demo_exp = get_demo_explanation(patient)
                demo_factors = [
                    RiskFactorItem(
                        feature=f["feature"],
                        value=f["value"],
                        importance=abs(f["contribution"]),
                        direction="increases_risk" if f["contribution"] > 0 else "decreases_risk"
                    )
                    for f in (demo_exp["risk_factors"] + demo_exp["protective_factors"])[:request.top_n_factors]
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
# CONFIG ENDPOINTS
# ============================================================================

@app.get("/config/demo-mode", response_model=DemoModeStatus, tags=["Config"])
async def get_demo_mode_status():
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
async def set_demo_mode(request: DemoModeRequest):
    """
    Włącz lub wyłącz tryb demo.

    Gdy tryb demo jest wyłączony i model nie jest załadowany, API zwróci błędy 503.
    Nie można włączyć trybu demo gdy zmienna FORCE_API_MODE jest ustawiona.
    """
    try:
        app_state.set_demo_mode(request.enabled)
        return await get_demo_mode_status()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/explain/shap", response_model=SHAPExplanation, tags=["XAI"])
async def explain_shap(request: ExplanationRequest):
    """
    Wygeneruj wyjaśnienie SHAP.

    Zwraca wartości SHAP dla cech pacjenta. Obsługuje XGBoost, Random Forest i LightGBM
    (parametr model_key).
    """
    try:
        model = _load_model_by_key(request.model_key)
        model_label = _MODEL_LABELS.get(request.model_key or "xgboost", "XGBoost")

        prediction = await predict(request.patient)
        features = patient_to_array(request.patient, app_state.feature_names)
        instance = np.array(features)

        # Użyj preinit SHAP explainer dla XGBoost, stwórz nowy dla innych modeli
        if (request.model_key in ("xgboost", None)) and app_state.shap_explainer is not None:
            real_exp = app_state.shap_explainer.explain_instance(instance)
        else:
            from src.xai.shap_explainer import SHAPExplainer
            X_bg = joblib.load(Path("models/saved/X_background.joblib"))
            explainer = SHAPExplainer(
                model=model,
                X_background=X_bg,
                feature_names=app_state.feature_names,
            )
            real_exp = explainer.explain_instance(instance)

        shap_values = {}
        contributions = []
        risk_factors = []
        protective_factors = []

        for fi in real_exp['feature_impacts'][:request.num_features]:
            shap_values[fi['feature']] = fi['shap_value']
            direction = "increases_risk" if fi['shap_value'] > 0 else "decreases_risk"
            entry = {
                "feature": fi['feature'],
                "value": fi['feature_value'],
                "contribution": fi['shap_value'],
                "direction": direction
            }
            contributions.append(entry)
            if fi['shap_value'] > 0:
                risk_factors.append(entry)
            elif fi['shap_value'] < 0:
                protective_factors.append(entry)

        return SHAPExplanation(
            base_value=real_exp['base_value'],
            shap_values=shap_values,
            feature_contributions=contributions,
            risk_factors=risk_factors,
            protective_factors=protective_factors,
            prediction=prediction
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Błąd SHAP: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain/lime", response_model=LIMEExplanation, tags=["XAI"])
async def explain_lime(request: ExplanationRequest):
    """
    Wygeneruj wyjaśnienie LIME.

    Zwraca wagi cech z lokalnego modelu zastępczego. Obsługuje XGBoost, Random Forest i LightGBM.
    """
    try:
        model = _load_model_by_key(request.model_key)
        prediction = await predict(request.patient)

        # Użyj preinit LIME explainer dla XGBoost, stwórz nowy dla innych modeli
        if (request.model_key in ("xgboost", None)) and app_state.lime_explainer is not None:
            lime_exp = app_state.lime_explainer
        else:
            from src.xai.lime_explainer import LIMEExplainer
            X_train = joblib.load(Path("models/saved/X_train.joblib"))
            lime_exp = LIMEExplainer(
                model=model,
                X_train=X_train,
                feature_names=app_state.feature_names,
            )

        features = patient_to_array(request.patient, app_state.feature_names)
        instance = np.array(features)
        real_exp = lime_exp.explain_instance(instance, num_features=request.num_features)

        feature_weights = []
        risk_factors = []
        protective_factors = []

        for feat_desc, weight in real_exp['feature_weights'][:request.num_features]:
            feature_weights.append({
                "feature": feat_desc,
                "weight": weight,
                "condition": feat_desc
            })
            entry = {"feature": feat_desc, "weight": weight}
            if weight > 0:
                risk_factors.append(entry)
            else:
                protective_factors.append(entry)

        return LIMEExplanation(
            intercept=real_exp['intercept'],
            feature_weights=feature_weights,
            risk_factors=risk_factors,
            protective_factors=protective_factors,
            local_prediction=real_exp['local_prediction'],
            prediction=prediction
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Błąd LIME: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain/comparison", response_model=ComparisonResult, tags=["XAI"])
async def explain_comparison(request: ExplanationRequest):
    """
    Porównaj wyjaśnienia SHAP i LIME dla pacjenta.

    Zwraca rankingi cech z obu metod oraz miary zgodności.
    Obsługuje XGBoost, Random Forest i LightGBM.
    """
    try:
        model = _load_model_by_key(request.model_key)
        features = patient_to_array(request.patient, app_state.feature_names)
        instance = np.array(features)
        num_features = request.num_features

        shap_ranking: List[str] = []
        lime_ranking: List[str] = []

        # SHAP ranking
        if (request.model_key in ("xgboost", None)) and app_state.shap_explainer is not None:
            shap_exp = app_state.shap_explainer.explain_instance(instance)
        else:
            from src.xai.shap_explainer import SHAPExplainer
            X_bg = joblib.load(Path("models/saved/X_background.joblib"))
            shap_expl = SHAPExplainer(model=model, X_background=X_bg, feature_names=app_state.feature_names)
            shap_exp = shap_expl.explain_instance(instance)
        shap_ranking = [fi['feature'] for fi in shap_exp['feature_impacts'][:num_features]]

        # LIME ranking
        if (request.model_key in ("xgboost", None)) and app_state.lime_explainer is not None:
            lime_exp_obj = app_state.lime_explainer
        else:
            from src.xai.lime_explainer import LIMEExplainer
            X_train = joblib.load(Path("models/saved/X_train.joblib"))
            lime_exp_obj = LIMEExplainer(model=model, X_train=X_train, feature_names=app_state.feature_names)
        lime_result = lime_exp_obj.explain_instance(instance, num_features=num_features)
        for feat_desc, _ in lime_result['feature_weights'][:num_features]:
            for fn in app_state.feature_names:
                if fn in feat_desc:
                    if fn not in lime_ranking:
                        lime_ranking.append(fn)
                    break

        # Oblicz zgodność
        shap_set = set(shap_ranking[:num_features])
        lime_set = set(lime_ranking[:num_features])
        common = list(shap_set & lime_set)
        union = shap_set | lime_set
        agreement = len(common) / len(union) if union else 0.0

        # Korelacja Spearmana (na wspólnych cechach)
        spearman_corr = 0.0
        if len(common) >= 2:
            from scipy.stats import spearmanr
            shap_ranks = [shap_ranking.index(f) for f in common if f in shap_ranking]
            lime_ranks = [lime_ranking.index(f) for f in common if f in lime_ranking]
            if len(shap_ranks) >= 2:
                corr, _ = spearmanr(shap_ranks, lime_ranks)
                spearman_corr = float(corr) if not np.isnan(corr) else 0.0

        return ComparisonResult(
            methods_compared=["SHAP", "LIME"],
            ranking_agreement=agreement,
            common_top_features=common,
            individual_rankings={"SHAP": shap_ranking, "LIME": lime_ranking},
            spearman_correlations={"SHAP_vs_LIME": spearman_corr}
        )

    except Exception as e:
        logger.error(f"Błąd comparison: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain/comparison/krishna", response_model=KrishnaComparisonResult, tags=["XAI"])
async def explain_comparison_krishna(request: ExplanationRequest):
    """Pełny panel metryk porównania XAI: Krishna et al. 2024 + RBO + Weighted Kendall.

    Liczy 6 metryk Krishna et al. (Feature/Rank/Sign/Signed-Rank/Pairwise-Rank
    Agreement) przy k=5 i k=10, RBO_min (Webber et al. 2010) z p=0.9, oraz
    Weighted Kendall tau (Vigna 2015) dla każdej pary metod XAI dostępnych
    dla wybranego modelu.
    """
    try:
        model = _load_model_by_key(request.model_key)
        features = patient_to_array(request.patient, app_state.feature_names)
        instance = np.array(features)
        num_features = max(request.num_features or 10, 10)

        from src.xai.shap_explainer import SHAPExplainer
        from src.xai.lime_explainer import LIMEExplainer
        from src.xai.dalex_wrapper import DALEXWrapper
        from src.xai.comparison import XAIComparison

        explanations: Dict[str, Dict[str, Any]] = {}

        # SHAP
        try:
            if (request.model_key in ("xgboost", None)) and app_state.shap_explainer is not None:
                explanations["SHAP"] = app_state.shap_explainer.explain_instance(instance)
            else:
                X_bg = joblib.load(Path("models/saved/X_background.joblib"))
                shap_expl = SHAPExplainer(model=model, X_background=X_bg, feature_names=app_state.feature_names)
                explanations["SHAP"] = shap_expl.explain_instance(instance)
        except Exception as e:
            logger.warning(f"SHAP nie powiodło się: {e}")

        # LIME
        try:
            if (request.model_key in ("xgboost", None)) and app_state.lime_explainer is not None:
                lime_obj = app_state.lime_explainer
            else:
                X_train = joblib.load(Path("models/saved/X_train.joblib"))
                lime_obj = LIMEExplainer(model=model, X_train=X_train, feature_names=app_state.feature_names)
            explanations["LIME"] = lime_obj.explain_instance(instance, num_features=num_features)
        except Exception as e:
            logger.warning(f"LIME nie powiodło się: {e}")

        # DALEX (break-down)
        try:
            X_bg = joblib.load(Path("models/saved/X_background.joblib"))
            y_bg = joblib.load(Path("models/saved/y_background.joblib"))
            dalex_w = DALEXWrapper(
                model=model, X=X_bg, y=y_bg,
                feature_names=app_state.feature_names,
                label=_MODEL_LABELS.get(request.model_key or "xgboost", "Model"),
            )
            bd = dalex_w.explain_instance_break_down(instance)
            if bd.get("contributions"):
                explanations["DALEX"] = bd
        except Exception as e:
            logger.warning(f"DALEX nie powiodło się: {e}")

        if len(explanations) < 2:
            raise HTTPException(
                status_code=503,
                detail="Krishna panel wymaga co najmniej 2 sprawnych metod XAI dla danego modelu.",
            )

        comp = XAIComparison(feature_names=app_state.feature_names)
        result = comp.compute_krishna_panel(explanations, ks=(5, 10), rbo_p=0.9)

        # Convert DataFrame panels to nested dicts for JSON
        panels_serialized: Dict[str, Dict[str, Dict[str, float]]] = {}
        for metric_name, df in result["panels"].items():
            panels_serialized[metric_name] = {
                row: {col: float(df.loc[row, col]) for col in df.columns}
                for row in df.index
            }

        return KrishnaComparisonResult(
            methods_compared=list(explanations.keys()),
            ks=result["config"]["ks"],
            rbo_p=result["config"]["rbo_p"],
            panels=panels_serialized,
            summary={k: float(v) for k, v in result["summary"].items()},
            rankings={m: list(r) for m, r in result["rankings"].items()},
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Błąd Krishna comparison: {e}")
        raise HTTPException(status_code=500, detail="Błąd obliczenia panelu Krishna")


# ============================================================================
# DALEX EXPLANATION
# ============================================================================

@app.post("/explain/dalex", response_model=DALEXExplanation, tags=["XAI"])
async def explain_dalex(request: ExplanationRequest):
    """
    Wyjaśnienie DALEX Break Down + Permutation Importance.

    Zwraca break-down analysis (lokalne wyjaśnienie) oraz globalną ważność cech
    za pomocą permutacji. Obsługuje XGBoost, Random Forest i LightGBM.
    """
    try:
        model = _load_model_by_key(request.model_key)
        model_label = _MODEL_LABELS.get(request.model_key or "xgboost", "XGBoost")

        from src.xai.dalex_wrapper import DALEXWrapper

        features = patient_to_array(request.patient, app_state.feature_names)
        X = np.array(features).reshape(1, -1)

        # Background data for explainer
        X_bg = joblib.load(Path("models/saved/X_background.joblib"))
        y_bg = joblib.load(Path("models/saved/y_background.joblib"))

        wrapper = DALEXWrapper(
            model=model,
            X=X_bg,
            y=y_bg,
            feature_names=app_state.feature_names,
            label=model_label
        )

        # Break-down for this patient
        bd = wrapper.explain_instance_break_down(X[0])

        # Variable importance (global)
        try:
            vi = wrapper.get_variable_importance(B=5)
        except Exception:
            vi = None

        risk_factors = [
            {"feature": c["variable"], "contribution": c["contribution"]}
            for c in bd.get("risk_factors", [])
        ]
        protective_factors = [
            {"feature": c["variable"], "contribution": c["contribution"]}
            for c in bd.get("protective_factors", [])
        ]

        return DALEXExplanation(
            intercept=bd["intercept"],
            prediction=bd["prediction"],
            risk_factors=risk_factors,
            protective_factors=protective_factors,
            variable_importance=vi,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Błąd DALEX: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# EBM EXPLANATION
# ============================================================================

# EBM is trained once on startup (lazy init)
_ebm_model = None
_ebm_lock = False


def _ensure_ebm():
    """Train EBM model on the training data (lazy, once)."""
    global _ebm_model, _ebm_lock
    if _ebm_model is not None:
        return _ebm_model
    if _ebm_lock:
        return None

    _ebm_lock = True
    try:
        from src.xai.ebm_explainer import EBMExplainer

        X_train = joblib.load(Path("models/saved/X_train.joblib"))
        y_train = joblib.load(Path("models/saved/y_train.joblib"))
        with open("models/saved/feature_names.json") as f:
            feature_names = json.load(f)

        ebm = EBMExplainer(
            feature_names=feature_names,
            max_rounds=500,
            interactions=0,
            random_state=42,
        )
        X_df = pd.DataFrame(X_train, columns=feature_names)
        ebm.fit(X_df, y_train, feature_names=feature_names)
        _ebm_model = ebm
        logger.info("EBM model wytrenowany")
        return ebm
    except Exception as e:
        logger.warning(f"Nie udało się wytrenować EBM: {e}")
        return None
    finally:
        _ebm_lock = False


@app.post("/explain/ebm", response_model=EBMExplanation, tags=["XAI"])
async def explain_ebm(request: ExplanationRequest):
    """
    Wyjaśnienie EBM (Explainable Boosting Machine).

    Zwraca lokalne wyjaśnienie + globalną ważność cech z inherentnie
    interpretowalnego modelu GAM.
    """
    try:
        import asyncio
        loop = asyncio.get_event_loop()
        ebm = await loop.run_in_executor(None, _ensure_ebm)
        if ebm is None:
            raise HTTPException(status_code=503, detail="Model EBM niedostępny")

        features = patient_to_array(request.patient, app_state.feature_names)
        X = np.array(features).reshape(1, -1)

        # Local explanation
        local = ebm.explain_local(X[0])

        # Global importance
        global_imp = ebm.get_feature_importance()

        prob = local["probability_positive"]
        pred = local["prediction"]

        contributions = [
            {"feature": c["feature"], "contribution": c["score"]}
            for c in local.get("contributions", [])
        ]

        # Get top interactions
        global_data = ebm.explain_global()
        interactions = global_data.get("interactions_detected", [])

        return EBMExplanation(
            prediction=pred,
            probability=prob,
            risk_level=get_risk_level_from_probability(prob),
            global_importance=global_imp,
            local_contributions=contributions,
            interactions=interactions[:10],
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Błąd EBM: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain/patient", response_model=PatientExplanation, tags=["XAI"])
async def explain_for_patient(request: PatientExplanationRequest):
    """
    Wygeneruj wyjaśnienie zrozumiałe dla pacjenta.

    Dostosowuje język i poziom szczegółowości do poziomu health literacy.
    Używa prawdziwych wartości SHAP, jeśli model jest załadowany.
    """
    try:
        prediction = await predict(request.patient)

        # Tłumaczenia cech
        translations = {
            "Wiek_rozpoznania": "Twój wiek w momencie rozpoznania",
            "Opoznienie_Rozpoznia": "Czas od objawów do diagnozy",
            "Manifestacja_Nerki": "Stan nerek",
            "Manifestacja_Sercowo-Naczyniowy": "Stan układu krążenia",
            "Manifestacja_Zajecie_CSN": "Zajęcie ośrodkowego układu nerwowego",
            "Manifestacja_Neurologiczny": "Objawy neurologiczne",
            "Manifestacja_Pokarmowy": "Stan układu pokarmowego",
            "Manifestacja_Skora": "Zmiany skórne",
            "Manifestacja_Wzrok": "Zajęcie narządu wzroku",
            "Manifestacja_Miesno-Szkiel": "Zajęcie układu mięśniowo-szkieletowego",
            "Manifestacja_Moczowo-Plciowy": "Zajęcie układu moczowo-płciowego",
            "Zaostrz_Wymagajace_OIT": "Przebyte poważne zaostrzenia (OIT)",
            "Zaostrz_Wymagajace_Hospital": "Zaostrzenia wymagające hospitalizacji",
            "Liczba_Zajetych_Narzadow": "Liczba dotkniętych narządów",
            "Kreatynina": "Wskaźnik czynności nerek (kreatynina)",
            "Plazmaferezy": "Przebyte zabiegi oczyszczania krwi",
            "Czas_Sterydow": "Czas leczenia sterydami",
            "Pulsy": "Leczenie pulsami sterydowymi",
            "Eozynofilia_Krwi_Obwodowych_Wartosc": "Poziom eozynofili we krwi",
            "Biopsja_Wynik": "Wynik biopsji",
        }

        # Spróbuj uzyskać prawdziwe wyjaśnienie SHAP
        risk_factors = []
        protective_factors = []
        base_value = 0.0
        used_real_shap = False

        if app_state.shap_explainer is not None and app_state.feature_names:
            try:
                features = patient_to_array(request.patient, app_state.feature_names)
                instance = np.array(features)
                shap_exp = app_state.shap_explainer.explain_instance(instance)

                base_value = shap_exp.get('base_value', 0.0)
                used_real_shap = True

                # Map risk/protective from real SHAP
                for fi in shap_exp.get('risk_factors', [])[:5]:
                    risk_factors.append({
                        "feature": fi["feature"],
                        "value": fi.get("feature_value", 0),
                        "contribution": fi.get("shap_value", 0),
                    })

                for fi in shap_exp.get('protective_factors', [])[:5]:
                    protective_factors.append({
                        "feature": fi["feature"],
                        "value": fi.get("feature_value", 0),
                        "contribution": fi.get("shap_value", 0),
                    })

            except Exception as e:
                logger.warning(f"SHAP explanation failed for patient: {e}")

        # Fallback na demo jeśli SHAP niedostępny
        if not used_real_shap:
            demo_exp = get_demo_explanation(request.patient)
            risk_factors = demo_exp.get("risk_factors", [])
            protective_factors = demo_exp.get("protective_factors", [])
            base_value = demo_exp.get("base_value", 0.0)

        # Poziom ryzyka
        if prediction.probability < 0.3:
            risk_desc = "Analiza wskazuje na niskie ryzyko. To dobra wiadomość!"
        elif prediction.probability < 0.7:
            risk_desc = "Analiza wskazuje na umiarkowane ryzyko. Warto zwrócić uwagę na kilka czynników."
        else:
            risk_desc = "Analiza wskazuje na podwyższone ryzyko. Ważna jest regularna opieka lekarska."

        # Tłumacz nazwy cech na język pacjenta
        main_concerns = [
            translations.get(rf["feature"], rf["feature"])
            for rf in risk_factors[:3]
        ]

        positive_factors = [
            translations.get(pf["feature"], pf["feature"])
            for pf in protective_factors[:3]
        ]

        return PatientExplanation(
            risk_level=prediction.risk_level.value,
            risk_description=risk_desc,
            main_concerns=main_concerns if main_concerns else ["Brak szczególnych czynników ryzyka"],
            positive_factors=positive_factors if positive_factors else ["Analiza trwa"],
            recommendations="Zalecamy omówienie tych wyników z lekarzem prowadzącym.",
            technical_summary={
                "probability": prediction.probability,
                "n_risk_factors": len(risk_factors),
                "n_protective_factors": len(protective_factors),
                "base_value": base_value,
                "method": "SHAP" if used_real_shap else "demo"
            } if request.health_literacy != HealthLiteracyLevel.BASIC else None,
            disclaimer="To narzędzie ma charakter informacyjny i nie zastępuje porady lekarza."
        )

    except Exception as e:
        logger.error(f"Błąd wyjaśnienia pacjenta: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/global-importance", response_model=GlobalImportance, tags=["Model"])
async def get_global_importance():
    """
    Pobierz globalną ważność cech.

    Zwraca ranking cech według ich wpływu na predykcje modelu.
    Wartości obliczane dynamicznie z SHAP (lub feature_importances_ jako fallback).
    """
    importance = app_state.get_global_importance()

    if not importance:
        # Brak modelu — zwróć puste dane
        return GlobalImportance(
            feature_importance={},
            top_features=[],
            method="Brak załadowanego modelu",
            n_samples=0
        )

    method = "SHAP TreeExplainer" if app_state.shap_explainer else "Feature Importances (tree-based)"

    return GlobalImportance(
        feature_importance=importance,
        top_features=list(importance.keys()),
        method=method,
        n_samples=100
    )


@app.get("/model/info", response_model=ModelInfo, tags=["Model"])
async def get_model_info():
    """
    Pobierz informacje o modelu.

    Zwraca metadane modelu i metryki wydajności z pliku evaluation_report.json.
    """
    metadata = app_state.model_metadata or _load_primary_model_metadata()
    model_type = metadata.get("model_type") or ("Primary Model" if app_state.is_loaded else "Demo Model")
    n_features = len(app_state.feature_names) if app_state.feature_names else 20
    feature_names = app_state.feature_names or []
    training_date = metadata.get("timestamp", "")[:10] if metadata.get("timestamp") else None
    performance_metrics = {
        "auc_roc": 0.0,
        "sensitivity": 0.0,
        "specificity": 0.0,
        "ppv": 0.0,
        "npv": 0.0
    }

    if metadata.get("auc_roc_mean") is not None:
        performance_metrics = {
            "auc_roc": round(metadata.get("auc_roc_mean", 0), 4),
            "auc_roc_std": round(metadata.get("auc_roc_std", 0), 4),
            "auc_pr": round(metadata.get("auc_pr_mean", 0), 4),
            "sensitivity": round(metadata.get("sensitivity_mean", 0), 4),
            "specificity": round(metadata.get("specificity_mean", 0), 4),
        }

    # Wczytaj metryki z evaluation_report.json
    if metadata.get("auc_roc_mean") is None:
        try:
            eval_report_path = Path("models/saved/evaluation_report.json")
            if eval_report_path.exists():
                with open(eval_report_path, 'r', encoding='utf-8') as f:
                    eval_data = json.load(f)
                preferred_key = (
                    metadata.get("single_best_model_type")
                    or metadata.get("best_model_type_by_nested_cv")
                    or "xgboost"
                )
                model_result = eval_data.get(preferred_key, eval_data.get("xgboost", eval_data.get("best", None)))
                if model_result:
                    metrics = model_result.get("metrics", {})
                    performance_metrics = {
                        "auc_roc": round(metrics.get("auc_roc", 0), 4),
                        "sensitivity": round(metrics.get("sensitivity", 0), 4),
                        "specificity": round(metrics.get("specificity", 0), 4),
                        "ppv": round(metrics.get("ppv", 0), 4),
                        "npv": round(metrics.get("npv", 0), 4),
                    }
        except Exception as e:
            logger.warning(f"Nie udało się wczytać evaluation_report.json: {e}")

    return ModelInfo(
        model_type=model_type,
        n_features=n_features,
        feature_names=feature_names if feature_names else ["Model nie załadowany"],
        training_date=training_date,
        performance_metrics=performance_metrics,
        version="1.0.0"
    )


# ============================================================================
# MODEL EVALUATION & CHARTS ENDPOINTS
# ============================================================================

@app.get("/model/evaluation", tags=["Model"])
async def get_model_evaluation():
    """
    Pobierz metryki ewaluacji wszystkich modeli.

    Zwraca pełne metryki medyczne (AUC, Sensitivity, Specificity, PPV, NPV, Brier Score)
    dla wszystkich wytrenowanych modeli z evaluation_report.json.
    """
    eval_report_path = Path("models/saved/evaluation_report.json")
    if not eval_report_path.exists():
        raise HTTPException(status_code=404, detail="Plik evaluation_report.json nie istnieje. Uruchom scripts/train_model.py")

    try:
        with open(eval_report_path, 'r', encoding='utf-8') as f:
            eval_data = json.load(f)

        # Przekształć do formy przyjaznej dla frontend
        result = {}
        for model_name, model_result in eval_data.items():
            metrics = model_result.get("metrics", {})
            result[model_name] = {
                "auc_roc": round(metrics.get("auc_roc", 0), 4),
                "auc_pr": round(metrics.get("auc_pr", 0), 4),
                "sensitivity": round(metrics.get("sensitivity", 0), 4),
                "specificity": round(metrics.get("specificity", 0), 4),
                "ppv": round(metrics.get("ppv", 0), 4),
                "npv": round(metrics.get("npv", 0), 4),
                "f1": round(metrics.get("f1", 0), 4),
                "accuracy": round(metrics.get("accuracy", 0), 4),
                "brier_score": round(metrics.get("brier_score", 0), 4),
                "mcc": round(metrics.get("mcc", 0), 4),
                "confusion_matrix": {
                    "tp": metrics.get("true_positives", 0),
                    "tn": metrics.get("true_negatives", 0),
                    "fp": metrics.get("false_positives", 0),
                    "fn": metrics.get("false_negatives", 0),
                }
            }

        return {"models": result}

    except Exception as e:
        logger.error(f"Błąd wczytywania evaluation_report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/comparison", tags=["Model"])
async def get_model_comparison():
    """
    Porównanie wszystkich modeli.

    Zwraca DataFrame porównania modeli z model_comparison.json.
    """
    comparison_path = Path("models/saved/model_comparison.json")
    if not comparison_path.exists():
        raise HTTPException(status_code=404, detail="Plik model_comparison.json nie istnieje. Uruchom scripts/train_model.py")

    try:
        with open(comparison_path, 'r', encoding='utf-8') as f:
            comparison_data = json.load(f)
        return {"comparison": comparison_data}
    except Exception as e:
        logger.error(f"Błąd wczytywania model_comparison: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/charts/roc", tags=["Model"])
async def get_roc_curves():
    """
    Dane do wykresów ROC dla wszystkich modeli.

    Zwraca fpr, tpr, auc dla każdego modelu — gotowe do narysowania krzywych ROC.
    Wymaga zapisanych X_test.joblib i y_test.joblib.
    """
    from sklearn.metrics import roc_curve, roc_auc_score

    X_test_path = Path("models/saved/X_test.joblib")
    y_test_path = Path("models/saved/y_test.joblib")

    if not X_test_path.exists() or not y_test_path.exists():
        raise HTTPException(status_code=404, detail="Brak X_test/y_test. Uruchom scripts/train_model.py")

    try:
        X_test = joblib.load(X_test_path)
        y_test = joblib.load(y_test_path)

        result = {}
        for model_key, filename in _MODEL_FILES.items():
            model_path = Path("models/saved") / filename
            if not model_path.exists():
                continue

            try:
                model = joblib.load(model_path)
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_test)[:, 1]
                else:
                    from scipy.special import expit
                    y_proba = expit(model.decision_function(X_test))

                fpr, tpr, _ = roc_curve(y_test, y_proba)
                auc = roc_auc_score(y_test, y_proba)

                result[model_key] = {
                    "label": _MODEL_LABELS.get(model_key, model_key),
                    "fpr": fpr.tolist(),
                    "tpr": tpr.tolist(),
                    "auc": round(float(auc), 4),
                }
            except Exception as e:
                logger.warning(f"Nie udało się obliczyć ROC dla {model_key}: {e}")
                continue

        return {"roc_curves": result}

    except Exception as e:
        logger.error(f"Błąd ROC curves: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/charts/confusion-matrix", tags=["Model"])
async def get_confusion_matrices():
    """
    Macierze konfuzji dla wszystkich modeli.

    Zwraca confusion matrix (raw + normalized) per model.
    """
    from sklearn.metrics import confusion_matrix

    X_test_path = Path("models/saved/X_test.joblib")
    y_test_path = Path("models/saved/y_test.joblib")

    if not X_test_path.exists() or not y_test_path.exists():
        raise HTTPException(status_code=404, detail="Brak X_test/y_test. Uruchom scripts/train_model.py")

    try:
        X_test = joblib.load(X_test_path)
        y_test = joblib.load(y_test_path)

        result = {}
        for model_key, filename in _MODEL_FILES.items():
            model_path = Path("models/saved") / filename
            if not model_path.exists():
                continue

            try:
                model = joblib.load(model_path)
                y_pred = model.predict(X_test)

                cm = confusion_matrix(y_test, y_pred)
                cm_normalized = (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]).round(4)

                result[model_key] = {
                    "label": _MODEL_LABELS.get(model_key, model_key),
                    "matrix": cm.tolist(),
                    "normalized": cm_normalized.tolist(),
                    "labels": ["Przeżycie", "Zgon"],
                }
            except Exception as e:
                logger.warning(f"Nie udało się obliczyć confusion matrix dla {model_key}: {e}")
                continue

        return {"confusion_matrices": result}

    except Exception as e:
        logger.error(f"Błąd confusion matrices: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain/counterfactuals", response_model=CounterfactualResponse, tags=["XAI"])
async def explain_counterfactuals(request: ExplanationRequest):
    """DiCE counterfactuals: minimalne zmiany cech modyfikowalnych do flipu klasy.

    Domyślnie zmienia tylko cechy modyfikowalne klinicznie (laboratoria, sterydy,
    plazmaferezy, leczenie immunosupresyjne) — wiek, płeć, typ ANCA, manifestacje
    i historia choroby pozostają niezmienione.

    OSTRZEŻENIE: counterfactuals są korelacyjne, nie przyczynowe (Prosperi 2020).
    Nie używać jako rekomendacji terapeutycznej bez walidacji klinicznej.
    """
    try:
        model = _load_model_by_key(request.model_key)
        features = patient_to_array(request.patient, app_state.feature_names)
        instance = np.array(features, dtype=float)

        X_train = joblib.load(Path("models/saved/X_train.joblib"))
        y_train = joblib.load(Path("models/saved/y_train.joblib"))

        from src.xai.dice_explainer import generate_counterfactuals as dice_gen

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: dice_gen(
                model=model,
                X_train=np.asarray(X_train),
                y_train=np.asarray(y_train),
                feature_names=app_state.feature_names,
                instance=instance,
                desired_class=0,
                total_CFs=request.num_features or 5,
                methods=("random",),
            ),
        )

        cfs_serialized = []
        for cf in result["cfs"]:
            cfs_serialized.append(CounterfactualExample(
                predicted_proba=cf["predicted_proba"],
                flipped_class=cf["flipped_class"],
                n_changes=cf["n_changes"],
                l1_distance=cf["l1_distance"],
                nearest_neighbor_distance=cf["nearest_neighbor_distance"],
                changes=[
                    CounterfactualChange(
                        feature=c["feature"],
                        **{"from": c["from"]},
                        to=c["to"],
                        delta=c["delta"],
                    )
                    for c in cf["changes"]
                ],
            ))

        return CounterfactualResponse(
            success=bool(result["success"]),
            method=result.get("method"),
            original_probability=float(result["original_proba"]),
            desired_class=0,
            cfs=cfs_serialized,
            metrics=CounterfactualMetrics(**result.get("metrics", {})) if result.get("metrics") else CounterfactualMetrics(),
            features_varied=list(result.get("features_varied", [])),
            message=str(result.get("message", "")),
        )
    except Exception as e:
        logger.error(f"Błąd counterfactuals: {e}")
        raise HTTPException(status_code=500, detail="Błąd generacji counterfactuals")


@app.get("/model/calibration", response_model=CalibrationResponse, tags=["Model"])
async def get_calibration():
    """Kalibracja wszystkich modeli na zbiorze testowym.

    Zwraca:
    - Brier score (im niższy, tym lepiej; 0 = doskonała kalibracja),
    - calibration slope i intercept (regresja logistyczna y ~ logit(p_pred);
      idealnie slope=1, intercept=0; slope<1 = nadmierna pewność),
    - punkty diagramu wiarygodności (mean_predicted vs fraction_positive).
    """
    try:
        from src.models.calibration_metrics import compute_calibration

        X_test = joblib.load(Path("models/saved/X_test.joblib"))
        y_test = joblib.load(Path("models/saved/y_test.joblib"))
        n_test = len(y_test)
        n_pos = int(np.asarray(y_test).sum())
        prevalence = n_pos / n_test if n_test else 0.0

        models_out: List[ModelCalibration] = []
        for model_key, filename in _MODEL_FILES.items():
            model_path = Path("models/saved") / filename
            if not model_path.exists():
                continue
            try:
                model = joblib.load(model_path)
                if not hasattr(model, "predict_proba"):
                    continue
                p = model.predict_proba(X_test)[:, 1]
                cal = compute_calibration(y_test, p, n_bins=10, strategy="quantile")
                models_out.append(ModelCalibration(
                    model=_MODEL_LABELS.get(model_key, model_key),
                    brier_score=float(cal["brier"]),
                    calibration_slope=float(cal["slope"]),
                    calibration_intercept=float(cal["intercept"]),
                    n_test=n_test,
                    curve=[CalibrationCurvePoint(**pt) for pt in cal["curve"]],
                ))
            except Exception as e:
                logger.warning(f"Kalibracja {model_key} nie powiodła się: {e}")
                continue

        return CalibrationResponse(
            n_test=n_test,
            n_positive=n_pos,
            prevalence=float(prevalence),
            models=models_out,
        )
    except Exception as e:
        logger.error(f"Błąd kalibracji: {e}")
        raise HTTPException(status_code=500, detail="Błąd obliczenia kalibracji")


@app.get("/model/decision-curve", response_model=DecisionCurveResponse, tags=["Model"])
async def get_decision_curve():
    """Decision Curve Analysis (DCA) — net benefit każdego modelu vs treat-all.

    Vickers & Elkin (2006): NB(t) = TP/N - FP/N * t/(1-t).
    Krzywa "treat all" jako odniesienie; "treat none" = 0 dla każdego progu.
    """
    try:
        from src.models.calibration_metrics import (
            compute_decision_curve, compute_treat_all_curve, default_threshold_grid,
        )

        X_test = joblib.load(Path("models/saved/X_test.joblib"))
        y_test = joblib.load(Path("models/saved/y_test.joblib"))
        n_test = len(y_test)
        n_pos = int(np.asarray(y_test).sum())
        prevalence = n_pos / n_test if n_test else 0.0

        thresholds = default_threshold_grid()
        treat_all = compute_treat_all_curve(y_test, thresholds)
        treat_all_pts = [NetBenefitPoint(**pt) for pt in treat_all]

        models_out: List[ModelDCA] = []
        for model_key, filename in _MODEL_FILES.items():
            model_path = Path("models/saved") / filename
            if not model_path.exists():
                continue
            try:
                model = joblib.load(model_path)
                if not hasattr(model, "predict_proba"):
                    continue
                p = model.predict_proba(X_test)[:, 1]
                pts = compute_decision_curve(y_test, p, thresholds)
                models_out.append(ModelDCA(
                    model=_MODEL_LABELS.get(model_key, model_key),
                    points=[NetBenefitPoint(**pt) for pt in pts],
                ))
            except Exception as e:
                logger.warning(f"DCA {model_key} nie powiodło się: {e}")
                continue

        return DecisionCurveResponse(
            n_test=n_test,
            n_positive=n_pos,
            prevalence=float(prevalence),
            threshold_grid=list(thresholds),
            treat_all=treat_all_pts,
            models=models_out,
        )
    except Exception as e:
        logger.error(f"Błąd DCA: {e}")
        raise HTTPException(status_code=500, detail="Błąd obliczenia DCA")


@app.get("/model/feature-selection", tags=["Model"])
async def get_feature_selection():
    """
    Wyniki feature selection.

    Zwraca dane z feature_selection_results.json i rfecv_results.json.
    """
    result = {}

    fs_path = Path("models/saved/feature_selection_results.json")
    if fs_path.exists():
        try:
            with open(fs_path, 'r', encoding='utf-8') as f:
                result["feature_selection"] = json.load(f)
        except Exception as e:
            logger.warning(f"Błąd wczytywania feature_selection_results: {e}")

    rfecv_path = Path("models/saved/rfecv_results.json")
    if rfecv_path.exists():
        try:
            with open(rfecv_path, 'r', encoding='utf-8') as f:
                result["rfecv"] = json.load(f)
        except Exception as e:
            logger.warning(f"Błąd wczytywania rfecv_results: {e}")

    if not result:
        raise HTTPException(status_code=404, detail="Brak plików feature selection")

    return result


# ============================================================================
# AGENT PREDICTION TOOL
# ============================================================================

async def run_prediction_tool(patient: PatientInput) -> AgentPredictionData:
    """
    Wykonaj predykcję z wyjaśnieniem SHAP jako narzędzie agenta.

    Zwraca dane predykcji + czynniki SHAP gotowe do wyświetlenia w czacie.
    """
    # 1. Predykcja
    pred = await predict(patient)

    factors: List[AgentPredictionFactor] = []
    base_value = 0.0

    # 2. SHAP explanation (jeśli dostępny)
    try:
        if app_state.shap_explainer is not None and app_state.feature_names:
            features = patient_to_array(patient, app_state.feature_names)
            instance = np.array(features)
            real_exp = app_state.shap_explainer.explain_instance(instance)
            base_value = real_exp.get('base_value', 0.0)

            for fi in real_exp['feature_impacts'][:10]:
                direction = "increases_risk" if fi['shap_value'] > 0 else "decreases_risk"
                factors.append(AgentPredictionFactor(
                    feature=fi['feature'],
                    contribution=fi['shap_value'],
                    direction=direction,
                ))
        else:
            # Fallback to demo explanation factors
            demo_exp = get_demo_explanation(patient)
            base_value = demo_exp.get('base_value', 0.0)
            for rf in demo_exp["risk_factors"]:
                factors.append(AgentPredictionFactor(
                    feature=rf["feature"],
                    contribution=rf["contribution"],
                    direction=rf.get("direction", "increases_risk"),
                ))
            for pf in demo_exp["protective_factors"]:
                factors.append(AgentPredictionFactor(
                    feature=pf["feature"],
                    contribution=pf["contribution"],
                    direction=pf.get("direction", "decreases_risk"),
                ))
    except Exception as e:
        logger.warning(f"SHAP explanation failed in agent tool: {e}")

    return AgentPredictionData(
        prediction=pred,
        factors=factors,
        base_value=base_value,
    )


@app.post("/agent/predict", response_model=AgentPredictionData, tags=["Agent"])
async def agent_predict(patient: PatientInput):
    """
    Narzędzie predykcji agenta — endpoint do wywoływania przez agenta.

    Wykonuje predykcję ryzyka śmiertelności i zwraca dane z wyjaśnieniem SHAP,
    gotowe do renderowania w interfejsie czatu (wykresy, czynniki).
    """
    try:
        return await run_prediction_tool(patient)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Błąd agent predict: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# AGENT CONVERSATION (conversational data collection)
# ============================================================================

# Kolekcja pytań do zebrania danych pacjenta
_COLLECTION_STEPS = [
    {
        "field": "wiek_rozpoznania",
        "question": "Witaj! Jestem asystentem medycznym systemu Vasculitis XAI.\n\nPomogę Ci ocenić ryzyko śmiertelności u pacjenta z zapaleniem naczyń. Zaczniemy od kilku pytań.\n\n**Jaki jest wiek pacjenta w momencie rozpoznania choroby?** (podaj liczbę lat)",
        "type": "number",
        "default": 50,
        "widget": "slider",
        "min": 0,
        "max": 100,
        "step": 1,
        "unit": "lat",
    },
    {
        "field": "opoznienie_rozpoznia",
        "question": "Ile miesięcy minęło od pierwszych objawów do postawienia diagnozy?\n(Jeśli nie wiesz, wpisz 0)",
        "type": "number",
        "default": 0,
        "widget": "slider",
        "min": 0,
        "max": 24,
        "step": 1,
        "unit": "mies.",
    },
    {
        "field": "manifestacja_nerki",
        "question": "Czy choroba dotknęła **nerek**? (tak/nie)",
        "type": "boolean",
        "default": 0,
        "widget": "buttons",
        "options": ["tak", "nie", "nie wiem"],
    },
    {
        "field": "manifestacja_sercowo_naczyniowy",
        "question": "Czy występują objawy ze strony **układu sercowo-naczyniowego**? (tak/nie)",
        "type": "boolean",
        "default": 0,
        "widget": "buttons",
        "options": ["tak", "nie", "nie wiem"],
    },
    {
        "field": "manifestacja_zajecie_csn",
        "question": "Czy choroba dotknęła **ośrodkowy układ nerwowy** (mózg, rdzeń)? (tak/nie)",
        "type": "boolean",
        "default": 0,
        "widget": "buttons",
        "options": ["tak", "nie", "nie wiem"],
    },
    {
        "field": "manifestacja_neurologiczny",
        "question": "Czy występują objawy **neurologiczne obwodowe** (neuropatia)? (tak/nie)",
        "type": "boolean",
        "default": 0,
        "widget": "buttons",
        "options": ["tak", "nie", "nie wiem"],
    },
    {
        "field": "liczba_zajetych_narzadow",
        "question": "Ile **narządów ogólnie** jest objętych chorobą? (podaj liczbę 1-10)",
        "type": "number",
        "default": 1,
        "widget": "slider",
        "min": 0,
        "max": 5,
        "step": 1,
        "unit": "",
    },
    {
        "field": "zaostrz_wymagajace_hospital",
        "question": "Czy wystąpiły zaostrzenia wymagające **hospitalizacji**? (tak/nie)",
        "type": "boolean",
        "default": 0,
        "widget": "buttons",
        "options": ["tak", "nie"],
    },
    {
        "field": "zaostrz_wymagajace_oit",
        "question": "Czy wystąpiły zaostrzenia wymagające pobytu na **OIT** (oddział intensywnej terapii)? (tak/nie)",
        "type": "boolean",
        "default": 0,
        "widget": "buttons",
        "options": ["tak", "nie"],
    },
    {
        "field": "kreatynina",
        "question": "Jaki jest poziom **kreatyniny** (μmol/L)?\n(Norma: 60-110. Jeśli nieznane, wpisz 0)",
        "type": "number",
        "default": 0,
        "widget": "slider",
        "min": 80,
        "max": 300,
        "step": 1,
        "unit": "μmol/L",
    },
    {
        "field": "czas_sterydow",
        "question": "Ile **miesięcy** pacjent jest leczony sterydami? (podaj liczbę lub 0)",
        "type": "number",
        "default": 0,
        "widget": "slider",
        "min": 0,
        "max": 36,
        "step": 1,
        "unit": "mies.",
    },
    {
        "field": "plazmaferezy",
        "question": "Czy pacjentowi wykonywano **plazmaferezy** (oczyszczanie krwi)? (tak/nie)",
        "type": "boolean",
        "default": 0,
        "widget": "buttons",
        "options": ["tak", "nie"],
    },
    {
        "field": "biopsja_wynik",
        "question": "Czy wynik **biopsji** był dodatni? (tak/nie/nie wiem)",
        "type": "boolean",
        "default": 0,
        "widget": "buttons",
        "options": ["tak", "nie", "nie wiem"],
    },
]

# Domyślne wartości dla pól, które nie są zbierane przez agenta
_DEFAULT_EXTRA_FIELDS = {
    "wiek": 0,
    "zap_gpa": 0,
    "manifestacja_objaw_ogol": 0,
    "manifestacja_miesno_szkiel": 0,
    "manifestacja_skora": 0,
    "manifestacja_wzrok": 0,
    "manifestacja_nos_ucho_gardlo": 0,
    "manifestacja_oddechowy": 0,
    "manifestacja_pokarmowy": 0,
    "manifestacja_moczowo_plciowy": 0,
    "eozynofilia_krwi_obwodowych_wartosc": 0,
    "pulsy": 0,
    "czas_sterydow": 0,
    "przebieg_scalony": 0,
    "powiklanie_skora": 0,
    "powiklania_hematologiczne": 0,
    "powiklania_infekcja": 0,
    "powiklania_autoimmunologiczne": 0,
    "powiklania_neurologiczne": 0,
    "powiklania_nowotwor_zlosliwy": 0,
    "powiklania_serc_pluca": 0,
}


def _parse_boolean(value: str) -> int:
    """Parsuj odpowiedź tak/nie na int 0/1."""
    v = value.strip().lower()
    if v in ('tak', 'yes', 't', 'y', '1'):
        return 1
    if v in ('nie', 'no', 'n', '0'):
        return 0
    if 'tak' in v:
        return 1
    if 'nie' in v:
        return 0
    return 0


def _parse_number(value: str) -> Optional[float]:
    """Parsuj odpowiedź liczbową."""
    import re
    match = re.search(r'[\d]+[.,]?[\d]*', value.replace(',', '.'))
    if match:
        return float(match.group().replace(',', '.'))
    return None


def _is_skip(value: str) -> bool:
    """Sprawdź czy użytkownik chce pominąć pytanie."""
    v = value.strip().lower()
    return v in ('nie wiem', 'pomin', 'skip', 'nie znam', '-', 'n/a', 'brak')


@app.post("/agent/chat", response_model=AgentConversationResponse, tags=["Agent"])
async def agent_conversation(request: AgentConversationRequest):
    """
    Konwersacyjny agent zbierający dane pacjenta i wykonujący predykcję.

    Flow:
    1. Agent zadaje pytania jedno po drugim
    2. Zbiera odpowiedzi i buduje profil pacjenta
    3. Gdy wszystkie dane zebrane → wykonuje predykcję
    4. Pokazuje wyniki z wykresami
    5. Pozwala na dalszą dyskusję o wynikach i chorobie
    """
    try:
        collected = dict(request.collected_data)
        current_step = request.current_step
        phase = request.phase
        user_message = request.message.strip()

        # ========================================
        # Phase: DISCUSSION (after prediction)
        # ========================================
        if phase == "discussion":
            message_lower = user_message.lower()

            if any(kw in message_lower for kw in ['nowy', 'nowa', 'restart', 'jeszcze raz', 'od nowa']):
                return AgentConversationResponse(
                    response="Okej, zaczynamy od nowa!\n\n" + _COLLECTION_STEPS[0]["question"],
                    collected_data={},
                    current_step=1,
                    phase="collecting",
                    missing_fields=[s["field"] for s in _COLLECTION_STEPS],
                    follow_up_suggestions=_COLLECTION_STEPS[0].get("suggestions", ["40", "55", "65", "75"]),
                )

            if any(kw in message_lower for kw in ['czynnik', 'wpływa', 'dlaczego', 'shap']):
                patient_data = dict(collected)
                patient_data.update(_DEFAULT_EXTRA_FIELDS)
                pred_data = await run_prediction_tool(PatientInput(**patient_data))
                factor_text = "Szczegółowe zestawienie czynników:\n\n"
                sorted_f = sorted(pred_data.factors, key=lambda f: abs(f.contribution), reverse=True)
                for i, f in enumerate(sorted_f[:8], 1):
                    direction = "↑ zwiększa" if f.contribution > 0 else "↓ zmniejsza"
                    factor_text += f"{i}. **{f.feature}**: {direction} ryzyko ({f.contribution:+.3f})\n"
                factor_text += "\nCzy chcesz dowiedzieć się więcej o konkretnym czynniku?"
                return AgentConversationResponse(
                    response=factor_text,
                    collected_data=collected,
                    current_step=current_step,
                    phase="discussion",
                    prediction_data=pred_data,
                    follow_up_suggestions=["Co mogę zrobić?", "Opowiedz o zapaleniu naczyń", "Nowy pacjent"],
                )

            if any(kw in message_lower for kw in ['zalec', 'co robić', 'co robic', 'pomoc', 'porad']):
                return AgentConversationResponse(
                    response="""Oto moje zalecenia:

1. **Regularne konsultacje** z lekarzem prowadzącym
2. **Przestrzeganie zaleceń** dotyczących leczenia
3. **Zgłaszanie objawów** — informuj lekarza o zmianach
4. **Badania kontrolne** — regularnie wykonuj zalecone badania

Pamiętaj, że ostateczne decyzje dotyczące leczenia podejmuje lekarz prowadzący.

Czy masz jeszcze pytania?""",
                    collected_data=collected,
                    current_step=current_step,
                    phase="discussion",
                    follow_up_suggestions=["Jakie czynniki wpływają na wynik?", "Opowiedz o zapaleniu naczyń", "Nowy pacjent"],
                )

            if any(kw in message_lower for kw in ['zapalenie naczyń', 'vasculitis', 'chorob', 'czego']):
                return AgentConversationResponse(
                    response="""**Zapalenie naczyń** (vasculitis) to grupa chorób, w których dochodzi do zapalenia ścian naczyń krwionośnych.

**Główne typy:**
- GPA (ziarniniakowatość z zapaleniem naczyń)
- MPA (mikroskopowe zapalenie naczyń)
- EGPA (eozynofilowa ziarniniakowatość, Churg-Strauss)

**Czynniki prognostyczne:**
- Wiek w momencie rozpoznania
- Zajęcie nerek, serca, OUN
- Liczba zajętych narządów
- Przebieg zaostrzeń

**Leczenie:** glikokortykosteroidy, cyklofosfamid, rytuksymab, a w ciężkich przypadkach plazmafereza.

Czy chcesz wiedzieć więcej?""",
                    collected_data=collected,
                    current_step=current_step,
                    phase="discussion",
                    follow_up_suggestions=["Jakie czynniki wpływają na wynik?", "Co mogę zrobić?", "Nowy pacjent"],
                )

            # General discussion
            return AgentConversationResponse(
                response="""Mogę pomóc Ci z:

- **Czynniki ryzyka** -- co wpływa na wynik
- **Zalecenia** -- na co zwrócić uwagę
- **Zapalenie naczyń** -- informacje o chorobie
- **Nowy pacjent** -- rozpocznij od nowa

O co chciałbyś zapytać?""",
                collected_data=collected,
                current_step=current_step,
                phase="discussion",
                follow_up_suggestions=["Jakie czynniki wpływają na wynik?", "Opowiedz o zapaleniu naczyń", "Nowy pacjent"],
            )

        # ========================================
        # Phase: COLLECTING data
        # ========================================
        if phase == "collecting":
            # Parse the user's answer for the current step
            if current_step > 0 and current_step <= len(_COLLECTION_STEPS):
                step = _COLLECTION_STEPS[current_step - 1]

                if _is_skip(user_message):
                    collected[step["field"]] = step["default"]
                elif step["type"] == "boolean":
                    collected[step["field"]] = _parse_boolean(user_message)
                elif step["type"] == "number":
                    parsed = _parse_number(user_message)
                    collected[step["field"]] = parsed if parsed is not None else step["default"]
                else:
                    collected[step["field"]] = step["default"]

            # Check if we've collected all steps
            if current_step >= len(_COLLECTION_STEPS):
                # All data collected → prediction!
                patient_data = dict(collected)
                patient_data.update(_DEFAULT_EXTRA_FIELDS)

                pred_data = await run_prediction_tool(PatientInput(**patient_data))

                risk_level_pl = {
                    "low": "Niskie",
                    "moderate": "Umiarkowane",
                    "high": "Wysokie",
                }.get(pred_data.prediction.risk_level.value, pred_data.prediction.risk_level.value)

                risk_factors = [f for f in pred_data.factors if f.contribution > 0][:5]
                protective_factors = [f for f in pred_data.factors if f.contribution < 0][:5]

                response = f"""**Zebrałem wszystkie dane! Oto wynik analizy:**

**Poziom ryzyka:** {risk_level_pl}
**Prawdopodobieństwo:** {pred_data.prediction.probability:.1%}
"""

                if risk_factors:
                    response += "\n**Czynniki zwiększające ryzyko:**\n"
                    for f in risk_factors:
                        response += f"- {f.feature}: +{f.contribution:.3f}\n"
                if protective_factors:
                    response += "\n**Czynniki zmniejszające ryzyko:**\n"
                    for f in protective_factors:
                        response += f"- {f.feature}: {f.contribution:.3f}\n"

                response += """\nPoniżej znajdziesz interaktywne wykresy. Możesz mnie teraz pytać o:\n- Czynniki wpływające na wynik\n- Zalecenia\n- Informacje o zapaleniu naczyń\n- Rozpocząć analizę nowego pacjenta"""

                return AgentConversationResponse(
                    response=response,
                    collected_data=collected,
                    current_step=current_step,
                    phase="discussion",
                    prediction_data=pred_data,
                    follow_up_suggestions=["Jakie czynniki wpływają na wynik?", "Co mogę zrobić?", "Nowy pacjent"],
                )

            # Ask the next question
            next_step_idx = current_step
            if next_step_idx < len(_COLLECTION_STEPS):
                next_step = _COLLECTION_STEPS[next_step_idx]
                progress = f"[{next_step_idx + 1}/{len(_COLLECTION_STEPS)}]"
                question = f"{progress} {next_step['question']}"

                # Build field_meta for frontend widget rendering
                field_meta = {
                    "field": next_step["field"],
                    "type": next_step["type"],
                    "widget": next_step.get("widget", "input"),
                }
                if next_step.get("min") is not None:
                    field_meta["min"] = next_step["min"]
                if next_step.get("max") is not None:
                    field_meta["max"] = next_step["max"]
                if next_step.get("step") is not None:
                    field_meta["step"] = next_step["step"]
                if next_step.get("unit"):
                    field_meta["unit"] = next_step["unit"]
                if next_step.get("options"):
                    field_meta["options"] = next_step["options"]

                return AgentConversationResponse(
                    response=question,
                    collected_data=collected,
                    current_step=next_step_idx + 1,
                    phase="collecting",
                    missing_fields=[s["field"] for s in _COLLECTION_STEPS[next_step_idx:]],
                    follow_up_suggestions=next_step.get("options", []),
                    field_meta=field_meta,
                )

        # Fallback — start over
        return AgentConversationResponse(
            response="Zaczynamy od nowa!\n\n" + _COLLECTION_STEPS[0]["question"],
            collected_data={},
            current_step=1,
            phase="collecting",
            missing_fields=[s["field"] for s in _COLLECTION_STEPS],
            follow_up_suggestions=_COLLECTION_STEPS[0].get("options", []),
            field_meta={
                "field": _COLLECTION_STEPS[0]["field"],
                "type": _COLLECTION_STEPS[0]["type"],
                "widget": _COLLECTION_STEPS[0].get("widget", "input"),
                **({"min": _COLLECTION_STEPS[0]["min"], "max": _COLLECTION_STEPS[0]["max"], "step": _COLLECTION_STEPS[0]["step"], "unit": _COLLECTION_STEPS[0]["unit"]} if _COLLECTION_STEPS[0].get("widget") == "slider" else {}),
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Błąd agent conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# CHAT (z narzędziem predykcji)
# ============================================================================

# Intencje związane z predykcją
_PREDICTION_KEYWORDS = [
    'predykc', 'ryzyko', 'śmiertelno', 'smiertelno', 'prawdopodob',
    'wynik', 'analiz', 'ocen', 'jakie szanse', 'jakie ryzyko',
    'prognos', 'zgon', 'ryzyk', 'mortality', 'risk',
    'jakie jest ryzyko', 'oblicz', 'policz', 'sprawdź', 'sprawdz',
]

_FACTOR_KEYWORDS = [
    'czynnik', 'wpływa', 'dlaczego', 'powód', 'powod', 'przyczyn',
    'co zwiększa', 'co zmniejsza', 'shap', 'ważno', 'wplyw',
]

_RECOMMENDATION_KEYWORDS = [
    'pomoc', 'co robić', 'co robic', 'zalec', 'rekomend', 'sugesti',
    'porad', 'co mogę', 'co powinien', 'co powinienem',
]


def _build_prediction_response(pred_result: PredictionOutput, factor_data: list) -> str:
    """Zbuduj tekstową odpowiedź agenta z wynikami predykcji."""
    risk_level_pl = {
        "low": "Niskie",
        "moderate": "Umiarkowane",
        "high": "Wysokie",
    }.get(pred_result.risk_level.value, pred_result.risk_level.value)

    response = f"""Na podstawie wprowadzonych danych przeprowadziłem analizę ryzyka śmiertelności:

**Poziom ryzyka:** {risk_level_pl}
**Prawdopodobieństwo:** {pred_result.probability:.1%}
"""

    if factor_data:
        risk_factors = [f for f in factor_data if f.contribution > 0]
        protective_factors = [f for f in factor_data if f.contribution < 0]

        if risk_factors:
            response += "\n**Czynniki zwiększające ryzyko:**\n"
            for f in risk_factors[:5]:
                response += f"- {f.feature}: +{f.contribution:.3f}\n"

        if protective_factors:
            response += "\n**Czynniki zmniejszające ryzyko:**\n"
            for f in protective_factors[:5]:
                response += f"- {f.feature}: {f.contribution:.3f}\n"

    response += """\nPoniżej znajdziesz interaktywne wykresy z szczegółową analizą.\n\nCzy chciałbyś/chciałabyś dowiedzieć się więcej o konkretnych czynnikach?"""

    return response


def _build_factor_response(factor_data: list) -> str:
    """Zbuduj odpowiedź o czynnikach ryzyka."""
    if not factor_data:
        return "Brak danych o czynnikach ryzyka. Najpierw przeprowadź analizę pacjenta."

    sorted_factors = sorted(factor_data, key=lambda f: abs(f.contribution), reverse=True)

    response = "Oto szczegółowe zestawienie czynników wpływających na ocenę ryzyka:\n\n"

    for i, f in enumerate(sorted_factors[:8], 1):
        direction = "↑ zwiększa" if f.contribution > 0 else "↓ zmniejsza"
        response += f"{i}. **{f.feature}**: {direction} ryzyko (wpływ: {f.contribution:+.3f})\n"

    response += "\nCzy chcesz dowiedzieć się więcej o którymś z tych czynników?"
    return response


def _build_recommendation_response() -> str:
    """Zbuduj odpowiedź z zaleceniami."""
    return """Oto moje zalecenia:

1. **Regularne konsultacje** z lekarzem prowadzącym
2. **Przestrzeganie zaleceń** dotyczących leczenia
3. **Zgłaszanie objawów** — informuj lekarza o wszelkich niepokojących zmianach
4. **Monitorowanie** regularnie wykonuj zalecone badania kontrolne

Pamiętaj, że ta analiza jest narzędziem wspierającym — ostateczne decyzje
dotyczące leczenia powinny być podejmowane wspólnie z lekarzem.

Czy masz dodatkowe pytania?"""


def _build_general_response() -> str:
    """Zbuduj ogólną odpowiedź powitalną."""
    return """Jestem asystentem pomagającym zrozumieć wyniki analizy ryzyka śmiertelności w zapaleniu naczyń.

Mogę pomóc Ci z:
- **Predykcją ryzyka** — obliczę prawdopodobieństwo na podstawie danych pacjenta
- **Wyjaśnieniem czynników** — powiem co wpływa na wynik i dlaczego
- **Zaleceniami** — podpowiem na co zwrócić uwagę

Aby przeprowadzić analizę, potrzebuję danych pacjenta. Wypełnij formularz
po lewej stronie, a następnie zapytaj mnie o wynik.

O czym chciałbyś/chciałabyś porozmawiać?"""


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Rozmowa z agentem AI.

    Odpowiada na pytania pacjenta/klinicysty o wyniki analizy.
    Może wywołać narzędzie predykcji i zwrócić dane do wyświetlenia wykresów.
    """
    try:
        message = request.message.lower()
        prediction_data: Optional[AgentPredictionData] = None

        # Wykryj intencję — predykcja
        is_prediction_intent = any(kw in message for kw in _PREDICTION_KEYWORDS)
        is_factor_intent = any(kw in message for kw in _FACTOR_KEYWORDS)
        is_recommendation_intent = any(kw in message for kw in _RECOMMENDATION_KEYWORDS)

        if is_prediction_intent:
            # Użyj narzędzia predykcji
            pred_result = await run_prediction_tool(request.patient)
            prediction_data = pred_result
            response = _build_prediction_response(pred_result.prediction, pred_result.factors)

        elif is_factor_intent:
            # Najpierw pobierz predykcję, żeby mieć czynniki
            pred_result = await run_prediction_tool(request.patient)
            prediction_data = pred_result
            response = _build_factor_response(pred_result.factors)

        elif is_recommendation_intent:
            response = _build_recommendation_response()

        else:
            response = _build_general_response()

        # Dodaj disclaimer dla pacjentów
        if request.health_literacy != HealthLiteracyLevel.CLINICIAN:
            response += "\n\n---\n*Pamiętaj: to narzędzie informacyjne, nie zastępuje porady lekarza.*"

        return ChatResponse(
            response=response,
            detected_concerns=None,
            follow_up_suggestions=[
                "Jakie czynniki wpływają na wynik?",
                "Co mogę zrobić, aby poprawić swoje zdrowie?",
                "Czy powinienem/powinnam martwić się wynikiem?",
            ],
            prediction_data=prediction_data,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Błąd chatu: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Nieobsłużony błąd: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
            code=500
        ).dict()
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
        reload=True
    )


if __name__ == "__main__":
    run_server()
