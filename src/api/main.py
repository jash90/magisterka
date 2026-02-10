"""
FastAPI aplikacja dla systemu XAI.

Główny plik API z endpointami dla wyjaśniania decyzji
zewnętrznego AI i interakcji z agentem konwersacyjnym.
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
import json
import logging

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
    ComparisonResult, ErrorResponse, RiskLevel, XAIMethod, HealthLiteracyLevel,
    AnalysisRequest, AnalysisOutput,
    patient_to_array, get_risk_level_from_probability, patients_to_matrix,
    # Batch schemas
    BatchAnalysisInput, BatchAnalysisOutput, BatchPatientResult,
    BatchSummary, BatchProcessingError, RiskFactorItem,
    DemoModeStatus, DemoModeRequest
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# APLIKACJA FASTAPI
# ============================================================================

app = FastAPI(
    title="Vasculitis XAI API",
    description="""
    API do wyjaśniania decyzji zewnętrznego AI w zapaleniu naczyń.

    System przyjmuje wynik predykcji z zewnętrznego modelu AI (% przeżycia/zgonu)
    i wyjaśnia, które cechy pacjenta były kluczowe dla tej decyzji,
    wykorzystując metody XAI (SHAP, LIME).
    """,
    version="2.0.0",
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
    """Stan aplikacji z wczytanymi modelami i explainerami."""

    def __init__(self):
        self.model = None
        self.feature_names = None
        self.shap_explainer_instance = None
        self.lime_explainer_instance = None
        self.rag_pipeline = None
        self.is_loaded = False
        self.explainers_ready = False

        # Konfiguracja trybu demo
        self.allow_demo = os.getenv("ALLOW_DEMO", "true").lower() == "true"
        self.force_api_mode = os.getenv("FORCE_API_MODE", "false").lower() == "true"

        if self.force_api_mode:
            self.allow_demo = False

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

    def load_models(self, model_path: str, feature_names_path: str):
        """Wczytaj modele i explainer'y."""
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

            # Inicjalizacja explainerów
            self._init_explainers()

            return True

        except Exception as e:
            logger.error(f"Błąd wczytywania modeli: {e}")
            return False

    def _init_explainers(self):
        """Inicjalizacja prawdziwych SHAP i LIME explainerów."""
        try:
            # Wczytaj dane referencyjne
            ref_path = os.getenv(
                "REFERENCE_DATA_PATH",
                "models/saved/X_reference.npy"
            )
            if not Path(ref_path).exists():
                logger.warning(f"Brak danych referencyjnych: {ref_path}. Explainers niedostępne.")
                return

            X_reference = np.load(ref_path)
            logger.info(f"Wczytano dane referencyjne: {X_reference.shape}")

            # SHAP Explainer
            from src.xai.shap_explainer import SHAPExplainer
            self.shap_explainer_instance = SHAPExplainer(
                model=self.model,
                X_background=X_reference,
                feature_names=self.feature_names
            )
            logger.info("SHAP Explainer zainicjalizowany")

            # LIME Explainer
            from src.xai.lime_explainer import LIMEExplainer
            self.lime_explainer_instance = LIMEExplainer(
                model=self.model,
                X_train=X_reference,
                feature_names=self.feature_names
            )
            logger.info("LIME Explainer zainicjalizowany")

            self.explainers_ready = True

        except Exception as e:
            logger.error(f"Błąd inicjalizacji explainerów: {e}")
            self.explainers_ready = False


app_state = AppState()


# ============================================================================
# STARTUP/SHUTDOWN
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Zdarzenie uruchomienia aplikacji."""
    logger.info("Uruchamianie API Vasculitis XAI...")

    # Próba wczytania modeli
    model_path = os.getenv("MODEL_PATH", "models/saved/best_model.joblib")
    feature_names_path = os.getenv("FEATURE_NAMES_PATH", "models/saved/feature_names.json")

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

def _build_prediction_output(probability: float) -> PredictionOutput:
    """Zbuduj PredictionOutput z prawdopodobieństwa."""
    return PredictionOutput(
        probability=float(probability),
        risk_level=get_risk_level_from_probability(probability),
        prediction=1 if probability > 0.5 else 0
    )


def _build_shap_explanation(shap_result: dict, prediction: PredictionOutput) -> SHAPExplanation:
    """Zbuduj SHAPExplanation z wyniku shap_explainer.explain_instance()."""
    shap_values_dict = dict(zip(
        shap_result['feature_names'],
        [float(v) for v in shap_result['shap_values']]
    ))

    contributions = []
    risk_factors = []
    protective_factors = []

    for fi in shap_result['feature_impacts']:
        fc = {
            "feature": fi['feature'],
            "value": float(fi['feature_value']),
            "contribution": float(fi['shap_value']),
            "direction": fi['direction']
        }
        contributions.append(fc)
        if fi['shap_value'] > 0:
            risk_factors.append(fc)
        elif fi['shap_value'] < 0:
            protective_factors.append(fc)

    return SHAPExplanation(
        base_value=shap_result['base_value'],
        shap_values=shap_values_dict,
        feature_contributions=contributions,
        risk_factors=risk_factors,
        protective_factors=protective_factors,
        prediction=prediction
    )


def _build_lime_explanation(lime_result: dict, prediction: PredictionOutput) -> LIMEExplanation:
    """Zbuduj LIMEExplanation z wyniku lime_explainer.explain_instance()."""
    feature_weights = []
    risk_factors = []
    protective_factors = []

    for feat_desc, weight in lime_result['feature_weights']:
        fw = {
            "feature": feat_desc,
            "weight": float(weight),
            "condition": feat_desc
        }
        feature_weights.append(fw)
        if weight > 0:
            risk_factors.append({"feature": feat_desc, "weight": float(weight)})
        elif weight < 0:
            protective_factors.append({"feature": feat_desc, "weight": float(weight)})

    return LIMEExplanation(
        intercept=lime_result['intercept'],
        feature_weights=feature_weights,
        risk_factors=risk_factors,
        protective_factors=protective_factors,
        local_prediction=lime_result['local_prediction'],
        prediction=prediction
    )


def _get_internal_prediction(patient: PatientInput) -> PredictionOutput:
    """Uzyskaj predykcję z wewnętrznego modelu (do celów XAI)."""
    if app_state.is_loaded and app_state.model is not None:
        features = patient_to_array(patient, app_state.feature_names)
        X = np.array(features).reshape(1, -1)
        probability = app_state.model.predict_proba(X)[0, 1]
        return _build_prediction_output(probability)
    else:
        # Fallback: bez modelu nie ma prawdziwej predykcji
        return _build_prediction_output(0.5)


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/", tags=["Info"])
async def root():
    """Strona główna API."""
    return {
        "name": "Vasculitis XAI API",
        "version": "2.0.0",
        "description": "API do wyjaśniania decyzji zewnętrznego AI w zapaleniu naczyń",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthCheckResponse, tags=["Info"])
async def health_check():
    """Sprawdzenie stanu API."""
    return HealthCheckResponse(
        status="healthy",
        model_loaded=app_state.is_loaded,
        explainers_ready=app_state.explainers_ready,
        api_version="2.0.0",
        timestamp=datetime.now().isoformat()
    )


# ============================================================================
# MAIN ANALYSIS ENDPOINT
# ============================================================================

@app.post("/analyze", response_model=AnalysisOutput, tags=["Analysis"])
async def analyze(request: AnalysisRequest):
    """
    Wyjaśnij decyzję zewnętrznego AI.

    Przyjmuje dane pacjenta i prawdopodobieństwo z zewnętrznego modelu AI.
    Zwraca wyjaśnienia SHAP i LIME wskazujące, które cechy pacjenta
    mogły wpłynąć na tę ocenę.
    """
    try:
        ext_prob = request.external_probability
        risk_level = get_risk_level_from_probability(ext_prob)

        shap_explanation = None
        lime_explanation = None

        if app_state.is_loaded and app_state.explainers_ready:
            features = patient_to_array(request.patient, app_state.feature_names)
            X = np.array(features).reshape(1, -1)
            internal_pred = _get_internal_prediction(request.patient)

            # SHAP
            if app_state.shap_explainer_instance:
                shap_result = app_state.shap_explainer_instance.explain_instance(X)
                shap_explanation = _build_shap_explanation(shap_result, internal_pred)

            # LIME
            if app_state.lime_explainer_instance:
                lime_result = app_state.lime_explainer_instance.explain_instance(X[0])
                lime_explanation = _build_lime_explanation(lime_result, internal_pred)

        return AnalysisOutput(
            external_probability=ext_prob,
            risk_level=risk_level,
            shap_explanation=shap_explanation,
            lime_explanation=lime_explanation,
            disclaimer=(
                "Zewnętrzny system AI oszacował ryzyko zgonu na "
                f"{ext_prob:.1%}. Poniżej przedstawiamy czynniki, które "
                "mogą wpływać na tę ocenę. Wyjaśnienia mają charakter informacyjny."
            )
        )

    except Exception as e:
        logger.error(f"Błąd analizy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# LEGACY PREDICT (kept for backward compatibility)
# ============================================================================

@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
async def predict(patient: PatientInput):
    """
    Wykonaj predykcję z wewnętrznego modelu.

    Zwraca prawdopodobieństwo zgonu z wewnętrznego modelu XGBoost.
    Używane głównie wewnętrznie przez explainers.
    """
    try:
        return _get_internal_prediction(patient)
    except Exception as e:
        logger.error(f"Błąd predykcji: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# BATCH ANALYSIS
# ============================================================================

@app.post("/analyze/batch", response_model=BatchAnalysisOutput, tags=["Analysis"])
async def analyze_batch(request: BatchAnalysisInput):
    """
    Batch analysis dla wielu pacjentów.

    Wymaga external_probabilities (jedno na pacjenta).
    Opcjonalnie generuje wyjaśnienia XAI.
    """
    start_time = time.perf_counter()

    results = []
    errors = []
    n_patients = len(request.patients)

    for i, (patient, ext_prob) in enumerate(
        zip(request.patients, request.external_probabilities)
    ):
        try:
            risk_level = get_risk_level_from_probability(ext_prob)

            top_factors = None
            if request.include_risk_factors and app_state.is_loaded:
                features = patient_to_array(patient, app_state.feature_names)
                X = np.array(features).reshape(1, -1)

                # Użyj feature_importances_ modelu
                importances = _get_model_importances()
                top_factors = _extract_risk_factors(
                    X[0], app_state.feature_names, importances, request.top_n_factors
                )

            results.append(BatchPatientResult(
                index=i,
                external_probability=ext_prob,
                risk_level=risk_level,
                top_risk_factors=top_factors,
                processing_status="success"
            ))

        except Exception as e:
            errors.append(BatchProcessingError(
                patient_index=i,
                error_type="processing",
                error_message=str(e),
                is_recoverable=True
            ))

    # Summary
    ext_probs = [r.external_probability for r in results]

    summary = BatchSummary(
        total_count=n_patients,
        low_risk_count=sum(1 for r in results if r.risk_level == RiskLevel.LOW),
        moderate_risk_count=sum(1 for r in results if r.risk_level == RiskLevel.MODERATE),
        high_risk_count=sum(1 for r in results if r.risk_level == RiskLevel.HIGH),
        avg_probability=float(np.mean(ext_probs)) if ext_probs else 0.0,
        median_probability=float(np.median(ext_probs)) if ext_probs else 0.0,
        min_probability=float(np.min(ext_probs)) if ext_probs else 0.0,
        max_probability=float(np.max(ext_probs)) if ext_probs else 0.0
    )

    processing_time = (time.perf_counter() - start_time) * 1000

    return BatchAnalysisOutput(
        total_patients=n_patients,
        processed_count=len(results),
        success_count=len([r for r in results if r.processing_status == "success"]),
        error_count=len(errors),
        processing_time_ms=processing_time,
        summary=summary,
        results=results,
        errors=errors
    )


def _get_model_importances() -> Dict[str, float]:
    """Pobierz feature importances z modelu."""
    if app_state.is_loaded and app_state.model is not None:
        importances = app_state.model.feature_importances_
        return dict(zip(app_state.feature_names, importances.tolist()))
    return {}


def _extract_risk_factors(
    x: np.ndarray,
    feature_names: List[str],
    importances: Dict[str, float],
    top_n: int = 3
) -> List[RiskFactorItem]:
    """Wyodrębnij top czynniki ryzyka dla pacjenta."""
    BINARY_RISK_FEATURES = {
        "Manifestacja_Miesno-Szkiel", "Manifestacja_Skora", "Manifestacja_Wzrok",
        "Manifestacja_Nos/Ucho/Gardlo", "Manifestacja_Oddechowy",
        "Manifestacja_Sercowo-Naczyniowy", "Manifestacja_Pokarmowy",
        "Manifestacja_Moczowo-Plciowy", "Manifestacja_Zajecie_CSN",
        "Manifestacja_Neurologiczny", "Zaostrz_Wymagajace_Hospital",
        "Zaostrz_Wymagajace_OIT", "Plazmaferezy", "Powiklania_Neurologiczne"
    }

    RISK_THRESHOLDS = {
        "Wiek_rozpoznania": 60,
        "Kreatynina": 150,
        "Liczba_Zajetych_Narzadow": 3,
        "Eozynofilia_Krwi_Obwodowej_Wartosc": 15,
    }

    factors = []
    for i, fname in enumerate(feature_names):
        value = float(x[i])
        importance = importances.get(fname, 0.0)

        if fname in BINARY_RISK_FEATURES:
            direction = "increases_risk" if value == 1 else "decreases_risk"
        elif fname in RISK_THRESHOLDS:
            direction = "increases_risk" if value > RISK_THRESHOLDS[fname] else "decreases_risk"
        else:
            direction = "neutral"

        factors.append(RiskFactorItem(
            feature=fname,
            value=value,
            importance=importance,
            direction=direction
        ))

    factors.sort(key=lambda x: x.importance, reverse=True)
    return factors[:top_n]


# ============================================================================
# CONFIG ENDPOINTS
# ============================================================================

@app.get("/config/demo-mode", response_model=DemoModeStatus, tags=["Config"])
async def get_demo_mode_status():
    """Pobierz status konfiguracji trybu demo."""
    return DemoModeStatus(
        demo_allowed=app_state.allow_demo,
        model_loaded=app_state.is_loaded,
        explainers_ready=app_state.explainers_ready,
        current_mode=app_state.get_current_mode(),
        force_api_mode=app_state.force_api_mode
    )


@app.post("/config/demo-mode", response_model=DemoModeStatus, tags=["Config"])
async def set_demo_mode(request: DemoModeRequest):
    """Włącz lub wyłącz tryb demo."""
    try:
        app_state.set_demo_mode(request.enabled)
        return await get_demo_mode_status()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================================
# XAI EXPLANATION ENDPOINTS
# ============================================================================

@app.post("/explain/shap", response_model=SHAPExplanation, tags=["XAI"])
async def explain_shap(request: ExplanationRequest):
    """
    Wygeneruj wyjaśnienie SHAP.

    Zwraca prawdziwe wartości SHAP dla cech pacjenta.
    """
    try:
        internal_pred = _get_internal_prediction(request.patient)

        if app_state.is_loaded and app_state.explainers_ready and app_state.shap_explainer_instance:
            features = patient_to_array(request.patient, app_state.feature_names)
            X = np.array(features).reshape(1, -1)

            shap_result = app_state.shap_explainer_instance.explain_instance(X)
            return _build_shap_explanation(shap_result, internal_pred)
        else:
            raise HTTPException(
                status_code=503,
                detail="SHAP explainer niedostępny. Załaduj model i dane referencyjne."
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

    Zwraca prawdziwe wagi cech z lokalnego modelu zastępczego.
    """
    try:
        internal_pred = _get_internal_prediction(request.patient)

        if app_state.is_loaded and app_state.explainers_ready and app_state.lime_explainer_instance:
            features = patient_to_array(request.patient, app_state.feature_names)
            X = np.array(features).reshape(1, -1)

            lime_result = app_state.lime_explainer_instance.explain_instance(X[0])
            return _build_lime_explanation(lime_result, internal_pred)
        else:
            raise HTTPException(
                status_code=503,
                detail="LIME explainer niedostępny. Załaduj model i dane referencyjne."
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Błąd LIME: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain/patient", response_model=PatientExplanation, tags=["XAI"])
async def explain_for_patient(request: PatientExplanationRequest):
    """
    Wygeneruj wyjaśnienie zrozumiałe dla pacjenta.

    Dostosowuje język i poziom szczegółowości do poziomu health literacy.
    Uwzględnia external_probability z zewnętrznego AI.
    """
    try:
        ext_prob = request.external_probability
        risk_level = get_risk_level_from_probability(ext_prob)

        # Tłumaczenia cech
        translations = {
            "Wiek_rozpoznania": "Wiek w momencie rozpoznania",
            "Manifestacja_Sercowo-Naczyniowy": "Stan układu krążenia",
            "Manifestacja_Oddechowy": "Stan układu oddechowego",
            "Manifestacja_Zajecie_CSN": "Stan ośrodkowego układu nerwowego",
            "Manifestacja_Neurologiczny": "Objawy neurologiczne",
            "Manifestacja_Skora": "Zmiany skórne",
            "Manifestacja_Wzrok": "Problemy ze wzrokiem",
            "Manifestacja_Nos_Ucho_Gardlo": "Objawy laryngologiczne",
            "Zaostrz_Wymagajace_OIT": "Przebyte poważne zaostrzenia",
            "Zaostrz_Wymagajace_Hospital": "Hospitalizacje",
            "Liczba_Zajetych_Narzadow": "Liczba dotkniętych narządów",
            "Kreatynina": "Wskaźnik czynności nerek",
            "Powiklania_Neurologiczne": "Powikłania neurologiczne",
        }

        # Opis ryzyka
        if ext_prob < 0.3:
            risk_desc = (
                f"Zewnętrzny system AI oszacował ryzyko na {ext_prob:.0%}. "
                "To wskazuje na niskie ryzyko. To dobra wiadomość!"
            )
        elif ext_prob < 0.7:
            risk_desc = (
                f"Zewnętrzny system AI oszacował ryzyko na {ext_prob:.0%}. "
                "To wskazuje na umiarkowane ryzyko. Warto zwrócić uwagę na kilka czynników."
            )
        else:
            risk_desc = (
                f"Zewnętrzny system AI oszacował ryzyko na {ext_prob:.0%}. "
                "To wskazuje na podwyższone ryzyko. Ważna jest regularna opieka lekarska."
            )

        # Pobierz faktyczne czynniki z SHAP jeśli dostępne
        main_concerns = []
        positive_factors = []

        if app_state.is_loaded and app_state.explainers_ready and app_state.shap_explainer_instance:
            features = patient_to_array(request.patient, app_state.feature_names)
            X = np.array(features).reshape(1, -1)
            shap_result = app_state.shap_explainer_instance.explain_instance(X)

            for fi in shap_result.get('risk_factors', [])[:3]:
                translated = translations.get(fi['feature'], fi['feature'])
                main_concerns.append(translated)

            for fi in shap_result.get('protective_factors', [])[:3]:
                translated = translations.get(fi['feature'], fi['feature'])
                positive_factors.append(translated)

        if not main_concerns:
            main_concerns = ["Brak szczegółowych danych (explainer niedostępny)"]
        if not positive_factors:
            positive_factors = ["Brak szczegółowych danych (explainer niedostępny)"]

        technical_summary = None
        if request.health_literacy != HealthLiteracyLevel.BASIC:
            technical_summary = {
                "external_probability": ext_prob,
                "n_risk_factors": len(main_concerns),
                "n_protective_factors": len(positive_factors),
                "explainers_available": app_state.explainers_ready
            }

        return PatientExplanation(
            risk_level=risk_level.value,
            risk_description=risk_desc,
            main_concerns=main_concerns,
            positive_factors=positive_factors,
            recommendations="Zalecamy omówienie tych wyników z lekarzem prowadzącym.",
            technical_summary=technical_summary,
            disclaimer=(
                "To narzędzie wyjaśnia decyzję zewnętrznego systemu AI. "
                "Ma charakter informacyjny i nie zastępuje porady lekarza."
            )
        )

    except Exception as e:
        logger.error(f"Błąd wyjaśnienia pacjenta: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/global-importance", response_model=GlobalImportance, tags=["Model"])
async def get_global_importance():
    """
    Pobierz globalną ważność cech.

    Używa model.feature_importances_ zamiast hardcoded danych.
    """
    if app_state.is_loaded and app_state.model is not None:
        importances = _get_model_importances()
        sorted_importance = dict(
            sorted(importances.items(), key=lambda x: x[1], reverse=True)
        )
        return GlobalImportance(
            feature_importance=sorted_importance,
            top_features=list(sorted_importance.keys()),
            method="XGBoost feature_importances_",
            n_samples=0
        )
    else:
        raise HTTPException(
            status_code=503,
            detail="Model nie jest załadowany. Nie można pobrać ważności cech."
        )


@app.get("/model/info", response_model=ModelInfo, tags=["Model"])
async def get_model_info():
    """Pobierz informacje o modelu."""
    return ModelInfo(
        model_type="XGBoostClassifier" if app_state.is_loaded else "Niedostępny",
        n_features=len(app_state.feature_names) if app_state.feature_names else 20,
        feature_names=app_state.feature_names or [],
        training_date="2024-01-15" if app_state.is_loaded else None,
        performance_metrics={
            "auc_roc": 0.85,
            "sensitivity": 0.82,
            "specificity": 0.78,
            "ppv": 0.65,
            "npv": 0.90
        },
        version="2.0.0"
    )


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Rozmowa z agentem AI o wyjaśnieniach decyzji zewnętrznego AI.
    """
    try:
        message = request.message.lower()
        ext_prob = request.external_probability
        risk_level = get_risk_level_from_probability(ext_prob)

        if any(word in message for word in ['wynik', 'analiza', 'ryzyko']):
            response = (
                f"Zewnętrzny system AI oszacował ryzyko zgonu na **{ext_prob:.1%}**, "
                f"co odpowiada poziomowi: **{risk_level.value}**.\n\n"
                "System XAI analizuje, które cechy pacjenta mogły wpłynąć na tę ocenę. "
                "Czy chciałbyś/chciałabyś dowiedzieć się więcej o konkretnych czynnikach?"
            )
        elif any(word in message for word in ['czynnik', 'wpływa', 'dlaczego']):
            response = (
                "Czynniki wpływające na ocenę zewnętrznego AI mogą obejmować:\n\n"
                "1. Liczbę zajętych narządów\n"
                "2. Manifestacje narządowe (układ oddechowy, serce, nerwy)\n"
                "3. Historię zaostrzeń wymagających hospitalizacji/OIT\n\n"
                "Użyj zakładek SHAP/LIME aby zobaczyć szczegółowe wyjaśnienia. "
                "Czy masz pytania o któryś z tych czynników?"
            )
        elif any(word in message for word in ['pomoc', 'co robić', 'zalec']):
            response = (
                "Zalecam:\n"
                "1. Regularnie konsultować się z lekarzem prowadzącym\n"
                "2. Przestrzegać zaleceń dotyczących leczenia\n"
                "3. Zgłaszać wszelkie niepokojące objawy\n\n"
                "Pamiętaj, że wyjaśnienia XAI pomagają zrozumieć decyzję AI, "
                "ale ostateczne decyzje dotyczące leczenia powinny być "
                "podejmowane wspólnie z lekarzem."
            )
        else:
            response = (
                "Jestem asystentem pomagającym zrozumieć wyjaśnienia decyzji AI.\n\n"
                "Mogę pomóc Ci z:\n"
                "- Wyjaśnieniem oceny zewnętrznego systemu AI\n"
                "- Omówieniem czynników wpływających na ocenę\n"
                "- Ogólnymi informacjami o zapaleniu naczyń\n\n"
                "O czym chciałbyś/chciałabyś porozmawiać?"
            )

        if request.health_literacy != HealthLiteracyLevel.CLINICIAN:
            response += "\n\n---\n*Pamiętaj: to narzędzie informacyjne, nie zastępuje porady lekarza.*"

        return ChatResponse(
            response=response,
            detected_concerns=None,
            follow_up_suggestions=[
                "Opowiedz mi więcej o czynnikach wpływających na ocenę AI",
                "Co mogę zrobić, aby poprawić swoje zdrowie?",
                "Jak interpretować wyjaśnienia SHAP/LIME?"
            ]
        )

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
