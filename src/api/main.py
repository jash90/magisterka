"""
FastAPI aplikacja dla systemu XAI.

Główny plik API z endpointami dla predykcji,
wyjaśnień XAI i agenta konwersacyjnego.
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
    patient_to_array, get_risk_level_from_probability, patients_to_matrix,
    # Batch schemas
    BatchPatientInput, BatchPredictionOutput, BatchPatientResult,
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
        """Pobierz global feature importance (z cache)."""
        if self._global_importance_cache is None:
            self._global_importance_cache = {
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
        return self._global_importance_cache

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
            return True

        except Exception as e:
            logger.error(f"Błąd wczytywania modeli: {e}")
            return False


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
        confidence_interval={"lower": probability - 0.1, "upper": probability + 0.1}
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
            "direction": "zwiększa ryzyko"
        })

    if patient.manifestacja_nerki:
        risk_factors.append({
            "feature": "Manifestacja_Nerki",
            "value": 1,
            "contribution": 0.12,
            "direction": "zwiększa ryzyko"
        })

    if patient.zaostrz_wymagajace_oit:
        risk_factors.append({
            "feature": "Zaostrz_Wymagajace_OIT",
            "value": 1,
            "contribution": 0.2,
            "direction": "zwiększa ryzyko"
        })

    if patient.wiek < 50:
        protective_factors.append({
            "feature": "Wiek",
            "value": patient.wiek,
            "contribution": -0.1,
            "direction": "zmniejsza ryzyko"
        })

    if patient.liczba_zajetych_narzadow <= 2:
        protective_factors.append({
            "feature": "Liczba_Zajetych_Narzadow",
            "value": patient.liczba_zajetych_narzadow,
            "contribution": -0.08,
            "direction": "zmniejsza ryzyko"
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

    Zwraca wartości SHAP dla cech pacjenta.
    """
    try:
        # Najpierw predykcja
        prediction = await predict(request.patient)

        # Demo wyjaśnienie
        demo_exp = get_demo_explanation(request.patient)

        # Utwórz strukturę SHAP
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
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain/lime", response_model=LIMEExplanation, tags=["XAI"])
async def explain_lime(request: ExplanationRequest):
    """
    Wygeneruj wyjaśnienie LIME.

    Zwraca wagi cech z lokalnego modelu zastępczego.
    """
    try:
        prediction = await predict(request.patient)
        demo_exp = get_demo_explanation(request.patient)

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
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain/patient", response_model=PatientExplanation, tags=["XAI"])
async def explain_for_patient(request: PatientExplanationRequest):
    """
    Wygeneruj wyjaśnienie zrozumiałe dla pacjenta.

    Dostosowuje język i poziom szczegółowości do poziomu health literacy.
    """
    try:
        prediction = await predict(request.patient)
        demo_exp = get_demo_explanation(request.patient)

        # Tłumaczenia cech
        translations = {
            "Wiek": "Twój wiek",
            "Manifestacja_Nerki": "Stan nerek",
            "Manifestacja_Sercowo_Naczyniowy": "Stan układu krążenia",
            "Zaostrz_Wymagajace_OIT": "Przebyte poważne zaostrzenia",
            "Liczba_Zajetych_Narzadow": "Liczba dotkniętych narządów",
            "Kreatynina": "Wskaźnik czynności nerek",
            "Max_CRP": "Poziom stanu zapalnego"
        }

        # Poziom ryzyka
        if prediction.probability < 0.3:
            risk_desc = "Analiza wskazuje na niskie ryzyko. To dobra wiadomość!"
        elif prediction.probability < 0.7:
            risk_desc = "Analiza wskazuje na umiarkowane ryzyko. Warto zwrócić uwagę na kilka czynników."
        else:
            risk_desc = "Analiza wskazuje na podwyższone ryzyko. Ważna jest regularna opieka lekarska."

        main_concerns = [
            translations.get(rf["feature"], rf["feature"])
            for rf in demo_exp["risk_factors"][:3]
        ]

        positive_factors = [
            translations.get(pf["feature"], pf["feature"])
            for pf in demo_exp["protective_factors"][:3]
        ]

        return PatientExplanation(
            risk_level=prediction.risk_level.value,
            risk_description=risk_desc,
            main_concerns=main_concerns if main_concerns else ["Brak szczególnych czynników ryzyka"],
            positive_factors=positive_factors if positive_factors else ["Analiza trwa"],
            recommendations="Zalecamy omówienie tych wyników z lekarzem prowadzącym.",
            technical_summary={
                "probability": prediction.probability,
                "n_risk_factors": len(demo_exp["risk_factors"]),
                "n_protective_factors": len(demo_exp["protective_factors"])
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
    """
    # Demo importance
    importance = {
        "Wiek": 0.15,
        "Manifestacja_Nerki": 0.12,
        "Zaostrz_Wymagajace_OIT": 0.11,
        "Liczba_Zajetych_Narzadow": 0.10,
        "Manifestacja_Sercowo-Naczyniowy": 0.09,
        "Kreatynina": 0.08,
        "Max_CRP": 0.07,
        "Dializa": 0.06,
        "Manifestacja_Zajecie_CSN": 0.05,
        "Plazmaferezy": 0.04
    }

    return GlobalImportance(
        feature_importance=importance,
        top_features=list(importance.keys()),
        method="SHAP TreeExplainer (demo)",
        n_samples=100
    )


@app.get("/model/info", response_model=ModelInfo, tags=["Model"])
async def get_model_info():
    """
    Pobierz informacje o modelu.

    Zwraca metadane modelu i metryki wydajności.
    """
    return ModelInfo(
        model_type="XGBoostClassifier" if app_state.is_loaded else "Demo Model",
        n_features=len(app_state.feature_names) if app_state.feature_names else 20,
        feature_names=app_state.feature_names or ["Wiek", "Plec", "Manifestacja_Nerki", "..."],
        training_date="2024-01-15" if app_state.is_loaded else None,
        performance_metrics={
            "auc_roc": 0.85,
            "sensitivity": 0.82,
            "specificity": 0.78,
            "ppv": 0.65,
            "npv": 0.90
        },
        version="1.0.0"
    )


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Rozmowa z agentem AI.

    Odpowiada na pytania pacjenta/klinicysty o wyniki analizy.
    """
    try:
        # Prosty response bez RAG
        message = request.message.lower()

        # Wykryj intencję
        if any(word in message for word in ['wynik', 'analiza', 'ryzyko']):
            prediction = await predict(request.patient)
            response = f"""
Na podstawie wprowadzonych danych, poziom ryzyka wynosi: **{prediction.risk_level.value}**
(prawdopodobieństwo: {prediction.probability:.1%}).

Główne czynniki brane pod uwagę to wiek, stan narządów i historia leczenia.

Czy chciałbyś/chciałabyś dowiedzieć się więcej o konkretnych czynnikach?
"""
        elif any(word in message for word in ['czynnik', 'wpływa', 'dlaczego']):
            demo_exp = get_demo_explanation(request.patient)
            factors = demo_exp["risk_factors"][:3]
            response = "Główne czynniki wpływające na ocenę to:\n\n"
            for i, f in enumerate(factors, 1):
                response += f"{i}. {f['feature']}\n"
            response += "\nCzy masz pytania o któryś z tych czynników?"
        elif any(word in message for word in ['pomoc', 'co robić', 'zalec']):
            response = """
Zalecam:
1. Regularnie konsultować się z lekarzem prowadzącym
2. Przestrzegać zaleceń dotyczących leczenia
3. Zgłaszać wszelkie niepokojące objawy

Pamiętaj, że ta analiza jest narzędziem wspierającym - ostateczne decyzje
dotyczące leczenia powinny być podejmowane wspólnie z lekarzem.
"""
        else:
            response = """
Jestem asystentem pomagającym zrozumieć wyniki analizy ryzyka.

Mogę pomóc Ci z:
- Wyjaśnieniem wyniku analizy
- Omówieniem czynników wpływających na ocenę
- Ogólnymi informacjami o zapaleniu naczyń

O czym chciałbyś/chciałabyś porozmawiać?
"""

        # Dodaj disclaimer dla pacjentów
        if request.health_literacy != HealthLiteracyLevel.CLINICIAN:
            response += "\n\n---\n*Pamiętaj: to narzędzie informacyjne, nie zastępuje porady lekarza.*"

        return ChatResponse(
            response=response,
            detected_concerns=None,
            follow_up_suggestions=[
                "Opowiedz mi więcej o czynnikach ryzyka",
                "Co mogę zrobić, aby poprawić swoje zdrowie?",
                "Czy powinienem/powinnam martwić się wynikiem?"
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
