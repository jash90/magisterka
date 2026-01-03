"""
Schematy Pydantic dla API.

Definiuje modele danych dla requestów i responsów API
systemu XAI do predykcji śmiertelności.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
from enum import Enum


class HealthLiteracyLevel(str, Enum):
    """Poziom health literacy."""
    BASIC = "basic"
    ADVANCED = "advanced"
    CLINICIAN = "clinician"


class RiskLevel(str, Enum):
    """Poziom ryzyka."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


class XAIMethod(str, Enum):
    """Metoda XAI."""
    LIME = "lime"
    SHAP = "shap"
    DALEX = "dalex"
    EBM = "ebm"


# ============================================================================
# INPUT SCHEMAS
# ============================================================================

class PatientInput(BaseModel):
    """Dane wejściowe pacjenta."""

    # Dane demograficzne
    wiek: float = Field(..., ge=0, le=120, description="Wiek pacjenta")
    plec: int = Field(..., ge=0, le=1, description="Płeć (0=K, 1=M)")
    wiek_rozpoznania: Optional[float] = Field(None, ge=0, le=120, description="Wiek w momencie rozpoznania")

    # Opóźnienie diagnostyczne
    opoznienie_rozpoznia: Optional[float] = Field(None, ge=0, description="Opóźnienie rozpoznania (miesiące)")

    # Manifestacje narządowe
    liczba_zajetych_narzadow: int = Field(0, ge=0, le=20, description="Liczba zajętych narządów")
    manifestacja_sercowo_naczyniowy: int = Field(0, ge=0, le=1)
    manifestacja_nerki: int = Field(0, ge=0, le=1)
    manifestacja_pokarmowy: int = Field(0, ge=0, le=1)
    manifestacja_zajecie_csn: int = Field(0, ge=0, le=1)
    manifestacja_neurologiczny: int = Field(0, ge=0, le=1)

    # Przebieg choroby
    zaostrz_wymagajace_oit: int = Field(0, ge=0, le=1, description="Zaostrzenia wymagające OIT")

    # Parametry laboratoryjne
    kreatynina: Optional[float] = Field(None, ge=0, description="Kreatynina (μmol/L)")
    max_crp: Optional[float] = Field(None, ge=0, description="Maksymalne CRP (mg/L)")

    # Leczenie
    plazmaferezy: int = Field(0, ge=0, le=1)
    dializa: int = Field(0, ge=0, le=1)
    sterydy_dawka_g: Optional[float] = Field(None, ge=0, description="Dawka sterydów (g)")
    czas_sterydow: Optional[float] = Field(None, ge=0, description="Czas sterydów (miesiące)")

    # Powikłania
    powiklania_serce_pluca: int = Field(0, ge=0, le=1)
    powiklania_infekcja: int = Field(0, ge=0, le=1)

    class Config:
        schema_extra = {
            "example": {
                "wiek": 55,
                "plec": 1,
                "wiek_rozpoznania": 50,
                "opoznienie_rozpoznia": 6,
                "liczba_zajetych_narzadow": 3,
                "manifestacja_sercowo_naczyniowy": 0,
                "manifestacja_nerki": 1,
                "manifestacja_pokarmowy": 0,
                "manifestacja_zajecie_csn": 0,
                "manifestacja_neurologiczny": 1,
                "zaostrz_wymagajace_oit": 0,
                "kreatynina": 120,
                "max_crp": 45,
                "plazmaferezy": 0,
                "dializa": 0,
                "sterydy_dawka_g": 0.5,
                "czas_sterydow": 12,
                "powiklania_serce_pluca": 0,
                "powiklania_infekcja": 0
            }
        }


class ExplanationRequest(BaseModel):
    """Request dla wyjaśnienia."""
    patient: PatientInput
    method: XAIMethod = Field(XAIMethod.SHAP, description="Metoda XAI")
    num_features: int = Field(10, ge=1, le=50, description="Liczba cech do wyświetlenia")


class PatientExplanationRequest(BaseModel):
    """Request dla wyjaśnienia dla pacjenta."""
    patient: PatientInput
    health_literacy: HealthLiteracyLevel = Field(
        HealthLiteracyLevel.BASIC,
        description="Poziom health literacy"
    )
    method: XAIMethod = Field(XAIMethod.SHAP, description="Metoda XAI")


class ChatRequest(BaseModel):
    """Request dla chatu."""
    message: str = Field(..., min_length=1, max_length=2000)
    patient: PatientInput
    health_literacy: HealthLiteracyLevel = Field(HealthLiteracyLevel.BASIC)
    conversation_history: List[Dict[str, str]] = Field(default_factory=list)


# ============================================================================
# OUTPUT SCHEMAS
# ============================================================================

class FeatureContribution(BaseModel):
    """Wkład pojedynczej cechy."""
    feature: str
    value: float
    contribution: float
    direction: str


class PredictionOutput(BaseModel):
    """Wynik predykcji."""
    probability: float = Field(..., ge=0, le=1, description="Prawdopodobieństwo zgonu")
    risk_level: RiskLevel
    prediction: int = Field(..., ge=0, le=1)
    confidence_interval: Optional[Dict[str, float]] = None

    class Config:
        schema_extra = {
            "example": {
                "probability": 0.35,
                "risk_level": "moderate",
                "prediction": 0,
                "confidence_interval": {"lower": 0.28, "upper": 0.42}
            }
        }


class SHAPExplanation(BaseModel):
    """Wyjaśnienie SHAP."""
    method: str = "SHAP"
    base_value: float
    shap_values: Dict[str, float]
    feature_contributions: List[FeatureContribution]
    risk_factors: List[FeatureContribution]
    protective_factors: List[FeatureContribution]
    prediction: PredictionOutput


class LIMEExplanation(BaseModel):
    """Wyjaśnienie LIME."""
    method: str = "LIME"
    intercept: float
    feature_weights: List[Dict[str, Any]]
    risk_factors: List[Dict[str, Any]]
    protective_factors: List[Dict[str, Any]]
    local_prediction: float
    prediction: PredictionOutput


class PatientExplanation(BaseModel):
    """Wyjaśnienie dla pacjenta."""
    risk_level: str
    risk_description: str
    main_concerns: List[str]
    positive_factors: List[str]
    recommendations: str
    technical_summary: Optional[Dict[str, Any]] = None
    disclaimer: str


class ModelInfo(BaseModel):
    """Informacje o modelu."""
    model_type: str
    n_features: int
    feature_names: List[str]
    training_date: Optional[str]
    performance_metrics: Dict[str, float]
    version: str


class GlobalImportance(BaseModel):
    """Globalna ważność cech."""
    feature_importance: Dict[str, float]
    top_features: List[str]
    method: str
    n_samples: int


class HealthCheckResponse(BaseModel):
    """Odpowiedź health check."""
    status: str
    model_loaded: bool
    api_version: str
    timestamp: str


class ChatResponse(BaseModel):
    """Odpowiedź z chatu."""
    response: str
    detected_concerns: Optional[List[str]] = None
    follow_up_suggestions: Optional[List[str]] = None


class ComparisonResult(BaseModel):
    """Wynik porównania metod XAI."""
    methods_compared: List[str]
    ranking_agreement: float
    common_top_features: List[str]
    individual_rankings: Dict[str, List[str]]
    spearman_correlations: Dict[str, float]


class ErrorResponse(BaseModel):
    """Odpowiedź błędu."""
    error: str
    detail: Optional[str] = None
    code: int


# ============================================================================
# BATCH PROCESSING SCHEMAS
# ============================================================================

class BatchPatientInput(BaseModel):
    """Dane wejściowe dla batch prediction."""
    patients: List[PatientInput] = Field(..., min_items=1, max_items=10000)
    include_risk_factors: bool = Field(True, description="Dołącz top czynniki ryzyka")
    top_n_factors: int = Field(3, ge=1, le=10, description="Liczba top czynników ryzyka")

    class Config:
        schema_extra = {
            "example": {
                "patients": [
                    {"wiek": 55, "plec": 1, "liczba_zajetych_narzadow": 3, "manifestacja_nerki": 1},
                    {"wiek": 45, "plec": 0, "liczba_zajetych_narzadow": 2, "manifestacja_nerki": 0}
                ],
                "include_risk_factors": True,
                "top_n_factors": 3
            }
        }


class RiskFactorItem(BaseModel):
    """Pojedynczy czynnik ryzyka."""
    feature: str
    value: float
    importance: float
    direction: str  # "increases_risk" | "decreases_risk"


class BatchPatientResult(BaseModel):
    """Wynik dla pojedynczego pacjenta w batch."""
    patient_id: Optional[str] = None
    index: int
    prediction: PredictionOutput
    top_risk_factors: Optional[List[RiskFactorItem]] = None
    processing_status: str = "success"  # success | demo | error
    error_message: Optional[str] = None


class BatchSummary(BaseModel):
    """Podsumowanie wyników batch."""
    total_count: int
    low_risk_count: int
    moderate_risk_count: int
    high_risk_count: int
    avg_probability: float
    median_probability: float
    min_probability: float
    max_probability: float


class BatchProcessingError(BaseModel):
    """Błąd przetwarzania pojedynczego pacjenta."""
    patient_index: int
    patient_id: Optional[str] = None
    error_type: str  # validation | prediction | timeout
    error_message: str
    is_recoverable: bool = True


class BatchPredictionOutput(BaseModel):
    """Wynik batch prediction."""
    total_patients: int
    processed_count: int
    success_count: int
    error_count: int
    processing_time_ms: float
    mode: str  # "api" | "demo"
    summary: BatchSummary
    results: List[BatchPatientResult]
    errors: List[BatchProcessingError] = []

    class Config:
        schema_extra = {
            "example": {
                "total_patients": 100,
                "processed_count": 100,
                "success_count": 100,
                "error_count": 0,
                "processing_time_ms": 150.5,
                "mode": "api",
                "summary": {
                    "total_count": 100,
                    "low_risk_count": 45,
                    "moderate_risk_count": 35,
                    "high_risk_count": 20,
                    "avg_probability": 0.42,
                    "median_probability": 0.38,
                    "min_probability": 0.05,
                    "max_probability": 0.92
                },
                "results": [],
                "errors": []
            }
        }


class DemoModeStatus(BaseModel):
    """Status trybu demo."""
    demo_allowed: bool
    model_loaded: bool
    current_mode: str  # "api" | "demo" | "unavailable"
    force_api_mode: bool


class DemoModeRequest(BaseModel):
    """Request do zmiany trybu demo."""
    enabled: bool


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def patient_to_array(patient: PatientInput, feature_order: List[str]) -> List[float]:
    """
    Konwertuj PatientInput do tablicy zgodnej z modelem.

    Args:
        patient: Dane pacjenta
        feature_order: Kolejność cech wymagana przez model

    Returns:
        Lista wartości cech
    """
    # Mapowanie nazw z Pydantic na nazwy w modelu
    field_mapping = {
        'wiek': 'Wiek',
        'plec': 'Plec',
        'wiek_rozpoznania': 'Wiek_rozpoznania',
        'opoznienie_rozpoznia': 'Opoznienie_Rozpoznia',
        'liczba_zajetych_narzadow': 'Liczba_Zajetych_Narzadow',
        'manifestacja_sercowo_naczyniowy': 'Manifestacja_Sercowo-Naczyniowy',
        'manifestacja_nerki': 'Manifestacja_Nerki',
        'manifestacja_pokarmowy': 'Manifestacja_Pokarmowy',
        'manifestacja_zajecie_csn': 'Manifestacja_Zajecie_CSN',
        'manifestacja_neurologiczny': 'Manifestacja_Neurologiczny',
        'zaostrz_wymagajace_oit': 'Zaostrz_Wymagajace_OIT',
        'kreatynina': 'Kreatynina',
        'max_crp': 'Max_CRP',
        'plazmaferezy': 'Plazmaferezy',
        'dializa': 'Dializa',
        'sterydy_dawka_g': 'Sterydy_Dawka_g',
        'czas_sterydow': 'Czas_Sterydow',
        'powiklania_serce_pluca': 'Powiklania_Serce/pluca',
        'powiklania_infekcja': 'Powiklania_Infekcja'
    }

    # Odwróć mapowanie
    reverse_mapping = {v: k for k, v in field_mapping.items()}

    patient_dict = patient.dict()
    values = []

    for feature in feature_order:
        pydantic_name = reverse_mapping.get(feature)
        if pydantic_name and pydantic_name in patient_dict:
            value = patient_dict[pydantic_name]
            values.append(value if value is not None else 0)
        else:
            values.append(0)

    return values


def get_risk_level_from_probability(probability: float) -> RiskLevel:
    """Określ poziom ryzyka na podstawie prawdopodobieństwa."""
    if probability < 0.3:
        return RiskLevel.LOW
    elif probability < 0.7:
        return RiskLevel.MODERATE
    else:
        return RiskLevel.HIGH


def patients_to_matrix(patients: List[PatientInput], feature_order: List[str]) -> 'np.ndarray':
    """
    Konwertuj listę PatientInput do macierzy numpy dla vectorized prediction.

    Args:
        patients: Lista danych pacjentów
        feature_order: Kolejność cech wymagana przez model

    Returns:
        np.ndarray o kształcie (n_patients, n_features)
    """
    import numpy as np

    n_patients = len(patients)
    n_features = len(feature_order)
    X = np.zeros((n_patients, n_features), dtype=np.float32)

    for i, patient in enumerate(patients):
        X[i] = patient_to_array(patient, feature_order)

    return X
