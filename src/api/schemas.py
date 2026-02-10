"""
Schematy Pydantic dla API.

Definiuje modele danych dla requestów i responsów API
systemu XAI do wyjaśniania decyzji zewnętrznego AI.
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
    """
    Dane wejściowe pacjenta - 20 cech dopasowanych do modelu.

    Pola odpowiadają 1:1 cechom z feature_names.json.
    """

    # Dane demograficzne / diagnostyczne
    wiek_rozpoznania: float = Field(50.0, ge=0, le=120, description="Wiek w momencie rozpoznania")
    opoznienie_rozpoznia: float = Field(0.0, ge=0, description="Opóźnienie rozpoznania (miesiące)")

    # Manifestacje narządowe
    manifestacja_miesno_szkiel: int = Field(0, ge=0, le=1, description="Manifestacja mięśniowo-szkieletowa")
    manifestacja_skora: int = Field(0, ge=0, le=1, description="Manifestacja skórna")
    manifestacja_wzrok: int = Field(0, ge=0, le=1, description="Manifestacja oczna")
    manifestacja_nos_ucho_gardlo: int = Field(0, ge=0, le=1, description="Manifestacja nos/ucho/gardło")
    manifestacja_oddechowy: int = Field(0, ge=0, le=1, description="Manifestacja układu oddechowego")
    manifestacja_sercowo_naczyniowy: int = Field(0, ge=0, le=1, description="Manifestacja sercowo-naczyniowa")
    manifestacja_pokarmowy: int = Field(0, ge=0, le=1, description="Manifestacja pokarmowa")
    manifestacja_moczowo_plciowy: int = Field(0, ge=0, le=1, description="Manifestacja moczowo-płciowa")
    manifestacja_zajecie_csn: int = Field(0, ge=0, le=1, description="Zajęcie ośrodkowego układu nerwowego")
    manifestacja_neurologiczny: int = Field(0, ge=0, le=1, description="Manifestacja neurologiczna obwodowa")

    # Przebieg choroby
    liczba_zajetych_narzadow: int = Field(0, ge=0, le=20, description="Liczba zajętych narządów")
    zaostrz_wymagajace_hospital: int = Field(0, ge=0, le=1, description="Zaostrzenia wymagające hospitalizacji")
    zaostrz_wymagajace_oit: int = Field(0, ge=0, le=1, description="Zaostrzenia wymagające OIT")

    # Parametry laboratoryjne
    kreatynina: float = Field(100.0, ge=0, description="Kreatynina (μmol/L)")

    # Leczenie
    czas_sterydow: float = Field(0.0, ge=0, description="Czas sterydów (miesiące)")
    plazmaferezy: int = Field(0, ge=0, le=1, description="Plazmaferezy")

    # Eozynofilia
    eozynofilia_krwi_obwodowej_wartosc: float = Field(0.0, ge=0, description="Eozynofilia krwi obwodowej (%)")

    # Powikłania
    powiklania_neurologiczne: int = Field(0, ge=0, le=1, description="Powikłania neurologiczne")

    class Config:
        schema_extra = {
            "example": {
                "wiek_rozpoznania": 50,
                "opoznienie_rozpoznia": 6,
                "manifestacja_miesno_szkiel": 0,
                "manifestacja_skora": 1,
                "manifestacja_wzrok": 0,
                "manifestacja_nos_ucho_gardlo": 0,
                "manifestacja_oddechowy": 1,
                "manifestacja_sercowo_naczyniowy": 0,
                "manifestacja_pokarmowy": 0,
                "manifestacja_moczowo_plciowy": 0,
                "manifestacja_zajecie_csn": 0,
                "manifestacja_neurologiczny": 1,
                "liczba_zajetych_narzadow": 3,
                "zaostrz_wymagajace_hospital": 1,
                "zaostrz_wymagajace_oit": 0,
                "kreatynina": 120,
                "czas_sterydow": 12,
                "plazmaferezy": 0,
                "eozynofilia_krwi_obwodowej_wartosc": 8.5,
                "powiklania_neurologiczne": 0
            }
        }


class AnalysisRequest(BaseModel):
    """Request do analizy: dane pacjenta + wynik z zewnętrznego AI."""
    patient: PatientInput
    external_probability: float = Field(
        ..., ge=0.0, le=1.0,
        description="Prawdopodobieństwo zgonu z zewnętrznego systemu AI (0.0-1.0)"
    )


class ExplanationRequest(BaseModel):
    """Request dla wyjaśnienia."""
    patient: PatientInput
    external_probability: float = Field(
        0.5, ge=0.0, le=1.0,
        description="Prawdopodobieństwo z zewnętrznego AI"
    )
    method: XAIMethod = Field(XAIMethod.SHAP, description="Metoda XAI")
    num_features: int = Field(10, ge=1, le=50, description="Liczba cech do wyświetlenia")


class PatientExplanationRequest(BaseModel):
    """Request dla wyjaśnienia dla pacjenta."""
    patient: PatientInput
    external_probability: float = Field(
        0.5, ge=0.0, le=1.0,
        description="Prawdopodobieństwo z zewnętrznego AI"
    )
    health_literacy: HealthLiteracyLevel = Field(
        HealthLiteracyLevel.BASIC,
        description="Poziom health literacy"
    )
    method: XAIMethod = Field(XAIMethod.SHAP, description="Metoda XAI")


class ChatRequest(BaseModel):
    """Request dla chatu."""
    message: str = Field(..., min_length=1, max_length=2000)
    patient: PatientInput
    external_probability: float = Field(
        0.5, ge=0.0, le=1.0,
        description="Prawdopodobieństwo z zewnętrznego AI"
    )
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
    """Wynik predykcji (wewnętrzny model, używany w wyjaśnieniach)."""
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


class AnalysisOutput(BaseModel):
    """Wynik analizy: external_probability + wyjaśnienia XAI."""
    external_probability: float = Field(..., ge=0, le=1)
    risk_level: RiskLevel
    shap_explanation: Optional[SHAPExplanation] = None
    lime_explanation: Optional[LIMEExplanation] = None
    disclaimer: str = "Wyjaśnienia XAI mają charakter informacyjny. Nie zastępują oceny klinicznej."


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
    explainers_ready: bool = False
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

class BatchAnalysisInput(BaseModel):
    """Dane wejściowe dla batch analysis."""
    patients: List[PatientInput] = Field(..., min_items=1, max_items=10000)
    external_probabilities: List[float] = Field(
        ..., min_items=1,
        description="Prawdopodobieństwa z zewnętrznego AI, jedno na pacjenta"
    )
    include_explanations: bool = Field(False, description="Dołącz wyjaśnienia XAI (wolniejsze)")
    include_risk_factors: bool = Field(True, description="Dołącz top czynniki ryzyka")
    top_n_factors: int = Field(3, ge=1, le=10, description="Liczba top czynników ryzyka")

    @validator('external_probabilities')
    def validate_probabilities(cls, v):
        for p in v:
            if not 0.0 <= p <= 1.0:
                raise ValueError(f"Prawdopodobieństwo musi być w zakresie [0, 1], otrzymano {p}")
        return v

    @validator('external_probabilities')
    def validate_length_match(cls, v, values):
        if 'patients' in values and len(v) != len(values['patients']):
            raise ValueError(
                f"Liczba external_probabilities ({len(v)}) musi odpowiadać "
                f"liczbie pacjentów ({len(values['patients'])})"
            )
        return v


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
    external_probability: float
    risk_level: RiskLevel
    top_risk_factors: Optional[List[RiskFactorItem]] = None
    processing_status: str = "success"
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
    error_type: str
    error_message: str
    is_recoverable: bool = True


class BatchAnalysisOutput(BaseModel):
    """Wynik batch analysis."""
    total_patients: int
    processed_count: int
    success_count: int
    error_count: int
    processing_time_ms: float
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
    explainers_ready: bool
    current_mode: str
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
    # Mapowanie 1:1 nazw Pydantic → nazw modelu (20 cech)
    field_mapping = {
        'wiek_rozpoznania': 'Wiek_rozpoznania',
        'opoznienie_rozpoznia': 'Opoznienie_Rozpoznia',
        'manifestacja_miesno_szkiel': 'Manifestacja_Miesno-Szkiel',
        'manifestacja_skora': 'Manifestacja_Skora',
        'manifestacja_wzrok': 'Manifestacja_Wzrok',
        'manifestacja_nos_ucho_gardlo': 'Manifestacja_Nos/Ucho/Gardlo',
        'manifestacja_oddechowy': 'Manifestacja_Oddechowy',
        'manifestacja_sercowo_naczyniowy': 'Manifestacja_Sercowo-Naczyniowy',
        'manifestacja_pokarmowy': 'Manifestacja_Pokarmowy',
        'manifestacja_moczowo_plciowy': 'Manifestacja_Moczowo-Plciowy',
        'manifestacja_zajecie_csn': 'Manifestacja_Zajecie_CSN',
        'manifestacja_neurologiczny': 'Manifestacja_Neurologiczny',
        'liczba_zajetych_narzadow': 'Liczba_Zajetych_Narzadow',
        'zaostrz_wymagajace_hospital': 'Zaostrz_Wymagajace_Hospital',
        'zaostrz_wymagajace_oit': 'Zaostrz_Wymagajace_OIT',
        'kreatynina': 'Kreatynina',
        'czas_sterydow': 'Czas_Sterydow',
        'plazmaferezy': 'Plazmaferezy',
        'eozynofilia_krwi_obwodowej_wartosc': 'Eozynofilia_Krwi_Obwodowej_Wartosc',
        'powiklania_neurologiczne': 'Powiklania_Neurologiczne',
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
    Konwertuj listę PatientInput do macierzy numpy.

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
