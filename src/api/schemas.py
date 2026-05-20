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
    """Dane wejściowe pacjenta — 30 cech zgodnych z modelem (SelectKBest)."""

    # Dane demograficzne
    wiek: Optional[float] = Field(None, ge=0, le=120, description="Wiek pacjenta")
    plec: int = Field(0, ge=0, le=1, description="Płeć pacjenta (0/1)")
    wiek_rozpoznania: Optional[float] = Field(None, ge=0, le=120, description="Wiek w momencie rozpoznania")
    opoznienie_rozpoznia: Optional[float] = Field(None, ge=0, description="Opóźnienie rozpoznania (miesiące)")

    # Typ zapalenia naczyń
    zap_gpa: int = Field(0, ge=0, le=1, description="Zapalenie GPA (ziarniniakowatość z zapaleniem naczyń)")

    # Manifestacje narządowe
    manifestacja_objaw_ogol: int = Field(0, ge=0, le=3, description="Objawy ogólne")
    manifestacja_miesno_szkiel: int = Field(0, ge=0, le=3, description="Mięśniowo-szkieletowy")
    manifestacja_skora: int = Field(0, ge=0, le=3, description="Skóra")
    manifestacja_wzrok: int = Field(0, ge=0, le=3, description="Wzrok")
    manifestacja_nos_ucho_gardlo: int = Field(0, ge=0, le=3, description="Nos/Ucho/Gardło")
    manifestacja_oddechowy: int = Field(0, ge=0, le=3, description="Oddechowy")
    manifestacja_sercowo_naczyniowy: int = Field(0, ge=0, le=3, description="Sercowo-naczyniowy")
    manifestacja_pokarmowy: int = Field(0, ge=0, le=3, description="Pokarmowy")
    manifestacja_nerki: int = Field(0, ge=0, le=3, description="Nerki")
    manifestacja_moczowo_plciowy: int = Field(0, ge=0, le=3, description="Moczowo-płciowy")
    manifestacja_zajecie_csn: int = Field(0, ge=0, le=3, description="Zajęcie CSN")
    manifestacja_neurologiczny: int = Field(0, ge=0, le=3, description="Neurologiczny")
    liczba_zajetych_narzadow: int = Field(0, ge=0, le=20, description="Liczba zajętych narządów")

    # Przebieg choroby
    zaostrz_wymagajace_hospital: int = Field(0, ge=0, le=3, description="Zaostrzenia wymagające hospitalizacji")
    zaostrz_wymagajace_oit: int = Field(0, ge=0, le=3, description="Zaostrzenia wymagające OIT")
    przebieg_scalony: Optional[float] = Field(None, ge=0, description="Przebieg scalony (skala)")

    # Parametry laboratoryjne
    kreatynina: Optional[float] = Field(None, ge=0, description="Kreatynina (μmol/L)")
    max_crp: Optional[float] = Field(None, ge=0, description="Maksymalne CRP")
    eozynofilia_krwi_obwodowej_wartosc: Optional[float] = Field(None, ge=0, description="Eozynofilia krwi obwodowej (wartość)")

    # Leczenie
    pulsy: int = Field(0, ge=0, le=1, description="Pulsy sterydowe IV")
    czas_sterydow: Optional[float] = Field(None, ge=0, description="Czas sterydów (miesiące)")
    dializa: int = Field(0, ge=0, le=1, description="Dializa")
    plazmaferezy: int = Field(0, ge=0, le=1, description="Plazmaferezy")

    # Diagnostyka
    biopsja_wynik: int = Field(0, ge=0, le=1, description="Wynik biopsji (0=brak/ujemny, 1=dodatni)")

    # Powikłania
    powiklanie_skora: int = Field(0, ge=0, le=1, description="Powikłania skórne")
    powiklania_hematologiczne: int = Field(0, ge=0, le=1, description="Powikłania hematologiczne")
    powiklania_infekcja: int = Field(0, ge=0, le=1, description="Powikłania infekcyjne")
    powiklania_autoimmunologiczne: int = Field(0, ge=0, le=1, description="Powikłania autoimmunologiczne")
    powiklania_neurologiczne: int = Field(0, ge=0, le=1, description="Powikłania neurologiczne")
    powiklania_nowotwor_zlosliwy: int = Field(0, ge=0, le=1, description="Powikłania nowotworowe")
    powiklania_serc_pluca: int = Field(0, ge=0, le=1, description="Powikłania serce/płuca")

    class Config:
        extra = "ignore"
        json_schema_extra = {
            "example": {
                "wiek": 55,
                "plec": 1,
                "wiek_rozpoznania": 50,
                "opoznienie_rozpoznia": 6,
                "zap_gpa": 1,
                "manifestacja_objaw_ogol": 0,
                "manifestacja_miesno_szkiel": 0,
                "manifestacja_skora": 0,
                "manifestacja_wzrok": 0,
                "manifestacja_nos_ucho_gardlo": 0,
                "manifestacja_oddechowy": 0,
                "manifestacja_sercowo_naczyniowy": 0,
                "manifestacja_nerki": 1,
                "manifestacja_pokarmowy": 0,
                "manifestacja_moczowo_plciowy": 0,
                "manifestacja_zajecie_csn": 0,
                "manifestacja_neurologiczny": 1,
                "liczba_zajetych_narzadow": 3,
                "zaostrz_wymagajace_hospital": 1,
                "zaostrz_wymagajace_oit": 0,
                "przebieg_scalony": 2,
                "kreatynina": 120,
                "max_crp": 30,
                "eozynofilia_krwi_obwodowej_wartosc": 0.5,
                "pulsy": 0,
                "czas_sterydow": 12,
                "dializa": 0,
                "plazmaferezy": 0,
                "biopsja_wynik": 1,
                "powiklanie_skora": 0,
                "powiklania_hematologiczne": 0,
                "powiklania_infekcja": 0,
                "powiklania_autoimmunologiczne": 0,
                "powiklania_neurologiczne": 0,
                "powiklania_nowotwor_zlosliwy": 0,
                "powiklania_serc_pluca": 0
            }
        }




class ExplanationRequest(BaseModel):
    """Request dla wyjaśnienia."""
    patient: PatientInput
    method: XAIMethod = Field(XAIMethod.SHAP, description="Metoda XAI")
    num_features: int = Field(10, ge=1, le=50, description="Liczba cech do wyświetlenia")
    model_key: Optional[str] = Field("xgboost", description="Model: xgboost, random_forest, lightgbm")


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


class ModelPrediction(BaseModel):
    """Predykcja pojedynczego modelu."""
    model_name: str = Field(..., description="Nazwa modelu")
    probability: float = Field(..., ge=0, le=1, description="Prawdopodobieństwo zgonu")
    risk_level: RiskLevel
    prediction: int = Field(..., ge=0, le=1)


class MultiModelPredictionOutput(BaseModel):
    """Predykcje ze wszystkich modeli."""
    models: List[ModelPrediction] = Field(default_factory=list)
    ensemble_probability: float = Field(..., ge=0, le=1, description="Średnie prawdopodobieństwo")
    ensemble_risk_level: RiskLevel
    primary_model: str = Field(default="xgboost", description="Model główny")


class PredictionOutput(BaseModel):
    """Wynik predykcji."""
    probability: float = Field(..., ge=0, le=1, description="Prawdopodobieństwo zgonu")
    risk_level: RiskLevel
    prediction: int = Field(..., ge=0, le=1)
    confidence_interval: Optional[Dict[str, float]] = None

    class Config:
        json_schema_extra = {
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
    prediction_data: Optional["AgentPredictionData"] = None


class DALEXExplanation(BaseModel):
    """Wyjaśnienie DALEX Break Down."""
    method: str = "DALEX"
    intercept: float
    prediction: float
    risk_factors: List[Dict[str, Any]] = Field(default_factory=list)
    protective_factors: List[Dict[str, Any]] = Field(default_factory=list)
    variable_importance: Optional[Dict[str, float]] = None


class EBMExplanation(BaseModel):
    """Wyjaśnienie EBM (global + local)."""
    method: str = "EBM"
    prediction: int
    probability: float
    risk_level: RiskLevel
    global_importance: Dict[str, float] = Field(default_factory=dict)
    local_contributions: List[Dict[str, Any]] = Field(default_factory=list)
    interactions: List[str] = Field(default_factory=list)


class ComparisonResult(BaseModel):
    """Wynik porównania metod XAI."""
    methods_compared: List[str]
    ranking_agreement: float
    common_top_features: List[str]
    individual_rankings: Dict[str, List[str]]
    spearman_correlations: Dict[str, float]


class KrishnaComparisonResult(BaseModel):
    """Pełny panel metryk porównania XAI: Krishna et al. 2024 + RBO + Weighted Kendall.

    Każda macierz to słownik {method: {method: value}} reprezentujący tabelę N×N.
    """
    methods_compared: List[str]
    ks: List[int]
    rbo_p: float
    panels: Dict[str, Dict[str, Dict[str, float]]] = Field(
        default_factory=dict,
        description="Macierze par metoda × metoda dla każdej metryki",
    )
    summary: Dict[str, float] = Field(
        default_factory=dict,
        description="Średnia górnego trójkąta (bez diagonali) dla każdej metryki",
    )
    rankings: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Pełny ranking cech zwracany przez każdą metodę",
    )


class ErrorResponse(BaseModel):
    """Odpowiedź błędu."""
    error: str
    detail: Optional[str] = None
    code: int


# ============================================================================
# CALIBRATION & DECISION CURVE ANALYSIS
# ============================================================================

class CalibrationCurvePoint(BaseModel):
    """Pojedynczy punkt diagramu wiarygodności."""
    mean_predicted: float
    fraction_positive: float


class ModelCalibration(BaseModel):
    """Pełna kalibracja jednego modelu na zbiorze testowym."""
    model: str
    brier_score: float
    calibration_slope: float
    calibration_intercept: float
    n_test: int
    curve: List[CalibrationCurvePoint] = Field(default_factory=list)


class CalibrationResponse(BaseModel):
    """Kalibracja wszystkich dostępnych modeli."""
    n_test: int
    n_positive: int
    prevalence: float
    models: List[ModelCalibration] = Field(default_factory=list)


class CounterfactualChange(BaseModel):
    feature: str
    from_value: float = Field(..., alias="from")
    to_value: float = Field(..., alias="to")
    delta: float

    class Config:
        populate_by_name = True


class CounterfactualExample(BaseModel):
    predicted_proba: float
    flipped_class: bool
    n_changes: int
    l1_distance: float
    nearest_neighbor_distance: float
    changes: List[CounterfactualChange] = Field(default_factory=list)


class CounterfactualMetrics(BaseModel):
    validity: float = 0.0
    n_changes_avg: float = 0.0
    l1_distance_avg: float = 0.0
    knn_distance_avg: float = 0.0
    n_cfs: int = 0


class CounterfactualResponse(BaseModel):
    """Wynik DiCE: kontrfaktyczne wyjaśnienia dla pacjenta."""
    success: bool
    method: Optional[str] = None
    original_probability: float
    desired_class: int
    cfs: List[CounterfactualExample] = Field(default_factory=list)
    metrics: CounterfactualMetrics = Field(default_factory=CounterfactualMetrics)
    features_varied: List[str] = Field(default_factory=list)
    message: str = ""


class NetBenefitPoint(BaseModel):
    threshold: float
    net_benefit: float


class ModelDCA(BaseModel):
    model: str
    points: List[NetBenefitPoint] = Field(default_factory=list)


class DecisionCurveResponse(BaseModel):
    """Decision Curve Analysis dla wszystkich modeli + linie odniesienia."""
    n_test: int
    n_positive: int
    prevalence: float
    threshold_grid: List[float] = Field(default_factory=list)
    treat_all: List[NetBenefitPoint] = Field(default_factory=list)
    models: List[ModelDCA] = Field(default_factory=list)


# ============================================================================
# BATCH PROCESSING SCHEMAS
# ============================================================================

class BatchPatientInput(BaseModel):
    """Dane wejściowe dla batch prediction."""
    patients: List[PatientInput] = Field(..., min_items=1, max_items=10000)
    include_risk_factors: bool = Field(True, description="Dołącz top czynniki ryzyka")
    top_n_factors: int = Field(3, ge=1, le=10, description="Liczba top czynników ryzyka")

    class Config:
        json_schema_extra = {
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
        json_schema_extra = {
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


class AgentPredictionFactor(BaseModel):
    """Czynnik ryzyka z predykcji agenta."""
    feature: str
    contribution: float
    direction: str = "increases_risk"


class AgentPredictionData(BaseModel):
    """Dane predykcji zwracane przez agenta w czacie."""
    prediction: PredictionOutput
    factors: List[AgentPredictionFactor] = []
    base_value: float = 0.0


# ============================================================================
# AGENT CONVERSATION SCHEMAS
# ============================================================================

class AgentConversationRequest(BaseModel):
    """Request dla konwersacyjnego agenta zbierającego dane."""
    message: str = Field(..., min_length=1, max_length=2000)
    conversation_history: List[Dict[str, str]] = Field(default_factory=list)
    collected_data: Dict[str, Any] = Field(default_factory=dict)
    current_step: int = 0
    phase: str = "collecting"  # collecting | prediction | discussion


class AgentConversationResponse(BaseModel):
    """Odpowiedź agenta konwersacyjnego."""
    response: str
    collected_data: Dict[str, Any] = Field(default_factory=dict)
    current_step: int = 0
    phase: str = "collecting"
    missing_fields: List[str] = Field(default_factory=list)
    prediction_data: Optional[AgentPredictionData] = None
    follow_up_suggestions: List[str] = Field(default_factory=list)
    field_meta: Optional[Dict[str, Any]] = None


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
    # Mapowanie nazw z Pydantic na nazwy w modelu (30 cech SelectKBest)
    field_mapping = {
        'wiek': 'Wiek',
        'plec': 'Plec',
        'wiek_rozpoznania': 'Wiek_rozpoznania',
        'opoznienie_rozpoznia': 'Opoznienie_Rozpoznia',
        'zap_gpa': 'Zap_GPA',
        'manifestacja_objaw_ogol': 'Manifestacja_Objaw_Ogol',
        'manifestacja_miesno_szkiel': 'Manifestacja_Miesno-Szkiel',
        'manifestacja_skora': 'Manifestacja_Skora',
        'manifestacja_wzrok': 'Manifestacja_Wzrok',
        'manifestacja_nos_ucho_gardlo': 'Manifestacja_Nos/Ucho/Gardlo',
        'manifestacja_oddechowy': 'Manifestacja_Oddechowy',
        'manifestacja_sercowo_naczyniowy': 'Manifestacja_Sercowo-Naczyniowy',
        'manifestacja_pokarmowy': 'Manifestacja_Pokarmowy',
        'manifestacja_nerki': 'Manifestacja_Nerki',
        'manifestacja_moczowo_plciowy': 'Manifestacja_Moczowo-Plciowy',
        'manifestacja_zajecie_csn': 'Manifestacja_Zajecie_CSN',
        'manifestacja_neurologiczny': 'Manifestacja_Neurologiczny',
        'liczba_zajetych_narzadow': 'Liczba_Zajetych_Narzadow',
        'zaostrz_wymagajace_hospital': 'Zaostrz_Wymagajace_Hospital',
        'zaostrz_wymagajace_oit': 'Zaostrz_Wymagajace_OIT',
        'przebieg_scalony': 'Przebieg_scalony',
        'kreatynina': 'Kreatynina',
        'max_crp': 'Max_CRP',
        'pulsy': 'Pulsy',
        'czas_sterydow': 'Czas_Sterydow',
        'dializa': 'Dializa',
        'plazmaferezy': 'Plazmaferezy',
        'eozynofilia_krwi_obwodowej_wartosc': 'Eozynofilia_Krwi_Obwodowej_Wartosc',
        'biopsja_wynik': 'Biopsja_Wynik',
        'powiklanie_skora': 'Powiklanie_Skora',
        'powiklania_hematologiczne': 'Powiklania_Hematologiczne',
        'powiklania_infekcja': 'Powiklania_Infekcja',
        'powiklania_autoimmunologiczne': 'Powiklania_Autoimmunologiczne',
        'powiklania_neurologiczne': 'Powiklania_Neurologiczne',
        'powiklania_nowotwor_zlosliwy': 'Powiklania_Nowotwor_Zlosliwy',
        'powiklania_serc_pluca': 'Powiklania_Serce/pluca',
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
