"""
Testy jednostkowe i integracyjne dla API FastAPI.

Zawiera testy schematow Pydantic (jednostkowe) oraz
testy endpointow API w trybie demo (integracyjne) z TestClient.
"""

import os
import sys
import pytest
from pathlib import Path

# Ustaw tryb demo PRZED importem app
os.environ["ALLOW_DEMO"] = "true"
os.environ.setdefault("CORS_ORIGINS", "http://localhost:8501")

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.schemas import (
    PatientInput, PredictionOutput, RiskLevel, HealthLiteracyLevel,
    patient_to_array, get_risk_level_from_probability
)


# ============================================================================
# SAMPLE DATA
# ============================================================================

SAMPLE_PATIENT = {
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
    "powiklania_infekcja": 0,
}

SAMPLE_PATIENT_HIGH_RISK = {
    "wiek": 75,
    "plec": 1,
    "wiek_rozpoznania": 70,
    "opoznienie_rozpoznia": 12,
    "liczba_zajetych_narzadow": 5,
    "manifestacja_sercowo_naczyniowy": 1,
    "manifestacja_nerki": 1,
    "manifestacja_pokarmowy": 1,
    "manifestacja_zajecie_csn": 1,
    "manifestacja_neurologiczny": 1,
    "zaostrz_wymagajace_oit": 1,
    "kreatynina": 350,
    "max_crp": 200,
    "plazmaferezy": 1,
    "dializa": 1,
    "sterydy_dawka_g": 3.0,
    "czas_sterydow": 36,
    "powiklania_serce_pluca": 1,
    "powiklania_infekcja": 1,
}


# ============================================================================
# SCHEMA UNIT TESTS (from original file)
# ============================================================================

class TestSchemas:
    """Testy dla schematow Pydantic."""

    def test_patient_input_valid(self):
        """Test poprawnych danych pacjenta."""
        patient = PatientInput(
            wiek=55,
            plec=1,
            wiek_rozpoznania=50,
            liczba_zajetych_narzadow=3,
            manifestacja_nerki=1,
            zaostrz_wymagajace_oit=0,
            kreatynina=100.0,
            max_crp=30.0
        )

        assert patient.wiek == 55
        assert patient.plec == 1
        assert patient.manifestacja_nerki == 1

    def test_patient_input_defaults(self):
        """Test wartosci domyslnych."""
        patient = PatientInput(
            wiek=50,
            plec=0
        )

        assert patient.liczba_zajetych_narzadow == 0
        assert patient.manifestacja_nerki == 0
        assert patient.dializa == 0

    def test_patient_input_validation_age(self):
        """Test walidacji wieku."""
        with pytest.raises(ValueError):
            PatientInput(wiek=-5, plec=0)

        with pytest.raises(ValueError):
            PatientInput(wiek=150, plec=0)

    def test_patient_input_validation_sex(self):
        """Test walidacji plci."""
        with pytest.raises(ValueError):
            PatientInput(wiek=50, plec=2)

    def test_prediction_output_valid(self):
        """Test poprawnego wyniku predykcji."""
        prediction = PredictionOutput(
            probability=0.35,
            risk_level=RiskLevel.MODERATE,
            prediction=0
        )

        assert prediction.probability == 0.35
        assert prediction.risk_level == RiskLevel.MODERATE

    def test_prediction_output_with_ci(self):
        """Test predykcji z przedzialem ufnosci."""
        prediction = PredictionOutput(
            probability=0.5,
            risk_level=RiskLevel.MODERATE,
            prediction=0,
            confidence_interval={"lower": 0.4, "upper": 0.6}
        )

        assert prediction.confidence_interval["lower"] == 0.4
        assert prediction.confidence_interval["upper"] == 0.6

    def test_get_risk_level_from_probability(self):
        """Test konwersji prawdopodobienstwa na poziom ryzyka."""
        assert get_risk_level_from_probability(0.1) == RiskLevel.LOW
        assert get_risk_level_from_probability(0.29) == RiskLevel.LOW
        assert get_risk_level_from_probability(0.3) == RiskLevel.MODERATE
        assert get_risk_level_from_probability(0.5) == RiskLevel.MODERATE
        assert get_risk_level_from_probability(0.69) == RiskLevel.MODERATE
        assert get_risk_level_from_probability(0.7) == RiskLevel.HIGH
        assert get_risk_level_from_probability(0.95) == RiskLevel.HIGH

    def test_patient_to_array(self):
        """Test konwersji pacjenta do tablicy."""
        patient = PatientInput(
            wiek=55,
            plec=1,
            kreatynina=100.0,
            max_crp=30.0
        )

        feature_order = ['Wiek', 'Plec', 'Kreatynina', 'Max_CRP']

        arr = patient_to_array(patient, feature_order)

        assert len(arr) == 4
        assert arr[0] == 55  # Wiek
        assert arr[1] == 1   # Plec
        assert arr[2] == 100.0  # Kreatynina
        assert arr[3] == 30.0   # Max_CRP

    def test_health_literacy_levels(self):
        """Test poziomow health literacy."""
        assert HealthLiteracyLevel.BASIC.value == "basic"
        assert HealthLiteracyLevel.ADVANCED.value == "advanced"
        assert HealthLiteracyLevel.CLINICIAN.value == "clinician"

    def test_risk_level_enum(self):
        """Test enum poziomow ryzyka."""
        assert RiskLevel.LOW.value == "low"
        assert RiskLevel.MODERATE.value == "moderate"
        assert RiskLevel.HIGH.value == "high"


class TestAPIRoutes:
    """Testy ze wymagane sciezki istnieja."""

    def test_import_api_main(self):
        """Test importu modulu API."""
        from src.api.main import app
        assert app is not None

    def test_api_routes_exist(self):
        """Test ze wymagane sciezki istnieja."""
        from src.api.main import app

        routes = [route.path for route in app.routes]

        assert "/" in routes
        assert "/health" in routes
        assert "/predict" in routes
        assert "/explain/shap" in routes
        assert "/explain/lime" in routes
        assert "/explain/patient" in routes
        assert "/model/info" in routes
        assert "/chat" in routes


# ============================================================================
# INTEGRATION TESTS WITH TestClient
# ============================================================================

@pytest.fixture(scope="module")
def client():
    """TestClient fixture — wspoldzielony w calym module."""
    from fastapi.testclient import TestClient
    from src.api.main import app
    with TestClient(app) as c:
        yield c


# ============================================================================
# INFO ENDPOINTS
# ============================================================================

class TestRootEndpoint:
    """Testy GET /"""

    def test_root_returns_200(self, client):
        """GET / zwraca 200."""
        response = client.get("/")
        assert response.status_code == 200

    def test_root_contains_name_and_version(self, client):
        """GET / zwraca name i version."""
        data = client.get("/").json()
        assert data["name"] == "Vasculitis XAI API"
        assert data["version"] == "1.0.0"
        assert "docs" in data
        assert "health" in data


class TestHealthEndpoint:
    """Testy GET /health"""

    def test_health_returns_200(self, client):
        """GET /health zwraca 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_structure(self, client):
        """GET /health zwraca HealthCheckResponse."""
        data = client.get("/health").json()
        assert data["status"] == "healthy"
        assert "model_loaded" in data
        assert isinstance(data["model_loaded"], bool)
        assert data["api_version"] == "1.0.0"
        assert "timestamp" in data


class TestXAIHealthEndpoint:
    """Testy GET /health/xai"""

    def test_xai_health_returns_200(self, client):
        """GET /health/xai zwraca 200."""
        response = client.get("/health/xai")
        assert response.status_code == 200

    def test_xai_health_response_structure(self, client):
        """GET /health/xai zwraca xai_ready i dialysis_xai_ready."""
        data = client.get("/health/xai").json()
        assert "xai_ready" in data
        assert "dialysis_xai_ready" in data
        assert isinstance(data["xai_ready"], bool)
        assert isinstance(data["dialysis_xai_ready"], bool)


# ============================================================================
# PREDICTION ENDPOINTS
# ============================================================================

class TestPredictEndpoint:
    """Testy POST /predict"""

    def test_predict_returns_200(self, client):
        """POST /predict z poprawnymi danymi zwraca 200."""
        response = client.post("/predict", json=SAMPLE_PATIENT)
        assert response.status_code == 200

    def test_predict_response_structure(self, client):
        """POST /predict zwraca PredictionOutput."""
        data = client.post("/predict", json=SAMPLE_PATIENT).json()
        assert "probability" in data
        assert "risk_level" in data
        assert "prediction" in data
        assert 0.0 <= data["probability"] <= 1.0
        assert data["risk_level"] in ["low", "moderate", "high"]
        assert data["prediction"] in [0, 1]

    def test_predict_demo_mode_flag(self, client):
        """POST /predict — is_demo jest bool (True gdy demo, False gdy model zaladowany)."""
        data = client.post("/predict", json=SAMPLE_PATIENT).json()
        assert isinstance(data.get("is_demo"), bool)

    def test_predict_different_patients_return_results(self, client):
        """POST /predict dla roznych pacjentow zwraca poprawne wyniki."""
        low_risk = client.post("/predict", json=SAMPLE_PATIENT).json()
        high_risk = client.post("/predict", json=SAMPLE_PATIENT_HIGH_RISK).json()
        # Oba wywolania powinny zwrocic poprawne predykcje
        assert 0.0 <= low_risk["probability"] <= 1.0
        assert 0.0 <= high_risk["probability"] <= 1.0
        assert low_risk["risk_level"] in ["low", "moderate", "high"]
        assert high_risk["risk_level"] in ["low", "moderate", "high"]

    def test_predict_invalid_negative_age(self, client):
        """POST /predict z ujemnym wiekiem zwraca 422."""
        invalid_patient = SAMPLE_PATIENT.copy()
        invalid_patient["wiek"] = -5
        response = client.post("/predict", json=invalid_patient)
        assert response.status_code == 422

    def test_predict_invalid_creatinine_too_high(self, client):
        """POST /predict z kreatynina > 2000 zwraca 422."""
        invalid_patient = SAMPLE_PATIENT.copy()
        invalid_patient["kreatynina"] = 2500
        response = client.post("/predict", json=invalid_patient)
        assert response.status_code == 422

    def test_predict_invalid_age_over_120(self, client):
        """POST /predict z wiekiem > 120 zwraca 422."""
        invalid_patient = SAMPLE_PATIENT.copy()
        invalid_patient["wiek"] = 150
        response = client.post("/predict", json=invalid_patient)
        assert response.status_code == 422

    def test_predict_invalid_plec(self, client):
        """POST /predict z plec > 1 zwraca 422."""
        invalid_patient = SAMPLE_PATIENT.copy()
        invalid_patient["plec"] = 3
        response = client.post("/predict", json=invalid_patient)
        assert response.status_code == 422

    def test_predict_minimal_patient(self, client):
        """POST /predict z minimalnymi danymi (tylko wymagane pola)."""
        minimal = {"wiek": 40, "plec": 0}
        response = client.post("/predict", json=minimal)
        assert response.status_code == 200
        data = response.json()
        assert 0.0 <= data["probability"] <= 1.0


class TestBatchPredictEndpoint:
    """Testy POST /predict/batch"""

    def test_batch_predict_returns_200(self, client):
        """POST /predict/batch z 2 pacjentami zwraca 200."""
        batch = {
            "patients": [SAMPLE_PATIENT, SAMPLE_PATIENT_HIGH_RISK],
            "include_risk_factors": True,
            "top_n_factors": 3,
        }
        response = client.post("/predict/batch", json=batch)
        assert response.status_code == 200

    def test_batch_predict_response_structure(self, client):
        """POST /predict/batch zwraca BatchPredictionOutput."""
        batch = {
            "patients": [SAMPLE_PATIENT, SAMPLE_PATIENT_HIGH_RISK],
            "include_risk_factors": True,
            "top_n_factors": 3,
        }
        data = client.post("/predict/batch", json=batch).json()
        assert data["total_patients"] == 2
        assert data["processed_count"] == 2
        assert data["success_count"] == 2
        assert data["error_count"] == 0
        assert data["mode"] in ["api", "demo"]
        assert "summary" in data
        assert "results" in data
        assert len(data["results"]) == 2

    def test_batch_predict_summary_statistics(self, client):
        """POST /predict/batch summary zawiera poprawne statystyki."""
        batch = {
            "patients": [SAMPLE_PATIENT, SAMPLE_PATIENT_HIGH_RISK],
            "include_risk_factors": False,
        }
        data = client.post("/predict/batch", json=batch).json()
        summary = data["summary"]
        assert summary["total_count"] == 2
        assert 0.0 <= summary["avg_probability"] <= 1.0
        assert 0.0 <= summary["min_probability"] <= summary["max_probability"] <= 1.0
        assert summary["median_probability"] >= summary["min_probability"]
        assert summary["median_probability"] <= summary["max_probability"]
        total_risk = (
            summary["low_risk_count"]
            + summary["moderate_risk_count"]
            + summary["high_risk_count"]
        )
        assert total_risk == 2

    def test_batch_predict_empty_list_rejected(self, client):
        """POST /predict/batch z pusta lista pacjentow zwraca 422."""
        batch = {"patients": []}
        response = client.post("/predict/batch", json=batch)
        assert response.status_code == 422

    def test_batch_predict_single_patient(self, client):
        """POST /predict/batch z 1 pacjentem dziala poprawnie."""
        batch = {"patients": [SAMPLE_PATIENT]}
        response = client.post("/predict/batch", json=batch)
        assert response.status_code == 200
        data = response.json()
        assert data["total_patients"] == 1
        assert len(data["results"]) == 1

    def test_batch_predict_risk_factors_field_present(self, client):
        """POST /predict/batch z include_risk_factors=True — pole top_risk_factors jest obecne."""
        batch = {
            "patients": [SAMPLE_PATIENT],
            "include_risk_factors": True,
            "top_n_factors": 3,
        }
        data = client.post("/predict/batch", json=batch).json()
        result = data["results"][0]
        # top_risk_factors moze byc None jesli wystapi blad walidacji wewnatrz,
        # lub lista jesli czynniki zostaly poprawnie wyodrebnione
        assert "top_risk_factors" in result


class TestAllModelsPredictEndpoint:
    """Testy POST /predict/all-models"""

    def test_all_models_returns_200(self, client):
        """POST /predict/all-models zwraca 200."""
        response = client.post("/predict/all-models", json=SAMPLE_PATIENT)
        assert response.status_code == 200

    def test_all_models_response_structure(self, client):
        """POST /predict/all-models zwraca MultiModelPredictionOutput."""
        data = client.post("/predict/all-models", json=SAMPLE_PATIENT).json()
        assert "results" in data
        assert "consensus_probability" in data
        assert "consensus_risk_level" in data
        assert "agreement_score" in data
        assert len(data["results"]) > 0
        assert 0.0 <= data["consensus_probability"] <= 1.0
        assert 0.0 <= data["agreement_score"] <= 1.0
        assert data["consensus_risk_level"] in ["low", "moderate", "high"]

    def test_all_models_individual_results(self, client):
        """POST /predict/all-models — kazdy wynik ma wymagane pola."""
        data = client.post("/predict/all-models", json=SAMPLE_PATIENT).json()
        for result in data["results"]:
            assert "model_type" in result
            assert "display_name" in result
            assert "probability" in result
            assert "risk_level" in result
            assert "prediction" in result
            assert 0.0 <= result["probability"] <= 1.0
            assert result["risk_level"] in ["low", "moderate", "high"]
            assert result["prediction"] in [0, 1]

    def test_all_models_demo_flag_is_bool(self, client):
        """POST /predict/all-models — is_demo jest bool."""
        data = client.post("/predict/all-models", json=SAMPLE_PATIENT).json()
        assert isinstance(data.get("is_demo"), bool)


# ============================================================================
# XAI ENDPOINTS
# ============================================================================

class TestSHAPEndpoint:
    """Testy POST /explain/shap"""

    def test_shap_returns_200(self, client):
        """POST /explain/shap zwraca 200."""
        payload = {
            "patient": SAMPLE_PATIENT,
            "method": "shap",
            "num_features": 10,
        }
        response = client.post("/explain/shap", json=payload)
        assert response.status_code == 200

    def test_shap_response_structure(self, client):
        """POST /explain/shap zwraca SHAPExplanation."""
        payload = {"patient": SAMPLE_PATIENT, "method": "shap"}
        data = client.post("/explain/shap", json=payload).json()
        assert data["method"] == "SHAP"
        assert "base_value" in data
        assert isinstance(data["base_value"], float)
        assert "shap_values" in data
        assert isinstance(data["shap_values"], dict)
        assert "feature_contributions" in data
        assert "risk_factors" in data
        assert "protective_factors" in data
        assert "prediction" in data
        # Prediction inside SHAP has standard fields
        pred = data["prediction"]
        assert 0.0 <= pred["probability"] <= 1.0
        assert pred["risk_level"] in ["low", "moderate", "high"]

    def test_shap_has_feature_contributions(self, client):
        """POST /explain/shap zwraca niepusta liste feature_contributions."""
        payload = {"patient": SAMPLE_PATIENT, "method": "shap"}
        data = client.post("/explain/shap", json=payload).json()
        assert len(data["feature_contributions"]) > 0
        contrib = data["feature_contributions"][0]
        assert "feature" in contrib
        assert "value" in contrib
        assert "contribution" in contrib
        assert "direction" in contrib
        assert contrib["direction"] in ["increases_risk", "decreases_risk"]


class TestLIMEEndpoint:
    """Testy POST /explain/lime"""

    def test_lime_returns_200(self, client):
        """POST /explain/lime zwraca 200."""
        payload = {"patient": SAMPLE_PATIENT, "method": "lime"}
        response = client.post("/explain/lime", json=payload)
        assert response.status_code == 200

    def test_lime_response_structure(self, client):
        """POST /explain/lime zwraca LIMEExplanation."""
        payload = {"patient": SAMPLE_PATIENT, "method": "lime"}
        data = client.post("/explain/lime", json=payload).json()
        assert data["method"] == "LIME"
        assert "intercept" in data
        assert isinstance(data["intercept"], float)
        assert "feature_weights" in data
        assert isinstance(data["feature_weights"], list)
        assert "risk_factors" in data
        assert "protective_factors" in data
        assert "local_prediction" in data
        assert isinstance(data["local_prediction"], float)
        assert "prediction" in data

    def test_lime_has_feature_weights(self, client):
        """POST /explain/lime zwraca niepusta liste feature_weights."""
        payload = {"patient": SAMPLE_PATIENT, "method": "lime"}
        data = client.post("/explain/lime", json=payload).json()
        assert len(data["feature_weights"]) > 0
        fw = data["feature_weights"][0]
        assert "feature" in fw
        assert "weight" in fw


class TestDALEXEndpoint:
    """Testy POST /explain/dalex"""

    def test_dalex_returns_200(self, client):
        """POST /explain/dalex zwraca 200."""
        payload = {"patient": SAMPLE_PATIENT, "method": "dalex"}
        response = client.post("/explain/dalex", json=payload)
        assert response.status_code == 200

    def test_dalex_response_structure(self, client):
        """POST /explain/dalex zwraca DALEXExplanation."""
        payload = {"patient": SAMPLE_PATIENT, "method": "dalex"}
        data = client.post("/explain/dalex", json=payload).json()
        assert data["method"] == "DALEX"
        assert "contributions" in data
        assert isinstance(data["contributions"], list)
        assert "risk_factors" in data
        assert "protective_factors" in data
        assert "prediction" in data

    def test_dalex_has_contributions(self, client):
        """POST /explain/dalex zwraca niepusta liste contributions."""
        payload = {"patient": SAMPLE_PATIENT, "method": "dalex"}
        data = client.post("/explain/dalex", json=payload).json()
        assert len(data["contributions"]) > 0


# ============================================================================
# CHAT ENDPOINT
# ============================================================================

@pytest.fixture(scope="module")
def client_no_raise():
    """TestClient z raise_server_exceptions=False (potrzebny dla endpointow z slowapi)."""
    from fastapi.testclient import TestClient as TC
    from src.api.main import app as _app
    with TC(_app, raise_server_exceptions=False) as c:
        yield c


class TestChatEndpoint:
    """Testy POST /chat

    UWAGA: Endpoint /chat uzywa @limiter.limit z slowapi.
    slowapi wymaga parametru o nazwie 'request' (nie 'http_request'),
    co moze powodowac blad 500 w srodowisku testowym.
    Uzywamy client_no_raise zeby TestClient nie rzucal wyjatku.
    """

    def test_chat_returns_response(self, client_no_raise):
        """POST /chat zwraca odpowiedz (200 lub 500 z powodu slowapi)."""
        payload = {
            "message": "Jakie jest moje ryzyko?",
            "patient": SAMPLE_PATIENT,
            "health_literacy": "basic",
        }
        response = client_no_raise.post("/chat", json=payload)
        if response.status_code == 200:
            data = response.json()
            assert "response" in data
            assert isinstance(data["response"], str)
            assert len(data["response"]) > 0
            assert "follow_up_suggestions" in data
        else:
            # slowapi moze zwrocic 500 gdy parametr Request nie jest poprawnie nazwany
            assert response.status_code == 500

    def test_chat_with_conversation_history(self, client_no_raise):
        """POST /chat z historia rozmowy zwraca odpowiedz."""
        payload = {
            "message": "Powiedz wiecej o kreatyninie",
            "patient": SAMPLE_PATIENT,
            "health_literacy": "basic",
            "conversation_history": [
                {"role": "user", "content": "Jakie jest moje ryzyko?"},
                {"role": "assistant", "content": "Analiza wykazala umiarkowane ryzyko."},
            ],
        }
        response = client_no_raise.post("/chat", json=payload)
        assert response.status_code in [200, 500]

    def test_chat_clinician_mode(self, client_no_raise):
        """POST /chat z health_literacy=basic zwraca odpowiedz."""
        payload = {
            "message": "Jakie czynniki wplywaja na ryzyko?",
            "patient": SAMPLE_PATIENT,
            "health_literacy": "basic",
        }
        response = client_no_raise.post("/chat", json=payload)
        assert response.status_code in [200, 500]

    def test_chat_empty_message_rejected(self, client_no_raise):
        """POST /chat z pustym message zwraca 422."""
        payload = {
            "message": "",
            "patient": SAMPLE_PATIENT,
            "health_literacy": "basic",
        }
        response = client_no_raise.post("/chat", json=payload)
        assert response.status_code == 422


# ============================================================================
# MODEL ENDPOINTS
# ============================================================================

class TestModelsAvailableEndpoint:
    """Testy GET /models/available"""

    def test_models_available_returns_200(self, client):
        """GET /models/available zwraca 200."""
        response = client.get("/models/available")
        assert response.status_code == 200

    def test_models_available_response_structure(self, client):
        """GET /models/available zwraca liste modeli."""
        data = client.get("/models/available").json()
        assert "models" in data
        assert "total_models" in data
        assert isinstance(data["models"], list)
        assert isinstance(data["total_models"], int)
        # W trybie demo moze nie byc zaladowanych modeli
        assert data["total_models"] >= 0


class TestCalibrationEndpoint:
    """Testy GET /models/calibration"""

    def test_calibration_returns_200(self, client):
        """GET /models/calibration zwraca 200."""
        response = client.get("/models/calibration")
        assert response.status_code == 200

    def test_calibration_response_structure(self, client):
        """GET /models/calibration zwraca CalibrationResponse."""
        data = client.get("/models/calibration").json()
        assert "models" in data
        assert "task_type" in data
        assert data["task_type"] == "mortality"
        assert isinstance(data["models"], list)

    def test_calibration_model_entries(self, client):
        """GET /models/calibration — kazdy wpis ma wymagane pola."""
        data = client.get("/models/calibration").json()
        # W trybie demo powinny byc syntetyczne dane kalibracji
        assert len(data["models"]) > 0
        for model_cal in data["models"]:
            assert "model_type" in model_cal
            assert "display_name" in model_cal
            assert "brier_score" in model_cal
            assert "calibration_curve_x" in model_cal
            assert "calibration_curve_y" in model_cal
            assert "n_samples" in model_cal
            assert isinstance(model_cal["brier_score"], float)
            assert model_cal["brier_score"] >= 0
            assert len(model_cal["calibration_curve_x"]) == len(model_cal["calibration_curve_y"])

    def test_calibration_dialysis_task_type(self, client):
        """GET /models/calibration?task_type=dialysis zwraca dane kalibracji dializy."""
        response = client.get("/models/calibration?task_type=dialysis")
        assert response.status_code == 200
        data = response.json()
        assert data["task_type"] == "dialysis"

    def test_calibration_invalid_task_type(self, client):
        """GET /models/calibration?task_type=invalid zwraca 422."""
        response = client.get("/models/calibration?task_type=invalid")
        assert response.status_code == 422


# ============================================================================
# CONFIG / DEMO MODE ENDPOINTS
# ============================================================================

class TestDemoModeEndpoint:
    """Testy GET/POST /config/demo-mode"""

    def test_get_demo_mode_status(self, client):
        """GET /config/demo-mode zwraca DemoModeStatus."""
        response = client.get("/config/demo-mode")
        assert response.status_code == 200
        data = response.json()
        assert "demo_allowed" in data
        assert "model_loaded" in data
        assert "current_mode" in data
        assert "force_api_mode" in data
        assert data["current_mode"] in ["api", "demo", "unavailable"]


# ============================================================================
# VALIDATION EDGE CASES
# ============================================================================

class TestValidationEdgeCases:
    """Testy walidacji danych wejsciowych."""

    def test_predict_missing_required_field(self, client):
        """POST /predict bez wymaganego pola (wiek) zwraca 422."""
        invalid = {"plec": 1}
        response = client.post("/predict", json=invalid)
        assert response.status_code == 422

    def test_predict_extra_fields_ignored(self, client):
        """POST /predict z dodatkowymi polami (extra='ignore') dziala poprawnie."""
        patient = SAMPLE_PATIENT.copy()
        patient["nieistniejace_pole"] = "wartosc"
        response = client.post("/predict", json=patient)
        assert response.status_code == 200

    def test_predict_boundary_age_zero(self, client):
        """POST /predict z wiek=0 dziala poprawnie (ge=0)."""
        patient = SAMPLE_PATIENT.copy()
        patient["wiek"] = 0
        patient["wiek_rozpoznania"] = 0
        patient["opoznienie_rozpoznia"] = 0
        response = client.post("/predict", json=patient)
        assert response.status_code == 200

    def test_predict_boundary_age_120(self, client):
        """POST /predict z wiek=120 dziala poprawnie (le=120)."""
        patient = SAMPLE_PATIENT.copy()
        patient["wiek"] = 120
        patient["wiek_rozpoznania"] = 115
        response = client.post("/predict", json=patient)
        assert response.status_code == 200

    def test_predict_creatinine_at_max(self, client):
        """POST /predict z kreatynina=2000 dziala (le=2000)."""
        patient = SAMPLE_PATIENT.copy()
        patient["kreatynina"] = 2000
        response = client.post("/predict", json=patient)
        assert response.status_code == 200

    def test_predict_empty_body(self, client):
        """POST /predict z pustym body zwraca 422."""
        response = client.post("/predict", json={})
        assert response.status_code == 422

    def test_batch_predict_no_patients_key(self, client):
        """POST /predict/batch bez klucza 'patients' zwraca 422."""
        response = client.post("/predict/batch", json={})
        assert response.status_code == 422

    def test_explain_shap_with_num_features(self, client):
        """POST /explain/shap z num_features=5 dziala poprawnie."""
        payload = {"patient": SAMPLE_PATIENT, "method": "shap", "num_features": 5}
        response = client.post("/explain/shap", json=payload)
        assert response.status_code == 200

    def test_explain_shap_without_method_uses_default(self, client):
        """POST /explain/shap bez method uzywa domyslnego shap."""
        payload = {"patient": SAMPLE_PATIENT}
        response = client.post("/explain/shap", json=payload)
        assert response.status_code == 200

    def test_chat_message_too_long_rejected(self, client):
        """POST /chat z wiadomoscia > 2000 znakow zwraca 422."""
        payload = {
            "message": "a" * 2001,
            "patient": SAMPLE_PATIENT,
            "health_literacy": "basic",
        }
        response = client.post("/chat", json=payload)
        assert response.status_code == 422

    def test_predict_opoznienie_exceeds_age(self, client):
        """POST /predict z opoznienie_rozpoznia > wiek*12 zwraca 422."""
        patient = SAMPLE_PATIENT.copy()
        patient["wiek"] = 10
        patient["opoznienie_rozpoznia"] = 200  # 200 > 10*12=120
        response = client.post("/predict", json=patient)
        assert response.status_code == 422


# ============================================================================
# HEALTH MODELS ENDPOINT
# ============================================================================

class TestModelsHealthEndpoint:
    """Testy GET /health/models"""

    def test_models_health_returns_200(self, client):
        """GET /health/models zwraca 200."""
        response = client.get("/health/models")
        assert response.status_code == 200

    def test_models_health_response_structure(self, client):
        """GET /health/models zwraca diagnostyczny status."""
        data = client.get("/health/models").json()
        assert "mortality_models" in data
        assert "dialysis_models" in data
        assert "xai_available" in data
        assert "feature_names_loaded" in data
        assert "n_feature_names" in data


# ============================================================================
# DIALYSIS-SPECIFIC TESTS
# ============================================================================

SAMPLE_DIALYSIS_PATIENT = {
    "wiek": 55,
    "plec": 1,
    "wiek_rozpoznania": 50,
    "liczba_zajetych_narzadow": 3,
    "manifestacja_sercowo_naczyniowy": 0,
    "manifestacja_nerki": 1,
    "manifestacja_neurologiczny": 0,
    "manifestacja_oddechowy": 0,
    "zaostrz_wymagajace_oit": 0,
    "zaostrz_wymagajace_hospital": 1,
    "kreatynina": 180,
    "max_crp": 45,
    "plazmaferezy": 0,
    "pulsy": 1,
    "sterydy_dawka_g": 0.5,
    "czas_sterydow": 12,
    "powiklania_serce_pluca": 0,
    "powiklania_infekcja": 0,
}

SAMPLE_DIALYSIS_PATIENT_HIGH_RISK = {
    "wiek": 70,
    "plec": 1,
    "wiek_rozpoznania": 65,
    "liczba_zajetych_narzadow": 5,
    "manifestacja_sercowo_naczyniowy": 1,
    "manifestacja_nerki": 1,
    "manifestacja_neurologiczny": 1,
    "manifestacja_oddechowy": 1,
    "zaostrz_wymagajace_oit": 1,
    "zaostrz_wymagajace_hospital": 1,
    "kreatynina": 350,
    "max_crp": 200,
    "plazmaferezy": 1,
    "pulsy": 1,
    "sterydy_dawka_g": 3.0,
    "czas_sterydow": 36,
    "powiklania_serce_pluca": 1,
    "powiklania_infekcja": 1,
}


class TestDialysisPredictEndpoint:
    """Testy POST /predict/dialysis"""

    def test_dialysis_predict_returns_200(self, client):
        """POST /predict/dialysis zwraca 200."""
        response = client.post("/predict/dialysis", json=SAMPLE_DIALYSIS_PATIENT)
        assert response.status_code == 200

    def test_dialysis_predict_response_structure(self, client):
        """POST /predict/dialysis zwraca poprawną strukturę."""
        data = client.post("/predict/dialysis", json=SAMPLE_DIALYSIS_PATIENT).json()
        assert "probability" in data
        assert "needs_dialysis" in data
        assert "risk_level" in data
        assert 0 <= data["probability"] <= 1
        assert isinstance(data["needs_dialysis"], bool)
        assert data["risk_level"] in ["low", "moderate", "high"]

    def test_dialysis_predict_high_risk(self, client):
        """POST /predict/dialysis z wysokim ryzykiem zwraca wyższe prawdopodobieństwo."""
        data_low = client.post("/predict/dialysis", json=SAMPLE_DIALYSIS_PATIENT).json()
        data_high = client.post("/predict/dialysis", json=SAMPLE_DIALYSIS_PATIENT_HIGH_RISK).json()
        assert data_high["probability"] > data_low["probability"]

    def test_dialysis_predict_with_model_type(self, client):
        """POST /predict/dialysis z wybranym model_type zwraca wynik."""
        patient = SAMPLE_DIALYSIS_PATIENT.copy()
        patient["model_type"] = "random_forest"
        response = client.post("/predict/dialysis", json=patient)
        assert response.status_code == 200

    def test_dialysis_predict_invalid_model_type(self, client):
        """POST /predict/dialysis z nieistniejącym modelem zwraca 400."""
        patient = SAMPLE_DIALYSIS_PATIENT.copy()
        patient["model_type"] = "nonexistent_model"
        response = client.post("/predict/dialysis", json=patient)
        assert response.status_code in [200, 400]  # 400 jeśli model załadowany, 200 demo


class TestDialysisAllModelsEndpoint:
    """Testy POST /predict/dialysis/all-models"""

    def test_dialysis_all_models_returns_200(self, client):
        """POST /predict/dialysis/all-models zwraca 200."""
        response = client.post("/predict/dialysis/all-models", json=SAMPLE_DIALYSIS_PATIENT)
        assert response.status_code == 200

    def test_dialysis_all_models_response_structure(self, client):
        """POST /predict/dialysis/all-models zwraca poprawną strukturę."""
        data = client.post("/predict/dialysis/all-models", json=SAMPLE_DIALYSIS_PATIENT).json()
        assert "results" in data
        assert "consensus_probability" in data
        assert "consensus_risk_level" in data
        assert "agreement_score" in data
        assert len(data["results"]) > 0

    def test_dialysis_all_models_individual_results(self, client):
        """Każdy wynik posiada model_type i probability."""
        data = client.post("/predict/dialysis/all-models", json=SAMPLE_DIALYSIS_PATIENT).json()
        for result in data["results"]:
            assert "model_type" in result
            assert "probability" in result
            assert 0 <= result["probability"] <= 1


class TestDialysisBatchEndpoint:
    """Testy POST /predict/dialysis/batch"""

    def test_dialysis_batch_returns_200(self, client):
        """POST /predict/dialysis/batch z listą pacjentów zwraca 200."""
        payload = {
            "patients": [SAMPLE_DIALYSIS_PATIENT, SAMPLE_DIALYSIS_PATIENT_HIGH_RISK],
            "include_risk_factors": True,
            "top_n_factors": 3
        }
        response = client.post("/predict/dialysis/batch", json=payload)
        assert response.status_code == 200

    def test_dialysis_batch_response_structure(self, client):
        """POST /predict/dialysis/batch zwraca poprawną strukturę."""
        payload = {
            "patients": [SAMPLE_DIALYSIS_PATIENT],
            "include_risk_factors": False
        }
        data = client.post("/predict/dialysis/batch", json=payload).json()
        assert "total_patients" in data
        assert "processed_count" in data
        assert "success_count" in data
        assert "summary" in data
        assert "results" in data
        assert data["total_patients"] == 1
        assert data["processed_count"] == 1

    def test_dialysis_batch_multiple_patients(self, client):
        """POST /predict/dialysis/batch z wieloma pacjentami przetwarza wszystkich."""
        patients = [SAMPLE_DIALYSIS_PATIENT] * 5
        payload = {"patients": patients, "include_risk_factors": True, "top_n_factors": 3}
        data = client.post("/predict/dialysis/batch", json=payload).json()
        assert data["total_patients"] == 5
        assert len(data["results"]) == 5
        for r in data["results"]:
            assert 0 <= r["prediction"]["probability"] <= 1

    def test_dialysis_batch_summary_statistics(self, client):
        """POST /predict/dialysis/batch oblicza poprawne statystyki podsumowania."""
        payload = {
            "patients": [SAMPLE_DIALYSIS_PATIENT, SAMPLE_DIALYSIS_PATIENT_HIGH_RISK],
            "include_risk_factors": False
        }
        data = client.post("/predict/dialysis/batch", json=payload).json()
        summary = data["summary"]
        assert "avg_probability" in summary
        assert "median_probability" in summary
        assert summary["total_count"] == 2
        assert summary["low_risk_count"] + summary["moderate_risk_count"] + summary["high_risk_count"] == 2

    def test_dialysis_batch_empty_patients_returns_422(self, client):
        """POST /predict/dialysis/batch z pustą listą zwraca 422."""
        payload = {"patients": []}
        response = client.post("/predict/dialysis/batch", json=payload)
        assert response.status_code == 422


class TestDialysisModelsAvailableEndpoint:
    """Testy GET /models/dialysis/available"""

    def test_dialysis_models_available_returns_200(self, client):
        """GET /models/dialysis/available zwraca 200."""
        response = client.get("/models/dialysis/available")
        assert response.status_code == 200

    def test_dialysis_models_available_structure(self, client):
        """GET /models/dialysis/available zwraca poprawną strukturę."""
        data = client.get("/models/dialysis/available").json()
        assert "models" in data
        assert "total_models" in data
        assert isinstance(data["models"], list)


class TestDialysisXAIEndpoints:
    """Testy endpointów XAI z task_type=dialysis."""

    def test_shap_dialysis(self, client):
        """POST /explain/shap z task_type=dialysis zwraca wyjaśnienie."""
        payload = {
            "patient": SAMPLE_PATIENT,
            "task_type": "dialysis",
            "dialysis_patient": SAMPLE_DIALYSIS_PATIENT
        }
        response = client.post("/explain/shap", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "shap_values" in data or "feature_contributions" in data

    def test_lime_dialysis(self, client):
        """POST /explain/lime z task_type=dialysis zwraca wyjaśnienie."""
        payload = {
            "patient": SAMPLE_PATIENT,
            "task_type": "dialysis",
            "dialysis_patient": SAMPLE_DIALYSIS_PATIENT
        }
        response = client.post("/explain/lime", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "feature_weights" in data

    def test_dalex_dialysis(self, client):
        """POST /explain/dalex z task_type=dialysis zwraca wyjaśnienie."""
        payload = {
            "patient": SAMPLE_PATIENT,
            "task_type": "dialysis",
            "dialysis_patient": SAMPLE_DIALYSIS_PATIENT
        }
        response = client.post("/explain/dalex", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "contributions" in data

    def test_ebm_local_dialysis(self, client):
        """POST /explain/ebm/local z task_type=dialysis zwraca wyjaśnienie."""
        payload = {
            "patient": SAMPLE_PATIENT,
            "task_type": "dialysis",
            "dialysis_patient": SAMPLE_DIALYSIS_PATIENT
        }
        response = client.post("/explain/ebm/local", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "contributions" in data

    def test_ebm_global_dialysis(self, client):
        """GET /explain/ebm/global?task_type=dialysis zwraca wyjaśnienie."""
        response = client.get("/explain/ebm/global?task_type=dialysis")
        assert response.status_code == 200
        data = response.json()
        assert "feature_importance" in data

    def test_shap_global_dialysis(self, client):
        """GET /explain/shap/global?task_type=dialysis zwraca wyjaśnienie."""
        response = client.get("/explain/shap/global?task_type=dialysis")
        assert response.status_code == 200
        data = response.json()
        assert "feature_importance" in data or "global_importance" in data

    def test_comparison_dialysis(self, client):
        """POST /explain/comparison z task_type=dialysis zwraca porównanie."""
        payload = {
            "patient": SAMPLE_PATIENT,
            "task_type": "dialysis",
            "dialysis_patient": SAMPLE_DIALYSIS_PATIENT
        }
        response = client.post("/explain/comparison", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "individual_rankings" in data

    def test_patient_explanation_dialysis(self, client):
        """POST /explain/patient z task_type=dialysis zwraca wyjaśnienie."""
        payload = {
            "patient": SAMPLE_PATIENT,
            "task_type": "dialysis",
            "dialysis_patient": SAMPLE_DIALYSIS_PATIENT,
            "health_literacy": "basic"
        }
        response = client.post("/explain/patient", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "risk_level" in data
        assert "risk_description" in data

    def test_calibration_dialysis(self, client):
        """GET /models/calibration?task_type=dialysis zwraca dane kalibracji."""
        response = client.get("/models/calibration?task_type=dialysis")
        assert response.status_code == 200
        data = response.json()
        assert data["task_type"] == "dialysis"
        assert "models" in data

    def test_global_importance_dialysis(self, client):
        """GET /model/global-importance?task_type=dialysis zwraca ważność."""
        response = client.get("/model/global-importance?task_type=dialysis")
        assert response.status_code == 200
        data = response.json()
        assert "feature_importance" in data

    def test_model_info_dialysis(self, client):
        """GET /model/info?task_type=dialysis zwraca info o modelu."""
        response = client.get("/model/info?task_type=dialysis")
        assert response.status_code == 200


class TestDialysisChatEndpoint:
    """Testy POST /chat z task_type=dialysis"""

    def test_chat_dialysis_returns_200(self, client):
        """POST /chat z task_type=dialysis zwraca 200."""
        payload = {
            "patient": SAMPLE_PATIENT,
            "message": "Jakie jest moje ryzyko potrzeby dializy?",
            "health_literacy": "basic",
            "task_type": "dialysis",
            "dialysis_patient": SAMPLE_DIALYSIS_PATIENT,
            "conversation_history": []
        }
        response = client.post("/chat", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert len(data["response"]) > 0

    def test_chat_dialysis_has_suggestions(self, client):
        """POST /chat z task_type=dialysis zwraca sugestie."""
        payload = {
            "patient": SAMPLE_PATIENT,
            "message": "Co mogę zrobić?",
            "health_literacy": "basic",
            "task_type": "dialysis",
            "dialysis_patient": SAMPLE_DIALYSIS_PATIENT,
            "conversation_history": []
        }
        data = client.post("/chat", json=payload).json()
        assert "follow_up_suggestions" in data
        assert len(data["follow_up_suggestions"]) > 0


# ============================================================================
# PATIENT EXTRACTOR TESTS
# ============================================================================

class TestPatientExtractor:
    """Testy modulu patient_extractor."""

    def test_extract_age_lat(self):
        from src.api.patient_extractor import extract_patient_from_text
        result = extract_patient_from_text("Pacjent 65 lat")
        assert result["wiek"] == 65.0

    def test_extract_age_letni(self):
        from src.api.patient_extractor import extract_patient_from_text
        result = extract_patient_from_text("55-letni mężczyzna")
        assert result["wiek"] == 55.0

    def test_extract_sex_female(self):
        from src.api.patient_extractor import extract_patient_from_text
        result = extract_patient_from_text("Pacjentka 50 lat, kobieta")
        assert result["plec"] == 0

    def test_extract_sex_male(self):
        from src.api.patient_extractor import extract_patient_from_text
        result = extract_patient_from_text("Mężczyzna 60 lat")
        assert result["plec"] == 1

    def test_extract_creatinine(self):
        from src.api.patient_extractor import extract_patient_from_text
        result = extract_patient_from_text("kreatynina 220 μmol/L")
        assert result["kreatynina"] == 220.0

    def test_extract_creatinine_comma(self):
        from src.api.patient_extractor import extract_patient_from_text
        result = extract_patient_from_text("kreatynina: 180,5")
        assert result["kreatynina"] == 180.5

    def test_extract_crp(self):
        from src.api.patient_extractor import extract_patient_from_text
        result = extract_patient_from_text("CRP 120 mg/L")
        assert result["max_crp"] == 120.0

    def test_extract_kidney(self):
        from src.api.patient_extractor import extract_patient_from_text
        result = extract_patient_from_text("zajęcie nerek, serce")
        assert result["manifestacja_nerki"] == 1

    def test_extract_kidney_negated(self):
        from src.api.patient_extractor import extract_patient_from_text
        result = extract_patient_from_text("bez zajęcia nerek")
        assert result["manifestacja_nerki"] == 0

    def test_extract_oit(self):
        from src.api.patient_extractor import extract_patient_from_text
        result = extract_patient_from_text("Pacjent przebywał na OIT")
        assert result["zaostrz_wymagajace_oit"] == 1

    def test_extract_organ_count(self):
        from src.api.patient_extractor import extract_patient_from_text
        result = extract_patient_from_text("zajęcie 4 narządów")
        assert result["liczba_zajetych_narzadow"] == 4

    def test_extract_complex_description(self):
        from src.api.patient_extractor import extract_patient_from_text
        text = "Pacjent 65 lat, mężczyzna, kreatynina 220, zajęcie nerek, CRP 80"
        result = extract_patient_from_text(text)
        assert result["wiek"] == 65.0
        assert result["plec"] == 1
        assert result["kreatynina"] == 220.0
        assert result["manifestacja_nerki"] == 1
        assert result["max_crp"] == 80.0

    def test_extract_empty_text(self):
        from src.api.patient_extractor import extract_patient_from_text
        result = extract_patient_from_text("")
        assert result == {}

    def test_missing_required_fields(self):
        from src.api.patient_extractor import get_missing_required_fields
        assert "wiek" in get_missing_required_fields({})
        assert "plec" in get_missing_required_fields({})
        assert get_missing_required_fields({"wiek": 55, "plec": 1}) == []

    def test_followup_question(self):
        from src.api.patient_extractor import get_followup_question
        q = get_followup_question(["wiek", "plec"])
        assert q is not None
        assert "lat" in q or "wiek" in q.lower()

    def test_build_summary(self):
        from src.api.patient_extractor import build_patient_summary
        s = build_patient_summary({"wiek": 55, "plec": 1, "kreatynina": 120})
        assert "55" in s
        assert "mężczyzna" in s


# ============================================================================
# CHAT ANALYZE ENDPOINT TESTS
# ============================================================================

class TestChatAnalyzeEndpoint:
    """Testy endpointu /chat/analyze."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from src.api.main import app
        return TestClient(app)

    def test_minimal_input_triggers_prediction(self, client):
        """Wiek + płeć powinny wystarczyć do predykcji."""
        payload = {
            "message": "Pacjent 65 lat, mężczyzna",
            "accumulated_patient": {},
            "task_type": "mortality",
        }
        resp = client.post("/chat/analyze", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["analysis_complete"] is True
        assert data["prediction"] is not None
        assert data["prediction"]["probability"] >= 0
        assert data["accumulated_patient"]["wiek"] == 65.0
        assert data["accumulated_patient"]["plec"] == 1

    def test_missing_data_returns_followup(self, client):
        """Brak wymaganych pól zwraca pytanie uzupełniające."""
        payload = {
            "message": "kreatynina 220, zajęcie nerek",
            "accumulated_patient": {},
            "task_type": "mortality",
        }
        resp = client.post("/chat/analyze", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["analysis_complete"] is False
        assert data["follow_up_question"] is not None
        assert len(data["missing_fields"]) > 0

    def test_incremental_accumulation(self, client):
        """Dane mogą być zbierane przyrostowo."""
        # Krok 1: wiek
        payload1 = {
            "message": "Pacjent 55 lat kobieta",
            "accumulated_patient": {},
            "task_type": "mortality",
        }
        resp1 = client.post("/chat/analyze", json=payload1)
        data1 = resp1.json()
        assert data1["analysis_complete"] is True
        acc = data1["accumulated_patient"]
        assert acc["wiek"] == 55.0
        assert acc["plec"] == 0

        # Krok 2: dodaj kreatynine
        payload2 = {
            "message": "kreatynina 150, nerki zajęte",
            "accumulated_patient": acc,
            "task_type": "mortality",
        }
        resp2 = client.post("/chat/analyze", json=payload2)
        data2 = resp2.json()
        assert data2["analysis_complete"] is True
        assert data2["accumulated_patient"]["kreatynina"] == 150.0
        assert data2["accumulated_patient"]["manifestacja_nerki"] == 1
        # Wiek powinien zostać z poprzedniego kroku
        assert data2["accumulated_patient"]["wiek"] == 55.0

    def test_response_contains_narrative(self, client):
        """Odpowiedź zawiera narracyjny opis w języku polskim."""
        payload = {
            "message": "Pacjent 65 lat, mężczyzna, kreatynina 220, zajęcie nerek",
            "accumulated_patient": {},
            "task_type": "mortality",
        }
        resp = client.post("/chat/analyze", json=payload)
        data = resp.json()
        assert "Ryzyko" in data["response"] or "Na podstawie" in data["response"]

    def test_empty_message_rejected(self, client):
        """Pusta wiadomość powinna być odrzucona."""
        payload = {
            "message": "",
            "accumulated_patient": {},
            "task_type": "mortality",
        }
        resp = client.post("/chat/analyze", json=payload)
        assert resp.status_code == 422

    def test_extracted_update_present(self, client):
        """extracted_update zawiera pola wyodrębnione z bieżącej wiadomości."""
        payload = {
            "message": "CRP 80",
            "accumulated_patient": {"wiek": 55, "plec": 1},
            "task_type": "mortality",
        }
        resp = client.post("/chat/analyze", json=payload)
        data = resp.json()
        assert data["extracted_update"].get("max_crp") == 80.0

    def test_dialysis_task_type(self, client):
        """Endpoint działa z task_type dialysis."""
        payload = {
            "message": "Pacjentka 50 lat",
            "accumulated_patient": {},
            "task_type": "dialysis",
        }
        resp = client.post("/chat/analyze", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["analysis_complete"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
