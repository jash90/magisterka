"""
Testy jednostkowe dla API FastAPI.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.schemas import (
    PatientInput, PredictionOutput, RiskLevel, HealthLiteracyLevel,
    AnalysisRequest, AnalysisOutput, BatchAnalysisInput,
    patient_to_array, get_risk_level_from_probability
)


class TestSchemas:
    """Testy dla schematów Pydantic."""

    def test_patient_input_valid(self):
        """Test poprawnych danych pacjenta."""
        patient = PatientInput(
            wiek_rozpoznania=50,
            opoznienie_rozpoznia=6,
            manifestacja_miesno_szkiel=0,
            manifestacja_skora=1,
            manifestacja_wzrok=0,
            manifestacja_nos_ucho_gardlo=0,
            manifestacja_oddechowy=1,
            manifestacja_sercowo_naczyniowy=0,
            manifestacja_pokarmowy=0,
            manifestacja_moczowo_plciowy=0,
            manifestacja_zajecie_csn=0,
            manifestacja_neurologiczny=1,
            liczba_zajetych_narzadow=3,
            zaostrz_wymagajace_hospital=1,
            zaostrz_wymagajace_oit=0,
            kreatynina=120.0,
            czas_sterydow=12,
            plazmaferezy=0,
            eozynofilia_krwi_obwodowej_wartosc=8.5,
            powiklania_neurologiczne=0
        )

        assert patient.wiek_rozpoznania == 50
        assert patient.manifestacja_skora == 1
        assert patient.liczba_zajetych_narzadow == 3

    def test_patient_input_defaults(self):
        """Test wartości domyślnych."""
        patient = PatientInput()

        assert patient.wiek_rozpoznania == 50.0
        assert patient.liczba_zajetych_narzadow == 0
        assert patient.manifestacja_oddechowy == 0
        assert patient.kreatynina == 100.0
        assert patient.eozynofilia_krwi_obwodowej_wartosc == 0.0
        assert patient.powiklania_neurologiczne == 0

    def test_patient_input_validation_age(self):
        """Test walidacji wieku rozpoznania."""
        with pytest.raises(ValueError):
            PatientInput(wiek_rozpoznania=-5)

        with pytest.raises(ValueError):
            PatientInput(wiek_rozpoznania=150)

    def test_patient_input_validation_binary(self):
        """Test walidacji pól binarnych."""
        with pytest.raises(ValueError):
            PatientInput(manifestacja_skora=2)

        with pytest.raises(ValueError):
            PatientInput(plazmaferezy=-1)

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
        """Test predykcji z przedziałem ufności."""
        prediction = PredictionOutput(
            probability=0.5,
            risk_level=RiskLevel.MODERATE,
            prediction=0,
            confidence_interval={"lower": 0.4, "upper": 0.6}
        )

        assert prediction.confidence_interval["lower"] == 0.4
        assert prediction.confidence_interval["upper"] == 0.6

    def test_get_risk_level_from_probability(self):
        """Test konwersji prawdopodobieństwa na poziom ryzyka."""
        assert get_risk_level_from_probability(0.1) == RiskLevel.LOW
        assert get_risk_level_from_probability(0.29) == RiskLevel.LOW
        assert get_risk_level_from_probability(0.3) == RiskLevel.MODERATE
        assert get_risk_level_from_probability(0.5) == RiskLevel.MODERATE
        assert get_risk_level_from_probability(0.69) == RiskLevel.MODERATE
        assert get_risk_level_from_probability(0.7) == RiskLevel.HIGH
        assert get_risk_level_from_probability(0.95) == RiskLevel.HIGH

    def test_patient_to_array(self):
        """Test konwersji pacjenta do tablicy (20 cech)."""
        patient = PatientInput(
            wiek_rozpoznania=55,
            kreatynina=120.0,
            manifestacja_oddechowy=1,
            liczba_zajetych_narzadow=3
        )

        feature_order = [
            'Wiek_rozpoznania', 'Opoznienie_Rozpoznia',
            'Manifestacja_Miesno-Szkiel', 'Manifestacja_Skora',
            'Manifestacja_Wzrok', 'Manifestacja_Nos/Ucho/Gardlo',
            'Manifestacja_Oddechowy', 'Manifestacja_Sercowo-Naczyniowy',
            'Manifestacja_Pokarmowy', 'Manifestacja_Moczowo-Plciowy',
            'Manifestacja_Zajecie_CSN', 'Manifestacja_Neurologiczny',
            'Liczba_Zajetych_Narzadow', 'Zaostrz_Wymagajace_Hospital',
            'Zaostrz_Wymagajace_OIT', 'Kreatynina',
            'Czas_Sterydow', 'Plazmaferezy',
            'Eozynofilia_Krwi_Obwodowej_Wartosc', 'Powiklania_Neurologiczne'
        ]

        arr = patient_to_array(patient, feature_order)

        assert len(arr) == 20
        assert arr[0] == 55    # Wiek_rozpoznania
        assert arr[6] == 1     # Manifestacja_Oddechowy
        assert arr[12] == 3    # Liczba_Zajetych_Narzadow
        assert arr[15] == 120.0  # Kreatynina

    def test_patient_to_array_partial(self):
        """Test konwersji z podzbiorem cech."""
        patient = PatientInput(
            wiek_rozpoznania=60,
            kreatynina=150.0
        )

        feature_order = ['Wiek_rozpoznania', 'Kreatynina']
        arr = patient_to_array(patient, feature_order)

        assert len(arr) == 2
        assert arr[0] == 60
        assert arr[1] == 150.0

    def test_health_literacy_levels(self):
        """Test poziomów health literacy."""
        assert HealthLiteracyLevel.BASIC.value == "basic"
        assert HealthLiteracyLevel.ADVANCED.value == "advanced"
        assert HealthLiteracyLevel.CLINICIAN.value == "clinician"

    def test_risk_level_enum(self):
        """Test enum poziomów ryzyka."""
        assert RiskLevel.LOW.value == "low"
        assert RiskLevel.MODERATE.value == "moderate"
        assert RiskLevel.HIGH.value == "high"

    def test_analysis_request(self):
        """Test schematu AnalysisRequest."""
        request = AnalysisRequest(
            patient=PatientInput(wiek_rozpoznania=55, kreatynina=120.0),
            external_probability=0.65
        )

        assert request.external_probability == 0.65
        assert request.patient.wiek_rozpoznania == 55

    def test_analysis_request_probability_bounds(self):
        """Test walidacji external_probability."""
        with pytest.raises(ValueError):
            AnalysisRequest(
                patient=PatientInput(),
                external_probability=1.5
            )

        with pytest.raises(ValueError):
            AnalysisRequest(
                patient=PatientInput(),
                external_probability=-0.1
            )

    def test_batch_analysis_input(self):
        """Test schematu BatchAnalysisInput."""
        batch = BatchAnalysisInput(
            patients=[PatientInput(), PatientInput()],
            external_probabilities=[0.3, 0.7]
        )

        assert len(batch.patients) == 2
        assert len(batch.external_probabilities) == 2

    def test_batch_analysis_length_mismatch(self):
        """Test walidacji: długość patients != external_probabilities."""
        with pytest.raises(ValueError):
            BatchAnalysisInput(
                patients=[PatientInput(), PatientInput()],
                external_probabilities=[0.3]
            )

    def test_batch_analysis_probability_validation(self):
        """Test walidacji prawdopodobieństw w batch."""
        with pytest.raises(ValueError):
            BatchAnalysisInput(
                patients=[PatientInput()],
                external_probabilities=[1.5]
            )


class TestAPIEndpoints:
    """Testy dla endpointów API (bez uruchamiania serwera)."""

    def test_import_api_main(self):
        """Test importu modułu API."""
        from src.api.main import app
        assert app is not None

    def test_api_routes_exist(self):
        """Test że wymagane ścieżki istnieją."""
        from src.api.main import app

        routes = [route.path for route in app.routes]

        assert "/" in routes
        assert "/health" in routes
        assert "/analyze" in routes
        assert "/explain/shap" in routes
        assert "/explain/lime" in routes
        assert "/explain/patient" in routes
        assert "/model/info" in routes
        assert "/chat" in routes

    def test_analyze_route_exists(self):
        """Test że /analyze route istnieje."""
        from src.api.main import app

        routes = [route.path for route in app.routes]
        assert "/analyze" in routes

    def test_batch_analyze_route_exists(self):
        """Test że /analyze/batch route istnieje."""
        from src.api.main import app

        routes = [route.path for route in app.routes]
        assert "/analyze/batch" in routes


# Testy integracyjne (wymagają uruchomionego serwera)
@pytest.mark.skipif(True, reason="Wymaga uruchomionego serwera API")
class TestAPIIntegration:
    """Testy integracyjne API."""

    @pytest.fixture
    def client(self):
        """Fixture dla klienta testowego."""
        from fastapi.testclient import TestClient
        from src.api.main import app
        return TestClient(app)

    def test_health_check(self, client):
        """Test health check."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_analyze(self, client):
        """Test analizy z external_probability."""
        request_data = {
            "patient": {
                "wiek_rozpoznania": 55,
                "kreatynina": 120.0,
                "manifestacja_oddechowy": 1,
                "liczba_zajetych_narzadow": 3
            },
            "external_probability": 0.65
        }

        response = client.post("/analyze", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert "external_probability" in data
        assert "risk_level" in data
        assert data["external_probability"] == 0.65

    def test_explain_shap(self, client):
        """Test wyjaśnienia SHAP."""
        request_data = {
            "patient": {
                "wiek_rozpoznania": 55,
                "kreatynina": 120.0
            },
            "external_probability": 0.5,
            "method": "shap",
            "num_features": 5
        }

        response = client.post("/explain/shap", json=request_data)
        assert response.status_code == 200


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
