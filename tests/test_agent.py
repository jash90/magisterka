"""
Testy jednostkowe dla modułu agenta konwersacyjnego.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.guardrails import GuardrailsChecker
from src.agent.prompts import (
    get_risk_level, translate_feature, format_risk_factors_patient,
    FEATURE_TRANSLATIONS, RISK_LEVEL_DESCRIPTIONS
)


class TestGuardrailsChecker:
    """Testy dla klasy GuardrailsChecker."""

    @pytest.fixture
    def checker(self):
        """Fixture dla checkera."""
        return GuardrailsChecker()

    def test_init(self, checker):
        """Test inicjalizacji."""
        assert len(checker.harmful_patterns) > 0
        assert len(checker.medical_advice_patterns) > 0
        assert len(checker.forbidden_patterns) > 0

    def test_detect_suicide_content(self, checker):
        """Test wykrywania treści o samobójstwie."""
        queries = [
            "Chcę popełnić samobójstwo",
            "Myślę o samookaleczeniu",
            "Nie chcę już żyć"
        ]

        for query in queries:
            response = checker.check_query(query)
            assert response is not None
            assert "116 123" in response  # Telefon zaufania

    def test_detect_medication_request(self, checker):
        """Test wykrywania prośby o leki."""
        queries = [
            "Jaki lek powinienem brać?",
            "Przepisz mi coś na ból",
            "Czy mogę zwiększyć dawkę?"
        ]

        for query in queries:
            response = checker.check_query(query)
            assert response is not None
            assert "lekarz" in response.lower()

    def test_detect_diagnosis_request(self, checker):
        """Test wykrywania prośby o diagnozę."""
        queries = [
            "Zdiagnozuj mnie",
            "Czy mam raka?",
            "Co mi jest?"
        ]

        for query in queries:
            response = checker.check_query(query)
            assert response is not None
            assert "diagnoz" in response.lower() or "lekarz" in response.lower()

    def test_detect_prognosis_request(self, checker):
        """Test wykrywania prośby o prognozę."""
        queries = [
            "Ile będę jeszcze żyć?",
            "Kiedy umrę?",
            "Jaka jest moja szansa na przeżycie?"
        ]

        for query in queries:
            response = checker.check_query(query)
            assert response is not None

    def test_safe_query_passes(self, checker):
        """Test że bezpieczne zapytania przechodzą."""
        safe_queries = [
            "Wyjaśnij mi wyniki analizy",
            "Co oznacza wysoki CRP?",
            "Jakie są czynniki ryzyka?",
            "Dziękuję za informację"
        ]

        for query in safe_queries:
            response = checker.check_query(query)
            assert response is None

    def test_validate_response_removes_forbidden(self, checker):
        """Test usuwania zakazanych treści z odpowiedzi."""
        # Odpowiedź z zakazanym wzorcem
        response = "Masz 50% szans na przeżycie w ciągu 5 lat."

        validated = checker.validate_response(response)

        assert "50%" not in validated or "usunięta" in validated.lower()

    def test_detect_emotional_distress(self, checker):
        """Test wykrywania stresu emocjonalnego."""
        distress_queries = [
            "Boję się wyników",
            "Jestem przerażona diagnozą",
            "Nie radzę sobie z chorobą"
        ]

        for query in distress_queries:
            assert checker.detect_emotional_distress(query) is True

        normal_queries = [
            "Jakie są wyniki?",
            "Wyjaśnij mi czynniki ryzyka"
        ]

        for query in normal_queries:
            assert checker.detect_emotional_distress(query) is False

    def test_health_literacy_check(self, checker):
        """Test sprawdzania odpowiedniości dla health literacy."""
        # Odpowiedź z technicznym językiem
        technical_response = "Model SHAP wskazuje, że wartość AUC-ROC wynosi 0.85."

        is_appropriate, problems = checker.check_health_literacy_appropriate(
            technical_response, 'basic'
        )

        assert is_appropriate is False
        assert len(problems) > 0


class TestPrompts:
    """Testy dla modułu promptów."""

    def test_get_risk_level(self):
        """Test określania poziomu ryzyka."""
        assert get_risk_level(0.1) == 'low'
        assert get_risk_level(0.3) == 'moderate'
        assert get_risk_level(0.5) == 'moderate'
        assert get_risk_level(0.7) == 'high'
        assert get_risk_level(0.9) == 'high'

    def test_translate_feature(self):
        """Test tłumaczenia nazw cech."""
        assert translate_feature('Wiek') == 'Twój wiek'
        assert translate_feature('Kreatynina') == 'Poziom wskaźnika czynności nerek'
        assert translate_feature('Unknown_Feature') == 'Unknown_Feature'

    def test_feature_translations_not_empty(self):
        """Test że słownik tłumaczeń nie jest pusty."""
        assert len(FEATURE_TRANSLATIONS) > 0
        assert 'Wiek' in FEATURE_TRANSLATIONS
        assert 'Kreatynina' in FEATURE_TRANSLATIONS

    def test_risk_level_descriptions(self):
        """Test opisów poziomów ryzyka."""
        for level in ['low', 'moderate', 'high']:
            assert level in RISK_LEVEL_DESCRIPTIONS
            assert 'patient' in RISK_LEVEL_DESCRIPTIONS[level]
            assert 'clinician' in RISK_LEVEL_DESCRIPTIONS[level]

    def test_format_risk_factors_patient(self):
        """Test formatowania czynników ryzyka."""
        factors = [
            {'feature': 'Wiek', 'contribution': 0.2},
            {'feature': 'Kreatynina', 'contribution': 0.15}
        ]

        formatted = format_risk_factors_patient(factors)

        assert '1.' in formatted
        assert '2.' in formatted

    def test_format_risk_factors_empty(self):
        """Test formatowania pustej listy."""
        formatted = format_risk_factors_patient([])
        assert 'Nie zidentyfikowano' in formatted


class TestRAGPipeline:
    """Testy dla RAG Pipeline."""

    def test_import_rag_pipeline(self):
        """Test importu RAGPipeline."""
        from src.agent.rag import RAGPipeline
        assert RAGPipeline is not None

    def test_rag_init_without_api_key(self):
        """Test inicjalizacji bez API key."""
        import os
        # Usuń klucz API jeśli istnieje
        original = os.environ.get('OPENAI_API_KEY')
        if original:
            del os.environ['OPENAI_API_KEY']

        from src.agent.rag import RAGPipeline

        pipeline = RAGPipeline()

        # Powinien działać bez klucza (tryb offline)
        assert pipeline.guardrails is not None

        # Przywróć klucz
        if original:
            os.environ['OPENAI_API_KEY'] = original


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
