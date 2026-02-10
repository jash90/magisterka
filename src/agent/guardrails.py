"""
Moduł guardrails dla bezpieczeństwa agenta konwersacyjnego.

Zawiera funkcje do wykrywania niebezpiecznych zapytań
i walidacji odpowiedzi agenta.
"""

import re
from typing import Optional, List, Dict, Any, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GuardrailsChecker:
    """
    Klasa do sprawdzania guardrails w konwersacji.

    Wykrywa niebezpieczne zapytania i waliduje odpowiedzi
    agenta przed wysłaniem do użytkownika.
    """

    def __init__(self):
        """Inicjalizacja checkera guardrails."""
        # Wzorce dla niebezpiecznych zapytań
        self._init_harmful_patterns()
        self._init_medical_advice_patterns()
        self._init_forbidden_response_patterns()

    def _init_harmful_patterns(self):
        """Inicjalizuj wzorce dla potencjalnie szkodliwych zapytań."""
        self.harmful_keywords = [
            # Samobójstwo/samookaleczenie
            r'samobójstw\w*',
            r'popełnić\s+samobójstwo',
            r'targnąć\s+się\s+na\s+życie',
            r'odebrać\s+(sobie\s+)?życie',
            r'skończyć\s+z\s+życiem',
            r'nie\s+chcę\s+(\w+\s+)?żyć',
            r'chcę\s+umrzeć',
            r'samookaleczen\w*',
            r'skaleczyć\s+się',
            r'zranić\s+się',
            r'ciąć\s+się',

            # Przemoc
            r'skrzywdzić\s+kogoś',
            r'zranić\s+kogoś',
            r'zabić\s+kogoś',
        ]

        self.harmful_patterns = [re.compile(p, re.IGNORECASE) for p in self.harmful_keywords]

    def _init_medical_advice_patterns(self):
        """Inicjalizuj wzorce dla próśb o poradę medyczną."""
        self.medical_advice_keywords = [
            # Prośby o diagnozę
            r'czy\s+mam\s+\w+',
            r'co\s+mi\s+jest',
            r'zdiagnozuj',
            r'postaw\s+diagnozę',
            r'jaka\s+to\s+choroba',

            # Prośby o leki
            r'jaki\s+lek\s+powinienem',
            r'jakie\s+leki\s+brać',
            r'przepisz\s+mi',
            r'co\s+powinienem\s+zażywać',
            r'zmień\s+mi\s+dawkę',
            r'zwięks\w*\s+dawkę',
            r'zmniejsz\s+dawkę',
            r'odstawić\s+lek',

            # Prośby o prognozę
            r'ile\s+będę\s+(\w+\s+)?żyć',
            r'jak\s+długo\s+jeszcze',
            r'kiedy\s+umrę',
            r'jaka\s+jest\s+moja\s+szansa',
            r'jaki\s+procent\s+szans',
        ]

        self.medical_advice_patterns = [re.compile(p, re.IGNORECASE) for p in self.medical_advice_keywords]

    def _init_forbidden_response_patterns(self):
        """Inicjalizuj wzorce zakazanych odpowiedzi."""
        self.forbidden_response_keywords = [
            # Konkretne prognozy
            r'\d+%\s+szans\w*\s+(na\s+)?(przeżycie|zgon|śmierć)',
            r'przeżyjesz\s+\d+\s+(lat|miesięcy)',
            r'rokowanie\s+wynosi',

            # Diagnozy
            r'masz\s+\w+\s+(chorobę|zaburzenie|zespół)',
            r'to\s+oznacza,?\s+że\s+(masz|cierpisz)',
            r'diagnozuję\s+cię',

            # Zalecenia lekowe
            r'(weź|zażyj|przyjmij)\s+\w+\s+mg',
            r'zwiększ\s+dawkę\s+do',
            r'zmień\s+lek\s+na',
            r'odstawić\s+\w+',

            # Krytyka lekarza
            r'lekarz\s+się\s+myli',
            r'twój\s+lekarz\s+nie\s+ma\s+racji',
            r'zmień\s+lekarza',
        ]

        self.forbidden_patterns = [re.compile(p, re.IGNORECASE) for p in self.forbidden_response_keywords]

    def check_query(self, query: str) -> Optional[str]:
        """
        Sprawdź zapytanie użytkownika.

        Args:
            query: Zapytanie użytkownika

        Returns:
            Specjalna odpowiedź jeśli wykryto problem, None w przeciwnym razie
        """
        # Sprawdź potencjalnie szkodliwe treści
        harmful = self._check_harmful_content(query)
        if harmful:
            return harmful

        # Sprawdź prośby o poradę medyczną
        medical = self._check_medical_advice(query)
        if medical:
            return medical

        return None

    def _check_harmful_content(self, query: str) -> Optional[str]:
        """Sprawdź czy zapytanie zawiera treści o samobójstwie/samookaleczeniu."""
        for pattern in self.harmful_patterns:
            if pattern.search(query):
                logger.warning(f"Wykryto potencjalnie szkodliwą treść w zapytaniu")
                return self._get_crisis_response()

        return None

    def _check_medical_advice(self, query: str) -> Optional[str]:
        """Sprawdź czy zapytanie to prośba o poradę medyczną."""
        for pattern in self.medical_advice_patterns:
            if pattern.search(query):
                # Określ typ prośby
                if any(re.search(p, query, re.IGNORECASE) for p in [
                    r'zdiagnozuj', r'postaw\s+diagnozę', r'jaka\s+to\s+choroba',
                    r'czy\s+mam', r'co\s+mi\s+jest'
                ]):
                    return self._get_diagnosis_redirect()

                if any(re.search(p, query, re.IGNORECASE) for p in [
                    r'jaki\s+lek', r'przepisz', r'dawkę', r'odstawić'
                ]):
                    return self._get_medication_redirect()

                if any(re.search(p, query, re.IGNORECASE) for p in [
                    r'ile\s+będę\s+(\w+\s+)?żyć', r'kiedy\s+umrę', r'szansa'
                ]):
                    return self._get_prognosis_redirect()

        return None

    def _get_crisis_response(self) -> str:
        """Odpowiedź w sytuacji kryzysowej."""
        return """
Rozumiem, że możesz przechodzić przez bardzo trudny czas.
Twoje uczucia są ważne i chcę, żebyś wiedział/a, że istnieje pomoc.

**Proszę, skontaktuj się teraz z jednym z tych miejsc:**

**Telefon Zaufania dla Dorosłych w Kryzysie Emocjonalnym**: 116 123
   (czynny codziennie 14:00-22:00)

**Centrum Wsparcia dla osób dorosłych w kryzysie psychicznym**: 800 70 2222
   (czynny codziennie 14:00-22:00)

**Telefon Zaufania dla Dzieci i Młodzieży**: 116 111
   (czynny codziennie 24/7)

**Lub udaj się na najbliższy oddział ratunkowy**

Nie jesteś sam/sama. Są ludzie, którzy chcą i mogą pomóc.
"""

    def _get_diagnosis_redirect(self) -> str:
        """Odpowiedź na prośbę o diagnozę."""
        return """
Rozumiem, że chciałbyś/chciałabyś poznać przyczynę swoich objawów.

Niestety, **nie jestem w stanie postawić diagnozy** - wymaga to:
- Bezpośredniego badania przez lekarza
- Analizy pełnej historii medycznej
- Często dodatkowych badań

**Co możesz zrobić:**
1. Umów wizytę u lekarza pierwszego kontaktu
2. Przygotuj listę objawów i pytań na wizytę
3. Zabierz ze sobą wyniki ostatnich badań

Mogę natomiast pomóc Ci zrozumieć wyniki analizy ryzyka,
które zostały przeprowadzone na podstawie Twoich danych.
Czy chciałbyś/chciałabyś o tym porozmawiać?
"""

    def _get_medication_redirect(self) -> str:
        """Odpowiedź na prośbę o leki."""
        return """
Rozumiem, że masz pytania dotyczące leków.

**Decyzje o leczeniu farmakologicznym** muszą być podejmowane
przez Twojego lekarza prowadzącego, ponieważ:
- Zna on Twoją pełną historię medyczną
- Może ocenić interakcje między lekami
- Monitoruje Twoją reakcję na leczenie

**Nie jestem w stanie:**
- Przepisywać leków
- Zmieniać dawkowania
- Zalecać odstawienia leków

**Zalecam:**
Umów się na wizytę kontrolną i omów swoje wątpliwości
z lekarzem prowadzącym.

Mogę natomiast wyjaśnić, jakie czynniki wpływają
na Twoją ocenę ryzyka. Czy mogę w tym pomóc?
"""

    def _get_prognosis_redirect(self) -> str:
        """Odpowiedź na prośbę o prognozę."""
        return """
Rozumiem, że niepewność dotycząca przyszłości może być bardzo trudna.

**Muszę być z Tobą szczery/a:**
Nie mogę podać konkretnych prognoz dotyczących czasu życia, ponieważ:
- Każdy przypadek jest indywidualny
- Wiele czynników wpływa na przebieg choroby
- Medycyna stale się rozwija

**Co mogę zrobić:**
- Wyjaśnić, które czynniki wpływają na ocenę ryzyka
- Pomóc zrozumieć, co możesz kontrolować
- Wskazać pozytywne aspekty Twojej sytuacji

**Zachęcam do rozmowy z lekarzem** o prognozach -
ma on pełen obraz Twojego stanu zdrowia.

Czy chciałbyś/chciałabyś porozmawiać o czynnikach
wpływających na Twoje zdrowie?
"""

    def validate_response(self, response: str) -> str:
        """
        Waliduj i oczyść odpowiedź przed wysłaniem.

        Args:
            response: Odpowiedź do walidacji

        Returns:
            Oczyszczona odpowiedź
        """
        # Sprawdź zakazane wzorce
        for pattern in self.forbidden_patterns:
            if pattern.search(response):
                logger.warning(f"Wykryto zakazany wzorzec w odpowiedzi: {pattern.pattern}")
                response = self._sanitize_response(response, pattern)

        return response

    def _sanitize_response(self, response: str, pattern) -> str:
        """Oczyść odpowiedź z zakazanych treści."""
        # Zamień problematyczną treść na bezpieczną alternatywę
        sanitized = pattern.sub('[treść usunięta ze względów bezpieczeństwa]', response)

        # Dodaj przypomnienie
        sanitized += "\n\n*Uwaga: Niektóre szczegóły zostały pominięte. Skonsultuj się z lekarzem.*"

        return sanitized

    def check_health_literacy_appropriate(
        self,
        response: str,
        target_literacy: str
    ) -> Tuple[bool, List[str]]:
        """
        Sprawdź czy odpowiedź jest odpowiednia dla poziomu health literacy.

        Args:
            response: Odpowiedź do sprawdzenia
            target_literacy: Docelowy poziom ('basic', 'advanced', 'clinician')

        Returns:
            Tuple (czy_odpowiednia, lista_problemów)
        """
        problems = []

        if target_literacy == 'basic':
            # Sprawdź czy nie ma zbyt technicznego języka
            technical_terms = [
                r'SHAP', r'LIME', r'model\s+ML', r'predykcja',
                r'algorytm', r'regresja', r'klasyfikator',
                r'AUC', r'ROC', r'sensitivity', r'specificity'
            ]

            for term in technical_terms:
                if re.search(term, response, re.IGNORECASE):
                    problems.append(f"Znaleziono techniczny termin: {term}")

            # Sprawdź długość zdań (zbyt długie mogą być trudne)
            sentences = re.split(r'[.!?]', response)
            for sentence in sentences:
                word_count = len(sentence.split())
                if word_count > 25:
                    problems.append(f"Zbyt długie zdanie ({word_count} słów)")

        return len(problems) == 0, problems

    def add_safety_wrapper(self, response: str) -> str:
        """
        Dodaj wrapper bezpieczeństwa do odpowiedzi.

        Args:
            response: Oryginalna odpowiedź

        Returns:
            Odpowiedź z wrapperem bezpieczeństwa
        """
        safety_note = """
---
*Ta analiza jest generowana automatycznie i ma charakter wyłącznie informacyjny.
Nie stanowi porady medycznej ani diagnozy. W przypadku jakichkolwiek
wątpliwości dotyczących zdrowia, skonsultuj się z lekarzem.*
"""
        return response + safety_note

    def detect_emotional_distress(self, query: str) -> bool:
        """
        Wykryj oznaki stresu emocjonalnego w zapytaniu.

        Args:
            query: Zapytanie użytkownika

        Returns:
            True jeśli wykryto oznaki stresu
        """
        distress_patterns = [
            r'boję\s+się',
            r'jestem\s+przerażon\w*',
            r'nie\s+mogę\s+spać',
            r'nie\s+radzę\s+sobie',
            r'załamuj\w*\s+się',
            r'nie\s+wiem\s+co\s+robić',
            r'jestem\s+zdesperowa\w*',
            r'tracę\s+nadzieję',
        ]

        for pattern in distress_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return True

        return False

    def get_empathetic_prefix(self) -> str:
        """Pobierz empatyczny prefiks dla odpowiedzi przy wykrytym stresie."""
        return """
Rozumiem, że to może być trudny i stresujący czas.
Twoje uczucia są całkowicie zrozumiałe w tej sytuacji.

Chcę Ci pomóc jak najlepiej potrafię.
"""

    def log_guardrail_trigger(
        self,
        query: str,
        trigger_type: str,
        action_taken: str
    ) -> None:
        """
        Zaloguj aktywację guardrails.

        Args:
            query: Zapytanie użytkownika
            trigger_type: Typ triggera
            action_taken: Podjęta akcja
        """
        logger.info(f"""
Guardrail Trigger:
  Type: {trigger_type}
  Action: {action_taken}
  Query preview: {query[:50]}...
""")
