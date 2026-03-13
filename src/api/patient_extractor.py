"""
Ekstraktor danych pacjenta z tekstu w języku polskim.

Moduł NLP do wyodrębniania parametrów klinicznych z opisów
słownych pacjentów w kontekście zapalenia naczyń.
"""

import re
import logging
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ============================================================================
# REGEX PATTERNS
# ============================================================================

# Wiek: "65 lat", "wiek 65", "wiek: 65", "ma 65 lat", "65-letni"
_AGE_PATTERNS = [
    re.compile(r"(\d{1,3})\s*lat", re.IGNORECASE),
    re.compile(r"(\d{1,3})\s*-?\s*letn[ia]", re.IGNORECASE),
    re.compile(r"wiek\D{0,10}(\d{1,3})", re.IGNORECASE),
    re.compile(r"ma\s+(\d{1,3})\s+lat", re.IGNORECASE),
]

# Płeć
_FEMALE_KEYWORDS = [
    "kobieta", "pacjentka", "kobiecie", "kobiety", "żeńska", "żeńskiej",
    "płeć żeńska", "plec zenska", "k\b",
]
_MALE_KEYWORDS = [
    "mężczyzna", "mezczyzna", "pacjent ", "mężczyźnie", "mężczyzny",
    "męska", "meska", "płeć męska", "plec meska",
]

# Kreatynina: "kreatynina 220", "kreat. 220", "kreatynina: 220 μmol/L"
_CREATININE_PATTERNS = [
    re.compile(r"kreatynin\w*\D{0,15}(\d+(?:[.,]\d+)?)", re.IGNORECASE),
    re.compile(r"kreat\.?\s*:?\s*(\d+(?:[.,]\d+)?)", re.IGNORECASE),
]

# CRP: "CRP 120", "max CRP 120", "crp: 120"
_CRP_PATTERNS = [
    re.compile(r"(?:max\.?\s*)?crp\D{0,10}(\d+(?:[.,]\d+)?)", re.IGNORECASE),
]

# Wiek rozpoznania: "rozpoznano w wieku 50", "rozpoznanie w 50 r.ż."
_AGE_DIAGNOSIS_PATTERNS = [
    re.compile(r"rozpoznan\w*\s+(?:w\s+)?(?:wieku\s+)?(\d{1,3})", re.IGNORECASE),
    re.compile(r"(?:wiek|w)\s+rozpoznani\w*\D{0,10}(\d{1,3})", re.IGNORECASE),
]

# Liczba narządów: "zajęcie 3 narządów", "3 narządy", "zajęte 4 narządy"
_ORGAN_COUNT_PATTERNS = [
    re.compile(r"(\d{1,2})\s*(?:narz[aą]d|organ)", re.IGNORECASE),
    re.compile(r"zaj[eę]\w*\s+(\d{1,2})\s*narz[aą]d", re.IGNORECASE),
    re.compile(r"liczba\s+zaj[eę]\w*\s+narz[aą]d\w*\D{0,5}(\d{1,2})", re.IGNORECASE),
]

# Sterydy dawka: "sterydy 0.5g", "dawka sterydów 1g", "sterydy dawka 0.5"
_STEROID_DOSE_PATTERNS = [
    re.compile(r"ster[yo]d\w*\s*(?:dawka\s*)?\D{0,10}(\d+(?:[.,]\d+)?)\s*g\b", re.IGNORECASE),
    re.compile(r"dawka?\s*ster[yo]d\w*\D{0,10}(\d+(?:[.,]\d+)?)", re.IGNORECASE),
]

# Czas sterydów: "sterydy przez 12 miesięcy", "sterydoterapia 6 mies"
_STEROID_DURATION_PATTERNS = [
    re.compile(r"ster[yo]d\w*\s+(?:przez\s+)?(\d+)\s*(?:miesi|mies)", re.IGNORECASE),
    re.compile(r"ster[yo]d\w*\s+(?:od\s+)?(\d+)\s*m\b", re.IGNORECASE),
]


# ============================================================================
# BOOLEAN FIELD KEYWORDS
# ============================================================================

_BOOLEAN_FIELD_KEYWORDS: Dict[str, List[Tuple[str, bool]]] = {
    "manifestacja_nerki": [
        ("nerek", True), ("nerki", True), ("nerk", True), ("zajęcie nerek", True),
        ("zajeciu nerek", True), ("nefropatia", True), ("nefrologiczn", True),
    ],
    "manifestacja_sercowo_naczyniowy": [
        ("serce", True), ("sercow", True), ("naczyniow", True),
        ("kardiologiczn", True), ("zajęcie serca", True),
    ],
    "manifestacja_pokarmowy": [
        ("pokarmow", True), ("jelitow", True), ("gastryczn", True),
        ("żołądek", True), ("jelita", True),
    ],
    "manifestacja_zajecie_csn": [
        ("ośrodkow", True), ("csn", True), ("mózg", True),
        ("oun", True), ("centraln", True),
    ],
    "manifestacja_neurologiczny": [
        ("neurologiczn", True), ("neuropatia", True), ("obwodow", True),
    ],
    "manifestacja_oddechowy": [
        ("oddechow", True), ("płucn", True), ("plucn", True),
        ("respirac", True),
    ],
    "zaostrz_wymagajace_oit": [
        ("oit", True), ("oiom", True), ("intensywn", True),
    ],
    "zaostrz_wymagajace_hospital": [
        ("hospitaliz", True), ("szpital", True),
    ],
    "plazmaferezy": [
        ("plazmaferez", True), ("aferez", True),
    ],
    "dializa": [
        ("dializ", True), ("hemodializ", True),
    ],
    "pulsy": [
        ("puls", True),
    ],
    "powiklania_serce_pluca": [
        ("powikłani", True), ("powiklani", True),
    ],
    "powiklania_infekcja": [
        ("infekcj", True), ("zakażeni", True), ("zakazeni", True),
    ],
}

# Negation words that flip boolean detection
_NEGATION_WORDS = ["nie", "bez", "brak", "negatywn", "ujemn", "wykluczon"]


def _extract_float(match_str: str) -> float:
    """Konwertuj dopasowany string na float (obsługuje przecinek)."""
    return float(match_str.replace(",", "."))


def _check_negation(text: str, keyword_pos: int) -> bool:
    """Sprawdź czy słowo kluczowe jest zanegowane (np. 'bez zajęcia nerek')."""
    # Check 30 chars before the keyword
    context = text[max(0, keyword_pos - 30):keyword_pos].lower()
    return any(neg in context for neg in _NEGATION_WORDS)


def extract_patient_from_text(text: str, task_type: str = "mortality") -> Dict[str, Any]:
    """
    Wyodrębnij dane pacjenta z tekstu w języku polskim.

    Args:
        text: Tekst opisujący pacjenta
        task_type: Typ zadania ('mortality' lub 'dialysis')

    Returns:
        Słownik z rozpoznanymi polami i ich wartościami
    """
    extracted: Dict[str, Any] = {}
    text_lower = text.lower()

    # --- Wiek ---
    for pattern in _AGE_PATTERNS:
        m = pattern.search(text)
        if m:
            age = int(m.group(1))
            if 0 < age <= 120:
                extracted["wiek"] = float(age)
                break

    # --- Płeć ---
    for kw in _FEMALE_KEYWORDS:
        if kw in text_lower:
            extracted["plec"] = 0
            break
    if "plec" not in extracted:
        for kw in _MALE_KEYWORDS:
            if kw in text_lower:
                extracted["plec"] = 1
                break

    # --- Kreatynina ---
    for pattern in _CREATININE_PATTERNS:
        m = pattern.search(text)
        if m:
            val = _extract_float(m.group(1))
            if 0 < val <= 2000:
                extracted["kreatynina"] = val
                break

    # --- CRP ---
    for pattern in _CRP_PATTERNS:
        m = pattern.search(text)
        if m:
            val = _extract_float(m.group(1))
            if 0 < val <= 500:
                extracted["max_crp"] = val
                break

    # --- Wiek rozpoznania ---
    for pattern in _AGE_DIAGNOSIS_PATTERNS:
        m = pattern.search(text)
        if m:
            val = int(m.group(1))
            if 0 < val <= 120:
                extracted["wiek_rozpoznania"] = float(val)
                break

    # --- Liczba zajętych narządów ---
    for pattern in _ORGAN_COUNT_PATTERNS:
        m = pattern.search(text)
        if m:
            val = int(m.group(1))
            if 0 <= val <= 20:
                extracted["liczba_zajetych_narzadow"] = val
                break

    # --- Dawka sterydów ---
    for pattern in _STEROID_DOSE_PATTERNS:
        m = pattern.search(text)
        if m:
            val = _extract_float(m.group(1))
            if 0 < val <= 100:
                extracted["sterydy_dawka_g"] = val
                break

    # --- Czas sterydów ---
    for pattern in _STEROID_DURATION_PATTERNS:
        m = pattern.search(text)
        if m:
            val = int(m.group(1))
            if 0 < val <= 600:
                extracted["czas_sterydow"] = float(val)
                break

    # --- Boolean fields ---
    for field_name, keywords in _BOOLEAN_FIELD_KEYWORDS.items():
        # Skip fields not relevant to current task type
        if task_type == "mortality" and field_name in ("manifestacja_oddechowy", "zaostrz_wymagajace_hospital", "pulsy"):
            continue
        if task_type == "dialysis" and field_name in ("manifestacja_pokarmowy", "manifestacja_zajecie_csn", "dializa"):
            continue

        for keyword, default_val in keywords:
            pos = text_lower.find(keyword)
            if pos >= 0:
                negated = _check_negation(text_lower, pos)
                extracted[field_name] = 0 if negated else (1 if default_val else 0)
                break

    return extracted


# ============================================================================
# MISSING FIELDS LOGIC
# ============================================================================

# Required for prediction (minimum)
_REQUIRED_FIELDS = ["wiek", "plec"]

# High-value fields (ask about but don't block prediction)
_HIGH_VALUE_FIELDS_MORTALITY = [
    "kreatynina", "manifestacja_nerki", "liczba_zajetych_narzadow",
    "manifestacja_sercowo_naczyniowy", "zaostrz_wymagajace_oit",
]

_HIGH_VALUE_FIELDS_DIALYSIS = [
    "kreatynina", "manifestacja_nerki", "liczba_zajetych_narzadow",
    "zaostrz_wymagajace_oit", "manifestacja_oddechowy",
]

_FOLLOWUP_QUESTIONS: Dict[str, str] = {
    "wiek": "Ile lat ma pacjent?",
    "plec": "Jaka jest płeć pacjenta (kobieta/mężczyzna)?",
    "kreatynina": "Jaki jest poziom kreatyniny (w μmol/L)?",
    "manifestacja_nerki": "Czy występuje zajęcie nerek?",
    "liczba_zajetych_narzadow": "Ile narządów jest zajętych przez chorobę?",
    "manifestacja_sercowo_naczyniowy": "Czy występuje zajęcie sercowo-naczyniowe?",
    "zaostrz_wymagajace_oit": "Czy były zaostrzenia wymagające OIT?",
    "manifestacja_oddechowy": "Czy występuje zajęcie układu oddechowego?",
    "max_crp": "Jakie było maksymalne CRP (mg/L)?",
}


def get_missing_required_fields(accumulated: Dict[str, Any], task_type: str = "mortality") -> List[str]:
    """Zwróć listę brakujących wymaganych pól."""
    return [f for f in _REQUIRED_FIELDS if f not in accumulated or accumulated[f] is None]


def get_missing_high_value_fields(accumulated: Dict[str, Any], task_type: str = "mortality") -> List[str]:
    """Zwróć listę brakujących pól o wysokiej wartości diagnostycznej."""
    hv_fields = _HIGH_VALUE_FIELDS_MORTALITY if task_type == "mortality" else _HIGH_VALUE_FIELDS_DIALYSIS
    return [f for f in hv_fields if f not in accumulated or accumulated[f] is None]


def get_followup_question(missing_fields: List[str], task_type: str = "mortality") -> Optional[str]:
    """Zwróć pytanie uzupełniające dla najważniejszego brakującego pola."""
    for field in missing_fields:
        if field in _FOLLOWUP_QUESTIONS:
            return _FOLLOWUP_QUESTIONS[field]
    return None


def get_followup_suggestions(accumulated: Dict[str, Any], task_type: str = "mortality") -> List[str]:
    """Zwróć sugestie pytań uzupełniających na podstawie brakujących danych."""
    missing_required = get_missing_required_fields(accumulated, task_type)
    missing_hv = get_missing_high_value_fields(accumulated, task_type)
    all_missing = missing_required + missing_hv

    suggestions = []
    for field in all_missing[:3]:
        q = _FOLLOWUP_QUESTIONS.get(field)
        if q:
            suggestions.append(q)
    return suggestions


def build_patient_summary(accumulated: Dict[str, Any], task_type: str = "mortality") -> str:
    """Zbuduj czytelne podsumowanie zebranych danych pacjenta."""
    parts = []

    if "wiek" in accumulated:
        parts.append(f"Wiek: {int(accumulated['wiek'])} lat")
    if "plec" in accumulated:
        parts.append(f"Płeć: {'mężczyzna' if accumulated['plec'] == 1 else 'kobieta'}")
    if "kreatynina" in accumulated:
        parts.append(f"Kreatynina: {accumulated['kreatynina']} μmol/L")
    if "max_crp" in accumulated:
        parts.append(f"Max CRP: {accumulated['max_crp']} mg/L")
    if accumulated.get("manifestacja_nerki"):
        parts.append("Zajęcie nerek: tak")
    if accumulated.get("manifestacja_sercowo_naczyniowy"):
        parts.append("Zajęcie sercowo-naczyniowe: tak")
    if accumulated.get("zaostrz_wymagajace_oit"):
        parts.append("Zaostrzenia OIT: tak")
    if "liczba_zajetych_narzadow" in accumulated:
        parts.append(f"Liczba zajętych narządów: {accumulated['liczba_zajetych_narzadow']}")
    if accumulated.get("plazmaferezy"):
        parts.append("Plazmaferezy: tak")
    if accumulated.get("dializa") and task_type == "mortality":
        parts.append("Dializa: tak")

    return "\n".join(f"• {p}" for p in parts) if parts else "(brak danych)"
