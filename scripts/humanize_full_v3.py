#!/usr/bin/env python3
"""
Pełna humanizacja pracy magisterskiej - Opcja B
Transformacja strony biernej na czynną, pierwsza osoba, konwersacyjny ton
"""

from docx import Document
from docx.shared import Pt
import re
from typing import Dict, List, Tuple

# === MAPA TRANSFORMACJI ===

# 1. Zamiana strony biernej na czynną (pierwsza osoba)
PASSIVE_TO_ACTIVE = {
    # Niniejsza praca → Ja
    r'Niniejsza praca magisterska przedstawia': 'W tej pracy przedstawiam',
    r'Niniejsza praca magisterska': 'Ta praca',
    r'niniejszej pracy magisterskiej': 'tej pracy',
    r'niniejszej pracy': 'tej pracy',
    r'niniejszym rozdziale': 'tym rozdziale',

    # Strona bierna → czynna
    r'została wyznaczona': 'wyznaczyłem',
    r'zostały wyznaczone': 'wyznaczyłem',
    r'został wybrany': 'wybrałem',
    r'została wybrana': 'wybrałem',
    r'zostało wybrane': 'wybrałem',
    r'został opracowany': 'opracowałem',
    r'została opracowana': 'opracowałem',
    r'zostało opracowane': 'opracowałem',
    r'został zaimplementowany': 'zaimplementowałem',
    r'została zaimplementowana': 'zaimplementowałem',
    r'zostało zaimplementowane': 'zaimplementowałem',
    r'został przeprowadzony': 'przeprowadziłem',
    r'została przeprowadzona': 'przeprowadziłem',
    r'zostało przeprowadzone': 'przeprowadziłem',
    r'został stworzony': 'stworzyłem',
    r'została stworzona': 'stworzyłem',
    r'zostało stworzone': 'stworzyłem',
    r'został wykonany': 'wykonałem',
    r'została wykonana': 'wykonałem',
    r'zostało wykonane': 'wykonałem',
    r'został zastosowany': 'zastosowałem',
    r'została zastosowana': 'zastosowałem',
    r'zostało zastosowane': 'zastosowałem',
    r'został wykorzystany': 'wykorzystałem',
    r'została wykorzystana': 'wykorzystałem',
    r'zostało wykorzystane': 'wykorzystałem',
    r'został zaprojektowany': 'zaprojektowałem',
    r'została zaprojektowana': 'zaprojektowałem',
    r'zostało zaprojektowane': 'zaprojektowałem',

    # Formy bezosobowe → pierwsza osoba
    r'Zaimplementowano': 'Zaimplementowałem',
    r'Opracowano': 'Opracowałem',
    r'Przeprowadzono': 'Przeprowadziłem',
    r'Stworzono': 'Stworzyłem',
    r'Wykonano': 'Wykonałem',
    r'Zastosowano': 'Zastosowałem',
    r'Wykorzystano': 'Wykorzystałem',
    r'Zaprojektowano': 'Zaprojektowałem',
    r'Zmierzono': 'Zmierzyłem',
    r'Porównano': 'Porównałem',
    r'Przeanalizowano': 'Przeanalizowałem',
    r'Zbadano': 'Zbadałem',
    r'Wykazano': 'Wykazałem',
    r'Zweryfikowano': 'Zweryfikowałem',
    r'Zintegrowano': 'Zintegrowałem',
    r'Omówiono': 'Omówiłem',
    r'Przedstawiono': 'Przedstawiłem',

    # Podmiot "praca/system" → "ja"
    r'Praca ta ma na celu': 'Moim celem jest',
    r'System umożliwia': 'Zaprojektowałem system, który umożliwia',
    r'Model osiągnął': 'Mój model osiągnął',
    r'Analiza wykazała': 'Moja analiza wykazała',

    # Archaiczne formy
    r'Głównym celem niniejszej pracy magisterskiej jest': 'Głównym celem tej pracy jest',
}

# 2. Transformacje konwersacyjne (dodanie łączników)
CONVERSATIONAL_ADDITIONS = {
    # Przejścia między sekcjami
    r'^Zapalenie naczyń \(vasculitis\) to': 'Zacznę od wyjaśnienia, że zapalenie naczyń (vasculitis) to',
    r'^Najlepsze wyniki uzyskał model XGBoost': 'Spośród testowanych modeli, najlepsze wyniki uzyskał XGBoost',
    r'^Analiza ważności cech wykazała': 'Co ciekawe, analiza ważności cech wykazała',
    r'^Dla najlepszego modelu': 'Przechodząc do wyjaśnialności, dla najlepszego modelu',
    r'^Zmierzono wydajność': 'Kluczowe było także zmierzenie wydajności',
}

# 3. Rozbicie mechanicznego paralelizmu (cele badawcze)
GOALS_TRANSFORMATIONS = {
    r'1\. Implementację i porównanie': '1. Pierwszym zadaniem była implementacja i porównanie',
    r'2\. Opracowanie i integrację': '2. Następnie opracowałem i zintegrowałem',
    r'3\. Przeprowadzenie systematycznego': '3. Istotne było przeprowadzenie systematycznego',
    r'4\. Zaprojektowanie i implementację': '4. Zaprojektowałem i zaimplementowałem',
    r'5\. Opracowanie systemu': '5. Kolejnym krokiem było opracowanie systemu',
    r'6\. Implementację mechanizmów': '6. Na końcu zaimplementowałem mechanizmy',
}

# 4. Transformacje sekcji wyników (bardziej narracyjny styl)
RESULTS_TRANSFORMATIONS = {
    # Zamiana "Model X osiągnął metrykę = Y" na bardziej narracyjne
    r'osiągając AUC-ROC = (\d+\.\d+) oraz czułość = (\d+\.\d+)': r'z AUC-ROC na poziomie \1 i czułością \2',
    r'Wysoka czułość w kontekście': 'Ta wysoka czułość ma szczególne znaczenie w kontekście',
    r'XGBoost wybrano jako model finalny': 'Zdecydowałem się na XGBoost jako model finalny',
    r'ze względu na:': 'z kilku powodów:',
}

# 5. Transformacje wniosków (osobisty głos)
CONCLUSIONS_TRANSFORMATIONS = {
    r'z powodzeniem zrealizowała założone cele badawcze': 'osiągnęła założone cele badawcze',
    r'Opracowany system': 'System, który opracowałem,',
    r'umożliwia predykcję': 'skutecznie przewiduje',
}

# 6. Redukcja formalności
FORMALITY_REDUCTIONS = {
    r'ewaluacja': 'ocena',
    r'implementacja': 'wdrożenie',
    r'predykcja': 'przewidywanie',
    r'integracja': 'połączenie',
    r'optymalizacja': 'usprawnienie',
    r'walidacja': 'sprawdzenie',
    r'konfiguracja': 'ustawienia',
    r'dokumentacja': 'opis',
}


def apply_transformations(text: str, transformations: Dict[str, str]) -> str:
    """Zastosuj transformacje do tekstu"""
    result = text
    for pattern, replacement in transformations.items():
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE if not pattern.startswith('^') else 0)
    return result


def humanize_paragraph(text: str, para_index: int) -> str:
    """Humanizuj pojedynczy paragraf z uwzględnieniem kontekstu"""
    if not text.strip():
        return text

    result = text

    # Zawsze stosuj transformacje strony biernej → czynnej
    result = apply_transformations(result, PASSIVE_TO_ACTIVE)

    # Sekcja Abstract (38) - pełna transformacja
    if para_index == 38:
        result = apply_transformations(result, CONVERSATIONAL_ADDITIONS)

    # Sekcja Wprowadzenie (88-95)
    if 88 <= para_index <= 95:
        result = apply_transformations(result, CONVERSATIONAL_ADDITIONS)

    # Cele badawcze (90)
    if para_index == 90:
        result = apply_transformations(result, GOALS_TRANSFORMATIONS)

    # Wyniki (140-160)
    if 140 <= para_index <= 160:
        result = apply_transformations(result, RESULTS_TRANSFORMATIONS)
        result = apply_transformations(result, CONVERSATIONAL_ADDITIONS)

    # Wnioski (166+)
    if para_index >= 166:
        result = apply_transformations(result, CONCLUSIONS_TRANSFORMATIONS)

    # Redukcja formalności - tylko dla wybranych słów, ostrożnie
    # (zakomentowane, bo może być zbyt agresywne)
    # result = apply_transformations(result, FORMALITY_REDUCTIONS)

    return result


def humanize_document(input_path: str, output_path: str):
    """Humanizuj cały dokument"""
    doc = Document(input_path)

    changes = []

    for i, para in enumerate(doc.paragraphs):
        original = para.text
        if not original.strip():
            continue

        humanized = humanize_paragraph(original, i)

        if humanized != original:
            changes.append({
                'para': i,
                'before': original[:100] + '...' if len(original) > 100 else original,
                'after': humanized[:100] + '...' if len(humanized) > 100 else humanized
            })

            # Zachowaj formatowanie, zmień tylko tekst
            for run in para.runs:
                run.text = ''
            if para.runs:
                para.runs[0].text = humanized
            else:
                para.text = humanized

    doc.save(output_path)

    return changes


def print_changes(changes: List[Dict]):
    """Wyświetl zmiany"""
    print(f"\n=== ZASTOSOWANE ZMIANY ({len(changes)} paragrafów) ===\n")

    for change in changes:
        print(f"--- Paragraf {change['para']} ---")
        print(f"PRZED: {change['before']}")
        print(f"PO:    {change['after']}")
        print()


if __name__ == '__main__':
    input_file = 'docs/magisterka_Zimny_sem1.docx'
    output_file = 'docs/magisterka_Zimny_sem1.docx'  # Nadpisz

    print("=== PEŁNA HUMANIZACJA - OPCJA B ===")
    print(f"Plik wejściowy: {input_file}")
    print(f"Plik wyjściowy: {output_file}")
    print()

    changes = humanize_document(input_file, output_file)
    print_changes(changes)

    print(f"\n Dokument zapisany: {output_file}")
    print(f" Zmieniono {len(changes)} paragrafów")
