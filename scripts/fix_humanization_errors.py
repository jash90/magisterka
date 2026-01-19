#!/usr/bin/env python3
"""
Naprawa błędów z pierwszej rundy humanizacji
"""

from docx import Document
import re

# Konkretne naprawy paragrafów
PARAGRAPH_FIXES = {
    # Para 95 - błędna wielka litera
    95: {
        'find': 'W tym rozdziale Przedstawiłem przegląd literatury naukowej stanowiącej podstawę teoretyczną pracy. Omówiłem',
        'replace': 'W tym rozdziale przedstawiam przegląd literatury naukowej stanowiącej podstawę teoretyczną pracy. Omawiam'
    },
    # Para 112 - błędna wielka litera
    112: {
        'find': 'Badania Przeprowadziłem zgodnie z następującym schematem:',
        'replace': 'Badania przeprowadziłem zgodnie z następującym schematem:'
    },
    # Para 142 - błędna gramatyka
    142: {
        'find': 'Globalna ważność cech wyznaczyłem przy użyciu trzech metod',
        'replace': 'Globalną ważność cech wyznaczyłem przy użyciu trzech metod'
    },
    # Para 166 - rozbite zdanie
    166: {
        'find': 'Ta praca osiągnęła założone cele badawcze. System, który opracowałem, wyjaśnialnej sztucznej inteligencji skutecznie przewiduje śmiertelności w zapaleniu naczyń.',
        'replace': 'Ta praca osiągnęła założone cele badawcze. System wyjaśnialnej sztucznej inteligencji, który opracowałem, skutecznie przewiduje ryzyko śmiertelności w zapaleniu naczyń.'
    },
}

# Globalne naprawy - wielkie litery po spacji
GLOBAL_FIXES = [
    # Wielkie litery w środku zdania po czasownikach
    (r'(\s)Przedstawiłem(\s)', r'\1przedstawiłem\2'),
    (r'(\s)Przeprowadziłem(\s)', r'\1przeprowadziłem\2'),
    (r'(\s)Zaimplementowałem(\s)', r'\1zaimplementowałem\2'),
    (r'(\s)Porównałem(\s)', r'\1porównałem\2'),
    (r'(\s)Zmierzyłem(\s)', r'\1zmierzyłem\2'),
    (r'(\s)Omówiłem(\s)', r'\1omówiłem\2'),
    (r'(\s)Zintegrowałem(\s)', r'\1zintegrowałem\2'),

    # Na początku zdania - wielka litera poprawna (ale nie po przecinku)
    (r'Poniżej Przedstawiłem', 'Poniżej przedstawiam'),
    (r'dla najlepszego modelu \(XGBoost\) Zaimplementowałem', 'dla najlepszego modelu (XGBoost) zaimplementowałem'),

    # Czytelność - zmiana czasu przeszłego na teraźniejszy gdzie to sensowne
    (r'API Zaimplementowałem przy użyciu', 'API zaimplementowałem przy użyciu'),
    (r'Interaktywny dashboard Zaimplementowałem', 'Interaktywny dashboard zaimplementowałem'),

    # Naprawa rozbicia realizacji celów w para 166
    (r'1\.  Zaimplementowałem i Porównałem', '1.  Zaimplementowałem i porównałem'),
    (r'2\.  Zintegrowałem', '2.  Zintegrowałem'),
]


def fix_document(input_path: str, output_path: str):
    """Napraw błędy w dokumencie"""
    doc = Document(input_path)
    changes = []

    for i, para in enumerate(doc.paragraphs):
        original = para.text
        if not original.strip():
            continue

        fixed = original

        # Najpierw zastosuj konkretne naprawy dla paragrafu
        if i in PARAGRAPH_FIXES:
            fix = PARAGRAPH_FIXES[i]
            if fix['find'] in fixed:
                fixed = fixed.replace(fix['find'], fix['replace'])

        # Potem zastosuj globalne naprawy
        for pattern, replacement in GLOBAL_FIXES:
            fixed = re.sub(pattern, replacement, fixed)

        if fixed != original:
            changes.append({
                'para': i,
                'before': original[:150] + '...' if len(original) > 150 else original,
                'after': fixed[:150] + '...' if len(fixed) > 150 else fixed
            })

            # Zachowaj formatowanie
            for run in para.runs:
                run.text = ''
            if para.runs:
                para.runs[0].text = fixed
            else:
                para.text = fixed

    doc.save(output_path)
    return changes


def main():
    input_file = 'docs/magisterka_Zimny_sem1.docx'
    output_file = 'docs/magisterka_Zimny_sem1.docx'

    print("=== NAPRAWA BŁĘDÓW HUMANIZACJI ===\n")

    changes = fix_document(input_file, output_file)

    print(f"Naprawiono {len(changes)} paragrafów:\n")
    for change in changes:
        print(f"--- Para {change['para']} ---")
        print(f"PRZED: {change['before']}")
        print(f"PO:    {change['after']}")
        print()

    print(f" Zapisano: {output_file}")


if __name__ == '__main__':
    main()
