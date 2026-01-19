#!/usr/bin/env python3
"""Analizuje czy imiesłowy kończą zdania."""

import re
from docx import Document
import sys

def analyze_sentence_endings(docx_path):
    """Sprawdza czy imiesłowy kończą zdania."""
    doc = Document(docx_path)

    participles = [
        (38, 'zapewniający'),
        (90, 'zapewniającego'),
        (166, 'zapewniający'),
        (38, 'oferując'),
        (88, 'oferując'),
        (97, 'oferując'),
        (129, 'oferując'),
        (168, 'oferując'),
        (166, 'dostarczając'),
    ]

    print("\n=== Analiza końcówek zdań z imiesłowami ===\n")

    for para_num, participle in participles:
        para = doc.paragraphs[para_num]
        text = para.text

        # Znajdź pozycję imiesłowu
        matches = list(re.finditer(re.escape(participle), text, re.IGNORECASE))

        if not matches:
            continue

        for match_idx, match in enumerate(matches):
            # Pobierz tekst po imiesłowie (do 100 znaków)
            after = text[match.end():match.end() + 100]

            # Sprawdź co jest bezpośrednio po imiesłowie
            next_chars = after[:30].strip()

            is_ending = False
            ending_type = ""

            if not next_chars or next_chars[0] in ['.', '\n', '•', '']:
                is_ending = True
                ending_type = "KOŃCZY ZDANIE/PUNKT"
            elif next_chars.startswith(','):
                ending_type = "Przecinek - kontynuacja"
            elif next_chars[0].islower():
                ending_type = "Kontynuacja zdania"
            else:
                ending_type = "Niejednoznaczne"

            # Pobierz kontekst (50 znaków przed i 50 po)
            start = max(0, match.start() - 50)
            end = min(len(text), match.end() + 50)
            context = text[start:end]

            print(f"\nParagraf #{para_num} (wystąpienie {match_idx + 1}):")
            print(f"  Imiesłów: '{participle}'")
            print(f"  Status: {ending_type} {' WYMAGA POPRAWKI!' if is_ending else '✓ OK'}")
            print(f"  Kontekst: ...{context}...")
            print(f"  Po imiesłowie: '{next_chars[:50]}'")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Użycie: python analyze_sentence_endings.py <plik.docx>")
        sys.exit(1)

    analyze_sentence_endings(sys.argv[1])
