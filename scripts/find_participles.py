#!/usr/bin/env python3
"""Znajduje wszystkie imiesłowy w dokumencie DOCX."""

import re
from docx import Document
import sys

def find_all_participles(docx_path):
    """Znajduje wszystkie imiesłowy w dokumencie."""
    doc = Document(docx_path)

    # Rozszerzona lista imiesłowów do znalezienia
    participle_patterns = [
        r'\bzapewniając\w*\b',
        r'\bumożliwiając\w*\b',
        r'\bpodkreślając\w*\b',
        r'\bwykazując\w*\b',
        r'\boferując\w*\b',
        r'\bdostarczając\w*\b',
    ]

    print(f"\n=== Szukanie imiesłowów w: {docx_path} ===\n")

    for pattern in participle_patterns:
        regex = re.compile(pattern, re.IGNORECASE)
        print(f"\nWzorzec: {pattern}")
        print("=" * 80)

        found_count = 0
        for idx, para in enumerate(doc.paragraphs):
            text = para.text
            matches = regex.finditer(text)

            for match in matches:
                found_count += 1
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end]

                print(f"\nParagraf #{idx}:")
                print(f"  Znalezione: '{match.group()}'")
                print(f"  Kontekst: ...{context}...")

                # Sprawdź czy kończy zdanie lub punkt listy
                after_match = text[match.end():match.end() + 20].strip()
                if after_match and after_match[0] in ['.', '\n', '•', '']:
                    print(f"    KOŃCZY ZDANIE/PUNKT!")

        print(f"\nZnaleziono: {found_count} wystąpień")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Użycie: python find_participles.py <plik.docx>")
        sys.exit(1)

    find_all_participles(sys.argv[1])
