#!/usr/bin/env python3
"""
Skrypt do generowania testowych danych pacjentów.

Użycie:
    python scripts/generate_test_data.py [liczba_pacjentów] [format]

Przykłady:
    python scripts/generate_test_data.py 1000 csv
    python scripts/generate_test_data.py 5000 json
"""

import sys
import json
import random
from pathlib import Path


def generate_patient(patient_id: int) -> dict:
    """Generuj losowe dane pacjenta."""
    wiek = random.randint(25, 85)
    plec = random.choice(['M', 'K'])

    # Im starszy pacjent, tym więcej narządów może być zajętych
    age_factor = (wiek - 25) / 60  # 0-1
    liczba_narzadow = min(5, max(1, int(random.gauss(2 + age_factor * 2, 1))))

    # Manifestacje narządowe - prawdopodobieństwo zależy od liczby narządów
    nerki = 1 if random.random() < 0.3 + age_factor * 0.3 else 0
    serce = 1 if random.random() < 0.2 + age_factor * 0.2 else 0
    csn = 1 if random.random() < 0.1 + age_factor * 0.1 else 0
    neuro = 1 if random.random() < 0.15 + age_factor * 0.15 else 0
    pokarm = 1 if random.random() < 0.1 else 0

    # Ciężkość choroby
    oit = 1 if random.random() < 0.1 + (nerki + serce) * 0.1 else 0
    dializa = 1 if nerki == 1 and random.random() < 0.3 else 0
    plazmaferezy = 1 if random.random() < 0.15 else 0

    # Parametry laboratoryjne
    base_creatinine = 80 + age_factor * 40
    kreatynina = max(50, base_creatinine + random.gauss(0, 30) + dializa * 80)

    base_crp = 20 + oit * 30 + serce * 15 + nerki * 10
    crp = max(5, base_crp + random.gauss(0, 20))

    # Leczenie
    sterydy = round(0.3 + random.random() * 0.7 + oit * 0.3, 1)

    return {
        'id': f'P{patient_id:05d}',
        'wiek': wiek,
        'plec': plec,
        'wiek_rozpoznania': max(20, wiek - random.randint(1, 15)),
        'liczba_narzadow': liczba_narzadow,
        'nerki': nerki,
        'serce': serce,
        'csn': csn,
        'neuro': neuro,
        'pokarm': pokarm,
        'oit': oit,
        'dializa': dializa,
        'kreatynina': round(kreatynina, 1),
        'crp': round(crp, 1),
        'plazmaferezy': plazmaferezy,
        'sterydy': sterydy
    }


def generate_patients(n: int) -> list:
    """Generuj listę pacjentów."""
    return [generate_patient(i + 1) for i in range(n)]


def save_csv(patients: list, filepath: Path):
    """Zapisz do CSV."""
    import csv

    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        if patients:
            writer = csv.DictWriter(f, fieldnames=patients[0].keys())
            writer.writeheader()
            writer.writerows(patients)

    print(f"Zapisano {len(patients)} pacjentów do {filepath}")


def save_json(patients: list, filepath: Path):
    """Zapisz do JSON."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(patients, f, ensure_ascii=False, indent=2)

    print(f"Zapisano {len(patients)} pacjentów do {filepath}")


def main():
    # Parsuj argumenty
    n_patients = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    format_type = sys.argv[2].lower() if len(sys.argv) > 2 else 'csv'

    # Generuj dane
    print(f"Generowanie {n_patients} pacjentów...")
    patients = generate_patients(n_patients)

    # Zapisz
    output_dir = Path(__file__).parent.parent / 'dashboard' / 'test_data'
    output_dir.mkdir(parents=True, exist_ok=True)

    if format_type == 'csv':
        filepath = output_dir / f'test_{n_patients}_patients.csv'
        save_csv(patients, filepath)
    elif format_type == 'json':
        filepath = output_dir / f'test_{n_patients}_patients.json'
        save_json(patients, filepath)
    else:
        print(f"Nieobsługiwany format: {format_type}")
        sys.exit(1)

    # Statystyki
    ages = [p['wiek'] for p in patients]
    nerki_count = sum(p['nerki'] for p in patients)
    oit_count = sum(p['oit'] for p in patients)

    print(f"\nStatystyki:")
    print(f"  Średni wiek: {sum(ages)/len(ages):.1f}")
    print(f"  Zakres wieku: {min(ages)}-{max(ages)}")
    print(f"  Zajęcie nerek: {nerki_count} ({nerki_count/len(patients)*100:.1f}%)")
    print(f"  Wymagający OIT: {oit_count} ({oit_count/len(patients)*100:.1f}%)")


if __name__ == '__main__':
    main()
