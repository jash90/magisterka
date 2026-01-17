#!/usr/bin/env python3
"""
Generator danych testowych dla analizy masowej pacjentów.
Tworzy pliki CSV i JSON z realistycznymi danymi pacjentów z zapaleniem naczyń.
"""

import numpy as np
import pandas as pd
import json
from datetime import datetime
from pathlib import Path

# Ustaw seed dla powtarzalności
np.random.seed(42)


def generate_patients(n: int) -> pd.DataFrame:
    """
    Generuj realistyczne dane pacjentów z zapaleniem naczyń.

    Args:
        n: liczba pacjentów do wygenerowania

    Returns:
        DataFrame z danymi pacjentów
    """
    patients = []

    for i in range(n):
        # Dane demograficzne
        wiek = np.random.randint(20, 86)
        plec = np.random.choice(['M', 'K'])
        wiek_rozpoznania = max(15, wiek - np.random.randint(0, 16))

        # Manifestacje narządowe (z realistycznymi prawdopodobieństwami)
        nerki = int(np.random.random() < 0.35)  # 35% szans
        serce = int(np.random.random() < 0.20)  # 20% szans
        csn = int(np.random.random() < 0.12)    # 12% szans
        neuro = int(np.random.random() < 0.28)  # 28% szans
        pokarmowy = int(np.random.random() < 0.15)  # 15% szans

        # Liczba zajętych narządów (skorelowana z manifestacjami)
        base_organs = nerki + serce + csn + neuro + pokarmowy
        liczba_narzadow = max(1, min(8, base_organs + np.random.randint(0, 3)))

        # Przebieg choroby
        # OIT bardziej prawdopodobne przy ciężkich manifestacjach
        oit_prob = 0.05 + 0.1 * serce + 0.15 * csn + 0.05 * nerki
        oit = int(np.random.random() < oit_prob)

        # Dializa tylko przy zajęciu nerek
        dializa = int(nerki and np.random.random() < 0.20)

        # Wartości laboratoryjne
        # Kreatynina wyższa przy zajęciu nerek
        if nerki:
            kreatynina = np.random.uniform(120, 350)
        else:
            kreatynina = np.random.uniform(60, 120)

        # CRP - rozkład log-normalny (więcej niskich wartości, niektóre bardzo wysokie)
        crp = np.exp(np.random.normal(3.5, 1.0))  # median ~33, może być 5-200+
        crp = min(250, max(5, crp))

        # Leczenie
        plazmaferezy = int(np.random.random() < (0.10 + 0.20 * nerki))  # częściej przy nerkach
        sterydy = np.random.uniform(0.3, 2.5)  # dawka w gramach
        czas_sterydow = np.random.randint(3, 36)  # miesiące

        # Powikłania
        powiklania_serce = int(np.random.random() < (0.10 + 0.15 * serce + 0.10 * oit))
        powiklania_infekcja = int(np.random.random() < (0.15 + 0.10 * oit + 0.05 * dializa))

        patient = {
            'id': f'P{i+1:04d}',
            'wiek': wiek,
            'plec': plec,
            'wiek_rozpoznania': wiek_rozpoznania,
            'liczba_narzadow': liczba_narzadow,
            'nerki': nerki,
            'serce': serce,
            'csn': csn,
            'neuro': neuro,
            'pokarmowy': pokarmowy,
            'oit': oit,
            'dializa': dializa,
            'kreatynina': round(kreatynina, 1),
            'crp': round(crp, 1),
            'plazmaferezy': plazmaferezy,
            'sterydy': round(sterydy, 2),
            'czas_sterydow': czas_sterydow,
            'powiklania_serce': powiklania_serce,
            'powiklania_infekcja': powiklania_infekcja
        }
        patients.append(patient)

    return pd.DataFrame(patients)


def save_csv(df: pd.DataFrame, filename: str):
    """Zapisz DataFrame do CSV."""
    df.to_csv(filename, index=False, encoding='utf-8')
    print(f"Zapisano {len(df)} pacjentów do {filename}")


def save_json_array(df: pd.DataFrame, filename: str):
    """Zapisz DataFrame jako tablica JSON."""
    records = df.to_dict(orient='records')
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"Zapisano {len(df)} pacjentów do {filename}")


def save_json_wrapped(df: pd.DataFrame, filename: str):
    """Zapisz DataFrame jako JSON z kluczem 'patients' i metadanymi."""
    records = df.to_dict(orient='records')
    data = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'total_patients': len(df),
            'description': 'Dane testowe pacjentów z zapaleniem naczyń',
            'source': 'generate_test_data.py'
        },
        'patients': records
    }
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Zapisano {len(df)} pacjentów do {filename}")


def main():
    """Generuj wszystkie pliki testowe."""
    output_dir = Path(__file__).parent

    print("Generowanie danych testowych...")
    print("=" * 50)

    # 1. CSV - 100 pacjentów
    df_100 = generate_patients(100)
    save_csv(df_100, output_dir / 'patients_100.csv')

    # 2. CSV - 500 pacjentów
    df_500 = generate_patients(500)
    save_csv(df_500, output_dir / 'patients_500.csv')

    # 3. CSV - 1000 pacjentów
    df_1000 = generate_patients(1000)
    save_csv(df_1000, output_dir / 'patients_1000.csv')

    # 4. JSON - 200 pacjentów (format tablicowy)
    df_200 = generate_patients(200)
    save_json_array(df_200, output_dir / 'patients_200.json')

    # 5. JSON - 300 pacjentów (format z kluczem)
    df_300 = generate_patients(300)
    save_json_wrapped(df_300, output_dir / 'patients_300_wrapped.json')

    print("=" * 50)
    print("Wygenerowano wszystkie pliki testowe!")

    # Podsumowanie statystyk
    print("\nStatystyki dla zestawu 1000 pacjentów:")
    print(f"  - Średni wiek: {df_1000['wiek'].mean():.1f} lat")
    print(f"  - Zajęcie nerek: {df_1000['nerki'].sum()} ({df_1000['nerki'].mean()*100:.1f}%)")
    print(f"  - Zajęcie serca: {df_1000['serce'].sum()} ({df_1000['serce'].mean()*100:.1f}%)")
    print(f"  - OIT: {df_1000['oit'].sum()} ({df_1000['oit'].mean()*100:.1f}%)")
    print(f"  - Dializa: {df_1000['dializa'].sum()} ({df_1000['dializa'].mean()*100:.1f}%)")
    print(f"  - Średnia kreatynina: {df_1000['kreatynina'].mean():.1f}")
    print(f"  - Średnie CRP: {df_1000['crp'].mean():.1f}")


if __name__ == '__main__':
    main()
