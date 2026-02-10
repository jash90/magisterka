"""
Generowanie syntetycznych danych referencyjnych dla SHAP/LIME.

Tworzy X_reference.npy z 300 próbkami na podstawie 20 cech z feature_names.json.
SHAP i LIME wymagają danych tła (background data) do obliczania wyjaśnień.
"""

import json
import numpy as np
from pathlib import Path


def generate_reference_data(n_samples: int = 300, seed: int = 42) -> np.ndarray:
    """
    Generuj syntetyczne dane referencyjne dla 20 cech modelu.

    Rozkłady dobrane na podstawie klinicznej wiedzy o zapaleniu naczyń:
    - Cechy binarne (manifestacje, powikłania): Bernoulli z prawdopodobieństwem 0.15-0.35
    - Wiek rozpoznania: N(50, 15), obcięty do [18, 90]
    - Opóźnienie rozpoznania: Exponential(6), obcięty do [0, 120]
    - Kreatynina: LogNormal(4.6, 0.4) ~ mediana ~100 μmol/L
    - Eozynofilia: LogNormal(2.3, 1.0) ~ mediana ~10%
    """
    rng = np.random.default_rng(seed)

    # Kolejność cech zgodna z feature_names.json
    features = np.zeros((n_samples, 20), dtype=np.float64)

    # 0: Wiek_rozpoznania - N(50, 15), obcięty [18, 90]
    ages = rng.normal(50, 15, n_samples)
    features[:, 0] = np.clip(ages, 18, 90)

    # 1: Opoznienie_Rozpoznia - Exponential(6), miesiące, obcięty [0, 120]
    delays = rng.exponential(6, n_samples)
    features[:, 1] = np.clip(delays, 0, 120)

    # 2-11: Cechy binarne (manifestacje narządowe i inne)
    binary_features = {
        2: 0.25,   # Manifestacja_Miesno-Szkiel
        3: 0.30,   # Manifestacja_Skora
        4: 0.15,   # Manifestacja_Wzrok
        5: 0.20,   # Manifestacja_Nos/Ucho/Gardlo
        6: 0.35,   # Manifestacja_Oddechowy
        7: 0.25,   # Manifestacja_Sercowo-Naczyniowy
        8: 0.20,   # Manifestacja_Pokarmowy
        9: 0.15,   # Manifestacja_Moczowo-Plciowy
        10: 0.10,  # Manifestacja_Zajecie_CSN
        11: 0.20,  # Manifestacja_Neurologiczny
    }
    for idx, prob in binary_features.items():
        features[:, idx] = rng.binomial(1, prob, n_samples)

    # 12: Liczba_Zajetych_Narzadow - Poisson(2.5), obcięta [0, 12]
    organ_counts = rng.poisson(2.5, n_samples)
    features[:, 12] = np.clip(organ_counts, 0, 12)

    # 13: Zaostrz_Wymagajace_Hospital - binarne, p=0.30
    features[:, 13] = rng.binomial(1, 0.30, n_samples)

    # 14: Zaostrz_Wymagajace_OIT - binarne, p=0.15
    features[:, 14] = rng.binomial(1, 0.15, n_samples)

    # 15: Kreatynina - LogNormal ~ mediana 100 μmol/L
    creatinine = rng.lognormal(4.6, 0.4, n_samples)
    features[:, 15] = np.clip(creatinine, 30, 800)

    # 16: Czas_Sterydow - Exponential(18), miesiące
    steroid_time = rng.exponential(18, n_samples)
    features[:, 16] = np.clip(steroid_time, 0, 120)

    # 17: Plazmaferezy - binarne, p=0.20
    features[:, 17] = rng.binomial(1, 0.20, n_samples)

    # 18: Eozynofilia_Krwi_Obwodowej_Wartosc - LogNormal ~ mediana ~10
    eosinophilia = rng.lognormal(2.3, 1.0, n_samples)
    features[:, 18] = np.clip(eosinophilia, 0, 100)

    # 19: Powiklania_Neurologiczne - binarne, p=0.15
    features[:, 19] = rng.binomial(1, 0.15, n_samples)

    return features


def main():
    project_root = Path(__file__).parent.parent
    feature_names_path = project_root / "models" / "saved" / "feature_names.json"
    output_path = project_root / "models" / "saved" / "X_reference.npy"

    # Wczytaj nazwy cech dla weryfikacji
    with open(feature_names_path, 'r') as f:
        feature_names = json.load(f)

    assert len(feature_names) == 20, f"Oczekiwano 20 cech, znaleziono {len(feature_names)}"

    # Generuj dane
    X_reference = generate_reference_data(n_samples=300)

    assert X_reference.shape == (300, 20), f"Zły kształt: {X_reference.shape}"

    # Zapisz
    np.save(output_path, X_reference)
    print(f"Zapisano X_reference.npy: {X_reference.shape} do {output_path}")

    # Podsumowanie
    print("\nStatystyki dla każdej cechy:")
    for i, name in enumerate(feature_names):
        col = X_reference[:, i]
        print(f"  {name:45s} min={col.min():.2f}  mean={col.mean():.2f}  max={col.max():.2f}")


if __name__ == "__main__":
    main()
