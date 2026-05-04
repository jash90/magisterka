# Raport testowy systemu Vasculitis XAI

## Cel

Raport podsumowuje weryfikację części aplikacyjnej pracy magisterskiej. Obejmuje backend FastAPI, frontend React/Vite oraz zgodność kontraktu danych pacjenta z aktualnym schematem 20 cech klinicznych.

## Zakres testów

- Walidacja schematów wejściowych i wyjściowych API.
- Konwersja danych pacjenta do kolejności cech używanej przez model.
- Obsługa predykcji pojedynczego pacjenta.
- Sprawdzenie endpointów wyjaśnialności i metadanych modelu.
- Produkcyjny build frontendu React/Vite.
- Kontrola integracji FastAPI z aktualnym kontraktem danych.
- Obsługa lokalnego artefaktu modelu o niezgodnej liczbie cech.

## Wyniki

| Obszar | Polecenie / procedura | Wynik |
| --- | --- | --- |
| Testy API | `venv/bin/python -m pytest tests/test_api.py -q` | 12 passed, 3 skipped |
| Składnia Python | `venv/bin/python -m py_compile src/api/main.py src/api/schemas.py` | OK |
| Build frontendu | `npm --prefix frontend run build` | OK |
| Predykcja przez FastAPI | `POST /predict` z payloadem 20 cech | 200 OK |
| Walidacja modelu | porównanie `n_features_in_` z `feature_names.json` | niezgodny model jest pomijany, API działa w trybie demo |

## Uwagi

Ostrzeżenie Vite dotyczące dużego rozmiaru bundla wynika głównie z biblioteki Plotly. Nie blokuje działania aplikacji, ale w dalszym rozwoju warto rozważyć dynamiczny import wykresów albo wydzielenie ciężkich widoków XAI do osobnych chunków.

Aktualny lokalny artefakt `models/saved/best_model.joblib` oczekuje 71 cech, natomiast kontrakt aplikacji i plik `feature_names.json` opisują 20 cech. Backend wykrywa tę niespójność podczas startu i przełącza się w tryb demo, dzięki czemu aplikacja pozostaje używalna i nie zwraca błędu predykcji.

## Wniosek

Część aplikacyjna spełnia wymagania pracy inżyniersko-aplikacyjnej na poziomie demonstratora: posiada działający backend, frontend, walidację danych, mechanizmy XAI oraz podstawowy zestaw testów regresyjnych. Przed wersją końcową należy uzupełnić lub wytrenować artefakt modelu zgodny z finalnym kontraktem 20 cech oraz wykonać walidację na niezależnym zbiorze danych.
