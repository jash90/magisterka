# Raport techniczny: Vasculitis XAI - System wyjasnialnej sztucznej inteligencji do predykcji smiertelnosci w zapaleniu naczyn

---

## 1. Cel i kontekst projektu

**Vasculitis XAI** to system wspomagania decyzji klinicznych (Clinical Decision Support System - CDSS) do predykcji ryzyka zgonu u pacjentow z zapaleniem naczyn. Projekt stanowi prace magisterska laczaca uczenie maszynowe z wyjasnialnoscia (Explainable AI / XAI) w kontekscie medycznym.

**Problem kliniczny:** Zapalenie naczyn (vasculitis) to grupa chorob autoimmunologicznych z niejednorodnym przebiegiem i trudna do przewidzenia prognoza. System dostarcza lekarzom i pacjentom narzedzie do oceny ryzyka z transparentnymi wyjasnieniami.

**Grupy docelowe:**
- Klinicysci (jezyk techniczny, wartosci SHAP, metryki)
- Pacjenci zaawansowani (podstawowa terminologia medyczna)
- Pacjenci bazowi (prosty jezyk, bez zargonu ML)

---

## 2. Architektura systemu

System sklada sie z 5 warstw:

```
[Dashboard Streamlit :8501] --> [API FastAPI :8000] --> [Model ML (XGBoost)]
                                      |                        |
                                      v                        v
                               [Agent RAG/LLM]          [Explainery XAI]
                               [ChromaDB :8001]    [SHAP | LIME | DALEX | EBM]
```

### Struktura katalogow

```
magisterka/
  src/
    api/          # REST API (FastAPI) - main.py, schemas.py
    data/         # Pipeline danych - preprocessing.py, imbalance.py, feature_engineering.py
    models/       # Modele ML - config.py, train.py, evaluate.py
    xai/          # Explainery - shap_explainer.py, lime_explainer.py, dalex_wrapper.py,
                  #              ebm_explainer.py, comparison.py
    agent/        # Agent konwersacyjny - rag.py, guardrails.py, prompts.py
  dashboard/      # Streamlit UI - streamlit_app.py
  models/saved/   # Artefakty modelu - best_model.joblib, feature_names.json
  data/           # Dane kliniczne - raw/, processed/
  tests/          # Testy - 5 modulow testowych
  scripts/        # Skrypty narzedzowe
  docker-compose.yml, Dockerfile, Dockerfile.streamlit
```

---

## 3. Pipeline danych (`src/data/`)

### 3.1 Preprocessing (`preprocessing.py` - 499 linii)

**Klasa `DataPreprocessor`** realizuje pelny pipeline przetwarzania danych:

1. **Wczytywanie** (`load_data`): CSV z separatorem `|` (specyficzny dla danych klinicznych). Automatyczne usuwanie kolumn identyfikatorow (Kod, ID, Patient_ID).

2. **Obsluga brakujacych wartosci** (`handle_missing_values`):
   - Traktowanie -1 jako brakujacych (konwencja w danych medycznych)
   - Strategie: median (domyslna), mean, mode, constant
   - Osobna obsluga dla kolumn numerycznych i kategorycznych

3. **Kodowanie kategoryczne** (`encode_categorical`): LabelEncoder z automatycznym wykrywaniem kolumn kategorycznych. Enkodery sa zapisywane do pozniejszego uzycia na nowych danych.

4. **Usuwanie korelacji** (`remove_high_correlation`): Macierz korelacji z progiem 0.95. Chroni kolumne docelowa przed usunieciem.

5. **Selekcja cech** (`select_features`): SelectKBest z mutual_info_classif lub f_classif. Domyslnie 20 cech.

6. **Skalowanie** (`scale_features`): StandardScaler lub MinMaxScaler z mozliwoscia fit/transform.

7. **Podzial danych** (`get_train_test_split`): Stratified split (80/20) z opcjonalnym zbiorem walidacyjnym.

8. **Pelny pipeline** (`prepare_pipeline`): Laczy wszystkie kroki w jednym wywolaniu. Zmienna docelowa: `Zgon` (smierc).

**19 kluczowych cech klinicznych** zdefiniowanych w `KLUCZOWE_CECHY`:
- Demografia: Wiek, Plec, Wiek_rozpoznania, Opoznienie_Rozpoznia
- Manifestacje: Sercowo-Naczyniowy, Nerki, Pokarmowy, Zajecie_CSN, Neurologiczny
- Przebieg: Liczba_Zajetych_Narzadow, Zaostrz_Wymagajace_OIT
- Laboratorium: Kreatynina, Max_CRP
- Leczenie: Plazmaferezy, Dializa, Sterydy_Dawka_g, Czas_Sterydow
- Powiklania: Serce/pluca, Infekcja

### 3.2 Inzynieria cech (`feature_engineering.py` - 394 linie)

**Klasa `FeatureEngineer`** tworzy cechy domenowe:

- **Wiekowe**: Kategorie wiekowe (0-40, 40-55, 55-65, 65-75, 75+), czas choroby, wczesne rozpoznanie (<50)
- **Narzadowe**: Zajecie narzadow krytycznych (serce/nerki/CSN), liczba narzadow krytycznych
- **Laboratoryjne**: Kategorie CRP (normalny/podwyzszony/wysoki/bardzo wysoki), Log_CRP, szacowane eGFR (wzor MDRD uproszczony: `175 * (kreatynina/88.4)^-1.154 * wiek^-0.203`), wskaznik wysokiej kreatyniny (>120)
- **Leczeniowe**: Wysokie dawki sterydow (>1g), dlugotrwale sterydy (>12 mies.), intensywne leczenie (plazmaferezy/dializa)
- **Powiklaniowe**: Suma powiklan, ciezkie powiklania
- **Risk Score**: Kliniczny score ryzyka (wiek, narzady, OIT, kreatynina, CRP, dializa) z kategoriami 0-3
- **Interakcje**: PolynomialFeatures degree=2, interaction_only=True

### 3.3 Obsluga niezbalansowania (`imbalance.py` - 482 linie)

**Klasa `ImbalanceHandler`** - 9 metod resamplingu:

| Metoda | Typ | Opis |
|--------|-----|------|
| SMOTE | Oversampling | Syntetyczne probki mniejszosciowe, k_neighbors adaptowane |
| ADASYN | Oversampling | Adaptacyjne - wiecej probek przy granicy decyzyjnej |
| RandomOverSampler | Oversampling | Duplikacja probek mniejszosciowych |
| RandomUnderSampler | Undersampling | Losowe usuwanie probek wiekszosciowych |
| TomekLinks | Undersampling | Usuwanie granicznych par probek |
| NearMiss | Undersampling | Wersje 1/2/3 - usuwanie na podstawie odleglosci |
| SMOTETomek | Kombinacja | SMOTE + czyszczenie TomekLinks |
| SMOTEENN | Kombinacja | SMOTE + Edited Nearest Neighbors |
| Combined | Kombinacja | Sekwencyjny SMOTE (50%) + undersampling (80%) |

**Automatyczna rekomendacja** (`recommend_strategy`):
- <10 probek mniejszosciowych -> random_oversampling
- Ratio <2 -> class_weights
- Ratio <5 -> SMOTE
- Ratio <10 -> SMOTETomek
- Ratio >=10 -> combined
- Maly dataset (<500) -> uproszczenie do SMOTE

**Dodatkowe**: `calculate_class_weights` (balanced/inverse/sqrt_inverse), `get_scale_pos_weight` (stosunek klas dla XGBoost).

---

## 4. Modele ML (`src/models/`)

### 4.1 Konfiguracja (`config.py` - 322 linie)

**7 typow modeli** z pelna konfiguracja:

| Model | Klasa | Kluczowe parametry |
|-------|-------|-------------------|
| Random Forest | sklearn.ensemble.RandomForestClassifier | n_estimators=100, max_depth=10, class_weight='balanced' |
| XGBoost | xgboost.XGBClassifier | learning_rate=0.1, max_depth=6, scale_pos_weight=5, eval_metric='auc' |
| LightGBM | lightgbm.LGBMClassifier | num_leaves=31, is_unbalance=True |
| Logistic Regression | sklearn.linear_model.LogisticRegression | penalty='l2', class_weight='balanced', solver='lbfgs' |
| SVM | sklearn.svm.SVC | kernel='rbf', class_weight='balanced', probability=True |
| Gradient Boosting | sklearn.ensemble.GradientBoostingClassifier | subsample=0.8 |
| Neural Network | sklearn.neural_network.MLPClassifier | hidden_layers=(100,50), activation='relu', early_stopping=True |

Kazdy model posiada siatki parametrow do GridSearchCV i RandomizedSearchCV.

**Progi medyczne** (MEDICAL_THRESHOLDS):
- AUC-ROC >= 0.75
- Sensitivity >= 0.80 (kluczowe - wykrywanie zagrozonego pacjenta)
- Specificity >= 0.60
- PPV >= 0.50
- NPV >= 0.90

**Kompatybilnosc TreeSHAP**: random_forest, xgboost, lightgbm, gradient_boosting

### 4.2 Trenowanie (`train.py` - 468 linii)

**Klasa `ModelTrainer`**:

- **`fit()`**: Trenowanie z opcjonalnym eval_set dla XGBoost/LightGBM. Logowanie historii trenowania (czas, n_samples, n_features).
- **`tune_hyperparameters()`**: GridSearchCV lub RandomizedSearchCV z StratifiedKFold (5 foldow). Scoring domyslny: roc_auc.
- **`cross_validate()`**: Multi-metric cross-validation.
- **`predict_proba()`**: Z fallback do decision_function + sigmoid dla modeli bez predict_proba.
- **`save_model()`**: joblib + metadata JSON (typ modelu, parametry, historia, timestamp).
- **`load_model()`**: Deserializacja z automatycznym wczytaniem metadanych.

**`train_multiple_models()`**: Trenowanie wielu modeli sekwencyjnie z opcjonalnym tuningiem.

### 4.3 Ewaluacja (`evaluate.py` - 607 linii)

**Klasa `ModelEvaluator`** - kompleksowa ewaluacja medyczna:

**Metryki**:
- Podstawowe: accuracy, precision, recall, F1
- Medyczne: sensitivity (=recall), specificity, PPV, NPV
- Zaawansowane: MCC (Matthews Correlation Coefficient), Cohen's Kappa, balanced accuracy
- Probabilistyczne: AUC-ROC, AUC-PR, Brier Score, Log Loss
- Macierz konfuzji: TP, TN, FP, FN

**Optymalizacja progu** (`find_optimal_threshold`):
- Metody: Youden's J (sensitivity + specificity - 1), F1, balanced accuracy
- Wymog minimalnej czulosci 0.80
- Testowanie 81 progow w zakresie [0.1, 0.9]

**Bootstrap CI** (`bootstrap_confidence_intervals`): 1000 iteracji, 95% CI dla AUC-ROC, AUC-PR, sensitivity, specificity, PPV, NPV.

**Wizualizacje**: ROC curve, Precision-Recall curve, macierz konfuzji (znormalizowana z wartosciami bezwzglednymi), porownanie modeli (bar chart).

### 4.4 Zapisany model

- **Plik**: `models/saved/best_model.joblib` (225 KB) - XGBoost
- **Metadane**: `best_model.json` - n_estimators=100, learning_rate=0.1, max_depth=6, scale_pos_weight=5, 719 probek treningowych, 20 cech, czas trenowania 0.228s
- **Cechy**: `feature_names.json` - 20 nazw cech w kolejnosci modelu

---

## 5. Explainery XAI (`src/xai/`)

### 5.1 SHAP Explainer (`shap_explainer.py` - 534 linie)

**Klasa `SHAPExplainer`** - wyjasnienia oparte na teorii gier (wartosci Shapleya):

**Automatyczny wybor explainera**:
- TreeExplainer: dla modeli drzewiastych (RF, XGBoost, LightGBM, GB) - dokladny, szybki O(TLD)
- LinearExplainer: dla modeli liniowych (LR, Ridge, Lasso)
- KernelExplainer: fallback model-agnostic - wolniejszy, probka tla max 100

**Wyjasnienia lokalne** (`explain_instance`):
- Wartosci SHAP dla kazdej cechy
- Base value (expected value modelu)
- Podzial na czynniki ryzyka (SHAP>0) i ochronne (SHAP<0)
- Ranking cech wg |SHAP value|

**Wyjasnienia globalne** (`get_global_importance`):
- Agregacja: mean_abs (domyslna), mean, max
- Na pelnym zbiorze danych lub probce

**Wizualizacje**: waterfall plot (shap.plots.waterfall), beeswarm/summary plot (shap.summary_plot), bar plot (custom), force plot (HTML interaktywny)

**Serializacja**:
- `to_json()`: Format dla LLM (top N czynnikow, podsumowanie)
- `to_patient_friendly()`: Tlumaczenie cech na polski (np. "Kreatynina" -> "Poziom wskaznika czynnosci nerek")

### 5.2 LIME Explainer (`lime_explainer.py` - 492 linie)

**Klasa `LIMEExplainer`** - lokalne modele zastecze:

**Mechanizm**: Perturbacja instancji (domyslnie 5000 probek), dopasowanie lokalnego modelu liniowego, wagi jako wyjasnienie.

**Konfiguracja**:
- LimeTabularExplainer ze statystykami z danych treningowych
- Dyskretyzacja zmiennych ciaglych (domyslnie wlaczona)
- Opcjonalne cechy kategoryczne

**Stabilnosc** (`calculate_stability`): 100 powtorzen, pomiar odchylenia standardowego wag, konsystencja top cechy, srednia niestabilnosc (std/|mean|).

**Wyjscie**: Wagi cech z warunkami (np. "Wiek > 60"), podzial na czynniki ryzyka/ochronne, intercept lokalnego modelu.

### 5.3 DALEX Wrapper (`dalex_wrapper.py` - 530 linii)

**Klasa `DALEXExplainer`** - model-agnostic:
- Feature importance (permutacyjna)
- Partial Dependence Plots (PDP)
- Accumulated Local Effects (ALE)
- Integracja z SHAP
- Analiza wydajnosci modelu

### 5.4 EBM Explainer (`ebm_explainer.py` - 541 linii)

**Klasa `EBMExplainer`** - Explainable Boosting Machine (Microsoft InterpretML):
- Model "glass-box" - interpretowalny z natury
- Rankingi waznosci cech
- Wykrywanie interakcji cech
- Addytywna dekompozycja modelu

### 5.5 Porownanie metod (`comparison.py` - 635 linii)

**Klasa `XAIComparison`** - cross-method analysis:

**Metryki zgodnosci**:
- Jaccard Similarity miedzy rankingami top-N cech
- Korelacja Spearmana miedzy scorami waznosci
- Zgodnosc kierunku wplywu cech (czy wszystkie metody zgadzaja sie ze cecha zwieksza/zmniejsza ryzyko)

**Stabilnosc**: Wielokrotne uruchomienia tego samego explainera, pomiar zmiennosci rankingu (Jaccard miedzy kolejnymi uruchomieniami), konsystencja top cechy.

**Wizualizacje**: Heatmapa pozycji w rankingu, heatmapa zgodnosci (Jaccard), raport tekstowy porownawczy.

---

## 6. Agent konwersacyjny (`src/agent/`)

### 6.1 RAG Pipeline (`rag.py` - 496 linii)

**Klasa `RAGPipeline`** - Retrieval-Augmented Generation:

**Stos technologiczny**: LangChain + ChromaDB + OpenAI GPT-4

**Baza wiedzy medycznej** (`add_medical_knowledge`):
- Przeglad zapalenia naczyn (typy: GPA, MPA, EGPA, AAV)
- Wskazniki laboratoryjne (CRP, kreatynina, ANCA)
- Leczenie (indukcja remisji, podtrzymanie, plazmafereza, dializa)

**Chunking**: RecursiveCharacterTextSplitter (chunk_size=1000, overlap=200)

**Retrieval**: ChromaDB similarity_search, top-k=3 dokumenty

**Generowanie odpowiedzi** (`generate_response`):
1. Sprawdzenie guardrails (pytanie uzytkownika)
2. Wybor promptu systemowego (basic/advanced/clinician)
3. Formatowanie kontekstu XAI (czynniki ryzyka/ochronne)
4. Pobranie kontekstu RAG z bazy wiedzy
5. Generacja z LLM (lub fallback bez API)
6. Walidacja odpowiedzi guardrails
7. Dodanie disclaimera (dla nie-klinicystow)

**Tryb fallback**: Gdy brak OPENAI_API_KEY - predefiniowane odpowiedzi tekstowe z poziomem ryzyka.

### 6.2 Guardrails (`guardrails.py` - 395 linii)

**Klasa `GuardrailsChecker`** - bezpieczenstwo agenta:

**Wykrywanie tresci szkodliwych** (regex po polsku):
- Samobojstwo/samookaleczenie: "chce umrzec", "odebrac sobie zycie", "targac sie na zycie"
- Przemoc: "skrzywdzic kogos", "zabic kogos"
- **Reakcja**: Numery telefonow kryzysowych (116 123, 800 70 2222, 116 111)

**Blokowanie porad medycznych**:
- Diagnozy: "czy mam...", "zdiagnozuj", "co mi jest"
- Leki: "jaki lek powinienem", "zmien dawke", "odstawic lek"
- Prognozy: "ile bede zyc", "kiedy umre"
- **Reakcja**: Przekierowanie do lekarza z konkretnymi wskazowkami

**Walidacja odpowiedzi** (forbidden patterns):
- Blokowanie liczbowych prognoz ("50% szans na przezycie")
- Blokowanie diagnoz ("masz chorobe X")
- Blokowanie zalecen lekowych ("zaz lek X mg")
- Blokowanie krytyki lekarza ("lekarz sie myli")
- Sanityzacja: zamiana na "[tresc usunieta]" + przypomnienie

**Health literacy validation**: Sprawdzenie odpowiednosci jezyka - dla poziomu basic wykrywa terminy techniczne (SHAP, LIME, predykcja, algorytm, AUC, ROC) i zbyt dlugie zdania (>25 slow).

**Wykrywanie stresu emocjonalnego**: Wzorce "boje sie", "nie radze sobie", "trace nadzieje" -> empatyczny prefiks odpowiedzi.

### 6.3 Prompty (`prompts.py` - 399 linii)

**3 poziomy promptow systemowych**:

| Poziom | Jezyk | Dozwolone | Zakazane |
|--------|-------|-----------|----------|
| Basic | "uczen 8 klasy" | czynniki, wskazniki zdrowia | SHAP, model, algorytm, predykcja, wartosci liczbowe |
| Advanced | Podstawowa terminologia | czynniki ryzyka, prawdopodobienstwo, CRP | wartosci SHAP, terminologia ML |
| Clinician | Pelna terminologia | wartosci SHAP/LIME, CI, ROC, PDP | zalecenia terapeutyczne |

**Tlumaczenia cech** (`FEATURE_TRANSLATIONS`): 19 mapowan np. "Manifestacja_Nerki" -> "Stan nerek", "Zaostrz_Wymagajace_OIT" -> "Historia pobytu na intensywnej terapii"

**Szablony wyjasnien**: Osobne dla pacjenta (prosty format) i klinicysty (Markdown z tabelami, CI, metoda XAI, base_value)

---

## 7. REST API (`src/api/`)

### 7.1 Aplikacja FastAPI (`main.py` - 902 linie)

**Konfiguracja**: CORS (all origins), Swagger UI (/docs), ReDoc (/redoc)

**AppState** - stan globalny:
- Model (joblib), feature_names (JSON), LIME/SHAP explainery, RAG pipeline
- Tryb demo: `ALLOW_DEMO=true`, `FORCE_API_MODE=false`
- Cache globalnej waznosci cech

**Endpointy**:

| Metoda | Endpoint | Funkcja |
|--------|----------|---------|
| GET | `/` | Info API |
| GET | `/health` | Health check (model_loaded, version, timestamp) |
| POST | `/predict` | Pojedyncza predykcja - model.predict_proba lub demo heurystyka |
| POST | `/predict/batch` | Batch do 10,000 pacjentow - vectorized numpy, chunking |
| POST | `/explain/shap` | Wyjasnienie SHAP (base_value, shap_values, contributions) |
| POST | `/explain/lime` | Wyjasnienie LIME (intercept, feature_weights) |
| POST | `/explain/patient` | Wyjasnienie przyjazne pacjentowi (tlumaczenia cech, ryzyko slowne) |
| GET | `/model/info` | Metadane modelu (typ, cechy, metryki) |
| GET | `/model/global-importance` | Ranking waznosci cech |
| POST | `/chat` | Rozmowa z agentem (wykrywanie intencji, disclaimery) |
| GET/POST | `/config/demo-mode` | Zarzadzanie trybem demo |

**Batch prediction** (`/predict/batch`):
- Vectorized: `patients_to_matrix()` -> numpy float32 -> `model.predict_proba()` na calej macierzy
- Fallback: Individual processing per patient
- Risk factors: Global importance + thresholds per feature
- Summary: count per risk level, avg/median/min/max probability
- Wydajnosc: 1000+ pacjentow w <100ms

**Demo mode**: Heurystyczna funkcja ryzyka (wiek >50, narzady *0.1, nerki +0.15, OIT +0.25, dializa +0.2), zakres [0.05, 0.95]

### 7.2 Schematy Pydantic (`schemas.py` - 435 linii)

**Enums**: HealthLiteracyLevel (basic/advanced/clinician), RiskLevel (low/moderate/high), XAIMethod (lime/shap/dalex/ebm)

**PatientInput** - 19 pol z walidacja:
- `wiek`: float, 0-120
- `plec`: int, 0-1 (K/M)
- Manifestacje: int 0-1 (binarne)
- Laboratorium: Optional[float] z dolnymi granicami
- Mapping pol Pydantic -> nazwy modelu (np. `manifestacja_sercowo_naczyniowy` -> `Manifestacja_Sercowo-Naczyniowy`)

**Risk levels**: <0.3 = LOW, 0.3-0.7 = MODERATE, >0.7 = HIGH

---

## 8. Dashboard Streamlit (`dashboard/streamlit_app.py` - 1560 linii)

### 8.1 Tryb pojedynczego pacjenta

**Sidebar - formularz**:
- Dane demograficzne: wiek, plec, wiek rozpoznania
- Manifestacje: 5 checkboxow + slider narzadow
- Przebieg: OIT, kreatynina, CRP
- Leczenie: plazmaferezy, dializa, sterydy
- Powiklania: serce/pluca, infekcje

**Wyniki**:
- Gauge chart (Plotly) z kolorami ryzyko: zielony/zolty/czerwony
- Risk level badge z opisem
- Top 5 czynnikow (strzalki gora/dol + wartosc contribution)

**Zakladki**:
1. **SHAP**: Waterfall chart (wplyw kazdego czynnika)
2. **LIME**: Bar chart (bezwzgledna waznosc)
3. **Porownanie**: Rankingi obok siebie + informacja o zgodnosci
4. **Chat AI**: Streamlit chat_input z prostym wykrywaniem intencji (wynik/czynnik/pomoc)

### 8.2 Tryb analizy masowej

**Upload**: CSV/JSON, auto-detect separatora (`,`, `;`, `\t`, `|`)

**Normalizacja kolumn** (`COLUMN_MAPPING`): 40+ mapowan nazw (polski/angielski), np. `age`->`wiek`, `kidneys`->`manifestacja_nerki`, `icu`->`zaostrz_wymagajace_oit`

**Przetwarzanie**: Chunking po 1000 pacjentow, batch API lub fallback demo, progress bar

**Wizualizacje**:
- Pie chart rozkladu ryzyka (Plotly)
- Histogram prawdopodobienstw z liniami progowymi (30%, 70%)
- Scatter plot wiek vs ryzyko (kolorowany wg poziomu)
- Tabela z kolorowaniem wierszy wg ryzyka
- Filtry: po poziomie ryzyka, sortowanie

**Eksport**: CSV (UTF-8-sig) i JSON z podsumowaniem i timestamp

### 8.3 Style CSS

Custom CSS z klasami: `.risk-low/moderate/high`, `.disclaimer`, `.info-card`, `.batch-header`, `.risk-badge`, `.upload-zone`. Dark mode support (transparent backgrounds, plotly_dark template).

---

## 9. Deployment (Docker)

**3 serwisy** w `docker-compose.yml`:

| Serwis | Obraz | Port | Opis |
|--------|-------|------|------|
| api | Python 3.11-slim + FastAPI | 8000 | Backend z modelem |
| dashboard | Python 3.11-slim + Streamlit | 8501 | Frontend UI |
| chroma | chromadb/chroma:latest | 8001 | Baza wektorowa RAG |

**Siec**: vasculitis-network (bridge)
**Volumes**: models/ i data/ (read-only), chroma-data (persistent)
**Health checks**: curl na /health (API) i /_stcore/health (Streamlit), interval 30s
**Zaleznosci**: Dashboard czeka na zdrowe API (`condition: service_healthy`)

---

## 10. Stos technologiczny

### ML/AI
| Biblioteka | Zastosowanie |
|-----------|-------------|
| scikit-learn | Preprocessing, modele (RF, LR, SVM, GB, MLP), metryki, CV |
| XGBoost | Gradient boosting (glowny model) |
| LightGBM | Alternatywny gradient boosting |
| imbalanced-learn | SMOTE, ADASYN, Tomek Links, NearMiss, kombinacje |
| SHAP | TreeSHAP/KernelSHAP wyjasnienia |
| LIME | Lokalne modele zastecze |
| DALEX | Model-agnostic XAI |
| InterpretML | EBM (glass-box model) |

### LLM/RAG
| Biblioteka | Zastosowanie |
|-----------|-------------|
| LangChain | Orkiestracja RAG, prompty, chat |
| langchain-openai | Integracja GPT-4 |
| ChromaDB | Baza wektorowa (embeddings) |
| sentence-transformers | Embeddings semantyczne |
| OpenAI API | GPT-4 (text-embedding-ada-002) |

### Web/API
| Biblioteka | Zastosowanie |
|-----------|-------------|
| FastAPI | REST API z automatyczna dokumentacja |
| Pydantic | Walidacja schematow I/O |
| Uvicorn | ASGI server |
| Streamlit | Dashboard interaktywny |
| Plotly | Wykresy interaktywne (gauge, waterfall, scatter, histogram, pie) |

### Inne
| Biblioteka | Zastosowanie |
|-----------|-------------|
| numpy/pandas | Operacje na danych |
| matplotlib/seaborn | Wizualizacje statyczne (ROC, confusion matrix) |
| scipy | Korelacja Spearmana w porownaniach XAI |
| joblib | Serializacja modeli |
| pytest | Testy jednostkowe |
| Docker/docker-compose | Konteneryzacja |

---

## 11. Bezpieczenstwo i etyka

### Guardrails - 7 kategorii zabezpieczen
1. **Wykrywanie tresci samobojczych** - telefony kryzysowe PL
2. **Blokowanie diagnoz** - przekierowanie do lekarza
3. **Blokowanie porad lekowych** - odmowa zmiany dawkowania
4. **Filtrowanie prognoz** - brak "ile bede zyc"
5. **Sanityzacja odpowiedzi** - usuwanie zakazanych wzorcow
6. **Obowiazkowe disclaimery** - "narzedzie informacyjne, nie zastepuje lekarza"
7. **Wykrywanie stresu emocjonalnego** - empatyczny prefiks

### Adaptowalnosc jezykowa
- 3 poziomy health literacy z dedykowanymi promptami
- Walidacja odpowiednosci jezyka technicznego
- Tlumaczenia 19 cech medycznych na jezyk zrozumialy

### Metryki medyczne
- Priorytet sensitivity >= 0.80 (wykrywanie zagrozonego pacjenta)
- NPV >= 0.90 (pewnosc negatywnej predykcji)
- Bootstrap 95% CI dla kwantyfikacji niepewnosci
- Optymalizacja progu z gwarancja minimalnej czulosci

---

## 12. Testy (`tests/`)

5 modulow testowych (77 testow, 74 passed, 3 skipped):
- `test_preprocessing.py`: DataPreprocessor, ImbalanceHandler, FeatureEngineer
- `test_models.py`: ModelTrainer, ModelEvaluator, konfiguracje
- `test_xai.py`: SHAP, LIME, DALEX, EBM, porownanie
- `test_agent.py`: RAGPipeline, GuardrailsChecker, Prompts
- `test_api.py`: Endpointy FastAPI, schematy Pydantic, walidacja request/response

3 testy skipped wymagaja uruchomionego serwera API (testy integracyjne).

---

## 13. Podsumowanie

| Metryka | Wartosc |
|---------|---------|
| Laczna liczba linii kodu Python | ~9,690 (src + dashboard) |
| Moduly zrodlowe | 17 plikow .py |
| Metody XAI | 4 (SHAP, LIME, DALEX, EBM) |
| Modele ML | 7 typow (XGBoost aktywny) |
| Endpointy API | 11 |
| Cechy kliniczne | 19 (+ engineered) |
| Metody resamplingu | 9 |
| Metryki ewaluacji | 15+ |
| Poziomy health literacy | 3 |
| Kategorie guardrails | 7 |
| Testy | 77 (74 passed, 3 skipped) |
| Konteneryzacja | 3 serwisy Docker |
