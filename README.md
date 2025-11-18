# Vasculitis XAI - System XAI do predykcji śmiertelności w zapaleniu naczyń

System wyjaśnialnej sztucznej inteligencji (XAI) do predykcji śmiertelności
u pacjentów z zapaleniem naczyń, z interfejsem dla klinicystów i pacjentów.

## Spis treści

- [Przegląd](#przegląd)
- [Architektura](#architektura)
- [Instalacja](#instalacja)
- [Użycie](#użycie)
- [Struktura projektu](#struktura-projektu)
- [Metody XAI](#metody-xai)
- [API](#api)
- [Testy](#testy)
- [Docker](#docker)

## Przegląd

System łączy modele uczenia maszynowego (XGBoost, Random Forest, LightGBM)
z metodami wyjaśnialnej AI (LIME, SHAP, DALEX, EBM) oraz agentem konwersacyjnym
opartym na LLM do generowania wyjaśnień zrozumiałych dla pacjentów i klinicystów.

### Główne funkcje

- **Predykcja ryzyka** - ocena prawdopodobieństwa zgonu na podstawie cech klinicznych
- **Wyjaśnienia XAI** - identyfikacja kluczowych czynników wpływających na predykcję
- **Agent konwersacyjny** - interaktywna rozmowa o wynikach z dostosowaniem języka
- **Dashboard** - interfejs graficzny dla klinicystów
- **API REST** - integracja z systemami zewnętrznymi

## Architektura

```
┌─────────────────────────────────────────────────────────────────┐
│                    INTERFEJS (Streamlit)                        │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                   AGENT LLM (LangChain + RAG)                   │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                      MODUŁ XAI                                  │
│    LIME │ SHAP │ DALEX │ EBM                                    │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                   KLASYFIKATORY ML                              │
│    Random Forest │ XGBoost │ LightGBM                           │
└─────────────────────────────────────────────────────────────────┘
```

## Instalacja

### Wymagania

- Python 3.10+
- pip lub conda

### Instalacja lokalna

```bash
# Klonowanie repozytorium
git clone https://github.com/user/vasculitis-xai.git
cd vasculitis-xai

# Utworzenie środowiska wirtualnego
python -m venv venv
source venv/bin/activate  # Linux/Mac
# lub: venv\Scripts\activate  # Windows

# Instalacja zależności
pip install -r requirements.txt

# Instalacja pakietu w trybie deweloperskim
pip install -e .
```

### Konfiguracja

Utwórz plik `.env` na podstawie `.env.example`:

```bash
cp .env.example .env
```

Uzupełnij wymagane zmienne:

```env
OPENAI_API_KEY=your-api-key-here
MODEL_PATH=models/saved/best_model.joblib
```

## Użycie

### Uruchomienie API

```bash
# Bezpośrednio
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Lub przez skrypt
python -m src.api.main
```

API będzie dostępne pod adresem: http://localhost:8000

Dokumentacja API: http://localhost:8000/docs

### Uruchomienie dashboardu

```bash
streamlit run dashboard/streamlit_app.py
```

Dashboard będzie dostępny pod adresem: http://localhost:8501

### Użycie w kodzie Python

```python
from src.data.preprocessing import DataPreprocessor
from src.models.train import ModelTrainer
from src.xai.shap_explainer import SHAPExplainer

# Preprocessing
preprocessor = DataPreprocessor()
df = preprocessor.load_data('data/raw/aktualne_dane.csv', separator='|')
X, y, feature_names = preprocessor.prepare_pipeline(df, target_col='Zgon')

# Trenowanie modelu
trainer = ModelTrainer('xgboost')
trainer.fit(X_train, y_train, feature_names=feature_names)

# Wyjaśnienie SHAP
shap_explainer = SHAPExplainer(trainer.model, X_train, feature_names)
explanation = shap_explainer.explain_instance(X_test[0])
print(shap_explainer.to_json(explanation))
```

## Struktura projektu

```
vasculitis-xai/
├── src/
│   ├── data/                    # Preprocessing i obsługa danych
│   │   ├── preprocessing.py     # Klasa DataPreprocessor
│   │   ├── imbalance.py         # Obsługa niezbalansowanych danych
│   │   └── feature_engineering.py
│   ├── models/                  # Modele ML
│   │   ├── config.py            # Konfiguracja hiperparametrów
│   │   ├── train.py             # Klasa ModelTrainer
│   │   └── evaluate.py          # Ewaluacja z metrykami medycznymi
│   ├── xai/                     # Metody XAI
│   │   ├── lime_explainer.py    # LIME
│   │   ├── shap_explainer.py    # SHAP
│   │   ├── dalex_wrapper.py     # DALEX
│   │   ├── ebm_explainer.py     # EBM
│   │   └── comparison.py        # Porównanie metod
│   ├── agent/                   # Agent konwersacyjny
│   │   ├── prompts.py           # Szablony promptów
│   │   ├── rag.py               # Pipeline RAG
│   │   └── guardrails.py        # Bezpieczeństwo
│   └── api/                     # API FastAPI
│       ├── main.py              # Endpointy
│       └── schemas.py           # Schematy Pydantic
├── dashboard/
│   └── streamlit_app.py         # Dashboard Streamlit
├── tests/                       # Testy jednostkowe
├── data/
│   ├── raw/                     # Dane surowe
│   └── processed/               # Dane przetworzone
├── models/
│   └── saved/                   # Zapisane modele
├── docs/                        # Dokumentacja
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## Metody XAI

### LIME (Local Interpretable Model-agnostic Explanations)

```python
from src.xai.lime_explainer import LIMEExplainer

explainer = LIMEExplainer(model, X_train, feature_names)
explanation = explainer.explain_instance(instance, num_features=10)
explainer.plot_explanation(explanation)
```

### SHAP (SHapley Additive exPlanations)

```python
from src.xai.shap_explainer import SHAPExplainer

explainer = SHAPExplainer(model, X_train, feature_names)
explanation = explainer.explain_instance(instance)
explainer.plot_waterfall(explanation)
```

### DALEX

```python
from src.xai.dalex_wrapper import DALEXWrapper

wrapper = DALEXWrapper(model, X, y, feature_names)
bd_exp = wrapper.explain_instance_break_down(instance)
wrapper.plot_break_down(bd_exp)
```

### EBM (Explainable Boosting Machine)

```python
from src.xai.ebm_explainer import EBMExplainer

ebm = EBMExplainer(feature_names=feature_names)
ebm.fit(X_train, y_train)
local_exp = ebm.explain_local(instance)
ebm.plot_local_explanation(local_exp)
```

## API

### Endpointy

| Metoda | Endpoint | Opis |
|--------|----------|------|
| GET | `/health` | Status API |
| POST | `/predict` | Predykcja ryzyka |
| POST | `/explain/shap` | Wyjaśnienie SHAP |
| POST | `/explain/lime` | Wyjaśnienie LIME |
| POST | `/explain/patient` | Wyjaśnienie dla pacjenta |
| GET | `/model/info` | Informacje o modelu |
| POST | `/chat` | Rozmowa z agentem |

### Przykład użycia API

```python
import requests

# Predykcja
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "wiek": 55,
        "plec": 1,
        "liczba_zajetych_narzadow": 3,
        "manifestacja_nerki": 1,
        "zaostrz_wymagajace_oit": 0
    }
)

result = response.json()
print(f"Ryzyko: {result['probability']:.1%}")
print(f"Poziom: {result['risk_level']}")
```

## Testy

```bash
# Uruchomienie wszystkich testów
pytest tests/ -v

# Testy z coverage
pytest tests/ --cov=src --cov-report=html

# Pojedynczy moduł testów
pytest tests/test_preprocessing.py -v
```

## Docker

### Budowanie obrazów

```bash
# API
docker build -t vasculitis-api -f Dockerfile .

# Dashboard
docker build -t vasculitis-dashboard -f Dockerfile.streamlit .
```

### Docker Compose

```bash
# Uruchomienie wszystkich usług
docker-compose up -d

# Zatrzymanie
docker-compose down

# Logi
docker-compose logs -f
```

Usługi:
- API: http://localhost:8000
- Dashboard: http://localhost:8501
- ChromaDB: http://localhost:8001

## Metryki medyczne

System używa metryk istotnych klinicznie:

- **AUC-ROC** - zdolność dyskryminacji modelu
- **Sensitivity (Recall)** - wykrywalność przypadków pozytywnych (kluczowe!)
- **Specificity** - wykrywalność przypadków negatywnych
- **PPV (Precision)** - dodatnia wartość predykcyjna
- **NPV** - ujemna wartość predykcyjna
- **Brier Score** - kalibracja prawdopodobieństw

## Guardrails bezpieczeństwa

System zawiera zabezpieczenia:

- Wykrywanie treści o samobójstwie → kierowanie do specjalistów
- Blokowanie prób uzyskania diagnoz medycznych
- Filtrowanie próśb o zalecenia farmakologiczne
- Usuwanie konkretnych prognoz liczbowych z odpowiedzi
- Obowiązkowe disclaimery medyczne

## Licencja

MIT License

## Autorzy

Projekt magisterski - System XAI do predykcji śmiertelności w zapaleniu naczyń

## Cytowanie

```bibtex
@thesis{vasculitis_xai_2024,
  title={System XAI do predykcji śmiertelności w zapaleniu naczyń},
  author={...},
  year={2024},
  school={...}
}
```
