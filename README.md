# Vasculitis XAI - Explainable AI System for Mortality Prediction in Vasculitis

An explainable artificial intelligence (XAI) system for predicting mortality risk in patients with vasculitis. The system combines machine learning classifiers with multiple XAI methods and a conversational LLM agent, providing transparent and interpretable predictions through interfaces designed for both clinicians and patients.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [Project Structure](#project-structure)
- [API Reference](#api-reference)
- [XAI Methods](#xai-methods)
- [Conversational Agent](#conversational-agent)
- [Dashboard](#dashboard)
- [Testing](#testing)
- [Docker Deployment](#docker-deployment)
- [Clinical Metrics](#clinical-metrics)
- [Safety Guardrails](#safety-guardrails)

## Overview

The system addresses the challenge of making machine learning predictions in healthcare transparent and trustworthy. It integrates three core components:

1. **ML Classification Pipeline** - Trains and evaluates XGBoost, Random Forest, and LightGBM models on clinical vasculitis data with support for handling class imbalance (SMOTE, ADASYN, undersampling).

2. **XAI Explanation Module** - Generates local and global explanations using four complementary methods (LIME, SHAP, DALEX, EBM), enabling cross-method comparison and validation of feature importance rankings.

3. **LLM Conversational Agent** - A RAG-powered (Retrieval-Augmented Generation) agent built on LangChain and ChromaDB that translates technical XAI outputs into natural language explanations, adapting its language to the audience's health literacy level (basic patient, advanced patient, or clinician).

### Key Features

- **Mortality risk prediction** based on 19 clinical features (demographics, organ manifestations, lab values, treatment history, complications)
- **Multi-method XAI explanations** with cross-method agreement analysis
- **Adaptive language** - explanations tailored to basic patients, advanced patients, and clinicians
- **Batch processing** - vectorized predictions for up to 10,000+ patients per request
- **Safety guardrails** - detection of suicidal content, blocking of medical diagnoses and medication recommendations, mandatory medical disclaimers
- **Demo mode** - the system works without a trained model using heuristic predictions, allowing UI and workflow evaluation
- **Interactive dashboard** - Streamlit-based interface for single-patient analysis and batch file processing

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                    INTERFACE (Streamlit Dashboard)                    │
│              Single patient analysis + Batch file upload              │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
┌───────────────────────────────▼──────────────────────────────────────┐
│                      REST API (FastAPI + Uvicorn)                     │
│    /predict  /predict/batch  /explain/*  /chat  /model/*  /health    │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
         ┌──────────────────────┼──────────────────────┐
         │                      │                      │
         ▼                      ▼                      ▼
┌─────────────────┐  ┌──────────────────┐  ┌───────────────────────┐
│  LLM Agent      │  │   XAI Module     │  │   ML Classifiers      │
│  (LangChain     │  │                  │  │                       │
│   + RAG)        │  │  LIME │ SHAP     │  │  Random Forest        │
│                 │  │  DALEX │ EBM     │  │  XGBoost              │
│  ChromaDB       │  │                  │  │  LightGBM             │
│  Guardrails     │  │  Comparison      │  │                       │
└─────────────────┘  └──────────────────┘  └───────────────────────┘
```

## Prerequisites

- **Python 3.10+**
- **pip** (or conda)
- **OpenAI API key** (optional, needed only for the LLM conversational agent; the rest of the system works without it)

## Installation

### 1. Clone and set up the virtual environment

```bash
git clone <repository-url>
cd vasculitis-xai

python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

### 2. Install the package in development mode (optional)

This registers the `vasculitis-api` console command and makes the `src` package importable from anywhere:

```bash
pip install -e .
```

## Configuration

Copy the example environment file and fill in the values:

```bash
cp .env.example .env
```

The `.env` file supports the following variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | *(none)* | OpenAI API key for the conversational agent (optional) |
| `MODEL_PATH` | `models/saved/best_model.joblib` | Path to the trained model file |
| `FEATURE_NAMES_PATH` | `models/saved/feature_names.json` | Path to the feature names JSON file |
| `API_HOST` | `0.0.0.0` | API server bind address |
| `API_PORT` | `8000` | API server port |
| `CHROMA_PERSIST_DIRECTORY` | `./chroma_db` | ChromaDB persistence directory |
| `LOG_LEVEL` | `INFO` | Logging level |
| `ALLOW_DEMO` | `true` | Enable demo mode when no model is loaded |
| `FORCE_API_MODE` | `false` | Disable demo mode entirely (returns 503 if model is missing) |

**Note:** If no trained model file is found at startup, the API automatically falls back to **demo mode**, which uses a heuristic risk scoring function instead of the ML model. This allows you to explore the full UI and API without training a model first.

## Running the Application

### API Server (FastAPI)

```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

Once running:
- API root: http://localhost:8000
- Interactive docs (Swagger UI): http://localhost:8000/docs
- Alternative docs (ReDoc): http://localhost:8000/redoc
- Health check: http://localhost:8000/health

### Dashboard (Streamlit)

The dashboard requires the API to be running (it communicates with `http://localhost:8000`). If the API is unavailable, the dashboard falls back to its own built-in demo mode.

```bash
streamlit run dashboard/streamlit_app.py
```

The dashboard will be available at: http://localhost:8501

### Running Both Together

Open two terminals:

```bash
# Terminal 1 - API
source venv/bin/activate
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2 - Dashboard
source venv/bin/activate
streamlit run dashboard/streamlit_app.py
```

### Using Python Code Directly

```python
from src.data.preprocessing import DataPreprocessor
from src.models.train import ModelTrainer
from src.xai.shap_explainer import SHAPExplainer

# Preprocessing
preprocessor = DataPreprocessor()
df = preprocessor.load_data('data/raw/aktualne_dane.csv', separator='|')
X, y, feature_names = preprocessor.prepare_pipeline(df, target_col='Zgon')

# Train model
trainer = ModelTrainer('xgboost')
trainer.fit(X_train, y_train, feature_names=feature_names)

# Generate SHAP explanation
shap_explainer = SHAPExplainer(trainer.model, X_train, feature_names)
explanation = shap_explainer.explain_instance(X_test[0])
print(shap_explainer.to_json(explanation))
```

## Project Structure

```
vasculitis-xai/
├── src/
│   ├── data/                        # Data loading and preprocessing
│   │   ├── preprocessing.py         # DataPreprocessor class (loading, cleaning, scaling, feature selection)
│   │   ├── imbalance.py             # Imbalanced data handling (SMOTE, ADASYN, undersampling)
│   │   └── feature_engineering.py   # Feature engineering utilities
│   ├── models/                      # ML model training and evaluation
│   │   ├── config.py                # Hyperparameter configurations for all classifiers
│   │   ├── train.py                 # ModelTrainer class (XGBoost, RF, LightGBM)
│   │   └── evaluate.py             # Evaluation with clinically relevant metrics
│   ├── xai/                         # Explainable AI methods
│   │   ├── lime_explainer.py        # LIME (Local Interpretable Model-agnostic Explanations)
│   │   ├── shap_explainer.py        # SHAP (SHapley Additive exPlanations)
│   │   ├── dalex_wrapper.py         # DALEX (Descriptive mAchine Learning EXplanations)
│   │   ├── ebm_explainer.py         # EBM (Explainable Boosting Machine from InterpretML)
│   │   └── comparison.py            # Cross-method comparison and ranking agreement
│   ├── agent/                       # Conversational AI agent
│   │   ├── prompts.py               # System prompts for different health literacy levels
│   │   ├── rag.py                   # RAG pipeline (LangChain + ChromaDB)
│   │   └── guardrails.py           # Safety guardrails (suicide detection, medical advice blocking)
│   └── api/                         # REST API
│       ├── main.py                  # FastAPI application with all endpoints
│       └── schemas.py               # Pydantic models for requests and responses
├── dashboard/
│   └── streamlit_app.py             # Streamlit dashboard (single + batch analysis)
├── tests/                           # Unit tests (pytest)
│   ├── test_preprocessing.py        # Data preprocessing tests
│   ├── test_models.py               # Model training and evaluation tests
│   ├── test_xai.py                  # XAI explainer tests
│   ├── test_agent.py                # Agent and guardrails tests
│   └── test_api.py                  # API endpoint tests
├── data/
│   ├── raw/                         # Raw clinical data
│   └── processed/                   # Preprocessed data
├── models/
│   └── saved/                       # Trained model artifacts (.joblib) and feature names (.json)
├── docs/                            # Documentation
├── scripts/                         # Utility scripts
├── requirements.txt                 # Python dependencies
├── setup.py                         # Package setup (registers vasculitis-api command)
├── .env.example                     # Environment variable template
├── Dockerfile                       # Docker image for the API
├── Dockerfile.streamlit             # Docker image for the dashboard
└── docker-compose.yml               # Multi-service orchestration (API + Dashboard + ChromaDB)
```

## API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API root - returns name, version, and links |
| `GET` | `/health` | Health check - returns model load status and API version |
| `POST` | `/predict` | Single patient mortality risk prediction |
| `POST` | `/predict/batch` | Batch prediction for multiple patients (vectorized, up to 10,000+) |
| `POST` | `/explain/shap` | SHAP explanation for a single patient |
| `POST` | `/explain/lime` | LIME explanation for a single patient |
| `POST` | `/explain/patient` | Patient-friendly explanation (adapts language to health literacy level) |
| `GET` | `/model/info` | Model metadata and performance metrics |
| `GET` | `/model/global-importance` | Global feature importance ranking |
| `POST` | `/chat` | Conversational agent interaction |
| `GET` | `/config/demo-mode` | Get current demo mode status |
| `POST` | `/config/demo-mode` | Enable or disable demo mode |

### Example: Single Prediction

```python
import requests

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
print(f"Risk: {result['probability']:.1%}")
print(f"Level: {result['risk_level']}")  # "low", "moderate", or "high"
```

### Example: Batch Prediction

```python
response = requests.post(
    "http://localhost:8000/predict/batch",
    json={
        "patients": [
            {"wiek": 55, "plec": 1, "liczba_zajetych_narzadow": 3, "manifestacja_nerki": 1},
            {"wiek": 45, "plec": 0, "liczba_zajetych_narzadow": 2, "manifestacja_nerki": 0}
        ],
        "include_risk_factors": True,
        "top_n_factors": 3
    }
)

result = response.json()
print(f"Processed: {result['processed_count']} patients in {result['processing_time_ms']:.1f}ms")
print(f"High risk: {result['summary']['high_risk_count']}")
```

### Patient Input Features

| Feature | Type | Description |
|---------|------|-------------|
| `wiek` | float | Patient age (0-120) |
| `plec` | int | Sex (0=Female, 1=Male) |
| `wiek_rozpoznania` | float | Age at diagnosis (optional) |
| `opoznienie_rozpoznia` | float | Diagnostic delay in months (optional) |
| `liczba_zajetych_narzadow` | int | Number of affected organs (0-20) |
| `manifestacja_nerki` | int | Kidney involvement (0/1) |
| `manifestacja_sercowo_naczyniowy` | int | Cardiovascular involvement (0/1) |
| `manifestacja_zajecie_csn` | int | Central nervous system involvement (0/1) |
| `manifestacja_neurologiczny` | int | Peripheral nervous system involvement (0/1) |
| `manifestacja_pokarmowy` | int | Gastrointestinal involvement (0/1) |
| `zaostrz_wymagajace_oit` | int | ICU-requiring flares (0/1) |
| `kreatynina` | float | Creatinine level in micromol/L (optional) |
| `max_crp` | float | Maximum CRP in mg/L (optional) |
| `plazmaferezy` | int | Plasmapheresis treatment (0/1) |
| `dializa` | int | Dialysis treatment (0/1) |
| `sterydy_dawka_g` | float | Steroid dose in grams (optional) |
| `czas_sterydow` | float | Steroid treatment duration in months (optional) |
| `powiklania_serce_pluca` | int | Cardiopulmonary complications (0/1) |
| `powiklania_infekcja` | int | Infection complications (0/1) |

## XAI Methods

The system implements four complementary XAI methods. Each provides a different perspective on feature importance, and the comparison module allows validation through ranking agreement analysis.

### LIME (Local Interpretable Model-agnostic Explanations)

Generates local surrogate models to explain individual predictions by perturbing the input and fitting a simple linear model around the instance of interest.

```python
from src.xai.lime_explainer import LIMEExplainer

explainer = LIMEExplainer(model, X_train, feature_names)
explanation = explainer.explain_instance(instance, num_features=10)
explainer.plot_explanation(explanation)
```

### SHAP (SHapley Additive exPlanations)

Computes Shapley values to assign each feature a contribution to the prediction, grounded in cooperative game theory. Uses TreeExplainer for tree-based models.

```python
from src.xai.shap_explainer import SHAPExplainer

explainer = SHAPExplainer(model, X_train, feature_names)
explanation = explainer.explain_instance(instance)
explainer.plot_waterfall(explanation)
```

### DALEX (Descriptive mAchine Learning EXplanations)

Provides model-level and instance-level explanations including break-down plots, variable importance, and partial dependence profiles.

```python
from src.xai.dalex_wrapper import DALEXWrapper

wrapper = DALEXWrapper(model, X, y, feature_names)
bd_exp = wrapper.explain_instance_break_down(instance)
wrapper.plot_break_down(bd_exp)
```

### EBM (Explainable Boosting Machine)

An inherently interpretable glass-box model from InterpretML. Provides both global and local explanations as a byproduct of its architecture.

```python
from src.xai.ebm_explainer import EBMExplainer

ebm = EBMExplainer(feature_names=feature_names)
ebm.fit(X_train, y_train)
local_exp = ebm.explain_local(instance)
ebm.plot_local_explanation(local_exp)
```

### Cross-Method Comparison

```python
from src.xai.comparison import XAIComparison

comparison = XAIComparison(explainers=[shap_exp, lime_exp, dalex_exp])
results = comparison.compare_feature_rankings(instance)
agreement = comparison.calculate_agreement()
```

## Conversational Agent

The conversational agent uses a RAG pipeline built on LangChain and ChromaDB to provide context-aware explanations. It adapts its language to three health literacy levels:

- **Basic** - Simple language, avoids technical terms, shorter sentences
- **Advanced** - More detailed explanations, includes some technical context
- **Clinician** - Full technical detail, SHAP values, model metrics

The agent includes a medical knowledge base about vasculitis (types, lab markers, treatments) that is embedded into ChromaDB and retrieved contextually during conversations.

## Dashboard

The Streamlit dashboard provides two analysis modes:

### Single Patient Mode
- Sidebar form for entering all 19 clinical features
- Gauge chart showing risk probability
- Waterfall chart (SHAP) and bar chart (LIME) of feature contributions
- Cross-method comparison tab
- Interactive chat with the AI agent

### Batch Analysis Mode
- Upload CSV or JSON files with multiple patients
- Automatic column name normalization (supports Polish and English column names)
- Risk distribution pie chart, probability histogram, age-vs-risk scatter plot
- Filterable and sortable results table with color-coded risk levels
- Export results to CSV or JSON

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run tests with coverage report
pytest tests/ --cov=src --cov-report=html

# Run a specific test module
pytest tests/test_preprocessing.py -v
pytest tests/test_models.py -v
pytest tests/test_xai.py -v
pytest tests/test_agent.py -v
pytest tests/test_api.py -v
```

## Docker Deployment

### Building Individual Images

```bash
# API
docker build -t vasculitis-api -f Dockerfile .

# Dashboard
docker build -t vasculitis-dashboard -f Dockerfile.streamlit .
```

### Docker Compose (all services)

The `docker-compose.yml` orchestrates three services:

```bash
# Start all services in the background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop all services
docker-compose down
```

| Service | Port | Description |
|---------|------|-------------|
| API | http://localhost:8000 | FastAPI backend |
| Dashboard | http://localhost:8501 | Streamlit frontend |
| ChromaDB | http://localhost:8001 | Vector database for RAG |

Both the API and Dashboard containers include health checks and run as non-root users. The API container mounts `models/` and `data/` as read-only volumes.

## Clinical Metrics

The model evaluation module uses clinically relevant metrics:

| Metric | Description | Relevance |
|--------|-------------|-----------|
| **AUC-ROC** | Area under the ROC curve | Overall model discrimination |
| **Sensitivity (Recall)** | True positive rate | Critical - ability to detect at-risk patients |
| **Specificity** | True negative rate | Avoiding unnecessary interventions |
| **PPV (Precision)** | Positive predictive value | Confidence in positive predictions |
| **NPV** | Negative predictive value | Confidence in negative predictions |
| **Brier Score** | Probability calibration | Reliability of predicted probabilities |

Risk thresholds: **< 30%** low risk, **30-70%** moderate risk, **> 70%** high risk.

## Safety Guardrails

The system includes multiple safety layers to ensure responsible use in a medical context:

- **Suicidal content detection** - Identifies expressions of self-harm intent and redirects to crisis hotlines (Polish emergency numbers)
- **Diagnosis request blocking** - Refuses to provide medical diagnoses and redirects to healthcare professionals
- **Medication advice blocking** - Refuses to prescribe, change dosage, or recommend stopping medications
- **Prognosis filtering** - Removes specific numerical life expectancy predictions from responses
- **Response sanitization** - Scans agent output for forbidden patterns before delivery
- **Mandatory disclaimers** - All patient-facing responses include a medical disclaimer
- **Emotional distress detection** - Adds empathetic prefixes when user anxiety is detected
- **Health literacy validation** - Checks that responses match the target audience's comprehension level

## License

MIT License

## Authors

Master's thesis project - Explainable AI System for Mortality Prediction in Vasculitis
