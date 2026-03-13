"""
Dashboard Streamlit dla systemu XAI.

Interfejs użytkownika do predykcji ryzyka śmiertelności
w zapaleniu naczyń z wyjaśnieniami XAI.
"""

import os
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import requests
import json
import io
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

# ============================================================================
# KONFIGURACJA
# ============================================================================

st.set_page_config(
    page_title="Vasculitis XAI - System wspomagania decyzji",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API URL — można nadpisać zmienną środowiskową API_URL
API_URL = os.getenv("API_URL", "http://localhost:8000")

# ============================================================================
# STYLE CSS
# ============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1a5276;
        text-align: center;
        padding: 1rem;
    }
    .risk-low {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        color: #155724;
    }
    .risk-moderate {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        color: #856404;
    }
    .risk-high {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #dc3545;
        color: #721c24;
    }
    .disclaimer {
        background-color: #e9ecef;
        padding: 1rem;
        border-radius: 5px;
        font-size: 0.9rem;
        color: #495057;
    }
    .factor-positive {
        color: #155724;
    }
    .factor-negative {
        color: #721c24;
    }
    .info-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #2874a6;
        color: #2c3e50;
    }
    .info-card h3 {
        color: #1a5276;
        margin-bottom: 0.5rem;
    }
    .info-card ul {
        color: #34495e;
    }
    .info-card ul li {
        color: #2c3e50;
        margin: 0.3rem 0;
    }
    /* Streamlit alerts - lepszy kontrast */
    .stAlert > div {
        color: #1a5276 !important;
    }
    /* Sidebar expanders - lepszy kontrast */
    .streamlit-expanderHeader {
        color: #1a5276 !important;
        font-weight: 600;
    }
    /* Plotly charts - przezroczyste tło dla integracji z dark mode */
    .js-plotly-plot .plotly .main-svg {
        background: transparent !important;
    }
    /* Batch analysis styles */
    .batch-header {
        background: linear-gradient(135deg, #2c3e50 0%, #1a252f 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        color: white;
    }
    .batch-stats {
        display: flex;
        gap: 1rem;
        flex-wrap: wrap;
    }
    .batch-stat-card {
        background: rgba(255,255,255,0.1);
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        flex: 1;
        min-width: 120px;
    }
    .batch-stat-card h4 {
        margin: 0;
        font-size: 2rem;
        font-weight: bold;
    }
    .batch-stat-card p {
        margin: 0.5rem 0 0 0;
        font-size: 0.9rem;
        opacity: 0.8;
    }
    .risk-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    .risk-badge-low {
        background-color: #28a745;
        color: white;
    }
    .risk-badge-moderate {
        background-color: #ffc107;
        color: #212529;
    }
    .risk-badge-high {
        background-color: #dc3545;
        color: white;
    }
    .upload-zone {
        border: 2px dashed #4a5568;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background: rgba(255,255,255,0.02);
        transition: all 0.3s ease;
    }
    .upload-zone:hover {
        border-color: #2874a6;
        background: rgba(40,116,166,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# FUNKCJE POMOCNICZE
# ============================================================================

def call_api(endpoint: str, method: str = "GET", data: Dict = None) -> Optional[Dict]:
    """Wywołaj API."""
    try:
        url = f"{API_URL}{endpoint}"
        if method == "GET":
            response = requests.get(url, timeout=10)
        else:
            response = requests.post(url, json=data, timeout=30)

        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Błąd API: {response.status_code}")
            return None
    except requests.exceptions.ConnectionError:
        st.warning("API niedostępne. Używam trybu demo.")
        return None
    except Exception as e:
        st.error(f"Błąd: {e}")
        return None


def call_api_cached(endpoint: str, method: str = "GET", data: Dict = None) -> Optional[Dict]:
    """Wywołaj API z cache w session_state. Cache czyszczony przy zmianie danych pacjenta."""
    if "_xai_cache" not in st.session_state:
        st.session_state["_xai_cache"] = {}
    cache_key = f"{method}:{endpoint}:{json.dumps(data, sort_keys=True, default=str) if data else ''}"
    if cache_key in st.session_state["_xai_cache"]:
        return st.session_state["_xai_cache"][cache_key]
    result = call_api(endpoint, method, data)
    if result is not None:
        st.session_state["_xai_cache"][cache_key] = result
    return result


def get_demo_prediction(patient_data: Dict) -> Dict:
    """Demo predykcja gdy API niedostępne."""
    risk_score = 0.0

    # Prosta heurystyka
    risk_score += max(0, (patient_data.get('wiek', 50) - 50) / 100)
    risk_score += patient_data.get('liczba_zajetych_narzadow', 0) * 0.1
    if patient_data.get('manifestacja_nerki'):
        risk_score += 0.15
    if patient_data.get('zaostrz_wymagajace_oit'):
        risk_score += 0.25
    if patient_data.get('dializa'):
        risk_score += 0.2

    probability = min(max(risk_score, 0.05), 0.95)

    if probability < 0.3:
        risk_level = "low"
    elif probability < 0.7:
        risk_level = "moderate"
    else:
        risk_level = "high"

    return {
        "probability": probability,
        "risk_level": risk_level,
        "prediction": 1 if probability > 0.5 else 0
    }


def get_demo_explanation(patient_data: Dict) -> Dict:
    """Demo wyjaśnienie."""
    risk_factors = []
    protective_factors = []

    if patient_data.get('wiek', 50) > 60:
        risk_factors.append({"feature": "Wiek", "contribution": 0.15})
    else:
        protective_factors.append({"feature": "Wiek", "contribution": -0.1})

    if patient_data.get('manifestacja_nerki'):
        risk_factors.append({"feature": "Zajęcie nerek", "contribution": 0.12})

    if patient_data.get('zaostrz_wymagajace_oit'):
        risk_factors.append({"feature": "Zaostrzenia OIT", "contribution": 0.2})

    if patient_data.get('liczba_zajetych_narzadow', 0) <= 2:
        protective_factors.append({"feature": "Liczba narządów", "contribution": -0.08})

    return {
        "risk_factors": risk_factors,
        "protective_factors": protective_factors,
        "base_value": 0.15
    }


def get_demo_dialysis_explanation(patient_data: Dict) -> Dict:
    """Demo wyjaśnienie dializy — heurystyki oparte na kreatyninie i funkcji nerek."""
    risk_factors = []
    protective_factors = []

    kreatynina = patient_data.get('kreatynina', 100)
    if kreatynina > 300:
        risk_factors.append({"feature": "Kreatynina", "contribution": 0.40})
    elif kreatynina > 200:
        risk_factors.append({"feature": "Kreatynina", "contribution": 0.25})
    elif kreatynina > 150:
        risk_factors.append({"feature": "Kreatynina", "contribution": 0.12})
    else:
        protective_factors.append({"feature": "Kreatynina", "contribution": -0.10})

    if patient_data.get('manifestacja_nerki'):
        risk_factors.append({"feature": "Zajęcie nerek", "contribution": 0.22})
    else:
        protective_factors.append({"feature": "Brak zajęcia nerek", "contribution": -0.12})

    if patient_data.get('zaostrz_wymagajace_oit'):
        risk_factors.append({"feature": "Zaostrzenia OIT", "contribution": 0.15})

    narzady = patient_data.get('liczba_zajetych_narzadow', 0)
    if narzady > 3:
        risk_factors.append({"feature": "Liczba narządów", "contribution": 0.08})
    elif narzady <= 1:
        protective_factors.append({"feature": "Liczba narządów", "contribution": -0.06})

    return {
        "risk_factors": risk_factors,
        "protective_factors": protective_factors,
        "base_value": 0.20
    }


def get_demo_dialysis_prediction(patient_data: Dict) -> Dict:
    """Demo predykcja dializy gdy API niedostępne."""
    risk_score = 0.0

    # Kreatynina — najsilniejszy predyktor
    kreatynina = patient_data.get('kreatynina', 100)
    if kreatynina > 300:
        risk_score += 0.45
    elif kreatynina > 200:
        risk_score += 0.30
    elif kreatynina > 150:
        risk_score += 0.15

    # Manifestacja nerek
    if patient_data.get('manifestacja_nerki'):
        risk_score += 0.25

    risk_score += patient_data.get('liczba_zajetych_narzadow', 0) * 0.03
    if patient_data.get('zaostrz_wymagajace_oit'):
        risk_score += 0.10
    if patient_data.get('plazmaferezy'):
        risk_score += 0.05

    probability = min(max(risk_score, 0.02), 0.98)

    if probability < 0.3:
        risk_level = "low"
    elif probability < 0.7:
        risk_level = "moderate"
    else:
        risk_level = "high"

    return {
        "probability": probability,
        "needs_dialysis": probability > 0.5,
        "prediction": int(probability > 0.5),
        "risk_level": risk_level,
    }


def create_gauge_chart(probability: float, title: str = "Ryzyko") -> go.Figure:
    """Utwórz wykres gauge."""
    if probability < 0.3:
        color = "#28a745"  # green
    elif probability < 0.7:
        color = "#ffc107"  # orange/yellow
    else:
        color = "#dc3545"  # red

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 20, 'color': '#ffffff'}},
        number={'suffix': '%', 'font': {'size': 40, 'color': '#ffffff'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': '#ffffff', 'tickfont': {'color': '#ffffff'}},
            'bar': {'color': color},
            'bgcolor': "#2d2d2d",
            'borderwidth': 2,
            'bordercolor': "#555555",
            'steps': [
                {'range': [0, 30], 'color': '#1e4620'},
                {'range': [30, 70], 'color': '#5c4a1e'},
                {'range': [70, 100], 'color': '#5c1e1e'}
            ],
            'threshold': {
                'line': {'color': "#ffffff", 'width': 4},
                'thickness': 0.75,
                'value': probability * 100
            }
        }
    ))

    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        font=dict(color='#ffffff', size=14),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        template='plotly_dark'
    )

    # Legenda stref ryzyka
    fig.add_annotation(
        x=0.15, y=0.22, text="Niskie<br><b>0–30%</b>",
        showarrow=False, font=dict(size=10, color='#28a745'),
        xref='paper', yref='paper'
    )
    fig.add_annotation(
        x=0.50, y=0.22, text="Umiarkowane<br><b>30–70%</b>",
        showarrow=False, font=dict(size=10, color='#ffc107'),
        xref='paper', yref='paper'
    )
    fig.add_annotation(
        x=0.85, y=0.22, text="Wysokie<br><b>70–100%</b>",
        showarrow=False, font=dict(size=10, color='#dc3545'),
        xref='paper', yref='paper'
    )

    return fig


def create_waterfall_chart(factors: list, title: str = "Wpływ czynników") -> go.Figure:
    """Utwórz wykres waterfall."""
    if not factors:
        return None

    names = [f["feature"] for f in factors]
    values = [f["contribution"] for f in factors]
    colors = ["red" if v > 0 else "green" for v in values]

    fig = go.Figure(go.Waterfall(
        name="Kontribucje",
        orientation="h",
        y=names,
        x=values,
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        decreasing={"marker": {"color": "#28a745"}},
        increasing={"marker": {"color": "#dc3545"}},
        text=[f"{v:+.3f}" for v in values],
        textposition="outside",
        textfont=dict(size=13, color='#ffffff', family='Arial Black'),
        showlegend=False
    ))

    # Legenda kolorów
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='markers',
        marker=dict(size=10, color='#dc3545', symbol='square'),
        name='Zwiększa ryzyko (wartość dodatnia)',
        showlegend=True
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='markers',
        marker=dict(size=10, color='#28a745', symbol='square'),
        name='Zmniejsza ryzyko (wartość ujemna)',
        showlegend=True
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color='#ffffff')),
        xaxis=dict(
            title=dict(text="Wpływ na predykcję (wartość SHAP / kontribucja)", font=dict(size=13, color='#ffffff')),
            tickfont=dict(color='#ffffff'),
            gridcolor='#444444',
            zerolinecolor='#888888'
        ),
        yaxis=dict(
            title=dict(text="Cecha kliniczna", font=dict(size=13, color='#ffffff')),
            tickfont=dict(color='#ffffff')
        ),
        height=400,
        margin=dict(l=20, r=20, t=50, b=40),
        font=dict(color='#ffffff', size=13),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        template='plotly_dark',
        showlegend=True,
        legend=dict(
            orientation='h', yanchor='top', y=-0.15, xanchor='center', x=0.5,
            font=dict(color='#ffffff', size=11)
        )
    )

    return fig


# ============================================================================
# FUNKCJE ANALIZY MASOWEJ
# ============================================================================

# Mapowanie kolumn CSV/JSON na wewnętrzne nazwy
COLUMN_MAPPING = {
    # Warianty polskie
    'wiek': 'wiek',
    'age': 'wiek',
    'plec': 'plec',
    'płeć': 'plec',
    'sex': 'plec',
    'gender': 'plec',
    'wiek_rozpoznania': 'wiek_rozpoznania',
    'age_at_diagnosis': 'wiek_rozpoznania',
    'liczba_narzadow': 'liczba_zajetych_narzadow',
    'liczba_zajetych_narzadow': 'liczba_zajetych_narzadow',
    'organ_count': 'liczba_zajetych_narzadow',
    'nerki': 'manifestacja_nerki',
    'kidneys': 'manifestacja_nerki',
    'manifestacja_nerki': 'manifestacja_nerki',
    'serce': 'manifestacja_sercowo_naczyniowy',
    'heart': 'manifestacja_sercowo_naczyniowy',
    'manifestacja_sercowo_naczyniowy': 'manifestacja_sercowo_naczyniowy',
    'csn': 'manifestacja_zajecie_csn',
    'cns': 'manifestacja_zajecie_csn',
    'manifestacja_zajecie_csn': 'manifestacja_zajecie_csn',
    'neuro': 'manifestacja_neurologiczny',
    'neurological': 'manifestacja_neurologiczny',
    'manifestacja_neurologiczny': 'manifestacja_neurologiczny',
    'pokarmowy': 'manifestacja_pokarmowy',
    'gi': 'manifestacja_pokarmowy',
    'gastrointestinal': 'manifestacja_pokarmowy',
    'manifestacja_pokarmowy': 'manifestacja_pokarmowy',
    'oit': 'zaostrz_wymagajace_oit',
    'icu': 'zaostrz_wymagajace_oit',
    'zaostrz_wymagajace_oit': 'zaostrz_wymagajace_oit',
    'kreatynina': 'kreatynina',
    'creatinine': 'kreatynina',
    'crp': 'max_crp',
    'max_crp': 'max_crp',
    'plazmaferezy': 'plazmaferezy',
    'plasmapheresis': 'plazmaferezy',
    'dializa': 'dializa',
    'dialysis': 'dializa',
    'sterydy': 'sterydy_dawka_g',
    'sterydy_dawka_g': 'sterydy_dawka_g',
    'steroids': 'sterydy_dawka_g',
    'czas_sterydow': 'czas_sterydow',
    'steroid_duration': 'czas_sterydow',
    'powiklania_serce': 'powiklania_serce_pluca',
    'powiklania_serce_pluca': 'powiklania_serce_pluca',
    'cardiac_complications': 'powiklania_serce_pluca',
    'powiklania_infekcja': 'powiklania_infekcja',
    'infections': 'powiklania_infekcja',
    'id': 'patient_id',
    'patient_id': 'patient_id',
    'id_pacjenta': 'patient_id',
    # Dialysis-specific aliases
    'manifestacja_oddechowy': 'manifestacja_oddechowy',
    'respiratory': 'manifestacja_oddechowy',
    'oddechowy': 'manifestacja_oddechowy',
    'zaostrz_wymagajace_hospital': 'zaostrz_wymagajace_hospital',
    'hospital': 'zaostrz_wymagajace_hospital',
    'hospitalization': 'zaostrz_wymagajace_hospital',
    'pulsy': 'pulsy',
    'pulses': 'pulsy',
    'pulse_steroids': 'pulsy',
    'wiek_w_chwili_zachorowania': 'wiek_rozpoznania',
    'opoznienie_rozpoznia': 'opoznienie_rozpoznia',
    'diagnosis_delay': 'opoznienie_rozpoznia',
}

# Wartości domyślne dla brakujących kolumn
DEFAULT_VALUES = {
    'wiek': 50,
    'plec': 0,
    'wiek_rozpoznania': 45,
    'liczba_zajetych_narzadow': 2,
    'manifestacja_nerki': 0,
    'manifestacja_sercowo_naczyniowy': 0,
    'manifestacja_zajecie_csn': 0,
    'manifestacja_neurologiczny': 0,
    'manifestacja_pokarmowy': 0,
    'zaostrz_wymagajace_oit': 0,
    'kreatynina': 100.0,
    'max_crp': 30.0,
    'plazmaferezy': 0,
    'dializa': 0,
    'sterydy_dawka_g': 0.5,
    'czas_sterydow': 12,
    'powiklania_serce_pluca': 0,
    'powiklania_infekcja': 0,
    'manifestacja_oddechowy': 0,
    'zaostrz_wymagajace_hospital': 0,
    'pulsy': 0,
}


def parse_uploaded_file(uploaded_file) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Parsuj wgrany plik CSV lub JSON.
    Zwraca: (DataFrame, komunikat błędu lub None)
    """
    try:
        file_name = uploaded_file.name.lower()

        if file_name.endswith('.csv'):
            # Próbuj różne separatory
            content = uploaded_file.getvalue().decode('utf-8')
            for sep in [',', ';', '\t', '|']:
                try:
                    df = pd.read_csv(io.StringIO(content), sep=sep)
                    if len(df.columns) > 1:
                        break
                except Exception:
                    continue
            else:
                df = pd.read_csv(io.StringIO(content))

        elif file_name.endswith('.json'):
            content = uploaded_file.getvalue().decode('utf-8')
            data = json.loads(content)

            # Obsłuż różne formaty JSON
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                if 'patients' in data:
                    df = pd.DataFrame(data['patients'])
                elif 'data' in data:
                    df = pd.DataFrame(data['data'])
                else:
                    # Pojedynczy pacjent
                    df = pd.DataFrame([data])
            else:
                return None, "Nieobsługiwany format JSON"
        else:
            return None, "Nieobsługiwany format pliku. Użyj CSV lub JSON."

        if df.empty:
            return None, "Plik jest pusty"

        return df, None

    except Exception as e:
        return None, f"Błąd podczas parsowania pliku: {str(e)}"


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizuj nazwy kolumn i uzupełnij brakujące wartości.
    """
    # Normalizuj nazwy kolumn (lowercase, usuń spacje)
    df.columns = [col.lower().strip().replace(' ', '_') for col in df.columns]

    # Mapuj kolumny
    rename_map = {}
    for col in df.columns:
        if col in COLUMN_MAPPING:
            rename_map[col] = COLUMN_MAPPING[col]

    df = df.rename(columns=rename_map)

    # Dodaj brakujące kolumny z wartościami domyślnymi
    for col, default_val in DEFAULT_VALUES.items():
        if col not in df.columns:
            df[col] = default_val

    # Dodaj ID pacjenta jeśli brak
    if 'patient_id' not in df.columns:
        df['patient_id'] = [f"P{i+1:04d}" for i in range(len(df))]

    # Konwertuj wartości tekstowe na numeryczne
    for col in df.columns:
        if df[col].dtype == 'object':
            # Konwersja płci
            if col == 'plec':
                df[col] = df[col].apply(lambda x: 1 if str(x).lower() in ['m', 'mężczyzna', 'male', '1'] else 0)
            # Konwersja boolean
            elif col in DEFAULT_VALUES and DEFAULT_VALUES[col] in [0, 1]:
                df[col] = df[col].apply(lambda x: 1 if str(x).lower() in ['tak', 'yes', 'true', '1', 't', 'y'] else 0)

    # Oblicz opóźnienie rozpoznania (nie może być ujemne)
    if 'opoznienie_rozpoznia' not in df.columns:
        df['opoznienie_rozpoznia'] = (df['wiek'] - df['wiek_rozpoznania']).clip(lower=0)

    return df


def prepare_patients_for_batch(df: pd.DataFrame, prediction_task: str = "Śmiertelność") -> List[Dict]:
    """Przygotuj listę pacjentów do wysłania do batch API."""
    patients = []
    for _, row in df.iterrows():
        if prediction_task == "Potrzeba dializy":
            patient = {
                "wiek": float(row.get('wiek', 50)),
                "plec": int(row.get('plec', 0)),
                "wiek_rozpoznania": float(row.get('wiek_rozpoznania', 45)),
                "liczba_zajetych_narzadow": int(row.get('liczba_zajetych_narzadow', 2)),
                "manifestacja_sercowo_naczyniowy": int(row.get('manifestacja_sercowo_naczyniowy', 0)),
                "manifestacja_nerki": int(row.get('manifestacja_nerki', 0)),
                "manifestacja_neurologiczny": int(row.get('manifestacja_neurologiczny', 0)),
                "manifestacja_oddechowy": int(row.get('manifestacja_oddechowy', 0)),
                "zaostrz_wymagajace_oit": int(row.get('zaostrz_wymagajace_oit', 0)),
                "zaostrz_wymagajace_hospital": int(row.get('zaostrz_wymagajace_hospital', 0)),
                "kreatynina": float(row.get('kreatynina', 100.0)),
                "max_crp": float(row.get('max_crp', 30.0)),
                "plazmaferezy": int(row.get('plazmaferezy', 0)),
                "pulsy": int(row.get('pulsy', 0)),
                "sterydy_dawka_g": float(row.get('sterydy_dawka_g', 0.5)),
                "czas_sterydow": float(row.get('czas_sterydow', 12)),
                "powiklania_serce_pluca": int(row.get('powiklania_serce_pluca', 0)),
                "powiklania_infekcja": int(row.get('powiklania_infekcja', 0))
            }
        else:
            patient = {
                "wiek": int(row.get('wiek', 50)),
                "plec": int(row.get('plec', 0)),
                "wiek_rozpoznania": int(row.get('wiek_rozpoznania', 45)),
                "opoznienie_rozpoznia": int(row.get('opoznienie_rozpoznia', 5)),
                "liczba_zajetych_narzadow": int(row.get('liczba_zajetych_narzadow', 2)),
                "manifestacja_sercowo_naczyniowy": int(row.get('manifestacja_sercowo_naczyniowy', 0)),
                "manifestacja_nerki": int(row.get('manifestacja_nerki', 0)),
                "manifestacja_pokarmowy": int(row.get('manifestacja_pokarmowy', 0)),
                "manifestacja_zajecie_csn": int(row.get('manifestacja_zajecie_csn', 0)),
                "manifestacja_neurologiczny": int(row.get('manifestacja_neurologiczny', 0)),
                "zaostrz_wymagajace_oit": int(row.get('zaostrz_wymagajace_oit', 0)),
                "kreatynina": float(row.get('kreatynina', 100.0)),
                "max_crp": float(row.get('max_crp', 30.0)),
                "plazmaferezy": int(row.get('plazmaferezy', 0)),
                "dializa": int(row.get('dializa', 0)),
                "sterydy_dawka_g": float(row.get('sterydy_dawka_g', 0.5)),
                "czas_sterydow": int(row.get('czas_sterydow', 12)),
                "powiklania_serce_pluca": int(row.get('powiklania_serce_pluca', 0)),
                "powiklania_infekcja": int(row.get('powiklania_infekcja', 0))
            }
        patients.append(patient)
    return patients


def call_batch_api(patients: List[Dict], include_risk_factors: bool = True, model_type: Optional[str] = None, endpoint: str = "/predict/batch") -> Optional[Dict]:
    """Wywołaj batch API endpoint."""
    try:
        payload = {
            "patients": patients,
            "include_risk_factors": include_risk_factors,
            "top_n_factors": 3
        }
        if model_type:
            payload["model_type"] = model_type

        response = requests.post(
            f"{API_URL}{endpoint}",
            json=payload,
            timeout=300  # 5 minut timeout dla dużych batch
        )
        if response.status_code == 200:
            data = response.json()
            # Store error info in session state for display
            st.session_state['batch_api_error_count'] = data.get('error_count', 0)
            st.session_state['batch_api_success_count'] = data.get('success_count', len(patients))
            st.session_state['batch_api_errors'] = data.get('errors', [])
            return data
        else:
            st.session_state['_batch_error'] = response.text
            return None
    except Exception as e:
        st.session_state['_batch_error'] = str(e)
        return None


def process_batch_patients(
    df: pd.DataFrame,
    prediction_task: str = "Śmiertelność",
    selected_model_type: Optional[str] = None,
    progress_callback=None
) -> pd.DataFrame:
    """
    Przetwórz pacjentów wsadowo używając batch API.

    Dla plików > 1000 pacjentów dzieli na chunki.
    """
    results = []
    total = len(df)
    patient_ids = df.get('patient_id', pd.Series([f"P{i+1:04d}" for i in range(total)]))

    # Konfiguracja chunków
    CHUNK_SIZE = 1000  # Pacjentów na request
    chunks = [df.iloc[i:i+CHUNK_SIZE] for i in range(0, total, CHUNK_SIZE)]

    processed = 0

    for chunk_idx, chunk_df in enumerate(chunks):
        chunk_start = chunk_idx * CHUNK_SIZE

        # Przygotuj pacjentów dla tego chunka
        patients = prepare_patients_for_batch(chunk_df, prediction_task=prediction_task)

        # Spróbuj użyć batch API
        if prediction_task == "Śmiertelność":
            batch_result = call_batch_api(patients, include_risk_factors=True, model_type=selected_model_type)
        else:
            batch_result = call_batch_api(patients, include_risk_factors=True, model_type=selected_model_type, endpoint="/predict/dialysis/batch")

        if batch_result is not None:
            # Użyj wyników z API
            for i, item in enumerate(batch_result.get('results', [])):
                global_idx = chunk_start + i
                pred = item.get('prediction', {})
                top_factors_list = item.get('top_risk_factors', [])
                top_factors = ", ".join([f.get('feature', '') for f in top_factors_list[:3]]) if top_factors_list else ""

                result = {
                    'patient_id': patient_ids.iloc[global_idx] if global_idx < len(patient_ids) else f"P{global_idx+1:04d}",
                    'wiek': patients[i]['wiek'],
                    'plec': 'M' if patients[i]['plec'] == 1 else 'K',
                    'liczba_narzadow': patients[i]['liczba_zajetych_narzadow'],
                    'probability': pred.get('probability', 0),
                    'probability_pct': f"{pred.get('probability', 0)*100:.1f}%",
                    'risk_level': pred.get('risk_level', 'low'),
                    'risk_level_pl': {'low': 'Niskie', 'moderate': 'Umiarkowane', 'high': 'Wysokie'}.get(pred.get('risk_level', 'low'), 'Niskie'),
                    'prediction': pred.get('prediction', 0),
                    'top_factors': top_factors,
                    'processing_mode': batch_result.get('mode', 'api')
                }
                results.append(result)

                processed += 1
                if progress_callback:
                    progress_callback(processed / total)
        else:
            # Fallback do pojedynczych wywołań (demo mode)
            for i, (_, row) in enumerate(chunk_df.iterrows()):
                global_idx = chunk_start + i
                patient_data = patients[i]

                # Prediction (zależne od celu predykcji)
                if prediction_task == "Potrzeba dializy":
                    api_pred = call_api("/predict/dialysis", "POST", patient_data)
                    prediction = api_pred if api_pred else get_demo_dialysis_prediction(patient_data)
                    explanation = get_demo_dialysis_explanation(patient_data)
                else:
                    api_pred = call_api("/predict", "POST", patient_data)
                    prediction = api_pred if api_pred else get_demo_prediction(patient_data)
                    explanation = get_demo_explanation(patient_data)

                all_factors = explanation["risk_factors"] + explanation["protective_factors"]
                all_factors_sorted = sorted(all_factors, key=lambda x: abs(x["contribution"]), reverse=True)
                top_factors = ", ".join([f["feature"] for f in all_factors_sorted[:3]])

                result = {
                    'patient_id': patient_ids.iloc[global_idx] if global_idx < len(patient_ids) else f"P{global_idx+1:04d}",
                    'wiek': patient_data['wiek'],
                    'plec': 'M' if patient_data['plec'] == 1 else 'K',
                    'liczba_narzadow': patient_data['liczba_zajetych_narzadow'],
                    'probability': prediction['probability'],
                    'probability_pct': f"{prediction['probability']*100:.1f}%",
                    'risk_level': prediction['risk_level'],
                    'risk_level_pl': {'low': 'Niskie', 'moderate': 'Umiarkowane', 'high': 'Wysokie'}[prediction['risk_level']],
                    'prediction': prediction['prediction'],
                    'top_factors': top_factors,
                    'processing_mode': 'demo'
                }
                results.append(result)

                processed += 1
                if progress_callback:
                    progress_callback(processed / total)

    return pd.DataFrame(results)


def get_api_status() -> Dict:
    """Pobierz status API (tryb demo, model załadowany)."""
    try:
        response = requests.get(f"{API_URL}/config/demo-mode", timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return {
        "demo_allowed": True,
        "model_loaded": False,
        "current_mode": "unavailable",
        "force_api_mode": False
    }




def create_risk_distribution_chart(results_df: pd.DataFrame) -> go.Figure:
    """Utwórz wykres rozkładu ryzyka."""
    risk_counts = results_df['risk_level'].value_counts()

    colors = {
        'low': '#28a745',
        'moderate': '#ffc107',
        'high': '#dc3545'
    }
    labels = {
        'low': 'Niskie',
        'moderate': 'Umiarkowane',
        'high': 'Wysokie'
    }

    fig = go.Figure(data=[
        go.Pie(
            labels=[labels.get(k, k) for k in risk_counts.index],
            values=risk_counts.values,
            marker_colors=[colors.get(k, '#666') for k in risk_counts.index],
            hole=0.4,
            textinfo='label+percent+value',
            textfont=dict(size=14, color='white'),
            hovertemplate='<b>%{label}</b><br>Liczba: %{value}<br>Udział: %{percent}<extra></extra>'
        )
    ])

    fig.update_layout(
        title=dict(text="Rozkład poziomów ryzyka", font=dict(size=18, color='#ffffff')),
        font=dict(color='#ffffff'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        template='plotly_dark',
        height=350,
        showlegend=True,
        legend=dict(
            title=dict(text="Poziom ryzyka", font=dict(color='#ffffff', size=12)),
            font=dict(color='#ffffff')
        ),
        annotations=[dict(
            text="Niskie: <30%<br>Umiarkowane: 30–70%<br>Wysokie: >70%",
            x=0.5, y=-0.15, xref='paper', yref='paper',
            showarrow=False, font=dict(size=10, color='#aaaaaa')
        )]
    )

    return fig


def create_probability_histogram(results_df: pd.DataFrame, prediction_task: str = "Śmiertelność") -> go.Figure:
    """Utwórz histogram prawdopodobieństw."""
    outcome_label = "zgonu" if prediction_task == "Śmiertelność" else "potrzeby dializy"
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=results_df['probability'] * 100,
        nbinsx=20,
        marker_color='#2874a6',
        opacity=0.8,
        name='Pacjenci',
        hovertemplate='Zakres: %{x}%<br>Liczba: %{y}<extra></extra>'
    ))

    # Linie progowe z legendą
    fig.add_trace(go.Scatter(
        x=[30, 30], y=[0, results_df.shape[0]],
        mode='lines', line=dict(dash='dash', color='#28a745', width=2),
        name='Próg niskie/umiarkowane (30%)', showlegend=True
    ))
    fig.add_trace(go.Scatter(
        x=[70, 70], y=[0, results_df.shape[0]],
        mode='lines', line=dict(dash='dash', color='#dc3545', width=2),
        name='Próg umiarkowane/wysokie (70%)', showlegend=True
    ))

    fig.update_layout(
        title=dict(text="Rozkład prawdopodobieństw ryzyka", font=dict(size=18, color='#ffffff')),
        xaxis=dict(
            title=dict(text=f"Prawdopodobieństwo {outcome_label} (%)", font=dict(color='#ffffff')),
            tickfont=dict(color='#ffffff'),
            gridcolor='#444444',
            range=[0, 100]
        ),
        yaxis=dict(
            title=dict(text="Liczba pacjentów", font=dict(color='#ffffff')),
            tickfont=dict(color='#ffffff'),
            gridcolor='#444444'
        ),
        font=dict(color='#ffffff'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        template='plotly_dark',
        height=350,
        bargap=0.1,
        showlegend=True,
        legend=dict(
            orientation='h', yanchor='top', y=-0.2, xanchor='center', x=0.5,
            font=dict(color='#ffffff', size=10)
        )
    )

    return fig


def create_age_risk_scatter(results_df: pd.DataFrame, prediction_task: str = "Śmiertelność") -> go.Figure:
    """Utwórz wykres punktowy wiek vs ryzyko."""
    outcome_label = "zgonu" if prediction_task == "Śmiertelność" else "potrzeby dializy"
    colors = {
        'low': '#28a745',
        'moderate': '#ffc107',
        'high': '#dc3545'
    }

    # Downsampling dla dużych zbiorów danych (>5000 punktów)
    MAX_SCATTER_POINTS = 5000
    if len(results_df) > MAX_SCATTER_POINTS:
        plot_df = results_df.sample(n=MAX_SCATTER_POINTS, random_state=42)
    else:
        plot_df = results_df

    fig = go.Figure()

    for risk_level in ['low', 'moderate', 'high']:
        mask = plot_df['risk_level'] == risk_level
        if mask.any():
            fig.add_trace(go.Scatter(
                x=plot_df.loc[mask, 'wiek'],
                y=plot_df.loc[mask, 'probability'] * 100,
                mode='markers',
                marker=dict(
                    size=10,
                    color=colors[risk_level],
                    opacity=0.7
                ),
                name={'low': 'Niskie', 'moderate': 'Umiarkowane', 'high': 'Wysokie'}[risk_level],
                text=plot_df.loc[mask, 'patient_id'],
                hovertemplate='<b>%{text}</b><br>Wiek: %{x}<br>Ryzyko: %{y:.1f}%<extra></extra>'
            ))

    # Linie progowe ryzyka
    fig.add_hline(y=30, line_dash="dot", line_color="#28a745", opacity=0.5,
                  annotation_text="30% — próg niskie/umiarkowane",
                  annotation_font_color="#28a745", annotation_font_size=10)
    fig.add_hline(y=70, line_dash="dot", line_color="#dc3545", opacity=0.5,
                  annotation_text="70% — próg umiarkowane/wysokie",
                  annotation_font_color="#dc3545", annotation_font_size=10)

    fig.update_layout(
        title=dict(text=f"Wiek a ryzyko {outcome_label}", font=dict(size=18, color='#ffffff')),
        xaxis=dict(
            title=dict(text="Wiek pacjenta (lata)", font=dict(color='#ffffff', size=13)),
            tickfont=dict(color='#ffffff'),
            gridcolor='#444444'
        ),
        yaxis=dict(
            title=dict(text=f"Prawdopodobieństwo {outcome_label} (%)", font=dict(color='#ffffff', size=13)),
            tickfont=dict(color='#ffffff'),
            gridcolor='#444444',
            range=[0, 100]
        ),
        font=dict(color='#ffffff'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        template='plotly_dark',
        height=400,
        legend=dict(
            title=dict(text="Poziom ryzyka", font=dict(color='#ffffff', size=12)),
            font=dict(color='#ffffff')
        )
    )

    return fig


def export_results_to_csv(results_df: pd.DataFrame) -> str:
    """Eksportuj wyniki do CSV."""
    export_df = results_df[['patient_id', 'wiek', 'plec', 'liczba_narzadow',
                            'probability_pct', 'risk_level_pl', 'top_factors']].copy()
    export_df.columns = ['ID Pacjenta', 'Wiek', 'Płeć', 'Liczba narządów',
                         'Ryzyko (%)', 'Poziom ryzyka', 'Główne czynniki']
    return export_df.to_csv(index=False, encoding='utf-8-sig')


def export_results_to_json(results_df: pd.DataFrame) -> str:
    """Eksportuj wyniki do JSON."""
    export_data = {
        'analysis_date': datetime.now().isoformat(),
        'total_patients': len(results_df),
        'summary': {
            'low_risk': int((results_df['risk_level'] == 'low').sum()),
            'moderate_risk': int((results_df['risk_level'] == 'moderate').sum()),
            'high_risk': int((results_df['risk_level'] == 'high').sum()),
            'avg_probability': float(results_df['probability'].mean())
        },
        'patients': results_df.to_dict(orient='records')
    }
    return json.dumps(export_data, indent=2, ensure_ascii=False)


def create_bar_chart(factors: list, title: str = "Ważność czynników") -> go.Figure:
    """Utwórz wykres słupkowy."""
    if not factors:
        return None

    names = [f["feature"] for f in factors]
    values = [abs(f["contribution"]) for f in factors]
    colors = ["#dc3545" if f["contribution"] > 0 else "#28a745" for f in factors]

    fig = go.Figure(go.Bar(
        y=names,
        x=values,
        orientation='h',
        marker_color=colors,
        text=[f"{v:.3f}" for v in values],
        textposition="outside",
        textfont=dict(size=13, color='#ffffff', family='Arial Black'),
        showlegend=False
    ))

    # Legenda kolorów
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='markers',
        marker=dict(size=10, color='#dc3545', symbol='square'),
        name='Czynnik ryzyka (zwiększa prawdopodobieństwo)',
        showlegend=True
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='markers',
        marker=dict(size=10, color='#28a745', symbol='square'),
        name='Czynnik ochronny (zmniejsza prawdopodobieństwo)',
        showlegend=True
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color='#ffffff')),
        xaxis=dict(
            title=dict(text="Bezwzględna waga cechy w modelu lokalnym", font=dict(size=13, color='#ffffff')),
            tickfont=dict(color='#ffffff'),
            gridcolor='#444444'
        ),
        yaxis=dict(
            title=dict(text="Cecha kliniczna", font=dict(size=13, color='#ffffff')),
            tickfont=dict(color='#ffffff')
        ),
        height=400,
        margin=dict(l=20, r=20, t=50, b=40),
        font=dict(color='#ffffff', size=13),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        template='plotly_dark',
        showlegend=True,
        legend=dict(
            orientation='h', yanchor='top', y=-0.15, xanchor='center', x=0.5,
            font=dict(color='#ffffff', size=11)
        )
    )

    return fig


def create_beeswarm_chart(shap_matrix: list, feature_values: list, feature_names: list) -> go.Figure:
    """Utwórz wykres beeswarm SHAP (global)."""
    _np = np
    shap_arr = _np.array(shap_matrix)
    feat_arr = _np.array(feature_values)
    n_samples, n_features = shap_arr.shape

    # Mean |SHAP| per feature for ordering
    mean_abs = _np.mean(_np.abs(shap_arr), axis=0)
    order = _np.argsort(mean_abs)[::-1][:min(15, n_features)]

    fig = go.Figure()
    for rank, idx in enumerate(order):
        fname = feature_names[idx] if idx < len(feature_names) else f"Feature {idx}"
        shap_vals = shap_arr[:, idx]
        feat_vals = feat_arr[:, idx]
        jitter = _np.random.RandomState(idx).uniform(-0.3, 0.3, n_samples)

        fig.add_trace(go.Scatter(
            x=shap_vals,
            y=[rank + j for j in jitter],
            mode='markers',
            marker=dict(
                size=5,
                color=feat_vals,
                colorscale='RdBu_r',
                showscale=(rank == 0),
                colorbar=dict(
                    title=dict(text="Wartość cechy<br>(niska → wysoka)", font=dict(color='#ffffff', size=11)),
                    tickfont=dict(color='#ffffff'),
                    len=0.6, y=0.5
                ),
                opacity=0.7,
            ),
            name=fname,
            showlegend=False,
            hovertemplate=f"<b>{fname}</b><br>SHAP: %{{x:.3f}}<br>Wartość: %{{marker.color:.2f}}<extra></extra>"
        ))

    ytick_labels = [feature_names[idx] if idx < len(feature_names) else f"Feature {idx}" for idx in order]

    fig.update_layout(
        title=dict(text="Beeswarm — SHAP values (globalny)", font=dict(size=18, color='#ffffff')),
        xaxis=dict(
            title=dict(text="Wartość SHAP (wpływ na predykcję: <0 = zmniejsza, >0 = zwiększa ryzyko)",
                       font=dict(size=11, color='#ffffff')),
            tickfont=dict(color='#ffffff'), gridcolor='#444444', zerolinecolor='#888888'
        ),
        yaxis=dict(
            title=dict(text="Cecha kliniczna (posortowane wg średniego |SHAP|)", font=dict(size=11, color='#ffffff')),
            tickvals=list(range(len(order))), ticktext=ytick_labels, tickfont=dict(color='#ffffff')
        ),
        height=500,
        margin=dict(l=20, r=20, t=50, b=40),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', template='plotly_dark',
    )

    # Linia zerowa z opisem
    fig.add_vline(x=0, line_dash="solid", line_color="#888888", line_width=1)

    return fig


def create_global_importance_bar(importance_dict: dict, title: str = "Globalna ważność cech") -> go.Figure:
    """Utwórz poziomy wykres słupkowy globalnej ważności."""
    sorted_items = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:15]
    names = [item[0] for item in reversed(sorted_items)]
    values = [item[1] for item in reversed(sorted_items)]

    fig = go.Figure(go.Bar(
        y=names, x=values, orientation='h',
        marker_color='#2874a6',
        text=[f"{v:.4f}" for v in values],
        textposition="outside",
        textfont=dict(size=12, color='#ffffff'),
        name='Średni |SHAP|',
        hovertemplate='<b>%{y}</b><br>Średnia |SHAP|: %{x:.4f}<extra></extra>'
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color='#ffffff')),
        xaxis=dict(
            title=dict(text="Średnia |SHAP value| — im wyższa, tym ważniejsza cecha",
                       font=dict(size=11, color='#ffffff')),
            tickfont=dict(color='#ffffff'), gridcolor='#444444'
        ),
        yaxis=dict(
            title=dict(text="Cecha kliniczna", font=dict(size=11, color='#ffffff')),
            tickfont=dict(color='#ffffff')
        ),
        height=450, margin=dict(l=20, r=80, t=50, b=40),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', template='plotly_dark',
        showlegend=False
    )
    return fig


def create_pdp_chart(x_values: list, y_values: list, feature_name: str) -> go.Figure:
    """Utwórz wykres Partial Dependence Profile."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_values, y=y_values,
        mode='lines', fill='tozeroy',
        line=dict(color='#2874a6', width=2),
        fillcolor='rgba(40, 116, 166, 0.2)',
        name=f'PDP: {feature_name}',
        hovertemplate=f'<b>{feature_name}</b>: %{{x:.2f}}<br>Średnia predykcja: %{{y:.4f}}<extra></extra>'
    ))
    fig.update_layout(
        title=dict(text=f"PDP — {feature_name}", font=dict(size=18, color='#ffffff')),
        xaxis=dict(
            title=dict(text=f"Wartość cechy: {feature_name}", font=dict(size=12, color='#ffffff')),
            tickfont=dict(color='#ffffff'), gridcolor='#444444'
        ),
        yaxis=dict(
            title=dict(text="Średnia predykcja modelu (ceteris paribus)", font=dict(size=12, color='#ffffff')),
            tickfont=dict(color='#ffffff'), gridcolor='#444444'
        ),
        height=400, margin=dict(l=20, r=20, t=50, b=40),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', template='plotly_dark',
        showlegend=False
    )

    # Opis interpretacji
    fig.add_annotation(
        x=0.5, y=-0.18, xref='paper', yref='paper',
        text="Linia pokazuje jak zmiana wartości cechy wpływa na średnią predykcję modelu (przy stałych pozostałych cechach)",
        showarrow=False, font=dict(size=9, color='#aaaaaa')
    )

    return fig


def create_variable_importance_chart(importance_dict: dict) -> go.Figure:
    """Utwórz wykres permutation variable importance (DALEX)."""
    sorted_items = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:15]
    names = [item[0] for item in reversed(sorted_items)]
    values = [item[1] for item in reversed(sorted_items)]

    fig = go.Figure(go.Bar(
        y=names, x=values, orientation='h',
        marker_color='#e74c3c',
        text=[f"{v:.4f}" for v in values],
        textposition="outside",
        textfont=dict(size=12, color='#ffffff'),
        name='Dropout loss',
        hovertemplate='<b>%{y}</b><br>Dropout loss: %{x:.4f}<extra></extra>'
    ))
    fig.update_layout(
        title=dict(text="Permutation Variable Importance (DALEX)", font=dict(size=18, color='#ffffff')),
        xaxis=dict(
            title=dict(text="Dropout loss — spadek AUC po permutacji cechy (im wyższy, tym ważniejsza cecha)",
                       font=dict(size=11, color='#ffffff')),
            tickfont=dict(color='#ffffff'), gridcolor='#444444'
        ),
        yaxis=dict(
            title=dict(text="Cecha kliniczna", font=dict(size=11, color='#ffffff')),
            tickfont=dict(color='#ffffff')
        ),
        height=450, margin=dict(l=20, r=80, t=50, b=40),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', template='plotly_dark',
        showlegend=False
    )
    return fig


def create_heatmap_chart(matrix: dict, labels: list, title: str = "Macierz zgodności") -> go.Figure:
    """Utwórz heatmap z annotacjami (Jaccard / Spearman)."""
    z_data = [[matrix.get(r, {}).get(c, 0) for c in labels] for r in labels]

    annotations = []
    for i, r in enumerate(labels):
        for j, c in enumerate(labels):
            val = matrix.get(r, {}).get(c, 0)
            annotations.append(dict(
                x=j, y=i, text=f"{val:.2f}",
                font=dict(color='white' if val < 0.7 else 'black', size=14),
                showarrow=False,
            ))

    fig = go.Figure(go.Heatmap(
        z=z_data, x=labels, y=labels,
        colorscale='Viridis', zmin=0, zmax=1,
        colorbar=dict(
            title=dict(text="Podobieństwo<br>(0=brak, 1=pełne)", font=dict(color='#ffffff', size=11)),
            tickfont=dict(color='#ffffff')
        ),
        hovertemplate='<b>%{x}</b> vs <b>%{y}</b><br>Podobieństwo: %{z:.2f}<extra></extra>'
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color='#ffffff')),
        xaxis=dict(
            title=dict(text="Metoda XAI", font=dict(size=11, color='#ffffff')),
            tickfont=dict(color='#ffffff')
        ),
        yaxis=dict(
            title=dict(text="Metoda XAI", font=dict(size=11, color='#ffffff')),
            tickfont=dict(color='#ffffff'), autorange='reversed'
        ),
        annotations=annotations,
        height=400, margin=dict(l=20, r=20, t=50, b=40),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', template='plotly_dark',
    )
    return fig


def create_ebm_local_chart(contributions: list) -> go.Figure:
    """Utwórz wykres waterfall dla EBM local (analogiczny do SHAP)."""
    if not contributions:
        return None
    sorted_contribs = sorted(contributions, key=lambda x: abs(x.get("contribution", x.get("score", 0))), reverse=True)[:10]
    names = [c.get("feature", c.get("name", "?")) for c in sorted_contribs]
    values = [c.get("contribution", c.get("score", 0)) for c in sorted_contribs]

    fig = go.Figure(go.Waterfall(
        name="Kontribucje EBM", orientation="h",
        y=names, x=values,
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        decreasing={"marker": {"color": "#28a745"}},
        increasing={"marker": {"color": "#dc3545"}},
        text=[f"{v:+.3f}" for v in values],
        textposition="outside",
        textfont=dict(size=13, color='#ffffff', family='Arial Black'),
        showlegend=False
    ))

    # Legenda kolorów
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='markers',
        marker=dict(size=10, color='#dc3545', symbol='square'),
        name='Zwiększa predykcję (score > 0)',
        showlegend=True
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='markers',
        marker=dict(size=10, color='#28a745', symbol='square'),
        name='Zmniejsza predykcję (score < 0)',
        showlegend=True
    ))

    fig.update_layout(
        title=dict(text="EBM — lokalne kontribucje", font=dict(size=18, color='#ffffff')),
        xaxis=dict(
            title=dict(text="Wpływ na predykcję (log-odds)", font=dict(size=12, color='#ffffff')),
            tickfont=dict(color='#ffffff'), gridcolor='#444444', zerolinecolor='#888888'
        ),
        yaxis=dict(
            title=dict(text="Cecha kliniczna", font=dict(size=12, color='#ffffff')),
            tickfont=dict(color='#ffffff')
        ),
        height=400, margin=dict(l=20, r=20, t=50, b=40),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', template='plotly_dark',
        showlegend=True,
        legend=dict(
            orientation='h', yanchor='top', y=-0.15, xanchor='center', x=0.5,
            font=dict(color='#ffffff', size=11)
        )
    )
    return fig


def create_feature_function_chart(names: list, scores: list, feature: str) -> go.Figure:
    """Utwórz wykres kształtu funkcji cechy EBM (shape function)."""
    all_numeric = all(isinstance(n, (int, float)) for n in names)

    fig = go.Figure()
    if all_numeric and len(names) > 5:
        # Line chart for numerical features
        fig.add_trace(go.Scatter(
            x=names, y=scores, mode='lines+markers',
            line=dict(color='#2874a6', width=2),
            marker=dict(size=4, color='#2874a6'),
            name=f'Score EBM: {feature}',
            hovertemplate=f'<b>{feature}</b>: %{{x:.2f}}<br>Score: %{{y:+.3f}}<extra></extra>'
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="#888888",
                      annotation_text="Linia bazowa (score = 0)",
                      annotation_font_color="#aaaaaa", annotation_font_size=10)
    else:
        # Bar chart for categorical/few bins
        fig.add_trace(go.Bar(
            x=[str(n) for n in names], y=scores,
            marker_color=['#dc3545' if s > 0 else '#28a745' for s in scores],
            text=[f"{s:+.3f}" for s in scores],
            textposition="outside",
            textfont=dict(size=11, color='#ffffff'),
            showlegend=False,
            hovertemplate=f'<b>{feature}</b>: %{{x}}<br>Score: %{{y:+.3f}}<extra></extra>'
        ))
        # Legenda kolorów dla wykresu słupkowego
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(size=10, color='#dc3545', symbol='square'),
            name='Zwiększa ryzyko (score > 0)', showlegend=True
        ))
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(size=10, color='#28a745', symbol='square'),
            name='Zmniejsza ryzyko (score < 0)', showlegend=True
        ))

    fig.update_layout(
        title=dict(text=f"Funkcja kształtu — {feature}", font=dict(size=18, color='#ffffff')),
        xaxis=dict(
            title=dict(text=f"Wartość cechy: {feature}", font=dict(size=12, color='#ffffff')),
            tickfont=dict(color='#ffffff'), gridcolor='#444444'
        ),
        yaxis=dict(
            title=dict(text="Score (log-odds) — wpływ na predykcję", font=dict(size=12, color='#ffffff')),
            tickfont=dict(color='#ffffff'), gridcolor='#444444', zerolinecolor='#888888'
        ),
        height=400, margin=dict(l=20, r=20, t=50, b=40),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', template='plotly_dark',
        showlegend=True,
        legend=dict(
            orientation='h', yanchor='top', y=-0.15, xanchor='center', x=0.5,
            font=dict(color='#ffffff', size=11)
        )
    )

    # Opis interpretacji
    fig.add_annotation(
        x=0.5, y=-0.22, xref='paper', yref='paper',
        text="Score > 0: cecha zwiększa prawdopodobieństwo zdarzenia | Score < 0: cecha zmniejsza prawdopodobieństwo",
        showarrow=False, font=dict(size=9, color='#aaaaaa')
    )

    return fig


def create_ranking_heatmap(rankings: dict) -> go.Figure:
    """Utwórz heatmap pozycji cech we wszystkich metodach XAI."""
    methods = list(rankings.keys())
    all_features = []
    for m in methods:
        for f in rankings[m]:
            if f not in all_features:
                all_features.append(f)
    all_features = all_features[:10]

    z_data = []
    annotations = []
    for i, feat in enumerate(all_features):
        row = []
        for j, method in enumerate(methods):
            r = rankings[method]
            pos = r.index(feat) + 1 if feat in r else len(r) + 1
            row.append(pos)
            annotations.append(dict(
                x=j, y=i, text=str(pos),
                font=dict(color='white' if pos > 3 else 'black', size=13),
                showarrow=False,
            ))
        z_data.append(row)

    fig = go.Figure(go.Heatmap(
        z=z_data, x=methods, y=all_features,
        colorscale='Viridis_r', zmin=1, zmax=10,
        colorbar=dict(
            title=dict(text="Pozycja w rankingu<br>(1=najważniejsza)", font=dict(color='#ffffff', size=11)),
            tickfont=dict(color='#ffffff')
        ),
        hovertemplate='<b>%{y}</b> w %{x}<br>Pozycja: %{z}<extra></extra>'
    ))
    fig.update_layout(
        title=dict(text="Rankingi cech — porównanie metod", font=dict(size=18, color='#ffffff')),
        xaxis=dict(
            title=dict(text="Metoda XAI", font=dict(size=11, color='#ffffff')),
            tickfont=dict(color='#ffffff')
        ),
        yaxis=dict(
            title=dict(text="Cecha kliniczna", font=dict(size=11, color='#ffffff')),
            tickfont=dict(color='#ffffff')
        ),
        annotations=annotations,
        height=450, margin=dict(l=20, r=20, t=50, b=40),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', template='plotly_dark',
    )

    # Opis interpretacji
    fig.add_annotation(
        x=0.5, y=-0.12, xref='paper', yref='paper',
        text="Ciemniejszy kolor = wyższa pozycja (ważniejsza cecha). Spójna pozycja cechy we wszystkich metodach zwiększa wiarygodność.",
        showarrow=False, font=dict(size=9, color='#aaaaaa')
    )

    return fig


# ============================================================================
# SIDEBAR - CEL PREDYKCJI
# ============================================================================

st.sidebar.markdown('<h2 style="color: #1a5276;">Cel predykcji</h2>', unsafe_allow_html=True)

prediction_task = st.sidebar.radio(
    "Cel predykcji:",
    options=["Śmiertelność", "Potrzeba dializy"],
    index=0,
    help="Wybierz zadanie predykcyjne: śmiertelność (Zgon) lub potrzeba dializy"
)

# Clear stale results when task changes
_prev_task = st.session_state.get("_prediction_task_prev")
if _prev_task is not None and _prev_task != prediction_task:
    for _key in ["batch_results", "comp_pred_2", "comp_pred_3", "analyzed", "messages", "_chat_prediction", "_chat_explanation"]:
        st.session_state.pop(_key, None)
st.session_state["_prediction_task_prev"] = prediction_task

st.sidebar.markdown("---")

# ============================================================================
# SIDEBAR - TRYB ANALIZY
# ============================================================================

st.sidebar.markdown('<h2 style="color: #1a5276;">Tryb analizy</h2>', unsafe_allow_html=True)

analysis_mode = st.sidebar.radio(
    "Wybierz tryb:",
    options=["Pojedynczy pacjent", "Analiza masowa"],
    index=0,
    help="Wybierz czy chcesz analizować pojedynczego pacjenta czy wiele pacjentów z pliku"
)

# Sposób wprowadzania danych (tylko w trybie pojedynczego pacjenta)
if analysis_mode == "Pojedynczy pacjent":
    input_mode = st.sidebar.radio(
        "Sposób wprowadzania danych:",
        ["Formularz", "Chat (opis słowny)"],
        index=0,
        help="Formularz: klasyczne pola. Chat: opisz pacjenta tekstem, dane zostaną wyodrębnione automatycznie."
    )
else:
    input_mode = "Formularz"

st.sidebar.markdown("---")

# ============================================================================
# SIDEBAR - WYBÓR MODELU
# ============================================================================

MODEL_DISPLAY_NAMES = {
    'random_forest': 'Random Forest',
    'naive_bayes': 'Naive Bayes',
    'calibrated_svm': 'Calibrated SVM',
    'xgboost': 'XGBoost',
    'stacking_ensemble': 'Stacking Ensemble',
    'logistic_regression': 'Logistic Regression',
    'svm': 'SVM',
}

@st.cache_data(ttl=60)
def get_available_models() -> Optional[Dict]:
    """Pobierz listę dostępnych modeli z API (cache 60s)."""
    try:
        response = requests.get(f"{API_URL}/models/available", timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return None

@st.cache_data(ttl=60)
def get_available_dialysis_models() -> Optional[Dict]:
    """Pobierz listę dostępnych modeli dializy z API (cache 60s)."""
    try:
        response = requests.get(f"{API_URL}/models/dialysis/available", timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return None

selected_model_type = None

# Fetch modeli potrzebny na stronie powitalnej
if prediction_task == "Śmiertelność":
    available_models_data = get_available_models()
else:
    available_models_data = get_available_dialysis_models()

# ============================================================================
# SIDEBAR - ANALIZA MASOWA (UPLOAD)
# ============================================================================

if analysis_mode == "Analiza masowa":
    # Status API i tryb pracy
    api_status = get_api_status()
    current_mode = api_status.get('current_mode', 'unavailable')
    model_loaded = api_status.get('model_loaded', False)

    # Wyświetl status trybu
    mode_icons = {
        'api': '[OK]',
        'demo': '[DEMO]',
        'unavailable': '[X]'
    }
    mode_labels = {
        'api': 'API (model ML)',
        'demo': 'Demo (symulacja)',
        'unavailable': 'Niedostępny'
    }
    mode_descriptions = {
        'api': 'Predykcje z wytrenowanego modelu XGBoost',
        'demo': 'Predykcje symulowane (bez modelu)',
        'unavailable': 'API niedostępne'
    }

    st.sidebar.markdown(
        f"""<div style="background: linear-gradient(135deg, #1a5276, #2874a6);
            padding: 12px; border-radius: 8px; margin-bottom: 15px;">
            <span style="font-size: 1.1em; font-weight: bold; color: white;">
                {mode_icons.get(current_mode, '⚪')} Tryb: {mode_labels.get(current_mode, current_mode)}
            </span><br>
            <span style="font-size: 0.85em; color: #d5dbdb;">
                {mode_descriptions.get(current_mode, '')}
            </span>
        </div>""",
        unsafe_allow_html=True
    )

    # Informacja o trybie (jeśli API dostępne)
    if current_mode != 'unavailable':
        with st.sidebar.expander("Ustawienia trybu", expanded=False):
            if model_loaded:
                st.info("Model ML załadowany. Możesz używać pełnych predykcji.")
            else:
                st.warning("Model ML niezaładowany. Uruchom API z modelem lub użyj trybu demo.")

            st.caption("""
            **Jak uruchomić z pełnym modelem:**
            1. Wytrenuj model: `python scripts/train_model.py`
            2. Uruchom API: `python -m src.api.main`
            """)

    st.sidebar.markdown('<h2 style="color: #1a5276;">Wgraj plik</h2>', unsafe_allow_html=True)

    uploaded_file = st.sidebar.file_uploader(
        "Wybierz plik CSV lub JSON",
        type=['csv', 'json'],
        help="Plik powinien zawierać dane pacjentów. Obsługiwane formaty: CSV, JSON. Max 100MB."
    )

    # Informacja o limitach
    st.sidebar.caption("**Obsługiwane:** do 50,000+ pacjentów | Max 100MB")

    with st.sidebar.expander("Format pliku", expanded=False):
        st.markdown("""
        **Wymagane kolumny:**
        - `wiek` (lub `age`) - wiek pacjenta
        - `plec` (lub `sex`) - płeć (K/M)

        **Opcjonalne kolumny:**
        - `wiek_rozpoznania`
        - `liczba_narzadow`
        - `nerki`, `serce`, `csn`, `neuro`
        - `oit`, `dializa`, `kreatynina`, `crp`

        **Przykład CSV:**
        ```
        id,wiek,plec,nerki,oit
        P001,65,M,1,0
        P002,45,K,0,1
        ```

        **Przykład JSON:**
        ```json
        [
          {"id": "P001", "wiek": 65, "plec": "M"},
          {"id": "P002", "wiek": 45, "plec": "K"}
        ]
        ```
        """)

    batch_analyze_button = st.sidebar.button(
        "Analizuj plik",
        type="primary",
        disabled=uploaded_file is None
    )

    # Pobierz przykładowy plik
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Pobierz przykładowy plik:**")

    sample_csv = """id,wiek,plec,wiek_rozpoznania,liczba_narzadow,nerki,serce,oit,dializa,kreatynina,crp
P001,65,M,60,3,1,0,0,0,120,45
P002,45,K,40,2,0,0,0,0,85,22
P003,72,M,68,4,1,1,1,0,180,88
P004,38,K,35,1,0,0,0,0,75,15
P005,55,M,50,2,1,0,0,0,110,35
P006,68,K,62,3,1,1,0,1,220,95
P007,42,M,38,2,0,0,0,0,90,28
P008,78,K,70,5,1,1,1,1,250,120
P009,51,M,48,2,0,0,0,0,95,30
P010,63,K,58,3,1,0,1,0,145,55"""

    st.sidebar.download_button(
        label="Pobierz przykład CSV",
        data=sample_csv,
        file_name="przykladowi_pacjenci.csv",
        mime="text/csv"
    )

# ============================================================================
# SIDEBAR - DANE PACJENTA (TRYB POJEDYNCZY)
# ============================================================================

if analysis_mode == "Pojedynczy pacjent" and input_mode == "Formularz":
    st.sidebar.markdown('<h2 style="color: #1a5276;">Dane pacjenta</h2>', unsafe_allow_html=True)

    with st.sidebar.expander("Dane demograficzne", expanded=True):
        wiek = st.number_input("Wiek", min_value=18, max_value=100, value=55)
        plec = st.selectbox("Płeć", options=["Kobieta", "Mężczyzna"])
        wiek_rozpoznania = st.number_input("Wiek rozpoznania", min_value=0, max_value=100, value=50)
        if wiek_rozpoznania > wiek:
            st.warning(f"Wiek rozpoznania ({wiek_rozpoznania}) nie może być większy niż wiek ({wiek}). Zostanie skorygowany.")

    with st.sidebar.expander("Manifestacje narządowe", expanded=True):
        liczba_narzadow = st.slider("Liczba zajętych narządów", 0, 10, 2)
        manifestacja_nerki = st.checkbox("Nerki")
        manifestacja_serce = st.checkbox("Serce/naczynia")
        if prediction_task == "Śmiertelność":
            manifestacja_csn = st.checkbox("Ośrodkowy układ nerwowy")
        manifestacja_neuro = st.checkbox("Obwodowy układ nerwowy")
        if prediction_task == "Śmiertelność":
            manifestacja_pokarm = st.checkbox("Układ pokarmowy")
        if prediction_task == "Potrzeba dializy":
            manifestacja_oddechowy = st.checkbox("Układ oddechowy")

    with st.sidebar.expander("Przebieg choroby", expanded=False):
        oit = st.checkbox("Zaostrzenia wymagające OIT")
        if prediction_task == "Potrzeba dializy":
            hospital = st.checkbox("Zaostrzenia wymagające hospitalizacji")
        kreatynina = st.number_input("Kreatynina (μmol/L)", min_value=0.0, value=100.0)
        crp = st.number_input("Max CRP (mg/L)", min_value=0.0, value=30.0)

    with st.sidebar.expander("Leczenie", expanded=False):
        plazmaferezy = st.checkbox("Plazmaferezy")
        if prediction_task == "Śmiertelność":
            dializa = st.checkbox("Dializa")
        if prediction_task == "Potrzeba dializy":
            pulsy = st.checkbox("Pulsy steroidowe")
        sterydy = st.number_input("Dawka sterydów (g)", min_value=0.0, value=0.5)
        czas_sterydow = st.number_input("Czas sterydów (mies.)", min_value=0, value=12)

    with st.sidebar.expander("Powikłania", expanded=False):
        powiklania_serce = st.checkbox("Powikłania sercowo-płucne")
        powiklania_infekcja = st.checkbox("Infekcje")

    # Przycisk analizy
    analyze_button = st.sidebar.button("Analizuj", type="primary")

    # Przygotuj dane pacjenta w zależności od celu predykcji
    if prediction_task == "Śmiertelność":
        patient_data = {
            "wiek": wiek,
            "plec": 1 if plec == "Mężczyzna" else 0,
            "wiek_rozpoznania": wiek_rozpoznania,
            "opoznienie_rozpoznia": max(0, wiek - wiek_rozpoznania),
            "liczba_zajetych_narzadow": liczba_narzadow,
            "manifestacja_sercowo_naczyniowy": 1 if manifestacja_serce else 0,
            "manifestacja_nerki": 1 if manifestacja_nerki else 0,
            "manifestacja_pokarmowy": 1 if manifestacja_pokarm else 0,
            "manifestacja_zajecie_csn": 1 if manifestacja_csn else 0,
            "manifestacja_neurologiczny": 1 if manifestacja_neuro else 0,
            "zaostrz_wymagajace_oit": 1 if oit else 0,
            "kreatynina": kreatynina,
            "max_crp": crp,
            "plazmaferezy": 1 if plazmaferezy else 0,
            "dializa": 1 if dializa else 0,
            "sterydy_dawka_g": sterydy,
            "czas_sterydow": czas_sterydow,
            "powiklania_serce_pluca": 1 if powiklania_serce else 0,
            "powiklania_infekcja": 1 if powiklania_infekcja else 0
        }
    else:
        patient_data = {
            "wiek": wiek,
            "plec": 1 if plec == "Mężczyzna" else 0,
            "wiek_rozpoznania": wiek_rozpoznania,
            "liczba_zajetych_narzadow": liczba_narzadow,
            "manifestacja_sercowo_naczyniowy": 1 if manifestacja_serce else 0,
            "manifestacja_nerki": 1 if manifestacja_nerki else 0,
            "manifestacja_neurologiczny": 1 if manifestacja_neuro else 0,
            "manifestacja_oddechowy": 1 if manifestacja_oddechowy else 0,
            "zaostrz_wymagajace_oit": 1 if oit else 0,
            "zaostrz_wymagajace_hospital": 1 if hospital else 0,
            "kreatynina": kreatynina,
            "max_crp": crp,
            "plazmaferezy": 1 if plazmaferezy else 0,
            "pulsy": 1 if pulsy else 0,
            "sterydy_dawka_g": sterydy,
            "czas_sterydow": czas_sterydow,
            "powiklania_serce_pluca": 1 if powiklania_serce else 0,
            "powiklania_infekcja": 1 if powiklania_infekcja else 0
        }
    # Bug #10: Resetuj 'analyzed' gdy dane formularza się zmieniają
    _form_hash = hash(json.dumps(patient_data, sort_keys=True, default=str))
    if st.session_state.get('_last_form_hash') != _form_hash:
        st.session_state['_last_form_hash'] = _form_hash
        st.session_state['analyzed'] = False
        st.session_state.pop('_xai_cache', None)

    # Zmienne dla trybu masowego (nieużywane w trybie pojedynczym)
    uploaded_file = None
    batch_analyze_button = False
elif analysis_mode == "Pojedynczy pacjent" and input_mode == "Chat (opis słowny)":
    # Tryb chat — dane zbierane konwersacyjnie
    st.sidebar.markdown('<h2 style="color: #1a5276;">Dane pacjenta (chat)</h2>', unsafe_allow_html=True)

    # Inicjalizuj accumulated_patient w session_state
    if "_accumulated_patient" not in st.session_state:
        st.session_state["_accumulated_patient"] = {}

    _acc = st.session_state["_accumulated_patient"]
    if _acc:
        with st.sidebar.expander("Zebrane dane", expanded=True):
            _field_labels = {
                "wiek": "Wiek", "plec": "Płeć", "kreatynina": "Kreatynina (μmol/L)",
                "max_crp": "Max CRP (mg/L)", "manifestacja_nerki": "Nerki",
                "manifestacja_sercowo_naczyniowy": "Serce/naczynia",
                "liczba_zajetych_narzadow": "Zajęte narządy",
                "zaostrz_wymagajace_oit": "OIT", "wiek_rozpoznania": "Wiek rozpoznania",
            }
            for k, v in _acc.items():
                label = _field_labels.get(k, k)
                if k == "plec":
                    st.sidebar.text(f"{label}: {'M' if v == 1 else 'K'}")
                elif isinstance(v, (int, float)):
                    st.sidebar.text(f"{label}: {v}")
        if st.sidebar.button("Wyczyść dane"):
            st.session_state["_accumulated_patient"] = {}
            st.session_state.pop("analyzed", None)
            st.session_state.pop("messages", None)
            st.rerun()
    else:
        st.sidebar.info("Opisz pacjenta w oknie czatu, a dane zostaną wyodrębnione automatycznie.")

    # Ustaw patient_data z accumulated lub pusty
    patient_data = dict(st.session_state.get("_accumulated_patient", {}))
    analyze_button = False
    uploaded_file = None
    batch_analyze_button = False
else:
    # Zmienne dla trybu masowego
    analyze_button = False
    patient_data = {}
    # uploaded_file i batch_analyze_button są już zdefiniowane w sekcji sidebar dla trybu masowego

# ============================================================================
# GŁÓWNA SEKCJA
# ============================================================================

if prediction_task == "Śmiertelność":
    st.markdown('<h1 class="main-header">System XAI do predykcji śmiertelności w zapaleniu naczyń</h1>', unsafe_allow_html=True)
else:
    st.markdown('<h1 class="main-header">System XAI do predykcji potrzeby dializy w zapaleniu naczyń</h1>', unsafe_allow_html=True)

_xai_task_type = "dialysis" if prediction_task != "Śmiertelność" else "mortality"

# ============================================================================
# TRYB ANALIZY MASOWEJ
# ============================================================================

if analysis_mode == "Analiza masowa":
    if batch_analyze_button and uploaded_file is not None:
        st.session_state['batch_analyzed'] = True
        st.session_state['batch_file_name'] = uploaded_file.name

        # Parsuj plik
        with st.spinner("Wczytuję plik..."):
            df, error = parse_uploaded_file(uploaded_file)

        if error:
            st.error(f"Błąd: {error}")
        else:
            # Normalizuj dane
            df = normalize_dataframe(df)

            st.success(f"Wczytano {len(df)} pacjentów z pliku {uploaded_file.name}")

            # Pokaż podgląd danych
            with st.expander("Podgląd wczytanych danych", expanded=False):
                st.dataframe(df.head(10), use_container_width=True)

            # Przetwarzaj pacjentów
            st.markdown("### Przetwarzanie pacjentów...")
            progress_bar = st.progress(0)
            status_text = st.empty()

            def update_progress(progress):
                progress_bar.progress(progress)
                status_text.text(f"Przetworzono: {int(progress * len(df))}/{len(df)} pacjentów")

            results_df = process_batch_patients(df, prediction_task, selected_model_type, update_progress)

            progress_bar.progress(1.0)
            status_text.text(f"Zakończono! Przetworzono {len(results_df)} pacjentów.")

            # Pokaż błędy wsadowe jeśli są
            _batch_err_count = st.session_state.get('batch_api_error_count', 0)
            _batch_ok_count = st.session_state.get('batch_api_success_count', len(results_df))
            _batch_errors = st.session_state.get('batch_api_errors', [])
            if _batch_err_count > 0:
                st.warning(f"⚠️ Przetworzone: {_batch_ok_count} pacjentów | Błędy: {_batch_err_count} rekordów")
                if _batch_errors:
                    with st.expander(f"Szczegóły błędów ({len(_batch_errors)})", expanded=False):
                        for err in _batch_errors[:50]:
                            st.text(str(err))

            # Pokaż tryb przetwarzania
            if 'processing_mode' in results_df.columns:
                mode = results_df['processing_mode'].iloc[0] if len(results_df) > 0 else 'unknown'
                if mode == 'api':
                    _api_model_desc = MODEL_DISPLAY_NAMES.get(selected_model_type, selected_model_type or "ML") if prediction_task == "Śmiertelność" else "modeli dializy"
                    st.success(f"**Tryb API** - Predykcje z wytrenowanego modelu {_api_model_desc}")
                elif mode == 'demo':
                    st.warning("**Tryb Demo** - Predykcje symulowane (bez modelu ML)")
                else:
                    st.info(f"Tryb: {mode}")

            # Zapisz wyniki w session state
            st.session_state['batch_results'] = results_df

    # Wyświetl wyniki jeśli są dostępne
    if st.session_state.get('batch_results') is not None:
        results_df = st.session_state['batch_results']

        st.markdown("---")

        # Podsumowanie statystyczne
        st.markdown("## Podsumowanie analizy")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label="Liczba pacjentów",
                value=len(results_df),
                delta=None
            )

        with col2:
            low_count = (results_df['risk_level'] == 'low').sum()
            st.metric(
                label="Niskie ryzyko",
                value=low_count,
                delta=f"{low_count/len(results_df)*100:.1f}%"
            )

        with col3:
            moderate_count = (results_df['risk_level'] == 'moderate').sum()
            st.metric(
                label="Umiarkowane ryzyko",
                value=moderate_count,
                delta=f"{moderate_count/len(results_df)*100:.1f}%"
            )

        with col4:
            high_count = (results_df['risk_level'] == 'high').sum()
            st.metric(
                label="Wysokie ryzyko",
                value=high_count,
                delta=f"{high_count/len(results_df)*100:.1f}%",
                delta_color="inverse"
            )

        # Wykresy
        st.markdown("---")
        st.markdown("## Wizualizacje")

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            fig_pie = create_risk_distribution_chart(results_df)
            st.plotly_chart(fig_pie, use_container_width=True)
            st.caption("Wykres kołowy pokazuje udział pacjentów w każdym poziomie ryzyka. "
                       "Progi: niskie <30%, umiarkowane 30–70%, wysokie >70%.")

        with chart_col2:
            fig_hist = create_probability_histogram(results_df, prediction_task)
            st.plotly_chart(fig_hist, use_container_width=True)
            outcome_hist_label = "zgonu" if prediction_task == "Śmiertelność" else "potrzeby dializy"
            st.caption(f"Histogram rozkładu prawdopodobieństw {outcome_hist_label} w populacji. "
                       "Przerywane linie oznaczają progi klasyfikacji ryzyka.")

        # Wykres scatter
        if prediction_task == "Potrzeba dializy":
            st.markdown("### Analiza wiek vs ryzyko dializy")
        else:
            st.markdown("### Analiza wiek vs ryzyko zgonu")
        fig_scatter = create_age_risk_scatter(results_df, prediction_task)
        st.plotly_chart(fig_scatter, use_container_width=True)
        outcome_scatter_label = "zgonu" if prediction_task == "Śmiertelność" else "potrzeby dializy"
        st.caption(f"Każdy punkt = jeden pacjent. Oś X = wiek, oś Y = prawdopodobieństwo {outcome_scatter_label} (%). "
                   "Kolor odpowiada poziomowi ryzyka: zielony = niskie, żółty = umiarkowane, czerwony = wysokie. "
                   "Przerywane linie na 30% i 70% wyznaczają granice stref.")

        # Tabela wyników
        st.markdown("---")
        st.markdown("## Szczegółowe wyniki")

        # Filtry
        filter_col1, filter_col2 = st.columns(2)

        with filter_col1:
            risk_filter = st.multiselect(
                "Filtruj po poziomie ryzyka:",
                options=['low', 'moderate', 'high'],
                default=['low', 'moderate', 'high'],
                format_func=lambda x: {'low': 'Niskie', 'moderate': 'Umiarkowane', 'high': 'Wysokie'}[x]
            )

        with filter_col2:
            sort_by = st.selectbox(
                "Sortuj po:",
                options=['probability', 'wiek', 'patient_id'],
                format_func=lambda x: {'probability': 'Prawdopodobieństwo', 'wiek': 'Wiek', 'patient_id': 'ID pacjenta'}[x]
            )

        # Zastosuj filtry
        filtered_df = results_df[results_df['risk_level'].isin(risk_filter)]
        filtered_df = filtered_df.sort_values(by=sort_by, ascending=(sort_by == 'patient_id'))

        # Wyświetl tabelę z kolorami
        def color_risk(val):
            if val == 'Niskie':
                return 'background-color: #28a745; color: white'
            elif val == 'Umiarkowane':
                return 'background-color: #ffc107; color: black'
            else:
                return 'background-color: #dc3545; color: white'

        display_df = filtered_df[['patient_id', 'wiek', 'plec', 'liczba_narzadow',
                                   'probability_pct', 'risk_level_pl', 'top_factors']].copy()
        display_df.columns = ['ID', 'Wiek', 'Płeć', 'Narządy', 'Ryzyko', 'Poziom', 'Główne czynniki']

        st.dataframe(
            display_df.style.map(color_risk, subset=['Poziom']),
            use_container_width=True,
            height=400
        )

        st.markdown(f"*Wyświetlono {len(filtered_df)} z {len(results_df)} pacjentów*")

        # Eksport wyników
        st.markdown("---")
        st.markdown("## Eksport wyników")

        export_col1, export_col2 = st.columns(2)

        with export_col1:
            csv_data = export_results_to_csv(results_df)
            st.download_button(
                label="Pobierz wyniki (CSV)",
                data=csv_data,
                file_name=f"wyniki_analizy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

        with export_col2:
            json_data = export_results_to_json(results_df)
            st.download_button(
                label="Pobierz wyniki (JSON)",
                data=json_data,
                file_name=f"wyniki_analizy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )

    elif not st.session_state.get('batch_analyzed', False):
        # Strona powitalna dla trybu masowego
        st.markdown("""
        <div class="upload-zone">
            <h2 style="color: #2874a6;">Analiza masowa pacjentów</h2>
            <p style="font-size: 1.1rem; color: #666;">Wgraj plik CSV lub JSON z listą pacjentów, aby przeprowadzić analizę ryzyka dla wielu osób jednocześnie.</p>
            <hr style="border-color: #4a5568; margin: 1.5rem 0;">
            <p style="color: #888;">← Użyj panelu bocznego, aby wgrać plik</p>
        </div>
        """, unsafe_allow_html=True)

        # Informacje o formacie
        st.markdown("### Obsługiwane formaty")

        format_col1, format_col2 = st.columns(2)

        with format_col1:
            st.markdown("""
            **CSV (Comma-Separated Values)**
            - Pierwszy wiersz: nagłówki kolumn
            - Separatory: przecinek, średnik, tab
            - Kodowanie: UTF-8
            """)

        with format_col2:
            st.markdown("""
            **JSON (JavaScript Object Notation)**
            - Tablica obiektów: `[{...}, {...}]`
            - Obiekt z kluczem `patients` lub `data`
            - Kodowanie: UTF-8
            """)

# ============================================================================
# TRYB POJEDYNCZEGO PACJENTA
# ============================================================================

elif analyze_button or st.session_state.get('analyzed', False):
    st.session_state['analyzed'] = True

    # Pobierz predykcję ze wszystkich modeli
    with st.spinner("Analizuję dane..."):
        if prediction_task == "Śmiertelność":
            multi_result = call_api("/predict/all-models", "POST", patient_data)
        else:
            multi_result = call_api("/predict/dialysis/all-models", "POST", patient_data)

        _demo_mode = False
        if multi_result is None:
            # Demo fallback — wygeneruj syntetyczne wyniki wielu modeli
            _demo_mode = True
            if prediction_task == "Śmiertelność":
                _demo_pred = get_demo_prediction(patient_data)
            else:
                _demo_pred = get_demo_dialysis_prediction(patient_data)
            base_prob = _demo_pred["probability"]
            rng = np.random.RandomState(42)
            if _xai_task_type == "dialysis":
                demo_models_list = [
                    {"model_type": "random_forest", "display_name": "Random Forest",
                     "probability": min(max(base_prob + rng.uniform(-0.08, 0.08), 0.01), 0.99),
                     "risk_level": "moderate", "metrics": {"auc_roc": 0.82, "accuracy": 0.77, "sensitivity": 0.79}},
                    {"model_type": "naive_bayes", "display_name": "Naive Bayes",
                     "probability": min(max(base_prob + rng.uniform(-0.08, 0.08), 0.01), 0.99),
                     "risk_level": "moderate", "metrics": {"auc_roc": 0.75, "accuracy": 0.72, "sensitivity": 0.70}},
                    {"model_type": "xgboost", "display_name": "XGBoost",
                     "probability": min(max(base_prob + rng.uniform(-0.08, 0.08), 0.01), 0.99),
                     "risk_level": "moderate", "metrics": {"auc_roc": 0.83, "accuracy": 0.78, "sensitivity": 0.76}},
                    {"model_type": "stacking_ensemble", "display_name": "Stacking Ensemble",
                     "probability": min(max(base_prob + rng.uniform(-0.08, 0.08), 0.01), 0.99),
                     "risk_level": "moderate", "metrics": {"auc_roc": 0.85, "accuracy": 0.80, "sensitivity": 0.82}},
                ]
            else:
                demo_models_list = [
                    {"model_type": "random_forest", "display_name": "Random Forest",
                     "probability": min(max(base_prob + rng.uniform(-0.08, 0.08), 0.01), 0.99),
                     "risk_level": "moderate", "metrics": {"auc_roc": 0.82, "accuracy": 0.77, "sensitivity": 0.79}},
                    {"model_type": "naive_bayes", "display_name": "Naive Bayes",
                     "probability": min(max(base_prob + rng.uniform(-0.08, 0.08), 0.01), 0.99),
                     "risk_level": "moderate", "metrics": {"auc_roc": 0.75, "accuracy": 0.72, "sensitivity": 0.70}},
                    {"model_type": "calibrated_svm", "display_name": "Calibrated SVM",
                     "probability": min(max(base_prob + rng.uniform(-0.08, 0.08), 0.01), 0.99),
                     "risk_level": "moderate", "metrics": {"auc_roc": 0.80, "accuracy": 0.76, "sensitivity": 0.74}},
                    {"model_type": "xgboost", "display_name": "XGBoost",
                     "probability": min(max(base_prob + rng.uniform(-0.08, 0.08), 0.01), 0.99),
                     "risk_level": "moderate", "metrics": {"auc_roc": 0.83, "accuracy": 0.78, "sensitivity": 0.76}},
                    {"model_type": "stacking_ensemble", "display_name": "Stacking Ensemble",
                     "probability": min(max(base_prob + rng.uniform(-0.08, 0.08), 0.01), 0.99),
                     "risk_level": "moderate", "metrics": {"auc_roc": 0.85, "accuracy": 0.80, "sensitivity": 0.82}},
                ]
            for m in demo_models_list:
                p = m["probability"]
                m["risk_level"] = "low" if p < 0.3 else ("high" if p >= 0.7 else "moderate")
            probs = [m["probability"] for m in demo_models_list]
            aucs = [m["metrics"]["auc_roc"] for m in demo_models_list]
            consensus_p = float(np.average(probs, weights=aucs))
            risk_levels = [m["risk_level"] for m in demo_models_list]
            most_common_rl = max(set(risk_levels), key=risk_levels.count)
            multi_result = {
                "results": demo_models_list,
                "consensus_probability": consensus_p,
                "consensus_risk_level": "low" if consensus_p < 0.3 else ("high" if consensus_p >= 0.7 else "moderate"),
                "agreement_score": risk_levels.count(most_common_rl) / len(risk_levels),
            }

        # Wyciągnij consensus jako prediction (używane dalej przez XAI taby i PDF)
        consensus_prob = multi_result.get("consensus_probability", 0)
        consensus_rl = multi_result.get("consensus_risk_level", "moderate")
        prediction = {
            "probability": consensus_prob,
            "risk_level": consensus_rl,
            "prediction": 1 if consensus_prob > 0.5 else 0,
            "model_used": "Consensus (wszystkie modele)",
        }
        results_list = multi_result.get("results", [])
        agreement = multi_result.get("agreement_score", 0)

        if prediction_task == "Śmiertelność":
            explanation = get_demo_explanation(patient_data)
        else:
            explanation = get_demo_dialysis_explanation(patient_data)

    st.session_state['_chat_prediction'] = prediction
    st.session_state['_chat_explanation'] = explanation

    if _demo_mode:
        st.warning("Wyniki poniżej są symulowane (tryb demo). Nie reprezentują rzeczywistej analizy modelu dla tego pacjenta.")

    # Tytuł i opisy w zależności od celu predykcji
    if prediction_task == "Śmiertelność":
        gauge_title = "Ryzyko zgonu"
        low_desc = "Niskie ryzyko zgonu. Wskaźniki w normie."
        moderate_desc = "Umiarkowane ryzyko zgonu. Zalecana zwiększona obserwacja."
        high_desc = "Wysokie ryzyko zgonu. Wymaga szczególnej uwagi."
    else:
        gauge_title = "Potrzeba dializy"
        low_desc = "Niskie prawdopodobieństwo potrzeby dializy."
        moderate_desc = "Umiarkowane ryzyko potrzeby dializy. Wskazane monitorowanie czynności nerek."
        high_desc = "Wysokie prawdopodobieństwo potrzeby dializy. Zalecana pilna ocena nefrologiczna."

    outcome_label_mm = "zgonu" if _xai_task_type == "mortality" else "potrzeby dializy"

    # ========== GŁÓWNA SEKCJA: WYNIKI MULTI-MODEL ==========
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown('<h3 class="section-header">Wynik analizy</h3>', unsafe_allow_html=True)
        st.caption("Consensus (wszystkie modele)")

        # Consensus gauge
        fig_gauge = create_gauge_chart(prediction["probability"], gauge_title)
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Risk level badge
        risk_level = prediction["risk_level"]
        if risk_level == "low":
            st.markdown(f'<div class="risk-low"><strong>Niskie ryzyko</strong><br>{low_desc}</div>', unsafe_allow_html=True)
        elif risk_level == "moderate":
            st.markdown(f'<div class="risk-moderate"><strong>Umiarkowane ryzyko</strong><br>{moderate_desc}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="risk-high"><strong>Wysokie ryzyko</strong><br>{high_desc}</div>', unsafe_allow_html=True)

        # Dodatkowa informacja dla dializy
        if prediction_task == "Potrzeba dializy" and prediction.get("needs_dialysis") is not None:
            if prediction["needs_dialysis"]:
                st.warning("Model wskazuje na potrzebę dializy.")
            else:
                st.success("Model nie wskazuje na potrzebę dializy.")

        # Zgodność modeli
        n_models = len(results_list)
        n_agree = int(agreement * n_models) if n_models > 0 else 0
        rl_pl = {"low": "Niskie", "moderate": "Umiarkowane", "high": "Wysokie"}.get(consensus_rl, consensus_rl)
        st.metric("Zgodność modeli", f"{n_agree}/{n_models}")
        st.metric("Agreement", f"{agreement:.0%}")

    with col2:
        # Tabela modeli
        if results_list:
            st.markdown('<h3 class="section-header">Porównanie modeli</h3>', unsafe_allow_html=True)
            table_data = []
            for r in sorted(results_list, key=lambda x: (x.get("metrics") or {}).get("auc_roc", 0), reverse=True):
                metrics = r.get("metrics", {}) or {}
                rl = r.get("risk_level", "moderate")
                rl_pl_t = {"low": "Niskie", "moderate": "Umiarkowane", "high": "Wysokie"}.get(rl, rl)
                table_data.append({
                    "Model": r.get("display_name", r.get("model_type", "")),
                    "AUC-ROC": metrics.get("auc_roc", "?"),
                    "Accuracy": metrics.get("accuracy", "?"),
                    "Sensitivity": metrics.get("sensitivity", "?"),
                    "Prawdopodobienstwo": f"{r.get('probability', 0):.1%}",
                    "Poziom ryzyka": rl_pl_t,
                })
            st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

        # Horizontal bar chart
        if results_list:
            model_names = [r.get("display_name", r.get("model_type", "")) for r in results_list]
            model_probs = [r.get("probability", 0) * 100 for r in results_list]
            model_colors = ["#28a745" if p < 30 else ("#ffc107" if p < 70 else "#dc3545") for p in model_probs]

            fig_bars = go.Figure(go.Bar(
                y=model_names, x=model_probs, orientation='h',
                marker_color=model_colors,
                text=[f"{p:.1f}%" for p in model_probs],
                textposition="outside",
                textfont=dict(size=13, color='#ffffff'),
                showlegend=False,
                hovertemplate='<b>%{y}</b><br>Prawdopodobieństwo: %{x:.1f}%<extra></extra>'
            ))
            fig_bars.add_trace(go.Scatter(
                x=[30, 30], y=[model_names[0], model_names[-1]],
                mode='lines', line=dict(dash='dash', color='#28a745', width=2),
                name='Próg niskie/umiarkowane (30%)', showlegend=True
            ))
            fig_bars.add_trace(go.Scatter(
                x=[70, 70], y=[model_names[0], model_names[-1]],
                mode='lines', line=dict(dash='dash', color='#dc3545', width=2),
                name='Próg umiarkowane/wysokie (70%)', showlegend=True
            ))
            fig_bars.add_trace(go.Scatter(
                x=[None], y=[None], mode='markers',
                marker=dict(size=10, color='#28a745', symbol='square'),
                name='Niskie ryzyko (<30%)', showlegend=True
            ))
            fig_bars.add_trace(go.Scatter(
                x=[None], y=[None], mode='markers',
                marker=dict(size=10, color='#ffc107', symbol='square'),
                name='Umiarkowane ryzyko (30-70%)', showlegend=True
            ))
            fig_bars.add_trace(go.Scatter(
                x=[None], y=[None], mode='markers',
                marker=dict(size=10, color='#dc3545', symbol='square'),
                name='Wysokie ryzyko (>70%)', showlegend=True
            ))
            fig_bars.update_layout(
                title=dict(text=f"Prawdopodobieństwo {outcome_label_mm} per model", font=dict(size=18, color='#ffffff')),
                xaxis=dict(
                    title=dict(text=f"Prawdopodobieństwo {outcome_label_mm} (%)", font=dict(size=12, color='#ffffff')),
                    range=[0, 105], tickfont=dict(color='#ffffff'), gridcolor='#444444'
                ),
                yaxis=dict(
                    title=dict(text="Model ML", font=dict(size=12, color='#ffffff')),
                    tickfont=dict(color='#ffffff')
                ),
                height=350, margin=dict(l=20, r=80, t=50, b=40),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', template='plotly_dark',
                showlegend=True,
                legend=dict(
                    orientation='h', yanchor='top', y=-0.15, xanchor='center', x=0.5,
                    font=dict(color='#ffffff', size=10)
                )
            )
            st.plotly_chart(fig_bars, use_container_width=True)

        # Kluczowe czynniki
        st.markdown('<h3 class="section-header">Kluczowe czynniki</h3>', unsafe_allow_html=True)
        all_factors = explanation["risk_factors"] + explanation["protective_factors"]
        all_factors_sorted = sorted(all_factors, key=lambda x: abs(x["contribution"]), reverse=True)
        for factor in all_factors_sorted[:5]:
            if factor["contribution"] > 0:
                arrow = "↑"
                direction = "zwiększa"
            else:
                arrow = "↓"
                direction = "zmniejsza"
            st.markdown(f'{arrow} **{factor["feature"]}** - {direction} ryzyko ({factor["contribution"]:+.3f})')

    # Zakładki z szczegółami
    st.markdown("---")
    tab_shap, tab_lime, tab_dalex, tab_ebm, tab_compare, tab_calib = st.tabs([
        "SHAP", "LIME", "DALEX", "EBM", "Porównanie XAI", "Kalibracja"
    ])

    # XAI extra params (task_type already set above)
    _xai_extra = {"task_type": _xai_task_type, "dialysis_patient": patient_data} if _xai_task_type == "dialysis" else {"task_type": "mortality"}

    # ========== TAB: SHAP ==========
    with tab_shap:
        st.subheader("SHAP - Wartości Shapleya")
        st.markdown("""
        **SHAP** (SHapley Additive exPlanations) rozkłada predykcje modelu na wkłady poszczególnych cech
        korzystając z teorii gier (wartości Shapleya).

        **Jak czytać wykresy:**
        - **Waterfall (lokalny):** Każdy słupek pokazuje wpływ jednej cechy na predykcje dla tego pacjenta.
          Czerwony = zwiększa ryzyko, zielony = zmniejsza ryzyko. Wartość liczbowa = wielkość wpływu.
        - **Beeswarm (globalny):** Każda kropka = jeden pacjent. Oś X = wartość SHAP (wpływ na predykcje).
          Kolor = wartość cechy (czerwony = wysoka, niebieski = niska). Cechy posortowane wg średniego |SHAP|.
        - **Słupkowy (globalny):** Średnia bezwzględna wartość SHAP dla każdej cechy — im wyższa, tym ważniejsza cecha globalnie.
        """)

        with st.spinner("Ładowanie wyjaśnień SHAP..."):
            shap_result = call_api_cached("/explain/shap", "POST", {"patient": patient_data, "method": "shap", **_xai_extra})
        if shap_result and shap_result.get("feature_contributions"):
            st.session_state["shap_api_result"] = shap_result
            shap_factors = shap_result["feature_contributions"]
            shap_factors_sorted = sorted(shap_factors, key=lambda x: abs(x.get("contribution", 0)), reverse=True)
            fig_waterfall = create_waterfall_chart(shap_factors_sorted[:10], "Wpływ czynników (SHAP)")
            if fig_waterfall:
                st.plotly_chart(fig_waterfall, use_container_width=True)

            if shap_result.get("base_value"):
                st.caption(f"Wartość bazowa modelu: {shap_result['base_value']:.4f}")
        else:
            all_factors_for_chart = explanation["risk_factors"] + explanation["protective_factors"]
            fig_waterfall = create_waterfall_chart(all_factors_for_chart, "Wpływ czynników (SHAP) — demo")
            if fig_waterfall:
                st.plotly_chart(fig_waterfall, use_container_width=True)
            st.warning("⚠️ Wyniki poniżej są symulowane (tryb demo). Nie reprezentują rzeczywistej analizy modelu dla tego pacjenta.")

        # Analiza globalna SHAP
        with st.expander("Analiza globalna SHAP", expanded=False):
            with st.spinner("Ładowanie globalnej analizy SHAP..."):
                shap_global = call_api_cached(f"/explain/shap/global?task_type={_xai_task_type}", "GET")
            if shap_global:
                col_bee, col_bar = st.columns(2)
                with col_bar:
                    fig_gimp = create_global_importance_bar(
                        shap_global.get("feature_importance", {}),
                        "Średnia |SHAP| per cecha"
                    )
                    st.plotly_chart(fig_gimp, use_container_width=True)
                with col_bee:
                    shap_matrix = shap_global.get("shap_values_matrix", [])
                    feat_matrix = shap_global.get("feature_values_matrix", [])
                    feat_names = shap_global.get("feature_names", [])
                    if shap_matrix and feat_matrix:
                        fig_bee = create_beeswarm_chart(shap_matrix, feat_matrix, feat_names)
                        st.plotly_chart(fig_bee, use_container_width=True)
                st.caption(f"Analiza na {shap_global.get('n_samples', '?')} próbkach")
            else:
                st.info("Globalna analiza SHAP niedostępna. Uruchom API z artefaktami XAI.")

    # ========== TAB: LIME ==========
    with tab_lime:
        st.subheader("LIME - Lokalne wyjasnienie")
        _lime_outcome = "zgonu" if _xai_task_type == "mortality" else "potrzeby dializy"
        st.markdown(f"""
        **LIME** (Local Interpretable Model-agnostic Explanations) buduje lokalny model zastępczy
        (regresja liniowa) wokół konkretnego pacjenta, aby wyjaśnić predykcje modelu.

        **Jak czytać wykres:**
        - Oś X = **bezwzględna waga cechy** w modelu lokalnym — im dłuższy słupek, tym większy wpływ cechy.
        - Kolor: **czerwony** = cecha zwiększa prawdopodobieństwo {_lime_outcome}, **zielony** = zmniejsza.
        - Cechy posortowane od najbardziej wpływowej (góra) do najmniej (dół).
        """)

        with st.spinner("Ładowanie wyjaśnień LIME..."):
            lime_result = call_api_cached("/explain/lime", "POST", {"patient": patient_data, "method": "lime", **_xai_extra})
        if lime_result and lime_result.get("feature_weights"):
            lime_weights = lime_result["feature_weights"]
            # Konwertuj do formatu wykresu
            lime_for_chart = [
                {"feature": w.get("feature", ""), "contribution": w.get("weight", 0)}
                for w in lime_weights
            ]
            lime_for_chart.sort(key=lambda x: abs(x["contribution"]), reverse=True)
            fig_bar = create_bar_chart(lime_for_chart[:10], "Waznosc czynnikow (LIME)")
            if fig_bar:
                st.plotly_chart(fig_bar, use_container_width=True)

            if lime_result.get("local_prediction") is not None:
                st.caption(f"Lokalna predykcja LIME: {lime_result['local_prediction']:.4f}")
        else:
            all_factors_for_chart = explanation["risk_factors"] + explanation["protective_factors"]
            fig_bar = create_bar_chart(all_factors_for_chart, "Waznosc czynnikow (LIME) — demo")
            if fig_bar:
                st.plotly_chart(fig_bar, use_container_width=True)
            st.warning("⚠️ Wyniki poniżej są symulowane (tryb demo). Nie reprezentują rzeczywistej analizy modelu dla tego pacjenta.")

    # ========== TAB: DALEX ==========
    with tab_dalex:
        st.subheader("DALEX — Break Down + analiza globalna")
        st.markdown("""
        **DALEX** to framework do wyjaśniania modeli ML oferujący analizę na poziomie instancji i globalnym.

        **Metody:**
        - **Break Down** — dekompozycja predykcji na addytywne kontribucje poszczególnych cech (analogicznie do SHAP).
        - **Permutation Variable Importance** — mierzy spadek jakości modelu (AUC) po losowej permutacji wartości cechy.
          Im większy spadek, tym ważniejsza cecha.
        - **PDP** (Partial Dependence Profile) — pokazuje jak zmiana jednej cechy wpływa na średnią predykcje modelu,
          przy stałych pozostałych cechach (ceteris paribus).
        """)

        # Instance-level Break Down
        st.markdown("### Break Down (instancja)")
        with st.spinner("Ładowanie Break Down (DALEX)..."):
            dalex_result = call_api_cached("/explain/dalex", "POST", {"patient": patient_data, "method": "dalex", **_xai_extra})
        if dalex_result and dalex_result.get("contributions"):
            dalex_contribs = dalex_result["contributions"]
            dalex_for_chart = [
                {"feature": c.get("feature", ""), "contribution": c.get("contribution", 0)}
                for c in dalex_contribs if c.get("feature")
            ]
            dalex_for_chart.sort(key=lambda x: abs(x["contribution"]), reverse=True)
            fig_dalex = create_waterfall_chart(dalex_for_chart[:10], "Break Down (DALEX)")
            if fig_dalex:
                st.plotly_chart(fig_dalex, use_container_width=True)
        else:
            st.info("DALEX Break Down: dane niedostępne. Uruchom API z artefaktami XAI.")

        st.markdown("---")

        # Global: Permutation Variable Importance
        st.markdown("### Permutation Variable Importance")
        with st.spinner("Ładowanie Variable Importance..."):
            dalex_vi = call_api_cached(f"/explain/dalex/variable-importance?task_type={_xai_task_type}", "GET")
        if dalex_vi and dalex_vi.get("feature_importance"):
            fig_vi = create_variable_importance_chart(dalex_vi["feature_importance"])
            st.plotly_chart(fig_vi, use_container_width=True)
        else:
            st.info("Variable importance niedostępne.")

        # Global: Partial Dependence Profile
        st.markdown("### Partial Dependence Profile (PDP)")
        model_info = call_api_cached(f"/model/info?task_type={_xai_task_type}", "GET")
        pdp_features = model_info.get("feature_names", []) if model_info else []
        if not pdp_features:
            pdp_features = ["Wiek_rozpoznania", "Kreatynina", "Liczba_Zajetych_Narzadow", "Zaostrz_Wymagajace_OIT"]

        pdp_feature = st.selectbox("Wybierz cechę dla PDP:", options=pdp_features, key="pdp_feature")
        if pdp_feature:
            with st.spinner("Ładowanie PDP..."):
                pdp_result = call_api_cached(f"/explain/dalex/pdp/{pdp_feature}?task_type={_xai_task_type}", "GET")
            if pdp_result and pdp_result.get("x_values"):
                fig_pdp = create_pdp_chart(pdp_result["x_values"], pdp_result["y_values"], pdp_feature)
                st.plotly_chart(fig_pdp, use_container_width=True)
            else:
                st.info(f"PDP dla cechy {pdp_feature} niedostępny.")

    # ========== TAB: EBM ==========
    with tab_ebm:
        st.subheader("EBM — Explainable Boosting Machine")
        st.markdown("""
        **EBM** (Explainable Boosting Machine) to model inherently interpretable — każda cecha ma jawną funkcję kształtu.
        Łączy dokładność gradient boosting z interpretowalnością uogólnionych modeli addytywnych (GAM).

        **Wykresy:**
        - **Globalna ważność cech** — średni bezwzględny wpływ każdej cechy na predykcje (w skali log-odds).
        - **Lokalne kontribucje** — rozkład predykcji na wkład poszczególnych cech dla tego pacjenta.
          Czerwony = zwiększa ryzyko, zielony = zmniejsza.
        - **Funkcja kształtu** (shape function) — jawna zależność score od wartości cechy.
          Score > 0: cecha zwiększa prawdopodobieństwo zdarzenia. Score < 0: zmniejsza.
          Linia bazowa (score = 0) oznacza neutralny wpływ.
        """)

        # Globalna ważność cech
        st.markdown("### Globalna ważność cech")
        with st.spinner("Ładowanie EBM..."):
            ebm_global = call_api_cached(f"/explain/ebm/global?task_type={_xai_task_type}", "GET")
        if ebm_global and ebm_global.get("feature_importance"):
            fig_ebm_gi = create_global_importance_bar(
                ebm_global["feature_importance"], "EBM — ważność cech"
            )
            st.plotly_chart(fig_ebm_gi, use_container_width=True)

            if ebm_global.get("interactions_detected"):
                st.caption(f"Wykryte interakcje: {', '.join(ebm_global['interactions_detected'][:5])}")

        st.markdown("---")

        # Lokalne wyjaśnienie
        st.markdown("### Lokalne wyjaśnienie (ten pacjent)")
        with st.spinner("Ładowanie EBM..."):
            ebm_local = call_api_cached("/explain/ebm/local", "POST", {"patient": patient_data, "method": "ebm", **_xai_extra})
        if ebm_local and ebm_local.get("contributions"):
            fig_ebm_local = create_ebm_local_chart(ebm_local["contributions"])
            if fig_ebm_local:
                st.plotly_chart(fig_ebm_local, use_container_width=True)
            st.caption(f"Prawdopodobieństwo (EBM): {ebm_local.get('probability_positive', 0):.1%}")
        else:
            st.info("Lokalne wyjaśnienie EBM niedostępne.")

        st.markdown("---")

        # Funkcja kształtu cechy
        st.markdown("### Funkcja kształtu cechy (Shape Function)")
        ebm_features = ebm_global.get("feature_names", []) if ebm_global else []
        if not ebm_features:
            ebm_features = pdp_features if pdp_features else ["Wiek_rozpoznania", "Kreatynina"]

        ff_feature = st.selectbox("Wybierz cechę:", options=ebm_features, key="ff_feature")
        if ff_feature:
            with st.spinner("Ładowanie Shape Function..."):
                ff_result = call_api_cached(f"/explain/ebm/feature-function/{ff_feature}?task_type={_xai_task_type}", "GET")
            if ff_result and ff_result.get("scores"):
                fig_ff = create_feature_function_chart(ff_result["names"], ff_result["scores"], ff_feature)
                st.plotly_chart(fig_ff, use_container_width=True)
            else:
                st.info(f"Shape function dla cechy {ff_feature} niedostępna.")

    # ========== TAB: Porownanie XAI ==========
    with tab_compare:
        st.subheader("Porównanie metod XAI")
        st.markdown("""
        Porównanie rankingów cech z różnych metod XAI pozwala ocenić **wiarygodność wyjaśnień**.
        Gdy różne metody wskazują te same cechy jako najważniejsze, możemy mieć większą pewność co do interpretacji.

        **Metryki:**
        - **Jaccard Agreement** — podobieństwo zbiorów top-5 cech między metodami (0% = brak wspólnych, 100% = identyczne).
        - **Macierz zgodności** — pairwise Jaccard similarity między każdą parą metod.
        - **Heatmap rankingów** — pozycja cechy w rankingu każdej metody (ciemniejszy = wyższa pozycja).
        """)

        # Instance-level comparison (existing)
        st.markdown("### Porównanie rankingow (instancja)")
        with st.spinner("Ładowanie porównania XAI..."):
            comparison_result = call_api_cached("/explain/comparison", "POST", {"patient": patient_data, "method": "shap", **_xai_extra})
        if comparison_result:
            rankings = comparison_result.get("individual_rankings", {})
            methods = list(rankings.keys())
            common_top = comparison_result.get("common_top_features", [])
            agreement_score = comparison_result.get("ranking_agreement", 0)

            if methods:
                max_len = max(len(rankings[m]) for m in methods) if methods else 0
                ranking_table = {"Pozycja": list(range(1, min(max_len + 1, 11)))}
                for m in methods:
                    r = rankings[m]
                    ranking_table[m] = [r[i] if i < len(r) else "" for i in range(min(max_len, 10))]
                st.dataframe(pd.DataFrame(ranking_table), use_container_width=True, hide_index=True)

            col_c1, col_c2 = st.columns(2)
            with col_c1:
                st.metric("Jaccard Agreement (top-5)", f"{agreement_score:.0%}")
            with col_c2:
                st.metric("Wspólne top cechy", ", ".join(common_top[:5]) if common_top else "brak")

            if agreement_score > 0.6:
                st.success("Wysoka zgodność rankingów — wyjaśnienia są wiarygodne.")
            elif agreement_score > 0.3:
                st.warning("Umiarkowana zgodność rankingów między metodami.")
            else:
                st.error("Niska zgodność — różne metody wskazują różne czynniki.")
        else:
            all_factors_sorted_chart = sorted(
                explanation["risk_factors"] + explanation["protective_factors"],
                key=lambda x: abs(x["contribution"]), reverse=True
            )
            col_comp1, col_comp2 = st.columns(2)
            with col_comp1:
                st.markdown("**Ranking SHAP:**")
                for i, f in enumerate(all_factors_sorted_chart[:5], 1):
                    st.write(f"{i}. {f['feature']}")
            with col_comp2:
                st.markdown("**Ranking LIME:**")
                for i, f in enumerate(all_factors_sorted_chart[:5], 1):
                    st.write(f"{i}. {f['feature']}")
            st.info("Wysoka zgodność rankingów między metodami zwiększa wiarygodność wyjaśnień.")

        # Globalna analiza porównawcza
        with st.expander("Globalna analiza porównawcza", expanded=False):
            with st.spinner("Ładowanie globalnego porównania..."):
                global_comp = call_api_cached(f"/explain/comparison/global?task_type={_xai_task_type}", "GET")
            if global_comp:
                comp_rankings = global_comp.get("rankings", {})
                comp_methods = global_comp.get("methods_compared", [])
                agreement_mat = global_comp.get("agreement_matrix", {})
                mean_agr = global_comp.get("mean_agreement", 0)

                col_rh, col_ah = st.columns(2)
                with col_rh:
                    if comp_rankings:
                        fig_rh = create_ranking_heatmap(comp_rankings)
                        st.plotly_chart(fig_rh, use_container_width=True)
                with col_ah:
                    if agreement_mat and comp_methods:
                        fig_ah = create_heatmap_chart(agreement_mat, comp_methods, "Jaccard Similarity")
                        st.plotly_chart(fig_ah, use_container_width=True)

                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    st.metric("Średnia zgodność (Jaccard)", f"{mean_agr:.0%}")
                with col_m2:
                    common_global = global_comp.get("common_top_features", [])
                    st.metric("Wspólne cechy (top-5, wszystkie metody)", ", ".join(common_global[:5]) if common_global else "brak")
            else:
                st.info("Globalna analiza porównawcza niedostępna. Uruchom API z artefaktami XAI.")

    # ========== TAB: Kalibracja ==========
    with tab_calib:
        st.subheader("Kalibracja modeli — krzywa niezawodności")
        st.markdown("""
        **Kalibracja** ocenia, czy prawdopodobieństwa wyznaczone przez model odpowiadają rzeczywistej częstości zdarzeń.
        Idealnie skalibrowany model leży na przekątnej (linia przerywana).

        - **Brier Score** — średni błąd kwadratowy prawdopodobieństw (0 = idealny, 1 = najgorszy). Niższy = lepszy.
        - **Krzywa niezawodności (reliability diagram)** — oś X: przewidywane prawdopodobieństwo, oś Y: obserwowana częstość.
          Krzywa powyżej przekątnej = model niedocenia ryzyka; poniżej = przecenia ryzyko.
        """)

        with st.spinner("Ładowanie danych kalibracji..."):
            calib_data = call_api_cached(f"/models/calibration?task_type={_xai_task_type}", "GET")
        if calib_data and calib_data.get("models"):
            calib_models = calib_data["models"]

            # Brier score metrics
            brier_cols = st.columns(min(len(calib_models), 5))
            for idx, m in enumerate(calib_models):
                with brier_cols[idx % len(brier_cols)]:
                    bs = m.get("brier_score", 0)
                    label = "demo" if m.get("is_demo") else ""
                    st.metric(
                        label=f"{m.get('display_name', m['model_type'])} {label}",
                        value=f"{bs:.4f}"
                    )

            st.markdown("---")

            # Reliability diagram
            fig_calib = go.Figure()
            # Perfect calibration reference line
            fig_calib.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines', name='Idealna kalibracja',
                line=dict(dash='dash', color='gray', width=1)
            ))
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            for idx, m in enumerate(calib_models):
                x_vals = m.get("calibration_curve_x", [])
                y_vals = m.get("calibration_curve_y", [])
                if x_vals and y_vals:
                    fig_calib.add_trace(go.Scatter(
                        x=x_vals, y=y_vals,
                        mode='lines+markers',
                        name=m.get("display_name", m["model_type"]),
                        line=dict(color=colors[idx % len(colors)], width=2),
                        marker=dict(size=6)
                    ))
            fig_calib.update_layout(
                title="Krzywa niezawodności (Reliability Diagram)",
                xaxis_title="Przewidywane prawdopodobieństwo",
                yaxis_title="Obserwowana częstość",
                xaxis=dict(range=[0, 1], tickformat='.0%'),
                yaxis=dict(range=[0, 1], tickformat='.0%'),
                height=450,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                template='plotly_dark',
                legend=dict(orientation='h', y=-0.2, xanchor='center', x=0.5)
            )
            st.plotly_chart(fig_calib, use_container_width=True)

            if calib_data.get("models") and calib_data["models"][0].get("is_demo"):
                st.warning("⚠️ Dane kalibracji są symulowane (brak danych testowych w API). Uruchom API z danymi testowymi.")
            st.caption(f"Liczba próbek: {calib_models[0].get('n_samples', '?')} | Zadanie: {calib_data.get('task_type', _xai_task_type)}")
        else:
            st.info("Dane kalibracji niedostępne. Uruchom API z wytrenowanymi modelami i danymi testowymi.")

    # ========== PDF EXPORT ==========
    st.markdown("---")
    st.markdown("### Eksport raportu PDF")
    if st.button("Generuj raport PDF", key="pdf_export_btn", use_container_width=False):
        try:
            from fpdf import FPDF
            pdf = FPDF()
            pdf.add_page()
            pdf.set_auto_page_break(auto=True, margin=15)

            # Dodaj font z obsługą polskich znaków (DejaVu Sans)
            _font_name = "Helvetica"  # fallback
            try:
                import subprocess
                # Szukaj DejaVu Sans na systemie
                font_paths = [
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                    "/usr/share/fonts/TTF/DejaVuSans.ttf",
                    "/Library/Fonts/DejaVuSans.ttf",
                    os.path.expanduser("~/.fonts/DejaVuSans.ttf"),
                ]
                # Na macOS sprawdź też matplotlib
                try:
                    import matplotlib
                    mpl_data = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
                    font_paths.append(os.path.join(mpl_data, "DejaVuSans.ttf"))
                except ImportError:
                    pass

                for fp in font_paths:
                    if os.path.exists(fp):
                        pdf.add_font("DejaVu", "", fp, uni=True)
                        pdf.add_font("DejaVu", "B", fp.replace("DejaVuSans.ttf", "DejaVuSans-Bold.ttf"), uni=True)
                        _font_name = "DejaVu"
                        break
            except Exception:
                pass  # Fallback do Helvetica

            # Title
            pdf.set_font(_font_name, "B", 16)
            task_label = "Śmiertelność" if prediction_task == "Śmiertelność" else "Potrzeba dializy"
            pdf.cell(0, 10, f"Raport XAI - {task_label}", ln=True, align="C")
            pdf.ln(5)

            # Prediction result
            pdf.set_font(_font_name, "B", 12)
            pdf.cell(0, 8, "Wynik predykcji", ln=True)
            pdf.set_font(_font_name, "", 11)
            prob_pct = prediction.get("probability", 0) * 100
            risk_lv = prediction.get("risk_level", "N/A")
            risk_map = {"low": "Niskie", "moderate": "Umiarkowane", "high": "Wysokie"}
            pdf.cell(0, 7, f"Prawdopodobieństwo: {prob_pct:.1f}%", ln=True)
            pdf.cell(0, 7, f"Poziom ryzyka: {risk_map.get(risk_lv, risk_lv)}", ln=True)
            if prediction_task == "Potrzeba dializy":
                needs_dialysis = prediction.get("needs_dialysis")
                if needs_dialysis is not None:
                    dialysis_label = "Tak" if needs_dialysis else "Nie"
                    pdf.cell(0, 7, f"Potrzeba dializy: {dialysis_label}", ln=True)
            pdf.ln(4)

            # Patient data (key fields)
            pdf.set_font(_font_name, "B", 12)
            pdf.cell(0, 8, "Dane pacjenta", ln=True)
            pdf.set_font(_font_name, "", 10)
            for k, v in list(patient_data.items())[:15]:
                pdf.cell(0, 6, f"  {k}: {v}", ln=True)
            pdf.ln(4)

            # Risk factors — use real SHAP result if available
            shap_api = st.session_state.get("shap_api_result")
            if shap_api and shap_api.get("feature_contributions"):
                all_factors_pdf = [{"feature": f["feature"], "contribution": f["contribution"]}
                                   for f in shap_api["feature_contributions"]]
            else:
                all_factors_pdf = explanation["risk_factors"] + explanation["protective_factors"]
            all_factors_pdf_sorted = sorted(all_factors_pdf, key=lambda x: abs(x["contribution"]), reverse=True)
            pdf.set_font(_font_name, "B", 12)
            pdf.cell(0, 8, "Główne czynniki ryzyka (top 10)", ln=True)
            pdf.set_font(_font_name, "", 10)
            for f in all_factors_pdf_sorted[:10]:
                direction_pl = "zwiększa ryzyko" if f["contribution"] > 0 else "zmniejsza ryzyko"
                pdf.cell(0, 6, f"  {f['feature']}: {f['contribution']:+.4f} ({direction_pl})", ln=True)
            pdf.ln(4)

            # Disclaimer
            pdf.set_font(_font_name, "", 9)
            pdf.multi_cell(0, 5, "OSTRZEŻENIE: Ten raport ma charakter informacyjny i pomocniczy. "
                           "Nie zastępuje diagnozy lekarskiej. Wszystkie decyzje kliniczne powinny być "
                           "podejmowane przez upoważnionych pracowników medycznych.")

            pdf_bytes = bytes(pdf.output())
            st.download_button(
                label="Pobierz raport PDF",
                data=pdf_bytes,
                file_name=f"raport_xai_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                use_container_width=False
            )
        except ImportError:
            st.error("Biblioteka fpdf2 nie jest zainstalowana. Uruchom: pip install fpdf2")
        except Exception as e:
            st.error(f"Błąd generowania PDF: {e}")

elif analysis_mode == "Pojedynczy pacjent":
    # Strona powitalna dla trybu pojedynczego pacjenta
    st.markdown("""
    <h2 style="color: #1a5276;">Witaj w systemie XAI!</h2>
    <p style="font-size: 1.1rem; color: #495057;">Ten system pomoże Ci:</p>
    <ul style="list-style: none; padding-left: 0; font-size: 1rem;">
        <li style="margin: 0.5rem 0;"><strong>Ocenić ryzyko</strong> - na podstawie danych klinicznych</li>
        <li style="margin: 0.5rem 0;"><strong>Zrozumieć przyczyny</strong> - dzięki wyjaśnieniom XAI</li>
        <li style="margin: 0.5rem 0;"><strong>Porozmawiać</strong> - z asystentem AI o wynikach</li>
    </ul>
    <h3 style="color: #1a5276; margin-top: 1.5rem;">Jak zacząć?</h3>
    <ol style="font-size: 1rem; color: #495057;">
        <li>Wprowadź dane pacjenta w panelu bocznym</li>
        <li>Kliknij przycisk <strong>Analizuj</strong></li>
        <li>Przeglądaj wyniki i wyjaśnienia</li>
    </ol>
    <hr style="margin-top: 1.5rem;">
    """, unsafe_allow_html=True)

    # Informacje o systemie
    col_info1, col_info2, col_info3 = st.columns(3)

    with col_info1:
        _welcome_models_data = available_models_data
        if _welcome_models_data and _welcome_models_data.get("models"):
            _model_items = "".join(
                f"<li>{m.get('display_name', m.get('model_type', ''))}</li>"
                for m in _welcome_models_data["models"]
            )
        else:
            _model_items = (
                "<li>XGBoost</li><li>Random Forest</li>"
                "<li>Calibrated SVM</li><li>Naive Bayes</li><li>Stacking Ensemble</li>"
            )
        st.markdown(f"""
        <div class="info-card">
            <h3>Modele ML</h3>
            <ul style="margin: 0; padding-left: 1.2rem;">
                {_model_items}
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_info2:
        st.markdown("""
        <div class="info-card">
            <h3>Metody XAI</h3>
            <ul style="margin: 0; padding-left: 1.2rem;">
                <li>SHAP</li>
                <li>LIME</li>
                <li>DALEX</li>
                <li>EBM</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_info3:
        st.markdown("""
        <div class="info-card">
            <h3>Metryki</h3>
            <ul style="margin: 0; padding-left: 1.2rem;">
                <li>AUC-ROC > 0.85</li>
                <li>Sensitivity > 0.80</li>
                <li>Specificity > 0.75</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# CHAT AI - ZAWSZE DOSTEPNY W TRYBIE POJEDYNCZEGO PACJENTA
# ============================================================================

if analysis_mode == "Pojedynczy pacjent":
    st.markdown("---")

    _use_chat_analyze = (input_mode == "Chat (opis słowny)")

    if _use_chat_analyze:
        st.markdown('<h3 class="section-header">Chat AI — Opisz pacjenta tekstem</h3>', unsafe_allow_html=True)
        st.markdown("Wpisz dane pacjenta w dowolnej formie, np. *\"Pacjent 65 lat, mężczyzna, kreatynina 220, zajęcie nerek\"*. Dane zostaną wyodrębnione automatycznie i wykonana będzie analiza.")
    else:
        st.markdown('<h3 class="section-header">Chat AI — Rozmowa o pacjencie</h3>', unsafe_allow_html=True)

        _analysis_done = st.session_state.get('analyzed', False)

        if _analysis_done:
            st.markdown("Chat AI korzysta z danych predykcji i wyjasnien XAI.")
        else:
            st.info("Analiza nie zostala jeszcze przeprowadzona. Chat AI moze odpowiadac na pytania o pacjenta na podstawie wprowadzonych danych. Kliknij **Analizuj**, aby uzyskac pelne wyniki XAI.")

    # Inicjalizuj historię chatu
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Wyświetl historię
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input użytkownika
    _chat_placeholder = "Opisz pacjenta lub zadaj pytanie..." if _use_chat_analyze else "Zadaj pytanie o pacjenta..."
    if prompt := st.chat_input(_chat_placeholder):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Generowanie odpowiedzi..."):
                prior_messages = st.session_state.get("messages", [])[:-1][-20:]

                if _use_chat_analyze:
                    # Tryb chat-driven analysis via /chat/analyze
                    _acc = st.session_state.get("_accumulated_patient", {})
                    chat_data = {
                        "message": prompt,
                        "accumulated_patient": _acc,
                        "task_type": _xai_task_type,
                        "health_literacy": "basic",
                        "conversation_history": prior_messages,
                    }
                    chat_response = call_api("/chat/analyze", "POST", chat_data)
                else:
                    # Tryb klasyczny via /chat
                    chat_data = {
                        "message": prompt,
                        "patient": patient_data,
                        "health_literacy": "basic",
                        "task_type": _xai_task_type,
                        "conversation_history": prior_messages
                    }
                    if _xai_task_type == "dialysis":
                        chat_data["dialysis_patient"] = patient_data
                    chat_response = call_api("/chat", "POST", chat_data)

            if _use_chat_analyze and chat_response:
                # Przetwórz odpowiedź z /chat/analyze
                response = chat_response.get("response", "")

                # Zaktualizuj accumulated_patient
                new_acc = chat_response.get("accumulated_patient", {})
                st.session_state["_accumulated_patient"] = new_acc

                # Jeśli analiza kompletna — ustaw patient_data i analyzed
                if chat_response.get("analysis_complete"):
                    # Wypełnij patient_data z accumulated by XAI tabs działały
                    _chat_patient = dict(new_acc)
                    # Ustaw domyślne wartości dla brakujących pól
                    if _xai_task_type == "mortality":
                        _chat_patient.setdefault("opoznienie_rozpoznia", None)
                        _chat_patient.setdefault("manifestacja_pokarmowy", 0)
                        _chat_patient.setdefault("manifestacja_zajecie_csn", 0)
                        _chat_patient.setdefault("dializa", 0)
                    for _def_key in ["manifestacja_sercowo_naczyniowy", "manifestacja_nerki",
                                     "manifestacja_neurologiczny", "zaostrz_wymagajace_oit",
                                     "plazmaferezy", "powiklania_serce_pluca", "powiklania_infekcja"]:
                        _chat_patient.setdefault(_def_key, 0)
                    _chat_patient.setdefault("liczba_zajetych_narzadow", 0)
                    _chat_patient.setdefault("kreatynina", None)
                    _chat_patient.setdefault("max_crp", None)
                    _chat_patient.setdefault("sterydy_dawka_g", None)
                    _chat_patient.setdefault("czas_sterydow", None)

                    patient_data = _chat_patient
                    st.session_state["analyzed"] = True

                    # Zapisz predykcję do session_state
                    if chat_response.get("prediction"):
                        st.session_state["_chat_prediction"] = chat_response["prediction"]
                    if chat_response.get("xai_summary"):
                        st.session_state["_chat_explanation"] = chat_response["xai_summary"]

                if not response:
                    response = "Nie udało się przetworzyć odpowiedzi."

            elif chat_response and chat_response.get("response"):
                response = chat_response["response"]
            else:
                # Fallback — użyj wyników analizy jeśli dostępne
                _fb_pred = st.session_state.get('_chat_prediction')
                _fb_expl = st.session_state.get('_chat_explanation')
                if _fb_pred and _fb_expl:
                    all_factors_sorted_chat = sorted(
                        _fb_expl.get("risk_factors", []) + _fb_expl.get("protective_factors", []),
                        key=lambda x: abs(x.get("contribution", 0)), reverse=True
                    )
                    risk_map = {'low': 'Niskie', 'moderate': 'Umiarkowane', 'high': 'Wysokie'}
                    risk_level_pl = risk_map.get(_fb_pred.get('risk_level', 'low'), _fb_pred.get('risk_level', ''))
                    if any(word in prompt.lower() for word in ['wynik', 'ryzyko', 'analiza']):
                        response = f"Poziom ryzyka: **{risk_level_pl}** ({_fb_pred['probability']:.1%}).\n\n"
                        if all_factors_sorted_chat:
                            response += "Glowne czynniki:\n"
                            for f in all_factors_sorted_chat[:3]:
                                response += f"- {f['feature']} ({f['contribution']:+.3f})\n"
                    elif any(word in prompt.lower() for word in ['czynnik', 'dlaczego']):
                        response = "Glowne czynniki wplywajace na ocene:\n\n"
                        for f in all_factors_sorted_chat[:3]:
                            response += f"- **{f['feature']}**: wplyw {f['contribution']:+.3f}\n"
                    else:
                        response = "Moge pomoc Ci zrozumiec wyniki analizy, czynniki ryzyka i zalecenia.\n\nO czym chcialbys porozmawiac?"
                else:
                    response = (
                        "Przepraszam, nie udalo sie polaczyc z API. "
                        "Sprobuj ponownie lub kliknij **Analizuj**, aby uzyskac pelna analize."
                    )
                response += "\n\n*Pamietaj: to narzedzie informacyjne, skonsultuj sie z lekarzem.*"

            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

            if len(st.session_state.messages) > 40:
                st.session_state.messages = st.session_state.messages[-40:]

            _suggestions = None
            if chat_response:
                _suggestions = chat_response.get("follow_up_suggestions")
            if _suggestions:
                st.markdown("**Proponowane pytania:**")
                for suggestion in _suggestions:
                    st.caption(f"- {suggestion}")

            # Rerun aby sidebar odświeżył "Zebrane dane" z _accumulated_patient
            if _use_chat_analyze and chat_response and chat_response.get("accumulated_patient"):
                st.rerun()

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div class="disclaimer">
<strong>Ważne:</strong> Ten system jest narzędziem wspierającym decyzje kliniczne.
Nie zastępuje profesjonalnej oceny medycznej. Wszystkie decyzje dotyczące
leczenia powinny być podejmowane przez wykwalifikowany personel medyczny
w oparciu o pełny obraz kliniczny pacjenta.
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; color: #5a6268; padding: 1rem; font-size: 0.9rem;">
Vasculitis XAI System v1.0.0 | © 2024-2026
</div>
""", unsafe_allow_html=True)
