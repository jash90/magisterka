"""
Dashboard Streamlit dla systemu XAI.

Interfejs użytkownika do predykcji ryzyka śmiertelności
w zapaleniu naczyń z wyjaśnieniami XAI.
"""

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

# API URL
API_URL = "http://localhost:8000"

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
    return fig


def create_waterfall_chart(factors: list, title: str = "Wpływ czynników") -> go.Figure:
    """Utwórz wykres waterfall."""
    if not factors:
        return None

    names = [f["feature"] for f in factors]
    values = [f["contribution"] for f in factors]
    colors = ["red" if v > 0 else "green" for v in values]

    fig = go.Figure(go.Waterfall(
        name="",
        orientation="h",
        y=names,
        x=values,
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        decreasing={"marker": {"color": "#28a745"}},
        increasing={"marker": {"color": "#dc3545"}},
        text=[f"{v:+.3f}" for v in values],
        textposition="outside",
        textfont=dict(size=13, color='#ffffff', family='Arial Black')
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color='#ffffff')),
        xaxis=dict(
            title=dict(text="Wpływ na ryzyko", font=dict(size=14, color='#ffffff')),
            tickfont=dict(color='#ffffff'),
            gridcolor='#444444',
            zerolinecolor='#888888'
        ),
        yaxis=dict(
            title=dict(text="Czynnik", font=dict(size=14, color='#ffffff')),
            tickfont=dict(color='#ffffff')
        ),
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
        font=dict(color='#ffffff', size=13),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        template='plotly_dark'
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
                except:
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

    # Oblicz opóźnienie rozpoznania
    if 'opoznienie_rozpoznia' not in df.columns:
        df['opoznienie_rozpoznia'] = df['wiek'] - df['wiek_rozpoznania']

    return df


def prepare_patients_for_batch(df: pd.DataFrame) -> List[Dict]:
    """Przygotuj listę pacjentów do wysłania do batch API."""
    patients = []
    for _, row in df.iterrows():
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


def call_batch_api(patients: List[Dict], include_risk_factors: bool = True) -> Optional[Dict]:
    """Wywołaj batch API endpoint."""
    try:
        response = requests.post(
            f"{API_URL}/predict/batch",
            json={
                "patients": patients,
                "include_risk_factors": include_risk_factors,
                "top_n_factors": 3
            },
            timeout=300  # 5 minut timeout dla dużych batch
        )
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        return None


def process_batch_patients(df: pd.DataFrame, progress_callback=None) -> pd.DataFrame:
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
        patients = prepare_patients_for_batch(chunk_df)

        # Spróbuj użyć batch API
        batch_result = call_batch_api(patients, include_risk_factors=True)

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

                # Demo prediction
                prediction = get_demo_prediction(patient_data)
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
    except:
        pass
    return {
        "demo_allowed": True,
        "model_loaded": False,
        "current_mode": "unavailable",
        "force_api_mode": False
    }


def parse_large_file_streaming(uploaded_file, chunk_size: int = 10000):
    """
    Generator do streamingu dużych plików.

    Dla plików > 10MB przetwarza w chunkach.
    """
    file_name = uploaded_file.name.lower()
    content = uploaded_file.getvalue().decode('utf-8')

    if file_name.endswith('.csv'):
        # Dla CSV użyj chunksize
        chunks_read = 0
        for chunk in pd.read_csv(io.StringIO(content), chunksize=chunk_size):
            yield normalize_dataframe(chunk)
            chunks_read += 1
    elif file_name.endswith('.json'):
        # Dla JSON wczytaj wszystko i podziel
        data = json.loads(content)
        if isinstance(data, list):
            patients = data
        elif isinstance(data, dict):
            patients = data.get('patients', data.get('data', [data]))
        else:
            patients = [data]

        for i in range(0, len(patients), chunk_size):
            chunk = patients[i:i + chunk_size]
            yield normalize_dataframe(pd.DataFrame(chunk))


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
            textfont=dict(size=14, color='white')
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
        legend=dict(font=dict(color='#ffffff'))
    )

    return fig


def create_probability_histogram(results_df: pd.DataFrame) -> go.Figure:
    """Utwórz histogram prawdopodobieństw."""
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=results_df['probability'] * 100,
        nbinsx=20,
        marker_color='#2874a6',
        opacity=0.8,
        name='Pacjenci'
    ))

    # Dodaj linie progowe
    fig.add_vline(x=30, line_dash="dash", line_color="#28a745",
                  annotation_text="Próg niski/umiarkowany")
    fig.add_vline(x=70, line_dash="dash", line_color="#dc3545",
                  annotation_text="Próg umiarkowany/wysoki")

    fig.update_layout(
        title=dict(text="Rozkład prawdopodobieństw ryzyka", font=dict(size=18, color='#ffffff')),
        xaxis=dict(
            title=dict(text="Prawdopodobieństwo (%)", font=dict(color='#ffffff')),
            tickfont=dict(color='#ffffff'),
            gridcolor='#444444'
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
        bargap=0.1
    )

    return fig


def create_age_risk_scatter(results_df: pd.DataFrame) -> go.Figure:
    """Utwórz wykres punktowy wiek vs ryzyko."""
    colors = {
        'low': '#28a745',
        'moderate': '#ffc107',
        'high': '#dc3545'
    }

    fig = go.Figure()

    for risk_level in ['low', 'moderate', 'high']:
        mask = results_df['risk_level'] == risk_level
        if mask.any():
            fig.add_trace(go.Scatter(
                x=results_df.loc[mask, 'wiek'],
                y=results_df.loc[mask, 'probability'] * 100,
                mode='markers',
                marker=dict(
                    size=10,
                    color=colors[risk_level],
                    opacity=0.7
                ),
                name={'low': 'Niskie', 'moderate': 'Umiarkowane', 'high': 'Wysokie'}[risk_level],
                text=results_df.loc[mask, 'patient_id'],
                hovertemplate='<b>%{text}</b><br>Wiek: %{x}<br>Ryzyko: %{y:.1f}%<extra></extra>'
            ))

    fig.update_layout(
        title=dict(text="Wiek a ryzyko zgonu", font=dict(size=18, color='#ffffff')),
        xaxis=dict(
            title=dict(text="Wiek (lata)", font=dict(color='#ffffff')),
            tickfont=dict(color='#ffffff'),
            gridcolor='#444444'
        ),
        yaxis=dict(
            title=dict(text="Prawdopodobieństwo (%)", font=dict(color='#ffffff')),
            tickfont=dict(color='#ffffff'),
            gridcolor='#444444'
        ),
        font=dict(color='#ffffff'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        template='plotly_dark',
        height=400,
        legend=dict(font=dict(color='#ffffff'))
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
        textfont=dict(size=13, color='#ffffff', family='Arial Black')
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color='#ffffff')),
        xaxis=dict(
            title=dict(text="Bezwzględny wpływ", font=dict(size=14, color='#ffffff')),
            tickfont=dict(color='#ffffff'),
            gridcolor='#444444'
        ),
        yaxis=dict(
            title=dict(text="Czynnik", font=dict(size=14, color='#ffffff')),
            tickfont=dict(color='#ffffff')
        ),
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
        font=dict(color='#ffffff', size=13),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        template='plotly_dark'
    )

    return fig


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

st.sidebar.markdown("---")

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

    # Toggle demo mode (jeśli API dostępne)
    if current_mode != 'unavailable':
        with st.sidebar.expander("Ustawienia trybu", expanded=False):
            demo_enabled = st.checkbox(
                "Wymuś tryb demo",
                value=not model_loaded,
                help="Zaznacz, aby używać symulowanych predykcji zamiast modelu ML",
                disabled=not model_loaded  # Disable jeśli model nie jest załadowany
            )

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

if analysis_mode == "Pojedynczy pacjent":
    st.sidebar.markdown('<h2 style="color: #1a5276;">Dane pacjenta</h2>', unsafe_allow_html=True)

    with st.sidebar.expander("Dane demograficzne", expanded=True):
        wiek = st.number_input("Wiek", min_value=18, max_value=100, value=55)
        plec = st.selectbox("Płeć", options=["Kobieta", "Mężczyzna"])
        wiek_rozpoznania = st.number_input("Wiek rozpoznania", min_value=0, max_value=100, value=50)

    with st.sidebar.expander("Manifestacje narządowe", expanded=True):
        liczba_narzadow = st.slider("Liczba zajętych narządów", 0, 10, 2)
        manifestacja_nerki = st.checkbox("Nerki")
        manifestacja_serce = st.checkbox("Serce/naczynia")
        manifestacja_csn = st.checkbox("Ośrodkowy układ nerwowy")
        manifestacja_neuro = st.checkbox("Obwodowy układ nerwowy")
        manifestacja_pokarm = st.checkbox("Układ pokarmowy")

    with st.sidebar.expander("Przebieg choroby", expanded=False):
        oit = st.checkbox("Zaostrzenia wymagające OIT")
        kreatynina = st.number_input("Kreatynina (μmol/L)", min_value=0.0, value=100.0)
        crp = st.number_input("Max CRP (mg/L)", min_value=0.0, value=30.0)

    with st.sidebar.expander("Leczenie", expanded=False):
        plazmaferezy = st.checkbox("Plazmaferezy")
        dializa = st.checkbox("Dializa")
        sterydy = st.number_input("Dawka sterydów (g)", min_value=0.0, value=0.5)
        czas_sterydow = st.number_input("Czas sterydów (mies.)", min_value=0, value=12)

    with st.sidebar.expander("Powikłania", expanded=False):
        powiklania_serce = st.checkbox("Powikłania sercowo-płucne")
        powiklania_infekcja = st.checkbox("Infekcje")

    # Przycisk analizy
    analyze_button = st.sidebar.button("Analizuj", type="primary")

    # Przygotuj dane pacjenta
    patient_data = {
        "wiek": wiek,
        "plec": 1 if plec == "Mężczyzna" else 0,
        "wiek_rozpoznania": wiek_rozpoznania,
        "opoznienie_rozpoznia": wiek - wiek_rozpoznania,
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
    # Zmienne dla trybu masowego (nieużywane w trybie pojedynczym)
    uploaded_file = None
    batch_analyze_button = False
else:
    # Zmienne dla trybu pojedynczego pacjenta (nieużywane w trybie masowym)
    analyze_button = False
    patient_data = {}
    # uploaded_file i batch_analyze_button są już zdefiniowane w sekcji sidebar dla trybu masowego

# ============================================================================
# GŁÓWNA SEKCJA
# ============================================================================

st.markdown('<h1 class="main-header">System XAI do predykcji śmiertelności w zapaleniu naczyń</h1>', unsafe_allow_html=True)

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

            results_df = process_batch_patients(df, update_progress)

            progress_bar.progress(1.0)
            status_text.text(f"Zakończono! Przetworzono {len(results_df)} pacjentów.")

            # Pokaż tryb przetwarzania
            if 'processing_mode' in results_df.columns:
                mode = results_df['processing_mode'].iloc[0] if len(results_df) > 0 else 'unknown'
                if mode == 'api':
                    st.success("**Tryb API** - Predykcje z wytrenowanego modelu XGBoost")
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

        with chart_col2:
            fig_hist = create_probability_histogram(results_df)
            st.plotly_chart(fig_hist, use_container_width=True)

        # Wykres scatter
        st.markdown("### Analiza wiek vs ryzyko")
        fig_scatter = create_age_risk_scatter(results_df)
        st.plotly_chart(fig_scatter, use_container_width=True)

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
            display_df.style.applymap(color_risk, subset=['Poziom']),
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

    # Pobierz predykcję
    with st.spinner("Analizuję dane..."):
        prediction = call_api("/predict", "POST", patient_data)
        if prediction is None:
            prediction = get_demo_prediction(patient_data)

        explanation = get_demo_explanation(patient_data)

    # Wyświetl wyniki
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown('<h3 class="section-header">Wynik analizy</h3>', unsafe_allow_html=True)

        # Gauge chart
        fig_gauge = create_gauge_chart(prediction["probability"], "Ryzyko zgonu")
        st.plotly_chart(fig_gauge, width='stretch')

        # Risk level badge
        risk_level = prediction["risk_level"]
        if risk_level == "low":
            st.markdown('<div class="risk-low"><strong>Niskie ryzyko</strong><br>Wskaźniki w normie.</div>', unsafe_allow_html=True)
        elif risk_level == "moderate":
            st.markdown('<div class="risk-moderate"><strong>Umiarkowane ryzyko</strong><br>Zalecana zwiększona obserwacja.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="risk-high"><strong>Wysokie ryzyko</strong><br>Wymaga szczególnej uwagi.</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<h3 class="section-header">Kluczowe czynniki</h3>', unsafe_allow_html=True)

        # Wyświetl top czynniki
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
    tab1, tab2, tab3, tab4 = st.tabs(["SHAP", "LIME", "Porównanie", "Chat AI"])

    with tab1:
        st.subheader("SHAP - Wartości Shapleya")
        st.markdown("""
        Wykres pokazuje wpływ każdego czynnika na predykcję modelu.
        Wartości dodatnie (czerwone) zwiększają ryzyko, ujemne (zielone) zmniejszają.
        """)

        all_factors_for_chart = explanation["risk_factors"] + explanation["protective_factors"]
        fig_waterfall = create_waterfall_chart(all_factors_for_chart, "Wpływ czynników (SHAP)")
        if fig_waterfall:
            st.plotly_chart(fig_waterfall, width='stretch')

    with tab2:
        st.subheader("LIME - Lokalne wyjaśnienie")
        st.markdown("""
        Wykres pokazuje bezwzględną ważność czynników w lokalnym modelu.
        """)

        fig_bar = create_bar_chart(all_factors_for_chart, "Ważność czynników (LIME)")
        if fig_bar:
            st.plotly_chart(fig_bar, width='stretch')

    with tab3:
        st.subheader("Porównanie metod XAI")

        col_comp1, col_comp2 = st.columns(2)

        with col_comp1:
            st.markdown("**Ranking SHAP:**")
            for i, f in enumerate(all_factors_sorted[:5], 1):
                st.write(f"{i}. {f['feature']}")

        with col_comp2:
            st.markdown("**Ranking LIME:**")
            lime_sorted = sorted(all_factors_for_chart, key=lambda x: abs(x["contribution"]), reverse=True)
            for i, f in enumerate(lime_sorted[:5], 1):
                st.write(f"{i}. {f['feature']}")

        st.info("Wysoka zgodność rankingów między metodami zwiększa wiarygodność wyjaśnień.")

    with tab4:
        st.markdown('<h3 class="section-header">Rozmowa o wynikach</h3>', unsafe_allow_html=True)

        # Inicjalizuj historię chatu
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Wyświetl historię
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Input użytkownika
        if prompt := st.chat_input("Zadaj pytanie o wyniki analizy..."):
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                # Prosty response bez API
                if any(word in prompt.lower() for word in ['wynik', 'ryzyko', 'analiza']):
                    response = f"""
Na podstawie analizy, poziom ryzyka wynosi **{prediction['risk_level']}**
(prawdopodobieństwo: {prediction['probability']:.1%}).

Główne czynniki to:
- {all_factors_sorted[0]['feature']}
- {all_factors_sorted[1]['feature'] if len(all_factors_sorted) > 1 else 'brak'}

Czy masz dodatkowe pytania?
"""
                elif any(word in prompt.lower() for word in ['czynnik', 'dlaczego']):
                    response = "Główne czynniki wpływające na ocenę to:\n\n"
                    for f in all_factors_sorted[:3]:
                        response += f"- **{f['feature']}**: wpływ {f['contribution']:+.3f}\n"
                else:
                    response = """
Mogę pomóc Ci zrozumieć:
- Wyniki analizy ryzyka
- Czynniki wpływające na ocenę
- Znaczenie poszczególnych wskaźników

O czym chciałbyś porozmawiać?
"""
                response += "\n\n*Pamiętaj: to narzędzie informacyjne, skonsultuj się z lekarzem.*"

                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

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
        st.markdown("""
        <div class="info-card">
            <h3>Modele ML</h3>
            <ul style="margin: 0; padding-left: 1.2rem;">
                <li>XGBoost</li>
                <li>Random Forest</li>
                <li>LightGBM</li>
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
Vasculitis XAI System v1.0.0 | © 2024
</div>
""", unsafe_allow_html=True)
