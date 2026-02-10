"""
Dashboard Streamlit dla systemu XAI.

Interfejs użytkownika do wyjaśniania decyzji zewnętrznego AI
w zapaleniu naczyń z wyjaśnieniami XAI (SHAP, LIME).
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
    page_title="Vasculitis XAI - Wyjaśnianie decyzji AI",
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
    .factor-positive { color: #155724; }
    .factor-negative { color: #721c24; }
    .info-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #2874a6;
        color: #2c3e50;
    }
    .info-card h3 { color: #1a5276; margin-bottom: 0.5rem; }
    .info-card ul { color: #34495e; }
    .info-card ul li { color: #2c3e50; margin: 0.3rem 0; }
    .stAlert > div { color: #1a5276 !important; }
    .streamlit-expanderHeader { color: #1a5276 !important; font-weight: 600; }
    .js-plotly-plot .plotly .main-svg { background: transparent !important; }
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
        st.warning("API niedostępne. Wyjaśnienia XAI będą niedostępne.")
        return None
    except Exception as e:
        st.error(f"Błąd: {e}")
        return None


def create_gauge_chart(probability: float, title: str = "Ryzyko") -> go.Figure:
    """Utwórz wykres gauge."""
    if probability < 0.3:
        color = "#28a745"
    elif probability < 0.7:
        color = "#ffc107"
    else:
        color = "#dc3545"

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

    names = [f.get("feature", "") for f in factors]
    values = [f.get("contribution", f.get("weight", 0)) for f in factors]

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


def create_bar_chart(factors: list, title: str = "Ważność czynników") -> go.Figure:
    """Utwórz wykres słupkowy."""
    if not factors:
        return None

    names = [f.get("feature", "") for f in factors]
    values = [abs(f.get("contribution", f.get("weight", 0))) for f in factors]
    colors = ["#dc3545" if f.get("contribution", f.get("weight", 0)) > 0 else "#28a745" for f in factors]

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
# FUNKCJE ANALIZY MASOWEJ
# ============================================================================

COLUMN_MAPPING = {
    'wiek_rozpoznania': 'wiek_rozpoznania',
    'age_at_diagnosis': 'wiek_rozpoznania',
    'opoznienie_rozpoznia': 'opoznienie_rozpoznia',
    'delay': 'opoznienie_rozpoznia',
    'miesno_szkiel': 'manifestacja_miesno_szkiel',
    'manifestacja_miesno_szkiel': 'manifestacja_miesno_szkiel',
    'skora': 'manifestacja_skora',
    'manifestacja_skora': 'manifestacja_skora',
    'wzrok': 'manifestacja_wzrok',
    'manifestacja_wzrok': 'manifestacja_wzrok',
    'nos_ucho_gardlo': 'manifestacja_nos_ucho_gardlo',
    'manifestacja_nos_ucho_gardlo': 'manifestacja_nos_ucho_gardlo',
    'oddechowy': 'manifestacja_oddechowy',
    'manifestacja_oddechowy': 'manifestacja_oddechowy',
    'serce': 'manifestacja_sercowo_naczyniowy',
    'heart': 'manifestacja_sercowo_naczyniowy',
    'manifestacja_sercowo_naczyniowy': 'manifestacja_sercowo_naczyniowy',
    'pokarmowy': 'manifestacja_pokarmowy',
    'gi': 'manifestacja_pokarmowy',
    'manifestacja_pokarmowy': 'manifestacja_pokarmowy',
    'moczowo_plciowy': 'manifestacja_moczowo_plciowy',
    'manifestacja_moczowo_plciowy': 'manifestacja_moczowo_plciowy',
    'csn': 'manifestacja_zajecie_csn',
    'cns': 'manifestacja_zajecie_csn',
    'manifestacja_zajecie_csn': 'manifestacja_zajecie_csn',
    'neuro': 'manifestacja_neurologiczny',
    'manifestacja_neurologiczny': 'manifestacja_neurologiczny',
    'liczba_narzadow': 'liczba_zajetych_narzadow',
    'liczba_zajetych_narzadow': 'liczba_zajetych_narzadow',
    'organ_count': 'liczba_zajetych_narzadow',
    'hospital': 'zaostrz_wymagajace_hospital',
    'zaostrz_wymagajace_hospital': 'zaostrz_wymagajace_hospital',
    'oit': 'zaostrz_wymagajace_oit',
    'icu': 'zaostrz_wymagajace_oit',
    'zaostrz_wymagajace_oit': 'zaostrz_wymagajace_oit',
    'kreatynina': 'kreatynina',
    'creatinine': 'kreatynina',
    'czas_sterydow': 'czas_sterydow',
    'steroid_duration': 'czas_sterydow',
    'plazmaferezy': 'plazmaferezy',
    'plasmapheresis': 'plazmaferezy',
    'eozynofilia': 'eozynofilia_krwi_obwodowej_wartosc',
    'eozynofilia_krwi_obwodowej_wartosc': 'eozynofilia_krwi_obwodowej_wartosc',
    'eosinophilia': 'eozynofilia_krwi_obwodowej_wartosc',
    'powiklania_neurologiczne': 'powiklania_neurologiczne',
    'neurological_complications': 'powiklania_neurologiczne',
    'external_probability': 'external_probability',
    'ext_prob': 'external_probability',
    'ryzyko_ai': 'external_probability',
    'id': 'patient_id',
    'patient_id': 'patient_id',
    'id_pacjenta': 'patient_id',
}

DEFAULT_VALUES = {
    'wiek_rozpoznania': 50,
    'opoznienie_rozpoznia': 0,
    'manifestacja_miesno_szkiel': 0,
    'manifestacja_skora': 0,
    'manifestacja_wzrok': 0,
    'manifestacja_nos_ucho_gardlo': 0,
    'manifestacja_oddechowy': 0,
    'manifestacja_sercowo_naczyniowy': 0,
    'manifestacja_pokarmowy': 0,
    'manifestacja_moczowo_plciowy': 0,
    'manifestacja_zajecie_csn': 0,
    'manifestacja_neurologiczny': 0,
    'liczba_zajetych_narzadow': 2,
    'zaostrz_wymagajace_hospital': 0,
    'zaostrz_wymagajace_oit': 0,
    'kreatynina': 100.0,
    'czas_sterydow': 0,
    'plazmaferezy': 0,
    'eozynofilia_krwi_obwodowej_wartosc': 0.0,
    'powiklania_neurologiczne': 0,
}


def parse_uploaded_file(uploaded_file) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Parsuj wgrany plik CSV lub JSON."""
    try:
        file_name = uploaded_file.name.lower()

        if file_name.endswith('.csv'):
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

            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                if 'patients' in data:
                    df = pd.DataFrame(data['patients'])
                elif 'data' in data:
                    df = pd.DataFrame(data['data'])
                else:
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
    """Normalizuj nazwy kolumn i uzupełnij brakujące wartości."""
    df.columns = [col.lower().strip().replace(' ', '_') for col in df.columns]

    rename_map = {}
    for col in df.columns:
        if col in COLUMN_MAPPING:
            rename_map[col] = COLUMN_MAPPING[col]
    df = df.rename(columns=rename_map)

    for col, default_val in DEFAULT_VALUES.items():
        if col not in df.columns:
            df[col] = default_val

    if 'patient_id' not in df.columns:
        df['patient_id'] = [f"P{i+1:04d}" for i in range(len(df))]

    # Konwertuj wartości boolean
    for col in df.columns:
        if df[col].dtype == 'object':
            if col in DEFAULT_VALUES and DEFAULT_VALUES[col] in [0, 1]:
                df[col] = df[col].apply(
                    lambda x: 1 if str(x).lower() in ['tak', 'yes', 'true', '1', 't', 'y'] else 0
                )

    return df


def process_batch_patients(df: pd.DataFrame, progress_callback=None) -> pd.DataFrame:
    """Przetwórz pacjentów wsadowo."""
    results = []
    total = len(df)
    patient_ids = df.get('patient_id', pd.Series([f"P{i+1:04d}" for i in range(total)]))

    has_ext_prob = 'external_probability' in df.columns

    for i, (_, row) in enumerate(df.iterrows()):
        ext_prob = float(row.get('external_probability', 0.5)) if has_ext_prob else 0.5

        if ext_prob < 0.3:
            risk_level = "low"
        elif ext_prob < 0.7:
            risk_level = "moderate"
        else:
            risk_level = "high"

        result = {
            'patient_id': patient_ids.iloc[i] if i < len(patient_ids) else f"P{i+1:04d}",
            'wiek_rozpoznania': row.get('wiek_rozpoznania', 50),
            'liczba_narzadow': row.get('liczba_zajetych_narzadow', 2),
            'external_probability': ext_prob,
            'probability_pct': f"{ext_prob*100:.1f}%",
            'risk_level': risk_level,
            'risk_level_pl': {'low': 'Niskie', 'moderate': 'Umiarkowane', 'high': 'Wysokie'}.get(risk_level, 'Niskie'),
        }
        results.append(result)

        if progress_callback:
            progress_callback((i + 1) / total)

    return pd.DataFrame(results)


def create_risk_distribution_chart(results_df: pd.DataFrame) -> go.Figure:
    """Utwórz wykres rozkładu ryzyka."""
    risk_counts = results_df['risk_level'].value_counts()

    colors = {'low': '#28a745', 'moderate': '#ffc107', 'high': '#dc3545'}
    labels = {'low': 'Niskie', 'moderate': 'Umiarkowane', 'high': 'Wysokie'}

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
        x=results_df['external_probability'] * 100,
        nbinsx=20,
        marker_color='#2874a6',
        opacity=0.8,
        name='Pacjenci'
    ))

    fig.add_vline(x=30, line_dash="dash", line_color="#28a745",
                  annotation_text="Próg niski/umiarkowany")
    fig.add_vline(x=70, line_dash="dash", line_color="#dc3545",
                  annotation_text="Próg umiarkowany/wysoki")

    fig.update_layout(
        title=dict(text="Rozkład ocen ryzyka z zewnętrznego AI", font=dict(size=18, color='#ffffff')),
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


def export_results_to_csv(results_df: pd.DataFrame) -> str:
    """Eksportuj wyniki do CSV."""
    export_df = results_df[['patient_id', 'wiek_rozpoznania', 'liczba_narzadow',
                            'probability_pct', 'risk_level_pl']].copy()
    export_df.columns = ['ID Pacjenta', 'Wiek rozpoznania', 'Liczba narządów',
                         'Ryzyko AI (%)', 'Poziom ryzyka']
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
            'avg_probability': float(results_df['external_probability'].mean())
        },
        'patients': results_df.to_dict(orient='records')
    }
    return json.dumps(export_data, indent=2, ensure_ascii=False)


def get_api_status() -> Dict:
    """Pobierz status API."""
    try:
        response = requests.get(f"{API_URL}/config/demo-mode", timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return {
        "demo_allowed": True,
        "model_loaded": False,
        "explainers_ready": False,
        "current_mode": "unavailable",
        "force_api_mode": False
    }


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
    api_status = get_api_status()

    st.sidebar.markdown('<h2 style="color: #1a5276;">Wgraj plik</h2>', unsafe_allow_html=True)

    uploaded_file = st.sidebar.file_uploader(
        "Wybierz plik CSV lub JSON",
        type=['csv', 'json'],
        help="Plik powinien zawierać dane pacjentów i kolumnę external_probability."
    )

    st.sidebar.caption("**Wymagana kolumna:** `external_probability` (0-1)")

    with st.sidebar.expander("Format pliku", expanded=False):
        st.markdown("""
        **Wymagane kolumny:**
        - `external_probability` (lub `ext_prob`, `ryzyko_ai`) - wynik z zewnętrznego AI (0-1)

        **Opcjonalne kolumny (cechy pacjenta):**
        - `wiek_rozpoznania`, `liczba_narzadow`
        - `serce`, `oddechowy`, `csn`, `neuro` (manifestacje)
        - `oit`, `hospital`, `kreatynina`
        - `eozynofilia`, `plazmaferezy`, `czas_sterydow`

        **Przykład CSV:**
        ```
        id,external_probability,wiek_rozpoznania,liczba_narzadow
        P001,0.72,65,3
        P002,0.25,45,2
        ```
        """)

    batch_analyze_button = st.sidebar.button(
        "Analizuj plik",
        type="primary",
        disabled=uploaded_file is None
    )

    # Przykładowy plik
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Pobierz przykładowy plik:**")

    sample_csv = """id,external_probability,wiek_rozpoznania,opoznienie_rozpoznia,liczba_narzadow,serce,oddechowy,oit,hospital,kreatynina
P001,0.72,65,6,3,0,1,0,1,120
P002,0.25,45,3,2,0,0,0,0,85
P003,0.85,72,12,4,1,1,1,1,180
P004,0.15,38,2,1,0,0,0,0,75
P005,0.45,55,5,2,0,1,0,0,110
P006,0.78,68,8,3,1,0,0,1,220
P007,0.30,42,4,2,0,0,0,0,90
P008,0.92,78,15,5,1,1,1,1,250
P009,0.35,51,3,2,0,0,0,0,95
P010,0.55,63,7,3,0,1,1,0,145"""

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

    # Predykcja zewnętrznego AI
    with st.sidebar.expander("Predykcja zewnętrznego AI", expanded=True):
        external_probability_pct = st.slider(
            "Prawdopodobieństwo zgonu z zewnętrznego AI (%)",
            0, 100, 50,
            help="Wprowadź wynik z zewnętrznego systemu AI"
        )

    with st.sidebar.expander("Dane demograficzne", expanded=True):
        wiek_rozpoznania = st.number_input("Wiek rozpoznania", min_value=0, max_value=100, value=50)
        opoznienie_rozpoznia = st.number_input("Opóźnienie rozpoznania (mies.)", min_value=0, max_value=120, value=6)

    with st.sidebar.expander("Manifestacje narządowe", expanded=True):
        liczba_narzadow = st.slider("Liczba zajętych narządów", 0, 12, 2)
        manifestacja_miesno_szkiel = st.checkbox("Mięśniowo-szkieletowe")
        manifestacja_skora = st.checkbox("Skóra")
        manifestacja_wzrok = st.checkbox("Wzrok")
        manifestacja_nos_ucho_gardlo = st.checkbox("Nos/ucho/gardło")
        manifestacja_oddechowy = st.checkbox("Układ oddechowy")
        manifestacja_serce = st.checkbox("Serce/naczynia")
        manifestacja_pokarm = st.checkbox("Układ pokarmowy")
        manifestacja_moczowo_plciowy = st.checkbox("Układ moczowo-płciowy")
        manifestacja_csn = st.checkbox("Ośrodkowy układ nerwowy")
        manifestacja_neuro = st.checkbox("Obwodowy układ nerwowy")

    with st.sidebar.expander("Przebieg choroby", expanded=False):
        zaostrz_hospital = st.checkbox("Zaostrzenia wymagające hospitalizacji")
        oit = st.checkbox("Zaostrzenia wymagające OIT")
        kreatynina = st.number_input("Kreatynina (μmol/L)", min_value=0.0, value=100.0)
        eozynofilia = st.number_input("Eozynofilia (%)", min_value=0.0, value=0.0)

    with st.sidebar.expander("Leczenie", expanded=False):
        czas_sterydow = st.number_input("Czas sterydów (mies.)", min_value=0.0, value=0.0)
        plazmaferezy = st.checkbox("Plazmaferezy")

    with st.sidebar.expander("Powikłania", expanded=False):
        powiklania_neuro = st.checkbox("Powikłania neurologiczne")

    analyze_button = st.sidebar.button("Analizuj", type="primary")

    # Dane pacjenta - 20 cech dopasowanych do modelu
    patient_data = {
        "wiek_rozpoznania": wiek_rozpoznania,
        "opoznienie_rozpoznia": opoznienie_rozpoznia,
        "manifestacja_miesno_szkiel": 1 if manifestacja_miesno_szkiel else 0,
        "manifestacja_skora": 1 if manifestacja_skora else 0,
        "manifestacja_wzrok": 1 if manifestacja_wzrok else 0,
        "manifestacja_nos_ucho_gardlo": 1 if manifestacja_nos_ucho_gardlo else 0,
        "manifestacja_oddechowy": 1 if manifestacja_oddechowy else 0,
        "manifestacja_sercowo_naczyniowy": 1 if manifestacja_serce else 0,
        "manifestacja_pokarmowy": 1 if manifestacja_pokarm else 0,
        "manifestacja_moczowo_plciowy": 1 if manifestacja_moczowo_plciowy else 0,
        "manifestacja_zajecie_csn": 1 if manifestacja_csn else 0,
        "manifestacja_neurologiczny": 1 if manifestacja_neuro else 0,
        "liczba_zajetych_narzadow": liczba_narzadow,
        "zaostrz_wymagajace_hospital": 1 if zaostrz_hospital else 0,
        "zaostrz_wymagajace_oit": 1 if oit else 0,
        "kreatynina": kreatynina,
        "czas_sterydow": czas_sterydow,
        "plazmaferezy": 1 if plazmaferezy else 0,
        "eozynofilia_krwi_obwodowej_wartosc": eozynofilia,
        "powiklania_neurologiczne": 1 if powiklania_neuro else 0,
    }
    external_probability = external_probability_pct / 100.0

    uploaded_file = None
    batch_analyze_button = False
else:
    analyze_button = False
    patient_data = {}
    external_probability = 0.5

# ============================================================================
# GŁÓWNA SEKCJA
# ============================================================================

st.markdown(
    '<h1 class="main-header">System XAI do wyjaśniania decyzji AI w zapaleniu naczyń</h1>',
    unsafe_allow_html=True
)

# ============================================================================
# TRYB ANALIZY MASOWEJ
# ============================================================================

if analysis_mode == "Analiza masowa":
    if batch_analyze_button and uploaded_file is not None:
        st.session_state['batch_analyzed'] = True

        with st.spinner("Wczytuję plik..."):
            df, error = parse_uploaded_file(uploaded_file)

        if error:
            st.error(f"Błąd: {error}")
        else:
            df = normalize_dataframe(df)

            if 'external_probability' not in df.columns:
                st.error("Brak kolumny `external_probability` w pliku. Dodaj wyniki z zewnętrznego AI.")
            else:
                st.success(f"Wczytano {len(df)} pacjentów z pliku {uploaded_file.name}")

                with st.expander("Podgląd wczytanych danych", expanded=False):
                    st.dataframe(df.head(10), use_container_width=True)

                st.markdown("### Przetwarzanie pacjentów...")
                progress_bar = st.progress(0)
                status_text = st.empty()

                def update_progress(progress):
                    progress_bar.progress(progress)
                    status_text.text(f"Przetworzono: {int(progress * len(df))}/{len(df)} pacjentów")

                results_df = process_batch_patients(df, update_progress)

                progress_bar.progress(1.0)
                status_text.text(f"Zakończono! Przetworzono {len(results_df)} pacjentów.")

                st.session_state['batch_results'] = results_df

    if st.session_state.get('batch_results') is not None:
        results_df = st.session_state['batch_results']

        st.markdown("---")
        st.markdown("## Podsumowanie analizy")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(label="Liczba pacjentów", value=len(results_df))
        with col2:
            low_count = (results_df['risk_level'] == 'low').sum()
            st.metric(label="Niskie ryzyko", value=low_count,
                      delta=f"{low_count/len(results_df)*100:.1f}%")
        with col3:
            moderate_count = (results_df['risk_level'] == 'moderate').sum()
            st.metric(label="Umiarkowane ryzyko", value=moderate_count,
                      delta=f"{moderate_count/len(results_df)*100:.1f}%")
        with col4:
            high_count = (results_df['risk_level'] == 'high').sum()
            st.metric(label="Wysokie ryzyko", value=high_count,
                      delta=f"{high_count/len(results_df)*100:.1f}%",
                      delta_color="inverse")

        st.markdown("---")
        st.markdown("## Wizualizacje")

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            fig_pie = create_risk_distribution_chart(results_df)
            st.plotly_chart(fig_pie, use_container_width=True)

        with chart_col2:
            fig_hist = create_probability_histogram(results_df)
            st.plotly_chart(fig_hist, use_container_width=True)

        # Tabela wyników
        st.markdown("---")
        st.markdown("## Szczegółowe wyniki")

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
                options=['external_probability', 'wiek_rozpoznania', 'patient_id'],
                format_func=lambda x: {
                    'external_probability': 'Ryzyko AI',
                    'wiek_rozpoznania': 'Wiek rozpoznania',
                    'patient_id': 'ID pacjenta'
                }[x]
            )

        filtered_df = results_df[results_df['risk_level'].isin(risk_filter)]
        filtered_df = filtered_df.sort_values(by=sort_by, ascending=(sort_by == 'patient_id'))

        def color_risk(val):
            if val == 'Niskie':
                return 'background-color: #28a745; color: white'
            elif val == 'Umiarkowane':
                return 'background-color: #ffc107; color: black'
            else:
                return 'background-color: #dc3545; color: white'

        display_df = filtered_df[['patient_id', 'wiek_rozpoznania', 'liczba_narzadow',
                                   'probability_pct', 'risk_level_pl']].copy()
        display_df.columns = ['ID', 'Wiek rozp.', 'Narządy', 'Ryzyko AI', 'Poziom']

        st.dataframe(
            display_df.style.applymap(color_risk, subset=['Poziom']),
            use_container_width=True,
            height=400
        )

        st.markdown(f"*Wyświetlono {len(filtered_df)} z {len(results_df)} pacjentów*")

        st.markdown("---")
        st.markdown("## Eksport wyników")

        export_col1, export_col2 = st.columns(2)

        with export_col1:
            csv_data = export_results_to_csv(results_df)
            st.download_button(
                label="Pobierz wyniki (CSV)",
                data=csv_data,
                file_name=f"wyniki_xai_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

        with export_col2:
            json_data = export_results_to_json(results_df)
            st.download_button(
                label="Pobierz wyniki (JSON)",
                data=json_data,
                file_name=f"wyniki_xai_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )

    elif not st.session_state.get('batch_analyzed', False):
        st.markdown("""
        <div class="upload-zone">
            <h2 style="color: #2874a6;">Analiza masowa - wyjaśnianie decyzji AI</h2>
            <p style="font-size: 1.1rem; color: #666;">
                Wgraj plik CSV lub JSON z danymi pacjentów i wynikami zewnętrznego AI,
                aby uzyskać wyjaśnienia XAI dla wielu osób jednocześnie.
            </p>
            <hr style="border-color: #4a5568; margin: 1.5rem 0;">
            <p style="color: #888;">← Użyj panelu bocznego, aby wgrać plik</p>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# TRYB POJEDYNCZEGO PACJENTA
# ============================================================================

elif analyze_button or st.session_state.get('analyzed', False):
    st.session_state['analyzed'] = True

    with st.spinner("Analizuję dane..."):
        # Wywołaj /analyze z external_probability
        analysis_result = call_api("/analyze", "POST", {
            "patient": patient_data,
            "external_probability": external_probability
        })

    # Wyświetl wyniki
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown('<h3 class="section-header">Ocena zewnętrznego AI</h3>', unsafe_allow_html=True)

        fig_gauge = create_gauge_chart(external_probability, "Ryzyko zgonu (zewnętrzne AI)")
        st.plotly_chart(fig_gauge, width='stretch')

        if external_probability < 0.3:
            risk_level = "low"
        elif external_probability < 0.7:
            risk_level = "moderate"
        else:
            risk_level = "high"

        if risk_level == "low":
            st.markdown('<div class="risk-low"><strong>Niskie ryzyko</strong><br>Zewnętrzny AI ocenia ryzyko jako niskie.</div>', unsafe_allow_html=True)
        elif risk_level == "moderate":
            st.markdown('<div class="risk-moderate"><strong>Umiarkowane ryzyko</strong><br>Zewnętrzny AI ocenia ryzyko jako umiarkowane.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="risk-high"><strong>Wysokie ryzyko</strong><br>Zewnętrzny AI ocenia ryzyko jako wysokie.</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<h3 class="section-header">Kluczowe czynniki (XAI)</h3>', unsafe_allow_html=True)

        st.info("Poniżej XAI wyjaśnia, które cechy pacjenta mogły wpłynąć na decyzję zewnętrznego AI.")

        if analysis_result and analysis_result.get('shap_explanation'):
            shap_exp = analysis_result['shap_explanation']
            all_factors = shap_exp.get('feature_contributions', [])
            all_factors_sorted = sorted(all_factors, key=lambda x: abs(x.get('contribution', 0)), reverse=True)

            for factor in all_factors_sorted[:5]:
                contrib = factor.get('contribution', 0)
                if contrib > 0:
                    arrow = "↑"
                    direction = "zwiększa"
                else:
                    arrow = "↓"
                    direction = "zmniejsza"
                st.markdown(f'{arrow} **{factor["feature"]}** - {direction} ryzyko ({contrib:+.3f})')
        else:
            st.warning("Wyjaśnienia SHAP niedostępne (model/explainer nie załadowany). Uruchom API z modelem.")

    # Zakładki z szczegółami
    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs(["SHAP", "LIME", "Porównanie", "Chat AI"])

    with tab1:
        st.subheader("SHAP - Wartości Shapleya")
        st.markdown("Wykres pokazuje wpływ każdego czynnika na ocenę modelu. "
                     "Wartości dodatnie (czerwone) zwiększają ryzyko, ujemne (zielone) zmniejszają.")

        if analysis_result and analysis_result.get('shap_explanation'):
            factors = analysis_result['shap_explanation'].get('feature_contributions', [])
            fig_waterfall = create_waterfall_chart(factors, "Wpływ czynników (SHAP)")
            if fig_waterfall:
                st.plotly_chart(fig_waterfall, width='stretch')
        else:
            st.info("SHAP niedostępny. Uruchom API z załadowanym modelem i danymi referencyjnymi.")

    with tab2:
        st.subheader("LIME - Lokalne wyjaśnienie")
        st.markdown("Wykres pokazuje bezwzględną ważność czynników w lokalnym modelu.")

        if analysis_result and analysis_result.get('lime_explanation'):
            lime_exp = analysis_result['lime_explanation']
            lime_factors = lime_exp.get('feature_weights', [])
            # Konwertuj do formatu z 'contribution' kluczem
            lime_for_chart = [
                {"feature": f.get("feature", ""), "contribution": f.get("weight", 0)}
                for f in lime_factors
            ]
            fig_bar = create_bar_chart(lime_for_chart, "Ważność czynników (LIME)")
            if fig_bar:
                st.plotly_chart(fig_bar, width='stretch')
        else:
            st.info("LIME niedostępny. Uruchom API z załadowanym modelem i danymi referencyjnymi.")

    with tab3:
        st.subheader("Porównanie metod XAI")

        if analysis_result and analysis_result.get('shap_explanation') and analysis_result.get('lime_explanation'):
            col_comp1, col_comp2 = st.columns(2)

            with col_comp1:
                st.markdown("**Ranking SHAP:**")
                shap_factors = analysis_result['shap_explanation'].get('feature_contributions', [])
                shap_sorted = sorted(shap_factors, key=lambda x: abs(x.get('contribution', 0)), reverse=True)
                for i, f in enumerate(shap_sorted[:5], 1):
                    st.write(f"{i}. {f['feature']}")

            with col_comp2:
                st.markdown("**Ranking LIME:**")
                lime_factors = analysis_result['lime_explanation'].get('feature_weights', [])
                lime_sorted = sorted(lime_factors, key=lambda x: abs(x.get('weight', 0)), reverse=True)
                for i, f in enumerate(lime_sorted[:5], 1):
                    st.write(f"{i}. {f['feature']}")

            st.info("Wysoka zgodność rankingów między metodami zwiększa wiarygodność wyjaśnień.")
        else:
            st.info("Porównanie wymaga dostępności obu metod XAI.")

    with tab4:
        st.markdown('<h3 class="section-header">Rozmowa o wyjaśnieniach</h3>', unsafe_allow_html=True)

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Zadaj pytanie o wyjaśnienia decyzji AI..."):
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                chat_result = call_api("/chat", "POST", {
                    "message": prompt,
                    "patient": patient_data,
                    "external_probability": external_probability
                })

                if chat_result:
                    response = chat_result.get("response", "Przepraszam, nie mogę odpowiedzieć.")
                else:
                    response = (
                        f"Zewnętrzny system AI oszacował ryzyko na {external_probability:.0%}. "
                        "Niestety API jest niedostępne - nie mogę podać szczegółowych wyjaśnień.\n\n"
                        "*Pamiętaj: to narzędzie informacyjne, skonsultuj się z lekarzem.*"
                    )

                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

elif analysis_mode == "Pojedynczy pacjent":
    # Strona powitalna
    st.markdown("""
    <h2 style="color: #1a5276;">Witaj w systemie XAI!</h2>
    <p style="font-size: 1.1rem; color: #495057;">
        Ten system wyjaśnia decyzje zewnętrznego systemu AI dotyczące ryzyka w zapaleniu naczyń.
    </p>
    <ul style="list-style: none; padding-left: 0; font-size: 1rem;">
        <li style="margin: 0.5rem 0;"><strong>Wprowadź wynik AI</strong> - podaj prawdopodobieństwo z zewnętrznego systemu</li>
        <li style="margin: 0.5rem 0;"><strong>Wprowadź dane pacjenta</strong> - cechy kliniczne</li>
        <li style="margin: 0.5rem 0;"><strong>Zrozum decyzję</strong> - zobacz które czynniki wpłynęły na ocenę AI</li>
    </ul>
    <h3 style="color: #1a5276; margin-top: 1.5rem;">Jak zacząć?</h3>
    <ol style="font-size: 1rem; color: #495057;">
        <li>Ustaw prawdopodobieństwo z zewnętrznego AI w panelu bocznym</li>
        <li>Wprowadź dane pacjenta</li>
        <li>Kliknij przycisk <strong>Analizuj</strong></li>
        <li>Przeglądaj wyjaśnienia SHAP/LIME</li>
    </ol>
    <hr style="margin-top: 1.5rem;">
    """, unsafe_allow_html=True)

    col_info1, col_info2, col_info3 = st.columns(3)

    with col_info1:
        st.markdown("""
        <div class="info-card">
            <h3>Koncept</h3>
            <ul style="margin: 0; padding-left: 1.2rem;">
                <li>Zewnętrzne AI predykuje</li>
                <li>XAI wyjaśnia decyzję</li>
                <li>SHAP + LIME</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_info2:
        st.markdown("""
        <div class="info-card">
            <h3>Metody XAI</h3>
            <ul style="margin: 0; padding-left: 1.2rem;">
                <li>SHAP (wartości Shapleya)</li>
                <li>LIME (lokalne wyjaśnienia)</li>
                <li>Porównanie metod</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_info3:
        st.markdown("""
        <div class="info-card">
            <h3>20 cech modelu</h3>
            <ul style="margin: 0; padding-left: 1.2rem;">
                <li>Dane demograficzne</li>
                <li>10 manifestacji narządowych</li>
                <li>Parametry kliniczne</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div class="disclaimer">
<strong>Ważne:</strong> Ten system wyjaśnia decyzje zewnętrznego systemu AI.
Jest narzędziem wspierającym decyzje kliniczne. Nie zastępuje profesjonalnej
oceny medycznej. Wszystkie decyzje dotyczące leczenia powinny być podejmowane
przez wykwalifikowany personel medyczny w oparciu o pełny obraz kliniczny pacjenta.
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; color: #5a6268; padding: 1rem; font-size: 0.9rem;">
Vasculitis XAI System v2.0.0 | Wyjaśnianie decyzji AI
</div>
""", unsafe_allow_html=True)
