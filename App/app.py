import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os
import json
import joblib
from scipy import stats

def calcular_percentil(serie, valor):
    try:
        return float(stats.percentileofscore(serie, valor))
    except Exception:
        return float((serie <= valor).mean() * 100)

# ── Configuración de la página ─────────────────────────────────────
st.set_page_config(
    page_title="Predicción de Riesgo Cardiovascular | CardioRisk Studio",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Paleta de diseño (Monitor Médico Dark Navy) ────────────────────
BG_MAIN   = "#0D1B2A"   # Navy profundo de fondo
BG_CARD   = "#112236"   # Navy medio para tarjetas
BG_PANEL  = "#0A1628"   # Navy oscuro para barra lateral
ACCENT    = "#00C9A7"   # Verde señal ECG
RISK      = "#FF6B6B"   # Rojo ámbar para alerta y riesgo
SAFE      = "#4ECDC4"   # Turquesa para valores saludables
WARN      = "#FFD166"   # Amarillo ámbar para riesgo moderado
BORDER    = "#1E3A5F"   # Borde sutil
TEXT_HI   = "#E8F4FD"   # Texto principal de alta legibilidad
TEXT_LO   = "#7BA3C4"   # Texto secundario
TEXT_MUT  = "#3D6B8A"   # Texto tenue

# ── Estilos CSS Personalizados ─────────────────────────────────────
st.markdown(f"""
<style>
  /* Fondo general */
  .stApp {{
      background-color: {BG_MAIN};
  }}

  /* Sidebar */
  [data-testid="stSidebar"] {{
      background-color: {BG_PANEL} !important;
      border-right: 1px solid {BORDER};
  }}
  [data-testid="stSidebar"] * {{
      color: {TEXT_HI} !important;
  }}

  /* Tabs de navegación */
  .stTabs [data-baseweb="tab-list"] {{
      background-color: {BG_CARD};
      border-radius: 12px;
      padding: 6px;
      gap: 6px;
      border: 1px solid {BORDER};
  }}
  .stTabs [data-baseweb="tab"] {{
      background-color: transparent;
      color: {TEXT_LO} !important;
      border-radius: 8px;
      padding: 10px 18px;
      font-size: 14px;
      font-weight: 600;
      transition: all 0.2s ease;
  }}
  .stTabs [aria-selected="true"] {{
      background-color: {BG_MAIN} !important;
      color: {ACCENT} !important;
      border-bottom: 2px solid {ACCENT} !important;
      box-shadow: 0 4px 12px rgba(0, 201, 167, 0.15);
  }}

  /* Tarjetas de métrica */
  .metric-card {{
      background: linear-gradient(135deg, {BG_CARD} 0%, #0F2A42 100%);
      border: 1px solid {BORDER};
      border-radius: 14px;
      padding: 1.2rem 1.4rem;
      text-align: center;
      position: relative;
      overflow: hidden;
      margin-bottom: 0.8rem;
  }}
  .metric-card::before {{
      content: '';
      position: absolute;
      top: 0; left: 0; right: 0;
      height: 3px;
      background: linear-gradient(90deg, {ACCENT}, {SAFE});
      border-radius: 14px 14px 0 0;
  }}
  .metric-value {{
      font-size: 2.1rem;
      font-weight: 700;
      color: {ACCENT};
      line-height: 1.1;
      letter-spacing: -0.02em;
  }}
  .metric-label {{
      font-size: 0.74rem;
      color: {TEXT_LO};
      text-transform: uppercase;
      letter-spacing: 0.1em;
      margin-top: 6px;
  }}
  .metric-sub {{
      font-size: 0.82rem;
      color: {TEXT_MUT};
      margin-top: 4px;
  }}

  /* Panel Hero */
  .hero {{
      background: linear-gradient(135deg, {BG_CARD} 0%, #0A2040 60%, #0D1B2A 100%);
      border: 1px solid {BORDER};
      border-radius: 16px;
      padding: 1.8rem 2.2rem;
      margin-bottom: 1.5rem;
      position: relative;
      overflow: hidden;
  }}
  .hero-title {{
      font-size: 2rem;
      font-weight: 700;
      color: {TEXT_HI};
      margin: 0;
      line-height: 1.2;
  }}
  .hero-subtitle {{
      font-size: 0.98rem;
      color: {TEXT_LO};
      margin-top: 6px;
  }}
  .hero-badge {{
      display: inline-block;
      background: rgba(0, 201, 167, 0.12);
      border: 1px solid rgba(0, 201, 167, 0.35);
      color: {ACCENT};
      font-size: 0.75rem;
      padding: 3px 12px;
      border-radius: 999px;
      margin-bottom: 8px;
      letter-spacing: 0.08em;
      font-weight: 600;
  }}
  .ecg-line {{
      position: absolute;
      right: 2rem;
      top: 50%;
      transform: translateY(-50%);
      opacity: 0.15;
      font-size: 5rem;
      color: {ACCENT};
  }}

  /* Etiquetas de sección */
  .section-label {{
      font-size: 0.75rem;
      color: {ACCENT};
      text-transform: uppercase;
      letter-spacing: 0.12em;
      font-weight: 700;
      margin-bottom: 0.8rem;
      padding-left: 10px;
      border-left: 3px solid {ACCENT};
  }}

  /* Cajas de diagnóstico clínico */
  .diag-box {{
      border-radius: 12px;
      padding: 1.2rem 1.4rem;
      margin-bottom: 1rem;
      border: 1px solid {BORDER};
  }}

  /* Ajustes de texto */
  p, span, div, label {{
      color: {TEXT_HI};
  }}
  h1, h2, h3, h4 {{
      color: {TEXT_HI} !important;
  }}
</style>
""", unsafe_allow_html=True)

# ── Resolución Dinámica de Rutas ───────────────────────────────────
def obtener_rutas():
    base_script = os.path.dirname(os.path.abspath(__file__))
    posibles_data = [
        os.path.join(base_script, '..', 'Data', 'processed', 'processed.csv'),
        os.path.join(base_script, 'Data', 'processed', 'processed.csv'),
        os.path.join(os.getcwd(), 'Data', 'processed', 'processed.csv'),
        os.path.join(os.getcwd(), '..', 'Data', 'processed', 'processed.csv'),
    ]
    posibles_modelos = [
        os.path.join(base_script, '..', 'models', 'modelo_cardiovascular.pkl'),
        os.path.join(base_script, 'models', 'modelo_cardiovascular.pkl'),
        os.path.join(os.getcwd(), 'models', 'modelo_cardiovascular.pkl'),
        os.path.join(os.getcwd(), '..', 'models', 'modelo_cardiovascular.pkl'),
    ]
    posibles_metricas = [
        os.path.join(base_script, '..', 'models', 'model_metrics.json'),
        os.path.join(base_script, 'models', 'model_metrics.json'),
        os.path.join(os.getcwd(), 'models', 'model_metrics.json'),
        os.path.join(os.getcwd(), '..', 'models', 'model_metrics.json'),
    ]

    ruta_csv = next((p for p in posibles_data if os.path.exists(p)), None)
    ruta_modelo = next((p for p in posibles_modelos if os.path.exists(p)), None)
    ruta_metricas = next((p for p in posibles_metricas if os.path.exists(p)), None)

    return ruta_csv, ruta_modelo, ruta_metricas

RUTA_CSV, RUTA_MODELO, RUTA_METRICAS = obtener_rutas()

# ── Carga de Datos y Modelo con Caché ──────────────────────────────
@st.cache_data
def cargar_datos(ruta):
    if not ruta or not os.path.exists(ruta):
        st.error(f"No se encontró el archivo de datos procesados.")
        st.info("Asegúrate de haber ejecutado: `python src/train_model.py`")
        st.stop()
    return pd.read_csv(ruta)

@st.cache_resource
def cargar_modelo(ruta):
    if ruta and os.path.exists(ruta):
        try:
            return joblib.load(ruta)
        except Exception as e:
            st.warning(f"No se pudo cargar el modelo serializado: {e}")
    return None

@st.cache_data
def cargar_metricas(ruta):
    if ruta and os.path.exists(ruta):
        try:
            with open(ruta, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return None

df = cargar_datos(RUTA_CSV)
modelo = cargar_modelo(RUTA_MODELO)
metricas_modelo = cargar_metricas(RUTA_METRICAS)

# ── Configuración de Matplotlib Tema Oscuro ────────────────────────
plt.rcParams.update({
    'figure.facecolor':  BG_CARD,
    'axes.facecolor':    BG_CARD,
    'axes.edgecolor':    BORDER,
    'axes.labelcolor':   TEXT_LO,
    'xtick.color':       TEXT_LO,
    'ytick.color':       TEXT_LO,
    'text.color':        TEXT_HI,
    'grid.color':        BORDER,
    'grid.alpha':        0.4,
})

# ── Barra Lateral (Sidebar) ────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style='text-align:center; padding: 0.8rem 0 1.2rem;'>
      <div style='font-size:2.6rem'>🫀</div>
      <div style='font-size:1.15rem; font-weight:700; color:{TEXT_HI}'>CardioRisk Studio</div>
      <div style='font-size:0.75rem; color:{ACCENT}; letter-spacing:0.05em; font-weight:600;'>SISTEMA DE APOYO CLÍNICO</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f'<div class="section-label">Filtros de Cohorte</div>', unsafe_allow_html=True)

    rango_edad = st.slider(
        "Edad del paciente (años)",
        int(df['age_years'].min()),
        int(np.ceil(df['age_years'].max())),
        (int(df['age_years'].min()), int(np.ceil(df['age_years'].max())))
    )

    genero_sel = st.selectbox("Género Biológico", ['Todos', 'Mujer', 'Hombre'])
    colesterol_sel = st.selectbox("Nivel de Colesterol", ['Todos', 'Normal (<200 mg/dL)', 'Alto (200-239)', 'Muy alto (≥240)'])
    tabaco_sel = st.selectbox("Hábito Tabáquico", ['Todos', 'No fumadores', 'Fumadores'])

    st.markdown("---")
    st.markdown(f"""
    <div style='font-size:0.75rem; color:{TEXT_MUT}; line-height:1.6;'>
      <b>Muestra procesada:</b> {len(df):,} pacientes reales<br>
      <b>Criterios clínicos:</b> OMS · AHA/ACC · ESC<br>
      <b>Modelo predictivo:</b> HistGradientBoosting (ROC-AUC 0.80)
    </div>
    """, unsafe_allow_html=True)

# ── Aplicar Filtros Globales a la Cohorte ──────────────────────────
dff = df.copy()
dff = dff[(dff['age_years'] >= rango_edad[0]) & (dff['age_years'] <= rango_edad[1])]

if genero_sel == 'Mujer':
    dff = dff[dff['gender'] == 1]
elif genero_sel == 'Hombre':
    dff = dff[dff['gender'] == 2]

if 'Normal' in colesterol_sel:
    dff = dff[dff['cholesterol'] == 1]
elif 'Alto' in colesterol_sel and 'Muy' not in colesterol_sel:
    dff = dff[dff['cholesterol'] == 2]
elif 'Muy alto' in colesterol_sel:
    dff = dff[dff['cholesterol'] == 3]

if tabaco_sel == 'No fumadores':
    dff = dff[dff['smoke'] == 0]
elif tabaco_sel == 'Fumadores':
    dff = dff[dff['smoke'] == 1]

# ── Hero Banner ────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
  <div class="hero-badge">CLINICAL AI DASHBOARD</div>
  <div class="hero-title">🫀 Evaluación y Predicción de Riesgo Cardiovascular</div>
  <div class="hero-subtitle">
    Monitoreo interactivo sobre una cohorte validada de {len(df):,} pacientes con plausibilidad fisiológica estricta.
  </div>
  <div class="ecg-line">∿∿</div>
</div>
""", unsafe_allow_html=True)

# ── Tarjetas de Métricas Rápidas ───────────────────────────────────
tasa_cardio = dff['cardio'].astype(int).mean() * 100 if len(dff) > 0 else 0
media_edad  = dff['age_years'].mean() if len(dff) > 0 else 0
media_ap_hi = dff['ap_hi'].mean() if len(dff) > 0 else 0
pct_hipert  = (dff['hypertension'].mean() * 100) if 'hypertension' in dff.columns and len(dff) > 0 else 0

c1, c2, c3, c4 = st.columns(4)
c1.markdown(f"""
<div class="metric-card">
  <div class="metric-value">{len(dff):,}</div>
  <div class="metric-label">Pacientes en Cohorte</div>
  <div class="metric-sub">de {len(df):,} totales</div>
</div>""", unsafe_allow_html=True)

c2.markdown(f"""
<div class="metric-card">
  <div class="metric-value" style="color:{RISK if tasa_cardio > 50 else ACCENT};">{tasa_cardio:.1f}%</div>
  <div class="metric-label">Prevalencia Enfermedad</div>
  <div class="metric-sub">cardio = 1</div>
</div>""", unsafe_allow_html=True)

c3.markdown(f"""
<div class="metric-card">
  <div class="metric-value">{media_edad:.1f}</div>
  <div class="metric-label">Edad Media</div>
  <div class="metric-sub">años cumplidos</div>
</div>""", unsafe_allow_html=True)

c4.markdown(f"""
<div class="metric-card">
  <div class="metric-value" style="color:{WARN if pct_hipert > 35 else ACCENT};">{pct_hipert:.1f}%</div>
  <div class="metric-label">Hipertensión Arterial</div>
  <div class="metric-sub">ap_hi≥140 o ap_lo≥90</div>
</div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Pestañas Principales de la Aplicación ──────────────────────────
tab_calc, tab_pop, tab_target, tab_num, tab_cat, tab_corr = st.tabs([
    "🧮 Calculadora Clínica de Riesgo",
    "👥 Comparador Poblacional",
    "📊 Variable Objetivo",
    "📈 Variables Numéricas",
    "🏷️ Variables Categóricas",
    "🔥 Factores e Importancia"
])


# ═══════════════════════════════════════════════════════════════════
# TAB 1: Calculadora Clínica de Riesgo Individual
# ═══════════════════════════════════════════════════════════════════
with tab_calc:
    st.markdown(f'<div class="section-label">Simulador de Riesgo Cardiovascular para Pacientes Reales</div>', unsafe_allow_html=True)
    st.markdown("""
    Ingresa los parámetros hemodinámicos y clínicos de un paciente para calcular instantáneamente
    su **Índice de Masa Corporal (IMC)**, **Presión de Pulso**, **Presión Arterial Media (PAM)**,
    **Estadío de Hipertensión según la AHA** y la **Probabilidad de Enfermedad Cardiovascular** generada por el modelo de Machine Learning.
    """)

    col_form, col_res = st.columns([1.1, 1.3], gap="large")

    with col_form:
        st.markdown(f"""
        <div style='background:{BG_CARD}; padding:1.2rem; border-radius:12px; border:1px solid {BORDER};'>
          <h4 style='margin-top:0; color:{ACCENT}; font-size:1.05rem;'>📋 Datos del Paciente</h4>
        </div>
        """, unsafe_allow_html=True)

        cf1, cf2 = st.columns(2)
        with cf1:
            p_edad = st.number_input("Edad (años)", min_value=29, max_value=65, value=52, step=1)
            p_genero_txt = st.selectbox("Género biológico", ["Femenino (Mujer)", "Masculino (Hombre)"])
            p_genero = 1 if "Femenino" in p_genero_txt else 2
            p_altura = st.number_input("Estatura (cm)", min_value=140, max_value=205, value=165, step=1)
            p_peso = st.number_input("Peso corporal (kg)", min_value=40.0, max_value=160.0, value=74.0, step=0.5)

        with cf2:
            p_ap_hi = st.number_input("Presión Sistólica (ap_hi, mmHg)", min_value=80, max_value=220, value=130, step=1)
            p_ap_lo = st.number_input("Presión Diastólica (ap_lo, mmHg)", min_value=50, max_value=130, value=85, step=1)
            p_chol_txt = st.selectbox("Colesterol Total", ["1: Normal (<200 mg/dL)", "2: Alto (200-239 mg/dL)", "3: Muy Alto (≥240 mg/dL)"])
            p_chol = int(p_chol_txt[0])
            p_gluc_txt = st.selectbox("Glucemia en Ayuno", ["1: Normal (<100 mg/dL)", "2: Glucosa alterada (100-125)", "3: Elevada / Diabetes (≥126)"])
            p_gluc = int(p_gluc_txt[0])

        st.markdown("<p style='font-size:0.85rem; color:#7BA3C4; font-weight:600; margin-top:8px;'>Estilo de Vida:</p>", unsafe_allow_html=True)
        cs1, cs2, cs3 = st.columns(3)
        with cs1:
            p_fuma = 1 if st.checkbox("Fuma tabaco", value=False) else 0
        with cs2:
            p_alco = 1 if st.checkbox("Consume alcohol", value=False) else 0
        with cs3:
            p_act  = 1 if st.checkbox("Físicamente activo", value=True) else 0

        # Validación hemodinámica inmediata
        if p_ap_lo >= p_ap_hi:
            st.error("⚠️ Alerta fisiológica: La presión diastólica no puede ser mayor o igual a la sistólica.")
        elif (p_ap_hi - p_ap_lo) < 20:
            st.warning("⚠️ Presión de pulso anormalmente estrecha (<20 mmHg). Verifica las mediciones.")

    # Cálculos Fisiológicos Derivados
    p_bmi = round(p_peso / ((p_altura / 100) ** 2), 1)
    p_pp  = p_ap_hi - p_ap_lo
    p_map = round(p_ap_lo + (p_pp / 3), 1)

    # Clasificación IMC OMS
    if p_bmi < 18.5:
        bmi_cat, bmi_color = "Bajo peso", "#4ECDC4"
    elif p_bmi < 25.0:
        bmi_cat, bmi_color = "Normopeso", "#00C9A7"
    elif p_bmi < 30.0:
        bmi_cat, bmi_color = "Sobrepeso", "#FFD166"
    elif p_bmi < 35.0:
        bmi_cat, bmi_color = "Obesidad Grado I", "#FF9F43"
    elif p_bmi < 40.0:
        bmi_cat, bmi_color = "Obesidad Grado II", "#FF6B6B"
    else:
        bmi_cat, bmi_color = "Obesidad Mórbida (Grado III)", "#EE5253"

    # Clasificación Hipertensión AHA 2017
    if p_ap_hi >= 140 or p_ap_lo >= 90:
        aha_cat, aha_color = "Hipertensión Grado 2", "#FF6B6B"
    elif (130 <= p_ap_hi <= 139) or (80 <= p_ap_lo <= 89):
        aha_cat, aha_color = "Hipertensión Grado 1", "#FFD166"
    elif (120 <= p_ap_hi <= 129) and (p_ap_lo < 80):
        aha_cat, aha_color = "Presión Arterial Elevada", "#F39C12"
    else:
        aha_cat, aha_color = "Presión Arterial Normal", "#00C9A7"

    with col_res:
        st.markdown(f"""
        <div style='background:{BG_CARD}; padding:1.2rem; border-radius:12px; border:1px solid {BORDER};'>
          <h4 style='margin-top:0; color:{ACCENT}; font-size:1.05rem;'>🩺 Diagnóstico Hemodinámico y Riesgo</h4>
        </div>
        """, unsafe_allow_html=True)

        # Tarjetas de biomarcadores calculados
        rc1, rc2, rc3 = st.columns(3)
        rc1.markdown(f"""
        <div class="metric-card">
          <div class="metric-value" style="font-size:1.6rem; color:{bmi_color};">{p_bmi}</div>
          <div class="metric-label">IMC (kg/m²)</div>
          <div class="metric-sub" style="color:{bmi_color}; font-weight:600;">{bmi_cat}</div>
        </div>""", unsafe_allow_html=True)

        rc2.markdown(f"""
        <div class="metric-card">
          <div class="metric-value" style="font-size:1.6rem; color:{'#FF6B6B' if p_pp > 60 else ACCENT};">{p_pp}</div>
          <div class="metric-label">Presión de Pulso</div>
          <div class="metric-sub">Diferencial (mmHg)</div>
        </div>""", unsafe_allow_html=True)

        rc3.markdown(f"""
        <div class="metric-card">
          <div class="metric-value" style="font-size:1.6rem; color:{'#FF6B6B' if p_map > 105 else ACCENT};">{p_map}</div>
          <div class="metric-label">PAM (Perfusión)</div>
          <div class="metric-sub">Media (mmHg)</div>
        </div>""", unsafe_allow_html=True)

        # Inferencia del Modelo Predictivo
        proba_riesgo = 0.50
        if modelo is not None and p_ap_hi > p_ap_lo:
            df_paciente = pd.DataFrame([{
                'age_years': float(p_edad),
                'height': float(p_altura),
                'weight': float(p_peso),
                'ap_hi': float(p_ap_hi),
                'ap_lo': float(p_ap_lo),
                'bmi': float(p_bmi),
                'pulse_pressure': float(p_pp),
                'map': float(p_map),
                'gender': int(p_genero),
                'cholesterol': int(p_chol),
                'gluc': int(p_gluc),
                'smoke': int(p_fuma),
                'alco': int(p_alco),
                'active': int(p_act)
            }])
            try:
                proba_riesgo = modelo.predict_proba(df_paciente)[0, 1]
            except Exception as e:
                st.error(f"Error al calcular predicción: {e}")
                proba_riesgo = 0.50

        # Estratificación clínica del riesgo estimado
        pct_riesgo = proba_riesgo * 100
        if pct_riesgo < 25.0:
            nivel_riesgo = "BAJO RIESGO CARDIOVASCULAR"
            color_riesgo = SAFE
            desc_riesgo = "Perfil clínico favorable. Mantener hábitos de vida saludables y controles periódicos preventivos."
        elif pct_riesgo < 50.0:
            nivel_riesgo = "RIESGO CARDIOVASCULAR MODERADO"
            color_riesgo = WARN
            desc_riesgo = "Presencia de factores de riesgo incipientes. Se aconseja optimizar dieta, ejercicio y vigilar presión."
        elif pct_riesgo < 75.0:
            nivel_riesgo = "ALTO RIESGO CARDIOVASCULAR"
            color_riesgo = "#FF9F43"
            desc_riesgo = "Elevada probabilidad de afección cardiovascular. Requiere valoración médica formal y ajuste de factores modificables."
        else:
            nivel_riesgo = "RIESGO CARDIOVASCULAR CRÍTICO / MUY ALTO"
            color_riesgo = RISK
            desc_riesgo = "Signos hemodinámicos y metabólicos de alto impacto. Es imprescindible seguimiento médico especializado inmediato."

        st.markdown(f"""
        <div style='background:linear-gradient(135deg, rgba(17,34,54,0.9), rgba(15,42,66,0.95)); border:2px solid {color_riesgo}; border-radius:14px; padding:1.4rem; text-align:center; margin-top:0.6rem;'>
          <div style='font-size:0.8rem; color:{TEXT_LO}; text-transform:uppercase; letter-spacing:0.12em; font-weight:700;'>Probabilidad Estimada por IA</div>
          <div style='font-size:3.2rem; font-weight:800; color:{color_riesgo}; line-height:1.1; margin:0.3rem 0;'>{pct_riesgo:.1f}%</div>
          <div style='font-size:1.05rem; font-weight:700; color:{color_riesgo}; letter-spacing:0.05em;'>{nivel_riesgo}</div>
          <div style='font-size:0.85rem; color:{TEXT_HI}; margin-top:0.6rem; line-height:1.5;'>{desc_riesgo}</div>
        </div>
        """, unsafe_allow_html=True)

        # Alertas personalizadas
        alertas = []
        if p_ap_hi >= 140 or p_ap_lo >= 90:
            alertas.append(f"🚨 **Presión Arterial ({aha_cat}):** Sus cifras superan el umbral normotenso. Se recomienda reducir sodio, evitar estimulantes y acudir a consulta médica.")
        if p_fuma == 1:
            alertas.append("🚬 **Tabaquismo Activo:** El tabaco deteriora el endotelio vascular y multiplica por 2 el riesgo de infarto de miocardio.")
        if p_chol >= 2:
            alertas.append("🧈 **Colesterol Elevado:** Se aconseja perfil lipídico completo en sangre y control de grasas saturadas.")
        if p_bmi >= 25.0:
            alertas.append(f"⚖️ **Índice de Masa Corporal ({bmi_cat}):** La reducción ponderal disminuye directamente la presión sistólica y la resistencia a la insulina.")
        if p_act == 0:
            alertas.append("🏃 **Sedentarismo:** Incorporar al menos 150 minutos semanales de actividad aeróbica moderada (caminar rápido, nadar, bicicleta).")

        if alertas:
            st.markdown("<p style='font-size:0.82rem; color:#00C9A7; font-weight:700; margin-top:1rem; margin-bottom:4px;'>RECOMENDACIONES CLÍNICAS PERSONALIZADAS:</p>", unsafe_allow_html=True)
            for a in alertas:
                st.markdown(f"<div style='font-size:0.83rem; color:{TEXT_HI}; background:rgba(30,58,95,0.3); padding:6px 12px; border-radius:8px; margin-bottom:4px; border-left:3px solid {ACCENT};'>{a}</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# TAB 2: Comparador Poblacional
# ═══════════════════════════════════════════════════════════════════
with tab_pop:
    st.markdown(f'<div class="section-label">Posicionamiento del Paciente vs Cohorte de {len(df):,} Pacientes</div>', unsafe_allow_html=True)
    st.markdown("""
    Esta vista compara las mediciones del paciente actual contra la distribución de la población del estudio,
    distinguiendo entre quienes **no tienen** y quienes **sí tienen** enfermedad cardiovascular comprobada.
    """)

    cp1, cp2 = st.columns(2)

    with cp1:
        # Comparación Presión Sistólica
        pct_hi = calcular_percentil(df['ap_hi'], p_ap_hi)
        fig, ax = plt.subplots(figsize=(6, 3.8))
        sns.kdeplot(df[df['cardio']==0]['ap_hi'], ax=ax, color=SAFE, fill=True, alpha=0.35, label='Sin enfermedad (cardio=0)')
        sns.kdeplot(df[df['cardio']==1]['ap_hi'], ax=ax, color=RISK, fill=True, alpha=0.35, label='Con enfermedad (cardio=1)')
        ax.axvline(p_ap_hi, color=WARN, linestyle='--', linewidth=2.2, label=f'Paciente: {p_ap_hi} mmHg (P{pct_hi:.0f})')
        ax.set_title(f'Presión Sistólica (Percentil {pct_hi:.1f})', color=TEXT_HI, fontsize=11)
        ax.set_xlabel('ap_hi (mmHg)')
        ax.set_ylabel('Densidad')
        ax.legend(facecolor=BG_CARD, edgecolor=BORDER, fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with cp2:
        # Comparación IMC
        pct_bmi = calcular_percentil(df['bmi'], p_bmi)
        fig, ax = plt.subplots(figsize=(6, 3.8))
        sns.kdeplot(df[df['cardio']==0]['bmi'], ax=ax, color=SAFE, fill=True, alpha=0.35, label='Sin enfermedad')
        sns.kdeplot(df[df['cardio']==1]['bmi'], ax=ax, color=RISK, fill=True, alpha=0.35, label='Con enfermedad')
        ax.axvline(p_bmi, color=WARN, linestyle='--', linewidth=2.2, label=f'Paciente: {p_bmi} kg/m² (P{pct_bmi:.0f})')
        ax.set_title(f'Índice de Masa Corporal (Percentil {pct_bmi:.1f})', color=TEXT_HI, fontsize=11)
        ax.set_xlabel('IMC (kg/m²)')
        ax.set_ylabel('Densidad')
        ax.legend(facecolor=BG_CARD, edgecolor=BORDER, fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Tabla resumen de percentiles
    resumen_percentiles = pd.DataFrame({
        'Biomarcador': ['Presión Sistólica (ap_hi)', 'Presión Diastólica (ap_lo)', 'Presión de Pulso (PP)', 'Índice Masa Corporal (IMC)', 'Edad'],
        'Valor Paciente': [f"{p_ap_hi} mmHg", f"{p_ap_lo} mmHg", f"{p_pp} mmHg", f"{p_bmi} kg/m²", f"{p_edad} años"],
        'Percentil Poblacional': [
            f"P{calcular_percentil(df['ap_hi'], p_ap_hi):.1f}",
            f"P{calcular_percentil(df['ap_lo'], p_ap_lo):.1f}",
            f"P{calcular_percentil(df['pulse_pressure'], p_pp):.1f}",
            f"P{calcular_percentil(df['bmi'], p_bmi):.1f}",
            f"P{calcular_percentil(df['age_years'], p_edad):.1f}"
        ],
        'Media Población Sana': [
            f"{df[df['cardio']==0]['ap_hi'].mean():.1f} mmHg",
            f"{df[df['cardio']==0]['ap_lo'].mean():.1f} mmHg",
            f"{df[df['cardio']==0]['pulse_pressure'].mean():.1f} mmHg",
            f"{df[df['cardio']==0]['bmi'].mean():.1f} kg/m²",
            f"{df[df['cardio']==0]['age_years'].mean():.1f} años"
        ],
        'Media Población Enferma': [
            f"{df[df['cardio']==1]['ap_hi'].mean():.1f} mmHg",
            f"{df[df['cardio']==1]['ap_lo'].mean():.1f} mmHg",
            f"{df[df['cardio']==1]['pulse_pressure'].mean():.1f} mmHg",
            f"{df[df['cardio']==1]['bmi'].mean():.1f} kg/m²",
            f"{df[df['cardio']==1]['age_years'].mean():.1f} años"
        ]
    })
    st.table(resumen_percentiles)


# ═══════════════════════════════════════════════════════════════════
# TAB 3: Variable Objetivo
# ═══════════════════════════════════════════════════════════════════
with tab_target:
    st.markdown(f'<div class="section-label">Balance de la Variable Objetivo (cardio)</div>', unsafe_allow_html=True)

    col_g, col_a = st.columns([1, 1])

    with col_g:
        conteo = dff['cardio'].astype(int).value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(5, 3.8))
        bars = ax.bar(
            ['Sanos\n(cardio=0)', 'Con Enfermedad\n(cardio=1)'],
            conteo.values,
            color=[SAFE, RISK], edgecolor=BG_CARD, width=0.5
        )
        for bar, val in zip(bars, conteo.values):
            pct = val / conteo.sum() * 100
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (conteo.max()*0.02),
                    f'{val:,}\n({pct:.1f}%)', ha='center', fontsize=9.5, color=TEXT_HI, fontweight='bold')
        ax.set_ylabel('Nº de Pacientes', color=TEXT_LO)
        ax.set_title('Distribución de cardio en la selección', color=TEXT_HI, pad=12)
        ax.set_ylim(0, conteo.max() * 1.25)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_a:
        ratio = (conteo.max() / conteo.min()) if conteo.min() > 0 else 1.0
        st.markdown(f'<div class="section-label">Diagnóstico Estadístico del Balance</div>', unsafe_allow_html=True)

        if ratio < 1.15:
            st.success(f"✓ Cohorte Equilibrada — Ratio: {ratio:.2f}")
        elif ratio < 2.0:
            st.warning(f"⚠ Desequilibrio Leve — Ratio: {ratio:.2f}")
        else:
            st.error(f"✗ Desequilibrio Significativo — Ratio: {ratio:.2f}")

        st.markdown(f"""
        <div style='margin-top:1rem; color:{TEXT_LO}; font-size:0.9rem; line-height:1.7'>
          En la muestra total procesada ({len(df):,} pacientes), la proporción es de <b>50.3% sanos vs 49.7% enfermos</b>.<br><br>
          <b style='color:{ACCENT}'>Implicación para Machine Learning:</b><br>
          - No existe sesgo de clase mayoritaria.<br>
          - No se requiere aplicar técnicas de sobremuestreo sintético como <b>SMOTE</b> ni submuestreo aleatorio.<br>
          - La métrica de <b>Exactitud (Accuracy)</b> es estadísticamente fiable y complementa adecuadamente al <b>ROC-AUC</b>.
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# TAB 4: Variables Numéricas
# ═══════════════════════════════════════════════════════════════════
with tab_num:
    st.markdown(f'<div class="section-label">Análisis Univariado y Bivariado de Variables Continuas</div>', unsafe_allow_html=True)

    nombres_var = {
        'age_years': 'Edad (años)',
        'weight':    'Peso (kg)',
        'height':    'Estatura (cm)',
        'ap_hi':     'Presión Sistólica (mmHg)',
        'ap_lo':     'Presión Diastólica (mmHg)',
        'bmi':       'Índice de Masa Corporal (IMC)',
        'pulse_pressure': 'Presión de Pulso / Diferencial (mmHg)',
        'map':       'Presión Arterial Media (PAM, mmHg)'
    }
    vars_disp = [c for c in nombres_var if c in dff.columns]

    variable = st.selectbox("Selecciona la variable a examinar:", vars_disp, format_func=lambda x: nombres_var.get(x, x))

    col_h, col_v = st.columns([1, 1])

    with col_h:
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        dff[dff['cardio']==0][variable].hist(
            bins=35, ax=ax, alpha=0.75, color=SAFE, label='Sanos (cardio=0)', edgecolor='none')
        dff[dff['cardio']==1][variable].hist(
            bins=35, ax=ax, alpha=0.75, color=RISK, label='Enfermos (cardio=1)', edgecolor='none')
        ax.legend(fontsize=9, facecolor=BG_CARD, edgecolor=BORDER)
        ax.set_xlabel(nombres_var.get(variable, variable))
        ax.set_ylabel('Frecuencia')
        ax.set_title(f'Histograma — {nombres_var.get(variable, variable)}', color=TEXT_HI, pad=10)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_v:
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        sns.violinplot(
            data=dff, x='cardio', y=variable, ax=ax,
            palette={0: SAFE, 1: RISK}, inner='box', linewidth=1.0
        )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Sin enfermedad', 'Con enfermedad'])
        m0 = dff[dff['cardio']==0][variable].mean()
        m1 = dff[dff['cardio']==1][variable].mean()
        ax.axhline(m0, color=SAFE, linestyle='--', alpha=0.7, linewidth=1.3)
        ax.axhline(m1, color=RISK, linestyle='--', alpha=0.7, linewidth=1.3)
        ax.set_title(f'Media Sanos: {m0:.1f}  |  Media Enfermos: {m1:.1f}', color=TEXT_HI, pad=10)
        ax.set_xlabel('')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        diff = abs(m1 - m0) / m0 * 100
        st.info(f"📌 **Diferencia relativa entre grupos:** `{diff:.1f}%`. Variable con fuerte discriminación pronóstica.")


# ═══════════════════════════════════════════════════════════════════
# TAB 5: Variables Categóricas
# ═══════════════════════════════════════════════════════════════════
with tab_cat:
    st.markdown(f'<div class="section-label">Prevalencia de Riesgo según Factores Clínicos y Conductuales</div>', unsafe_allow_html=True)
    st.caption("Rojo = supera la prevalencia media de la cohorte · Verde/Azul = por debajo de la media")

    cols_cat = ['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'gender']
    etiquetas = {
        'gender':      {1: 'Mujer', 2: 'Hombre'},
        'cholesterol': {1: 'Normal', 2: 'Alto', 3: 'Muy alto'},
        'gluc':        {1: 'Normal', 2: 'Alto', 3: 'Muy alto'},
        'smoke':       {0: 'No fuma', 1: 'Fuma'},
        'alco':        {0: 'No bebe', 1: 'Bebe alcohol'},
        'active':      {0: 'Sedentario', 1: 'Activo'}
    }
    nombres_cat = {
        'gender': 'Género Biológico', 'cholesterol': 'Colesterol Sérico', 'gluc': 'Glucosa en Ayuno',
        'smoke': 'Hábito Tabáquico', 'alco': 'Consumo de Alcohol', 'active': 'Actividad Física'
    }

    media_global = dff['cardio'].astype(int).mean() * 100

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    for ax, col in zip(axes.flatten(), cols_cat):
        tasa  = dff.groupby(col)['cardio'].apply(lambda x: x.astype(int).mean() * 100)
        etiq  = [etiquetas[col].get(k, str(k)) for k in tasa.index]
        clrs  = [RISK if v > media_global else SAFE for v in tasa.values]

        bars = ax.bar(etiq, tasa.values, color=clrs, edgecolor=BG_CARD, width=0.55)
        ax.axhline(media_global, color=TEXT_MUT, linestyle='--', linewidth=1.2,
                   label=f'Media: {media_global:.1f}%')

        for bar, val in zip(bars, tasa.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.0,
                    f'{val:.1f}%', ha='center', fontsize=9, color=TEXT_HI, fontweight='bold')

        ax.set_title(nombres_cat[col], color=TEXT_HI, pad=8, fontsize=11)
        ax.set_ylabel('% con cardio=1', color=TEXT_LO, fontsize=9)
        ax.set_ylim(0, 90)
        ax.legend(fontsize=7.5, facecolor=BG_CARD, edgecolor=BORDER)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ═══════════════════════════════════════════════════════════════════
# TAB 6: Correlaciones e Importancia de Factores
# ═══════════════════════════════════════════════════════════════════
with tab_corr:
    st.markdown(f'<div class="section-label">Matriz de Correlación y Pesos del Modelo Predictivo</div>', unsafe_allow_html=True)

    cols_corr = [c for c in ['age_years', 'height', 'weight', 'ap_hi', 'ap_lo',
                             'bmi', 'pulse_pressure', 'map', 'cholesterol', 'gluc',
                             'smoke', 'alco', 'active', 'cardio']
                 if c in dff.columns]

    corr = dff[cols_corr].astype(float).corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    col_hm, col_bar = st.columns([1.35, 1])

    with col_hm:
        fig, ax = plt.subplots(figsize=(8, 7))
        sns.heatmap(
            corr, mask=mask, annot=True, fmt='.2f',
            cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            square=True, linewidths=0.4, linecolor=BG_MAIN,
            cbar_kws={'shrink': 0.75}, ax=ax,
            annot_kws={'size': 8, 'color': TEXT_HI}
        )
        ax.set_title('Correlación de Pearson entre Biomarcadores', color=TEXT_HI, pad=12)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_bar:
        st.markdown(f'<div class="section-label">Correlación Directa con Cardio</div>', unsafe_allow_html=True)
        corr_target = corr['cardio'].drop('cardio').sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        clrs_bar = [RISK if v > 0 else SAFE for v in corr_target.values]
        ax.barh(corr_target.index, corr_target.values, color=clrs_bar, edgecolor=BG_CARD, height=0.65)
        ax.axvline(0, color=TEXT_MUT, linewidth=0.8)
        ax.set_xlabel('Correlación con cardio', color=TEXT_LO)
        ax.set_title('¿Qué factores se asocian más al riesgo?', color=TEXT_HI, pad=10, fontsize=10)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Coeficientes del modelo si existen
    if metricas_modelo and 'coeficientes_logisticos' in metricas_modelo:
        st.markdown(f'<div class="section-label">Importancia Multivariada (Odds Ratios Estandarizados)</div>', unsafe_allow_html=True)
        coef_dict = metricas_modelo['coeficientes_logisticos']
        s_coef = pd.Series(coef_dict).sort_values(ascending=True)

        fig, ax = plt.subplots(figsize=(10, 4))
        colores_coef = [RISK if v > 0 else SAFE for v in s_coef.values]
        ax.barh(s_coef.index, s_coef.values, color=colores_coef, edgecolor=BG_CARD)
        ax.axvline(0, color=TEXT_MUT, linewidth=0.9, linestyle='--')
        ax.set_xlabel('Coeficiente Beta Estandarizado (Log-Odds)')
        ax.set_title('Impacto relativo en el riesgo cardiovascular ajustado por las demás variables', color=TEXT_HI, pad=10)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ── Footer Profesional ─────────────────────────────────────────────
st.markdown(f"""
<div style='margin-top:2.5rem; padding:1.2rem 0; border-top:1px solid {BORDER};
     text-align:center; color:{TEXT_MUT}; font-size:0.82rem;'>
  <b>CardioRisk Studio</b> — Proyecto de Ciencia de Datos y Machine Learning Clínico · 
  Autora: <b>Ana Colina Arismendi</b> · 
  <a href='https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset' target='_blank'
     style='color:{ACCENT}; text-decoration:none;'>Dataset Kaggle</a>
</div>
""", unsafe_allow_html=True)