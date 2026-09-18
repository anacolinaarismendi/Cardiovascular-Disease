import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import joblib
from scipy import stats

def calcular_percentil(serie, valor):
    try:
        return float(stats.percentileofscore(serie.dropna(), valor))
    except Exception:
        return float((serie <= valor).mean() * 100)

# ── Configuración de la página ─────────────────────────────────────
st.set_page_config(
    page_title="Predicción de Riesgo Cardiovascular | Framingham Clinical AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Paleta de diseño (Monitor Médico Dark Navy) ────────────────────
BG_MAIN   = "#0D1B2A"   # Navy profundo de fondo
BG_CARD   = "#112236"   # Navy medio para tarjetas
BG_PANEL  = "#0A1628"   # Navy oscuro para barra lateral
ACCENT    = "#00C9A7"   # Verde señal ECG
RISK      = "#FF6B6B"   # Rojo coral para alerta y riesgo
SAFE      = "#4ECDC4"   # Turquesa para valores saludables
WARN      = "#FFD166"   # Amarillo ámbar para riesgo moderado
BORDER    = "#1E3A5F"   # Borde sutil
TEXT_HI   = "#E8F4FD"   # Texto principal de alta legibilidad
TEXT_LO   = "#7BA3C4"   # Texto secundario
TEXT_MUT  = "#3D6B8A"   # Texto tenue

# ── Estilos CSS Personalizados ─────────────────────────────────────
st.markdown(f"""
<style>
  .stApp {{
      background-color: {BG_MAIN};
  }}
  [data-testid="stSidebar"] {{
      background-color: {BG_PANEL} !important;
      border-right: 1px solid {BORDER};
  }}
  [data-testid="stSidebar"] * {{
      color: {TEXT_HI} !important;
  }}
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
      padding: 10px 16px;
      font-size: 13.5px;
      font-weight: 600;
      transition: all 0.2s ease;
  }}
  .stTabs [aria-selected="true"] {{
      background-color: {BG_MAIN} !important;
      color: {ACCENT} !important;
      border-bottom: 2px solid {ACCENT} !important;
      box-shadow: 0 4px 12px rgba(0, 201, 167, 0.15);
  }}
  .metric-card {{
      background: linear-gradient(135deg, {BG_CARD} 0%, #0F2A42 100%);
      border: 1px solid {BORDER};
      border-radius: 14px;
      padding: 1.1rem 1.3rem;
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
  .hero {{
      background: linear-gradient(135deg, {BG_CARD} 0%, #0A2040 60%, #0D1B2A 100%);
      border: 1px solid {BORDER};
      border-radius: 16px;
      padding: 1.6rem 2.0rem;
      margin-bottom: 1.2rem;
  }}
  .hero-title {{
      font-size: 1.85rem;
      font-weight: 700;
      color: {TEXT_HI};
      margin: 0;
      line-height: 1.2;
  }}
  .hero-subtitle {{
      font-size: 0.95rem;
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
  }}
  .section-label {{
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: {ACCENT};
      font-weight: 700;
      margin-bottom: 0.6rem;
  }}
</style>
""", unsafe_allow_html=True)

# ── Localización de Rutas ──────────────────────────────────────────
def obtener_rutas():
    base_script = os.path.dirname(os.path.abspath(__file__))
    posibles_data = [
        os.path.join(base_script, '..', 'Data', 'processed', 'processed.csv'),
        os.path.join(base_script, 'Data', 'processed', 'processed.csv'),
        os.path.join(os.getcwd(), 'Data', 'processed', 'processed.csv'),
    ]
    posibles_modelos = [
        os.path.join(base_script, '..', 'models', 'modelo_cardiovascular.pkl'),
        os.path.join(base_script, 'models', 'modelo_cardiovascular.pkl'),
        os.path.join(os.getcwd(), 'models', 'modelo_cardiovascular.pkl'),
    ]
    posibles_metricas = [
        os.path.join(base_script, '..', 'models', 'model_metrics.json'),
        os.path.join(base_script, 'models', 'model_metrics.json'),
        os.path.join(os.getcwd(), 'models', 'model_metrics.json'),
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
            st.warning(f"No se pudo cargar el modelo: {e}")
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

# ── Configuración Matplotlib Tema Oscuro ───────────────────────────
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
    <div style='text-align:center; padding: 0.6rem 0 1rem;'>
      <div style='font-size:2.5rem'>🫀</div>
      <div style='font-size:1.15rem; font-weight:700; color:{TEXT_HI}'>Framingham AI Studio</div>
      <div style='font-size:0.75rem; color:{ACCENT}; letter-spacing:0.05em; font-weight:600;'>RIESGO CARDIOVASCULAR A 10 AÑOS</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f'<div class="section-label">Filtros de Cohorte</div>', unsafe_allow_html=True)

    min_edad = int(df['age'].min())
    max_edad = int(df['age'].max())
    rango_edad = st.slider("Rango de Edad", min_edad, max_edad, (min_edad, max_edad))

    genero_sel = st.selectbox("Sexo Biológico", ['Todos', 'Mujeres', 'Hombres'])
    tabaco_sel = st.selectbox("Hábito Tabáquico", ['Todos', 'No fumadores', 'Fumadores activos'])
    hipert_sel = st.selectbox("Hipertensión Arterial", ['Todos', 'Normotensos', 'Hipertensos'])
    diab_sel   = st.selectbox("Diabetes Mellitus", ['Todos', 'No Diabéticos', 'Diabéticos'])

    st.markdown("---")
    st.markdown(f"""
    <div style='font-size:0.75rem; color:{TEXT_MUT}; line-height:1.6;'>
      <b>Cohorte:</b> Framingham Heart Study (Longitudinal)<br>
      <b>Población procesada:</b> {len(df):,} pacientes reales<br>
      <b>Asociaciones:</b> Odds Ratios Clínicos Auténticos<br>
      <b>Modelo:</b> Pipeline Clínico Calibrado (ROC-AUC {metricas_modelo.get('roc_auc', 0.69):.2f})
    </div>
    """, unsafe_allow_html=True)

# ── Aplicar Filtros a la Cohorte ──────────────────────────────────
dff = df.copy()
dff = dff[(dff['age'] >= rango_edad[0]) & (dff['age'] <= rango_edad[1])]

if genero_sel == 'Mujeres':
    dff = dff[dff['male'] == 0]
elif genero_sel == 'Hombres':
    dff = dff[dff['male'] == 1]

if tabaco_sel == 'No fumadores':
    dff = dff[dff['currentSmoker'] == 0]
elif tabaco_sel == 'Fumadores activos':
    dff = dff[dff['currentSmoker'] == 1]

if hipert_sel == 'Normotensos':
    dff = dff[dff['prevalentHyp'] == 0]
elif hipert_sel == 'Hipertensos':
    dff = dff[dff['prevalentHyp'] == 1]

if diab_sel == 'No Diabéticos':
    dff = dff[dff['diabetes'] == 0]
elif diab_sel == 'Diabéticos':
    dff = dff[dff['diabetes'] == 1]

# ── Hero Banner ────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
  <div class="hero-badge">EPIDEMIOLOGICAL CLINICAL ENGINE</div>
  <div class="hero-title">🫀 Evaluación de Riesgo Coronario a 10 Años (Framingham)</div>
  <div class="hero-subtitle">
    Plataforma interactiva basada en el histórico estudio longitudinal de Framingham. Monitoreo de factores con plausibilidad fisiológica estricta: tabaquismo, glucosa, colesterol, presión sistólica y edad con Odds Ratios positivos reales.
  </div>
</div>
""", unsafe_allow_html=True)

# ── Tarjetas de Métricas Rápidas ───────────────────────────────────
tasa_cardio = dff['cardio'].astype(int).mean() * 100 if len(dff) > 0 else 0
media_edad  = dff['age'].mean() if len(dff) > 0 else 0
media_sys   = dff['sysBP'].mean() if len(dff) > 0 else 0
media_chol  = dff['totChol'].mean() if len(dff) > 0 else 0

c1, c2, c3, c4 = st.columns(4)
c1.markdown(f"""
<div class="metric-card">
  <div class="metric-value">{len(dff):,}</div>
  <div class="metric-label">Pacientes en Filtro</div>
  <div class="metric-sub">de {len(df):,} totales</div>
</div>""", unsafe_allow_html=True)

c2.markdown(f"""
<div class="metric-card">
  <div class="metric-value" style="color:{RISK if tasa_cardio > 20 else ACCENT};">{tasa_cardio:.1f}%</div>
  <div class="metric-label">Incidencia a 10 Años</div>
  <div class="metric-sub">Cardiopatía Coronaria</div>
</div>""", unsafe_allow_html=True)

c3.markdown(f"""
<div class="metric-card">
  <div class="metric-value">{media_sys:.1f}</div>
  <div class="metric-label">Presión Sistólica Media</div>
  <div class="metric-sub">mmHg</div>
</div>""", unsafe_allow_html=True)

c4.markdown(f"""
<div class="metric-card">
  <div class="metric-value" style="color:{WARN if media_chol > 235 else ACCENT};">{media_chol:.1f}</div>
  <div class="metric-label">Colesterol Total Medio</div>
  <div class="metric-sub">mg/dL</div>
</div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Pestañas Principales ───────────────────────────────────────────
tab_calc, tab_pop, tab_target, tab_num, tab_cat, tab_corr = st.tabs([
    "🧮 Calculadora Clínica de Riesgo",
    "👥 Comparador Poblacional",
    "📊 Variable Objetivo (Incidencia 10a)",
    "📈 Biomarcadores Cuantitativos",
    "🏷️ Factores de Riesgo Clínicos",
    "🔥 Odds Ratios e Importancia"
])

# ═══════════════════════════════════════════════════════════════════
# TAB 1: Calculadora Clínica de Riesgo Individual
# ═══════════════════════════════════════════════════════════════════
with tab_calc:
    st.markdown(f'<div class="section-label">Simulador de Riesgo Coronario a 10 Años para Pacientes Reales</div>', unsafe_allow_html=True)
    st.markdown("""
    Ingresa los parámetros clínicos y hemodinámicos del paciente para calcular en tiempo real
    su **Índice de Masa Corporal (IMC)**, **Presión de Pulso (PP)**, **Presión Arterial Media (PAM)**,
    **Estadío de Hipertensión (AHA)** y la **Probabilidad de Evento Coronario a 10 Años** calibrada por Machine Learning.
    """)

    col_form, col_res = st.columns([1.1, 1.3], gap="large")

    with col_form:
        st.markdown(f"""
        <div style='background:{BG_CARD}; padding:1.2rem; border-radius:12px; border:1px solid {BORDER};'>
          <h4 style='margin-top:0; color:{ACCENT}; font-size:1.05rem;'>📋 Parámetros del Paciente</h4>
        </div>
        """, unsafe_allow_html=True)

        cf1, cf2 = st.columns(2)
        with cf1:
            p_edad = st.number_input("Edad (años)", min_value=30, max_value=75, value=52, step=1)
            p_sexo_txt = st.selectbox("Sexo Biológico", ["Femenino (Mujer)", "Masculino (Hombre)"])
            p_male = 1 if "Masculino" in p_sexo_txt else 0
            p_cigs = st.slider("Cigarrillos al día", min_value=0, max_value=60, value=0, step=1, help="0 si no fuma actualmente")
            p_smoker = 1 if p_cigs > 0 else 0
            p_totchol = st.number_input("Colesterol Total (mg/dL)", min_value=120, max_value=450, value=220, step=5)
            p_glucose = st.number_input("Glucosa en Ayunas (mg/dL)", min_value=50, max_value=300, value=95, step=5)

        with cf2:
            p_sys = st.number_input("Presión Sistólica (sysBP, mmHg)", min_value=85, max_value=220, value=130, step=1)
            p_dia = st.number_input("Presión Diastólica (diaBP, mmHg)", min_value=50, max_value=130, value=82, step=1)
            p_hr  = st.number_input("Frecuencia Cardíaca (lpm)", min_value=45, max_value=130, value=72, step=1)
            p_peso = st.number_input("Peso corporal (kg)", min_value=40.0, max_value=160.0, value=72.0, step=0.5)
            p_alt  = st.number_input("Estatura (cm)", min_value=140, max_value=205, value=168, step=1)

        st.markdown("<p style='font-size:0.85rem; color:#7BA3C4; font-weight:600; margin-top:8px;'>Antecedentes Médicos y Tratamiento:</p>", unsafe_allow_html=True)
        cs1, cs2, cs3 = st.columns(3)
        with cs1:
            p_bpmeds = 1 if st.checkbox("Usa Antihipertensivos", value=False) else 0
        with cs2:
            p_diabetes = 1 if st.checkbox("Diagnóstico Diabetes", value=False) else (1 if p_glucose >= 126 else 0)
        with cs3:
            p_stroke = 1 if st.checkbox("Antecedente de Ictus / ACV", value=False) else 0

        # Validación hemodinámica
        if p_dia >= p_sys:
            st.error("⚠️ Alerta: La presión diastólica no puede ser mayor o igual a la sistólica.")
        elif (p_sys - p_dia) < 20:
            st.warning("⚠️ Presión diferencial estrecha (<20 mmHg). Verifica la toma de presión.")

    # Cálculos fisiológicos
    p_bmi = round(p_peso / ((p_alt / 100) ** 2), 1)
    p_pp  = p_sys - p_dia
    p_map = round(p_dia + (p_pp / 3), 1)
    p_hyp = 1 if (p_sys >= 140 or p_dia >= 90 or p_bpmeds == 1) else 0

    # Categorías clínicas
    if p_bmi < 18.5:
        bmi_cat, bmi_color = "Bajo peso", SAFE
    elif p_bmi < 25.0:
        bmi_cat, bmi_color = "Normopeso", ACCENT
    elif p_bmi < 30.0:
        bmi_cat, bmi_color = "Sobrepeso", WARN
    elif p_bmi < 35.0:
        bmi_cat, bmi_color = "Obesidad Grado I", "#FF9F43"
    else:
        bmi_cat, bmi_color = "Obesidad Grado II/III", RISK

    if p_sys >= 140 or p_dia >= 90:
        aha_cat, aha_color = "Hipertensión Grado 2", RISK
    elif (130 <= p_sys <= 139) or (80 <= p_dia <= 89):
        aha_cat, aha_color = "Hipertensión Grado 1", WARN
    elif (120 <= p_sys <= 129) and (p_dia < 80):
        aha_cat, aha_color = "Presión Arterial Elevada", "#F39C12"
    else:
        aha_cat, aha_color = "Presión Arterial Normal", ACCENT

    with col_res:
        st.markdown(f"""
        <div style='background:{BG_CARD}; padding:1.2rem; border-radius:12px; border:1px solid {BORDER};'>
          <h4 style='margin-top:0; color:{ACCENT}; font-size:1.05rem;'>🩺 Diagnóstico Hemodinámico y Riesgo</h4>
        </div>
        """, unsafe_allow_html=True)

        rc1, rc2, rc3 = st.columns(3)
        rc1.markdown(f"""
        <div class="metric-card">
          <div class="metric-value" style="font-size:1.6rem; color:{bmi_color};">{p_bmi}</div>
          <div class="metric-label">IMC (kg/m²)</div>
          <div class="metric-sub" style="color:{bmi_color}; font-weight:600;">{bmi_cat}</div>
        </div>""", unsafe_allow_html=True)

        rc2.markdown(f"""
        <div class="metric-card">
          <div class="metric-value" style="font-size:1.6rem; color:{RISK if p_pp > 60 else ACCENT};">{p_pp}</div>
          <div class="metric-label">Presión de Pulso</div>
          <div class="metric-sub">Diferencial (mmHg)</div>
        </div>""", unsafe_allow_html=True)

        rc3.markdown(f"""
        <div class="metric-card">
          <div class="metric-value" style="font-size:1.6rem; color:{RISK if p_map > 105 else ACCENT};">{p_map}</div>
          <div class="metric-label">PAM (Perfusión)</div>
          <div class="metric-sub">Media (mmHg)</div>
        </div>""", unsafe_allow_html=True)

        # Inferencia del Modelo Predictivo
        proba_riesgo = 0.20
        if modelo is not None and p_sys > p_dia:
            df_paciente = pd.DataFrame([{
                'age': float(p_edad),
                'cigsPerDay': float(p_cigs),
                'totChol': float(p_totchol),
                'sysBP': float(p_sys),
                'diaBP': float(p_dia),
                'BMI': float(p_bmi),
                'heartRate': float(p_hr),
                'glucose': float(p_glucose),
                'male': int(p_male),
                'currentSmoker': int(p_smoker),
                'BPMeds': int(p_bpmeds),
                'prevalentStroke': int(p_stroke),
                'prevalentHyp': int(p_hyp),
                'diabetes': int(p_diabetes)
            }])
            try:
                proba_riesgo = modelo.predict_proba(df_paciente)[0, 1]
            except Exception as e:
                st.error(f"Error en inferencia clínica: {e}")
                proba_riesgo = 0.20

        pct_riesgo = proba_riesgo * 100
        if pct_riesgo < 15.0:
            nivel_riesgo = "BAJO RIESGO CORONARIO (<15%)"
            color_riesgo = SAFE
            desc_riesgo = "Perfil cardiovascular favorable. Mantener estilo de vida saludable y chequeo preventivo periódico."
        elif pct_riesgo < 30.0:
            nivel_riesgo = "RIESGO MODERADO (15% - 30%)"
            color_riesgo = WARN
            desc_riesgo = "Presencia de factores de riesgo ateroscleróticos. Optimizar dieta mediterránea, control tensional y actividad física."
        elif pct_riesgo < 50.0:
            nivel_riesgo = "ALTO RIESGO CORONARIO (30% - 50%)"
            color_riesgo = "#FF9F43"
            desc_riesgo = "Elevada probabilidad de sufrir angina de pecho o infarto en 10 años. Se recomienda intervención médica formal."
        else:
            nivel_riesgo = "RIESGO MUY ALTO / CRÍTICO (>50%)"
            color_riesgo = RISK
            desc_riesgo = "Múltiples factores aterogénicos activos de alto impacto. Requiere seguimiento cardiológico prioritario."

        st.markdown(f"""
        <div style='background:linear-gradient(135deg, rgba(17,34,54,0.95), rgba(15,42,66,0.98)); border:2px solid {color_riesgo}; border-radius:14px; padding:1.4rem; text-align:center; margin-top:0.6rem;'>
          <div style='font-size:0.8rem; color:{TEXT_LO}; text-transform:uppercase; letter-spacing:0.12em; font-weight:700;'>Riesgo a 10 Años de Evento Coronario (Framingham)</div>
          <div style='font-size:3.2rem; font-weight:800; color:{color_riesgo}; line-height:1.1; margin:0.3rem 0;'>{pct_riesgo:.1f}%</div>
          <div style='font-size:1.05rem; font-weight:700; color:{color_riesgo}; letter-spacing:0.05em;'>{nivel_riesgo}</div>
          <div style='font-size:0.85rem; color:{TEXT_HI}; margin-top:0.6rem; line-height:1.5;'>{desc_riesgo}</div>
        </div>
        """, unsafe_allow_html=True)

        alertas = []
        if p_sys >= 140 or p_dia >= 90:
            alertas.append(f"🚨 **Presión Arterial ({aha_cat}):** Sus valores superan el rango normotenso. Controlar ingesta de sodio y evaluar terapia antihipertensiva.")
        if p_smoker == 1:
            alertas.append(f"🚬 **Tabaquismo Activo ({p_cigs} cig/día):** El tabaco lesiona el endotelio vascular y acelera la formación de placas ateroscleróticas.")
        if p_totchol >= 240:
            alertas.append("🧈 **Hipercolesterolemia Severa (≥240 mg/dL):** Se sugiere dosaje de LDL/HDL y valoración de tratamiento con estatinas.")
        elif p_totchol >= 200:
            alertas.append("🧈 **Colesterol Limítrofe (200-239 mg/dL):** Cuidar grasas saturadas y aumentar ingesta de fibra soluble.")
        if p_glucose >= 126 or p_diabetes == 1:
            alertas.append("🩸 **Hiperglucemia / Diabetes:** El exceso de glucosa daña micro y macrovasculatura arterial aumentando el riesgo coronario en más del 50%.")
        if p_bmi >= 25.0:
            alertas.append(f"⚖️ **Índice de Masa Corporal ({bmi_cat}):** Reducir entre 5% y 10% del peso corporal normaliza la tensión arterial y mejora el perfil glucémico.")

        if alertas:
            st.markdown("<p style='font-size:0.82rem; color:#00C9A7; font-weight:700; margin-top:1rem; margin-bottom:4px;'>RECOMENDACIONES CLÍNICAS INDIVIDUALES:</p>", unsafe_allow_html=True)
            for a in alertas:
                st.markdown(f"<div style='font-size:0.83rem; color:{TEXT_HI}; background:rgba(30,58,95,0.35); padding:7px 12px; border-radius:8px; margin-bottom:5px; border-left:3px solid {ACCENT};'>{a}</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# TAB 2: Comparador Poblacional
# ═══════════════════════════════════════════════════════════════════
with tab_pop:
    st.markdown(f'<div class="section-label">Posicionamiento del Paciente vs Cohorte Framingham ({len(df):,} Registros)</div>', unsafe_allow_html=True)
    st.markdown("""
    Esta vista ubica al paciente dentro de las distribuciones reales de la población sana vs quienes sufrieron un evento coronario en los 10 años de seguimiento.
    """)

    cp1, cp2 = st.columns(2)

    with cp1:
        pct_sys = calcular_percentil(df['sysBP'], p_sys)
        fig, ax = plt.subplots(figsize=(6, 3.8))
        sns.kdeplot(df[df['cardio']==0]['sysBP'], ax=ax, color=SAFE, fill=True, alpha=0.35, label='Sin Evento Coronario (0)')
        sns.kdeplot(df[df['cardio']==1]['sysBP'], ax=ax, color=RISK, fill=True, alpha=0.35, label='Con Evento a 10a (1)')
        ax.axvline(p_sys, color=WARN, linestyle='--', linewidth=2.2, label=f'Paciente: {p_sys} mmHg (P{pct_sys:.0f})')
        ax.set_title(f'Presión Sistólica (Percentil {pct_sys:.1f})', color=TEXT_HI, fontsize=11)
        ax.set_xlabel('Presión Sistólica (mmHg)')
        ax.set_ylabel('Densidad')
        ax.legend(facecolor=BG_CARD, edgecolor=BORDER, fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with cp2:
        pct_chol = calcular_percentil(df['totChol'], p_totchol)
        fig, ax = plt.subplots(figsize=(6, 3.8))
        sns.kdeplot(df[df['cardio']==0]['totChol'], ax=ax, color=SAFE, fill=True, alpha=0.35, label='Sin Evento (0)')
        sns.kdeplot(df[df['cardio']==1]['totChol'], ax=ax, color=RISK, fill=True, alpha=0.35, label='Con Evento (1)')
        ax.axvline(p_totchol, color=WARN, linestyle='--', linewidth=2.2, label=f'Paciente: {p_totchol} mg/dL (P{pct_chol:.0f})')
        ax.set_title(f'Colesterol Total (Percentil {pct_chol:.1f})', color=TEXT_HI, fontsize=11)
        ax.set_xlabel('Colesterol Total (mg/dL)')
        ax.set_ylabel('Densidad')
        ax.legend(facecolor=BG_CARD, edgecolor=BORDER, fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    cp3, cp4 = st.columns(2)
    with cp3:
        pct_gluc = calcular_percentil(df['glucose'], p_glucose)
        fig, ax = plt.subplots(figsize=(6, 3.8))
        sns.kdeplot(df[df['cardio']==0]['glucose'], ax=ax, color=SAFE, fill=True, alpha=0.35, label='Sin Evento')
        sns.kdeplot(df[df['cardio']==1]['glucose'], ax=ax, color=RISK, fill=True, alpha=0.35, label='Con Evento')
        ax.axvline(p_glucose, color=WARN, linestyle='--', linewidth=2.2, label=f'Paciente: {p_glucose} mg/dL (P{pct_gluc:.0f})')
        ax.set_title(f'Glucosa en Ayunas (Percentil {pct_gluc:.1f})', color=TEXT_HI, fontsize=11)
        ax.set_xlabel('Glucosa (mg/dL)')
        ax.set_ylabel('Densidad')
        ax.legend(facecolor=BG_CARD, edgecolor=BORDER, fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with cp4:
        pct_bmi = calcular_percentil(df['BMI'], p_bmi)
        fig, ax = plt.subplots(figsize=(6, 3.8))
        sns.kdeplot(df[df['cardio']==0]['BMI'], ax=ax, color=SAFE, fill=True, alpha=0.35, label='Sin Evento')
        sns.kdeplot(df[df['cardio']==1]['BMI'], ax=ax, color=RISK, fill=True, alpha=0.35, label='Con Evento')
        ax.axvline(p_bmi, color=WARN, linestyle='--', linewidth=2.2, label=f'Paciente: {p_bmi} kg/m² (P{pct_bmi:.0f})')
        ax.set_title(f'Índice de Masa Corporal (Percentil {pct_bmi:.1f})', color=TEXT_HI, fontsize=11)
        ax.set_xlabel('IMC (kg/m²)')
        ax.set_ylabel('Densidad')
        ax.legend(facecolor=BG_CARD, edgecolor=BORDER, fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    resumen_percentiles = pd.DataFrame({
        'Biomarcador': ['Presión Sistólica (sysBP)', 'Presión Diastólica (diaBP)', 'Colesterol Total (totChol)', 'Glucosa (glucose)', 'IMC (BMI)', 'Edad'],
        'Valor Paciente': [f"{p_sys} mmHg", f"{p_dia} mmHg", f"{p_totchol} mg/dL", f"{p_glucose} mg/dL", f"{p_bmi} kg/m²", f"{p_edad} años"],
        'Percentil Poblacional': [
            f"P{calcular_percentil(df['sysBP'], p_sys):.1f}",
            f"P{calcular_percentil(df['diaBP'], p_dia):.1f}",
            f"P{calcular_percentil(df['totChol'], p_totchol):.1f}",
            f"P{calcular_percentil(df['glucose'], p_glucose):.1f}",
            f"P{calcular_percentil(df['BMI'], p_bmi):.1f}",
            f"P{calcular_percentil(df['age'], p_edad):.1f}"
        ],
        'Media Población Sin Evento (0)': [
            f"{df[df['cardio']==0]['sysBP'].mean():.1f} mmHg",
            f"{df[df['cardio']==0]['diaBP'].mean():.1f} mmHg",
            f"{df[df['cardio']==0]['totChol'].mean():.1f} mg/dL",
            f"{df[df['cardio']==0]['glucose'].mean():.1f} mg/dL",
            f"{df[df['cardio']==0]['BMI'].mean():.1f} kg/m²",
            f"{df[df['cardio']==0]['age'].mean():.1f} años"
        ],
        'Media Población Con Evento (1)': [
            f"{df[df['cardio']==1]['sysBP'].mean():.1f} mmHg",
            f"{df[df['cardio']==1]['diaBP'].mean():.1f} mmHg",
            f"{df[df['cardio']==1]['totChol'].mean():.1f} mg/dL",
            f"{df[df['cardio']==1]['glucose'].mean():.1f} mg/dL",
            f"{df[df['cardio']==1]['BMI'].mean():.1f} kg/m²",
            f"{df[df['cardio']==1]['age'].mean():.1f} años"
        ]
    })
    st.table(resumen_percentiles)


# ═══════════════════════════════════════════════════════════════════
# TAB 3: Variable Objetivo
# ═══════════════════════════════════════════════════════════════════
with tab_target:
    st.markdown(f'<div class="section-label">Incidencia a 10 Años de Enfermedad Coronaria (cardio / TenYearCHD)</div>', unsafe_allow_html=True)

    col_g, col_a = st.columns([1, 1])

    with col_g:
        conteo = dff['cardio'].astype(int).value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(5, 3.8))
        bars = ax.bar(
            ['Sin Evento (0)\n(Libre de ECV)', 'Con Evento (1)\n(Cardiopatía a 10a)'],
            conteo.values,
            color=[SAFE, RISK], edgecolor=BG_CARD, width=0.45
        )
        for bar, val in zip(bars, conteo.values):
            pct = val / conteo.sum() * 100
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (conteo.max()*0.02),
                    f'{val:,}\n({pct:.1f}%)', ha='center', fontsize=9.5, color=TEXT_HI, fontweight='bold')
        ax.set_ylabel('Nº de Pacientes', color=TEXT_LO)
        ax.set_title('Distribución en la cohorte filtrada', color=TEXT_HI, pad=12)
        ax.set_ylim(0, conteo.max() * 1.25)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_a:
        st.markdown(f'<div class="section-label">Contexto Epidemiológico Longitudinal</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style='color:{TEXT_LO}; font-size:0.9rem; line-height:1.7'>
          En estudios prospectivos como el <b>Framingham Heart Study</b>, los pacientes son seguidos a lo largo de 10 años.<br>
          - <b>Incidencia real:</b> ~15% desarrolla enfermedad coronaria comprobada (infarto agudo de miocardio o muerte coronaria).<br>
          - <b>Estrategia de Modelado:</b> Dado el desbalance natural (85% vs 15%), el pipeline clínico incorpora <code>class_weight='balanced'</code> y penalización L2 para evitar falsos negativos y maximizar la sensibilidad de cribado.<br>
          - <b>Métrica de Oro:</b> El <b>ROC-AUC ({metricas_modelo.get('roc_auc', 0.69):.3f})</b> evalúa la capacidad de discriminación sin verse sesgado por el desbalance.
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# TAB 4: Biomarcadores Cuantitativos
# ═══════════════════════════════════════════════════════════════════
with tab_num:
    st.markdown(f'<div class="section-label">Distribución y Discriminación de Biomarcadores Cuantitativos</div>', unsafe_allow_html=True)

    nombres_var = {
        'sysBP': 'Presión Sistólica (mmHg)',
        'diaBP': 'Presión Diastólica (mmHg)',
        'totChol': 'Colesterol Total (mg/dL)',
        'glucose': 'Glucosa en Ayunas (mg/dL)',
        'BMI': 'Índice de Masa Corporal (kg/m²)',
        'age': 'Edad (años)',
        'cigsPerDay': 'Cigarrillos al día',
        'heartRate': 'Frecuencia Cardíaca (lpm)',
        'pulse_pressure': 'Presión de Pulso (mmHg)',
        'map': 'Presión Arterial Media (mmHg)'
    }
    vars_disp = [c for c in nombres_var if c in dff.columns]

    variable = st.selectbox("Selecciona biomarcador a inspeccionar:", vars_disp, format_func=lambda x: nombres_var.get(x, x))

    col_h, col_v = st.columns([1, 1])

    with col_h:
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        dff[dff['cardio']==0][variable].dropna().hist(
            bins=35, ax=ax, alpha=0.75, color=SAFE, label='Sin Evento (cardio=0)', edgecolor='none')
        dff[dff['cardio']==1][variable].dropna().hist(
            bins=35, ax=ax, alpha=0.75, color=RISK, label='Con Evento (cardio=1)', edgecolor='none')
        ax.legend(fontsize=9, facecolor=BG_CARD, edgecolor=BORDER)
        ax.set_xlabel(nombres_var.get(variable, variable))
        ax.set_ylabel('Frecuencia')
        ax.set_title(f'Histograma — {nombres_var.get(variable, variable)}', color=TEXT_HI, pad=10)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_v:
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        palette_dict = {0: SAFE, 1: RISK, '0': SAFE, '1': RISK}
        if len(dff) > 0 and dff['cardio'].nunique() > 1:
            sns.violinplot(
                data=dff, x='cardio', y=variable, hue='cardio', ax=ax,
                palette=palette_dict, inner='box', linewidth=1.0, legend=False
            )
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['Sin Evento (0)', 'Con Evento (1)'])

        m0 = dff[dff['cardio']==0][variable].mean() if (dff['cardio']==0).any() else 0
        m1 = dff[dff['cardio']==1][variable].mean() if (dff['cardio']==1).any() else 0
        if (dff['cardio']==0).any():
            ax.axhline(m0, color=SAFE, linestyle='--', alpha=0.8, linewidth=1.3)
        if (dff['cardio']==1).any():
            ax.axhline(m1, color=RISK, linestyle='--', alpha=0.8, linewidth=1.3)

        ax.set_title(f'Media Sanos: {m0:.1f}  |  Media Evento: {m1:.1f}', color=TEXT_HI, pad=10)
        ax.set_xlabel('')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        diff = abs(m1 - m0) / m0 * 100 if m0 != 0 else 0
        st.info(f"📌 **Diferencia relativa entre medias:** `{diff:.1f}%`. Los pacientes que sufrieron cardiopatía coronaria muestran niveles significativamente superiores.")


# ═══════════════════════════════════════════════════════════════════
# TAB 5: Factores de Riesgo Clínicos
# ═══════════════════════════════════════════════════════════════════
with tab_cat:
    st.markdown(f'<div class="section-label">Prevalencia de Evento Coronario según Factores Clínicos Reales</div>', unsafe_allow_html=True)
    st.caption("Rojo = supera la tasa basal poblacional · Verde = por debajo de la media")

    cols_cat = ['currentSmoker', 'diabetes', 'prevalentHyp', 'BPMeds', 'male', 'bp_stage']
    etiquetas = {
        'male': {0: 'Mujer', 1: 'Hombre'},
        'currentSmoker': {0: 'No Fuma', 1: 'Fumador Activo'},
        'diabetes': {0: 'No Diabético', 1: 'Diabético'},
        'prevalentHyp': {0: 'Normotenso', 1: 'Hipertenso'},
        'BPMeds': {0: 'Sin Antihip.', 1: 'Con Antihip.'},
        'bp_stage': {1: 'Normal', 2: 'Elevada', 3: 'HTA Grado 1', 4: 'HTA Grado 2'}
    }
    titulos_map = {
        'male': 'Sexo Biológico',
        'currentSmoker': 'Hábito Tabáquico',
        'diabetes': 'Diagnóstico Diabetes',
        'prevalentHyp': 'Hipertensión Prevalente',
        'BPMeds': 'Uso de Antihipertensivos',
        'bp_stage': 'Estadío Presión (AHA)'
    }

    media_global = dff['cardio'].astype(int).mean() * 100

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    for ax, col in zip(axes.flatten(), cols_cat):
        tasa = dff.groupby(col)['cardio'].apply(lambda x: x.astype(int).mean() * 100)
        etiq = [etiquetas[col].get(k, str(k)) for k in tasa.index]
        clrs = [RISK if v > media_global else SAFE for v in tasa.values]

        bars = ax.bar(etiq, tasa.values, color=clrs, edgecolor=BG_CARD, width=0.52)
        ax.axhline(media_global, color='#FFD166', linestyle='--', linewidth=1.1, label=f'Media: {media_global:.1f}%')

        for bar, val in zip(bars, tasa.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.0,
                    f'{val:.1f}%', ha='center', fontsize=9, color=TEXT_HI, fontweight='bold')

        ax.set_title(titulos_map[col], color=TEXT_HI, pad=8, fontsize=11)
        ax.set_ylabel('% Evento Coronario', color=TEXT_LO, fontsize=9)
        ax.set_ylim(0, max(tasa.values)*1.35 if len(tasa)>0 else 50)
        ax.legend(fontsize=7.5, facecolor=BG_CARD, edgecolor=BORDER)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ═══════════════════════════════════════════════════════════════════
# TAB 6: Odds Ratios e Importancia
# ═══════════════════════════════════════════════════════════════════
with tab_corr:
    st.markdown(f'<div class="section-label">Correlaciones Clínicas y Odds Ratios Estandarizados</div>', unsafe_allow_html=True)

    cols_corr = [c for c in ['age', 'cigsPerDay', 'totChol', 'sysBP', 'diaBP',
                             'BMI', 'heartRate', 'glucose', 'diabetes', 'prevalentHyp', 'cardio']
                 if c in dff.columns]

    corr = dff[cols_corr].astype(float).corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    col_hm, col_bar = st.columns([1.3, 1])

    with col_hm:
        fig, ax = plt.subplots(figsize=(8, 7))
        sns.heatmap(
            corr, mask=mask, annot=True, fmt='.2f',
            cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            square=True, linewidths=0.4, linecolor=BG_MAIN,
            cbar_kws={'shrink': 0.75}, ax=ax,
            annot_kws={'size': 8, 'color': TEXT_HI}
        )
        ax.set_title('Matriz de Correlación de Pearson (Framingham Cohort)', color=TEXT_HI, pad=12)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_bar:
        st.markdown(f'<div class="section-label">Asociación Directa con Evento Coronario</div>', unsafe_allow_html=True)
        corr_target = corr['cardio'].drop('cardio').sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        clrs_bar = [RISK if v > 0 else SAFE for v in corr_target.values]
        ax.barh(corr_target.index, corr_target.values, color=clrs_bar, edgecolor=BG_CARD, height=0.65)
        ax.axvline(0, color=TEXT_MUT, linewidth=0.8)
        ax.set_xlabel('Correlación (r) con cardio')
        ax.set_title('Factores con mayor impacto directo', color=TEXT_HI, pad=10, fontsize=10)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    if metricas_modelo and 'odds_ratios' in metricas_modelo:
        st.markdown(f'<div class="section-label">Odds Ratios Clínicos Estandarizados (Multivariados)</div>', unsafe_allow_html=True)
        ors = pd.Series(metricas_modelo['odds_ratios']).sort_values(ascending=True)

        fig, ax = plt.subplots(figsize=(10, 4.5))
        clrs_or = [RISK if v > 1.0 else SAFE for v in ors.values]
        ax.barh(ors.index, ors.values, color=clrs_or, edgecolor=BG_CARD)
        ax.axvline(1.0, color='#FFD166', linewidth=1.5, linestyle='--', label='OR = 1.0 (Sin Efecto)')
        for i, (idx, val) in enumerate(ors.items()):
            ax.text(val + 0.02, i, f'{val:.2f}x', va='center', fontsize=9, color=TEXT_HI, fontweight='bold')
        ax.set_xlabel('Odds Ratio (OR)')
        ax.set_title('Impacto en la razón de momios de sufrir enfermedad coronaria ajustado por covariables', color=TEXT_HI, pad=10)
        ax.legend(fontsize=8, facecolor=BG_CARD, edgecolor=BORDER)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ── Footer ─────────────────────────────────────────────────────────
st.markdown(f"""
<div style='margin-top:2.5rem; padding:1.2rem 0; border-top:1px solid {BORDER};
     text-align:center; color:{TEXT_MUT}; font-size:0.82rem;'>
  <b>Framingham AI Clinical Studio</b> — Machine Learning & Modelado Epidemiológico Cardiovascular · 
  Autora: <b>Ana Colina Arismendi</b> · 
  Dataset de Referencia: <b>Framingham Heart Study (NIH / NHLBI)</b>
</div>
""", unsafe_allow_html=True)
