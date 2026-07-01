import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os

# ── Configuración ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Riesgo Cardiovascular",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Paleta de diseño ───────────────────────────────────────────────
# Inspirada en monitores médicos: fondo dark navy, señal ECG en verde lima,
# riesgo en rojo ámbar, neutro en azul acero

BG_MAIN   = "#0D1B2A"   # navy profundo
BG_CARD   = "#112236"   # navy medio para tarjetas
BG_PANEL  = "#0A1628"   # navy oscuro para sidebar
ACCENT    = "#00C9A7"   # verde señal ECG
RISK      = "#FF6B6B"   # rojo ámbar suave
SAFE      = "#4ECDC4"   # teal apagado
BORDER    = "#1E3A5F"   # borde sutil
TEXT_HI   = "#E8F4FD"   # texto principal
TEXT_LO   = "#7BA3C4"   # texto secundario
TEXT_MUT  = "#3D6B8A"   # texto muy apagado

# ── Estilos CSS ────────────────────────────────────────────────────
st.markdown(f"""
<style>
  /* Fondo global */
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

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {{
      background-color: {BG_CARD};
      border-radius: 10px;
      padding: 4px;
      gap: 4px;
  }}
  .stTabs [data-baseweb="tab"] {{
      background-color: transparent;
      color: {TEXT_LO} !important;
      border-radius: 8px;
      padding: 8px 18px;
      font-size: 14px;
      font-weight: 500;
  }}
  .stTabs [aria-selected="true"] {{
      background-color: {BG_MAIN} !important;
      color: {ACCENT} !important;
      border-bottom: 2px solid {ACCENT} !important;
  }}

  /* Tarjetas de métrica */
  .metric-card {{
      background: linear-gradient(135deg, {BG_CARD} 0%, #0F2A42 100%);
      border: 1px solid {BORDER};
      border-radius: 14px;
      padding: 1.4rem 1.6rem;
      text-align: center;
      position: relative;
      overflow: hidden;
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
      font-size: 2.2rem;
      font-weight: 700;
      color: {ACCENT};
      line-height: 1.1;
      letter-spacing: -0.02em;
  }}
  .metric-label {{
      font-size: 0.72rem;
      color: {TEXT_LO};
      text-transform: uppercase;
      letter-spacing: 0.1em;
      margin-top: 5px;
  }}
  .metric-sub {{
      font-size: 0.82rem;
      color: {TEXT_MUT};
      margin-top: 3px;
  }}

  /* Header hero */
  .hero {{
      background: linear-gradient(135deg, {BG_CARD} 0%, #0A2040 60%, #0D1B2A 100%);
      border: 1px solid {BORDER};
      border-radius: 16px;
      padding: 2rem 2.5rem;
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
      font-size: 1rem;
      color: {TEXT_LO};
      margin-top: 6px;
  }}
  .hero-badge {{
      display: inline-block;
      background: rgba(0,201,167,0.12);
      border: 1px solid rgba(0,201,167,0.3);
      color: {ACCENT};
      font-size: 0.75rem;
      padding: 3px 10px;
      border-radius: 999px;
      margin-bottom: 10px;
      letter-spacing: 0.06em;
      font-weight: 600;
  }}

  /* ECG decorativo en hero */
  .ecg-line {{
      position: absolute;
      right: 2rem;
      top: 50%;
      transform: translateY(-50%);
      opacity: 0.15;
      font-size: 5rem;
      color: {ACCENT};
  }}

  /* Section labels */
  .section-label {{
      font-size: 0.72rem;
      color: {ACCENT};
      text-transform: uppercase;
      letter-spacing: 0.12em;
      font-weight: 600;
      margin-bottom: 0.8rem;
      padding-left: 10px;
      border-left: 2px solid {ACCENT};
  }}

  /* Textos globales */
  p, span, div, label {{
      color: {TEXT_HI};
  }}
  h1, h2, h3 {{
      color: {TEXT_HI} !important;
  }}

  /* Slider y selectbox */
  .stSlider > div > div > div > div {{
      background: {ACCENT} !important;
  }}
</style>
""", unsafe_allow_html=True)

# ── Carga de datos ─────────────────────────────────────────────────
@st.cache_data
def cargar_datos():
    ruta = '../Data/processed/processed.csv'
    if not os.path.exists(ruta):
        st.error(f"No se encontró el archivo en: {ruta}")
        st.info("Asegúrate de que processed.csv esté en Data/processed/")
        st.stop()
    return pd.read_csv(ruta)

df = cargar_datos()

# ── Matplotlib: tema oscuro consistente con la app ─────────────────
plt.rcParams.update({
    'figure.facecolor':  BG_CARD,
    'axes.facecolor':    BG_CARD,
    'axes.edgecolor':    BORDER,
    'axes.labelcolor':   TEXT_LO,
    'xtick.color':       TEXT_LO,
    'ytick.color':       TEXT_LO,
    'text.color':        TEXT_HI,
    'grid.color':        BORDER,
    'grid.alpha':        0.5,
})

# ── Sidebar ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style='text-align:center; padding: 1rem 0 1.5rem;'>
      <div style='font-size:2.5rem'>🫀</div>
      <div style='font-size:1rem; font-weight:700; color:{TEXT_HI}'>CardioRisk</div>
      <div style='font-size:0.75rem; color:{TEXT_LO}'>Explorador de datos</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"<div style='font-size:0.72rem; color:{ACCENT}; text-transform:uppercase; letter-spacing:0.1em; font-weight:600; margin-bottom:0.5rem'>Filtros</div>", unsafe_allow_html=True)

    rango_edad = st.slider(
        "Edad (años)",
        int(df['age_years'].min()),
        int(np.ceil(df['age_years'].max())),
        (int(df['age_years'].min()), int(np.ceil(df['age_years'].max())))
    )

    genero_sel     = st.selectbox("Género",     ['Todos', 'Mujer', 'Hombre'])
    colesterol_sel = st.selectbox("Colesterol", ['Todos', 'Normal', 'Alto', 'Muy alto'])

    st.markdown("---")
    st.markdown(f"<div style='font-size:0.72rem; color:{TEXT_MUT}'>Dataset: 68,640 pacientes<br>Fuente: Kaggle — sulianova</div>", unsafe_allow_html=True)

# ── Aplicar filtros ────────────────────────────────────────────────
dff = df.copy()
dff = dff[(dff['age_years'] >= rango_edad[0]) & (dff['age_years'] <= rango_edad[1])]

if genero_sel == 'Mujer':
    dff = dff[dff['gender'] == 1]
elif genero_sel == 'Hombre':
    dff = dff[dff['gender'] == 2]

if colesterol_sel == 'Normal':
    dff = dff[dff['cholesterol'] == 1]
elif colesterol_sel == 'Alto':
    dff = dff[dff['cholesterol'] == 2]
elif colesterol_sel == 'Muy alto':
    dff = dff[dff['cholesterol'] == 3]

# ── Hero header ────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
  <div class="hero-badge">Studio</div>
  <div class="hero-title">🫀 Predicción de Riesgo Cardiovascular</div>
  <div class="hero-subtitle">
    Explorador interactivo — 68,640 pacientes · Kaggle Cardiovascular Disease Dataset
  </div>
  <div class="ecg-line">∿</div>
</div>
""", unsafe_allow_html=True)

# ── Métricas ───────────────────────────────────────────────────────
tasa_cardio  = dff['cardio'].astype(int).mean() * 100
media_edad   = dff['age_years'].mean()
media_ap_hi  = dff['ap_hi'].mean()
pct_hipert   = dff['hypertension'].mean() * 100 if 'hypertension' in dff.columns else (((dff['ap_hi']>=140)|(dff['ap_lo']>=90)).mean()*100)

c1, c2, c3, c4 = st.columns(4)
for col, valor, label, sub in [
    (c1, f"{len(dff):,}",       "Pacientes",          "en la selección"),
    (c2, f"{tasa_cardio:.1f}%", "Tasa de enfermedad", "cardio = 1"),
    (c3, f"{media_edad:.1f}",   "Edad media",         "años"),
    (c4, f"{pct_hipert:.1f}%",  "Hipertensos",        "ap_hi≥140 o ap_lo≥90"),
]:
    col.markdown(f"""
    <div class="metric-card">
      <div class="metric-value">{valor}</div>
      <div class="metric-label">{label}</div>
      <div class="metric-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Variable objetivo",
    "📈 Numéricas",
    "🏷️ Categóricas",
    "🔥 Correlaciones"
])

# ─── TAB 1: Variable objetivo ──────────────────────────────────────
with tab1:
    st.markdown(f'<div class="section-label">Balance de cardio en la selección</div>', unsafe_allow_html=True)

    col_g, col_a = st.columns([1, 1])

    with col_g:
        conteo = dff['cardio'].astype(int).value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(5, 4))
        bars = ax.bar(
            ['Sin enfermedad\n(cardio=0)', 'Con enfermedad\n(cardio=1)'],
            conteo.values,
            color=[SAFE, RISK], edgecolor=BG_CARD, width=0.5
        )
        for bar, val in zip(bars, conteo.values):
            pct = val / conteo.sum() * 100
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                    f'{val:,}\n({pct:.1f}%)', ha='center', fontsize=10, color=TEXT_HI)
        ax.set_ylabel('Pacientes', color=TEXT_LO)
        ax.set_title('Distribución de cardio', color=TEXT_HI, pad=12)
        ax.set_ylim(0, conteo.max() * 1.25)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_a:
        ratio = conteo.max() / conteo.min()
        st.markdown(f'<div class="section-label">Diagnóstico del balance</div>', unsafe_allow_html=True)

        if ratio < 1.5:
            st.success(f"✓ Dataset balanceado — ratio {ratio:.2f}")
        elif ratio < 3:
            st.warning(f"⚠ Leve desequilibrio — ratio {ratio:.2f}")
        else:
            st.error(f"✗ Desequilibrio severo — ratio {ratio:.2f}")

        st.markdown(f"""
        <div style='margin-top:1rem; color:{TEXT_LO}; font-size:0.9rem; line-height:1.7'>
          Un dataset balanceado significa que el modelo no puede hacer trampa
          prediciendo siempre la clase mayoritaria.<br><br>
          Un ratio cercano a <b style='color:{ACCENT}'>1.0</b> indica que hay
          aproximadamente el mismo número de pacientes con y sin enfermedad.<br><br>
          En este dataset no se necesita SMOTE ni corrección — basta con usar
          <code style='color:{ACCENT}'>stratify=y</code> al dividir train/test.
        </div>
        """, unsafe_allow_html=True)

# ─── TAB 2: Numéricas ──────────────────────────────────────────────
with tab2:
    st.markdown(f'<div class="section-label">Distribución por grupo</div>', unsafe_allow_html=True)

    nombres_var = {
        'age_years': 'Edad (años)',
        'weight':    'Peso (kg)',
        'ap_hi':     'Presión sistólica (mmHg)',
        'ap_lo':     'Presión diastólica (mmHg)',
        'bmi':       'IMC',
        'pulse_pressure': 'Presión diferencial'
    }
    vars_disp = [c for c in nombres_var if c in dff.columns]

    variable = st.selectbox(
        "Variable a explorar",
        vars_disp,
        format_func=lambda x: nombres_var.get(x, x)
    )

    col_h, col_v = st.columns([1, 1])

    with col_h:
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        dff[dff['cardio']==0][variable].hist(
            bins=40, ax=ax, alpha=0.75, color=SAFE, label='Sin enfermedad (0)', edgecolor='none')
        dff[dff['cardio']==1][variable].hist(
            bins=40, ax=ax, alpha=0.75, color=RISK, label='Con enfermedad (1)', edgecolor='none')
        ax.legend(fontsize=9, facecolor=BG_CARD, edgecolor=BORDER)
        ax.set_xlabel(nombres_var.get(variable, variable))
        ax.set_title(f'Distribución — {nombres_var.get(variable, variable)}', color=TEXT_HI, pad=10)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_v:
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        sns.violinplot(
            data=dff, x='cardio', y=variable, ax=ax,
            hue='cardio', palette={0: SAFE, 1: RISK},
            inner='box', linewidth=0.8, legend=False
        )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Sin enfermedad', 'Con enfermedad'])
        m0 = dff[dff['cardio']==0][variable].mean()
        m1 = dff[dff['cardio']==1][variable].mean()
        ax.axhline(m0, color=SAFE, linestyle='--', alpha=0.6, linewidth=1.2)
        ax.axhline(m1, color=RISK, linestyle='--', alpha=0.6, linewidth=1.2)
        ax.set_title(f'Media 0: {m0:.1f}  |  Media 1: {m1:.1f}', color=TEXT_HI, pad=10)
        ax.set_xlabel('')
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        diff = abs(m1 - m0) / m0 * 100
        if diff > 10:
            st.info(f"📌 Diferencia entre grupos: **{diff:.1f}%** — variable con alto poder predictivo.")
        else:
            st.caption(f"Diferencia entre grupos: {diff:.1f}%")

# ─── TAB 3: Categóricas ────────────────────────────────────────────
with tab3:
    st.markdown(f'<div class="section-label">Tasa de cardio=1 por categoría</div>', unsafe_allow_html=True)
    st.caption("Rojo = supera la media global · Azul = está por debajo")

    cols_cat = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
    etiquetas = {
        'gender':      {1: 'Mujer', 2: 'Hombre'},
        'cholesterol': {1: 'Normal', 2: 'Alto', 3: 'Muy alto'},
        'gluc':        {1: 'Normal', 2: 'Alto', 3: 'Muy alto'},
        'smoke':       {0: 'No fuma', 1: 'Fuma'},
        'alco':        {0: 'No bebe', 1: 'Bebe'},
        'active':      {0: 'Sedentario', 1: 'Activo'},
    }
    nombres_cat = {
        'gender': 'Género', 'cholesterol': 'Colesterol', 'gluc': 'Glucosa',
        'smoke': 'Tabaco', 'alco': 'Alcohol', 'active': 'Actividad física'
    }

    media_global = dff['cardio'].astype(int).mean() * 100

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, col in zip(axes.flatten(), cols_cat):
        tasa  = dff.groupby(col)['cardio'].apply(lambda x: x.astype(int).mean() * 100)
        etiq  = [etiquetas[col].get(k, str(k)) for k in tasa.index]
        clrs  = [RISK if v > media_global else SAFE for v in tasa.values]

        bars = ax.bar(etiq, tasa.values, color=clrs, edgecolor=BG_CARD, width=0.55)
        ax.axhline(media_global, color=TEXT_MUT, linestyle='--', linewidth=1.2,
                   label=f'Media: {media_global:.1f}%')

        for bar, val in zip(bars, tasa.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                    f'{val:.1f}%', ha='center', fontsize=9, color=TEXT_HI)

        ax.set_title(nombres_cat[col], color=TEXT_HI, pad=8)
        ax.set_ylabel('% con cardio=1', color=TEXT_LO, fontsize=9)
        ax.set_ylim(0, 85)
        ax.legend(fontsize=7, facecolor=BG_CARD, edgecolor=BORDER)
        ax.tick_params(axis='x', labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ─── TAB 4: Correlaciones ──────────────────────────────────────────
with tab4:
    st.markdown(f'<div class="section-label">Mapa de correlaciones</div>', unsafe_allow_html=True)

    cols_corr = [c for c in ['age_years','height','weight','ap_hi','ap_lo',
                              'bmi','cholesterol','gluc','smoke','alco','active','cardio']
                 if c in dff.columns]

    corr = dff[cols_corr].astype(float).corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    col_hm, col_bar = st.columns([1.4, 1])

    with col_hm:
        fig, ax = plt.subplots(figsize=(8, 7))
        sns.heatmap(
            corr, mask=mask, annot=True, fmt='.2f',
            cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            square=True, linewidths=0.4, linecolor=BG_MAIN,
            cbar_kws={'shrink': 0.75}, ax=ax,
            annot_kws={'size': 8, 'color': TEXT_HI}
        )
        ax.set_title('Correlación de Pearson entre variables', color=TEXT_HI, pad=12)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_bar:
        st.markdown(f'<div class="section-label">Correlación con cardio</div>', unsafe_allow_html=True)

        corr_target = corr['cardio'].drop('cardio').sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        clrs_bar = [RISK if v > 0 else SAFE for v in corr_target.values]
        ax.barh(corr_target.index, corr_target.values, color=clrs_bar, edgecolor=BG_CARD, height=0.65)
        ax.axvline(0, color=TEXT_MUT, linewidth=0.8)
        ax.set_xlabel('Correlación con cardio', color=TEXT_LO)
        ax.set_title('¿Qué variables se asocian más\ncon enfermedad cardiovascular?',
                     color=TEXT_HI, pad=10, fontsize=10)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        top_var = corr_target.idxmax()
        top_val = corr_target.max()
        st.markdown(f"""
        <div style='margin-top:1rem; color:{TEXT_LO}; font-size:0.88rem; line-height:1.7'>
          Variable más correlada con <code style='color:{ACCENT}'>cardio</code>:<br>
          <b style='color:{ACCENT}'>{top_var}</b> (r = {top_val:.3f})<br><br>
          Ninguna variable supera 0.5 — el riesgo cardiovascular
          es <b>multifactorial</b>, no depende de una sola medición.
        </div>
        """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────
st.markdown(f"""
<div style='margin-top:2rem; padding:1rem 0; border-top:1px solid {BORDER};
     text-align:center; color:{TEXT_MUT}; font-size:0.8rem;'>
  Ana Colina Arismendi 
  <a href='https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset'
     style='color:{ACCENT}; text-decoration:none;'>Dataset en Kaggle</a>
</div>
""", unsafe_allow_html=True)