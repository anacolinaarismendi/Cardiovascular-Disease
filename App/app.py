import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ── Configuración de página ──────────────────────────────────────
st.set_page_config(
    page_title="Riesgo Cardiovascular",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Estilos ───────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #1e1e2e;
        border: 1px solid #313244;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #cba6f7;
    }
    .metric-label {
        font-size: 0.78rem;
        color: #a6adc8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 4px;
    }
    .metric-sub {
        font-size: 0.85rem;
        color: #6c7086;
        margin-top: 2px;
    }
</style>
""", unsafe_allow_html=True)

# ── Carga de datos ────────────────────────────────────────────────
@st.cache_data
def cargar_datos():
    rutas = [
        '../Data/processed/processed.csv',
        '../Data/cardio_train.csv',
    ]
    for ruta in rutas:
        if os.path.exists(ruta):
            sep = ';' if ruta.endswith('cardio_train.csv') else ','
            df = pd.read_csv(ruta, sep=sep)
            if 'age' in df.columns and 'age_years' not in df.columns:
                df['age_years'] = (df['age'] / 365.25).round(1)
            if 'id' in df.columns:
                df = df.drop(columns=['id'])
            return df
    st.error("No se encontró el dataset.")
    st.stop()

df = cargar_datos()

# ── Sidebar con filtros ──────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🫀 Filtros")

    rango_edad = st.slider(
        "Edad (años)",
        int(df['age_years'].min()),
        int(df['age_years'].max()),
        (int(df['age_years'].min()), int(df['age_years'].max()))
    )

    genero_sel     = st.selectbox("Género",     ['Todos', 'Mujer', 'Hombre'])
    cardio_sel     = st.selectbox("Cardio",     ['Todos', 'Sin enfermedad (0)', 'Con enfermedad (1)'])
    colesterol_sel = st.selectbox("Colesterol", ['Todos', 'Normal', 'Alto', 'Muy alto'])

# ── Aplicar filtros ───────────────────────────────────────────────
dff = df.copy()
dff = dff[(dff['age_years'] >= rango_edad[0]) & (dff['age_years'] <= rango_edad[1])]

if genero_sel == 'Mujer':
    dff = dff[dff['gender'] == 1]
elif genero_sel == 'Hombre':
    dff = dff[dff['gender'] == 2]

if cardio_sel == 'Sin enfermedad (0)':
    dff = dff[dff['cardio'] == 0]
elif cardio_sel == 'Con enfermedad (1)':
    dff = dff[dff['cardio'] == 1]

if colesterol_sel == 'Normal':
    dff = dff[dff['cholesterol'] == 1]
elif colesterol_sel == 'Alto':
    dff = dff[dff['cholesterol'] == 2]
elif colesterol_sel == 'Muy alto':
    dff = dff[dff['cholesterol'] == 3]

# ── Métricas ──────────────────────────────────────────────────────
tasa_cardio = dff['cardio'].astype(int).mean() * 100
media_edad  = dff['age_years'].mean()
media_ap_hi = dff['ap_hi'].mean()

c1, c2, c3 = st.columns(3)

c1.markdown(f"""
<div class="metric-card">
    <div class="metric-value">{len(dff):,}</div>
    <div class="metric-label">Pacientes</div>
    <div class="metric-sub">en selección</div>
</div>""", unsafe_allow_html=True)

c2.markdown(f"""
<div class="metric-card">
    <div class="metric-value">{tasa_cardio:.1f}%</div>
    <div class="metric-label">Tasa cardio=1</div>
    <div class="metric-sub">con enfermedad</div>
</div>""", unsafe_allow_html=True)

c3.markdown(f"""
<div class="metric-card">
    <div class="metric-value">{media_edad:.1f}</div>
    <div class="metric-label">Edad media</div>
    <div class="metric-sub">años</div>
</div>""", unsafe_allow_html=True)

# ── Tabs (se crean ANTES de usarse) ───────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Variable objetivo",
    "📈 Numéricas",
    "🏷️ Categóricas",
    "🔥 Correlaciones"
])

with tab1:
    conteo = dff['cardio'].astype(int).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(['Sin enfermedad', 'Con enfermedad'], conteo.values, color=['#89b4fa', '#f38ba8'])
    st.pyplot(fig)
    plt.close()

    ratio = conteo.max() / conteo.min()
    if ratio < 1.5:
        st.success(f"✓ Dataset balanceado (ratio {ratio:.2f})")
    elif ratio < 3:
        st.warning(f"⚠ Leve desequilibrio (ratio {ratio:.2f})")
    else:
        st.error(f"✗ Desequilibrio severo (ratio {ratio:.2f})")

with tab2:
    variable = st.selectbox("Variable", ['age_years', 'weight', 'ap_hi', 'ap_lo', 'bmi'])
    fig, ax = plt.subplots(figsize=(10, 4))
    dff[dff['cardio']==0][variable].hist(bins=40, ax=ax, alpha=0.7, color='#89b4fa', label='cardio=0')
    dff[dff['cardio']==1][variable].hist(bins=40, ax=ax, alpha=0.7, color='#f38ba8', label='cardio=1')
    ax.legend()
    st.pyplot(fig)
    plt.close()

with tab3:
    cols_cat = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
    etiquetas = {
        'gender':      {1: 'Mujer', 2: 'Hombre'},
        'cholesterol': {1: 'Normal', 2: 'Alto', 3: 'Muy alto'},
        'gluc':        {1: 'Normal', 2: 'Alto', 3: 'Muy alto'},
        'smoke':       {0: 'No fuma', 1: 'Fuma'},
        'alco':        {0: 'No bebe', 1: 'Bebe'},
        'active':      {0: 'Sedentario', 1: 'Activo'},
    }
    nombres = {
        'gender': 'Género', 'cholesterol': 'Colesterol', 'gluc': 'Glucosa',
        'smoke': 'Tabaco', 'alco': 'Alcohol', 'active': 'Actividad física'
    }

    st.markdown("Tasa de enfermedad cardiovascular por categoría")

    media_global = dff['cardio'].astype(int).mean() * 100

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    for ax, col in zip(axes.flatten(), cols_cat):
        tasa = dff.groupby(col)['cardio'].apply(lambda x: x.astype(int).mean() * 100)
        etiq = [etiquetas[col].get(k, str(k)) for k in tasa.index]
        colores = ['#f38ba8' if v > media_global else '#89b4fa' for v in tasa.values]

        bars = ax.bar(etiq, tasa.values, color=colores, edgecolor='white')
        ax.axhline(media_global, color='gray', linestyle='--', linewidth=1.2,
                   label=f'Media: {media_global:.1f}%')

        for bar, val in zip(bars, tasa.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                    f'{val:.1f}%', ha='center', fontsize=8)

        ax.set_title(nombres[col])
        ax.set_ylabel('% cardio=1')
        ax.set_ylim(0, 85)
        ax.legend(fontsize=7)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with tab4:
    cols_corr = ['age_years', 'height', 'weight', 'ap_hi', 'ap_lo', 'bmi',
                 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'cardio']

    corr = dff[cols_corr].astype(float).corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    st.markdown("Mapa de correlaciones")

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
                cmap='coolwarm', center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.5, ax=ax)
    st.pyplot(fig)
    plt.close()

    st.markdown("Correlación de cada variable con cardio")

    corr_target = corr['cardio'].drop('cardio').sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    colores_bar = ['#f38ba8' if v > 0 else '#89b4fa' for v in corr_target.values]
    ax.barh(corr_target.index, corr_target.values, color=colores_bar)
    ax.axvline(0, color='gray', linewidth=0.8)
    ax.set_xlabel('Correlación con cardio')
    st.pyplot(fig)
    plt.close()