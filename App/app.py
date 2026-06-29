import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.set_page_config(
    page_title="Riesgo Cardiovascular",
    page_icon="🫀",
    layout="wide",                  # ocupa todo el ancho de la pantalla
    initial_sidebar_state="expanded"
)
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
    .section-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #cdd6f4;
        border-left: 3px solid #cba6f7;
        padding-left: 10px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)
@st.cache_data                      # ← guarda en memoria, no recarga en cada interacción
def cargar_datos():
    rutas = [
        '../Data/processed/processed.csv',
        '../Data/cardio_train.csv',
    ]
    for ruta in rutas:
        if os.path.exists(ruta):
            sep = ';' if ruta.endswith('cardio_train.csv') else ','
            df = pd.read_csv(ruta, sep=sep)
            # limpieza mínima si viene del raw
            if 'age' in df.columns and 'age_years' not in df.columns:
                df['age_years'] = (df['age'] / 365.25).round(1)
            if 'id' in df.columns:
                df = df.drop(columns=['id'])
            return df
    st.error("No se encontró el dataset.")
    st.stop()

df = cargar_datos()
with st.sidebar:
    st.markdown("## 🫀 Filtros")

    rango_edad = st.slider(
        "Edad (años)",
        int(df['age_years'].min()),
        int(df['age_years'].max()),
        (int(df['age_years'].min()), int(df['age_years'].max()))  # valor inicial = rango completo
    )

    genero_sel     = st.selectbox("Género",     ['Todos', 'Mujer', 'Hombre'])
    cardio_sel     = st.selectbox("Cardio",     ['Todos', 'Sin enfermedad (0)', 'Con enfermedad (1)'])
    colesterol_sel = st.selectbox("Colesterol", ['Todos', 'Normal', 'Alto', 'Muy alto'])

    dff = df.copy()      # dff = dataframe filtrado, df = original sin tocar

dff = dff[(dff['age_years'] >= rango_edad[0]) & (dff['age_years'] <= rango_edad[1])]

if genero_sel == 'Mujer':
    dff = dff[dff['gender'] == 1]
elif genero_sel == 'Hombre':
    dff = dff[dff['gender'] == 2]

if cardio_sel == 'Sin enfermedad (0)':
    dff = dff[dff['cardio'] == 0]
elif cardio_sel == 'Con enfermedad (1)':
    dff = dff[dff['cardio'] == 1]

    tasa_cardio = dff['cardio'].astype(int).mean() * 100
media_edad  = dff['age_years'].mean()
media_ap_hi = dff['ap_hi'].mean()

c1, c2, c3 = st.columns(3)     # divide la pantalla en 3 columnas iguales

c1.markdown(f"""
<div class="metric-card">
    <div class="metric-value">{len(dff):,}</div>
    <div class="metric-label">Pacientes</div>
    <div class="metric-sub">en selección</div>
</div>""", unsafe_allow_html=True)

c2.markdown(f"""
<div class="metric-card">
    <div class="metric-label">Tasa cardio=1</div>
    <div class="metric-sub">con enfermedad</div>
</div>""", unsafe_allow_html=True)

c3.markdown(f"""
<div class="metric-card">
    <div class="metric-value">{media_edad:.1f}</div>
    <div class="metric-label">Edad media</div>
    <div class="metric-sub">años</div>
</div>""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Variable objetivo",
    "📈 Numéricas",
    "🏷️ Categóricas",
    "🔥 Correlaciones"
])

with tab1:
    # todo lo que pongas aquí aparece solo cuando el usuario hace clic en esta pestaña
    conteo = dff['cardio'].astype(int).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(['Sin enfermedad', 'Con enfermedad'], conteo.values, color=['#89b4fa', '#f38ba8'])
    st.pyplot(fig)          # ← así se muestra un gráfico matplotlib en Streamlit
    plt.close()             # ← siempre cerrar para liberar memoria

with tab2:
    variable = st.selectbox("Variable", ['age_years', 'weight', 'ap_hi', 'ap_lo', 'bmi'])
    fig, ax = plt.subplots(figsize=(10, 4))
    dff[dff['cardio']==0][variable].hist(bins=40, ax=ax, alpha=0.7, color='#89b4fa', label='cardio=0')
    dff[dff['cardio']==1][variable].hist(bins=40, ax=ax, alpha=0.7, color='#f38ba8', label='cardio=1')
    ax.legend()
    st.pyplot(fig)
    plt.close()

    ratio = conteo.max() / conteo.min()

if ratio < 1.5:
    st.success(f"✓ Dataset balanceado (ratio {ratio:.2f})")   # verde
elif ratio < 3:
    st.warning(f"⚠ Leve desequilibrio (ratio {ratio:.2f})")   # amarillo
else:
    st.error(f"✗ Desequilibrio severo (ratio {ratio:.2f})")   # rojo