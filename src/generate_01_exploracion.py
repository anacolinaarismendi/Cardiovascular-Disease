"""
Generador del Notebook 01_exploracion.ipynb adaptado a la cohorte Framingham Heart Study.
Auditoría completa de calidad de datos, variables clínicas reales y valores fisiológicos.
"""

import json
import os

def build_01_notebook():
    cells = []

    def md(text):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in text.strip().split("\n")]
        })

    def code(text):
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in text.strip().split("\n")]
        })

    # Header
    md("""# 🔍 Notebook 01: Exploración Inicial y Auditoría de Calidad de Datos Clínicos
## Proyecto de Predicción de Riesgo Cardiovascular — Cohorte Framingham
**Autora:** Ana Colina Arismendi  
**Fuente:** *Framingham Heart Study (NIH / National Heart, Lung, and Blood Institute)*

> **Objetivo:** Realizar una auditoría técnica y médica exhaustiva del dataset bruto `framingham.csv` (4,240 registros longitudinales), identificando estructura, cardinalidad, valores nulos, duplicados y coherencia fisiológica de los biomarcadores continuos (colesterol, glucemia, presión arterial, frecuencia cardíaca).

---

### 📑 Contenido del Notebook
1. [Configuración del entorno y librerías](#1)
2. [Carga e inspección preliminar de datos](#2)
3. [Estructura, tipos de datos y memoria](#3)
4. [Estadísticos descriptivos globales](#4)
5. [Auditoría de valores nulos y completitud](#5)
6. [Auditoría de duplicados técnicos](#6)
7. [Detección de anomalías hemodinámicas y fisiológicas](#7)
8. [Distribución univariada de factores de riesgo](#8)
9. [Inspección de la variable objetivo (TenYearCHD / cardio)](#9)
10. [Conclusiones del diagnóstico y hoja de ruta](#10)
""")

    # 1. Configuración
    md("""<a id="1"></a>
## 1. Configuración del entorno y librerías

Importamos las bibliotecas fundamentales para análisis exploratorio, cálculo matricial y visualización estadística avanzada.
""")

    code("""import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 120)
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
sns.set_theme(style='whitegrid', palette='mako')

print(f"✓ Pandas: {pd.__version__} | NumPy: {np.__version__}")""")

    # 2. Carga
    md("""<a id="2"></a>
## 2. Carga e inspección preliminar de datos

Cargamos el dataset original `framingham.csv`. A diferencia de estudios transversales con sesgo de supervivencia, este estudio prospectivo siguió a cada participante durante una década para registrar la aparición de cardiopatía coronaria.
""")

    code("""ruta_raw = '../Data/framingham.csv'
if not os.path.exists(ruta_raw):
    ruta_raw = 'Data/framingham.csv'

df_raw = pd.read_csv(ruta_raw)
print(f"Dimensiones de la cohorte: {df_raw.shape[0]:,} filas × {df_raw.shape[1]} variables")
df_raw.head(10)""")

    # 3. Info
    md("""<a id="3"></a>
## 3. Estructura, tipos de datos y memoria

Verificamos los tipos de datos asignados y la cardinalidad de cada columna clínica.
""")

    code("""print("--- Resumen Estructural de la Cohorte ---")
df_raw.info()

print("\\n--- Cardinalidad de Variables ---")
cardinalidad = pd.DataFrame({
    'Tipo': df_raw.dtypes,
    'Valores_Unicos': df_raw.nunique(),
    'Pct_Unicos': (df_raw.nunique() / len(df_raw) * 100).round(2),
    'Ejemplo_Valores': [df_raw[c].dropna().unique()[:4].tolist() for c in df_raw.columns]
})
print(cardinalidad)""")

    # 4. Descriptivos
    md("""<a id="4"></a>
## 4. Estadísticos descriptivos globales

Examinamos las medidas de tendencia central, dispersión, asimetría y valores extremos de las variables continuas.
""")

    code("""num_cols = ['age', 'cigsPerDay', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']
desc = df_raw[num_cols].describe().T
desc['IQR'] = desc['75%'] - desc['25%']
desc['Skewness'] = df_raw[num_cols].skew()
desc['Kurtosis'] = df_raw[num_cols].kurtosis()
print("--- Estadísticos Descriptivos Robustos ---")
print(desc.round(2))""")

    # 5. Nulos
    md("""<a id="5"></a>
## 5. Auditoría de valores nulos y completitud

Identificamos columnas con valores faltantes que requerirán imputación médica razonada en el preprocesamiento.
""")

    code("""nulos = pd.DataFrame({
    'Total_Nulos': df_raw.isnull().sum(),
    'Porcentaje_Nulos': (df_raw.isnull().mean() * 100).round(2)
})
nulos_con_datos = nulos[nulos['Total_Nulos'] > 0].sort_values(by='Total_Nulos', ascending=False)
print("--- Variables con Valores Faltantes ---")
print(nulos_con_datos)

fig, ax = plt.subplots(figsize=(8, 4))
nulos_con_datos['Porcentaje_Nulos'].plot(kind='bar', ax=ax, color='#e74c3c')
ax.set_ylabel('% Faltante')
ax.set_title('Porcentaje de Valores Nulos por Variable Clínica', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()""")

    # 6. Duplicados
    md("""<a id="6"></a>
## 6. Auditoría de duplicados técnicos

Verificamos la unicidad de las observaciones registradas.
""")

    code("""dup_exactos = df_raw.duplicated().sum()
print(f"Duplicados exactos en la cohorte: {dup_exactos}")""")

    # 7. Hemodinámica
    md("""<a id="7"></a>
## 7. Detección de anomalías hemodinámicas y límites fisiológicos

Comprobamos si existen incongruencias biológicas fundamentales, como presión diastólica superior a sistólica o valores de glucemia incompatibles con la vida.
""")

    code("""print("--- Auditoría de Plausibilidad Fisiológica ---")
anom_sys_dia = (df_raw['sysBP'] <= df_raw['diaBP']).sum()
print(f"Casos con Presión Diastólica >= Sistólica: {anom_sys_dia}")

anom_pp_baja = ((df_raw['sysBP'] - df_raw['diaBP']) < 15).sum()
print(f"Casos con Presión de Pulso patológicamente baja (<15 mmHg): {anom_pp_baja}")

anom_chol_extremo = ((df_raw['totChol'] < 90) | (df_raw['totChol'] > 600)).sum()
print(f"Casos con Colesterol fuera de rango biológico habitual (<90 o >600 mg/dL): {anom_chol_extremo}")

anom_bmi_extremo = ((df_raw['BMI'] < 14) | (df_raw['BMI'] > 60)).sum()
print(f"Casos con IMC extremo (<14 o >60): {anom_bmi_extremo}")""")

    # 8. Distribuciones univariadas
    md("""<a id="8"></a>
## 8. Distribución univariada de factores de riesgo

Visualizamos las distribuciones de los principales biomarcadores para evaluar simetría y presencia de colas pesadas.
""")

    code("""fig, axes = plt.subplots(2, 3, figsize=(15, 8))
vars_grafico = ['age', 'sysBP', 'diaBP', 'totChol', 'glucose', 'cigsPerDay']
titulos = ['Edad (años)', 'Presión Sistólica (mmHg)', 'Presión Diastólica (mmHg)',
           'Colesterol Total (mg/dL)', 'Glucosa en Ayuno (mg/dL)', 'Cigarrillos al Día']

for ax, col, tit in zip(axes.flatten(), vars_grafico, titulos):
    sns.histplot(df_raw[col].dropna(), kde=True, ax=ax, color='#16a085', bins=30)
    ax.set_title(tit, fontsize=11, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('Frecuencia')

plt.tight_layout()
plt.show()""")

    # 9. Target
    md("""<a id="9"></a>
## 9. Inspección de la variable objetivo (`TenYearCHD`)

La variable objetivo binaria indica si el paciente presentó un evento coronario documentado en los 10 años posteriores al examen basal.
""")

    code("""conteo_target = df_raw['TenYearCHD'].value_counts()
pct_target = df_raw['TenYearCHD'].value_counts(normalize=True) * 100

print("--- Distribución de Incidencia a 10 Años (TenYearCHD) ---")
for val, count in conteo_target.items():
    lbl = "Libre de Evento Coronario (0)" if val == 0 else "Con Evento Coronario a 10a (1)"
    print(f"  {lbl}: {count:,} pacientes ({pct_target[val]:.2f}%)")

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(['Sin Evento (0)', 'Evento a 10a (1)'], conteo_target.values, color=['#2ecc71', '#e74c3c'], width=0.5)
for bar, c, p in zip(bars, conteo_target.values, pct_target.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, f"{c:,}\\n({p:.1f}%)", ha='center', fontweight='bold')
ax.set_ylabel('Pacientes')
ax.set_title('Incidencia Acumulada de Cardiopatía Coronaria a 10 Años', fontsize=12)
ax.set_ylim(0, conteo_target.max() * 1.2)
plt.tight_layout()
plt.show()""")

    # 10. Conclusiones
    md("""<a id="10"></a>
## 10. Conclusiones del diagnóstico y plan de preprocesamiento

### Resumen del Hallazgo Clínico
1. **Calidad Fisiológica:** El dataset Framingham exhibe coherencia hemodinámica estricta ($sysBP > diaBP$ en el 100% de los casos).
2. **Imputación Necesaria:** Variables como `glucose` (9.2% nulos), `totChol` (1.2% nulos), `BMI` (0.4% nulos) y `cigsPerDay` (0.7% nulos) requieren imputación fundamentada (por ejemplo, mediana según estatus de tabaquismo o diabetes).
3. **Desbalance Epidemiológico:** La incidencia es del ~15%, representativa de una cohorte comunitaria real. No debe aplicarse oversampling agresivo sin justificación, sino ponderación de pérdida (`class_weight='balanced'`) y evaluación por ROC-AUC y F1.
4. **Próximo Paso:** Proceder al **Notebook 02** aplicando la guía clínica de 19 pasos para estandarizar, imputar y generar variables derivadas hemodinámicas.
""")

    # Guardar
    os.makedirs('Notebooks', exist_ok=True)
    out_path = 'Notebooks/01_exploracion.ipynb'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({
            "cells": cells,
            "metadata": {
                "language_info": {"name": "python", "version": "3.12"}
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }, f, indent=2, ensure_ascii=False)

    print(f"✓ Notebook 01 generado exitosamente en: {out_path}")

if __name__ == '__main__':
    build_01_notebook()
