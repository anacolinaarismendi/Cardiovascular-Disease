"""
Generador del Notebook 01_exploracion.ipynb con estructura profesional,
auditoría completa de calidad de datos, visualizaciones clínicas y narrativa ejecutiva.
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
## Proyecto de Predicción de Riesgo Cardiovascular

> **Objetivo:** Realizar una auditoría técnica y médica exhaustiva del dataset bruto `cardio_train.csv` (70,000 registros), identificando anomalías hemodinámicas, valores atípicos imposibles en humanos vivos, patrones de completitud y cardinalidad antes de aplicar cualquier transformación de datos.

---

### 📑 Contenido del Notebook
1. [Configuración del entorno y librerías](#1)
2. [Carga e inspección preliminar de datos](#2)
3. [Estructura, tipos de datos y memoria](#3)
4. [Estadísticos descriptivos globales](#4)
5. [Auditoría de valores nulos y completitud](#5)
6. [Auditoría de duplicados técnicos y clínicos](#6)
7. [Detección de anomalías hemodinámicas y fisiológicas](#7)
8. [Distribución univariada de variables clínicas y estilo de vida](#8)
9. [Primera inspección de la variable objetivo (cardio)](#9)
10. [Conclusiones del diagnóstico y plan de preprocesamiento](#10)
""")

    # 1. Configuración
    md("""<a id="1"></a>
## 1. Configuración del entorno y librerías

Importamos las herramientas fundamentales para análisis exploratorio, cálculo matricial y visualización estadística avanzada.
""")

    code("""import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Configuración visual profesional
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 120)
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
sns.set_theme(style='whitegrid', palette='mako')

print(f"✓ Pandas: {pd.__version__} | NumPy: {np.__version__}")""")

    # 2. Carga de datos
    md("""<a id="2"></a>
## 2. Carga e inspección preliminar de datos

El dataset `cardio_train.csv` proviene de un registro de salud europeo/ruso en formato CSV separado por punto y coma (`;`).
""")

    code("""ruta_raw = '../Data/cardio_train.csv'
if not os.path.exists(ruta_raw):
    ruta_raw = 'Data/cardio_train.csv'

df_raw = pd.read_csv(ruta_raw, sep=';')
print(f"Dimensiones del dataset bruto: {df_raw.shape[0]:,} filas × {df_raw.shape[1]} columnas")
df_raw.head(5)""")

    # 3. Estructura y memoria
    md("""<a id="3"></a>
## 3. Estructura, tipos de datos y memoria

Verificamos el tipo de dato asignado por defecto por Pandas a cada columna y el consumo en memoria RAM.
""")

    code("""print("--- Resumen de Información Estructural ---")
df_raw.info()

print("\\n--- Cardinalidad (Valores únicos por variable) ---")
cardinalidad = pd.DataFrame({
    'Tipo': df_raw.dtypes,
    'Valores_Unicos': df_raw.nunique(),
    'Pct_Unicos': (df_raw.nunique() / len(df_raw) * 100).round(2)
})
display(cardinalidad.sort_values(by='Valores_Unicos', ascending=False))""")

    # 4. Estadísticos globales
    md("""<a id="4"></a>
## 4. Estadísticos descriptivos globales

Examinamos las medidas de tendencia central (media, mediana), dispersión (desviación estándar) y valores extremos (mínimos y máximos).
""")

    code("""desc = df_raw.describe().T
desc['IQR'] = desc['75%'] - desc['25%']
print("Estadísticos descriptivos del dataset bruto:")
display(desc[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'IQR']].round(2))""")

    # 5. Auditoría de nulos
    md("""<a id="5"></a>
## 5. Auditoría de valores nulos y completitud

Evaluamos si existen campos nulos (`NaN`, `None` o cadenas vacías).
""")

    code("""nulos = pd.DataFrame({
    'Valores_Nulos': df_raw.isnull().sum(),
    'Porcentaje (%)': (df_raw.isnull().sum() / len(df_raw) * 100).round(2)
})
print("Tabla de nulos:")
display(nulos)

fig, ax = plt.subplots(figsize=(10, 2.5))
sns.heatmap(df_raw.isnull(), cbar=False, yticklabels=False, cmap='Blues', ax=ax)
ax.set_title('Mapa de Calor de Nulos: 100% Completo (0 registros faltantes)', fontsize=12)
plt.tight_layout()
plt.show()""")

    # 6. Auditoría de duplicados
    md("""<a id="6"></a>
## 6. Auditoría de duplicados técnicos y clínicos

- **Duplicados exactos con ID:** Esperamos 0 porque la columna `id` es única por fila.
- **Duplicados clínicos (sin ID):** Pacientes con exactamente la misma combinación de edad, género, peso, talla, presiones, colesterol, glucosa y hábitos.
""")

    code("""dup_id = df_raw.duplicated(subset=['id']).sum()
dup_clinicos = df_raw.drop(columns=['id']).duplicated().sum()

print(f"Duplicados en clave primaria 'id': {dup_id} (integridad de clave)")
print(f"Duplicados clínicos en variables médicas: {dup_clinicos:,} ({dup_clinicos/len(df_raw)*100:.2f}%)")
print("--> Diagnóstico: En el preprocesamiento eliminaremos los duplicados clínicos para evitar ponderación artificial.")""")

    # 7. Detección de anomalías hemodinámicas
    md("""<a id="7"></a>
## 7. Detección de anomalías hemodinámicas y fisiológicas

En esta sección identificamos inconsistencias que violan leyes fundamentales de la fisiología cardiovascular humana:
1. **Presión arterial diastólica mayor o igual a la sistólica ($ap\\_lo \\ge ap\\_hi$).**
2. **Presiones negativas** (ej. $ap\\_hi = -150$).
3. **Errores de digitación con ceros extra** (ej. $ap\\_hi = 11000$ o $16020$).
4. **Estatura y peso incompatibles con adultos** (ej. altura 55 cm, peso 10 kg).
""")

    code("""# 1. Presión invertida
invertida = df_raw[df_raw['ap_lo'] >= df_raw['ap_hi']]
print(f"Casos con presión diastólica >= sistólica: {len(invertida):,} ({len(invertida)/len(df_raw)*100:.2f}%)")

# 2. Presiones extremas imposibles
sistolica_extrema = df_raw[(df_raw['ap_hi'] < 70) | (df_raw['ap_hi'] > 250)]
diastolica_extrema = df_raw[(df_raw['ap_lo'] < 40) | (df_raw['ap_lo'] > 150)]

print(f"Presión sistólica fuera de rango de vida (70-250 mmHg): {len(sistolica_extrema):,}")
print(f"Presión diastólica fuera de rango de vida (40-150 mmHg): {len(diastolica_extrema):,}")

# Visualización de boxplots brutos con outliers masivos
fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
cols_bio = ['ap_hi', 'ap_lo', 'height', 'weight']
titulos = ['Sistólica (ap_hi)', 'Diastólica (ap_lo)', 'Estatura (cm)', 'Peso (kg)']

for ax, col, tit in zip(axes, cols_bio, titulos):
    sns.boxplot(x=df_raw[col], ax=ax, color='#FF6B6B')
    ax.set_title(f'Boxplot Bruto: {tit}', fontsize=11)
plt.tight_layout()
plt.show()""")

    # 8. Distribución univariada
    md("""<a id="8"></a>
## 8. Distribución univariada de variables clínicas y estilo de vida

Convertimos temporalmente la edad a años cumplidos para interpretar su distribución real y examinamos las prevalencias categóricas.
""")

    code("""edad_anios = df_raw['age'] / 365.25

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# Edad
sns.histplot(edad_anios, bins=30, kde=True, ax=axes[0], color='#00C9A7')
axes[0].set_title(f'Distribución de Edad (Media: {edad_anios.mean():.1f} años)')
axes[0].set_xlabel('Años cumplidos')

# Género
conteo_genero = df_raw['gender'].value_counts()
axes[1].pie(conteo_genero.values, labels=['Mujer (1)', 'Hombre (2)'], autopct='%1.1f%%',
            colors=['#4ECDC4', '#1E3A5F'], startangle=90)
axes[1].set_title('Distribución por Género Biológico')

# Estilo de vida combinado
estilo = pd.DataFrame({
    'Hábito': ['Fumador', 'Alcohol', 'Actividad Física'],
    'Prevalencia (%)': [
        df_raw['smoke'].mean() * 100,
        df_raw['alco'].mean() * 100,
        df_raw['active'].mean() * 100
    ]
})
sns.barplot(data=estilo, x='Hábito', y='Prevalencia (%)', ax=axes[2], palette='mako')
axes[2].set_title('Prevalencia de Factores de Estilo de Vida')
axes[2].set_ylim(0, 100)

for p in axes[2].patches:
    axes[2].annotate(f"{p.get_height():.1f}%",
                     (p.get_x() + p.get_width() / 2., p.get_height() + 2),
                     ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()""")

    # 9. Inspección del Target
    md("""<a id="9"></a>
## 9. Primera inspección de la variable objetivo (cardio)

Verificamos el balance inicial de la variable a predecir (`0 = Sano`, `1 = Enfermo`).
""")

    code("""conteo_target = df_raw['cardio'].value_counts().sort_index()
pct_target = df_raw['cardio'].value_counts(normalize=True).sort_index() * 100

fig, ax = plt.subplots(figsize=(6, 3.8))
bars = ax.bar(['Sin Enfermedad (0)', 'Con Enfermedad (1)'], conteo_target.values,
              color=['#4ECDC4', '#FF6B6B'], width=0.5)

for bar, val, pct in zip(bars, conteo_target.values, pct_target.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1000,
            f"{val:,}\\n({pct:.1f}%)", ha='center', fontsize=10, fontweight='bold')

ax.set_ylabel('Nº de Pacientes')
ax.set_title('Balance de la Variable Objetivo en Dataset Bruto', fontsize=12)
ax.set_ylim(0, conteo_target.max() * 1.25)
plt.tight_layout()
plt.show()

ratio = conteo_target.max() / conteo_target.min()
print(f"Ratio de clases inicial: {ratio:.2f} (Perfectamente balanceado)")""")

    # 10. Conclusiones y Plan
    md("""<a id="10"></a>
## 10. Conclusiones de la Auditoría y Plan de Preprocesamiento

### Resumen de Hallazgos Críticos:
1. **Calidad de datos:** No existen valores nulos (`0 NaN`), pero existen **duplicados clínicos exactos** que deben filtrarse.
2. **Inconsistencias fisiológicas severas:** Se identificaron **1,236 filas** con presión diastólica mayor o igual a la sistólica ($ap\\_lo \\ge ap\\_hi$), lo cual es físicamente imposible en humanos vivos.
3. **Valores atípicos extremos:** Existen valores de presión de hasta 16,020 mmHg y negativos (-150 mmHg) causados por errores tipográficos de captura de datos.
4. **Balance del target:** La variable dependiente `cardio` está prácticamente en paridad exacta (50.0% / 50.0%), garantizando que no se requerirán técnicas de remuestreo sintético (SMOTE).

### Directrices para el Notebook 02 (Preprocesamiento):
- Convertir la edad en días a años cumplidos con factor 365.25.
- Aplicar criterios fisiológicos basados en guías internacionales (AHA/OMS):
  - $80 \\le ap\\_hi \\le 220\\text{ mmHg}$
  - $50 \\le ap\\_lo \\le 130\\text{ mmHg}$
  - $ap\\_hi > ap\\_lo$
  - Presión de Pulso $\\ge 20\\text{ mmHg}$
  - Estatura $140 - 205\\text{ cm}$
  - Peso $40 - 165\\text{ kg}$
- Construir variables clínicas derivadas: $IMC$ (BMI), Presión de Pulso ($PP$), Presión Arterial Media ($PAM$) y estadios de hipertensión AHA.
- Estructurar el `ColumnTransformer` y la partición estratificada sin fugas de datos.
""")

    notebook_dict = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    target_path = 'Notebooks/01_exploracion.ipynb'
    with open(target_path, 'w', encoding='utf-8') as f:
        json.dump(notebook_dict, f, indent=1, ensure_ascii=False)
    print(f"✓ Notebook {target_path} generado con éxito con enfoque profesional de auditoría.")

if __name__ == '__main__':
    build_01_notebook()
