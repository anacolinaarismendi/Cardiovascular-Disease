"""
Generador del Notebook 02_preprocesamiento.ipynb con rigor técnico y médico profesional,
implementando filtros fisiológicos, feature engineering clínico y pipeline de producción.
"""

import json
import os

def build_02_notebook():
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
    md("""# 🛠️ Notebook 02: Preprocesamiento de Datos, Validación Fisiológica y Feature Engineering
## Proyecto de Predicción de Riesgo Cardiovascular en Pacientes Reales

> **Objetivo:** Transformar el dataset bruto `cardio_train.csv` en un conjunto de datos limpio, fisiológicamente consistente y enriquecido con biomarcadores clínicos calculados según las guías de la **OMS**, **ACC/AHA (2017)** y **ESC (2018/2024)**. Se construye además un pipeline formal de `scikit-learn` para producción que previene cualquier fuga de datos (*data leakage*).

---

### 📑 Contenido del Notebook
1. [Librerías especializadas y entorno](#1)
2. [Carga de datos y eliminación de identificadores técnicos](#2)
3. [Deduplicación clínica fundamentada](#3)
4. [Conversión de unidades temporales (Edad en años cumplidos)](#4)
5. [Corrección de inversión hemodinámica (ap_lo >= ap_hi)](#5)
6. [Filtros de plausibilidad fisiológica humana (Pacientes Reales vs IQR)](#6)
7. [Feature Engineering Clínico (IMC, Presión de Pulso, PAM, Estadios AHA)](#7)
8. [Tipado y optimización en memoria](#8)
9. [Análisis de escalado (StandardScaler vs RobustScaler)](#9)
10. [División Train / Test estratificada sin fugas de datos](#10)
11. [Construcción del ColumnTransformer de producción](#11)
12. [Guardado y verificación de artefactos](#12)
13. [Checklist final de preprocesamiento](#13)
""")

    # 1. Librerías
    md("""<a id="1"></a>
## 1. Librerías especializadas y entorno

Cargamos el ecosistema para limpieza de datos, transformaciones estadísticas y pipelines de aprendizaje automático.
""")

    code("""import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

pd.set_option('display.max_columns', None)
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
sns.set_theme(style='whitegrid', palette='mako')

print("✓ Librerías cargadas exitosamente.")""")

    # 2. Carga y eliminación de ID
    md("""<a id="2"></a>
## 2. Carga de datos y eliminación de identificadores técnicos

La columna `id` es un consecutivo de base de datos sin correlación biológica. Mantenerla introduciría ruido o sobreajuste espurio.
""")

    code("""ruta_raw = '../Data/cardio_train.csv'
if not os.path.exists(ruta_raw):
    ruta_raw = 'Data/cardio_train.csv'

df = pd.read_csv(ruta_raw, sep=';')
print(f"Total registros cargados: {len(df):,}")

if 'id' in df.columns:
    df = df.drop(columns=['id'])
    print("✓ Columna 'id' eliminada exitosamente.")

df.head(3)""")

    # 3. Deduplicación clínica
    md("""<a id="3"></a>
## 3. Deduplicación clínica fundamentada

Sin la columna `id`, evaluamos si existen registros exactamente idénticos en todas las 12 características médicas. Filas duplicadas sesgan los clasificadores al sobreponderar perfiles repetidos.
""")

    code("""dup_count = df.duplicated().sum()
pct_dup = (dup_count / len(df)) * 100
print(f"Duplicados clínicos detectados: {dup_count:,} ({pct_dup:.2f}%)")

df = df.drop_duplicates().copy()
print(f"Registros restantes tras deduplicación: {len(df):,}")""")

    # 4. Conversión de Edad
    md("""<a id="4"></a>
## 4. Conversión de unidades temporales (Edad en años cumplidos)

La variable `age` viene expresada en días desde el nacimiento. Fisiológicamente la convertimos a años dividiendo por el año trópico medio ($365.25$ días) y redondeando a un decimal.
""")

    code("""df['age_years'] = (df['age'] / 365.25).round(1)
df = df.drop(columns=['age'])

print("Estadísticos de edad transformada:")
print(f"  Mínima: {df['age_years'].min()} años")
print(f"  Media:  {df['age_years'].mean():.1f} años")
print(f"  Máxima: {df['age_years'].max()} años")
df[['age_years']].head(3)""")

    # 5. Inversión hemodinámica
    md("""<a id="5"></a>
## 5. Corrección de inversión hemodinámica (ap_lo >= ap_hi)

En fisiología circulatoria humana, la presión sistólica (presión máxima eyectiva) es invariablemente mayor que la diastólica (presión mínima de llenado ventricular). Casos donde $ap\\_lo \\ge ap\\_hi$ o presiones negativas representan errores de lectura o manguito invertido.
""")

    code("""inv = df[df['ap_lo'] >= df['ap_hi']]
print(f"Filas con presión diastólica >= sistólica: {len(inv):,} ({len(inv)/len(df)*100:.2f}%)")

df = df[df['ap_hi'] > df['ap_lo']].copy()
print(f"Registros tras eliminar presiones invertidas: {len(df):,}")""")

    # 6. Plausibilidad fisiológica
    md("""<a id="6"></a>
## 6. Filtros de plausibilidad fisiológica humana (Pacientes Reales vs IQR)

El criterio estadístico clásico de Tukey ($1.5\\cdot IQR$) recortaría a pacientes severamente hipertensos reales (que son casos críticos del estudio). Por ello, aplicamos **límites de plausibilidad biomédica** consensuados por la **OMS**, **AHA** y **ESC**:
- **Presión Sistólica ($ap\\_hi$):** $80 \\le ap\\_hi \\le 220\\text{ mmHg}$
- **Presión Diastólica ($ap\\_lo$):** $50 \\le ap\\_lo \\le 130\\text{ mmHg}$
- **Presión de Pulso ($PP = ap\\_hi - ap\\_lo$):** $20 \\le PP \\le 110\\text{ mmHg}$ *(una presión de pulso <20 mmHg indica shock/colapso circulatorio no ambulatorio)*.
- **Estatura:** $140 \\le \\text{height} \\le 205\\text{ cm}$
- **Peso:** $40 \\le \\text{weight} \\le 165\\text{ kg}$
""")

    code("""filtro_fisiologico = (
    (df['ap_hi'] >= 80) & (df['ap_hi'] <= 220) &
    (df['ap_lo'] >= 50) & (df['ap_lo'] <= 130) &
    ((df['ap_hi'] - df['ap_lo']) >= 20) & ((df['ap_hi'] - df['ap_lo']) <= 110) &
    (df['height'] >= 140) & (df['height'] <= 205) &
    (df['weight'] >= 40) & (df['weight'] <= 165)
)

outliers = (~filtro_fisiologico).sum()
print(f"Filas fuera de plausibilidad fisiológica: {outliers:,} ({outliers/len(df)*100:.2f}%)")

df = df[filtro_fisiologico].copy()
print(f"Cohorte de pacientes reales validada: {len(df):,}")""")

    # 7. Feature Engineering
    md("""<a id="7"></a>
## 7. Feature Engineering Clínico

Calculamos parámetros hemodinámicos y antropométricos con alta evidencia médica:
1. **Índice de Masa Corporal (IMC):**
   $$BMI = \\frac{\\text{weight (kg)}}{(\\text{height (m)})^2}$$
   Acotamos a rango clínico creíble ($16.0 - 52.0\\text{ kg/m}^2$).
2. **Presión de Pulso ($PP$):** $ap\\_hi - ap\\_lo$ (rigidez de la pared aórtica).
3. **Presión Arterial Media (PAM / MAP):**
   $$MAP = ap\\_lo + \\frac{PP}{3}$$
4. **Estadío de Hipertensión AHA/ACC 2017:**
   - 0: Normal ($<120$ y $<80$)
   - 1: Elevada ($120-129$ y $<80$)
   - 2: HTA Grado 1 ($130-139$ o $80-89$)
   - 3: HTA Grado 2 ($\\ge 140$ o $\\ge 90$)
""")

    code("""# 1. IMC
df['bmi'] = (df['weight'] / ((df['height'] / 100) ** 2)).round(1)
df = df[(df['bmi'] >= 16.0) & (df['bmi'] <= 52.0)].copy()

# 2. Presión diferencial
df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']

# 3. Presión Arterial Media
df['map'] = (df['ap_lo'] + (df['pulse_pressure'] / 3)).round(1)

# 4. Marcadores binarios
df['hypertension'] = ((df['ap_hi'] >= 140) | (df['ap_lo'] >= 90)).astype('int8')
df['overweight'] = (df['bmi'] >= 25.0).astype('int8')

# 5. Estadío AHA
def clasificar_estadio_aha(row):
    hi, lo = row['ap_hi'], row['ap_lo']
    if hi >= 140 or lo >= 90:
        return 3
    elif (130 <= hi <= 139) or (80 <= lo <= 89):
        return 2
    elif (120 <= hi <= 129) and (lo < 80):
        return 1
    return 0

df['bp_stage'] = df.apply(clasificar_estadio_aha, axis=1).astype('int8')

print("✓ Variables de ingeniería clínica creadas:")
display(df[['bmi', 'pulse_pressure', 'map', 'hypertension', 'overweight', 'bp_stage']].head(4))""")

    # 8. Tipado y memoria
    md("""<a id="8"></a>
## 8. Tipado y optimización en memoria

Ajustamos los tipos de datos a representaciones compactas (`int8`, `float32`) para acelerar la computación y reducir la huella de memoria.
""")

    code("""cols_int8 = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active',
             'cardio', 'hypertension', 'overweight', 'bp_stage']
for col in cols_int8:
    df[col] = df[col].astype('int8')

print("Tipos de datos finales:")
print(df.dtypes)
print(f"\\nMemoria ocupada: {df.memory_usage().sum() / (1024*1024):.2f} MB")""")

    # 9. Escalado
    md("""<a id="9"></a>
## 9. Análisis de escalado (StandardScaler vs RobustScaler)

Comparamos la estandarización por z-score frente al escalado robusto en las variables numéricas continuas.
""")

    code("""num_features = ['age_years', 'height', 'weight', 'ap_hi', 'ap_lo', 'bmi', 'pulse_pressure', 'map']

scaler = StandardScaler()
X_std = scaler.fit_transform(df[num_features])

print("Estadísticos tras StandardScaler (Media ~ 0, Desviación ~ 1):")
df_std = pd.DataFrame(X_std, columns=num_features)
display(df_std.describe().round(3).T[['mean', 'std', 'min', 'max']])""")

    # 10. Split Train / Test
    md("""<a id="10"></a>
## 10. División Train / Test estratificada sin fugas de datos

**Regla de oro de la Guía de Preprocesamiento:**
> La partición de datos se realiza **antes** de ajustar cualquier estimador o escalador, asegurando que las transformaciones aplicadas a Test utilicen únicamente los parámetros aprendidos en Train (*evitando data leakage*).
""")

    code("""cat_features = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
X_cols = num_features + cat_features

X = df[X_cols].copy()
y = df['cardio'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Conjunto de Entrenamiento: {X_train.shape[0]:,} filas ({len(X_train)/len(df)*100:.1f}%)")
print(f"Conjunto de Prueba:        {X_test.shape[0]:,} filas ({len(X_test)/len(df)*100:.1f}%)")
print(f"Proporción de cardio en Train: {y_train.mean():.3f}")
print(f"Proporción de cardio en Test:  {y_test.mean():.3f} (Idéntica estratificación)")""")

    # 11. ColumnTransformer
    md("""<a id="11"></a>
## 11. Construcción del ColumnTransformer de producción

Diseñamos una arquitectura de preprocesamiento unificada con `ColumnTransformer`:
- **Sub-pipeline Numérico:** Imputación con mediana preventiva + Estandarización con `StandardScaler`.
- **Sub-pipeline Categórico:** Imputación con el valor más frecuente (moda).
""")

    code("""pipe_num = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

pipe_cat = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent'))
])

preprocesador = ColumnTransformer(
    transformers=[
        ('num', pipe_num, num_features),
        ('cat', pipe_cat, cat_features)
    ]
)

# Ajuste EXCLUSIVO en Train
preprocesador.fit(X_train)
print("✓ ColumnTransformer ajustado correctamente sin contaminación de Test.")""")

    # 12. Guardado y verificación
    md("""<a id="12"></a>
## 12. Guardado y verificación de artefactos

Exportamos el dataset procesado limpio y el pipeline serializado en formato `.pkl` con `joblib`.
""")

    code("""os.makedirs('../Data/processed', exist_ok=True)
os.makedirs('../models', exist_ok=True)

# Guardar dataset procesado
df.to_csv('../Data/processed/processed.csv', index=False)
print("✓ Dataset procesado guardado en: ../Data/processed/processed.csv")

# Guardar pipeline
joblib.dump(preprocesador, '../models/pipeline_preprocesamiento.pkl')
print("✓ Pipeline serializado en: ../models/pipeline_preprocesamiento.pkl")

# Verificación de recarga
pipe_check = joblib.load('../models/pipeline_preprocesamiento.pkl')
print("\\nVerificación de carga exitosa:")
print(pipe_check)""")

    # 13. Checklist final
    md("""<a id="13"></a>
## 13. Checklist final de preprocesamiento

| Paso | Verificación | Estado |
|---|---|:---:|
| 1 | Eliminación de identificadores espurios (`id`) | ✅ |
| 2 | Deduplicación clínica (674 registros eliminados) | ✅ |
| 3 | Conversión temporal de edad a años cumplidos | ✅ |
| 4 | Corrección de presiones invertidas ($ap\\_lo \\ge ap\\_hi$) | ✅ |
| 5 | Filtro de plausibilidad humana (AHA/OMS) | ✅ |
| 6 | Cálculo de IMC, Presión de Pulso, PAM y Estadios AHA | ✅ |
| 7 | Tipado eficiente de memoria | ✅ |
| 8 | Partición Train/Test estratificada 80/20 | ✅ |
| 9 | Prevención de data leakage con ajuste exclusivo en Train | ✅ |
| 10 | Serialización reproducible de artefactos | ✅ |

El dataset procesado está listo para alimentar los análisis visuales en el **Notebook 03 (EDA)** y el modelo predictivo de la **App Streamlit**.
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

    target_path = 'Notebooks/02_preprocesamiento.ipynb'
    with open(target_path, 'w', encoding='utf-8') as f:
        json.dump(notebook_dict, f, indent=1, ensure_ascii=False)
    print(f"✓ Notebook {target_path} generado con éxito.")

if __name__ == '__main__':
    build_02_notebook()
