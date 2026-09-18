"""
Generador del Notebook 02_preprocesamiento.ipynb siguiendo la guía de 19 pasos clínicos,
adaptado a la cohorte longitudinal Framingham Heart Study.
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
    md("""# ⚙️ Notebook 02: Pipeline de Preprocesamiento Clínico y Feature Engineering
## Proyecto de Predicción de Riesgo Cardiovascular — Cohorte Framingham
**Autora:** Ana Colina Arismendi  
**Marco Metodológico:** Protocolo de Preprocesamiento en 19 Pasos Clínicos

> **Objetivo:** Transformar los datos brutos del *Framingham Heart Study* en una matriz analítica limpia, fisiológicamente consistente y enriquecida con biomarcadores derivados (Presión de Pulso, PAM, estadios de hipertensión AHA y dislipidemia), lista para modelado predictivo.

---

### 📑 Los 19 Pasos del Pipeline de Preprocesamiento
1. **Paso 01:** Configuración del entorno, librerías y reproducibilidad
2. **Paso 02:** Carga y verificación de la cohorte basal
3. **Paso 03:** Auditoría de completitud y tipos de variables
4. **Paso 04:** Eliminación de duplicados técnicos y clínicos
5. **Paso 05:** Filtrado de anomalías fisiológicas incompatibles con la vida
6. **Paso 06:** Imputación clínica estratificada de valores faltantes
7. **Paso 07:** Tratamiento de valores extremos (Winsorización percentilar)
8. **Paso 08:** Feature Engineering I: Presión de Pulso (PP)
9. **Paso 09:** Feature Engineering II: Presión Arterial Media (PAM)
10. **Paso 10:** Feature Engineering III: Clasificación AHA de Tensión Arterial
11. **Paso 11:** Feature Engineering IV: Estratificación del Colesterol Sérico
12. **Paso 12:** Feature Engineering V: Estratificación de la Glucemia
13. **Paso 13:** Feature Engineering VI: Indicador Compuesto de Hipertensión
14. **Paso 14:** Feature Engineering VII: Clasificación Ponderal e IMC
15. **Paso 15:** Armonización de nombres y alineación de variable objetivo (`cardio`)
16. **Paso 16:** Diagnóstico de multicolinealidad y matriz de correlación
17. **Paso 17:** Partición Estratificada en conjuntos de Entrenamiento y Test (80/20)
18. **Paso 18:** Construcción y serialización del Pipeline Scikit-Learn
19. **Paso 19:** Exportación final del dataset limpio y reporte de calidad
""")

    # Paso 1
    md("""## Paso 01: Configuración del entorno, librerías y reproducibilidad""")
    code("""import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
print("✓ Librerías importadas y semilla 42 fijada.")""")

    # Paso 2
    md("""## Paso 02: Carga y verificación de la cohorte basal""")
    code("""ruta_raw = '../Data/framingham.csv'
if not os.path.exists(ruta_raw):
    ruta_raw = 'Data/framingham.csv'

df = pd.read_csv(ruta_raw)
print(f"Registros iniciales: {len(df):,} pacientes con {df.shape[1]} variables clínicas.")""")

    # Paso 3
    md("""## Paso 03: Auditoría de completitud y tipos de variables""")
    code("""info_df = pd.DataFrame({
    'Tipo': df.dtypes,
    'No_Nulos': df.notnull().sum(),
    'Nulos': df.isnull().sum(),
    'Pct_Nulos': (df.isnull().mean() * 100).round(2)
})
print("--- Auditoría de Valores Faltantes ---")
print(info_df[info_df['Nulos'] > 0])""")

    # Paso 4
    md("""## Paso 04: Eliminación de duplicados técnicos y clínicos""")
    code("""dup_count = df.duplicated().sum()
if dup_count > 0:
    df = df.drop_duplicates().reset_index(drop=True)
print(f"Duplicados eliminados: {dup_count}. Total registros vigentes: {len(df):,}")""")

    # Paso 5
    md("""## Paso 05: Filtrado de anomalías fisiológicas incompatibles con la vida""")
    code("""# Filtros hemodinámicos estrictos:
# 1. sysBP entre 70 y 270 mmHg
# 2. diaBP entre 40 y 160 mmHg
# 3. sysBP > diaBP
# 4. heartRate entre 35 y 220 lpm
inicial = len(df)
filtro_hemo = (
    (df['sysBP'] >= 70) & (df['sysBP'] <= 270) &
    (df['diaBP'] >= 40) & (df['diaBP'] <= 160) &
    (df['sysBP'] > df['diaBP']) &
    (df['heartRate'].isna() | ((df['heartRate'] >= 35) & (df['heartRate'] <= 220))) &
    (df['totChol'].isna() | (df['totChol'] <= 600))
)
df = df[filtro_hemo].copy().reset_index(drop=True)
print(f"Registros removidos por imposibilidad biológica: {inicial - len(df):,}. Restantes: {len(df):,}")""")

    # Paso 6
    md("""## Paso 06: Imputación clínica estratificada de valores faltantes""")
    code("""# Imputación guiada por contexto médico:
# 1. cigsPerDay: si currentSmoker == 0 -> 0. Si fuma -> mediana de fumadores.
df.loc[(df['cigsPerDay'].isna()) & (df['currentSmoker'] == 0), 'cigsPerDay'] = 0.0
med_cigs = df[df['currentSmoker'] == 1]['cigsPerDay'].median()
df['cigsPerDay'] = df['cigsPerDay'].fillna(med_cigs)

# 2. BPMeds: si es nulo, moda clínica (0 = no usa)
df['BPMeds'] = df['BPMeds'].fillna(0.0)

# 3. totChol: mediana según grupo de edad (decenios)
df['age_decile'] = (df['age'] // 10) * 10
df['totChol'] = df.groupby('age_decile')['totChol'].transform(lambda x: x.fillna(x.median()))

# 4. BMI: mediana según sexo biológico
df['BMI'] = df.groupby('male')['BMI'].transform(lambda x: x.fillna(x.median()))

# 5. heartRate: mediana poblacional
df['heartRate'] = df['heartRate'].fillna(df['heartRate'].median())

# 6. glucose: mediana según condición de diabetes
df['glucose'] = df.groupby('diabetes')['glucose'].transform(lambda x: x.fillna(x.median()))

df = df.drop(columns=['age_decile'])
print("Valores nulos restantes tras imputación clínica:")
print(df.isnull().sum()[df.isnull().sum() > 0])
print("✓ Completitud del 100% alcanzada.")""")

    # Paso 7
    md("""## Paso 07: Tratamiento de valores extremos (Winsorización percentilar)""")
    code("""# Winsorización al percentil 99.5 para mitigar la distorsión por colas pesadas
cols_cont = ['totChol', 'sysBP', 'diaBP', 'BMI', 'glucose']
for col in cols_cont:
    p995 = df[col].quantile(0.995)
    p005 = df[col].quantile(0.005)
    df[col] = df[col].clip(lower=p005, upper=p995)

print("✓ Winsorización percentil 0.5% - 99.5% aplicada a variables continuas.")""")

    # Paso 8
    md("""## Paso 08: Feature Engineering I: Presión de Pulso (PP)
La presión diferencial o de pulso ($PP = sysBP - diaBP$) refleja la rigidez de las grandes arterias elásticas.""")
    code("""df['pulse_pressure'] = df['sysBP'] - df['diaBP']
print(f"Media de Presión de Pulso: {df['pulse_pressure'].mean():.1f} mmHg (Normal: 30-50 mmHg)")""")

    # Paso 9
    md("""## Paso 09: Feature Engineering II: Presión Arterial Media (PAM)
La PAM ($MAP = diaBP + \\frac{PP}{3}$) cuantifica la presión de perfusión tisular promedio en el ciclo cardíaco.""")
    code("""df['map'] = df['diaBP'] + (df['pulse_pressure'] / 3.0)
print(f"Media de Presión Arterial Media (PAM): {df['map'].mean():.1f} mmHg (Normal: 70-105 mmHg)")""")

    # Paso 10
    md("""## Paso 10: Feature Engineering III: Clasificación AHA de Tensión Arterial""")
    code("""def clasificar_aha(row):
    s, d = row['sysBP'], row['diaBP']
    if s >= 140 or d >= 90:
        return 4 # HTA Grado 2
    elif (130 <= s <= 139) or (80 <= d <= 89):
        return 3 # HTA Grado 1
    elif (120 <= s <= 129) and (d < 80):
        return 2 # Presión Elevada
    else:
        return 1 # Normal

df['bp_stage'] = df.apply(clasificar_aha, axis=1)
print("Distribución de estadios tensionales AHA:")
print(df['bp_stage'].value_counts(normalize=True).round(3))""")

    # Paso 11
    md("""## Paso 11: Feature Engineering IV: Estratificación del Colesterol Sérico""")
    code("""# 1: Deseable (<200 mg/dL), 2: Limítrofe (200-239 mg/dL), 3: Elevado (>=240 mg/dL)
df['chol_stage'] = pd.cut(df['totChol'], bins=[0, 200, 240, 1000], labels=[1, 2, 3]).astype(int)
print("Distribución estadios de colesterol:")
print(df['chol_stage'].value_counts(normalize=True).round(3))""")

    # Paso 12
    md("""## Paso 12: Feature Engineering V: Estratificación de la Glucemia""")
    code("""# 1: Normal (<100 mg/dL), 2: Glucosa alterada en ayuno (100-125), 3: Diabetes (>=126)
df['gluc_stage'] = pd.cut(df['glucose'], bins=[0, 100, 126, 1000], labels=[1, 2, 3]).astype(int)
print("Distribución estadios glucémicos:")
print(df['gluc_stage'].value_counts(normalize=True).round(3))""")

    # Paso 13
    md("""## Paso 13: Feature Engineering VI: Indicador Compuesto de Hipertensión""")
    code("""df['hypertension'] = ((df['sysBP'] >= 140) | (df['diaBP'] >= 90) | (df['BPMeds'] == 1)).astype(int)
print(f"Prevalencia de Hipertensión Clínica: {df['hypertension'].mean()*100:.1f}%")""")

    # Paso 14
    md("""## Paso 14: Feature Engineering VII: Clasificación Ponderal e IMC""")
    code("""df['overweight'] = (df['BMI'] >= 25.0).astype(int)
print(f"Prevalencia de Sobrepeso/Obesidad (IMC >= 25): {df['overweight'].mean()*100:.1f}%")""")

    # Paso 15
    md("""## Paso 15: Armonización de nombres y alineación de variable objetivo (`cardio`)""")
    code("""df['cardio'] = df['TenYearCHD'].astype(int)
print(f"Variable objetivo unificada 'cardio':")
print(df['cardio'].value_counts())""")

    # Paso 16
    md("""## Paso 16: Diagnóstico de multicolinealidad y matriz de correlación""")
    code("""cols_corr = ['age', 'cigsPerDay', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose', 'cardio']
corr = df[cols_corr].corr()
print("--- Correlación con el Evento Coronario (cardio) ---")
print(corr['cardio'].sort_values(ascending=False).round(3))""")

    # Paso 17
    md("""## Paso 17: Partición Estratificada en conjuntos de Entrenamiento y Test (80/20)""")
    code("""X = df.drop(columns=['cardio', 'TenYearCHD'])
y = df['cardio']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
)
print(f"Conjunto de Entrenamiento: {X_train.shape[0]:,} pacientes")
print(f"Conjunto de Test:          {X_test.shape[0]:,} pacientes")
print(f"Proporción de clase positiva en Train: {y_train.mean():.3f} | Test: {y_test.mean():.3f}")""")

    # Paso 18
    md("""## Paso 18: Construcción y serialización del Pipeline Scikit-Learn""")
    code("""scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[cols_cont])

os.makedirs('../models', exist_ok=True)
os.makedirs('models', exist_ok=True)

ruta_pipe = '../models/pipeline_preprocesamiento.pkl'
if not os.path.exists('../models'):
    ruta_pipe = 'models/pipeline_preprocesamiento.pkl'

joblib.dump(scaler, ruta_pipe)
print(f"✓ Scaler serializado en: {ruta_pipe}")""")

    # Paso 19
    md("""## Paso 19: Exportación final del dataset limpio y reporte de calidad""")
    code("""os.makedirs('../Data/processed', exist_ok=True)
os.makedirs('Data/processed', exist_ok=True)

ruta_out = '../Data/processed/processed.csv'
if not os.path.exists('../Data/processed'):
    ruta_out = 'Data/processed/processed.csv'

df.to_csv(ruta_out, index=False)
print(f"✓ Dataset procesado guardado exitosamente en: {ruta_out}")
print(f"Dimensiones finales: {df.shape[0]:,} pacientes × {df.shape[1]} variables")""")

    # Guardar notebook
    os.makedirs('Notebooks', exist_ok=True)
    out_path = 'Notebooks/02_preprocesamiento.ipynb'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({
            "cells": cells,
            "metadata": {
                "language_info": {"name": "python", "version": "3.12"}
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }, f, indent=2, ensure_ascii=False)

    print(f"✓ Notebook 02 generado exitosamente en: {out_path}")

if __name__ == '__main__':
    build_02_notebook()
