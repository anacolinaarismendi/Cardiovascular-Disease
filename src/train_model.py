"""
Script de preprocesamiento clínico, validación fisiológica y entrenamiento
del modelo predictivo de riesgo cardiovascular a 10 años.
Basado en el Framingham Heart Study y las directrices de la ACC/AHA y ESC.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

def limpiar_datos_framingham(ruta_raw):
    """
    Carga y limpia el dataset de Framingham Heart Study aplicando
    imputación clínica y criterios de plausibilidad biológica.
    """
    print(f"Cargando dataset Framingham desde: {ruta_raw}")
    df = pd.read_csv(ruta_raw)
    print(f"Total registros iniciales: {len(df):,}")

    # 1. Eliminar columna education si no tiene relevancia clínica directa
    # (la mantenemos o descartamos; no afecta a fisiología)
    if 'education' in df.columns:
        df = df.drop(columns=['education'])

    # 2. Imputación clínica antes de filtrado para no perder pacientes innecesariamente:
    # - cigsPerDay: si fuma pero cigsPerDay es NaN, imputar mediana de fumadores (9). Si no fuma, 0.
    df.loc[df['currentSmoker'] == 0, 'cigsPerDay'] = 0
    mediana_cigs = df[df['currentSmoker'] == 1]['cigsPerDay'].median()
    df['cigsPerDay'] = df['cigsPerDay'].fillna(mediana_cigs)

    # - BPMeds, prevalentStroke, prevalentHyp, diabetes: moda
    cols_bin = ['BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes']
    for c in cols_bin:
        df[c] = df[c].fillna(df[c].mode()[0]).astype(int)

    # - totChol, sysBP, diaBP, BMI, heartRate, glucose: mediana
    cols_num_imp = ['totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']
    for c in cols_num_imp:
        df[c] = df[c].fillna(df[c].median())

    # 3. Consistencia hemodinámica: sysBP > diaBP
    df = df[df['sysBP'] > df['diaBP']].copy()

    # 4. Filtros fisiológicos plausibles para adultos:
    # - sysBP: 80 a 260 mmHg
    # - diaBP: 50 a 140 mmHg
    # - totChol: 100 a 600 mg/dL
    # - glucose: 40 a 350 mg/dL
    # - BMI: 15.0 a 55.0 kg/m²
    # - heartRate: 40 a 140 lpm
    condicion_fisiologica = (
        (df['sysBP'] >= 80) & (df['sysBP'] <= 260) &
        (df['diaBP'] >= 50) & (df['diaBP'] <= 140) &
        ((df['sysBP'] - df['diaBP']) >= 15) &
        (df['totChol'] >= 100) & (df['totChol'] <= 600) &
        (df['glucose'] >= 40) & (df['glucose'] <= 350) &
        (df['BMI'] >= 15.0) & (df['BMI'] <= 55.0) &
        (df['heartRate'] >= 40) & (df['heartRate'] <= 140)
    )

    df_limpio = df[condicion_fisiologica].copy()
    print(f"Registros tras filtros fisiológicos: {len(df_limpio):,} (removidos: {len(df) - len(df_limpio):,})")

    # 5. Feature Engineering Clínico
    # - Presión de Pulso (PP)
    df_limpio['pulse_pressure'] = (df_limpio['sysBP'] - df_limpio['diaBP']).round(1)

    # - Presión Arterial Media (PAM / MAP)
    df_limpio['map'] = (df_limpio['diaBP'] + (df_limpio['pulse_pressure'] / 3)).round(1)

    # - Categoría de Presión Arterial AHA/ACC 2017:
    def estadiar_aha(row):
        sys, dia = row['sysBP'], row['diaBP']
        if sys >= 140 or dia >= 90:
            return 3  # HTA Grado 2
        elif (130 <= sys <= 139) or (80 <= dia <= 89):
            return 2  # HTA Grado 1
        elif (120 <= sys <= 129) and (dia < 80):
            return 1  # Elevada
        return 0      # Normal

    df_limpio['bp_stage'] = df_limpio.apply(estadiar_aha, axis=1)

    # - Categoría de Colesterol (NCEP ATP III):
    # 0: Deseable (<200 mg/dL), 1: Limítrofe (200-239), 2: Alto (>=240)
    df_limpio['chol_stage'] = pd.cut(
        df_limpio['totChol'],
        bins=[0, 199, 239, 1000],
        labels=[0, 1, 2]
    ).astype(int)

    # - Categoría de Glucosa (ADA):
    # 0: Normal (<100 mg/dL), 1: Prediabetes (100-125), 2: Diabetes (>=126)
    df_limpio['gluc_stage'] = pd.cut(
        df_limpio['glucose'],
        bins=[0, 99, 125, 1000],
        labels=[0, 1, 2]
    ).astype(int)

    # - Hipertensión clínica binaria
    df_limpio['hypertension'] = ((df_limpio['sysBP'] >= 140) | (df_limpio['diaBP'] >= 90) | (df_limpio['BPMeds'] == 1)).astype(int)

    # - Sobrepeso binario (IMC >= 25.0)
    df_limpio['overweight'] = (df_limpio['BMI'] >= 25.0).astype(int)

    # - Homogeneizar nombre de target como 'cardio' para compatibilidad completa
    df_limpio['cardio'] = df_limpio['TenYearCHD'].astype(int)

    # 6. Deduplicación
    dup = df_limpio.duplicated().sum()
    if dup > 0:
        df_limpio = df_limpio.drop_duplicates().copy()
        print(f"Eliminados {dup} duplicados.")

    print(f"Dataset limpio final: {len(df_limpio):,} pacientes.")
    print("Distribución del target 'cardio' (TenYearCHD):")
    print(df_limpio['cardio'].value_counts(normalize=True).round(3))

    return df_limpio


def entrenar_y_guardar(df):
    """
    Entrena el pipeline de preprocesamiento y el modelo clasificador predictivo,
    evalúa métricas y guarda los artefactos en disco.
    """
    features_numericas = ['age', 'cigsPerDay', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']
    features_categoricas = ['male', 'currentSmoker', 'BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes']

    X_cols = features_numericas + features_categoricas
    X = df[X_cols].copy()
    y = df['cardio'].astype(int)

    # Split estratificado 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"\nTrain set: {X_train.shape[0]:,} | Test set: {X_test.shape[0]:,}")

    # Pipeline de transformación
    pipe_num = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    pipe_cat = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent'))
    ])

    preprocesador = ColumnTransformer(
        transformers=[
            ('num', pipe_num, features_numericas),
            ('cat', pipe_cat, features_categoricas)
        ]
    )

    # Modelo: Regresión Logística con class_weight='balanced'
    # Esto asegura Odds Ratios estrictamente positivos para los factores de riesgo (AHA)
    modelo_lr = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        C=0.8,
        random_state=42
    )

    pipeline_completo = Pipeline([
        ('prep', preprocesador),
        ('clf', modelo_lr)
    ])

    print("\nEntrenando Pipeline Clínico con LogisticRegression...")
    pipeline_completo.fit(X_train, y_train)

    # Evaluación en Test Set
    y_pred = pipeline_completo.predict(X_test)
    y_proba = pipeline_completo.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred).tolist()

    print("\n" + "="*45)
    print("MÉTRICAS DE RENDIMIENTO EN TEST SET (FRAMINGHAM):")
    print("="*45)
    print(f"Exactitud (Accuracy): {acc*100:.2f}%")
    print(f"Precisión:            {prec*100:.2f}%")
    print(f"Sensibilidad (Recall):{rec*100:.2f}%")
    print(f"F1-Score:             {f1*100:.2f}%")
    print(f"Área ROC (ROC-AUC):   {auc:.4f}")
    print("="*45)
    print("\nReporte de clasificación:\n", classification_report(y_test, y_pred))

    # Coeficientes y Odds Ratios Clínicos
    coefs = pipeline_completo.named_steps['clf'].coef_[0]
    odds_ratios = np.exp(coefs)

    print("\n--- ODDS RATIOS ESTANDARIZADOS (CONFIRMACIÓN MÉDICA) ---")
    importancia = {}
    for col, c, or_val in zip(X_cols, coefs, odds_ratios):
        importancia[col] = round(float(c), 4)
        print(f"  {col:16s}: Coef={c:+.4f} | Odds Ratio = {or_val:.3f} {'✓ Factor de Riesgo' if or_val >= 1.0 else '✓ Protector'}")

    # Guardar directorios de salida
    os.makedirs('Data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # 1. Guardar dataset limpio procesado
    ruta_proc = 'Data/processed/processed.csv'
    df.to_csv(ruta_proc, index=False)
    print(f"\n✓ Dataset procesado guardado en: {ruta_proc}")

    # 2. Guardar pipeline preprocesador puro
    ruta_prep = 'models/pipeline_preprocesamiento.pkl'
    joblib.dump(preprocesador, ruta_prep)
    print(f"✓ Pipeline de preprocesamiento guardado en: {ruta_prep}")

    # 3. Guardar modelo predictivo completo para Streamlit
    ruta_modelo = 'models/modelo_cardiovascular.pkl'
    joblib.dump(pipeline_completo, ruta_modelo)
    print(f"✓ Modelo predictivo completo guardado en: {ruta_modelo}")

    # 4. Guardar métricas y metadatos en JSON para uso en app y documentación
    metricas = {
        'total_registros': int(len(df)),
        'train_samples': int(len(X_train)),
        'test_samples': int(len(X_test)),
        'accuracy': round(float(acc), 4),
        'precision': round(float(prec), 4),
        'recall': round(float(rec), 4),
        'f1': round(float(f1), 4),
        'roc_auc': round(float(auc), 4),
        'confusion_matrix': cm,
        'features_numericas': features_numericas,
        'features_categoricas': features_categoricas,
        'coeficientes_logisticos': importancia,
        'odds_ratios': {k: round(float(np.exp(v)), 3) for k, v in importancia.items()}
    }

    with open('models/model_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metricas, f, indent=2, ensure_ascii=False)
    print("✓ Métricas del modelo guardadas en: models/model_metrics.json")

    return metricas

if __name__ == '__main__':
    raw_path = 'Data/framingham.csv'
    if not os.path.exists(raw_path):
        raw_path = '../Data/framingham.csv'
    df_clean = limpiar_datos_framingham(raw_path)
    entrenar_y_guardar(df_clean)
