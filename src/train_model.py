"""
Script de preprocesamiento clínico, validación fisiológica y entrenamiento
del modelo predictivo de riesgo cardiovascular.
Basado en las directrices de la Guía de Preprocesamiento de 19 pasos
y en las guías médicas internacionales (OMS, AHA/ACC y ESC).
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

def limpiar_datos_clinicos(ruta_raw):
    """
    Carga y limpia el dataset original de Kaggle aplicando
    criterios de plausibilidad fisiológica para pacientes reales.
    """
    print(f"Cargando dataset desde: {ruta_raw}")
    df = pd.read_csv(ruta_raw, sep=';')
    total_inicial = len(df)
    print(f"Total registros iniciales: {total_inicial:,}")

    # 1. Eliminar identificador sin valor clínico predictivo
    if 'id' in df.columns:
        df = df.drop(columns=['id'])

    # 2. Conversión de edad: días a años redondeado a 1 decimal
    df['age_years'] = (df['age'] / 365.25).round(1)
    df = df.drop(columns=['age'])

    # 3. Consistencia hemodinámica estricta: ap_hi > ap_lo
    filas_invertidas = (df['ap_lo'] >= df['ap_hi']).sum()
    print(f"Eliminando {filas_invertidas:,} filas con presión invertida (ap_lo >= ap_hi)...")
    df = df[df['ap_hi'] > df['ap_lo']].copy()

    # 4. Filtros fisiológicos de plausibilidad para pacientes adultos (OMS / AHA / ESC):
    # - Presión Sistólica: 80 a 220 mmHg
    # - Presión Diastólica: 50 a 130 mmHg
    # - Presión de Pulso (ap_hi - ap_lo): 20 a 110 mmHg
    # - Estatura: 140 a 205 cm
    # - Peso: 40 a 165 kg
    condicion_fisiologica = (
        (df['ap_hi'] >= 80) & (df['ap_hi'] <= 220) &
        (df['ap_lo'] >= 50) & (df['ap_lo'] <= 130) &
        ((df['ap_hi'] - df['ap_lo']) >= 20) & ((df['ap_hi'] - df['ap_lo']) <= 110) &
        (df['height'] >= 140) & (df['height'] <= 205) &
        (df['weight'] >= 40) & (df['weight'] <= 165)
    )

    df_limpio = df[condicion_fisiologica].copy()
    print(f"Filas tras filtros fisiológicos: {len(df_limpio):,} (removidas: {len(df) - len(df_limpio):,})")

    # 5. Feature Engineering Clínico
    # - IMC (Índice de Masa Corporal)
    df_limpio['bmi'] = (df_limpio['weight'] / ((df_limpio['height'] / 100) ** 2)).round(1)
    # Acotar IMC fisiológicamente entre 16.0 y 52.0
    df_limpio = df_limpio[(df_limpio['bmi'] >= 16.0) & (df_limpio['bmi'] <= 52.0)].copy()

    # - Presión diferencial o de pulso (PP)
    df_limpio['pulse_pressure'] = df_limpio['ap_hi'] - df_limpio['ap_lo']

    # - Presión Arterial Media (PAM / MAP): PAM = ap_lo + (pulse_pressure / 3)
    df_limpio['map'] = (df_limpio['ap_lo'] + (df_limpio['pulse_pressure'] / 3)).round(1)

    # - Hipertensión binaria (criterio clínico: ap_hi >= 140 o ap_lo >= 90)
    df_limpio['hypertension'] = (
        (df_limpio['ap_hi'] >= 140) | (df_limpio['ap_lo'] >= 90)
    ).astype(int)

    # - Sobrepeso binario (IMC >= 25.0)
    df_limpio['overweight'] = (df_limpio['bmi'] >= 25.0).astype(int)

    # - Estadio de Presión Arterial AHA/ACC 2017:
    # 0: Normal (<120 y <80)
    # 1: Elevada (120-129 y <80)
    # 2: Hipertensión Grado 1 (130-139 o 80-89)
    # 3: Hipertensión Grado 2 (>=140 o >=90)
    def clasificar_aha(row):
        hi, lo = row['ap_hi'], row['ap_lo']
        if hi >= 140 or lo >= 90:
            return 3  # HTA Grado 2
        elif (130 <= hi <= 139) or (80 <= lo <= 89):
            return 2  # HTA Grado 1
        elif (120 <= hi <= 129) and (lo < 80):
            return 1  # Elevada
        else:
            return 0  # Normal

    df_limpio['bp_stage'] = df_limpio.apply(clasificar_aha, axis=1)

    # 6. Duplicados clínicos
    duplicados = df_limpio.duplicated().sum()
    if duplicados > 0:
        print(f"Eliminando {duplicados:,} registros duplicados exactos...")
        df_limpio = df_limpio.drop_duplicates().copy()

    print(f"Dataset limpio final: {len(df_limpio):,} filas y {df_limpio.shape[1]} columnas.")
    print(f"Balance del target 'cardio':\n{df_limpio['cardio'].value_counts(normalize=True).round(3)}")

    return df_limpio


def entrenar_y_guardar(df):
    """
    Entrena el pipeline de preprocesamiento y el modelo clasificador predictivo,
    evalúa métricas y guarda los artefactos en disco.
    """
    # Definir variables
    features_numericas = ['age_years', 'height', 'weight', 'ap_hi', 'ap_lo',
                          'bmi', 'pulse_pressure', 'map']
    features_categoricas = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']

    X_cols = features_numericas + features_categoricas
    X = df[X_cols].copy()
    y = df['cardio'].astype(int)

    # Split estratificado 80 / 20 para evitar data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"\nTrain set: {X_train.shape[0]:,} muestras")
    print(f"Test set:  {X_test.shape[0]:,} muestras")

    # Pipeline de transformación numérica y categórica
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

    # Modelo: HistGradientBoostingClassifier para capturar interacciones no lineales
    modelo_hgb = HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.08,
        max_depth=6,
        random_state=42
    )

    pipeline_completo = Pipeline([
        ('prep', preprocesador),
        ('clf', modelo_hgb)
    ])

    # Ajustar SOLO con datos de entrenamiento
    print("\nEntrenando Pipeline con HistGradientBoostingClassifier...")
    pipeline_completo.fit(X_train, y_train)

    # Evaluación en Test
    y_pred = pipeline_completo.predict(X_test)
    y_proba = pipeline_completo.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred).tolist()

    print("\n" + "="*45)
    print("MÉTRICAS DE RENDIMIENTO EN TEST SET:")
    print("="*45)
    print(f"Exactitud (Accuracy): {acc*100:.2f}%")
    print(f"Precisión:            {prec*100:.2f}%")
    print(f"Exhaustividad/Recall: {rec*100:.2f}%")
    print(f"F1-Score:             {f1*100:.2f}%")
    print(f"Área ROC (ROC-AUC):   {auc:.4f}")
    print("="*45)
    print("\nReporte de clasificación:\n", classification_report(y_test, y_pred))

    # También entrenamos una Regresión Logística para coeficientes interpretables
    pipe_lr = Pipeline([
        ('prep', preprocesador),
        ('clf', LogisticRegression(max_iter=1000, random_state=42))
    ])
    pipe_lr.fit(X_train, y_train)
    auc_lr = roc_auc_score(y_test, pipe_lr.predict_proba(X_test)[:, 1])
    print(f"ROC-AUC Regresión Logística (referencia lineal): {auc_lr:.4f}")

    # Coeficientes de Regresión Logística para explicabilidad clínica
    coefs = pipe_lr.named_steps['clf'].coef_[0]
    importancia_features = sorted(
        zip(X_cols, [round(float(c), 4) for c in coefs]),
        key=lambda x: abs(x[1]),
        reverse=True
    )

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
        'roc_auc_lr': round(float(auc_lr), 4),
        'confusion_matrix': cm,
        'features_numericas': features_numericas,
        'features_categoricas': features_categoricas,
        'coeficientes_logisticos': dict(importancia_features)
    }

    with open('models/model_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metricas, f, indent=2, ensure_ascii=False)
    print("✓ Métricas del modelo guardadas en: models/model_metrics.json")

    return metricas

if __name__ == '__main__':
    raw_path = 'Data/cardio_train.csv'
    if not os.path.exists(raw_path):
        raw_path = '../Data/cardio_train.csv'
    df_clean = limpiar_datos_clinicos(raw_path)
    entrenar_y_guardar(df_clean)
