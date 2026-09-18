# 🫀 Predicción de Riesgo Cardiovascular en Pacientes Reales

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.58-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9-F7931E.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Producci%C3%B3n-success.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Sistema integral de ciencia de datos, análisis exploratorio clínico y machine learning para la **evaluación, estratificación y predicción del riesgo de enfermedad cardiovascular** a partir de biomarcadores hemodinámicos y hábitos de vida en pacientes reales.

---

## 📑 Tabla de Contenidos
1. [Pregunta Clínica y Objetivos del Proyecto](#-pregunta-clínica-y-objetivos)
2. [Dataset y Criterios Fisiológicos para Pacientes Reales](#-dataset-y-criterios-fisiológicos)
3. [Diccionario Clínico de Variables](#-diccionario-clínico-de-variables)
4. [Metodología: Guía de Preprocesamiento en 19 Pasos](#-metodología-guía-de-preprocesamiento-en-19-pasos)
5. [Hallazgos Principales del Análisis Exploratorio](#-hallazgos-principales-del-eda)
6. [Modelo Predictivo y Rendimiento](#-modelo-predictivo-y-rendimiento)
7. [CardioRisk Studio: Aplicación Interactiva en Streamlit](#-cardiorisk-studio-aplicación-interactiva)
8. [Instrucciones de Ejecución Local y Despliegue](#-instrucciones-de-ejecución)
9. [Estructura del Proyecto](#-estructura-del-proyecto)
10. [Autora y Licencia](#-autora-y-licencia)

---

## 🎯 Pregunta Clínica y Objetivos

> **¿Es posible predecir con alta fiabilidad la presencia de enfermedad cardiovascular utilizando únicamente parámetros clínicos ambulatorios de bajo costo (presión arterial, medidas antropométricas, colesterol y estilo de vida)?**

Las enfermedades cardiovasculares representan la **primera causa de muerte prematura en el mundo** según la Organización Mundial de la Salud (OMS). Este proyecto resuelve tres necesidades médicas esenciales:

| Caso de Uso | Aplicación Práctica | Destinatarios |
|---|---|---|
| **Screening Clínico Temprano** | Detección precoz de pacientes asintomáticos en consulta antes de pruebas costosas (ecocardiograma, angiografía). | Médicos de Atención Primaria |
| **Salud Pública Preventiva** | Identificación del impacto relativo de factores modificables (tabaco, sobrepeso, sedentarismo) para diseñar campañas focalizadas. | Epidemiólogos y Sistemas de Salud |
| **Estratificación de Riesgo** | Segmentación en 4 niveles de riesgo (Bajo, Moderado, Alto, Crítico) con recomendaciones personalizadas. | Clínicas y Pacientes |

---

## 🩺 Dataset y Criterios Fisiológicos

- **Fuente:** [Cardiovascular Disease Dataset (Kaggle - sulianova)](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset).
- **Muestra Original:** 70,000 observaciones.
- **Muestra Validada Final:** **67,514 pacientes reales**.

### Criterios de Compatibilidad Hemodinámica (OMS / ACC/AHA / ESC)
En el dataset bruto existen anomalías como presiones sistólicas negativas (-150 mmHg), errores de tipeo con ceros extra (16,020 mmHg), presiones diastólicas superiores a las sistólicas ($ap\_lo \ge ap\_hi$) o estaturas de 55 cm. Siguiendo las directrices internacionales del **American College of Cardiology (ACC/AHA 2017)** y de la **European Society of Cardiology (ESC/ESH 2018/2024)**, se definieron los siguientes rangos de validez fisiológica para adultos de 29 a 65 años:

* **Presión Sistólica ($ap\_hi$):** $80 \le ap\_hi \le 220\text{ mmHg}$ *(cifras <80 implican shock o colapso circulatorio incompatible con control ambulatorio; >220 corresponden a emergencias extremas o error de captura)*.
* **Presión Diastólica ($ap\_lo$):** $50 \le ap\_lo \le 130\text{ mmHg}$.
* **Consistencia Hemodinámica:** $ap\_hi > ap\_lo$ de forma estricta.
* **Presión de Pulso ($PP = ap\_hi - ap\_lo$):** $20 \le PP \le 110\text{ mmHg}$ *(una presión de pulso menor a 20 mmHg es fisiológicamente inviable en pacientes ambulatorios)*.
* **Estatura:** $140 \le \text{height} \le 205\text{ cm}$.
* **Peso:** $40 \le \text{weight} \le 165\text{ kg}$.
* **Índice de Masa Corporal (IMC):** $16.0 \le \text{BMI} \le 52.0\text{ kg/m}^2$.

---

## 📖 Diccionario Clínico de Variables

| Variable | Tipo | Unidad / Rango | Definición Clínica e Interpretación |
|---|---|---|---|
| `age_years` | Numérica | $29.6 - 64.9$ años | Edad del paciente calculada a partir de los días ($age / 365.25$). |
| `gender` | Categórica | 1 = Mujer · 2 = Hombre | Sexo biológico del paciente. |
| `height` | Numérica | $140 - 205$ cm | Estatura en bipedestación. |
| `weight` | Numérica | $40 - 165$ kg | Peso corporal medido en balanza clínica. |
| `ap_hi` | Numérica | $80 - 220$ mmHg | Presión arterial sistólica (presión pico durante la contracción ventricular). |
| `ap_lo` | Numérica | $50 - 130$ mmHg | Presión arterial diastólica (presión mínima durante la relajación ventricular). |
| `bmi` | Calculada | $16.0 - 52.0$ kg/m² | Índice de Masa Corporal ($weight / height^2$). Clasificado según la OMS. |
| `pulse_pressure`| Calculada | $20 - 110$ mmHg | Presión diferencial ($ap\_hi - ap\_lo$). Marcador directo de rigidez aórtica. |
| `map` | Calculada | $60 - 160$ mmHg | Presión Arterial Media ($ap\_lo + PP / 3$). Marcador de perfusión sistémica. |
| `cholesterol`| Categórica ordinal | 1, 2, 3 | Nivel de colesterol sérico total (1: Normal <200 · 2: Alto 200-239 · 3: Muy alto $\ge$240 mg/dL). |
| `gluc` | Categórica ordinal | 1, 2, 3 | Nivel de glucemia en ayuno (1: Normal <100 · 2: Alterada 100-125 · 3: Diabetes $\ge$126 mg/dL). |
| `smoke` | Binaria | 0 = No · 1 = Sí | Hábito tabáquico activo regular. |
| `alco` | Binaria | 0 = No · 1 = Sí | Consumo habitual de bebidas alcohólicas. |
| `active` | Binaria | 0 = No · 1 = Sí | Realización de actividad física regular ($\ge$150 min/semana). |
| `cardio` | Binaria (**Target**)| 0 = No · 1 = Sí | **Presencia confirmada de enfermedad cardiovascular**. |

---

## 🔬 Metodología: Guía de Preprocesamiento en 19 Pasos

El análisis y la ingeniería de datos se implementaron siguiendo rigurosamente la **Guía Maestra de Preprocesamiento** (`Notebooks/03_eda.ipynb`):

```
[1. Librerías] ────> [2. Carga de Datos] ───> [3. Exploración Inicial] ───> [4. Análisis de Nulos]
                                                                                   │
[8. Consistencia] <── [7. Tipos de Datos] <── [6. Duplicados] <── [5. Estrategia Nulos]
       │
       ▼
[9. Outliers Clínicos] ─> [10. Codificación] ──> [11. Escalado] ─────────> [12. Distribuciones]
                                                                                   │
[16. Split Train/Test] <─ [15. Balance Target] <─ [14. Correlaciones] <── [13. Feature Engineering]
       │
       ▼
[17. ColumnTransformer] ─> [18. Serialización de Modelos] ───────────────> [19. Checklist Final]
```

1. **Librerías especializadas:** Configuración de `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy` y `scikit-learn`.
2. **Carga robusta:** Carga con delimitador `;` y verificación de integridad.
3. **Exploración inicial:** Comprobación de dimensiones (70,000 filas, 13 columnas) y cardinalidades.
4. **Análisis de nulos:** Comprobación de 0 datos faltantes en la matriz.
5. **Tratamiento de nulos:** Inclusión preventiva de `SimpleImputer(strategy='median')` en el pipeline de producción.
6. **Detección de duplicados:** Identificación y eliminación de **674 duplicados clínicos exactos**.
7. **Tipos de datos:** Conversión de días a años (`age_years`) y tipado eficiente (`int8`).
8. **Consistencia hemodinámica:** Eliminación de **1,236 filas** con presión invertida ($ap\_lo \ge ap\_hi$).
9. **Outliers fisiológicos:** Filtrado de **463 registros** fuera de los límites de vida humana ambulatoria.
10. **Codificación de categóricas:** Mapeo semántico para gráficos y numérico ordinal para modelos.
11. **Escalado y normalización:** Aplicación de `StandardScaler` sobre variables continuas en el pipeline.
12. **Transformación de distribuciones:** Evaluación de asimetría (*skewness*) y estabilización de variables asimétricas.
13. **Feature Engineering Clínico:** Creación de `bmi`, `pulse_pressure`, `map`, `hypertension`, `overweight` y `bp_stage` (AHA).
14. **Selección y multicolinealidad:** Heatmap de correlación Pearson/Spearman y análisis de pares colineales.
15. **Evaluación de desbalance:** Comprobación de paridad **50.3% sanos vs 49.7% enfermos** (ratio 1.01). Se demostró que **no se requiere SMOTE**.
16. **División Train/Test:** Partición estratificada 80/20 (`stratify=y`, 54,011 train / 13,503 test) previa a cualquier ajuste para **prevenir data leakage**.
17. **Pipeline y ColumnTransformer:** Encapsulamiento de sub-pipelines numéricos y categóricos.
18. **Guardado de artefactos:** Exportación de `processed.csv`, `pipeline_preprocesamiento.pkl` y `modelo_cardiovascular.pkl`.
19. **Checklist final:** Validación de 15 puntos clínicos y computacionales.

---

## 📊 Hallazgos Principales del EDA

### 1. Balance de la Variable Objetivo
El dataset se encuentra en equilibrio óptimo para clasificación binaria sin necesidad de remuestreo sintético:
- **Sin enfermedad (`cardio = 0`):** 33,962 pacientes (50.3%)
- **Con enfermedad (`cardio = 1`):** 33,552 pacientes (49.7%)
- **Ratio mayoría/minoría:** `1.01`

![Balance de la Variable Objetivo](img/01_target_balance.png)

### 2. Diferencias en Variables Numéricas Clave
Los pacientes con enfermedad cardiovascular presentan un desplazamiento estadísticamente significativo en todas las variables continuas:
- **Presión Sistólica (`ap_hi`):** Media de **132.8 mmHg** en enfermos vs **121.0 mmHg** en sanos ($\Delta = +11.8\text{ mmHg}$, $p < 0.001$).
- **Presión de Pulso (`pulse_pressure`):** Media de **48.7 mmHg** en enfermos vs **38.5 mmHg** en sanos ($\Delta = +10.2\text{ mmHg}$, rigidez arterial marcada).
- **Edad (`age_years`):** Media de **54.9 años** en enfermos vs **51.7 años** en sanos ($\Delta = +3.2\text{ años}$).
- **Índice de Masa Corporal (`bmi`):** Media de **28.5 kg/m²** en enfermos vs **26.5 kg/m²** en sanos.

![Violinplots de Variables Numéricas](img/s2a_violinplots.png)

### 3. Prevalencia de Riesgo según Factores de Vida y Metabólicos
- **Efecto Umbral del Colesterol:** El riesgo salta drásticamente del **44.0%** en colesterol normal al **60.2%** en colesterol alto y alcanza el **76.5%** en colesterol muy alto.
- **Glucosa Alterada:** Pacientes con glucemia muy alta superan el **62%** de prevalencia de afección cardiovascular.
- **Sedentarismo:** Los pacientes inactivos presentan una tasa de enfermedad superior a los que realizan ejercicio regular.

![Factores Categóricos vs Riesgo](img/s2b_categoricas_vs_cardio.png)

### 4. Matriz de Correlación
Ninguna variable individual supera $r = 0.45$ de correlación lineal con `cardio`. Esto demuestra empíricamente que **el riesgo cardiovascular es de naturaleza multifactorial** y requiere modelos capaces de capturar interacciones no lineales.

![Matriz de Correlaciones](img/s4a_heatmap_correlaciones.png)

---

## 🤖 Modelo Predictivo y Rendimiento

Se implementó un pipeline en `scikit-learn` acoplado a un clasificador **HistGradientBoostingClassifier** (optimizado para árboles de decisión basados en histogramas, robusto ante valores extremos y capaz de modelar interacciones complejas):

### Métricas de Evaluación en Test Set Independiente (13,503 pacientes)
| Métrica | Valor Obtenido | Interpretación Clínica |
|---|:---:|---|
| **ROC-AUC** | **0.7989 (~0.80)** | Capacidad de discriminación excelente entre enfermos y sanos. |
| **Exactitud (Accuracy)** | **73.07%** | Porcentaje global de clasificaciones correctas. |
| **Precisión (Precision)** | **75.09%** | Cuando el modelo predice enfermedad, acierta 3 de cada 4 veces. |
| **Exhaustividad (Recall)** | **68.60%** | Sensibilidad para capturar pacientes verdaderamente enfermos. |
| **F1-Score** | **71.70%** | Balance armónico entre precisión y cobertura. |

### Matriz de Confusión en Test
```
                       Predicho Sano (0)    Predicho Enfermo (1)
Verdadero Sano (0)           5,261                 1,528
Verdadero Enfermo (1)        2,108                 4,606
```

### Importancia Multivariada (Odds Ratios Estandarizados - Regresión Logística)
1. **Colesterol Sérico Total:** $\beta = +0.4997$ (Factor metabólico de mayor impacto).
2. **Edad (`age_years`):** $\beta = +0.3472$ (Factor acumulativo no modificable).
3. **Presión Sistólica (`ap_hi`):** $\beta = +0.3263$ (Tensión hemodinámica parietal).
4. **Presión de Pulso (`pulse_pressure`):** $\beta = +0.3235$ (Marcador de esclerosis arterial).
5. **Presión Arterial Media (`map`):** $\beta = +0.3178$.
6. **Actividad Física regular (`active`):** $\beta = -0.2317$ (Factor protector fundamental).

---

## 🎛️ CardioRisk Studio: Aplicación Interactiva

La aplicación web construida en **Streamlit** (`App/app.py`) fue completamente rediseñada bajo una estética de monitor médico dark navy con señal ECG:

### Módulos Principales:
1. **🧮 Calculadora Clínica de Riesgo Individual:**
   - Controles interactivos limitados a rangos fisiológicos reales.
   - Cálculo instantáneo de **IMC** (con badge de la OMS: Normopeso, Sobrepeso, Obesidad I/II/III).
   - Cálculo de **Presión de Pulso** y **Presión Arterial Media (PAM)**.
   - Diagnóstico del **Estadío de Hipertensión según la AHA/ACC 2017**.
   - **Predicción de probabilidad de riesgo (%) por Machine Learning** con gauge visual de 4 niveles (*Bajo <25%*, *Moderado 25-50%*, *Alto 50-75%*, *Crítico >75%*).
   - **Recomendaciones clínicas personalizadas** basadas en los factores específicos del paciente.
2. **👥 Comparador Poblacional:**
   - Gráficos de densidad de Kernel que ubican la presión y el IMC del paciente en percentiles respecto a los 67,514 casos de la cohorte.
3. **📊 Variable Objetivo:**
   - Análisis interactivo del balance de clases.
4. **📈 Variables Numéricas:**
   - Histogramas y gráficos de violín dinámicos filtrables por edad y género.
5. **🏷️ Variables Categóricas:**
   - Prevalencia cruzada de tabaquismo, colesterol, glucosa y sedentarismo.
6. **🔥 Factores e Importancia:**
   - Matriz de correlación de Pearson y coeficientes beta del modelo.

---

## 🚀 Instrucciones de Ejecución

### 1. Clonar el repositorio y configurar el entorno
```bash
# Clonar
git clone https://github.com/anacolinaarismendi/Cardiovascular-Disease.git
cd "Cardiovascular Disease"

# Crear y activar entorno virtual
python -m venv .venv
source .venv/bin/activate       # macOS / Linux
# .venv\Scripts\activate        # Windows

# Instalar dependencias oficiales
pip install -r requirements.txt
```

### 2. Ejecutar el pipeline de datos y entrenamiento (opcional)
Los datos procesados y modelos ya vienen pre-entrenados en el repositorio, pero puedes re-ejecutarlos cuando desees:
```bash
python src/train_model.py
```

### 3. Lanzar la aplicación Streamlit
La aplicación cuenta con **resolución dinámica de rutas** y puede ejecutarse indistintamente desde cualquier ubicación:
```bash
# Desde la raíz del repositorio:
streamlit run App/app.py

# O desde la carpeta App:
cd App
streamlit run app.py
```
La aplicación se abrirá automáticamente en tu navegador en `http://localhost:8501`.

### 4. Despliegue en Streamlit Cloud
1. Realiza el push de tu repositorio a GitHub asegurando que `Data/processed/processed.csv`, `models/` y `requirements.txt` estén incluidos.
2. Ingresa a [share.streamlit.io](https://share.streamlit.io).
3. Selecciona tu repositorio y rama `main`.
4. En **Main file path**, escribe: `App/app.py`.
5. Haz clic en **Deploy**.

---

## 📁 Estructura del Proyecto

```
Cardiovascular-Disease/
├── README.md                          # Documentación maestra del proyecto
├── LICENSE                            # Licencia de código abierto MIT
├── requirements.txt                   # Dependencias de Python requeridas
├── App/
│   └── app.py                         # Aplicación web interactiva Streamlit
├── Data/
│   ├── cardio_train.csv               # Dataset original de Kaggle (70,000 registros)
│   └── processed/
│       └── processed.csv              # Dataset limpio con plausibilidad fisiológica (67,514 registros)
├── Notebooks/
│   ├── 01_exploracion.ipynb           # Primer acercamiento a los datos brutos
│   ├── 02_preprocesamiento.ipynb      # Pruebas preliminares de transformación
│   └── 03_eda.ipynb                   # EDA Completo estructurado en los 19 pasos de la guía
├── models/
│   ├── pipeline_preprocesamiento.pkl  # Pipeline de transformación ColumnTransformer
│   ├── modelo_cardiovascular.pkl      # Modelo predictivo completo entrenado
│   └── model_metrics.json             # Métricas de validación y coeficientes
├── src/
│   ├── train_model.py                 # Script de limpieza clínica, pipeline y entrenamiento
│   ├── generate_eda_notebook.py       # Generador del notebook 03_eda.ipynb en 19 pasos
│   └── export_plots.py                # Generador de gráficos de alta resolución
└── img/                               # Gráficos clínicos exportados
    ├── 01_target_balance.png
    ├── s2a_violinplots.png
    ├── s2b_categoricas_vs_cardio.png
    └── s4a_heatmap_correlaciones.png
```

---

## 👩‍💻 Autora y Licencia

- **Autora:** Ana Colina Arismendi
- **GitHub:** [@anacolinaarismendi](https://github.com/anacolinaarismendi)
- **Licencia:** Este proyecto se distribuye bajo la licencia [MIT](LICENSE).
