# Predicción de Riesgo Cardiovascular

Proyecto de análisis de datos y machine learning para predecir la presencia de enfermedad cardiovascular a partir de variables clínicas y de estilo de vida.

🔗 **App en vivo:** http://localhost:8503

---

## Pregunta que guía el proyecto

> ¿Es posible predecir si un paciente tiene enfermedad cardiovascular a partir de indicadores clínicos básicos como la presión arterial, el colesterol y el estilo de vida?

Se identificaron tres usos concretos para este modelo:

| Uso | Descripción | A quién sirve |
|---|---|---|
| Screening clínico | Alerta temprana antes de exámenes costosos | Médicos de atención primaria |
| Salud pública | Identificar qué factores pesan más para enfocar campañas preventivas | Equipos de salud pública |
| Segmentación | Agrupar pacientes por perfil de riesgo combinado | Aseguradoras y sistemas de salud |

---

## Dataset

| Campo | Detalle |
|---|---|
| Nombre | Cardiovascular Disease Dataset |
| Fuente | [Kaggle — sulianova](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset) |
| Licencia | Uso público (Kaggle) |
| Filas (tras limpieza) | 68,640 pacientes |
| Variable objetivo | `cardio` (0 = sin enfermedad, 1 = con enfermedad) |

### Variables del dataset

| Variable | Tipo | Descripción |
|---|---|---|
| `age_years` | Numérica | Edad en años |
| `gender` | Categórica | Género (1 = mujer, 2 = hombre) |
| `height`, `weight` | Numérica | Talla (cm) y peso (kg) |
| `ap_hi`, `ap_lo` | Numérica | Presión sistólica y diastólica (mmHg) |
| `bmi` | Numérica calculada | Índice de masa corporal |
| `pulse_pressure` | Numérica calculada | Presión diferencial (ap_hi − ap_lo) |
| `cholesterol`, `gluc` | Categórica ordinal | 1 = normal · 2 = alto · 3 = muy alto |
| `smoke`, `alco`, `active` | Binaria | Fuma, bebe alcohol, hace actividad física |
| `cardio` | Binaria | **Variable objetivo** |

---

## Hallazgos principales

- El dataset está prácticamente balanceado (50.5% / 49.5%), apto para clasificación binaria sin técnicas de balanceo adicionales.
- La **presión diferencial** (`pulse_pressure`) es la variable individual con mayor poder discriminante (+19.0% de diferencia entre grupos), seguida de la **presión sistólica** (+11.9%).
- El **IMC** es el factor modificable más relevante (+7.5%) — el único de los tres factores principales sobre el que se puede intervenir directamente.
- El **colesterol alto/muy alto** muestra una relación por umbral, no lineal: el riesgo salta notablemente al pasar de "normal" a "alto".
- Ninguna variable individual supera 0.5 de correlación directa con `cardio` — el riesgo cardiovascular es multifactorial.

---

## Estructura del proyecto

```
mi-proyecto-m1/
├── README.md
├── requirements.txt
├── Data/
│   ├── raw/                    # Dataset original de Kaggle (no versionado)
│   └── processed/
│       └── processed.csv       # Datos limpios, listos para análisis
├── Notebooks/
│   ├── 01_exploracion.ipynb
│   ├── 02_preprocesamiento.ipynb
│   └── 03_eda.ipynb
├── App/
│   └── app.py                  # Dashboard interactivo Streamlit
└── img/                        # Gráficos exportados para el README
```

---

## Cómo ejecutar localmente

```bash
# 1. Clonar el repositorio
git clone https://github.com/tu-usuario/mi-proyecto-m1.git
cd mi-proyecto-m1

# 2. Crear entorno virtual
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
# .venv\Scripts\activate       # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Añadir el dataset
# Descargar cardio_train.csv desde Kaggle y colocarlo en Data/raw/
# Ejecutar los notebooks 01 → 02 para generar Data/processed/processed.csv

# 5. Lanzar la app
cd App
streamlit run app.py
```

---

## Cómo desplegar en Streamlit Cloud

1. Sube el repositorio completo a GitHub, **incluyendo** `Data/processed/processed.csv` (los datos procesados sí se versionan; el raw no).
2. Entra a [share.streamlit.io](https://share.streamlit.io) y conecta tu cuenta de GitHub.
3. Selecciona el repositorio, la rama `main`, y como **Main file path** indica:
   ```
   App/app.py
   ```
4. Streamlit Cloud detecta automáticamente `requirements.txt` en la raíz del repo e instala las dependencias.
5. Despliega — la URL pública queda disponible en un par de minutos.

---

## Autora

**Ana Colina Arismendi** ·