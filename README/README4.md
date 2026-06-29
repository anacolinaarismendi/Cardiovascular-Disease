# Proyecto de Predicción de Enfermedades Cardiovasculares

Este proyecto está diseñado para analizar datos médicos relacionados con enfermedades cardiovasculares y construir un modelo capaz de predecir el riesgo de enfermedad en pacientes.

## Objetivos

- Analizar variables clínicas y demográficas asociadas a enfermedades cardiovasculares.
- Construir un modelo de clasificación para identificar pacientes con riesgo elevado.
- Documentar el proceso y resultados del análisis.

## Contenido del proyecto

- `data/`: conjunto de datos y archivos relacionados.
- `notebooks/`: análisis exploratorio y pruebas de modelos en notebooks.
- `src/`: código fuente para el procesamiento de datos y entrenamiento del modelo.
- `README/`: documentación del proyecto.

## Requisitos

- Python 3.8 o superior
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## Instalación

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/usuario/Proyecto-Cardio.git
   ```
2. Entrar en el directorio del proyecto:
   ```bash
   cd "Cardiovascular Disease"
   ```
3. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```

## Uso

1. Preprocesar los datos.
2. Ejecutar el entrenamiento del modelo.
3. Evaluar la predicción con datos de prueba.

Ejemplo:
```bash
python src/train_model.py
```

## Resultados esperados

- Informe de análisis exploratorio.
- Métricas de rendimiento del modelo (precisión, recall, F1).
- Modelo entrenado para inferencia.

## Notas

Este README proporciona una visión general del proyecto y puede adaptarse según los requisitos específicos de los datos y el enfoque de modelado.
