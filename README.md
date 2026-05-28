# Ejercicio 1 — Análisis de Sentimientos en Reseñas de Productos

Proyecto en Python para reemplazar el flujo propuesto en KNIME usando un **Pipeline de scikit-learn**: limpieza básica, TF-IDF, Naive Bayes, evaluación y guardado del modelo.

## ¿Por qué Pipeline y no PyTorch/TensorFlow?

Para este ejercicio, la opción más alineada con el enunciado es **Pipeline + TF-IDF + Naive Bayes**. Es rápida, explicable, fácil de evaluar y reproduce casi exactamente los nodos de KNIME: Text Processing, TF-IDF, Naive Bayes Learner, Scorer y Model Writer.

PyTorch/TensorFlow servirían para modelos más avanzados, pero requieren más tiempo, GPU opcional, embeddings y más ajustes. Para entregar el ejercicio CRISP-DM, esta solución es más limpia.

## Estructura

```text
amazon_sentiment_project/
├── data/
│   ├── raw/                 # dataset descargado desde Kaggle
│   └── processed/
├── models/                  # modelo .joblib
├── reports/
│   ├── metrics.json
│   ├── classification_report.txt
│   └── figures/confusion_matrix.png
├── scripts/
│   ├── eda.py
│   ├── train.py
│   └── predict.py
├── src/amazon_sentiment/
│   ├── data.py
│   ├── download_data.py
│   ├── evaluate.py
│   └── model.py
├── requirements.txt
└── pyproject.toml
```

## 1. Crear entorno

```bash
cd amazon_sentiment_project
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

## 2. Instalar dependencias

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## 3. Configurar Kaggle

1. En Kaggle: Account → Create New API Token.
2. Descarga `kaggle.json`.
3. Colócalo en:

Windows:

```text
C:\Users\TU_USUARIO\.kaggle\kaggle.json
```

macOS/Linux:

```text
~/.kaggle/kaggle.json
```

En macOS/Linux ejecuta:

```bash
chmod 600 ~/.kaggle/kaggle.json
```

## 4. Descargar dataset automáticamente

```bash
python -m amazon_sentiment.download_data
```

Esto descarga `bittlingmayer/amazonreviews` y descomprime los archivos en `data/raw/`.

Para probar el flujo sin Kaggle:

```bash
python -m amazon_sentiment.download_data --tiny-sample
```

## 5. Comprensión de datos / EDA

```bash
python scripts/eda.py --limit 10000
```

## 6. Entrenar modelo

Entrenamiento recomendado para empezar:

```bash
python scripts/train.py --limit-train 200000 --limit-test 50000
```

Usar todo el dataset:

```bash
python scripts/train.py --limit-train 0 --limit-test 0
```

Salidas:

- `models/sentiment_pipeline.joblib`
- `reports/metrics.json`
- `reports/classification_report.txt`
- `reports/figures/confusion_matrix.png`

## 7. Probar predicciones

```bash
python scripts/predict.py "This product is excellent and arrived fast"
python scripts/predict.py "Very bad quality, I want a refund"
```

## Fases CRISP-DM cubiertas

1. **Comprensión del negocio:** las reseñas influyen en reputación, intención de compra, devoluciones y priorización de mejoras.
2. **Comprensión de los datos:** `scripts/eda.py` explora tamaño, distribución de clases, ejemplos y longitud de reseñas.
3. **Preparación de datos:** se eliminan nulos/duplicados y se transforma texto con TF-IDF.
4. **Modelado:** `MultinomialNB` dentro de un `Pipeline`.
5. **Evaluación:** Accuracy, Precision, Recall, F1-score y Matriz de Confusión.
6. **Despliegue:** guardado del pipeline completo con `joblib` para reutilizarlo en predicciones.

## Petición recomendada para Codex

Copia y pega esto en Codex:

```text
Tengo un proyecto Python para el Ejercicio 1: Análisis de Sentimientos en Amazon Reviews con metodología CRISP-DM. Quiero que revises el repositorio y verifiques que el flujo completo funciona: instalación, descarga con Kaggle API, EDA, entrenamiento con TF-IDF + Multinomial Naive Bayes, evaluación con accuracy/precision/recall/F1/matriz de confusión y predicción con modelo guardado. Si algo falla, corrígelo manteniendo una estructura simple y documentada. No cambies el objetivo del ejercicio ni reemplaces el modelo base salvo que propongas una mejora opcional separada.
```

## Mejora opcional

Después de entregar la versión base, puedes agregar un segundo modelo con `LogisticRegression` o un modelo Transformer. Compara ambos contra Naive Bayes usando las mismas métricas.
