# Analisis de Sentimientos en Resenas de Amazon

Proyecto universitario en Python para clasificar resenas de productos de Amazon como positivas o negativas. El flujo usa `scikit-learn`, `TF-IDF` y `Multinomial Naive Bayes`, siguiendo la metodologia CRISP-DM.

## Objetivo

Construir y evaluar un modelo de analisis de sentimientos que permita identificar si una resena de producto expresa una opinion positiva o negativa.

## Dataset

Dataset usado: `bittlingmayer/amazonreviews` de Kaggle.

El dataset viene en formato FastText:

- `__label__1`: resena negativa.
- `__label__2`: resena positiva.

Los archivos principales se descargan y descomprimen en `data/raw/`:

- `train.ft.txt`
- `test.ft.txt`

## Metodologia CRISP-DM

1. Comprension del negocio: las resenas afectan reputacion, ventas, devoluciones y decisiones de mejora de productos.
2. Comprension de datos: se revisa distribucion de clases, longitud promedio y ejemplos de resenas positivas y negativas.
3. Preparacion de datos: se leen archivos FastText, se transforman etiquetas y se eliminan nulos/duplicados.
4. Modelado: se entrena un `Pipeline` con `TfidfVectorizer` y `MultinomialNB`.
5. Evaluacion: se calculan Accuracy, Precision, Recall, F1-score y Matriz de Confusion.
6. Despliegue: se guarda el pipeline completo con `joblib` para usarlo desde terminal con nuevas resenas.

## Estructura

```text
amazon_sentiment_project/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── reports/
├── scripts/
│   ├── eda.py
│   ├── train.py
│   ├── predict.py
│   └── decompress_bz2.py
├── src/
│   └── amazon_sentiment/
│       ├── data.py
│       ├── download_data.py
│       ├── evaluate.py
│       └── model.py
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Instalacion

Crear y activar entorno virtual:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

Instalar dependencias:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## Configuracion de Kaggle API

En Kaggle, ir a Account y crear un API Token. Eso descarga un archivo `kaggle.json`.

Opcion recomendada para este proyecto: definir la variable `KAGGLE_API_TOKEN` con el contenido JSON del token:

```powershell
$env:KAGGLE_API_TOKEN = '{"username":"TU_USUARIO","key":"TU_API_KEY"}'
```

Tambien puedes definirla como ruta al archivo:

```powershell
$env:KAGGLE_API_TOKEN = "C:\Users\TU_USUARIO\.kaggle\kaggle.json"
```

No subas `kaggle.json` al repositorio.

## Comandos de Ejecucion

Descargar y descomprimir dataset:

```powershell
python src\amazon_sentiment\download_data.py
```

Exploracion de datos:

```powershell
python scripts\eda.py --limit 10000
```

Entrenamiento recomendado para la entrega:

```powershell
python scripts\train.py --limit-train 50000 --limit-test 10000
```

Prediccion desde terminal:

```powershell
python scripts\predict.py "This product is amazing"
```

## Salidas Generadas

Despues del EDA:

- `reports/eda_summary.txt`

Despues del entrenamiento:

- `models/sentiment_pipeline.joblib`
- `reports/metrics.txt`
- `reports/metrics.json`
- `reports/classification_report.txt`
- `reports/confusion_matrix.png`

## Explicacion de Metricas

- Accuracy: proporcion total de predicciones correctas.
- Precision: de las resenas predichas como positivas, cuantas realmente eran positivas.
- Recall: de las resenas positivas reales, cuantas encontro correctamente el modelo.
- F1-score: promedio armonico entre Precision y Recall; resume el equilibrio entre ambas.
- Matriz de Confusion: tabla que muestra aciertos y errores por clase: negativos correctos, positivos correctos y confusiones entre clases.

## Resultados Obtenidos

Con el comando:

```powershell
python scripts\train.py --limit-train 50000 --limit-test 10000
```

Se obtuvo:

- Train: 50000 resenas.
- Test: 10000 resenas.
- Clases balanceadas: positivas y negativas.
- Accuracy aproximado: 0.87.
- F1-score aproximado: 0.87.

Estos resultados son adecuados para un modelo base clasico con TF-IDF + Naive Bayes.

## Limitaciones

- El dataset esta en ingles, por lo que el modelo funciona mejor con resenas en ingles.
- Puede fallar o perder confianza con textos en espanol.
- Naive Bayes es rapido y explicable, pero no captura contexto profundo como sarcasmo, negaciones complejas o significado semantico avanzado.
- El modelo depende de la calidad y representatividad del dataset usado.

## Mejoras Futuras

- Comparar con `LogisticRegression` o `LinearSVC`.
- Probar embeddings o modelos Transformer.
- Agregar soporte para resenas en espanol con un dataset adecuado o traduccion previa.
- Crear una interfaz web simple para predicciones.
- Guardar experimentos con diferentes tamanos de entrenamiento y comparar metricas.
