from __future__ import annotations

import argparse
from pathlib import Path

import joblib


def main() -> None:
    parser = argparse.ArgumentParser(description="Predice el sentimiento de una resena.")
    parser.add_argument("text", nargs="+", help="Texto de la resena")
    parser.add_argument("--model-path", default="models/sentiment_pipeline.joblib")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"No existe el modelo en {model_path}. Ejecuta primero scripts/train.py para entrenarlo."
        )

    model = joblib.load(model_path)
    review = " ".join(args.text)
    pred = int(model.predict([review])[0])
    probabilities = model.predict_proba([review])[0]
    confidence = float(max(probabilities))
    label = "positive" if pred == 1 else "negative"

    print("Advertencia: este modelo fue entrenado con resenas en ingles; funciona mejor con texto en ingles.")
    print(f"Resena: {review}")
    print(f"Prediccion: {label}")
    print(f"Confianza: {confidence:.4f}")


if __name__ == "__main__":
    main()
