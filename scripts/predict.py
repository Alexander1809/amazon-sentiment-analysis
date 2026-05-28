from __future__ import annotations
import argparse
import joblib


def main():
    parser = argparse.ArgumentParser(description="Predice sentimiento de una reseña.")
    parser.add_argument("text", nargs="+", help="Texto de la reseña")
    parser.add_argument("--model-path", default="models/sentiment_pipeline.joblib")
    args = parser.parse_args()
    model = joblib.load(args.model_path)
    review = " ".join(args.text)
    pred = int(model.predict([review])[0])
    proba = model.predict_proba([review])[0]
    print({"review": review, "prediction": "positive" if pred == 1 else "negative", "confidence": float(max(proba))})

if __name__ == "__main__":
    main()
