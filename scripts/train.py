from __future__ import annotations
import argparse
from pathlib import Path
import joblib
from amazon_sentiment.data import load_dataset
from amazon_sentiment.model import build_pipeline
from amazon_sentiment.evaluate import evaluate_and_save


def main():
    parser = argparse.ArgumentParser(description="Entrena análisis de sentimientos con TF-IDF + Naive Bayes.")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--model-path", default="models/sentiment_pipeline.joblib")
    parser.add_argument("--limit-train", type=int, default=200000, help="Usa None/0 para todo el dataset. Recomendado empezar con 200000.")
    parser.add_argument("--limit-test", type=int, default=50000)
    args = parser.parse_args()
    limit_train = None if args.limit_train == 0 else args.limit_train
    limit_test = None if args.limit_test == 0 else args.limit_test

    train_df, test_df = load_dataset(args.raw_dir, limit_train, limit_test)
    print(f"Train: {train_df.shape} | Test: {test_df.shape}")
    print("Distribución train:\n", train_df["sentiment"].value_counts())

    model = build_pipeline()
    model.fit(train_df["review"], train_df["label"])

    metrics = evaluate_and_save(model, test_df["review"], test_df["label"])
    Path(args.model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model_path)
    print(f"Modelo guardado en {args.model_path}")
    print(metrics["classification_report"])

if __name__ == "__main__":
    main()
