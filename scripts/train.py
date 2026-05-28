from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from amazon_sentiment.data import load_dataset
from amazon_sentiment.evaluate import evaluate_and_save
from amazon_sentiment.model import build_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena analisis de sentimientos con TF-IDF + Naive Bayes.")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--model-path", default="models/sentiment_pipeline.joblib")
    parser.add_argument("--report-dir", default="reports")
    parser.add_argument(
        "--limit-train",
        type=int,
        default=200000,
        help="Usa 0 para entrenar con todo el dataset. Recomendado empezar con 50000 o 200000.",
    )
    parser.add_argument("--limit-test", type=int, default=50000, help="Usa 0 para evaluar con todo el dataset.")
    args = parser.parse_args()

    limit_train = None if args.limit_train == 0 else args.limit_train
    limit_test = None if args.limit_test == 0 else args.limit_test

    train_df, test_df = load_dataset(args.raw_dir, limit_train, limit_test)
    print(f"Train: {train_df.shape[0]} filas | Test: {test_df.shape[0]} filas")
    print("Distribucion train:")
    print(train_df["sentiment"].value_counts())

    model = build_pipeline()
    model.fit(train_df["review"], train_df["label"])

    metrics = evaluate_and_save(model, test_df["review"], test_df["label"], report_dir=args.report_dir)

    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    print(f"\nModelo guardado en {model_path}")
    print(f"Metricas guardadas en {Path(args.report_dir) / 'metrics.txt'}")
    print(f"Matriz de confusion guardada en {Path(args.report_dir) / 'confusion_matrix.png'}")
    print("\nMetricas principales:")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-score:  {metrics['f1_score']:.4f}")
    print("\nReporte de clasificacion:")
    print(metrics["classification_report"])


if __name__ == "__main__":
    main()
