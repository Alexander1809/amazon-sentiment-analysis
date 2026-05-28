from __future__ import annotations

import argparse
from pathlib import Path

from amazon_sentiment.data import load_dataset


def summarize_split(name: str, df) -> str:
    class_counts = df["sentiment"].value_counts()
    avg_length = df["review"].str.len().mean()

    lines = [
        f"{name}: {df.shape[0]} filas, {df.shape[1]} columnas",
        "Distribucion de clases:",
        class_counts.to_string(),
        f"Longitud promedio de resenas: {avg_length:.2f} caracteres",
        "",
        "Ejemplos positivos:",
        df[df["sentiment"] == "positive"].head(3)[["sentiment", "review"]].to_string(index=False),
        "",
        "Ejemplos negativos:",
        df[df["sentiment"] == "negative"].head(3)[["sentiment", "review"]].to_string(index=False),
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Exploracion basica del dataset Amazon Reviews.")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument("--report-path", default="reports/eda_summary.txt")
    args = parser.parse_args()

    train_df, test_df = load_dataset(args.raw_dir, args.limit, args.limit)
    summary = "\n\n".join(
        [
            "Resumen EDA - Amazon Reviews",
            "============================",
            summarize_split("train", train_df),
            summarize_split("test", test_df),
        ]
    )

    print(summary)

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(summary, encoding="utf-8")
    print(f"\nResumen guardado en {report_path}")


if __name__ == "__main__":
    main()
