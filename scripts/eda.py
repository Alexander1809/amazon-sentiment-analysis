from __future__ import annotations
import argparse
from amazon_sentiment.data import load_dataset


def main():
    parser = argparse.ArgumentParser(description="Exploración básica del dataset.")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--limit", type=int, default=10000)
    args = parser.parse_args()
    train_df, test_df = load_dataset(args.raw_dir, args.limit, args.limit)
    for name, df in [("train", train_df), ("test", test_df)]:
        print(f"\n{name}: {df.shape}")
        print(df["sentiment"].value_counts())
        print("Longitud promedio:", df["review"].str.len().mean())
        print("Ejemplos:")
        print(df.sample(min(3, len(df)), random_state=42)[["sentiment", "review"]].to_string(index=False))

if __name__ == "__main__":
    main()
