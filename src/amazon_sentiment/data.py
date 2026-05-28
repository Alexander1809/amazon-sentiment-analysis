from __future__ import annotations
from pathlib import Path
import pandas as pd


def parse_fasttext_file(path: str | Path, limit: int | None = None) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            label, _, text = line.partition(" ")
            if label == "__label__1":
                y = 0
            elif label == "__label__2":
                y = 1
            else:
                continue
            rows.append({"label": y, "sentiment": "positive" if y == 1 else "negative", "review": text})
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No se pudieron leer filas válidas desde {path}")
    return df.dropna(subset=["review", "label"]).drop_duplicates()


def load_dataset(raw_dir: str | Path = "data/raw", limit_train: int | None = None, limit_test: int | None = None):
    raw_dir = Path(raw_dir)
    train_path = raw_dir / "train.ft.txt"
    test_path = raw_dir / "test.ft.txt"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Faltan train.ft.txt o test.ft.txt. Ejecuta primero python -m amazon_sentiment.download_data")
    return parse_fasttext_file(train_path, limit_train), parse_fasttext_file(test_path, limit_test)
