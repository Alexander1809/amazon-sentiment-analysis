from __future__ import annotations
import argparse, gzip, shutil, subprocess, sys
from pathlib import Path

DATASET = "bittlingmayer/amazonreviews"

def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)

def maybe_decompress_gz(raw_dir: Path) -> None:
    for gz_path in raw_dir.glob("*.gz"):
        out_path = raw_dir / gz_path.stem
        if out_path.exists():
            continue
        print(f"Descomprimiendo {gz_path.name} -> {out_path.name}")
        with gzip.open(gz_path, "rb") as src, open(out_path, "wb") as dst:
            shutil.copyfileobj(src, dst)

def create_tiny_sample(raw_dir: Path) -> None:
    sample = """__label__2 Excelente producto, llegó rápido y funciona perfecto.\n__label__1 Muy mala calidad, se dañó en dos días.\n__label__2 Me encantó, lo recomiendo totalmente.\n__label__1 No cumple lo prometido, dinero perdido.\n"""
    (raw_dir / "train.ft.txt").write_text(sample, encoding="utf-8")
    (raw_dir / "test.ft.txt").write_text(sample, encoding="utf-8")
    print("Se creó un dataset pequeño de prueba en data/raw/. Úsalo solo para verificar el flujo.")

def main() -> None:
    parser = argparse.ArgumentParser(description="Descarga el dataset Amazon Reviews desde Kaggle.")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--tiny-sample", action="store_true", help="Crea datos mínimos sin usar Kaggle.")
    args = parser.parse_args()
    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    if args.tiny_sample:
        create_tiny_sample(raw_dir)
        return

    try:
        run([sys.executable, "-m", "kaggle", "datasets", "download", "-d", DATASET, "-p", str(raw_dir), "--unzip"])
        maybe_decompress_gz(raw_dir)
    except Exception as exc:
        print("\nNo se pudo descargar desde Kaggle.")
        print("Configura tus credenciales en ~/.kaggle/kaggle.json o usa: python -m amazon_sentiment.download_data --tiny-sample")
        raise exc

if __name__ == "__main__":
    main()
