from __future__ import annotations
import argparse, bz2, gzip, shutil, subprocess, sys
from pathlib import Path

DATASET = "bittlingmayer/amazonreviews"
EXPECTED_FILES = ("train.ft.txt", "test.ft.txt")

def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)

def should_decompress(archive_path: Path, out_path: Path) -> bool:
    if not out_path.exists():
        return True
    return out_path.stat().st_size < archive_path.stat().st_size


def decompress_archive(archive_path: Path, opener) -> Path:
    out_path = archive_path.with_suffix("")
    if not should_decompress(archive_path, out_path):
        print(f"{out_path.name} ya existe y parece completo. Se omite.")
        return out_path

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    print(f"Descomprimiendo {archive_path.name} -> {out_path.name}")
    with opener(archive_path, "rb") as src, open(tmp_path, "wb") as dst:
        shutil.copyfileobj(src, dst)
    tmp_path.replace(out_path)
    print(f"{out_path.name}: {out_path.stat().st_size:,} bytes")
    return out_path


def maybe_decompress_archives(raw_dir: Path) -> None:
    for bz2_path in raw_dir.glob("*.bz2"):
        decompress_archive(bz2_path, bz2.open)
    for gz_path in raw_dir.glob("*.gz"):
        decompress_archive(gz_path, gzip.open)


def expected_files_exist(raw_dir: Path) -> bool:
    return all((raw_dir / name).exists() for name in EXPECTED_FILES)

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
        if list(raw_dir.glob("*.bz2")) or list(raw_dir.glob("*.gz")):
            maybe_decompress_archives(raw_dir)
            if expected_files_exist(raw_dir):
                return

        run([sys.executable, "-m", "kaggle", "datasets", "download", "-d", DATASET, "-p", str(raw_dir), "--unzip"])
        maybe_decompress_archives(raw_dir)
    except Exception as exc:
        print("\nNo se pudo descargar desde Kaggle.")
        print("Configura tus credenciales en ~/.kaggle/kaggle.json o usa: python -m amazon_sentiment.download_data --tiny-sample")
        raise exc

if __name__ == "__main__":
    main()
