from __future__ import annotations

import argparse
import bz2
import gzip
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

DATASET = "bittlingmayer/amazonreviews"
EXPECTED_FILES = ("train.ft.txt", "test.ft.txt")
MIN_EXPECTED_BYTES = {
    "train.ft.txt": 100_000_000,
    "test.ft.txt": 10_000_000,
}


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def configure_kaggle_from_token() -> None:
    token = os.getenv("KAGGLE_API_TOKEN")
    if not token or (os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY")):
        return

    if token.lstrip().startswith("{"):
        credentials = json.loads(token)
    else:
        credentials = json.loads(Path(token).read_text(encoding="utf-8"))

    os.environ["KAGGLE_USERNAME"] = credentials["username"]
    os.environ["KAGGLE_KEY"] = credentials["key"]


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


def expected_files_are_complete(raw_dir: Path) -> bool:
    return all((raw_dir / name).stat().st_size >= min_bytes for name, min_bytes in MIN_EXPECTED_BYTES.items())


def create_tiny_sample(raw_dir: Path) -> None:
    sample = """__label__2 Excelente producto, llego rapido y funciona perfecto.\n__label__1 Muy mala calidad, se dano en dos dias.\n__label__2 Me encanto, lo recomiendo totalmente.\n__label__1 No cumple lo prometido, dinero perdido.\n"""
    (raw_dir / "train.ft.txt").write_text(sample, encoding="utf-8")
    (raw_dir / "test.ft.txt").write_text(sample, encoding="utf-8")
    print("Se creo un dataset pequeno de prueba en data/raw/. Usalo solo para verificar el flujo.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Descarga el dataset Amazon Reviews desde Kaggle.")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--tiny-sample", action="store_true", help="Crea datos minimos sin usar Kaggle.")
    args = parser.parse_args()
    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    if args.tiny_sample:
        create_tiny_sample(raw_dir)
        return

    try:
        if expected_files_exist(raw_dir) and expected_files_are_complete(raw_dir):
            print("train.ft.txt y test.ft.txt ya existen y parecen completos. Se omite descarga.")
            return

        if list(raw_dir.glob("*.bz2")) or list(raw_dir.glob("*.gz")):
            maybe_decompress_archives(raw_dir)
            if expected_files_exist(raw_dir) and expected_files_are_complete(raw_dir):
                return

        configure_kaggle_from_token()
        run([sys.executable, "-m", "kaggle", "datasets", "download", "-d", DATASET, "-p", str(raw_dir), "--unzip"])
        maybe_decompress_archives(raw_dir)
    except Exception as exc:
        print("\nNo se pudo descargar desde Kaggle.")
        print("Configura KAGGLE_API_TOKEN, KAGGLE_USERNAME/KAGGLE_KEY o usa --tiny-sample para una prueba minima.")
        raise exc


if __name__ == "__main__":
    main()
