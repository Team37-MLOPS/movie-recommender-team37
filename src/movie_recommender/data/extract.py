from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

from movie_recommender.config import load_settings

EXPECTED_FILES = {"ratings.dat", "movies.dat", "users.dat", "README"}


def extract_dataset(zip_path: Path, raw_dir: Path) -> list[Path]:
    if not zip_path.exists():
        raise FileNotFoundError(f"Dataset zip not found: {zip_path}")

    raw_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(raw_dir.parent)

    extracted = [raw_dir / file_name for file_name in sorted(EXPECTED_FILES)]
    missing = [path for path in extracted if not path.exists()]
    if missing:
        missing_names = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing extracted dataset files: {missing_names}")
    return extracted


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract MovieLens 1M dataset.")
    parser.add_argument("--config", default="configs/settings.yaml")
    args = parser.parse_args()

    settings = load_settings(args.config)
    extracted = extract_dataset(settings.paths.dataset_zip, settings.paths.raw_dir)
    for path in extracted:
        print(path)


if __name__ == "__main__":
    main()
