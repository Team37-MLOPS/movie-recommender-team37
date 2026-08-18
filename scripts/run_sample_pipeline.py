from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from movie_recommender.config import load_settings
from movie_recommender.data.extract import extract_dataset
from movie_recommender.data.preprocess_spark import preprocess
from movie_recommender.models.train import train


def main() -> None:
    settings = load_settings()
    extract_dataset(settings.paths.dataset_zip, settings.paths.raw_dir)
    preprocess(settings, sample_fraction=0.05)
    train(settings, sample_fraction=0.05)


if __name__ == "__main__":
    main()
