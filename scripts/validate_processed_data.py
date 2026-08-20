from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from movie_recommender.config import load_settings

MAX_DROPPED_RATIO = 0.2


def validate(processed_dir: Path) -> None:
    ratings_path = processed_dir / "ratings"
    if not ratings_path.exists():
        raise FileNotFoundError(f"Missing processed ratings at {ratings_path}")

    ratings = pd.read_parquet(ratings_path)
    if ratings.empty:
        raise ValueError("Processed ratings dataset is empty.")

    for column in ("user_id", "movie_id", "rating"):
        null_count = ratings[column].isnull().sum()
        if null_count > 0:
            raise ValueError(f"Processed ratings has {null_count} nulls in required column '{column}'.")

    if not ratings["rating"].between(1.0, 5.0).all():
        raise ValueError("Found ratings outside the valid 1.0-5.0 range.")

    report_path = processed_dir / "data_quality_report.json"
    if not report_path.exists():
        raise FileNotFoundError(f"Missing data quality report at {report_path}")

    with report_path.open("r", encoding="utf-8") as fh:
        report = json.load(fh)

    for name, stats in report.items():
        dropped_ratio = stats.get("dropped_ratio", 0.0)
        if dropped_ratio > MAX_DROPPED_RATIO:
            raise ValueError(
                f"'{name}' dropped {dropped_ratio:.1%} of raw rows during cleaning, "
                f"exceeding the {MAX_DROPPED_RATIO:.0%} threshold."
            )

    print(f"Validated processed dataset: {ratings.shape[0]} rows, {ratings.shape[1]} columns.")
    print(f"Data quality report: {json.dumps(report, indent=2, sort_keys=True)}")


def main() -> None:
    settings = load_settings()
    validate(settings.paths.processed_dir)


if __name__ == "__main__":
    main()
