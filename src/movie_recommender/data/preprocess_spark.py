from __future__ import annotations

import argparse
import json
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import Window
from pyspark.sql import functions as F
from pyspark.sql import types as T

from movie_recommender.config import Settings, load_settings

MIN_VALID_AGE = 1
MAX_VALID_AGE = 120
VALID_GENDERS = ("M", "F")
NO_GENRES_LISTED = "(no genres listed)"

RATINGS_SCHEMA = T.StructType(
    [
        T.StructField("user_id", T.IntegerType(), nullable=False),
        T.StructField("movie_id", T.IntegerType(), nullable=False),
        T.StructField("rating", T.DoubleType(), nullable=False),
        T.StructField("timestamp", T.LongType(), nullable=False),
    ]
)

MOVIES_SCHEMA = T.StructType(
    [
        T.StructField("movie_id", T.IntegerType(), nullable=False),
        T.StructField("title", T.StringType(), nullable=False),
        T.StructField("genres", T.StringType(), nullable=False),
    ]
)

USERS_SCHEMA = T.StructType(
    [
        T.StructField("user_id", T.IntegerType(), nullable=False),
        T.StructField("gender", T.StringType(), nullable=True),
        T.StructField("age", T.IntegerType(), nullable=True),
        T.StructField("occupation", T.IntegerType(), nullable=True),
        T.StructField("zip_code", T.StringType(), nullable=True),
    ]
)


def build_spark(settings: Settings) -> SparkSession:
    return (
        SparkSession.builder.appName(settings.spark.app_name)
        .master(settings.spark.master)
        .config("spark.sql.shuffle.partitions", settings.spark.shuffle_partitions)
        .getOrCreate()
    )


def read_dat(spark: SparkSession, path: Path, schema: T.StructType) -> DataFrame:
    return (
        spark.read.option("sep", "::")
        .option("encoding", "ISO-8859-1")
        .option("mode", "DROPMALFORMED")
        .schema(schema)
        .csv(str(path))
    )


def load_raw_frames(spark: SparkSession, raw_dir: Path) -> tuple[DataFrame, DataFrame, DataFrame]:
    ratings = read_dat(spark, raw_dir / "ratings.dat", RATINGS_SCHEMA)
    movies = read_dat(spark, raw_dir / "movies.dat", MOVIES_SCHEMA)
    users = read_dat(spark, raw_dir / "users.dat", USERS_SCHEMA)
    return ratings, movies, users


def clean_ratings(ratings: DataFrame, sample_fraction: float, seed: int) -> DataFrame:
    deduped_by_event = ratings.dropDuplicates(["user_id", "movie_id", "timestamp"])

    # A user can re-rate the same movie at a different time; keep only the
    # most recent rating per (user, movie) so aggregates don't double-count
    # that pair.
    latest_per_pair = Window.partitionBy("user_id", "movie_id").orderBy(F.col("timestamp").desc())
    deduped = (
        deduped_by_event.withColumn("_rank", F.row_number().over(latest_per_pair))
        .where(F.col("_rank") == 1)
        .drop("_rank")
    )

    cleaned = (
        deduped.where(F.col("user_id").isNotNull())
        .where(F.col("movie_id").isNotNull())
        .where(F.col("rating").between(1.0, 5.0))
        .where(F.col("timestamp").isNotNull())
        .where(F.col("timestamp") > 0)
        .withColumn("rating_datetime", F.from_unixtime("timestamp").cast("timestamp"))
        .withColumn("rating_year", F.year("rating_datetime"))
        .withColumn("rating_month", F.month("rating_datetime"))
    )
    if 0 < sample_fraction < 1:
        cleaned = cleaned.sample(withReplacement=False, fraction=sample_fraction, seed=seed)
    return cleaned


def clean_movies(movies: DataFrame) -> DataFrame:
    cleaned = (
        movies.dropDuplicates(["movie_id"])
        .where(F.col("movie_id").isNotNull())
        .where(F.col("title").isNotNull())
        .withColumn("release_year", F.regexp_extract("title", r"\((\d{4})\)$", 1).cast("int"))
        .withColumn("release_year_missing", F.col("release_year").isNull())
        .withColumn("clean_title", F.trim(F.regexp_replace("title", r"\s*\(\d{4}\)$", "")))
        .withColumn(
            "genre_array",
            F.when(F.col("genres") == NO_GENRES_LISTED, F.array())
            .otherwise(F.split(F.col("genres"), r"\|")),
        )
    )
    return cleaned


def clean_users(users: DataFrame) -> DataFrame:
    return (
        users.dropDuplicates(["user_id"])
        .where(F.col("user_id").isNotNull())
        .withColumn(
            "gender",
            F.when(F.col("gender").isin(*VALID_GENDERS), F.col("gender")).otherwise(F.lit("unknown")),
        )
        .withColumn(
            "age",
            F.when(
                F.col("age").isNotNull() & F.col("age").between(MIN_VALID_AGE, MAX_VALID_AGE),
                F.col("age"),
            ).otherwise(F.lit(0)),
        )
        .withColumn("occupation", F.coalesce(F.col("occupation"), F.lit(0)))
        .withColumn(
            "zip_code",
            F.when(
                F.col("zip_code").rlike(r"^\d{5}"),
                F.substring(F.col("zip_code"), 1, 5),
            ).otherwise(F.lit("unknown")),
        )
    )


def build_features(ratings: DataFrame, movies: DataFrame, users: DataFrame) -> dict[str, DataFrame]:
    user_features = ratings.groupBy("user_id").agg(
        F.count("*").alias("user_rating_count"),
        F.avg("rating").alias("user_mean_rating"),
        F.stddev("rating").alias("user_rating_stddev"),
    )

    movie_features = ratings.groupBy("movie_id").agg(
        F.count("*").alias("movie_rating_count"),
        F.avg("rating").alias("movie_mean_rating"),
        F.stddev("rating").alias("movie_rating_stddev"),
    )

    interactions = (
        ratings.join(users, on="user_id", how="left")
        .join(movies, on="movie_id", how="left")
        .join(user_features, on="user_id", how="left")
        .join(movie_features, on="movie_id", how="left")
        .withColumn("is_positive", (F.col("rating") >= 4.0).cast("int"))
    )

    genre_counts = movies.select(F.explode("genre_array").alias("genre")).groupBy("genre").count()
    rating_distribution = ratings.groupBy("rating").count().orderBy("rating")

    return {
        "ratings": ratings,
        "movies": movies,
        "users": users,
        "interactions": interactions,
        "user_features": user_features,
        "movie_features": movie_features,
        "genre_counts": genre_counts,
        "rating_distribution": rating_distribution,
    }


def write_outputs(frames: dict[str, DataFrame], processed_dir: Path) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        frame.write.mode("overwrite").parquet(str(processed_dir / name))


def compute_data_quality_report(raw_counts: dict[str, int], cleaned_counts: dict[str, int]) -> dict:
    report = {}
    for name, raw_count in raw_counts.items():
        cleaned_count = cleaned_counts.get(name, 0)
        dropped = raw_count - cleaned_count
        report[name] = {
            "raw_rows": raw_count,
            "cleaned_rows": cleaned_count,
            "dropped_rows": dropped,
            "dropped_ratio": (dropped / raw_count) if raw_count else 0.0,
        }
    return report


def write_data_quality_report(report: dict, processed_dir: Path) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    report_path = processed_dir / "data_quality_report.json"
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)


def preprocess(settings: Settings, sample_fraction: float | None = None) -> None:
    spark = build_spark(settings)
    try:
        ratings_raw, movies_raw, users_raw = load_raw_frames(spark, settings.paths.raw_dir)
        raw_counts = {
            "ratings": ratings_raw.count(),
            "movies": movies_raw.count(),
            "users": users_raw.count(),
        }

        fraction = settings.training.sample_fraction if sample_fraction is None else sample_fraction
        ratings = clean_ratings(ratings_raw, fraction, settings.random_seed)
        movies = clean_movies(movies_raw)
        users = clean_users(users_raw)
        cleaned_counts = {
            "ratings": ratings.count(),
            "movies": movies.count(),
            "users": users.count(),
        }

        frames = build_features(ratings, movies, users)
        write_outputs(frames, settings.paths.processed_dir)

        report = compute_data_quality_report(raw_counts, cleaned_counts)
        write_data_quality_report(report, settings.paths.processed_dir)
    finally:
        spark.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess MovieLens data with Spark.")
    parser.add_argument("--config", default="configs/settings.yaml")
    parser.add_argument("--sample-fraction", type=float, default=None)
    args = parser.parse_args()

    preprocess(load_settings(args.config), sample_fraction=args.sample_fraction)


if __name__ == "__main__":
    main()
