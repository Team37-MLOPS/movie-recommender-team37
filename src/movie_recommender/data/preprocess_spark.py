from __future__ import annotations

import argparse
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

from movie_recommender.config import Settings, load_settings

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
    cleaned = (
        ratings.dropDuplicates(["user_id", "movie_id", "timestamp"])
        .where(F.col("user_id").isNotNull())
        .where(F.col("movie_id").isNotNull())
        .where(F.col("rating").between(1.0, 5.0))
        .where(F.col("timestamp").isNotNull())
        .withColumn("rating_datetime", F.from_unixtime("timestamp").cast("timestamp"))
        .withColumn("rating_year", F.year("rating_datetime"))
        .withColumn("rating_month", F.month("rating_datetime"))
    )
    if 0 < sample_fraction < 1:
        cleaned = cleaned.sample(withReplacement=False, fraction=sample_fraction, seed=seed)
    return cleaned


def clean_movies(movies: DataFrame) -> DataFrame:
    return (
        movies.dropDuplicates(["movie_id"])
        .where(F.col("movie_id").isNotNull())
        .where(F.col("title").isNotNull())
        .withColumn("release_year", F.regexp_extract("title", r"\((\d{4})\)$", 1).cast("int"))
        .withColumn("clean_title", F.trim(F.regexp_replace("title", r"\s*\(\d{4}\)$", "")))
        .withColumn("genre_array", F.split(F.col("genres"), r"\|"))
    )


def clean_users(users: DataFrame) -> DataFrame:
    return (
        users.dropDuplicates(["user_id"])
        .where(F.col("user_id").isNotNull())
        .withColumn("gender", F.coalesce(F.col("gender"), F.lit("unknown")))
        .withColumn("age", F.coalesce(F.col("age"), F.lit(0)))
        .withColumn("occupation", F.coalesce(F.col("occupation"), F.lit(0)))
        .withColumn("zip_code", F.coalesce(F.col("zip_code"), F.lit("unknown")))
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


def preprocess(settings: Settings, sample_fraction: float | None = None) -> None:
    spark = build_spark(settings)
    try:
        ratings_raw, movies_raw, users_raw = load_raw_frames(spark, settings.paths.raw_dir)
        fraction = settings.training.sample_fraction if sample_fraction is None else sample_fraction
        ratings = clean_ratings(ratings_raw, fraction, settings.random_seed)
        movies = clean_movies(movies_raw)
        users = clean_users(users_raw)
        frames = build_features(ratings, movies, users)
        write_outputs(frames, settings.paths.processed_dir)
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
