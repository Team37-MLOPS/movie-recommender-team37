from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from movie_recommender.config import Settings, load_settings
from movie_recommender.data.preprocess_spark import build_spark


def read_processed(spark: SparkSession, processed_dir: Path) -> tuple[DataFrame, DataFrame]:
    ratings = spark.read.parquet(str(processed_dir / "ratings"))
    movies = spark.read.parquet(str(processed_dir / "movies"))
    return ratings, movies


def build_popularity(ratings: DataFrame, movies: DataFrame) -> DataFrame:
    popularity = (
        ratings.groupBy("movie_id")
        .agg(
            F.count("*").alias("rating_count"),
            F.avg("rating").alias("mean_rating"),
        )
        .withColumn("score", F.col("mean_rating") * F.log1p(F.col("rating_count")))
        .join(
            movies.select("movie_id", "clean_title", "title", "genres", "release_year"),
            on="movie_id",
            how="left",
        )
        .orderBy(F.desc("score"), F.desc("rating_count"), F.desc("mean_rating"))
    )
    return popularity


def evaluate_precision_at_k(recommendations: DataFrame, test: DataFrame, k: int) -> float:
    positives = (
        test.where(F.col("rating") >= 4.0)
        .groupBy("user_id")
        .agg(F.collect_set("movie_id").alias("actual_movies"))
    )
    predicted = recommendations.where(F.col("rank") <= k).groupBy("user_id").agg(
        F.collect_list("movie_id").alias("predicted_movies")
    )
    scored = predicted.join(positives, on="user_id", how="inner").withColumn(
        "hits", F.size(F.array_intersect("predicted_movies", "actual_movies"))
    )
    result = scored.select((F.col("hits") / F.lit(k)).alias("precision_at_k")).agg(
        F.avg("precision_at_k").alias("precision_at_k")
    )
    row = result.first()
    return float(row["precision_at_k"] or 0.0) if row else 0.0


def flatten_als_recommendations(recommendations: DataFrame, movies: DataFrame) -> DataFrame:
    return (
        recommendations.select(
            "user_id",
            F.posexplode("recommendations").alias("rank_zero", "recommendation"),
        )
        .select(
            "user_id",
            (F.col("rank_zero") + F.lit(1)).alias("rank"),
            F.col("recommendation.movie_id").alias("movie_id"),
            F.col("recommendation.rating").alias("score"),
        )
        .join(
            movies.select("movie_id", "clean_title", "title", "genres", "release_year"),
            on="movie_id",
            how="left",
        )
        .orderBy("user_id", "rank")
    )


def log_to_mlflow(settings: Settings, params: dict[str, Any], metrics: dict[str, float]) -> None:
    try:
        import mlflow

        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", settings.mlflow.tracking_uri)
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(settings.mlflow.experiment_name)
        with mlflow.start_run(run_name="popularity-vs-als"):
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            if settings.paths.metrics_path.exists():
                mlflow.log_artifact(str(settings.paths.metrics_path))
    except Exception as exc:  # pragma: no cover - depends on external MLflow service
        print(f"MLflow logging skipped: {exc}")


def train(settings: Settings, sample_fraction: float | None = None) -> dict[str, float]:
    spark = build_spark(settings)
    try:
        ratings, movies = read_processed(spark, settings.paths.processed_dir)
        fraction = settings.training.sample_fraction if sample_fraction is None else sample_fraction
        if 0 < fraction < 1:
            ratings = ratings.sample(
                withReplacement=False,
                fraction=fraction,
                seed=settings.random_seed,
            )

        train_df, test_df = ratings.randomSplit(
            [settings.training.train_ratio, 1 - settings.training.train_ratio],
            seed=settings.random_seed,
        )

        popularity = build_popularity(train_df, movies)
        settings.paths.model_dir.mkdir(parents=True, exist_ok=True)
        settings.paths.artifact_dir.mkdir(parents=True, exist_ok=True)
        popularity.write.mode("overwrite").parquet(str(settings.paths.popularity_path))

        als = ALS(
            userCol="user_id",
            itemCol="movie_id",
            ratingCol="rating",
            rank=settings.training.rank,
            maxIter=settings.training.max_iter,
            regParam=settings.training.reg_param,
            implicitPrefs=settings.training.implicit_prefs,
            coldStartStrategy=settings.training.cold_start_strategy,
            seed=settings.random_seed,
        )
        model = als.fit(train_df)
        predictions = model.transform(test_df)

        evaluator = RegressionEvaluator(
            metricName="rmse",
            labelCol="rating",
            predictionCol="prediction",
        )
        rmse = float(evaluator.evaluate(predictions))
        mae = float(
            predictions.select(F.abs(F.col("rating") - F.col("prediction")).alias("ae"))
            .agg(F.avg("ae").alias("mae"))
            .first()["mae"]
        )

        als_recommendations = flatten_als_recommendations(
            model.recommendForAllUsers(settings.training.top_k),
            movies,
        )
        precision_at_k = evaluate_precision_at_k(
            als_recommendations,
            test_df,
            settings.training.top_k,
        )
        als_recommendations.write.mode("overwrite").parquet(str(settings.paths.recommendation_path))

        model_path = settings.paths.model_dir / "als_model"
        model.write().overwrite().save(str(model_path))

        metrics = {
            "als_rmse": rmse,
            "als_mae": mae,
            f"als_precision_at_{settings.training.top_k}": precision_at_k,
        }
        with settings.paths.metrics_path.open("w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2, sort_keys=True)

        params = {
            "rank": settings.training.rank,
            "max_iter": settings.training.max_iter,
            "reg_param": settings.training.reg_param,
            "top_k": settings.training.top_k,
            "sample_fraction": fraction,
        }
        log_to_mlflow(settings, params, metrics)
        return metrics
    finally:
        spark.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MovieLens recommendation models.")
    parser.add_argument("--config", default="configs/settings.yaml")
    parser.add_argument("--sample-fraction", type=float, default=None)
    args = parser.parse_args()

    metrics = train(load_settings(args.config), sample_fraction=args.sample_fraction)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
