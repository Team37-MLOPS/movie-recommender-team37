from __future__ import annotations

import os
from datetime import datetime

try:
    from airflow import DAG
    from airflow.operators.bash import BashOperator
except Exception:  # pragma: no cover - lets lightweight CI import without Airflow installed
    DAG = None
    BashOperator = None


PROJECT_DIR = os.getenv("PROJECT_DIR", "/opt/movie-recommender")
SCHEDULE = os.getenv("MOVIE_RECOMMENDER_DAG_SCHEDULE")


if DAG and BashOperator:
    with DAG(
        dag_id="movie_recommender_pipeline",
        description="Extract, preprocess, train, and export MovieLens recommendations.",
        start_date=datetime(2026, 1, 1),
        schedule=SCHEDULE,
        catchup=False,
        tags=["mlops", "movies", "recommendations"],
    ) as dag:
        extract = BashOperator(
            task_id="extract_dataset",
            bash_command=f"cd {PROJECT_DIR} && python -m movie_recommender.data.extract",
        )

        preprocess = BashOperator(
            task_id="spark_preprocess",
            bash_command=f"cd {PROJECT_DIR} && python -m movie_recommender.data.preprocess_spark",
        )

        train_svd = BashOperator(
            task_id="train_svd",
            bash_command=f"cd {PROJECT_DIR} && python -m src.models.train_svd",
        )

        train_als = BashOperator(
            task_id="train_als",
            bash_command=f"cd {PROJECT_DIR} && python -m src.models.train_als",
        )

        select_best = BashOperator(
            task_id="select_best_model",
            bash_command=f"cd {PROJECT_DIR} && python -m src.models.select_best",
        )

        extract >> preprocess >> [train_svd, train_als] >> select_best
