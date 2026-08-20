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
SCHEDULE = os.getenv("MOVIE_RECOMMENDER_DAG_SCHEDULE", "@weekly")


if DAG and BashOperator:
    with DAG(
        dag_id="movie_recommender_pipeline",
        description="Extract, preprocess, validate, and train MovieLens recommendation models.",
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

        validate = BashOperator(
            task_id="validate_processed_data",
            bash_command=f"cd {PROJECT_DIR} && python scripts/validate_processed_data.py",
        )

        train = BashOperator(
            task_id="train_and_log_models",
            bash_command=f"cd {PROJECT_DIR} && python -m movie_recommender.models.train",
        )

        train_svd = BashOperator(
            task_id="train_svd",
            bash_command=f"cd {PROJECT_DIR} && python -m src.models.train_svd",
        )

        train_als = BashOperator(
            task_id="train_als",
            bash_command=f"cd {PROJECT_DIR} && python -m src.models.train_als",
        )

        train_xgb_hybrid = BashOperator(
            task_id="train_xgb_hybrid",
            bash_command=f"cd {PROJECT_DIR} && python -m src.models.train_xgb_hybrid",
        )

        train_ncf = BashOperator(
            task_id="train_ncf",
            bash_command=f"cd {PROJECT_DIR} && python -m src.models.train_ncf",
        )

        select_best_model = BashOperator(
            task_id="select_best_model",
            bash_command=f"cd {PROJECT_DIR} && python -m src.models.select_best",
        )

        extract >> preprocess >> validate
        validate >> train
        validate >> [train_svd, train_als, train_xgb_hybrid, train_ncf] >> select_best_model
