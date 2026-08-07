"""
Airflow DAG orchestrating the MovieLens-1M pipeline:

    check_raw_data -> run_beam_preprocessing -> validate_processed_data -> train_baseline_models

Airflow is the compulsory orchestrator for this project; Apache Beam is the
data engineering tool used for the actual preprocessing (see
``ml_pipeline/data_pipeline/beam_preprocess.py`` for why Beam was chosen).
This DAG simply schedules/monitors that Beam job and the downstream training
step as a reproducible, automated workflow.
"""

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

ML_PIPELINE_DIR = os.environ.get(
    "ML_PIPELINE_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
)
RAW_DIR = os.path.join(ML_PIPELINE_DIR, "data", "raw", "ml-1m")
PROCESSED_DIR = os.path.join(ML_PIPELINE_DIR, "data", "processed")
PROCESSED_OUTPUT = os.path.join(PROCESSED_DIR, "movielens_features")
PROCESSED_CSV = PROCESSED_OUTPUT + ".csv"

default_args = {
    "owner": "team37-mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


def check_raw_data(**_):
    required = ["ratings.dat", "movies.dat", "users.dat"]
    missing = [f for f in required if not os.path.exists(os.path.join(RAW_DIR, f))]
    if missing:
        raise FileNotFoundError(f"Missing raw MovieLens files: {missing} in {RAW_DIR}")


def run_beam_preprocessing(**_):
    from ml_pipeline.data_pipeline.beam_preprocess import run as run_beam

    if os.path.exists(PROCESSED_CSV):
        os.remove(PROCESSED_CSV)

    run_beam([
        "--ratings", os.path.join(RAW_DIR, "ratings.dat"),
        "--movies", os.path.join(RAW_DIR, "movies.dat"),
        "--users", os.path.join(RAW_DIR, "users.dat"),
        "--output", PROCESSED_OUTPUT,
    ])


def validate_processed_data(**_):
    import pandas as pd

    df = pd.read_csv(PROCESSED_CSV)
    if df.empty:
        raise ValueError("Processed dataset is empty.")
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        raise ValueError(f"Processed dataset contains nulls:\n{null_counts[null_counts > 0]}")
    if not df["rating"].between(1, 5).all():
        raise ValueError("Found ratings outside the valid 1-5 range.")
    print(f"Validated processed dataset: {df.shape[0]} rows, {df.shape[1]} columns.")


def train_baseline_models(**_):
    from ml_pipeline.training.train_models import main as train_main

    train_main(data_path=PROCESSED_CSV)


with DAG(
    dag_id="movielens_data_and_training_pipeline",
    default_args=default_args,
    description="MovieLens-1M: Beam preprocessing + baseline model training",
    schedule_interval=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["movielens", "beam", "mlflow"],
) as dag:

    t1_check_raw = PythonOperator(
        task_id="check_raw_data",
        python_callable=check_raw_data,
    )

    t2_preprocess = PythonOperator(
        task_id="run_beam_preprocessing",
        python_callable=run_beam_preprocessing,
    )

    t3_validate = PythonOperator(
        task_id="validate_processed_data",
        python_callable=validate_processed_data,
    )

    t4_train = PythonOperator(
        task_id="train_baseline_models",
        python_callable=train_baseline_models,
    )

    t1_check_raw >> t2_preprocess >> t3_validate >> t4_train
