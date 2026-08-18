"""Train an ALS (implicit-feedback) collaborative-filtering recommender,
tuning hyperparameters with Ray Tune and tracking every trial + the final
best model in MLflow.

Unlike SVD, ALS has no rating-scale output, so tuning and model selection
are driven by ranking quality (f1_at_10, the harmonic mean of precision
and recall) instead of RMSE. Each Ray Tune trial trains one ALSModel on
the training set and evaluates Precision@10/Recall@10/F1@10 on the
validation set, logging params/metrics as an MLflow run. After tuning,
the best config is retrained on the training set, re-evaluated on
validation, and registered as a new version of the "movie-recommender-als"
model in the MLflow Model Registry.

The held-out test set is not used in training. Will be used post deployment.

Run with: python -m src.models.train_als
"""
import logging
import pickle
import tempfile
from pathlib import Path

import mlflow
import mlflow.pyfunc
import ray
from ray import tune

from src.data.data_utils import (
    build_relevant_items_map,
    build_user_items_map,
    load_ratings,
    train_val_test_split_by_user,
)
from src.models.als_model import ALSModel
from src.models.metrics import precision_recall_at_k
from src.models.recommender_pyfunc import RecommenderPyfunc

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH}"
EXPERIMENT_NAME = "movie-recommender"
REGISTERED_MODEL_NAME = "movie-recommender-als"
TOP_K = 10
NUM_TUNE_SAMPLES = 8


def evaluate_ranking(model: ALSModel, users, user_items_map, relevant_items_map, k=TOP_K) -> dict:
    recs = model.recommend_batch(users, k, user_items_map)
    return precision_recall_at_k(recs, relevant_items_map, k=k)


def train_trial(config, train_df, val_df):
    """One Ray Tune trial: fit an ALSModel with `config` on the training
    set, evaluate ranking quality on the validation set, log it as its own
    MLflow run, and report f1_at_10 back to the tuner."""
    trial_name = tune.get_context().get_trial_name()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=trial_name):
        mlflow.log_params(config)
        mlflow.set_tag("model_family", "als")
        mlflow.set_tag("stage", "tuning_trial")

        model = ALSModel(**config, random_state=42).fit(train_df)

        user_items_map = build_user_items_map(train_df)
        relevant_items_map = build_relevant_items_map(val_df)
        val_users = list(relevant_items_map.keys())

        metrics = evaluate_ranking(model, val_users, user_items_map, relevant_items_map, k=TOP_K)
        mlflow.log_metrics(metrics)

        tune.report(metrics)


def log_pyfunc_model(model: ALSModel, artifact_path: str, registered_model_name: str = None) -> None:
    """Pickle the raw model, then log it wrapped as an MLflow pyfunc model
    so it's servable via mlflow.pyfunc.load_model and registerable in the
    Model Registry."""
    with tempfile.TemporaryDirectory() as tmp:
        pkl_path = Path(tmp) / "model.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(model, f)
        mlflow.pyfunc.log_model(
            artifact_path=artifact_path,
            python_model=RecommenderPyfunc(),
            artifacts={"model": str(pkl_path)},
            pip_requirements=["implicit==0.7.3", "pandas", "numpy", "scipy"],
            registered_model_name=registered_model_name,
        )


def main() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    logger.info("Loading MovieLens 1M ratings")
    ratings = load_ratings()
    train_df, val_df, test_df = train_val_test_split_by_user(ratings, val_frac=0.15, test_frac=0.15, seed=42)
    logger.info(
        "Train: %d rows, Val: %d rows, Test (reserved, unused): %d rows",
        len(train_df), len(val_df), len(test_df),
    )

    ray.init(ignore_reinit_error=True)
    logger.info("Ray cluster resources: %s", ray.cluster_resources())

    search_space = {
        "factors": tune.choice([20, 60, 100]),
        "regularization": tune.loguniform(1e-3, 1e-1),
        "iterations": tune.choice([10, 15, 20]),
    }

    trainable = tune.with_parameters(train_trial, train_df=train_df, val_df=val_df)
    tuner = tune.Tuner(
        trainable,
        param_space=search_space,
        tune_config=tune.TuneConfig(metric="f1_at_10", mode="max", num_samples=NUM_TUNE_SAMPLES),
    )
    results = tuner.fit()

    best_result = results.get_best_result(metric="f1_at_10", mode="max")
    best_config = best_result.config
    logger.info(
        "Best config (selected on validation f1_at_10): %s (f1_at_10=%.4f)",
        best_config, best_result.metrics["f1_at_10"],
    )

    logger.info("Retraining best config on train set for final validation evaluation + registration")
    with mlflow.start_run(run_name="best_model"):
        mlflow.log_params(best_config)
        mlflow.set_tag("model_family", "als")
        mlflow.set_tag("stage", "best_model")

        best_model = ALSModel(**best_config, random_state=42).fit(train_df)

        user_items_map = build_user_items_map(train_df)
        relevant_items_map = build_relevant_items_map(val_df)
        val_users = list(relevant_items_map.keys())

        metrics = evaluate_ranking(best_model, val_users, user_items_map, relevant_items_map, k=TOP_K)
        mlflow.log_metrics(metrics)
        logger.info("Best model validation metrics: %s", metrics)

        log_pyfunc_model(best_model, "model", registered_model_name=REGISTERED_MODEL_NAME)

    logger.info("Done. Run `mlflow ui --backend-store-uri %s` to inspect runs and the registry.", MLFLOW_TRACKING_URI)
    logger.info("Test set (%d rows) was not used - reserved for future streamed-inference evaluation.", len(test_df))


if __name__ == "__main__":
    main()
