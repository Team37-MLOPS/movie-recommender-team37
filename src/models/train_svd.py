"""Train an SVD collaborative-filtering recommender, tuning hyperparameters
with Ray Tune and tracking every trial + the final best model in MLflow.

Each Ray Tune trial trains one SVDModel on the training set and evaluates
it on the validation set, logging its params/metrics as an MLflow run.
After tuning, the best config (selected on validation RMSE) is
retrained on the training set, evaluated on validation with the full
metric set (rating + ranking), and registered as a new version of the
"movie-recommender-svd" model in the MLflow Model Registry.

The held-out test set is not used in training. Will be used post deployment
Run with: python -m src.models.train_svd
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
from src.models.metrics import mae, precision_recall_at_k, rmse
from src.models.recommender_pyfunc import RecommenderPyfunc
from src.models.svd_model import SVDModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH}"
EXPERIMENT_NAME = "movie-recommender"
REGISTERED_MODEL_NAME = "movie-recommender-svd"
TOP_K = 10
NUM_TUNE_SAMPLES = 8


def evaluate_rating_predictions(model: SVDModel, eval_df) -> dict:
    y_true = eval_df["rating"].to_numpy()
    y_pred = [model.predict_rating(u, m) for u, m in zip(eval_df["user_id"], eval_df["movie_id"])]
    return {"rmse": rmse(y_true, y_pred), "mae": mae(y_true, y_pred)}


def train_trial(config, train_df, val_df):
    """One Ray Tune trial: fit an SVDModel with `config` on the training
    set, evaluate on the validation set, log it as its own MLflow run, and
    report RMSE back to the tuner. Only rating-prediction metrics are
    evaluated per trial (ranking metrics are computed once, later, for the
    winning config) to keep tuning fast."""
    trial_name = tune.get_context().get_trial_name()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=trial_name):
        mlflow.log_params(config)
        mlflow.set_tag("model_family", "svd")
        mlflow.set_tag("stage", "tuning_trial")

        model = SVDModel(**config, random_state=42).fit(train_df)
        metrics = evaluate_rating_predictions(model, val_df)
        mlflow.log_metrics(metrics)

        tune.report(metrics)


def log_pyfunc_model(model: SVDModel, artifact_path: str, registered_model_name: str = None) -> None:
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
            pip_requirements=["scikit-surprise==1.1.5", "pandas", "numpy", "scipy"],
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
        "n_factors": tune.choice([20, 50, 100]),
        "n_epochs": tune.choice([10, 20, 30]),
        "lr_all": tune.loguniform(1e-3, 1e-1),
        "reg_all": tune.uniform(0.01, 0.1),
    }

    trainable = tune.with_parameters(train_trial, train_df=train_df, val_df=val_df)
    tuner = tune.Tuner(
        trainable,
        param_space=search_space,
        tune_config=tune.TuneConfig(metric="rmse", mode="min", num_samples=NUM_TUNE_SAMPLES),
    )
    results = tuner.fit()

    best_result = results.get_best_result(metric="rmse", mode="min")
    best_config = best_result.config
    logger.info("Best config (selected on validation rmse): %s (rmse=%.4f)", best_config, best_result.metrics["rmse"])

    logger.info("Retraining best config on train set for final validation evaluation + registration")
    with mlflow.start_run(run_name="best_model"):
        mlflow.log_params(best_config)
        mlflow.set_tag("model_family", "svd")
        mlflow.set_tag("stage", "best_model")

        best_model = SVDModel(**best_config, random_state=42).fit(train_df)

        user_items_map = build_user_items_map(train_df)
        relevant_items_map = build_relevant_items_map(val_df)
        val_users = list(relevant_items_map.keys())

        metrics = evaluate_rating_predictions(best_model, val_df)
        recs = best_model.recommend_batch(val_users, TOP_K, user_items_map)
        metrics.update(precision_recall_at_k(recs, relevant_items_map, k=TOP_K))
        mlflow.log_metrics(metrics)
        logger.info("Best model validation metrics: %s", metrics)

        log_pyfunc_model(best_model, "model", registered_model_name=REGISTERED_MODEL_NAME)

    logger.info("Done. Run `mlflow ui --backend-store-uri %s` to inspect runs and the registry.", MLFLOW_TRACKING_URI)
    logger.info("Test set (%d rows) was not used - reserved for future streamed-inference evaluation.", len(test_df))


if __name__ == "__main__":
    main()

