"""Train a Neural Collaborative Filtering (PyTorch) recommender, tuning
hyperparameters with Ray Tune and tracking every trial + the final best
model in MLflow.

All of one invocation's Ray Tune trial runs and its final best-model run
are nested under a single timestamped parent run, so the MLflow UI run
list shows one collapsible row per training run instead of dumping every
trial flat in the list - expand it to drill into individual trials.

Each Ray Tune trial trains one NCFModel on the training set and evaluates
it on the validation set (RMSE), logging its params/metrics as an MLflow
run. After tuning, the top NUM_RERANK_CANDIDATES trials by validation RMSE
are retrained and re-evaluated on validation F1@10 (the metric
select_best.py actually compares model families on), and the one with the
best F1@10 - not necessarily the lowest RMSE - is registered as a new
version of the "movie-recommender-ncf" model in the MLflow Model Registry.

The held-out test set is not used in training. Will be used post deployment.

Run with: python -m movie_recommender.models.train_ncf
"""
import logging
import os
import pickle
import tempfile
from datetime import datetime
from pathlib import Path

import mlflow
import mlflow.pyfunc
import ray
from ray import tune

from movie_recommender.data.data_utils import (
    build_relevant_items_map,
    build_user_items_map,
    load_ratings,
    train_val_test_split_by_user,
)
from movie_recommender.models.metrics import mae, precision_recall_at_k, rmse
from movie_recommender.models.ncf_model import NCFModel
from movie_recommender.models.recommender_pyfunc import RecommenderPyfunc

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{MLFLOW_DB_PATH}")
EXPERIMENT_NAME = "movie-recommender"
REGISTERED_MODEL_NAME = "movie-recommender-ncf"
TOP_K = 10
NUM_TUNE_SAMPLES = 8
NUM_RERANK_CANDIDATES = 3
FIXED_EPOCHS = 15
FIXED_BATCH_SIZE = 4096


def evaluate_rating_predictions(model: NCFModel, eval_df) -> dict:
    y_true = eval_df["rating"].to_numpy()
    y_pred = [model.predict_rating(u, m) for u, m in zip(eval_df["user_id"], eval_df["movie_id"])]
    return {"rmse": rmse(y_true, y_pred), "mae": mae(y_true, y_pred)}


def train_trial(config, train_df, val_df, parent_run_id):
    """One Ray Tune trial: fit an NCFModel with `config` on the training
    set, evaluate on the validation set, log it as its own MLflow run
    nested under `parent_run_id`, and report RMSE back to the tuner. Only
    rating-prediction metrics are evaluated per trial (ranking metrics are
    computed once, later, for the winning config) to keep tuning fast.

    Ray runs each trial in its own worker process, so the usual
    `nested=True` context-manager trick (which relies on an in-process
    "active run" stack) doesn't reach across processes. Setting the
    `mlflow.parentRunId` tag explicitly achieves the same nesting in the
    UI regardless of which process created the run."""
    trial_name = tune.get_context().get_trial_name()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=trial_name, tags={"mlflow.parentRunId": parent_run_id}):
        mlflow.log_params(config)
        mlflow.set_tag("model_family", "ncf")
        mlflow.set_tag("stage", "tuning_trial")

        model = NCFModel(epochs=FIXED_EPOCHS, batch_size=FIXED_BATCH_SIZE, random_state=42, **config).fit(train_df)
        for step, epoch_loss in enumerate(model.epoch_losses_):
            mlflow.log_metric("train_loss", epoch_loss, step=step)

        metrics = evaluate_rating_predictions(model, val_df)
        mlflow.log_metrics(metrics)

        tune.report(metrics)


def log_pyfunc_model(model: NCFModel, artifact_path: str, registered_model_name: str = None) -> None:
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
            pip_requirements=["torch==2.13.0", "pandas", "numpy", "scipy"],
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

    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    with mlflow.start_run(run_name=f"ncf_tuning_{run_timestamp}") as parent_run:
        parent_run_id = parent_run.info.run_id
        mlflow.set_tag("model_family", "ncf")
        mlflow.set_tag("stage", "tuning_parent")

        ray.init(ignore_reinit_error=True)
        logger.info("Ray cluster resources: %s", ray.cluster_resources())

        search_space = {
            "n_factors": tune.choice([32, 64, 128]),
            "dropout": tune.uniform(0.1, 0.4),
            "lr": tune.loguniform(1e-4, 1e-2),
            "weight_decay": tune.loguniform(1e-6, 1e-3),
        }

        trainable = tune.with_parameters(train_trial, train_df=train_df, val_df=val_df, parent_run_id=parent_run_id)
        tuner = tune.Tuner(
            trainable,
            param_space=search_space,
            tune_config=tune.TuneConfig(metric="rmse", mode="min", num_samples=NUM_TUNE_SAMPLES),
        )
        results = tuner.fit()

        # RMSE and F1@10 don't always agree on the best config, and F1@10
        # is what select_best.py actually compares model families on.
        # Re-rank the top RMSE candidates by validation F1@10 and keep the
        # one that ranks best, rather than blindly trusting the RMSE
        # optimum.
        top_results = sorted(results, key=lambda r: r.metrics["rmse"])[:NUM_RERANK_CANDIDATES]
        user_items_map = build_user_items_map(train_df)
        relevant_items_map = build_relevant_items_map(val_df)
        val_users = list(relevant_items_map.keys())

        candidates = []
        for result in top_results:
            config = result.config
            model = NCFModel(epochs=FIXED_EPOCHS, batch_size=FIXED_BATCH_SIZE, random_state=42, **config).fit(train_df)
            metrics = evaluate_rating_predictions(model, val_df)
            recs = model.recommend_batch(val_users, TOP_K, user_items_map)
            metrics.update(precision_recall_at_k(recs, relevant_items_map, k=TOP_K))
            logger.info("Re-rank candidate config=%s metrics=%s", config, metrics)
            candidates.append((config, model, metrics))

        best_config, best_model, metrics = max(candidates, key=lambda c: c[2]["f1_at_10"])
        logger.info(
            "Selected config (best validation f1_at_10 among top-%d rmse candidates): %s",
            NUM_RERANK_CANDIDATES, best_config,
        )

        logger.info("Logging selected model for registration")
        with mlflow.start_run(run_name="best_model", nested=True):
            mlflow.log_params(best_config)
            mlflow.set_tag("model_family", "ncf")
            mlflow.set_tag("stage", "best_model")
            mlflow.log_metrics(metrics)
            logger.info("Best model validation metrics: %s", metrics)

            log_pyfunc_model(best_model, "model", registered_model_name=REGISTERED_MODEL_NAME)

    logger.info("Done. Run `mlflow ui --backend-store-uri %s` to inspect runs and the registry.", MLFLOW_TRACKING_URI)
    logger.info("Test set (%d rows) was not used - reserved for future streamed-inference evaluation.", len(test_df))


if __name__ == "__main__":
    main()
