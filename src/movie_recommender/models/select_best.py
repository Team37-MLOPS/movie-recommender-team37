"""Compare the best-ever run from each trained model family (svd, als,
xgb_hybrid, ncf) and promote the winner (by f1_at_10) to a single, unified
"movie-recommender-prod" model with the "champion" alias in the MLflow
Model Registry.

Best-ever, not most-recent: with repeated scheduled retraining, a fresh
tune can land on a worse hyperparameter config than a previous one by
chance, and only ever comparing the latest run would let the production
champion regress.

Run with: python -m movie_recommender.models.select_best
"""
import logging
import os
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{MLFLOW_DB_PATH}")
EXPERIMENT_NAME = "movie-recommender"
REGISTERED_MODEL_NAME = "movie-recommender-prod"
CHAMPION_ALIAS = "champion"
SELECTION_METRIC = "f1_at_10"
ARTIFACT_PATH = "model"


def get_best_ever_model_runs(client: MlflowClient, experiment_id: str):
    """One run per model family: across all runs tagged stage=best_model
    for that family, the one with the highest SELECTION_METRIC (runs
    missing the metric are skipped, not treated as a win by omission)."""
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string="tags.stage = 'best_model'",
        order_by=["attributes.start_time DESC"],
        max_results=1000,
    )
    best_by_family = {}
    for run in runs:
        family = run.data.tags.get("model_family")
        score = run.data.metrics.get(SELECTION_METRIC)
        if not family or score is None:
            continue
        current_best = best_by_family.get(family)
        if current_best is None or score > current_best.data.metrics[SELECTION_METRIC]:
            best_by_family[family] = run
    return best_by_family


def main() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(f"Experiment '{EXPERIMENT_NAME}' not found. Run a training script first.")

    best_by_family = get_best_ever_model_runs(client, experiment.experiment_id)
    if not best_by_family:
        raise RuntimeError(
            f"No best_model runs log the selection metric '{SELECTION_METRIC}'. "
            "Run a training script (e.g. train_svd.py, train_als.py, train_xgb_hybrid.py, train_ncf.py) first."
        )

    logger.info("Best-ever best_model run per family:")
    for family, run in best_by_family.items():
        logger.info(
            "  %-10s run_id=%s %s=%s all_metrics=%s",
            family, run.info.run_id, SELECTION_METRIC, run.data.metrics.get(SELECTION_METRIC), run.data.metrics,
        )

    best_run = max(best_by_family.values(), key=lambda r: r.data.metrics[SELECTION_METRIC])
    best_family = best_run.data.tags.get("model_family")

    logger.info(
        "Winner: family=%s run_id=%s %s=%.4f",
        best_family, best_run.info.run_id, SELECTION_METRIC, best_run.data.metrics[SELECTION_METRIC],
    )

    model_uri = f"runs:/{best_run.info.run_id}/{ARTIFACT_PATH}"
    result = mlflow.register_model(model_uri=model_uri, name=REGISTERED_MODEL_NAME)
    logger.info("Registered '%s' version %s (source family: %s)", REGISTERED_MODEL_NAME, result.version, best_family)

    client.set_registered_model_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS, result.version)
    logger.info(
        "Set alias '%s' -> version %s. Load it with: mlflow.pyfunc.load_model('models:/%s@%s')",
        CHAMPION_ALIAS, result.version, REGISTERED_MODEL_NAME, CHAMPION_ALIAS,
    )


if __name__ == "__main__":
    main()
