"""Compare each model family's best run, promote the winner (by f1_at_10)
to "movie-recommender-prod" with the "champion" alias.

Run with: python -m src.models.select_best
"""
import logging
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH}"
EXPERIMENT_NAME = "movie-recommender"
REGISTERED_MODEL_NAME = "movie-recommender-prod"
CHAMPION_ALIAS = "champion"
SELECTION_METRIC = "f1_at_10"
ARTIFACT_PATH = "model"


def get_latest_best_model_runs(client: MlflowClient, experiment_id: str):
    """One run per model family: the most recent run tagged
    stage=best_model for that family."""
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string="tags.stage = 'best_model'",
        order_by=["attributes.start_time DESC"],
        max_results=200,
    )
    latest_by_family = {}
    for run in runs:
        family = run.data.tags.get("model_family")
        if family and family not in latest_by_family:
            latest_by_family[family] = run
    return latest_by_family


def main() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(f"Experiment '{EXPERIMENT_NAME}' not found. Run a training script first.")

    latest_by_family = get_latest_best_model_runs(client, experiment.experiment_id)
    if not latest_by_family:
        raise RuntimeError("No best_model runs found. Run a training script (e.g. train_svd.py, train_als.py) first.")

    logger.info("Latest best_model run per family:")
    for family, run in latest_by_family.items():
        logger.info(
            "  %-6s run_id=%s %s=%s all_metrics=%s",
            family, run.info.run_id, SELECTION_METRIC, run.data.metrics.get(SELECTION_METRIC), run.data.metrics,
        )

    scored = [r for r in latest_by_family.values() if SELECTION_METRIC in r.data.metrics]
    if not scored:
        raise RuntimeError(f"None of the best_model runs log the selection metric '{SELECTION_METRIC}'.")
    best_run = max(scored, key=lambda r: r.data.metrics[SELECTION_METRIC])
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
