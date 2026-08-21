"""Train an XGBoost residual-correction hybrid recommender, tuning
hyperparameters with Ray Tune and tracking every trial + the final best
model in MLflow.

The hybrid combines SVD's collaborative-filtering signal with Spark-
engineered content features (genre one-hot, demographics, train-only
rating aggregates): a fixed-hyperparameter SVDModel is fit once per trial,
then an XGBRegressor is tuned to predict the *residual* of SVD's
prediction using content features (see `xgb_hybrid_model.py` for why
residual correction, not raw feature-stacking).

All of one invocation's Ray Tune trial runs and its final best-model run
are nested under a single timestamped parent run, so the MLflow UI run
list shows one collapsible row per training run instead of dumping every
trial flat in the list - expand it to drill into individual trials.

After tuning, the top NUM_RERANK_CANDIDATES trials by validation RMSE are
retrained and re-evaluated on validation F1@10 (the metric select_best.py
actually compares model families on), and the one with the best F1@10 -
not necessarily the lowest RMSE - is registered as a new version of the
"movie-recommender-xgb-hybrid" model in the MLflow Model Registry.

The held-out test set is not used in training. Will be used post deployment.

Run with: python -m src.models.train_xgb_hybrid
"""
import logging
import pickle
import tempfile
from datetime import datetime
from pathlib import Path

import mlflow
import mlflow.pyfunc
import ray
from mlflow.tracking import MlflowClient
from ray import tune

from movie_recommender.config import load_settings
from src.data.data_utils import (
    build_relevant_items_map,
    build_user_items_map,
    load_engineered_features,
    load_ratings,
    train_val_test_split_by_user,
)
from src.models.metrics import mae, precision_recall_at_k, rmse
from src.models.recommender_pyfunc import RecommenderPyfunc
from src.models.xgb_hybrid_model import XGBHybridModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH}"
EXPERIMENT_NAME = "movie-recommender"
REGISTERED_MODEL_NAME = "movie-recommender-xgb-hybrid"
TOP_K = 10
NUM_TUNE_SAMPLES = 8
NUM_RERANK_CANDIDATES = 3

# SVD is fit once per trial with a fixed configuration (not tuned itself
# here - train_svd.py already tunes SVD as its own family); only the
# XGBoost residual model's hyperparameters are searched. Used only if no
# svd `best_model` run exists yet to pull a tuned config from (see
# `_load_svd_params`) - e.g. the very first time this script runs before
# train_svd.py has ever produced one.
FALLBACK_SVD_PARAMS = dict(n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02)


def _load_svd_params() -> dict:
    """Pull the hyperparameters of the best-ever tuned SVD model (by
    validation f1_at_10) from MLflow, so the hybrid's base model always
    tracks train_svd.py's actual tuned optimum instead of a hardcoded
    snapshot that can silently drift stale - fitting residuals against a
    weak base model erases whatever signal the tree model would otherwise
    learn."""
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        logger.warning("No '%s' experiment yet, using fallback SVD params", EXPERIMENT_NAME)
        return FALLBACK_SVD_PARAMS

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.stage = 'best_model' and tags.model_family = 'svd'",
        order_by=["attributes.start_time DESC"],
        max_results=1000,
    )
    best_run = max(runs, key=lambda r: r.data.metrics.get("f1_at_10", -1.0), default=None)
    if best_run is None or "f1_at_10" not in best_run.data.metrics:
        logger.warning("No svd best_model run found, using fallback SVD params")
        return FALLBACK_SVD_PARAMS

    params = {
        "n_factors": int(best_run.data.params["n_factors"]),
        "n_epochs": int(best_run.data.params["n_epochs"]),
        "lr_all": float(best_run.data.params["lr_all"]),
        "reg_all": float(best_run.data.params["reg_all"]),
    }
    logger.info("Using tuned SVD params from run %s: %s", best_run.info.run_id, params)
    return params


def evaluate_rating_predictions(model: XGBHybridModel, eval_df) -> dict:
    y_true = eval_df["rating"].to_numpy()
    y_pred = [model.predict_rating(u, m) for u, m in zip(eval_df["user_id"], eval_df["movie_id"])]
    return {"rmse": rmse(y_true, y_pred), "mae": mae(y_true, y_pred)}


def train_trial(config, train_df, val_df, movies_features, users_features, svd_params, parent_run_id):
    """One Ray Tune trial: fit an XGBHybridModel with `config` on the
    training set, evaluate rating-prediction accuracy on the validation
    set, log it as its own MLflow run nested under `parent_run_id`, and
    report RMSE back to the tuner.

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
        mlflow.set_tag("model_family", "xgb_hybrid")
        mlflow.set_tag("stage", "tuning_trial")

        model = XGBHybridModel(svd_params=svd_params, **config).fit(train_df, movies_features, users_features)
        metrics = evaluate_rating_predictions(model, val_df)
        mlflow.log_metrics(metrics)

        tune.report(metrics)


def log_pyfunc_model(model: XGBHybridModel, artifact_path: str, registered_model_name: str = None) -> None:
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
            pip_requirements=["xgboost==3.2.0", "scikit-surprise==1.1.5", "pandas", "numpy", "scipy"],
            registered_model_name=registered_model_name,
        )


def main() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    logger.info("Loading MovieLens 1M ratings and Spark-engineered content features")
    ratings = load_ratings()
    settings = load_settings()
    movies_features, users_features = load_engineered_features(settings.paths.processed_dir)
    svd_params = _load_svd_params()

    train_df, val_df, test_df = train_val_test_split_by_user(ratings, val_frac=0.15, test_frac=0.15, seed=42)
    logger.info(
        "Train: %d rows, Val: %d rows, Test (reserved, unused): %d rows",
        len(train_df), len(val_df), len(test_df),
    )

    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    with mlflow.start_run(run_name=f"xgb_hybrid_tuning_{run_timestamp}") as parent_run:
        parent_run_id = parent_run.info.run_id
        mlflow.set_tag("model_family", "xgb_hybrid")
        mlflow.set_tag("stage", "tuning_parent")

        ray.init(ignore_reinit_error=True)
        logger.info("Ray cluster resources: %s", ray.cluster_resources())

        search_space = {
            "n_estimators": tune.choice([100, 150, 200]),
            "max_depth": tune.choice([3, 5, 7]),
            "learning_rate": tune.loguniform(1e-2, 2e-1),
            "reg_lambda": tune.loguniform(1e-1, 2e1),
        }

        trainable = tune.with_parameters(
            train_trial,
            train_df=train_df,
            val_df=val_df,
            movies_features=movies_features,
            users_features=users_features,
            svd_params=svd_params,
            parent_run_id=parent_run_id,
        )
        tuner = tune.Tuner(
            trainable,
            param_space=search_space,
            tune_config=tune.TuneConfig(metric="rmse", mode="min", num_samples=NUM_TUNE_SAMPLES),
        )
        results = tuner.fit()

        # RMSE (rating-value accuracy) and F1@10 (ranking quality) don't
        # always agree on which config is best, and F1@10 is what this
        # project actually selects/serves models on (see select_best.py).
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
            model = XGBHybridModel(svd_params=svd_params, **config).fit(train_df, movies_features, users_features)
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
            mlflow.set_tag("model_family", "xgb_hybrid")
            mlflow.set_tag("stage", "best_model")
            mlflow.log_metrics(metrics)
            logger.info("Best model validation metrics: %s", metrics)

            log_pyfunc_model(best_model, "model", registered_model_name=REGISTERED_MODEL_NAME)

    logger.info("Done. Run `mlflow ui --backend-store-uri %s` to inspect runs and the registry.", MLFLOW_TRACKING_URI)
    logger.info("Test set (%d rows) was not used - reserved for future streamed-inference evaluation.", len(test_df))


if __name__ == "__main__":
    main()
