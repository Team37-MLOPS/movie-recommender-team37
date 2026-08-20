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

# SVD is fit once per trial with a fixed, reasonable configuration (not
# tuned itself here - train_svd.py already tunes SVD as its own family);
# only the XGBoost residual model's hyperparameters are searched.
SVD_PARAMS = dict(n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02)


def evaluate_ranking(model: XGBHybridModel, users, user_items_map, relevant_items_map, k=TOP_K) -> dict:
    recs = model.recommend_batch(users, k, user_items_map)
    return precision_recall_at_k(recs, relevant_items_map, k=k)


def evaluate_rating_predictions(model: XGBHybridModel, eval_df) -> dict:
    y_true = eval_df["rating"].to_numpy()
    y_pred = [model.predict_rating(u, m) for u, m in zip(eval_df["user_id"], eval_df["movie_id"])]
    return {"rmse": rmse(y_true, y_pred), "mae": mae(y_true, y_pred)}


def train_trial(config, train_df, val_df, movies_features, users_features, parent_run_id):
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

        model = XGBHybridModel(svd_params=SVD_PARAMS, **config).fit(train_df, movies_features, users_features)
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
            parent_run_id=parent_run_id,
        )
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
        with mlflow.start_run(run_name="best_model", nested=True):
            mlflow.log_params(best_config)
            mlflow.set_tag("model_family", "xgb_hybrid")
            mlflow.set_tag("stage", "best_model")

            best_model = XGBHybridModel(svd_params=SVD_PARAMS, **best_config).fit(
                train_df, movies_features, users_features
            )

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
