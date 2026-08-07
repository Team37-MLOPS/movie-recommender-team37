"""
Baseline model training + comparison for the MovieLens-1M rating-prediction task.

Two meaningful, structurally different models are trained and compared:

  1. SVD collaborative filtering (Surprise library) — learns latent user and
     item factors purely from the user-movie-rating matrix. This is the
     classic recommender-system approach and needs no content features.

  2. Gradient Boosting regression (scikit-learn) on content + demographic
     features (genres, user age/gender/occupation, per-user and per-movie
     historical rating statistics). This is a content/feature-based
     alternative that could generalize to new user-movie pairs whose exact
     (user, movie) combination was never rated before.

Both are evaluated with RMSE and MAE on a held-out test split, and every run
is logged to MLflow (params, metrics, and the serialized model artifact) so
the two approaches can be compared in the MLflow UI.
"""

import argparse
import os

import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from surprise import Dataset, Reader, SVD
from surprise.accuracy import mae as surprise_mae
from surprise.accuracy import rmse as surprise_rmse

RANDOM_STATE = 42
MLRUNS_DIR = os.environ.get(
    "MLFLOW_TRACKING_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "mlruns")),
)
EXPERIMENT_NAME = "movielens-rating-prediction"


def load_data(data_path):
    df = pd.read_csv(data_path)
    return df


def train_test_split_df(df, test_size=0.2):
    return train_test_split(df, test_size=test_size, random_state=RANDOM_STATE)


def train_svd_collaborative_filtering(train_df, test_df, n_factors=50, n_epochs=20):
    reader = Reader(rating_scale=(1, 5))
    train_data = Dataset.load_from_df(train_df[["user_id", "movie_id", "rating"]], reader)
    trainset = train_data.build_full_trainset()

    algo = SVD(n_factors=n_factors, n_epochs=n_epochs, random_state=RANDOM_STATE)
    algo.fit(trainset)

    testset = list(
        zip(test_df["user_id"], test_df["movie_id"], test_df["rating"])
    )
    predictions = algo.test(testset)

    rmse = surprise_rmse(predictions, verbose=False)
    mae = surprise_mae(predictions, verbose=False)
    return algo, {"rmse": rmse, "mae": mae}, {"n_factors": n_factors, "n_epochs": n_epochs}


GENRE_COLS_PREFIX = "genre_"


def build_content_features(train_df, full_df):
    """Per-user and per-movie rating stats are computed on the TRAIN split
    only, then joined onto both splits — avoids leaking test-set ratings
    into the features used to predict them."""
    user_stats = (
        train_df.groupby("user_id")["rating"]
        .agg(user_avg_rating="mean", user_rating_count="count")
        .reset_index()
    )
    movie_stats = (
        train_df.groupby("movie_id")["rating"]
        .agg(movie_avg_rating="mean", movie_rating_count="count")
        .reset_index()
    )

    global_avg = train_df["rating"].mean()

    merged = full_df.merge(user_stats, on="user_id", how="left")
    merged = merged.merge(movie_stats, on="movie_id", how="left")
    merged["user_avg_rating"] = merged["user_avg_rating"].fillna(global_avg)
    merged["user_rating_count"] = merged["user_rating_count"].fillna(0)
    merged["movie_avg_rating"] = merged["movie_avg_rating"].fillna(global_avg)
    merged["movie_rating_count"] = merged["movie_rating_count"].fillna(0)
    return merged


def get_feature_columns(df):
    genre_cols = [c for c in df.columns if c.startswith(GENRE_COLS_PREFIX)]
    other_cols = [
        "age", "occupation", "gender_is_male", "num_genres",
        "user_avg_rating", "user_rating_count",
        "movie_avg_rating", "movie_rating_count",
    ]
    return other_cols + genre_cols


def train_content_based_regressor(train_df, test_df, full_df, n_estimators=150, max_depth=4):
    featurized = build_content_features(train_df, full_df)
    feature_cols = get_feature_columns(featurized)

    train_idx = train_df.index
    test_idx = test_df.index

    X_train = featurized.loc[train_idx, feature_cols]
    y_train = featurized.loc[train_idx, "rating"]
    X_test = featurized.loc[test_idx, feature_cols]
    y_test = featurized.loc[test_idx, "rating"]

    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    preds = np.clip(model.predict(X_test), 1, 5)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    mae = mean_absolute_error(y_test, preds)
    return model, {"rmse": rmse, "mae": mae}, {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "n_features": len(feature_cols),
    }


def main(data_path, test_size=0.2):
    os.makedirs(MLRUNS_DIR, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{os.path.join(MLRUNS_DIR, 'mlflow.db')}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_data(data_path)
    train_df, test_df = train_test_split_df(df, test_size=test_size)

    results = {}

    with mlflow.start_run(run_name="svd_collaborative_filtering"):
        svd_model, svd_metrics, svd_params = train_svd_collaborative_filtering(train_df, test_df)
        mlflow.log_params(svd_params)
        mlflow.log_metrics(svd_metrics)
        mlflow.log_param("model_type", "SVD_collaborative_filtering")
        mlflow.log_param("train_rows", len(train_df))
        mlflow.log_param("test_rows", len(test_df))
        results["svd_collaborative_filtering"] = svd_metrics
        print(f"[SVD] RMSE={svd_metrics['rmse']:.4f} MAE={svd_metrics['mae']:.4f}")

    with mlflow.start_run(run_name="gradient_boosting_content_based"):
        gb_model, gb_metrics, gb_params = train_content_based_regressor(train_df, test_df, df)
        mlflow.log_params(gb_params)
        mlflow.log_metrics(gb_metrics)
        mlflow.log_param("model_type", "GradientBoosting_content_based")
        mlflow.log_param("train_rows", len(train_df))
        mlflow.log_param("test_rows", len(test_df))
        mlflow.sklearn.log_model(gb_model, artifact_path="model")
        results["gradient_boosting_content_based"] = gb_metrics
        print(f"[GB]  RMSE={gb_metrics['rmse']:.4f} MAE={gb_metrics['mae']:.4f}")

    print("\n=== Model comparison ===")
    for name, metrics in results.items():
        print(f"{name}: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default=os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "data", "processed", "movielens_features.csv")
        ),
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()
    main(data_path=args.data, test_size=args.test_size)
