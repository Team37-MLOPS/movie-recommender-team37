"""
Baseline + advanced model training and comparison for the MovieLens-1M
rating-prediction task.

Four structurally different models are trained and compared:

  1. SVD collaborative filtering (Surprise library) — learns latent user and
     item factors purely from the user-movie-rating matrix. This is the
     classic recommender-system approach and needs no content features.

  2. Gradient Boosting regression (scikit-learn) on content + demographic
     features (genres, user age/gender/occupation, per-user and per-movie
     historical rating statistics). This is a content/feature-based
     alternative that could generalize to new user-movie pairs whose exact
     (user, movie) combination was never rated before.

  3. XGBoost hybrid — a residual-correction model on top of (1): XGBoost is
     trained on the same content/demographic features as (2) to predict the
     SVD model's prediction error, and the final rating is svd_pred +
     residual_pred. Correcting SVD's residuals with content signal beats SVD
     alone without letting the tree model simply re-derive (and drown out
     with) the strong SVD signal.

  4. Neural Collaborative Filtering (PyTorch) — user/item embedding tables
     followed by an MLP, trained end-to-end on the interaction matrix. A
     distinct, deep-learning architecture family from the tree-based/SVD
     models above.

All four are evaluated with RMSE and MAE on the same held-out test split, and
every run is logged to MLflow (params, metrics, and the serialized model
artifact) so the approaches can be compared in the MLflow UI.
"""

import argparse
import os

import mlflow
import mlflow.pytorch
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from surprise import Dataset, Reader, SVD
from surprise.accuracy import mae as surprise_mae
from surprise.accuracy import rmse as surprise_rmse
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor

RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
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


def train_xgboost_hybrid(train_df, test_df, full_df, svd_algo, n_estimators=150, max_depth=5, learning_rate=0.03, reg_lambda=15.0):
    """Residual-correction hybrid: the already-trained SVD model (fit on
    train_df only) supplies a base rating prediction, and XGBoost is trained
    on the content/demographic features to predict SVD's error on the train
    split. Final prediction = svd_pred + residual_pred. Correcting the
    residual (rather than predicting the rating directly from content
    features plus svd_pred_rating as an input) avoids the model simply
    re-deriving svd_pred_rating and drowning out the weaker content signal."""
    featurized = build_content_features(train_df, full_df)
    featurized["svd_pred_rating"] = [
        svd_algo.predict(uid, iid).est
        for uid, iid in zip(featurized["user_id"], featurized["movie_id"])
    ]
    featurized["residual"] = featurized["rating"] - featurized["svd_pred_rating"]
    feature_cols = get_feature_columns(featurized)

    train_idx = train_df.index
    test_idx = test_df.index

    X_train = featurized.loc[train_idx, feature_cols]
    y_train_residual = featurized.loc[train_idx, "residual"]
    X_test = featurized.loc[test_idx, feature_cols]
    y_test = featurized.loc[test_idx, "rating"]
    svd_pred_test = featurized.loc[test_idx, "svd_pred_rating"].to_numpy()

    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        reg_lambda=reg_lambda,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train_residual)

    preds = np.clip(svd_pred_test + model.predict(X_test), 1, 5)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    mae = mean_absolute_error(y_test, preds)
    return model, {"rmse": rmse, "mae": mae}, {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "reg_lambda": reg_lambda,
        "n_features": len(feature_cols),
        "target": "svd_residual",
    }


class NCFModel(nn.Module):
    """User/item embeddings + per-user/per-movie bias terms -> concat -> MLP,
    added to a fixed global-mean offset. Index 0 in each embedding/bias
    table is reserved as an UNK slot for ids unseen at train time."""

    def __init__(self, n_users, n_movies, n_factors=64, dropout=0.2, global_mean=0.0):
        super().__init__()
        self.user_embedding = nn.Embedding(n_users + 1, n_factors)
        self.movie_embedding = nn.Embedding(n_movies + 1, n_factors)
        self.user_bias = nn.Embedding(n_users + 1, 1)
        self.movie_bias = nn.Embedding(n_movies + 1, 1)
        self.mlp = nn.Sequential(
            nn.Linear(2 * n_factors, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        nn.init.normal_(self.user_embedding.weight, std=0.05)
        nn.init.normal_(self.movie_embedding.weight, std=0.05)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.movie_bias.weight)
        self.register_buffer("global_mean", torch.tensor(float(global_mean)))

    def forward(self, user_idx, movie_idx):
        x = torch.cat([self.user_embedding(user_idx), self.movie_embedding(movie_idx)], dim=1)
        mlp_out = self.mlp(x).squeeze(-1)
        bias = self.user_bias(user_idx).squeeze(-1) + self.movie_bias(movie_idx).squeeze(-1)
        return mlp_out + bias + self.global_mean


def train_ncf(train_df, test_df, epochs=15, batch_size=4096, lr=1e-3, n_factors=64, dropout=0.2, weight_decay=1e-5):
    """Trains an embedding + bias + MLP model directly on the interaction
    matrix. id -> index vocabularies and the global mean are computed from
    train_df only; unseen ids in the test split map to the reserved UNK
    index 0 (no test-set leakage)."""
    user_ids = sorted(train_df["user_id"].unique())
    movie_ids = sorted(train_df["movie_id"].unique())
    user_to_idx = {uid: i + 1 for i, uid in enumerate(user_ids)}
    movie_to_idx = {mid: i + 1 for i, mid in enumerate(movie_ids)}
    global_mean = train_df["rating"].mean()

    def encode(df):
        u = torch.tensor(df["user_id"].map(user_to_idx).fillna(0).astype(int).values, dtype=torch.long)
        m = torch.tensor(df["movie_id"].map(movie_to_idx).fillna(0).astype(int).values, dtype=torch.long)
        r = torch.tensor(df["rating"].astype(float).values, dtype=torch.float32)
        return u, m, r

    train_u, train_m, train_r = encode(train_df)
    test_u, test_m, test_r = encode(test_df)

    device = torch.device("cpu")
    model = NCFModel(
        len(user_ids), len(movie_ids), n_factors=n_factors, dropout=dropout, global_mean=global_mean
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    train_loader = DataLoader(
        TensorDataset(train_u, train_m, train_r), batch_size=batch_size, shuffle=True
    )

    for epoch in range(epochs):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for u_batch, m_batch, r_batch in train_loader:
            optimizer.zero_grad()
            preds = model(u_batch, m_batch)
            loss = loss_fn(preds, r_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        avg_loss = epoch_loss / n_batches
        mlflow.log_metric("train_mse", avg_loss, step=epoch)
        print(f"[NCF] epoch {epoch + 1}/{epochs} train_mse={avg_loss:.4f}")

    model.eval()
    with torch.no_grad():
        preds = np.clip(model(test_u, test_m).numpy(), 1, 5)
    y_test = test_r.numpy()
    rmse = mean_squared_error(y_test, preds) ** 0.5
    mae = mean_absolute_error(y_test, preds)
    input_example = (test_u[:5], test_m[:5])
    return model, {"rmse": rmse, "mae": mae}, {
        "n_factors": n_factors,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "dropout": dropout,
        "weight_decay": weight_decay,
    }, input_example


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

    with mlflow.start_run(run_name="xgboost_hybrid"):
        xgb_model, xgb_metrics, xgb_params = train_xgboost_hybrid(train_df, test_df, df, svd_model)
        mlflow.log_params(xgb_params)
        mlflow.log_metrics(xgb_metrics)
        mlflow.log_param("model_type", "XGBoost_hybrid")
        mlflow.log_param("train_rows", len(train_df))
        mlflow.log_param("test_rows", len(test_df))
        mlflow.xgboost.log_model(xgb_model, artifact_path="model")
        results["xgboost_hybrid"] = xgb_metrics
        print(f"[XGB] RMSE={xgb_metrics['rmse']:.4f} MAE={xgb_metrics['mae']:.4f}")

    with mlflow.start_run(run_name="ncf_pytorch"):
        ncf_model, ncf_metrics, ncf_params, ncf_input_example = train_ncf(train_df, test_df)
        mlflow.log_params(ncf_params)
        mlflow.log_metrics(ncf_metrics)
        mlflow.log_param("model_type", "NCF_pytorch")
        mlflow.log_param("train_rows", len(train_df))
        mlflow.log_param("test_rows", len(test_df))
        mlflow.pytorch.log_model(
            ncf_model, artifact_path="model", input_example=ncf_input_example, serialization_format="pickle"
        )
        results["ncf_pytorch"] = ncf_metrics
        print(f"[NCF] RMSE={ncf_metrics['rmse']:.4f} MAE={ncf_metrics['mae']:.4f}")

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
