"""XGBoost residual-correction hybrid model.

Fits a fixed-hyperparameter `SVDModel` (collaborative-filtering signal)
first, then trains an `XGBRegressor` to predict the *residual*
(`rating - svd_pred`) from Spark-engineered content features (genre
one-hot, demographics, train-only rating aggregates) rather than the raw
rating. Final prediction = svd_pred + residual_pred, clipped to [1, 5].

Why residual correction and not feeding `svd_pred` in as just another
XGBoost input feature: giving a weak learner the strong model's own
prediction as an input feature lets it get "lazy" and mostly reproduce
that feature, while any overfit noise on the weaker content features
still drags predictions down - on this dataset that approach failed to
beat SVD alone. Targeting the base model's *error* instead makes every
unit of the tree model's capacity work on signal SVD doesn't already
capture.
"""
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from src.data.data_utils import CONTENT_FEATURE_COLUMNS
from src.models.svd_model import SVDModel


class XGBHybridModel:
    def __init__(
        self,
        svd_params: dict,
        n_estimators: int = 150,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        reg_lambda: float = 1.0,
        random_state: int = 42,
    ):
        self.svd_params = svd_params
        self.xgb_params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            reg_lambda=reg_lambda,
            random_state=random_state,
        )
        self.svd_model = SVDModel(**svd_params, random_state=random_state)
        self.xgb_model = XGBRegressor(**self.xgb_params)

        self.movies_features_: Optional[pd.DataFrame] = None
        self.users_features_: Optional[pd.DataFrame] = None
        self.user_stats_: Optional[pd.DataFrame] = None
        self.movie_stats_: Optional[pd.DataFrame] = None
        self.all_movie_ids_: List[int] = []

    def _svd_predict_batch(self, df: pd.DataFrame) -> np.ndarray:
        testset = list(zip(df["user_id"], df["movie_id"], [0.0] * len(df)))
        predictions = self.svd_model.model.test(testset)
        return np.array([p.est for p in predictions])

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        merged = (
            df.merge(self.movies_features_, on="movie_id", how="left")
            .merge(self.users_features_, on="user_id", how="left")
            .merge(self.user_stats_, on="user_id", how="left")
            .merge(self.movie_stats_, on="movie_id", how="left")
        )
        merged[CONTENT_FEATURE_COLUMNS] = merged[CONTENT_FEATURE_COLUMNS].fillna(0.0)
        return merged[CONTENT_FEATURE_COLUMNS]

    def fit(
        self,
        train_df: pd.DataFrame,
        movies_features: pd.DataFrame,
        users_features: pd.DataFrame,
        svd_model: Optional[SVDModel] = None,
    ) -> "XGBHybridModel":
        self.movies_features_ = movies_features
        self.users_features_ = users_features

        # Reuse an already-fit SVD model when the caller shares one across
        # multiple hyperparameter trials (fixed SVD hyperparameters mean it
        # doesn't need to be refit per trial); otherwise fit our own.
        self.svd_model = svd_model if svd_model is not None else self.svd_model.fit(train_df)

        self.user_stats_ = (
            train_df.groupby("user_id")["rating"]
            .agg(user_rating_count="count", user_mean_rating="mean", user_rating_stddev="std")
            .reset_index()
        )
        self.movie_stats_ = (
            train_df.groupby("movie_id")["rating"]
            .agg(movie_rating_count="count", movie_mean_rating="mean", movie_rating_stddev="std")
            .reset_index()
        )

        svd_preds = self._svd_predict_batch(train_df)
        residuals = train_df["rating"].to_numpy() - svd_preds
        X = self._build_features(train_df)
        self.xgb_model.fit(X, residuals)

        self.all_movie_ids_ = train_df["movie_id"].unique().tolist()
        return self

    def predict_rating(self, user_id: int, movie_id: int) -> float:
        svd_pred = self.svd_model.predict_rating(user_id, movie_id)
        row = pd.DataFrame([{"user_id": user_id, "movie_id": movie_id}])
        residual_pred = float(self.xgb_model.predict(self._build_features(row))[0])
        return float(np.clip(svd_pred + residual_pred, 1.0, 5.0))

    def recommend(self, user_id: int, k: int, exclude_items: set) -> List[int]:
        candidates = [m for m in self.all_movie_ids_ if m not in exclude_items]
        if not candidates:
            return []
        candidates_df = pd.DataFrame({"user_id": user_id, "movie_id": candidates})
        svd_preds = self._svd_predict_batch(candidates_df)
        residual_preds = self.xgb_model.predict(self._build_features(candidates_df))
        scores = np.clip(svd_preds + residual_preds, 1.0, 5.0)
        top_k_idx = np.argsort(-scores)[:k]
        return [candidates[i] for i in top_k_idx]

    def recommend_batch(self, user_ids: List[int], k: int, user_items_map: Dict[int, set]) -> Dict[int, List[int]]:
        return {u: self.recommend(u, k, user_items_map.get(u, set())) for u in user_ids}
