"""ALS model (implicit feedback) via the `implicit` library. Evaluated on
ranking quality (Precision@K/Recall@K), not RMSE/MAE.
"""
from typing import Dict, List

import numpy as np
import pandas as pd
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares


class ALSModel:
    def __init__(self, factors: int = 60, regularization: float = 0.01, iterations: int = 15, random_state: int = 42):
        self.params = dict(factors=factors, regularization=regularization, iterations=iterations, random_state=random_state)
        self.model = AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            iterations=iterations,
            random_state=random_state,
        )
        self.user_id_to_idx: Dict[int, int] = {}
        self.idx_to_movie_id: Dict[int, int] = {}
        self.user_items_: sp.csr_matrix = None

    def fit(self, train_df: pd.DataFrame) -> "ALSModel":
        user_ids = train_df["user_id"].unique()
        movie_ids = train_df["movie_id"].unique()
        self.user_id_to_idx = {u: i for i, u in enumerate(user_ids)}
        movie_id_to_idx = {m: i for i, m in enumerate(movie_ids)}
        self.idx_to_movie_id = {i: m for m, i in movie_id_to_idx.items()}

        rows = train_df["user_id"].map(self.user_id_to_idx).to_numpy()
        cols = train_df["movie_id"].map(movie_id_to_idx).to_numpy()
        # every observed rating counts as a positive implicit signal, regardless of its value
        data = np.ones(len(train_df), dtype=np.float32)

        self.user_items_ = sp.csr_matrix((data, (rows, cols)), shape=(len(user_ids), len(movie_ids)))
        self.model.fit(self.user_items_)
        return self

    def recommend(self, user_id: int, k: int, exclude_items: set) -> List[int]:
        if user_id not in self.user_id_to_idx:
            return []
        user_idx = self.user_id_to_idx[user_id]
        # over-fetch so that excluding already-seen items still leaves k recommendations
        n_candidates = k + len(exclude_items)
        item_idxs, _scores = self.model.recommend(
            user_idx, self.user_items_[user_idx], N=n_candidates, filter_already_liked_items=False
        )
        movie_ids = [self.idx_to_movie_id[i] for i in item_idxs if self.idx_to_movie_id[i] not in exclude_items]
        return movie_ids[:k]

    def recommend_batch(self, user_ids: List[int], k: int, user_items_map: Dict[int, set]) -> Dict[int, List[int]]:
        return {u: self.recommend(u, k, user_items_map.get(u, set())) for u in user_ids}
