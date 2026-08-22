"""Neural Collaborative Filtering model (PyTorch).

Learns user/item embedding tables plus per-user/per-item bias terms, fed
through a small MLP, to predict ratings directly - the deep-learning
counterpart to the matrix-factorization SVD model. Ids unseen at train
time map to a reserved index 0 ("UNK") embedding so at inference an
unseen user/movie degrades gracefully instead of crashing (a basic
cold-start strategy).
"""
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Ray Tune pins each trial to 1 CPU (via OMP_NUM_THREADS), which is why a
# tuning trial fits in ~9 min - torch's default of spinning up one thread
# per core adds thread-launch/sync overhead per op that dwarfs the actual
# compute on this model's tiny batches (<=4096 rows, <=256-dim). Outside
# Ray (e.g. re-ranking retrains in train_*.py) that pin doesn't apply, so
# pin it explicitly here for consistent, fast performance in every context.
torch.set_num_threads(1)

UNK_INDEX = 0


class NCFNet(nn.Module):
    def __init__(self, n_users: int, n_movies: int, n_factors: int, dropout: float):
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, n_factors)
        self.movie_embedding = nn.Embedding(n_movies, n_factors)
        self.user_bias = nn.Embedding(n_users, 1)
        self.movie_bias = nn.Embedding(n_movies, 1)
        self.mlp = nn.Sequential(
            nn.Linear(2 * n_factors, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, user_idx: torch.Tensor, movie_idx: torch.Tensor) -> torch.Tensor:
        user_vec = self.user_embedding(user_idx)
        movie_vec = self.movie_embedding(movie_idx)
        x = torch.cat([user_vec, movie_vec], dim=-1)
        mlp_out = self.mlp(x).squeeze(-1)
        bias = self.user_bias(user_idx).squeeze(-1) + self.movie_bias(movie_idx).squeeze(-1)
        return mlp_out + bias


class NCFModel:
    def __init__(
        self,
        n_factors: int = 64,
        dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        epochs: int = 15,
        batch_size: int = 4096,
        random_state: int = 42,
    ):
        self.params = dict(
            n_factors=n_factors, dropout=dropout, lr=lr, weight_decay=weight_decay,
            epochs=epochs, batch_size=batch_size, random_state=random_state,
        )
        self.n_factors = n_factors
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state

        self.user_id_to_idx: Dict[int, int] = {}
        self.movie_id_to_idx: Dict[int, int] = {}
        self.global_mean_: float = 0.0
        self.net: NCFNet = None
        self.all_movie_ids_: List[int] = []
        self.epoch_losses_: List[float] = []

    def _index_ids(self, ids: pd.Series) -> Dict[int, int]:
        # index 0 reserved as UNK; real ids start at 1.
        unique_ids = ids.unique()
        return {raw_id: idx + 1 for idx, raw_id in enumerate(unique_ids)}

    def _to_indices(self, ids: pd.Series, id_to_idx: Dict[int, int]) -> np.ndarray:
        return np.array([id_to_idx.get(i, UNK_INDEX) for i in ids], dtype=np.int64)

    def fit(self, train_df: pd.DataFrame) -> "NCFModel":
        torch.manual_seed(self.random_state)

        self.user_id_to_idx = self._index_ids(train_df["user_id"])
        self.movie_id_to_idx = self._index_ids(train_df["movie_id"])
        self.global_mean_ = float(train_df["rating"].mean())
        self.all_movie_ids_ = train_df["movie_id"].unique().tolist()

        n_users = len(self.user_id_to_idx) + 1
        n_movies = len(self.movie_id_to_idx) + 1
        self.net = NCFNet(n_users, n_movies, self.n_factors, self.dropout)

        user_idx = torch.tensor(self._to_indices(train_df["user_id"], self.user_id_to_idx))
        movie_idx = torch.tensor(self._to_indices(train_df["movie_id"], self.movie_id_to_idx))
        targets = torch.tensor((train_df["rating"] - self.global_mean_).to_numpy(), dtype=torch.float32)

        dataset = TensorDataset(user_idx, movie_idx, targets)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.MSELoss()

        self.net.train()
        self.epoch_losses_ = []
        for _epoch in range(self.epochs):
            total_loss, n_batches = 0.0, 0
            for user_batch, movie_batch, target_batch in loader:
                optimizer.zero_grad()
                preds = self.net(user_batch, movie_batch)
                loss = loss_fn(preds, target_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1
            self.epoch_losses_.append(total_loss / max(n_batches, 1))

        return self

    def predict_rating(self, user_id: int, movie_id: int) -> float:
        self.net.eval()
        with torch.no_grad():
            user_idx = torch.tensor([self.user_id_to_idx.get(user_id, UNK_INDEX)])
            movie_idx = torch.tensor([self.movie_id_to_idx.get(movie_id, UNK_INDEX)])
            pred = self.net(user_idx, movie_idx).item() + self.global_mean_
        return float(np.clip(pred, 1.0, 5.0))

    def recommend(self, user_id: int, k: int, exclude_items: set) -> List[int]:
        candidates = [m for m in self.all_movie_ids_ if m not in exclude_items]
        if not candidates:
            return []
        self.net.eval()
        with torch.no_grad():
            user_idx = torch.tensor([self.user_id_to_idx.get(user_id, UNK_INDEX)] * len(candidates))
            movie_idx = torch.tensor(self._to_indices(pd.Series(candidates), self.movie_id_to_idx))
            preds = self.net(user_idx, movie_idx).numpy() + self.global_mean_
        top_k_idx = np.argsort(-preds)[:k]
        return [candidates[i] for i in top_k_idx]

    def recommend_batch(self, user_ids: List[int], k: int, user_items_map: Dict[int, set]) -> Dict[int, List[int]]:
        return {u: self.recommend(u, k, user_items_map.get(u, set())) for u in user_ids}
